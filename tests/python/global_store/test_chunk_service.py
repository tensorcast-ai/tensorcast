#  Copyright (c) 2025-2026, TensorCast Team.

"""Tests for ChunkService (chunk directory operations)."""

import duckdb
import pytest

from tensorcast.global_store.db_utils import init_db
from tensorcast.global_store.models import Worker
from tensorcast.global_store.repositories.chunk_directory_repository import (
    ChunkDirectoryRepository,
)
from tensorcast.global_store.repositories.worker_repository import WorkerRepository
from tensorcast.global_store.services.chunk_service import ChunkService
from tensorcast.proto.global_store.v1 import global_store_pb2


@pytest.fixture()
def db_conn():
    conn = duckdb.connect()
    init_db(conn)
    return conn


@pytest.fixture()
def repos(db_conn):
    return {
        "chunk": ChunkDirectoryRepository(db_conn),
        "worker": WorkerRepository(db_conn),
    }


@pytest.fixture()
def chunk_service(repos):
    return ChunkService(repos["chunk"])


def _insert_worker(
    worker_repo: WorkerRepository,
    node_id: str = "node1",
    node_address: str = "192.168.50.10",
) -> Worker:
    worker = Worker(
        worker_id="worker_node1_1",
        daemon_id=f"daemon_{node_id}",
        node_id=node_id,
        node_address=node_address,
        grpc_port=50051,
        p2p_port=50052,
        mem_pool_total_size=1024,
        mem_pool_available_size=1024,
        accepting_new_requests=True,
    )
    worker_repo.create(worker)
    return worker


def test_batch_update_and_query(chunk_service: ChunkService, repos):
    # Ensure worker exists for JOIN in query_chunk_locations
    _insert_worker(repos["worker"], node_id="node1")

    updates = [
        global_store_pb2.ChunkStateUpdate(
            artifact_id="art1",
            chunk_idx=0,
            state=global_store_pb2.CHUNK_STATE_HOT,
            device_uuid="gpu-uuid-0",
            replica=0,
        ),
        global_store_pb2.ChunkStateUpdate(
            artifact_id="art1",
            chunk_idx=1,
            state=global_store_pb2.CHUNK_STATE_COPIED_GPU,
            device_uuid="gpu-uuid-0",
            replica=0,
        ),
    ]

    applied = chunk_service.batch_update_chunk_states(
        worker_id="worker_node1_1", node_id="node1", updates=updates
    )
    assert applied == 2

    # Query all chunks
    locations = chunk_service.query_chunk_locations("art1")
    assert len(locations) == 2
    # Verify structure/order by (chunk_idx, state, node_load_ratio)
    assert {loc.chunk_idx for loc in locations} == {0, 1}
    # Verify fields copied from repo row mapping
    by_idx = {loc.chunk_idx: loc for loc in locations}
    assert by_idx[0].node_id == "node1"
    assert by_idx[0].node_address == "192.168.50.10"
    assert by_idx[0].p2p_port == 50052
    assert by_idx[0].device_uuid == "gpu-uuid-0"

    # Query subset
    sub = chunk_service.query_chunk_locations("art1", [1])
    assert len(sub) == 1
    assert sub[0].chunk_idx == 1


def test_query_excludes_evicted(chunk_service: ChunkService, repos):
    _insert_worker(repos["worker"], node_id="node1")

    updates = [
        global_store_pb2.ChunkStateUpdate(
            artifact_id="art2",
            chunk_idx=0,
            state=global_store_pb2.CHUNK_STATE_EVICTED,
            device_uuid="gpu-uuid-0",
            replica=0,
        ),
        global_store_pb2.ChunkStateUpdate(
            artifact_id="art2",
            chunk_idx=1,
            state=global_store_pb2.CHUNK_STATE_HOT,
            device_uuid="gpu-uuid-0",
            replica=0,
        ),
    ]

    chunk_service.batch_update_chunk_states("w", "node1", updates)

    locations = chunk_service.query_chunk_locations("art2")
    # Evicted chunk (idx 0) should be excluded
    assert [loc.chunk_idx for loc in locations] == [1]


def test_cleanup_stale_chunks(chunk_service: ChunkService, repos, db_conn):
    _insert_worker(repos["worker"], node_id="node1")

    # Insert one old row by manually adjusting last_update_time
    updates = [
        global_store_pb2.ChunkStateUpdate(
            artifact_id="art3",
            chunk_idx=0,
            state=global_store_pb2.CHUNK_STATE_HOT,
            device_uuid="gpu-uuid-0",
            replica=0,
        )
    ]
    chunk_service.batch_update_chunk_states("w", "node1", updates)

    # Make it stale: set last_update_time to long ago
    db_conn.execute(
        "UPDATE chunk_directory SET last_update_time = TIMESTAMP '1970-01-01 00:00:00' WHERE artifact_id = 'art3'"
    )

    deleted = chunk_service.cleanup_stale_chunks(stale_threshold_seconds=1)
    assert deleted >= 1


def test_get_chunk_distribution(chunk_service: ChunkService, repos):
    _insert_worker(repos["worker"], node_id="node1")
    artifact_id = "art4"
    updates = [
        global_store_pb2.ChunkStateUpdate(
            artifact_id=artifact_id,
            chunk_idx=0,
            state=global_store_pb2.CHUNK_STATE_HOT,
            device_uuid="gpu-uuid-0",
            replica=0,
        ),
        global_store_pb2.ChunkStateUpdate(
            artifact_id=artifact_id,
            chunk_idx=1,
            state=global_store_pb2.CHUNK_STATE_COLD,
            device_uuid="gpu-uuid-0",
            replica=0,
        ),
        global_store_pb2.ChunkStateUpdate(
            artifact_id=artifact_id,
            chunk_idx=2,
            state=global_store_pb2.CHUNK_STATE_COLD,
            device_uuid="gpu-uuid-1",
            replica=1,
        ),
    ]
    chunk_service.batch_update_chunk_states("w", "node1", updates)

    dist = chunk_service.get_chunk_distribution(artifact_id)
    # state_distribution: keys are enum ints
    assert dist["state_distribution"][global_store_pb2.CHUNK_STATE_HOT] == 1
    assert dist["state_distribution"][global_store_pb2.CHUNK_STATE_COLD] == 2
    # node_distribution: all on node1
    assert dist["node_distribution"]["node1"] == 3


def test_batch_update_empty_returns_zero(chunk_service: ChunkService):
    applied = chunk_service.batch_update_chunk_states("w", "nodeX", [])
    assert applied == 0


def test_query_nonexistent_artifact_returns_empty(chunk_service: ChunkService, repos):
    _insert_worker(repos["worker"], node_id="node1")
    assert chunk_service.query_chunk_locations("nonexistent") == []


def test_get_chunk_distribution_nonexistent_artifact_is_empty(
    chunk_service: ChunkService,
):
    dist = chunk_service.get_chunk_distribution("nonexistent")
    assert dist["state_distribution"] == {}
    assert dist["node_distribution"] == {}


def test_upsert_replaces_state_for_same_pk(chunk_service: ChunkService, repos):
    _insert_worker(repos["worker"], node_id="node1")
    artifact_id = "art_upsert"
    updates1 = [
        global_store_pb2.ChunkStateUpdate(
            artifact_id=artifact_id,
            chunk_idx=0,
            state=global_store_pb2.CHUNK_STATE_HOT,
            device_uuid="gpu-uuid-0",
            replica=0,
        )
    ]
    chunk_service.batch_update_chunk_states("w", "node1", updates1)

    updates2 = [
        global_store_pb2.ChunkStateUpdate(
            artifact_id=artifact_id,
            chunk_idx=0,
            state=global_store_pb2.CHUNK_STATE_COLD,
            device_uuid="gpu-uuid-0",
            replica=0,
        )
    ]
    chunk_service.batch_update_chunk_states("w", "node1", updates2)

    res = chunk_service.query_chunk_locations(artifact_id)
    assert len(res) == 1
    assert res[0].state == global_store_pb2.CHUNK_STATE_COLD


def test_query_with_empty_indices_returns_empty(chunk_service: ChunkService, repos):
    _insert_worker(repos["worker"], node_id="node1")
    artifact_id = "art_empty_indices"
    updates = [
        global_store_pb2.ChunkStateUpdate(
            artifact_id=artifact_id,
            chunk_idx=0,
            state=global_store_pb2.CHUNK_STATE_HOT,
            device_uuid="gpu-uuid-0",
            replica=0,
        )
    ]
    chunk_service.batch_update_chunk_states("w", "node1", updates)

    # Explicit empty list should return empty instead of full-scan
    res_empty = chunk_service.query_chunk_locations(artifact_id, [])
    assert res_empty == []

    # Sanity: without indices we do get results
    res_all = chunk_service.query_chunk_locations(artifact_id)
    assert len(res_all) == 1


def test_query_requires_existing_worker_join(chunk_service: ChunkService):
    artifact_id = "art_join"
    # Insert chunk rows without inserting workers entry
    updates = [
        global_store_pb2.ChunkStateUpdate(
            artifact_id=artifact_id,
            chunk_idx=0,
            state=global_store_pb2.CHUNK_STATE_HOT,
            device_uuid="gpu-uuid-0",
            replica=0,
        )
    ]
    chunk_service.batch_update_chunk_states("w", "node_orphan", updates)

    # JOIN with workers should filter out rows since worker(node_id) does not exist
    res = chunk_service.query_chunk_locations(artifact_id)
    assert res == []


def test_get_chunk_distribution_across_multiple_nodes(
    chunk_service: ChunkService, repos
):
    # Insert two workers on different nodes
    worker_repo = repos["worker"]
    worker_repo.create(
        Worker(
            worker_id="worker_node1_1",
            daemon_id="daemon_node1",
            node_id="node1",
            node_address="192.168.50.10",
            grpc_port=50051,
            p2p_port=50052,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            accepting_new_requests=True,
        )
    )
    worker_repo.create(
        Worker(
            worker_id="worker_node2_1",
            daemon_id="daemon_node2",
            node_id="node2",
            node_address="192.168.50.11",
            grpc_port=50061,
            p2p_port=50062,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            accepting_new_requests=True,
        )
    )

    artifact_id = "art_multi_nodes"
    chunk_service.batch_update_chunk_states(
        "w1",
        "node1",
        [
            global_store_pb2.ChunkStateUpdate(
                artifact_id=artifact_id,
                chunk_idx=0,
                state=global_store_pb2.CHUNK_STATE_HOT,
                device_uuid="gpu-uuid-0",
                replica=0,
            )
        ],
    )
    chunk_service.batch_update_chunk_states(
        "w2",
        "node2",
        [
            global_store_pb2.ChunkStateUpdate(
                artifact_id=artifact_id,
                chunk_idx=1,
                state=global_store_pb2.CHUNK_STATE_COLD,
                device_uuid="gpu-uuid-1",
                replica=0,
            )
        ],
    )

    dist = chunk_service.get_chunk_distribution(artifact_id)
    assert dist["node_distribution"]["node1"] == 1
    assert dist["node_distribution"]["node2"] == 1


def test_query_specific_indices_returns_multiple_nodes(
    chunk_service: ChunkService, repos
):
    # Two workers on node1/node2 for the same chunk index
    worker_repo = repos["worker"]
    worker_repo.create(
        Worker(
            worker_id="worker_node1_2",
            daemon_id="daemon_node1",
            node_id="node1",
            node_address="192.168.50.10",
            grpc_port=50071,
            p2p_port=50072,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            accepting_new_requests=True,
        )
    )
    worker_repo.create(
        Worker(
            worker_id="worker_node2_2",
            daemon_id="daemon_node2",
            node_id="node2",
            node_address="192.168.50.11",
            grpc_port=50081,
            p2p_port=50082,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            accepting_new_requests=True,
        )
    )

    artifact_id = "art_specific_indices"
    idx = 7
    chunk_service.batch_update_chunk_states(
        "w1",
        "node1",
        [
            global_store_pb2.ChunkStateUpdate(
                artifact_id=artifact_id,
                chunk_idx=idx,
                state=global_store_pb2.CHUNK_STATE_HOT,
                device_uuid="gpu-uuid-0",
                replica=0,
            )
        ],
    )
    chunk_service.batch_update_chunk_states(
        "w2",
        "node2",
        [
            global_store_pb2.ChunkStateUpdate(
                artifact_id=artifact_id,
                chunk_idx=idx,
                state=global_store_pb2.CHUNK_STATE_HOT,
                device_uuid="gpu-uuid-1",
                replica=0,
            )
        ],
    )

    res = chunk_service.query_chunk_locations(artifact_id, [idx])
    # Expect two locations (one per node)
    assert len(res) == 2
    assert {r.node_id for r in res} == {"node1", "node2"}


def test_query_ordering_by_node_load_ratio(chunk_service: ChunkService, repos, db_conn):
    # Prepare two workers on different nodes
    worker_repo = repos["worker"]
    worker_repo.create(
        Worker(
            worker_id="worker_node1_3",
            daemon_id="daemon_node1",
            node_id="node1",
            node_address="192.168.50.10",
            grpc_port=50091,
            p2p_port=50092,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            accepting_new_requests=True,
        )
    )
    worker_repo.create(
        Worker(
            worker_id="worker_node2_3",
            daemon_id="daemon_node2",
            node_id="node2",
            node_address="192.168.50.11",
            grpc_port=50101,
            p2p_port=50102,
            mem_pool_total_size=1024,
            mem_pool_available_size=1024,
            accepting_new_requests=True,
        )
    )

    artifact_id = "art_ordering"
    idx = 9
    # Insert two identical-state rows for the same chunk index on different nodes
    chunk_service.batch_update_chunk_states(
        "w1",
        "node1",
        [
            global_store_pb2.ChunkStateUpdate(
                artifact_id=artifact_id,
                chunk_idx=idx,
                state=global_store_pb2.CHUNK_STATE_HOT,
                device_uuid="gpu-uuid-0",
                replica=0,
            )
        ],
    )
    chunk_service.batch_update_chunk_states(
        "w2",
        "node2",
        [
            global_store_pb2.ChunkStateUpdate(
                artifact_id=artifact_id,
                chunk_idx=idx,
                state=global_store_pb2.CHUNK_STATE_HOT,
                device_uuid="gpu-uuid-1",
                replica=0,
            )
        ],
    )

    # Adjust node_load_ratio to enforce ordering (node1 smaller than node2)
    db_conn.execute(
        """
        UPDATE chunk_directory
        SET node_load_ratio = CASE node_id
            WHEN 'node1' THEN 0.2
            WHEN 'node2' THEN 0.5
        END
        WHERE artifact_id = ? AND chunk_idx = ? AND chunk_state = ?
        """,
        [artifact_id, idx, int(global_store_pb2.CHUNK_STATE_HOT)],
    )

    res = chunk_service.query_chunk_locations(artifact_id, [idx])
    assert len(res) == 2
    # Expect node1 (0.2) to come before node2 (0.5)
    assert [r.node_id for r in res] == ["node1", "node2"]


def test_cleanup_stale_chunks_when_recent_returns_zero(
    chunk_service: ChunkService, repos
):
    _insert_worker(repos["worker"], node_id="node1")
    artifact_id = "art_recent"
    updates = [
        global_store_pb2.ChunkStateUpdate(
            artifact_id=artifact_id,
            chunk_idx=0,
            state=global_store_pb2.CHUNK_STATE_HOT,
            device_uuid="gpu-uuid-0",
            replica=0,
        )
    ]
    chunk_service.batch_update_chunk_states("w", "node1", updates)

    deleted = chunk_service.cleanup_stale_chunks(stale_threshold_seconds=10_000_000)
    assert deleted == 0
