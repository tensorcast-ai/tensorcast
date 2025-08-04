#  Copyright (c) 2025, StepCast Team.

import time
import uuid

import pytest

from scstore.proto import global_store_pb2
from tests.python.interaction.utils import FakeContext
from tests.python.interaction.test_basic import _make_servicer


# Test previously failed because Global Store erroneously removed replicas when
# the worker sent an empty inventory.  The recovery logic has been fixed so the
# replica must remain registered.


def test_state_sync_should_not_remove_existing_local_replica(global_store_service):
    """Reproduce bug where a valid local replica is removed after state sync.

    Steps:
    1.  Start a Store-Daemon with HA enabled so that it registers a worker.
    2.  Manually register a GPU replica for that worker directly in the Global Store
        (simulating a previously persisted entry).
    3.  Tell the daemon's connection-manager that it already hosts the replica.
    4.  Perform a *full* state-sync.  Because the daemon currently sends an *empty*
        ``local_replicas`` list the Global Store will think the replica is stale and
        return a ``REMOVE_REPLICA`` change which causes the daemon to forget the
        replica.
    5.  The replica **should** still be considered registered on the daemon, but the
        current implementation removes it – the assertion therefore fails and the
        test is expected to xfail until the bug is fixed.
    """

    # ------------------------------------------------------------------
    # 1) Spin up daemon and wait until worker_id is assigned
    # ------------------------------------------------------------------
    daemon = _make_servicer(enable_global=True, gs_addr=global_store_service._address)

    timeout_s = 2.0
    start = time.time()
    while not daemon.worker_id and time.time() - start < timeout_s:
        time.sleep(0.01)

    assert daemon.worker_id, "worker_id not assigned within timeout – test setup failed"

    # Convenience handles
    worker_id = daemon.worker_id
    node_id = daemon.node_id
    conn_mgr = daemon.connection_manager

    # ------------------------------------------------------------------
    # 2) Manually register a replica for *this* worker directly in the GS
    # ------------------------------------------------------------------
    model_name = "debug-model.ckpt"
    replica_uuid = str(uuid.uuid4())

    mem_info = global_store_pb2.MemoryInfo(
        node_id=node_id,
        node_address="127.0.0.1",
        node_port=daemon.node_port,
        memory_size=2 * 1024 * 1024,  # 2 MiB
        memory_type=global_store_pb2.MemoryType.GPU,
        device_id=0,
        remote_memory_keys=[f"key-{replica_uuid}"],
        buffer_sizes=[2 * 1024 * 1024],
    )

    reg_req = global_store_pb2.RegisterModelReplicaRequest(
        model_name=model_name,
        mem_info=mem_info,
        max_concurrency=1,
        worker_id=worker_id,
    )
    reg_resp = global_store_service.RegisterModelReplica(reg_req, FakeContext())
    assert reg_resp.status == global_store_pb2.Status.OK, "Replica registration failed during test setup"

    # ------------------------------------------------------------------
    # 3) Pretend the daemon already knows about this replica locally
    # ------------------------------------------------------------------
    conn_mgr._add_registered_model(model_name)  # pyright: ignore[reportOptionalMemberAccess]
    assert model_name in conn_mgr.registered_models # pyright: ignore[reportOptionalMemberAccess]

    # ------------------------------------------------------------------
    # 4) Trigger a *full* state synchronisation
    # ------------------------------------------------------------------
    conn_mgr._perform_state_sync(force_full=True)  # pyright: ignore[reportOptionalMemberAccess]

    # ------------------------------------------------------------------
    # 5) Replica should still be in the daemon's registered set – but bug removes it
    # ------------------------------------------------------------------
    assert model_name in conn_mgr.registered_models, (  # pyright: ignore[reportOptionalMemberAccess]
        "Replica was erroneously removed after state sync – bug reproduced"
    )