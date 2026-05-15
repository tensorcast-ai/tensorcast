#  Copyright (c) 2026, TensorCast Team.

"""Group version-set realization coverage."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest

from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.global_store.config.settings import get_config, set_config
from tensorcast.global_store.exceptions import DatabaseError
from tensorcast.global_store.grpc_service import GlobalStoreServicer
from tensorcast.global_store.repositories.key_mapping_repository import (
    KeyMappingRepository,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


def _source_replica_id(part_id: str) -> str:
    suffix = int(part_id) + 1
    return f"00000000-0000-0000-0000-{suffix:012d}"


@pytest.fixture
def group_servicer():
    try:
        original_config = get_config()
    except RuntimeError:
        original_config = None
    set_config(GlobalStoreConfig(group_realization={"enabled": True}))
    servicer = GlobalStoreServicer()
    try:
        yield servicer
    finally:
        servicer.worker_service.close()
        if original_config is not None:
            set_config(original_config)


@pytest.fixture
def explicit_group_servicer():
    try:
        original_config = get_config()
    except RuntimeError:
        original_config = None
    set_config(
        GlobalStoreConfig(
            group_realization={
                "enabled": True,
                "publish_authority_mode": "COORDINATOR_EXPLICIT",
                "min_wait_poll_interval_ms": 1,
                "max_wait_poll_interval_ms": 2,
            }
        )
    )
    servicer = GlobalStoreServicer()
    try:
        yield servicer
    finally:
        servicer.worker_service.close()
        if original_config is not None:
            set_config(original_config)


def _selection(
    artifact_id: str,
    *,
    view_id: str = "",
    selection_hash: bytes | None = None,
) -> common_pb2.ArtifactSelection:
    return common_pb2.ArtifactSelection(
        artifact_id=artifact_id,
        view_id=view_id,
        selection_hash=selection_hash or f"sel:{artifact_id}:{view_id}".encode(),
    )


def _context(
    group_id: str,
    part_id: str,
    *,
    total_parts: int = 2,
) -> global_store_pb2.GroupRealizationContext:
    required_part_ids = [str(index) for index in range(total_parts)]
    return global_store_pb2.GroupRealizationContext(
        group_kind="tp_serving",
        group_id=group_id,
        epoch=7,
        total_parts=total_parts,
        part_id=part_id,
        required_part_ids=required_part_ids,
    )


def _begin_selection(
    servicer: GlobalStoreServicer,
    context_factory,
    *,
    group_id: str,
    part_id: str,
    selection: common_pb2.ArtifactSelection,
    total_parts: int = 2,
):
    return servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(explicit_selection=selection),
            context=_context(group_id, part_id, total_parts=total_parts),
            daemon_id=f"daemon-{part_id}",
            daemon_session_id=f"session-{part_id}",
            worker_id=f"worker-{part_id}",
        ),
        context_factory(),
    )


def _same_selection_part(
    part_id: str,
    artifact_id: str,
    *,
    view_id: str = "",
    byte_space: common_pb2.ByteSpaceRef | None = None,
) -> global_store_pb2.GroupVersionSetPart:
    return global_store_pb2.GroupVersionSetPart(
        part_id=part_id,
        selection=_selection(artifact_id, view_id=view_id),
        requested_byte_space=byte_space
        or common_pb2.ByteSpaceRef(kind=common_pb2.BYTE_SPACE_KIND_CANONICAL),
    )


def _prepared_request(
    *,
    transaction_id: str,
    part_id: str,
    source_export_generation: int = 1,
) -> global_store_pb2.ReportGroupRealizationPreparedRequest:
    return global_store_pb2.ReportGroupRealizationPreparedRequest(
        transaction_id=transaction_id,
        part_id=part_id,
        staged_value=global_store_pb2.StagedBindingValueRef(
            daemon_id=f"daemon-{part_id}",
            daemon_session_id=f"session-{part_id}",
            binding_id=f"binding-{part_id}",
            binding_value_id=f"value-{part_id}",
            staging_token=f"stage-{part_id}",
            staging_epoch=1,
        ),
        daemon_id=f"daemon-{part_id}",
        daemon_session_id=f"session-{part_id}",
        worker_id=f"worker-{part_id}",
        materialization_attempt_id=f"attempt-{part_id}",
        source_replica_id=_source_replica_id(part_id),
        source_export_generation=source_export_generation,
        child_transport_request_id=f"child-{part_id}",
    )


def _prepared_request_without_source(
    *,
    transaction_id: str,
    part_id: str,
) -> global_store_pb2.ReportGroupRealizationPreparedRequest:
    request = _prepared_request(transaction_id=transaction_id, part_id=part_id)
    request.source_replica_id = ""
    request.source_export_generation = 0
    return request


def _register_source_replica(
    group_servicer,
    *,
    part_id: str,
    artifact_id: str,
    export_generation: int = 1,
    is_available: bool = True,
) -> None:
    replica_id = _source_replica_id(part_id)
    group_servicer.connection.execute(
        """
        INSERT INTO artifact_replicas (
            replica_id,
            artifact_id,
            node_id,
            node_address,
            node_port,
            memory_size,
            memory_type,
            device_id,
            is_available,
            remote_memory_keys,
            buffer_sizes,
            export_state,
            export_generation,
            worker_id
        ) VALUES (?, ?, ?, '127.0.0.1', 50051, 1, 'GPU', 0, ?, ['key'], [1],
                  'EXPORTABLE', ?, ?)
        """,
        [
            replica_id,
            artifact_id,
            f"node-{part_id}",
            is_available,
            export_generation,
            f"worker-{part_id}",
        ],
    )
    group_servicer.connection.execute(
        """
        INSERT INTO replica_counters (replica_id, current_requests)
        VALUES (?, 0)
        """,
        [replica_id],
    )


def _group_publish_lease(
    group_servicer,
    transaction_id: str,
    *,
    kind: str = "group_realization_publish",
    target_artifact_id: str | None = None,
) -> str:
    acquired, lease = group_servicer.operation_repository.acquire_lease(
        operation_id=f"publish:{transaction_id}:{kind}:{target_artifact_id or transaction_id}",
        kind=kind,
        target_artifact_id=target_artifact_id or transaction_id,
        owner_id="test-coordinator",
        ttl_ms=60_000,
        initial_state="running",
        initial_status_proto=b"",
    )
    assert acquired is True
    return str(lease["lease_token"])


def test_key_target_history_rejects_in_place_target_change(db_connection) -> None:
    repo = KeyMappingRepository(db_connection)
    repo.upsert(key="model:latest", artifact_id="aid-1")

    with pytest.raises(DatabaseError, match="target changes"):
        repo.upsert(key="model:latest", artifact_id="aid-2")

    result = repo.swap(key="model:latest", new_artifact_id="aid-2")
    assert result["ok"] is True
    assert result["generation"] == 1
    current = repo.get_current_target(key="model:latest")
    assert current is not None
    assert current["artifact_id"] == "aid-2"
    assert current["generation"] == 1
    old = repo.get_target_for_generation(key="model:latest", generation=0)
    assert old is not None
    assert old["artifact_id"] == "aid-1"
    assert repo.audit_target_history_consistency() == []


def test_artifact_target_swap_clears_stale_selection_hash_for_group_resolution(
    group_servicer,
    test_context,
) -> None:
    repo = group_servicer.key_mapping_repository
    repo.upsert(
        key="model:swapped",
        artifact_id="aid-old",
        selection_hash=b"old-selection-hash",
    )

    result = repo.swap(key="model:swapped", new_artifact_id="aid-new")
    assert result["ok"] is True
    assert result["generation"] == 1
    current = repo.get_current_target(key="model:swapped")
    assert current is not None
    assert current["artifact_id"] == "aid-new"
    assert current["selection_hash"] is None
    assert repo.audit_target_history_consistency() == []

    response = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                key_reference=global_store_pb2.KeyVersionReference(
                    key="model:swapped",
                    alias="latest",
                )
            ),
            context=_context("group-swapped-key", "0"),
        ),
        type(test_context)(),
    )

    assert response.status == global_store_pb2.STATUS_OK
    assert response.part.selection.artifact_id == "aid-new"
    assert bytes(response.part.selection_hash) != b"old-selection-hash"
    after = repo.get_current_target(key="model:swapped")
    assert after is not None
    assert after["selection_hash"] == bytes(response.part.selection_hash)
    assert repo.audit_target_history_consistency() == []


def test_key_target_history_audit_detects_hash_drift(db_connection) -> None:
    repo = KeyMappingRepository(db_connection)
    repo.upsert(
        key="model:hash-drift",
        artifact_id="aid-drift",
        selection_hash=b"history-selection-hash",
    )

    db_connection.execute(
        "UPDATE key_mappings SET selection_hash = ? WHERE key = ?",
        [b"fast-pointer-drift", "model:hash-drift"],
    )

    drifts = repo.audit_target_history_consistency()
    assert len(drifts) == 1
    assert drifts[0]["key"] == "model:hash-drift"
    assert drifts[0]["fast_selection_hash"] == b"fast-pointer-drift"
    assert drifts[0]["history_selection_hash"] == b"history-selection-hash"


def test_same_selection_transaction_publishes_after_all_required_parts(
    group_servicer,
    test_context,
) -> None:
    selection = _selection("aid-shared")
    first = _begin_selection(
        group_servicer,
        type(test_context),
        group_id="group-same",
        part_id="0",
        selection=selection,
    )
    second = _begin_selection(
        group_servicer,
        type(test_context),
        group_id="group-same",
        part_id="1",
        selection=selection,
    )

    assert first.status == global_store_pb2.STATUS_OK
    assert second.status == global_store_pb2.STATUS_OK
    assert first.transaction_id == second.transaction_id
    assert first.version_set.version_set_id == second.version_set.version_set_id
    assert first.part.selection.artifact_id == "aid-shared"
    assert second.part.selection.artifact_id == "aid-shared"
    assert first.transaction_fingerprint == second.transaction_fingerprint

    _register_source_replica(
        group_servicer,
        part_id="0",
        artifact_id="aid-shared",
    )
    _register_source_replica(
        group_servicer,
        part_id="1",
        artifact_id="aid-shared",
    )

    prepared0 = group_servicer.ReportGroupRealizationPrepared(
        _prepared_request(transaction_id=first.transaction_id, part_id="0"),
        type(test_context)(),
    )
    assert prepared0.status == global_store_pb2.STATUS_OK
    assert prepared0.state == global_store_pb2.GROUP_REALIZATION_STATE_PREPARING

    prepared1 = group_servicer.ReportGroupRealizationPrepared(
        _prepared_request(transaction_id=first.transaction_id, part_id="1"),
        type(test_context)(),
    )
    assert prepared1.status == global_store_pb2.STATUS_OK
    assert prepared1.state == global_store_pb2.GROUP_REALIZATION_STATE_PUBLISHED

    diagnostic = group_servicer.GetGroupRealization(
        global_store_pb2.GetGroupRealizationRequest(
            transaction_id=first.transaction_id
        ),
        type(test_context)(),
    )
    assert diagnostic.status == global_store_pb2.STATUS_OK
    assert diagnostic.state == global_store_pb2.GROUP_REALIZATION_STATE_PUBLISHED
    assert list(diagnostic.missing_part_ids) == []
    assert diagnostic.prepared_count == 2
    assert diagnostic.published_count == 2


def test_register_same_selection_manifest_returns_identical_parts(
    group_servicer,
    test_context,
) -> None:
    register = group_servicer.RegisterGroupVersionSet(
        global_store_pb2.RegisterGroupVersionSetRequest(
            realization_kind=global_store_pb2.GROUP_REALIZATION_KIND_SAME_SELECTION,
            parts=[
                _same_selection_part("0", "aid-controlled"),
                _same_selection_part("1", "aid-controlled"),
            ],
        ),
        type(test_context)(),
    )

    assert register.status == global_store_pb2.STATUS_OK
    assert (
        register.realization_kind
        == global_store_pb2.GROUP_REALIZATION_KIND_SAME_SELECTION
    )
    assert len(register.parts) == 2
    assert {part.part_id for part in register.parts} == {"0", "1"}
    assert {part.selection.artifact_id for part in register.parts} == {"aid-controlled"}
    assert len({bytes(part.selection_hash) for part in register.parts}) == 1

    first = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                explicit_version_set=register.version_set
            ),
            context=_context("group-controlled-same", "0"),
        ),
        type(test_context)(),
    )
    second = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                explicit_version_set=register.version_set
            ),
            context=_context("group-controlled-same", "1"),
        ),
        type(test_context)(),
    )

    assert first.status == global_store_pb2.STATUS_OK
    assert second.status == global_store_pb2.STATUS_OK
    assert first.transaction_id == second.transaction_id
    assert first.version_set.version_set_id == register.version_set.version_set_id
    assert second.version_set.version_set_id == register.version_set.version_set_id
    assert first.part.selection.artifact_id == "aid-controlled"
    assert second.part.selection.artifact_id == "aid-controlled"


def test_register_same_selection_manifest_rejects_mixed_parts(
    group_servicer,
    test_context,
) -> None:
    mixed_selection_context = type(test_context)()
    mixed_selection = group_servicer.RegisterGroupVersionSet(
        global_store_pb2.RegisterGroupVersionSetRequest(
            realization_kind=global_store_pb2.GROUP_REALIZATION_KIND_SAME_SELECTION,
            parts=[
                _same_selection_part("0", "aid-a"),
                _same_selection_part("1", "aid-b"),
            ],
        ),
        mixed_selection_context,
    )
    assert mixed_selection.status == global_store_pb2.STATUS_ERROR
    assert mixed_selection_context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "same_selection requires identical selections" in (
        mixed_selection_context.details
    )

    mixed_byte_space_context = type(test_context)()
    mixed_byte_space = group_servicer.RegisterGroupVersionSet(
        global_store_pb2.RegisterGroupVersionSetRequest(
            realization_kind=global_store_pb2.GROUP_REALIZATION_KIND_SAME_SELECTION,
            parts=[
                _same_selection_part("0", "aid-a"),
                _same_selection_part(
                    "1",
                    "aid-a",
                    byte_space=common_pb2.ByteSpaceRef(
                        kind=common_pb2.BYTE_SPACE_KIND_VIEW,
                        id="view-a",
                    ),
                ),
            ],
        ),
        mixed_byte_space_context,
    )
    assert mixed_byte_space.status == global_store_pb2.STATUS_ERROR
    assert mixed_byte_space_context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "same_selection requires identical byte spaces" in (
        mixed_byte_space_context.details
    )


def test_auto_publish_handles_parallel_distinct_group_transactions(
    group_servicer,
    test_context,
) -> None:
    total_parts = 4
    total_transactions = 24
    context_type = type(test_context)

    def realize(index: int) -> str:
        group_id = f"group-parallel-{index}"
        selection = _selection(f"aid-parallel-{index}")
        transaction_id = ""
        for part_index in range(total_parts):
            part_id = str(part_index)
            response = _begin_selection(
                group_servicer,
                context_type,
                group_id=group_id,
                part_id=part_id,
                selection=selection,
                total_parts=total_parts,
            )
            assert response.status == global_store_pb2.STATUS_OK
            if not transaction_id:
                transaction_id = response.transaction_id
            assert response.transaction_id == transaction_id

        for part_index in range(total_parts):
            part_id = str(part_index)
            prepared = group_servicer.ReportGroupRealizationPrepared(
                _prepared_request_without_source(
                    transaction_id=transaction_id,
                    part_id=part_id,
                ),
                context_type(),
            )
            assert prepared.status == global_store_pb2.STATUS_OK

        diagnostic = group_servicer.GetGroupRealization(
            global_store_pb2.GetGroupRealizationRequest(transaction_id=transaction_id),
            context_type(),
        )
        assert diagnostic.status == global_store_pb2.STATUS_OK
        assert diagnostic.state == global_store_pb2.GROUP_REALIZATION_STATE_PUBLISHED
        assert diagnostic.prepared_count == total_parts
        assert diagnostic.published_count == total_parts
        return transaction_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        transaction_ids = list(executor.map(realize, range(total_transactions)))

    assert len(set(transaction_ids)) == total_transactions


def test_group_realization_ignores_transport_group_metadata_for_semantic_identity(
    group_servicer,
    test_context,
) -> None:
    group_servicer.connection.execute(
        """
        INSERT INTO artifact_transports (
            transport_id,
            replica_id,
            artifact_id,
            requested_view_id,
            source_node_id,
            source_address,
            source_port,
            request_id,
            group_id,
            group_kind,
            group_total_parts,
            group_part_id,
            group_epoch
        ) VALUES (
            '00000000-0000-0000-0000-000000000901',
            '00000000-0000-0000-0000-000000000902',
            'transport-artifact',
            'transport-view',
            'node-transport',
            '127.0.0.1',
            50051,
            'transport-request-ignored',
            'group-transport-metadata',
            'tp_serving',
            2,
            '0',
            7
        )
        """
    )

    first = _begin_selection(
        group_servicer,
        type(test_context),
        group_id="group-transport-metadata",
        part_id="0",
        selection=_selection("semantic-artifact"),
    )
    second = _begin_selection(
        group_servicer,
        type(test_context),
        group_id="group-transport-metadata",
        part_id="1",
        selection=_selection("semantic-artifact"),
    )

    assert first.status == global_store_pb2.STATUS_OK
    assert second.status == global_store_pb2.STATUS_OK
    assert first.transaction_id == second.transaction_id
    assert first.part.selection.artifact_id == "semantic-artifact"
    assert second.part.selection.artifact_id == "semantic-artifact"


def test_per_part_manifest_returns_part_specific_selection(
    group_servicer,
    test_context,
) -> None:
    register = group_servicer.RegisterGroupVersionSet(
        global_store_pb2.RegisterGroupVersionSetRequest(
            realization_kind=global_store_pb2.GROUP_REALIZATION_KIND_PER_PART_SELECTION,
            parts=[
                global_store_pb2.GroupVersionSetPart(
                    part_id="0",
                    selection=_selection("aid-rank-0"),
                    requested_byte_space=common_pb2.ByteSpaceRef(
                        kind=common_pb2.BYTE_SPACE_KIND_CANONICAL
                    ),
                ),
                global_store_pb2.GroupVersionSetPart(
                    part_id="1",
                    selection=_selection("aid-rank-1", view_id="view-1"),
                    requested_byte_space=common_pb2.ByteSpaceRef(
                        kind=common_pb2.BYTE_SPACE_KIND_VIEW,
                        id="view-1",
                    ),
                ),
            ],
        ),
        type(test_context)(),
    )
    assert register.status == global_store_pb2.STATUS_OK

    first = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                explicit_version_set=register.version_set
            ),
            context=_context("group-per-part", "0"),
        ),
        type(test_context)(),
    )
    second = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                explicit_version_set=register.version_set
            ),
            context=_context("group-per-part", "1"),
        ),
        type(test_context)(),
    )

    assert first.status == global_store_pb2.STATUS_OK
    assert second.status == global_store_pb2.STATUS_OK
    assert first.version_set.version_set_id == second.version_set.version_set_id
    assert first.part.selection.artifact_id == "aid-rank-0"
    assert second.part.selection.artifact_id == "aid-rank-1"
    assert second.part.selection.view_id == "view-1"


def test_key_reference_join_uses_frozen_version_set_after_key_swap(
    group_servicer,
    test_context,
) -> None:
    group_servicer.key_mapping_repository.upsert(
        key="model:frozen",
        artifact_id="aid-before",
    )
    first = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                key_reference=global_store_pb2.KeyVersionReference(
                    key="model:frozen",
                    alias="latest",
                )
            ),
            context=_context("group-key-freeze", "0"),
        ),
        type(test_context)(),
    )
    assert first.status == global_store_pb2.STATUS_OK
    assert first.part.selection.artifact_id == "aid-before"

    swap = group_servicer.key_mapping_repository.swap(
        key="model:frozen",
        new_artifact_id="aid-after",
    )
    assert swap["ok"] is True

    second = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                key_reference=global_store_pb2.KeyVersionReference(
                    key="model:frozen",
                    alias="latest",
                )
            ),
            context=_context("group-key-freeze", "1"),
        ),
        type(test_context)(),
    )
    assert second.status == global_store_pb2.STATUS_OK
    assert second.version_set.version_set_id == first.version_set.version_set_id
    assert second.part.selection.artifact_id == "aid-before"


def test_key_reference_replay_rejects_new_expected_generation_after_key_swap(
    group_servicer,
    test_context,
) -> None:
    group_servicer.key_mapping_repository.upsert(
        key="model:frozen-generation",
        artifact_id="aid-before",
    )
    first = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                key_reference=global_store_pb2.KeyVersionReference(
                    key="model:frozen-generation",
                    alias="latest",
                )
            ),
            context=_context("group-key-generation-freeze", "0"),
        ),
        type(test_context)(),
    )
    assert first.status == global_store_pb2.STATUS_OK
    assert first.key_generation == 0

    swap = group_servicer.key_mapping_repository.swap(
        key="model:frozen-generation",
        new_artifact_id="aid-after",
    )
    assert swap["ok"] is True
    assert swap["generation"] == 1

    replay_context = type(test_context)()
    replay = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                key_reference=global_store_pb2.KeyVersionReference(
                    key="model:frozen-generation",
                    alias="latest",
                    expected_generation=1,
                )
            ),
            context=_context("group-key-generation-freeze", "1"),
        ),
        replay_context,
    )
    assert replay.status == global_store_pb2.STATUS_ERROR
    assert replay_context.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "key generation mismatch" in str(replay_context.details)


def test_explicit_conflict_on_same_semantic_slot_fails_before_materialization(
    group_servicer,
    test_context,
) -> None:
    first = _begin_selection(
        group_servicer,
        type(test_context),
        group_id="group-conflict",
        part_id="0",
        selection=_selection("aid-a"),
    )
    assert first.status == global_store_pb2.STATUS_OK

    conflict_context = type(test_context)()
    conflict = _begin_selection(
        group_servicer,
        lambda: conflict_context,
        group_id="group-conflict",
        part_id="1",
        selection=_selection("aid-b"),
    )
    assert conflict.status == global_store_pb2.STATUS_ERROR
    assert conflict_context.code == grpc.StatusCode.ALREADY_EXISTS


def test_existing_semantic_slot_rejects_different_required_part_set(
    group_servicer,
    test_context,
) -> None:
    first = _begin_selection(
        group_servicer,
        type(test_context),
        group_id="group-required-conflict",
        part_id="0",
        selection=_selection("aid-required-conflict"),
        total_parts=2,
    )
    assert first.status == global_store_pb2.STATUS_OK

    conflict_context = type(test_context)()
    conflict = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                explicit_version_set=first.version_set
            ),
            context=global_store_pb2.GroupRealizationContext(
                group_kind="tp_serving",
                group_id="group-required-conflict",
                epoch=7,
                total_parts=1,
                part_id="0",
                required_part_ids=["0"],
            ),
        ),
        conflict_context,
    )

    assert conflict.status == global_store_pb2.STATUS_ERROR
    assert conflict_context.code == grpc.StatusCode.ALREADY_EXISTS


def test_existing_part_join_rejects_different_daemon_identity(
    group_servicer,
    test_context,
) -> None:
    first = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                explicit_selection=_selection("aid-join-owner")
            ),
            context=_context("group-join-owner", "0"),
            daemon_id="daemon-a",
            daemon_session_id="session-a",
            worker_id="worker-a",
        ),
        type(test_context)(),
    )
    assert first.status == global_store_pb2.STATUS_OK

    conflict_context = type(test_context)()
    conflict = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                explicit_version_set=first.version_set
            ),
            context=_context("group-join-owner", "0"),
            daemon_id="daemon-b",
            daemon_session_id="session-b",
            worker_id="worker-b",
        ),
        conflict_context,
    )

    assert conflict.status == global_store_pb2.STATUS_ERROR
    assert conflict_context.code == grpc.StatusCode.ALREADY_EXISTS


def test_key_reference_strict_resolution_backfills_migrated_selection_hash(
    group_servicer,
    test_context,
) -> None:
    group_servicer.key_mapping_repository.upsert(
        key="model:migrated",
        artifact_id="aid-migrated",
    )
    before = group_servicer.key_mapping_repository.get_current_target(
        key="model:migrated"
    )
    assert before is not None
    assert before["selection_hash"] is None

    response = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                key_reference=global_store_pb2.KeyVersionReference(
                    key="model:migrated",
                    alias="latest",
                )
            ),
            context=_context("group-key-hash", "0"),
        ),
        type(test_context)(),
    )

    assert response.status == global_store_pb2.STATUS_OK
    after = group_servicer.key_mapping_repository.get_current_target(
        key="model:migrated"
    )
    assert after is not None
    assert after["selection_hash"] is not None


def test_member_fingerprint_rejects_conflicting_duplicate_prepared_report(
    explicit_group_servicer,
    test_context,
) -> None:
    selection = _selection("aid-duplicate")
    first = _begin_selection(
        explicit_group_servicer,
        type(test_context),
        group_id="group-duplicate",
        part_id="0",
        selection=selection,
    )
    assert first.status == global_store_pb2.STATUS_OK
    _register_source_replica(
        explicit_group_servicer,
        part_id="0",
        artifact_id="aid-duplicate",
    )

    prepared = explicit_group_servicer.ReportGroupRealizationPrepared(
        _prepared_request(transaction_id=first.transaction_id, part_id="0"),
        type(test_context)(),
    )
    assert prepared.status == global_store_pb2.STATUS_OK

    conflicting = _prepared_request(transaction_id=first.transaction_id, part_id="0")
    conflicting.materialization_attempt_id = "attempt-conflict"
    conflict_context = type(test_context)()
    conflict = explicit_group_servicer.ReportGroupRealizationPrepared(
        conflicting,
        conflict_context,
    )
    assert conflict.status == global_store_pb2.STATUS_ERROR
    assert conflict_context.code == grpc.StatusCode.ALREADY_EXISTS


def test_explicit_publish_requires_authority_and_all_parts_ready(
    explicit_group_servicer,
    test_context,
) -> None:
    selection = _selection("aid-explicit")
    first = _begin_selection(
        explicit_group_servicer,
        type(test_context),
        group_id="group-explicit",
        part_id="0",
        selection=selection,
    )
    second = _begin_selection(
        explicit_group_servicer,
        type(test_context),
        group_id="group-explicit",
        part_id="1",
        selection=selection,
    )
    assert first.status == global_store_pb2.STATUS_OK
    assert second.status == global_store_pb2.STATUS_OK
    _register_source_replica(
        explicit_group_servicer,
        part_id="0",
        artifact_id="aid-explicit",
    )
    _register_source_replica(
        explicit_group_servicer,
        part_id="1",
        artifact_id="aid-explicit",
    )

    explicit_group_servicer.ReportGroupRealizationPrepared(
        _prepared_request(transaction_id=first.transaction_id, part_id="0"),
        type(test_context)(),
    )
    lease_token = _group_publish_lease(
        explicit_group_servicer,
        first.transaction_id,
    )
    not_ready_context = type(test_context)()
    not_ready = explicit_group_servicer.PublishGroupRealization(
        global_store_pb2.PublishGroupRealizationRequest(
            transaction_id=first.transaction_id,
            require_ready_to_publish=True,
            authority=global_store_pb2.GroupPublishAuthority(
                operation_lease_id=lease_token,
            ),
        ),
        not_ready_context,
    )
    assert not_ready.status == global_store_pb2.STATUS_ERROR
    assert not_ready_context.code == grpc.StatusCode.INVALID_ARGUMENT

    explicit_group_servicer.ReportGroupRealizationPrepared(
        _prepared_request(transaction_id=first.transaction_id, part_id="1"),
        type(test_context)(),
    )
    unauthorized_context = type(test_context)()
    unauthorized = explicit_group_servicer.PublishGroupRealization(
        global_store_pb2.PublishGroupRealizationRequest(
            transaction_id=first.transaction_id,
            require_ready_to_publish=True,
        ),
        unauthorized_context,
    )
    assert unauthorized.status == global_store_pb2.STATUS_ERROR
    assert unauthorized_context.code == grpc.StatusCode.PERMISSION_DENIED

    capability_context = type(test_context)()
    capability = explicit_group_servicer.PublishGroupRealization(
        global_store_pb2.PublishGroupRealizationRequest(
            transaction_id=first.transaction_id,
            require_ready_to_publish=True,
            authority=global_store_pb2.GroupPublishAuthority(
                capability_token=b"coordinator-token"
            ),
        ),
        capability_context,
    )
    assert capability.status == global_store_pb2.STATUS_ERROR
    assert capability_context.code == grpc.StatusCode.PERMISSION_DENIED

    wrong_kind_context = type(test_context)()
    wrong_kind = explicit_group_servicer.PublishGroupRealization(
        global_store_pb2.PublishGroupRealizationRequest(
            transaction_id=first.transaction_id,
            require_ready_to_publish=True,
            authority=global_store_pb2.GroupPublishAuthority(
                operation_lease_id=_group_publish_lease(
                    explicit_group_servicer,
                    first.transaction_id,
                    kind="unrelated_operation",
                )
            ),
        ),
        wrong_kind_context,
    )
    assert wrong_kind.status == global_store_pb2.STATUS_ERROR
    assert wrong_kind_context.code == grpc.StatusCode.PERMISSION_DENIED

    wrong_target_context = type(test_context)()
    wrong_target = explicit_group_servicer.PublishGroupRealization(
        global_store_pb2.PublishGroupRealizationRequest(
            transaction_id=first.transaction_id,
            require_ready_to_publish=True,
            authority=global_store_pb2.GroupPublishAuthority(
                operation_lease_id=_group_publish_lease(
                    explicit_group_servicer,
                    first.transaction_id,
                    target_artifact_id="another-transaction",
                )
            ),
        ),
        wrong_target_context,
    )
    assert wrong_target.status == global_store_pb2.STATUS_ERROR
    assert wrong_target_context.code == grpc.StatusCode.PERMISSION_DENIED

    published = explicit_group_servicer.PublishGroupRealization(
        global_store_pb2.PublishGroupRealizationRequest(
            transaction_id=first.transaction_id,
            require_ready_to_publish=True,
            authority=global_store_pb2.GroupPublishAuthority(
                operation_lease_id=lease_token,
            ),
        ),
        type(test_context)(),
    )
    assert published.status == global_store_pb2.STATUS_OK
    assert published.state == global_store_pb2.GROUP_REALIZATION_STATE_PUBLISHED


def test_publish_rejects_stale_source_export_generation(
    explicit_group_servicer,
    test_context,
) -> None:
    selection = _selection("aid-stale")
    first = _begin_selection(
        explicit_group_servicer,
        type(test_context),
        group_id="group-stale-source",
        part_id="0",
        selection=selection,
    )
    second = _begin_selection(
        explicit_group_servicer,
        type(test_context),
        group_id="group-stale-source",
        part_id="1",
        selection=selection,
    )
    _register_source_replica(
        explicit_group_servicer,
        part_id="0",
        artifact_id="aid-stale",
        export_generation=2,
    )
    _register_source_replica(
        explicit_group_servicer,
        part_id="1",
        artifact_id="aid-stale",
    )
    explicit_group_servicer.ReportGroupRealizationPrepared(
        _prepared_request(
            transaction_id=first.transaction_id,
            part_id="0",
            source_export_generation=1,
        ),
        type(test_context)(),
    )
    ready = explicit_group_servicer.ReportGroupRealizationPrepared(
        _prepared_request(transaction_id=first.transaction_id, part_id="1"),
        type(test_context)(),
    )
    assert ready.state == global_store_pb2.GROUP_REALIZATION_STATE_READY_TO_PUBLISH

    publish_context = type(test_context)()
    publish = explicit_group_servicer.PublishGroupRealization(
        global_store_pb2.PublishGroupRealizationRequest(
            transaction_id=first.transaction_id,
            require_ready_to_publish=True,
            authority=global_store_pb2.GroupPublishAuthority(
                operation_lease_id=_group_publish_lease(
                    explicit_group_servicer,
                    first.transaction_id,
                )
            ),
        ),
        publish_context,
    )
    assert second.transaction_id == first.transaction_id
    assert publish.status == global_store_pb2.STATUS_ERROR
    assert publish_context.code == grpc.StatusCode.ALREADY_EXISTS

    diagnostic = explicit_group_servicer.GetGroupRealization(
        global_store_pb2.GetGroupRealizationRequest(
            transaction_id=first.transaction_id
        ),
        type(test_context)(),
    )
    assert diagnostic.state == global_store_pb2.GROUP_REALIZATION_STATE_ABORTED
    assert diagnostic.failure_code == "source_visibility_stale"


def test_publish_rejects_source_marked_unavailable_via_lifecycle_rpc(
    explicit_group_servicer,
    test_context,
) -> None:
    selection = _selection("aid-source-unavailable")
    first = _begin_selection(
        explicit_group_servicer,
        type(test_context),
        group_id="group-source-unavailable",
        part_id="0",
        selection=selection,
    )
    second = _begin_selection(
        explicit_group_servicer,
        type(test_context),
        group_id="group-source-unavailable",
        part_id="1",
        selection=selection,
    )
    assert first.status == global_store_pb2.STATUS_OK
    assert second.status == global_store_pb2.STATUS_OK
    _register_source_replica(
        explicit_group_servicer,
        part_id="0",
        artifact_id="aid-source-unavailable",
    )
    _register_source_replica(
        explicit_group_servicer,
        part_id="1",
        artifact_id="aid-source-unavailable",
    )

    explicit_group_servicer.ReportGroupRealizationPrepared(
        _prepared_request(transaction_id=first.transaction_id, part_id="0"),
        type(test_context)(),
    )
    explicit_group_servicer.ReportGroupRealizationPrepared(
        _prepared_request(transaction_id=first.transaction_id, part_id="1"),
        type(test_context)(),
    )

    unavailable = explicit_group_servicer.MarkReplicaUnavailable(
        global_store_pb2.MarkReplicaUnavailableRequest(
            artifact_id="aid-source-unavailable",
            replica_id=_source_replica_id("0"),
            reason="group-source-fence-test",
        ),
        type(test_context)(),
    )
    assert unavailable.status == global_store_pb2.STATUS_OK
    assert unavailable.updated is True

    publish_context = type(test_context)()
    publish = explicit_group_servicer.PublishGroupRealization(
        global_store_pb2.PublishGroupRealizationRequest(
            transaction_id=first.transaction_id,
            require_ready_to_publish=True,
            authority=global_store_pb2.GroupPublishAuthority(
                operation_lease_id=_group_publish_lease(
                    explicit_group_servicer,
                    first.transaction_id,
                )
            ),
        ),
        publish_context,
    )
    assert publish.status == global_store_pb2.STATUS_ERROR
    assert publish_context.code == grpc.StatusCode.ALREADY_EXISTS

    diagnostic = explicit_group_servicer.GetGroupRealization(
        global_store_pb2.GetGroupRealizationRequest(
            transaction_id=first.transaction_id
        ),
        type(test_context)(),
    )
    assert diagnostic.state == global_store_pb2.GROUP_REALIZATION_STATE_ABORTED
    assert diagnostic.failure_code == "source_visibility_stale"


def test_publish_rejects_source_unregistered_via_lifecycle_rpc(
    explicit_group_servicer,
    test_context,
) -> None:
    selection = _selection("aid-source-unregistered")
    first = _begin_selection(
        explicit_group_servicer,
        type(test_context),
        group_id="group-source-unregistered",
        part_id="0",
        selection=selection,
    )
    second = _begin_selection(
        explicit_group_servicer,
        type(test_context),
        group_id="group-source-unregistered",
        part_id="1",
        selection=selection,
    )
    assert first.status == global_store_pb2.STATUS_OK
    assert second.status == global_store_pb2.STATUS_OK
    _register_source_replica(
        explicit_group_servicer,
        part_id="0",
        artifact_id="aid-source-unregistered",
    )
    _register_source_replica(
        explicit_group_servicer,
        part_id="1",
        artifact_id="aid-source-unregistered",
    )

    explicit_group_servicer.ReportGroupRealizationPrepared(
        _prepared_request(transaction_id=first.transaction_id, part_id="0"),
        type(test_context)(),
    )
    explicit_group_servicer.ReportGroupRealizationPrepared(
        _prepared_request(transaction_id=first.transaction_id, part_id="1"),
        type(test_context)(),
    )

    unregister = explicit_group_servicer.UnregisterReplica(
        global_store_pb2.UnregisterReplicaRequest(
            artifact_id="aid-source-unregistered",
            replica_id=_source_replica_id("0"),
        ),
        type(test_context)(),
    )
    assert unregister.status == global_store_pb2.STATUS_OK

    publish_context = type(test_context)()
    publish = explicit_group_servicer.PublishGroupRealization(
        global_store_pb2.PublishGroupRealizationRequest(
            transaction_id=first.transaction_id,
            require_ready_to_publish=True,
            authority=global_store_pb2.GroupPublishAuthority(
                operation_lease_id=_group_publish_lease(
                    explicit_group_servicer,
                    first.transaction_id,
                )
            ),
        ),
        publish_context,
    )
    assert publish.status == global_store_pb2.STATUS_ERROR
    assert publish_context.code == grpc.StatusCode.ALREADY_EXISTS

    diagnostic = explicit_group_servicer.GetGroupRealization(
        global_store_pb2.GetGroupRealizationRequest(
            transaction_id=first.transaction_id
        ),
        type(test_context)(),
    )
    assert diagnostic.state == global_store_pb2.GROUP_REALIZATION_STATE_ABORTED
    assert diagnostic.failure_code == "source_visibility_stale"


def test_wait_group_realization_published_deadline_zero_is_non_waiting(
    group_servicer,
    test_context,
) -> None:
    response = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                explicit_selection=_selection("aid-no-wait")
            ),
            context=_context("group-no-wait", "0"),
        ),
        type(test_context)(),
    )
    assert response.status == global_store_pb2.STATUS_OK

    wait = group_servicer.WaitGroupRealizationPublished(
        global_store_pb2.WaitGroupRealizationPublishedRequest(
            transaction_id=response.transaction_id,
            deadline_unix_nanos=0,
        ),
        type(test_context)(),
    )
    assert wait.status == global_store_pb2.STATUS_TIMED_OUT
    assert wait.state in {
        global_store_pb2.GROUP_REALIZATION_STATE_OPEN,
        global_store_pb2.GROUP_REALIZATION_STATE_RESOLVED,
        global_store_pb2.GROUP_REALIZATION_STATE_PREPARING,
        global_store_pb2.GROUP_REALIZATION_STATE_READY_TO_PUBLISH,
    }

    diagnostic = group_servicer.GetGroupRealization(
        global_store_pb2.GetGroupRealizationRequest(
            transaction_id=response.transaction_id
        ),
        type(test_context)(),
    )
    assert diagnostic.status == global_store_pb2.STATUS_OK
    assert diagnostic.state == wait.state


def test_group_realization_deadline_expiration_is_diagnostic_terminal(
    group_servicer,
    test_context,
) -> None:
    response = group_servicer.BeginOrJoinGroupRealization(
        global_store_pb2.BeginOrJoinGroupRealizationRequest(
            version=global_store_pb2.VersionReference(
                explicit_selection=_selection("aid-expire")
            ),
            context=_context("group-expire", "0"),
            deadline_unix_nanos=time.time_ns() - 1,
        ),
        type(test_context)(),
    )
    assert response.status == global_store_pb2.STATUS_OK

    diagnostic = group_servicer.GetGroupRealization(
        global_store_pb2.GetGroupRealizationRequest(
            transaction_id=response.transaction_id
        ),
        type(test_context)(),
    )
    assert diagnostic.status == global_store_pb2.STATUS_OK
    assert diagnostic.state == global_store_pb2.GROUP_REALIZATION_STATE_EXPIRED
    assert diagnostic.deadline_unix_nanos > 0
    assert diagnostic.publish_authority_mode == "AUTO_WHEN_READY"


def test_group_realization_deadline_is_capped_by_transaction_ttl(test_context) -> None:
    try:
        original_config = get_config()
    except RuntimeError:
        original_config = None
    set_config(
        GlobalStoreConfig(
            group_realization={
                "enabled": True,
                "default_deadline_ms": 10_000,
                "transaction_ttl_ms": 1_000,
            }
        )
    )
    servicer = GlobalStoreServicer()
    try:
        before_ns = time.time_ns()
        response = servicer.BeginOrJoinGroupRealization(
            global_store_pb2.BeginOrJoinGroupRealizationRequest(
                version=global_store_pb2.VersionReference(
                    explicit_selection=_selection("aid-deadline-cap")
                ),
                context=_context("group-deadline-cap", "0"),
                deadline_unix_nanos=before_ns + 3_600_000_000_000,
            ),
            type(test_context)(),
        )
        after_ns = time.time_ns()
        assert response.status == global_store_pb2.STATUS_OK

        diagnostic = servicer.GetGroupRealization(
            global_store_pb2.GetGroupRealizationRequest(
                transaction_id=response.transaction_id
            ),
            type(test_context)(),
        )

        assert diagnostic.status == global_store_pb2.STATUS_OK
        assert diagnostic.deadline_unix_nanos >= before_ns
        assert diagnostic.deadline_unix_nanos <= after_ns + 1_000_000_000
    finally:
        servicer.worker_service.close()
        if original_config is not None:
            set_config(original_config)


def test_group_realization_schema_constraints_and_indexes(db_connection) -> None:
    tables = {row[0].lower() for row in db_connection.execute("SHOW TABLES").fetchall()}
    assert {
        "key_version_targets",
        "group_version_sets",
        "group_version_set_parts",
        "group_realization_transactions",
        "group_realization_members",
    }.issubset(tables)

    indexes = {
        row[0].lower()
        for row in db_connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    }
    assert {
        "idx_key_version_targets_artifact",
        "idx_group_version_set_parts_artifact",
        "idx_group_realization_transactions_state_deadline",
        "idx_group_realization_members_state",
    }.issubset(indexes)

    with pytest.raises(Exception):  # noqa: B017
        db_connection.execute(
            """
            INSERT INTO key_version_targets (
                namespace,
                key,
                generation,
                target_kind,
                artifact_id,
                group_version_set_id
            ) VALUES ('', 'bad-key', 0, 'artifact_selection', NULL, 'gvs-bad')
            """
        )

    with pytest.raises(Exception):  # noqa: B017
        db_connection.execute(
            """
            INSERT INTO group_realization_transactions (
                transaction_id,
                group_kind,
                group_id,
                epoch,
                version_set_id,
                realization_kind,
                transaction_fingerprint,
                required_part_ids_json,
                total_parts,
                state
            ) VALUES (
                'bad-txn',
                'tp',
                'g',
                1,
                'gvs',
                'same_selection',
                'hash'::BLOB,
                '["0"]',
                1,
                'bad_state'
            )
            """
        )
