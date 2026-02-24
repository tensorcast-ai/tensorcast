#  Copyright (c) 2025-2026, TensorCast Team.

"""ArtifactBinding and KeyMapping RPC coverage for GlobalStoreServicer."""

import grpc
from google.protobuf import duration_pb2

from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.global_store.config.settings import get_config, set_config
from tensorcast.global_store.grpc_service import GlobalStoreServicer
from tensorcast.proto.global_store.v1 import global_store_pb2


def test_upsert_and_get_artifact_binding_round_trip(servicer, test_context):
    upsert_req = global_store_pb2.UpsertArtifactBindingRequest(
        binding=global_store_pb2.ArtifactBinding(
            from_artifact_id="assembly-1",
            to_artifact_id="sealed-1",
            kind=global_store_pb2.ARTIFACT_BINDING_KIND_SEAL,
        )
    )
    upsert_resp = servicer.UpsertArtifactBinding(upsert_req, test_context)

    assert upsert_resp.status == global_store_pb2.Status.STATUS_OK
    assert upsert_resp.created is True
    assert upsert_resp.binding.from_artifact_id == "assembly-1"
    assert upsert_resp.binding.to_artifact_id == "sealed-1"
    assert test_context.code is None

    get_context = type(test_context)()
    get_resp = servicer.GetArtifactBinding(
        global_store_pb2.GetArtifactBindingRequest(artifact_id="assembly-1"),
        get_context,
    )
    assert get_resp.status == global_store_pb2.Status.STATUS_OK
    assert get_resp.binding.from_artifact_id == "assembly-1"
    assert get_resp.binding.to_artifact_id == "sealed-1"
    assert get_context.code is None


def test_upsert_artifact_binding_conflict_returns_failed_precondition(
    servicer,
    test_context,
):
    first_req = global_store_pb2.UpsertArtifactBindingRequest(
        binding=global_store_pb2.ArtifactBinding(
            from_artifact_id="assembly-2",
            to_artifact_id="sealed-2a",
            kind=global_store_pb2.ARTIFACT_BINDING_KIND_SEAL,
        )
    )
    first_resp = servicer.UpsertArtifactBinding(first_req, test_context)
    assert first_resp.status == global_store_pb2.Status.STATUS_OK

    conflict_context = type(test_context)()
    conflict_req = global_store_pb2.UpsertArtifactBindingRequest(
        binding=global_store_pb2.ArtifactBinding(
            from_artifact_id="assembly-2",
            to_artifact_id="sealed-2b",
            kind=global_store_pb2.ARTIFACT_BINDING_KIND_SEAL,
        )
    )
    conflict_resp = servicer.UpsertArtifactBinding(conflict_req, conflict_context)
    assert conflict_resp.status == global_store_pb2.Status.STATUS_ERROR
    assert conflict_context.code == grpc.StatusCode.FAILED_PRECONDITION
    assert "artifact binding conflict" in (conflict_context.details or "")


def test_upsert_key_mapping_conflict_exposes_reason(servicer, test_context):
    create_resp = servicer.UpsertKeyMapping(
        global_store_pb2.UpsertKeyMappingRequest(
            key="model:latest",
            artifact_id="aid-1",
        ),
        test_context,
    )
    assert create_resp.status == global_store_pb2.Status.STATUS_OK

    conflict_context = type(test_context)()
    conflict_resp = servicer.UpsertKeyMapping(
        global_store_pb2.UpsertKeyMappingRequest(
            key="model:latest",
            artifact_id="aid-2",
        ),
        conflict_context,
    )
    assert conflict_resp.status == global_store_pb2.Status.STATUS_ERROR
    assert "aid-1" in conflict_resp.conflict_reason
    assert conflict_context.code is None


def test_resolve_key_mapping_ttl_then_alias_sets_cache_policy(servicer, test_context):
    upsert_req = global_store_pb2.UpsertKeyMappingRequest(
        key="model:stable",
        artifact_id="aid-stable-1",
        ttl=duration_pb2.Duration(seconds=120),
    )
    upsert_resp = servicer.UpsertKeyMapping(upsert_req, test_context)
    assert upsert_resp.status == global_store_pb2.Status.STATUS_OK

    resolve_context = type(test_context)()
    resolve_resp = servicer.ResolveKeyMapping(
        global_store_pb2.ResolveKeyMappingRequest(key="model:stable"),
        resolve_context,
    )
    assert resolve_resp.status == global_store_pb2.Status.STATUS_OK
    assert resolve_resp.artifact_id == "aid-stable-1"
    assert resolve_resp.cache_ttl_seconds == 120

    swap_context = type(test_context)()
    swap_resp = servicer.SwapKeyMapping(
        global_store_pb2.SwapKeyMappingRequest(
            key="model:stable",
            new_artifact_id="aid-stable-2",
        ),
        swap_context,
    )
    assert swap_resp.status == global_store_pb2.Status.STATUS_OK

    resolve_after_context = type(test_context)()
    resolve_after_resp = servicer.ResolveKeyMapping(
        global_store_pb2.ResolveKeyMappingRequest(key="model:stable"),
        resolve_after_context,
    )
    assert resolve_after_resp.status == global_store_pb2.Status.STATUS_OK
    assert resolve_after_resp.artifact_id == "aid-stable-2"
    assert resolve_after_resp.cache_ttl_seconds == 1


def test_swap_key_mapping_generation_mismatch_returns_error(servicer, test_context):
    upsert_resp = servicer.UpsertKeyMapping(
        global_store_pb2.UpsertKeyMappingRequest(
            key="model:canary",
            artifact_id="aid-canary-1",
        ),
        test_context,
    )
    assert upsert_resp.status == global_store_pb2.Status.STATUS_OK

    mismatch_context = type(test_context)()
    mismatch_resp = servicer.SwapKeyMapping(
        global_store_pb2.SwapKeyMappingRequest(
            key="model:canary",
            new_artifact_id="aid-canary-2",
            expected_generation=99,
        ),
        mismatch_context,
    )
    assert mismatch_resp.status == global_store_pb2.Status.STATUS_ERROR
    assert mismatch_resp.artifact_id == "aid-canary-1"
    assert mismatch_resp.generation == 0


def test_revoke_key_mapping_not_found(servicer, test_context):
    resp = servicer.RevokeKeyMapping(
        global_store_pb2.RevokeKeyMappingRequest(key="missing:key"),
        test_context,
    )
    assert resp.status == global_store_pb2.Status.STATUS_NOT_FOUND


def test_resolve_key_mapping_alias_cache_ttl_can_be_configured(test_context):
    try:
        original_config = get_config()
    except RuntimeError:
        original_config = None
    set_config(
        GlobalStoreConfig(
            key_mapping_policy={"alias_cache_ttl_ms": 3000},
        )
    )
    servicer = GlobalStoreServicer()
    try:
        upsert_resp = servicer.UpsertKeyMapping(
            global_store_pb2.UpsertKeyMappingRequest(
                key="model:alias-config",
                artifact_id="aid-alias-config-1",
            ),
            test_context,
        )
        assert upsert_resp.status == global_store_pb2.Status.STATUS_OK

        swap_resp = servicer.SwapKeyMapping(
            global_store_pb2.SwapKeyMappingRequest(
                key="model:alias-config",
                new_artifact_id="aid-alias-config-2",
            ),
            type(test_context)(),
        )
        assert swap_resp.status == global_store_pb2.Status.STATUS_OK

        resolve_resp = servicer.ResolveKeyMapping(
            global_store_pb2.ResolveKeyMappingRequest(key="model:alias-config"),
            type(test_context)(),
        )
        assert resolve_resp.status == global_store_pb2.Status.STATUS_OK
        assert resolve_resp.cache_ttl_seconds == 3
    finally:
        servicer.worker_service.close()
        if original_config is not None:
            set_config(original_config)
