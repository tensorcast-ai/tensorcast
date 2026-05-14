#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest
import torch

import tensorcast as tc
from tensorcast.api.store.binding import Binding, _reject_live_swap_group_realization
from tensorcast.api.store.binding_state import parse_binding_value_or_raise
from tensorcast.api.store.owned_binding_layout import BindingLayout
from tensorcast.api.store.owned_binding_slot import OwnedBindingSlot
from tensorcast.api.store.types import ArtifactError
from tensorcast.daemon_ctl import _copy_group_realization_options
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2


def test_context_is_pure_data_container() -> None:
    group = tc.GroupRealization(
        group_kind="serving_prefetch",
        group_id="model-a",
        epoch=7,
        total_parts=2,
        part_id="rank0",
        required_part_ids=("rank0", "rank1"),
        require_staged_publish=True,
    )
    ctx = tc.context(
        request_id="req-1",
        qos="background",
        deadline_ms=123,
        idempotency_key="idem-1",
        tags={"stage": "warm", "attempt": 2},
        group_realization=group,
    )
    assert isinstance(ctx, tc.CallContext)
    assert ctx.request_id == "req-1"
    assert ctx.qos == "background"
    assert ctx.deadline_ms == 123
    assert ctx.idempotency_key == "idem-1"
    assert ctx.tags is not None and ctx.tags["stage"] == "warm"
    assert ctx.group_realization == group
    assert not hasattr(ctx, "artifact")

    with pytest.raises(FrozenInstanceError):
        ctx.request_id = "req-2"  # type: ignore[misc]


def test_handle_factories_are_context_free() -> None:
    sig = inspect.signature(tc.artifact)
    assert "ctx" not in sig.parameters


def test_group_realization_context_serializes_to_daemon_options() -> None:
    group = tc.GroupRealization(
        group_kind="serving_prefetch",
        group_id="model-a",
        epoch=7,
        total_parts=2,
        part_id="rank0",
        required_part_ids=("rank0", "rank1"),
        require_staged_publish=True,
        deadline_unix_nanos=123456,
    )
    options = store_daemon_pb2.GroupRealizationOptions()

    _copy_group_realization_options(
        group_realization=group,
        version_selection=common_pb2.ArtifactSelection(artifact_id="aid"),
        target=options,
    )

    assert options.enabled is True
    assert options.require_staged_publish is True
    assert options.deadline_unix_nanos == 123456
    assert options.version.explicit_selection.artifact_id == "aid"
    assert options.group.group_kind == "serving_prefetch"
    assert options.group.group_id == "model-a"
    assert options.group.part_id == "rank0"
    assert list(options.group.required_part_ids) == ["rank0", "rank1"]


def test_group_realization_context_serializes_explicit_version_set() -> None:
    group = tc.GroupRealization(
        group_kind="serving_prefetch",
        group_id="model-a",
        epoch=7,
        total_parts=2,
        part_id="rank0",
        required_part_ids=("rank0", "rank1"),
        require_staged_publish=True,
        version_set=tc.GroupVersionSetRef(
            version_set_id="gvs-1",
            manifest_hash=b"manifest-hash",
            manifest_generation=3,
        ),
    )
    options = store_daemon_pb2.GroupRealizationOptions()

    _copy_group_realization_options(
        group_realization=group,
        version_selection=common_pb2.ArtifactSelection(artifact_id="ignored"),
        target=options,
    )

    assert options.version.WhichOneof("value") == "explicit_version_set"
    assert options.version.explicit_version_set.version_set_id == "gvs-1"
    assert options.version.explicit_version_set.manifest_hash == b"manifest-hash"
    assert options.version.explicit_version_set.manifest_generation == 3
    assert options.group.part_id == "rank0"


def test_group_realization_is_rejected_for_live_binding_swap() -> None:
    group = tc.GroupRealization(
        group_kind="serving_prefetch",
        group_id="model-a",
        epoch=7,
        total_parts=2,
        part_id="rank0",
        required_part_ids=("rank0", "rank1"),
        require_staged_publish=True,
    )

    with pytest.raises(ArtifactError, match="Binding.swap group_realization"):
        _reject_live_swap_group_realization(tc.context(group_realization=group))


def test_staged_owned_binding_response_is_not_current_value() -> None:
    value = store_daemon_pb2.BindingValue(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="staged-1",
        seal_generation=0,
        is_artifact_backed=True,
        verification_state=(
            store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_LOCAL_ONLY
        ),
    )
    value.source_artifact_id = "mi2:artifact"
    value.selection.artifact_id = "mi2:artifact"
    metadata = parse_binding_value_or_raise(
        value,
        rpc_name="CreateOwnedBinding staged_value",
        expected_binding_id="binding-1",
        expected_binding_layout_id="layout-1",
    )
    assert metadata is not None
    group_acquire = tc.GroupRealizationAcquireRef(
        transaction_id="txn-1",
        version_set_id="gvs-1",
        part_id="rank-0",
        staging_token="stage-1",
    )

    class _Runtime:
        closed = False

        def ensure_client(self) -> object:
            return object()

    class _Store:
        pass

    slot = OwnedBindingSlot(
        store=_Store(),  # type: ignore[arg-type]
        runtime=_Runtime(),  # type: ignore[arg-type]
        tensors={"weight": object()},  # type: ignore[dict-item]
        layout=BindingLayout(
            binding_layout_id="layout-1",
            target_layout=store_daemon_pb2.TargetLayout(),
            target_index_bytes=b"{}",
        ),
        binding_id="binding-1",
        current_value_metadata=None,
        staged_value_metadata=metadata,
        group_realization_acquire=group_acquire,
        device=torch.device("cuda:0"),
        device_id=0,
        target_publication_token=None,
    )
    binding = Binding(slot)

    assert binding.current_value is None
    assert binding.staged_value is not None
    assert binding.staged_value.group_realization_acquire == group_acquire
    assert binding.artifact_id == "mi2:artifact"
    with pytest.raises(ArtifactError, match="publish_replica"):
        binding.publish_replica()
    with pytest.raises(ArtifactError, match="Binding.swap"):
        binding.swap("mi2:next")
    with pytest.raises(ArtifactError, match="Binding.realize_from"):
        binding.realize_from("mi2:next", realization_plan=())
