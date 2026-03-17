#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from tensorcast.api._config import PlanType
from tensorcast.api._view_ops import NarrowOp, TransposeOp, ViewSpecBuildResult
from tensorcast.api.context import CallContext
from tensorcast.api.operation import Operation, OperationStatus, PollingOperation
from tensorcast.api.store.binding_state import clone_selection
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.inplace_slot import InplaceSlot, _ctx_timeout_s
from tensorcast.api.store.owned_binding_slot import OwnedBindingSlot
from tensorcast.api.store.types import ArtifactError
from tensorcast.api.store.views import TransformPlacement
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import AssemblyAttemptRef, PartialSealResult

if TYPE_CHECKING:
    from tensorcast.api.store.artifact import Artifact
    from tensorcast.api.store.owned_binding_layout import BindingLayout
    from tensorcast.api.store.runtime import StoreRuntimeContext
    from tensorcast.daemon_ctl import DaemonCtl
    from tensorcast.proto.common.v1 import common_pb2


_TRANSPORT_GROUP_OPID_MARKER = "#tcg:"
_TRANSPORT_GROUP_KIND_TAG = "tc.transport.group.kind"
_TRANSPORT_GROUP_ID_TAG = "tc.transport.group.id"
_TRANSPORT_GROUP_TOTAL_PARTS_TAG = "tc.transport.group.total_parts"
_TRANSPORT_GROUP_PART_ID_TAG = "tc.transport.group.part_id"
_TRANSPORT_GROUP_PRIORITY_TAG = "tc.transport.group.priority"
_TRANSPORT_GROUP_EPOCH_TAG = "tc.transport.group.epoch"
_TRANSPORT_REQUEST_ID_TAG = "tc.transport.request_id"
_CANONICAL_FULL_CONTRIBUTION_VIEW_ID = "__canonical_full__"
_ALLOWED_OPERATION_TOKEN_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:"
)


def _sanitize_operation_token(value: str | None) -> str:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return ""
    return "".join(ch if ch in _ALLOWED_OPERATION_TOKEN_CHARS else "_" for ch in raw)


def _read_context_tag_str(
    tags: Mapping[str, object] | None,
    key: str,
) -> str:
    if not tags:
        return ""
    value = tags.get(key)
    if value is None:
        return ""
    return _sanitize_operation_token(str(value))


def _read_context_tag_int(
    tags: Mapping[str, object] | None,
    key: str,
    *,
    default: int,
) -> int:
    if not tags:
        return default
    value = tags.get(key)
    if value is None:
        return default
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _build_transport_operation_id(
    *,
    base_operation_id: str,
    ctx: CallContext | None,
) -> str:
    if ctx is None:
        return base_operation_id

    tags = ctx.tags
    group_kind = _read_context_tag_str(tags, _TRANSPORT_GROUP_KIND_TAG)
    group_id = _read_context_tag_str(tags, _TRANSPORT_GROUP_ID_TAG)
    part_id = _read_context_tag_str(tags, _TRANSPORT_GROUP_PART_ID_TAG)
    total_parts = _read_context_tag_int(
        tags,
        _TRANSPORT_GROUP_TOTAL_PARTS_TAG,
        default=0,
    )
    priority = _read_context_tag_int(
        tags,
        _TRANSPORT_GROUP_PRIORITY_TAG,
        default=0,
    )
    epoch = _read_context_tag_int(
        tags,
        _TRANSPORT_GROUP_EPOCH_TAG,
        default=0,
    )
    request_id = _read_context_tag_str(tags, _TRANSPORT_REQUEST_ID_TAG)

    if group_kind and group_id and part_id and total_parts > 0:
        if not request_id:
            request_id = _sanitize_operation_token(f"{group_id}:{part_id}")
        metadata = (
            f"kind={group_kind};gid={group_id};tot={int(total_parts)};"
            f"part={part_id};pri={int(priority)};ep={int(epoch)};rid={request_id}"
        )
        return f"{base_operation_id}{_TRANSPORT_GROUP_OPID_MARKER}{metadata}"

    if request_id:
        return f"{base_operation_id}{_TRANSPORT_GROUP_OPID_MARKER}rid={request_id}"

    return base_operation_id


def _clone_view_spec(
    view_spec: common_pb2.ViewSpec | None,
) -> common_pb2.ViewSpec | None:
    if view_spec is None or not view_spec.tensors:
        return None
    cloned = common_pb2.ViewSpec()
    cloned.CopyFrom(view_spec)
    return cloned


def _build_view_result_from_proto(
    view_spec: common_pb2.ViewSpec,
) -> ViewSpecBuildResult:
    tensor_ops: dict[str, tuple[NarrowOp | TransposeOp, ...]] = {}
    for name, ops in sorted(view_spec.tensors.items()):
        entries: list[NarrowOp | TransposeOp] = []
        for op in ops.ops:
            if op.HasField("narrow"):
                entries.append(
                    NarrowOp(
                        dim=int(op.narrow.dim),
                        start=int(op.narrow.start),
                        length=int(op.narrow.length),
                    )
                )
            elif op.HasField("transpose"):
                entries.append(
                    TransposeOp(
                        dim0=int(op.transpose.dim0),
                        dim1=int(op.transpose.dim1),
                    )
                )
        if entries:
            tensor_ops[str(name)] = tuple(entries)
    if not tensor_ops:
        raise ArtifactError(
            "binding contribution requires a non-empty view_spec",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    proto = common_pb2.ViewSpec()
    proto.CopyFrom(view_spec)
    return ViewSpecBuildResult(
        proto=proto,
        tensor_ops=MappingProxyType(tensor_ops),
    )


def _load_artifact_index_bytes(
    binding: "Binding",
    artifact_id: str,
) -> bytes:
    runtime = binding._runtime
    cached = runtime.get_artifact_index_cached(artifact_id)
    if cached is not None and cached.canonical_index_bytes:
        return bytes(cached.canonical_index_bytes)
    return bytes(runtime.ensure_client().get_artifact_index_by_id(artifact_id))


def _slot_contribution_source_artifact_id(
    slot: InplaceSlot | OwnedBindingSlot,
) -> str | None:
    value = getattr(slot, "contribution_source_artifact_id", None)
    if value:
        return str(value)
    metadata = slot.current_value_metadata
    if metadata is not None and metadata.source_artifact_id:
        return str(metadata.source_artifact_id)
    return None


def _slot_contribution_view_spec(
    slot: InplaceSlot | OwnedBindingSlot,
) -> common_pb2.ViewSpec | None:
    view_spec = getattr(slot, "_view_spec", None)
    cloned = _clone_view_spec(view_spec)
    if cloned is not None:
        return cloned
    selection = getattr(slot, "contribution_selection", None)
    if selection is not None and selection.HasField("view_spec"):
        return _clone_view_spec(selection.view_spec)
    live_selection = getattr(slot, "selection", None)
    if live_selection is not None and live_selection.HasField("view_spec"):
        return _clone_view_spec(live_selection.view_spec)
    return None


def _slot_contribution_tensor_names(
    slot: InplaceSlot | OwnedBindingSlot,
) -> tuple[str, ...]:
    selection = getattr(slot, "contribution_selection", None)
    if selection is not None and selection.tensor_names:
        return tuple(str(name) for name in selection.tensor_names)
    live_selection = getattr(slot, "selection", None)
    if live_selection is not None and live_selection.tensor_names:
        return tuple(str(name) for name in live_selection.tensor_names)
    return ()


def _view_spec_is_subset_identity(
    view_spec: common_pb2.ViewSpec,
    canonical_index,
    tensor_names: tuple[str, ...],
) -> bool:
    if not tensor_names:
        return False
    entry_by_name = {str(entry.name): entry for entry in canonical_index.entries}
    if set(view_spec.tensors.keys()) != set(tensor_names):
        return False
    for name in tensor_names:
        entry = entry_by_name.get(str(name))
        ops = view_spec.tensors.get(str(name))
        if entry is None or ops is None or len(ops.ops) != 1 or not entry.shape:
            return False
        op = ops.ops[0]
        if not op.HasField("narrow"):
            return False
        if (
            int(op.narrow.dim) != 0
            or int(op.narrow.start) != 0
            or int(op.narrow.length) != int(entry.shape[0])
        ):
            return False
    return True


def _compute_coverage_plan_hash(
    *,
    contribution_kind: str,
    binding: "Binding",
    view_id: str | None,
    canonical_ranges: tuple[object, ...],
) -> str:
    payload = {
        "binding_layout_id": binding.binding_layout_id,
        "contribution_kind": contribution_kind,
        "view_id": view_id or "",
        "ranges": [[int(item.offset), int(item.length)] for item in canonical_ranges],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"bcp1:{digest}"


def _binding_contribution_plan() -> PlanType:
    return PlanType.VRAM_COALESCED


def _prepare_contribution_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    if os.environ.get("TENSORCAST_CUDA_BACKEND", "").strip() != "fake":
        return tensors
    prepared: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        prepared[name] = tensor.detach().cpu() if tensor.is_cuda else tensor
    return prepared


def _wait_for_events(wait_events: tuple[object, ...] | list[object] | None) -> None:
    if not wait_events:
        return
    for idx, event in enumerate(wait_events):
        if event is None:
            raise ArtifactError(
                f"wait_events[{idx}] must not be None",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        synchronize = getattr(event, "synchronize", None)
        if not callable(synchronize):
            raise ArtifactError(
                f"wait_events[{idx}] must provide synchronize()",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        try:
            synchronize()
        except ArtifactError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(
                f"wait_events[{idx}] synchronize() failed: {exc}",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            ) from exc


class _PublishedLeaseKeepalive:
    def __init__(self, client: "DaemonCtl", ttl_ms: int) -> None:
        self._client = client
        self._ttl_ms = int(ttl_ms)
        self._lease_id: str | None = None
        self._epoch = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self, lease_id: str) -> None:
        if not lease_id or self._ttl_ms <= 0:
            return
        with self._lock:
            if self._lease_id == lease_id and self._thread and self._thread.is_alive():
                return
            self._stop_locked()
            self._epoch = 0
            self._lease_id = lease_id
            self._stop.clear()

            interval = max(1.0, self._ttl_ms / 2000.0)

            def _run() -> None:
                import random

                while not self._stop.wait(interval * (0.9 + 0.2 * random.random())):
                    try:
                        self._client.keep_alive_registered_artifact(
                            lease_id, self._ttl_ms, self._epoch
                        )
                        self._epoch += 1
                    except Exception:  # noqa: BLE001
                        continue

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            self._thread = t

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        thread = self._thread
        self._thread = None
        self._lease_id = None
        self._epoch = 0
        self._stop.set()
        if thread and thread.is_alive():
            thread.join(timeout=1.0)


@dataclass(frozen=True, slots=True)
class BindingUpdateEpoch:
    binding_id: str
    update_epoch: str

    def __str__(self) -> str:
        return self.update_epoch


@dataclass(frozen=True, slots=True)
class SealedBindingValue:
    binding_id: str
    binding_layout_id: str
    binding_value_id: str
    seal_generation: int
    source_artifact_id: str | None
    selection: "common_pb2.ArtifactSelection | None"
    is_artifact_backed: bool
    _binding_ref: weakref.ReferenceType["Binding"] = field(
        repr=False,
        compare=False,
    )

    @property
    def artifact_id(self) -> str | None:
        return self.source_artifact_id

    @property
    def is_current(self) -> bool:
        binding = self._binding_ref()
        current_value = None if binding is None else binding.current_value
        return (
            current_value is not None
            and current_value.binding_value_id == self.binding_value_id
            and current_value.seal_generation == self.seal_generation
        )

    @property
    def is_published(self) -> bool:
        binding = self._binding_ref()
        current_value = None if binding is None else binding.current_value
        return (
            self.is_current
            and current_value is not None
            and binding is not None
            and binding._slot.published_lease_id is not None
        )

    def publish_replica(self, *, ctx: CallContext | None = None) -> None:
        binding = self._require_current_binding()
        binding.publish_replica(ctx=ctx)

    def publish_replica_operation(
        self, *, ctx: CallContext | None = None
    ) -> "Operation[SealedBindingValue]":
        binding = self._require_current_binding()
        return binding.publish_replica_operation(ctx=ctx)

    def activate_key(
        self,
        key: str,
        *,
        expected_active_artifact_id: str | None = None,
        expected_active_generation: int | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        binding = self._require_current_binding()
        binding._activate_key(
            key,
            expected_active_artifact_id=expected_active_artifact_id,
            expected_active_generation=expected_active_generation,
            operation_id=_build_transport_operation_id(
                base_operation_id=uuid.uuid4().hex,
                ctx=ctx,
            ),
            ctx=ctx,
        )

    def contribute_to_assembly(
        self,
        *,
        attempt: AssemblyAttemptRef,
        ctx: CallContext | None = None,
    ) -> PartialSealResult:
        binding = self._require_current_binding()
        slot = binding._slot
        store = slot._store
        pipeline = store._registration

        view_spec = _slot_contribution_view_spec(slot)
        selection_names = _slot_contribution_tensor_names(slot)
        contribution_kind: str
        contributed_view_id: str | None

        if view_spec is not None:
            source_artifact_id = _slot_contribution_source_artifact_id(slot)
            if not source_artifact_id:
                if (
                    slot.layout.target_layout.index_kind
                    != store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW
                ):
                    canonical_index_bytes = bytes(slot.layout.target_index_bytes)
                else:
                    raise ArtifactError(
                        "binding contribution requires a source artifact-backed canonical index",
                        status_code="FAILED_PRECONDITION",
                        retryable=False,
                    )
            else:
                canonical_index_bytes = _load_artifact_index_bytes(
                    binding, source_artifact_id
                )
            canonical_index = canonical_index_from_bytes(canonical_index_bytes)
            if selection_names and _view_spec_is_subset_identity(
                view_spec,
                canonical_index,
                selection_names,
            ):
                view_ctx, upload_tensors = pipeline._build_subset_piece_registration(
                    canonical_index_bytes=canonical_index_bytes,
                    canonical_index=canonical_index,
                    tensor_names=selection_names,
                    view_spec_proto=view_spec,
                    tensors=binding.tensors,
                )
            else:
                build_result = _build_view_result_from_proto(view_spec)
                piece_selection_names = (
                    tuple(build_result.tensor_ops.keys())
                    if build_result.tensor_ops
                    else selection_names
                )
                view_ctx, upload_tensors = pipeline._build_view_registration(
                    canonical_index_bytes=canonical_index_bytes,
                    canonical_index=canonical_index,
                    build_result=build_result,
                    tensors=binding.tensors,
                    placement_enum=TransformPlacement.TRANSFORM_PLACEMENT_SERVER,
                    registration_kind=store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE,
                    selection_names=piece_selection_names,
                )
            registered = pipeline._perform_registration(
                _prepare_contribution_tensors(upload_tensors),
                artifact_id=attempt.assembly_id,
                key=None,
                plan=_binding_contribution_plan(),
                view_registration=view_ctx,
            )
            contribution_kind = "piece_partial"
            reg_result = registered.registration_result
            if reg_result is None or not reg_result.view_id:
                raise ArtifactError(
                    "piece contribution registration did not return a view_id",
                    status_code="DATA_LOSS",
                    retryable=False,
                )
            contributed_view_id = str(reg_result.view_id)
            coverage_plan_hash = _compute_coverage_plan_hash(
                contribution_kind=contribution_kind,
                binding=binding,
                view_id=contributed_view_id,
                canonical_ranges=tuple(reg_result.canonical_ranges),
            )
            submit_kind = store_daemon_pb2.BINDING_CONTRIBUTION_KIND_PIECE_PARTIAL
        else:
            if attempt.expected_view_ids:
                raise ArtifactError(
                    "binding does not expose a piece contribution view_spec for this layout contract",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            pipeline._perform_registration(
                _prepare_contribution_tensors(binding.tensors),
                artifact_id=attempt.assembly_id,
                key=None,
                plan=_binding_contribution_plan(),
            )
            contribution_kind = "canonical_full"
            contributed_view_id = None
            coverage_plan_hash = _compute_coverage_plan_hash(
                contribution_kind=contribution_kind,
                binding=binding,
                view_id=_CANONICAL_FULL_CONTRIBUTION_VIEW_ID,
                canonical_ranges=(),
            )
            submit_kind = store_daemon_pb2.BINDING_CONTRIBUTION_KIND_CANONICAL_FULL

        timeout_s = _ctx_timeout_s(ctx)
        response = binding._runtime.ensure_client().submit_binding_contribution(
            assembly_id=attempt.assembly_id,
            layout_id=attempt.layout_id,
            contribution_contract_hash=attempt.contribution_contract_hash,
            binding_id=self.binding_id,
            binding_value_id=self.binding_value_id,
            coverage_plan_hash=coverage_plan_hash,
            contribution_kind=submit_kind,
            coordinator_operation_id=attempt.coordinator_operation_id,
            coordinator_generation=attempt.coordinator_generation,
            view_id=contributed_view_id,
            timeout_s=timeout_s if timeout_s is not None else 30.0,
        )
        return PartialSealResult(
            assembly_id=attempt.assembly_id,
            binding_id=self.binding_id,
            binding_value_id=self.binding_value_id,
            contribution_kind=contribution_kind,
            view_id=contributed_view_id,
            coverage_plan_hash=coverage_plan_hash,
            accepted=bool(getattr(response, "accepted", False)),
            already_exists=bool(getattr(response, "already_exists", False)),
        )

    def _require_current_binding(self) -> "Binding":
        binding = self._binding_ref()
        if binding is None:
            raise ArtifactError(
                "Binding is no longer available",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not self.is_current:
            raise ArtifactError(
                "SealedBindingValue is no longer current for this binding",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return binding


class Binding:
    """Stable, client-owned CUDA layout that can be refilled in-place."""

    def __init__(
        self,
        slot: InplaceSlot | OwnedBindingSlot,
        *,
        publish: bool = False,
        ctx: CallContext | None = None,
    ) -> None:
        self._slot = slot
        runtime: StoreRuntimeContext = slot._runtime
        self._runtime = runtime
        self._publish_ttl_ms = 0
        self._keepalive = _PublishedLeaseKeepalive(
            runtime.ensure_client(), self._publish_ttl_ms
        )
        if publish:
            self.publish_replica(ctx=ctx)

    def __enter__(self) -> "Binding":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def tensors(self) -> Mapping[str, torch.Tensor]:
        return self._slot.tensors

    @property
    def binding_id(self) -> str:
        return self._slot.binding_id

    @property
    def binding_layout_id(self) -> str:
        return self._slot.binding_layout_id

    @property
    def layout(self) -> "BindingLayout":
        return self._slot.layout

    @property
    def current_value(self) -> SealedBindingValue | None:
        metadata = self._slot.current_value_metadata
        if metadata is None:
            return None
        return SealedBindingValue(
            binding_id=metadata.binding_id,
            binding_layout_id=metadata.binding_layout_id,
            binding_value_id=metadata.binding_value_id,
            seal_generation=metadata.seal_generation,
            source_artifact_id=metadata.source_artifact_id,
            selection=clone_selection(metadata.selection),
            is_artifact_backed=metadata.is_artifact_backed,
            _binding_ref=weakref.ref(self),
        )

    @property
    def artifact_id(self) -> str | None:
        current_value = self.current_value
        if current_value is None or not current_value.is_artifact_backed:
            return None
        return current_value.source_artifact_id

    @property
    def selection(self) -> "common_pb2.ArtifactSelection | None":
        current_value = self.current_value
        if current_value is None or not current_value.is_artifact_backed:
            return None
        return current_value.selection

    def swap(
        self,
        artifact: "Artifact | str",
        *,
        publish: bool = False,
        activate_key: str | None = None,
        expected_active_artifact_id: str | None = None,
        expected_active_generation: int | None = None,
        wait: bool = True,
        drain_timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> SealedBindingValue:
        operation_id = _build_transport_operation_id(
            base_operation_id=uuid.uuid4().hex,
            ctx=ctx,
        )
        self._stop_keepalive()
        try:
            self._slot.swap(
                artifact,
                publish=publish,
                wait=wait,
                drain_timeout_s=drain_timeout_s,
                ctx=ctx,
                operation_id=operation_id,
                publish_ttl_ms=self._publish_ttl_ms if publish else None,
            )
        except Exception:
            if self._slot.published_lease_id:
                self._start_keepalive()
            raise
        if publish:
            self._start_keepalive()
        if activate_key:
            self._activate_key(
                activate_key,
                expected_active_artifact_id=expected_active_artifact_id,
                expected_active_generation=expected_active_generation,
                operation_id=operation_id,
                ctx=ctx,
            )
        current_value = self.current_value
        if current_value is None:
            raise ArtifactError(
                "swap() completed without a current sealed value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        return current_value

    def begin_update(
        self,
        *,
        wait_events: tuple[object, ...] | list[object] | None = None,
        drain_timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> BindingUpdateEpoch:
        _wait_for_events(wait_events)
        self._stop_keepalive()
        return self._slot.begin_update(
            drain_timeout_s=drain_timeout_s,
            ctx=ctx,
        )

    def seal_current(
        self,
        *,
        update_epoch: BindingUpdateEpoch | str | int,
        wait_events: tuple[object, ...] | list[object] | None = None,
        ctx: CallContext | None = None,
    ) -> SealedBindingValue:
        _wait_for_events(wait_events)
        self._stop_keepalive()
        self._slot.seal_current(update_epoch=update_epoch, ctx=ctx)
        current_value = self.current_value
        if current_value is None:
            raise ArtifactError(
                "seal_current() completed without a current sealed value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        return current_value

    def close(self) -> None:
        self._stop_keepalive()
        self._slot.close()

    def retire(
        self,
        *,
        drain_timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        self._stop_keepalive()
        self._slot.retire(
            wait=True,
            drain_timeout_s=drain_timeout_s,
            ctx=ctx,
        )

    def publish_replica(
        self,
        *,
        ctx: CallContext | None = None,
    ) -> SealedBindingValue:
        self._slot.publish_replica(ttl_ms=self._publish_ttl_ms, ctx=ctx)
        self._start_keepalive()
        current_value = self.current_value
        if current_value is None:
            raise ArtifactError(
                "publish_replica() requires a current sealed value",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return current_value

    def publish_replica_operation(
        self,
        *,
        ctx: CallContext | None = None,
    ) -> "Operation[SealedBindingValue]":
        slot_op = self._slot.publish_replica_operation(
            ttl_ms=self._publish_ttl_ms,
            ctx=ctx,
        )

        def _status() -> OperationStatus:
            return slot_op.status()

        def _result() -> SealedBindingValue:
            _ = slot_op.wait()
            self._start_keepalive()
            current_value = self.current_value
            if current_value is None:
                raise ArtifactError(
                    "publish_replica_operation() requires a current sealed value",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            return current_value

        return PollingOperation(
            operation_id=slot_op.operation_id,
            status_fn=_status,
            result_fn=_result,
            cancel_fn=slot_op.cancel,
            ctx=ctx,
        )

    def _start_keepalive(self) -> None:
        lease_id = self._slot.published_lease_id
        if lease_id:
            self._keepalive.start(lease_id)

    def _stop_keepalive(self) -> None:
        self._keepalive.stop()

    def _activate_key(
        self,
        key: str,
        *,
        expected_active_artifact_id: str | None,
        expected_active_generation: int | None,
        operation_id: str,
        ctx: CallContext | None,
    ) -> None:
        if not key:
            raise ArtifactError(
                "activate_key must be non-empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        timeout_s = _ctx_timeout_s(ctx)
        current_artifact_id = self._slot.artifact_id
        if not current_artifact_id:
            raise ArtifactError(
                "activate_key requires an artifact-backed current value",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        result = self._runtime.ensure_client().swap_key_mapping(
            key=key,
            new_artifact_id=current_artifact_id,
            expected_artifact_id=expected_active_artifact_id,
            expected_generation=expected_active_generation,
            operation_id=operation_id,
            timeout_s=timeout_s if timeout_s is not None else 10.0,
        )
        if not result.ok:
            message = (
                f"Activation key '{key}' conflict: current artifact_id="
                f"{result.artifact_id or 'unknown'} generation={result.generation}"
            )
            raise ArtifactError(
                message,
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        self._runtime.invalidate_artifact(None, key=key, reason="activate_key")


__all__ = [
    "Binding",
    "BindingUpdateEpoch",
    "SealedBindingValue",
]
