#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence

import torch

from tensorcast._c_ext import (
    get_cuda_memory_handle,
    get_cuda_memory_handle_with_offset,
)
from tensorcast.api._config import RegisterArtifactOptions, StorePolicy
from tensorcast.api._device import device_uuid_for, resolve_device
from tensorcast.api._materialize import (
    MaterializationPayload,
    materialize_artifact_v2,
)
from tensorcast.api._region_cache import (
    register_region as _cache_register_region,
)
from tensorcast.api._region_cache import (
    unregister_region as _cache_unregister_region,
)
from tensorcast.api._register import RegistrationResult, _register_artifact_core
from tensorcast.api._runtime import require_runtime
from tensorcast.api.context import CallContext
from tensorcast.api.operation import (
    DaemonGlobalStoreOperation,
    Operation,
    OperationError,
    OperationStatus,
    OperationTimeoutError,
    PollingOperation,
)
from tensorcast.api.store.artifact import (
    Artifact,
    ArtifactDescriptor,
    MaterializationDiagnostics,
    PlacementPin,
    PrefetchedReplica,
    TensorDictMaterializationResult,
    TensorMeta,
)
from tensorcast.api.store.async_ops import ArtifactFuture
from tensorcast.api.store.batch_context import (
    BatchContext,
    MaterializationBatcher,
)
from tensorcast.api.store.binding import Binding, BindingUpdateEpoch, SealedBindingValue
from tensorcast.api.store.binding_state import parse_binding_value_or_raise
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.handles import RegisteredArtifact
from tensorcast.api.store.inplace_slot import InplaceSlot, _ctx_timeout_s
from tensorcast.api.store.mapped_binding import (
    CopyPlan,
    CopyPlanEntry,
    Range,
    TargetTensors,
    normalize_copy_plan,
)
from tensorcast.api.store.materialization import MaterializationPipeline
from tensorcast.api.store.owned_binding_layout import BindingLayout
from tensorcast.api.store.owned_binding_slot import (
    OwnedBindingSlot,
    restore_owned_binding_tensors,
)
from tensorcast.api.store.region_utils import collect_storage_bases
from tensorcast.api.store.registration import RegistrationPipeline
from tensorcast.api.store.runtime import (
    StoreRuntimeContext,
    shutdown_context,
)
from tensorcast.api.store.runtime import (
    get_context as get_runtime_context,
)
from tensorcast.api.store.types import (
    ArtifactError,
    ArtifactStatusCode,
    CanonicalIndex,
    CanonicalIndexEntry,
    FallbackOptions,
    LeaseHandle,
    PersistenceShardStatus,
    PersistenceStatusResult,
    ReplicaInfo,
    RetryPolicy,
    StoreCapabilities,
    StoreOptions,
    TensorDict,
)
from tensorcast.api.store.views import TransformPlacement, ViewOrchestrator
from tensorcast.common.identity import ArtifactIdKind, infer_artifact_id_kind
from tensorcast.daemon_ctl import get_daemon_client
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import (
    ArtifactDescriptor as TypedArtifactDescriptor,
)
from tensorcast.types import (
    AssemblyAttemptRef,
    AssemblyCloseoutContract,
    AssemblyReadinessPolicy,
    AssemblyRequirementSetRef,
    DeregisterArtifactOutcome,
    PartialSealResult,
    PublishedModelVersion,
    SealAssemblyResult,
    VramRegionHandle,
)

logger = logging.getLogger(__name__)


def _copy_plan_to_proto(
    mapping: CopyPlan,
) -> store_daemon_pb2.CopyPlan:
    normalized = normalize_copy_plan(mapping)
    proto = store_daemon_pb2.CopyPlan(version=1)
    for entry in normalized:
        entry_proto = proto.entries.add()
        entry_proto.ckpt_name = str(entry.ckpt_name)
        entry_proto.dst_name = str(entry.dst_name)
        if entry.ckpt_range is not None:
            entry_proto.ckpt_range.dim = int(entry.ckpt_range.dim)
            entry_proto.ckpt_range.start = int(entry.ckpt_range.start)
            entry_proto.ckpt_range.end = int(entry.ckpt_range.end)
        if entry.dst_range is not None:
            entry_proto.dst_range.dim = int(entry.dst_range.dim)
            entry_proto.dst_range.start = int(entry.dst_range.start)
            entry_proto.dst_range.end = int(entry.dst_range.end)
    return proto


def _normalize_target_layout_contract(
    target_layout: store_daemon_pb2.TargetLayout,
) -> tuple[
    int,
    int,
    int,
    str,
    bytes,
    tuple[tuple[int, int], ...],
    tuple[tuple[str, int, int, int], ...],
]:
    storage_order = {
        str(storage.storage_id): idx
        for idx, storage in enumerate(target_layout.storages)
    }
    storages = tuple(
        (int(storage.device_id), int(storage.storage_length))
        for storage in target_layout.storages
    )
    offsets = tuple(
        sorted(
            (
                str(offset.name),
                int(storage_order[str(offset.storage_id)]),
                int(offset.storage_offset),
                int(offset.logical_length),
            )
            for offset in target_layout.offsets
        )
    )
    return (
        int(target_layout.layout_kind),
        int(target_layout.index_kind),
        int(target_layout.tensor_spec_kind),
        str(target_layout.view_id or ""),
        bytes(target_layout.logical_layout_hash),
        storages,
        offsets,
    )


def _validate_client_binding_targets(
    *,
    layout: BindingLayout,
    target_tensors: Mapping[str, torch.Tensor],
    device_id: int,
    pipeline: MaterializationPipeline,
    expected_index: CanonicalIndex,
) -> None:
    if layout.target_layout.index_kind == store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW:
        selection_order = tuple(
            str(offset.name) for offset in layout.target_layout.offsets
        )
        derived_layout = pipeline._build_region_backed_layout(
            canonical_index=expected_index,
            canonical_index_bytes=layout.target_index_bytes,
            target=target_tensors,
            device_id=device_id,
            tensor_names=selection_order,
            view_spec=None,
            view_id=str(layout.target_layout.view_id or "") or None,
            view_index_hint=layout.target_index_bytes,
            selection_order=selection_order,
        )
        if bytes(derived_layout.view_index_bytes or b"") != bytes(
            layout.target_index_bytes
        ):
            raise ArtifactError(
                "target_tensors do not match BindingLayout index bytes",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
    else:
        derived_layout = pipeline._build_region_backed_layout(
            canonical_index=expected_index,
            canonical_index_bytes=layout.target_index_bytes,
            target=target_tensors,
            device_id=device_id,
            tensor_names=None,
            view_spec=None,
            view_id=None,
            view_index_hint=None,
            selection_order=None,
        )
    if _normalize_target_layout_contract(
        derived_layout.layout
    ) != _normalize_target_layout_contract(layout.target_layout):
        raise ArtifactError(
            "target_tensors do not match the BindingLayout storage contract",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )


def _artifact_id_kind_from_proto(kind: int, artifact_id: str) -> ArtifactIdKind:
    if kind == common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_MI2:
        return ArtifactIdKind.MI2
    if kind == common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_CGID:
        return ArtifactIdKind.CGID
    inferred = infer_artifact_id_kind(artifact_id)
    return inferred or ArtifactIdKind.MI2


def _decode_published_model_version_from_response(
    resp: operation_pb2.GetOperationResponse, *, assembly_id: str
) -> PublishedModelVersion:
    if resp.status.state != operation_pb2.OPERATION_STATE_SUCCESS:
        message = (
            resp.status.message or "Assembly attempt did not complete successfully"
        )
        if resp.status.HasField("error"):
            message = resp.status.error.message or message
        raise ArtifactError(
            message,
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    payload = store_daemon_pb2.SealAssemblyResult()
    if not resp.status.result.Unpack(payload):
        raise ArtifactError(
            "Unexpected assembly attempt result type",
            status_code="INTERNAL",
            retryable=False,
        )
    artifact = payload.artifact
    descriptor = TypedArtifactDescriptor(
        artifact_id=str(artifact.artifact_id),
        index_multihash=str(artifact.index_multihash or "") or None,
        data_multihash=str(artifact.data_multihash or "") or None,
        schema_version=str(artifact.schema_version or "") or None,
        encoding=str(artifact.encoding or "") or None,
        total_size=int(artifact.total_size),
        id_kind=_artifact_id_kind_from_proto(artifact.id_kind, artifact.artifact_id),
    )
    return PublishedModelVersion(
        assembly_id=assembly_id,
        source_artifact_id=descriptor.artifact_id,
        source_descriptor=descriptor,
        serving_artifact_id=(
            str(payload.serving_artifact.artifact_id)
            if payload.HasField("serving_artifact")
            and payload.serving_artifact.artifact_id
            else None
        ),
        serving_descriptor=(
            TypedArtifactDescriptor(
                artifact_id=str(payload.serving_artifact.artifact_id),
                index_multihash=str(payload.serving_artifact.index_multihash or "")
                or None,
                data_multihash=str(payload.serving_artifact.data_multihash or "")
                or None,
                schema_version=str(payload.serving_artifact.schema_version or "")
                or None,
                encoding=str(payload.serving_artifact.encoding or "") or None,
                total_size=int(payload.serving_artifact.total_size),
                id_kind=_artifact_id_kind_from_proto(
                    payload.serving_artifact.id_kind,
                    payload.serving_artifact.artifact_id,
                ),
            )
            if payload.HasField("serving_artifact")
            and payload.serving_artifact.artifact_id
            else None
        ),
        source_version_key=str(payload.source_version_key or "") or None,
        serving_version_key=str(payload.serving_version_key or "") or None,
        representation_contract_hash=(
            str(payload.representation_contract_hash or "") or None
        ),
        serving_manifest_ref=str(payload.serving_manifest_ref or "") or None,
    )


def _split_mi2_artifact_id(artifact_id: str) -> tuple[str | None, str | None]:
    if not artifact_id.startswith("mi2:"):
        return None, None
    remainder = artifact_id[len("mi2:") :]
    parts = remainder.split(":", 1)
    if len(parts) != 2:
        return None, None
    index_multihash = parts[0].strip()
    data_multihash = parts[1].strip()
    if not index_multihash or not data_multihash:
        return None, None
    return index_multihash, data_multihash


def _parse_artifact_ref(
    ref: str | None,
    *,
    artifact_id: str | None,
    key: str | None,
) -> tuple[str | None, str | None]:
    if ref is None:
        return artifact_id, key
    if artifact_id or key:
        raise ValueError("ref cannot be combined with artifact_id or key")
    ref_value = str(ref)
    if not ref_value:
        raise ValueError("ref must be non-empty")
    if ref_value.startswith(("mi2:", "cgid:")):
        return ref_value, None
    if ref_value.startswith("disk:"):
        raise ValueError(
            "disk: ref is no longer supported; use Store.from_disk(...) to import "
            "and then reference the artifact by id or key."
        )
    return None, ref_value


def _should_show_from_disk_progress(show_progress: bool | None) -> bool:
    if show_progress is not None:
        return bool(show_progress)
    with contextlib.suppress(Exception):
        return bool(sys.stderr.isatty())
    return False


_IMPORT_STREAM_ERROR_STATUS: dict[int, ArtifactStatusCode] = {
    int(store_daemon_pb2.IMPORT_ARTIFACT_ERROR_CODE_SOURCE_NOT_FOUND): "NOT_FOUND",
    int(
        store_daemon_pb2.IMPORT_ARTIFACT_ERROR_CODE_SOURCE_PERMISSION_DENIED
    ): "PERMISSION_DENIED",
    int(
        store_daemon_pb2.IMPORT_ARTIFACT_ERROR_CODE_SOURCE_FORMAT_INVALID
    ): "INVALID_ARGUMENT",
    int(
        store_daemon_pb2.IMPORT_ARTIFACT_ERROR_CODE_SOURCE_MUTATED
    ): "FAILED_PRECONDITION",
    int(store_daemon_pb2.IMPORT_ARTIFACT_ERROR_CODE_REGISTRY_IO_FAILURE): "UNAVAILABLE",
    int(
        store_daemon_pb2.IMPORT_ARTIFACT_ERROR_CODE_POLICY_DENIED_NON_LOCAL_PEER
    ): "PERMISSION_DENIED",
}
_IMPORT_STREAM_DEFAULT_STATUS: ArtifactStatusCode = "INTERNAL"


def _stream_error_from_import_event(
    event: store_daemon_pb2.ImportArtifactFromPathStreamEvent,
) -> ArtifactError:
    message = (
        str(getattr(event, "message", "") or "")
        or "ImportArtifactFromPathStream reported an error"
    )
    status_code = _IMPORT_STREAM_ERROR_STATUS.get(
        int(getattr(event, "error_code", 0) or 0), _IMPORT_STREAM_DEFAULT_STATUS
    )
    return ArtifactError(
        message,
        status_code=status_code,
        retryable=(status_code == "UNAVAILABLE"),
    )


def _consume_import_artifact_stream_with_tqdm(
    stream,
    *,
    disk_path: str,
) -> store_daemon_pb2.ImportArtifactFromPathResponse:
    from tqdm.auto import tqdm

    desc = f"resolve:{os.path.basename(disk_path) or disk_path}"
    bar = tqdm(total=None, unit="B", unit_scale=True, unit_divisor=1024, desc=desc)
    final_response: store_daemon_pb2.ImportArtifactFromPathResponse | None = None
    try:
        for event in stream:
            total_bytes = int(getattr(event, "total_bytes", 0) or 0)
            processed_bytes = int(getattr(event, "processed_bytes", 0) or 0)
            if total_bytes > 0 and bar.total != total_bytes:
                bar.total = total_bytes
            if processed_bytes > bar.n:
                bar.update(processed_bytes - bar.n)

            message = str(getattr(event, "message", "") or "")
            if message:
                bar.set_postfix_str(message, refresh=False)

            if bool(getattr(event, "done", False)):
                if total_bytes > 0 and bar.n < total_bytes:
                    bar.update(total_bytes - bar.n)
                if bool(getattr(event, "error", False)):
                    raise _stream_error_from_import_event(event)
                if not event.HasField("result"):
                    raise ArtifactError(
                        "ImportArtifactFromPathStream done event missing result",
                        status_code="DATA_LOSS",
                        retryable=False,
                    )
                final_response = event.result
                break
    finally:
        bar.close()

    if final_response is None:
        raise ArtifactError(
            "ImportArtifactFromPathStream ended without terminal result",
            status_code="DATA_LOSS",
            retryable=False,
        )
    return final_response


def _consume_import_artifact_stream(
    stream,
) -> store_daemon_pb2.ImportArtifactFromPathResponse:
    final_response: store_daemon_pb2.ImportArtifactFromPathResponse | None = None
    for event in stream:
        if bool(getattr(event, "done", False)):
            if bool(getattr(event, "error", False)):
                raise _stream_error_from_import_event(event)
            if not event.HasField("result"):
                raise ArtifactError(
                    "ImportArtifactFromPathStream done event missing result",
                    status_code="DATA_LOSS",
                    retryable=False,
                )
            final_response = event.result
            break
    if final_response is None:
        raise ArtifactError(
            "ImportArtifactFromPathStream ended without terminal result",
            status_code="DATA_LOSS",
            retryable=False,
        )
    return final_response


class Store:
    """Store façade delegating to runtime, registration, and materialization pipelines."""

    _PERSISTENCE_STATE_FROM_PROTO = {
        store_daemon_pb2.PERSISTENCE_STATE_PENDING: "pending",
        store_daemon_pb2.PERSISTENCE_STATE_RUNNING: "running",
        store_daemon_pb2.PERSISTENCE_STATE_DEGRADED: "degraded",
        store_daemon_pb2.PERSISTENCE_STATE_SUCCESS: "success",
        store_daemon_pb2.PERSISTENCE_STATE_FAILED: "failed",
    }

    def __init__(
        self,
        daemon_endpoint: str,
        *,
        opts: StoreOptions | None = None,
        runtime: StoreRuntimeContext | None = None,
        register_fn: Callable[..., RegistrationResult] | None = None,
        materialize_fn: Callable[..., MaterializationPayload] | None = None,
    ) -> None:
        self._runtime = runtime or StoreRuntimeContext(
            daemon_endpoint, opts=opts, client_factory=get_daemon_client
        )
        self._views = ViewOrchestrator(self._runtime)
        self._registration = RegistrationPipeline(
            self._runtime,
            self._views,
            register_fn=register_fn or _register_artifact_core,
        )
        self._materialization = MaterializationPipeline(
            self._runtime,
            self._views,
            materialize_fn=materialize_fn or materialize_artifact_v2,
        )
        self._enable_batcher = os.getenv(
            "TENSORCAST_STORE_ENABLE_BATCHER", "1"
        ).lower() not in ("0", "false", "no")
        self._enable_prefetch = os.getenv(
            "TENSORCAST_STORE_ENABLE_PREFETCH", "1"
        ).lower() not in ("0", "false", "no")
        self._batcher: MaterializationBatcher | None = (
            MaterializationBatcher(
                self._runtime,
                self._materialization,
            )
            if self._enable_batcher
            else None
        )

    def set_register_fn(self, register_fn: Callable[..., RegistrationResult]) -> None:
        self._registration.set_register_fn(register_fn)

    def set_materialize_fn(
        self, materialize_fn: Callable[..., MaterializationPayload]
    ) -> None:
        self._materialization.set_materialize_fn(materialize_fn)

    # ------------------------------------------------------------------
    # Registration APIs
    # ------------------------------------------------------------------
    def register(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        ttl_ms: int | None = None,
    ) -> RegisteredArtifact:
        return self._registration.register(
            tensors,
            artifact_id=artifact_id,
            key=key,
            policy=policy,
            options=options,
            ttl_ms=ttl_ms,
        )

    def register_async(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        ttl_ms: int | None = None,
    ) -> ArtifactFuture[RegisteredArtifact]:
        return self._registration.register_async(
            tensors,
            artifact_id=artifact_id,
            key=key,
            policy=policy,
            options=options,
            ttl_ms=ttl_ms,
        )

    def register_view(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        slices: Mapping[str, Sequence[object]] | None = None,
        transpose: Mapping[str, Sequence[tuple[int, int]]] | None = None,
        view_id: str | None = None,
        placement: str | None = None,
        ttl_ms: int | None = None,
        options: RegisterArtifactOptions | None = None,
        canonical_index_bytes: bytes | None = None,
        registration_kind: str | int | None = None,
    ) -> RegisteredArtifact:
        return self._registration.register_view(
            tensors,
            artifact_id=artifact_id,
            key=key,
            slices=slices,
            transpose=transpose,
            view_id=view_id,
            placement=placement,
            ttl_ms=ttl_ms,
            options=options,
            canonical_index_bytes=canonical_index_bytes,
            registration_kind=registration_kind,
            resolver=self._views.resolve_view_inputs,
        )

    def register_piece(
        self,
        tensors: TensorDict,
        *,
        assembly_id: str,
        key: str | None = None,
        slices: Mapping[str, Sequence[object]] | None = None,
        canonical_index_bytes: bytes | None = None,
        placement: str | None = None,
        ttl_ms: int | None = None,
        options: RegisterArtifactOptions | None = None,
    ) -> RegisteredArtifact:
        return self._registration.register_piece(
            tensors,
            assembly_id=assembly_id,
            key=key,
            slices=slices,
            canonical_index_bytes=canonical_index_bytes,
            placement=placement,
            ttl_ms=ttl_ms,
            options=options,
            resolver=self._views.resolve_view_inputs,
        )

    def put(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        device: int | torch.device | None = None,
    ) -> RegisteredArtifact:
        return self._registration.put(
            tensors,
            artifact_id=artifact_id,
            key=key,
            policy=policy,
            options=options,
            device=device,
        )

    def put_async(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        device: int | torch.device | None = None,
    ) -> ArtifactFuture[RegisteredArtifact]:
        return self._registration.put_async(
            tensors,
            artifact_id=artifact_id,
            key=key,
            policy=policy,
            options=options,
            device=device,
        )

    def create_binding(
        self,
        layout: BindingLayout,
        *,
        ownership: str = "daemon",
        device: torch.device | str | None = None,
        target_tensors: Mapping[str, torch.Tensor] | None = None,
        mapping: CopyPlan | None = None,
        ctx: CallContext | None = None,
    ) -> Binding:
        if not isinstance(layout, BindingLayout):
            raise ArtifactError(
                "layout must be a BindingLayout",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        normalized_mapping: tuple[CopyPlanEntry, ...] | None = None
        copy_plan_proto: store_daemon_pb2.CopyPlan | None = None
        if mapping is not None:
            normalized_mapping = normalize_copy_plan(mapping)
            if not layout.dst_specs:
                raise ArtifactError(
                    "mapping requires a mapped BindingLayout with dst_specs",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            copy_plan_proto = _copy_plan_to_proto(normalized_mapping)
        elif layout.dst_specs:
            raise ArtifactError(
                "mapped BindingLayout requires mapping",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        runtime = self._runtime
        client = runtime.ensure_client()
        timeout_s = _ctx_timeout_s(ctx)
        mode = str(ownership).strip().lower()
        if mode == "daemon":
            if device is None:
                raise ArtifactError(
                    "device is required for daemon-owned bindings",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if target_tensors is not None:
                raise ArtifactError(
                    "target_tensors must be omitted for daemon-owned bindings",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            device_obj = torch.device(device)
            device_id = resolve_device(device_obj, allow_cpu=False)
            response = client.create_binding(
                ownership=store_daemon_pb2.BindingOwnership.BINDING_OWNERSHIP_DAEMON,
                target_layout=layout.target_layout,
                target_index_bytes=layout.target_index_bytes,
                device_uuid=device_uuid_for(device_id),
                binding_layout_id=layout.binding_layout_id,
                copy_plan=copy_plan_proto,
                dst_specs=layout.dst_specs if copy_plan_proto is not None else None,
                timeout_s=timeout_s if timeout_s is not None else 600.0,
            )
            try:
                tensors = restore_owned_binding_tensors(
                    response=response,
                    runtime=runtime,
                    device_id=device_id,
                )
                current_value_metadata = parse_binding_value_or_raise(
                    response.current_value
                    if hasattr(response, "current_value")
                    else None,
                    rpc_name="CreateBinding",
                    expected_binding_id=str(response.binding_id),
                    expected_binding_layout_id=layout.binding_layout_id,
                )
            except Exception:
                with contextlib.suppress(Exception):
                    client.close_owned_binding(binding_id=str(response.binding_id))
                raise
            slot = OwnedBindingSlot(
                store=self,
                runtime=runtime,
                tensors=tensors,
                layout=layout,
                binding_id=str(response.binding_id),
                current_value_metadata=current_value_metadata,
                device=device_obj,
                device_id=device_id,
                fallback=None,
                target_publication_token=None,
            )
            return Binding(slot)

        if mode != "client":
            raise ArtifactError(
                "ownership must be 'daemon' or 'client'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not target_tensors:
            raise ArtifactError(
                "target_tensors are required for client-owned bindings",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        expected_index = canonical_index_from_bytes(layout.target_index_bytes)
        expected_names = {entry.name for entry in expected_index.entries}
        if {str(name) for name in target_tensors} != expected_names:
            raise ArtifactError(
                "target_tensors must match the BindingLayout tensor set",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if device is not None:
            requested_device_id = resolve_device(torch.device(device), allow_cpu=False)
        else:
            requested_device_id = None
        first_tensor = next(iter(target_tensors.values()))
        if not first_tensor.is_cuda:
            raise ArtifactError(
                "client-owned bindings require CUDA target tensors",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        device_id = resolve_device(first_tensor.device, allow_cpu=False)
        if requested_device_id is not None and requested_device_id != device_id:
            raise ArtifactError(
                "device does not match target_tensors",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        for name, tensor in target_tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise ArtifactError(
                    f"target tensor '{name}' must be a torch.Tensor",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if not tensor.is_cuda:
                raise ArtifactError(
                    f"target tensor '{name}' must be CUDA",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if resolve_device(tensor.device, allow_cpu=False) != device_id:
                raise ArtifactError(
                    "target_tensors must share the same CUDA device",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
        for entry in expected_index.entries:
            tensor = target_tensors[entry.name]
            if tensor.dtype != entry.dtype:
                raise ArtifactError(
                    f"target tensor '{entry.name}' dtype does not match BindingLayout",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if tuple(int(v) for v in tensor.shape) != tuple(
                int(v) for v in entry.shape
            ):
                raise ArtifactError(
                    f"target tensor '{entry.name}' shape does not match BindingLayout",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if tuple(int(v) for v in tensor.stride()) != tuple(
                int(v) for v in entry.stride
            ):
                raise ArtifactError(
                    f"target tensor '{entry.name}' stride does not match BindingLayout",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )

        region_ids: list[str] = []
        try:
            for base_ptr, nbytes in sorted(
                collect_storage_bases(target_tensors).items()
            ):
                handle = self.register_vram_region(
                    device_id=device_id,
                    base_ptr=base_ptr,
                    size_bytes=nbytes,
                    ttl_ms=0,
                )
                region_ids.append(handle.region_id)
            _validate_client_binding_targets(
                layout=layout,
                target_tensors=target_tensors,
                device_id=device_id,
                pipeline=self._materialization,
                expected_index=expected_index,
            )
            response = client.create_binding(
                ownership=store_daemon_pb2.BindingOwnership.BINDING_OWNERSHIP_CLIENT,
                target_layout=layout.target_layout,
                target_index_bytes=layout.target_index_bytes,
                device_uuid=device_uuid_for(device_id),
                binding_layout_id=layout.binding_layout_id,
                copy_plan=copy_plan_proto,
                dst_specs=layout.dst_specs if copy_plan_proto is not None else None,
                timeout_s=timeout_s if timeout_s is not None else 600.0,
            )
        except Exception:
            with contextlib.suppress(Exception):
                for region_id in region_ids:
                    self.unregister_vram_region(region_id)
            raise
        try:
            current_value_metadata = parse_binding_value_or_raise(
                response.current_value if hasattr(response, "current_value") else None,
                rpc_name="CreateBinding",
                expected_binding_id=str(response.binding_id),
                expected_binding_layout_id=layout.binding_layout_id,
            )
        except Exception:
            with contextlib.suppress(Exception):
                client.close_owned_binding(binding_id=str(response.binding_id))
            with contextlib.suppress(Exception):
                for region_id in region_ids:
                    self.unregister_vram_region(region_id)
            raise
        slot = InplaceSlot(
            store=self,
            runtime=runtime,
            pipeline=self._materialization,
            tensors=target_tensors,
            device=first_tensor.device,
            device_id=device_id,
            layout=layout,
            binding_id=str(response.binding_id),
            region_ids=tuple(region_ids),
            selection_names=tuple(
                offset.name for offset in layout.target_layout.offsets
            ),
            view_id=str(layout.target_layout.view_id or "") or None,
            view_subset_hash=None,
            view_spec=None,
            fallback=None,
            current_value_metadata=current_value_metadata,
            target_publication_token=None,
            copy_plan=normalized_mapping,
        )
        return Binding(slot)

    def query_persistence_status(
        self, *, task_id: str | None = None, artifact_id: str | None = None
    ) -> PersistenceStatusResult:
        """Query persistence task state via the local daemon."""
        if not task_id and not artifact_id:
            raise ArtifactError(
                "task_id or artifact_id must be provided",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        resp = self._runtime.ensure_client().query_persistence_status(
            task_id=task_id, artifact_id=artifact_id
        )
        return self._persistence_status_from_proto(resp)

    def seal_assembly(
        self,
        assembly_id: str,
        *,
        publish_canonical: bool = True,
        wait: bool = True,
        layout_id: str | None = None,
        timeout_s: float = 120.0,
        ctx: CallContext | None = None,
    ) -> SealAssemblyResult | Operation[SealAssemblyResult]:
        if publish_canonical is False:
            # Legacy synchronous path (operation-based sealing always publishes canonical).
            return self._runtime.ensure_client().seal_assembly(
                assembly_id,
                publish_canonical=False,
                timeout_s=timeout_s,
            )

        op = self.seal_assembly_operation(
            assembly_id,
            layout_id=layout_id,
            ctx=ctx,
        )
        if wait:
            return op.wait(timeout_s=timeout_s)
        return op

    def seal_assembly_operation(
        self,
        assembly_id: str,
        *,
        layout_id: str | None = None,
        ctx: CallContext | None = None,
    ) -> Operation[SealAssemblyResult]:
        start_resp = self._runtime.ensure_client().start_seal_assembly(
            assembly_id=assembly_id,
            layout_id=layout_id,
        )
        operation_id = start_resp.operation.operation_id
        context: dict[str, str] = {"assembly_id": assembly_id}
        if layout_id:
            context["layout_id"] = str(layout_id)

        def _decode(resp) -> SealAssemblyResult:
            payload = store_daemon_pb2.SealAssemblyResult()
            if not resp.status.result.Unpack(payload):
                raise ArtifactError(
                    f"Unexpected seal assembly result type (assembly_id={assembly_id})",
                    status_code="INTERNAL",
                    retryable=False,
                )
            artifact = payload.artifact
            descriptor = TypedArtifactDescriptor(
                artifact_id=str(artifact.artifact_id),
                index_multihash=str(artifact.index_multihash or "") or None,
                data_multihash=str(artifact.data_multihash or "") or None,
                schema_version=str(artifact.schema_version or "") or None,
                encoding=str(artifact.encoding or "") or None,
                total_size=int(artifact.total_size),
                id_kind=_artifact_id_kind_from_proto(
                    artifact.id_kind, artifact.artifact_id
                ),
            )
            return SealAssemblyResult(
                sealed_artifact_id=descriptor.artifact_id,
                descriptor=descriptor,
                already_sealed=False,
            )

        return DaemonGlobalStoreOperation(
            operation_id=operation_id,
            runtime_ref=weakref.ref(self._runtime),
            ctx=ctx,
            context=context,
            result_factory=_decode,
        )

    def _persistence_status_from_proto(
        self, resp: store_daemon_pb2.QueryPersistenceStatusResponse
    ) -> PersistenceStatusResult:
        shards: list[PersistenceShardStatus] = []
        for shard in resp.shards:
            state = self._PERSISTENCE_STATE_FROM_PROTO.get(shard.state, "unknown")
            shards.append(
                PersistenceShardStatus(
                    shard_id=shard.shard_id,
                    shard_idx=int(shard.shard_idx),
                    state=state,
                    progress=float(shard.progress),
                    degraded_reason=shard.degraded_reason or None,
                    last_error=shard.last_error or None,
                    target_nodes=tuple(shard.target_nodes),
                    lease_ids=tuple(shard.lease_ids),
                )
            )
        state = self._PERSISTENCE_STATE_FROM_PROTO.get(resp.state, "unknown")
        return PersistenceStatusResult(
            task_id=resp.task_id,
            artifact_id=resp.artifact_id,
            plan_id=resp.plan_id,
            state=state,
            progress=float(resp.progress),
            degraded_reason=resp.degraded_reason or None,
            last_error=resp.last_error or None,
            shards=tuple(shards),
        )

    def start_assembly_attempt(
        self,
        *,
        layout_id: str,
        requirements: AssemblyRequirementSetRef | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        closeout_contract: AssemblyCloseoutContract | None = None,
        ctx: CallContext | None = None,
    ) -> AssemblyAttemptRef:
        del ctx
        if requirements is None:
            raise ValueError(
                "requirements are required; construct them explicitly with "
                "AssemblyRequirementSetRef.pp_from_structural_views(...), "
                "AssemblyRequirementSetRef.ep_from_structural_views(...), "
                "or AssemblyRequirementSetRef.canonical_full()"
            )
        kwargs: dict[str, object] = {"layout_id": layout_id}
        kwargs["requirements"] = requirements
        if readiness_policy is not None:
            kwargs["readiness_policy"] = readiness_policy
        if closeout_contract is not None:
            kwargs["closeout_contract"] = closeout_contract
        return self._runtime.ensure_client().start_assembly_attempt(**kwargs)

    def seal_assembly_attempt(
        self,
        attempt: AssemblyAttemptRef,
        *,
        ctx: CallContext | None = None,
    ) -> Operation[PublishedModelVersion]:
        if not attempt.coordinator_operation_id:
            raise ArtifactError(
                "AssemblyAttemptRef is missing coordinator operation metadata",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        resp = self._runtime.ensure_client().seal_assembly_attempt(
            attempt_id=attempt.attempt_id,
            timeout_s=10.0,
        )
        operation_ref = operation_pb2.OperationRef()
        operation_ref.CopyFrom(resp.operation)
        context = {
            "attempt_id": attempt.attempt_id,
            "workspace_assembly_id": attempt.workspace_assembly_id,
            "layout_id": attempt.layout_id,
        }

        def _decode(
            op_resp: operation_pb2.GetOperationResponse,
        ) -> PublishedModelVersion:
            return _decode_published_model_version_from_response(
                op_resp, assembly_id=attempt.workspace_assembly_id
            )

        return DaemonGlobalStoreOperation(
            operation_id=str(operation_ref.operation_id),
            runtime_ref=weakref.ref(self._runtime),
            ctx=ctx,
            context=context,
            result_factory=_decode,
            operation_ref=operation_ref,
        )

    def wait_assembly_attempt(
        self,
        attempt: AssemblyAttemptRef | Operation[PublishedModelVersion],
        *,
        timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> PublishedModelVersion:
        del ctx
        if isinstance(attempt, Operation):
            return attempt.wait(timeout_s=timeout_s)

        client = self._runtime.ensure_client()
        operation_id: str
        assembly_id: str
        operation_ref: operation_pb2.OperationRef | None = None
        if isinstance(attempt, AssemblyAttemptRef):
            if not attempt.coordinator_operation_id:
                raise ArtifactError(
                    "AssemblyAttemptRef is missing coordinator operation metadata",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            operation_id = attempt.coordinator_operation_id
            assembly_id = attempt.workspace_assembly_id
            operation_ref = operation_pb2.OperationRef()
            operation_ref.CopyFrom(attempt.coordinator_operation)
        wait_timeout_s = 120.0 if timeout_s is None else float(timeout_s)
        resp = client.wait_operation(
            operation_id,
            operation_ref=operation_ref,
            timeout_ms=max(1, int(wait_timeout_s * 1000)),
            timeout_s=wait_timeout_s + 5.0,
        )
        return _decode_published_model_version_from_response(
            resp, assembly_id=assembly_id
        )

    def persistence_operation(
        self,
        *,
        task_id: str | None = None,
        artifact_id: str | None = None,
        ctx: CallContext | None = None,
    ) -> Operation[PersistenceStatusResult]:
        op_id = task_id or f"persist:{artifact_id or ''}"
        created_at = time.monotonic()

        def _ctx_remaining_timeout_s() -> float | None:
            if ctx is None or ctx.deadline_ms is None:
                return None
            remaining = (float(ctx.deadline_ms) / 1000.0) - (
                time.monotonic() - created_at
            )
            return max(0.0, remaining)

        def _query() -> PersistenceStatusResult:
            timeout_s = _ctx_remaining_timeout_s()
            if timeout_s is not None and timeout_s <= 0:
                raise OperationTimeoutError(
                    "Operation deadline exceeded (ctx.deadline_ms)",
                    retryable=True,
                )
            resp = self._runtime.ensure_client().query_persistence_status(
                task_id=task_id,
                artifact_id=artifact_id,
                timeout_s=timeout_s if timeout_s is not None else 10.0,
            )
            return self._persistence_status_from_proto(resp)

        def _status() -> OperationStatus:
            timeout_s = _ctx_remaining_timeout_s()
            if timeout_s is not None and timeout_s <= 0:
                return OperationStatus(
                    state="degraded",
                    message="CallContext deadline exceeded",
                    progress=0.0,
                    as_of_ms=int(time.time() * 1000),
                    error=OperationError(
                        status_code="DEADLINE_EXCEEDED",
                        message="CallContext deadline exceeded",
                        retryable=True,
                        context={
                            "task_id": task_id or "",
                            "artifact_id": artifact_id or "",
                        },
                    ),
                )
            try:
                result = _query()
            except OperationTimeoutError as exc:
                return OperationStatus(
                    state="degraded",
                    message=str(exc),
                    progress=0.0,
                    as_of_ms=int(time.time() * 1000),
                    error=OperationError(
                        status_code="DEADLINE_EXCEEDED",
                        message=str(exc),
                        retryable=True,
                        context={
                            "task_id": task_id or "",
                            "artifact_id": artifact_id or "",
                        },
                    ),
                )
            state: str
            if result.state in {"pending", "running", "unknown"}:
                state = "running"
            elif result.state == "success":
                state = "success"
            elif result.state == "failed":
                state = "failed"
            elif result.state == "degraded":
                state = "degraded"
            else:
                state = "running"
            error: OperationError | None = None
            if state in {"failed", "degraded"}:
                message = (
                    result.last_error
                    or result.degraded_reason
                    or "persistence degraded"
                )
                error = OperationError(
                    status_code="INTERNAL"
                    if state == "failed"
                    else "DEADLINE_EXCEEDED",
                    message=message,
                    retryable=True,
                    context={
                        "task_id": result.task_id,
                        "artifact_id": result.artifact_id,
                    },
                )
            return OperationStatus(
                state=state,  # type: ignore[arg-type]
                message=result.degraded_reason or None,
                progress=float(result.progress),
                as_of_ms=int(time.time() * 1000),
                error=error,
            )

        def _result() -> PersistenceStatusResult:
            return _query()

        return PollingOperation(
            operation_id=op_id,
            status_fn=_status,
            result_fn=_result,
            ctx=ctx,
        )

    # ------------------------------------------------------------------
    # Retrieval APIs
    # ------------------------------------------------------------------
    def artifact(
        self,
        ref: str | None = None,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        fallback: FallbackOptions | str | None = None,
    ) -> Artifact:
        artifact_id, key = _parse_artifact_ref(
            ref,
            artifact_id=artifact_id,
            key=key,
        )
        effective_fallback = FallbackOptions.parse(fallback)
        return Artifact(
            store_ref=weakref.ref(self),
            artifact_id=artifact_id,
            key=key,
            fallback=effective_fallback,
        )

    async def artifact_async(
        self,
        ref: str | None = None,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        fallback: FallbackOptions | str | None = None,
    ) -> Artifact:
        return self.artifact(
            ref,
            artifact_id=artifact_id,
            key=key,
            fallback=fallback,
        )

    def from_disk(
        self,
        path: str,
        *,
        key: str | None = None,
        verify_checksums: bool = True,
        show_progress: bool | None = None,
    ) -> Artifact:
        disk_path = os.fspath(path)
        if not disk_path:
            raise ValueError("path is required")
        client = self._runtime.ensure_client()
        use_progress = _should_show_from_disk_progress(show_progress)
        stream = client.import_artifact_from_path_stream_v2(
            path=disk_path,
            verify_checksums=bool(verify_checksums),
        )
        if use_progress:
            response = _consume_import_artifact_stream_with_tqdm(
                stream,
                disk_path=disk_path,
            )
        else:
            response = _consume_import_artifact_stream(stream)
        artifact_id = response.artifact_id or ""
        if not artifact_id:
            raise ArtifactError(
                "ImportArtifactFromPath returned empty artifact_id",
                status_code="DATA_LOSS",
                retryable=False,
            )
        canonical_index_bytes = bytes(response.canonical_index_bytes)
        if not canonical_index_bytes:
            raise ArtifactError(
                "ImportArtifactFromPath returned empty canonical index bytes",
                status_code="DATA_LOSS",
                retryable=False,
            )
        canonical_index = canonical_index_from_bytes(canonical_index_bytes)
        generation_value: int | None = (
            int(response.generation) if response.generation else None
        )
        entry = ArtifactCacheEntry(
            artifact_id=artifact_id,
            canonical_index_bytes=canonical_index_bytes,
            parsed_index=canonical_index,
            generation=generation_value,
            expires_at=time.monotonic(),
        )
        self._runtime.cache_artifact_index(entry)
        if key:
            index_multihash, data_multihash = _split_mi2_artifact_id(artifact_id)
            id_kind = infer_artifact_id_kind(artifact_id) or ArtifactIdKind.MI2
            descriptor = TypedArtifactDescriptor(
                artifact_id=artifact_id,
                index_multihash=index_multihash,
                data_multihash=data_multihash,
                schema_version=None,
                encoding=None,
                total_size=canonical_index.total_size_bytes,
                id_kind=id_kind,
            )
            try:
                ok = client.publish_replica_key(key=key, descriptor=descriptor)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to publish key %s via daemon", key)
            else:
                if not ok:
                    logger.warning(
                        "Key mapping for %s already exists; keeping existing mapping",
                        key,
                    )
                else:
                    self._runtime.cache_key_mapping(key, artifact_id=artifact_id)
        return Artifact(
            store_ref=weakref.ref(self),
            artifact_id=artifact_id,
            key=key,
            fallback=None,
            canonical_index_bytes=canonical_index_bytes or None,
            canonical_index=canonical_index,
            generation=generation_value,
        )

    # ------------------------------------------------------------------
    # Region-backed registration
    # ------------------------------------------------------------------
    def register_vram_region(
        self,
        *,
        device_id: int,
        base_ptr: int,
        size_bytes: int,
        ttl_ms: int,
        name: str | None = None,
    ) -> VramRegionHandle:
        client = self._runtime.ensure_client()
        base_ptr_value = int(base_ptr)
        size_value = int(size_bytes)
        base_offset = 0
        try:
            handle_bytes, base_offset = get_cuda_memory_handle_with_offset(
                int(device_id), base_ptr_value
            )
        except Exception:  # noqa: BLE001
            handle_bytes = get_cuda_memory_handle(int(device_id), base_ptr_value)
            base_offset = 0
        if base_offset:
            base_ptr_value -= int(base_offset)
            size_value += int(base_offset)
        handle = client.register_vram_region(
            device_id=int(device_id),
            size_bytes=int(size_value),
            ttl_ms=int(ttl_ms),
            cuda_ipc_handle=handle_bytes,
            region_name=name,
        )
        with contextlib.suppress(Exception):
            _cache_register_region(
                region_id=handle.region_id,
                device_id=int(device_id),
                base_ptr=int(base_ptr_value),
                size_bytes=int(size_value),
                ttl_ms=int(ttl_ms),
            )
        return handle

    def unregister_vram_region(
        self, region_id: str, *, force: bool | None = None
    ) -> bool:
        client = self._runtime.ensure_client()
        released = client.unregister_vram_region(region_id, force=force)
        if released:
            with contextlib.suppress(Exception):
                _cache_unregister_region(region_id)
        return released

    def deregister_artifact(
        self,
        artifact_id: str,
        *,
        wait: bool = True,
        drain_timeout_s: float | None = None,
        extend_ttl_ms: int | None = None,
        device_id: int | None = None,
        byte_space: common_pb2.ByteSpaceRef | None = None,
        keep_shared_disk_copy: bool = False,
        operation_id: str | None = None,
    ) -> DeregisterArtifactOutcome:
        client = self._runtime.ensure_client()
        drain_ms = int(drain_timeout_s * 1000) if drain_timeout_s is not None else None
        outcome = client.deregister_artifact(
            artifact_id,
            wait_for_drain=bool(wait),
            drain_timeout_ms=drain_ms,
            extend_ttl_ms=extend_ttl_ms,
            device_id=device_id,
            byte_space=byte_space,
            keep_shared_disk_copy=keep_shared_disk_copy,
            operation_id=operation_id,
        )
        self._runtime.invalidate_artifact(artifact_id, key=None, reason="deregister")
        return outcome

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def capabilities(self) -> StoreCapabilities:
        return self._runtime.capabilities

    @property
    def daemon_endpoint(self) -> str:
        return self._runtime.daemon_endpoint

    @property
    def closed(self) -> bool:
        return self._runtime.closed

    @property
    def batcher(self) -> MaterializationBatcher:
        if self._batcher is None:
            raise RuntimeError("Materialization batcher is disabled")
        return self._batcher

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._batcher is not None:
                self._batcher.close()
        self._runtime.close()


_PROCESS_STORE_LOCK = threading.RLock()
_PROCESS_STORE: Store | None = None
_PROCESS_STORE_OPTS: StoreOptions | None = None


def _ensure_process_store(
    *,
    opts: StoreOptions | None = None,
    force_recreate: bool = False,
) -> Store:
    global _PROCESS_STORE, _PROCESS_STORE_OPTS

    context = get_runtime_context(
        opts=opts,
        force_recreate=force_recreate,
        client_factory=get_daemon_client,
        runtime_provider=require_runtime,
    )

    with _PROCESS_STORE_LOCK:
        current = _PROCESS_STORE
        current_closed = current.closed if current is not None else False
        if current is None or force_recreate or current_closed:
            prior = _PROCESS_STORE
            opts_marker: StoreOptions | None
            if opts is not None:
                opts_marker = opts
            elif current_closed and not force_recreate:
                opts_marker = _PROCESS_STORE_OPTS
            else:
                opts_marker = None
            effective_opts: StoreOptions = opts_marker or context.opts
            _PROCESS_STORE = Store(
                context.daemon_endpoint, opts=effective_opts, runtime=context
            )
            _PROCESS_STORE_OPTS = opts_marker
            if prior is not None:
                with contextlib.suppress(Exception):
                    prior.close()
        elif (
            opts is not None
            and _PROCESS_STORE_OPTS is not None
            and opts != _PROCESS_STORE_OPTS
        ):
            raise RuntimeError(
                "Store already initialized with different options. Pass force_recreate=True to replace."
            )
        result = _PROCESS_STORE
        if result is None:
            raise RuntimeError("Failed to initialize process Store")
        return result


def store(
    *,
    opts: StoreOptions | None = None,
    force_recreate: bool = False,
) -> Store:
    """Return the process-wide Store session, creating it lazily as needed."""

    return _ensure_process_store(opts=opts, force_recreate=force_recreate)


def shutdown_process_store() -> None:
    """Close and clear the process-wide Store if it exists."""

    global _PROCESS_STORE, _PROCESS_STORE_OPTS

    with _PROCESS_STORE_LOCK:
        current = _PROCESS_STORE
        _PROCESS_STORE = None
        _PROCESS_STORE_OPTS = None
    if current is not None:
        with contextlib.suppress(Exception):
            current.close()
    shutdown_context()


def _coerce_store() -> Store:
    current = _PROCESS_STORE
    if current is not None and not current.closed:
        return current
    return store()


def register(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    policy: StorePolicy | str | None = None,
    options: RegisterArtifactOptions | None = None,
    ttl_ms: int | None = None,
) -> RegisteredArtifact:
    return _coerce_store().register(
        tensors,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        options=options,
        ttl_ms=ttl_ms,
    )


def register_async(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    policy: StorePolicy | str | None = None,
    options: RegisterArtifactOptions | None = None,
    ttl_ms: int | None = None,
) -> ArtifactFuture[RegisteredArtifact]:
    return _coerce_store().register_async(
        tensors,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        options=options,
        ttl_ms=ttl_ms,
    )


def register_view(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    slices: Mapping[str, Sequence[object]] | None = None,
    transpose: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    view_id: str | None = None,
    placement: str | None = None,
    ttl_ms: int | None = None,
    options: RegisterArtifactOptions | None = None,
    canonical_index_bytes: bytes | None = None,
    registration_kind: str | int | None = None,
) -> RegisteredArtifact:
    return _coerce_store().register_view(
        tensors,
        artifact_id=artifact_id,
        key=key,
        slices=slices,
        transpose=transpose,
        view_id=view_id,
        placement=placement,
        ttl_ms=ttl_ms,
        options=options,
        canonical_index_bytes=canonical_index_bytes,
        registration_kind=registration_kind,
    )


def register_piece(
    tensors: TensorDict,
    *,
    assembly_id: str,
    key: str | None = None,
    slices: Mapping[str, Sequence[object]] | None = None,
    canonical_index_bytes: bytes | None = None,
    placement: str | None = None,
    ttl_ms: int | None = None,
    options: RegisterArtifactOptions | None = None,
) -> RegisteredArtifact:
    return _coerce_store().register_piece(
        tensors,
        assembly_id=assembly_id,
        key=key,
        slices=slices,
        canonical_index_bytes=canonical_index_bytes,
        placement=placement,
        ttl_ms=ttl_ms,
        options=options,
    )


def register_vram_region(
    *,
    device_id: int,
    base_ptr: int,
    size_bytes: int,
    ttl_ms: int,
    name: str | None = None,
) -> VramRegionHandle:
    return _coerce_store().register_vram_region(
        device_id=device_id,
        base_ptr=base_ptr,
        size_bytes=size_bytes,
        ttl_ms=ttl_ms,
        name=name,
    )


def unregister_vram_region(region_id: str, *, force: bool | None = None) -> bool:
    return _coerce_store().unregister_vram_region(region_id, force=force)


def deregister_artifact(
    artifact_id: str,
    *,
    wait: bool = True,
    drain_timeout_s: float | None = None,
    extend_ttl_ms: int | None = None,
    device_id: int | None = None,
    byte_space: common_pb2.ByteSpaceRef | None = None,
    keep_shared_disk_copy: bool = False,
    operation_id: str | None = None,
) -> DeregisterArtifactOutcome:
    return _coerce_store().deregister_artifact(
        artifact_id,
        wait=wait,
        drain_timeout_s=drain_timeout_s,
        extend_ttl_ms=extend_ttl_ms,
        device_id=device_id,
        byte_space=byte_space,
        keep_shared_disk_copy=keep_shared_disk_copy,
        operation_id=operation_id,
    )


def put(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    policy: StorePolicy | str | None = None,
    options: RegisterArtifactOptions | None = None,
    device: int | torch.device | None = None,
) -> RegisteredArtifact:
    return _coerce_store().put(
        tensors,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        options=options,
        device=device,
    )


def put_async(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    policy: StorePolicy | str | None = None,
    options: RegisterArtifactOptions | None = None,
    device: int | torch.device | None = None,
) -> ArtifactFuture[RegisteredArtifact]:
    return _coerce_store().put_async(
        tensors,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        options=options,
        device=device,
    )


def create_binding(
    layout: BindingLayout,
    *,
    ownership: str = "daemon",
    device: torch.device | str | None = None,
    target_tensors: Mapping[str, torch.Tensor] | None = None,
    mapping: CopyPlan | None = None,
    ctx: CallContext | None = None,
) -> Binding:
    return _coerce_store().create_binding(
        layout,
        ownership=ownership,
        device=device,
        target_tensors=target_tensors,
        mapping=mapping,
        ctx=ctx,
    )


def query_persistence_status(
    *, task_id: str | None = None, artifact_id: str | None = None
) -> PersistenceStatusResult:
    return _coerce_store().query_persistence_status(
        task_id=task_id, artifact_id=artifact_id
    )


def persistence_operation(
    *,
    task_id: str | None = None,
    artifact_id: str | None = None,
    ctx: CallContext | None = None,
) -> Operation[PersistenceStatusResult]:
    return _coerce_store().persistence_operation(
        task_id=task_id, artifact_id=artifact_id, ctx=ctx
    )


def start_assembly_attempt(
    *,
    layout_id: str,
    requirements: AssemblyRequirementSetRef | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    closeout_contract: AssemblyCloseoutContract | None = None,
    ctx: CallContext | None = None,
) -> AssemblyAttemptRef:
    return _coerce_store().start_assembly_attempt(
        layout_id=layout_id,
        requirements=requirements,
        readiness_policy=readiness_policy,
        closeout_contract=closeout_contract,
        ctx=ctx,
    )


def seal_assembly_attempt(
    attempt: AssemblyAttemptRef,
    *,
    ctx: CallContext | None = None,
) -> Operation[PublishedModelVersion]:
    return _coerce_store().seal_assembly_attempt(attempt, ctx=ctx)


def wait_assembly_attempt(
    attempt: AssemblyAttemptRef | Operation[PublishedModelVersion],
    *,
    timeout_s: float | None = None,
    ctx: CallContext | None = None,
) -> PublishedModelVersion:
    return _coerce_store().wait_assembly_attempt(
        attempt,
        timeout_s=timeout_s,
        ctx=ctx,
    )


def seal_assembly(
    assembly_id: str,
    *,
    publish_canonical: bool = True,
    wait: bool = True,
    layout_id: str | None = None,
    timeout_s: float = 120.0,
    ctx: CallContext | None = None,
) -> SealAssemblyResult | Operation[SealAssemblyResult]:
    return _coerce_store().seal_assembly(
        assembly_id,
        publish_canonical=publish_canonical,
        wait=wait,
        layout_id=layout_id,
        timeout_s=timeout_s,
        ctx=ctx,
    )


def artifact(
    ref: str | None = None,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    fallback: FallbackOptions | str | None = None,
) -> Artifact:
    store = _coerce_store()
    if ref is None:
        return store.artifact(
            artifact_id=artifact_id,
            key=key,
            fallback=fallback,
        )
    return store.artifact(
        ref=ref,
        artifact_id=artifact_id,
        key=key,
        fallback=fallback,
    )


async def artifact_async(
    ref: str | None = None,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    fallback: FallbackOptions | str | None = None,
) -> Artifact:
    store = _coerce_store()
    if ref is None:
        return await store.artifact_async(
            artifact_id=artifact_id,
            key=key,
            fallback=fallback,
        )
    return await store.artifact_async(
        ref=ref,
        artifact_id=artifact_id,
        key=key,
        fallback=fallback,
    )


def from_disk(
    path: str,
    *,
    key: str | None = None,
    verify_checksums: bool = True,
    show_progress: bool | None = None,
) -> Artifact:
    return _coerce_store().from_disk(
        path,
        key=key,
        verify_checksums=verify_checksums,
        show_progress=show_progress,
    )


__all__ = [
    "Artifact",
    "ArtifactDescriptor",
    "ArtifactError",
    "ArtifactFuture",
    "ArtifactStatusCode",
    "AssemblyAttemptRef",
    "AssemblyCloseoutContract",
    "AssemblyReadinessPolicy",
    "AssemblyRequirementSetRef",
    "Binding",
    "BindingLayout",
    "BindingUpdateEpoch",
    "CanonicalIndex",
    "CanonicalIndexEntry",
    "CopyPlan",
    "CopyPlanEntry",
    "FallbackOptions",
    "LeaseHandle",
    "MaterializationPayload",
    "MaterializationDiagnostics",
    "MaterializationBatcher",
    "PlacementPin",
    "PrefetchedReplica",
    "PartialSealResult",
    "PublishedModelVersion",
    "RegisteredArtifact",
    "ReplicaInfo",
    "RetryPolicy",
    "SealedBindingValue",
    "StoreCapabilities",
    "Store",
    "StoreOptions",
    "TensorMeta",
    "TensorDictMaterializationResult",
    "TensorDict",
    "TransformPlacement",
    "Range",
    "TargetTensors",
    "PersistenceStatusResult",
    "PersistenceShardStatus",
    "artifact",
    "artifact_async",
    "create_binding",
    "from_disk",
    "store",
    "shutdown_process_store",
    "BatchContext",
    "register",
    "register_async",
    "put",
    "put_async",
    "query_persistence_status",
    "persistence_operation",
    "start_assembly_attempt",
    "register_view",
    "register_piece",
    "register_vram_region",
    "unregister_vram_region",
    "deregister_artifact",
    "seal_assembly",
    "seal_assembly_attempt",
    "wait_assembly_attempt",
    "get_daemon_client",
    "require_runtime",
    "_register_artifact_core",
    "materialize_artifact_v2",
]
