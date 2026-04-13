#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import contextlib
import uuid
import weakref
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Sequence

import torch

from tensorcast.api import _metrics as store_metrics
from tensorcast.api import _region_cache as region_cache
from tensorcast.api._config import GetArtifactOptions
from tensorcast.api._device import device_uuid_for
from tensorcast.api.context import CallContext
from tensorcast.api.operation import (
    DaemonGlobalStoreOperation,
    Operation,
    OperationStatus,
    PollingOperation,
)
from tensorcast.api.store.binding_state import (
    BindingValueMetadata,
    clone_selection,
    parse_binding_value_or_raise,
)
from tensorcast.api.store.mapped_binding import (
    CopyPlanEntry,
    compute_mapped_view_id,
    validate_copy_plan,
    view_narrow_ranges,
)
from tensorcast.api.store.materialization import _resolve_source_policy_from_options
from tensorcast.api.store.owned_binding_layout import BindingLayout
from tensorcast.api.store.region_utils import collect_storage_bases
from tensorcast.api.store.retry import map_materialization_error
from tensorcast.api.store.types import ArtifactError
from tensorcast.common.selection_contract import (
    build_artifact_selection,
    compute_selected_index_bytes,
)
from tensorcast.common.selection_identity import (
    compute_logical_layout_hash,
    compute_selection_hash,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import ServingRuntimePolicy

if TYPE_CHECKING:
    from tensorcast.api.store import Store
    from tensorcast.api.store.artifact import Artifact
    from tensorcast.api.store.binding import BindingUpdateEpoch
    from tensorcast.api.store.materialization import (
        MaterializationPipeline,
        _RegionBackedLayout,
    )
    from tensorcast.api.store.runtime import StoreRuntimeContext


def _ctx_timeout_s(ctx: CallContext | None) -> float | None:
    if ctx is None or ctx.deadline_ms is None:
        return None
    timeout_s = float(ctx.deadline_ms) / 1000.0
    if timeout_s <= 0:
        raise ArtifactError(
            "CallContext deadline exceeded",
            status_code="DEADLINE_EXCEEDED",
            retryable=False,
        )
    return max(0.001, timeout_s)


def _map_slot_error(exc: Exception) -> ArtifactError:
    if isinstance(exc, ArtifactError):
        return exc
    if isinstance(exc, ValueError):
        return ArtifactError(str(exc), status_code="INVALID_ARGUMENT", retryable=False)
    if isinstance(exc, TimeoutError):
        return ArtifactError(str(exc), status_code="DEADLINE_EXCEEDED", retryable=True)
    return map_materialization_error(exc)


def _normalize_view_id(view_id: str | None) -> str:
    return "" if view_id is None else str(view_id)


def _selection_publishable(
    *,
    selection_names: Sequence[str],
    view_subset_hash: bytes | None,
    view_id: str,
) -> bool:
    subset_hash = view_subset_hash or b""
    # Full canonical selections are always publishable. For packed/subset
    # selections, require a stable view_id so publication is scoped to that
    # byte-space instead of canonical routing.
    if not selection_names and subset_hash == b"":
        return True
    return bool(view_id)


def _is_region_error(error: ArtifactError) -> bool:
    return error.status_code in {"DATA_LOSS", "FAILED_PRECONDITION", "NOT_FOUND"}


def _selection_from_region_layout(
    *,
    artifact_id: str,
    canonical_index_bytes: bytes,
    region_layout: "_RegionBackedLayout",
    view_spec: common_pb2.ViewSpec | None,
) -> common_pb2.ArtifactSelection:
    subset_hash = bytes(region_layout.view_subset_hash or b"")
    selection_names: tuple[str, ...] = (
        tuple(region_layout.selection_names) if subset_hash else ()
    )
    view_spec_for_selection = (
        view_spec if view_spec is not None and view_spec.tensors else None
    )
    layout_index_bytes: bytes | None = None
    if region_layout.layout.index_kind == store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW:
        if region_layout.view_index_bytes:
            layout_index_bytes = bytes(region_layout.view_index_bytes)
        else:
            layout_index_bytes = compute_selected_index_bytes(
                canonical_index_bytes=canonical_index_bytes,
                view_spec=view_spec_for_selection,
                tensor_names=selection_names,
            )
    return build_artifact_selection(
        artifact_id=artifact_id,
        canonical_index_bytes=canonical_index_bytes,
        layout_index_bytes=layout_index_bytes,
        view_spec=view_spec_for_selection,
        tensor_names=selection_names,
        view_subset_hash=subset_hash if subset_hash else None,
        view_id=_normalize_view_id(region_layout.view_id),
        allow_view_id_without_spec=bool(
            region_layout.view_id and view_spec_for_selection is None
        ),
    )


class InplaceSlot:
    """Stable, client-owned CUDA layout that can be refilled in-place."""

    def __init__(
        self,
        *,
        store: "Store",
        runtime: "StoreRuntimeContext",
        pipeline: "MaterializationPipeline",
        tensors: Mapping[str, torch.Tensor],
        device: torch.device,
        device_id: int,
        layout: BindingLayout,
        binding_id: str,
        region_ids: Sequence[str] | None,
        selection_names: Sequence[str] | None,
        view_id: str | None,
        view_subset_hash: bytes | None,
        view_spec: common_pb2.ViewSpec | None,
        current_value_metadata: BindingValueMetadata | None,
        target_publication_token: bytes | None,
        copy_plan: Sequence[CopyPlanEntry] | None = None,
    ) -> None:
        if not tensors:
            raise ArtifactError(
                "InplaceSlot requires at least one tensor",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._store = store
        self._runtime = runtime
        self._pipeline = pipeline
        self._tensors = dict(tensors)
        self._tensors_view: Mapping[str, torch.Tensor] = MappingProxyType(self._tensors)
        self._layout = layout
        self._binding_id = str(binding_id)
        self._binding_layout_id = str(layout.binding_layout_id)
        self._device = device
        self._device_id = int(device_id)
        self._region_ids = tuple(region_ids or ())
        self._view_spec = view_spec
        self._current_value_metadata = current_value_metadata
        self._artifact_id = (
            str(current_value_metadata.source_artifact_id)
            if current_value_metadata is not None
            and current_value_metadata.source_artifact_id is not None
            else None
        )
        self._contribution_source_artifact_id = self._artifact_id
        self._target_publication_token = (
            bytes(target_publication_token) if target_publication_token else None
        )
        self._copy_plan = tuple(copy_plan) if copy_plan is not None else None

        self._selection_names = tuple(str(name) for name in (selection_names or ()))
        self._view_id = _normalize_view_id(view_id)
        self._view_subset_hash = bytes(view_subset_hash or b"")
        self._logical_layout_hash = bytes(layout.target_layout.logical_layout_hash)
        if not self._logical_layout_hash:
            index_bytes = bytes(layout.target_index_bytes)
            needs_view_index = (
                layout.target_layout.index_kind
                == store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW
            )
            self._logical_layout_hash = compute_logical_layout_hash(
                index_bytes=index_bytes,
                needs_view_index=needs_view_index,
            )
        self._selection_hash = compute_selection_hash(
            view_id=self._view_id,
            view_subset_hash=self._view_subset_hash,
        )
        self._seal_generation_counter = (
            int(current_value_metadata.seal_generation)
            if current_value_metadata is not None
            else 0
        )
        self._active_update_epoch: str | None = None
        self._published_lease_id: str | None = None
        self._published_replica_id: str | None = None
        self._dirty = False
        self._closed = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def tensors(self) -> Mapping[str, torch.Tensor]:
        return self._tensors_view

    @property
    def binding_id(self) -> str:
        return self._binding_id

    @property
    def binding_layout_id(self) -> str:
        return self._binding_layout_id

    @property
    def layout(self) -> BindingLayout:
        return self._layout

    @property
    def current_value_metadata(self) -> BindingValueMetadata | None:
        return self._current_value_metadata

    @property
    def artifact_id(self) -> str | None:
        return self._artifact_id

    @property
    def contribution_source_artifact_id(self) -> str | None:
        return self._contribution_source_artifact_id

    @property
    def selection(self) -> common_pb2.ArtifactSelection | None:
        if self._current_value_metadata is None:
            return None
        return clone_selection(self._current_value_metadata.selection)

    @property
    def byte_space(self) -> common_pb2.ByteSpaceRef:
        space = common_pb2.ByteSpaceRef()
        if self._view_id:
            space.kind = common_pb2.BYTE_SPACE_KIND_VIEW
            space.id = self._view_id
        else:
            space.kind = common_pb2.BYTE_SPACE_KIND_CANONICAL
            space.id = ""
        return space

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def published_lease_id(self) -> str | None:
        return self._published_lease_id

    @property
    def published_replica_id(self) -> str | None:
        return self._published_replica_id

    @property
    def dirty(self) -> bool:
        return self._dirty

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def publish_replica_operation(
        self,
        *,
        ttl_ms: int | None = None,
        owner_pid: int | None = None,
        ctx: CallContext | None = None,
    ) -> Operation[None]:
        self._ensure_open()
        if self._dirty:
            raise ArtifactError(
                "Slot contents are dirty; materialize again before publishing",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if self._published_lease_id is not None:
            return PollingOperation(
                operation_id=self._published_lease_id,
                status_fn=lambda: OperationStatus(
                    state="success",
                    message="publish already completed",
                ),
                result_fn=lambda: None,
                ctx=ctx,
            )
        if self._artifact_id is None or self.selection is None:
            raise ArtifactError(
                "publish_replica requires an artifact-backed current value",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not _selection_publishable(
            selection_names=self._selection_names,
            view_subset_hash=self._view_subset_hash,
            view_id=self._view_id,
        ):
            raise ArtifactError(
                "Slot selection is not publishable (packed or subset)",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not self._target_publication_token:
            raise ArtifactError(
                "target_publication_token missing; daemon publish not available",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        timeout_s = _ctx_timeout_s(ctx)
        operation_id = uuid.uuid4().hex
        client = self._runtime.ensure_client()
        try:
            start_resp = client.start_publish_target_replica(
                target_publication_token=self._target_publication_token,
                byte_space=self.byte_space,
                ttl_ms=ttl_ms,
                owner_pid=owner_pid,
                operation_id=operation_id,
                timeout_s=timeout_s if timeout_s is not None else 10.0,
            )
        except Exception as exc:  # noqa: BLE001
            error = _map_slot_error(exc)
            raise ArtifactError(
                str(error),
                status_code=error.status_code,
                retryable=error.retryable,
            ) from exc

        operation_ref = (
            start_resp.operation if hasattr(start_resp, "operation") else None
        )
        resolved_operation_id = (
            str(getattr(operation_ref, "operation_id", "") or "") or operation_id
        )

        def _decode(resp) -> None:
            payload = store_daemon_pb2.PublishTargetReplicaResponse()
            if not resp.status.result.Unpack(payload):
                raise ArtifactError(
                    "Unexpected publish target replica result type",
                    status_code="INTERNAL",
                    retryable=False,
                )
            lease_id = str(payload.lease_id or "")
            if not lease_id:
                raise ArtifactError(
                    "Publish target replica operation returned empty lease_id",
                    status_code="DATA_LOSS",
                    retryable=False,
                )
            self._published_lease_id = lease_id
            self._published_replica_id = (
                str(payload.replica_id) if payload.replica_id else None
            )
            return None

        return DaemonGlobalStoreOperation(
            operation_id=resolved_operation_id,
            runtime_ref=weakref.ref(self._runtime),
            ctx=ctx,
            context={
                "artifact_id": self._artifact_id or "",
                "byte_space_kind": str(int(self.byte_space.kind)),
            },
            result_factory=_decode,
            operation_ref=operation_ref,
        )

    def publish_replica(
        self,
        *,
        ttl_ms: int | None = None,
        owner_pid: int | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        self._ensure_open()
        if self._dirty:
            raise ArtifactError(
                "Slot contents are dirty; materialize again before publishing",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if self._published_lease_id is not None:
            return
        if self._artifact_id is None or self.selection is None:
            raise ArtifactError(
                "publish_replica requires an artifact-backed current value",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not _selection_publishable(
            selection_names=self._selection_names,
            view_subset_hash=self._view_subset_hash,
            view_id=self._view_id,
        ):
            raise ArtifactError(
                "Slot selection is not publishable (packed or subset)",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not self._target_publication_token:
            raise ArtifactError(
                "target_publication_token missing; daemon publish not available",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        timeout_s = _ctx_timeout_s(ctx)
        operation_id = uuid.uuid4().hex
        client = self._runtime.ensure_client()
        try:
            resp = client.publish_target_replica(
                target_publication_token=self._target_publication_token,
                byte_space=self.byte_space,
                ttl_ms=ttl_ms,
                owner_pid=owner_pid,
                operation_id=operation_id,
                timeout_s=timeout_s if timeout_s is not None else 60.0,
            )
        except Exception as exc:  # noqa: BLE001
            error = _map_slot_error(exc)
            raise ArtifactError(
                str(error),
                status_code=error.status_code,
                retryable=error.retryable,
            ) from exc
        lease_id = resp.lease_id if hasattr(resp, "lease_id") else ""
        if not lease_id:
            raise ArtifactError(
                "PublishTargetReplica returned empty lease_id",
                status_code="DATA_LOSS",
                retryable=False,
            )
        self._published_lease_id = lease_id
        self._published_replica_id = (
            resp.replica_id if hasattr(resp, "replica_id") and resp.replica_id else None
        )

    def retire(
        self,
        *,
        wait: bool = True,
        drain_timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        self._ensure_open()
        if self._published_lease_id is None:
            return
        operation_id = uuid.uuid4().hex
        self._retire_published(
            operation_id=operation_id,
            wait=wait,
            drain_timeout_s=drain_timeout_s,
            ctx=ctx,
        )

    def begin_update(
        self,
        *,
        drain_timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> "BindingUpdateEpoch":
        from tensorcast.api.store.binding import BindingUpdateEpoch

        self._ensure_open()
        operation_id = uuid.uuid4().hex
        if self._published_lease_id is not None:
            self._retire_published(
                operation_id=operation_id,
                wait=True,
                drain_timeout_s=drain_timeout_s,
                ctx=ctx,
            )
        timeout_s = _ctx_timeout_s(ctx)
        response = self._runtime.ensure_client().begin_binding_update(
            binding_id=self._binding_id,
            timeout_s=timeout_s if timeout_s is not None else 30.0,
        )
        update_epoch = str(getattr(response, "update_epoch", "") or "")
        if not update_epoch:
            raise ArtifactError(
                "BeginBindingUpdate returned an empty update_epoch",
                status_code="DATA_LOSS",
                retryable=False,
            )
        self._active_update_epoch = update_epoch
        self._current_value_metadata = None
        self._artifact_id = None
        self._target_publication_token = None
        self._dirty = False
        return BindingUpdateEpoch(
            binding_id=self._binding_id,
            update_epoch=update_epoch,
        )

    def seal_current(
        self,
        *,
        update_epoch: "BindingUpdateEpoch | str | int",
        ctx: CallContext | None = None,
    ) -> None:
        self._ensure_open()
        update_epoch_token = self._normalize_update_epoch(update_epoch)
        timeout_s = _ctx_timeout_s(ctx)
        try:
            response = self._runtime.ensure_client().seal_binding(
                binding_id=self._binding_id,
                update_epoch=update_epoch_token,
                timeout_s=timeout_s if timeout_s is not None else 30.0,
            )
        except Exception as exc:  # noqa: BLE001
            self._enter_dirty_state()
            error = _map_slot_error(exc)
            raise ArtifactError(
                str(error),
                status_code=error.status_code,
                retryable=error.retryable,
            ) from exc
        metadata = parse_binding_value_or_raise(
            response.current_value if hasattr(response, "current_value") else None,
            rpc_name="SealBinding",
            expected_binding_id=self._binding_id,
            expected_binding_layout_id=self._binding_layout_id,
        )
        if metadata is None:
            self._enter_dirty_state()
            raise ArtifactError(
                "SealBinding returned empty current_value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        self._active_update_epoch = None
        self._seal_generation_counter = int(metadata.seal_generation)
        self._current_value_metadata = metadata
        self._artifact_id = None
        self._target_publication_token = None
        self._dirty = False

    def swap(
        self,
        artifact: "Artifact | str",
        *,
        options: GetArtifactOptions | None = None,
        publish: bool = False,
        serving_runtime_policy: ServingRuntimePolicy | None = None,
        wait: bool = True,
        drain_timeout_s: float | None = None,
        ctx: CallContext | None = None,
        operation_id: str | None = None,
        publish_ttl_ms: int | None = None,
        publish_owner_pid: int | None = None,
    ) -> None:
        self._ensure_open()
        resolved = self._resolve_artifact(artifact)
        store, _, pipeline = resolved._require_components()
        if store is not self._store:
            raise ArtifactError(
                "Swap artifact must come from the same Store",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        resolved._ensure_metadata()
        canonical_index = resolved._canonical_index
        canonical_index_bytes = resolved._canonical_index_bytes
        if canonical_index is None or canonical_index_bytes is None:
            raise ArtifactError(
                "Missing canonical index for swap",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

        if self._copy_plan is not None:
            view_narrows = view_narrow_ranges(resolved._view_spec)
            validate_copy_plan(
                plan=self._copy_plan,
                canonical_index=canonical_index,
                target_tensors=self._tensors,
                view_narrows=view_narrows,
                require_full_coverage=True,
            )
            view_spec_proto = None
            if resolved._view_spec is not None and not resolved._view_spec.is_identity:
                view_spec_proto = resolved._view_spec.proto
            elif self._view_spec is not None and self._view_spec.tensors:
                view_spec_proto = self._view_spec
            selection_index_bytes = bytes(canonical_index_bytes)
            if view_spec_proto is not None and view_spec_proto.tensors:
                selection_index_bytes = compute_selected_index_bytes(
                    canonical_index_bytes=canonical_index_bytes,
                    view_spec=view_spec_proto,
                    tensor_names=None,
                )
            elif (
                resolved._view_metadata is not None
                and resolved._view_metadata.view_index_bytes
                and not str(resolved._view_metadata.view_id).startswith("mapped:v1:")
            ):
                selection_index_bytes = bytes(resolved._view_metadata.view_index_bytes)

            selection_order = self._selection_names if self._selection_names else None
            operation_id = operation_id or uuid.uuid4().hex
            if self._published_lease_id is not None:
                self._retire_published(
                    operation_id=operation_id,
                    wait=wait,
                    drain_timeout_s=drain_timeout_s,
                    ctx=ctx,
                )

            source_policy = _resolve_source_policy_from_options(options)
            artifact_id = resolved._ensure_identified()
            client = self._runtime.ensure_client()
            rpc_timeout_s = _ctx_timeout_s(ctx)
            source_view_id = resolved._mapped_source_view_id(self._runtime)
            mapped_view_id = compute_mapped_view_id(
                canonical_index_bytes=canonical_index_bytes,
                source_view_id=source_view_id,
                plan=self._copy_plan,
                target_tensors=self._tensors,
            )
            attempt = 0
            response = None
            region_layout = None
            while attempt < 2:
                region_layout = pipeline._build_mapped_region_backed_layout(
                    target=self._tensors,
                    device_id=self._device_id,
                    selection_order=selection_order,
                    mapped_view_id=mapped_view_id,
                    selection_index_bytes=selection_index_bytes,
                )
                self._ensure_layout_match(region_layout)
                try:
                    selection = _selection_from_region_layout(
                        artifact_id=artifact_id,
                        canonical_index_bytes=canonical_index_bytes,
                        region_layout=region_layout,
                        view_spec=view_spec_proto,
                    )
                    response = client.materialize_into_mapped_target(
                        selection=selection,
                        target_layout=region_layout.layout,
                        device_uuid=device_uuid_for(self._device_id),
                        source_policy=source_policy,
                        serving_runtime_policy=serving_runtime_policy,
                        copy_plan=self._copy_plan,
                        dst_tensors=self._tensors,
                        operation_id=operation_id,
                        timeout_s=rpc_timeout_s if rpc_timeout_s is not None else 600.0,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._enter_dirty_state()
                    message = str(exc)
                    if (
                        "MaterializeIntoMappedTarget" in message
                        and "not supported" in message.lower()
                    ):
                        raise ArtifactError(
                            "Mapped binding is not supported by the connected StoreDaemon",
                            status_code="FAILED_PRECONDITION",
                            retryable=False,
                        ) from exc
                    error = map_materialization_error(exc)
                    if attempt == 0 and _is_region_error(error):
                        self._refresh_regions()
                        attempt += 1
                        continue
                    raise ArtifactError(
                        str(error),
                        status_code=error.status_code,
                        retryable=False,
                    ) from exc

                if (
                    response.status
                    != store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
                ):
                    self._enter_dirty_state()
                    raise ArtifactError(
                        "MaterializeIntoMappedTarget returned non-success status",
                        status_code="DATA_LOSS",
                        retryable=False,
                    )
                break
            if response is None or region_layout is None:
                self._enter_dirty_state()
                raise ArtifactError(
                    "MaterializeIntoMappedTarget retry failed to produce a response",
                    status_code="DATA_LOSS",
                    retryable=False,
                )
            store_metrics.record_region_backed_verification_skipped(
                self._runtime.daemon_endpoint
            )
            self._dirty = False

            self._update_state_from_layout(
                region_layout=region_layout,
                view_spec=view_spec_proto,
                target_publication_token=getattr(
                    response, "target_publication_token", None
                ),
            )
            self._mark_artifact_backed_current(
                artifact_id=artifact_id,
                selection=selection,
                target_publication_token=getattr(
                    response, "target_publication_token", None
                ),
            )
            if publish:
                self._publish_with_operation_id(
                    operation_id,
                    ctx=ctx,
                    ttl_ms=publish_ttl_ms,
                    owner_pid=publish_owner_pid,
                )
            return
        view_spec_proto = None
        if resolved._view_spec is not None and not resolved._view_spec.is_identity:
            view_spec_proto = resolved._view_spec.proto
        elif self._view_spec is not None and self._view_spec.tensors:
            view_spec_proto = self._view_spec
        selection_order = self._selection_names if self._selection_names else None
        view_index_hint = None
        if selection_order is None and resolved._view_metadata is not None:
            view_index_hint = resolved._view_metadata.view_index_bytes
        operation_id = operation_id or uuid.uuid4().hex
        if self._published_lease_id is not None:
            self._retire_published(
                operation_id=operation_id,
                wait=wait,
                drain_timeout_s=drain_timeout_s,
                ctx=ctx,
            )

        source_policy = _resolve_source_policy_from_options(options)
        artifact_id = resolved._ensure_identified()
        client = self._runtime.ensure_client()
        rpc_timeout_s = _ctx_timeout_s(ctx)
        publish_checked = False
        attempt = 0
        response = None
        region_layout = None
        while attempt < 2:
            region_layout = pipeline._build_region_backed_layout(
                canonical_index=canonical_index,
                canonical_index_bytes=canonical_index_bytes,
                target=self._tensors,
                device_id=self._device_id,
                tensor_names=selection_order,
                view_spec=view_spec_proto,
                view_id=None,
                view_index_hint=view_index_hint,
                selection_order=selection_order,
            )
            self._ensure_layout_match(region_layout)
            if publish and not publish_checked:
                if not _selection_publishable(
                    selection_names=region_layout.selection_names,
                    view_subset_hash=region_layout.view_subset_hash,
                    view_id=_normalize_view_id(region_layout.view_id),
                ):
                    raise ArtifactError(
                        "Slot selection is not publishable (packed or subset)",
                        status_code="FAILED_PRECONDITION",
                        retryable=False,
                    )
                publish_checked = True
            try:
                selection = _selection_from_region_layout(
                    artifact_id=artifact_id,
                    canonical_index_bytes=canonical_index_bytes,
                    region_layout=region_layout,
                    view_spec=view_spec_proto,
                )
                response = client.materialize_into_target_v2(
                    selection=selection,
                    target_layout=region_layout.layout,
                    device_uuid=device_uuid_for(self._device_id),
                    source_policy=source_policy,
                    serving_runtime_policy=serving_runtime_policy,
                    operation_id=operation_id,
                    timeout_s=rpc_timeout_s if rpc_timeout_s is not None else 600.0,
                )
            except Exception as exc:  # noqa: BLE001
                self._enter_dirty_state()
                error = map_materialization_error(exc)
                if attempt == 0 and _is_region_error(error):
                    self._refresh_regions()
                    attempt += 1
                    continue
                raise ArtifactError(
                    str(error),
                    status_code=error.status_code,
                    retryable=False,
                ) from exc

            if (
                response.status
                != store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
            ):
                self._enter_dirty_state()
                raise ArtifactError(
                    "MaterializeIntoTarget returned non-success status",
                    status_code="DATA_LOSS",
                    retryable=False,
                )
            break
        if response is None or region_layout is None:
            self._enter_dirty_state()
            raise ArtifactError(
                "MaterializeIntoTarget retry failed to produce a response",
                status_code="DATA_LOSS",
                retryable=False,
            )
        store_metrics.record_region_backed_verification_skipped(
            self._runtime.daemon_endpoint
        )
        self._dirty = False

        self._update_state_from_layout(
            region_layout=region_layout,
            view_spec=view_spec_proto,
            target_publication_token=getattr(
                response, "target_publication_token", None
            ),
        )
        self._mark_artifact_backed_current(
            artifact_id=artifact_id,
            selection=selection,
            target_publication_token=getattr(
                response, "target_publication_token", None
            ),
        )

        if publish:
            try:
                self._publish_with_operation_id(
                    operation_id,
                    ctx=ctx,
                    ttl_ms=publish_ttl_ms,
                    owner_pid=publish_owner_pid,
                )
            except ArtifactError:
                raise

    def close(self) -> None:
        if self._closed:
            return
        with contextlib.suppress(Exception):
            if self._published_lease_id is not None:
                self.retire(wait=False)
        with contextlib.suppress(Exception):
            if self._binding_id:
                self._runtime.ensure_client().close_owned_binding(
                    binding_id=self._binding_id
                )
        self._closed = True
        self._release_regions()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._closed:
            raise ArtifactError(
                "InplaceSlot is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if self._runtime.closed:
            raise ArtifactError(
                "Store is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

    def _resolve_artifact(self, artifact: "Artifact | str") -> "Artifact":
        if isinstance(artifact, str):
            return self._store.artifact(ref=str(artifact))
        return artifact

    def _ensure_layout_match(self, region_layout: "_RegionBackedLayout") -> None:
        layout_hash = bytes(region_layout.layout.logical_layout_hash)
        if not layout_hash:
            index_bytes = (
                region_layout.view_index_bytes or self._layout.target_index_bytes
            )
            needs_view_index = (
                region_layout.layout.index_kind
                == store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW
            )
            layout_hash = compute_logical_layout_hash(
                index_bytes=index_bytes,
                needs_view_index=needs_view_index,
            )
        if layout_hash != self._logical_layout_hash:
            raise ArtifactError(
                "Swap target layout does not match slot layout",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        view_id = _normalize_view_id(region_layout.view_id)
        if view_id != self._view_id:
            raise ArtifactError(
                "Swap view_id does not match slot selection",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        subset_hash = bytes(region_layout.view_subset_hash or b"")
        if subset_hash != self._view_subset_hash:
            raise ArtifactError(
                "Swap view subset does not match slot selection",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        selection_names = tuple(region_layout.selection_names)
        if selection_names != self._selection_names:
            raise ArtifactError(
                "Swap tensor_names order does not match slot selection",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

    def _refresh_regions(self) -> None:
        stale_ids = tuple(self._region_ids)
        self._release_regions()
        for region_id in stale_ids:
            if not region_id:
                continue
            with contextlib.suppress(Exception):
                region_cache.unregister_region(region_id)
        ttl_ms = 0
        bases = collect_storage_bases(self._tensors)
        new_ids: list[str] = []
        for base_ptr, nbytes in sorted(bases.items()):
            handle = self._store.register_vram_region(
                device_id=self._device_id,
                base_ptr=base_ptr,
                size_bytes=nbytes,
                ttl_ms=int(ttl_ms),
            )
            new_ids.append(handle.region_id)
        self._region_ids = tuple(new_ids)

    def _retire_published(
        self,
        *,
        operation_id: str,
        wait: bool,
        drain_timeout_s: float | None,
        ctx: CallContext | None,
    ) -> None:
        lease_id = self._published_lease_id
        if lease_id is None:
            return
        timeout_s = _ctx_timeout_s(ctx)
        drain_timeout_ms = (
            int(drain_timeout_s * 1000) if drain_timeout_s is not None else None
        )
        client = self._runtime.ensure_client()
        try:
            _ = client.retire_published_replica(
                artifact_id=self._artifact_id or "",
                byte_space=self.byte_space,
                lease_id=lease_id,
                device_id=self._device_id,
                wait_for_drain=bool(wait),
                drain_timeout_ms=drain_timeout_ms,
                operation_id=operation_id,
                timeout_s=timeout_s if timeout_s is not None else 60.0,
            )
        except Exception as exc:  # noqa: BLE001
            error = _map_slot_error(exc)
            raise ArtifactError(
                str(error),
                status_code=error.status_code,
                retryable=error.retryable,
            ) from exc
        self._published_lease_id = None
        self._published_replica_id = None

    def _publish_with_operation_id(
        self,
        operation_id: str,
        *,
        ttl_ms: int | None = None,
        owner_pid: int | None = None,
        ctx: CallContext | None,
    ) -> None:
        if not self._target_publication_token:
            raise ArtifactError(
                "target_publication_token missing; daemon publish not available",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        timeout_s = _ctx_timeout_s(ctx)
        client = self._runtime.ensure_client()
        try:
            resp = client.publish_target_replica(
                target_publication_token=self._target_publication_token,
                byte_space=self.byte_space,
                ttl_ms=ttl_ms,
                owner_pid=owner_pid,
                operation_id=operation_id,
                timeout_s=timeout_s if timeout_s is not None else 60.0,
            )
        except Exception as exc:  # noqa: BLE001
            error = _map_slot_error(exc)
            raise ArtifactError(
                str(error),
                status_code=error.status_code,
                retryable=error.retryable,
            ) from exc
        lease_id = resp.lease_id if hasattr(resp, "lease_id") else ""
        if not lease_id:
            raise ArtifactError(
                "PublishTargetReplica returned empty lease_id",
                status_code="DATA_LOSS",
                retryable=False,
            )
        self._published_lease_id = lease_id
        self._published_replica_id = (
            resp.replica_id if hasattr(resp, "replica_id") and resp.replica_id else None
        )

    def _release_regions(self) -> None:
        for region_id in self._region_ids:
            if not region_id:
                continue
            with contextlib.suppress(Exception):
                self._store.unregister_vram_region(region_id)
        self._region_ids = ()

    def _update_state_from_layout(
        self,
        *,
        region_layout: "_RegionBackedLayout",
        view_spec: common_pb2.ViewSpec | None,
        target_publication_token: bytes | None,
    ) -> None:
        self._region_ids = tuple(region_layout.region_ids)
        self._view_spec = view_spec
        self._selection_names = tuple(region_layout.selection_names)
        self._view_id = _normalize_view_id(region_layout.view_id)
        self._view_subset_hash = bytes(region_layout.view_subset_hash or b"")
        self._logical_layout_hash = bytes(region_layout.layout.logical_layout_hash)
        if not self._logical_layout_hash:
            index_bytes = (
                region_layout.view_index_bytes or self._layout.target_index_bytes
            )
            needs_view_index = (
                region_layout.layout.index_kind
                == store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW
            )
            self._logical_layout_hash = compute_logical_layout_hash(
                index_bytes=index_bytes,
                needs_view_index=needs_view_index,
            )
        self._selection_hash = compute_selection_hash(
            view_id=self._view_id,
            view_subset_hash=self._view_subset_hash,
        )
        self._target_publication_token = (
            bytes(target_publication_token) if target_publication_token else None
        )

    def _mark_artifact_backed_current(
        self,
        *,
        artifact_id: str,
        selection: common_pb2.ArtifactSelection,
        target_publication_token: bytes | None,
    ) -> None:
        self._artifact_id = str(artifact_id)
        self._contribution_source_artifact_id = str(artifact_id)
        try:
            response = self._runtime.ensure_client().commit_binding_artifact(
                binding_id=self._binding_id,
                selection=selection,
                source_artifact_id=artifact_id,
                target_publication_token=target_publication_token,
                timeout_s=30.0,
            )
        except Exception as exc:  # noqa: BLE001
            self._enter_dirty_state()
            error = _map_slot_error(exc)
            raise ArtifactError(
                str(error),
                status_code=error.status_code,
                retryable=error.retryable,
            ) from exc
        metadata = parse_binding_value_or_raise(
            response.current_value
            if response is not None and hasattr(response, "current_value")
            else None,
            rpc_name="CommitBindingArtifact",
            expected_binding_id=self._binding_id,
            expected_binding_layout_id=self._binding_layout_id,
        )
        if metadata is None:
            self._enter_dirty_state()
            raise ArtifactError(
                "CommitBindingArtifact returned empty current_value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        self._seal_generation_counter = int(metadata.seal_generation)
        self._current_value_metadata = metadata
        self._target_publication_token = (
            bytes(target_publication_token) if target_publication_token else None
        )
        self._dirty = False

    def _enter_dirty_state(self) -> None:
        self._active_update_epoch = None
        self._current_value_metadata = None
        self._artifact_id = None
        self._target_publication_token = None
        self._dirty = True

    def _normalize_update_epoch(
        self,
        update_epoch: "BindingUpdateEpoch | str | int",
    ) -> str:
        if hasattr(update_epoch, "update_epoch"):
            binding_id = getattr(update_epoch, "binding_id", None)
            if binding_id is not None and str(binding_id) != self._binding_id:
                raise ArtifactError(
                    "BindingUpdateEpoch does not belong to this binding",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            return str(update_epoch.update_epoch)
        token = str(update_epoch)
        if token.startswith("bue:"):
            parts = token.split(":", 2)
            if len(parts) == 3 and parts[1] and parts[1] != self._binding_id:
                raise ArtifactError(
                    "update_epoch token does not belong to this binding",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
        return token


__all__ = ["InplaceSlot"]
