#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import contextlib
import uuid
import weakref
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Sequence, cast

import torch

from tensorcast._c_ext import get_cuda_memory_ptr, restore_tensors
from tensorcast.api._config import GetArtifactOptions
from tensorcast.api._device import device_uuid_for
from tensorcast.api._materialize import _tensor_payload_from_proto
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
from tensorcast.api.store.inplace_slot import (
    _ctx_timeout_s,
    _map_slot_error,
    _normalize_view_id,
    _selection_publishable,
)
from tensorcast.api.store.materialization import _resolve_source_policy_from_options
from tensorcast.api.store.owned_binding_layout import BindingLayout
from tensorcast.api.store.realization_plan import binding_realization_plan_to_proto
from tensorcast.api.store.retry import map_materialization_error
from tensorcast.api.store.types import ArtifactError
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import (
    ExecutionDiagnostics,
    PublicDiskSourceHandle,
    ServerConfig,
    ServingRuntimePolicy,
    SourceBoundCapability,
    SourceBoundPlanDiagnostics,
)

if TYPE_CHECKING:
    from tensorcast.api.store import Store
    from tensorcast.api.store.artifact import Artifact
    from tensorcast.api.store.binding import BindingUpdateEpoch
    from tensorcast.api.store.runtime import StoreRuntimeContext


_MIN_SOURCE_BOUND_CONTRACT_VERSION = 4
_REQUIRED_SOURCE_BOUND_CAPABILITIES = (
    SourceBoundCapability.FIRST_CLASS_COLLECTIVE_INGRESS,
    SourceBoundCapability.TYPED_EXECUTION_DIAGNOSTICS,
    SourceBoundCapability.SINGLE_MINT_BINDING_CLOSEOUT,
)


def _raise_if_published_for_mutation(
    *,
    op_name: str,
    published_lease_id: str | None,
) -> None:
    if published_lease_id is None:
        return
    raise ArtifactError(
        f"{op_name}() requires an unpublished binding; call retire() first",
        status_code="FAILED_PRECONDITION",
        retryable=False,
    )


def _resolve_server_config(runtime: "StoreRuntimeContext") -> ServerConfig | None:
    server_config = runtime.capabilities.server_config
    if server_config is not None:
        return server_config
    return runtime.ensure_client().get_server_config()


def _source_bound_capability_names(flags: int) -> list[str]:
    return [
        str(capability.name)
        for capability in SourceBoundCapability
        if capability.name is not None and flags & int(capability)
    ]


def _require_execution_only_realize_contract(
    runtime: "StoreRuntimeContext",
) -> None:
    server_config = _resolve_server_config(runtime)
    version = int(getattr(server_config, "source_bound_contract_version", 0) or 0)
    flags = int(getattr(server_config, "source_bound_capability_flags", 0) or 0)
    missing = [
        capability.name
        for capability in _REQUIRED_SOURCE_BOUND_CAPABILITIES
        if not (flags & int(capability))
    ]
    if version >= _MIN_SOURCE_BOUND_CONTRACT_VERSION and not missing:
        return
    required = [capability.name for capability in _REQUIRED_SOURCE_BOUND_CAPABILITIES]
    raise ArtifactError(
        "realize_from() requires source-bound contract v4 with capabilities "
        f"{required}; daemon reported version={version}, "
        f"capabilities={_source_bound_capability_names(flags)}",
        status_code="FAILED_PRECONDITION",
        retryable=False,
    )


def restore_owned_binding_tensors(
    *,
    response: store_daemon_pb2.CreateBindingResponse
    | store_daemon_pb2.CreateOwnedBindingResponse,
    runtime: "StoreRuntimeContext",
    device_id: int,
) -> dict[str, torch.Tensor]:
    if not response.HasField("mem_handle"):
        raise ArtifactError(
            "CreateOwnedBinding returned empty mem_handle",
            status_code="DATA_LOSS",
            retryable=False,
        )
    mem_handle = response.mem_handle
    if mem_handle.WhichOneof("handle") != "cuda_ipc_handle":
        raise ArtifactError(
            "CreateOwnedBinding returned a non-GPU handle",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    cuda_ipc_handle = bytes(mem_handle.cuda_ipc_handle)
    if not cuda_ipc_handle:
        raise ArtifactError(
            "CreateOwnedBinding returned empty cuda_ipc_handle",
            status_code="DATA_LOSS",
            retryable=False,
        )
    server_config = runtime.capabilities.server_config
    if server_config is None:
        server_config = runtime.ensure_client().get_server_config()
    local_handle_socket_path = (
        server_config.local_handle_socket_path if server_config is not None else ""
    )
    lease_token = bytes(mem_handle.lease_token)
    if lease_token and not local_handle_socket_path:
        raise ArtifactError(
            "lease_token present but local_handle_socket_path is missing",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    device_uuid = None
    try:
        device_uuid = device_uuid_for(device_id)
    except Exception:  # noqa: BLE001
        device_uuid = None
    descriptors = [
        _tensor_payload_from_proto(desc, default_device_uuid=device_uuid)
        for desc in response.payloads
    ]
    meta_state_dict = {
        desc.name: (
            list(desc.shape),
            list(desc.stride),
            desc.dtype,
            int(desc.storage_offset),
        )
        for desc in descriptors
    }
    tensor_offsets = {desc.name: int(desc.buffer_offset) for desc in descriptors}
    cuda_memory_ptr = get_cuda_memory_ptr(device_id, cuda_ipc_handle)
    return restore_tensors(
        meta_state_dict,
        {int(device_id): int(cuda_memory_ptr)},
        {int(device_id): tensor_offsets},
        True,
        lease_token=lease_token,
        local_handle_socket_path=local_handle_socket_path,
    )


def _collective_group_to_proto(
    group: object | None,
) -> store_daemon_pb2.CollectiveLoadGroup | None:
    if group is None:
        return None
    group_id = str(getattr(group, "group_id", "") or "")
    raw_world_size = getattr(group, "world_size", 0)
    world_size = 0 if raw_world_size is None else int(raw_world_size)
    # rank=0 is valid for collective ingress; avoid falsy normalization that
    # rewrites it to -1.
    raw_rank = getattr(group, "rank", -1)
    rank = -1 if raw_rank is None else int(raw_rank)
    if not group_id:
        return None
    if world_size <= 1 or rank < 0 or rank >= world_size:
        raise ArtifactError(
            "collective_group requires a non-empty group_id with world_size > 1 and a valid rank",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    proto = store_daemon_pb2.CollectiveLoadGroup()
    proto.group_id = group_id
    proto.world_size = world_size
    proto.rank = rank
    return proto


def _source_locality_to_proto(
    value: object,
) -> store_daemon_pb2.SourceLocality:
    normalized = str(getattr(value, "value", value) or "auto").strip().lower()
    if normalized == "host_local":
        return store_daemon_pb2.SOURCE_LOCALITY_HOST_LOCAL
    if normalized == "shared_source":
        return store_daemon_pb2.SOURCE_LOCALITY_SHARED_SOURCE
    return store_daemon_pb2.SOURCE_LOCALITY_AUTO


def _build_source_execution_contract(
    *,
    options: GetArtifactOptions | None,
    ctx: CallContext | None,
) -> tuple[
    store_daemon_pb2.SourceExecutionTopology | None,
    store_daemon_pb2.CollectivePolicy | None,
]:
    execution_topology = None if options is None else options.execution_topology
    explicit_collective_group = (
        None if execution_topology is None else execution_topology.collective_group
    )
    ctx_collective_group = None if ctx is None else ctx.collective
    if ctx_collective_group is not None:
        raise ArtifactError(
            "ctx.collective is no longer accepted for daemon-owned source-bound "
            "materialization; use options.execution_topology.collective_group",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )

    policy_mode = (
        None
        if execution_topology is None
        else getattr(execution_topology, "collective_policy", None)
    )
    if policy_mode is None and explicit_collective_group is not None:
        from tensorcast.api._config import CollectivePolicyMode

        policy_mode = CollectivePolicyMode.REQUIRE_COLLECTIVE

    if str(getattr(policy_mode, "value", policy_mode) or "") == "disable_collective":
        if explicit_collective_group is not None:
            raise ArtifactError(
                "collective_policy=disable_collective conflicts with a collective group",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
    elif policy_mode is not None and explicit_collective_group is None:
        raise ArtifactError(
            "collective_policy requires execution_topology.collective_group",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )

    topology_proto: store_daemon_pb2.SourceExecutionTopology | None = None
    source_locality = (
        execution_topology.source_locality if execution_topology is not None else None
    )
    source_sharing_domain = (
        None
        if execution_topology is None
        else str(execution_topology.source_sharing_domain or "") or None
    )
    if (
        explicit_collective_group is not None
        or source_locality is not None
        or source_sharing_domain is not None
    ):
        topology_proto = store_daemon_pb2.SourceExecutionTopology()
        collective_group_proto = _collective_group_to_proto(explicit_collective_group)
        if collective_group_proto is not None:
            topology_proto.collective_load_group.CopyFrom(collective_group_proto)
        if source_locality is not None:
            topology_proto.source_locality = _source_locality_to_proto(source_locality)
        if source_sharing_domain is not None:
            topology_proto.source_sharing_domain = source_sharing_domain

    if policy_mode is None:
        return topology_proto, None

    normalized_policy = str(getattr(policy_mode, "value", policy_mode)).strip().lower()
    if normalized_policy == "require_collective":
        return (
            topology_proto,
            store_daemon_pb2.COLLECTIVE_POLICY_REQUIRE_COLLECTIVE,
        )
    if normalized_policy == "collective_first":
        return (
            topology_proto,
            store_daemon_pb2.COLLECTIVE_POLICY_COLLECTIVE_FIRST,
        )
    return topology_proto, store_daemon_pb2.COLLECTIVE_POLICY_DISABLE_COLLECTIVE


def _execution_diagnostics_from_response(
    response: object,
) -> ExecutionDiagnostics | None:
    diagnostics_proto = getattr(response, "execution_diagnostics", None)
    if diagnostics_proto is None:
        return None
    has_field = getattr(response, "HasField", None)
    if callable(has_field):
        try:
            if not has_field("execution_diagnostics"):
                return None
        except ValueError:
            pass
    return ExecutionDiagnostics.from_proto(diagnostics_proto)


def _source_bound_plan_diagnostics_from_response(
    response: object,
) -> SourceBoundPlanDiagnostics | None:
    diagnostics_proto = getattr(response, "source_bound_plan_diagnostics", None)
    if diagnostics_proto is None:
        return None
    has_field = getattr(response, "HasField", None)
    if callable(has_field):
        try:
            if not has_field("source_bound_plan_diagnostics"):
                return None
        except ValueError:
            pass
    return SourceBoundPlanDiagnostics.from_proto(diagnostics_proto)


class OwnedBindingSlot:
    """Stable, daemon-owned CUDA layout that can be refilled in-place."""

    def __init__(
        self,
        *,
        store: "Store",
        runtime: "StoreRuntimeContext",
        tensors: Mapping[str, torch.Tensor],
        layout: BindingLayout,
        binding_id: str,
        current_value_metadata: BindingValueMetadata | None,
        device: torch.device,
        device_id: int,
        target_publication_token: bytes | None,
        target_publication_operation_id: str | None = None,
    ) -> None:
        if not tensors:
            raise ArtifactError(
                "OwnedBindingSlot requires at least one tensor",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._store = store
        self._runtime = runtime
        self._tensors = dict(tensors)
        self._tensors_view: Mapping[str, torch.Tensor] = MappingProxyType(self._tensors)
        self._layout = layout
        self._binding_id = str(binding_id)
        self._binding_layout_id = str(layout.binding_layout_id)
        self._current_value_metadata = current_value_metadata
        self._selection = clone_selection(
            None if current_value_metadata is None else current_value_metadata.selection
        )
        self._contribution_selection = clone_selection(self._selection)
        self._contribution_source_artifact_id = (
            None
            if current_value_metadata is None
            or current_value_metadata.source_artifact_id is None
            else str(current_value_metadata.source_artifact_id)
        )
        self._device = device
        self._device_id = int(device_id)
        self._target_publication_token = (
            bytes(target_publication_token) if target_publication_token else None
        )
        self._target_publication_operation_id = (
            str(target_publication_operation_id)
            if target_publication_operation_id
            else None
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
        self._last_execution_diagnostics: ExecutionDiagnostics | None = None
        self._last_source_bound_plan_diagnostics: SourceBoundPlanDiagnostics | None = (
            None
        )

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
        if self._current_value_metadata is None:
            return None
        return self._current_value_metadata.source_artifact_id

    @property
    def contribution_selection(self) -> common_pb2.ArtifactSelection | None:
        return clone_selection(self._contribution_selection)

    @property
    def contribution_source_artifact_id(self) -> str | None:
        return self._contribution_source_artifact_id

    @property
    def selection(self) -> common_pb2.ArtifactSelection | None:
        return clone_selection(self._selection)

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

    @property
    def last_execution_diagnostics(self) -> ExecutionDiagnostics | None:
        return self._last_execution_diagnostics

    @property
    def last_source_bound_plan_diagnostics(self) -> SourceBoundPlanDiagnostics | None:
        return self._last_source_bound_plan_diagnostics

    @property
    def byte_space(self) -> common_pb2.ByteSpaceRef:
        space = common_pb2.ByteSpaceRef()
        if self._selection is not None and self._selection.view_id:
            space.kind = common_pb2.BYTE_SPACE_KIND_VIEW
            space.id = str(self._selection.view_id)
        else:
            space.kind = common_pb2.BYTE_SPACE_KIND_CANONICAL
            space.id = ""
        return space

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
                "Binding contents are dirty; materialize again before publishing",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if self._published_lease_id is not None:
            return
        if self._selection is None or self.artifact_id is None:
            raise ArtifactError(
                "publish_replica requires an artifact-backed current value",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        selection_names = tuple(str(name) for name in self._selection.tensor_names)
        if not _selection_publishable(
            selection_names=selection_names,
            view_subset_hash=bytes(self._selection.view_subset_hash or b""),
            view_id=_normalize_view_id(self._selection.view_id),
        ):
            raise ArtifactError(
                "Binding selection is not publishable (packed or subset)",
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
        operation_id = self._target_publication_operation_id or uuid.uuid4().hex
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
        self._published_lease_id = str(lease_id)
        self._published_replica_id = (
            str(resp.replica_id)
            if hasattr(resp, "replica_id") and resp.replica_id
            else None
        )

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
                "Binding contents are dirty; materialize again before publishing",
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
        if self._selection is None or self.artifact_id is None:
            raise ArtifactError(
                "publish_replica requires an artifact-backed current value",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        selection_names = tuple(str(name) for name in self._selection.tensor_names)
        if not _selection_publishable(
            selection_names=selection_names,
            view_subset_hash=bytes(self._selection.view_subset_hash or b""),
            view_id=_normalize_view_id(self._selection.view_id),
        ):
            raise ArtifactError(
                "Binding selection is not publishable (packed or subset)",
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
        operation_id = self._target_publication_operation_id or uuid.uuid4().hex
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
                "artifact_id": self.artifact_id or "",
                "binding_id": self._binding_id,
            },
            result_factory=_decode,
            operation_ref=operation_ref,
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
        timeout_s = _ctx_timeout_s(ctx)
        drain_timeout_ms = (
            int(drain_timeout_s * 1000) if drain_timeout_s is not None else None
        )
        client = self._runtime.ensure_client()
        try:
            _ = client.retire_published_replica(
                artifact_id=self.artifact_id or "",
                byte_space=self.byte_space,
                lease_id=self._published_lease_id,
                device_id=self._device_id,
                wait_for_drain=bool(wait),
                drain_timeout_ms=drain_timeout_ms,
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

    def begin_update(
        self,
        *,
        drain_timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> "BindingUpdateEpoch":
        from tensorcast.api.store.binding import BindingUpdateEpoch

        self._ensure_open()
        _raise_if_published_for_mutation(
            op_name="begin_update",
            published_lease_id=self._published_lease_id,
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
        self._selection = None
        self._target_publication_token = None
        self._target_publication_operation_id = None
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
        self._selection = None
        self._target_publication_token = None
        self._target_publication_operation_id = None
        self._dirty = False

    def promote_current_value(
        self,
        *,
        binding_value_id: str,
        ctx: CallContext | None = None,
    ) -> store_daemon_pb2.PromoteBindingCurrentValueResponse:
        self._ensure_open()
        self._last_execution_diagnostics = None
        self._last_source_bound_plan_diagnostics = None
        timeout_s = _ctx_timeout_s(ctx)
        try:
            response = self._runtime.ensure_client().promote_binding_current_value(
                binding_id=self._binding_id,
                binding_value_id=str(binding_value_id),
                timeout_s=timeout_s if timeout_s is not None else 30.0,
            )
        except Exception as exc:  # noqa: BLE001
            error = _map_slot_error(exc)
            raise ArtifactError(
                str(error),
                status_code=error.status_code,
                retryable=error.retryable,
            ) from exc
        metadata = parse_binding_value_or_raise(
            response.current_value if hasattr(response, "current_value") else None,
            rpc_name="PromoteBindingCurrentValue",
            expected_binding_id=self._binding_id,
            expected_binding_layout_id=self._binding_layout_id,
        )
        if metadata is None:
            raise ArtifactError(
                "PromoteBindingCurrentValue returned empty current_value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        self._seal_generation_counter = int(metadata.seal_generation)
        self._current_value_metadata = metadata
        self._selection = clone_selection(metadata.selection)
        self._contribution_selection = clone_selection(metadata.selection)
        self._contribution_source_artifact_id = metadata.source_artifact_id
        self._target_publication_token = None
        self._target_publication_operation_id = None
        self._dirty = False
        self._last_execution_diagnostics = _execution_diagnostics_from_response(
            response
        )
        self._last_source_bound_plan_diagnostics = None
        return response

    def realize_from(
        self,
        artifact: "Artifact | PublicDiskSourceHandle | str",
        *,
        realization_plan: object,
        options: GetArtifactOptions | None = None,
        ctx: CallContext | None = None,
        operation_id: str | None = None,
    ) -> "BindingUpdateEpoch":
        from tensorcast.api.store.binding import BindingUpdateEpoch

        self._ensure_open()
        _raise_if_published_for_mutation(
            op_name="realize_from",
            published_lease_id=self._published_lease_id,
        )
        _require_execution_only_realize_contract(self._runtime)
        public_disk_source = (
            artifact if isinstance(artifact, PublicDiskSourceHandle) else None
        )
        resolved = None
        if public_disk_source is None:
            resolved = self._resolve_artifact(cast("Artifact | str", artifact))
        if resolved is not None:
            store, _, _ = resolved._require_components()
            if store is not self._store:
                raise ArtifactError(
                    "Realization source artifact must come from the same Store",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
        else:
            resolved = None
        source_policy = _resolve_source_policy_from_options(options)
        execution_topology, collective_policy = _build_source_execution_contract(
            options=options,
            ctx=ctx,
        )
        self._last_execution_diagnostics = None
        self._last_source_bound_plan_diagnostics = None
        rpc_timeout_s = _ctx_timeout_s(ctx)
        try:
            source_selection = None
            if resolved is not None and hasattr(
                resolved, "_build_owner_source_selection"
            ):
                ensure_metadata = getattr(resolved, "_ensure_metadata", None)
                if callable(ensure_metadata):
                    ensure_metadata()
                canonical_index_bytes = getattr(
                    resolved, "_canonical_index_bytes", None
                )
                if canonical_index_bytes is not None:
                    view_spec = getattr(resolved, "_view_spec", None)
                    view_spec_proto = view_spec.proto if view_spec is not None else None
                    source_selection = resolved._build_owner_source_selection(
                        packing="byte_space",
                        view_spec_proto=view_spec_proto,
                        canonical_index_bytes=canonical_index_bytes,
                    )
            response = self._runtime.ensure_client().refill_owned_binding(
                binding_id=self._binding_id,
                artifact_id=(
                    ""
                    if public_disk_source is None
                    else str(public_disk_source.artifact_id)
                )
                if resolved is None
                else resolved._ensure_identified(),
                public_disk_source=(
                    None
                    if public_disk_source is None
                    else public_disk_source.to_proto()
                ),
                source_selection=source_selection,
                realization_plan=binding_realization_plan_to_proto(
                    cast(Sequence[object], realization_plan),
                    target_index_bytes=self._layout.target_index_bytes,
                ),
                source_policy=source_policy,
                execution_topology=execution_topology,
                collective_policy=collective_policy,
                operation_id=operation_id,
                timeout_s=rpc_timeout_s if rpc_timeout_s is not None else 600.0,
            )
        except Exception as exc:  # noqa: BLE001
            error = map_materialization_error(exc)
            if error.status_code == "DATA_LOSS":
                self._enter_dirty_state()
            raise ArtifactError(
                str(error),
                status_code=error.status_code,
                retryable=False,
            ) from exc
        update_epoch = str(getattr(response, "update_epoch", "") or "")
        if not update_epoch:
            self._enter_dirty_state()
            raise ArtifactError(
                "RefillOwnedBinding returned empty update_epoch",
                status_code="DATA_LOSS",
                retryable=False,
            )
        self._dirty = False
        self._active_update_epoch = update_epoch
        self._selection = None
        self._contribution_selection = None
        self._contribution_source_artifact_id = None
        self._target_publication_token = None
        self._target_publication_operation_id = None
        self._current_value_metadata = None
        self._last_execution_diagnostics = _execution_diagnostics_from_response(
            response
        )
        self._last_source_bound_plan_diagnostics = (
            _source_bound_plan_diagnostics_from_response(response)
        )
        return BindingUpdateEpoch(
            binding_id=self._binding_id,
            update_epoch=update_epoch,
        )

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
        store, _, _ = resolved._require_components()
        if store is not self._store:
            raise ArtifactError(
                "Swap artifact must come from the same Store",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if self._published_lease_id is not None:
            self.retire(
                wait=wait,
                drain_timeout_s=drain_timeout_s,
                ctx=ctx,
            )
        source_policy = _resolve_source_policy_from_options(options)
        execution_topology, collective_policy = _build_source_execution_contract(
            options=options,
            ctx=ctx,
        )
        self._last_execution_diagnostics = None
        self._last_source_bound_plan_diagnostics = None
        rpc_timeout_s = _ctx_timeout_s(ctx)
        try:
            source_selection = None
            if hasattr(resolved, "_build_owner_source_selection"):
                ensure_metadata = getattr(resolved, "_ensure_metadata", None)
                if callable(ensure_metadata):
                    ensure_metadata()
                canonical_index_bytes = getattr(
                    resolved, "_canonical_index_bytes", None
                )
                if canonical_index_bytes is not None:
                    view_spec = getattr(resolved, "_view_spec", None)
                    view_spec_proto = view_spec.proto if view_spec is not None else None
                    source_selection = resolved._build_owner_source_selection(
                        packing="byte_space",
                        view_spec_proto=view_spec_proto,
                        canonical_index_bytes=canonical_index_bytes,
                    )
            response = self._runtime.ensure_client().refill_owned_binding(
                binding_id=self._binding_id,
                artifact_id=resolved._ensure_identified(),
                source_selection=source_selection,
                source_policy=source_policy,
                execution_topology=execution_topology,
                collective_policy=collective_policy,
                serving_runtime_policy=serving_runtime_policy,
                operation_id=operation_id,
                timeout_s=rpc_timeout_s if rpc_timeout_s is not None else 600.0,
            )
        except Exception as exc:  # noqa: BLE001
            error = map_materialization_error(exc)
            if error.status_code == "DATA_LOSS":
                self._enter_dirty_state()
            raise ArtifactError(
                str(error),
                status_code=error.status_code,
                retryable=False,
            ) from exc
        self._dirty = False
        self._selection = clone_selection(response.resolved_selection)
        self._contribution_selection = clone_selection(response.resolved_selection)
        self._contribution_source_artifact_id = str(response.artifact_id)
        self._target_publication_token = (
            bytes(response.target_publication_token)
            if getattr(response, "target_publication_token", b"")
            else None
        )
        self._target_publication_operation_id = (
            str(operation_id)
            if operation_id and self._target_publication_token
            else None
        )
        metadata = parse_binding_value_or_raise(
            response.current_value if hasattr(response, "current_value") else None,
            rpc_name="RefillOwnedBinding",
            expected_binding_id=self._binding_id,
            expected_binding_layout_id=self._binding_layout_id,
        )
        if metadata is None:
            self._enter_dirty_state()
            raise ArtifactError(
                "RefillOwnedBinding returned empty current_value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        self._seal_generation_counter = int(metadata.seal_generation)
        self._current_value_metadata = metadata
        self._last_execution_diagnostics = _execution_diagnostics_from_response(
            response
        )
        self._last_source_bound_plan_diagnostics = (
            _source_bound_plan_diagnostics_from_response(response)
        )
        if publish:
            self.publish_replica(
                ttl_ms=publish_ttl_ms,
                owner_pid=publish_owner_pid,
                ctx=ctx,
            )

    def close(self) -> None:
        if self._closed:
            return
        with contextlib.suppress(Exception):
            if self._published_lease_id is not None:
                self.retire(wait=False)
        self._runtime.ensure_client().close_owned_binding(binding_id=self._binding_id)
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise ArtifactError(
                "OwnedBindingSlot is closed",
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

    def _enter_dirty_state(self) -> None:
        self._active_update_epoch = None
        self._current_value_metadata = None
        self._selection = None
        self._target_publication_token = None
        self._target_publication_operation_id = None
        self._dirty = True


__all__ = ["OwnedBindingSlot", "restore_owned_binding_tensors"]
