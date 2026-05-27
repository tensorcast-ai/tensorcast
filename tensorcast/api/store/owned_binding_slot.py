#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import functools
import logging
import time
import uuid
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
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
from tensorcast.api.store.realization_kernel import (
    binding_materialization_diagnostics_from_response,
    execution_diagnostics_from_response,
    source_bound_plan_diagnostics_from_response,
)
from tensorcast.api.store.realization_plan import binding_realization_plan_to_proto
from tensorcast.api.store.retry import map_materialization_error
from tensorcast.api.store.types import ArtifactError
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import (
    BindingValueRef,
    ExecutionDiagnostics,
    GroupRealizationAcquireRef,
    PublicDiskSourceHandle,
    RuntimeArtifactPolicy,
    ServerConfig,
    SourceBoundCapability,
    SourceBoundPlanDiagnostics,
)

logger = logging.getLogger(__name__)

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

_RESTORE_EXECUTOR: ThreadPoolExecutor | None = None


def _owned_binding_restore_executor() -> ThreadPoolExecutor:
    global _RESTORE_EXECUTOR
    if _RESTORE_EXECUTOR is None:
        _RESTORE_EXECUTOR = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tc-owned-binding-restore",
        )
    return _RESTORE_EXECUTOR


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
    profile_start = time.perf_counter()
    profile_last = profile_start
    logger.info(
        "tc_profile_py restore_owned_binding_tensors enter device_id=%d payloads=%d has_mem_handle=%s",
        device_id,
        len(getattr(response, "payloads", ())),
        response.HasField("mem_handle"),
    )
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
    now = time.perf_counter()
    logger.info(
        "tc_profile_py restore_owned_binding_tensors parsed_handle handle_bytes=%d "
        "lease_token_bytes=%d step_sec=%.6f total_sec=%.6f",
        len(cuda_ipc_handle),
        len(bytes(mem_handle.lease_token)),
        now - profile_last,
        now - profile_start,
    )
    profile_last = now
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
    now = time.perf_counter()
    logger.info(
        "tc_profile_py restore_owned_binding_tensors resolved_server_config "
        "local_handle_socket_path=%s step_sec=%.6f total_sec=%.6f",
        bool(local_handle_socket_path),
        now - profile_last,
        now - profile_start,
    )
    profile_last = now
    device_uuid = None
    try:
        device_uuid = device_uuid_for(device_id)
    except Exception:  # noqa: BLE001
        device_uuid = None
    descriptors = [
        _tensor_payload_from_proto(desc, default_device_uuid=device_uuid)
        for desc in response.payloads
    ]
    now = time.perf_counter()
    logger.info(
        "tc_profile_py restore_owned_binding_tensors built_descriptors "
        "device_uuid=%s descriptor_count=%d step_sec=%.6f total_sec=%.6f",
        device_uuid,
        len(descriptors),
        now - profile_last,
        now - profile_start,
    )
    profile_last = now
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
    now = time.perf_counter()
    logger.info(
        "tc_profile_py restore_owned_binding_tensors built_metadata "
        "tensor_count=%d step_sec=%.6f total_sec=%.6f",
        len(meta_state_dict),
        now - profile_last,
        now - profile_start,
    )
    profile_last = now
    logger.info("tc_profile_py restore_owned_binding_tensors get_cuda_memory_ptr_start")
    cuda_memory_ptr = get_cuda_memory_ptr(device_id, cuda_ipc_handle)
    now = time.perf_counter()
    logger.info(
        "tc_profile_py restore_owned_binding_tensors get_cuda_memory_ptr_done "
        "ptr=%d step_sec=%.6f total_sec=%.6f",
        int(cuda_memory_ptr),
        now - profile_last,
        now - profile_start,
    )
    profile_last = now
    logger.info("tc_profile_py restore_owned_binding_tensors restore_tensors_start")
    tensors = restore_tensors(
        meta_state_dict,
        {int(device_id): int(cuda_memory_ptr)},
        {int(device_id): tensor_offsets},
        True,
        lease_token=lease_token,
        local_handle_socket_path=local_handle_socket_path,
    )
    now = time.perf_counter()
    logger.info(
        "tc_profile_py restore_owned_binding_tensors restore_tensors_done "
        "tensor_count=%d step_sec=%.6f total_sec=%.6f",
        len(tensors),
        now - profile_last,
        now - profile_start,
    )
    return tensors


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

        policy_mode = CollectivePolicyMode.COLLECTIVE_FIRST

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


class OwnedBindingSlot:
    """Stable, daemon-owned CUDA layout that can be refilled in-place."""

    def __init__(
        self,
        *,
        store: "Store",
        runtime: "StoreRuntimeContext",
        tensors: Mapping[str, torch.Tensor] | None,
        layout: BindingLayout,
        binding_id: str,
        current_value_metadata: BindingValueMetadata | None,
        device: torch.device,
        device_id: int,
        binding_current_value_publication_token: bytes | None,
        staged_value_metadata: BindingValueMetadata | None = None,
        group_realization_acquire: GroupRealizationAcquireRef | None = None,
        binding_current_value_publication_operation_id: str | None = None,
        restore_response: store_daemon_pb2.CreateBindingResponse
        | store_daemon_pb2.CreateOwnedBindingResponse
        | None = None,
        start_restore: bool = False,
        materialization_diagnostics: Mapping[str, object] | None = None,
    ) -> None:
        if not tensors and restore_response is None:
            raise ArtifactError(
                "OwnedBindingSlot requires tensors or a deferred restore response",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._store = store
        self._runtime = runtime
        self._tensors = dict(tensors or {})
        self._tensors_view: Mapping[str, torch.Tensor] = MappingProxyType(self._tensors)
        self._tensors_restored = bool(tensors)
        self._restore_response = restore_response
        self._restore_future: Future[dict[str, torch.Tensor]] | None = None
        self._layout = layout
        self._binding_id = str(binding_id)
        self._binding_layout_id = str(layout.binding_layout_id)
        self._current_value_metadata = current_value_metadata
        self._staged_value_metadata = staged_value_metadata
        self._group_realization_acquire = group_realization_acquire
        self._staged_value_acquired = False
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
        self._binding_current_value_publication_token = (
            bytes(binding_current_value_publication_token)
            if binding_current_value_publication_token
            else None
        )
        self._binding_current_value_publication_operation_id = (
            str(binding_current_value_publication_operation_id)
            if binding_current_value_publication_operation_id
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
        self._last_materialization_diagnostics: Mapping[str, object] | None = (
            None
            if materialization_diagnostics is None
            else dict(materialization_diagnostics)
        )
        if start_restore:
            self.start_tensor_restore()

    @property
    def tensors(self) -> Mapping[str, torch.Tensor]:
        self.ensure_tensors_restored()
        return self._tensors_view

    def start_tensor_restore(self) -> None:
        self._ensure_open()
        if self._tensors_restored or self._restore_future is not None:
            return
        if self._restore_response is None:
            raise ArtifactError(
                "OwnedBindingSlot has no tensors and no deferred restore response",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        logger.info(
            "tc_profile_py owned_binding_slot start_tensor_restore_async "
            "binding_id=%s device_id=%d",
            self._binding_id,
            self._device_id,
        )
        self._restore_future = _owned_binding_restore_executor().submit(
            functools.partial(
                restore_owned_binding_tensors,
                response=self._restore_response,
                runtime=self._runtime,
                device_id=self._device_id,
            )
        )

    def ensure_tensors_restored(self) -> None:
        if self._tensors_restored:
            return
        self._ensure_open()
        if self._restore_response is None and self._restore_future is None:
            raise ArtifactError(
                "OwnedBindingSlot has no tensors and no deferred restore response",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        profile_start = time.perf_counter()
        logger.info(
            "tc_profile_py owned_binding_slot ensure_tensors_restored_start "
            "binding_id=%s future_started=%s",
            self._binding_id,
            self._restore_future is not None,
        )
        if self._restore_future is None:
            tensors = restore_owned_binding_tensors(
                response=cast(
                    store_daemon_pb2.CreateBindingResponse
                    | store_daemon_pb2.CreateOwnedBindingResponse,
                    self._restore_response,
                ),
                runtime=self._runtime,
                device_id=self._device_id,
            )
        else:
            tensors = self._restore_future.result()
        if not tensors:
            raise ArtifactError(
                "OwnedBindingSlot tensor restore returned no tensors",
                status_code="DATA_LOSS",
                retryable=False,
            )
        self._tensors = dict(tensors)
        self._tensors_view = MappingProxyType(self._tensors)
        self._tensors_restored = True
        self._restore_response = None
        self._restore_future = None
        logger.info(
            "tc_profile_py owned_binding_slot ensure_tensors_restored_done "
            "binding_id=%s tensor_count=%d total_sec=%.6f",
            self._binding_id,
            len(self._tensors),
            time.perf_counter() - profile_start,
        )

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
    def staged_value_metadata(self) -> BindingValueMetadata | None:
        return self._staged_value_metadata

    @property
    def group_realization_acquire(self) -> GroupRealizationAcquireRef | None:
        return self._group_realization_acquire

    @property
    def staged_value_acquired(self) -> bool:
        return self._staged_value_acquired

    @property
    def artifact_id(self) -> str | None:
        if self._current_value_metadata is None:
            return (
                None
                if self._staged_value_metadata is None
                else self._staged_value_metadata.source_artifact_id
            )
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
    def last_materialization_diagnostics(self) -> Mapping[str, object] | None:
        if self._last_materialization_diagnostics is None:
            return None
        return dict(self._last_materialization_diagnostics)

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
        self._reject_staged_value_mutation("publish_replica")
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
        if not self._binding_current_value_publication_token:
            raise ArtifactError(
                "binding_current_value_publication_token missing; daemon publish not available",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        timeout_s = _ctx_timeout_s(ctx)
        operation_id = (
            self._binding_current_value_publication_operation_id or uuid.uuid4().hex
        )
        client = self._runtime.ensure_client()
        try:
            resp = client.publish_target_replica(
                binding_current_value_publication_token=self._binding_current_value_publication_token,
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
        self._reject_staged_value_mutation("publish_replica_operation")
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
        if not self._binding_current_value_publication_token:
            raise ArtifactError(
                "binding_current_value_publication_token missing; daemon publish not available",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        timeout_s = _ctx_timeout_s(ctx)
        operation_id = (
            self._binding_current_value_publication_operation_id or uuid.uuid4().hex
        )
        client = self._runtime.ensure_client()
        try:
            start_resp = client.start_publish_target_replica(
                binding_current_value_publication_token=self._binding_current_value_publication_token,
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
        self._reject_staged_value_mutation("begin_update")
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
        self._binding_current_value_publication_token = None
        self._binding_current_value_publication_operation_id = None
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
        self._reject_staged_value_mutation("seal_current")
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
            response.current_value,
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
        self._binding_current_value_publication_token = None
        self._binding_current_value_publication_operation_id = None
        self._dirty = False

    def freeze_current(
        self,
        *,
        update_epoch: "BindingUpdateEpoch | str | int",
        source_artifact_ref: str | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        self._ensure_open()
        self._reject_staged_value_mutation("freeze_current")
        update_epoch_token = self._normalize_update_epoch(update_epoch)
        timeout_s = _ctx_timeout_s(ctx)
        try:
            response = self._runtime.ensure_client().freeze_binding_current_value(
                binding_id=self._binding_id,
                update_epoch=update_epoch_token,
                source_artifact_ref=source_artifact_ref,
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
            response.current_value,
            rpc_name="FreezeBindingCurrentValue",
            expected_binding_id=self._binding_id,
            expected_binding_layout_id=self._binding_layout_id,
        )
        if metadata is None:
            self._enter_dirty_state()
            raise ArtifactError(
                "FreezeBindingCurrentValue returned empty current_value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        self._active_update_epoch = None
        self._seal_generation_counter = int(metadata.seal_generation)
        self._current_value_metadata = metadata
        self._selection = None
        self._binding_current_value_publication_token = None
        self._binding_current_value_publication_operation_id = None
        self._dirty = False

    def promote_current_value(
        self,
        *,
        binding_value_id: str,
        ctx: CallContext | None = None,
    ) -> store_daemon_pb2.PromoteBindingCurrentValueResponse:
        self._ensure_open()
        self._reject_staged_value_mutation("promote_current_value")
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
            response.current_value,
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
        token = bytes(response.binding_current_value_publication_token)
        self._binding_current_value_publication_token = bytes(token) if token else None
        self._binding_current_value_publication_operation_id = None
        self._dirty = False
        self._last_execution_diagnostics = execution_diagnostics_from_response(response)
        self._last_source_bound_plan_diagnostics = None
        return response

    def start_promote_current_value(
        self,
        *,
        binding_value_id: str,
        ctx: CallContext | None = None,
    ) -> store_daemon_pb2.BindingPromotionStatus:
        self._ensure_open()
        self._reject_staged_value_mutation("start_promote_current_value")
        timeout_s = _ctx_timeout_s(ctx)
        try:
            response = (
                self._runtime.ensure_client().start_promote_binding_current_value(
                    binding_id=self._binding_id,
                    binding_value_id=str(binding_value_id),
                    timeout_s=timeout_s if timeout_s is not None else 30.0,
                )
            )
        except Exception as exc:  # noqa: BLE001
            error = _map_slot_error(exc)
            raise ArtifactError(
                str(error),
                status_code=error.status_code,
                retryable=error.retryable,
            ) from exc
        if not response.HasField("status"):
            raise ArtifactError(
                "StartPromoteBindingCurrentValue returned empty status",
                status_code="DATA_LOSS",
                retryable=False,
            )
        metadata = parse_binding_value_or_raise(
            response.status.current_value
            if response.status.HasField("current_value")
            else None,
            rpc_name="StartPromoteBindingCurrentValue",
            expected_binding_id=self._binding_id,
            expected_binding_layout_id=self._binding_layout_id,
        )
        if metadata is not None:
            self._current_value_metadata = metadata
        token = getattr(response.status, "binding_current_value_publication_token", b"")
        if token:
            self._binding_current_value_publication_token = bytes(token)
        return response.status

    def get_promotion_status(
        self,
        *,
        verification_job_id: str | None = None,
        binding_value_id: str,
        ctx: CallContext | None = None,
    ) -> store_daemon_pb2.BindingPromotionStatus:
        self._ensure_open()
        timeout_s = _ctx_timeout_s(ctx)
        try:
            response = self._runtime.ensure_client().get_binding_promotion_status(
                verification_job_id=verification_job_id,
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
        if not response.HasField("status"):
            raise ArtifactError(
                "GetBindingPromotionStatus returned empty status",
                status_code="DATA_LOSS",
                retryable=False,
            )
        metadata = parse_binding_value_or_raise(
            response.status.current_value
            if response.status.HasField("current_value")
            else None,
            rpc_name="GetBindingPromotionStatus",
            expected_binding_id=self._binding_id,
            expected_binding_layout_id=self._binding_layout_id,
        )
        if metadata is not None:
            self._current_value_metadata = metadata
        token = getattr(response.status, "binding_current_value_publication_token", b"")
        if token:
            self._binding_current_value_publication_token = bytes(token)
        return response.status

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
        self._reject_staged_value_mutation("realize_from")
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
        self._last_materialization_diagnostics = None
        rpc_timeout_s = _ctx_timeout_s(ctx)
        try:
            source_selection = (
                None
                if resolved is None
                else resolved._resolve_realization_selection().proto
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
        self._binding_current_value_publication_token = None
        self._binding_current_value_publication_operation_id = None
        self._current_value_metadata = None
        self._last_execution_diagnostics = execution_diagnostics_from_response(response)
        self._last_source_bound_plan_diagnostics = (
            source_bound_plan_diagnostics_from_response(response)
        )
        self._last_materialization_diagnostics = (
            binding_materialization_diagnostics_from_response(
                response,
                layout=self._layout,
            )
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
        runtime_artifact_policy: RuntimeArtifactPolicy | None = None,
        wait: bool = True,
        drain_timeout_s: float | None = None,
        ctx: CallContext | None = None,
        operation_id: str | None = None,
        publish_ttl_ms: int | None = None,
        publish_owner_pid: int | None = None,
    ) -> None:
        self._ensure_open()
        self._reject_staged_value_mutation("swap")
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
        self._last_materialization_diagnostics = None
        rpc_timeout_s = _ctx_timeout_s(ctx)
        try:
            source_selection = resolved._resolve_realization_selection().proto
            response = self._runtime.ensure_client().refill_owned_binding(
                binding_id=self._binding_id,
                artifact_id=resolved._ensure_identified(),
                source_selection=source_selection,
                source_policy=source_policy,
                execution_topology=execution_topology,
                collective_policy=collective_policy,
                runtime_artifact_policy=runtime_artifact_policy,
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
        self._binding_current_value_publication_token = (
            bytes(response.binding_current_value_publication_token)
            if response.binding_current_value_publication_token
            else None
        )
        self._binding_current_value_publication_operation_id = (
            str(operation_id)
            if operation_id and self._binding_current_value_publication_token
            else None
        )
        metadata = parse_binding_value_or_raise(
            response.current_value,
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
        self._last_execution_diagnostics = execution_diagnostics_from_response(response)
        self._last_source_bound_plan_diagnostics = (
            source_bound_plan_diagnostics_from_response(response)
        )
        self._last_materialization_diagnostics = (
            binding_materialization_diagnostics_from_response(
                response,
                layout=self._layout,
            )
        )
        if publish:
            self.publish_replica(
                ttl_ms=publish_ttl_ms,
                owner_pid=publish_owner_pid,
                ctx=ctx,
            )

    def acquire_staged_value(
        self,
        *,
        wait_for_publish: bool = True,
        wait_timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> BindingValueMetadata:
        self._ensure_open()
        if self._staged_value_metadata is None:
            raise ArtifactError(
                "acquire_staged_value() requires a staged group-realization value",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if self._group_realization_acquire is None:
            raise ArtifactError(
                "staged binding is missing group_realization_acquire",
                status_code="DATA_LOSS",
                retryable=False,
            )
        timeout_s = _ctx_timeout_s(ctx)
        if wait_timeout_s is not None:
            timeout_s = float(wait_timeout_s)
        acquire_ref = self._group_realization_acquire.model_copy(
            update={
                "wait_for_publish": bool(wait_for_publish),
                "wait_timeout_ms": max(
                    0,
                    int((timeout_s if timeout_s is not None else 30.0) * 1000.0),
                ),
            }
        )
        value_ref = self._binding_value_ref(self._staged_value_metadata)
        response = self._runtime.ensure_client().acquire_group_staged_binding_value(
            binding_value_ref=value_ref,
            group_realization_acquire=acquire_ref,
            expected_device_uuid=self._device_uuid(),
            caller_pid=None,
            timeout_s=timeout_s if timeout_s is not None else 30.0,
        )
        if not bool(getattr(response, "acquired_staged_value", False)):
            raise ArtifactError(
                "AcquireBindingValue did not return a staged value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        metadata = parse_binding_value_or_raise(
            response.acquired_value if response.HasField("acquired_value") else None,
            rpc_name="AcquireBindingValue",
            expected_binding_id=self._binding_id,
            expected_binding_layout_id=self._binding_layout_id,
        )
        if metadata is None:
            raise ArtifactError(
                "AcquireBindingValue returned empty acquired_value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        if metadata.binding_value_id != self._staged_value_metadata.binding_value_id:
            raise ArtifactError(
                "AcquireBindingValue returned a different staged value",
                status_code="DATA_LOSS",
                retryable=False,
            )
        lease_token = bytes(response.mem_handle.lease_token)
        if lease_token:
            try:
                self._runtime.ensure_client().release_placement_lease(
                    lease_token=lease_token,
                    timeout_s=5.0,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "owned_binding_slot.acquire_staged_value failed to release "
                    "placement lease (binding_id=%s)",
                    self._binding_id,
                )
        self._staged_value_acquired = True
        self._staged_value_metadata = metadata
        return metadata

    def close(self) -> None:
        if self._closed:
            return
        if self._restore_future is not None:
            try:
                self._restore_future.result()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "owned_binding_slot.close observed failed restore future "
                    "(binding_id=%s)",
                    self._binding_id,
                )
        if self._published_lease_id is not None:
            try:
                self.retire(wait=False)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "owned_binding_slot.close failed to retire published binding "
                    "lease (binding_id=%s, lease_id=%s)",
                    self._binding_id,
                    self._published_lease_id,
                )
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

    def _reject_staged_value_mutation(self, op_name: str) -> None:
        if self._staged_value_metadata is None:
            return
        raise ArtifactError(
            f"{op_name}() is disabled for staged group-realization values; "
            "wait/acquire the staged value or close it",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )

    def _device_uuid(self) -> str:
        if self._staged_value_metadata is None and self._current_value_metadata is None:
            return ""
        try:
            from tensorcast.api._device import device_uuid_for

            return device_uuid_for(self._device_id)
        except Exception:
            return ""

    def _binding_value_ref(
        self,
        metadata: BindingValueMetadata,
    ) -> BindingValueRef:
        return BindingValueRef(
            binding_id=metadata.binding_id,
            binding_layout_id=metadata.binding_layout_id,
            binding_value_id=metadata.binding_value_id,
            seal_generation=metadata.seal_generation,
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
        self._binding_current_value_publication_token = None
        self._binding_current_value_publication_operation_id = None
        self._dirty = True


__all__ = ["OwnedBindingSlot", "restore_owned_binding_tensors"]
