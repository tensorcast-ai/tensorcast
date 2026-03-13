#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import contextlib
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

import torch

from tensorcast._c_ext import get_cuda_memory_ptr, restore_tensors
from tensorcast.api._device import device_uuid_for
from tensorcast.api._materialize import _tensor_payload_from_proto
from tensorcast.api.context import CallContext
from tensorcast.api.store.inplace_slot import (
    _ctx_timeout_s,
    _map_slot_error,
    _normalize_view_id,
    _selection_publishable,
)
from tensorcast.api.store.materialization import _build_source_policy
from tensorcast.api.store.retry import map_materialization_error
from tensorcast.api.store.types import ArtifactError, FallbackOptions
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2

if TYPE_CHECKING:
    from tensorcast.api.store import Store
    from tensorcast.api.store.artifact import Artifact
    from tensorcast.api.store.runtime import StoreRuntimeContext


def restore_owned_binding_tensors(
    *,
    response: store_daemon_pb2.CreateOwnedBindingResponse,
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


class OwnedBindingSlot:
    """Stable, daemon-owned CUDA layout that can be refilled in-place."""

    def __init__(
        self,
        *,
        store: "Store",
        runtime: "StoreRuntimeContext",
        tensors: Mapping[str, torch.Tensor],
        binding_id: str,
        selection: common_pb2.ArtifactSelection,
        device: torch.device,
        device_id: int,
        fallback: FallbackOptions | None,
        target_publication_token: bytes | None,
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
        self._binding_id = str(binding_id)
        self._selection = common_pb2.ArtifactSelection()
        self._selection.CopyFrom(selection)
        self._device = device
        self._device_id = int(device_id)
        self._fallback = fallback
        self._target_publication_token = (
            bytes(target_publication_token) if target_publication_token else None
        )
        self._published_lease_id: str | None = None
        self._published_replica_id: str | None = None
        self._dirty = False
        self._closed = False

    @property
    def tensors(self) -> Mapping[str, torch.Tensor]:
        return self._tensors_view

    @property
    def artifact_id(self) -> str:
        return str(self._selection.artifact_id)

    @property
    def selection(self) -> common_pb2.ArtifactSelection:
        selection = common_pb2.ArtifactSelection()
        selection.CopyFrom(self._selection)
        return selection

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
    def byte_space(self) -> common_pb2.ByteSpaceRef:
        space = common_pb2.ByteSpaceRef()
        if self._selection.view_id:
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
        client = self._runtime.ensure_client()
        try:
            resp = client.publish_target_replica(
                target_publication_token=self._target_publication_token,
                byte_space=self.byte_space,
                ttl_ms=ttl_ms,
                owner_pid=owner_pid,
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
                artifact_id=self.artifact_id,
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

    def swap(
        self,
        artifact: "Artifact | str",
        *,
        publish: bool = False,
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
        preference, source_policy = self._resolve_source_policy(resolved._fallback)
        rpc_timeout_s = _ctx_timeout_s(ctx)
        try:
            response = self._runtime.ensure_client().refill_owned_binding(
                binding_id=self._binding_id,
                artifact_id=resolved._ensure_identified(),
                preference=preference,
                source_policy=source_policy,
                operation_id=operation_id,
                timeout_s=rpc_timeout_s if rpc_timeout_s is not None else 600.0,
            )
        except Exception as exc:  # noqa: BLE001
            self._dirty = True
            error = map_materialization_error(exc)
            raise ArtifactError(
                str(error),
                status_code=error.status_code,
                retryable=False,
            ) from exc
        self._dirty = False
        self._selection.CopyFrom(response.resolved_selection)
        self._target_publication_token = (
            bytes(response.target_publication_token)
            if getattr(response, "target_publication_token", b"")
            else None
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

    def _resolve_source_policy(
        self, fallback: FallbackOptions | None
    ) -> tuple[
        store_daemon_pb2.SourcePreference,
        store_daemon_pb2.SourcePolicy,
    ]:
        preference = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        effective_prefer = fallback.prefer if fallback is not None else "auto"
        if fallback is not None:
            if fallback.prefer == "p2p":
                preference = (
                    store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_P2P
                )
            elif fallback.prefer == "disk":
                preference = (
                    store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK
                )
        allow_p2p = True if fallback is None else bool(fallback.allow_p2p)
        if effective_prefer == "local":
            allow_p2p = False
        allow_disk = True if fallback is None else bool(fallback.allow_disk)
        if effective_prefer == "local":
            allow_disk = False
        source_policy = _build_source_policy(
            preference=preference,
            allow_p2p=allow_p2p,
            allow_disk=allow_disk,
        )
        return preference, source_policy


__all__ = ["OwnedBindingSlot", "restore_owned_binding_tensors"]
