#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import array
import contextlib
import fcntl
import logging
import os
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from tensorcast._c_ext import (
    get_cuda_memory_ptr,
    restore_tensors,
    restore_tensors_from_cpu_fd_with_lease,
)
from tensorcast.api._config import GetArtifactOptions
from tensorcast.api._device import CPU_DEVICE_ID, device_uuid_for, resolve_device
from tensorcast.api._errors import DaemonUnavailable, IndexParseError
from tensorcast.api._runtime import apply_client_load_defaults_if_present
from tensorcast.api._utils import new_uuid
from tensorcast.api.context import CallContext, CollectiveLoadGroup
from tensorcast.common.selection_contract import (
    build_artifact_selection,
    compute_selected_index_bytes,
)
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.observability.otel import ensure_client_otel
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import ServerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorPayloadDescriptor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    buffer_offset: int
    byte_length: int
    storage_offset: int
    device_uuid: str | None = None


PayloadIterator = Iterator[tuple[TensorPayloadDescriptor, torch.Tensor]]


@dataclass(frozen=True)
class MaterializationPayload:
    artifact_id: str
    canonical_index_bytes: bytes
    descriptors: Sequence[TensorPayloadDescriptor]
    payload_iter: Callable[[], PayloadIterator]
    replica_uuid: str
    generation: int | None = None
    state_dict: dict[str, torch.Tensor] | None = None
    state_dict_loader: Callable[[], dict[str, torch.Tensor]] | None = None
    disk_path: str | None = None
    view_index_bytes: bytes | None = None
    view_data_hash: str | None = None
    source: store_daemon_pb2.MaterializationSource | None = None
    device_uuid: str | None = None
    ticket_replica_uuid: str | None = None
    ticket_status: store_daemon_pb2.MaterializeReplicaStatus | None = None
    ticket_created_at_ts: float | None = None
    ticket_expires_at_ts: float | None = None
    retry_attempts: int = 1
    retry_reason_buckets: Mapping[str, int] | None = None
    budget_deadline_sec: float | None = None
    budget_elapsed_sec: float | None = None
    budget_remaining_sec: float | None = None
    budget_exit_reason: str | None = None
    materialize_timing: dict[str, float] | None = None
    bind_timing: dict[str, float] | None = None


_LOCAL_HANDLE_RESP_LABELS: dict[int, str] = {
    0: "ok",
    1: "not_found",
    2: "failed_precondition",
    3: "permission_denied",
    4: "internal",
}


def _ensure_fd_cloexec(fd: int) -> None:
    flags = fcntl.fcntl(fd, fcntl.F_GETFD)
    fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)


def _request_cpu_memfd_fd(
    *,
    local_handle_socket_path: str,
    lease_token: bytes,
) -> int:
    if not local_handle_socket_path:
        raise DaemonUnavailable("Local handle socket path is required for CPU memfd")
    if not lease_token:
        raise DaemonUnavailable("Daemon returned empty lease_token for CPU memfd")
    if len(lease_token) > 1024:
        raise DaemonUnavailable("lease_token too large for local handle protocol")

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect(local_handle_socket_path)
        except OSError as exc:
            raise DaemonUnavailable(
                f"Failed to connect LocalHandle socket at {local_handle_socket_path}"
            ) from exc
        msg = bytes([1]) + struct.pack("=I", len(lease_token)) + lease_token
        try:
            sock.sendall(msg)
        except OSError as exc:
            raise DaemonUnavailable("LocalHandle GetCpuMemfdFd send failed") from exc

        recv_flags = getattr(socket, "MSG_CMSG_CLOEXEC", 0)
        try:
            data, ancdata, _, _ = sock.recvmsg(
                1, socket.CMSG_SPACE(struct.calcsize("i")), recv_flags
            )
        except (OSError, TimeoutError) as exc:
            raise DaemonUnavailable("LocalHandle GetCpuMemfdFd recv failed") from exc
        if not data:
            raise DaemonUnavailable("Local handle server returned empty response")
        code = int(data[0])
        if code != 0:
            label = _LOCAL_HANDLE_RESP_LABELS.get(code, f"unknown({code})")
            raise DaemonUnavailable(f"LocalHandle GetCpuMemfdFd failed: {label}")

        recv_fds: list[int] = []
        for level, ctype, cmsg_data in ancdata:
            if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
                fds = array.array("i")
                fds.frombytes(cmsg_data)
                recv_fds.extend(int(fd) for fd in fds)

        if not recv_fds:
            raise DaemonUnavailable(
                "LocalHandle GetCpuMemfdFd returned no file descriptor"
            )

        fd = recv_fds[0]
        for extra_fd in recv_fds[1:]:
            with contextlib.suppress(OSError):
                os.close(extra_fd)
        _ensure_fd_cloexec(fd)
        return fd


def _tensor_payload_from_proto(
    proto: store_daemon_pb2.TensorPayloadDescriptor,
    *,
    default_device_uuid: str | None,
) -> TensorPayloadDescriptor:
    device_uuid = proto.device_uuid or default_device_uuid
    return TensorPayloadDescriptor(
        name=str(proto.name),
        dtype=str(proto.dtype),
        shape=tuple(int(dim) for dim in proto.shape),
        stride=tuple(int(dim) for dim in proto.stride),
        buffer_offset=int(proto.buffer_offset),
        byte_length=int(proto.byte_length),
        storage_offset=int(proto.storage_offset),
        device_uuid=str(device_uuid) if device_uuid else None,
    )


def _export_policy_to_proto(value: str | None) -> store_daemon_pb2.ExportPolicy:
    normalized = "never" if value is None else str(value).strip().lower()
    if normalized == "force":
        return store_daemon_pb2.ExportPolicy.EXPORT_POLICY_FORCE
    if normalized == "auto":
        return store_daemon_pb2.ExportPolicy.EXPORT_POLICY_AUTO
    return store_daemon_pb2.ExportPolicy.EXPORT_POLICY_NEVER


def _resolve_collective_load_group(
    ctx: CallContext | None,
) -> store_daemon_pb2.CollectiveLoadGroup | None:
    # Collective disk load must be an explicit API choice. Ambient environment
    # state must not silently rewrite a normal materialization request.
    if ctx is None or ctx.collective is None:
        return None
    collective: CollectiveLoadGroup = ctx.collective
    if not collective.group_id:
        return None
    if collective.world_size <= 1:
        return None
    if collective.rank < 0 or collective.rank >= collective.world_size:
        return None
    group = store_daemon_pb2.CollectiveLoadGroup()
    group.group_id = collective.group_id
    group.world_size = int(collective.world_size)
    group.rank = int(collective.rank)
    return group


def _build_artifact_selection(
    *,
    artifact_id: str,
    view: common_pb2.ViewSpec | None,
    view_id: str | None,
    tensor_names: Sequence[str] | None,
    view_subset_hash: bytes | None,
    canonical_index_hint: bytes | None,
    view_index_hint: bytes | None,
) -> common_pb2.ArtifactSelection:
    ordered_names: tuple[str, ...] = tuple(str(name) for name in (tensor_names or ()))
    has_subset = bool(ordered_names)
    has_transform = bool(view is not None and view.tensors)
    resolved_view_index_hint = bytes(view_index_hint or b"")

    if (has_transform or has_subset) and not resolved_view_index_hint:
        canonical_bytes = bytes(canonical_index_hint or b"")
        if canonical_bytes:
            resolved_view_index_hint = compute_selected_index_bytes(
                canonical_index_bytes=canonical_bytes,
                view_spec=view if has_transform else None,
                tensor_names=ordered_names if has_subset else None,
            )

    return build_artifact_selection(
        artifact_id=artifact_id,
        canonical_index_bytes=bytes(canonical_index_hint or b""),
        layout_index_bytes=resolved_view_index_hint,
        view_spec=view,
        tensor_names=ordered_names,
        view_subset_hash=view_subset_hash,
        view_id=view_id,
        allow_view_id_without_spec=bool(view_id and not has_transform),
    )


def materialize_artifact_v2(
    *,
    client: DaemonCtl,
    daemon_address: str,
    server_config: ServerConfig | None = None,
    device_id: int | torch.device,
    artifact_id: str | None,
    key: str | None,
    options: GetArtifactOptions | None = None,
    view: common_pb2.ViewSpec | None = None,
    view_id: str | None = None,
    placement: store_daemon_pb2.TransformPlacement | None = None,
    canonical_index_hint: bytes | None = None,
    preference: store_daemon_pb2.SourcePreference | None = None,
    source_policy: store_daemon_pb2.SourcePolicy | None = None,
    tensor_names: Sequence[str] | None = None,
    verify_checksums: bool = True,
    view_subset_hash: bytes | None = None,
    replica_uuid: str | None = None,
    view_index_hint: bytes | None = None,
    generation_hint: int | None = None,
    ctx: CallContext | None = None,
    timeout_s: float | None = None,
    lease_mode: store_daemon_pb2.LeaseMode = store_daemon_pb2.LeaseMode.LEASE_MODE_UNSPECIFIED,
) -> MaterializationPayload:
    if artifact_id is not None and key is not None:
        raise ValueError("Exactly one of artifact_id or key must be provided")
    if artifact_id is None and key is None:
        raise ValueError("Either artifact_id or key is required")
    if view is not None and view_id is not None:
        raise ValueError("Specify at most one of view or view_id")

    ensure_client_otel("tensorcast-client", role="client")
    tracer = trace.get_tracer(__name__)

    opts = options or GetArtifactOptions()
    dev_id = resolve_device(device_id, allow_cpu=True)
    target_device_type = (
        store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU
        if dev_id == CPU_DEVICE_ID
        else store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU
    )
    (
        pinned_ms,
        enable_ver,
        wait_for_completion,
        region_backed_mode,
    ) = apply_client_load_defaults_if_present(
        opts.pinned_allocation_timeout_ms,
        opts.enable_verification,
        opts.wait_for_completion,
        opts.region_backed_mode,
        runtime_address=daemon_address,
    )

    opts = opts.model_copy(
        update={
            "pinned_allocation_timeout_ms": pinned_ms,
            "enable_verification": enable_ver,
            "wait_for_completion": wait_for_completion,
            "region_backed_mode": region_backed_mode,
        }
    )

    disk_path: str | None = None
    view_index_bytes: bytes | None = None
    materialized_source: store_daemon_pb2.MaterializationSource | None = None
    materialize_timing: dict[str, float] = {}

    with tracer.start_as_current_span(
        "Client/MaterializeArtifactV2", kind=SpanKind.INTERNAL
    ) as span:
        if ctx is not None:
            if ctx.request_id:
                span.set_attribute("tc.request_id", str(ctx.request_id))
            if ctx.qos:
                span.set_attribute("tc.qos", str(ctx.qos))
            if ctx.idempotency_key:
                span.set_attribute("tc.idempotency_key", str(ctx.idempotency_key))
            if ctx.tags:
                for k, v in ctx.tags.items():
                    span.set_attribute(f"tc.tags.{k}", v)

        effective_timeout_s = timeout_s
        if (
            effective_timeout_s is None
            and ctx is not None
            and ctx.deadline_ms is not None
        ):
            effective_timeout_s = max(0.001, float(ctx.deadline_ms) / 1000.0)

        if preference is not None:
            preference_value = preference
        elif source_policy is not None:
            preference_value = (
                source_policy.preference
                if source_policy.preference
                != store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_UNSPECIFIED
                else store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
            )
        else:
            preference_value = store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        export_policy = _export_policy_to_proto(opts.export_policy)
        resolved_artifact_id = artifact_id
        if resolved_artifact_id is None:
            if key is None:
                raise ValueError(
                    "artifact_id or key is required for materialize_artifact_v2"
                )
            mapping = client.resolve_key_mapping(key)
            if not mapping.artifact_id:
                raise DaemonUnavailable(
                    f"ResolveKeyMapping returned empty artifact_id for key '{key}'"
                )
            resolved_artifact_id = mapping.artifact_id

        resolved_canonical_index_hint = canonical_index_hint
        if resolved_canonical_index_hint is None:
            resolved_canonical_index_hint = client.get_artifact_index_by_id(
                resolved_artifact_id
            )

        selection_start = time.perf_counter()
        selection = _build_artifact_selection(
            artifact_id=resolved_artifact_id,
            view=view,
            view_id=view_id,
            tensor_names=tensor_names,
            view_subset_hash=view_subset_hash,
            canonical_index_hint=resolved_canonical_index_hint,
            view_index_hint=view_index_hint,
        )
        materialize_timing["build_selection_sec"] = (
            time.perf_counter() - selection_start
        )
        replica_uuid_value = replica_uuid or new_uuid()
        collective_load_group = _resolve_collective_load_group(ctx)

        request_device_uuid = (
            ""
            if target_device_type == store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU
            else device_uuid_for(dev_id)
        )
        rpc_start = time.perf_counter()
        response = client.materialize_by_artifact_id_v2(
            selection=selection,
            replica_uuid=replica_uuid_value,
            device_uuid=request_device_uuid,
            pinned_allocation_timeout_ms=opts.pinned_allocation_timeout_ms,
            wait_for_completion=opts.wait_for_completion,
            wait_for_shared_disk_ms=opts.wait_for_shared_disk_ms,
            placement=placement,
            return_response=True,
            preference=preference_value,
            source_policy=source_policy,
            export_policy=export_policy,
            need_view_data_hash=bool(opts.need_view_data_hash),
            target_device_type=target_device_type,
            lease_mode=lease_mode,
            collective_load_group=collective_load_group,
            timeout_s=effective_timeout_s,
            timing_out=materialize_timing,
        )
        materialize_timing["materialize_call_sec"] = time.perf_counter() - rpc_start
        if not isinstance(response, store_daemon_pb2.MaterializeReplicaResponse):
            raise DaemonUnavailable(
                "Daemon returned unexpected response type for materialization v2"
            )
        disk_path = response.disk_path or disk_path
        materialized_source = response.source

    response_decode_start = time.perf_counter()

    ticket_replica_uuid: str | None = None
    ticket_status: store_daemon_pb2.MaterializeReplicaStatus | None = None
    ticket_created_at_ts: float | None = None
    ticket_expires_at_ts: float | None = None
    if response.HasField("ticket"):
        ticket_replica_uuid = response.ticket.replica_uuid or None
        ticket_status = response.ticket.status
        try:
            ticket_created_at_ts = response.ticket.created_at.ToDatetime().timestamp()
        except Exception:  # noqa: BLE001
            ticket_created_at_ts = None
        try:
            if response.ticket.HasField("expires_at"):
                ticket_expires_at_ts = (
                    response.ticket.expires_at.ToDatetime().timestamp()
                )
        except Exception:  # noqa: BLE001
            ticket_expires_at_ts = None

    mem_handle = response.mem_handle if response.HasField("mem_handle") else None
    handle_kind = mem_handle.WhichOneof("handle") if mem_handle is not None else None
    lease_token = bytes(mem_handle.lease_token) if mem_handle is not None else b""
    local_handle_socket_path = (
        server_config.local_handle_socket_path if server_config is not None else ""
    )
    cpu_shared_memory_enabled = bool(
        server_config.cpu_shared_memory_enabled if server_config is not None else True
    )

    cuda_ipc_handle = b""
    cpu_memfd_size_bytes = 0
    cpu_memfd_offset_bytes = 0
    if mem_handle is not None:
        if handle_kind == "cuda_ipc_handle":
            cuda_ipc_handle = bytes(mem_handle.cuda_ipc_handle)
        elif handle_kind == "cpu_memfd":
            cpu_memfd_size_bytes = int(mem_handle.cpu_memfd.size_bytes)
            cpu_memfd_offset_bytes = int(mem_handle.cpu_memfd.offset_bytes)

    if wait_for_completion:
        if mem_handle is None or handle_kind is None:
            raise DaemonUnavailable(
                "Daemon returned empty mem_handle for materialization v2"
            )
        if (
            target_device_type == store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU
            and handle_kind != "cpu_memfd"
        ):
            raise DaemonUnavailable(
                "Daemon returned non-CPU handle for CPU materialization v2"
            )
        if (
            target_device_type == store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU
            and handle_kind != "cuda_ipc_handle"
        ):
            raise DaemonUnavailable(
                "Daemon returned non-GPU handle for GPU materialization v2"
            )
        if handle_kind == "cuda_ipc_handle" and not cuda_ipc_handle:
            raise DaemonUnavailable(
                "Daemon returned empty cuda_ipc_handle for materialization v2"
            )
        if handle_kind == "cpu_memfd":
            if not cpu_shared_memory_enabled:
                raise DaemonUnavailable(
                    "Daemon cpu_shared_memory_enabled is false for CPU materialization v2"
                )
            if not local_handle_socket_path:
                raise DaemonUnavailable(
                    "Daemon local_handle_socket_path is missing for CPU materialization v2"
                )
            if not lease_token:
                raise DaemonUnavailable(
                    "Daemon returned empty lease_token for CPU materialization v2"
                )
            if cpu_memfd_size_bytes <= 0:
                raise DaemonUnavailable(
                    "Daemon returned empty cpu_memfd handle for CPU materialization v2"
                )

    view_data_hash = getattr(response, "view_data_hash", "") or None

    canonical_index_bytes = bytes(response.canonical_index_bytes)
    raw_generation = int(getattr(response, "generation", 0))
    generation_value: int | None = raw_generation if raw_generation != 0 else None
    if generation_value is None and generation_hint is not None:
        generation_value = generation_hint
    if canonical_index_bytes:
        resolved_artifact_id = response.artifact_id or resolved_artifact_id
    else:
        fallback_hint = resolved_canonical_index_hint
        if fallback_hint is not None:
            canonical_index_bytes = fallback_hint
            if generation_value is None and generation_hint is not None:
                generation_value = generation_hint
        else:
            raise IndexParseError(
                f"Failed to fetch canonical index for artifact_id={resolved_artifact_id}"
            )

    if hasattr(response, "view_index_bytes") and response.view_index_bytes:
        view_index_bytes = bytes(response.view_index_bytes)
    elif view_index_hint is not None:
        view_index_bytes = view_index_hint

    # Translate descriptors from proto into SDK form.
    device_uuid: str | None
    try:
        if target_device_type == store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU:
            device_uuid = None
        else:
            device_uuid = device_uuid_for(dev_id)
    except Exception:  # noqa: BLE001
        device_uuid = None
    descriptors = [
        _tensor_payload_from_proto(desc, default_device_uuid=device_uuid)
        for desc in response.payloads
    ]
    materialize_timing["decode_response_sec"] = (
        time.perf_counter() - response_decode_start
    )

    meta_state_dict = {
        desc.name: (
            list(desc.shape),
            list(desc.stride),
            desc.dtype,
            int(desc.storage_offset),
        )
        for desc in descriptors
    }
    tensor_offsets_by_name: Mapping[str, int] = {
        desc.name: int(desc.buffer_offset) for desc in descriptors
    }

    tensors_cache: dict[str, torch.Tensor] | None = None
    bind_timing: dict[str, float] = {}

    def _ensure_tensors_cache() -> dict[str, torch.Tensor]:
        nonlocal tensors_cache
        if not wait_for_completion:
            raise DaemonUnavailable(
                "wait_for_completion=true is required to create tensor views"
            )
        if tensors_cache is None:
            if handle_kind == "cuda_ipc_handle":
                if dev_id == CPU_DEVICE_ID:
                    raise DaemonUnavailable(
                        "GPU materialization returned a CUDA IPC handle but device is CPU"
                    )
                if not cuda_ipc_handle:
                    raise DaemonUnavailable(
                        "Materialization payload is missing a cuda_ipc_handle"
                    )
                if lease_token and not local_handle_socket_path:
                    raise DaemonUnavailable(
                        "lease_token present but local_handle_socket_path is missing"
                    )
                ipc_open_start = time.perf_counter()
                cuda_memory_ptr = get_cuda_memory_ptr(dev_id, cuda_ipc_handle)
                bind_timing["ipc_open_sec"] = time.perf_counter() - ipc_open_start
                tensor_offsets: Mapping[int | torch.device, Mapping[str, int]] = {
                    dev_id: tensor_offsets_by_name
                }
                memory_ptrs: Mapping[int | torch.device, int] = {
                    dev_id: cuda_memory_ptr
                }
                restore_start = time.perf_counter()
                tensors_cache = restore_tensors(
                    meta_state_dict,
                    memory_ptrs,
                    tensor_offsets,
                    True,
                    lease_token=lease_token,
                    local_handle_socket_path=local_handle_socket_path,
                )
                bind_timing["restore_tensors_sec"] = time.perf_counter() - restore_start
            elif handle_kind == "cpu_memfd":
                if not cpu_shared_memory_enabled:
                    raise DaemonUnavailable("cpu_shared_memory_enabled is false")
                if not local_handle_socket_path:
                    raise DaemonUnavailable("local_handle_socket_path is missing")
                if not lease_token:
                    raise DaemonUnavailable("lease_token is missing for CPU memfd")
                fd_request_start = time.perf_counter()
                fd = _request_cpu_memfd_fd(
                    local_handle_socket_path=local_handle_socket_path,
                    lease_token=lease_token,
                )
                bind_timing["cpu_fd_request_sec"] = (
                    time.perf_counter() - fd_request_start
                )
                restore_start = time.perf_counter()
                tensors_cache = restore_tensors_from_cpu_fd_with_lease(
                    meta_state_dict,
                    fd=fd,
                    size_bytes=cpu_memfd_size_bytes,
                    offset_bytes=cpu_memfd_offset_bytes,
                    tensor_device_offsets=tensor_offsets_by_name,
                    lease_token=lease_token,
                    local_handle_socket_path=local_handle_socket_path,
                )
                bind_timing["restore_tensors_sec"] = time.perf_counter() - restore_start
            else:
                raise DaemonUnavailable(
                    "Materialization payload is missing a mem_handle"
                )
        return tensors_cache

    def _iter() -> PayloadIterator:
        cache = _ensure_tensors_cache()
        for desc in descriptors:
            yield desc, cache[desc.name]

    if opts.enable_verification:
        verification_timeout_ms = opts.pinned_allocation_timeout_ms + 30000
        t = threading.Thread(
            target=_monitor_verification,
            args=(
                client,
                artifact_id or key or "",
                replica_uuid_value,
                verification_timeout_ms,
            ),
            daemon=True,
        )
        t.start()

    return MaterializationPayload(
        artifact_id=resolved_artifact_id,
        canonical_index_bytes=canonical_index_bytes,
        descriptors=tuple(descriptors),
        payload_iter=_iter,
        state_dict_loader=_ensure_tensors_cache,
        replica_uuid=replica_uuid_value,
        disk_path=disk_path,
        view_index_bytes=view_index_bytes,
        view_data_hash=view_data_hash,
        source=materialized_source,
        device_uuid=device_uuid,
        ticket_replica_uuid=ticket_replica_uuid,
        ticket_status=ticket_status,
        ticket_created_at_ts=ticket_created_at_ts,
        ticket_expires_at_ts=ticket_expires_at_ts,
        generation=generation_value,
        materialize_timing=materialize_timing,
        bind_timing=bind_timing,
    )


def _monitor_verification(
    ctl: DaemonCtl,
    identifier: str,
    replica: str,
    timeout: int,
) -> None:  # pragma: no cover
    try:
        resp = ctl.wait_artifact_verification(
            artifact_identifier=identifier,
            replica_uuid=replica,
            timeout_ms=timeout,
        )
        if resp is None:
            return
        if (
            resp.status
            == store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_FAILED
        ):
            os._exit(1)
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "MaterializationPayload",
    "TensorPayloadDescriptor",
    "materialize_artifact_v2",
]
