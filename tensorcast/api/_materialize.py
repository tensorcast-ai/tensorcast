#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from tensorcast._c_ext import get_cuda_memory_ptr, restore_tensors
from tensorcast.api._config import GetArtifactOptions
from tensorcast.api._device import device_uuid_for, resolve_device
from tensorcast.api._errors import DaemonUnavailable, IndexParseError
from tensorcast.api._runtime import apply_client_load_defaults_if_present
from tensorcast.api._utils import new_uuid
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.observability.otel import ensure_client_otel
from tensorcast.proto.daemon.v1 import store_daemon_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2 as store_daemon_v2_pb2


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
    disk_path: str | None = None
    view_index_bytes: bytes | None = None
    view_data_hash: str | None = None
    source: store_daemon_pb2.MaterializationSource | None = None
    device_uuid: str | None = None
    ticket_replica_uuid: str | None = None
    ticket_status: store_daemon_pb2.MaterializeReplicaStatus | None = None
    ticket_created_at_ts: float | None = None
    ticket_expires_at_ts: float | None = None


def _tensor_payload_from_proto(
    proto: store_daemon_v2_pb2.TensorPayloadDescriptor,
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


def materialize_artifact_v2(
    *,
    client: DaemonCtl,
    daemon_address: str,
    device_id: int | torch.device,
    artifact_id: str | None,
    key: str | None,
    options: GetArtifactOptions | None = None,
    view: store_daemon_pb2.ViewSpec | None = None,
    view_id: str | None = None,
    placement: store_daemon_pb2.TransformPlacement | None = None,
    canonical_index_hint: bytes | None = None,
    disk_path_hint: str | None = None,
    preference: store_daemon_pb2.SourcePreference | None = None,
    tensor_names: Sequence[str] | None = None,
    verify_checksums: bool = True,
    view_subset_hash: bytes | None = None,
    replica_uuid: str | None = None,
    view_index_hint: bytes | None = None,
    generation_hint: int | None = None,
) -> MaterializationPayload:
    if artifact_id is not None and key is not None:
        raise ValueError("Exactly one of artifact_id or key must be provided")
    if artifact_id is None and key is None and not disk_path_hint:
        raise ValueError("Either artifact_id, key, or disk_path_hint is required")
    if view is not None and view_id is not None:
        raise ValueError("Specify at most one of view or view_id")

    ensure_client_otel("tensorcast-client", role="client")
    tracer = trace.get_tracer(__name__)

    opts = options or GetArtifactOptions()
    dev_id = resolve_device(device_id)
    (
        pinned_ms,
        enable_ver,
        wait_for_completion,
    ) = apply_client_load_defaults_if_present(
        opts.pinned_allocation_timeout_ms,
        opts.enable_verification,
        opts.wait_for_completion,
        runtime_address=daemon_address,
    )

    opts = opts.model_copy(
        update={
            "pinned_allocation_timeout_ms": pinned_ms,
            "enable_verification": enable_ver,
            "wait_for_completion": wait_for_completion,
        }
    )

    replica_uuid_value = replica_uuid or new_uuid()
    disk_path: str | None = disk_path_hint
    view_index_bytes: bytes | None = None
    materialized_source: store_daemon_pb2.MaterializationSource | None = None

    with tracer.start_as_current_span(
        "Client/MaterializeArtifactV2", kind=SpanKind.INTERNAL
    ):
        preference_value = (
            preference
            if preference is not None
            else store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        )
        response: (
            store_daemon_v2_pb2.MaterializeReplicaResponse
            | store_daemon_v2_pb2.MaterializeByKeyResponse
        )
        if artifact_id is not None:
            response = client.materialize_by_artifact_id_v2(
                artifact_id=artifact_id,
                replica_uuid=replica_uuid_value,
                device_uuid=device_uuid_for(dev_id),
                pinned_allocation_timeout_ms=opts.pinned_allocation_timeout_ms,
                wait_for_completion=opts.wait_for_completion,
                view=view,
                view_id=view_id,
                placement=placement,
                return_response=True,
                disk_path=disk_path_hint,
                preference=preference_value,
                tensor_names=tensor_names,
                verify_checksums=verify_checksums,
                view_subset_hash=view_subset_hash,
            )
            if not isinstance(response, store_daemon_v2_pb2.MaterializeReplicaResponse):
                raise DaemonUnavailable(
                    "Daemon returned unexpected response type for materialization v2"
                )
            disk_path = response.disk_path or disk_path
            materialized_source = response.source
        elif key is not None:
            response = client.materialize_by_key_v2(
                key=key or "",
                replica_uuid=replica_uuid_value,
                device_id=int(dev_id),
                pinned_allocation_timeout_ms=opts.pinned_allocation_timeout_ms,
                wait_for_completion=opts.wait_for_completion,
                return_response=True,
                tensor_names=tensor_names,
                view_subset_hash=view_subset_hash,
            )
            if not isinstance(response, store_daemon_v2_pb2.MaterializeByKeyResponse):
                raise DaemonUnavailable(
                    "Daemon returned unexpected response type for key materialization v2"
                )
            disk_path = response.used_disk_path or disk_path
            materialized_source = response.source
        elif disk_path_hint:
            # Disk-only materialization: no artifact_id or key, just a disk path.
            # The daemon loads directly from disk via DiskFallbackHint.
            response = client.materialize_by_artifact_id_v2(
                artifact_id="",
                replica_uuid=replica_uuid_value,
                device_uuid=device_uuid_for(dev_id),
                pinned_allocation_timeout_ms=opts.pinned_allocation_timeout_ms,
                wait_for_completion=opts.wait_for_completion,
                view=view,
                view_id=view_id,
                placement=placement,
                return_response=True,
                disk_path=disk_path_hint,
                preference=store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_PREFER_DISK,
                tensor_names=tensor_names,
                verify_checksums=verify_checksums,
                view_subset_hash=view_subset_hash,
            )
            if not isinstance(response, store_daemon_v2_pb2.MaterializeReplicaResponse):
                raise DaemonUnavailable(
                    "Daemon returned unexpected response type for disk materialization v2"
                )
            disk_path = response.disk_path or disk_path
            materialized_source = response.source
        else:
            raise ValueError(
                "artifact_id, key, or disk_path_hint is required for materialize_artifact_v2"
            )

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

    handle_bytes = b""
    if response.HasField("mem_handle"):
        handle_bytes = response.mem_handle.cuda_ipc_handle
    if wait_for_completion and not handle_bytes:
        raise DaemonUnavailable(
            "Daemon returned empty mem_handle for materialization v2"
        )

    view_data_hash: str | None = None
    try:
        subset_hash_bytes = (
            bytes(response.view_subset.subset_hash)
            if hasattr(response, "view_subset")
            else b""
        )
        if subset_hash_bytes:
            view_data_hash = subset_hash_bytes.hex()
    except Exception:  # noqa: BLE001
        view_data_hash = None
    if view_data_hash is None and view_subset_hash:
        view_data_hash = view_subset_hash.hex()

    canonical_index_bytes = bytes(response.canonical_index_bytes)
    raw_generation = int(getattr(response, "generation", 0))
    generation_value: int | None = raw_generation if raw_generation != 0 else None
    if generation_value is None and generation_hint is not None:
        generation_value = generation_hint
    if canonical_index_bytes:
        resolved_artifact_id = response.artifact_id or artifact_id or key or ""
    else:
        fallback_hint = canonical_index_hint or view_index_hint
        if fallback_hint is not None:
            canonical_index_bytes = fallback_hint
            resolved_artifact_id = artifact_id or key or ""
            if generation_value is None and generation_hint is not None:
                generation_value = generation_hint
        else:
            raise IndexParseError(
                f"Failed to fetch canonical index for artifact_id={artifact_id or key or ''}"
            )

    if hasattr(response, "view_index_bytes") and response.view_index_bytes:
        view_index_bytes = bytes(response.view_index_bytes)
    elif view_index_hint is not None:
        view_index_bytes = view_index_hint

    # Translate descriptors from proto into SDK form.
    device_uuid: str | None
    try:
        resolved_device = resolve_device(device_id)
        device_uuid = device_uuid_for(resolved_device)
    except Exception:  # noqa: BLE001
        device_uuid = None
    descriptors = [
        _tensor_payload_from_proto(desc, default_device_uuid=device_uuid)
        for desc in response.payloads
    ]
    cuda_memory_ptr = (
        get_cuda_memory_ptr(dev_id, handle_bytes) if handle_bytes else None
    )

    def _iter() -> PayloadIterator:
        if cuda_memory_ptr is None:
            raise DaemonUnavailable(
                "Materialization payload is missing a mem_handle; tensor data is not available"
            )
        for desc in descriptors:
            meta_state_dict = {
                desc.name: (
                    list(desc.shape),
                    list(desc.stride),
                    desc.dtype,
                    int(desc.storage_offset),
                )
            }
            tensor_offsets: Mapping[int | torch.device, Mapping[str, int]] = {
                dev_id: {desc.name: int(desc.buffer_offset)}
            }
            memory_ptrs: Mapping[int | torch.device, int] = {dev_id: cuda_memory_ptr}
            tensors = restore_tensors(
                meta_state_dict,
                memory_ptrs,
                tensor_offsets,
                True,
            )
            yield desc, tensors[desc.name]

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
