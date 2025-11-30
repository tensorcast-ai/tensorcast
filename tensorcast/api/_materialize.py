#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, replace

import torch
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from tensorcast._C import get_cuda_memory_ptr, restore_tensors
from tensorcast.api._config import GetArtifactOptions
from tensorcast.api._device import device_uuid_for, resolve_device
from tensorcast.api._errors import DaemonUnavailable, IndexParseError
from tensorcast.api._indices import (
    TensorDataIndex,
    TensorMetaIndex,
    calculate_tensor_device_offsets,
)
from tensorcast.api._runtime import apply_client_load_defaults_if_present
from tensorcast.api._utils import new_uuid
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.observability.otel import ensure_client_otel
from tensorcast.proto.daemon.v1 import store_daemon_pb2


@dataclass(frozen=True)
class MaterializedArtifact:
    artifact_id: str
    state_dict: dict[str, torch.Tensor]
    canonical_index_bytes: bytes
    replica_uuid: str
    disk_path: str | None = None
    view_index_bytes: bytes | None = None
    view_data_hash: str | None = None
    source: store_daemon_pb2.MaterializationSource | None = None


def materialize_artifact(
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
) -> MaterializedArtifact:
    if artifact_id is not None and key is not None:
        raise ValueError("Exactly one of artifact_id or key must be provided")
    if artifact_id is None and key is None and not disk_path_hint:
        raise ValueError("Either artifact_id, key, or disk_path_hint is required")
    if view is not None and view_id is not None:
        raise ValueError("Specify at most one of view or view_id")
    if artifact_id is None and (view is not None or view_id is not None):
        raise ValueError("artifact_id is required when requesting a view")

    ensure_client_otel("tensorcast-client", role="client")
    tracer = trace.get_tracer(__name__)

    opts = options or GetArtifactOptions()
    if not opts.wait_for_completion:
        raise DaemonUnavailable(
            "wait_for_completion=False is not supported for materialize_artifact"
        )

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

    opts = replace(
        opts,
        pinned_allocation_timeout_ms=pinned_ms,
        enable_verification=enable_ver,
        wait_for_completion=wait_for_completion,
    )

    replica_uuid = new_uuid()
    disk_path: str | None = disk_path_hint
    view_index_bytes: bytes | None = None
    view_data_hash: str | None = None
    materialized_source: store_daemon_pb2.MaterializationSource | None = None

    with tracer.start_as_current_span(
        "Client/MaterializeArtifact", kind=SpanKind.INTERNAL
    ):
        preference_value = (
            preference
            if preference is not None
            else store_daemon_pb2.SourcePreference.SOURCE_PREFERENCE_AUTO
        )
        if artifact_id is not None:
            if view is not None or view_id is not None:
                response = client.materialize_by_artifact_id(
                    artifact_id,
                    replica_uuid,
                    device_uuid_for(dev_id),
                    pinned_allocation_timeout_ms=opts.pinned_allocation_timeout_ms,
                    wait_for_completion=opts.wait_for_completion,
                    view=view,
                    view_id=view_id,
                    placement=placement,
                    return_response=True,
                    disk_path=disk_path_hint,
                    preference=preference_value,
                )
                if not isinstance(
                    response, store_daemon_pb2.MaterializeReplicaResponse
                ):
                    raise DaemonUnavailable(
                        "Daemon returned unexpected response type for view materialization"
                    )
                disk_path = response.disk_path or disk_path
                if (
                    response.mem_handle is None
                    or not response.mem_handle.cuda_ipc_handle
                ):
                    raise DaemonUnavailable(
                        "Daemon returned empty mem_handle for view materialization"
                    )
                handle_bytes = response.mem_handle.cuda_ipc_handle
                if response.view_index_json:
                    view_index_bytes = bytes(response.view_index_json)
                if response.view_data_hash:
                    view_data_hash = str(response.view_data_hash)
                materialized_source = response.source
                resolved_artifact_id = response.artifact_id or artifact_id
            else:
                response = client.materialize_by_artifact_id(
                    artifact_id,
                    replica_uuid,
                    device_uuid_for(dev_id),
                    pinned_allocation_timeout_ms=opts.pinned_allocation_timeout_ms,
                    wait_for_completion=opts.wait_for_completion,
                    return_response=True,
                    disk_path=disk_path_hint,
                    preference=preference_value,
                )
                if not isinstance(
                    response, store_daemon_pb2.MaterializeReplicaResponse
                ):
                    raise DaemonUnavailable(
                        "Daemon returned unexpected response type for materialization"
                    )
                handle_bytes = response.mem_handle.cuda_ipc_handle
                disk_path = response.disk_path or disk_path
                materialized_source = response.source
                resolved_artifact_id = response.artifact_id or artifact_id
                if response.view_index_json:
                    view_index_bytes = bytes(response.view_index_json)
        elif key is not None:
            response = client.materialize_by_key(
                key or "",
                replica_uuid,
                dev_id,
                pinned_allocation_timeout_ms=opts.pinned_allocation_timeout_ms,
                wait_for_completion=opts.wait_for_completion,
                return_response=True,
            )
            handle_bytes = response.mem_handle.cuda_ipc_handle
            disk_path = response.used_disk_path
            resolved_artifact_id = response.artifact_id
            materialized_source = response.source
        else:
            response = client.materialize_by_artifact_id(
                artifact_id or "",
                replica_uuid,
                device_uuid_for(dev_id),
                pinned_allocation_timeout_ms=opts.pinned_allocation_timeout_ms,
                wait_for_completion=opts.wait_for_completion,
                return_response=True,
                disk_path=disk_path_hint,
                preference=preference_value,
            )
            if not isinstance(response, store_daemon_pb2.MaterializeReplicaResponse):
                raise DaemonUnavailable(
                    "Daemon returned unexpected response type for disk materialization"
                )
            handle_bytes = response.mem_handle.cuda_ipc_handle
            disk_path = response.disk_path or disk_path_hint
            materialized_source = response.source
            resolved_artifact_id = (
                response.artifact_id or artifact_id or (disk_path or "")
            )
            if response.view_index_json:
                view_index_bytes = bytes(response.view_index_json)

    if not resolved_artifact_id:
        raise IndexParseError(
            "Daemon did not provide artifact_id for materialized artifact"
        )

    canonical_index_bytes = canonical_index_hint
    if canonical_index_bytes is None and view_index_bytes is not None:
        canonical_index_bytes = view_index_bytes
    if canonical_index_bytes is None:
        try:
            canonical_index_bytes = client.get_artifact_index_by_id(
                resolved_artifact_id
            )
        except RuntimeError as exc:
            # Provide clearer error when daemon could not supply index and GS is unavailable.
            raise IndexParseError(
                f"Failed to fetch canonical index for artifact_id={resolved_artifact_id}: {exc}"
            ) from exc
    if not canonical_index_bytes:
        raise IndexParseError(
            f"Failed to fetch canonical index for artifact_id={resolved_artifact_id}"
        )

    try:
        layout_index_bytes = view_index_bytes or canonical_index_bytes
        index_obj = json.loads(layout_index_bytes)
    except Exception as exc:  # noqa: BLE001
        raise IndexParseError("Invalid canonical index JSON from Global Store") from exc

    tensor_meta_index: TensorMetaIndex = {}
    tensor_data_index: TensorDataIndex = {}
    for name, meta in index_obj.items():
        if len(meta) != 6:
            raise IndexParseError(
                f"Invalid canonical index entry for '{name}': expected 6 fields [offset,size,shape,stride,dtype,storage_offset], got {len(meta)}"
            )
        offset, size, shape, stride, dtype, storage_offset = meta
        tensor_meta_index[name] = (shape, stride, dtype, storage_offset)
        tensor_data_index[name] = (int(offset), int(size))

    tensor_device_offsets, _ = calculate_tensor_device_offsets(
        tensor_data_index, dev_id
    )

    cuda_memory_ptr = get_cuda_memory_ptr(dev_id, handle_bytes)
    state_dict = restore_tensors(
        tensor_meta_index,
        {dev_id: cuda_memory_ptr},
        tensor_device_offsets,
        True,
    )

    if opts.enable_verification:
        verification_timeout_ms = opts.pinned_allocation_timeout_ms + 30000
        t = threading.Thread(
            target=_monitor_verification,
            args=(
                client,
                resolved_artifact_id,
                replica_uuid,
                verification_timeout_ms,
            ),
            daemon=True,
        )
        t.start()

    return MaterializedArtifact(
        artifact_id=resolved_artifact_id,
        state_dict=state_dict,
        canonical_index_bytes=canonical_index_bytes,
        replica_uuid=replica_uuid,
        disk_path=disk_path,
        view_index_bytes=view_index_bytes,
        view_data_hash=view_data_hash,
        source=materialized_source,
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


__all__ = ["MaterializedArtifact", "materialize_artifact"]
