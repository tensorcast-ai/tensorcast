#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, replace
from typing import cast

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


def materialize_artifact(
    *,
    client: DaemonCtl,
    daemon_address: str,
    device_id: int | torch.device,
    artifact_id: str | None,
    key: str | None,
    options: GetArtifactOptions | None = None,
) -> MaterializedArtifact:
    if (artifact_id is None and key is None) or (artifact_id and key):
        raise ValueError("Exactly one of artifact_id or key must be provided")

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
    disk_path: str | None = None

    with tracer.start_as_current_span(
        "Client/MaterializeArtifact", kind=SpanKind.INTERNAL
    ):
        if artifact_id is not None:
            handle_bytes = client.materialize_by_artifact_id(
                artifact_id,
                replica_uuid,
                device_uuid_for(dev_id),
                pinned_allocation_timeout_ms=opts.pinned_allocation_timeout_ms,
                wait_for_completion=opts.wait_for_completion,
            )
            resolved_artifact_id = artifact_id
        else:
            mat_result = client.materialize_by_key(
                key or "",
                replica_uuid,
                dev_id,
                pinned_allocation_timeout_ms=opts.pinned_allocation_timeout_ms,
                wait_for_completion=opts.wait_for_completion,
            )
            handle_bytes, disk_path, resolved_artifact_id = cast(
                tuple[bytes, str, str], mat_result
            )

    idx_bytes = client.get_artifact_index_by_id(resolved_artifact_id)
    if not idx_bytes:
        raise IndexParseError(
            f"Failed to fetch canonical index for artifact_id={resolved_artifact_id}"
        )

    try:
        index_obj = json.loads(idx_bytes)
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
        canonical_index_bytes=idx_bytes,
        replica_uuid=replica_uuid,
        disk_path=disk_path,
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
