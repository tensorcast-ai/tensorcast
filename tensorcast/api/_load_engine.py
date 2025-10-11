#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable, Protocol, cast

import torch
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from tensorcast._C import get_cuda_memory_ptr, restore_tensors
from tensorcast.api._errors import DaemonUnavailable
from tensorcast.api._indices import (
    TensorDeviceOffsets,
    TensorMetaIndex,
    calculate_tensor_device_offsets,
    load_tensor_indices_from_dir,
)
from tensorcast.api._io_disk import load_dict_from_disk
from tensorcast.api._runtime_handle import RuntimeHandle
from tensorcast.api._utils import ensure_artifact_descriptor_safe
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.observability.otel import ensure_client_otel, set_span_attributes
from tensorcast.proto.daemon.v1 import store_daemon_pb2

if torch.__version__ is None:  # pragma: no cover
    raise RuntimeError("torch must be importable before using tensorcast.api loaders")


class LoadHandle:
    class _StateDictProxy(dict[str, torch.Tensor]):
        def __init__(self, inner: dict[str, torch.Tensor], ready_evt: threading.Event):
            super().__init__()
            self._inner = inner
            self._ready = ready_evt

        def _ensure_ready(self) -> None:
            if not self._ready.is_set():
                raise DaemonUnavailable(
                    "state_dict not ready; call wait() or result() before tensor access"
                )

        def __getitem__(self, k: str) -> torch.Tensor:
            self._ensure_ready()
            return self._inner[k]

        def items(self):
            self._ensure_ready()
            return self._inner.items()

        def keys(self):
            self._ensure_ready()
            return self._inner.keys()

        def values(self):
            self._ensure_ready()
            return self._inner.values()

    def __init__(
        self, state_dict: dict[str, torch.Tensor], confirm_fn: Callable[[], bool]
    ):
        self._state = state_dict
        self._confirm = confirm_fn
        self._ready = threading.Event()
        self._proxy = LoadHandle._StateDictProxy(state_dict, self._ready)

    def ready(self) -> bool:
        return self._ready.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        ok = self._confirm()
        if ok:
            self._ready.set()
        return ok

    def result(self) -> dict[str, torch.Tensor]:
        if not self.ready():
            ok = self.wait()
            if not ok:
                raise DaemonUnavailable("Artifact loading failed during confirmation")
        return self._state

    @property
    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._proxy


class Loader(Protocol):
    def load(self) -> dict[str, torch.Tensor] | LoadHandle:  # pragma: no cover - prot
        ...


class DiskLoader:
    def __init__(self, *, artifact_dir: Path, device_id: int) -> None:
        self._artifact_dir = artifact_dir
        self._device_id = device_id

    def load(self) -> dict[str, torch.Tensor]:
        return load_dict_from_disk(self._artifact_dir, device_id=self._device_id)


class DaemonLoader:
    def __init__(
        self,
        *,
        runtime: RuntimeHandle,
        disk_path: str | os.PathLike,
        replica_uuid: str,
        device_uuid: str,
        device_id: int,
        artifact_dir: Path,
        tensor_meta_index: TensorMetaIndex,
        tensor_device_offsets: TensorDeviceOffsets,
        pinned_allocation_timeout_ms: int,
        wait_for_completion: bool,
        enable_verification: bool,
    ) -> None:
        self.runtime = runtime
        self.disk_path = str(disk_path)
        self.replica_uuid = replica_uuid
        self.device_uuid = device_uuid
        self.device_id = int(device_id)
        self.artifact_dir = artifact_dir
        self.tensor_meta_index = tensor_meta_index
        self.tensor_device_offsets = tensor_device_offsets
        self.pinned_allocation_timeout_ms = int(pinned_allocation_timeout_ms)
        self.wait_for_completion = bool(wait_for_completion)
        self.enable_verification = bool(enable_verification)
        self.tracer = trace.get_tracer(__name__)

    @property
    def client(self) -> DaemonCtl:
        return self.runtime.client

    def load(
        self,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], Callable[[], bool]]:
        with self.tracer.start_as_current_span(
            "Client/LoadDictDaemon", kind=SpanKind.INTERNAL
        ):
            ensure_client_otel("tensorcast-client", role="client")
            set_span_attributes(
                {
                    "tc.disk.path": str(self.disk_path),
                    "tc.device.id": int(self.device_id),
                    "tc.source": "daemon",
                    "tc.pinned_allocation_timeout_ms": int(
                        self.pinned_allocation_timeout_ms
                    ),
                    "tc.wait_for_completion": bool(self.wait_for_completion),
                }
            )
            result = self.client.load_into_gpu(
                str(self.disk_path),
                self.replica_uuid,
                self.device_uuid,
                pinned_allocation_timeout_ms=self.pinned_allocation_timeout_ms,
                wait_for_completion=self.wait_for_completion,
            )

        if self.wait_for_completion:
            cuda_memory_handle = result
            load_status = None
        else:
            cuda_memory_handle, load_status = cast(tuple[bytes, object], result)

        cuda_memory_ptr = get_cuda_memory_ptr(self.device_id, cuda_memory_handle)
        state_dict = restore_tensors(
            self.tensor_meta_index,
            {self.device_id: cuda_memory_ptr},
            self.tensor_device_offsets,
            True,
        )

        if self.enable_verification:
            verification_timeout_ms = self.pinned_allocation_timeout_ms + 30000
            t = threading.Thread(
                target=_monitor_verification,
                args=(
                    self.client,
                    self.disk_path,
                    self.replica_uuid,
                    verification_timeout_ms,
                ),
                daemon=True,
            )
            t.start()

        ensure_artifact_descriptor_safe(self.artifact_dir)

        if self.wait_for_completion:
            return state_dict

        def confirm_load() -> bool:
            try:
                if (
                    load_status
                    and load_status
                    == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_FAILED
                ):
                    return False
                return self.client.confirm_replica_loaded(
                    str(self.disk_path),
                    self.replica_uuid,
                )
            except Exception:
                return False

        return state_dict, confirm_load


def prepare_artifact_layout(
    artifact_dir: Path, *, device_id: int
) -> tuple[TensorMetaIndex, TensorDeviceOffsets]:
    tensor_meta_index, tensor_data_index = load_tensor_indices_from_dir(artifact_dir)
    tensor_device_offsets, _ = calculate_tensor_device_offsets(
        tensor_data_index, device_id
    )
    return tensor_meta_index, tensor_device_offsets


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
    except Exception:
        pass


__all__ = [
    "DaemonLoader",
    "DiskLoader",
    "LoadHandle",
    "prepare_artifact_layout",
]
