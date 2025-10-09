#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, cast

import torch
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from tensorcast import startup
from tensorcast._C import (
    get_cuda_memory_ptr,
    restore_tensors,
)
from tensorcast.client_runtime import client_defaults, daemon_target_default

if TYPE_CHECKING:  # for static type checkers only
    from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.observability.otel import ensure_client_otel, set_span_attributes
from tensorcast.proto.daemon.v1 import store_daemon_pb2

# Avoid importing Global Store client/protos in API layer
from ._config import (
    DEFAULT_PINNED_TIMEOUT_MS,
    GetArtifactOptions,
)
from ._device import device_uuid_for, resolve_device
from ._errors import DaemonUnavailable, IndexParseError
from ._indices import (
    TensorDataIndex,
    TensorDeviceOffsets,
    TensorMetaIndex,
    calculate_tensor_device_offsets,
    load_tensor_indices_from_dir,
)
from ._io_disk import load_dict_from_disk
from ._utils import ensure_artifact_descriptor_safe, new_uuid


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
    def load(
        self,
    ) -> dict[str, torch.Tensor] | LoadHandle:  # pragma: no cover - protocol
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
        client: DaemonCtl,
        tracer: trace.Tracer,
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
        self.client = client
        self.tracer = tracer
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

    def load(
        self,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], Callable[[], bool]]:
        with self.tracer.start_as_current_span(
            "Client/LoadDictDaemon", kind=SpanKind.INTERNAL
        ):
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
    ) = _apply_client_defaults_if_present(
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
    except Exception:
        pass


def _apply_client_defaults_if_present(
    pinned_allocation_timeout_ms: int,
    enable_verification: bool,
    wait_for_completion: bool,
    *,
    runtime_address: str,
) -> tuple[int, bool, bool]:
    cfg_timeout_ms, cfg_enable_ver, cfg_wait = client_defaults()
    cfg_target = daemon_target_default()
    if cfg_target and cfg_target != runtime_address:
        raise RuntimeError(
            "ClientConfig daemon target does not match initialized daemon address. "
            "Call tensorcast.startup.init() with the desired daemon first."
        )

    if (
        pinned_allocation_timeout_ms == DEFAULT_PINNED_TIMEOUT_MS
        and cfg_timeout_ms is not None
    ):
        pinned_allocation_timeout_ms = int(cfg_timeout_ms)
    if enable_verification is True and cfg_enable_ver is not None:
        enable_verification = bool(cfg_enable_ver)
    if wait_for_completion is True and cfg_wait is not None:
        wait_for_completion = bool(cfg_wait)

    return pinned_allocation_timeout_ms, enable_verification, wait_for_completion


def load_dict_sync(
    *,
    disk_path: str | os.PathLike | None = None,
    device_id: int | torch.device = 0,
    storage_path: str | os.PathLike | None = None,
    enable_verification: bool = True,
    pinned_allocation_timeout_ms: int = DEFAULT_PINNED_TIMEOUT_MS,
) -> dict[str, torch.Tensor]:
    """Synchronously load an artifact directory into GPU memory.

    Always returns a state_dict with fixed type.
    """
    ensure_client_otel("tensorcast-client", role="client")
    tracer = trace.get_tracer(__name__)
    runtime_ctx = startup.require_initialized()
    client = runtime_ctx.client

    # Unified path semantics: user-provided disk_path is the artifact root.

    device_id_int: int = resolve_device(device_id)
    if disk_path is None:
        raise ValueError("disk_path must be provided")

    raw_disk_path = Path(str(disk_path))
    artifact_dir = raw_disk_path
    tensor_meta_index, tensor_data_index = load_tensor_indices_from_dir(artifact_dir)

    tensor_device_offsets, _ = calculate_tensor_device_offsets(
        tensor_data_index, device_id_int
    )

    device_uuid = device_uuid_for(device_id_int)
    replica_uuid = new_uuid()
    # Use torch cuda check
    if not torch.cuda.is_available():
        return load_dict_from_disk(artifact_dir, device_id=device_id_int)

    (
        pinned_allocation_timeout_ms,
        enable_verification,
        _wait_for_completion,
    ) = _apply_client_defaults_if_present(
        pinned_allocation_timeout_ms,
        enable_verification,
        True,
        runtime_address=runtime_ctx.address,
    )
    try:
        loader = DaemonLoader(
            client=client,
            tracer=tracer,
            disk_path=str(disk_path),
            replica_uuid=replica_uuid,
            device_uuid=device_uuid,
            device_id=device_id_int,
            artifact_dir=artifact_dir,
            tensor_meta_index=tensor_meta_index,
            tensor_device_offsets=tensor_device_offsets,
            pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
            wait_for_completion=True,
            enable_verification=enable_verification,
        )
        result = loader.load()
        if isinstance(result, tuple):
            state, _ = result
            return state
        return result
    except RuntimeError as e:
        if "Local StoreDaemon" in str(e) or "not available" in str(e):
            return load_dict_from_disk(artifact_dir, device_id=device_id_int)
        raise DaemonUnavailable(str(e)) from e


def load_dict_async(
    disk_path: str | os.PathLike | None = None,
    device_id: int | torch.device = 0,
    storage_path: str | os.PathLike | None = None,
    enable_verification: bool = True,
    pinned_allocation_timeout_ms: int = DEFAULT_PINNED_TIMEOUT_MS,
) -> LoadHandle:
    ensure_client_otel("tensorcast-client", role="client")
    tracer = trace.get_tracer(__name__)
    runtime_ctx = startup.require_initialized()
    client = runtime_ctx.client

    # Unified path semantics: user-provided disk_path is the artifact root.

    device_id_int: int = resolve_device(device_id)
    if disk_path is None:
        raise ValueError("disk_path must be provided")

    raw_disk_path = Path(str(disk_path))
    artifact_dir = raw_disk_path
    tensor_meta_index, tensor_data_index = load_tensor_indices_from_dir(artifact_dir)
    tensor_device_offsets, _ = calculate_tensor_device_offsets(
        tensor_data_index, device_id_int
    )
    device_uuid = device_uuid_for(device_id_int)
    replica_uuid = new_uuid()
    if not torch.cuda.is_available():
        state = load_dict_from_disk(artifact_dir, device_id=device_id_int)

        def _confirm() -> bool:
            return True

        return LoadHandle(state, _confirm)

    (
        pinned_allocation_timeout_ms,
        enable_verification,
        _,
    ) = _apply_client_defaults_if_present(
        pinned_allocation_timeout_ms,
        enable_verification,
        False,
        runtime_address=runtime_ctx.address,
    )

    loader = DaemonLoader(
        client=client,
        tracer=tracer,
        disk_path=str(disk_path),
        replica_uuid=replica_uuid,
        device_uuid=device_uuid,
        device_id=device_id_int,
        artifact_dir=artifact_dir,
        tensor_meta_index=tensor_meta_index,
        tensor_device_offsets=tensor_device_offsets,
        pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
        wait_for_completion=False,
        enable_verification=enable_verification,
    )
    state, confirm = cast(
        tuple[dict[str, torch.Tensor], Callable[[], bool]], loader.load()
    )
    return LoadHandle(state, confirm)
