#  Copyright (c) 2025, TensorCast Team.

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, cast

import grpc
import torch
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.logger import init_logger
from tensorcast.observability.otel import ensure_client_otel, set_span_attributes
from tensorcast.proto.daemon.v1 import store_daemon_pb2 as store_daemon_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2 as gs_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2_grpc as gs_pb2_grpc
from tensorcast.types import (
    ArtifactDescriptor,
    CoalescedHandshake,
    CoalescedPlan,
    DVMPPlan,
    Handshake,
    LeasePlan,
    LeaseSegment,
)

logger = init_logger(__name__)
# Global daemon address configuration
_global_daemon_address = "127.0.0.1:8073"
_global_store_address = os.environ.get("TENSORCAST_GLOBAL_STORE", "127.0.0.1:8085")

# ---------------------------------------------------------------------------
# Module-level constants and type aliases to improve readability
# ---------------------------------------------------------------------------

# Alignment used for coalesced layouts (bytes)
DEFAULT_ALIGN: int = 8

# Client defaults
DEFAULT_PINNED_TIMEOUT_MS: int = 30000
DEFAULT_DVMP_CHUNK_SIZE: int = 4 * 1024 * 1024

# Type aliases for index structures
TensorMetaIndex = dict[str, tuple[list[int], list[int], str, int]]
TensorDataIndex = dict[str, tuple[int, int]]
TensorDeviceOffsets = dict[int | torch.device, dict[str, int]]
CopyChunk = tuple[int, int, int, int]  # (src_offset, size, dst_offset, stream_idx)


def resolve_device(device: int | torch.device) -> int:
    """Convert device specification to device ID.

    Args:
        device: Device as int ID or torch.device object

    Returns:
        Device ID as integer

    Raises:
        ValueError: If device is CPU (not supported yet)
    """
    if device is None:
        raise ValueError("device is required")

    if isinstance(device, torch.device):
        if device.type == "cpu":
            raise ValueError("CPU is not supported yet")
        return device.index if device.index is not None else 0
    # Already an int
    return int(device)


def set_daemon_address(address: str) -> None:
    """Set the global daemon address for all torch_util functions."""
    global _global_daemon_address
    _global_daemon_address = address
    logger.debug(f"Set global daemon address to: {address}")


def get_daemon_address() -> str:
    """Get the current global daemon address."""
    return _global_daemon_address


def set_global_store_address(address: str) -> None:
    global _global_store_address
    _global_store_address = address


def get_global_store_address() -> str:
    return _global_store_address


from tensorcast._C import (  # noqa: E402
    build_canonical_index_from_safetensors,
    get_cuda_memory_handle,
    get_cuda_memory_ptr,
    get_device_uuid_map,
    inspect_or_generate_descriptor,
    restore_tensors,
    restore_tensors_from_disk,
    save_model_to_disk,
)


def _get_uuid():
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# RFC-0014 Python SDK helpers
# ---------------------------------------------------------------------------


class RegisterArtifactOptions:
    def __init__(
        self,
        *,
        plan: str = "vram_coalesced",
        p2p_prefer: str = "vram",
        max_inflight_bytes: int = 512 * 1024 * 1024,
        release_on_tensor_commit: bool = True,
        min_tensor_bytes: int = 64 * 1024,
        max_tensor_count: int = 8192,
        lease_bytes_limit: int = 0,
        # RFC-0014 additions
        key: str | None = None,
        disk_path: str | None = None,
    ) -> None:
        self.plan = plan
        self.p2p_prefer = p2p_prefer
        self.max_inflight_bytes = int(max_inflight_bytes)
        self.release_on_tensor_commit = bool(release_on_tensor_commit)
        self.min_tensor_bytes = int(min_tensor_bytes)
        self.max_tensor_count = int(max_tensor_count)
        self.lease_bytes_limit = int(lease_bytes_limit)
        # RFC-0014 additions
        self.key = key
        self.disk_path = disk_path


class GetArtifactOptions:
    def __init__(
        self,
        *,
        prefer: str = "p2p",  # "p2p" or "disk"
        pinned_allocation_timeout_ms: int = DEFAULT_PINNED_TIMEOUT_MS,
        wait_for_completion: bool = True,
        enable_verification: bool = True,
    ) -> None:
        self.prefer = prefer
        self.pinned_allocation_timeout_ms = int(pinned_allocation_timeout_ms)
        self.wait_for_completion = bool(wait_for_completion)
        self.enable_verification = bool(enable_verification)


class RegisteredArtifact:
    def __init__(
        self, registration_id: str, daemon_address: str, *, ttl_ms: int | None = None
    ) -> None:
        self.registration_id = registration_id
        self._addr = daemon_address
        self._ttl_ms = int(ttl_ms) if ttl_ms and ttl_ms > 0 else 0
        self._ka_thread: threading.Thread | None = None
        self._ka_stop = threading.Event()
        self._epoch: int = 0

    def __enter__(self) -> "RegisteredArtifact":
        if self._ttl_ms > 0 and self._ka_thread is None:

            def _keepalive() -> None:
                ctl = DaemonCtl(self._addr)
                interval = max(1.0, self._ttl_ms / 2000.0)
                while not self._ka_stop.wait(interval):
                    try:
                        ctl.keep_alive_registered_artifact(
                            self.registration_id, self._ttl_ms, self._epoch
                        )
                        self._epoch += 1
                    except Exception:  # noqa: BLE001
                        logger.exception("KeepAliveRegisterArtifact failed")
                        continue

            t = threading.Thread(target=_keepalive, daemon=True)
            t.start()
            self._ka_thread = t
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._ka_stop.set()
        if self._ka_thread and self._ka_thread.is_alive():
            self._ka_thread.join(timeout=1.0)

    def commit(self, timeout_s: float = 60.0) -> ArtifactDescriptor:
        ctl = DaemonCtl(self._addr)
        desc = ctl.commit_registered_artifact(self.registration_id, timeout_s=timeout_s)
        self.__exit__(None, None, None)
        return desc

    def abort(self, timeout_s: float = 15.0) -> bool:
        self.__exit__(None, None, None)
        ctl = DaemonCtl(self._addr)
        return ctl.abort_registered_artifact(self.registration_id, timeout_s=timeout_s)

    def revoke(self, reason: str = "", timeout_s: float = 10.0) -> bool:
        self.__exit__(None, None, None)
        ctl = DaemonCtl(self._addr)
        return ctl.revoke_registered_artifact(self.registration_id, reason)


def begin_register_artifact_sdk(
    *,
    device_id: int,
    total_size_bytes: int,
    ttl_ms: int | None,
    tensor_index_data: bytes,
    plan: CoalescedPlan | DVMPPlan | LeasePlan,
    daemon_address: str,
) -> tuple[RegisteredArtifact, Handshake]:
    ctl = DaemonCtl(daemon_address)
    out = ctl.begin_register_artifact(
        device_id=device_id,
        total_size_bytes=total_size_bytes,
        ttl_ms=ttl_ms,
        tensor_index_data=tensor_index_data,
        encoding="json",
        schema_version="v2",
        plan=plan,
        timeout_s=60.0,
    )
    handle = RegisteredArtifact(out.registration_id, daemon_address, ttl_ms=ttl_ms or 0)
    return handle, out.handshake


def _gs_stub(addr: str | None = None) -> gs_pb2_grpc.GlobalStoreServiceStub:
    target = addr or get_global_store_address()
    channel = grpc.insecure_channel(target)
    return gs_pb2_grpc.GlobalStoreServiceStub(channel)


def _validate_disk_index_matches(index_bytes: bytes, disk_path: str) -> None:
    """Validate that disk_path/tensor_index.json canonicalized matches index_bytes.

    Raises ValueError if mismatch or file missing.
    """
    p = Path(disk_path)
    index_file = p / "tensor_index.json"
    if not index_file.exists():
        raise ValueError(f"Disk index not found at {index_file}")
    try:
        disk_raw = index_file.read_bytes()
        disk_obj = json.loads(disk_raw)
        disk_canon = json.dumps(disk_obj, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Failed to parse disk index at {index_file}: {e}") from e
    if disk_canon != index_bytes:
        # Provide short diff hint (length and head)
        raise ValueError(
            "Disk index does not match registration index (canonical JSON mismatch)"
        )


def _upsert_key_mapping_if_needed(
    *,
    key: str | None,
    artifact_id: str,
    disk_path: str | None,
    descriptor: ArtifactDescriptor | None = None,
) -> None:
    if not key:
        return
    try:
        # Prefer daemon-assisted publish; fallback to Global Store direct
        if descriptor is not None:
            ctl = DaemonCtl(get_daemon_address())
            ok = ctl.publish_replica_key(
                key=key, descriptor=descriptor, disk_path=disk_path or ""
            )
            if ok:
                return
        stub = _gs_stub()
        _ = stub.UpsertKeyMapping(
            gs_pb2.UpsertKeyMappingRequest(
                key=key, artifact_id=artifact_id, disk_path=disk_path or ""
            ),
            timeout=5.0,
        )
    except Exception:
        logger.exception("Failed to upsert key mapping to Global Store")


def calculate_tensor_device_offsets(
    tensor_index: dict[str, tuple[int, int]],
    device_id: int | torch.device = 0,
):
    """Generate the device-side offsets for each tensor.

    We iterate over *tensor_index* in declaration order and copy each unique
    storage block into a contiguous buffer on *device_id*.  Duplicate storages
    are detected via the *(offset, size)* pair and share the same destination
    offset.

    Returns
    -------
    tuple
        (tensor_device_offsets, tensor_copy_chunks)

        • *tensor_device_offsets* – ``dict[device_id, dict[tensor_name, int]]``
          mapping every tensor to its destination offset inside the allocated
          CUDA buffer.
        • *tensor_copy_chunks* – ``dict[device_id, list[tuple[int, int, int, int]]]``
          describing the copy schedule used by the C++ backend.  The tuple is
          *(src_offset, size, dst_offset, stream_idx)*.
    """

    tensor_device_offsets: dict[int | torch.device, dict[str, int]] = {device_id: {}}
    tensor_copy_chunks: dict[int | torch.device, list[tuple[int, int, int, int]]] = {
        device_id: []
    }

    current_offset: int = 0
    ALIGN: int = DEFAULT_ALIGN  # writer guarantees 64-bit alignment
    seen: dict[tuple[int, int], int] = {}

    # Enforce deterministic iteration order to stabilize coalesced layout
    for tensor_name in sorted(tensor_index.keys()):
        src_offset, size = tensor_index[tensor_name]
        if (src_offset, size) in seen:
            dst_offset = seen[(src_offset, size)]
        else:
            # Ensure the next unique storage block is aligned to 64-bit
            # boundaries to mirror the layout produced by the C++ checkpoint
            # writer.  Misalignment here results in off-by-one (or worse)
            # errors when restoring tensors because the C++ reader *always*
            # performs the same alignment logic.

            if current_offset % ALIGN:
                current_offset = (current_offset + (ALIGN - 1)) // ALIGN * ALIGN

            dst_offset = current_offset
            seen[(src_offset, size)] = dst_offset
            tensor_copy_chunks[device_id].append((src_offset, size, dst_offset, 0))
            current_offset += size

        tensor_device_offsets[device_id][tensor_name] = dst_offset

    return tensor_device_offsets, tensor_copy_chunks


def build_indices_from_safetensors(
    artifact_dir: os.PathLike | Path,
) -> tuple[TensorMetaIndex, TensorDataIndex]:
    """Build unified indices using C++ canonical safetensors index builder.

    Returns (tensor_meta_index, tensor_data_index).
    """
    artifact_dir = Path(str(artifact_dir))
    # Use C++ helper to build canonical index bytes (strict canonical JSON)
    index_bytes = build_canonical_index_from_safetensors(str(artifact_dir))
    try:
        index_obj = json.loads(index_bytes)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Failed to parse canonical index bytes: {e}") from e

    tensor_meta_index: TensorMetaIndex = {}
    tensor_data_index: TensorDataIndex = {}
    for name, meta in index_obj.items():
        offset, size, shape, stride, dtype, storage_offset = meta
        tensor_meta_index[name] = (
            list(shape),
            list(stride),
            str(dtype),
            int(storage_offset),
        )
        tensor_data_index[name] = (int(offset), int(size))
    return tensor_meta_index, tensor_data_index


# ---------------------------------------------------------------------------
# Internal helpers to reduce duplication and clarify control flow
# ---------------------------------------------------------------------------


def _build_tensor_meta_and_source_indices(
    artifact: dict[str, torch.Tensor],
) -> tuple[TensorMetaIndex, TensorDataIndex]:
    """Build canonical meta and source storage indices from an artifact dict."""
    tensor_meta_index: TensorMetaIndex = {}
    tensor_source_index: TensorDataIndex = {}
    for name, t in artifact.items():
        if not isinstance(name, str) or not name:
            raise ValueError("All artifact keys must be non-empty strings")
        storage = t.untyped_storage()
        data_ptr = int(storage.data_ptr())
        size_bytes = int(storage.size())
        tensor_source_index[name] = (data_ptr, size_bytes)
        tensor_meta_index[name] = (
            list(map(int, t.shape)),
            list(map(int, t.stride())),
            str(t.dtype),
            int(t.storage_offset()),
        )
    return tensor_meta_index, tensor_source_index


def _compute_coalesced_layout(
    tensor_source_index: TensorDataIndex,
    device_id: int,
) -> tuple[TensorDeviceOffsets, list[CopyChunk], int]:
    """Compute coalesced offsets, unique copy chunks and total size."""
    tensor_device_offsets, tensor_copy_chunks = calculate_tensor_device_offsets(
        tensor_source_index, device_id
    )
    unique_chunks = tensor_copy_chunks.get(device_id, [])
    if not unique_chunks:
        raise ValueError("Failed to compute coalesced layout for artifact")
    total_size_bytes = max(dst + sz for _, sz, dst, _ in unique_chunks)
    return tensor_device_offsets, unique_chunks, total_size_bytes


def _build_v2_index_bytes(
    tensor_meta_index: TensorMetaIndex,
    tensor_source_index: TensorDataIndex,
    tensor_device_offsets: TensorDeviceOffsets,
    device_id: int,
) -> bytes:
    """Build canonical v2 index JSON bytes using destination offsets."""
    tensor_index_v2: dict[str, tuple[int, int, list[int], list[int], str, int]] = {}
    for name in sorted(tensor_meta_index.keys()):
        shape, stride, dtype, storage_offset = tensor_meta_index[name]
        _, storage_size = tensor_source_index[name]
        dst_off = int(tensor_device_offsets[device_id][name])
        tensor_index_v2[name] = (
            dst_off,
            int(storage_size),
            list(shape),
            list(stride),
            dtype,
            int(storage_offset),
        )
    return json.dumps(tensor_index_v2, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _compose_artifact_dir(
    raw_disk_path: Path, storage_path: str | os.PathLike | None
) -> Path:
    """Compose the artifact directory from the storage root and relative path.

    Rules:
    - storage_path is None or "": use raw_disk_path as-is
    - otherwise: join storage_path/raw_disk_path
    """
    if storage_path is None:
        return raw_disk_path
    s = str(storage_path)
    if s == "":
        return raw_disk_path
    return Path(s) / raw_disk_path


def _load_tensor_indices(
    artifact_dir: Path,
) -> tuple[
    TensorMetaIndex,
    TensorDataIndex,
]:
    """Load tensor meta and data indices from artifact directory.

    Prefers `tensor_index.json` when present; otherwise builds from safetensors.
    Handles legacy v1 format (5-tuple without storage_offset).
    """
    index_path = artifact_dir / "tensor_index.json"
    safetensors_files: list[Path] = sorted(artifact_dir.glob("*.safetensors"))

    if safetensors_files and not index_path.exists():
        return build_indices_from_safetensors(artifact_dir)

    with open(index_path, "r") as f:
        tensor_index = json.load(f)

    tensor_meta_index: TensorMetaIndex = {}
    tensor_data_index: TensorDataIndex = {}
    for name, meta in tensor_index.items():
        if len(meta) == 5:
            offset, size, shape, stride, dtype = meta
            storage_offset = 0
        else:
            offset, size, shape, stride, dtype, storage_offset = meta

        tensor_meta_index[name] = (shape, stride, dtype, storage_offset)
        tensor_data_index[name] = (offset, size)

    return tensor_meta_index, tensor_data_index


def _apply_client_defaults_if_present(
    pinned_allocation_timeout_ms: int,
    enable_verification: bool,
    wait_for_completion: bool,
) -> tuple[int, bool, bool]:
    """Apply client defaults when available from runtime config.

    Also updates daemon address if a default target is configured.
    """
    try:
        from tensorcast.client_runtime import client_defaults, daemon_target_default

        cfg_timeout_ms, cfg_enable_ver, cfg_wait = client_defaults()
        cfg_target = daemon_target_default()
        if cfg_target:
            from tensorcast.torch_util import set_daemon_address as _set_addr

            _set_addr(cfg_target)
    except Exception:
        cfg_timeout_ms, cfg_enable_ver, cfg_wait = (None, None, None)

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


def _ensure_artifact_descriptor_safe(artifact_dir: Path) -> None:
    """Ensure `artifact_descriptor.json` exists without raising to caller."""
    try:
        _ = inspect_or_generate_descriptor(str(artifact_dir))
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to ensure artifact_descriptor.json: %s", e)


def _load_from_disk_with_tracing(
    tracer: trace.Tracer,
    disk_path: str | os.PathLike,
    device_id_int: int,
    artifact_dir: Path,
    *,
    span_name: str,
    is_fallback: bool = False,
) -> dict[str, torch.Tensor]:
    """Helper for local disk load with span attributes and fallback flag."""
    with tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL):
        set_span_attributes(
            {
                "tc.disk.path": str(disk_path),
                "tc.device.id": int(device_id_int),
                "tc.source": "disk",
                **({"tc.fallback": True} if is_fallback else {}),
            }
        )
        return load_dict_from_disk(
            artifact_dir,
            device_id=device_id_int,
            storage_path="",
        )


def _monitor_verification(
    ctl: DaemonCtl,
    identifier: str,
    replica: str,
    timeout: int,
) -> None:  # pragma: no cover – simple helper
    """Background monitor that checks artifact verification status and exits on failure."""
    try:
        resp = ctl.wait_artifact_verification(
            artifact_identifier=identifier,
            replica_uuid=replica,
            timeout_ms=timeout,
        )

        # If RPC itself failed, proceed optimistically – already logged
        if resp is None:
            return

        if (
            resp.status
            == store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_FAILED
        ):
            logger.fatal(
                "Artifact verification failed (identifier=%s, replica=%s): %s",
                identifier,
                replica,
                resp.err_msg or "no details",
            )
            # Abort the entire process to avoid serving corrupted artifact
            os._exit(1)
        elif (
            resp.status
            == store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_PASSED
        ):
            logger.info(
                "Artifact verification passed (identifier=%s, replica=%s)",
                identifier,
                replica,
            )
        else:
            # UNKNOWN / IN_PROGRESS after timeout – treat as pass but warn
            logger.warning(
                "Artifact verification not complete (status=%s, id=%s, replica=%s): %s",
                resp.status,
                identifier,
                replica,
                resp.err_msg,
            )
    except Exception as exc:  # noqa: BLE001
        # Any unexpected error – log and continue optimistically
        logger.exception("Verification monitor encountered error: %s", exc)


def save_dict(
    state_dict: dict[str, torch.Tensor],
    disk_path: str | os.PathLike,
    use_streaming: bool = True,
    streaming_config: dict | None = None,
) -> dict:
    # Prepare inputs for unified C++ save: tensor_names, tensor_data, meta_state_dict
    tensor_names = sorted(state_dict.keys())
    tensor_data_index: dict[str, tuple[int, int]] = {}
    meta_state_dict: dict[str, tuple[list[int], list[int], str, int]] = {}
    for name, param in state_dict.items():
        storage = param.untyped_storage()
        tensor_data_index[name] = (int(storage.data_ptr()), int(storage.size()))
        meta_state_dict[name] = (
            list(map(int, param.shape)),
            list(map(int, param.stride())),
            str(param.dtype),
            int(param.storage_offset()),
        )

    config = streaming_config or {}
    # Delegate to C++ to write partitions, tensor_index.json, and artifact_descriptor.json
    descriptor = save_model_to_disk(
        tensor_names, tensor_data_index, meta_state_dict, str(disk_path), config
    )
    return dict(descriptor)


def load_dict_from_disk(
    disk_path: str | os.PathLike,
    *,
    device_id: int | torch.device = 0,
    storage_path: str | os.PathLike | None = None,
) -> dict[str, torch.Tensor]:
    """Load a checkpoint directly from disk files, without contacting the daemon.

    - Uses tensor_index.json (or safetensors headers) and partition files under artifact directory.
    - Loads to CUDA device if available; falls back to CPU when CUDA is not available.
    """
    # Resolve artifact directory and indices
    raw_disk_path = Path(str(disk_path))
    artifact_dir = _compose_artifact_dir(raw_disk_path, storage_path)
    tensor_meta_index, tensor_data_index = _load_tensor_indices(artifact_dir)

    device_id_int: int = resolve_device(device_id)
    # Compute coalesced offsets and adapt to the flat mapping API expected by restore_tensors_from_disk
    tensor_device_offsets, _ = calculate_tensor_device_offsets(
        tensor_data_index, device_id_int
    )
    per_tensor_offsets: dict[str, int] = dict(
        tensor_device_offsets.get(device_id_int, {})
    )

    # Choose device: load to CPU (-1) if CUDA unavailable
    target_device_for_local = device_id_int if torch.cuda.is_available() else -1

    state_dict = restore_tensors_from_disk(
        tensor_meta_index,
        str(artifact_dir),
        per_tensor_offsets,
        target_device_for_local,
    )
    return state_dict


def load_dict(
    disk_path: str | os.PathLike | None = None,
    device_id: int | torch.device = 0,
    storage_path: str | os.PathLike | None = None,
    enable_verification: bool = True,
    pinned_allocation_timeout_ms: int = 30000,
    wait_for_completion: bool = True,
):
    """Load a artifact checkpoint into memory.

    Args:
        disk_path: Path to the artifact checkpoint
        device_id: Target device (int or torch.device)
        storage_path: Base storage path for artifacts
        enable_verification: Whether to enable async artifact verification
        pinned_allocation_timeout_ms: Timeout for pinned memory allocation
        wait_for_completion: If True (default), wait for artifact to be fully loaded.
                           If False, return immediately after memory allocation and
                           return a tuple (state_dict, confirm_fn) where confirm_fn
                           must be called to ensure loading is complete.

    Returns:
        If wait_for_completion=True: The loaded state_dict
        If wait_for_completion=False: Tuple of (state_dict, confirm_fn) where
            confirm_fn() -> bool indicates if loading succeeded
    """
    # Ensure client-side OTel is initialized in a library-friendly manner.
    ensure_client_otel("tensorcast-client", role="client")
    tracer = trace.get_tracer(__name__)
    client = DaemonCtl(get_daemon_address())

    # Respect explicit empty string ("") to mean "use disk_path as-is".
    # In unified config mode: storage_path must be provided or present in ClientConfig.
    if storage_path is None:
        try:
            from tensorcast.client_runtime import storage_root_default

            cfg_root = storage_root_default()
        except Exception:
            cfg_root = None
        if cfg_root is None:
            raise ValueError(
                "storage_path must be provided or set via ClientConfig.storage.default_root"
            )
        storage_path = cfg_root

    # Normalize device_id early to avoid runtime type checks
    device_id_int: int = resolve_device(device_id)

    if disk_path is None or storage_path is None:
        raise ValueError("disk_path and storage_path must be provided")

    raw_disk_path = Path(str(disk_path))
    artifact_dir = _compose_artifact_dir(raw_disk_path, storage_path)
    tensor_meta_index, tensor_data_index = _load_tensor_indices(artifact_dir)

    # tensor_device_offsets: tensor_name to offset on device
    # tensor_copy_chunks: (offset, size, device_offset[device], 0)
    #                       now offset = device_offset[device]
    tensor_device_offsets, tensor_copy_chunks = calculate_tensor_device_offsets(
        tensor_data_index, device_id_int
    )

    device_uuid_map = get_device_uuid_map()
    device_uuid = device_uuid_map[device_id_int]
    replica_uuid = _get_uuid()

    # If CUDA is not available, or daemon is unavailable at runtime, fall back to local disk loading.
    if not torch.cuda.is_available():
        return _load_from_disk_with_tracing(
            tracer,
            str(disk_path),
            device_id_int,
            artifact_dir,
            span_name="Client/LoadDictLocal",
            is_fallback=False,
        )

    # Apply ClientConfig defaults when caller used function defaults
    (
        pinned_allocation_timeout_ms,
        enable_verification,
        wait_for_completion,
    ) = _apply_client_defaults_if_present(
        pinned_allocation_timeout_ms, enable_verification, wait_for_completion
    )

    with tracer.start_as_current_span("Client/LoadDictDaemon", kind=SpanKind.INTERNAL):
        set_span_attributes(
            {
                "tc.disk.path": str(disk_path),
                "tc.device.id": int(device_id_int),
                "tc.source": "daemon",
                "tc.pinned_allocation_timeout_ms": int(pinned_allocation_timeout_ms),
                "tc.wait_for_completion": bool(wait_for_completion),
            }
        )
        try:
            result = client.load_into_gpu(
                str(disk_path),
                replica_uuid,
                device_uuid,
                pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
                wait_for_completion=wait_for_completion,
            )
        except RuntimeError as e:
            # Gracefully degrade to local load when daemon is not running
            if "Local StoreDaemon" in str(e) or "not available" in str(e):
                return _load_from_disk_with_tracing(
                    tracer,
                    str(disk_path),
                    device_id_int,
                    artifact_dir,
                    span_name="Client/LoadDictLocalFallback",
                    is_fallback=True,
                )
            raise

    if wait_for_completion:
        cuda_memory_handle = result
        load_status = None
    else:
        # For async mode, result is (handle, status)
        cuda_memory_handle, load_status = result

    # Map the CUDA IPC handle returned by the daemon to a local device pointer
    cuda_memory_ptr = get_cuda_memory_ptr(device_id_int, cuda_memory_handle)

    # load artifact state_dict
    start = time.time()
    state_dict = restore_tensors(
        tensor_meta_index,
        {device_id_int: cuda_memory_ptr},
        tensor_device_offsets,
        True,  # from_ipc_shm
    )
    logger.info(f"restore state_dict takes {time.time() - start} seconds")
    # ------------------------------------------------------------------
    # Asynchronous integrity verification – handled by StoreDaemon.  We
    # optimistically return the state_dict immediately and monitor the
    # verification result in a background daemon thread.  Any failure is
    # treated as fatal and will terminate the process.
    # ------------------------------------------------------------------

    if enable_verification:
        verification_timeout_ms = pinned_allocation_timeout_ms + 30000
        t = threading.Thread(
            target=_monitor_verification,
            args=(client, disk_path, replica_uuid, verification_timeout_ms),
            daemon=True,
        )

        t.start()

    # Per RFC-0007: after load completes, ensure artifact_descriptor.json exists.
    _ensure_artifact_descriptor_safe(artifact_dir)

    # Return based on mode
    if wait_for_completion:
        return state_dict
    else:

        def confirm_load() -> bool:
            """Confirm that the async artifact loading has completed.

            Returns:
                True if loading succeeded, False otherwise
            """
            try:
                # Check the load status first
                if (
                    load_status
                    and load_status
                    == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_FAILED
                ):
                    logger.error(f"Artifact allocation failed for {disk_path}")
                    return False

                # Call ConfirmReplica to wait for loading completion
                success = client.confirm_replica_loaded(
                    str(disk_path),
                    replica_uuid,
                )

                if not success:
                    logger.error(f"Failed to confirm artifact loading for {disk_path}")
                    return False

                logger.info(f"Artifact {disk_path} loading confirmed successfully")
                return True

            except Exception as e:
                logger.exception(f"Error confirming artifact load: {e}")
                return False

        return state_dict, confirm_load


def register_artifact(
    artifact: dict[str, torch.Tensor],
    *,
    options: RegisterArtifactOptions,
    device_id: int | torch.device | None = None,
    ttl_ms: int | None = None,
    daemon_address: str | None = None,
) -> tuple[dict[str, torch.Tensor], ArtifactDescriptor]:
    """Register an in-memory tensor dict per RFC-0014.

    Returns (state_dict, descriptor_dict). For non-coalesced plans, state_dict is the input mapping.
    """
    if not artifact:
        raise ValueError("artifact must not be empty")

    # Resolve and validate device per spec:
    # a) device_id is None: infer from artifact → all tensors must be CUDA and on the same device
    # b) device_id is provided: artifact must be CPU tensors only
    if device_id is None:
        # Require all tensors to be CUDA and on the same device
        dev_index: int | None = None
        for t in artifact.values():
            if not t.is_cuda:
                raise ValueError(
                    "When device_id is None, all tensors must be CUDA tensors on the same device"
                )
            if dev_index is None:
                dev_index = t.device.index if t.device.index is not None else 0
            else:
                if (t.device.index if t.device.index is not None else 0) != dev_index:
                    raise ValueError(
                        "All CUDA tensors must be on the same device when inferring device_id"
                    )
        if dev_index is None:
            raise ValueError("artifact is empty or has no tensors to infer device from")
        target_device_id = int(dev_index)
        input_mode = "cuda"
    else:
        target_device_id = resolve_device(device_id)
        # Enforce CPU-only input in this mode
        for t in artifact.values():
            if t.is_cuda:
                raise ValueError(
                    "When device_id is specified, artifact must contain CPU tensors only"
                )
        input_mode = "cpu"

    # Build canonical meta index and source storage info
    tensor_meta_index, tensor_source_index = _build_tensor_meta_and_source_indices(
        artifact
    )

    # Plan coalesced layout (8B aligned) and compute total size
    tensor_device_offsets, unique_chunks, total_size_bytes = _compute_coalesced_layout(
        tensor_source_index, target_device_id
    )

    # Build v2-equivalent index JSON using destination offsets
    index_bytes = _build_v2_index_bytes(
        tensor_meta_index, tensor_source_index, tensor_device_offsets, target_device_id
    )

    # RFC-0014: Validate existing disk_path metadata when provided (strict equality)
    if options.disk_path is not None and options.disk_path.strip() != "":
        cand = Path(options.disk_path)
        if cand.exists() and cand.is_dir():
            _validate_disk_index_matches(index_bytes, str(cand))

    ensure_client_otel("tensorcast-client", role="client")
    tracer = trace.get_tracer(__name__)
    ctl = DaemonCtl(daemon_address or get_daemon_address())

    # Build a canonical index bytes and a Begin request per plan
    kind = options.plan
    if kind == "vram_coalesced":
        with tracer.start_as_current_span(
            "Client/RegisterArtifact.Coalesced", kind=SpanKind.INTERNAL
        ):
            plan_model = CoalescedPlan(
                kind="coalesced",
                max_inflight_bytes=options.max_inflight_bytes,
                release_on_tensor_commit=options.release_on_tensor_commit,
            )
            begin = ctl.begin_register_artifact(
                device_id=target_device_id,
                total_size_bytes=total_size_bytes,
                ttl_ms=ttl_ms if ttl_ms else 0,
                tensor_index_data=index_bytes,
                encoding="json",
                schema_version="v2",
                plan=plan_model,
            )
            if not isinstance(begin.handshake, CoalescedHandshake):
                raise RuntimeError("Unexpected handshake type for coalesced plan")
            cuda_handle = begin.handshake.daemon_ipc_handle
            base_ptr = get_cuda_memory_ptr(target_device_id, cuda_handle)
            dest_state_dict = restore_tensors(
                tensor_meta_index,
                {target_device_id: int(base_ptr)},
                tensor_device_offsets,
                True,
            )
            for name, src in artifact.items():
                dst = dest_state_dict[name]
                local = src
                if input_mode == "cpu":
                    local = local.to(
                        torch.device("cuda", target_device_id), non_blocking=True
                    )
                else:
                    if local.device.index != target_device_id:
                        raise ValueError(
                            f"Tensor '{name}' device mismatch: expected cuda:{target_device_id}, got {local.device}"
                        )
                if local.dtype != dst.dtype:
                    local = local.to(dst.dtype)
                if tuple(local.shape) != tuple(dst.shape):
                    raise ValueError(
                        f"Shape mismatch for tensor '{name}': {tuple(local.shape)} vs {tuple(dst.shape)}"
                    )
                dst.copy_(local, non_blocking=True)
            torch.cuda.synchronize(target_device_id)
            desc = ctl.commit_registered_artifact(begin.registration_id, timeout_s=60.0)
            # RFC-0014: persist+publish when disk_path == "", else publish with provided path
            if options.disk_path is not None and options.disk_path.strip() == "":
                try:
                    from tensorcast.client_runtime import storage_root_default

                    root = storage_root_default()
                    if not root:
                        raise RuntimeError(
                            "ClientConfig.storage.default_root is required when disk_path==''"
                        )
                    out_dir = Path(root) / desc.artifact_id
                    save_dict(dest_state_dict, str(out_dir))
                    _upsert_key_mapping_if_needed(
                        key=options.key,
                        artifact_id=desc.artifact_id,
                        disk_path=str(out_dir),
                        descriptor=desc,
                    )
                except Exception:
                    logger.exception("Persist+publish failed for key mapping")
            else:
                _upsert_key_mapping_if_needed(
                    key=options.key,
                    artifact_id=desc.artifact_id,
                    disk_path=options.disk_path,
                    descriptor=desc,
                )
            return dest_state_dict, desc

    elif kind == "dvmp":
        # DVMP: upload bytes into daemon-owned CPU buffer via FeedRegisterArtifactStream
        with tracer.start_as_current_span(
            "Client/RegisterArtifact.DVMP", kind=SpanKind.INTERNAL
        ):
            # Begin
            plan_model = DVMPPlan(kind="dvmp", preferred_channel=2, ring_bytes=0)
            begin = ctl.begin_register_artifact(
                device_id=target_device_id,
                total_size_bytes=total_size_bytes,
                ttl_ms=ttl_ms if ttl_ms else 0,
                tensor_index_data=index_bytes,
                encoding="json",
                schema_version="v2",
                plan=plan_model,
            )
            # Stream in daemon-preferred chunk size without assembling a giant buffer
            # Build generator that yields contiguous slices mapped to their destination offsets
            frames: list[tuple[int, bytes]] = []
            # Reconstruct offsets and sizes from our inputs to avoid keeping a second dict
            for name in sorted(tensor_meta_index.keys()):
                dst_off = int(tensor_device_offsets[target_device_id][name])
                _ptr, storage_size = tensor_source_index[name]
                src = artifact[name]
                b = src.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                if len(b) != int(storage_size):
                    raise ValueError(
                        f"Tensor '{name}' raw byte size mismatch: {len(b)} vs {storage_size}"
                    )
                frames.append((int(dst_off), b))

            # Concatenate frames into one bytes is suboptimal for memory; stream each frame with proper offset
            ok_all = True
            # Determine chunk size from daemon
            cfg = ctl.get_server_config()
            chunk_size = int(cfg.chunk_size)
            if chunk_size <= 0:
                chunk_size = DEFAULT_DVMP_CHUNK_SIZE

            # Send each frame, possibly split by chunk_size
            for off, data_bytes in frames:
                ok = ctl.feed_register_artifact_dvmp_stream_data(
                    begin.registration_id,
                    data_bytes,
                    offset=int(off),
                    chunk_size=chunk_size,
                    timeout_s=60.0,
                )
                if not ok:
                    ok_all = False
                    break
            ok = ok_all
            if not ok:
                raise RuntimeError("DVMP feed failed")
            desc = ctl.commit_registered_artifact(begin.registration_id, timeout_s=60.0)
            if options.disk_path is not None and options.disk_path.strip() == "":
                try:
                    from tensorcast.client_runtime import storage_root_default

                    root = storage_root_default()
                    if not root:
                        raise RuntimeError(
                            "ClientConfig.storage.default_root is required when disk_path==''"
                        )
                    out_dir = Path(root) / desc.artifact_id
                    save_dict(artifact, str(out_dir))
                    _upsert_key_mapping_if_needed(
                        key=options.key,
                        artifact_id=desc.artifact_id,
                        disk_path=str(out_dir),
                        descriptor=desc,
                    )
                except Exception:
                    logger.exception("Persist+publish failed for key mapping")
            else:
                _upsert_key_mapping_if_needed(
                    key=options.key,
                    artifact_id=desc.artifact_id,
                    disk_path=options.disk_path,
                    descriptor=desc,
                )
            return artifact, desc

    elif kind == "vram_leased":
        # VRAM Lease: export CUDA IPC handles for unique storages and feed as LeaseSegments
        with tracer.start_as_current_span(
            "Client/RegisterArtifact.Lease", kind=SpanKind.INTERNAL
        ):
            if input_mode != "cuda":
                raise ValueError(
                    "vram_leased plan requires CUDA tensors (device_id must be inferred)"
                )
            # Build unique storage chunks. Sorting by destination offset keeps logs
            # stable, but the daemon no longer requires ordering (dst_offset is explicit).
            # Reuse the unique chunks computed earlier.
            chunks = list(unique_chunks)  # (src_offset, size, dst_offset, stream_idx)
            chunks.sort(key=lambda x: int(x[2]))

            # Export CUDA IPC handle via pybind (core implementation)
            def _export_cuda_ipc_handle(ptr: int) -> bytes:
                # Returns raw cudaIpcMemHandle_t bytes
                return get_cuda_memory_handle(target_device_id, int(ptr))

            segments: list[LeaseSegment] = []
            for src_off, size_bytes, _dst_off, _stream in chunks:
                handle_bytes = _export_cuda_ipc_handle(int(src_off))
                segments.append(
                    LeaseSegment(
                        device_id=int(target_device_id),
                        cuda_ipc_handle=handle_bytes,
                        base_addr=0,
                        length=int(size_bytes),
                        dst_offset=int(_dst_off),
                    )
                )
            plan_model = LeasePlan(
                kind="lease",
                min_tensor_bytes=options.min_tensor_bytes,
                max_tensor_count=options.max_tensor_count,
                lease_bytes_limit=options.lease_bytes_limit,
            )
            begin = ctl.begin_register_artifact(
                device_id=target_device_id,
                total_size_bytes=total_size_bytes,
                ttl_ms=ttl_ms if ttl_ms else 0,
                tensor_index_data=index_bytes,
                encoding="json",
                schema_version="v2",
                plan=plan_model,
            )
            ok = ctl.feed_register_artifact_lease_segments(
                begin.registration_id, segments
            )
            if not ok:
                raise RuntimeError("Lease segments feed failed")
            desc = ctl.commit_registered_artifact(begin.registration_id, timeout_s=60.0)
            if options.disk_path is not None and options.disk_path.strip() == "":
                try:
                    from tensorcast.client_runtime import storage_root_default

                    root = storage_root_default()
                    if not root:
                        raise RuntimeError(
                            "ClientConfig.storage.default_root is required when disk_path==''"
                        )
                    out_dir = Path(root) / desc.artifact_id
                    save_dict(artifact, str(out_dir))
                    _upsert_key_mapping_if_needed(
                        key=options.key,
                        artifact_id=desc.artifact_id,
                        disk_path=str(out_dir),
                        descriptor=desc,
                    )
                except Exception:
                    logger.exception("Persist+publish failed for key mapping")
            else:
                _upsert_key_mapping_if_needed(
                    key=options.key,
                    artifact_id=desc.artifact_id,
                    disk_path=options.disk_path,
                    descriptor=desc,
                )
            return artifact, desc

    else:
        raise ValueError(f"Unknown plan: {kind}")


def get_artifact(
    *,
    key: str,
    device_id: int | torch.device = 0,
    options: GetArtifactOptions | None = None,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], Callable[[], bool]]:
    """Fetch an artifact by RFC-0014 key using the daemon's MaterializeByKey.

    Behavior:
    - prefer == 'p2p' (default): call daemon MaterializeByKey; reconstruct from
      GPU memory using indices loaded from `used_disk_path` returned by daemon; on
      daemon failure, optionally fall back to disk by resolving key mapping.
    - prefer == 'disk': resolve key mapping and load from disk.

    Returns a state_dict; in async mode returns (state_dict, confirm_fn).
    """
    if not key or not isinstance(key, str):
        raise ValueError("key must be a non-empty string")

    opts = options or GetArtifactOptions()
    ensure_client_otel("tensorcast-client", role="client")
    tracer = trace.get_tracer(__name__)

    # Helper: Resolve key mapping for disk fallback
    def _resolve_mapping():
        stub = _gs_stub()
        return stub.ResolveKeyMapping(
            gs_pb2.ResolveKeyMappingRequest(key=key), timeout=5.0
        )

    dev_id = resolve_device(device_id)
    # device_uuid unused with MaterializeByKey path
    replica_uuid = _get_uuid()

    # Disk-preferred path
    if opts.prefer == "disk":
        res = _resolve_mapping()
        if res.status != gs_pb2.Status.STATUS_OK or not res.disk_path:
            raise RuntimeError("Key resolved without disk_path; disk load not possible")
        return load_dict(
            disk_path=res.disk_path,
            device_id=dev_id,
            storage_path=None,
            enable_verification=opts.enable_verification,
            pinned_allocation_timeout_ms=opts.pinned_allocation_timeout_ms,
            wait_for_completion=opts.wait_for_completion,
        )

    # Prefer P2P via daemon MaterializeByKey
    client = DaemonCtl(get_daemon_address())
    (
        pinned_ms,
        enable_ver,
        wait_comp,
    ) = _apply_client_defaults_if_present(
        opts.pinned_allocation_timeout_ms,
        opts.enable_verification,
        opts.wait_for_completion,
    )
    if pinned_ms is not None:
        opts.pinned_allocation_timeout_ms = pinned_ms
    if enable_ver is not None:
        opts.enable_verification = enable_ver
    if wait_comp is not None:
        opts.wait_for_completion = wait_comp

    try:
        with tracer.start_as_current_span(
            "Client/GetArtifactByKey.P2P", kind=SpanKind.INTERNAL
        ):
            if opts.wait_for_completion:
                _res_sync = client.materialize_by_key(
                    key,
                    replica_uuid,
                    dev_id,
                    pinned_allocation_timeout_ms=opts.pinned_allocation_timeout_ms,
                    wait_for_completion=True,
                )
                handle_bytes, _used_disk_path, artifact_id = cast(
                    tuple[bytes, str, str], _res_sync
                )
                load_status = None
            else:
                _res_async = client.materialize_by_key(
                    key,
                    replica_uuid,
                    dev_id,
                    pinned_allocation_timeout_ms=opts.pinned_allocation_timeout_ms,
                    wait_for_completion=False,
                )
                handle_bytes, load_status, _used_disk_path, artifact_id = cast(
                    tuple[bytes, object, str, str], _res_async
                )

        # Reconstruct indices via Global Store by artifact_id
        gs = _gs_stub()
        idx_resp = gs.GetArtifactIndexById(
            gs_pb2.GetArtifactIndexByIdRequest(artifact_id=artifact_id), timeout=5.0
        )
        if idx_resp.status != gs_pb2.Status.STATUS_OK or not idx_resp.tensor_index_data:
            raise RuntimeError(
                f"Failed to fetch canonical index for artifact_id={artifact_id}"
            )
        # Parse canonical index JSON -> meta/data indices
        try:
            index_obj = json.loads(idx_resp.tensor_index_data)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Invalid canonical index JSON from Global Store") from e
        tensor_meta_index: TensorMetaIndex = {}
        tensor_data_index: TensorDataIndex = {}
        for name, meta in index_obj.items():
            if len(meta) == 5:
                offset, size, shape, stride, dtype = meta
                storage_offset = 0
            else:
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

        # Optional background verification (daemon-side)
        if opts.enable_verification:
            verification_timeout_ms = opts.pinned_allocation_timeout_ms + 30000
            t = threading.Thread(
                target=_monitor_verification,
                args=(client, artifact_id, replica_uuid, verification_timeout_ms),
                daemon=True,
            )
            t.start()

        if opts.wait_for_completion:
            return state_dict

        def confirm_load() -> bool:
            try:
                if (
                    load_status
                    and load_status
                    == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_FAILED
                ):
                    logger.error(f"Artifact allocation failed (key) for key={key}")
                    return False
                return client.confirm_replica_loaded("", replica_uuid)
            except Exception:
                logger.exception("Error confirming key-based artifact load")
                return False

        return state_dict, confirm_load

    except Exception:  # noqa: BLE001
        # No compatibility fallback: rely on daemon orchestrator; propagate error
        raise
