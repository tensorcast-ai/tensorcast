#  Copyright (c) 2025, StepCast Team.

import ctypes
import json
import os
import threading
import time
import uuid
from pathlib import Path

import torch

import scstore.proto.store_daemon_pb2 as store_daemon_pb2
from scstore.daemon_ctl import DaemonCtl
from scstore.logger import init_logger

logger = init_logger(__name__)
# Global daemon address configuration
_global_daemon_address = "127.0.0.1:8073"


def resolve_device(device: int | torch.device) -> int:
    """Convert device specification to device ID.

    Args:
        device: Device as int ID or torch.device object

    Returns:
        Device ID as integer

    Raises:
        ValueError: If device is CPU (not supported yet)
    """
    # LEGACY-DYNAMIC: Required for torch.device vs int handling
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


ctypes.CDLL(os.path.join(os.path.dirname(__file__), "lib/libscstore.so"))
from scstore._C import (  # noqa: E402
    generate_model_verification_info,
    get_cuda_memory_ptr,
    get_device_uuid_map,
    restore_tensors,
    restore_tensors_from_model_path,
    save_tensors,
    save_tensors_streaming,
)


def _get_uuid():
    return str(uuid.uuid4())


def _torch_dtype_from_safetensors(dtype: str) -> str:
    """Map safetensors dtype to our torch dtype string used in meta index.

    We intentionally return the canonical torch dtype string expected by the C++ bridge.
    """
    mapping = {
        "F16": "torch.float16",
        "BF16": "torch.bfloat16",
        "F32": "torch.float32",
        "F64": "torch.float64",
        "I8": "torch.int8",
        "I16": "torch.int16",
        "I32": "torch.int32",
        "I64": "torch.int64",
        "U8": "torch.uint8",
        "BOOL": "torch.uint8",  # stored as byte
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")
    return mapping[dtype]


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
    ALIGN: int = 8  # bytes – writer guarantees 64-bit alignment
    seen: dict[tuple[int, int], int] = {}

    for tensor_name, (src_offset, size) in tensor_index.items():
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
    model_dir: os.PathLike | Path,
) -> tuple[
    dict[str, tuple[list[int], list[int], str, int]], dict[str, tuple[int, int]]
]:
    """Parse all .safetensors files in the directory and build unified indices.

    Returns (tensor_meta_index, tensor_data_index).
    """
    model_dir_path = Path(str(model_dir))
    safetensors_files: list[Path] = sorted(model_dir_path.glob("*.safetensors"))
    if not safetensors_files:
        raise ValueError(f"No .safetensors found in {model_dir_path}")

    tensor_meta_index: dict[str, tuple[list[int], list[int], str, int]] = {}
    tensor_data_index: dict[str, tuple[int, int]] = {}
    base_offset = 0
    for st_path in safetensors_files:
        with open(st_path, "rb") as fbin:
            header_len_bytes = fbin.read(8)
            if len(header_len_bytes) != 8:
                raise ValueError(f"Invalid safetensors file: {st_path}")
            header_len = int.from_bytes(
                header_len_bytes, byteorder="little", signed=False
            )
            header_json = fbin.read(header_len)
            if len(header_json) != header_len:
                raise ValueError(f"Truncated safetensors header: {st_path}")
            try:
                header = json.loads(header_json)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Malformed safetensors header in {st_path}: {e}"
                ) from e

        # Compute data_start and data_size
        file_size = st_path.stat().st_size
        data_start = 8 + header_len
        if data_start > file_size:
            raise ValueError(f"Invalid safetensors layout in {st_path}")

        for name, meta in header.items():
            if name == "__metadata__":
                continue
            # Validate tensor name for safety
            if not name or "/" in name or "\\" in name or name.startswith("."):
                raise ValueError(f"Invalid tensor name: {name}")
            if name in tensor_meta_index or name in tensor_data_index:
                raise ValueError(
                    f"Duplicate tensor key across safetensors files: {name}"
                )
            dtype = meta["dtype"]
            shape = meta["shape"]
            begin, end = meta["data_offsets"]
            if end < begin:
                raise ValueError(f"Invalid data_offsets for tensor {name} in {st_path}")
            length = end - begin
            # Row-major stride
            stride: list[int] = []
            if len(shape) == 0:
                stride = []
            else:
                stride = [0] * len(shape)
                acc = 1
                for i in range(len(shape) - 1, -1, -1):
                    stride[i] = acc
                    acc *= int(shape[i])
            tensor_meta_index[name] = (
                list(map(int, shape)),
                stride,
                _torch_dtype_from_safetensors(dtype),
                0,
            )
            tensor_data_index[name] = (base_offset + begin, length)

        # Advance base_offset by payload size of this file
        base_offset += file_size - data_start

    return tensor_meta_index, tensor_data_index


def save_dict(
    state_dict: dict[str, torch.Tensor],
    model_path: str | os.PathLike,
    use_streaming: bool = True,
    streaming_config: dict | None = None,
):
    tensor_names = list(state_dict.keys())
    tensor_data_index = {}
    for name, param in state_dict.items():
        param_storage = param.untyped_storage()
        data_ptr = param_storage.data_ptr()
        size = param_storage.size()  # storage.size is the memory size in bytes
        tensor_data_index[name] = (data_ptr, size)

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    # Choose between streaming or traditional save
    if use_streaming:
        config = streaming_config or {}
        logger.info(f"Using streaming save with config: {config}")
        tensor_offsets = save_tensors_streaming(
            tensor_names, tensor_data_index, model_path, config
        )
    else:
        tensor_offsets = save_tensors(tensor_names, tensor_data_index, model_path)

    # Build a mapping from offset -> max storage size.  This mirrors the
    # pointer-based deduplication performed in the C++ layer where the backing
    # storage is written exactly once using the *largest* slice.
    offset_max_size: dict[int, int] = {}
    for name in tensor_names:
        off = tensor_offsets[name]
        sz = tensor_data_index[name][1]
        offset_max_size[off] = max(offset_max_size.get(off, 0), sz)

    # The new v2 index additionally records each tensor's *storage offset* (in
    # **elements**, not bytes) so that views/slices that start at a non-zero
    # offset inside the underlying storage can be reconstructed correctly on
    # load.  The tuple layout therefore becomes:
    #   (file_offset, storage_size, shape, stride, dtype, storage_offset)
    #
    # Older checkpoints did not include the final element.  The loader keeps
    # backward-compatibility by treating the offset as 0 when the 6th element
    # is absent.

    tensor_index: dict[
        str, tuple[int, int, tuple[int, ...], tuple[int, ...], str, int]
    ] = {}
    for name, param in state_dict.items():
        off = tensor_offsets[name]
        # Use the canonical (max) size for all aliases that map to the same storage.
        sz = offset_max_size[off]
        tensor_index[name] = (
            off,  # File offset inside tensor.data_*
            sz,  # Canonical storage size (bytes)
            tuple(param.shape),
            tuple(param.stride()),
            str(param.dtype),
            int(param.storage_offset()),  # <-- NEW FIELD (elements)
        )

    with open(os.path.join(model_path, "tensor_index.json"), "w") as f:
        json.dump(tensor_index, f)

    # Skip verification generation if no tensors were saved (e.g., empty model)
    if not tensor_names:
        logger.info("No tensors to verify; skipping verification info generation.")
        return

    # Generate and save verification information
    try:
        logger.info("Generating model verification information...")
        start_time = time.time()

        # Generate verification info for saved partitions
        verification_info = generate_model_verification_info(model_path)

        # Save verification info to file
        verification_path = os.path.join(model_path, "verification.json")
        with open(verification_path, "w") as f:
            json.dump(verification_info, f, indent=2)

        duration = time.time() - start_time
        logger.info(
            f"Model verification info generated and saved in {duration:.3f}s to {verification_path}"
        )

    except Exception as e:
        logger.warning(f"Failed to generate model verification info: {e}")


def load_dict(
    model_path: str | os.PathLike | None = None,
    device_id: int | torch.device = 0,
    storage_path: str | os.PathLike | None = None,
    enable_verification: bool = True,
    pinned_allocation_timeout_ms: int = 30000,
    wait_for_completion: bool = True,
):
    """Load a model checkpoint into memory.

    Args:
        model_path: Path to the model checkpoint
        device_id: Target device (int or torch.device)
        storage_path: Base storage path for models
        enable_verification: Whether to enable async model verification
        pinned_allocation_timeout_ms: Timeout for pinned memory allocation
        wait_for_completion: If True (default), wait for model to be fully loaded.
                           If False, return immediately after memory allocation and
                           return a tuple (state_dict, confirm_fn) where confirm_fn
                           must be called to ensure loading is complete.

    Returns:
        If wait_for_completion=True: The loaded state_dict
        If wait_for_completion=False: Tuple of (state_dict, confirm_fn) where
            confirm_fn() -> bool indicates if loading succeeded
    """
    client = DaemonCtl(get_daemon_address())

    if not storage_path:
        storage_path = os.getenv("STORAGE_PATH", "./models")

    # Normalize device_id early to avoid runtime type checks
    device_id_int: int = resolve_device(device_id)

    if model_path is None or storage_path is None:
        raise ValueError("model_path and storage_path must be provided")

    model_dir = Path(str(storage_path)) / str(model_path)

    index_path = model_dir / "tensor_index.json"
    safetensors_files: list[Path] = sorted(model_dir.glob("*.safetensors"))

    # If safetensors present and no tensor_index.json, build indices from safetensors headers
    if safetensors_files and not index_path.exists():
        tensor_meta_index, tensor_data_index = build_indices_from_safetensors(model_dir)
    else:
        with open(index_path, "r") as f:
            tensor_index = json.load(f)

        tensor_meta_index = {}
        tensor_data_index = {}
        for name, meta in tensor_index.items():
            # Legacy checkpoints (<= v1) store a 5-tuple without the storage_offset.
            if len(meta) == 5:
                offset, size, shape, stride, dtype = meta
                storage_offset = 0
            else:
                offset, size, shape, stride, dtype, storage_offset = meta

            tensor_meta_index[name] = (shape, stride, dtype, storage_offset)
            tensor_data_index[name] = (offset, size)

    # tensor_device_offsets: tensor_name to offset on device
    # tensor_copy_chunks: (offset, size, device_offset[device], 0)
    #                       now offset = device_offset[device]
    tensor_device_offsets, tensor_copy_chunks = calculate_tensor_device_offsets(
        tensor_data_index, device_id_int
    )

    device_uuid_map = get_device_uuid_map()
    device_uuid = device_uuid_map[device_id_int]
    replica_uuid = _get_uuid()

    result = client.load_into_gpu(
        str(model_path),
        replica_uuid,
        device_uuid,
        pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
        wait_for_completion=wait_for_completion,
    )

    if wait_for_completion:
        cuda_memory_handle = result
        load_status = None
    else:
        # For async mode, result is (handle, status)
        cuda_memory_handle, load_status = result

    # Map the CUDA IPC handle returned by the daemon to a local device pointer
    cuda_memory_ptr = get_cuda_memory_ptr(device_id_int, cuda_memory_handle)

    # load model state_dict
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

    def _monitor_verification(
        ctl: DaemonCtl,
        identifier: str,
        replica: str,
        timeout: int,
    ) -> None:  # pragma: no cover – simple helper
        try:
            resp = ctl.wait_model_verification(
                model_identifier=identifier,
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
                    "Model verification failed (identifier=%s, replica=%s): %s",
                    identifier,
                    replica,
                    resp.err_msg or "no details",
                )
                # Abort the entire process to avoid serving corrupted model
                os._exit(1)
            elif (
                resp.status
                == store_daemon_pb2.VerificationStatus.VERIFICATION_STATUS_PASSED
            ):
                logger.info(
                    "Model verification passed (identifier=%s, replica=%s)",
                    identifier,
                    replica,
                )
            else:
                # UNKNOWN / IN_PROGRESS after timeout – treat as pass but warn
                logger.warning(
                    "Model verification not complete (status=%s, id=%s, replica=%s): %s",
                    resp.status,
                    identifier,
                    replica,
                    resp.err_msg,
                )
        except Exception as exc:  # noqa: BLE001
            # Any unexpected error – log and continue optimistically
            logger.exception("Verification monitor encountered error: %s", exc)

    if enable_verification:
        verification_timeout_ms = pinned_allocation_timeout_ms + 30000
        t = threading.Thread(
            target=_monitor_verification,
            args=(client, model_path, replica_uuid, verification_timeout_ms),
            daemon=True,
        )

        t.start()

    # Return based on mode
    if wait_for_completion:
        return state_dict
    else:

        def confirm_load() -> bool:
            """Confirm that the async model loading has completed.

            Returns:
                True if loading succeeded, False otherwise
            """
            try:
                # Check the load status first
                if (
                    load_status
                    and load_status
                    == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_FAILED
                ):
                    logger.error(f"Model allocation failed for {model_path}")
                    return False

                # Call ConfirmModel to wait for loading completion
                success = client.confirm_model_loaded(
                    str(model_path),
                    replica_uuid,
                )

                if not success:
                    logger.error(f"Failed to confirm model loading for {model_path}")
                    return False

                logger.info(f"Model {model_path} loading confirmed successfully")
                return True

            except Exception as e:
                logger.exception(f"Error confirming model load: {e}")
                return False

        return state_dict, confirm_load


def load_dict_pure_local(
    model_path: str | os.PathLike, device_id: int | torch.device = 0
):
    # Normalize device_id early to avoid runtime type checks
    device_id_int: int = resolve_device(device_id)

    with open(os.path.join(model_path, "tensor_index.json"), "r") as f:
        tensor_index = json.load(f)

    tensor_meta_index = {}
    tensor_data_index = {}
    for name, meta in tensor_index.items():
        # Legacy checkpoints (<= v1) store a 5-tuple without the storage_offset.
        if len(meta) == 5:
            offset, size, shape, stride, dtype = meta
            storage_offset = 0
        else:
            offset, size, shape, stride, dtype, storage_offset = meta

        tensor_meta_index[name] = (shape, stride, dtype, storage_offset)
        tensor_data_index[name] = (offset, size)

    start = time.time()

    # ------------------------------------------------------------------
    # Pure-local restore expects the *in-memory* layout to match the on-disk
    # byte representation **exactly** because we simply stream every byte of
    # each partition into a contiguous GPU buffer in
    # `restore_tensors_from_model_path`.  Any additional alignment or
    # compaction here would introduce gaps and therefore corrupt the logical
    # view of a tensor.  In distributed/daemon-based paths we still rely on
    # `calculate_tensor_device_offsets` for efficiency, but for the local
    # fast-path we must adopt the identity mapping (dst_offset == src_offset).
    # ------------------------------------------------------------------

    tensor_device_offsets: dict[int, dict[str, int]] = {
        device_id_int: {name: offset for name, (offset, _) in tensor_data_index.items()}
    }

    # The copy-schedule returned from `calculate_tensor_device_offsets` is not
    # required here because the C++ helper streams the entire file verbatim.
    # We therefore skip its computation to avoid the additional 8-byte
    # realignment logic that caused the observed 4-byte data shift on certain
    # filesystems when O_DIRECT is enabled.
    # ------------------------------------------------------------------

    state_dict = restore_tensors_from_model_path(
        tensor_meta_index,
        model_path,
        tensor_device_offsets[device_id_int],  # Use offsets for the single device
    )

    logger.info(f"restore state_dict takes {time.time() - start} seconds")

    # To maintain backwards-compatibility with older call-sites that expect
    # `load_dict_pure_local()` to return a 2-tuple of `(meta, state_dict)` we
    # now return `None` as a placeholder for the deprecated metadata value.
    #
    # Newer callers should simply ignore the first element (e.g. “_ , sd = …”) or
    # switch to the single-value form if the metadata is not required.
    return None, state_dict
