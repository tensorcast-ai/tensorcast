#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import threading
import weakref
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from tensorcast._c_ext import (
    compute_view_registration_plan,
    get_cuda_memory_handle_with_offset,
    get_cuda_memory_ptr,
    restore_tensors,
)
from tensorcast.api import _region_cache as region_cache
from tensorcast.api._tensor_graph import TensorStorageGraph, build_tensor_storage_graph

if TYPE_CHECKING:  # for static type checkers only
    from tensorcast.daemon_ctl import DaemonCtl

from tensorcast.api._config import PlanType, RegisterArtifactOptions, StorePolicy
from tensorcast.api._device import resolve_device
from tensorcast.api._errors import (
    DeviceMismatch,
    FeedFailed,
    InvalidPlan,
    TensorCastError,
)
from tensorcast.api._indices import (
    TensorDataIndex,
    TensorDeviceOffsets,
    TensorMetaIndex,
    calculate_tensor_device_offsets,
)
from tensorcast.api._runtime import require_runtime
from tensorcast.api._utils import validate_disk_index_matches
from tensorcast.api._view_ops import NarrowOp, TransposeOp, ViewSpecBuildResult
from tensorcast.common.identity import ArtifactIdKind, validate_client_generated_id
from tensorcast.observability.otel import ensure_client_otel
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import (
    ArtifactDescriptor,
    CanonicalRange,
    CoalescedHandshake,
    CoalescedPlan,
    Handshake,
    LeasePlan,
    LeaseSegment,
    LocalStableTierResult,
    RegisterStorage,
    RegisterTensorAlias,
    StableDramHandshake,
    StableDramPlan,
)

# Internal alignment hint for layout computation.
# Currently unused by the underlying calculator but kept for API clarity.
DEFAULT_ALIGN = 1


@dataclass
class RegistrationResult:
    """Unified result for registration APIs.

    Provides additional build metadata so higher-level callers (e.g., Store
    sessions) can construct canonical index models and replica descriptors
    without recomputing tensor layout.
    """

    state_dict: dict[str, torch.Tensor] | None
    descriptor: ArtifactDescriptor
    lease: RegisteredLease | None
    build: "BuildContext"
    layout: "CoalescedLayout"
    index_bytes: bytes
    plan: PlanType
    view_id: str | None = None
    view_index_json: bytes | None = None
    view_data_hash: str | None = None
    canonical_ranges: tuple[CanonicalRange, ...] = ()
    registration_kind: Literal["canonical", "piece"] = "canonical"
    allow_partial: bool = False
    local_stable_tier: LocalStableTierResult | None = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ViewPlanChunk:
    canonical_offset: int
    view_offset: int
    length: int
    segment_aligned: bool


@dataclass(frozen=True)
class InverseTensorPlan:
    tensor_name: str
    dst_offset: int
    canonical_offset: int
    storage_offset_elements: int
    canonical_shape: tuple[int, ...]
    canonical_stride: tuple[int, ...]
    view_shape: tuple[int, ...]
    view_stride: tuple[int, ...]
    permutation: tuple[int, ...]
    dtype: str
    element_size_bytes: int


@dataclass(frozen=True)
class ViewPlanMetadata:
    view_size_bytes: int
    view_index_json: bytes
    write_chunks: tuple[ViewPlanChunk, ...]
    inverse_requires_materialization: bool
    inverse_tensors: tuple[InverseTensorPlan, ...]


@dataclass
class ViewRegistrationContext:
    canonical_index_bytes: bytes
    view_options: "store_daemon_pb2.ViewRegistrationOptions"
    placement: int
    plan: ViewPlanMetadata
    tensors: dict[str, torch.Tensor]
    canonical_ranges: tuple[CanonicalRange, ...]
    registration_kind: int


class RegisteredArtifact:
    def __init__(
        self,
        registration_id: str,
        daemon_address: str,
        *,
        ttl_ms: int | None = None,
        client: DaemonCtl | None = None,
    ) -> None:
        import threading

        self.registration_id = registration_id
        self._addr = daemon_address
        self._ttl_ms = int(ttl_ms) if ttl_ms and ttl_ms > 0 else 0
        self._ka_thread: threading.Thread | None = None
        self._ka_stop = threading.Event()
        self._epoch: int = 0
        if client is None:
            raise RuntimeError(
                "RegisteredArtifact requires an active TensorCast session. "
                "Call tensorcast.startup.init(mode='connect'|'create') before using registration APIs."
            )
        # Cache client for this handle's lifetime
        self._ctl = client

    @property
    def client(self) -> DaemonCtl:
        return self._ctl

    def __enter__(self) -> "RegisteredArtifact":
        import threading

        if self._ttl_ms > 0 and self._ka_thread is None:

            def _keepalive() -> None:
                ctl = self.client
                interval = max(1.0, self._ttl_ms / 2000.0)
                while not self._ka_stop.wait(interval):
                    try:
                        ctl.keep_alive_registered_artifact(
                            self.registration_id, self._ttl_ms, self._epoch
                        )
                        self._epoch += 1
                    except Exception:  # noqa: BLE001
                        continue

            t = threading.Thread(target=_keepalive, daemon=True)
            t.start()
            self._ka_thread = t
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._ka_stop.set()
        if self._ka_thread and self._ka_thread.is_alive():
            self._ka_thread.join(timeout=1.0)

    def commit(self, timeout_s: float = 60.0):
        """Commit and return CommitResult(descriptor, existed)."""
        commit_res = self.client.commit_registered_artifact(
            self.registration_id, timeout_s=timeout_s
        )
        self.__exit__(None, None, None)
        return commit_res

    def abort(self, timeout_s: float = 15.0) -> bool:
        self.__exit__(None, None, None)
        return self.client.abort_registered_artifact(
            self.registration_id, timeout_s=timeout_s
        )

    def revoke(self, reason: str = "", timeout_s: float = 10.0) -> bool:
        self.__exit__(None, None, None)
        return self.client.revoke_registered_artifact(self.registration_id, reason)


class RegisteredLease:
    """Post-commit lease keepalive and best-effort revoke helper.

    Use as a context manager to maintain keepalive at ttl/2 cadence. On exit,
    attempts to revoke immediately; if revoke fails, relies on TTL/pid watcher.
    """

    _live = None  # weakref.WeakSet[RegisteredLease]
    _atexit_installed = False

    def __init__(
        self,
        registration_id: str,
        daemon_address: str,
        ttl_ms: int,
        owner_pid: int,
        *,
        client: DaemonCtl,
    ) -> None:
        if RegisteredLease._live is None:
            RegisteredLease._live = weakref.WeakSet()

        self.registration_id = registration_id
        self._addr = daemon_address
        self._ttl_ms = int(ttl_ms)
        self._epoch: int = 0
        self._owner_pid: int = int(owner_pid)
        self._ka_thread: threading.Thread | None = None
        self._ka_stop = threading.Event()
        self._client = client

        RegisteredLease._live.add(self)
        if not RegisteredLease._atexit_installed:
            atexit.register(RegisteredLease._revoke_all)
            RegisteredLease._atexit_installed = True
        # GC finalizer
        weakref.finalize(self, self._best_effort_revoke)

    def __enter__(self) -> "RegisteredLease":
        if self._ttl_ms > 0 and self._ka_thread is None:
            # Jitter ±10%
            interval = max(1.0, (self._ttl_ms / 2000.0))

            def _keepalive() -> None:
                import random

                ctl = self._client
                while not self._ka_stop.wait(interval * (0.9 + 0.2 * random.random())):
                    try:
                        ctl.keep_alive_registered_artifact(
                            self.registration_id, self._ttl_ms, self._epoch
                        )
                        self._epoch += 1
                    except Exception:  # noqa: BLE001
                        continue

            t = threading.Thread(target=_keepalive, daemon=True)
            t.start()
            self._ka_thread = t
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_keepalive()
        self._best_effort_revoke()

    def _stop_keepalive(self) -> None:
        if self._ka_thread and self._ka_thread.is_alive():
            self._ka_stop.set()
            self._ka_thread.join(timeout=1.0)

    def _best_effort_revoke(self) -> None:
        try:
            self._client.revoke_registered_artifact(self.registration_id)
        except Exception:  # noqa: BLE001
            with contextlib.suppress(Exception):
                logger.debug(
                    "Failed to revoke registered artifact %s",
                    self.registration_id,
                    exc_info=True,
                )
            # rely on TTL + pid_watcher
            pass

    @classmethod
    def _revoke_all(cls) -> None:
        if cls._live is None:
            return
        for inst in list(cls._live):
            try:
                inst._best_effort_revoke()
            except Exception:  # noqa: BLE001
                continue

    @property
    def ttl_ms(self) -> int:
        return self._ttl_ms

    @property
    def owner_pid(self) -> int:
        return self._owner_pid

    @property
    def daemon_address(self) -> str:
        return self._addr


def _upsert_key_mapping_if_needed(
    *,
    key: str | None,
    artifact_id: str,
    disk_path: str | None,
    descriptor: ArtifactDescriptor | None = None,
    client: DaemonCtl | None,
) -> None:
    if not key:
        return
    if client is None:
        raise RuntimeError("publish_replica_key requires an active daemon client")
    if descriptor is None:
        logger.warning("Skipping key publish for %s: missing artifact descriptor", key)
        return
    try:
        ok = client.publish_replica_key(
            key=key, descriptor=descriptor, disk_path=disk_path or ""
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to publish key %s via daemon", key)
        return
    if not ok:
        logger.warning(
            "Key mapping for %s already exists; keeping registration for %s",
            key,
            artifact_id,
        )
        return


def _persist_publish_if_needed(
    *,
    desc: ArtifactDescriptor,
    options: RegisterArtifactOptions,
    state_dict_to_save: dict[str, torch.Tensor] | None,
    client: DaemonCtl,
) -> None:
    if options.disk_path is not None and options.disk_path.strip() == "":
        if state_dict_to_save is not None:
            raise TensorCastError(
                "disk_path=='' local persistence is test-only and disabled in production. "
                "Provide an explicit disk_path, or persist via your own pipeline; "
                "tests may use tensorcast.testing.io_disk.save_dict."
            )
        return
    else:
        _upsert_key_mapping_if_needed(
            key=options.key,
            artifact_id=desc.artifact_id,
            disk_path=options.disk_path,
            descriptor=desc,
            client=client,
        )


class BuildContext:
    """Holds derived metadata and device context for artifact registration."""

    def __init__(
        self,
        *,
        device_id: int,
        input_mode: str,
        tensor_meta_index: TensorMetaIndex,
        tensor_source_index: TensorDataIndex,
        storage_graph: TensorStorageGraph,
    ) -> None:
        self.device_id = int(device_id)
        self.input_mode = input_mode  # "cpu" or "cuda"
        self.tensor_meta_index = tensor_meta_index
        self.tensor_source_index = tensor_source_index
        self.storage_graph = storage_graph

    @staticmethod
    def from_artifact(
        artifact: dict[str, torch.Tensor], device_id: int | torch.device | None
    ) -> "BuildContext":
        has_cuda = False
        has_cpu = False
        cuda_device_id: int | None = None
        for t in artifact.values():
            if t.is_cuda:
                has_cuda = True
                this_idx = t.device.index if t.device.index is not None else 0
                if cuda_device_id is None:
                    cuda_device_id = this_idx
                elif this_idx != cuda_device_id:
                    raise DeviceMismatch("All CUDA tensors must be on the same device")
            else:
                has_cpu = True

        if has_cuda and has_cpu:
            raise DeviceMismatch(
                "Artifact tensors must be all CPU or all CUDA tensors on the same device"
            )

        if has_cuda:
            input_mode = "cuda"
            inferred_device_id = int(cuda_device_id or 0)
        else:
            input_mode = "cpu"
            inferred_device_id = None

        target_device_id: int
        if device_id is None:
            if input_mode == "cuda":
                if inferred_device_id is None:
                    raise DeviceMismatch("CUDA tensors require a resolved device id")
                target_device_id = inferred_device_id
            else:
                if torch.cuda.is_available():
                    try:
                        target_device_id = int(torch.cuda.current_device())
                    except Exception:  # noqa: BLE001
                        target_device_id = 0
                else:
                    target_device_id = 0
        else:
            target_device_id = resolve_device(device_id)
            if (
                input_mode == "cuda"
                and inferred_device_id is not None
                and inferred_device_id != target_device_id
            ):
                raise DeviceMismatch(
                    f"Tensor device mismatch: expected cuda:{target_device_id}, got cuda:{inferred_device_id}"
                )

        storage_graph = build_tensor_storage_graph(artifact)
        if input_mode == "cuda":
            for entry in storage_graph.storages.values():
                if entry.device_id != target_device_id:
                    raise DeviceMismatch(
                        f"Tensor storage device mismatch: expected cuda:{target_device_id}, "
                        f"got {entry.device_id}"
                    )

        return BuildContext(
            device_id=target_device_id,
            input_mode=input_mode,
            tensor_meta_index=storage_graph.tensor_meta_index,
            tensor_source_index=storage_graph.tensor_source_index,
            storage_graph=storage_graph,
        )


class CoalescedLayout:
    """Represents the computed device layout for the artifact tensors."""

    def __init__(
        self,
        *,
        device_id: int,
        offsets: TensorDeviceOffsets,
        unique_chunks: list[tuple[int, int, int, int]],
        total_size: int,
    ) -> None:
        self.device_id = int(device_id)
        self.offsets = offsets
        self.unique_chunks = unique_chunks
        self.total_size = int(total_size)

    @staticmethod
    def compute(
        tensor_source_index: TensorDataIndex,
        device_id: int,
        *,
        align: int = DEFAULT_ALIGN,
    ) -> "CoalescedLayout":
        # Note: current implementation ignores `align` to match existing API.

        tensor_device_offsets, unique_chunks_all = calculate_tensor_device_offsets(
            tensor_source_index, int(device_id)
        )
        unique_chunks = unique_chunks_all.get(int(device_id), [])
        total_size_bytes = (
            max(dst + sz for _, sz, dst, _ in unique_chunks) if unique_chunks else 0
        )
        return CoalescedLayout(
            device_id=int(device_id),
            offsets=tensor_device_offsets,
            unique_chunks=unique_chunks,
            total_size=total_size_bytes,
        )


def build_canonical_index_bytes(
    tensor_meta_index: TensorMetaIndex,
    tensor_source_index: TensorDataIndex,
    tensor_device_offsets: TensorDeviceOffsets,
    device_id: int,
) -> bytes:
    canonical_index: dict[str, tuple[int, int, list[int], list[int], str, int]] = {}
    for name in sorted(tensor_meta_index.keys()):
        shape, stride, dtype, storage_offset = tensor_meta_index[name]
        _, storage_size = tensor_source_index[name]
        dst_off = int(tensor_device_offsets[int(device_id)][name])
        canonical_index[name] = (
            dst_off,
            int(storage_size),
            list(shape),
            list(stride),
            dtype,
            int(storage_offset),
        )
    return json.dumps(canonical_index, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _prepare_build(
    artifact: dict[str, torch.Tensor],
    device_id: int | torch.device | None,
) -> tuple[BuildContext, CoalescedLayout, bytes]:
    """Compute BuildContext, layout, and index bytes for an artifact."""
    ctx = BuildContext.from_artifact(artifact, device_id)
    layout = CoalescedLayout.compute(
        ctx.tensor_source_index, ctx.device_id, align=DEFAULT_ALIGN
    )
    index_bytes = build_canonical_index_bytes(
        ctx.tensor_meta_index, ctx.tensor_source_index, layout.offsets, ctx.device_id
    )
    return ctx, layout, index_bytes


def _tensor_indices_from_canonical_index_bytes(
    index_bytes: bytes,
) -> tuple[TensorMetaIndex, TensorDataIndex]:
    try:
        index_obj = json.loads(index_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise TensorCastError(
            "Failed to parse canonical index bytes for view registration"
        ) from exc
    tensor_meta_index: TensorMetaIndex = {}
    tensor_data_index: TensorDataIndex = {}
    for name, meta in index_obj.items():
        if not isinstance(meta, list | tuple) or len(meta) != 6:
            raise TensorCastError(
                f"Invalid canonical index entry for '{name}': expected 6 fields"
            )
        offset, size, shape, stride, dtype, storage_offset = meta
        tensor_meta_index[name] = (
            [int(v) for v in shape],
            [int(v) for v in stride],
            str(dtype),
            int(storage_offset),
        )
        tensor_data_index[name] = (int(offset), int(size))
    return tensor_meta_index, tensor_data_index


def _build_context_from_canonical_index(
    index_bytes: bytes,
    device_id: int,
) -> tuple["BuildContext", "CoalescedLayout"]:
    tensor_meta_index, tensor_data_index = _tensor_indices_from_canonical_index_bytes(
        index_bytes
    )
    tensor_device_offsets, unique_chunks = calculate_tensor_device_offsets(
        tensor_data_index, device_id
    )
    chunk_list = unique_chunks.get(device_id, [])
    total_size = 0
    for offset, size, _, _ in chunk_list:
        total_size = max(total_size, int(offset) + int(size))

    storage_graph = TensorStorageGraph(
        storages={},
        aliases={},
        tensor_meta_index=tensor_meta_index,
        tensor_source_index=tensor_data_index,
    )
    ctx = BuildContext(
        device_id=device_id,
        input_mode="cpu",
        tensor_meta_index=tensor_meta_index,
        tensor_source_index=tensor_data_index,
        storage_graph=storage_graph,
    )
    layout = CoalescedLayout(
        device_id=device_id,
        offsets=tensor_device_offsets,
        unique_chunks=chunk_list,
        total_size=total_size,
    )
    return ctx, layout


def _merge_canonical_ranges(
    chunks: Sequence[ViewPlanChunk],
) -> tuple[CanonicalRange, ...]:
    sorted_chunks = sorted(
        (chunk.canonical_offset, chunk.canonical_offset + chunk.length)
        for chunk in chunks
        if chunk.length > 0
    )
    if not sorted_chunks:
        return ()
    merged: list[CanonicalRange] = []
    current_start, current_end = sorted_chunks[0]
    for start, end in sorted_chunks[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append(
                CanonicalRange(
                    offset=int(current_start), length=int(current_end - current_start)
                )
            )
            current_start, current_end = start, end
    merged.append(
        CanonicalRange(
            offset=int(current_start), length=int(current_end - current_start)
        )
    )
    return tuple(merged)


def _compute_view_plan_metadata(
    canonical_index_bytes: bytes,
    build_result: ViewSpecBuildResult,
) -> ViewPlanMetadata:
    if build_result.is_identity:
        raise TensorCastError("View registration requires explicit view operations")
    normalized_ops = build_result.to_normalized_dict()
    native_plan = compute_view_registration_plan(canonical_index_bytes, normalized_ops)
    forward = native_plan["forward"]
    write_chunks = tuple(
        ViewPlanChunk(
            canonical_offset=int(chunk["canonical_offset"]),
            view_offset=int(chunk["view_offset"]),
            length=int(chunk["length"]),
            segment_aligned=bool(chunk["segment_aligned"]),
        )
        for chunk in native_plan.get("write_chunks", [])
    )
    inverse_native = native_plan.get("inverse_transform", {})
    inverse_tensors = tuple(
        InverseTensorPlan(
            tensor_name=str(tp["tensor_name"]),
            dst_offset=int(tp["dst_offset"]),
            canonical_offset=int(tp["canonical_offset"]),
            storage_offset_elements=int(tp["storage_offset_elements"]),
            canonical_shape=tuple(int(v) for v in tp["canonical_shape"]),
            canonical_stride=tuple(int(v) for v in tp["canonical_stride"]),
            view_shape=tuple(int(v) for v in tp["view_shape"]),
            view_stride=tuple(int(v) for v in tp["view_stride"]),
            permutation=tuple(int(v) for v in tp["permutation"]),
            dtype=str(tp["dtype"]),
            element_size_bytes=int(tp["element_size_bytes"]),
        )
        for tp in inverse_native.get("tensors", [])
    )
    return ViewPlanMetadata(
        view_size_bytes=int(forward["view_size_bytes"]),
        view_index_json=bytes(forward["view_index_json"]),
        write_chunks=write_chunks,
        inverse_requires_materialization=bool(
            inverse_native.get("requires_materialization", False)
        ),
        inverse_tensors=inverse_tensors,
    )


def _linearize_view_tensors(
    tensors: Mapping[str, torch.Tensor],
    plan: ViewPlanMetadata,
) -> bytearray:
    if plan.view_size_bytes == 0:
        return bytearray()
    try:
        layout_index = json.loads(plan.view_index_json.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise TensorCastError("Failed to parse view_index_json from plan") from exc

    buffer = bytearray(plan.view_size_bytes)
    mv = memoryview(buffer)
    for name in sorted(layout_index.keys()):
        if name not in tensors:
            raise TensorCastError(
                f"View tensor '{name}' missing from registration payload"
            )
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TensorCastError(f"View tensor '{name}' must be a torch.Tensor")
        offset, size_bytes, _, _, dtype_str, _ = layout_index[name]
        expected_bytes = int(size_bytes)
        src_dtype = str(tensor.dtype)
        if src_dtype != str(dtype_str):
            raise TensorCastError(
                f"Tensor '{name}' dtype mismatch: expected {dtype_str}, found {src_dtype}"
            )
        contiguous = tensor.detach()
        if contiguous.device.type != "cpu":
            contiguous = contiguous.to(torch.device("cpu"), non_blocking=False)
        contiguous = contiguous.contiguous()
        data_bytes = contiguous.numpy().tobytes()
        if len(data_bytes) != expected_bytes:
            raise TensorCastError(
                f"Tensor '{name}' byte size mismatch: expected {expected_bytes}, got {len(data_bytes)}"
            )
        start = int(offset)
        mv[start : start + expected_bytes] = data_bytes
    return buffer


def _torch_dtype_from_string(dtype_str: str) -> torch.dtype:
    if hasattr(torch, dtype_str.split(".")[-1]):
        return getattr(torch, dtype_str.split(".")[-1])
    raise TensorCastError(f"Unsupported dtype for view registration: {dtype_str}")


def _materialize_canonical_tensors(
    canonical_index_bytes: bytes,
    build_result: ViewSpecBuildResult,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    tensor_meta_index, _ = _tensor_indices_from_canonical_index_bytes(
        canonical_index_bytes
    )
    canonical_tensors: dict[str, torch.Tensor] = {}
    for name, (shape, _, dtype_str, _) in tensor_meta_index.items():
        if name not in tensors:
            raise TensorCastError(f"View registration missing tensor '{name}'")
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TensorCastError(f"Tensor '{name}' must be a torch.Tensor")
        tensor_cpu = tensor.detach()
        if tensor_cpu.device.type != "cpu":
            tensor_cpu = tensor_cpu.to(torch.device("cpu"), non_blocking=False)
        tensor_cpu = tensor_cpu.contiguous()
        ops = list(build_result.tensor_ops.get(name, ()))
        # Apply inverse operations (reverse order)
        for op in reversed(ops):
            if isinstance(op, TransposeOp):
                tensor_cpu = tensor_cpu.transpose(int(op.dim0), int(op.dim1))
            elif isinstance(op, NarrowOp):
                dim = int(op.dim)
                start = int(op.start)
                length = int(op.length)
                full = torch.zeros(
                    tuple(int(d) for d in shape),
                    dtype=tensor_cpu.dtype,
                )
                slice_spec = [slice(None)] * full.ndim
                slice_spec[dim] = slice(start, start + length)
                full[tuple(slice_spec)].copy_(tensor_cpu)
                tensor_cpu = full
            else:
                raise TensorCastError(
                    f"Unsupported view operation type: {type(op).__name__}"
                )
        # Ensure dtype matches canonical requirements
        target_dtype = _torch_dtype_from_string(dtype_str)
        if tensor_cpu.dtype != target_dtype:
            tensor_cpu = tensor_cpu.to(target_dtype)
        expected_shape = tuple(int(d) for d in shape)
        if tuple(tensor_cpu.shape) != expected_shape:
            raise TensorCastError(
                f"Tensor '{name}' has shape {tuple(tensor_cpu.shape)}, expected {expected_shape}"
            )
        canonical_tensors[name] = tensor_cpu
    return canonical_tensors


def make_plan_model(
    options: RegisterArtifactOptions, total_size_bytes: int | None = None
) -> CoalescedPlan | LeasePlan | StableDramPlan:
    plan_type: PlanType = options.plan
    if plan_type is PlanType.DRAM_STABLE:
        if not options.stage_on_gpu:
            raise InvalidPlan("dram_stable with stage_on_gpu=false is not implemented")
        return StableDramPlan(
            kind="dram_stable",
            stage_on_gpu=options.stage_on_gpu,
            release_gpu_on_commit=options.release_gpu_on_commit,
        )
    if plan_type is PlanType.VRAM_COALESCED:
        return CoalescedPlan(
            kind="coalesced",
            max_inflight_bytes=options.max_inflight_bytes,
            release_on_tensor_commit=options.release_on_tensor_commit,
        )
    if plan_type is PlanType.VRAM_LEASED:
        # Current release only supports Lease-In-Place (LIP)
        in_place = options.lease_in_place
        if not in_place:
            raise InvalidPlan(
                "vram_leased (in_place=false) is not implemented; set lease_in_place=True"
            )
        return LeasePlan(
            kind="lease",
            min_tensor_bytes=options.min_tensor_bytes,
            max_tensor_count=options.max_tensor_count,
            lease_bytes_limit=options.lease_bytes_limit,
            in_place=True,
        )
    raise InvalidPlan(f"Unknown plan: {plan_type}")


class _CoalescedUploader:
    def upload(
        self,
        *,
        artifact: dict[str, torch.Tensor],
        ctx: BuildContext,
        layout: CoalescedLayout,
        handle: RegisteredArtifact,
        handshake: Handshake,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(handshake, CoalescedHandshake):
            raise TensorCastError("Unexpected handshake type for coalesced plan")
        cuda_handle = handshake.daemon_ipc_handle
        base_ptr = get_cuda_memory_ptr(ctx.device_id, cuda_handle)
        dest_state_dict = restore_tensors(
            ctx.tensor_meta_index,
            {ctx.device_id: int(base_ptr)},
            layout.offsets,
            True,
        )
        if cancel_event and cancel_event.is_set():
            raise CancelledError
        for name, src in artifact.items():
            if cancel_event and cancel_event.is_set():
                raise CancelledError
            dst = dest_state_dict[name]
            local = src
            if not local.is_cuda:
                local = local.to(torch.device("cuda", ctx.device_id), non_blocking=True)
            else:
                if (local.device.index or 0) != int(ctx.device_id):
                    raise DeviceMismatch(
                        f"Tensor '{name}' device mismatch: expected cuda:{ctx.device_id}, got {local.device}"
                    )
            if local.dtype != dst.dtype:
                local = local.to(dst.dtype)
            if tuple(local.shape) != tuple(dst.shape):
                raise DeviceMismatch(
                    f"Shape mismatch for tensor '{name}': {tuple(local.shape)} vs {tuple(dst.shape)}"
                )
            dst.copy_(local, non_blocking=True)
        if cancel_event and cancel_event.is_set():
            raise CancelledError
        torch.cuda.synchronize(ctx.device_id)
        return dest_state_dict


class _StableDramUploader:
    def upload(
        self,
        *,
        artifact: dict[str, torch.Tensor],
        ctx: BuildContext,
        layout: CoalescedLayout,
        handle: RegisteredArtifact,
        handshake: Handshake,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(handshake, StableDramHandshake):
            raise TensorCastError("Unexpected handshake type for dram_stable plan")
        if not handshake.staging_cuda_ipc_handle:
            raise TensorCastError("dram_stable requires a staging CUDA IPC handle")
        base_ptr = get_cuda_memory_ptr(ctx.device_id, handshake.staging_cuda_ipc_handle)
        dest_state_dict = restore_tensors(
            ctx.tensor_meta_index,
            {ctx.device_id: int(base_ptr)},
            layout.offsets,
            True,
        )
        if cancel_event and cancel_event.is_set():
            raise CancelledError
        for name, src in artifact.items():
            if cancel_event and cancel_event.is_set():
                raise CancelledError
            dst = dest_state_dict[name]
            local = src
            if not local.is_cuda:
                local = local.to(torch.device("cuda", ctx.device_id), non_blocking=True)
            else:
                if (local.device.index or 0) != int(ctx.device_id):
                    raise DeviceMismatch(
                        f"Tensor '{name}' device mismatch: expected cuda:{ctx.device_id}, got {local.device}"
                    )
            if local.dtype != dst.dtype:
                local = local.to(dst.dtype)
            if tuple(local.shape) != tuple(dst.shape):
                raise DeviceMismatch(
                    f"Shape mismatch for tensor '{name}': {tuple(local.shape)} vs {tuple(dst.shape)}"
                )
            dst.copy_(local, non_blocking=True)
        if cancel_event and cancel_event.is_set():
            raise CancelledError
        torch.cuda.synchronize(ctx.device_id)
        return dest_state_dict


class _LeaseUploader:
    def upload(
        self,
        *,
        artifact: dict[str, torch.Tensor],
        ctx: BuildContext,
        layout: CoalescedLayout,
        handle: RegisteredArtifact,
        handshake: Handshake,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, torch.Tensor]:
        def _export_cuda_ipc_handle(ptr: int) -> tuple[bytes, int]:
            return get_cuda_memory_handle_with_offset(ctx.device_id, int(ptr))

        graph = ctx.storage_graph
        offsets_for_device = layout.offsets.get(int(ctx.device_id), {})
        if not offsets_for_device:
            raise TensorCastError(
                f"No layout offsets recorded for device {ctx.device_id} during lease registration"
            )

        storage_to_dst: dict[str, int] = {}
        aliases_payload: list[RegisterTensorAlias] = []
        for name in sorted(graph.aliases.keys()):
            alias = graph.aliases[name]
            if cancel_event and cancel_event.is_set():
                raise CancelledError
            if name not in offsets_for_device:
                raise TensorCastError(
                    f"Missing layout offset for tensor '{name}' in lease plan"
                )
            dst_offset = int(offsets_for_device[name])
            storage_to_dst.setdefault(alias.storage_id, dst_offset)
            aliases_payload.append(
                RegisterTensorAlias(
                    name=alias.name,
                    storage_id=alias.storage_id,
                    storage_offset=int(alias.storage_offset),
                    logical_length=int(alias.logical_length),
                    shape=list(alias.shape),
                    stride=list(alias.stride),
                    dtype=alias.dtype,
                )
            )

        segments: list[LeaseSegment] = []
        storages_payload: list[RegisterStorage] = []
        for storage_id in sorted(graph.storages.keys()):
            if cancel_event and cancel_event.is_set():
                raise CancelledError
            entry = graph.storages[storage_id]
            if entry.device_id != int(ctx.device_id):
                raise DeviceMismatch(
                    f"Storage '{storage_id}' device mismatch: expected cuda:{ctx.device_id}, got {entry.device_id}"
                )
            dst_offset = storage_to_dst[storage_id]
            length_bytes = int(entry.size_bytes)
            # Prefer region-backed registration when the storage is fully covered
            # by a pre-registered VRAM region for this device.
            rec = region_cache.find_region_for(
                int(ctx.device_id), int(entry.base_ptr), int(entry.size_bytes)
            )
            if rec is not None:
                # Emit a region-referenced storage entry; drop redundant handle export.
                base_offset = int(entry.base_ptr) - int(rec.base_ptr)
                segments.append(
                    LeaseSegment(
                        storage_id=storage_id,
                        storage_offset=0,
                        artifact_offset=int(dst_offset),
                        length=length_bytes,
                    )
                )
                storages_payload.append(
                    RegisterStorage(
                        storage_id=storage_id,
                        device_id=int(ctx.device_id),
                        cuda_ipc_handle=None,
                        storage_length=length_bytes,
                        vram_region_id=rec.region_id,
                        mapping_base_offset=int(base_offset),
                    )
                )
            else:
                # Fallback to legacy handle-based path.
                handle_bytes, base_offset = _export_cuda_ipc_handle(int(entry.base_ptr))
                if not handle_bytes:
                    raise TensorCastError(
                        f"Failed to export CUDA IPC handle for storage '{storage_id}'"
                    )
                segments.append(
                    LeaseSegment(
                        storage_id=storage_id,
                        storage_offset=0,
                        artifact_offset=int(dst_offset),
                        length=length_bytes,
                    )
                )
                storages_payload.append(
                    RegisterStorage(
                        storage_id=storage_id,
                        device_id=int(ctx.device_id),
                        cuda_ipc_handle=handle_bytes,
                        storage_length=length_bytes,
                        mapping_base_offset=int(base_offset),
                    )
                )
        if not storages_payload:
            raise TensorCastError(
                "No storage entries resolved during lease registration; artifact tensors must supply shared storage metadata"
            )
        if not aliases_payload:
            raise TensorCastError(
                "No tensor aliases resolved during lease registration; artifact tensors must supply alias metadata"
            )
        if cancel_event and cancel_event.is_set():
            raise CancelledError
        ctl = handle.client
        ok = ctl.feed_register_artifact_lease_segments(
            handle.registration_id,
            segments,
            storages=storages_payload,
            tensor_aliases=aliases_payload,
        )
        if not ok:
            raise FeedFailed("Lease segments feed failed")
        return artifact


PLAN_REGISTRY: dict[PlanType, object] = {
    PlanType.DRAM_STABLE: _StableDramUploader(),
    PlanType.VRAM_COALESCED: _CoalescedUploader(),
    PlanType.VRAM_LEASED: _LeaseUploader(),
}


def _register_artifact_core(
    *,
    artifact: dict[str, torch.Tensor],
    options: RegisterArtifactOptions,
    device_id: int | torch.device | None,
    ttl_ms: int | None,
    client_artifact_id: str | None = None,
    force_lease_in_place: bool = False,
    prevalidate_disk: bool = True,
    client: DaemonCtl | None = None,
    daemon_address: str | None = None,
    cancel_event: threading.Event | None = None,
    on_begin: Callable[[RegisteredArtifact], None] | None = None,
    view: ViewRegistrationContext | None = None,
) -> RegistrationResult:
    if client is None:
        runtime = require_runtime()
        ctl = runtime.client
        addr = runtime.address
    else:
        ctl = client
        resolved_addr = daemon_address or client.server_address
        addr = str(resolved_addr)

    if not artifact:
        raise TensorCastError("artifact must not be empty")

    target_device_id: int | None = None
    if isinstance(device_id, torch.device):
        target_device_id = resolve_device(device_id)
    elif isinstance(device_id, int):
        target_device_id = int(device_id)

    normalized_artifact_id: str | None = None
    identity_kind = ArtifactIdKind.MI2
    if client_artifact_id:
        candidate = client_artifact_id.strip()
        if candidate:
            try:
                validate_client_generated_id(candidate)
            except ValueError as exc:
                raise TensorCastError(str(exc)) from exc
            normalized_artifact_id = candidate
            identity_kind = ArtifactIdKind.CGID
        else:
            normalized_artifact_id = None

    if view is None:
        ctx, layout, index_bytes = _prepare_build(artifact, device_id)
        if target_device_id is None:
            target_device_id = ctx.device_id
        else:
            ctx.device_id = target_device_id
    else:
        if target_device_id is None:
            target_device_id = 0
        ctx, layout = _build_context_from_canonical_index(
            view.canonical_index_bytes, target_device_id
        )
        index_bytes = view.canonical_index_bytes

    if (
        view is None
        and prevalidate_disk
        and options.disk_path is not None
        and options.disk_path.strip() != ""
    ):
        cand = Path(options.disk_path)
        if cand.exists() and cand.is_dir():
            validate_disk_index_matches(index_bytes, str(cand))

    ensure_client_otel("tensorcast-client", role="client")
    tracer = trace.get_tracer(__name__)

    if force_lease_in_place:
        plan_type: PlanType = PlanType.VRAM_LEASED
        plan_model = LeasePlan(
            kind="lease",
            min_tensor_bytes=options.min_tensor_bytes,
            max_tensor_count=options.max_tensor_count,
            lease_bytes_limit=options.lease_bytes_limit,
            in_place=True,
        )
    else:
        plan_type = options.plan
        plan_model = make_plan_model(options, layout.total_size)

    # Plan input-mode constraints
    if plan_type is PlanType.DRAM_STABLE and not options.stage_on_gpu:
        raise InvalidPlan("dram_stable with stage_on_gpu=false is not implemented")
    if plan_type is PlanType.VRAM_LEASED and ctx.input_mode != "cuda":
        raise DeviceMismatch(
            "vram_leased plan requires CUDA tensors (device_id must be inferred)"
        )
    if view is not None and plan_type is not PlanType.VRAM_COALESCED:
        raise InvalidPlan("View registration requires vram_coalesced plan")

    span_names = {
        PlanType.DRAM_STABLE: "Client/RegisterArtifact.StableDram",
        PlanType.VRAM_COALESCED: "Client/RegisterArtifact.Coalesced",
        PlanType.VRAM_LEASED: "Client/RegisterArtifact.Lease",
    }

    with tracer.start_as_current_span(span_names[plan_type], kind=SpanKind.INTERNAL):
        span = trace.get_current_span()
        span.set_attribute("tc.artifact.identity_kind", identity_kind.value)
        if normalized_artifact_id:
            span.set_attribute("tc.artifact.client_artifact_id", normalized_artifact_id)
        policy = StorePolicy.parse(options.policy)
        total_size_bytes = layout.total_size
        if (
            view is not None
            and view.registration_kind == store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE
        ):
            total_size_bytes = view.plan.view_size_bytes
        begin_response = ctl.begin_register_artifact(
            device_id=ctx.device_id,
            total_size_bytes=total_size_bytes,
            ttl_ms=ttl_ms,
            tensor_index_data=index_bytes,
            encoding="json",
            schema_version="v3",
            client_artifact_id=normalized_artifact_id,
            plan=plan_model,
            policy=policy,
            view=view.view_options if view is not None else None,
            timeout_s=60.0,
        )
        handle = RegisteredArtifact(
            begin_response.registration_id,
            addr,
            ttl_ms=ttl_ms or 0,
            client=ctl,
        )
        hs = begin_response.handshake
        if on_begin is not None:
            on_begin(handle)
        if cancel_event and cancel_event.is_set():
            with contextlib.suppress(Exception):
                handle.abort(timeout_s=5.0)
            raise CancelledError

        if view is not None:
            if view.placement == store_daemon_pb2.TRANSFORM_PLACEMENT_SERVER:
                with handle:
                    try:
                        if cancel_event and cancel_event.is_set():
                            raise CancelledError
                        payload = _linearize_view_tensors(view.tensors, view.plan)
                        ok = ctl.feed_register_artifact_view_chunks(
                            handle.registration_id,
                            payload,
                        )
                        if not ok:
                            raise TensorCastError(
                                "Failed to stream view registration bytes to daemon"
                            )
                        if cancel_event and cancel_event.is_set():
                            raise CancelledError
                        commit_res = handle.commit(timeout_s=60.0)
                        desc = commit_res.descriptor
                        _persist_publish_if_needed(
                            desc=desc,
                            options=options,
                            state_dict_to_save=None,
                            client=ctl,
                        )
                        return RegistrationResult(
                            state_dict=None,
                            descriptor=desc,
                            lease=None,
                            build=ctx,
                            layout=layout,
                            index_bytes=index_bytes,
                            plan=plan_type,
                            view_id=commit_res.view_id,
                            view_index_json=commit_res.view_index_json,
                            view_data_hash=commit_res.view_data_hash,
                            canonical_ranges=commit_res.canonical_ranges
                            or view.canonical_ranges,
                            registration_kind=commit_res.registration_kind,
                            allow_partial=commit_res.registration_kind == "piece",
                            local_stable_tier=commit_res.local_stable_tier,
                        )
                    except CancelledError:
                        with contextlib.suppress(Exception):
                            handle.abort(timeout_s=5.0)
                        raise
            elif view.placement != store_daemon_pb2.TRANSFORM_PLACEMENT_CLIENT:
                raise TensorCastError(
                    "Unknown transform placement for view registration"
                )

        registrar = PLAN_REGISTRY[plan_type]
        with handle:
            try:
                # Upload per plan
                if isinstance(registrar, (_CoalescedUploader, _StableDramUploader)):
                    state_dict: dict[str, torch.Tensor] | None = registrar.upload(
                        artifact=artifact,
                        ctx=ctx,
                        layout=layout,
                        handle=handle,
                        handshake=hs,
                        cancel_event=cancel_event,
                    )
                    if cancel_event and cancel_event.is_set():
                        raise CancelledError
                    commit_res = handle.commit(timeout_s=60.0)
                    desc = commit_res.descriptor
                    if isinstance(registrar, _StableDramUploader):
                        _persist_publish_if_needed(
                            desc=desc,
                            options=options,
                            state_dict_to_save=artifact,
                            client=ctl,
                        )
                        state_dict = None
                    else:
                        _persist_publish_if_needed(
                            desc=desc,
                            options=options,
                            state_dict_to_save=state_dict,
                            client=ctl,
                        )
                    return RegistrationResult(
                        state_dict=state_dict,
                        descriptor=desc,
                        lease=None,
                        build=ctx,
                        layout=layout,
                        index_bytes=index_bytes,
                        plan=plan_type,
                        view_id=commit_res.view_id,
                        view_index_json=commit_res.view_index_json,
                        view_data_hash=commit_res.view_data_hash,
                        canonical_ranges=commit_res.canonical_ranges,
                        registration_kind=commit_res.registration_kind,
                        allow_partial=commit_res.allow_partial,
                        local_stable_tier=commit_res.local_stable_tier,
                    )

                if isinstance(registrar, _LeaseUploader):
                    _ = registrar.upload(
                        artifact=artifact,
                        ctx=ctx,
                        layout=layout,
                        handle=handle,
                        handshake=hs,
                        cancel_event=cancel_event,
                    )
                    if cancel_event and cancel_event.is_set():
                        raise CancelledError
                    commit_res = handle.commit(timeout_s=60.0)
                    desc = commit_res.descriptor
                    _persist_publish_if_needed(
                        desc=desc,
                        options=options,
                        state_dict_to_save=None,
                        client=ctl,
                    )
                    lease_obj: RegisteredLease = RegisteredLease(
                        registration_id=handle.registration_id,
                        daemon_address=addr,
                        ttl_ms=int(ttl_ms) if ttl_ms and ttl_ms > 0 else 600_000,
                        owner_pid=ctl._get_effective_pid(),
                        client=ctl,
                    )
                    # For lease plans, return original artifact as the state_dict
                    return RegistrationResult(
                        state_dict=artifact,
                        descriptor=desc,
                        lease=lease_obj,
                        build=ctx,
                        layout=layout,
                        index_bytes=index_bytes,
                        plan=plan_type,
                        view_id=commit_res.view_id,
                        view_index_json=commit_res.view_index_json,
                        view_data_hash=commit_res.view_data_hash,
                        canonical_ranges=commit_res.canonical_ranges,
                        registration_kind=commit_res.registration_kind,
                        allow_partial=commit_res.allow_partial,
                        local_stable_tier=commit_res.local_stable_tier,
                    )
            except CancelledError:
                with contextlib.suppress(Exception):
                    handle.abort(timeout_s=5.0)
                raise

    raise InvalidPlan(f"Unknown plan: {plan_type}")


def register_artifact(
    artifact: dict[str, torch.Tensor],
    *,
    options: RegisterArtifactOptions,
    device_id: int | torch.device | None = None,
    ttl_ms: int | None = None,
) -> RegistrationResult:
    """Unified high-level API returning a structured result.

    - For Coalesced/UMA/VS: returns destination state dict and descriptor.
    - For Lease (in_place according to options): returns original artifact as state_dict and descriptor.
    """
    return _register_artifact_core(
        artifact=artifact,
        options=options,
        device_id=device_id,
        ttl_ms=ttl_ms,
        force_lease_in_place=False,
        prevalidate_disk=True,
    )
