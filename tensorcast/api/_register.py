#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import atexit
import contextlib
import logging
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import grpc
import torch
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from tensorcast import startup
from tensorcast._C import (
    get_cuda_memory_handle,
    get_cuda_memory_ptr,
    restore_tensors,
)

if TYPE_CHECKING:  # for static type checkers only
    from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.api._config import (
    PlanType,
    RegisterArtifactOptions,
    get_global_store_address,
)
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
    build_v2_index_bytes,
)
from tensorcast.api._io_disk import save_dict
from tensorcast.api._utils import validate_disk_index_matches
from tensorcast.observability.otel import ensure_client_otel
from tensorcast.proto.global_store.v1 import global_store_pb2 as gs_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2_grpc as gs_pb2_grpc
from tensorcast.types import (
    ArtifactDescriptor,
    CoalescedHandshake,
    CoalescedPlan,
    Handshake,
    LeasePlan,
    LeaseSegment,
)

# Internal alignment hint for layout computation.
# Currently unused by the underlying calculator but kept for API clarity.
DEFAULT_ALIGN = 1


@dataclass
class RegistrationResult:
    """Unified result for registration APIs.

    - state_dict: destination state dict for coalesced/UMA/VS; original artifact for lease
    - descriptor: content-addressed artifact descriptor
    - lease: post-commit lease handle when using lease-in-place; otherwise None
    """

    state_dict: dict[str, torch.Tensor] | None
    descriptor: ArtifactDescriptor
    lease: RegisteredLease | None


logger = logging.getLogger(__name__)


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
                "Call tensorcast.startup.init() before using registration APIs."
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


def begin_register_artifact_sdk(
    *,
    device_id: int,
    total_size_bytes: int,
    ttl_ms: int | None,
    tensor_index_data: bytes,
    plan: CoalescedPlan | LeasePlan,
) -> tuple[RegisteredArtifact, Handshake]:
    runtime_ctx = startup.require_initialized()
    ctl = runtime_ctx.client
    daemon_addr = runtime_ctx.address
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
    handle = RegisteredArtifact(
        out.registration_id, daemon_addr, ttl_ms=ttl_ms or 0, client=ctl
    )
    return handle, out.handshake


def _gs_stub(addr: str | None = None) -> gs_pb2_grpc.GlobalStoreServiceStub:
    target = addr or get_global_store_address()
    channel = grpc.insecure_channel(target)
    return gs_pb2_grpc.GlobalStoreServiceStub(channel)


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
    try:
        if descriptor is not None and client is not None:
            ok = client.publish_replica_key(
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
        # log-less helper (API layer will log)
        pass


def _persist_publish_if_needed(
    *,
    desc: ArtifactDescriptor,
    options: RegisterArtifactOptions,
    state_dict_to_save: dict[str, torch.Tensor] | None,
    client: DaemonCtl,
) -> None:
    if options.disk_path is not None and options.disk_path.strip() == "":
        try:
            from tensorcast.client_runtime import storage_root_default

            root = storage_root_default()
            if not root:
                raise TensorCastError(
                    "ClientConfig.storage.default_root is required when disk_path==''"
                )
            out_dir = Path(root) / desc.artifact_id
            if state_dict_to_save is not None:
                save_dict(state_dict_to_save, str(out_dir))
            _upsert_key_mapping_if_needed(
                key=options.key,
                artifact_id=desc.artifact_id,
                disk_path=str(out_dir),
                descriptor=desc,
                client=client,
            )
        except Exception:
            pass
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
    ) -> None:
        self.device_id = int(device_id)
        self.input_mode = input_mode  # "cpu" or "cuda"
        self.tensor_meta_index = tensor_meta_index
        self.tensor_source_index = tensor_source_index

    @staticmethod
    def from_artifact(
        artifact: dict[str, torch.Tensor], device_id: int | torch.device | None
    ) -> "BuildContext":
        if device_id is None:
            dev_index: int | None = None
            for t in artifact.values():
                if not t.is_cuda:
                    raise DeviceMismatch(
                        "When device_id is None, all tensors must be CUDA tensors on the same device"
                    )
                this_idx = t.device.index if t.device.index is not None else 0
                if dev_index is None:
                    dev_index = this_idx
                elif this_idx != dev_index:
                    raise DeviceMismatch(
                        "All CUDA tensors must be on the same device when inferring device_id"
                    )
            if dev_index is None:
                raise TensorCastError(
                    "artifact is empty or has no tensors to infer device from"
                )
            target_device_id = int(dev_index)
            input_mode = "cuda"
        else:
            target_device_id = resolve_device(device_id)
            for t in artifact.values():
                if t.is_cuda:
                    raise DeviceMismatch(
                        "When device_id is specified, artifact must contain CPU tensors only"
                    )
            input_mode = "cpu"

        tensor_meta_index: TensorMetaIndex = {}
        tensor_source_index: TensorDataIndex = {}
        for name, t in artifact.items():
            storage = t.untyped_storage()
            tensor_source_index[name] = (int(storage.data_ptr()), int(storage.size()))
            tensor_meta_index[name] = (
                list(map(int, t.shape)),
                list(map(int, t.stride())),
                str(t.dtype),
                int(t.storage_offset()),
            )

        return BuildContext(
            device_id=target_device_id,
            input_mode=input_mode,
            tensor_meta_index=tensor_meta_index,
            tensor_source_index=tensor_source_index,
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
        from ._indices import calculate_tensor_device_offsets

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


class IndexV2:
    @staticmethod
    def build_bytes(
        tensor_meta_index: TensorMetaIndex,
        tensor_source_index: TensorDataIndex,
        tensor_device_offsets: TensorDeviceOffsets,
        device_id: int,
    ) -> bytes:
        return build_v2_index_bytes(
            tensor_meta_index,
            tensor_source_index,
            tensor_device_offsets,
            int(device_id),
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
    index_bytes = IndexV2.build_bytes(
        ctx.tensor_meta_index, ctx.tensor_source_index, layout.offsets, ctx.device_id
    )
    return ctx, layout, index_bytes


def make_plan_model(
    options: RegisterArtifactOptions, total_size_bytes: int | None = None
) -> CoalescedPlan | LeasePlan:
    plan_type = cast(PlanType, options.plan)
    if plan_type is PlanType.VRAM_COALESCED:
        return CoalescedPlan(
            kind="coalesced",
            max_inflight_bytes=options.max_inflight_bytes,
            release_on_tensor_commit=options.release_on_tensor_commit,
        )
    if plan_type is PlanType.VRAM_LEASED:
        # Current release only supports Lease-In-Place (LIP)
        in_place = bool(getattr(options, "lease_in_place", False))
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
        for name, src in artifact.items():
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
    ) -> dict[str, torch.Tensor]:
        def _export_cuda_ipc_handle(ptr: int) -> bytes:
            return get_cuda_memory_handle(ctx.device_id, int(ptr))

        segments: list[LeaseSegment] = []
        chunks = list(layout.unique_chunks)
        chunks.sort(key=lambda x: int(x[2]))
        for src_off, size_bytes, dst_off, _stream in chunks:
            handle_bytes = _export_cuda_ipc_handle(int(src_off))
            segments.append(
                LeaseSegment(
                    device_id=int(ctx.device_id),
                    cuda_ipc_handle=handle_bytes,
                    base_addr=0,
                    length=int(size_bytes),
                    dst_offset=int(dst_off),
                )
            )
        ctl = handle.client
        ok = ctl.feed_register_artifact_lease_segments(handle.registration_id, segments)
        if not ok:
            raise FeedFailed("Lease segments feed failed")
        return artifact


PLAN_REGISTRY: dict[PlanType, object] = {
    PlanType.VRAM_COALESCED: _CoalescedUploader(),
    PlanType.VRAM_LEASED: _LeaseUploader(),
}


def _register_artifact_core(
    *,
    artifact: dict[str, torch.Tensor],
    options: RegisterArtifactOptions,
    device_id: int | torch.device | None,
    ttl_ms: int | None,
    force_lease_in_place: bool = False,
    prevalidate_disk: bool = True,
) -> RegistrationResult:
    runtime_ctx = startup.require_initialized()
    addr = runtime_ctx.address

    if not artifact:
        raise TensorCastError("artifact must not be empty")

    ctx, layout, index_bytes = _prepare_build(artifact, device_id)

    if (
        prevalidate_disk
        and options.disk_path is not None
        and options.disk_path.strip() != ""
    ):
        cand = Path(options.disk_path)
        if cand.exists() and cand.is_dir():
            validate_disk_index_matches(index_bytes, str(cand))

    ensure_client_otel("tensorcast-client", role="client")
    tracer = trace.get_tracer(__name__)

    if force_lease_in_place:
        plan_type = PlanType.VRAM_LEASED
        plan_model = LeasePlan(
            kind="lease",
            min_tensor_bytes=options.min_tensor_bytes,
            max_tensor_count=options.max_tensor_count,
            lease_bytes_limit=options.lease_bytes_limit,
            in_place=True,
        )
    else:
        plan_type = cast(PlanType, options.plan)
        plan_model = make_plan_model(options, layout.total_size)

    # Plan input-mode constraints
    if plan_type is PlanType.VRAM_LEASED and ctx.input_mode != "cuda":
        raise DeviceMismatch(
            "vram_leased plan requires CUDA tensors (device_id must be inferred)"
        )

    span_names = {
        PlanType.VRAM_COALESCED: "Client/RegisterArtifact.Coalesced",
        PlanType.VRAM_LEASED: "Client/RegisterArtifact.Lease",
    }

    with tracer.start_as_current_span(span_names[plan_type], kind=SpanKind.INTERNAL):
        handle, hs = begin_register_artifact_sdk(
            device_id=ctx.device_id,
            total_size_bytes=layout.total_size,
            ttl_ms=ttl_ms,
            tensor_index_data=index_bytes,
            plan=plan_model,
        )

        registrar = PLAN_REGISTRY[plan_type]
        with handle:
            # Upload per plan
            if isinstance(registrar, _CoalescedUploader):
                state_dict = registrar.upload(
                    artifact=artifact,
                    ctx=ctx,
                    layout=layout,
                    handle=handle,
                    handshake=hs,
                )
                commit_res = handle.commit(timeout_s=60.0)
                desc = commit_res.descriptor
                _persist_publish_if_needed(
                    desc=desc,
                    options=options,
                    state_dict_to_save=state_dict,
                    client=runtime_ctx.client,
                )
                return RegistrationResult(
                    state_dict=state_dict, descriptor=desc, lease=None
                )

            if isinstance(registrar, _LeaseUploader):
                _ = registrar.upload(
                    artifact=artifact,
                    ctx=ctx,
                    layout=layout,
                    handle=handle,
                    handshake=hs,
                )
                commit_res = handle.commit(timeout_s=60.0)
                desc = commit_res.descriptor
                _persist_publish_if_needed(
                    desc=desc,
                    options=options,
                    state_dict_to_save=None,
                    client=runtime_ctx.client,
                )
                ctl = runtime_ctx.client
                lease_obj: RegisteredLease = RegisteredLease(
                    registration_id=handle.registration_id,
                    daemon_address=addr,
                    ttl_ms=int(ttl_ms) if ttl_ms and ttl_ms > 0 else 600_000,
                    owner_pid=ctl._get_effective_pid(),
                    client=ctl,
                )
                # For lease plans, return original artifact as the state_dict
                return RegistrationResult(
                    state_dict=artifact, descriptor=desc, lease=lease_obj
                )

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
