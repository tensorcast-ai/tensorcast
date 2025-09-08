#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import atexit
import threading
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, cast

import grpc
import torch
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from tensorcast._C import (
    get_cuda_memory_handle,
    get_cuda_memory_ptr,
    restore_tensors,
)
from tensorcast.daemon_ctl import get_daemon_client

if TYPE_CHECKING:  # for static type checkers only
    from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.observability.otel import ensure_client_otel
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

from ._config import (
    PlanType,
    RegisterArtifactOptions,
    get_daemon_address,
    get_global_store_address,
)
from ._device import resolve_device
from ._errors import DeviceMismatch, FeedFailed, InvalidPlan, TensorCastError
from ._indices import (
    TensorDataIndex,
    TensorDeviceOffsets,
    TensorMetaIndex,
    build_v2_index_bytes,
)
from ._io_disk import save_dict
from ._utils import validate_disk_index_matches

# Internal alignment hint for layout computation.
# Currently unused by the underlying calculator but kept for API clarity.
DEFAULT_ALIGN = 1


class RegisteredArtifact:
    def __init__(
        self,
        registration_id: str,
        daemon_address: str,
        *,
        ttl_ms: int | None = None,
        client: "DaemonCtl" | None = None,
    ) -> None:
        import threading

        self.registration_id = registration_id
        self._addr = daemon_address
        self._ttl_ms = int(ttl_ms) if ttl_ms and ttl_ms > 0 else 0
        self._ka_thread: threading.Thread | None = None
        self._ka_stop = threading.Event()
        self._epoch: int = 0
        # Cache client for this handle's lifetime
        self._ctl = client or get_daemon_client(self._addr)

    @property
    def client(self) -> "DaemonCtl":
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

    def commit(self, timeout_s: float = 60.0) -> ArtifactDescriptor:
        desc = self.client.commit_registered_artifact(
            self.registration_id, timeout_s=timeout_s
        )
        self.__exit__(None, None, None)
        return desc

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
        self, registration_id: str, daemon_address: str, ttl_ms: int, owner_pid: int
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

                ctl = DaemonCtl(self._addr)
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
            ctl = DaemonCtl(self._addr)
            ctl.revoke_registered_artifact(self.registration_id)
        except Exception:  # noqa: BLE001
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


def register_artifact_lease_in_place(
    artifact: dict[str, torch.Tensor],
    *,
    options: RegisterArtifactOptions,
    ttl_ms: int | None = None,
    daemon_address: str | None = None,
) -> tuple[ArtifactDescriptor, RegisteredLease]:
    """Register a CUDA artifact as a Lease-In-Place (LIP) and return a lease handle.

    Returns a content-addressed descriptor and a RegisteredLease context manager
    that maintains post-Commit keepalives and performs best-effort revoke on exit.
    """
    if not artifact:
        raise TensorCastError("artifact must not be empty")

    # Build context & layout, then index bytes (CUDA input required)
    ctx = BuildContext.from_artifact(artifact, device_id=None)
    if ctx.input_mode != "cuda":
        raise DeviceMismatch(
            "lease_in_place requires CUDA tensors without explicit device_id"
        )
    layout = CoalescedLayout.compute(
        ctx.tensor_source_index, ctx.device_id, align=DEFAULT_ALIGN
    )
    index_bytes = IndexV2.build_bytes(
        ctx.tensor_meta_index, ctx.tensor_source_index, layout.offsets, ctx.device_id
    )

    ensure_client_otel("tensorcast-client", role="client")
    addr = daemon_address or get_daemon_address()
    # Force in_place regardless of caller option to avoid ambiguity
    plan = LeasePlan(
        kind="lease",
        min_tensor_bytes=options.min_tensor_bytes,
        max_tensor_count=options.max_tensor_count,
        lease_bytes_limit=options.lease_bytes_limit,
        in_place=True,
    )

    handle, hs = begin_register_artifact_sdk(
        device_id=ctx.device_id,
        total_size_bytes=layout.total_size,
        ttl_ms=ttl_ms,
        tensor_index_data=index_bytes,
        plan=plan,
        daemon_address=addr,
    )

    registrar = _LeaseUploader()
    with handle:
        _ = registrar.upload(
            artifact=artifact,
            ctx=ctx,
            layout=layout,
            handle=handle,
            handshake=hs,
            daemon_address=addr,
        )
        desc = handle.commit(timeout_s=60.0)
        # Create RegisteredLease tied to the same registration id
        lease = RegisteredLease(
            registration_id=handle.registration_id,
            daemon_address=addr,
            ttl_ms=int(ttl_ms) if ttl_ms and ttl_ms > 0 else 600_000,
            owner_pid=DaemonCtl(
                addr
            )._get_effective_pid(),  # uses host pid if configured
        )
        return desc, lease


def begin_register_artifact_sdk(
    *,
    device_id: int,
    total_size_bytes: int,
    ttl_ms: int | None,
    tensor_index_data: bytes,
    plan: CoalescedPlan | DVMPPlan | LeasePlan,
    daemon_address: str,
) -> tuple[RegisteredArtifact, Handshake]:
    ctl = get_daemon_client(daemon_address)
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
        out.registration_id, daemon_address, ttl_ms=ttl_ms or 0, client=ctl
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
) -> None:
    if not key:
        return
    try:
        if descriptor is not None:
            ctl = get_daemon_client(get_daemon_address())
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
        # log-less helper (API layer will log)
        pass


def _persist_publish_if_needed(
    *,
    desc: ArtifactDescriptor,
    options: RegisterArtifactOptions,
    state_dict_to_save: dict[str, torch.Tensor] | None,
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
            )
        except Exception:
            pass
    else:
        _upsert_key_mapping_if_needed(
            key=options.key,
            artifact_id=desc.artifact_id,
            disk_path=options.disk_path,
            descriptor=desc,
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


def make_plan_model(
    options: RegisterArtifactOptions, total_size_bytes: int | None = None
) -> CoalescedPlan | DVMPPlan | LeasePlan:
    plan_type = cast(PlanType, options.plan)
    if plan_type is PlanType.VRAM_COALESCED:
        return CoalescedPlan(
            kind="coalesced",
            max_inflight_bytes=options.max_inflight_bytes,
            release_on_tensor_commit=options.release_on_tensor_commit,
        )
    if plan_type is PlanType.DVMP:
        return DVMPPlan(
            kind="dvmp",
            preferred_channel=int(options.dvmp_preferred_channel),  # pyright: ignore[reportArgumentType]
            ring_bytes=int(options.dvmp_ring_bytes),
        )
    if plan_type is PlanType.VRAM_LEASED:
        return LeasePlan(
            kind="lease",
            min_tensor_bytes=options.min_tensor_bytes,
            max_tensor_count=options.max_tensor_count,
            lease_bytes_limit=options.lease_bytes_limit,
            in_place=bool(getattr(options, "lease_in_place", False)),
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
        daemon_address: str,
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


class _DVMPUploader:
    def upload(
        self,
        *,
        artifact: dict[str, torch.Tensor],
        ctx: BuildContext,
        layout: CoalescedLayout,
        handle: RegisteredArtifact,
        handshake: Handshake,
        daemon_address: str,
    ) -> dict[str, torch.Tensor]:
        frames: list[tuple[int, bytes]] = []
        for name in sorted(ctx.tensor_meta_index.keys()):
            dst_off = int(layout.offsets[ctx.device_id][name])
            _ptr, storage_size = ctx.tensor_source_index[name]
            src = artifact[name]
            b = src.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            if len(b) != int(storage_size):
                raise TensorCastError(
                    f"Tensor '{name}' raw byte size mismatch: {len(b)} vs {storage_size}"
                )
            frames.append((int(dst_off), b))

        ctl = handle.client
        cfg = ctl.get_server_config()
        chunk_size = int(cfg.chunk_size) if int(cfg.chunk_size) > 0 else 8 * 1024 * 1024

        ok_all = True
        for off, data_bytes in frames:
            ok = ctl.feed_register_artifact_dvmp_stream_data(
                handle.registration_id,
                data_bytes,
                offset=int(off),
                chunk_size=chunk_size,
                timeout_s=60.0,
            )
            if not ok:
                ok_all = False
                break
        if not ok_all:
            raise FeedFailed("DVMP feed failed")
        return artifact


class _LeaseUploader:
    def upload(
        self,
        *,
        artifact: dict[str, torch.Tensor],
        ctx: BuildContext,
        layout: CoalescedLayout,
        handle: RegisteredArtifact,
        handshake: Handshake,
        daemon_address: str,
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
    PlanType.DVMP: _DVMPUploader(),
    PlanType.VRAM_LEASED: _LeaseUploader(),
}


def register_artifact(
    artifact: dict[str, torch.Tensor],
    *,
    options: RegisterArtifactOptions,
    device_id: int | torch.device | None = None,
    ttl_ms: int | None = None,
    daemon_address: str | None = None,
) -> tuple[dict[str, torch.Tensor], ArtifactDescriptor]:
    if not artifact:
        raise TensorCastError("artifact must not be empty")

    # 1) Build context and layout, then index bytes
    ctx = BuildContext.from_artifact(artifact, device_id)
    layout = CoalescedLayout.compute(
        ctx.tensor_source_index, ctx.device_id, align=DEFAULT_ALIGN
    )
    index_bytes = IndexV2.build_bytes(
        ctx.tensor_meta_index, ctx.tensor_source_index, layout.offsets, ctx.device_id
    )

    # Optional pre-validation against existing disk path
    if options.disk_path is not None and options.disk_path.strip() != "":
        cand = Path(options.disk_path)
        if cand.exists() and cand.is_dir():
            validate_disk_index_matches(index_bytes, str(cand))

    ensure_client_otel("tensorcast-client", role="client")
    tracer = trace.get_tracer(__name__)
    plan_type = cast(PlanType, options.plan)
    addr = daemon_address or get_daemon_address()

    # Plan input-mode constraints
    if plan_type is PlanType.DVMP and ctx.input_mode != "cpu":
        raise DeviceMismatch(
            "dvmp plan requires CPU tensors (specify device_id to enforce CPU input)"
        )
    if plan_type is PlanType.VRAM_LEASED and ctx.input_mode != "cuda":
        raise DeviceMismatch(
            "vram_leased plan requires CUDA tensors (device_id must be inferred)"
        )

    span_names = {
        PlanType.VRAM_COALESCED: "Client/RegisterArtifact.Coalesced",
        PlanType.DVMP: "Client/RegisterArtifact.DVMP",
        PlanType.VRAM_LEASED: "Client/RegisterArtifact.Lease",
    }
    plan_model = make_plan_model(options, layout.total_size)

    with tracer.start_as_current_span(span_names[plan_type], kind=SpanKind.INTERNAL):
        handle, hs = begin_register_artifact_sdk(
            device_id=ctx.device_id,
            total_size_bytes=layout.total_size,
            ttl_ms=ttl_ms,
            tensor_index_data=index_bytes,
            plan=plan_model,
            daemon_address=addr,
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
                    daemon_address=addr,
                )
                desc = handle.commit(timeout_s=60.0)
                _persist_publish_if_needed(
                    desc=desc, options=options, state_dict_to_save=state_dict
                )
                return state_dict, desc
            if isinstance(registrar, _DVMPUploader):
                state_dict = registrar.upload(
                    artifact=artifact,
                    ctx=ctx,
                    layout=layout,
                    handle=handle,
                    handshake=hs,
                    daemon_address=addr,
                )
                desc = handle.commit(timeout_s=60.0)
                _persist_publish_if_needed(
                    desc=desc, options=options, state_dict_to_save=state_dict
                )
                return state_dict, desc
            if isinstance(registrar, _LeaseUploader):
                state_dict = registrar.upload(
                    artifact=artifact,
                    ctx=ctx,
                    layout=layout,
                    handle=handle,
                    handshake=hs,
                    daemon_address=addr,
                )
                desc = handle.commit(timeout_s=60.0)
                _persist_publish_if_needed(
                    desc=desc, options=options, state_dict_to_save=None
                )
                return state_dict, desc

    raise InvalidPlan(f"Unknown plan: {plan_type}")
