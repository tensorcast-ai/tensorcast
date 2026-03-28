#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import grpc
import pytest
import torch

from tensorcast import ArtifactError, GetArtifactOptions, artifact, put, startup
from tensorcast.global_store.config.settings import GlobalStoreConfig, set_config
from tensorcast.global_store.grpc_service import (
    GlobalStoreServicer,
    register_global_store_servicers,
)
from tests.python.utils.daemon import start_daemon_binary
from tests.python.utils.ports import get_free_port

pytestmark = pytest.mark.requires_cuda_or_fake


def _target_device() -> str:
    if os.environ.get("TENSORCAST_CUDA_BACKEND") == "fake":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _stop_proc(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _make_demo_tensors() -> dict[str, torch.Tensor]:
    return {
        "model.embed_tokens.weight": torch.arange(
            0, 8 * 16, dtype=torch.float32
        ).reshape(8, 16),
        "model.layers.0.self_attn.q_proj.weight": torch.arange(
            0, 16 * 16, dtype=torch.float32
        ).reshape(16, 16),
        "model.layers.0.self_attn.k_proj.weight": torch.arange(
            0, 12 * 16, dtype=torch.float32
        ).reshape(12, 16),
        "model.layers.0.mlp.gate_proj.weight": torch.arange(
            0, 16 * 8, dtype=torch.float32
        ).reshape(16, 8),
        "model.layers.0.mlp.down_proj.weight": torch.arange(
            0, 16 * 4, dtype=torch.float32
        ).reshape(16, 4),
        "model.norm.weight": torch.arange(0, 16, dtype=torch.float32),
    }


def _pick_split_dim(shape: tuple[int, ...], tp: int) -> int | None:
    if len(shape) < 2:
        return None
    candidate_dims = [
        dim for dim, extent in enumerate(shape) if extent >= tp and extent % tp == 0
    ]
    if not candidate_dims:
        return None
    max_extent = max(shape[dim] for dim in candidate_dims)
    for dim in reversed(candidate_dims):
        if shape[dim] == max_extent:
            return dim
    return None


def _build_tp_slices(
    tensors: dict[str, torch.Tensor],
    *,
    tp: int,
    tp_rank: int,
) -> dict[str, list[tuple[int, slice]]]:
    if tp == 1:
        return {}
    slices: dict[str, list[tuple[int, slice]]] = {}
    for name, tensor in tensors.items():
        split_dim = _pick_split_dim(tuple(tensor.shape), tp)
        if split_dim is None:
            continue
        extent = int(tensor.shape[split_dim])
        shard_size = extent // tp
        start = tp_rank * shard_size
        stop = start + shard_size
        slices[name] = [(split_dim, slice(start, stop, None))]
    return slices


def _apply_slices(
    tensor: torch.Tensor,
    view_ops: list[tuple[int, slice]],
) -> torch.Tensor:
    if not view_ops:
        return tensor
    sliced = tensor
    for dim, part in view_ops:
        index = [slice(None)] * sliced.ndim
        index[dim] = part
        sliced = sliced[tuple(index)]
    return sliced


def test_tp_view_get_across_two_real_daemons_with_global_store(tmp_path: Path) -> None:
    set_config(GlobalStoreConfig())
    gs_servicer = GlobalStoreServicer()
    gs_server = grpc.server(ThreadPoolExecutor(max_workers=8))
    register_global_store_servicers(gs_server, gs_servicer)
    gs_port = gs_server.add_insecure_port("127.0.0.1:0")
    if gs_port <= 0:
        raise RuntimeError("failed to bind Global Store server port")
    gs_server.start()

    gs_addr = f"127.0.0.1:{gs_port}"
    daemon_a_addr = f"127.0.0.1:{get_free_port()}"
    daemon_b_addr = f"127.0.0.1:{get_free_port()}"
    device_selector = _target_device()
    cpu_target = device_selector == "cpu"

    daemon_a_local_handle_socket_path: str | None = None
    daemon_b_local_handle_socket_path: str | None = None
    if cpu_target:
        daemon_a_local_handle_socket_path = str(tmp_path / "daemon_a_local_handle.sock")
        daemon_b_local_handle_socket_path = str(tmp_path / "daemon_b_local_handle.sock")

    daemon_a_proc = start_daemon_binary(
        daemon_a_addr,
        tmp_path / "daemon_a_storage",
        daemon_id=f"tp-view-src-{uuid4().hex}",
        global_store_addr=gs_addr,
        p2p_port=get_free_port(),
        cpu_shared_memory_enabled=cpu_target,
        local_handle_socket_path=daemon_a_local_handle_socket_path,
        stable_bytes=64 * 1024 * 1024,
    )
    daemon_b_proc = start_daemon_binary(
        daemon_b_addr,
        tmp_path / "daemon_b_storage",
        daemon_id=f"tp-view-dst-{uuid4().hex}",
        global_store_addr=gs_addr,
        p2p_port=get_free_port(),
        cpu_shared_memory_enabled=cpu_target,
        local_handle_socket_path=daemon_b_local_handle_socket_path,
        stable_bytes=64 * 1024 * 1024,
    )

    key = f"model:test-dual-daemon-tp:{uuid4().hex}"
    tensors = _make_demo_tensors()

    try:
        startup.init(mode="connect", address=daemon_a_addr)
        try:
            put(tensors, key=key, policy="pinned")
        finally:
            startup.shutdown()

        startup.init(mode="connect", address=daemon_b_addr)
        try:
            with pytest.raises(ArtifactError):
                artifact(key=key).tensor_dict(
                    device=device_selector,
                    options=GetArtifactOptions(source="local_only"),
                )

            handle = artifact(key=key)
            materialize_options = GetArtifactOptions(
                source={"preference": "prefer_p2p", "allow_p2p": True, "allow_disk": False},
                verify_checksums=False,
            )

            tp = 4
            for tp_rank in range(tp):
                rank_slices = _build_tp_slices(tensors, tp=tp, tp_rank=tp_rank)
                loaded = handle.view(slices=rank_slices).tensor_dict(
                    device=device_selector,
                    options=materialize_options,
                )
                for name, original in tensors.items():
                    expected = _apply_slices(original, rank_slices.get(name, []))
                    actual = loaded[name]
                    expected_cpu = expected.detach().cpu()
                    actual_cpu = actual.detach().cpu()
                    assert torch.equal(actual_cpu, expected_cpu), (
                        f"tensor mismatch on rank={tp_rank}, name={name}, "
                        f"actual_shape={tuple(actual_cpu.shape)}, expected_shape={tuple(expected_cpu.shape)}"
                    )
        finally:
            startup.shutdown()
    finally:
        _stop_proc(daemon_a_proc)
        _stop_proc(daemon_b_proc)
        gs_server.stop(grace=None)
