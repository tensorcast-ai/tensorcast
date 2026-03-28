#  Copyright (c) 2025-2026, TensorCast Team.

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import grpc
import pytest
import torch
from safetensors.torch import save_file as st_save

from tensorcast import GetArtifactOptions, from_disk, startup
from tensorcast.global_store.composite_stub import GlobalStoreCompositeStub
from tensorcast.global_store.config.settings import GlobalStoreConfig, set_config
from tensorcast.global_store.grpc_service import (
    GlobalStoreServicer,
    register_global_store_servicers,
)
from tensorcast.proto.global_store.v1 import global_store_pb2
from tensorcast.testing.io_disk import save_dict
from tests.python.utils.daemon import start_daemon_binary
from tests.python.utils.ports import get_free_port

pytestmark = pytest.mark.requires_cuda_or_fake


def test_shared_storage_roundtrip(tmp_path):
    """Ensure tensors that share underlying storage round-trip correctly."""

    tmp_path = Path(tmp_path)

    # -----------------------
    # Shared group 1
    # -----------------------
    base1 = torch.arange(1000, dtype=torch.float32)
    view1_a = base1.view(100, 10)
    view1_b = base1[:500]  # smaller slice – still same storage

    # -----------------------
    # Shared group 2
    # -----------------------
    base2 = torch.linspace(0, 1, 2000, dtype=torch.float32)
    view2 = base2.view(400, 5)

    # -----------------------
    # Independent tensors (no shared storage)
    # -----------------------
    indep1 = torch.randn(123, dtype=torch.float32)
    indep2 = torch.randn(64, 64, dtype=torch.float32)

    state_dict = {
        "base1": base1,
        "view1_a": view1_a,
        "view1_b": view1_b,
        "base2": base2,
        "view2": view2,
        "indep1": indep1,
        "indep2": indep2,
    }

    storage_root = tmp_path / "daemon-storage"
    save_path = storage_root / "artifact"
    # Save using the unified writer
    descriptor = save_dict(state_dict, str(save_path))

    listen = f"127.0.0.1:{get_free_port()}"
    cpu_target = os.environ.get("TENSORCAST_CUDA_BACKEND") == "fake"
    local_handle_socket_path: str | None = None
    if cpu_target:
        local_handle_dir = Path(tempfile.mkdtemp(prefix="tc_local_handle_"))
        local_handle_socket_path = str(local_handle_dir / "local_handle.sock")
    set_config(GlobalStoreConfig())
    gs_servicer = GlobalStoreServicer()
    gs_server = grpc.server(ThreadPoolExecutor(max_workers=4))
    register_global_store_servicers(gs_server, gs_servicer)
    gs_port = gs_server.add_insecure_port("127.0.0.1:0")
    if gs_port <= 0:
        raise RuntimeError("failed to bind Global Store server port")
    gs_server.start()
    try:
        channel = grpc.insecure_channel(f"127.0.0.1:{gs_port}")
        stub = GlobalStoreCompositeStub(channel)
        try:
            info = stub.GetServerInfo(global_store_pb2.GetServerInfoRequest())
            cluster_id = info.cluster_id
            rel_path = save_path.relative_to(storage_root)
            resp = stub.UpsertArtifactDiskLocation(
                global_store_pb2.UpsertArtifactDiskLocationRequest(
                    artifact_id=descriptor["artifact_id"],
                    cluster_id=cluster_id,
                    relative_path=str(rel_path),
                    kind=global_store_pb2.DISK_LOCATION_KIND_MANAGED,
                )
            )
            if resp.status != global_store_pb2.Status.STATUS_OK:
                raise RuntimeError("failed to upsert artifact disk location")
        finally:
            channel.close()

        daemon_proc = start_daemon_binary(
            listen,
            storage_root,
            cpu_shared_memory_enabled=cpu_target,
            local_handle_socket_path=local_handle_socket_path,
            stable_bytes=64 * 1024 * 1024,
            global_store_addr=f"127.0.0.1:{gs_port}",
        )
        try:
            startup.init(mode="connect", address=listen)
            try:
                device_selector = (
                    "cpu"
                    if cpu_target
                    else ("cuda:0" if torch.cuda.is_available() else "cpu")
                )
                artifact_handle = from_disk(str(save_path), verify_checksums=False)
                loaded_state_dict = artifact_handle.tensor_dict(
                    device=device_selector,
                    options=GetArtifactOptions(
                        source="disk_only",
                        verify_checksums=False,
                    ),
                )
            finally:
                startup.shutdown()
        finally:
            try:
                daemon_proc.terminate()
                daemon_proc.wait(timeout=3)
            except Exception:
                pass
    finally:
        gs_server.stop(grace=None)

    # For value comparisons, normalize to CPU to avoid device mismatch errors
    loaded_for_compare: dict[str, torch.Tensor] = {
        k: (v.cpu() if v.is_cuda else v) for k, v in loaded_state_dict.items()
    }

    # -----------------------
    # (1) Value equality check for every tensor
    # -----------------------
    for name, original in state_dict.items():
        assert torch.equal(original, loaded_for_compare[name]), (
            f"Tensor content mismatch for {name}, {original} != {loaded_for_compare[name]}"
        )

    # -----------------------
    # (2) Storage sharing semantics helpers
    # -----------------------
    def assert_shared(names: Sequence[str]):
        """Assert tensors referenced by `names` share storage in `loaded_state_dict`."""
        ptrs = {loaded_state_dict[n].storage().data_ptr() for n in names}
        assert len(ptrs) == 1, (
            f"Tensors {names} are expected to share storage after load but found {len(ptrs)} distinct storages"
        )

    # Group 1 should share
    assert_shared(["base1", "view1_a", "view1_b"])
    # Group 2 should share
    assert_shared(["base2", "view2"])

    # Groups should not share with each other
    ptr_group1 = loaded_state_dict["base1"].storage().data_ptr()
    ptr_group2 = loaded_state_dict["base2"].storage().data_ptr()
    assert ptr_group1 != ptr_group2, (
        "Separate storage groups share the same backing storage unexpectedly"
    )

    # Independent tensors should each have unique storage
    indep_ptrs = {
        loaded_state_dict["indep1"].storage().data_ptr(),
        loaded_state_dict["indep2"].storage().data_ptr(),
    }
    assert len(indep_ptrs) == 2
    # And they should not collide with shared groups
    assert ptr_group1 not in indep_ptrs
    assert ptr_group2 not in indep_ptrs


def test_from_disk_tensor_dict_without_global_store(tmp_path):
    tmp_path = Path(tmp_path)
    storage_root = tmp_path / "daemon-storage"
    save_path = storage_root / "artifact"
    expected = {"weights": torch.arange(32, dtype=torch.float32)}
    save_dict(expected, str(save_path))

    listen = f"127.0.0.1:{get_free_port()}"
    cpu_target = os.environ.get("TENSORCAST_CUDA_BACKEND") == "fake"
    local_handle_socket_path: str | None = None
    if cpu_target:
        local_handle_dir = Path(tempfile.mkdtemp(prefix="tc_local_handle_"))
        local_handle_socket_path = str(local_handle_dir / "local_handle.sock")

    daemon_proc = start_daemon_binary(
        listen,
        storage_root,
        cpu_shared_memory_enabled=cpu_target,
        local_handle_socket_path=local_handle_socket_path,
        stable_bytes=64 * 1024 * 1024,
    )
    try:
        startup.init(mode="connect", address=listen)
        try:
            artifact_handle = from_disk(str(save_path), verify_checksums=False)
            device_selector = (
                "cpu"
                if cpu_target
                else ("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            loaded = artifact_handle.tensor_dict(
                device=device_selector,
                options=GetArtifactOptions(
                    source="disk_only",
                    verify_checksums=False,
                ),
            )
        finally:
            startup.shutdown()
    finally:
        try:
            daemon_proc.terminate()
            daemon_proc.wait(timeout=3)
        except Exception:
            pass

    loaded_cpu = (
        loaded["weights"].cpu() if loaded["weights"].is_cuda else loaded["weights"]
    )
    assert torch.equal(loaded_cpu, expected["weights"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_from_disk_bind_and_bind_into_respect_safetensors_source_layout(
    tmp_path,
):
    tmp_path = Path(tmp_path)
    storage_root = tmp_path / "daemon-storage"
    save_path = storage_root / "artifact"
    save_path.mkdir(parents=True, exist_ok=True)
    expected = torch.arange(32, dtype=torch.bfloat16)
    st_save({"weights": expected}, str(save_path / "weights.safetensors"))
    compat_libs = [
        "/data/cuda/compat",
        "/data/cuda/cuda-12.8/lib64",
        "/usr/local/nvidia/lib64",
    ]
    available_compat_libs = [path for path in compat_libs if Path(path).exists()]
    if available_compat_libs:
        current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        compat_ld_library_path = ":".join(available_compat_libs)
        if current_ld_library_path:
            os.environ["LD_LIBRARY_PATH"] = (
                f"{compat_ld_library_path}:{current_ld_library_path}"
            )
        else:
            os.environ["LD_LIBRARY_PATH"] = compat_ld_library_path

    listen = f"127.0.0.1:{get_free_port()}"
    local_handle_dir = tmp_path / "local-handle"
    local_handle_dir.mkdir(parents=True, exist_ok=True)
    daemon_proc = start_daemon_binary(
        listen,
        storage_root,
        stable_bytes=64 * 1024 * 1024,
        local_handle_socket_path=str(local_handle_dir / "local_handle.sock"),
    )
    try:
        startup.init(mode="connect", address=listen)
        try:
            artifact_handle = from_disk(str(save_path), verify_checksums=False)
            ref = artifact_handle.tensor_dict(device="cuda:0")["weights"]

            binding = artifact_handle.bind(
                device="cuda:0",
                packing="byte_space",
                options=GetArtifactOptions(
                    source="disk_only",
                    verify_checksums=False,
                ),
            )
            try:
                bound = dict(binding.tensors)["weights"]
                torch.cuda.synchronize()
                assert torch.equal(bound.cpu(), expected)
                assert torch.equal(bound, ref)
            finally:
                binding.close()

            target = torch.empty_like(ref)
            binding = artifact_handle.bind_into(
                {"weights": target}, packing="byte_space"
            )
            try:
                torch.cuda.synchronize()
                assert torch.equal(target.cpu(), expected)
                assert torch.equal(target, ref)
            finally:
                binding.close()
        finally:
            startup.shutdown()
    finally:
        try:
            daemon_proc.terminate()
            daemon_proc.wait(timeout=3)
        except Exception:
            pass


if __name__ == "__main__":
    test_shared_storage_roundtrip(tmp_path="/tmp/test_shared_storage")
