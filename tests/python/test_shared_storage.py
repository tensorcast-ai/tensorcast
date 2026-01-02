#  Copyright (c) 2025-2026, TensorCast Team.

from pathlib import Path
from typing import Sequence

import pytest
import torch

from tensorcast import FallbackOptions, artifact, startup
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
    daemon_proc = start_daemon_binary(listen, storage_root)
    try:
        startup.init(mode="connect", address=listen)
        try:
            fallback = FallbackOptions(
                disk_path=str(save_path),
                prefer_disk=True,
                allow_p2p=False,
                verify_checksums=False,
            )
            device_selector = "cuda:0" if torch.cuda.is_available() else "cpu"
            loaded_state_dict = artifact(
                artifact_id=descriptor["artifact_id"],
                fallback=fallback,
            ).tensor_dict(device=device_selector)
        finally:
            startup.shutdown()
    finally:
        try:
            daemon_proc.terminate()
            daemon_proc.wait(timeout=3)
        except Exception:
            pass

    # For value comparisons, normalize to CPU to avoid device mismatch errors
    loaded_for_compare: dict[str, torch.Tensor] = {
        k: (v.cpu() if v.is_cuda else v) for k, v in loaded_state_dict.items()
    }

    # -----------------------
    # (1) Value equality check for every tensor
    # -----------------------
    for name, original in state_dict.items():
        assert torch.equal(
            original, loaded_for_compare[name]
        ), f"Tensor content mismatch for {name}, {original} != {loaded_for_compare[name]}"

    # -----------------------
    # (2) Storage sharing semantics helpers
    # -----------------------
    def assert_shared(names: Sequence[str]):
        """Assert tensors referenced by `names` share storage in `loaded_state_dict`."""
        ptrs = {loaded_state_dict[n].storage().data_ptr() for n in names}
        assert (
            len(ptrs) == 1
        ), f"Tensors {names} are expected to share storage after load but found {len(ptrs)} distinct storages"

    # Group 1 should share
    assert_shared(["base1", "view1_a", "view1_b"])
    # Group 2 should share
    assert_shared(["base2", "view2"])

    # Groups should not share with each other
    ptr_group1 = loaded_state_dict["base1"].storage().data_ptr()
    ptr_group2 = loaded_state_dict["base2"].storage().data_ptr()
    assert ptr_group1 != ptr_group2, "Separate storage groups share the same backing storage unexpectedly"

    # Independent tensors should each have unique storage
    indep_ptrs = {
        loaded_state_dict["indep1"].storage().data_ptr(),
        loaded_state_dict["indep2"].storage().data_ptr(),
    }
    assert len(indep_ptrs) == 2
    # And they should not collide with shared groups
    assert ptr_group1 not in indep_ptrs
    assert ptr_group2 not in indep_ptrs

if __name__ == "__main__":
    test_shared_storage_roundtrip(tmp_path="/tmp/test_shared_storage")
