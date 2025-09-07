#  Copyright (c) 2025, TensorCast Team.

import os
from pathlib import Path

import torch

from tensorcast.torch_util import (
    RegisterArtifactOptions,
    register_artifact,
    set_global_store_address,
)


def main() -> None:
    # Optionally point SDK to Global Store
    gs_addr = os.environ.get("TENSORCAST_GLOBAL_STORE", "127.0.0.1:8085")
    set_global_store_address(gs_addr)

    # Create a tiny dummy state_dict
    state_dict: dict[str, torch.Tensor] = {
        "linear.weight": torch.randn(16, 16, dtype=torch.float32),
        "linear.bias": torch.randn(16, dtype=torch.float32),
    }

    # Directory to persist artifact files; must exist if provided
    out_dir = Path("./artifacts/demo-key-artifact").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = RegisterArtifactOptions(
        plan="vram_coalesced",
        key="demo:model:001",
        disk_path=str(out_dir),  # validate/record disk source
    )
    _, desc = register_artifact(state_dict, options=opts, device_id=0)
    print("Committed artifact_id:", desc.artifact_id)


if __name__ == "__main__":
    main()
