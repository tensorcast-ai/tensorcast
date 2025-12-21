#  Copyright (c) 2025, TensorCast Team.

from pathlib import Path

import torch

import tensorcast as tc
from tensorcast import PlanType, RegisterArtifactOptions


def main() -> None:
    # Connect to the Store Daemon
    tc.init(mode="connect", address="127.0.0.1:50052")

    # Create a tiny dummy state_dict
    state_dict: dict[str, torch.Tensor] = {
        "linear.weight": torch.randn(16, 16, dtype=torch.float32),
        "linear.bias": torch.randn(16, dtype=torch.float32),
    }

    # Directory to persist artifact files; must exist if provided
    out_dir = Path("./artifacts/demo-key-artifact").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = RegisterArtifactOptions(
        plan=PlanType.VRAM_COALESCED,
        disk_path=str(out_dir),  # validate/record disk source
    )
    registered = tc.put(
        state_dict,
        key="demo:model:001",
        options=opts,
        device=0,
    )
    print("Committed artifact_id:", registered.artifact_id)


if __name__ == "__main__":
    main()
