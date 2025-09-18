#  Copyright (c) 2025, TensorCast Team.

from pathlib import Path

import torch

from tensorcast import startup
from tensorcast.api import register_artifact
from tensorcast.api._config import RegisterArtifactOptions


def main() -> None:
    # Connect to the Store Daemon (assumes tensorcast.startup.init() env/config)
    startup.init()

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
    res = register_artifact(state_dict, options=opts, device_id=0)
    desc = res.descriptor
    print("Committed artifact_id:", desc.artifact_id)


if __name__ == "__main__":
    main()
