#  Copyright (c) 2025, TensorCast Team.

from pathlib import Path

import torch

from tensorcast import startup
from tensorcast.api import Store
from tensorcast.api._config import RegisterArtifactOptions


def main() -> None:
    # Connect to the Store Daemon (assumes tensorcast.startup.init() env/config)
    ctx = startup.init()
    store = Store(ctx.address)

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
        disk_path=str(out_dir),  # validate/record disk source
    )
    device_arg = "cuda:0" if torch.cuda.is_available() else None
    registered = store.put(
        state_dict,
        key="demo:model:001",
        options=opts,
        device=device_arg,
    )
    desc = registered.registration_result.descriptor
    print("Committed artifact_id:", desc.artifact_id)


if __name__ == "__main__":
    main()
