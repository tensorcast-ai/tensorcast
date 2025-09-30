#  Copyright (c) 2025, TensorCast Team.

import torch

from tensorcast import startup
from tensorcast.api import FallbackOptions, Store


def main() -> None:
    ctx = startup.init()
    store = Store(ctx.address)

    # Prefer daemon MaterializeByKey path (no client-side disk fallback)
    fallback = FallbackOptions(prefer_disk=False, allow_p2p=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state_dict = store.get(key="demo:model:001", device=device, fallback=fallback)

    total_params = sum(int(t.numel()) for t in state_dict.values())
    print("Loaded params:", total_params)


if __name__ == "__main__":
    main()
