#  Copyright (c) 2025, TensorCast Team.

import os

from tensorcast.torch_util import (
    GetArtifactOptions,
    get_artifact,
    set_global_store_address,
)


def main() -> None:
    gs_addr = os.environ.get("TENSORCAST_GLOBAL_STORE", "127.0.0.1:8085")
    set_global_store_address(gs_addr)

    # Prefer daemon MaterializeByKey path (no client-side fallback)
    opts = GetArtifactOptions(prefer="p2p")
    state_dict = get_artifact(key="demo:model:001", device_id=0, options=opts)

    total_params = sum(int(t.numel()) for t in state_dict.values())  # type: ignore[union-attr]
    print("Loaded params:", total_params)


if __name__ == "__main__":
    main()
