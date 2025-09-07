#  Copyright (c) 2025, TensorCast Team.

import os

from tensorcast.api._config import GetArtifactOptions, set_global_store_address
from tensorcast.api._loader import get_artifact_sync


def main() -> None:
    gs_addr = os.environ.get("TENSORCAST_GLOBAL_STORE", "127.0.0.1:8085")
    set_global_store_address(gs_addr)

    # Prefer daemon MaterializeByKey path (no client-side fallback)
    opts = GetArtifactOptions(prefer="p2p")
    state_dict = get_artifact_sync(key="demo:model:001", device_id=0, options=opts)

    total_params = sum(int(t.numel()) for t in state_dict.values())
    print("Loaded params:", total_params)


if __name__ == "__main__":
    main()
