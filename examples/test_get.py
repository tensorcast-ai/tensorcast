#  Copyright (c) 2026, TensorCast Team.

import tensorcast as tc

tc.init(mode="connect")

device = "cuda:0"
key = "demo:model:002"

handle = tc.artifact(key=key)
loaded = handle.tensor_dict(device=device)
print(loaded)
