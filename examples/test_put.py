#  Copyright (c) 2026, TensorCast Team.

import torch

import tensorcast as tc

tc.init(mode="connect")

device = "cuda:0"
key = "demo:model:002"
state_dict = {"layer.weight": torch.randn(8, 8, device=device)}
expected_cpu = {name: tensor.cpu() for name, tensor in state_dict.items()}

registered = tc.put(state_dict, key=key)
print("artifact_id:", registered.artifact_id)
