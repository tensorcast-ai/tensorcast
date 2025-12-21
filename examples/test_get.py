#  Copyright (c) 2025, TensorCast Team.

import torch

import tensorcast as tc

tc.init(mode="connect", address="127.0.0.1:50052")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
state_dict = tc.get(key="test:model:001", device=device)
print(state_dict)
