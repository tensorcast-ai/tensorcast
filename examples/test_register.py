#  Copyright (c) 2025, TensorCast Team.

import time

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

import tensorcast as tc
from tensorcast import PlanType, RegisterArtifactOptions
from tensorcast.testing.dict import assert_state_dict_equal

hf_model_name = "Qwen/Qwen3-0.6B"

model = AutoModelForCausalLM.from_pretrained(
    hf_model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda:0",
)

# Print every tensor size
# for name, param in artifact.state_dict().items():
#     MB = 1024 * 1024
#     if param.numel() * param.dtype.itemsize > 1 * MB:
#         continue
#     print(f"{name}: {param.numel()}, {param.dtype}")

# print("=" * 100)

state_dict = model.state_dict()

# tc.init(address="127.0.0.1:50052")
opts = RegisterArtifactOptions(plan=PlanType.VRAM_COALESCED, key="test:model:001")
res = tc.register(state_dict, options=opts)
saved_dict = res.state_dict
assert saved_dict is not None

# Validate equality between the original and registered dicts
assert_state_dict_equal(state_dict, saved_dict)
print("All tensors match ✅")


while True:
    time.sleep(1)
