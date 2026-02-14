#  Copyright (c) 2026, TensorCast Team.

import os

from safetensors.torch import load_file

import tensorcast as tc
from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig

# tc.init(global_store_address="100.102.113.111:50051", mode="create")
tc.init(mode="connect")


model_name = "qwen3-0.6b-base"
weight_version = 123

cfg = WeightPublisherConfig(
    model_name=model_name,
    key_template="model:{model_name}:v{weight_version}",
    # Keep replicas resident in local stable DRAM for repeated local get().
    policy="pinned",
    overflow_policy="reject",
    wait_persistence=True,  # default
    keep_last=2,  # keep rollback window
    history_path="/tmp/weights_history.json",
    trigger_reload=False,  # publish only
)

publisher = WeightPublisher(cfg)

# Tensors can be either:
# - all CUDA tensors on the same device (recommended), or
# - all CPU tensors (Tensorcast will stage them to CUDA during `put`).
# tensors = {
#     "transformer.wte.weight": torch.empty((10, 10), device="cuda:0"),
# }
# load tensros from safetensors: /mnt/step3-alignment/inference/Qwen3-0.6B-Base

tensors = {}
weights_path = "/mnt/step3-alignment/inference/Qwen3-0.6B-Base"
for filename in os.listdir(weights_path):
    if filename.endswith(".safetensors"):
        file_path = os.path.join(weights_path, filename)
        # The load_file function returns a dictionary (state_dict) of tensors
        # For a single file, you can just load it directly.
        # For multiple files that are part of the same model, you might need to merge them.
        loaded_tensors = load_file(file_path, device="cpu")  # specify 'cuda' if needed
        tensors.update(loaded_tensors)

artifact_id = publisher.publish(tensors, version=123)
print("published", artifact_id)
