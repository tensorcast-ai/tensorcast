#  Copyright (c) 2026, TensorCast Team.

import torch

import tensorcast as tc
from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig

# tc.init(global_store_address="100.102.113.111:50051", mode="create")
tc.init(mode="connect")

model_name = "llama7b"
weight_version = 123

cfg = WeightPublisherConfig(
    model_name="llama7b",
    key_template="model:{model_name}:v{weight_version}",
    policy="durable",  # default
    wait_persistence=True,  # default
    keep_last=2,  # keep rollback window
    history_path="/tmp/weights_history.json",
    trigger_reload=False,  # publish only
)

publisher = WeightPublisher(cfg)

# Tensors can be either:
# - all CUDA tensors on the same device (recommended), or
# - all CPU tensors (Tensorcast will stage them to CUDA during `put`).
tensors = {
    "transformer.wte.weight": torch.empty((10, 10), device="cuda:0"),
}

artifact_id = publisher.publish(tensors, version=123)
print("published", artifact_id)
