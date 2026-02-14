#  Copyright (c) 2026, TensorCast Team.

import argparse

import tensorcast as tc

tc.init(mode="connect")

device = "cuda:0"
key = "model:qwen3-0.6b-base:v123"

parser = argparse.ArgumentParser()
parser.add_argument("--use_view", action="store_true")
args = parser.parse_args()
handle = tc.artifact(key=key)

if args.use_view:
    tensor_keys = handle.tensor_names
    desc = handle.describe()
    tensor_slices = {}
    for name in tensor_keys:
        shape = desc.tensor_metas[name].shape
        if not shape:
            raise ValueError(
                f"Tensor '{name}' is scalar; cannot build dim-based full slice"
            )
        dim = len(shape) - 1
        tensor_slices[name] = [(dim, slice(0, int(shape[dim]), None))]
    view_handle = handle.view(names=tensor_keys, slices=tensor_slices)
    loaded = view_handle.tensor_dict(device=device)
    print(loaded)
else:
    loaded = handle.tensor_dict(device=device)
    print(loaded)
