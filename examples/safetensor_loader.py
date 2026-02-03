#  Copyright (c) 2026, TensorCast Team.

# import tensorcast as tc

# path = "/mnt/step3-alignment/inference/Qwen2-1.5B-Instruct"
# artifact = tc.from_disk(
#     path,
#     fallback=tc.FallbackOptions.for_disk(path, verify=False),
# )
# names = artifact.tensor_names

# tensor_dict = artifact.tensor_dict(device="cuda:0")
# print(tensor_dict)


# with artifact.deferred_loader(device="cuda:0", packing="byte_space") as loader:
#     for name in names:
#         _ = loader.tensor(name)
#     slot = loader.commit()

# tensor_dict = dict(slot.tensors)

# print(tensor_dict)

import tensorcast as tc

path = "/mnt/step3-alignment/inference/Qwen2-1.5B-Instruct"

# HF safetensors 目录通常没有 artifact_descriptor.json；要么先 backfill，要么先关掉校验
artifact = tc.from_disk(path, fallback=tc.FallbackOptions.for_disk(path, verify=False))

desc = artifact.describe()  # 一次拿到所有 tensor 的 shape/stride/dtype 元信息

tp_world = 2
slices_tp0: dict[str, list[object]] = {}
slices_tp1: dict[str, list[object]] = {}

for name in desc.tensor_names:
    meta = desc.tensor_metas[name]
    if len(meta.shape) == 0:
        raise RuntimeError(f"scalar tensor cannot be TP-sharded: {name}")

    # 示例：统一按最后一维做 TP=2 切分（每个 tensor 都一分为二）
    dim = len(meta.shape) - 1
    extent = int(meta.shape[dim])
    if extent % tp_world != 0:
        raise RuntimeError(
            f"tensor {name} shape[{dim}]={extent} not divisible by TP={tp_world}"
        )

    part = extent // tp_world
    slices_tp0[name] = [(dim, slice(0, part))]
    slices_tp1[name] = [(dim, slice(part, 2 * part))]

artifact_tp0 = artifact.view(slices=slices_tp0)
artifact_tp1 = artifact.view(slices=slices_tp1)

tp0_tensor_dict = artifact_tp0.tensor_dict(device="cuda:0")
tp1_tensor_dict = artifact_tp1.tensor_dict(device="cuda:0")

print("--------------------------------")
print(tp0_tensor_dict)
print("--------------------------------")
print(tp1_tensor_dict)
print("--------------------------------")
