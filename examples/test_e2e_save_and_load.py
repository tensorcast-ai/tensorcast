#  Copyright (c) 2025, StepCast Team.

import torch

from scstore.torch_util import load_dict, save_dict

# sudo python examples/save_vllm_model.py --artifact-name DeepSeek-R1-0528 --local-artifact-path /mnt/host0/DeepSeek-R1-0528  --storage-path /mnt/host0/scstore --tensor-parallel-size 8
directory = "/mnt/host0/scstore/DeepSeek-R1-0528-layer-8-tp-1/rank_0"
ori_path = f"{directory}/original_state_dict.pth"

ori_dict = torch.load(ori_path, map_location="cuda:0")

for key, tensor in ori_dict.items():
    ori_dict[key] = tensor.contiguous()


def assert_dict_equal(
    ori_dict: dict[str, torch.Tensor], sc_dict: dict[str, torch.Tensor]
) -> None:
    """Compare two dictionaries of tensors and assert they are equal within a tolerance.

    Args:
        ori_dict: The original dictionary of tensors.
        sc_dict: The dictionary loaded back from disk.

    Raises:
        AssertionError: If the dictionaries differ in keys, shapes, or data beyond *atol*.
    """
    if ori_dict.keys() != sc_dict.keys():
        missing_keys = ori_dict.keys() ^ sc_dict.keys()
        raise AssertionError(f"Key sets differ: {missing_keys}")

    for key in ori_dict:
        t1, t2 = ori_dict[key].bfloat16().cuda(), sc_dict[key].bfloat16().cuda()

        if not torch.equal(t1, t2):
            raise AssertionError(f"Tensor '{key}' differs: {t1}, {t2}")
        else:
            print(f"Tensor '{key}' matches.")


tmp_dir = "/mnt/host0/scstore/DeepSeek-R1-0528-layer-8-tp-1/rank_test"  # ssd
# tmp_dir = "/tmp/rank_test" # tmpfs
save_dict(ori_dict, tmp_dir, use_streaming=True)
sc_dict = load_dict(tmp_dir, device_id=0, storage_path="", enable_verification=False)

# Validate that tensors are identical within numerical tolerance.
assert_dict_equal(ori_dict, sc_dict)

print("All tensors match ✅")
