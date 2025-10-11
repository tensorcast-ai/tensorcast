#  Copyright (c) 2025, TensorCast Team.

import gc

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

import tensorcast as tc
from tensorcast.testing.dict import assert_state_dict_equal

tc.init(address="127.0.0.1:50052")

hf_model_name = "Qwen/Qwen3-0.6B"
# Load a artifact from HuggingFace artifact hub.
artifact = AutoModelForCausalLM.from_pretrained(
    hf_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
)

state_dict = artifact.state_dict()

registered = tc.put(state_dict, device=0)
saved_dict = registered.state_dict
assert saved_dict is not None

# Validate equality between the original and registered dicts
assert_state_dict_equal(state_dict, saved_dict)
print("All tensors match ✅")

# Verify GPU memory held by state_dict is freed after deletion
if torch.cuda.is_available():
    device = torch.device("cuda", 0)

    # Baseline after registration/validation (saved_dict may occupy GPU memory)
    torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device)

    # Move state_dict to GPU to simulate/use VRAM for it explicitly
    state_dict = {
        k: v.to(device=device, non_blocking=True) for k, v in state_dict.items()
    }
    torch.cuda.synchronize(device)
    mem_with_state = torch.cuda.memory_allocated(device)

    used_by_state_dict = mem_with_state - mem_before
    print(
        f"state_dict moved to GPU. Allocated +{used_by_state_dict / (1024**2):.2f} MiB "
        f"(before={mem_before / (1024**2):.2f} MiB, after={mem_with_state / (1024**2):.2f} MiB)"
    )

    # Compare again on-GPU to validate correctness in this configuration as well
    assert_state_dict_equal(state_dict, saved_dict)

    # Delete and ensure VRAM used by state_dict is freed
    del state_dict
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    mem_after_del = torch.cuda.memory_allocated(device)

    # Allow a tiny tolerance for allocator bookkeeping
    tolerance_bytes = 8 * 1024 * 1024  # 8 MiB
    if mem_after_del > mem_before + tolerance_bytes:
        raise AssertionError(
            (
                "GPU memory not fully released after deleting state_dict: "
                f"before={mem_before / (1024**2):.2f} MiB, "
                f"after_del={mem_after_del / (1024**2):.2f} MiB"
            )
        )

    print(
        "state_dict GPU memory released ✅ "
        f"(baseline={mem_before / (1024**2):.2f} MiB, after_del={mem_after_del / (1024**2):.2f} MiB)"
    )
else:
    print("CUDA not available; skipping VRAM release test.")
