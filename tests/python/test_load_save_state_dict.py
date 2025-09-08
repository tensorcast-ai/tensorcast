#  Copyright (c) 2025, TensorCast Team.

import torch
from typing import Any
from tensorcast.api import load_dict_sync as load_dict


def compare_tensor_dicts(
    dict1: dict[str, torch.Tensor],
    dict2: dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict[str, str | dict[str, Any]]:
    """
    Compares PyTorch tensors in two dictionaries.

    Args:
        dict1: The first dictionary, with string keys and PyTorch tensor values.
        dict2: The second dictionary, with string keys and PyTorch tensor values.
        rtol: Relative tolerance (default: 1e-5).
        atol: Absolute tolerance (default: 1e-8).

    Returns:
        A dictionary containing the comparison results, with keys from the original
        dictionaries and values as difference information.
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise TypeError("Inputs must be dictionaries.")

    results: dict[str, str | dict[str, Any]] = {}

    # Get the set of all keys from the first dictionary.
    # We are assuming dict1 is the reference or superset for keys to check.
    # If keys present only in dict2 should also be reported,
    # all_keys = set(dict1.keys()).union(set(dict2.keys())) would be more appropriate.
    all_keys = dict1.keys()

    # Check each key
    for key in all_keys:
        # Check if the key exists in both dictionaries
        if key not in dict1:
            # This case should not happen if all_keys = dict1.keys()
            # If all_keys can contain keys not in dict1, this is relevant.
            results[key] = "Only exists in the second dictionary"
            continue
        if key not in dict2:
            results[key] = "Only exists in the first dictionary"
            continue

        tensor1 = dict1[key].to("cuda").half()
        tensor2 = dict2[key].to("cuda").half()

        # Check if shapes are the same
        if tensor1.shape != tensor2.shape:
            results[key] = f"Shapes differ: {tensor1.shape} vs {tensor2.shape}"
            continue

        # Calculate differences
        abs_diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(abs_diff).item()
        mean_diff = torch.mean(abs_diff).item()

        # Check if tensors are close within tolerance
        is_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

        results[key] = {
            "shape": tensor1.shape,
            "max_difference": max_diff,
            "mean_difference": mean_diff,
            "is_close": is_close,
            "nbytes": tensor1.nbytes,
        }

        print(key, results[key])

    return results


if __name__ == "__main__":
    # Example usage:
    # Replace with actual paths to your artifact files
    # Ensure the artifact files exist at these paths before running.
    path_to_torch_state_dict = (
        "/data/workspace/tensorcast-store/test-models/Qwen/Qwen3-0.6B/state_dict.pth"
    )
    path_to_sc_model_dir = (
        "/data/workspace/tensorcast-store/test-models/Qwen/Qwen3-0.6B"
    )

    try:
        torch_state_dict = torch.load(path_to_torch_state_dict)
        sc_state_dict = load_dict(
            disk_path=path_to_sc_model_dir,
            device_id=0,
            storage_path="",
            enable_verification=False,
        )

        comparison_results = compare_tensor_dicts(torch_state_dict, sc_state_dict)
        # Optionally, print or process comparison_results further
        # print("\nFinal Comparison Results:")
        # for key, result in comparison_results.items():
        #     print(f"{key}: {result}")

    except FileNotFoundError as e:
        print(f"Error: Artifact file not found. Please check the paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
