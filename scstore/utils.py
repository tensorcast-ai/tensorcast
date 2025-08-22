#  Copyright (c) 2025, StepCast Team.

import re

import torch
from torch import nn


def set_module_buffer_to_device(
    module: nn.Module,
    target: str,
    device: torch.device,
):
    module_path, _, buffer_name = target.rpartition(".")

    mod: torch.nn.Module = module.get_submodule(module_path)

    buffer = mod.get_buffer(buffer_name)
    mod._buffers[buffer_name] = buffer.to(device)


def send_module_buffers_to_device(
    module: nn.Module,
    device_map: dict,
):
    if "" in device_map and len(device_map) != 1:
        raise RuntimeError(
            f"Device map {device_map} is invalid. If you want to specify the default device, use key ''."  # noqa: E501
        )

    buffer_names = [name for name, _ in module.named_buffers()]
    for tensor_or_module, device_id in device_map.items():
        if tensor_or_module == "":
            for buffer_name in buffer_names:
                set_module_buffer_to_device(module, buffer_name, device_id)
        else:
            for buffer_name in buffer_names:
                if buffer_name.startswith(tensor_or_module):
                    set_module_buffer_to_device(module, buffer_name, device_id)


def get_total_parameter_size(module):
    total_param_size = 0
    for param in module.parameters():
        total_param_size += param.numel() * dtype_byte_size(param.dtype)
    return total_param_size


def get_parameter_size(artifact: nn.Module, param_path: str):
    param = artifact.get_parameter(param_path)
    return param.numel() * dtype_byte_size(param.dtype)


def get_no_split_modules(artifact, no_split_modules_list, parent_name=""):
    no_split_modules = {}
    for name, submodule in artifact.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        module_class_name = submodule.__class__.__name__
        # If the module is a leaf module or in the no_split_modules_list, we don't split it # noqa: E501
        if not list(submodule.children()) or module_class_name in no_split_modules_list:
            no_split_modules[full_name] = get_total_parameter_size(submodule)
            continue
        no_split_modules.update(
            get_no_split_modules(submodule, no_split_modules_list, full_name)
        )

    return no_split_modules


def dtype_byte_size(dtype: torch.dtype) -> int:
    return torch.finfo(dtype).bits // 8


def to_num_bytes(value: str) -> int:
    """
    Convert a string representing a data size to its equivalent number of bytes.

    The input must strictly follow the format:
        <number><unit>

    - <number>: A positive integer.
    - <unit>: One of the following case-sensitive units:
        B, KB, MB, GB, TB, PB, EB, ZB, YB

    No leading, trailing, or middle spaces or other characters are allowed.

    Examples:
        "1GB"  -> 1073741824
        "512MB" -> 536870912

    Args:
        value (str): The string to convert.

    Returns:
        int: The equivalent number of bytes.

    Raises:
        ValueError: If the input format is incorrect.
    """
    # Define the regular expression pattern for validation
    pattern = r"^(\d+)(B|KB|MB|GB|TB|PB|EB|ZB|YB)$"
    match = re.fullmatch(pattern, value)

    if not match:
        error_message = (
            "Invalid format. The input must be a positive integer "
            "followed immediately by a unit "
            "(B, KB, MB, GB, TB, PB, EB, ZB, YB), case sensitive, "
            "with no spaces or other characters."
        )
        raise ValueError(error_message)

    number_str, unit = match.groups()
    number = int(number_str)

    # Define the multiplier for each unit
    unit_multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "PB": 1024**5,
        "EB": 1024**6,
        "ZB": 1024**7,
        "YB": 1024**8,
    }

    bytes_value = number * unit_multipliers[unit]
    return bytes_value
