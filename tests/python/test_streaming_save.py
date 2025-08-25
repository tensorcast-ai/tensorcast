#!/usr/bin/env python3
#  Copyright (c) 2025, StepCast Team.

"""Tests for streaming save functionality (pytest style)."""

import json
import os

import pytest
import torch

from scstore.torch_util import save_dict, load_dict_from_disk


@pytest.fixture
def has_cuda():
    """Whether CUDA is available on the test host."""
    return torch.cuda.is_available()


def create_test_artifact(num_tensors=10, tensor_size_mb=100, use_cuda=False):
    """Create a test artifact with specified characteristics.

    Args:
        num_tensors: Number of tensors to create.
        tensor_size_mb: Size per tensor in MB (approx, float32 assumed).
        use_cuda: Whether to allocate tensors on CUDA.

    Returns:
        Dict[str, torch.Tensor]: Mapping from tensor name to tensor.
    """
    state_dict = {}
    elements_per_tensor = (tensor_size_mb * 1024 * 1024) // 4  # float32

    # Resolve a stable target device once per artifact to avoid cross-test interference
    if use_cuda and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            device_index = torch.cuda.current_device()
        except Exception:  # Fallback if current device not initialized
            device_index = 0
        # Ensure process device is set explicitly to avoid stale ambient device state from other tests
        try:
            torch.cuda.set_device(device_index)
        except Exception:
            pass
        device = torch.device('cuda', device_index)
        # Ensure a clean allocator state before large allocations
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device_index)
            # Warm up CUDA context on the target device
            _ = torch.randn((1,), device=device, dtype=torch.float32)
            torch.cuda.synchronize(device_index)
        except Exception:
            pass
    else:
        device = 'cpu'

    for i in range(num_tensors):
        shape = (1024, elements_per_tensor // 1024)
        print(f"Creating tensor {i} on {device} with shape {shape}")
        tensor = torch.randn(shape, device=device, dtype=torch.float32)
        state_dict[f"tensor_{i}"] = tensor

    return state_dict


def test_streaming_save_basic(tmp_path, has_cuda):
    """Test basic streaming save functionality."""
    state_dict = create_test_artifact(num_tensors=5, tensor_size_mb=50, use_cuda=has_cuda)

    save_path = os.path.join(str(tmp_path), "streaming_test")
    save_dict(state_dict, save_path, use_streaming=True)

    assert os.path.exists(os.path.join(save_path, "tensor_index.json"))
    assert os.path.exists(os.path.join(save_path, "tensor.data_0"))

    loaded_state_dict = load_dict_from_disk(save_path, device_id=0, storage_path="")

    for name, original_tensor in state_dict.items():
        loaded_tensor = loaded_state_dict[name]
        if original_tensor.is_cuda:
            original_tensor = original_tensor.cpu()
        if loaded_tensor.is_cuda:
            loaded_tensor = loaded_tensor.cpu()
        assert torch.allclose(original_tensor, loaded_tensor)


def test_streaming_vs_traditional_equivalence(tmp_path, has_cuda):
    """Test that streaming and traditional save produce equivalent results."""
    state_dict = create_test_artifact(num_tensors=3, tensor_size_mb=30, use_cuda=has_cuda)

    traditional_path = os.path.join(str(tmp_path), "traditional")
    save_dict(state_dict, traditional_path, use_streaming=False)

    streaming_path = os.path.join(str(tmp_path), "streaming")
    save_dict(state_dict, streaming_path, use_streaming=True)

    traditional_loaded = load_dict_from_disk(traditional_path, device_id=0, storage_path="")
    streaming_loaded = load_dict_from_disk(streaming_path, device_id=0, storage_path="")

    for name in state_dict.keys():
        trad_tensor = traditional_loaded[name].cpu()
        stream_tensor = streaming_loaded[name].cpu()
        assert torch.allclose(trad_tensor, stream_tensor)


def test_streaming_custom_config(tmp_path, has_cuda):
    """Test streaming save with custom configuration."""
    if has_cuda:
        torch.cuda.synchronize()
    state_dict = create_test_artifact(num_tensors=4, tensor_size_mb=25, use_cuda=has_cuda)

    custom_config = {
        "num_buffers": 2,
        "buffer_size_mb": 64,
        "enable_async_write": False,  # Synchronous for predictable testing
    }

    save_path = os.path.join(str(tmp_path), "custom_config")
    save_dict(state_dict, save_path, use_streaming=True, streaming_config=custom_config)

    assert os.path.exists(os.path.join(save_path, "tensor_index.json"))

    loaded_state_dict = load_dict_from_disk(save_path, device_id=0, storage_path="")
    assert len(loaded_state_dict) == len(state_dict)


def test_empty_artifact(tmp_path):
    """Test saving an empty artifact."""
    state_dict = {}

    save_path = os.path.join(str(tmp_path), "empty")
    save_dict(state_dict, save_path, use_streaming=True)

    assert os.path.exists(os.path.join(save_path, "tensor_index.json"))

    with open(os.path.join(save_path, "tensor_index.json"), 'r') as f:
        index = json.load(f)
        assert len(index) == 0


def test_mixed_tensor_sizes(tmp_path):
    """Test saving artifact with various tensor sizes."""
    state_dict = {
        "tiny": torch.randn(10, device='cpu'),
        "small": torch.randn(100, 100, device='cpu'),
        "medium": torch.randn(1000, 1000, device='cpu'),
        "large": torch.randn(5000, 5000, device='cpu'),
    }

    save_path = os.path.join(str(tmp_path), "mixed_sizes")
    save_dict(state_dict, save_path, use_streaming=True)

    loaded_state_dict = load_dict_from_disk(save_path, device_id=0, storage_path="")

    for name, original_tensor in state_dict.items():
        loaded_tensor = loaded_state_dict[name]
        if original_tensor.is_cuda:
            original_tensor = original_tensor.cpu()
        if loaded_tensor.is_cuda:
            loaded_tensor = loaded_tensor.cpu()
        assert torch.allclose(original_tensor, loaded_tensor)