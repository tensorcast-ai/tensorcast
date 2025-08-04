#!/usr/bin/env python3
#  Copyright (c) 2025, StepCast Team.

"""Tests for streaming save functionality."""

import json
import os
import tempfile
import unittest

import torch

from scstore.torch_util import save_dict, load_dict_pure_local


class TestStreamingSave(unittest.TestCase):
    """Test cases for streaming tensor save functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.has_cuda = torch.cuda.is_available()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_model(self, num_tensors=10, tensor_size_mb=100):
        """Create a test model with specified characteristics."""
        state_dict = {}
        elements_per_tensor = (tensor_size_mb * 1024 * 1024) // 4  # float32

        for i in range(num_tensors):
            shape = (1024, elements_per_tensor // 1024)
            if self.has_cuda:
                tensor = torch.randn(shape, device='cuda', dtype=torch.float32)
            else:
                tensor = torch.randn(shape, dtype=torch.float32)
            state_dict[f"tensor_{i}"] = tensor

        return state_dict

    def test_streaming_save_basic(self):
        """Test basic streaming save functionality."""
        # Create test model
        state_dict = self.create_test_model(num_tensors=5, tensor_size_mb=50)

        # Save using streaming
        save_path = os.path.join(self.test_dir, "streaming_test")
        save_dict(state_dict, save_path, use_streaming=True)

        # Verify files exist
        self.assertTrue(os.path.exists(os.path.join(save_path, "tensor_index.json")))
        self.assertTrue(os.path.exists(os.path.join(save_path, "tensor.data_0")))

        # Load and verify
        _, loaded_state_dict = load_dict_pure_local(save_path)

        # Compare tensors
        for name, original_tensor in state_dict.items():
            loaded_tensor = loaded_state_dict[name]
            # Move to CPU for comparison
            if original_tensor.is_cuda:
                original_tensor = original_tensor.cpu()
            if loaded_tensor.is_cuda:
                loaded_tensor = loaded_tensor.cpu()
            self.assertTrue(torch.allclose(original_tensor, loaded_tensor))

    def test_streaming_vs_traditional_equivalence(self):
        """Test that streaming and traditional save produce equivalent results."""
        # Create test model
        state_dict = self.create_test_model(num_tensors=3, tensor_size_mb=30)

        # Save using traditional method
        traditional_path = os.path.join(self.test_dir, "traditional")
        save_dict(state_dict, traditional_path, use_streaming=False)

        # Save using streaming method
        streaming_path = os.path.join(self.test_dir, "streaming")
        save_dict(state_dict, streaming_path, use_streaming=True)

        # Load both
        _, traditional_loaded = load_dict_pure_local(traditional_path)
        _, streaming_loaded = load_dict_pure_local(streaming_path)

        # Compare all tensors
        for name in state_dict.keys():
            trad_tensor = traditional_loaded[name].cpu()
            stream_tensor = streaming_loaded[name].cpu()
            self.assertTrue(torch.allclose(trad_tensor, stream_tensor))

    def test_streaming_custom_config(self):
        """Test streaming save with custom configuration."""
        # Create test model
        state_dict = self.create_test_model(num_tensors=4, tensor_size_mb=25)

        # Custom config
        custom_config = {
            "num_buffers": 2,
            "buffer_size_mb": 64,
            "enable_async_write": False  # Synchronous for predictable testing
        }

        # Save with custom config
        save_path = os.path.join(self.test_dir, "custom_config")
        save_dict(state_dict, save_path, use_streaming=True, streaming_config=custom_config)

        # Verify save completed successfully
        self.assertTrue(os.path.exists(os.path.join(save_path, "tensor_index.json")))

        # Load and verify
        _, loaded_state_dict = load_dict_pure_local(save_path)
        self.assertEqual(len(loaded_state_dict), len(state_dict))

    def test_empty_model(self):
        """Test saving an empty model."""
        state_dict = {}

        save_path = os.path.join(self.test_dir, "empty")
        save_dict(state_dict, save_path, use_streaming=True)

        # Should still create tensor_index.json
        self.assertTrue(os.path.exists(os.path.join(save_path, "tensor_index.json")))

        with open(os.path.join(save_path, "tensor_index.json"), 'r') as f:
            index = json.load(f)
            self.assertEqual(len(index), 0)

    def test_mixed_tensor_sizes(self):
        """Test saving model with various tensor sizes."""
        state_dict = {
            "tiny": torch.randn(10, device='cuda' if self.has_cuda else 'cpu'),
            "small": torch.randn(100, 100, device='cuda' if self.has_cuda else 'cpu'),
            "medium": torch.randn(1000, 1000, device='cuda' if self.has_cuda else 'cpu'),
            "large": torch.randn(5000, 5000, device='cuda' if self.has_cuda else 'cpu'),
        }

        save_path = os.path.join(self.test_dir, "mixed_sizes")
        save_dict(state_dict, save_path, use_streaming=True)

        _, loaded_state_dict = load_dict_pure_local(save_path)

        for name, original_tensor in state_dict.items():
            loaded_tensor = loaded_state_dict[name]
            if original_tensor.is_cuda:
                original_tensor = original_tensor.cpu()
            if loaded_tensor.is_cuda:
                loaded_tensor = loaded_tensor.cpu()
            self.assertTrue(torch.allclose(original_tensor, loaded_tensor))


if __name__ == '__main__':
    unittest.main()