#!/usr/bin/env python3
#  Copyright (c) 2025, TensorCast Team.

"""
Example demonstrating the streaming save functionality for GPU tensors.

This example shows how to use the new StreamingTensorWriter backend for
efficient GPU tensor saving with configurable buffer sizes and async I/O.
"""

import os
import time

import torch

from scstore.torch_util import save_dict


def create_example_model(size_gb: float = 1.0) -> dict[str, torch.Tensor]:
    """Create an example artifact with specified size in GB."""
    # Calculate number of float32 elements for the target size
    bytes_per_float32 = 4
    total_bytes = int(size_gb * 1024 * 1024 * 1024)
    total_elements = total_bytes // bytes_per_float32

    # Create multiple tensors to simulate a real artifact
    tensors_per_layer = 4
    layers = 12
    elements_per_tensor = total_elements // (tensors_per_layer * layers)

    state_dict = {}

    for layer in range(layers):
        # Simulate different tensor shapes
        shapes = [
            (1024, elements_per_tensor // 1024),
            (elements_per_tensor // 512, 512),
            (256, elements_per_tensor // 256),
            (elements_per_tensor,),
        ]

        for i, shape in enumerate(shapes[:tensors_per_layer]):
            tensor_name = f"layer_{layer}.tensor_{i}"
            # Create tensor on GPU if available
            if torch.cuda.is_available():
                state_dict[tensor_name] = torch.randn(
                    shape, device="cuda", dtype=torch.float32
                )
            else:
                state_dict[tensor_name] = torch.randn(shape, dtype=torch.float32)

    return state_dict


def compare_save_methods(state_dict: dict[str, torch.Tensor], output_dir: str):
    """Compare traditional and streaming save methods."""

    # Traditional save
    traditional_path = os.path.join(output_dir, "traditional_save")
    print(f"\nSaving with traditional method to {traditional_path}...")
    start_time = time.time()
    save_dict(state_dict, traditional_path, use_streaming=False)
    traditional_time = time.time() - start_time
    print(f"Traditional save completed in {traditional_time:.2f} seconds")

    # Streaming save with default config
    streaming_path = os.path.join(output_dir, "streaming_save_default")
    print(f"\nSaving with streaming method (default config) to {streaming_path}...")
    start_time = time.time()
    save_dict(state_dict, streaming_path, use_streaming=True)
    streaming_default_time = time.time() - start_time
    print(f"Streaming save (default) completed in {streaming_default_time:.2f} seconds")

    # Streaming save with custom config
    streaming_custom_path = os.path.join(output_dir, "streaming_save_custom")
    custom_config = {
        "num_buffers": 8,  # More buffers for better pipelining
        "buffer_size_mb": 512,  # Larger buffers
        "enable_async_write": True,  # Async I/O
    }
    print(
        f"\nSaving with streaming method (custom config) to {streaming_custom_path}..."
    )
    print(f"Config: {custom_config}")
    start_time = time.time()
    save_dict(
        state_dict,
        streaming_custom_path,
        use_streaming=True,
        streaming_config=custom_config,
    )
    streaming_custom_time = time.time() - start_time
    print(f"Streaming save (custom) completed in {streaming_custom_time:.2f} seconds")

    # Summary
    print("\n" + "=" * 50)
    print("Performance Summary:")
    print(f"Traditional save: {traditional_time:.2f}s")
    print(
        f"Streaming save (default): {streaming_default_time:.2f}s ({traditional_time / streaming_default_time:.2f}x)"
    )
    print(
        f"Streaming save (custom): {streaming_custom_time:.2f}s ({traditional_time / streaming_custom_time:.2f}x)"
    )


def main():
    """Main example function."""
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        print("CUDA is not available. Using CPU tensors for demonstration.")

    # Create example artifact
    artifact_size_gb = 2.0  # Adjust based on available GPU memory
    print(f"\nCreating example artifact ({artifact_size_gb} GB)...")
    state_dict = create_example_model(artifact_size_gb)

    # Calculate actual size
    total_bytes = sum(
        tensor.element_size() * tensor.nelement() for tensor in state_dict.values()
    )
    print(
        f"Artifact created with {len(state_dict)} tensors, total size: {total_bytes / 1024**3:.2f} GB"
    )

    # Compare save methods
    output_dir = "./streaming_save_test"
    os.makedirs(output_dir, exist_ok=True)
    compare_save_methods(state_dict, output_dir)

    # Environment variable configuration example
    print("\n" + "=" * 50)
    print("Environment Variable Configuration:")
    print("You can also configure streaming save via environment variables:")
    print("  export STREAMING_CHUNK_SIZE_MB=512")
    print("  export STREAMING_POOL_SIZE_GB=20")
    print("  export STREAMING_NUM_BUFFERS=8")


if __name__ == "__main__":
    main()
