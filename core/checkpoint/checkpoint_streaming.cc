// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/checkpoint/checkpoint_streaming.h"

#include <filesystem>
#include "core/common/cuda_api.h"

#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "core/checkpoint/progress_bar.h"

namespace stepcast::store {

std::unordered_map<std::string, uint64_t> save_tensors_streaming(
    const std::vector<std::string>& tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const std::string& path,
    const StreamingTensorWriter::Config& config) {
  // Create output directory if it doesn't exist
  std::filesystem::create_directories(path);

  const std::string tensor_filename = std::filesystem::path(path) / "tensor.data";

  // Read configuration from environment variables with defaults
  const char* chunk_size_env = std::getenv("STREAMING_CHUNK_SIZE_MB");
  const char* pool_size_env = std::getenv("STREAMING_POOL_SIZE_GB");
  const char* num_buffers_env = std::getenv("STREAMING_NUM_BUFFERS");

  // Override config with environment variables if set
  StreamingTensorWriter::Config final_config = config;
  if (chunk_size_env) {
    final_config.buffer_size_mb = std::stoull(chunk_size_env);
  }
  if (num_buffers_env) {
    final_config.num_buffers = std::stoull(num_buffers_env);
  }

  const size_t pool_size_gb = pool_size_env ? std::stoull(pool_size_env) : 10; // Default: 10 GB
  const size_t pool_size = pool_size_gb << 30; // Convert GB to bytes
  const size_t chunk_size = final_config.buffer_size_mb << 20; // Convert MB to bytes

  LOG(INFO) << "Streaming tensor save configuration: "
            << "num_buffers=" << final_config.num_buffers << ", "
            << "buffer_size=" << final_config.buffer_size_mb << "MB, "
            << "pool_size=" << pool_size_gb << "GB, "
            << "async_write=" << (final_config.enable_async_write ? "enabled" : "disabled");

  // Create pinned memory pool
  auto pinned_pool = std::make_shared<PinnedMemoryPool>(pool_size, chunk_size);

  // Create streaming writer
  StreamingTensorWriter writer(tensor_filename, final_config, pinned_pool);

  // Initialize writer
  auto init_status = writer.initialize();
  if (!init_status.ok()) {
    LOG(FATAL) << "Failed to initialize StreamingTensorWriter: " << init_status;
  }

  // Tensor offset tracking
  std::unordered_map<std::string, uint64_t> tensor_offsets;

  // ------------------------------------------------------------------
  // Deduplication (pointer-based, see checkpoint.cc for rationale)
  // ------------------------------------------------------------------
  struct StorageMeta {
    uint64_t max_size{0};
    std::string owner_name;
  };

  std::unordered_map<const char*, StorageMeta> ptr_meta;
  for (const auto& n : tensor_names) {
    const auto& [base, size] = tensor_data[n];
    const char* p = reinterpret_cast<const char*>(base);
    auto& meta = ptr_meta[p];
    if (size > meta.max_size) {
      meta.max_size = size;
      meta.owner_name = n;
    }
  }

  std::unordered_map<const char*, uint64_t> ptr_written_offset;

  const int total = tensor_names.size();
  int count = 0;

  // Create CUDA stream for async operations
  cudaStream_t stream = nullptr;
  auto stream_status = cuda::stream_create(&stream);
  if (!stream_status.ok()) {
    LOG(WARNING) << "Streaming save - stream_create failed (will fallback to sync transfers): "
                 << stream_status.message();
    stream = nullptr;
  }

  for (const auto& name : tensor_names) {
    const auto& [base, logical_size_unused] = tensor_data[name];
    (void)logical_size_unused; // Suppress unused variable warning
    const char* data_ptr = reinterpret_cast<const char*>(base);

    auto it_ptr = ptr_written_offset.find(data_ptr);
    uint64_t offset_for_tensor = 0;
    if (it_ptr != ptr_written_offset.end()) {
      offset_for_tensor = it_ptr->second;
    } else {
      // Need to write the backing storage (max_size bytes)
      const uint64_t size_to_write = ptr_meta[data_ptr].max_size;

      // Detect whether the pointer is device (CUDA) memory or host
      bool is_device_ptr = false;
      cudaPointerAttributes attr;
      auto attr_status =
          cuda::pointer_get_attributes_full(const_cast<void*>(reinterpret_cast<const void*>(data_ptr)), &attr);
      if (attr_status.ok()) {
#if CUDART_VERSION >= 10000
        is_device_ptr = (attr.type == cudaMemoryTypeDevice);
#else
        is_device_ptr = (attr.memoryType == cudaMemoryTypeDevice);
#endif
      } else {
        ABSL_CHECK_OK(cuda::get_last_error());
      }

      auto offset_result = writer.write_tensor(data_ptr, size_to_write, is_device_ptr, stream);
      if (!offset_result.ok()) {
        LOG(FATAL) << "Failed to write tensor '" << name << "': " << offset_result.status();
      }

      offset_for_tensor = offset_result.value();
      ptr_written_offset[data_ptr] = offset_for_tensor;
    }

    tensor_offsets[name] = offset_for_tensor;

    // Update progress bar
    count++;
    const float progress = static_cast<float>(count) / total;
    show_progress_bar(progress, "Saving tensors (streaming): ");
  }

  // Cleanup CUDA stream
  if (stream) {
    auto destroy_status = cuda::stream_destroy(stream);
    if (!destroy_status.ok()) {
      LOG(ERROR) << "Failed to destroy CUDA stream: " << destroy_status.message();
    }
  }

  // Finalize writing
  auto finalize_status = writer.finalize();
  if (!finalize_status.ok()) {
    LOG(ERROR) << "Failed to finalize StreamingTensorWriter: " << finalize_status;
  }

  LOG(INFO) << "Successfully saved " << tensor_names.size()
            << " tensors using streaming approach. Total bytes: " << writer.total_bytes_written();

  return tensor_offsets;
}

} // namespace stepcast::store