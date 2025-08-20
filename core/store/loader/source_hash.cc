// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/source_hash.h"

#include <algorithm>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "core/common/cuda_api.h"
#include "core/common/model_hash.h"

namespace stepcast::store::loader {

namespace {

using stepcast::store::model_hash::compute_tree_hash_root_sha256;
using stepcast::store::model_hash::multibase_multihash_sha256;
using stepcast::store::model_hash::sha256_digest_bytes;

class CpuMemorySourceLocal : public SeekableSource {
 public:
  CpuMemorySourceLocal(const void* base_ptr, uint64_t total_size)
      : base_ptr_(static_cast<const uint8_t*>(base_ptr)), total_size_(total_size) {}

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto st = read_at(current_offset_, dst, max_bytes);
    if (!st.ok())
      return st;
    current_offset_ += *st;
    return st;
  }
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= total_size_)
      return static_cast<size_t>(0);
    const size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, total_size_ - offset));
    std::memcpy(dst, base_ptr_ + offset, to_copy);
    return to_copy;
  }

 private:
  const uint8_t* base_ptr_;
  uint64_t total_size_;
  uint64_t current_offset_ = 0;
};

class GpuMemorySourceLocal : public SeekableSource {
 public:
  GpuMemorySourceLocal(void* device_ptr, uint64_t total_size, int device_id)
      : device_ptr_(static_cast<uint8_t*>(device_ptr)), total_size_(total_size), device_id_(device_id) {}

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto st = read_at(current_offset_, dst, max_bytes);
    if (!st.ok())
      return st;
    current_offset_ += *st;
    return st;
  }
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= total_size_)
      return static_cast<size_t>(0);
    const size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, total_size_ - offset));
    if (auto st = stepcast::cuda::set_device(device_id_); !st.ok())
      return st;
    auto st = stepcast::cuda::memcpy(dst, device_ptr_ + offset, to_copy, cudaMemcpyDeviceToHost);
    if (!st.ok())
      return st;
    if (auto sync = stepcast::cuda::device_synchronize(); !sync.ok())
      return sync;
    return to_copy;
  }

 private:
  uint8_t* device_ptr_;
  uint64_t total_size_;
  int device_id_;
  uint64_t current_offset_ = 0;
};

} // namespace

absl::StatusOr<std::string> compute_data_multihash_from_seekable_source(
    SeekableSource& source,
    uint64_t total_size,
    size_t leaf_chunk_bytes) {
  if (total_size == 0) {
    return absl::InvalidArgumentError("total_size must be > 0");
  }
  if (leaf_chunk_bytes == 0) {
    return absl::InvalidArgumentError("leaf_chunk_bytes must be > 0");
  }

  std::vector<std::vector<uint8_t>> leaves;
  leaves.reserve(static_cast<size_t>((total_size + leaf_chunk_bytes - 1) / leaf_chunk_bytes));

  std::vector<uint8_t> buffer(leaf_chunk_bytes);
  uint64_t processed = 0;
  while (processed < total_size) {
    const size_t to_read = static_cast<size_t>(std::min<uint64_t>(leaf_chunk_bytes, total_size - processed));
    auto n_or = source.read_at(processed, buffer.data(), to_read);
    if (!n_or.ok()) {
      return n_or.status();
    }
    const size_t got = n_or.value();
    if (got == 0) {
      // Unexpected short read
      return absl::DataLossError("short read while hashing source");
    }
    leaves.push_back(sha256_digest_bytes(absl::Span<const uint8_t>(buffer.data(), got)));
    processed += got;
  }
  std::vector<uint8_t> root = compute_tree_hash_root_sha256(leaves);
  return multibase_multihash_sha256(root);
}

absl::StatusOr<std::string> compute_data_multihash_from_cpu_memory(
    const void* base_ptr,
    uint64_t total_size,
    size_t leaf_chunk_bytes) {
  if (base_ptr == nullptr) {
    return absl::InvalidArgumentError("base_ptr must not be null");
  }
  CpuMemorySourceLocal src(base_ptr, total_size);
  return compute_data_multihash_from_seekable_source(src, total_size, leaf_chunk_bytes);
}

absl::StatusOr<std::string> compute_data_multihash_from_gpu_memory(
    void* device_ptr,
    uint64_t total_size,
    int device_id,
    size_t leaf_chunk_bytes) {
  if (device_ptr == nullptr) {
    return absl::InvalidArgumentError("device_ptr must not be null");
  }
  GpuMemorySourceLocal src(device_ptr, total_size, device_id);
  return compute_data_multihash_from_seekable_source(src, total_size, leaf_chunk_bytes);
}

} // namespace stepcast::store::loader
