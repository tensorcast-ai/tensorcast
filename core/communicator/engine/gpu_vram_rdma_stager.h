// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "gsl/pointers"

#include "core/communicator/base/constants.h"
#include "core/communicator/engine/memory_stager.h"
#include "core/communicator/transport/partition_tensor.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::communicator::engine {

class GpuVramStagingPool {
 public:
  struct Slice {
    void* ptr = nullptr;
    size_t bytes = 0;
    int index = -1;
  };

  GpuVramStagingPool(int device_id, size_t pool_bytes, size_t slice_bytes);
  ~GpuVramStagingPool();

  GpuVramStagingPool(const GpuVramStagingPool&) = delete;
  GpuVramStagingPool& operator=(const GpuVramStagingPool&) = delete;

  absl::Status initialize();

  absl::StatusOr<Slice> acquire_slice();
  absl::Status release_slice(int index);

  [[nodiscard]] int device_id() const {
    return device_id_;
  }

  [[nodiscard]] size_t pool_bytes() const {
    return pool_bytes_;
  }

  [[nodiscard]] size_t slice_bytes() const {
    return slice_bytes_;
  }

  [[nodiscard]] size_t num_slices() const {
    return num_slices_;
  }

  [[nodiscard]] void* base_ptr() const {
    return base_ptr_;
  }

  [[nodiscard]] bool contains_ptr(gsl::not_null<void*> ptr) const;
  [[nodiscard]] std::optional<MemoryStager::MrSlab> mr_slab() const;

 private:
  absl::Status release_pool();

  int device_id_ = -1;
  size_t pool_bytes_ = 0;
  size_t slice_bytes_ = 0;
  size_t num_slices_ = 0;
  void* base_ptr_ = nullptr;

  mutable absl::Mutex mu_;
  std::vector<int> free_indices_ ABSL_GUARDED_BY(mu_);
  std::vector<bool> in_use_ ABSL_GUARDED_BY(mu_);
  bool initialized_ = false;
};

class GpuVramRdmaStager : public MemoryStager {
 public:
  explicit GpuVramRdmaStager(std::shared_ptr<GpuVramStagingPool> pool);
  ~GpuVramRdmaStager() override = default;

  absl::StatusOr<void*> stage(
      const std::shared_ptr<transport::PartitionTensor>& tensor,
      uint64_t offset,
      uint64_t bytes,
      StageMode mode = StageMode::kBlocking) override;

  absl::Status release_staged_buffer(gsl::not_null<void*> exposed_ptr) override;

  [[nodiscard]] size_t get_chunk_size() const override;
  [[nodiscard]] size_t get_num_buffers() const override;

  std::optional<MrSlab> mr_slab_for_ptr(gsl::not_null<void*> exposed_ptr) const override;

  std::shared_ptr<GpuVramStagingPool> pool() const {
    return pool_;
  }

 private:
  std::shared_ptr<GpuVramStagingPool> pool_;

  mutable absl::Mutex mu_;
  std::unordered_map<void*, int> staged_slices_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::communicator::engine
