// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gsl/pointers"

#include "core/communicator/engine/memory_stager.h"
#include "core/communicator/engine/gpu_tcp_stager.h"

namespace tensorcast::communicator {

// Adapter that exposes GPU staging via the unified MemoryStager interface.
// Internally delegates to GpuTcpStager for D2H copies and buffer lifecycle.
class GpuNetStager : public MemoryStager {
 public:
  explicit GpuNetStager(std::shared_ptr<GpuTcpStager> delegate)
      : delegate_(std::move(delegate)) {}

  absl::StatusOr<void*> stage(
      const std::shared_ptr<PartitionTensor>& tensor,
      uint64_t offset,
      uint64_t bytes) override {
    if (!delegate_) return absl::FailedPreconditionError("GpuNetStager not initialized");
    return delegate_->stage(tensor, offset, bytes);
  }

  absl::Status release_staged_buffer(gsl::not_null<void*> host_ptr) override {
    if (!delegate_) return absl::FailedPreconditionError("GpuNetStager not initialized");
    return delegate_->release_staged_buffer(host_ptr.get());
  }

  size_t get_chunk_size() const override {
    return delegate_ ? delegate_->get_chunk_size() : 0;
  }

  size_t get_num_buffers() const override {
    return delegate_ ? delegate_->get_num_buffers() : 0;
  }

 private:
  std::shared_ptr<GpuTcpStager> delegate_;
};

} // namespace tensorcast::communicator

