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
  explicit GpuNetStager(const gsl::not_null<std::shared_ptr<GpuTcpStager>>& delegate)
      : delegate_(delegate) {}

  absl::StatusOr<void*> stage(
      const std::shared_ptr<PartitionTensor>& tensor,
      uint64_t offset,
      uint64_t bytes) override {
    return delegate_->stage(tensor, offset, bytes);
  }

  absl::Status release_staged_buffer(gsl::not_null<void*> host_ptr) override {
    return delegate_->release_staged_buffer(host_ptr.get());
  }

  size_t get_chunk_size() const override {
    return delegate_->get_chunk_size();
  }

  size_t get_num_buffers() const override {
    return delegate_->get_num_buffers();
  }

 private:
  gsl::not_null<std::shared_ptr<GpuTcpStager>> delegate_;
};

} // namespace tensorcast::communicator

