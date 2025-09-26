// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/communicator/engine/memory_stager.h"
#include "gsl/pointers"

namespace tensorcast::communicator::engine {

// DRAM (CPU) stager: memcpy from source VA into host-pinned pool buffer.
// Supports UMA-backed short pin leases via an optional LeaseProvider injected
// by the StoreEngine. When set, stage() acquires a short lease for the
// [offset, bytes] region and releases it after memcpy completes.
class DRAMStager : public MemoryStager {
 public:
  explicit DRAMStager(
      gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> pool,
      size_t num_buffers_hint = 4);
  ~DRAMStager() override = default;

  // UMA pin-lease provider interface (optional). If set, stage() will
  // acquire a lease for the [offset, bytes] region and release it immediately
  // after memcpy completes.
  struct LeaseHandle {
    virtual ~LeaseHandle() = default;
  };

  struct LeaseProvider {
    virtual ~LeaseProvider() = default;
    virtual std::unique_ptr<LeaseHandle> acquire(const std::string& tensor_key, uint64_t offset, uint64_t bytes) = 0;
  };

  void set_lease_provider(std::shared_ptr<LeaseProvider> provider) {
    lease_provider_ = std::move(provider);
  }

  static std::shared_ptr<LeaseProvider> make_noop_lease_provider();

  absl::StatusOr<void*> stage(
      const std::shared_ptr<communicator::transport::PartitionTensor>& tensor,
      uint64_t offset,
      uint64_t bytes,
      StageMode mode = StageMode::kBlocking) override;

  absl::Status release_staged_buffer(gsl::not_null<void*> host_ptr) override;

  size_t get_chunk_size() const override {
    return chunk_size_;
  }

  size_t get_num_buffers() const override {
    return num_buffers_hint_;
  }

 private:
  gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> pool_;
  const size_t chunk_size_;
  const size_t num_buffers_hint_;

  // Track allocations so that we can deallocate correctly
  mutable absl::Mutex mu_;
  std::unordered_map<void*, std::vector<gsl::not_null<char*>>> allocations_ ABSL_GUARDED_BY(mu_);

  std::shared_ptr<LeaseProvider> lease_provider_;
};

} // namespace tensorcast::communicator::engine
