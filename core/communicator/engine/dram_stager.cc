// Copyright (c) 2025, TensorCast Team.

#include "core/communicator/engine/dram_stager.h"

#include <cstring>

#include "absl/log/log.h"
#include "core/communicator/transport/partition_tensor.h"

namespace tensorcast::communicator {

namespace {
class NoOpLeaseHandle : public DRAMStager::LeaseHandle {};
class NoOpLeaseProvider : public DRAMStager::LeaseProvider {
 public:
  std::unique_ptr<DRAMStager::LeaseHandle> acquire(
      const std::string& /*tensor_key*/, uint64_t /*offset*/, uint64_t /*bytes*/) override {
    return std::make_unique<NoOpLeaseHandle>();
  }
};
} // namespace

std::shared_ptr<DRAMStager::LeaseProvider> DRAMStager::make_noop_lease_provider() {
  static auto provider = std::make_shared<NoOpLeaseProvider>();
  return provider;
}

DRAMStager::DRAMStager(std::shared_ptr<store::PinnedMemoryPool> pool, size_t num_buffers_hint)
    : pool_(std::move(pool)),
      chunk_size_(pool_ ? pool_->chunk_size() : static_cast<size_t>(64 * 1024 * 1024)),
      num_buffers_hint_(num_buffers_hint) {
  if (!pool_) {
    LOG(WARNING) << "DRAMStager constructed without a pinned memory pool; staging may fail";
  }
}

absl::StatusOr<void*> DRAMStager::stage(
    const std::shared_ptr<PartitionTensor>& tensor,
    uint64_t offset,
    uint64_t bytes) {
  if (!pool_) {
    return absl::FailedPreconditionError("PinnedMemoryPool is not configured for DRAMStager");
  }
  if (bytes == 0 || bytes > chunk_size_) {
    return absl::InvalidArgumentError("bytes must be in [1, chunk_size]");
  }

  std::vector<char*> bufs;
  // Immediate allocation attempt; the send loop can back off if needed
  int ret = pool_->allocate(chunk_size_, bufs);
  if (ret != 0 || bufs.empty()) {
    return absl::ResourceExhaustedError("PinnedMemoryPool out of buffers for staging");
  }

  void* host_ptr = bufs[0];
  // memcpy from source VA
  std::unique_ptr<LeaseHandle> lease;
  if (lease_provider_) {
    lease = lease_provider_->acquire(tensor->get_key(), offset, bytes);
  }
  auto* src = tensor->get_addr<uint8_t>() + offset;
  std::memcpy(host_ptr, src, static_cast<size_t>(bytes));
  
  {
    absl::MutexLock lock(&mu_);
    std::vector<gsl::not_null<char*>> wrapped;
    wrapped.reserve(bufs.size());
    for (auto* p : bufs) wrapped.emplace_back(gsl::not_null<char*>{p});
    allocations_[host_ptr] = std::move(wrapped);
  }
  return host_ptr;
}

absl::Status DRAMStager::release_staged_buffer(gsl::not_null<void*> host_ptr) {
  if (!pool_) {
    return absl::FailedPreconditionError("PinnedMemoryPool is not configured for DRAMStager");
  }
  std::vector<char*> bufs;
  {
    absl::MutexLock lock(&mu_);
    auto it = allocations_.find(host_ptr.get());
    if (it == allocations_.end()) {
      return absl::InvalidArgumentError("Unknown host_ptr for DRAMStager::release_staged_buffer");
    }
    bufs.reserve(it->second.size());
    for (auto p : it->second) bufs.push_back(p.get());
    allocations_.erase(it);
  }
  int ret = pool_->deallocate(bufs);
  if (ret != 0) {
    return absl::InternalError("Failed to deallocate pinned buffers");
  }
  return absl::OkStatus();
}

} // namespace tensorcast::communicator
