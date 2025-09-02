// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "core/communicator/engine/dram_stager.h"
#include "core/store/device_types.h"
#include "core/store/replica/replica_memory_coordinator.h"
#include "gsl/pointers"

namespace tensorcast::store::components {

// UMA-backed lease provider for DRAM staging.
// Maps registered tensor keys (exported DVMP ranges) to UMA + base VA offset.
class UmaLeaseProvider : public communicator::engine::DRAMStager::LeaseProvider {
 public:
  struct Entry {
    loading::ReplicaKey key;
    uint64_t base_va_off = 0;
    std::weak_ptr<replica::ReplicaMemoryCoordinator> uma;
  };

  static std::shared_ptr<UmaLeaseProvider> instance();

  // Register a mapping for a DVMP-exported tensor key.
  void register_mapping(
      const std::string& tensor_key,
      const loading::ReplicaKey& key,
      uint64_t base_va_off,
      gsl::not_null<std::shared_ptr<replica::ReplicaMemoryCoordinator>> uma);

  std::unique_ptr<communicator::engine::DRAMStager::LeaseHandle> acquire(
      const std::string& tensor_key,
      uint64_t offset,
      uint64_t bytes) override;

  // Lightweight residency query for DirectMR policy checks. Returns true only
  // if all chunks fully covering [offset, offset+bytes) are in HOT state.
  bool is_range_hot(const std::string& tensor_key, uint64_t offset, uint64_t bytes) const;

 private:
  UmaLeaseProvider() = default;

  struct TokenLeaseHandle : public communicator::engine::DRAMStager::LeaseHandle {
    explicit TokenLeaseHandle(std::shared_ptr<void> keep) : keepalive(std::move(keep)) {}
    std::shared_ptr<void> keepalive;
  };

  mutable absl::Mutex mu_;
  std::unordered_map<std::string, Entry> map_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::store::components
