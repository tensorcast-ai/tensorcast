// Copyright (c) 2025, TensorCast Team.

#include "core/store/components/eviction_service.h"

#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/common/memory/memory_location.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/device_types.h"

namespace tensorcast::store::components {

namespace detail {

absl::Status evict_core(
    absl::FunctionRef<std::vector<loading::ReplicaKey>()> get_candidates,
    absl::FunctionRef<absl::Status(const loading::ReplicaKey&)> try_release,
    absl::FunctionRef<void()> record_eviction,
    absl::FunctionRef<bool()> has_freed_enough,
    absl::FunctionRef<void(const loading::ReplicaKey&)> on_evicted) {
  auto candidates = get_candidates();
  for (const auto& key : candidates) {
    if (!try_release(key).ok()) {
      continue;
    }
    record_eviction();
    on_evicted(key);
    if (has_freed_enough()) {
      return absl::OkStatus();
    }
  }
  return absl::ResourceExhaustedError("Could not free enough memory");
}

} // namespace detail

absl::Status evict_for_gpu(
    ReplicaRegistry& registry,
    DeviceManager& device_manager,
    MetricsCollector& metrics,
    int device_id,
    size_t required_bytes) {
  auto free_before_or = device_manager.get_free_memory(device_id);
  if (!free_before_or.ok()) {
    return free_before_or.status();
  }
  const size_t free_before = free_before_or.value();

  return detail::evict_core(
      [&] { return registry.get_lru_instances(); },
      [&](const loading::ReplicaKey& key) -> absl::Status {
        if (key.device.type != DeviceType::GPU || key.device.ordinal != device_id) {
          return absl::FailedPreconditionError("not target device");
        }
        auto replica_or = registry.find(key);
        if (!replica_or.ok()) {
          return replica_or.status();
        }
        const auto& replica = replica_or.value();
        if (replica->get_memory_state(common::memory::MemoryLocation::GPU) != replica::MemoryState::LOADED) {
          return absl::FailedPreconditionError("not loaded");
        }
        return replica->release_memory(common::memory::MemoryLocation::GPU);
      },
      [&] { metrics.record_memory_eviction(); },
      [&] {
        auto free_now_or = device_manager.get_free_memory(device_id);
        return free_now_or.ok() && (free_now_or.value() - free_before) >= required_bytes;
      },
      [](const loading::ReplicaKey&) {});
}

absl::Status evict_for_cpu(
    ReplicaRegistry& registry,
    tensorcast::common::memory::PinnedBufferPool& memory_pool,
    MetricsCollector& metrics,
    size_t required_pinned_pool_bytes) {
  return detail::evict_core(
      [&] { return registry.get_lru_instances(); },
      [&](const loading::ReplicaKey& key) -> absl::Status {
        auto replica_or = registry.find(key);
        if (!replica_or.ok()) {
          return replica_or.status();
        }
        return replica_or.value()->release_memory(common::memory::MemoryLocation::CPU);
      },
      [&] { metrics.record_memory_eviction(); },
      [&] { return memory_pool.get_available_size() >= required_pinned_pool_bytes; },
      [&](const loading::ReplicaKey& key) {
        LOG(INFO) << "Evicted replica " << key.artifact_id
                  << " (device=" << (key.device.type == DeviceType::CPU ? "CPU" : "GPU") << ":" << key.device.ordinal
                  << ") from CPU memory";
      });
}

} // namespace tensorcast::store::components
