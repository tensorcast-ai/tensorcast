// Copyright (c) 2025, TensorCast Team.

#include "core/store/components/eviction_service.h"

#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/memory_state.h"

using tensorcast::DeviceType;
using tensorcast::store::DeviceKey;
using tensorcast::store::components::detail::evict_core;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::loading::ReplicaKeyHash;

namespace {

struct FakeDeviceManager {
  absl::StatusOr<size_t> get_free_memory(int /*device_id*/) const {
    return free_bytes;
  }

  void increase_free(size_t delta) {
    free_bytes += delta;
  }

  size_t free_bytes{0};
};

struct FakeMemoryPool {
  size_t get_available_size() const {
    return available;
  }

  void increase(size_t delta) {
    available += delta;
  }

  size_t available{0};
};

struct FakeMetricsCollector {
  void record_memory_eviction() {
    ++eviction_count;
  }

  int eviction_count{0};
};

struct FakeReplica {
  absl::Status release_memory(tensorcast::common::memory::MemoryLocation location) {
    switch (location) {
      case tensorcast::common::memory::MemoryLocation::GPU:
        if (!gpu_loaded) {
          return absl::FailedPreconditionError("GPU not loaded");
        }
        gpu_loaded = false;
        if (device_manager != nullptr) {
          device_manager->increase_free(gpu_bytes);
        }
        return absl::OkStatus();
      case tensorcast::common::memory::MemoryLocation::CPU:
        if (!cpu_loaded) {
          return absl::FailedPreconditionError("CPU not loaded");
        }
        cpu_loaded = false;
        if (memory_pool != nullptr) {
          memory_pool->increase(cpu_bytes);
        }
        return absl::OkStatus();
      default:
        return absl::InvalidArgumentError("unsupported location");
    }
  }

  tensorcast::store::replica::MemoryState get_memory_state(tensorcast::common::memory::MemoryLocation location) const {
    switch (location) {
      case tensorcast::common::memory::MemoryLocation::GPU:
        return gpu_loaded ? tensorcast::store::replica::MemoryState::LOADED
                          : tensorcast::store::replica::MemoryState::UNALLOCATED;
      case tensorcast::common::memory::MemoryLocation::CPU:
        return cpu_loaded ? tensorcast::store::replica::MemoryState::LOADED
                          : tensorcast::store::replica::MemoryState::UNALLOCATED;
      default:
        return tensorcast::store::replica::MemoryState::UNALLOCATED;
    }
  }

  bool gpu_loaded{false};
  bool cpu_loaded{false};
  size_t gpu_bytes{0};
  size_t cpu_bytes{0};
  FakeDeviceManager* device_manager{nullptr};
  FakeMemoryPool* memory_pool{nullptr};
};

struct FakeRegistry {
  std::vector<ReplicaKey> get_lru_instances() const {
    return order;
  }

  absl::StatusOr<std::shared_ptr<FakeReplica>> find(const ReplicaKey& key) const {
    auto it = entries.find(key);
    if (it == entries.end()) {
      return absl::NotFoundError("not found");
    }
    return it->second;
  }

  std::vector<ReplicaKey> order;
  absl::flat_hash_map<ReplicaKey, std::shared_ptr<FakeReplica>, ReplicaKeyHash> entries;
};

ReplicaKey make_gpu_key(int device_id) {
  DeviceKey device{.type = DeviceType::GPU, .ordinal = device_id, .uuid = ""};
  return ReplicaKey{.artifact_id = "artifact", .view_id = std::nullopt, .device = device, .replica = 0};
}

ReplicaKey make_cpu_key() {
  DeviceKey device{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  return ReplicaKey{.artifact_id = "artifact_cpu", .view_id = std::nullopt, .device = device, .replica = 0};
}

} // namespace

TEST_CASE("EvictionService GPU path frees memory and records metrics", "[eviction][gpu]") {
  FakeDeviceManager device_manager;
  device_manager.free_bytes = 128;

  FakeMetricsCollector metrics;
  FakeRegistry registry;

  ReplicaKey key = make_gpu_key(0);
  auto replica = std::make_shared<FakeReplica>();
  replica->gpu_loaded = true;
  replica->gpu_bytes = 256;
  replica->device_manager = &device_manager;

  registry.entries.insert({key, replica});
  registry.order.push_back(key);

  const size_t required = 200;
  const size_t free_before = device_manager.free_bytes;
  auto status = evict_core(
      [&] { return registry.get_lru_instances(); },
      [&](const ReplicaKey& k) -> absl::Status {
        if (k.device.type != DeviceType::GPU || k.device.ordinal != 0) {
          return absl::FailedPreconditionError("not target device");
        }
        auto r = registry.find(k);
        if (!r.ok())
          return r.status();
        if (r.value()->get_memory_state(tensorcast::common::memory::MemoryLocation::GPU) !=
            tensorcast::store::replica::MemoryState::LOADED) {
          return absl::FailedPreconditionError("not loaded");
        }
        return r.value()->release_memory(tensorcast::common::memory::MemoryLocation::GPU);
      },
      [&] { metrics.record_memory_eviction(); },
      [&] { return (device_manager.free_bytes - free_before) >= required; },
      [](const ReplicaKey&) {});
  CHECK(status.ok());
  CHECK_FALSE(replica->gpu_loaded);
  CHECK(device_manager.free_bytes == 384);
  CHECK(metrics.eviction_count == 1);
}

TEST_CASE("EvictionService GPU path returns resource exhausted when insufficient memory", "[eviction][gpu]") {
  FakeDeviceManager device_manager;
  device_manager.free_bytes = 64;

  FakeMetricsCollector metrics;
  FakeRegistry registry;

  ReplicaKey key = make_gpu_key(1);
  auto replica = std::make_shared<FakeReplica>();
  replica->gpu_loaded = true;
  replica->gpu_bytes = 32;
  replica->device_manager = &device_manager;

  registry.entries.insert({key, replica});
  registry.order.push_back(key);

  const size_t required = 256;
  const size_t free_before = device_manager.free_bytes;
  auto status = evict_core(
      [&] { return registry.get_lru_instances(); },
      [&](const ReplicaKey& k) -> absl::Status {
        if (k.device.type != DeviceType::GPU || k.device.ordinal != 1) {
          return absl::FailedPreconditionError("not target device");
        }
        auto r = registry.find(k);
        if (!r.ok())
          return r.status();
        if (r.value()->get_memory_state(tensorcast::common::memory::MemoryLocation::GPU) !=
            tensorcast::store::replica::MemoryState::LOADED) {
          return absl::FailedPreconditionError("not loaded");
        }
        return r.value()->release_memory(tensorcast::common::memory::MemoryLocation::GPU);
      },
      [&] { metrics.record_memory_eviction(); },
      [&] { return (device_manager.free_bytes - free_before) >= required; },
      [](const ReplicaKey&) {});
  CHECK_FALSE(status.ok());
  CHECK(status.code() == absl::StatusCode::kResourceExhausted);
  CHECK_FALSE(replica->gpu_loaded);
  CHECK(metrics.eviction_count == 1);
}

TEST_CASE("EvictionService CPU path frees memory and triggers callback", "[eviction][cpu]") {
  FakeMemoryPool pool;
  FakeMetricsCollector metrics;
  FakeRegistry registry;

  ReplicaKey key = make_cpu_key();
  auto replica = std::make_shared<FakeReplica>();
  replica->cpu_loaded = true;
  replica->cpu_bytes = 512;
  replica->memory_pool = &pool;

  registry.entries.insert({key, replica});
  registry.order.push_back(key);

  bool callback_invoked = false;
  auto status = evict_core(
      [&] { return registry.get_lru_instances(); },
      [&](const ReplicaKey& k) -> absl::Status {
        auto r = registry.find(k);
        if (!r.ok())
          return r.status();
        return r.value()->release_memory(tensorcast::common::memory::MemoryLocation::CPU);
      },
      [&] { metrics.record_memory_eviction(); },
      [&] { return pool.get_available_size() >= 256; },
      [&](const ReplicaKey& evicted_key) { callback_invoked = (evicted_key.artifact_id == key.artifact_id); });

  CHECK(status.ok());
  CHECK_FALSE(replica->cpu_loaded);
  CHECK(pool.available == 512);
  CHECK(metrics.eviction_count == 1);
  CHECK(callback_invoked);
}
