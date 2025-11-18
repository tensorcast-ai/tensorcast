// Copyright (c) 2025, TensorCast Team.

#include "core/store/components/runtime/replica_service.h"

#include <limits>
#include <optional>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/cuda_api.h"
#include "core/store/components/eviction_service.h"

namespace tensorcast::store::components::runtime {

ReplicaService::ReplicaService(gsl::not_null<ComponentCatalog*> catalog) : catalog_(catalog) {}

size_t ReplicaService::get_available_memory() const {
  return pinned_pool()->get_available_size();
}

void ReplicaService::update_memory_pool_metrics() {
  metrics().update_all_metrics(*pinned_pool(), registry(), device_manager());
}

std::vector<ReplicaInfo> ReplicaService::get_all_replicas_info() const {
  std::vector<ReplicaInfo> result;

  const auto replica_keys = registry().get_lru_instances();

  for (const auto& key : replica_keys) {
    auto replica_or = registry().find(key);
    if (!replica_or.ok()) {
      continue;
    }

    const auto& replica = replica_or.value();
    ReplicaInfo info;
    info.artifact_id = key.artifact_id;

    auto size_result = replica->get_artifact_size();
    info.size_bytes = size_result.ok() ? size_result.value() : 0;

    auto cpu_state = replica->get_memory_state(common::memory::MemoryLocation::CPU);
    auto gpu_state = replica->get_memory_state(common::memory::MemoryLocation::GPU);

    auto is_present = [](replica::MemoryState st) {
      return st == replica::MemoryState::ALLOCATED || st == replica::MemoryState::LOADING ||
          st == replica::MemoryState::LOADED;
    };

    info.cpu_state = is_present(cpu_state) ? common::memory::MemoryLocation::CPU : common::memory::MemoryLocation::NONE;
    info.gpu_state = is_present(gpu_state) ? common::memory::MemoryLocation::GPU : common::memory::MemoryLocation::NONE;

    info.gpu_device_id = -1;
    info.gpu_device_uuid.clear();

    if (key.device.type == DeviceType::GPU && is_present(gpu_state)) {
      info.gpu_device_id = key.device.ordinal;
      if (!key.device.uuid.empty()) {
        info.gpu_device_uuid = key.device.uuid;
      } else {
        const auto gpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
        if (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) {
          cudaPointerAttributes attrs;
          auto attr_status = cuda::pointer_get_attributes_full(gpu_ptrs[0], &attrs);
          if (attr_status.ok() && attrs.type == cudaMemoryTypeDevice) {
            auto gpu_info_result = device_manager().get_gpu_info(attrs.device);
            if (gpu_info_result.ok()) {
              info.gpu_device_uuid = (*gpu_info_result)->uuid;
            }
          }
        }
      }
    }

    info.is_registered_for_comm = communication_manager()->is_enabled() &&
        (cpu_state == replica::MemoryState::LOADED || gpu_state == replica::MemoryState::LOADED);
    auto now = std::chrono::system_clock::now();
    info.last_access_time = now;
    info.load_time = now;
    result.push_back(info);
  }

  return result;
}

std::vector<DeviceKey> ReplicaService::get_resident_devices(std::string_view artifact_id) const {
  absl::flat_hash_set<DeviceKey, DeviceKeyHash> unique_devices;
  std::vector<DeviceKey> devices;

  const auto replica_keys = registry().find_by_artifact(artifact_id);
  for (const auto& key : replica_keys) {
    auto replica_or = registry().find(key);
    if (!replica_or.ok()) {
      continue;
    }
    const auto& replica = replica_or.value();

    auto is_present = [](replica::MemoryState st) {
      return st == replica::MemoryState::ALLOCATED || st == replica::MemoryState::LOADING ||
          st == replica::MemoryState::LOADED;
    };

    if (key.device.type == DeviceType::CPU) {
      if (is_present(replica->get_memory_state(common::memory::MemoryLocation::CPU))) {
        unique_devices.insert(DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""});
      }
    } else if (key.device.type == DeviceType::GPU) {
      if (is_present(replica->get_memory_state(common::memory::MemoryLocation::GPU))) {
        unique_devices.insert(
            DeviceKey{.type = DeviceType::GPU, .ordinal = key.device.ordinal, .uuid = key.device.uuid});
      }
    }
  }

  devices.assign(unique_devices.begin(), unique_devices.end());
  return devices;
}

std::vector<loading::ReplicaKey> ReplicaService::list_device_replicas(const DeviceKey& device) const {
  std::vector<loading::ReplicaKey> list;
  const auto inst_keys = registry().find_by_device(device);
  for (const auto& key : inst_keys) {
    auto replica_or = registry().find(key);
    if (!replica_or.ok()) {
      continue;
    }
    const auto& replica = replica_or.value();
    auto is_present = [](replica::MemoryState st) {
      return st == replica::MemoryState::ALLOCATED || st == replica::MemoryState::LOADING ||
          st == replica::MemoryState::LOADED;
    };

    if (device.type == DeviceType::CPU) {
      if (is_present(replica->get_memory_state(common::memory::MemoryLocation::CPU))) {
        list.push_back(key);
      }
    } else if (device.type == DeviceType::GPU) {
      if (is_present(replica->get_memory_state(common::memory::MemoryLocation::GPU))) {
        list.push_back(key);
      }
    }
  }
  return list;
}

int ReplicaService::wait_replica_ready(const loading::ReplicaKey& key) const {
  auto replica_or = registry().find(key);
  if (!replica_or.ok()) {
    return 1;
  }
  const auto& replica = replica_or.value();
  common::memory::MemoryLocation loc =
      (key.device.type == DeviceType::CPU) ? common::memory::MemoryLocation::CPU : common::memory::MemoryLocation::GPU;
  absl::Status st = replica->wait_until_loaded(loc, absl::InfiniteDuration());
  return st.ok() ? 0 : 1;
}

int ReplicaService::unload_replica(const loading::ReplicaKey& key) const {
  auto replica_or = registry().find(key);
  if (!replica_or.ok()) {
    VLOG(1) << "ReplicaService::unload_replica artifact=" << key.artifact_id << " device=" << key.device.ordinal
            << " not found in registry: " << replica_or.status();
    return 1;
  }

  const auto& replica = replica_or.value();
  common::memory::MemoryLocation loc =
      (key.device.type == DeviceType::CPU) ? common::memory::MemoryLocation::CPU : common::memory::MemoryLocation::GPU;

  replica::MemoryState before_state = replica->get_memory_state(loc);

  if (before_state <= replica::MemoryState::UNALLOCATED) {
    constexpr absl::Duration kLoadProgressProbe = absl::Milliseconds(250);
    constexpr absl::Duration kProbeInterval = absl::Milliseconds(5);
    const absl::Time probe_deadline = absl::Now() + kLoadProgressProbe;

    replica::MemoryState observed_state = before_state;
    while (observed_state <= replica::MemoryState::UNALLOCATED && absl::Now() < probe_deadline) {
      absl::SleepFor(kProbeInterval);
      observed_state = replica->get_memory_state(loc);
    }

    if (observed_state <= replica::MemoryState::UNALLOCATED) {
      VLOG(1) << "ReplicaService::unload_replica artifact=" << key.artifact_id << " device=" << key.device.ordinal
              << ": no allocation observed after probe window; treating as no-op unload.";
      return -1;
    }

    before_state = observed_state;
  }

  absl::Status release_status = replica->release_memory(loc);

  if (absl::IsFailedPrecondition(release_status)) {
    constexpr absl::Duration kUnloadRetryTimeout = absl::Seconds(30);
    absl::Status wait_status = replica->wait_until_loaded(loc, kUnloadRetryTimeout);

    if (!wait_status.ok() && !absl::IsFailedPrecondition(wait_status)) {
      VLOG(1) << "ReplicaService::unload_replica artifact=" << key.artifact_id << " device=" << key.device.ordinal
              << ": wait for load completion returned " << wait_status;
      return -1;
    }

    release_status = replica->release_memory(loc);
  }

  if (!release_status.ok()) {
    VLOG(1) << "ReplicaService::unload_replica artifact=" << key.artifact_id << " device=" << key.device.ordinal
            << ": unload failed with " << release_status;
  }

  return release_status.ok() ? 0 : -1;
}

replica::MemoryState ReplicaService::get_replica_state(const loading::ReplicaKey& key, DeviceType memory_type) const {
  auto replica_or = registry().find(key);
  if (!replica_or.ok()) {
    return replica::MemoryState::UNINITIALIZED;
  }
  common::memory::MemoryLocation loc =
      (memory_type == DeviceType::CPU) ? common::memory::MemoryLocation::CPU : common::memory::MemoryLocation::GPU;
  return replica_or.value()->get_memory_state(loc);
}

absl::StatusOr<uint64_t> ReplicaService::get_replica_gpu_ptr(const loading::ReplicaKey& key) const {
  if (key.device.type != DeviceType::GPU) {
    return absl::InvalidArgumentError("ReplicaKey does not reference a GPU device");
  }
  auto replica_or = registry().find(key);
  if (!replica_or.ok()) {
    return absl::NotFoundError("Replica not found");
  }
  const auto ptrs = replica_or.value()->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
  if (ptrs.empty() || ptrs[0] == nullptr) {
    return absl::FailedPreconditionError("GPU memory not available");
  }
  return reinterpret_cast<uint64_t>(ptrs[0]);
}

absl::StatusOr<uint64_t> ReplicaService::get_replica_size(const loading::ReplicaKey& key) const {
  auto replica_or = registry().find(key);
  if (!replica_or.ok()) {
    return absl::NotFoundError("Replica not found");
  }
  auto size_or = replica_or.value()->get_artifact_size();
  if (!size_or.ok()) {
    return size_or.status();
  }
  return *size_or;
}

absl::StatusOr<ExportRegistration> ReplicaService::enable_remote_replica_access(
    const loading::ReplicaKey& key,
    common::memory::MemoryLocation location) const {
  auto comm = communication_manager();
  if (!comm || !comm->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }
  auto replica_or = registry().find(key);
  if (!replica_or.ok()) {
    return absl::NotFoundError("Replica not found");
  }
  return replica_or.value()->enable_remote_memory_access(location, comm->get_engine());
}

absl::Status ReplicaService::disable_remote_replica_access(
    const loading::ReplicaKey& key,
    common::memory::MemoryLocation location) const {
  auto comm = communication_manager();
  if (!comm || !comm->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }
  auto replica_or = registry().find(key);
  if (!replica_or.ok()) {
    return absl::NotFoundError("Replica not found");
  }
  return replica_or.value()->disable_remote_memory_access(location, comm->get_engine());
}

absl::Status ReplicaService::try_evict_memory_for_replica(size_t required_size) {
  return tensorcast::store::components::evict_for_cpu(registry(), *pinned_pool(), metrics(), required_size);
}

std::shared_ptr<replica::Replica> ReplicaService::get_or_create_replica(
    const std::string& artifact_identifier,
    const replica::ReplicaConfig& config) {
  DeviceKey device_key;
  device_key.type = config.device_type;
  device_key.ordinal = (config.device_type == DeviceType::GPU) ? config.local_device_id : -1;
  loading::ReplicaKey inst_key{
      .artifact_id = artifact_identifier,
      .view_id = config.view_id,
      .device = device_key,
  };

  if (auto existing_or = registry().find(inst_key); existing_or.ok()) {
    return existing_or.value();
  }

  auto replica_create_or = replica::Replica::create(config);
  if (!replica_create_or.ok()) {
    LOG(ERROR) << "Failed to create replica: " << replica_create_or.status().message();
    return nullptr;
  }
  auto replica = std::shared_ptr<replica::Replica>(std::move(replica_create_or.value()));

  const bool hydrate_inline_cpu = [&]() -> bool {
    if (config.device_type != DeviceType::CPU) {
      return false;
    }
    const auto* inline_src = std::get_if<loading::InlineBufferSource>(&config.source);
    if (inline_src == nullptr) {
      return false;
    }
    return inline_src->data != nullptr && inline_src->size_bytes > 0;
  }();

  if (hydrate_inline_cpu) {
    auto load_future = replica->ensure_loaded_async(common::memory::MemoryLocation::CPU);
    const absl::Status& load_status = load_future.get();
    if (!load_status.ok()) {
      LOG(ERROR) << "Failed to hydrate inline CPU replica " << artifact_identifier << ": " << load_status;
      return nullptr;
    }
  }

  absl::Status emplace_status = registry().emplace(inst_key, gsl::not_null<std::shared_ptr<replica::Replica>>{replica});

  if (absl::IsAlreadyExists(emplace_status)) {
    if (auto existing_or = registry().find(inst_key); existing_or.ok()) {
      return existing_or.value();
    }
  } else if (!emplace_status.ok()) {
    LOG(ERROR) << "Failed to register replica: " << emplace_status.message();
    return nullptr;
  }

  return replica;
}

int ReplicaService::clear_mem() {
  auto replicas = registry().clear_all();
  std::vector<absl::Status> errors;

  for (const auto& [inst_key, replica] : replicas) {
    auto cpu_status = replica->release_memory(common::memory::MemoryLocation::CPU);
    if (!cpu_status.ok()) {
      LOG(WARNING) << "Failed to release CPU memory for " << inst_key << ": " << cpu_status.message();
      errors.push_back(cpu_status);
    }

    auto gpu_status = replica->release_memory(common::memory::MemoryLocation::GPU);
    if (!gpu_status.ok() && !absl::IsNotFound(gpu_status)) {
      LOG(WARNING) << "Failed to release GPU memory for " << inst_key << ": " << gpu_status.message();
      errors.push_back(gpu_status);
    }
  }

  metrics().update_all_metrics(*pinned_pool(), registry(), device_manager());

  if (!errors.empty()) {
    LOG(ERROR) << "Failed to release memory for " << errors.size() << " replica(s) during shutdown";
    return -1;
  }

  return 0;
}

std::vector<replica::ChunkState> ReplicaService::get_chunk_states_telemetry(std::string_view artifact_id) const {
  return get_chunk_states_cpu_uma(artifact_id);
}

std::vector<replica::ChunkState> ReplicaService::get_chunk_states_for_device(
    std::string_view artifact_id,
    int device_id) const {
  std::vector<replica::ChunkState> out;
  auto keys = registry().find_by_artifact(artifact_id);
  for (const auto& key : keys) {
    if (key.device.type == DeviceType::GPU && key.device.ordinal == device_id) {
      auto rep_or = registry().find(key);
      if (!rep_or.ok() || !*rep_or) {
        return out;
      }
      auto& rep = *rep_or;
      auto& mm = rep->get_memory_manager();
      return mm.get_chunk_states_uma(common::memory::MemoryLocation::GPU, device_id);
    }
  }
  return out;
}

std::vector<replica::ChunkState> ReplicaService::get_chunk_states_cpu_uma(std::string_view artifact_id) const {
  std::vector<replica::ChunkState> out;
  auto keys = registry().find_by_artifact(artifact_id);
  if (keys.empty()) {
    return out;
  }
  std::optional<loading::ReplicaKey> chosen;
  int min_gpu_ord = std::numeric_limits<int>::max();
  for (const auto& k : keys) {
    if (k.device.type == DeviceType::CPU) {
      chosen = k;
      break;
    }
    if (k.device.type == DeviceType::GPU) {
      if (!chosen.has_value() || (chosen->device.type != DeviceType::CPU && k.device.ordinal < min_gpu_ord)) {
        chosen = k;
        min_gpu_ord = k.device.ordinal;
      }
    }
  }
  if (!chosen.has_value()) {
    return out;
  }

  auto rep_or = registry().find(*chosen);
  if (!rep_or.ok() || !*rep_or) {
    return out;
  }
  auto& rep = *rep_or;
  auto& mm = rep->get_memory_manager();
  return mm.get_chunk_states_uma(common::memory::MemoryLocation::CPU);
}

ReplicaRegistry& ReplicaService::registry() {
  return catalog_->replica_registry();
}

const ReplicaRegistry& ReplicaService::registry() const {
  return catalog_->replica_registry();
}

DeviceManager& ReplicaService::device_manager() {
  return catalog_->device_manager();
}

const DeviceManager& ReplicaService::device_manager() const {
  return catalog_->device_manager();
}

std::shared_ptr<common::memory::PinnedBufferPool> ReplicaService::pinned_pool() const {
  return catalog_->pinned_buffer_pool();
}

std::shared_ptr<CommunicationManager> ReplicaService::communication_manager() const {
  return catalog_->communication_manager();
}

MetricsCollector& ReplicaService::metrics() {
  return catalog_->metrics_collector();
}

const MetricsCollector& ReplicaService::metrics() const {
  return catalog_->metrics_collector();
}

} // namespace tensorcast::store::components::runtime
