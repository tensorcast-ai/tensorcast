// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/replica/replica_runtime.h"

#include <algorithm>
#include <format>
#include <limits>
#include <optional>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/cuda/cuda_api.h"
#include "core/store/components/eviction_service.h"
#include "core/store/device_registry.h"

namespace tensorcast::store::runtime {

namespace {

DeviceKey normalize_device_key(DeviceKey key) {
  if (key.type == DeviceType::GPU) {
    if (key.ordinal < 0) {
      key.ordinal = 0;
    }
    return DeviceRegistry::instance().normalize(key);
  }
  if (key.type == DeviceType::CPU) {
    return DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  }
  return key;
}

loading::ReplicaKey normalize_replica_key(const loading::ReplicaKey& key) {
  loading::ReplicaKey normalized = key;
  normalized.device = normalize_device_key(key.device);
  return normalized;
}

uint64_t compute_bytes_for_chunks(
    const replica::UnifiedMemoryAuthority::ArtifactLayout& layout,
    absl::Span<const uint32_t> chunk_ids) {
  if (layout.artifact_chunk_bytes == 0 || layout.artifact_bytes == 0) {
    return 0;
  }
  std::vector<uint32_t> ids(chunk_ids.begin(), chunk_ids.end());
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  uint64_t total = 0;
  for (uint32_t idx : ids) {
    const uint64_t start = static_cast<uint64_t>(idx) * static_cast<uint64_t>(layout.artifact_chunk_bytes);
    if (start >= layout.artifact_bytes) {
      continue;
    }
    const uint64_t end =
        std::min<uint64_t>(layout.artifact_bytes, start + static_cast<uint64_t>(layout.artifact_chunk_bytes));
    if (end > start) {
      total += end - start;
    }
  }
  return total;
}

struct ResolvedUmaReplica {
  loading::ReplicaKey registry_key;
  loading::ReplicaKey uma_key;
  std::shared_ptr<replica::Replica> replica;
};

std::optional<loading::ReplicaKey> resolve_uma_key_for_replica(
    const loading::ReplicaKey& requested_key,
    const std::shared_ptr<replica::Replica>& replica_instance) {
  auto uma = replica_instance->get_memory_manager().memory_authority();
  if (!uma) {
    return std::nullopt;
  }
  const loading::ReplicaKey normalized_requested = normalize_replica_key(requested_key);
  if (uma->has_allocation(normalized_requested)) {
    return normalized_requested;
  }
  const loading::ReplicaKey physical_key = normalize_replica_key(replica_instance->replica_key());
  if (uma->has_allocation(physical_key)) {
    return physical_key;
  }
  return std::nullopt;
}

absl::StatusOr<ResolvedUmaReplica> resolve_replica_and_uma_key(
    const components::ReplicaRegistry& registry,
    const loading::ReplicaKey& requested_key) {
  const loading::ReplicaKey normalized_requested = normalize_replica_key(requested_key);
  auto replica_or = registry.find(normalized_requested);
  if (replica_or.ok()) {
    auto uma_key = resolve_uma_key_for_replica(normalized_requested, *replica_or);
    if (!uma_key.has_value()) {
      return absl::NotFoundError("Replica not found in unified memory");
    }
    return ResolvedUmaReplica{
        .registry_key = normalized_requested,
        .uma_key = *uma_key,
        .replica = *replica_or,
    };
  }

  const auto candidates = registry.find_by_device(normalize_device_key(normalized_requested.device));
  for (const auto& candidate_key : candidates) {
    auto candidate_or = registry.find(candidate_key);
    if (!candidate_or.ok()) {
      continue;
    }
    auto uma = candidate_or.value()->get_memory_manager().memory_authority();
    if (uma == nullptr || !uma->has_allocation(normalized_requested)) {
      continue;
    }
    return ResolvedUmaReplica{
        .registry_key = candidate_key,
        .uma_key = normalized_requested,
        .replica = *candidate_or,
    };
  }

  return absl::NotFoundError("Replica not found");
}

} // namespace

ReplicaRuntime::ReplicaRuntime(gsl::not_null<RuntimeContext*> context)
    : ReplicaRuntime(Config{.runtime_context = context}) {}

ReplicaRuntime::ReplicaRuntime(Config config) : context_(config.runtime_context) {
  if (context_ != nullptr) {
    event_publisher_ = context_->event_publisher();
    if (context_->ingestion_event_hub() != nullptr) {
      ingestion_event_subscription_ = context_->ingestion_event_hub()->subscribe_completed(
          [this](const IngestionCompletedEvent& event) { record_ingestion_result(event); });
    }
  }
}

size_t ReplicaRuntime::get_available_memory() const {
  return pinned_pool()->get_available_size();
}

void ReplicaRuntime::update_memory_pool_metrics() {
  metrics().update_all_metrics(*pinned_pool(), registry(), device_manager());
  if (context_ != nullptr && context_->options().pinned_memory_authority) {
    metrics().update_pinned_authority_metrics(*context_->options().pinned_memory_authority);
  }
}

std::vector<ReplicaInfo> ReplicaRuntime::get_all_replicas_info() const {
  std::vector<ReplicaInfo> result;

  const auto replica_keys = registry().get_lru_instances();

  for (const auto& key : replica_keys) {
    auto replica_or = registry().find(key);
    if (!replica_or.ok()) {
      continue;
    }

    const auto& replica = replica_or.value();
    ReplicaInfo info;
    info.key = key;
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

std::vector<ReplicaInventoryEntry> ReplicaRuntime::get_ha_inventory() const {
  std::vector<ReplicaInventoryEntry> result;

  const auto replica_keys = registry().get_lru_instances();
  result.reserve(replica_keys.size());

  auto is_present = [](replica::MemoryState st) {
    return st == replica::MemoryState::ALLOCATED || st == replica::MemoryState::LOADING ||
        st == replica::MemoryState::LOADED;
  };

  for (const auto& key : replica_keys) {
    auto replica_or = registry().find(key);
    if (!replica_or.ok()) {
      continue;
    }
    const auto& replica = replica_or.value();

    const bool is_gpu = key.device.type == DeviceType::GPU;
    if (is_gpu && key.device.ordinal < 0) {
      continue;
    }
    const auto location = is_gpu ? common::memory::MemoryLocation::GPU : common::memory::MemoryLocation::CPU;
    const auto state = replica->get_memory_state(location);
    if (!is_present(state)) {
      continue;
    }

    const auto publish_state = get_replica_publish_state(key);
    if (publish_state != ReplicaPublishState::kPublishPending && publish_state != ReplicaPublishState::kPublished) {
      continue;
    }

    ReplicaInventoryEntry entry;
    entry.key = key;
    entry.size_bytes = get_replica_size_or_zero(key);
    entry.memory_location = location;
    const auto transport_state = get_transport_state(key);
    entry.export_state = transport_state.export_state;
    entry.export_generation = transport_state.export_generation;
    entry.remote_memory_keys = transport_state.remote_memory_keys;
    entry.buffer_sizes = transport_state.buffer_sizes;
    entry.verification_json = transport_state.verification_json;
    const bool comm_enabled = communication_manager()->is_enabled();
    entry.is_available = comm_enabled && state == replica::MemoryState::LOADED;
    entry.publish_state = publish_state;
    result.push_back(std::move(entry));
  }

  return result;
}

std::vector<DeviceKey> ReplicaRuntime::get_resident_devices(
    std::string_view artifact_id,
    std::optional<std::string_view> view_id) const {
  absl::flat_hash_set<DeviceKey, DeviceKeyHash> unique_devices;
  std::vector<DeviceKey> devices;

  const auto replica_keys = registry().find_by_artifact(artifact_id);
  for (const auto& key : replica_keys) {
    if (view_id.has_value()) {
      if (view_id->empty()) {
        if (key.view_id.has_value() && !key.view_id->empty()) {
          continue;
        }
      } else if (!key.view_id.has_value() || key.view_id->empty() || *key.view_id != *view_id) {
        continue;
      }
    } else if (key.view_id.has_value() && !key.view_id->empty()) {
      continue;
    }
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

absl::StatusOr<int> ReplicaRuntime::get_unique_gpu_residency(
    std::string_view artifact_id,
    std::optional<std::string_view> view_id) const {
  int unique_gpu_device = -2; // -2: unknown, -1: none, >=0: unique device
  for (const auto& info : get_all_replicas_info()) {
    if (info.artifact_id != artifact_id) {
      continue;
    }
    if (view_id.has_value()) {
      if (view_id->empty()) {
        if (info.key.view_id.has_value() && !info.key.view_id->empty()) {
          continue;
        }
      } else if (!info.key.view_id.has_value() || info.key.view_id->empty() || *info.key.view_id != *view_id) {
        continue;
      }
    } else if (info.key.view_id.has_value() && !info.key.view_id->empty()) {
      continue;
    }
    if (info.gpu_state == common::memory::MemoryLocation::GPU) {
      if (unique_gpu_device == -2) {
        unique_gpu_device = info.gpu_device_id;
      } else if (unique_gpu_device != info.gpu_device_id) {
        return absl::InvalidArgumentError("ambiguous artifact residency across multiple GPUs; device_id required");
      }
    }
  }
  if (unique_gpu_device == -2) {
    return -1;
  }
  return unique_gpu_device;
}

std::vector<loading::ReplicaKey> ReplicaRuntime::list_device_replicas(const DeviceKey& device) const {
  std::vector<loading::ReplicaKey> list;
  const auto inst_keys = registry().find_by_device(normalize_device_key(device));
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

int ReplicaRuntime::wait_replica_ready(const loading::ReplicaKey& key) const {
  const auto normalized_key = normalize_replica_key(key);
  auto replica_or = registry().find(normalized_key);
  if (!replica_or.ok()) {
    return 1;
  }
  const auto& replica = replica_or.value();
  common::memory::MemoryLocation loc = (normalized_key.device.type == DeviceType::CPU)
      ? common::memory::MemoryLocation::CPU
      : common::memory::MemoryLocation::GPU;
  absl::Status st = replica->wait_until_loaded(loc, absl::InfiniteDuration());
  return st.ok() ? 0 : 1;
}

absl::Status ReplicaRuntime::unload_replica_status(const loading::ReplicaKey& key) const {
  const auto normalized_key = normalize_replica_key(key);
  auto replica_or = registry().find(normalized_key);
  if (!replica_or.ok()) {
    VLOG(1) << "ReplicaRuntime::unload_replica artifact=" << normalized_key.artifact_id
            << " device=" << normalized_key.device.ordinal << " not found in registry: " << replica_or.status();
    return absl::NotFoundError(
        std::format(
            "ReplicaRuntime::unload_replica: artifact={} device={} not found in registry: {}",
            normalized_key.artifact_id,
            normalized_key.device.ordinal,
            replica_or.status().ToString()));
  }

  const auto& replica = replica_or.value();
  size_t replica_size = 0;
  if (auto size_or = replica->get_artifact_size(); size_or.ok()) {
    replica_size = *size_or;
  }
  common::memory::MemoryLocation loc = (normalized_key.device.type == DeviceType::CPU)
      ? common::memory::MemoryLocation::CPU
      : common::memory::MemoryLocation::GPU;

  absl::Status unexport_status = disable_remote_access_for_replica(normalized_key, replica, loc);
  if (!unexport_status.ok()) {
    return absl::Status(
        unexport_status.code(),
        std::format(
            "ReplicaRuntime::unload_replica: disable_remote_memory_access failed for artifact={} device={} "
            "location={} status={}",
            normalized_key.artifact_id,
            normalized_key.device.ordinal,
            common::memory::location_to_string(loc),
            unexport_status.ToString()));
  }

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
      VLOG(1) << "ReplicaRuntime::unload_replica artifact=" << normalized_key.artifact_id
              << " device=" << normalized_key.device.ordinal
              << ": no allocation observed after probe window; treating as no-op unload.";
      return absl::OkStatus();
    }

    before_state = observed_state;
  }

  absl::Status release_status = replica->release_memory(loc);

  if (absl::IsFailedPrecondition(release_status)) {
    constexpr absl::Duration kUnloadRetryTimeout = absl::Seconds(30);
    absl::Status wait_status = replica->wait_until_loaded(loc, kUnloadRetryTimeout);

    if (!wait_status.ok() && !absl::IsFailedPrecondition(wait_status)) {
      const auto current_state = replica->get_memory_state(loc);
      VLOG(1) << "ReplicaRuntime::unload_replica artifact=" << normalized_key.artifact_id
              << " device=" << normalized_key.device.ordinal << ": wait for load completion returned " << wait_status;
      return absl::Status(
          wait_status.code(),
          std::format(
              "ReplicaRuntime::unload_replica: wait_until_loaded failed for artifact={} device={} location={} "
              "state={} status={}",
              normalized_key.artifact_id,
              normalized_key.device.ordinal,
              common::memory::location_to_string(loc),
              replica::state_to_string(current_state),
              wait_status.ToString()));
    }

    release_status = replica->release_memory(loc);
  }

  if (!release_status.ok()) {
    const auto current_state = replica->get_memory_state(loc);
    VLOG(1) << "ReplicaRuntime::unload_replica artifact=" << normalized_key.artifact_id
            << " device=" << normalized_key.device.ordinal << ": unload failed with " << release_status;
    return absl::Status(
        release_status.code(),
        std::format(
            "ReplicaRuntime::unload_replica: release_memory failed for artifact={} device={} location={} state={} "
            "status={}",
            normalized_key.artifact_id,
            normalized_key.device.ordinal,
            common::memory::location_to_string(loc),
            replica::state_to_string(current_state),
            release_status.ToString()));
  }

  publish_replica_event(RuntimeEventType::kReplicaEvicted, normalized_key, replica_size);
  return absl::OkStatus();
}

int ReplicaRuntime::unload_replica(const loading::ReplicaKey& key) const {
  absl::Status unload_status = unload_replica_status(normalize_replica_key(key));
  if (unload_status.ok()) {
    return 0;
  }
  if (absl::IsNotFound(unload_status)) {
    return 1;
  }
  return -1;
}

absl::Status ReplicaRuntime::retire_replica_status(const loading::ReplicaKey& key) {
  const auto normalized_key = normalize_replica_key(key);
  auto removed = registry().erase(normalized_key);
  if (!removed.has_value()) {
    return absl::NotFoundError(
        std::format(
            "ReplicaRuntime::retire_replica: artifact={} device={} not found in registry",
            normalized_key.artifact_id,
            normalized_key.device.ordinal));
  }

  auto [removed_key, replica] = std::move(*removed);
  const loading::ReplicaKey canonical_key = normalize_replica_key(removed_key);
  size_t replica_size = 0;
  if (auto size_or = replica->get_artifact_size(); size_or.ok()) {
    replica_size = *size_or;
  }
  bool released_any = false;
  std::vector<absl::Status> release_errors;
  teardown_replica_memory(
      canonical_key,
      replica,
      /*release_cpu=*/true,
      /*release_gpu=*/true,
      &released_any,
      &release_errors);
  if (released_any) {
    publish_replica_event(RuntimeEventType::kReplicaEvicted, canonical_key, replica_size);
  }

  clear_replica_runtime_state(canonical_key);
  if (auto stable_cache = context_->stable_cache_manager(); stable_cache != nullptr) {
    stable_cache->on_replica_evicted(canonical_key, "deregister");
  }
  metrics().update_all_metrics(*pinned_pool(), registry(), device_manager());

  if (!release_errors.empty()) {
    return release_errors.front();
  }
  return absl::OkStatus();
}

replica::MemoryState ReplicaRuntime::get_replica_state(const loading::ReplicaKey& key, DeviceType memory_type) const {
  auto replica_or = registry().find(normalize_replica_key(key));
  if (!replica_or.ok()) {
    return replica::MemoryState::UNINITIALIZED;
  }
  common::memory::MemoryLocation loc =
      (memory_type == DeviceType::CPU) ? common::memory::MemoryLocation::CPU : common::memory::MemoryLocation::GPU;
  return replica_or.value()->get_memory_state(loc);
}

absl::StatusOr<uint64_t> ReplicaRuntime::get_replica_gpu_ptr(const loading::ReplicaKey& key) const {
  if (key.device.type != DeviceType::GPU) {
    return absl::InvalidArgumentError("ReplicaKey does not reference a GPU device");
  }
  auto replica_or = registry().find(normalize_replica_key(key));
  if (!replica_or.ok()) {
    return absl::NotFoundError("Replica not found");
  }
  const auto ptrs = replica_or.value()->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
  if (ptrs.empty() || ptrs[0] == nullptr) {
    return absl::FailedPreconditionError("GPU memory not available");
  }
  return reinterpret_cast<uint64_t>(ptrs[0]);
}

absl::StatusOr<uint64_t> ReplicaRuntime::get_replica_size(const loading::ReplicaKey& key) const {
  auto replica_or = registry().find(normalize_replica_key(key));
  if (!replica_or.ok()) {
    return absl::NotFoundError("Replica not found");
  }
  auto size_or = replica_or.value()->get_artifact_size();
  if (!size_or.ok()) {
    return size_or.status();
  }
  return *size_or;
}

void ReplicaRuntime::set_replica_publish_state(const loading::ReplicaKey& key, ReplicaPublishState state) {
  absl::MutexLock lock(&publish_state_mu_);
  publish_states_[normalize_replica_key(key)] = state;
}

ReplicaPublishState ReplicaRuntime::get_replica_publish_state(const loading::ReplicaKey& key) const {
  absl::MutexLock lock(&publish_state_mu_);
  auto it = publish_states_.find(normalize_replica_key(key));
  if (it == publish_states_.end()) {
    return ReplicaPublishState::kLocalOnly;
  }
  return it->second;
}

ReplicaTransportState ReplicaRuntime::get_transport_state(const loading::ReplicaKey& key) const {
  absl::MutexLock lock(&transport_state_mu_);
  auto it = transport_states_.find(normalize_replica_key(key));
  if (it == transport_states_.end()) {
    return ReplicaTransportState{};
  }
  return it->second;
}

void ReplicaRuntime::update_transport_state(const loading::ReplicaKey& key, const ReplicaTransportState& state) {
  absl::MutexLock lock(&transport_state_mu_);
  transport_states_[normalize_replica_key(key)] = state;
}

void ReplicaRuntime::clear_transport_state(const loading::ReplicaKey& key) {
  absl::MutexLock lock(&transport_state_mu_);
  transport_states_.erase(normalize_replica_key(key));
}

void ReplicaRuntime::set_replica_global_id(const loading::ReplicaKey& key, std::string replica_id) {
  if (replica_id.empty()) {
    return;
  }
  absl::MutexLock lock(&replica_id_mu_);
  replica_ids_[normalize_replica_key(key)] = std::move(replica_id);
}

std::optional<std::string> ReplicaRuntime::get_replica_global_id(const loading::ReplicaKey& key) const {
  absl::MutexLock lock(&replica_id_mu_);
  auto it = replica_ids_.find(normalize_replica_key(key));
  if (it == replica_ids_.end()) {
    return std::nullopt;
  }
  return it->second;
}

void ReplicaRuntime::clear_replica_global_id(const loading::ReplicaKey& key) {
  absl::MutexLock lock(&replica_id_mu_);
  replica_ids_.erase(normalize_replica_key(key));
}

absl::StatusOr<ExportRegistration> ReplicaRuntime::enable_remote_replica_access(
    const loading::ReplicaKey& key,
    common::memory::MemoryLocation location) const {
  const auto normalized_key = normalize_replica_key(key);
  auto comm = communication_manager();
  if (!comm || !comm->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }
  auto replica_or = registry().find(normalized_key);
  if (!replica_or.ok()) {
    return absl::NotFoundError("Replica not found");
  }
  auto registration_or = replica_or.value()->enable_remote_memory_access(location, comm->get_engine());
  if (registration_or.ok()) {
    publish_remote_access_event(normalized_key, location, true);
  }
  return registration_or;
}

absl::Status ReplicaRuntime::disable_remote_replica_access(
    const loading::ReplicaKey& key,
    common::memory::MemoryLocation location) const {
  const auto normalized_key = normalize_replica_key(key);
  auto comm = communication_manager();
  if (!comm || !comm->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }
  auto replica_or = registry().find(normalized_key);
  if (!replica_or.ok()) {
    return absl::NotFoundError("Replica not found");
  }
  absl::Status status = replica_or.value()->disable_remote_memory_access(location, comm->get_engine());
  if (status.ok()) {
    publish_remote_access_event(normalized_key, location, false);
  }
  return status;
}

absl::StatusOr<replica::UnifiedMemoryAuthority::ExportRegistration> ReplicaRuntime::set_replica_exported(
    const loading::ReplicaKey& key,
    common::memory::MemoryLocation location,
    absl::Span<const uint32_t> chunks,
    bool on) const {
  auto resolved_or = resolve_replica_and_uma_key(registry(), key);
  if (!resolved_or.ok()) {
    return resolved_or.status();
  }
  auto uma = resolved_or->replica->get_memory_manager().memory_authority();
  if (!uma) {
    return absl::FailedPreconditionError("UMA is unavailable for replica");
  }
  return uma->set_exported(resolved_or->uma_key, location, chunks, on);
}

absl::StatusOr<replica::UnifiedMemoryAuthority::StableLease> ReplicaRuntime::acquire_replica_stable_lease(
    const loading::ReplicaKey& key,
    absl::Span<const uint32_t> chunks) const {
  auto resolved_or = resolve_replica_and_uma_key(registry(), key);
  if (!resolved_or.ok()) {
    return resolved_or.status();
  }
  auto uma = resolved_or->replica->get_memory_manager().memory_authority();
  if (!uma) {
    return absl::FailedPreconditionError("UMA is unavailable for replica");
  }
  auto lease_or = uma->acquire_stable_lease(resolved_or->uma_key, chunks);
  if (lease_or.ok() || !absl::IsResourceExhausted(lease_or.status())) {
    return lease_or;
  }
  auto stable_cache = context_->stable_cache_manager();
  if (!stable_cache) {
    return lease_or;
  }
  auto layout_or = uma->get_layout(resolved_or->uma_key);
  if (!layout_or.ok()) {
    return lease_or;
  }
  const uint64_t required_bytes = compute_bytes_for_chunks(*layout_or, chunks);
  if (required_bytes == 0) {
    return lease_or;
  }
  const absl::Status evict_status = stable_cache->preempt_for_export(required_bytes, resolved_or->registry_key);
  if (!evict_status.ok()) {
    LOG(WARNING) << "stable_cache.preempt_for_export failed: key=" << resolved_or->registry_key
                 << " bytes=" << required_bytes << " status=" << evict_status;
    return lease_or;
  }
  return uma->acquire_stable_lease(resolved_or->uma_key, chunks);
}

absl::Status ReplicaRuntime::release_replica_stable_lease(
    const replica::UnifiedMemoryAuthority::StableLease& lease) const {
  auto resolved_or = resolve_replica_and_uma_key(registry(), lease.key);
  if (!resolved_or.ok()) {
    return resolved_or.status();
  }
  auto uma = resolved_or->replica->get_memory_manager().memory_authority();
  if (!uma) {
    return absl::FailedPreconditionError("UMA is unavailable for replica");
  }
  auto adjusted = lease;
  adjusted.key = resolved_or->uma_key;
  return uma->release_stable_lease(adjusted);
}

absl::Status ReplicaRuntime::try_evict_memory_for_replica(size_t required_size) {
  auto stable_cache = context_->stable_cache_manager();
  const absl::Time now = absl::Now();
  return tensorcast::store::components::detail::evict_core(
      [&] { return registry().get_lru_instances(); },
      [&](const loading::ReplicaKey& key) -> absl::Status {
        if (stable_cache && !stable_cache->is_evictable(key, now)) {
          return absl::FailedPreconditionError("stable cache policy prevents eviction");
        }
        auto replica_or = registry().find(key);
        if (!replica_or.ok()) {
          return replica_or.status();
        }
        return replica_or.value()->release_memory(common::memory::MemoryLocation::CPU);
      },
      [&] { metrics().record_memory_eviction(); },
      [&] { return pinned_pool()->get_available_size() >= required_size; },
      [&](const loading::ReplicaKey& evicted_key) {
        if (stable_cache) {
          stable_cache->on_replica_evicted(evicted_key, "pinned_pool_eviction");
        }
        publish_replica_event(RuntimeEventType::kReplicaEvicted, evicted_key, get_replica_size_or_zero(evicted_key));
      });
}

std::shared_ptr<replica::Replica> ReplicaRuntime::get_or_create_replica(
    const std::string& artifact_identifier,
    replica::ReplicaConfig config) {
  DeviceKey device_key;
  device_key.type = config.device_type;
  if (config.device_type == DeviceType::GPU) {
    const int ordinal = (config.local_device_id >= 0) ? config.local_device_id : 0;
    device_key = DeviceRegistry::instance().gpu_key(ordinal);
  } else {
    device_key.type = DeviceType::CPU;
    device_key.ordinal = -1;
    device_key.uuid.clear();
  }
  loading::ReplicaKey inst_key{
      .artifact_id = artifact_identifier,
      .view_id = config.view_id,
      .device = device_key,
  };

  if (!config.memory_tier_config.has_value() && context_->options().memory_tier_config.has_value()) {
    config.memory_tier_config = context_->options().memory_tier_config;
  }
  config.byte_mapping_config = context_->options().byte_mapping;
  config.cpu_shared_memory_enabled = context_->options().cpu_shared_memory_enabled;

  if (auto existing_or = registry().find(inst_key); existing_or.ok()) {
    return existing_or.value();
  }

  auto replica_create_or = replica::Replica::create(config);
  if (!replica_create_or.ok()) {
    LOG(ERROR) << "Failed to create replica: " << replica_create_or.status().message();
    return nullptr;
  }
  auto replica = std::shared_ptr<replica::Replica>(std::move(replica_create_or.value()));
  if (auto budget = context_->memory_tier_budget(); budget) {
    replica->get_memory_manager().set_memory_tier_budget(budget);
  }

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
    absl::Status load_status = std::move(replica->ensure_loaded_async(common::memory::MemoryLocation::CPU)).get();
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

int ReplicaRuntime::clear_mem() {
  auto replicas = registry().clear_all();
  absl::flat_hash_set<const replica::Replica*> processed;
  processed.reserve(replicas.size());
  std::vector<absl::Status> errors;

  for (const auto& [inst_key, _] : replicas) {
    clear_replica_runtime_state(inst_key);
  }

  for (const auto& entry : replicas) {
    const auto& replica = entry.second;
    if (!replica) {
      continue;
    }
    if (!processed.insert(replica.get()).second) {
      continue;
    }

    const loading::ReplicaKey canonical_key = normalize_replica_key(replica->replica_key());
    clear_replica_runtime_state(canonical_key);

    size_t replica_size = 0;
    if (auto size_or = replica->get_artifact_size(); size_or.ok()) {
      replica_size = *size_or;
    }
    bool released_any = false;
    teardown_replica_memory(
        canonical_key,
        replica,
        /*release_cpu=*/true,
        /*release_gpu=*/true,
        &released_any,
        &errors);
    if (released_any) {
      publish_replica_event(RuntimeEventType::kReplicaEvicted, canonical_key, replica_size);
    }
    if (auto stable_cache = context_->stable_cache_manager(); stable_cache != nullptr) {
      stable_cache->on_replica_evicted(canonical_key, "clear_mem");
    }
  }

  metrics().update_all_metrics(*pinned_pool(), registry(), device_manager());

  if (!errors.empty()) {
    LOG(ERROR) << "Failed to release memory for " << errors.size() << " replica(s) during shutdown";
    return -1;
  }

  return 0;
}

std::vector<replica::ChunkState> ReplicaRuntime::get_chunk_states_telemetry(std::string_view artifact_id) const {
  return get_chunk_states_cpu_uma(artifact_id);
}

std::vector<replica::ChunkState> ReplicaRuntime::get_chunk_states_for_device(
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

std::vector<replica::ChunkState> ReplicaRuntime::get_chunk_states_cpu_uma(std::string_view artifact_id) const {
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

absl::StatusOr<size_t> ReplicaRuntime::get_device_total_memory(int device_id) const {
  auto info_or = device_manager().get_gpu_info(device_id);
  if (!info_or.ok()) {
    return info_or.status();
  }
  return static_cast<size_t>((*info_or)->total_memory);
}

absl::StatusOr<size_t> ReplicaRuntime::get_device_free_memory(int device_id) const {
  return device_manager().get_free_memory(device_id);
}

void ReplicaRuntime::record_ingestion_result(const IngestionResultEvent& event) {
  auto& metrics = context_->metrics_collector();
  const bool success = event.status.ok();
  const bool from_p2p = event.source == IngestionSource::kP2P;

  if (from_p2p) {
    metrics.record_p2p_transfer(success ? event.bytes_transferred : 0, success);
  }

  metrics.record_operation(from_p2p ? "load_from_p2p" : "load_from_disk", event.duration_seconds);

  const std::string device_scope = (event.target_device.type == DeviceType::GPU) ? "gpu" : "cpu";
  std::optional<std::string_view> view_scope;
  if (event.view_id.has_value()) {
    view_scope = std::string_view(*event.view_id);
  }
  metrics.record_artifact_load(
      from_p2p ? "remote" : "disk", device_scope, success ? "finalize" : "error", event.duration_seconds, view_scope);

  if (success) {
    if (event.replica_key.has_value()) {
      set_replica_publish_state(
          *event.replica_key,
          event.publish_to_global_store ? ReplicaPublishState::kPublishPending : ReplicaPublishState::kLocalOnly);
    }
    update_memory_pool_metrics();
    if (event.replica_key.has_value()) {
      publish_replica_event(
          RuntimeEventType::kReplicaLoaded, *event.replica_key, get_replica_size_or_zero(*event.replica_key));
    }
  }
}

ReplicaRegistry& ReplicaRuntime::registry() {
  return context_->replica_registry();
}

const ReplicaRegistry& ReplicaRuntime::registry() const {
  return context_->replica_registry();
}

DeviceManager& ReplicaRuntime::device_manager() {
  return context_->device_manager();
}

const DeviceManager& ReplicaRuntime::device_manager() const {
  return context_->device_manager();
}

std::shared_ptr<common::memory::PinnedBufferPool> ReplicaRuntime::pinned_pool() const {
  return context_->pinned_buffer_pool();
}

std::shared_ptr<CommunicationManager> ReplicaRuntime::communication_manager() const {
  return context_->communication_manager();
}

MetricsCollector& ReplicaRuntime::metrics() {
  return context_->metrics_collector();
}

const MetricsCollector& ReplicaRuntime::metrics() const {
  return context_->metrics_collector();
}

absl::Status ReplicaRuntime::disable_remote_access_for_replica(
    const loading::ReplicaKey& key,
    const std::shared_ptr<replica::Replica>& replica,
    common::memory::MemoryLocation location) const {
  auto comm = communication_manager();
  if (!comm || !comm->is_enabled() || !replica) {
    return absl::OkStatus();
  }
  absl::Status status = replica->disable_remote_memory_access(location, comm->get_engine());
  if (absl::IsNotFound(status)) {
    return absl::OkStatus();
  }
  if (status.ok()) {
    publish_remote_access_event(key, location, /*enabled=*/false);
  }
  return status;
}

void ReplicaRuntime::teardown_replica_memory(
    const loading::ReplicaKey& key,
    const std::shared_ptr<replica::Replica>& replica,
    bool release_cpu,
    bool release_gpu,
    bool* released_any,
    std::vector<absl::Status>* errors) const {
  if (released_any == nullptr || errors == nullptr || !replica) {
    return;
  }

  auto release_location = [&](common::memory::MemoryLocation location) {
    const std::string location_name = common::memory::location_to_string(location);
    absl::Status unexport_status = disable_remote_access_for_replica(key, replica, location);
    if (!unexport_status.ok()) {
      LOG(WARNING) << "Failed to disable remote access for " << key << " at " << location_name << ": "
                   << unexport_status;
      errors->push_back(unexport_status);
    }

    absl::Status release_status = replica->release_memory(location);
    if (!release_status.ok() && !absl::IsNotFound(release_status)) {
      LOG(WARNING) << "Failed to release " << location_name << " memory for " << key << ": " << release_status;
      errors->push_back(release_status);
      return;
    }
    if (release_status.ok()) {
      *released_any = true;
    }
  };

  if (release_cpu) {
    release_location(common::memory::MemoryLocation::CPU);
  }
  if (release_gpu) {
    release_location(common::memory::MemoryLocation::GPU);
  }
}

void ReplicaRuntime::clear_replica_runtime_state(const loading::ReplicaKey& key) {
  const loading::ReplicaKey normalized_key = normalize_replica_key(key);
  {
    absl::MutexLock lock(&publish_state_mu_);
    publish_states_.erase(normalized_key);
  }
  clear_transport_state(normalized_key);
  clear_replica_global_id(normalized_key);
}

void ReplicaRuntime::publish_replica_event(RuntimeEventType type, const loading::ReplicaKey& key, size_t size_bytes)
    const {
  if (!event_publisher_) {
    return;
  }
  RuntimeEvent event;
  event.type = type;
  ReplicaLifecycleEvent payload;
  payload.key = key;
  payload.size_bytes = size_bytes;
  event.payload = std::move(payload);
  event_publisher_.publish(std::move(event));
}

void ReplicaRuntime::publish_remote_access_event(
    const loading::ReplicaKey& key,
    common::memory::MemoryLocation location,
    bool enabled) const {
  if (!event_publisher_) {
    return;
  }
  RuntimeEvent event;
  event.type = RuntimeEventType::kRemoteAccessToggled;
  RemoteAccessEvent payload;
  payload.key = key;
  payload.location = location;
  payload.enabled = enabled;
  event.payload = std::move(payload);
  event_publisher_.publish(std::move(event));
}

size_t ReplicaRuntime::get_replica_size_or_zero(const loading::ReplicaKey& key) const {
  auto size_or = get_replica_size(key);
  if (!size_or.ok()) {
    VLOG(1) << "ReplicaRuntime could not determine size for artifact=" << key.artifact_id
            << " device=" << key.device.ordinal << ": " << size_or.status();
    return 0;
  }
  return *size_or;
}

} // namespace tensorcast::store::runtime
