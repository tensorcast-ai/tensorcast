// Copyright (c) 2025-2026, TensorCast Team.

#include "store_engine.h"

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <utility>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/memory/memory_location.h"
#include "core/communicator/misc/common.h"
#include "core/store/components/stable_dram_cache_manager.h"
#include "core/store/materialization/common/view_hash_utils.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/unified_memory_authority.h"
#include "gsl/pointers"

namespace tensorcast::store {

using common::memory::MemoryLocation;
using loading::ReplicaHandle;
using loading::ReplicaKey;
using replica::MemoryState;

namespace {

using tensorcast::store::components::MemoryTierLeaseDescriptor;

absl::StatusOr<std::vector<uint32_t>> normalize_chunk_ids_for_lease(
    const MemoryTierLeaseDescriptor& lease,
    const replica::UnifiedMemoryAuthority::ArtifactLayout& layout) {
  if (layout.artifact_chunk_bytes == 0) {
    return absl::FailedPreconditionError("artifact_chunk_bytes must be non-zero for memory tier lease binding");
  }
  const uint64_t num_chunks = (layout.artifact_bytes + layout.artifact_chunk_bytes - 1) / layout.artifact_chunk_bytes;
  if (num_chunks == 0) {
    return absl::FailedPreconditionError("artifact has no CPU chunks available for stable lease");
  }

  std::vector<uint32_t> ids;
  if (!lease.chunk_ids.empty()) {
    ids = lease.chunk_ids;
  } else {
    const uint32_t start = lease.chunk_start;
    uint32_t count = lease.chunk_count;
    if (start >= num_chunks) {
      return absl::OutOfRangeError(
          absl::StrFormat("chunk_start=%u is outside available chunks=%llu", start, static_cast<uint64_t>(num_chunks)));
    }
    if (count == 0) {
      count = static_cast<uint32_t>(num_chunks - start);
    }
    const uint64_t limit = std::min<uint64_t>(count, num_chunks - start);
    ids.reserve(limit);
    for (uint64_t i = 0; i < limit; ++i) {
      ids.push_back(static_cast<uint32_t>(start + i));
    }
  }

  if (ids.empty()) {
    ids.reserve(num_chunks);
    for (uint64_t i = 0; i < num_chunks; ++i) {
      ids.push_back(static_cast<uint32_t>(i));
    }
  }

  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  for (uint32_t id : ids) {
    if (id >= num_chunks) {
      return absl::OutOfRangeError(
          absl::StrFormat("chunk index %u exceeds available chunks=%llu", id, static_cast<uint64_t>(num_chunks)));
    }
  }
  return ids;
}

uint64_t compute_bytes_for_chunks(
    const replica::UnifiedMemoryAuthority::ArtifactLayout& layout,
    absl::Span<const uint32_t> chunk_ids) {
  if (layout.artifact_chunk_bytes == 0) {
    return 0;
  }
  uint64_t total = 0;
  for (uint32_t idx : chunk_ids) {
    const uint64_t start = static_cast<uint64_t>(idx) * layout.artifact_chunk_bytes;
    if (start >= layout.artifact_bytes) {
      continue;
    }
    const uint64_t end = std::min<uint64_t>(layout.artifact_bytes, start + layout.artifact_chunk_bytes);
    if (end > start) {
      total += end - start;
    }
  }
  return total;
}

} // namespace

// (hashing utilities moved to core/common/artifact_hash.*)
// GPU eviction helper kept internal to this translation unit.
// ═══════════════════════════════════════════════════════════════════════════
// Construction and Destruction
// ═══════════════════════════════════════════════════════════════════════════

// New unified constructor based on StoreEngineOptions (Phase-3+)
StoreEngine::StoreEngine(const StoreEngineOptions& opts)
    : options_(opts),
      storage_path_(opts.storage_path),
      memory_pool_size_(opts.memory_pool_size),
      artifact_chunk_bytes_(
          opts.artifact_chunk_bytes == 0 ? tensorcast::common::consts::kArtifactChunkDefault
                                         : opts.artifact_chunk_bytes),
      num_thread_(opts.num_thread),
      tx_slice_bytes_(opts.tx_slice_bytes),
      pinned_memory_timeout_(opts.pinned_memory_timeout),
      runtime_env_(std::make_unique<runtime::RuntimeEnv>(opts)) {
  LOG(INFO) << "Initializing StoreEngine with unified Options constructor";
  LOG(INFO) << "Storage path: "
            << (storage_path_.empty() ? "<empty - artifact_identifier will be full path>" : storage_path_.string());
  LOG(INFO) << "Memory pool size: " << memory_pool_size_ / communicator::misc::GB << "GB";
  LOG(INFO) << "I/O threads: " << num_thread_ << ", tx_slice_bytes: " << tx_slice_bytes_ / communicator::misc::MB
            << "MB";
  LOG(INFO) << "CPU shared memory enabled: " << (options_.cpu_shared_memory_enabled ? "true" : "false");

  auto init_status = runtime_env_->initialize();
  CHECK(init_status.ok()) << "Failed to initialize RuntimeEnv: " << init_status;

  auto& context = runtime_env_->runtime_context();
  runtime::ReplicaRuntime::Config replica_config{
      .runtime_context = &context,
  };
  replica_runtime_ = std::make_unique<runtime::ReplicaRuntime>(replica_config);
  runtime::ReplicaPromotionManager::Config promotion_config{
      .runtime_context = &context,
      .replica_runtime = replica_runtime_.get(),
  };
  promotion_manager_ = std::make_unique<runtime::ReplicaPromotionManager>(promotion_config);
  metadata_gateway_ = std::make_unique<runtime::metadata::MetadataGateway>(runtime::metadata::MetadataGateway::Config{
      .runtime_context = &context,
      .replica_runtime = replica_runtime_.get(),
      .promotion_manager = promotion_manager_.get(),
      .artifact_chunk_bytes = artifact_chunk_bytes_,
      .pinned_memory_timeout = pinned_memory_timeout_,
      .max_concurrency = options_.promotion.max_concurrency,
      .replica_factory = {},
  });
  runtime::IngestionRuntime::Config ingestion_config{
      .runtime_context = &context,
      .replica_runtime = replica_runtime_.get(),
      .metadata_gateway = metadata_gateway_.get(),
      .storage_path = storage_path_,
      .artifact_chunk_bytes = artifact_chunk_bytes_,
      .pinned_memory_timeout = pinned_memory_timeout_,
      .num_threads = num_thread_,
      .options = &options_,
  };
  ingestion_runtime_ = std::make_unique<runtime::IngestionRuntime>(std::move(ingestion_config));

  context.metrics_collector().update_all_metrics(
      *context.pinned_buffer_pool(), replica_runtime_->registry(), context.device_manager());
  if (context.options().pinned_memory_authority) {
    context.metrics_collector().update_pinned_authority_metrics(*context.options().pinned_memory_authority);
  }
}

StoreEngine::~StoreEngine() {
  LOG(INFO) << "Shutting down StoreEngine";
  if (replica_runtime_) {
    replica_runtime_->clear_mem();
  }
  if (runtime_env_) {
    runtime_env_->shutdown();
  }
}

void StoreEngine::set_global_store_client_for_testing(std::shared_ptr<components::IGlobalStoreClient> client) {
  if (runtime_env_) {
    runtime_env_->set_global_store_client_for_testing(client);
  }
  if (metadata_gateway_) {
    metadata_gateway_->set_client_override(std::move(client));
  }
}

void StoreEngine::set_promotion_sync_hooks(runtime::PromotionSyncHooks hooks) {
  if (promotion_manager_) {
    promotion_manager_->set_sync_hooks(std::move(hooks));
  }
}

void StoreEngine::set_stable_cache_spill_evictable(
    std::function<bool(const loading::ReplicaKey&, const components::StableDramCachePolicy&)> callback) {
  auto cache_manager = runtime_env_->runtime_context().stable_cache_manager();
  if (!cache_manager) {
    return;
  }
  cache_manager->set_spill_evictable_callback(std::move(callback));
}

absl::StatusOr<StoreEngine::StableCacheAdmissionResult> StoreEngine::admit_stable_cache_policy(
    const loading::ReplicaKey& key,
    const components::StableDramCachePolicy& policy) {
  if (key.device.type != DeviceType::CPU) {
    return absl::FailedPreconditionError("stable cache admission requires CPU ReplicaKey");
  }
  auto cache_manager = runtime_env_->runtime_context().stable_cache_manager();
  if (!cache_manager) {
    return absl::FailedPreconditionError("stable cache manager unavailable");
  }
  auto replica_or = runtime_env_->runtime_context().replica_registry().find(key);
  if (!replica_or.ok()) {
    return replica_or.status();
  }
  auto size_or = get_replica_size(key);
  if (!size_or.ok()) {
    return size_or.status();
  }
  components::StableDramCacheManager::AdmissionRequest request;
  request.key = key;
  request.replica = *replica_or;
  request.size_bytes = *size_or;
  request.policy = policy;
  auto admit_or = cache_manager->admit(request);
  if (!admit_or.ok()) {
    return admit_or.status();
  }
  StableCacheAdmissionResult result;
  result.admitted = admit_or->admitted;
  result.skipped = admit_or->skipped;
  return result;
}

absl::Status StoreEngine::update_stable_cache_policy(
    const loading::ReplicaKey& key,
    const components::StableDramCachePolicy& policy,
    std::optional<absl::Time> retention_deadline) {
  if (key.device.type != DeviceType::CPU) {
    return absl::FailedPreconditionError("stable cache policy update requires CPU ReplicaKey");
  }
  auto cache_manager = runtime_env_->runtime_context().stable_cache_manager();
  if (!cache_manager) {
    return absl::FailedPreconditionError("stable cache manager unavailable");
  }
  return cache_manager->update_policy(key, policy, retention_deadline);
}

// ═══════════════════════════════════════════════════════════════════════════
// Status Queries
// ═══════════════════════════════════════════════════════════════════════════

size_t StoreEngine::get_available_memory() const {
  return replica_runtime_->get_available_memory();
}

void StoreEngine::update_memory_pool_metrics() {
  replica_runtime_->update_memory_pool_metrics();
}

std::vector<StoreEngine::ReplicaInfo> StoreEngine::get_all_replicas_info() const {
  return replica_runtime_->get_all_replicas_info();
}

std::optional<MemoryTierBudget::Snapshot> StoreEngine::get_memory_tier_snapshot() const {
  auto budget = runtime_env_->runtime_context().memory_tier_budget();
  if (!budget) {
    return std::nullopt;
  }
  return budget->snapshot();
}

std::optional<MemoryTierConfig> StoreEngine::get_memory_tier_config() const {
  return runtime_env_->runtime_context().options().memory_tier_config;
}

absl::StatusOr<components::MemoryTierLeaseDescriptor> StoreEngine::acquire_memory_tier_lease(
    const components::MemoryTierLeaseDescriptor& lease) {
  if (lease.artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required for memory tier lease binding");
  }

  // Locate CPU replica for the artifact from the active runtime registry (UMA-backed).
  auto cpu_keys = replica_runtime_->registry().find_by_artifact(lease.artifact_id);
  auto runtime_cpu_keys = replica_runtime_->list_device_replicas(DeviceKey{DeviceType::CPU, -1, ""});
  loading::ReplicaKey cpu_key;
  auto select_cpu_key = [&](const std::vector<loading::ReplicaKey>& keys) -> bool {
    for (const auto& k : keys) {
      if (k.device.type == DeviceType::CPU) {
        cpu_key = k;
        return true;
      }
    }
    return false;
  };

  bool found_cpu = select_cpu_key(cpu_keys) || select_cpu_key(runtime_cpu_keys);
  if (!found_cpu) {
    return absl::NotFoundError(absl::StrFormat("CPU replica for artifact %s not found", lease.artifact_id));
  }

  auto replica_or = replica_runtime_->registry().find(cpu_key);
  if (!replica_or.ok()) {
    // Registry may be stale; retry with runtime list.
    if (select_cpu_key(runtime_cpu_keys)) {
      replica_or = replica_runtime_->registry().find(cpu_key);
    }
    if (!replica_or.ok()) {
      return replica_or.status();
    }
  }
  auto replica = *replica_or;
  auto& mm = replica->get_memory_manager();
  auto uma = mm.memory_authority();

  auto layout_or = uma->get_layout(cpu_key);
  if (!layout_or.ok()) {
    return layout_or.status();
  }
  auto chunk_ids_or = normalize_chunk_ids_for_lease(lease, *layout_or);
  if (!chunk_ids_or.ok()) {
    return chunk_ids_or.status();
  }
  auto chunk_ids = std::move(*chunk_ids_or);

  // Ensure CPU residency is ready before binding the lease.
  if (replica->get_memory_state(MemoryLocation::CPU) != MemoryState::LOADED) {
    auto load_status = std::move(replica->ensure_loaded_async(MemoryLocation::CPU)).get();
    if (!load_status.ok()) {
      return load_status;
    }
  }

  // Skip re-acquisition if the stable lease is already held for all chunks.
  const auto snapshot = uma->snapshot_cpu_chunks(cpu_key);
  bool already_held = true;
  for (uint32_t idx : chunk_ids) {
    if (idx >= snapshot.size() || snapshot[idx].stable_lease_count == 0) {
      already_held = false;
      break;
    }
  }

  components::MemoryTierLeaseDescriptor result = lease;
  result.chunk_ids = chunk_ids;

  if (already_held) {
    auto ledger_or = uma->get_ledger_version(cpu_key);
    if (!ledger_or.ok()) {
      return ledger_or.status();
    }
    result.ledger_version = *ledger_or;
    result.bytes = compute_bytes_for_chunks(*layout_or, result.chunk_ids);
    return result;
  }

  auto lease_or = acquire_replica_stable_lease(cpu_key, absl::MakeSpan(result.chunk_ids));
  if (!lease_or.ok()) {
    return lease_or.status();
  }

  result.chunk_ids = lease_or->chunk_indices;
  result.ledger_version = lease_or->ledger_version;
  result.bytes = lease_or->bytes;
  return result;
}

absl::StatusOr<components::MemoryTierLeaseDescriptor> StoreEngine::release_memory_tier_lease(
    const components::MemoryTierLeaseDescriptor& lease) {
  if (lease.artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required for memory tier lease release");
  }

  auto cpu_keys = runtime_env_->runtime_context().replica_registry().find_by_artifact(lease.artifact_id);
  auto runtime_cpu_keys = replica_runtime_->list_device_replicas(DeviceKey{DeviceType::CPU, -1, ""});
  loading::ReplicaKey cpu_key;
  auto select_cpu_key = [&](const std::vector<loading::ReplicaKey>& keys) -> bool {
    for (const auto& k : keys) {
      if (k.device.type == DeviceType::CPU) {
        cpu_key = k;
        return true;
      }
    }
    return false;
  };

  bool found_cpu = select_cpu_key(cpu_keys) || select_cpu_key(runtime_cpu_keys);
  if (!found_cpu) {
    return absl::NotFoundError(absl::StrFormat("CPU replica for artifact %s not found", lease.artifact_id));
  }

  auto replica_or = replica_runtime_->registry().find(cpu_key);
  if (!replica_or.ok()) {
    if (select_cpu_key(runtime_cpu_keys)) {
      replica_or = replica_runtime_->registry().find(cpu_key);
    }
    if (!replica_or.ok()) {
      return replica_or.status();
    }
  }
  auto replica = *replica_or;
  auto& mm = replica->get_memory_manager();
  auto uma = mm.memory_authority();

  auto layout_or = uma->get_layout(cpu_key);
  if (!layout_or.ok()) {
    return layout_or.status();
  }
  auto chunk_ids_or = normalize_chunk_ids_for_lease(lease, *layout_or);
  if (!chunk_ids_or.ok()) {
    return chunk_ids_or.status();
  }
  auto chunk_ids = std::move(*chunk_ids_or);

  const auto snapshot = uma->snapshot_cpu_chunks(cpu_key);
  bool held = false;
  for (uint32_t idx : chunk_ids) {
    if (idx < snapshot.size() && snapshot[idx].stable_lease_count > 0) {
      held = true;
      break;
    }
  }

  if (held) {
    auto st = uma->release_stable_lease(cpu_key, absl::MakeSpan(chunk_ids));
    if (!st.ok()) {
      return st;
    }
  }

  auto ledger_or = uma->get_ledger_version(cpu_key);
  components::MemoryTierLeaseDescriptor result = lease;
  result.chunk_ids = std::move(chunk_ids);
  result.bytes = compute_bytes_for_chunks(*layout_or, result.chunk_ids);
  result.ledger_version = ledger_or.ok() ? *ledger_or : 0;
  return result;
}

absl::StatusOr<int> StoreEngine::get_unique_gpu_residency(
    std::string_view artifact_id,
    std::optional<std::string_view> view_id) const {
  return replica_runtime_->get_unique_gpu_residency(artifact_id, view_id);
}

absl::StatusOr<loading::ReplicaHandle> StoreEngine::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return ingestion_runtime_->ingest_from_p2p(artifact_identifier, source, target, hints);
}

absl::StatusOr<loading::ReplicaHandle> StoreEngine::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return ingestion_runtime_->ingest_from_disk(artifact_identifier, source, target, hints);
}

absl::StatusOr<loading::ReplicaHandle> StoreEngine::ingest_from_buffer_internal(
    const std::string& artifact_identifier,
    const loading::InlineBufferSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  (void)artifact_identifier;
  (void)source;
  (void)target;
  (void)hints;
  return absl::UnimplementedError("InlineBufferSource loading not yet implemented");
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------
std::vector<DeviceKey> StoreEngine::get_resident_devices(
    std::string_view artifact_id,
    std::optional<std::string_view> view_id) const {
  return replica_runtime_->get_resident_devices(artifact_id, view_id);
}

std::vector<StoreEngine::ReplicaInventoryEntry> StoreEngine::get_ha_inventory() const {
  return replica_runtime_->get_ha_inventory();
}

std::optional<std::string> StoreEngine::get_replica_global_store_id(const loading::ReplicaKey& key) const {
  return replica_runtime_->get_replica_global_id(key);
}

void StoreEngine::set_replica_global_store_id(const loading::ReplicaKey& key, std::string replica_id) {
  replica_runtime_->set_replica_global_id(key, std::move(replica_id));
}

std::vector<ReplicaKey> StoreEngine::list_device_replicas(const DeviceKey& device) const {
  return replica_runtime_->list_device_replicas(device);
}

// ---------------------------------------------------------------------------
// Multi-Device Binding – GPU-aware memory eviction (NEW in Phase 3.2)
// ---------------------------------------------------------------------------

int StoreEngine::wait_replica_ready(const ReplicaKey& key) {
  return replica_runtime_->wait_replica_ready(key);
}

absl::Status StoreEngine::unload_replica_status(const ReplicaKey& key) {
  return replica_runtime_->unload_replica_status(key);
}

int StoreEngine::unload_replica(const ReplicaKey& key) {
  return replica_runtime_->unload_replica(key);
}

absl::Status StoreEngine::retire_replica_status(const ReplicaKey& key) {
  return replica_runtime_->retire_replica_status(key);
}

MemoryState StoreEngine::get_replica_state(const ReplicaKey& key, DeviceType memory_type) const {
  return replica_runtime_->get_replica_state(key, memory_type);
}

absl::StatusOr<uint64_t> StoreEngine::get_replica_gpu_ptr(const ReplicaKey& key) {
  return replica_runtime_->get_replica_gpu_ptr(key);
}

absl::StatusOr<uint64_t> StoreEngine::get_replica_size(const ReplicaKey& key) {
  return replica_runtime_->get_replica_size(key);
}

void StoreEngine::set_replica_publish_state(const ReplicaKey& key, ReplicaPublishState state) {
  replica_runtime_->set_replica_publish_state(key, state);
}

StoreEngine::ReplicaPublishState StoreEngine::get_replica_publish_state(const ReplicaKey& key) const {
  return replica_runtime_->get_replica_publish_state(key);
}

std::unique_ptr<runtime::RuntimeContextEvents::Subscription> StoreEngine::subscribe_to_runtime_events(
    runtime::RuntimeContextEvents::Callback callback) {
  return runtime_env_->runtime_context().subscribe_to_events(std::move(callback));
}

absl::StatusOr<ExportRegistration> StoreEngine::enable_remote_replica_access(
    const ReplicaKey& key,
    MemoryLocation location) {
  return replica_runtime_->enable_remote_replica_access(key, location);
}

absl::Status StoreEngine::disable_remote_replica_access(const ReplicaKey& key, MemoryLocation location) {
  return replica_runtime_->disable_remote_replica_access(key, location);
}

absl::StatusOr<replica::UnifiedMemoryAuthority::ExportRegistration> StoreEngine::set_replica_exported(
    const ReplicaKey& key,
    MemoryLocation location,
    absl::Span<const uint32_t> chunks,
    bool on) {
  return replica_runtime_->set_replica_exported(key, location, chunks, on);
}

absl::StatusOr<replica::UnifiedMemoryAuthority::StableLease> StoreEngine::acquire_replica_stable_lease(
    const ReplicaKey& key,
    absl::Span<const uint32_t> chunks) {
  return replica_runtime_->acquire_replica_stable_lease(key, chunks);
}

absl::Status StoreEngine::release_replica_stable_lease(const replica::UnifiedMemoryAuthority::StableLease& lease) {
  return replica_runtime_->release_replica_stable_lease(lease);
}

// ═══════════════════════════════════════════════════════════════════════════
// Memory cleanup
// ═══════════════════════════════════════════════════════════════════════════

int StoreEngine::clear_mem() {
  return replica_runtime_->clear_mem();
}

absl::StatusOr<loading::ReplicaHandle> StoreEngine::materialize_replica(
    const DeviceKey& target_device,
    MaterializeMode mode,
    const loading::MaterializeHints& hints,
    std::optional<loading::DiskSource> disk_source) {
  return ingestion_runtime_->materialize_replica(target_device, mode, hints, std::move(disk_source));
}

absl::StatusOr<loading::MaterializeIntoTargetResult> StoreEngine::materialize_into_target(
    const DeviceKey& target_device,
    const loading::IntoTargetLayout& target_layout,
    std::string_view canonical_index_json,
    uint64_t generation,
    const loading::MaterializeHints& hints,
    std::optional<loading::DiskSource> disk_source) {
  return ingestion_runtime_->materialize_into_target(
      target_device, target_layout, canonical_index_json, generation, hints, std::move(disk_source));
}

absl::StatusOr<loading::MaterializeIntoTargetResult> StoreEngine::materialize_mapped_into_target(
    const DeviceKey& target_device,
    const loading::IntoTargetLayout& target_layout,
    const loader::ByteRangeMap& mapping,
    std::string_view canonical_index_json,
    uint64_t generation,
    const loading::MaterializeHints& hints,
    std::optional<loading::DiskSource> disk_source) {
  return ingestion_runtime_->materialize_mapped_into_target(
      target_device, target_layout, mapping, canonical_index_json, generation, hints, std::move(disk_source));
}

absl::StatusOr<loading::MaterializeIntoTargetResult> StoreEngine::materialize_mapped_into_target(
    const DeviceKey& target_device,
    const loading::IntoTargetLayout& target_layout,
    const loader::ByteRangeMap& mapping,
    std::string_view canonical_index_json,
    uint64_t generation,
    const loading::MaterializeHints& hints) {
  return ingestion_runtime_->materialize_mapped_into_target(
      target_device, target_layout, mapping, canonical_index_json, generation, hints);
}

absl::StatusOr<loading::ReplicaHandle> StoreEngine::materialize_view_from_assembly(
    std::string_view assembly_id,
    std::string_view target_artifact_id,
    std::string_view view_id,
    std::string_view view_spec_json,
    const DeviceKey& target_device,
    loading::TransformPlacement placement,
    const std::vector<std::string>* allowed_view_ids) {
  return ingestion_runtime_->materialize_view_from_assembly(
      assembly_id, target_artifact_id, view_id, view_spec_json, target_device, placement, allowed_view_ids);
}

absl::StatusOr<SealAssemblyResult> StoreEngine::seal_assembly(
    std::string_view assembly_id,
    bool publish_canonical,
    runtime::ingestion::MaterializationFacade::SealProgressCallback progress_cb,
    const std::vector<std::string>* allowed_view_ids) {
  return ingestion_runtime_->seal_assembly(assembly_id, publish_canonical, std::move(progress_cb), allowed_view_ids);
}

// ═══════════════════════════════════════════════════════════════════════════
// Global Store registration helper for already-loaded replicas
// ═══════════════════════════════════════════════════════════════════════════
// 注册本地到global store
absl::Status StoreEngine::register_replica_with_global_store(
    const ReplicaKey& key,
    std::string_view artifact_id_override) {
  return ingestion_runtime_->register_replica_with_global_store(key, artifact_id_override);
}

absl::Status StoreEngine::unregister_replica_from_global_store(std::string_view artifact_id, int device_id) {
  return metadata_gateway_->unregister_replica(artifact_id, device_id);
}

// ═══════════════════════════════════════════════════════════════════════════
// Key-mapping wrappers
// ═══════════════════════════════════════════════════════════════════════════

absl::StatusOr<components::KeyMapping> StoreEngine::resolve_key_mapping(
    std::string_view key,
    const components::RpcOptions& rpc_options) {
  return metadata_gateway_->resolve_key_mapping(key, rpc_options);
}

absl::StatusOr<components::KeyMappingSwapResult> StoreEngine::swap_key_mapping(
    std::string_view key,
    std::string_view new_artifact_id,
    std::optional<std::string_view> expected_artifact_id,
    std::optional<uint64_t> expected_generation) {
  return metadata_gateway_->swap_key_mapping(key, new_artifact_id, expected_artifact_id, expected_generation);
}

absl::StatusOr<std::string> StoreEngine::get_canonical_index_by_id(std::string_view artifact_id) {
  auto local_or = [&]() -> absl::StatusOr<std::string> {
    auto keys = replica_runtime_->registry().find_by_artifact(artifact_id);
    if (keys.empty()) {
      return absl::NotFoundError("local canonical index not found");
    }
    for (const auto& key : keys) {
      auto replica_or = replica_runtime_->registry().find(key);
      if (!replica_or.ok()) {
        continue;
      }
      const auto& canonical = (*replica_or)->canonical_index_json();
      if (canonical.has_value() && !canonical->empty()) {
        return *canonical;
      }
    }
    return absl::NotFoundError("local canonical index not found");
  }();

  if (local_or.ok()) {
    return *local_or;
  }
  auto remote_or = metadata_gateway_->get_canonical_index(artifact_id);
  if (remote_or.ok()) {
    return *remote_or;
  }
  if (absl::IsFailedPrecondition(remote_or.status())) {
    return local_or.status();
  }
  return remote_or.status();
}

absl::StatusOr<components::ViewMetadata> StoreEngine::get_view_metadata(
    std::string_view artifact_id,
    std::string_view view_id) {
  return metadata_gateway_->get_view_metadata(artifact_id, view_id);
}

absl::Status StoreEngine::upsert_key_mapping(std::string_view key, std::string_view artifact_id, absl::Duration ttl) {
  return metadata_gateway_->upsert_key_mapping(key, artifact_id, ttl);
}

absl::Status StoreEngine::revoke_key_mapping(std::string_view key) {
  return metadata_gateway_->revoke_key_mapping(key);
}

// ═══════════════════════════════════════════════════════════════════════════
// RFC-0006 – Memory Artifact Registration (coalesced)
// ═══════════════════════════════════════════════════════════════════════════

absl::StatusOr<StoreEngine::RegistrationBeginResult> StoreEngine::begin_register_artifact(
    const ArtifactRegistration& reg) {
  return metadata_gateway_->begin_registration(reg);
}

absl::StatusOr<uint64_t> StoreEngine::get_registration_gpu_ptr(std::string_view registration_id) const {
  return metadata_gateway_->get_registration_gpu_ptr(registration_id);
}

absl::StatusOr<StoreEngine::RegistrationCommitResult> StoreEngine::commit_registered_artifact(
    std::string_view registration_id) {
  return metadata_gateway_->commit_registration(registration_id);
}

absl::Status StoreEngine::ingest_view_registration_chunk(
    std::string_view registration_id,
    uint64_t view_offset,
    absl::Span<const std::byte> data) {
  return metadata_gateway_->ingest_view_chunk(registration_id, view_offset, data);
}

absl::Status StoreEngine::ingest_registration_chunk(
    std::string_view registration_id,
    uint64_t offset,
    absl::Span<const std::byte> data) {
  return metadata_gateway_->ingest_registration_chunk(registration_id, offset, data);
}

absl::StatusOr<uint64_t> StoreEngine::get_view_registration_ingested_bytes(std::string_view registration_id) {
  return metadata_gateway_->get_view_ingested_bytes(registration_id);
}

absl::Status StoreEngine::keep_alive_registered_artifact(std::string_view registration_id, uint32_t ttl_ms) {
  return metadata_gateway_->keep_alive_registration(registration_id, ttl_ms);
}

absl::Status StoreEngine::abort_registered_artifact(std::string_view registration_id) {
  return metadata_gateway_->abort_registration(registration_id);
}

std::vector<replica::ChunkState> StoreEngine::get_chunk_states_telemetry(std::string_view artifact_id) const {
  return replica_runtime_->get_chunk_states_telemetry(artifact_id);
}

std::vector<replica::ChunkState> StoreEngine::get_chunk_states_for_device(std::string_view artifact_id, int device_id)
    const {
  return replica_runtime_->get_chunk_states_for_device(artifact_id, device_id);
}

// GPU device queries (exposed for status/health reporting)
absl::StatusOr<size_t> StoreEngine::get_device_total_memory(int device_id) const {
  return replica_runtime_->get_device_total_memory(device_id);
}

absl::StatusOr<size_t> StoreEngine::get_device_free_memory(int device_id) const {
  return replica_runtime_->get_device_free_memory(device_id);
}

std::vector<replica::ChunkState> StoreEngine::get_chunk_states_cpu_uma(std::string_view artifact_id) const {
  return replica_runtime_->get_chunk_states_cpu_uma(artifact_id);
}

absl::StatusOr<void*> StoreEngine::get_replica_cpu_base_ptr(std::string_view artifact_id) const {
  auto cpu_keys = replica_runtime_->registry().find_by_artifact(artifact_id);
  auto runtime_cpu_keys = replica_runtime_->list_device_replicas(DeviceKey{DeviceType::CPU, -1, ""});
  loading::ReplicaKey cpu_key;
  auto select_cpu_key = [&](const std::vector<loading::ReplicaKey>& keys) -> bool {
    for (const auto& k : keys) {
      if (k.device.type == DeviceType::CPU) {
        cpu_key = k;
        return true;
      }
    }
    return false;
  };

  bool found_cpu = select_cpu_key(cpu_keys) || select_cpu_key(runtime_cpu_keys);
  if (!found_cpu) {
    return absl::NotFoundError(absl::StrFormat("CPU replica for artifact %s not found", artifact_id));
  }

  auto replica_or = replica_runtime_->registry().find(cpu_key);
  if (!replica_or.ok()) {
    if (select_cpu_key(runtime_cpu_keys)) {
      replica_or = replica_runtime_->registry().find(cpu_key);
    }
    if (!replica_or.ok()) {
      return replica_or.status();
    }
  }
  auto replica = *replica_or;
  if (replica->get_memory_state(common::memory::MemoryLocation::CPU) != replica::MemoryState::LOADED) {
    return absl::FailedPreconditionError("CPU replica is not loaded");
  }
  auto ptrs = replica->get_data_pointer(common::memory::MemoryLocation::CPU);
  if (ptrs.empty() || ptrs.front() == nullptr) {
    return absl::FailedPreconditionError("CPU replica base pointer unavailable");
  }
  return ptrs.front();
}

gsl::not_null<std::shared_ptr<components::CommunicationManager>> StoreEngine::get_shared_comm_manager() const {
  auto comm_mgr = runtime_env_->runtime_context().communication_manager();
  CHECK(comm_mgr != nullptr) << "RuntimeContext returned null CommunicationManager";
  return gsl::not_null<std::shared_ptr<components::CommunicationManager>>{comm_mgr};
}

components::MetricsCollector::P2PTransferSnapshot StoreEngine::get_p2p_transfer_snapshot() const {
  return replica_runtime_->metrics().get_p2p_transfer_snapshot();
}

absl::StatusOr<loader::ViewPlan> StoreEngine::compute_view_plan(
    std::string_view canonical_index_json,
    const loader::ViewSpec& spec) {
  return loader::ViewPlanner::compute_view_plan(canonical_index_json, spec);
}

absl::StatusOr<loader::ViewPlan> StoreEngine::compute_view_plan(
    std::string_view canonical_index_json,
    const loader::ViewSpec& spec,
    absl::Span<const std::string> subset_names) {
  return loader::ViewPlanner::compute_view_plan(canonical_index_json, spec, subset_names);
}

bool StoreEngine::view_plan_allows_alias(const loader::ViewPlan& plan) {
  if (plan.selection.map.total_bytes == 0) {
    return false;
  }
  if (plan.selection.requires_materialization) {
    return false;
  }
  if (plan.transform.requires_materialization || !plan.transform.tensors.empty()) {
    return false;
  }
  return plan.selection.is_contiguous && plan.selection.is_segment_aligned;
}

absl::StatusOr<std::string> StoreEngine::compute_view_data_hash_from_source(
    loader::SeekableSource& base_source,
    const loader::ViewPlan& plan,
    size_t leaf_chunk_bytes) {
  ViewHashComputer computer;
  return computer.hash_view_from_source(base_source, plan, leaf_chunk_bytes);
}

} // namespace tensorcast::store
