// Copyright (c) 2025, TensorCast Team.

#include "store_engine.h"

#include <unistd.h>

#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <utility>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/communicator/misc/common.h"
#include "core/store/materialization/common/view_hash_utils.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/replica/memory_state.h"
#include "gsl/pointers"

namespace tensorcast::store {

using common::memory::MemoryLocation;
using loading::ReplicaHandle;
using loading::ReplicaKey;
using replica::MemoryState;

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

  auto init_status = runtime_env_->initialize();
  CHECK(init_status.ok()) << "Failed to initialize RuntimeEnv: " << init_status;

  auto& context = runtime_env_->runtime_context();
  runtime::ReplicaRuntime::Config replica_config{
      .runtime_context = &context,
  };
  replica_runtime_ = std::make_unique<runtime::ReplicaRuntime>(replica_config);
  metadata_gateway_ = std::make_unique<runtime::metadata::MetadataGateway>(runtime::metadata::MetadataGateway::Config{
      .runtime_context = &context,
      .replica_runtime = replica_runtime_.get(),
      .artifact_chunk_bytes = artifact_chunk_bytes_,
      .pinned_memory_timeout = pinned_memory_timeout_,
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

absl::StatusOr<int> StoreEngine::get_unique_gpu_residency(std::string_view artifact_id) const {
  return replica_runtime_->get_unique_gpu_residency(artifact_id);
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
std::vector<DeviceKey> StoreEngine::get_resident_devices(std::string_view artifact_id) const {
  return replica_runtime_->get_resident_devices(artifact_id);
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

int StoreEngine::unload_replica(const ReplicaKey& key) {
  return replica_runtime_->unload_replica(key);
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

absl::StatusOr<ExportRegistration> StoreEngine::enable_remote_replica_access(
    const ReplicaKey& key,
    MemoryLocation location) {
  return replica_runtime_->enable_remote_replica_access(key, location);
}

absl::Status StoreEngine::disable_remote_replica_access(const ReplicaKey& key, MemoryLocation location) {
  return replica_runtime_->disable_remote_replica_access(key, location);
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
    const loading::MaterializeHints& hints) {
  return ingestion_runtime_->materialize_replica(target_device, mode, hints);
}

// ═══════════════════════════════════════════════════════════════════════════
// Global Store registration helper for already-loaded replicas
// ═══════════════════════════════════════════════════════════════════════════

absl::Status StoreEngine::register_replica_with_global_store(
    const ReplicaKey& key,
    std::string_view artifact_id_override) {
  return ingestion_runtime_->register_replica_with_global_store(key, artifact_id_override);
}

absl::Status StoreEngine::unregister_replica_from_global_store(std::string_view artifact_id, int device_id) {
  return metadata_gateway_->unregister_replica(artifact_id, device_id);
}

// ═══════════════════════════════════════════════════════════════════════════
// RFC-0014: Key-mapping wrappers
// ═══════════════════════════════════════════════════════════════════════════

absl::StatusOr<components::KeyMapping> StoreEngine::resolve_key_mapping(std::string_view key) {
  return metadata_gateway_->resolve_key_mapping(key);
}

absl::StatusOr<std::string> StoreEngine::get_canonical_index_by_id(std::string_view artifact_id) {
  return metadata_gateway_->get_canonical_index(artifact_id);
}

absl::Status StoreEngine::upsert_key_mapping(
    std::string_view key,
    std::string_view artifact_id,
    std::string_view disk_path,
    absl::Duration ttl) {
  return metadata_gateway_->upsert_key_mapping(key, artifact_id, disk_path, ttl);
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

gsl::not_null<std::shared_ptr<components::CommunicationManager>> StoreEngine::get_shared_comm_manager() const {
  auto comm_mgr = runtime_env_->runtime_context().communication_manager();
  CHECK(comm_mgr != nullptr) << "RuntimeContext returned null CommunicationManager";
  return gsl::not_null<std::shared_ptr<components::CommunicationManager>>{comm_mgr};
}

absl::StatusOr<loader::ViewPlan> StoreEngine::compute_view_plan(
    std::string_view canonical_index_json,
    const loader::ViewSpec& spec) {
  return loader::ViewPlanner::compute_view_plan(canonical_index_json, spec);
}

bool StoreEngine::view_plan_allows_alias(const loader::ViewPlan& plan) {
  if (plan.selection.total_bytes == 0) {
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
