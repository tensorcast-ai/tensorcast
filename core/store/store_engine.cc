// Copyright (c) 2025, TensorCast Team.

#include "store_engine.h"

#include <unistd.h>

#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/communicator/misc/common.h"
#include "core/store/components/eviction_service.h"
#include "core/store/materialization/common/view_hash_utils.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/materialization/runtime/pipeline/ingestion_pipeline.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/replica_config.h"
#include "gsl/pointers"

namespace tensorcast::store {

using common::memory::MemoryLocation;
using loading::ReplicaHandle;
using loading::ReplicaKey;
using replica::MemoryState;
using replica::Replica;

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
      component_catalog_(std::make_unique<components::runtime::ComponentCatalog>(opts)) {
  LOG(INFO) << "Initializing StoreEngine with unified Options constructor";
  LOG(INFO) << "Storage path: "
            << (storage_path_.empty() ? "<empty - artifact_identifier will be full path>" : storage_path_.string());
  LOG(INFO) << "Memory pool size: " << memory_pool_size_ / communicator::misc::GB << "GB";
  LOG(INFO) << "I/O threads: " << num_thread_ << ", tx_slice_bytes: " << tx_slice_bytes_ / communicator::misc::MB
            << "MB";

  auto catalog_status = component_catalog_->start();
  CHECK(catalog_status.ok()) << "Failed to initialize ComponentCatalog: " << catalog_status;
  replica_service_ = std::make_unique<components::runtime::ReplicaService>(component_catalog_.get());
  global_store_publisher_ =
      std::make_unique<components::runtime::GlobalStorePublisher>(components::runtime::GlobalStorePublisher::Config{
          .component_catalog = component_catalog_.get(), .replica_service = replica_service_.get()});
  telemetry_service_ =
      std::make_unique<components::runtime::TelemetryService>(components::runtime::TelemetryService::Config{
          .component_catalog = component_catalog_.get(), .replica_service = replica_service_.get()});
  materialization::runtime::pipeline::IngestionPipeline::Config pipeline_config{
      .storage_path = storage_path_,
      .num_threads = num_thread_,
      .artifact_chunk_bytes = artifact_chunk_bytes_,
      .pinned_memory_timeout = pinned_memory_timeout_,
      .engine_options = &options_,
      .replica_service = replica_service_.get(),
      .component_catalog = component_catalog_.get(),
      .telemetry_service = telemetry_service_.get(),
      .global_store_publisher = global_store_publisher_.get(),
  };
  ingestion_pipeline_ =
      std::make_unique<materialization::runtime::pipeline::IngestionPipeline>(std::move(pipeline_config));
  materialization::control::MaterializationCoordinator::Config coordinator_config{
      .replica_service = replica_service_.get(),
      .pipeline = ingestion_pipeline_.get(),
      .component_catalog = component_catalog_.get(),
      .global_store_publisher = global_store_publisher_.get(),
      .artifact_chunk_bytes = artifact_chunk_bytes_,
      .pinned_memory_timeout = pinned_memory_timeout_,
      .num_threads = num_thread_,
  };
  materialization_coordinator_ =
      std::make_unique<materialization::control::MaterializationCoordinator>(std::move(coordinator_config));

  {
    auto* device_manager_ptr = &component_catalog_->device_manager();
    auto* replica_registry_ptr = &replica_service_->registry();
    auto* metrics_collector_ptr = &component_catalog_->metrics_collector();
    components::RegistrationResources registration_resources{
        .device_manager = gsl::not_null<components::DeviceManager*>{device_manager_ptr},
        .replica_registry = gsl::not_null<components::ReplicaRegistry*>{replica_registry_ptr},
        .metrics_collector = gsl::not_null<components::MetricsCollector*>{metrics_collector_ptr},
        .memory_pool =
            gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{component_catalog_->pinned_buffer_pool()},
        .communication_manager = component_catalog_->communication_manager()};
    components::ReplicaFactory replica_factory =
        [](const replica::ReplicaConfig& config) -> absl::StatusOr<std::shared_ptr<replica::Replica>> {
      auto create_or = Replica::create(config);
      if (!create_or.ok()) {
        return create_or.status();
      }
      return std::shared_ptr<replica::Replica>(std::move(create_or.value()));
    };
    registration_facade_ = std::make_unique<components::RegistrationFacade>(
        std::move(registration_resources),
        std::move(replica_factory),
        artifact_chunk_bytes_,
        pinned_memory_timeout_,
        global_store_publisher_.get());
  }

  component_catalog_->metrics_collector().update_all_metrics(
      *component_catalog_->pinned_buffer_pool(), replica_service_->registry(), component_catalog_->device_manager());
}

StoreEngine::~StoreEngine() {
  LOG(INFO) << "Shutting down StoreEngine";
  if (replica_service_) {
    replica_service_->clear_mem();
  }
  if (component_catalog_) {
    component_catalog_->shutdown();
  }
}

void StoreEngine::set_global_store_client_for_testing(std::shared_ptr<components::IGlobalStoreClient> client) {
  component_catalog_->set_global_store_client_for_testing(client);
  if (global_store_publisher_) {
    global_store_publisher_->set_client_override(std::move(client));
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Status Queries
// ═══════════════════════════════════════════════════════════════════════════

size_t StoreEngine::get_available_memory() const {
  return telemetry_service_->get_available_memory();
}

void StoreEngine::update_memory_pool_metrics() {
  telemetry_service_->update_memory_pool_metrics();
}

std::vector<StoreEngine::ReplicaInfo> StoreEngine::get_all_replicas_info() const {
  return telemetry_service_->get_all_replicas_info();
}

absl::StatusOr<int> StoreEngine::get_unique_gpu_residency(std::string_view artifact_id) const {
  return telemetry_service_->get_unique_gpu_residency(artifact_id);
}

absl::StatusOr<loading::ReplicaHandle> StoreEngine::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return ingestion_pipeline_->ingest_from_p2p(artifact_identifier, source, target, hints);
}

absl::StatusOr<loading::ReplicaHandle> StoreEngine::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return ingestion_pipeline_->ingest_from_disk(artifact_identifier, source, target, hints);
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
  return telemetry_service_->get_resident_devices(artifact_id);
}

std::vector<ReplicaKey> StoreEngine::list_device_replicas(const DeviceKey& device) const {
  return telemetry_service_->list_device_replicas(device);
}

// ---------------------------------------------------------------------------
// Multi-Device Binding – GPU-aware memory eviction (NEW in Phase 3.2)
// ---------------------------------------------------------------------------

int StoreEngine::wait_replica_ready(const ReplicaKey& key) {
  return replica_service_->wait_replica_ready(key);
}

int StoreEngine::unload_replica(const ReplicaKey& key) {
  return replica_service_->unload_replica(key);
}

MemoryState StoreEngine::get_replica_state(const ReplicaKey& key, DeviceType memory_type) const {
  return replica_service_->get_replica_state(key, memory_type);
}

absl::StatusOr<uint64_t> StoreEngine::get_replica_gpu_ptr(const ReplicaKey& key) {
  return replica_service_->get_replica_gpu_ptr(key);
}

absl::StatusOr<uint64_t> StoreEngine::get_replica_size(const ReplicaKey& key) {
  return replica_service_->get_replica_size(key);
}

absl::StatusOr<ExportRegistration> StoreEngine::enable_remote_replica_access(
    const ReplicaKey& key,
    MemoryLocation location) {
  return replica_service_->enable_remote_replica_access(key, location);
}

absl::Status StoreEngine::disable_remote_replica_access(const ReplicaKey& key, MemoryLocation location) {
  return replica_service_->disable_remote_replica_access(key, location);
}

// ═══════════════════════════════════════════════════════════════════════════
// Memory cleanup
// ═══════════════════════════════════════════════════════════════════════════

int StoreEngine::clear_mem() {
  return replica_service_->clear_mem();
}

absl::StatusOr<loading::ReplicaHandle> StoreEngine::materialize_replica(
    const DeviceKey& target_device,
    MaterializeMode mode,
    const loading::MaterializeHints& hints) {
  return materialization_coordinator_->Materialize(target_device, mode, hints);
}

// ═══════════════════════════════════════════════════════════════════════════
// Global Store registration helper for already-loaded replicas
// ═══════════════════════════════════════════════════════════════════════════
// 注册本地到global store
absl::Status StoreEngine::register_replica_with_global_store(
    const ReplicaKey& key,
    std::string_view artifact_id_override) {
  return materialization_coordinator_->register_replica_with_global_store(key, artifact_id_override);
}

absl::Status StoreEngine::unregister_replica_from_global_store(std::string_view artifact_id, int device_id) {
  return global_store_publisher_->unregister_replica(artifact_id, device_id);
}

// ═══════════════════════════════════════════════════════════════════════════
// RFC-0014: Key-mapping wrappers
// ═══════════════════════════════════════════════════════════════════════════

absl::StatusOr<components::KeyMapping> StoreEngine::resolve_key_mapping(std::string_view key) {
  return global_store_publisher_->resolve_key_mapping(key);
}

absl::StatusOr<std::string> StoreEngine::get_canonical_index_by_id(std::string_view artifact_id) {
  return global_store_publisher_->get_canonical_index(artifact_id);
}

absl::Status StoreEngine::upsert_key_mapping(
    std::string_view key,
    std::string_view artifact_id,
    std::string_view disk_path,
    absl::Duration ttl) {
  return global_store_publisher_->upsert_key_mapping(key, artifact_id, disk_path, ttl);
}

absl::Status StoreEngine::revoke_key_mapping(std::string_view key) {
  return global_store_publisher_->revoke_key_mapping(key);
}

// ═══════════════════════════════════════════════════════════════════════════
// RFC-0006 – Memory Artifact Registration (coalesced)
// ═══════════════════════════════════════════════════════════════════════════

absl::StatusOr<StoreEngine::RegistrationBeginResult> StoreEngine::begin_register_artifact(
    const ArtifactRegistration& reg) {
  if (!registration_facade_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_facade_->begin(reg);
}

absl::StatusOr<StoreEngine::RegistrationCommitResult> StoreEngine::commit_registered_artifact(
    std::string_view registration_id) {
  if (!registration_facade_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_facade_->commit(registration_id);
}

absl::Status StoreEngine::ingest_view_registration_chunk(
    std::string_view registration_id,
    uint64_t view_offset,
    absl::Span<const std::byte> data) {
  if (!registration_facade_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_facade_->ingest_view_chunk(registration_id, view_offset, data);
}

absl::StatusOr<uint64_t> StoreEngine::get_view_registration_ingested_bytes(std::string_view registration_id) {
  if (!registration_facade_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_facade_->get_view_ingested_bytes(registration_id);
}

absl::Status StoreEngine::keep_alive_registered_artifact(std::string_view registration_id, uint32_t ttl_ms) {
  if (!registration_facade_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_facade_->keep_alive(registration_id, ttl_ms);
}

absl::Status StoreEngine::abort_registered_artifact(std::string_view registration_id) {
  if (!registration_facade_) {
    return absl::FailedPreconditionError("registration manager is not initialized");
  }
  return registration_facade_->abort(registration_id);
}

std::vector<replica::ChunkState> StoreEngine::get_chunk_states_telemetry(std::string_view artifact_id) const {
  return telemetry_service_->get_chunk_states_telemetry(artifact_id);
}

std::vector<replica::ChunkState> StoreEngine::get_chunk_states_for_device(std::string_view artifact_id, int device_id)
    const {
  return telemetry_service_->get_chunk_states_for_device(artifact_id, device_id);
}

// GPU device queries (exposed for status/health reporting)
absl::StatusOr<size_t> StoreEngine::get_device_total_memory(int device_id) const {
  return telemetry_service_->get_device_total_memory(device_id);
}

absl::StatusOr<size_t> StoreEngine::get_device_free_memory(int device_id) const {
  return telemetry_service_->get_device_free_memory(device_id);
}

std::vector<replica::ChunkState> StoreEngine::get_chunk_states_cpu_uma(std::string_view artifact_id) const {
  return telemetry_service_->get_chunk_states_cpu_uma(artifact_id);
}

gsl::not_null<std::shared_ptr<components::CommunicationManager>> StoreEngine::get_shared_comm_manager() const {
  auto comm_mgr = component_catalog_->communication_manager();
  CHECK(comm_mgr != nullptr) << "StoreEngine ComponentCatalog returned null CommunicationManager";
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
