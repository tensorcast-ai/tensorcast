// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <string_view>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/stable_dram_cache_policy.h"
#include "core/store/components/worker_identity.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/contracts/loader.h"
#include "core/store/memory_tier_budget.h"
#include "core/store/memory_tier_config.h"
#include "core/store/replica/chunk_state.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/unified_memory_authority.h"
#include "core/store/runtime/context/runtime_context_events.h"
#include "core/store/runtime/ingestion/artifact_lowering_plan.h"
#include "core/store/runtime/ingestion/ingestion_runtime.h"
#include "core/store/runtime/metadata/metadata_gateway.h"
#include "core/store/runtime/metadata/metadata_types.h"
#include "core/store/runtime/replica/replica_info.h"
#include "core/store/runtime/replica/replica_promotion_manager.h"
#include "core/store/runtime/replica/replica_runtime.h"
#include "core/store/runtime/runtime_env.h"
#include "core/store/seal_assembly_result.h"
#include "core/store/store_engine_options.h"
#include "gsl/pointers"

namespace tensorcast::store {

class StoreEngine {
 public:
  // ═══════════════════════════════════════════════════════════════════════════
  // Type Definitions (using new unified type system)
  // ═══════════════════════════════════════════════════════════════════════════

  // Legacy AsyncLoadResult and load() interface have been fully removed;
  // callers should use ReplicaHandle returned from materialize_replica().

  using ReplicaInfo = runtime::ReplicaInfo;
  using ReplicaInventoryEntry = runtime::ReplicaInventoryEntry;
  using ReplicaPublishState = runtime::ReplicaPublishState;

  // ═══════════════════════════════════════════════════════════════════════════
  // Construction and Initialization
  // ═══════════════════════════════════════════════════════════════════════════

  explicit StoreEngine(const StoreEngineOptions& opts);

  ~StoreEngine();

  // ═══════════════════════════════════════════════════════════════════════════════════════
  // Public API
  // ═══════════════════════════════════════════════════════════════════════════════════════

  // ─────────────────────────────────────────────────────────────────────────
  // New materialize_replica() API (multi-device binding)
  // ─────────────────────────────────────────────────────────────────────────

  using MaterializeMode = loading::MaterializeMode;

  // ─────────────────────────────────────────────────────────────────────────
  // Lightweight handle that callers receive from materialize_replica().
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * @brief Unified materialization entry. Replaces materialize_replica().
   */
  absl::StatusOr<loading::ReplicaHandle> materialize_replica(
      const DeviceKey& target_device,
      MaterializeMode mode = MaterializeMode::AUTO,
      const loading::MaterializeHints& hints = {},
      std::optional<loading::DiskSource> disk_source = std::nullopt);

  absl::StatusOr<loading::MaterializeIntoTargetResult> materialize_into_target(
      const DeviceKey& target_device,
      const loading::IntoTargetLayout& target_layout,
      std::string_view canonical_index_json,
      uint64_t generation,
      const loading::MaterializeHints& hints = {},
      std::optional<loading::DiskSource> disk_source = std::nullopt);

  absl::StatusOr<loading::MaterializeIntoTargetResult> materialize_mapped_into_target(
      const DeviceKey& target_device,
      const loading::IntoTargetLayout& target_layout,
      const loader::ByteRangeMap& mapping,
      std::string_view canonical_index_json,
      uint64_t generation,
      const loading::MaterializeHints& hints,
      std::optional<loading::DiskSource> disk_source);

  absl::StatusOr<loading::MaterializeIntoTargetResult> materialize_mapped_into_target(
      const DeviceKey& target_device,
      const loading::IntoTargetLayout& target_layout,
      const loader::ByteRangeMap& mapping,
      std::string_view canonical_index_json,
      uint64_t generation,
      const loading::MaterializeHints& hints = {});

  absl::StatusOr<loading::MaterializeIntoTargetResult> materialize_mapped_loader_into_target(
      const DeviceKey& target_device,
      const loading::IntoTargetLayout& target_layout,
      std::unique_ptr<IArtifactLoader> loader,
      const loader::ByteRangeMap& mapping,
      const loading::MaterializeHints& hints,
      loading::MaterializationSource source_kind = loading::MaterializationSource::kLocalReplica);

  absl::StatusOr<runtime::ingestion::ArtifactLoweringResult> execute_artifact_lowering_plan(
      runtime::ingestion::ArtifactLoweringPlan plan);

  absl::StatusOr<loading::ReplicaHandle> materialize_view_from_assembly(
      std::string_view assembly_id,
      std::string_view target_artifact_id,
      std::string_view view_id,
      std::string_view view_spec_json,
      const DeviceKey& target_device,
      loading::TransformPlacement placement,
      const std::vector<std::string>* allowed_view_ids = nullptr);

  absl::StatusOr<SealAssemblyResult> seal_assembly(
      std::string_view assembly_id,
      bool publish_canonical,
      runtime::ingestion::MaterializationFacade::SealProgressCallback progress_cb = {},
      const std::vector<std::string>* allowed_view_ids = nullptr);

  absl::StatusOr<loading::ReplicaHandle> ingest_from_p2p(
      const std::string& artifact_identifier,
      const P2PSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints);

  absl::StatusOr<loading::ReplicaHandle> ingest_from_disk(
      const std::string& artifact_identifier,
      const loading::DiskSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints);

  // View planning/execution helpers (variant-aware path)
  static absl::StatusOr<loader::ViewPlan> compute_view_plan(
      std::string_view canonical_index_json,
      const loader::ViewSpec& spec);

  static absl::StatusOr<loader::ViewPlan> compute_view_plan(
      std::string_view canonical_index_json,
      const loader::ViewSpec& spec,
      absl::Span<const std::string> subset_names);

  static bool view_plan_allows_alias(const loader::ViewPlan& plan);

  static absl::StatusOr<std::string> compute_view_data_hash_from_source(
      loader::SeekableSource& base_source,
      const loader::ViewPlan& plan,
      size_t leaf_chunk_bytes = 4ULL * 1024 * 1024);

  using ViewPlacement = runtime::metadata::ViewPlacement;
  using ViewRegistrationKind = runtime::metadata::ViewRegistrationKind;
  using CanonicalRange = runtime::metadata::CanonicalRange;
  using ViewRegistration = runtime::metadata::ViewRegistration;

  // ------------------------------------------------------------------------
  // Memory Artifact Registration (coalesced) – Phase A (RFC-0006)
  // ------------------------------------------------------------------------

  using ArtifactRegistration = runtime::metadata::ArtifactRegistration;
  using RegistrationBeginResult = runtime::metadata::RegistrationBeginResult;
  using RegistrationCpuMemfdInfo = runtime::metadata::RegistrationCpuMemfdInfo;

  /**
   * @brief Begin registering an in-memory tensor dict replica.
   * Allocates target GPU memory and returns a CUDA IPC handle for the caller
   * (user process) to write tensor bytes directly into daemon-owned memory.
   */
  absl::StatusOr<RegistrationBeginResult> begin_register_artifact(const ArtifactRegistration& reg);

  // Return the in-process GPU pointer for a pending registration's staging buffer.
  // This is intended for daemon-internal copy paths; external clients must use
  // the CUDA IPC handle returned by begin_register_artifact().
  absl::StatusOr<uint64_t> get_registration_gpu_ptr(std::string_view registration_id) const;
  absl::StatusOr<RegistrationCpuMemfdInfo> get_registration_cpu_memfd_info(std::string_view registration_id) const;

  // CPU registration path removed

  /**
   * @brief Commit a previously begun registration.  Finalizes the replica by
   * exporting remote memory keys (if communication engine is enabled) and
   * registering the memory replica with Global Store.  On success, memory
   * ownership remains with the daemon and becomes discoverable by peers.
   */
  using RegistrationCommitResult = runtime::metadata::RegistrationCommitResult;

  absl::StatusOr<RegistrationCommitResult> commit_registered_artifact(std::string_view registration_id);

  absl::Status ingest_view_registration_chunk(
      std::string_view registration_id,
      uint64_t view_offset,
      absl::Span<const std::byte> data);
  absl::Status ingest_registration_chunk(
      std::string_view registration_id,
      uint64_t offset,
      absl::Span<const std::byte> data);
  absl::Status ingest_registration_written_range(std::string_view registration_id, uint64_t offset, uint64_t length);

  absl::StatusOr<uint64_t> get_view_registration_ingested_bytes(std::string_view registration_id);

  /**
   * @brief Abort a pending registration and release allocated memory.
   */
  absl::Status abort_registered_artifact(std::string_view registration_id);

  /**
   * @brief Refresh TTL for a pending registration to keep it alive.
   *
   * Extends the internal expiry_time used for TTL enforcement during
   * CommitRegisteredArtifact. No-op when ttl_ms == 0.
   */
  [[nodiscard]] absl::Status keep_alive_registered_artifact(std::string_view registration_id, uint32_t ttl_ms);

  // ------------------------------------------------------------------------
  // Query helpers (multi-device binding)
  // ------------------------------------------------------------------------

  /**
   * @brief Returns the set of devices where a given artifact_id is already loaded.
   */
  [[nodiscard]] std::vector<DeviceKey> get_resident_devices(
      std::string_view artifact_id,
      std::optional<std::string_view> view_id = std::nullopt) const;

  [[nodiscard]] std::vector<ReplicaInventoryEntry> get_ha_inventory() const;
  [[nodiscard]] std::optional<std::string> get_replica_global_store_id(const loading::ReplicaKey& key) const;
  void set_replica_global_store_id(const loading::ReplicaKey& key, std::string replica_id);

  [[nodiscard]] const StoreEngineOptions& options() const {
    return options_;
  }

  /**
   * @brief Returns a unique GPU device ordinal if the artifact resides on exactly
   *        one GPU; returns -1 if not present on any GPU; returns InvalidArgument
   *        when the artifact resides on multiple GPUs (ambiguous without a device).
   */
  [[nodiscard]] absl::StatusOr<int> get_unique_gpu_residency(
      std::string_view artifact_id,
      std::optional<std::string_view> view_id = std::nullopt) const;

  /**
   * @brief Inject a custom Global Store client (primarily for testing).
   * Ownership is shared so tests may reuse the stub beyond the StoreEngine lifetime.
   */
  void set_global_store_client_for_testing(std::shared_ptr<components::IGlobalStoreClient> client);
  void set_promotion_sync_hooks(runtime::PromotionSyncHooks hooks);

  [[nodiscard]] runtime::ReplicaPromotionManager* promotion_manager() const {
    return promotion_manager_.get();
  }

  void set_stable_cache_spill_evictable(
      std::function<bool(const loading::ReplicaKey&, const components::StableDramCachePolicy&)> callback);

  struct StableCacheAdmissionResult {
    bool admitted{false};
    bool skipped{false};
  };

  struct ReplicaBackingObservation {
    loading::ReplicaKey key;
    std::uint64_t size_bytes{0};
    common::memory::MemoryLocation memory_location{common::memory::MemoryLocation::NONE};
    bool cpu_memfd_available{false};
    bool cuda_ipc_available{false};
    runtime::ReplicaExportState remote_export_state{runtime::ReplicaExportState::kPresenceOnly};
    std::uint64_t remote_export_generation{0};
    bool remote_access_enabled{false};
  };

  // Apply/upgrade stable-DRAM cache policy for an existing CPU replica.
  // Returns {admitted=true} when the replica is tracked (including upgrades),
  // {skipped=true} when admission was best-effort and could not be satisfied.
  [[nodiscard]] absl::StatusOr<StableCacheAdmissionResult> admit_stable_cache_policy(
      const loading::ReplicaKey& key,
      const components::StableDramCachePolicy& policy);

  [[nodiscard]] absl::Status update_stable_cache_policy(
      const loading::ReplicaKey& key,
      const components::StableDramCachePolicy& policy,
      std::optional<absl::Time> retention_deadline = std::nullopt);

  /**
   * @brief Returns all ReplicaKey(s) that reside on a particular device.
   */
  [[nodiscard]] std::vector<loading::ReplicaKey> list_device_replicas(const DeviceKey& device) const;

  [[nodiscard]] std::optional<MemoryTierBudget::Snapshot> get_memory_tier_snapshot() const;
  [[nodiscard]] std::optional<MemoryTierConfig> get_memory_tier_config() const;
  absl::StatusOr<components::MemoryTierLeaseDescriptor> acquire_memory_tier_lease(
      const components::MemoryTierLeaseDescriptor& lease);
  absl::StatusOr<components::MemoryTierLeaseDescriptor> release_memory_tier_lease(
      const components::MemoryTierLeaseDescriptor& lease);

  // Replica management
  // ─────────────────────────────────────────────────────────────────────
  // NEW ReplicaKey-centric APIs (Multi-Device Binding)
  // ─────────────────────────────────────────────────────────────────────
  int wait_replica_ready(const loading::ReplicaKey& key);
  absl::Status unload_replica_status(const loading::ReplicaKey& key);
  int unload_replica(const loading::ReplicaKey& key);
  absl::Status retire_replica_status(const loading::ReplicaKey& key);
  [[nodiscard]] replica::MemoryState get_replica_state(const loading::ReplicaKey& key, DeviceType memory_type) const;
  absl::StatusOr<uint64_t> get_replica_gpu_ptr(const loading::ReplicaKey& key);
  // Return total artifact size in bytes for the given replica.
  absl::StatusOr<uint64_t> get_replica_size(const loading::ReplicaKey& key);
  [[nodiscard]] absl::StatusOr<ReplicaBackingObservation> inspect_replica_backing(const loading::ReplicaKey& key) const;
  [[nodiscard]] absl::StatusOr<std::unique_ptr<IArtifactLoader>> open_local_replica_loader(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location = common::memory::MemoryLocation::CPU) const;

  void set_replica_publish_state(const loading::ReplicaKey& key, ReplicaPublishState state);
  [[nodiscard]] ReplicaPublishState get_replica_publish_state(const loading::ReplicaKey& key) const;

  std::unique_ptr<runtime::RuntimeContextEvents::Subscription> subscribe_to_runtime_events(
      runtime::RuntimeContextEvents::Callback callback);

  // Remote memory registration helpers (ReplicaKey version)
  [[nodiscard]] absl::StatusOr<ExportRegistration> enable_remote_replica_access(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location);
  [[nodiscard]] absl::Status disable_remote_replica_access(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location);

  [[nodiscard]] absl::StatusOr<replica::UnifiedMemoryAuthority::ExportRegistration> set_replica_exported(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location,
      absl::Span<const uint32_t> chunks,
      bool on);

  [[nodiscard]] absl::StatusOr<replica::UnifiedMemoryAuthority::StableLease> acquire_replica_stable_lease(
      const loading::ReplicaKey& key,
      absl::Span<const uint32_t> chunks);

  [[nodiscard]] absl::Status release_replica_stable_lease(const replica::UnifiedMemoryAuthority::StableLease& lease);

  // Register a loaded replica with the Global Store if connected. When
  // artifact_id_override is provided it must be a canonical `mi2:` identifier
  // (e.g., disk-ingested replicas that computed hashes locally); otherwise the
  // `ReplicaKey.artifact_id` is published.
  [[nodiscard]] absl::Status register_replica_with_global_store(
      const loading::ReplicaKey& key,
      std::string_view artifact_id_override = {});

  // Deregister a memory replica from Global Store using worker identity.
  [[nodiscard]] absl::Status unregister_replica_from_global_store(std::string_view artifact_id, int device_id);

  // Key-mapping wrappers delegating to Global Store client. These
  // avoid exposing the client to callers and ensure we always use the Engine's
  // configured Global Store connection.
  absl::StatusOr<components::KeyMapping> resolve_key_mapping(
      std::string_view key,
      const components::RpcOptions& rpc_options = components::RpcOptions{});
  absl::Status upsert_key_mapping(
      std::string_view key,
      std::string_view artifact_id,
      absl::Duration ttl = absl::ZeroDuration());
  absl::StatusOr<components::KeyMappingSwapResult> swap_key_mapping(
      std::string_view key,
      std::string_view new_artifact_id,
      std::optional<std::string_view> expected_artifact_id = std::nullopt,
      std::optional<uint64_t> expected_generation = std::nullopt);
  absl::StatusOr<std::string> get_canonical_index_by_id(std::string_view artifact_id);
  absl::StatusOr<components::ViewMetadata> get_view_metadata(std::string_view artifact_id, std::string_view view_id);
  absl::Status revoke_key_mapping(std::string_view key);

  // --------------------------------------------------------------------
  // Memory & Registration helpers
  // --------------------------------------------------------------------
  int clear_mem();

  // Status queries
  [[nodiscard]] size_t get_mem_pool_size() const {
    return memory_pool_size_;
  }

  // New canonical name for transfer window size (bytes).
  [[nodiscard]] size_t get_tx_slice_bytes() const {
    return tx_slice_bytes_;
  }

  // UMA/VS artifact chunk size (bytes). Public read-only accessor for daemon status APIs
  // and controllers that need the authoritative artifact granularity.
  [[nodiscard]] size_t get_artifact_chunk_bytes() const {
    return artifact_chunk_bytes_;
  }

  [[nodiscard]] size_t get_available_memory() const;
  void update_memory_pool_metrics();
  [[nodiscard]] std::vector<ReplicaInfo> get_all_replicas_info() const;

  // GPU device queries (for status/health reporting)
  [[nodiscard]] int get_num_gpus() const {
    return replica_runtime_->device_manager().get_num_gpus();
  }

  [[nodiscard]] absl::StatusOr<size_t> get_device_total_memory(int device_id) const;
  [[nodiscard]] absl::StatusOr<size_t> get_device_free_memory(int device_id) const;

  // VS chunk-state snapshot API for daemon observers (read-only, non-authoritative telemetry).
  [[nodiscard]] std::vector<replica::ChunkState> get_chunk_states_telemetry(std::string_view artifact_id) const;

  // UMA-backed per-device chunk state snapshot for a given artifact. Returns
  // device-aware GPU states when device_id >= 0; returns an empty vector if
  // the replica for the given device is not found. Intended for daemon sync.
  [[nodiscard]] std::vector<replica::ChunkState> get_chunk_states_for_device(
      std::string_view artifact_id,
      int device_id) const;

  // UMA CPU states (authoritative): artifact-level accessor. When multiple
  // replicas exist, chooses a primary instance deterministically: prefer a CPU
  // instance if present; otherwise pick the GPU instance with the smallest
  // device ordinal. Returns empty on miss.
  [[nodiscard]] std::vector<replica::ChunkState> get_chunk_states_cpu_uma(std::string_view artifact_id) const;

  // Returns base pointer for the CPU replica if loaded.
  [[nodiscard]] absl::StatusOr<void*> get_replica_cpu_base_ptr(std::string_view artifact_id) const;

  // VS chunk locking APIs have been removed in UMA V3 final state.

  // Expose the configured communication manager to daemon for P2P export paths
  // that are not bound to a loaded replica (e.g., LIP-backed staged transfers).
  // Always non-null; may be disabled (see is_enabled()).
  [[nodiscard]] gsl::not_null<std::shared_ptr<components::CommunicationManager>> get_shared_comm_manager() const;

  [[nodiscard]] components::MetricsCollector::P2PTransferSnapshot get_p2p_transfer_snapshot() const;

 private:
  // ═══════════════════════════════════════════════════════════════════════════
  // Configuration
  // ═══════════════════════════════════════════════════════════════════════════

  const StoreEngineOptions options_;
  const std::filesystem::path storage_path_;
  const size_t memory_pool_size_;
  const size_t artifact_chunk_bytes_;
  const int num_thread_;
  const size_t tx_slice_bytes_;
  const std::chrono::milliseconds pinned_memory_timeout_;

  // ═══════════════════════════════════════════════════════════════════════════
  // Core Components
  // ═══════════════════════════════════════════════════════════════════════════

  std::unique_ptr<runtime::RuntimeEnv> runtime_env_;
  std::unique_ptr<runtime::ReplicaRuntime> replica_runtime_;
  std::unique_ptr<runtime::ReplicaPromotionManager> promotion_manager_;
  std::unique_ptr<runtime::metadata::MetadataGateway> metadata_gateway_;
  std::unique_ptr<runtime::IngestionRuntime> ingestion_runtime_;
  absl::StatusOr<loading::ReplicaHandle> ingest_from_buffer_internal(
      const std::string& artifact_identifier,
      const loading::InlineBufferSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints);

 public:
  // Inject worker identity after successful registration with Global Store
  void set_worker_identity(
      std::string worker_id,
      std::string node_id,
      std::string node_address,
      uint32_t grpc_port,
      uint32_t p2p_port) {
    if (runtime_env_) {
      runtime_env_->update_worker_identity(
          std::move(worker_id), std::move(node_id), std::move(node_address), grpc_port, p2p_port);
    }
    if (metadata_gateway_) {
      metadata_gateway_->refresh_override_endpoint();
    }
  }
};

} // namespace tensorcast::store
