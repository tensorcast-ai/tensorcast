// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <string_view>
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/worker_identity.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/memory_tier_budget.h"
#include "core/store/memory_tier_config.h"
#include "core/store/replica/chunk_state.h"
#include "core/store/replica/memory_state.h"
#include "core/store/runtime/ingestion/ingestion_runtime.h"
#include "core/store/runtime/metadata/metadata_gateway.h"
#include "core/store/runtime/metadata/metadata_types.h"
#include "core/store/runtime/replica/replica_info.h"
#include "core/store/runtime/replica/replica_runtime.h"
#include "core/store/runtime/runtime_env.h"
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
      const loading::MaterializeHints& hints = {});

  absl::StatusOr<loading::MaterializeIntoTargetResult> materialize_into_target(
      const DeviceKey& target_device,
      gsl::not_null<void*> target_ptr,
      uint64_t total_size,
      std::string_view canonical_index_json,
      uint64_t generation,
      const loading::MaterializeHints& hints = {});

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

  static bool view_plan_allows_alias(const loader::ViewPlan& plan);

  static absl::StatusOr<std::string> compute_view_data_hash_from_source(
      loader::SeekableSource& base_source,
      const loader::ViewPlan& plan,
      size_t leaf_chunk_bytes = 4ULL * 1024 * 1024);

  using ViewPlacement = runtime::metadata::ViewPlacement;
  using CanonicalRange = runtime::metadata::CanonicalRange;
  using ViewRegistration = runtime::metadata::ViewRegistration;

  // ------------------------------------------------------------------------
  // Memory Artifact Registration (coalesced) – Phase A (RFC-0006)
  // ------------------------------------------------------------------------

  using ArtifactRegistration = runtime::metadata::ArtifactRegistration;
  using RegistrationBeginResult = runtime::metadata::RegistrationBeginResult;

  /**
   * @brief Begin registering an in-memory tensor dict replica.
   * Allocates target GPU memory and returns a CUDA IPC handle for the caller
   * (user process) to write tensor bytes directly into daemon-owned memory.
   */
  absl::StatusOr<RegistrationBeginResult> begin_register_artifact(const ArtifactRegistration& reg);

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
  [[nodiscard]] std::vector<DeviceKey> get_resident_devices(std::string_view artifact_id) const;

  /**
   * @brief Returns a unique GPU device ordinal if the artifact resides on exactly
   *        one GPU; returns -1 if not present on any GPU; returns InvalidArgument
   *        when the artifact resides on multiple GPUs (ambiguous without a device).
   */
  [[nodiscard]] absl::StatusOr<int> get_unique_gpu_residency(std::string_view artifact_id) const;

  /**
   * @brief Inject a custom Global Store client (primarily for testing).
   * Ownership is shared so tests may reuse the stub beyond the StoreEngine lifetime.
   */
  void set_global_store_client_for_testing(std::shared_ptr<components::IGlobalStoreClient> client);

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
  int unload_replica(const loading::ReplicaKey& key);
  [[nodiscard]] replica::MemoryState get_replica_state(const loading::ReplicaKey& key, DeviceType memory_type) const;
  absl::StatusOr<uint64_t> get_replica_gpu_ptr(const loading::ReplicaKey& key);
  // Return total artifact size in bytes for the given replica.
  absl::StatusOr<uint64_t> get_replica_size(const loading::ReplicaKey& key);

  // Remote memory registration helpers (ReplicaKey version)
  [[nodiscard]] absl::StatusOr<ExportRegistration> enable_remote_replica_access(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location);
  [[nodiscard]] absl::Status disable_remote_replica_access(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location);

  // Register a loaded replica with the Global Store if connected. When
  // artifact_id_override is provided it must be a canonical `mi2:` identifier
  // (e.g., disk-ingested replicas that computed hashes locally); otherwise the
  // `ReplicaKey.artifact_id` is published.
  [[nodiscard]] absl::Status register_replica_with_global_store(
      const loading::ReplicaKey& key,
      std::string_view artifact_id_override = {});

  // Deregister a memory replica from Global Store using worker identity.
  [[nodiscard]] absl::Status unregister_replica_from_global_store(std::string_view artifact_id, int device_id);

  // RFC-0014: Key-mapping wrappers delegating to Global Store client. These
  // avoid exposing the client to callers and ensure we always use the Engine's
  // configured Global Store connection.
  absl::StatusOr<components::KeyMapping> resolve_key_mapping(std::string_view key);
  absl::Status upsert_key_mapping(
      std::string_view key,
      std::string_view artifact_id,
      std::string_view disk_path = {},
      absl::Duration ttl = absl::ZeroDuration());
  absl::StatusOr<std::string> get_canonical_index_by_id(std::string_view artifact_id);
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

  // VS chunk locking APIs have been removed in UMA V3 final state.

  // Expose the configured communication manager to daemon for P2P export paths
  // that are not bound to a loaded replica (e.g., LIP-backed staged transfers).
  // Always non-null; may be disabled (see is_enabled()).
  [[nodiscard]] gsl::not_null<std::shared_ptr<components::CommunicationManager>> get_shared_comm_manager() const;

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
  std::unique_ptr<runtime::metadata::MetadataGateway> metadata_gateway_;
  std::unique_ptr<runtime::IngestionRuntime> ingestion_runtime_;
  static absl::StatusOr<loading::ReplicaHandle> ingest_from_buffer_internal(
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
