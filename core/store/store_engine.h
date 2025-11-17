// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <optional>
#include <string_view>
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_identity.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/registration/artifact_registration_manager.h"
#include "core/store/components/replica_registry.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/loading/materialization/materialization_backend.h"
#include "core/store/replica/chunk_state.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/replica.h"
#include "core/store/store_engine_options.h"
#include "core/store/view_utils.h"
#include "gsl/pointers"

namespace tensorcast::store {

namespace loading {
struct MaterializationDeps;
} // namespace loading

namespace loader {
struct ViewPlan;
struct ViewSpec;
struct ViewWritePlan;
struct BidirectionalViewPlan;
class ViewIngestExecutor;
class SeekableSource;
} // namespace loader

class StoreEngine : public loading::MaterializationBackend {
 public:
  // ═══════════════════════════════════════════════════════════════════════════
  // Type Definitions (using new unified type system)
  // ═══════════════════════════════════════════════════════════════════════════

  // Legacy AsyncLoadResult and load() interface have been fully removed;
  // callers should use ReplicaHandle returned from materialize_replica().

  struct ReplicaInfo {
    std::string artifact_id;
    uint64_t size_bytes;
    common::memory::MemoryLocation cpu_state;
    common::memory::MemoryLocation gpu_state;
    int gpu_device_id;
    std::string gpu_device_uuid;
    bool is_registered_for_comm;
    std::chrono::time_point<std::chrono::system_clock> last_access_time;
    std::chrono::time_point<std::chrono::system_clock> load_time;
  };

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

  // loading::MaterializationBackend
  absl::StatusOr<loading::ReplicaHandle> ingest_from_p2p(
      const std::string& artifact_identifier,
      const P2PSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints) override;

  absl::StatusOr<loading::ReplicaHandle> ingest_from_disk(
      const std::string& artifact_identifier,
      const loading::DiskSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints) override;

  // View planning/execution helpers (variant-aware path)
  static absl::StatusOr<loader::ViewPlan> compute_view_plan(
      std::string_view canonical_index_json,
      const loader::ViewSpec& spec);

  static bool view_plan_allows_alias(const loader::ViewPlan& plan);

  static absl::StatusOr<std::string> compute_view_data_hash_from_source(
      loader::SeekableSource& base_source,
      const loader::ViewPlan& plan,
      size_t leaf_chunk_bytes = 4ULL * 1024 * 1024);

  using ViewPlacement = components::ViewPlacement;
  using CanonicalRange = components::CanonicalRange;
  using ViewRegistration = components::ViewRegistration;

  // ------------------------------------------------------------------------
  // Memory Artifact Registration (coalesced) – Phase A (RFC-0006)
  // ------------------------------------------------------------------------

  using ArtifactRegistration = components::ArtifactRegistration;
  using RegistrationBeginResult = components::RegistrationBeginResult;

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
  using RegistrationCommitResult = components::RegistrationCommitResult;

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
  absl::StatusOr<int> get_unique_gpu_residency(std::string_view artifact_id) const;

  /**
   * @brief Inject a custom Global Store client (primarily for testing).
   * Ownership is shared so tests may reuse the stub beyond the StoreEngine lifetime.
   */
  void set_global_store_client_for_testing(std::shared_ptr<components::IGlobalStoreClient> client);

  /**
   * @brief Returns all ReplicaKey(s) that reside on a particular device.
   */
  [[nodiscard]] std::vector<loading::ReplicaKey> list_device_replicas(const DeviceKey& device) const;

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
      std::string_view artifact_id_override = {}) override;

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
    return device_manager_->get_num_gpus();
  }

  absl::StatusOr<size_t> get_device_total_memory(int device_id) const;
  absl::StatusOr<size_t> get_device_free_memory(int device_id) const;

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
  [[nodiscard]] gsl::not_null<std::shared_ptr<components::CommunicationManager>> get_shared_comm_manager() const {
    return comm_manager_;
  }

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

  gsl::not_null<std::unique_ptr<components::DeviceManager>> device_manager_;
  gsl::not_null<std::unique_ptr<components::ReplicaRegistry>> replica_registry_;
  gsl::not_null<std::unique_ptr<components::MetricsCollector>> metrics_collector_;
  std::shared_ptr<components::IGlobalStoreClient> global_store_client_;
  gsl::not_null<std::shared_ptr<components::CommunicationManager>> comm_manager_;
  gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> memory_pool_;
  // ═══════════════════════════════════════════════════════════════════════════
  // Internal Helper Methods
  // ═══════════════════════════════════════════════════════════════════════════

  // Constructor helpers
  void initialize_components();
  void initialize_global_store(const StoreEngineOptions& opts);
  void initialize_communication_manager(const StoreEngineOptions& opts);

  // Replica loading helpers - using new unified types
  absl::StatusOr<loading::ReplicaHandle> ingest_from_disk_internal(
      const std::string& artifact_identifier,
      const loading::DiskSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints);

  absl::StatusOr<loading::ReplicaHandle> ingest_from_p2p_internal(
      const std::string& artifact_identifier,
      const P2PSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints);

  static absl::StatusOr<loading::ReplicaHandle> ingest_from_buffer_internal(
      const std::string& artifact_identifier,
      const loading::InlineBufferSource& source,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints);
  [[nodiscard]] loading::MaterializationDeps make_materialization_deps();

  // Memory management helpers
  absl::Status try_evict_memory_for_replica(size_t required_size);
  std::shared_ptr<replica::Replica> get_or_create_replica(
      const std::string& artifact_identifier,
      const replica::ReplicaConfig& config);

  // Utility methods
  [[nodiscard]] size_t get_num_chunk_from_tensor_size(size_t tensor_size) const;

  std::unique_ptr<components::ArtifactRegistrationManager> registration_manager_;

  // Worker identity (for Global Store registrations)
  std::string worker_id_;
  std::string node_id_;
  std::string node_address_;
  uint32_t grpc_port_{0};
  uint32_t p2p_port_{0};

 public:
  // Inject worker identity after successful registration with Global Store
  void set_worker_identity(
      std::string worker_id,
      std::string node_id,
      std::string node_address,
      uint32_t grpc_port,
      uint32_t p2p_port) {
    worker_id_ = std::move(worker_id);
    node_id_ = std::move(node_id);
    node_address_ = std::move(node_address);
    grpc_port_ = grpc_port;
    p2p_port_ = p2p_port;
    if (global_store_client_) {
      global_store_client_->update_local_endpoint(node_id_, node_address_, grpc_port_, p2p_port_);
    }
    if (registration_manager_) {
      components::WorkerIdentity identity{
          .worker_id = worker_id_,
          .node_id = node_id_,
          .node_address = node_address_,
          .grpc_port = grpc_port_,
          .p2p_port = p2p_port_};
      registration_manager_->set_worker_identity(std::move(identity));
    }
  }
};

} // namespace tensorcast::store
