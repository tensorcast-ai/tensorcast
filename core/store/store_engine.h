// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/replica/chunk_meta.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/replica.h"
#include "core/store/store_engine_options.h"
#include "gsl/pointers"

namespace tensorcast::store {

namespace loading {
class MaterializeOrchestrator;
} // namespace loading

class StoreEngine {
  friend class loading::MaterializeOrchestrator;

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

  enum class MaterializeMode : uint8_t { AUTO, COPY_ONLY, LOAD_ONLY };

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

  // ------------------------------------------------------------------------
  // Memory Artifact Registration (coalesced) – Phase A (RFC-0006)
  // ------------------------------------------------------------------------

  struct ArtifactRegistration {
    std::string artifact_id; // Logical artifact identifier (old artifact_id)
    std::string tensor_index_key; // Canonical JSON SHA-256 hex (lowercase)
    std::optional<std::string> tensor_index_data; // Optional canonical JSON bytes for UPSERT
    std::string schema_version{"v2"}; // Data-format schema version
    std::string encoding{"json"}; // Encoding of index_data (if provided)
    int device_id{0}; // Local CUDA device ordinal
    uint64_t total_size_bytes{0}; // Total coalesced byte size (8B-aligned)
    bool enable_p2p{true}; // Whether to enable remote access
    uint32_t ttl_ms{0}; // Optional TTL for Begin→Commit (0 = no TTL)
  };

  struct RegistrationBeginResult {
    std::string registration_id; // Opaque id for Commit/Abort
    std::array<std::byte, sizeof(cudaIpcMemHandle_t)> cuda_ipc_handle_bytes{}; // CUDA IPC handle
    int device_id{0};
    uint64_t size_bytes{0};
  };

  /**
   * @brief Begin registering an in-memory tensor dict replica.
   * Allocates target GPU memory and returns a CUDA IPC handle for the caller
   * (user process) to write tensor bytes directly into daemon-owned memory.
   */
  absl::StatusOr<RegistrationBeginResult> begin_register_artifact(const ArtifactRegistration& reg);

  // DVMP (CPU UMA) registration begin. Allocates DVMP-backed PAGEABLE_CPU region
  // and prepares a pending registration tracked by registration_id. No GPU memory
  // is allocated in this path.
  absl::StatusOr<RegistrationBeginResult> begin_register_artifact_dvmp(const ArtifactRegistration& reg);

  // Writes a DVMP data chunk into the pending registration's DVMP region at the
  // specified virtual address offset. Updates DVMP chunk metadata accordingly.
  absl::Status feed_register_dvmp_chunk(
      std::string_view registration_id,
      uint64_t offset,
      const void* data,
      size_t bytes);

  /**
   * @brief Commit a previously begun registration.  Finalizes the replica by
   * exporting remote memory keys (if communication engine is enabled) and
   * registering the memory replica with Global Store.  On success, memory
   * ownership remains with the daemon and becomes discoverable by peers.
   */
  struct RegistrationCommitResult {
    std::string registration_id;
    std::string artifact_id;
    int device_id{0};
    uint64_t size_bytes{0};
    // RFC-0007: Expose content-address components for callers who need
    // authoritative descriptor details without recomputation.
    std::string index_multihash; // multibase(base32)-encoded multihash of canonical index
    std::string data_multihash; // multibase(base32)-encoded multihash of tree-hash root
    std::string schema_version; // e.g. "v2"
    std::string encoding; // e.g. "json" or "cbor"
  };

  absl::StatusOr<RegistrationCommitResult> commit_registered_artifact(std::string_view registration_id);

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
  absl::Status keep_alive_registered_artifact(std::string_view registration_id, uint32_t ttl_ms);

  // ------------------------------------------------------------------------
  // Query helpers (multi-device binding)
  // ------------------------------------------------------------------------

  /**
   * @brief Returns the set of devices where a given artifact_id is already loaded.
   */
  [[nodiscard]] std::vector<DeviceKey> get_resident_devices(std::string_view artifact_id) const;

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
  absl::StatusOr<CommRegistrationInfo> enable_remote_replica_access(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location);
  absl::Status disable_remote_replica_access(const loading::ReplicaKey& key, common::memory::MemoryLocation location);

  // Register a loaded replica with the Global Store if connected. When
  // artifact_id_override is provided, it is used as the identifier (e.g.,
  // content-addressed mi2:...); otherwise key.artifact_id is used.
  absl::Status register_replica_with_global_store(
      const loading::ReplicaKey& key,
      std::string_view artifact_id_override = {});

  // RFC-0014: Key-mapping wrappers delegating to Global Store client. These
  // avoid exposing the client to callers and ensure we always use the Engine's
  // configured Global Store connection.
  absl::StatusOr<components::GlobalStoreClient::KeyMapping> resolve_key_mapping(std::string_view key);
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
  [[nodiscard]] size_t get_chunk_size() const {
    return chunk_size_;
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

  // DVMP chunk-state snapshot API for daemon observers (read-only).
  [[nodiscard]] std::vector<replica::ChunkState> get_chunk_states(std::string_view artifact_id) const;

  // ─────────────────────────────────────────────────────────────────────────
  // Distributed Memory Pool (DVMP) chunk locking API
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * @brief Lock chunks for H2D or P2P transfer to prevent concurrent eviction.
   *
   * @param replica_key Fully-qualified key identifying the concrete replica.
   * @param chunk_indices List of chunk indices to lock.
   *
   * @return absl::Status OK on success, ResourceExhausted if any chunk is already locked.
   */
  absl::Status lock_chunks(const loading::ReplicaKey& replica_key, absl::Span<const uint32_t> chunk_indices);

  /**
   * @brief Unlock chunks after H2D or P2P transfer completion.
   *
   * @param replica_key Same ReplicaKey that was previously passed to lock_chunks().
   * @param chunk_indices List of chunk indices to unlock.
   * @param copied_gpu Whether the chunks were successfully copied to GPU.
   *
   * @return absl::Status OK on success.
   */
  absl::Status unlock_chunks(
      const loading::ReplicaKey& replica_key,
      absl::Span<const uint32_t> chunk_indices,
      bool copied_gpu);

 private:
  // ═══════════════════════════════════════════════════════════════════════════
  // Configuration
  // ═══════════════════════════════════════════════════════════════════════════

  const StoreEngineOptions options_;
  const std::filesystem::path storage_path_;
  const size_t memory_pool_size_;
  const int num_thread_;
  const size_t chunk_size_;
  const std::chrono::milliseconds pinned_memory_timeout_;

  // ═══════════════════════════════════════════════════════════════════════════
  // Core Components
  // ═══════════════════════════════════════════════════════════════════════════

  gsl::not_null<std::unique_ptr<components::DeviceManager>> device_manager_;
  gsl::not_null<std::unique_ptr<components::ReplicaRegistry>> replica_registry_;
  gsl::not_null<std::unique_ptr<components::MetricsCollector>> metrics_collector_;
  std::unique_ptr<components::GlobalStoreClient> global_store_client_;
  std::shared_ptr<components::CommunicationManager> comm_manager_;
  gsl::not_null<std::shared_ptr<common::memory::PinnedMemoryPool>> memory_pool_;
  gsl::not_null<std::shared_ptr<common::memory::DistributedVirtualMemoryPool>> dvmp_; // NEW: System-wide DVMP instance
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

  // Memory management helpers
  absl::Status try_evict_memory_for_replica(size_t required_size);
  std::shared_ptr<replica::Replica> get_or_create_replica(
      const std::string& artifact_identifier,
      const replica::ReplicaConfig& config);

  // Utility methods
  [[nodiscard]] size_t get_num_chunk_from_tensor_size(size_t tensor_size) const;

  // ═══════════════════════════════════════════════════════════════════════════
  // Pending Memory Registration State (RFC-0006)
  // ═══════════════════════════════════════════════════════════════════════════
  struct PendingRegistrationEntry {
    std::string registration_id;
    std::string artifact_id;
    int device_id{0};
    uint64_t size_bytes{0};
    std::string tensor_index_key;
    std::optional<std::string> tensor_index_data;
    std::string schema_version;
    std::string encoding;
    bool enable_p2p{true};
    std::shared_ptr<replica::Replica> replica; // Backing replica for memory ownership
    void* gpu_ptr{nullptr}; // Base GPU pointer (for diagnostics)
    cudaIpcMemHandle_t ipc_handle{}; // CUDA IPC handle bytes
    std::chrono::steady_clock::time_point expiry_time; // For TTL cleanup
    enum class Plan : uint8_t { COALESCED = 0, DVMP = 1 } plan{Plan::COALESCED};
  };

  std::mutex pending_mutex_;
  std::unordered_map<std::string, PendingRegistrationEntry> pending_regs_;

  // Worker identity (for Global Store registrations)
  std::string worker_id_;
  std::string node_id_;

 public:
  // Inject worker identity after successful registration with Global Store
  void set_worker_identity(std::string worker_id, std::string node_id) {
    worker_id_ = std::move(worker_id);
    node_id_ = std::move(node_id);
  }
};

} // namespace tensorcast::store
