// Copyright (c) 2025, StepCast Team. All rights reserved.

//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
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
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/store/checkpoint_store_options.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/model_registry.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/model/memory_state.h"
#include "core/store/model/model.h"
#include "gsl/pointers"

namespace stepcast::store {

class PrepareOrchestrator; // Forward declaration for friend access

class CheckpointStore {
  friend class PrepareOrchestrator;

 public:
  // ═══════════════════════════════════════════════════════════════════════════
  // Type Definitions (using new unified type system)
  // ═══════════════════════════════════════════════════════════════════════════

  // Legacy AsyncLoadResult and load() interface have been fully removed;
  // callers should use ModelHandle returned from prepare().

  struct ModelInfo {
    std::string model_id;
    uint64_t size_bytes;
    ModelLocation cpu_state;
    ModelLocation gpu_state;
    int gpu_device_id;
    std::string gpu_device_uuid;
    bool is_registered_for_comm;
    std::chrono::time_point<std::chrono::system_clock> last_access_time;
    std::chrono::time_point<std::chrono::system_clock> load_time;
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // Construction and Initialization
  // ═══════════════════════════════════════════════════════════════════════════

  explicit CheckpointStore(const CheckpointStoreOptions& opts);

  ~CheckpointStore();

  // ═══════════════════════════════════════════════════════════════════════════════════════
  // Public API
  // ═══════════════════════════════════════════════════════════════════════════════════════

  // ─────────────────────────────────────────────────────────────────────────
  // New prepare() API (multi-device binding)
  // ─────────────────────────────────────────────────────────────────────────

  enum class PrepareMode : uint8_t { AUTO, COPY_ONLY, LOAD_ONLY };

  // ─────────────────────────────────────────────────────────────────────────
  // Lightweight handle that callers receive from prepare().
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * @brief Unified load / copy entry. Replaces load(ModelLoadSpec).
   *
   * @param model_id      Model identifier (logical name, e.g. "llama-7b")
   * @param target_device Target device where the model instance should reside.
   * @param mode          Loading strategy hint.
   * @param hints         Advanced tuning knobs (reuse existing struct).
   */
  absl::StatusOr<ModelHandle> prepare(
      std::string_view model_id,
      const DeviceKey& target_device,
      PrepareMode mode = PrepareMode::AUTO,
      const LoadingHints& hints = {});

  // ------------------------------------------------------------------------
  // Memory TensorDict Registration (coalesced) – Phase A (RFC-0006)
  // ------------------------------------------------------------------------

  struct TensorDictRegistration {
    std::string model_id; // Logical model identifier
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
  absl::StatusOr<RegistrationBeginResult> begin_register_tensor_dict(const TensorDictRegistration& reg);

  /**
   * @brief Commit a previously begun registration.  Finalizes the replica by
   * exporting remote memory keys (if communication engine is enabled) and
   * registering the memory replica with Global Store.  On success, memory
   * ownership remains with the daemon and becomes discoverable by peers.
   */
  struct RegistrationCommitResult {
    std::string registration_id;
    std::string model_id;
    int device_id{0};
    uint64_t size_bytes{0};
  };

  absl::StatusOr<RegistrationCommitResult> commit_registered_tensor_dict(std::string_view registration_id);

  /**
   * @brief Abort a pending registration and release allocated memory.
   */
  absl::Status abort_registered_tensor_dict(std::string_view registration_id);

  // ------------------------------------------------------------------------
  // Query helpers (multi-device binding)
  // ------------------------------------------------------------------------

  /**
   * @brief Returns the set of devices where a given model_id is already loaded.
   */
  [[nodiscard]] std::vector<DeviceKey> get_loaded_devices(std::string_view model_id) const;

  /**
   * @brief Returns all InstanceKey(s) that reside on a particular device.
   */
  [[nodiscard]] std::vector<InstanceKey> list_device_models(const DeviceKey& device) const;

  // Model management
  // ─────────────────────────────────────────────────────────────────────
  // NEW InstanceKey-centric APIs (Multi-Device Binding)
  // ─────────────────────────────────────────────────────────────────────
  int wait_instance_ready(const InstanceKey& key);
  int unload_instance(const InstanceKey& key);
  [[nodiscard]] MemoryState get_instance_state(const InstanceKey& key, DeviceType memory_type) const;
  absl::StatusOr<uint64_t> get_instance_gpu_ptr(const InstanceKey& key);

  // Remote memory registration helpers (InstanceKey version)
  absl::StatusOr<CommRegistrationInfo> enable_remote_instance_access(const InstanceKey& key, ModelLocation location);
  absl::Status disable_remote_instance_access(const InstanceKey& key, ModelLocation location);

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
  [[nodiscard]] std::vector<ModelInfo> get_all_models_info() const;

  // ─────────────────────────────────────────────────────────────────────────
  // Distributed Memory Pool (DVMP) chunk locking API
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * @brief Lock chunks for H2D or P2P transfer to prevent concurrent eviction.
   *
   * @param instance_key Fully-qualified key identifying the concrete model instance.
   *        Using InstanceKey avoids ambiguity when the same model_id is loaded on
   *        multiple GPUs.
   * @param chunk_indices List of chunk indices to lock.
   *
   * @return absl::Status OK on success, ResourceExhausted if any chunk is already locked.
   */
  absl::Status lock_chunks(const InstanceKey& instance_key, absl::Span<const uint32_t> chunk_indices);

  /**
   * @brief Unlock chunks after H2D or P2P transfer completion.
   *
   * @param instance_key Same InstanceKey that was previously passed to lock_chunks().
   * @param chunk_indices List of chunk indices to unlock.
   * @param copied_gpu Whether the chunks were successfully copied to GPU.
   *
   * @return absl::Status OK on success.
   */
  absl::Status unlock_chunks(
      const InstanceKey& instance_key,
      absl::Span<const uint32_t> chunk_indices,
      bool copied_gpu);

 private:
  // ═══════════════════════════════════════════════════════════════════════════
  // Configuration
  // ═══════════════════════════════════════════════════════════════════════════

  const std::filesystem::path storage_path_;
  const size_t memory_pool_size_;
  const int num_thread_;
  const size_t chunk_size_;
  const std::chrono::milliseconds pinned_memory_timeout_;

  // ═══════════════════════════════════════════════════════════════════════════
  // Core Components
  // ═══════════════════════════════════════════════════════════════════════════

  gsl::not_null<std::unique_ptr<DeviceManager>> device_manager_;
  gsl::not_null<std::unique_ptr<ModelRegistry>> model_registry_;
  gsl::not_null<std::unique_ptr<MetricsCollector>> metrics_collector_;
  std::unique_ptr<GlobalStoreClient> global_store_client_;
  std::shared_ptr<CommunicationManager> comm_manager_;
  gsl::not_null<std::shared_ptr<PinnedMemoryPool>> memory_pool_;
  gsl::not_null<std::shared_ptr<memory::DistributedVirtualMemoryPool>> dvmp_; // NEW: System-wide DVMP instance
  // ═══════════════════════════════════════════════════════════════════════════
  // Internal Helper Methods
  // ═══════════════════════════════════════════════════════════════════════════

  // Constructor helpers
  void initialize_components();
  void initialize_global_store(const CheckpointStoreOptions& opts);
  void initialize_communication_manager(const CheckpointStoreOptions& opts);

  // Model loading helpers - using new unified types
  absl::StatusOr<::stepcast::store::ModelHandle> load_from_disk_internal(
      const std::string& model_identifier,
      const DiskSource& source,
      const ModelTarget& target,
      const LoadingHints& hints);

  absl::StatusOr<::stepcast::store::ModelHandle> load_from_p2p_internal(
      const std::string& model_identifier,
      const P2PSource& source,
      const ModelTarget& target,
      const LoadingHints& hints);

  absl::StatusOr<::stepcast::store::ModelHandle> load_from_buffer_internal(
      const std::string& model_identifier,
      const InlineBufferSource& source,
      const ModelTarget& target,
      const LoadingHints& hints);

  // Memory management helpers
  absl::Status try_evict_memory_for_model(size_t required_size);
  std::shared_ptr<Model> get_or_create_model(const std::string& model_identifier, const ModelConfig& config);

  // Utility methods
  [[nodiscard]] size_t get_num_chunk_from_tensor_size(size_t tensor_size) const;

  // ═══════════════════════════════════════════════════════════════════════════
  // Pending Memory Registration State (RFC-0006)
  // ═══════════════════════════════════════════════════════════════════════════
  struct PendingRegistrationEntry {
    std::string registration_id;
    std::string model_id;
    int device_id{0};
    uint64_t size_bytes{0};
    std::string tensor_index_key;
    std::optional<std::string> tensor_index_data;
    std::string schema_version;
    std::string encoding;
    bool enable_p2p{true};
    std::shared_ptr<Model> model; // Backing model for memory ownership
    void* gpu_ptr{nullptr}; // Base GPU pointer (for diagnostics)
    cudaIpcMemHandle_t ipc_handle{}; // CUDA IPC handle bytes
    std::chrono::steady_clock::time_point expiry_time; // For TTL cleanup
  };

  std::mutex pending_mutex_;
  std::unordered_map<std::string, PendingRegistrationEntry> pending_regs_;
};

} // namespace stepcast::store