// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/store/model/model.h"

namespace stepcast::store {

/**
 * @brief Thread-safe registry for managing loaded models.
 *
 * This component handles:
 * - Model storage and retrieval
 * - Access time tracking
 * - Thread-safe operations on the model map
 * - Model lifecycle coordination
 */
class ModelRegistry {
 public:
  ModelRegistry() = default;
  ~ModelRegistry() = default;

  // Disable copy and move
  ModelRegistry(const ModelRegistry&) = delete;
  ModelRegistry& operator=(const ModelRegistry&) = delete;
  ModelRegistry(ModelRegistry&&) = delete;
  ModelRegistry& operator=(ModelRegistry&&) = delete;

  /**
   * @brief Remove every model instance from the registry, returning the removed
   *        models so the caller can perform any necessary cleanup.
   */
  std::vector<std::pair<InstanceKey, std::shared_ptr<Model>>> clear_all() ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Returns the number of registered model instances.
   */
  size_t size() const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Counts how many model instances are currently LOADED at the given
   *        memory location (CPU / GPU).
   */
  size_t get_model_count_by_location(ModelLocation location) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Calculates the total size (in bytes) of all loaded models, ignoring
   *        any instances where the size cannot be determined.
   */
  uint64_t get_total_model_size() const ABSL_LOCKS_EXCLUDED(mutex_);

  // -----------------------------------------------------------------------
  // New API (Multi-Device Binding) – Works with InstanceKey & DeviceKey
  // -----------------------------------------------------------------------

  /**
   * @brief Insert a new model instance into the registry.
   *        Returns AlreadyExists if the same InstanceKey already exists.
   */
  absl::Status emplace(const InstanceKey& key, std::shared_ptr<Model> model) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Find the model instance associated with the given InstanceKey.
   *        Updates its last_access timestamp on success.
   */
  absl::StatusOr<std::shared_ptr<Model>> find(const InstanceKey& key) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Return InstanceKeys for all instances whose model_id matches the query.
   */
  std::vector<InstanceKey> find_by_model(std::string_view model_id) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Return InstanceKeys for all instances bound to the given device.
   */
  std::vector<InstanceKey> find_by_device(const DeviceKey& device) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Return InstanceKeys sorted by least-recently-used order (oldest first).
   */
  std::vector<InstanceKey> get_lru_instances() const ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  mutable absl::Mutex mutex_;
  // ────────────────────────────────────────────────────────────────────────
  // New multi-index storage (prototyped, not yet implemented)
  // ────────────────────────────────────────────────────────────────────────

  struct Entry {
    InstanceKey key;
    std::shared_ptr<Model> model;
    mutable std::chrono::time_point<std::chrono::system_clock> last_access;
  };

  std::vector<Entry> entries_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<InstanceKey, size_t, InstanceKeyHash> by_instance_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<std::string, std::vector<size_t>> by_model_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<DeviceKey, std::vector<size_t>, DeviceKeyHash> by_device_ ABSL_GUARDED_BY(mutex_);
};

} // namespace stepcast::store