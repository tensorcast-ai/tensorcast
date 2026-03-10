// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/store/replica/replica.h"
#include "gsl/pointers"

namespace tensorcast::store::components {

/**
 * @brief Thread-safe registry for managing loaded replicas.
 *
 * This component handles:
 * - Replica storage and retrieval
 * - Access time tracking
 * - Thread-safe operations on the replica map
 * - Replica lifecycle coordination
 */
class ReplicaRegistry {
 public:
  ReplicaRegistry() = default;
  ~ReplicaRegistry() = default;

  // Disable copy and move
  ReplicaRegistry(const ReplicaRegistry&) = delete;
  ReplicaRegistry& operator=(const ReplicaRegistry&) = delete;
  ReplicaRegistry(ReplicaRegistry&&) = delete;
  ReplicaRegistry& operator=(ReplicaRegistry&&) = delete;

  /**
   * @brief Remove every replica instance from the registry, returning the removed
   *        replicas so the caller can perform any necessary cleanup.
   */
  std::vector<std::pair<loading::ReplicaKey, std::shared_ptr<replica::Replica>>> clear_all()
      ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Returns the number of registered replica instances.
   */
  size_t size() const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Counts how many replica instances are currently LOADED at the given
   *        memory location (CPU / GPU).
   */
  size_t get_replica_count_by_location(common::memory::MemoryLocation location) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Calculates the total size (in bytes) of all loaded replicas, ignoring
   *        any instances where the size cannot be determined.
   */
  uint64_t get_total_replica_size() const ABSL_LOCKS_EXCLUDED(mutex_);

  // -----------------------------------------------------------------------
  // New API (Multi-Device Binding) – Works with ReplicaKey & DeviceKey
  // -----------------------------------------------------------------------

  /**
   * @brief Insert a new replica instance into the registry.
   *        Returns AlreadyExists if the same ReplicaKey already exists.
   */
  absl::Status emplace(const loading::ReplicaKey& key, gsl::not_null<std::shared_ptr<replica::Replica>> replica)
      ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Find the replica instance associated with the given ReplicaKey.
   *        Updates its last_access timestamp on success.
   */
  absl::StatusOr<std::shared_ptr<replica::Replica>> find(const loading::ReplicaKey& key) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Return ReplicaKeys for all instances whose artifact_id matches the query.
   */
  std::vector<loading::ReplicaKey> find_by_artifact(std::string_view artifact_id) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Return ReplicaKeys for all instances bound to the given device.
   */
  std::vector<loading::ReplicaKey> find_by_device(const DeviceKey& device) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Return ReplicaKeys sorted by least-recently-used order (oldest first).
   */
  std::vector<loading::ReplicaKey> get_lru_instances() const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Erase the exact replica instance for `key`.
   *        Returns std::nullopt when the key is not present.
   */
  std::optional<std::pair<loading::ReplicaKey, std::shared_ptr<replica::Replica>>> erase(const loading::ReplicaKey& key)
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  void rebuild_indices_locked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_;

  // ────────────────────────────────────────────────────────────────────────
  // New multi-index storage (prototyped, not yet implemented)
  // ────────────────────────────────────────────────────────────────────────

  struct Entry {
    loading::ReplicaKey key;
    std::shared_ptr<replica::Replica> replica;
    mutable std::chrono::time_point<std::chrono::system_clock> last_access;
  };

  std::vector<Entry> entries_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<loading::ReplicaKey, size_t, loading::ReplicaKeyHash> by_instance_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<std::string, std::vector<size_t>> by_artifact_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<DeviceKey, std::vector<size_t>, DeviceKeyHash> by_device_ ABSL_GUARDED_BY(mutex_);
};

} // namespace tensorcast::store::components
