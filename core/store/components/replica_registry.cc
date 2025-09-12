// Copyright (c) 2025, TensorCast Team.

#include "replica_registry.h"

#include <algorithm>

#include "core/store/replica/memory_state.h"

namespace tensorcast::store::components {

// ─────────────────────────────────────────────────────────────────────────────
// New ReplicaKey-based API implementation
// ─────────────────────────────────────────────────────────────────────────────

absl::Status ReplicaRegistry::emplace(
    const loading::ReplicaKey& key,
    gsl::not_null<std::shared_ptr<replica::Replica>> replica) {
  absl::MutexLock lock(&mutex_);
  if (by_instance_.contains(key)) {
    return absl::AlreadyExistsError("Instance already registered");
  }

  const size_t idx = entries_.size();
  entries_.push_back(Entry{.key = key, .replica = std::move(replica), .last_access = std::chrono::system_clock::now()});
  by_instance_.emplace(key, idx);
  by_artifact_[key.artifact_id].push_back(idx);
  by_device_[key.device].push_back(idx);

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<replica::Replica>> ReplicaRegistry::find(const loading::ReplicaKey& key) const {
  absl::MutexLock lock(&mutex_);
  auto it = by_instance_.find(key);
  if (it == by_instance_.end()) {
    return absl::NotFoundError("Replica not found");
  }
  const size_t idx = it->second;
  entries_[idx].last_access = std::chrono::system_clock::now();
  return entries_[idx].replica;
}

std::vector<loading::ReplicaKey> ReplicaRegistry::find_by_artifact(std::string_view artifact_id) const {
  absl::MutexLock lock(&mutex_);
  std::vector<loading::ReplicaKey> result;
  auto it = by_artifact_.find(std::string(artifact_id));
  if (it == by_artifact_.end()) {
    return result;
  }
  const auto& indices = it->second;
  result.reserve(indices.size());
  for (size_t idx : indices) {
    result.push_back(entries_[idx].key);
    entries_[idx].last_access = std::chrono::system_clock::now();
  }
  return result;
}

std::vector<loading::ReplicaKey> ReplicaRegistry::find_by_device(const DeviceKey& device) const {
  absl::MutexLock lock(&mutex_);
  std::vector<loading::ReplicaKey> result;
  auto it = by_device_.find(device);
  if (it == by_device_.end()) {
    return result;
  }
  const auto& indices = it->second;
  result.reserve(indices.size());
  for (size_t idx : indices) {
    result.push_back(entries_[idx].key);
    entries_[idx].last_access = std::chrono::system_clock::now();
  }
  return result;
}

// -----------------------------------------------------------------------------

std::vector<loading::ReplicaKey> ReplicaRegistry::get_lru_instances() const {
  absl::MutexLock lock(&mutex_);

  struct TimedKey {
    loading::ReplicaKey key;
    std::chrono::time_point<std::chrono::system_clock> ts;
  };

  std::vector<TimedKey> tmp;
  tmp.reserve(entries_.size());
  for (const auto& e : entries_) {
    tmp.push_back(TimedKey{.key = e.key, .ts = e.last_access});
  }

  std::ranges::sort(tmp, [](const TimedKey& a, const TimedKey& b) { return a.ts < b.ts; });

  std::vector<loading::ReplicaKey> ordered;
  ordered.reserve(tmp.size());
  for (const auto& tk : tmp) {
    ordered.push_back(tk.key);
  }
  return ordered;
}

// -----------------------------------------------------------------------------

std::vector<std::pair<loading::ReplicaKey, std::shared_ptr<replica::Replica>>> ReplicaRegistry::clear_all() {
  absl::MutexLock lock(&mutex_);

  std::vector<std::pair<loading::ReplicaKey, std::shared_ptr<replica::Replica>>> removed;
  removed.reserve(entries_.size());

  for (auto& entry : entries_) {
    removed.emplace_back(entry.key, std::move(entry.replica));
  }

  // Wipe all indices and storage vectors
  entries_.clear();
  by_instance_.clear();
  by_artifact_.clear();
  by_device_.clear();

  return removed;
}

size_t ReplicaRegistry::size() const {
  absl::MutexLock lock(&mutex_);
  return entries_.size();
}

size_t ReplicaRegistry::get_replica_count_by_location(common::memory::MemoryLocation location) const {
  absl::MutexLock lock(&mutex_);
  size_t count = 0;
  for (const auto& entry : entries_) {
    if (entry.replica->get_memory_state(location) == replica::MemoryState::LOADED) {
      ++count;
    }
  }
  return count;
}

uint64_t ReplicaRegistry::get_total_replica_size() const {
  absl::MutexLock lock(&mutex_);
  uint64_t total = 0;
  for (const auto& entry : entries_) {
    auto size_or = entry.replica->get_artifact_size();
    if (size_or.ok()) {
      total += *size_or;
    }
  }
  return total;
}

} // namespace tensorcast::store::components
