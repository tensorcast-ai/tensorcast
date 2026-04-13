// Copyright (c) 2025-2026, TensorCast Team.

#include "replica_registry.h"

#include <algorithm>

#include "absl/container/flat_hash_set.h"
#include "core/store/replica/memory_state.h"

namespace tensorcast::store::components {

void ReplicaRegistry::append_to_secondary_indices_locked(size_t idx) {
  auto& entry = entries_[idx];
  auto& artifact_bucket = by_artifact_[entry.key.artifact_id];
  entry.artifact_bucket_pos = artifact_bucket.size();
  artifact_bucket.push_back(idx);

  auto& device_bucket = by_device_[entry.key.device];
  entry.device_bucket_pos = device_bucket.size();
  device_bucket.push_back(idx);
}

void ReplicaRegistry::remove_from_artifact_bucket_locked(size_t idx) {
  Entry& entry = entries_[idx];
  auto it = by_artifact_.find(entry.key.artifact_id);
  DCHECK(it != by_artifact_.end());
  auto& bucket = it->second;
  DCHECK_LT(entry.artifact_bucket_pos, bucket.size());
  const size_t swapped_idx = bucket.back();
  bucket[entry.artifact_bucket_pos] = swapped_idx;
  bucket.pop_back();
  if (entry.artifact_bucket_pos < bucket.size()) {
    entries_[swapped_idx].artifact_bucket_pos = entry.artifact_bucket_pos;
  }
  if (bucket.empty()) {
    by_artifact_.erase(it);
  }
}

void ReplicaRegistry::remove_from_device_bucket_locked(size_t idx) {
  Entry& entry = entries_[idx];
  auto it = by_device_.find(entry.key.device);
  DCHECK(it != by_device_.end());
  auto& bucket = it->second;
  DCHECK_LT(entry.device_bucket_pos, bucket.size());
  const size_t swapped_idx = bucket.back();
  bucket[entry.device_bucket_pos] = swapped_idx;
  bucket.pop_back();
  if (entry.device_bucket_pos < bucket.size()) {
    entries_[swapped_idx].device_bucket_pos = entry.device_bucket_pos;
  }
  if (bucket.empty()) {
    by_device_.erase(it);
  }
}

void ReplicaRegistry::patch_moved_entry_indices_locked(size_t idx) {
  const Entry& entry = entries_[idx];
  by_instance_.insert_or_assign(entry.key, idx);
  auto artifact_it = by_artifact_.find(entry.key.artifact_id);
  DCHECK(artifact_it != by_artifact_.end());
  DCHECK_LT(entry.artifact_bucket_pos, artifact_it->second.size());
  artifact_it->second[entry.artifact_bucket_pos] = idx;

  auto device_it = by_device_.find(entry.key.device);
  DCHECK(device_it != by_device_.end());
  DCHECK_LT(entry.device_bucket_pos, device_it->second.size());
  device_it->second[entry.device_bucket_pos] = idx;
}

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
  entries_.push_back(
      Entry{
          .key = key,
          .replica = std::move(replica),
          .last_access = std::chrono::system_clock::now(),
      });
  by_instance_.emplace(key, idx);
  append_to_secondary_indices_locked(idx);

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

  // De-duplicate aliases that point to the same replica instance. Use the
  // most recent access timestamp among aliases so hot replicas are not evicted
  // due to stale alias keys.
  absl::flat_hash_map<const replica::Replica*, TimedKey> dedup;
  dedup.reserve(entries_.size());
  for (const auto& e : entries_) {
    const auto* replica_ptr = e.replica.get();
    auto it = dedup.find(replica_ptr);
    if (it == dedup.end() || e.last_access > it->second.ts) {
      dedup[replica_ptr] = TimedKey{.key = e.key, .ts = e.last_access};
    }
  }
  std::vector<TimedKey> tmp;
  tmp.reserve(dedup.size());
  for (const auto& [_, timed_key] : dedup) {
    tmp.push_back(timed_key);
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

std::optional<std::pair<loading::ReplicaKey, std::shared_ptr<replica::Replica>>> ReplicaRegistry::erase(
    const loading::ReplicaKey& key) {
  absl::MutexLock lock(&mutex_);
  auto it = by_instance_.find(key);
  if (it == by_instance_.end()) {
    return std::nullopt;
  }

  const size_t idx = it->second;
  const size_t last_idx = entries_.size() - 1;
  Entry removed_entry{
      .key = entries_[idx].key,
      .replica = std::move(entries_[idx].replica),
      .last_access = entries_[idx].last_access,
      .artifact_bucket_pos = entries_[idx].artifact_bucket_pos,
      .device_bucket_pos = entries_[idx].device_bucket_pos,
  };
  remove_from_artifact_bucket_locked(idx);
  remove_from_device_bucket_locked(idx);
  by_instance_.erase(it);
  if (idx != last_idx) {
    entries_[idx] = std::move(entries_[last_idx]);
    patch_moved_entry_indices_locked(idx);
  }
  entries_.pop_back();
  return std::pair<loading::ReplicaKey, std::shared_ptr<replica::Replica>>{
      std::move(removed_entry.key),
      std::move(removed_entry.replica),
  };
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
  absl::flat_hash_set<const replica::Replica*> seen;
  seen.reserve(entries_.size());
  for (const auto& entry : entries_) {
    const auto* replica_ptr = entry.replica.get();
    if (!seen.insert(replica_ptr).second) {
      continue;
    }
    if (entry.replica->get_memory_state(location) == replica::MemoryState::LOADED) {
      ++count;
    }
  }
  return count;
}

uint64_t ReplicaRegistry::get_total_replica_size() const {
  absl::MutexLock lock(&mutex_);
  uint64_t total = 0;
  absl::flat_hash_set<const replica::Replica*> seen;
  seen.reserve(entries_.size());
  for (const auto& entry : entries_) {
    const auto* replica_ptr = entry.replica.get();
    if (!seen.insert(replica_ptr).second) {
      continue;
    }
    auto size_or = entry.replica->get_artifact_size();
    if (size_or.ok()) {
      total += *size_or;
    }
  }
  return total;
}

} // namespace tensorcast::store::components
