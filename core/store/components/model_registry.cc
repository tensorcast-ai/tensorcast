// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "model_registry.h"

#include <algorithm>

#include "core/store/model/memory_state.h"

namespace stepcast::store {

// ─────────────────────────────────────────────────────────────────────────────
// New InstanceKey-based API implementation
// ─────────────────────────────────────────────────────────────────────────────

absl::Status ModelRegistry::emplace(const InstanceKey& key, std::shared_ptr<Model> model) {
  if (!model) {
    return absl::InvalidArgumentError("Cannot register null model");
  }

  absl::MutexLock lock(&mutex_);
  if (by_instance_.contains(key)) {
    return absl::AlreadyExistsError("Instance already registered");
  }

  const size_t idx = entries_.size();
  entries_.push_back(Entry{key, std::move(model), std::chrono::system_clock::now()});
  by_instance_.emplace(key, idx);
  by_model_[key.model_id].push_back(idx);
  by_device_[key.device].push_back(idx);

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<Model>> ModelRegistry::find(const InstanceKey& key) const {
  absl::MutexLock lock(&mutex_);
  auto it = by_instance_.find(key);
  if (it == by_instance_.end()) {
    return absl::NotFoundError("Model instance not found");
  }
  const size_t idx = it->second;
  entries_[idx].last_access = std::chrono::system_clock::now();
  return entries_[idx].model;
}

std::vector<InstanceKey> ModelRegistry::find_by_model(std::string_view model_id) const {
  absl::MutexLock lock(&mutex_);
  std::vector<InstanceKey> result;
  auto it = by_model_.find(std::string(model_id));
  if (it == by_model_.end()) {
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

std::vector<InstanceKey> ModelRegistry::find_by_device(const DeviceKey& device) const {
  absl::MutexLock lock(&mutex_);
  std::vector<InstanceKey> result;
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

std::vector<InstanceKey> ModelRegistry::get_lru_instances() const {
  absl::MutexLock lock(&mutex_);

  struct TimedKey {
    InstanceKey key;
    std::chrono::time_point<std::chrono::system_clock> ts;
  };

  std::vector<TimedKey> tmp;
  tmp.reserve(entries_.size());
  for (const auto& e : entries_) {
    tmp.push_back(TimedKey{e.key, e.last_access});
  }

  std::sort(tmp.begin(), tmp.end(), [](const TimedKey& a, const TimedKey& b) { return a.ts < b.ts; });

  std::vector<InstanceKey> ordered;
  ordered.reserve(tmp.size());
  for (const auto& tk : tmp) {
    ordered.push_back(tk.key);
  }
  return ordered;
}

// -----------------------------------------------------------------------------

std::vector<std::pair<InstanceKey, std::shared_ptr<Model>>> ModelRegistry::clear_all() {
  absl::MutexLock lock(&mutex_);

  std::vector<std::pair<InstanceKey, std::shared_ptr<Model>>> removed;
  removed.reserve(entries_.size());

  for (auto& entry : entries_) {
    removed.emplace_back(entry.key, std::move(entry.model));
  }

  // Wipe all indices and storage vectors
  entries_.clear();
  by_instance_.clear();
  by_model_.clear();
  by_device_.clear();

  return removed;
}

size_t ModelRegistry::size() const {
  absl::MutexLock lock(&mutex_);
  return entries_.size();
}

size_t ModelRegistry::get_model_count_by_location(ModelLocation location) const {
  absl::MutexLock lock(&mutex_);
  size_t count = 0;
  for (const auto& entry : entries_) {
    if (entry.model && entry.model->get_memory_state(location) == MemoryState::LOADED) {
      ++count;
    }
  }
  return count;
}

uint64_t ModelRegistry::get_total_model_size() const {
  absl::MutexLock lock(&mutex_);
  uint64_t total = 0;
  for (const auto& entry : entries_) {
    if (!entry.model)
      continue;
    auto size_or = entry.model->get_model_size();
    if (size_or.ok()) {
      total += *size_or;
    }
  }
  return total;
}

} // namespace stepcast::store