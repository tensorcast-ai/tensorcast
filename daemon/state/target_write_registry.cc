// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/target_write_registry.h"

#include <utility>
#include <vector>

namespace tensorcast::daemon {

TargetWriteRegistry::TargetWriteRegistry(Options opts) : opts_(std::move(opts)) {}

TargetWriteRegistry::Record TargetWriteRegistry::insert(Record record) {
  absl::MutexLock lock(&mu_);
  const absl::Time now = absl::Now();
  prune_locked(now);
  if (record.expires_at == absl::InfinitePast()) {
    record.expires_at = now + opts_.ttl;
  }
  latest_by_layout_[record.layout_key] = record.write_id;
  records_[record.write_id] = record;
  if (opts_.capacity == 0) {
    return record;
  }
  while (records_.size() > opts_.capacity) {
    auto it = records_.begin();
    if (it == records_.end()) {
      break;
    }
    const std::string evict_id = it->first;
    const std::string layout_key = it->second.layout_key;
    records_.erase(it);
    auto latest_it = latest_by_layout_.find(layout_key);
    if (latest_it != latest_by_layout_.end() && latest_it->second == evict_id) {
      latest_by_layout_.erase(latest_it);
    }
  }
  return record;
}

std::optional<TargetWriteRegistry::Record> TargetWriteRegistry::lookup(
    std::string_view write_id,
    absl::Time now,
    bool require_not_expired) const {
  absl::MutexLock lock(&mu_);
  auto it = records_.find(std::string(write_id));
  if (it == records_.end()) {
    return std::nullopt;
  }
  if (require_not_expired && it->second.expires_at != absl::InfinitePast() && now > it->second.expires_at) {
    return std::nullopt;
  }
  return it->second;
}

bool TargetWriteRegistry::is_current_for_layout(std::string_view layout_key, std::string_view write_id) const {
  absl::MutexLock lock(&mu_);
  auto it = latest_by_layout_.find(std::string(layout_key));
  if (it == latest_by_layout_.end()) {
    return false;
  }
  return it->second == write_id;
}

void TargetWriteRegistry::erase(std::string_view write_id) {
  absl::MutexLock lock(&mu_);
  auto it = records_.find(std::string(write_id));
  if (it == records_.end()) {
    return;
  }
  const std::string layout_key = it->second.layout_key;
  records_.erase(it);
  auto latest_it = latest_by_layout_.find(layout_key);
  if (latest_it != latest_by_layout_.end() && latest_it->second == write_id) {
    latest_by_layout_.erase(latest_it);
  }
}

void TargetWriteRegistry::prune(absl::Time now) {
  absl::MutexLock lock(&mu_);
  prune_locked(now);
}

void TargetWriteRegistry::prune_locked(absl::Time now) {
  if (records_.empty()) {
    return;
  }
  std::vector<std::string> expired;
  expired.reserve(records_.size());
  for (const auto& [key, record] : records_) {
    if (record.expires_at != absl::InfinitePast() && now > record.expires_at) {
      expired.push_back(key);
    }
  }
  for (const auto& key : expired) {
    auto it = records_.find(key);
    if (it == records_.end()) {
      continue;
    }
    const std::string layout_key = it->second.layout_key;
    records_.erase(it);
    auto latest_it = latest_by_layout_.find(layout_key);
    if (latest_it != latest_by_layout_.end() && latest_it->second == key) {
      latest_by_layout_.erase(latest_it);
    }
  }
}

} // namespace tensorcast::daemon
