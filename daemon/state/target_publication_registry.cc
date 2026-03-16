// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/target_publication_registry.h"

#include <utility>
#include <vector>

namespace tensorcast::daemon {

TargetPublicationRegistry::TargetPublicationRegistry(Options opts) : opts_(std::move(opts)) {}

TargetPublicationRegistry::Record TargetPublicationRegistry::insert(Record record) {
  absl::MutexLock lock(&mu_);
  const absl::Time now = absl::Now();
  prune_locked(now);
  if (record.expires_at == absl::InfinitePast()) {
    record.expires_at = now + opts_.ttl;
  }
  latest_by_target_[record.publication_key] = record.publication_id;
  records_[record.publication_id] = record;
  if (opts_.capacity == 0) {
    return record;
  }
  while (records_.size() > opts_.capacity) {
    auto it = records_.begin();
    if (it == records_.end()) {
      break;
    }
    const std::string evict_id = it->first;
    const std::string publication_key = it->second.publication_key;
    records_.erase(it);
    auto latest_it = latest_by_target_.find(publication_key);
    if (latest_it != latest_by_target_.end() && latest_it->second == evict_id) {
      latest_by_target_.erase(latest_it);
    }
  }
  return record;
}

std::optional<TargetPublicationRegistry::Record> TargetPublicationRegistry::lookup(
    std::string_view publication_id,
    absl::Time now,
    bool require_not_expired) const {
  absl::MutexLock lock(&mu_);
  auto it = records_.find(std::string(publication_id));
  if (it == records_.end()) {
    return std::nullopt;
  }
  if (require_not_expired && it->second.expires_at != absl::InfinitePast() && now > it->second.expires_at) {
    return std::nullopt;
  }
  return it->second;
}

bool TargetPublicationRegistry::is_current_for_target(std::string_view publication_key, std::string_view publication_id)
    const {
  absl::MutexLock lock(&mu_);
  auto it = latest_by_target_.find(std::string(publication_key));
  if (it == latest_by_target_.end()) {
    return false;
  }
  return it->second == publication_id;
}

void TargetPublicationRegistry::erase(std::string_view publication_id) {
  absl::MutexLock lock(&mu_);
  auto it = records_.find(std::string(publication_id));
  if (it == records_.end()) {
    return;
  }
  const std::string publication_key = it->second.publication_key;
  records_.erase(it);
  auto latest_it = latest_by_target_.find(publication_key);
  if (latest_it != latest_by_target_.end() && latest_it->second == publication_id) {
    latest_by_target_.erase(latest_it);
  }
}

void TargetPublicationRegistry::prune(absl::Time now) {
  absl::MutexLock lock(&mu_);
  prune_locked(now);
}

void TargetPublicationRegistry::prune_locked(absl::Time now) {
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
    const std::string publication_key = it->second.publication_key;
    records_.erase(it);
    auto latest_it = latest_by_target_.find(publication_key);
    if (latest_it != latest_by_target_.end() && latest_it->second == key) {
      latest_by_target_.erase(latest_it);
    }
  }
}

} // namespace tensorcast::daemon
