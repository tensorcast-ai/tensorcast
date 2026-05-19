// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/target_publication_registry.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"

namespace tensorcast::daemon {

PublicationSubjectKey build_publication_subject_key(
    const tensorcast::common::v1::ArtifactSelection& selection,
    const tensorcast::common::v1::ByteSpaceRef& byte_space,
    std::string_view target_layout_hash,
    std::string_view device_uuid) {
  PublicationSubjectKey key{
      .value = absl::StrCat(
          selection.artifact_id(),
          "|",
          selection.view_id(),
          "|",
          selection.logical_layout_hash(),
          "|",
          selection.selection_hash(),
          "|",
          selection.view_subset_hash(),
          "|",
          static_cast<int>(byte_space.kind()),
          "|",
          byte_space.id(),
          "|",
          target_layout_hash,
          "|",
          device_uuid),
  };
  for (const auto& name : selection.tensor_names()) {
    absl::StrAppend(&key.value, "|t:", name);
  }
  return key;
}

TargetPublicationRegistry::TargetPublicationRegistry(Options opts) : opts_(std::move(opts)) {}

TargetPublicationRegistry::Record TargetPublicationRegistry::insert(Record record) {
  absl::MutexLock lock(&mu_);
  const absl::Time now = absl::Now();
  prune_locked(now);
  if (record.subject_generation == 0) {
    record.subject_generation = assign_subject_generation_locked(record);
  }
  if (record.expires_at == absl::InfinitePast()) {
    record.expires_at = now + opts_.ttl;
  }
  latest_by_subject_[record.publication_subject_key.value] = record.publication_id.value;
  records_[record.publication_id.value] = record;
  if (opts_.capacity == 0) {
    return record;
  }
  while (records_.size() > opts_.capacity) {
    auto it = std::find_if(records_.begin(), records_.end(), [](const auto& item) { return !item.second.published; });
    if (it == records_.end()) {
      break;
    }
    const std::string evict_id = it->first;
    const std::string publication_subject_key = it->second.publication_subject_key.value;
    records_.erase(it);
    auto latest_it = latest_by_subject_.find(publication_subject_key);
    if (latest_it != latest_by_subject_.end() && latest_it->second == evict_id) {
      latest_by_subject_.erase(latest_it);
    }
  }
  return record;
}

TargetPublicationRegistry::StagedRecord TargetPublicationRegistry::insert_staged(StagedRecord record) {
  absl::MutexLock lock(&mu_);
  const absl::Time now = absl::Now();
  prune_locked(now);
  if (record.publication.subject_generation == 0) {
    record.publication.subject_generation = assign_subject_generation_locked(record.publication);
  }
  if (record.publication.expires_at == absl::InfinitePast()) {
    record.publication.expires_at = now + opts_.ttl;
  }
  staged_records_[record.publication.publication_id.value] = record;
  if (opts_.capacity == 0) {
    return record;
  }
  while (records_.size() + staged_records_.size() > opts_.capacity) {
    auto it = staged_records_.begin();
    if (it != staged_records_.end()) {
      staged_records_.erase(it);
      continue;
    }
    auto record_it =
        std::find_if(records_.begin(), records_.end(), [](const auto& item) { return !item.second.published; });
    if (record_it == records_.end()) {
      break;
    }
    const std::string evict_id = record_it->first;
    const std::string publication_subject_key = record_it->second.publication_subject_key.value;
    records_.erase(record_it);
    auto latest_it = latest_by_subject_.find(publication_subject_key);
    if (latest_it != latest_by_subject_.end() && latest_it->second == evict_id) {
      latest_by_subject_.erase(latest_it);
    }
  }
  return record;
}

std::optional<TargetPublicationRegistry::StagedRecord> TargetPublicationRegistry::lookup_staged(
    std::string_view publication_id,
    absl::Time now) const {
  absl::MutexLock lock(&mu_);
  auto it = staged_records_.find(std::string(publication_id));
  if (it == staged_records_.end()) {
    return std::nullopt;
  }
  if (it->second.publication.expires_at != absl::InfinitePast() && now > it->second.publication.expires_at) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<TargetPublicationRegistry::StagedRecord> TargetPublicationRegistry::publish_staged(
    std::string_view publication_id) {
  absl::MutexLock lock(&mu_);
  const absl::Time now = absl::Now();
  prune_locked(now);
  auto it = staged_records_.find(std::string(publication_id));
  if (it == staged_records_.end()) {
    return std::nullopt;
  }
  StagedRecord staged = it->second;
  staged_records_.erase(it);
  staged.staging.publish_admitted = true;
  latest_by_subject_[staged.publication.publication_subject_key.value] = staged.publication.publication_id.value;
  records_[staged.publication.publication_id.value] = staged.publication;
  return staged;
}

size_t TargetPublicationRegistry::erase_staged_for_transaction(std::string_view transaction_id, size_t max_to_erase) {
  absl::MutexLock lock(&mu_);
  size_t erased = 0;
  const bool unlimited = max_to_erase == 0;
  for (auto it = staged_records_.begin(); it != staged_records_.end();) {
    if (it->second.staging.transaction_id != transaction_id) {
      ++it;
      continue;
    }
    auto erase_it = it++;
    staged_records_.erase(erase_it);
    erased++;
    if (!unlimited && erased >= max_to_erase) {
      return erased;
    }
  }
  return erased;
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
  if (it->second.terminal) {
    return std::nullopt;
  }
  if (require_not_expired && it->second.expires_at != absl::InfinitePast() && now > it->second.expires_at) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<TargetPublicationRegistry::Record> TargetPublicationRegistry::lookup_current_for_subject(
    std::string_view publication_subject_key,
    absl::Time now,
    bool require_not_expired) const {
  absl::MutexLock lock(&mu_);
  auto it = latest_by_subject_.find(std::string(publication_subject_key));
  if (it == latest_by_subject_.end()) {
    return std::nullopt;
  }
  auto record_it = records_.find(it->second);
  if (record_it == records_.end()) {
    return std::nullopt;
  }
  if (record_it->second.terminal) {
    return std::nullopt;
  }
  if (require_not_expired && record_it->second.expires_at != absl::InfinitePast() &&
      now > record_it->second.expires_at) {
    return std::nullopt;
  }
  return record_it->second;
}

bool TargetPublicationRegistry::is_current_for_subject(
    std::string_view publication_subject_key,
    std::string_view publication_id) const {
  absl::MutexLock lock(&mu_);
  auto it = latest_by_subject_.find(std::string(publication_subject_key));
  if (it == latest_by_subject_.end()) {
    return false;
  }
  return it->second == publication_id;
}

std::optional<TargetPublicationRegistry::Record> TargetPublicationRegistry::mark_published(
    std::string_view publication_id,
    std::string_view operation_id,
    std::string_view lease_id,
    std::string_view replica_id,
    std::uint64_t published_lifecycle_lease_id,
    absl::Time now) {
  absl::MutexLock lock(&mu_);
  auto it = records_.find(std::string(publication_id));
  if (it == records_.end() || it->second.terminal) {
    return std::nullopt;
  }
  it->second.published = true;
  it->second.published_operation_id = std::string(operation_id);
  it->second.published_lease_id = std::string(lease_id);
  it->second.published_replica_id = std::string(replica_id);
  it->second.published_lifecycle_lease_id = published_lifecycle_lease_id;
  it->second.published_at = now;
  return it->second;
}

std::optional<TargetPublicationRegistry::Record> TargetPublicationRegistry::terminalize(
    std::string_view publication_id,
    std::string_view reason) {
  absl::MutexLock lock(&mu_);
  auto it = records_.find(std::string(publication_id));
  if (it == records_.end()) {
    staged_records_.erase(std::string(publication_id));
    return std::nullopt;
  }
  Record record = it->second;
  record.terminal = true;
  record.terminal_reason = std::string(reason);
  record.terminal_at = absl::Now();
  const std::string publication_subject_key = it->second.publication_subject_key.value;
  records_.erase(it);
  auto latest_it = latest_by_subject_.find(publication_subject_key);
  if (latest_it != latest_by_subject_.end() && latest_it->second == publication_id) {
    latest_by_subject_.erase(latest_it);
  }
  return record;
}

void TargetPublicationRegistry::erase(std::string_view publication_id) {
  absl::MutexLock lock(&mu_);
  auto it = records_.find(std::string(publication_id));
  if (it == records_.end()) {
    staged_records_.erase(std::string(publication_id));
    return;
  }
  const std::string publication_subject_key = it->second.publication_subject_key.value;
  records_.erase(it);
  auto latest_it = latest_by_subject_.find(publication_subject_key);
  if (latest_it != latest_by_subject_.end() && latest_it->second == publication_id) {
    latest_by_subject_.erase(latest_it);
  }
}

void TargetPublicationRegistry::prune(absl::Time now) {
  absl::MutexLock lock(&mu_);
  prune_locked(now);
}

void TargetPublicationRegistry::prune_locked(absl::Time now) {
  if (records_.empty() && staged_records_.empty()) {
    return;
  }
  std::vector<std::string> expired;
  expired.reserve(records_.size());
  for (const auto& [key, record] : records_) {
    if (!record.published && record.expires_at != absl::InfinitePast() && now > record.expires_at) {
      expired.push_back(key);
    }
  }
  for (const auto& key : expired) {
    auto it = records_.find(key);
    if (it == records_.end()) {
      continue;
    }
    const std::string publication_subject_key = it->second.publication_subject_key.value;
    records_.erase(it);
    auto latest_it = latest_by_subject_.find(publication_subject_key);
    if (latest_it != latest_by_subject_.end() && latest_it->second == key) {
      latest_by_subject_.erase(latest_it);
    }
  }

  std::vector<std::string> expired_staged;
  expired_staged.reserve(staged_records_.size());
  for (const auto& [key, record] : staged_records_) {
    if (record.publication.expires_at != absl::InfinitePast() && now > record.publication.expires_at) {
      expired_staged.push_back(key);
    }
  }
  for (const auto& key : expired_staged) {
    staged_records_.erase(key);
  }
}

std::uint64_t TargetPublicationRegistry::assign_subject_generation_locked(const Record& record) const {
  if (record.publication_id.value.empty()) {
    return 1;
  }
  auto current_it = latest_by_subject_.find(record.publication_subject_key.value);
  if (current_it != latest_by_subject_.end()) {
    auto record_it = records_.find(current_it->second);
    if (record_it != records_.end()) {
      if (record_it->second.publication_id == record.publication_id) {
        return record_it->second.subject_generation == 0 ? 1 : record_it->second.subject_generation;
      }
      const std::uint64_t current_generation =
          record_it->second.subject_generation == 0 ? 1 : record_it->second.subject_generation;
      return record_it->second.owner_pid == record.owner_pid ? current_generation : current_generation + 1;
    }
  }

  std::uint64_t max_generation = 0;
  std::optional<int> max_generation_owner_pid;
  for (const auto& [publication_key, existing] : records_) {
    (void)publication_key;
    if (existing.publication_subject_key != record.publication_subject_key) {
      continue;
    }
    const std::uint64_t existing_generation = existing.subject_generation == 0 ? 1 : existing.subject_generation;
    if (existing_generation > max_generation) {
      max_generation = existing_generation;
      max_generation_owner_pid = existing.owner_pid;
    }
  }
  if (max_generation == 0) {
    return 1;
  }
  if (max_generation_owner_pid.has_value() && *max_generation_owner_pid != record.owner_pid) {
    return max_generation + 1;
  }
  return max_generation;
}

} // namespace tensorcast::daemon
