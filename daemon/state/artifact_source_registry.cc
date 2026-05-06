// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/artifact_source_registry.h"

#include <utility>

namespace tensorcast::daemon {

void ArtifactSourceRegistry::upsert_binding(std::string artifact_id, Entry entry) {
  absl::MutexLock lock(&mu_);
  auto now = absl::Now();
  entry.updated_at = now;
  auto it = entries_.find(artifact_id);
  if (it == entries_.end()) {
    if (entry.created_at == absl::UnixEpoch()) {
      entry.created_at = now;
    }
    entries_.emplace(std::move(artifact_id), std::move(entry));
    return;
  }
  entry.created_at = it->second.created_at;
  it->second = std::move(entry);
}

std::optional<ArtifactSourceRegistry::Entry> ArtifactSourceRegistry::lookup_binding(
    std::string_view artifact_id) const {
  absl::MutexLock lock(&mu_);
  auto it = entries_.find(artifact_id);
  if (it == entries_.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool ArtifactSourceRegistry::erase_binding(std::string_view artifact_id) {
  absl::MutexLock lock(&mu_);
  return entries_.erase(std::string(artifact_id)) > 0;
}

bool ArtifactSourceRegistry::note_promoted_artifact_for_binding(
    std::string_view artifact_id,
    std::string_view promoted_artifact_id,
    PromotionOrigin origin) {
  if (artifact_id.empty() || promoted_artifact_id.empty()) {
    return false;
  }
  absl::MutexLock lock(&mu_);
  auto it = entries_.find(std::string(artifact_id));
  if (it == entries_.end()) {
    return false;
  }
  it->second.promoted_content_artifact_id = std::string(promoted_artifact_id);
  it->second.promoted_content_origin = origin;
  it->second.updated_at = absl::Now();
  return true;
}

size_t ArtifactSourceRegistry::note_promoted_artifact_for_source(
    std::string_view canonical_source_path,
    const FingerprintMap& file_fingerprints,
    std::string_view promoted_artifact_id,
    PromotionOrigin origin) {
  if (canonical_source_path.empty() || promoted_artifact_id.empty()) {
    return 0;
  }
  absl::MutexLock lock(&mu_);
  size_t updated = 0;
  const auto now = absl::Now();
  for (auto& [artifact_id, entry] : entries_) {
    (void)artifact_id;
    if (entry.source_kind != SourceKind::kMountedSourceArtifact) {
      continue;
    }
    if (entry.canonical_source_path != canonical_source_path) {
      continue;
    }
    if (entry.file_fingerprints != file_fingerprints) {
      continue;
    }
    entry.promoted_content_artifact_id = std::string(promoted_artifact_id);
    entry.promoted_content_origin = origin;
    entry.updated_at = now;
    ++updated;
  }
  return updated;
}

} // namespace tensorcast::daemon
