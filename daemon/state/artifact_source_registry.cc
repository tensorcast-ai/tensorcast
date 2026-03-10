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

} // namespace tensorcast::daemon
