// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/local_disk_import_catalog.h"

#include <utility>

namespace tensorcast::daemon {

void LocalDiskImportCatalog::upsert_import(std::string artifact_id, Entry entry) {
  absl::MutexLock lock(&mu_);
  entries_.insert_or_assign(std::move(artifact_id), std::move(entry));
}

std::optional<LocalDiskImportCatalog::Entry> LocalDiskImportCatalog::lookup_import(std::string_view artifact_id) const {
  absl::MutexLock lock(&mu_);
  auto it = entries_.find(artifact_id);
  if (it == entries_.end()) {
    return std::nullopt;
  }
  return it->second;
}

} // namespace tensorcast::daemon
