// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace tensorcast::daemon {

class LocalDiskImportCatalog {
 public:
  struct Entry {
    std::string normalized_disk_path;
    bool descriptor_present{false};
    std::optional<std::string> index_multihash;
    std::optional<std::string> data_multihash;
    std::optional<uint64_t> generation;
    absl::Time created_at{absl::Now()};
  };

  void upsert_import(std::string artifact_id, Entry entry);
  [[nodiscard]] std::optional<Entry> lookup_import(std::string_view artifact_id) const;

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, Entry> entries_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
