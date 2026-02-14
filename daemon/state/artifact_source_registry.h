// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace tensorcast::daemon {

class ArtifactSourceRegistry {
 public:
  enum class SourceKind : std::uint8_t {
    kManagedSharedDisk = 0,
    kLocalImport = 1,
  };

  struct SourceFileFingerprint {
    std::uint64_t inode{0};
    std::uint64_t size{0};
    std::int64_t mtime_ns{0};

    bool operator==(const SourceFileFingerprint&) const = default;
  };

  using FingerprintMap = absl::flat_hash_map<std::string, SourceFileFingerprint>;

  struct Entry {
    SourceKind source_kind{SourceKind::kLocalImport};
    std::string canonical_source_path;
    std::optional<std::string> source_disk_path;
    bool descriptor_present{false};
    std::optional<std::string> index_multihash;
    std::optional<std::string> data_multihash;
    std::optional<std::uint64_t> generation;
    FingerprintMap file_fingerprints;
    absl::Time created_at{absl::Now()};
    absl::Time updated_at{absl::Now()};
  };

  void upsert_binding(std::string artifact_id, Entry entry);
  [[nodiscard]] std::optional<Entry> lookup_binding(std::string_view artifact_id) const;
  [[nodiscard]] bool erase_binding(std::string_view artifact_id);

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, Entry> entries_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
