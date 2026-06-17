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
    kMountedSourceArtifact = 2,
  };

  enum class PromotionOrigin : std::uint8_t {
    kUnspecified = 0,
    kImport = 1,
    kMountedVerify = 2,
    kBindingSeal = 3,
  };

  enum class SourceFormatKind : std::uint8_t {
    kUnspecified = 0,
    kPartitioned = 1,
    kSafetensors = 2,
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
    std::string canonical_index_json;
    std::optional<std::string> source_index_json;
    std::optional<std::string> source_disk_path;
    SourceFormatKind source_format_kind{SourceFormatKind::kUnspecified};
    bool descriptor_present{false};
    std::optional<std::string> index_multihash;
    std::optional<std::string> data_multihash;
    std::optional<std::string> trusted_content_artifact_id;
    std::optional<std::string> promoted_content_artifact_id;
    std::optional<PromotionOrigin> promoted_content_origin;
    std::optional<std::string> policy_id;
    std::optional<std::string> snapshot_digest;
    std::optional<std::uint64_t> generation;
    bool tensor_aware_metadata{true};
    bool validate_before_read{false};
    FingerprintMap file_fingerprints;
    absl::Time created_at{absl::Now()};
    absl::Time updated_at{absl::Now()};
  };

  void upsert_binding(std::string artifact_id, Entry entry);
  [[nodiscard]] std::optional<Entry> lookup_binding(std::string_view artifact_id) const;
  [[nodiscard]] bool erase_binding(std::string_view artifact_id);
  [[nodiscard]] bool note_promoted_artifact_for_binding(
      std::string_view artifact_id,
      std::string_view promoted_artifact_id,
      PromotionOrigin origin = PromotionOrigin::kBindingSeal);
  [[nodiscard]] size_t note_promoted_artifact_for_source(
      std::string_view canonical_source_path,
      const FingerprintMap& file_fingerprints,
      std::string_view promoted_artifact_id,
      PromotionOrigin origin = PromotionOrigin::kImport);

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, Entry> entries_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
