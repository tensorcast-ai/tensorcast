// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/memory/memory_location.h"
#include "core/store/loader/source.h"

namespace tensorcast::store::loader::verification {

class VerificationMetadataGuard {
  struct Entry;

 public:
  class ScopedLock {
   public:
    ScopedLock() = default;
    ScopedLock(ScopedLock&& other) noexcept;
    ScopedLock& operator=(ScopedLock&& other) noexcept;
    ScopedLock(const ScopedLock&) = delete;
    ScopedLock& operator=(const ScopedLock&) = delete;
    ~ScopedLock();

    [[nodiscard]] absl::Duration wait_duration() const {
      return wait_duration_;
    }

    [[nodiscard]] std::string_view artifact_id() const {
      return artifact_id_;
    }

   private:
    friend class VerificationMetadataGuard;
    ScopedLock(std::string artifact_id, absl::Duration wait_duration, std::shared_ptr<Entry> entry_handle);

    std::string artifact_id_;
    absl::Duration wait_duration_{};
    std::shared_ptr<Entry> entry_handle_;

    void Reset() ABSL_NO_THREAD_SAFETY_ANALYSIS;
  };

  static ScopedLock Acquire(std::string artifact_id) ABSL_NO_THREAD_SAFETY_ANALYSIS;
  static ScopedLock Acquire(std::string artifact_id, absl::Duration warn_after) ABSL_NO_THREAD_SAFETY_ANALYSIS;

  using EntryPtr = std::shared_ptr<Entry>;

 private:
  VerificationMetadataGuard() = default;
};

struct MemoryView {
  common::memory::MemoryLocation location{common::memory::MemoryLocation::CPU};
  void* base_ptr{nullptr};
  uint64_t size_bytes{0};
  std::optional<int> gpu_device_id;
};

struct ViewHashResult {
  std::string multihash;
  std::vector<std::vector<uint8_t>> leaf_digests;
};

absl::StatusOr<std::string> compute_data_multihash(const MemoryView& mem);

absl::StatusOr<ViewHashResult> compute_view_tree_hash_and_leaves(
    loader::SeekableSource& base_source,
    uint64_t total_size,
    size_t leaf_chunk_bytes);

absl::Status reuse_or_generate_verification_json(
    const std::filesystem::path& artifact_dir,
    std::string expected_byte_space_id,
    const MemoryView& mem);

// Test hook to clear in-process verification metadata cache. No-op in
// production code paths.
void ClearVerificationMetadataCacheForTesting();

absl::Status write_descriptor_if_absent(
    const std::filesystem::path& artifact_dir,
    std::string_view index_multihash,
    std::string_view data_multihash,
    uint64_t total_size_bytes,
    std::string_view encoding);

absl::StatusOr<std::vector<std::vector<uint8_t>>> compute_canonical_leaf_digests(
    const MemoryView& mem,
    absl::Span<const uint64_t> leaf_indices,
    size_t chunk_bytes);

} // namespace tensorcast::store::loader::verification
