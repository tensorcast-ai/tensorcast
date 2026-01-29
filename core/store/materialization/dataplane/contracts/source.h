// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>

#include "absl/status/statusor.h"
#include "core/store/replica/types/direct_write_grant.h"

namespace tensorcast::store::loader {

class Source {
 public:
  virtual ~Source() = default;

  virtual absl::StatusOr<size_t> read(void* dst, size_t max_bytes) = 0;
};

class SeekableSource : public Source {
 public:
  [[nodiscard]] virtual uint64_t total_bytes() const = 0;

  virtual absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) = 0;

  // Optional zero-copy capability: direct write into destination address space.
  // Default implementations disable the feature.
  [[nodiscard]] virtual bool supports_direct_write_at() const {
    return false;
  }

  virtual absl::StatusOr<size_t> read_into_at(
      [[maybe_unused]] uint64_t src_offset,
      [[maybe_unused]] uint64_t dest_va_offset,
      [[maybe_unused]] size_t bytes,
      const DirectWriteGrant& /*grant*/) {
    return absl::UnimplementedError("direct write not supported");
  }
};

} // namespace tensorcast::store::loader
