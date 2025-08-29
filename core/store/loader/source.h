// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>

#include "absl/status/statusor.h"
#include "core/store/direct_write.h"

namespace tensorcast::store::loader {

class Source {
 public:
  virtual ~Source() = default;

  virtual absl::StatusOr<size_t> read(void* dst, size_t max_bytes) = 0;
};

class SeekableSource : public Source {
 public:
  virtual absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) = 0;

  // Optional zero-copy capability: direct write into destination address space.
  // Default implementations disable the feature.
  [[nodiscard]] virtual bool supports_direct_write() const {
    return false;
  }
  virtual absl::StatusOr<size_t> read_into(uint64_t dest_va_offset, size_t bytes, const DirectWriteToken& /*token*/) {
    (void)dest_va_offset;
    (void)bytes;
    return absl::UnimplementedError("direct write not supported");
  }
};

} // namespace tensorcast::store::loader
