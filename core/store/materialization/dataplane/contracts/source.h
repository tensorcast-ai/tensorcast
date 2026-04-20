// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/store/replica/types/direct_write_grant.h"

namespace tensorcast::store::loader {

struct DirectWriteOp {
  uint64_t src_offset = 0;
  uint64_t dest_va_offset = 0;
  uint64_t bytes = 0;
};

class Source {
 public:
  virtual ~Source() = default;

  virtual absl::StatusOr<size_t> read(void* dst, size_t max_bytes) = 0;
};

class SeekableSource : public Source {
 public:
  [[nodiscard]] virtual uint64_t total_bytes() const = 0;

  virtual absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) = 0;

  // Optional raw CPU memory pointer for fast-path gather.
  [[nodiscard]] virtual const uint8_t* cpu_base_ptr() const {
    return nullptr;
  }

  // Optional zero-copy capability: direct write into destination address space.
  // Default implementations disable the feature.
  [[nodiscard]] virtual bool supports_direct_write_at() const {
    return false;
  }

  // Optional vectored direct-write capability. Unlike read_into_at(), callers
  // may treat Unimplemented/FailedPrecondition from readv_into_at() as a
  // pre-issue capability miss and fall back before any writes are assumed to
  // have been committed. Sources that inherit the default readv loop must keep
  // this disabled because the loop may have already executed earlier ops when a
  // later op fails.
  [[nodiscard]] virtual bool supports_batched_direct_write_at() const {
    return false;
  }

  virtual absl::StatusOr<size_t> read_into_at(
      [[maybe_unused]] uint64_t src_offset,
      [[maybe_unused]] uint64_t dest_va_offset,
      [[maybe_unused]] size_t bytes,
      const DirectWriteGrant& /*grant*/) {
    return absl::UnimplementedError("direct write not supported");
  }

  virtual absl::StatusOr<size_t> readv_into_at(absl::Span<const DirectWriteOp> ops, const DirectWriteGrant& grant) {
    size_t total_bytes = 0;
    for (const auto& op : ops) {
      if (op.bytes == 0) {
        continue;
      }
      auto wrote_or = read_into_at(op.src_offset, op.dest_va_offset, static_cast<size_t>(op.bytes), grant);
      if (!wrote_or.ok()) {
        return wrote_or.status();
      }
      if (*wrote_or != op.bytes) {
        return absl::DataLossError("direct write batch short write");
      }
      total_bytes += *wrote_or;
    }
    return total_bytes;
  }
};

} // namespace tensorcast::store::loader
