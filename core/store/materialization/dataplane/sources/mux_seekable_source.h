// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include "gsl/pointers"

#include "absl/status/statusor.h"
#include "core/store/materialization/dataplane/contracts/source.h"

namespace tensorcast::store::loader {

// Tries primary read first; on short read or error, falls back to secondary.
class MuxSeekableSource : public SeekableSource {
 public:
  MuxSeekableSource(
      gsl::not_null<std::shared_ptr<SeekableSource>> primary,
      gsl::not_null<std::shared_ptr<SeekableSource>> fallback);
  ~MuxSeekableSource() override = default;

  [[nodiscard]] uint64_t total_bytes() const override;

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override;

  [[nodiscard]] bool supports_direct_write_at() const override;
  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const DirectWriteGrant& grant) override;

 private:
  gsl::not_null<std::shared_ptr<SeekableSource>> primary_;
  gsl::not_null<std::shared_ptr<SeekableSource>> fallback_;
  uint64_t current_offset_ = 0;
};

} // namespace tensorcast::store::loader
