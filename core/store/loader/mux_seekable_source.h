// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "core/store/loader/source.h"

namespace tensorcast::store::loader {

// Tries primary read first; on short read or error, falls back to secondary.
class MuxSeekableSource : public SeekableSource {
 public:
  MuxSeekableSource(std::shared_ptr<SeekableSource> primary, std::shared_ptr<SeekableSource> fallback);
  ~MuxSeekableSource() override = default;

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override;

 private:
  std::shared_ptr<SeekableSource> primary_;
  std::shared_ptr<SeekableSource> fallback_;
  uint64_t current_offset_ = 0;
};

} // namespace tensorcast::store::loader
