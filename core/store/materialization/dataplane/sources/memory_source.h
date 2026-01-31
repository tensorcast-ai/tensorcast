// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>

#include "absl/status/statusor.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "gsl/pointers"

namespace tensorcast::store::loader {

class CpuMemorySource final : public SeekableSource {
 public:
  CpuMemorySource(gsl::not_null<const void*> base_ptr, uint64_t total_size);

  [[nodiscard]] uint64_t total_bytes() const override {
    return total_size_;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override;

 private:
  const uint8_t* base_ptr_{nullptr};
  uint64_t total_size_{0};
  uint64_t current_offset_{0};
};

class GpuMemorySource final : public SeekableSource {
 public:
  GpuMemorySource(gsl::not_null<void*> device_ptr, int device_id, uint64_t total_size);

  [[nodiscard]] uint64_t total_bytes() const override {
    return total_size_;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override;

 private:
  gsl::not_null<void*> device_ptr_;
  int device_id_{0};
  uint64_t total_size_{0};
  uint64_t current_offset_{0};
};

} // namespace tensorcast::store::loader
