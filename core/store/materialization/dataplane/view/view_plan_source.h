// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/sources/byte_range_mapped_source.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/store_engine_options.h"
#include "gsl/pointers"

namespace tensorcast::store::loader {

// ViewPlanSource adapts a canonical SeekableSource according to a SelectionPlan.
// It materializes PAD regions as zero bytes and streams DATA ranges in order,
// allowing callers to consume the variant ByteSpace sequentially.
class ViewPlanSource final : public SeekableSource {
 public:
  ViewPlanSource(
      gsl::not_null<SeekableSource*> base,
      SelectionPlan plan,
      StoreEngineOptions::ByteMappingConfig config = {});
  ~ViewPlanSource() override = default;

  [[nodiscard]] bool alias_eligible() const {
    return alias_eligible_;
  }

  [[nodiscard]] bool requires_materialization() const {
    return requires_materialization_;
  }

  [[nodiscard]] uint64_t total_bytes() const override {
    return total_bytes_;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override;
  [[nodiscard]] const uint8_t* cpu_base_ptr() const override;
  [[nodiscard]] bool supports_direct_write_at() const override;
  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const DirectWriteGrant& grant) override;
  absl::StatusOr<size_t> readv_into_at(absl::Span<const DirectWriteOp> ops, const DirectWriteGrant& grant) override;

 private:
  gsl::not_null<SeekableSource*> base_;
  std::unique_ptr<ByteRangeMappedSource> mapped_source_;
  uint64_t total_bytes_{0};
  bool alias_eligible_{false};
  bool requires_materialization_{false};
};

std::unique_ptr<SeekableSource> make_view_plan_source(
    std::unique_ptr<SeekableSource> base,
    SelectionPlan plan,
    StoreEngineOptions::ByteMappingConfig config = {});

} // namespace tensorcast::store::loader
