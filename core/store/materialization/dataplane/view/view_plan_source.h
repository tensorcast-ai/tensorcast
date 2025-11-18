// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "gsl/pointers"

namespace tensorcast::store::loader {

// ViewPlanSource adapts a canonical SeekableSource according to a SelectionPlan.
// It materializes PAD regions as zero bytes and streams DATA ranges in order,
// allowing callers to consume the variant ByteSpace sequentially.
class ViewPlanSource final : public SeekableSource {
 public:
  ViewPlanSource(gsl::not_null<SeekableSource*> base, SelectionPlan plan);

  [[nodiscard]] bool alias_eligible() const {
    return alias_eligible_;
  }

  [[nodiscard]] bool requires_materialization() const {
    return requires_materialization_;
  }

  [[nodiscard]] uint64_t total_bytes() const {
    return total_bytes_;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override;

 private:
  absl::StatusOr<size_t> copy_from_range(
      const SelectionPlan::Range& range,
      uint64_t range_offset,
      uint8_t* dst,
      size_t bytes) const;

  gsl::not_null<SeekableSource*> base_;
  std::vector<SelectionPlan::Range> ranges_;
  uint64_t total_bytes_{0};
  bool alias_eligible_{false};
  bool requires_materialization_{false};
  uint64_t cursor_{0};
};

std::unique_ptr<SeekableSource> make_view_plan_source(std::unique_ptr<SeekableSource> base, SelectionPlan plan);

} // namespace tensorcast::store::loader
