// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <atomic>
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
  ~ViewPlanSource() override;

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
  struct ExecutionRun {
    enum class Kind : uint8_t { kPad = 0, kContiguous = 1, kStrided = 2 };

    Kind kind{Kind::kPad};
    uint64_t dst_begin{0};
    uint64_t dst_end{0};
    uint64_t src_begin{0};
    uint64_t src_base{0};
    uint64_t row_len{0};
    uint64_t stride{0};
    uint64_t rows{0};
    uint64_t rows_per_block{0};
    size_t range_index{0};
    size_t range_count{0};
    bool strided_candidate{false};
  };

  struct Stats {
    std::atomic<uint64_t> base_read_calls{0};
    std::atomic<uint64_t> base_read_bytes{0};
    std::atomic<uint64_t> output_bytes{0};
    std::atomic<uint64_t> pack_bytes{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> strided_runtime_fallbacks{0};
  };

  struct StridedBlockCache;

  void build_runs();
  [[nodiscard]] size_t find_run_index(uint64_t offset) const;
  static bool should_enable_strided(const ExecutionRun& run);
  static uint64_t compute_rows_per_block(const ExecutionRun& run);

  absl::StatusOr<size_t> copy_from_strided_ranges(
      const ExecutionRun& run,
      uint64_t run_offset,
      uint8_t* dst,
      size_t bytes);

  absl::StatusOr<size_t> fill_strided_run(
      size_t run_index,
      const ExecutionRun& run,
      uint64_t run_offset,
      uint8_t* dst,
      size_t bytes);

  absl::StatusOr<size_t> read_base(uint64_t offset, uint8_t* dst, size_t bytes);

  absl::StatusOr<size_t> copy_from_range(
      const SelectionPlan::Range& range,
      uint64_t range_offset,
      uint8_t* dst,
      size_t bytes);

  gsl::not_null<SeekableSource*> base_;
  std::vector<SelectionPlan::Range> ranges_;
  std::vector<ExecutionRun> runs_;
  std::vector<uint64_t> run_starts_;
  std::unique_ptr<std::atomic<uint8_t>[]> strided_disabled_;
  std::unique_ptr<StridedBlockCache> strided_cache_;
  Stats stats_;
  uint64_t strided_runs_total_{0};
  uint64_t strided_fallback_runs_total_{0};
  uint64_t total_bytes_{0};
  bool alias_eligible_{false};
  bool requires_materialization_{false};
  uint64_t cursor_{0};
};

std::unique_ptr<SeekableSource> make_view_plan_source(std::unique_ptr<SeekableSource> base, SelectionPlan plan);

} // namespace tensorcast::store::loader
