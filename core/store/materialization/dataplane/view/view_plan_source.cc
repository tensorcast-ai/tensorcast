// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/view/view_plan_source.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::loader {

namespace {

constexpr uint64_t STRIDED_RUN_MIN_RANGES = 128;
constexpr uint64_t STRIDED_MIN_ROW_LEN_BYTES = 4096;
constexpr uint64_t STRIDED_MAX_AMPLIFICATION = 8;
constexpr uint64_t STRIDED_BLOCK_TARGET_BYTES = 16ULL * 1024 * 1024;
constexpr uint64_t STRIDED_BLOCK_MAX_BYTES = 64ULL * 1024 * 1024;

struct StridedBlock {
  size_t run_index{0};
  uint64_t first_row{0};
  uint64_t rows{0};
  uint64_t src_begin{0};
  std::vector<uint8_t> data;
};

struct MetricsHandles {
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> meter;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> base_read_calls;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> base_read_bytes;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> output_bytes;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> pack_bytes;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> strided_runs;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> strided_fallback_runs;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> cache_hits;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> cache_misses;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> amplification_ratio;
};

MetricsHandles& metrics_handles() {
  static MetricsHandles handles;
  if (!handles.meter) {
    handles.meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  }
  if (!handles.base_read_calls) {
    handles.base_read_calls = handles.meter->CreateDoubleCounter("tc_view_plan_source_base_read_calls_total");
    handles.base_read_bytes = handles.meter->CreateDoubleCounter("tc_view_plan_source_base_read_bytes_total");
    handles.output_bytes = handles.meter->CreateDoubleCounter("tc_view_plan_source_output_bytes_total");
    handles.pack_bytes = handles.meter->CreateDoubleCounter("tc_view_plan_source_pack_bytes_total");
    handles.strided_runs = handles.meter->CreateDoubleCounter("tc_view_plan_source_strided_runs_total");
    handles.strided_fallback_runs =
        handles.meter->CreateDoubleCounter("tc_view_plan_source_strided_fallback_runs_total");
    handles.cache_hits = handles.meter->CreateDoubleCounter("tc_view_plan_source_strided_cache_hits_total");
    handles.cache_misses = handles.meter->CreateDoubleCounter("tc_view_plan_source_strided_cache_misses_total");
    handles.amplification_ratio = handles.meter->CreateDoubleHistogram("tc_view_plan_source_amplification_ratio");
  }
  return handles;
}

void record_view_plan_source_metrics(
    uint64_t base_read_calls,
    uint64_t base_read_bytes,
    uint64_t output_bytes,
    uint64_t pack_bytes,
    uint64_t strided_runs,
    uint64_t strided_fallback_runs,
    uint64_t cache_hits,
    uint64_t cache_misses,
    double amplification_ratio) {
  try {
    static const std::map<std::string, opentelemetry::common::AttributeValue> kEmptyAttrs;
    auto& handles = metrics_handles();
    handles.base_read_calls->Add(
        static_cast<double>(base_read_calls),
        opentelemetry::common::KeyValueIterableView(kEmptyAttrs),
        opentelemetry::context::Context{});
    handles.base_read_bytes->Add(
        static_cast<double>(base_read_bytes),
        opentelemetry::common::KeyValueIterableView(kEmptyAttrs),
        opentelemetry::context::Context{});
    handles.output_bytes->Add(
        static_cast<double>(output_bytes),
        opentelemetry::common::KeyValueIterableView(kEmptyAttrs),
        opentelemetry::context::Context{});
    handles.pack_bytes->Add(
        static_cast<double>(pack_bytes),
        opentelemetry::common::KeyValueIterableView(kEmptyAttrs),
        opentelemetry::context::Context{});
    handles.strided_runs->Add(
        static_cast<double>(strided_runs),
        opentelemetry::common::KeyValueIterableView(kEmptyAttrs),
        opentelemetry::context::Context{});
    handles.strided_fallback_runs->Add(
        static_cast<double>(strided_fallback_runs),
        opentelemetry::common::KeyValueIterableView(kEmptyAttrs),
        opentelemetry::context::Context{});
    handles.cache_hits->Add(
        static_cast<double>(cache_hits),
        opentelemetry::common::KeyValueIterableView(kEmptyAttrs),
        opentelemetry::context::Context{});
    handles.cache_misses->Add(
        static_cast<double>(cache_misses),
        opentelemetry::common::KeyValueIterableView(kEmptyAttrs),
        opentelemetry::context::Context{});
    handles.amplification_ratio->Record(
        amplification_ratio,
        opentelemetry::common::KeyValueIterableView(kEmptyAttrs),
        opentelemetry::context::Context{});
  } catch (...) {
    VLOG(1) << "metrics counters tc_view_plan_source_* unavailable";
  }
}

bool amplification_within_limit(uint64_t stride, uint64_t row_len) {
  if (row_len == 0) {
    return false;
  }
  if (STRIDED_MAX_AMPLIFICATION == 0) {
    return false;
  }
  if (row_len > std::numeric_limits<uint64_t>::max() / STRIDED_MAX_AMPLIFICATION) {
    return false;
  }
  return stride <= STRIDED_MAX_AMPLIFICATION * row_len;
}

} // namespace

struct ViewPlanSource::StridedBlockCache {
  absl::Mutex mutex;
  std::shared_ptr<const StridedBlock> block ABSL_GUARDED_BY(mutex);
};

namespace {

class OwningViewPlanSource final : public SeekableSource {
 public:
  OwningViewPlanSource(std::unique_ptr<SeekableSource> base, SelectionPlan plan)
      : base_(std::move(base)), adapter_(gsl::not_null<SeekableSource*>{base_.get()}, std::move(plan)) {}

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    return adapter_.read(dst, max_bytes);
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    return adapter_.read_at(offset, dst, bytes);
  }

  [[nodiscard]] bool supports_direct_write() const override {
    return false;
  }

  absl::StatusOr<size_t> read_into(uint64_t, size_t, const DirectWriteGrant&) override {
    return absl::UnimplementedError("direct write not supported for view plan sources");
  }

 private:
  std::unique_ptr<SeekableSource> base_;
  ViewPlanSource adapter_;
};

} // namespace

ViewPlanSource::ViewPlanSource(gsl::not_null<SeekableSource*> base, SelectionPlan plan)
    : base_(base),
      ranges_(plan.ranges.begin(), plan.ranges.end()),
      total_bytes_(plan.total_bytes),
      alias_eligible_(plan.is_segment_aligned),
      requires_materialization_(plan.requires_materialization) {
  std::sort(ranges_.begin(), ranges_.end(), [](const SelectionPlan::Range& a, const SelectionPlan::Range& b) {
    return a.dst_offset < b.dst_offset;
  });
  build_runs();
  if (!runs_.empty()) {
    strided_disabled_ = std::make_unique<std::atomic<uint8_t>[]>(runs_.size());
    for (size_t idx = 0; idx < runs_.size(); ++idx) {
      strided_disabled_[idx].store(0, std::memory_order_relaxed);
    }
  }
  if (strided_runs_total_ > 0) {
    strided_cache_ = std::make_unique<StridedBlockCache>();
  }
}

ViewPlanSource::~ViewPlanSource() {
  const uint64_t base_read_calls = stats_.base_read_calls.load(std::memory_order_relaxed);
  const uint64_t base_read_bytes = stats_.base_read_bytes.load(std::memory_order_relaxed);
  const uint64_t output_bytes = stats_.output_bytes.load(std::memory_order_relaxed);
  const uint64_t pack_bytes = stats_.pack_bytes.load(std::memory_order_relaxed);
  const uint64_t cache_hits = stats_.cache_hits.load(std::memory_order_relaxed);
  const uint64_t cache_misses = stats_.cache_misses.load(std::memory_order_relaxed);
  const uint64_t runtime_fallbacks = stats_.strided_runtime_fallbacks.load(std::memory_order_relaxed);
  const uint64_t strided_fallback_runs = strided_fallback_runs_total_ + runtime_fallbacks;
  const double amplification =
      output_bytes == 0 ? 0.0 : static_cast<double>(base_read_bytes) / static_cast<double>(output_bytes);

  VLOG(1) << "ViewPlanSource summary: output_bytes=" << output_bytes << " base_read_calls=" << base_read_calls
          << " base_read_bytes=" << base_read_bytes << " pack_bytes=" << pack_bytes
          << " strided_runs=" << strided_runs_total_ << " strided_fallback_runs=" << strided_fallback_runs
          << " cache_hits=" << cache_hits << " cache_misses=" << cache_misses << " amplification=" << amplification;

  record_view_plan_source_metrics(
      base_read_calls,
      base_read_bytes,
      output_bytes,
      pack_bytes,
      strided_runs_total_,
      strided_fallback_runs,
      cache_hits,
      cache_misses,
      amplification);
}

bool ViewPlanSource::should_enable_strided(const ExecutionRun& run) {
  if (run.rows < STRIDED_RUN_MIN_RANGES) {
    return false;
  }
  if (run.row_len < STRIDED_MIN_ROW_LEN_BYTES) {
    return false;
  }
  if (run.row_len > STRIDED_BLOCK_MAX_BYTES) {
    return false;
  }
  return amplification_within_limit(run.stride, run.row_len);
}

uint64_t ViewPlanSource::compute_rows_per_block(const ExecutionRun& run) {
  if (run.stride == 0) {
    return 1;
  }
  uint64_t rows_per_block = STRIDED_BLOCK_TARGET_BYTES / run.stride;
  if (rows_per_block == 0) {
    rows_per_block = 1;
  }
  uint64_t max_rows = 1;
  if (STRIDED_BLOCK_MAX_BYTES > run.row_len) {
    max_rows = 1 + (STRIDED_BLOCK_MAX_BYTES - run.row_len) / run.stride;
  }
  if (rows_per_block > max_rows) {
    rows_per_block = max_rows;
  }
  return std::max<uint64_t>(rows_per_block, 1);
}

void ViewPlanSource::build_runs() {
  runs_.clear();
  run_starts_.clear();
  strided_runs_total_ = 0;
  strided_fallback_runs_total_ = 0;

  if (ranges_.empty()) {
    return;
  }

  runs_.reserve(ranges_.size());
  size_t idx = 0;
  while (idx < ranges_.size()) {
    const auto& range = ranges_[idx];
    if (range.length == 0) {
      ++idx;
      continue;
    }
    if (range.kind == SelectionPlan::Range::Kind::kPad) {
      size_t start = idx;
      uint64_t dst_begin = range.dst_offset;
      uint64_t dst_end = dst_begin + range.length;
      ++idx;
      while (idx < ranges_.size()) {
        const auto& next = ranges_[idx];
        if (next.kind != SelectionPlan::Range::Kind::kPad) {
          break;
        }
        if (next.length == 0 || next.dst_offset != dst_end) {
          break;
        }
        dst_end += next.length;
        ++idx;
      }
      ExecutionRun run;
      run.kind = ExecutionRun::Kind::kPad;
      run.dst_begin = dst_begin;
      run.dst_end = dst_end;
      run.range_index = start;
      run.range_count = idx - start;
      runs_.push_back(run);
      continue;
    }

    bool strided_candidate = false;
    uint64_t stride = 0;
    if (idx + 1 < ranges_.size()) {
      const auto& next = ranges_[idx + 1];
      if (next.kind == SelectionPlan::Range::Kind::kData && next.length == range.length &&
          next.dst_offset == range.dst_offset + range.length && next.src_offset > range.src_offset) {
        stride = next.src_offset - range.src_offset;
        strided_candidate = stride > range.length;
      }
    }

    if (strided_candidate) {
      const size_t start = idx;
      const uint64_t row_len = range.length;
      const uint64_t dst_begin = range.dst_offset;
      const uint64_t src_base = range.src_offset;
      size_t cursor = idx;
      uint64_t expected_dst = dst_begin;
      uint64_t expected_src = src_base;
      while (cursor < ranges_.size()) {
        const auto& current = ranges_[cursor];
        if (current.kind != SelectionPlan::Range::Kind::kData || current.length != row_len ||
            current.dst_offset != expected_dst || current.src_offset != expected_src) {
          break;
        }
        expected_dst += row_len;
        expected_src += stride;
        ++cursor;
      }
      const uint64_t rows = static_cast<uint64_t>(cursor - start);
      ExecutionRun run;
      run.kind = ExecutionRun::Kind::kStrided;
      run.dst_begin = dst_begin;
      run.dst_end = dst_begin + (row_len * rows);
      run.src_base = src_base;
      run.row_len = row_len;
      run.stride = stride;
      run.rows = rows;
      run.range_index = start;
      run.range_count = cursor - start;
      run.strided_candidate = should_enable_strided(run);
      if (run.strided_candidate) {
        run.rows_per_block = compute_rows_per_block(run);
      } else {
        ++strided_fallback_runs_total_;
      }
      ++strided_runs_total_;
      runs_.push_back(run);
      idx = cursor;
      continue;
    }

    size_t start = idx;
    uint64_t dst_begin = range.dst_offset;
    uint64_t dst_end = dst_begin + range.length;
    uint64_t src_begin = range.src_offset;
    uint64_t src_end = src_begin + range.length;
    ++idx;
    while (idx < ranges_.size()) {
      const auto& next = ranges_[idx];
      if (next.kind != SelectionPlan::Range::Kind::kData) {
        break;
      }
      if (next.dst_offset != dst_end || next.src_offset != src_end) {
        break;
      }
      dst_end += next.length;
      src_end += next.length;
      ++idx;
    }
    ExecutionRun run;
    run.kind = ExecutionRun::Kind::kContiguous;
    run.dst_begin = dst_begin;
    run.dst_end = dst_end;
    run.src_begin = src_begin;
    run.range_index = start;
    run.range_count = idx - start;
    runs_.push_back(run);
  }

  run_starts_.reserve(runs_.size());
  for (const auto& run : runs_) {
    run_starts_.push_back(run.dst_begin);
  }
}

size_t ViewPlanSource::find_run_index(uint64_t offset) const {
  if (run_starts_.empty()) {
    return 0;
  }
  auto it = std::upper_bound(run_starts_.begin(), run_starts_.end(), offset);
  if (it == run_starts_.begin()) {
    return 0;
  }
  return static_cast<size_t>(it - run_starts_.begin() - 1);
}

absl::StatusOr<size_t> ViewPlanSource::read_base(uint64_t offset, uint8_t* dst, size_t bytes) {
  if (bytes == 0) {
    return static_cast<size_t>(0);
  }
  stats_.base_read_calls.fetch_add(1, std::memory_order_relaxed);
  auto read_or = base_->read_at(offset, dst, bytes);
  if (!read_or.ok()) {
    return read_or.status();
  }
  stats_.base_read_bytes.fetch_add(*read_or, std::memory_order_relaxed);
  if (*read_or != bytes) {
    return absl::InternalError("short read while executing view selection plan");
  }
  return *read_or;
}

absl::StatusOr<size_t> ViewPlanSource::copy_from_range(
    const SelectionPlan::Range& range,
    uint64_t range_offset,
    uint8_t* dst,
    size_t bytes) {
  if (bytes == 0) {
    return static_cast<size_t>(0);
  }
  if (range.kind == SelectionPlan::Range::Kind::kPad) {
    std::memset(dst, 0, bytes);
    return bytes;
  }
  const uint64_t source_offset = range.src_offset + range_offset;
  return read_base(source_offset, dst, bytes);
}

absl::StatusOr<size_t> ViewPlanSource::copy_from_strided_ranges(
    const ExecutionRun& run,
    uint64_t run_offset,
    uint8_t* dst,
    size_t bytes) {
  if (bytes == 0) {
    return static_cast<size_t>(0);
  }
  if (run.row_len == 0) {
    return absl::InternalError("invalid strided run row length");
  }
  const uint64_t first_row = run_offset / run.row_len;
  uint64_t row_offset = run_offset % run.row_len;
  size_t range_index = run.range_index + static_cast<size_t>(first_row);
  const size_t range_limit = run.range_index + run.range_count;
  size_t remaining = bytes;
  uint8_t* out = dst;

  while (remaining > 0 && range_index < range_limit) {
    const auto& range = ranges_[range_index];
    const size_t available = static_cast<size_t>(range.length - row_offset);
    const size_t take = std::min(remaining, available);
    auto copied_or = copy_from_range(range, row_offset, out, take);
    if (!copied_or.ok()) {
      return copied_or.status();
    }
    out += *copied_or;
    remaining -= *copied_or;
    row_offset = 0;
    ++range_index;
  }
  return bytes - remaining;
}

absl::StatusOr<size_t> ViewPlanSource::fill_strided_run(
    size_t run_index,
    const ExecutionRun& run,
    uint64_t run_offset,
    uint8_t* dst,
    size_t bytes) {
  if (bytes == 0) {
    return static_cast<size_t>(0);
  }
  if (!run.strided_candidate || !strided_cache_) {
    return copy_from_strided_ranges(run, run_offset, dst, bytes);
  }
  if (strided_disabled_ && strided_disabled_[run_index].load(std::memory_order_relaxed) != 0) {
    return copy_from_strided_ranges(run, run_offset, dst, bytes);
  }
  if (run.row_len == 0 || run.rows == 0) {
    return absl::InternalError("invalid strided run");
  }

  auto load_block = [&](uint64_t row) -> absl::StatusOr<std::shared_ptr<const StridedBlock>> {
    if (strided_cache_) {
      absl::MutexLock lock(&strided_cache_->mutex);
      const auto cached = strided_cache_->block;
      if (cached && cached->run_index == run_index && row >= cached->first_row &&
          row < cached->first_row + cached->rows) {
        stats_.cache_hits.fetch_add(1, std::memory_order_relaxed);
        return cached;
      }
    }

    stats_.cache_misses.fetch_add(1, std::memory_order_relaxed);
    const uint64_t rows_per_block = std::max<uint64_t>(run.rows_per_block, 1);
    const uint64_t block_first_row = row - (row % rows_per_block);
    uint64_t block_rows = std::min(rows_per_block, run.rows - block_first_row);

    while (block_rows > 0) {
      const uint64_t block_bytes = (block_rows - 1) * run.stride + run.row_len;
      if (block_bytes > STRIDED_BLOCK_MAX_BYTES) {
        block_rows /= 2;
        continue;
      }
      auto block = std::make_shared<StridedBlock>();
      block->run_index = run_index;
      block->first_row = block_first_row;
      block->rows = block_rows;
      block->src_begin = run.src_base + block_first_row * run.stride;
      try {
        block->data.resize(static_cast<size_t>(block_bytes));
      } catch (const std::bad_alloc&) {
        block_rows /= 2;
        continue;
      }
      auto read_or = read_base(block->src_begin, block->data.data(), block->data.size());
      if (!read_or.ok()) {
        return read_or.status();
      }
      if (strided_cache_) {
        absl::MutexLock lock(&strided_cache_->mutex);
        strided_cache_->block = block;
      }
      return block;
    }

    return absl::ResourceExhaustedError("unable to allocate strided block buffer");
  };

  uint64_t local_offset = run_offset;
  size_t remaining = bytes;
  uint8_t* out = dst;

  while (remaining > 0) {
    const uint64_t row = local_offset / run.row_len;
    uint64_t row_offset = local_offset % run.row_len;
    if (row >= run.rows) {
      return absl::InternalError("strided run offset out of range");
    }
    auto block_or = load_block(row);
    if (!block_or.ok()) {
      if (absl::IsResourceExhausted(block_or.status())) {
        if (strided_disabled_) {
          uint8_t expected = 0;
          if (strided_disabled_[run_index].compare_exchange_strong(
                  expected, 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
            stats_.strided_runtime_fallbacks.fetch_add(1, std::memory_order_relaxed);
          }
        }
        return copy_from_strided_ranges(run, local_offset, out, remaining);
      }
      return block_or.status();
    }
    const auto& block = *block_or;
    const uint64_t block_end_row = block->first_row + block->rows;
    uint64_t active_row = row;
    while (remaining > 0 && active_row < block_end_row) {
      const size_t available = static_cast<size_t>(run.row_len - row_offset);
      const size_t take = std::min(remaining, available);
      const uint64_t block_offset = (active_row - block->first_row) * run.stride + row_offset;
      std::memcpy(out, block->data.data() + block_offset, take);
      stats_.pack_bytes.fetch_add(take, std::memory_order_relaxed);
      out += take;
      remaining -= take;
      local_offset += take;
      row_offset += take;
      if (row_offset == run.row_len) {
        ++active_row;
        row_offset = 0;
      }
    }
  }

  return bytes;
}

absl::StatusOr<size_t> ViewPlanSource::read(void* dst, size_t max_bytes) {
  auto bytes_or = read_at(cursor_, dst, max_bytes);
  if (!bytes_or.ok()) {
    return bytes_or;
  }
  cursor_ += *bytes_or;
  return bytes_or;
}

absl::StatusOr<size_t> ViewPlanSource::read_at(uint64_t offset, void* dst, size_t bytes) {
  if (offset >= total_bytes_ || bytes == 0) {
    return static_cast<size_t>(0);
  }
  const uint64_t remaining_bytes = total_bytes_ - offset;
  size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, remaining_bytes));
  uint8_t* out = static_cast<uint8_t*>(dst);
  size_t copied = 0;
  uint64_t cursor = offset;

  size_t run_index = find_run_index(offset);
  while (run_index < runs_.size() && copied < to_copy) {
    const auto& run = runs_[run_index];
    if (cursor >= run.dst_end) {
      ++run_index;
      continue;
    }
    if (cursor < run.dst_begin) {
      return absl::InternalError("selection plan contains uncovered gaps");
    }
    const uint64_t run_offset = cursor - run.dst_begin;
    const size_t available = static_cast<size_t>(run.dst_end - cursor);
    const size_t chunk = std::min(to_copy - copied, available);
    absl::StatusOr<size_t> copied_or;
    switch (run.kind) {
      case ExecutionRun::Kind::kPad:
        std::memset(out + copied, 0, chunk);
        copied_or = chunk;
        break;
      case ExecutionRun::Kind::kContiguous:
        copied_or = read_base(run.src_begin + run_offset, out + copied, chunk);
        break;
      case ExecutionRun::Kind::kStrided:
        copied_or = fill_strided_run(run_index, run, run_offset, out + copied, chunk);
        break;
    }
    if (!copied_or.ok()) {
      return copied_or.status();
    }
    copied += *copied_or;
    cursor += *copied_or;
    if (*copied_or == 0) {
      break;
    }
    ++run_index;
  }

  stats_.output_bytes.fetch_add(copied, std::memory_order_relaxed);
  return copied;
}

std::unique_ptr<SeekableSource> make_view_plan_source(std::unique_ptr<SeekableSource> base, SelectionPlan plan) {
  if (!base) {
    return nullptr;
  }
  if (plan.total_bytes == 0 || plan.ranges.empty()) {
    return base;
  }
  return std::make_unique<OwningViewPlanSource>(std::move(base), std::move(plan));
}

} // namespace tensorcast::store::loader
