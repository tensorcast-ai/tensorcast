// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/byte_range_mapped_source.h"

#include <algorithm>
#include <cstring>
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
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> pad_bytes;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> pack_bytes;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> strided_runs;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> strided_fallback_runs;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> cache_hits;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> cache_misses;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> direct_write_calls;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> direct_write_bytes;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> direct_write_fallback_calls;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> amplification_ratio;
};

MetricsHandles init_metrics_handles() {
  MetricsHandles handles;
  handles.meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  handles.base_read_calls = handles.meter->CreateDoubleCounter("tc_byte_range_base_read_calls_total");
  handles.base_read_bytes = handles.meter->CreateDoubleCounter("tc_byte_range_base_read_bytes_total");
  handles.output_bytes = handles.meter->CreateDoubleCounter("tc_byte_range_output_bytes_total");
  handles.pad_bytes = handles.meter->CreateDoubleCounter("tc_byte_range_pad_bytes_total");
  handles.pack_bytes = handles.meter->CreateDoubleCounter("tc_byte_range_pack_bytes_total");
  handles.strided_runs = handles.meter->CreateDoubleCounter("tc_byte_range_strided_runs_total");
  handles.strided_fallback_runs = handles.meter->CreateDoubleCounter("tc_byte_range_strided_fallback_runs_total");
  handles.cache_hits = handles.meter->CreateDoubleCounter("tc_byte_range_strided_cache_hits_total");
  handles.cache_misses = handles.meter->CreateDoubleCounter("tc_byte_range_strided_cache_misses_total");
  handles.direct_write_calls = handles.meter->CreateDoubleCounter("tc_byte_range_direct_write_calls_total");
  handles.direct_write_bytes = handles.meter->CreateDoubleCounter("tc_byte_range_direct_write_bytes_total");
  handles.direct_write_fallback_calls =
      handles.meter->CreateDoubleCounter("tc_byte_range_direct_write_fallback_calls_total");
  handles.amplification_ratio = handles.meter->CreateDoubleHistogram("tc_byte_range_amplification_ratio");
  return handles;
}

MetricsHandles& metrics_handles() {
  static MetricsHandles handles = init_metrics_handles();
  return handles;
}

void record_counter(
    const opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>>& counter,
    std::string_view path,
    uint64_t value) {
  if (value == 0) {
    return;
  }
  try {
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("path", opentelemetry::common::AttributeValue(std::string(path)));
    counter->Add(
        static_cast<double>(value),
        opentelemetry::common::KeyValueIterableView(attrs),
        opentelemetry::context::Context{});
  } catch (...) {
    VLOG(2) << "metrics counter tc_byte_range_* unavailable";
  }
}

void record_histogram(
    const opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>>& histogram,
    std::string_view path,
    double value) {
  try {
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("path", opentelemetry::common::AttributeValue(std::string(path)));
    histogram->Record(value, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
    VLOG(2) << "metrics histogram tc_byte_range_amplification_ratio unavailable";
  }
}

} // namespace

struct ByteRangeMappedSource::StridedBlockCache {
  absl::Mutex mutex;
  std::shared_ptr<const StridedBlock> block ABSL_GUARDED_BY(mutex);
};

absl::StatusOr<std::unique_ptr<ByteRangeMappedSource>> ByteRangeMappedSource::Create(
    ByteRangeMap map,
    std::shared_ptr<const ByteRangeProgram> program,
    std::vector<std::shared_ptr<SeekableSource>> sources,
    Options options) {
  if (!program) {
    return absl::InvalidArgumentError("ByteRangeMappedSource requires a program");
  }
  if (map.total_bytes != program->total_bytes) {
    return absl::InvalidArgumentError("ByteRangeMappedSource map/program total_bytes mismatch");
  }
  if (sources.size() < map.num_sources) {
    return absl::InvalidArgumentError("ByteRangeMappedSource missing sources for map");
  }
  bool direct_write_supported = options.enable_direct_write_at && !program->has_strided_runs;
  if (direct_write_supported) {
    for (const auto& run : program->runs) {
      if (run.kind != ByteRangeRun::Kind::kContiguous) {
        continue;
      }
      if (run.source_index >= sources.size()) {
        direct_write_supported = false;
        break;
      }
      if (!sources[run.source_index]->supports_direct_write_at()) {
        direct_write_supported = false;
        break;
      }
    }
  }

  auto source = std::unique_ptr<ByteRangeMappedSource>(new ByteRangeMappedSource(
      std::move(map), std::move(program), std::move(sources), std::move(options), direct_write_supported));
  if (auto st = source->validate_sources(source->map_); !st.ok()) {
    return st;
  }
  return source;
}

ByteRangeMappedSource::ByteRangeMappedSource(
    ByteRangeMap map,
    std::shared_ptr<const ByteRangeProgram> program,
    std::vector<std::shared_ptr<SeekableSource>> sources,
    Options options,
    bool direct_write_supported)
    : map_(std::move(map)),
      program_(std::move(program)),
      sources_(std::move(sources)),
      options_(std::move(options)),
      direct_write_supported_(direct_write_supported) {
  if (!program_->runs.empty()) {
    strided_disabled_ = std::make_unique<std::atomic<uint8_t>[]>(program_->runs.size());
    for (size_t idx = 0; idx < program_->runs.size(); ++idx) {
      strided_disabled_[idx].store(0, std::memory_order_relaxed);
    }
  }
  if (program_->has_strided_runs) {
    strided_cache_ = std::make_unique<StridedBlockCache>();
  }
}

ByteRangeMappedSource::~ByteRangeMappedSource() {
  const uint64_t base_read_calls = stats_.base_read_calls.load(std::memory_order_relaxed);
  const uint64_t base_read_bytes = stats_.base_read_bytes.load(std::memory_order_relaxed);
  const uint64_t output_bytes = stats_.output_bytes.load(std::memory_order_relaxed);
  const uint64_t pad_bytes = stats_.pad_bytes.load(std::memory_order_relaxed);
  const uint64_t pack_bytes = stats_.pack_bytes.load(std::memory_order_relaxed);
  const uint64_t cache_hits = stats_.cache_hits.load(std::memory_order_relaxed);
  const uint64_t cache_misses = stats_.cache_misses.load(std::memory_order_relaxed);
  const uint64_t runtime_fallbacks = stats_.strided_runtime_fallbacks.load(std::memory_order_relaxed);
  const uint64_t direct_write_calls = stats_.direct_write_calls.load(std::memory_order_relaxed);
  const uint64_t direct_write_bytes = stats_.direct_write_bytes.load(std::memory_order_relaxed);
  const uint64_t direct_write_fallbacks = stats_.direct_write_fallback_calls.load(std::memory_order_relaxed);

  const uint64_t strided_fallback_runs = program_->strided_compile_fallback_runs + runtime_fallbacks;
  const double amplification =
      output_bytes == 0 ? 0.0 : static_cast<double>(base_read_bytes) / static_cast<double>(output_bytes);

  VLOG(1) << "ByteRangeMappedSource summary: output_bytes=" << output_bytes << " base_read_calls=" << base_read_calls
          << " base_read_bytes=" << base_read_bytes << " pad_bytes=" << pad_bytes << " pack_bytes=" << pack_bytes
          << " strided_runs=" << program_->strided_candidate_runs << " strided_fallback_runs=" << strided_fallback_runs
          << " cache_hits=" << cache_hits << " cache_misses=" << cache_misses << " amplification=" << amplification
          << " direct_write_calls=" << direct_write_calls << " direct_write_bytes=" << direct_write_bytes
          << " direct_write_fallback_calls=" << direct_write_fallbacks;

  auto& handles = metrics_handles();
  record_counter(handles.base_read_calls, options_.path, base_read_calls);
  record_counter(handles.base_read_bytes, options_.path, base_read_bytes);
  record_counter(handles.output_bytes, options_.path, output_bytes);
  record_counter(handles.pad_bytes, options_.path, pad_bytes);
  record_counter(handles.pack_bytes, options_.path, pack_bytes);
  record_counter(handles.strided_runs, options_.path, program_->strided_candidate_runs);
  record_counter(handles.strided_fallback_runs, options_.path, strided_fallback_runs);
  record_counter(handles.cache_hits, options_.path, cache_hits);
  record_counter(handles.cache_misses, options_.path, cache_misses);
  record_counter(handles.direct_write_calls, options_.path, direct_write_calls);
  record_counter(handles.direct_write_bytes, options_.path, direct_write_bytes);
  record_counter(handles.direct_write_fallback_calls, options_.path, direct_write_fallbacks);
  record_histogram(handles.amplification_ratio, options_.path, amplification);
}

uint64_t ByteRangeMappedSource::total_bytes() const {
  return program_->total_bytes;
}

bool ByteRangeMappedSource::supports_direct_write_at() const {
  return direct_write_supported_;
}

absl::Status ByteRangeMappedSource::validate_sources(const ByteRangeMap& map) const {
  if (sources_.size() < map.num_sources) {
    return absl::InvalidArgumentError("ByteRangeMappedSource sources do not satisfy map.num_sources");
  }
  for (const auto& seg : map.segments) {
    if (seg.kind != ByteRangeSegment::Kind::kData) {
      continue;
    }
    if (seg.source_index >= sources_.size()) {
      return absl::InvalidArgumentError("ByteRangeMappedSource source index out of range");
    }
    const uint64_t total = sources_[seg.source_index]->total_bytes();
    if (seg.src_offset > total || seg.length > total - seg.src_offset) {
      return absl::InvalidArgumentError("ByteRangeMappedSource map references out-of-bounds source bytes");
    }
  }
  return absl::OkStatus();
}

size_t ByteRangeMappedSource::find_run_index(uint64_t offset) const {
  if (program_->run_starts.empty()) {
    return 0;
  }
  auto it = std::upper_bound(program_->run_starts.begin(), program_->run_starts.end(), offset);
  if (it == program_->run_starts.begin()) {
    return 0;
  }
  return static_cast<size_t>(it - program_->run_starts.begin() - 1);
}

absl::StatusOr<size_t> ByteRangeMappedSource::read_base(
    uint32_t source_index,
    uint64_t offset,
    uint8_t* dst,
    size_t bytes) {
  if (bytes == 0) {
    return static_cast<size_t>(0);
  }
  if (source_index >= sources_.size()) {
    return absl::InvalidArgumentError("ByteRangeMappedSource source index out of range");
  }
  stats_.base_read_calls.fetch_add(1, std::memory_order_relaxed);
  auto read_or = sources_[source_index]->read_at(offset, dst, bytes);
  if (!read_or.ok()) {
    return read_or.status();
  }
  stats_.base_read_bytes.fetch_add(*read_or, std::memory_order_relaxed);
  if (*read_or != bytes) {
    return absl::DataLossError("short read while executing byte range program");
  }
  return *read_or;
}

absl::StatusOr<size_t> ByteRangeMappedSource::copy_from_strided_rows(
    const ByteRangeRun& run,
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
  uint64_t row = first_row;
  size_t remaining = bytes;
  uint8_t* out = dst;

  while (remaining > 0 && row < run.rows) {
    const size_t available = static_cast<size_t>(run.row_len - row_offset);
    const size_t take = std::min(remaining, available);
    const uint64_t src_offset = run.src_base + row * run.stride + row_offset;
    auto copied_or = read_base(run.source_index, src_offset, out, take);
    if (!copied_or.ok()) {
      return copied_or.status();
    }
    out += *copied_or;
    remaining -= *copied_or;
    row_offset = 0;
    ++row;
  }
  return bytes - remaining;
}

absl::StatusOr<size_t> ByteRangeMappedSource::fill_strided_run(
    size_t run_index,
    const ByteRangeRun& run,
    uint64_t run_offset,
    uint8_t* dst,
    size_t bytes) {
  if (bytes == 0) {
    return static_cast<size_t>(0);
  }
  if (!strided_cache_) {
    return copy_from_strided_rows(run, run_offset, dst, bytes);
  }
  if (strided_disabled_ && strided_disabled_[run_index].load(std::memory_order_relaxed) != 0) {
    return copy_from_strided_rows(run, run_offset, dst, bytes);
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
      if (block_bytes > program_->strided_block_max_bytes) {
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
      auto read_or = read_base(run.source_index, block->src_begin, block->data.data(), block->data.size());
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
        const size_t already_copied = bytes - remaining;
        auto fallback_or = copy_from_strided_rows(run, local_offset, out, remaining);
        if (!fallback_or.ok()) {
          return fallback_or.status();
        }
        return already_copied + *fallback_or;
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

absl::StatusOr<size_t> ByteRangeMappedSource::read(void* dst, size_t max_bytes) {
  auto bytes_or = read_at(cursor_, dst, max_bytes);
  if (!bytes_or.ok()) {
    return bytes_or;
  }
  cursor_ += *bytes_or;
  return bytes_or;
}

absl::StatusOr<size_t> ByteRangeMappedSource::read_at(uint64_t offset, void* dst, size_t bytes) {
  if (offset >= program_->total_bytes || bytes == 0) {
    return static_cast<size_t>(0);
  }
  const uint64_t remaining_bytes = program_->total_bytes - offset;
  size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, remaining_bytes));
  uint8_t* out = static_cast<uint8_t*>(dst);
  size_t copied = 0;
  uint64_t cursor = offset;

  size_t run_index = find_run_index(offset);
  while (run_index < program_->runs.size() && copied < to_copy) {
    const auto& run = program_->runs[run_index];
    if (cursor >= run.dst_end) {
      ++run_index;
      continue;
    }
    if (cursor < run.dst_begin) {
      return absl::InternalError("byte range program contains uncovered gaps");
    }
    const uint64_t run_offset = cursor - run.dst_begin;
    const size_t available = static_cast<size_t>(run.dst_end - cursor);
    const size_t chunk = std::min(to_copy - copied, available);
    absl::StatusOr<size_t> copied_or;
    switch (run.kind) {
      case ByteRangeRun::Kind::kPad:
        std::memset(out + copied, 0, chunk);
        stats_.pad_bytes.fetch_add(chunk, std::memory_order_relaxed);
        copied_or = chunk;
        break;
      case ByteRangeRun::Kind::kContiguous:
        copied_or = read_base(run.source_index, run.src_begin + run_offset, out + copied, chunk);
        break;
      case ByteRangeRun::Kind::kStrided:
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

absl::Status ByteRangeMappedSource::zero_fill_to_grant(
    uint64_t dest_va_offset,
    size_t bytes,
    const DirectWriteGrant& grant) {
  size_t remaining = bytes;
  uint64_t cursor = dest_va_offset;
  while (remaining > 0) {
    const DirectWriteGrant::Window* target = nullptr;
    for (const auto& window : grant.windows) {
      if (cursor >= window.va_offset && cursor < window.va_offset + window.length) {
        target = &window;
        break;
      }
    }
    if (target == nullptr) {
      return absl::InvalidArgumentError("No direct-write window covers requested range");
    }
    const uint64_t window_offset = cursor - target->va_offset;
    const size_t available = static_cast<size_t>(target->length - window_offset);
    const size_t take = std::min(remaining, available);
    std::memset(reinterpret_cast<void*>(target->local_addr + window_offset), 0, take);
    remaining -= take;
    cursor += take;
  }
  return absl::OkStatus();
}

absl::StatusOr<size_t> ByteRangeMappedSource::read_into_at(
    uint64_t src_offset,
    uint64_t dest_va_offset,
    size_t bytes,
    const DirectWriteGrant& grant) {
  if (!supports_direct_write_at()) {
    stats_.direct_write_fallback_calls.fetch_add(1, std::memory_order_relaxed);
    return absl::UnimplementedError("direct write not supported for byte range program");
  }
  if (src_offset >= program_->total_bytes || bytes == 0) {
    return static_cast<size_t>(0);
  }

  stats_.direct_write_calls.fetch_add(1, std::memory_order_relaxed);
  const uint64_t remaining_bytes = program_->total_bytes - src_offset;
  size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, remaining_bytes));
  size_t copied = 0;
  uint64_t cursor = src_offset;

  size_t run_index = find_run_index(src_offset);
  while (run_index < program_->runs.size() && copied < to_copy) {
    const auto& run = program_->runs[run_index];
    if (cursor >= run.dst_end) {
      ++run_index;
      continue;
    }
    if (cursor < run.dst_begin) {
      return absl::InternalError("byte range program contains uncovered gaps");
    }
    const uint64_t run_offset = cursor - run.dst_begin;
    const size_t available = static_cast<size_t>(run.dst_end - cursor);
    const size_t chunk = std::min(to_copy - copied, available);
    switch (run.kind) {
      case ByteRangeRun::Kind::kPad: {
        auto st = zero_fill_to_grant(dest_va_offset + copied, chunk, grant);
        if (!st.ok()) {
          return st;
        }
        stats_.pad_bytes.fetch_add(chunk, std::memory_order_relaxed);
        copied += chunk;
        cursor += chunk;
        break;
      }
      case ByteRangeRun::Kind::kContiguous: {
        stats_.base_read_calls.fetch_add(1, std::memory_order_relaxed);
        auto got =
            sources_[run.source_index]->read_into_at(run.src_begin + run_offset, dest_va_offset + copied, chunk, grant);
        if (!got.ok()) {
          return got.status();
        }
        if (*got != chunk) {
          return absl::DataLossError("short direct write while executing byte range program");
        }
        stats_.base_read_bytes.fetch_add(*got, std::memory_order_relaxed);
        copied += *got;
        cursor += *got;
        break;
      }
      case ByteRangeRun::Kind::kStrided:
        stats_.direct_write_fallback_calls.fetch_add(1, std::memory_order_relaxed);
        return absl::UnimplementedError("direct write not supported for strided runs");
    }
    ++run_index;
  }

  stats_.direct_write_bytes.fetch_add(copied, std::memory_order_relaxed);
  stats_.output_bytes.fetch_add(copied, std::memory_order_relaxed);
  return copied;
}

} // namespace tensorcast::store::loader
