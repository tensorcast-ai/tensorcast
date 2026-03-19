// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/byte_range_mapped_source.h"

#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <typeinfo>
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

pid_t current_tid() {
  return static_cast<pid_t>(::syscall(SYS_gettid));
}

int current_cpu() {
  return ::sched_getcpu();
}

int addr_numa_node(const void* addr) {
  if (addr == nullptr) {
    return -1;
  }
  int node = -1;
  long rc = ::syscall(SYS_get_mempolicy, &node, nullptr, 0, const_cast<void*>(addr), MPOL_F_NODE | MPOL_F_ADDR);
  if (rc != 0) {
    return -1;
  }
  return node;
}

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

const char* run_kind_to_cstr(ByteRangeRun::Kind kind) {
  switch (kind) {
    case ByteRangeRun::Kind::kPad:
      return "pad";
    case ByteRangeRun::Kind::kContiguous:
      return "contiguous";
    case ByteRangeRun::Kind::kStrided:
      return "strided";
  }
  return "unknown";
}

uint64_t elapsed_us(std::chrono::steady_clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
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
  const auto start = std::chrono::steady_clock::now();
  if (bytes == 0) {
    VLOG(2) << "byte_range.read_base bytes=0 source_index=" << source_index << " offset=" << offset << " duration_us=0";
    return static_cast<size_t>(0);
  }
  if (source_index >= sources_.size()) {
    VLOG(2) << "byte_range.read_base invalid_source source_index=" << source_index
            << " source_count=" << sources_.size() << " offset=" << offset << " bytes=" << bytes;
    return absl::InvalidArgumentError("ByteRangeMappedSource source index out of range");
  }
  stats_.base_read_calls.fetch_add(1, std::memory_order_relaxed);
  auto read_or = sources_[source_index]->read_at(offset, dst, bytes);
  if (!read_or.ok()) {
    VLOG(2) << "byte_range.read_base read_failed source_index=" << source_index << " offset=" << offset
            << " bytes=" << bytes << " duration_us=" << elapsed_us(start) << " status=" << read_or.status();
    return read_or.status();
  }
  stats_.base_read_bytes.fetch_add(*read_or, std::memory_order_relaxed);
  if (*read_or != bytes) {
    VLOG(2) << "byte_range.read_base short_read source_index=" << source_index << " offset=" << offset
            << " requested=" << bytes << " got=" << *read_or << " duration_us=" << elapsed_us(start)
            << " tid=" << current_tid() << " cpu=" << current_cpu();
    return absl::DataLossError("short read while executing byte range program");
  }
  VLOG(2) << "byte_range.read_base ok source_index=" << source_index << " offset=" << offset << " bytes=" << bytes
          << " duration_us=" << elapsed_us(start) << " tid=" << current_tid() << " cpu=" << current_cpu();
  return *read_or;
}

absl::StatusOr<size_t> ByteRangeMappedSource::copy_from_strided_rows(
    const ByteRangeRun& run,
    uint64_t run_offset,
    uint8_t* dst,
    size_t bytes) {
  const auto start = std::chrono::steady_clock::now();
  if (bytes == 0) {
    VLOG(2) << "byte_range.strided.row_copy bytes=0 source_index=" << run.source_index << " run_offset=" << run_offset
            << " duration_us=0";
    return static_cast<size_t>(0);
  }
  if (run.source_index >= sources_.size()) {
    VLOG(2) << "byte_range.strided.row_copy invalid_source source_index=" << run.source_index
            << " source_count=" << sources_.size() << " run_offset=" << run_offset << " bytes=" << bytes;
    return absl::InvalidArgumentError("ByteRangeMappedSource source index out of range");
  }
  if (run.row_len == 0) {
    VLOG(2) << "byte_range.strided.row_copy invalid_row_len source_index=" << run.source_index
            << " run_offset=" << run_offset << " bytes=" << bytes;
    return absl::InternalError("invalid strided run row length");
  }
  const auto& source = sources_[run.source_index];
  const uint64_t first_row = run_offset / run.row_len;
  uint64_t row_offset = run_offset % run.row_len;
  uint64_t row = first_row;
  size_t remaining = bytes;
  uint8_t* out = dst;
  uint64_t local_base_read_calls = 0;
  uint64_t local_base_read_bytes = 0;

  while (remaining > 0 && row < run.rows) {
    const size_t available = static_cast<size_t>(run.row_len - row_offset);
    const size_t take = std::min(remaining, available);
    const uint64_t src_offset = run.src_base + row * run.stride + row_offset;
    ++local_base_read_calls;
    auto copied_or = source->read_at(src_offset, out, take);
    if (!copied_or.ok()) {
      if (local_base_read_calls != 0) {
        stats_.base_read_calls.fetch_add(local_base_read_calls, std::memory_order_relaxed);
      }
      if (local_base_read_bytes != 0) {
        stats_.base_read_bytes.fetch_add(local_base_read_bytes, std::memory_order_relaxed);
      }
      VLOG(2) << "byte_range.strided.row_copy read_failed source_index=" << run.source_index << " row=" << row
              << " src_offset=" << src_offset << " bytes=" << take << " duration_us=" << elapsed_us(start)
              << " status=" << copied_or.status();
      return copied_or.status();
    }
    if (*copied_or != take) {
      if (local_base_read_calls != 0) {
        stats_.base_read_calls.fetch_add(local_base_read_calls, std::memory_order_relaxed);
      }
      if (local_base_read_bytes != 0) {
        stats_.base_read_bytes.fetch_add(local_base_read_bytes, std::memory_order_relaxed);
      }
      VLOG(2) << "byte_range.strided.row_copy short_read source_index=" << run.source_index << " row=" << row
              << " src_offset=" << src_offset << " requested=" << take << " got=" << *copied_or
              << " duration_us=" << elapsed_us(start);
      return absl::DataLossError("short read while executing strided row copy");
    }
    local_base_read_bytes += *copied_or;
    out += *copied_or;
    remaining -= *copied_or;
    row_offset = 0;
    ++row;
  }
  if (local_base_read_calls != 0) {
    stats_.base_read_calls.fetch_add(local_base_read_calls, std::memory_order_relaxed);
  }
  if (local_base_read_bytes != 0) {
    stats_.base_read_bytes.fetch_add(local_base_read_bytes, std::memory_order_relaxed);
  }
  VLOG(2) << "byte_range.strided.row_copy done source_index=" << run.source_index << " first_row=" << first_row
          << " bytes=" << (bytes - remaining) << " rows_touched=" << (row > first_row ? (row - first_row) : 0)
          << " duration_us=" << elapsed_us(start);
  return bytes - remaining;
}

absl::StatusOr<size_t> ByteRangeMappedSource::fill_strided_run(
    size_t run_index,
    const ByteRangeRun& run,
    uint64_t run_offset,
    uint8_t* dst,
    size_t bytes,
    uint64_t* pack_us_total,
    size_t* pack_bytes_total,
    uint64_t* cache_lookup_us_total,
    uint64_t* block_prepare_us_total,
    uint64_t* block_load_us_total,
    uint64_t* row_copy_us_total,
    size_t* row_copy_bytes_total) {
  const auto run_start = std::chrono::steady_clock::now();
  uint64_t local_pack_us_total = 0;
  size_t local_pack_bytes_total = 0;
  uint64_t local_cache_lookup_us_total = 0;
  uint64_t local_block_prepare_us_total = 0;
  uint64_t local_block_load_us_total = 0;
  uint64_t local_row_copy_us_total = 0;
  size_t local_row_copy_bytes_total = 0;
  uint64_t local_block_pick_us_total = 0;
  uint64_t local_block_resize_us_total = 0;
  uint64_t local_pack_memcpy_calls = 0;
  size_t local_cache_hit_count = 0;
  size_t local_cache_miss_count = 0;
  size_t local_block_load_count = 0;
  size_t local_block_reuse_count = 0;
  size_t local_block_new_count = 0;
  size_t local_block_bytes_total = 0;
  auto flush_local_stats = [&]() {
    if (pack_us_total != nullptr) {
      *pack_us_total += local_pack_us_total;
    }
    if (pack_bytes_total != nullptr) {
      *pack_bytes_total += local_pack_bytes_total;
    }
    if (cache_lookup_us_total != nullptr) {
      *cache_lookup_us_total += local_cache_lookup_us_total;
    }
    if (block_prepare_us_total != nullptr) {
      *block_prepare_us_total += local_block_prepare_us_total;
    }
    if (block_load_us_total != nullptr) {
      *block_load_us_total += local_block_load_us_total;
    }
    if (row_copy_us_total != nullptr) {
      *row_copy_us_total += local_row_copy_us_total;
    }
    if (row_copy_bytes_total != nullptr) {
      *row_copy_bytes_total += local_row_copy_bytes_total;
    }
  };
  VLOG(2) << "byte_range.strided.begin run_index=" << run_index << " source_index=" << run.source_index
          << " run_offset=" << run_offset << " bytes=" << bytes << " row_len=" << run.row_len
          << " stride=" << run.stride << " rows=" << run.rows;
  if (bytes == 0) {
    VLOG(2) << "byte_range.strided.end run_index=" << run_index << " copied=0 duration_us=0";
    flush_local_stats();
    return static_cast<size_t>(0);
  }
  if (!strided_cache_) {
    const auto row_copy_start = std::chrono::steady_clock::now();
    auto copied_or = copy_from_strided_rows(run, run_offset, dst, bytes);
    const uint64_t row_copy_us = elapsed_us(row_copy_start);
    local_row_copy_us_total += row_copy_us;
    if (copied_or.ok()) {
      local_row_copy_bytes_total += *copied_or;
    }
    if (copied_or.ok()) {
      VLOG(2) << "byte_range.strided.end run_index=" << run_index << " mode=no_cache copied=" << *copied_or
              << " row_copy_us=" << row_copy_us << " duration_us=" << elapsed_us(run_start);
    } else {
      VLOG(2) << "byte_range.strided.end run_index=" << run_index << " mode=no_cache status=" << copied_or.status()
              << " row_copy_us=" << row_copy_us << " duration_us=" << elapsed_us(run_start);
    }
    flush_local_stats();
    return copied_or;
  }
  if (strided_disabled_ && strided_disabled_[run_index].load(std::memory_order_relaxed) != 0) {
    const auto row_copy_start = std::chrono::steady_clock::now();
    auto copied_or = copy_from_strided_rows(run, run_offset, dst, bytes);
    const uint64_t row_copy_us = elapsed_us(row_copy_start);
    local_row_copy_us_total += row_copy_us;
    if (copied_or.ok()) {
      local_row_copy_bytes_total += *copied_or;
    }
    if (copied_or.ok()) {
      VLOG(2) << "byte_range.strided.end run_index=" << run_index << " mode=disabled copied=" << *copied_or
              << " row_copy_us=" << row_copy_us << " duration_us=" << elapsed_us(run_start);
    } else {
      VLOG(2) << "byte_range.strided.end run_index=" << run_index << " mode=disabled status=" << copied_or.status()
              << " row_copy_us=" << row_copy_us << " duration_us=" << elapsed_us(run_start);
    }
    flush_local_stats();
    return copied_or;
  }
  if (run.row_len == 0 || run.rows == 0) {
    VLOG(2) << "byte_range.strided.invalid run_index=" << run_index << " row_len=" << run.row_len
            << " rows=" << run.rows;
    flush_local_stats();
    return absl::InternalError("invalid strided run");
  }
  const uint64_t first_row = run_offset / run.row_len;
  const uint64_t last_offset_exclusive = run_offset + static_cast<uint64_t>(bytes);
  const uint64_t last_row_exclusive = (last_offset_exclusive + run.row_len - 1) / run.row_len;
  const uint64_t rows_touched = last_row_exclusive > first_row ? (last_row_exclusive - first_row) : 0;
  auto& source = sources_[run.source_index];
  const uint8_t* cpu_source_base_ptr = source->cpu_base_ptr();
  const uint64_t first_src_offset = run.src_base + first_row * run.stride + (run_offset % run.row_len);
  VLOG(2) << "byte_range.strided.direct_gather_probe run_index=" << run_index << " source_index=" << run.source_index
          << " source_type=" << typeid(*source).name()
          << " has_cpu_base_ptr=" << (cpu_source_base_ptr != nullptr ? 1 : 0) << " row_len=" << run.row_len
          << " stride=" << run.stride << " bytes=" << bytes << " rows_touched=" << rows_touched
          << " tid=" << current_tid() << " cpu=" << current_cpu()
          << " source_base_numa=" << addr_numa_node(cpu_source_base_ptr) << " first_src_addr_numa="
          << addr_numa_node(cpu_source_base_ptr != nullptr ? cpu_source_base_ptr + first_src_offset : nullptr)
          << " dst_numa=" << addr_numa_node(dst);
  const bool force_mmap_direct_gather = false;
  const bool threshold_direct_gather = run.stride > run.row_len &&
      run.row_len >= options_.direct_gather_min_row_len_bytes && bytes >= options_.direct_gather_min_total_bytes &&
      rows_touched <= options_.direct_gather_max_rows_touched && cpu_source_base_ptr != nullptr;
  const bool direct_gather_candidate = force_mmap_direct_gather || threshold_direct_gather;
  if (direct_gather_candidate) {
    const auto gather_start = std::chrono::steady_clock::now();
    const uint8_t* base_ptr = cpu_source_base_ptr;
    size_t copied = 0;
    uint64_t row = first_row;
    uint64_t row_offset = run_offset % run.row_len;
    uint8_t* out_ptr = dst;
    uint64_t direct_gather_memcpy_calls = 0;
    while (copied < bytes && row < run.rows) {
      const size_t available = static_cast<size_t>(run.row_len - row_offset);
      const size_t take = std::min(bytes - copied, available);
      const uint64_t src_offset = run.src_base + row * run.stride + row_offset;
      const uint64_t source_total = sources_[run.source_index]->total_bytes();
      if (src_offset > source_total || static_cast<uint64_t>(take) > source_total - src_offset) {
        const uint64_t gather_us = elapsed_us(gather_start);
        VLOG(2) << "byte_range.strided.direct_gather_oob run_index=" << run_index
                << " source_index=" << run.source_index << " src_offset=" << src_offset << " take=" << take
                << " source_total=" << source_total << " copied=" << copied << " duration_us=" << gather_us;
        flush_local_stats();
        return absl::OutOfRangeError("direct gather source range out of bounds");
      }
      std::memcpy(out_ptr, base_ptr + src_offset, take);
      ++direct_gather_memcpy_calls;
      copied += take;
      out_ptr += take;
      row_offset = 0;
      ++row;
    }
    const uint64_t gather_us = elapsed_us(gather_start);
    local_row_copy_us_total += gather_us;
    local_row_copy_bytes_total += copied;
    if (copied != bytes) {
      VLOG(2) << "byte_range.strided.direct_gather_short_copy run_index=" << run_index
              << " source_index=" << run.source_index << " requested=" << bytes << " copied=" << copied
              << " rows_touched=" << rows_touched << " duration_us=" << gather_us;
      flush_local_stats();
      return absl::DataLossError("direct gather short copy");
    }
    stats_.base_read_calls.fetch_add(direct_gather_memcpy_calls, std::memory_order_relaxed);
    stats_.base_read_bytes.fetch_add(copied, std::memory_order_relaxed);
    VLOG(2) << "byte_range.strided.direct_gather run_index=" << run_index << " source_index=" << run.source_index
            << " forced=" << (force_mmap_direct_gather ? 1 : 0) << " rows_touched=" << rows_touched
            << " bytes=" << copied << " memcpy_calls=" << direct_gather_memcpy_calls << " duration_us=" << gather_us
            << " tid=" << current_tid() << " cpu=" << current_cpu()
            << " source_base_numa=" << addr_numa_node(cpu_source_base_ptr) << " first_src_addr_numa="
            << addr_numa_node(cpu_source_base_ptr != nullptr ? cpu_source_base_ptr + first_src_offset : nullptr)
            << " dst_numa=" << addr_numa_node(dst);
    flush_local_stats();
    return copied;
  }

  auto load_block = [&](uint64_t row) -> absl::StatusOr<std::shared_ptr<const StridedBlock>> {
    const auto cache_lookup_start = std::chrono::steady_clock::now();
    if (strided_cache_) {
      absl::MutexLock lock(&strided_cache_->mutex);
      const auto cached = strided_cache_->block;
      if (cached && cached->run_index == run_index && row >= cached->first_row &&
          row < cached->first_row + cached->rows) {
        local_cache_lookup_us_total += elapsed_us(cache_lookup_start);
        stats_.cache_hits.fetch_add(1, std::memory_order_relaxed);
        ++local_cache_hit_count;
        VLOG(2) << "byte_range.strided.cache_hit run_index=" << run_index << " row=" << row
                << " block_first_row=" << cached->first_row << " block_rows=" << cached->rows;
        return cached;
      }
    }
    local_cache_lookup_us_total += elapsed_us(cache_lookup_start);

    stats_.cache_misses.fetch_add(1, std::memory_order_relaxed);
    ++local_cache_miss_count;
    const uint64_t rows_per_block = std::max<uint64_t>(run.rows_per_block, 1);
    const uint64_t block_first_row = row - (row % rows_per_block);
    uint64_t block_rows = std::min(rows_per_block, run.rows - block_first_row);

    while (block_rows > 0) {
      const auto block_prepare_start = std::chrono::steady_clock::now();
      const uint64_t block_bytes = (block_rows - 1) * run.stride + run.row_len;
      if (block_bytes > program_->strided_block_max_bytes) {
        local_block_prepare_us_total += elapsed_us(block_prepare_start);
        block_rows /= 2;
        continue;
      }
      std::shared_ptr<StridedBlock> block;
      bool reused_block = false;
      const auto block_pick_start = std::chrono::steady_clock::now();
      if (strided_cache_) {
        absl::MutexLock lock(&strided_cache_->mutex);
        if (strided_cache_->block && strided_cache_->block.use_count() == 1) {
          block = std::const_pointer_cast<StridedBlock>(strided_cache_->block);
          strided_cache_->block.reset();
          reused_block = true;
        }
      }
      if (!block) {
        block = std::make_shared<StridedBlock>();
      }
      local_block_pick_us_total += elapsed_us(block_pick_start);
      if (reused_block) {
        ++local_block_reuse_count;
      } else {
        ++local_block_new_count;
      }
      block->run_index = run_index;
      block->first_row = block_first_row;
      block->rows = block_rows;
      block->src_begin = run.src_base + block_first_row * run.stride;
      try {
        const auto block_resize_start = std::chrono::steady_clock::now();
        const size_t target_bytes = static_cast<size_t>(block_bytes);
        if (block->data.capacity() < target_bytes) {
          block->data.reserve(target_bytes);
        }
        if (block->data.size() != target_bytes) {
          block->data.resize(target_bytes);
        }
        local_block_resize_us_total += elapsed_us(block_resize_start);
      } catch (const std::bad_alloc&) {
        local_block_prepare_us_total += elapsed_us(block_prepare_start);
        VLOG(2) << "byte_range.strided.block_alloc_retry run_index=" << run_index
                << " block_first_row=" << block_first_row << " block_rows=" << block_rows
                << " block_bytes=" << block_bytes;
        block_rows /= 2;
        continue;
      }
      local_block_prepare_us_total += elapsed_us(block_prepare_start);
      const auto block_read_start = std::chrono::steady_clock::now();
      auto read_or = read_base(run.source_index, block->src_begin, block->data.data(), block->data.size());
      const uint64_t block_read_us = elapsed_us(block_read_start);
      local_block_load_us_total += block_read_us;
      if (!read_or.ok()) {
        VLOG(2) << "byte_range.strided.cache_miss_read_failed run_index=" << run_index
                << " block_first_row=" << block_first_row << " block_rows=" << block_rows
                << " block_bytes=" << block->data.size() << " duration_us=" << block_read_us
                << " status=" << read_or.status();
        return read_or.status();
      }
      VLOG(2) << "byte_range.strided.cache_miss_loaded run_index=" << run_index
              << " block_first_row=" << block_first_row << " block_rows=" << block_rows
              << " block_bytes=" << block->data.size() << " duration_us=" << block_read_us;
      ++local_block_load_count;
      local_block_bytes_total += block->data.size();
      const double load_mib = static_cast<double>(block->data.size()) / (1024.0 * 1024.0);
      const double load_seconds = static_cast<double>(block_read_us) / 1e6;
      const double throughput_mib_s = load_seconds > 0.0 ? (load_mib / load_seconds) : 0.0;
      VLOG(2) << "byte_range.strided.block_summary run_index=" << run_index << " source_index=" << run.source_index
              << " block_first_row=" << block_first_row << " block_rows=" << block_rows
              << " block_bytes=" << block->data.size() << " reused_block=" << (reused_block ? 1 : 0)
              << " block_pick_us=" << local_block_pick_us_total << " block_resize_us=" << local_block_resize_us_total
              << " block_prepare_us=" << local_block_prepare_us_total << " block_load_us=" << block_read_us
              << " block_load_throughput_mib_s=" << throughput_mib_s;
      const std::shared_ptr<const StridedBlock> immutable_block = block;
      if (strided_cache_) {
        absl::MutexLock lock(&strided_cache_->mutex);
        strided_cache_->block = immutable_block;
      }
      return immutable_block;
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
        const auto fallback_start = std::chrono::steady_clock::now();
        auto fallback_or = copy_from_strided_rows(run, local_offset, out, remaining);
        const uint64_t fallback_us = elapsed_us(fallback_start);
        local_row_copy_us_total += fallback_us;
        if (!fallback_or.ok()) {
          VLOG(2) << "byte_range.strided.fallback_failed run_index=" << run_index
                  << " already_copied=" << already_copied << " remaining=" << remaining
                  << " duration_us=" << fallback_us << " status=" << fallback_or.status();
          flush_local_stats();
          return fallback_or.status();
        }
        local_row_copy_bytes_total += *fallback_or;
        VLOG(2) << "byte_range.strided.fallback_done run_index=" << run_index << " already_copied=" << already_copied
                << " fallback_copied=" << *fallback_or << " duration_us=" << fallback_us;
        flush_local_stats();
        return already_copied + *fallback_or;
      }
      VLOG(2) << "byte_range.strided.block_load_failed run_index=" << run_index << " local_offset=" << local_offset
              << " remaining=" << remaining << " status=" << block_or.status();
      flush_local_stats();
      return block_or.status();
    }
    const auto& block = *block_or;
    const uint64_t block_end_row = block->first_row + block->rows;
    uint64_t active_row = row;
    const auto pack_start = std::chrono::steady_clock::now();
    size_t packed_bytes = 0;
    while (remaining > 0 && active_row < block_end_row) {
      const size_t available = static_cast<size_t>(run.row_len - row_offset);
      const size_t take = std::min(remaining, available);
      const uint64_t block_offset = (active_row - block->first_row) * run.stride + row_offset;
      std::memcpy(out, block->data.data() + block_offset, take);
      ++local_pack_memcpy_calls;
      stats_.pack_bytes.fetch_add(take, std::memory_order_relaxed);
      packed_bytes += take;
      out += take;
      remaining -= take;
      local_offset += take;
      row_offset += take;
      if (row_offset == run.row_len) {
        ++active_row;
        row_offset = 0;
      }
    }
    VLOG(2) << "byte_range.strided.pack_block run_index=" << run_index << " block_first_row=" << block->first_row
            << " block_rows=" << block->rows << " packed_bytes=" << packed_bytes
            << " duration_us=" << elapsed_us(pack_start);
    local_pack_us_total += elapsed_us(pack_start);
    local_pack_bytes_total += packed_bytes;
  }

  VLOG(2) << "byte_range.strided.run_summary run_index=" << run_index << " source_index=" << run.source_index
          << " requested_bytes=" << bytes << " copied_bytes=" << bytes << " cache_hits=" << local_cache_hit_count
          << " cache_misses=" << local_cache_miss_count << " blocks_loaded=" << local_block_load_count
          << " blocks_reused=" << local_block_reuse_count << " blocks_new=" << local_block_new_count
          << " block_bytes_total=" << local_block_bytes_total << " block_pick_us_total=" << local_block_pick_us_total
          << " block_resize_us_total=" << local_block_resize_us_total
          << " block_prepare_us_total=" << local_block_prepare_us_total
          << " block_load_us_total=" << local_block_load_us_total << " pack_us_total=" << local_pack_us_total
          << " pack_memcpy_calls=" << local_pack_memcpy_calls << " pack_bytes_total=" << local_pack_bytes_total
          << " row_copy_us_total=" << local_row_copy_us_total << " total_us=" << elapsed_us(run_start);

  VLOG(2) << "byte_range.strided.end run_index=" << run_index << " copied=" << bytes
          << " duration_us=" << elapsed_us(run_start);
  flush_local_stats();
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
  const auto read_start = std::chrono::steady_clock::now();
  uint64_t pad_us_total = 0;
  uint64_t contiguous_us_total = 0;
  uint64_t strided_us_total = 0;
  uint64_t strided_pack_us_total = 0;
  uint64_t strided_cache_lookup_us_total = 0;
  uint64_t strided_block_prepare_us_total = 0;
  uint64_t strided_block_load_us_total = 0;
  uint64_t strided_row_copy_us_total = 0;
  size_t pad_bytes_total = 0;
  size_t contiguous_bytes_total = 0;
  size_t strided_bytes_total = 0;
  size_t strided_pack_bytes_total = 0;
  size_t strided_row_copy_bytes_total = 0;
  if (offset >= program_->total_bytes || bytes == 0) {
    VLOG(2) << "byte_range.read_at trivial offset=" << offset << " bytes=" << bytes
            << " total_bytes=" << program_->total_bytes << " copied=0 duration_us=0";
    return static_cast<size_t>(0);
  }
  const uint64_t remaining_bytes = program_->total_bytes - offset;
  size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, remaining_bytes));
  uint8_t* out = static_cast<uint8_t*>(dst);
  size_t copied = 0;
  uint64_t cursor = offset;

  size_t run_index = find_run_index(offset);
  VLOG(2) << "byte_range.read_at.begin offset=" << offset << " req_bytes=" << bytes << " to_copy=" << to_copy
          << " run_index=" << run_index << " total_runs=" << program_->runs.size();
  while (run_index < program_->runs.size() && copied < to_copy) {
    const auto& run = program_->runs[run_index];
    const auto run_start = std::chrono::steady_clock::now();
    if (cursor >= run.dst_end) {
      VLOG(2) << "byte_range.read_at.skip run_index=" << run_index << " kind=" << run_kind_to_cstr(run.kind)
              << " cursor=" << cursor << " dst_end=" << run.dst_end;
      ++run_index;
      continue;
    }
    if (cursor < run.dst_begin) {
      VLOG(2) << "byte_range.read_at.uncovered_gap run_index=" << run_index << " cursor=" << cursor
              << " dst_begin=" << run.dst_begin << " duration_us=" << elapsed_us(run_start);
      return absl::InternalError("byte range program contains uncovered gaps");
    }
    const uint64_t run_offset = cursor - run.dst_begin;
    const size_t available = static_cast<size_t>(run.dst_end - cursor);
    const size_t chunk = std::min(to_copy - copied, available);
    VLOG(2) << "byte_range.read_at.run_begin run_index=" << run_index << " kind=" << run_kind_to_cstr(run.kind)
            << " source_index=" << run.source_index << " run_offset=" << run_offset << " chunk=" << chunk
            << " dst_begin=" << run.dst_begin << " dst_end=" << run.dst_end;
    absl::StatusOr<size_t> copied_or;
    const auto kind_start = std::chrono::steady_clock::now();
    switch (run.kind) {
      case ByteRangeRun::Kind::kPad:
        std::memset(out + copied, 0, chunk);
        stats_.pad_bytes.fetch_add(chunk, std::memory_order_relaxed);
        copied_or = chunk;
        break;
      case ByteRangeRun::Kind::kContiguous:
        copied_or = read_base(run.source_index, run.src_begin + run_offset, out + copied, chunk);
        break;
      case ByteRangeRun::Kind::kStrided: {
        uint64_t strided_pack_us = 0;
        size_t strided_pack_bytes = 0;
        uint64_t strided_cache_lookup_us = 0;
        uint64_t strided_block_prepare_us = 0;
        uint64_t strided_block_load_us = 0;
        uint64_t strided_row_copy_us = 0;
        size_t strided_row_copy_bytes = 0;
        copied_or = fill_strided_run(
            run_index,
            run,
            run_offset,
            out + copied,
            chunk,
            &strided_pack_us,
            &strided_pack_bytes,
            &strided_cache_lookup_us,
            &strided_block_prepare_us,
            &strided_block_load_us,
            &strided_row_copy_us,
            &strided_row_copy_bytes);
        strided_pack_us_total += strided_pack_us;
        strided_pack_bytes_total += strided_pack_bytes;
        strided_cache_lookup_us_total += strided_cache_lookup_us;
        strided_block_prepare_us_total += strided_block_prepare_us;
        strided_block_load_us_total += strided_block_load_us;
        strided_row_copy_us_total += strided_row_copy_us;
        strided_row_copy_bytes_total += strided_row_copy_bytes;
        break;
      }
    }
    const uint64_t kind_us = elapsed_us(kind_start);
    switch (run.kind) {
      case ByteRangeRun::Kind::kPad:
        pad_us_total += kind_us;
        break;
      case ByteRangeRun::Kind::kContiguous:
        contiguous_us_total += kind_us;
        break;
      case ByteRangeRun::Kind::kStrided:
        strided_us_total += kind_us;
        break;
    }
    if (!copied_or.ok()) {
      VLOG(2) << "byte_range.read_at.run_failed run_index=" << run_index << " kind=" << run_kind_to_cstr(run.kind)
              << " chunk=" << chunk << " duration_us=" << elapsed_us(run_start) << " status=" << copied_or.status();
      return copied_or.status();
    }
    VLOG(2) << "byte_range.read_at.run_end run_index=" << run_index << " kind=" << run_kind_to_cstr(run.kind)
            << " copied_chunk=" << *copied_or << " duration_us=" << elapsed_us(run_start);
    switch (run.kind) {
      case ByteRangeRun::Kind::kPad:
        pad_bytes_total += *copied_or;
        break;
      case ByteRangeRun::Kind::kContiguous:
        contiguous_bytes_total += *copied_or;
        break;
      case ByteRangeRun::Kind::kStrided:
        strided_bytes_total += *copied_or;
        break;
    }
    copied += *copied_or;
    cursor += *copied_or;
    if (*copied_or == 0) {
      VLOG(2) << "byte_range.read_at.stop_zero_progress run_index=" << run_index;
      break;
    }
    ++run_index;
  }

  stats_.output_bytes.fetch_add(copied, std::memory_order_relaxed);
  VLOG(2) << "byte_range.read_at.end offset=" << offset << " req_bytes=" << bytes << " to_copy=" << to_copy
          << " copied=" << copied << " duration_us=" << elapsed_us(read_start);
  VLOG(2) << "byte_range.read_at.summary offset=" << offset << " req_bytes=" << bytes << " copied=" << copied
          << " duration_us=" << elapsed_us(read_start) << " pad_us_total=" << pad_us_total
          << " contiguous_us_total=" << contiguous_us_total << " strided_us_total=" << strided_us_total
          << " strided_pack_us_total=" << strided_pack_us_total
          << " strided_cache_lookup_us_total=" << strided_cache_lookup_us_total
          << " strided_block_prepare_us_total=" << strided_block_prepare_us_total
          << " strided_block_load_us_total=" << strided_block_load_us_total
          << " strided_row_copy_us_total=" << strided_row_copy_us_total << " pad_bytes_total=" << pad_bytes_total
          << " contiguous_bytes_total=" << contiguous_bytes_total << " strided_bytes_total=" << strided_bytes_total
          << " strided_pack_bytes_total=" << strided_pack_bytes_total
          << " strided_row_copy_bytes_total=" << strided_row_copy_bytes_total;
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
