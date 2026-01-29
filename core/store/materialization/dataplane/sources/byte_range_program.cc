// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/byte_range_program.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/contracts/byte_range/byte_range_fingerprint_encoding.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::loader {

namespace {

constexpr std::string_view kConfigFingerprintVersion = "byte_range_config_v1";

using byte_range_fingerprint_internal::append_u32;
using byte_range_fingerprint_internal::append_u64;

bool amplification_within_limit(uint64_t stride, uint64_t row_len, uint32_t max_amplification) {
  if (row_len == 0 || max_amplification == 0) {
    return false;
  }
  if (row_len > std::numeric_limits<uint64_t>::max() / max_amplification) {
    return false;
  }
  return stride <= static_cast<uint64_t>(max_amplification) * row_len;
}

bool should_enable_strided(const ByteRangeRun& run, const StoreEngineOptions::ByteMappingConfig& cfg) {
  if (run.rows < cfg.strided_run_min_ranges) {
    return false;
  }
  if (run.row_len < cfg.strided_min_row_len_bytes) {
    return false;
  }
  if (run.row_len > cfg.strided_block_max_bytes) {
    return false;
  }
  return amplification_within_limit(run.stride, run.row_len, cfg.strided_max_amplification);
}

uint64_t compute_rows_per_block(const ByteRangeRun& run, const StoreEngineOptions::ByteMappingConfig& cfg) {
  if (run.stride == 0) {
    return 1;
  }
  uint64_t rows_per_block = cfg.strided_block_target_bytes / run.stride;
  if (rows_per_block == 0) {
    rows_per_block = 1;
  }
  uint64_t max_rows = 1;
  if (cfg.strided_block_max_bytes > run.row_len) {
    max_rows = 1 + (cfg.strided_block_max_bytes - run.row_len) / run.stride;
  }
  if (rows_per_block > max_rows) {
    rows_per_block = max_rows;
  }
  return std::max<uint64_t>(rows_per_block, 1);
}

struct CacheKey {
  ByteRangeFingerprint map_fingerprint;
  ByteRangeFingerprint config_fingerprint;

  bool operator==(const CacheKey& other) const = default;
};

template <typename H>
H AbslHashValue(H h, const CacheKey& key) {
  return H::combine(std::move(h), key.map_fingerprint, key.config_fingerprint);
}

struct CacheEntry {
  CacheKey key;
  std::shared_ptr<const ByteRangeProgram> program;
  uint64_t total_bytes{0};
  uint32_t num_sources{0};
  size_t segment_count{0};
  std::list<CacheKey>::iterator lru_it;
};

struct CacheMetricsHandles {
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> meter;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> hits;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> misses;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> evictions;
};

CacheMetricsHandles init_cache_metrics() {
  CacheMetricsHandles handles;
  handles.meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  handles.hits = handles.meter->CreateDoubleCounter("tc_byte_range_program_cache_hits_total");
  handles.misses = handles.meter->CreateDoubleCounter("tc_byte_range_program_cache_misses_total");
  handles.evictions = handles.meter->CreateDoubleCounter("tc_byte_range_program_cache_evictions_total");
  return handles;
}

CacheMetricsHandles& cache_metrics() {
  static CacheMetricsHandles handles = init_cache_metrics();
  return handles;
}

void record_cache_counter(
    const opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>>& counter,
    std::string_view path) {
  try {
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("path", opentelemetry::common::AttributeValue(std::string(path)));
    counter->Add(1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
    VLOG(2) << "metrics counter tc_byte_range_program_cache_* unavailable";
  }
}

class ByteRangeProgramCache {
 public:
  explicit ByteRangeProgramCache(size_t capacity) : capacity_(std::max<size_t>(capacity, 1)) {}

  void SetCapacity(size_t capacity) {
    absl::MutexLock lock(&mutex_);
    capacity_ = std::max<size_t>(capacity, 1);
    evict_if_needed();
  }

  std::shared_ptr<const ByteRangeProgram> Lookup(const CacheKey& key, const ByteRangeMap& map, std::string_view path) {
    absl::MutexLock lock(&mutex_);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      record_cache_counter(cache_metrics().misses, path);
      return nullptr;
    }
    const auto& entry = it->second;
    if (entry.total_bytes != map.total_bytes || entry.num_sources != map.num_sources ||
        entry.segment_count != map.segments.size()) {
      LOG(ERROR) << "ByteRangeProgram cache key collision detected; evicting entry";
      lru_.erase(entry.lru_it);
      entries_.erase(it);
      record_cache_counter(cache_metrics().misses, path);
      return nullptr;
    }
    lru_.splice(lru_.begin(), lru_, entry.lru_it);
    record_cache_counter(cache_metrics().hits, path);
    return entry.program;
  }

  void Insert(
      const CacheKey& key,
      const ByteRangeMap& map,
      std::shared_ptr<const ByteRangeProgram> program,
      std::string_view path) {
    absl::MutexLock lock(&mutex_);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      it->second.program = std::move(program);
      it->second.total_bytes = map.total_bytes;
      it->second.num_sources = map.num_sources;
      it->second.segment_count = map.segments.size();
      lru_.splice(lru_.begin(), lru_, it->second.lru_it);
      return;
    }
    lru_.push_front(key);
    CacheEntry entry{
        .key = key,
        .program = std::move(program),
        .total_bytes = map.total_bytes,
        .num_sources = map.num_sources,
        .segment_count = map.segments.size(),
        .lru_it = lru_.begin(),
    };
    entries_.emplace(key, std::move(entry));
    evict_if_needed(path);
  }

 private:
  void evict_if_needed(std::string_view path = "") ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    while (entries_.size() > capacity_ && !lru_.empty()) {
      auto last_it = std::prev(lru_.end());
      auto entry_it = entries_.find(*last_it);
      if (entry_it != entries_.end()) {
        entries_.erase(entry_it);
      }
      lru_.erase(last_it);
      if (!path.empty()) {
        record_cache_counter(cache_metrics().evictions, path);
      }
    }
  }

  size_t capacity_{1};
  absl::Mutex mutex_;
  std::list<CacheKey> lru_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<CacheKey, CacheEntry> entries_ ABSL_GUARDED_BY(mutex_);
};

ByteRangeProgramCache& program_cache() {
  static ByteRangeProgramCache cache(/*capacity=*/256);
  return cache;
}

} // namespace

ByteRangeFingerprint fingerprint_byte_range_config(const StoreEngineOptions::ByteMappingConfig& config) {
  std::string payload;
  payload.reserve(64);
  payload.append(kConfigFingerprintVersion);
  payload.push_back('\0');
  payload.push_back(config.enable_strided_execution ? '\1' : '\0');
  append_u32(&payload, config.strided_run_min_ranges);
  append_u64(&payload, config.strided_min_row_len_bytes);
  append_u32(&payload, config.strided_max_amplification);
  append_u64(&payload, config.strided_block_target_bytes);
  append_u64(&payload, config.strided_block_max_bytes);

  std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  ByteRangeFingerprint fp;
  std::memcpy(fp.bytes.data(), digest.data(), fp.bytes.size());
  return fp;
}

ByteRangeCompiler::ByteRangeCompiler(StoreEngineOptions::ByteMappingConfig config, std::string path)
    : config_(std::move(config)), config_fingerprint_(fingerprint_byte_range_config(config_)), path_(std::move(path)) {}

absl::StatusOr<std::shared_ptr<const ByteRangeProgram>> ByteRangeCompiler::Compile(ByteRangeMap map) {
  auto normalized_or = normalize_byte_range_map(std::move(map));
  if (!normalized_or.ok()) {
    return normalized_or.status();
  }
  ByteRangeMap normalized = std::move(*normalized_or);
  auto map_fp_or = fingerprint_byte_range_map(normalized);
  if (!map_fp_or.ok()) {
    return map_fp_or.status();
  }
  const CacheKey key{*map_fp_or, config_fingerprint_};

  auto& cache = program_cache();
  cache.SetCapacity(config_.program_cache_entries);
  if (auto cached = cache.Lookup(key, normalized, path_)) {
    return cached;
  }

  auto program = std::make_shared<ByteRangeProgram>();
  program->total_bytes = normalized.total_bytes;
  program->map_fingerprint = *map_fp_or;
  program->config_fingerprint = config_fingerprint_;
  program->strided_block_max_bytes = config_.strided_block_max_bytes;

  const auto& segments = normalized.segments;
  size_t idx = 0;
  while (idx < segments.size()) {
    const auto& seg = segments[idx];
    if (seg.length == 0) {
      ++idx;
      continue;
    }
    if (seg.kind == ByteRangeSegment::Kind::kPad) {
      uint64_t dst_begin = seg.dst_offset;
      uint64_t dst_end = seg.dst_offset + seg.length;
      size_t cursor = idx + 1;
      while (cursor < segments.size()) {
        const auto& next = segments[cursor];
        if (next.kind != ByteRangeSegment::Kind::kPad || next.length == 0 || next.dst_offset != dst_end) {
          break;
        }
        dst_end += next.length;
        ++cursor;
      }
      ByteRangeRun run;
      run.kind = ByteRangeRun::Kind::kPad;
      run.dst_begin = dst_begin;
      run.dst_end = dst_end;
      program->runs.push_back(run);
      idx = cursor;
      continue;
    }

    bool strided_candidate = false;
    uint64_t stride = 0;
    if (idx + 1 < segments.size()) {
      const auto& next = segments[idx + 1];
      if (next.kind == ByteRangeSegment::Kind::kData && next.source_index == seg.source_index &&
          next.length == seg.length && next.dst_offset == seg.dst_offset + seg.length &&
          next.src_offset > seg.src_offset) {
        stride = next.src_offset - seg.src_offset;
        strided_candidate = stride > seg.length;
      }
    }

    if (strided_candidate) {
      program->strided_candidate_runs += 1;
      const size_t start = idx;
      const uint64_t row_len = seg.length;
      const uint64_t dst_begin = seg.dst_offset;
      const uint64_t src_base = seg.src_offset;
      uint64_t expected_dst = dst_begin;
      uint64_t expected_src = src_base;
      size_t cursor = idx;
      while (cursor < segments.size()) {
        const auto& current = segments[cursor];
        if (current.kind != ByteRangeSegment::Kind::kData || current.source_index != seg.source_index ||
            current.length != row_len || current.dst_offset != expected_dst || current.src_offset != expected_src) {
          break;
        }
        expected_dst += row_len;
        expected_src += stride;
        ++cursor;
      }
      const uint64_t rows = static_cast<uint64_t>(cursor - start);
      ByteRangeRun run;
      run.kind = ByteRangeRun::Kind::kStrided;
      run.dst_begin = dst_begin;
      run.dst_end = dst_begin + (row_len * rows);
      run.source_index = seg.source_index;
      run.src_base = src_base;
      run.row_len = row_len;
      run.stride = stride;
      run.rows = rows;

      const bool enable_strided = config_.enable_strided_execution && should_enable_strided(run, config_);
      if (enable_strided) {
        run.rows_per_block = compute_rows_per_block(run, config_);
        program->runs.push_back(run);
        program->has_strided_runs = true;
      } else {
        program->strided_compile_fallback_runs += 1;
        for (size_t j = start; j < cursor; ++j) {
          const auto& piece = segments[j];
          ByteRangeRun fallback;
          fallback.kind = ByteRangeRun::Kind::kContiguous;
          fallback.dst_begin = piece.dst_offset;
          fallback.dst_end = piece.dst_offset + piece.length;
          fallback.source_index = piece.source_index;
          fallback.src_begin = piece.src_offset;
          program->runs.push_back(fallback);
        }
      }
      idx = cursor;
      continue;
    }

    uint64_t dst_begin = seg.dst_offset;
    uint64_t dst_end = seg.dst_offset + seg.length;
    uint64_t src_begin = seg.src_offset;
    uint64_t src_end = src_begin + seg.length;
    size_t cursor = idx + 1;
    while (cursor < segments.size()) {
      const auto& next = segments[cursor];
      if (next.kind != ByteRangeSegment::Kind::kData || next.source_index != seg.source_index ||
          next.dst_offset != dst_end || next.src_offset != src_end) {
        break;
      }
      dst_end += next.length;
      src_end += next.length;
      ++cursor;
    }
    ByteRangeRun run;
    run.kind = ByteRangeRun::Kind::kContiguous;
    run.dst_begin = dst_begin;
    run.dst_end = dst_end;
    run.source_index = seg.source_index;
    run.src_begin = src_begin;
    program->runs.push_back(run);
    idx = cursor;
  }

  program->run_starts.reserve(program->runs.size());
  for (const auto& run : program->runs) {
    program->run_starts.push_back(run.dst_begin);
  }

  cache.Insert(key, normalized, program, path_);
  return program;
}

} // namespace tensorcast::store::loader
