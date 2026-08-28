// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_layout_utils.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon::materialization_layout {

namespace {

constexpr size_t kCanonicalIndexParseCacheMaxEntries = 16;
constexpr size_t kCanonicalIndexCacheSampleBytes = 256;

struct InflightCanonicalIndexParse {
  absl::Mutex mu;
  bool done ABSL_GUARDED_BY(mu){false};
  absl::Status status ABSL_GUARDED_BY(mu);
  std::shared_ptr<const CanonicalIndexTable> table ABSL_GUARDED_BY(mu);
  size_t byte_size ABSL_GUARDED_BY(mu){0};
  std::string prefix_sample ABSL_GUARDED_BY(mu);
  std::string suffix_sample ABSL_GUARDED_BY(mu);
};

struct CachedCanonicalIndexParse {
  std::shared_ptr<const CanonicalIndexTable> table;
  size_t byte_size{0};
  std::string prefix_sample;
  std::string suffix_sample;
};

struct CanonicalIndexParseCache {
  absl::Mutex mu;
  absl::flat_hash_map<std::string, CachedCanonicalIndexParse> entries;
  absl::flat_hash_map<std::string, std::shared_ptr<InflightCanonicalIndexParse>> inflight;
  std::deque<std::string> insertion_order;
};

CanonicalIndexParseCache& canonical_index_parse_cache() {
  static auto* cache = new CanonicalIndexParseCache();
  return *cache;
}

std::string canonical_index_parse_cache_key(std::string_view index_json) {
  const std::vector<uint8_t> digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(index_json.data()), index_json.size()));
  return absl::StrCat(index_json.size(), ":", common::multibase_multihash_sha256(digest));
}

std::string canonical_index_identity_parse_cache_key(std::string_view index_json, std::string_view identity_key) {
  (void)index_json;
  return absl::StrCat("identity:", identity_key);
}

std::string index_prefix_sample(std::string_view index_json) {
  return std::string(index_json.substr(0, std::min(index_json.size(), kCanonicalIndexCacheSampleBytes)));
}

std::string index_suffix_sample(std::string_view index_json) {
  const size_t count = std::min(index_json.size(), kCanonicalIndexCacheSampleBytes);
  return std::string(index_json.substr(index_json.size() - count, count));
}

bool sample_matches(
    std::string_view index_json,
    size_t byte_size,
    std::string_view prefix_sample,
    std::string_view suffix_sample) {
  return index_json.size() == byte_size && index_prefix_sample(index_json) == prefix_sample &&
      index_suffix_sample(index_json) == suffix_sample;
}

absl::Status identity_cache_mismatch_status() {
  return absl::FailedPreconditionError("canonical index identity cache key was reused for different index bytes");
}

absl::StatusOr<CanonicalIndexTable> parse_canonical_index_uncached(std::string_view index_json) {
  if (index_json.empty()) {
    return absl::InvalidArgumentError("canonical index JSON is empty");
  }
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(index_json, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON: ", e.what()));
  }
  CanonicalIndexTable table;
  for (auto it = j.begin(); it != j.end(); ++it) {
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() != 6) {
      return absl::InvalidArgumentError("Invalid canonical index entry");
    }
    CanonicalIndexEntry entry;
    entry.logical_offset = arr[0].get<uint64_t>();
    entry.logical_length = arr[1].get<uint64_t>();
    entry.shape.reserve(arr[2].size());
    for (const auto& dim : arr[2]) {
      entry.shape.push_back(dim.get<int64_t>());
    }
    entry.stride.reserve(arr[3].size());
    for (const auto& dim : arr[3]) {
      entry.stride.push_back(dim.get<int64_t>());
    }
    entry.dtype = arr[4].get<std::string>();
    entry.storage_offset = arr[5].get<uint64_t>();
    table.logical_total_size =
        std::max<uint64_t>(table.logical_total_size, entry.logical_offset + entry.logical_length);
    table.entries.emplace(it.key(), std::move(entry));
  }
  return table;
}

absl::StatusOr<std::shared_ptr<const CanonicalIndexTable>> parse_canonical_index_shared_with_key(
    std::string_view index_json,
    std::string key) {
  if (index_json.empty()) {
    return absl::InvalidArgumentError("canonical index JSON is empty");
  }
  auto& cache = canonical_index_parse_cache();
  std::shared_ptr<InflightCanonicalIndexParse> inflight;
  bool is_leader = false;
  {
    absl::MutexLock lock(&cache.mu);
    if (auto it = cache.entries.find(key); it != cache.entries.end()) {
      if (!sample_matches(index_json, it->second.byte_size, it->second.prefix_sample, it->second.suffix_sample)) {
        return identity_cache_mismatch_status();
      }
      return it->second.table;
    }
    auto [it, inserted] = cache.inflight.try_emplace(key, std::make_shared<InflightCanonicalIndexParse>());
    inflight = it->second;
    is_leader = inserted;
  }

  if (!is_leader) {
    absl::MutexLock wait_lock(&inflight->mu);
    while (!inflight->done) {
      inflight->mu.Await(absl::Condition(&inflight->done));
    }
    if (!inflight->status.ok()) {
      return inflight->status;
    }
    if (!sample_matches(index_json, inflight->byte_size, inflight->prefix_sample, inflight->suffix_sample)) {
      return identity_cache_mismatch_status();
    }
    return inflight->table;
  }

  auto table_or = parse_canonical_index_uncached(index_json);
  if (!table_or.ok()) {
    const absl::Status status = table_or.status();
    {
      absl::MutexLock wait_lock(&inflight->mu);
      inflight->status = status;
      inflight->done = true;
    }
    {
      absl::MutexLock lock(&cache.mu);
      cache.inflight.erase(key);
    }
    return status;
  }
  auto table = std::make_shared<const CanonicalIndexTable>(std::move(*table_or));
  const size_t byte_size = index_json.size();
  const std::string prefix_sample = index_prefix_sample(index_json);
  const std::string suffix_sample = index_suffix_sample(index_json);
  {
    absl::MutexLock wait_lock(&inflight->mu);
    inflight->status = absl::OkStatus();
    inflight->table = table;
    inflight->byte_size = byte_size;
    inflight->prefix_sample = prefix_sample;
    inflight->suffix_sample = suffix_sample;
    inflight->done = true;
  }
  {
    absl::MutexLock lock(&cache.mu);
    cache.inflight.erase(key);
    if (!cache.entries.contains(key)) {
      cache.entries.emplace(
          key,
          CachedCanonicalIndexParse{
              .table = table,
              .byte_size = byte_size,
              .prefix_sample = prefix_sample,
              .suffix_sample = suffix_sample,
          });
      cache.insertion_order.push_back(key);
      while (cache.entries.size() > kCanonicalIndexParseCacheMaxEntries && !cache.insertion_order.empty()) {
        cache.entries.erase(cache.insertion_order.front());
        cache.insertion_order.pop_front();
      }
    }
  }
  return table;
}

} // namespace

absl::StatusOr<std::shared_ptr<const CanonicalIndexTable>> parse_canonical_index_shared(std::string_view index_json) {
  if (index_json.empty()) {
    return absl::InvalidArgumentError("canonical index JSON is empty");
  }
  return parse_canonical_index_shared_with_key(index_json, canonical_index_parse_cache_key(index_json));
}

absl::StatusOr<std::shared_ptr<const CanonicalIndexTable>> parse_canonical_index_shared_with_identity(
    std::string_view index_json,
    std::string_view identity_key) {
  if (identity_key.empty()) {
    return parse_canonical_index_shared(index_json);
  }
  if (index_json.empty()) {
    return absl::InvalidArgumentError("canonical index JSON is empty");
  }
  return parse_canonical_index_shared_with_key(
      index_json, canonical_index_identity_parse_cache_key(index_json, identity_key));
}

absl::StatusOr<CanonicalIndexTable> parse_canonical_index(std::string_view index_json) {
  auto table_or = parse_canonical_index_shared(index_json);
  if (!table_or.ok()) {
    return table_or.status();
  }
  return **table_or;
}

absl::StatusOr<uint64_t> dtype_element_size(std::string_view dtype) {
  static const absl::flat_hash_map<std::string_view, uint64_t> kSizeMap = {
      {"torch.float16", 2},
      {"torch.bfloat16", 2},
      {"torch.float32", 4},
      {"torch.float64", 8},
      {"torch.float8_e4m3fn", 1},
      {"torch.float8_e5m2", 1},
      {"torch.float8_e4m3fnuz", 1},
      {"torch.float8_e5m2fnuz", 1},
      {"torch.float8_e8m0fnu", 1},
      {"torch.int8", 1},
      {"torch.uint8", 1},
      {"torch.int16", 2},
      {"torch.int32", 4},
      {"torch.int64", 8},
      {"torch.bool", 1},
      {"torch.float", 4},
      {"torch.double", 8},
  };
  auto it = kSizeMap.find(dtype);
  if (it == kSizeMap.end()) {
    return absl::InvalidArgumentError(absl::StrCat("unsupported dtype: ", dtype));
  }
  return it->second;
}

absl::StatusOr<uint64_t> product_dims(absl::Span<const int64_t> dims) {
  uint64_t acc = 1;
  for (int64_t dim : dims) {
    if (dim <= 0) {
      return absl::InvalidArgumentError("shape dims must be positive");
    }
    if (acc > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
      return absl::OutOfRangeError("shape size overflow");
    }
    acc *= static_cast<uint64_t>(dim);
  }
  return acc;
}

absl::StatusOr<std::vector<TargetOffsetEntry>> resolve_target_offsets(const v2::TargetLayout& layout) {
  std::vector<TargetOffsetEntry> offsets;
  offsets.reserve(layout.offsets_size() + layout.aliases_size());
  if (layout.tensor_spec_kind() == v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS) {
    for (const auto& entry : layout.offsets()) {
      TargetOffsetEntry resolved;
      resolved.name = entry.name();
      resolved.storage_id = entry.storage_id();
      resolved.storage_offset = entry.storage_offset();
      resolved.logical_length = entry.logical_length();
      offsets.push_back(std::move(resolved));
    }
    return offsets;
  }
  if (layout.tensor_spec_kind() == v2::TargetLayout::TENSOR_SPEC_KIND_ALIAS_UNSPECIFIED) {
    for (const auto& entry : layout.aliases()) {
      TargetOffsetEntry resolved;
      resolved.name = entry.name();
      resolved.storage_id = entry.storage_id();
      resolved.storage_offset = entry.storage_offset();
      resolved.logical_length = entry.logical_length();
      offsets.push_back(std::move(resolved));
    }
    return offsets;
  }
  return absl::InvalidArgumentError("Unsupported tensor_spec_kind for target layout");
}

} // namespace tensorcast::daemon::materialization_layout
