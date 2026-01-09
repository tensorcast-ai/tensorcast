// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/segment_plan_source.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/cuda/cuda_api.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::loader {

using nlohmann::json;

absl::StatusOr<std::vector<SegmentPiece>> build_segment_plan_from_canonical_index_json(
    std::string_view index_json,
    uint64_t total_size,
    uint64_t /*align_bytes*/) {
  if (total_size == 0) {
    return absl::InvalidArgumentError("total_size must be > 0");
  }
  if (index_json.empty()) {
    return absl::InvalidArgumentError("index_json must not be empty");
  }
  json j;
  try {
    j = json::parse(index_json, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON: ", e.what()));
  }
  // Collect (offset, size)
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  ranges.reserve(j.size());
  for (auto it = j.begin(); it != j.end(); ++it) {
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() < 2) {
      continue;
    }
    uint64_t off = arr[0].get<uint64_t>();
    uint64_t sz = arr[1].get<uint64_t>();
    ranges.emplace_back(off, sz);
  }
  std::sort(ranges.begin(), ranges.end(), [](auto& a, auto& b) { return a.first < b.first; });

  std::vector<SegmentPiece> plan;
  plan.reserve(ranges.size() * 2 + 1);
  uint64_t cur = 0;
  for (const auto& [off, sz] : ranges) {
    if (off > cur) {
      // PAD gap
      plan.push_back(SegmentPiece{SegmentPiece::PAD, cur, off - cur, 0});
      cur = off;
    }
    if (sz > 0) {
      plan.push_back(SegmentPiece{SegmentPiece::DATA, off, sz, off});
      cur = off + sz;
    }
  }
  if (cur < total_size) {
    plan.push_back(SegmentPiece{SegmentPiece::PAD, cur, total_size - cur, 0});
  }
  return plan;
}

LinearizedGpuPlanSource::LinearizedGpuPlanSource(
    gsl::not_null<void*> device_ptr,
    int device_id,
    absl::Span<const SegmentPiece> plan,
    uint64_t total_size)
    : device_ptr_(device_ptr), device_id_(device_id), plan_(plan.begin(), plan.end()), total_size_(total_size) {}

absl::StatusOr<size_t> LinearizedGpuPlanSource::read(void* dst, size_t max_bytes) {
  auto st = read_at(current_offset_, dst, max_bytes);
  if (!st.ok())
    return st;
  current_offset_ += *st;
  return st;
}

absl::StatusOr<size_t> LinearizedGpuPlanSource::read_at(uint64_t offset, void* dst, size_t bytes) {
  if (offset >= total_size_)
    return static_cast<size_t>(0);
  size_t remaining = static_cast<size_t>(std::min<uint64_t>(bytes, total_size_ - offset));
  uint8_t* out = static_cast<uint8_t*>(dst);

  // Find starting piece index by linear scan (plans are modest in size)
  // Optimization: cache could be added later if needed.
  size_t idx = 0;
  uint64_t piece_start = 0;
  for (; idx < plan_.size(); ++idx) {
    const auto& p = plan_[idx];
    if (offset < p.dst_offset + p.length) {
      piece_start = p.dst_offset;
      break;
    }
  }
  if (idx == plan_.size()) {
    // Offset beyond plan range
    return static_cast<size_t>(0);
  }

  while (remaining > 0 && idx < plan_.size()) {
    const auto& p = plan_[idx];
    const uint64_t local = offset - p.dst_offset;
    const size_t avail = static_cast<size_t>(p.length - local);
    const size_t take = std::min(remaining, avail);
    if (p.kind == SegmentPiece::PAD) {
      std::memset(out, 0, take);
    } else {
      // Copy from GPU: device_ptr_ + p.src_offset + local → out
      if (auto st = tensorcast::cuda::set_device(device_id_); !st.ok())
        return st;
      auto st = tensorcast::cuda::memcpy(
          out, static_cast<uint8_t*>(device_ptr_.get()) + (p.src_offset + local), take, cudaMemcpyDeviceToHost);
      if (!st.ok())
        return st;
      if (auto sync = tensorcast::cuda::device_synchronize(); !sync.ok())
        return sync;
    }
    out += take;
    offset += take;
    remaining -= take;
    if (take == avail) {
      ++idx;
    }
  }
  return static_cast<size_t>(out - static_cast<uint8_t*>(dst));
}

absl::StatusOr<std::string> compute_data_multihash_from_gpu_plan(
    gsl::not_null<void*> device_ptr,
    int device_id,
    absl::Span<const SegmentPiece> plan,
    uint64_t total_size,
    size_t leaf_chunk_bytes) {
  LinearizedGpuPlanSource src(device_ptr, device_id, plan, total_size);
  return compute_data_multihash_from_seekable_source(src, total_size, leaf_chunk_bytes);
}

} // namespace tensorcast::store::loader
