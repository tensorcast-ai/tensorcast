// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/common/piece_view_state_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/dataplane/verification/verification_utils.h"
#include "core/store/materialization/dataplane/view/view_ingest_executor.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::materialization::common {

namespace {

using ::tensorcast::store::view::CanonicalRange;

constexpr size_t kDefaultLeafChunkBytes = 4ULL * 1024 * 1024;
constexpr uint64_t kProofChunkBytesV1 = 4ULL * 1024 * 1024;
constexpr std::string_view kProofSchemaV1 = "v1";

struct TensorInterval {
  std::string tensor_name;
  uint64_t offset{0};
  uint64_t size_bytes{0};
};

struct CanonicalToViewSpan {
  uint64_t canonical_offset{0};
  uint64_t view_offset{0};
  uint64_t length{0};
};

absl::StatusOr<std::vector<TensorInterval>> parse_tensor_intervals(std::string_view canonical_index_json) {
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("canonical_index_json must not be empty");
  }
  nlohmann::json json;
  try {
    json = nlohmann::json::parse(canonical_index_json, nullptr, true);
  } catch (const std::exception& ex) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON: ", ex.what()));
  }
  if (!json.is_object()) {
    return absl::InvalidArgumentError("canonical index JSON must be an object");
  }

  std::vector<TensorInterval> out;
  out.reserve(json.size());
  for (auto it = json.begin(); it != json.end(); ++it) {
    if (!it.value().is_array() || it.value().size() < 2) {
      continue;
    }
    TensorInterval interval;
    interval.tensor_name = it.key();
    interval.offset = it.value()[0].get<uint64_t>();
    interval.size_bytes = it.value()[1].get<uint64_t>();
    out.push_back(std::move(interval));
  }
  std::sort(out.begin(), out.end(), [](const TensorInterval& lhs, const TensorInterval& rhs) {
    return lhs.offset < rhs.offset;
  });
  return out;
}

bool ranges_cover_interval(const std::vector<CanonicalRange>& ranges, uint64_t start, uint64_t length) {
  if (length == 0) {
    return true;
  }
  uint64_t cursor = start;
  const uint64_t end = start + length;
  for (const auto& range : ranges) {
    if (range.length == 0) {
      continue;
    }
    const uint64_t range_start = range.offset;
    const uint64_t range_end = range.offset + range.length;
    if (range_end <= cursor) {
      continue;
    }
    if (range_start > cursor) {
      return false;
    }
    cursor = std::min(end, range_end);
    if (cursor >= end) {
      return true;
    }
  }
  return cursor >= end;
}

absl::StatusOr<std::vector<CanonicalToViewSpan>> canonical_spans_for_tensor(
    const loader::ViewWritePlan& write_plan,
    uint64_t tensor_offset,
    uint64_t tensor_bytes,
    uint64_t view_size_bytes) {
  const uint64_t tensor_end = tensor_offset + tensor_bytes;
  std::vector<CanonicalToViewSpan> spans;
  spans.reserve(write_plan.chunks.size());

  for (const auto& chunk : write_plan.chunks) {
    const uint64_t chunk_end = chunk.canonical_offset + chunk.length;
    if (chunk.length == 0 || chunk_end <= tensor_offset || chunk.canonical_offset >= tensor_end) {
      continue;
    }
    const uint64_t start = std::max<uint64_t>(chunk.canonical_offset, tensor_offset);
    const uint64_t end = std::min<uint64_t>(chunk_end, tensor_end);
    CanonicalToViewSpan span;
    span.canonical_offset = start;
    span.view_offset = chunk.view_offset + (start - chunk.canonical_offset);
    span.length = end - start;
    if (span.view_offset > view_size_bytes || span.view_offset + span.length > view_size_bytes) {
      return absl::OutOfRangeError("view offset out of bounds while building proof spans");
    }
    spans.push_back(std::move(span));
  }

  std::sort(spans.begin(), spans.end(), [](const CanonicalToViewSpan& lhs, const CanonicalToViewSpan& rhs) {
    return lhs.canonical_offset < rhs.canonical_offset;
  });

  uint64_t cursor = tensor_offset;
  for (const auto& span : spans) {
    if (span.length == 0) {
      continue;
    }
    if (span.canonical_offset != cursor) {
      return absl::FailedPreconditionError("tensor canonical span coverage is not contiguous");
    }
    cursor = span.canonical_offset + span.length;
  }
  if (cursor != tensor_end) {
    return absl::FailedPreconditionError("tensor canonical span coverage is incomplete");
  }
  return spans;
}

size_t normalize_leaf_chunk_bytes(size_t leaf_chunk_bytes) {
  return leaf_chunk_bytes == 0 ? kDefaultLeafChunkBytes : leaf_chunk_bytes;
}

} // namespace

std::vector<CanonicalRange> canonical_ranges_from_write_plan(const loader::ViewWritePlan& write_plan) {
  std::vector<CanonicalRange> ranges;
  ranges.reserve(write_plan.chunks.size());
  for (const auto& chunk : write_plan.chunks) {
    CanonicalRange range;
    range.offset = chunk.canonical_offset;
    range.length = chunk.length;
    ranges.push_back(range);
  }
  std::sort(ranges.begin(), ranges.end(), [](const CanonicalRange& lhs, const CanonicalRange& rhs) {
    return lhs.offset < rhs.offset;
  });

  std::vector<CanonicalRange> merged;
  merged.reserve(ranges.size());
  for (const auto& range : ranges) {
    if (range.length == 0) {
      continue;
    }
    if (merged.empty()) {
      merged.push_back(range);
      continue;
    }
    auto& last = merged.back();
    const uint64_t last_end = last.offset + last.length;
    if (range.offset <= last_end) {
      const uint64_t new_end = std::max(last_end, range.offset + range.length);
      last.length = new_end - last.offset;
    } else {
      merged.push_back(range);
    }
  }
  return merged;
}

absl::StatusOr<PieceViewStatePayload> build_piece_view_state_payload(const PieceViewStateRequest& request) {
  if (request.view_id.empty()) {
    return absl::StatusOr<PieceViewStatePayload>(absl::InvalidArgumentError("view_id is required"));
  }

  PieceViewStatePayload payload;
  payload.view_size_bytes = request.plan.forward.view_size_bytes;
  if (payload.view_size_bytes == 0) {
    return absl::FailedPreconditionError("piece registration requires non-empty view_size_bytes");
  }

  const size_t leaf_chunk_bytes = normalize_leaf_chunk_bytes(request.leaf_chunk_bytes);
  auto view_hash_or = loader::verification::compute_view_tree_hash_and_leaves(
      request.view_source, payload.view_size_bytes, leaf_chunk_bytes);
  if (!view_hash_or.ok()) {
    return view_hash_or.status();
  }
  payload.view_data_hash = view_hash_or->multihash;
  payload.leaf_writes.reserve(view_hash_or->leaf_digests.size());
  for (size_t idx = 0; idx < view_hash_or->leaf_digests.size(); ++idx) {
    global_store::LeafWrite leaf;
    auto* hash_space = leaf.mutable_hash_space();
    hash_space->mutable_byte_space()->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
    hash_space->mutable_byte_space()->set_id(std::string(request.view_id));
    leaf.set_leaf_idx(static_cast<uint64_t>(idx));
    const auto& digest = view_hash_or->leaf_digests[idx];
    leaf.set_digest(digest.data(), static_cast<int>(digest.size()));
    payload.leaf_writes.push_back(std::move(leaf));
  }

  payload.canonical_ranges = canonical_ranges_from_write_plan(request.plan.write);
  for (const auto& range : payload.canonical_ranges) {
    payload.canonical_bytes_covered += range.length;
  }

  auto intervals_or = parse_tensor_intervals(request.canonical_index_json);
  if (!intervals_or.ok()) {
    return intervals_or.status();
  }
  payload.canonical_size_bytes = request.canonical_size_bytes;
  if (payload.canonical_size_bytes == 0) {
    for (const auto& interval : *intervals_or) {
      payload.canonical_size_bytes = std::max(payload.canonical_size_bytes, interval.offset + interval.size_bytes);
    }
  }
  if (payload.canonical_size_bytes > 0 && payload.canonical_bytes_covered > payload.canonical_size_bytes) {
    return absl::FailedPreconditionError("canonical_bytes_covered exceeds canonical_size_bytes");
  }

  std::unordered_map<std::string, uint64_t> transpose_view_offsets;
  std::unordered_map<std::string, loader::TensorTransformPlan> transpose_inverse_plans;
  if (request.plan.forward.transform.requires_materialization) {
    for (const auto& tensor_plan : request.plan.forward.transform.tensors) {
      transpose_view_offsets.emplace(tensor_plan.tensor_name, tensor_plan.dst_offset);
    }
    for (const auto& tensor_plan : request.plan.inverse_transform.tensors) {
      transpose_inverse_plans.emplace(tensor_plan.tensor_name, tensor_plan);
    }
  }

  for (const auto& interval : *intervals_or) {
    if (interval.size_bytes == 0) {
      continue;
    }
    if (!ranges_cover_interval(payload.canonical_ranges, interval.offset, interval.size_bytes)) {
      continue;
    }
    const auto inverse_it = transpose_inverse_plans.find(interval.tensor_name);
    if (inverse_it != transpose_inverse_plans.end()) {
      const auto offset_it = transpose_view_offsets.find(interval.tensor_name);
      if (offset_it == transpose_view_offsets.end()) {
        return absl::FailedPreconditionError("missing transpose tensor dst_offset for proof digests");
      }
      const uint64_t tensor_view_offset = offset_it->second;
      if (tensor_view_offset + interval.size_bytes > payload.view_size_bytes) {
        return absl::OutOfRangeError("transpose tensor view range exceeds view buffer size");
      }
      if (interval.size_bytes > std::numeric_limits<size_t>::max()) {
        return absl::OutOfRangeError("transpose tensor exceeds host memory limits");
      }
      std::vector<uint8_t> view_bytes(static_cast<size_t>(interval.size_bytes));
      auto read_or = request.view_source.read_at(tensor_view_offset, view_bytes.data(), view_bytes.size());
      if (!read_or.ok()) {
        return read_or.status();
      }
      if (*read_or != view_bytes.size()) {
        return absl::OutOfRangeError("failed to read full transpose tensor bytes from view source");
      }

      std::vector<uint8_t> canonical_bytes = view_bytes;
      loader::ViewWritePlan write_plan;
      loader::ViewWritePlan::Chunk write_chunk;
      write_chunk.canonical_offset = 0;
      write_chunk.view_offset = 0;
      write_chunk.length = interval.size_bytes;
      write_chunk.segment_aligned = false;
      write_plan.chunks.push_back(std::move(write_chunk));

      loader::TransformPlan inverse_transform;
      inverse_transform.requires_materialization = true;
      loader::TensorTransformPlan tensor_transform = inverse_it->second;
      tensor_transform.dst_offset = 0;
      tensor_transform.canonical_offset = 0;
      tensor_transform.storage_offset_elements = 0;
      inverse_transform.tensors.push_back(std::move(tensor_transform));

      loader::ViewIngestExecutor executor(
          std::move(write_plan), std::move(inverse_transform), loader::ViewIngestExecutor::IngestTarget::kCanonical);
      absl::Status ingest_status = executor.ingest_chunk(
          /*view_offset=*/0,
          absl::Span<const std::byte>(reinterpret_cast<const std::byte*>(view_bytes.data()), view_bytes.size()),
          tensorcast::common::memory::MemoryLocation::CPU,
          canonical_bytes.data(),
          /*device_id=*/-1);
      if (!ingest_status.ok()) {
        return ingest_status;
      }
      absl::Status finalize_status = executor.finalize(
          tensorcast::common::memory::MemoryLocation::CPU,
          canonical_bytes.data(),
          /*device_id=*/-1);
      if (!finalize_status.ok()) {
        return finalize_status;
      }

      const uint64_t expected_chunks = (interval.size_bytes + kProofChunkBytesV1 - 1) / kProofChunkBytesV1;
      for (uint64_t proof_chunk_idx = 0; proof_chunk_idx < expected_chunks; ++proof_chunk_idx) {
        const uint64_t local_start = proof_chunk_idx * kProofChunkBytesV1;
        const uint64_t local_end = std::min<uint64_t>(interval.size_bytes, local_start + kProofChunkBytesV1);
        if (local_end <= local_start) {
          continue;
        }
        if (local_end > std::numeric_limits<size_t>::max()) {
          return absl::OutOfRangeError("proof chunk exceeds host memory limits");
        }
        const size_t chunk_bytes = static_cast<size_t>(local_end - local_start);
        std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(
            absl::Span<const uint8_t>(canonical_bytes.data() + local_start, chunk_bytes));
        if (digest.size() != 32) {
          return absl::InternalError("sha256 digest size mismatch");
        }
        global_store::PieceProofDigestWrite proof;
        proof.set_view_id(std::string(request.view_id));
        proof.set_tensor_name(interval.tensor_name);
        proof.set_proof_schema_version(std::string(kProofSchemaV1));
        proof.set_proof_chunk_idx(proof_chunk_idx);
        proof.set_digest(digest.data(), static_cast<int>(digest.size()));
        payload.proof_digests.push_back(std::move(proof));
      }
      continue;
    }

    auto spans_or =
        canonical_spans_for_tensor(request.plan.write, interval.offset, interval.size_bytes, payload.view_size_bytes);
    if (!spans_or.ok()) {
      return spans_or.status();
    }
    std::vector<CanonicalToViewSpan> spans = std::move(*spans_or);
    size_t span_idx = 0;
    const uint64_t expected_chunks = (interval.size_bytes + kProofChunkBytesV1 - 1) / kProofChunkBytesV1;
    for (uint64_t proof_chunk_idx = 0; proof_chunk_idx < expected_chunks; ++proof_chunk_idx) {
      const uint64_t local_start = proof_chunk_idx * kProofChunkBytesV1;
      const uint64_t local_end = std::min<uint64_t>(interval.size_bytes, local_start + kProofChunkBytesV1);
      const uint64_t abs_start = interval.offset + local_start;
      const uint64_t abs_end = interval.offset + local_end;
      if (abs_end <= abs_start) {
        continue;
      }
      if (abs_end - abs_start > std::numeric_limits<size_t>::max()) {
        return absl::OutOfRangeError("proof chunk exceeds host memory limits");
      }
      std::vector<uint8_t> buffer(static_cast<size_t>(abs_end - abs_start));
      uint64_t cursor = abs_start;
      while (cursor < abs_end) {
        while (span_idx < spans.size() && spans[span_idx].canonical_offset + spans[span_idx].length <= cursor) {
          ++span_idx;
        }
        if (span_idx >= spans.size()) {
          return absl::FailedPreconditionError("missing canonical span while computing proof digests");
        }
        const auto& span = spans[span_idx];
        if (span.canonical_offset > cursor) {
          return absl::FailedPreconditionError("canonical span gap while computing proof digests");
        }
        const uint64_t take_end = std::min<uint64_t>(abs_end, span.canonical_offset + span.length);
        const size_t take = static_cast<size_t>(take_end - cursor);
        const uint64_t src_view_offset = span.view_offset + (cursor - span.canonical_offset);
        auto dst = absl::MakeSpan(buffer).subspan(static_cast<size_t>(cursor - abs_start), take);
        auto read_or = request.view_source.read_at(src_view_offset, dst.data(), dst.size());
        if (!read_or.ok()) {
          return read_or.status();
        }
        if (*read_or != dst.size()) {
          return absl::OutOfRangeError("failed to read full proof span from view source");
        }
        cursor = take_end;
      }
      std::vector<uint8_t> digest =
          tensorcast::common::sha256_digest_bytes(absl::Span<const uint8_t>(buffer.data(), buffer.size()));
      if (digest.size() != 32) {
        return absl::InternalError("sha256 digest size mismatch");
      }
      global_store::PieceProofDigestWrite proof;
      proof.set_view_id(std::string(request.view_id));
      proof.set_tensor_name(interval.tensor_name);
      proof.set_proof_schema_version(std::string(kProofSchemaV1));
      proof.set_proof_chunk_idx(proof_chunk_idx);
      proof.set_digest(digest.data(), static_cast<int>(digest.size()));
      payload.proof_digests.push_back(std::move(proof));
    }
  }

  return payload;
}

} // namespace tensorcast::store::materialization::common
