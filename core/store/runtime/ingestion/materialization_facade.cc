// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/ingestion/materialization_facade.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "core/cuda/cuda_ipc.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/worker_identity.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/control/materialize_orchestrator.h"
#include "core/store/materialization/dataplane/loaders/disk_loader.h"
#include "core/store/materialization/dataplane/loaders/p2p_loader.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/runtime/pump.h"
#include "core/store/materialization/dataplane/runtime/streaming_buffer_adapter.h"
#include "core/store/materialization/dataplane/sinks/target_layout_gpu_sink.h"
#include "core/store/materialization/dataplane/sources/remote_key_source.h"
#include "core/store/materialization/dataplane/sources/segment_plan_source.h"
#include "core/store/materialization/dataplane/view/view_plan_source.h"
#include "core/store/materialization/dataplane/view/view_transform_executor.h"
#include "core/store/replica/replica.h"
#include "core/store/view_utils.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::runtime::ingestion {

namespace pipeline = tensorcast::store::materialization::runtime::pipeline;
using materialization::control::MaterializeOrchestrator;

namespace {

bool is_local_identity(const components::WorkerIdentity& local) {
  return !local.node_id.empty() || !local.node_address.empty();
}

bool is_local_replica(const components::RemoteReplicaInfo& remote, const components::WorkerIdentity& local) {
  if (!is_local_identity(local)) {
    return false;
  }
  if (!local.node_id.empty() && !remote.node_id.empty()) {
    return local.node_id == remote.node_id;
  }
  if (!local.node_address.empty() && !remote.node_address.empty() && local.node_address == remote.node_address) {
    if (local.p2p_port == 0 || remote.node_port == 0) {
      return true;
    }
    return local.p2p_port == remote.node_port;
  }
  return false;
}

absl::Status stale_local_route_status(std::string_view artifact_id) {
  return absl::UnavailableError(
      absl::StrCat("Global Store route stale for artifact_id=", artifact_id, "; retry or provide disk_path"));
}

absl::StatusOr<std::pair<std::string, std::string>> parse_mi2_multihashes(std::string_view artifact_id) {
  constexpr std::string_view kPrefix = common::kMi2Prefix;
  if (!artifact_id.starts_with(kPrefix)) {
    return absl::InvalidArgumentError("sealed_artifact_id must start with \"mi2:\"");
  }
  const size_t index_begin = kPrefix.size();
  const size_t sep = artifact_id.find(':', index_begin);
  if (sep == std::string_view::npos) {
    return absl::InvalidArgumentError("sealed_artifact_id must be of form mi2:<index_multihash>:<data_multihash>");
  }
  const std::string_view index_mh = artifact_id.substr(index_begin, sep - index_begin);
  const std::string_view data_mh = artifact_id.substr(sep + 1);
  if (index_mh.empty() || data_mh.empty()) {
    return absl::InvalidArgumentError("sealed_artifact_id must include index and data multihash components");
  }
  return std::make_pair(std::string(index_mh), std::string(data_mh));
}

absl::StatusOr<DeviceKey> select_seal_target_device(components::DeviceManager& device_manager) {
  if (device_manager.get_num_gpus() <= 0) {
    return absl::FailedPreconditionError("seal_assembly requires at least one GPU device");
  }
  return DeviceRegistry::instance().gpu_key(0);
}

} // namespace

class PlanBackedSeekableSource final : public loader::SeekableSource {
 public:
  PlanBackedSeekableSource(
      std::unique_ptr<loader::SeekableSource> source,
      std::shared_ptr<const std::vector<loader::SegmentPiece>> plan,
      uint64_t total_size)
      : source_(std::move(source)), plan_(std::move(plan)), total_size_(total_size) {
    piece_starts_.reserve(plan_->size());
    for (const auto& piece : *plan_) {
      piece_starts_.push_back(piece.dst_offset);
    }
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto st = read_at(current_offset_, dst, max_bytes);
    if (!st.ok()) {
      return st;
    }
    current_offset_ += *st;
    return st;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= total_size_) {
      return static_cast<size_t>(0);
    }
    size_t remaining = static_cast<size_t>(std::min<uint64_t>(bytes, total_size_ - offset));
    auto* out = static_cast<uint8_t*>(dst);

    if (piece_starts_.empty()) {
      return static_cast<size_t>(0);
    }
    auto it = std::upper_bound(piece_starts_.begin(), piece_starts_.end(), offset);
    size_t idx = 0;
    if (it == piece_starts_.begin()) {
      idx = 0;
    } else {
      idx = static_cast<size_t>(it - piece_starts_.begin() - 1);
    }
    if (idx >= plan_->size()) {
      return static_cast<size_t>(0);
    }

    while (remaining > 0 && idx < plan_->size()) {
      const auto& p = (*plan_)[idx];
      const uint64_t local = offset - p.dst_offset;
      const size_t avail = static_cast<size_t>(p.length - local);
      const size_t take = std::min(remaining, avail);
      if (p.kind == loader::SegmentPiece::PAD) {
        std::memset(out, 0, take);
      } else {
        auto read_or = source_->read_at(p.src_offset + local, out, take);
        if (!read_or.ok()) {
          return read_or.status();
        }
        if (*read_or != take) {
          return absl::DataLossError("Short read while materializing into target");
        }
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

 private:
  std::unique_ptr<loader::SeekableSource> source_;
  std::shared_ptr<const std::vector<loader::SegmentPiece>> plan_;
  std::vector<uint64_t> piece_starts_;
  uint64_t total_size_{0};
  uint64_t current_offset_{0};
};

class CpuMemorySource final : public loader::SeekableSource {
 public:
  CpuMemorySource(gsl::not_null<const void*> base_ptr, uint64_t total_size)
      : base_ptr_(static_cast<const uint8_t*>(base_ptr.get())), total_size_(total_size) {}

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(current_offset_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or;
    }
    current_offset_ += *read_or;
    return read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= total_size_) {
      return static_cast<size_t>(0);
    }
    const size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, total_size_ - offset));
    std::memcpy(dst, base_ptr_ + offset, to_copy);
    return to_copy;
  }

 private:
  const uint8_t* base_ptr_;
  uint64_t total_size_;
  uint64_t current_offset_{0};
};

class LocalReplicaSource final : public loader::SeekableSource {
 public:
  static absl::StatusOr<std::shared_ptr<loader::SeekableSource>> Create(
      std::shared_ptr<replica::Replica> replica,
      common::memory::MemoryLocation location,
      int device_id,
      uint64_t total_size) {
    if (!replica) {
      return absl::InvalidArgumentError("local replica source requires replica");
    }
    if (total_size == 0) {
      return absl::InvalidArgumentError("local replica source requires non-zero size");
    }
    std::shared_ptr<loader::SeekableSource> source;
    if (location == common::memory::MemoryLocation::GPU) {
      const auto gpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
      if (gpu_ptrs.empty() || gpu_ptrs[0] == nullptr) {
        return absl::FailedPreconditionError("local replica GPU pointer unavailable");
      }
      std::array<loader::SegmentPiece, 1> plan{loader::SegmentPiece{loader::SegmentPiece::DATA, 0, total_size, 0}};
      source = std::make_shared<loader::LinearizedGpuPlanSource>(
          gsl::not_null<void*>{gpu_ptrs[0]}, device_id, absl::MakeSpan(plan), total_size);
    } else {
      const auto cpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::CPU);
      if (cpu_ptrs.empty() || cpu_ptrs[0] == nullptr) {
        return absl::FailedPreconditionError("local replica CPU pointer unavailable");
      }
      source = std::make_shared<CpuMemorySource>(gsl::not_null<const void*>{cpu_ptrs[0]}, total_size);
    }
    return std::shared_ptr<loader::SeekableSource>(new LocalReplicaSource(std::move(replica), std::move(source)));
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    return source_->read(dst, max_bytes);
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    return source_->read_at(offset, dst, bytes);
  }

 private:
  LocalReplicaSource(std::shared_ptr<replica::Replica> replica, std::shared_ptr<loader::SeekableSource> source)
      : replica_(std::move(replica)), source_(std::move(source)) {}

  std::shared_ptr<replica::Replica> replica_;
  std::shared_ptr<loader::SeekableSource> source_;
};

absl::StatusOr<uint64_t> compute_logical_total_size(std::string_view canonical_index_json) {
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("canonical index JSON must not be empty");
  }
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(canonical_index_json, nullptr, true);
  } catch (const std::exception& ex) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON: ", ex.what()));
  }
  if (!j.is_object()) {
    return absl::InvalidArgumentError("canonical index JSON must be an object");
  }
  uint64_t total_size = 0;
  for (auto it = j.begin(); it != j.end(); ++it) {
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() < 2) {
      continue;
    }
    const uint64_t offset = arr[0].get<uint64_t>();
    const uint64_t size = arr[1].get<uint64_t>();
    total_size = std::max<uint64_t>(total_size, offset + size);
  }
  if (total_size == 0) {
    return absl::InvalidArgumentError("canonical index total_size is zero");
  }
  return total_size;
}

struct AssemblyTargetRange {
  enum class Kind : uint8_t { kData = 0, kPad = 1 };
  Kind kind{Kind::kData};
  uint64_t canonical_offset{0};
  uint64_t target_offset{0};
  uint64_t length{0};
};

struct AssemblySourceInfo {
  std::string view_id;
  components::TransportSession session;
  uint64_t view_size_bytes{0};
  components::TransportLease transport_lease;
};

struct AssemblySegment {
  enum class Kind : uint8_t { kData = 0, kPad = 1 };
  Kind kind{Kind::kData};
  uint64_t dst_offset{0};
  uint64_t length{0};
  size_t source_index{0};
  uint64_t src_offset{0};
};

struct AssemblyPlan {
  uint64_t total_size{0};
  std::vector<AssemblySourceInfo> sources;
  std::vector<AssemblySegment> segments;
  std::vector<view::CanonicalRange> missing_ranges;
};

struct PieceInterval {
  uint64_t canonical_offset{0};
  uint64_t length{0};
  size_t source_index{0};
  uint64_t view_offset{0};
};

class AssemblySource final : public loader::SeekableSource {
 public:
  AssemblySource(
      std::vector<std::shared_ptr<loader::SeekableSource>> sources,
      std::vector<AssemblySegment> segments,
      uint64_t total_size)
      : sources_(std::move(sources)), segments_(std::move(segments)), total_size_(total_size) {
    segment_starts_.reserve(segments_.size());
    for (const auto& seg : segments_) {
      segment_starts_.push_back(seg.dst_offset);
    }
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto st = read_at(current_offset_, dst, max_bytes);
    if (!st.ok()) {
      return st;
    }
    current_offset_ += *st;
    return st;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= total_size_) {
      return static_cast<size_t>(0);
    }
    size_t remaining = static_cast<size_t>(std::min<uint64_t>(bytes, total_size_ - offset));
    auto* out = static_cast<uint8_t*>(dst);

    if (segments_.empty()) {
      return static_cast<size_t>(0);
    }
    auto it = std::upper_bound(segment_starts_.begin(), segment_starts_.end(), offset);
    size_t idx = 0;
    if (it == segment_starts_.begin()) {
      idx = 0;
    } else {
      idx = static_cast<size_t>(it - segment_starts_.begin() - 1);
    }
    if (idx >= segments_.size()) {
      return static_cast<size_t>(0);
    }

    while (remaining > 0 && idx < segments_.size()) {
      const auto& seg = segments_[idx];
      if (offset < seg.dst_offset) {
        return absl::DataLossError("Assembly source gap detected");
      }
      const uint64_t local = offset - seg.dst_offset;
      const uint64_t seg_end = seg.dst_offset + seg.length;
      if (local >= seg.length) {
        ++idx;
        continue;
      }
      const size_t avail = static_cast<size_t>(seg_end - offset);
      const size_t take = std::min(remaining, avail);
      if (seg.kind == AssemblySegment::Kind::kPad) {
        std::memset(out, 0, take);
      } else {
        if (seg.source_index >= sources_.size()) {
          return absl::InternalError("Assembly source index out of range");
        }
        auto read_or = sources_[seg.source_index]->read_at(seg.src_offset + local, out, take);
        if (!read_or.ok()) {
          return read_or.status();
        }
        if (*read_or != take) {
          return absl::DataLossError("Assembly source short read");
        }
      }
      out += take;
      offset += take;
      remaining -= take;
      if (offset >= seg_end) {
        ++idx;
      }
    }
    return static_cast<size_t>(out - static_cast<uint8_t*>(dst));
  }

  [[nodiscard]] bool supports_direct_write() const override {
    return false;
  }

 private:
  std::vector<std::shared_ptr<loader::SeekableSource>> sources_;
  std::vector<AssemblySegment> segments_;
  std::vector<uint64_t> segment_starts_;
  uint64_t total_size_{0};
  uint64_t current_offset_{0};
};

std::vector<AssemblyTargetRange> build_target_ranges_from_view_plan(const loader::ViewPlan& plan) {
  std::vector<AssemblyTargetRange> ranges;
  ranges.reserve(plan.selection.ranges.size());
  for (const auto& range : plan.selection.ranges) {
    AssemblyTargetRange out;
    out.target_offset = range.dst_offset;
    out.length = range.length;
    if (range.kind == loader::SelectionPlan::Range::Kind::kPad) {
      out.kind = AssemblyTargetRange::Kind::kPad;
    } else {
      out.kind = AssemblyTargetRange::Kind::kData;
      out.canonical_offset = range.src_offset;
    }
    ranges.push_back(out);
  }
  return ranges;
}

absl::StatusOr<std::vector<AssemblyTargetRange>> build_target_ranges_for_canonical(
    std::string_view canonical_index_json,
    uint64_t total_size) {
  auto plan_or =
      loader::build_segment_plan_from_canonical_index_json(canonical_index_json, total_size, /*align_bytes=*/8);
  if (!plan_or.ok()) {
    return plan_or.status();
  }
  std::vector<AssemblyTargetRange> ranges;
  ranges.reserve(plan_or->size());
  for (const auto& segment : *plan_or) {
    AssemblyTargetRange out;
    out.target_offset = segment.dst_offset;
    out.length = segment.length;
    if (segment.kind == loader::SegmentPiece::PAD) {
      out.kind = AssemblyTargetRange::Kind::kPad;
    } else {
      out.kind = AssemblyTargetRange::Kind::kData;
      out.canonical_offset = segment.dst_offset;
    }
    if (out.length > 0) {
      ranges.push_back(out);
    }
  }
  return ranges;
}

void coalesce_missing_ranges(std::vector<view::CanonicalRange>* ranges) {
  if (!ranges) {
    return;
  }
  auto& vec = *ranges;
  if (vec.empty()) {
    return;
  }
  std::sort(vec.begin(), vec.end(), [](const auto& lhs, const auto& rhs) { return lhs.offset < rhs.offset; });
  std::vector<view::CanonicalRange> merged;
  merged.reserve(vec.size());
  for (const auto& range : vec) {
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
  vec = std::move(merged);
}

std::string format_missing_ranges(absl::Span<const view::CanonicalRange> ranges, size_t limit = 5) {
  std::string out;
  size_t count = 0;
  for (const auto& range : ranges) {
    if (count++ > 0) {
      absl::StrAppend(&out, ", ");
    }
    absl::StrAppend(&out, "[", range.offset, "+", range.length, ")");
    if (count >= limit) {
      break;
    }
  }
  if (ranges.size() > limit) {
    absl::StrAppend(&out, ", ...");
  }
  return out;
}

absl::Status build_assembly_segments(
    absl::Span<const AssemblyTargetRange> target_ranges,
    const std::vector<PieceInterval>& intervals,
    std::vector<AssemblySegment>* segments,
    std::vector<view::CanonicalRange>* missing_ranges) {
  if (segments == nullptr || missing_ranges == nullptr) {
    return absl::InvalidArgumentError("assembly segments outputs must not be null");
  }
  segments->clear();
  missing_ranges->clear();

  std::vector<uint64_t> interval_starts;
  interval_starts.reserve(intervals.size());
  for (const auto& interval : intervals) {
    interval_starts.push_back(interval.canonical_offset);
  }

  auto find_interval_index = [&](uint64_t offset) -> size_t {
    if (intervals.empty()) {
      return 0;
    }
    auto it = std::upper_bound(interval_starts.begin(), interval_starts.end(), offset);
    if (it == interval_starts.begin()) {
      return 0;
    }
    return static_cast<size_t>(it - interval_starts.begin() - 1);
  };

  for (const auto& target : target_ranges) {
    if (target.length == 0) {
      continue;
    }
    if (target.kind == AssemblyTargetRange::Kind::kPad) {
      AssemblySegment seg;
      seg.kind = AssemblySegment::Kind::kPad;
      seg.dst_offset = target.target_offset;
      seg.length = target.length;
      segments->push_back(seg);
      continue;
    }

    uint64_t cursor = target.canonical_offset;
    const uint64_t end = target.canonical_offset + target.length;
    size_t idx = find_interval_index(cursor);
    while (cursor < end) {
      while (idx < intervals.size() && intervals[idx].canonical_offset + intervals[idx].length <= cursor) {
        ++idx;
      }
      if (idx >= intervals.size() || intervals[idx].canonical_offset > cursor) {
        const uint64_t gap_end = (idx < intervals.size()) ? std::min(end, intervals[idx].canonical_offset) : end;
        missing_ranges->push_back(view::CanonicalRange{.offset = cursor, .length = gap_end - cursor});
        cursor = gap_end;
        continue;
      }

      const auto& interval = intervals[idx];
      const uint64_t interval_end = interval.canonical_offset + interval.length;
      const uint64_t take_end = std::min(interval_end, end);
      const uint64_t take_len = take_end - cursor;
      AssemblySegment seg;
      seg.kind = AssemblySegment::Kind::kData;
      seg.dst_offset = target.target_offset + (cursor - target.canonical_offset);
      seg.length = take_len;
      seg.source_index = interval.source_index;
      seg.src_offset = interval.view_offset + (cursor - interval.canonical_offset);
      segments->push_back(seg);
      cursor = take_end;
    }
  }

  coalesce_missing_ranges(missing_ranges);
  return absl::OkStatus();
}

absl::StatusOr<AssemblyPlan> build_assembly_plan(
    components::IGlobalStoreClient& gs_client,
    std::string_view assembly_id,
    std::string_view canonical_index_json,
    uint64_t canonical_total_size,
    absl::Span<const AssemblyTargetRange> target_ranges,
    uint64_t target_total_size,
    const DeviceKey& target_device,
    const components::WorkerIdentity& local_identity) {
  auto variants_or = gs_client.list_variants(assembly_id);
  if (!variants_or.ok()) {
    return variants_or.status();
  }
  const auto& variants = *variants_or;
  if (variants.empty()) {
    return absl::NotFoundError(absl::StrCat("no variants found for assembly_id=", assembly_id));
  }

  std::vector<AssemblySourceInfo> sources;
  std::vector<PieceInterval> intervals;
  std::vector<std::pair<view::CanonicalRange, size_t>> coverage_spans;

  for (const auto& variant : variants) {
    if (variant.view_id.empty()) {
      continue;
    }
    if (variant.canonical_ranges.empty()) {
      return absl::FailedPreconditionError(absl::StrCat("coverage metadata missing for view_id=", variant.view_id));
    }
    if (variant.canonical_size_bytes > 0 && canonical_total_size > 0 &&
        variant.canonical_size_bytes != canonical_total_size) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "canonical size mismatch for view_id=",
              variant.view_id,
              " expected=",
              canonical_total_size,
              " got=",
              variant.canonical_size_bytes));
    }

    auto spec_or = view::parse_view_spec_json(variant.view_spec_json);
    if (!spec_or.ok()) {
      return spec_or.status();
    }
    auto plan_or = loader::ViewPlanner::compute_bidirectional_view_plan(canonical_index_json, *spec_or);
    if (!plan_or.ok()) {
      return plan_or.status();
    }
    const auto& forward = plan_or->forward;
    if (forward.transform.requires_materialization || forward.selection.requires_materialization) {
      return absl::FailedPreconditionError(
          absl::StrCat("variant view requires transforms (unsupported) view_id=", variant.view_id));
    }

    auto session_or = gs_client.request_view_transport(
        assembly_id,
        variant.view_id,
        local_identity.node_id,
        local_identity.node_address,
        local_identity.p2p_port,
        target_device,
        /*wait_timeout_ms=*/30000);
    if (!session_or.ok()) {
      if (absl::IsNotFound(session_or.status())) {
        continue;
      }
      return session_or.status();
    }
    auto session = std::move(*session_or);
    components::TransportLease transport_lease(&gs_client, session.transport_id);
    if (variant.view_size_bytes > 0 && session.remote_replica.memory_size != variant.view_size_bytes) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "variant size mismatch for view_id=",
              variant.view_id,
              " expected=",
              variant.view_size_bytes,
              " got=",
              session.remote_replica.memory_size));
    }

    const size_t source_index = sources.size();
    sources.push_back(
        AssemblySourceInfo{
            .view_id = variant.view_id,
            .session = std::move(session),
            .view_size_bytes = variant.view_size_bytes,
            .transport_lease = std::move(transport_lease),
        });

    for (const auto& range : variant.canonical_ranges) {
      coverage_spans.emplace_back(view::CanonicalRange{.offset = range.offset, .length = range.length}, source_index);
    }

    for (const auto& range : forward.selection.ranges) {
      if (range.kind != loader::SelectionPlan::Range::Kind::kData || range.length == 0) {
        continue;
      }
      intervals.push_back(
          PieceInterval{
              .canonical_offset = range.src_offset,
              .length = range.length,
              .source_index = source_index,
              .view_offset = range.dst_offset});
    }
  }

  if (sources.empty()) {
    return absl::NotFoundError(absl::StrCat("no available piece replicas for assembly_id=", assembly_id));
  }

  std::sort(coverage_spans.begin(), coverage_spans.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first.offset < rhs.first.offset;
  });
  for (size_t i = 1; i < coverage_spans.size(); ++i) {
    const auto& prev = coverage_spans[i - 1];
    const auto& cur = coverage_spans[i];
    const uint64_t prev_end = prev.first.offset + prev.first.length;
    if (cur.first.offset < prev_end && prev.second != cur.second) {
      const auto& prev_view = sources[prev.second].view_id;
      const auto& cur_view = sources[cur.second].view_id;
      return absl::FailedPreconditionError(
          absl::StrCat("overlapping canonical coverage across view_id=", prev_view, " and view_id=", cur_view));
    }
  }

  std::sort(intervals.begin(), intervals.end(), [](const PieceInterval& a, const PieceInterval& b) {
    if (a.canonical_offset != b.canonical_offset) {
      return a.canonical_offset < b.canonical_offset;
    }
    return a.source_index < b.source_index;
  });
  for (size_t i = 1; i < intervals.size(); ++i) {
    const auto& prev = intervals[i - 1];
    const auto& cur = intervals[i];
    const uint64_t prev_end = prev.canonical_offset + prev.length;
    if (cur.canonical_offset < prev_end && prev.source_index != cur.source_index) {
      const auto& prev_view = sources[prev.source_index].view_id;
      const auto& cur_view = sources[cur.source_index].view_id;
      return absl::FailedPreconditionError(
          absl::StrCat("overlapping canonical ranges across view_id=", prev_view, " and view_id=", cur_view));
    }
  }

  AssemblyPlan plan;
  plan.total_size = target_total_size;
  plan.sources = std::move(sources);
  absl::Status seg_status = build_assembly_segments(target_ranges, intervals, &plan.segments, &plan.missing_ranges);
  if (!seg_status.ok()) {
    return seg_status;
  }
  return plan;
}

absl::StatusOr<std::shared_ptr<loader::SeekableSource>> make_local_piece_source(
    components::ReplicaRegistry& registry,
    std::string_view assembly_id,
    std::string_view view_id,
    const DeviceKey& device,
    common::memory::MemoryLocation location,
    int device_id,
    uint64_t view_size_bytes) {
  loading::ReplicaKey key;
  key.artifact_id = std::string(assembly_id);
  key.view_id = std::string(view_id);
  key.device = device;
  key.replica = 0;
  auto replica_or = registry.find(key);
  if (!replica_or.ok()) {
    if (!absl::IsNotFound(replica_or.status())) {
      return replica_or.status();
    }
    const auto candidates = registry.find_by_artifact(assembly_id);
    for (const auto& candidate : candidates) {
      if (!candidate.view_id.has_value()) {
        continue;
      }
      if (candidate.view_id.value() != view_id) {
        continue;
      }
      auto candidate_or = registry.find(candidate);
      if (candidate_or.ok()) {
        replica_or = std::move(candidate_or);
        break;
      }
    }
  }
  if (!replica_or.ok()) {
    return absl::NotFoundError(absl::StrCat("local replica missing for view_id=", view_id));
  }
  return LocalReplicaSource::Create(std::move(*replica_or), location, device_id, view_size_bytes);
}

MaterializationFacade::MaterializationFacade(Config config)
    : config_(std::move(config)),
      hooks_(config_.hooks),
      ingestion_event_hub_(config_.runtime_context->ingestion_event_hub()) {
  ABSL_CHECK(config_.runtime_context != nullptr) << "RuntimeContext is required";
  ABSL_CHECK(config_.replica_runtime != nullptr) << "ReplicaRuntime is required";
  ABSL_CHECK(config_.metadata_gateway != nullptr) << "MetadataGateway is required";
  ABSL_CHECK(config_.options != nullptr) << "StoreEngineOptions must not be null";
  ABSL_CHECK(ingestion_event_hub_ != nullptr) << "RuntimeContext missing ingestion event hub";

  pipeline::IngestionPipeline::Config pipeline_config{
      .storage_path = config_.storage_path,
      .num_threads = config_.num_threads,
      .artifact_chunk_bytes = config_.artifact_chunk_bytes,
      .pinned_memory_timeout = config_.pinned_memory_timeout,
      .engine_options = config_.options,
      .replica_runtime = config_.replica_runtime,
      .runtime_context = config_.runtime_context.get(),
  };
  if (hooks_ && hooks_->pipeline_factory) {
    pipeline_ = hooks_->pipeline_factory(pipeline_config);
  } else {
    pipeline_ = std::make_unique<pipeline::IngestionPipeline>(pipeline_config);
  }
  ABSL_CHECK(pipeline_ != nullptr) << "Ingestion pipeline factory returned null";

  auto& registry = config_.replica_runtime->registry();
  auto pinned_pool = config_.runtime_context->pinned_buffer_pool();
  MaterializationDeps deps(
      gsl::not_null<components::ReplicaRegistry*>{&registry},
      gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{pinned_pool});
  deps.async_runtime = config_.runtime_context->async_runtime();
  deps.artifact_chunk_bytes = config_.artifact_chunk_bytes;
  deps.pinned_memory_timeout = config_.pinned_memory_timeout;
  deps.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);
  deps.num_threads = config_.num_threads;
  deps.view_hash_computer = config_.runtime_context->view_hash_computer();
  deps.ingest_from_disk = [this](
                              const std::string& artifact_identifier,
                              const loading::DiskSource& source,
                              const loading::ReplicaTarget& target,
                              const loading::MaterializeHints& hints) {
    return run_disk_ingestion_internal(artifact_identifier, source, target, hints, /*publish_to_global_store=*/false);
  };
  deps.run_auto = [this](const loading::MaterializationRequest& request) -> absl::StatusOr<loading::ReplicaHandle> {
    auto client = config_.runtime_context->global_store_client();
    if (!client || !client->is_connected()) {
      return absl::FailedPreconditionError("GlobalStoreClient not connected");
    }
    MaterializeOrchestrator orchestrator(
        gsl::not_null<materialization::control::MaterializationBackend*>{this},
        gsl::not_null<components::IGlobalStoreClient*>{client.get()},
        config_.runtime_context->worker_identity());
    auto orchestrated_or = orchestrator.run(request.canonical_artifact_id(), request.target_device(), request.hints());
    if (orchestrated_or.ok()) {
      return *orchestrated_or;
    }
    if (absl::IsNotFound(orchestrated_or.status())) {
      const auto id_kind = common::infer_artifact_id_kind(request.canonical_artifact_id());
      if (id_kind == common::ArtifactIdKind::kCgid) {
        auto assembled_or = assemble_from_pieces(request);
        if (assembled_or.ok()) {
          return *assembled_or;
        }
        return assembled_or.status();
      }
    }
    return orchestrated_or.status();
  };

  if (hooks_ && hooks_->materialization_service_factory) {
    materialization_service_ = hooks_->materialization_service_factory(std::move(deps));
  } else {
    materialization_service_ = std::make_unique<MaterializationService>(std::move(deps));
  }
  ABSL_CHECK(materialization_service_ != nullptr) << "Materialization service factory returned null";
}

MaterializationFacade::~MaterializationFacade() = default;

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::materialize_replica(
    const DeviceKey& target_device,
    loading::MaterializeMode mode,
    const loading::MaterializeHints& hints) {
  auto request_or =
      loading::MaterializationRequest::Create(target_device, mode, hints, config_.replica_runtime->device_manager());
  if (!request_or.ok()) {
    return request_or.status();
  }
  return materialization_service_->execute(request_or.value());
}

absl::StatusOr<loading::MaterializeIntoTargetResult> MaterializationFacade::materialize_into_target(
    const DeviceKey& target_device,
    const loading::IntoTargetLayout& target_layout,
    std::string_view canonical_index_json,
    uint64_t generation,
    const loading::MaterializeHints& hints) {
  if (target_device.type != DeviceType::GPU) {
    return absl::InvalidArgumentError("materialize_into_target requires GPU target device");
  }
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("materialize_into_target requires canonical index bytes");
  }
  if (hints.artifact_id.empty()) {
    return absl::InvalidArgumentError("materialize_into_target requires hints.artifact_id");
  }
  if (target_layout.storages.empty()) {
    return absl::InvalidArgumentError("materialize_into_target requires at least one target storage");
  }

  uint64_t total_size = target_layout.total_size;
  uint64_t computed_total = 0;
  for (const auto& storage : target_layout.storages) {
    if (storage.length == 0) {
      return absl::InvalidArgumentError("materialize_into_target requires non-empty storage length");
    }
    if (storage.length > std::numeric_limits<uint64_t>::max() - computed_total) {
      return absl::OutOfRangeError("materialize_into_target storage length overflow");
    }
    computed_total += storage.length;
  }
  if (total_size == 0) {
    total_size = computed_total;
  } else if (total_size != computed_total) {
    return absl::InvalidArgumentError("materialize_into_target total_size does not match storage lengths");
  }
  if (total_size == 0) {
    return absl::InvalidArgumentError("materialize_into_target requires total_size > 0");
  }

  auto canonical_total_or = compute_logical_total_size(canonical_index_json);
  if (!canonical_total_or.ok()) {
    return canonical_total_or.status();
  }
  const uint64_t canonical_total_size = *canonical_total_or;

  std::optional<loader::ViewPlan> view_plan;
  if (hints.variant && hints.variant->cached_plan.has_value()) {
    view_plan = *hints.variant->cached_plan;
  }
  const loading::TransformPlacement placement =
      hints.variant ? hints.variant->placement : loading::TransformPlacement::kServer;
  if (view_plan.has_value() && view_plan->transform.requires_materialization &&
      placement == loading::TransformPlacement::kServer && target_layout.storages.size() != 1) {
    return absl::InvalidArgumentError(
        "materialize_into_target requires single storage for server-side view transforms");
  }
  if (view_plan.has_value() && view_plan->view_size_bytes > 0 && view_plan->view_size_bytes != total_size) {
    return absl::InvalidArgumentError("materialize_into_target view size does not match target layout size");
  }

  auto plan_key = [&]() -> std::string {
    auto mh_or = common::compute_index_multihash(std::optional<std::string>(canonical_index_json), "");
    if (mh_or.ok()) {
      return absl::StrCat(generation, ":", *mh_or);
    }
    const size_t fallback_hash = std::hash<std::string_view>{}(canonical_index_json);
    return absl::StrCat(generation, ":raw:", fallback_hash);
  }();

  std::shared_ptr<std::vector<loader::SegmentPiece>> plan_ptr;
  {
    absl::MutexLock lock(&segment_plan_mu_);
    auto it = segment_plan_cache_.find(plan_key);
    if (it != segment_plan_cache_.end()) {
      plan_ptr = it->second;
    }
  }
  if (!plan_ptr) {
    auto plan_or = loader::build_segment_plan_from_canonical_index_json(canonical_index_json, canonical_total_size);
    if (!plan_or.ok()) {
      return plan_or.status();
    }
    plan_ptr = std::make_shared<std::vector<loader::SegmentPiece>>(std::move(*plan_or));
    absl::MutexLock lock(&segment_plan_mu_);
    segment_plan_cache_.emplace(plan_key, plan_ptr);
  }

  auto run_source =
      [&](std::unique_ptr<IArtifactLoader> loader,
          loading::MaterializationSource source_kind) -> absl::StatusOr<loading::MaterializeIntoTargetResult> {
    auto init_status = loader->initialize();
    if (!init_status.ok()) {
      return init_status;
    }
    auto source_or = loader->open_source();
    if (!source_or.ok()) {
      return source_or.status();
    }

    std::unique_ptr<loader::SeekableSource> plan_source =
        std::make_unique<PlanBackedSeekableSource>(std::move(*source_or), plan_ptr, canonical_total_size);
    if (view_plan.has_value() && !view_plan->is_identity) {
      plan_source = loader::make_view_plan_source(std::move(plan_source), view_plan->selection);
    }
    if (!plan_source) {
      return absl::InternalError("materialize_into_target failed to build view plan source");
    }

    const size_t slice_bytes = config_.runtime_context->tx_slice_bytes();
    if (slice_bytes == 0 || config_.artifact_chunk_bytes == 0) {
      return absl::FailedPreconditionError("tx_slice_bytes or artifact_chunk_bytes is zero");
    }
    const size_t num_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);
    auto session_spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
        /*num_chunks=*/num_chunks, slice_bytes, config_.runtime_context->pinned_buffer_pool());
    const std::chrono::milliseconds timeout =
        hints.pinned_timeout.count() > 0 ? hints.pinned_timeout : config_.pinned_memory_timeout;
    auto init_spb_status = session_spb->initialize(timeout);
    if (!init_spb_status.ok()) {
      return init_spb_status;
    }
    loader::StreamingBufferAdapter adapter(session_spb);

    std::vector<loader::TargetStorage> storages;
    storages.reserve(target_layout.storages.size());
    std::vector<loader::Range> ranges;
    ranges.reserve(target_layout.storages.size());
    uint64_t range_cursor = 0;
    for (const auto& storage : target_layout.storages) {
      if (storage.length == 0) {
        return absl::InvalidArgumentError("materialize_into_target requires non-empty storage length");
      }
      if (storage.length > std::numeric_limits<size_t>::max()) {
        return absl::OutOfRangeError("materialize_into_target storage length exceeds host limits");
      }
      storages.push_back(loader::TargetStorage{storage.base_ptr, storage.length});
      ranges.emplace_back(range_cursor, static_cast<size_t>(storage.length));
      range_cursor += storage.length;
    }
    if (range_cursor != total_size) {
      return absl::InvalidArgumentError("materialize_into_target storage ranges do not span total_size");
    }

    loader::TargetLayoutGpuSink::Options sink_opts{
        .storages = std::move(storages),
        .chunk_size = config_.artifact_chunk_bytes,
        .device_id = target_device.ordinal,
    };
    loader::TargetLayoutGpuSink sink(std::move(sink_opts));

    const int concurrency = hints.pipeline_concurrency > 0 ? static_cast<int>(hints.pipeline_concurrency)
                                                           : std::max(1, config_.num_threads);
    auto pump_status = loader::pump_ranges(
        *plan_source,
        sink,
        adapter,
        absl::MakeSpan(ranges),
        concurrency,
        config_.runtime_context->async_runtime()->blocking_executor());
    if (!pump_status.ok()) {
      return absl::DataLossError(absl::StrCat("materialize_into_target pump failed: ", pump_status.message()));
    }
    auto close_status = sink.close();
    if (!close_status.ok()) {
      return absl::DataLossError(absl::StrCat("materialize_into_target sink close failed: ", close_status.message()));
    }
    if (view_plan.has_value() && view_plan->transform.requires_materialization &&
        placement == loading::TransformPlacement::kServer) {
      auto transform_status = loader::execute_transform(
          view_plan->transform,
          common::memory::MemoryLocation::GPU,
          target_layout.storages.front().base_ptr.get(),
          target_device.ordinal);
      if (!transform_status.ok()) {
        return absl::DataLossError(
            absl::StrCat("materialize_into_target view transform failed: ", transform_status.message()));
      }
    }
    return loading::MaterializeIntoTargetResult{.source = source_kind};
  };

  auto gs_client = config_.runtime_context->global_store_client();
  const bool gs_connected = gs_client && gs_client->is_connected();
  const bool prefer_disk = hints.source_preference == loading::SourcePreference::kPreferDisk;
  const bool prefer_p2p = hints.source_preference == loading::SourcePreference::kPreferP2P;
  const bool allow_p2p = hints.allow_p2p;
  const bool allow_disk = hints.allow_disk;
  const bool has_disk_path = !hints.disk_path.empty();
  components::WorkerIdentity local_identity = config_.runtime_context->worker_identity();
  if (!is_local_identity(local_identity)) {
    const auto& options = config_.runtime_context->options();
    if (!options.p2p_listen_host.empty()) {
      local_identity.node_address = options.p2p_listen_host;
    }
    local_identity.p2p_port = options.p2p_port;
  }

  if (prefer_disk && !allow_disk) {
    return absl::InvalidArgumentError("source_policy disallows disk but preference=PREFER_DISK was requested");
  }
  if (prefer_p2p && !allow_p2p) {
    return absl::InvalidArgumentError("source_policy disallows P2P but preference=PREFER_P2P was requested");
  }

  if (prefer_disk && has_disk_path && allow_disk) {
    loading::DiskSource disk_src;
    disk_src.path = std::filesystem::path(hints.disk_path);
    disk_src.require_descriptor = tensorcast::common::is_mi2_artifact_id(hints.artifact_id);
    auto disk_or = run_source(std::make_unique<DiskLoader>(disk_src), loading::MaterializationSource::kDisk);
    if (disk_or.ok()) {
      return disk_or;
    }
    if (!gs_connected || !allow_p2p) {
      return disk_or.status();
    }
  }

  if (!gs_connected && (!has_disk_path || !allow_disk)) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  if (allow_p2p && gs_connected && !hints.artifact_id.empty()) {
    auto transport_or = gs_client->request_replica_transport(
        hints.artifact_id,
        local_identity.node_id,
        local_identity.node_address,
        local_identity.p2p_port,
        target_device,
        /*wait_timeout_ms=*/30000);
    if (transport_or.ok()) {
      const auto& session = *transport_or;
      const auto& remote = session.remote_replica;
      if (is_local_replica(remote, local_identity)) {
        LOG(WARNING) << "Global Store returned local replica for artifact_id=" << hints.artifact_id
                     << "; treating route as stale";
        auto complete_status = gs_client->complete_replica_transport(session.transport_id);
        if (!complete_status.ok()) {
          LOG(WARNING) << "complete_replica_transport after stale-local route returned error: " << complete_status;
        }
        if (!has_disk_path) {
          return stale_local_route_status(hints.artifact_id);
        }
      } else {
        P2PSource p2p_src;
        p2p_src.size_bytes = remote.memory_size;
        p2p_src.ip = remote.node_address;
        p2p_src.port = static_cast<uint16_t>(remote.node_port);
        p2p_src.memory_keys = remote.remote_memory_keys;
        p2p_src.buf_sizes = remote.buffer_sizes;
        p2p_src.verification_json = remote.verification_json;
        p2p_src.enable_checksum = false;
        p2p_src.location.type = remote.memory_type;
        p2p_src.location.device_id = remote.device_id;
        auto p2p_or = run_source(std::make_unique<P2PLoader>(p2p_src), loading::MaterializationSource::kP2P);
        auto complete_status = gs_client->complete_replica_transport(session.transport_id);
        if (!complete_status.ok()) {
          LOG(WARNING) << "complete_replica_transport returned error: " << complete_status;
        }
        if (p2p_or.ok()) {
          return p2p_or;
        }
        if (!allow_disk || !has_disk_path || prefer_p2p) {
          return p2p_or.status();
        }
      }
    } else if (!allow_disk || !has_disk_path) {
      return transport_or.status();
    }
  }

  if (allow_disk && has_disk_path) {
    loading::DiskSource disk_src;
    disk_src.path = std::filesystem::path(hints.disk_path);
    disk_src.require_descriptor = tensorcast::common::is_mi2_artifact_id(hints.artifact_id);
    return run_source(std::make_unique<DiskLoader>(disk_src), loading::MaterializationSource::kDisk);
  }

  if (!allow_p2p && !allow_disk) {
    return absl::FailedPreconditionError("source_policy disallows P2P and disk for materialize_into_target");
  }

  return absl::FailedPreconditionError("materialize_into_target requires disk_path or Global Store connectivity");
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::assemble_from_pieces(
    const loading::MaterializationRequest& request) {
  auto gs_client = config_.runtime_context->global_store_client();
  if (!gs_client || !gs_client->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (request.canonical_artifact_id().empty()) {
    return absl::InvalidArgumentError("assemble_from_pieces requires artifact_id");
  }
  if (!request.target_is_gpu()) {
    return absl::FailedPreconditionError("assemble_from_pieces requires GPU target device");
  }

  std::string canonical_index_json;
  if (request.hints().variant && request.hints().variant->canonical_index_json.has_value()) {
    canonical_index_json = *request.hints().variant->canonical_index_json;
  } else {
    auto index_or = gs_client->get_artifact_index_by_id(request.canonical_artifact_id());
    if (!index_or.ok()) {
      return index_or.status();
    }
    canonical_index_json = std::move(*index_or);
  }

  auto canonical_total_or = compute_logical_total_size(canonical_index_json);
  if (!canonical_total_or.ok()) {
    return canonical_total_or.status();
  }
  const uint64_t canonical_total_size = *canonical_total_or;

  std::optional<loader::ViewPlan> target_view_plan;
  std::vector<AssemblyTargetRange> target_ranges;
  uint64_t target_total_size = 0;

  if (request.requested_view_id().has_value()) {
    const auto& view_id = *request.requested_view_id();
    if (view_id.empty()) {
      return absl::InvalidArgumentError("requested view_id must be non-empty for assembly");
    }
    if (request.hints().variant && request.hints().variant->cached_plan.has_value()) {
      target_view_plan = *request.hints().variant->cached_plan;
    } else {
      std::optional<loader::ViewSpec> view_spec;
      if (request.hints().variant && request.hints().variant->view_spec.has_value()) {
        view_spec = *request.hints().variant->view_spec;
      } else {
        auto meta_or = gs_client->get_view_metadata(request.canonical_artifact_id(), view_id);
        if (!meta_or.ok()) {
          return meta_or.status();
        }
        auto spec_or = view::parse_view_spec_json(meta_or->view_spec_json);
        if (!spec_or.ok()) {
          return spec_or.status();
        }
        view_spec = std::move(*spec_or);
      }
      if (!view_spec.has_value()) {
        return absl::InvalidArgumentError("view_spec is required to assemble requested view_id");
      }
      auto plan_or = loader::ViewPlanner::compute_view_plan(canonical_index_json, *view_spec);
      if (!plan_or.ok()) {
        return plan_or.status();
      }
      target_view_plan = std::move(*plan_or);
    }

    if (target_view_plan->transform.requires_materialization || target_view_plan->selection.requires_materialization) {
      return absl::FailedPreconditionError("assembly requires selection-only view (no transforms)");
    }
    target_ranges = build_target_ranges_from_view_plan(*target_view_plan);
    target_total_size = target_view_plan->view_size_bytes;
  } else {
    auto ranges_or = build_target_ranges_for_canonical(canonical_index_json, canonical_total_size);
    if (!ranges_or.ok()) {
      return ranges_or.status();
    }
    target_ranges = std::move(*ranges_or);
    target_total_size = canonical_total_size;
  }

  if (target_total_size == 0) {
    return absl::InvalidArgumentError("assembly target size must be > 0");
  }

  auto plan_or = build_assembly_plan(
      *gs_client,
      request.canonical_artifact_id(),
      canonical_index_json,
      canonical_total_size,
      absl::MakeSpan(target_ranges),
      target_total_size,
      request.target_device(),
      config_.runtime_context->worker_identity());
  if (!plan_or.ok()) {
    return plan_or.status();
  }
  AssemblyPlan plan = std::move(*plan_or);
  if (!plan.missing_ranges.empty()) {
    return absl::UnavailableError(
        absl::StrCat("assembly missing canonical ranges: ", format_missing_ranges(plan.missing_ranges)));
  }

  components::WorkerIdentity local_identity = config_.runtime_context->worker_identity();
  if (!is_local_identity(local_identity)) {
    const auto& options = config_.runtime_context->options();
    if (!options.p2p_listen_host.empty()) {
      local_identity.node_address = options.p2p_listen_host;
    }
    local_identity.p2p_port = options.p2p_port;
  }
  auto comm_manager = config_.runtime_context->communication_manager();
  const bool comm_enabled = comm_manager && comm_manager->is_enabled();
  auto& replica_registry = config_.replica_runtime->registry();
  std::vector<std::shared_ptr<loader::SeekableSource>> piece_sources;
  piece_sources.reserve(plan.sources.size());
  for (const auto& source : plan.sources) {
    const auto& remote = source.session.remote_replica;
    if (is_local_replica(remote, local_identity)) {
      DeviceKey local_device;
      if (remote.memory_type == common::memory::MemoryLocation::CPU) {
        local_device = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
      } else {
        const int local_device_id = remote.device_id >= 0 ? remote.device_id : request.target_device().ordinal;
        local_device = DeviceRegistry::instance().gpu_key(local_device_id);
      }
      auto local_or = make_local_piece_source(
          replica_registry,
          request.canonical_artifact_id(),
          source.view_id,
          local_device,
          remote.memory_type,
          remote.device_id,
          source.view_size_bytes);
      if (!local_or.ok()) {
        return local_or.status();
      }
      piece_sources.push_back(*local_or);
      continue;
    }
    if (!comm_enabled) {
      return absl::FailedPreconditionError("Communication not enabled");
    }
    if (remote.remote_memory_keys.empty()) {
      return absl::FailedPreconditionError(absl::StrCat("remote memory keys missing for view_id=", source.view_id));
    }
    if (remote.buffer_sizes.size() != remote.remote_memory_keys.size()) {
      return absl::FailedPreconditionError(absl::StrCat("buffer size mismatch for view_id=", source.view_id));
    }
    std::vector<size_t> buffer_sizes;
    buffer_sizes.reserve(remote.buffer_sizes.size());
    for (uint64_t size : remote.buffer_sizes) {
      buffer_sizes.push_back(static_cast<size_t>(size));
    }
    loader::RemoteKeySource::Options opts{
        .comm_engine =
            gsl::not_null<std::shared_ptr<tensorcast::communicator::engine::Communicator>>{
                comm_manager->get_shared_engine()},
        .memory_keys = remote.remote_memory_keys,
        .buffer_sizes = std::move(buffer_sizes),
        .ip = remote.node_address,
        .port = static_cast<uint16_t>(remote.node_port),
        .total_size = remote.memory_size,
    };
    piece_sources.push_back(std::make_shared<loader::RemoteKeySource>(std::move(opts)));
  }

  loading::ReplicaKey key;
  key.artifact_id = request.canonical_artifact_id();
  key.view_id = request.requested_view_id();
  key.device = request.target_device();
  key.replica = 0;

  auto existing_or = replica_registry.find(key);
  if (existing_or.ok()) {
    const auto& existing = existing_or.value();
    auto ready = existing->ready_signal_for(request.target_location());
    loading::ReplicaHandle handle;
    handle.replica_key = key;
    handle.ready_signal = ready;
    handle.cpu_state = existing->get_memory_state(common::memory::MemoryLocation::CPU);
    handle.gpu_state = existing->get_memory_state(common::memory::MemoryLocation::GPU);
    handle.source = loading::MaterializationSource::kLocalReplica;
    return handle;
  }
  if (!absl::IsNotFound(existing_or.status())) {
    return existing_or.status();
  }

  loading::InlineBufferSource inline_source{.data = nullptr, .size_bytes = plan.total_size};
  replica::ReplicaConfig cfg{
      .source = inline_source,
      .artifact_identifier = key.artifact_id,
      .device_type = DeviceType::GPU,
      .local_device_id = key.device.ordinal,
      .pinned_buffer_pool = config_.runtime_context->pinned_buffer_pool(),
      .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{config_.runtime_context->async_runtime()},
      .artifact_chunk_bytes = config_.artifact_chunk_bytes,
      .expected_artifact_size = plan.total_size,
      .view_plan = target_view_plan,
      .memory_tier_config = config_.options->memory_tier_config,
  };
  cfg.pinned_memory_timeout = config_.pinned_memory_timeout;
  cfg.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);
  cfg.view_id = request.requested_view_id();
  cfg.transform_placement =
      request.hints().variant ? request.hints().variant->placement : loading::TransformPlacement::kServer;

  auto replica_or = replica::Replica::create(cfg);
  if (!replica_or.ok()) {
    return replica_or.status();
  }
  auto replica = std::shared_ptr<replica::Replica>(std::move(replica_or.value()));

  auto source = std::make_unique<AssemblySource>(std::move(piece_sources), std::move(plan.segments), plan.total_size);
  const int concurrency = request.hints().pipeline_concurrency > 0
      ? static_cast<int>(request.hints().pipeline_concurrency)
      : std::max(1, config_.num_threads);
  auto load_future = replica->get_memory_manager().load_async_from_source(
      std::move(source), request.target_location(), concurrency, std::nullopt, std::function<absl::Status()>{});
  absl::Status load_status = std::move(load_future).get();
  if (!load_status.ok()) {
    return load_status;
  }
  replica->set_ready_signal(request.target_location(), absl::OkStatus());

  absl::Status emplace_status = replica_registry.emplace(key, gsl::not_null{replica});
  if (absl::IsAlreadyExists(emplace_status)) {
    auto existing_or = replica_registry.find(key);
    if (!existing_or.ok()) {
      return existing_or.status();
    }
    const auto& existing = existing_or.value();
    loading::ReplicaHandle handle;
    handle.replica_key = key;
    handle.ready_signal = existing->ready_signal_for(request.target_location());
    handle.cpu_state = existing->get_memory_state(common::memory::MemoryLocation::CPU);
    handle.gpu_state = existing->get_memory_state(common::memory::MemoryLocation::GPU);
    handle.source = loading::MaterializationSource::kLocalReplica;
    return handle;
  }
  if (!emplace_status.ok()) {
    return emplace_status;
  }

  loading::ReplicaHandle handle;
  handle.replica_key = key;
  handle.ready_signal = replica->ready_signal_for(request.target_location());
  handle.cpu_state = replica->get_memory_state(common::memory::MemoryLocation::CPU);
  handle.gpu_state = replica->get_memory_state(common::memory::MemoryLocation::GPU);
  handle.source = loading::MaterializationSource::kP2P;
  if (request.target_is_gpu()) {
    const auto gpu_ptrs = replica->get_data_pointer(common::memory::MemoryLocation::GPU);
    handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;
    auto ipc_or = replica->get_memory_manager().get_ipc_handle();
    if (ipc_or.ok()) {
      handle.cuda_ipc_handle = cuda::IpcHandleBytes::from_native(*ipc_or);
    }
  }
  const auto& view_plan = replica->view_plan();
  if (view_plan.has_value() && !view_plan->is_identity) {
    handle.view_index_json = view_plan->view_index_json;
    const uint64_t view_size = view_plan->view_size_bytes;
    if (view_size > 0) {
      auto computer = config_.runtime_context->view_hash_computer();
      if (computer) {
        auto hash = computer->hash_replica_view(
            *replica,
            request.target_location(),
            view_size,
            request.target_is_gpu() ? std::optional<int>(request.target_device().ordinal) : std::nullopt);
        if (hash.has_value()) {
          handle.view_data_hash = std::move(hash);
        }
      }
    }
  }

  return handle;
}

absl::StatusOr<store::SealAssemblyResult> MaterializationFacade::seal_assembly(
    std::string_view assembly_id,
    bool publish_canonical) {
  if (assembly_id.empty()) {
    return absl::InvalidArgumentError("seal_assembly requires non-empty assembly_id");
  }

  auto gs_client = config_.runtime_context->global_store_client();
  if (!gs_client || !gs_client->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  store::SealAssemblyResult result;
  result.assembly_id = std::string(assembly_id);
  result.schema_version = "v3";
  result.encoding = "json";

  const auto id_kind = common::infer_artifact_id_kind(assembly_id);
  if (id_kind == common::ArtifactIdKind::kMi2) {
    result.sealed_artifact_id = std::string(assembly_id);
    result.already_sealed = true;
    auto parse_or = parse_mi2_multihashes(assembly_id);
    if (!parse_or.ok()) {
      return parse_or.status();
    }
    result.index_multihash = parse_or->first;
    result.data_multihash = parse_or->second;
    if (!publish_canonical) {
      return result;
    }
  }

  std::string sealed_artifact_id;
  if (result.sealed_artifact_id.empty()) {
    auto binding_or = gs_client->get_artifact_binding(assembly_id);
    if (binding_or.ok()) {
      result.sealed_artifact_id = binding_or->to_artifact_id;
      result.already_sealed = true;
      auto parse_or = parse_mi2_multihashes(result.sealed_artifact_id);
      if (!parse_or.ok()) {
        return parse_or.status();
      }
      result.index_multihash = parse_or->first;
      result.data_multihash = parse_or->second;
      if (!publish_canonical) {
        return result;
      }
    } else if (!absl::IsNotFound(binding_or.status())) {
      return binding_or.status();
    }
  }

  if (id_kind == common::ArtifactIdKind::kUnspecified) {
    return absl::InvalidArgumentError(R"(seal_assembly requires "cgid:" or "mi2:" artifact id)");
  }

  auto index_or = gs_client->get_artifact_index_by_id(assembly_id);
  if (!index_or.ok()) {
    return index_or.status();
  }
  std::string canonical_index_json = std::move(*index_or);

  auto canonical_total_or = compute_logical_total_size(canonical_index_json);
  if (!canonical_total_or.ok()) {
    return canonical_total_or.status();
  }
  const uint64_t canonical_total_size = *canonical_total_or;
  result.total_size = canonical_total_size;

  auto index_mh_or =
      common::compute_index_multihash(std::optional<std::string>(canonical_index_json), std::string_view());
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }
  if (!result.index_multihash.empty() && result.index_multihash != *index_mh_or) {
    return absl::FailedPreconditionError("index multihash does not match sealed artifact id");
  }
  result.index_multihash = *index_mh_or;

  auto target_device_or = select_seal_target_device(config_.runtime_context->device_manager());
  if (!target_device_or.ok()) {
    return target_device_or.status();
  }
  const DeviceKey target_device = *target_device_or;

  auto target_ranges_or = build_target_ranges_for_canonical(canonical_index_json, canonical_total_size);
  if (!target_ranges_or.ok()) {
    return target_ranges_or.status();
  }
  std::vector<AssemblyTargetRange> target_ranges = std::move(*target_ranges_or);

  auto plan_or = build_assembly_plan(
      *gs_client,
      assembly_id,
      canonical_index_json,
      canonical_total_size,
      absl::MakeSpan(target_ranges),
      canonical_total_size,
      target_device,
      config_.runtime_context->worker_identity());
  if (!plan_or.ok()) {
    return plan_or.status();
  }
  AssemblyPlan plan = std::move(*plan_or);
  if (!plan.missing_ranges.empty()) {
    return absl::UnavailableError(
        absl::StrCat("seal_assembly missing canonical ranges: ", format_missing_ranges(plan.missing_ranges)));
  }

  auto comm_manager = config_.runtime_context->communication_manager();
  if (!comm_manager || !comm_manager->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }

  std::vector<std::shared_ptr<loader::SeekableSource>> piece_sources;
  piece_sources.reserve(plan.sources.size());
  for (const auto& source : plan.sources) {
    const auto& remote = source.session.remote_replica;
    if (remote.remote_memory_keys.empty()) {
      return absl::FailedPreconditionError(absl::StrCat("remote memory keys missing for view_id=", source.view_id));
    }
    if (remote.buffer_sizes.size() != remote.remote_memory_keys.size()) {
      return absl::FailedPreconditionError(absl::StrCat("buffer size mismatch for view_id=", source.view_id));
    }
    std::vector<size_t> buffer_sizes;
    buffer_sizes.reserve(remote.buffer_sizes.size());
    for (uint64_t size : remote.buffer_sizes) {
      buffer_sizes.push_back(static_cast<size_t>(size));
    }
    loader::RemoteKeySource::Options opts{
        .comm_engine =
            gsl::not_null<std::shared_ptr<tensorcast::communicator::engine::Communicator>>{
                comm_manager->get_shared_engine()},
        .memory_keys = remote.remote_memory_keys,
        .buffer_sizes = std::move(buffer_sizes),
        .ip = remote.node_address,
        .port = static_cast<uint16_t>(remote.node_port),
        .total_size = remote.memory_size,
    };
    piece_sources.push_back(std::make_shared<loader::RemoteKeySource>(std::move(opts)));
  }

  const size_t leaf_chunk_bytes =
      config_.artifact_chunk_bytes == 0 ? static_cast<size_t>(4ULL * 1024 * 1024) : config_.artifact_chunk_bytes;
  if (result.sealed_artifact_id.empty()) {
    std::vector<AssemblySegment> hash_segments = plan.segments;
    AssemblySource hash_source(piece_sources, std::move(hash_segments), plan.total_size);
    auto data_mh_or =
        loader::compute_data_multihash_from_seekable_source(hash_source, plan.total_size, leaf_chunk_bytes);
    if (!data_mh_or.ok()) {
      return data_mh_or.status();
    }
    result.data_multihash = *data_mh_or;
    result.sealed_artifact_id = absl::StrCat("mi2:", result.index_multihash, ":", result.data_multihash);

    components::ArtifactBinding binding;
    binding.from_artifact_id = std::string(assembly_id);
    binding.to_artifact_id = result.sealed_artifact_id;
    binding.kind = tensorcast::global_store::v1::ARTIFACT_BINDING_KIND_SEAL;
    auto upsert_or = gs_client->upsert_artifact_binding(binding);
    if (!upsert_or.ok()) {
      return upsert_or.status();
    }
    if (!upsert_or->created) {
      result.already_sealed = true;
      result.sealed_artifact_id = upsert_or->binding.to_artifact_id;
      auto parse_or = parse_mi2_multihashes(result.sealed_artifact_id);
      if (!parse_or.ok()) {
        return parse_or.status();
      }
      result.index_multihash = parse_or->first;
      result.data_multihash = parse_or->second;
    }
  }

  if (!publish_canonical) {
    return result;
  }

  auto registry = &config_.replica_runtime->registry();
  loading::ReplicaKey key;
  key.artifact_id = result.sealed_artifact_id;
  key.device = target_device;
  key.replica = 0;

  auto existing_or = registry->find(key);
  if (!existing_or.ok() && !absl::IsNotFound(existing_or.status())) {
    return existing_or.status();
  }
  if (!existing_or.ok()) {
    loading::InlineBufferSource inline_source{.data = nullptr, .size_bytes = plan.total_size};
    replica::ReplicaConfig cfg{
        .source = inline_source,
        .artifact_identifier = key.artifact_id,
        .device_type = DeviceType::GPU,
        .local_device_id = key.device.ordinal,
        .pinned_buffer_pool = config_.runtime_context->pinned_buffer_pool(),
        .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{config_.runtime_context->async_runtime()},
        .artifact_chunk_bytes = config_.artifact_chunk_bytes,
        .expected_artifact_size = plan.total_size,
        .memory_tier_config = config_.options->memory_tier_config,
    };
    cfg.pinned_memory_timeout = config_.pinned_memory_timeout;
    cfg.streaming_buffer_chunks = std::max<size_t>(1, config_.runtime_context->options().streaming_buffer_chunks);

    auto replica_or = replica::Replica::create(cfg);
    if (!replica_or.ok()) {
      return replica_or.status();
    }
    auto replica = std::shared_ptr<replica::Replica>(std::move(replica_or.value()));

    auto source = std::make_unique<AssemblySource>(piece_sources, plan.segments, plan.total_size);
    const int concurrency = std::max(1, config_.num_threads);
    auto load_future = replica->get_memory_manager().load_async_from_source(
        std::move(source),
        common::memory::MemoryLocation::GPU,
        concurrency,
        std::nullopt,
        std::function<absl::Status()>{});
    absl::Status load_status = std::move(load_future).get();
    if (!load_status.ok()) {
      return load_status;
    }
    replica->set_ready_signal(common::memory::MemoryLocation::GPU, absl::OkStatus());

    absl::Status emplace_status = registry->emplace(key, gsl::not_null{replica});
    if (!emplace_status.ok() && !absl::IsAlreadyExists(emplace_status)) {
      return emplace_status;
    }
  }

  absl::Status publish_status = register_replica_with_global_store(key, {});
  if (!publish_status.ok()) {
    return publish_status;
  }

  return result;
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return run_disk_ingestion_internal(artifact_identifier, source, target, hints, /*publish_to_global_store=*/false);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::ingest_from_disk(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  return run_disk_ingestion_internal(artifact_identifier, source, target, hints, publish_to_global_store);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints) {
  return run_p2p_ingestion_internal(artifact_identifier, source, target, hints, /*publish_to_global_store=*/false);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::ingest_from_p2p(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  return run_p2p_ingestion_internal(artifact_identifier, source, target, hints, publish_to_global_store);
}

absl::Status MaterializationFacade::register_replica_with_global_store(
    const loading::ReplicaKey& key,
    std::string_view artifact_id_override,
    std::string_view publish_context_id) {
  std::string context = publish_context_id.empty() ? "" : std::string(publish_context_id);
  if (context.empty()) {
    auto stored_context = lookup_publish_context_for_replica(key);
    if (stored_context.has_value()) {
      context = *stored_context;
    }
  } else {
    record_publish_context_for_replica(key, context);
  }

  if (hooks_ && hooks_->register_replica_override) {
    return hooks_->register_replica_override(key, artifact_id_override, context);
  }
  return config_.metadata_gateway->register_replica(key, artifact_id_override, context);
}

template <typename SourceT, typename RunnerFn>
absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::run_pipeline_ingestion(
    IngestionSource source_type,
    const std::string& artifact_identifier,
    const SourceT& /*source*/,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store,
    RunnerFn&& runner) {
  const std::string request_id = make_request_id(source_type == IngestionSource::kDisk ? "disk" : "p2p");
  const std::string publish_context_id =
      publish_to_global_store ? config_.runtime_context->mint_publish_context_id() : std::string();
  const loading::MaterializeMode mode =
      source_type == IngestionSource::kP2P ? loading::MaterializeMode::COPY_ONLY : loading::MaterializeMode::LOAD_ONLY;
  IngestionRequestMetadata metadata{
      .request_id = request_id,
      .artifact_identifier = artifact_identifier,
      .source = source_type,
      .target = target,
      .publish_context_id = publish_context_id,
      .publish_to_global_store = publish_to_global_store,
      .materialize_mode = mode,
      .hints = hints,
  };
  maybe_invoke_before_pipeline_start(metadata);

  const IngestionStartedEvent started_event = make_started_event(
      request_id, artifact_identifier, source_type, target, publish_context_id, publish_to_global_store, mode, hints);
  publish_started_event(started_event);

  IngestionResultEvent defaults = make_ingestion_event_seed(
      request_id, artifact_identifier, source_type, target, publish_to_global_store, publish_context_id, mode, hints);

  if (auto override_result = maybe_override_result(); override_result.has_value()) {
    IngestionResultEvent event = defaults;
    if (!override_result->ok()) {
      event.status = override_result->status();
      maybe_mutate_completion_event(event);
      publish_completed_event(std::move(event));
      return override_result->status();
    }
    auto handle = std::move(override_result->value());
    event.replica_key = handle.key();
    maybe_mutate_completion_event(event);
    if (publish_to_global_store && !publish_context_id.empty()) {
      record_publish_context_for_replica(handle.key(), publish_context_id);
    }
    publish_completed_event(event);
    return handle;
  }

  IngestionResultEvent pipeline_event;
  auto handle_or = runner(request_id, publish_context_id, &pipeline_event);
  if (!handle_or.ok()) {
    IngestionResultEvent failure_event = pipeline_event.request_id.empty() ? defaults : pipeline_event;
    apply_event_defaults(failure_event, defaults);
    failure_event.status = handle_or.status();
    maybe_mutate_completion_event(failure_event);
    publish_completed_event(std::move(failure_event));
    return handle_or.status();
  }

  auto handle = std::move(handle_or.value());
  apply_event_defaults(pipeline_event, defaults);
  if (!pipeline_event.replica_key.has_value()) {
    pipeline_event.replica_key = handle.key();
  }
  maybe_mutate_completion_event(pipeline_event);
  if (publish_to_global_store && !pipeline_event.publish_context_id.empty()) {
    record_publish_context_for_replica(handle.key(), pipeline_event.publish_context_id);
  }
  publish_completed_event(pipeline_event);
  return handle;
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::run_disk_ingestion_internal(
    const std::string& artifact_identifier,
    const loading::DiskSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  auto runner = [&](const std::string& request_id,
                    const std::string& publish_context_id,
                    IngestionResultEvent* event_out) {
    return pipeline_->ingest_from_disk(
        artifact_identifier, source, target, hints, publish_to_global_store, event_out, request_id, publish_context_id);
  };

  return run_pipeline_ingestion(
      IngestionSource::kDisk, artifact_identifier, source, target, hints, publish_to_global_store, runner);
}

absl::StatusOr<loading::ReplicaHandle> MaterializationFacade::run_p2p_ingestion_internal(
    const std::string& artifact_identifier,
    const P2PSource& source,
    const loading::ReplicaTarget& target,
    const loading::MaterializeHints& hints,
    bool publish_to_global_store) {
  auto runner = [&](const std::string& request_id,
                    const std::string& publish_context_id,
                    IngestionResultEvent* event_out) {
    return pipeline_->ingest_from_p2p(
        artifact_identifier, source, target, hints, publish_to_global_store, event_out, request_id, publish_context_id);
  };

  return run_pipeline_ingestion(
      IngestionSource::kP2P, artifact_identifier, source, target, hints, publish_to_global_store, runner);
}

std::string MaterializationFacade::make_request_id(std::string_view prefix) {
  const uint64_t sequence = request_counter_.fetch_add(1, std::memory_order_relaxed);
  const int64_t timestamp = absl::ToUnixNanos(absl::Now());
  return absl::StrCat(prefix, "_", timestamp, "_", sequence);
}

IngestionResultEvent MaterializationFacade::make_ingestion_event_seed(
    const std::string& request_id,
    std::string_view artifact_identifier,
    IngestionSource source,
    const loading::ReplicaTarget& target,
    bool publish_to_global_store,
    const std::string& publish_context_id,
    loading::MaterializeMode mode,
    const loading::MaterializeHints& hints) const {
  IngestionResultEvent event;
  event.request_id = request_id;
  event.source = source;
  event.materialize_mode = mode;
  event.artifact_id = std::string(artifact_identifier);
  event.target_device = target.location.to_device_key();
  event.target_location = target.location.type;
  event.publish_to_global_store = publish_to_global_store;
  event.publish_context_id = publish_context_id;
  event.status = absl::OkStatus();
  if (hints.variant && hints.variant->view_id.has_value()) {
    event.view_id = hints.variant->view_id;
  }
  return event;
}

IngestionStartedEvent MaterializationFacade::make_started_event(
    const std::string& request_id,
    std::string_view artifact_identifier,
    IngestionSource source,
    const loading::ReplicaTarget& target,
    const std::string& publish_context_id,
    bool publish_to_global_store,
    loading::MaterializeMode mode,
    const loading::MaterializeHints& hints) const {
  IngestionStartedEvent started;
  started.request_id = request_id;
  started.artifact_id = std::string(artifact_identifier);
  started.source = source;
  started.target = target;
  started.publish_context_id = publish_context_id;
  started.publish_to_global_store = publish_to_global_store;
  started.materialize_mode = mode;
  if (hints.variant && hints.variant->view_id.has_value()) {
    started.view_id = hints.variant->view_id;
  }
  return started;
}

void MaterializationFacade::publish_started_event(const IngestionStartedEvent& event) const {
  if (ingestion_event_hub_ != nullptr) {
    ingestion_event_hub_->publish_started(event);
  }
}

void MaterializationFacade::publish_completed_event(IngestionCompletedEvent event) const {
  if (ingestion_event_hub_ != nullptr) {
    ingestion_event_hub_->publish_completed(event);
  }
}

void MaterializationFacade::apply_event_defaults(IngestionResultEvent& event, const IngestionResultEvent& defaults)
    const {
  if (event.request_id.empty()) {
    event.request_id = defaults.request_id;
  }
  if (event.artifact_id.empty()) {
    event.artifact_id = defaults.artifact_id;
  }
  event.source = defaults.source;
  event.materialize_mode = defaults.materialize_mode;
  event.target_device = defaults.target_device;
  event.target_location = defaults.target_location;
  if (!event.view_id.has_value() && defaults.view_id.has_value()) {
    event.view_id = defaults.view_id;
  }
  event.publish_to_global_store = defaults.publish_to_global_store;
  if (event.publish_context_id.empty()) {
    event.publish_context_id = defaults.publish_context_id;
  }
}

std::optional<absl::StatusOr<loading::ReplicaHandle>> MaterializationFacade::maybe_override_result() const {
  if (!hooks_ || !hooks_->override_result) {
    return std::nullopt;
  }
  return hooks_->override_result();
}

void MaterializationFacade::maybe_invoke_before_pipeline_start(const IngestionRequestMetadata& metadata) const {
  if (!hooks_ || !hooks_->before_pipeline_start) {
    return;
  }
  hooks_->before_pipeline_start(metadata);
}

void MaterializationFacade::maybe_mutate_completion_event(IngestionResultEvent& event) const {
  if (!hooks_ || !hooks_->mutate_completion_event) {
    return;
  }
  hooks_->mutate_completion_event(event);
}

void MaterializationFacade::record_publish_context_for_replica(
    const loading::ReplicaKey& key,
    std::string_view publish_context_id) {
  if (publish_context_id.empty()) {
    return;
  }
  absl::MutexLock lock(&publish_context_mu_);
  publish_context_by_replica_[key] = std::string(publish_context_id);
}

std::optional<std::string> MaterializationFacade::lookup_publish_context_for_replica(
    const loading::ReplicaKey& key) const {
  absl::MutexLock lock(&publish_context_mu_);
  auto it = publish_context_by_replica_.find(key);
  if (it == publish_context_by_replica_.end()) {
    return std::nullopt;
  }
  return it->second;
}

} // namespace tensorcast::store::runtime::ingestion
