// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/metadata/registration_backend.h"

#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <unordered_map>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_verification.h"
#include "core/common/memory/cuda_memory.h"
#include "core/common/memory/host_memory.h"
#include "core/common/trace/trace_macros.h"
#include "core/cuda/cuda_api.h"
#include "core/store/components/eviction_service.h"
#include "core/store/components/stable_dram_cache_manager.h"
#include "core/store/device_registry.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/sources/byte_range_map_builder.h"
#include "core/store/materialization/dataplane/sources/byte_range_mapped_source.h"
#include "core/store/materialization/dataplane/sources/byte_range_program.h"
#include "core/store/materialization/dataplane/sources/memory_source.h"
#include "core/store/materialization/dataplane/verification/verification_utils.h"
#include "core/store/materialization/dataplane/view/view_ingest_executor.h"
#include "core/store/materialization/dataplane/view/view_plan_source.h"
#include "core/store/memory_tier_budget.h"
#include "core/store/view_utils.h"
#include "nlohmann/json.hpp"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::store::runtime::metadata {

namespace {

constexpr uint64_t kBytesPerGiB = 1024ULL * 1024ULL * 1024ULL;
constexpr uint64_t kStableBeginMinHeadroomBytes = 2ULL * kBytesPerGiB;

uint64_t sum_view_write_bytes(const loader::ViewWritePlan& write_plan) {
  uint64_t total = 0;
  for (const auto& chunk : write_plan.chunks) {
    total += chunk.length;
  }
  return total;
}

std::vector<CanonicalRange> canonical_ranges_from_write_plan(const loader::ViewWritePlan& write_plan) {
  std::vector<CanonicalRange> ranges;
  ranges.reserve(write_plan.chunks.size());
  for (const auto& chunk : write_plan.chunks) {
    CanonicalRange range;
    range.offset = chunk.canonical_offset;
    range.length = chunk.length;
    ranges.push_back(range);
  }
  std::sort(ranges.begin(), ranges.end(), [](const CanonicalRange& a, const CanonicalRange& b) {
    return a.offset < b.offset;
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

absl::Status add_non_overlapping_ingested_range(
    std::map<uint64_t, uint64_t>& ranges,
    uint64_t offset,
    uint64_t length,
    uint64_t* ingested_unique_bytes) {
  if (length == 0) {
    return absl::OkStatus();
  }
  const uint64_t end = offset + length;
  uint64_t merged_start = offset;
  uint64_t merged_end = end;
  auto it = ranges.lower_bound(offset);

  if (it != ranges.begin()) {
    auto prev = std::prev(it);
    if (prev->second > offset) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "registration chunk overlaps previously ingested range: existing=[",
              prev->first,
              ",",
              prev->second,
              "), incoming=[",
              offset,
              ",",
              end,
              ")"));
    }
    if (prev->second == offset) {
      merged_start = prev->first;
      merged_end = std::max<uint64_t>(merged_end, prev->second);
      it = ranges.erase(prev);
    }
  }

  while (it != ranges.end() && it->first <= merged_end) {
    if (it->first < merged_end) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "registration chunk overlaps previously ingested range: existing=[",
              it->first,
              ",",
              it->second,
              "), incoming=[",
              offset,
              ",",
              end,
              ")"));
    }
    merged_end = std::max<uint64_t>(merged_end, it->second);
    it = ranges.erase(it);
  }

  ranges.emplace(merged_start, merged_end);
  *ingested_unique_bytes += length;
  return absl::OkStatus();
}

bool ingested_ranges_cover_full_span(const std::map<uint64_t, uint64_t>& ranges, uint64_t total_size_bytes) {
  if (total_size_bytes == 0) {
    return true;
  }
  if (ranges.size() != 1) {
    return false;
  }
  const auto it = ranges.begin();
  return it->first == 0 && it->second == total_size_bytes;
}

uint64_t ingested_prefix_bytes(const std::map<uint64_t, uint64_t>& ranges) {
  uint64_t cursor = 0;
  for (const auto& [start, end] : ranges) {
    if (start != cursor) {
      break;
    }
    cursor = end;
  }
  return cursor;
}

absl::Status enforce_stable_begin_runtime_memory_guard(
    uint64_t artifact_size_bytes,
    const std::shared_ptr<MemoryTierBudget>& memory_tier_budget) {
  auto available_or = common::memory::detect_host_memory_available_bytes();
  if (!available_or.ok()) {
    VLOG(1) << "stable begin runtime memory guard skipped: " << available_or.status();
    return absl::OkStatus();
  }

  uint64_t reusable_stable_bytes = 0;
  if (memory_tier_budget != nullptr) {
    const auto budget = memory_tier_budget->snapshot();
    reusable_stable_bytes = budget.stable_total_bytes > budget.stable_used_bytes
        ? (budget.stable_total_bytes - budget.stable_used_bytes)
        : 0;
  }
  const uint64_t bytes_requiring_new_allocation =
      artifact_size_bytes > reusable_stable_bytes ? (artifact_size_bytes - reusable_stable_bytes) : 0;
  const uint64_t guard_headroom = std::max<uint64_t>(kStableBeginMinHeadroomBytes, artifact_size_bytes / 10U);
  const uint64_t required_available = bytes_requiring_new_allocation + guard_headroom;
  if (*available_or >= required_available) {
    return absl::OkStatus();
  }

  return absl::ResourceExhaustedError(
      absl::StrCat(
          "insufficient available host memory for stable registration begin: available_bytes=",
          *available_or,
          ", required_bytes=",
          required_available,
          " (artifact_bytes=",
          artifact_size_bytes,
          ", reusable_stable_bytes=",
          reusable_stable_bytes,
          ", allocation_bytes=",
          bytes_requiring_new_allocation,
          ", headroom_bytes=",
          guard_headroom,
          "). rejecting begin to avoid daemon OOM"));
}

std::vector<components::CanonicalRange> to_component_ranges(const std::vector<CanonicalRange>& ranges) {
  std::vector<components::CanonicalRange> out;
  out.reserve(ranges.size());
  for (const auto& range : ranges) {
    components::CanonicalRange converted;
    converted.offset = range.offset;
    converted.length = range.length;
    out.push_back(converted);
  }
  return out;
}

absl::Status zero_view_padding(
    const loader::ViewWritePlan& write_plan,
    uint64_t view_size_bytes,
    common::memory::MemoryLocation location,
    void* base_ptr,
    int device_id) {
  if (view_size_bytes == 0) {
    return absl::InvalidArgumentError("view_size_bytes must be > 0");
  }
  if (base_ptr == nullptr) {
    return absl::InvalidArgumentError("view base pointer is null");
  }

  std::vector<loader::ViewWritePlan::Chunk> chunks = write_plan.chunks;
  std::sort(chunks.begin(), chunks.end(), [](const auto& a, const auto& b) { return a.view_offset < b.view_offset; });

  auto zero_span = [&](uint64_t offset, uint64_t length) -> absl::Status {
    if (length == 0) {
      return absl::OkStatus();
    }
    auto* dst = static_cast<uint8_t*>(base_ptr) + static_cast<std::ptrdiff_t>(offset);
    switch (location) {
      case common::memory::MemoryLocation::CPU:
        std::memset(dst, 0, static_cast<size_t>(length));
        return absl::OkStatus();
      case common::memory::MemoryLocation::GPU: {
        if (!cuda::is_fake()) {
          auto dev_status = cuda::set_device(device_id);
          if (!dev_status.ok()) {
            return dev_status;
          }
        }
        auto ms = cuda::memset(dst, 0, static_cast<size_t>(length));
        if (!ms.ok()) {
          return ms;
        }
        return absl::OkStatus();
      }
      default:
        return absl::InvalidArgumentError("unsupported memory location for view padding");
    }
  };

  uint64_t cursor = 0;
  for (const auto& chunk : chunks) {
    if (chunk.view_offset > cursor) {
      auto st = zero_span(cursor, chunk.view_offset - cursor);
      if (!st.ok()) {
        return st;
      }
    }
    const uint64_t end = chunk.view_offset + chunk.length;
    cursor = std::max(cursor, end);
  }
  if (cursor < view_size_bytes) {
    auto st = zero_span(cursor, view_size_bytes - cursor);
    if (!st.ok()) {
      return st;
    }
  }

  if (location == common::memory::MemoryLocation::GPU && !cuda::is_fake()) {
    auto sync = cuda::device_synchronize();
    if (!sync.ok()) {
      return sync;
    }
  }

  return absl::OkStatus();
}

std::string make_registration_id() {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;
  return absl::StrCat("reg_", absl::ToUnixNanos(absl::Now()), "_", getpid(), "_", dis(gen));
}

absl::StatusOr<std::pair<std::string, std::string>> parse_mi2_multihashes(std::string_view artifact_id) {
  constexpr std::string_view kPrefix = "mi2:";
  if (!artifact_id.starts_with(kPrefix)) {
    return absl::InvalidArgumentError("artifact_id_override must start with \"mi2:\"");
  }
  const size_t index_begin = kPrefix.size();
  const size_t sep = artifact_id.find(':', index_begin);
  if (sep == std::string_view::npos) {
    return absl::InvalidArgumentError("artifact_id_override must be of form mi2:<index_multihash>:<data_multihash>");
  }
  const std::string_view index_mh = artifact_id.substr(index_begin, sep - index_begin);
  const std::string_view data_mh = artifact_id.substr(sep + 1);
  if (index_mh.empty() || data_mh.empty()) {
    return absl::InvalidArgumentError("artifact_id_override must include index and data multihash components");
  }
  return std::make_pair(std::string(index_mh), std::string(data_mh));
}

constexpr uint64_t kProofChunkBytesV1 = 4ULL * 1024 * 1024;
constexpr std::string_view kProofSchemaV1 = "v1";

struct TensorInterval {
  std::string tensor_name;
  uint64_t offset{0};
  uint64_t size_bytes{0};
};

absl::StatusOr<std::vector<TensorInterval>> parse_tensor_intervals(std::string_view canonical_index_json) {
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("canonical_index_json must not be empty");
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

  std::vector<TensorInterval> out;
  out.reserve(j.size());
  for (auto it = j.begin(); it != j.end(); ++it) {
    const std::string tensor_name = it.key();
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() < 2) {
      continue;
    }
    TensorInterval interval;
    interval.tensor_name = tensor_name;
    interval.offset = arr[0].get<uint64_t>();
    interval.size_bytes = arr[1].get<uint64_t>();
    out.push_back(std::move(interval));
  }

  std::sort(
      out.begin(), out.end(), [](const TensorInterval& a, const TensorInterval& b) { return a.offset < b.offset; });
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

absl::Status read_view_bytes(
    common::memory::MemoryLocation location,
    void* base_ptr,
    int device_id,
    uint64_t view_offset,
    absl::Span<uint8_t> dst) {
  if (dst.empty()) {
    return absl::OkStatus();
  }
  if (base_ptr == nullptr) {
    return absl::InvalidArgumentError("read_view_bytes requires non-null base_ptr");
  }
  auto* src = static_cast<uint8_t*>(base_ptr) + static_cast<std::ptrdiff_t>(view_offset);
  switch (location) {
    case common::memory::MemoryLocation::CPU:
      std::memcpy(dst.data(), src, dst.size());
      return absl::OkStatus();
    case common::memory::MemoryLocation::GPU: {
      if (!cuda::is_fake()) {
        auto set_status = cuda::set_device(device_id);
        if (!set_status.ok()) {
          return set_status;
        }
      }
      return cuda::memcpy(dst.data(), src, dst.size(), cudaMemcpyDeviceToHost);
    }
    default:
      return absl::InvalidArgumentError("unsupported memory location for view reads");
  }
}

struct CanonicalToViewSpan {
  uint64_t canonical_offset{0};
  uint64_t view_offset{0};
  uint64_t length{0};
};

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

  std::sort(spans.begin(), spans.end(), [](const CanonicalToViewSpan& a, const CanonicalToViewSpan& b) {
    return a.canonical_offset < b.canonical_offset;
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

} // namespace

struct RegistrationBackend::PendingRegistrationContext {
  enum class Plan : uint8_t { kCoalesced = 0, kStableDram = 1 };
  enum class StableCpuIngestMode : uint8_t { kUnset = 0, kStreamMemcpy = 1, kRangeAckOnly = 2 };

  mutable std::mutex stream_ingest_mu;
  std::string registration_id;
  std::string artifact_id;
  std::string client_artifact_id;
  std::optional<std::string> artifact_id_override;
  int device_id{0};
  uint64_t size_bytes{0};
  std::string tensor_index_key;
  std::optional<std::string> tensor_index_data;
  std::string schema_version;
  std::string encoding;
  bool enable_p2p{true};
  common::ArtifactIdKind id_kind{common::ArtifactIdKind::kMi2};
  std::shared_ptr<replica::Replica> replica;
  loading::ReplicaKey pending_registry_key;
  void* gpu_ptr{nullptr};
  std::byte* stable_dram_cpu_base{nullptr};
  cudaIpcMemHandle_t ipc_handle{};
  std::unique_ptr<common::memory::GpuDeviceMemory> staging_gpu;
  StableDramOptions stable_dram;
  uint64_t stream_ingested_bytes{0};
  uint64_t stream_copied_bytes{0};
  uint64_t stream_chunk_count{0};
  uint64_t stream_inflight_chunks{0};
  uint64_t stream_memcpy_nanos{0};
  std::map<uint64_t, uint64_t> stream_ingested_ranges;
  StableCpuIngestMode stable_cpu_ingest_mode{StableCpuIngestMode::kUnset};
  std::optional<components::StableDramCachePolicy> stable_cache_policy;
  std::chrono::steady_clock::time_point expiry_time;
  std::chrono::steady_clock::time_point begin_time;
  Plan plan{Plan::kCoalesced};

  struct ViewState {
    ViewRegistration options;
    loader::BidirectionalViewPlan plan;
    uint64_t expected_view_bytes{0};
    uint64_t view_size_bytes{0};
    uint64_t ingested_bytes{0};
    bool finalized{false};
    std::unique_ptr<loader::ViewIngestExecutor> executor;
  };

  std::unique_ptr<ViewState> view_state;
};

RegistrationBackend::RegistrationBackend(
    RegistrationResources resources,
    ReplicaFactory replica_factory,
    size_t artifact_chunk_bytes,
    std::chrono::milliseconds pinned_memory_timeout,
    size_t streaming_buffer_chunks,
    RegistrationPublisher* publisher)
    : device_manager_(resources.device_manager),
      replica_registry_(resources.replica_registry),
      metrics_collector_(resources.metrics_collector),
      memory_pool_(resources.memory_pool),
      communication_manager_(std::move(resources.communication_manager)),
      stable_cache_manager_(std::move(resources.stable_cache_manager)),
      async_runtime_(std::move(resources.async_runtime)),
      memory_tier_budget_(std::move(resources.memory_tier_budget)),
      memory_tier_config_(resources.memory_tier_config),
      cpu_shared_memory_enabled_(resources.cpu_shared_memory_enabled),
      promotion_manager_(resources.promotion_manager),
      replica_factory_(std::move(replica_factory)),
      artifact_chunk_bytes_(artifact_chunk_bytes),
      pinned_memory_timeout_(pinned_memory_timeout),
      streaming_buffer_chunks_(std::max<size_t>(1, streaming_buffer_chunks)),
      publisher_(publisher),
      byte_mapping_config_(resources.byte_mapping_config) {
  ABSL_CHECK(replica_factory_) << "ReplicaFactory must be provided";
  ABSL_CHECK(async_runtime_ != nullptr) << "RegistrationResources.async_runtime is required";
}

absl::StatusOr<RegistrationBeginResult> RegistrationBackend::begin(const ArtifactRegistration& reg) {
  std::string trace_request_id = reg.tensor_index_key;
  if (trace_request_id.empty()) {
    trace_request_id = reg.artifact_id;
  }
  if (trace_request_id.empty()) {
    trace_request_id = "registration";
  }
  std::string trace_artifact_id = reg.artifact_id.empty() ? trace_request_id : reg.artifact_id;
  SC_TRACE_INIT_GUARD(trace_request_id, trace_artifact_id, "registration_begin");

  if (reg.total_size_bytes == 0) {
    return absl::InvalidArgumentError("total_size_bytes must be > 0");
  }
  const bool stable_dram = reg.plan == RegistrationPlan::kStableDram;
  if (reg.device_id < 0) {
    return absl::InvalidArgumentError("device_id must be >= 0");
  }
  if (stable_dram) {
    auto guard_status = enforce_stable_begin_runtime_memory_guard(reg.total_size_bytes, memory_tier_budget_);
    if (!guard_status.ok()) {
      return guard_status;
    }
  }
  if (!reg.schema_version.empty() && reg.schema_version != "v3") {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported ArtifactRegistration.schema_version='", reg.schema_version, "'; expected 'v3'"));
  }
  if (reg.tensor_index_key.empty() && !reg.tensor_index_data.has_value()) {
    return absl::InvalidArgumentError("tensor index key or data must be provided");
  }
  if (reg.client_artifact_id.has_value() && reg.artifact_id_override.has_value()) {
    return absl::InvalidArgumentError("artifact_id_override cannot be set when client_artifact_id is provided");
  }
  if (reg.artifact_id_override.has_value()) {
    if (reg.artifact_id_override->empty()) {
      return absl::InvalidArgumentError("artifact_id_override must be non-empty when provided");
    }
    auto parsed_or = parse_mi2_multihashes(*reg.artifact_id_override);
    if (!parsed_or.ok()) {
      return parsed_or.status();
    }
  }

  uint64_t canonical_size = reg.total_size_bytes;

  std::optional<loader::BidirectionalViewPlan> view_plan;
  uint64_t expected_view_bytes = 0;
  uint64_t view_size_bytes = 0;
  std::vector<CanonicalRange> canonical_ranges;
  if (reg.view.has_value()) {
    const auto& view_opts = *reg.view;
    if (!reg.tensor_index_data.has_value() || reg.tensor_index_data->empty()) {
      return absl::InvalidArgumentError("view registration requires inline canonical index data");
    }
    if (view_opts.view_id.empty()) {
      return absl::InvalidArgumentError("view registration requires a non-empty view_id");
    }
    if (view_opts.placement == ViewPlacement::kUnspecified) {
      return absl::InvalidArgumentError("view registration requires explicit placement");
    }
    if (view_opts.registration_kind == ViewRegistrationKind::kUnspecified) {
      return absl::InvalidArgumentError("view.registration_kind must be specified");
    }
    if (view_opts.registration_kind == ViewRegistrationKind::kPiece) {
      if (view_opts.canonical_size_bytes == 0) {
        return absl::InvalidArgumentError("piece registration requires canonical_size_bytes");
      }
      canonical_size = view_opts.canonical_size_bytes;
      if (!reg.client_artifact_id.has_value() || reg.client_artifact_id->empty()) {
        return absl::InvalidArgumentError("piece registration requires client_artifact_id (cgid)");
      }
    } else {
      if (view_opts.canonical_size_bytes != 0 && view_opts.canonical_size_bytes != reg.total_size_bytes) {
        return absl::InvalidArgumentError("view.canonical_size_bytes must match total_size_bytes");
      }
      canonical_size = view_opts.canonical_size_bytes != 0 ? view_opts.canonical_size_bytes : reg.total_size_bytes;
    }
    auto plan_or = loader::ViewPlanner::compute_bidirectional_view_plan(*reg.tensor_index_data, view_opts.spec);
    if (!plan_or.ok()) {
      return plan_or.status();
    }
    view_plan = std::move(*plan_or);
    expected_view_bytes = sum_view_write_bytes(view_plan->write);
    view_size_bytes = view_plan->forward.view_size_bytes;
    canonical_ranges = canonical_ranges_from_write_plan(view_plan->write);
    uint64_t covered_bytes = 0;
    for (const auto& range : canonical_ranges) {
      covered_bytes += range.length;
    }
    if (covered_bytes > canonical_size) {
      return absl::InvalidArgumentError("view registration exceeds canonical byte space");
    }
    if (view_opts.registration_kind == ViewRegistrationKind::kPiece) {
      if (reg.total_size_bytes != view_size_bytes) {
        return absl::InvalidArgumentError("piece registration total_size_bytes must equal view_size_bytes");
      }
      if (ranges_cover_interval(canonical_ranges, 0, canonical_size)) {
        return absl::InvalidArgumentError(
            "piece registration must not fully cover canonical bytes; use registration_kind=CANONICAL");
      }
    } else {
      if (covered_bytes != canonical_size) {
        return absl::InvalidArgumentError(
            "canonical view registration must fully cover canonical bytes; use registration_kind=PIECE for partial");
      }
      if (view_opts.placement == ViewPlacement::kServer && view_plan->inverse_transform.requires_materialization) {
        auto info_or = device_manager_->get_gpu_info(reg.device_id);
        if (!info_or.ok()) {
          return absl::FailedPreconditionError(
              absl::StrCat(
                  "SERVER placement for view registration requires GPU transpose support on device ",
                  reg.device_id,
                  "; retry with placement=CLIENT (",
                  info_or.status().message(),
                  ")"));
        }
      }
    }
  }

  loading::InlineBufferSource ib_source{.data = nullptr, .size_bytes = reg.total_size_bytes};
  replica::ReplicaConfig cfg{
      .source = ib_source,
      .artifact_identifier = reg.artifact_id,
      .device_type = stable_dram ? DeviceType::CPU : DeviceType::GPU,
      .local_device_id = stable_dram ? -1 : reg.device_id,
      .pinned_buffer_pool = memory_pool_,
      .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{async_runtime_},
      .artifact_chunk_bytes = artifact_chunk_bytes_,
      .expected_artifact_size = reg.total_size_bytes,
      .byte_mapping_config = byte_mapping_config_};
  cfg.cpu_shared_memory_enabled = cpu_shared_memory_enabled_;
  if (!reg.view.has_value() && reg.tensor_index_data.has_value() && !reg.tensor_index_data->empty()) {
    cfg.canonical_index_json = *reg.tensor_index_data;
  }
  cfg.pinned_memory_timeout = pinned_memory_timeout_;
  cfg.streaming_buffer_chunks = streaming_buffer_chunks_;
  if (memory_tier_config_.has_value()) {
    cfg.memory_tier_config = memory_tier_config_;
  }

  const bool needs_gpu_staging = !stable_dram || reg.stable_dram.stage_on_gpu;
  if (needs_gpu_staging) {
    if (auto free_or = device_manager_->get_free_memory(reg.device_id); free_or.ok()) {
      size_t free_bytes = free_or.value();
      if (reg.total_size_bytes > free_bytes) {
        auto evict_status = components::evict_for_gpu(
            *replica_registry_,
            *device_manager_,
            *metrics_collector_,
            reg.device_id,
            reg.total_size_bytes - free_bytes);
        if (!evict_status.ok()) {
          return absl::ResourceExhaustedError(
              absl::StrCat(
                  "Insufficient GPU memory available. Requested: ",
                  reg.total_size_bytes,
                  " bytes, Free: ",
                  free_bytes,
                  ". ",
                  evict_status.message()));
        }
      }
    }
  }

  auto replica_or = replica_factory_(cfg);
  if (!replica_or.ok()) {
    return replica_or.status();
  }
  auto replica = std::move(replica_or.value());

  if (memory_tier_budget_) {
    replica->get_memory_manager().set_memory_tier_budget(memory_tier_budget_);
  }

  void* base_ptr = nullptr;
  std::byte* stable_dram_cpu_base = nullptr;
  cudaIpcMemHandle_t ipc_handle{};
  std::unique_ptr<common::memory::GpuDeviceMemory> staging_gpu;
  if (stable_dram) {
    absl::Status alloc_status = replica->get_memory_manager().allocate_memory(common::memory::MemoryLocation::CPU);
    if (!alloc_status.ok()) {
      return alloc_status;
    }
    const auto cpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::CPU);
    if (cpu_ptrs.empty() || cpu_ptrs[0] == nullptr) {
      return absl::FailedPreconditionError("CPU pointer unavailable after stable_dram allocation");
    }
    stable_dram_cpu_base = static_cast<std::byte*>(cpu_ptrs[0]);
    if (reg.stable_dram.stage_on_gpu) {
      staging_gpu = std::make_unique<common::memory::GpuDeviceMemory>();
      absl::Status staging_status = staging_gpu->allocate(reg.total_size_bytes, reg.device_id);
      if (!staging_status.ok()) {
        return staging_status;
      }
      base_ptr = staging_gpu->get();
      if (base_ptr == nullptr) {
        return absl::InternalError("Stable DRAM staging pointer is null");
      }
      auto ipc_status = cuda::get_ipc_mem_handle(&ipc_handle, base_ptr);
      if (!ipc_status.ok()) {
        return ipc_status;
      }
    }
  } else {
    absl::Status alloc_status = replica->get_memory_manager().allocate_memory(common::memory::MemoryLocation::GPU);
    if (!alloc_status.ok()) {
      return alloc_status;
    }

    auto ipc_or = replica->get_memory_manager().get_ipc_handle();
    if (!ipc_or.ok()) {
      return ipc_or.status();
    }
    ipc_handle = *ipc_or;
    const auto gpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
    base_ptr = (!gpu_ptrs.empty() ? gpu_ptrs[0] : nullptr);
  }

  DeviceKey dev_key = stable_dram ? DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""}
                                  : DeviceRegistry::instance().gpu_key(reg.device_id);
  loading::ReplicaKey inst_key{.artifact_id = reg.artifact_id, .device = dev_key, .replica = 0};
  {
    absl::Status emplace_status =
        replica_registry_->emplace(inst_key, gsl::not_null<std::shared_ptr<replica::Replica>>{replica});
    if (absl::IsAlreadyExists(emplace_status)) {
      VLOG(1) << "Pending registry already had instance for key=" << inst_key.artifact_id;
    } else if (!emplace_status.ok()) {
      return emplace_status;
    }
  }

  auto entry = std::make_shared<PendingRegistrationContext>();
  entry->registration_id = make_registration_id();
  entry->artifact_id = reg.artifact_id;
  if (reg.client_artifact_id.has_value()) {
    entry->client_artifact_id = *reg.client_artifact_id;
    entry->id_kind = common::ArtifactIdKind::kCgid;
  } else {
    entry->client_artifact_id.clear();
    entry->id_kind = common::ArtifactIdKind::kMi2;
  }
  entry->artifact_id_override = reg.artifact_id_override;
  entry->device_id = reg.device_id;
  entry->size_bytes = reg.total_size_bytes;
  entry->tensor_index_key = reg.tensor_index_key;
  entry->tensor_index_data = reg.tensor_index_data;
  entry->schema_version = reg.schema_version;
  entry->encoding = reg.encoding;
  entry->enable_p2p = reg.enable_p2p;
  entry->pending_registry_key = inst_key;
  if (reg.ttl_ms > 0) {
    entry->expiry_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(reg.ttl_ms);
  }
  entry->replica = replica;
  entry->gpu_ptr = base_ptr;
  entry->stable_dram_cpu_base = stable_dram_cpu_base;
  entry->ipc_handle = ipc_handle;
  entry->staging_gpu = std::move(staging_gpu);
  entry->stable_dram = reg.stable_dram;
  entry->stream_ingested_bytes = 0;
  entry->stream_copied_bytes = 0;
  entry->stream_chunk_count = 0;
  entry->stream_inflight_chunks = 0;
  entry->stream_memcpy_nanos = 0;
  entry->stable_cpu_ingest_mode = PendingRegistrationContext::StableCpuIngestMode::kUnset;
  entry->stable_cache_policy = reg.stable_cache_policy;
  entry->plan =
      stable_dram ? PendingRegistrationContext::Plan::kStableDram : PendingRegistrationContext::Plan::kCoalesced;
  entry->begin_time = std::chrono::steady_clock::now();

  if (view_plan.has_value()) {
    entry->view_state = std::make_unique<PendingRegistrationContext::ViewState>();
    entry->view_state->options = *reg.view;
    entry->view_state->options.canonical_size_bytes = canonical_size;
    entry->view_state->options.canonical_ranges = canonical_ranges;
    entry->view_state->plan = *view_plan;
    entry->view_state->expected_view_bytes = expected_view_bytes;
    entry->view_state->view_size_bytes = view_size_bytes;
    entry->view_state->ingested_bytes = 0;
    if (entry->view_state->options.placement == ViewPlacement::kServer) {
      const auto target = (entry->view_state->options.registration_kind == ViewRegistrationKind::kPiece)
          ? loader::ViewIngestExecutor::IngestTarget::kView
          : loader::ViewIngestExecutor::IngestTarget::kCanonical;
      entry->view_state->executor = std::make_unique<loader::ViewIngestExecutor>(
          entry->view_state->plan.write, entry->view_state->plan.inverse_transform, target);
    }
  }

  size_t pending_size_after = 0;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_regs_.emplace(entry->registration_id, entry);
    pending_size_after = pending_regs_.size();
  }
  record_pending_gauge(pending_size_after);

  RegistrationBeginResult out;
  out.registration_id = entry->registration_id;
  out.device_id = reg.device_id;
  out.size_bytes = reg.total_size_bytes;
  out.cuda_ipc_handle_bytes = cuda::IpcHandleBytes::from_native(ipc_handle);
  return out;
}

absl::StatusOr<RegistrationCommitResult> RegistrationBackend::commit(std::string_view registration_id) {
  std::string request_id(registration_id);
  if (request_id.empty()) {
    request_id = "registration";
  }
  std::shared_ptr<PendingRegistrationContext> entry;
  std::shared_ptr<PendingRegistrationContext> expired_entry;
  size_t pending_size_after_expire = 0;
  std::string trace_artifact_id = request_id;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_regs_.find(std::string(registration_id));
    if (it == pending_regs_.end()) {
      return absl::NotFoundError("registration_id not found");
    }
    if (!it->second->artifact_id.empty()) {
      trace_artifact_id = it->second->artifact_id;
    }
    const bool has_ttl = it->second->expiry_time.time_since_epoch().count() > 0;
    if (has_ttl && std::chrono::steady_clock::now() > it->second->expiry_time) {
      expired_entry = it->second;
      pending_regs_.erase(it);
      pending_size_after_expire = pending_regs_.size();
    } else {
      entry = it->second;
    }
  }

  SC_TRACE_INIT_GUARD(request_id, trace_artifact_id, "registration_commit");

  if (expired_entry) {
    record_pending_gauge(pending_size_after_expire);
    record_commit_latency(*expired_entry, "expired");
    erase_pending_registry_alias(*expired_entry);
    const auto location = expired_entry->plan == PendingRegistrationContext::Plan::kStableDram
        ? common::memory::MemoryLocation::CPU
        : common::memory::MemoryLocation::GPU;
    release_replica_memory(expired_entry->replica, location);
    return absl::DeadlineExceededError("registration expired (TTL)");
  }
  if (!entry) {
    return absl::InternalError("pending registration entry missing after TTL check");
  }

  if (entry->view_state && entry->view_state->options.placement == ViewPlacement::kServer) {
    auto& view_state = *entry->view_state;
    if (!view_state.executor) {
      return absl::FailedPreconditionError("view executor missing for server placement registration");
    }
    if (!view_state.executor->is_complete()) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "view bytes incomplete; expected ",
              view_state.expected_view_bytes,
              " received ",
              view_state.executor->ingested_bytes()));
    }
    auto finalize_status =
        view_state.executor->finalize(common::memory::MemoryLocation::GPU, entry->gpu_ptr, entry->device_id);
    if (!finalize_status.ok()) {
      return finalize_status;
    }
    view_state.finalized = true;
  }

  if (entry->view_state && entry->view_state->options.registration_kind == ViewRegistrationKind::kPiece) {
    const auto location =
        (entry->plan == PendingRegistrationContext::Plan::kStableDram && !entry->stable_dram.stage_on_gpu)
        ? common::memory::MemoryLocation::CPU
        : common::memory::MemoryLocation::GPU;
    void* base_ptr = nullptr;
    if (location == common::memory::MemoryLocation::CPU) {
      const auto cpu_ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::CPU);
      base_ptr = (!cpu_ptrs.empty() ? cpu_ptrs[0] : nullptr);
    } else {
      base_ptr = entry->gpu_ptr;
      if (base_ptr == nullptr) {
        const auto ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
        base_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
      }
    }
    if (base_ptr == nullptr) {
      return absl::FailedPreconditionError("view buffer base pointer unavailable for padding");
    }
    auto pad_status = zero_view_padding(
        entry->view_state->plan.write, entry->view_state->view_size_bytes, location, base_ptr, entry->device_id);
    if (!pad_status.ok()) {
      return pad_status;
    }
  }

  if (entry->plan == PendingRegistrationContext::Plan::kStableDram) {
    if (entry->stable_dram.stage_on_gpu) {
      if (entry->gpu_ptr == nullptr) {
        return absl::FailedPreconditionError("Stable DRAM staging pointer is null");
      }
      const auto cpu_ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::CPU);
      if (cpu_ptrs.empty() || cpu_ptrs[0] == nullptr) {
        return absl::FailedPreconditionError("CPU pointer unavailable for stable DRAM commit");
      }
      auto dev_status = cuda::set_device(entry->device_id);
      if (!dev_status.ok()) {
        return dev_status;
      }
      auto copy_status =
          cuda::memcpy(cpu_ptrs[0], entry->gpu_ptr, static_cast<size_t>(entry->size_bytes), cudaMemcpyDeviceToHost);
      if (!copy_status.ok()) {
        return copy_status;
      }
      auto sync_status = cuda::device_synchronize();
      if (!sync_status.ok()) {
        return sync_status;
      }
      LOG(INFO) << "Stable DRAM commit path=staging_gpu bytes=" << entry->size_bytes
                << " device_id=" << entry->device_id;
    } else {
      uint64_t stream_ingested_bytes = 0;
      uint64_t stream_copied_bytes = 0;
      uint64_t stream_chunk_count = 0;
      uint64_t stream_inflight_chunks = 0;
      uint64_t stream_memcpy_nanos = 0;
      size_t stream_range_count = 0;
      uint64_t stream_prefix_bytes = 0;
      bool stream_cover_full_span = false;
      PendingRegistrationContext::StableCpuIngestMode stream_mode =
          PendingRegistrationContext::StableCpuIngestMode::kUnset;
      {
        std::lock_guard<std::mutex> lock(entry->stream_ingest_mu);
        stream_ingested_bytes = entry->stream_ingested_bytes;
        stream_copied_bytes = entry->stream_copied_bytes;
        stream_chunk_count = entry->stream_chunk_count;
        stream_inflight_chunks = entry->stream_inflight_chunks;
        stream_memcpy_nanos = entry->stream_memcpy_nanos;
        stream_range_count = entry->stream_ingested_ranges.size();
        stream_prefix_bytes = ingested_prefix_bytes(entry->stream_ingested_ranges);
        stream_cover_full_span = ingested_ranges_cover_full_span(entry->stream_ingested_ranges, entry->size_bytes);
        stream_mode = entry->stable_cpu_ingest_mode;
      }
      if (stream_mode == PendingRegistrationContext::StableCpuIngestMode::kUnset) {
        return absl::FailedPreconditionError(
            absl::StrCat(
                "stable_dram ingestion missing: expected=",
                entry->size_bytes,
                " acked=",
                stream_ingested_bytes,
                " copied=",
                stream_copied_bytes,
                " inflight=",
                stream_inflight_chunks,
                " prefix=",
                stream_prefix_bytes,
                " ranges=",
                stream_range_count));
      }
      if (stream_mode == PendingRegistrationContext::StableCpuIngestMode::kStreamMemcpy) {
        if (stream_ingested_bytes != entry->size_bytes || stream_copied_bytes != entry->size_bytes ||
            stream_inflight_chunks != 0 || !stream_cover_full_span) {
          return absl::FailedPreconditionError(
              absl::StrCat(
                  "stable_dram stream ingestion incomplete: expected=",
                  entry->size_bytes,
                  " got=",
                  stream_ingested_bytes,
                  " copied=",
                  stream_copied_bytes,
                  " inflight=",
                  stream_inflight_chunks,
                  " prefix=",
                  stream_prefix_bytes,
                  " ranges=",
                  stream_range_count));
        }
        const double memcpy_seconds = static_cast<double>(stream_memcpy_nanos) / 1e9;
        const double memcpy_gibps = memcpy_seconds > 0.0
            ? (static_cast<double>(entry->size_bytes) / static_cast<double>(1ULL << 30)) / memcpy_seconds
            : 0.0;
        LOG(INFO) << "Stable DRAM commit path=cpu_stream bytes=" << entry->size_bytes
                  << " chunks=" << stream_chunk_count << " memcpy_s=" << memcpy_seconds
                  << " memcpy_gibps=" << memcpy_gibps;
      } else {
        if (stream_ingested_bytes != entry->size_bytes || stream_inflight_chunks != 0 || !stream_cover_full_span) {
          return absl::FailedPreconditionError(
              absl::StrCat(
                  "stable_dram cpu_memfd publish incomplete: expected=",
                  entry->size_bytes,
                  " acked=",
                  stream_ingested_bytes,
                  " inflight=",
                  stream_inflight_chunks,
                  " prefix=",
                  stream_prefix_bytes,
                  " ranges=",
                  stream_range_count));
        }
        LOG(INFO) << "Stable DRAM commit path=cpu_memfd_publish bytes=" << entry->size_bytes
                  << " ranges=" << stream_chunk_count;
      }
    }
  }

  std::string index_multihash;
  std::string data_multihash;
  std::optional<loader::ByteRangeMap> canonical_map;
  auto ensure_canonical_map = [&]() {
    if (canonical_map.has_value()) {
      return;
    }
    if (entry->tensor_index_data.has_value() && !entry->tensor_index_data->empty() && entry->encoding == "json") {
      auto map_or =
          loader::build_byte_range_map_from_canonical_index_json(*entry->tensor_index_data, entry->size_bytes);
      if (map_or.ok()) {
        canonical_map = std::move(*map_or);
      } else {
        LOG(WARNING) << "Failed to rebuild canonical byte map: " << map_or.status();
      }
    }
  };

  const bool has_artifact_id_override =
      entry->artifact_id_override.has_value() && !entry->artifact_id_override->empty();
  if (has_artifact_id_override) {
    if (!entry->client_artifact_id.empty()) {
      return absl::InvalidArgumentError("artifact_id_override cannot be set when client_artifact_id is present");
    }
    auto parsed_or = parse_mi2_multihashes(*entry->artifact_id_override);
    if (!parsed_or.ok()) {
      return parsed_or.status();
    }
    index_multihash = parsed_or->first;
    data_multihash = parsed_or->second;
    entry->artifact_id = *entry->artifact_id_override;
    entry->id_kind = common::ArtifactIdKind::kMi2;
  } else {
    absl::StatusOr<std::string> index_mh_or =
        common::compute_index_multihash(entry->tensor_index_data, entry->tensor_index_key);
    if (!index_mh_or.ok()) {
      return index_mh_or.status();
    }
    index_multihash = *index_mh_or;

    if (!entry->schema_version.empty() && entry->schema_version != "v3") {
      return absl::FailedPreconditionError(
          absl::StrCat("Pending registration schema_version must be 'v3'; found '", entry->schema_version, "'"));
    }

    if (entry->id_kind == common::ArtifactIdKind::kMi2 || entry->client_artifact_id.empty()) {
      absl::StatusOr<std::string> data_mh_or;
      if (entry->plan == PendingRegistrationContext::Plan::kStableDram) {
        if (!entry->replica) {
          return absl::FailedPreconditionError("CPU registration missing backing replica");
        }
        const auto cpu_ptrs = entry->replica->get_data_pointer(common::memory::MemoryLocation::CPU);
        if (cpu_ptrs.empty() || cpu_ptrs[0] == nullptr) {
          return absl::FailedPreconditionError("CPU data pointer unavailable for hashing");
        }
        gsl::not_null<const void*> cpu_base{static_cast<const void*>(cpu_ptrs[0])};
        auto mh_or = loader::compute_data_multihash_from_cpu_memory(cpu_base, entry->size_bytes);
        if (!mh_or.ok()) {
          return mh_or.status();
        }
        data_mh_or = *mh_or;
      } else {
        void* gpu_ptr = entry->gpu_ptr;
        if (gpu_ptr == nullptr) {
          const auto ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
          gpu_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
        }
        if (!gpu_ptr) {
          return absl::FailedPreconditionError("GPU pointer is null; cannot hash GPU data");
        }
        ensure_canonical_map();
        if (canonical_map.has_value()) {
          auto base_source = std::make_shared<loader::GpuMemorySource>(
              gsl::not_null<void*>{gpu_ptr}, entry->device_id, entry->size_bytes);
          loader::ByteRangeCompiler compiler(byte_mapping_config_, "registration_canonical");
          auto program_or = compiler.Compile(*canonical_map);
          if (!program_or.ok()) {
            return program_or.status();
          }
          std::vector<std::shared_ptr<loader::SeekableSource>> sources;
          sources.emplace_back(std::move(base_source));
          loader::ByteRangeMappedSource::Options map_opts{
              .path = "registration_canonical",
              .enable_direct_write_at = byte_mapping_config_.enable_direct_write_at,
          };
          auto mapped_or = loader::ByteRangeMappedSource::Create(
              *canonical_map, *program_or, std::move(sources), std::move(map_opts));
          if (!mapped_or.ok()) {
            return mapped_or.status();
          }
          auto mh_or =
              loader::compute_data_multihash_from_seekable_source(*mapped_or.value(), canonical_map->total_bytes);
          if (!mh_or.ok()) {
            return mh_or.status();
          }
          data_mh_or = *mh_or;
        } else {
          auto mh_or = common::compute_data_multihash_from_gpu(gpu_ptr, entry->size_bytes, entry->device_id);
          if (!mh_or.ok()) {
            return mh_or.status();
          }
          data_mh_or = *mh_or;
        }
      }
      if (!data_mh_or.ok()) {
        return data_mh_or.status();
      }
      data_multihash = *data_mh_or;
      entry->artifact_id = absl::StrCat("mi2:", index_multihash, ":", data_multihash);
      entry->id_kind = common::ArtifactIdKind::kMi2;
    } else {
      entry->artifact_id = entry->client_artifact_id;
      entry->id_kind = common::ArtifactIdKind::kCgid;
      data_multihash.clear();
    }
  }

  DeviceKey dev_key = entry->plan == PendingRegistrationContext::Plan::kStableDram
      ? DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""}
      : DeviceRegistry::instance().gpu_key(entry->device_id);
  std::optional<std::string> view_id;
  if (entry->view_state && !entry->view_state->options.view_id.empty()) {
    view_id = entry->view_state->options.view_id;
  }
  loading::ReplicaKey mi2_key{.artifact_id = entry->artifact_id, .view_id = view_id, .device = dev_key, .replica = 0};
  const bool allow_idempotent = entry->view_state == nullptr && entry->id_kind == common::ArtifactIdKind::kMi2;
  bool reuse_existing = false;
  bool stable_cache_admitted = false;
  bool key_already_exists = false;
  if (allow_idempotent) {
    if (auto existing_or = replica_registry_->find(mi2_key); existing_or.ok()) {
      reuse_existing = true;
    }
  }
  if (!reuse_existing) {
    absl::Status emplace_status =
        replica_registry_->emplace(mi2_key, gsl::not_null<std::shared_ptr<replica::Replica>>{entry->replica});
    if (absl::IsAlreadyExists(emplace_status)) {
      key_already_exists = true;
      if (allow_idempotent) {
        reuse_existing = true;
      } else {
        VLOG(1) << "mi2 mapping already present for artifact_id=" << entry->artifact_id;
      }
    } else if (!emplace_status.ok()) {
      return emplace_status;
    }
  }

  if (reuse_existing) {
    erase_pending_registry_alias(*entry, /*keep_key=*/mi2_key);
    const auto location = entry->plan == PendingRegistrationContext::Plan::kStableDram
        ? common::memory::MemoryLocation::CPU
        : common::memory::MemoryLocation::GPU;
    release_replica_memory(entry->replica, location);
    size_t pending_size_after = 0;
    erase_pending(registration_id, &pending_size_after);
    record_pending_gauge(pending_size_after);

    RegistrationCommitResult result;
    result.registration_id = std::string(registration_id);
    result.artifact_id = entry->artifact_id;
    result.device_id = entry->device_id;
    result.device = dev_key;
    result.size_bytes = entry->size_bytes;
    result.existed = true;
    result.stable_cache_admitted = false;
    result.index_multihash = index_multihash;
    result.data_multihash = data_multihash;
    result.schema_version = entry->schema_version;
    result.encoding = entry->encoding;
    result.id_kind = entry->id_kind;
    record_commit_latency(*entry, "existed");
    return result;
  }

  // Piece/view retries may hit an existing logical key. The newly ingested
  // replica is transient in that case, so avoid publishing fresh transport
  // keys that would become stale once this commit scope ends.
  const bool skip_transport_publication = key_already_exists && entry->view_state != nullptr;

  if (entry->plan == PendingRegistrationContext::Plan::kStableDram && entry->stable_cache_policy.has_value() &&
      stable_cache_manager_) {
    components::StableDramCacheManager::AdmissionRequest admit_request;
    admit_request.key = mi2_key;
    admit_request.replica = entry->replica;
    admit_request.size_bytes = entry->size_bytes;
    admit_request.policy = *entry->stable_cache_policy;
    auto admit_or = stable_cache_manager_->admit(admit_request);
    if (!admit_or.ok()) {
      release_replica_memory(entry->replica, common::memory::MemoryLocation::CPU);
      return admit_or.status();
    }
    stable_cache_admitted = admit_or->admitted && !admit_or->skipped;
  }

  std::vector<std::string> remote_keys;
  std::vector<uint64_t> buffer_sizes;
  std::optional<ExportRegistration> export_registration;
  if (!skip_transport_publication && entry->enable_p2p && communication_manager_ &&
      communication_manager_->is_enabled()) {
    const auto location = entry->plan == PendingRegistrationContext::Plan::kStableDram
        ? common::memory::MemoryLocation::CPU
        : common::memory::MemoryLocation::GPU;
    auto reg_info_or = entry->replica->enable_remote_memory_access(location, communication_manager_->get_engine());
    if (!reg_info_or.ok()) {
      return reg_info_or.status();
    }
    export_registration = *reg_info_or;
    remote_keys = reg_info_or->remote_memory_keys;
    buffer_sizes.reserve(reg_info_or->buffer_sizes.size());
    for (const auto& sz : reg_info_or->buffer_sizes) {
      buffer_sizes.push_back(static_cast<uint64_t>(sz));
    }
  }

  DeviceKey device = entry->plan == PendingRegistrationContext::Plan::kStableDram
      ? DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""}
      : DeviceRegistry::instance().gpu_key(entry->device_id);
  std::optional<std::string> verification_json;
  if (!remote_keys.empty()) {
    const auto location = entry->plan == PendingRegistrationContext::Plan::kStableDram
        ? common::memory::MemoryLocation::CPU
        : common::memory::MemoryLocation::GPU;
    std::vector<void*> data_ptrs = entry->replica->get_memory_manager().get_pointer(location);
    if (!data_ptrs.empty() && data_ptrs[0] != nullptr) {
      std::vector<size_t> data_sizes{static_cast<size_t>(entry->size_bytes)};
      const int verify_device = (location == common::memory::MemoryLocation::GPU) ? entry->device_id : -1;
      auto info_or = common::ArtifactVerifier::generate_verification_info(
          data_ptrs, data_sizes, verify_device, common::VerificationLevel::KEY_POINTS);
      if (info_or.ok()) {
        verification_json = info_or->to_json();
      }
    }
  }

  if (promotion_manager_ && export_registration.has_value()) {
    auto promotion_status =
        promotion_manager_->record_export_registration(mi2_key, *export_registration, verification_json);
    if (!promotion_status.ok()) {
      LOG(WARNING) << "record_export_registration failed for " << mi2_key << ": " << promotion_status;
    }
  }

  std::optional<std::string> view_data_hash;
  std::optional<std::string> view_spec_json;
  std::vector<global_store::v1::LeafWrite> leaf_writes;
  std::vector<global_store::v1::PieceProofDigestWrite> proof_digests;
  std::vector<uint64_t> canonical_leaf_indices;
  std::optional<size_t> leaf_chunk_bytes;
  if (entry->view_state) {
    view_spec_json = view::build_view_spec_json(entry->view_state->options.spec);
    leaf_chunk_bytes = artifact_chunk_bytes_ == 0 ? static_cast<size_t>(4ULL * 1024 * 1024) : artifact_chunk_bytes_;

    if (entry->view_state->options.registration_kind == ViewRegistrationKind::kPiece) {
      if (!leaf_chunk_bytes.has_value() || entry->view_state->view_size_bytes == 0) {
        return absl::FailedPreconditionError("piece registration requires view_size_bytes for hashing");
      }
      const auto location =
          (entry->plan == PendingRegistrationContext::Plan::kStableDram && !entry->stable_dram.stage_on_gpu)
          ? common::memory::MemoryLocation::CPU
          : common::memory::MemoryLocation::GPU;
      void* base_ptr = nullptr;
      if (location == common::memory::MemoryLocation::CPU) {
        const auto cpu_ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::CPU);
        base_ptr = (!cpu_ptrs.empty() ? cpu_ptrs[0] : nullptr);
      } else {
        base_ptr = entry->gpu_ptr;
        if (base_ptr == nullptr) {
          const auto ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
          base_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
        }
      }
      if (base_ptr == nullptr) {
        return absl::FailedPreconditionError("view buffer base pointer unavailable for hashing");
      }
      absl::StatusOr<loader::verification::ViewHashResult> view_hash_or;
      if (location == common::memory::MemoryLocation::CPU) {
        loader::CpuMemorySource src(gsl::not_null<const void*>{base_ptr}, entry->view_state->view_size_bytes);
        view_hash_or = loader::verification::compute_view_tree_hash_and_leaves(
            src, entry->view_state->view_size_bytes, *leaf_chunk_bytes);
      } else {
        loader::GpuMemorySource view_source(
            gsl::not_null<void*>{base_ptr}, entry->device_id, entry->view_state->view_size_bytes);
        view_hash_or = loader::verification::compute_view_tree_hash_and_leaves(
            view_source, entry->view_state->view_size_bytes, *leaf_chunk_bytes);
      }
      if (!view_hash_or.ok()) {
        return view_hash_or.status();
      }
      view_data_hash = view_hash_or->multihash;
      const auto& digests = view_hash_or->leaf_digests;
      leaf_writes.reserve(leaf_writes.size() + digests.size());
      for (size_t idx = 0; idx < digests.size(); ++idx) {
        global_store::v1::LeafWrite leaf;
        auto* hash_space = leaf.mutable_hash_space();
        hash_space->mutable_byte_space()->set_kind(common::v1::BYTE_SPACE_KIND_VIEW);
        hash_space->mutable_byte_space()->set_id(entry->view_state->options.view_id);
        leaf.set_leaf_idx(static_cast<uint64_t>(idx));
        const auto& digest = digests[idx];
        leaf.set_digest(digest.data(), static_cast<int>(digest.size()));
        leaf_writes.push_back(std::move(leaf));
      }

      if (entry->tensor_index_data.has_value() && !entry->tensor_index_data->empty()) {
        auto intervals_or = parse_tensor_intervals(*entry->tensor_index_data);
        if (!intervals_or.ok()) {
          return intervals_or.status();
        }
        std::unordered_map<std::string, uint64_t> transpose_view_offsets;
        std::unordered_map<std::string, loader::TensorTransformPlan> transpose_inverse_plans;
        if (entry->view_state->plan.forward.transform.requires_materialization) {
          for (const auto& tensor_plan : entry->view_state->plan.forward.transform.tensors) {
            transpose_view_offsets.emplace(tensor_plan.tensor_name, tensor_plan.dst_offset);
          }
          for (const auto& tensor_plan : entry->view_state->plan.inverse_transform.tensors) {
            transpose_inverse_plans.emplace(tensor_plan.tensor_name, tensor_plan);
          }
        }
        for (const auto& interval : *intervals_or) {
          if (interval.size_bytes == 0) {
            continue;
          }
          if (!ranges_cover_interval(
                  entry->view_state->options.canonical_ranges, interval.offset, interval.size_bytes)) {
            continue;
          }
          const auto inverse_it = transpose_inverse_plans.find(interval.tensor_name);
          if (inverse_it != transpose_inverse_plans.end()) {
            const auto offset_it = transpose_view_offsets.find(interval.tensor_name);
            if (offset_it == transpose_view_offsets.end()) {
              return absl::FailedPreconditionError("missing transpose tensor dst_offset for proof digests");
            }
            const uint64_t view_offset = offset_it->second;
            if (view_offset + interval.size_bytes > entry->view_state->view_size_bytes) {
              return absl::OutOfRangeError("transpose tensor view range exceeds view buffer size");
            }
            if (interval.size_bytes > std::numeric_limits<size_t>::max()) {
              return absl::OutOfRangeError("transpose tensor exceeds host memory limits");
            }
            std::vector<uint8_t> view_bytes(static_cast<size_t>(interval.size_bytes));
            auto read_status =
                read_view_bytes(location, base_ptr, entry->device_id, view_offset, absl::MakeSpan(view_bytes));
            if (!read_status.ok()) {
              return read_status;
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
                std::move(write_plan),
                std::move(inverse_transform),
                loader::ViewIngestExecutor::IngestTarget::kCanonical);
            absl::Status ingest_status = executor.ingest_chunk(
                /*view_offset=*/0,
                absl::Span<const std::byte>(reinterpret_cast<const std::byte*>(view_bytes.data()), view_bytes.size()),
                common::memory::MemoryLocation::CPU,
                canonical_bytes.data(),
                /*device_id=*/-1);
            if (!ingest_status.ok()) {
              return ingest_status;
            }
            absl::Status finalize_status =
                executor.finalize(common::memory::MemoryLocation::CPU, canonical_bytes.data(), /*device_id=*/-1);
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
              std::vector<uint8_t> digest = common::sha256_digest_bytes(
                  absl::Span<const uint8_t>(canonical_bytes.data() + local_start, chunk_bytes));
              if (digest.size() != 32) {
                return absl::InternalError("sha256 digest size mismatch");
              }
              global_store::v1::PieceProofDigestWrite proof;
              proof.set_view_id(entry->view_state->options.view_id);
              proof.set_tensor_name(interval.tensor_name);
              proof.set_proof_schema_version(std::string(kProofSchemaV1));
              proof.set_proof_chunk_idx(proof_chunk_idx);
              proof.set_digest(digest.data(), static_cast<int>(digest.size()));
              proof_digests.push_back(std::move(proof));
            }
            continue;
          }

          auto spans_or = canonical_spans_for_tensor(
              entry->view_state->plan.write, interval.offset, interval.size_bytes, entry->view_state->view_size_bytes);
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
              auto st = read_view_bytes(location, base_ptr, entry->device_id, src_view_offset, dst);
              if (!st.ok()) {
                return st;
              }
              cursor = take_end;
            }
            std::vector<uint8_t> digest =
                common::sha256_digest_bytes(absl::Span<const uint8_t>(buffer.data(), buffer.size()));
            if (digest.size() != 32) {
              return absl::InternalError("sha256 digest size mismatch");
            }
            global_store::v1::PieceProofDigestWrite proof;
            proof.set_view_id(entry->view_state->options.view_id);
            proof.set_tensor_name(interval.tensor_name);
            proof.set_proof_schema_version(std::string(kProofSchemaV1));
            proof.set_proof_chunk_idx(proof_chunk_idx);
            proof.set_digest(digest.data(), static_cast<int>(digest.size()));
            proof_digests.push_back(std::move(proof));
          }
        }
      }
    } else {
      ensure_canonical_map();
      const uint64_t view_total = entry->view_state->plan.forward.selection.map.total_bytes;
      if (canonical_map.has_value() && view_total > 0 && leaf_chunk_bytes.has_value()) {
        void* gpu_ptr = entry->gpu_ptr;
        if (gpu_ptr == nullptr) {
          const auto ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
          gpu_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
        }
        if (gpu_ptr != nullptr) {
          auto base_source = std::make_shared<loader::GpuMemorySource>(
              gsl::not_null<void*>{gpu_ptr}, entry->device_id, entry->size_bytes);
          loader::ByteRangeCompiler compiler(byte_mapping_config_, "registration_view");
          auto program_or = compiler.Compile(*canonical_map);
          if (!program_or.ok()) {
            LOG(WARNING) << "Failed to compile canonical byte map for view hash: " << program_or.status();
          } else {
            std::vector<std::shared_ptr<loader::SeekableSource>> sources;
            sources.emplace_back(std::move(base_source));
            loader::ByteRangeMappedSource::Options map_opts{
                .path = "registration_view",
                .enable_direct_write_at = byte_mapping_config_.enable_direct_write_at,
            };
            auto mapped_or = loader::ByteRangeMappedSource::Create(
                *canonical_map, *program_or, std::move(sources), std::move(map_opts));
            if (!mapped_or.ok()) {
              LOG(WARNING) << "Failed to build canonical mapped source for view hash: " << mapped_or.status();
            } else {
              auto view_source = loader::make_view_plan_source(
                  std::move(*mapped_or), entry->view_state->plan.forward.selection, byte_mapping_config_);
              if (!view_source) {
                LOG(WARNING) << "Failed to build view plan source for view hash";
              } else {
                auto view_hash_or = loader::verification::compute_view_tree_hash_and_leaves(
                    *view_source, view_total, *leaf_chunk_bytes);
                if (view_hash_or.ok()) {
                  view_data_hash = view_hash_or->multihash;
                  const auto& digests = view_hash_or->leaf_digests;
                  leaf_writes.reserve(leaf_writes.size() + digests.size());
                  for (size_t idx = 0; idx < digests.size(); ++idx) {
                    global_store::v1::LeafWrite leaf;
                    auto* hash_space = leaf.mutable_hash_space();
                    hash_space->mutable_byte_space()->set_kind(common::v1::BYTE_SPACE_KIND_VIEW);
                    hash_space->mutable_byte_space()->set_id(entry->view_state->options.view_id);
                    leaf.set_leaf_idx(static_cast<uint64_t>(idx));
                    const auto& digest = digests[idx];
                    leaf.set_digest(digest.data(), static_cast<int>(digest.size()));
                    leaf_writes.push_back(std::move(leaf));
                  }
                } else {
                  LOG(WARNING) << "ComputeTreeHashAndLeaves (view) failed: " << view_hash_or.status();
                }
              }
            }
          }
        }
      }
      canonical_leaf_indices = view::compute_fully_covered_canonical_leaf_indices(
          entry->view_state->options.canonical_ranges, leaf_chunk_bytes.value_or(0));
    }
  }

  if (entry->plan == PendingRegistrationContext::Plan::kStableDram && entry->stable_dram.stage_on_gpu &&
      entry->stable_dram.release_gpu_on_commit && entry->staging_gpu) {
    entry->staging_gpu.reset();
    entry->gpu_ptr = nullptr;
  }

  if (entry->id_kind == common::ArtifactIdKind::kMi2 && entry->view_state && leaf_chunk_bytes.has_value() &&
      !canonical_leaf_indices.empty() && !index_multihash.empty()) {
    loader::verification::MemoryView canonical_view;
    canonical_view.size_bytes = entry->size_bytes;
    if (entry->plan == PendingRegistrationContext::Plan::kStableDram) {
      canonical_view.location = common::memory::MemoryLocation::CPU;
      const auto cpu_ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::CPU);
      if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr) {
        canonical_view.base_ptr = const_cast<void*>(cpu_ptrs[0]);
      }
    } else {
      canonical_view.location = common::memory::MemoryLocation::GPU;
      canonical_view.gpu_device_id = entry->device_id;
      void* gpu_ptr = entry->gpu_ptr;
      if (gpu_ptr == nullptr) {
        const auto ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
        gpu_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
      }
      canonical_view.base_ptr = gpu_ptr;
    }

    auto canonical_digest_or = loader::verification::compute_canonical_leaf_digests(
        canonical_view, absl::Span<const uint64_t>(canonical_leaf_indices), *leaf_chunk_bytes);
    if (!canonical_digest_or.ok()) {
      LOG(WARNING) << "Failed to compute canonical leaf digests for view registration: "
                   << canonical_digest_or.status();
    } else if (canonical_digest_or->size() != canonical_leaf_indices.size()) {
      LOG(WARNING) << "Canonical leaf digest count mismatch: expected " << canonical_leaf_indices.size() << " got "
                   << canonical_digest_or->size();
    } else {
      leaf_writes.reserve(leaf_writes.size() + canonical_leaf_indices.size());
      for (size_t i = 0; i < canonical_leaf_indices.size(); ++i) {
        global_store::v1::LeafWrite leaf;
        auto* hash_space = leaf.mutable_hash_space();
        hash_space->mutable_byte_space()->set_kind(common::v1::BYTE_SPACE_KIND_CANONICAL);
        hash_space->set_canonical_index_multihash(index_multihash);
        leaf.set_leaf_idx(canonical_leaf_indices[i]);
        const auto& digest = (*canonical_digest_or)[i];
        leaf.set_digest(digest.data(), static_cast<int>(digest.size()));
        leaf_writes.push_back(std::move(leaf));
      }
    }
  }

  size_t pending_size_after = 0;
  erase_pending_registry_alias(*entry, /*keep_key=*/mi2_key);
  erase_pending(registration_id, &pending_size_after);
  record_pending_gauge(pending_size_after);

  RegistrationCommitResult result;
  result.registration_id = std::string(registration_id);
  result.artifact_id = entry->artifact_id;
  result.device_id = entry->device_id;
  result.device = device;
  result.size_bytes = entry->size_bytes;
  result.existed = skip_transport_publication;
  result.stable_cache_admitted = stable_cache_admitted;
  result.index_multihash = index_multihash;
  result.data_multihash = data_multihash;
  result.schema_version = entry->schema_version;
  result.encoding = entry->encoding;
  result.id_kind = entry->id_kind;
  uint64_t covered_bytes = 0;
  if (entry->view_state) {
    if (!entry->view_state->options.view_id.empty()) {
      result.view_id = entry->view_state->options.view_id;
    }
    if (view_data_hash.has_value()) {
      result.view_data_multihash = view_data_hash;
    }
    if (!entry->view_state->plan.forward.view_index_json.empty()) {
      result.view_index_json = entry->view_state->plan.forward.view_index_json;
    }
    result.canonical_ranges = entry->view_state->options.canonical_ranges;
    result.registration_kind = entry->view_state->options.registration_kind;
    for (const auto& range : result.canonical_ranges) {
      covered_bytes += range.length;
    }
  }

  if (entry->view_state && !entry->view_state->options.view_id.empty() && publisher_) {
    components::ViewStateUpdate update;
    update.artifact_id = entry->artifact_id;
    update.view_id = entry->view_state->options.view_id;
    update.view_spec_json = view_spec_json.value_or(view::build_view_spec_json(entry->view_state->options.spec));
    update.view_size_bytes = entry->view_state->plan.forward.view_size_bytes;
    update.view_data_hash = view_data_hash;
    update.mark_verified = true;
    update.canonical_size_bytes = entry->view_state->options.canonical_size_bytes;
    update.canonical_bytes_covered = covered_bytes;
    update.canonical_ranges = to_component_ranges(entry->view_state->options.canonical_ranges);
    update.leaf_writes = std::move(leaf_writes);
    update.proof_digests = std::move(proof_digests);
    absl::Status update_status = publisher_->update_view_state(update);
    if (!update_status.ok()) {
      LOG(WARNING) << "UpdateArtifactViewState failed for artifact " << entry->artifact_id
                   << " view_id=" << entry->view_state->options.view_id << ": " << update_status;
      if (entry->view_state->options.registration_kind == ViewRegistrationKind::kPiece) {
        return update_status;
      }
    }
  }

  if (publisher_ && !skip_transport_publication) {
    RegistrationPublication publication{
        .artifact_id = entry->artifact_id,
        .device = device,
        .size_bytes = entry->size_bytes,
        .view_id = (entry->view_state && entry->view_state->options.registration_kind == ViewRegistrationKind::kPiece &&
                    !entry->view_state->options.view_id.empty())
            ? std::optional<std::string>(entry->view_state->options.view_id)
            : std::nullopt,
        .tensor_index_key = entry->tensor_index_key,
        .remote_memory_keys = remote_keys,
        .buffer_sizes = buffer_sizes,
        .tensor_index_data = entry->tensor_index_data,
        .encoding = entry->encoding,
        .schema_version = entry->schema_version,
        .verification_json = verification_json,
        .index_multihash = index_multihash,
        .data_multihash = data_multihash,
        .id_kind = entry->id_kind};
    absl::Status registration_status = publisher_->publish_registration(publication);
    if (!registration_status.ok()) {
      // GlobalStore not connected - skip publication in standalone mode.
      // Local registration remains valid; artifact is usable on this node.
      if (absl::IsFailedPrecondition(registration_status) &&
          absl::StrContains(registration_status.message(), "not connected")) {
        LOG(INFO) << "Skipping GlobalStore publication (not connected): " << registration_status.message();
      } else {
        return registration_status;
      }
    }
  }

  {
    const auto location = entry->plan == PendingRegistrationContext::Plan::kStableDram
        ? common::memory::MemoryLocation::CPU
        : common::memory::MemoryLocation::GPU;
    auto mark_status = entry->replica->mark_loaded(location);
    if (!mark_status.ok()) {
      return mark_status;
    }
    entry->replica->set_ready_signal(location, absl::OkStatus());
  }
  record_commit_latency(*entry, "ok");
  return result;
}

absl::Status RegistrationBackend::ingest_view_chunk(
    std::string_view registration_id,
    uint64_t view_offset,
    absl::Span<const std::byte> data) {
  if (data.empty()) {
    return absl::OkStatus();
  }
  auto entry = lookup_pending(registration_id);
  if (!entry) {
    return absl::NotFoundError("registration_id not found");
  }
  if (!entry->view_state) {
    return absl::FailedPreconditionError("registration is not view-enabled");
  }
  auto& view_state = *entry->view_state;
  if (view_state.options.placement != ViewPlacement::kServer) {
    return absl::FailedPreconditionError("view chunk ingestion is only valid for SERVER placement");
  }
  if (!view_state.executor) {
    return absl::FailedPreconditionError("view executor missing for SERVER placement ingestion");
  }
  void* canonical_base = entry->gpu_ptr;
  if (canonical_base == nullptr) {
    const auto ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
    canonical_base = (!ptrs.empty() ? ptrs[0] : nullptr);
    if (canonical_base == nullptr) {
      return absl::FailedPreconditionError("GPU pointer unavailable for view ingestion");
    }
    entry->gpu_ptr = canonical_base;
  }
  auto status = view_state.executor->ingest_chunk(
      view_offset, data, common::memory::MemoryLocation::GPU, canonical_base, entry->device_id);
  if (!status.ok()) {
    return status;
  }
  view_state.ingested_bytes = view_state.executor->ingested_bytes();
  return absl::OkStatus();
}

absl::Status RegistrationBackend::ingest_registration_chunk(
    std::string_view registration_id,
    uint64_t offset,
    absl::Span<const std::byte> data) {
  if (data.empty()) {
    return absl::OkStatus();
  }
  auto entry = lookup_pending(registration_id);
  if (!entry) {
    return absl::NotFoundError("registration_id not found");
  }
  if (entry->plan != PendingRegistrationContext::Plan::kStableDram || entry->stable_dram.stage_on_gpu) {
    return absl::FailedPreconditionError("registration chunk ingestion requires stable_dram stage_on_gpu=false");
  }
  if (entry->view_state) {
    return absl::FailedPreconditionError("registration chunk ingestion is unavailable for view-enabled registrations");
  }
  if (offset > entry->size_bytes || data.size() > entry->size_bytes - offset) {
    return absl::OutOfRangeError(
        absl::StrCat(
            "registration chunk out of range: offset=", offset, " bytes=", data.size(), " total=", entry->size_bytes));
  }
  if (entry->stable_dram_cpu_base == nullptr) {
    return absl::FailedPreconditionError("CPU pointer unavailable for registration chunk ingestion");
  }
  const auto chunk_size = static_cast<uint64_t>(data.size());
  {
    std::lock_guard<std::mutex> lock(entry->stream_ingest_mu);
    if (entry->stable_cpu_ingest_mode == PendingRegistrationContext::StableCpuIngestMode::kRangeAckOnly) {
      return absl::FailedPreconditionError(
          "registration chunk ingestion cannot mix with stable_dram written-range mode");
    }
    entry->stable_cpu_ingest_mode = PendingRegistrationContext::StableCpuIngestMode::kStreamMemcpy;
    auto range_status = add_non_overlapping_ingested_range(
        entry->stream_ingested_ranges, offset, chunk_size, &entry->stream_ingested_bytes);
    if (!range_status.ok()) {
      return range_status;
    }
    entry->stream_chunk_count += 1;
    entry->stream_inflight_chunks += 1;
  }

  const auto copy_start = std::chrono::steady_clock::now();
  auto* dst = entry->stable_dram_cpu_base + static_cast<size_t>(offset);
  std::memcpy(dst, data.data(), data.size());
  const auto copy_nanos = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - copy_start).count());
  {
    std::lock_guard<std::mutex> lock(entry->stream_ingest_mu);
    entry->stream_copied_bytes += chunk_size;
    entry->stream_memcpy_nanos += copy_nanos;
    ABSL_CHECK(entry->stream_inflight_chunks > 0);
    entry->stream_inflight_chunks -= 1;
  }
  return absl::OkStatus();
}

absl::Status RegistrationBackend::ingest_registration_written_range(
    std::string_view registration_id,
    uint64_t offset,
    uint64_t length) {
  if (length == 0) {
    return absl::OkStatus();
  }
  auto entry = lookup_pending(registration_id);
  if (!entry) {
    return absl::NotFoundError("registration_id not found");
  }
  if (entry->plan != PendingRegistrationContext::Plan::kStableDram || entry->stable_dram.stage_on_gpu) {
    return absl::FailedPreconditionError(
        "registration written-range ingestion requires stable_dram stage_on_gpu=false");
  }
  if (entry->view_state) {
    return absl::FailedPreconditionError(
        "registration written-range ingestion is unavailable for view-enabled registrations");
  }
  if (offset > entry->size_bytes || length > entry->size_bytes - offset) {
    return absl::OutOfRangeError(
        absl::StrCat(
            "registration written-range out of range: offset=",
            offset,
            " length=",
            length,
            " total=",
            entry->size_bytes));
  }

  std::lock_guard<std::mutex> lock(entry->stream_ingest_mu);
  if (entry->stable_cpu_ingest_mode == PendingRegistrationContext::StableCpuIngestMode::kStreamMemcpy) {
    return absl::FailedPreconditionError(
        "registration written-range ingestion cannot mix with stable_dram cpu_stream mode");
  }
  entry->stable_cpu_ingest_mode = PendingRegistrationContext::StableCpuIngestMode::kRangeAckOnly;
  auto range_status =
      add_non_overlapping_ingested_range(entry->stream_ingested_ranges, offset, length, &entry->stream_ingested_bytes);
  if (!range_status.ok()) {
    return range_status;
  }
  entry->stream_chunk_count += 1;
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> RegistrationBackend::get_view_ingested_bytes(std::string_view registration_id) const {
  auto entry = lookup_pending(registration_id);
  if (!entry) {
    return absl::NotFoundError("registration_id not found");
  }
  if (!entry->view_state) {
    return absl::FailedPreconditionError("registration has no view state");
  }
  if (entry->view_state->executor) {
    return entry->view_state->executor->ingested_bytes();
  }
  return entry->view_state->expected_view_bytes;
}

absl::StatusOr<uint64_t> RegistrationBackend::get_registration_gpu_ptr(std::string_view registration_id) const {
  auto entry = lookup_pending(registration_id);
  if (!entry) {
    return absl::NotFoundError("registration_id not found");
  }
  if (entry->gpu_ptr == nullptr) {
    return absl::FailedPreconditionError("registration has no GPU pointer");
  }
  return reinterpret_cast<uint64_t>(entry->gpu_ptr);
}

absl::StatusOr<RegistrationCpuMemfdInfo> RegistrationBackend::get_registration_cpu_memfd_info(
    std::string_view registration_id) const {
  auto entry = lookup_pending(registration_id);
  if (!entry) {
    return absl::NotFoundError("registration_id not found");
  }
  if (entry->plan != PendingRegistrationContext::Plan::kStableDram || entry->stable_dram.stage_on_gpu) {
    return absl::FailedPreconditionError("registration has no stable_dram cpu memory");
  }
  auto uma = entry->replica->get_memory_manager().memory_authority();
  if (uma == nullptr) {
    return absl::FailedPreconditionError("unified memory authority is unavailable");
  }
  auto region_or = uma->get_cpu_memfd_region(entry->pending_registry_key);
  if (!region_or.ok()) {
    return region_or.status();
  }
  const auto& region = *region_or;
  RegistrationCpuMemfdInfo info;
  info.replica_key = entry->pending_registry_key;
  info.fd = region.fd;
  info.size_bytes = region.size_bytes;
  info.offset_bytes = region.offset_bytes;
  return info;
}

absl::Status RegistrationBackend::abort(std::string_view registration_id) {
  size_t pending_size_after = 0;
  auto entry = erase_pending(registration_id, &pending_size_after);
  if (!entry) {
    return absl::NotFoundError("registration_id not found");
  }
  record_pending_gauge(pending_size_after);
  record_commit_latency(*entry, "aborted");
  erase_pending_registry_alias(*entry);
  const auto location = entry->plan == PendingRegistrationContext::Plan::kStableDram
      ? common::memory::MemoryLocation::CPU
      : common::memory::MemoryLocation::GPU;
  release_replica_memory(entry->replica, location);
  return absl::OkStatus();
}

absl::Status RegistrationBackend::keep_alive(std::string_view registration_id, uint32_t ttl_ms) {
  if (ttl_ms == 0) {
    return absl::OkStatus();
  }
  std::lock_guard<std::mutex> lock(pending_mutex_);
  auto it = pending_regs_.find(std::string(registration_id));
  if (it == pending_regs_.end()) {
    return absl::NotFoundError("registration_id not found");
  }
  it->second->expiry_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(ttl_ms);
  return absl::OkStatus();
}

std::shared_ptr<RegistrationBackend::PendingRegistrationContext> RegistrationBackend::erase_pending(
    std::string_view registration_id,
    size_t* pending_size_after) {
  std::shared_ptr<PendingRegistrationContext> entry;
  size_t new_size = 0;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_regs_.find(std::string(registration_id));
    if (it == pending_regs_.end()) {
      new_size = pending_regs_.size();
    } else {
      entry = it->second;
      pending_regs_.erase(it);
      new_size = pending_regs_.size();
    }
  }
  if (pending_size_after != nullptr) {
    *pending_size_after = new_size;
  }
  return entry;
}

std::shared_ptr<RegistrationBackend::PendingRegistrationContext> RegistrationBackend::lookup_pending(
    std::string_view registration_id) const {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  auto it = pending_regs_.find(std::string(registration_id));
  if (it == pending_regs_.end()) {
    return nullptr;
  }
  return it->second;
}

void RegistrationBackend::release_replica_memory(
    const std::shared_ptr<replica::Replica>& replica,
    common::memory::MemoryLocation location) {
  if (!replica) {
    return;
  }
  absl::Status st = replica->release_memory(location);
  if (!st.ok()) {
    VLOG(1) << "release_memory(" << static_cast<int>(location) << ") failed: " << st;
  }
}

void RegistrationBackend::erase_pending_registry_alias(
    const PendingRegistrationContext& entry,
    std::optional<loading::ReplicaKey> keep_key) {
  const auto& pending_key = entry.pending_registry_key;
  if (pending_key.artifact_id.empty()) {
    return;
  }
  if (keep_key.has_value() && pending_key == *keep_key) {
    return;
  }

  auto removed = replica_registry_->erase(pending_key);
  if (!removed.has_value()) {
    VLOG(2) << "RegistrationBackend: pending alias already absent registration_id=" << entry.registration_id
            << " key=" << pending_key;
    return;
  }

  if (entry.replica && removed->second.get() != entry.replica.get()) {
    LOG(WARNING) << "RegistrationBackend: pending alias key mapped to unexpected replica registration_id="
                 << entry.registration_id << " key=" << pending_key;
  } else {
    VLOG(1) << "RegistrationBackend: removed pending alias registration_id=" << entry.registration_id
            << " key=" << pending_key;
  }
}

void RegistrationBackend::record_pending_gauge(size_t pending_count) const {
  metrics_collector_->record_registration_pending(pending_count);
}

void RegistrationBackend::record_commit_latency(const PendingRegistrationContext& ctx, std::string_view status) const {
  if (ctx.begin_time.time_since_epoch().count() == 0) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  const double duration_seconds = std::chrono::duration<double>(now - ctx.begin_time).count();
  metrics_collector_->record_registration_commit(duration_seconds, status);
}

} // namespace tensorcast::store::runtime::metadata
