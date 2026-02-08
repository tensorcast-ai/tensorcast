// Copyright (c) 2025-2026, TensorCast Team.

// Implementation of RegistrationController

#include "daemon/service/controllers/registration_controller.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "core/cuda/cuda_ipc.h"
#include "core/cuda/device_guard.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "core/store/materialization/dataplane/sources/byte_range_map_builder.h"
#include "core/store/materialization/dataplane/verification/verification_utils.h"
#include "core/store/materialization/dataplane/view/view_ingest_executor.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/view_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/state/store_policy_resolver.h"
#include "daemon/util/status_utils.h"
#include "nlohmann/json.hpp"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

void ReleaseRegionRefs(IpcRegionRegistry& registry, const absl::flat_hash_map<std::string, uint32_t>& refs) {
  for (const auto& [region_id, count] : refs) {
    for (uint32_t i = 0; i < count; ++i) {
      absl::Status st = registry.release(region_id);
      if (!st.ok()) {
        LOG(WARNING) << "ReleaseRegionRefs: release failed for region=" << region_id << ": " << st;
      }
    }
  }
}

void EraseRegistrationRegionRefs(
    RegistrationManager& registration_manager,
    IpcRegionRegistry& registry,
    std::string_view registration_id) {
  auto refs = registration_manager.erase_all_for(std::string(registration_id));
  ReleaseRegionRefs(registry, refs);
}

class RegionPinGuard {
 public:
  explicit RegionPinGuard(IpcRegionRegistry& registry) : registry_(registry) {}

  ~RegionPinGuard() {
    if (!active_)
      return;
    ReleaseRegionRefs(registry_, refs_);
  }

  void add(const std::string& region_id) {
    ++refs_[region_id];
  }

  void release() {
    active_ = false;
    refs_.clear();
  }

  const absl::flat_hash_map<std::string, uint32_t>& refs() const {
    return refs_;
  }

 private:
  IpcRegionRegistry& registry_;
  absl::flat_hash_map<std::string, uint32_t> refs_;
  bool active_{true};
};

tensorcast::common::v1::ArtifactIdKind ToProtoKind(tensorcast::common::ArtifactIdKind kind) {
  using ProtoKind = tensorcast::common::v1::ArtifactIdKind;
  switch (kind) {
    case tensorcast::common::ArtifactIdKind::kCgid:
      return ProtoKind::ARTIFACT_ID_KIND_CGID;
    case tensorcast::common::ArtifactIdKind::kMi2:
      return ProtoKind::ARTIFACT_ID_KIND_MI2;
    case tensorcast::common::ArtifactIdKind::kUnspecified:
    default:
      return ProtoKind::ARTIFACT_ID_KIND_UNSPECIFIED;
  }
}

const char* requirement_level_label(RequirementLevel level) {
  switch (level) {
    case RequirementLevel::kNone:
      return "none";
    case RequirementLevel::kMay:
      return "may";
    case RequirementLevel::kShould:
      return "should";
    case RequirementLevel::kMust:
      return "must";
    default:
      return "unknown";
  }
}

const char* local_stable_op_label(RegistrationManager::RegPlan plan) {
  return plan == RegistrationManager::RegPlan::STABLE_DRAM ? "put" : "register";
}

const char* stable_retention_label(store::components::StableRetentionPolicy policy) {
  switch (policy) {
    case store::components::StableRetentionPolicy::kPinned:
      return "pinned";
    case store::components::StableRetentionPolicy::kTtl:
      return "ttl";
    case store::components::StableRetentionPolicy::kBestEffort:
    default:
      return "best_effort";
  }
}

const char* stable_overflow_label(store::components::StableOverflowPolicy policy) {
  switch (policy) {
    case store::components::StableOverflowPolicy::kEvict:
      return "evict";
    case store::components::StableOverflowPolicy::kSpill:
      return "spill";
    case store::components::StableOverflowPolicy::kReject:
    default:
      return "reject";
  }
}

using materialization_policy::compute_view_id_from_spec;
using materialization_policy::convert_view_spec;

void record_local_stable_tier_metrics(
    const char* op,
    const char* status,
    const char* requirement,
    std::optional<double> seconds);

absl::StatusOr<store::StoreEngine::StableCacheAdmissionResult> ensure_local_stable_admission(
    store::StoreEngine& engine,
    LipManager& lip_mgr,
    IpcRegionRegistry& regions,
    const ResolvedStorePolicy& policy,
    std::string_view artifact_id,
    tensorcast::common::ArtifactIdKind id_kind,
    uint64_t total_size,
    int owner_pid,
    std::string_view canonical_index_json,
    std::string_view index_key_hex);

absl::StatusOr<ResolvedStorePolicy> resolve_effective_store_policy(const RegistrationManager::RegMeta& meta) {
  if (meta.resolved_policy.has_value()) {
    return *meta.resolved_policy;
  }
  return resolve_store_policy(nullptr);
}

absl::Status apply_local_stable_tier(
    store::StoreEngine& engine,
    LipManager& lip,
    IpcRegionRegistry& regions,
    const RegistrationManager::RegMeta& meta,
    std::string_view artifact_id,
    tensorcast::common::ArtifactIdKind id_kind,
    uint64_t total_size,
    v2::LocalStableTierResult* local_stable,
    const std::function<void()>& on_must_failure_cleanup) {
  auto resolved_or = resolve_effective_store_policy(meta);
  if (!resolved_or.ok()) {
    return resolved_or.status();
  }
  const ResolvedStorePolicy& resolved = *resolved_or;
  const char* op_label = local_stable_op_label(meta.plan);
  const char* requirement = requirement_level_label(resolved.local_requirement);

  if (meta.view_registration) {
    local_stable->set_status(v2::LOCAL_STABLE_TIER_STATUS_SKIPPED);
    local_stable->set_message("view registrations do not satisfy the local stable tier");
    record_local_stable_tier_metrics(op_label, "skipped", requirement, std::nullopt);
    return absl::OkStatus();
  }
  if (static_cast<int>(resolved.local_requirement) < static_cast<int>(RequirementLevel::kShould)) {
    local_stable->set_status(v2::LOCAL_STABLE_TIER_STATUS_SKIPPED);
    record_local_stable_tier_metrics(op_label, "skipped", requirement, std::nullopt);
    return absl::OkStatus();
  }

  const auto stable_policy_opt = stable_cache_policy_from_resolved(resolved);
  const auto local_stable_start = std::chrono::steady_clock::now();
  auto admit_or = ensure_local_stable_admission(
      engine,
      lip,
      regions,
      resolved,
      artifact_id,
      id_kind,
      total_size,
      meta.owner_pid,
      meta.index_data,
      meta.index_key_hex);
  const double local_stable_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - local_stable_start).count();
  if (!admit_or.ok() || admit_or->skipped) {
    const std::string message = admit_or.ok()
        ? "local stable tier admission skipped"
        : std::string(admit_or.status().message().data(), admit_or.status().message().size());
    const char* retention =
        stable_policy_opt.has_value() ? stable_retention_label(stable_policy_opt->retention_policy) : "unknown";
    const char* overflow =
        stable_policy_opt.has_value() ? stable_overflow_label(stable_policy_opt->overflow_policy) : "unknown";
    const char* outcome = resolved.local_requirement == RequirementLevel::kMust ? "failed" : "degraded";
    LOG(WARNING) << "local_stable_tier." << outcome << ": artifact_id=" << artifact_id << " op=" << op_label
                 << " requirement=" << requirement << " retention=" << retention << " overflow=" << overflow
                 << " seconds=" << local_stable_seconds << " message=\"" << message << "\"";
    if (resolved.local_requirement == RequirementLevel::kMust) {
      record_local_stable_tier_metrics(op_label, "failed", requirement, local_stable_seconds);
      on_must_failure_cleanup();
      return admit_or.ok() ? absl::FailedPreconditionError(message) : admit_or.status();
    }
    local_stable->set_status(v2::LOCAL_STABLE_TIER_STATUS_DEGRADED);
    local_stable->set_message(message);
    record_local_stable_tier_metrics(op_label, "degraded", requirement, local_stable_seconds);
    return absl::OkStatus();
  }

  local_stable->set_status(v2::LOCAL_STABLE_TIER_STATUS_READY);
  const char* retention =
      stable_policy_opt.has_value() ? stable_retention_label(stable_policy_opt->retention_policy) : "unknown";
  const char* overflow =
      stable_policy_opt.has_value() ? stable_overflow_label(stable_policy_opt->overflow_policy) : "unknown";
  LOG(INFO) << "local_stable_tier.ready: artifact_id=" << artifact_id << " op=" << op_label
            << " requirement=" << requirement << " retention=" << retention << " overflow=" << overflow
            << " seconds=" << local_stable_seconds;
  record_local_stable_tier_metrics(op_label, "ready", requirement, local_stable_seconds);
  return absl::OkStatus();
}

void record_local_stable_tier_metrics(
    const char* op,
    const char* status,
    const char* requirement,
    std::optional<double> seconds) {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_local_stable_tier_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("op", opentelemetry::common::AttributeValue(std::string(op)));
    attrs.emplace("status", opentelemetry::common::AttributeValue(std::string(status)));
    attrs.emplace("requirement", opentelemetry::common::AttributeValue(std::string(requirement)));
    counter->Add(1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
    // Metrics must not affect control flow.
  }

  if (!seconds.has_value()) {
    return;
  }
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto hist = meter->CreateDoubleHistogram("tc_local_stable_tier_seconds");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("op", opentelemetry::common::AttributeValue(std::string(op)));
    attrs.emplace("status", opentelemetry::common::AttributeValue(std::string(status)));
    hist->Record(*seconds, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
    // Metrics must not affect control flow.
  }
}

store::StoreEngine::ViewPlacement ToPlacement(v2::TransformPlacement placement) {
  switch (placement) {
    case v2::TRANSFORM_PLACEMENT_SERVER:
      return store::StoreEngine::ViewPlacement::kServer;
    case v2::TRANSFORM_PLACEMENT_CLIENT:
      return store::StoreEngine::ViewPlacement::kClient;
    case v2::TRANSFORM_PLACEMENT_UNSPECIFIED:
    default:
      return store::StoreEngine::ViewPlacement::kUnspecified;
  }
}

store::StoreEngine::ViewRegistrationKind ToRegistrationKind(v2::ViewRegistrationKind kind) {
  switch (kind) {
    case v2::VIEW_REGISTRATION_KIND_CANONICAL:
      return store::StoreEngine::ViewRegistrationKind::kCanonical;
    case v2::VIEW_REGISTRATION_KIND_PIECE:
      return store::StoreEngine::ViewRegistrationKind::kPiece;
    case v2::VIEW_REGISTRATION_KIND_UNSPECIFIED:
    default:
      return store::StoreEngine::ViewRegistrationKind::kUnspecified;
  }
}

void record_view_bytes_metric(double bytes) {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_register_view_bytes_total");
    counter->Add(bytes);
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_view_bytes_total unavailable";
  }
}

void record_view_partial_metric() {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_register_view_partials_total");
    counter->Add(1.0);
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_view_partials_total unavailable";
  }
}

struct RegistrationAbortGuard {
  store::StoreEngine* engine{nullptr};
  std::string registration_id;
  bool active{true};

  ~RegistrationAbortGuard() {
    if (!active || engine == nullptr || registration_id.empty()) {
      return;
    }
    absl::Status st = engine->abort_registered_artifact(registration_id);
    if (!st.ok()) {
      LOG(WARNING) << "RegistrationAbortGuard: abort_registered_artifact failed for id=" << registration_id << ": "
                   << st;
    }
  }

  void release() {
    active = false;
  }
};

absl::Status copy_to_staging_from_coalesced_gpu(
    int device_id,
    gsl::not_null<void*> dst_dev,
    gsl::not_null<void*> src_dev,
    absl::Span<const store::loader::ByteRangeSegment> segments,
    uint64_t total_size) {
  cuda::CudaDeviceGuard guard(device_id);
  if (!guard.status().ok()) {
    return guard.status();
  }
  if (segments.empty()) {
    return cuda::memcpy(dst_dev, src_dev, static_cast<size_t>(total_size), cudaMemcpyDeviceToDevice);
  }
  for (const auto& seg : segments) {
    if (seg.length == 0) {
      continue;
    }
    if (seg.kind == store::loader::ByteRangeSegment::Kind::kPad) {
      auto st = cuda::memset(static_cast<uint8_t*>(dst_dev.get()) + seg.dst_offset, 0, static_cast<size_t>(seg.length));
      if (!st.ok()) {
        return st;
      }
      continue;
    }
    auto st = cuda::memcpy(
        static_cast<uint8_t*>(dst_dev.get()) + seg.dst_offset,
        static_cast<const uint8_t*>(src_dev.get()) + seg.src_offset,
        static_cast<size_t>(seg.length),
        cudaMemcpyDeviceToDevice);
    if (!st.ok()) {
      return st;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<cuda::IpcMapping*> get_or_open_mapping_for_storage(
    const RegisterStorageMeta& storage,
    absl::flat_hash_map<std::string, std::unique_ptr<cuda::IpcMapping>>& cache,
    RegionPinGuard& region_pin,
    IpcRegionRegistry& regions,
    int owner_pid) {
  const std::string cache_key =
      storage.has_handle() ? absl::StrCat("h:", storage.handle_bytes) : absl::StrCat("r:", storage.region_id);
  auto it = cache.find(cache_key);
  if (it != cache.end()) {
    return it->second.get();
  }

  std::unique_ptr<cuda::IpcMapping> mapping;
  if (storage.has_handle()) {
    auto map_or =
        cuda::IpcMapping::open(storage.handle_bytes, cuda::OpenOptions{.flags = cudaIpcMemLazyEnablePeerAccess});
    if (!map_or.ok()) {
      return map_or.status();
    }
    mapping = std::make_unique<cuda::IpcMapping>(std::move(*map_or));
  } else if (storage.has_region()) {
    auto acq_or = regions.acquire(storage.region_id, owner_pid);
    if (!acq_or.ok()) {
      return acq_or.status();
    }
    region_pin.add(storage.region_id);
    auto handle_or = regions.get_handle_bytes(storage.region_id);
    if (!handle_or.ok()) {
      return handle_or.status();
    }
    auto map_or = cuda::IpcMapping::open(*handle_or, cuda::OpenOptions{.flags = cudaIpcMemLazyEnablePeerAccess});
    if (!map_or.ok()) {
      return map_or.status();
    }
    mapping = std::make_unique<cuda::IpcMapping>(std::move(*map_or));
  } else {
    return absl::InvalidArgumentError("storage entry missing source handle or region");
  }

  auto [insert_it, _] = cache.emplace(cache_key, std::move(mapping));
  return insert_it->second.get();
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

std::vector<store::view::CanonicalRange> canonical_ranges_from_write_plan(
    const store::loader::ViewWritePlan& write_plan) {
  std::vector<store::view::CanonicalRange> ranges;
  ranges.reserve(write_plan.chunks.size());
  for (const auto& chunk : write_plan.chunks) {
    store::view::CanonicalRange range;
    range.offset = chunk.canonical_offset;
    range.length = chunk.length;
    ranges.push_back(range);
  }
  std::sort(
      ranges.begin(), ranges.end(), [](const store::view::CanonicalRange& a, const store::view::CanonicalRange& b) {
        return a.offset < b.offset;
      });
  std::vector<store::view::CanonicalRange> merged;
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

bool ranges_cover_interval(const std::vector<store::view::CanonicalRange>& ranges, uint64_t start, uint64_t length) {
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

struct CanonicalToViewSpan {
  uint64_t canonical_offset{0};
  uint64_t view_offset{0};
  uint64_t length{0};
};

absl::StatusOr<std::vector<CanonicalToViewSpan>> canonical_spans_for_tensor(
    const store::loader::ViewWritePlan& write_plan,
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

class LipDenseViewSource final : public store::loader::SeekableSource {
 public:
  struct Segment {
    int device_id{0};
    cuda::IpcMapping* mapping{nullptr};
    uint64_t base_offset{0};
    uint64_t view_offset{0};
    uint64_t length{0};
  };

  explicit LipDenseViewSource(std::vector<Segment> segments, uint64_t total_bytes)
      : segments_(std::move(segments)), total_bytes_(total_bytes) {
    std::sort(segments_.begin(), segments_.end(), [](const Segment& a, const Segment& b) {
      return a.view_offset < b.view_offset;
    });
  }

  ~LipDenseViewSource() override = default;

  [[nodiscard]] uint64_t total_bytes() const override {
    return total_bytes_;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(cursor_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or;
    }
    cursor_ += *read_or;
    return read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= total_bytes_) {
      return static_cast<size_t>(0);
    }
    size_t remaining = static_cast<size_t>(std::min<uint64_t>(bytes, total_bytes_ - offset));
    auto* out = static_cast<uint8_t*>(dst);
    while (remaining > 0) {
      const Segment* seg = nullptr;
      for (const auto& candidate : segments_) {
        if (offset >= candidate.view_offset && offset < candidate.view_offset + candidate.length) {
          seg = &candidate;
          break;
        }
      }
      if (seg == nullptr || seg->mapping == nullptr) {
        return absl::FailedPreconditionError("missing LIP segment coverage while reading view bytes");
      }
      const uint64_t local = offset - seg->view_offset;
      const size_t avail = static_cast<size_t>(seg->length - local);
      const size_t take = std::min(remaining, avail);
      cuda::CudaDeviceGuard guard(seg->device_id);
      if (!guard.status().ok()) {
        return guard.status();
      }
      auto st = cuda::memcpy(
          out,
          static_cast<const uint8_t*>(seg->mapping->get()) + static_cast<std::ptrdiff_t>(seg->base_offset + local),
          take,
          cudaMemcpyDeviceToHost);
      if (!st.ok()) {
        return st;
      }
      if (auto sync = cuda::device_synchronize(); !sync.ok()) {
        return sync;
      }
      out += take;
      offset += take;
      remaining -= take;
    }
    return static_cast<size_t>(out - static_cast<uint8_t*>(dst));
  }

  [[nodiscard]] bool supports_direct_write_at() const override {
    return false;
  }

 private:
  std::vector<Segment> segments_;
  uint64_t total_bytes_{0};
  uint64_t cursor_{0};
};

absl::Status copy_to_staging_from_lip_sources(
    int device_id,
    gsl::not_null<void*> dst_dev,
    absl::Span<const store::loader::ByteRangeSegment> plan_segments,
    uint64_t total_size,
    absl::Span<const LeaseSegMeta> segments,
    absl::Span<const RegisterStorageMeta> storages,
    IpcRegionRegistry& regions,
    int owner_pid) {
  cuda::CudaDeviceGuard guard(device_id);
  if (!guard.status().ok()) {
    return guard.status();
  }
  if (plan_segments.empty()) {
    return absl::InvalidArgumentError("LIP local-stable copy requires a canonical segment plan");
  }

  absl::flat_hash_map<std::string, const RegisterStorageMeta*> storage_by_id;
  storage_by_id.reserve(storages.size());
  for (const auto& s : storages) {
    storage_by_id.emplace(s.storage_id, &s);
  }

  struct OpenedSeg {
    int device_id;
    cuda::IpcMapping* map;
    uint64_t base;
    uint64_t len;
    uint64_t dst;
  };

  RegionPinGuard pin_guard(regions);
  absl::flat_hash_map<std::string, std::unique_ptr<cuda::IpcMapping>> mapping_cache;
  mapping_cache.reserve(storages.size());

  std::vector<OpenedSeg> opened;
  opened.reserve(segments.size());
  for (const auto& seg : segments) {
    auto it = storage_by_id.find(seg.storage_id);
    if (it == storage_by_id.end()) {
      return absl::InvalidArgumentError("segment references unknown storage_id");
    }
    const RegisterStorageMeta* storage = it->second;
    if (seg.length != storage->storage_length) {
      return absl::InvalidArgumentError(
          absl::StrCat("segment length does not match storage_length for storage_id=", storage->storage_id));
    }
    if (seg.storage_offset != 0) {
      return absl::InvalidArgumentError("segment storage_offset must be 0 for full-storage registrations");
    }
    auto map_or = get_or_open_mapping_for_storage(*storage, mapping_cache, pin_guard, regions, owner_pid);
    if (!map_or.ok()) {
      return map_or.status();
    }
    OpenedSeg opened_seg{
        .device_id = storage->device_id,
        .map = *map_or,
        .base = storage->mapping_base_offset + seg.storage_offset,
        .len = seg.length,
        .dst = seg.artifact_offset,
    };
    if (opened_seg.device_id != device_id) {
      return absl::UnimplementedError("multi-device lease sources are not supported for local stable materialization");
    }
    opened.push_back(opened_seg);
  }
  std::sort(opened.begin(), opened.end(), [](const OpenedSeg& a, const OpenedSeg& b) { return a.dst < b.dst; });

  auto find_covering = [&](uint64_t off) -> const OpenedSeg* {
    for (const auto& s : opened) {
      if (off >= s.dst && off < (s.dst + s.len)) {
        return &s;
      }
      if (off < s.dst) {
        break;
      }
    }
    return nullptr;
  };

  for (const auto& plan : plan_segments) {
    if (plan.length == 0) {
      continue;
    }
    if (plan.source_index != 0) {
      return absl::UnimplementedError("local stable materialization supports only source_index=0 for LIP sources");
    }
    if (plan.dst_offset > total_size || plan.length > total_size || (plan.dst_offset + plan.length) > total_size) {
      return absl::OutOfRangeError("segment plan dst range out of bounds");
    }
    if (plan.kind == store::loader::ByteRangeSegment::Kind::kPad) {
      auto st =
          cuda::memset(static_cast<uint8_t*>(dst_dev.get()) + plan.dst_offset, 0, static_cast<size_t>(plan.length));
      if (!st.ok()) {
        return st;
      }
      continue;
    }
    if (plan.src_offset > total_size || plan.length > total_size || (plan.src_offset + plan.length) > total_size) {
      return absl::OutOfRangeError("segment plan src range out of bounds");
    }
    uint64_t remaining = plan.length;
    uint64_t src_off = plan.src_offset;
    uint64_t dst_off = plan.dst_offset;
    while (remaining > 0) {
      const OpenedSeg* seg = find_covering(src_off);
      if (seg == nullptr) {
        return absl::FailedPreconditionError("segment plan references source offsets not covered by lease segments");
      }
      const uint64_t seg_end = seg->dst + seg->len;
      const uint64_t avail = seg_end - src_off;
      const uint64_t take64 = std::min<uint64_t>(remaining, avail);
      auto st = cuda::memcpy(
          static_cast<uint8_t*>(dst_dev.get()) + dst_off,
          static_cast<uint8_t*>(seg->map->get()) + seg->base + (src_off - seg->dst),
          static_cast<size_t>(take64),
          cudaMemcpyDeviceToDevice);
      if (!st.ok()) {
        return st;
      }
      src_off += take64;
      dst_off += take64;
      remaining -= take64;
    }
  }
  return absl::OkStatus();
}

template <typename CopyFn>
absl::StatusOr<store::StoreEngine::RegistrationCommitResult> run_stable_dram_registration(
    store::StoreEngine& engine,
    int staging_device_id,
    uint64_t total_size,
    std::string_view canonical_index_json,
    std::string_view index_key_hex,
    std::string_view artifact_id,
    tensorcast::common::ArtifactIdKind id_kind,
    std::optional<store::components::StableDramCachePolicy> stable_cache_policy,
    CopyFn&& copy_fn) {
  store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = absl::StrCat("local_stable:", absl::ToUnixNanos(absl::Now()), ":", getpid());
  reg.plan = store::runtime::metadata::RegistrationPlan::kStableDram;
  reg.device_id = staging_device_id;
  reg.total_size_bytes = total_size;
  reg.enable_p2p = true;
  reg.schema_version = "v3";
  reg.encoding = "json";
  if (!index_key_hex.empty()) {
    reg.tensor_index_key = std::string(index_key_hex);
  }
  if (!canonical_index_json.empty()) {
    reg.tensor_index_data = std::string(canonical_index_json);
  }
  if (id_kind == tensorcast::common::ArtifactIdKind::kMi2) {
    reg.artifact_id_override = std::string(artifact_id);
  } else if (id_kind == tensorcast::common::ArtifactIdKind::kCgid) {
    reg.client_artifact_id = std::string(artifact_id);
  } else {
    return absl::InvalidArgumentError("unsupported artifact_id kind for stable registration");
  }
  if (stable_cache_policy.has_value()) {
    reg.stable_cache_policy = *stable_cache_policy;
  }
  reg.stable_dram.stage_on_gpu = true;
  reg.stable_dram.release_gpu_on_commit = true;

  auto begin_or = engine.begin_register_artifact(reg);
  if (!begin_or.ok()) {
    return begin_or.status();
  }
  const auto& begin = *begin_or;
  RegistrationAbortGuard abort_guard{.engine = &engine, .registration_id = begin.registration_id};

  auto dst_ptr_or = engine.get_registration_gpu_ptr(begin.registration_id);
  if (!dst_ptr_or.ok()) {
    return dst_ptr_or.status();
  }
  auto* dst_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(*dst_ptr_or));
  if (dst_ptr == nullptr) {
    return absl::FailedPreconditionError("stable registration staging pointer is null");
  }
  absl::Status copy_status = copy_fn(gsl::not_null<void*>{dst_ptr});
  if (!copy_status.ok()) {
    return copy_status;
  }
  if (auto sync = cuda::device_synchronize(); !sync.ok()) {
    return sync;
  }
  auto commit_or = engine.commit_registered_artifact(begin.registration_id);
  if (!commit_or.ok()) {
    return commit_or.status();
  }
  abort_guard.release();
  return *commit_or;
}

absl::StatusOr<store::loading::ReplicaKey> find_loaded_cpu_replica(store::StoreEngine& engine, std::string_view id) {
  const auto devices = engine.get_resident_devices(id);
  for (const auto& device : devices) {
    if (device.type != DeviceType::CPU) {
      continue;
    }
    store::loading::ReplicaKey key;
    key.artifact_id = std::string(id);
    key.device = device;
    key.replica = 0;
    if (engine.get_replica_state(key, DeviceType::CPU) != store::replica::MemoryState::LOADED) {
      return absl::FailedPreconditionError("stable DRAM replica is not loaded");
    }
    return key;
  }
  return absl::NotFoundError("stable DRAM replica not found");
}

std::optional<store::loading::ReplicaKey> find_loaded_gpu_replica(store::StoreEngine& engine, std::string_view id) {
  const auto devices = engine.get_resident_devices(id);
  for (const auto& device : devices) {
    if (device.type != DeviceType::GPU) {
      continue;
    }
    store::loading::ReplicaKey key;
    key.artifact_id = std::string(id);
    key.device = device;
    key.replica = 0;
    if (engine.get_replica_state(key, DeviceType::GPU) != store::replica::MemoryState::LOADED) {
      continue;
    }
    return key;
  }
  return std::nullopt;
}

absl::StatusOr<store::StoreEngine::StableCacheAdmissionResult> ensure_local_stable_admission(
    store::StoreEngine& engine,
    LipManager& lip_mgr,
    IpcRegionRegistry& regions,
    const ResolvedStorePolicy& policy,
    std::string_view artifact_id,
    tensorcast::common::ArtifactIdKind id_kind,
    uint64_t total_size,
    int owner_pid,
    std::string_view canonical_index_json,
    std::string_view index_key_hex) {
  auto stable_policy_opt = stable_cache_policy_from_resolved(policy);
  if (!stable_policy_opt.has_value()) {
    return absl::FailedPreconditionError("stable cache policy missing for local stable tier");
  }

  auto cpu_key_or = find_loaded_cpu_replica(engine, artifact_id);
  if (cpu_key_or.ok()) {
    auto admit_or = engine.admit_stable_cache_policy(*cpu_key_or, *stable_policy_opt);
    if (!admit_or.ok()) {
      return admit_or.status();
    }
    return *admit_or;
  }
  if (!absl::IsNotFound(cpu_key_or.status())) {
    return cpu_key_or.status();
  }

  if (auto gpu_key_opt = find_loaded_gpu_replica(engine, artifact_id); gpu_key_opt.has_value()) {
    const int device_id = gpu_key_opt->device.ordinal;
    auto ptr_or = engine.get_replica_gpu_ptr(*gpu_key_opt);
    if (!ptr_or.ok()) {
      return ptr_or.status();
    }
    auto* src_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(*ptr_or));
    if (src_ptr == nullptr) {
      return absl::FailedPreconditionError("GPU source pointer is null for local stable materialization");
    }
    std::optional<store::loader::ByteRangeMap> canonical_map;
    if (!canonical_index_json.empty()) {
      auto map_or = store::loader::build_byte_range_map_from_canonical_index_json(canonical_index_json, total_size);
      if (!map_or.ok()) {
        return map_or.status();
      }
      canonical_map = std::move(*map_or);
    }
    const std::optional<store::components::StableDramCachePolicy> required_policy =
        policy.local_requirement == RequirementLevel::kMust ? stable_policy_opt : std::nullopt;
    auto commit_or = run_stable_dram_registration(
        engine,
        device_id,
        total_size,
        canonical_index_json,
        index_key_hex,
        artifact_id,
        id_kind,
        required_policy,
        [&](gsl::not_null<void*> dst_dev) -> absl::Status {
          return copy_to_staging_from_coalesced_gpu(
              device_id,
              dst_dev,
              gsl::not_null<void*>{src_ptr},
              canonical_map.has_value() ? absl::MakeSpan(canonical_map->segments)
                                        : absl::Span<const store::loader::ByteRangeSegment>(),
              total_size);
        });
    if (!commit_or.ok()) {
      return commit_or.status();
    }
  } else {
    auto lip_opt = lip_mgr.find_active_by_artifact_id(std::string(artifact_id));
    if (!lip_opt.has_value()) {
      return absl::FailedPreconditionError("no eligible source found for local stable materialization");
    }
    const auto& lip = *lip_opt;
    const std::string_view index_json = !canonical_index_json.empty() ? canonical_index_json : lip.index_data;
    if (index_json.empty()) {
      return absl::FailedPreconditionError("canonical index JSON required for local stable materialization from LIP");
    }
    auto map_or = store::loader::build_byte_range_map_from_canonical_index_json(index_json, total_size);
    if (!map_or.ok()) {
      return map_or.status();
    }
    const std::optional<store::components::StableDramCachePolicy> required_policy =
        policy.local_requirement == RequirementLevel::kMust ? stable_policy_opt : std::nullopt;
    const int device_id = lip.device_id;
    auto commit_or = run_stable_dram_registration(
        engine,
        device_id,
        total_size,
        index_json,
        std::string_view(),
        artifact_id,
        id_kind,
        required_policy,
        [&](gsl::not_null<void*> dst_dev) -> absl::Status {
          return copy_to_staging_from_lip_sources(
              device_id,
              dst_dev,
              absl::MakeSpan(map_or->segments),
              total_size,
              absl::MakeSpan(lip.segments),
              absl::MakeSpan(lip.storages),
              regions,
              owner_pid);
        });
    if (!commit_or.ok()) {
      return commit_or.status();
    }
  }

  auto cpu_key_or2 = find_loaded_cpu_replica(engine, artifact_id);
  if (!cpu_key_or2.ok()) {
    return cpu_key_or2.status();
  }
  auto admit_or = engine.admit_stable_cache_policy(*cpu_key_or2, *stable_policy_opt);
  if (!admit_or.ok()) {
    return admit_or.status();
  }
  return *admit_or;
}

template <typename NextRequestFn>
grpc::Status process_feed_requests(
    RegistrationController::Dep& dep,
    NextRequestFn&& next_request,
    const char* source_label) {
  v2::FeedRegisterArtifactStreamRequest req;
  std::string reg_id;
  RegistrationManager::RegMeta current_meta;
  bool have_meta = false;
  while (next_request(&req)) {
    if (reg_id.empty()) {
      reg_id = req.registration_id();
      if (!dep.reg.has_meta(reg_id)) {
        return {StatusCode::NOT_FOUND, "registration_id not found"};
      }
      absl::flat_hash_map<std::string, uint32_t> expired_refs;
      if (dep.reg.expire_if_ttl_elapsed(reg_id, &expired_refs)) {
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto counter = meter->CreateDoubleCounter("tc_register_ttl_expired_feed_total");
          counter->Add(1.0);
        } catch (...) {
          VLOG(1) << "metrics counter tc_register_ttl_expired_feed_total unavailable";
        }
        ReleaseRegionRefs(dep.regions, expired_refs);
        return {StatusCode::DEADLINE_EXCEEDED, "registration expired (TTL)"};
      }
      auto meta_opt = dep.reg.get_meta(reg_id);
      if (!meta_opt.has_value()) {
        return {StatusCode::NOT_FOUND, "registration metadata missing"};
      }
      current_meta = *meta_opt;
      have_meta = true;
    } else if (req.registration_id() != reg_id) {
      return {StatusCode::INVALID_ARGUMENT, "registration_id changed in stream"};
    }

    const uint32_t extend_ms = dep.reg.extend_if_has_ttl(reg_id);
    if (extend_ms > 0) {
      auto st = dep.engine.keep_alive_registered_artifact(reg_id, extend_ms);
      if (!st.ok()) {
        LOG(WARNING) << "keep_alive_registered_artifact failed (" << source_label << "): reg_id=" << reg_id << ": "
                     << st;
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto ctr = meter->CreateDoubleCounter("tc_register_keepalive_failed_total");
          ctr->Add(1.0);
        } catch (...) {
        }
      }
    }

    if (req.has_lease_segments()) {
      std::vector<LeaseSegMeta> to_add;
      to_add.reserve(req.lease_segments().segments_size());
      for (const auto& s : req.lease_segments().segments()) {
        LeaseSegMeta m;
        m.storage_id = s.storage_id();
        m.storage_offset = s.storage_offset();
        m.artifact_offset = s.artifact_offset();
        m.length = s.length();
        if (m.storage_id.empty()) {
          return {StatusCode::INVALID_ARGUMENT, "lease segment missing storage_id"};
        }
        to_add.push_back(std::move(m));
      }
      dep.reg.append_lease_segments(reg_id, std::move(to_add));
    } else if (req.has_view_chunk()) {
      const std::string& payload = req.view_chunk().data();
      absl::Span<const std::byte> bytes(reinterpret_cast<const std::byte*>(payload.data()), payload.size());
      auto ingest_status = dep.engine.ingest_view_registration_chunk(reg_id, req.view_chunk().view_offset(), bytes);
      if (!ingest_status.ok()) {
        return to_grpc_status(ingest_status);
      }
      record_view_bytes_metric(static_cast<double>(payload.size()));
      auto ingested_or = dep.engine.get_view_registration_ingested_bytes(reg_id);
      if (ingested_or.ok()) {
        dep.reg.update_view_ingested_bytes(reg_id, *ingested_or);
      }
    } else if (!req.storage_entries().empty() || !req.tensor_aliases().empty()) {
      // allow metadata-only payloads
    } else {
      return {StatusCode::INVALID_ARGUMENT, "missing feed payload"};
    }

    if (!req.storage_entries().empty()) {
      if (!have_meta) {
        auto meta_opt = dep.reg.get_meta(reg_id);
        if (!meta_opt.has_value()) {
          return {StatusCode::NOT_FOUND, "registration metadata missing"};
        }
        current_meta = *meta_opt;
        have_meta = true;
      }
      RegionPinGuard pin_guard(dep.regions);
      std::vector<RegisterStorageMeta> storages;
      storages.reserve(req.storage_entries().size());
      for (const auto& entry : req.storage_entries()) {
        RegisterStorageMeta meta;
        meta.storage_id = entry.storage_id();
        meta.device_id = entry.device_id();
        if (!entry.cuda_ipc_handle().empty()) {
          meta.handle_bytes = entry.cuda_ipc_handle();
        }
        meta.storage_length = entry.storage_length();
        const bool has_region = !entry.vram_region_id().empty();
        if (has_region) {
          meta.region_id = entry.vram_region_id();
        }
        meta.mapping_base_offset = entry.mapping_base_offset();
        if (meta.handle_bytes.empty() == meta.region_id.empty()) {
          return {StatusCode::INVALID_ARGUMENT, "storage entry must specify exactly one source"};
        }
        if (has_region) {
          // Validate using describe() first to avoid leaking a ref on failure.
          auto desc_or = dep.regions.describe(meta.region_id);
          if (!desc_or.ok()) {
            return to_grpc_status(desc_or.status());
          }
          const auto& desc = *desc_or;
          if (desc.device_id != meta.device_id) {
            return {StatusCode::FAILED_PRECONDITION, "region device does not match storage device"};
          }
          const uint64_t offset = meta.mapping_base_offset;
          const uint64_t length = meta.storage_length;
          if (length > desc.size_bytes || offset > (desc.size_bytes - length)) {
            return {StatusCode::FAILED_PRECONDITION, "region-backed storage exceeds region bounds"};
          }
          // Acquire only after validation and track immediately for cleanup safety.
          auto acq_or = dep.regions.acquire(meta.region_id, current_meta.owner_pid);
          if (!acq_or.ok()) {
            return to_grpc_status(acq_or.status());
          }
          pin_guard.add(meta.region_id);
        }
        storages.push_back(std::move(meta));
      }
      if (!storages.empty()) {
        dep.reg.append_storage_entries(reg_id, std::move(storages));
        for (const auto& [region_id, count] : pin_guard.refs()) {
          dep.reg.add_region_reference(reg_id, region_id, count);
        }
        pin_guard.release();
      }
    }

    if (!req.tensor_aliases().empty()) {
      std::vector<RegisterTensorAliasMeta> aliases;
      aliases.reserve(req.tensor_aliases().size());
      for (const auto& alias : req.tensor_aliases()) {
        RegisterTensorAliasMeta meta;
        meta.name = alias.name();
        meta.storage_id = alias.storage_id();
        meta.storage_offset = alias.storage_offset();
        meta.logical_length = alias.logical_length();
        meta.shape.reserve(alias.shape().size());
        for (int64_t v : alias.shape()) {
          meta.shape.push_back(v);
        }
        meta.stride.reserve(alias.stride().size());
        for (int64_t v : alias.stride()) {
          meta.stride.push_back(v);
        }
        meta.dtype = alias.dtype();
        aliases.push_back(std::move(meta));
      }
      if (!aliases.empty()) {
        dep.reg.append_tensor_aliases(reg_id, std::move(aliases));
      }
    }
  }
  return Status::OK;
}

absl::StatusOr<bool> is_piece_assembly_sealed(
    store::components::IGlobalStoreClient* global_store_client,
    std::string_view client_artifact_id) {
  if (global_store_client == nullptr || !global_store_client->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  auto binding_or = global_store_client->get_artifact_binding(std::string(client_artifact_id));
  if (binding_or.ok()) {
    return true;
  }
  if (absl::IsNotFound(binding_or.status())) {
    return false;
  }
  return binding_or.status();
}

grpc::Status commit_piece_view_registration(
    RegistrationController::Dep& dep,
    RpcContext& rctx,
    const std::string& registration_id,
    const RegistrationManager::RegMeta& meta,
    std::vector<LeaseSegMeta> lease_vec,
    std::vector<RegisterStorageMeta> storage_entries,
    v2::CommitRegisteredArtifactResponse& resp) {
  if (!dep.global_store_client || !dep.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }
  if (meta.client_artifact_id.empty()) {
    return {StatusCode::INVALID_ARGUMENT, "piece registration requires client_artifact_id (cgid)"};
  }
  if (meta.view_id.empty()) {
    return {StatusCode::FAILED_PRECONDITION, "piece registration missing view_id"};
  }
  if (meta.view_spec_json.empty()) {
    return {StatusCode::FAILED_PRECONDITION, "piece registration missing view_spec_json"};
  }
  if (meta.index_data.empty()) {
    return {StatusCode::FAILED_PRECONDITION, "piece registration missing tensor_index_data"};
  }

  const auto view_spec_or = store::view::parse_view_spec_json(meta.view_spec_json);
  if (!view_spec_or.ok()) {
    return to_grpc_status(view_spec_or.status());
  }
  const auto plan_or = store::loader::ViewPlanner::compute_bidirectional_view_plan(meta.index_data, *view_spec_or);
  if (!plan_or.ok()) {
    return to_grpc_status(plan_or.status());
  }
  const auto& view_plan = *plan_or;
  const uint64_t view_total = view_plan.forward.view_size_bytes;
  if (view_total == 0) {
    return {StatusCode::FAILED_PRECONDITION, "piece registration requires non-empty view_size_bytes"};
  }
  if (meta.total_size != 0 && meta.total_size != view_total) {
    return {
        StatusCode::FAILED_PRECONDITION,
        absl::StrCat("piece registration total_size mismatch: requested=", meta.total_size, " computed=", view_total)};
  }

  // Validate dense segment coverage [0, view_total) so remote keys can be sequenced.
  {
    std::vector<LeaseSegMeta> sorted = lease_vec;
    std::sort(sorted.begin(), sorted.end(), [](const LeaseSegMeta& a, const LeaseSegMeta& b) {
      return a.artifact_offset < b.artifact_offset;
    });
    uint64_t cursor = 0;
    for (const auto& seg : sorted) {
      if (seg.length == 0) {
        continue;
      }
      if (seg.artifact_offset != cursor) {
        return {StatusCode::FAILED_PRECONDITION, "piece LIP segments must densely cover view bytes"};
      }
      cursor = seg.artifact_offset + seg.length;
    }
    if (cursor != view_total) {
      return {StatusCode::FAILED_PRECONDITION, "piece LIP segments must densely cover view bytes"};
    }
  }

  auto stable_index_or = store::loader::rebuild_stable_canonical_index(meta.index_data, meta.device_id);
  if (!stable_index_or.ok()) {
    return to_grpc_status(stable_index_or.status());
  }
  const std::string stable_index_json = std::move(*stable_index_or);

  std::string index_key_hex = meta.index_key_hex;
  if (index_key_hex.empty()) {
    const auto digest = common::sha256_digest_bytes(
        absl::Span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(stable_index_json.data()), stable_index_json.size()));
    index_key_hex.reserve(digest.size() * 2);
    for (uint8_t byte : digest) {
      absl::StrAppendFormat(&index_key_hex, "%02x", byte);
    }
  }
  auto index_mh_or = common::compute_index_multihash(std::optional<std::string>(stable_index_json), index_key_hex);
  if (!index_mh_or.ok()) {
    return to_grpc_status(index_mh_or.status());
  }

  // Build a seekable view source over the leased view ByteSpace.
  absl::flat_hash_map<std::string, const RegisterStorageMeta*> storage_by_id;
  storage_by_id.reserve(storage_entries.size());
  for (const auto& s : storage_entries) {
    storage_by_id.emplace(s.storage_id, &s);
  }

  RegionPinGuard pin_guard(dep.regions);
  absl::flat_hash_map<std::string, std::unique_ptr<cuda::IpcMapping>> mapping_cache;
  mapping_cache.reserve(storage_entries.size());

  std::vector<LipDenseViewSource::Segment> view_segments;
  view_segments.reserve(lease_vec.size());
  for (const auto& seg : lease_vec) {
    auto it = storage_by_id.find(seg.storage_id);
    if (it == storage_by_id.end()) {
      return {StatusCode::INVALID_ARGUMENT, "lease segment references unknown storage_id"};
    }
    const RegisterStorageMeta* storage = it->second;
    if (storage->device_id != meta.device_id) {
      return {StatusCode::FAILED_PRECONDITION, "storage device_id mismatch for piece LIP registration"};
    }
    if (seg.length != storage->storage_length) {
      return {StatusCode::INVALID_ARGUMENT, "segment length does not match storage_length"};
    }
    if (seg.storage_offset != 0) {
      return {StatusCode::INVALID_ARGUMENT, "segment storage_offset must be 0 for full-storage registrations"};
    }
    if (seg.artifact_offset > view_total || seg.length > view_total || seg.artifact_offset + seg.length > view_total) {
      return {StatusCode::OUT_OF_RANGE, "lease segment view range out of bounds"};
    }
    auto map_or = get_or_open_mapping_for_storage(*storage, mapping_cache, pin_guard, dep.regions, meta.owner_pid);
    if (!map_or.ok()) {
      return to_grpc_status(map_or.status());
    }
    LipDenseViewSource::Segment view_segment;
    view_segment.device_id = storage->device_id;
    view_segment.mapping = *map_or;
    view_segment.base_offset = storage->mapping_base_offset + seg.storage_offset;
    view_segment.view_offset = seg.artifact_offset;
    view_segment.length = seg.length;
    view_segments.push_back(view_segment);
  }
  LipDenseViewSource view_source(std::move(view_segments), view_total);

  size_t leaf_chunk_bytes = dep.engine.get_artifact_chunk_bytes();
  if (leaf_chunk_bytes == 0) {
    leaf_chunk_bytes = static_cast<size_t>(4ULL * 1024 * 1024);
  }

  auto view_hash_or =
      store::loader::verification::compute_view_tree_hash_and_leaves(view_source, view_total, leaf_chunk_bytes);
  if (!view_hash_or.ok()) {
    return to_grpc_status(view_hash_or.status());
  }
  const std::string view_data_hash = view_hash_or->multihash;

  std::vector<tensorcast::global_store::v1::LeafWrite> leaf_writes;
  leaf_writes.reserve(view_hash_or->leaf_digests.size());
  for (size_t idx = 0; idx < view_hash_or->leaf_digests.size(); ++idx) {
    tensorcast::global_store::v1::LeafWrite leaf;
    auto* hash_space = leaf.mutable_hash_space();
    hash_space->mutable_byte_space()->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
    hash_space->mutable_byte_space()->set_id(meta.view_id);
    leaf.set_leaf_idx(static_cast<uint64_t>(idx));
    const auto& digest = view_hash_or->leaf_digests[idx];
    leaf.set_digest(digest.data(), static_cast<int>(digest.size()));
    leaf_writes.push_back(std::move(leaf));
  }

  const std::vector<store::view::CanonicalRange> canonical_ranges = canonical_ranges_from_write_plan(view_plan.write);
  uint64_t covered_bytes = 0;
  for (const auto& range : canonical_ranges) {
    covered_bytes += range.length;
  }

  std::vector<tensorcast::global_store::v1::PieceProofDigestWrite> proof_digests;
  auto intervals_or = parse_tensor_intervals(stable_index_json);
  if (!intervals_or.ok()) {
    return to_grpc_status(intervals_or.status());
  }
  uint64_t canonical_size_bytes = meta.view_canonical_size_bytes;
  if (canonical_size_bytes == 0) {
    for (const auto& interval : *intervals_or) {
      canonical_size_bytes = std::max(canonical_size_bytes, interval.offset + interval.size_bytes);
    }
  }
  if (canonical_size_bytes > 0 && covered_bytes > canonical_size_bytes) {
    return {StatusCode::FAILED_PRECONDITION, "canonical_bytes_covered exceeds canonical_size_bytes"};
  }

  std::unordered_map<std::string, uint64_t> transpose_view_offsets;
  std::unordered_map<std::string, store::loader::TensorTransformPlan> transpose_inverse_plans;
  if (view_plan.forward.transform.requires_materialization) {
    for (const auto& tensor_plan : view_plan.forward.transform.tensors) {
      transpose_view_offsets.emplace(tensor_plan.tensor_name, tensor_plan.dst_offset);
    }
    for (const auto& tensor_plan : view_plan.inverse_transform.tensors) {
      transpose_inverse_plans.emplace(tensor_plan.tensor_name, tensor_plan);
    }
  }

  for (const auto& interval : *intervals_or) {
    if (interval.size_bytes == 0) {
      continue;
    }
    if (!ranges_cover_interval(canonical_ranges, interval.offset, interval.size_bytes)) {
      continue;
    }
    const auto inverse_it = transpose_inverse_plans.find(interval.tensor_name);
    if (inverse_it != transpose_inverse_plans.end()) {
      const auto offset_it = transpose_view_offsets.find(interval.tensor_name);
      if (offset_it == transpose_view_offsets.end()) {
        return {StatusCode::FAILED_PRECONDITION, "missing transpose tensor dst_offset for proof digests"};
      }
      const uint64_t tensor_view_offset = offset_it->second;
      if (tensor_view_offset + interval.size_bytes > view_total) {
        return {StatusCode::OUT_OF_RANGE, "transpose tensor view range exceeds view buffer size"};
      }
      if (interval.size_bytes > std::numeric_limits<size_t>::max()) {
        return {StatusCode::OUT_OF_RANGE, "transpose tensor exceeds host memory limits"};
      }

      std::vector<uint8_t> view_bytes(static_cast<size_t>(interval.size_bytes));
      auto read_or = view_source.read_at(tensor_view_offset, view_bytes.data(), view_bytes.size());
      if (!read_or.ok()) {
        return to_grpc_status(read_or.status());
      }
      if (*read_or != view_bytes.size()) {
        return {StatusCode::OUT_OF_RANGE, "failed to read full transpose tensor bytes from view source"};
      }

      std::vector<uint8_t> canonical_bytes = view_bytes;
      store::loader::ViewWritePlan write_plan;
      store::loader::ViewWritePlan::Chunk write_chunk;
      write_chunk.canonical_offset = 0;
      write_chunk.view_offset = 0;
      write_chunk.length = interval.size_bytes;
      write_chunk.segment_aligned = false;
      write_plan.chunks.push_back(std::move(write_chunk));

      store::loader::TransformPlan inverse_transform;
      inverse_transform.requires_materialization = true;
      store::loader::TensorTransformPlan tensor_transform = inverse_it->second;
      tensor_transform.dst_offset = 0;
      tensor_transform.canonical_offset = 0;
      tensor_transform.storage_offset_elements = 0;
      inverse_transform.tensors.push_back(std::move(tensor_transform));

      store::loader::ViewIngestExecutor executor(
          std::move(write_plan),
          std::move(inverse_transform),
          store::loader::ViewIngestExecutor::IngestTarget::kCanonical);
      absl::Status ingest_status = executor.ingest_chunk(
          /*view_offset=*/0,
          absl::Span<const std::byte>(reinterpret_cast<const std::byte*>(view_bytes.data()), view_bytes.size()),
          tensorcast::common::memory::MemoryLocation::CPU,
          canonical_bytes.data(),
          /*device_id=*/-1);
      if (!ingest_status.ok()) {
        return to_grpc_status(ingest_status);
      }
      absl::Status finalize_status =
          executor.finalize(tensorcast::common::memory::MemoryLocation::CPU, canonical_bytes.data(), /*device_id=*/-1);
      if (!finalize_status.ok()) {
        return to_grpc_status(finalize_status);
      }

      const uint64_t expected_chunks = (interval.size_bytes + kProofChunkBytesV1 - 1) / kProofChunkBytesV1;
      for (uint64_t proof_chunk_idx = 0; proof_chunk_idx < expected_chunks; ++proof_chunk_idx) {
        const uint64_t local_start = proof_chunk_idx * kProofChunkBytesV1;
        const uint64_t local_end = std::min<uint64_t>(interval.size_bytes, local_start + kProofChunkBytesV1);
        if (local_end <= local_start) {
          continue;
        }
        if (local_end > std::numeric_limits<size_t>::max()) {
          return {StatusCode::OUT_OF_RANGE, "proof chunk exceeds host memory limits"};
        }
        const size_t chunk_bytes = static_cast<size_t>(local_end - local_start);
        std::vector<uint8_t> digest =
            common::sha256_digest_bytes(absl::Span<const uint8_t>(canonical_bytes.data() + local_start, chunk_bytes));
        if (digest.size() != 32) {
          return {StatusCode::INTERNAL, "sha256 digest size mismatch"};
        }
        tensorcast::global_store::v1::PieceProofDigestWrite proof;
        proof.set_view_id(meta.view_id);
        proof.set_tensor_name(interval.tensor_name);
        proof.set_proof_schema_version(std::string(kProofSchemaV1));
        proof.set_proof_chunk_idx(proof_chunk_idx);
        proof.set_digest(digest.data(), static_cast<int>(digest.size()));
        proof_digests.push_back(std::move(proof));
      }
      continue;
    }

    auto spans_or = canonical_spans_for_tensor(view_plan.write, interval.offset, interval.size_bytes, view_total);
    if (!spans_or.ok()) {
      return to_grpc_status(spans_or.status());
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
        return {StatusCode::OUT_OF_RANGE, "proof chunk exceeds host memory limits"};
      }
      std::vector<uint8_t> buffer(static_cast<size_t>(abs_end - abs_start));
      uint64_t cursor = abs_start;
      while (cursor < abs_end) {
        while (span_idx < spans.size() && spans[span_idx].canonical_offset + spans[span_idx].length <= cursor) {
          ++span_idx;
        }
        if (span_idx >= spans.size()) {
          return {StatusCode::FAILED_PRECONDITION, "missing canonical span while computing proof digests"};
        }
        const auto& span = spans[span_idx];
        if (span.canonical_offset > cursor) {
          return {StatusCode::FAILED_PRECONDITION, "canonical span gap while computing proof digests"};
        }
        const uint64_t take_end = std::min<uint64_t>(abs_end, span.canonical_offset + span.length);
        const size_t take = static_cast<size_t>(take_end - cursor);
        const uint64_t src_view_offset = span.view_offset + (cursor - span.canonical_offset);
        auto dst = absl::MakeSpan(buffer).subspan(static_cast<size_t>(cursor - abs_start), take);
        auto read_or = view_source.read_at(src_view_offset, dst.data(), dst.size());
        if (!read_or.ok()) {
          return to_grpc_status(read_or.status());
        }
        if (*read_or != dst.size()) {
          return {StatusCode::OUT_OF_RANGE, "failed to read full proof span from view source"};
        }
        cursor = take_end;
      }
      std::vector<uint8_t> digest =
          common::sha256_digest_bytes(absl::Span<const uint8_t>(buffer.data(), buffer.size()));
      if (digest.size() != 32) {
        return {StatusCode::INTERNAL, "sha256 digest size mismatch"};
      }
      tensorcast::global_store::v1::PieceProofDigestWrite proof;
      proof.set_view_id(meta.view_id);
      proof.set_tensor_name(interval.tensor_name);
      proof.set_proof_schema_version(std::string(kProofSchemaV1));
      proof.set_proof_chunk_idx(proof_chunk_idx);
      proof.set_digest(digest.data(), static_cast<int>(digest.size()));
      proof_digests.push_back(std::move(proof));
    }
  }

  struct LipRollback {
    LipManager* lip{nullptr};
    std::string registration_id;
    bool active{true};

    ~LipRollback() {
      if (!active || lip == nullptr) {
        return;
      }
      absl::Status st = lip->revoke_by_registration_id(registration_id);
      if (!st.ok()) {
        LOG(WARNING) << "LipRollback: revoke_by_registration_id failed for id=" << registration_id << ": " << st;
      }
    }

    void release() {
      active = false;
    }
  } lip_rollback{.lip = &dep.lip, .registration_id = registration_id};

  auto lease_or = dep.lip.commit_routable_view_lease_in_place(
      registration_id,
      meta.client_artifact_id,
      meta.view_id,
      meta.device_id,
      meta.owner_pid,
      meta.ttl_ms,
      meta.epoch,
      view_total,
      std::move(lease_vec),
      std::move(storage_entries));
  if (!lease_or.ok()) {
    lip_rollback.release();
    return to_grpc_status(lease_or.status());
  }

  std::string worker_id = dep.identity->worker_id();
  if (worker_id.empty()) {
    worker_id = "local";
  }
  const store::DeviceKey device = store::DeviceRegistry::instance().gpu_key(meta.device_id);

  auto replica_id_or = dep.global_store_client->register_memory_replica(
      meta.client_artifact_id,
      worker_id,
      device,
      view_total,
      index_key_hex,
      lease_or->remote_memory_keys,
      lease_or->buffer_sizes,
      stable_index_json,
      /*encoding=*/"json",
      /*schema_version=*/"v3",
      /*max_concurrency=*/1,
      /*verification_json=*/std::nullopt,
      std::optional<std::string_view>(meta.view_id));
  if (!replica_id_or.ok()) {
    return to_grpc_status(replica_id_or.status());
  }
  const std::string replica_id = *replica_id_or;
  dep.lip.attach_replica_id(registration_id, replica_id);

  struct ReplicaRollback {
    store::components::IGlobalStoreClient* client{nullptr};
    std::string artifact_id;
    std::string replica_id;
    bool active{true};

    ~ReplicaRollback() {
      if (!active || client == nullptr || replica_id.empty()) {
        return;
      }
      if (!client->is_connected()) {
        return;
      }
      absl::Status st = client->unregister_replica(artifact_id, replica_id);
      if (!st.ok()) {
        LOG(WARNING) << "ReplicaRollback: unregister_replica failed for artifact_id=" << artifact_id
                     << " replica_id=" << replica_id << ": " << st;
      }
    }

    void release() {
      active = false;
    }
  } replica_rollback{
      .client = dep.global_store_client.get(), .artifact_id = meta.client_artifact_id, .replica_id = replica_id};

  store::components::ViewStateUpdate update;
  update.artifact_id = meta.client_artifact_id;
  update.view_id = meta.view_id;
  update.view_spec_json = meta.view_spec_json;
  update.view_size_bytes = view_total;
  update.view_data_hash = view_data_hash;
  update.mark_verified = true;
  update.canonical_size_bytes = canonical_size_bytes;
  update.canonical_bytes_covered = covered_bytes;
  update.canonical_ranges.reserve(canonical_ranges.size());
  for (const auto& range : canonical_ranges) {
    store::components::CanonicalRange canonical_range;
    canonical_range.offset = range.offset;
    canonical_range.length = range.length;
    update.canonical_ranges.push_back(std::move(canonical_range));
  }
  update.leaf_writes = std::move(leaf_writes);
  update.proof_digests = std::move(proof_digests);
  absl::Status update_status = dep.global_store_client->update_artifact_view_state(update);
  if (!update_status.ok()) {
    return to_grpc_status(update_status);
  }

  // Publish response metadata.
  auto* desc = resp.mutable_artifact_descriptor();
  desc->set_artifact_id(meta.client_artifact_id);
  desc->set_index_multihash(*index_mh_or);
  desc->set_schema_version("v3");
  desc->set_encoding("json");
  desc->set_total_size(view_total);
  desc->set_id_kind(ToProtoKind(tensorcast::common::ArtifactIdKind::kCgid));
  resp.set_existed(false);
  resp.set_view_id(meta.view_id);
  if (!view_plan.forward.view_index_json.empty()) {
    resp.set_view_index_json(view_plan.forward.view_index_json);
  }
  if (!view_data_hash.empty()) {
    resp.set_view_data_hash(view_data_hash);
  }
  for (const auto& range : canonical_ranges) {
    auto* response_range = resp.add_canonical_ranges();
    response_range->set_offset(range.offset);
    response_range->set_length(range.length);
  }
  resp.set_allow_partial(true);

  auto* local_stable = resp.mutable_local_stable_tier();
  auto local_stable_status = apply_local_stable_tier(
      dep.engine,
      dep.lip,
      dep.regions,
      meta,
      meta.client_artifact_id,
      tensorcast::common::ArtifactIdKind::kCgid,
      view_total,
      local_stable,
      []() {});
  if (!local_stable_status.ok()) {
    return to_grpc_status(local_stable_status);
  }

  lip_rollback.release();
  replica_rollback.release();
  EraseRegistrationRegionRefs(dep.reg, dep.regions, registration_id);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status validate_piece_assembly_not_sealed(
    RegistrationController::Dep& dep,
    const std::string& registration_id,
    const RegistrationManager::RegMeta& meta) {
  if (meta.client_artifact_id.empty()) {
    return {StatusCode::INVALID_ARGUMENT, "piece registration requires client_artifact_id (cgid)"};
  }
  auto sealed_or = is_piece_assembly_sealed(dep.global_store_client.get(), meta.client_artifact_id);
  if (!sealed_or.ok()) {
    return to_grpc_status(sealed_or.status());
  }
  if (*sealed_or) {
    absl::Status abort_status = dep.engine.abort_registered_artifact(registration_id);
    if (!abort_status.ok()) {
      LOG(WARNING) << "abort_registered_artifact failed after sealed binding: " << abort_status;
    }
    EraseRegistrationRegionRefs(dep.reg, dep.regions, registration_id);
    return {StatusCode::FAILED_PRECONDITION, "assembly is already sealed; new pieces are not allowed"};
  }
  return Status::OK;
}

grpc::Status commit_lease_registration(
    RegistrationController::Dep& dep,
    RpcContext& rctx,
    const std::string& registration_id,
    const RegistrationManager::RegMeta& meta,
    v2::CommitRegisteredArtifactResponse& resp) {
  // Only LIP (in_place=true) is supported in current release.
  if (!meta.lease_in_place) {
    return {StatusCode::UNIMPLEMENTED, "vram_leased (in_place=false) is not implemented; set lease_in_place=true"};
  }
  if (meta.expiry.time_since_epoch().count() > 0 && std::chrono::steady_clock::now() > meta.expiry) {
    EraseRegistrationRegionRefs(dep.reg, dep.regions, registration_id);
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_ttl_expired_commit_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_ttl_expired_commit_total unavailable";
    }
    return {StatusCode::DEADLINE_EXCEEDED, "registration expired (TTL)"};
  }

  // Always commit lease in-place: do not allocate destination GPU memory.
  auto lease_vec = dep.reg.get_lease_segments(registration_id);
  if (lease_vec.empty()) {
    return {StatusCode::FAILED_PRECONDITION, "no lease segments fed"};
  }

  auto storage_entries = dep.reg.get_storage_entries(registration_id);
  if (storage_entries.empty()) {
    return {StatusCode::FAILED_PRECONDITION, "registration missing storage_entries payload"};
  }

  if (meta.view_registration && meta.view_registration_kind == store::StoreEngine::ViewRegistrationKind::kPiece) {
    return commit_piece_view_registration(
        dep, rctx, registration_id, meta, std::move(lease_vec), std::move(storage_entries), resp);
  }

  auto alias_vec = dep.reg.get_tensor_aliases(registration_id);
  if (alias_vec.empty()) {
    return {StatusCode::FAILED_PRECONDITION, "registration missing tensor_aliases payload"};
  }
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto storage_counter = meter->CreateDoubleCounter("tc_register_storage_count");
    static auto alias_counter = meter->CreateDoubleCounter("tc_register_tensor_count");
    storage_counter->Add(static_cast<double>(storage_entries.size()));
    alias_counter->Add(static_cast<double>(alias_vec.size()));
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_storage_count/tc_register_tensor_count unavailable";
  }

  auto out_or = dep.lip.commit_lease_in_place(
      registration_id,
      meta.device_id,
      meta.owner_pid,
      meta.ttl_ms,
      meta.epoch,
      meta.total_size,
      meta.id_kind,
      meta.client_artifact_id,
      meta.index_data,
      meta.index_key_hex,
      std::move(lease_vec),
      std::move(storage_entries),
      std::move(alias_vec));
  if (!out_or.ok()) {
    if (absl::IsAlreadyExists(out_or.status())) {
      try {
        static auto meter =
            opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
        static auto counter = meter->CreateDoubleCounter("tc_register_commit_denied_total");
        counter->Add(1.0);
      } catch (...) {
        VLOG(1) << "metrics counter tc_register_commit_denied_total unavailable";
      }
    }
    return to_grpc_status(out_or.status());
  }

  const auto& out = *out_or;
  auto* desc = resp.mutable_artifact_descriptor();
  desc->set_artifact_id(out.artifact_id);
  desc->set_index_multihash(out.index_multihash);
  desc->set_data_multihash(out.data_multihash);
  desc->set_schema_version(out.schema_version);
  desc->set_encoding(out.encoding);
  desc->set_total_size(out.total_size);
  desc->set_id_kind(ToProtoKind(out.id_kind));

  auto* local_stable = resp.mutable_local_stable_tier();
  auto local_stable_status = apply_local_stable_tier(
      dep.engine, dep.lip, dep.regions, meta, out.artifact_id, out.id_kind, out.total_size, local_stable, [&]() {
        (void)dep.lip.revoke_by_registration_id(registration_id);
        EraseRegistrationRegionRefs(dep.reg, dep.regions, registration_id);
      });
  if (!local_stable_status.ok()) {
    return to_grpc_status(local_stable_status);
  }

  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_register_commit_lip_total");
    counter->Add(1.0);
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_commit_lip_total unavailable";
  }
  // Create CommitLease for VRAM_LEASED in-place ownership (device-unique).
  if (dep.lifecycle) {
    SessionLifecycleManager::CommitSubject subj{.artifact_id = out.artifact_id, .device_id = meta.device_id};
    auto lid_or = dep.lifecycle->create_commit_lease(subj, meta.owner_pid);
    if (!lid_or.ok()) {
      LOG(WARNING) << "create_commit_lease failed: artifact_id=" << out.artifact_id << " dev=" << meta.device_id << ": "
                   << lid_or.status();
      try {
        static auto meter =
            opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
        static auto ctr = meter->CreateDoubleCounter("tc_lease_create_failed_total");
        ctr->Add(1.0);
      } catch (...) {
      }
    }
  }
  LOG(INFO) << "Registered memory replica: " << out.artifact_id
            << " plan=vram_leased(in_place) device=gpu:" << meta.device_id << " size=" << out.total_size << "B";
  EraseRegistrationRegionRefs(dep.reg, dep.regions, registration_id);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status commit_engine_registration(
    RegistrationController::Dep& dep,
    RpcContext& rctx,
    const std::string& registration_id,
    RegistrationManager::RegMeta meta,
    v2::CommitRegisteredArtifactResponse& resp) {
  auto commit_or = dep.engine.commit_registered_artifact(registration_id);
  if (!commit_or.ok()) {
    return to_grpc_status(commit_or.status());
  }
  const auto& out = commit_or.value();
  auto* desc = resp.mutable_artifact_descriptor();
  desc->set_artifact_id(out.artifact_id);
  desc->set_index_multihash(out.index_multihash);
  desc->set_data_multihash(out.data_multihash);
  desc->set_schema_version(out.schema_version);
  desc->set_encoding(out.encoding);
  desc->set_total_size(out.size_bytes);
  desc->set_id_kind(ToProtoKind(out.id_kind));
  resp.set_existed(out.existed);
  if (out.view_id.has_value()) {
    resp.set_view_id(*out.view_id);
  }
  if (out.view_index_json.has_value()) {
    resp.set_view_index_json(*out.view_index_json);
  }
  if (out.view_data_multihash.has_value()) {
    resp.set_view_data_hash(*out.view_data_multihash);
  }
  for (const auto& range : out.canonical_ranges) {
    auto* r = resp.add_canonical_ranges();
    r->set_offset(range.offset);
    r->set_length(range.length);
  }
  resp.set_allow_partial(out.registration_kind == store::StoreEngine::ViewRegistrationKind::kPiece);
  if (out.view_id.has_value()) {
    meta.view_registration = true;
    meta.view_id = *out.view_id;
    meta.view_canonical_ranges = out.canonical_ranges;
    meta.view_data_multihash = out.view_data_multihash;
    meta.view_registration_kind = out.registration_kind;
    uint64_t covered_bytes = 0;
    for (const auto& range : out.canonical_ranges) {
      covered_bytes += range.length;
    }
    uint64_t canonical_size_bytes =
        meta.view_canonical_size_bytes > 0 ? meta.view_canonical_size_bytes : out.size_bytes;
    if (!out.existed && covered_bytes < canonical_size_bytes) {
      record_view_partial_metric();
      LOG(INFO) << "View registration partial coverage: artifact_id=" << out.artifact_id << " view_id=" << *out.view_id
                << " covered_bytes=" << covered_bytes << " canonical_bytes=" << canonical_size_bytes
                << " ingested_view_bytes=" << meta.view_ingested_bytes;
    } else {
      VLOG(1) << "View registration coverage: artifact_id=" << out.artifact_id << " view_id=" << *out.view_id
              << " covered_bytes=" << covered_bytes << " canonical_bytes=" << canonical_size_bytes
              << " ingested_view_bytes=" << meta.view_ingested_bytes;
    }
  }

  auto* local_stable = resp.mutable_local_stable_tier();
  auto local_stable_status = apply_local_stable_tier(
      dep.engine, dep.lip, dep.regions, meta, out.artifact_id, out.id_kind, out.size_bytes, local_stable, [&]() {
        if (!out.existed) {
          store::loading::ReplicaKey base_key{.artifact_id = out.artifact_id, .device = out.device, .replica = 0};
          (void)dep.engine.unload_replica(base_key);
          auto unreg_status = dep.engine.unregister_replica_from_global_store(out.artifact_id, out.device.ordinal);
          if (!unreg_status.ok()) {
            LOG(WARNING) << "unregister_replica_from_global_store failed after must local stable failure: artifact_id="
                         << out.artifact_id << " dev=" << out.device.ordinal << ": " << unreg_status;
          }
        }
        dep.reg.erase_meta(registration_id);
      });
  if (!local_stable_status.ok()) {
    return to_grpc_status(local_stable_status);
  }

  if (out.existed) {
    store::loading::ReplicaKey key{.artifact_id = out.artifact_id, .device = out.device, .replica = 0};
    dep.refs.add_ref(key, meta.owner_pid);
    if (dep.lifecycle && meta.ttl_ms > 0 && out.device.type == DeviceType::GPU) {
      auto lease_or = dep.lifecycle->create_ttl_use_lease(key, meta.owner_pid, absl::Milliseconds(meta.ttl_ms));
      if (lease_or.ok()) {
        meta.use_lease_id = *lease_or;
      } else {
        LOG(ERROR) << "failed to create ttl use lease: " << lease_or.status();
      }
    }
    meta.device_id = out.device.ordinal;
    meta.joined_existing = true;
    meta.artifact_id_mi2 = out.artifact_id;
    dep.reg.set_meta(registration_id, meta);
  } else {
    dep.reg.erase_meta(registration_id);
  }
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto coalesced_counter = meter->CreateDoubleCounter("tc_register_commit_coalesced_total");
    static auto stable_counter = meter->CreateDoubleCounter("tc_register_commit_stable_dram_total");
    if (meta.plan == RegistrationManager::RegPlan::STABLE_DRAM) {
      stable_counter->Add(1.0);
    } else {
      coalesced_counter->Add(1.0);
    }
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_commit_(stable_dram|coalesced)_total unavailable";
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status apply_begin_view_registration(
    const v2::BeginRegisterArtifactRequest& req,
    RegistrationManager::RegMeta& meta,
    store::StoreEngine::ArtifactRegistration& reg) {
  if (!req.has_view()) {
    return Status::OK;
  }
  if (req.view().allow_partial()) {
    return {
        StatusCode::INVALID_ARGUMENT,
        "allow_partial is deprecated; use registration_kind=VIEW_REGISTRATION_KIND_PIECE"};
  }
  if (!req.has_tensor_index_data()) {
    return {StatusCode::INVALID_ARGUMENT, "view registration requires tensor_index_data"};
  }
  if (req.view().spec().tensors().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "view registration requires non-empty view spec"};
  }
  auto placement = ToPlacement(req.view().placement());
  if (placement == store::StoreEngine::ViewPlacement::kUnspecified) {
    return {StatusCode::INVALID_ARGUMENT, "view placement must be specified"};
  }
  auto registration_kind = ToRegistrationKind(req.view().registration_kind());
  if (registration_kind == store::StoreEngine::ViewRegistrationKind::kUnspecified) {
    return {StatusCode::INVALID_ARGUMENT, "view.registration_kind must be specified"};
  }
  if (registration_kind == store::StoreEngine::ViewRegistrationKind::kPiece && req.view().ranges_size() > 0) {
    return {StatusCode::INVALID_ARGUMENT, "view.ranges not supported for piece registration"};
  }
  auto spec_or = convert_view_spec(req.view().spec());
  if (!spec_or.ok()) {
    return to_grpc_status(spec_or.status());
  }
  store::StoreEngine::ViewRegistration view_reg;
  auto view_id_or = compute_view_id_from_spec(req.view().spec(), meta.index_data);
  if (!view_id_or.ok()) {
    return to_grpc_status(view_id_or.status());
  }
  std::string resolved_view_id = req.view().view_id();
  if (resolved_view_id.empty()) {
    resolved_view_id = *view_id_or;
  } else if (resolved_view_id != *view_id_or) {
    return {StatusCode::INVALID_ARGUMENT, "view_id does not match view spec"};
  }
  view_reg.view_id = resolved_view_id;
  view_reg.spec = std::move(*spec_or);
  meta.view_spec_json = store::view::build_view_spec_json(view_reg.spec);
  view_reg.placement = placement;
  view_reg.canonical_size_bytes = req.view().canonical_size_bytes();
  view_reg.registration_kind = registration_kind;
  reg.view = view_reg;
  meta.view_registration = true;
  meta.view_placement = placement;
  meta.view_id = view_reg.view_id;
  meta.view_registration_kind = registration_kind;
  meta.view_canonical_size_bytes = view_reg.canonical_size_bytes;
  return Status::OK;
}

grpc::Status begin_coalesced_or_stable_registration(
    RegistrationController::Dep& dep,
    RpcContext& rctx,
    const v2::BeginRegisterArtifactRequest& req,
    RegistrationManager::RegPlan plan,
    RegistrationManager::RegMeta& meta,
    store::StoreEngine::ArtifactRegistration& reg,
    v2::BeginRegisterArtifactResponse& resp) {
  if (req.has_tensor_index_key()) {
    reg.tensor_index_key = req.tensor_index_key();
  }
  if (req.has_tensor_index_data()) {
    reg.tensor_index_data = meta.index_data;
    reg.schema_version = req.tensor_index_data().schema_version();
    reg.encoding = req.tensor_index_data().encoding();
  }
  if (!meta.client_artifact_id.empty()) {
    reg.client_artifact_id = meta.client_artifact_id;
  }
  if (plan == RegistrationManager::RegPlan::STABLE_DRAM) {
    reg.plan = store::runtime::metadata::RegistrationPlan::kStableDram;
    reg.stable_dram.stage_on_gpu = meta.stage_on_gpu;
    reg.stable_dram.release_gpu_on_commit = meta.release_gpu_on_commit;
    if (!meta.stage_on_gpu) {
      return {StatusCode::UNIMPLEMENTED, "dram_stable with stage_on_gpu=false is not implemented"};
    }
  }
  auto begin_or = dep.engine.begin_register_artifact(reg);
  if (!begin_or.ok()) {
    return to_grpc_status(begin_or.status());
  }
  const auto& out = begin_or.value();
  resp.set_registration_id(out.registration_id);
  auto handle_view = out.cuda_ipc_handle_bytes.as_string_view();
  if (plan == RegistrationManager::RegPlan::STABLE_DRAM) {
    auto* hs = resp.mutable_stable_dram();
    hs->set_staging_cuda_ipc_handle(handle_view.data(), handle_view.size());
  } else {
    auto* hs = resp.mutable_coalesced();
    hs->set_daemon_ipc_handle(handle_view.data(), handle_view.size());
  }
  resp.set_device_id(out.device_id);
  resp.set_total_size(out.size_bytes);
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto coalesced_counter = meter->CreateDoubleCounter("tc_register_begin_coalesced_total");
    static auto stable_counter = meter->CreateDoubleCounter("tc_register_begin_stable_dram_total");
    if (plan == RegistrationManager::RegPlan::STABLE_DRAM) {
      stable_counter->Add(1.0);
    } else {
      coalesced_counter->Add(1.0);
    }
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_begin_(stable_dram|coalesced)_total unavailable";
  }
  dep.reg.set_meta(out.registration_id, meta);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status begin_lease_registration(
    RegistrationController::Dep& dep,
    RpcContext& rctx,
    const RegistrationManager::RegMeta& meta,
    v2::BeginRegisterArtifactResponse& resp) {
  if (!meta.lease_in_place) {
    return {StatusCode::UNIMPLEMENTED, "vram_leased (in_place=false) is not implemented; set lease_in_place=true"};
  }
  std::string reg_id = absl::StrCat("reg_", absl::ToUnixNanos(absl::Now()), "_", getpid());
  resp.set_registration_id(reg_id);
  resp.mutable_lease();
  resp.set_device_id(meta.device_id);
  resp.set_total_size(meta.total_size);
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_register_begin_lease_total");
    counter->Add(1.0);
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_begin_lease_total unavailable";
  }
  dep.reg.set_meta(reg_id, meta);
  rctx.mark_success();
  return Status::OK;
}

} // namespace

grpc::Status RegistrationController::begin(
    RpcContext& rctx,
    const v2::BeginRegisterArtifactRequest& req,
    v2::BeginRegisterArtifactResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.device.id", static_cast<int64_t>(req.device_id()));
  span->SetAttribute("tc.size.bytes", static_cast<int64_t>(req.total_size()));

  store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = absl::StrCat("mem_reg:", absl::ToUnixNanos(absl::Now()), ":", getpid());
  reg.device_id = req.device_id();
  reg.total_size_bytes = req.total_size();
  reg.enable_p2p = true;
  if (req.has_ttl_ms())
    reg.ttl_ms = req.ttl_ms();
  if (req.owner_pid() <= 0) {
    return {StatusCode::INVALID_ARGUMENT, "owner_pid is required (>0)"};
  }
  RegistrationManager::RegPlan plan = RegistrationManager::RegPlan::COALESCED;
  if (req.has_lease())
    plan = RegistrationManager::RegPlan::LEASE;
  if (req.has_stable_dram())
    plan = RegistrationManager::RegPlan::STABLE_DRAM;
  RegistrationManager::RegMeta meta;
  meta.plan = plan;
  meta.total_size = req.total_size();
  meta.device_id = req.device_id();
  meta.owner_pid = req.owner_pid();
  {
    auto policy_or = resolve_store_policy(req.has_policy() ? &req.policy() : nullptr);
    if (!policy_or.ok()) {
      return to_grpc_status(policy_or.status());
    }
    meta.resolved_policy = *policy_or;
    if (plan == RegistrationManager::RegPlan::STABLE_DRAM && policy_or->local_requirement == RequirementLevel::kMust) {
      reg.stable_cache_policy = stable_cache_policy_from_resolved(*policy_or);
    }
  }
  if (!req.client_artifact_id().empty()) {
    auto id_status = common::validate_client_generated_id(req.client_artifact_id());
    if (!id_status.ok()) {
      return {StatusCode::INVALID_ARGUMENT, std::string(id_status.message())};
    }
    meta.id_kind = common::ArtifactIdKind::kCgid;
    meta.client_artifact_id = req.client_artifact_id();
  } else {
    meta.id_kind = common::ArtifactIdKind::kMi2;
    meta.client_artifact_id.clear();
  }
  if (req.has_lease())
    meta.lease_in_place = req.lease().in_place();
  if (req.has_stable_dram()) {
    meta.stage_on_gpu = req.stable_dram().stage_on_gpu();
    meta.release_gpu_on_commit = req.stable_dram().release_gpu_on_commit();
  }
  if (req.has_ttl_ms() && req.ttl_ms() > 0) {
    meta.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(req.ttl_ms());
    meta.ttl_ms = static_cast<uint32_t>(req.ttl_ms());
  }
  if (req.has_tensor_index_key())
    meta.index_key_hex = req.tensor_index_key();
  else if (req.has_tensor_index_data())
    meta.index_data = std::string(req.tensor_index_data().data().begin(), req.tensor_index_data().data().end());

  auto view_status = apply_begin_view_registration(req, meta, reg);
  if (!view_status.ok()) {
    return view_status;
  }

  if (meta.view_registration && meta.view_registration_kind == store::StoreEngine::ViewRegistrationKind::kPiece) {
    if (meta.client_artifact_id.empty()) {
      return {StatusCode::INVALID_ARGUMENT, "piece registration requires client_artifact_id (cgid)"};
    }
    auto sealed_or = is_piece_assembly_sealed(d_.global_store_client.get(), meta.client_artifact_id);
    if (!sealed_or.ok()) {
      return to_grpc_status(sealed_or.status());
    }
    if (*sealed_or) {
      return {StatusCode::FAILED_PRECONDITION, "assembly is already sealed; new pieces are not allowed"};
    }
  }

  if (plan == RegistrationManager::RegPlan::COALESCED || plan == RegistrationManager::RegPlan::STABLE_DRAM) {
    return begin_coalesced_or_stable_registration(d_, rctx, req, plan, meta, reg, resp);
  }
  if (plan == RegistrationManager::RegPlan::LEASE) {
    return begin_lease_registration(d_, rctx, meta, resp);
  }
  return Status::OK;
}

grpc::Status RegistrationController::feed_stream(
    RpcContext& rctx,
    ::grpc::ServerReader<v2::FeedRegisterArtifactStreamRequest>& reader,
    v2::FeedRegisterArtifactStreamResponse& /*resp*/) {
  auto status = process_feed_requests(
      d_, [&](v2::FeedRegisterArtifactStreamRequest* next_req) { return reader.Read(next_req); }, "stream");
  if (!status.ok()) {
    return status;
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status RegistrationController::feed_vector(const std::vector<v2::FeedRegisterArtifactStreamRequest>& reqs) {
  size_t cursor = 0;
  return process_feed_requests(
      d_,
      [&](v2::FeedRegisterArtifactStreamRequest* next_req) {
        if (cursor >= reqs.size()) {
          return false;
        }
        *next_req = reqs[cursor++];
        return true;
      },
      "vector");
}

grpc::Status RegistrationController::keep_alive(
    RpcContext& rctx,
    const v2::KeepAliveRegisterArtifactRequest& req,
    v2::KeepAliveRegisterArtifactResponse& /*resp*/) {
  auto st = d_.reg.keepalive_precommit(
      req.registration_id(), req.owner_pid(), req.epoch(), req.ttl_ms() > 0 ? req.ttl_ms() : 0, d_.engine);
  if (!st.ok()) {
    if (!absl::IsNotFound(st)) {
      return to_grpc_status(st);
    }
    st = d_.lip.keepalive_lease(
        req.registration_id(), req.owner_pid(), req.epoch(), req.ttl_ms() > 0 ? req.ttl_ms() : 0);
    if (!st.ok()) {
      return to_grpc_status(st);
    }
  }

  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_register_keepalive_total");
    counter->Add(1.0);
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_keepalive_total unavailable";
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status RegistrationController::commit(
    RpcContext& rctx,
    const v2::CommitRegisteredArtifactRequest& req,
    v2::CommitRegisteredArtifactResponse& resp) {
  auto meta_opt = d_.reg.get_meta(req.registration_id());
  RegistrationManager::RegMeta meta;
  if (meta_opt.has_value()) {
    meta = *meta_opt;
    LOG(INFO) << "RegistrationController::commit: " << req.registration_id() << ",  plan=" << meta.plan
              << ", lease_in_place=" << meta.lease_in_place << ", expiry=" << meta.expiry.time_since_epoch().count();
  } else {
    LOG(INFO) << "RegistrationController::commit: " << req.registration_id() << " no meta";
  }

  if (meta.view_registration && meta.view_registration_kind == store::StoreEngine::ViewRegistrationKind::kPiece) {
    auto validate_status = validate_piece_assembly_not_sealed(d_, req.registration_id(), meta);
    if (!validate_status.ok()) {
      return validate_status;
    }
  }

  if (meta.plan == RegistrationManager::RegPlan::LEASE) {
    return commit_lease_registration(d_, rctx, req.registration_id(), meta, resp);
  }
  return commit_engine_registration(d_, rctx, req.registration_id(), meta, resp);
}

grpc::Status RegistrationController::abort(
    RpcContext& rctx,
    const v2::AbortRegisteredArtifactRequest& req,
    v2::AbortRegisteredArtifactResponse& /*resp*/) {
  auto st = d_.engine.abort_registered_artifact(req.registration_id());
  if (!st.ok())
    return to_grpc_status(st);
  EraseRegistrationRegionRefs(d_.reg, d_.regions, req.registration_id());
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_register_abort_total");
    counter->Add(1.0);
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_abort_total unavailable";
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status RegistrationController::revoke(
    RpcContext& rctx,
    const v2::RevokeRegisteredArtifactRequest& req,
    v2::RevokeRegisteredArtifactResponse& /*resp*/) {
  // Capture meta for potential joined-reference cleanup
  auto meta_opt = d_.reg.get_meta(req.registration_id());
  auto revoke_status = d_.lip.revoke_by_registration_id(req.registration_id());
  if (!revoke_status.ok()) {
    absl::Status abort_status = d_.engine.abort_registered_artifact(req.registration_id());
    if (!abort_status.ok()) {
      LOG(WARNING) << "revoke: abort_registered_artifact failed for id=" << req.registration_id() << ": "
                   << abort_status;
    }
  }
  EraseRegistrationRegionRefs(d_.reg, d_.regions, req.registration_id());

  if (meta_opt.has_value() && meta_opt->joined_existing) {
    const auto& m = *meta_opt;
    store::DeviceKey dev_key = store::DeviceRegistry::instance().gpu_key(m.device_id);
    store::loading::ReplicaKey key{.artifact_id = m.artifact_id_mi2, .device = dev_key, .replica = 0};
    // Release lifecycle UseLease precisely, if recorded
    bool lease_released = false;
    if (m.use_lease_id != 0) {
      d_.lifecycle->release_lease(static_cast<SessionLifecycleManager::LeaseId>(m.use_lease_id));
      lease_released = true;
    } else {
      // Fallback by subject+pid
      auto st = d_.lifecycle->release_use_lease(key, m.owner_pid);
      if (st.ok()) {
        lease_released = true;
      } else {
        LOG(WARNING) << "release_use_lease failed (revoke fallback): artifact_id=" << m.artifact_id_mi2
                     << " dev=" << m.device_id << ": " << st;
      }
    }
    if (!lease_released) {
      d_.refs.drop_ref(key, m.owner_pid);
    }
  }
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_register_revoke_total");
    counter->Add(1.0);
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_revoke_total unavailable";
  }
  rctx.mark_success();
  return Status::OK;
}

} // namespace tensorcast::daemon
