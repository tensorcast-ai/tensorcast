// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/local_stable_tier_service.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <unistd.h>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/cuda/cuda_ipc.h"
#include "core/cuda/device_guard.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/dataplane/sources/byte_range_map_builder.h"
#include "daemon/service/controllers/registration_storage_mapping_utils.h"
#include "daemon/state/store_policy_resolver.h"
#include "gsl/pointers"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

namespace {

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
  for (const auto& storage : storages) {
    storage_by_id.emplace(storage.storage_id, &storage);
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
    for (const auto& segment : opened) {
      if (off >= segment.dst && off < (segment.dst + segment.len)) {
        return &segment;
      }
      if (off < segment.dst) {
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
    LocalStableTierService::Dep d,
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

  auto cpu_key_or = find_loaded_cpu_replica(d.engine, artifact_id);
  if (cpu_key_or.ok()) {
    auto admit_or = d.engine.admit_stable_cache_policy(*cpu_key_or, *stable_policy_opt);
    if (!admit_or.ok()) {
      return admit_or.status();
    }
    return *admit_or;
  }
  if (!absl::IsNotFound(cpu_key_or.status())) {
    return cpu_key_or.status();
  }

  if (auto gpu_key_opt = find_loaded_gpu_replica(d.engine, artifact_id); gpu_key_opt.has_value()) {
    const int device_id = gpu_key_opt->device.ordinal;
    auto ptr_or = d.engine.get_replica_gpu_ptr(*gpu_key_opt);
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
        d.engine,
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
    auto lip_opt = d.lip.find_active_by_artifact_id(std::string(artifact_id));
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
        d.engine,
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
              d.regions,
              owner_pid);
        });
    if (!commit_or.ok()) {
      return commit_or.status();
    }
  }

  auto cpu_key_or2 = find_loaded_cpu_replica(d.engine, artifact_id);
  if (!cpu_key_or2.ok()) {
    return cpu_key_or2.status();
  }
  auto admit_or = d.engine.admit_stable_cache_policy(*cpu_key_or2, *stable_policy_opt);
  if (!admit_or.ok()) {
    return admit_or.status();
  }
  return *admit_or;
}

absl::StatusOr<ResolvedStorePolicy> resolve_effective_store_policy(const RegistrationManager::RegMeta& meta) {
  if (meta.resolved_policy.has_value()) {
    return *meta.resolved_policy;
  }
  return resolve_store_policy(nullptr);
}

} // namespace

LocalStableTierService::LocalStableTierService(Dep d) : d_(d) {}

absl::Status LocalStableTierService::apply_local_stable_tier(
    const RegistrationManager::RegMeta& meta,
    std::string_view artifact_id,
    tensorcast::common::ArtifactIdKind id_kind,
    uint64_t total_size,
    v2::LocalStableTierResult& local_stable,
    const std::function<void()>& on_must_failure_cleanup) const {
  auto resolved_or = resolve_effective_store_policy(meta);
  if (!resolved_or.ok()) {
    return resolved_or.status();
  }
  const ResolvedStorePolicy& resolved = *resolved_or;
  const char* op_label = local_stable_op_label(meta.plan);
  const char* requirement = requirement_level_label(resolved.local_requirement);

  if (meta.view_registration) {
    local_stable.set_status(v2::LOCAL_STABLE_TIER_STATUS_SKIPPED);
    local_stable.set_message("view registrations do not satisfy the local stable tier");
    record_local_stable_tier_metrics(op_label, "skipped", requirement, std::nullopt);
    return absl::OkStatus();
  }
  if (static_cast<int>(resolved.local_requirement) < static_cast<int>(RequirementLevel::kShould)) {
    local_stable.set_status(v2::LOCAL_STABLE_TIER_STATUS_SKIPPED);
    record_local_stable_tier_metrics(op_label, "skipped", requirement, std::nullopt);
    return absl::OkStatus();
  }

  const auto stable_policy_opt = stable_cache_policy_from_resolved(resolved);
  const auto local_stable_start = std::chrono::steady_clock::now();
  auto admit_or = ensure_local_stable_admission(
      d_, resolved, artifact_id, id_kind, total_size, meta.owner_pid, meta.index_data, meta.index_key_hex);
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
    local_stable.set_status(v2::LOCAL_STABLE_TIER_STATUS_DEGRADED);
    local_stable.set_message(message);
    record_local_stable_tier_metrics(op_label, "degraded", requirement, local_stable_seconds);
    return absl::OkStatus();
  }

  local_stable.set_status(v2::LOCAL_STABLE_TIER_STATUS_READY);
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

} // namespace tensorcast::daemon
