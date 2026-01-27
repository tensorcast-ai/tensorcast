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

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "core/cuda/cuda_ipc.h"
#include "core/cuda/device_guard.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/dataplane/sources/segment_plan_source.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "daemon/state/store_policy_resolver.h"
#include "daemon/util/status_utils.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"
#include "tensorcast/common/v1/common.pb.h"

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

absl::StatusOr<std::string> serialize_deterministic_proto(const google::protobuf::Message& message) {
  const size_t size = message.ByteSizeLong();
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return absl::OutOfRangeError("proto message too large for deterministic serialization");
  }
  std::string buffer;
  buffer.resize(size);
  google::protobuf::io::ArrayOutputStream stream(buffer.data(), static_cast<int>(size));
  google::protobuf::io::CodedOutputStream coded(&stream);
  coded.SetSerializationDeterministic(true);
  if (!message.SerializeToCodedStream(&coded) || coded.HadError()) {
    return absl::InternalError("deterministic proto serialization failed");
  }
  return buffer;
}

absl::StatusOr<std::string> compute_view_id_from_spec(
    const v2::ViewSpec& view_spec,
    std::string_view canonical_index_json) {
  auto index_mh_or = common::compute_index_multihash(std::optional<std::string>(canonical_index_json), "");
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }
  auto proto_bytes_or = serialize_deterministic_proto(view_spec);
  if (!proto_bytes_or.ok()) {
    return proto_bytes_or.status();
  }
  std::vector<uint8_t> buffer;
  buffer.reserve(proto_bytes_or->size() + index_mh_or->size());
  buffer.insert(buffer.end(), proto_bytes_or->begin(), proto_bytes_or->end());
  buffer.insert(buffer.end(), index_mh_or->begin(), index_mh_or->end());
  const std::vector<uint8_t> digest = common::sha256_digest_bytes(absl::Span<const uint8_t>(buffer));
  return common::multibase_multihash_sha256(digest);
}

absl::StatusOr<store::loader::ViewSpec> BuildViewSpecFromProto(const v2::ViewSpec& spec_proto) {
  store::loader::ViewSpec spec;
  for (const auto& [tensor_name, ops_proto] : spec_proto.tensors()) {
    store::loader::TensorViewOps tensor_ops;
    for (const auto& op_proto : ops_proto.ops()) {
      if (op_proto.has_narrow()) {
        const auto& narrow = op_proto.narrow();
        store::loader::NarrowOp narrow_op;
        narrow_op.dim = static_cast<int32_t>(narrow.dim());
        narrow_op.start = narrow.start();
        narrow_op.length = narrow.length();
        tensor_ops.ops.push_back(store::loader::ViewOp::Narrow(narrow_op));
      } else if (op_proto.has_transpose()) {
        const auto& transpose = op_proto.transpose();
        store::loader::TransposeOp transpose_op;
        transpose_op.dim0 = static_cast<int32_t>(transpose.dim0());
        transpose_op.dim1 = static_cast<int32_t>(transpose.dim1());
        tensor_ops.ops.push_back(store::loader::ViewOp::Transpose(transpose_op));
      } else {
        return absl::InvalidArgumentError("unsupported view operation in ViewSpec");
      }
    }
    spec.tensors.emplace(tensor_name, std::move(tensor_ops));
  }
  return spec;
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
    absl::Span<const store::loader::SegmentPiece> plan,
    uint64_t total_size) {
  cuda::CudaDeviceGuard guard(device_id);
  if (!guard.status().ok()) {
    return guard.status();
  }
  if (plan.empty()) {
    return cuda::memcpy(dst_dev, src_dev, static_cast<size_t>(total_size), cudaMemcpyDeviceToDevice);
  }
  for (const auto& p : plan) {
    if (p.length == 0) {
      continue;
    }
    if (p.kind == store::loader::SegmentPiece::PAD) {
      auto st = cuda::memset(static_cast<uint8_t*>(dst_dev.get()) + p.dst_offset, 0, static_cast<size_t>(p.length));
      if (!st.ok()) {
        return st;
      }
      continue;
    }
    auto st = cuda::memcpy(
        static_cast<uint8_t*>(dst_dev.get()) + p.dst_offset,
        static_cast<const uint8_t*>(src_dev.get()) + p.src_offset,
        static_cast<size_t>(p.length),
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

absl::Status copy_to_staging_from_lip_sources(
    int device_id,
    gsl::not_null<void*> dst_dev,
    absl::Span<const store::loader::SegmentPiece> plan,
    uint64_t total_size,
    absl::Span<const LeaseSegMeta> segments,
    absl::Span<const RegisterStorageMeta> storages,
    IpcRegionRegistry& regions,
    int owner_pid) {
  cuda::CudaDeviceGuard guard(device_id);
  if (!guard.status().ok()) {
    return guard.status();
  }
  if (plan.empty()) {
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

  for (const auto& p : plan) {
    if (p.length == 0) {
      continue;
    }
    if (p.dst_offset > total_size || p.length > total_size || (p.dst_offset + p.length) > total_size) {
      return absl::OutOfRangeError("segment plan dst range out of bounds");
    }
    if (p.kind == store::loader::SegmentPiece::PAD) {
      auto st = cuda::memset(static_cast<uint8_t*>(dst_dev.get()) + p.dst_offset, 0, static_cast<size_t>(p.length));
      if (!st.ok()) {
        return st;
      }
      continue;
    }
    uint64_t remaining = p.length;
    uint64_t src_off = p.src_offset;
    uint64_t dst_off = p.dst_offset;
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
    std::vector<store::loader::SegmentPiece> plan;
    if (!canonical_index_json.empty()) {
      auto plan_or = store::loader::build_segment_plan_from_canonical_index_json(
          canonical_index_json, total_size, /*align_bytes=*/8);
      if (!plan_or.ok()) {
        return plan_or.status();
      }
      plan = std::move(*plan_or);
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
              device_id, dst_dev, gsl::not_null<void*>{src_ptr}, absl::MakeSpan(plan), total_size);
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
    auto plan_or =
        store::loader::build_segment_plan_from_canonical_index_json(index_json, total_size, /*align_bytes=*/8);
    if (!plan_or.ok()) {
      return plan_or.status();
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
              absl::MakeSpan(*plan_or),
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

  if (req.has_view()) {
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
    auto spec_or = BuildViewSpecFromProto(req.view().spec());
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
    view_reg.placement = placement;
    view_reg.canonical_size_bytes = req.view().canonical_size_bytes();
    view_reg.registration_kind = registration_kind;
    reg.view = view_reg;
    meta.view_registration = true;
    meta.view_placement = placement;
    meta.view_id = view_reg.view_id;
    meta.view_registration_kind = registration_kind;
    meta.view_canonical_size_bytes = view_reg.canonical_size_bytes;
  }

  if (meta.view_registration && meta.view_registration_kind == store::StoreEngine::ViewRegistrationKind::kPiece) {
    if (plan == RegistrationManager::RegPlan::LEASE) {
      return {StatusCode::FAILED_PRECONDITION, "piece registration does not support lease-in-place"};
    }
    if (meta.client_artifact_id.empty()) {
      return {StatusCode::INVALID_ARGUMENT, "piece registration requires client_artifact_id (cgid)"};
    }
    if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
      return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
    }
    auto binding_or = d_.global_store_client->get_artifact_binding(meta.client_artifact_id);
    if (binding_or.ok()) {
      return {StatusCode::FAILED_PRECONDITION, "assembly is already sealed; new pieces are not allowed"};
    }
    if (!absl::IsNotFound(binding_or.status())) {
      return to_grpc_status(binding_or.status());
    }
  }

  if (plan == RegistrationManager::RegPlan::COALESCED || plan == RegistrationManager::RegPlan::STABLE_DRAM) {
    if (req.has_tensor_index_key())
      reg.tensor_index_key = req.tensor_index_key();
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
    auto begin_or = d_.engine.begin_register_artifact(reg);
    if (!begin_or.ok())
      return to_grpc_status(begin_or.status());
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
    d_.reg.set_meta(out.registration_id, meta);
    rctx.mark_success();
    return Status::OK;
  }
  // CPU plan removed
  if (plan == RegistrationManager::RegPlan::LEASE) {
    // Only LIP (in_place=true) is supported in current release
    if (!meta.lease_in_place) {
      return {StatusCode::UNIMPLEMENTED, "vram_leased (in_place=false) is not implemented; set lease_in_place=true"};
    }
    std::string reg_id = absl::StrCat("reg_", absl::ToUnixNanos(absl::Now()), "_", getpid());
    resp.set_registration_id(reg_id);
    resp.mutable_lease();
    resp.set_device_id(req.device_id());
    resp.set_total_size(req.total_size());
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_begin_lease_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_begin_lease_total unavailable";
    }
    d_.reg.set_meta(reg_id, meta);
    rctx.mark_success();
    return Status::OK;
  }
  return Status::OK;
}

grpc::Status RegistrationController::feed_stream(
    RpcContext& rctx,
    ::grpc::ServerReader<v2::FeedRegisterArtifactStreamRequest>& reader,
    v2::FeedRegisterArtifactStreamResponse& /*resp*/) {
  v2::FeedRegisterArtifactStreamRequest req;
  std::string reg_id;
  RegistrationManager::RegMeta current_meta;
  bool have_meta = false;
  while (reader.Read(&req)) {
    if (reg_id.empty()) {
      reg_id = req.registration_id();
      if (!d_.reg.has_meta(reg_id)) {
        return {StatusCode::NOT_FOUND, "registration_id not found"};
      }
      absl::flat_hash_map<std::string, uint32_t> expired_refs;
      if (d_.reg.expire_if_ttl_elapsed(reg_id, &expired_refs)) {
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto counter = meter->CreateDoubleCounter("tc_register_ttl_expired_feed_total");
          counter->Add(1.0);
        } catch (...) {
          VLOG(1) << "metrics counter tc_register_ttl_expired_feed_total unavailable";
        }
        ReleaseRegionRefs(d_.regions, expired_refs);
        return {StatusCode::DEADLINE_EXCEEDED, "registration expired (TTL)"};
      }
      auto meta_opt = d_.reg.get_meta(reg_id);
      if (!meta_opt.has_value()) {
        return {StatusCode::NOT_FOUND, "registration metadata missing"};
      }
      current_meta = *meta_opt;
      have_meta = true;
    } else if (req.registration_id() != reg_id) {
      return {StatusCode::INVALID_ARGUMENT, "registration_id changed in stream"};
    }

    uint32_t extend_ms = d_.reg.extend_if_has_ttl(reg_id);
    if (extend_ms > 0) {
      auto st = d_.engine.keep_alive_registered_artifact(reg_id, extend_ms);
      if (!st.ok()) {
        LOG(WARNING) << "keep_alive_registered_artifact failed (stream): reg_id=" << reg_id << ": " << st;
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
      d_.reg.append_lease_segments(reg_id, std::move(to_add));
    } else if (req.has_view_chunk()) {
      const std::string& payload = req.view_chunk().data();
      absl::Span<const std::byte> bytes(reinterpret_cast<const std::byte*>(payload.data()), payload.size());
      auto ingest_status = d_.engine.ingest_view_registration_chunk(reg_id, req.view_chunk().view_offset(), bytes);
      if (!ingest_status.ok()) {
        return to_grpc_status(ingest_status);
      }
      record_view_bytes_metric(static_cast<double>(payload.size()));
      auto ingested_or = d_.engine.get_view_registration_ingested_bytes(reg_id);
      if (ingested_or.ok()) {
        d_.reg.update_view_ingested_bytes(reg_id, *ingested_or);
      }
    } else if (!req.storage_entries().empty() || !req.tensor_aliases().empty()) {
      // allow requests that only carry storage/alias metadata without segments
    } else {
      return {StatusCode::INVALID_ARGUMENT, "missing feed payload"};
    }
    if (!req.storage_entries().empty()) {
      if (!have_meta) {
        auto meta_opt = d_.reg.get_meta(reg_id);
        if (!meta_opt.has_value()) {
          return {StatusCode::NOT_FOUND, "registration metadata missing"};
        }
        current_meta = *meta_opt;
        have_meta = true;
      }
      RegionPinGuard pin_guard(d_.regions);
      std::vector<RegisterStorageMeta> storages;
      storages.reserve(req.storage_entries().size());
      for (const auto& entry : req.storage_entries()) {
        RegisterStorageMeta meta;
        meta.storage_id = entry.storage_id();
        meta.device_id = entry.device_id();
        if (!entry.cuda_ipc_handle().empty())
          meta.handle_bytes = entry.cuda_ipc_handle();
        meta.storage_length = entry.storage_length();
        const bool has_region = !entry.vram_region_id().empty();
        if (has_region)
          meta.region_id = entry.vram_region_id();
        meta.mapping_base_offset = entry.mapping_base_offset();
        if (meta.handle_bytes.empty() == meta.region_id.empty()) {
          return {StatusCode::INVALID_ARGUMENT, "storage entry must specify exactly one source"};
        }
        if (has_region) {
          // Validate using describe() first to avoid leaking a ref on failure
          auto desc_or = d_.regions.describe(meta.region_id);
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
          // Only acquire after validation succeeds; immediately track in pin_guard
          auto acq_or = d_.regions.acquire(meta.region_id, current_meta.owner_pid);
          if (!acq_or.ok()) {
            return to_grpc_status(acq_or.status());
          }
          pin_guard.add(meta.region_id);
        }
        storages.push_back(std::move(meta));
      }
      if (!storages.empty()) {
        d_.reg.append_storage_entries(reg_id, std::move(storages));
        for (const auto& [region_id, count] : pin_guard.refs()) {
          d_.reg.add_region_reference(reg_id, region_id, count);
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
        for (int64_t v : alias.shape())
          meta.shape.push_back(v);
        meta.stride.reserve(alias.stride().size());
        for (int64_t v : alias.stride())
          meta.stride.push_back(v);
        meta.dtype = alias.dtype();
        aliases.push_back(std::move(meta));
      }
      if (!aliases.empty())
        d_.reg.append_tensor_aliases(reg_id, std::move(aliases));
    }
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status RegistrationController::feed_vector(const std::vector<v2::FeedRegisterArtifactStreamRequest>& reqs) {
  std::string reg_id;
  RegistrationManager::RegMeta current_meta;
  bool have_meta = false;
  for (const auto& req : reqs) {
    if (reg_id.empty()) {
      reg_id = req.registration_id();
      if (!d_.reg.has_meta(reg_id)) {
        return {StatusCode::NOT_FOUND, "registration_id not found"};
      }
      absl::flat_hash_map<std::string, uint32_t> expired_refs;
      if (d_.reg.expire_if_ttl_elapsed(reg_id, &expired_refs)) {
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto counter = meter->CreateDoubleCounter("tc_register_ttl_expired_feed_total");
          counter->Add(1.0);
        } catch (...) {
          VLOG(1) << "metrics counter tc_register_ttl_expired_feed_total unavailable";
        }
        ReleaseRegionRefs(d_.regions, expired_refs);
        return {StatusCode::DEADLINE_EXCEEDED, "registration expired (TTL)"};
      }
      auto meta_opt = d_.reg.get_meta(reg_id);
      if (!meta_opt.has_value()) {
        return {StatusCode::NOT_FOUND, "registration metadata missing"};
      }
      current_meta = *meta_opt;
      have_meta = true;
    } else if (req.registration_id() != reg_id) {
      return {StatusCode::INVALID_ARGUMENT, "registration_id changed in stream"};
    }
    {
      uint32_t extend_ms = d_.reg.extend_if_has_ttl(reg_id);
      if (extend_ms > 0) {
        auto st = d_.engine.keep_alive_registered_artifact(reg_id, extend_ms);
        if (!st.ok()) {
          LOG(WARNING) << "keep_alive_registered_artifact failed (vector): reg_id=" << reg_id << ": " << st;
          try {
            static auto meter =
                opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
            static auto ctr = meter->CreateDoubleCounter("tc_register_keepalive_failed_total");
            ctr->Add(1.0);
          } catch (...) {
          }
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
      d_.reg.append_lease_segments(reg_id, std::move(to_add));
    } else if (req.has_view_chunk()) {
      const std::string& payload = req.view_chunk().data();
      absl::Span<const std::byte> bytes(reinterpret_cast<const std::byte*>(payload.data()), payload.size());
      auto ingest_status = d_.engine.ingest_view_registration_chunk(reg_id, req.view_chunk().view_offset(), bytes);
      if (!ingest_status.ok()) {
        return to_grpc_status(ingest_status);
      }
      record_view_bytes_metric(static_cast<double>(payload.size()));
      auto ingested_or = d_.engine.get_view_registration_ingested_bytes(reg_id);
      if (ingested_or.ok()) {
        d_.reg.update_view_ingested_bytes(reg_id, *ingested_or);
      }
    } else if (!req.storage_entries().empty() || !req.tensor_aliases().empty()) {
      // allow metadata-only payloads
    } else {
      return {StatusCode::INVALID_ARGUMENT, "missing feed payload"};
    }
    if (!req.storage_entries().empty()) {
      if (!have_meta) {
        auto meta_opt = d_.reg.get_meta(reg_id);
        if (!meta_opt.has_value()) {
          return {StatusCode::NOT_FOUND, "registration metadata missing"};
        }
        current_meta = *meta_opt;
        have_meta = true;
      }
      RegionPinGuard pin_guard(d_.regions);
      std::vector<RegisterStorageMeta> storages;
      storages.reserve(req.storage_entries().size());
      for (const auto& entry : req.storage_entries()) {
        RegisterStorageMeta meta;
        meta.storage_id = entry.storage_id();
        meta.device_id = entry.device_id();
        if (!entry.cuda_ipc_handle().empty())
          meta.handle_bytes = entry.cuda_ipc_handle();
        meta.storage_length = entry.storage_length();
        const bool has_region = !entry.vram_region_id().empty();
        if (has_region)
          meta.region_id = entry.vram_region_id();
        meta.mapping_base_offset = entry.mapping_base_offset();
        if (meta.handle_bytes.empty() == meta.region_id.empty()) {
          return {StatusCode::INVALID_ARGUMENT, "storage entry must specify exactly one source"};
        }
        if (has_region) {
          // Validate using describe() first to avoid leaking a ref on failure
          auto desc_or = d_.regions.describe(meta.region_id);
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
          // Only acquire after validation succeeds; immediately track in pin_guard
          auto acq_or = d_.regions.acquire(meta.region_id, current_meta.owner_pid);
          if (!acq_or.ok()) {
            return to_grpc_status(acq_or.status());
          }
          pin_guard.add(meta.region_id);
        }
        storages.push_back(std::move(meta));
      }
      if (!storages.empty()) {
        d_.reg.append_storage_entries(reg_id, std::move(storages));
        for (const auto& [region_id, count] : pin_guard.refs()) {
          d_.reg.add_region_reference(reg_id, region_id, count);
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
        for (int64_t v : alias.shape())
          meta.shape.push_back(v);
        meta.stride.reserve(alias.stride().size());
        for (int64_t v : alias.stride())
          meta.stride.push_back(v);
        meta.dtype = alias.dtype();
        aliases.push_back(std::move(meta));
      }
      if (!aliases.empty())
        d_.reg.append_tensor_aliases(reg_id, std::move(aliases));
    }
  }
  return Status::OK;
}

grpc::Status RegistrationController::keep_alive(
    RpcContext& rctx,
    const v2::KeepAliveRegisterArtifactRequest& req,
    v2::KeepAliveRegisterArtifactResponse& /*resp*/) {
  {
    auto st = d_.reg.keepalive_precommit(
        req.registration_id(), req.owner_pid(), req.epoch(), req.ttl_ms() > 0 ? req.ttl_ms() : 0, d_.engine);
    if (st.ok())
      goto KEEPALIVE_OK;
    else if (!absl::IsNotFound(st))
      return to_grpc_status(st);
  }
  {
    auto st = d_.lip.keepalive_lease(
        req.registration_id(), req.owner_pid(), req.epoch(), req.ttl_ms() > 0 ? req.ttl_ms() : 0);
    if (!st.ok())
      return to_grpc_status(st);
  }
KEEPALIVE_OK:
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
    if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
      return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
    }
    auto binding_or = d_.global_store_client->get_artifact_binding(meta.client_artifact_id);
    if (binding_or.ok()) {
      absl::Status abort_status = d_.engine.abort_registered_artifact(req.registration_id());
      if (!abort_status.ok()) {
        LOG(WARNING) << "abort_registered_artifact failed after sealed binding: " << abort_status;
      }
      auto refs = d_.reg.erase_all_for(req.registration_id());
      ReleaseRegionRefs(d_.regions, refs);
      return {StatusCode::FAILED_PRECONDITION, "assembly is already sealed; new pieces are not allowed"};
    }
    if (!absl::IsNotFound(binding_or.status())) {
      return to_grpc_status(binding_or.status());
    }
  }

  if (meta.plan == RegistrationManager::RegPlan::LEASE) {
    // Only LIP (in_place=true) is supported in current release
    if (!meta.lease_in_place) {
      return {StatusCode::UNIMPLEMENTED, "vram_leased (in_place=false) is not implemented; set lease_in_place=true"};
    }
    if (meta.expiry.time_since_epoch().count() > 0 && std::chrono::steady_clock::now() > meta.expiry) {
      auto refs = d_.reg.erase_all_for(req.registration_id());
      ReleaseRegionRefs(d_.regions, refs);
      try {
        static auto meter =
            opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
        static auto counter = meter->CreateDoubleCounter("tc_register_ttl_expired_commit_total");
        counter->Add(1.0);
      } catch (...) {
        VLOG(1) << "metrics counter tc_register_ttl_expired_commit_total unavailable";
      }
      return {StatusCode::DEADLINE_EXCEEDED, "registration expired (TTL)"};
    }

    // Always commit lease in-place: do not allocate destination GPU memory
    auto lease_vec = d_.reg.get_lease_segments(req.registration_id());
    if (lease_vec.empty()) {
      return {StatusCode::FAILED_PRECONDITION, "no lease segments fed"};
    }

    auto storage_entries = d_.reg.get_storage_entries(req.registration_id());
    auto alias_vec = d_.reg.get_tensor_aliases(req.registration_id());
    if (storage_entries.empty()) {
      return {StatusCode::FAILED_PRECONDITION, "registration missing storage_entries payload"};
    }
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

    auto out_or = d_.lip.commit_lease_in_place(
        req.registration_id(),
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

    ResolvedStorePolicy resolved;
    if (meta.resolved_policy.has_value()) {
      resolved = *meta.resolved_policy;
    } else {
      auto default_or = resolve_store_policy(nullptr);
      if (!default_or.ok()) {
        return to_grpc_status(default_or.status());
      }
      resolved = *default_or;
    }
    auto* local_stable = resp.mutable_local_stable_tier();
    const char* op_label = local_stable_op_label(meta.plan);
    const char* requirement = requirement_level_label(resolved.local_requirement);
    if (meta.view_registration) {
      local_stable->set_status(v2::LOCAL_STABLE_TIER_STATUS_SKIPPED);
      local_stable->set_message("view registrations do not satisfy the local stable tier");
      record_local_stable_tier_metrics(op_label, "skipped", requirement, std::nullopt);
    } else if (static_cast<int>(resolved.local_requirement) < static_cast<int>(RequirementLevel::kShould)) {
      local_stable->set_status(v2::LOCAL_STABLE_TIER_STATUS_SKIPPED);
      record_local_stable_tier_metrics(op_label, "skipped", requirement, std::nullopt);
    } else {
      const auto stable_policy_opt = stable_cache_policy_from_resolved(resolved);
      const auto local_stable_start = std::chrono::steady_clock::now();
      auto admit_or = ensure_local_stable_admission(
          d_.engine,
          d_.lip,
          d_.regions,
          resolved,
          out.artifact_id,
          out.id_kind,
          out.total_size,
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
        LOG(WARNING) << "local_stable_tier." << outcome << ": artifact_id=" << out.artifact_id << " op=" << op_label
                     << " requirement=" << requirement << " retention=" << retention << " overflow=" << overflow
                     << " seconds=" << local_stable_seconds << " message=\"" << message << "\"";
        if (resolved.local_requirement == RequirementLevel::kMust) {
          record_local_stable_tier_metrics(op_label, "failed", requirement, local_stable_seconds);
          (void)d_.lip.revoke_by_registration_id(req.registration_id());
          auto refs = d_.reg.erase_all_for(req.registration_id());
          ReleaseRegionRefs(d_.regions, refs);
          return to_grpc_status(admit_or.ok() ? absl::FailedPreconditionError(message) : admit_or.status());
        }
        local_stable->set_status(v2::LOCAL_STABLE_TIER_STATUS_DEGRADED);
        local_stable->set_message(message);
        record_local_stable_tier_metrics(op_label, "degraded", requirement, local_stable_seconds);
      } else {
        local_stable->set_status(v2::LOCAL_STABLE_TIER_STATUS_READY);
        const char* retention =
            stable_policy_opt.has_value() ? stable_retention_label(stable_policy_opt->retention_policy) : "unknown";
        const char* overflow =
            stable_policy_opt.has_value() ? stable_overflow_label(stable_policy_opt->overflow_policy) : "unknown";
        LOG(INFO) << "local_stable_tier.ready: artifact_id=" << out.artifact_id << " op=" << op_label
                  << " requirement=" << requirement << " retention=" << retention << " overflow=" << overflow
                  << " seconds=" << local_stable_seconds;
        record_local_stable_tier_metrics(op_label, "ready", requirement, local_stable_seconds);
      }
    }

    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_commit_lip_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_commit_lip_total unavailable";
    }
    // Create CommitLease for VRAM_LEASED in-place ownership (device-unique)
    if (d_.lifecycle) {
      SessionLifecycleManager::CommitSubject subj{.artifact_id = out.artifact_id, .device_id = meta.device_id};
      auto lid_or = d_.lifecycle->create_commit_lease(subj, meta.owner_pid);
      if (!lid_or.ok()) {
        LOG(WARNING) << "create_commit_lease failed: artifact_id=" << out.artifact_id << " dev=" << meta.device_id
                     << ": " << lid_or.status();
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto ctr = meter->CreateDoubleCounter("tc_lease_create_failed_total");
          ctr->Add(1.0);
        } catch (...) {
        }
      }
    }
    // Log lease-in-place registration summary including plan.
    LOG(INFO) << "Registered memory replica: " << out.artifact_id
              << " plan=vram_leased(in_place) device=gpu:" << meta.device_id << " size=" << out.total_size << "B";
    auto refs = d_.reg.erase_all_for(req.registration_id());
    ReleaseRegionRefs(d_.regions, refs);
    rctx.mark_success();
    return Status::OK;
  }
  {
    auto commit_or = d_.engine.commit_registered_artifact(req.registration_id());
    if (!commit_or.ok())
      return to_grpc_status(commit_or.status());
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
        LOG(INFO) << "View registration partial coverage: artifact_id=" << out.artifact_id
                  << " view_id=" << *out.view_id << " covered_bytes=" << covered_bytes
                  << " canonical_bytes=" << canonical_size_bytes << " ingested_view_bytes=" << meta.view_ingested_bytes;
      } else {
        VLOG(1) << "View registration coverage: artifact_id=" << out.artifact_id << " view_id=" << *out.view_id
                << " covered_bytes=" << covered_bytes << " canonical_bytes=" << canonical_size_bytes
                << " ingested_view_bytes=" << meta.view_ingested_bytes;
      }
    }

    ResolvedStorePolicy resolved;
    if (meta.resolved_policy.has_value()) {
      resolved = *meta.resolved_policy;
    } else {
      auto default_or = resolve_store_policy(nullptr);
      if (!default_or.ok()) {
        return to_grpc_status(default_or.status());
      }
      resolved = *default_or;
    }
    auto* local_stable = resp.mutable_local_stable_tier();
    const char* op_label = local_stable_op_label(meta.plan);
    const char* requirement = requirement_level_label(resolved.local_requirement);
    if (meta.view_registration) {
      local_stable->set_status(v2::LOCAL_STABLE_TIER_STATUS_SKIPPED);
      local_stable->set_message("view registrations do not satisfy the local stable tier");
      record_local_stable_tier_metrics(op_label, "skipped", requirement, std::nullopt);
    } else if (static_cast<int>(resolved.local_requirement) < static_cast<int>(RequirementLevel::kShould)) {
      local_stable->set_status(v2::LOCAL_STABLE_TIER_STATUS_SKIPPED);
      record_local_stable_tier_metrics(op_label, "skipped", requirement, std::nullopt);
    } else {
      const auto stable_policy_opt = stable_cache_policy_from_resolved(resolved);
      const auto local_stable_start = std::chrono::steady_clock::now();
      auto admit_or = ensure_local_stable_admission(
          d_.engine,
          d_.lip,
          d_.regions,
          resolved,
          out.artifact_id,
          out.id_kind,
          out.size_bytes,
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
        LOG(WARNING) << "local_stable_tier." << outcome << ": artifact_id=" << out.artifact_id << " op=" << op_label
                     << " requirement=" << requirement << " retention=" << retention << " overflow=" << overflow
                     << " seconds=" << local_stable_seconds << " message=\"" << message << "\"";
        if (resolved.local_requirement == RequirementLevel::kMust) {
          record_local_stable_tier_metrics(op_label, "failed", requirement, local_stable_seconds);
          if (!out.existed) {
            store::loading::ReplicaKey base_key{.artifact_id = out.artifact_id, .device = out.device, .replica = 0};
            (void)d_.engine.unload_replica(base_key);
            auto unreg_status = d_.engine.unregister_replica_from_global_store(out.artifact_id, out.device.ordinal);
            if (!unreg_status.ok()) {
              LOG(WARNING)
                  << "unregister_replica_from_global_store failed after must local stable failure: artifact_id="
                  << out.artifact_id << " dev=" << out.device.ordinal << ": " << unreg_status;
            }
          }
          d_.reg.erase_meta(req.registration_id());
          return to_grpc_status(admit_or.ok() ? absl::FailedPreconditionError(message) : admit_or.status());
        }
        local_stable->set_status(v2::LOCAL_STABLE_TIER_STATUS_DEGRADED);
        local_stable->set_message(message);
        record_local_stable_tier_metrics(op_label, "degraded", requirement, local_stable_seconds);
      } else {
        local_stable->set_status(v2::LOCAL_STABLE_TIER_STATUS_READY);
        const char* retention =
            stable_policy_opt.has_value() ? stable_retention_label(stable_policy_opt->retention_policy) : "unknown";
        const char* overflow =
            stable_policy_opt.has_value() ? stable_overflow_label(stable_policy_opt->overflow_policy) : "unknown";
        LOG(INFO) << "local_stable_tier.ready: artifact_id=" << out.artifact_id << " op=" << op_label
                  << " requirement=" << requirement << " retention=" << retention << " overflow=" << overflow
                  << " seconds=" << local_stable_seconds;
        record_local_stable_tier_metrics(op_label, "ready", requirement, local_stable_seconds);
      }
    }

    if (out.existed) {
      store::loading::ReplicaKey key{.artifact_id = out.artifact_id, .device = out.device, .replica = 0};
      d_.refs.add_ref(key, meta.owner_pid);
      if (d_.lifecycle && meta.ttl_ms > 0 && out.device.type == DeviceType::GPU) {
        auto lease_or = d_.lifecycle->create_ttl_use_lease(key, meta.owner_pid, absl::Milliseconds(meta.ttl_ms));
        if (lease_or.ok()) {
          meta.use_lease_id = *lease_or;
        } else {
          LOG(ERROR) << "failed to create ttl use lease: " << lease_or.status();
        }
      }
      meta.device_id = out.device.ordinal;
      meta.joined_existing = true;
      meta.artifact_id_mi2 = out.artifact_id;
      d_.reg.set_meta(req.registration_id(), meta);
    } else {
      d_.reg.erase_meta(req.registration_id());
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
}

grpc::Status RegistrationController::abort(
    RpcContext& rctx,
    const v2::AbortRegisteredArtifactRequest& req,
    v2::AbortRegisteredArtifactResponse& /*resp*/) {
  auto st = d_.engine.abort_registered_artifact(req.registration_id());
  if (!st.ok())
    return to_grpc_status(st);
  auto refs = d_.reg.erase_all_for(req.registration_id());
  ReleaseRegionRefs(d_.regions, refs);
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
  {
    auto st = d_.lip.revoke_by_registration_id(req.registration_id());
    if (st.ok()) {
      auto refs = d_.reg.erase_all_for(req.registration_id());
      ReleaseRegionRefs(d_.regions, refs);
      goto REVOKE_DONE;
    }
  }
  {
    absl::Status _st = d_.engine.abort_registered_artifact(req.registration_id());
    if (!_st.ok()) {
      LOG(WARNING) << "revoke: abort_registered_artifact failed for id=" << req.registration_id() << ": " << _st;
    }
    auto refs = d_.reg.erase_all_for(req.registration_id());
    ReleaseRegionRefs(d_.regions, refs);
  }
REVOKE_DONE:
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
