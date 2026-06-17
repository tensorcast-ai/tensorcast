// Copyright (c) 2025-2026, TensorCast Team.

// Implementation of RegistrationController

#include "daemon/service/controllers/registration_controller.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

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
#include "core/store/materialization/common/piece_view_state_utils.h"
#include "core/store/materialization/common/view_registration_normalization_utils.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/view_utils.h"
#include "daemon/service/controllers/local_stable_tier_service.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/service/controllers/materialization_replica_handle_utils.h"
#include "daemon/service/controllers/registration_storage_mapping_utils.h"
#include "daemon/state/store_policy_resolver.h"
#include "daemon/util/status_utils.h"
#include "opentelemetry/metrics/provider.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

constexpr std::uint64_t kFeedStreamProgressLogIntervalBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;

grpc::Status await_state_sync_barrier(const RegistrationController::Dep& dep) {
  if (!dep.await_state_sync_barrier) {
    return Status::OK;
  }
  const absl::Status barrier_status = dep.await_state_sync_barrier();
  if (barrier_status.ok()) {
    return Status::OK;
  }
  LOG(WARNING) << "Piece registration state sync barrier failed: " << barrier_status;
  return to_grpc_status(barrier_status);
}

void EraseRegistrationRegionRefs(
    RegistrationManager& registration_manager,
    HandleLeaseRegistry* handle_leases,
    IpcRegionRegistry& registry,
    std::string_view registration_id) {
  auto meta_opt = registration_manager.get_meta(std::string(registration_id));
  if (handle_leases != nullptr && meta_opt.has_value() && !meta_opt->stable_dram_publish_lease_token.empty()) {
    auto release_status = handle_leases->release(meta_opt->stable_dram_publish_lease_token);
    if (!release_status.ok() && !absl::IsNotFound(release_status)) {
      LOG(WARNING) << "release stable_dram publish lease failed for registration_id=" << registration_id << ": "
                   << release_status;
    }
  }
  auto refs = registration_manager.erase_all_for(std::string(registration_id));
  release_region_refs(registry, refs);
}

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

void release_stable_publish_lease_if_any(
    RegistrationController::Dep& dep,
    RegistrationManager::RegMeta& meta,
    std::string_view registration_id) {
  if (meta.stable_dram_publish_lease_token.empty()) {
    return;
  }
  if (dep.handle_leases == nullptr) {
    LOG(WARNING) << "stable_dram publish lease exists but handle lease registry is unavailable; registration_id="
                 << registration_id;
    meta.stable_dram_publish_lease_token.clear();
    return;
  }
  auto release_status = dep.handle_leases->release(meta.stable_dram_publish_lease_token);
  if (!release_status.ok() && !absl::IsNotFound(release_status)) {
    LOG(WARNING) << "release stable_dram publish lease failed for registration_id=" << registration_id << ": "
                 << release_status;
  }
  meta.stable_dram_publish_lease_token.clear();
}

using materialization_policy::compute_view_id_from_spec;
using materialization_policy::convert_view_spec;

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
  if (local_stable == nullptr) {
    return absl::InvalidArgumentError("local_stable result is required");
  }
  LocalStableTierService service(LocalStableTierService::Dep{.engine = engine, .lip = lip, .regions = regions});
  return service.apply_local_stable_tier(
      meta, artifact_id, id_kind, total_size, *local_stable, on_must_failure_cleanup);
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

v2::ViewRegistrationKind ToProtoRegistrationKind(store::StoreEngine::ViewRegistrationKind kind) {
  switch (kind) {
    case store::StoreEngine::ViewRegistrationKind::kPiece:
      return v2::VIEW_REGISTRATION_KIND_PIECE;
    case store::StoreEngine::ViewRegistrationKind::kCanonical:
    case store::StoreEngine::ViewRegistrationKind::kUnspecified:
    default:
      return v2::VIEW_REGISTRATION_KIND_CANONICAL;
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

template <typename NextRequestFn>
grpc::Status process_feed_requests(
    RegistrationController::Dep& dep,
    NextRequestFn&& next_request,
    const char* source_label) {
  v2::FeedRegisterArtifactStreamRequest req;
  std::string reg_id;
  RegistrationManager::RegMeta current_meta;
  bool have_meta = false;
  const absl::Time stream_start = absl::Now();
  std::uint64_t streamed_view_bytes = 0;
  std::uint64_t streamed_view_chunks = 0;
  std::uint64_t next_progress_log_bytes = kFeedStreamProgressLogIntervalBytes;
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
        release_region_refs(dep.regions, expired_refs);
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
      const uint64_t view_offset = req.view_chunk().view_offset();
      const std::uint64_t payload_bytes = static_cast<std::uint64_t>(payload.size());
      absl::Status ingest_status;
      if (have_meta && current_meta.plan == RegistrationManager::RegPlan::STABLE_DRAM && !current_meta.stage_on_gpu &&
          !current_meta.view_registration) {
        ingest_status = dep.engine.ingest_registration_chunk(reg_id, view_offset, bytes);
      } else {
        ingest_status = dep.engine.ingest_view_registration_chunk(reg_id, view_offset, bytes);
      }
      if (!ingest_status.ok()) {
        return to_grpc_status(ingest_status);
      }
      record_view_bytes_metric(static_cast<double>(payload.size()));
      if (!(have_meta && current_meta.plan == RegistrationManager::RegPlan::STABLE_DRAM && !current_meta.stage_on_gpu &&
            !current_meta.view_registration)) {
        auto ingested_or = dep.engine.get_view_registration_ingested_bytes(reg_id);
        if (ingested_or.ok()) {
          dep.reg.update_view_ingested_bytes(reg_id, *ingested_or);
        }
      }
      streamed_view_bytes += payload_bytes;
      streamed_view_chunks += 1;
      if (streamed_view_bytes >= next_progress_log_bytes) {
        const absl::Duration elapsed = absl::Now() - stream_start;
        const double elapsed_s = std::max(1e-6, absl::ToDoubleSeconds(elapsed));
        const double gib = static_cast<double>(streamed_view_bytes) / static_cast<double>(1ULL << 30);
        LOG(INFO) << absl::StrFormat(
            "feed_stream progress source=%s registration_id=%s chunks=%d bytes=%d elapsed_s=%.3f avg_gibps=%.3f",
            source_label,
            reg_id,
            static_cast<long long>(streamed_view_chunks),
            static_cast<long long>(streamed_view_bytes),
            elapsed_s,
            gib / elapsed_s);
        next_progress_log_bytes += kFeedStreamProgressLogIntervalBytes;
      }
    } else if (req.has_stable_dram_write_progress()) {
      if (!(have_meta && current_meta.plan == RegistrationManager::RegPlan::STABLE_DRAM && !current_meta.stage_on_gpu &&
            !current_meta.view_registration)) {
        return {
            StatusCode::FAILED_PRECONDITION,
            "stable_dram write progress requires non-view stable_dram stage_on_gpu=false"};
      }
      std::uint64_t acked_bytes = 0;
      std::uint64_t acked_ranges = 0;
      for (const auto& range : req.stable_dram_write_progress().ranges()) {
        const uint64_t canonical_offset = range.canonical_offset();
        const uint64_t length = range.length();
        auto ingest_status = dep.engine.ingest_registration_written_range(reg_id, canonical_offset, length);
        if (!ingest_status.ok()) {
          return to_grpc_status(ingest_status);
        }
        acked_bytes += length;
        acked_ranges += 1;
      }
      if (acked_bytes > 0) {
        record_view_bytes_metric(static_cast<double>(acked_bytes));
        streamed_view_bytes += acked_bytes;
        streamed_view_chunks += acked_ranges;
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

  if (streamed_view_chunks > 0) {
    const absl::Duration elapsed = absl::Now() - stream_start;
    const double elapsed_s = std::max(1e-6, absl::ToDoubleSeconds(elapsed));
    const double gib = static_cast<double>(streamed_view_bytes) / static_cast<double>(1ULL << 30);
    LOG(INFO) << absl::StrFormat(
        "feed_stream done source=%s registration_id=%s chunks=%d bytes=%d elapsed_s=%.3f avg_gibps=%.3f",
        source_label,
        reg_id,
        static_cast<long long>(streamed_view_chunks),
        static_cast<long long>(streamed_view_bytes),
        elapsed_s,
        gib / elapsed_s);
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

  const auto parsed_or = store::view::parse_view_selection_json(meta.view_spec_json);
  if (!parsed_or.ok()) {
    return to_grpc_status(parsed_or.status());
  }
  const auto plan_or = parsed_or->tensor_names.empty()
      ? store::loader::ViewPlanner::compute_bidirectional_view_plan(meta.index_data, parsed_or->spec)
      : store::loader::ViewPlanner::compute_bidirectional_view_plan(
            meta.index_data, parsed_or->spec, parsed_or->tensor_names);
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
  auto piece_payload_or = store::materialization::common::build_piece_view_state_payload(
      store::materialization::common::PieceViewStateRequest{
          .canonical_index_json = stable_index_json,
          .view_id = meta.view_id,
          .plan = view_plan,
          .view_source = view_source,
          .leaf_chunk_bytes = leaf_chunk_bytes,
          .canonical_size_bytes = meta.view_canonical_size_bytes,
      });
  if (!piece_payload_or.ok()) {
    return to_grpc_status(piece_payload_or.status());
  }
  auto piece_payload = std::move(*piece_payload_or);
  const std::string view_data_hash = piece_payload.view_data_hash;
  const auto& canonical_ranges = piece_payload.canonical_ranges;
  const uint64_t covered_bytes = piece_payload.canonical_bytes_covered;
  const uint64_t canonical_size_bytes = piece_payload.canonical_size_bytes;
  auto leaf_writes = std::move(piece_payload.leaf_writes);
  auto proof_digests = std::move(piece_payload.proof_digests);

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
    return {
        StatusCode::UNAVAILABLE,
        absl::StrCat(
            "worker identity unavailable while registering memory replica for artifact_id=",
            meta.client_artifact_id,
            " view_id=",
            meta.view_id,
            "; routable replicas require completed worker lifecycle registration")};
  }
  const store::DeviceKey device = store::DeviceRegistry::instance().gpu_key(meta.device_id);

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
      /*max_concurrency=*/std::max<uint32_t>(1, dep.max_concurrency),
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

  const auto barrier_status = await_state_sync_barrier(dep);
  if (!barrier_status.ok()) {
    return barrier_status;
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
  resp.set_registration_kind(v2::VIEW_REGISTRATION_KIND_PIECE);

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
  EraseRegistrationRegionRefs(dep.reg, dep.handle_leases, dep.regions, registration_id);
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
    EraseRegistrationRegionRefs(dep.reg, dep.handle_leases, dep.regions, registration_id);
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
    EraseRegistrationRegionRefs(dep.reg, dep.handle_leases, dep.regions, registration_id);
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
        EraseRegistrationRegionRefs(dep.reg, dep.handle_leases, dep.regions, registration_id);
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
  EraseRegistrationRegionRefs(dep.reg, dep.handle_leases, dep.regions, registration_id);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status commit_engine_registration(
    RegistrationController::Dep& dep,
    RpcContext& rctx,
    const std::string& registration_id,
    RegistrationManager::RegMeta meta,
    v2::CommitRegisteredArtifactResponse& resp) {
  const absl::Time controller_start = absl::Now();
  // Release temporary publish handle lease before commit admission so stable
  // budget is not double-counted (publish guard lease + stable cache lease).
  release_stable_publish_lease_if_any(dep, meta, registration_id);
  if (dep.reg.has_meta(registration_id)) {
    dep.reg.set_meta(registration_id, meta);
  }

  const absl::Time engine_commit_start = absl::Now();
  auto commit_or = dep.engine.commit_registered_artifact(registration_id);
  const double engine_commit_ms = absl::ToDoubleMilliseconds(absl::Now() - engine_commit_start);
  if (!commit_or.ok()) {
    LOG(INFO) << absl::StrFormat(
        "CommitRegisteredArtifact profile registration_id=%s phase=engine_commit status=error engine_commit_ms=%.3f "
        "total_ms=%.3f error=%s",
        registration_id,
        engine_commit_ms,
        absl::ToDoubleMilliseconds(absl::Now() - controller_start),
        commit_or.status().ToString());
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
  resp.set_registration_kind(ToProtoRegistrationKind(out.registration_kind));
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
  const absl::Time local_stable_start = absl::Now();
  if (out.stable_cache_admitted) {
    local_stable->set_status(v2::LOCAL_STABLE_TIER_STATUS_READY);
    local_stable->set_message("local stable tier admitted during stable_dram commit");
  } else {
    auto local_stable_status = apply_local_stable_tier(
        dep.engine, dep.lip, dep.regions, meta, out.artifact_id, out.id_kind, out.size_bytes, local_stable, [&]() {
          if (!out.existed) {
            store::loading::ReplicaKey base_key{.artifact_id = out.artifact_id, .device = out.device, .replica = 0};
            (void)dep.engine.unload_replica(base_key);
            auto unreg_status = dep.engine.unregister_replica_from_global_store(out.artifact_id, out.device.ordinal);
            if (!unreg_status.ok()) {
              LOG(WARNING)
                  << "unregister_replica_from_global_store failed after must local stable failure: artifact_id="
                  << out.artifact_id << " dev=" << out.device.ordinal << ": " << unreg_status;
            }
          }
          release_stable_publish_lease_if_any(dep, meta, registration_id);
          dep.reg.erase_meta(registration_id);
        });
    if (!local_stable_status.ok()) {
      LOG(INFO) << absl::StrFormat(
          "CommitRegisteredArtifact profile registration_id=%s phase=local_stable status=error "
          "engine_commit_ms=%.3f local_stable_ms=%.3f total_ms=%.3f error=%s",
          registration_id,
          engine_commit_ms,
          absl::ToDoubleMilliseconds(absl::Now() - local_stable_start),
          absl::ToDoubleMilliseconds(absl::Now() - controller_start),
          local_stable_status.ToString());
      return to_grpc_status(local_stable_status);
    }
  }
  const double local_stable_ms = absl::ToDoubleMilliseconds(absl::Now() - local_stable_start);

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
  LOG(INFO) << absl::StrFormat(
      "CommitRegisteredArtifact profile registration_id=%s status=ok engine_commit_ms=%.3f local_stable_ms=%.3f "
      "total_ms=%.3f artifact_id=%s existed=%d stable_cache_admitted=%d",
      registration_id,
      engine_commit_ms,
      local_stable_ms,
      absl::ToDoubleMilliseconds(absl::Now() - controller_start),
      out.artifact_id,
      out.existed ? 1 : 0,
      out.stable_cache_admitted ? 1 : 0);
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
  if (!req.has_tensor_index_data()) {
    return {StatusCode::INVALID_ARGUMENT, "view registration requires tensor_index_data"};
  }
  if (req.view().spec().tensors().empty() && req.view().tensor_names().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "view registration requires a view spec or tensor_names"};
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
  view_reg.view_id = req.view().view_id();
  view_reg.spec = std::move(*spec_or);
  for (const auto& tensor_name : req.view().tensor_names()) {
    if (!tensor_name.empty()) {
      view_reg.tensor_names.push_back(tensor_name);
    }
  }
  view_reg.placement = placement;
  view_reg.canonical_size_bytes = req.view().canonical_size_bytes();
  view_reg.registration_kind = registration_kind;
  auto normalized_view_or =
      store::materialization::common::normalize_view_registration(view_reg, meta.index_data, meta.total_size);
  if (!normalized_view_or.ok()) {
    return to_grpc_status(normalized_view_or.status());
  }
  reg.view = normalized_view_or->registration;
  meta.view_spec_json = normalized_view_or->view_spec_json;
  meta.view_registration = true;
  meta.view_placement = placement;
  meta.view_id = reg.view->view_id;
  meta.view_registration_kind = registration_kind;
  meta.view_canonical_size_bytes = normalized_view_or->canonical_size_bytes;
  return Status::OK;
}

void maybe_attach_stable_dram_cpu_publish_handle(
    RegistrationController::Dep& dep,
    const std::string& registration_id,
    RegistrationManager::RegMeta& meta,
    v2::StableDramHandshake* hs) {
  if (hs == nullptr || meta.stage_on_gpu || meta.view_registration || meta.owner_pid <= 0) {
    return;
  }
  if (dep.handle_leases == nullptr) {
    VLOG(1) << "stable_dram cpu publish handle unavailable: handle leases disabled";
    return;
  }

  auto memfd_or = dep.engine.get_registration_cpu_memfd_info(registration_id);
  if (!memfd_or.ok()) {
    VLOG(1) << "stable_dram cpu publish handle unavailable for registration_id=" << registration_id << ": "
            << memfd_or.status();
    return;
  }
  const auto& memfd = *memfd_or;
  if (memfd.fd < 0 || memfd.size_bytes == 0) {
    VLOG(1) << "stable_dram cpu publish handle unavailable for registration_id=" << registration_id
            << ": invalid memfd descriptor";
    return;
  }

  auto chunks_or =
      materialization_replica_handle::build_export_chunks_for_replica(dep.engine, memfd.replica_key, memfd.size_bytes);
  if (!chunks_or.ok()) {
    LOG(WARNING) << "stable_dram cpu publish failed to build export chunks for registration_id=" << registration_id
                 << ": " << chunks_or.status();
    return;
  }

  HandleLeaseRegistry::CpuMemfdDescriptor memfd_desc{
      .fd = memfd.fd,
      .size_bytes = memfd.size_bytes,
      .offset_bytes = memfd.offset_bytes,
  };
  auto token_or = dep.handle_leases->mint_cpu_memfd_lease(memfd.replica_key, meta.owner_pid, memfd_desc, *chunks_or);
  if (!token_or.ok()) {
    LOG(WARNING) << "stable_dram cpu publish failed to mint memfd lease for registration_id=" << registration_id << ": "
                 << token_or.status();
    return;
  }

  auto* publish = hs->mutable_publish_cpu_memfd();
  publish->set_size_bytes(memfd.size_bytes);
  publish->set_offset_bytes(memfd.offset_bytes);
  hs->set_publish_cpu_memfd_lease_token(*token_or);
  meta.stable_dram_publish_lease_token = *token_or;
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
    if (meta.stage_on_gpu) {
      hs->set_staging_cuda_ipc_handle(handle_view.data(), handle_view.size());
    } else {
      maybe_attach_stable_dram_cpu_publish_handle(dep, out.registration_id, meta, hs);
    }
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

store::StoreEngine::ArtifactRegistration make_begin_registration(const v2::BeginRegisterArtifactRequest& req) {
  store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = absl::StrCat("mem_reg:", absl::ToUnixNanos(absl::Now()), ":", getpid());
  reg.device_id = req.device_id();
  reg.total_size_bytes = req.total_size();
  reg.enable_p2p = true;
  if (req.has_ttl_ms()) {
    reg.ttl_ms = req.ttl_ms();
  }
  return reg;
}

RegistrationManager::RegPlan select_begin_plan(const v2::BeginRegisterArtifactRequest& req) {
  RegistrationManager::RegPlan plan = RegistrationManager::RegPlan::COALESCED;
  if (req.has_lease()) {
    plan = RegistrationManager::RegPlan::LEASE;
  }
  if (req.has_stable_dram()) {
    plan = RegistrationManager::RegPlan::STABLE_DRAM;
  }
  return plan;
}

absl::StatusOr<RegistrationManager::RegMeta> build_begin_meta_from_request(
    const v2::BeginRegisterArtifactRequest& req,
    RegistrationManager::RegPlan plan,
    store::StoreEngine::ArtifactRegistration& reg) {
  RegistrationManager::RegMeta meta;
  meta.plan = plan;
  meta.total_size = req.total_size();
  meta.device_id = req.device_id();
  meta.owner_pid = req.owner_pid();

  auto policy_or = resolve_store_policy(req.has_policy() ? &req.policy() : nullptr);
  if (!policy_or.ok()) {
    return policy_or.status();
  }
  meta.resolved_policy = *policy_or;
  if (plan == RegistrationManager::RegPlan::STABLE_DRAM && policy_or->local_requirement == RequirementLevel::kMust) {
    reg.stable_cache_policy = stable_cache_policy_from_resolved(*policy_or);
  }

  if (!req.client_artifact_id().empty()) {
    auto id_status = common::validate_client_generated_id(req.client_artifact_id());
    if (!id_status.ok()) {
      return absl::InvalidArgumentError(std::string(id_status.message()));
    }
    meta.id_kind = common::ArtifactIdKind::kCgid;
    meta.client_artifact_id = req.client_artifact_id();
  } else {
    meta.id_kind = common::ArtifactIdKind::kMi2;
    meta.client_artifact_id.clear();
  }

  if (req.has_lease()) {
    meta.lease_in_place = req.lease().in_place();
  }
  if (req.has_stable_dram()) {
    meta.stage_on_gpu = req.stable_dram().stage_on_gpu();
    meta.release_gpu_on_commit = req.stable_dram().release_gpu_on_commit();
  }
  if (req.has_ttl_ms() && req.ttl_ms() > 0) {
    meta.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(req.ttl_ms());
    meta.ttl_ms = static_cast<uint32_t>(req.ttl_ms());
  }
  if (req.has_tensor_index_key()) {
    meta.index_key_hex = req.tensor_index_key();
  } else if (req.has_tensor_index_data()) {
    meta.index_data = std::string(req.tensor_index_data().data().begin(), req.tensor_index_data().data().end());
  }
  return meta;
}

grpc::Status validate_piece_begin_registration(
    RegistrationController::Dep& dep,
    const RegistrationManager::RegMeta& meta) {
  if (!meta.view_registration || meta.view_registration_kind != store::StoreEngine::ViewRegistrationKind::kPiece) {
    return Status::OK;
  }
  if (meta.client_artifact_id.empty()) {
    return {StatusCode::INVALID_ARGUMENT, "piece registration requires client_artifact_id (cgid)"};
  }
  auto sealed_or = is_piece_assembly_sealed(dep.global_store_client.get(), meta.client_artifact_id);
  if (!sealed_or.ok()) {
    return to_grpc_status(sealed_or.status());
  }
  if (*sealed_or) {
    return {StatusCode::FAILED_PRECONDITION, "assembly is already sealed; new pieces are not allowed"};
  }
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

  if (req.owner_pid() <= 0) {
    return {StatusCode::INVALID_ARGUMENT, "owner_pid is required (>0)"};
  }

  store::StoreEngine::ArtifactRegistration reg = make_begin_registration(req);
  const RegistrationManager::RegPlan plan = select_begin_plan(req);
  auto meta_or = build_begin_meta_from_request(req, plan, reg);
  if (!meta_or.ok()) {
    return to_grpc_status(meta_or.status());
  }
  RegistrationManager::RegMeta meta = *meta_or;

  auto view_status = apply_begin_view_registration(req, meta, reg);
  if (!view_status.ok()) {
    return view_status;
  }

  auto piece_validation_status = validate_piece_begin_registration(d_, meta);
  if (!piece_validation_status.ok()) {
    return piece_validation_status;
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
  EraseRegistrationRegionRefs(d_.reg, d_.handle_leases, d_.regions, req.registration_id());
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
  EraseRegistrationRegionRefs(d_.reg, d_.handle_leases, d_.regions, req.registration_id());

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
