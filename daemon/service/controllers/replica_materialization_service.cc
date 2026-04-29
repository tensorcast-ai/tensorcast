// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/replica_materialization_service.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "opentelemetry/metrics/provider.h"

#include "core/store/device_registry.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/view_utils.h"
#include "daemon/service/controllers/materialization_index_source_utils.h"
#include "daemon/service/controllers/materialization_payload_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/service/controllers/materialization_post_seal_utils.h"
#include "daemon/service/controllers/materialization_replica_handle_utils.h"
#include "daemon/service/controllers/materialization_request_common_utils.h"
#include "daemon/service/controllers/selection_validation_utils.h"
#include "daemon/service/controllers/serving_artifact_manifest_utils.h"
#include "daemon/util/deadline_utils.h"
#include "daemon/util/status_utils.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

using materialization_index_source::DescriptorMetadata;
using materialization_index_source::ensure_tensor_index_present;
using materialization_index_source::load_descriptor_metadata;
using materialization_index_source::validate_descriptor_against_index;
using materialization_payload::populate_materialize_payloads;
using materialization_payload::resolve_layout_json;
using materialization_policy::apply_request_context_to_hints;
using materialization_policy::build_view_spec_proto;
using materialization_policy::compute_view_id_from_spec;
using materialization_policy::convert_view_spec;
using materialization_policy::NormalizedMaterializationRequestContext;
using materialization_policy::resolve_collective_group_hint;
using materialization_policy::resolve_materialization_request_context;
using materialization_policy::resolve_transport_scheduling_group_hint;
using materialization_policy::resolve_transform_placement;
using materialization_policy::to_hint_export_policy;
using materialization_post_seal::check_post_seal_view_reuse_safe;
using materialization_replica_handle::bind_replica_handle_for_response;
using materialization_request_common::LeaseContext;
using materialization_request_common::LipFastPathRequest;
using materialization_request_common::materialize_with_shared_disk_retry;
using materialization_request_common::resolve_artifact_and_disk_source;
using materialization_request_common::try_satisfy_lip_fast_path;
using materialization_request_common::validate_and_compute_lease_context;
using selection_validation::compute_and_validate_view_subset_hash;
using selection_validation::validate_hashes_and_build_resolved_selection;
using selection_validation::validate_request_tensor_names;
using store::loader::ViewSpec;

using store::loading::MaterializationSource;

void record_lease_create_failed() {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto ctr = meter->CreateDoubleCounter("tc_lease_create_failed_total");
    ctr->Add(1.0);
  } catch (...) {
  }
}

v2::MaterializationSource to_proto_source(MaterializationSource source) {
  switch (source) {
    case MaterializationSource::kDisk:
      return v2::MaterializationSource::MATERIALIZATION_SOURCE_DISK;
    case MaterializationSource::kP2P:
      return v2::MaterializationSource::MATERIALIZATION_SOURCE_P2P;
    case MaterializationSource::kLocalReplica:
      return v2::MaterializationSource::MATERIALIZATION_SOURCE_LOCAL_REPLICA;
    case MaterializationSource::kUnspecified:
    default:
      return v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED;
  }
}

std::optional<std::string> build_internal_subset_view_id(std::string_view view_subset_hash) {
  if (view_subset_hash.empty()) {
    return std::nullopt;
  }
  return absl::StrCat("subset:", absl::BytesToHexString(view_subset_hash));
}

absl::Status validate_impossible_cpu_stable_budget(const store::StoreEngine& engine, uint64_t required_bytes) {
  if (required_bytes == 0) {
    return absl::OkStatus();
  }
  auto snapshot_opt = engine.get_memory_tier_snapshot();
  if (!snapshot_opt.has_value()) {
    return absl::OkStatus();
  }
  const auto& snapshot = *snapshot_opt;
  if (required_bytes <= snapshot.stable_total_bytes) {
    return absl::OkStatus();
  }
  return absl::ResourceExhaustedError(
      absl::StrFormat(
          "Insufficient stable bytes: requested=%llu used=%llu total=%llu",
          static_cast<unsigned long long>(required_bytes),
          static_cast<unsigned long long>(snapshot.stable_used_bytes),
          static_cast<unsigned long long>(snapshot.stable_total_bytes)));
}

absl::Status bind_materialized_handle(
    store::StoreEngine& engine,
    SessionsService& sessions,
    RefTracker& refs,
    SessionLifecycleManager* lifecycle,
    HandleLeaseRegistry* handle_leases,
    const store::loading::ReplicaHandle& handle,
    std::string_view replica_uuid,
    int32_t effective_pid,
    bool allow_pid_ref,
    bool cpu_target,
    std::string_view lease_log_context,
    v2::MemCopyHandle& out_mem_handle) {
  return bind_replica_handle_for_response(
      engine,
      sessions,
      refs,
      lifecycle,
      handle_leases,
      handle,
      replica_uuid,
      effective_pid,
      allow_pid_ref,
      cpu_target,
      lease_log_context,
      [&]() { record_lease_create_failed(); },
      out_mem_handle);
}

absl::StatusOr<store::loading::ReplicaHandle> retry_materialize_from_shared_disk(
    const absl::Status& initial_status,
    store::StoreEngine& engine,
    const store::DeviceKey& dev,
    store::StoreEngine::MaterializeMode mode,
    store::loading::MaterializeHints& hints,
    store::components::IGlobalStoreClient* global_store_client,
    const std::filesystem::path& storage_path,
    std::string_view resolved_artifact_id,
    const NormalizedMaterializationRequestContext& request_context,
    const grpc::ServerContext& server_context,
    std::optional<std::filesystem::path>& normalized_disk_path,
    const materialization_request_common::PrepareRetryDiskSourceFn& prepare_retry_disk_source) {
  return materialize_with_shared_disk_retry(
      initial_status,
      global_store_client,
      storage_path,
      resolved_artifact_id,
      request_context,
      server_context,
      normalized_disk_path,
      [&](const std::optional<store::loading::DiskSource>& retry_disk_source) {
        return engine.materialize_replica(dev, mode, hints, retry_disk_source);
      },
      prepare_retry_disk_source);
}

std::chrono::milliseconds resolve_materialization_request_budget(
    const grpc::ServerContext& server_context,
    const v2::MaterializeReplicaRequest& req) {
  using clock = std::chrono::system_clock;
  constexpr std::chrono::milliseconds kDefaultBudget{600000};
  constexpr std::chrono::milliseconds kHardCap{1800000};
  const std::chrono::milliseconds requested_budget = req.pinned_allocation_timeout_ms() > 0
      ? std::chrono::milliseconds(req.pinned_allocation_timeout_ms())
      : kDefaultBudget;
  std::chrono::milliseconds effective_budget = std::min(requested_budget, kHardCap);
  const auto grpc_deadline = server_context.deadline();
  if (grpc_deadline != clock::time_point::max()) {
    const auto now = clock::now();
    if (grpc_deadline <= now) {
      return std::chrono::milliseconds(0);
    }
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(grpc_deadline - now);
    if (remaining.count() <= 0) {
      return std::chrono::milliseconds(1);
    }
    effective_budget = std::min(effective_budget, remaining);
  }
  return effective_budget.count() > 0 ? effective_budget : std::chrono::milliseconds(1);
}

} // namespace

ReplicaMaterializationService::ReplicaMaterializationService(Dep d) : d_(std::move(d)) {
  if (!d_.storage_path.empty()) {
    std::error_code ec;
    storage_path_ = std::filesystem::weakly_canonical(d_.storage_path, ec);
    if (ec) {
      ec.clear();
      storage_path_ = d_.storage_path.lexically_normal();
    }
  }
}

grpc::Status ReplicaMaterializationService::materialize_replica(
    RpcContext& rctx,
    const v2::MaterializeReplicaRequest& req,
    v2::MaterializeReplicaResponse& resp) {
  auto& span = rctx.span();
  store::loading::ExecutionTopologyContext execution_topology;
  // Collective disk load is an explicit request contract. Do not infer it from
  // replica_uuid or ambient process state.
  execution_topology.collective_load_group =
      resolve_collective_group_hint(req.has_collective_load_group() ? &req.collective_load_group() : nullptr);
  auto request_context_or = resolve_materialization_request_context(
      req.has_source_policy() ? &req.source_policy() : nullptr,
      std::move(execution_topology),
      req.replica_uuid().empty() ? std::nullopt : std::optional<std::string>(req.replica_uuid()),
      /*verify_checksums=*/true,
      static_cast<int32_t>(req.wait_for_shared_disk_ms()));
  if (!request_context_or.ok()) {
    resp.set_status(v2::MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(request_context_or.status());
  }
  const NormalizedMaterializationRequestContext request_context = *request_context_or;
  const auto& retrieval_policy = request_context.retrieval_policy;
  const bool prefer_disk = retrieval_policy.preference == store::loading::SourcePreference::kPreferDisk;
  bool verify_checksums = request_context.verify_checksums;

  span->SetAttribute("tc.store.verify_checksums", verify_checksums);

  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.device.uuid", req.device_uuid());
  }
  span->SetAttribute("tc.size.bytes", static_cast<int64_t>(req.size_bytes()));
  span->SetAttribute("tc.store.preference", static_cast<int64_t>(retrieval_policy.preference));
  span->SetAttribute("tc.store.allow_p2p", retrieval_policy.allow_p2p);
  span->SetAttribute("tc.store.allow_disk", retrieval_policy.allow_disk);
  span->SetAttribute("tc.view.need_data_hash", req.has_need_view_data_hash() ? req.need_view_data_hash() : true);

  using v2::MaterializeReplicaStatus;
  if (d_.shutdown_signal.is_shutting_down()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  if (!req.has_selection() || req.selection().artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "selection.artifact_id is required"};
  }
  const auto& selection = req.selection();

  const auto dev = d_.devices.From(req.target_device_type(), req.device_uuid(), std::nullopt);
  const bool cpu_target = dev.type == DeviceType::CPU;
  auto lease_context_or = validate_and_compute_lease_context(
      rctx.server_context().peer(),
      req.lease_mode(),
      req.wait_for_completion(),
      req.pid(),
      cpu_target,
      d_.cpu_shared_memory_enabled,
      d_.handle_leases != nullptr);
  if (!lease_context_or.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(lease_context_or.status());
  }

  const LeaseContext lease_context = *lease_context_or;
  const bool loopback_peer = lease_context.loopback_peer;
  const bool no_lease = lease_context.no_lease;
  const int32_t effective_pid = lease_context.effective_pid;

  auto artifact_resolution_or = resolve_artifact_and_disk_source(
      d_.global_store_client,
      &d_.disk_imports,
      storage_path_,
      selection.artifact_id(),
      retrieval_policy.allow_disk,
      /*allow_local_import_fallback=*/true,
      loopback_peer);
  if (!artifact_resolution_or.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(artifact_resolution_or.status());
  }
  auto artifact_resolution = std::move(*artifact_resolution_or);
  std::string resolved_artifact_id = std::move(artifact_resolution.resolved_artifact_id);
  std::optional<std::string> bound_artifact_id = std::move(artifact_resolution.bound_artifact_id);
  std::optional<std::string> fallback_artifact_id = std::move(artifact_resolution.fallback_artifact_id);
  std::optional<bool> post_seal_view_reuse_safe;
  const bool gs_connected = artifact_resolution.gs_connected;
  std::optional<std::filesystem::path> normalized_disk_path = std::move(artifact_resolution.normalized_disk_path);
  std::optional<ArtifactSourceRegistry::Entry> local_import = std::move(artifact_resolution.local_import);
  std::optional<store::loading::DiskSource> disk_source = std::move(artifact_resolution.disk_source);
  span->SetAttribute("tc.store.gs_connected", gs_connected);
  span->SetAttribute("tc.store.local_import_fallback", local_import.has_value());

  span->SetAttribute("tc.artifact.id", resolved_artifact_id);
  resp.set_artifact_id(resolved_artifact_id);
  if (bound_artifact_id.has_value()) {
    span->SetAttribute("tc.artifact.bound", *bound_artifact_id);
  }

  if (normalized_disk_path.has_value()) {
    resp.set_disk_path(normalized_disk_path->string());
    if (rctx.allow_high_card_attrs()) {
      span->SetAttribute("tc.disk.path", normalized_disk_path->string());
    }
  }

  DescriptorMetadata descriptor_meta;
  std::optional<store::loader::IndexInfo> disk_index;
  std::optional<store::loading::DiskMetadata> disk_metadata;
  if (normalized_disk_path.has_value()) {
    auto descriptor_or = load_descriptor_metadata(*normalized_disk_path);
    if (!descriptor_or.ok()) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(descriptor_or.status());
    }
    descriptor_meta = *descriptor_or;
    if (verify_checksums) {
      auto index_or = store::loader::read_from_artifact_dir(*normalized_disk_path, dev.ordinal);
      if (!index_or.ok()) {
        resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
        return to_grpc_status(index_or.status());
      }

      // Safetensors directories commonly omit artifact_descriptor.json. In that case we cannot
      // verify checksums, so we warn and continue with verification disabled.
      if (!descriptor_meta.found && index_or->is_safetensors) {
        LOG(WARNING) << "verify_checksums requested but artifact_descriptor.json missing for safetensors at "
                     << normalized_disk_path->string() << "; skipping descriptor validation";
        verify_checksums = false;
        span->SetAttribute("tc.store.verify_checksums", verify_checksums);
      } else {
        if (!descriptor_meta.found) {
          resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
          return {StatusCode::FAILED_PRECONDITION, "artifact_descriptor.json required when verify_checksums=true"};
        }
        auto validation_status =
            validate_descriptor_against_index(descriptor_meta, *index_or, /*verify_checksums=*/true);
        if (!validation_status.ok()) {
          resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
          return to_grpc_status(validation_status);
        }
      }
      disk_index = std::move(*index_or);
    } else if (prefer_disk) {
      auto idx_status = ensure_tensor_index_present(*normalized_disk_path);
      if (!idx_status.ok()) {
        resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
        return to_grpc_status(idx_status);
      }
    }

    store::loading::DiskMetadata metadata;
    metadata.descriptor_present = descriptor_meta.found;
    metadata.schema_version = descriptor_meta.schema_version;
    metadata.index_multihash = descriptor_meta.index_multihash;
    metadata.data_multihash = descriptor_meta.data_multihash;
    if (disk_index.has_value()) {
      metadata.canonical_index_json = disk_index->canonical_index_json;
      if (disk_index->source_index_json.has_value()) {
        metadata.source_index_json = disk_index->source_index_json;
      }
      if (!disk_index->index_multihash.empty()) {
        metadata.index_multihash = disk_index->index_multihash;
      }
      if (disk_index->total_size_bytes > 0) {
        metadata.logical_total_size = disk_index->total_size_bytes;
      }
      if (disk_index->source_total_size_bytes > 0) {
        metadata.source_total_size_bytes = disk_index->source_total_size_bytes;
      }
      metadata.is_safetensors = disk_index->is_safetensors;
    }
    disk_metadata = std::move(metadata);
  }
  if (local_import.has_value()) {
    if (!disk_metadata.has_value()) {
      disk_metadata = store::loading::DiskMetadata{};
    }
    auto& metadata = *disk_metadata;
    metadata.descriptor_present = metadata.descriptor_present || local_import->descriptor_present;
    if (!metadata.source_index_json.has_value() && local_import->source_index_json.has_value()) {
      metadata.source_index_json = *local_import->source_index_json;
    }
    if (!metadata.index_multihash.has_value() && local_import->index_multihash.has_value()) {
      metadata.index_multihash = *local_import->index_multihash;
    }
    if (!metadata.data_multihash.has_value() && local_import->data_multihash.has_value()) {
      metadata.data_multihash = *local_import->data_multihash;
    }
  }

  if (normalized_disk_path.has_value()) {
    std::optional<uint64_t> expected_size;
    if (disk_metadata.has_value() && disk_metadata->logical_total_size.has_value()) {
      expected_size = disk_metadata->logical_total_size;
    }
    disk_source = store::loading::DiskSource{
        .path = *normalized_disk_path,
        .expected_size = expected_size,
        .require_descriptor = true,
    };
  }
  std::optional<std::string> disk_source_artifact_id;
  if (disk_source.has_value()) {
    disk_source_artifact_id = resolved_artifact_id;
  }

  if (descriptor_meta.artifact_id.has_value() && resolved_artifact_id != *descriptor_meta.artifact_id) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::FAILED_PRECONDITION, "artifact_id mismatch between request and artifact_descriptor.json"};
  }
  const bool has_disk = disk_source.has_value();
  const bool has_artifact = !resolved_artifact_id.empty();

  std::vector<std::string> selection_names;
  std::string_view selection_error_reason;
  auto selection_names_status = validate_request_tensor_names(selection, selection_names, &selection_error_reason);
  if (!selection_names_status.ok()) {
    if (!selection_error_reason.empty()) {
      span->SetAttribute("tc.selection.error_reason", std::string(selection_error_reason));
    }
    return selection_names_status;
  }
  std::string view_subset_hash;
  auto subset_hash_status = compute_and_validate_view_subset_hash(
      selection, absl::MakeSpan(selection_names), view_subset_hash, &selection_error_reason);
  if (!subset_hash_status.ok()) {
    if (!selection_error_reason.empty()) {
      span->SetAttribute("tc.selection.error_reason", std::string(selection_error_reason));
    }
    return subset_hash_status;
  }
  const bool selection_requests_view = selection.has_view_spec() || !selection.view_id().empty();
  const bool selection_requests_subset = !selection_names.empty() || !view_subset_hash.empty();
  LOG(INFO) << "materialize_replica.selection"
            << " artifact_id=" << resolved_artifact_id << " request_tensor_names=" << selection.tensor_names_size()
            << " parsed_selection_names=" << selection_names.size()
            << " has_view_spec=" << (selection.has_view_spec() ? 1 : 0) << " request_view_id=" << selection.view_id()
            << " request_subset_hash_len=" << selection.view_subset_hash().size();
  if (has_artifact && dev.type == DeviceType::GPU && !selection_requests_view && !selection_requests_subset &&
      d_.lip.has_active_on_device(resolved_artifact_id, dev.ordinal)) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::FAILED_PRECONDITION, "lease_in_place not supported for same device_id consumers"};
  }

  // View identity handling
  std::optional<ViewSpec> view_spec;
  std::optional<tensorcast::common::v1::ViewSpec> view_spec_proto;
  std::optional<store::loader::ViewPlan> view_plan;
  std::optional<std::string> canonical_index_json;
  std::optional<std::string> resolved_view_id;
  tensorcast::common::v1::ArtifactSelection resolved_selection;
  const std::string& index_source_artifact_id =
      fallback_artifact_id.has_value() ? *fallback_artifact_id : resolved_artifact_id;

  auto read_canonical_from_disk = [&]() -> absl::StatusOr<std::string> {
    if (!normalized_disk_path.has_value()) {
      return absl::FailedPreconditionError("disk source path required for disk-backed canonical planning");
    }
    if (disk_index.has_value() && !disk_index->canonical_index_json.empty()) {
      return disk_index->canonical_index_json;
    }
    auto idx_status = ensure_tensor_index_present(*normalized_disk_path);
    if (!idx_status.ok()) {
      return idx_status;
    }
    auto local_or = store::loader::read_from_artifact_dir(*normalized_disk_path, dev.ordinal);
    if (!local_or.ok()) {
      return local_or.status();
    }
    return local_or->canonical_index_json;
  };

  if (!has_artifact && !has_disk) {
    return {StatusCode::INVALID_ARGUMENT, "selection requires artifact_id or disk_path for canonical planning"};
  }

  const bool prefer_disk_index = has_disk && (!gs_connected || prefer_disk);
  absl::StatusOr<std::string> index_or =
      prefer_disk_index ? read_canonical_from_disk() : d_.engine.get_canonical_index_by_id(index_source_artifact_id);
  if (!index_or.ok() && has_disk && !prefer_disk_index) {
    auto disk_or = read_canonical_from_disk();
    if (disk_or.ok()) {
      index_or = std::move(disk_or);
    }
  }
  if (!index_or.ok()) {
    return to_grpc_status(index_or.status());
  }
  canonical_index_json = std::move(index_or).value();

  if (selection.has_view_spec()) {
    auto spec_or = convert_view_spec(selection.view_spec());
    if (!spec_or.ok()) {
      return to_grpc_status(spec_or.status());
    }
    view_spec = std::move(*spec_or);
    view_spec_proto = selection.view_spec();
  } else if (!selection.view_id().empty()) {
    if (!has_artifact) {
      return {StatusCode::INVALID_ARGUMENT, "selection.view_id requires artifact_id for routing"};
    }
    auto view_meta_or = d_.engine.get_view_metadata(resolved_artifact_id, selection.view_id());
    if (!view_meta_or.ok() && fallback_artifact_id.has_value() && absl::IsNotFound(view_meta_or.status()) &&
        d_.post_seal_policy.reuse_views_if_safe) {
      if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
        return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
      }
      auto safe_or =
          check_post_seal_view_reuse_safe(*d_.global_store_client, *fallback_artifact_id, resolved_artifact_id);
      if (!safe_or.ok()) {
        return to_grpc_status(safe_or.status());
      }
      post_seal_view_reuse_safe = *safe_or;
      if (*safe_or) {
        view_meta_or = d_.engine.get_view_metadata(*fallback_artifact_id, selection.view_id());
      }
    }
    if (!view_meta_or.ok()) {
      return to_grpc_status(view_meta_or.status());
    }
    auto spec_or = store::view::parse_view_spec_json(view_meta_or->view_spec_json);
    if (!spec_or.ok()) {
      return to_grpc_status(spec_or.status());
    }
    view_spec = std::move(*spec_or);
    view_spec_proto = build_view_spec_proto(*view_spec);
  }

  if (view_spec.has_value() || !selection_names.empty()) {
    auto plan_or =
        store::StoreEngine::compute_view_plan(*canonical_index_json, view_spec.value_or(ViewSpec{}), selection_names);
    if (!plan_or.ok()) {
      return to_grpc_status(plan_or.status());
    }
    view_plan = std::move(*plan_or);
  }

  const bool has_view_transform = view_spec.has_value() && view_plan.has_value() && !view_plan->is_identity;
  if (has_view_transform) {
    if (!view_spec_proto.has_value()) {
      view_spec_proto = build_view_spec_proto(*view_spec);
    }
    if (!view_spec_proto.has_value()) {
      return {StatusCode::FAILED_PRECONDITION, "view spec proto missing while resolving selection"};
    }
    auto view_id_or = compute_view_id_from_spec(*view_spec_proto, *canonical_index_json);
    if (!view_id_or.ok()) {
      return to_grpc_status(view_id_or.status());
    }
    if (!selection.view_id().empty() && selection.view_id() != *view_id_or) {
      return {StatusCode::INVALID_ARGUMENT, "selection.view_id does not match selection.view_spec"};
    }
    resolved_view_id = *view_id_or;
    span->SetAttribute("tc.view.id", *resolved_view_id);
  } else {
    if (!selection.view_id().empty()) {
      if (selection_names.empty()) {
        return {
            StatusCode::INVALID_ARGUMENT, "selection.view_id requires a non-identity view spec or tensor_names subset"};
      }
      resolved_view_id = selection.view_id();
      span->SetAttribute("tc.view.id", *resolved_view_id);
    }
    view_spec.reset();
    view_spec_proto.reset();
  }

  std::string selected_index_json = *canonical_index_json;
  if (view_plan.has_value() && !view_plan->is_identity) {
    selected_index_json = view_plan->view_index_json;
  }
  const bool needs_view_index = !selection_names.empty() || resolved_view_id.has_value();
  const tensorcast::common::v1::ViewSpec* resolved_view_spec =
      (resolved_view_id.has_value() && view_spec_proto.has_value()) ? &*view_spec_proto : nullptr;
  auto selection_identity_status = validate_hashes_and_build_resolved_selection(
      selection,
      resolved_artifact_id,
      resolved_view_id.value_or(""),
      selected_index_json,
      needs_view_index,
      absl::MakeSpan(selection_names),
      view_subset_hash,
      resolved_view_spec,
      resolved_selection,
      &selection_error_reason);
  if (!selection_identity_status.ok()) {
    if (!selection_error_reason.empty()) {
      span->SetAttribute("tc.selection.error_reason", std::string(selection_error_reason));
    }
    return selection_identity_status;
  }

  auto finalize_response = [&]() -> grpc::Status {
    if (no_lease) {
      resp.clear_mem_handle();
    }
    if (!resp.view_index_json().empty() && resp.view_index_bytes().empty()) {
      resp.set_view_index_bytes(resp.view_index_json());
    }

    auto layout_or = resolve_layout_json(resp, req, d_.engine);
    if (!layout_or.ok()) {
      return to_grpc_status(layout_or.status());
    }
    const bool prefer_view_plan = view_plan.has_value() && !view_plan->is_identity && resp.view_index_json().empty();
    const std::string* ticket_device_uuid = req.device_uuid().empty() ? nullptr : &req.device_uuid();
    absl::Status payload_status = populate_materialize_payloads(
        resp,
        *layout_or,
        resolved_selection.tensor_names(),
        req.device_uuid(),
        resolved_selection.view_subset_hash(),
        req.wait_for_completion(),
        req.replica_uuid(),
        ticket_device_uuid,
        view_plan,
        prefer_view_plan,
        /*fill_view_index_bytes=*/false);
    if (!payload_status.ok()) {
      return to_grpc_status(payload_status);
    }
    resp.mutable_resolved_selection()->CopyFrom(resolved_selection);
    rctx.mark_success();
    return Status::OK;
  };

  // Artifact LIP fast path: try cross-device consumption
  const bool view_requested = view_spec.has_value() || resolved_view_id.has_value();
  if (has_artifact && !view_requested && dev.type == DeviceType::GPU) {
    LipFastPathRequest lip_request{
        .artifact_id = resolved_artifact_id,
        .target_device_id = dev.ordinal,
        .replica_uuid = req.replica_uuid(),
        .effective_pid = effective_pid,
        .allow_pid_ref = loopback_peer,
        .lease_log_context = "LIP path",
    };
    auto lip_or = try_satisfy_lip_fast_path(
        d_.lip, d_.sessions, d_.refs, d_.handle_leases, d_.lifecycle, lip_request, *resp.mutable_mem_handle(), [&]() {
          record_lease_create_failed();
        });
    if (!lip_or.ok()) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(lip_or.status());
    }
    if (lip_or->satisfied) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
      resp.set_source(v2::MaterializationSource::MATERIALIZATION_SOURCE_LOCAL_REPLICA);
      span->SetAttribute("tc.store.source", static_cast<int64_t>(resp.source()));
      return finalize_response();
    }
  }

  // Engine-backed materialization
  store::loading::MaterializeHints hints;
  const bool prefer_direct_disk_for_local_import = local_import.has_value() && disk_source.has_value() &&
      retrieval_policy.allow_disk && retrieval_policy.preference != store::loading::SourcePreference::kPreferP2P &&
      (view_spec.has_value() || resolved_view_id.has_value() || !selection_names.empty());
  if (req.pinned_allocation_timeout_ms() > 0) {
    hints.pinned_timeout = std::chrono::milliseconds(req.pinned_allocation_timeout_ms());
  }
  const std::chrono::milliseconds request_budget = resolve_materialization_request_budget(rctx.server_context(), req);
  hints.request_budget = request_budget;
  hints.transport_wait_timeout = request_budget;
  hints.verify = verify_checksums ? store::loading::MaterializeHints::Verify::CHECKSUM
                                  : store::loading::MaterializeHints::Verify::NONE;
  apply_request_context_to_hints(request_context, &hints);
  if (!req.transport_request_id().empty()) {
    hints.transport_request_id = req.transport_request_id();
  }
  if (req.has_transport_scheduling_group()) {
    auto group_hint =
        resolve_transport_scheduling_group_hint(&req.transport_scheduling_group());
    if (group_hint.has_value()) {
      hints.transport_scheduling_group = std::move(*group_hint);
    }
  }
  if (prefer_direct_disk_for_local_import) {
    hints.set_retrieval_policy(
        store::loading::RetrievalPolicy{
            .preference = store::loading::SourcePreference::kPreferDisk,
            .allow_p2p = false,
            .allow_disk = request_context.retrieval_policy.allow_disk,
        });
  }
  if (prefer_direct_disk_for_local_import) {
    LOG(INFO) << "Using disk-first materialization for local import artifact_id=" << resolved_artifact_id
              << " (view_requested="
              << ((view_spec.has_value() || resolved_view_id.has_value() || !selection_names.empty()) ? 1 : 0)
              << ", selection_tensors=" << selection_names.size() << ")";
  }
  if (disk_source.has_value()) {
    hints.source_mutation_policy = store::loading::SourceMutationPolicy::kReadOnly;
  }
  hints.export_policy = to_hint_export_policy(req.export_policy());
  hints.need_view_data_hash = req.has_need_view_data_hash() ? req.need_view_data_hash() : true;
  if (has_artifact)
    hints.artifact_id = resolved_artifact_id;
  if (disk_metadata.has_value()) {
    hints.disk_metadata = std::move(*disk_metadata);
  }
  if (view_spec.has_value() || resolved_view_id.has_value() || !selection_names.empty()) {
    store::loading::VariantIdentity variant;
    if (has_artifact) {
      variant.canonical_artifact_id = resolved_artifact_id;
    }
    if (view_spec.has_value()) {
      variant.view_spec = view_spec;
    }
    if (canonical_index_json.has_value()) {
      variant.canonical_index_json = canonical_index_json;
    }
    if (view_plan.has_value()) {
      variant.cached_plan = view_plan;
    }
    if (resolved_view_id.has_value()) {
      variant.view_id = resolved_view_id;
    } else if (!selection_names.empty()) {
      auto subset_view_id = build_internal_subset_view_id(view_subset_hash);
      if (subset_view_id.has_value()) {
        variant.view_id = std::move(*subset_view_id);
      }
    }
    variant.placement = resolve_transform_placement(req.placement(), view_spec);
    LOG(INFO) << "materialize_replica.variant"
              << " artifact_id=" << resolved_artifact_id << " selection_names=" << selection_names.size()
              << " has_view_spec=" << (view_spec.has_value() ? 1 : 0)
              << " resolved_view_id=" << (resolved_view_id.has_value() ? *resolved_view_id : std::string())
              << " cached_plan=" << (view_plan.has_value() ? 1 : 0)
              << " cached_plan_identity=" << (view_plan.has_value() ? (view_plan->is_identity ? 1 : 0) : -1)
              << " cached_plan_view_bytes=" << (view_plan.has_value() ? view_plan->view_size_bytes : 0);
    hints.variant = std::move(variant);
  }

  if (cpu_target && selection_names.empty() && !resolved_view_id.has_value() && !view_spec.has_value()) {
    std::optional<uint64_t> required_stable_bytes;
    if (hints.disk_metadata.has_value() && hints.disk_metadata->logical_total_size.has_value()) {
      required_stable_bytes = hints.disk_metadata->logical_total_size;
    } else if (req.size_bytes() > 0) {
      required_stable_bytes = req.size_bytes();
    }
    if (required_stable_bytes.has_value()) {
      auto stable_budget_status = validate_impossible_cpu_stable_budget(d_.engine, *required_stable_bytes);
      if (!stable_budget_status.ok()) {
        resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
        return to_grpc_status(stable_budget_status);
      }
    }
  }

  const auto mode = (has_disk && !has_artifact && !prefer_disk) ? store::StoreEngine::MaterializeMode::LOAD_ONLY
                                                                : store::StoreEngine::MaterializeMode::AUTO;

  auto result = d_.engine.materialize_replica(dev, mode, hints, disk_source);
  if (!result.ok() && view_requested && fallback_artifact_id.has_value() && absl::IsNotFound(result.status())) {
    bool allow_reuse = false;
    if (d_.post_seal_policy.reuse_views_if_safe) {
      if (!post_seal_view_reuse_safe.has_value()) {
        if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
          resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
          return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
        }
        auto safe_or =
            check_post_seal_view_reuse_safe(*d_.global_store_client, *fallback_artifact_id, resolved_artifact_id);
        if (!safe_or.ok()) {
          LOG(WARNING) << "post-seal view reuse check failed for assembly=" << *fallback_artifact_id
                       << " mi2=" << resolved_artifact_id << ": " << safe_or.status();
          resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
          return to_grpc_status(safe_or.status());
        }
        post_seal_view_reuse_safe = *safe_or;
      }
      allow_reuse = *post_seal_view_reuse_safe;
      if (!allow_reuse) {
        LOG(WARNING) << "post-seal view reuse disabled: proof commitments mismatch for assembly="
                     << *fallback_artifact_id << " mi2=" << resolved_artifact_id;
      }
    }

    if (allow_reuse) {
      hints.artifact_id = *fallback_artifact_id;
      if (hints.variant.has_value()) {
        hints.variant->canonical_artifact_id = *fallback_artifact_id;
      }
      std::optional<store::loading::DiskSource> fallback_disk_source;
      if (disk_source_artifact_id.has_value() && *disk_source_artifact_id == *fallback_artifact_id) {
        fallback_disk_source = disk_source;
      }
      auto fallback_or = d_.engine.materialize_replica(dev, mode, hints, fallback_disk_source);
      if (fallback_or.ok()) {
        result = std::move(fallback_or);
      } else {
        resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
        return to_grpc_status(fallback_or.status());
      }
    }
  }
  if (!result.ok()) {
    auto retry_or = retry_materialize_from_shared_disk(
        result.status(),
        d_.engine,
        dev,
        store::StoreEngine::MaterializeMode::AUTO,
        hints,
        d_.global_store_client.get(),
        storage_path_,
        resolved_artifact_id,
        request_context,
        rctx.server_context(),
        normalized_disk_path,
        [&](const std::filesystem::path& ready_disk_path) -> absl::StatusOr<std::optional<store::loading::DiskSource>> {
          auto descriptor_or = load_descriptor_metadata(ready_disk_path);
          if (!descriptor_or.ok()) {
            return descriptor_or.status();
          }
          descriptor_meta = *descriptor_or;

          auto index_or = store::loader::read_from_artifact_dir(ready_disk_path, dev.ordinal);
          if (!index_or.ok()) {
            return index_or.status();
          }
          if (!descriptor_meta.found) {
            return absl::FailedPreconditionError("artifact_descriptor.json required for managed shared-disk loads");
          }
          auto validation_status =
              validate_descriptor_against_index(descriptor_meta, *index_or, /*verify_checksums=*/true);
          if (!validation_status.ok()) {
            return validation_status;
          }
          if (descriptor_meta.artifact_id.has_value() && resolved_artifact_id != *descriptor_meta.artifact_id) {
            return absl::FailedPreconditionError("artifact_id mismatch between request and artifact_descriptor.json");
          }

          disk_index = std::move(*index_or);
          store::loading::DiskMetadata metadata;
          metadata.descriptor_present = descriptor_meta.found;
          metadata.schema_version = descriptor_meta.schema_version;
          metadata.index_multihash = descriptor_meta.index_multihash;
          metadata.data_multihash = descriptor_meta.data_multihash;
          metadata.canonical_index_json = disk_index->canonical_index_json;
          if (disk_index->source_index_json.has_value()) {
            metadata.source_index_json = disk_index->source_index_json;
          }
          if (!disk_index->index_multihash.empty()) {
            metadata.index_multihash = disk_index->index_multihash;
          }
          if (disk_index->total_size_bytes > 0) {
            metadata.logical_total_size = disk_index->total_size_bytes;
          }
          if (disk_index->source_total_size_bytes > 0) {
            metadata.source_total_size_bytes = disk_index->source_total_size_bytes;
          }
          metadata.is_safetensors = disk_index->is_safetensors;

          hints.disk_metadata = std::move(metadata);
          hints.set_retrieval_policy(
              store::loading::RetrievalPolicy{
                  .preference = store::loading::SourcePreference::kPreferDisk,
                  .allow_p2p = false,
                  .allow_disk = true,
              });
          hints.source_mutation_policy = store::loading::SourceMutationPolicy::kReadOnly;

          std::optional<uint64_t> expected_size;
          if (hints.disk_metadata.has_value() && hints.disk_metadata->logical_total_size.has_value()) {
            expected_size = hints.disk_metadata->logical_total_size;
          }
          disk_source_artifact_id = resolved_artifact_id;
          return store::loading::DiskSource{
              .path = ready_disk_path,
              .expected_size = expected_size,
              .require_descriptor = true,
          };
        });
    result = std::move(retry_or);
    if (normalized_disk_path.has_value()) {
      resp.set_disk_path(normalized_disk_path->string());
      if (rctx.allow_high_card_attrs()) {
        span->SetAttribute("tc.disk.path", normalized_disk_path->string());
      }
    }
  }

  if (!result.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(result.status());
  }
  const auto& handle = *result;
  auto preflight_canonical_index_or = d_.engine.get_canonical_index_by_id(handle.replica_key.artifact_id);
  if (!preflight_canonical_index_or.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(preflight_canonical_index_or.status());
  }
  std::optional<store::loading::DiskSource> preflight_disk_source;
  if (disk_source_artifact_id.has_value() && *disk_source_artifact_id == handle.replica_key.artifact_id) {
    preflight_disk_source = disk_source;
  }
  auto preflight_or = serving_artifact_manifest::preflight_serving_artifact(
      &d_.engine,
      serving_artifact_manifest::build_preflight_request(
          handle.replica_key.artifact_id,
          *preflight_canonical_index_or,
          preflight_disk_source,
          req.has_serving_artifact_policy() ? &req.serving_artifact_policy() : nullptr));
  if (!preflight_or.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(preflight_or.status());
  }
  resp.set_source(to_proto_source(handle.source));
  span->SetAttribute("tc.store.source", static_cast<int64_t>(resp.source()));
  if (normalized_disk_path.has_value()) {
    resp.set_disk_path(normalized_disk_path->string());
  }
  if (cpu_target && !handle.cpu_memfd_region.has_value()) {
    LOG(WARNING) << "MaterializationController: cpu_target but engine handle missing cpu_memfd_region for key="
                 << handle.replica_key << " cpu_state=" << static_cast<int>(handle.cpu_state)
                 << " gpu_state=" << static_cast<int>(handle.gpu_state);
  }
  auto bind_status = bind_materialized_handle(
      d_.engine,
      d_.sessions,
      d_.refs,
      d_.lifecycle,
      d_.handle_leases,
      handle,
      req.replica_uuid(),
      effective_pid,
      loopback_peer,
      cpu_target,
      "engine path",
      *resp.mutable_mem_handle());
  if (!bind_status.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(bind_status);
  }
  resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  if (handle.view_index_json.has_value()) {
    resp.set_view_index_json(*handle.view_index_json);
  }
  if (resp.view_index_json().empty() && normalized_disk_path.has_value()) {
    // Prefer disk-local canonical index to avoid Global Store dependency when the client provides a disk_path,
    // even if the engine serves the request from an already-loaded local replica.
    if (disk_index.has_value()) {
      resp.set_view_index_json(disk_index->canonical_index_json);
    } else {
      auto local_index_or = store::loader::read_from_artifact_dir(*normalized_disk_path, dev.ordinal);
      if (local_index_or.ok()) {
        resp.set_view_index_json(local_index_or->canonical_index_json);
        VLOG(1) << "MaterializationController: filled view_index_json from disk for artifact_id="
                << handle.replica_key.artifact_id;
      } else {
        LOG(WARNING) << "Failed to read canonical index from disk for artifact_id=" << handle.replica_key.artifact_id
                     << ": " << local_index_or.status();
      }
    }
  }
  if (resp.view_index_json().empty() && handle.source == store::loading::MaterializationSource::kDisk) {
    auto index_or = d_.engine.get_canonical_index_by_id(handle.replica_key.artifact_id);
    if (index_or.ok()) {
      resp.set_view_index_json(*index_or);
      VLOG(1) << "MaterializationController: filled view_index_json from engine for disk artifact_id="
              << handle.replica_key.artifact_id;
    } else {
      LOG(WARNING) << "Failed to fetch canonical index for disk materialization response: " << index_or.status();
    }
  }
  if (handle.view_data_hash.has_value()) {
    resp.set_view_data_hash(*handle.view_data_hash);
  }
  return finalize_response();
}

} // namespace tensorcast::daemon
