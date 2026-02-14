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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "opentelemetry/metrics/provider.h"

#include "core/store/device_registry.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "daemon/service/controllers/materialization_index_source_utils.h"
#include "daemon/service/controllers/materialization_payload_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/service/controllers/materialization_post_seal_utils.h"
#include "daemon/service/controllers/materialization_replica_handle_utils.h"
#include "daemon/service/controllers/materialization_request_common_utils.h"
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
using materialization_payload::resolve_layout_json_by_key;
using materialization_policy::compute_view_id_from_spec;
using materialization_policy::convert_view_spec;
using materialization_policy::resolve_source_policy;
using materialization_policy::resolve_transform_placement;
using materialization_policy::ResolvedSourcePolicy;
using materialization_policy::to_hint_export_policy;
using materialization_policy::to_hint_preference;
using materialization_policy::validate_source_policy;
using materialization_post_seal::check_post_seal_view_reuse_safe;
using materialization_replica_handle::bind_replica_handle_for_response;
using materialization_request_common::LeaseContext;
using materialization_request_common::LipFastPathRequest;
using materialization_request_common::materialize_with_shared_disk_retry;
using materialization_request_common::resolve_artifact_and_disk_source;
using materialization_request_common::try_satisfy_lip_fast_path;
using materialization_request_common::validate_and_compute_lease_context;
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

absl::StatusOr<ResolvedSourcePolicy> resolve_and_validate_effective_policy(
    const v2::SourcePolicy* source_policy,
    v2::SourcePreference preference) {
  ResolvedSourcePolicy effective_policy = resolve_source_policy(source_policy, preference);
  const absl::Status policy_status = validate_source_policy(effective_policy);
  if (!policy_status.ok()) {
    return policy_status;
  }
  return effective_policy;
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
    int wait_for_shared_disk_ms,
    bool allow_disk,
    const grpc::ServerContext& server_context,
    std::optional<std::filesystem::path>& normalized_disk_path,
    const materialization_request_common::PrepareRetryDiskSourceFn& prepare_retry_disk_source) {
  return materialize_with_shared_disk_retry(
      initial_status,
      global_store_client,
      storage_path,
      resolved_artifact_id,
      wait_for_shared_disk_ms,
      allow_disk,
      server_context,
      normalized_disk_path,
      [&](const std::optional<store::loading::DiskSource>& retry_disk_source) {
        return engine.materialize_replica(dev, mode, hints, retry_disk_source);
      },
      prepare_retry_disk_source);
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
  auto policy_or =
      resolve_and_validate_effective_policy(req.has_source_policy() ? &req.source_policy() : nullptr, req.preference());
  if (!policy_or.ok()) {
    resp.set_status(v2::MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(policy_or.status());
  }
  ResolvedSourcePolicy effective_policy = *policy_or;
  const bool prefer_disk = effective_policy.preference == v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK;
  bool verify_checksums = true;

  span->SetAttribute("tc.store.verify_checksums", verify_checksums);

  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.device.uuid", req.device_uuid());
  }
  span->SetAttribute("tc.size.bytes", static_cast<int64_t>(req.size_bytes()));
  span->SetAttribute("tc.store.preference", static_cast<int64_t>(effective_policy.preference));
  span->SetAttribute("tc.store.allow_p2p", effective_policy.allow_p2p);
  span->SetAttribute("tc.store.allow_disk", effective_policy.allow_disk);

  using v2::MaterializeReplicaStatus;
  if (d_.shutdown_signal.is_shutting_down()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  const bool request_has_artifact = req.has_artifact_id() && !req.artifact_id().empty();
  if (!request_has_artifact) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }

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
      req.artifact_id(),
      effective_policy.allow_disk,
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

  // View identity handling
  std::optional<ViewSpec> view_spec;
  std::optional<store::loader::ViewPlan> view_plan;
  std::optional<std::string> canonical_index_json;
  std::optional<std::string> request_view_id;
  const std::string& index_source_artifact_id =
      fallback_artifact_id.has_value() ? *fallback_artifact_id : resolved_artifact_id;

  switch (req.view_identity_case()) {
    case v2::MaterializeReplicaRequest::kView: {
      if (!has_artifact && !has_disk) {
        return {StatusCode::INVALID_ARGUMENT, "view spec requires artifact_id or disk_path for canonical planning"};
      }
      auto spec_or = convert_view_spec(req.view());
      if (!spec_or.ok()) {
        return to_grpc_status(spec_or.status());
      }
      view_spec = std::move(*spec_or);
      auto read_canonical_from_disk = [&]() -> absl::StatusOr<std::string> {
        if (!normalized_disk_path.has_value()) {
          return absl::FailedPreconditionError("disk source path required for disk-backed view planning");
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

      const bool prefer_disk_index = has_disk && (!gs_connected || prefer_disk);

      absl::StatusOr<std::string> index_or = prefer_disk_index
          ? read_canonical_from_disk()
          : d_.engine.get_canonical_index_by_id(index_source_artifact_id);
      if (!index_or.ok() && has_disk && !prefer_disk_index) {
        // If the canonical lookup fails (e.g., local-only daemon without Global Store), use the disk index when
        // present.
        auto disk_or = read_canonical_from_disk();
        if (disk_or.ok()) {
          index_or = std::move(disk_or);
        }
      }
      if (!index_or.ok()) {
        return to_grpc_status(index_or.status());
      }
      canonical_index_json = std::move(index_or).value();
      auto plan_or = store::StoreEngine::compute_view_plan(*canonical_index_json, *view_spec);
      if (!plan_or.ok()) {
        return to_grpc_status(plan_or.status());
      }
      if (!plan_or->is_identity) {
        view_plan = *plan_or;
        auto view_id_or = compute_view_id_from_spec(req.view(), *canonical_index_json);
        if (!view_id_or.ok()) {
          return to_grpc_status(view_id_or.status());
        }
        request_view_id = std::move(*view_id_or);
      } else {
        // Identity view collapses to canonical path
        view_spec.reset();
        view_plan.reset();
        canonical_index_json.reset();
      }
      break;
    }
    case v2::MaterializeReplicaRequest::kViewId: {
      if (!req.view_id().empty()) {
        if (!has_artifact) {
          return {StatusCode::INVALID_ARGUMENT, "view_id requires artifact_id for routing"};
        }
        request_view_id = req.view_id();
      }
      break;
    }
    case v2::MaterializeReplicaRequest::VIEW_IDENTITY_NOT_SET:
      break;
  }
  if (request_view_id.has_value()) {
    span->SetAttribute("tc.view.id", *request_view_id);
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
    const bool prefer_view_plan =
        req.view_identity_case() == v2::MaterializeReplicaRequest::kView && resp.view_index_json().empty();
    const std::string* ticket_device_uuid = req.device_uuid().empty() ? nullptr : &req.device_uuid();
    absl::Status payload_status = populate_materialize_payloads(
        resp,
        *layout_or,
        req.tensor_names(),
        req.device_uuid(),
        req.view_subset_hash(),
        req.wait_for_completion(),
        req.replica_uuid(),
        ticket_device_uuid,
        view_plan,
        prefer_view_plan,
        /*fill_view_index_bytes=*/false);
    if (!payload_status.ok()) {
      return to_grpc_status(payload_status);
    }
    rctx.mark_success();
    return Status::OK;
  };

  // Artifact LIP fast path: try cross-device consumption
  const bool view_requested = view_spec.has_value() || request_view_id.has_value();
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
  if (req.pinned_allocation_timeout_ms() > 0) {
    hints.pinned_timeout = std::chrono::milliseconds(req.pinned_allocation_timeout_ms());
  }
  hints.verify = verify_checksums ? store::loading::MaterializeHints::Verify::CHECKSUM
                                  : store::loading::MaterializeHints::Verify::NONE;
  hints.source_preference = to_hint_preference(effective_policy.preference);
  hints.allow_p2p = effective_policy.allow_p2p;
  hints.allow_disk = effective_policy.allow_disk;
  if (disk_source.has_value()) {
    hints.source_mutation_policy = store::loading::SourceMutationPolicy::kReadOnly;
  }
  hints.export_policy = to_hint_export_policy(req.export_policy());
  if (has_artifact)
    hints.artifact_id = resolved_artifact_id;
  if (disk_metadata.has_value()) {
    hints.disk_metadata = std::move(*disk_metadata);
  }
  if (view_spec.has_value() || request_view_id.has_value()) {
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
    if (request_view_id.has_value()) {
      variant.view_id = request_view_id;
    }
    variant.placement = resolve_transform_placement(req.placement(), view_spec);
    hints.variant = std::move(variant);
  }
  const auto mode = (has_disk && !has_artifact && !prefer_disk) ? store::StoreEngine::MaterializeMode::LOAD_ONLY
                                                                : store::StoreEngine::MaterializeMode::AUTO;

  auto result = d_.engine.materialize_replica(dev, mode, hints, disk_source);
  if (!result.ok() && view_requested && fallback_artifact_id.has_value() && absl::IsNotFound(result.status())) {
    bool allow_reuse = false;
    if (d_.post_seal_policy.reuse_views_if_safe) {
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
      allow_reuse = *safe_or;
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
        static_cast<int>(req.wait_for_shared_disk_ms()),
        effective_policy.allow_disk,
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
          hints.source_preference = to_hint_preference(v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
          hints.allow_p2p = false;
          hints.allow_disk = true;
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

grpc::Status ReplicaMaterializationService::materialize_by_key(
    RpcContext& rctx,
    const v2::MaterializeByKeyRequest& req,
    v2::MaterializeByKeyResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req.key());
  const v2::DeviceType requested_type = (req.target_device_type() == v2::DeviceType::DEVICE_TYPE_CPU)
      ? v2::DeviceType::DEVICE_TYPE_CPU
      : v2::DeviceType::DEVICE_TYPE_GPU;
  const bool cpu_target = requested_type == v2::DeviceType::DEVICE_TYPE_CPU;
  span->SetAttribute("tc.device.type", static_cast<int64_t>(requested_type));
  using v2::MaterializeReplicaStatus;
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

  auto policy_or =
      resolve_and_validate_effective_policy(req.has_source_policy() ? &req.source_policy() : nullptr, req.preference());
  if (!policy_or.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(policy_or.status());
  }
  ResolvedSourcePolicy effective_policy = *policy_or;
  span->SetAttribute("tc.store.preference", static_cast<int64_t>(effective_policy.preference));
  span->SetAttribute("tc.store.allow_p2p", effective_policy.allow_p2p);
  span->SetAttribute("tc.store.allow_disk", effective_policy.allow_disk);

  if (d_.shutdown_signal.is_shutting_down()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req.key().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "key is required"};
  }

  auto finalize_response = [&]() -> grpc::Status {
    if (no_lease) {
      resp.clear_mem_handle();
    }
    auto layout_or = resolve_layout_json_by_key(resp, d_.engine);
    if (!layout_or.ok()) {
      return to_grpc_status(layout_or.status());
    }
    absl::Status payload_status = populate_materialize_payloads(
        resp,
        *layout_or,
        req.tensor_names(),
        /*device_uuid=*/"",
        req.view_subset_hash(),
        req.wait_for_completion(),
        req.replica_uuid(),
        /*ticket_device_uuid=*/nullptr,
        /*view_plan=*/std::nullopt,
        /*prefer_view_plan=*/false,
        /*fill_view_index_bytes=*/true);
    if (!payload_status.ok()) {
      return to_grpc_status(payload_status);
    }
    rctx.mark_success();
    return Status::OK;
  };

  auto mapping_or = d_.engine.resolve_key_mapping(req.key());
  if (!mapping_or.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(mapping_or.status());
  }
  const auto& mapping = *mapping_or;

  // Engine path device resolution (also used for local short-circuit checks).
  store::DeviceKey dev;
  if (cpu_target) {
    dev = d_.devices.From(v2::DeviceType::DEVICE_TYPE_CPU, /*uuid=*/"", /*ordinal_hint=*/std::nullopt);
  } else {
    if (req.device_id() < 0 || req.device_id() >= d_.engine.get_num_gpus()) {
      return {StatusCode::INVALID_ARGUMENT, "invalid device_id"};
    }
    dev = store::DeviceRegistry::instance().gpu_key(req.device_id());
  }

  std::string resolved_artifact_id = mapping.artifact_id;
  store::loading::MaterializeHints hints;
  if (req.pinned_allocation_timeout_ms() > 0) {
    hints.pinned_timeout = std::chrono::milliseconds(req.pinned_allocation_timeout_ms());
  }
  hints.artifact_id = resolved_artifact_id;
  hints.source_preference = to_hint_preference(effective_policy.preference);
  hints.allow_p2p = effective_policy.allow_p2p;
  hints.allow_disk = effective_policy.allow_disk;

  auto finalize_materialized = [&](const store::loading::ReplicaHandle& handle, std::string_view used_disk_path) {
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
        "engine by-key",
        *resp.mutable_mem_handle());
    if (!bind_status.ok()) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(bind_status);
    }
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
    resp.set_artifact_id(resolved_artifact_id);
    resp.set_used_disk_path(std::string(used_disk_path));
    resp.set_source(to_proto_source(handle.source));
    span->SetAttribute("tc.store.source", static_cast<int64_t>(resp.source()));
    return finalize_response();
  };

  // Local-first short-circuit: if local CPU/GPU residency is already available,
  // try a no-remote/no-disk materialization before querying Global Store.
  store::loading::ReplicaKey cpu_lookup{
      .artifact_id = resolved_artifact_id,
      .device = store::DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      .replica = 0,
  };
  const bool local_cpu_loaded =
      d_.engine.get_replica_state(cpu_lookup, DeviceType::CPU) == store::replica::MemoryState::LOADED;
  bool local_gpu_loaded = false;
  if (!cpu_target) {
    store::loading::ReplicaKey gpu_lookup{
        .artifact_id = resolved_artifact_id,
        .device = dev,
        .replica = 0,
    };
    local_gpu_loaded = d_.engine.get_replica_state(gpu_lookup, DeviceType::GPU) == store::replica::MemoryState::LOADED;
  }
  if (local_cpu_loaded || local_gpu_loaded) {
    auto local_hints = hints;
    local_hints.allow_p2p = false;
    local_hints.allow_disk = false;
    auto local_result = d_.engine.materialize_replica(dev, store::StoreEngine::MaterializeMode::AUTO, local_hints);
    if (local_result.ok()) {
      span->SetAttribute("tc.store.local_short_circuit", true);
      span->SetAttribute("tc.store.gs_connected", false);
      return finalize_materialized(*local_result, /*used_disk_path=*/"");
    }
    VLOG(1) << "Local short-circuit miss for key=" << req.key() << ": " << local_result.status();
  }

  auto artifact_resolution_or = resolve_artifact_and_disk_source(
      d_.global_store_client,
      /*disk_imports=*/nullptr,
      storage_path_,
      mapping.artifact_id,
      effective_policy.allow_disk,
      /*allow_local_import_fallback=*/false,
      loopback_peer);
  if (!artifact_resolution_or.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(artifact_resolution_or.status());
  }
  auto artifact_resolution = std::move(*artifact_resolution_or);
  resolved_artifact_id = std::move(artifact_resolution.resolved_artifact_id);
  hints.artifact_id = resolved_artifact_id;
  std::optional<std::string> bound_artifact_id = std::move(artifact_resolution.bound_artifact_id);
  if (bound_artifact_id.has_value()) {
    span->SetAttribute("tc.artifact.bound", *bound_artifact_id);
  }
  span->SetAttribute("tc.artifact.id", resolved_artifact_id);
  std::optional<std::filesystem::path> normalized_disk_path = std::move(artifact_resolution.normalized_disk_path);
  std::optional<store::loading::DiskSource> disk_source = std::move(artifact_resolution.disk_source);
  if (disk_source.has_value()) {
    hints.source_mutation_policy = store::loading::SourceMutationPolicy::kReadOnly;
  }
  std::string used_disk_path = normalized_disk_path.has_value() ? normalized_disk_path->string() : std::string();
  span->SetAttribute("tc.store.gs_connected", artifact_resolution.gs_connected);

  // Try LIP fast path first (GPU only)
  if (!cpu_target) {
    LipFastPathRequest lip_request{
        .artifact_id = resolved_artifact_id,
        .target_device_id = req.device_id(),
        .replica_uuid = req.replica_uuid(),
        .effective_pid = effective_pid,
        .allow_pid_ref = loopback_peer,
        .lease_log_context = "LIP by-key",
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
      resp.set_artifact_id(mapping.artifact_id);
      resp.set_used_disk_path(used_disk_path);
      resp.set_source(v2::MaterializationSource::MATERIALIZATION_SOURCE_LOCAL_REPLICA);
      span->SetAttribute("tc.store.source", static_cast<int64_t>(resp.source()));
      return finalize_response();
    }
  }

  // Engine path
  auto result = d_.engine.materialize_replica(dev, store::StoreEngine::MaterializeMode::AUTO, hints, disk_source);
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
        static_cast<int>(req.wait_for_shared_disk_ms()),
        effective_policy.allow_disk,
        rctx.server_context(),
        normalized_disk_path,
        [&](const std::filesystem::path& ready_disk_path) -> absl::StatusOr<std::optional<store::loading::DiskSource>> {
          hints.source_preference = to_hint_preference(v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
          hints.allow_p2p = false;
          hints.allow_disk = true;
          hints.source_mutation_policy = store::loading::SourceMutationPolicy::kReadOnly;
          return store::loading::DiskSource{
              .path = ready_disk_path,
              .expected_size = std::nullopt,
              .require_descriptor = true,
          };
        });
    result = std::move(retry_or);
    if (normalized_disk_path.has_value()) {
      used_disk_path = normalized_disk_path->string();
    }
  }
  if (!result.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(result.status());
  }
  return finalize_materialized(*result, used_disk_path);
}

} // namespace tensorcast::daemon
