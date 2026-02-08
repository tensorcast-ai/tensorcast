// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_controller.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/protobuf/util/time_util.h"
#include "gsl/pointers"
#include "opentelemetry/metrics/provider.h"

#include "core/common/artifact_hash.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/materialization/dataplane/sources/memory_source.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "daemon/service/controllers/materialization_index_source_utils.h"
#include "daemon/service/controllers/materialization_payload_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/service/controllers/materialization_replica_handle_utils.h"
#include "daemon/service/controllers/materialization_request_common_utils.h"
#include "daemon/util/grpc_peer_utils.h"
#include "daemon/util/status_utils.h"
#include "folly/futures/Future.h"
#include "nlohmann/json.hpp"

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
using materialization_policy::spec_includes_transpose;
using materialization_policy::to_hint_export_policy;
using materialization_policy::to_hint_preference;
using materialization_policy::validate_source_policy;
using materialization_replica_handle::bind_replica_handle_for_response;
using materialization_request_common::LeaseContext;
using materialization_request_common::LipFastPathRequest;
using materialization_request_common::materialize_with_shared_disk_retry;
using materialization_request_common::resolve_artifact_and_disk_source;
using materialization_request_common::try_satisfy_lip_fast_path;
using materialization_request_common::validate_and_compute_lease_context;
using store::loader::ViewSpec;

absl::StatusOr<bool> check_post_seal_view_reuse_safe(
    store::components::IGlobalStoreClient& client,
    std::string_view assembly_id,
    std::string_view mi2_id);
using store::loading::MaterializationSource;

void record_lease_create_failed() {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto ctr = meter->CreateDoubleCounter("tc_lease_create_failed_total");
    ctr->Add(1.0);
  } catch (...) {
  }
}

class OperationLeaseGuard {
 public:
  OperationLeaseGuard(
      std::shared_ptr<store::components::IGlobalStoreClient> client,
      std::string lease_token,
      std::string operation_id)
      : client_(std::move(client)), lease_token_(std::move(lease_token)), operation_id_(std::move(operation_id)) {}

  ~OperationLeaseGuard() {
    release();
  }

  void release() {
    if (released_ || client_ == nullptr || lease_token_.empty()) {
      released_ = true;
      return;
    }
    tensorcast::operation::v1::ReleaseOperationLeaseRequest release_req;
    release_req.set_lease_token(lease_token_);
    auto release_or = client_->release_operation_lease(release_req);
    if (!release_or.ok()) {
      LOG(WARNING) << "release_operation_lease failed for op=" << operation_id_ << ": " << release_or.status();
    }
    released_ = true;
  }

 private:
  std::shared_ptr<store::components::IGlobalStoreClient> client_;
  std::string lease_token_;
  std::string operation_id_;
  bool released_{false};
};

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

} // namespace

MaterializationController::MaterializationController(Dep d)
    : d_(std::move(d)),
      seal_operation_tracker_(std::make_shared<SealOperationTracker>()),
      disk_artifact_service_(
          DiskArtifactService::Dep{
              .engine = d_.engine,
              .disk_imports = d_.disk_imports,
              .shutdown_signal = d_.shutdown_signal,
              .storage_path = d_.storage_path,
          }),
      replica_lifecycle_service_(
          ReplicaLifecycleService::Dep{
              .engine = d_.engine,
              .refs = d_.refs,
              .sessions = d_.sessions,
              .lifecycle = d_.lifecycle,
              .devices = d_.devices,
          }),
      target_materialization_service_(
          TargetMaterializationService::Dep{
              .engine = d_.engine,
              .lip_manager = d_.lip_manager,
              .devices = d_.devices,
              .regions = d_.regions,
              .disk_imports = d_.disk_imports,
              .shutdown_signal = d_.shutdown_signal,
              .identity = d_.identity,
              .global_store_client = d_.global_store_client,
              .capability_tokens = d_.capability_tokens,
              .external_target_verification_enabled = d_.external_target_verification_enabled,
              .storage_path = d_.storage_path,
          }) {
  if (!d_.storage_path.empty()) {
    std::error_code ec;
    storage_path_ = std::filesystem::weakly_canonical(d_.storage_path, ec);
    if (ec) {
      ec.clear();
      storage_path_ = d_.storage_path.lexically_normal();
    }
  }
}

TargetWriteRegistry::Record MaterializationController::insert_target_write_for_testing(
    TargetWriteRegistry::Record record) {
  return target_materialization_service_.insert_target_write_for_testing(std::move(record));
}

grpc::Status MaterializationController::materialize_replica(
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
  std::optional<LocalDiskImportCatalog::Entry> local_import = std::move(artifact_resolution.local_import);
  std::optional<store::loading::DiskSource> disk_source = std::move(artifact_resolution.disk_source);

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
    auto retry_or = materialize_with_shared_disk_retry(
        result.status(),
        d_.global_store_client.get(),
        storage_path_,
        resolved_artifact_id,
        static_cast<int>(req.wait_for_shared_disk_ms()),
        effective_policy.allow_disk,
        rctx.server_context(),
        normalized_disk_path,
        [&](const std::optional<store::loading::DiskSource>& retry_disk_source) {
          return d_.engine.materialize_replica(
              dev, store::StoreEngine::MaterializeMode::AUTO, hints, retry_disk_source);
        },
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
  auto bind_status = bind_replica_handle_for_response(
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
      [&]() { record_lease_create_failed(); },
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

grpc::Status MaterializationController::materialize_by_key(
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
  std::string resolved_artifact_id = std::move(artifact_resolution.resolved_artifact_id);
  std::optional<std::string> bound_artifact_id = std::move(artifact_resolution.bound_artifact_id);
  if (bound_artifact_id.has_value()) {
    span->SetAttribute("tc.artifact.bound", *bound_artifact_id);
  }
  span->SetAttribute("tc.artifact.id", resolved_artifact_id);
  std::optional<std::filesystem::path> normalized_disk_path = std::move(artifact_resolution.normalized_disk_path);
  std::optional<store::loading::DiskSource> disk_source = std::move(artifact_resolution.disk_source);
  std::string used_disk_path = normalized_disk_path.has_value() ? normalized_disk_path->string() : std::string();

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
  store::DeviceKey dev;
  if (cpu_target) {
    dev = d_.devices.From(v2::DeviceType::DEVICE_TYPE_CPU, /*uuid=*/"", /*ordinal_hint=*/std::nullopt);
  } else {
    // Validate device_id
    if (req.device_id() < 0 || req.device_id() >= d_.engine.get_num_gpus()) {
      return {StatusCode::INVALID_ARGUMENT, "invalid device_id"};
    }
    dev = store::DeviceRegistry::instance().gpu_key(req.device_id());
  }
  store::loading::MaterializeHints hints;
  if (req.pinned_allocation_timeout_ms() > 0) {
    hints.pinned_timeout = std::chrono::milliseconds(req.pinned_allocation_timeout_ms());
  }
  hints.artifact_id = resolved_artifact_id;
  hints.source_preference = to_hint_preference(effective_policy.preference);
  hints.allow_p2p = effective_policy.allow_p2p;
  hints.allow_disk = effective_policy.allow_disk;

  auto result = d_.engine.materialize_replica(dev, store::StoreEngine::MaterializeMode::AUTO, hints, disk_source);
  if (!result.ok()) {
    auto retry_or = materialize_with_shared_disk_retry(
        result.status(),
        d_.global_store_client.get(),
        storage_path_,
        resolved_artifact_id,
        static_cast<int>(req.wait_for_shared_disk_ms()),
        effective_policy.allow_disk,
        rctx.server_context(),
        normalized_disk_path,
        [&](const std::optional<store::loading::DiskSource>& retry_disk_source) {
          return d_.engine.materialize_replica(
              dev, store::StoreEngine::MaterializeMode::AUTO, hints, retry_disk_source);
        },
        [&](const std::filesystem::path& ready_disk_path) -> absl::StatusOr<std::optional<store::loading::DiskSource>> {
          hints.source_preference = to_hint_preference(v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
          hints.allow_p2p = false;
          hints.allow_disk = true;
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
  const auto& handle = *result;
  auto bind_status = bind_replica_handle_for_response(
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
      [&]() { record_lease_create_failed(); },
      *resp.mutable_mem_handle());
  if (!bind_status.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(bind_status);
  }
  resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  resp.set_artifact_id(resolved_artifact_id);
  resp.set_used_disk_path(used_disk_path);
  resp.set_source(to_proto_source(handle.source));
  span->SetAttribute("tc.store.source", static_cast<int64_t>(resp.source()));
  return finalize_response();
}

grpc::Status MaterializationController::materialize_into_target(
    RpcContext& rctx,
    const v2::MaterializeIntoTargetRequest& req,
    v2::MaterializeIntoTargetResponse& resp) {
  return target_materialization_service_.materialize_into_target(rctx, req, resp);
}

grpc::Status MaterializationController::materialize_into_mapped_target(
    RpcContext& rctx,
    const v2::MaterializeIntoMappedTargetRequest& req,
    v2::MaterializeIntoTargetResponse& resp) {
  return target_materialization_service_.materialize_into_mapped_target(rctx, req, resp);
}

grpc::Status MaterializationController::publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::PublishTargetReplicaResponse& resp) {
  return target_materialization_service_.publish_target_replica(rctx, req, resp);
}

grpc::Status MaterializationController::resolve_artifact_from_disk(
    RpcContext& rctx,
    const v2::ResolveArtifactFromDiskRequest& req,
    v2::ResolveArtifactFromDiskResponse& resp) {
  return disk_artifact_service_.resolve_artifact_from_disk(rctx, req, resp);
}

grpc::Status MaterializationController::get_artifact_index_by_id(
    RpcContext& rctx,
    const v2::GetArtifactIndexByIdRequest& req,
    v2::GetArtifactIndexByIdResponse& resp) {
  return disk_artifact_service_.get_artifact_index_by_id(rctx, req, resp);
}

grpc::Status MaterializationController::seal_assembly(
    RpcContext& rctx,
    const v2::SealAssemblyRequest& req,
    v2::SealAssemblyResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.assembly_id());

  if (req.assembly_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "assembly_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  auto result_or = d_.engine.seal_assembly(req.assembly_id(), req.publish_canonical());
  if (!result_or.ok()) {
    return to_grpc_status(result_or.status());
  }
  const auto& result = *result_or;
  resp.set_sealed_artifact_id(result.sealed_artifact_id);
  resp.set_already_sealed(result.already_sealed);
  auto* desc = resp.mutable_descriptor_();
  desc->set_artifact_id(result.sealed_artifact_id);
  if (!result.index_multihash.empty()) {
    desc->set_index_multihash(result.index_multihash);
  }
  if (!result.data_multihash.empty()) {
    desc->set_data_multihash(result.data_multihash);
  }
  if (!result.schema_version.empty()) {
    desc->set_schema_version(result.schema_version);
  }
  if (!result.encoding.empty()) {
    desc->set_encoding(result.encoding);
  }
  if (result.total_size > 0) {
    desc->set_total_size(result.total_size);
  }
  desc->set_id_kind(tensorcast::common::v1::ArtifactIdKind::ARTIFACT_ID_KIND_MI2);
  rctx.mark_success();
  return Status::OK;
}

namespace {

std::string compute_seal_operation_id(std::string_view assembly_id) {
  const std::string payload = absl::StrCat("seal_assembly:", assembly_id);
  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  const std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(bytes);
  return tensorcast::common::multibase_multihash_sha256(digest);
}

std::string owner_id_for_operation(const WorkerIdentityStore& identity) {
  auto daemon_id = identity.daemon_id();
  if (!daemon_id.empty()) {
    return daemon_id;
  }
  auto worker_id = identity.worker_id();
  if (!worker_id.empty()) {
    return worker_id;
  }
  return "unknown";
}

bool retryable_status(const absl::Status& st) {
  return absl::IsUnavailable(st) || absl::IsDeadlineExceeded(st) || absl::IsAborted(st) || absl::IsInternal(st) ||
      absl::IsUnknown(st);
}

constexpr uint64_t kProofChunkBytesV1 = 4ULL * 1024 * 1024;

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

std::vector<uint8_t> compute_view_meta_digest(const store::components::ViewInfo& view) {
  std::vector<store::components::CanonicalRange> ranges = view.canonical_ranges;
  std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
    if (a.offset != b.offset) {
      return a.offset < b.offset;
    }
    return a.length < b.length;
  });

  std::string payload;
  payload.reserve(256 + ranges.size() * 32);
  absl::StrAppend(&payload, "view_id=", view.view_id, ";");
  absl::StrAppend(&payload, "view_data_hash=", view.view_data_hash.value_or(""), ";");
  absl::StrAppend(&payload, "view_size_bytes=", view.view_size_bytes, ";");
  absl::StrAppend(&payload, "canonical_size_bytes=", view.canonical_size_bytes, ";");
  absl::StrAppend(&payload, "canonical_bytes_covered=", view.canonical_bytes_covered, ";");
  for (const auto& range : ranges) {
    absl::StrAppend(&payload, range.offset, ":", range.length, ";");
  }

  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  return tensorcast::common::sha256_digest_bytes(bytes);
}

absl::StatusOr<bool> check_post_seal_view_reuse_safe(
    store::components::IGlobalStoreClient& client,
    std::string_view assembly_id,
    std::string_view mi2_id) {
  if (assembly_id.empty() || mi2_id.empty()) {
    return absl::InvalidArgumentError("check_post_seal_view_reuse_safe requires assembly_id and mi2_id");
  }

  auto layouts_or = client.list_artifact_layouts(mi2_id);
  if (!layouts_or.ok()) {
    return layouts_or.status();
  }
  if (layouts_or->empty()) {
    return true;
  }

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> tensors_by_schema;
  for (const auto& layout_id : *layouts_or) {
    if (layout_id.empty()) {
      continue;
    }
    auto spec_or = client.get_layout_spec(layout_id);
    if (!spec_or.ok()) {
      return spec_or.status();
    }
    const auto& layout_spec = spec_or->layout();
    const std::string schema_version = layout_spec.proof_schema_version();
    for (const auto& entry : layout_spec.tensors()) {
      if (entry.second.overlap_mode() == tensorcast::layout::v1::OVERLAP_MODE_REPLICATE_EQUAL) {
        if (schema_version.empty()) {
          return absl::FailedPreconditionError("proof_schema_version required for replicated tensors");
        }
        tensors_by_schema[schema_version].insert(entry.first);
      }
    }
  }

  if (tensors_by_schema.empty()) {
    return true;
  }

  for (const auto& [schema_version, tensors] : tensors_by_schema) {
    if (tensors.empty()) {
      continue;
    }
    tensorcast::global_store::v1::CheckProofCommitmentsMatchRequest req;
    req.set_assembly_id(std::string(assembly_id));
    req.set_mi2_id(std::string(mi2_id));
    req.set_proof_schema_version(schema_version);
    for (const auto& name : tensors) {
      if (!name.empty()) {
        req.add_tensor_names(name);
      }
    }
    auto resp_or = client.check_proof_commitments_match(req);
    if (!resp_or.ok()) {
      return resp_or.status();
    }
    if (!resp_or->match()) {
      return false;
    }
  }
  return true;
}

} // namespace

grpc::Status MaterializationController::start_seal_assembly(
    RpcContext& rctx,
    const v2::StartSealAssemblyRequest& req,
    v2::StartSealAssemblyResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.assembly_id());

  if (req.assembly_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "assembly_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  const std::string operation_id = compute_seal_operation_id(req.assembly_id());
  auto* out_ref = resp.mutable_operation();
  out_ref->set_operation_id(operation_id);
  out_ref->set_kind("seal_assembly");
  out_ref->set_target_artifact_id(req.assembly_id());

  tensorcast::operation::v1::AcquireOperationLeaseRequest lease_req;
  lease_req.set_operation_id(operation_id);
  lease_req.set_kind("seal_assembly");
  lease_req.set_target_artifact_id(req.assembly_id());
  lease_req.set_owner_id(owner_id_for_operation(d_.identity));
  // Let Global Store apply defaults and clamp to limits.
  lease_req.set_ttl_ms(0);

  auto lease_or = d_.global_store_client->acquire_operation_lease(lease_req);
  if (!lease_or.ok()) {
    if (absl::IsAlreadyExists(lease_or.status())) {
      rctx.mark_success();
      return Status::OK;
    }
    return to_grpc_status(lease_or.status());
  }
  const auto& lease_resp = *lease_or;
  if (!lease_resp.acquired()) {
    rctx.mark_success();
    return Status::OK;
  }

  const auto lease = lease_resp.lease();
  const uint64_t lease_generation = lease.lease_generation();
  const std::string lease_token = lease.lease_token();
  const std::string assembly_id = req.assembly_id();
  const std::string layout_id = req.layout_id();

  auto seal_tracker = seal_operation_tracker_;
  bool should_start = false;
  {
    absl::MutexLock lock(&seal_tracker->mu);
    should_start = seal_tracker->active_operations.insert(operation_id).second;
  }

  if (should_start) {
    auto client_sp = d_.global_store_client;
    auto executor = d_.async_runtime.blocking_executor();
    auto* async_runtime = &d_.async_runtime;
    auto* engine = &d_.engine;
    auto* devices = &d_.devices;
    auto* identity = &d_.identity;
    const DaemonOptions::PostSealPolicy post_seal_policy = d_.post_seal_policy;
    executor->add(
        [seal_tracker,
         client_sp = std::move(client_sp),
         async_runtime,
         engine,
         devices,
         identity,
         post_seal_policy,
         operation_id,
         assembly_id,
         layout_id,
         lease_generation,
         lease_token]() mutable -> void {
          if (client_sp == nullptr) {
            return;
          }
          absl::Status final_status = absl::OkStatus();
          OperationLeaseGuard lease_guard(client_sp, lease_token, operation_id);
          auto cleanup = absl::MakeCleanup([seal_tracker, operation_id]() {
            absl::MutexLock lock(&seal_tracker->mu);
            seal_tracker->active_operations.erase(operation_id);
          });

          auto keepalive_stop = std::make_shared<std::atomic<bool>>(false);
          auto keepalive_exec = async_runtime->blocking_executor();
          auto keepalive = std::make_shared<std::function<void()>>();
          std::weak_ptr<std::function<void()>> keepalive_weak = keepalive;
          *keepalive = [client_sp,
                        keepalive_stop,
                        keepalive_exec,
                        keepalive_weak,
                        &timekeeper = async_runtime->timekeeper(),
                        lease_token]() mutable {
            if (keepalive_stop->load(std::memory_order_relaxed)) {
              return;
            }
            timekeeper.after(std::chrono::milliseconds(5000))
                .via(keepalive_exec)
                .thenValue([client_sp, keepalive_stop, lease_token, keepalive_weak](folly::Unit) mutable {
                  if (keepalive_stop->load(std::memory_order_relaxed)) {
                    return;
                  }
                  tensorcast::operation::v1::KeepaliveOperationLeaseRequest req;
                  req.set_lease_token(lease_token);
                  req.set_ttl_ms(0);
                  auto resp_or = client_sp->keepalive_operation_lease(req);
                  if (!resp_or.ok()) {
                    LOG(WARNING) << "keepalive_operation_lease failed for op=" << lease_token << ": "
                                 << resp_or.status();
                  }
                  auto next = keepalive_weak.lock();
                  if (next != nullptr) {
                    (*next)();
                  }
                });
          };
          (*keepalive)();

          tensorcast::daemon::v2::SealAssemblySnapshot snapshot_msg;
          google::protobuf::Any snapshot_any;
          bool snapshot_loaded = false;
          {
            tensorcast::operation::v1::GetOperationRequest get_req;
            get_req.set_operation_id(operation_id);
            auto existing_or = client_sp->get_operation(get_req);
            if (existing_or.ok()) {
              const auto& existing_snapshot = existing_or->snapshot();
              const bool has_snapshot = !existing_snapshot.type_url().empty() || !existing_snapshot.value().empty();
              if (has_snapshot) {
                snapshot_any = existing_snapshot;
                snapshot_loaded = snapshot_any.UnpackTo(&snapshot_msg);
                if (!snapshot_loaded) {
                  final_status = absl::FailedPreconditionError("unsupported seal snapshot payload");
                }
              }
            } else {
              LOG(WARNING) << "get_operation failed while loading seal snapshot (op=" << operation_id
                           << "): " << existing_or.status();
            }
          }

          if (!snapshot_loaded && final_status.ok()) {
            snapshot_msg.set_assembly_id(assembly_id);
            if (!layout_id.empty()) {
              snapshot_msg.set_layout_id(layout_id);
              snapshot_msg.set_assembly_layout_binding_version(0);
            } else {
              auto binding_or = client_sp->get_assembly_layout_binding(assembly_id);
              if (binding_or.ok()) {
                snapshot_msg.set_layout_id(binding_or->layout_id());
                snapshot_msg.set_assembly_layout_binding_version(binding_or->binding_version());
              }
            }

            auto views_or = client_sp->list_views(assembly_id);
            if (!views_or.ok()) {
              final_status = views_or.status();
            } else {
              std::vector<store::components::ViewInfo> views = std::move(*views_or);
              std::sort(views.begin(), views.end(), [](const auto& a, const auto& b) { return a.view_id < b.view_id; });
              for (const auto& view : views) {
                if (view.view_id.empty()) {
                  continue;
                }
                auto* out = snapshot_msg.add_views();
                out->set_view_id(view.view_id);
                auto digest = compute_view_meta_digest(view);
                out->set_meta_digest(digest.data(), static_cast<int>(digest.size()));
              }
            }

            if (final_status.ok()) {
              snapshot_any.PackFrom(snapshot_msg);
              snapshot_loaded = true;
            }
          }

          tensorcast::operation::v1::UpdateOperationRequest running;
          running.set_operation_id(operation_id);
          running.set_lease_generation(lease_generation);
          auto* status = running.mutable_status();
          status->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);
          status->set_message("sealing");
          status->set_progress(0.0);
          *status->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
          if (snapshot_loaded) {
            running.mutable_snapshot()->CopyFrom(snapshot_any);
          }
          final_status = client_sp->update_operation(running);
          if (!final_status.ok()) {
            LOG(WARNING) << "update_operation(RUNNING) failed for op=" << operation_id << ": " << final_status;
            keepalive_stop->store(true, std::memory_order_relaxed);
            return;
          }

          std::vector<std::string> allowed_view_ids;
          allowed_view_ids.reserve(static_cast<size_t>(snapshot_msg.views_size()));
          for (const auto& view : snapshot_msg.views()) {
            if (!view.view_id().empty()) {
              allowed_view_ids.push_back(view.view_id());
            }
          }

          if (snapshot_msg.views_size() > 0) {
            auto current_views_or = client_sp->list_views(assembly_id);
            if (!current_views_or.ok()) {
              final_status = current_views_or.status();
            } else {
              absl::flat_hash_map<std::string, std::string> expected;
              expected.reserve(static_cast<size_t>(snapshot_msg.views_size()));
              for (const auto& view : snapshot_msg.views()) {
                expected.emplace(view.view_id(), view.meta_digest());
              }
              for (const auto& view : *current_views_or) {
                auto it = expected.find(view.view_id);
                if (it == expected.end()) {
                  continue;
                }
                auto digest = compute_view_meta_digest(view);
                const std::string computed(reinterpret_cast<const char*>(digest.data()), digest.size());
                if (computed != it->second) {
                  final_status = absl::FailedPreconditionError(
                      absl::StrCat("seal snapshot view metadata mismatch for view_id=", view.view_id));
                  break;
                }
                expected.erase(it);
              }
              if (final_status.ok() && !expected.empty()) {
                final_status = absl::FailedPreconditionError("seal snapshot view missing from current view set");
              }
            }
          }

          if (!final_status.ok()) {
            keepalive_stop->store(true, std::memory_order_relaxed);
            // Fall through to FAILED status update below.
          }

          auto last_progress_ms = std::make_shared<std::atomic<int64_t>>(0);
          auto max_hashed = std::make_shared<std::atomic<uint64_t>>(0);
          auto enable_updates = std::make_shared<std::atomic<bool>>(true);
          store::runtime::ingestion::MaterializationFacade::SealProgressCallback progress_cb =
              [client_sp, operation_id, lease_generation, last_progress_ms, max_hashed, enable_updates](
                  uint64_t hashed_leaf_count, uint64_t total_hash_leaves) mutable {
                if (!enable_updates->load(std::memory_order_relaxed) || total_hash_leaves == 0) {
                  return;
                }
                const uint64_t prev_max = max_hashed->load(std::memory_order_relaxed);
                if (hashed_leaf_count <= prev_max && hashed_leaf_count != total_hash_leaves) {
                  return;
                }
                max_hashed->store(std::max(prev_max, hashed_leaf_count), std::memory_order_relaxed);

                const int64_t now_ms = absl::ToUnixMillis(absl::Now());
                const int64_t last_ms = last_progress_ms->load(std::memory_order_relaxed);
                if (hashed_leaf_count != total_hash_leaves && last_ms != 0 && now_ms - last_ms < 1000) {
                  return;
                }
                last_progress_ms->store(now_ms, std::memory_order_relaxed);

                tensorcast::operation::v1::UpdateOperationRequest update;
                update.set_operation_id(operation_id);
                update.set_lease_generation(lease_generation);
                auto* status = update.mutable_status();
                status->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);
                status->set_message(absl::StrCat("hashing ", hashed_leaf_count, "/", total_hash_leaves));
                status->set_progress(static_cast<double>(hashed_leaf_count) / static_cast<double>(total_hash_leaves));
                *status->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
                absl::Status st = client_sp->update_operation(update);
                if (!st.ok()) {
                  enable_updates->store(false, std::memory_order_relaxed);
                  LOG(WARNING) << "update_operation(progress) failed for op=" << operation_id << ": " << st;
                }
              };

          const std::vector<std::string>* allowed_ptr = snapshot_loaded ? &allowed_view_ids : nullptr;
          auto seal_or = final_status.ok()
              ? engine->seal_assembly(assembly_id, /*publish_canonical=*/true, std::move(progress_cb), allowed_ptr)
              : absl::StatusOr<store::SealAssemblyResult>(final_status);
          if (!seal_or.ok()) {
            final_status = seal_or.status();
          } else {
            const std::string sealed_artifact_id = seal_or->sealed_artifact_id;
            if (final_status.ok() && !snapshot_msg.layout_id().empty()) {
              final_status = client_sp->attach_layout_to_artifact(sealed_artifact_id, snapshot_msg.layout_id());
            }

            std::optional<tensorcast::layout::v1::LayoutSpec> layout_spec_for_post_seal;
            if (final_status.ok() && !snapshot_msg.layout_id().empty()) {
              auto layout_or = client_sp->get_layout_spec(snapshot_msg.layout_id());
              if (!layout_or.ok()) {
                final_status = layout_or.status();
              } else {
                layout_spec_for_post_seal = layout_or->layout();
                const auto& layout_spec = *layout_spec_for_post_seal;
                const std::string proof_schema_version = layout_spec.proof_schema_version();
                absl::flat_hash_set<std::string> replicated_tensors;
                replicated_tensors.reserve(layout_spec.tensors_size());
                for (const auto& entry : layout_spec.tensors()) {
                  if (entry.second.overlap_mode() == tensorcast::layout::v1::OVERLAP_MODE_REPLICATE_EQUAL) {
                    replicated_tensors.insert(entry.first);
                  }
                }

                if (!replicated_tensors.empty()) {
                  if (proof_schema_version.empty()) {
                    final_status =
                        absl::FailedPreconditionError("proof_schema_version required for replicated tensors");
                  } else if (proof_schema_version != "v1") {
                    final_status = absl::UnimplementedError("unsupported proof_schema_version");
                  } else {
                    auto index_or = client_sp->get_artifact_index_by_id(sealed_artifact_id);
                    if (!index_or.ok()) {
                      final_status = index_or.status();
                    } else {
                      auto intervals_or = parse_tensor_intervals(*index_or);
                      if (!intervals_or.ok()) {
                        final_status = intervals_or.status();
                      } else {
                        auto resident_devices = engine->get_resident_devices(sealed_artifact_id);
                        auto gpu_it = std::find_if(
                            resident_devices.begin(), resident_devices.end(), [](const store::DeviceKey& d) {
                              return d.type == DeviceType::GPU;
                            });
                        if (gpu_it == resident_devices.end()) {
                          final_status =
                              absl::FailedPreconditionError("sealed artifact GPU replica unavailable for proofs");
                        } else {
                          store::loading::ReplicaKey replica_key;
                          replica_key.artifact_id = sealed_artifact_id;
                          replica_key.view_id = std::nullopt;
                          replica_key.device = *gpu_it;
                          replica_key.replica = 0;

                          auto size_or = engine->get_replica_size(replica_key);
                          auto ptr_or = engine->get_replica_gpu_ptr(replica_key);
                          if (!size_or.ok()) {
                            final_status = size_or.status();
                          } else if (!ptr_or.ok()) {
                            final_status = ptr_or.status();
                          } else {
                            store::loader::GpuMemorySource src(
                                gsl::not_null<void*>{reinterpret_cast<void*>(*ptr_or)},
                                /*device_id=*/gpu_it->ordinal,
                                *size_or);

                            std::vector<tensorcast::global_store::v1::TensorProofCommitmentWrite> writes;
                            for (const auto& interval : *intervals_or) {
                              if (interval.size_bytes == 0) {
                                continue;
                              }
                              if (!replicated_tensors.contains(interval.tensor_name)) {
                                continue;
                              }
                              const uint64_t expected_chunks =
                                  (interval.size_bytes + kProofChunkBytesV1 - 1) / kProofChunkBytesV1;
                              for (uint64_t chunk_idx = 0; chunk_idx < expected_chunks; ++chunk_idx) {
                                const uint64_t local_start = chunk_idx * kProofChunkBytesV1;
                                const uint64_t local_end =
                                    std::min<uint64_t>(interval.size_bytes, local_start + kProofChunkBytesV1);
                                if (local_end <= local_start) {
                                  continue;
                                }
                                const uint64_t abs_start = interval.offset + local_start;
                                const uint64_t read_len = local_end - local_start;
                                if (read_len > std::numeric_limits<size_t>::max()) {
                                  final_status = absl::OutOfRangeError("proof chunk exceeds host memory limits");
                                  break;
                                }
                                std::vector<uint8_t> buffer(static_cast<size_t>(read_len));
                                auto read_or = src.read_at(abs_start, buffer.data(), static_cast<size_t>(read_len));
                                if (!read_or.ok()) {
                                  final_status = read_or.status();
                                  break;
                                }
                                if (*read_or != buffer.size()) {
                                  final_status = absl::DataLossError("short read while computing MI2 proof digest");
                                  break;
                                }
                                std::vector<uint8_t> digest =
                                    tensorcast::common::sha256_digest_bytes(absl::MakeSpan(buffer));
                                if (digest.size() != 32) {
                                  final_status = absl::InternalError("sha256 digest size mismatch");
                                  break;
                                }
                                tensorcast::global_store::v1::TensorProofCommitmentWrite write;
                                write.set_tensor_name(interval.tensor_name);
                                write.set_proof_chunk_idx(chunk_idx);
                                write.set_digest(digest.data(), static_cast<int>(digest.size()));
                                writes.push_back(std::move(write));
                              }
                              if (!final_status.ok()) {
                                break;
                              }
                            }

                            if (final_status.ok() && !writes.empty()) {
                              constexpr size_t kBatchEntries = 1024;
                              for (size_t i = 0; i < writes.size(); i += kBatchEntries) {
                                tensorcast::global_store::v1::WriteTensorProofCommitmentsRequest write_req;
                                write_req.set_mi2_id(sealed_artifact_id);
                                write_req.set_proof_schema_version(proof_schema_version);
                                const size_t end = std::min(writes.size(), i + kBatchEntries);
                                for (size_t j = i; j < end; ++j) {
                                  *write_req.add_commitments() = writes[j];
                                }
                                auto write_resp_or = client_sp->write_tensor_proof_commitments(write_req);
                                if (!write_resp_or.ok()) {
                                  final_status = write_resp_or.status();
                                  break;
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }

            if (final_status.ok()) {
              const auto& policy = post_seal_policy;
              const bool allow_migration = policy.migrate_views;
              const bool allow_retire = policy.retire_pieces;
              if (allow_retire && !policy.migrate_views && !policy.reuse_views_if_safe) {
                LOG(WARNING) << "post-seal retire_pieces enabled without migrate_views or reuse_views_if_safe; "
                             << "view reads may fail after seal";
              }

              if (allow_migration) {
                auto views_or = client_sp->list_views(assembly_id);
                if (!views_or.ok()) {
                  LOG(WARNING) << "post-seal migrate_views list_views failed for assembly=" << assembly_id << ": "
                               << views_or.status();
                } else {
                  absl::flat_hash_set<std::string> allowed_set;
                  if (!allowed_view_ids.empty()) {
                    allowed_set.reserve(allowed_view_ids.size());
                    for (const auto& id : allowed_view_ids) {
                      if (!id.empty()) {
                        allowed_set.insert(id);
                      }
                    }
                  }
                  absl::flat_hash_set<std::string> expected_set;
                  if (layout_spec_for_post_seal.has_value()) {
                    const auto& expected = layout_spec_for_post_seal->expected_view_ids();
                    expected_set.reserve(static_cast<size_t>(expected.size()));
                    for (const auto& id : expected) {
                      if (!id.empty()) {
                        expected_set.insert(id);
                      }
                    }
                  }

                  const std::vector<std::string>* allowed_ptr = allowed_view_ids.empty() ? nullptr : &allowed_view_ids;
                  if (engine->get_num_gpus() == 0) {
                    LOG(WARNING) << "post-seal migrate_views skipped: no GPU devices available";
                  } else {
                    for (const auto& view : *views_or) {
                      if (view.view_id.empty()) {
                        continue;
                      }
                      if (!allowed_set.empty() && !allowed_set.contains(view.view_id)) {
                        continue;
                      }
                      if (!expected_set.empty() && !expected_set.contains(view.view_id)) {
                        continue;
                      }
                      if (view.view_spec_json.empty()) {
                        LOG(WARNING) << "post-seal migrate_views skipping view_id=" << view.view_id
                                     << " (missing view_spec_json)";
                        continue;
                      }
                      if (view.view_size_bytes == 0) {
                        LOG(WARNING) << "post-seal migrate_views skipping view_id=" << view.view_id
                                     << " (view_size_bytes=0)";
                        continue;
                      }
                      if (policy.migrate_transpose_only) {
                        auto spec_or = store::view::parse_view_spec_json(view.view_spec_json);
                        if (!spec_or.ok()) {
                          LOG(WARNING) << "post-seal migrate_views skipping view_id=" << view.view_id
                                       << " (invalid view_spec_json): " << spec_or.status();
                          continue;
                        }
                        if (!spec_includes_transpose(*spec_or)) {
                          continue;
                        }
                      }

                      const store::DeviceKey target_device = devices->DefaultGpu();
                      auto handle_or = engine->materialize_view_from_assembly(
                          assembly_id,
                          sealed_artifact_id,
                          view.view_id,
                          view.view_spec_json,
                          target_device,
                          store::loading::TransformPlacement::kServer,
                          allowed_ptr);
                      if (!handle_or.ok()) {
                        LOG(WARNING) << "post-seal migrate_views failed for view_id=" << view.view_id << ": "
                                     << handle_or.status();
                        continue;
                      }

                      auto publish_status = engine->register_replica_with_global_store(handle_or->replica_key, {});
                      if (!publish_status.ok() && !absl::IsAlreadyExists(publish_status)) {
                        LOG(WARNING) << "post-seal migrate_views register_replica failed for view_id=" << view.view_id
                                     << ": " << publish_status;
                      }

                      store::components::ViewStateUpdate update;
                      update.artifact_id = sealed_artifact_id;
                      update.view_id = view.view_id;
                      update.view_spec_json = view.view_spec_json;
                      update.view_size_bytes = view.view_size_bytes;
                      if (view.view_data_hash.has_value()) {
                        update.view_data_hash = view.view_data_hash;
                      }
                      update.mark_verified = view.verified_at.has_value();
                      update.canonical_size_bytes = view.canonical_size_bytes;
                      update.canonical_bytes_covered = view.canonical_bytes_covered;
                      update.canonical_ranges = view.canonical_ranges;
                      auto view_status = client_sp->update_artifact_view_state(update);
                      if (!view_status.ok()) {
                        LOG(WARNING) << "post-seal migrate_views update_view_state failed for view_id=" << view.view_id
                                     << ": " << view_status;
                      }
                    }
                  }
                }
              }

              if (allow_retire) {
                const std::string worker_id = identity->worker_id();
                if (!worker_id.empty()) {
                  auto unreg_status = client_sp->unregister_replica_by_worker(assembly_id, worker_id);
                  if (!unreg_status.ok()) {
                    LOG(WARNING) << "post-seal retire_pieces unregister_replica_by_worker failed for assembly="
                                 << assembly_id << ": " << unreg_status;
                  }
                } else {
                  LOG(WARNING) << "post-seal retire_pieces skipped unregister_replica_by_worker: worker_id unavailable";
                }

                std::vector<store::loading::ReplicaKey> to_unload;
                for (const auto& info : engine->get_all_replicas_info()) {
                  if (info.key.artifact_id == assembly_id) {
                    to_unload.push_back(info.key);
                  }
                }
                for (const auto& key : to_unload) {
                  auto unload_status = engine->unload_replica_status(key);
                  if (!unload_status.ok()) {
                    LOG(WARNING) << "post-seal retire_pieces unload_replica failed for key=" << key << ": "
                                 << unload_status;
                  }
                }
              }
            }

            tensorcast::daemon::v2::SealAssemblyResult result_msg;
            auto* artifact = result_msg.mutable_artifact();
            artifact->set_artifact_id(sealed_artifact_id);
            if (!seal_or->index_multihash.empty()) {
              artifact->set_index_multihash(seal_or->index_multihash);
            }
            if (!seal_or->data_multihash.empty()) {
              artifact->set_data_multihash(seal_or->data_multihash);
            }
            if (!seal_or->schema_version.empty()) {
              artifact->set_schema_version(seal_or->schema_version);
            }
            if (!seal_or->encoding.empty()) {
              artifact->set_encoding(seal_or->encoding);
            }
            if (seal_or->total_size > 0) {
              artifact->set_total_size(seal_or->total_size);
            }

            tensorcast::operation::v1::UpdateOperationRequest success;
            success.set_operation_id(operation_id);
            success.set_lease_generation(lease_generation);
            auto* out = success.mutable_status();
            out->set_state(tensorcast::operation::v1::OPERATION_STATE_SUCCESS);
            out->set_message("sealed");
            out->set_progress(1.0);
            *out->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
            out->mutable_result()->PackFrom(result_msg);
            if (final_status.ok()) {
              final_status = client_sp->update_operation(success);
            }
          }

          if (!final_status.ok()) {
            tensorcast::operation::v1::UpdateOperationRequest failed;
            failed.set_operation_id(operation_id);
            failed.set_lease_generation(lease_generation);
            auto* out = failed.mutable_status();
            out->set_state(tensorcast::operation::v1::OPERATION_STATE_FAILED);
            out->set_message("seal failed");
            out->set_progress(0.0);
            *out->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
            auto* err = out->mutable_error();
            err->set_status_code(absl::StatusCodeToString(final_status.code()));
            err->set_message(std::string(final_status.message()));
            err->set_retryable(retryable_status(final_status));
            absl::Status update_st = client_sp->update_operation(failed);
            if (!update_st.ok()) {
              LOG(WARNING) << "update_operation(FAILED) failed for op=" << operation_id << ": " << update_st;
            }
          }

          keepalive_stop->store(true, std::memory_order_relaxed);
          lease_guard.release();
        });
  }

  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::get_operation(
    RpcContext& rctx,
    const tensorcast::operation::v1::GetOperationRequest& req,
    tensorcast::operation::v1::GetOperationResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.operation.id", req.operation_id());

  if (req.operation_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "operation_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  auto op_or = d_.global_store_client->get_operation(req);
  if (!op_or.ok()) {
    return to_grpc_status(op_or.status());
  }
  resp = std::move(*op_or);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::wait_operation(
    RpcContext& rctx,
    const v2::WaitOperationRequest& req,
    v2::WaitOperationResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.operation.id", req.operation_id());

  if (req.operation_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "operation_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  const uint64_t timeout_ms = req.timeout_ms();
  const absl::Time start = absl::Now();
  const absl::Time deadline = timeout_ms > 0 ? start + absl::Milliseconds(timeout_ms) : absl::InfiniteFuture();

  absl::Duration sleep = absl::Milliseconds(50);
  tensorcast::operation::v1::GetOperationRequest op_req;
  op_req.set_operation_id(req.operation_id());

  while (absl::Now() < deadline) {
    auto op_or = d_.global_store_client->get_operation(op_req);
    if (!op_or.ok()) {
      return to_grpc_status(op_or.status());
    }
    const auto state = op_or->status().state();
    resp.mutable_operation()->Swap(&(*op_or));
    if (state == tensorcast::operation::v1::OPERATION_STATE_SUCCESS ||
        state == tensorcast::operation::v1::OPERATION_STATE_FAILED ||
        state == tensorcast::operation::v1::OPERATION_STATE_CANCELLED) {
      rctx.mark_success();
      return Status::OK;
    }
    absl::SleepFor(sleep);
    sleep = std::min(sleep * 12 / 10, absl::Milliseconds(500));
  }

  auto op_or = d_.global_store_client->get_operation(op_req);
  if (!op_or.ok()) {
    return to_grpc_status(op_or.status());
  }
  resp.mutable_operation()->Swap(&(*op_or));
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::confirm(
    RpcContext& rctx,
    const v2::ConfirmReplicaRequest& req,
    v2::ConfirmReplicaResponse& resp) const {
  return replica_lifecycle_service_.confirm(rctx, req, resp);
}

grpc::Status MaterializationController::unload(
    RpcContext& rctx,
    const v2::UnloadReplicaRequest& req,
    v2::UnloadReplicaResponse& resp) {
  return replica_lifecycle_service_.unload(rctx, req, resp);
}

grpc::Status MaterializationController::wait_verification(
    RpcContext& rctx,
    const v2::WaitReplicaVerificationRequest& req,
    v2::WaitReplicaVerificationResponse& resp) {
  return replica_lifecycle_service_.wait_verification(rctx, req, resp);
}

} // namespace tensorcast::daemon
