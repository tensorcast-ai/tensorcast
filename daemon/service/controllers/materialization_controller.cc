// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_controller.h"

#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "daemon/util/grpc_peer_utils.h"
#include "daemon/util/status_utils.h"

namespace tensorcast::daemon {

using status_utils::to_grpc_status;
namespace global_store = tensorcast::global_store::v1;

namespace {

void merge_operation_ref_metadata(
    const tensorcast::operation::v1::OperationRef& source,
    tensorcast::operation::v1::OperationRef* target) {
  if (target == nullptr) {
    return;
  }
  if (target->operation_id().empty() && !source.operation_id().empty()) {
    target->set_operation_id(source.operation_id());
  }
  if (target->kind().empty() && !source.kind().empty()) {
    target->set_kind(source.kind());
  }
  if (target->target_artifact_id().empty() && !source.target_artifact_id().empty()) {
    target->set_target_artifact_id(source.target_artifact_id());
  }
  if (target->authority_scope_kind().empty() && !source.authority_scope_kind().empty()) {
    target->set_authority_scope_kind(source.authority_scope_kind());
  }
  if (target->authority_scope_id().empty() && !source.authority_scope_id().empty()) {
    target->set_authority_scope_id(source.authority_scope_id());
  }
  if (target->attachment_kind().empty() && !source.attachment_kind().empty()) {
    target->set_attachment_kind(source.attachment_kind());
  }
  if (target->recovery_class().empty() && !source.recovery_class().empty()) {
    target->set_recovery_class(source.recovery_class());
  }
  if (target->fencing_digest().empty() && !source.fencing_digest().empty()) {
    target->set_fencing_digest(source.fencing_digest());
  }
}

bool operation_state_is_terminal(tensorcast::operation::v1::OperationState state) {
  using State = tensorcast::operation::v1::OperationState;
  return state == State::OPERATION_STATE_SUCCESS || state == State::OPERATION_STATE_FAILED ||
      state == State::OPERATION_STATE_CANCELLED;
}

absl::StatusOr<v2::TargetLayout> parse_serving_target_layout(std::string_view target_layout_bytes) {
  if (target_layout_bytes.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return absl::InvalidArgumentError("serving target resolved_layout.target_layout is too large");
  }
  v2::TargetLayout target_layout;
  if (!target_layout.ParseFromArray(target_layout_bytes.data(), static_cast<int>(target_layout_bytes.size()))) {
    return absl::InvalidArgumentError("serving target resolved_layout.target_layout must be a TargetLayout proto");
  }
  if (target_layout.storages_size() == 0 || target_layout.offsets_size() == 0) {
    return absl::InvalidArgumentError("serving target TargetLayout requires storages and offsets");
  }
  return target_layout;
}

uint64_t logical_total_size(const v2::TargetLayout& target_layout) {
  uint64_t total = 0;
  for (const auto& offset : target_layout.offsets()) {
    const uint64_t end = offset.storage_offset() + offset.logical_length();
    if (end > total) {
      total = end;
    }
  }
  return total;
}

std::chrono::milliseconds duration_from_policy_ms(
    bool has_value,
    uint64_t value_ms,
    std::chrono::milliseconds default_value) {
  if (!has_value) {
    return default_value;
  }
  const uint64_t max_ms = static_cast<uint64_t>(std::chrono::milliseconds::max().count());
  return std::chrono::milliseconds(static_cast<int64_t>(std::min(value_ms, max_ms)));
}

std::string verification_state_string(v2::BindingValueVerificationState state) {
  switch (state) {
    case v2::BINDING_VALUE_VERIFICATION_STATE_PENDING:
      return "pending";
    case v2::BINDING_VALUE_VERIFICATION_STATE_VERIFIED:
      return "verified";
    case v2::BINDING_VALUE_VERIFICATION_STATE_FAILED:
      return "failed";
    case v2::BINDING_VALUE_VERIFICATION_STATE_LOCAL_ONLY:
      return "local_only";
    case v2::BINDING_VALUE_VERIFICATION_STATE_UNSPECIFIED:
    default:
      return "local_only";
  }
}

void cleanup_prefetch_created_binding(
    BindingRegistry& registry,
    HandleLeaseRegistry* handle_leases,
    const v2::CreateOwnedBindingResponse& create_resp,
    std::string_view reason) {
  if (create_resp.binding_id().empty()) {
    return;
  }

  bool safe_to_close = true;
  if (create_resp.created_staged_value() && !create_resp.staged_value().binding_value_id().empty()) {
    const auto remove_status =
        registry.remove_staged_value(create_resp.binding_id(), create_resp.staged_value().binding_value_id(), reason);
    if (!remove_status.ok() && !absl::IsNotFound(remove_status)) {
      LOG(WARNING) << "failed to remove staged serving binding value during cleanup reason=" << reason
                   << " binding_id=" << create_resp.binding_id()
                   << " binding_value_id=" << create_resp.staged_value().binding_value_id() << ": " << remove_status;
      if (absl::IsFailedPrecondition(remove_status)) {
        safe_to_close = false;
      }
    }
  }

  if (!create_resp.mem_handle().lease_token().empty() && handle_leases != nullptr) {
    const auto release_status = handle_leases->release(create_resp.mem_handle().lease_token());
    if (!release_status.ok() && !absl::IsNotFound(release_status)) {
      LOG(WARNING) << "failed to release serving prefetch bootstrap lease during cleanup reason=" << reason
                   << " binding_id=" << create_resp.binding_id() << ": " << release_status;
    }
  }

  if (!safe_to_close) {
    return;
  }
  const auto retire_status = registry.retire_retained(create_resp.binding_id(), reason);
  if (!retire_status.ok()) {
    if (absl::IsNotFound(retire_status)) {
      VLOG(1) << "serving prefetch cleanup could not find binding_id=" << create_resp.binding_id()
              << " reason=" << reason;
      return;
    }
    LOG(WARNING) << "failed to retire serving prefetch binding during cleanup reason=" << reason
                 << " binding_id=" << create_resp.binding_id() << ": " << retire_status;
  }
}

std::string grpc_code_name(grpc::StatusCode code) {
  switch (code) {
    case grpc::StatusCode::OK:
      return "OK";
    case grpc::StatusCode::CANCELLED:
      return "CANCELLED";
    case grpc::StatusCode::UNKNOWN:
      return "UNKNOWN";
    case grpc::StatusCode::INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case grpc::StatusCode::DEADLINE_EXCEEDED:
      return "DEADLINE_EXCEEDED";
    case grpc::StatusCode::NOT_FOUND:
      return "NOT_FOUND";
    case grpc::StatusCode::ALREADY_EXISTS:
      return "ALREADY_EXISTS";
    case grpc::StatusCode::PERMISSION_DENIED:
      return "PERMISSION_DENIED";
    case grpc::StatusCode::RESOURCE_EXHAUSTED:
      return "RESOURCE_EXHAUSTED";
    case grpc::StatusCode::FAILED_PRECONDITION:
      return "FAILED_PRECONDITION";
    case grpc::StatusCode::ABORTED:
      return "ABORTED";
    case grpc::StatusCode::OUT_OF_RANGE:
      return "OUT_OF_RANGE";
    case grpc::StatusCode::UNIMPLEMENTED:
      return "UNIMPLEMENTED";
    case grpc::StatusCode::INTERNAL:
      return "INTERNAL";
    case grpc::StatusCode::UNAVAILABLE:
      return "UNAVAILABLE";
    case grpc::StatusCode::DATA_LOSS:
      return "DATA_LOSS";
    case grpc::StatusCode::UNAUTHENTICATED:
      return "UNAUTHENTICATED";
    default:
      return "UNKNOWN";
  }
}

void fill_binding_value_ref(const BindingRegistry::Record& record, tensorcast::publication::v1::BindingValueRef* ref) {
  ref->set_binding_id(record.binding_id);
  ref->set_binding_layout_id(record.binding_layout_id);
  ref->set_binding_value_id(record.current_binding_value_id);
  ref->set_seal_generation(record.seal_generation);
}

void fill_binding_value_ref(const v2::BindingValue& value, tensorcast::publication::v1::BindingValueRef* ref) {
  ref->set_binding_id(value.binding_id());
  ref->set_binding_layout_id(value.binding_layout_id());
  ref->set_binding_value_id(value.binding_value_id());
  ref->set_seal_generation(value.seal_generation());
}

std::string make_daemon_session_id() {
  return absl::StrCat("session-", ::getpid(), "-", std::chrono::steady_clock::now().time_since_epoch().count());
}

} // namespace

MaterializationController::MaterializationController(Dep d)
    : global_store_client_(d.global_store_client),
      shutdown_signal_(&d.shutdown_signal),
      binding_registry_(&d.binding_registry),
      handle_leases_(d.handle_leases),
      serving_prefetch_(d.serving_prefetch),
      daemon_id_(d.daemon_id.empty() ? std::string("daemon-local") : d.daemon_id),
      daemon_session_id_(make_daemon_session_id()),
      assembly_operation_service_(
          AssemblyOperationService::Dep{
              .engine = d.engine,
              .devices = d.devices,
              .shutdown_signal = d.shutdown_signal,
              .async_runtime = d.async_runtime,
              .identity = d.identity,
              .bindings = d.binding_registry,
              .global_store_client = d.global_store_client,
              .lip_manager = &d.lip_manager,
              .lifecycle = d.lifecycle,
              .post_seal_policy = d.post_seal_policy,
              .await_state_sync_barrier = d.await_state_sync_barrier,
          }),
      disk_artifact_service_(
          DiskArtifactService::Dep{
              .engine = d.engine,
              .source_registry = d.disk_imports,
              .shutdown_signal = d.shutdown_signal,
              .global_store_client = d.global_store_client,
              .storage_path = d.storage_path,
              .public_disk_source_policy = d.public_disk_source_policy,
          }),
      replica_materialization_service_(
          ReplicaMaterializationService::Dep{
              .engine = d.engine,
              .refs = d.refs,
              .sessions = d.sessions,
              .lip = d.lip,
              .devices = d.devices,
              .disk_imports = d.disk_imports,
              .shutdown_signal = d.shutdown_signal,
              .global_store_client = d.global_store_client,
              .identity = &d.identity,
              .lifecycle = d.lifecycle,
              .handle_leases = d.handle_leases,
              .cpu_shared_memory_enabled = d.cpu_shared_memory_enabled,
              .post_seal_policy = d.post_seal_policy,
              .progressive_replication = d.progressive_replication,
              .daemon_id = daemon_id_,
              .daemon_session_id = daemon_session_id_,
              .storage_path = d.storage_path,
          }),
      replica_lifecycle_service_(
          ReplicaLifecycleService::Dep{
              .engine = d.engine,
              .refs = d.refs,
              .sessions = d.sessions,
              .lifecycle = d.lifecycle,
              .devices = d.devices,
          }),
      target_materialization_service_(
          TargetMaterializationService::Dep{
              .engine = d.engine,
              .lip_manager = d.lip_manager,
              .devices = d.devices,
              .regions = d.regions,
              .disk_imports = d.disk_imports,
              .lifecycle = *d.lifecycle,
              .lifecycle_kernel = *d.lifecycle_kernel,
              .shutdown_signal = d.shutdown_signal,
              .async_runtime = d.async_runtime,
              .identity = d.identity,
              .external_target_access_service = d.external_target_access_service,
              .global_store_client = d.global_store_client,
              .max_concurrency = d.max_concurrency,
              .capability_tokens = d.capability_tokens,
              .external_target_verification_enabled = d.external_target_verification_enabled,
              .progressive_replication = d.progressive_replication,
              .daemon_id = daemon_id_,
              .daemon_session_id = daemon_session_id_,
              .storage_path = d.storage_path,
              .await_state_sync_barrier = d.await_state_sync_barrier,
          }),
      owner_binding_service_(
          OwnedBindingService::Dep{
              .engine = d.engine,
              .devices = d.devices,
              .disk_imports = d.disk_imports,
              .bindings = d.binding_registry,
              .shutdown_signal = d.shutdown_signal,
              .async_runtime = d.async_runtime,
              .identity = d.identity,
              .global_store_client = d.global_store_client,
              .lifecycle = d.lifecycle,
              .handle_leases = d.handle_leases,
              .registration_manager = &d.registration_manager,
              .lip_manager = &d.lip_manager,
              .refs = &d.refs,
              .regions = &d.regions,
              .max_concurrency = d.max_concurrency,
              .capability_tokens = d.capability_tokens,
              .target_materialization_service = &target_materialization_service_,
              .daemon_id = daemon_id_,
              .daemon_session_id = daemon_session_id_,
              .storage_path = d.storage_path,
              .await_state_sync_barrier = d.await_state_sync_barrier,
          }) {
  // Register publish replay admission as one child-owner policy behind the
  // shared observation-path dispatcher. The dispatcher itself does not own
  // publish semantics.
  public_operation_admission_service_.register_handler(
      std::string(TargetPublishService::public_operation_kind()),
      [this](const tensorcast::operation::v1::OperationRef& operation_ref, absl::Time now) {
        return target_materialization_service_.admit_public_operation(operation_ref, now);
      });
}

namespace {

absl::Status validate_serving_member_target(const tensorcast::operation::v1::ServingBindingTarget& target) {
  if (target.runtime().empty()) {
    return absl::InvalidArgumentError("serving target runtime is required");
  }
  if (target.device_uuid().empty()) {
    return absl::InvalidArgumentError("serving target device_uuid is required");
  }
  if (target.serving_build_digest().empty()) {
    return absl::InvalidArgumentError("serving target serving_build_digest is required");
  }
  if (!target.has_member() || target.member().member_id().empty()) {
    return absl::InvalidArgumentError("serving target member is required");
  }
  if (!target.has_resolved_layout()) {
    return absl::FailedPreconditionError("serving target resolved_layout is required before allocation");
  }
  const auto& layout = target.resolved_layout();
  if (layout.binding_layout_id().empty() || layout.target_layout_hash().empty() ||
      layout.tensor_schema_hash().empty()) {
    return absl::FailedPreconditionError("serving target resolved_layout is incomplete");
  }
  if (layout.target_layout().empty() || layout.target_index_bytes().empty()) {
    return absl::FailedPreconditionError("serving target layout/index bytes are required before allocation");
  }
  if (auto target_layout_or = parse_serving_target_layout(layout.target_layout()); !target_layout_or.ok()) {
    return target_layout_or.status();
  }
  if (layout.member().member_id() != target.member().member_id() ||
      layout.member().member_index() != target.member().member_index() ||
      layout.member().member_count() != target.member().member_count() ||
      layout.member().group_id() != target.member().group_id()) {
    return absl::FailedPreconditionError("serving target resolved_layout member mismatch");
  }
  if (layout.topology().schema_topology_digest() != target.topology().schema_topology_digest()) {
    return absl::FailedPreconditionError("serving target resolved_layout topology mismatch");
  }
  return absl::OkStatus();
}

bool same_serving_member_ref(
    const tensorcast::operation::v1::ServingBindingMemberRef& lhs,
    const tensorcast::operation::v1::ServingBindingMemberRef& rhs) {
  return lhs.member_id() == rhs.member_id() && lhs.member_index() == rhs.member_index() &&
      lhs.member_count() == rhs.member_count() && lhs.group_id() == rhs.group_id();
}

absl::Status validate_serving_prefetch_request(const v2::PrefetchServingBindingRequest& req) {
  switch (req.target_case()) {
    case v2::PrefetchServingBindingRequest::kServingBindingTarget:
      return validate_serving_member_target(req.serving_binding_target());
    case v2::PrefetchServingBindingRequest::kServingBindingSetTarget: {
      const auto& target_set = req.serving_binding_set_target();
      if (target_set.members().empty()) {
        return absl::InvalidArgumentError("serving binding set target requires members");
      }
      for (const auto& member_target : target_set.members()) {
        if (auto status = validate_serving_member_target(member_target); !status.ok()) {
          return status;
        }
      }
      return absl::OkStatus();
    }
    case v2::PrefetchServingBindingRequest::TARGET_NOT_SET:
      return absl::InvalidArgumentError("serving prefetch target is required");
  }
  return absl::InvalidArgumentError("serving prefetch target is required");
}

void fill_current_binding_value(const BindingRegistry::Record& record, v2::BindingValue* value) {
  value->set_binding_id(record.binding_id);
  value->set_binding_layout_id(record.binding_layout_id);
  value->set_binding_value_id(record.current_binding_value_id);
  value->set_seal_generation(record.seal_generation);
  value->mutable_selection()->CopyFrom(record.current_selection);
  value->set_is_artifact_backed(record.state == v2::BINDING_STATE_READY_ARTIFACT);
  value->set_verification_state(record.verification_state);
  value->set_verification_job_id(record.verification_job_id);
  value->set_source_artifact_ref(record.source_artifact_ref);
  value->set_local_serving_ref(record.local_serving_ref);
  if (!record.serving_artifact_id.empty()) {
    value->set_serving_artifact_id(record.serving_artifact_id);
  }
  if (!record.verification_failure_reason.empty()) {
    value->set_verification_failure_reason(record.verification_failure_reason);
  }
  if (!record.current_artifact_id.empty()) {
    value->set_source_artifact_id(record.current_artifact_id);
  }
}

void fill_staged_binding_value(
    const BindingRegistry::Record& record,
    const BindingRegistry::StagedBindingValue& staged,
    v2::BindingValue* value) {
  value->set_binding_id(record.binding_id);
  value->set_binding_layout_id(record.binding_layout_id);
  value->set_binding_value_id(staged.binding_value_id);
  value->set_seal_generation(staged.expected_previous_seal_generation);
  if (!staged.artifact_id.empty()) {
    value->set_source_artifact_id(staged.artifact_id);
  }
  if (!staged.selection.artifact_id().empty()) {
    value->mutable_selection()->CopyFrom(staged.selection);
  }
  value->set_is_artifact_backed(!staged.artifact_id.empty());
  value->set_verification_state(staged.verification_state);
  if (!staged.verification_failure_reason.empty()) {
    value->set_verification_failure_reason(staged.verification_failure_reason);
  }
}

struct GroupRealizationPublishCheck {
  grpc::Status status;
  bool terminal_not_published{false};
};

GroupRealizationPublishCheck group_realization_status_to_grpc(
    global_store::Status status,
    global_store::GroupRealizationState state) {
  if (status == global_store::STATUS_OK && state == global_store::GROUP_REALIZATION_STATE_PUBLISHED) {
    return {.status = grpc::Status::OK};
  }
  if (status == global_store::STATUS_NOT_FOUND) {
    return {.status = {grpc::StatusCode::NOT_FOUND, "group realization transaction not found"}};
  }
  if (status == global_store::STATUS_TIMED_OUT) {
    return {
        .status = {
            grpc::StatusCode::DEADLINE_EXCEEDED, "group realization transaction is not published before deadline"}};
  }
  if (state == global_store::GROUP_REALIZATION_STATE_ABORTED ||
      state == global_store::GROUP_REALIZATION_STATE_EXPIRED) {
    return {
        .status =
            {grpc::StatusCode::FAILED_PRECONDITION, "group realization transaction is terminal but not published"},
        .terminal_not_published = true};
  }
  return {.status = {grpc::StatusCode::FAILED_PRECONDITION, "group realization transaction is not published"}};
}

GroupRealizationPublishCheck verify_group_realization_published(
    const std::shared_ptr<store::components::IGlobalStoreClient>& global_store_client,
    const v2::GroupRealizationAcquireRef& acquire) {
  if (acquire.transaction_id().empty()) {
    return {.status = {grpc::StatusCode::INVALID_ARGUMENT, "group_realization_acquire.transaction_id is required"}};
  }
  if (global_store_client == nullptr || !global_store_client->is_connected()) {
    return {.status = {grpc::StatusCode::FAILED_PRECONDITION, "GlobalStoreClient is required for group-aware acquire"}};
  }
  global_store::WaitGroupRealizationPublishedRequest request;
  request.set_transaction_id(acquire.transaction_id());
  if (acquire.wait_for_publish()) {
    const uint32_t timeout_ms = acquire.wait_timeout_ms() == 0 ? 1U : acquire.wait_timeout_ms();
    request.set_deadline_unix_nanos(
        static_cast<uint64_t>(absl::ToUnixNanos(absl::Now() + absl::Milliseconds(timeout_ms))));
  } else {
    request.set_deadline_unix_nanos(0);
  }
  auto response_or = global_store_client->wait_group_realization_published(request);
  if (!response_or.ok()) {
    return {.status = to_grpc_status(response_or.status())};
  }
  return group_realization_status_to_grpc(response_or->status(), response_or->state());
}

} // namespace

grpc::Status MaterializationController::prefetch_serving_binding(
    RpcContext& rctx,
    const v2::PrefetchServingBindingRequest& req,
    v2::PrefetchServingBindingResponse& resp) {
  if (auto status = validate_serving_prefetch_request(req); !status.ok()) {
    return to_grpc_status(status);
  }
  if (req.target_case() == v2::PrefetchServingBindingRequest::kServingBindingSetTarget) {
    tensorcast::operation::v1::PrefetchServingBindingSetResult set_result;
    const auto& target_set = req.serving_binding_set_target();
    set_result.set_runtime(target_set.runtime());
    set_result.mutable_topology()->CopyFrom(target_set.topology());
    set_result.set_group_id(target_set.group_id());
    set_result.set_readiness(tensorcast::operation::v1::SERVING_BINDING_READINESS_LOCAL_READY);
    bool failed = false;
    for (const auto& member_target : target_set.members()) {
      v2::PrefetchServingBindingRequest member_req;
      member_req.mutable_source_selection()->CopyFrom(req.source_selection());
      member_req.mutable_source()->CopyFrom(req.source());
      member_req.mutable_serving_binding_target()->CopyFrom(member_target);
      member_req.set_requested_readiness(req.requested_readiness());
      member_req.mutable_retention_policy()->CopyFrom(req.retention_policy());
      if (req.has_operation_id() && !req.operation_id().empty()) {
        member_req.set_operation_id(absl::StrCat(req.operation_id(), ":", member_target.member().member_id()));
      }
      if (req.has_group_realization()) {
        member_req.mutable_group_realization()->CopyFrom(req.group_realization());
        member_req.mutable_group_realization()->mutable_group()->set_part_id(member_target.member().member_id());
      }

      v2::PrefetchServingBindingResponse member_resp;
      const grpc::Status member_status = prefetch_serving_binding(rctx, member_req, member_resp);
      if (!member_status.ok() || member_resp.status().state() != tensorcast::operation::v1::OPERATION_STATE_SUCCESS ||
          !member_resp.status().has_result()) {
        failed = true;
        auto* failure = set_result.add_member_failures();
        failure->mutable_member()->CopyFrom(member_target.member());
        failure->set_code(member_status.ok() ? "FAILED_PRECONDITION" : grpc_code_name(member_status.error_code()));
        failure->set_message(
            member_status.ok() ? member_resp.status().message() : std::string(member_status.error_message()));
        failure->set_phase("member_materialization");
        continue;
      }
      tensorcast::operation::v1::PrefetchServingBindingResult member_result;
      if (!member_resp.status().result().UnpackTo(&member_result)) {
        failed = true;
        auto* failure = set_result.add_member_failures();
        failure->mutable_member()->CopyFrom(member_target.member());
        failure->set_code("INTERNAL");
        failure->set_message("member result did not contain PrefetchServingBindingResult");
        failure->set_phase("member_result_decode");
        continue;
      }
      set_result.add_members()->CopyFrom(member_result);
    }

    if (failed) {
      for (const auto& member : set_result.members()) {
        const auto cleanup_status =
            binding_registry_->retire_retained(member.binding_value_ref().binding_id(), "partitioned_set_failure");
        if (!cleanup_status.ok()) {
          LOG(WARNING) << "failed to retire partitioned serving sibling binding_id="
                       << member.binding_value_ref().binding_id() << ": " << cleanup_status;
        }
      }
      set_result.set_partial(set_result.members_size() > 0);
    }

    auto* op_ref = resp.mutable_operation_ref();
    op_ref->set_operation_id(
        req.has_operation_id() && !req.operation_id().empty() ? req.operation_id() : "prefetch-serving-binding-set");
    op_ref->set_kind("prefetch_serving_binding_set");
    op_ref->set_target_artifact_id(req.source_selection().artifact_id());
    op_ref->set_authority_scope_kind("daemon_retained_binding_set");
    op_ref->set_authority_scope_id(target_set.group_id());
    op_ref->set_attachment_kind("serving_binding_value_set");
    op_ref->set_recovery_class("daemon_retained_local");

    auto* status = resp.mutable_status();
    status->set_state(
        failed ? tensorcast::operation::v1::OPERATION_STATE_FAILED
               : tensorcast::operation::v1::OPERATION_STATE_SUCCESS);
    status->set_message(failed ? "serving binding set materialization failed" : "serving binding set is local-ready");
    status->set_progress(1.0);
    status->mutable_result()->PackFrom(set_result);
    rctx.mark_success();
    return grpc::Status::OK;
  }
  if (!req.has_source_selection() || req.source_selection().artifact_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "source_selection.artifact_id is required"};
  }
  if (req.requested_readiness() == tensorcast::operation::v1::SERVING_BINDING_READINESS_PUBLISHED_READY) {
    return {grpc::StatusCode::UNIMPLEMENTED, "serving_published_ready promotion is not implemented"};
  }

  const auto& target = req.serving_binding_target();
  const auto& layout = target.resolved_layout();
  const auto reuse_mode = layout.source_reuse().mode();
  if (reuse_mode == tensorcast::operation::v1::SERVING_BINDING_SOURCE_REUSE_MODE_SERVING_TRANSFORM_REQUIRED) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "serving transform required before allocation"};
  }
  if (reuse_mode == tensorcast::operation::v1::SERVING_BINDING_SOURCE_REUSE_MODE_UNSUPPORTED ||
      reuse_mode == tensorcast::operation::v1::SERVING_BINDING_SOURCE_REUSE_MODE_UNSPECIFIED) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "serving source reuse mode is unsupported"};
  }

  auto target_layout_or = parse_serving_target_layout(layout.target_layout());
  if (!target_layout_or.ok()) {
    return to_grpc_status(target_layout_or.status());
  }
  v2::CreateOwnedBindingRequest create_req;
  create_req.mutable_source_selection()->CopyFrom(req.source_selection());
  create_req.mutable_target_layout()->CopyFrom(*target_layout_or);
  create_req.set_target_index_bytes(layout.target_index_bytes());
  create_req.set_pid(static_cast<int32_t>(::getpid()));
  create_req.set_device_uuid(target.device_uuid());
  create_req.set_binding_layout_id(layout.binding_layout_id());
  if (req.has_operation_id() && !req.operation_id().empty()) {
    create_req.set_operation_id(req.operation_id());
  }
  if (!layout.copy_plan_bytes().empty()) {
    return {grpc::StatusCode::UNIMPLEMENTED, "serving binding mapped copy plans are not implemented"};
  }
  if (req.has_group_realization()) {
    create_req.mutable_group_realization()->CopyFrom(req.group_realization());
  }

  v2::CreateOwnedBindingResponse create_resp;
  const grpc::Status create_status = owner_binding_service_.create_owned_binding(rctx, create_req, create_resp);
  if (!create_status.ok()) {
    return create_status;
  }
  if (rctx.server_context().IsCancelled()) {
    cleanup_prefetch_created_binding(
        *binding_registry_, handle_leases_, create_resp, "prefetch_serving_caller_cancelled");
    return {grpc::StatusCode::CANCELLED, "PrefetchServingBinding request was cancelled"};
  }

  auto record_or = binding_registry_->get(create_resp.binding_id());
  if (!record_or.ok()) {
    cleanup_prefetch_created_binding(
        *binding_registry_, handle_leases_, create_resp, "prefetch_serving_missing_record");
    return to_grpc_status(record_or.status());
  }
  const bool created_staged_value = create_resp.created_staged_value();
  const v2::BindingValue& serving_value =
      created_staged_value ? create_resp.staged_value() : create_resp.current_value();
  if (serving_value.binding_value_id().empty()) {
    cleanup_prefetch_created_binding(*binding_registry_, handle_leases_, create_resp, "prefetch_serving_empty_value");
    return {grpc::StatusCode::FAILED_PRECONDITION, "serving prefetch produced no binding value"};
  }

  const absl::Time now = absl::Now();
  const auto& policy = req.retention_policy();
  const auto unacquired_ttl = duration_from_policy_ms(
      policy.has_expire_if_unacquired_after_ms(),
      policy.expire_if_unacquired_after_ms(),
      serving_prefetch_.default_expire_if_unacquired);
  const auto idle_ttl = duration_from_policy_ms(
      policy.has_idle_ttl_after_last_release_ms(),
      policy.idle_ttl_after_last_release_ms(),
      serving_prefetch_.default_idle_ttl_after_last_release);
  (void)idle_ttl;
  const uint64_t reservation_bytes = logical_total_size(*target_layout_or);
  const std::string capability_id =
      absl::StrCat("capability:", create_resp.binding_id(), ":", serving_value.binding_value_id());
  const absl::Time expires_at = now + absl::Milliseconds(unacquired_ttl.count());

  tensorcast::operation::v1::PrefetchServingBindingResult result;
  bool staged_value_missing = false;
  {
    const auto record = *record_or;
    absl::MutexLock lock(&record->mu);
    record->control_lifetime = BindingRegistry::ControlLifetime::kDaemonRetained;
    record->retained_ref = true;
    record->creator_pid = static_cast<int32_t>(::getpid());
    record->owner_pid = 0;
    record->state = v2::BINDING_STATE_READY_LOCAL;
    record->daemon_id = daemon_id_;
    record->daemon_session_id = daemon_session_id_;
    record->serving_member = target.member();
    record->serving_build_digest = target.serving_build_digest();
    record->target_layout_hash = layout.target_layout_hash();
    record->tensor_schema_hash = layout.tensor_schema_hash();
    record->reservation_capability_id = capability_id;
    record->reservation_expires_at = expires_at;
    record->unacquired_deadline = expires_at;
    record->idle_deadline = absl::InfiniteFuture();
    record->materialization_deadline = absl::InfiniteFuture();
    record->verification_state = v2::BINDING_VALUE_VERIFICATION_STATE_LOCAL_ONLY;
    record->source_artifact_ref = target.source().has_source_artifact_ref() ? target.source().source_artifact_ref()
                                                                            : req.source_selection().artifact_id();
    record->local_serving_ref =
        absl::StrCat("binding-local:", record->binding_id, ":", serving_value.binding_value_id());
    record->serving_artifact_id.clear();
    if (created_staged_value) {
      auto staged_it = record->staged_values_by_id.find(serving_value.binding_value_id());
      if (staged_it == record->staged_values_by_id.end()) {
        staged_value_missing = true;
      } else {
        staged_it->second.target_layout_hash = record->target_layout_hash;
        staged_it->second.tensor_schema_hash = record->tensor_schema_hash;
        staged_it->second.expires_at = expires_at;
        staged_it->second.verification_state = record->verification_state;
      }
    }

    if (!staged_value_missing) {
      result.set_local_serving_ref(record->local_serving_ref);
      if (created_staged_value) {
        fill_binding_value_ref(serving_value, result.mutable_binding_value_ref());
      } else {
        fill_binding_value_ref(*record, result.mutable_binding_value_ref());
      }
      result.set_daemon_id(record->daemon_id);
      result.set_daemon_session_id(record->daemon_session_id);
      result.set_device_uuid(record->device_uuid);
      result.mutable_member()->CopyFrom(record->serving_member);
      result.set_reservation_bytes(reservation_bytes);
      result.set_readiness(tensorcast::operation::v1::SERVING_BINDING_READINESS_LOCAL_READY);
      result.set_verification_state(verification_state_string(record->verification_state));
      result.set_expires_at_ms(static_cast<uint64_t>(absl::ToUnixMillis(expires_at)));
      result.set_staged_value(created_staged_value);
      if (created_staged_value) {
        result.set_group_realization_transaction_id(create_resp.group_realization_acquire().transaction_id());
        result.set_group_realization_version_set_id(create_resp.group_realization_acquire().version_set_id());
        result.set_group_realization_part_id(create_resp.group_realization_acquire().part_id());
        result.set_group_realization_staging_token(create_resp.group_realization_acquire().staging_token());
      }
      auto* capability = result.mutable_reservation_capability();
      capability->set_capability_id(capability_id);
      capability->mutable_binding_value_ref()->CopyFrom(result.binding_value_ref());
      capability->set_daemon_id(record->daemon_id);
      capability->set_daemon_session_id(record->daemon_session_id);
      capability->set_device_uuid(record->device_uuid);
      capability->mutable_member()->CopyFrom(record->serving_member);
      capability->set_reservation_bytes(reservation_bytes);
      capability->set_scope_digest(
          absl::StrCat(record->target_layout_hash, ":", record->tensor_schema_hash, ":", record->serving_build_digest));
      capability->set_expires_at_ms(static_cast<uint64_t>(absl::ToUnixMillis(expires_at)));
    }
  }

  if (staged_value_missing) {
    cleanup_prefetch_created_binding(
        *binding_registry_, handle_leases_, create_resp, "prefetch_serving_missing_staged");
    return {grpc::StatusCode::FAILED_PRECONDITION, "staged serving binding value missing from registry"};
  }
  if (rctx.server_context().IsCancelled()) {
    cleanup_prefetch_created_binding(
        *binding_registry_, handle_leases_, create_resp, "prefetch_serving_caller_cancelled");
    return {grpc::StatusCode::CANCELLED, "PrefetchServingBinding request was cancelled"};
  }

  if (!create_resp.mem_handle().lease_token().empty()) {
    auto release_status = handle_leases_->release(create_resp.mem_handle().lease_token());
    if (!release_status.ok()) {
      return to_grpc_status(release_status);
    }
  }

  auto* op_ref = resp.mutable_operation_ref();
  op_ref->set_operation_id(
      req.has_operation_id() && !req.operation_id().empty()
          ? req.operation_id()
          : absl::StrCat("prefetch-serving-binding:", create_resp.binding_id()));
  op_ref->set_kind("prefetch_serving_binding");
  op_ref->set_target_artifact_id(req.source_selection().artifact_id());
  op_ref->set_authority_scope_kind("daemon_retained_binding");
  op_ref->set_authority_scope_id(create_resp.binding_id());
  op_ref->set_attachment_kind("serving_binding_value");
  op_ref->set_recovery_class("daemon_retained_local");

  auto* status = resp.mutable_status();
  status->set_state(tensorcast::operation::v1::OPERATION_STATE_SUCCESS);
  status->set_message("serving binding is local-ready");
  status->set_progress(1.0);
  status->mutable_result()->PackFrom(result);
  rctx.mark_success();
  return grpc::Status::OK;
}

grpc::Status MaterializationController::acquire_binding_value(
    RpcContext& rctx,
    const v2::AcquireBindingValueRequest& req,
    v2::AcquireBindingValueResponse& resp) {
  if (binding_registry_ == nullptr) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "binding registry unavailable"};
  }
  if (handle_leases_ == nullptr) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "local handle plane is disabled (no handle leases)"};
  }
  if (!req.has_caller_pid() || req.caller_pid() <= 0) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "caller_pid is required"};
  }
  if (!is_loopback_grpc_peer(rctx.server_context().peer())) {
    return {grpc::StatusCode::PERMISSION_DENIED, "AcquireBindingValue is local-only (loopback/UDS)"};
  }
  if (!req.has_group_realization_acquire() && req.binding_value_ref().binding_id().empty() &&
      !req.local_serving_ref().empty()) {
    if (req.expected_device_uuid().empty()) {
      return {grpc::StatusCode::INVALID_ARGUMENT, "expected_device_uuid is required"};
    }
    if (!req.has_expected_member() || req.expected_member().member_id().empty()) {
      return {grpc::StatusCode::INVALID_ARGUMENT, "expected_member is required"};
    }
    if (req.expected_tensor_schema_hash().empty()) {
      return {grpc::StatusCode::INVALID_ARGUMENT, "expected_tensor_schema_hash is required"};
    }
    if (req.expected_serving_build_digest().empty()) {
      return {grpc::StatusCode::INVALID_ARGUMENT, "expected_serving_build_digest is required"};
    }

    auto record_or = binding_registry_->get_by_local_serving_ref(req.local_serving_ref());
    if (!record_or.ok()) {
      return to_grpc_status(record_or.status());
    }
    const auto record = *record_or;
    std::string binding_id;
    bool retained_ref = false;
    {
      absl::MutexLock lock(&record->mu);
      if (record->retired) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "binding is retired"};
      }
      retained_ref = record->retained_ref;
      if (!retained_ref && record->owner_pid != req.caller_pid()) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "binding is not retained"};
      }
      if (record->current_binding_value_id.empty()) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "binding has no current value"};
      }
      if (!req.expected_daemon_id().empty() && record->daemon_id != req.expected_daemon_id()) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "daemon_id mismatch"};
      }
      if (!req.expected_daemon_session_id().empty() && record->daemon_session_id != req.expected_daemon_session_id()) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "daemon_session_id mismatch"};
      }
      if (record->device_uuid != req.expected_device_uuid()) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "device_uuid mismatch"};
      }
      if (!record->serving_member.member_id().empty() &&
          !same_serving_member_ref(record->serving_member, req.expected_member())) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "member mismatch"};
      }
      if (!req.expected_target_layout_hash().empty() &&
          record->target_layout_hash != req.expected_target_layout_hash()) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "target_layout_hash mismatch"};
      }
      if (retained_ref && record->tensor_schema_hash != req.expected_tensor_schema_hash()) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "tensor_schema_hash mismatch"};
      }
      if (retained_ref && record->serving_build_digest != req.expected_serving_build_digest()) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "serving_build_digest mismatch"};
      }
      if (record->allowed_caller_pid.has_value() && *record->allowed_caller_pid != req.caller_pid()) {
        return {grpc::StatusCode::PERMISSION_DENIED, "caller_pid mismatch"};
      }
      binding_id = record->binding_id;
    }

    if (retained_ref) {
      if (auto status = binding_registry_->acquire_attachment_ref(binding_id, absl::Now()); !status.ok()) {
        return to_grpc_status(status);
      }
    }
    auto token_or = retained_ref ? handle_leases_->mint_external_cuda_lease(
                                       req.caller_pid(),
                                       [registry = binding_registry_, binding_id]() {
                                         registry->release_attachment_ref(binding_id, absl::Now());
                                       })
                                 : handle_leases_->mint_external_cuda_lease(req.caller_pid(), []() {});
    if (!token_or.ok()) {
      if (retained_ref) {
        binding_registry_->release_attachment_ref(binding_id, absl::Now());
      }
      return to_grpc_status(token_or.status());
    }
    auto refreshed_or = binding_registry_->get(binding_id);
    if (!refreshed_or.ok()) {
      auto release_status = handle_leases_->release(*token_or);
      if (!release_status.ok()) {
        LOG(WARNING) << "failed to release local-ref acquire lease after binding lookup failure: " << release_status;
      }
      return to_grpc_status(refreshed_or.status());
    }
    const auto refreshed = *refreshed_or;
    absl::MutexLock lock(&refreshed->mu);
    resp.set_lease_token(*token_or);
    resp.mutable_mem_handle()->set_cuda_ipc_handle(
        refreshed->handle_bytes.as_string_view().data(), refreshed->handle_bytes.as_string_view().size());
    resp.mutable_mem_handle()->set_lease_token(*token_or);
    resp.set_target_index_bytes(refreshed->target_index_json);
    for (const auto& payload : refreshed->payloads) {
      *resp.add_payloads() = payload;
    }
    resp.set_reservation_bytes(logical_total_size(refreshed->target_layout));
    fill_current_binding_value(*refreshed, resp.mutable_current_value());
    resp.mutable_acquired_value()->CopyFrom(resp.current_value());
    return grpc::Status::OK;
  }
  if (req.has_group_realization_acquire()) {
    const auto publish_check =
        verify_group_realization_published(global_store_client_, req.group_realization_acquire());
    if (!publish_check.status.ok()) {
      if (publish_check.terminal_not_published) {
        const auto cleanup_status = binding_registry_->remove_staged_value(
            req.binding_value_ref().binding_id(),
            req.binding_value_ref().binding_value_id(),
            "group_realization_terminal_not_published");
        if (!cleanup_status.ok() && !absl::IsNotFound(cleanup_status)) {
          LOG(WARNING) << "failed to remove terminal group staged binding value binding_id="
                       << req.binding_value_ref().binding_id()
                       << " binding_value_id=" << req.binding_value_ref().binding_value_id() << ": " << cleanup_status;
        }
      }
      return publish_check.status;
    }
    if (auto status = binding_registry_->validate_and_acquire_group_staged_attachment_ref(req, absl::Now());
        !status.ok()) {
      return to_grpc_status(status);
    }
    const std::string binding_id = req.binding_value_ref().binding_id();
    const std::string binding_value_id = req.binding_value_ref().binding_value_id();
    auto token_or = handle_leases_->mint_external_cuda_lease(
        req.caller_pid(),
        [registry = binding_registry_, binding_id]() { registry->release_attachment_ref(binding_id, absl::Now()); });
    if (!token_or.ok()) {
      binding_registry_->release_attachment_ref(binding_id, absl::Now());
      return to_grpc_status(token_or.status());
    }

    auto record_or = binding_registry_->get(binding_id);
    if (!record_or.ok()) {
      auto release_status = handle_leases_->release(*token_or);
      if (!release_status.ok()) {
        LOG(WARNING) << "failed to release staged acquire lease after binding lookup failure: " << release_status;
      }
      return to_grpc_status(record_or.status());
    }
    auto staged_or = binding_registry_->get_staged_value(binding_id, binding_value_id);
    if (!staged_or.ok()) {
      auto release_status = handle_leases_->release(*token_or);
      if (!release_status.ok()) {
        LOG(WARNING) << "failed to release staged acquire lease after staged lookup failure: " << release_status;
      }
      return to_grpc_status(staged_or.status());
    }
    const auto record = *record_or;
    const auto staged = *staged_or;
    resp.set_lease_token(*token_or);
    resp.mutable_mem_handle()->set_cuda_ipc_handle(
        staged.handle_bytes.as_string_view().data(), staged.handle_bytes.as_string_view().size());
    resp.mutable_mem_handle()->set_lease_token(*token_or);
    resp.set_target_index_bytes(staged.target_index_json);
    for (const auto& payload : staged.payloads) {
      *resp.add_payloads() = payload;
    }
    resp.set_reservation_bytes(req.reservation_capability().reservation_bytes());
    resp.set_acquired_staged_value(true);
    fill_staged_binding_value(*record, staged, resp.mutable_acquired_value());
    return grpc::Status::OK;
  }
  if (auto status = binding_registry_->validate_and_acquire_attachment_ref(req, absl::Now()); !status.ok()) {
    return to_grpc_status(status);
  }
  const std::string binding_id = req.binding_value_ref().binding_id();
  auto token_or = handle_leases_->mint_external_cuda_lease(
      req.caller_pid(),
      [registry = binding_registry_, binding_id]() { registry->release_attachment_ref(binding_id, absl::Now()); });
  if (!token_or.ok()) {
    binding_registry_->release_attachment_ref(binding_id, absl::Now());
    return to_grpc_status(token_or.status());
  }

  auto record_or = binding_registry_->get(binding_id);
  if (!record_or.ok()) {
    auto release_status = handle_leases_->release(*token_or);
    if (!release_status.ok()) {
      LOG(WARNING) << "failed to release acquire lease after binding lookup failure: " << release_status;
    }
    return to_grpc_status(record_or.status());
  }
  const auto record = *record_or;
  absl::MutexLock lock(&record->mu);
  resp.set_lease_token(*token_or);
  resp.mutable_mem_handle()->set_cuda_ipc_handle(
      record->handle_bytes.as_string_view().data(), record->handle_bytes.as_string_view().size());
  resp.mutable_mem_handle()->set_lease_token(*token_or);
  resp.set_target_index_bytes(record->target_index_json);
  for (const auto& payload : record->payloads) {
    *resp.add_payloads() = payload;
  }
  resp.set_reservation_bytes(req.reservation_capability().reservation_bytes());
  fill_current_binding_value(*record, resp.mutable_current_value());
  resp.mutable_acquired_value()->CopyFrom(resp.current_value());
  return grpc::Status::OK;
}

absl::StatusOr<TargetPublicationRegistry::Record> MaterializationController::insert_target_publication_for_testing(
    TargetPublicationRegistry::Record record) {
  return target_materialization_service_.insert_target_publication_for_testing(std::move(record));
}

grpc::Status MaterializationController::materialize_replica(
    RpcContext& rctx,
    const v2::MaterializeReplicaRequest& req,
    v2::MaterializeReplicaResponse& resp) {
  return replica_materialization_service_.materialize_replica(rctx, req, resp);
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

grpc::Status MaterializationController::create_owned_binding(
    RpcContext& rctx,
    const v2::CreateOwnedBindingRequest& req,
    v2::CreateOwnedBindingResponse& resp) {
  return owner_binding_service_.create_owned_binding(rctx, req, resp);
}

grpc::Status MaterializationController::create_binding(
    RpcContext& rctx,
    const v2::CreateBindingRequest& req,
    v2::CreateBindingResponse& resp) {
  return owner_binding_service_.create_binding(rctx, req, resp);
}

grpc::Status MaterializationController::commit_binding_artifact(
    RpcContext& rctx,
    const v2::CommitBindingArtifactRequest& req,
    v2::CommitBindingArtifactResponse& resp) {
  return owner_binding_service_.commit_binding_artifact(rctx, req, resp);
}

grpc::Status MaterializationController::begin_binding_update(
    RpcContext& rctx,
    const v2::BeginBindingUpdateRequest& req,
    v2::BeginBindingUpdateResponse& resp) {
  return owner_binding_service_.begin_binding_update(rctx, req, resp);
}

grpc::Status MaterializationController::submit_binding_contribution(
    RpcContext& rctx,
    const v2::SubmitBindingContributionRequest& req,
    v2::SubmitBindingContributionResponse& resp) {
  return owner_binding_service_.submit_binding_contribution(rctx, req, resp);
}

grpc::Status MaterializationController::freeze_binding_current_value(
    RpcContext& rctx,
    const v2::FreezeBindingCurrentValueRequest& req,
    v2::FreezeBindingCurrentValueResponse& resp) {
  return owner_binding_service_.freeze_binding_current_value(rctx, req, resp);
}

grpc::Status MaterializationController::seal_binding(
    RpcContext& rctx,
    const v2::SealBindingRequest& req,
    v2::SealBindingResponse& resp) {
  return owner_binding_service_.seal_binding(rctx, req, resp);
}

grpc::Status MaterializationController::promote_binding_current_value(
    RpcContext& rctx,
    const v2::PromoteBindingCurrentValueRequest& req,
    v2::PromoteBindingCurrentValueResponse& resp) {
  return owner_binding_service_.promote_binding_current_value(rctx, req, resp);
}

grpc::Status MaterializationController::start_promote_binding_current_value(
    RpcContext& rctx,
    const v2::StartPromoteBindingCurrentValueRequest& req,
    v2::StartPromoteBindingCurrentValueResponse& resp) {
  return owner_binding_service_.start_promote_binding_current_value(rctx, req, resp);
}

grpc::Status MaterializationController::get_binding_promotion_status(
    RpcContext& rctx,
    const v2::GetBindingPromotionStatusRequest& req,
    v2::GetBindingPromotionStatusResponse& resp) {
  return owner_binding_service_.get_binding_promotion_status(rctx, req, resp);
}

grpc::Status MaterializationController::refill_owned_binding(
    RpcContext& rctx,
    const v2::RefillOwnedBindingRequest& req,
    v2::RefillOwnedBindingResponse& resp) {
  return owner_binding_service_.refill_owned_binding(rctx, req, resp);
}

grpc::Status MaterializationController::close_owned_binding(
    RpcContext& rctx,
    const v2::CloseOwnedBindingRequest& req,
    v2::CloseOwnedBindingResponse& resp) {
  return owner_binding_service_.close_owned_binding(rctx, req, resp);
}

grpc::Status MaterializationController::publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::PublishTargetReplicaResponse& resp) {
  return target_materialization_service_.publish_target_replica(rctx, req, resp);
}

grpc::Status MaterializationController::start_publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::StartPublishTargetReplicaResponse& resp) {
  return target_materialization_service_.start_publish_target_replica(rctx, req, resp);
}

absl::StatusOr<TargetPublishService::TargetPublicationFrontDoorContext> MaterializationController::
    inspect_target_publication_context_for_testing(const v2::PublishTargetReplicaRequest& req, absl::Time now) {
  return target_materialization_service_.inspect_target_publication_context_for_testing(req, now);
}

absl::StatusOr<RoutedAuthorityRequest> MaterializationController::
    build_target_publication_workflow_routed_request_for_testing(
        const v2::PublishTargetReplicaRequest& req,
        absl::Time now) const {
  return target_materialization_service_.build_target_publication_workflow_routed_request_for_testing(req, now);
}

absl::StatusOr<RoutedAuthorityRequest> MaterializationController::
    build_target_publication_workflow_continuation_request_for_testing(
        const RoutedAuthorityRequest& routed_request,
        const OwnerStageReply& workflow_gate_reply) const {
  return target_materialization_service_.build_target_publication_workflow_continuation_request_for_testing(
      routed_request, workflow_gate_reply);
}

absl::StatusOr<std::optional<OwnerStageReply>> MaterializationController::maybe_route_authority_stage(
    const RoutedAuthorityRequest& routed_request,
    absl::Time now) {
  return target_materialization_service_.maybe_route_authority_stage(routed_request, now);
}

grpc::Status MaterializationController::import_artifact_from_path(
    RpcContext& rctx,
    const v2::ImportArtifactFromPathRequest& req,
    v2::ImportArtifactFromPathResponse& resp) {
  return disk_artifact_service_.import_artifact_from_path(rctx, req, resp);
}

grpc::Status MaterializationController::resolve_public_disk_source(
    RpcContext& rctx,
    const v2::ResolvePublicDiskSourceRequest& req,
    v2::ResolvePublicDiskSourceResponse& resp) {
  return disk_artifact_service_.resolve_public_disk_source(rctx, req, resp);
}

grpc::Status MaterializationController::promote_mounted_source_artifact(
    RpcContext& rctx,
    const v2::PromoteMountedSourceArtifactRequest& req,
    v2::PromoteMountedSourceArtifactResponse& resp) {
  return disk_artifact_service_.promote_mounted_source_artifact(rctx, req, resp);
}

grpc::Status MaterializationController::import_artifact_from_path_stream(
    RpcContext& rctx,
    const v2::ImportArtifactFromPathRequest& req,
    grpc::ServerWriter<v2::ImportArtifactFromPathStreamEvent>& writer) {
  return disk_artifact_service_.import_artifact_from_path_stream(rctx, req, writer);
}

grpc::Status MaterializationController::get_artifact_index_by_id(
    RpcContext& rctx,
    const v2::GetArtifactIndexByIdRequest& req,
    v2::GetArtifactIndexByIdResponse& resp) {
  return disk_artifact_service_.get_artifact_index_by_id(rctx, req, resp);
}

grpc::Status MaterializationController::list_artifact_layouts(
    RpcContext& rctx,
    const v2::ListArtifactLayoutsRequest& req,
    v2::ListArtifactLayoutsResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.artifact_id());
  if (req.artifact_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }
  auto layouts_or = global_store_client_->list_artifact_layouts(req.artifact_id());
  if (!layouts_or.ok()) {
    return to_grpc_status(layouts_or.status());
  }
  for (const auto& layout_id : *layouts_or) {
    resp.add_layout_ids(layout_id);
  }
  return grpc::Status::OK;
}

grpc::Status MaterializationController::seal_assembly(
    RpcContext& rctx,
    const v2::SealAssemblyRequest& req,
    v2::SealAssemblyResponse& resp) {
  return assembly_operation_service_.seal_assembly(rctx, req, resp);
}

grpc::Status MaterializationController::start_assembly_attempt(
    RpcContext& rctx,
    const v2::StartAssemblyAttemptRequest& req,
    v2::StartAssemblyAttemptResponse& resp) {
  return assembly_operation_service_.start_assembly_attempt(rctx, req, resp);
}

grpc::Status MaterializationController::seal_assembly_attempt(
    RpcContext& rctx,
    const v2::SealAssemblyAttemptRequest& req,
    v2::SealAssemblyAttemptResponse& resp) {
  return assembly_operation_service_.seal_assembly_attempt(rctx, req, resp);
}

grpc::Status MaterializationController::start_seal_assembly(
    RpcContext& rctx,
    const v2::StartSealAssemblyRequest& req,
    v2::StartSealAssemblyResponse& resp) {
  return assembly_operation_service_.start_seal_assembly(rctx, req, resp);
}

grpc::Status MaterializationController::get_operation(
    RpcContext& rctx,
    const tensorcast::operation::v1::GetOperationRequest& req,
    tensorcast::operation::v1::GetOperationResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.operation.id", req.operation_id());

  if (req.operation_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "operation_id is required"};
  }
  if (shutdown_signal_ != nullptr && shutdown_signal_->is_shutting_down()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  auto op_or = global_store_client_->get_operation(req);
  if (!op_or.ok()) {
    return to_grpc_status(op_or.status());
  }
  if (req.has_ref()) {
    merge_operation_ref_metadata(req.ref(), op_or->mutable_ref());
  }
  auto admit_status = public_operation_admission_service_.admit(op_or->ref(), absl::Now());
  if (!admit_status.ok()) {
    return to_grpc_status(admit_status);
  }
  resp = std::move(*op_or);
  rctx.mark_success();
  return grpc::Status::OK;
}

grpc::Status MaterializationController::wait_operation(
    RpcContext& rctx,
    const v2::WaitOperationRequest& req,
    v2::WaitOperationResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.operation.id", req.operation_id());

  if (req.operation_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "operation_id is required"};
  }
  if (shutdown_signal_ != nullptr && shutdown_signal_->is_shutting_down()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  const uint64_t timeout_ms = req.timeout_ms();
  const absl::Time start = absl::Now();
  const absl::Time deadline = timeout_ms > 0 ? start + absl::Milliseconds(timeout_ms) : absl::InfiniteFuture();
  absl::Duration sleep = absl::Milliseconds(50);

  tensorcast::operation::v1::GetOperationRequest get_req;
  get_req.set_operation_id(req.operation_id());
  if (req.has_ref()) {
    get_req.mutable_ref()->CopyFrom(req.ref());
  }

  while (absl::Now() < deadline) {
    auto op_or = global_store_client_->get_operation(get_req);
    if (!op_or.ok()) {
      return to_grpc_status(op_or.status());
    }
    if (req.has_ref()) {
      merge_operation_ref_metadata(req.ref(), op_or->mutable_ref());
    }
    auto admit_status = public_operation_admission_service_.admit(op_or->ref(), absl::Now());
    if (!admit_status.ok()) {
      return to_grpc_status(admit_status);
    }
    const auto state = op_or->status().state();
    resp.mutable_operation()->Swap(&(*op_or));
    if (operation_state_is_terminal(state)) {
      rctx.mark_success();
      return grpc::Status::OK;
    }
    absl::SleepFor(sleep);
    sleep = std::min(sleep * 12 / 10, absl::Milliseconds(500));
  }

  auto op_or = global_store_client_->get_operation(get_req);
  if (!op_or.ok()) {
    return to_grpc_status(op_or.status());
  }
  if (req.has_ref()) {
    merge_operation_ref_metadata(req.ref(), op_or->mutable_ref());
  }
  auto admit_status = target_materialization_service_.admit_public_operation(op_or->ref(), absl::Now());
  if (!admit_status.ok()) {
    return to_grpc_status(admit_status);
  }
  resp.mutable_operation()->Swap(&(*op_or));
  rctx.mark_success();
  return grpc::Status::OK;
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
