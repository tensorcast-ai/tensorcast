// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/grpc_service_impl.h"

#include <nlohmann/json.hpp>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "core/store/components/global_store_client.h"
#include "daemon/state/store_policy_resolver.h"
#include "daemon/util/path_utils.h"
#include "daemon/util/status_utils.h"
#include "folly/futures/Future.h"
#include "google/protobuf/util/time_util.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

std::string status_code_name(grpc::StatusCode code) {
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

bool is_retryable(grpc::StatusCode code) {
  switch (code) {
    case grpc::StatusCode::UNAVAILABLE:
    case grpc::StatusCode::DEADLINE_EXCEEDED:
    case grpc::StatusCode::RESOURCE_EXHAUSTED:
    case grpc::StatusCode::ABORTED:
      return true;
    default:
      return false;
  }
}

std::string replica_key_hash_bytes(const store::loading::ReplicaKey& key) {
  const uint64_t h = static_cast<uint64_t>(store::loading::ReplicaKeyHash{}(key));
  std::string out(sizeof(h), '\0');
  std::memcpy(out.data(), &h, sizeof(h));
  return out;
}

void fill_success_status(v2::ReplicaOperationStatus& out) {
  out.set_state(v2::ReplicaOperationState::REPLICA_OPERATION_STATE_SUCCESS);
  *out.mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
}

void fill_running_status(v2::ReplicaOperationStatus& out) {
  out.set_state(v2::ReplicaOperationState::REPLICA_OPERATION_STATE_RUNNING);
  *out.mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
}

void fill_failed_status(v2::ReplicaOperationStatus& out, const absl::Status& st) {
  grpc::Status grpc_st = status_utils::to_grpc_status(st);
  out.set_state(v2::ReplicaOperationState::REPLICA_OPERATION_STATE_FAILED);
  out.set_message(std::string(st.message()));
  *out.mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
  auto* err = out.mutable_error();
  err->set_status_code(status_code_name(grpc_st.error_code()));
  err->set_message(std::string(st.message()));
  err->set_retryable(is_retryable(grpc_st.error_code()));
}

absl::StatusOr<std::optional<std::string>> parse_view_id(const tensorcast::common::v1::ByteSpaceRef& space) {
  switch (space.kind()) {
    case tensorcast::common::v1::BYTE_SPACE_KIND_UNSPECIFIED:
    case tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL:
      return std::nullopt;
    case tensorcast::common::v1::BYTE_SPACE_KIND_VIEW:
      if (space.id().empty()) {
        return absl::InvalidArgumentError("byte_space VIEW requires id");
      }
      return std::optional<std::string>(space.id());
    default:
      return absl::InvalidArgumentError("unsupported byte_space kind");
  }
}

uint32_t remaining_timeout_ms(absl::Time deadline, uint32_t timeout_ms) {
  if (timeout_ms == 0) {
    return 0;
  }
  const absl::Duration remaining = deadline - absl::Now();
  if (remaining <= absl::ZeroDuration()) {
    return 0;
  }
  const int64_t remaining_ms = absl::ToInt64Milliseconds(remaining);
  const int64_t clamped_ms = std::min<int64_t>(remaining_ms, timeout_ms);
  return static_cast<uint32_t>(clamped_ms);
}

struct DrainOutcome {
  bool drained{true};
  std::optional<std::string> replica_id;
};

absl::StatusOr<DrainOutcome> retire_replica_with_drain(
    LipManager* lip_manager,
    store::components::IGlobalStoreClient* global_store_client,
    std::string_view artifact_id,
    const ArtifactDeviceKey& key,
    bool wait_for_drain,
    uint32_t timeout_ms,
    std::optional<std::string_view> operation_id) {
  DrainOutcome outcome{.drained = !wait_for_drain, .replica_id = std::nullopt};
  const absl::Time deadline = absl::Now() + absl::Milliseconds(timeout_ms);

  outcome.replica_id = lip_manager->find_replica_id(key);
  if (outcome.replica_id.has_value() && !outcome.replica_id->empty()) {
    if (global_store_client == nullptr || !global_store_client->is_connected()) {
      return absl::FailedPreconditionError("Global Store client unavailable");
    }
    auto mark_or = global_store_client->mark_replica_unavailable(
        artifact_id,
        *outcome.replica_id,
        /*reason=*/"retire",
        operation_id);
    if (!mark_or.ok() && !absl::IsNotFound(mark_or.status())) {
      return mark_or.status();
    }
    if (wait_for_drain) {
      const uint32_t remaining_ms = remaining_timeout_ms(deadline, timeout_ms);
      auto drain_or = global_store_client->wait_replica_drain(*outcome.replica_id, remaining_ms, operation_id);
      if (!drain_or.ok() && !absl::IsNotFound(drain_or.status())) {
        return drain_or.status();
      }
      if (drain_or.ok() && !drain_or->drained) {
        return absl::DeadlineExceededError("drain timed out; artifact remains quiesced");
      }
    }
  }

  if (wait_for_drain) {
    const bool drained = lip_manager->wait_exports_drained(key, deadline);
    if (!drained) {
      return absl::DeadlineExceededError("drain timed out; artifact remains quiesced");
    }
    outcome.drained = true;
  }
  return outcome;
}

void fill_degraded_timeout_status(v2::ReplicaOperationStatus& out, std::string_view message) {
  out.set_state(v2::ReplicaOperationState::REPLICA_OPERATION_STATE_DEGRADED);
  out.set_message(std::string(message));
  *out.mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
  auto* err = out.mutable_error();
  err->set_status_code("DEADLINE_EXCEEDED");
  err->set_message(std::string(message));
  err->set_retryable(true);
}

void append_deregister_message(v2::DeregisterArtifactResponse* resp, std::string_view msg) {
  if (resp == nullptr || msg.empty()) {
    return;
  }
  if (!resp->message().empty()) {
    resp->set_message(absl::StrCat(resp->message(), "; ", msg));
    return;
  }
  resp->set_message(std::string(msg));
}

bool is_relative_path_under_prefix(
    const std::filesystem::path& relative_path,
    const std::filesystem::path& expected_prefix) {
  if (!std::filesystem::path{relative_path}.has_relative_path() || relative_path.is_absolute()) {
    return false;
  }
  auto rel_it = relative_path.begin();
  for (auto prefix_it = expected_prefix.begin(); prefix_it != expected_prefix.end(); ++prefix_it, ++rel_it) {
    if (rel_it == relative_path.end() || *rel_it != *prefix_it) {
      return false;
    }
  }
  return true;
}

void purge_managed_shared_disk_artifact(
    store::components::IGlobalStoreClient* global_store_client,
    std::string_view artifact_id,
    const std::filesystem::path& storage_path,
    v2::DeregisterArtifactResponse* resp) {
  if (global_store_client == nullptr || !global_store_client->is_connected()) {
    append_deregister_message(resp, "shared disk purge skipped: Global Store client unavailable");
    return;
  }

  auto cluster_or = global_store_client->get_cluster_id();
  if (!cluster_or.ok() || cluster_or->empty()) {
    append_deregister_message(resp, "shared disk purge skipped: cluster_id unavailable");
    return;
  }

  const std::string& cluster_id = *cluster_or;
  auto locations_or =
      global_store_client->list_artifact_disk_locations(std::string(artifact_id), /*include_deleted=*/true);
  if (!locations_or.ok()) {
    append_deregister_message(resp, "shared disk purge skipped: disk locations unavailable");
    return;
  }

  const std::filesystem::path expected_prefix = std::filesystem::path("clusters") / cluster_id / "objects";
  std::vector<std::string> managed_paths;
  managed_paths.reserve(locations_or->size());
  for (const auto& location : *locations_or) {
    if (location.cluster_id != cluster_id) {
      continue;
    }
    if (location.kind != tensorcast::global_store::v1::DISK_LOCATION_KIND_MANAGED) {
      continue;
    }
    if (location.relative_path.empty()) {
      continue;
    }
    const std::filesystem::path relative_path(location.relative_path);
    if (!is_relative_path_under_prefix(relative_path, expected_prefix)) {
      continue;
    }
    managed_paths.push_back(location.relative_path);
  }

  for (const auto& relative_path : managed_paths) {
    // Tombstone first so disk fallback stops advertising the path.
    absl::Status st = global_store_client->upsert_artifact_disk_location(
        std::string(artifact_id),
        cluster_id,
        relative_path,
        tensorcast::global_store::v1::DISK_LOCATION_KIND_MANAGED,
        /*is_deleted=*/true);
    if (!st.ok()) {
      append_deregister_message(resp, absl::StrCat("shared disk tombstone failed: ", st.message()));
    }
  }

  if (storage_path.empty()) {
    append_deregister_message(resp, "shared disk purge skipped: storage_path missing");
    return;
  }

  for (const auto& relative_path : managed_paths) {
    auto normalized_or = normalize_disk_path(relative_path, storage_path);
    if (!normalized_or.ok()) {
      append_deregister_message(
          resp, absl::StrCat("shared disk purge path rejected: ", normalized_or.status().message()));
      continue;
    }
    std::error_code error_code;
    std::filesystem::remove_all(*normalized_or, error_code);
    if (error_code) {
      append_deregister_message(resp, absl::StrCat("shared disk purge failed: ", error_code.message()));
    }
  }
}

} // namespace

StoreDaemonServiceImpl::StoreDaemonServiceImpl(Deps deps, Options opts)
    : engine_(&deps.engine),
      materialization_controller_(&deps.materialization_controller),
      registration_controller_(&deps.registration_controller),
      transport_controller_(&deps.transport_controller),
      status_controller_(&deps.status_controller),
      region_registry_(&deps.region_registry),
      lip_manager_(&deps.lip_manager),
      global_store_client_(std::move(deps.global_store_client)),
      persistence_manager_(deps.persistence_manager),
      sessions_service_(&deps.sessions_service),
      lifecycle_manager_(&deps.lifecycle_manager),
      lease_controller_(&deps.lease_controller),
      shutdown_signal_(&deps.shutdown_signal),
      opts_(std::move(opts)) {}

Status StoreDaemonServiceImpl::MaterializeReplica(
    grpc::ServerContext* ctx,
    const v2::MaterializeReplicaRequest* req,
    v2::MaterializeReplicaResponse* resp) {
  RpcContext rctx{"MaterializeReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->materialize_replica(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::MaterializeIntoTarget(
    grpc::ServerContext* ctx,
    const v2::MaterializeIntoTargetRequest* req,
    v2::MaterializeIntoTargetResponse* resp) {
  RpcContext rctx{"MaterializeIntoTarget", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->materialize_into_target(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::MaterializeIntoMappedTarget(
    grpc::ServerContext* ctx,
    const v2::MaterializeIntoMappedTargetRequest* req,
    v2::MaterializeIntoTargetResponse* resp) {
  RpcContext rctx{"MaterializeIntoMappedTarget", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->materialize_into_mapped_target(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ConfirmReplica(
    grpc::ServerContext* ctx,
    const v2::ConfirmReplicaRequest* req,
    v2::ConfirmReplicaResponse* resp) {
  RpcContext rctx{"ConfirmReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->confirm(rctx, *req, *resp);
}

// Key-based materialization via Global Store mapping.
Status StoreDaemonServiceImpl::MaterializeByKey(
    grpc::ServerContext* ctx,
    const v2::MaterializeByKeyRequest* req,
    v2::MaterializeByKeyResponse* resp) {
  RpcContext rctx{"MaterializeByKey", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->materialize_by_key(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ResolveArtifactFromDisk(
    grpc::ServerContext* ctx,
    const v2::ResolveArtifactFromDiskRequest* req,
    v2::ResolveArtifactFromDiskResponse* resp) {
  RpcContext rctx{"ResolveArtifactFromDisk", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->resolve_artifact_from_disk(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::QueryReplicaStatus(
    grpc::ServerContext* ctx,
    const v2::QueryReplicaStatusRequest* req,
    v2::QueryReplicaStatusResponse* resp) {
  RpcContext rctx{"QueryReplicaStatus", *ctx, opts_.allow_high_card_attrs};
  if (!req->has_ticket() || req->ticket().replica_uuid().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "ticket.replica_uuid is required"};
  }
  const std::string& replica_uuid = req->ticket().replica_uuid();
  auto entry = sessions_service_->get(replica_uuid);
  if (!entry.has_value()) {
    return {StatusCode::NOT_FOUND, "replica_uuid not found"};
  }

  resp->mutable_ticket()->set_replica_uuid(replica_uuid);
  resp->set_replica_key_hash(replica_key_hash_bytes(entry->key));

  auto* status = resp->mutable_status();
  if (!entry->ready_signal) {
    fill_success_status(*status);
  } else if (!entry->ready_signal->is_ready()) {
    fill_running_status(*status);
  } else {
    const absl::Status st = entry->wait_ready(std::chrono::milliseconds(0));
    if (st.ok()) {
      fill_success_status(*status);
    } else {
      fill_failed_status(*status, st);
    }
  }

  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::WaitReplicaStatus(
    grpc::ServerContext* ctx,
    const v2::WaitReplicaStatusRequest* req,
    v2::WaitReplicaStatusResponse* resp) {
  RpcContext rctx{"WaitReplicaStatus", *ctx, opts_.allow_high_card_attrs};
  if (!req->has_ticket() || req->ticket().replica_uuid().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "ticket.replica_uuid is required"};
  }
  const std::string& replica_uuid = req->ticket().replica_uuid();
  auto entry = sessions_service_->get(replica_uuid);
  if (!entry.has_value()) {
    return {StatusCode::NOT_FOUND, "replica_uuid not found"};
  }

  resp->mutable_ticket()->set_replica_uuid(replica_uuid);
  resp->set_replica_key_hash(replica_key_hash_bytes(entry->key));

  auto* status = resp->mutable_status();
  if (!entry->ready_signal) {
    fill_success_status(*status);
    rctx.mark_success();
    return Status::OK;
  }
  if (entry->ready_signal->is_ready()) {
    const absl::Status st = entry->wait_ready(std::chrono::milliseconds(0));
    if (st.ok()) {
      fill_success_status(*status);
    } else {
      fill_failed_status(*status, st);
    }
    rctx.mark_success();
    return Status::OK;
  }

  // timeout_ms is a server-side wait budget. When unset/0, the server waits until the replica reaches a terminal
  // state or until the RPC deadline expires.
  const std::chrono::milliseconds user_timeout(static_cast<int64_t>(req->timeout_ms()));
  const auto start = std::chrono::steady_clock::now();
  const std::optional<std::chrono::steady_clock::time_point> user_deadline = user_timeout.count() > 0
      ? std::optional<std::chrono::steady_clock::time_point>(start + user_timeout)
      : std::nullopt;

  constexpr std::chrono::milliseconds k_wait_slice(250);

  while (true) {
    std::optional<std::chrono::milliseconds> remaining;
    if (user_deadline.has_value()) {
      remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(*user_deadline - std::chrono::steady_clock::now());
    }

    using clock = std::chrono::system_clock;
    const auto grpc_deadline = ctx->deadline();
    if (grpc_deadline != clock::time_point::max()) {
      const auto now = clock::now();
      const auto d = grpc_deadline <= now ? std::chrono::milliseconds(0)
                                          : std::chrono::duration_cast<std::chrono::milliseconds>(grpc_deadline - now);
      remaining = remaining.has_value() ? std::min(*remaining, d) : d;
    }

    if (remaining.has_value() && remaining->count() <= 0) {
      fill_degraded_timeout_status(*status, "replica wait budget exhausted");
      rctx.mark_success();
      return Status::OK;
    }

    if (ctx->IsCancelled()) {
      return {StatusCode::CANCELLED, "request cancelled"};
    }

    const std::chrono::milliseconds slice = remaining.has_value() ? std::min(*remaining, k_wait_slice) : k_wait_slice;
    if (slice.count() <= 0) {
      fill_degraded_timeout_status(*status, "replica wait budget exhausted");
      rctx.mark_success();
      return Status::OK;
    }

    absl::Status wait_st;
    try {
      wait_st = std::move(entry->subscribe_ready()).get(slice);
    } catch (const folly::FutureTimeout&) {
      continue;
    } catch (const std::exception& ex) {
      wait_st = absl::InternalError(ex.what());
    } catch (...) {
      wait_st = absl::InternalError("replica wait_ready failed with unknown exception");
    }

    if (wait_st.ok()) {
      fill_success_status(*status);
    } else if (absl::IsDeadlineExceeded(wait_st)) {
      fill_degraded_timeout_status(*status, wait_st.message());
    } else {
      fill_failed_status(*status, wait_st);
    }
    rctx.mark_success();
    return Status::OK;
  }
}

Status StoreDaemonServiceImpl::ReleaseReplica(
    grpc::ServerContext* ctx,
    const v2::ReleaseReplicaRequest* req,
    v2::ReleaseReplicaResponse* resp) {
  RpcContext rctx{"ReleaseReplica", *ctx, opts_.allow_high_card_attrs};
  if (!req->has_ticket() || req->ticket().replica_uuid().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "ticket.replica_uuid is required"};
  }
  const std::string& replica_uuid = req->ticket().replica_uuid();
  const bool erased = sessions_service_->erase(replica_uuid);
  lifecycle_manager_->release_session(replica_uuid);
  resp->set_released(erased);
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::CreatePlacementLease(
    grpc::ServerContext* ctx,
    const v2::CreatePlacementLeaseRequest* req,
    v2::CreatePlacementLeaseResponse* resp) {
  RpcContext rctx{"CreatePlacementLease", *ctx, opts_.allow_high_card_attrs};
  return lease_controller_->create_placement_lease(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RenewPlacementLease(
    grpc::ServerContext* ctx,
    const v2::RenewPlacementLeaseRequest* req,
    v2::RenewPlacementLeaseResponse* resp) {
  RpcContext rctx{"RenewPlacementLease", *ctx, opts_.allow_high_card_attrs};
  return lease_controller_->renew_placement_lease(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ReleasePlacementLease(
    grpc::ServerContext* ctx,
    const v2::ReleasePlacementLeaseRequest* req,
    v2::ReleasePlacementLeaseResponse* resp) {
  RpcContext rctx{"ReleasePlacementLease", *ctx, opts_.allow_high_card_attrs};
  return lease_controller_->release_placement_lease(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::AcquireRetentionHandle(
    grpc::ServerContext* ctx,
    const v2::AcquireRetentionHandleRequest* req,
    v2::AcquireRetentionHandleResponse* resp) {
  RpcContext rctx{"AcquireRetentionHandle", *ctx, opts_.allow_high_card_attrs};
  return lease_controller_->acquire_retention_handle(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RenewRetentionHandle(
    grpc::ServerContext* ctx,
    const v2::RenewRetentionHandleRequest* req,
    v2::RenewRetentionHandleResponse* resp) {
  RpcContext rctx{"RenewRetentionHandle", *ctx, opts_.allow_high_card_attrs};
  return lease_controller_->renew_retention_handle(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ReleaseRetentionHandle(
    grpc::ServerContext* ctx,
    const v2::ReleaseRetentionHandleRequest* req,
    v2::ReleaseRetentionHandleResponse* resp) {
  RpcContext rctx{"ReleaseRetentionHandle", *ctx, opts_.allow_high_card_attrs};
  return lease_controller_->release_retention_handle(rctx, *req, *resp);
}

// Publish key mapping via Global Store.
Status StoreDaemonServiceImpl::PublishReplicaKey(
    grpc::ServerContext* ctx,
    const v2::PublishReplicaKeyRequest* req,
    v2::PublishReplicaKeyResponse* resp) {
  RpcContext rctx{"PublishReplicaKey", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req->key());

  if (req->key().empty() || !req->has_artifact_descriptor() || req->artifact_descriptor().artifact_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "key and artifact_descriptor.artifact_id are required"};
  }
  if (shutdown_signal_->is_shutting_down()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  // Use engine's configured Global Store client for upsert.
  auto up = engine_->upsert_key_mapping(req->key(), req->artifact_descriptor().artifact_id());
  if (!up.ok()) {
    // For conflicts, return OK with ok=false for idempotency.
    if (absl::IsAlreadyExists(up)) {
      resp->set_ok(false);
      resp->set_conflict_reason(std::string(up.message()));
      rctx.mark_success();
      return grpc::Status::OK;
    }
    return to_grpc_status(up);
  }
  resp->set_ok(true);
  rctx.mark_success();
  return grpc::Status::OK;
}

Status StoreDaemonServiceImpl::ResolveKeyMapping(
    grpc::ServerContext* ctx,
    const v2::ResolveKeyMappingRequest* req,
    v2::ResolveKeyMappingResponse* resp) {
  RpcContext rctx{"ResolveKeyMapping", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req->key());

  if (req->key().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "key is required"};
  }
  if (shutdown_signal_->is_shutting_down()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  auto mapping_or = engine_->resolve_key_mapping(req->key());
  if (!mapping_or.ok()) {
    return to_grpc_status(mapping_or.status());
  }
  const auto& m = *mapping_or;
  resp->set_artifact_id(m.artifact_id);
  resp->set_generation(m.generation);
  resp->set_cache_ttl_seconds(m.cache_ttl_seconds);
  rctx.mark_success();
  return grpc::Status::OK;
}

Status StoreDaemonServiceImpl::SwapKeyMapping(
    grpc::ServerContext* ctx,
    const v2::SwapKeyMappingRequest* req,
    v2::SwapKeyMappingResponse* resp) {
  RpcContext rctx{"SwapKeyMapping", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req->key());
  if (req->key().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "key is required"};
  }
  if (req->new_artifact_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "new_artifact_id is required"};
  }
  if (shutdown_signal_->is_shutting_down()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  std::optional<std::string_view> expected_artifact_id;
  if (!req->expected_artifact_id().empty()) {
    expected_artifact_id = req->expected_artifact_id();
  }
  std::optional<uint64_t> expected_generation;
  if (req->has_expected_generation()) {
    expected_generation = req->expected_generation();
  }

  auto result_or =
      engine_->swap_key_mapping(req->key(), req->new_artifact_id(), expected_artifact_id, expected_generation);
  if (!result_or.ok()) {
    return to_grpc_status(result_or.status());
  }
  const auto& result = *result_or;
  resp->set_ok(result.ok);
  resp->set_artifact_id(result.artifact_id);
  resp->set_generation(result.generation);
  rctx.mark_success();
  return grpc::Status::OK;
}

Status StoreDaemonServiceImpl::GetArtifactIndexById(
    grpc::ServerContext* ctx,
    const v2::GetArtifactIndexByIdRequest* req,
    v2::GetArtifactIndexByIdResponse* resp) {
  RpcContext rctx{"GetArtifactIndexById", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->get_artifact_index_by_id(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::SealAssembly(
    grpc::ServerContext* ctx,
    const v2::SealAssemblyRequest* req,
    v2::SealAssemblyResponse* resp) {
  RpcContext rctx{"SealAssembly", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->seal_assembly(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::StartSealAssembly(
    grpc::ServerContext* ctx,
    const v2::StartSealAssemblyRequest* req,
    v2::StartSealAssemblyResponse* resp) {
  RpcContext rctx{"StartSealAssembly", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->start_seal_assembly(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::GetOperation(
    grpc::ServerContext* ctx,
    const tensorcast::operation::v1::GetOperationRequest* req,
    tensorcast::operation::v1::GetOperationResponse* resp) {
  RpcContext rctx{"GetOperation", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->get_operation(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::WaitOperation(
    grpc::ServerContext* ctx,
    const v2::WaitOperationRequest* req,
    v2::WaitOperationResponse* resp) {
  RpcContext rctx{"WaitOperation", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->wait_operation(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::StartPersistence(
    grpc::ServerContext* ctx,
    const v2::StartPersistenceRequest* req,
    v2::StartPersistenceResponse* resp) {
  RpcContext rctx{"StartPersistence", *ctx, opts_.allow_high_card_attrs};
  if (req->artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  if (shutdown_signal_->is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!persistence_manager_) {
    return {StatusCode::FAILED_PRECONDITION, "persistence manager unavailable"};
  }
  auto policy_or = resolve_store_policy(req->has_policy() ? &req->policy() : nullptr);
  if (!policy_or.ok()) {
    return to_grpc_status(policy_or.status());
  }
  auto task_or = persistence_manager_->start_task(req->artifact_id(), *policy_or, req->key_hint());
  if (!task_or.ok()) {
    return to_grpc_status(task_or.status());
  }
  const auto& task = *task_or;

  resp->set_task_id(task.task_id);
  resp->set_plan_id(task.plan_id);
  resp->set_state(task.state);
  resp->set_progress(task.progress);
  if (!task.degraded_reason.empty()) {
    resp->set_degraded_reason(task.degraded_reason);
  }
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::QueryPersistenceStatus(
    grpc::ServerContext* ctx,
    const v2::QueryPersistenceStatusRequest* req,
    v2::QueryPersistenceStatusResponse* resp) {
  RpcContext rctx{"QueryPersistenceStatus", *ctx, opts_.allow_high_card_attrs};
  if (req->task_id().empty() && req->artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "task_id or artifact_id is required"};
  }
  std::optional<std::string> task_key;
  if (!req->task_id().empty()) {
    task_key = req->task_id();
  }
  if (!persistence_manager_) {
    return {StatusCode::FAILED_PRECONDITION, "persistence manager unavailable"};
  }
  absl::optional<PersistenceTaskState> task;
  if (task_key.has_value()) {
    task = persistence_manager_->get_by_task_id(*task_key);
  } else {
    task = persistence_manager_->get_latest_for_artifact(req->artifact_id());
  }

  if (!task.has_value()) {
    return {StatusCode::NOT_FOUND, "persistence task not found"};
  }
  resp->set_task_id(task_key.value_or(task->task_id));
  resp->set_artifact_id(task->artifact_id);
  resp->set_plan_id(task->plan_id);
  resp->set_state(task->state);
  resp->set_progress(task->progress);
  if (!task->degraded_reason.empty()) {
    resp->set_degraded_reason(task->degraded_reason);
  }
  if (!task->last_error.empty()) {
    resp->set_last_error(task->last_error);
  }
  for (const auto& shard : task->shards) {
    auto* out = resp->add_shards();
    out->set_shard_id(shard.shard_id);
    out->set_shard_idx(shard.shard_idx);
    out->set_state(shard.state);
    out->set_progress(shard.progress);
    if (!shard.degraded_reason.empty()) {
      out->set_degraded_reason(shard.degraded_reason);
    }
    if (!shard.last_error.empty()) {
      out->set_last_error(shard.last_error);
    }
    out->mutable_target_nodes()->Reserve(static_cast<int>(shard.targets.size()));
    out->mutable_lease_ids()->Reserve(static_cast<int>(shard.targets.size()));
    for (const auto& target : shard.targets) {
      out->add_target_nodes(target.node_id);
      out->add_lease_ids(target.lease_id);
    }
  }
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::UnloadReplica(
    grpc::ServerContext* ctx,
    const v2::UnloadReplicaRequest* req,
    v2::UnloadReplicaResponse* resp) {
  RpcContext rctx{"UnloadReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->unload(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::ClearMem(
    grpc::ServerContext* ctx,
    const v2::ClearMemRequest* /*req*/,
    v2::ClearMemResponse* /*resp*/) {
  RpcContext rctx{"ClearMem", *ctx, opts_.allow_high_card_attrs};
  const int rc = engine_->clear_mem();
  if (rc == 0) {
    rctx.mark_success();
    return Status::OK;
  }
  return {StatusCode::INTERNAL, absl::StrFormat("clear_mem() returned %d", rc)};
}

Status StoreDaemonServiceImpl::GetServerConfig(
    grpc::ServerContext* ctx,
    const v2::GetServerConfigRequest* /*req*/,
    v2::GetServerConfigResponse* resp) {
  RpcContext rctx{"GetServerConfig", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_server_config(rctx, *resp);
}

// ──────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────

// Legacy device/key helpers have been removed in favor of DeviceResolver

Status StoreDaemonServiceImpl::WaitReplicaVerification(
    grpc::ServerContext* ctx,
    const v2::WaitReplicaVerificationRequest* req,
    v2::WaitReplicaVerificationResponse* resp) {
  RpcContext rctx{"WaitReplicaVerification", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->wait_verification(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::LockTransportChunks(
    grpc::ServerContext* ctx,
    const v2::LockTransportChunksRequest* req,
    v2::LockTransportChunksResponse* resp) {
  RpcContext rctx{"LockTransportChunks", *ctx, opts_.allow_high_card_attrs};
  return transport_controller_->lock(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RegisterVramRegion(
    grpc::ServerContext* ctx,
    const v2::RegisterVramRegionRequest* req,
    v2::RegisterVramRegionResponse* resp) {
  RpcContext rctx{"RegisterVramRegion", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.device.id", static_cast<int64_t>(req->device_id()));
  span->SetAttribute("tc.region.size_bytes", static_cast<int64_t>(req->size_bytes()));
  span->SetAttribute("tc.region.ttl_ms", static_cast<int64_t>(req->ttl_ms()));

  if (shutdown_signal_->is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req->owner_pid() <= 0) {
    return {StatusCode::INVALID_ARGUMENT, "owner_pid must be > 0"};
  }
  if (req->device_id() < 0) {
    return {StatusCode::INVALID_ARGUMENT, "device_id must be >= 0"};
  }
  if (req->size_bytes() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "size_bytes must be > 0"};
  }
  if (req->cuda_ipc_handle().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "cuda_ipc_handle must not be empty"};
  }

  IpcRegionRegistry::RegisterParams params;
  params.device_id = req->device_id();
  params.owner_pid = req->owner_pid();
  params.size_bytes = req->size_bytes();
  params.ttl_ms = req->ttl_ms();
  if (req->has_session_id()) {
    params.session_id = req->session_id();
  }
  if (req->has_region_name()) {
    params.region_name = req->region_name();
  }
  params.handle_bytes = std::string(req->cuda_ipc_handle());

  auto desc_or = region_registry_->register_region(params);
  if (!desc_or.ok()) {
    return to_grpc_status(desc_or.status());
  }
  const auto& desc = *desc_or;
  if (desc.ttl_ms == 0 && lifecycle_manager_ != nullptr) {
    lifecycle_manager_->watch_pid(static_cast<pid_t>(desc.owner_pid));
  }
  resp->set_region_id(desc.region_id);
  resp->set_ttl_ms(desc.ttl_ms);
  if (desc.expires_at != absl::InfiniteFuture()) {
    const int64_t micros = absl::ToUnixMicros(desc.expires_at);
    auto* ts = resp->mutable_expires_at();
    ts->set_seconds(micros / 1'000'000);
    ts->set_nanos(static_cast<int32_t>((micros % 1'000'000) * 1'000));
  }
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::UnregisterVramRegion(
    grpc::ServerContext* ctx,
    const v2::UnregisterVramRegionRequest* req,
    v2::UnregisterVramRegionResponse* resp) {
  RpcContext rctx{"UnregisterVramRegion", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.region.id", req->region_id());

  if (req->region_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "region_id is required"};
  }
  if (req->owner_pid() <= 0) {
    return {StatusCode::INVALID_ARGUMENT, "owner_pid must be > 0"};
  }

  const bool force = req->has_force() ? req->force() : false;
  auto released_or = region_registry_->unregister_region(req->region_id(), req->owner_pid(), force);
  if (!released_or.ok()) {
    return to_grpc_status(released_or.status());
  }
  resp->set_released(*released_or);
  if (*released_or && lifecycle_manager_ != nullptr) {
    lifecycle_manager_->unwatch_pid(static_cast<pid_t>(req->owner_pid()));
  }
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::DeregisterArtifact(
    grpc::ServerContext* ctx,
    const v2::DeregisterArtifactRequest* req,
    v2::DeregisterArtifactResponse* resp) {
  RpcContext rctx{"DeregisterArtifact", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  if (!req->artifact_id().empty()) {
    span->SetAttribute("tc.artifact.id", req->artifact_id());
  }
  if (rctx.allow_high_card_attrs() && req->has_operation_id()) {
    span->SetAttribute("tc.operation.id", req->operation_id());
  }
  if (shutdown_signal_->is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req->artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  const std::string artifact_id = req->artifact_id();
  const bool keep_shared_disk_copy = req->has_keep_shared_disk_copy() ? req->keep_shared_disk_copy() : false;

  auto view_id_or = parse_view_id(req->byte_space());
  if (!view_id_or.ok()) {
    return to_grpc_status(view_id_or.status());
  }
  std::optional<std::string> view_id = *view_id_or;

  std::optional<LipLeaseEntry> lease;
  ArtifactDeviceKey key;
  if (req->has_device_id()) {
    key = ArtifactDeviceKey{.artifact_id = artifact_id, .view_id = view_id.value_or(""), .device_id = req->device_id()};
    lease = lip_manager_->find_active_by_key(key);
    if (!lease.has_value()) {
      return {StatusCode::NOT_FOUND, "no active lease for artifact"};
    }
  } else {
    auto leases = lip_manager_->list_active_by_artifact_id(artifact_id, view_id);
    if (leases.empty()) {
      return {StatusCode::NOT_FOUND, "no active lease for artifact"};
    }
    if (leases.size() > 1) {
      return {StatusCode::INVALID_ARGUMENT, "device_id required to disambiguate replicas"};
    }
    lease = leases.front();
    key = ArtifactDeviceKey{.artifact_id = artifact_id, .view_id = view_id.value_or(""), .device_id = lease->device_id};
  }

  if (req->has_owner_pid() && lease->owner_pid != req->owner_pid()) {
    return {StatusCode::PERMISSION_DENIED, "owner_pid mismatch for active lease"};
  }

  if (req->has_extend_ttl_ms() && req->extend_ttl_ms() > 0) {
    auto st = lip_manager_->extend_ttl_for_artifact(artifact_id, req->extend_ttl_ms(), view_id);
    if (!st.ok()) {
      return to_grpc_status(st);
    }
  }

  lip_manager_->quiesce_lease(key);

  const bool wait_for_drain = req->wait_for_drain();
  const uint32_t timeout_ms = req->has_drain_timeout_ms() ? req->drain_timeout_ms() : 30000U;
  auto drain_or = retire_replica_with_drain(
      lip_manager_,
      global_store_client_.get(),
      artifact_id,
      key,
      wait_for_drain,
      timeout_ms,
      req->has_operation_id() ? std::optional<std::string_view>(req->operation_id()) : std::nullopt);
  if (!drain_or.ok()) {
    return to_grpc_status(drain_or.status());
  }
  const auto& drain = *drain_or;

  absl::Status revoke_status = lip_manager_->revoke_by_registration_id(lease->registration_id);
  if (!revoke_status.ok()) {
    return to_grpc_status(revoke_status);
  }
  resp->set_drained(drain.drained);
  resp->set_removed(true);

  if ((!drain.replica_id.has_value() || drain.replica_id->empty()) && !view_id.has_value()) {
    absl::Status gs_st = engine_->unregister_replica_from_global_store(artifact_id, key.device_id);
    if (!gs_st.ok()) {
      resp->set_message(absl::StrCat("Global Store deregister failed: ", gs_st.message()));
    }
  }

  absl::flat_hash_set<std::string> unique_regions;
  for (const auto& s : lease->storages) {
    if (s.has_region()) {
      unique_regions.insert(s.region_id);
    }
  }
  for (const auto& rid : unique_regions) {
    resp->add_released_region_ids(rid);
  }

  if (!keep_shared_disk_copy && !view_id.has_value()) {
    purge_managed_shared_disk_artifact(global_store_client_.get(), artifact_id, opts_.storage_path, resp);
  }

  rctx.mark_success();
  return grpc::Status::OK;
}

Status StoreDaemonServiceImpl::PublishTargetReplica(
    grpc::ServerContext* ctx,
    const v2::PublishTargetReplicaRequest* req,
    v2::PublishTargetReplicaResponse* resp) {
  RpcContext rctx{"PublishTargetReplica", *ctx, opts_.allow_high_card_attrs};
  return materialization_controller_->publish_target_replica(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RetirePublishedReplica(
    grpc::ServerContext* ctx,
    const v2::RetirePublishedReplicaRequest* req,
    v2::RetirePublishedReplicaResponse* resp) {
  RpcContext rctx{"RetirePublishedReplica", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  if (!req->artifact_id().empty()) {
    span->SetAttribute("tc.artifact.id", req->artifact_id());
  }
  if (rctx.allow_high_card_attrs() && req->has_operation_id()) {
    span->SetAttribute("tc.operation.id", req->operation_id());
  }
  if (shutdown_signal_->is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req->artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }

  auto view_id_or = parse_view_id(req->byte_space());
  if (!view_id_or.ok()) {
    return to_grpc_status(view_id_or.status());
  }
  std::optional<std::string> view_id = *view_id_or;

  std::optional<LipLeaseEntry> lease;
  ArtifactDeviceKey key;
  if (req->has_lease_id() && !req->lease_id().empty()) {
    auto key_opt = lip_manager_->find_key_by_registration_id(req->lease_id());
    if (!key_opt.has_value()) {
      return {StatusCode::NOT_FOUND, "lease_id not found"};
    }
    key = *key_opt;
    lease = lip_manager_->find_active_by_key(key);
    if (!lease.has_value()) {
      return {StatusCode::NOT_FOUND, "no active lease for lease_id"};
    }
    if (key.artifact_id != req->artifact_id()) {
      return {StatusCode::INVALID_ARGUMENT, "lease_id does not match artifact_id"};
    }
    if (view_id.has_value() && key.view_id != *view_id) {
      return {StatusCode::INVALID_ARGUMENT, "lease_id does not match byte_space"};
    }
    if (req->has_device_id() && key.device_id != req->device_id()) {
      return {StatusCode::INVALID_ARGUMENT, "lease_id does not match device_id"};
    }
  } else {
    if (req->has_device_id()) {
      key = ArtifactDeviceKey{
          .artifact_id = req->artifact_id(), .view_id = view_id.value_or(""), .device_id = req->device_id()};
      lease = lip_manager_->find_active_by_key(key);
      if (!lease.has_value()) {
        return {StatusCode::NOT_FOUND, "no active lease for artifact"};
      }
    } else {
      auto leases = lip_manager_->list_active_by_artifact_id(req->artifact_id(), view_id);
      if (leases.empty()) {
        return {StatusCode::NOT_FOUND, "no active lease for artifact"};
      }
      if (leases.size() > 1) {
        return {StatusCode::INVALID_ARGUMENT, "device_id required to disambiguate replicas"};
      }
      lease = leases.front();
      key = ArtifactDeviceKey{
          .artifact_id = req->artifact_id(), .view_id = view_id.value_or(""), .device_id = lease->device_id};
    }
  }

  if (req->has_owner_pid() && lease->owner_pid != req->owner_pid()) {
    return {StatusCode::PERMISSION_DENIED, "owner_pid mismatch for active lease"};
  }

  lip_manager_->quiesce_lease(key);

  const bool wait_for_drain = req->wait_for_drain();
  const uint32_t timeout_ms = req->has_drain_timeout_ms() ? req->drain_timeout_ms() : 30000U;
  auto drain_or = retire_replica_with_drain(
      lip_manager_,
      global_store_client_.get(),
      req->artifact_id(),
      key,
      wait_for_drain,
      timeout_ms,
      req->has_operation_id() ? std::optional<std::string_view>(req->operation_id()) : std::nullopt);
  if (!drain_or.ok()) {
    return to_grpc_status(drain_or.status());
  }
  const auto& drain = *drain_or;

  absl::Status revoke_status = lip_manager_->revoke_by_registration_id(lease->registration_id);
  if (!revoke_status.ok()) {
    return to_grpc_status(revoke_status);
  }
  resp->set_drained(drain.drained);
  resp->set_removed(true);
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::UnlockTransportChunks(
    grpc::ServerContext* ctx,
    const v2::UnlockTransportChunksRequest* req,
    v2::UnlockTransportChunksResponse* /*resp*/) {
  RpcContext rctx{"UnlockTransportChunks", *ctx, opts_.allow_high_card_attrs};
  v2::UnlockTransportChunksResponse dummy;
  return transport_controller_->unlock(rctx, *req, dummy);
}

Status StoreDaemonServiceImpl::BeginRegisterArtifact(
    grpc::ServerContext* ctx,
    const v2::BeginRegisterArtifactRequest* req,
    v2::BeginRegisterArtifactResponse* resp) {
  RpcContext rctx{"BeginRegisterArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->begin(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::CommitRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v2::CommitRegisteredArtifactRequest* req,
    v2::CommitRegisteredArtifactResponse* resp) {
  RpcContext rctx{"CommitRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->commit(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::AbortRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v2::AbortRegisteredArtifactRequest* req,
    v2::AbortRegisteredArtifactResponse* resp) {
  RpcContext rctx{"AbortRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->abort(rctx, *req, *resp);
}

// Removed unary FeedRegisterArtifact; use streaming variant only

Status StoreDaemonServiceImpl::FeedRegisterArtifactStream(
    grpc::ServerContext* ctx,
    ::grpc::ServerReader<v2::FeedRegisterArtifactStreamRequest>* reader,
    v2::FeedRegisterArtifactStreamResponse* resp) {
  RpcContext rctx{"FeedRegisterArtifactStream", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->feed_stream(rctx, *reader, *resp);
}

grpc::Status StoreDaemonServiceImpl::feed_register_artifact_stream_vector(
    const std::vector<v2::FeedRegisterArtifactStreamRequest>& reqs) {
  return registration_controller_->feed_vector(reqs);
}

Status StoreDaemonServiceImpl::KeepAliveRegisterArtifact(
    grpc::ServerContext* ctx,
    const v2::KeepAliveRegisterArtifactRequest* req,
    v2::KeepAliveRegisterArtifactResponse* resp) {
  RpcContext rctx{"KeepAliveRegisterArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->keep_alive(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::RevokeRegisteredArtifact(
    grpc::ServerContext* ctx,
    const v2::RevokeRegisteredArtifactRequest* req,
    v2::RevokeRegisteredArtifactResponse* resp) {
  RpcContext rctx{"RevokeRegisteredArtifact", *ctx, opts_.allow_high_card_attrs};
  return registration_controller_->revoke(rctx, *req, *resp);
}

Status StoreDaemonServiceImpl::GetWorkerStatus(
    grpc::ServerContext* ctx,
    const v2::GetWorkerStatusRequest* /*req*/,
    v2::GetWorkerStatusResponse* resp) {
  RpcContext rctx{"GetWorkerStatus", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_worker_status(rctx, *resp);
}

Status StoreDaemonServiceImpl::GetDetailedStatus(
    grpc::ServerContext* ctx,
    const v2::GetDetailedStatusRequest* /*req*/,
    v2::GetDetailedStatusResponse* resp) {
  RpcContext rctx{"GetDetailedStatus", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_detailed_status(rctx, *resp);
}

// verification tracking moved to VerificationTracker

// Legacy GetLoadedReplicas removed; use V2

Status StoreDaemonServiceImpl::GetLoadedReplicasV2(
    grpc::ServerContext* ctx,
    const v2::GetLoadedReplicasV2Request* req,
    v2::GetLoadedReplicasV2Response* resp) {
  RpcContext rctx{"GetLoadedReplicasV2", *ctx, opts_.allow_high_card_attrs};
  return status_controller_->get_loaded_replicas_v2(rctx, *req, *resp, opts_.use_cursor_pagination);
}

} // namespace tensorcast::daemon
