// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/replica_session_controller.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string_view>

#include "absl/status/status.h"
#include "daemon/util/status_utils.h"
#include "folly/futures/Future.h"
#include "google/protobuf/util/time_util.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;

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
  const grpc::Status grpc_st = status_utils::to_grpc_status(st);
  out.set_state(v2::ReplicaOperationState::REPLICA_OPERATION_STATE_FAILED);
  out.set_message(std::string(st.message()));
  *out.mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
  auto* err = out.mutable_error();
  err->set_status_code(status_code_name(grpc_st.error_code()));
  err->set_message(std::string(st.message()));
  err->set_retryable(is_retryable(grpc_st.error_code()));
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

} // namespace

grpc::Status ReplicaSessionController::query_replica_status(
    RpcContext& rctx,
    const v2::QueryReplicaStatusRequest& req,
    v2::QueryReplicaStatusResponse& resp) {
  if (!req.has_ticket() || req.ticket().replica_uuid().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "ticket.replica_uuid is required"};
  }
  const std::string& replica_uuid = req.ticket().replica_uuid();
  auto entry = d_.sessions.get(replica_uuid);
  if (!entry.has_value()) {
    return {StatusCode::NOT_FOUND, "replica_uuid not found"};
  }

  resp.mutable_ticket()->set_replica_uuid(replica_uuid);
  resp.set_replica_key_hash(replica_key_hash_bytes(entry->key));

  auto* status = resp.mutable_status();
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

grpc::Status ReplicaSessionController::wait_replica_status(
    RpcContext& rctx,
    grpc::ServerContext& ctx,
    const v2::WaitReplicaStatusRequest& req,
    v2::WaitReplicaStatusResponse& resp) {
  if (!req.has_ticket() || req.ticket().replica_uuid().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "ticket.replica_uuid is required"};
  }
  const std::string& replica_uuid = req.ticket().replica_uuid();
  auto entry = d_.sessions.get(replica_uuid);
  if (!entry.has_value()) {
    return {StatusCode::NOT_FOUND, "replica_uuid not found"};
  }

  resp.mutable_ticket()->set_replica_uuid(replica_uuid);
  resp.set_replica_key_hash(replica_key_hash_bytes(entry->key));

  auto* status = resp.mutable_status();
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
  const std::chrono::milliseconds user_timeout(static_cast<int64_t>(req.timeout_ms()));
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
    const auto grpc_deadline = ctx.deadline();
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

    if (ctx.IsCancelled()) {
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

grpc::Status ReplicaSessionController::release_replica(
    RpcContext& rctx,
    const v2::ReleaseReplicaRequest& req,
    v2::ReleaseReplicaResponse& resp) {
  if (!req.has_ticket() || req.ticket().replica_uuid().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "ticket.replica_uuid is required"};
  }
  const std::string& replica_uuid = req.ticket().replica_uuid();
  const bool erased = d_.sessions.erase(replica_uuid);
  d_.lifecycle.release_session(replica_uuid);
  resp.set_released(erased);
  rctx.mark_success();
  return Status::OK;
}

} // namespace tensorcast::daemon
