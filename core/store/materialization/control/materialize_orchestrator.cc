// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/control/materialize_orchestrator.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "core/store/components/global_store_client.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "grpcpp/grpcpp.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/provider.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::store::materialization::control {

using tensorcast::store::P2PSource;
using tensorcast::store::components::RemoteReplicaInfo;
using tensorcast::store::components::WorkerIdentity;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::loading::MaterializeHints;
using tensorcast::store::loading::ReplicaHandle;
using tensorcast::store::loading::ReplicaTarget;

namespace {

constexpr uint32_t kDefaultTransportWaitTimeoutMs = 30000;
constexpr uint32_t kViewTransportProbeTimeoutMs = 1000;
constexpr std::chrono::milliseconds kDerivedViewRouteSettleTimeout{10000};
constexpr std::chrono::milliseconds kDerivedViewRouteSettleAfterRetryableWaitError{30000};
constexpr int kMaxReselectionAttemptsWithoutBudget = 64;
constexpr std::chrono::milliseconds kMinReselectionBudget{1};
constexpr std::chrono::milliseconds kMinReselectionBackoff{50};

struct SourceReselectionMetrics {
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> meter;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> stale_source_detected_total;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> reselection_attempt_total;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> reselection_success_total;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> reselection_exhausted_total;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> route_selected_total;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> fallback_total;
};

SourceReselectionMetrics& source_reselection_metrics() {
  static SourceReselectionMetrics metrics;
  if (!metrics.meter) {
    metrics.meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  }
  return metrics;
}

std::string route_kind_to_string(tensorcast::global_store::v1::TransportRouteKind route_kind) {
  switch (route_kind) {
    case tensorcast::global_store::v1::TRANSPORT_ROUTE_KIND_CANONICAL:
      return "canonical";
    case tensorcast::global_store::v1::TRANSPORT_ROUTE_KIND_RESIDENT_VIEW:
      return "resident_view";
    case tensorcast::global_store::v1::TRANSPORT_ROUTE_KIND_DERIVED_VIEW_FROM_CANONICAL:
      return "derived_view_from_canonical";
    case tensorcast::global_store::v1::TRANSPORT_ROUTE_KIND_UNSPECIFIED:
      return "unspecified";
    default:
      break;
  }
  return absl::StrCat("unknown(", static_cast<int>(route_kind), ")");
}

void record_route_selected(
    tensorcast::global_store::v1::TransportRouteKind route_kind,
    bool view_scoped,
    bool canonical_fallback) {
  try {
    auto& metrics = source_reselection_metrics();
    if (!metrics.route_selected_total) {
      metrics.route_selected_total = metrics.meter->CreateDoubleCounter("tc_view_transport_route_selected_total");
    }
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("route_kind", route_kind_to_string(route_kind));
    attrs.emplace("scope", std::string(view_scoped ? "view" : "canonical"));
    attrs.emplace("canonical_fallback", canonical_fallback ? "true" : "false");
    metrics.route_selected_total->Add(
        1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void record_view_transport_fallback(std::string_view reason, std::string_view stage, bool view_scoped) {
  try {
    auto& metrics = source_reselection_metrics();
    if (!metrics.fallback_total) {
      metrics.fallback_total = metrics.meter->CreateDoubleCounter("tc_view_transport_fallback_total");
    }
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("reason", std::string(reason));
    attrs.emplace("stage", std::string(stage));
    attrs.emplace("scope", std::string(view_scoped ? "view" : "canonical"));
    metrics.fallback_total->Add(
        1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void add_source_reselection_counter(
    const opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>>& counter,
    std::string_view reason,
    bool view_scoped,
    int attempt) {
  if (!counter) {
    return;
  }
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("reason", std::string(reason));
  attrs.emplace("scope", std::string(view_scoped ? "view" : "canonical"));
  attrs.emplace("attempt", std::string(std::to_string(std::max(0, attempt))));
  counter->Add(1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
}

void record_stale_source_detected(std::string_view reason, bool view_scoped, int attempt) {
  try {
    auto& metrics = source_reselection_metrics();
    if (!metrics.stale_source_detected_total) {
      metrics.stale_source_detected_total = metrics.meter->CreateDoubleCounter("tc_p2p_stale_source_detected_total");
    }
    add_source_reselection_counter(metrics.stale_source_detected_total, reason, view_scoped, attempt);
  } catch (...) {
  }
}

void record_source_reselection_attempt(std::string_view reason, bool view_scoped, int attempt) {
  try {
    auto& metrics = source_reselection_metrics();
    if (!metrics.reselection_attempt_total) {
      metrics.reselection_attempt_total = metrics.meter->CreateDoubleCounter("tc_p2p_source_reselection_attempt_total");
    }
    add_source_reselection_counter(metrics.reselection_attempt_total, reason, view_scoped, attempt);
  } catch (...) {
  }
}

void record_source_reselection_success(std::string_view reason, bool view_scoped, int attempt) {
  try {
    auto& metrics = source_reselection_metrics();
    if (!metrics.reselection_success_total) {
      metrics.reselection_success_total = metrics.meter->CreateDoubleCounter("tc_p2p_source_reselection_success_total");
    }
    add_source_reselection_counter(metrics.reselection_success_total, reason, view_scoped, attempt);
  } catch (...) {
  }
}

void record_source_reselection_exhausted(std::string_view reason, bool view_scoped, int attempt) {
  try {
    auto& metrics = source_reselection_metrics();
    if (!metrics.reselection_exhausted_total) {
      metrics.reselection_exhausted_total =
          metrics.meter->CreateDoubleCounter("tc_p2p_source_reselection_exhausted_total");
    }
    add_source_reselection_counter(metrics.reselection_exhausted_total, reason, view_scoped, attempt);
  } catch (...) {
  }
}

bool is_local_identity(const WorkerIdentity& local) {
  return !local.node_id.empty() || !local.node_address.empty();
}

bool is_local_replica(const RemoteReplicaInfo& remote, const WorkerIdentity& local) {
  if (!is_local_identity(local)) {
    return false;
  }

  const bool same_node_id = !local.node_id.empty() && !remote.node_id.empty() && local.node_id == remote.node_id;
  const bool same_node_address =
      !local.node_address.empty() && !remote.node_address.empty() && local.node_address == remote.node_address;
  const bool has_local_p2p_port = local.p2p_port != 0;
  const bool has_remote_p2p_port = remote.node_port != 0;
  if (has_local_p2p_port && has_remote_p2p_port) {
    if (local.p2p_port != remote.node_port) {
      return false;
    }
    return same_node_id || same_node_address;
  }

  if (same_node_address) {
    return true;
  }
  if (same_node_id) {
    return true;
  }
  return false;
}

absl::Status stale_local_route_status(std::string_view artifact_id) {
  return absl::UnavailableError(
      absl::StrCat("Global Store route stale for artifact_id=", artifact_id, "; retry or provide disk source"));
}

bool should_retry_source_selection(const absl::Status& status) {
  return absl::IsUnavailable(status) || absl::IsNotFound(status) || absl::IsFailedPrecondition(status) ||
      absl::IsDeadlineExceeded(status);
}

bool is_retryable_grpc_status(const grpc::Status& status) {
  return status.error_code() == grpc::StatusCode::UNAVAILABLE ||
      status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED;
}

std::optional<std::chrono::steady_clock::time_point> resolve_request_deadline(const MaterializeHints& hints) {
  if (hints.request_budget.count() <= 0) {
    return std::nullopt;
  }
  return std::chrono::steady_clock::now() + hints.request_budget;
}

std::chrono::milliseconds remaining_request_budget(
    const std::optional<std::chrono::steady_clock::time_point>& deadline) {
  if (!deadline.has_value()) {
    return std::chrono::milliseconds::max();
  }
  const auto now = std::chrono::steady_clock::now();
  if (now >= *deadline) {
    return std::chrono::milliseconds(0);
  }
  return std::chrono::duration_cast<std::chrono::milliseconds>(*deadline - now);
}

uint32_t clamp_timeout_to_u32_ms(std::chrono::milliseconds timeout) {
  if (timeout.count() <= 0) {
    return 0;
  }
  if (timeout.count() > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return std::numeric_limits<uint32_t>::max();
  }
  return static_cast<uint32_t>(timeout.count());
}

uint32_t effective_transport_wait_timeout_ms(
    const MaterializeHints& hints,
    const std::optional<std::chrono::steady_clock::time_point>& deadline) {
  const std::chrono::milliseconds configured = hints.transport_wait_timeout.count() > 0
      ? hints.transport_wait_timeout
      : std::chrono::milliseconds(kDefaultTransportWaitTimeoutMs);
  std::chrono::milliseconds timeout = configured;
  const std::chrono::milliseconds remaining = remaining_request_budget(deadline);
  if (remaining != std::chrono::milliseconds::max()) {
    timeout = std::min(timeout, remaining);
  }
  if (timeout.count() <= 0) {
    return 0;
  }
  return std::max<uint32_t>(1, clamp_timeout_to_u32_ms(timeout));
}

std::string budget_ms_label(std::chrono::milliseconds timeout) {
  if (timeout == std::chrono::milliseconds::max()) {
    return "unbounded";
  }
  return std::to_string(timeout.count());
}

bool is_source_side_upgrade_terminal_timeout(const absl::Status& status) {
  if (absl::IsDeadlineExceeded(status)) {
    return true;
  }
  return (absl::IsFailedPrecondition(status) || absl::IsUnavailable(status)) &&
      absl::StrContains(status.message(), "budget exhausted");
}

uint32_t view_transport_probe_timeout_ms(uint32_t transport_wait_timeout_ms) {
  if (transport_wait_timeout_ms == 0) {
    return 0;
  }
  return std::min(transport_wait_timeout_ms, kViewTransportProbeTimeoutMs);
}

bool can_retry_source_selection(
    const absl::Status& status,
    int reselection_attempt,
    const std::optional<std::chrono::steady_clock::time_point>& deadline,
    int max_reselection_attempts) {
  if (!should_retry_source_selection(status)) {
    return false;
  }
  if (reselection_attempt >= max_reselection_attempts) {
    return false;
  }
  const std::chrono::milliseconds remaining = remaining_request_budget(deadline);
  if (remaining != std::chrono::milliseconds::max()) {
    return remaining >= kMinReselectionBudget;
  }
  return true;
}

int resolve_max_reselection_attempts(const MaterializeHints& hints) {
  if (hints.request_budget.count() <= 0) {
    return kMaxReselectionAttemptsWithoutBudget;
  }
  const int64_t computed = std::max<int64_t>(1, hints.request_budget.count() / kMinReselectionBackoff.count());
  if (computed > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(computed);
}

std::optional<components::TransportSchedulingGroupHint> to_transport_scheduling_group_hint(
    const MaterializeHints& hints) {
  if (!hints.transport_scheduling_group.has_value()) {
    return std::nullopt;
  }
  const auto& group = *hints.transport_scheduling_group;
  if (group.group_id.empty() || group.group_kind.empty() || group.part_id.empty()) {
    return std::nullopt;
  }
  components::TransportSchedulingGroupHint out;
  out.group_id = group.group_id;
  out.group_kind = group.group_kind;
  out.total_parts = group.total_parts;
  out.part_id = group.part_id;
  out.priority = group.priority;
  out.epoch = group.epoch;
  return out;
}

bool should_log_reselection_attempt(int reselection_attempt) {
  return reselection_attempt <= 5 || (reselection_attempt % 10) == 0;
}

absl::Status source_reselection_exhausted_status(
    std::string_view artifact_id,
    int reselection_attempt,
    const std::optional<std::chrono::steady_clock::time_point>& deadline,
    const absl::Status& terminal_status) {
  const std::chrono::milliseconds remaining = remaining_request_budget(deadline);
  if (remaining != std::chrono::milliseconds::max() && remaining < kMinReselectionBudget) {
    return absl::DeadlineExceededError(
        absl::StrCat(
            "source reselection deadline exhausted for artifact_id=", artifact_id, " attempts=", reselection_attempt));
  }
  if (absl::IsNotFound(terminal_status)) {
    return absl::NotFoundError(
        absl::StrCat(
            "source reselection budget exhausted for artifact_id=", artifact_id, " attempts=", reselection_attempt));
  }
  return absl::UnavailableError(
      absl::StrCat(
          "source reselection budget exhausted for artifact_id=", artifact_id, " attempts=", reselection_attempt));
}

struct PreparedRemoteReplicaCleanup {
  std::string daemon_address;
  std::string replica_uuid;
};

struct RemoteReplicaFetchCleanup {
  std::string daemon_address;
  std::string transport_id;
};

absl::Status release_prepared_remote_replica(const PreparedRemoteReplicaCleanup& cleanup) {
  auto channel = grpc::CreateChannel(cleanup.daemon_address, grpc::InsecureChannelCredentials());
  auto stub = tensorcast::daemon::v2::StoreDaemonService::NewStub(channel);
  grpc::ClientContext release_ctx;
  release_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
  tensorcast::daemon::v2::ReleaseReplicaRequest release_req;
  release_req.mutable_ticket()->set_replica_uuid(cleanup.replica_uuid);
  tensorcast::daemon::v2::ReleaseReplicaResponse release_resp;
  const grpc::Status release_status = stub->ReleaseReplica(&release_ctx, release_req, &release_resp);
  if (!release_status.ok()) {
    return absl::UnavailableError(
        absl::StrCat(
            "source daemon ReleaseReplica failed for derived view export: address=",
            cleanup.daemon_address,
            " replica_uuid=",
            cleanup.replica_uuid,
            " status=",
            release_status.error_message()));
  }
  return absl::OkStatus();
}

tensorcast::daemon::v2::DeviceType to_daemon_device_type(common::memory::MemoryLocation location) {
  switch (location) {
    case common::memory::MemoryLocation::CPU:
      return tensorcast::daemon::v2::DEVICE_TYPE_CPU;
    case common::memory::MemoryLocation::GPU:
      return tensorcast::daemon::v2::DEVICE_TYPE_GPU;
    default:
      return tensorcast::daemon::v2::DEVICE_TYPE_UNSPECIFIED;
  }
}

absl::StatusOr<bool> begin_remote_replica_fetch(
    const components::TransportSession& session,
    std::string_view artifact_id,
    std::string_view view_id) {
  if (session.remote_replica.grpc_port == 0 || session.remote_replica.node_address.empty()) {
    return absl::UnimplementedError("source daemon fetch lifecycle requires routable daemon gRPC address");
  }
  auto channel = grpc::CreateChannel(
      absl::StrCat(session.remote_replica.node_address, ":", session.remote_replica.grpc_port),
      grpc::InsecureChannelCredentials());
  auto stub = tensorcast::daemon::v2::StoreDaemonService::NewStub(channel);
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
  tensorcast::daemon::v2::BeginReplicaFetchRequest req;
  req.set_transport_id(session.transport_id);
  req.set_artifact_id(std::string(artifact_id));
  req.set_view_id(std::string(view_id));
  req.set_device_type(to_daemon_device_type(session.remote_replica.memory_type));
  if (session.remote_replica.memory_type == common::memory::MemoryLocation::GPU) {
    req.set_device_id(session.remote_replica.device_id);
  }
  tensorcast::daemon::v2::BeginReplicaFetchResponse resp;
  const grpc::Status status = stub->BeginReplicaFetch(&ctx, req, &resp);
  if (status.error_code() == grpc::StatusCode::UNIMPLEMENTED) {
    return false;
  }
  if (!status.ok()) {
    return absl::UnavailableError(
        absl::StrCat(
            "source daemon BeginReplicaFetch failed: transport_id=",
            session.transport_id,
            " status=",
            status.error_message()));
  }
  return resp.managed();
}

absl::Status end_remote_replica_fetch(const RemoteReplicaFetchCleanup& cleanup, std::string_view reason) {
  auto channel = grpc::CreateChannel(cleanup.daemon_address, grpc::InsecureChannelCredentials());
  auto stub = tensorcast::daemon::v2::StoreDaemonService::NewStub(channel);
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
  tensorcast::daemon::v2::EndReplicaFetchRequest req;
  req.set_transport_id(cleanup.transport_id);
  req.set_reason(std::string(reason));
  tensorcast::daemon::v2::EndReplicaFetchResponse resp;
  const grpc::Status status = stub->EndReplicaFetch(&ctx, req, &resp);
  if (status.error_code() == grpc::StatusCode::UNIMPLEMENTED) {
    return absl::OkStatus();
  }
  if (!status.ok()) {
    return absl::UnavailableError(
        absl::StrCat(
            "source daemon EndReplicaFetch failed: transport_id=",
            cleanup.transport_id,
            " status=",
            status.error_message()));
  }
  return absl::OkStatus();
}

} // namespace

MaterializeOrchestrator::MaterializeOrchestrator(
    gsl::not_null<MaterializationBackend*> backend,
    gsl::not_null<components::IGlobalStoreClient*> gs_client,
    WorkerIdentity local_identity)
    : backend_(backend), gs_client_(gs_client), local_identity_(std::move(local_identity)) {}

absl::StatusOr<ReplicaHandle> MaterializeOrchestrator::run(
    std::string_view artifact_id,
    const DeviceKey& target_device,
    const MaterializeHints& hints,
    const std::optional<loading::DiskSource>& disk_source) {
  // ------------------------------------------------------------------
  // 1. Preference handling and Global Store connectivity guard
  // ------------------------------------------------------------------
  const bool gs_connected = gs_client_->is_connected();
  const auto preference = hints.source_preference;
  const bool has_disk_source = disk_source.has_value();
  const bool has_artifact_id_hint = !hints.artifact_id.empty();
  const bool allow_p2p = hints.allow_p2p;
  const bool allow_disk = hints.allow_disk;
  if (!allow_p2p && !allow_disk) {
    return absl::FailedPreconditionError("source_policy disallows both P2P and disk materialization");
  }
  if (preference == loading::SourcePreference::kPreferP2P && !has_artifact_id_hint) {
    return absl::InvalidArgumentError("preference=PREFER_P2P requires a canonical artifact_id");
  }
  if (preference == loading::SourcePreference::kPreferP2P && !allow_p2p) {
    return absl::InvalidArgumentError("source_policy disallows P2P but preference=PREFER_P2P was requested");
  }
  if (preference == loading::SourcePreference::kPreferDisk && !allow_disk) {
    return absl::InvalidArgumentError("source_policy disallows disk but preference=PREFER_DISK was requested");
  }
  if (!gs_connected && (!has_disk_source || !allow_disk)) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  auto register_with_global_store = [&](const loading::ReplicaHandle& handle) {
    absl::Status reg_status = backend_->register_replica_with_global_store(handle.key(), {});
    if (!reg_status.ok()) {
      LOG(WARNING) << "register_replica_with_global_store returned error: " << reg_status;
    }
  };

  bool attempted_disk_first = false;
  absl::Status last_p2p_status = absl::OkStatus();

  const std::optional<std::string> view_id = hints.variant ? hints.variant->view_id : std::nullopt;

  // ------------------------------------------------------------------
  // 2. Disk-first path when requested
  // ------------------------------------------------------------------
  if (preference == loading::SourcePreference::kPreferDisk && has_disk_source && allow_disk) {
    loading::DiskSource disk_src = *disk_source;

    loading::ReplicaTarget target;
    target.location.type = (target_device.type == DeviceType::GPU) ? common::memory::MemoryLocation::GPU
                                                                   : common::memory::MemoryLocation::CPU;
    target.location.device_id = target_device.ordinal;

    auto disk_or = backend_->ingest_from_disk(std::string(artifact_id), disk_src, target, hints);
    attempted_disk_first = true;
    if (disk_or.ok()) {
      register_with_global_store(*disk_or);
      return disk_or;
    }
    // If caller only provided disk hints, surface the disk failure directly.
    if (!has_artifact_id_hint || !gs_connected || !allow_p2p) {
      return disk_or.status();
    }
    LOG(INFO) << "Disk-preferred load failed, retrying via P2P: " << disk_or.status();
  }

  // ------------------------------------------------------------------
  // 3. Request transport session – Global Store will choose a suitable source
  // ------------------------------------------------------------------
  absl::StatusOr<components::TransportSession> transport_or = allow_p2p
      ? absl::FailedPreconditionError("GlobalStoreClient not connected")
      : absl::FailedPreconditionError("source_policy disallows P2P");
  bool used_canonical_transport_fallback = false;
  absl::Status view_transport_status;
  int reselection_attempt = 0;
  bool had_transport_request = false;
  const auto request_deadline = resolve_request_deadline(hints);
  const int max_reselection_attempts = resolve_max_reselection_attempts(hints);
  const auto scheduling_group_hint = to_transport_scheduling_group_hint(hints);
  const std::string_view requester_worker_id = hints.transport_requester_worker_id.empty()
      ? std::string_view(local_identity_.worker_id)
      : std::string_view(hints.transport_requester_worker_id);
  const std::string_view transport_request_id = hints.transport_request_id;

  auto request_transport = [&]() -> absl::StatusOr<components::TransportSession> {
    const uint32_t wait_timeout_ms = effective_transport_wait_timeout_ms(hints, request_deadline);
    if (wait_timeout_ms == 0) {
      return absl::DeadlineExceededError(absl::StrCat("transport wait budget exhausted for artifact_id=", artifact_id));
    }
    const uint32_t view_probe_timeout_ms = view_transport_probe_timeout_ms(wait_timeout_ms);
    used_canonical_transport_fallback = false;
    view_transport_status = absl::OkStatus();
    if (view_id.has_value()) {
      auto view_transport_or = gs_client_->request_view_transport(
          artifact_id,
          *view_id,
          local_identity_.node_id,
          local_identity_.node_address,
          local_identity_.p2p_port,
          target_device,
          view_probe_timeout_ms,
          scheduling_group_hint,
          requester_worker_id,
          transport_request_id);
      if (!view_transport_or.ok() &&
          (absl::IsNotFound(view_transport_or.status()) || absl::IsUnimplemented(view_transport_or.status()) ||
           absl::IsDeadlineExceeded(view_transport_or.status()))) {
        view_transport_status = view_transport_or.status();
        record_view_transport_fallback("view_transport_probe_unavailable", "request_transport", /*view_scoped=*/true);
        LOG(INFO) << "route_kind=canonical_fallback"
                  << " fallback_reason=view_transport_probe_unavailable"
                  << " artifact_id=" << artifact_id << " view_id=" << *view_id
                  << " probe_status=" << view_transport_or.status() << " probe_timeout_ms=" << view_probe_timeout_ms
                  << " canonical_wait_timeout_ms=" << wait_timeout_ms;
        auto canonical_transport_or = gs_client_->request_replica_transport(
            artifact_id,
            local_identity_.node_id,
            local_identity_.node_address,
            local_identity_.p2p_port,
            target_device,
            wait_timeout_ms,
            scheduling_group_hint,
            requester_worker_id,
            transport_request_id);
        used_canonical_transport_fallback = true;
        return canonical_transport_or;
      }
      return view_transport_or;
    }
    return gs_client_->request_replica_transport(
        artifact_id,
        local_identity_.node_id,
        local_identity_.node_address,
        local_identity_.p2p_port,
        target_device,
        wait_timeout_ms,
        scheduling_group_hint,
        requester_worker_id,
        transport_request_id);
  };

  struct UpgradedDerivedViewTransport {
    components::TransportSession session;
    PreparedRemoteReplicaCleanup cleanup;
  };

  auto try_upgrade_derived_view_transport = [&](const components::TransportSession& source_session)
      -> absl::StatusOr<std::optional<UpgradedDerivedViewTransport>> {
    if (!view_id.has_value()) {
      return std::optional<UpgradedDerivedViewTransport>();
    }
    if (source_session.route_kind == tensorcast::global_store::v1::TRANSPORT_ROUTE_KIND_RESIDENT_VIEW) {
      return std::optional<UpgradedDerivedViewTransport>();
    }
    if (is_local_replica(source_session.remote_replica, local_identity_)) {
      return std::optional<UpgradedDerivedViewTransport>();
    }
    if (source_session.remote_replica.grpc_port == 0 || source_session.remote_replica.node_address.empty()) {
      return absl::UnavailableError("source-side derived view transport requires a routable source daemon gRPC port");
    }

    auto fill_proto_view_spec = [&](tensorcast::common::v1::ViewSpec* out) {
      out->clear_tensors();
      if (!hints.variant.has_value() || !hints.variant->view_spec.has_value()) {
        return;
      }
      for (const auto& [tensor_name, tensor_ops] : hints.variant->view_spec->tensors) {
        auto& proto_tensor_ops = (*out->mutable_tensors())[tensor_name];
        for (const auto& op : tensor_ops.ops) {
          auto* proto_op = proto_tensor_ops.add_ops();
          switch (op.kind) {
            case store::materialization::view::ViewOp::Kind::kNarrow: {
              auto* narrow = proto_op->mutable_narrow();
              narrow->set_dim(op.narrow.dim);
              narrow->set_start(op.narrow.start);
              narrow->set_length(op.narrow.length);
              break;
            }
            case store::materialization::view::ViewOp::Kind::kTranspose: {
              auto* transpose = proto_op->mutable_transpose();
              transpose->set_dim0(op.transpose.dim0);
              transpose->set_dim1(op.transpose.dim1);
              break;
            }
          }
        }
      }
    };

    const uint32_t wait_timeout_ms = effective_transport_wait_timeout_ms(hints, request_deadline);
    if (wait_timeout_ms == 0) {
      return absl::DeadlineExceededError(
          absl::StrCat("transport wait budget exhausted before source-side derive: artifact_id=", artifact_id));
    }
    const std::string daemon_address =
        absl::StrCat(source_session.remote_replica.node_address, ":", source_session.remote_replica.grpc_port);
    const std::chrono::milliseconds remaining_before_upgrade = remaining_request_budget(request_deadline);
    LOG(INFO) << "event=source_side_upgrade_budget"
              << " artifact_id=" << artifact_id << " view_id=" << *view_id << " route_kind=derived_view_from_canonical"
              << " request_budget_ms=" << hints.request_budget.count()
              << " transport_wait_timeout_ms=" << hints.transport_wait_timeout.count()
              << " remaining_request_budget_ms=" << budget_ms_label(remaining_before_upgrade)
              << " source_prepare_wait_budget_ms=" << wait_timeout_ms << " source_daemon=" << daemon_address;
    const std::string prepare_replica_uuid = absl::StrCat("derived-view-export:", source_session.transport_id);
    tensorcast::daemon::v2::MaterializeReplicaRequest prepare_req;
    prepare_req.mutable_selection()->set_artifact_id(std::string(artifact_id));
    prepare_req.mutable_selection()->set_view_id(*view_id);
    fill_proto_view_spec(prepare_req.mutable_selection()->mutable_view_spec());
    prepare_req.set_replica_uuid(prepare_replica_uuid);
    prepare_req.set_target_device_type(tensorcast::daemon::v2::DEVICE_TYPE_CPU);
    prepare_req.set_size_bytes(
        source_session.view_transport_metadata.has_value() ? source_session.view_transport_metadata->view_size_bytes
                                                           : source_session.remote_replica.memory_size);
    prepare_req.mutable_source_policy()->set_preference(tensorcast::daemon::v2::SOURCE_PREFERENCE_AUTO);
    prepare_req.mutable_source_policy()->set_allow_p2p(true);
    prepare_req.mutable_source_policy()->set_allow_disk(false);
    prepare_req.set_lease_mode(tensorcast::daemon::v2::LEASE_MODE_NO_LEASE);
    prepare_req.set_export_policy(tensorcast::daemon::v2::EXPORT_POLICY_FORCE);
    prepare_req.set_wait_for_completion(false);
    prepare_req.set_need_view_data_hash(hints.need_view_data_hash);

    tensorcast::daemon::v2::MaterializeReplicaResponse prepare_resp;
    auto call_prepare_materialize = [&]() -> grpc::Status {
      auto channel = grpc::CreateChannel(daemon_address, grpc::InsecureChannelCredentials());
      auto stub = tensorcast::daemon::v2::StoreDaemonService::NewStub(channel);
      grpc::ClientContext prepare_ctx;
      prepare_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(wait_timeout_ms));
      prepare_resp.Clear();
      return stub->MaterializeReplica(&prepare_ctx, prepare_req, &prepare_resp);
    };

    grpc::Status prepare_status;
    for (int attempt = 1; attempt <= 2; ++attempt) {
      prepare_status = call_prepare_materialize();
      if (prepare_status.ok()) {
        break;
      }
      if (attempt == 2 || !is_retryable_grpc_status(prepare_status)) {
        return absl::UnavailableError(
            absl::StrCat(
                "source daemon MaterializeReplica failed for derived view export: address=",
                daemon_address,
                " status=",
                prepare_status.error_message()));
      }
      LOG(INFO) << "Retrying source daemon MaterializeReplica for derived view export after retryable gRPC error: "
                << "address=" << daemon_address << " attempt=" << attempt
                << " status=" << prepare_status.error_message();
    }
    if (prepare_resp.status() != tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "source daemon MaterializeReplica did not allocate derived view export: address=",
              daemon_address,
              " status=",
              static_cast<int>(prepare_resp.status())));
    }
    if (!prepare_resp.has_ticket() || prepare_resp.ticket().replica_uuid().empty()) {
      return absl::FailedPreconditionError("source daemon derived view export did not return a ticket");
    }
    PreparedRemoteReplicaCleanup cleanup{
        .daemon_address = daemon_address,
        .replica_uuid = prepare_resp.ticket().replica_uuid(),
    };
    auto cleanup_guard = absl::MakeCleanup([&]() {
      const absl::Status release_status = release_prepared_remote_replica(cleanup);
      if (!release_status.ok()) {
        LOG(WARNING) << "best-effort release of derived view export failed during upgrade: " << release_status;
      }
    });

    tensorcast::daemon::v2::WaitReplicaStatusRequest wait_req;
    wait_req.mutable_ticket()->set_replica_uuid(prepare_resp.ticket().replica_uuid());
    wait_req.set_timeout_ms(wait_timeout_ms);
    tensorcast::daemon::v2::WaitReplicaStatusResponse wait_resp;
    auto call_wait_replica_status = [&]() -> grpc::Status {
      auto channel = grpc::CreateChannel(daemon_address, grpc::InsecureChannelCredentials());
      auto stub = tensorcast::daemon::v2::StoreDaemonService::NewStub(channel);
      grpc::ClientContext wait_ctx;
      wait_ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(wait_timeout_ms));
      wait_resp.Clear();
      return stub->WaitReplicaStatus(&wait_ctx, wait_req, &wait_resp);
    };

    grpc::Status wait_status;
    bool wait_status_retryable_unknown = false;
    for (int attempt = 1; attempt <= 2; ++attempt) {
      wait_status = call_wait_replica_status();
      if (wait_status.ok()) {
        break;
      }
      if (attempt == 2 || !is_retryable_grpc_status(wait_status)) {
        if (is_retryable_grpc_status(wait_status)) {
          wait_status_retryable_unknown = true;
          LOG(INFO) << "source daemon WaitReplicaStatus remained retryable for derived view export; "
                    << "probing resident view route directly: address=" << daemon_address
                    << " status=" << wait_status.error_message();
          break;
        }
        return absl::UnavailableError(
            absl::StrCat(
                "source daemon WaitReplicaStatus failed for derived view export: address=",
                daemon_address,
                " status=",
                wait_status.error_message()));
      }
      LOG(INFO) << "Retrying source daemon WaitReplicaStatus for derived view export after retryable gRPC error: "
                << "address=" << daemon_address << " attempt=" << attempt << " status=" << wait_status.error_message();
    }
    if (!wait_status_retryable_unknown &&
        wait_resp.status().state() != tensorcast::daemon::v2::REPLICA_OPERATION_STATE_SUCCESS) {
      if (wait_resp.status().state() == tensorcast::daemon::v2::REPLICA_OPERATION_STATE_DEGRADED &&
          absl::StrContains(wait_resp.status().message(), "budget exhausted")) {
        return absl::DeadlineExceededError(
            absl::StrCat(
                "source daemon derived view export wait budget exhausted: address=",
                daemon_address,
                " message=",
                wait_resp.status().message()));
      }
      return absl::FailedPreconditionError(
          absl::StrCat(
              "source daemon derived view export did not become ready: state=",
              static_cast<int>(wait_resp.status().state()),
              " message=",
              wait_resp.status().message()));
    }

    const std::chrono::milliseconds remaining = remaining_request_budget(request_deadline);
    std::chrono::milliseconds settle_budget;
    if (remaining == std::chrono::milliseconds::max()) {
      settle_budget = wait_status_retryable_unknown ? kDerivedViewRouteSettleAfterRetryableWaitError
                                                    : kDerivedViewRouteSettleTimeout;
    } else {
      settle_budget = std::max(kMinReselectionBudget, remaining);
    }
    LOG(INFO) << "event=source_side_upgrade_route_settle_budget"
              << " artifact_id=" << artifact_id << " view_id=" << *view_id << " route_kind=derived_view_from_canonical"
              << " remaining_request_budget_ms=" << budget_ms_label(remaining)
              << " settle_budget_ms=" << budget_ms_label(settle_budget)
              << " wait_status_retryable_unknown=" << (wait_status_retryable_unknown ? "true" : "false");
    const auto settle_deadline = std::chrono::steady_clock::now() + settle_budget;
    int resident_view_attempt = 0;
    int last_route_kind = 0;
    while (true) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= settle_deadline) {
        return absl::DeadlineExceededError(
            absl::StrCat(
                "source-side derived view export completed but resident view route did not appear within "
                "settle_timeout_ms=",
                settle_budget.count(),
                " last_route_kind=",
                last_route_kind));
      }
      const auto poll_remaining = std::chrono::duration_cast<std::chrono::milliseconds>(settle_deadline - now);
      const uint32_t poll_wait_timeout_ms = std::max<uint32_t>(1, clamp_timeout_to_u32_ms(poll_remaining));
      auto upgraded_view_transport_or = gs_client_->request_view_transport(
          artifact_id,
          *view_id,
          local_identity_.node_id,
          local_identity_.node_address,
          local_identity_.p2p_port,
          target_device,
          poll_wait_timeout_ms,
          scheduling_group_hint,
          requester_worker_id,
          std::string_view());
      if (!upgraded_view_transport_or.ok()) {
        if (!should_retry_source_selection(upgraded_view_transport_or.status())) {
          return upgraded_view_transport_or.status();
        }
        ++resident_view_attempt;
        VLOG(1) << "resident view route not ready after source-side prepare: artifact_id=" << artifact_id
                << " view_id=" << *view_id << " attempt=" << resident_view_attempt
                << " status=" << upgraded_view_transport_or.status();
        std::this_thread::sleep_for(kMinReselectionBackoff);
        continue;
      }
      last_route_kind = static_cast<int>(upgraded_view_transport_or->route_kind);
      if (upgraded_view_transport_or->route_kind == tensorcast::global_store::v1::TRANSPORT_ROUTE_KIND_RESIDENT_VIEW) {
        std::move(cleanup_guard).Cancel();
        return std::optional<UpgradedDerivedViewTransport>(UpgradedDerivedViewTransport{
            .session = std::move(*upgraded_view_transport_or),
            .cleanup = std::move(cleanup),
        });
      }

      absl::Status cancel_status = gs_client_->complete_replica_transport(
          upgraded_view_transport_or->transport_id,
          components::TransportCompletionOutcome::kCancelled,
          "waiting_for_resident_view_after_source_prepare");
      if (!cancel_status.ok()) {
        LOG(WARNING) << "complete_replica_transport while waiting for resident view returned error: " << cancel_status;
      }
      ++resident_view_attempt;
      VLOG(1) << "resident view route still unresolved after source-side prepare: artifact_id=" << artifact_id
              << " view_id=" << *view_id << " attempt=" << resident_view_attempt << " route_kind=" << last_route_kind;
      std::this_thread::sleep_for(kMinReselectionBackoff);
    }
  };

  while (gs_connected && allow_p2p) {
    const std::chrono::milliseconds remaining = remaining_request_budget(request_deadline);
    if (remaining != std::chrono::milliseconds::max() && remaining < kMinReselectionBudget) {
      had_transport_request = true;
      transport_or =
          source_reselection_exhausted_status(artifact_id, reselection_attempt, request_deadline, last_p2p_status);
      break;
    }
    transport_or = request_transport();
    had_transport_request = true;
    if (!transport_or.ok()) {
      last_p2p_status = transport_or.status();
      if (can_retry_source_selection(
              last_p2p_status, reselection_attempt, request_deadline, max_reselection_attempts)) {
        reselection_attempt += 1;
        record_source_reselection_attempt("request_transport_failure", view_id.has_value(), reselection_attempt);
        if (should_log_reselection_attempt(reselection_attempt)) {
          const std::chrono::milliseconds retry_remaining = remaining_request_budget(request_deadline);
          const std::string remaining_label = retry_remaining == std::chrono::milliseconds::max()
              ? "unbounded"
              : std::to_string(retry_remaining.count());
          LOG(WARNING) << "Retrying source selection after transport request failure: artifact_id=" << artifact_id
                       << " attempt=" << reselection_attempt << "/" << max_reselection_attempts
                       << " remaining_budget_ms=" << remaining_label << " status=" << last_p2p_status;
        }
        continue;
      }
      if (should_retry_source_selection(last_p2p_status)) {
        last_p2p_status =
            source_reselection_exhausted_status(artifact_id, reselection_attempt, request_deadline, last_p2p_status);
        record_source_reselection_exhausted("request_transport_failure", view_id.has_value(), reselection_attempt);
        transport_or = last_p2p_status;
      }
      break;
    }
    components::TransportSession session = *transport_or;
    std::optional<PreparedRemoteReplicaCleanup> prepared_remote_cleanup;
    std::optional<RemoteReplicaFetchCleanup> remote_fetch_cleanup;
    std::string remote_fetch_reason = "scope_exit";
    auto prepared_cleanup_guard = absl::MakeCleanup([&]() {
      if (!prepared_remote_cleanup.has_value()) {
        return;
      }
      const absl::Status release_status = release_prepared_remote_replica(*prepared_remote_cleanup);
      if (!release_status.ok()) {
        LOG(WARNING) << "best-effort release of derived view export failed after transport session: " << release_status;
      }
    });
    auto remote_fetch_cleanup_guard = absl::MakeCleanup([&]() {
      if (!remote_fetch_cleanup.has_value()) {
        return;
      }
      const absl::Status end_status = end_remote_replica_fetch(*remote_fetch_cleanup, remote_fetch_reason);
      if (!end_status.ok()) {
        LOG(WARNING) << "best-effort EndReplicaFetch failed after transport session: " << end_status;
      }
    });

    if (view_id.has_value() &&
        session.route_kind == tensorcast::global_store::v1::TRANSPORT_ROUTE_KIND_DERIVED_VIEW_FROM_CANONICAL) {
      auto upgraded_transport_or = try_upgrade_derived_view_transport(session);
      if (!upgraded_transport_or.ok()) {
        if (is_source_side_upgrade_terminal_timeout(upgraded_transport_or.status())) {
          LOG(WARNING) << "event=source_side_upgrade_deadline_exhausted"
                       << " artifact_id=" << artifact_id << " view_id=" << *view_id << " route_kind=terminal_timeout"
                       << " terminal_reason=" << upgraded_transport_or.status();
          return upgraded_transport_or.status();
        }
        used_canonical_transport_fallback = true;
        record_view_transport_fallback("source_side_upgrade_unavailable", "upgrade_to_resident_view", true);
        LOG(INFO) << "source-side derived view transport upgrade unavailable for artifact_id=" << artifact_id
                  << " view_id=" << *view_id << " route_kind=canonical_fallback"
                  << " fallback_reason=" << upgraded_transport_or.status() << "; continuing with canonical transport";
      } else if (upgraded_transport_or->has_value()) {
        absl::Status cancel_status = gs_client_->complete_replica_transport(
            session.transport_id,
            components::TransportCompletionOutcome::kCancelled,
            "replaced_by_resident_view_transport");
        if (!cancel_status.ok()) {
          LOG(WARNING) << "complete_replica_transport for superseded canonical route returned error: " << cancel_status;
        }
        prepared_remote_cleanup = std::move((*upgraded_transport_or)->cleanup);
        session = std::move((*upgraded_transport_or)->session);
      }
    }
    const auto& remote = session.remote_replica;
    const std::string route_kind = route_kind_to_string(session.route_kind);
    record_route_selected(session.route_kind, view_id.has_value(), used_canonical_transport_fallback);
    LOG(INFO) << "event=route_selected"
              << " artifact_id=" << artifact_id << " view_id=" << (view_id.has_value() ? *view_id : std::string())
              << " target_device=" << target_device.ordinal << " route_kind=" << route_kind
              << " transport_id=" << session.transport_id << " remote_node=" << remote.node_address
              << " remote_port=" << remote.node_port << " remote_grpc_port=" << remote.grpc_port
              << " canonical_fallback=" << (used_canonical_transport_fallback ? "true" : "false");

    if (is_local_replica(remote, local_identity_)) {
      record_stale_source_detected("local_route", view_id.has_value(), reselection_attempt);
      LOG(WARNING) << "Global Store returned local replica for artifact_id=" << artifact_id
                   << "; treating route as stale";
      absl::Status comp_status = gs_client_->complete_replica_transport(
          session.transport_id, components::TransportCompletionOutcome::kFailed, "stale_local_route");
      if (!comp_status.ok()) {
        LOG(WARNING) << "complete_replica_transport after stale-local route returned error: " << comp_status;
      }
      last_p2p_status = stale_local_route_status(artifact_id);
      if (can_retry_source_selection(
              last_p2p_status, reselection_attempt, request_deadline, max_reselection_attempts)) {
        reselection_attempt += 1;
        record_source_reselection_attempt("local_route", view_id.has_value(), reselection_attempt);
        if (should_log_reselection_attempt(reselection_attempt)) {
          const std::chrono::milliseconds retry_remaining = remaining_request_budget(request_deadline);
          const std::string remaining_label = retry_remaining == std::chrono::milliseconds::max()
              ? "unbounded"
              : std::to_string(retry_remaining.count());
          LOG(WARNING) << "Retrying source selection after stale local route: artifact_id=" << artifact_id
                       << " attempt=" << reselection_attempt << "/" << max_reselection_attempts
                       << " remaining_budget_ms=" << remaining_label;
        }
        continue;
      }
      last_p2p_status =
          source_reselection_exhausted_status(artifact_id, reselection_attempt, request_deadline, last_p2p_status);
      record_source_reselection_exhausted("local_route", view_id.has_value(), reselection_attempt);
      break;
    }

    // Build P2PSource from server-selected remote replica
    P2PSource p2p_src;
    p2p_src.size_bytes = remote.memory_size;
    p2p_src.ip = remote.node_address;
    p2p_src.port = static_cast<uint16_t>(remote.node_port);
    p2p_src.memory_keys = remote.remote_memory_keys;
    p2p_src.buf_sizes = remote.buffer_sizes;
    p2p_src.verification_json = remote.verification_json;
    p2p_src.enable_checksum = true;
    p2p_src.source_is_view = session.route_kind == tensorcast::global_store::v1::TRANSPORT_ROUTE_KIND_RESIDENT_VIEW;
    p2p_src.location.type = remote.memory_type;
    p2p_src.location.device_id = remote.device_id;
    if (has_disk_source && allow_disk && preference != loading::SourcePreference::kPreferP2P) {
      p2p_src.fallback_disk_dir = disk_source->path.string();
    }

    // Build target description
    ReplicaTarget target;
    target.location.type = (target_device.type == DeviceType::GPU) ? common::memory::MemoryLocation::GPU
                                                                   : common::memory::MemoryLocation::CPU;
    target.location.device_id = target_device.ordinal;

    if (view_id.has_value() && session.route_kind == tensorcast::global_store::v1::TRANSPORT_ROUTE_KIND_RESIDENT_VIEW &&
        !is_local_replica(remote, local_identity_)) {
      auto begin_fetch_or = begin_remote_replica_fetch(session, artifact_id, *view_id);
      if (!begin_fetch_or.ok()) {
        LOG(WARNING) << "source-side fetch lifecycle unavailable for artifact_id=" << artifact_id
                     << " view_id=" << *view_id << " transport_id=" << session.transport_id << ": "
                     << begin_fetch_or.status();
      } else if (*begin_fetch_or) {
        remote_fetch_cleanup = RemoteReplicaFetchCleanup{
            .daemon_address = absl::StrCat(remote.node_address, ":", remote.grpc_port),
            .transport_id = session.transport_id,
        };
      }
    }

    auto load_or = backend_->ingest_from_p2p(std::string(artifact_id), p2p_src, target, hints);
    remote_fetch_reason = load_or.ok() ? "ingest_success" : "ingest_failure";
    if (remote_fetch_cleanup.has_value()) {
      const absl::Status end_status = end_remote_replica_fetch(*remote_fetch_cleanup, remote_fetch_reason);
      if (!end_status.ok()) {
        LOG(WARNING) << "EndReplicaFetch after ingest_from_p2p failed: " << end_status;
      }
      remote_fetch_cleanup.reset();
    }
    if (load_or.ok()) {
      // Notify GS that transport finished
      absl::Status comp_status = gs_client_->complete_replica_transport(
          session.transport_id, components::TransportCompletionOutcome::kSuccess);
      if (!comp_status.ok()) {
        LOG(WARNING) << "complete_replica_transport returned error: " << comp_status;
      }
      const auto& handle = *load_or;
      absl::Status reg_status = backend_->register_replica_with_global_store(handle.key(), {});
      if (!reg_status.ok()) {
        LOG(WARNING) << "register_replica_with_global_store returned error: " << reg_status;
      }

      if (used_canonical_transport_fallback && view_id.has_value()) {
        LOG(INFO) << "event=view_materialized_via_fallback"
                  << " route_kind=canonical_fallback"
                  << " artifact_id=" << artifact_id << " view_id=" << *view_id
                  << " transport_id=" << session.transport_id;
      }
      if (reselection_attempt > 0) {
        record_source_reselection_success("p2p_load", view_id.has_value(), reselection_attempt);
      }
      return load_or;
    } // Loading via P2P failed – close transport and log
    absl::Status comp_status = gs_client_->complete_replica_transport(
        session.transport_id, components::TransportCompletionOutcome::kFailed, load_or.status().ToString());
    if (!comp_status.ok()) {
      LOG(WARNING) << "complete_replica_transport after failure returned error: " << comp_status;
    }
    last_p2p_status = load_or.status();
    if (view_id.has_value()) {
      LOG(WARNING) << "P2P load failed for view_id=" << *view_id << ": " << load_or.status();
    } else {
      LOG(WARNING) << "P2P load failed: " << load_or.status();
    }

    if (can_retry_source_selection(last_p2p_status, reselection_attempt, request_deadline, max_reselection_attempts)) {
      record_stale_source_detected("p2p_load_failure", view_id.has_value(), reselection_attempt);
      reselection_attempt += 1;
      record_source_reselection_attempt("p2p_load_failure", view_id.has_value(), reselection_attempt);
      if (should_log_reselection_attempt(reselection_attempt)) {
        const std::chrono::milliseconds retry_remaining = remaining_request_budget(request_deadline);
        const std::string remaining_label =
            retry_remaining == std::chrono::milliseconds::max() ? "unbounded" : std::to_string(retry_remaining.count());
        LOG(WARNING) << "Retrying source selection after P2P load failure: artifact_id=" << artifact_id
                     << " attempt=" << reselection_attempt << "/" << max_reselection_attempts
                     << " remaining_budget_ms=" << remaining_label << " status=" << last_p2p_status;
      }
      continue;
    }
    if (should_retry_source_selection(last_p2p_status)) {
      last_p2p_status =
          source_reselection_exhausted_status(artifact_id, reselection_attempt, request_deadline, last_p2p_status);
      record_source_reselection_exhausted("p2p_load_failure", view_id.has_value(), reselection_attempt);
    }
    break;
  }

  if (had_transport_request && !transport_or.ok()) {
    // Not found or GS unavailable → fall back to disk
    const char* route_name = "request_replica_transport";
    if (view_id.has_value() && !used_canonical_transport_fallback) {
      route_name = "request_view_transport";
    }
    std::string fallback_note;
    if (used_canonical_transport_fallback && !view_transport_status.ok()) {
      fallback_note = absl::StrCat(" (view_transport_status=", view_transport_status.ToString(), ")");
    }
    LOG(INFO) << route_name << " failed: " << transport_or.status() << "; falling back to disk"
              << (view_id ? absl::StrCat(" (view_id=", *view_id, ")") : "") << fallback_note;
    if (!has_disk_source || !allow_disk) {
      return transport_or.status();
    }
  } else if (!gs_connected || !allow_p2p) {
    if (!has_disk_source || !allow_disk) {
      return transport_or.status();
    }
  } else if (!last_p2p_status.ok() && (!has_disk_source || !allow_disk)) {
    return last_p2p_status;
  }

  // ------------------------------------------------------------------
  // 3. Disk fallback
  // ------------------------------------------------------------------
  DiskSource disk_src = *disk_source;

  ReplicaTarget target;
  target.location.type = (target_device.type == DeviceType::GPU) ? common::memory::MemoryLocation::GPU
                                                                 : common::memory::MemoryLocation::CPU;
  target.location.device_id = target_device.ordinal;

  if (preference == loading::SourcePreference::kPreferDisk && attempted_disk_first) {
    if (!last_p2p_status.ok()) {
      return last_p2p_status;
    }
    return transport_or.ok() ? absl::FailedPreconditionError("disk source already attempted") : transport_or.status();
  }

  if (!allow_disk) {
    return absl::FailedPreconditionError("source_policy disallows disk materialization");
  }
  auto disk_or = backend_->ingest_from_disk(std::string(artifact_id), disk_src, target, hints);
  if (disk_or.ok()) {
    const auto& handle = *disk_or;
    register_with_global_store(handle);
  }
  return disk_or;
}

} // namespace tensorcast::store::materialization::control
