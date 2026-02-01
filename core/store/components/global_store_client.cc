// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/components/global_store_client.h"

#include <unistd.h>
#include <chrono>
#include <random>
#include <thread>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/otel/grpc_propagation.h"
#include "core/communicator/misc/utils.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/global_store/v1/global_store.grpc.pb.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::store::components {

// Backward-compatibility: map unversioned global_store symbols to v1
namespace global_store = tensorcast::global_store::v1;
namespace memory_tier = tensorcast::memory_tier::v1;

TransportLease::TransportLease(IGlobalStoreClient* client, std::string transport_id)
    : client_(client), transport_id_(std::move(transport_id)) {}

TransportLease::TransportLease(TransportLease&& other) noexcept
    : client_(other.client_), transport_id_(std::move(other.transport_id_)) {
  other.client_ = nullptr;
}

TransportLease& TransportLease::operator=(TransportLease&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  complete();
  client_ = other.client_;
  transport_id_ = std::move(other.transport_id_);
  other.client_ = nullptr;
  return *this;
}

TransportLease::~TransportLease() {
  complete();
}

void TransportLease::release() {
  client_ = nullptr;
}

void TransportLease::complete() {
  if (!client_ || transport_id_.empty()) {
    return;
  }
  absl::Status st = client_->complete_replica_transport(transport_id_);
  if (!st.ok()) {
    LOG(WARNING) << "complete_replica_transport failed for " << transport_id_ << ": " << st;
  }
  client_ = nullptr;
  transport_id_.clear();
}

namespace {
// Map Global Store Status enum to readable string
const char* status_to_cstr(global_store::Status s) {
  switch (s) {
    case global_store::STATUS_UNSPECIFIED:
      return "STATUS_UNSPECIFIED";
    case global_store::STATUS_NOT_FOUND:
      return "STATUS_NOT_FOUND";
    case global_store::STATUS_TIMED_OUT:
      return "STATUS_TIMED_OUT";
    case global_store::STATUS_TOO_MANY_REQUESTS:
      return "STATUS_TOO_MANY_REQUESTS";
    case global_store::STATUS_STATE_SYNC_REQUIRED:
      return "STATUS_STATE_SYNC_REQUIRED";
    case global_store::STATUS_ERROR:
      return "STATUS_ERROR";
    case global_store::STATUS_OK:
      return "STATUS_OK";
    default:
      return "<unknown>";
  }
}

memory_tier::LeaseKind to_proto_kind(MemoryTierLeaseKind kind) {
  switch (kind) {
    case MemoryTierLeaseKind::kPreemptible:
      return memory_tier::LEASE_KIND_PREEMPTIBLE;
    case MemoryTierLeaseKind::kStable:
    default:
      return memory_tier::LEASE_KIND_STABLE;
  }
}

MemoryTierLeaseKind from_proto_kind(memory_tier::LeaseKind kind) {
  switch (kind) {
    case memory_tier::LEASE_KIND_PREEMPTIBLE:
      return MemoryTierLeaseKind::kPreemptible;
    case memory_tier::LEASE_KIND_STABLE:
    default:
      return MemoryTierLeaseKind::kStable;
  }
}

memory_tier::LeaseState to_proto_state(MemoryTierLeaseState state) {
  switch (state) {
    case MemoryTierLeaseState::kActive:
      return memory_tier::LEASE_STATE_ACTIVE;
    case MemoryTierLeaseState::kRevoking:
      return memory_tier::LEASE_STATE_REVOKING;
    case MemoryTierLeaseState::kExpired:
      return memory_tier::LEASE_STATE_EXPIRED;
    case MemoryTierLeaseState::kPending:
    default:
      return memory_tier::LEASE_STATE_PENDING;
  }
}

MemoryTierLeaseState from_proto_state(memory_tier::LeaseState state) {
  switch (state) {
    case memory_tier::LEASE_STATE_ACTIVE:
      return MemoryTierLeaseState::kActive;
    case memory_tier::LEASE_STATE_REVOKING:
      return MemoryTierLeaseState::kRevoking;
    case memory_tier::LEASE_STATE_EXPIRED:
      return MemoryTierLeaseState::kExpired;
    case memory_tier::LEASE_STATE_PENDING:
    default:
      return MemoryTierLeaseState::kPending;
  }
}

memory_tier::LeaseAckAction to_proto_action(MemoryTierAckAction action) {
  switch (action) {
    case MemoryTierAckAction::kReleased:
      return memory_tier::LEASE_ACK_ACTION_RELEASED;
    case MemoryTierAckAction::kAcquired:
    default:
      return memory_tier::LEASE_ACK_ACTION_ACQUIRED;
  }
}

MemoryTierLeaseDescriptor from_proto_lease(const memory_tier::MemoryTierLease& lease) {
  MemoryTierLeaseDescriptor out;
  out.lease_id = lease.lease_id();
  out.node_id = lease.node_id();
  out.kind = from_proto_kind(lease.kind());
  out.artifact_id = lease.artifact_id();
  out.chunk_start = lease.chunk_range().start();
  out.chunk_count = lease.chunk_range().count();
  out.chunk_ids.assign(lease.chunk_ids().begin(), lease.chunk_ids().end());
  out.ledger_version = lease.ledger_version();
  out.bytes = lease.bytes();
  out.workload_id = lease.workload_id();
  out.state = from_proto_state(lease.state());
  out.request_id = lease.request_id();
  out.ack_epoch_ns = lease.ack_epoch_ns();
  out.issued_at_ns = lease.issued_at_ns();
  out.expires_at_ns = lease.expires_at_ns();
  return out;
}

bool is_loopback_or_unspecified(absl::string_view addr) {
  if (addr.empty())
    return true;
  if (addr == "localhost" || addr == "ip6-localhost" || addr == "*" || addr == "0.0.0.0")
    return true;

  absl::string_view trimmed = addr;
  if (trimmed.front() == '[' && trimmed.back() == ']') {
    trimmed.remove_prefix(1);
    trimmed.remove_suffix(1);
  }

  return trimmed == "127.0.0.1" || trimmed == "::" || trimmed == "::1" || trimmed == "0:0:0:0:0:0:0:1";
}

std::optional<absl::Time> timestamp_to_absl(const google::protobuf::Timestamp& ts) {
  if (ts.seconds() == 0 && ts.nanos() == 0) {
    return std::nullopt;
  }
  return absl::UnixEpoch() + absl::Seconds(ts.seconds()) + absl::Nanoseconds(ts.nanos());
}
} // namespace

using common::memory::MemoryLocation;
using replica::ChunkState;

namespace {

std::shared_ptr<grpc::Channel> make_channel(const GlobalStoreClientConfig& config) {
  grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 30000);
  args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 20000);
  args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);

  auto channel = grpc::CreateCustomChannel(config.global_store_address, grpc::InsecureChannelCredentials(), args);
  ABSL_CHECK(channel != nullptr);
  return channel;
}

} // namespace

GlobalStoreClient::GlobalStoreClient(GlobalStoreClientConfig config)
    : config_(std::move(config)),
      channel_(gsl::not_null<std::shared_ptr<grpc::Channel>>(make_channel(config_))),
      stub_(
          gsl::not_null<std::unique_ptr<global_store::GlobalStoreService::Stub>>(
              global_store::GlobalStoreService::NewStub(channel_.get()))),
      memory_tier_stub_(
          gsl::not_null<std::unique_ptr<memory_tier::MemoryTierService::Stub>>(
              memory_tier::MemoryTierService::NewStub(channel_.get()))) {}

GlobalStoreClient::~GlobalStoreClient() = default;

absl::Status GlobalStoreClient::initialize() {
  // Test connection with a health check
  global_store::HealthCheckRequest req;
  global_store::HealthCheckResponse resp;

  grpc::ClientContext context;
  context.set_deadline(
      std::chrono::system_clock::now() + std::chrono::seconds(absl::ToInt64Seconds(config_.connection_timeout)));
  if (!config_.cluster_token.empty()) {
    context.AddMetadata("x-tensorcast-cluster-token", config_.cluster_token);
  }

  // OpenTelemetry: create a client span for HealthCheck and inject context.
  namespace otel = opentelemetry;
  auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
  otel::trace::StartSpanOptions opts;
  opts.kind = otel::trace::SpanKind::kClient;
  auto span = tracer->StartSpan("GlobalStore/HealthCheck", opts);
  otel::trace::Scope scope(span);
  span->SetAttribute("rpc.system", "grpc");
  span->SetAttribute("rpc.service", "tensorcast.global_store.GlobalStoreService");
  span->SetAttribute("rpc.method", "HealthCheck");
  tensorcast::common::otel::inject_into_client_metadata(context);

  auto status = stub_->HealthCheck(&context, req, &resp);
  if (!status.ok()) {
    return absl::UnavailableError(
        absl::StrFormat(
            "Failed to connect to Global Store at %s: %s", config_.global_store_address, status.error_message()));
  }

  if (!config_.cluster_token.empty()) {
    if (resp.cluster_token().empty()) {
      return absl::FailedPreconditionError(
          "Global Store health check did not return a cluster_token; refusing to continue with configured token.");
    }
    if (resp.cluster_token() != config_.cluster_token) {
      return absl::FailedPreconditionError(
          absl::StrFormat(
              "Global Store cluster_token mismatch: expected '%s', got '%s'",
              config_.cluster_token,
              resp.cluster_token()));
    }
  }

  LOG(INFO) << "Successfully connected to Global Store at " << config_.global_store_address;
  return absl::OkStatus();
}

absl::Status GlobalStoreClient::publish_memory_tier_status(const MemoryTierStatusPayload& status) {
  memory_tier::PublishMemoryTierStatusRequest request;
  auto* s = request.mutable_status();
  s->set_node_id(status.node_id);
  s->set_worker_id(status.worker_id);
  s->set_stable_total_bytes(status.stable_total_bytes);
  s->set_stable_used_bytes(status.stable_used_bytes);
  s->set_preemptible_total_bytes(status.preemptible_total_bytes);
  s->set_preemptible_marked_bytes(status.preemptible_marked_bytes);
  s->set_faults_per_sec(status.faults_per_sec);
  s->set_rehydrate_p99_ns(status.rehydrate_p99_ns);
  s->set_enable_preemptible(status.enable_preemptible);
  s->set_memory_tier_config_json(status.memory_tier_config_json);
  const uint64_t epoch_ns =
      status.epoch_ns != 0 ? status.epoch_ns : static_cast<uint64_t>(absl::ToUnixNanos(absl::Now()));
  s->set_epoch_ns(epoch_ns);

  memory_tier::PublishMemoryTierStatusResponse response;
  return execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) {
        return memory_tier_stub_->PublishMemoryTierStatus(ctx, req, resp);
      },
      "PublishMemoryTierStatus");
}

absl::StatusOr<MemoryTierLeaseDescriptor> GlobalStoreClient::request_memory_tier_lease(
    const MemoryTierLeaseDescriptor& request_desc) {
  memory_tier::RequestMemoryTierLeaseRequest request;
  request.set_node_id(request_desc.node_id);
  request.set_kind(to_proto_kind(request_desc.kind));
  request.set_artifact_id(request_desc.artifact_id);
  request.set_ledger_version(request_desc.ledger_version);
  request.set_bytes(request_desc.bytes);
  request.set_workload_id(request_desc.workload_id);
  request.set_request_id(request_desc.request_id);
  if (request_desc.issued_at_ns != 0) {
    request.set_issued_at_ns(request_desc.issued_at_ns);
  }
  auto* cr = request.mutable_chunk_range();
  cr->set_start(request_desc.chunk_start);
  cr->set_count(request_desc.chunk_count);
  for (uint32_t id : request_desc.chunk_ids)
    request.add_chunk_ids(id);

  memory_tier::RequestMemoryTierLeaseResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) {
        return memory_tier_stub_->RequestMemoryTierLease(ctx, req, resp);
      },
      "RequestMemoryTierLease");
  if (!status.ok()) {
    return status;
  }
  return from_proto_lease(response.lease());
}

absl::StatusOr<MemoryTierLeaseDescriptor> GlobalStoreClient::acknowledge_memory_tier_lease(
    const MemoryTierLeaseAckPayload& ack) {
  if (ack.artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required for MemoryTierLease acknowledgements");
  }
  memory_tier::AcknowledgeMemoryTierLeaseRequest request;
  request.set_lease_id(ack.lease_id);
  request.set_node_id(ack.node_id);
  request.set_action(to_proto_action(ack.action));
  request.set_ledger_version(ack.ledger_version);
  request.set_bytes(ack.bytes);
  request.set_request_id(ack.request_id);
  request.set_artifact_id(ack.artifact_id);
  if (ack.ack_epoch_ns != 0) {
    request.set_ack_epoch_ns(ack.ack_epoch_ns);
  }
  auto* cr = request.mutable_chunk_range();
  cr->set_start(ack.chunk_start);
  cr->set_count(ack.chunk_count);
  for (uint32_t id : ack.chunk_ids)
    request.add_chunk_ids(id);

  memory_tier::AcknowledgeMemoryTierLeaseResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) {
        return memory_tier_stub_->AcknowledgeMemoryTierLease(ctx, req, resp);
      },
      "AcknowledgeMemoryTierLease");
  if (!status.ok()) {
    return status;
  }
  return from_proto_lease(response.lease());
}

absl::StatusOr<std::vector<MemoryTierLeaseDescriptor>> GlobalStoreClient::list_memory_tier_leases(
    std::string_view node_id) {
  memory_tier::ListOutstandingLeasesRequest request;
  request.set_node_id(std::string(node_id));
  // Default states: pending, active, revoking (mirror server default)
  request.add_states(memory_tier::LEASE_STATE_PENDING);
  request.add_states(memory_tier::LEASE_STATE_ACTIVE);
  request.add_states(memory_tier::LEASE_STATE_REVOKING);

  memory_tier::ListOutstandingLeasesResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) {
        return memory_tier_stub_->ListOutstandingLeases(ctx, req, resp);
      },
      "ListOutstandingLeases");
  if (!status.ok()) {
    return status;
  }
  std::vector<MemoryTierLeaseDescriptor> out;
  out.reserve(response.leases_size());
  for (const auto& l : response.leases()) {
    out.push_back(from_proto_lease(l));
  }
  return out;
}

absl::StatusOr<MemoryTierLeaseDescriptor> GlobalStoreClient::revoke_memory_tier_lease(std::string_view lease_id) {
  memory_tier::RevokeMemoryTierLeaseRequest request;
  request.set_lease_id(std::string(lease_id));

  memory_tier::RevokeMemoryTierLeaseResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) {
        return memory_tier_stub_->RevokeMemoryTierLease(ctx, req, resp);
      },
      "RevokeMemoryTierLease");
  if (!status.ok()) {
    return status;
  }
  return from_proto_lease(response.lease());
}

absl::StatusOr<WorkerRegistrationInfo> GlobalStoreClient::register_worker(
    std::string_view node_id,
    std::string_view node_address,
    uint32_t grpc_port,
    uint32_t p2p_port,
    uint64_t mem_pool_total_size,
    uint64_t mem_pool_available_size,
    bool is_recovery_registration,
    std::string_view previous_worker_id,
    std::string_view daemon_id,
    uint64_t capability_flags) {
  if (daemon_id.empty()) {
    return absl::InvalidArgumentError(
        "register_worker requires a non-empty daemon_id; configure DaemonConfig.daemon_id.");
  }
  if (is_loopback_or_unspecified(node_address)) {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "Invalid node_address '%s'. Global Store requires a routable (non-loopback) address; configure --advertise_host "
            "or listen_addr accordingly.",
            node_address));
  }

  if (grpc_port == 0) {
    return absl::InvalidArgumentError(
        "register_worker requires a non-zero grpc_port; ensure the daemon listen_addr includes a port component.");
  }

  if (p2p_port == 0) {
    return absl::InvalidArgumentError(
        "register_worker requires a non-zero p2p_port; configure the daemon with --p2p_listen or provide a valid port.");
  }

  global_store::RegisterWorkerRequest request;
  request.set_node_id(std::string(node_id));
  request.set_node_address(std::string(node_address));
  request.set_grpc_port(grpc_port);
  request.set_p2p_port(p2p_port);
  request.set_mem_pool_total_size(mem_pool_total_size);
  request.set_mem_pool_available_size(mem_pool_available_size);
  request.set_capability_flags(capability_flags);
  if (is_recovery_registration) {
    request.set_is_recovery_registration(true);
    if (!previous_worker_id.empty()) {
      request.set_previous_worker_id(std::string(previous_worker_id));
    }
  }
  request.set_daemon_id(std::string(daemon_id));

  global_store::RegisterWorkerResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RegisterWorker(ctx, req, resp); },
      "RegisterWorker");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat(
            "RegisterWorker failed: %s (%d). This usually means the Global Store rejected the registration.\n"
            "Common causes: duplicate worker on %s:%u from another host (node_id mismatch), or server-side error.\n"
            "Suggested actions: ensure no other daemon owns %s:%u, verify node_id/host configuration, and check Global Store logs.",
            status_to_cstr(response.status()),
            static_cast<int>(response.status()),
            request.node_address().c_str(),
            request.grpc_port(),
            request.node_address().c_str(),
            request.grpc_port()));
  }

  if (response.expected_state_version() == 0) {
    return absl::FailedPreconditionError(
        "RegisterWorker response missing expected_state_version; Global Store must return a non-zero state version.");
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    worker_id_ = response.worker_id();
    node_id_ = std::string(node_id);
    node_address_ = std::string(node_address);
    grpc_port_ = grpc_port;
    p2p_port_ = p2p_port;
  }

  LOG(INFO) << "Registered worker with ID: " << response.worker_id();
  WorkerRegistrationInfo info;
  info.worker_id = response.worker_id();
  info.expected_state_version = response.expected_state_version();
  info.state_sync_required = response.state_sync_required();
  info.heartbeat_interval_ms = response.heartbeat_interval_ms();
  return info;
}

absl::StatusOr<global_store::WorkerHeartbeatResponse> GlobalStoreClient::send_heartbeat_enhanced(
    std::string_view worker_id,
    uint64_t mem_pool_available_size,
    bool accepting_new_requests,
    uint64_t state_version,
    std::string_view state_checksum,
    const std::vector<std::string>& registered_artifact_ids,
    int64_t last_successful_sync,
    global_store::ConnectionStatus connection_status,
    const RpcOptions& rpc_options,
    std::string_view daemon_id,
    uint64_t capability_flags) {
  global_store::WorkerHeartbeatRequest request;
  request.set_worker_id(std::string(worker_id));
  request.set_mem_pool_available_size(mem_pool_available_size);
  request.set_accepting_new_requests(accepting_new_requests);
  request.set_state_version(state_version);
  request.set_state_checksum(std::string(state_checksum));
  for (const auto& id : registered_artifact_ids) {
    request.add_registered_artifact_ids(id);
  }
  request.set_last_successful_sync(last_successful_sync);
  request.set_global_store_status(connection_status);
  request.set_capability_flags(capability_flags);
  if (!daemon_id.empty()) {
    request.set_daemon_id(std::string(daemon_id));
  }

  global_store::WorkerHeartbeatResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->WorkerHeartbeat(ctx, req, resp); },
      "WorkerHeartbeat(enhanced)",
      rpc_options);
  if (!status.ok())
    return status;
  if (response.status() != global_store::STATUS_OK && response.status() != global_store::STATUS_STATE_SYNC_REQUIRED) {
    return absl::InternalError(
        absl::StrFormat(
            "WorkerHeartbeat(enhanced) failed: %s (%d)",
            status_to_cstr(response.status()),
            static_cast<int>(response.status())));
  }
  return response;
}

absl::Status GlobalStoreClient::unregister_worker(std::string_view worker_id, bool is_graceful_shutdown) {
  global_store::UnregisterWorkerRequest request;
  request.set_worker_id(std::string(worker_id));
  request.set_is_graceful_shutdown(is_graceful_shutdown);

  global_store::UnregisterWorkerResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->UnregisterWorker(ctx, req, resp); },
      "UnregisterWorker");

  if (!status.ok()) {
    return status;
  }

  if (response.status() == global_store::STATUS_OK) {
    VLOG(1) << "Unregistered worker with ID: " << worker_id
            << (is_graceful_shutdown ? " (graceful)" : " (non-graceful)");
    return absl::OkStatus();
  }
  if (response.status() == global_store::STATUS_NOT_FOUND) {
    // Idempotency: if the worker is already absent (e.g., previous unregister
    // succeeded or GS restarted), treat as success to avoid noisy shutdown logs.
    VLOG(1) << "UnregisterWorker returned NOT_FOUND for worker_id=" << worker_id << "; treating as success";
    return absl::OkStatus();
  }
  return absl::InternalError(
      absl::StrFormat(
          "UnregisterWorker failed: %s (%d)", status_to_cstr(response.status()), static_cast<int>(response.status())));
}

absl::StatusOr<std::string> GlobalStoreClient::register_replica(
    std::string_view artifact_id,
    std::string_view worker_id,
    const DeviceKey& device,
    MemoryLocation location,
    uint64_t memory_size,
    uint32_t max_concurrency,
    std::optional<std::string_view> view_id) {
  global_store::RegisterReplicaRequest request;
  request.set_artifact_id(std::string(artifact_id));
  request.set_worker_id(std::string(worker_id));
  request.set_max_concurrency(max_concurrency);

  auto* mem_info = request.mutable_mem_info();
  if (auto fill_st = fill_memory_info(mem_info, device, location, memory_size, view_id); !fill_st.ok()) {
    return fill_st;
  }

  global_store::RegisterReplicaResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RegisterReplica(ctx, req, resp); },
      "RegisterReplica");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat(
            "RegisterReplica failed: %s (%d)", status_to_cstr(response.status()), static_cast<int>(response.status())));
  }

  LOG(INFO) << "Registered replica: " << artifact_id << " with ID: " << response.replica_id();
  return response.replica_id();
}

absl::Status GlobalStoreClient::record_view_residency(
    std::string_view canonical_artifact_id,
    std::string_view view_id,
    uint64_t view_size_bytes,
    std::optional<std::string_view> view_data_hash) {
  (void)canonical_artifact_id;
  (void)view_id;
  (void)view_size_bytes;
  (void)view_data_hash;
  // A dedicated RPC for view metadata will be introduced for view-residency signals.
  // Until that lands, treat this as a best-effort noop so core plumbing can wire
  // the call sites without coupling to server availability.
  return absl::UnimplementedError("Global Store view residency RPC not yet implemented");
}

absl::Status GlobalStoreClient::update_artifact_view_state(const ViewStateUpdate& update) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (update.artifact_id.empty()) {
    return absl::InvalidArgumentError("update_artifact_view_state requires artifact_id");
  }
  const bool has_view = !update.view_id.empty() || !update.view_spec_json.empty() || update.view_size_bytes > 0 ||
      update.view_data_hash.has_value() || update.mark_verified || update.canonical_size_bytes > 0 ||
      update.canonical_bytes_covered > 0 || !update.canonical_ranges.empty();
  if (has_view && update.view_id.empty()) {
    return absl::InvalidArgumentError("update_artifact_view_state view payload requires view_id");
  }
  if (!update.proof_digests.empty() && update.view_id.empty()) {
    return absl::InvalidArgumentError("update_artifact_view_state proof digests require view_id");
  }

  global_store::UpdateArtifactViewStateRequest request;
  request.set_artifact_id(update.artifact_id);
  if (has_view) {
    auto* view = request.mutable_view();
    view->set_view_id(update.view_id);
    view->set_view_spec_json(update.view_spec_json);
    view->set_view_size(update.view_size_bytes);
    if (update.view_data_hash.has_value()) {
      view->set_view_data_hash(*update.view_data_hash);
    }
    if (update.canonical_size_bytes > 0 || update.canonical_bytes_covered > 0) {
      auto* coverage = view->mutable_canonical_coverage();
      coverage->set_total_bytes(update.canonical_size_bytes);
      coverage->set_covered_bytes(update.canonical_bytes_covered);
    }
    for (const auto& range : update.canonical_ranges) {
      auto* out = view->add_canonical_ranges();
      out->set_off(range.offset);
      out->set_len(range.length);
    }
    if (update.mark_verified) {
      const absl::Time now = absl::Now();
      auto* ts = view->mutable_verified_at();
      const int64_t seconds = absl::ToUnixSeconds(now);
      const int64_t nanos = absl::ToInt64Nanoseconds(now - absl::UnixEpoch() - absl::Seconds(seconds));
      ts->set_seconds(seconds);
      ts->set_nanos(static_cast<int32_t>(nanos));
    }
  }
  for (const auto& leaf : update.leaf_writes) {
    *request.add_leaf_writes() = leaf;
  }
  for (const auto& digest : update.proof_digests) {
    *request.add_proof_digests() = digest;
  }

  global_store::UpdateArtifactViewStateResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->UpdateArtifactViewState(ctx, req, resp); },
      "UpdateArtifactViewState");
  if (!status.ok()) {
    return status;
  }
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat("UpdateArtifactViewState failed: status=%s", status_to_cstr(response.status())));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<ViewInfo>> GlobalStoreClient::list_views(std::string_view artifact_id) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("list_views requires artifact_id");
  }

  std::vector<ViewInfo> views;
  std::string next_page_token;
  do {
    global_store::ListViewsRequest request;
    request.set_artifact_id(std::string(artifact_id));
    auto* pagination = request.mutable_pagination();
    pagination->set_page_size(200);
    if (!next_page_token.empty()) {
      pagination->set_page_token(next_page_token);
    }

    global_store::ListViewsResponse response;
    auto status = execute_rpc_with_retry(
        request,
        &response,
        [this](auto* ctx, const auto& req, auto* resp) { return stub_->ListViews(ctx, req, resp); },
        "ListViews");
    if (!status.ok()) {
      return status;
    }
    if (response.status() == global_store::STATUS_NOT_FOUND) {
      return absl::NotFoundError(absl::StrCat("views not found for artifact_id=", artifact_id));
    }
    if (response.status() != global_store::STATUS_OK) {
      return absl::InternalError(absl::StrFormat("ListViews failed: status=%s", status_to_cstr(response.status())));
    }

    for (const auto& entry : response.views()) {
      ViewInfo info;
      info.view_id = entry.view_id();
      info.view_spec_json = entry.view_spec_json();
      info.view_size_bytes = entry.view_size();
      if (!entry.view_data_hash().empty()) {
        info.view_data_hash = entry.view_data_hash();
      }
      if (entry.has_verified_at()) {
        info.verified_at = timestamp_to_absl(entry.verified_at());
      }
      if (entry.has_canonical_coverage()) {
        info.canonical_size_bytes = entry.canonical_coverage().total_bytes();
        info.canonical_bytes_covered = entry.canonical_coverage().covered_bytes();
      }
      info.canonical_ranges.reserve(static_cast<size_t>(entry.canonical_ranges_size()));
      for (const auto& range : entry.canonical_ranges()) {
        info.canonical_ranges.push_back(CanonicalRange{.offset = range.off(), .length = range.len()});
      }
      views.push_back(std::move(info));
    }

    next_page_token = response.has_page_info() ? response.page_info().next_page_token() : std::string();
  } while (!next_page_token.empty());

  return views;
}

absl::StatusOr<global_store::AssemblyLayoutBinding> GlobalStoreClient::get_assembly_layout_binding(
    std::string_view assembly_id) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (assembly_id.empty()) {
    return absl::InvalidArgumentError("get_assembly_layout_binding requires assembly_id");
  }

  global_store::GetAssemblyLayoutBindingRequest request;
  request.set_assembly_id(std::string(assembly_id));

  global_store::GetAssemblyLayoutBindingResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->GetAssemblyLayoutBinding(ctx, req, resp); },
      "GetAssemblyLayoutBinding");
  if (!status.ok()) {
    return status;
  }
  if (response.status() == global_store::STATUS_NOT_FOUND) {
    return absl::NotFoundError(absl::StrCat("assembly layout binding not found for ", assembly_id));
  }
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat("GetAssemblyLayoutBinding failed: status=%s", status_to_cstr(response.status())));
  }
  if (!response.has_binding()) {
    return absl::InternalError("GetAssemblyLayoutBinding returned empty binding");
  }
  return response.binding();
}

absl::StatusOr<layout::LayoutSpecRecord> GlobalStoreClient::get_layout_spec(std::string_view layout_id) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (layout_id.empty()) {
    return absl::InvalidArgumentError("get_layout_spec requires layout_id");
  }

  global_store::GetLayoutSpecRequest request;
  request.set_layout_id(std::string(layout_id));

  global_store::GetLayoutSpecResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->GetLayoutSpec(ctx, req, resp); },
      "GetLayoutSpec");
  if (!status.ok()) {
    return status;
  }
  if (response.status() == global_store::STATUS_NOT_FOUND) {
    return absl::NotFoundError(absl::StrCat("layout spec not found for ", layout_id));
  }
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(absl::StrFormat("GetLayoutSpec failed: status=%s", status_to_cstr(response.status())));
  }
  if (!response.has_record()) {
    return absl::InternalError("GetLayoutSpec returned empty record");
  }
  return response.record();
}

absl::Status GlobalStoreClient::attach_layout_to_artifact(std::string_view mi2_id, std::string_view layout_id) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (mi2_id.empty() || layout_id.empty()) {
    return absl::InvalidArgumentError("attach_layout_to_artifact requires mi2_id and layout_id");
  }

  global_store::AttachLayoutToArtifactRequest request;
  request.set_mi2_id(std::string(mi2_id));
  request.set_layout_id(std::string(layout_id));

  global_store::AttachLayoutToArtifactResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->AttachLayoutToArtifact(ctx, req, resp); },
      "AttachLayoutToArtifact");
  if (!status.ok()) {
    return status;
  }
  if (response.status() != global_store::STATUS_OK) {
    if (response.status() == global_store::STATUS_NOT_FOUND) {
      return absl::NotFoundError(
          absl::StrCat("AttachLayoutToArtifact not found: mi2_id=", mi2_id, " layout_id=", layout_id));
    }
    return absl::InternalError(
        absl::StrFormat("AttachLayoutToArtifact failed: status=%s", status_to_cstr(response.status())));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::string>> GlobalStoreClient::list_artifact_layouts(std::string_view mi2_id) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (mi2_id.empty()) {
    return absl::InvalidArgumentError("list_artifact_layouts requires mi2_id");
  }

  global_store::ListArtifactLayoutsRequest request;
  request.set_mi2_id(std::string(mi2_id));

  global_store::ListArtifactLayoutsResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->ListArtifactLayouts(ctx, req, resp); },
      "ListArtifactLayouts");
  if (!status.ok()) {
    return status;
  }
  if (response.status() == global_store::STATUS_NOT_FOUND) {
    return absl::NotFoundError(absl::StrCat("artifact layouts not found for ", mi2_id));
  }
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat("ListArtifactLayouts failed: status=%s", status_to_cstr(response.status())));
  }
  std::vector<std::string> layout_ids;
  layout_ids.reserve(static_cast<size_t>(response.layout_ids_size()));
  for (const auto& layout_id : response.layout_ids()) {
    layout_ids.push_back(layout_id);
  }
  return layout_ids;
}

absl::StatusOr<global_store::WriteTensorProofCommitmentsResponse> GlobalStoreClient::write_tensor_proof_commitments(
    const global_store::WriteTensorProofCommitmentsRequest& request) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (request.mi2_id().empty()) {
    return absl::InvalidArgumentError("write_tensor_proof_commitments requires mi2_id");
  }
  if (request.proof_schema_version().empty()) {
    return absl::InvalidArgumentError("write_tensor_proof_commitments requires proof_schema_version");
  }

  global_store::WriteTensorProofCommitmentsResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->WriteTensorProofCommitments(ctx, req, resp); },
      "WriteTensorProofCommitments");
  if (!status.ok()) {
    return status;
  }
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat("WriteTensorProofCommitments failed: status=%s", status_to_cstr(response.status())));
  }
  return response;
}

absl::StatusOr<global_store::CheckProofCommitmentsMatchResponse> GlobalStoreClient::check_proof_commitments_match(
    const global_store::CheckProofCommitmentsMatchRequest& request) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (request.assembly_id().empty() || request.mi2_id().empty()) {
    return absl::InvalidArgumentError("check_proof_commitments_match requires assembly_id and mi2_id");
  }
  if (request.proof_schema_version().empty()) {
    return absl::InvalidArgumentError("check_proof_commitments_match requires proof_schema_version");
  }
  if (request.tensor_names().empty()) {
    return absl::InvalidArgumentError("check_proof_commitments_match requires tensor_names");
  }

  global_store::CheckProofCommitmentsMatchResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->CheckProofCommitmentsMatch(ctx, req, resp); },
      "CheckProofCommitmentsMatch");
  if (!status.ok()) {
    return status;
  }
  if (response.status() == global_store::STATUS_NOT_FOUND) {
    return absl::NotFoundError("CheckProofCommitmentsMatch not found");
  }
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat("CheckProofCommitmentsMatch failed: status=%s", status_to_cstr(response.status())));
  }
  return response;
}

absl::StatusOr<global_store::AssemblyRuntimePolicy> GlobalStoreClient::get_assembly_runtime_policy(
    std::string_view assembly_id) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (assembly_id.empty()) {
    return absl::InvalidArgumentError("get_assembly_runtime_policy requires assembly_id");
  }

  global_store::GetAssemblyRuntimePolicyRequest request;
  request.set_assembly_id(std::string(assembly_id));

  global_store::GetAssemblyRuntimePolicyResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->GetAssemblyRuntimePolicy(ctx, req, resp); },
      "GetAssemblyRuntimePolicy");
  if (!status.ok()) {
    return status;
  }
  if (response.status() == global_store::STATUS_NOT_FOUND) {
    return absl::NotFoundError(absl::StrCat("assembly runtime policy not found for ", assembly_id));
  }
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat("GetAssemblyRuntimePolicy failed: status=%s", status_to_cstr(response.status())));
  }
  if (!response.has_policy()) {
    return absl::InternalError("GetAssemblyRuntimePolicy returned empty policy");
  }
  return response.policy();
}

absl::StatusOr<operation::AcquireOperationLeaseResponse> GlobalStoreClient::acquire_operation_lease(
    const operation::AcquireOperationLeaseRequest& request) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  operation::AcquireOperationLeaseResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->AcquireOperationLease(ctx, req, resp); },
      "AcquireOperationLease");
  if (!status.ok()) {
    return status;
  }
  return response;
}

absl::StatusOr<operation::KeepaliveOperationLeaseResponse> GlobalStoreClient::keepalive_operation_lease(
    const operation::KeepaliveOperationLeaseRequest& request) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  operation::KeepaliveOperationLeaseResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->KeepaliveOperationLease(ctx, req, resp); },
      "KeepaliveOperationLease");
  if (!status.ok()) {
    return status;
  }
  return response;
}

absl::StatusOr<operation::ReleaseOperationLeaseResponse> GlobalStoreClient::release_operation_lease(
    const operation::ReleaseOperationLeaseRequest& request) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  operation::ReleaseOperationLeaseResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->ReleaseOperationLease(ctx, req, resp); },
      "ReleaseOperationLease");
  if (!status.ok()) {
    return status;
  }
  return response;
}

absl::StatusOr<operation::GetOperationResponse> GlobalStoreClient::get_operation(
    const operation::GetOperationRequest& request) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  operation::GetOperationResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->GetOperation(ctx, req, resp); },
      "GetOperation");
  if (!status.ok()) {
    return status;
  }
  return response;
}

absl::Status GlobalStoreClient::update_operation(const operation::UpdateOperationRequest& request) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  operation::UpdateOperationResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->UpdateOperation(ctx, req, resp); },
      "UpdateOperation");
  if (!status.ok()) {
    return status;
  }
  return absl::OkStatus();
}

absl::StatusOr<ArtifactBinding> GlobalStoreClient::get_artifact_binding(std::string_view artifact_id) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("get_artifact_binding requires artifact_id");
  }
  global_store::GetArtifactBindingRequest request;
  request.set_artifact_id(std::string(artifact_id));
  global_store::GetArtifactBindingResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->GetArtifactBinding(ctx, req, resp); },
      "GetArtifactBinding");
  if (!status.ok()) {
    return status;
  }
  if (response.status() == global_store::STATUS_NOT_FOUND) {
    return absl::NotFoundError(absl::StrCat("artifact binding not found for ", artifact_id));
  }
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat("GetArtifactBinding failed: status=%s", status_to_cstr(response.status())));
  }
  if (!response.has_binding()) {
    return absl::InternalError("GetArtifactBinding returned empty binding");
  }
  const auto& binding_proto = response.binding();
  ArtifactBinding binding;
  binding.from_artifact_id = binding_proto.from_artifact_id();
  binding.to_artifact_id = binding_proto.to_artifact_id();
  binding.kind = binding_proto.kind();
  if (binding_proto.has_created_at()) {
    binding.created_at = timestamp_to_absl(binding_proto.created_at());
  }
  return binding;
}

absl::StatusOr<ArtifactBindingResult> GlobalStoreClient::upsert_artifact_binding(const ArtifactBinding& binding) {
  if (!is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (binding.from_artifact_id.empty() || binding.to_artifact_id.empty()) {
    return absl::InvalidArgumentError("upsert_artifact_binding requires from/to artifact ids");
  }
  global_store::UpsertArtifactBindingRequest request;
  auto* binding_proto = request.mutable_binding();
  binding_proto->set_from_artifact_id(binding.from_artifact_id);
  binding_proto->set_to_artifact_id(binding.to_artifact_id);
  binding_proto->set_kind(binding.kind);

  global_store::UpsertArtifactBindingResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->UpsertArtifactBinding(ctx, req, resp); },
      "UpsertArtifactBinding");
  if (!status.ok()) {
    return status;
  }
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat("UpsertArtifactBinding failed: status=%s", status_to_cstr(response.status())));
  }
  if (!response.has_binding()) {
    return absl::InternalError("UpsertArtifactBinding returned empty binding");
  }
  ArtifactBindingResult out;
  const auto& resp_binding = response.binding();
  out.binding.from_artifact_id = resp_binding.from_artifact_id();
  out.binding.to_artifact_id = resp_binding.to_artifact_id();
  out.binding.kind = resp_binding.kind();
  if (resp_binding.has_created_at()) {
    out.binding.created_at = timestamp_to_absl(resp_binding.created_at());
  }
  out.created = response.created();
  return out;
}

absl::StatusOr<std::string> GlobalStoreClient::register_memory_replica(
    std::string_view artifact_id,
    std::string_view worker_id,
    const DeviceKey& device,
    uint64_t memory_size,
    std::string_view tensor_index_key,
    const std::vector<std::string>& remote_memory_keys,
    const std::vector<uint64_t>& buffer_sizes,
    const std::optional<std::string>& tensor_index_data,
    std::string_view encoding,
    std::string_view schema_version,
    uint32_t max_concurrency,
    const std::optional<std::string>& verification_json,
    std::optional<std::string_view> view_id,
    const std::optional<common::v1::ArtifactDescriptor>& descriptor) {
  // NOTE: This implementation relies on proto/global_store.proto support for
  // memory replicas with tensor index key. If the server does not support the
  // new fields it will still accept the request but ignore extra data.

  global_store::RegisterReplicaRequest request;
  request.set_artifact_id(std::string(artifact_id));
  request.set_worker_id(std::string(worker_id));
  request.set_max_concurrency(max_concurrency);

  auto* mem_info = request.mutable_mem_info();
  if (auto fill_st = fill_memory_info(mem_info, device, MemoryLocation::GPU, memory_size, view_id); !fill_st.ok()) {
    return fill_st;
  }
  // If server supports memory replica fields, populate them via extension fields in MemoryInfo
  // For current proto, we include memory-replica metadata by overloading fields when available via
  // Global Store server. As a fallback, embed keys in the request's optional fields.

  // Mark as in-memory replica and attach tensor index key/metadata
  mem_info->set_is_memory_replica(true);
  if (!tensor_index_key.empty()) {
    mem_info->set_tensor_index_key(std::string(tensor_index_key));
  }
  if (verification_json.has_value() && !verification_json->empty()) {
    mem_info->set_verification_json(*verification_json);
  }
  // Best-effort creation metadata
  {
    const auto now = std::chrono::system_clock::now();
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    auto* ts = mem_info->mutable_creation_ts();
    ts->set_seconds(static_cast<int64_t>(secs));
    ts->set_nanos(0);
  }
  mem_info->set_source_process_id(std::to_string(getpid()));
  if (descriptor.has_value()) {
    *request.mutable_descriptor_() = *descriptor;
  }

  for (const auto& key : remote_memory_keys) {
    mem_info->add_remote_memory_keys(key);
  }
  for (const auto& sz : buffer_sizes) {
    mem_info->add_buffer_sizes(sz);
  }

  // Attach remote memory keys and buffer sizes if present
  // The header signature includes vectors; however, to maintain backward
  // compatibility with existing callers we do not require them.
  // Callers using enable_remote_replica_access should populate these via
  // the Replica/CommunicationManager before calling this method.
  // NOTE: We cannot reference parameters by name due to preexisting signature –
  // adjust after refactor if needed.

  // Optionally include canonical index data for UPSERT on first write.
  if (tensor_index_data.has_value() && !tensor_index_data->empty()) {
    request.set_tensor_index_data(*tensor_index_data);
    if (!encoding.empty()) {
      request.set_encoding(std::string(encoding));
    }
    if (!schema_version.empty()) {
      request.set_schema_version(std::string(schema_version));
    }
  }
  // Note: descriptor attachment is handled at server-side using the parsed mi2 artifact_id
  // Optionally include descriptor when available to allow GS to upsert the artifact registry directly.
  // Server will parse mi2 artifact_id and upsert the artifacts table as needed.

  global_store::RegisterReplicaResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RegisterReplica(ctx, req, resp); },
      "RegisterReplica(memory)");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat(
            "RegisterMemoryReplica failed: %s (%d)",
            status_to_cstr(response.status()),
            static_cast<int>(response.status())));
  }

  // Enrich log with plan type and basic context. We infer plan from device type:
  // - CPU device ⇒ UMA single ledger
  // - GPU device ⇒ VRAM_COALESCED (including materialized Lease)
  const char* plan_str = (device.type == DeviceType::CPU) ? "uma_single_ledger" : "vram_coalesced";
  const char* dev_kind = (device.type == DeviceType::CPU) ? "cpu" : "gpu";
  LOG(INFO) << "Registered memory replica: " << artifact_id << " plan=" << plan_str << " device=" << dev_kind << ":"
            << device.ordinal;
  LOG(INFO) << "RegisterReplica(memory) request mem_info: \n" << request.mem_info().DebugString();
  return response.replica_id();
}

absl::Status GlobalStoreClient::unregister_replica(std::string_view artifact_id, std::string_view replica_id) {
  global_store::UnregisterReplicaRequest request;
  request.set_artifact_id(std::string(artifact_id));
  request.set_replica_id(std::string(replica_id));

  global_store::UnregisterReplicaResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->UnregisterReplica(ctx, req, resp); },
      "UnregisterReplica");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(absl::StrFormat("UnregisterReplica failed with status: %d", response.status()));
  }

  return absl::OkStatus();
}

absl::Status GlobalStoreClient::unregister_replica_by_worker(
    std::string_view artifact_id,
    std::string_view worker_id,
    std::optional<common::memory::MemoryLocation> memory_type,
    std::optional<uint32_t> device_id) {
  global_store::UnregisterReplicaByWorkerRequest request;
  request.set_artifact_id(std::string(artifact_id));
  request.set_worker_id(std::string(worker_id));
  if (memory_type.has_value()) {
    request.set_memory_type(convert_to_proto_memory_type(*memory_type));
  }
  if (device_id.has_value()) {
    request.set_device_id(*device_id);
  }

  global_store::UnregisterReplicaByWorkerResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->UnregisterReplicaByWorker(ctx, req, resp); },
      "UnregisterReplicaByWorker");
  if (!status.ok()) {
    return status;
  }
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(absl::StrFormat("UnregisterReplicaByWorker failed with status: %d", response.status()));
  }
  return absl::OkStatus();
}

absl::StatusOr<TransportSession> GlobalStoreClient::request_replica_transport(
    std::string_view artifact_id,
    std::string_view source_node_id,
    std::string_view source_address,
    uint32_t source_port,
    const DeviceKey& target_device,
    uint32_t wait_timeout_ms) {
  global_store::RequestReplicaTransportRequest request;
  request.set_artifact_id(std::string(artifact_id));
  request.set_source_node_id(std::string(source_node_id));
  request.set_source_address(std::string(source_address));
  request.set_source_port(source_port);
  request.mutable_requested_byte_space()->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  // Use standard duration
  auto* dur = request.mutable_wait_timeout_dur();
  dur->set_seconds(static_cast<int64_t>(wait_timeout_ms / 1000));
  dur->set_nanos(static_cast<int32_t>((wait_timeout_ms % 1000) * 1000000));

  auto* local_mem_info = request.mutable_local_memory_info();
  if (auto fill_st = fill_memory_info(local_mem_info, target_device, MemoryLocation::GPU, 0); !fill_st.ok()) {
    return fill_st;
  }

  global_store::RequestReplicaTransportResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RequestReplicaTransport(ctx, req, resp); },
      "RequestReplicaTransport");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    if (response.status() == global_store::STATUS_NOT_FOUND) {
      return absl::NotFoundError(absl::StrFormat("No available replicas for replica: %s", artifact_id));
    }
    return absl::InternalError(
        absl::StrFormat(
            "RequestReplicaTransport failed: %s (%d)",
            status_to_cstr(response.status()),
            static_cast<int>(response.status())));
  }

  TransportSession session;
  session.transport_id = response.transport_id();
  session.remote_replica = convert_from_proto_memory_info(response.remote_memory_info());
  session.start_time = absl::Now();

  LOG(INFO) << "Started P2P transport " << session.transport_id << " from " << session.remote_replica.node_id;

  return session;
}

absl::StatusOr<TransportSession> GlobalStoreClient::request_view_transport(
    std::string_view artifact_id,
    std::string_view view_id,
    std::string_view source_node_id,
    std::string_view source_address,
    uint32_t source_port,
    const DeviceKey& target_device,
    uint32_t wait_timeout_ms) {
  if (view_id.empty()) {
    return absl::InvalidArgumentError("view_id must be non-empty for view transport");
  }

  global_store::RequestReplicaTransportRequest request;
  request.set_artifact_id(std::string(artifact_id));
  request.set_source_node_id(std::string(source_node_id));
  request.set_source_address(std::string(source_address));
  request.set_source_port(source_port);
  auto* dur = request.mutable_wait_timeout_dur();
  dur->set_seconds(static_cast<int64_t>(wait_timeout_ms / 1000));
  dur->set_nanos(static_cast<int32_t>((wait_timeout_ms % 1000) * 1000000));

  auto* local_mem_info = request.mutable_local_memory_info();
  if (auto fill_st = fill_memory_info(local_mem_info, target_device, MemoryLocation::GPU, 0); !fill_st.ok()) {
    return fill_st;
  }

  auto* requested_space = request.mutable_requested_byte_space();
  requested_space->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
  requested_space->set_id(std::string(view_id));

  global_store::RequestReplicaTransportResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RequestReplicaTransport(ctx, req, resp); },
      "RequestReplicaTransport(view)");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    if (response.status() == global_store::STATUS_NOT_FOUND) {
      return absl::NotFoundError(
          absl::StrFormat("No available replicas for view transport: %s view_id=%s", artifact_id, view_id));
    }
    return absl::InternalError(
        absl::StrFormat(
            "RequestReplicaTransport(view) failed: %s (%d)",
            status_to_cstr(response.status()),
            static_cast<int>(response.status())));
  }

  TransportSession session;
  session.transport_id = response.transport_id();
  session.remote_replica = convert_from_proto_memory_info(response.remote_memory_info());
  session.start_time = absl::Now();

  LOG(INFO) << "Started view transport " << session.transport_id << " from " << session.remote_replica.node_id
            << " view_id=" << view_id;

  return session;
}

absl::Status GlobalStoreClient::complete_replica_transport(std::string_view transport_id) {
  global_store::CompleteReplicaTransportRequest request;
  request.set_transport_id(std::string(transport_id));

  global_store::CompleteReplicaTransportResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->CompleteReplicaTransport(ctx, req, resp); },
      "CompleteReplicaTransport");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(absl::StrFormat("CompleteReplicaTransport failed with status: %d", response.status()));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<RemoteReplicaInfo>> GlobalStoreClient::get_artifact_replicas(
    std::string_view artifact_id,
    std::optional<std::string_view> view_id) {
  global_store::GetArtifactInfoByIdRequest request;
  request.set_artifact_id(std::string(artifact_id));
  auto* byte_space = request.mutable_requested_byte_space();
  if (view_id.has_value()) {
    byte_space->set_kind(common::v1::BYTE_SPACE_KIND_VIEW);
    byte_space->set_id(std::string(*view_id));
  } else {
    byte_space->set_kind(common::v1::BYTE_SPACE_KIND_CANONICAL);
  }

  global_store::GetArtifactInfoByIdResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->GetArtifactInfoById(ctx, req, resp); },
      "GetArtifactInfoById");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    if (response.status() == global_store::STATUS_NOT_FOUND) {
      return absl::NotFoundError(absl::StrFormat("Artifact not found: %s", artifact_id));
    }
    return absl::InternalError(
        absl::StrFormat(
            "GetArtifactInfoById failed: %s (%d)",
            status_to_cstr(response.status()),
            static_cast<int>(response.status())));
  }

  std::vector<RemoteReplicaInfo> replicas;
  for (const auto& mem_info : response.replicas()) {
    replicas.push_back(convert_from_proto_memory_info(mem_info));
  }

  return replicas;
}

bool GlobalStoreClient::is_connected() const {
  return channel_->GetState(false) == GRPC_CHANNEL_READY;
}

absl::Status GlobalStoreClient::batch_update_chunk_states(
    std::string_view worker_id,
    std::string_view node_id,
    const std::vector<ChunkStateUpdate>& updates) {
  if (updates.empty())
    return absl::OkStatus();
  global_store::BatchUpdateChunkStatesRequest req;
  req.set_worker_id(std::string(worker_id));
  req.set_node_id(std::string(node_id));
  for (const auto& u : updates) {
    auto* up = req.add_updates();
    up->set_artifact_id(u.artifact_id);
    up->set_chunk_idx(u.chunk_idx);
    switch (u.state) {
      case ChunkState::HOT:
        up->set_state(global_store::CHUNK_STATE_HOT);
        break;
      case ChunkState::LOCKED_TX:
        up->set_state(global_store::CHUNK_STATE_LOCKED_TX);
        break;
      case ChunkState::COPIED_GPU:
        up->set_state(global_store::CHUNK_STATE_COPIED_GPU);
        break;
      case ChunkState::EVICTED:
        up->set_state(global_store::CHUNK_STATE_EVICTED);
        break;
      case ChunkState::COLD:
      case ChunkState::PREEMPTIBLE:
      default:
        up->set_state(global_store::CHUNK_STATE_COLD);
        break;
    }
    up->set_device_uuid(u.device_uuid);
    up->set_replica(u.replica);
  }
  global_store::BatchUpdateChunkStatesResponse resp;
  auto st = execute_rpc_with_retry(
      req,
      &resp,
      [this](auto* ctx, const auto& r, auto* o) { return stub_->BatchUpdateChunkStates(ctx, r, o); },
      "BatchUpdateChunkStates");
  if (!st.ok())
    return st;
  if (resp.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat(
            "BatchUpdateChunkStates failed: %s (%d)", status_to_cstr(resp.status()), static_cast<int>(resp.status())));
  }
  return absl::OkStatus();
}

// Static conversion methods
tensorcast::common::v1::MemoryType GlobalStoreClient::convert_to_proto_memory_type(MemoryLocation location) {
  switch (location) {
    case MemoryLocation::GPU:
      return tensorcast::common::v1::MEMORY_TYPE_GPU;
    case MemoryLocation::CPU:
      return tensorcast::common::v1::MEMORY_TYPE_RAM;
    case MemoryLocation::DISK:
      return tensorcast::common::v1::MEMORY_TYPE_DISK;
    case MemoryLocation::REMOTE:
      return tensorcast::common::v1::MEMORY_TYPE_DISK; // Fallback mapping for REMOTE
    default:
      return tensorcast::common::v1::MEMORY_TYPE_DISK;
  }
}

MemoryLocation GlobalStoreClient::convert_from_proto_memory_type(tensorcast::common::v1::MemoryType type) {
  switch (type) {
    case tensorcast::common::v1::MEMORY_TYPE_GPU:
      return MemoryLocation::GPU;
    case tensorcast::common::v1::MEMORY_TYPE_RAM:
      return MemoryLocation::CPU;
    case tensorcast::common::v1::MEMORY_TYPE_DISK:
      return MemoryLocation::DISK;
    default:
      return MemoryLocation::DISK;
  }
}

absl::Status GlobalStoreClient::fill_memory_info(
    tensorcast::common::v1::MemoryInfo* info,
    const DeviceKey& device,
    MemoryLocation location,
    uint64_t memory_size,
    std::optional<std::string_view> view_id) {
  std::string node_id;
  std::string node_address;
  uint32_t node_port = 0;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    node_id = node_id_;
    node_address = node_address_;
    node_port = p2p_port_;
  }

  if (node_id.empty()) {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
      node_id = hostname;
    } else {
      node_id = "unknown";
    }
  }

  if (node_address.empty() || is_loopback_or_unspecified(node_address)) {
    const std::string default_ip = communicator::misc::get_default_ip();
    if (!default_ip.empty()) {
      node_address = default_ip;
    }
  }

  if (node_address.empty() || is_loopback_or_unspecified(node_address)) {
    return absl::FailedPreconditionError(
        "GlobalStoreClient requires a routable advertised address before exporting memory info.");
  }

  if (node_port == 0) {
    return absl::FailedPreconditionError(
        "GlobalStoreClient requires a non-zero P2P port before exporting memory info; configure --p2p_listen or pass a valid p2p_port to register_worker().");
  }

  info->set_node_id(node_id);
  info->set_node_address(node_address);
  info->set_node_port(node_port);
  info->set_memory_size(memory_size);
  info->set_memory_type(convert_to_proto_memory_type(location));

  if (device.type == DeviceType::GPU) {
    info->set_device_id(device.ordinal);
  }

  auto* byte_space = info->mutable_byte_space();
  if (view_id.has_value() && !view_id->empty()) {
    byte_space->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
    byte_space->set_id(std::string(*view_id));
  } else {
    byte_space->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  }

  return absl::OkStatus();
}

RemoteReplicaInfo GlobalStoreClient::convert_from_proto_memory_info(const tensorcast::common::v1::MemoryInfo& info) {
  RemoteReplicaInfo replica;
  replica.node_id = info.node_id();
  replica.node_address = info.node_address();
  replica.node_port = info.node_port();
  replica.memory_size = info.memory_size();
  replica.memory_type = convert_from_proto_memory_type(info.memory_type());
  replica.device_id = info.device_id();

  for (const auto& key : info.remote_memory_keys()) {
    replica.remote_memory_keys.push_back(key);
  }

  for (const auto& size : info.buffer_sizes()) {
    replica.buffer_sizes.push_back(size);
  }
  if (!info.verification_json().empty()) {
    replica.verification_json = info.verification_json();
  }
  if (info.has_byte_space() && info.byte_space().kind() == tensorcast::common::v1::BYTE_SPACE_KIND_VIEW) {
    if (!info.byte_space().id().empty()) {
      replica.view_id = info.byte_space().id();
    }
  }

  return replica;
}

void GlobalStoreClient::update_local_endpoint(
    std::string node_id,
    std::string node_address,
    uint32_t grpc_port,
    uint32_t p2p_port) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!node_id.empty()) {
    node_id_ = std::move(node_id);
  }
  if (!node_address.empty()) {
    node_address_ = std::move(node_address);
  }
  if (grpc_port != 0) {
    grpc_port_ = grpc_port;
  }
  if (p2p_port != 0) {
    p2p_port_ = p2p_port;
  }
}

template <typename Request, typename Response, typename RpcMethod>
absl::Status GlobalStoreClient::execute_rpc_with_retry(
    const Request& request,
    Response* response,
    RpcMethod method,
    const std::string& method_name,
    const RpcOptions& rpc_options) {
  const absl::Duration timeout = rpc_options.timeout.value_or(config_.rpc_timeout);
  const uint32_t max_retries = rpc_options.max_retries.value_or(config_.max_retries);
  const absl::Duration retry_backoff = rpc_options.retry_backoff.value_or(config_.retry_backoff);
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  for (uint32_t attempt = 0; attempt <= max_retries; ++attempt) {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(absl::ToInt64Seconds(timeout)));
    if (!config_.cluster_token.empty()) {
      context.AddMetadata("x-tensorcast-cluster-token", config_.cluster_token);
    }

    // OpenTelemetry: create client span and inject W3C Trace Context.
    namespace otel = opentelemetry;
    auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
    otel::trace::StartSpanOptions opts;
    opts.kind = otel::trace::SpanKind::kClient;
    auto span = tracer->StartSpan(absl::StrFormat("GlobalStore/%s", method_name), opts);
    otel::trace::Scope scope(span);
    span->SetAttribute("rpc.system", "grpc");
    span->SetAttribute("rpc.service", "tensorcast.global_store.GlobalStoreService");
    span->SetAttribute("rpc.method", method_name);
    tensorcast::common::otel::inject_into_client_metadata(context);

    auto status = method(&context, request, response);
    if (!status.ok()) {
      span->SetAttribute("rpc.grpc.status_code", static_cast<int64_t>(status.error_code()));
      span->SetAttribute("rpc.grpc.message", status.error_message());
    }

    if (status.ok()) {
      return absl::OkStatus();
    }

    const auto code = status.error_code();
    switch (code) {
      case grpc::StatusCode::FAILED_PRECONDITION:
        return absl::FailedPreconditionError(status.error_message());
      case grpc::StatusCode::INVALID_ARGUMENT:
        return absl::InvalidArgumentError(status.error_message());
      case grpc::StatusCode::NOT_FOUND:
        return absl::NotFoundError(status.error_message());
      case grpc::StatusCode::ALREADY_EXISTS:
        return absl::AlreadyExistsError(status.error_message());
      case grpc::StatusCode::PERMISSION_DENIED:
        return absl::PermissionDeniedError(status.error_message());
      case grpc::StatusCode::UNAUTHENTICATED:
        return absl::UnauthenticatedError(status.error_message());
      case grpc::StatusCode::RESOURCE_EXHAUSTED:
        return absl::ResourceExhaustedError(status.error_message());
      case grpc::StatusCode::ABORTED:
        return absl::AbortedError(status.error_message());
      case grpc::StatusCode::OUT_OF_RANGE:
        return absl::OutOfRangeError(status.error_message());
      case grpc::StatusCode::UNIMPLEMENTED:
        return absl::UnimplementedError(status.error_message());
      case grpc::StatusCode::DATA_LOSS:
        return absl::DataLossError(status.error_message());
      case grpc::StatusCode::CANCELLED:
        return absl::CancelledError(status.error_message());
      default:
        break;
    }

    if (attempt < max_retries) {
      auto base = retry_backoff * (1 << attempt);
      // Jitter within +/- 50%
      double jitter = std::uniform_real_distribution<double>(0.5, 1.5)(rng);
      auto jittered = absl::Milliseconds(static_cast<int64_t>(absl::ToInt64Milliseconds(base) * jitter));
      LOG(WARNING) << "RPC " << method_name << " failed (attempt " << attempt + 1 << "/" << max_retries + 1
                   << "): " << status.error_message() << ". Retrying in " << absl::ToInt64Milliseconds(jittered)
                   << "ms";
      std::this_thread::sleep_for(std::chrono::milliseconds(absl::ToInt64Milliseconds(jittered)));
    }
  }

  return absl::UnavailableError(absl::StrFormat("RPC %s failed after %d retries", method_name, max_retries + 1));
}

absl::StatusOr<std::vector<ChunkLocationInfo>> GlobalStoreClient::query_chunk_locations(
    std::string_view artifact_id,
    const std::vector<uint32_t>& chunk_indices) {
  global_store::QueryChunkLocationsRequest request;
  request.set_artifact_id(std::string(artifact_id));
  for (uint32_t idx : chunk_indices) {
    request.add_chunk_indices(idx);
  }

  global_store::QueryChunkLocationsResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->QueryChunkLocations(ctx, req, resp); },
      "QueryChunkLocations");

  if (!status.ok()) {
    return status;
  }

  if (response.status() != global_store::STATUS_OK) {
    if (response.status() == global_store::STATUS_NOT_FOUND) {
      return absl::NotFoundError(absl::StrFormat("Artifact not found: %s", artifact_id));
    }
    return absl::InternalError(absl::StrFormat("QueryChunkLocations failed with status: %d", response.status()));
  }

  std::vector<ChunkLocationInfo> locations;
  locations.reserve(response.locations_size());

  for (const auto& loc : response.locations()) {
    ChunkLocationInfo info;
    info.chunk_idx = loc.chunk_idx();
    info.node_id = loc.node_id();
    info.node_address = loc.node_address();
    info.p2p_port = loc.p2p_port();

    // Convert proto ChunkState to our ChunkState
    switch (loc.state()) {
      case global_store::CHUNK_STATE_HOT:
        info.state = ChunkState::HOT;
        break;
      case global_store::CHUNK_STATE_LOCKED_TX:
        info.state = ChunkState::LOCKED_TX;
        break;
      case global_store::CHUNK_STATE_COPIED_GPU:
        info.state = ChunkState::COPIED_GPU;
        break;
      case global_store::CHUNK_STATE_COLD:
        info.state = ChunkState::COLD;
        break;
      case global_store::CHUNK_STATE_EVICTED:
        info.state = ChunkState::EVICTED;
        break;
      default:
        info.state = ChunkState::EVICTED; // Safe default
        break;
    }

    info.node_load_ratio = loc.node_load_ratio();
    info.device_uuid = loc.device_uuid();
    info.replica = loc.replica();

    locations.push_back(std::move(info));
  }

  return locations;
}

absl::StatusOr<StateSyncResult> GlobalStoreClient::synchronize_worker_state(
    const global_store::WorkerLocalState& local_state,
    bool force_full_sync,
    const StateSyncToken& token,
    const RpcOptions& rpc_options) {
  global_store::SynchronizeWorkerStateRequest request;
  request.set_worker_id(local_state.worker_id());
  *request.mutable_local_state() = local_state;
  request.set_force_full_sync(force_full_sync);
  request.set_sync_epoch(token.epoch);
  request.set_sync_request_id(token.request_id);

  global_store::SynchronizeWorkerStateResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->SynchronizeWorkerState(ctx, req, resp); },
      "SynchronizeWorkerState",
      rpc_options);
  if (!status.ok())
    return status;
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat(
            "SynchronizeWorkerState failed: %s (%d)",
            status_to_cstr(response.status()),
            static_cast<int>(response.status())));
  }
  StateSyncResult result;
  result.new_state_version = response.new_state_version();
  result.new_state_checksum = response.new_state_checksum();
  result.ignored = response.ignored();
  result.state_changes.reserve(response.state_changes_size());
  for (const auto& ch : response.state_changes()) {
    result.state_changes.push_back(ch);
  }
  return result;
}

absl::StatusOr<FullStateSyncResult> GlobalStoreClient::request_full_state_sync(
    std::string_view worker_id,
    uint64_t current_state_version,
    const StateSyncToken& token,
    const RpcOptions& rpc_options) {
  global_store::RequestFullStateSyncRequest request;
  request.set_worker_id(std::string(worker_id));
  request.set_current_state_version(current_state_version);
  request.set_sync_epoch(token.epoch);
  request.set_sync_request_id(token.request_id);

  global_store::RequestFullStateSyncResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RequestFullStateSync(ctx, req, resp); },
      "RequestFullStateSync",
      rpc_options);
  if (!status.ok())
    return status;
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat(
            "RequestFullStateSync failed: %s (%d)",
            status_to_cstr(response.status()),
            static_cast<int>(response.status())));
  }
  FullStateSyncResult result;
  result.new_state_version = response.new_state_version();
  result.new_state_checksum = response.new_state_checksum();
  result.ignored = response.ignored();
  result.expected_replicas.reserve(response.expected_replicas_size());
  for (const auto& rep : response.expected_replicas()) {
    result.expected_replicas.push_back(rep);
  }
  return result;
}

// ========== Key Mapping ==========

absl::StatusOr<KeyMapping> GlobalStoreClient::resolve_key_mapping(std::string_view key) {
  global_store::ResolveKeyMappingRequest request;
  request.set_key(std::string(key));

  global_store::ResolveKeyMappingResponse response;

  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->ResolveKeyMapping(ctx, req, resp); },
      "ResolveKeyMapping");

  if (!status.ok()) {
    return status;
  }
  if (response.status() != global_store::STATUS_OK) {
    return absl::NotFoundError("key not found");
  }

  KeyMapping out{
      .artifact_id = response.artifact_id(),
      .replica_uuid = response.replica_uuid(),
      .daemon_address = response.daemon_address(),
      .disk_path = response.disk_path(),
  };
  return out;
}

absl::Status GlobalStoreClient::upsert_key_mapping(
    std::string_view key,
    std::string_view artifact_id,
    std::string_view disk_path,
    absl::Duration ttl) {
  global_store::UpsertKeyMappingRequest request;
  request.set_key(std::string(key));
  request.set_artifact_id(std::string(artifact_id));
  request.set_disk_path(std::string(disk_path));
  if (ttl > absl::ZeroDuration()) {
    auto* d = request.mutable_ttl();
    d->set_seconds(absl::ToInt64Seconds(ttl));
    d->set_nanos(0);
  }

  global_store::UpsertKeyMappingResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->UpsertKeyMapping(ctx, req, resp); },
      "UpsertKeyMapping");
  if (!status.ok())
    return status;
  if (response.status() != global_store::STATUS_OK) {
    return absl::AlreadyExistsError("key mapping conflict or error");
  }
  return absl::OkStatus();
}

absl::Status GlobalStoreClient::revoke_key_mapping(std::string_view key) {
  global_store::RevokeKeyMappingRequest request;
  request.set_key(std::string(key));

  global_store::RevokeKeyMappingResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->RevokeKeyMapping(ctx, req, resp); },
      "RevokeKeyMapping");
  if (!status.ok())
    return status;
  if (response.status() != global_store::STATUS_OK) {
    return absl::NotFoundError("key not found");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> GlobalStoreClient::get_artifact_index_by_id(std::string_view artifact_id) {
  global_store::GetArtifactIndexByIdRequest request;
  request.set_artifact_id(std::string(artifact_id));

  global_store::GetArtifactIndexByIdResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->GetArtifactIndexById(ctx, req, resp); },
      "GetArtifactIndexById");
  if (!status.ok())
    return status;
  if (response.status() != global_store::STATUS_OK) {
    return absl::NotFoundError("artifact index not found");
  }

  const std::string schema_version = response.schema_version();
  if (schema_version.empty()) {
    return absl::FailedPreconditionError(
        "Global Store returned canonical index without schema_version; schema_version='v3' is required");
  }
  if (schema_version != "v3") {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported canonical index schema_version='", schema_version, "'; expected 'v3'"));
  }

  // In proto3, singular bytes fields do not have presence; check emptiness instead.
  if (response.tensor_index_data().empty()) {
    return absl::NotFoundError("artifact index not found");
  }
  return response.tensor_index_data();
}

absl::StatusOr<ViewMetadata> GlobalStoreClient::get_view_metadata(
    std::string_view artifact_id,
    std::string_view view_id) {
  if (artifact_id.empty() || view_id.empty()) {
    return absl::InvalidArgumentError("get_view_metadata requires artifact_id and view_id");
  }
  global_store::GetArtifactInfoByIdRequest request;
  request.set_artifact_id(std::string(artifact_id));
  request.mutable_requested_byte_space()->set_kind(common::v1::BYTE_SPACE_KIND_VIEW);
  request.mutable_requested_byte_space()->set_id(std::string(view_id));
  request.set_include_view_meta(true);

  global_store::GetArtifactInfoByIdResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->GetArtifactInfoById(ctx, req, resp); },
      "GetArtifactInfoById");
  if (!status.ok()) {
    return status;
  }
  if (response.status() != global_store::STATUS_OK) {
    if (response.status() == global_store::STATUS_NOT_FOUND) {
      return absl::NotFoundError(
          absl::StrFormat("View metadata not found for artifact_id=%s view_id=%s", artifact_id, view_id));
    }
    return absl::InternalError(
        absl::StrFormat(
            "GetArtifactInfoById failed: %s (%d)",
            status_to_cstr(response.status()),
            static_cast<int>(response.status())));
  }
  if (!response.has_view_meta()) {
    return absl::NotFoundError(
        absl::StrFormat("View metadata missing for artifact_id=%s view_id=%s", artifact_id, view_id));
  }
  const auto& meta = response.view_meta();
  if (meta.view_spec_json().empty()) {
    return absl::NotFoundError(
        absl::StrFormat("view_spec_json missing for artifact_id=%s view_id=%s", artifact_id, view_id));
  }
  ViewMetadata result;
  result.view_spec_json = meta.view_spec_json();
  result.view_size_bytes = meta.view_size();
  if (!meta.view_data_hash().empty()) {
    result.view_data_hash = meta.view_data_hash();
  }
  return result;
}

absl::StatusOr<PlacementPlanResult> GlobalStoreClient::plan_placement(
    std::string_view artifact_id,
    global_store::PlacementPolicy policy,
    const std::vector<PlacementShardSpec>& shards,
    std::string_view source_node_id) {
  global_store::PlanPlacementRequest request;
  request.set_artifact_id(std::string(artifact_id));
  request.set_placement_policy(policy);
  request.set_source_node_id(std::string(source_node_id));
  for (const auto& shard : shards) {
    auto* out = request.add_shards();
    out->set_shard_id(shard.shard_id);
    out->set_shard_idx(shard.shard_idx);
    out->set_size_bytes(shard.size_bytes);
    out->set_content_digest(shard.content_digest);
    out->set_byte_range_start(shard.byte_range_start);
    out->set_byte_range_length(shard.byte_range_length);
    out->mutable_chunk_ids()->Add(shard.chunk_ids.begin(), shard.chunk_ids.end());
  }

  global_store::PlanPlacementResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->PlanPlacement(ctx, req, resp); },
      "PlanPlacement");
  if (!status.ok()) {
    return status;
  }
  if (response.plan_id().empty()) {
    return absl::InternalError("PlanPlacement returned empty plan_id");
  }

  PlacementPlanResult result;
  result.plan_id = response.plan_id();
  result.effective_policy = response.effective_policy();
  result.degraded = response.degraded();
  result.degraded_reason = response.degraded_reason();
  result.placements.reserve(response.placements_size());

  for (const auto& placement_proto : response.placements()) {
    PlacementShardSpec shard{
        .shard_id = placement_proto.shard().shard_id(),
        .shard_idx = placement_proto.shard().shard_idx(),
        .size_bytes = placement_proto.shard().size_bytes(),
        .content_digest = placement_proto.shard().content_digest(),
        .byte_range_start = placement_proto.shard().byte_range_start(),
        .byte_range_length = placement_proto.shard().byte_range_length(),
        .chunk_ids = {placement_proto.shard().chunk_ids().begin(), placement_proto.shard().chunk_ids().end()}};
    ShardPlacement placement;
    placement.shard = std::move(shard);
    placement.degraded_reason = placement_proto.degraded_reason();
    placement.targets.reserve(placement_proto.targets_size());
    for (const auto& target_proto : placement_proto.targets()) {
      PlacementTargetStatus target;
      target.node_id = target_proto.node_id();
      target.lease_id = target_proto.lease_id();
      target.target_state = target_proto.target_state();
      target.degraded_reason = target_proto.degraded_reason();
      placement.targets.push_back(std::move(target));
    }
    result.placements.push_back(std::move(placement));
  }
  return result;
}

absl::Status GlobalStoreClient::report_persistence_status(const PersistenceReport& report) {
  global_store::ReportPersistenceStatusRequest request;
  request.set_task_id(report.task_id);
  request.set_artifact_id(report.artifact_id);
  request.set_plan_id(report.plan_id);
  request.set_state(report.state);
  request.set_progress(report.progress);
  if (!report.last_error.empty()) {
    request.set_last_error(report.last_error);
  }
  if (!report.degraded_reason.empty()) {
    request.set_degraded_reason(report.degraded_reason);
  }
  for (const auto& shard : report.shards) {
    auto* shard_proto = request.add_shard_statuses();
    shard_proto->set_shard_id(shard.shard_id);
    shard_proto->set_shard_idx(shard.shard_idx);
    shard_proto->set_state(shard.state);
    shard_proto->set_progress(shard.progress);
    if (!shard.degraded_reason.empty()) {
      shard_proto->set_degraded_reason(shard.degraded_reason);
    }
    if (!shard.last_error.empty()) {
      shard_proto->set_last_error(shard.last_error);
    }
    for (const auto& target : shard.targets) {
      auto* target_proto = shard_proto->add_targets();
      target_proto->set_node_id(target.node_id);
      if (!target.lease_id.empty()) {
        target_proto->set_lease_id(target.lease_id);
      }
      target_proto->set_target_state(target.target_state);
      if (!target.degraded_reason.empty()) {
        target_proto->set_degraded_reason(target.degraded_reason);
      }
    }
  }

  global_store::ReportPersistenceStatusResponse response;
  auto status = execute_rpc_with_retry(
      request,
      &response,
      [this](auto* ctx, const auto& req, auto* resp) { return stub_->ReportPersistenceStatus(ctx, req, resp); },
      "ReportPersistenceStatus");
  if (!status.ok()) {
    return status;
  }
  if (response.status() != global_store::STATUS_OK) {
    return absl::InternalError(
        absl::StrFormat("ReportPersistenceStatus failed: status=%s", status_to_cstr(response.status())));
  }
  return absl::OkStatus();
}

} // namespace tensorcast::store::components
