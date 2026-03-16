// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/control/materialize_orchestrator.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/store/components/endpoint_id.h"
#include "core/store/components/global_store_client.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/provider.h"

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
constexpr int kMaxReselectionAttemptsWithoutBudget = 64;
constexpr std::chrono::milliseconds kMinReselectionBudget{1};
constexpr std::chrono::milliseconds kMinReselectionBackoff{50};

struct SourceReselectionMetrics {
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> meter;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> stale_source_detected_total;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> reselection_attempt_total;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> reselection_success_total;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> reselection_exhausted_total;
};

SourceReselectionMetrics& source_reselection_metrics() {
  static SourceReselectionMetrics metrics;
  if (!metrics.meter) {
    metrics.meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  }
  return metrics;
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
        if (has_disk_source && allow_disk && preference != loading::SourcePreference::kPreferP2P) {
          LOG(INFO) << "request_view_transport unavailable for artifact_id=" << artifact_id << " view_id=" << *view_id
                    << " within probe_timeout_ms=" << view_probe_timeout_ms
                    << "; bypassing canonical transport route and falling back to disk";
          return absl::AbortedError(
              absl::StrCat("view transport unavailable for artifact_id=", artifact_id, "; disk fallback available"));
        }
        LOG(INFO) << "request_view_transport unavailable for artifact_id=" << artifact_id << " view_id=" << *view_id
                  << " within probe_timeout_ms=" << view_probe_timeout_ms
                  << "; retrying canonical transport route with wait_timeout_ms=" << wait_timeout_ms;
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
    const auto& session = *transport_or;
    const auto& remote = session.remote_replica;

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
    p2p_src.local_endpoint_id = components::derive_endpoint_id(local_identity_, target_device);
    p2p_src.remote_endpoint_id = remote.endpoint_id;
    p2p_src.memory_keys = remote.remote_memory_keys;
    p2p_src.buf_sizes = remote.buffer_sizes;
    p2p_src.verification_json = remote.verification_json;
    p2p_src.enable_checksum = true;
    p2p_src.location.type = remote.memory_type;
    p2p_src.location.device_id = remote.device_id;
    p2p_src.request_budget = hints.request_budget;
    p2p_src.artifact_id = std::string(artifact_id);
    backend_->prepare_p2p_source(&p2p_src);
    if (has_disk_source && allow_disk && preference != loading::SourcePreference::kPreferP2P) {
      p2p_src.fallback_disk_dir = disk_source->path.string();
    }

    // Build target description
    ReplicaTarget target;
    target.location.type = (target_device.type == DeviceType::GPU) ? common::memory::MemoryLocation::GPU
                                                                   : common::memory::MemoryLocation::CPU;
    target.location.device_id = target_device.ordinal;

    auto load_or = backend_->ingest_from_p2p(std::string(artifact_id), p2p_src, target, hints);
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
        LOG(INFO) << "materialize_view loaded via canonical transport fallback: artifact_id=" << artifact_id
                  << " view_id=" << *view_id;
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
