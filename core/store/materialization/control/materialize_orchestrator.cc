// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/control/materialize_orchestrator.h"

#include <algorithm>
#include <filesystem>
#include <map>
#include <utility>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
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

constexpr int kMaxStaleSourceReselectionAttempts = 3;

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
  return absl::IsUnavailable(status) || absl::IsNotFound(status) || absl::IsFailedPrecondition(status);
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

  auto request_transport = [&]() -> absl::StatusOr<components::TransportSession> {
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
          /*wait_timeout_ms=*/30000);
      if (!view_transport_or.ok() &&
          (absl::IsNotFound(view_transport_or.status()) || absl::IsUnimplemented(view_transport_or.status()))) {
        view_transport_status = view_transport_or.status();
        LOG(INFO) << "request_view_transport unavailable for artifact_id=" << artifact_id << " view_id=" << *view_id
                  << "; retrying canonical transport route";
        auto canonical_transport_or = gs_client_->request_replica_transport(
            artifact_id,
            local_identity_.node_id,
            local_identity_.node_address,
            local_identity_.p2p_port,
            target_device,
            /*wait_timeout_ms=*/30000);
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
        /*wait_timeout_ms=*/30000);
  };

  while (gs_connected && allow_p2p) {
    transport_or = request_transport();
    had_transport_request = true;
    if (!transport_or.ok()) {
      break;
    }
    const auto& session = *transport_or;
    const auto& remote = session.remote_replica;

    if (is_local_replica(remote, local_identity_)) {
      record_stale_source_detected("local_route", view_id.has_value(), reselection_attempt);
      LOG(WARNING) << "Global Store returned local replica for artifact_id=" << artifact_id
                   << "; treating route as stale";
      absl::Status comp_status = gs_client_->complete_replica_transport(session.transport_id);
      if (!comp_status.ok()) {
        LOG(WARNING) << "complete_replica_transport after stale-local route returned error: " << comp_status;
      }
      last_p2p_status = stale_local_route_status(artifact_id);
      if (reselection_attempt < kMaxStaleSourceReselectionAttempts) {
        reselection_attempt += 1;
        record_source_reselection_attempt("local_route", view_id.has_value(), reselection_attempt);
        LOG(WARNING) << "Retrying source selection after stale local route: artifact_id=" << artifact_id
                     << " attempt=" << reselection_attempt << "/" << kMaxStaleSourceReselectionAttempts;
        continue;
      }
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

    auto load_or = backend_->ingest_from_p2p(std::string(artifact_id), p2p_src, target, hints);
    if (load_or.ok()) {
      // Notify GS that transport finished
      absl::Status comp_status = gs_client_->complete_replica_transport(session.transport_id);
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
    absl::Status comp_status = gs_client_->complete_replica_transport(session.transport_id);
    if (!comp_status.ok()) {
      LOG(WARNING) << "complete_replica_transport after failure returned error: " << comp_status;
    }
    last_p2p_status = load_or.status();
    if (view_id.has_value()) {
      LOG(WARNING) << "P2P load failed for view_id=" << *view_id << ": " << load_or.status();
    } else {
      LOG(WARNING) << "P2P load failed: " << load_or.status();
    }

    if (should_retry_source_selection(last_p2p_status) && reselection_attempt < kMaxStaleSourceReselectionAttempts) {
      record_stale_source_detected("p2p_load_failure", view_id.has_value(), reselection_attempt);
      reselection_attempt += 1;
      record_source_reselection_attempt("p2p_load_failure", view_id.has_value(), reselection_attempt);
      LOG(WARNING) << "Retrying source selection after P2P load failure: artifact_id=" << artifact_id
                   << " attempt=" << reselection_attempt << "/" << kMaxStaleSourceReselectionAttempts
                   << " status=" << last_p2p_status;
      continue;
    }
    if (should_retry_source_selection(last_p2p_status)) {
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
