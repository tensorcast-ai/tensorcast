// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/control/materialize_orchestrator.h"

#include <filesystem>
#include <utility>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/store/components/global_store_client.h"
#include "core/store/materialization/contracts/loading_spec.h"

namespace tensorcast::store::materialization::control {

using tensorcast::store::P2PSource;
using tensorcast::store::components::RemoteReplicaInfo;
using tensorcast::store::components::WorkerIdentity;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::loading::MaterializeHints;
using tensorcast::store::loading::ReplicaHandle;
using tensorcast::store::loading::ReplicaTarget;

namespace {

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
  if (gs_connected && allow_p2p) {
    if (view_id.has_value()) {
      transport_or = gs_client_->request_view_transport(
          artifact_id,
          *view_id,
          local_identity_.node_id,
          local_identity_.node_address,
          local_identity_.p2p_port,
          target_device,
          /*wait_timeout_ms=*/30000);
      if (!transport_or.ok() &&
          (absl::IsNotFound(transport_or.status()) || absl::IsUnimplemented(transport_or.status()))) {
        view_transport_status = transport_or.status();
        LOG(INFO) << "request_view_transport unavailable for artifact_id=" << artifact_id << " view_id=" << *view_id
                  << "; retrying canonical transport route";
        transport_or = gs_client_->request_replica_transport(
            artifact_id,
            local_identity_.node_id,
            local_identity_.node_address,
            local_identity_.p2p_port,
            target_device,
            /*wait_timeout_ms=*/30000);
        used_canonical_transport_fallback = true;
      }
    } else {
      transport_or = gs_client_->request_replica_transport(
          artifact_id,
          local_identity_.node_id,
          local_identity_.node_address,
          local_identity_.p2p_port,
          target_device,
          /*wait_timeout_ms=*/30000);
    }
  }

  if (transport_or.ok()) {
    const auto& session = *transport_or;
    const auto& remote = session.remote_replica;

    if (is_local_replica(remote, local_identity_)) {
      LOG(WARNING) << "Global Store returned local replica for artifact_id=" << artifact_id
                   << "; treating route as stale";
      absl::Status comp_status = gs_client_->complete_replica_transport(session.transport_id);
      if (!comp_status.ok()) {
        LOG(WARNING) << "complete_replica_transport after stale-local route returned error: " << comp_status;
      }
      last_p2p_status = stale_local_route_status(artifact_id);
      if (!has_disk_source || !allow_disk) {
        return last_p2p_status;
      }
    } else {
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

      if (!has_disk_source || !allow_disk) {
        return load_or.status();
      }
    }

  } else {
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
