// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/control/materialize_orchestrator.h"

#include <filesystem>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/store/components/global_store_client.h"
#include "core/store/materialization/contracts/loading_spec.h"

namespace tensorcast::store::materialization::control {

using tensorcast::store::P2PSource;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::loading::MaterializeHints;
using tensorcast::store::loading::ReplicaHandle;
using tensorcast::store::loading::ReplicaTarget;

MaterializeOrchestrator::MaterializeOrchestrator(
    gsl::not_null<MaterializationBackend*> backend,
    gsl::not_null<components::IGlobalStoreClient*> gs_client)
    : backend_(backend), gs_client_(gs_client) {}

absl::StatusOr<ReplicaHandle> MaterializeOrchestrator::run(
    std::string_view artifact_id,
    const DeviceKey& target_device,
    const MaterializeHints& hints) {
  // ------------------------------------------------------------------
  // 1. Preference handling and Global Store connectivity guard
  // ------------------------------------------------------------------
  const bool gs_connected = gs_client_->is_connected();
  const auto preference = hints.source_preference;
  const bool has_disk_path = !hints.disk_path.empty();
  const bool has_artifact_id_hint = !hints.artifact_id.empty();
  if (preference == loading::SourcePreference::kPreferP2P && !has_artifact_id_hint) {
    return absl::InvalidArgumentError("preference=PREFER_P2P requires a canonical artifact_id");
  }
  if (!gs_connected && !has_disk_path) {
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
  if (preference == loading::SourcePreference::kPreferDisk && has_disk_path) {
    loading::DiskSource disk_src;
    disk_src.path = std::filesystem::path(hints.disk_path);

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
    if (!has_artifact_id_hint || !gs_connected) {
      return disk_or.status();
    }
    LOG(INFO) << "Disk-preferred load failed, retrying via P2P: " << disk_or.status();
  }

  // ------------------------------------------------------------------
  // 3. Request transport session – Global Store will choose a suitable source
  // ------------------------------------------------------------------
  absl::StatusOr<components::TransportSession> transport_or =
      absl::FailedPreconditionError("GlobalStoreClient not connected");
  if (gs_connected) {
    if (view_id.has_value()) {
      transport_or = gs_client_->request_view_transport(
          artifact_id,
          *view_id,
          /*source_node_id=*/"",
          /*source_address=*/"",
          /*source_port=*/0,
          target_device,
          /*wait_timeout_ms=*/30000);
      if (!transport_or.ok()) {
        if (absl::IsNotFound(transport_or.status()) || absl::IsUnimplemented(transport_or.status())) {
          LOG(INFO) << "Variant transport unavailable for view_id=" << *view_id << " (" << transport_or.status()
                    << "); falling back to canonical routing";
          transport_or = gs_client_->request_replica_transport(
              artifact_id,
              /*source_node_id=*/"",
              /*source_address=*/"",
              /*source_port=*/0,
              target_device,
              /*wait_timeout_ms=*/30000);
        }
      }
    } else {
      transport_or = gs_client_->request_replica_transport(
          artifact_id,
          /*source_node_id=*/"", // Local node info optional – left empty here
          /*source_address=*/"",
          /*source_port=*/0,
          target_device,
          /*wait_timeout_ms=*/30000);
    }
  }

  if (transport_or.ok()) {
    const auto& session = *transport_or;
    const auto& remote = session.remote_replica;

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

    if (hints.disk_path.empty()) {
      return load_or.status();
    }

  } else {
    // Not found or GS unavailable → fall back to disk
    LOG(INFO) << "request_replica_transport failed: " << transport_or.status() << "; falling back to disk"
              << (view_id ? absl::StrCat(" (view_id=", *view_id, ")") : "");
    if (hints.disk_path.empty()) {
      return transport_or.status();
    }
  }

  // ------------------------------------------------------------------
  // 3. Disk fallback
  // ------------------------------------------------------------------
  DiskSource disk_src;
  disk_src.path = std::filesystem::path(hints.disk_path);

  ReplicaTarget target;
  target.location.type = (target_device.type == DeviceType::GPU) ? common::memory::MemoryLocation::GPU
                                                                 : common::memory::MemoryLocation::CPU;
  target.location.device_id = target_device.ordinal;

  if (preference == loading::SourcePreference::kPreferDisk && attempted_disk_first) {
    if (!last_p2p_status.ok()) {
      return last_p2p_status;
    }
    return transport_or.ok() ? absl::FailedPreconditionError("disk path already attempted") : transport_or.status();
  }

  auto disk_or = backend_->ingest_from_disk(std::string(artifact_id), disk_src, target, hints);
  if (disk_or.ok()) {
    const auto& handle = *disk_or;
    register_with_global_store(handle);
  }
  return disk_or;
}

} // namespace tensorcast::store::materialization::control
