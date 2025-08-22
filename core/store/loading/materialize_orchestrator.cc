// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loading/materialize_orchestrator.h"

#include <filesystem>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/store/components/global_store_client.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/loading/replica_registration_helper.h"
#include "core/store/store_engine.h"

namespace stepcast::store {

MaterializeOrchestrator::MaterializeOrchestrator(StoreEngine* store, GlobalStoreClient* gs_client)
    : store_(store), gs_client_(gs_client) {}

absl::StatusOr<ReplicaHandle> MaterializeOrchestrator::run(
    std::string_view artifact_id,
    const DeviceKey& target_device,
    const MaterializeHints& hints) {
  // ------------------------------------------------------------------
  // 1. Query Global Store for existing replicas
  // ------------------------------------------------------------------
  if (!gs_client_ || !gs_client_->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  // ------------------------------------------------------------------
  // 2. Request transport session – Global Store will choose a suitable source
  // ------------------------------------------------------------------
  auto transport_or = gs_client_->request_replica_transport(
      artifact_id,
      /*source_node_id=*/"", // Local node info optional – left empty here
      /*source_address=*/"",
      /*source_port=*/0,
      target_device,
      /*wait_timeout_ms=*/30000);

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
    p2p_src.enable_checksum = true;
    p2p_src.location.type = remote.memory_type;
    p2p_src.location.device_id = remote.device_id;

    // Build target description
    ReplicaTarget target;
    target.location.type = (target_device.type == DeviceType::GPU) ? MemoryLocation::GPU : MemoryLocation::PAGEABLE_CPU;
    target.location.device_id = target_device.ordinal;

    auto load_or = store_->ingest_from_p2p_internal(std::string(artifact_id), p2p_src, target, hints);
    if (load_or.ok()) {
      // Notify GS that transport finished
      absl::Status comp_status = gs_client_->complete_replica_transport(session.transport_id);
      if (!comp_status.ok()) {
        LOG(WARNING) << "complete_replica_transport returned error: " << comp_status;
      }

      // Register replica with size info from remote (best effort)
      absl::Status reg_status = ReplicaRegistrationHelper::register_local_replica(
          gs_client_, artifact_id, target_device, target.location.type, remote.memory_size);
      if (!reg_status.ok()) {
        LOG(WARNING) << "register_local_replica returned error: " << reg_status;
      }

      return load_or;
    } // Loading via P2P failed – close transport and log
    absl::Status comp_status = gs_client_->complete_replica_transport(session.transport_id);
    if (!comp_status.ok()) {
      LOG(WARNING) << "complete_replica_transport after failure returned error: " << comp_status;
    }
    LOG(WARNING) << "P2P load failed: " << load_or.status();

  } else {
    // Not found or GS unavailable → fall back to disk
    LOG(INFO) << "request_replica_transport failed: " << transport_or.status() << "; falling back to disk";
  }

  // ------------------------------------------------------------------
  // 3. Disk fallback
  // ------------------------------------------------------------------
  // Content-addressed IDs (mi2:...) are not paths. Disk fallback requires an explicit hints.disk_path.
  if (hints.disk_path.empty()) {
    return absl::FailedPreconditionError(
        "Disk fallback requires hints.disk_path; content-addressed artifact_id must route via Global Store");
  }
  DiskSource disk_src;
  disk_src.path = std::filesystem::path(hints.disk_path);

  ReplicaTarget target;
  target.location.type = (target_device.type == DeviceType::GPU) ? MemoryLocation::GPU : MemoryLocation::PAGEABLE_CPU;
  target.location.device_id = target_device.ordinal;

  auto disk_or = store_->ingest_from_disk_internal(hints.disk_path, disk_src, target, hints);
  if (disk_or.ok()) {
    // Attempt to register replica with actual size if available
    // Query size via a public API to avoid peeking internals; tolerate failures.
    uint64_t artifact_size = 0;
    if (auto size_or = store_->get_replica_size(disk_or->key()); size_or.ok()) {
      artifact_size = *size_or;
    }
    // For disk fallback, register using disk_path as a legacy attribute stored on the replica.
    // Content-addressed routing remains via artifact_id in other flows.
    absl::Status reg_status = ReplicaRegistrationHelper::register_local_replica(
        gs_client_, hints.disk_path, target_device, target.location.type, artifact_size);
    if (!reg_status.ok()) {
      LOG(WARNING) << "register_local_replica (disk path) returned error: " << reg_status;
    }
  }
  return disk_or;
}

} // namespace stepcast::store