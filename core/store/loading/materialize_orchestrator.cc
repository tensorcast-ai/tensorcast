// Copyright (c) 2025, TensorCast Team.

#include "core/store/loading/materialize_orchestrator.h"

#include <filesystem>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/store/components/global_store_client.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/loading/replica_registration_helper.h"
#include "core/store/store_engine.h"

namespace tensorcast::store::loading {

MaterializeOrchestrator::MaterializeOrchestrator(
    gsl::not_null<StoreEngine*> store,
    gsl::not_null<components::GlobalStoreClient*> gs_client)
    : store_(store), gs_client_(gs_client) {}

absl::StatusOr<ReplicaHandle> MaterializeOrchestrator::run(
    std::string_view artifact_id,
    const DeviceKey& target_device,
    const MaterializeHints& hints) {
  // ------------------------------------------------------------------
  // 1. Query Global Store for existing replicas
  // ------------------------------------------------------------------
  if (!gs_client_->is_connected()) {
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
    p2p_src.verification_json = remote.verification_json;
    p2p_src.enable_checksum = true;
    p2p_src.location.type = remote.memory_type;
    p2p_src.location.device_id = remote.device_id;

    // Build target description
    ReplicaTarget target;
    target.location.type = (target_device.type == DeviceType::GPU) ? common::memory::MemoryLocation::GPU
                                                                   : common::memory::MemoryLocation::CPU;
    target.location.device_id = target_device.ordinal;

    auto load_or = store_->ingest_from_p2p_internal(std::string(artifact_id), p2p_src, target, hints);
    if (load_or.ok()) {
      // Notify GS that transport finished
      absl::Status comp_status = gs_client_->complete_replica_transport(session.transport_id);
      if (!comp_status.ok()) {
        LOG(WARNING) << "complete_replica_transport returned error: " << comp_status;
      }

      // Register replica with Global Store using the engine's worker identity
      // and computed key. This avoids placeholder worker IDs and keeps
      // registrations consistent with WorkerLifecycleManager.
      const auto& handle = *load_or;
      absl::Status reg_status = store_->register_replica_with_global_store(handle.key());
      if (!reg_status.ok()) {
        LOG(WARNING) << "register_replica_with_global_store returned error: " << reg_status;
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
  target.location.type = (target_device.type == DeviceType::GPU) ? common::memory::MemoryLocation::GPU
                                                                 : common::memory::MemoryLocation::CPU;
  target.location.device_id = target_device.ordinal;

  auto disk_or = store_->ingest_from_disk_internal(hints.disk_path, disk_src, target, hints);
  if (disk_or.ok()) {
    // Register with Global Store using the engine helper, overriding the
    // artifact id with the disk path for legacy visibility.
    const auto& handle = *disk_or;
    absl::Status reg_status =
        store_->register_replica_with_global_store(handle.key(), /*artifact_id_override=*/hints.disk_path);
    if (!reg_status.ok()) {
      LOG(WARNING) << "register_replica_with_global_store (disk path) returned error: " << reg_status;
    }
  }
  return disk_or;
}

} // namespace tensorcast::store::loading
