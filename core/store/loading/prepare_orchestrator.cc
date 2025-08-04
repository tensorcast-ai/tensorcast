// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loading/prepare_orchestrator.h"

#include <filesystem>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/store/checkpoint_store.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/model_registry.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/loading/replica_registration_helper.h"

namespace stepcast::store {

PrepareOrchestrator::PrepareOrchestrator(
    CheckpointStore* store,
    GlobalStoreClient* gs_client,
    ModelRegistry* registry,
    DeviceManager* device_manager)
    : store_(store), gs_client_(gs_client), registry_(registry), device_manager_(device_manager) {}

absl::StatusOr<ModelHandle> PrepareOrchestrator::run(
    std::string_view model_id,
    const DeviceKey& target_device,
    const LoadingHints& hints) {
  // ------------------------------------------------------------------
  // 1. Query Global Store for existing replicas
  // ------------------------------------------------------------------
  if (!gs_client_ || !gs_client_->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  // ------------------------------------------------------------------
  // 2. Request transport session – Global Store will choose a suitable source
  // ------------------------------------------------------------------
  auto transport_or = gs_client_->request_model_transport(
      model_id,
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
    ModelTarget target;
    target.location.type = (target_device.type == DeviceType::GPU) ? ModelLocation::GPU : ModelLocation::PAGEABLE_CPU;
    target.location.device_id = target_device.ordinal;

    auto load_or = store_->load_from_p2p_internal(std::string(model_id), p2p_src, target, hints);
    if (load_or.ok()) {
      // Notify GS that transport finished
      absl::Status comp_status = gs_client_->complete_model_transport(session.transport_id);
      if (!comp_status.ok()) {
        LOG(WARNING) << "complete_model_transport returned error: " << comp_status;
      }

      // Register replica with size info from remote (best effort)
      absl::Status reg_status = ReplicaRegistrationHelper::register_local_replica(
          gs_client_, model_id, target_device, target.location.type, remote.memory_size);
      if (!reg_status.ok()) {
        LOG(WARNING) << "register_local_replica returned error: " << reg_status;
      }

      return load_or;
    } // Loading via P2P failed – close transport and log
    absl::Status comp_status = gs_client_->complete_model_transport(session.transport_id);
    if (!comp_status.ok()) {
      LOG(WARNING) << "complete_model_transport after failure returned error: " << comp_status;
    }
    LOG(WARNING) << "P2P load failed: " << load_or.status();

  } else {
    // Not found or GS unavailable → fall back to disk
    LOG(INFO) << "request_model_transport failed: " << transport_or.status() << "; falling back to disk";
  }

  // ------------------------------------------------------------------
  // 3. Disk fallback
  // ------------------------------------------------------------------
  DiskSource disk_src;
  disk_src.path = std::filesystem::path(std::string(model_id));

  ModelTarget target;
  target.location.type = (target_device.type == DeviceType::GPU) ? ModelLocation::GPU : ModelLocation::PAGEABLE_CPU;
  target.location.device_id = target_device.ordinal;

  auto disk_or = store_->load_from_disk_internal(std::string(model_id), disk_src, target, hints);
  if (disk_or.ok()) {
    // Attempt to register replica (size unknown here, pass 0)
    absl::Status reg_status = ReplicaRegistrationHelper::register_local_replica(
        gs_client_, model_id, target_device, target.location.type, /*size_bytes=*/0);
    if (!reg_status.ok()) {
      LOG(WARNING) << "register_local_replica (disk path) returned error: " << reg_status;
    }
  }
  return disk_or;
}

} // namespace stepcast::store