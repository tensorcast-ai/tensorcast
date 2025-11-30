// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/runtime/pipeline/handle_stage.h"

#include <cstring>
#include <optional>

#include "absl/status/statusor.h"

namespace tensorcast::store::materialization::runtime::pipeline {

absl::StatusOr<loading::ReplicaHandle> HandleStage::build(IngestionContext& ctx) {
  loading::ReplicaHandle handle;

  DeviceKey dev_key = ctx.target_device;
  if (!ctx.target_is_gpu) {
    dev_key = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  }

  handle.replica_key = loading::ReplicaKey{
      .artifact_id = ctx.artifact_identifier,
      .view_id = ctx.hints.variant ? ctx.hints.variant->view_id : std::optional<std::string>(),
      .device = dev_key,
      .replica = 0};
  handle.ready_future = ctx.load_future;
  handle.cpu_state = ctx.replica->get_memory_state(common::memory::MemoryLocation::CPU);
  handle.gpu_state = ctx.replica->get_memory_state(common::memory::MemoryLocation::GPU);

  if (ctx.target_is_gpu) {
    const auto gpu_ptrs = ctx.replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
    handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

    auto ipc_or = ctx.replica->get_memory_manager().get_ipc_handle();
    if (ipc_or.ok()) {
      std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
    }
  }

  if (ctx.resolved_view_plan.has_value() && !ctx.resolved_view_plan->is_identity) {
    handle.view_index_json = ctx.resolved_view_plan->view_index_json;
  }
  if (!handle.view_index_json.has_value() && ctx.source_type == SourceType::kDisk) {
    if (ctx.verification.canonical_index_json.has_value()) {
      handle.view_index_json = ctx.verification.canonical_index_json;
      LOG(INFO) << "HandleStage: attached canonical index from disk for artifact=" << ctx.artifact_identifier;
    } else {
      LOG(INFO) << "HandleStage: no canonical index available for disk source artifact=" << ctx.artifact_identifier;
    }
  }
  if (ctx.verification.view_data_hash.has_value()) {
    handle.view_data_hash = ctx.verification.view_data_hash;
  }

  switch (ctx.source_type) {
    case SourceType::kDisk:
      handle.source = loading::MaterializationSource::kDisk;
      break;
    case SourceType::kP2P:
      handle.source = loading::MaterializationSource::kP2P;
      break;
  }

  return handle;
}

} // namespace tensorcast::store::materialization::runtime::pipeline
