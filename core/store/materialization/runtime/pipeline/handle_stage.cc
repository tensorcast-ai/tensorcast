// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/runtime/pipeline/handle_stage.h"

#include <chrono>
#include <cstring>
#include <optional>
#include <string_view>

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
  if (ctx.verification.view_data_hash.has_value()) {
    handle.view_data_hash = ctx.verification.view_data_hash;
  }

  return handle;
}

} // namespace tensorcast::store::materialization::runtime::pipeline
