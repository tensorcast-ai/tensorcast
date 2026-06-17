// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/runtime/pipeline/handle_stage.h"

#include <optional>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "core/cuda/cuda_ipc.h"
#include "core/store/replica/unified_memory_authority.h"

namespace tensorcast::store::materialization::runtime::pipeline {

absl::StatusOr<loading::ReplicaHandle> HandleStage::build(IngestionContext& ctx) {
  loading::ReplicaHandle handle;

  // The replica already carries the canonicalized device identity chosen by
  // the engine. Reuse that key verbatim so later session joins (for example
  // prefetch followed by tensor_dict on the same replica_uuid) see identical
  // ReplicaKeys instead of mixing normalized and non-normalized DeviceKeys.
  handle.replica_key = ctx.replica->replica_key();
  handle.ready_signal = ctx.load_signal;
  handle.cpu_state = ctx.replica->get_memory_state(common::memory::MemoryLocation::CPU);
  handle.gpu_state = ctx.replica->get_memory_state(common::memory::MemoryLocation::GPU);

  if (ctx.target_is_gpu) {
    const auto gpu_ptrs = ctx.replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
    handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

    auto ipc_or = ctx.replica->get_memory_manager().get_ipc_handle();
    if (ipc_or.ok()) {
      handle.cuda_ipc_handle = cuda::IpcHandleBytes::from_native(*ipc_or);
    } else {
      LOG(WARNING) << "HandleStage: get_ipc_handle failed for key=" << handle.replica_key
                   << " gpu_state=" << static_cast<int>(handle.gpu_state) << " status=" << ipc_or.status();
    }
  } else {
    auto uma = ctx.replica->get_memory_manager().memory_authority();
    if (uma) {
      const loading::ReplicaKey& allocation_key = ctx.replica->replica_key();
      auto region_or = uma->get_cpu_memfd_region(allocation_key);
      if (region_or.ok()) {
        handle.cpu_memfd_region = loading::CpuMemfdRegion{
            .fd = region_or->fd,
            .size_bytes = region_or->size_bytes,
            .offset_bytes = region_or->offset_bytes,
        };
        if (ctx.options != nullptr && ctx.options->cpu_shared_memory_enabled) {
          VLOG(1) << "HandleStage: attached CPU memfd region for key=" << handle.replica_key
                  << " size_bytes=" << region_or->size_bytes << " offset_bytes=" << region_or->offset_bytes;
        }
      } else if (ctx.options != nullptr && ctx.options->cpu_shared_memory_enabled) {
        LOG(WARNING) << "HandleStage: cpu_shared_memory_enabled but get_cpu_memfd_region failed for allocation_key="
                     << allocation_key << " (request_key=" << handle.replica_key << "): " << region_or.status();
      }
    } else if (ctx.options != nullptr && ctx.options->cpu_shared_memory_enabled) {
      LOG(WARNING) << "HandleStage: cpu_shared_memory_enabled but UMA is unavailable for key=" << handle.replica_key;
    }
  }

  if (ctx.resolved_view_plan.has_value() && !ctx.resolved_view_plan->is_identity) {
    handle.view_index_json = ctx.resolved_view_plan->view_index_json;
  }
  if (!handle.view_index_json.has_value() && ctx.source_type == SourceType::kDisk) {
    if (ctx.verification.canonical_index_json.has_value()) {
      handle.view_index_json = ctx.verification.canonical_index_json;
      VLOG(1) << "HandleStage: attached canonical index from disk for artifact=" << ctx.artifact_identifier;
    } else {
      VLOG(1) << "HandleStage: no canonical index available for disk source artifact=" << ctx.artifact_identifier;
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
