// Copyright (c) 2025-2026, TensorCast Team.

// StatusAssembler: assemble status responses from StoreEngine and RefTracker

#pragma once

#include <chrono>
#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "core/store/store_engine.h"
#include "daemon/state/ref_tracker.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class StatusAssembler {
 public:
  // Populates memory pool info, GPU aggregation, CPU replicas, communication summary,
  // and global totals in the provided response. Caller is responsible for setting
  // top-level worker fields (is_registered, is_healthy, is_shutting_down, uptime, worker_id).
  static void FillDetailedStatus(store::StoreEngine& engine, RefTracker& refs, v2::GetDetailedStatusResponse& resp) {
    auto* mp = resp.mutable_memory_pool_info();
    mp->set_total_size_bytes(engine.get_mem_pool_size());
    mp->set_available_bytes(engine.get_available_memory());
    mp->set_allocated_bytes(engine.get_mem_pool_size() - engine.get_available_memory());
    mp->set_allocated_chunks_count(0);
    mp->set_chunk_size_bytes(engine.get_tx_slice_bytes());

    uint64_t total_bytes = 0;
    int32_t total_replicas = 0;

    struct GpuAgg {
      v2::GpuDeviceInfo* out;
      bool mem_filled{false};
    };

    absl::flat_hash_map<int, GpuAgg> gpu_map;

    for (const auto& info : engine.get_all_replicas_info()) {
      if (info.gpu_state != common::memory::MemoryLocation::NONE) {
        auto it = gpu_map.find(info.gpu_device_id);
        if (it == gpu_map.end()) {
          auto* gpu = resp.add_gpu_devices();
          gpu->set_device_id(info.gpu_device_id);
          gpu->set_device_uuid(info.gpu_device_uuid);
          it = gpu_map.emplace(info.gpu_device_id, GpuAgg{.out = gpu, .mem_filled = false}).first;
        }
        if (!it->second.mem_filled) {
          size_t total_mem = 0;
          size_t free_mem = 0;
          if (auto t = engine.get_device_total_memory(info.gpu_device_id); t.ok())
            total_mem = *t;
          if (auto f = engine.get_device_free_memory(info.gpu_device_id); f.ok())
            free_mem = *f;
          it->second.out->set_total_memory_bytes(static_cast<uint64_t>(total_mem));
          it->second.out->set_free_memory_bytes(static_cast<uint64_t>(free_mem));
          uint64_t used = (total_mem > free_mem) ? static_cast<uint64_t>(total_mem - free_mem) : 0ULL;
          it->second.out->set_used_memory_bytes(used);
          it->second.mem_filled = true;
        }
        auto* r = it->second.out->add_loaded_replicas();
        r->set_artifact_id(info.artifact_id);
        r->set_artifact_size_bytes(info.size_bytes);
        r->set_location(v2::MemoryLocation::MEMORY_LOCATION_GPU);
        r->set_loaded_timestamp(
            std::chrono::duration_cast<std::chrono::seconds>(info.load_time.time_since_epoch()).count());
        r->set_last_access_timestamp(
            std::chrono::duration_cast<std::chrono::seconds>(info.last_access_time.time_since_epoch()).count());
        r->add_replica_uuids("");
        r->set_is_registered_for_comm(info.is_registered_for_comm);
        total_replicas += 1;
        total_bytes += info.size_bytes;
      }
      if (info.cpu_state != common::memory::MemoryLocation::NONE) {
        auto* r = resp.add_cpu_replicas();
        r->set_artifact_id(info.artifact_id);
        r->set_artifact_size_bytes(info.size_bytes);
        r->set_location(v2::MemoryLocation::MEMORY_LOCATION_PAGEABLE_CPU);
        r->set_loaded_timestamp(
            std::chrono::duration_cast<std::chrono::seconds>(info.load_time.time_since_epoch()).count());
        r->set_last_access_timestamp(
            std::chrono::duration_cast<std::chrono::seconds>(info.last_access_time.time_since_epoch()).count());
        r->add_replica_uuids("");
        r->set_is_registered_for_comm(info.is_registered_for_comm);
        total_replicas += 1;
        total_bytes += info.size_bytes;
      }
    }

    bool any_comm = false;
    for (const auto& info : engine.get_all_replicas_info()) {
      any_comm = any_comm || info.is_registered_for_comm;
    }
    resp.mutable_communication_info()->set_enabled(any_comm);
    resp.set_total_replicas_loaded(total_replicas);
    resp.set_total_artifact_size_bytes(static_cast<int64_t>(total_bytes));
    resp.set_storage_path("");
    resp.set_num_worker_threads(0);
  }
};

} // namespace tensorcast::daemon
