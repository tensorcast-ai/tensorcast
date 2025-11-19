// Copyright (c) 2025, TensorCast Team.

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"
#include "core/common/memory/memory_location.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"
#include "core/store/runtime/component_catalog.h"
#include "core/store/runtime/replica_runtime.h"
#include "core/store/runtime/runtime_event_hub.h"
#include "core/store/store_engine_options.h"
#include "core/testing/test_helpers.h"
#include "gsl/pointers"

using tensorcast::DeviceType;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::DeviceKey;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::loading::InlineBufferSource;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::replica::Replica;
using tensorcast::store::replica::ReplicaConfig;
using tensorcast::store::runtime::ComponentCatalog;
using tensorcast::store::runtime::ReplicaRuntime;
using tensorcast::store::runtime::RuntimeEventHub;

namespace {

StoreEngineOptions MakeTestOptions() {
  StoreEngineOptions opts;
  opts.storage_path.clear();
  opts.memory_pool_size = 32ull * 1024 * 1024;
  opts.tx_slice_bytes = 256 * 1024;
  opts.artifact_chunk_bytes = opts.tx_slice_bytes;
  opts.num_thread = 1;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  opts.p2p_listen_host = "127.0.0.1";
  opts.p2p_port = 0;
  opts.enable_rdma = false;
  return opts;
}

ReplicaConfig MakeInlineReplicaConfig(
    ComponentCatalog& catalog,
    std::string artifact_id,
    DeviceType device_type,
    int device_id,
    const std::shared_ptr<const void>& view,
    size_t bytes) {
  ReplicaConfig config{
      .source = InlineBufferSource{.data = view, .size_bytes = bytes},
      .artifact_identifier = std::move(artifact_id),
      .device_type = device_type,
      .local_device_id = device_id,
      .pinned_buffer_pool =
          gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{catalog.pinned_buffer_pool()},
      .artifact_chunk_bytes = catalog.artifact_chunk_bytes(),
      .expected_artifact_size = bytes,
      .max_buffer_bytes = bytes,
      .pinned_memory_timeout = std::chrono::milliseconds(0),
  };
  return config;
}

ReplicaKey MakeReplicaKey(std::string artifact_id, DeviceKey device) {
  ReplicaKey key{
      .artifact_id = std::move(artifact_id),
      .view_id = std::nullopt,
      .device = device,
      .replica = 0,
  };
  return key;
}

} // namespace

TEST_CASE("ReplicaRuntime evicts CPU replicas and reports telemetry", "[runtime][replica_runtime]") {
  SKIP_IF_NO_CUDA();

  auto opts = MakeTestOptions();
  ComponentCatalog catalog(opts);
  CHECK_OK(catalog.start());

  RuntimeEventHub hub;
  ReplicaRuntime runtime(ReplicaRuntime::Config{.component_catalog = &catalog, .event_hub = &hub});

  constexpr size_t kCpuBytes = 16 * 1024;
  auto cpu_backing = std::make_shared<std::vector<uint8_t>>(kCpuBytes, 0x5C);
  auto cpu_view = std::shared_ptr<const void>(cpu_backing, static_cast<const void*>(cpu_backing->data()));
  auto cpu_config = MakeInlineReplicaConfig(catalog, "cpu_only_artifact", DeviceType::CPU, -1, cpu_view, kCpuBytes);

  auto cpu_replica = runtime.get_or_create_replica("cpu_only_artifact", cpu_config);
  REQUIRE(cpu_replica != nullptr);

  auto chunk_states = runtime.get_chunk_states_telemetry("cpu_only_artifact");
  REQUIRE_FALSE(chunk_states.empty());

  auto info = runtime.get_all_replicas_info();
  bool found_cpu_info = false;
  for (const auto& replica_info : info) {
    if (replica_info.artifact_id == "cpu_only_artifact") {
      found_cpu_info = true;
      CHECK(replica_info.cpu_state == MemoryLocation::CPU);
      CHECK(replica_info.size_bytes == kCpuBytes);
    }
  }
  CHECK(found_cpu_info);

  auto residents_before = runtime.get_resident_devices("cpu_only_artifact");
  REQUIRE(residents_before.size() == 1);
  CHECK(residents_before.front().type == DeviceType::CPU);

  CHECK_OK(runtime.try_evict_memory_for_replica(catalog.tx_slice_bytes()));

  auto residents_after = runtime.get_resident_devices("cpu_only_artifact");
  CHECK(residents_after.empty());

  CHECK(runtime.clear_mem() == 0);
  catalog.shutdown();
}

TEST_CASE("ReplicaRuntime toggles GPU exports and chunk snapshots", "[runtime][replica_runtime]") {
  SKIP_IF_NO_CUDA();

  auto opts = MakeTestOptions();
  ComponentCatalog catalog(opts);
  CHECK_OK(catalog.start());

  RuntimeEventHub hub;
  ReplicaRuntime runtime(ReplicaRuntime::Config{.component_catalog = &catalog, .event_hub = &hub});

  constexpr size_t kGpuBytes = 32 * 1024;
  auto gpu_backing = std::make_shared<std::vector<uint8_t>>(kGpuBytes, 0xA5);
  auto gpu_view = std::shared_ptr<const void>(gpu_backing, static_cast<const void*>(gpu_backing->data()));
  auto gpu_config = MakeInlineReplicaConfig(catalog, "gpu_artifact", DeviceType::GPU, 0, gpu_view, kGpuBytes);

  auto gpu_replica = runtime.get_or_create_replica("gpu_artifact", gpu_config);
  REQUIRE(gpu_replica != nullptr);

  auto cpu_stage_future = gpu_replica->ensure_loaded_async(MemoryLocation::CPU);
  const absl::Status cpu_stage_status = cpu_stage_future.get();
  CHECK(cpu_stage_status.ok());

  auto load_future = gpu_replica->ensure_loaded_async(MemoryLocation::GPU, 1, 0);
  const absl::Status load_status = load_future.get();
  CHECK(load_status.ok());

  DeviceKey gpu_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  ReplicaKey gpu_key = MakeReplicaKey("gpu_artifact", gpu_device);
  CHECK(runtime.wait_replica_ready(gpu_key) == 0);

  auto export_or = runtime.enable_remote_replica_access(gpu_key, MemoryLocation::GPU);
  REQUIRE(export_or.ok());
  CHECK_FALSE(export_or->buffer_addresses.empty());
  CHECK(export_or->device_id == 0);

  auto disable_status = runtime.disable_remote_replica_access(gpu_key, MemoryLocation::GPU);
  CHECK(disable_status.ok());

  auto gpu_chunk_states = runtime.get_chunk_states_for_device("gpu_artifact", 0);
  REQUIRE_FALSE(gpu_chunk_states.empty());

  auto gpu_residents = runtime.get_resident_devices("gpu_artifact");
  bool found_gpu = false;
  for (const auto& device : gpu_residents) {
    if (device.type == DeviceType::GPU && device.ordinal == 0) {
      found_gpu = true;
      break;
    }
  }
  CHECK(found_gpu);

  auto unique_gpu_or = runtime.get_unique_gpu_residency("gpu_artifact");
  REQUIRE(unique_gpu_or.ok());
  CHECK(unique_gpu_or.value() == 0);

  CHECK(runtime.clear_mem() == 0);
  catalog.shutdown();
}
