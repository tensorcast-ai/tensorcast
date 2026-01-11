// Copyright (c) 2025-2026, TensorCast Team.

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "catch2/catch_test_macros.hpp"
#include "core/common/memory/memory_location.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/runtime/context/runtime_context_events.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/runtime/replica/replica_runtime.h"
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
using tensorcast::store::runtime::IngestionResultEvent;
using tensorcast::store::runtime::RemoteAccessEvent;
using tensorcast::store::runtime::ReplicaLifecycleEvent;
using tensorcast::store::runtime::ReplicaPublishState;
using tensorcast::store::runtime::ReplicaRuntime;
using tensorcast::store::runtime::RuntimeContext;
using tensorcast::store::runtime::RuntimeEvent;
using tensorcast::store::runtime::RuntimeEventType;

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
    RuntimeContext& catalog,
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
      .async_runtime = gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{catalog.async_runtime()},
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
  RuntimeContext catalog(opts);
  CHECK_OK(catalog.start());
  ReplicaRuntime runtime(ReplicaRuntime::Config{.runtime_context = &catalog});

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

TEST_CASE("ReplicaRuntime publishes lifecycle events", "[runtime][replica_runtime][events]") {
  SKIP_IF_NO_CUDA();

  auto opts = MakeTestOptions();
  RuntimeContext catalog(opts);
  CHECK_OK(catalog.start());
  ReplicaRuntime runtime(ReplicaRuntime::Config{.runtime_context = &catalog});

  absl::Mutex mu;
  std::vector<RuntimeEvent> observed;
  auto subscription = catalog.subscribe_to_events([&](const RuntimeEvent& event) {
    absl::MutexLock lock(&mu);
    observed.push_back(event);
  });
  REQUIRE(subscription != nullptr);

  constexpr size_t kCpuBytes = 8192;
  auto cpu_backing = std::make_shared<std::vector<uint8_t>>(kCpuBytes, 0xAB);
  auto cpu_view = std::shared_ptr<const void>(cpu_backing, static_cast<const void*>(cpu_backing->data()));
  auto cpu_config = MakeInlineReplicaConfig(catalog, "cpu_lifecycle", DeviceType::CPU, -1, cpu_view, kCpuBytes);
  auto cpu_replica = runtime.get_or_create_replica("cpu_lifecycle", cpu_config);
  REQUIRE(cpu_replica != nullptr);

  DeviceKey cpu_device{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  ReplicaKey cpu_key = MakeReplicaKey("cpu_lifecycle", cpu_device);

  IngestionResultEvent result_event;
  result_event.source = tensorcast::store::runtime::IngestionSource::kDisk;
  result_event.artifact_id = "cpu_lifecycle";
  result_event.target_device = cpu_device;
  result_event.target_location = MemoryLocation::CPU;
  result_event.bytes_transferred = kCpuBytes;
  result_event.duration_seconds = 0.01;
  result_event.status = absl::OkStatus();
  result_event.replica_key = cpu_key;
  runtime.record_ingestion_result(result_event);

  catalog.drain_events();
  std::vector<RuntimeEvent> snapshot;
  {
    absl::MutexLock lock(&mu);
    snapshot = observed;
    observed.clear();
  }

  bool saw_loaded = false;
  for (const auto& event : snapshot) {
    if (event.type != RuntimeEventType::kReplicaLoaded) {
      continue;
    }
    const auto* payload = std::get_if<ReplicaLifecycleEvent>(&event.payload);
    REQUIRE(payload != nullptr);
    CHECK(payload->key.artifact_id == "cpu_lifecycle");
    CHECK(payload->size_bytes == kCpuBytes);
    saw_loaded = true;
  }
  CHECK(saw_loaded);

  CHECK(runtime.unload_replica(cpu_key) == 0);
  catalog.drain_events();
  {
    absl::MutexLock lock(&mu);
    snapshot = observed;
    observed.clear();
  }
  bool saw_evicted = false;
  for (const auto& event : snapshot) {
    if (event.type != RuntimeEventType::kReplicaEvicted) {
      continue;
    }
    const auto* payload = std::get_if<ReplicaLifecycleEvent>(&event.payload);
    REQUIRE(payload != nullptr);
    CHECK(payload->key.artifact_id == "cpu_lifecycle");
    CHECK(payload->size_bytes == kCpuBytes);
    saw_evicted = true;
  }
  CHECK(saw_evicted);

  CHECK(runtime.clear_mem() == 0);
  catalog.shutdown();
}

TEST_CASE("ReplicaRuntime tracks publish state for HA inventory", "[runtime][replica_runtime][ha]") {
  SKIP_IF_NO_CUDA();

  auto opts = MakeTestOptions();
  RuntimeContext catalog(opts);
  CHECK_OK(catalog.start());
  ReplicaRuntime runtime(ReplicaRuntime::Config{.runtime_context = &catalog});

  constexpr size_t kCpuBytes = 4096;
  auto cpu_backing = std::make_shared<std::vector<uint8_t>>(kCpuBytes, 0x11);
  auto cpu_view = std::shared_ptr<const void>(cpu_backing, static_cast<const void*>(cpu_backing->data()));
  auto cpu_config = MakeInlineReplicaConfig(catalog, "cpu_publish_state", DeviceType::CPU, -1, cpu_view, kCpuBytes);
  auto cpu_replica = runtime.get_or_create_replica("cpu_publish_state", cpu_config);
  REQUIRE(cpu_replica != nullptr);

  DeviceKey cpu_device{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  ReplicaKey cpu_key = MakeReplicaKey("cpu_publish_state", cpu_device);

  auto inventory = runtime.get_ha_inventory();
  CHECK(inventory.empty());

  IngestionResultEvent result_event;
  result_event.source = tensorcast::store::runtime::IngestionSource::kDisk;
  result_event.artifact_id = "cpu_publish_state";
  result_event.target_device = cpu_device;
  result_event.target_location = MemoryLocation::CPU;
  result_event.bytes_transferred = kCpuBytes;
  result_event.duration_seconds = 0.01;
  result_event.status = absl::OkStatus();
  result_event.replica_key = cpu_key;
  result_event.publish_to_global_store = true;
  runtime.record_ingestion_result(result_event);

  inventory = runtime.get_ha_inventory();
  REQUIRE(inventory.size() == 1);
  CHECK(inventory.front().key == cpu_key);
  CHECK(inventory.front().memory_location == MemoryLocation::CPU);
  CHECK(inventory.front().publish_state == ReplicaPublishState::kPublishPending);

  runtime.set_replica_publish_state(cpu_key, ReplicaPublishState::kPublished);
  inventory = runtime.get_ha_inventory();
  REQUIRE(inventory.size() == 1);
  CHECK(inventory.front().publish_state == ReplicaPublishState::kPublished);

  runtime.set_replica_publish_state(cpu_key, ReplicaPublishState::kRetiring);
  inventory = runtime.get_ha_inventory();
  CHECK(inventory.empty());

  runtime.set_replica_publish_state(cpu_key, ReplicaPublishState::kPublished);
  CHECK(runtime.unload_replica(cpu_key) == 0);
  inventory = runtime.get_ha_inventory();
  CHECK(inventory.empty());

  CHECK(runtime.clear_mem() == 0);
  catalog.shutdown();
}

TEST_CASE("ReplicaRuntime toggles GPU exports and chunk snapshots", "[runtime][replica_runtime]") {
  SKIP_IF_NO_CUDA();

  auto opts = MakeTestOptions();
  RuntimeContext catalog(opts);
  CHECK_OK(catalog.start());
  ReplicaRuntime runtime(ReplicaRuntime::Config{.runtime_context = &catalog});

  absl::Mutex remote_mu;
  std::vector<RemoteAccessEvent> remote_events;
  auto remote_subscription = catalog.subscribe_to_events([&](const RuntimeEvent& event) {
    if (event.type != RuntimeEventType::kRemoteAccessToggled) {
      return;
    }
    const auto* payload = std::get_if<RemoteAccessEvent>(&event.payload);
    if (payload == nullptr) {
      return;
    }
    absl::MutexLock lock(&remote_mu);
    remote_events.push_back(*payload);
  });
  REQUIRE(remote_subscription != nullptr);

  constexpr size_t kGpuBytes = 32 * 1024;
  auto gpu_backing = std::make_shared<std::vector<uint8_t>>(kGpuBytes, 0xA5);
  auto gpu_view = std::shared_ptr<const void>(gpu_backing, static_cast<const void*>(gpu_backing->data()));
  auto gpu_config = MakeInlineReplicaConfig(catalog, "gpu_artifact", DeviceType::GPU, 0, gpu_view, kGpuBytes);

  auto gpu_replica = runtime.get_or_create_replica("gpu_artifact", gpu_config);
  REQUIRE(gpu_replica != nullptr);

  auto cpu_stage_future = gpu_replica->ensure_loaded_async(MemoryLocation::CPU);
  const absl::Status cpu_stage_status = std::move(cpu_stage_future).get();
  CHECK(cpu_stage_status.ok());

  auto load_future = gpu_replica->ensure_loaded_async(MemoryLocation::GPU, 1, 0);
  const absl::Status load_status = std::move(load_future).get();
  CHECK(load_status.ok());

  DeviceKey gpu_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  ReplicaKey gpu_key = MakeReplicaKey("gpu_artifact", gpu_device);
  CHECK(runtime.wait_replica_ready(gpu_key) == 0);

  auto export_or = runtime.enable_remote_replica_access(gpu_key, MemoryLocation::GPU);
  REQUIRE(export_or.ok());
  CHECK_FALSE(export_or->buffer_addresses.empty());
  CHECK(export_or->device_id == 0);

  catalog.drain_events();
  std::vector<RemoteAccessEvent> snapshot;
  {
    absl::MutexLock lock(&remote_mu);
    snapshot = remote_events;
    remote_events.clear();
  }
  REQUIRE_FALSE(snapshot.empty());
  const auto& enable_payload = snapshot.back();
  CHECK(enable_payload.enabled);
  CHECK(enable_payload.key.artifact_id == "gpu_artifact");

  auto disable_status = runtime.disable_remote_replica_access(gpu_key, MemoryLocation::GPU);
  CHECK(disable_status.ok());

  catalog.drain_events();
  {
    absl::MutexLock lock(&remote_mu);
    snapshot = remote_events;
    remote_events.clear();
  }
  REQUIRE_FALSE(snapshot.empty());
  const auto& disable_payload = snapshot.back();
  CHECK_FALSE(disable_payload.enabled);
  CHECK(disable_payload.key.artifact_id == "gpu_artifact");

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
