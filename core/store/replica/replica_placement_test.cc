// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/replica/replica_placement.h"

#include <memory>

#include "catch2/catch_test_macros.hpp"
#include "core/common/async_runtime.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/device_registry.h"
#include "core/store/device_types.h"
#include "gsl/pointers"

using tensorcast::DeviceType;
using tensorcast::store::DeviceKey;
using tensorcast::store::DeviceRegistry;
using tensorcast::store::replica::normalize_replica_device_key;
using tensorcast::store::replica::ReplicaConfig;
using tensorcast::store::replica::resolve_replica_config_device_key;

namespace {

ReplicaConfig make_config(DeviceType device_type, int local_device_id) {
  auto pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(
      4096, 4096, tensorcast::common::memory::PinnedBufferPool::Options{.register_on_create = false});
  auto async_runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  return ReplicaConfig{
      .device_type = device_type,
      .local_device_id = local_device_id,
      .pinned_buffer_pool = gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      .async_runtime = gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{async_runtime},
  };
}

} // namespace

TEST_CASE("ReplicaConfig placement uses device_type as the authority", "[replica][placement]") {
  ReplicaConfig cpu_config = make_config(DeviceType::CPU, -1);
  auto cpu_or = resolve_replica_config_device_key(cpu_config);
  REQUIRE(cpu_or.ok());
  CHECK(cpu_or->type == DeviceType::CPU);
  CHECK(cpu_or->ordinal == -1);

  ReplicaConfig gpu_config = make_config(DeviceType::GPU, 0);
  auto gpu_or = resolve_replica_config_device_key(gpu_config);
  REQUIRE(gpu_or.ok());
  CHECK(gpu_or->type == DeviceType::GPU);
  CHECK(gpu_or->ordinal == 0);

  ReplicaConfig ambiguous_cpu_config = make_config(DeviceType::CPU, 0);
  CHECK_FALSE(resolve_replica_config_device_key(ambiguous_cpu_config).ok());

  ReplicaConfig missing_gpu_config = make_config(DeviceType::GPU, -1);
  CHECK_FALSE(resolve_replica_config_device_key(missing_gpu_config).ok());
}

TEST_CASE("Replica device normalization does not default GPU placement to ordinal zero", "[replica][placement]") {
  DeviceKey missing_gpu{.type = DeviceType::GPU, .ordinal = -1, .uuid = ""};
  const DeviceKey still_missing = normalize_replica_device_key(missing_gpu);
  CHECK(still_missing.type == DeviceType::GPU);
  CHECK(still_missing.ordinal == -1);

  DeviceRegistry::instance().register_gpu(7, "replica-placement-test-gpu");
  DeviceKey uuid_only_gpu{.type = DeviceType::GPU, .ordinal = -1, .uuid = "replica-placement-test-gpu"};
  const DeviceKey resolved_gpu = normalize_replica_device_key(uuid_only_gpu);
  CHECK(resolved_gpu.type == DeviceType::GPU);
  CHECK(resolved_gpu.ordinal == 7);
  CHECK(resolved_gpu.uuid == "replica-placement-test-gpu");

  DeviceKey cpu_with_ordinal{.type = DeviceType::CPU, .ordinal = 3, .uuid = "ignored"};
  const DeviceKey canonical_cpu = normalize_replica_device_key(cpu_with_ordinal);
  CHECK(canonical_cpu.type == DeviceType::CPU);
  CHECK(canonical_cpu.ordinal == -1);
  CHECK(canonical_cpu.uuid.empty());
}
