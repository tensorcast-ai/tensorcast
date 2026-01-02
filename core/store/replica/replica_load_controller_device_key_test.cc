// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <memory>

#include "core/common/async_runtime.h"
#include "core/common/const/granularity.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/device_types.h"
#include "core/store/replica/replica_load_controller.h"

using tensorcast::DeviceType;
using tensorcast::common::AsyncRuntime;
using tensorcast::common::memory::PinnedBufferPool;
using tensorcast::store::DeviceKey;
using tensorcast::store::replica::ReplicaLoadController;

TEST_CASE("ReplicaLoadController binds UMA allocation to canonical device key", "[replica][uma][device_key]") {
  const uint64_t artifact_size = 64 * 1024; // 64 KiB CPU replica
  auto pinned_pool = std::make_shared<PinnedBufferPool>(
      /*total_bytes=*/512 * 1024, /*chunk_bytes=*/tensorcast::common::consts::kArtifactChunkDefault);
  auto async_runtime = std::make_shared<AsyncRuntime>();
  const DeviceKey cpu_device{DeviceType::CPU, -1, ""};

  auto controller = std::make_shared<ReplicaLoadController>(
      "cpu_artifact_for_uma",
      cpu_device,
      pinned_pool,
      async_runtime,
      tensorcast::common::consts::kArtifactChunkDefault,
      /*max_buffer_bytes=*/256 * 1024,
      std::chrono::milliseconds::zero(),
      /*streaming_buffer_chunks=*/16,
      artifact_size);

  const auto key = controller->replica_key();
  REQUIRE(key.device.type == DeviceType::CPU);
  REQUIRE(key.device.ordinal == -1);

  auto uma = controller->memory_authority();
  REQUIRE(uma->has_allocation(key));

  auto layout_or = uma->get_layout(key);
  REQUIRE(layout_or.ok());
  CHECK(layout_or->artifact_bytes == artifact_size);
  CHECK(layout_or->artifact_chunk_bytes == tensorcast::common::consts::kArtifactChunkDefault);
}
