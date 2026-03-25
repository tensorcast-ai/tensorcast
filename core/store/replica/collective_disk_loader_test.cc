// Copyright (c) 2026, TensorCast Team.

#include "core/store/replica/collective_disk_loader.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <vector>

#include "catch2/catch_test_macros.hpp"

namespace tensorcast::store::replica {

namespace {

std::shared_ptr<const loader::DiskArtifactContext> make_disk_context() {
  return std::make_shared<loader::DiskArtifactContext>(
      std::filesystem::path("artifact"),
      std::vector<std::filesystem::path>{},
      std::vector<size_t>{},
      /*total_size=*/16,
      /*is_safetensors=*/true,
      /*descriptor_present=*/false,
      /*tensor_index_json_present=*/false,
      /*tensor_index_cbor_present=*/false,
      std::vector<loader::SharedSafetensorsSegment>{});
}

} // namespace

TEST_CASE(
    "try_local_batched_disk_load falls back when representation work is unavailable",
    "[collective_disk_loader][local_batched][fallback]") {
  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_local_batched_disk_load = true;

  LocalBatchedDiskLoadRequest request{
      .replica_key = loading::ReplicaKey{.artifact_id = "artifact_local_batched"},
      .disk_context = make_disk_context(),
      .strategy_config = strategy,
      .gpu_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(1)),
      .device_id = 0,
  };

  std::shared_ptr<common::memory::PinnedBufferPool> pinned_pool;
  const auto result = try_local_batched_disk_load(request, pinned_pool, std::chrono::milliseconds(0));

  REQUIRE_FALSE(result.handled);
  REQUIRE(result.status.ok());
}

} // namespace tensorcast::store::replica
