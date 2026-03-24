// Copyright (c) 2026, TensorCast Team.

#include "core/store/replica/collective_disk_loader.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "core/store/materialization/contracts/view/view_spec.h"

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

loading::VariantIdentity make_transpose_variant() {
  loading::VariantIdentity variant;
  variant.view_spec.emplace();
  materialization::view::TensorViewOps ops;
  ops.ops.push_back(materialization::view::ViewOp::Transpose(materialization::view::TransposeOp{.dim0 = 0, .dim1 = 1}));
  variant.view_spec->tensors.emplace("tensor", std::move(ops));
  return variant;
}

} // namespace

TEST_CASE(
    "try_local_batched_disk_load falls back on unsupported view ops",
    "[collective_disk_loader][local_batched][fallback]") {
  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_local_batched_disk_load = true;

  LocalBatchedDiskLoadRequest request{
      .replica_key = loading::ReplicaKey{.artifact_id = "artifact_local_batched"},
      .disk_context = make_disk_context(),
      .source_index_json = R"({"tensor":[0,16,[4,4],[4,1],"torch.uint8",0]})",
      .view_index_json = R"({"tensor":[0,16,[4,4],[4,1],"torch.uint8",0]})",
      .variant_identity = make_transpose_variant(),
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
