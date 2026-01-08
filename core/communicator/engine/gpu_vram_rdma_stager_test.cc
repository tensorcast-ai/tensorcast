// Copyright (c) 2025-2026, TensorCast Team.

#include "core/communicator/engine/gpu_vram_rdma_stager.h"

#include "catch2/catch_test_macros.hpp"

#include "core/communicator/base/constants.h"
#include "core/communicator/transport/partition_tensor.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::communicator::engine {

TEST_CASE("GpuVramRdmaStager stages and releases slices") {
  constexpr size_t kSliceBytes = 4096;
  constexpr size_t kPoolBytes = kSliceBytes * 2;

  auto pool = std::make_shared<GpuVramStagingPool>(/*device_id=*/0, kPoolBytes, kSliceBytes);
  REQUIRE(pool->initialize().ok());
  GpuVramRdmaStager stager(pool);

  void* gpu_ptr = nullptr;
  REQUIRE(cuda::malloc(&gpu_ptr, kSliceBytes).ok());
  auto tensor = std::make_shared<transport::PartitionTensor>(
      "gpu_tensor", reinterpret_cast<uint64_t>(gpu_ptr), kSliceBytes, base::COMMUNICATE_ENGINE_DEV_GPU, nullptr);
  tensor->set_device_id(0);

  auto staged_or = stager.stage(tensor, /*offset=*/0, /*bytes=*/kSliceBytes, MemoryStager::StageMode::kTry);
  REQUIRE(staged_or.ok());
  void* staged_ptr = *staged_or;

  auto slab = stager.mr_slab_for_ptr(gsl::not_null<void*>{staged_ptr});
  REQUIRE(slab.has_value());
  CHECK(slab->base.get() == pool->base_ptr());
  CHECK(slab->bytes == pool->pool_bytes());

  REQUIRE(stager.release_staged_buffer(gsl::not_null<void*>{staged_ptr}).ok());
  REQUIRE(cuda::free(gpu_ptr).ok());
}

} // namespace tensorcast::communicator::engine
