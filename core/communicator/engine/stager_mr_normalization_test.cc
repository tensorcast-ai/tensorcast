// Copyright (c) 2025-2026, TensorCast Team.

#include "core/communicator/engine/host_pinned_cpu_stager.h"
#include "core/communicator/engine/host_pinned_gpu_stager.h"
#include "core/communicator/engine/memory_stager.h"

#include <vector>

#include "catch2/catch_test_macros.hpp"

#include "core/common/cuda_api.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/communicator/base/constants.h"
#include "core/communicator/transport/partition_tensor.h"

namespace tensorcast::communicator::engine {

TEST_CASE("NormalizeMrRegion uses slab base for host-pinned CPU staging") {
  constexpr size_t kSliceBytes = 4096;
  auto pool = std::make_shared<common::memory::PinnedBufferPool>(kSliceBytes * 2, kSliceBytes);
  HostPinnedCpuStager stager(
      gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{pool}, /*num_buffers_hint=*/2);

  std::vector<uint8_t> src(kSliceBytes, 0xAB);
  auto tensor = std::make_shared<transport::PartitionTensor>(
      "cpu_tensor", reinterpret_cast<uint64_t>(src.data()), kSliceBytes, base::COMMUNICATE_ENGINE_DEV_CPU, nullptr);

  auto staged_or = stager.stage(tensor, /*offset=*/0, /*bytes=*/kSliceBytes, MemoryStager::StageMode::kBlocking);
  REQUIRE(staged_or.ok());
  void* staged_ptr = *staged_or;

  auto slab = NormalizeMrRegion(stager, gsl::not_null<void*>{staged_ptr}, kSliceBytes);
  auto slabs = pool->list_slabs();
  REQUIRE(!slabs.empty());
  CHECK(slab.base.get() == slabs.front().base.get());
  CHECK(slab.bytes == slabs.front().bytes);

  REQUIRE(stager.release_staged_buffer(gsl::not_null<void*>{staged_ptr}).ok());
}

TEST_CASE("NormalizeMrRegion uses slab base for host-pinned GPU staging") {
  constexpr size_t kSliceBytes = 4096;
  auto pool = std::make_shared<common::memory::PinnedBufferPool>(kSliceBytes * 2, kSliceBytes);
  HostPinnedGpuStager stager(kSliceBytes, /*num_buffers=*/2, pool);

  void* gpu_ptr = nullptr;
  REQUIRE(cuda::malloc(&gpu_ptr, kSliceBytes).ok());
  auto tensor = std::make_shared<transport::PartitionTensor>(
      "gpu_tensor", reinterpret_cast<uint64_t>(gpu_ptr), kSliceBytes, base::COMMUNICATE_ENGINE_DEV_GPU, nullptr);
  tensor->set_device_id(0);

  auto staged_or = stager.stage(tensor, /*offset=*/0, /*bytes=*/kSliceBytes, MemoryStager::StageMode::kTry);
  REQUIRE(staged_or.ok());
  void* staged_ptr = *staged_or;

  auto slab = NormalizeMrRegion(stager, gsl::not_null<void*>{staged_ptr}, kSliceBytes);
  auto slabs = pool->list_slabs();
  REQUIRE(!slabs.empty());
  CHECK(slab.base.get() == slabs.front().base.get());
  CHECK(slab.bytes == slabs.front().bytes);

  REQUIRE(stager.release_staged_buffer(gsl::not_null<void*>{staged_ptr}).ok());
  REQUIRE(cuda::free(gpu_ptr).ok());
}

} // namespace tensorcast::communicator::engine
