// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/testing/common.h"

#include <filesystem>
#include <future>
#include <vector>

#include "absl/status/status.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/common/memory/virtual_address_space.h"
#include "core/store/loader/disk_loader.h"
#include "core/store/loader/source.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/replica/replica_load_controller.h"

namespace fs = std::filesystem;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::common::memory::PinnedBufferPool;
using tensorcast::store::DiskLoader;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::replica::MemoryState;
using tensorcast::store::replica::ReplicaLoadController;
using tensorcast::testing::create_dummy_file;
using tensorcast::testing::is_cuda_available;
using tensorcast::testing::read_file_content;
using tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir;

TEST_CASE("DiskLoader streaming disk load to GPU", "[loader][disk][streaming][gpu]") {
  if (!is_cuda_available()) {
    SKIP("CUDA not available. Skipping DiskLoader streaming GPU test.");
  }

  // Setup dummy replica directory with two partitions
  const std::string artifact_dir = "loader_streaming_files";
  const std::string p0 = "tensor.data_0";
  const std::string p1 = "tensor.data_1";
  const size_t size0 = 1024 * 5;
  const size_t size1 = 1024 * 3;
  const size_t total_size = size0 + size1;

  fs::path base = fs::temp_directory_path() / "disk_loader_streaming_test";
  if (fs::exists(base))
    fs::remove_all(base);
  fs::create_directories(base / artifact_dir);

  fs::path path0 = base / artifact_dir / p0;
  fs::path path1 = base / artifact_dir / p1;
  REQUIRE(create_dummy_file(path0, size0, 'S'));
  REQUIRE(create_dummy_file(path1, size1, 'T'));

  // RFC-0007: ensure descriptor and canonical index exist
  REQUIRE(write_rfc0007_descriptor_for_standard_artifact_dir(base / artifact_dir).ok());

  // Read combined data for verification
  auto data0 = read_file_content(path0);
  auto data1 = read_file_content(path1);
  std::vector<char> combined;
  combined.reserve(total_size);
  combined.insert(combined.end(), data0.begin(), data0.end());
  combined.insert(combined.end(), data1.begin(), data1.end());
  REQUIRE(combined.size() == total_size);

  // Setup DiskLoader
  DiskSource source;
  source.path = base / artifact_dir;
  DiskLoader loader(source);
  REQUIRE(loader.initialize().ok());
  auto size_status = loader.get_artifact_size();
  REQUIRE(size_status.ok());
  const uint64_t artifact_size = *size_status;
  REQUIRE(artifact_size == total_size);

  // Setup ReplicaLoadController with streaming enabled
  const size_t pool_total = 1024 * 1024;
  const size_t pool_chunk = 4096;
  auto pool = std::make_shared<PinnedBufferPool>(pool_total, pool_chunk);
  auto virtual_addr_space = std::make_shared<tensorcast::common::memory::VirtualAddressSpace>();
  auto memmgr = std::make_shared<ReplicaLoadController>(
      "loader_stream_artifact",
      /*device=*/0,
      pool,
      virtual_addr_space,
      /*max_buffer_bytes=*/static_cast<size_t>(1024 * 2), // 2 KB buffer to force streaming
      std::chrono::milliseconds::zero(),
      artifact_size);

  // Launch streaming load to GPU
  auto src_or = loader.open_source();
  REQUIRE(src_or.ok());
  auto fut = memmgr->load_async_from_source(std::move(*src_or), MemoryLocation::GPU, /*concurrency=*/2);
  REQUIRE(fut.valid());
  auto st = fut.get();
  LOG(INFO) << "Streaming load result: " << st.message();
  REQUIRE(st.ok());

  // Verify GPU memory state
  REQUIRE(memmgr->get_state(MemoryLocation::GPU) == MemoryState::LOADED);

  // Copy back to host and verify content
  auto gpu_ptrs = memmgr->get_pointer(MemoryLocation::GPU);
  REQUIRE(gpu_ptrs.size() == 1);
  void* gpu_ptr = gpu_ptrs[0];
  REQUIRE(gpu_ptr != nullptr);

  std::vector<char> host_buf(artifact_size);
  absl::Status copy_status = tensorcast::cuda::memcpy(host_buf.data(), gpu_ptr, artifact_size, cudaMemcpyDeviceToHost);
  REQUIRE(copy_status.ok());
  REQUIRE(host_buf == combined);

  // Cleanup
  fs::remove_all(base);
}
