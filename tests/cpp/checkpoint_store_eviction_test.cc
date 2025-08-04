// Copyright (c) 2025, StepCast Team. All rights reserved.

// New unit-test for CheckpointStore CPU pinned-memory eviction scenario
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "catch2/catch_test_macros.hpp"

#include "core/common/cuda_api.h"
#include "core/store/checkpoint_store.h"
#include "core/store/checkpoint_store_options.h"
#include "core/store/loading/loading_spec.h"

namespace fs = std::filesystem;
using namespace stepcast::store;
using stepcast::DeviceType;
using stepcast::store::InstanceKey;

namespace {
// Helper that creates a dummy binary file of the requested size.
void create_dummy_file(const fs::path& path, size_t size_bytes) {
  std::error_code ec;
  fs::create_directories(path.parent_path(), ec);
  REQUIRE(!ec);

  std::vector<char> buf(size_bytes, 'X');
  std::ofstream out(path, std::ios::binary);
  REQUIRE(out.good());
  out.write(buf.data(), size_bytes);
  out.close();
  REQUIRE(out.good());
}

constexpr size_t operator"" _MiB(unsigned long long mib) {
  return static_cast<size_t>(mib) * 1024ULL * 1024ULL;
}
} // namespace

TEST_CASE(
    "CheckpointStore CPU eviction frees pinned memory and allows second model to load",
    "[checkpoint_store][eviction][cpu]") {
  const size_t kPoolSize = 4_MiB; // Total pinned memory pool
  const size_t kChunkSize = 1_MiB; // Pool chunk granularity
  const size_t kModelSize = 3_MiB; // Each model ~3 MiB → needs 3 chunks

  // Temporary directory for disk source files.
  fs::path tmp_root = fs::temp_directory_path() / "checkpoint_store_eviction_test";
  std::error_code ec;
  fs::remove_all(tmp_root, ec);
  fs::create_directories(tmp_root, ec);
  REQUIRE(!ec);

  // -----------------------------------------------------------------------
  // 1. Create two dummy model files.
  // -----------------------------------------------------------------------
  const fs::path modelA_path = tmp_root / "modelA";
  const fs::path modelB_path = tmp_root / "modelB";
  create_dummy_file(modelA_path / "tensor.data_0", kModelSize);
  create_dummy_file(modelB_path / "tensor.data_0", kModelSize);

  // -----------------------------------------------------------------------
  // 2. Instantiate CheckpointStore with a very small pinned pool.
  // -----------------------------------------------------------------------
  CheckpointStoreOptions opts;
  opts.storage_path = tmp_root.string();
  opts.memory_pool_size = kPoolSize;
  opts.chunk_size = kChunkSize;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);

  CheckpointStore store(opts);

  // Convenience lambda for a CPU DeviceKey.
  auto make_cpu_key = []() { return DeviceKey{DeviceType::CPU, /*ordinal=*/-1, /*uuid=*/""}; };

  // -----------------------------------------------------------------------
  // 3. Load first model – should succeed and occupy most of the pool.
  // -----------------------------------------------------------------------
  // 3a) First model – should succeed and occupy most of the pool.
  auto handleA_or = store.prepare("modelA", make_cpu_key());
  REQUIRE(handleA_or.ok());
  auto handleA = std::move(handleA_or).value();
  REQUIRE(handleA.ready_future.get().ok());
  {
    DeviceKey cpu{DeviceType::CPU, -1, ""};
    InstanceKey key{"modelA", cpu, 0};
    REQUIRE(store.get_instance_state(key, DeviceType::CPU) == MemoryState::LOADED);
  }

  // At this point, available pinned memory should be < kChunkSize.
  REQUIRE(store.get_available_memory() < kChunkSize);

  // -----------------------------------------------------------------------
  // 4. Load second model – allocation initially fails, eviction frees modelA.
  // -----------------------------------------------------------------------
  // 4) Load second model – this may trigger eviction of modelA.
  auto handleB_or = store.prepare("modelB", make_cpu_key());
  REQUIRE(handleB_or.ok());
  auto handleB = std::move(handleB_or).value();
  REQUIRE(handleB.ready_future.get().ok());
  {
    DeviceKey cpu{DeviceType::CPU, -1, ""};
    InstanceKey key{"modelB", cpu, 0};
    REQUIRE(store.get_instance_state(key, DeviceType::CPU) == MemoryState::LOADED);
  }

  // ModelA should now have been evicted from CPU memory (UNALLOCATED or less).
  auto stateA = store.get_instance_state(InstanceKey{"modelA", DeviceKey{DeviceType::CPU, -1, ""}, 0}, DeviceType::CPU);
  REQUIRE(
      (stateA == MemoryState::UNALLOCATED || stateA == MemoryState::ALLOCATED || stateA == MemoryState::UNINITIALIZED));

  // -----------------------------------------------------------------------
  // 5. Cleanup.
  // -----------------------------------------------------------------------
  REQUIRE(store.clear_mem() == 0);
  fs::remove_all(tmp_root, ec);
}

TEST_CASE(
    "CheckpointStore GPU eviction frees GPU memory and allows second model to load",
    "[checkpoint_store][eviction][gpu]") {
  // Skip test if CUDA is not available
  int device_count = 0;
  auto count_status = stepcast::cuda::get_device_count(&device_count);
  if (!count_status.ok() || device_count == 0) {
    SKIP("CUDA not available, skipping GPU eviction test");
  }

  // Get available GPU memory to size our test appropriately
  size_t free_gpu_mem = 0, total_gpu_mem = 0;
  auto mem_status = stepcast::cuda::get_memory_info(&free_gpu_mem, &total_gpu_mem, 0);
  REQUIRE(mem_status.ok());

  // Use conservative sizes to avoid OOM on smaller GPUs
  const size_t kPoolSize = 128_MiB; // CPU pinned memory pool
  const size_t kChunkSize = 8_MiB; // Pool chunk granularity
  const size_t kModelSize = std::min(64_MiB, free_gpu_mem / 4); // Each model uses 1/4 of free GPU memory

  // Temporary directory for disk source files
  fs::path tmp_root = fs::temp_directory_path() / "checkpoint_store_gpu_eviction_test";
  std::error_code ec;
  fs::remove_all(tmp_root, ec);
  fs::create_directories(tmp_root, ec);
  REQUIRE(!ec);

  // Create three dummy model files
  const fs::path modelA_path = tmp_root / "modelA";
  const fs::path modelB_path = tmp_root / "modelB";
  const fs::path modelC_path = tmp_root / "modelC";
  create_dummy_file(modelA_path / "tensor.data_0", kModelSize);
  create_dummy_file(modelB_path / "tensor.data_0", kModelSize);
  create_dummy_file(modelC_path / "tensor.data_0", kModelSize);

  // Instantiate CheckpointStore
  CheckpointStoreOptions opts;
  opts.storage_path = tmp_root.string();
  opts.memory_pool_size = kPoolSize;
  opts.chunk_size = kChunkSize;
  opts.num_thread = 4;
  opts.pinned_memory_timeout = std::chrono::milliseconds(5000);

  CheckpointStore store(opts);

  // Helper for GPU DeviceKey
  auto make_gpu0_key = []() { return DeviceKey{DeviceType::GPU, /*ordinal=*/0, /*uuid=*/""}; };

  // Load first two models to GPU 0 – should succeed.
  auto gpu_handleA_or = store.prepare("modelA", make_gpu0_key());
  REQUIRE(gpu_handleA_or.ok());
  auto gpu_handleA = std::move(gpu_handleA_or).value();
  REQUIRE(gpu_handleA.ready_future.get().ok());

  auto gpu_handleB_or = store.prepare("modelB", make_gpu0_key());
  REQUIRE(gpu_handleB_or.ok());
  auto gpu_handleB = std::move(gpu_handleB_or).value();
  REQUIRE(gpu_handleB.ready_future.get().ok());

  // Check available GPU memory - should be significantly reduced
  size_t free_after_two = 0;
  mem_status = stepcast::cuda::get_memory_info(&free_after_two, &total_gpu_mem, 0);
  REQUIRE(mem_status.ok());
  REQUIRE(free_after_two < free_gpu_mem - (2 * kModelSize));

  // Loading third model should trigger eviction of least-recently-used (modelA).
  auto gpu_handleC_or = store.prepare("modelC", make_gpu0_key());
  REQUIRE(gpu_handleC_or.ok());
  auto gpu_handleC = std::move(gpu_handleC_or).value();
  REQUIRE(gpu_handleC.ready_future.get().ok());

  // ModelA should have been evicted from GPU
  auto stateA_gpu =
      store.get_instance_state(InstanceKey{"modelA", DeviceKey{DeviceType::GPU, 0, ""}, 0}, DeviceType::GPU);
  REQUIRE((stateA_gpu == MemoryState::UNALLOCATED || stateA_gpu == MemoryState::UNINITIALIZED));

  // ModelB should still be loaded (it was accessed more recently than A)
  {
    DeviceKey gpu0{DeviceType::GPU, 0, ""};
    InstanceKey key{"modelB", gpu0, 0};
    REQUIRE(store.get_instance_state(key, DeviceType::GPU) == MemoryState::LOADED);
  }

  // Cleanup
  REQUIRE(store.clear_mem() == 0);
  fs::remove_all(tmp_root, ec);
}

TEST_CASE("CheckpointStore GPU-to-GPU copy with eviction", "[checkpoint_store][eviction][gpu][p2p]") {
  // Skip test if CUDA is not available or only one GPU
  int device_count = 0;
  auto count_status = stepcast::cuda::get_device_count(&device_count);
  if (!count_status.ok() || device_count < 2) {
    SKIP("Need at least 2 GPUs for GPU-to-GPU copy test");
  }

  // Conservative sizes
  const size_t kPoolSize = 128_MiB;
  const size_t kChunkSize = 8_MiB;
  const size_t kModelSize = 32_MiB;

  // Temporary directory
  fs::path tmp_root = fs::temp_directory_path() / "checkpoint_store_p2p_eviction_test";
  std::error_code ec;
  fs::remove_all(tmp_root, ec);
  fs::create_directories(tmp_root, ec);
  REQUIRE(!ec);

  // Create model files
  create_dummy_file(tmp_root / "modelX" / "tensor.data_0", kModelSize);
  create_dummy_file(tmp_root / "modelY" / "tensor.data_0", kModelSize);

  // Instantiate CheckpointStore
  CheckpointStoreOptions opts;
  opts.storage_path = tmp_root.string();
  opts.memory_pool_size = kPoolSize;
  opts.chunk_size = kChunkSize;
  opts.num_thread = 4;

  CheckpointStore store(opts);

  // Load modelX to GPU 0
  {
    DeviceKey gpu0{DeviceType::GPU, 0, ""};
    auto handle = store.prepare("modelX", gpu0);
    REQUIRE(handle.ok());
    REQUIRE(handle->wait_ready(std::chrono::seconds(30)).ok());
  }

  // Load modelY to GPU 0 (to consume memory)
  {
    DeviceKey gpu0{DeviceType::GPU, 0, ""};
    auto handle = store.prepare("modelY", gpu0);
    REQUIRE(handle.ok());
    REQUIRE(handle->wait_ready(std::chrono::seconds(30)).ok());
  }

  // Now try to copy modelX to GPU 1 - this will trigger P2P copy
  // If GPU 1 doesn't have enough memory, it should evict something
  {
    DeviceKey gpu1{DeviceType::GPU, 1, ""};
    auto handle = store.prepare("modelX", gpu1);
    REQUIRE(handle.ok());
    auto status = handle->wait_ready(std::chrono::seconds(30));
    REQUIRE(status.ok());

    // Verify modelX is now on both GPUs
    auto devices = store.get_loaded_devices("modelX");
    bool on_gpu0 = false, on_gpu1 = false;
    for (const auto& dev : devices) {
      if (dev.type == DeviceType::GPU) {
        if (dev.ordinal == 0)
          on_gpu0 = true;
        if (dev.ordinal == 1)
          on_gpu1 = true;
      }
    }
    REQUIRE(on_gpu0);
    REQUIRE(on_gpu1);
  }

  // Cleanup
  REQUIRE(store.clear_mem() == 0);
  fs::remove_all(tmp_root, ec);
}