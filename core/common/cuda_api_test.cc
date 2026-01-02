// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <cstring>
#include <vector>

#include "core/common/cuda_api.h"

TEST_CASE("CUDA API abstraction layer", "[cuda]") {
  namespace cuda = tensorcast::cuda;

  const bool cuda_available = cuda::is_available();
  if (!cuda_available) {
    WARN("CUDA not available; skipping CUDA API tests.");
    return;
  }

  SECTION("Basic device operations") {
    // Check if CUDA is available
    REQUIRE(cuda_available == true);

    // Check if using fake backend
    bool fake_backend = cuda::is_fake();
    INFO("Using fake CUDA backend: " << fake_backend);

    // Get device count
    int device_count = 0;
    auto status = cuda::get_device_count(&device_count);
    REQUIRE(status.ok());
    REQUIRE(device_count >= 1);

    // Set and get device
    status = cuda::set_device(0);
    REQUIRE(status.ok());

    int current_device = -1;
    status = cuda::get_device(&current_device);
    REQUIRE(status.ok());
    REQUIRE(current_device == 0);

    // Get device name
    auto name_or = cuda::get_device_name(0);
    REQUIRE(name_or.ok());
    INFO("Device name: " << name_or.value());
  }

  SECTION("Memory allocation and deallocation") {
    void* ptr = nullptr;
    size_t size = 1024 * 1024; // 1MB

    // Allocate memory
    auto status = cuda::malloc(&ptr, size);
    REQUIRE(status.ok());
    REQUIRE(ptr != nullptr);

    // Get memory info
    size_t free_before = 0;
    size_t total = 0;
    status = cuda::get_memory_info(&free_before, &total, 0);
    REQUIRE(status.ok());
    REQUIRE(total > 0);

    // Free memory
    status = cuda::free(ptr);
    REQUIRE(status.ok());

    // Check memory info after free
    size_t free_after = 0;
    status = cuda::get_memory_info(&free_after, &total, 0);
    REQUIRE(status.ok());

    // In fake backend, we might not see exact memory changes
    if (!cuda::is_fake()) {
      REQUIRE(free_after >= free_before);
    }
  }

  SECTION("Pinned memory allocation") {
    void* ptr = nullptr;
    size_t size = 1024 * 1024; // 1MB

    // Allocate pinned memory
    auto status = cuda::malloc_host(&ptr, size);
    REQUIRE(status.ok());
    REQUIRE(ptr != nullptr);

    // Free pinned memory
    status = cuda::free_host(ptr);
    REQUIRE(status.ok());
  }

  SECTION("Memory operations") {
    size_t size = 1024;
    void* gpu_ptr = nullptr;
    void* host_ptr = nullptr;

    // Allocate GPU and host memory
    auto status = cuda::malloc(&gpu_ptr, size);
    REQUIRE(status.ok());

    status = cuda::malloc_host(&host_ptr, size);
    REQUIRE(status.ok());

    // Initialize host memory
    std::vector<uint8_t> test_data(size);
    for (size_t i = 0; i < size; ++i) {
      test_data[i] = static_cast<uint8_t>(i % 256);
    }
    std::memcpy(host_ptr, test_data.data(), size);

    // Copy to GPU
    status = cuda::memcpy(gpu_ptr, host_ptr, size, cudaMemcpyHostToDevice);
    REQUIRE(status.ok());

    // Clear host buffer
    std::memset(host_ptr, 0, size);

    // Copy back from GPU
    status = cuda::memcpy(host_ptr, gpu_ptr, size, cudaMemcpyDeviceToHost);
    REQUIRE(status.ok());

    // Verify data
    bool data_matches = std::memcmp(host_ptr, test_data.data(), size) == 0;
    REQUIRE(data_matches);

    // Test memset
    status = cuda::memset(gpu_ptr, 42, size);
    REQUIRE(status.ok());

    status = cuda::memcpy(host_ptr, gpu_ptr, size, cudaMemcpyDeviceToHost);
    REQUIRE(status.ok());

    auto* host_bytes = static_cast<uint8_t*>(host_ptr);
    bool all_42 = true;
    for (size_t i = 0; i < size; ++i) {
      if (host_bytes[i] != 42) {
        all_42 = false;
        break;
      }
    }
    REQUIRE(all_42);

    // Cleanup
    status = cuda::free(gpu_ptr);
    REQUIRE(status.ok());

    status = cuda::free_host(host_ptr);
    REQUIRE(status.ok());
  }

  SECTION("IPC handle operations") {
    void* ptr = nullptr;
    size_t size = 1024 * 1024; // 1MB

    // Allocate memory
    auto status = cuda::malloc(&ptr, size);
    REQUIRE(status.ok());

    // Get IPC handle
    std::string handle;
    status = cuda::get_ipc_handle(ptr, &handle);
    REQUIRE(status.ok());
    REQUIRE(!handle.empty());
    INFO("IPC handle size: " << handle.size());

    // In a real test, we would open this handle in another process
    // For now, test that we can open it in the same process (fake backend supports this)
    if (cuda::is_fake()) {
      void* opened_ptr = nullptr;
      status = cuda::open_ipc_handle(handle, &opened_ptr);
      REQUIRE(status.ok());
      REQUIRE(opened_ptr == ptr); // In fake backend, should get same pointer

      // Close IPC handle
      status = cuda::close_ipc_handle(opened_ptr);
      REQUIRE(status.ok());
    }

    // Free original allocation
    status = cuda::free(ptr);
    REQUIRE(status.ok());
  }

  SECTION("CUDA VMM (Driver API) wrappers") {
    // This unit test is primarily for FakeCuda. The real backend may require a
    // functional NVIDIA driver (libcuda.so) and compatible GPU.
    if (!cuda::is_fake()) {
      SUCCEED("Skipping CUDA VMM wrapper test on real backend");
      return;
    }

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;

    size_t granularity = 0;
    REQUIRE(cuda::cu_mem_get_allocation_granularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM).ok());
    REQUIRE(granularity > 0);

    CUdeviceptr reserved = 0;
    REQUIRE(cuda::cu_mem_address_reserve(&reserved, granularity, granularity, 0, 0).ok());
    REQUIRE(reserved != 0);

    CUmemGenericAllocationHandle handle = 0;
    REQUIRE(cuda::cu_mem_create(&handle, granularity, &prop, 0).ok());
    REQUIRE(handle != 0);

    REQUIRE(cuda::cu_mem_map(reserved, granularity, 0, handle, 0).ok());

    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    REQUIRE(cuda::cu_mem_set_access(reserved, granularity, &access, 1).ok());

    REQUIRE(cuda::cu_mem_unmap(reserved, granularity).ok());
    REQUIRE(cuda::cu_mem_release(handle).ok());
    REQUIRE(cuda::cu_mem_address_free(reserved, granularity).ok());
  }

  SECTION("Synchronization operations") {
    // These should always succeed
    auto status = cuda::device_synchronize();
    REQUIRE(status.ok());

    // Create a stream for synchronization test
    cudaStream_t stream = nullptr;
    status = cuda::stream_create(&stream);
    REQUIRE(status.ok());

    status = cuda::stream_synchronize(stream);
    REQUIRE(status.ok());

    // Clean up
    status = cuda::stream_destroy(stream);
    REQUIRE(status.ok());
  }

  SECTION("Error handling") {
    // Test invalid device
    auto status = cuda::set_device(999);
    REQUIRE(!status.ok());

    // Test null pointer
    status = cuda::free(nullptr);
    REQUIRE(status.ok()); // Should succeed (no-op)

    // Test zero allocation size - this is actually valid in CUDA
    void* ptr = nullptr;
    status = cuda::malloc(&ptr, 0);
    REQUIRE(status.ok());
    if (status.ok() && ptr != nullptr) {
      // Clean up the allocation
      status = cuda::free(ptr);
      REQUIRE(status.ok());
    }
  }

  SECTION("Asynchronous stream operations") {
    constexpr size_t kSize = 256 * 1024;
    void* gpu_ptr = nullptr;
    void* host_ptr = nullptr;

    REQUIRE(cuda::malloc(&gpu_ptr, kSize).ok());
    REQUIRE(cuda::malloc_host(&host_ptr, kSize).ok());

    auto* host_bytes = static_cast<uint8_t*>(host_ptr);
    for (size_t i = 0; i < kSize; ++i) {
      host_bytes[i] = static_cast<uint8_t>((i * 7) % 251);
    }

    cudaStream_t stream = nullptr;
    REQUIRE(cuda::stream_create(&stream).ok());

    std::atomic<int> callback_count{0};
    REQUIRE(cuda::memset_async(gpu_ptr, 0, kSize, stream).ok());
    REQUIRE(cuda::memcpy_async(gpu_ptr, host_ptr, kSize, cudaMemcpyHostToDevice, stream).ok());
    REQUIRE(
        cuda::stream_add_callback(
            stream,
            [](cudaStream_t, cudaError_t, void* ctx) {
              auto* counter = static_cast<std::atomic<int>*>(ctx);
              counter->fetch_add(1, std::memory_order_relaxed);
            },
            &callback_count,
            0)
            .ok());

    REQUIRE(cuda::stream_synchronize(stream).ok());
    REQUIRE(callback_count.load(std::memory_order_relaxed) == 1);

    std::vector<uint8_t> verify(kSize);
    REQUIRE(cuda::memcpy(verify.data(), gpu_ptr, kSize, cudaMemcpyDeviceToHost).ok());
    REQUIRE(std::memcmp(verify.data(), host_ptr, kSize) == 0);

    // Ensure device-level synchronize drains all streams
    std::memset(host_ptr, 0, kSize);
    REQUIRE(cuda::memcpy_async(host_ptr, gpu_ptr, kSize, cudaMemcpyDeviceToHost, stream).ok());
    REQUIRE(cuda::device_synchronize().ok());
    REQUIRE(std::memcmp(host_ptr, verify.data(), kSize) == 0);

    REQUIRE(cuda::stream_destroy(stream).ok());
    REQUIRE(cuda::free(gpu_ptr).ok());
    REQUIRE(cuda::free_host(host_ptr).ok());
  }
}
