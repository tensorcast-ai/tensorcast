// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <vector>

#include "core/common/cuda_api.h"

TEST_CASE("CUDA API abstraction layer", "[cuda]") {
  namespace cuda = stepcast::cuda;

  SECTION("Basic device operations") {
    // Check if CUDA is available
    bool cuda_available = cuda::is_available();
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
    size_t free_before = 0, total = 0;
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

    uint8_t* host_bytes = static_cast<uint8_t*>(host_ptr);
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
}