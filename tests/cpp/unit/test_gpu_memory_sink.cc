// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "core/common/cuda/cuda_api.h"
#include "core/common/device_guard.h"
#include "core/store/loader/gpu_memory_sink.h"

using namespace stepcast::store::loader;
using namespace stepcast::common;

class GPUMemoryFixture {
 public:
  GPUMemoryFixture(size_t size = 10 * 1024 * 1024) : size_(size) {
    // Check if CUDA is available
    int device_count = 0;
    auto status = stepcast::cuda::get_device_count(&device_count);
    if (!status.ok() || device_count == 0) {
      cuda_available_ = false;
      return;
    }
    cuda_available_ = true;

    // Allocate GPU memory
    DeviceGuard guard(0);
    status = stepcast::cuda::malloc(&gpu_ptr_, size_);
    if (!status.ok()) {
      gpu_ptr_ = nullptr;
      cuda_available_ = false;
    }

    // Allocate pinned host memory for testing
    status = stepcast::cuda::malloc_host(&host_ptr_, size_);
    if (!status.ok()) {
      host_ptr_ = nullptr;
      if (gpu_ptr_) {
        stepcast::cuda::free(gpu_ptr_);
        gpu_ptr_ = nullptr;
      }
      cuda_available_ = false;
    }
  }

  ~GPUMemoryFixture() {
    if (gpu_ptr_) {
      stepcast::cuda::free(gpu_ptr_);
    }
    if (host_ptr_) {
      stepcast::cuda::free_host(host_ptr_);
    }
  }

  bool is_cuda_available() const { return cuda_available_; }
  void* gpu_ptr() const { return gpu_ptr_; }
  void* host_ptr() const { return host_ptr_; }
  size_t size() const { return size_; }

  void fill_host_buffer(char pattern = 'A') {
    if (!host_ptr_) return;
    
    char* buffer = static_cast<char*>(host_ptr_);
    for (size_t i = 0; i < size_; ++i) {
      buffer[i] = static_cast<char>(pattern + (i % 26));
    }
  }

  bool verify_gpu_content(size_t bytes, char pattern = 'A') {
    if (!gpu_ptr_ || !host_ptr_) return false;

    std::vector<char> temp(bytes);
    
    // Copy from GPU to host for verification
    auto status = stepcast::cuda::memcpy(temp.data(), gpu_ptr_, bytes, cudaMemcpyDeviceToHost);
    if (!status.ok()) return false;

    // Verify content
    for (size_t i = 0; i < bytes; ++i) {
      if (temp[i] != static_cast<char>(pattern + (i % 26))) {
        return false;
      }
    }
    return true;
  }

 private:
  size_t size_;
  void* gpu_ptr_ = nullptr;
  void* host_ptr_ = nullptr;
  bool cuda_available_ = false;
};

TEST_CASE("GPUMemorySink basic functionality", "[gpu_memory_sink]") {
  GPUMemoryFixture fixture(10 * 1024 * 1024);  // 10MB
  
  if (!fixture.is_cuda_available()) {
    SKIP("CUDA not available");
  }

  SECTION("Simple write") {
    GPUMemorySink::Options options;
    options.gpu_base_ptr = fixture.gpu_ptr();
    options.total_size = fixture.size();
    options.device_id = 0;

    GPUMemorySink sink(options);

    // Prepare test data
    size_t write_size = 1024;
    fixture.fill_host_buffer('A');

    // Write to GPU
    auto status = sink.write(fixture.host_ptr(), write_size);
    REQUIRE(status.ok());

    // Close to ensure transfer completes
    status = sink.close();
    REQUIRE(status.ok());

    // Verify content
    REQUIRE(fixture.verify_gpu_content(write_size, 'A'));
  }

  SECTION("Multiple writes") {
    GPUMemorySink::Options options;
    options.gpu_base_ptr = fixture.gpu_ptr();
    options.total_size = fixture.size();
    options.device_id = 0;

    GPUMemorySink sink(options);

    // Write multiple chunks
    size_t chunk_size = 1024;
    size_t num_chunks = 10;
    fixture.fill_host_buffer('B');

    for (size_t i = 0; i < num_chunks; ++i) {
      auto status = sink.write(fixture.host_ptr(), chunk_size);
      REQUIRE(status.ok());
    }

    // Close and verify
    auto status = sink.close();
    REQUIRE(status.ok());

    REQUIRE(fixture.verify_gpu_content(chunk_size * num_chunks, 'B'));
  }

  SECTION("Write full buffer") {
    GPUMemorySink::Options options;
    options.gpu_base_ptr = fixture.gpu_ptr();
    options.total_size = 1024 * 1024;  // 1MB
    options.device_id = 0;

    GPUMemorySink sink(options);

    // Fill and write entire buffer
    fixture.fill_host_buffer('C');
    
    auto status = sink.write(fixture.host_ptr(), options.total_size);
    REQUIRE(status.ok());

    status = sink.close();
    REQUIRE(status.ok());

    REQUIRE(fixture.verify_gpu_content(options.total_size, 'C'));
  }

  SECTION("Async transfer completion") {
    GPUMemorySink::Options options;
    options.gpu_base_ptr = fixture.gpu_ptr();
    options.total_size = fixture.size();
    options.device_id = 0;

    GPUMemorySink sink(options);

    // Write large amount to test async
    size_t large_write = 5 * 1024 * 1024;  // 5MB
    fixture.fill_host_buffer('D');

    auto status = sink.write(fixture.host_ptr(), large_write);
    REQUIRE(status.ok());

    // Close should wait for async transfer
    status = sink.close();
    REQUIRE(status.ok());

    // Verify all data transferred
    REQUIRE(fixture.verify_gpu_content(large_write, 'D'));
  }
}

TEST_CASE("GPUMemorySink error handling", "[gpu_memory_sink]") {
  SECTION("Null GPU pointer") {
    GPUMemorySink::Options options;
    options.gpu_base_ptr = nullptr;
    options.total_size = 1024;
    options.device_id = 0;

    GPUMemorySink sink(options);

    std::vector<char> data(1024);
    auto status = sink.write(data.data(), 1024);
    
    REQUIRE(!status.ok());
    REQUIRE(status.code() == absl::StatusCode::kInvalidArgument);
  }

  SECTION("Write exceeds total size") {
    GPUMemoryFixture fixture(1024);
    
    if (!fixture.is_cuda_available()) {
      SKIP("CUDA not available");
    }

    GPUMemorySink::Options options;
    options.gpu_base_ptr = fixture.gpu_ptr();
    options.total_size = 1024;
    options.device_id = 0;

    GPUMemorySink sink(options);

    // Try to write more than total size
    std::vector<char> data(2048);
    auto status = sink.write(data.data(), 2048);
    
    REQUIRE(!status.ok());
    REQUIRE(status.message().find("exceed total GPU memory size") != std::string::npos);
  }

  SECTION("Invalid device ID") {
    GPUMemoryFixture fixture(1024);
    
    int device_count = 0;
    auto status = stepcast::cuda::get_device_count(&device_count);
    if (!status.ok() || device_count == 0) {
      SKIP("CUDA not available");
    }

    GPUMemorySink::Options options;
    options.gpu_base_ptr = fixture.gpu_ptr();
    options.total_size = 1024;
    options.device_id = 999;  // Invalid device

    GPUMemorySink sink(options);

    std::vector<char> data(1024);
    status = sink.write(data.data(), 1024);
    
    // Should fail due to invalid device
    REQUIRE(!status.ok());
  }

  SECTION("Multiple writes exceeding limit") {
    GPUMemoryFixture fixture(2048);
    
    if (!fixture.is_cuda_available()) {
      SKIP("CUDA not available");
    }

    GPUMemorySink::Options options;
    options.gpu_base_ptr = fixture.gpu_ptr();
    options.total_size = 2048;
    options.device_id = 0;

    GPUMemorySink sink(options);

    // First write succeeds
    std::vector<char> data(1024);
    auto status = sink.write(data.data(), 1024);
    REQUIRE(status.ok());

    // Second write within limit
    status = sink.write(data.data(), 1024);
    REQUIRE(status.ok());

    // Third write exceeds limit
    status = sink.write(data.data(), 1);
    REQUIRE(!status.ok());
  }
}

TEST_CASE("GPUMemorySink validation", "[gpu_memory_sink]") {
  GPUMemoryFixture fixture(10 * 1024 * 1024);
  
  if (!fixture.is_cuda_available()) {
    SKIP("CUDA not available");
  }

  SECTION("Verify total bytes written on close") {
    GPUMemorySink::Options options;
    options.gpu_base_ptr = fixture.gpu_ptr();
    options.total_size = fixture.size();
    options.device_id = 0;

    GPUMemorySink sink(options);

    // Write specific amount
    size_t expected_total = 5 * 1024 * 1024;  // 5MB
    size_t chunk_size = 1024 * 1024;  // 1MB chunks
    fixture.fill_host_buffer('E');

    for (size_t written = 0; written < expected_total; written += chunk_size) {
      auto status = sink.write(fixture.host_ptr(), chunk_size);
      REQUIRE(status.ok());
    }

    // Close should complete all transfers
    auto status = sink.close();
    REQUIRE(status.ok());
    
    // The sink should have written exactly expected_total bytes
    // We verify this by checking GPU content
    REQUIRE(fixture.verify_gpu_content(expected_total, 'E'));
  }

  SECTION("Close without writes") {
    GPUMemorySink::Options options;
    options.gpu_base_ptr = fixture.gpu_ptr();
    options.total_size = fixture.size();
    options.device_id = 0;

    GPUMemorySink sink(options);

    // Close without any writes
    auto status = sink.close();
    REQUIRE(status.ok());
  }

  SECTION("Stream synchronization on close") {
    GPUMemorySink::Options options;
    options.gpu_base_ptr = fixture.gpu_ptr();
    options.total_size = fixture.size();
    options.device_id = 0;

    GPUMemorySink sink(options);

    // Write data
    size_t write_size = 2 * 1024 * 1024;  // 2MB
    fixture.fill_host_buffer('F');
    
    auto status = sink.write(fixture.host_ptr(), write_size);
    REQUIRE(status.ok());

    // Close should synchronize the stream
    status = sink.close();
    REQUIRE(status.ok());

    // After close, all data should be transferred
    REQUIRE(fixture.verify_gpu_content(write_size, 'F'));
  }
}

TEST_CASE("GPUMemorySink proposed fix validation", "[gpu_memory_sink]") {
  GPUMemoryFixture fixture(10 * 1024 * 1024);
  
  if (!fixture.is_cuda_available()) {
    SKIP("CUDA not available");
  }

  SECTION("Validate all data written before close") {
    // This test validates the proposed fix:
    // close() should check that current_offset_ == options_.total_size
    
    GPUMemorySink::Options options;
    options.gpu_base_ptr = fixture.gpu_ptr();
    options.total_size = 1024 * 1024;  // 1MB expected
    options.device_id = 0;

    GPUMemorySink sink(options);

    // Write only partial data
    size_t partial_write = 512 * 1024;  // 512KB (half of expected)
    fixture.fill_host_buffer('G');
    
    auto status = sink.write(fixture.host_ptr(), partial_write);
    REQUIRE(status.ok());

    // Close should detect incomplete transfer
    // After implementing the fix, this should either:
    // 1. Return an error indicating incomplete transfer
    // 2. Log a warning about incomplete data
    status = sink.close();
    
    // For now, it succeeds (current behavior)
    // After fix, we might want to change this behavior
    REQUIRE(status.ok());
    
    // Document expected behavior after fix
    // Option 1: Return error if not all expected data written
    // Option 2: Log warning but still succeed
    // Option 3: Track this in metrics for monitoring
  }

  SECTION("Validate exact amount written") {
    GPUMemorySink::Options options;
    options.gpu_base_ptr = fixture.gpu_ptr();
    options.total_size = 1024 * 1024;  // 1MB expected
    options.device_id = 0;

    GPUMemorySink sink(options);

    // Write exact amount
    fixture.fill_host_buffer('H');
    
    auto status = sink.write(fixture.host_ptr(), options.total_size);
    REQUIRE(status.ok());

    // Close should succeed with exact amount
    status = sink.close();
    REQUIRE(status.ok());
    
    // Verify all data transferred
    REQUIRE(fixture.verify_gpu_content(options.total_size, 'H'));
  }
}