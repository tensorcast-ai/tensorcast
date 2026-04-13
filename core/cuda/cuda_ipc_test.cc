// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstring>

#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_ipc.h"

TEST_CASE("CUDA IPC handle bytes conversions are stable", "[cuda][ipc]") {
  namespace cuda = tensorcast::cuda;

  REQUIRE(cuda::IpcHandleBytes::kHandleSize == sizeof(cudaIpcMemHandle_t));

  cuda::IpcHandleBytes empty;
  REQUIRE_FALSE(empty.is_valid());

  auto non_zero = empty;
  non_zero.bytes[0] = 1;
  REQUIRE(non_zero.is_valid());

  std::array<std::uint8_t, cuda::IpcHandleBytes::kHandleSize> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<std::uint8_t>(i);
  }

  cudaIpcMemHandle_t native{};
  std::memcpy(&native, pattern.data(), pattern.size());

  auto bytes = cuda::IpcHandleBytes::from_native(native);
  REQUIRE(bytes.bytes == pattern);

  auto roundtrip = bytes.to_native();
  REQUIRE(std::memcmp(&roundtrip, &native, sizeof(native)) == 0);

  auto view = bytes.as_string_view();
  REQUIRE(view.size() == pattern.size());
  REQUIRE(std::memcmp(view.data(), pattern.data(), pattern.size()) == 0);
}

TEST_CASE("CUDA IPC mapping close uses the import device and restores caller device", "[cuda][ipc]") {
  namespace cuda = tensorcast::cuda;

  if (!cuda::is_available()) {
    WARN("CUDA not available; skipping CUDA IPC close test.");
    return;
  }

  int device_count = 0;
  auto device_count_status = cuda::get_device_count(&device_count);
  REQUIRE(device_count_status.ok());
  if (device_count < 2) {
    WARN("Fewer than 2 CUDA devices available; skipping cross-device CUDA IPC close test.");
    return;
  }

  constexpr size_t kAllocationBytes = 4096;
  constexpr int kExportDevice = 0;
  constexpr int kCallerDevice = 1;

  REQUIRE(cuda::set_device(kExportDevice).ok());
  void* allocation = nullptr;
  REQUIRE(cuda::malloc(&allocation, kAllocationBytes).ok());

  cudaIpcMemHandle_t handle{};
  auto handle_status = cuda::get_ipc_mem_handle(&handle, allocation);
  REQUIRE(handle_status.ok());

  auto mapping_or = cuda::IpcMapping::open(handle);
  REQUIRE(mapping_or.ok());
  cuda::IpcMapping mapping = std::move(*mapping_or);

  REQUIRE(cuda::set_device(kCallerDevice).ok());
  auto close_status = mapping.close();
  REQUIRE(close_status.ok());

  int current_device = -1;
  REQUIRE(cuda::get_device(&current_device).ok());
  REQUIRE(current_device == kCallerDevice);

  REQUIRE(cuda::set_device(kExportDevice).ok());
  REQUIRE(cuda::free(allocation).ok());
}
