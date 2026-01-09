// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstring>

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
