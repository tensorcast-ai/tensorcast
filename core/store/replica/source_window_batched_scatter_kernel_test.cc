// Copyright (c) 2026, TensorCast Team.

#include "core/store/replica/source_window_batched_scatter_kernel.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"
#include "core/common/memory/cuda_memory.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::store::replica {
namespace {

bool has_real_cuda_device() {
  if (cuda::is_fake()) {
    return false;
  }
  int device_count = 0;
  if (!cuda::get_device_count(&device_count).ok()) {
    return false;
  }
  return device_count > 0;
}

int real_cuda_device_count() {
  if (cuda::is_fake()) {
    return 0;
  }
  int device_count = 0;
  if (!cuda::get_device_count(&device_count).ok()) {
    return 0;
  }
  return device_count;
}

uint64_t device_address(void* base, uint64_t offset) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<std::uint8_t*>(base) + offset));
}

void apply_expected_copy(
    const std::vector<std::uint8_t>& src,
    std::vector<std::uint8_t>& expected,
    uint64_t src_offset,
    uint64_t dst_offset,
    uint64_t row_bytes,
    uint64_t row_count,
    uint64_t source_stride_bytes,
    uint64_t target_stride_bytes) {
  for (uint64_t row = 0; row < row_count; ++row) {
    for (uint64_t col = 0; col < row_bytes; ++col) {
      expected[dst_offset + row * target_stride_bytes + col] = src[src_offset + row * source_stride_bytes + col];
    }
  }
}

} // namespace

TEST_CASE("source-window batched scatter kernel copies linear and strided descriptors", "[source_window][cuda]") {
  if (!has_real_cuda_device()) {
    SUCCEED("CUDA device unavailable; source-window batched scatter kernel test skipped");
    return;
  }

  constexpr int kDeviceId = 0;
  REQUIRE(cuda::set_device(kDeviceId).ok());
  const absl::Status prewarm_status = prewarm_source_window_batched_scatter_kernel_for_device(kDeviceId);
  if (absl::IsUnavailable(prewarm_status)) {
    SUCCEED("NVRTC unavailable in the test environment; source-window batched scatter kernel test skipped");
    return;
  }
  INFO(prewarm_status);
  REQUIRE(prewarm_status.ok());

  constexpr size_t kSrcBytes = 256;
  constexpr size_t kDstBytes = 256;
  std::vector<std::uint8_t> src(kSrcBytes);
  std::vector<std::uint8_t> expected(kDstBytes, 0xEE);
  std::vector<std::uint8_t> actual(kDstBytes, 0);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<std::uint8_t>((i * 37 + 11) & 0xFF);
  }

  common::memory::GpuDeviceMemory src_gpu;
  common::memory::GpuDeviceMemory dst_gpu;
  common::memory::GpuDeviceMemory descriptor_gpu;
  REQUIRE(src_gpu.allocate(kSrcBytes, kDeviceId).ok());
  REQUIRE(dst_gpu.allocate(kDstBytes, kDeviceId).ok());

  std::vector<SourceWindowBatchedScatterDescriptor> descriptors{
      SourceWindowBatchedScatterDescriptor{
          .src_ptr = device_address(src_gpu.get(), 3),
          .dst_ptr = device_address(dst_gpu.get(), 5),
          .row_bytes = 17,
          .row_count = 1,
          .source_stride_bytes = 17,
          .target_stride_bytes = 17,
      },
      SourceWindowBatchedScatterDescriptor{
          .src_ptr = device_address(src_gpu.get(), 64),
          .dst_ptr = device_address(dst_gpu.get(), 64),
          .row_bytes = 16,
          .row_count = 3,
          .source_stride_bytes = 24,
          .target_stride_bytes = 32,
      },
      SourceWindowBatchedScatterDescriptor{
          .src_ptr = device_address(src_gpu.get(), 151),
          .dst_ptr = device_address(dst_gpu.get(), 177),
          .row_bytes = 7,
          .row_count = 4,
          .source_stride_bytes = 11,
          .target_stride_bytes = 13,
      },
  };
  REQUIRE(
      descriptor_gpu.allocate(descriptors.size() * source_window_batched_scatter_descriptor_bytes(), kDeviceId).ok());

  apply_expected_copy(src, expected, 3, 5, 17, 1, 17, 17);
  apply_expected_copy(src, expected, 64, 64, 16, 3, 24, 32);
  apply_expected_copy(src, expected, 151, 177, 7, 4, 11, 13);

  std::vector<std::uint8_t> initial_dst(kDstBytes, 0xEE);
  REQUIRE(cuda::memcpy(src_gpu.get(), src.data(), src.size(), cudaMemcpyHostToDevice).ok());
  REQUIRE(cuda::memcpy(dst_gpu.get(), initial_dst.data(), initial_dst.size(), cudaMemcpyHostToDevice).ok());

  cudaStream_t stream = nullptr;
  REQUIRE(cuda::stream_create(&stream).ok());
  REQUIRE(
      launch_source_window_batched_scatter(descriptors, descriptor_gpu.get(), descriptor_gpu.size(), kDeviceId, stream)
          .ok());
  REQUIRE(cuda::stream_synchronize(stream).ok());
  REQUIRE(cuda::stream_destroy(stream).ok());

  REQUIRE(cuda::memcpy(actual.data(), dst_gpu.get(), actual.size(), cudaMemcpyDeviceToHost).ok());
  REQUIRE(actual == expected);
}

TEST_CASE("source-window batched scatter kernel supports alternating devices", "[source_window][cuda]") {
  const int device_count = real_cuda_device_count();
  if (device_count < 2) {
    SUCCEED("fewer than two CUDA devices available; multi-device batched scatter kernel test skipped");
    return;
  }

  constexpr size_t kSrcBytes = 128;
  constexpr size_t kDstBytes = 128;
  std::vector<std::uint8_t> src(kSrcBytes);
  std::vector<std::uint8_t> expected(kDstBytes, 0xA5);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<std::uint8_t>((i * 13 + 7) & 0xFF);
  }
  apply_expected_copy(src, expected, 8, 16, 24, 1, 24, 24);
  apply_expected_copy(src, expected, 48, 64, 8, 4, 12, 10);

  struct PerDeviceState {
    int device_id{-1};
    common::memory::GpuDeviceMemory src_gpu;
    common::memory::GpuDeviceMemory dst_gpu;
    common::memory::GpuDeviceMemory descriptor_gpu;
    cudaStream_t stream{nullptr};
    std::vector<SourceWindowBatchedScatterDescriptor> descriptors;
  };

  std::vector<PerDeviceState> states(2);
  for (int ordinal = 0; ordinal < 2; ++ordinal) {
    auto& state = states[static_cast<size_t>(ordinal)];
    state.device_id = ordinal;
    REQUIRE(cuda::set_device(state.device_id).ok());
    const absl::Status prewarm_status = prewarm_source_window_batched_scatter_kernel_for_device(state.device_id);
    if (absl::IsUnavailable(prewarm_status)) {
      SUCCEED("NVRTC unavailable in the test environment; multi-device batched scatter kernel test skipped");
      return;
    }
    INFO(prewarm_status);
    REQUIRE(prewarm_status.ok());
    REQUIRE(state.src_gpu.allocate(kSrcBytes, state.device_id).ok());
    REQUIRE(state.dst_gpu.allocate(kDstBytes, state.device_id).ok());
    state.descriptors = {
        SourceWindowBatchedScatterDescriptor{
            .src_ptr = device_address(state.src_gpu.get(), 8),
            .dst_ptr = device_address(state.dst_gpu.get(), 16),
            .row_bytes = 24,
            .row_count = 1,
            .source_stride_bytes = 24,
            .target_stride_bytes = 24,
        },
        SourceWindowBatchedScatterDescriptor{
            .src_ptr = device_address(state.src_gpu.get(), 48),
            .dst_ptr = device_address(state.dst_gpu.get(), 64),
            .row_bytes = 8,
            .row_count = 4,
            .source_stride_bytes = 12,
            .target_stride_bytes = 10,
        },
    };
    REQUIRE(state.descriptor_gpu
                .allocate(state.descriptors.size() * source_window_batched_scatter_descriptor_bytes(), state.device_id)
                .ok());
    REQUIRE(cuda::stream_create(&state.stream).ok());
    std::vector<std::uint8_t> initial_dst(kDstBytes, 0xA5);
    REQUIRE(cuda::memcpy(state.src_gpu.get(), src.data(), src.size(), cudaMemcpyHostToDevice).ok());
    REQUIRE(cuda::memcpy(state.dst_gpu.get(), initial_dst.data(), initial_dst.size(), cudaMemcpyHostToDevice).ok());
  }

  for (int round = 0; round < 8; ++round) {
    for (auto& state : states) {
      REQUIRE(cuda::set_device(state.device_id).ok());
      REQUIRE(
          launch_source_window_batched_scatter(
              state.descriptors, state.descriptor_gpu.get(), state.descriptor_gpu.size(), state.device_id, state.stream)
              .ok());
    }
  }

  for (auto& state : states) {
    REQUIRE(cuda::set_device(state.device_id).ok());
    REQUIRE(cuda::stream_synchronize(state.stream).ok());
    std::vector<std::uint8_t> actual(kDstBytes, 0);
    REQUIRE(cuda::memcpy(actual.data(), state.dst_gpu.get(), actual.size(), cudaMemcpyDeviceToHost).ok());
    REQUIRE(actual == expected);
    REQUIRE(cuda::stream_destroy(state.stream).ok());
    state.stream = nullptr;
  }
}

} // namespace tensorcast::store::replica
