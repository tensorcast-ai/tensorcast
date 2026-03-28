// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstddef>
#include <string>

#include "core/cuda/cuda_api.h"
#include "core/cuda/device_guard.h"
#include "core/store/materialization/dataplane/sinks/target_layout_gpu_sink.h"
#include "gsl/pointers"

using tensorcast::cuda::DeviceGuard;
using tensorcast::store::loader::TargetLayoutGpuSink;
using tensorcast::store::loader::TargetStorage;

class MultiStorageFixture {
 public:
  MultiStorageFixture(size_t size0, size_t size1) : size0_(size0), size1_(size1) {
    int device_count = 0;
    auto status = tensorcast::cuda::get_device_count(&device_count);
    if (!status.ok() || device_count == 0) {
      available_ = false;
      return;
    }

    DeviceGuard guard(0);
    if (!guard.status().ok()) {
      available_ = false;
      return;
    }

    status = tensorcast::cuda::malloc(&gpu_ptr0_, size0_);
    if (!status.ok()) {
      available_ = false;
      return;
    }
    status = tensorcast::cuda::malloc(&gpu_ptr1_, size1_);
    if (!status.ok()) {
      auto st = tensorcast::cuda::free(gpu_ptr0_);
      (void)st;
      gpu_ptr0_ = nullptr;
      available_ = false;
      return;
    }

    available_ = true;
  }

  ~MultiStorageFixture() {
    if (gpu_ptr0_) {
      auto st = tensorcast::cuda::free(gpu_ptr0_);
      (void)st;
    }
    if (gpu_ptr1_) {
      auto st = tensorcast::cuda::free(gpu_ptr1_);
      (void)st;
    }
  }

  bool is_available() const {
    return available_;
  }

  void* gpu_ptr0() const {
    return gpu_ptr0_;
  }

  void* gpu_ptr1() const {
    return gpu_ptr1_;
  }

  size_t size0() const {
    return size0_;
  }

  size_t size1() const {
    return size1_;
  }

 private:
  size_t size0_{0};
  size_t size1_{0};
  void* gpu_ptr0_{nullptr};
  void* gpu_ptr1_{nullptr};
  bool available_{false};
};

TargetLayoutGpuSink::Options make_options(const MultiStorageFixture& fixture) {
  TargetLayoutGpuSink::Options options;
  options.device_id = 0;
  options.chunk_size = 4;
  options.storages = {
      TargetStorage{gsl::not_null<void*>{fixture.gpu_ptr0()}, fixture.size0()},
      TargetStorage{gsl::not_null<void*>{fixture.gpu_ptr1()}, fixture.size1()},
  };
  return options;
}

TEST_CASE("TargetLayoutGpuSink routes writes to storages", "[target_layout_gpu_sink]") {
  MultiStorageFixture fixture(/*size0=*/4, /*size1=*/4);
  if (!fixture.is_available()) {
    SKIP("CUDA not available");
  }

  SECTION("Writes stay within each storage") {
    TargetLayoutGpuSink sink(make_options(fixture));

    const std::array<char, 4> first = {'A', 'A', 'A', 'A'};
    const std::array<char, 4> second = {'B', 'B', 'B', 'B'};

    REQUIRE(sink.write_at(0, first.data(), first.size()).ok());
    REQUIRE(sink.write_at(4, second.data(), second.size()).ok());
    REQUIRE(sink.close().ok());

    std::array<char, 4> out_first{};
    std::array<char, 4> out_second{};
    DeviceGuard guard(0);
    REQUIRE(guard.status().ok());
    auto status =
        tensorcast::cuda::memcpy(out_first.data(), fixture.gpu_ptr0(), out_first.size(), cudaMemcpyDeviceToHost);
    REQUIRE(status.ok());
    status = tensorcast::cuda::memcpy(out_second.data(), fixture.gpu_ptr1(), out_second.size(), cudaMemcpyDeviceToHost);
    REQUIRE(status.ok());

    REQUIRE(out_first == first);
    REQUIRE(out_second == second);
  }

  SECTION("Overlapping writes still satisfy close coverage") {
    TargetLayoutGpuSink sink(make_options(fixture));

    const std::array<char, 4> first = {'A', 'A', 'A', 'A'};
    const std::array<char, 4> second = {'B', 'B', 'B', 'B'};
    const std::array<char, 2> overlap = {'C', 'C'};

    REQUIRE(sink.write_at(0, first.data(), first.size()).ok());
    REQUIRE(sink.write_at(4, second.data(), second.size()).ok());
    REQUIRE(sink.write_at(6, overlap.data(), overlap.size()).ok());
    REQUIRE(sink.close().ok());

    std::array<char, 4> out_first{};
    std::array<char, 4> out_second{};
    DeviceGuard guard(0);
    REQUIRE(guard.status().ok());
    auto status =
        tensorcast::cuda::memcpy(out_first.data(), fixture.gpu_ptr0(), out_first.size(), cudaMemcpyDeviceToHost);
    REQUIRE(status.ok());
    status = tensorcast::cuda::memcpy(out_second.data(), fixture.gpu_ptr1(), out_second.size(), cudaMemcpyDeviceToHost);
    REQUIRE(status.ok());

    REQUIRE(out_first == first);
    CHECK(out_second[0] == 'B');
    CHECK(out_second[1] == 'B');
    CHECK(out_second[2] == 'C');
    CHECK(out_second[3] == 'C');
  }

  SECTION("Writes spanning storages are rejected") {
    TargetLayoutGpuSink sink(make_options(fixture));

    const std::array<char, 4> data = {'C', 'C', 'C', 'C'};
    auto status = sink.write_at(2, data.data(), data.size());
    REQUIRE_FALSE(status.ok());
    REQUIRE(status.message().find("spans multiple target storages") != std::string::npos);
  }
}
