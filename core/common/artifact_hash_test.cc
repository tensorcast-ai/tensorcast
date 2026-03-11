// Copyright (c) 2026, TensorCast Team.

#include "core/common/artifact_hash.h"

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::common {

TEST_CASE("GPU hash NVRTC prewarm rejects negative device id", "[artifact_hash][gpu]") {
  const absl::Status status = prewarm_gpu_hash_nvrtc_for_device(-1);
  REQUIRE_FALSE(status.ok());
  REQUIRE(absl::IsInvalidArgument(status));
}

TEST_CASE("GPU hash NVRTC prewarm is a no-op under FakeCuda backend", "[artifact_hash][gpu][fake]") {
  if (!cuda::is_fake()) {
    return;
  }

  REQUIRE(prewarm_gpu_hash_nvrtc_for_device(0).ok());
  REQUIRE(prewarm_gpu_hash_nvrtc_for_visible_devices().ok());
}

TEST_CASE("GPU hash NVRTC prewarm tolerates empty visible device set", "[artifact_hash][gpu]") {
  if (cuda::is_fake()) {
    return;
  }

  int device_count = 0;
  REQUIRE(cuda::get_device_count(&device_count).ok());
  if (device_count != 0) {
    return;
  }

  REQUIRE(prewarm_gpu_hash_nvrtc_for_visible_devices().ok());
  const absl::Status device_status = prewarm_gpu_hash_nvrtc_for_device(std::max(device_count, 0));
  REQUIRE_FALSE(device_status.ok());
  REQUIRE(absl::IsInvalidArgument(device_status));
}

} // namespace tensorcast::common
