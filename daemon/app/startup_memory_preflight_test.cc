// Copyright (c) 2026, TensorCast Team.

#include "daemon/app/startup_memory_preflight.h"

#include <catch2/catch_test_macros.hpp>

namespace tensorcast::daemon {
namespace {

constexpr uint64_t kGiB = 1024ULL * 1024 * 1024;

} // namespace

TEST_CASE("Startup memory headroom is min(10%, 10GiB)", "[startup_memory_preflight]") {
  REQUIRE(compute_startup_memory_headroom_bytes(100) == 10);
  REQUIRE(compute_startup_memory_headroom_bytes(50 * kGiB) == 5 * kGiB);
  REQUIRE(compute_startup_memory_headroom_bytes(300 * kGiB) == 10 * kGiB);
}

TEST_CASE("Startup memory preflight fails when available < required+headroom", "[startup_memory_preflight]") {
  set_startup_memory_available_override_for_testing(21 * kGiB);
  const absl::Status st = preflight_startup_memory(/*pinned_bytes=*/20 * kGiB, /*stable_bytes=*/0);
  REQUIRE(!st.ok());
  REQUIRE(st.code() == absl::StatusCode::kResourceExhausted);

  set_startup_memory_available_override_for_testing(22 * kGiB);
  const absl::Status ok = preflight_startup_memory(/*pinned_bytes=*/20 * kGiB, /*stable_bytes=*/0);
  REQUIRE(ok.ok());

  // Clear override to avoid test cross-talk.
  set_startup_memory_available_override_for_testing(std::nullopt);
}

TEST_CASE("Engine pinned concurrency sizing honors fake CUDA mode", "[startup_memory_preflight]") {
  const auto real = compute_engine_pinned_concurrency_sizing(
      /*streaming_buffer_chunks=*/16,
      /*detected_gpu_count=*/4,
      /*fake_cuda_backend=*/false);
  REQUIRE(real.effective_gpu_count == 4);
  REQUIRE(real.required_slices == 64);

  const auto fake_multi = compute_engine_pinned_concurrency_sizing(
      /*streaming_buffer_chunks=*/16,
      /*detected_gpu_count=*/4,
      /*fake_cuda_backend=*/true);
  REQUIRE(fake_multi.effective_gpu_count == 1);
  REQUIRE(fake_multi.required_slices == 16);

  const auto fake_zero = compute_engine_pinned_concurrency_sizing(
      /*streaming_buffer_chunks=*/16,
      /*detected_gpu_count=*/0,
      /*fake_cuda_backend=*/true);
  REQUIRE(fake_zero.effective_gpu_count == 1);
  REQUIRE(fake_zero.required_slices == 16);
}

} // namespace tensorcast::daemon
