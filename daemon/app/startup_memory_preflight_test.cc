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

} // namespace tensorcast::daemon
