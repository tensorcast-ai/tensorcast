// Copyright (c) 2025, TensorCast Team.

#include "core/common/memory/host_memory.h"

#include <catch2/catch_test_macros.hpp>

using tensorcast::common::memory::detect_host_memory_capacity_bytes;
using tensorcast::common::memory::set_host_memory_capacity_override_for_testing;

TEST_CASE("Host memory detection honors override", "[host_memory]") {
  set_host_memory_capacity_override_for_testing(1234);
  auto cap = detect_host_memory_capacity_bytes();
  REQUIRE(cap.ok());
  REQUIRE(*cap == 1234);

  // Clear override to avoid test cross-talk.
  set_host_memory_capacity_override_for_testing(std::nullopt);
}
