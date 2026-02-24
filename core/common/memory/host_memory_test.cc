// Copyright (c) 2025-2026, TensorCast Team.

#include "core/common/memory/host_memory.h"

#include <catch2/catch_test_macros.hpp>

using tensorcast::common::memory::detect_host_memory_available_bytes;
using tensorcast::common::memory::detect_host_memory_capacity_bytes;
using tensorcast::common::memory::set_host_memory_available_override_for_testing;
using tensorcast::common::memory::set_host_memory_capacity_override_for_testing;

TEST_CASE("Host memory detection honors override", "[host_memory]") {
  set_host_memory_capacity_override_for_testing(1234);
  auto cap = detect_host_memory_capacity_bytes();
  REQUIRE(cap.ok());
  REQUIRE(*cap == 1234);

  // Clear override to avoid test cross-talk.
  set_host_memory_capacity_override_for_testing(std::nullopt);
}

TEST_CASE("Host available memory detection honors override", "[host_memory]") {
  set_host_memory_available_override_for_testing(5678);
  auto avail = detect_host_memory_available_bytes();
  REQUIRE(avail.ok());
  REQUIRE(*avail == 5678);

  // Clear override to avoid test cross-talk.
  set_host_memory_available_override_for_testing(std::nullopt);
}
