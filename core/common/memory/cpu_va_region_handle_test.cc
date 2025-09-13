// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/common/memory/virtual_address_space.h"

using tensorcast::common::memory::VirtualAddressSpace;

TEST_CASE("CPU VA region handle open and basic delegation") {
  VirtualAddressSpace virtual_addr_space(VirtualAddressSpace::kDefaultChunkSize);
  auto not_found = virtual_addr_space.open("missing");
  REQUIRE(!not_found.ok());

  // Allocate and open
  auto region_or = virtual_addr_space.allocate("m1", 256ULL * 1024 * 1024);
  REQUIRE(region_or.ok());
  auto handle_or = virtual_addr_space.open("m1");
  REQUIRE(handle_or.ok());

  // Basic write via region handle
  const char payload[] = "x";
  auto st = handle_or->write_at(/*va_offset=*/0, payload, sizeof(payload));
  REQUIRE(st.ok());
}
