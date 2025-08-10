// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/common/memory/distributed_virtual_memory_pool.h"

using stepcast::memory::DistributedVirtualMemoryPool;

TEST_CASE("DvmpRegion handle open and basic delegation") {
  DistributedVirtualMemoryPool dvmp;
  auto not_found = dvmp.open("missing");
  REQUIRE(!not_found.ok());

  // Allocate and open
  auto region_or = dvmp.allocate("m1", 256ULL * 1024 * 1024);
  REQUIRE(region_or.ok());
  auto handle_or = dvmp.open("m1");
  REQUIRE(handle_or.ok());

  // Delegated calls succeed (no-ops for empty spans)
  auto st = handle_or->lock_chunks({});
  REQUIRE(st.ok());
}
