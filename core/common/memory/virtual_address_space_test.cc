// Copyright (c) 2025, TensorCast Team.

#include "core/common/memory/virtual_address_space.h"

#include <cstring>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include "absl/status/statusor.h"

using tensorcast::common::memory::VirtualAddressSpace;

TEST_CASE("VirtualAddressSpace alloc/open/write_at basic", "[virtual_address_space]") {
  VirtualAddressSpace vs(VirtualAddressSpace::kDefaultChunkSize);
  const std::string artifact_id = "vs_test_artifact";

  // Allocate a small region (two chunks worth)
  auto reg_or = vs.allocate(artifact_id, VirtualAddressSpace::kDefaultChunkSize * 2);
  REQUIRE(reg_or.ok());
  auto region = *reg_or;
  REQUIRE(region.cpu_base != nullptr);
  REQUIRE(region.bytes == VirtualAddressSpace::kDefaultChunkSize * 2);

  // Open a region handle and write some bytes
  auto handle_or = vs.open(artifact_id);
  REQUIRE(handle_or.ok());
  auto handle = *handle_or;

  const char payload[] = "hello_vs";
  auto st = handle.write_at(/*va_offset=*/0, payload, sizeof(payload));
  REQUIRE(st.ok());

  // Verify memory content at base address
  char* base = static_cast<char*>(region.cpu_base);
  REQUIRE(std::memcmp(base, payload, sizeof(payload)) == 0);
}
