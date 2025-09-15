// Copyright (c) 2025, TensorCast Team.

// Basic UMA unit test validating allocation and chunk mapping state.

#include "core/store/replica/unified_memory_authority.h"

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "absl/status/statusor.h"

using tensorcast::common::memory::MemoryLocation;
using tensorcast::common::memory::VirtualAddressSpace;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::replica::UnifiedMemoryAuthority;

static std::shared_ptr<VirtualAddressSpace> make_vs() {
  return std::make_shared<VirtualAddressSpace>(VirtualAddressSpace::kDefaultChunkSize);
}

TEST_CASE("UMA allocate + mappings + get_missing_chunks", "[uma]") {
  auto vs = make_vs();
  UnifiedMemoryAuthority uma(gsl::not_null<std::shared_ptr<VirtualAddressSpace>>{vs});

  ReplicaKey key{
      .artifact_id = std::string("uma_unit_test"), .device = {tensorcast::DeviceType::CPU, -1, ""}, .replica = 0};

  // Allocate 2 chunks worth of CPU memory via UMA (through VS)
  auto st = uma.allocate(key, VirtualAddressSpace::kDefaultChunkSize * 2);
  REQUIRE(st.ok());

  // Initially, CPU missing chunks for target GPU should be all chunks
  auto miss_gpu = uma.get_missing_chunks(key, MemoryLocation::GPU, /*device_id*/ 0);
  REQUIRE(miss_gpu.size() >= 2);

  // After a synthetic plan/commit, GPU should have fewer missing chunks
  std::vector<uint32_t> first_chunk{0};
  auto plan_or = uma.plan_load(key, MemoryLocation::GPU, /*device_id*/ 0, absl::MakeSpan(first_chunk));
  REQUIRE(plan_or.ok());
  auto cst = uma.commit(plan_or->session_id, MemoryLocation::GPU, absl::MakeSpan(first_chunk), /*device_id*/ 0);
  REQUIRE(cst.ok());
  auto miss_gpu2 = uma.get_missing_chunks(key, MemoryLocation::GPU, /*device_id*/ 0);
  REQUIRE(miss_gpu2.size() + 1 == miss_gpu.size());
}
