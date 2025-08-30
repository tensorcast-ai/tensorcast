// Copyright (c) 2025, TensorCast Team.

#include "catch2/catch_test_macros.hpp"

#include "core/store/components/uma_lease_provider.h"
#include "core/store/replica/replica_memory_coordinator.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"

using tensorcast::memory::DistributedVirtualMemoryPool;
using tensorcast::store::ReplicaKey;
using tensorcast::store::DeviceKey;
using tensorcast::DeviceType;

TEST_CASE("UMA Lease Provider hot residency checks", "[store][uma][residency]") {
  // Setup DVMP and ReplicaMemoryCoordinator
  auto dvmp = std::make_shared<DistributedVirtualMemoryPool>(DistributedVirtualMemoryPool::kDefaultChunkSize);
  auto rmc = std::make_shared<tensorcast::store::ReplicaMemoryCoordinator>(dvmp);

  const std::string artifact_id = "rfc0009-uma-test";
  const size_t two_chunks = 2 * DistributedVirtualMemoryPool::kDefaultChunkSize;

  DeviceKey cpu_dev;
  cpu_dev.type = DeviceType::CPU;
  cpu_dev.ordinal = -1;

  ReplicaKey key{.artifact_id = artifact_id, .device = cpu_dev, .replica = 0};
  REQUIRE(rmc->allocate(key, two_chunks).ok());

  // Register UMA mapping (base VA offset = 0)
  const std::string tensor_key = "tensor-key-uma-hot";
  tensorcast::store::UmaLeaseProvider::instance()->register_mapping(tensor_key, key, /*base_va_off=*/0, rmc);

  auto* uma = tensorcast::store::UmaLeaseProvider::instance().get();

  // Initially COLD – not hot
  REQUIRE_FALSE(uma->is_range_hot(tensor_key, /*offset=*/0, /*bytes=*/1));

  // Write a single byte into the first chunk to mark it HOT
  uint8_t one = 0x42;
  REQUIRE(dvmp->write_at(artifact_id, /*va_offset=*/0, &one, 1).ok());

  // Now first chunk should be reported HOT for any subrange within it
  REQUIRE(uma->is_range_hot(tensor_key, /*offset=*/0, /*bytes=*/1));
  REQUIRE(uma->is_range_hot(tensor_key, /*offset=*/DistributedVirtualMemoryPool::kDefaultChunkSize - 16, /*bytes=*/16));

  // A range that spans the boundary into the next (COLD) chunk should not be hot yet
  REQUIRE_FALSE(uma->is_range_hot(
      tensor_key,
      /*offset=*/DistributedVirtualMemoryPool::kDefaultChunkSize - 8,
      /*bytes=*/16));

  // Cross into second chunk – still COLD
  REQUIRE_FALSE(uma->is_range_hot(tensor_key, /*offset=*/DistributedVirtualMemoryPool::kDefaultChunkSize, /*bytes=*/1));

  // Mark second chunk HOT
  REQUIRE(dvmp->write_at(artifact_id, /*va_offset=*/DistributedVirtualMemoryPool::kDefaultChunkSize, &one, 1).ok());

  // Entire two-chunk range should now be HOT
  REQUIRE(uma->is_range_hot(tensor_key, /*offset=*/0, /*bytes=*/two_chunks));
  // And the boundary-spanning subrange becomes hot as well
  REQUIRE(uma->is_range_hot(
      tensor_key,
      /*offset=*/DistributedVirtualMemoryPool::kDefaultChunkSize - 8,
      /*bytes=*/16));
}
