// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/engine/store.h"

#include "catch2/catch_test_macros.hpp"
#include "core/communicator/base/constants.h"
#include "core/communicator/transport/partition_tensor.h"

namespace tensorcast::communicator::engine {

TEST_CASE("PartitionTensorStore overwrites existing key on re-registration", "[communicator][store]") {
  PartitionTensorStore store;

  auto first = std::make_shared<transport::PartitionTensor>(
      "artifact_chunk_0",
      /*addr=*/0x1000,
      /*bytes=*/256,
      base::COMMUNICATE_ENGINE_DEV_CPU);
  store.register_tensor(first);

  auto second = std::make_shared<transport::PartitionTensor>(
      "artifact_chunk_0",
      /*addr=*/0x2000,
      /*bytes=*/512,
      base::COMMUNICATE_ENGINE_DEV_CPU);
  store.register_tensor(second);

  auto current = store.get_tensor("artifact_chunk_0");
  REQUIRE(current != nullptr);
  CHECK(current->get_uint64_addr() == 0x2000);
  CHECK(current->get_bytes() == 512);
}

} // namespace tensorcast::communicator::engine
