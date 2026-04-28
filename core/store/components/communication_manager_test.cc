// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "core/store/components/communication_manager.h"

namespace tensorcast::store::components {
namespace {

TEST_CASE("CommunicationManager register_memory uses unique keys for repeated exports", "[communication_manager]") {
  CommunicationManager manager;
  REQUIRE(manager.initialize("127.0.0.1", 0, /*enable_rdma=*/false).ok());

  std::vector<char> buffer(64, '\0');
  std::vector<void*> addresses{buffer.data()};
  std::vector<size_t> sizes{buffer.size()};

  auto first_or = manager.register_memory(addresses, sizes, /*device_id=*/-1);
  REQUIRE(first_or.ok());
  auto second_or = manager.register_memory(addresses, sizes, /*device_id=*/-1);
  REQUIRE(second_or.ok());

  REQUIRE(first_or->remote_memory_keys.size() == 1);
  REQUIRE(second_or->remote_memory_keys.size() == 1);
  CHECK(first_or->remote_memory_keys.front() != second_or->remote_memory_keys.front());

  CHECK(manager.get_engine().unregister_tensor(first_or->remote_memory_keys.front()).ok());
  CHECK(manager.get_engine().unregister_tensor(second_or->remote_memory_keys.front()).ok());
}

} // namespace
} // namespace tensorcast::store::components
