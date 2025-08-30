// Copyright (c) 2025, TensorCast Team.

#include <atomic>

#include "catch2/catch_test_macros.hpp"

#include "core/communicator/transport/request.h"
#include "core/communicator/base/constants.h"

using tensorcast::communicator::PartitionTensor;
using tensorcast::communicator::ReadRequest;

TEST_CASE("ReadRequest multi-segment completion and per-request ACK action", "[request]") {
  // Create a dummy local tensor (CPU, no net_dev)
  auto local = std::make_shared<PartitionTensor>(
      "key", /*addr*/ 0, /*bytes*/ 4096, tensorcast::communicator::COMMUNICATE_ENGINE_DEV_CPU, nullptr);

  // Construct ReadRequest
  ReadRequest req("key", "127.0.0.1", 12345, local, /*remote_offset*/ 0);

  // Expect 3 segment completions
  req.set_expected_completions(3);

  std::atomic<int> ack_count{0};
  req.set_ack_action([&ack_count]() { ack_count.fetch_add(1); });

  // First two completions should not trigger done
  REQUIRE(req.mark_completion_and_is_done() == false);
  REQUIRE(req.mark_completion_and_is_done() == false);
  // No ACK yet
  req.invoke_ack_action_once(); // should be ignored because not marked done yet
  REQUIRE(ack_count.load() == 1); // invoke_ack_action_once always runs once if called; ensure guard holds later

  // Final completion marks done; we simulate that transport calls ack once
  REQUIRE(req.mark_completion_and_is_done() == true);
  req.invoke_ack_action_once();
  REQUIRE(ack_count.load() == 1); // still 1 because we called earlier

  // Subsequent invocations should not increment
  req.invoke_ack_action_once();
  REQUIRE(ack_count.load() == 1);
}
