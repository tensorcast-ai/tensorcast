// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/materialization_policy_utils.h"

#include <catch2/catch_test_macros.hpp>

namespace {

using tensorcast::daemon::materialization_policy::default_collective_policy_for_mapped_target;
using tensorcast::store::loading::CollectiveLoadGroupHint;
using tensorcast::store::loading::ExecutionTopologyContext;
namespace v2 = tensorcast::daemon::v2;

TEST_CASE(
    "Mapped target defaults to collective-first when collective topology is present",
    "[daemon][materialization][policy]") {
  ExecutionTopologyContext execution_topology;
  execution_topology.collective_load_group =
      CollectiveLoadGroupHint{.group_id = "same-host-tp-load", .world_size = 8, .rank = 3};

  CHECK(
      default_collective_policy_for_mapped_target(execution_topology) ==
      v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST);
}

TEST_CASE(
    "Mapped target defaults to disable-collective without collective topology",
    "[daemon][materialization][policy]") {
  CHECK(
      default_collective_policy_for_mapped_target(ExecutionTopologyContext{}) ==
      v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE);
}

} // namespace
