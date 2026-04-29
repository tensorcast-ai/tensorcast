// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/materialization_policy_utils.h"

#include <catch2/catch_test_macros.hpp>

namespace {

using tensorcast::daemon::materialization_policy::default_collective_policy_for_mapped_target;
using tensorcast::daemon::materialization_policy::resolve_transport_scheduling_group_hint;
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

TEST_CASE("Transport scheduling group hint maps daemon proto", "[daemon][materialization][policy]") {
  v2::TransportSchedulingGroupHint proto;
  proto.set_group_kind("weight_broadcast");
  proto.set_group_id("model-a:v42");
  proto.set_total_parts(16);
  proto.set_part_id("daemon-1");
  proto.set_priority(7);
  proto.set_epoch(42);

  auto hint = resolve_transport_scheduling_group_hint(&proto);

  REQUIRE(hint.has_value());
  CHECK(hint->group_kind == "weight_broadcast");
  CHECK(hint->group_id == "model-a:v42");
  CHECK(hint->total_parts == 16);
  CHECK(hint->part_id == "daemon-1");
  CHECK(hint->priority == 7);
  CHECK(hint->epoch == 42);
}

} // namespace
