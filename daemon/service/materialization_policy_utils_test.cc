// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/materialization_policy_utils.h"

#include <memory>
#include <string_view>

#include <catch2/catch_test_macros.hpp>
#include "core/store/testing/global_store_client_stub.h"

namespace {

using tensorcast::daemon::materialization_policy::apply_group_realization_begin_context_to_transport_context;
using tensorcast::daemon::materialization_policy::begin_or_join_group_realization_if_enabled;
using tensorcast::daemon::materialization_policy::default_collective_policy_for_mapped_target;
using tensorcast::daemon::materialization_policy::GroupRealizationPreparedMemberContext;
using tensorcast::daemon::materialization_policy::report_group_realization_prepared_if_enabled;
using tensorcast::daemon::materialization_policy::resolve_group_realization_transport_context;
using tensorcast::daemon::materialization_policy::validate_group_realization_staged_publish_supported;
using tensorcast::store::loading::CollectiveLoadGroupHint;
using tensorcast::store::loading::ExecutionTopologyContext;
namespace global_store = tensorcast::global_store::v1;
namespace common = tensorcast::common::v1;
namespace v2 = tensorcast::daemon::v2;

class RecordingGroupRealizationClient : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  global_store::BeginOrJoinGroupRealizationRequest last_request;
  global_store::ReportGroupRealizationPreparedRequest last_prepared_request;

  absl::StatusOr<global_store::BeginOrJoinGroupRealizationResponse> begin_or_join_group_realization(
      const global_store::BeginOrJoinGroupRealizationRequest& request,
      const tensorcast::store::components::RpcOptions&) override {
    last_request = request;
    global_store::BeginOrJoinGroupRealizationResponse response;
    response.set_status(global_store::STATUS_OK);
    response.set_transaction_id("txn-17");
    response.mutable_version_set()->set_version_set_id("vs-17");
    response.set_realization_kind(global_store::GROUP_REALIZATION_KIND_PER_PART_SELECTION);
    response.mutable_part()->set_part_id(request.context().part_id());
    response.mutable_part()->mutable_selection()->set_artifact_id("artifact-frozen-rank0");
    response.mutable_part()->mutable_selection()->set_view_id("view-rank0");
    response.mutable_part()->mutable_requested_byte_space()->set_kind(common::BYTE_SPACE_KIND_VIEW);
    response.mutable_part()->mutable_requested_byte_space()->set_id("view-rank0");
    response.mutable_part()->set_selection_hash("hash-rank0");
    response.set_state(global_store::GROUP_REALIZATION_STATE_RESOLVED);
    response.set_key_generation(42);
    return response;
  }

  absl::StatusOr<global_store::ReportGroupRealizationPreparedResponse> report_group_realization_prepared(
      const global_store::ReportGroupRealizationPreparedRequest& request,
      const tensorcast::store::components::RpcOptions&) override {
    last_prepared_request = request;
    global_store::ReportGroupRealizationPreparedResponse response;
    response.set_status(global_store::STATUS_OK);
    response.set_state(global_store::GROUP_REALIZATION_STATE_PREPARING);
    response.set_member_state(global_store::GROUP_REALIZATION_MEMBER_STATE_PREPARED);
    response.set_member_fingerprint("member-fingerprint");
    return response;
  }
};

v2::GroupRealizationOptions build_group_realization_options() {
  v2::GroupRealizationOptions options;
  options.set_enabled(true);
  options.mutable_version()->mutable_explicit_selection()->set_artifact_id("artifact-a");
  auto* group = options.mutable_group();
  group->set_group_kind("weight_publish");
  group->set_group_id("model-v17");
  group->set_epoch(7);
  group->set_total_parts(2);
  group->set_part_id("rank0");
  group->add_required_part_ids("rank0");
  group->add_required_part_ids("rank1");
  return options;
}

TEST_CASE(
    "Group realization begin joins through Global Store and returns frozen part selection",
    "[daemon][materialization][policy]") {
  auto options = build_group_realization_options();
  auto client = std::make_shared<RecordingGroupRealizationClient>();

  auto begin_or = begin_or_join_group_realization_if_enabled(client, &options, "daemon-a", "session-a", "worker-a");

  REQUIRE(begin_or.ok());
  REQUIRE(begin_or->has_value());
  CHECK((*begin_or)->transaction_id == "txn-17");
  CHECK((*begin_or)->part_id == "rank0");
  CHECK((*begin_or)->part_selection.artifact_id() == "artifact-frozen-rank0");
  CHECK((*begin_or)->key_generation == 42);
  CHECK(client->last_request.daemon_id() == "daemon-a");
  CHECK(client->last_request.daemon_session_id() == "session-a");
  CHECK(client->last_request.worker_id() == "worker-a");
  CHECK(client->last_request.context().group_kind() == "weight_publish");
  CHECK(client->last_request.context().required_part_ids_size() == 2);
  CHECK(client->last_request.version().explicit_selection().artifact_id() == "artifact-a");

  auto transport_or = resolve_group_realization_transport_context("op-17", &options);
  REQUIRE(transport_or.ok());
  apply_group_realization_begin_context_to_transport_context(**begin_or, &*transport_or);
  REQUIRE(transport_or->transport_scheduling_group.has_value());
  const std::string& group_id = transport_or->transport_scheduling_group->group_id;
  CHECK(group_id.find("txn:txn-17") != std::string::npos);
  CHECK(group_id.find("vs:vs-17") != std::string::npos);
  CHECK(group_id.find("artifact:artifact-frozen-rank0") != std::string::npos);
  CHECK(group_id.find("part:rank0") != std::string::npos);
  CHECK(group_id.find("byte_space:2:view-rank0") != std::string::npos);
  CHECK(group_id.find("view:view-rank0") != std::string::npos);
  CHECK(group_id.find("selection_hash:686173682d72616e6b30") != std::string::npos);
}

TEST_CASE("Group realization prepared report carries staged member fences", "[daemon][materialization][policy]") {
  auto options = build_group_realization_options();
  auto client = std::make_shared<RecordingGroupRealizationClient>();
  auto begin_or = begin_or_join_group_realization_if_enabled(client, &options, "daemon-a", "session-a", "worker-a");
  REQUIRE(begin_or.ok());
  REQUIRE(begin_or->has_value());

  auto prepared_or = report_group_realization_prepared_if_enabled(
      client,
      &options,
      &**begin_or,
      GroupRealizationPreparedMemberContext{
          .binding_id = "binding-1",
          .binding_value_id = "staged-value-1",
          .staging_token = "staging-token-1",
          .staging_epoch = 11,
          .expected_previous_seal_generation = 7,
          .materialization_attempt_id = "attempt-1",
          .prepared_value_hash = "prepared-hash",
          .source_replica_id = "replica-1",
          .source_export_generation = 5,
          .child_transport_request_id = "transport-1",
      },
      "daemon-a",
      "session-a",
      "worker-a");

  REQUIRE(prepared_or.ok());
  REQUIRE(prepared_or->has_value());
  CHECK((*prepared_or)->state() == global_store::GROUP_REALIZATION_STATE_PREPARING);
  CHECK((*prepared_or)->member_state() == global_store::GROUP_REALIZATION_MEMBER_STATE_PREPARED);
  CHECK(client->last_prepared_request.transaction_id() == "txn-17");
  CHECK(client->last_prepared_request.part_id() == "rank0");
  CHECK(client->last_prepared_request.staged_value().daemon_id() == "daemon-a");
  CHECK(client->last_prepared_request.staged_value().daemon_session_id() == "session-a");
  CHECK(client->last_prepared_request.staged_value().binding_id() == "binding-1");
  CHECK(client->last_prepared_request.staged_value().binding_value_id() == "staged-value-1");
  CHECK(client->last_prepared_request.staged_value().staging_token() == "staging-token-1");
  CHECK(client->last_prepared_request.staged_value().staging_epoch() == 11);
  CHECK(client->last_prepared_request.expected_previous_seal_generation() == 7);
  CHECK(client->last_prepared_request.materialization_attempt_id() == "attempt-1");
  CHECK(client->last_prepared_request.prepared_value_hash() == "prepared-hash");
  CHECK(client->last_prepared_request.source_replica_id() == "replica-1");
  CHECK(client->last_prepared_request.source_export_generation() == 5);
  CHECK(client->last_prepared_request.child_transport_request_id() == "transport-1");
}

TEST_CASE(
    "Group realization prepared report validates staged identity before RPC",
    "[daemon][materialization][policy]") {
  auto options = build_group_realization_options();
  auto client = std::make_shared<RecordingGroupRealizationClient>();
  auto begin_or = begin_or_join_group_realization_if_enabled(client, &options, "daemon-a", "session-a", "worker-a");
  REQUIRE(begin_or.ok());
  REQUIRE(begin_or->has_value());

  auto prepared_or = report_group_realization_prepared_if_enabled(
      client,
      &options,
      &**begin_or,
      GroupRealizationPreparedMemberContext{
          .binding_id = "binding-1",
          .binding_value_id = "staged-value-1",
          .staging_token = "",
          .staging_epoch = 11,
      },
      "daemon-a",
      "session-a",
      "worker-a");

  REQUIRE_FALSE(prepared_or.ok());
  CHECK(prepared_or.status().message().find("staged binding value identity") != std::string_view::npos);
}

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

TEST_CASE(
    "Group realization typed context derives strict transport scheduling group",
    "[daemon][materialization][policy]") {
  auto options = build_group_realization_options();

  auto context_or = resolve_group_realization_transport_context("op-17", &options);

  REQUIRE(context_or.ok());
  REQUIRE(context_or->transport_scheduling_group.has_value());
  const auto& group = *context_or->transport_scheduling_group;
  CHECK(context_or->transport_request_id == "op-17");
  CHECK(group.group_kind == "group_realization_transport");
  CHECK(group.group_id == "weight_publish:model-v17");
  CHECK(group.total_parts == 2);
  CHECK(group.part_id == "rank0");
  CHECK(group.epoch == 7);
}

TEST_CASE(
    "Group realization typed context uses typed group and opaque operation id",
    "[daemon][materialization][policy]") {
  auto options = build_group_realization_options();

  auto context_or = resolve_group_realization_transport_context("op-17-with-diagnostics", &options);

  REQUIRE(context_or.ok());
  REQUIRE(context_or->transport_scheduling_group.has_value());
  const auto& group = *context_or->transport_scheduling_group;
  CHECK(context_or->transport_request_id == "op-17-with-diagnostics");
  CHECK(group.group_kind == "group_realization_transport");
  CHECK(group.group_id == "weight_publish:model-v17");
  CHECK(group.part_id == "rank0");
  CHECK(group.priority == 0);
}

TEST_CASE("Group realization rejects invalid typed membership", "[daemon][materialization][policy]") {
  auto options = build_group_realization_options();
  options.mutable_group()->clear_required_part_ids();
  options.mutable_group()->add_required_part_ids("rank1");
  options.mutable_group()->add_required_part_ids("rank2");

  auto context_or = resolve_group_realization_transport_context("op-17", &options);

  CHECK_FALSE(context_or.ok());
}

TEST_CASE(
    "Group realization staged publish requirement fails on non-staged paths",
    "[daemon][materialization][policy]") {
  auto options = build_group_realization_options();
  options.set_require_staged_publish(true);

  CHECK_FALSE(validate_group_realization_staged_publish_supported(&options, false).ok());
  CHECK(validate_group_realization_staged_publish_supported(&options, true).ok());
  options.set_require_staged_publish(false);
  CHECK(validate_group_realization_staged_publish_supported(&options, false).ok());
  CHECK(validate_group_realization_staged_publish_supported(nullptr, false).ok());
}

} // namespace
