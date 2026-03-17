// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include "core/store/store_engine.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"
#include "tensorcast/node_agent/v1/node_agent.pb.h"

namespace {

tensorcast::store::StoreEngineOptions make_opts_small() {
  tensorcast::store::StoreEngineOptions opts;
  opts.memory_pool_size = 32ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  return opts;
}

std::unique_ptr<tensorcast::daemon::DaemonServiceHarness> make_harness() {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_small());
  tensorcast::daemon::DaemonOptions options;
  options.daemon_id = "daemon-local";
  options.gateway_ingress_enabled = true;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, options);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

tensorcast::common::v1::ArtifactSelection make_selection() {
  tensorcast::common::v1::ArtifactSelection selection;
  selection.set_artifact_id("mi2:test");
  selection.set_logical_layout_hash("logical");
  selection.set_selection_hash("selection");
  return selection;
}

tensorcast::node_agent::v1::ExecutePlanResponse parse_terminal(
    const tensorcast::daemon::v2::ExecutePlanResponse& response) {
  tensorcast::node_agent::v1::ExecutePlanResponse terminal;
  REQUIRE_FALSE(response.terminal_result().empty());
  REQUIRE(terminal.ParseFromString(response.terminal_result()));
  return terminal;
}

} // namespace

TEST_CASE("ExecutePlan serves local dry-run worker plans", "[daemon][ingress][plan]") {
  auto harness = make_harness();
  auto& svc = harness->service();
  const std::string daemon_id = harness->kernel().worker_identity_store().daemon_id();
  REQUIRE_FALSE(daemon_id.empty());

  tensorcast::daemon::v2::ExecutePlanRequest request;
  request.set_execution_class(tensorcast::daemon::v2::PLAN_EXECUTION_CLASS_TERMINAL_ONLY);
  request.set_dry_run(true);
  request.mutable_plan()->mutable_context()->set_request_id("req-local-dry-run");
  auto* step = request.mutable_plan()->add_steps();
  step->set_step_id("s1");
  step->mutable_target()->set_target_type(tensorcast::plan::v1::TARGET_TYPE_WORKER);
  step->mutable_target()->set_target_id(daemon_id);
  step->mutable_action()->mutable_prefetch()->mutable_selection()->CopyFrom(make_selection());
  step->mutable_action()->mutable_prefetch()->set_device_id(-1);

  tensorcast::daemon::v2::ExecutePlanResponse response;
  grpc::ServerContext ctx;
  const auto status = svc.ExecutePlan(&ctx, &request, &response);
  REQUIRE(status.ok());
  REQUIRE(response.ok());
  REQUIRE(response.request_id() == "req-local-dry-run");

  const auto terminal = parse_terminal(response);
  REQUIRE(terminal.ok());
  REQUIRE(terminal.request_id() == "req-local-dry-run");
  REQUIRE(terminal.steps_size() == 1);
  REQUIRE(terminal.steps(0).step_id() == "s1");
  REQUIRE(terminal.steps(0).status().state() == tensorcast::node_agent::v1::OPERATION_STATE_SUCCESS);
  REQUIRE(terminal.steps(0).status().message() == "dry-run");
}

TEST_CASE("ExecutePlan rejects remote worker targets before dispatch", "[daemon][ingress][plan]") {
  auto harness = make_harness();
  auto& svc = harness->service();

  tensorcast::daemon::v2::ExecutePlanRequest request;
  request.set_execution_class(tensorcast::daemon::v2::PLAN_EXECUTION_CLASS_TERMINAL_ONLY);
  request.set_dry_run(true);
  request.mutable_plan()->mutable_context()->set_request_id("req-remote-worker");
  auto* step = request.mutable_plan()->add_steps();
  step->set_step_id("s1");
  step->mutable_target()->set_target_type(tensorcast::plan::v1::TARGET_TYPE_WORKER);
  step->mutable_target()->set_target_id("daemon-remote");
  step->mutable_action()->mutable_prefetch()->mutable_selection()->CopyFrom(make_selection());
  step->mutable_action()->mutable_prefetch()->set_device_id(-1);

  tensorcast::daemon::v2::ExecutePlanResponse response;
  grpc::ServerContext ctx;
  const auto status = svc.ExecutePlan(&ctx, &request, &response);
  REQUIRE(status.ok());
  REQUIRE_FALSE(response.ok());

  const auto terminal = parse_terminal(response);
  REQUIRE_FALSE(terminal.ok());
  REQUIRE(terminal.steps_size() == 1);
  REQUIRE(terminal.steps(0).status().state() == tensorcast::node_agent::v1::OPERATION_STATE_FAILED);
  REQUIRE(terminal.steps(0).status().error().status_code() == "FAILED_PRECONDITION");
  REQUIRE(terminal.steps(0).status().message() == "worker target does not match this agent");
}

TEST_CASE("ExecutePlan rejects instance and cluster targets in the first ingress slice", "[daemon][ingress][plan]") {
  auto harness = make_harness();
  auto& svc = harness->service();

  tensorcast::daemon::v2::ExecutePlanRequest request;
  request.set_execution_class(tensorcast::daemon::v2::PLAN_EXECUTION_CLASS_TERMINAL_ONLY);
  request.mutable_plan()->mutable_context()->set_request_id("req-nonlocal-targets");

  auto* instance_step = request.mutable_plan()->add_steps();
  instance_step->set_step_id("instance");
  instance_step->mutable_target()->set_target_type(tensorcast::plan::v1::TARGET_TYPE_INSTANCE);
  instance_step->mutable_target()->set_target_id("inst-1");
  instance_step->mutable_action()->mutable_manifest()->set_engine_request_id("eng-1");

  auto* cluster_step = request.mutable_plan()->add_steps();
  cluster_step->set_step_id("cluster");
  cluster_step->add_depends_on("instance");
  cluster_step->mutable_target()->set_target_type(tensorcast::plan::v1::TARGET_TYPE_CLUSTER);
  cluster_step->mutable_target()->set_target_id("cluster-1");
  cluster_step->mutable_action()->mutable_cluster_action()->set_action_ref("wf-1");

  tensorcast::daemon::v2::ExecutePlanResponse response;
  grpc::ServerContext ctx;
  const auto status = svc.ExecutePlan(&ctx, &request, &response);
  REQUIRE(status.ok());
  REQUIRE_FALSE(response.ok());

  const auto terminal = parse_terminal(response);
  REQUIRE_FALSE(terminal.ok());
  REQUIRE(terminal.steps_size() == 2);
  REQUIRE(terminal.steps(0).step_id() == "instance");
  REQUIRE(terminal.steps(0).status().state() == tensorcast::node_agent::v1::OPERATION_STATE_FAILED);
  REQUIRE(terminal.steps(0).status().error().status_code() == "FAILED_PRECONDITION");
  REQUIRE(terminal.steps(0).status().message() == "instance target does not match this agent");
  REQUIRE(terminal.steps(1).step_id() == "cluster");
  REQUIRE(terminal.steps(1).status().state() == tensorcast::node_agent::v1::OPERATION_STATE_CANCELLED);
}

TEST_CASE("ExecutePlan prefetch_set fails closed on mismatched set digest", "[daemon][ingress][plan]") {
  auto harness = make_harness();
  auto& svc = harness->service();
  const std::string daemon_id = harness->kernel().worker_identity_store().daemon_id();
  REQUIRE_FALSE(daemon_id.empty());

  tensorcast::daemon::v2::ExecutePlanRequest request;
  request.set_execution_class(tensorcast::daemon::v2::PLAN_EXECUTION_CLASS_TERMINAL_ONLY);
  request.mutable_plan()->mutable_context()->set_request_id("req-prefetch-set");
  auto* step = request.mutable_plan()->add_steps();
  step->set_step_id("s1");
  step->mutable_target()->set_target_type(tensorcast::plan::v1::TARGET_TYPE_WORKER);
  step->mutable_target()->set_target_id(daemon_id);
  auto* action = step->mutable_action()->mutable_prefetch_set();
  action->set_device_id(-1);
  action->mutable_artifact_set()->set_set_digest_hex("deadbeef");
  action->mutable_artifact_set()->set_item_count(1);
  action->mutable_artifact_set()->set_carrier_form("inline");
  action->mutable_artifact_set()->add_inline_items()->CopyFrom(make_selection());

  tensorcast::daemon::v2::ExecutePlanResponse response;
  grpc::ServerContext ctx;
  const auto status = svc.ExecutePlan(&ctx, &request, &response);
  REQUIRE(status.ok());
  REQUIRE_FALSE(response.ok());

  const auto terminal = parse_terminal(response);
  REQUIRE_FALSE(terminal.ok());
  REQUIRE(terminal.steps_size() == 1);
  REQUIRE(terminal.steps(0).status().state() == tensorcast::node_agent::v1::OPERATION_STATE_FAILED);
  REQUIRE(terminal.steps(0).status().error().status_code() == "FAILED_PRECONDITION");
  REQUIRE(
      terminal.steps(0).status().message() ==
      "prefetch_set failed: ArtifactSetRef digest does not match resolved canonical item set");
}
