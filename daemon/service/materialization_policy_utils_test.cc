// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/materialization_policy_utils.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "core/store/testing/global_store_client_stub.h"

namespace {

using tensorcast::daemon::materialization_policy::apply_group_realization_begin_context_to_transport_context;
using tensorcast::daemon::materialization_policy::begin_or_join_group_realization_if_enabled;
using tensorcast::daemon::materialization_policy::build_controller_realization_plan;
using tensorcast::daemon::materialization_policy::controller_resource_manager_linkage_for;
using tensorcast::daemon::materialization_policy::ControllerRealizationPlan;
using tensorcast::daemon::materialization_policy::default_collective_policy_for_mapped_target;
using tensorcast::daemon::materialization_policy::GroupRealizationPreparedMemberContext;
using tensorcast::daemon::materialization_policy::report_group_realization_prepared_if_enabled;
using tensorcast::daemon::materialization_policy::require_controller_export_kind;
using tensorcast::daemon::materialization_policy::require_controller_resource_authority;
using tensorcast::daemon::materialization_policy::resolve_group_realization_transport_context;
using tensorcast::daemon::materialization_policy::resolve_materialization_request_context;
using tensorcast::daemon::materialization_policy::resolve_operation_transport_context;
using tensorcast::daemon::materialization_policy::validate_controller_process_visible_export_authorities;
using tensorcast::daemon::materialization_policy::validate_controller_realization_plan;
using tensorcast::daemon::materialization_policy::validate_group_realization_staged_publish_supported;
using tensorcast::store::loading::CollectiveLoadGroupHint;
using tensorcast::store::loading::ExecutionTopologyContext;
namespace global_store = tensorcast::global_store::v1;
namespace common = tensorcast::common::v1;
namespace operation = tensorcast::operation::v1;
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
    response.mutable_version_set()->set_manifest_hash("manifest-hash-17");
    response.mutable_version_set()->set_manifest_generation(17);
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

bool has_resource_authority(const ControllerRealizationPlan& plan, std::string_view authority) {
  return std::any_of(
      plan.resource_envelope.resource_authorities.begin(),
      plan.resource_envelope.resource_authorities.end(),
      [authority](const std::string& current) { return std::string_view(current) == authority; });
}

bool is_sha256_hex(std::string_view value) {
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char ch) {
           return std::isxdigit(static_cast<unsigned char>(ch)) != 0;
         });
}

void check_controller_source_selection_digest(const std::optional<std::string>& digest) {
  REQUIRE(digest.has_value());
  CHECK(is_sha256_hex(*digest));
  CHECK(*digest != "73656c656374696f6e2d68617368");
}

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

void fill_target_layout(v2::TargetLayout* layout) {
  layout->set_tensor_spec_kind(v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);
  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(1024);
  auto* offset = layout->add_offsets();
  offset->set_name("tensor");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(1024);
}

std::string target_layout_bytes() {
  v2::TargetLayout layout;
  fill_target_layout(&layout);
  return layout.SerializeAsString();
}

void fill_checkpoint_source(operation::ServingBindingSourceRef* source) {
  source->set_source_kind(operation::SERVING_BINDING_SOURCE_KIND_CHECKPOINT_ARTIFACT);
  source->set_artifact_selection_digest("source-selection");
  source->set_source_artifact_ref("mi2:checkpoint");
  source->set_source_schema_hash("source-schema");
}

void fill_member(operation::ServingBindingMemberRef* member, std::string_view member_id, uint32_t index) {
  member->set_member_id(std::string(member_id));
  member->set_member_index(index);
  member->set_member_count(2);
  member->set_group_id("group-1");
}

void fill_serving_target(
    operation::ServingBindingTarget* target,
    std::string_view member_id,
    uint32_t index,
    std::string_view device_uuid) {
  target->set_runtime("vllm");
  target->set_device(std::format("cuda:{}", index));
  target->set_device_uuid(std::string(device_uuid));
  fill_checkpoint_source(target->mutable_source());
  target->mutable_topology()->set_schema_topology_digest("topology-schema");
  fill_member(target->mutable_member(), member_id, index);
  target->set_model_config_digest("model-config");
  target->set_serving_build_digest("serving-build");
  auto* layout = target->mutable_resolved_layout();
  layout->set_binding_layout_id(std::format("layout-{}", index));
  layout->mutable_source()->CopyFrom(target->source());
  layout->mutable_source_reuse()->set_mode(operation::SERVING_BINDING_SOURCE_REUSE_MODE_CHECKPOINT_TO_SERVING);
  layout->mutable_topology()->CopyFrom(target->topology());
  layout->mutable_member()->CopyFrom(target->member());
  layout->set_target_layout(target_layout_bytes());
  layout->set_target_index_bytes("target-index");
  layout->set_target_layout_hash(std::format("target-layout-{}", index));
  layout->set_tensor_schema_hash("tensor-schema");
  layout->set_spec_digest(std::format("spec-{}", index));
}

TEST_CASE("Controller realization plan mirrors caller target materialization", "[daemon][materialization][policy]") {
  v2::MaterializeIntoTargetRequest request;
  request.mutable_selection()->set_artifact_id("mi2:test:artifact");
  request.mutable_selection()->set_selection_hash("selection-hash");
  request.set_operation_id("op-1");
  request.set_pid(1234);
  request.set_device_uuid("GPU-0");
  fill_target_layout(request.mutable_target_layout());
  auto request_context_or = resolve_materialization_request_context(nullptr);
  REQUIRE(request_context_or.ok());
  auto transport_context = resolve_operation_transport_context("op-1");

  auto plan_or =
      build_controller_realization_plan(request, *request_context_or, transport_context, nullptr, "mi2:test:artifact");

  REQUIRE(plan_or.ok());
  CHECK(plan_or->target.target_kind == "caller_tensors");
  CHECK(plan_or->target.resolved_artifact_id == "mi2:test:artifact");
  CHECK_FALSE(plan_or->target.target_layout_digest.empty());
  CHECK(plan_or->target.member_count == 1);
  CHECK(plan_or->strategy.source_selection_mode == "single_selection");
  CHECK(plan_or->strategy.source_coordination == "single_request");
  check_controller_source_selection_digest(plan_or->strategy.source_selection_digest);
  CHECK(plan_or->lifecycle.capability == "caller_tensors");
  CHECK((plan_or->lifecycle.release_policy == std::vector<std::string>{"release_external_target_storage_lease"}));
  CHECK(plan_or->resource_envelope.projection_kind == "completion");
  CHECK(plan_or->resource_envelope.body_backing_intent.retention_intent == "ephemeral");
  CHECK(plan_or->resource_envelope.body_backing_intent.sharing_intent == "private_local");
  CHECK(has_resource_authority(*plan_or, "caller_allocation"));
  REQUIRE(require_controller_export_kind(*plan_or, "registered_region_direct_write", "MaterializeIntoTarget").ok());
  REQUIRE(require_controller_resource_authority(*plan_or, "caller_allocation", "MaterializeIntoTarget").ok());
}

TEST_CASE(
    "Controller source selection digest is independent from target layout digest",
    "[daemon][materialization][policy]") {
  v2::MaterializeIntoTargetRequest base_request;
  base_request.mutable_selection()->set_artifact_id("mi2:test:artifact");
  base_request.mutable_selection()->set_logical_layout_hash("logical-layout");
  base_request.mutable_selection()->set_selection_hash("selection-hash");
  base_request.set_pid(1234);
  base_request.set_device_uuid("GPU-0");
  fill_target_layout(base_request.mutable_target_layout());
  auto request_context_or = resolve_materialization_request_context(nullptr);
  REQUIRE(request_context_or.ok());

  const auto first_transport = resolve_operation_transport_context("op-identity-1");
  auto first_or = build_controller_realization_plan(
      base_request, *request_context_or, first_transport, nullptr, "mi2:test:artifact");
  REQUIRE(first_or.ok());

  v2::MaterializeIntoTargetRequest target_changed_request = base_request;
  target_changed_request.mutable_target_layout()->mutable_storages(0)->set_storage_length(2048);
  const auto target_changed_transport = resolve_operation_transport_context("op-identity-2");
  auto target_changed_or = build_controller_realization_plan(
      target_changed_request, *request_context_or, target_changed_transport, nullptr, "mi2:test:artifact");
  REQUIRE(target_changed_or.ok());

  v2::MaterializeIntoTargetRequest source_changed_request = base_request;
  source_changed_request.mutable_selection()->set_logical_layout_hash("other-logical-layout");
  const auto source_changed_transport = resolve_operation_transport_context("op-identity-3");
  auto source_changed_or = build_controller_realization_plan(
      source_changed_request, *request_context_or, source_changed_transport, nullptr, "mi2:test:artifact");
  REQUIRE(source_changed_or.ok());

  check_controller_source_selection_digest(first_or->strategy.source_selection_digest);
  CHECK(first_or->strategy.source_selection_digest == target_changed_or->strategy.source_selection_digest);
  CHECK(first_or->target.target_layout_digest != target_changed_or->target.target_layout_digest);
  CHECK(first_or->strategy.source_selection_digest != source_changed_or->strategy.source_selection_digest);
}

TEST_CASE("Controller realization plan mirrors binding creation ownership", "[daemon][materialization][policy]") {
  v2::CreateBindingRequest daemon_request;
  daemon_request.set_ownership(v2::BindingOwnership::BINDING_OWNERSHIP_DAEMON);
  daemon_request.set_pid(1234);
  daemon_request.set_device_uuid("GPU-0");
  daemon_request.set_binding_layout_id("layout-0");
  daemon_request.set_target_index_bytes("target-index");
  fill_target_layout(daemon_request.mutable_target_layout());

  auto daemon_plan_or = build_controller_realization_plan(daemon_request);

  REQUIRE(daemon_plan_or.ok());
  CHECK(daemon_plan_or->target.target_kind == "binding_owned");
  CHECK(daemon_plan_or->target.resolved_artifact_id.empty());
  CHECK(daemon_plan_or->strategy.source_selection_mode == "none");
  CHECK(daemon_plan_or->strategy.source_coordination == "binding_allocation");
  CHECK(daemon_plan_or->lifecycle.export_lifetime_kind == "handle_lease");
  CHECK(daemon_plan_or->lifecycle.mutability_contract == "binding_controlled_mutable");
  CHECK(
      (daemon_plan_or->lifecycle.release_policy ==
       std::vector<std::string>{
           "release_handle_lease",
           "close_binding",
       }));
  CHECK(daemon_plan_or->resource_envelope.backing_kind == "daemon_binding");
  CHECK(daemon_plan_or->resource_envelope.export_kind == "cuda_ipc_lease");
  CHECK(daemon_plan_or->resource_envelope.projection_kind == "binding");
  CHECK(daemon_plan_or->resource_envelope.body_backing_intent.preferred_residency == "gpu");
  CHECK(daemon_plan_or->resource_envelope.body_backing_intent.retention_intent == "retained");
  CHECK(has_resource_authority(*daemon_plan_or, "BodyBackingManager"));
  CHECK(has_resource_authority(*daemon_plan_or, "HandleLeaseRegistry"));
  CHECK(has_resource_authority(*daemon_plan_or, "SessionLifecycleManager"));
  CHECK(has_resource_authority(*daemon_plan_or, "LifecycleKernel"));

  v2::CreateBindingRequest adopted_request;
  adopted_request.set_ownership(v2::BindingOwnership::BINDING_OWNERSHIP_CLIENT);
  adopted_request.set_pid(1234);
  adopted_request.set_device_uuid("GPU-0");
  adopted_request.set_binding_layout_id("layout-0");
  adopted_request.set_target_index_bytes("target-index");
  adopted_request.set_source_artifact_id("mi2:source-override");
  adopted_request.mutable_initial_selection()->set_artifact_id("mi2:source");
  adopted_request.mutable_initial_selection()->set_selection_hash("selection-hash");
  fill_target_layout(adopted_request.mutable_target_layout());

  auto adopted_plan_or = build_controller_realization_plan(adopted_request);

  REQUIRE(adopted_plan_or.ok());
  CHECK(adopted_plan_or->target.target_kind == "binding_adopted");
  CHECK(adopted_plan_or->target.resolved_artifact_id == "mi2:source-override");
  CHECK(adopted_plan_or->strategy.source_selection_mode == "single_selection");
  CHECK(adopted_plan_or->strategy.source_coordination == "binding_initial_value");
  check_controller_source_selection_digest(adopted_plan_or->strategy.source_selection_digest);
  CHECK(adopted_plan_or->lifecycle.export_lifetime_kind == "binding_registry");
  CHECK(adopted_plan_or->lifecycle.mutability_contract == "caller_region_borrowed");
  CHECK(adopted_plan_or->resource_envelope.backing_kind == "caller_region");
  CHECK(adopted_plan_or->resource_envelope.export_kind == "publication_token_or_none");
  CHECK(adopted_plan_or->resource_envelope.body_backing_intent.retention_intent == "ephemeral");
  CHECK(has_resource_authority(*adopted_plan_or, "caller_allocation"));
  CHECK(has_resource_authority(*adopted_plan_or, "BindingRegistry"));
}

TEST_CASE(
    "Controller realization plan rejects process-visible exports without token-backed lifetime",
    "[daemon][materialization][policy]") {
  ControllerRealizationPlan plan;
  plan.target.target_kind = "tensor_dict";
  plan.lifecycle.capability = "tensor_dict";
  plan.lifecycle.export_lifetime_kind = "request_scoped";
  plan.lifecycle.release_policy = {"release_handle_lease"};
  plan.resource_envelope.backing_kind = "daemon_replica";
  plan.resource_envelope.export_kind = "cuda_ipc_lease";
  plan.resource_envelope.projection_kind = "tensor_dict";
  plan.resource_envelope.owner_kind = "caller_pid";
  plan.resource_envelope.release_policy = plan.lifecycle.release_policy;
  plan.resource_envelope.resource_authorities = {"HandleLeaseRegistry"};

  auto status = validate_controller_realization_plan(plan);

  CHECK_FALSE(status.ok());
  CHECK(std::string(status.message()) == "process-visible controller exports require token-backed lifetime");
}

TEST_CASE(
    "Controller execution admission checks export kind and resource authorities",
    "[daemon][materialization][policy]") {
  v2::CreateBindingRequest daemon_request;
  daemon_request.set_ownership(v2::BindingOwnership::BINDING_OWNERSHIP_DAEMON);
  daemon_request.set_pid(1234);
  daemon_request.set_device_uuid("GPU-0");
  daemon_request.set_binding_layout_id("layout-0");
  daemon_request.set_target_index_bytes("target-index");
  fill_target_layout(daemon_request.mutable_target_layout());
  auto plan_or = build_controller_realization_plan(daemon_request);
  REQUIRE(plan_or.ok());

  REQUIRE(validate_controller_process_visible_export_authorities(*plan_or, "CreateBinding").ok());
  REQUIRE(require_controller_export_kind(*plan_or, "cuda_ipc_lease", "CreateBinding").ok());
  CHECK(plan_or->resource_envelope.manager_linkage == controller_resource_manager_linkage_for(*plan_or));

  auto mismatched_export = require_controller_export_kind(*plan_or, "publication_token_or_none", "CreateBinding");
  REQUIRE_FALSE(mismatched_export.ok());
  CHECK(
      mismatched_export.message().find("expected controller export_kind=publication_token_or_none") !=
      std::string::npos);

  plan_or->resource_envelope.resource_authorities = {"HandleLeaseRegistry"};
  auto missing_authority = validate_controller_process_visible_export_authorities(*plan_or, "CreateBinding");
  REQUIRE_FALSE(missing_authority.ok());
  CHECK(missing_authority.message().find("SessionLifecycleManager") != std::string::npos);
}

TEST_CASE(
    "Controller resource envelope maps manager linkage and execution commit concepts",
    "[daemon][materialization][policy]") {
  v2::MaterializeReplicaRequest request;
  request.mutable_selection()->set_artifact_id("mi2:tensor-dict");
  request.set_wait_for_completion(true);
  request.set_pid(1234);
  request.set_device_uuid("GPU-0");
  request.set_target_device_type(v2::DeviceType::DEVICE_TYPE_GPU);
  request.set_size_bytes(4096);
  ExecutionTopologyContext execution_topology;
  execution_topology.collective_load_group =
      CollectiveLoadGroupHint{.group_id = "same-host-load", .world_size = 2, .rank = 0};
  auto request_context_or = resolve_materialization_request_context(nullptr, execution_topology);
  REQUIRE(request_context_or.ok());
  auto transport_context = resolve_operation_transport_context("");
  common::ArtifactSelection resolved_selection;
  resolved_selection.set_artifact_id("mi2:tensor-dict");
  resolved_selection.set_selection_hash("selection-hash");
  resolved_selection.set_logical_layout_hash("logical-layout");

  auto plan_or = build_controller_realization_plan(
      request,
      *request_context_or,
      transport_context,
      nullptr,
      "mi2:tensor-dict",
      resolved_selection,
      /*cpu_target=*/false,
      /*no_lease=*/false);

  REQUIRE(plan_or.ok());
  CHECK(
      plan_or->resource_envelope.manager_linkage.body_backing_manager ==
      "BodyBackingManager:daemon_replica:ephemeral:local_read_mostly");
  CHECK(plan_or->resource_envelope.manager_linkage.handle_lease_registry == "HandleLeaseRegistry:cuda_ipc_lease");
  CHECK(
      plan_or->resource_envelope.manager_linkage.session_lifecycle_manager ==
      "SessionLifecycleManager:cuda_ipc_lease:pid_use_lease");
  CHECK(plan_or->resource_envelope.manager_linkage.lifecycle_kernel == "LifecycleKernel:handle_lease_capability");
  CHECK(
      plan_or->resource_envelope.manager_linkage.execution_commit_report ==
      "ExecutionCommitReport:replica_materialization:collective_capable");

  auto mismatched_plan = *plan_or;
  mismatched_plan.resource_envelope.manager_linkage.execution_commit_report = "none";
  const auto status = validate_controller_realization_plan(mismatched_plan);
  REQUIRE_FALSE(status.ok());
  CHECK(status.message().find("resource manager linkage") != std::string::npos);
}

TEST_CASE("Controller realization plan mirrors owned binding materialization", "[daemon][materialization][policy]") {
  auto options = build_group_realization_options();
  options.set_require_staged_publish(true);
  auto client = std::make_shared<RecordingGroupRealizationClient>();
  auto begin_or = begin_or_join_group_realization_if_enabled(client, &options, "daemon-a", "session-a", "worker-a");
  REQUIRE(begin_or.ok());
  REQUIRE(begin_or->has_value());
  auto transport_or = resolve_group_realization_transport_context("create-op", &options);
  REQUIRE(transport_or.ok());
  apply_group_realization_begin_context_to_transport_context(**begin_or, &*transport_or);
  ExecutionTopologyContext execution_topology;
  execution_topology.collective_load_group =
      CollectiveLoadGroupHint{.group_id = "same-host-load", .world_size = 2, .rank = 0};
  auto request_context_or = resolve_materialization_request_context(nullptr, execution_topology);
  REQUIRE(request_context_or.ok());

  v2::CreateOwnedBindingRequest request;
  request.mutable_source_selection()->set_artifact_id("mi2:source");
  request.mutable_source_selection()->set_selection_hash("selection-hash");
  request.set_pid(1234);
  request.set_device_uuid("GPU-0");
  request.set_operation_id("create-op");
  request.set_binding_layout_id("layout-0");
  request.set_target_index_bytes("target-index");
  request.mutable_group_realization()->CopyFrom(options);
  fill_target_layout(request.mutable_target_layout());
  common::ArtifactSelection resolved_selection = request.source_selection();
  resolved_selection.set_artifact_id("artifact-frozen-rank0");

  auto plan_or = build_controller_realization_plan(
      request, *request_context_or, *transport_or, &**begin_or, "artifact-frozen-rank0", resolved_selection);

  REQUIRE(plan_or.ok());
  CHECK(plan_or->target.target_kind == "binding_owned");
  CHECK(plan_or->target.member_count == 2);
  REQUIRE(plan_or->target.group_id.has_value());
  CHECK(*plan_or->target.group_id == "model-v17");
  REQUIRE(plan_or->target.part_id.has_value());
  CHECK(*plan_or->target.part_id == "rank0");
  REQUIRE(plan_or->target.operation_id.has_value());
  CHECK(*plan_or->target.operation_id == "create-op");
  CHECK(plan_or->strategy.source_selection_mode == "per_part_selection");
  CHECK(plan_or->strategy.source_coordination == "group_realization_transport");
  CHECK(plan_or->strategy.collective_policy == v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST);
  CHECK(
      (plan_or->strategy.group_barriers ==
       std::vector<std::string>{"member_readiness", "group_acquire", "staged_values", "publish_barrier"}));
  CHECK(plan_or->lifecycle.capability == "binding_owned");
  CHECK(plan_or->lifecycle.staged_value_count == 1);
  CHECK(plan_or->lifecycle.acquire_claim_count == 1);
  CHECK(plan_or->lifecycle.publish_barrier);
  CHECK(plan_or->resource_envelope.backing_kind == "daemon_binding");
  CHECK(plan_or->resource_envelope.export_kind == "cuda_ipc_lease");
  CHECK(plan_or->resource_envelope.projection_kind == "staged_binding");
}

TEST_CASE("Controller realization plan mirrors owned binding refill", "[daemon][materialization][policy]") {
  v2::RefillOwnedBindingRequest request;
  request.set_binding_id("binding-1");
  request.set_artifact_id("mi2:new-source");
  request.set_operation_id("refill-op");
  request.set_collective_policy(v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE);
  request.mutable_source_selection()->set_artifact_id("mi2:new-source");
  request.mutable_source_selection()->set_selection_hash("selection-hash");
  ExecutionTopologyContext execution_topology;
  execution_topology.collective_load_group =
      CollectiveLoadGroupHint{.group_id = "refill-load", .world_size = 2, .rank = 0};
  auto request_context_or = resolve_materialization_request_context(nullptr, execution_topology);
  REQUIRE(request_context_or.ok());
  auto transport_context = resolve_operation_transport_context("refill-op");
  v2::TargetLayout target_layout;
  fill_target_layout(&target_layout);

  common::ArtifactSelection resolved_selection = request.source_selection();
  auto plan_or = build_controller_realization_plan(
      request,
      *request_context_or,
      transport_context,
      nullptr,
      "mi2:new-source",
      resolved_selection,
      target_layout,
      "GPU-0",
      1234,
      /*mapped=*/true,
      v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE,
      /*execution_only_mutable=*/false);

  REQUIRE(plan_or.ok());
  CHECK(plan_or->target.target_kind == "binding_owned_refill");
  CHECK(plan_or->target.resolved_artifact_id == "mi2:new-source");
  REQUIRE(plan_or->target.operation_id.has_value());
  CHECK(*plan_or->target.operation_id == "refill-op");
  CHECK(plan_or->strategy.source_selection_mode == "single_selection");
  CHECK(plan_or->strategy.collective_policy == v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE);
  check_controller_source_selection_digest(plan_or->strategy.source_selection_digest);
  CHECK(plan_or->lifecycle.capability == "binding_owned");
  CHECK(plan_or->lifecycle.export_lifetime_kind == "binding_current_value");
  CHECK(plan_or->lifecycle.mutability_contract == "binding_controlled_read_only");
  CHECK(
      (plan_or->lifecycle.release_policy ==
       std::vector<std::string>{
           "replace_binding_current_value",
           "retire_publication_token",
       }));
  CHECK(plan_or->resource_envelope.backing_kind == "daemon_binding");
  CHECK(plan_or->resource_envelope.export_kind == "publication_token_or_none");
  CHECK(plan_or->resource_envelope.projection_kind == "binding_mapped_refill");
  CHECK(plan_or->resource_envelope.owner_kind == "binding_registry");
  REQUIRE(require_controller_export_kind(*plan_or, "publication_token_or_none", "RefillOwnedBinding").ok());
  REQUIRE(require_controller_resource_authority(*plan_or, "BodyBackingManager", "RefillOwnedBinding").ok());
  REQUIRE(require_controller_resource_authority(*plan_or, "BindingRegistry", "RefillOwnedBinding").ok());

  auto mutable_plan_or = build_controller_realization_plan(
      request,
      *request_context_or,
      transport_context,
      nullptr,
      "mi2:new-source",
      resolved_selection,
      target_layout,
      "GPU-0",
      1234,
      /*mapped=*/false,
      v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE,
      /*execution_only_mutable=*/true);

  REQUIRE(mutable_plan_or.ok());
  CHECK(mutable_plan_or->resource_envelope.export_kind == "none");
  CHECK(mutable_plan_or->resource_envelope.projection_kind == "binding_refill");
  REQUIRE(require_controller_export_kind(*mutable_plan_or, "none", "RefillOwnedBinding").ok());
  REQUIRE(require_controller_resource_authority(*mutable_plan_or, "BodyBackingManager", "RefillOwnedBinding").ok());
  REQUIRE(require_controller_resource_authority(*mutable_plan_or, "BindingRegistry", "RefillOwnedBinding").ok());
}

v2::AssemblyRequirementSetRef make_assembly_requirements() {
  v2::AssemblyRequirementSetRef requirements;
  requirements.set_carrier_form("inline");
  requirements.set_requirement_count(2);
  auto* first = requirements.add_inline_requirements();
  first->set_slot_id("rank0");
  first->mutable_target()->set_kind(v2::ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW);
  first->mutable_target()->set_structural_view_id("view-rank0");
  first->set_coverage_contract("pp_structural_view");
  auto* second = requirements.add_inline_requirements();
  second->set_slot_id("rank1");
  second->mutable_target()->set_kind(v2::ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW);
  second->mutable_target()->set_structural_view_id("view-rank1");
  second->set_coverage_contract("pp_structural_view");
  return requirements;
}

v2::AssemblyAttemptIntent make_assembly_intent(v2::AssemblyCloseoutKind closeout_kind) {
  v2::AssemblyAttemptIntent intent;
  intent.set_layout_id("layout-assembly");
  intent.mutable_requirements()->CopyFrom(make_assembly_requirements());
  intent.mutable_closeout_contract()->set_kind(closeout_kind);
  intent.set_attempt_intent_digest("attempt-intent-digest");
  return intent;
}

TEST_CASE("Controller realization plan mirrors assembly attempt coordinator", "[daemon][materialization][policy]") {
  v2::StartAssemblyAttemptRequest request;
  request.set_layout_id("layout-assembly");
  request.mutable_requirements()->CopyFrom(make_assembly_requirements());
  auto intent = make_assembly_intent(v2::ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY);

  auto plan_or =
      build_controller_realization_plan(request, intent, "attempt-1", "cgid:assembly-workspace-1", "assembly-op-1");

  REQUIRE(plan_or.ok());
  CHECK(plan_or->target.target_kind == "assembly_attempt");
  CHECK(plan_or->target.resolved_artifact_id == "cgid:assembly-workspace-1");
  CHECK(plan_or->target.member_count == 2);
  REQUIRE(plan_or->target.group_id.has_value());
  CHECK(*plan_or->target.group_id == "attempt-1");
  REQUIRE(plan_or->target.operation_id.has_value());
  CHECK(*plan_or->target.operation_id == "assembly-op-1");
  CHECK(plan_or->strategy.source_selection_mode == "assembly_requirements");
  CHECK(plan_or->strategy.source_coordination == "assembly_attempt_coordinator");
  CHECK(
      (plan_or->strategy.group_barriers ==
       std::vector<std::string>{
           "requirement_registration",
           "multi_requirement_readiness",
           "operation_coordinator",
       }));
  REQUIRE(plan_or->strategy.transaction_id.has_value());
  CHECK(*plan_or->strategy.transaction_id == "attempt-1");
  REQUIRE(plan_or->strategy.source_selection_digest.has_value());
  CHECK(*plan_or->strategy.source_selection_digest == "attempt-intent-digest");
  CHECK(plan_or->lifecycle.capability == "assembly_attempt");
  CHECK(plan_or->lifecycle.export_lifetime_kind == "operation_lease");
  CHECK(plan_or->lifecycle.staged_value_count == 2);
  CHECK(plan_or->lifecycle.acquire_claim_count == 1);
  CHECK(plan_or->resource_envelope.backing_kind == "assembly_workspace");
  CHECK(plan_or->resource_envelope.export_kind == "operation_lease");
  CHECK(plan_or->resource_envelope.projection_kind == "operation_ref");
  CHECK(plan_or->resource_envelope.body_backing_intent.retention_intent == "retained");
  CHECK(plan_or->resource_envelope.body_backing_intent.sharing_intent == "private_local");
  CHECK(has_resource_authority(*plan_or, "assembly_registry"));
  CHECK(has_resource_authority(*plan_or, "OperationLeaseRegistry"));
  CHECK(has_resource_authority(*plan_or, "SessionLifecycleManager"));
}

TEST_CASE("Controller realization plan mirrors assembly attempt sealing", "[daemon][materialization][policy]") {
  v2::SealAssemblyAttemptRequest request;
  request.set_attempt_id("attempt-1");
  v2::AssemblyAttemptRecord record;
  record.set_attempt_id("attempt-1");
  record.set_workspace_assembly_id("cgid:assembly-workspace-1");
  record.mutable_intent()->CopyFrom(make_assembly_intent(v2::ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH));

  auto plan_or = build_controller_realization_plan(request, record, "assembly-op-1");

  REQUIRE(plan_or.ok());
  CHECK(plan_or->target.target_kind == "assembly_attempt_seal");
  CHECK(plan_or->target.resolved_artifact_id == "cgid:assembly-workspace-1");
  CHECK(plan_or->target.member_count == 2);
  REQUIRE(plan_or->target.group_id.has_value());
  CHECK(*plan_or->target.group_id == "attempt-1");
  CHECK(plan_or->strategy.source_selection_mode == "assembly_requirements");
  CHECK(plan_or->strategy.source_coordination == "representation_publish_closeout");
  CHECK(
      (plan_or->strategy.group_barriers ==
       std::vector<std::string>{
           "readiness_cut",
           "seal_cut",
           "representation_publish_closeout",
       }));
  CHECK(plan_or->lifecycle.capability == "assembly_seal");
  CHECK(plan_or->lifecycle.mutability_contract == "sealed_artifact_immutable");
  CHECK(plan_or->lifecycle.publish_barrier);
  CHECK(
      (plan_or->lifecycle.release_policy ==
       std::vector<std::string>{
           "release_operation_lease",
           "finalize_slot_occupancies",
           "publish_workspace_seal_binding",
       }));
  CHECK(plan_or->resource_envelope.owner_kind == "daemon_coordinator");
  CHECK(has_resource_authority(*plan_or, "assembly_registry"));
  CHECK(has_resource_authority(*plan_or, "OperationLeaseRegistry"));
  CHECK(has_resource_authority(*plan_or, "SessionLifecycleManager"));
}

TEST_CASE("Controller realization plan mirrors low-level assembly sealing", "[daemon][materialization][policy]") {
  v2::SealAssemblyRequest sync_request;
  sync_request.set_assembly_id("cgid:assembly-workspace-1");
  sync_request.set_publish_canonical(true);

  auto sync_plan_or = build_controller_realization_plan(sync_request);

  REQUIRE(sync_plan_or.ok());
  CHECK(sync_plan_or->target.target_kind == "assembly_seal");
  CHECK(sync_plan_or->target.resolved_artifact_id == "cgid:assembly-workspace-1");
  CHECK(sync_plan_or->strategy.source_selection_mode == "assembly_workspace");
  CHECK(sync_plan_or->strategy.source_coordination == "seal_and_publish_canonical");
  CHECK((sync_plan_or->strategy.group_barriers == std::vector<std::string>{"seal_cut"}));
  CHECK(sync_plan_or->lifecycle.export_lifetime_kind == "request_scoped");
  CHECK(sync_plan_or->resource_envelope.export_kind == "none");
  CHECK(sync_plan_or->resource_envelope.projection_kind == "artifact_descriptor");

  v2::StartSealAssemblyRequest async_request;
  async_request.set_assembly_id("cgid:assembly-workspace-1");
  async_request.set_layout_id("layout-assembly");

  auto async_plan_or = build_controller_realization_plan(async_request, "seal-op-1");

  REQUIRE(async_plan_or.ok());
  CHECK(async_plan_or->target.target_kind == "assembly_seal");
  CHECK(async_plan_or->target.resolved_artifact_id == "cgid:assembly-workspace-1");
  REQUIRE(async_plan_or->target.operation_id.has_value());
  CHECK(*async_plan_or->target.operation_id == "seal-op-1");
  CHECK(async_plan_or->strategy.source_coordination == "explicit_layout");
  CHECK(
      (async_plan_or->strategy.group_barriers ==
       std::vector<std::string>{
           "operation_coordinator",
           "seal_cut",
           "post_seal_policy",
       }));
  CHECK(async_plan_or->lifecycle.export_lifetime_kind == "operation_lease");
  CHECK(async_plan_or->lifecycle.acquire_claim_count == 1);
  CHECK(async_plan_or->resource_envelope.export_kind == "operation_lease");
  CHECK(async_plan_or->resource_envelope.projection_kind == "operation_ref");
}

TEST_CASE(
    "Controller realization plan mirrors tensor dict replica materialization",
    "[daemon][materialization][policy]") {
  v2::MaterializeReplicaRequest request;
  request.mutable_selection()->set_artifact_id("mi2:tensor-dict");
  request.set_wait_for_completion(true);
  request.set_pid(1234);
  request.set_device_uuid("GPU-0");
  request.set_target_device_type(v2::DeviceType::DEVICE_TYPE_GPU);
  request.set_size_bytes(4096);
  request.set_export_policy(v2::ExportPolicy::EXPORT_POLICY_FORCE);
  request.mutable_collective_load_group()->set_group_id("same-host-load");
  request.mutable_collective_load_group()->set_world_size(2);
  request.mutable_collective_load_group()->set_rank(0);
  ExecutionTopologyContext execution_topology;
  execution_topology.collective_load_group =
      CollectiveLoadGroupHint{.group_id = "same-host-load", .world_size = 2, .rank = 0};
  auto request_context_or = resolve_materialization_request_context(nullptr, execution_topology);
  REQUIRE(request_context_or.ok());
  auto transport_context = resolve_operation_transport_context("");
  common::ArtifactSelection resolved_selection;
  resolved_selection.set_artifact_id("mi2:tensor-dict");
  resolved_selection.set_selection_hash("selection-hash");
  resolved_selection.set_logical_layout_hash("logical-layout");

  auto plan_or = build_controller_realization_plan(
      request,
      *request_context_or,
      transport_context,
      nullptr,
      "mi2:tensor-dict",
      resolved_selection,
      /*cpu_target=*/false,
      /*no_lease=*/false);

  REQUIRE(plan_or.ok());
  CHECK(plan_or->target.target_kind == "tensor_dict");
  CHECK(plan_or->target.resolved_artifact_id == "mi2:tensor-dict");
  CHECK(plan_or->target.target_layout_digest == "6c6f676963616c2d6c61796f7574");
  CHECK(plan_or->target.device_uuid == "GPU-0");
  CHECK(plan_or->target.owner_pid == 1234);
  CHECK(plan_or->strategy.source_selection_mode == "single_selection");
  CHECK(plan_or->strategy.source_coordination == "single_request");
  CHECK(plan_or->strategy.collective_policy == v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST);
  check_controller_source_selection_digest(plan_or->strategy.source_selection_digest);
  CHECK(plan_or->lifecycle.capability == "tensor_dict");
  CHECK(plan_or->lifecycle.export_lifetime_kind == "handle_lease");
  CHECK(
      (plan_or->lifecycle.release_policy ==
       std::vector<std::string>{
           "release_handle_lease",
           "release_pid_ref",
           "release_replica_session",
       }));
  CHECK(plan_or->lifecycle.acquire_claim_count == 1);
  CHECK(plan_or->resource_envelope.backing_kind == "daemon_replica");
  CHECK(plan_or->resource_envelope.export_kind == "cuda_ipc_lease");
  CHECK(plan_or->resource_envelope.projection_kind == "tensor_dict");
  CHECK(plan_or->resource_envelope.owner_kind == "caller_pid");
  CHECK(plan_or->resource_envelope.body_backing_intent.preferred_residency == "gpu");
  CHECK(plan_or->resource_envelope.body_backing_intent.retention_intent == "ephemeral");
  CHECK(plan_or->resource_envelope.body_backing_intent.sharing_intent == "local_read_mostly");
  CHECK(has_resource_authority(*plan_or, "BodyBackingManager"));
  CHECK(has_resource_authority(*plan_or, "HandleLeaseRegistry"));
  CHECK(has_resource_authority(*plan_or, "SessionLifecycleManager"));
  CHECK(has_resource_authority(*plan_or, "LifecycleKernel"));
}

TEST_CASE(
    "Controller realization plan mirrors retained no-lease replica materialization",
    "[daemon][materialization][policy]") {
  v2::MaterializeReplicaRequest request;
  request.mutable_selection()->set_artifact_id("mi2:prefetch");
  request.set_wait_for_completion(false);
  request.set_lease_mode(v2::LeaseMode::LEASE_MODE_NO_LEASE);
  request.set_device_uuid("GPU-0");
  request.set_target_device_type(v2::DeviceType::DEVICE_TYPE_GPU);
  request.set_size_bytes(4096);
  auto request_context_or = resolve_materialization_request_context(nullptr);
  REQUIRE(request_context_or.ok());
  auto transport_context = resolve_operation_transport_context("");
  common::ArtifactSelection resolved_selection;
  resolved_selection.set_artifact_id("mi2:prefetch");
  resolved_selection.set_selection_hash("selection-hash");
  resolved_selection.set_logical_layout_hash("logical-layout");

  auto plan_or = build_controller_realization_plan(
      request,
      *request_context_or,
      transport_context,
      nullptr,
      "mi2:prefetch",
      resolved_selection,
      /*cpu_target=*/false,
      /*no_lease=*/true);

  REQUIRE(plan_or.ok());
  CHECK(plan_or->target.target_kind == "retained_replica");
  CHECK(plan_or->target.owner_pid == 0);
  CHECK(plan_or->strategy.collective_policy == v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE);
  CHECK(plan_or->lifecycle.capability == "retained_replica");
  CHECK(plan_or->lifecycle.export_lifetime_kind == "daemon_retained");
  CHECK(
      (plan_or->lifecycle.release_policy ==
       std::vector<std::string>{
           "retain_daemon_replica",
           "release_operation_ticket",
       }));
  CHECK(plan_or->lifecycle.acquire_claim_count == 0);
  CHECK(plan_or->resource_envelope.backing_kind == "daemon_retained_replica");
  CHECK(plan_or->resource_envelope.export_kind == "none");
  CHECK(plan_or->resource_envelope.projection_kind == "operation_ticket");
  CHECK(plan_or->resource_envelope.owner_kind == "daemon_session");
  CHECK(plan_or->resource_envelope.body_backing_intent.retention_intent == "retained");
  CHECK(plan_or->resource_envelope.body_backing_intent.stable_retention_requirement == "prefer_stable");
  CHECK(has_resource_authority(*plan_or, "BodyBackingManager"));
  CHECK(has_resource_authority(*plan_or, "SessionLifecycleManager"));
}

TEST_CASE(
    "Controller realization plan mirrors target-set group strategy and lifecycle",
    "[daemon][materialization][policy]") {
  auto options = build_group_realization_options();
  auto client = std::make_shared<RecordingGroupRealizationClient>();
  auto begin_or = begin_or_join_group_realization_if_enabled(client, &options, "daemon-a", "session-a", "worker-a");
  REQUIRE(begin_or.ok());
  REQUIRE(begin_or->has_value());
  auto transport_or = resolve_group_realization_transport_context("op-17", &options);
  REQUIRE(transport_or.ok());
  apply_group_realization_begin_context_to_transport_context(**begin_or, &*transport_or);
  ExecutionTopologyContext execution_topology;
  execution_topology.collective_load_group =
      CollectiveLoadGroupHint{.group_id = "same-host-tp-load", .world_size = 2, .rank = 0};
  auto request_context_or = resolve_materialization_request_context(nullptr, execution_topology);
  REQUIRE(request_context_or.ok());
  v2::MaterializeIntoMappedTargetRequest request;
  request.mutable_group_realization()->CopyFrom(options);
  request.set_operation_id("op-17");
  request.set_pid(1234);
  request.set_device_uuid("GPU-0");
  fill_target_layout(request.mutable_target_layout());

  auto plan_or = build_controller_realization_plan(
      request, *request_context_or, *transport_or, &**begin_or, "artifact-frozen-rank0");

  REQUIRE(plan_or.ok());
  CHECK(plan_or->target.target_kind == "binding_adopted");
  CHECK(plan_or->target.member_count == 2);
  REQUIRE(plan_or->target.group_id.has_value());
  CHECK(*plan_or->target.group_id == "model-v17");
  REQUIRE(plan_or->target.part_id.has_value());
  CHECK(*plan_or->target.part_id == "rank0");
  CHECK(plan_or->strategy.source_selection_mode == "per_part_selection");
  CHECK(plan_or->strategy.source_coordination == "group_realization_transport");
  CHECK(plan_or->strategy.collective_policy == v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST);
  CHECK((plan_or->strategy.group_barriers == std::vector<std::string>{"member_readiness", "group_acquire"}));
  REQUIRE(plan_or->strategy.transaction_id.has_value());
  CHECK(*plan_or->strategy.transaction_id == "txn-17");
  REQUIRE(plan_or->strategy.version_set_id.has_value());
  CHECK(*plan_or->strategy.version_set_id == "vs-17");
  CHECK(plan_or->lifecycle.capability == "binding_adopted");
  CHECK(plan_or->lifecycle.acquire_claim_count == 1);
  CHECK(plan_or->resource_envelope.projection_kind == "binding");
}

TEST_CASE(
    "Controller realization plan mirrors retained serving target-set prefetch",
    "[daemon][materialization][policy]") {
  v2::PrefetchServingBindingRequest request;
  request.mutable_source_selection()->set_artifact_id("mi2:checkpoint");
  fill_checkpoint_source(request.mutable_source());
  request.set_operation_id("prefetch-op");
  request.set_requested_readiness(operation::SERVING_BINDING_READINESS_LOCAL_READY);
  auto* target_set = request.mutable_serving_binding_set_target();
  target_set->set_runtime("vllm");
  target_set->mutable_source()->CopyFrom(request.source());
  target_set->mutable_topology()->set_schema_topology_digest("topology-schema");
  target_set->set_group_id("group-1");
  fill_serving_target(target_set->add_members(), "member-0", 0, "GPU-0");
  fill_serving_target(target_set->add_members(), "member-1", 1, "GPU-1");
  target_set->mutable_source()->CopyFrom(request.source());
  request.mutable_group_realization()->CopyFrom(build_group_realization_options());
  request.mutable_group_realization()->mutable_version()->mutable_explicit_version_set()->set_version_set_id(
      "vs-requested");
  request.mutable_group_realization()->set_require_staged_publish(true);

  auto plan_or = build_controller_realization_plan(request);

  REQUIRE(plan_or.ok());
  CHECK(plan_or->target.target_kind == "target_set");
  CHECK(plan_or->target.member_count == 2);
  REQUIRE(plan_or->target.group_id.has_value());
  CHECK(*plan_or->target.group_id == "group-1");
  CHECK_FALSE(plan_or->target.target_layout_digest.empty());
  CHECK(plan_or->strategy.source_selection_mode == "same_selection");
  CHECK(plan_or->strategy.source_coordination == "group_realization_transport");
  CHECK(plan_or->strategy.collective_policy == v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST);
  REQUIRE(plan_or->strategy.version_set_id.has_value());
  CHECK(*plan_or->strategy.version_set_id == "vs-requested");
  REQUIRE(plan_or->strategy.source_selection_digest.has_value());
  CHECK(*plan_or->strategy.source_selection_digest == "source-selection");
  CHECK(
      (plan_or->strategy.group_barriers ==
       std::vector<std::string>{"member_readiness", "group_acquire", "staged_values", "publish_barrier"}));
  CHECK(plan_or->lifecycle.capability == "target_set");
  CHECK(plan_or->lifecycle.staged_value_count == 2);
  CHECK(plan_or->lifecycle.acquire_claim_count == 2);
  CHECK(plan_or->lifecycle.publish_barrier);
  CHECK(plan_or->resource_envelope.export_kind == "binding_reservation_set");
  REQUIRE(require_controller_export_kind(*plan_or, "binding_reservation_set", "PrefetchServingBinding").ok());
  REQUIRE(require_controller_resource_authority(*plan_or, "BindingRegistry", "PrefetchServingBinding").ok());
  REQUIRE(require_controller_resource_authority(*plan_or, "SessionLifecycleManager", "PrefetchServingBinding").ok());
}

TEST_CASE("Controller realization plan mirrors retained serving member prefetch", "[daemon][materialization][policy]") {
  v2::PrefetchServingBindingRequest request;
  request.mutable_source_selection()->set_artifact_id("mi2:checkpoint");
  fill_checkpoint_source(request.mutable_source());
  request.set_operation_id("prefetch-member-op");
  request.set_requested_readiness(operation::SERVING_BINDING_READINESS_LOCAL_READY);
  fill_serving_target(request.mutable_serving_binding_target(), "member-0", 0, "GPU-0");
  request.mutable_group_realization()->CopyFrom(build_group_realization_options());
  request.mutable_group_realization()->set_require_staged_publish(true);

  auto plan_or = build_controller_realization_plan(request);

  REQUIRE(plan_or.ok());
  CHECK(plan_or->target.target_kind == "retained_binding");
  CHECK(plan_or->target.member_count == 2);
  REQUIRE(plan_or->target.group_id.has_value());
  CHECK(*plan_or->target.group_id == "group-1");
  REQUIRE(plan_or->target.part_id.has_value());
  CHECK(*plan_or->target.part_id == "member-0");
  CHECK(plan_or->lifecycle.capability == "retained_binding");
  CHECK(plan_or->lifecycle.staged_value_count == 1);
  CHECK(plan_or->lifecycle.acquire_claim_count == 1);
  CHECK(plan_or->lifecycle.publish_barrier);
  CHECK(plan_or->resource_envelope.export_kind == "binding_reservation");
  REQUIRE(require_controller_export_kind(*plan_or, "binding_reservation", "PrefetchServingBinding").ok());
  REQUIRE(require_controller_resource_authority(*plan_or, "BindingRegistry", "PrefetchServingBinding").ok());
  REQUIRE(require_controller_resource_authority(*plan_or, "SessionLifecycleManager", "PrefetchServingBinding").ok());
}

TEST_CASE(
    "Controller realization plan mirrors retained binding acquire lifecycle",
    "[daemon][materialization][policy]") {
  v2::AcquireBindingValueRequest request;
  request.set_caller_pid(1234);
  request.set_expected_daemon_id("daemon-a");
  request.set_expected_daemon_session_id("session-a");
  request.set_expected_device_uuid("GPU-0");
  request.set_expected_target_layout_hash("target-layout-0");
  request.set_expected_tensor_schema_hash("tensor-schema");
  request.set_expected_serving_build_digest("serving-build");
  request.mutable_binding_value_ref()->set_binding_id("binding-1");
  request.mutable_binding_value_ref()->set_binding_layout_id("layout-0");
  request.mutable_binding_value_ref()->set_binding_value_id("value-1");
  request.mutable_binding_value_ref()->set_seal_generation(1);
  request.mutable_reservation_capability()->set_capability_id("capability-1");
  request.mutable_reservation_capability()->mutable_binding_value_ref()->CopyFrom(request.binding_value_ref());
  request.mutable_reservation_capability()->set_daemon_id("daemon-a");
  request.mutable_reservation_capability()->set_daemon_session_id("session-a");
  request.mutable_reservation_capability()->set_device_uuid("GPU-0");
  fill_member(request.mutable_reservation_capability()->mutable_member(), "member-0", 0);
  request.mutable_reservation_capability()->set_reservation_bytes(1024);
  request.mutable_reservation_capability()->set_scope_digest("scope-digest");
  request.mutable_group_realization_acquire()->set_transaction_id("txn-17");
  request.mutable_group_realization_acquire()->set_version_set_id("vs-17");
  request.mutable_group_realization_acquire()->set_part_id("member-0");
  request.mutable_group_realization_acquire()->set_staging_token("stage-0");
  request.mutable_group_realization_acquire()->set_wait_for_publish(true);

  auto plan_or = build_controller_realization_plan(request);

  REQUIRE(plan_or.ok());
  CHECK(plan_or->target.target_kind == "runtime_attachment");
  CHECK(plan_or->target.resolved_artifact_id == "binding:binding-1");
  CHECK(plan_or->target.target_layout_digest == "target-layout-0");
  CHECK(plan_or->target.member_count == 2);
  REQUIRE(plan_or->target.group_id.has_value());
  CHECK(*plan_or->target.group_id == "group-1");
  REQUIRE(plan_or->target.part_id.has_value());
  CHECK(*plan_or->target.part_id == "member-0");
  CHECK(plan_or->strategy.source_selection_mode == "per_part_selection");
  CHECK(plan_or->strategy.source_coordination == "group_realization_acquire");
  CHECK((plan_or->strategy.group_barriers == std::vector<std::string>{"group_acquire", "publish_barrier"}));
  REQUIRE(plan_or->strategy.transaction_id.has_value());
  CHECK(*plan_or->strategy.transaction_id == "txn-17");
  REQUIRE(plan_or->strategy.version_set_id.has_value());
  CHECK(*plan_or->strategy.version_set_id == "vs-17");
  CHECK(plan_or->lifecycle.capability == "retained_acquire");
  CHECK(plan_or->lifecycle.staged_value_count == 1);
  CHECK(plan_or->lifecycle.acquire_claim_count == 1);
  CHECK(plan_or->lifecycle.publish_barrier);
  CHECK(plan_or->resource_envelope.export_kind == "cuda_ipc_lease");
  CHECK(plan_or->resource_envelope.projection_kind == "runtime_attachment");
}

TEST_CASE("Controller realization plan mirrors target publication lifecycle", "[daemon][materialization][policy]") {
  v2::PublishTargetReplicaRequest request;
  request.set_binding_current_value_publication_token("publication-token");
  request.mutable_byte_space()->set_kind(common::BYTE_SPACE_KIND_CANONICAL);
  request.set_owner_pid(1234);
  request.set_operation_id("publish-op-request");

  common::BindingCurrentValuePublicationScope scope;
  scope.set_publication_id("publication-1");
  scope.mutable_selection()->set_artifact_id("mi2:published");
  scope.mutable_selection()->set_selection_hash("selection-hash");
  scope.mutable_byte_space()->CopyFrom(request.byte_space());
  scope.set_device_uuid("GPU-0");
  scope.set_owner_pid(4321);
  scope.set_target_layout_hash("layout-hash");
  scope.set_operation_id("publish-op-scope");
  scope.set_binding_id("binding-1");
  scope.set_binding_layout_id("layout-0");
  scope.set_binding_value_id("value-1");
  scope.set_seal_generation(7);
  scope.set_daemon_id("daemon-a");
  scope.set_daemon_session_id("session-a");

  common::ByteSpaceRef normalized_byte_space;
  normalized_byte_space.set_kind(common::BYTE_SPACE_KIND_CANONICAL);

  auto plan_or = build_controller_realization_plan(request, scope, normalized_byte_space);

  REQUIRE(plan_or.ok());
  CHECK(plan_or->target.target_kind == "publication");
  CHECK(plan_or->target.resolved_artifact_id == "mi2:published");
  CHECK(plan_or->target.target_layout_digest == "6c61796f75742d68617368");
  CHECK(plan_or->target.device_uuid == "GPU-0");
  CHECK(plan_or->target.owner_pid == 1234);
  CHECK(plan_or->target.member_count == 1);
  REQUIRE(plan_or->target.operation_id.has_value());
  CHECK(*plan_or->target.operation_id == "publish-op-request");
  CHECK(plan_or->strategy.source_selection_mode == "single_selection");
  CHECK(plan_or->strategy.source_coordination == "publication_lifecycle");
  CHECK(plan_or->strategy.collective_policy == v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE);
  check_controller_source_selection_digest(plan_or->strategy.source_selection_digest);
  CHECK(plan_or->lifecycle.capability == "publication");
  CHECK(plan_or->lifecycle.export_lifetime_kind == "publication_lease");
  CHECK(plan_or->lifecycle.mutability_contract == "published_read_only");
  CHECK(
      (plan_or->lifecycle.release_policy ==
       std::vector<std::string>{
           "retire_published_replica",
           "release_publication_lease",
           "release_lifecycle_use_guard",
       }));
  CHECK(plan_or->lifecycle.acquire_claim_count == 1);
  CHECK(plan_or->lifecycle.publish_barrier);
  CHECK(plan_or->resource_envelope.backing_kind == "daemon_published_replica");
  CHECK(plan_or->resource_envelope.export_kind == "publication_lease");
  CHECK(plan_or->resource_envelope.projection_kind == "published_replica");
  CHECK(plan_or->resource_envelope.owner_kind == "runtime_publication");
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

TEST_CASE(
    "Group realization begin context derives stable child transport request id from frozen identity",
    "[daemon][materialization][policy]") {
  auto options = build_group_realization_options();
  auto client = std::make_shared<RecordingGroupRealizationClient>();
  auto begin_or = begin_or_join_group_realization_if_enabled(client, &options, "daemon-a", "session-a", "worker-a");
  REQUIRE(begin_or.ok());
  REQUIRE(begin_or->has_value());

  auto context_or = resolve_group_realization_transport_context("op-17", &options);
  REQUIRE(context_or.ok());
  apply_group_realization_begin_context_to_transport_context(**begin_or, &*context_or);
  const std::string child_transport_request_id = context_or->transport_request_id;

  CHECK(child_transport_request_id.rfind("grt:", 0) == 0);
  CHECK(child_transport_request_id != "op-17");

  auto replay_or = resolve_group_realization_transport_context("op-17", &options);
  REQUIRE(replay_or.ok());
  apply_group_realization_begin_context_to_transport_context(**begin_or, &*replay_or);
  CHECK(replay_or->transport_request_id == child_transport_request_id);

  auto different_part_context = **begin_or;
  different_part_context.part_id = "rank1";
  different_part_context.part_selection.set_artifact_id("artifact-frozen-rank1");
  different_part_context.part_selection.set_view_id("view-rank1");
  different_part_context.requested_byte_space.set_id("view-rank1");
  different_part_context.selection_hash = "hash-rank1";
  auto different_part_or = resolve_group_realization_transport_context("op-17", &options);
  REQUIRE(different_part_or.ok());
  apply_group_realization_begin_context_to_transport_context(different_part_context, &*different_part_or);
  CHECK(different_part_or->transport_request_id != child_transport_request_id);

  auto different_attempt_or = resolve_group_realization_transport_context("op-18", &options);
  REQUIRE(different_attempt_or.ok());
  apply_group_realization_begin_context_to_transport_context(**begin_or, &*different_attempt_or);
  CHECK(different_attempt_or->transport_request_id != child_transport_request_id);
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
