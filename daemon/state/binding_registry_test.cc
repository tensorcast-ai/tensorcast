// Copyright (c) 2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/store/testing/global_store_client_stub.h"
#include "daemon/state/binding_registry.h"
#include "daemon/state/sweep_tasks.h"

namespace tensorcast::daemon {
namespace {

namespace global_store = tensorcast::global_store::v1;

class GroupRealizationStatusClient final : public store::testing::GlobalStoreClientStub {
 public:
  std::unordered_map<std::string, global_store::GroupRealizationState> states;
  std::vector<std::string> requested_transaction_ids;

  absl::StatusOr<global_store::GetGroupRealizationResponse> get_group_realization(
      const global_store::GetGroupRealizationRequest& request,
      const store::components::RpcOptions&) override {
    requested_transaction_ids.push_back(request.transaction_id());
    global_store::GetGroupRealizationResponse response;
    response.set_transaction_id(request.transaction_id());
    const auto it = states.find(request.transaction_id());
    if (it == states.end()) {
      response.set_status(global_store::STATUS_NOT_FOUND);
      return response;
    }
    response.set_status(global_store::STATUS_OK);
    response.set_state(it->second);
    return response;
  }
};

std::shared_ptr<BindingRegistry::Record> make_record(std::string binding_id) {
  auto record = std::make_shared<BindingRegistry::Record>();
  record->binding_id = std::move(binding_id);
  record->binding_layout_id = "layout-1";
  record->owner_pid = 1234;
  record->creator_pid = 1234;
  record->device_uuid = "GPU-0";
  record->state = v2::BINDING_STATE_READY_LOCAL;
  record->control_lifetime = BindingRegistry::ControlLifetime::kDaemonRetained;
  record->retained_ref = true;
  record->current_binding_value_id = "value-1";
  record->seal_generation = 7;
  record->target_layout_hash = "layout-hash";
  record->tensor_schema_hash = "tensor-schema";
  record->daemon_id = "daemon-1";
  record->daemon_session_id = "session-1";
  record->serving_build_digest = "serving-build";
  record->reservation_capability_id = "capability-1";
  record->reservation_expires_at = absl::UnixEpoch() + absl::Hours(1);
  record->local_serving_ref = "binding-local:binding-1:value-1";
  record->serving_member.set_member_id("member-0");
  record->serving_member.set_member_index(0);
  record->serving_member.set_member_count(1);
  record->serving_member.set_group_id("group-1");
  return record;
}

v2::AcquireBindingValueRequest make_acquire_request() {
  v2::AcquireBindingValueRequest request;
  auto* ref = request.mutable_binding_value_ref();
  ref->set_binding_id("binding-acquire");
  ref->set_binding_layout_id("layout-1");
  ref->set_binding_value_id("value-1");
  ref->set_seal_generation(7);
  auto* capability = request.mutable_reservation_capability();
  capability->set_capability_id("capability-1");
  capability->mutable_binding_value_ref()->CopyFrom(*ref);
  capability->set_daemon_id("daemon-1");
  capability->set_daemon_session_id("session-1");
  capability->set_device_uuid("GPU-0");
  capability->mutable_member()->CopyFrom(request.expected_member());
  capability->set_reservation_bytes(1024);
  capability->set_scope_digest("scope");
  capability->set_expires_at_ms(3'600'000);
  request.set_expected_daemon_id("daemon-1");
  request.set_expected_daemon_session_id("session-1");
  request.set_expected_device_uuid("GPU-0");
  auto* member = request.mutable_expected_member();
  member->set_member_id("member-0");
  member->set_member_index(0);
  member->set_member_count(1);
  member->set_group_id("group-1");
  capability->mutable_member()->CopyFrom(*member);
  request.set_expected_target_layout_hash("layout-hash");
  request.set_expected_tensor_schema_hash("tensor-schema");
  request.set_expected_serving_build_digest("serving-build");
  request.set_local_serving_ref("binding-local:binding-1:value-1");
  request.set_caller_pid(4321);
  return request;
}

BindingRegistry::StagedBindingValue make_staged_value(std::string binding_value_id) {
  BindingRegistry::StagedBindingValue value;
  value.transaction_id = "txn-1";
  value.version_set_id = "version-set-1";
  value.part_id = "part-0";
  value.binding_value_id = std::move(binding_value_id);
  value.staging_token = "staging-token-1";
  value.staging_epoch = 11;
  value.materialization_attempt_id = "attempt-1";
  value.selection.set_artifact_id("artifact-staged");
  value.expected_previous_seal_generation = 7;
  value.target_index_json = R"({"tensors":[]})";
  value.target_layout_hash = "staged-target-layout-hash";
  value.tensor_schema_hash = "staged-tensor-schema-hash";
  value.logical_total_size = 4096;
  value.source_replica_id = "replica-1";
  value.source_export_generation = 5;
  value.child_transport_request_id = "transport-1";
  value.expires_at = absl::UnixEpoch() + absl::Seconds(30);
  return value;
}

} // namespace

TEST_CASE("pid exit does not reclaim daemon-retained binding", "[daemon][binding][retention]") {
  BindingRegistry registry;
  auto record = make_record("binding-retained");
  record->control_lifetime = BindingRegistry::ControlLifetime::kDaemonRetained;
  record->retained_ref = true;

  REQUIRE(registry.insert(record).ok());
  registry.handle_pid_exit(1234);

  auto retained_or = registry.get("binding-retained");
  REQUIRE(retained_or.ok());
  {
    absl::MutexLock lock(&(*retained_or)->mu);
    REQUIRE((*retained_or)->retained_ref);
    REQUIRE_FALSE((*retained_or)->closed);
    REQUIRE((*retained_or)->owner_pid == 0);
  }
}

TEST_CASE("pid exit reclaims pid-bound binding without export refs", "[daemon][binding][retention]") {
  BindingRegistry registry;
  auto record = make_record("binding-pid-bound");
  record->control_lifetime = BindingRegistry::ControlLifetime::kPidBound;

  REQUIRE(registry.insert(record).ok());
  registry.handle_pid_exit(1234);

  REQUIRE_FALSE(registry.get("binding-pid-bound").ok());
}

TEST_CASE("unacquired ttl retires retained binding", "[daemon][binding][retention]") {
  BindingRegistry registry;
  auto record = make_record("binding-unacquired");
  record->control_lifetime = BindingRegistry::ControlLifetime::kDaemonRetained;
  record->retained_ref = true;
  record->unacquired_deadline = absl::UnixEpoch() + absl::Seconds(10);

  REQUIRE(registry.insert(record).ok());

  REQUIRE(registry.sweep_retention(absl::UnixEpoch() + absl::Seconds(11)) == 1);
  REQUIRE_FALSE(registry.get("binding-unacquired").ok());
}

TEST_CASE("active attachment prevents free but not retire", "[daemon][binding][retention]") {
  BindingRegistry registry;
  auto record = make_record("binding-active");
  record->control_lifetime = BindingRegistry::ControlLifetime::kDaemonRetained;
  record->retained_ref = true;
  record->unacquired_deadline = absl::UnixEpoch() + absl::Seconds(10);

  REQUIRE(registry.insert(record).ok());
  REQUIRE(registry.acquire_attachment_ref("binding-active", absl::UnixEpoch() + absl::Seconds(1)).ok());

  REQUIRE(registry.sweep_retention(absl::UnixEpoch() + absl::Seconds(11)) == 0);
  REQUIRE(registry.get("binding-active").ok());

  REQUIRE(registry.retire_retained("binding-active", "manual").ok());
  REQUIRE(registry.get("binding-active").ok());

  registry.release_attachment_ref("binding-active", absl::UnixEpoch() + absl::Seconds(12));
  REQUIRE_FALSE(registry.get("binding-active").ok());
}

TEST_CASE(
    "binding attachment release arms idle ttl and keepalive requires active attachment",
    "[daemon][binding][retention]") {
  BindingRegistry registry;
  auto record = make_record("binding-idle");

  REQUIRE(registry.insert(record).ok());
  REQUIRE(registry.acquire_attachment_ref("binding-idle", absl::UnixEpoch() + absl::Seconds(1)).ok());
  REQUIRE(registry.keepalive_attachment_ref("binding-idle", absl::UnixEpoch() + absl::Seconds(2)).ok());

  registry.release_attachment_ref("binding-idle", absl::UnixEpoch() + absl::Seconds(3), absl::Seconds(5));
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->active_attachment_refs == 0);
    REQUIRE(record->idle_deadline == absl::UnixEpoch() + absl::Seconds(8));
  }
  REQUIRE_FALSE(registry.keepalive_attachment_ref("binding-idle", absl::UnixEpoch() + absl::Seconds(4)).ok());
  REQUIRE(registry.sweep_retention(absl::UnixEpoch() + absl::Seconds(7)) == 0);
  REQUIRE(registry.sweep_retention(absl::UnixEpoch() + absl::Seconds(8)) == 1);
  REQUIRE_FALSE(registry.get("binding-idle").ok());
}

TEST_CASE(
    "binding attachment release uses record idle ttl by default",
    "[daemon][binding][retention]") {
  BindingRegistry registry;
  auto record = make_record("binding-idle-default");
  record->idle_ttl_after_last_release = absl::Seconds(5);

  REQUIRE(registry.insert(record).ok());
  REQUIRE(registry.acquire_attachment_ref("binding-idle-default", absl::UnixEpoch() + absl::Seconds(1)).ok());

  registry.release_attachment_ref("binding-idle-default", absl::UnixEpoch() + absl::Seconds(3));
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->active_attachment_refs == 0);
    REQUIRE(record->idle_deadline == absl::UnixEpoch() + absl::Seconds(8));
  }
  REQUIRE(registry.sweep_retention(absl::UnixEpoch() + absl::Seconds(7)) == 0);
  REQUIRE(registry.sweep_retention(absl::UnixEpoch() + absl::Seconds(8)) == 1);
  REQUIRE_FALSE(registry.get("binding-idle-default").ok());
}

TEST_CASE(
    "binding attachment bootstrap release does not arm idle ttl before first acquire",
    "[daemon][binding][retention]") {
  BindingRegistry registry;
  auto record = make_record("binding-bootstrap");
  record->idle_ttl_after_last_release = absl::Seconds(1);
  record->active_attachment_refs = 1;

  REQUIRE(registry.insert(record).ok());

  registry.release_attachment_ref("binding-bootstrap", absl::UnixEpoch() + absl::Seconds(3));
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->active_attachment_refs == 0);
    REQUIRE(record->first_acquired_at == absl::InfinitePast());
    REQUIRE(record->idle_deadline == absl::InfiniteFuture());
  }
  REQUIRE(registry.sweep_retention(absl::UnixEpoch() + absl::Seconds(10)) == 0);
  REQUIRE(registry.get("binding-bootstrap").ok());
}

TEST_CASE("materialization timeout retires not-ready retained binding", "[daemon][binding][retention]") {
  BindingRegistry registry;
  auto record = make_record("binding-materializing");
  record->control_lifetime = BindingRegistry::ControlLifetime::kDaemonRetained;
  record->retained_ref = true;
  record->state = v2::BINDING_STATE_MUTABLE;
  record->materialization_deadline = absl::UnixEpoch() + absl::Seconds(5);

  REQUIRE(registry.insert(record).ok());

  REQUIRE(registry.sweep_retention(absl::UnixEpoch() + absl::Seconds(6)) == 1);
  REQUIRE_FALSE(registry.get("binding-materializing").ok());
}

TEST_CASE("acquire validation admits matching retained binding value", "[daemon][binding][acquire]") {
  BindingRegistry registry;
  auto record = make_record("binding-acquire");
  record->allowed_caller_pid = 4321;

  REQUIRE(registry.insert(record).ok());

  auto request = make_acquire_request();
  REQUIRE(registry.validate_and_acquire_attachment_ref(request, absl::UnixEpoch() + absl::Seconds(1)).ok());
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->active_attachment_refs == 1);
  }
}

TEST_CASE("staged binding value stays separate from current value", "[daemon][binding][staged]") {
  BindingRegistry registry;
  auto record = make_record("binding-acquire");
  REQUIRE(registry.insert(record).ok());

  REQUIRE(registry
              .insert_staged_value(
                  "binding-acquire", make_staged_value("staged-value-1"), absl::UnixEpoch() + absl::Seconds(1))
              .ok());

  auto staged_or = registry.get_staged_value("binding-acquire", "staged-value-1");
  REQUIRE(staged_or.ok());
  CHECK(staged_or->transaction_id == "txn-1");
  CHECK(staged_or->daemon_id == "daemon-1");
  CHECK(staged_or->daemon_session_id == "session-1");
  CHECK(staged_or->created_at == absl::UnixEpoch() + absl::Seconds(1));
  CHECK(staged_or->target_layout_hash == "staged-target-layout-hash");
  CHECK(staged_or->logical_total_size == 4096);
  {
    absl::MutexLock lock(&record->mu);
    CHECK(record->current_binding_value_id == "value-1");
    CHECK(record->staged_values_by_id.size() == 1);
  }
}

TEST_CASE("ordinary acquire rejects staged binding value", "[daemon][binding][staged]") {
  BindingRegistry registry;
  auto record = make_record("binding-acquire");
  REQUIRE(registry.insert(record).ok());
  REQUIRE(registry
              .insert_staged_value(
                  "binding-acquire", make_staged_value("staged-value-1"), absl::UnixEpoch() + absl::Seconds(1))
              .ok());

  auto request = make_acquire_request();
  request.mutable_binding_value_ref()->set_binding_value_id("staged-value-1");
  request.mutable_reservation_capability()->mutable_binding_value_ref()->CopyFrom(request.binding_value_ref());

  const auto status = registry.validate_acquire_request(request, absl::UnixEpoch() + absl::Seconds(2));
  REQUIRE(status.code() == absl::StatusCode::kFailedPrecondition);
  CHECK(status.message() == "binding_value_id mismatch");
  {
    absl::MutexLock lock(&record->mu);
    CHECK(record->active_attachment_refs == 0);
    CHECK(record->current_binding_value_id == "value-1");
  }
}

TEST_CASE("group-aware acquire admits staged value only with matching published fences", "[daemon][binding][staged]") {
  BindingRegistry registry;
  auto record = make_record("binding-acquire");
  record->allowed_caller_pid = 4321;
  REQUIRE(registry.insert(record).ok());
  REQUIRE(registry
              .insert_staged_value(
                  "binding-acquire", make_staged_value("staged-value-1"), absl::UnixEpoch() + absl::Seconds(1))
              .ok());

  auto request = make_acquire_request();
  request.mutable_binding_value_ref()->set_binding_value_id("staged-value-1");
  request.mutable_reservation_capability()->mutable_binding_value_ref()->CopyFrom(request.binding_value_ref());
  request.set_expected_target_layout_hash("staged-target-layout-hash");
  request.set_expected_tensor_schema_hash("staged-tensor-schema-hash");
  auto* group = request.mutable_group_realization_acquire();
  group->set_transaction_id("txn-1");
  group->set_version_set_id("version-set-1");
  group->set_part_id("part-0");
  group->set_staging_token("staging-token-1");

  REQUIRE(
      registry.validate_and_acquire_group_staged_attachment_ref(request, absl::UnixEpoch() + absl::Seconds(2)).ok());
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->active_attachment_refs == 1);
    REQUIRE(record->current_binding_value_id == "value-1");
  }
}

TEST_CASE("owner-created staged binding acquire does not require serving reservation", "[daemon][binding][staged]") {
  BindingRegistry registry;
  auto record = make_record("binding-acquire");
  record->control_lifetime = BindingRegistry::ControlLifetime::kPidBound;
  record->retained_ref = false;
  record->reservation_capability_id.clear();
  record->serving_member.Clear();
  record->serving_build_digest.clear();
  record->allowed_caller_pid.reset();
  REQUIRE(registry.insert(record).ok());
  auto staged = make_staged_value("staged-value-1");
  staged.daemon_id = "daemon-1";
  staged.daemon_session_id = "session-1";
  REQUIRE(registry.insert_staged_value("binding-acquire", std::move(staged), absl::UnixEpoch()).ok());

  v2::AcquireBindingValueRequest request;
  auto* ref = request.mutable_binding_value_ref();
  ref->set_binding_id("binding-acquire");
  ref->set_binding_layout_id("layout-1");
  ref->set_binding_value_id("staged-value-1");
  ref->set_seal_generation(7);
  request.set_expected_daemon_id("daemon-1");
  request.set_expected_daemon_session_id("session-1");
  request.set_expected_device_uuid("GPU-0");
  request.set_expected_target_layout_hash("staged-target-layout-hash");
  request.set_caller_pid(1234);
  auto* group = request.mutable_group_realization_acquire();
  group->set_transaction_id("txn-1");
  group->set_version_set_id("version-set-1");
  group->set_part_id("part-0");
  group->set_staging_token("staging-token-1");

  REQUIRE(registry.validate_and_acquire_group_staged_attachment_ref(request, absl::UnixEpoch()).ok());
  {
    absl::MutexLock lock(&record->mu);
    CHECK(record->active_attachment_refs == 1);
    CHECK(record->retained_ref == false);
  }
}

TEST_CASE("group-aware acquire rejects unpublished or mismatched staged fences", "[daemon][binding][staged]") {
  BindingRegistry registry;
  auto record = make_record("binding-acquire");
  REQUIRE(registry.insert(record).ok());
  REQUIRE(registry
              .insert_staged_value(
                  "binding-acquire", make_staged_value("staged-value-1"), absl::UnixEpoch() + absl::Seconds(1))
              .ok());

  auto request = make_acquire_request();
  request.mutable_binding_value_ref()->set_binding_value_id("staged-value-1");
  request.mutable_reservation_capability()->mutable_binding_value_ref()->CopyFrom(request.binding_value_ref());
  request.set_expected_target_layout_hash("staged-target-layout-hash");
  request.set_expected_tensor_schema_hash("staged-tensor-schema-hash");
  auto* group = request.mutable_group_realization_acquire();
  group->set_transaction_id("txn-1");
  group->set_version_set_id("version-set-1");
  group->set_part_id("part-0");

  auto missing_token_status =
      registry.validate_group_staged_acquire_request(request, absl::UnixEpoch() + absl::Seconds(2));
  REQUIRE(missing_token_status.code() == absl::StatusCode::kInvalidArgument);

  group->set_staging_token("other-token");
  auto mismatched_token_status =
      registry.validate_group_staged_acquire_request(request, absl::UnixEpoch() + absl::Seconds(2));
  REQUIRE(mismatched_token_status.code() == absl::StatusCode::kFailedPrecondition);
}

TEST_CASE("staged binding value cleanup covers ttl transaction and shutdown paths", "[daemon][binding][staged]") {
  BindingRegistry registry;
  auto record = make_record("binding-acquire");
  REQUIRE(registry.insert(record).ok());

  auto expired = make_staged_value("staged-expired");
  expired.expires_at = absl::UnixEpoch() + absl::Seconds(5);
  REQUIRE(registry.insert_staged_value("binding-acquire", std::move(expired), absl::UnixEpoch()).ok());
  REQUIRE(registry
              .insert_staged_value(
                  "binding-acquire", make_staged_value("staged-abort"), absl::UnixEpoch() + absl::Seconds(1))
              .ok());
  auto shutdown_value = make_staged_value("staged-shutdown");
  shutdown_value.transaction_id = "txn-2";
  REQUIRE(registry.insert_staged_value("binding-acquire", std::move(shutdown_value), absl::UnixEpoch()).ok());

  CHECK(registry.sweep_staged_values(absl::UnixEpoch() + absl::Seconds(6), 1) == 1);
  CHECK_FALSE(registry.get_staged_value("binding-acquire", "staged-expired").ok());

  CHECK(registry.remove_staged_values_for_transaction("txn-1", "abort", 0) == 1);
  CHECK_FALSE(registry.get_staged_value("binding-acquire", "staged-abort").ok());

  CHECK(registry.clear_staged_values("daemon_shutdown") == 1);
  CHECK_FALSE(registry.get_staged_value("binding-acquire", "staged-shutdown").ok());
}

TEST_CASE(
    "binding retention sweep removes staged values for terminal group transactions",
    "[daemon][binding][staged]") {
  BindingRegistry registry;
  auto record = make_record("binding-acquire");
  REQUIRE(registry.insert(record).ok());

  auto aborted = make_staged_value("staged-aborted");
  aborted.transaction_id = "txn-aborted";
  aborted.expires_at = absl::InfiniteFuture();
  REQUIRE(registry.insert_staged_value("binding-acquire", std::move(aborted), absl::UnixEpoch()).ok());

  auto expired = make_staged_value("staged-expired-terminal");
  expired.transaction_id = "txn-expired";
  expired.expires_at = absl::InfiniteFuture();
  REQUIRE(registry.insert_staged_value("binding-acquire", std::move(expired), absl::UnixEpoch()).ok());

  auto published = make_staged_value("staged-published");
  published.transaction_id = "txn-published";
  published.expires_at = absl::InfiniteFuture();
  REQUIRE(registry.insert_staged_value("binding-acquire", std::move(published), absl::UnixEpoch()).ok());

  auto missing = make_staged_value("staged-missing");
  missing.transaction_id = "txn-missing";
  missing.expires_at = absl::InfiniteFuture();
  REQUIRE(registry.insert_staged_value("binding-acquire", std::move(missing), absl::UnixEpoch()).ok());

  auto client = std::make_shared<GroupRealizationStatusClient>();
  client->states.emplace("txn-aborted", global_store::GROUP_REALIZATION_STATE_ABORTED);
  client->states.emplace("txn-expired", global_store::GROUP_REALIZATION_STATE_EXPIRED);
  client->states.emplace("txn-published", global_store::GROUP_REALIZATION_STATE_PUBLISHED);

  BindingRetentionSweepTask task(registry, client);
  task.run_once();

  CHECK_FALSE(registry.get_staged_value("binding-acquire", "staged-aborted").ok());
  CHECK_FALSE(registry.get_staged_value("binding-acquire", "staged-expired-terminal").ok());
  CHECK_FALSE(registry.get_staged_value("binding-acquire", "staged-missing").ok());
  CHECK(registry.get_staged_value("binding-acquire", "staged-published").ok());
  CHECK(client->requested_transaction_ids.size() == 4);
}

TEST_CASE("staged binding insertion rejects current value and stale previous generation", "[daemon][binding][staged]") {
  BindingRegistry registry;
  auto record = make_record("binding-acquire");
  REQUIRE(registry.insert(record).ok());

  auto current_value = make_staged_value("value-1");
  auto current_status =
      registry.insert_staged_value("binding-acquire", std::move(current_value), absl::UnixEpoch() + absl::Seconds(1));
  REQUIRE(current_status.code() == absl::StatusCode::kFailedPrecondition);

  auto stale_generation = make_staged_value("staged-stale-generation");
  stale_generation.expected_previous_seal_generation = 8;
  auto generation_status = registry.insert_staged_value(
      "binding-acquire", std::move(stale_generation), absl::UnixEpoch() + absl::Seconds(1));
  REQUIRE(generation_status.code() == absl::StatusCode::kFailedPrecondition);
}

TEST_CASE("acquire validation rejects stale binding identity and layout", "[daemon][binding][acquire]") {
  BindingRegistry registry;
  auto record = make_record("binding-acquire");
  REQUIRE(registry.insert(record).ok());

  auto stale_value = make_acquire_request();
  stale_value.mutable_binding_value_ref()->set_binding_value_id("stale-value");
  stale_value.mutable_reservation_capability()->mutable_binding_value_ref()->CopyFrom(stale_value.binding_value_ref());
  REQUIRE_FALSE(registry.validate_acquire_request(stale_value, absl::UnixEpoch() + absl::Seconds(1)).ok());

  auto stale_generation = make_acquire_request();
  stale_generation.mutable_binding_value_ref()->set_seal_generation(6);
  stale_generation.mutable_reservation_capability()->mutable_binding_value_ref()->CopyFrom(
      stale_generation.binding_value_ref());
  REQUIRE_FALSE(registry.validate_acquire_request(stale_generation, absl::UnixEpoch() + absl::Seconds(1)).ok());

  auto layout_mismatch = make_acquire_request();
  layout_mismatch.set_expected_target_layout_hash("other-layout");
  REQUIRE_FALSE(registry.validate_acquire_request(layout_mismatch, absl::UnixEpoch() + absl::Seconds(1)).ok());

  auto schema_mismatch = make_acquire_request();
  schema_mismatch.set_expected_tensor_schema_hash("other-schema");
  REQUIRE_FALSE(registry.validate_acquire_request(schema_mismatch, absl::UnixEpoch() + absl::Seconds(1)).ok());

  auto build_mismatch = make_acquire_request();
  build_mismatch.set_expected_serving_build_digest("other-build");
  REQUIRE_FALSE(registry.validate_acquire_request(build_mismatch, absl::UnixEpoch() + absl::Seconds(1)).ok());
}

TEST_CASE("acquire validation rejects authority and caller mismatches", "[daemon][binding][acquire]") {
  BindingRegistry registry;
  auto record = make_record("binding-acquire");
  record->allowed_caller_pid = 4321;
  REQUIRE(registry.insert(record).ok());

  auto daemon_mismatch = make_acquire_request();
  daemon_mismatch.set_expected_daemon_session_id("other-session");
  REQUIRE_FALSE(registry.validate_acquire_request(daemon_mismatch, absl::UnixEpoch() + absl::Seconds(1)).ok());

  auto device_mismatch = make_acquire_request();
  device_mismatch.set_expected_device_uuid("GPU-1");
  REQUIRE_FALSE(registry.validate_acquire_request(device_mismatch, absl::UnixEpoch() + absl::Seconds(1)).ok());

  auto member_mismatch = make_acquire_request();
  member_mismatch.mutable_expected_member()->set_member_index(1);
  member_mismatch.mutable_reservation_capability()->mutable_member()->CopyFrom(member_mismatch.expected_member());
  REQUIRE_FALSE(registry.validate_acquire_request(member_mismatch, absl::UnixEpoch() + absl::Seconds(1)).ok());

  auto caller_mismatch = make_acquire_request();
  caller_mismatch.set_caller_pid(9999);
  REQUIRE_FALSE(registry.validate_acquire_request(caller_mismatch, absl::UnixEpoch() + absl::Seconds(1)).ok());

  auto capability_mismatch = make_acquire_request();
  capability_mismatch.mutable_reservation_capability()->set_capability_id("other-capability");
  REQUIRE_FALSE(registry.validate_acquire_request(capability_mismatch, absl::UnixEpoch() + absl::Seconds(1)).ok());

  auto expired = make_acquire_request();
  expired.mutable_reservation_capability()->set_expires_at_ms(1);
  REQUIRE_FALSE(registry.validate_acquire_request(expired, absl::UnixEpoch() + absl::Seconds(2)).ok());
}

TEST_CASE("acquire validation rejects retired and local-ref-only requests", "[daemon][binding][acquire]") {
  BindingRegistry registry;
  auto record = make_record("binding-acquire");
  REQUIRE(registry.insert(record).ok());

  auto local_ref_only = make_acquire_request();
  local_ref_only.clear_binding_value_ref();
  REQUIRE_FALSE(registry.validate_acquire_request(local_ref_only, absl::UnixEpoch() + absl::Seconds(1)).ok());

  REQUIRE(registry.acquire_attachment_ref("binding-acquire", absl::UnixEpoch() + absl::Seconds(1)).ok());
  REQUIRE(registry.retire_retained("binding-acquire", "manual").ok());
  REQUIRE_FALSE(registry.validate_acquire_request(make_acquire_request(), absl::UnixEpoch() + absl::Seconds(2)).ok());
}

} // namespace tensorcast::daemon
