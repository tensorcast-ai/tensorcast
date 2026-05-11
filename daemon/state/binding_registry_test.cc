// Copyright (c) 2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>

#include "absl/time/time.h"
#include "daemon/state/binding_registry.h"

namespace tensorcast::daemon {
namespace {

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
