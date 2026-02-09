// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "core/common/ready_signal.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/store_engine.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace {

tensorcast::store::StoreEngineOptions make_opts_basic() {
  tensorcast::store::StoreEngineOptions opts;
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  return opts;
}

tensorcast::daemon::DaemonOptions make_daemon_options() {
  tensorcast::daemon::DaemonOptions opts;
  opts.storage_path = std::filesystem::temp_directory_path();
  return opts;
}

std::unique_ptr<tensorcast::daemon::DaemonServiceHarness> make_harness(
    const std::shared_ptr<tensorcast::store::StoreEngine>& engine,
    const tensorcast::daemon::DaemonOptions& options) {
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, options);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

tensorcast::store::loading::ReplicaKey make_replica_key(std::string artifact_id) {
  tensorcast::store::loading::ReplicaKey key;
  key.artifact_id = std::move(artifact_id);
  key.device = tensorcast::store::DeviceKey{.type = tensorcast::DeviceType::CPU, .ordinal = -1, .uuid = ""};
  key.replica = 0;
  return key;
}

} // namespace

TEST_CASE("ConfirmReplica returns code=0 for unknown replica_uuid", "[daemon][replica-lifecycle]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  tensorcast::daemon::v2::ConfirmReplicaRequest req;
  req.set_replica_uuid("unknown-replica");
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_CPU);
  tensorcast::daemon::v2::ConfirmReplicaResponse resp;
  grpc::ServerContext ctx;
  const auto st = svc.ConfirmReplica(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.code() == 0);
}

TEST_CASE("UnloadReplica keeps session when another pid still references replica", "[daemon][replica-lifecycle]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string replica_uuid = "replica-unload-shared-ref";
  const auto key = make_replica_key("mi2:unload:shared");
  auto ready = std::make_shared<tensorcast::common::ReadySignal<absl::Status>>();
  ready->set_value(absl::OkStatus());
  REQUIRE(harness->kernel().sessions_service().put_with_verification(replica_uuid, key, ready).ok());

  harness->kernel().ref_tracker().add_ref(key, 1001);
  harness->kernel().ref_tracker().add_ref(key, 2002);

  tensorcast::daemon::v2::UnloadReplicaRequest req;
  req.set_replica_uuid(replica_uuid);
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_CPU);
  req.set_pid(1001);

  tensorcast::daemon::v2::UnloadReplicaResponse resp;
  grpc::ServerContext ctx;
  const auto st = svc.UnloadReplica(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.code() == 0);
  REQUIRE(harness->kernel().ref_tracker().ref_count(key) == 1);
  REQUIRE(harness->kernel().sessions_service().get(replica_uuid).has_value());
}

TEST_CASE("WaitReplicaVerification persists failure state for later polling", "[daemon][replica-lifecycle]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string replica_uuid = "replica-verification-fail";
  const auto key = make_replica_key("mi2:verify:fail");
  auto ready = std::make_shared<tensorcast::common::ReadySignal<absl::Status>>();
  REQUIRE(harness->kernel().sessions_service().put_with_verification(replica_uuid, key, ready).ok());

  tensorcast::daemon::v2::WaitReplicaVerificationRequest req;
  req.set_replica_uuid(replica_uuid);
  req.set_timeout_ms(500);

  std::thread failure([ready]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ready->set_value(absl::InternalError("injected verification failure"));
  });

  tensorcast::daemon::v2::WaitReplicaVerificationResponse resp;
  grpc::ServerContext ctx;
  const auto st = svc.WaitReplicaVerification(&ctx, &req, &resp);
  failure.join();
  REQUIRE(st.error_code() == grpc::StatusCode::INTERNAL);
  REQUIRE(resp.status() == tensorcast::daemon::v2::VerificationStatus::VERIFICATION_STATUS_FAILED);
  REQUIRE(resp.err_msg().find("injected verification failure") != std::string::npos);

  tensorcast::daemon::v2::WaitReplicaVerificationResponse resp2;
  grpc::ServerContext ctx2;
  const auto st2 = svc.WaitReplicaVerification(&ctx2, &req, &resp2);
  REQUIRE(st2.ok());
  REQUIRE(resp2.status() == tensorcast::daemon::v2::VerificationStatus::VERIFICATION_STATUS_FAILED);
  REQUIRE(resp2.err_msg().find("injected verification failure") != std::string::npos);
}
