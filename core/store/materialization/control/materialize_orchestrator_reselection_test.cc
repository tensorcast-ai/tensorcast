// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/control/materialize_orchestrator.h"

#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "core/common/memory/memory_location.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/testing/recording_global_store_client.h"

namespace tensorcast::store::materialization::control {
namespace {

using tensorcast::store::components::TransportCompletionOutcome;
using tensorcast::store::components::TransportSession;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::loading::MaterializationSource;
using tensorcast::store::loading::MaterializeHints;
using tensorcast::store::loading::ReplicaHandle;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::loading::ReplicaTarget;
using tensorcast::store::testing::RecordingGlobalStoreClient;

class FakeMaterializationBackend final : public MaterializationBackend {
 public:
  struct P2pAttempt {
    std::string source_ip;
    uint16_t source_port{0};
  };

  std::vector<P2pAttempt> p2p_attempts;
  std::vector<absl::Status> p2p_scripted_statuses;
  std::vector<absl::Status> register_scripted_statuses;
  absl::Status register_status{absl::OkStatus()};
  std::function<void()> on_register;
  int register_calls{0};

  absl::StatusOr<ReplicaHandle> ingest_from_p2p(
      const std::string& artifact_identifier,
      const P2PSource& source,
      const ReplicaTarget& target,
      const MaterializeHints&) override {
    p2p_attempts.push_back(P2pAttempt{.source_ip = source.ip, .source_port = source.port});
    const size_t call_index = p2p_attempts.size() - 1;
    if (call_index < p2p_scripted_statuses.size() && !p2p_scripted_statuses[call_index].ok()) {
      return p2p_scripted_statuses[call_index];
    }

    DeviceKey device{
        .type = (target.location.type == common::memory::MemoryLocation::GPU) ? DeviceType::GPU : DeviceType::CPU,
        .ordinal = target.location.device_id,
        .uuid = "",
    };
    if (device.type == DeviceType::CPU) {
      device.ordinal = -1;
    }

    ReplicaHandle handle;
    handle.replica_key = ReplicaKey{
        .artifact_id = artifact_identifier,
        .view_id = std::nullopt,
        .device = device,
        .replica = 0,
    };
    handle.source = MaterializationSource::kP2P;
    return handle;
  }

  absl::StatusOr<ReplicaHandle> ingest_from_disk(
      const std::string&,
      const DiskSource&,
      const ReplicaTarget&,
      const MaterializeHints&) override {
    return absl::UnimplementedError("disk path should not be used in this test");
  }

  absl::Status register_replica_with_global_store(const ReplicaKey&, std::string_view, std::string_view) override {
    const size_t call_index = static_cast<size_t>(register_calls);
    register_calls += 1;
    if (on_register) {
      on_register();
    }
    if (call_index < register_scripted_statuses.size()) {
      return register_scripted_statuses[call_index];
    }
    return register_status;
  }
};

TransportSession make_transport_session(
    std::string_view transport_id,
    std::string_view node_id,
    std::string_view node_address,
    uint32_t node_port,
    common::memory::MemoryLocation memory_type,
    int32_t device_id) {
  TransportSession session;
  session.transport_id = std::string(transport_id);
  session.remote_replica.node_id = std::string(node_id);
  session.remote_replica.node_address = std::string(node_address);
  session.remote_replica.node_port = node_port;
  session.remote_replica.memory_size = 1024;
  session.remote_replica.memory_type = memory_type;
  session.remote_replica.device_id = device_id;
  session.remote_replica.remote_memory_keys = {"tensor.data_0"};
  session.remote_replica.buffer_sizes = {1024};
  session.remote_replica.verification_json = "{}";
  return session;
}

DeviceKey make_gpu_target(int ordinal) {
  return DeviceKey{
      .type = DeviceType::GPU,
      .ordinal = ordinal,
      .uuid = "",
  };
}

absl::StatusOr<TransportSession> request_replica_transport_with_legacy_positional_args(
    components::IGlobalStoreClient& client,
    const DeviceKey& target_device) {
  return client.request_replica_transport(
      "artifact-legacy",
      "node-legacy",
      "10.9.9.1",
      50090,
      target_device,
      5000,
      std::nullopt,
      "worker-legacy",
      "request-legacy");
}

TEST_CASE("MaterializeOrchestrator accepts local route returned by Global Store", "[store][materialize][reselection]") {
  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->allow_replica_transport = true;
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-local", "node-local", "10.1.1.1", 50001, common::memory::MemoryLocation::GPU, 0));

  FakeMaterializationBackend backend;
  MaterializeHints hints;
  hints.artifact_id = "artifact-reselect-local";
  hints.allow_p2p = true;
  hints.allow_disk = false;

  components::WorkerIdentity local_identity{
      .node_id = "node-local",
      .node_address = "10.1.1.1",
      .p2p_port = 50001,
  };
  MaterializeOrchestrator orchestrator(
      gsl::not_null<MaterializationBackend*>{&backend},
      gsl::not_null<components::IGlobalStoreClient*>{gs_client.get()},
      local_identity);

  auto result = orchestrator.run("artifact-reselect-local", make_gpu_target(0), hints, std::nullopt);
  REQUIRE(result.ok());
  REQUIRE(backend.register_calls == 1);
  REQUIRE(backend.p2p_attempts.size() == 1);
  CHECK(backend.p2p_attempts.front().source_ip == "10.1.1.1");
  CHECK(backend.p2p_attempts.front().source_port == 50001);
  REQUIRE(gs_client->replica_requests.size() == 1);
  REQUIRE(gs_client->completed_transport_outcomes.size() == 1);
  CHECK(gs_client->completed_transport_outcomes[0] == TransportCompletionOutcome::kSuccess);
}

TEST_CASE("GlobalStoreClient request transport keeps legacy positional arguments", "[store][materialize][reselection]") {
  RecordingGlobalStoreClient gs_client;
  gs_client.connected = true;
  gs_client.allow_replica_transport = true;

  auto result = request_replica_transport_with_legacy_positional_args(gs_client, make_gpu_target(0));

  REQUIRE(result.ok());
  REQUIRE(gs_client.replica_requests.size() == 1);
  CHECK(gs_client.replica_request_requester_worker_ids.front() == "worker-legacy");
  CHECK(gs_client.replica_request_request_ids.front() == "request-legacy");
  REQUIRE(gs_client.replica_request_broadcast_hints.size() == 1);
  CHECK_FALSE(gs_client.replica_request_broadcast_hints.front().has_value());
}

TEST_CASE(
    "MaterializeOrchestrator preserves non-broadcast transport completion before best-effort registration",
    "[store][materialize][reselection]") {
  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->allow_replica_transport = true;
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-non-broadcast", "node-remote", "10.1.1.2", 50002, common::memory::MemoryLocation::GPU, 0));

  bool completed_before_register = false;
  FakeMaterializationBackend backend;
  backend.register_status = absl::UnavailableError("best-effort register failed");
  backend.on_register = [&]() {
    completed_before_register = !gs_client->completed_transport_ids.empty();
  };

  MaterializeHints hints;
  hints.artifact_id = "artifact-non-broadcast-register-fails";
  hints.allow_p2p = true;
  hints.allow_disk = false;

  components::WorkerIdentity local_identity{
      .worker_id = "worker-local",
      .node_id = "node-local",
      .node_address = "10.1.1.1",
      .p2p_port = 50001,
  };
  MaterializeOrchestrator orchestrator(
      gsl::not_null<MaterializationBackend*>{&backend},
      gsl::not_null<components::IGlobalStoreClient*>{gs_client.get()},
      local_identity);

  auto result = orchestrator.run("artifact-non-broadcast-register-fails", make_gpu_target(0), hints, std::nullopt);
  REQUIRE(result.ok());
  REQUIRE(backend.register_calls == 1);
  REQUIRE(gs_client->completed_transport_ids.size() == 1);
  CHECK(gs_client->completed_transport_outcomes.front() == TransportCompletionOutcome::kSuccess);
  CHECK(completed_before_register);
}

TEST_CASE("MaterializeOrchestrator reselects source after retryable P2P failure", "[store][materialize][reselection]") {
  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->allow_replica_transport = true;
  gs_client->push_scripted_transport_session(
      make_transport_session("transport-a", "node-a", "10.2.2.1", 50011, common::memory::MemoryLocation::GPU, 0));
  gs_client->push_scripted_transport_session(
      make_transport_session("transport-b", "node-b", "10.2.2.2", 50012, common::memory::MemoryLocation::GPU, 0));

  FakeMaterializationBackend backend;
  backend.p2p_scripted_statuses = {absl::UnavailableError("stale source"), absl::OkStatus()};

  MaterializeHints hints;
  hints.artifact_id = "artifact-reselect-p2p-failure";
  hints.allow_p2p = true;
  hints.allow_disk = false;

  components::WorkerIdentity local_identity{
      .node_id = "node-local",
      .node_address = "10.2.2.99",
      .p2p_port = 50999,
  };
  MaterializeOrchestrator orchestrator(
      gsl::not_null<MaterializationBackend*>{&backend},
      gsl::not_null<components::IGlobalStoreClient*>{gs_client.get()},
      local_identity);

  auto result = orchestrator.run("artifact-reselect-p2p-failure", make_gpu_target(0), hints, std::nullopt);
  REQUIRE(result.ok());
  REQUIRE(backend.register_calls == 1);
  REQUIRE(backend.p2p_attempts.size() == 2);
  CHECK(backend.p2p_attempts[0].source_ip == "10.2.2.1");
  CHECK(backend.p2p_attempts[1].source_ip == "10.2.2.2");
  REQUIRE(gs_client->replica_requests.size() == 2);
  REQUIRE(gs_client->completed_transport_outcomes.size() == 2);
  CHECK(gs_client->completed_transport_outcomes[0] == TransportCompletionOutcome::kFailed);
  CHECK(gs_client->completed_transport_outcomes[1] == TransportCompletionOutcome::kSuccess);
}

TEST_CASE(
    "MaterializeOrchestrator reselection is deadline-aware instead of fixed attempts",
    "[store][materialize][reselection]") {
  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->allow_replica_transport = true;
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-local-1", "node-local", "10.3.3.1", 50021, common::memory::MemoryLocation::GPU, 0));
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-local-2", "node-local", "10.3.3.1", 50021, common::memory::MemoryLocation::GPU, 0));
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-local-3", "node-local", "10.3.3.1", 50021, common::memory::MemoryLocation::GPU, 0));
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-local-4", "node-local", "10.3.3.1", 50021, common::memory::MemoryLocation::GPU, 0));
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-remote", "node-remote", "10.3.3.2", 50022, common::memory::MemoryLocation::GPU, 0));

  FakeMaterializationBackend backend;
  backend.p2p_scripted_statuses = {
      absl::UnavailableError("retryable local source 1"),
      absl::UnavailableError("retryable local source 2"),
      absl::UnavailableError("retryable local source 3"),
      absl::UnavailableError("retryable local source 4"),
      absl::OkStatus(),
  };
  MaterializeHints hints;
  hints.artifact_id = "artifact-reselect-deadline-aware";
  hints.allow_p2p = true;
  hints.allow_disk = false;
  hints.request_budget = std::chrono::seconds(2);
  hints.transport_wait_timeout = std::chrono::milliseconds(1200);

  components::WorkerIdentity local_identity{
      .node_id = "node-local",
      .node_address = "10.3.3.1",
      .p2p_port = 50021,
  };
  MaterializeOrchestrator orchestrator(
      gsl::not_null<MaterializationBackend*>{&backend},
      gsl::not_null<components::IGlobalStoreClient*>{gs_client.get()},
      local_identity);

  auto result = orchestrator.run("artifact-reselect-deadline-aware", make_gpu_target(0), hints, std::nullopt);
  REQUIRE(result.ok());
  REQUIRE(backend.p2p_attempts.size() == 5);
  CHECK(backend.p2p_attempts.back().source_ip == "10.3.3.2");
  REQUIRE(gs_client->replica_requests.size() == 5);
  for (uint32_t timeout_ms : gs_client->replica_request_wait_timeouts_ms) {
    CHECK(timeout_ms <= 1200);
    CHECK(timeout_ms > 0);
  }
}

TEST_CASE(
    "MaterializeOrchestrator probes view transport briefly before canonical fallback",
    "[store][materialize][reselection][view_fallback]") {
  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->allow_view_transport = false;
  gs_client->allow_replica_transport = true;
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-view-fallback", "node-remote", "10.4.4.2", 50032, common::memory::MemoryLocation::GPU, 0));

  FakeMaterializationBackend backend;
  MaterializeHints hints;
  hints.artifact_id = "artifact-view-fallback-probe";
  hints.allow_p2p = true;
  hints.allow_disk = false;
  hints.transport_wait_timeout = std::chrono::milliseconds(5000);
  loading::VariantIdentity variant;
  variant.canonical_artifact_id = hints.artifact_id;
  variant.view_id = std::string("view:tp0");
  hints.variant = std::move(variant);

  components::WorkerIdentity local_identity{
      .node_id = "node-local",
      .node_address = "10.4.4.1",
      .p2p_port = 50031,
  };
  MaterializeOrchestrator orchestrator(
      gsl::not_null<MaterializationBackend*>{&backend},
      gsl::not_null<components::IGlobalStoreClient*>{gs_client.get()},
      local_identity);

  auto result = orchestrator.run("artifact-view-fallback-probe", make_gpu_target(0), hints, std::nullopt);
  REQUIRE(result.ok());
  REQUIRE(backend.register_calls == 1);
  REQUIRE(backend.p2p_attempts.size() == 1);
  REQUIRE(gs_client->view_requests.size() == 1);
  REQUIRE(gs_client->replica_requests.size() == 1);
  REQUIRE(gs_client->view_request_wait_timeouts_ms.size() == 1);
  REQUIRE(gs_client->replica_request_wait_timeouts_ms.size() == 1);
  CHECK(gs_client->view_request_wait_timeouts_ms.front() > 0);
  CHECK(gs_client->view_request_wait_timeouts_ms.front() < gs_client->replica_request_wait_timeouts_ms.front());
  CHECK(gs_client->replica_request_wait_timeouts_ms.front() == 5000);
}

TEST_CASE(
    "MaterializeOrchestrator propagates broadcast transport parent hint",
    "[store][materialize][reselection][broadcast]") {
  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->allow_replica_transport = true;
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-broadcast", "node-remote", "10.6.6.2", 50052, common::memory::MemoryLocation::GPU, 0));

  FakeMaterializationBackend backend;
  MaterializeHints hints;
  hints.artifact_id = "artifact-broadcast-hint";
  hints.allow_p2p = true;
  hints.allow_disk = false;
  hints.transport_request_id = "request-broadcast-1";
  hints.broadcast = loading::BroadcastHint{
      .session_id = "session-a",
      .strict_parent = true,
  };

  components::WorkerIdentity local_identity{
      .worker_id = "worker-local",
      .node_id = "node-local",
      .node_address = "10.6.6.1",
      .p2p_port = 50051,
  };
  MaterializeOrchestrator orchestrator(
      gsl::not_null<MaterializationBackend*>{&backend},
      gsl::not_null<components::IGlobalStoreClient*>{gs_client.get()},
      local_identity);

  auto result = orchestrator.run("artifact-broadcast-hint", make_gpu_target(0), hints, std::nullopt);
  REQUIRE(result.ok());
  REQUIRE(gs_client->replica_request_broadcast_hints.size() == 1);
  REQUIRE(gs_client->replica_request_broadcast_hints.front().has_value());
  CHECK(gs_client->replica_request_broadcast_hints.front()->session_id == "session-a");
  CHECK(gs_client->replica_request_broadcast_hints.front()->strict_parent);
}

TEST_CASE(
    "MaterializeOrchestrator completes broadcast transport as failed when local registration fails",
    "[store][materialize][reselection][broadcast]") {
  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->allow_replica_transport = false;
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-broadcast", "node-remote", "10.7.7.2", 50062, common::memory::MemoryLocation::GPU, 0));

  FakeMaterializationBackend backend;
  backend.register_status = absl::UnavailableError("register failed");

  MaterializeHints hints;
  hints.artifact_id = "artifact-broadcast-register-fails";
  hints.allow_p2p = true;
  hints.allow_disk = false;
  hints.transport_request_id = "request-broadcast-register-fails";
  hints.broadcast = loading::BroadcastHint{
      .session_id = "session-register-fails",
      .strict_parent = true,
  };

  components::WorkerIdentity local_identity{
      .worker_id = "worker-local",
      .node_id = "node-local",
      .node_address = "10.7.7.1",
      .p2p_port = 50061,
  };
  MaterializeOrchestrator orchestrator(
      gsl::not_null<MaterializationBackend*>{&backend},
      gsl::not_null<components::IGlobalStoreClient*>{gs_client.get()},
      local_identity);

  auto result = orchestrator.run("artifact-broadcast-register-fails", make_gpu_target(0), hints, std::nullopt);
  REQUIRE_FALSE(result.ok());
  REQUIRE(backend.register_calls == 1);
  REQUIRE(gs_client->completed_transport_ids.size() == 1);
  CHECK(gs_client->completed_transport_ids.front() == "transport-broadcast");
  REQUIRE(gs_client->completed_transport_outcomes.size() == 1);
  CHECK(gs_client->completed_transport_outcomes.front() == TransportCompletionOutcome::kFailed);
}

TEST_CASE(
    "MaterializeOrchestrator retries broadcast source selection after registration failure",
    "[store][materialize][reselection][broadcast]") {
  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->allow_replica_transport = true;
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-broadcast-a", "node-a", "10.8.8.2", 50072, common::memory::MemoryLocation::GPU, 0));
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-broadcast-b", "node-b", "10.8.8.3", 50073, common::memory::MemoryLocation::GPU, 0));

  FakeMaterializationBackend backend;
  backend.register_scripted_statuses = {absl::UnavailableError("register failed"), absl::OkStatus()};

  MaterializeHints hints;
  hints.artifact_id = "artifact-broadcast-register-retry";
  hints.allow_p2p = true;
  hints.allow_disk = false;
  hints.transport_request_id = "request-broadcast-register-retry";
  hints.broadcast = loading::BroadcastHint{
      .session_id = "session-register-retry",
      .strict_parent = true,
  };

  components::WorkerIdentity local_identity{
      .worker_id = "worker-local",
      .node_id = "node-local",
      .node_address = "10.8.8.1",
      .p2p_port = 50071,
  };
  MaterializeOrchestrator orchestrator(
      gsl::not_null<MaterializationBackend*>{&backend},
      gsl::not_null<components::IGlobalStoreClient*>{gs_client.get()},
      local_identity);

  auto result = orchestrator.run("artifact-broadcast-register-retry", make_gpu_target(0), hints, std::nullopt);
  REQUIRE(result.ok());
  REQUIRE(backend.register_calls == 2);
  REQUIRE(backend.p2p_attempts.size() == 2);
  CHECK(backend.p2p_attempts[0].source_ip == "10.8.8.2");
  CHECK(backend.p2p_attempts[1].source_ip == "10.8.8.3");
  REQUIRE(gs_client->completed_transport_ids.size() == 2);
  CHECK(gs_client->completed_transport_ids[0] == "transport-broadcast-a");
  CHECK(gs_client->completed_transport_outcomes[0] == TransportCompletionOutcome::kFailed);
  CHECK(gs_client->completed_transport_ids[1] == "transport-broadcast-b");
  CHECK(gs_client->completed_transport_outcomes[1] == TransportCompletionOutcome::kSuccess);
}

TEST_CASE(
    "MaterializeOrchestrator returns terminal broadcast registration failure status",
    "[store][materialize][reselection][broadcast]") {
  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->allow_replica_transport = true;
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-broadcast-terminal", "node-remote", "10.9.9.2", 50092, common::memory::MemoryLocation::GPU, 0));

  FakeMaterializationBackend backend;
  backend.register_status = absl::InvalidArgumentError("invalid child replica");

  MaterializeHints hints;
  hints.artifact_id = "artifact-broadcast-terminal-register-fails";
  hints.allow_p2p = true;
  hints.allow_disk = false;
  hints.transport_request_id = "request-broadcast-terminal-register-fails";
  hints.broadcast = loading::BroadcastHint{
      .session_id = "session-terminal-register-fails",
      .strict_parent = true,
  };

  components::WorkerIdentity local_identity{
      .worker_id = "worker-local",
      .node_id = "node-local",
      .node_address = "10.9.9.1",
      .p2p_port = 50091,
  };
  MaterializeOrchestrator orchestrator(
      gsl::not_null<MaterializationBackend*>{&backend},
      gsl::not_null<components::IGlobalStoreClient*>{gs_client.get()},
      local_identity);

  auto result =
      orchestrator.run("artifact-broadcast-terminal-register-fails", make_gpu_target(0), hints, std::nullopt);
  REQUIRE_FALSE(result.ok());
  CHECK(absl::IsInvalidArgument(result.status()));
  REQUIRE(gs_client->completed_transport_ids.size() == 1);
  CHECK(gs_client->completed_transport_ids.front() == "transport-broadcast-terminal");
  CHECK(gs_client->completed_transport_outcomes.front() == TransportCompletionOutcome::kFailed);
}

TEST_CASE(
    "MaterializeOrchestrator propagates transport scheduler hint metadata",
    "[store][materialize][reselection][scheduler_hint]") {
  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->allow_replica_transport = true;
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-hint", "node-remote", "10.5.5.2", 50042, common::memory::MemoryLocation::GPU, 0));

  FakeMaterializationBackend backend;
  MaterializeHints hints;
  hints.artifact_id = "artifact-scheduler-hint";
  hints.allow_p2p = true;
  hints.allow_disk = false;
  hints.transport_request_id = "request-hint-1";
  hints.transport_requester_worker_id = "worker-hint-1";
  hints.transport_scheduling_group = loading::TransportSchedulingGroupHint{
      .group_id = "group-a",
      .group_kind = "tp_rank",
      .total_parts = 4,
      .part_id = "part-0",
      .priority = 7,
      .epoch = 9,
  };

  components::WorkerIdentity local_identity{
      .worker_id = "worker-local",
      .node_id = "node-local",
      .node_address = "10.5.5.1",
      .p2p_port = 50041,
  };
  MaterializeOrchestrator orchestrator(
      gsl::not_null<MaterializationBackend*>{&backend},
      gsl::not_null<components::IGlobalStoreClient*>{gs_client.get()},
      local_identity);

  auto result = orchestrator.run("artifact-scheduler-hint", make_gpu_target(0), hints, std::nullopt);
  REQUIRE(result.ok());
  REQUIRE(gs_client->replica_request_request_ids.size() == 1);
  CHECK(gs_client->replica_request_request_ids.front() == "request-hint-1");
  REQUIRE(gs_client->replica_request_requester_worker_ids.size() == 1);
  CHECK(gs_client->replica_request_requester_worker_ids.front() == "worker-hint-1");
  REQUIRE(gs_client->replica_request_groups.size() == 1);
  REQUIRE(gs_client->replica_request_groups.front().has_value());
  CHECK(gs_client->replica_request_groups.front()->group_id == "group-a");
  CHECK(gs_client->replica_request_groups.front()->group_kind == "tp_rank");
  CHECK(gs_client->replica_request_groups.front()->total_parts == 4);
  CHECK(gs_client->replica_request_groups.front()->part_id == "part-0");
  CHECK(gs_client->replica_request_groups.front()->priority == 7);
  CHECK(gs_client->replica_request_groups.front()->epoch == 9);
}

} // namespace
} // namespace tensorcast::store::materialization::control
