// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/control/materialize_orchestrator.h"

#include <chrono>
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
    register_calls += 1;
    return absl::OkStatus();
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

TEST_CASE("MaterializeOrchestrator reselects source after stale-local route", "[store][materialize][reselection]") {
  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->allow_replica_transport = true;
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-local", "node-local", "10.1.1.1", 50001, common::memory::MemoryLocation::GPU, 0));
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-remote", "node-remote", "10.1.1.2", 50002, common::memory::MemoryLocation::GPU, 0));

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
  CHECK(backend.p2p_attempts.front().source_ip == "10.1.1.2");
  CHECK(backend.p2p_attempts.front().source_port == 50002);
  REQUIRE(gs_client->replica_requests.size() == 2);
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
  REQUIRE(backend.p2p_attempts.size() == 1);
  CHECK(backend.p2p_attempts.front().source_ip == "10.3.3.2");
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

} // namespace
} // namespace tensorcast::store::materialization::control
