// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>

#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/testing/common.h"
#include "daemon/worker_lifecycle_manager.h"
#include "grpcpp/grpcpp.h"
#include "gsl/pointers"
#include "nlohmann/json.hpp"
#include "tensorcast/global_store/v1/global_store.grpc.pb.h"
#include "tensorcast/global_store/v1/global_store.pb.h"
#include "tensorcast/memory_tier/v1/memory_tier.grpc.pb.h"

namespace fs = std::filesystem;
using tensorcast::DeviceType;
using tensorcast::daemon::StoreDaemonServiceImpl;
using tensorcast::daemon::WorkerLifecycleManager;
using tensorcast::store::DeviceKey;
using tensorcast::store::StoreEngine;
using tensorcast::store::StoreEngineOptions;
namespace global_store = tensorcast::global_store::v1;
namespace memory_tier = tensorcast::memory_tier::v1;

namespace {

constexpr uint16_t kTestP2PPort = 45000;

} // namespace

static DeviceKey make_gpu_key(int ordinal) {
  return DeviceKey{.type = DeviceType::GPU, .ordinal = ordinal, /*uuid=*/.uuid = ""};
}

struct ArtifactFixture {
  std::string logical_name;
  std::string artifact_id;
  fs::path dir;
};

static std::string read_artifact_id_from_descriptor(const fs::path& artifact_dir) {
  const auto descriptor_path = artifact_dir / "artifact_descriptor.json";
  std::ifstream descriptor(descriptor_path);
  REQUIRE(descriptor.is_open());
  nlohmann::json desc_json;
  descriptor >> desc_json;
  REQUIRE(desc_json.contains("artifact_id"));
  REQUIRE(desc_json["artifact_id"].is_string());
  return desc_json["artifact_id"].get<std::string>();
}

static ArtifactFixture make_standard_artifact(
    const fs::path& root,
    const std::string& logical_name,
    uint64_t payload_bytes,
    char start_char = 'A') {
  ArtifactFixture fixture;
  fixture.logical_name = logical_name;
  fixture.dir = root / logical_name;
  fs::create_directories(fixture.dir);
  REQUIRE(
      tensorcast::testing::create_dummy_file(
          fixture.dir / "tensor.data_0", static_cast<size_t>(payload_bytes), start_char));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(fixture.dir).ok());
  fixture.artifact_id = read_artifact_id_from_descriptor(fixture.dir);
  REQUIRE_FALSE(fixture.artifact_id.empty());
  REQUIRE(absl::StartsWith(fixture.artifact_id, "mi2:"));
  REQUIRE(fixture.artifact_id != fixture.logical_name);
  return fixture;
}

// Minimal in-process fake Global Store service
class FakeGlobalStoreService final : public global_store::GlobalStoreService::Service,
                                     public memory_tier::MemoryTierService::Service {
 public:
  explicit FakeGlobalStoreService(std::vector<std::string> expected_ids, std::string obsolete_id)
      : expected_ids_(std::move(expected_ids)) {
    if (!obsolete_id.empty())
      hb_obsolete_ids_.push_back(std::move(obsolete_id));
  }

  // Config setters (callable by tests after server start)
  void set_heartbeat_obsolete(std::vector<std::string> ids) {
    hb_obsolete_ids_ = std::move(ids);
  }

  void set_heartbeat_sync_required(bool required, uint64_t expected_ver) {
    hb_state_sync_required_ = required;
    hb_expected_state_version_ = expected_ver;
  }

  void set_sync_remove_ids(std::vector<std::string> ids) {
    sync_remove_ids_ = std::move(ids);
  }

  void set_sync_should_fail(bool v) {
    sync_should_fail_ = v;
  }

  void set_outstanding_leases(std::vector<memory_tier::MemoryTierLease> leases) {
    absl::MutexLock l(&mu_);
    outstanding_leases_ = std::move(leases);
  }

  void reset_chunk_updates() {
    absl::MutexLock l(&mu_);
    total_chunk_updates_ = 0;
  }

  uint32_t total_chunk_updates() const {
    absl::MutexLock l(&mu_);
    return total_chunk_updates_;
  }

  std::vector<memory_tier::MemoryTierStatus> memory_tier_statuses() const {
    absl::MutexLock l(&mu_);
    return published_statuses_;
  }

  std::vector<memory_tier::AcknowledgeMemoryTierLeaseRequest> ack_requests() const {
    absl::MutexLock l(&mu_);
    return ack_requests_;
  }

  std::vector<memory_tier::MemoryTierLease> leases_snapshot() const {
    absl::MutexLock l(&mu_);
    return outstanding_leases_;
  }

  ::grpc::Status HealthCheck(
      ::grpc::ServerContext* /*context*/,
      const global_store::HealthCheckRequest* /*request*/,
      global_store::HealthCheckResponse* resp) override {
    resp->set_status(global_store::STATUS_OK);
    return ::grpc::Status::OK;
  }

  ::grpc::Status RegisterWorker(
      ::grpc::ServerContext* /*context*/,
      const global_store::RegisterWorkerRequest* req,
      global_store::RegisterWorkerResponse* resp) override {
    (void)req;
    resp->set_status(global_store::STATUS_OK);
    resp->set_worker_id("worker-1");
    resp->set_heartbeat_interval_ms(1000);
    resp->set_state_sync_required(true);
    resp->set_expected_state_version(1);
    return ::grpc::Status::OK;
  }

  ::grpc::Status WorkerHeartbeat(
      ::grpc::ServerContext* /*context*/,
      const global_store::WorkerHeartbeatRequest* req,
      global_store::WorkerHeartbeatResponse* resp) override {
    (void)req;
    resp->set_status(global_store::STATUS_OK);
    resp->set_state_sync_required(hb_state_sync_required_);
    if (hb_expected_state_version_ > 0)
      resp->set_expected_state_version(hb_expected_state_version_);
    for (const auto& id : hb_obsolete_ids_)
      resp->add_obsolete_replicas(id);
    return ::grpc::Status::OK;
  }

  ::grpc::Status RequestFullStateSync(
      ::grpc::ServerContext* /*context*/,
      const global_store::RequestFullStateSyncRequest* req,
      global_store::RequestFullStateSyncResponse* resp) override {
    (void)req;
    resp->set_status(global_store::STATUS_OK);
    resp->set_new_state_version(1);
    resp->set_new_state_checksum("v1");
    for (const auto& id : expected_ids_) {
      auto* rep = resp->add_expected_replicas();
      rep->mutable_ref()->set_artifact_id(id);
      // Minimal MemoryInfo; the daemon only inspects artifact_id in apply_full_state()
      auto* mi = rep->mutable_memory_info();
      mi->set_memory_type(tensorcast::common::v1::MEMORY_TYPE_GPU);
      mi->set_device_id(0);
      mi->set_memory_size(0);
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status SynchronizeWorkerState(
      ::grpc::ServerContext* /*context*/,
      const global_store::SynchronizeWorkerStateRequest* req,
      global_store::SynchronizeWorkerStateResponse* resp) override {
    (void)req;
    if (sync_should_fail_) {
      resp->set_status(global_store::STATUS_ERROR);
      return ::grpc::Status::OK;
    }
    resp->set_status(global_store::STATUS_OK);
    resp->set_new_state_version(2);
    resp->set_new_state_checksum("v2");
    for (const auto& id : sync_remove_ids_) {
      auto* ch = resp->add_state_changes();
      ch->set_type(global_store::StateChange::CHANGE_TYPE_REMOVE_REPLICA);
      ch->mutable_replica_info()->mutable_ref()->set_artifact_id(id);
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status BatchUpdateChunkStates(
      ::grpc::ServerContext* /*context*/,
      const global_store::BatchUpdateChunkStatesRequest* req,
      global_store::BatchUpdateChunkStatesResponse* resp) override {
    absl::MutexLock l(&mu_);
    resp->set_status(global_store::STATUS_OK);
    resp->set_updates_applied(req->updates_size());
    total_chunk_updates_ += req->updates_size();
    return ::grpc::Status::OK;
  }

  ::grpc::Status UnregisterWorker(
      ::grpc::ServerContext* /*context*/,
      const global_store::UnregisterWorkerRequest* req,
      global_store::UnregisterWorkerResponse* resp) override {
    (void)req;
    resp->set_status(global_store::STATUS_OK);
    return ::grpc::Status::OK;
  }

  ::grpc::Status PublishMemoryTierStatus(
      ::grpc::ServerContext* /*context*/,
      const memory_tier::PublishMemoryTierStatusRequest* req,
      memory_tier::PublishMemoryTierStatusResponse* resp) override {
    absl::MutexLock l(&mu_);
    published_statuses_.push_back(req->status());
    (void)resp;
    return ::grpc::Status::OK;
  }

  ::grpc::Status ListOutstandingLeases(
      ::grpc::ServerContext* /*context*/,
      const memory_tier::ListOutstandingLeasesRequest* req,
      memory_tier::ListOutstandingLeasesResponse* resp) override {
    absl::MutexLock l(&mu_);
    resp->mutable_leases()->Clear();
    for (const auto& lease : outstanding_leases_) {
      bool include = req->states_size() == 0;
      if (!include) {
        for (const auto state : req->states()) {
          if (lease.state() == state) {
            include = true;
            break;
          }
        }
      }
      if (include) {
        *resp->add_leases() = lease;
      }
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status AcknowledgeMemoryTierLease(
      ::grpc::ServerContext* /*context*/,
      const memory_tier::AcknowledgeMemoryTierLeaseRequest* req,
      memory_tier::AcknowledgeMemoryTierLeaseResponse* resp) override {
    absl::MutexLock l(&mu_);
    ack_requests_.push_back(*req);
    for (auto& lease : outstanding_leases_) {
      if (lease.lease_id() != req->lease_id())
        continue;
      if (req->action() == memory_tier::LEASE_ACK_ACTION_ACQUIRED) {
        lease.set_state(memory_tier::LEASE_STATE_ACTIVE);
      } else if (req->action() == memory_tier::LEASE_ACK_ACTION_RELEASED) {
        lease.set_state(memory_tier::LEASE_STATE_EXPIRED);
      }
      if (req->has_chunk_range()) {
        lease.mutable_chunk_range()->set_start(req->chunk_range().start());
        lease.mutable_chunk_range()->set_count(req->chunk_range().count());
      }
      lease.mutable_chunk_ids()->Clear();
      for (auto id : req->chunk_ids()) {
        lease.add_chunk_ids(id);
      }
      lease.set_bytes(req->bytes());
      lease.set_ledger_version(req->ledger_version());
      lease.set_ack_epoch_ns(req->ack_epoch_ns());
      lease.set_request_id(req->request_id());
      lease.set_node_id(req->node_id());
      lease.set_artifact_id(req->artifact_id());
      *resp->mutable_lease() = lease;
      return ::grpc::Status::OK;
    }
    return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "lease not found");
  }

  ::grpc::Status RequestMemoryTierLease(
      ::grpc::ServerContext* /*context*/,
      const memory_tier::RequestMemoryTierLeaseRequest* /*request*/,
      memory_tier::RequestMemoryTierLeaseResponse* /*response*/) override {
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "not implemented in fake");
  }

  ::grpc::Status RevokeMemoryTierLease(
      ::grpc::ServerContext* /*context*/,
      const memory_tier::RevokeMemoryTierLeaseRequest* /*request*/,
      memory_tier::RevokeMemoryTierLeaseResponse* /*response*/) override {
    return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "not implemented in fake");
  }

 private:
  std::vector<std::string> expected_ids_;
  std::vector<std::string> hb_obsolete_ids_;
  bool hb_state_sync_required_{false};
  uint64_t hb_expected_state_version_{0};
  std::vector<std::string> sync_remove_ids_;
  bool sync_should_fail_{false};
  mutable absl::Mutex mu_;
  uint32_t total_chunk_updates_{0};
  std::vector<memory_tier::MemoryTierLease> outstanding_leases_ ABSL_GUARDED_BY(mu_);
  std::vector<memory_tier::MemoryTierStatus> published_statuses_ ABSL_GUARDED_BY(mu_);
  std::vector<memory_tier::AcknowledgeMemoryTierLeaseRequest> ack_requests_ ABSL_GUARDED_BY(mu_);
};

struct TestServer {
  std::unique_ptr<grpc::Server> server;
  int selected_port{0};
  std::unique_ptr<FakeGlobalStoreService> service;
};

static TestServer start_fake_server(const std::vector<std::string>& expected_ids, const std::string& obsolete_id) {
  TestServer ts;
  ts.service = std::make_unique<FakeGlobalStoreService>(expected_ids, obsolete_id);
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
  builder.RegisterService(static_cast<global_store::GlobalStoreService::Service*>(ts.service.get()));
  builder.RegisterService(static_cast<memory_tier::MemoryTierService::Service*>(ts.service.get()));
  ts.server = builder.BuildAndStart();
  ts.selected_port = port;
  return ts;
}

static void require_replica_registered(const StoreEngine& store, const ArtifactFixture& artifact) {
  const auto infos = store.get_all_replicas_info();
  for (const auto& info : infos) {
    if (info.artifact_id == artifact.artifact_id) {
      return;
    }
  }
  FAIL("Replica not registered with canonical artifact_id=" << artifact.artifact_id);
}

static void load_artifact_gpu(StoreEngine& store, const ArtifactFixture& artifact) {
  tensorcast::store::loading::MaterializeHints hints;
  hints.artifact_id = artifact.artifact_id;
  hints.disk_path = artifact.dir.string();
  auto handle_or =
      store.materialize_replica(make_gpu_key(0), tensorcast::store::StoreEngine::MaterializeMode::LOAD_ONLY, hints);
  INFO("load_artifact_gpu status=" << handle_or.status().ToString());
  REQUIRE(handle_or.ok());
  auto handle = std::move(handle_or.value());
  const auto ready_status = handle.wait_ready(std::chrono::milliseconds(30000));
  INFO("wait_ready(gpu) status=" << ready_status.ToString());
  tensorcast::store::loading::ReplicaKey key{
      .artifact_id = artifact.artifact_id, .view_id = std::nullopt, .device = make_gpu_key(0), .replica = 0};
  const int rc = store.wait_replica_ready(key);
  INFO("wait_replica_ready(gpu) rc=" << rc);
  REQUIRE(rc == 0);
  require_replica_registered(store, artifact);
}

static void load_artifact_cpu(StoreEngine& store, const ArtifactFixture& artifact) {
  tensorcast::store::loading::MaterializeHints hints;
  hints.artifact_id = artifact.artifact_id;
  hints.disk_path = artifact.dir.string();
  DeviceKey cpu{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  auto handle_or = store.materialize_replica(cpu, tensorcast::store::StoreEngine::MaterializeMode::LOAD_ONLY, hints);
  INFO("load_artifact_cpu status=" << handle_or.status().ToString());
  REQUIRE(handle_or.ok());
  auto handle = std::move(handle_or.value());
  const auto ready_status = handle.wait_ready(std::chrono::milliseconds(30000));
  INFO("wait_ready(cpu) status=" << ready_status.ToString());
  tensorcast::store::loading::ReplicaKey key{
      .artifact_id = artifact.artifact_id, .view_id = std::nullopt, .device = cpu, .replica = 0};
  const int rc = store.wait_replica_ready(key);
  INFO("wait_replica_ready(cpu) rc=" << rc);
  REQUIRE(rc == 0);
  require_replica_registered(store, artifact);
}

TEST_CASE("WorkerLifecycleManager initial full state sync removes drift", "[daemon][ha][sync]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping HA sync test.");
    return;
  }

  // Prepare dummy artifact directories
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test";
  fs::create_directories(temp_root);
  const auto keep = make_standard_artifact(temp_root, "artifact_keep", 1ULL * 1024 * 1024, 'A');
  const auto remove = make_standard_artifact(temp_root, "artifact_remove", 1ULL * 1024 * 1024, 'B');

  // Start fake GS server
  auto test_server = start_fake_server({keep.artifact_id}, /*obsolete_id=*/"");
  REQUIRE(test_server.selected_port > 0);

  // Build engine and load both replicas locally (A+B)
  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, keep);
  load_artifact_gpu(*engine_ptr, remove);
  // Sanity: both present
  {
    auto infos = engine_ptr->get_all_replicas_info();
    bool found_keep = false;
    bool found_remove = false;
    for (const auto& i : infos) {
      if (i.artifact_id == keep.artifact_id)
        found_keep = true;
      if (i.artifact_id == remove.artifact_id)
        found_remove = true;
    }
    REQUIRE(found_keep);
    REQUIRE(found_remove);
  }

  // Start lifecycle manager pointing to fake GS
  tensorcast::daemon::StoreDaemonServiceImpl dummy_service(engine_ptr);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 5000; // long enough to avoid race
  wopts.chunk_sync_interval_ms = 0; // disable chunk sync thread

  WorkerLifecycleManager wlm(
      gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr},
      gsl::not_null<StoreDaemonServiceImpl*>{&dummy_service},
      wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());

  // Initial full-state sync happens in start(); poll until removal applied
  bool removed = false;
  bool kept = false;
  for (int i = 0; i < 200; ++i) { // up to ~2s
    auto infos = engine_ptr->get_all_replicas_info();
    kept = false;
    bool remove_present = false;
    for (const auto& in : infos) {
      if (in.artifact_id == keep.artifact_id) {
        // kept means still has any residency on CPU/GPU
        if (in.cpu_state != tensorcast::common::memory::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::common::memory::MemoryLocation::NONE) {
          kept = true;
        }
      }
      if (in.artifact_id == remove.artifact_id) {
        if (in.cpu_state != tensorcast::common::memory::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::common::memory::MemoryLocation::NONE) {
          remove_present = true;
        }
      }
    }
    removed = !remove_present;
    if (kept && removed)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(kept);
  REQUIRE(removed);

  wlm.stop();
  test_server.server->Shutdown();

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("WorkerLifecycleManager heartbeat applies obsolete removals", "[daemon][ha][heartbeat]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping HA heartbeat test.");
    return;
  }

  // Prepare dummy artifact directories
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_hb";
  fs::create_directories(temp_root);
  const auto keep = make_standard_artifact(temp_root, "artifact_keep2", 1ULL * 1024 * 1024, 'C');
  const auto remove = make_standard_artifact(temp_root, "artifact_remove2", 1ULL * 1024 * 1024, 'D');

  // Start fake GS server: expected replicas include both (so initial full sync keeps both),
  // but heartbeat will advise 'remove_id' as obsolete to trigger removal.
  auto test_server = start_fake_server({keep.artifact_id, remove.artifact_id}, /*obsolete_id=*/remove.artifact_id);
  REQUIRE(test_server.selected_port > 0);

  // Build engine and load both replicas locally (A+B)
  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, keep);
  load_artifact_gpu(*engine_ptr, remove);

  tensorcast::daemon::StoreDaemonServiceImpl dummy_service(engine_ptr);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50; // fast heartbeat to apply obsolete list
  wopts.chunk_sync_interval_ms = 0; // disable chunk sync thread

  WorkerLifecycleManager wlm(
      gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr},
      gsl::not_null<StoreDaemonServiceImpl*>{&dummy_service},
      wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());

  // Poll until heartbeat-applied obsolete removal happens
  bool removed = false;
  bool kept = false;
  for (int i = 0; i < 200; ++i) { // up to ~2s
    auto infos = engine_ptr->get_all_replicas_info();
    kept = false;
    bool remove_present = false;
    for (const auto& in : infos) {
      if (in.artifact_id == keep.artifact_id) {
        if (in.cpu_state != tensorcast::common::memory::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::common::memory::MemoryLocation::NONE) {
          kept = true;
        }
      }
      if (in.artifact_id == remove.artifact_id) {
        if (in.cpu_state != tensorcast::common::memory::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::common::memory::MemoryLocation::NONE) {
          remove_present = true;
        }
      }
    }
    removed = !remove_present;
    if (kept && removed)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(kept);
  REQUIRE(removed);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("state_checksum_is_stable_and_availability_sensitive") {
  std::vector<StoreEngine::ReplicaInfo> infos;
  StoreEngine::ReplicaInfo gpu;
  gpu.artifact_id = "artifact-a";
  gpu.gpu_state = tensorcast::common::memory::MemoryLocation::GPU;
  gpu.cpu_state = tensorcast::common::memory::MemoryLocation::NONE;
  gpu.gpu_device_id = 1;
  gpu.is_registered_for_comm = true;
  infos.push_back(gpu);

  StoreEngine::ReplicaInfo cpu;
  cpu.artifact_id = "artifact-b";
  cpu.gpu_state = tensorcast::common::memory::MemoryLocation::NONE;
  cpu.cpu_state = tensorcast::common::memory::MemoryLocation::CPU;
  cpu.gpu_device_id = -1;
  cpu.is_registered_for_comm = true;
  infos.push_back(cpu);

  const std::string node_id = "node-xyz";
  const auto checksum1 = WorkerLifecycleManager::compute_state_checksum(node_id, infos);
  std::reverse(infos.begin(), infos.end());
  const auto checksum2 = WorkerLifecycleManager::compute_state_checksum(node_id, infos);
  REQUIRE(checksum1 == checksum2);

  infos.front().is_registered_for_comm = false;
  const auto checksum3 = WorkerLifecycleManager::compute_state_checksum(node_id, infos);
  REQUIRE(checksum3 != checksum1);
}

TEST_CASE("WorkerLifecycleManager applies REMOVE via SynchronizeWorkerState", "[daemon][ha][delta]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping HA delta test.");
    return;
  }

  // Prepare dummy artifact directories
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_delta";
  fs::create_directories(temp_root);
  const auto keep = make_standard_artifact(temp_root, "artifact_keep3", 1ULL * 1024 * 1024, 'E');
  const auto remove = make_standard_artifact(temp_root, "artifact_remove3", 1ULL * 1024 * 1024, 'F');

  // Start fake GS server: full-state expects both; heartbeat demands a sync and SynchronizeWorkerState returns REMOVE
  // for remove_id
  auto test_server = start_fake_server({keep.artifact_id, remove.artifact_id}, /*obsolete_id=*/"");
  test_server.service->set_heartbeat_sync_required(true, /*expected_ver=*/2);
  test_server.service->set_sync_remove_ids({remove.artifact_id});
  REQUIRE(test_server.selected_port > 0);

  // Build engine and load both replicas locally (A+B)
  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, keep);
  load_artifact_gpu(*engine_ptr, remove);

  tensorcast::daemon::StoreDaemonServiceImpl dummy_service(engine_ptr);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50; // fast heartbeat to drive synchronize call
  wopts.chunk_sync_interval_ms = 0;

  WorkerLifecycleManager wlm(
      gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr},
      gsl::not_null<StoreDaemonServiceImpl*>{&dummy_service},
      wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());

  // Poll until REMOVE applied
  bool removed = false;
  bool kept = false;
  for (int i = 0; i < 200; ++i) {
    auto infos = engine_ptr->get_all_replicas_info();
    kept = false;
    bool remove_present = false;
    for (const auto& in : infos) {
      if (in.artifact_id == keep.artifact_id) {
        if (in.cpu_state != tensorcast::common::memory::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::common::memory::MemoryLocation::NONE)
          kept = true;
      }
      if (in.artifact_id == remove.artifact_id) {
        if (in.cpu_state != tensorcast::common::memory::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::common::memory::MemoryLocation::NONE)
          remove_present = true;
      }
    }
    removed = !remove_present;
    if (kept && removed)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(kept);
  REQUIRE(removed);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("WorkerLifecycleManager falls back to full-state sync on sync failure", "[daemon][ha][fallback]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping HA fallback test.");
    return;
  }

  // Prepare dummy artifact directories
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_fallback";
  fs::create_directories(temp_root);
  const auto keep = make_standard_artifact(temp_root, "artifact_keep4", 1ULL * 1024 * 1024, 'G');
  const auto remove = make_standard_artifact(temp_root, "artifact_remove4", 1ULL * 1024 * 1024, 'H');

  // Start fake GS server: heartbeat demands sync, but SynchronizeWorkerState fails; RequestFullStateSync expects only
  // keep_id
  auto test_server = start_fake_server({keep.artifact_id}, /*obsolete_id=*/"");
  test_server.service->set_heartbeat_sync_required(true, /*expected_ver=*/2);
  test_server.service->set_sync_should_fail(true);
  REQUIRE(test_server.selected_port > 0);

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, keep);
  load_artifact_gpu(*engine_ptr, remove);

  tensorcast::daemon::StoreDaemonServiceImpl dummy_service(engine_ptr);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50;
  wopts.chunk_sync_interval_ms = 0;

  WorkerLifecycleManager wlm(
      gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr},
      gsl::not_null<StoreDaemonServiceImpl*>{&dummy_service},
      wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());

  // Poll until fallback full-sync applied (remove_id removed, keep_id present)
  bool removed = false;
  bool kept = false;
  for (int i = 0; i < 200; ++i) {
    auto infos = engine_ptr->get_all_replicas_info();
    kept = false;
    bool remove_present = false;
    for (const auto& in : infos) {
      if (in.artifact_id == keep.artifact_id) {
        if (in.cpu_state != tensorcast::common::memory::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::common::memory::MemoryLocation::NONE)
          kept = true;
      }
      if (in.artifact_id == remove.artifact_id) {
        if (in.cpu_state != tensorcast::common::memory::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::common::memory::MemoryLocation::NONE)
          remove_present = true;
      }
    }
    removed = !remove_present;
    if (kept && removed)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(kept);
  REQUIRE(removed);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("WorkerLifecycleManager sends batch chunk state updates", "[daemon][ha][chunks]") {
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_chunks";
  fs::create_directories(temp_root);
  const auto artifact = make_standard_artifact(temp_root, "artifact_chunks", 1ULL * 1024 * 1024, 'I');

  // Start fake GS: expect same artifact to keep; no heartbeat removals needed
  auto test_server = start_fake_server({artifact.artifact_id}, /*obsolete_id=*/"");
  test_server.service->reset_chunk_updates();
  REQUIRE(test_server.selected_port > 0);

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, artifact);

  tensorcast::daemon::StoreDaemonServiceImpl dummy_service(engine_ptr);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 200; // slower heartbeat; chunk sync drives updates
  wopts.chunk_sync_interval_ms = 50; // fast chunk sync loop

  WorkerLifecycleManager wlm(
      gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr},
      gsl::not_null<StoreDaemonServiceImpl*>{&dummy_service},
      wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());

  // Poll for updates
  bool got_updates = false;
  for (int i = 0; i < 200; ++i) {
    if (test_server.service->total_chunk_updates() > 0) {
      got_updates = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(got_updates);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("WorkerLifecycleManager publishes memory tier status and reconciles leases", "[daemon][ha][memory_tier]") {
  fs::path temp_root = fs::temp_directory_path() / "wlm_memory_tier_test";
  fs::create_directories(temp_root);
  const auto artifact = make_standard_artifact(temp_root, "memory_tier_artifact_daemon", 2ULL * 1024 * 1024, 'J');

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL * 1024 * 1024;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  tensorcast::store::MemoryTierConfig tiers;
  tiers.enable_preemptible_memory = false;
  tiers.stable_bytes = 8ULL * 1024 * 1024;
  tiers.preemptible_limit_bytes = 0;
  tiers.preemptible_low_watermark_ratio = 0.4;
  opts.memory_tier_config = tiers;

  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_cpu(*engine_ptr, artifact);
  auto infos = engine_ptr->get_all_replicas_info();
  for (const auto& info : infos) {
    INFO(
        "replica_info artifact_id=" << info.artifact_id << " cpu_state=" << static_cast<int>(info.cpu_state)
                                    << " gpu_state=" << static_cast<int>(info.gpu_state));
  }
  bool found_cpu = false;
  for (const auto& info : infos) {
    if (info.artifact_id == artifact.artifact_id &&
        info.cpu_state != tensorcast::common::memory::MemoryLocation::NONE) {
      found_cpu = true;
      break;
    }
  }
  REQUIRE(found_cpu);

  const DeviceKey cpu_dev{DeviceType::CPU, -1, ""};
  const auto cpu_replicas = engine_ptr->list_device_replicas(cpu_dev);
  std::optional<std::string> canonical_id_opt;
  for (const auto& key : cpu_replicas) {
    if (key.artifact_id == artifact.artifact_id) {
      canonical_id_opt = key.artifact_id;
      break;
    }
  }
  REQUIRE(canonical_id_opt.has_value());
  const std::string canonical_id = *canonical_id_opt;

  tensorcast::store::components::MemoryTierLeaseDescriptor lease_req;
  lease_req.artifact_id = canonical_id;
  lease_req.kind = tensorcast::store::components::MemoryTierLeaseKind::kStable;
  lease_req.chunk_start = 0;
  lease_req.chunk_count = 0;
  auto acquired_or = engine_ptr->acquire_memory_tier_lease(lease_req);
  INFO(acquired_or.status().ToString());
  tensorcast::store::components::MemoryTierLeaseDescriptor acquired = lease_req;
  if (acquired_or.ok()) {
    acquired = *acquired_or;
  } else {
    uint64_t artifact_bytes = 0;
    for (const auto& info : infos) {
      if (info.artifact_id == canonical_id) {
        artifact_bytes = info.size_bytes;
        break;
      }
    }
    acquired.chunk_ids = {0};
    acquired.chunk_count = 1;
    acquired.chunk_start = 0;
    acquired.bytes = artifact_bytes;
    acquired.ledger_version = 1;
    WARN("acquire_memory_tier_lease fallback: " << acquired_or.status());
  }

  auto test_server = start_fake_server({canonical_id}, /*obsolete_id=*/"");
  REQUIRE(test_server.selected_port > 0);

  memory_tier::MemoryTierLease pending;
  pending.set_lease_id("lease-pending");
  pending.set_kind(memory_tier::LEASE_KIND_STABLE);
  pending.set_artifact_id(canonical_id);
  pending.mutable_chunk_range()->set_start(acquired.chunk_ids.empty() ? 0 : acquired.chunk_ids.front());
  pending.mutable_chunk_range()->set_count(acquired.chunk_ids.size());
  pending.clear_chunk_ids();
  for (auto id : acquired.chunk_ids) {
    pending.add_chunk_ids(id);
  }
  pending.set_ledger_version(acquired.ledger_version);
  pending.set_bytes(acquired.bytes);
  pending.set_state(memory_tier::LEASE_STATE_PENDING);
  pending.set_request_id("req-pending");

  memory_tier::MemoryTierLease revoking = pending;
  revoking.set_lease_id("lease-revoking");
  revoking.set_state(memory_tier::LEASE_STATE_REVOKING);
  revoking.clear_chunk_ids();
  revoking.add_chunk_ids(0);

  test_server.service->set_outstanding_leases({pending, revoking});

  tensorcast::daemon::StoreDaemonServiceImpl dummy_service(engine_ptr);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50;
  wopts.chunk_sync_interval_ms = 0;

  WorkerLifecycleManager wlm(
      gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr},
      gsl::not_null<StoreDaemonServiceImpl*>{&dummy_service},
      wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());

  bool saw_acquired = false;
  bool saw_released = false;
  for (int i = 0; i < 200; ++i) {
    auto acks = test_server.service->ack_requests();
    for (const auto& ack : acks) {
      if (ack.lease_id() == pending.lease_id() && ack.action() == memory_tier::LEASE_ACK_ACTION_ACQUIRED &&
          ack.ack_epoch_ns() > 0 && !ack.chunk_ids().empty()) {
        saw_acquired = true;
      }
      if (ack.lease_id() == revoking.lease_id() && ack.action() == memory_tier::LEASE_ACK_ACTION_RELEASED &&
          ack.ack_epoch_ns() > 0) {
        saw_released = true;
      }
    }
    if (saw_acquired && saw_released)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(saw_acquired);
  REQUIRE(saw_released);

  bool pending_active = false;
  bool revoking_expired = false;
  for (const auto& lease : test_server.service->leases_snapshot()) {
    if (lease.lease_id() == pending.lease_id()) {
      pending_active = lease.state() == memory_tier::LEASE_STATE_ACTIVE;
    } else if (lease.lease_id() == revoking.lease_id()) {
      revoking_expired = lease.state() == memory_tier::LEASE_STATE_EXPIRED;
    }
  }
  REQUIRE(pending_active);
  REQUIRE(revoking_expired);

  bool got_status = false;
  memory_tier::MemoryTierStatus latest_status;
  for (int i = 0; i < 200; ++i) {
    auto statuses = test_server.service->memory_tier_statuses();
    if (!statuses.empty()) {
      latest_status = statuses.back();
      got_status = true;
      if (latest_status.worker_id() == "worker-1")
        break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(got_status);
  REQUIRE(latest_status.worker_id() == "worker-1");
  REQUIRE_FALSE(latest_status.node_id().empty());
  REQUIRE(latest_status.stable_total_bytes() == tiers.stable_bytes);
  CHECK(latest_status.enable_preemptible() == tiers.enable_preemptible_memory);
  CHECK(latest_status.preemptible_total_bytes() == 0);
  CHECK_FALSE(latest_status.memory_tier_config_json().empty());

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("WorkerLifecycleManager syncs on version mismatch without sync flag", "[daemon][ha][version_mismatch]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping HA version mismatch test.");
    return;
  }

  // Prepare dummy artifact directories
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_ver_mismatch";
  fs::create_directories(temp_root);
  const auto keep = make_standard_artifact(temp_root, "artifact_keep5", 1ULL * 1024 * 1024, 'K');
  const auto remove = make_standard_artifact(temp_root, "artifact_remove5", 1ULL * 1024 * 1024, 'L');

  // Start fake GS server: full-state expects both; heartbeat indicates expected_version=2 (mismatch) but
  // sync_required=false.
  auto test_server = start_fake_server({keep.artifact_id, remove.artifact_id}, /*obsolete_id=*/"");
  test_server.service->set_heartbeat_sync_required(false, /*expected_ver=*/2);
  test_server.service->set_sync_remove_ids({remove.artifact_id});
  REQUIRE(test_server.selected_port > 0);

  // Build engine and load both replicas locally
  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, keep);
  load_artifact_gpu(*engine_ptr, remove);

  tensorcast::daemon::StoreDaemonServiceImpl dummy_service(engine_ptr);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50; // drive heartbeat quickly
  wopts.chunk_sync_interval_ms = 0; // disable chunk sync

  WorkerLifecycleManager wlm(
      gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr},
      gsl::not_null<StoreDaemonServiceImpl*>{&dummy_service},
      wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());

  // Poll until version-only hint triggers synchronize and removal applied
  bool removed = false;
  bool kept = false;
  for (int i = 0; i < 200; ++i) {
    auto infos = engine_ptr->get_all_replicas_info();
    kept = false;
    bool remove_present = false;
    for (const auto& in : infos) {
      if (in.artifact_id == keep.artifact_id) {
        if (in.cpu_state != tensorcast::common::memory::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::common::memory::MemoryLocation::NONE)
          kept = true;
      }
      if (in.artifact_id == remove.artifact_id) {
        if (in.cpu_state != tensorcast::common::memory::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::common::memory::MemoryLocation::NONE)
          remove_present = true;
      }
    }
    removed = !remove_present;
    if (kept && removed)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(kept);
  REQUIRE(removed);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}
