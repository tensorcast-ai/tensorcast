// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>

#include "absl/log/log.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/testing/common.h"
#include "daemon/worker_lifecycle_manager.h"
#include "grpcpp/grpcpp.h"
#include "proto/global_store.grpc.pb.h"
#include "proto/global_store.pb.h"

namespace fs = std::filesystem;
using tensorcast::DeviceType;
using tensorcast::daemon::WorkerLifecycleManager;
using tensorcast::store::DeviceKey;
using tensorcast::store::StoreEngine;
using tensorcast::store::StoreEngineOptions;

static DeviceKey make_gpu_key(int ordinal) {
  return DeviceKey{DeviceType::GPU, ordinal, /*uuid=*/""};
}

// Minimal in-process fake Global Store service
class FakeGlobalStoreService final : public ::global_store::GlobalStore::Service {
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
  void reset_chunk_updates() {
    std::lock_guard<std::mutex> l(mu_);
    total_chunk_updates_ = 0;
  }
  uint32_t total_chunk_updates() const {
    std::lock_guard<std::mutex> l(mu_);
    return total_chunk_updates_;
  }

  ::grpc::Status HealthCheck(
      ::grpc::ServerContext*,
      const ::global_store::HealthCheckRequest*,
      ::global_store::HealthCheckResponse* resp) override {
    resp->set_status(::global_store::OK);
    return ::grpc::Status::OK;
  }

  ::grpc::Status RegisterWorker(
      ::grpc::ServerContext*,
      const ::global_store::RegisterWorkerRequest* req,
      ::global_store::RegisterWorkerResponse* resp) override {
    (void)req;
    resp->set_status(::global_store::OK);
    resp->set_worker_id("worker-1");
    resp->set_heartbeat_interval_ms(1000);
    resp->set_state_sync_required(true);
    resp->set_expected_state_version(1);
    return ::grpc::Status::OK;
  }

  ::grpc::Status WorkerHeartbeat(
      ::grpc::ServerContext*,
      const ::global_store::WorkerHeartbeatRequest* req,
      ::global_store::WorkerHeartbeatResponse* resp) override {
    (void)req;
    resp->set_status(::global_store::OK);
    resp->set_state_sync_required(hb_state_sync_required_);
    if (hb_expected_state_version_ > 0)
      resp->set_expected_state_version(hb_expected_state_version_);
    for (const auto& id : hb_obsolete_ids_)
      resp->add_obsolete_replicas(id);
    return ::grpc::Status::OK;
  }

  ::grpc::Status RequestFullStateSync(
      ::grpc::ServerContext*,
      const ::global_store::RequestFullStateSyncRequest* req,
      ::global_store::RequestFullStateSyncResponse* resp) override {
    (void)req;
    resp->set_status(::global_store::OK);
    resp->set_new_state_version(1);
    resp->set_new_state_checksum("v1");
    for (const auto& id : expected_ids_) {
      auto* rep = resp->add_expected_replicas();
      rep->set_artifact_id(id);
      // Minimal MemoryInfo; the daemon only inspects artifact_id in apply_full_state()
      auto* mi = rep->mutable_memory_info();
      mi->set_memory_type(::global_store::GPU);
      mi->set_device_id(0);
      mi->set_memory_size(0);
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status SynchronizeWorkerState(
      ::grpc::ServerContext*,
      const ::global_store::SynchronizeWorkerStateRequest* req,
      ::global_store::SynchronizeWorkerStateResponse* resp) override {
    (void)req;
    if (sync_should_fail_) {
      resp->set_status(::global_store::ERROR);
      return ::grpc::Status::OK;
    }
    resp->set_status(::global_store::OK);
    resp->set_new_state_version(2);
    resp->set_new_state_checksum("v2");
    for (const auto& id : sync_remove_ids_) {
      auto* ch = resp->add_state_changes();
      ch->set_type(::global_store::StateChange::REMOVE_REPLICA);
      ch->mutable_replica_info()->set_artifact_id(id);
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status BatchUpdateChunkStates(
      ::grpc::ServerContext*,
      const ::global_store::BatchUpdateChunkStatesRequest* req,
      ::global_store::BatchUpdateChunkStatesResponse* resp) override {
    std::lock_guard<std::mutex> l(mu_);
    resp->set_status(::global_store::OK);
    resp->set_updates_applied(req->updates_size());
    total_chunk_updates_ += req->updates_size();
    return ::grpc::Status::OK;
  }

  ::grpc::Status UnregisterWorker(
      ::grpc::ServerContext*,
      const ::global_store::UnregisterWorkerRequest* req,
      ::global_store::UnregisterWorkerResponse* resp) override {
    (void)req;
    resp->set_status(::global_store::OK);
    return ::grpc::Status::OK;
  }

 private:
  std::vector<std::string> expected_ids_;
  std::vector<std::string> hb_obsolete_ids_;
  bool hb_state_sync_required_{false};
  uint64_t hb_expected_state_version_{0};
  std::vector<std::string> sync_remove_ids_;
  bool sync_should_fail_{false};
  mutable std::mutex mu_;
  uint32_t total_chunk_updates_{0};
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
  builder.RegisterService(ts.service.get());
  ts.server = builder.BuildAndStart();
  ts.selected_port = port;
  return ts;
}

static StoreEngine make_store(const fs::path& storage_root) {
  StoreEngineOptions opts;
  opts.storage_path = storage_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.chunk_size = 1ULL << 20;
  opts.num_thread = 2;
  // Tests default pinned timeout to 0 (wait indefinitely)
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  return StoreEngine(opts);
}

static void load_artifact_gpu(StoreEngine& store, const std::string& artifact_id) {
  tensorcast::store::MaterializeHints hints;
  hints.disk_path = artifact_id;
  auto handle_or =
      store.materialize_replica(make_gpu_key(0), tensorcast::store::StoreEngine::MaterializeMode::LOAD_ONLY, hints);
  REQUIRE(handle_or.ok());
  auto handle = std::move(handle_or.value());
  REQUIRE(handle.wait_ready(std::chrono::milliseconds(30000)).ok());
}

TEST_CASE("WorkerLifecycleManager initial full state sync removes drift", "[daemon][ha][sync]") {
  if (!tensorcast::tests::is_cuda_available()) {
    WARN("CUDA not available – skipping HA sync test.");
    return;
  }

  const std::string keep_id = "artifact_keep";
  const std::string remove_id = "artifact_remove";

  // Prepare dummy artifact directories
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test";
  fs::create_directories(temp_root);
  fs::create_directories(temp_root / keep_id);
  fs::create_directories(temp_root / remove_id);
  REQUIRE(tensorcast::tests::create_dummy_file(temp_root / keep_id / "tensor.data_0", 1 * 1024 * 1024));
  REQUIRE(tensorcast::tests::create_dummy_file(temp_root / remove_id / "tensor.data_0", 1 * 1024 * 1024));
  REQUIRE(tensorcast::tests::write_rfc0007_descriptor_for_standard_artifact_dir(temp_root / keep_id).ok());
  REQUIRE(tensorcast::tests::write_rfc0007_descriptor_for_standard_artifact_dir(temp_root / remove_id).ok());

  // Start fake GS server
  auto test_server = start_fake_server({keep_id}, /*obsolete_id=*/"");
  REQUIRE(test_server.selected_port > 0);

  // Build engine and load both replicas locally (A+B)
  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.chunk_size = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, keep_id);
  load_artifact_gpu(*engine_ptr, remove_id);
  // Sanity: both present
  {
    auto infos = engine_ptr->get_all_replicas_info();
    bool found_keep = false, found_remove = false;
    for (const auto& i : infos) {
      if (i.artifact_id == keep_id)
        found_keep = true;
      if (i.artifact_id == remove_id)
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
  wopts.p2p_port = 0;
  wopts.heartbeat_interval_ms = 5000; // long enough to avoid race
  wopts.chunk_sync_interval_ms = 0; // disable chunk sync thread

  WorkerLifecycleManager wlm(engine_ptr, &dummy_service, wopts);
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
      if (in.artifact_id == keep_id) {
        // kept means still has any residency on CPU/GPU
        if (in.cpu_state != tensorcast::store::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::store::MemoryLocation::NONE) {
          kept = true;
        }
      }
      if (in.artifact_id == remove_id) {
        if (in.cpu_state != tensorcast::store::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::store::MemoryLocation::NONE) {
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
  if (!tensorcast::tests::is_cuda_available()) {
    WARN("CUDA not available – skipping HA heartbeat test.");
    return;
  }

  const std::string keep_id = "artifact_keep2";
  const std::string remove_id = "artifact_remove2";

  // Prepare dummy artifact directories
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_hb";
  fs::create_directories(temp_root);
  fs::create_directories(temp_root / keep_id);
  fs::create_directories(temp_root / remove_id);
  REQUIRE(tensorcast::tests::create_dummy_file(temp_root / keep_id / "tensor.data_0", 1 * 1024 * 1024));
  REQUIRE(tensorcast::tests::create_dummy_file(temp_root / remove_id / "tensor.data_0", 1 * 1024 * 1024));
  REQUIRE(tensorcast::tests::write_rfc0007_descriptor_for_standard_artifact_dir(temp_root / keep_id).ok());
  REQUIRE(tensorcast::tests::write_rfc0007_descriptor_for_standard_artifact_dir(temp_root / remove_id).ok());

  // Start fake GS server: expected replicas include both (so initial full sync keeps both),
  // but heartbeat will advise 'remove_id' as obsolete to trigger removal.
  auto test_server = start_fake_server({keep_id, remove_id}, /*obsolete_id=*/remove_id);
  REQUIRE(test_server.selected_port > 0);

  // Build engine and load both replicas locally (A+B)
  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.chunk_size = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, keep_id);
  load_artifact_gpu(*engine_ptr, remove_id);

  tensorcast::daemon::StoreDaemonServiceImpl dummy_service(engine_ptr);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = 0;
  wopts.heartbeat_interval_ms = 50; // fast heartbeat to apply obsolete list
  wopts.chunk_sync_interval_ms = 0; // disable chunk sync thread

  WorkerLifecycleManager wlm(engine_ptr, &dummy_service, wopts);
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
      if (in.artifact_id == keep_id) {
        if (in.cpu_state != tensorcast::store::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::store::MemoryLocation::NONE) {
          kept = true;
        }
      }
      if (in.artifact_id == remove_id) {
        if (in.cpu_state != tensorcast::store::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::store::MemoryLocation::NONE) {
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

TEST_CASE("WorkerLifecycleManager applies REMOVE via SynchronizeWorkerState", "[daemon][ha][delta]") {
  if (!tensorcast::tests::is_cuda_available()) {
    WARN("CUDA not available – skipping HA delta test.");
    return;
  }

  const std::string keep_id = "artifact_keep3";
  const std::string remove_id = "artifact_remove3";

  // Prepare dummy artifact directories
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_delta";
  fs::create_directories(temp_root);
  for (const auto& id : {keep_id, remove_id}) {
    fs::create_directories(temp_root / id);
    REQUIRE(tensorcast::tests::create_dummy_file(temp_root / id / "tensor.data_0", 1 * 1024 * 1024));
    REQUIRE(tensorcast::tests::write_rfc0007_descriptor_for_standard_artifact_dir(temp_root / id).ok());
  }

  // Start fake GS server: full-state expects both; heartbeat demands a sync and SynchronizeWorkerState returns REMOVE
  // for remove_id
  auto test_server = start_fake_server({keep_id, remove_id}, /*obsolete_id=*/"");
  test_server.service->set_heartbeat_sync_required(true, /*expected_ver=*/2);
  test_server.service->set_sync_remove_ids({remove_id});
  REQUIRE(test_server.selected_port > 0);

  // Build engine and load both replicas locally (A+B)
  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.chunk_size = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, keep_id);
  load_artifact_gpu(*engine_ptr, remove_id);

  tensorcast::daemon::StoreDaemonServiceImpl dummy_service(engine_ptr);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = 0;
  wopts.heartbeat_interval_ms = 50; // fast heartbeat to drive synchronize call
  wopts.chunk_sync_interval_ms = 0;

  WorkerLifecycleManager wlm(engine_ptr, &dummy_service, wopts);
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
      if (in.artifact_id == keep_id) {
        if (in.cpu_state != tensorcast::store::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::store::MemoryLocation::NONE)
          kept = true;
      }
      if (in.artifact_id == remove_id) {
        if (in.cpu_state != tensorcast::store::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::store::MemoryLocation::NONE)
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
  if (!tensorcast::tests::is_cuda_available()) {
    WARN("CUDA not available – skipping HA fallback test.");
    return;
  }

  const std::string keep_id = "artifact_keep4";
  const std::string remove_id = "artifact_remove4";

  // Prepare dummy artifact directories
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_fallback";
  fs::create_directories(temp_root);
  for (const auto& id : {keep_id, remove_id}) {
    fs::create_directories(temp_root / id);
    REQUIRE(tensorcast::tests::create_dummy_file(temp_root / id / "tensor.data_0", 1 * 1024 * 1024));
    REQUIRE(tensorcast::tests::write_rfc0007_descriptor_for_standard_artifact_dir(temp_root / id).ok());
  }

  // Start fake GS server: heartbeat demands sync, but SynchronizeWorkerState fails; RequestFullStateSync expects only
  // keep_id
  auto test_server = start_fake_server({keep_id}, /*obsolete_id=*/"");
  test_server.service->set_heartbeat_sync_required(true, /*expected_ver=*/2);
  test_server.service->set_sync_should_fail(true);
  REQUIRE(test_server.selected_port > 0);

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.chunk_size = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, keep_id);
  load_artifact_gpu(*engine_ptr, remove_id);

  tensorcast::daemon::StoreDaemonServiceImpl dummy_service(engine_ptr);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = 0;
  wopts.heartbeat_interval_ms = 50;
  wopts.chunk_sync_interval_ms = 0;

  WorkerLifecycleManager wlm(engine_ptr, &dummy_service, wopts);
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
      if (in.artifact_id == keep_id) {
        if (in.cpu_state != tensorcast::store::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::store::MemoryLocation::NONE)
          kept = true;
      }
      if (in.artifact_id == remove_id) {
        if (in.cpu_state != tensorcast::store::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::store::MemoryLocation::NONE)
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
  const std::string art_id = "artifact_chunks";
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_chunks";
  fs::create_directories(temp_root);
  fs::create_directories(temp_root / art_id);
  REQUIRE(tensorcast::tests::create_dummy_file(temp_root / art_id / "tensor.data_0", 1 * 1024 * 1024));
  REQUIRE(tensorcast::tests::write_rfc0007_descriptor_for_standard_artifact_dir(temp_root / art_id).ok());

  // Start fake GS: expect same artifact to keep; no heartbeat removals needed
  auto test_server = start_fake_server({art_id}, /*obsolete_id=*/"");
  test_server.service->reset_chunk_updates();
  REQUIRE(test_server.selected_port > 0);

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.chunk_size = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, art_id);

  tensorcast::daemon::StoreDaemonServiceImpl dummy_service(engine_ptr);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = 0;
  wopts.heartbeat_interval_ms = 200; // slower heartbeat; chunk sync drives updates
  wopts.chunk_sync_interval_ms = 50; // fast chunk sync loop

  WorkerLifecycleManager wlm(engine_ptr, &dummy_service, wopts);
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

TEST_CASE("WorkerLifecycleManager syncs on version mismatch without sync flag", "[daemon][ha][version_mismatch]") {
  if (!tensorcast::tests::is_cuda_available()) {
    WARN("CUDA not available – skipping HA version mismatch test.");
    return;
  }

  const std::string keep_id = "artifact_keep5";
  const std::string remove_id = "artifact_remove5";

  // Prepare dummy artifact directories
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_ver_mismatch";
  fs::create_directories(temp_root);
  for (const auto& id : {keep_id, remove_id}) {
    fs::create_directories(temp_root / id);
    REQUIRE(tensorcast::tests::create_dummy_file(temp_root / id / "tensor.data_0", 1 * 1024 * 1024));
    REQUIRE(tensorcast::tests::write_rfc0007_descriptor_for_standard_artifact_dir(temp_root / id).ok());
  }

  // Start fake GS server: full-state expects both; heartbeat indicates expected_version=2 (mismatch) but
  // sync_required=false.
  auto test_server = start_fake_server({keep_id, remove_id}, /*obsolete_id=*/"");
  test_server.service->set_heartbeat_sync_required(false, /*expected_ver=*/2);
  test_server.service->set_sync_remove_ids({remove_id});
  REQUIRE(test_server.selected_port > 0);

  // Build engine and load both replicas locally
  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.chunk_size = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, keep_id);
  load_artifact_gpu(*engine_ptr, remove_id);

  tensorcast::daemon::StoreDaemonServiceImpl dummy_service(engine_ptr);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = 0;
  wopts.heartbeat_interval_ms = 50; // drive heartbeat quickly
  wopts.chunk_sync_interval_ms = 0; // disable chunk sync

  WorkerLifecycleManager wlm(engine_ptr, &dummy_service, wopts);
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
      if (in.artifact_id == keep_id) {
        if (in.cpu_state != tensorcast::store::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::store::MemoryLocation::NONE)
          kept = true;
      }
      if (in.artifact_id == remove_id) {
        if (in.cpu_state != tensorcast::store::MemoryLocation::NONE ||
            in.gpu_state != tensorcast::store::MemoryLocation::NONE)
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
