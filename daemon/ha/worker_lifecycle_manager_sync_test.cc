// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/testing/common.h"
#include "daemon/ha/worker_lifecycle_manager.h"
#include "daemon/ha/worker_lifecycle_ports.h"
#include "daemon/state/daemon_kernel.h"
#include "daemon/state/daemon_options.h"
#include "grpcpp/grpcpp.h"
#include "gsl/pointers"
#include "nlohmann/json.hpp"
#include "tensorcast/global_store/v1/global_store.grpc.pb.h"
#include "tensorcast/global_store/v1/global_store.pb.h"
#include "tensorcast/memory_tier/v1/memory_tier.grpc.pb.h"

namespace fs = std::filesystem;
using tensorcast::DeviceType;
using tensorcast::daemon::DaemonKernel;
using tensorcast::daemon::WorkerLifecycleManager;
using tensorcast::daemon::WorkerLifecyclePorts;
using tensorcast::store::DeviceKey;
using tensorcast::store::DeviceRegistry;
using tensorcast::store::StoreEngine;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::loading::ReplicaKey;
namespace commonpb = tensorcast::common::v1;
namespace global_store = tensorcast::global_store::v1;
namespace memory_tier = tensorcast::memory_tier::v1;

namespace {

constexpr uint16_t kTestP2PPort = 45000;

} // namespace

static DeviceKey make_gpu_key(int ordinal) {
  return DeviceRegistry::instance().gpu_key(ordinal);
}

static std::unique_ptr<DaemonKernel> make_kernel(
    const std::shared_ptr<StoreEngine>& engine,
    const fs::path& storage_root) {
  tensorcast::daemon::DaemonOptions opts;
  opts.storage_path = storage_root;
  opts.persistence_log_path = storage_root / "persistence.log";
  opts.daemon_id = "daemon-test";
  return std::make_unique<DaemonKernel>(engine, nullptr, opts);
}

struct HaFixture {
  std::shared_ptr<StoreEngine> engine;
  std::unique_ptr<DaemonKernel> kernel;
  WorkerLifecyclePorts ports;

  HaFixture(std::shared_ptr<StoreEngine> engine_in, const fs::path& storage_root)
      : engine(std::move(engine_in)),
        kernel(make_kernel(engine, storage_root)),
        ports{
            kernel->worker_identity_store(),
            kernel->retire_gates(),
            kernel->shutdown_signal(),
            kernel->async_runtime()} {}
};

struct ArtifactFixture {
  std::string logical_name;
  std::string artifact_id;
  fs::path dir;
};

struct RemoveSpec {
  std::string artifact_id;
  commonpb::MemoryType memory_type{commonpb::MEMORY_TYPE_GPU};
  uint32_t device_id{0};
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

static commonpb::ReplicaInfo make_expected_replica(
    std::string_view artifact_id,
    commonpb::MemoryType memory_type,
    uint32_t device_id) {
  commonpb::ReplicaInfo info;
  info.mutable_ref()->set_artifact_id(std::string(artifact_id));
  auto* mi = info.mutable_memory_info();
  mi->set_memory_type(memory_type);
  mi->set_device_id(device_id);
  mi->set_memory_size(0);
  return info;
}

// Minimal in-process fake Global Store service
class FakeGlobalStoreService final : public global_store::ClusterRuntimeService::Service,
                                     public global_store::ClusterAdminService::Service,
                                     public memory_tier::MemoryTierService::Service {
 public:
  explicit FakeGlobalStoreService(std::vector<std::string> expected_ids, std::string obsolete_id)
      : expected_ids_(std::move(expected_ids)) {
    if (!obsolete_id.empty())
      hb_obsolete_ids_.push_back(std::move(obsolete_id));
  }

  // Config setters (callable by tests after server start)
  void set_heartbeat_obsolete(std::vector<std::string> ids) {
    absl::MutexLock l(&mu_);
    hb_obsolete_ids_ = std::move(ids);
  }

  void set_heartbeat_sync_required(bool required, uint64_t expected_ver) {
    absl::MutexLock l(&mu_);
    hb_state_sync_required_ = required;
    hb_expected_state_version_ = expected_ver;
  }

  void set_heartbeat_rpc_failure(grpc::StatusCode code, std::string message, int fail_count = -1) {
    absl::MutexLock l(&mu_);
    hb_failure_code_ = code;
    hb_failure_message_ = std::move(message);
    hb_failure_remaining_ = fail_count < 0 ? -1 : std::max(0, fail_count);
  }

  void set_sync_remove_ids(std::vector<std::string> ids) {
    std::vector<RemoveSpec> specs;
    specs.reserve(ids.size());
    for (const auto& id : ids) {
      specs.push_back(RemoveSpec{.artifact_id = id});
    }
    set_sync_remove_specs(std::move(specs));
  }

  void set_sync_remove_specs(std::vector<RemoveSpec> specs) {
    absl::MutexLock l(&mu_);
    sync_remove_specs_ = std::move(specs);
  }

  void set_sync_should_fail(bool v) {
    absl::MutexLock l(&mu_);
    sync_should_fail_ = v;
  }

  void set_expected_replicas(std::vector<commonpb::ReplicaInfo> replicas) {
    absl::MutexLock l(&mu_);
    expected_replicas_ = std::move(replicas);
  }

  void set_reconcile_result_kind(global_store::ReconcileResultKind kind, uint32_t retry_after_ms = 0) {
    absl::MutexLock l(&mu_);
    reconcile_result_kind_ = kind;
    retry_after_ms_ = retry_after_ms;
    retry_later_remaining_ = 0;
  }

  void set_retry_later_then_apply(int retries, uint32_t retry_after_ms) {
    absl::MutexLock l(&mu_);
    retry_later_remaining_ = std::max(0, retries);
    retry_after_ms_ = retry_after_ms;
    if (reconcile_result_kind_ == global_store::RECONCILE_RESULT_KIND_RETRY_LATER) {
      reconcile_result_kind_ = global_store::RECONCILE_RESULT_KIND_APPLIED;
    }
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

  std::string last_registered_daemon_id() const {
    absl::MutexLock l(&mu_);
    return last_registered_daemon_id_;
  }

  uint32_t register_requests() const {
    absl::MutexLock l(&mu_);
    return register_requests_;
  }

  uint32_t reconcile_requests() const {
    absl::MutexLock l(&mu_);
    return reconcile_requests_;
  }

  std::vector<uint64_t> reconcile_request_sequences() const {
    absl::MutexLock l(&mu_);
    return reconcile_request_sequences_;
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
    {
      absl::MutexLock l(&mu_);
      last_registered_daemon_id_ = req->daemon_id();
      ++register_requests_;
    }
    resp->set_status(global_store::STATUS_OK);
    resp->set_worker_id("worker-1");
    resp->set_heartbeat_interval_ms(1000);
    resp->set_state_sync_required(true);
    resp->set_expected_state_version(1);
    resp->set_reconcile_generation(1);
    return ::grpc::Status::OK;
  }

  ::grpc::Status WorkerHeartbeat(
      ::grpc::ServerContext* /*context*/,
      const global_store::WorkerHeartbeatRequest* req,
      global_store::WorkerHeartbeatResponse* resp) override {
    (void)req;
    grpc::StatusCode failure_code = grpc::StatusCode::OK;
    std::string failure_message;
    bool state_sync_required = false;
    uint64_t expected_state_version = 0;
    std::vector<std::string> obsolete_ids;
    {
      absl::MutexLock l(&mu_);
      if (hb_failure_code_ != grpc::StatusCode::OK && hb_failure_remaining_ != 0) {
        failure_code = hb_failure_code_;
        failure_message = hb_failure_message_;
        if (hb_failure_remaining_ > 0) {
          --hb_failure_remaining_;
        }
      }
      state_sync_required = hb_state_sync_required_;
      expected_state_version = hb_expected_state_version_;
      obsolete_ids = hb_obsolete_ids_;
    }
    if (failure_code != grpc::StatusCode::OK) {
      return ::grpc::Status(failure_code, failure_message);
    }
    resp->set_status(global_store::STATUS_OK);
    resp->set_state_sync_required(state_sync_required);
    if (expected_state_version > 0)
      resp->set_expected_state_version(expected_state_version);
    for (const auto& id : obsolete_ids)
      resp->add_obsolete_replicas(id);
    return ::grpc::Status::OK;
  }

  ::grpc::Status ReconcileWorkerState(
      ::grpc::ServerContext* /*context*/,
      const global_store::ReconcileWorkerStateRequest* req,
      global_store::ReconcileWorkerStateResponse* resp) override {
    std::vector<RemoveSpec> remove_specs;
    std::vector<commonpb::ReplicaInfo> expected_replicas;
    global_store::ReconcileResultKind result_kind;
    uint32_t retry_after_ms = 0;
    {
      absl::MutexLock l(&mu_);
      ++reconcile_requests_;
      reconcile_request_sequences_.push_back(req->request_seq());
      if (sync_should_fail_) {
        return ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "forced reconcile transport failure");
      }
      if (retry_later_remaining_ > 0) {
        --retry_later_remaining_;
        result_kind = global_store::RECONCILE_RESULT_KIND_RETRY_LATER;
        retry_after_ms = retry_after_ms_;
      } else {
        result_kind = reconcile_result_kind_;
        retry_after_ms = retry_after_ms_;
      }
      remove_specs = sync_remove_specs_;
      expected_replicas = expected_replicas_;
    }
    if (expected_replicas.empty()) {
      expected_replicas.reserve(expected_ids_.size());
      for (const auto& id : expected_ids_) {
        commonpb::ReplicaInfo rep;
        rep.mutable_ref()->set_artifact_id(id);
        auto* mi = rep.mutable_memory_info();
        mi->set_memory_type(commonpb::MEMORY_TYPE_GPU);
        mi->set_device_id(0);
        mi->set_memory_size(0);
        expected_replicas.push_back(std::move(rep));
      }
    }
    resp->set_result_kind(result_kind);
    resp->set_new_state_version(2);
    resp->set_new_state_checksum("v2");
    if (result_kind == global_store::RECONCILE_RESULT_KIND_RETRY_LATER) {
      resp->set_retry_after_ms(retry_after_ms > 0 ? retry_after_ms : 25);
      return ::grpc::Status::OK;
    }
    if (result_kind == global_store::RECONCILE_RESULT_KIND_REBASE_REQUIRED) {
      for (const auto& rep : expected_replicas) {
        *resp->add_expected_replicas() = rep;
      }
      return ::grpc::Status::OK;
    }
    if (result_kind == global_store::RECONCILE_RESULT_KIND_FATAL ||
        result_kind == global_store::RECONCILE_RESULT_KIND_NOOP ||
        result_kind == global_store::RECONCILE_RESULT_KIND_IGNORED_STALE ||
        result_kind == global_store::RECONCILE_RESULT_KIND_UNSPECIFIED) {
      return ::grpc::Status::OK;
    }

    auto has_expected_replica =
        [&expected_replicas](std::string_view artifact_id, commonpb::MemoryType memory_type, uint32_t device_id) {
          for (const auto& rep : expected_replicas) {
            if (rep.ref().artifact_id() != artifact_id) {
              continue;
            }
            if (rep.memory_info().memory_type() != memory_type) {
              continue;
            }
            if (rep.memory_info().device_id() != device_id) {
              continue;
            }
            return true;
          }
          return false;
        };
    for (const auto& local_replica : req->inventory()) {
      const auto memory_type = local_replica.memory_info().memory_type();
      if (memory_type != commonpb::MEMORY_TYPE_GPU && memory_type != commonpb::MEMORY_TYPE_RAM) {
        continue;
      }
      const uint32_t device_id =
          memory_type == commonpb::MEMORY_TYPE_GPU ? static_cast<uint32_t>(local_replica.memory_info().device_id()) : 0;
      if (has_expected_replica(local_replica.ref().artifact_id(), memory_type, device_id)) {
        continue;
      }
      const RemoveSpec spec{
          .artifact_id = local_replica.ref().artifact_id(),
          .memory_type = memory_type,
          .device_id = device_id,
      };
      const bool already_present = std::any_of(remove_specs.begin(), remove_specs.end(), [&](const RemoveSpec& entry) {
        return entry.artifact_id == spec.artifact_id && entry.memory_type == spec.memory_type &&
            entry.device_id == spec.device_id;
      });
      if (!already_present) {
        remove_specs.push_back(spec);
      }
    }

    for (const auto& spec : remove_specs) {
      auto* ch = resp->add_state_changes();
      ch->set_type(global_store::StateChange::CHANGE_TYPE_REMOVE_REPLICA);
      auto* rep = ch->mutable_replica_info();
      rep->mutable_ref()->set_artifact_id(spec.artifact_id);
      auto* mi = rep->mutable_memory_info();
      mi->set_memory_type(spec.memory_type);
      mi->set_device_id(spec.device_id);
      mi->set_memory_size(0);
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
  mutable absl::Mutex mu_;
  std::vector<std::string> hb_obsolete_ids_ ABSL_GUARDED_BY(mu_);
  bool hb_state_sync_required_ ABSL_GUARDED_BY(mu_){false};
  uint64_t hb_expected_state_version_ ABSL_GUARDED_BY(mu_){0};
  grpc::StatusCode hb_failure_code_ ABSL_GUARDED_BY(mu_){grpc::StatusCode::OK};
  std::string hb_failure_message_ ABSL_GUARDED_BY(mu_);
  int hb_failure_remaining_ ABSL_GUARDED_BY(mu_){0};
  std::vector<RemoveSpec> sync_remove_specs_ ABSL_GUARDED_BY(mu_);
  bool sync_should_fail_ ABSL_GUARDED_BY(mu_){false};
  global_store::ReconcileResultKind reconcile_result_kind_ ABSL_GUARDED_BY(mu_){
      global_store::RECONCILE_RESULT_KIND_APPLIED};
  uint32_t retry_after_ms_ ABSL_GUARDED_BY(mu_){0};
  int retry_later_remaining_ ABSL_GUARDED_BY(mu_){0};
  uint32_t register_requests_ ABSL_GUARDED_BY(mu_){0};
  uint32_t reconcile_requests_ ABSL_GUARDED_BY(mu_){0};
  std::vector<uint64_t> reconcile_request_sequences_ ABSL_GUARDED_BY(mu_);
  std::vector<commonpb::ReplicaInfo> expected_replicas_ ABSL_GUARDED_BY(mu_);
  uint32_t total_chunk_updates_ ABSL_GUARDED_BY(mu_){0};
  std::string last_registered_daemon_id_ ABSL_GUARDED_BY(mu_);
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
  builder.RegisterService(static_cast<global_store::ClusterRuntimeService::Service*>(ts.service.get()));
  builder.RegisterService(static_cast<global_store::ClusterAdminService::Service*>(ts.service.get()));
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

static void load_artifact_gpu(StoreEngine& store, const ArtifactFixture& artifact, bool publish = true) {
  tensorcast::store::loading::MaterializeHints hints;
  hints.artifact_id = artifact.artifact_id;
  tensorcast::store::loading::DiskSource disk_source{.path = artifact.dir, .expected_size = std::nullopt};
  auto handle_or = store.materialize_replica(
      make_gpu_key(0), tensorcast::store::StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);
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
  if (publish) {
    store.set_replica_publish_state(key, StoreEngine::ReplicaPublishState::kPublishPending);
  }
}

static void load_artifact_cpu(StoreEngine& store, const ArtifactFixture& artifact, bool publish = true) {
  tensorcast::store::loading::MaterializeHints hints;
  hints.artifact_id = artifact.artifact_id;
  tensorcast::store::loading::DiskSource disk_source{.path = artifact.dir, .expected_size = std::nullopt};
  DeviceKey cpu{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  auto handle_or =
      store.materialize_replica(cpu, tensorcast::store::StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);
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
  if (publish) {
    store.set_replica_publish_state(key, StoreEngine::ReplicaPublishState::kPublishPending);
  }
}

TEST_CASE("WorkerLifecycleManager bootstrap reconcile removes drift", "[daemon][ha][sync]") {
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
  load_artifact_cpu(*engine_ptr, keep);
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
  HaFixture fixture(engine_ptr, temp_root);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 5000; // long enough to avoid race
  wopts.chunk_sync_interval_ms = 0; // disable chunk sync thread

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());
  REQUIRE(test_server.service->last_registered_daemon_id() == "daemon-test");

  // Bootstrap reconcile happens in start(); poll until removal applied.
  bool keep_gpu_present = false;
  bool keep_cpu_present = false;
  bool remove_present = false;
  for (int i = 0; i < 200; ++i) { // up to ~2s
    keep_gpu_present = false;
    keep_cpu_present = false;
    remove_present = false;
    const auto keep_devices = engine_ptr->get_resident_devices(keep.artifact_id);
    for (const auto& dev : keep_devices) {
      if (dev.type == DeviceType::GPU) {
        keep_gpu_present = true;
      } else if (dev.type == DeviceType::CPU) {
        keep_cpu_present = true;
      }
    }
    remove_present = !engine_ptr->get_resident_devices(remove.artifact_id).empty();
    if (keep_gpu_present && !keep_cpu_present && !remove_present)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(keep_gpu_present);
  REQUIRE_FALSE(keep_cpu_present);
  REQUIRE_FALSE(remove_present);

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

  HaFixture fixture(engine_ptr, temp_root);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50; // fast heartbeat to apply obsolete list
  wopts.chunk_sync_interval_ms = 0; // disable chunk sync thread

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());
  REQUIRE(test_server.service->last_registered_daemon_id() == "daemon-test");

  // Poll to ensure heartbeat obsolete hints do not trigger unloads.
  bool kept = false;
  bool remove_present = false;
  for (int i = 0; i < 200; ++i) { // up to ~2s
    auto infos = engine_ptr->get_all_replicas_info();
    kept = false;
    remove_present = false;
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
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(kept);
  REQUIRE(remove_present);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("state_checksum_is_stable_and_availability_sensitive") {
  std::vector<StoreEngine::ReplicaInventoryEntry> infos;
  StoreEngine::ReplicaInventoryEntry gpu;
  gpu.key = ReplicaKey{
      .artifact_id = "artifact-a",
      .view_id = std::nullopt,
      .device = DeviceRegistry::instance().gpu_key(1),
      .replica = 0};
  gpu.memory_location = tensorcast::common::memory::MemoryLocation::GPU;
  gpu.is_available = true;
  infos.push_back(gpu);

  StoreEngine::ReplicaInventoryEntry cpu;
  cpu.key = ReplicaKey{
      .artifact_id = "artifact-b",
      .view_id = std::nullopt,
      .device = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      .replica = 0};
  cpu.memory_location = tensorcast::common::memory::MemoryLocation::CPU;
  cpu.is_available = true;
  infos.push_back(cpu);

  const std::string node_id = "node-xyz";
  const std::string node_address = "10.0.0.1";
  const uint32_t node_port = 50050;
  const auto checksum1 = WorkerLifecycleManager::compute_state_checksum(node_id, node_address, node_port, infos);
  std::reverse(infos.begin(), infos.end());
  const auto checksum2 = WorkerLifecycleManager::compute_state_checksum(node_id, node_address, node_port, infos);
  REQUIRE(checksum1 == checksum2);

  infos.front().is_available = false;
  const auto checksum3 = WorkerLifecycleManager::compute_state_checksum(node_id, node_address, node_port, infos);
  REQUIRE(checksum3 != checksum1);
}

TEST_CASE("WorkerLifecycleManager applies REMOVE via ReconcileWorkerState", "[daemon][ha][delta]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping HA delta test.");
    return;
  }

  // Prepare dummy artifact directories
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_delta";
  fs::create_directories(temp_root);
  const auto keep = make_standard_artifact(temp_root, "artifact_keep3", 1ULL * 1024 * 1024, 'E');
  const auto remove = make_standard_artifact(temp_root, "artifact_remove3", 1ULL * 1024 * 1024, 'F');

  // Start fake GS server: expected snapshot includes both, but reconcile requests an explicit GPU remove for remove_id.
  auto test_server = start_fake_server({keep.artifact_id, remove.artifact_id}, /*obsolete_id=*/"");
  test_server.service->set_heartbeat_sync_required(true, /*expected_ver=*/2);
  test_server.service->set_expected_replicas(
      {make_expected_replica(keep.artifact_id, commonpb::MEMORY_TYPE_GPU, 0),
       make_expected_replica(remove.artifact_id, commonpb::MEMORY_TYPE_GPU, 0),
       make_expected_replica(remove.artifact_id, commonpb::MEMORY_TYPE_RAM, 0)});
  test_server.service->set_sync_remove_specs(
      {RemoveSpec{.artifact_id = remove.artifact_id, .memory_type = commonpb::MEMORY_TYPE_GPU, .device_id = 0}});
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
  load_artifact_cpu(*engine_ptr, remove);

  HaFixture fixture(engine_ptr, temp_root);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50; // fast heartbeat to drive reconcile calls
  wopts.chunk_sync_interval_ms = 0;

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());
  REQUIRE(test_server.service->last_registered_daemon_id() == "daemon-test");

  // Poll until REMOVE applied
  bool keep_present = false;
  bool remove_gpu_present = false;
  bool remove_cpu_present = false;
  for (int i = 0; i < 200; ++i) {
    keep_present = !engine_ptr->get_resident_devices(keep.artifact_id).empty();
    remove_gpu_present = false;
    remove_cpu_present = false;
    const auto remove_devices = engine_ptr->get_resident_devices(remove.artifact_id);
    for (const auto& dev : remove_devices) {
      if (dev.type == DeviceType::GPU) {
        remove_gpu_present = true;
      } else if (dev.type == DeviceType::CPU) {
        remove_cpu_present = true;
      }
    }
    if (keep_present && remove_cpu_present && !remove_gpu_present)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(keep_present);
  REQUIRE(remove_cpu_present);
  REQUIRE_FALSE(remove_gpu_present);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("WorkerLifecycleManager retire waits for refs", "[daemon][ha][retire]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping HA retire test.");
    return;
  }

  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_retire_ref";
  fs::create_directories(temp_root);
  const auto remove = make_standard_artifact(temp_root, "artifact_remove_ref", 1ULL * 1024 * 1024, 'R');

  auto test_server = start_fake_server({remove.artifact_id}, /*obsolete_id=*/"");
  test_server.service->set_heartbeat_sync_required(true, /*expected_ver=*/2);
  test_server.service->set_sync_remove_specs(
      {RemoveSpec{.artifact_id = remove.artifact_id, .memory_type = commonpb::MEMORY_TYPE_GPU, .device_id = 0}});
  REQUIRE(test_server.selected_port > 0);

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, remove);

  HaFixture fixture(engine_ptr, temp_root);
  ReplicaKey key{.artifact_id = remove.artifact_id, .view_id = std::nullopt, .device = make_gpu_key(0), .replica = 0};
  const int32_t live_pid = static_cast<int32_t>(::getpid());
  fixture.kernel->ref_tracker().add_ref(key, live_pid);

  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50;
  wopts.chunk_sync_interval_ms = 0;

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());
  REQUIRE(test_server.service->last_registered_daemon_id() == "daemon-test");

  bool still_present = false;
  for (int i = 0; i < 200; ++i) {
    still_present = !engine_ptr->get_resident_devices(remove.artifact_id).empty();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(still_present);

  fixture.kernel->ref_tracker().drop_ref(key, live_pid);
  bool removed = false;
  for (int i = 0; i < 200; ++i) {
    removed = engine_ptr->get_resident_devices(remove.artifact_id).empty();
    if (removed)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(removed);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("WorkerLifecycleManager retire waits for transport locks", "[daemon][ha][retire]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping HA retire test.");
    return;
  }

  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_retire_lock";
  fs::create_directories(temp_root);
  const auto remove = make_standard_artifact(temp_root, "artifact_remove_lock", 1ULL * 1024 * 1024, 'L');

  auto test_server = start_fake_server({remove.artifact_id}, /*obsolete_id=*/"");
  test_server.service->set_heartbeat_sync_required(true, /*expected_ver=*/2);
  test_server.service->set_sync_remove_specs(
      {RemoveSpec{.artifact_id = remove.artifact_id, .memory_type = commonpb::MEMORY_TYPE_GPU, .device_id = 0}});
  REQUIRE(test_server.selected_port > 0);

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);
  load_artifact_gpu(*engine_ptr, remove);

  HaFixture fixture(engine_ptr, temp_root);
  ReplicaKey key{.artifact_id = remove.artifact_id, .view_id = std::nullopt, .device = make_gpu_key(0), .replica = 0};
  auto& locks = fixture.kernel->transport_lock_manager();
  const std::string token = locks.mint_token();
  locks.put(token, key, {0});

  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50;
  wopts.chunk_sync_interval_ms = 0;

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());
  REQUIRE(test_server.service->last_registered_daemon_id() == "daemon-test");

  bool still_present = false;
  for (int i = 0; i < 200; ++i) {
    still_present = !engine_ptr->get_resident_devices(remove.artifact_id).empty();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(still_present);

  locks.erase(token);
  bool removed = false;
  for (int i = 0; i < 200; ++i) {
    removed = engine_ptr->get_resident_devices(remove.artifact_id).empty();
    if (removed)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(removed);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("WorkerLifecycleManager applies REBASE_REQUIRED expected snapshot", "[daemon][ha][rebase]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping HA rebase test.");
    return;
  }

  // Prepare dummy artifact directories
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_rebase";
  fs::create_directories(temp_root);
  const auto keep = make_standard_artifact(temp_root, "artifact_keep4", 1ULL * 1024 * 1024, 'G');
  const auto remove = make_standard_artifact(temp_root, "artifact_remove4", 1ULL * 1024 * 1024, 'H');

  // Start fake GS server: reconcile requests a REBASE to expected snapshot containing only keep_id.
  auto test_server = start_fake_server({keep.artifact_id}, /*obsolete_id=*/"");
  test_server.service->set_heartbeat_sync_required(true, /*expected_ver=*/2);
  test_server.service->set_reconcile_result_kind(global_store::RECONCILE_RESULT_KIND_REBASE_REQUIRED);
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

  HaFixture fixture(engine_ptr, temp_root);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50;
  wopts.chunk_sync_interval_ms = 0;

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());
  REQUIRE(test_server.service->last_registered_daemon_id() == "daemon-test");

  // Poll until REBASE expected snapshot is applied (remove_id removed, keep_id present).
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

TEST_CASE("WorkerLifecycleManager retries on RETRY_LATER and eventually applies reconcile", "[daemon][ha][retry]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping HA retry test.");
    return;
  }

  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_retry";
  fs::create_directories(temp_root);
  const auto keep = make_standard_artifact(temp_root, "artifact_keep_retry", 1ULL * 1024 * 1024, 'M');
  const auto remove = make_standard_artifact(temp_root, "artifact_remove_retry", 1ULL * 1024 * 1024, 'N');

  auto test_server = start_fake_server({keep.artifact_id, remove.artifact_id}, /*obsolete_id=*/"");
  test_server.service->set_heartbeat_sync_required(true, /*expected_ver=*/2);
  test_server.service->set_retry_later_then_apply(/*retries=*/2, /*retry_after_ms=*/20);
  test_server.service->set_sync_remove_specs(
      {RemoveSpec{.artifact_id = remove.artifact_id, .memory_type = commonpb::MEMORY_TYPE_GPU, .device_id = 0}});
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

  HaFixture fixture(engine_ptr, temp_root);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50;
  wopts.chunk_sync_interval_ms = 0;

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());
  REQUIRE(test_server.service->last_registered_daemon_id() == "daemon-test");

  bool removed = false;
  bool kept = false;
  for (int i = 0; i < 300; ++i) {
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
    if (kept && removed && test_server.service->reconcile_requests() >= 3) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(kept);
  REQUIRE(removed);
  REQUIRE(test_server.service->reconcile_requests() >= 3);
  const auto reconcile_sequences = test_server.service->reconcile_request_sequences();
  REQUIRE(reconcile_sequences.size() >= 3);
  REQUIRE(reconcile_sequences[0] == 1);
  REQUIRE(reconcile_sequences[1] == 1);
  REQUIRE(reconcile_sequences[2] == 1);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("WorkerLifecycleManager suppresses re-registration on heartbeat conflicts", "[daemon][ha][retry]") {
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_hb_conflict";
  fs::create_directories(temp_root);

  auto test_server = start_fake_server({}, /*obsolete_id=*/"");
  test_server.service->set_heartbeat_rpc_failure(
      grpc::StatusCode::ABORTED,
      "WorkerHeartbeat transaction conflict: TransactionContext Error: Conflict on tuple deletion!",
      /*fail_count=*/8);
  REQUIRE(test_server.selected_port > 0);

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);

  HaFixture fixture(engine_ptr, temp_root);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50;
  wopts.chunk_sync_interval_ms = 0;
  wopts.heartbeat_rpc_timeout_ms = 50;
  wopts.heartbeat_rpc_max_retries = 0;

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());

  for (int i = 0; i < 80; ++i) {
    if (test_server.service->register_requests() > 1) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  REQUIRE(test_server.service->register_requests() == 1);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("WorkerLifecycleManager keeps request_seq stable on transport failures", "[daemon][ha][retry]") {
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_transport_retry";
  fs::create_directories(temp_root);

  auto test_server = start_fake_server({}, /*obsolete_id=*/"");
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

  HaFixture fixture(engine_ptr, temp_root);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50;
  wopts.chunk_sync_interval_ms = 0;
  wopts.state_sync_rpc_timeout_ms = 50;
  wopts.state_sync_rpc_max_retries = 0;

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());

  for (int i = 0; i < 300; ++i) {
    if (test_server.service->reconcile_requests() >= 3) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  const auto reconcile_sequences = test_server.service->reconcile_request_sequences();
  REQUIRE(reconcile_sequences.size() >= 3);
  REQUIRE(reconcile_sequences[0] == 1);
  REQUIRE(reconcile_sequences[1] == 1);
  REQUIRE(reconcile_sequences[2] == 1);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("WorkerLifecycleManager bounds re-registration under repeated identity failures", "[daemon][ha][retry]") {
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_hb_not_found";
  fs::create_directories(temp_root);

  auto test_server = start_fake_server({}, /*obsolete_id=*/"");
  test_server.service->set_heartbeat_rpc_failure(
      grpc::StatusCode::NOT_FOUND,
      "WorkerHeartbeat failed: worker not found",
      /*fail_count=*/-1);
  REQUIRE(test_server.selected_port > 0);

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);

  HaFixture fixture(engine_ptr, temp_root);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50;
  wopts.chunk_sync_interval_ms = 0;
  wopts.heartbeat_rpc_timeout_ms = 50;
  wopts.heartbeat_rpc_max_retries = 0;

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());

  for (int i = 0; i < 200; ++i) {
    if (test_server.service->register_requests() >= 2) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));

  const auto register_requests = test_server.service->register_requests();
  REQUIRE(register_requests >= 2);
  REQUIRE(register_requests <= 7);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("WorkerLifecycleManager backs off reconcile retries on transport failures", "[daemon][ha][retry]") {
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_transport_backoff";
  fs::create_directories(temp_root);

  auto test_server = start_fake_server({}, /*obsolete_id=*/"");
  test_server.service->set_sync_should_fail(true);
  REQUIRE(test_server.selected_port > 0);

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);

  HaFixture fixture(engine_ptr, temp_root);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50;
  wopts.chunk_sync_interval_ms = 0;
  wopts.state_sync_rpc_timeout_ms = 50;
  wopts.state_sync_rpc_max_retries = 0;

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());

  std::this_thread::sleep_for(std::chrono::milliseconds(1800));
  const auto reconcile_requests = test_server.service->reconcile_requests();
  REQUIRE(reconcile_requests >= 2);
  REQUIRE(reconcile_requests <= 8);

  wlm.stop();
  test_server.server->Shutdown();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE(
    "WorkerLifecycleManager tracks outage mode and suppresses duplicate reconcile enqueue",
    "[daemon][ha][retry][metrics]") {
  fs::path temp_root = fs::temp_directory_path() / "wlm_sync_test_outage_metrics";
  fs::create_directories(temp_root);

  auto test_server = start_fake_server({}, /*obsolete_id=*/"");
  test_server.service->set_sync_should_fail(true);
  test_server.service->set_heartbeat_sync_required(true, /*expected_ver=*/1);
  REQUIRE(test_server.selected_port > 0);

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  auto engine_ptr = std::make_shared<StoreEngine>(opts);

  HaFixture fixture(engine_ptr, temp_root);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 30;
  wopts.chunk_sync_interval_ms = 0;
  wopts.state_sync_rpc_timeout_ms = 50;
  wopts.state_sync_rpc_max_retries = 0;

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());

  bool outage_observed = false;
  bool suppress_observed = false;
  for (int i = 0; i < 200; ++i) {
    outage_observed = outage_observed || wlm.state_sync_outage_mode_active();
    suppress_observed = suppress_observed || wlm.state_sync_enqueue_suppressed() > 0;
    if (outage_observed && suppress_observed) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(outage_observed);
  REQUIRE(suppress_observed);

  test_server.service->set_sync_should_fail(false);
  const auto baseline_sync_success = wlm.sync_success();
  bool recovered = false;
  for (int i = 0; i < 200; ++i) {
    if (wlm.sync_success() > baseline_sync_success && !wlm.state_sync_outage_mode_active()) {
      recovered = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(recovered);
  REQUIRE(wlm.last_reconnect_latency_ms() >= 0);

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

  HaFixture fixture(engine_ptr, temp_root);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 200; // slower heartbeat; chunk sync drives updates
  wopts.chunk_sync_interval_ms = 50; // fast chunk sync loop

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());
  REQUIRE(test_server.service->last_registered_daemon_id() == "daemon-test");

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
  load_artifact_cpu(*engine_ptr, artifact, /*publish=*/false);
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

  HaFixture fixture(engine_ptr, temp_root);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50;
  wopts.chunk_sync_interval_ms = 0;

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());
  REQUIRE(test_server.service->last_registered_daemon_id() == "daemon-test");

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

  // Start fake GS server: expected snapshot includes both; heartbeat indicates expected_version=2 (mismatch) while
  // state_sync_required=false.
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

  HaFixture fixture(engine_ptr, temp_root);
  WorkerLifecycleManager::Options wopts;
  wopts.global_store_addr = std::string("127.0.0.1:") + std::to_string(test_server.selected_port);
  wopts.listen_addr = "127.0.0.1:50051";
  wopts.p2p_port = kTestP2PPort;
  wopts.heartbeat_interval_ms = 50; // drive heartbeat quickly
  wopts.chunk_sync_interval_ms = 0; // disable chunk sync

  WorkerLifecycleManager wlm(gsl::not_null<std::shared_ptr<StoreEngine>>{engine_ptr}, fixture.ports, wopts);
  auto st = wlm.start();
  REQUIRE(st.ok());
  REQUIRE(test_server.service->last_registered_daemon_id() == "daemon-test");

  // Poll until version-only hint triggers reconcile and removal applied.
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
