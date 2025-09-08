// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <future>
#include <memory>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "core/store/store_engine.h"
#include "daemon/ref_tracker.h"
#include "daemon/replica_session_manager.h"
#include "daemon/transport_lock_manager.h"
#include "grpcpp/grpcpp.h"
#include "tensorcast/daemon/v1/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class StoreDaemonServiceImpl final : public v1::StoreDaemonService::Service {
 public:
  struct Options {
    // Sweep/TTL configuration
    std::chrono::seconds sessions_ttl{std::chrono::seconds(60)};
    std::chrono::seconds locks_ttl{std::chrono::seconds(120)};
    std::chrono::milliseconds sessions_sweep_interval{std::chrono::milliseconds(10000)};
    std::chrono::milliseconds locks_sweep_interval{std::chrono::milliseconds(10000)};
    std::chrono::milliseconds verification_sweep_interval{std::chrono::milliseconds(500)};
    std::chrono::milliseconds proc_check_interval{std::chrono::milliseconds(5000)};

    // Eviction policy
    bool enable_periodic_eviction{false};
    double gpu_memory_limit_fraction{0.90};
    std::chrono::milliseconds eviction_check_interval{std::chrono::milliseconds(1000)};

    // Observability
    bool allow_high_card_attrs{false};
  };

  explicit StoreDaemonServiceImpl(std::shared_ptr<tensorcast::store::StoreEngine> engine)
      : StoreDaemonServiceImpl(std::move(engine), Options{}) {}

  explicit StoreDaemonServiceImpl(std::shared_ptr<tensorcast::store::StoreEngine> engine, Options opts)
      : engine_(std::move(engine)), sessions_(opts.sessions_ttl), locks_(opts.locks_ttl), opts_(opts) {
    start_sweepers();
  }

  ~StoreDaemonServiceImpl() override;

  // Initiate graceful shutdown: reject new materialization requests and allow
  // in-flight operations to complete. Called by server_main signal handler.
  void begin_shutdown() {
    is_shutting_down_.store(true);
  }

  bool is_shutting_down() const {
    return is_shutting_down_.load();
  }

  // Worker lifecycle reflection for status RPCs
  void set_worker_registered(std::string worker_id) {
    {
      absl::MutexLock l(&worker_mu_);
      worker_id_ = std::move(worker_id);
    }
    is_registered_.store(true);
  }
  bool is_registered() const {
    return is_registered_.load();
  }
  std::string worker_id() const {
    absl::MutexLock l(&worker_mu_);
    return worker_id_;
  }

  grpc::Status MaterializeReplica(
      grpc::ServerContext* ctx,
      const v1::MaterializeReplicaRequest* req,
      v1::MaterializeReplicaResponse* resp) override;

  grpc::Status ConfirmReplica(
      grpc::ServerContext* ctx,
      const v1::ConfirmReplicaRequest* req,
      v1::ConfirmReplicaResponse* resp) override;

  grpc::Status UnloadReplica(
      grpc::ServerContext* ctx,
      const v1::UnloadReplicaRequest* req,
      v1::UnloadReplicaResponse* resp) override;

  grpc::Status ClearMem(grpc::ServerContext* ctx, const v1::ClearMemRequest* req, v1::ClearMemResponse* resp) override;

  grpc::Status GetServerConfig(
      grpc::ServerContext* ctx,
      const v1::GetServerConfigRequest* req,
      v1::GetServerConfigResponse* resp) override;

  grpc::Status WaitReplicaVerification(
      grpc::ServerContext* ctx,
      const v1::WaitReplicaVerificationRequest* req,
      v1::WaitReplicaVerificationResponse* resp) override;

  grpc::Status LockTransportChunks(
      grpc::ServerContext* ctx,
      const v1::LockTransportChunksRequest* req,
      v1::LockTransportChunksResponse* resp) override;

  grpc::Status UnlockTransportChunks(
      grpc::ServerContext* ctx,
      const v1::UnlockTransportChunksRequest* req,
      v1::UnlockTransportChunksResponse* resp) override;

  grpc::Status BeginRegisterArtifact(
      grpc::ServerContext* ctx,
      const v1::BeginRegisterArtifactRequest* req,
      v1::BeginRegisterArtifactResponse* resp) override;

  grpc::Status CommitRegisteredArtifact(
      grpc::ServerContext* ctx,
      const v1::CommitRegisteredArtifactRequest* req,
      v1::CommitRegisteredArtifactResponse* resp) override;

  grpc::Status AbortRegisteredArtifact(
      grpc::ServerContext* ctx,
      const v1::AbortRegisteredArtifactRequest* req,
      v1::AbortRegisteredArtifactResponse* resp) override;

  grpc::Status FeedRegisterArtifactStream(
      grpc::ServerContext* ctx,
      ::grpc::ServerReader<v1::FeedRegisterArtifactStreamRequest>* reader,
      v1::FeedRegisterArtifactStreamResponse* resp) override;

  // Testing/helper overload: process a vector of streaming requests without standing up a gRPC server
  grpc::Status feed_register_artifact_stream_vector(const std::vector<v1::FeedRegisterArtifactStreamRequest>& reqs);

  grpc::Status KeepAliveRegisterArtifact(
      grpc::ServerContext* ctx,
      const v1::KeepAliveRegisterArtifactRequest* req,
      v1::KeepAliveRegisterArtifactResponse* resp) override;

  grpc::Status RevokeRegisteredArtifact(
      grpc::ServerContext* ctx,
      const v1::RevokeRegisteredArtifactRequest* req,
      v1::RevokeRegisteredArtifactResponse* resp) override;

  // RFC-0014
  grpc::Status MaterializeByKey(
      grpc::ServerContext* ctx,
      const v1::MaterializeByKeyRequest* req,
      v1::MaterializeByKeyResponse* resp) override;
  grpc::Status PublishReplicaKey(
      grpc::ServerContext* ctx,
      const v1::PublishReplicaKeyRequest* req,
      v1::PublishReplicaKeyResponse* resp) override;

  grpc::Status ResolveKeyMapping(
      grpc::ServerContext* ctx,
      const v1::ResolveKeyMappingRequest* req,
      v1::ResolveKeyMappingResponse* resp) override;

  grpc::Status GetArtifactIndexById(
      grpc::ServerContext* ctx,
      const v1::GetArtifactIndexByIdRequest* req,
      v1::GetArtifactIndexByIdResponse* resp) override;

  // Status & listing RPCs
  grpc::Status GetWorkerStatus(
      grpc::ServerContext* ctx,
      const v1::GetWorkerStatusRequest* req,
      v1::GetWorkerStatusResponse* resp) override;

  grpc::Status GetDetailedStatus(
      grpc::ServerContext* ctx,
      const v1::GetDetailedStatusRequest* req,
      v1::GetDetailedStatusResponse* resp) override;

  // Legacy GetLoadedReplicas removed; use V2

  grpc::Status GetLoadedReplicasV2(
      grpc::ServerContext* ctx,
      const v1::GetLoadedReplicasV2Request* req,
      v1::GetLoadedReplicasV2Response* resp) override;

  // Expose current ref-count for a given replica key (for HA state reporting)
  size_t ref_count_for(const store::loading::ReplicaKey& key) const {
    return refs_.ref_count(key);
  }

 private:
  std::shared_ptr<tensorcast::store::StoreEngine> engine_;
  ReplicaSessionManager sessions_;
  TransportLockManager locks_;
  RefTracker refs_;

  // Background sweepers
  std::atomic<bool> stop_{false};
  std::thread sweep_sessions_th_;
  std::thread sweep_locks_th_;
  std::thread pid_watcher_th_;
  std::thread verif_sweeper_th_;
  std::thread eviction_th_;
  std::chrono::time_point<std::chrono::steady_clock> start_time_{std::chrono::steady_clock::now()};
  void start_sweepers();
  void stop_sweepers();

  // Helpers
  static tensorcast::store::DeviceKey resolve_device(const v1::MaterializeReplicaRequest& req);
  static tensorcast::store::DeviceKey resolve_device(const v1::ConfirmReplicaRequest& req);
  static tensorcast::store::DeviceKey resolve_device(const v1::UnloadReplicaRequest& req);
  static store::loading::ReplicaKey make_replica_key(const std::string& artifact_id);

  // Shutdown gating
  std::atomic<bool> is_shutting_down_{false};
  // Compatibility configuration removed

  // Worker lifecycle reflection
  std::atomic<bool> is_registered_{false};
  mutable absl::Mutex worker_mu_;
  std::string worker_id_ ABSL_GUARDED_BY(worker_mu_);

  // Lightweight verification registry keyed by replica_uuid. Tracks
  // verification progress decoupled from ready_future state for parity with
  // Python daemon semantics.
  struct VerifEntry {
    v1::VerificationStatus status{v1::VerificationStatus::VERIFICATION_STATUS_IN_PROGRESS};
    std::string err;
  };
  absl::Mutex verif_mu_;
  absl::flat_hash_map<std::string, VerifEntry> verif_ ABSL_GUARDED_BY(verif_mu_);
  void set_verif_status(const std::string& uuid, v1::VerificationStatus st, std::string err = "");

  // Background task queue for verification completion and auto-registration
  struct VerifTask {
    std::string uuid;
    std::shared_future<absl::Status> ready;
  };
  struct AutoRegTask {
    store::loading::ReplicaKey key;
    std::string disk_path;
    std::shared_future<absl::Status> ready;
  };
  absl::Mutex bg_tasks_mu_;
  std::deque<VerifTask> verif_tasks_ ABSL_GUARDED_BY(bg_tasks_mu_);
  std::deque<AutoRegTask> auto_reg_tasks_ ABSL_GUARDED_BY(bg_tasks_mu_);

  // Lightweight registration metadata for unified Begin/Feed/KeepAlive/Commit lifecycle.
  enum class RegPlan : uint8_t { COALESCED = 0, DVMP = 1, LEASE = 2 };
  struct RegMeta {
    RegPlan plan{RegPlan::COALESCED};
    std::chrono::time_point<std::chrono::steady_clock> expiry{};
    // Remember TTL duration so stream frames can refresh expiry without extra RPCs
    uint32_t ttl_ms{0};
    uint64_t epoch{0};
    uint64_t total_size{0};
    int device_id{0};
    int owner_pid{0};
    bool lease_in_place{false};
    std::string index_key_hex; // optional
    std::string index_data; // optional canonical index JSON bytes
  };
  absl::Mutex reg_mu_;
  absl::flat_hash_map<std::string, RegMeta> reg_meta_ ABSL_GUARDED_BY(reg_mu_);
  absl::flat_hash_map<std::string, std::vector<uint8_t>> reg_buffers_ ABSL_GUARDED_BY(reg_mu_);

  struct LeaseSegMeta {
    int device_id{0};
    std::string handle_bytes; // raw cudaIpcMemHandle_t bytes
    uint64_t base_offset{0}; // offset within mapped handle
    uint64_t length{0};
    uint64_t dst_offset{0}; // destination offset in coalesced buffer
  };
  absl::flat_hash_map<std::string, std::vector<LeaseSegMeta>> reg_leases_ ABSL_GUARDED_BY(reg_mu_);

  // LIP Registry (post-Commit leases)
  struct LipLeaseEntry {
    std::string registration_id; // original registration id for keepalive/revoke
    std::string artifact_id;
    int device_id{0};
    int owner_pid{0};
    uint32_t ttl_ms{0};
    std::chrono::time_point<std::chrono::steady_clock> expiry{};
    uint64_t epoch{0};
    uint64_t total_size{0};
    std::string index_data; // canonical JSON (for verification hashing if needed)
    std::vector<LeaseSegMeta> segments; // mapped via cuda IPC when used
    std::string verification_json; // optional stored verification metadata (JSON)
  };
  // Keyed by (artifact_id + "@" + device_id)
  absl::Mutex lip_mu_;
  absl::flat_hash_map<std::string, LipLeaseEntry> lip_by_key_ ABSL_GUARDED_BY(lip_mu_);
  // Map registration_id -> key for quick keepalive/revoke
  absl::flat_hash_map<std::string, std::string> lip_key_by_reg_ ABSL_GUARDED_BY(lip_mu_);

  // LIP P2P export records keyed by transport lock token. These records track
  // temporary CUDA IPC mappings and corresponding registered tensor keys in the
  // CommunicateEngine so we can unregister/cleanup on UnlockTransportChunks.
  struct LipExportRecord {
    std::string artifact_id;
    int device_id{0};
    std::vector<void*> opened_ptrs; // cudaIpcOpenMemHandle() pointers to close
    std::vector<std::string> tensor_keys; // registered keys to unregister
  };
  absl::Mutex lip_export_mu_;
  absl::flat_hash_map<std::string, LipExportRecord> lip_exports_ ABSL_GUARDED_BY(lip_export_mu_);

  // Helper: D2D copy from LIP segments into a new coalesced destination on target device; returns CUDA IPC handle
  // bytes.
  absl::StatusOr<std::vector<uint8_t>> lip_copy_to_new_coalesced_int(
      int target_device_id,
      const std::string& canonical_index_json,
      uint64_t total_size,
      absl::Span<const LeaseSegMeta> segments);

  Options opts_;
};

} // namespace tensorcast::daemon
