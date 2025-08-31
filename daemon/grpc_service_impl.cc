// Copyright (c) 2025, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <nlohmann/json.hpp>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <optional>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "core/store/device_registry.h"
#include "core/store/device_types.h"
#include "core/store/loading/loading_spec.h"
#include "daemon/status_utils.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

Status StoreDaemonServiceImpl::MaterializeReplica(
    grpc::ServerContext* ctx,
    const ::store_daemon::MaterializeReplicaRequest* req,
    ::store_daemon::MaterializeReplicaResponse* resp) {
  using ::store_daemon::MaterializeReplicaStatus;

  // Reject new materialization while shutting down to align with Python daemon
  if (is_shutting_down_.load()) {
    resp->set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  // Validate one-of inputs: exactly one of artifact_id or disk_path
  const bool has_artifact = req->has_artifact_id() && !req->artifact_id().empty();
  const bool has_disk = req->has_disk_path() && !req->disk_path().empty();
  if (has_artifact == has_disk) {
    return {StatusCode::INVALID_ARGUMENT, "Exactly one of artifact_id or disk_path must be provided"};
  }

  // Build DeviceKey
  const auto device = resolve_device(*req);

  // Build hints
  tensorcast::store::MaterializeHints hints;
  if (req->pinned_allocation_timeout_ms() > 0) {
    hints.pinned_timeout = std::chrono::milliseconds(req->pinned_allocation_timeout_ms());
  }
  if (has_artifact)
    hints.artifact_id = req->artifact_id();
  if (has_disk)
    hints.disk_path = req->disk_path();

  // Choose mode
  tensorcast::store::StoreEngine::MaterializeMode mode = tensorcast::store::StoreEngine::MaterializeMode::AUTO;
  if (has_disk) {
    mode = tensorcast::store::StoreEngine::MaterializeMode::LOAD_ONLY;
  }

  auto result = engine_->materialize_replica(device, mode, hints);
  if (!result.ok()) {
    // Allocation/init failed
    resp->set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(result.status());
  }

  const auto& handle = *result;
  // Store session mapping for Confirm/Unload
  if (!req->replica_uuid().empty()) {
    sessions_.put(req->replica_uuid(), handle.replica_key, handle.ready_future);
    // Initialize verification registry entry and enqueue a background task to
    // update status to PASSED/FAILED after the ready_future resolves.
    set_verif_status(req->replica_uuid(), ::store_daemon::VerificationStatus::VERIFICATION_STATUS_IN_PROGRESS);
    {
      absl::MutexLock l(&bg_tasks_mu_);
      verif_tasks_.push_back(VerifTask{req->replica_uuid(), handle.ready_future});
    }
  }
  // Track initial PID reference and keep_for_global if provided
  if (req->pid() > 0) {
    bool keep = req->keep_for_global();
    refs_.add_ref(handle.replica_key, req->pid(), keep);
  }

  // Populate response
  if (has_disk)
    resp->set_disk_path(req->disk_path());
  resp->set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  if (handle.cuda_ipc_handle.is_valid()) {
    auto* mem = resp->mutable_mem_handle();
    mem->set_cuda_ipc_handle(handle.cuda_ipc_handle.to_string());
  }

  // Optionally enqueue auto-registration of disk loads with Global Store after ready
  if (has_disk && compat_.auto_register_disk_loads) {
    absl::MutexLock l(&bg_tasks_mu_);
    auto_reg_tasks_.push_back(AutoRegTask{handle.replica_key, req->disk_path(), handle.ready_future});
  }
  return Status::OK;
}

Status StoreDaemonServiceImpl::ConfirmReplica(
    grpc::ServerContext* ctx,
    const ::store_daemon::ConfirmReplicaRequest* req,
    ::store_daemon::ConfirmReplicaResponse* resp) {
  resp->set_disk_path(req->disk_path());

  // Compatibility checks when strict confirm mode is enabled
  if (compat_.confirm_requires_disk_path) {
    if (req->disk_path().empty()) {
      return {StatusCode::INVALID_ARGUMENT, "disk_path is required by confirm_requires_disk_path"};
    }
    if (req->target_device_type() != ::store_daemon::DeviceType::DEVICE_TYPE_GPU) {
      return {StatusCode::UNIMPLEMENTED, "confirm on non-GPU path disabled by compatibility flag"};
    }
  }

  if (req->replica_uuid().empty()) {
    resp->set_code(0);
    return Status::OK;
  }

  auto entry = sessions_.get(req->replica_uuid());
  if (!entry.has_value()) {
    // Parity: unknown replica_uuid → code=0 OK
    resp->set_code(0);
    return Status::OK;
  }

  // Compute bounded wait timeout: min(30s, remaining gRPC deadline)
  std::chrono::milliseconds wait_ms(30000);
  const auto deadline = ctx->deadline();
  const auto now = std::chrono::system_clock::now();
  if (deadline != std::chrono::system_clock::time_point::max()) {
    if (deadline <= now) {
      wait_ms = std::chrono::milliseconds(0);
    } else {
      auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
      if (remaining < wait_ms)
        wait_ms = remaining;
    }
  }

  absl::Status st = entry->ready.wait_for(wait_ms) == std::future_status::ready
      ? entry->ready.get()
      : absl::DeadlineExceededError("confirm timeout");
  if (st.ok()) {
    resp->set_code(0);
    return Status::OK;
  }
  resp->set_code(1);
  return to_grpc_status(st);
}

Status StoreDaemonServiceImpl::UnloadReplica(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::UnloadReplicaRequest* req,
    ::store_daemon::UnloadReplicaResponse* resp) {
  resp->set_disk_path(req->disk_path());
  // Python parity: DISK-target unload is a no-op idempotent success
  if (req->target_device_type() == ::store_daemon::DeviceType::DEVICE_TYPE_DISK) {
    resp->set_code(0);
    return Status::OK;
  }
  // Two paths:
  // 1) If replica_uuid present and known -> use session key
  // 2) Else, fall back to disk_path + target device to form a ReplicaKey for idempotent unload
  tensorcast::store::ReplicaKey key;
  bool have_key = false;
  if (!req->replica_uuid().empty()) {
    auto entry = sessions_.get(req->replica_uuid());
    if (entry.has_value()) {
      key = entry->key;
      have_key = true;
    }
  }
  if (!have_key) {
    if (!req->disk_path().empty()) {
      key.artifact_id = req->disk_path();
      key.device = resolve_device(*req);
      key.replica = 0;
      have_key = true;
    } else {
      // Idempotent success when no identification provided
      resp->set_code(0);
      return Status::OK;
    }
  }
  // Drop PID ref if provided; if refs remain, skip unload per parity
  if (req->has_pid()) {
    refs_.drop_ref(key, req->pid());
    if (refs_.ref_count(key) > 0) {
      resp->set_code(0);
      return Status::OK;
    }
  }
  const int rc = engine_->unload_replica(key);
  if (rc == 0) {
    if (!req->replica_uuid().empty()) {
      sessions_.erase(req->replica_uuid());
    }
    resp->set_code(0);
    return Status::OK;
  }
  resp->set_code(1);
  return {StatusCode::INTERNAL, absl::StrFormat("unload_replica() returned %d", rc)};
}

Status StoreDaemonServiceImpl::ClearMem(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::ClearMemRequest* /*req*/,
    ::store_daemon::ClearMemResponse* /*resp*/) {
  const int rc = engine_->clear_mem();
  if (rc == 0)
    return Status::OK;
  return {StatusCode::INTERNAL, absl::StrFormat("clear_mem() returned %d", rc)};
}

Status StoreDaemonServiceImpl::GetServerConfig(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::GetServerConfigRequest* /*req*/,
    ::store_daemon::GetServerConfigResponse* resp) {
  resp->set_mem_pool_size(static_cast<int64_t>(engine_->get_mem_pool_size()));
  resp->set_chunk_size(static_cast<int64_t>(engine_->get_chunk_size()));
  return Status::OK;
}

// Destructor: stop sweepers
StoreDaemonServiceImpl::~StoreDaemonServiceImpl() {
  stop_sweepers();
}

// ──────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────

static tensorcast::store::DeviceKey default_gpu_key() {
  return tensorcast::store::DeviceRegistry::instance().gpu_key(0);
}

tensorcast::store::DeviceKey StoreDaemonServiceImpl::resolve_device(
    const ::store_daemon::MaterializeReplicaRequest& req) {
  using tensorcast::DeviceType;
  using tensorcast::store::DeviceKey;
  if (!req.device_uuid().empty()) {
    DeviceKey key{DeviceType::GPU, 0, req.device_uuid()};
    return tensorcast::store::DeviceRegistry::instance().normalize(key);
  }
  switch (req.target_device_type()) {
    case ::store_daemon::DeviceType::DEVICE_TYPE_CPU:
      return DeviceKey{DeviceType::CPU, -1, ""};
    case ::store_daemon::DeviceType::DEVICE_TYPE_DISK:
      // Treat as ingest-from-disk to default GPU for v1 parity
      return default_gpu_key();
    case ::store_daemon::DeviceType::DEVICE_TYPE_GPU:
    default:
      return default_gpu_key();
  }
}

tensorcast::store::DeviceKey StoreDaemonServiceImpl::resolve_device(const ::store_daemon::ConfirmReplicaRequest& req) {
  using tensorcast::DeviceType;
  using tensorcast::store::DeviceKey;
  switch (req.target_device_type()) {
    case ::store_daemon::DeviceType::DEVICE_TYPE_CPU:
      return DeviceKey{DeviceType::CPU, -1, ""};
    case ::store_daemon::DeviceType::DEVICE_TYPE_DISK:
      return default_gpu_key();
    case ::store_daemon::DeviceType::DEVICE_TYPE_GPU:
    default:
      return default_gpu_key();
  }
}

tensorcast::store::DeviceKey StoreDaemonServiceImpl::resolve_device(const ::store_daemon::UnloadReplicaRequest& req) {
  using tensorcast::DeviceType;
  using tensorcast::store::DeviceKey;
  switch (req.target_device_type()) {
    case ::store_daemon::DeviceType::DEVICE_TYPE_CPU:
      return DeviceKey{DeviceType::CPU, -1, ""};
    case ::store_daemon::DeviceType::DEVICE_TYPE_DISK:
      return default_gpu_key();
    case ::store_daemon::DeviceType::DEVICE_TYPE_GPU:
    default:
      return default_gpu_key();
  }
}

tensorcast::store::ReplicaKey StoreDaemonServiceImpl::make_replica_key(const std::string& artifact_id) {
  tensorcast::store::ReplicaKey key;
  key.artifact_id = artifact_id;
  key.device = default_gpu_key();
  key.replica = 0;
  return key;
}

Status StoreDaemonServiceImpl::WaitReplicaVerification(
    grpc::ServerContext* ctx,
    const ::store_daemon::ReplicaVerificationRequest* req,
    ::store_daemon::ReplicaVerificationResponse* resp) {
  // Consult verification registry (populated by MaterializeReplica) to see
  // if we already have a terminal status. If so, return immediately; otherwise
  // proceed to wait on the readiness future with bounded timeout.
  std::optional<::store_daemon::VerificationStatus> known_status;
  std::string known_err;
  {
    absl::MutexLock l(&verif_mu_);
    auto it = verif_.find(req->replica_uuid());
    if (it != verif_.end()) {
      known_status = it->second.status;
      known_err = it->second.err;
      if (*known_status == ::store_daemon::VerificationStatus::VERIFICATION_STATUS_PASSED ||
          *known_status == ::store_daemon::VerificationStatus::VERIFICATION_STATUS_FAILED) {
        resp->set_status(*known_status);
        if (!known_err.empty())
          resp->set_err_msg(known_err);
        return Status::OK;
      }
    }
  }
  // If no session, treat as unknown UUID
  auto entry = sessions_.get(req->replica_uuid());
  if (!entry.has_value()) {
    resp->set_status(::store_daemon::VerificationStatus::VERIFICATION_STATUS_UNKNOWN);
    return Status::OK;
  }
  // Bounded wait
  std::chrono::milliseconds wait_ms(30000);
  if (req->timeout_ms() > 0)
    wait_ms = std::chrono::milliseconds(req->timeout_ms());
  const auto deadline = ctx->deadline();
  const auto now = std::chrono::system_clock::now();
  if (deadline != std::chrono::system_clock::time_point::max()) {
    if (deadline <= now) {
      wait_ms = std::chrono::milliseconds(0);
    } else {
      auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
      if (remaining < wait_ms)
        wait_ms = remaining;
    }
  }
  auto st_wait = entry->ready.wait_for(wait_ms);
  if (st_wait == std::future_status::timeout) {
    if (compat_.verification_timeout_deadline) {
      return {StatusCode::DEADLINE_EXCEEDED, "verification wait timeout"};
    }
    // Return last-known status (default to IN_PROGRESS) with timeout hint
    auto st = known_status.value_or(::store_daemon::VerificationStatus::VERIFICATION_STATUS_IN_PROGRESS);
    resp->set_status(st);
    resp->set_err_msg("timeout");
    return Status::OK;
  }
  absl::Status st = entry->ready.get();
  if (st.ok()) {
    resp->set_status(::store_daemon::VerificationStatus::VERIFICATION_STATUS_PASSED);
    return Status::OK;
  }
  resp->set_status(::store_daemon::VerificationStatus::VERIFICATION_STATUS_FAILED);
  resp->set_err_msg(std::string(st.message()));
  return to_grpc_status(st);
}

Status StoreDaemonServiceImpl::LockTransportChunks(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::LockChunksRequest* req,
    ::store_daemon::LockChunksResponse* resp) {
  // Resolve ReplicaKey, prefer explicit device_id then infer residency
  auto key = make_replica_key(req->artifact_id());
  if (req->has_device_id()) {
    key.device = tensorcast::store::DeviceRegistry::instance().gpu_key(req->device_id());
  } else {
    int unique_gpu_device = -2; // -2=unknown, -1=none, >=0=single device
    for (const auto& info : engine_->get_all_replicas_info()) {
      if (info.artifact_id == req->artifact_id() && info.gpu_state == tensorcast::store::MemoryLocation::GPU) {
        if (unique_gpu_device == -2) {
          unique_gpu_device = info.gpu_device_id;
        } else if (unique_gpu_device != info.gpu_device_id) {
          unique_gpu_device = -3; // ambiguous
          break;
        }
      }
    }
    if (unique_gpu_device >= 0) {
      key.device = tensorcast::store::DeviceRegistry::instance().gpu_key(unique_gpu_device);
    } else if (unique_gpu_device == -3) {
      return {StatusCode::INVALID_ARGUMENT, "ambiguous artifact residency across multiple GPUs; device_id required"};
    }
  }
  std::vector<uint32_t> indices(req->chunk_indices().begin(), req->chunk_indices().end());
  auto st = engine_->lock_chunks(key, absl::MakeSpan(indices));
  if (!st.ok())
    return to_grpc_status(st);
  std::string token = locks_.mint_token();
  locks_.put(token, key, std::move(indices));
  resp->set_lock_token(token);
  return Status::OK;
}

Status StoreDaemonServiceImpl::UnlockTransportChunks(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::UnlockChunksRequest* req,
    ::store_daemon::UnlockChunksResponse* /*resp*/) {
  auto entry = locks_.get(req->lock_token());
  if (!entry.has_value()) {
    return {StatusCode::NOT_FOUND, "unknown lock token"};
  }
  auto st = engine_->unlock_chunks(entry->key, absl::MakeSpan(entry->chunk_indices), /*copied_gpu=*/false);
  if (!st.ok())
    return to_grpc_status(st);
  locks_.erase(req->lock_token());
  return Status::OK;
}

Status StoreDaemonServiceImpl::BeginRegisterArtifact(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::BeginRegisterArtifactRequest* req,
    ::store_daemon::BeginRegisterArtifactResponse* resp) {
  tensorcast::store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = req->artifact_id();
  reg.device_id = req->device_id();
  reg.total_size_bytes = req->total_size();
  reg.enable_p2p = compat_.enable_p2p_access && req->enable_p2p();
  if (req->has_ttl_ms())
    reg.ttl_ms = req->ttl_ms();
  if (req->has_tensor_index_key()) {
    reg.tensor_index_key = req->tensor_index_key();
  } else if (req->has_tensor_index_data()) {
    reg.tensor_index_data = std::string(req->tensor_index_data().data().begin(), req->tensor_index_data().data().end());
    reg.schema_version = req->tensor_index_data().schema_version();
    reg.encoding = req->tensor_index_data().encoding();
  }
  auto begin_or = engine_->begin_register_artifact(reg);
  if (!begin_or.ok())
    return to_grpc_status(begin_or.status());
  const auto& out = begin_or.value();
  resp->set_registration_id(out.registration_id);
  // Copy IPC handle bytes
  resp->set_daemon_ipc_handle(
      reinterpret_cast<const char*>(out.cuda_ipc_handle_bytes.data()), out.cuda_ipc_handle_bytes.size());
  resp->set_device_id(out.device_id);
  resp->set_size(out.size_bytes);
  return Status::OK;
}

Status StoreDaemonServiceImpl::CommitRegisteredArtifact(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::CommitRegisteredArtifactRequest* req,
    ::store_daemon::CommitRegisteredArtifactResponse* resp) {
  auto commit_or = engine_->commit_registered_artifact(req->registration_id());
  if (!commit_or.ok())
    return to_grpc_status(commit_or.status());
  const auto& out = commit_or.value();
  resp->set_registration_id(out.registration_id);
  resp->set_artifact_id(out.artifact_id);
  resp->set_device_id(out.device_id);
  resp->set_size(out.size_bytes);
  // Populate RFC-0007 content-addressed descriptor for parity with Python daemon
  // and client expectations.
  auto* desc = resp->mutable_descriptor_();
  desc->set_artifact_id(out.artifact_id);
  desc->set_index_multihash(out.index_multihash);
  desc->set_data_multihash(out.data_multihash);
  desc->set_schema_version(out.schema_version);
  desc->set_encoding(out.encoding);
  desc->set_total_size(out.size_bytes);
  return Status::OK;
}

Status StoreDaemonServiceImpl::AbortRegisteredArtifact(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::AbortRegisteredArtifactRequest* req,
    ::store_daemon::AbortRegisteredArtifactResponse* resp) {
  auto st = engine_->abort_registered_artifact(req->registration_id());
  if (!st.ok())
    return to_grpc_status(st);
  resp->set_ok(true);
  return Status::OK;
}

void StoreDaemonServiceImpl::start_sweepers() {
  stop_.store(false);
  sweep_sessions_th_ = std::thread([this]() {
    while (!stop_.load()) {
      for (const auto& k : sessions_.keys()) {
        sessions_.remove_if_expired(k);
      }
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  });
  sweep_locks_th_ = std::thread([this]() {
    while (!stop_.load()) {
      for (const auto& tok : locks_.tokens()) {
        auto expired = locks_.remove_if_expired(tok);
        if (expired.has_value()) {
          (void)engine_->unlock_chunks(expired->key, absl::MakeSpan(expired->chunk_indices), /*copied_gpu=*/false);
        }
      }
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  });
  // Verification and auto-registration sweeper: updates verification status and
  // registers disk-ingested replicas with Global Store once ready.
  verif_sweeper_th_ = std::thread([this]() {
    using namespace std::chrono_literals;
    while (!stop_.load()) {
      // Collect completed verification tasks
      std::vector<std::pair<std::string, absl::Status>> verif_done;
      {
        absl::MutexLock l(&bg_tasks_mu_);
        for (auto it = verif_tasks_.begin(); it != verif_tasks_.end();) {
          if (it->ready.wait_for(0ms) == std::future_status::ready) {
            verif_done.emplace_back(it->uuid, it->ready.get());
            it = verif_tasks_.erase(it);
          } else {
            ++it;
          }
        }
      }
      for (auto& p : verif_done) {
        const std::string& uuid = p.first;
        const absl::Status& st = p.second;
        if (st.ok()) {
          set_verif_status(uuid, ::store_daemon::VerificationStatus::VERIFICATION_STATUS_PASSED);
        } else {
          set_verif_status(
              uuid, ::store_daemon::VerificationStatus::VERIFICATION_STATUS_FAILED, std::string(st.message()));
        }
      }

      // Collect completed auto-registration tasks
      std::vector<AutoRegTask> reg_ready;
      {
        absl::MutexLock l(&bg_tasks_mu_);
        for (auto it = auto_reg_tasks_.begin(); it != auto_reg_tasks_.end();) {
          if (it->ready.wait_for(0ms) == std::future_status::ready) {
            reg_ready.push_back(*it);
            it = auto_reg_tasks_.erase(it);
          } else {
            ++it;
          }
        }
      }
      for (auto& task : reg_ready) {
        // Wait for load completion result (ignore status; parity with previous behavior)
        (void)task.ready.get();
        // Try to read descriptor for mi2 ID
        std::string mi2_id;
        try {
          std::filesystem::path desc_path = std::filesystem::path(task.disk_path) / "artifact_descriptor.json";
          if (std::filesystem::exists(desc_path)) {
            std::ifstream f(desc_path);
            if (f.is_open()) {
              nlohmann::json j;
              f >> j;
              if (j.contains("artifact_id") && j["artifact_id"].is_string()) {
                mi2_id = j["artifact_id"].get<std::string>();
              } else if (
                  j.contains("index_multihash") && j.contains("data_multihash") && j["index_multihash"].is_string() &&
                  j["data_multihash"].is_string()) {
                mi2_id = absl::StrCat(
                    "mi2:", j["index_multihash"].get<std::string>(), ":", j["data_multihash"].get<std::string>());
              }
            }
          }
        } catch (...) {
          // Ignore descriptor parsing errors; fall back to key.artifact_id
        }
        auto st = engine_->register_replica_with_global_store(task.key, mi2_id);
        if (!st.ok()) {
          VLOG(1) << "Auto-register disk load failed: " << st;
        }
      }

      std::this_thread::sleep_for(500ms);
    }
  });
  // PID watcher: drop dead PID refs to avoid leaked references pinning memory
  pid_watcher_th_ = std::thread([this]() {
    while (!stop_.load()) {
      auto keys = refs_.keys();
      for (const auto& key : keys) {
        auto plist = refs_.pids(key);
        for (int32_t pid : plist) {
          // Check /proc/<pid>
          std::string proc_path = absl::StrCat("/proc/", pid);
          if (::access(proc_path.c_str(), F_OK) != 0) {
            refs_.drop_ref(key, pid);
            if (compat_.evict_on_dead_pid && refs_.ref_count(key) == 0 && !refs_.keep_for_global(key)) {
              (void)engine_->unload_replica(key);
            }
          }
        }
      }
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  });
  // Optional periodic eviction policy: unload least-recently-used GPU replicas
  // on devices where used memory exceeds the configured fraction. Disabled by default.
  if (compat_.enable_periodic_eviction) {
    eviction_th_ = std::thread([this]() {
      using namespace std::chrono_literals;
      while (!stop_.load()) {
        // Iterate all GPU devices; if usage exceeds threshold, evict LRU replicas with no refs and not keep_for_global
        const int num_gpus = engine_->get_num_gpus();
        for (int dev = 0; dev < num_gpus; ++dev) {
          auto tot_or = engine_->get_device_total_memory(dev);
          auto free_or = engine_->get_device_free_memory(dev);
          if (!tot_or.ok() || !free_or.ok())
            continue;
          const double total = static_cast<double>(*tot_or);
          const double used = static_cast<double>(*tot_or - *free_or);
          if (total <= 0.0)
            continue;
          double ratio = used / total;
          if (ratio <= compat_.gpu_memory_limit_fraction)
            continue;

          // Build LRU list of candidates on this device
          struct Cand {
            tensorcast::store::ReplicaKey key;
            std::chrono::time_point<std::chrono::system_clock> last_access;
            size_t size;
          };
          std::vector<Cand> cands;
          for (const auto& info : engine_->get_all_replicas_info()) {
            if (info.gpu_state == tensorcast::store::MemoryLocation::NONE)
              continue;
            if (info.gpu_device_id != dev)
              continue;
            tensorcast::store::ReplicaKey key{
                .artifact_id = info.artifact_id,
                .device = tensorcast::store::DeviceRegistry::instance().gpu_key(dev),
                .replica = 0};
            if (refs_.ref_count(key) > 0 || refs_.keep_for_global(key))
              continue;
            cands.push_back(Cand{key, info.last_access_time, static_cast<size_t>(info.size_bytes)});
          }
          std::sort(
              cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.last_access < b.last_access; });

          // Evict until under threshold or no candidates
          for (const auto& c : cands) {
            if (ratio <= compat_.gpu_memory_limit_fraction)
              break;
            (void)engine_->unload_replica(c.key);
            // Recompute ratio after unload attempt
            auto f2 = engine_->get_device_free_memory(dev);
            if (!f2.ok())
              break;
            const double used2 = static_cast<double>(*tot_or - *f2);
            ratio = used2 / total;
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(compat_.eviction_check_interval_ms));
      }
    });
  }
}

void StoreDaemonServiceImpl::stop_sweepers() {
  stop_.store(true);
  if (sweep_sessions_th_.joinable())
    sweep_sessions_th_.join();
  if (sweep_locks_th_.joinable())
    sweep_locks_th_.join();
  if (pid_watcher_th_.joinable())
    pid_watcher_th_.join();
  if (verif_sweeper_th_.joinable())
    verif_sweeper_th_.join();
  if (eviction_th_.joinable())
    eviction_th_.join();
}

Status StoreDaemonServiceImpl::GetWorkerStatus(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::GetWorkerStatusRequest* /*req*/,
    ::store_daemon::GetWorkerStatusResponse* resp) {
  resp->set_is_registered(is_registered());
  resp->set_is_healthy(true);
  resp->set_is_shutting_down(is_shutting_down_.load());
  resp->set_mem_pool_total_size(engine_->get_mem_pool_size());
  resp->set_mem_pool_available_size(engine_->get_available_memory());
  auto uptime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);
  resp->set_uptime_seconds(uptime.count());
  resp->set_worker_id(is_registered() ? worker_id() : "");
  return Status::OK;
}

Status StoreDaemonServiceImpl::GetDetailedStatus(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::GetDetailedStatusRequest* /*req*/,
    ::store_daemon::GetDetailedStatusResponse* resp) {
  resp->set_is_registered(is_registered());
  resp->set_is_healthy(true);
  resp->set_is_shutting_down(is_shutting_down_.load());
  auto uptime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_);
  resp->set_uptime_seconds(uptime.count());
  resp->set_worker_id(is_registered() ? worker_id() : "");

  auto* mp = resp->mutable_memory_pool_info();
  mp->set_total_size_bytes(engine_->get_mem_pool_size());
  mp->set_available_bytes(engine_->get_available_memory());
  mp->set_allocated_bytes(engine_->get_mem_pool_size() - engine_->get_available_memory());
  mp->set_allocated_chunks_count(0);
  mp->set_chunk_size_bytes(engine_->get_chunk_size());

  // Aggregate by GPU device to avoid duplicate device entries. Also compute totals.
  uint64_t total_bytes = 0;
  int32_t total_replicas = 0;
  struct GpuAgg {
    ::store_daemon::GpuDeviceInfo* out;
    bool mem_filled{false};
  };
  absl::flat_hash_map<int, GpuAgg> gpu_map;

  for (const auto& info : engine_->get_all_replicas_info()) {
    if (info.gpu_state != tensorcast::store::MemoryLocation::NONE) {
      auto it = gpu_map.find(info.gpu_device_id);
      if (it == gpu_map.end()) {
        auto* gpu = resp->add_gpu_devices();
        gpu->set_device_id(info.gpu_device_id);
        gpu->set_device_uuid(info.gpu_device_uuid);
        it = gpu_map.emplace(info.gpu_device_id, GpuAgg{gpu, false}).first;
      }
      // Populate GPU memory totals once per device
      if (!it->second.mem_filled) {
        size_t total_mem = 0;
        size_t free_mem = 0;
        if (auto t = engine_->get_device_total_memory(info.gpu_device_id); t.ok()) {
          total_mem = *t;
        }
        if (auto f = engine_->get_device_free_memory(info.gpu_device_id); f.ok()) {
          free_mem = *f;
        }
        it->second.out->set_total_memory_bytes(static_cast<uint64_t>(total_mem));
        it->second.out->set_free_memory_bytes(static_cast<uint64_t>(free_mem));
        uint64_t used = (total_mem > free_mem) ? static_cast<uint64_t>(total_mem - free_mem) : 0ULL;
        it->second.out->set_used_memory_bytes(used);
        it->second.mem_filled = true;
      }
      auto* r = it->second.out->add_loaded_replicas();
      r->set_artifact_id(info.artifact_id);
      r->set_artifact_size_bytes(info.size_bytes);
      r->set_location(::store_daemon::MemoryLocation::MEMORY_LOCATION_GPU);
      r->set_loaded_timestamp(
          std::chrono::duration_cast<std::chrono::seconds>(info.load_time.time_since_epoch()).count());
      r->set_last_access_timestamp(
          std::chrono::duration_cast<std::chrono::seconds>(info.last_access_time.time_since_epoch()).count());
      r->add_replica_uuids("");
      r->set_is_registered_for_comm(info.is_registered_for_comm);
      total_replicas += 1;
      total_bytes += info.size_bytes;
    }
    if (info.cpu_state != tensorcast::store::MemoryLocation::NONE) {
      auto* r = resp->add_cpu_replicas();
      r->set_artifact_id(info.artifact_id);
      r->set_artifact_size_bytes(info.size_bytes);
      r->set_location(::store_daemon::MemoryLocation::MEMORY_LOCATION_PAGEABLE_CPU);
      r->set_loaded_timestamp(
          std::chrono::duration_cast<std::chrono::seconds>(info.load_time.time_since_epoch()).count());
      r->set_last_access_timestamp(
          std::chrono::duration_cast<std::chrono::seconds>(info.last_access_time.time_since_epoch()).count());
      r->add_replica_uuids("");
      r->set_is_registered_for_comm(info.is_registered_for_comm);
      total_replicas += 1;
      total_bytes += info.size_bytes;
    }
  }

  bool any_comm = false;
  for (const auto& info : engine_->get_all_replicas_info()) {
    any_comm = any_comm || info.is_registered_for_comm;
  }
  resp->mutable_communication_info()->set_enabled(compat_.enable_p2p_access && any_comm);
  resp->set_total_replicas_loaded(total_replicas);
  resp->set_total_artifact_size_bytes(static_cast<int64_t>(total_bytes));
  resp->set_storage_path("");
  resp->set_num_worker_threads(0);
  return Status::OK;
}

void StoreDaemonServiceImpl::set_verif_status(
    const std::string& uuid,
    ::store_daemon::VerificationStatus st,
    std::string err) {
  absl::MutexLock l(&verif_mu_);
  verif_[uuid] = VerifEntry{st, std::move(err)};
}

Status StoreDaemonServiceImpl::GetLoadedReplicas(
    grpc::ServerContext* /*ctx*/,
    const ::store_daemon::GetLoadedReplicasRequest* req,
    ::store_daemon::GetLoadedReplicasResponse* resp) {
  int32_t total = 0;
  uint64_t total_bytes = 0;
  for (const auto& info : engine_->get_all_replicas_info()) {
    // Derive device id (-1 for CPU entries, device id for GPU)
    int device_id = -1;
    if (info.gpu_state != tensorcast::store::MemoryLocation::NONE) {
      device_id = info.gpu_device_id;
    }
    if (req->has_artifact_id_filter() && info.artifact_id.find(req->artifact_id_filter()) == std::string::npos)
      continue;
    if (req->has_device_id_filter() && device_id != req->device_id_filter())
      continue;

    tensorcast::store::ReplicaKey key;
    key.artifact_id = info.artifact_id;
    key.device = (device_id >= 0) ? tensorcast::store::DeviceRegistry::instance().gpu_key(device_id)
                                  : tensorcast::store::DeviceKey{tensorcast::DeviceType::CPU, -1, ""};
    key.replica = 0;

    auto* out = resp->add_replicas();
    out->set_artifact_id(info.artifact_id);
    out->set_device_id(device_id);
    out->set_ref_count(static_cast<int32_t>(refs_.ref_count(key)));
    // pids
    for (int32_t pid : refs_.pids(key)) {
      out->add_pids(pid);
    }
    out->set_size_bytes(info.size_bytes);
    out->set_keep_for_global(refs_.keep_for_global(key));
    out->set_last_access_timestamp(
        std::chrono::duration_cast<std::chrono::seconds>(info.last_access_time.time_since_epoch()).count());
    total++;
    total_bytes += info.size_bytes;
  }
  resp->set_total_replicas(total);
  resp->set_total_size_bytes(total_bytes);
  return Status::OK;
}

} // namespace tensorcast::daemon
