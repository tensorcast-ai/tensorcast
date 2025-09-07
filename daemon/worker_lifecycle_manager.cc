// Copyright (c) 2025, TensorCast Team.

#include "daemon/worker_lifecycle_manager.h"

#include <unistd.h>

#include <chrono>
#include <cstring>
#include <exception>

#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "core/store/device_registry.h"

namespace tensorcast::daemon {

namespace global_store = tensorcast::global_store::v1;
namespace commonpb = common::v1;

using namespace std::chrono_literals;

absl::Status WorkerLifecycleManager::start() {
  if (opts_.global_store_addr.empty()) {
    return absl::OkStatus();
  }
  store::components::GlobalStoreClientConfig cfg;
  cfg.global_store_address = opts_.global_store_addr;
  gs_ = std::make_unique<store::components::GlobalStoreClient>(cfg);
  auto st = gs_->initialize();
  if (!st.ok())
    return st;

  char hostname[256];
  if (::gethostname(hostname, sizeof(hostname)) != 0) {
    std::strncpy(hostname, "unknown", sizeof(hostname));
    hostname[sizeof(hostname) - 1] = '\0';
  }
  node_id_ = hostname;
  const std::string node_addr = host_from_listen(opts_.listen_addr);
  const uint32_t grpc_port = port_from_listen(opts_.listen_addr);

  auto reg_or = gs_->register_worker(
      node_id_, node_addr, grpc_port, opts_.p2p_port, engine_->get_mem_pool_size(), engine_->get_available_memory());
  if (!reg_or.ok())
    return reg_or.status();
  worker_id_ = *reg_or;
  service_->set_worker_registered(worker_id_);
  // Propagate worker identity into the engine so subsequent GS registrations
  // use the real worker_id instead of a placeholder.
  if (engine_) {
    engine_->set_worker_identity(worker_id_, node_id_);
  }

  // Initial full-state sync: query GS for expected replicas and evict local
  // replicas not present in the expected set to remove drift.
  if (gs_) {
    std::vector<commonpb::ReplicaInfo> expected;
    auto full_or = gs_->request_full_state_sync(worker_id_, /*current_state_version=*/0, &expected);
    if (full_or.ok()) {
      state_version_ = full_or->first;
      state_checksum_ = full_or->second;
      apply_full_state(expected);
      last_sync_success_ts_ = static_cast<int64_t>(
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
              .count());
    } else {
      LOG(WARNING) << "Initial RequestFullStateSync failed: " << full_or.status();
    }
  }

  stop_.store(false);
  hb_thread_ = std::thread(&WorkerLifecycleManager::heartbeat_loop, this);
  if (opts_.chunk_sync_interval_ms > 0) {
    sync_thread_ = std::thread(&WorkerLifecycleManager::chunk_sync_loop, this);
  }
  monitor_thread_ = std::thread(&WorkerLifecycleManager::monitor_loop, this);
  return absl::OkStatus();
}

void WorkerLifecycleManager::stop() {
  if (!gs_)
    return;
  stop_.store(true);
  if (hb_thread_.joinable())
    hb_thread_.join();
  if (sync_thread_.joinable())
    sync_thread_.join();
  if (monitor_thread_.joinable())
    monitor_thread_.join();
  // Best-effort disable remote access before unregistering worker
  for (const auto& info : engine_->get_all_replicas_info()) {
    if (!info.is_registered_for_comm)
      continue;
    store::DeviceKey dev = store::DeviceRegistry::instance().gpu_key(info.gpu_device_id);
    store::loading::ReplicaKey key{.artifact_id = info.artifact_id, .device = dev, .replica = 0};
    (void)engine_->disable_remote_replica_access(key, common::memory::MemoryLocation::GPU);
  }
  if (!worker_id_.empty()) {
    (void)gs_->unregister_worker(worker_id_, /*is_graceful_shutdown=*/true);
  }
}

void WorkerLifecycleManager::heartbeat_loop() {
  hb_alive_.store(true);
  const auto interval = std::chrono::milliseconds(opts_.heartbeat_interval_ms);
  try {
    while (!stop_.load()) {
      // Prepare enhanced heartbeat fields
      const bool accepting = !service_->is_shutting_down();
      const auto infos = engine_->get_all_replicas_info();
      std::vector<std::string> registered_ids;
      registered_ids.reserve(infos.size());
      for (const auto& i : infos) {
        registered_ids.push_back(i.artifact_id);
      }
      // Compute simple checksum over current snapshot
      state_checksum_ = compute_state_checksum(infos);
      auto hb_or = gs_->send_heartbeat_enhanced(
          worker_id_,
          engine_->get_available_memory(),
          accepting,
          state_version_,
          state_checksum_,
          registered_ids,
          last_sync_success_ts_,
          global_store::CONNECTION_STATUS_CONNECTED);
      if (!hb_or.ok()) {
        LOG(WARNING) << "Enhanced heartbeat failed: " << hb_or.status().message();
        hb_failure_.fetch_add(1);
      } else {
        const auto& hb = *hb_or;
        hb_success_.fetch_add(1);
        last_hb_ts_s_.store(
            static_cast<int64_t>(
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                    .count()));
        // Apply any explicit obsolete replicas immediately
        if (hb.obsolete_replicas_size() > 0) {
          std::vector<std::string> obs;
          obs.reserve(hb.obsolete_replicas_size());
          for (const auto& id : hb.obsolete_replicas())
            obs.push_back(id);
          apply_obsolete_replicas(obs);
        }
        const bool needs_sync = hb.state_sync_required() ||
            (hb.expected_state_version() > 0 && hb.expected_state_version() != state_version_);
        if (needs_sync) {
          // Build local state for synchronize call
          global_store::WorkerLocalState local_state;
          local_state.set_worker_id(worker_id_);
          local_state.set_state_version(state_version_);
          local_state.set_state_checksum(state_checksum_);
          {
            auto* ts = local_state.mutable_last_update_ts();
            ts->set_seconds(
                static_cast<int64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                         std::chrono::system_clock::now().time_since_epoch())
                                         .count()));
            ts->set_nanos(0);
          }
          for (const auto& i : infos) {
            auto* rep = local_state.add_local_replicas();
            rep->mutable_ref()->set_artifact_id(i.artifact_id);
            rep->mutable_ref()->set_replica_id("");
            auto* mi = rep->mutable_memory_info();
            mi->set_memory_size(i.size_bytes);
            if (i.gpu_state != common::memory::MemoryLocation::NONE) {
              mi->set_memory_type(commonpb::MEMORY_TYPE_GPU);
              mi->set_device_id(i.gpu_device_id);
            } else if (i.cpu_state != common::memory::MemoryLocation::NONE) {
              mi->set_memory_type(commonpb::MEMORY_TYPE_RAM);
              mi->set_device_id(0);
            } else {
              mi->set_memory_type(commonpb::MEMORY_TYPE_DISK);
              mi->set_device_id(0);
            }
            rep->mutable_stats()->set_max_concurrency(1);
            // Reconcile current_requests with active PID refs tracked by the service
            store::loading::ReplicaKey rkey{
                .artifact_id = i.artifact_id,
                .device = (i.gpu_state != common::memory::MemoryLocation::NONE)
                    ? store::DeviceRegistry::instance().gpu_key(i.gpu_device_id)
                    : store::DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
                .replica = 0};
            rep->mutable_stats()->set_current_requests(static_cast<uint32_t>(service_->ref_count_for(rkey)));
            rep->mutable_stats()->set_is_available(true);
            rep->mutable_stats()->mutable_registered_ts()->CopyFrom(local_state.last_update_ts());
          }

          std::vector<global_store::StateChange> changes;
          auto sync_or = gs_->synchronize_worker_state(local_state, /*force_full_sync=*/false, &changes);
          if (sync_or.ok()) {
            state_version_ = sync_or->first;
            state_checksum_ = sync_or->second;
            sync_success_.fetch_add(1);
            // Apply server-suggested removals
            std::vector<std::string> obsolete;
            obsolete.reserve(changes.size());
            for (const auto& ch : changes) {
              switch (ch.type()) {
                case global_store::StateChange::CHANGE_TYPE_REMOVE_REPLICA: {
                  obsolete.push_back(ch.replica_info().ref().artifact_id());
                  break;
                }
                case global_store::StateChange::CHANGE_TYPE_ADD_REPLICA: {
                  // Proactively materialize the replica locally on the indicated memory
                  const auto& ri = ch.replica_info();
                  store::DeviceKey dev{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
                  if (ri.memory_info().memory_type() == commonpb::MEMORY_TYPE_GPU) {
                    dev = store::DeviceRegistry::instance().gpu_key(static_cast<int>(ri.memory_info().device_id()));
                  } else if (ri.memory_info().memory_type() == commonpb::MEMORY_TYPE_RAM) {
                    dev = store::DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
                  } else {
                    // Ignore DISK-only add in daemon prefetch
                    break;
                  }
                  std::string artifact_id = ri.ref().artifact_id();
                  auto engine = engine_;
                  std::thread([engine, dev, artifact_id]() {
                    store::loading::MaterializeHints hints;
                    hints.artifact_id = artifact_id;
                    (void)engine->materialize_replica(dev, store::StoreEngine::MaterializeMode::LOAD_ONLY, hints);
                  }).detach();
                  break;
                }
                case global_store::StateChange::CHANGE_TYPE_UPDATE_REPLICA: {
                  // Reconcile availability (enable/disable remote access) if applicable
                  const auto& ri = ch.replica_info();
                  const auto artifact_id = ri.ref().artifact_id();
                  // Find local replica info to get device id and comm registration
                  for (const auto& li : engine_->get_all_replicas_info()) {
                    if (li.artifact_id != artifact_id)
                      continue;
                    if (li.gpu_state == common::memory::MemoryLocation::NONE)
                      continue;
                    auto dev = store::DeviceRegistry::instance().gpu_key(li.gpu_device_id);
                    store::loading::ReplicaKey key{.artifact_id = li.artifact_id, .device = dev, .replica = 0};
                    if (!ri.stats().is_available() && li.is_registered_for_comm) {
                      (void)engine_->disable_remote_replica_access(key, common::memory::MemoryLocation::GPU);
                    } else if (ri.stats().is_available() && !li.is_registered_for_comm) {
                      (void)engine_->enable_remote_replica_access(key, common::memory::MemoryLocation::GPU);
                    }
                  }
                  break;
                }
                default:
                  break;
              }
            }
            if (!obsolete.empty())
              apply_obsolete_replicas(obsolete);
            last_sync_success_ts_ = local_state.last_update_ts().seconds();
            last_sync_ts_s_.store(last_sync_success_ts_);
          } else {
            VLOG(1) << "SynchronizeWorkerState returned: " << sync_or.status();
            sync_failure_.fetch_add(1);
            // Fallback to full-state sync if server indicates desync or errors persist
            std::vector<commonpb::ReplicaInfo> expected;
            auto full_or = gs_->request_full_state_sync(worker_id_, state_version_, &expected);
            if (full_or.ok()) {
              state_version_ = full_or->first;
              state_checksum_ = full_or->second;
              apply_full_state(expected);
              last_sync_success_ts_ = static_cast<int64_t>(
                  std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                      .count());
              last_sync_ts_s_.store(last_sync_success_ts_);
            }
          }
        }
      }
      std::this_thread::sleep_for(interval);
      hb_ticks_.fetch_add(1);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "heartbeat_loop crashed: " << e.what();
  } catch (...) {
    LOG(ERROR) << "heartbeat_loop crashed: unknown exception";
  }
  hb_alive_.store(false);
}

void WorkerLifecycleManager::chunk_sync_loop() {
  sync_alive_.store(true);
  const auto interval = std::chrono::milliseconds(opts_.chunk_sync_interval_ms);
  try {
    while (!stop_.load()) {
      std::vector<store::components::GlobalStoreClient::ChunkStateUpdate> updates;
      for (const auto& info : engine_->get_all_replicas_info()) {
        if (info.gpu_state == common::memory::MemoryLocation::NONE)
          continue;
        auto states = engine_->get_chunk_states(info.artifact_id);
        updates.reserve(updates.size() + states.size());
        for (size_t i = 0; i < states.size(); ++i) {
          store::components::GlobalStoreClient::ChunkStateUpdate u;
          u.artifact_id = info.artifact_id;
          u.chunk_idx = static_cast<uint32_t>(i);
          u.state = states[i];
          u.device_uuid = info.gpu_device_uuid;
          u.replica = 0;
          updates.push_back(std::move(u));
        }
      }
      if (!updates.empty()) {
        (void)gs_->batch_update_chunk_states(worker_id_, node_id_, updates);
      }
      std::this_thread::sleep_for(interval);
      sync_ticks_.fetch_add(1);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "chunk_sync_loop crashed: " << e.what();
  } catch (...) {
    LOG(ERROR) << "chunk_sync_loop crashed: unknown exception";
  }
  sync_alive_.store(false);
}

void WorkerLifecycleManager::monitor_loop() {
  // Simple liveness/restart loop for lifecycle threads
  using namespace std::chrono_literals;
  uint64_t last_hb_ticks = 0;
  uint64_t last_sync_ticks = 0;
  auto last_hb_change = std::chrono::steady_clock::now();
  auto last_sync_change = std::chrono::steady_clock::now();
  while (!stop_.load()) {
    // Detect stalled heartbeat: ticks not increasing for > 3 * heartbeat_interval
    if (hb_ticks_.load() != last_hb_ticks) {
      last_hb_ticks = hb_ticks_.load();
      last_hb_change = std::chrono::steady_clock::now();
    } else if (
        std::chrono::steady_clock::now() - last_hb_change >
        std::chrono::milliseconds(opts_.heartbeat_interval_ms * 3)) {
      LOG(WARNING) << "heartbeat_loop appears stalled; restarting";
      hb_alive_.store(false);
    }
    if (!hb_alive_.load() && !stop_.load()) {
      if (hb_thread_.joinable())
        hb_thread_.join();
      hb_restarts_.fetch_add(1);
      hb_thread_ = std::thread(&WorkerLifecycleManager::heartbeat_loop, this);
    }
    // Detect stalled sync: ticks not increasing for > 3 * chunk_sync_interval
    if (opts_.chunk_sync_interval_ms > 0) {
      if (sync_ticks_.load() != last_sync_ticks) {
        last_sync_ticks = sync_ticks_.load();
        last_sync_change = std::chrono::steady_clock::now();
      } else if (
          std::chrono::steady_clock::now() - last_sync_change >
          std::chrono::milliseconds(opts_.chunk_sync_interval_ms * 3)) {
        LOG(WARNING) << "chunk_sync_loop appears stalled; restarting";
        sync_alive_.store(false);
      }
    }
    if (opts_.chunk_sync_interval_ms > 0 && !sync_alive_.load() && !stop_.load()) {
      if (sync_thread_.joinable())
        sync_thread_.join();
      sync_restarts_.fetch_add(1);
      sync_thread_ = std::thread(&WorkerLifecycleManager::chunk_sync_loop, this);
    }
    std::this_thread::sleep_for(5s);
  }
}

void WorkerLifecycleManager::apply_obsolete_replicas(const std::vector<std::string>& artifact_ids) {
  if (artifact_ids.empty())
    return;
  auto infos = engine_->get_all_replicas_info();
  for (const auto& id : artifact_ids) {
    for (const auto& info : infos) {
      if (info.artifact_id != id)
        continue;
      // Unload both GPU and CPU replicas with id-match
      if (info.gpu_state != common::memory::MemoryLocation::NONE) {
        auto dev = store::DeviceRegistry::instance().gpu_key(info.gpu_device_id);
        store::loading::ReplicaKey key{.artifact_id = info.artifact_id, .device = dev, .replica = 0};
        (void)engine_->unload_replica(key);
      }
      if (info.cpu_state != common::memory::MemoryLocation::NONE) {
        store::DeviceKey cpu{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
        store::loading::ReplicaKey key{.artifact_id = info.artifact_id, .device = cpu, .replica = 0};
        (void)engine_->unload_replica(key);
      }
    }
  }
}

void WorkerLifecycleManager::apply_full_state(const std::vector<commonpb::ReplicaInfo>& expected) {
  // Build a set of expected artifact_ids for quick lookup
  absl::flat_hash_set<std::string> expected_ids;
  expected_ids.reserve(expected.size());
  for (const auto& r : expected)
    expected_ids.insert(r.ref().artifact_id());

  std::vector<std::string> obsolete;
  for (const auto& info : engine_->get_all_replicas_info()) {
    if (expected_ids.find(info.artifact_id) == expected_ids.end()) {
      obsolete.push_back(info.artifact_id);
    }
  }
  apply_obsolete_replicas(obsolete);
}

std::string WorkerLifecycleManager::compute_state_checksum(const std::vector<store::StoreEngine::ReplicaInfo>& infos) {
  // Simple stable hash over (artifact_id, gpu_state, cpu_state, device_id)
  size_t h = 0;
  for (const auto& i : infos) {
    h = absl::HashOf(h, i.artifact_id, static_cast<int>(i.gpu_state), static_cast<int>(i.cpu_state), i.gpu_device_id);
  }
  // Format as hex for readability
  std::string out;
  out.resize(sizeof(size_t) * 2);
  static const char* hex = "0123456789abcdef";
  for (size_t i = 0; i < sizeof(size_t); ++i) {
    auto byte = static_cast<uint8_t>((h >> ((sizeof(size_t) - 1 - i) * 8)) & 0xFF);
    out[i * 2] = hex[(byte >> 4) & 0xF];
    out[i * 2 + 1] = hex[byte & 0xF];
  }
  return out;
}

} // namespace tensorcast::daemon
