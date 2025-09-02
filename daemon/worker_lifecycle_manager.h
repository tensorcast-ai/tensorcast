// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/grpc_service_impl.h"
#include "tensorcast/global/global_store.pb.h"

namespace tensorcast::daemon {

class WorkerLifecycleManager {
 public:
  struct Options {
    std::string global_store_addr;
    std::string listen_addr; // host:port
    uint16_t p2p_port{0};
    int heartbeat_interval_ms{5000};
    int chunk_sync_interval_ms{10000}; // 0 to disable
  };

  WorkerLifecycleManager(std::shared_ptr<store::StoreEngine> engine, StoreDaemonServiceImpl* service, Options opts)
      : engine_(std::move(engine)), service_(service), opts_(std::move(opts)) {}

  ~WorkerLifecycleManager() {
    stop();
  }

  absl::Status start();
  void stop();

 private:
  static std::string host_from_listen(const std::string& listen) {
    auto pos = listen.find(':');
    if (pos == std::string::npos) {
      return listen;
    }
    return listen.substr(0, pos);
  }
  static uint32_t port_from_listen(const std::string& listen) {
    auto pos = listen.rfind(':');
    if (pos == std::string::npos) {
      return 0;
    }
    return static_cast<uint32_t>(std::stoi(listen.substr(pos + 1)));
  }

  void heartbeat_loop();
  void chunk_sync_loop();
  void monitor_loop();
  void apply_obsolete_replicas(const std::vector<std::string>& artifact_ids);
  void apply_full_state(const std::vector<common::ReplicaInfo>& expected);
  static std::string compute_state_checksum(const std::vector<store::StoreEngine::ReplicaInfo>& infos);

  std::shared_ptr<store::StoreEngine> engine_;
  StoreDaemonServiceImpl* service_;
  Options opts_;

  std::unique_ptr<store::components::GlobalStoreClient> gs_;
  std::string worker_id_;
  std::string node_id_;

  // HA state tracking
  uint64_t state_version_{0};
  std::string state_checksum_;
  int64_t last_sync_success_ts_{0};

  std::atomic<bool> stop_{false};
  std::thread hb_thread_;
  std::thread sync_thread_;
  std::thread monitor_thread_;

  // Lightweight metrics/counters for observability
  std::atomic<uint64_t> hb_success_{0};
  std::atomic<uint64_t> hb_failure_{0};
  std::atomic<uint64_t> sync_success_{0};
  std::atomic<uint64_t> sync_failure_{0};
  std::atomic<int64_t> last_hb_ts_s_{0};
  std::atomic<int64_t> last_sync_ts_s_{0};
  std::atomic<uint64_t> hb_restarts_{0};
  std::atomic<uint64_t> sync_restarts_{0};
  std::atomic<uint64_t> hb_ticks_{0};
  std::atomic<uint64_t> sync_ticks_{0};
  std::atomic<bool> hb_alive_{false};
  std::atomic<bool> sync_alive_{false};

 public:
  // Read-only accessors for metrics exporter
  uint64_t hb_success() const {
    return hb_success_.load();
  }
  uint64_t hb_failure() const {
    return hb_failure_.load();
  }
  uint64_t sync_success() const {
    return sync_success_.load();
  }
  uint64_t sync_failure() const {
    return sync_failure_.load();
  }
  int64_t last_hb_ts_s() const {
    return last_hb_ts_s_.load();
  }
  int64_t last_sync_ts_s() const {
    return last_sync_ts_s_.load();
  }
  uint64_t hb_restarts() const {
    return hb_restarts_.load();
  }
  uint64_t sync_restarts() const {
    return sync_restarts_.load();
  }
  bool hb_alive() const {
    return hb_alive_.load();
  }
  bool sync_alive() const {
    return sync_alive_.load();
  }
};

} // namespace tensorcast::daemon
