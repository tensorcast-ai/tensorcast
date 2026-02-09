// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "core/store/components/global_store_client.h"
#include "core/store/runtime/context/runtime_context_events.h"
#include "core/store/store_engine.h"
#include "daemon/ha/worker_lifecycle_ports.h"
#include "gsl/pointers"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::daemon {

namespace commonpb = ::tensorcast::common::v1;

class WorkerLifecycleManager {
 public:
  struct Options {
    std::string global_store_addr;
    std::string listen_addr; // host:port
    // Optional externally advertised host for registration (overrides listen host)
    std::string advertise_host;
    uint16_t p2p_port{0};
    int heartbeat_interval_ms{5000};
    int chunk_sync_interval_ms{10000}; // 0 to disable
    int heartbeat_rpc_timeout_ms{0};
    int state_sync_rpc_timeout_ms{0};
    int full_sync_rpc_timeout_ms{0};
    std::optional<int32_t> heartbeat_rpc_max_retries;
    std::optional<int32_t> state_sync_rpc_max_retries;
    std::optional<int32_t> full_sync_rpc_max_retries;
    // When true, an empty local inventory is treated as authoritative and will
    // drive removals via force_full_sync during synchronization.
    bool force_full_sync_on_empty_inventory{false};
    // Optional cluster identity guard; if set, daemon will refuse to register
    // with a Global Store reporting a different token.
    std::string cluster_token;
    // Optional shared Global Store client to reuse across the daemon.
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    // Capability directory bitset for Global Store registration/heartbeats.
    uint64_t capability_flags{0};
  };

  WorkerLifecycleManager(
      gsl::not_null<std::shared_ptr<store::StoreEngine>> engine,
      WorkerLifecyclePorts ports,
      Options opts);

  ~WorkerLifecycleManager() {
    stop();
  }

  absl::Status start();
  void stop();

  // Exposed for deterministic HA synchronization and unit tests.
  static std::string compute_state_checksum(
      std::string_view node_id,
      std::string_view node_address,
      uint32_t node_port,
      const std::vector<store::StoreEngine::ReplicaInventoryEntry>& inventory);

 private:
  enum class AdvertisedAddressSource {
    kExplicit,
    kListen,
    kRoute,
    kDefault,
  };

  struct ResolvedAdvertisedAddress {
    std::string host;
    AdvertisedAddressSource source{AdvertisedAddressSource::kDefault};
  };

  static const char* advertised_source_to_cstr(AdvertisedAddressSource source);

  static absl::StatusOr<ResolvedAdvertisedAddress> resolve_advertised_address(
      const WorkerLifecycleManager::Options& opts);

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
  void state_sync_loop(uint64_t epoch);
  void chunk_sync_loop();
  void monitor_loop();
  void apply_obsolete_replicas(const std::vector<std::string>& artifact_ids);
  void apply_full_state(const std::vector<commonpb::ReplicaInfo>& expected);
  void enqueue_retire_keys(std::vector<store::loading::ReplicaKey> keys, std::string_view context);
  void process_retire_queue();
  void schedule_demotion_task(const store::loading::ReplicaKey& key, std::string_view context);
  bool wait_for_state_sync_success(uint64_t baseline, std::chrono::milliseconds timeout);
  absl::Status reregister_worker(bool preserve_identity);
  void reconcile_memory_tier_leases_once();
  bool wait_for_stop(std::chrono::milliseconds interval);
  void request_state_sync();
  void queue_obsolete_replicas(std::vector<std::string> obsolete);
  void perform_state_sync(uint64_t epoch);
  void retire_thread(std::thread* thread);
  store::components::RpcOptions build_rpc_options(int timeout_ms, std::optional<int32_t> max_retries) const;
  store::components::StateSyncToken next_state_sync_token(uint64_t epoch);
  void mark_state_sync_progress();
  std::optional<std::chrono::milliseconds> state_sync_stall_budget() const;

  const gsl::not_null<std::shared_ptr<store::StoreEngine>> engine_;
  const WorkerLifecyclePorts ports_;
  const Options opts_;
  const std::string daemon_id_;

  static gsl::not_null<std::shared_ptr<store::components::IGlobalStoreClient>> make_global_store_client(
      const Options& opts);
  const gsl::not_null<std::shared_ptr<store::components::IGlobalStoreClient>> global_store_;
  const std::string node_id_;
  std::string worker_id_;
  std::string node_address_;

  // HA state tracking
  uint64_t state_version_{0};
  std::string state_checksum_;
  int64_t last_sync_success_ts_{0};
  mutable std::mutex state_mu_;

  std::atomic<bool> stop_{false};
  // Ensure stop() is idempotent even if invoked from multiple places (e.g.,
  // explicit shutdown path and destructor). When true, subsequent calls to
  // stop() are no-ops.
  std::atomic<bool> stop_called_{false};
  std::mutex stop_mu_;
  std::condition_variable stop_cv_;
  std::thread hb_thread_;
  std::thread state_sync_thread_;
  std::thread sync_thread_;
  std::thread monitor_thread_;
  std::mutex retired_threads_mu_;
  std::vector<std::thread> retired_threads_;

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
  std::atomic<uint64_t> state_sync_ticks_{0};
  std::atomic<uint64_t> sync_ticks_{0};
  std::atomic<bool> hb_alive_{false};
  std::atomic<bool> state_sync_alive_{false};
  std::atomic<bool> state_sync_inflight_{false};
  std::atomic<bool> sync_alive_{false};
  std::atomic<uint64_t> hb_epoch_{0};
  std::atomic<uint64_t> state_sync_epoch_{0};
  std::atomic<uint64_t> state_sync_requests_{0};
  std::atomic<uint64_t> state_sync_request_id_{0};
  std::atomic<int64_t> state_sync_last_progress_ns_{0};
  std::atomic<bool> state_sync_restart_pending_{false};
  std::mutex state_sync_mu_;
  std::condition_variable state_sync_cv_;
  std::mutex obsolete_mu_;
  std::vector<std::string> pending_obsolete_replicas_;
  std::atomic<bool> obsolete_pending_{false};

  struct RetireEntry {
    size_t attempts{0};
    int64_t last_log_ts_s{0};
    bool demotion_started{false};
    bool demotion_complete{false};
  };

  std::mutex retire_mu_;
  absl::flat_hash_map<store::loading::ReplicaKey, RetireEntry, store::loading::ReplicaKeyHash> retire_queue_;
  std::atomic<bool> retire_pending_{false};
  std::mutex sync_success_mu_;
  std::condition_variable sync_success_cv_;
  std::unique_ptr<store::runtime::RuntimeContextEvents::Subscription> runtime_event_subscription_;
  bool memory_tier_enabled_{false};

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
