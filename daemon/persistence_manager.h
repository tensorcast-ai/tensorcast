// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/utility/utility.h"
#include "core/common/memory/memory_location.h"
#include "core/store/components/global_store_client.h"
#include "core/store/device_types.h"
#include "daemon/background_scheduler.h"
#include "daemon/types.h"
#include "tensorcast/daemon/v1/store_daemon.pb.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::daemon {

class LipManager;

struct PersistenceTargetState {
  std::string node_id;
  std::string lease_id;
  tensorcast::global_store::v1::PlacementTargetState target_state{
      tensorcast::global_store::v1::PLACEMENT_TARGET_STATE_PENDING};
  std::string degraded_reason;
  bool is_shared_disk{false};
  uint32_t attempts{0};
  uint32_t cooldown_ticks{0};
  bool lease_acked{false};
  bool replica_registered{false};
};

struct PersistenceShardState {
  std::string shard_id;
  uint32_t shard_idx{0};
  uint64_t byte_range_start{0};
  uint64_t byte_range_length{0};
  uint64_t size_bytes{0};
  std::string content_digest;
  std::vector<uint32_t> chunk_ids;
  v1::PersistenceState state{v1::PERSISTENCE_STATE_PENDING};
  double progress{0.0};
  std::string degraded_reason;
  std::string last_error;
  std::vector<PersistenceTargetState> targets;
};

struct PersistenceTaskState {
  std::string task_id;
  std::string plan_id;
  std::string artifact_id;
  v1::PlacementPolicy placement_policy{v1::PLACEMENT_POLICY_UNSPECIFIED};
  bool persist_to_shared_disk{false};
  v1::PersistenceState state{v1::PERSISTENCE_STATE_PENDING};
  double progress{0.0};
  std::string degraded_reason;
  std::string last_error;
  std::vector<PersistenceShardState> shards;
  bool metrics_active{false};
  bool metrics_closed{false};
};

// Persistence task tracker with lightweight shard planning/state machine.
class PersistenceManager {
 public:
  explicit PersistenceManager(
      BackgroundScheduler* scheduler,
      LipManager* lip_mgr,
      size_t artifact_chunk_bytes,
      std::chrono::milliseconds tick_interval = std::chrono::milliseconds(500),
      std::filesystem::path task_log_path = {});

  absl::StatusOr<PersistenceTaskState> start_task(
      std::string artifact_id,
      v1::PlacementPolicy placement_policy,
      bool persist_to_shared_disk);

  PersistenceTaskState start_task_for_test(
      std::string artifact_id,
      v1::PlacementPolicy placement_policy,
      bool persist_to_shared_disk,
      uint64_t total_size_bytes = 64ULL * 1024 * 1024) {
    LipLeaseEntry lease;
    lease.artifact_id = artifact_id;
    lease.total_size = total_size_bytes;
    return start_task_with_lease(std::move(lease), placement_policy, persist_to_shared_disk).value();
  }

  absl::optional<PersistenceTaskState> get_by_task_id(absl::string_view task_id) const;
  absl::optional<PersistenceTaskState> get_latest_for_artifact(absl::string_view artifact_id) const;

  // Tick progress forward (pending -> running -> success). Exposed for tests.
  void advance_once_for_test();

  void set_local_node_id(std::string node_id);
  void set_global_store_client(store::components::IGlobalStoreClient* client);

  void set_fail_shared_disk_for_test(bool fail) {
    fail_shared_disk_for_test_.store(fail);
  }

 private:
  using PlacementPlanResult = store::components::PlacementPlanResult;
  using PlacementShardSpec = store::components::PlacementShardSpec;

  absl::StatusOr<PersistenceTaskState> start_task_with_lease(
      LipLeaseEntry lease,
      v1::PlacementPolicy placement_policy,
      bool persist_to_shared_disk);
  absl::StatusOr<std::vector<PersistenceShardState>> plan_shards(const LipLeaseEntry& lease) const;
  absl::Status apply_placement_plan(PersistenceTaskState& task, std::vector<PersistenceShardState>& shards);
  absl::StatusOr<PlacementPlanResult> request_plan(
      const PersistenceTaskState& task,
      const std::vector<PersistenceShardState>& shards) const;
  void propagate_degraded_reason(PersistenceTaskState& task) const;
  static double compute_task_progress(const PersistenceTaskState& task);
  void attach_shared_disk_targets(bool persist_to_shared_disk, std::vector<PersistenceShardState>& shards)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void advance_shard_locked(PersistenceTaskState& task, PersistenceShardState& shard)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void maybe_report_status_locked(const PersistenceTaskState& task) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status ack_and_register_remote(
      const PersistenceTaskState& task,
      const PersistenceShardState& shard,
      PersistenceTargetState& target);
  static bool is_terminal(v1::PersistenceState state);
  static tensorcast::global_store::v1::PersistenceState to_global_state(v1::PersistenceState state);
  void persist_task_locked(const PersistenceTaskState& task) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void load_task_log();
  void update_counter_from_task_id(absl::string_view task_id) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void tick();
  void advance_locked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  LipManager* lip_mgr_; // Not owned.
  store::components::IGlobalStoreClient* global_store_{nullptr}; // Not owned.
  size_t artifact_chunk_bytes_;
  std::string local_node_id_;
  BackgroundScheduler* scheduler_; // Not owned.
  std::chrono::milliseconds tick_interval_;
  std::filesystem::path task_log_path_;

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, PersistenceTaskState> tasks_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::string> artifact_to_task_ ABSL_GUARDED_BY(mu_);
  std::atomic<uint64_t> counter_{0};
  absl::flat_hash_set<std::string> shared_disk_index_ ABSL_GUARDED_BY(mu_);
  std::atomic<bool> fail_shared_disk_for_test_{false};
};

} // namespace tensorcast::daemon
