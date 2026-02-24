// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
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
#include "core/common/async_runtime.h"
#include "core/common/memory/memory_location.h"
#include "core/store/components/global_store_client.h"
#include "core/store/device_types.h"
#include "daemon/state/background_scheduler.h"
#include "daemon/state/store_policy_resolver.h"
#include "daemon/state/types.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::store {
class StoreEngine;
} // namespace tensorcast::store

namespace tensorcast::daemon {

class LipManager;

struct DiskWriteState {
  std::atomic<uint64_t> bytes_written{0};
  uint64_t total_bytes{0};
  std::atomic<bool> done{false};
  std::string part_path;
  mutable absl::Mutex mu;
  absl::Status status ABSL_GUARDED_BY(mu){absl::OkStatus()};

  absl::Status get_status() const {
    absl::MutexLock lock(&mu);
    return status;
  }

  void set_status(absl::Status new_status) {
    absl::MutexLock lock(&mu);
    status = std::move(new_status);
  }
};

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
  std::shared_ptr<DiskWriteState> disk_write_state;
};

struct PersistenceShardState {
  std::string shard_id;
  uint32_t shard_idx{0};
  uint64_t byte_range_start{0};
  uint64_t byte_range_length{0};
  uint64_t size_bytes{0};
  std::string content_digest;
  std::vector<uint32_t> chunk_ids;
  v2::PersistenceState state{v2::PERSISTENCE_STATE_PENDING};
  double progress{0.0};
  std::string degraded_reason;
  std::string last_error;
  std::vector<PersistenceTargetState> targets;
};

struct PersistenceTaskState {
  std::string task_id;
  std::string plan_id;
  std::string artifact_id;
  std::string key_hint;
  v2::PlacementPolicy placement_policy{v2::PLACEMENT_POLICY_UNSPECIFIED};
  bool persist_to_shared_disk{false};
  RequirementLevel remote_requirement{RequirementLevel::kNone};
  RequirementLevel shared_disk_requirement{RequirementLevel::kNone};
  v2::PolicyLayout layout{v2::POLICY_LAYOUT_AUTO};
  v2::PersistenceState state{v2::PERSISTENCE_STATE_PENDING};
  double progress{0.0};
  std::string degraded_reason;
  std::string last_error;
  std::vector<PersistenceShardState> shards;
  uint64_t total_size_bytes{0};
  std::string disk_relative_path;
  bool disk_directory_ready{false};
  bool disk_metadata_written{false};
  bool disk_location_registered{false};
  bool by_key_linked{false};
  bool metrics_active{false};
  bool metrics_closed{false};
  uint64_t last_report_signature{0};
  int64_t last_report_ts_ns{0};
};

// Persistence task tracker with lightweight shard planning/state machine.
class PersistenceManager {
 public:
  struct PersistenceSource {
    std::string artifact_id;
    uint64_t total_size_bytes{0};
  };

  explicit PersistenceManager(
      BackgroundScheduler* scheduler,
      LipManager* lip_mgr,
      store::StoreEngine* store_engine,
      std::shared_ptr<common::AsyncRuntime> async_runtime,
      size_t artifact_chunk_bytes,
      std::chrono::milliseconds tick_interval = std::chrono::milliseconds(500),
      std::filesystem::path task_log_path = {});

  absl::StatusOr<PersistenceTaskState> start_task(
      std::string artifact_id,
      const ResolvedStorePolicy& policy,
      std::string_view key_hint = {});

  PersistenceTaskState start_task_for_test(
      std::string artifact_id,
      v2::PlacementPolicy placement_policy,
      bool persist_to_shared_disk,
      uint64_t total_size_bytes = 64ULL * 1024 * 1024) {
    ResolvedStorePolicy resolved;
    resolved.shared_disk_requirement = persist_to_shared_disk ? RequirementLevel::kMust : RequirementLevel::kNone;
    resolved.remote_requirement =
        placement_policy == v2::PLACEMENT_POLICY_LOCAL_ONLY ? RequirementLevel::kNone : RequirementLevel::kShould;
    if (placement_policy == v2::PLACEMENT_POLICY_SHARDED) {
      resolved.layout = v2::POLICY_LAYOUT_SHARDED;
    } else if (placement_policy == v2::PLACEMENT_POLICY_REPLICATED) {
      resolved.layout = v2::POLICY_LAYOUT_UNSHARDED;
    } else {
      resolved.layout = v2::POLICY_LAYOUT_AUTO;
    }
    PersistenceSource source;
    source.artifact_id = std::move(artifact_id);
    source.total_size_bytes = total_size_bytes;
    return start_task_with_source(std::move(source), resolved).value();
  }

  PersistenceTaskState start_task_for_test_with_policy(
      std::string artifact_id,
      const ResolvedStorePolicy& policy,
      uint64_t total_size_bytes = 64ULL * 1024 * 1024) {
    PersistenceSource source;
    source.artifact_id = std::move(artifact_id);
    source.total_size_bytes = total_size_bytes;
    return start_task_with_source(std::move(source), policy).value();
  }

  absl::optional<PersistenceTaskState> get_by_task_id(absl::string_view task_id) const;
  absl::optional<PersistenceTaskState> get_latest_for_artifact(absl::string_view artifact_id) const;

  // Tick progress forward (pending -> running -> success). Exposed for tests.
  void advance_once_for_test();

  void set_local_node_id(std::string node_id);
  void set_global_store_client(store::components::IGlobalStoreClient* client);
  void set_storage_path(std::filesystem::path storage_root);
  void set_max_concurrency(uint32_t max_concurrency);

  [[nodiscard]] bool is_spill_evictable(
      absl::string_view artifact_id,
      bool require_shared_disk,
      bool require_remote_stable) const;

  void set_fail_shared_disk_for_test(bool fail) {
    fail_shared_disk_for_test_.store(fail);
  }

 private:
  using PlacementPlanResult = store::components::PlacementPlanResult;
  using PlacementShardSpec = store::components::PlacementShardSpec;

  absl::StatusOr<PersistenceTaskState> start_task_with_source(
      PersistenceSource source,
      const ResolvedStorePolicy& policy,
      std::string_view key_hint = {});
  absl::StatusOr<PersistenceSource> resolve_source(std::string_view artifact_id) const;
  absl::StatusOr<PersistenceSource> stable_source(std::string_view artifact_id) const;
  static v2::PlacementPolicy select_placement_policy(const ResolvedStorePolicy& policy, uint64_t total_size_bytes);
  absl::StatusOr<std::vector<PersistenceShardState>> plan_shards(
      const PersistenceSource& source,
      v2::PolicyLayout layout) const;
  absl::Status apply_placement_plan(PersistenceTaskState& task, std::vector<PersistenceShardState>& shards);
  absl::StatusOr<PlacementPlanResult> request_plan(
      const PersistenceTaskState& task,
      const std::vector<PersistenceShardState>& shards) const;
  void propagate_degraded_reason(PersistenceTaskState& task) const;
  static double compute_task_progress(const PersistenceTaskState& task);
  void attach_shared_disk_targets(bool persist_to_shared_disk, std::vector<PersistenceShardState>& shards)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::StatusOr<std::filesystem::path> ensure_disk_directory(PersistenceTaskState& task)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::StatusOr<std::filesystem::path> disk_object_dir(const PersistenceTaskState& task) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status start_disk_write(
      PersistenceTaskState& task,
      const PersistenceShardState& shard,
      PersistenceTargetState& target) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status finalize_disk_directory(PersistenceTaskState& task) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status register_disk_location(PersistenceTaskState& task) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void advance_shard_locked(PersistenceTaskState& task, PersistenceShardState& shard)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void update_durability_locked(const PersistenceTaskState& task) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void maybe_report_status_locked(
      PersistenceTaskState& task,
      std::vector<store::components::PersistenceReport>& reports) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status ack_and_register_remote(
      const PersistenceTaskState& task,
      const PersistenceShardState& shard,
      PersistenceTargetState& target);
  static bool is_terminal(v2::PersistenceState state);
  static store::components::PersistenceReport build_report(const PersistenceTaskState& task);
  static tensorcast::global_store::v1::PersistenceState to_global_state(v2::PersistenceState state);
  void persist_task_locked(const PersistenceTaskState& task) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void load_task_log();
  void update_counter_from_task_id(absl::string_view task_id) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void tick();
  void advance_locked(std::vector<store::components::PersistenceReport>& reports) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  LipManager* lip_mgr_; // Not owned.
  store::StoreEngine* store_engine_{nullptr}; // Not owned.
  store::components::IGlobalStoreClient* global_store_{nullptr}; // Not owned.
  std::shared_ptr<common::AsyncRuntime> async_runtime_;
  size_t artifact_chunk_bytes_;
  std::atomic<uint32_t> max_concurrency_{4};
  std::string local_node_id_;
  BackgroundScheduler* scheduler_; // Not owned.
  std::chrono::milliseconds tick_interval_;
  std::filesystem::path task_log_path_;
  std::filesystem::path storage_root_;
  std::string cluster_id_;

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, PersistenceTaskState> tasks_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::string> artifact_to_task_ ABSL_GUARDED_BY(mu_);
  std::atomic<uint64_t> counter_{0};

  struct DurabilityState {
    bool shared_disk_complete{false};
    bool remote_stable_complete{false};
  };

  absl::flat_hash_map<std::string, DurabilityState> durability_index_ ABSL_GUARDED_BY(mu_);
  std::atomic<bool> fail_shared_disk_for_test_{false};
};

} // namespace tensorcast::daemon
