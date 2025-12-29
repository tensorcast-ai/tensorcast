// Copyright (c) 2025, TensorCast Team.

#include "daemon/persistence_manager.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/memory_state.h"
#include "core/store/store_engine.h"
#include "daemon/lip_manager.h"
#include "nlohmann/json.hpp"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

namespace gs = tensorcast::global_store::v1;
namespace comps = tensorcast::store::components;

namespace {
constexpr uint64_t kShardMinBytes = 64ULL * 1024 * 1024;
constexpr uint64_t kShardMaxBytes = 256ULL * 1024 * 1024;
constexpr uint64_t kShardThresholdBytes = 128ULL * 1024 * 1024;
constexpr uint64_t kDefaultChunkBytes = 4ULL * 1024 * 1024;
constexpr uint32_t kMaxLeaseAttempts = 3;
constexpr uint32_t kMaxCooldownTicks = 8;

std::string make_digest(absl::string_view artifact_id, uint32_t shard_idx, uint64_t start, uint64_t length) {
  const std::string payload = absl::StrCat(artifact_id, ":", shard_idx, ":", start, ":", length);
  const auto digest_bytes = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return absl::BytesToHexString(
      absl::string_view(reinterpret_cast<const char*>(digest_bytes.data()), digest_bytes.size()));
}

gs::PlacementPolicy to_global_policy(v1::PlacementPolicy policy) {
  switch (policy) {
    case v1::PLACEMENT_POLICY_LOCAL_ONLY:
      return gs::PLACEMENT_POLICY_LOCAL_ONLY;
    case v1::PLACEMENT_POLICY_REPLICATED:
      return gs::PLACEMENT_POLICY_REPLICATED;
    case v1::PLACEMENT_POLICY_SHARDED:
      return gs::PLACEMENT_POLICY_SHARDED;
    default:
      return gs::PLACEMENT_POLICY_UNSPECIFIED;
  }
}

void adjust_active_metric(int64_t delta) {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateInt64UpDownCounter("tc_persist_tasks_active");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    counter->Add(delta, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
    // Metrics must not affect control flow
  }
}

void record_error_metric(absl::string_view stage) {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_persist_errors_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    if (!stage.empty()) {
      attrs.emplace("stage", std::string(stage));
    }
    counter->Add(1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void record_progress_metric(double progress) {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto hist = meter->CreateDoubleHistogram("tc_persist_progress_ratio");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    hist->Record(progress, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void record_retry_metric(absl::string_view stage) {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_persist_retries_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    if (!stage.empty()) {
      attrs.emplace("stage", std::string(stage));
    }
    counter->Add(1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

uint32_t compute_backoff_ticks(uint32_t attempts) {
  const uint32_t shift = std::min<uint32_t>(attempts, 3);
  const uint32_t backoff = 1u << shift;
  return std::min<uint32_t>(backoff, kMaxCooldownTicks);
}

void send_reports(comps::IGlobalStoreClient* client, const std::vector<comps::PersistenceReport>& reports) {
  if (client == nullptr) {
    return;
  }
  for (const auto& report : reports) {
    const absl::Status st = client->report_persistence_status(report);
    if (!st.ok()) {
      LOG(WARNING) << "persistence.report_status_failed: " << st.message();
    }
  }
}

nlohmann::json target_to_json(const PersistenceTargetState& target) {
  nlohmann::json j;
  j["node_id"] = target.node_id;
  j["lease_id"] = target.lease_id;
  j["state"] = target.target_state;
  j["degraded_reason"] = target.degraded_reason;
  j["is_shared_disk"] = target.is_shared_disk;
  j["attempts"] = target.attempts;
  j["cooldown_ticks"] = target.cooldown_ticks;
  j["lease_acked"] = target.lease_acked;
  j["replica_registered"] = target.replica_registered;
  return j;
}

nlohmann::json shard_to_json(const PersistenceShardState& shard) {
  nlohmann::json j;
  j["shard_id"] = shard.shard_id;
  j["shard_idx"] = shard.shard_idx;
  j["byte_range_start"] = shard.byte_range_start;
  j["byte_range_length"] = shard.byte_range_length;
  j["size_bytes"] = shard.size_bytes;
  j["content_digest"] = shard.content_digest;
  j["chunk_ids"] = shard.chunk_ids;
  j["state"] = shard.state;
  j["progress"] = shard.progress;
  j["degraded_reason"] = shard.degraded_reason;
  j["last_error"] = shard.last_error;
  nlohmann::json targets = nlohmann::json::array();
  for (const auto& tgt : shard.targets) {
    targets.push_back(target_to_json(tgt));
  }
  j["targets"] = std::move(targets);
  return j;
}

nlohmann::json task_to_json(const PersistenceTaskState& task) {
  nlohmann::json j;
  j["task_id"] = task.task_id;
  j["plan_id"] = task.plan_id;
  j["artifact_id"] = task.artifact_id;
  j["placement_policy"] = task.placement_policy;
  j["persist_to_shared_disk"] = task.persist_to_shared_disk;
  j["remote_requirement"] = static_cast<int>(task.remote_requirement);
  j["shared_disk_requirement"] = static_cast<int>(task.shared_disk_requirement);
  j["layout"] = static_cast<int>(task.layout);
  j["state"] = task.state;
  j["progress"] = task.progress;
  j["degraded_reason"] = task.degraded_reason;
  j["last_error"] = task.last_error;
  nlohmann::json shards = nlohmann::json::array();
  for (const auto& shard : task.shards) {
    shards.push_back(shard_to_json(shard));
  }
  j["shards"] = std::move(shards);
  return j;
}

PersistenceTargetState target_from_json(const nlohmann::json& j) {
  PersistenceTargetState tgt;
  tgt.node_id = j.value("node_id", "");
  tgt.lease_id = j.value("lease_id", "");
  tgt.target_state = static_cast<gs::PlacementTargetState>(j.value("state", 0));
  tgt.degraded_reason = j.value("degraded_reason", "");
  tgt.is_shared_disk = j.value("is_shared_disk", false);
  tgt.attempts = j.value("attempts", 0);
  tgt.cooldown_ticks = j.value("cooldown_ticks", 0);
  tgt.lease_acked = j.value("lease_acked", false);
  tgt.replica_registered = j.value("replica_registered", false);
  return tgt;
}

absl::optional<PersistenceTaskState> task_from_json(const nlohmann::json& j) {
  try {
    PersistenceTaskState task;
    task.task_id = j.value("task_id", "");
    if (task.task_id.empty()) {
      return absl::nullopt;
    }
    task.plan_id = j.value("plan_id", "");
    task.artifact_id = j.value("artifact_id", "");
    task.placement_policy = static_cast<v1::PlacementPolicy>(j.value("placement_policy", 0));
    task.persist_to_shared_disk = j.value("persist_to_shared_disk", false);
    task.remote_requirement = static_cast<RequirementLevel>(j.value("remote_requirement", 0));
    task.shared_disk_requirement = static_cast<RequirementLevel>(j.value("shared_disk_requirement", 0));
    task.layout = static_cast<v1::PolicyLayout>(j.value("layout", v1::POLICY_LAYOUT_AUTO));
    task.state = static_cast<v1::PersistenceState>(j.value("state", 0));
    task.progress = j.value("progress", 0.0);
    task.degraded_reason = j.value("degraded_reason", "");
    task.last_error = j.value("last_error", "");
    if (j.contains("shards")) {
      for (const auto& shard_json : j.at("shards")) {
        PersistenceShardState shard;
        shard.shard_id = shard_json.value("shard_id", "");
        shard.shard_idx = shard_json.value("shard_idx", 0);
        shard.byte_range_start = shard_json.value("byte_range_start", 0);
        shard.byte_range_length = shard_json.value("byte_range_length", 0);
        shard.size_bytes = shard_json.value("size_bytes", 0);
        shard.content_digest = shard_json.value("content_digest", "");
        shard.state = static_cast<v1::PersistenceState>(shard_json.value("state", 0));
        shard.progress = shard_json.value("progress", 0.0);
        shard.degraded_reason = shard_json.value("degraded_reason", "");
        shard.last_error = shard_json.value("last_error", "");
        if (shard_json.contains("chunk_ids")) {
          for (const auto& cid : shard_json.at("chunk_ids")) {
            shard.chunk_ids.push_back(static_cast<uint32_t>(cid.get<uint64_t>()));
          }
        }
        if (shard_json.contains("targets")) {
          for (const auto& target_json : shard_json.at("targets")) {
            shard.targets.push_back(target_from_json(target_json));
          }
        }
        task.shards.push_back(std::move(shard));
      }
    }
    return task;
  } catch (...) {
    return absl::nullopt;
  }
}
} // namespace

PersistenceManager::PersistenceManager(
    BackgroundScheduler* scheduler,
    LipManager* lip_mgr,
    store::StoreEngine* store_engine,
    size_t artifact_chunk_bytes,
    std::chrono::milliseconds tick_interval,
    std::filesystem::path task_log_path)
    : lip_mgr_(lip_mgr),
      store_engine_(store_engine),
      global_store_(nullptr),
      artifact_chunk_bytes_(artifact_chunk_bytes),
      local_node_id_("local"),
      scheduler_(scheduler),
      tick_interval_(tick_interval),
      task_log_path_(std::move(task_log_path)) {
  if (scheduler_) {
    scheduler_->add_task(TaskKind::kPersistence, tick_interval_, [this]() { this->tick(); });
  }
  load_task_log();
}

absl::StatusOr<PersistenceTaskState> PersistenceManager::start_task(
    std::string artifact_id,
    const ResolvedStorePolicy& policy) {
  auto source_or = resolve_source(artifact_id);
  if (!source_or.ok()) {
    return source_or.status();
  }
  return start_task_with_source(std::move(*source_or), policy);
}

absl::StatusOr<PersistenceTaskState> PersistenceManager::start_task_with_source(
    PersistenceSource source,
    const ResolvedStorePolicy& policy) {
  const uint64_t id = ++counter_;
  const v1::PlacementPolicy placement_policy = select_placement_policy(policy, source.total_size_bytes);
  const bool persist_to_shared_disk = policy.shared_disk_requirement != RequirementLevel::kNone;
  PersistenceTaskState task{
      .task_id = absl::StrFormat("persist-%016x", id),
      .plan_id = absl::StrFormat("plan-%016x", id),
      .artifact_id = source.artifact_id,
      .placement_policy = placement_policy,
      .persist_to_shared_disk = persist_to_shared_disk,
      .remote_requirement = policy.remote_requirement,
      .shared_disk_requirement = policy.shared_disk_requirement,
      .layout = policy.layout,
      .state = v1::PERSISTENCE_STATE_PENDING,
      .progress = 0.0,
      .degraded_reason = "",
      .last_error = "",
  };

  auto shards_or = plan_shards(source, policy.layout);
  if (!shards_or.ok()) {
    task.state = v1::PERSISTENCE_STATE_FAILED;
    task.last_error = std::string(shards_or.status().message());
  } else {
    auto shards = std::move(*shards_or);
    const absl::Status plan_status = apply_placement_plan(task, shards);
    if (!plan_status.ok()) {
      task.state = v1::PERSISTENCE_STATE_FAILED;
      task.last_error = std::string(plan_status.message());
    }
    {
      absl::MutexLock lock(&mu_);
      attach_shared_disk_targets(task.persist_to_shared_disk, shards);
    }
    task.shards = std::move(shards);
  }
  propagate_degraded_reason(task);
  task.progress = compute_task_progress(task);
  if (task.state == v1::PERSISTENCE_STATE_FAILED) {
    record_error_metric("plan");
  }
  if (!task.metrics_active && task.state != v1::PERSISTENCE_STATE_FAILED) {
    adjust_active_metric(1);
    task.metrics_active = true;
  }

  {
    absl::MutexLock lock(&mu_);
    artifact_to_task_[task.artifact_id] = task.task_id;
    tasks_[task.task_id] = task;
    persist_task_locked(task);
  }
  if (scheduler_) {
    scheduler_->notify(TaskKind::kPersistence);
  }
  return task;
}

absl::StatusOr<PersistenceManager::PersistenceSource> PersistenceManager::resolve_source(
    std::string_view artifact_id) const {
  auto stable_or = stable_source(artifact_id);
  if (stable_or.ok()) {
    return stable_or;
  }
  if (lip_mgr_ != nullptr) {
    auto lease_opt = lip_mgr_->find_active_by_artifact_id(std::string(artifact_id));
    if (lease_opt.has_value()) {
      PersistenceSource source;
      source.artifact_id = lease_opt->artifact_id;
      source.total_size_bytes = lease_opt->total_size;
      return source;
    }
  }
  return absl::FailedPreconditionError(
      absl::StrCat("no stable DRAM replica or active lease for artifact: ", stable_or.status().message()));
}

absl::StatusOr<PersistenceManager::PersistenceSource> PersistenceManager::stable_source(
    std::string_view artifact_id) const {
  if (store_engine_ == nullptr) {
    return absl::FailedPreconditionError("store engine unavailable for persistence");
  }
  const auto devices = store_engine_->get_resident_devices(artifact_id);
  for (const auto& device : devices) {
    if (device.type != DeviceType::CPU) {
      continue;
    }
    store::loading::ReplicaKey key;
    key.artifact_id = std::string(artifact_id);
    key.device = device;
    key.replica = 0;
    if (store_engine_->get_replica_state(key, DeviceType::CPU) != store::replica::MemoryState::LOADED) {
      return absl::FailedPreconditionError("stable DRAM replica is not loaded");
    }
    auto size_or = store_engine_->get_replica_size(key);
    if (!size_or.ok()) {
      return size_or.status();
    }
    if (*size_or == 0) {
      return absl::FailedPreconditionError("stable DRAM replica size is 0");
    }
    PersistenceSource source;
    source.artifact_id = std::string(artifact_id);
    source.total_size_bytes = *size_or;
    return source;
  }
  return absl::FailedPreconditionError("stable DRAM replica not found for artifact");
}

v1::PlacementPolicy PersistenceManager::select_placement_policy(
    const ResolvedStorePolicy& policy,
    uint64_t total_size_bytes) {
  if (policy.remote_requirement == RequirementLevel::kNone) {
    return v1::PLACEMENT_POLICY_LOCAL_ONLY;
  }
  if (policy.layout == v1::POLICY_LAYOUT_SHARDED) {
    return v1::PLACEMENT_POLICY_SHARDED;
  }
  if (policy.layout == v1::POLICY_LAYOUT_UNSHARDED) {
    return v1::PLACEMENT_POLICY_REPLICATED;
  }
  return total_size_bytes >= kShardThresholdBytes ? v1::PLACEMENT_POLICY_SHARDED : v1::PLACEMENT_POLICY_REPLICATED;
}

absl::optional<PersistenceTaskState> PersistenceManager::get_by_task_id(absl::string_view task_id) const {
  absl::MutexLock lock(&mu_);
  auto it = tasks_.find(task_id);
  if (it == tasks_.end()) {
    return absl::nullopt;
  }
  return it->second;
}

absl::optional<PersistenceTaskState> PersistenceManager::get_latest_for_artifact(absl::string_view artifact_id) const {
  absl::MutexLock lock(&mu_);
  auto latest_it = artifact_to_task_.find(artifact_id);
  if (latest_it == artifact_to_task_.end()) {
    return absl::nullopt;
  }
  const auto task_it = tasks_.find(latest_it->second);
  if (task_it == tasks_.end()) {
    return absl::nullopt;
  }
  return task_it->second;
}

void PersistenceManager::advance_once_for_test() {
  comps::IGlobalStoreClient* gs_client = nullptr;
  std::vector<comps::PersistenceReport> reports;
  {
    absl::MutexLock lock(&mu_);
    gs_client = global_store_;
    advance_locked(reports);
  }
  send_reports(gs_client, reports);
}

void PersistenceManager::set_local_node_id(std::string node_id) {
  absl::MutexLock lock(&mu_);
  local_node_id_ = std::move(node_id);
}

void PersistenceManager::set_global_store_client(store::components::IGlobalStoreClient* client) {
  absl::MutexLock lock(&mu_);
  global_store_ = client;
}

absl::Status PersistenceManager::ack_and_register_remote(
    const PersistenceTaskState& task,
    const PersistenceShardState& shard,
    PersistenceTargetState& target) {
  if (global_store_ == nullptr) {
    return absl::FailedPreconditionError("global store unavailable for remote registration");
  }
  comps::MemoryTierLeaseAckPayload ack;
  ack.lease_id = target.lease_id;
  ack.node_id = target.node_id;
  ack.artifact_id = task.artifact_id;
  ack.chunk_ids.assign(shard.chunk_ids.begin(), shard.chunk_ids.end());
  ack.chunk_start = ack.chunk_ids.empty() ? 0u : shard.chunk_ids.front();
  ack.chunk_count = static_cast<uint32_t>(shard.chunk_ids.size());
  ack.bytes = shard.size_bytes;
  ack.request_id = task.task_id;
  auto ack_or = global_store_->acknowledge_memory_tier_lease(ack);
  if (!ack_or.ok()) {
    return ack_or.status();
  }
  target.lease_acked = true;

  tensorcast::store::DeviceKey device;
  device.type = tensorcast::DeviceType::REMOTE;
  device.ordinal = 0;
  device.uuid = target.node_id;
  auto reg_or = global_store_->register_replica(
      task.artifact_id,
      target.node_id,
      device,
      common::memory::MemoryLocation::CPU,
      shard.size_bytes,
      /*max_concurrency=*/1);
  if (!reg_or.ok()) {
    return reg_or.status();
  }
  target.replica_registered = true;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<PersistenceShardState>> PersistenceManager::plan_shards(
    const PersistenceSource& source,
    v1::PolicyLayout layout) const {
  const uint64_t chunk_bytes = artifact_chunk_bytes_ == 0 ? kDefaultChunkBytes : artifact_chunk_bytes_;
  if (chunk_bytes == 0) {
    return absl::FailedPreconditionError("artifact_chunk_bytes must be > 0");
  }
  if (source.total_size_bytes == 0) {
    return absl::FailedPreconditionError("artifact total_size is 0");
  }

  std::vector<PersistenceShardState> shards;
  const uint64_t chunk_count = (source.total_size_bytes + chunk_bytes - 1) / chunk_bytes;
  shards.reserve(chunk_count);

  auto append_shard = [&](uint64_t start, uint64_t length, std::vector<uint32_t>&& chunk_ids) {
    PersistenceShardState shard;
    shard.shard_idx = static_cast<uint32_t>(shards.size());
    shard.shard_id = absl::StrFormat("%s:%u", source.artifact_id, shard.shard_idx);
    shard.byte_range_start = start;
    shard.byte_range_length = length;
    shard.size_bytes = length;
    shard.content_digest = make_digest(source.artifact_id, shard.shard_idx, start, length);
    shard.chunk_ids = std::move(chunk_ids);
    shard.state = v1::PERSISTENCE_STATE_PENDING;
    shard.progress = 0.0;
    shards.push_back(std::move(shard));
  };

  const bool force_unsharded = layout == v1::POLICY_LAYOUT_UNSHARDED;
  const bool force_sharded = layout == v1::POLICY_LAYOUT_SHARDED;
  if (force_unsharded || (!force_sharded && source.total_size_bytes < kShardThresholdBytes)) {
    std::vector<uint32_t> chunk_ids;
    chunk_ids.reserve(static_cast<size_t>(chunk_count));
    for (uint32_t idx = 0; idx < chunk_count; ++idx) {
      chunk_ids.push_back(idx);
    }
    append_shard(0, source.total_size_bytes, std::move(chunk_ids));
    return shards;
  }

  uint64_t shard_start = 0;
  uint64_t shard_size = 0;
  std::vector<uint32_t> shard_chunks;
  shard_chunks.reserve(static_cast<size_t>(chunk_count));

  for (uint32_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
    const uint64_t chunk_start = static_cast<uint64_t>(chunk_idx) * chunk_bytes;
    const uint64_t chunk_len = std::min<uint64_t>(chunk_bytes, source.total_size_bytes - chunk_start);
    const bool should_flush =
        !shard_chunks.empty() && shard_size >= kShardMinBytes && shard_size + chunk_len > kShardMaxBytes;
    if (should_flush) {
      append_shard(shard_start, shard_size, std::move(shard_chunks));
      shard_chunks.clear();
      shard_size = 0;
      shard_start = chunk_start;
    }
    if (shard_chunks.empty()) {
      shard_start = chunk_start;
    }
    shard_chunks.push_back(chunk_idx);
    shard_size += chunk_len;
  }
  if (!shard_chunks.empty()) {
    append_shard(shard_start, shard_size, std::move(shard_chunks));
  }
  return shards;
}

absl::StatusOr<PersistenceManager::PlacementPlanResult> PersistenceManager::request_plan(
    const PersistenceTaskState& task,
    const std::vector<PersistenceShardState>& shards) const {
  std::string node_id_copy;
  comps::IGlobalStoreClient* gs_client = nullptr;
  {
    absl::MutexLock lock(&mu_);
    node_id_copy = local_node_id_;
    gs_client = global_store_;
  }
  if (gs_client == nullptr) {
    return absl::FailedPreconditionError("global store unavailable");
  }
  if (node_id_copy.empty()) {
    return absl::FailedPreconditionError("local node id unavailable for placement");
  }
  std::vector<PlacementShardSpec> specs;
  specs.reserve(shards.size());
  for (const auto& shard : shards) {
    PlacementShardSpec spec;
    spec.shard_id = shard.shard_id;
    spec.shard_idx = shard.shard_idx;
    spec.size_bytes = shard.size_bytes;
    spec.content_digest = shard.content_digest;
    spec.byte_range_start = shard.byte_range_start;
    spec.byte_range_length = shard.byte_range_length;
    spec.chunk_ids.assign(shard.chunk_ids.begin(), shard.chunk_ids.end());
    specs.push_back(std::move(spec));
  }
  return gs_client->plan_placement(task.artifact_id, to_global_policy(task.placement_policy), specs, node_id_copy);
}

absl::Status PersistenceManager::apply_placement_plan(
    PersistenceTaskState& task,
    std::vector<PersistenceShardState>& shards) {
  if (shards.empty()) {
    return absl::OkStatus();
  }

  auto plan_or = request_plan(task, shards);
  if (!plan_or.ok()) {
    if (task.remote_requirement == RequirementLevel::kMust) {
      task.state = v1::PERSISTENCE_STATE_FAILED;
      task.last_error = std::string(plan_or.status().message());
    } else if (task.remote_requirement == RequirementLevel::kShould) {
      task.degraded_reason = std::string(plan_or.status().message());
    } else if (task.last_error.empty() && task.remote_requirement == RequirementLevel::kNone) {
      task.last_error = std::string(plan_or.status().message());
    }
    LOG(WARNING) << "persistence.plan.fallback: artifact_id=" << task.artifact_id << " status=" << plan_or.status();
    std::string node_id_copy;
    {
      absl::MutexLock lock(&mu_);
      node_id_copy = local_node_id_;
    }
    for (auto& shard : shards) {
      PersistenceTargetState target;
      target.node_id = node_id_copy.empty() ? "local" : node_id_copy;
      shard.targets.push_back(std::move(target));
    }
    return absl::OkStatus();
  }

  const auto& plan = *plan_or;
  if (!plan.plan_id.empty()) {
    task.plan_id = plan.plan_id;
  }
  if (plan.degraded && task.degraded_reason.empty()) {
    if (task.remote_requirement == RequirementLevel::kMust) {
      task.state = v1::PERSISTENCE_STATE_FAILED;
      task.last_error = plan.degraded_reason.empty() ? "placement_plan_degraded" : plan.degraded_reason;
    } else if (task.remote_requirement == RequirementLevel::kShould) {
      task.degraded_reason = plan.degraded_reason;
    }
  }

  for (const auto& placement : plan.placements) {
    if (placement.shard.shard_idx >= shards.size()) {
      continue;
    }
    auto& shard = shards[placement.shard.shard_idx];
    if (!placement.shard.shard_id.empty()) {
      shard.shard_id = placement.shard.shard_id;
    }
    shard.degraded_reason = placement.degraded_reason.empty() ? shard.degraded_reason : placement.degraded_reason;
    shard.targets.clear();
    shard.targets.reserve(placement.targets.size());
    for (const auto& target : placement.targets) {
      PersistenceTargetState tgt;
      tgt.node_id = target.node_id;
      tgt.lease_id = target.lease_id;
      tgt.target_state = target.target_state;
      tgt.degraded_reason = target.degraded_reason;
      shard.targets.push_back(std::move(tgt));
    }
  }

  // Ensure every shard has at least one target (fallback to local)
  std::string node_id_copy;
  {
    absl::MutexLock lock(&mu_);
    node_id_copy = local_node_id_;
  }
  for (auto& shard : shards) {
    if (shard.targets.empty()) {
      PersistenceTargetState target;
      target.node_id = node_id_copy.empty() ? "local" : node_id_copy;
      shard.targets.push_back(std::move(target));
    }
  }
  return absl::OkStatus();
}

void PersistenceManager::attach_shared_disk_targets(
    bool persist_to_shared_disk,
    std::vector<PersistenceShardState>& shards) {
  if (!persist_to_shared_disk) {
    return;
  }
  std::string node_id_copy = local_node_id_.empty() ? "local" : local_node_id_;
  for (auto& shard : shards) {
    PersistenceTargetState disk_target;
    disk_target.node_id = node_id_copy;
    disk_target.target_state = gs::PLACEMENT_TARGET_STATE_PENDING;
    disk_target.is_shared_disk = true;
    shard.targets.push_back(std::move(disk_target));
  }
}

void PersistenceManager::propagate_degraded_reason(PersistenceTaskState& task) const {
  if (task.state == v1::PERSISTENCE_STATE_FAILED) {
    return;
  }
  if (task.degraded_reason.empty()) {
    for (const auto& shard : task.shards) {
      if (!shard.degraded_reason.empty()) {
        task.degraded_reason = shard.degraded_reason;
        break;
      }
    }
  }
  if (!task.degraded_reason.empty()) {
    for (auto& shard : task.shards) {
      if (shard.degraded_reason.empty()) {
        shard.degraded_reason = task.degraded_reason;
      }
    }
  }
}

double PersistenceManager::compute_task_progress(const PersistenceTaskState& task) {
  if (task.shards.empty()) {
    return task.state == v1::PERSISTENCE_STATE_SUCCESS ? 1.0 : 0.0;
  }
  double accum = 0.0;
  for (const auto& shard : task.shards) {
    accum += shard.progress;
  }
  return accum / static_cast<double>(task.shards.size());
}

void PersistenceManager::advance_shard_locked(PersistenceTaskState& task, PersistenceShardState& shard) {
  if (is_terminal(shard.state)) {
    return;
  }
  size_t done_targets = 0;
  const size_t total_targets = shard.targets.size();
  bool shard_failed = false;
  bool shard_degraded = !shard.degraded_reason.empty();
  bool remote_seen = false;
  bool remote_failed = false;
  bool remote_success = false;
  bool disk_seen = false;
  bool disk_failed = false;
  bool disk_success = false;
  std::string remote_failure_reason;
  std::string disk_failure_reason;
  const bool remote_required = task.remote_requirement == RequirementLevel::kMust;
  const bool remote_should = task.remote_requirement == RequirementLevel::kShould;
  const bool disk_required = task.shared_disk_requirement == RequirementLevel::kMust;
  const bool disk_should = task.shared_disk_requirement == RequirementLevel::kShould;

  for (auto& target : shard.targets) {
    if (target.is_shared_disk) {
      disk_seen = true;
      if (target.target_state == gs::PLACEMENT_TARGET_STATE_COMPLETE ||
          target.target_state == gs::PLACEMENT_TARGET_STATE_SKIPPED) {
        ++done_targets;
        disk_success = true;
        continue;
      }
      if (target.target_state == gs::PLACEMENT_TARGET_STATE_FAILED) {
        disk_failed = true;
        if (disk_failure_reason.empty()) {
          disk_failure_reason = target.degraded_reason;
        }
        ++done_targets;
        continue;
      }
      bool dedup_hit = false;
      {
        dedup_hit = shared_disk_index_.contains(shard.content_digest);
      }
      if (dedup_hit) {
        target.target_state = gs::PLACEMENT_TARGET_STATE_SKIPPED;
        ++done_targets;
        disk_success = true;
        continue;
      }
      if (target.target_state == gs::PLACEMENT_TARGET_STATE_PENDING) {
        target.target_state = gs::PLACEMENT_TARGET_STATE_COPYING;
        continue;
      }
      if (fail_shared_disk_for_test_.load()) {
        target.target_state = gs::PLACEMENT_TARGET_STATE_FAILED;
        target.degraded_reason = "shared_disk_write_failed";
        disk_failed = true;
        if (disk_failure_reason.empty()) {
          disk_failure_reason = target.degraded_reason;
        }
        ++done_targets;
        continue;
      }
      // COPY->COMPLETE on the next tick to simulate async disk writes.
      target.target_state = gs::PLACEMENT_TARGET_STATE_COMPLETE;
      {
        shared_disk_index_.insert(shard.content_digest);
      }
      ++done_targets;
      disk_success = true;
      continue;
    }

    const bool is_remote = !target.node_id.empty() && target.node_id != local_node_id_;
    if (is_remote) {
      remote_seen = true;
    }
    if (target.target_state == gs::PLACEMENT_TARGET_STATE_COMPLETE ||
        target.target_state == gs::PLACEMENT_TARGET_STATE_SKIPPED) {
      ++done_targets;
      if (is_remote) {
        remote_success = true;
      }
      continue;
    }
    if (target.target_state == gs::PLACEMENT_TARGET_STATE_FAILED) {
      if (target.attempts >= kMaxLeaseAttempts) {
        remote_failed = true;
        if (remote_failure_reason.empty()) {
          remote_failure_reason = target.degraded_reason;
        }
        ++done_targets;
        continue;
      }
      if (target.cooldown_ticks > 0) {
        --target.cooldown_ticks;
        continue;
      }
      target.target_state = gs::PLACEMENT_TARGET_STATE_PENDING;
    }
    if (target.cooldown_ticks > 0) {
      --target.cooldown_ticks;
      continue;
    }
    if (target.target_state == gs::PLACEMENT_TARGET_STATE_COPYING) {
      absl::Status reg_status = absl::OkStatus();
      if (is_remote) {
        reg_status = ack_and_register_remote(task, shard, target);
      }
      if (!reg_status.ok()) {
        target.target_state = gs::PLACEMENT_TARGET_STATE_FAILED;
        target.degraded_reason = std::string(reg_status.message());
        record_retry_metric("register");
        ++target.attempts;
        target.cooldown_ticks = compute_backoff_ticks(target.attempts);
        if (target.attempts >= kMaxLeaseAttempts) {
          remote_failed = true;
          if (remote_failure_reason.empty()) {
            remote_failure_reason = target.degraded_reason;
          }
          ++done_targets;
        }
        continue;
      }
      target.target_state = gs::PLACEMENT_TARGET_STATE_COMPLETE;
      ++done_targets;
      if (is_remote) {
        remote_success = true;
      }
      continue;
    }
    if (target.node_id.empty()) {
      target.target_state = gs::PLACEMENT_TARGET_STATE_FAILED;
      target.degraded_reason = "missing_target";
      shard_failed = true;
      if (remote_failure_reason.empty()) {
        remote_failure_reason = target.degraded_reason;
      }
      ++done_targets;
      continue;
    }
    if (target.node_id == local_node_id_) {
      target.target_state = gs::PLACEMENT_TARGET_STATE_COMPLETE;
      ++done_targets;
      continue;
    }
    if (global_store_ == nullptr) {
      target.target_state = gs::PLACEMENT_TARGET_STATE_FAILED;
      target.degraded_reason = "global_store_unavailable";
      remote_failed = true;
      if (remote_failure_reason.empty()) {
        remote_failure_reason = target.degraded_reason;
      }
      ++done_targets;
      continue;
    }
    comps::MemoryTierLeaseDescriptor lease_req;
    lease_req.node_id = target.node_id;
    lease_req.kind = comps::MemoryTierLeaseKind::kStable;
    lease_req.artifact_id = task.artifact_id;
    lease_req.chunk_ids = shard.chunk_ids;
    lease_req.chunk_start = shard.chunk_ids.empty() ? 0u : shard.chunk_ids.front();
    lease_req.chunk_count = static_cast<uint32_t>(shard.chunk_ids.size());
    lease_req.bytes = shard.size_bytes;
    lease_req.workload_id = task.task_id;
    auto lease_or = global_store_->request_memory_tier_lease(lease_req);
    if (!lease_or.ok()) {
      target.target_state = gs::PLACEMENT_TARGET_STATE_FAILED;
      target.degraded_reason = "lease_denied";
      if (remote_failure_reason.empty()) {
        remote_failure_reason = target.degraded_reason;
      }
      record_retry_metric("lease");
      ++target.attempts;
      target.cooldown_ticks = compute_backoff_ticks(target.attempts);
      if (target.attempts >= kMaxLeaseAttempts) {
        remote_failed = true;
        if (remote_failure_reason.empty()) {
          remote_failure_reason = target.degraded_reason;
        }
        ++done_targets;
      }
      continue;
    }
    target.lease_id = lease_or->lease_id;
    target.target_state = gs::PLACEMENT_TARGET_STATE_COPYING;
    target.cooldown_ticks = 0;
  }

  shard.progress = total_targets == 0 ? 1.0 : static_cast<double>(done_targets) / static_cast<double>(total_targets);

  if ((remote_required || remote_should) && !remote_seen) {
    if (remote_failure_reason.empty()) {
      remote_failure_reason = "remote_target_missing";
    }
    if (remote_required) {
      shard_failed = true;
    } else {
      shard_degraded = true;
    }
  }
  if ((disk_required || disk_should) && !disk_seen) {
    if (disk_failure_reason.empty()) {
      disk_failure_reason = "shared_disk_target_missing";
    }
    if (disk_required) {
      shard_failed = true;
    } else {
      shard_degraded = true;
    }
  }

  if (remote_failed && !remote_success) {
    if (remote_required) {
      shard_failed = true;
    } else if (remote_should) {
      shard_degraded = true;
    }
  }
  if (disk_failed && !disk_success) {
    if (disk_required) {
      shard_failed = true;
    } else if (disk_should) {
      shard_degraded = true;
    }
  }

  if (shard_failed && shard.last_error.empty()) {
    if (!remote_failure_reason.empty()) {
      shard.last_error = remote_failure_reason;
    } else if (!disk_failure_reason.empty()) {
      shard.last_error = disk_failure_reason;
    } else {
      shard.last_error = "target_failed";
    }
  }
  if (shard_degraded && shard.degraded_reason.empty()) {
    if (!remote_failure_reason.empty()) {
      shard.degraded_reason = remote_failure_reason;
    } else if (!disk_failure_reason.empty()) {
      shard.degraded_reason = disk_failure_reason;
    } else {
      shard.degraded_reason = "target_failed";
    }
  }
  if (shard_failed) {
    shard.state = v1::PERSISTENCE_STATE_FAILED;
    return;
  }
  const bool all_done = done_targets == total_targets;
  if (all_done) {
    shard.state = (shard_degraded || !task.degraded_reason.empty()) ? v1::PERSISTENCE_STATE_DEGRADED
                                                                    : v1::PERSISTENCE_STATE_SUCCESS;
  } else {
    shard.state = v1::PERSISTENCE_STATE_RUNNING;
  }
}

void PersistenceManager::maybe_report_status_locked(
    const PersistenceTaskState& task,
    std::vector<comps::PersistenceReport>& reports) {
  if (global_store_ == nullptr) {
    return;
  }
  reports.push_back(build_report(task));
}

store::components::PersistenceReport PersistenceManager::build_report(const PersistenceTaskState& task) {
  comps::PersistenceReport report;
  report.task_id = task.task_id;
  report.plan_id = task.plan_id;
  report.artifact_id = task.artifact_id;
  report.state = to_global_state(task.state);
  report.progress = task.progress;
  report.last_error = task.last_error;
  report.degraded_reason = task.degraded_reason;

  for (const auto& shard : task.shards) {
    comps::PersistenceShardReport shard_report;
    shard_report.shard_id = shard.shard_id;
    shard_report.shard_idx = shard.shard_idx;
    shard_report.state = to_global_state(shard.state);
    shard_report.progress = shard.progress;
    shard_report.degraded_reason = shard.degraded_reason;
    shard_report.last_error = shard.last_error;
    for (const auto& target : shard.targets) {
      comps::PlacementTargetStatus tgt;
      tgt.node_id = target.node_id;
      tgt.lease_id = target.lease_id;
      tgt.target_state = target.target_state;
      tgt.degraded_reason = target.degraded_reason;
      shard_report.targets.push_back(std::move(tgt));
    }
    report.shards.push_back(std::move(shard_report));
  }

  return report;
}

void PersistenceManager::persist_task_locked(const PersistenceTaskState& task) const {
  if (task_log_path_.empty()) {
    return;
  }
  std::error_code ec;
  const auto parent = task_log_path_.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
  }
  std::ofstream out(task_log_path_, std::ios::app);
  if (!out.is_open()) {
    LOG(WARNING) << "persistence.log.write_failed path=" << task_log_path_.string();
    return;
  }
  out << task_to_json(task).dump() << "\n";
}

void PersistenceManager::update_counter_from_task_id(absl::string_view task_id) {
  static constexpr absl::string_view kPrefix = "persist-";
  if (task_id.size() <= kPrefix.size() || !absl::StartsWith(task_id, kPrefix)) {
    return;
  }
  const std::string hex(task_id.substr(kPrefix.size()));
  uint64_t parsed = 0;
  std::stringstream ss;
  ss << std::hex << hex;
  if (!(ss >> parsed)) {
    return;
  }
  const uint64_t current = counter_.load();
  if (parsed > current) {
    counter_.store(parsed);
  }
}

void PersistenceManager::load_task_log() {
  if (task_log_path_.empty()) {
    return;
  }
  std::ifstream in(task_log_path_);
  if (!in.is_open()) {
    return;
  }

  std::string line;
  absl::flat_hash_map<std::string, PersistenceTaskState> recovered;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    auto json = nlohmann::json::parse(line, nullptr, false);
    if (json.is_discarded()) {
      continue;
    }
    auto task_opt = task_from_json(json);
    if (!task_opt.has_value()) {
      continue;
    }
    recovered[task_opt->task_id] = std::move(*task_opt);
  }
  in.close();

  bool notify_scheduler = false;
  {
    absl::MutexLock lock(&mu_);
    for (auto& entry : recovered) {
      update_counter_from_task_id(entry.second.task_id);
      artifact_to_task_[entry.second.artifact_id] = entry.second.task_id;
      if (!is_terminal(entry.second.state) && !entry.second.metrics_active) {
        adjust_active_metric(1);
        entry.second.metrics_active = true;
        notify_scheduler = true;
      } else if (is_terminal(entry.second.state)) {
        entry.second.metrics_closed = true;
      }
      for (const auto& shard : entry.second.shards) {
        for (const auto& target : shard.targets) {
          if (target.is_shared_disk &&
              (target.target_state == gs::PLACEMENT_TARGET_STATE_COMPLETE ||
               target.target_state == gs::PLACEMENT_TARGET_STATE_SKIPPED)) {
            shared_disk_index_.insert(shard.content_digest);
          }
        }
      }
      tasks_[entry.first] = std::move(entry.second);
      update_durability_locked(tasks_[entry.first]);
    }
  }
  if (scheduler_ != nullptr && notify_scheduler) {
    scheduler_->notify(TaskKind::kPersistence);
  }
}

bool PersistenceManager::is_terminal(v1::PersistenceState state) {
  return state == v1::PERSISTENCE_STATE_SUCCESS || state == v1::PERSISTENCE_STATE_FAILED ||
      state == v1::PERSISTENCE_STATE_DEGRADED;
}

bool PersistenceManager::is_spill_evictable(
    absl::string_view artifact_id,
    bool require_shared_disk,
    bool require_remote_stable) const {
  absl::MutexLock lock(&mu_);
  auto it = durability_index_.find(std::string(artifact_id));
  if (it == durability_index_.end()) {
    return false;
  }
  const auto& state = it->second;
  if (!state.shared_disk_complete && !state.remote_stable_complete) {
    return false;
  }
  if (require_shared_disk && !state.shared_disk_complete) {
    return false;
  }
  if (require_remote_stable && !state.remote_stable_complete) {
    return false;
  }
  return true;
}

gs::PersistenceState PersistenceManager::to_global_state(v1::PersistenceState state) {
  switch (state) {
    case v1::PERSISTENCE_STATE_PENDING:
      return gs::PERSISTENCE_STATE_PENDING;
    case v1::PERSISTENCE_STATE_RUNNING:
      return gs::PERSISTENCE_STATE_RUNNING;
    case v1::PERSISTENCE_STATE_DEGRADED:
      return gs::PERSISTENCE_STATE_DEGRADED;
    case v1::PERSISTENCE_STATE_SUCCESS:
      return gs::PERSISTENCE_STATE_SUCCESS;
    case v1::PERSISTENCE_STATE_FAILED:
      return gs::PERSISTENCE_STATE_FAILED;
    default:
      return gs::PERSISTENCE_STATE_UNSPECIFIED;
  }
}

void PersistenceManager::tick() {
  comps::IGlobalStoreClient* gs_client = nullptr;
  std::vector<comps::PersistenceReport> reports;
  {
    absl::MutexLock lock(&mu_);
    gs_client = global_store_;
    advance_locked(reports);
  }
  send_reports(gs_client, reports);
}

void PersistenceManager::advance_locked(std::vector<comps::PersistenceReport>& reports) {
  for (auto& entry : tasks_) {
    auto& task = entry.second;
    if (is_terminal(task.state)) {
      continue;
    }
    if (task.state == v1::PERSISTENCE_STATE_PENDING) {
      task.state = v1::PERSISTENCE_STATE_RUNNING;
      for (auto& shard : task.shards) {
        shard.state = v1::PERSISTENCE_STATE_RUNNING;
        shard.progress = 0.0;
      }
      propagate_degraded_reason(task);
      task.progress = compute_task_progress(task);
      record_progress_metric(task.progress);
      maybe_report_status_locked(task, reports);
      continue;
    }
    bool any_failed = false;
    bool any_degraded = !task.degraded_reason.empty();
    bool all_done = true;

    for (auto& shard : task.shards) {
      advance_shard_locked(task, shard);
      if (shard.state == v1::PERSISTENCE_STATE_FAILED) {
        any_failed = true;
      }
      if (shard.state == v1::PERSISTENCE_STATE_DEGRADED || !shard.degraded_reason.empty()) {
        any_degraded = true;
      }
      if (shard.state != v1::PERSISTENCE_STATE_SUCCESS && shard.state != v1::PERSISTENCE_STATE_DEGRADED) {
        all_done = false;
      }
    }
    update_durability_locked(task);

    if (any_failed && task.last_error.empty()) {
      task.last_error = "one or more persistence shards failed";
    }
    propagate_degraded_reason(task);
    if (any_failed) {
      task.state = v1::PERSISTENCE_STATE_FAILED;
    } else if (all_done) {
      task.state = any_degraded ? v1::PERSISTENCE_STATE_DEGRADED : v1::PERSISTENCE_STATE_SUCCESS;
    } else {
      task.state = v1::PERSISTENCE_STATE_RUNNING;
    }
    task.progress = compute_task_progress(task);
    if (is_terminal(task.state) && task.metrics_active && !task.metrics_closed) {
      adjust_active_metric(-1);
      task.metrics_closed = true;
      if (task.state == v1::PERSISTENCE_STATE_FAILED) {
        record_error_metric("task");
      }
    }
    record_progress_metric(task.progress);
    maybe_report_status_locked(task, reports);
    persist_task_locked(task);
  }
}

void PersistenceManager::update_durability_locked(const PersistenceTaskState& task) {
  bool shared_disk_complete = false;
  bool remote_stable_complete = false;
  if (task.persist_to_shared_disk && !task.shards.empty()) {
    shared_disk_complete = true;
    for (const auto& shard : task.shards) {
      bool has_disk_target = false;
      bool shard_complete = false;
      for (const auto& target : shard.targets) {
        if (!target.is_shared_disk) {
          continue;
        }
        has_disk_target = true;
        if (target.target_state == gs::PLACEMENT_TARGET_STATE_COMPLETE ||
            target.target_state == gs::PLACEMENT_TARGET_STATE_SKIPPED) {
          shard_complete = true;
          break;
        }
      }
      if (!has_disk_target || !shard_complete) {
        shared_disk_complete = false;
        break;
      }
    }
  }

  if (task.remote_requirement != RequirementLevel::kNone && !task.shards.empty()) {
    remote_stable_complete = true;
    for (const auto& shard : task.shards) {
      bool has_remote_target = false;
      bool shard_complete = false;
      for (const auto& target : shard.targets) {
        if (target.is_shared_disk) {
          continue;
        }
        const bool is_remote = !target.node_id.empty() && target.node_id != local_node_id_;
        if (!is_remote) {
          continue;
        }
        has_remote_target = true;
        if (target.target_state == gs::PLACEMENT_TARGET_STATE_COMPLETE ||
            target.target_state == gs::PLACEMENT_TARGET_STATE_SKIPPED) {
          shard_complete = true;
          break;
        }
      }
      if (!has_remote_target || !shard_complete) {
        remote_stable_complete = false;
        break;
      }
    }
  }

  if (!shared_disk_complete && !remote_stable_complete) {
    return;
  }
  auto& entry = durability_index_[task.artifact_id];
  if (shared_disk_complete) {
    entry.shared_disk_complete = true;
  }
  if (remote_stable_complete) {
    entry.remote_stable_complete = true;
  }
}

} // namespace tensorcast::daemon
