// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/persistence_manager.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/memory_state.h"
#include "core/store/store_engine.h"
#include "daemon/state/lip_manager.h"
#include "folly/futures/Future.h"
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

enum class DiskFailureStage { kWrite, kFinalize, kRegister };

std::string make_digest(absl::string_view artifact_id, uint32_t shard_idx, uint64_t start, uint64_t length) {
  const std::string payload = absl::StrCat(artifact_id, ":", shard_idx, ":", start, ":", length);
  const auto digest_bytes = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return absl::BytesToHexString(
      absl::string_view(reinterpret_cast<const char*>(digest_bytes.data()), digest_bytes.size()));
}

gs::PlacementPolicy to_global_policy(v2::PlacementPolicy policy) {
  switch (policy) {
    case v2::PLACEMENT_POLICY_LOCAL_ONLY:
      return gs::PLACEMENT_POLICY_LOCAL_ONLY;
    case v2::PLACEMENT_POLICY_REPLICATED:
      return gs::PLACEMENT_POLICY_REPLICATED;
    case v2::PLACEMENT_POLICY_SHARDED:
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

constexpr std::string_view kClustersDir = "clusters";
constexpr std::string_view kObjectsDir = "objects";
constexpr std::string_view kByKeyDir = "by_key";

std::string sanitize_segment(std::string_view input) {
  std::string out;
  out.reserve(input.size());
  for (char c : input) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  if (out.empty()) {
    out = "unnamed";
  }
  return out;
}

absl::StatusOr<std::pair<std::string, std::string>> parse_mi2_identity(std::string_view artifact_id) {
  if (!absl::StartsWith(artifact_id, "mi2:")) {
    return absl::InvalidArgumentError("artifact_id must start with mi2:");
  }
  const std::string_view rest = artifact_id.substr(4);
  const auto pos = rest.find(':');
  if (pos == std::string_view::npos) {
    return absl::InvalidArgumentError("artifact_id must be of form mi2:<index_multihash>:<data_multihash>");
  }
  const std::string_view index_mh = rest.substr(0, pos);
  const std::string_view data_mh = rest.substr(pos + 1);
  if (index_mh.empty() || data_mh.empty()) {
    return absl::InvalidArgumentError("artifact_id must include index and data multihash");
  }
  return std::make_pair(std::string(index_mh), std::string(data_mh));
}

std::string short_artifact_hint(std::string_view artifact_id) {
  auto parsed = parse_mi2_identity(artifact_id);
  if (parsed.ok()) {
    const std::string& data_mh = parsed->second;
    return sanitize_segment(data_mh.substr(0, std::min<size_t>(12, data_mh.size())));
  }
  return sanitize_segment(std::string(artifact_id).substr(0, 12));
}

std::string format_persist_timestamp(absl::Time now) {
  const std::string base = absl::FormatTime("%Y%m%dT%H%M%S", now, absl::UTCTimeZone());
  const int64_t millis = absl::ToUnixMillis(now) % 1000;
  return std::format("{}{:03d}Z", base, static_cast<int>(millis));
}

uint32_t compute_backoff_ticks(uint32_t attempts) {
  const uint32_t shift = std::min<uint32_t>(attempts, 3);
  const uint32_t backoff = 1u << shift;
  return std::min<uint32_t>(backoff, kMaxCooldownTicks);
}

bool is_safe_relative_path(const std::filesystem::path& path) {
  if (path.empty() || path.is_absolute()) {
    return false;
  }
  for (const auto& part : path) {
    if (part == "." || part == "..") {
      return false;
    }
  }
  return true;
}

bool has_path_prefix(const std::filesystem::path& path, const std::filesystem::path& prefix) {
  auto path_it = path.begin();
  for (auto prefix_it = prefix.begin(); prefix_it != prefix.end(); ++prefix_it) {
    if (path_it == path.end() || *path_it != *prefix_it) {
      return false;
    }
    ++path_it;
  }
  return true;
}

std::string disk_failure_token(const absl::Status& status, DiskFailureStage stage) {
  const std::string_view message = status.message();
  if (absl::StartsWith(message, "storage_path_missing")) {
    return "storage_path_missing";
  }
  if (absl::StartsWith(message, "cluster_id_missing")) {
    return "cluster_id_missing";
  }
  if (absl::StartsWith(message, "disk_relative_path_missing")) {
    return "disk_relative_path_missing";
  }
  if (stage == DiskFailureStage::kRegister) {
    return "disk_location_upsert_failed";
  }
  return "shared_disk_write_failed";
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
    task.placement_policy = static_cast<v2::PlacementPolicy>(j.value("placement_policy", 0));
    task.persist_to_shared_disk = j.value("persist_to_shared_disk", false);
    task.remote_requirement = static_cast<RequirementLevel>(j.value("remote_requirement", 0));
    task.shared_disk_requirement = static_cast<RequirementLevel>(j.value("shared_disk_requirement", 0));
    task.layout = static_cast<v2::PolicyLayout>(j.value("layout", v2::POLICY_LAYOUT_AUTO));
    task.state = static_cast<v2::PersistenceState>(j.value("state", 0));
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
        shard.state = static_cast<v2::PersistenceState>(shard_json.value("state", 0));
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
    std::shared_ptr<common::AsyncRuntime> async_runtime,
    size_t artifact_chunk_bytes,
    std::chrono::milliseconds tick_interval,
    std::filesystem::path task_log_path)
    : lip_mgr_(lip_mgr),
      store_engine_(store_engine),
      global_store_(nullptr),
      async_runtime_(async_runtime ? std::move(async_runtime) : std::make_shared<common::AsyncRuntime>()),
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
    const ResolvedStorePolicy& policy,
    std::string_view key_hint) {
  auto source_or = resolve_source(artifact_id);
  if (!source_or.ok()) {
    return source_or.status();
  }
  return start_task_with_source(std::move(*source_or), policy, key_hint);
}

absl::StatusOr<PersistenceTaskState> PersistenceManager::start_task_with_source(
    PersistenceSource source,
    const ResolvedStorePolicy& policy,
    std::string_view key_hint) {
  const uint64_t id = ++counter_;
  std::filesystem::path storage_root_copy;
  std::string cluster_id_copy;
  {
    absl::MutexLock lock(&mu_);
    storage_root_copy = storage_root_;
    cluster_id_copy = cluster_id_;
  }
  const v2::PlacementPolicy placement_policy = select_placement_policy(policy, source.total_size_bytes);
  const bool persist_to_shared_disk = policy.shared_disk_requirement != RequirementLevel::kNone;
  PersistenceTaskState task{
      .task_id = absl::StrFormat("persist-%016x", id),
      .plan_id = absl::StrFormat("plan-%016x", id),
      .artifact_id = source.artifact_id,
      .key_hint = std::string(key_hint),
      .placement_policy = placement_policy,
      .persist_to_shared_disk = persist_to_shared_disk,
      .remote_requirement = policy.remote_requirement,
      .shared_disk_requirement = policy.shared_disk_requirement,
      .layout = policy.layout,
      .state = v2::PERSISTENCE_STATE_PENDING,
      .progress = 0.0,
      .degraded_reason = "",
      .last_error = "",
      .total_size_bytes = source.total_size_bytes,
  };

  const bool shared_disk_required = policy.shared_disk_requirement == RequirementLevel::kMust;
  if (shared_disk_required && storage_root_copy.empty()) {
    task.state = v2::PERSISTENCE_STATE_FAILED;
    task.last_error = "storage_path_missing";
  } else if (shared_disk_required && cluster_id_copy.empty()) {
    task.state = v2::PERSISTENCE_STATE_FAILED;
    task.last_error = "cluster_id_missing";
  } else {
    auto shards_or = plan_shards(source, policy.layout);
    if (!shards_or.ok()) {
      task.state = v2::PERSISTENCE_STATE_FAILED;
      task.last_error = std::string(shards_or.status().message());
    } else {
      auto shards = std::move(*shards_or);
      const absl::Status plan_status = apply_placement_plan(task, shards);
      if (!plan_status.ok()) {
        task.state = v2::PERSISTENCE_STATE_FAILED;
        task.last_error = std::string(plan_status.message());
      }
      {
        absl::MutexLock lock(&mu_);
        attach_shared_disk_targets(task.persist_to_shared_disk, shards);
      }
      task.shards = std::move(shards);
    }
  }
  propagate_degraded_reason(task);
  task.progress = compute_task_progress(task);
  if (task.state == v2::PERSISTENCE_STATE_FAILED) {
    record_error_metric("plan");
  }
  if (!task.metrics_active && task.state != v2::PERSISTENCE_STATE_FAILED) {
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

v2::PlacementPolicy PersistenceManager::select_placement_policy(
    const ResolvedStorePolicy& policy,
    uint64_t total_size_bytes) {
  if (policy.remote_requirement == RequirementLevel::kNone) {
    return v2::PLACEMENT_POLICY_LOCAL_ONLY;
  }
  if (policy.layout == v2::POLICY_LAYOUT_SHARDED) {
    return v2::PLACEMENT_POLICY_SHARDED;
  }
  if (policy.layout == v2::POLICY_LAYOUT_UNSHARDED) {
    return v2::PLACEMENT_POLICY_REPLICATED;
  }
  return total_size_bytes >= kShardThresholdBytes ? v2::PLACEMENT_POLICY_SHARDED : v2::PLACEMENT_POLICY_REPLICATED;
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
  std::string cluster_id;
  if (client != nullptr) {
    auto cluster_or = client->get_cluster_id();
    if (cluster_or.ok()) {
      cluster_id = *cluster_or;
    } else {
      LOG(WARNING) << "persistence.cluster_id_unavailable: " << cluster_or.status().message();
    }
  }
  absl::MutexLock lock(&mu_);
  global_store_ = client;
  if (!cluster_id.empty()) {
    cluster_id_ = std::move(cluster_id);
  }
}

void PersistenceManager::set_storage_path(std::filesystem::path storage_root) {
  absl::MutexLock lock(&mu_);
  storage_root_ = std::move(storage_root);
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
    v2::PolicyLayout layout) const {
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
    shard.state = v2::PERSISTENCE_STATE_PENDING;
    shard.progress = 0.0;
    shards.push_back(std::move(shard));
  };

  const bool force_unsharded = layout == v2::POLICY_LAYOUT_UNSHARDED;
  const bool force_sharded = layout == v2::POLICY_LAYOUT_SHARDED;
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
      task.state = v2::PERSISTENCE_STATE_FAILED;
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
      task.state = v2::PERSISTENCE_STATE_FAILED;
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

absl::StatusOr<std::filesystem::path> PersistenceManager::disk_object_dir(const PersistenceTaskState& task) const {
  if (storage_root_.empty()) {
    return absl::FailedPreconditionError("storage_path_missing");
  }
  if (cluster_id_.empty()) {
    return absl::FailedPreconditionError("cluster_id_missing");
  }
  std::filesystem::path relative;
  if (!task.disk_relative_path.empty()) {
    relative = task.disk_relative_path;
  } else {
    const std::string artifact_segment = sanitize_segment(task.artifact_id);
    relative = std::filesystem::path(kClustersDir) / cluster_id_ / kObjectsDir / artifact_segment;
  }
  if (!is_safe_relative_path(relative)) {
    return absl::FailedPreconditionError("disk_relative_path_missing");
  }
  const std::filesystem::path expected_prefix = std::filesystem::path(kClustersDir) / cluster_id_;
  if (!has_path_prefix(relative, expected_prefix)) {
    return absl::FailedPreconditionError("disk_relative_path_missing");
  }
  return storage_root_ / relative;
}

absl::StatusOr<std::filesystem::path> PersistenceManager::ensure_disk_directory(PersistenceTaskState& task) {
  if (storage_root_.empty()) {
    return absl::FailedPreconditionError("storage_path_missing");
  }
  if (cluster_id_.empty()) {
    return absl::FailedPreconditionError("cluster_id_missing");
  }
  if (task.disk_relative_path.empty()) {
    const std::string artifact_segment = sanitize_segment(task.artifact_id);
    const std::filesystem::path relative =
        std::filesystem::path(kClustersDir) / cluster_id_ / kObjectsDir / artifact_segment;
    task.disk_relative_path = relative.generic_string();
  }
  const std::filesystem::path object_dir = storage_root_ / task.disk_relative_path;
  if (!is_safe_relative_path(task.disk_relative_path)) {
    return absl::FailedPreconditionError("disk_relative_path_missing");
  }
  const std::filesystem::path expected_prefix = std::filesystem::path(kClustersDir) / cluster_id_;
  if (!has_path_prefix(task.disk_relative_path, expected_prefix)) {
    return absl::FailedPreconditionError("disk_relative_path_missing");
  }
  if (!task.disk_directory_ready) {
    std::error_code ec;
    std::filesystem::create_directories(object_dir, ec);
    if (ec) {
      return absl::ErrnoToStatus(ec.value(), "shared_disk_directory_create_failed");
    }
    task.disk_directory_ready = true;
  }

  if (!task.by_key_linked && !task.key_hint.empty()) {
    const std::string key_segment = sanitize_segment(task.key_hint);
    const std::filesystem::path by_key_root =
        storage_root_ / std::filesystem::path(kClustersDir) / cluster_id_ / kByKeyDir / key_segment;
    std::error_code ec;
    std::filesystem::create_directories(by_key_root, ec);
    if (!ec) {
      const std::string base_name =
          std::format("{}_{}", format_persist_timestamp(absl::Now()), short_artifact_hint(task.artifact_id));
      std::filesystem::path link_path = by_key_root / base_name;
      std::filesystem::path target_rel;
      std::error_code rel_ec;
      target_rel = std::filesystem::relative(object_dir, link_path.parent_path(), rel_ec);
      if (rel_ec) {
        target_rel = object_dir;
      }
      auto pick_suffix = []() {
        static thread_local std::mt19937 rng{std::random_device{}()};
        static constexpr char kHex[] = "0123456789abcdef";
        std::string out;
        out.reserve(6);
        for (int i = 0; i < 6; ++i) {
          out.push_back(kHex[rng() % 16]);
        }
        return out;
      };
      for (int attempt = 0; attempt < 3; ++attempt) {
        if (std::filesystem::exists(link_path, ec)) {
          link_path = by_key_root / std::format("{}_{}", base_name, pick_suffix());
          continue;
        }
        std::filesystem::create_symlink(target_rel, link_path, ec);
        if (!ec) {
          task.by_key_linked = true;
          break;
        }
      }
      if (!task.by_key_linked) {
        LOG(WARNING) << "persistence.by_key_symlink_failed path=" << link_path.string() << " error=" << ec.message();
      }
    }
  }

  return object_dir;
}

absl::Status PersistenceManager::start_disk_write(
    PersistenceTaskState& task,
    const PersistenceShardState& shard,
    PersistenceTargetState& target) {
  if (store_engine_ == nullptr) {
    return absl::FailedPreconditionError("store_engine_unavailable");
  }
  auto object_dir_or = ensure_disk_directory(task);
  if (!object_dir_or.ok()) {
    return object_dir_or.status();
  }
  const std::filesystem::path part_path = *object_dir_or / std::format("tensor.data_{}", shard.shard_idx);
  std::error_code ec;
  if (std::filesystem::exists(part_path, ec)) {
    if (!ec && std::filesystem::file_size(part_path, ec) == shard.size_bytes) {
      auto state = std::make_shared<DiskWriteState>();
      state->total_bytes = shard.size_bytes;
      state->bytes_written.store(shard.size_bytes);
      state->part_path = part_path.string();
      state->done.store(true);
      target.disk_write_state = state;
      target.target_state = gs::PLACEMENT_TARGET_STATE_COPYING;
      return absl::OkStatus();
    }
  }

  if (!async_runtime_) {
    return absl::FailedPreconditionError("async_runtime_unavailable");
  }

  auto base_ptr_or = store_engine_->get_replica_cpu_base_ptr(task.artifact_id);
  if (!base_ptr_or.ok()) {
    return base_ptr_or.status();
  }
  const auto base_ptr = static_cast<const uint8_t*>(*base_ptr_or);
  const uint64_t offset = shard.byte_range_start;
  const uint64_t length = shard.size_bytes;

  auto state = std::make_shared<DiskWriteState>();
  state->total_bytes = length;
  state->part_path = part_path.string();
  target.disk_write_state = state;
  target.target_state = gs::PLACEMENT_TARGET_STATE_COPYING;

  auto executor = async_runtime_->blocking_executor();
  executor->add([state, base_ptr, offset, length, part_path]() {
    absl::Status st = absl::OkStatus();
    std::ofstream out(part_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      st = absl::FailedPreconditionError("shared_disk_write_failed");
    } else {
      constexpr uint64_t kChunk = 4ULL * 1024 * 1024;
      uint64_t remaining = length;
      const uint8_t* src = base_ptr + offset;
      while (st.ok() && remaining > 0) {
        const uint64_t to_write = std::min<uint64_t>(remaining, kChunk);
        out.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(to_write));
        if (!out.good()) {
          st = absl::FailedPreconditionError("shared_disk_write_failed");
          break;
        }
        state->bytes_written.fetch_add(to_write);
        remaining -= to_write;
        src += to_write;
      }
      out.flush();
      if (st.ok() && !out.good()) {
        st = absl::FailedPreconditionError("shared_disk_write_failed");
      }
    }
    state->set_status(std::move(st));
    state->done.store(true);
  });

  return absl::OkStatus();
}

absl::Status PersistenceManager::finalize_disk_directory(PersistenceTaskState& task) {
  if (task.disk_metadata_written) {
    return absl::OkStatus();
  }
  auto object_dir_or = disk_object_dir(task);
  if (!object_dir_or.ok()) {
    return object_dir_or.status();
  }
  const std::filesystem::path object_dir = *object_dir_or;
  std::error_code ec;
  std::filesystem::create_directories(object_dir, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), "shared_disk_directory_create_failed");
  }

  for (const auto& shard : task.shards) {
    const std::filesystem::path part_path = object_dir / std::format("tensor.data_{}", shard.shard_idx);
    if (!std::filesystem::exists(part_path, ec)) {
      return absl::NotFoundError("shared_disk_part_missing");
    }
    if (std::filesystem::file_size(part_path, ec) != shard.size_bytes) {
      return absl::FailedPreconditionError("shared_disk_part_size_mismatch");
    }
  }

  if (store_engine_ == nullptr) {
    return absl::FailedPreconditionError("store_engine_unavailable");
  }
  auto index_or = store_engine_->get_canonical_index_by_id(task.artifact_id);
  if (!index_or.ok()) {
    return index_or.status();
  }
  const std::filesystem::path index_path = object_dir / "tensor_index.json";
  std::ofstream idx_out(index_path, std::ios::trunc);
  if (!idx_out.is_open()) {
    return absl::FailedPreconditionError("shared_disk_index_write_failed");
  }
  idx_out << *index_or;
  idx_out.flush();
  if (!idx_out.good()) {
    return absl::FailedPreconditionError("shared_disk_index_write_failed");
  }

  auto parsed_or = parse_mi2_identity(task.artifact_id);
  if (!parsed_or.ok()) {
    return parsed_or.status();
  }
  const auto& index_mh = parsed_or->first;
  const auto& data_mh = parsed_or->second;

  nlohmann::json desc;
  desc["artifact_id"] = task.artifact_id;
  desc["index_multihash"] = index_mh;
  desc["data_multihash"] = data_mh;
  desc["schema_version"] = "v3";
  desc["encoding"] = "json";
  desc["total_size"] = task.total_size_bytes;

  const std::filesystem::path descriptor_path = object_dir / "artifact_descriptor.json";
  std::ofstream desc_out(descriptor_path, std::ios::trunc);
  if (!desc_out.is_open()) {
    return absl::FailedPreconditionError("shared_disk_descriptor_write_failed");
  }
  desc_out << desc.dump(2);
  desc_out.flush();
  if (!desc_out.good()) {
    return absl::FailedPreconditionError("shared_disk_descriptor_write_failed");
  }

  task.disk_metadata_written = true;
  return absl::OkStatus();
}

absl::Status PersistenceManager::register_disk_location(PersistenceTaskState& task) {
  if (task.disk_location_registered) {
    return absl::OkStatus();
  }
  if (global_store_ == nullptr) {
    return absl::FailedPreconditionError("global_store_unavailable");
  }
  if (cluster_id_.empty()) {
    return absl::FailedPreconditionError("cluster_id_missing");
  }
  if (task.disk_relative_path.empty()) {
    return absl::FailedPreconditionError("disk_relative_path_missing");
  }
  const absl::Status st = global_store_->upsert_artifact_disk_location(
      task.artifact_id, cluster_id_, task.disk_relative_path, gs::DISK_LOCATION_KIND_MANAGED);
  if (!st.ok()) {
    return st;
  }
  task.disk_location_registered = true;
  return absl::OkStatus();
}

void PersistenceManager::propagate_degraded_reason(PersistenceTaskState& task) const {
  if (task.state == v2::PERSISTENCE_STATE_FAILED) {
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
    return task.state == v2::PERSISTENCE_STATE_SUCCESS ? 1.0 : 0.0;
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
  double progress_sum = 0.0;
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
  const bool disk_commit_ready = task.disk_location_registered;

  for (auto& target : shard.targets) {
    if (target.is_shared_disk) {
      disk_seen = true;
      double target_progress = 0.0;
      if (target.target_state == gs::PLACEMENT_TARGET_STATE_COMPLETE ||
          target.target_state == gs::PLACEMENT_TARGET_STATE_SKIPPED) {
        ++done_targets;
        disk_success = true;
        target_progress = 1.0;
        progress_sum += target_progress;
        continue;
      }
      if (target.target_state == gs::PLACEMENT_TARGET_STATE_FAILED) {
        disk_failed = true;
        if (disk_failure_reason.empty()) {
          disk_failure_reason = target.degraded_reason;
        }
        ++done_targets;
        target_progress = 1.0;
        progress_sum += target_progress;
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
        target_progress = 1.0;
        progress_sum += target_progress;
        continue;
      }
      if (target.disk_write_state) {
        const auto state = target.disk_write_state;
        if (state->total_bytes > 0) {
          target_progress = static_cast<double>(state->bytes_written.load()) / static_cast<double>(state->total_bytes);
          target_progress = std::min(1.0, target_progress);
        }
        if (state->done.load()) {
          const absl::Status st = state->get_status();
          if (!st.ok()) {
            target.target_state = gs::PLACEMENT_TARGET_STATE_FAILED;
            target.degraded_reason = disk_failure_token(st, DiskFailureStage::kWrite);
            disk_failed = true;
            if (disk_failure_reason.empty()) {
              disk_failure_reason = target.degraded_reason;
            }
            ++done_targets;
            target_progress = 1.0;
          } else if (disk_commit_ready) {
            target.target_state = gs::PLACEMENT_TARGET_STATE_COMPLETE;
            ++done_targets;
            disk_success = true;
            target_progress = 1.0;
          } else {
            target.target_state = gs::PLACEMENT_TARGET_STATE_COPYING;
          }
        } else {
          target.target_state = gs::PLACEMENT_TARGET_STATE_COPYING;
        }
        progress_sum += target_progress;
        continue;
      }
      if (target.target_state == gs::PLACEMENT_TARGET_STATE_PENDING) {
        auto st = start_disk_write(task, shard, target);
        if (!st.ok()) {
          target.target_state = gs::PLACEMENT_TARGET_STATE_FAILED;
          target.degraded_reason = disk_failure_token(st, DiskFailureStage::kWrite);
          disk_failed = true;
          if (disk_failure_reason.empty()) {
            disk_failure_reason = target.degraded_reason;
          }
          ++done_targets;
          target_progress = 1.0;
        }
        progress_sum += target_progress;
        continue;
      }
      progress_sum += target_progress;
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
      progress_sum += 1.0;
      continue;
    }
    if (target.target_state == gs::PLACEMENT_TARGET_STATE_FAILED) {
      if (target.attempts >= kMaxLeaseAttempts) {
        remote_failed = true;
        if (remote_failure_reason.empty()) {
          remote_failure_reason = target.degraded_reason;
        }
        ++done_targets;
        progress_sum += 1.0;
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
          progress_sum += 1.0;
        }
        continue;
      }
      target.target_state = gs::PLACEMENT_TARGET_STATE_COMPLETE;
      ++done_targets;
      if (is_remote) {
        remote_success = true;
      }
      progress_sum += 1.0;
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
      progress_sum += 1.0;
      continue;
    }
    if (target.node_id == local_node_id_) {
      target.target_state = gs::PLACEMENT_TARGET_STATE_COMPLETE;
      ++done_targets;
      progress_sum += 1.0;
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
      progress_sum += 1.0;
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
        progress_sum += 1.0;
      }
      continue;
    }
    target.lease_id = lease_or->lease_id;
    target.target_state = gs::PLACEMENT_TARGET_STATE_COPYING;
    target.cooldown_ticks = 0;
  }

  shard.progress = total_targets == 0 ? 1.0 : progress_sum / static_cast<double>(total_targets);

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
    shard.state = v2::PERSISTENCE_STATE_FAILED;
    return;
  }
  const bool all_done = done_targets == total_targets;
  if (all_done) {
    shard.state = (shard_degraded || !task.degraded_reason.empty()) ? v2::PERSISTENCE_STATE_DEGRADED
                                                                    : v2::PERSISTENCE_STATE_SUCCESS;
  } else {
    shard.state = v2::PERSISTENCE_STATE_RUNNING;
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
      tasks_[entry.first] = std::move(entry.second);
      update_durability_locked(tasks_[entry.first]);
    }
  }
  if (scheduler_ != nullptr && notify_scheduler) {
    scheduler_->notify(TaskKind::kPersistence);
  }
}

bool PersistenceManager::is_terminal(v2::PersistenceState state) {
  return state == v2::PERSISTENCE_STATE_SUCCESS || state == v2::PERSISTENCE_STATE_FAILED ||
      state == v2::PERSISTENCE_STATE_DEGRADED;
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

gs::PersistenceState PersistenceManager::to_global_state(v2::PersistenceState state) {
  switch (state) {
    case v2::PERSISTENCE_STATE_PENDING:
      return gs::PERSISTENCE_STATE_PENDING;
    case v2::PERSISTENCE_STATE_RUNNING:
      return gs::PERSISTENCE_STATE_RUNNING;
    case v2::PERSISTENCE_STATE_DEGRADED:
      return gs::PERSISTENCE_STATE_DEGRADED;
    case v2::PERSISTENCE_STATE_SUCCESS:
      return gs::PERSISTENCE_STATE_SUCCESS;
    case v2::PERSISTENCE_STATE_FAILED:
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
    if (task.state == v2::PERSISTENCE_STATE_PENDING) {
      task.state = v2::PERSISTENCE_STATE_RUNNING;
      for (auto& shard : task.shards) {
        shard.state = v2::PERSISTENCE_STATE_RUNNING;
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
      if (shard.state == v2::PERSISTENCE_STATE_FAILED) {
        any_failed = true;
      }
      if (shard.state == v2::PERSISTENCE_STATE_DEGRADED || !shard.degraded_reason.empty()) {
        any_degraded = true;
      }
      if (shard.state != v2::PERSISTENCE_STATE_SUCCESS && shard.state != v2::PERSISTENCE_STATE_DEGRADED) {
        all_done = false;
      }
    }

    const bool disk_writes_complete = [&]() {
      if (!task.persist_to_shared_disk) {
        return false;
      }
      for (const auto& shard : task.shards) {
        bool has_disk_target = false;
        bool shard_written = false;
        for (const auto& target : shard.targets) {
          if (!target.is_shared_disk) {
            continue;
          }
          has_disk_target = true;
          if (target.target_state == gs::PLACEMENT_TARGET_STATE_FAILED) {
            return false;
          }
          if (target.disk_write_state && target.disk_write_state->done.load()) {
            const absl::Status st = target.disk_write_state->get_status();
            if (!st.ok()) {
              return false;
            }
            shard_written = true;
            break;
          }
          if (target.target_state == gs::PLACEMENT_TARGET_STATE_COMPLETE ||
              target.target_state == gs::PLACEMENT_TARGET_STATE_SKIPPED) {
            shard_written = true;
            break;
          }
        }
        if (!has_disk_target || !shard_written) {
          return false;
        }
      }
      return true;
    }();

    if (disk_writes_complete && !task.disk_location_registered) {
      absl::Status st = finalize_disk_directory(task);
      DiskFailureStage stage = DiskFailureStage::kFinalize;
      if (st.ok()) {
        stage = DiskFailureStage::kRegister;
        st = register_disk_location(task);
      }
      if (!st.ok()) {
        const bool disk_required = task.shared_disk_requirement == RequirementLevel::kMust;
        const bool disk_should = task.shared_disk_requirement == RequirementLevel::kShould;
        const std::string token = disk_failure_token(st, stage);
        if (disk_required) {
          any_failed = true;
          if (task.last_error.empty()) {
            task.last_error = token;
          }
          for (auto& shard : task.shards) {
            shard.state = v2::PERSISTENCE_STATE_FAILED;
            if (shard.last_error.empty()) {
              shard.last_error = token;
            }
          }
        } else if (disk_should) {
          any_degraded = true;
          if (task.degraded_reason.empty()) {
            task.degraded_reason = token;
          }
          for (auto& shard : task.shards) {
            if (shard.degraded_reason.empty()) {
              shard.degraded_reason = token;
            }
          }
        }
      }
    }

    update_durability_locked(task);

    if (any_failed && task.last_error.empty()) {
      for (const auto& shard : task.shards) {
        if (!shard.last_error.empty()) {
          task.last_error = shard.last_error;
          break;
        }
      }
      if (task.last_error.empty()) {
        task.last_error = "one or more persistence shards failed";
      }
    }
    propagate_degraded_reason(task);
    if (any_failed) {
      task.state = v2::PERSISTENCE_STATE_FAILED;
    } else if (all_done) {
      task.state = any_degraded ? v2::PERSISTENCE_STATE_DEGRADED : v2::PERSISTENCE_STATE_SUCCESS;
    } else {
      task.state = v2::PERSISTENCE_STATE_RUNNING;
    }
    task.progress = compute_task_progress(task);
    if (is_terminal(task.state) && task.metrics_active && !task.metrics_closed) {
      adjust_active_metric(-1);
      task.metrics_closed = true;
      if (task.state == v2::PERSISTENCE_STATE_FAILED) {
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
  if (task.persist_to_shared_disk && task.disk_location_registered && !task.shards.empty()) {
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
