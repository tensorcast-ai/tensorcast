// Copyright (c) 2025, TensorCast Team.

#include "core/store/components/stable_dram_cache_manager.h"

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/types/span.h"
#include "core/common/memory/memory_location.h"
#include "core/store/components/replica_registry.h"
#include "core/store/device_types.h"
#include "core/store/memory_tier_budget.h"
#include "core/store/replica/replica.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::components {

namespace {

constexpr absl::string_view kMetricsScope = "tensorcast.daemon";

const char* retention_policy_label(StableRetentionPolicy policy) {
  switch (policy) {
    case StableRetentionPolicy::kPinned:
      return "pinned";
    case StableRetentionPolicy::kTtl:
      return "ttl";
    case StableRetentionPolicy::kBestEffort:
    default:
      return "best_effort";
  }
}

const char* overflow_policy_label(StableOverflowPolicy policy) {
  switch (policy) {
    case StableOverflowPolicy::kReject:
      return "reject";
    case StableOverflowPolicy::kSpill:
      return "spill";
    case StableOverflowPolicy::kEvict:
    default:
      return "evict";
  }
}

bool is_cpu_key(const loading::ReplicaKey& key) {
  return key.device.type == DeviceType::CPU;
}

} // namespace

StableDramCacheManager::StableDramCacheManager(Config config)
    : registry_(config.registry),
      memory_tier_budget_(std::move(config.memory_tier_budget)),
      spill_guard_(std::move(config.spill_guard)),
      spill_evictable_(std::move(config.spill_evictable)) {}

uint64_t StableDramCacheManager::bytes_used() const {
  return bytes_used_.load();
}

absl::StatusOr<StableDramCacheManager::AdmissionResult> StableDramCacheManager::admit(const AdmissionRequest& request) {
  AdmissionResult result;
  if (!request.replica) {
    return absl::InvalidArgumentError("stable cache admission requires replica");
  }
  if (request.size_bytes == 0) {
    return absl::InvalidArgumentError("stable cache admission requires size_bytes > 0");
  }
  if (!is_cpu_key(request.key)) {
    return absl::FailedPreconditionError("stable cache admission requires CPU replica");
  }

  {
    absl::MutexLock lock(&mu_);
    if (entries_.contains(request.key)) {
      record_hit();
      result.admitted = true;
      return result;
    }
  }
  record_miss();

  const uint64_t stable_budget = memory_tier_budget_ ? memory_tier_budget_->stable_total_bytes() : 0;
  if (memory_tier_budget_ && stable_budget > 0 && request.size_bytes > stable_budget) {
    LOG(WARNING) << "stable_cache.reject size_exceeds_budget artifact_id=" << request.key.artifact_id
                 << " size_bytes=" << request.size_bytes << " stable_bytes=" << stable_budget;
    if (request.policy.required) {
      return absl::ResourceExhaustedError("stable cache admission exceeds stable tier budget");
    }
    result.skipped = true;
    return result;
  }

  auto lease_or = acquire_stable_lease(request.key, request.replica);
  if (!lease_or.ok() && absl::IsResourceExhausted(lease_or.status()) &&
      request.policy.overflow_policy != StableOverflowPolicy::kReject) {
    if (request.policy.overflow_policy == StableOverflowPolicy::kSpill) {
      absl::Status spill_status;
      if (spill_guard_) {
        spill_status = spill_guard_(request.key);
      } else {
        spill_status = absl::FailedPreconditionError("stable cache spill requires shared disk availability");
      }
      if (!spill_status.ok()) {
        LOG(WARNING) << "stable_cache.spill_blocked artifact_id=" << request.key.artifact_id
                     << " status=" << spill_status;
        return absl::ResourceExhaustedError(absl::StrCat("stable cache spill blocked: ", spill_status.message()));
      }
    }
    const absl::Status evict_status = evict_for_bytes(
        request.size_bytes, request.key, absl::Now(), request.policy.overflow_policy == StableOverflowPolicy::kSpill);
    if (evict_status.ok()) {
      lease_or = acquire_stable_lease(request.key, request.replica);
    } else if (request.policy.overflow_policy == StableOverflowPolicy::kSpill) {
      return evict_status;
    }
  }

  if (!lease_or.ok()) {
    if (request.policy.overflow_policy == StableOverflowPolicy::kSpill) {
      return lease_or.status();
    }
    if (request.policy.overflow_policy == StableOverflowPolicy::kReject || request.policy.required) {
      return lease_or.status();
    }
    result.skipped = true;
    return result;
  }

  CacheEntry entry;
  entry.key = request.key;
  entry.size_bytes = request.size_bytes;
  entry.stable_bytes = lease_or->bytes;
  entry.policy = request.policy;
  if (request.policy.retention_ttl.has_value()) {
    entry.retention_deadline = absl::Now() + absl::FromChrono(*request.policy.retention_ttl);
  }
  entry.stable_lease = std::move(*lease_or);

  const uint64_t stable_bytes = entry.stable_bytes;
  bool inserted = false;
  {
    absl::MutexLock lock(&mu_);
    if (entries_.contains(entry.key)) {
      inserted = false;
    } else {
      entries_.emplace(entry.key, std::move(entry));
      inserted = true;
    }
  }

  if (!inserted) {
    auto uma = request.replica->get_memory_manager().memory_authority();
    if (uma && entry.stable_lease.has_value()) {
      absl::Status st = uma->release_stable_lease(*entry.stable_lease);
      if (!st.ok()) {
        LOG(WARNING) << "stable_cache.release_lease_failed artifact_id=" << request.key.artifact_id << " status=" << st;
      }
    }
    result.admitted = true;
    return result;
  }

  bytes_used_.fetch_add(stable_bytes);
  record_bytes_delta(static_cast<int64_t>(stable_bytes));

  LOG(INFO) << "stable_cache.admit artifact_id=" << request.key.artifact_id << " size_bytes=" << request.size_bytes
            << " retention=" << retention_policy_label(request.policy.retention_policy)
            << " overflow=" << overflow_policy_label(request.policy.overflow_policy);
  result.admitted = true;
  return result;
}

bool StableDramCacheManager::is_evictable(const loading::ReplicaKey& key, absl::Time now) const {
  if (!is_cpu_key(key)) {
    return true;
  }
  std::optional<CacheEntry> entry;
  {
    absl::MutexLock lock(&mu_);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      entry = it->second;
    }
  }

  auto replica_or = registry_->find(key);
  if (!replica_or.ok()) {
    return false;
  }
  const auto& replica = replica_or.value();
  if (replica->get_memory_state(common::memory::MemoryLocation::CPU) == replica::MemoryState::LOADING) {
    return false;
  }
  auto uma = replica->get_memory_manager().memory_authority();
  if (!uma) {
    return false;
  }
  const auto snapshot = uma->snapshot_cpu_chunks(key);
  const bool tracked = entry.has_value();
  for (const auto& rec : snapshot) {
    if (rec.pin_refcnt > 0) {
      return false;
    }
    if (!tracked && rec.stable_lease_count > 0) {
      return false;
    }
    if (tracked && rec.stable_lease_count > 1) {
      return false;
    }
  }

  if (!entry.has_value()) {
    return true;
  }
  return is_entry_evictable(*entry, now);
}

void StableDramCacheManager::on_replica_evicted(const loading::ReplicaKey& key, absl::string_view reason) {
  CacheEntry entry;
  bool found = false;
  {
    absl::MutexLock lock(&mu_);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return;
    }
    entry = it->second;
    entries_.erase(it);
    found = true;
  }
  if (!found) {
    return;
  }

  auto replica_or = registry_->find(key);
  if (replica_or.ok()) {
    auto uma = replica_or.value()->get_memory_manager().memory_authority();
    if (uma && entry.stable_lease.has_value()) {
      absl::Status st = uma->release_stable_lease(*entry.stable_lease);
      if (!st.ok()) {
        LOG(WARNING) << "stable_cache.release_lease_failed artifact_id=" << key.artifact_id << " status=" << st;
      }
    }
  }

  bytes_used_.fetch_sub(entry.stable_bytes);
  record_bytes_delta(-static_cast<int64_t>(entry.stable_bytes));
  record_eviction();
  if (entry.policy.retention_policy == StableRetentionPolicy::kTtl && entry.retention_deadline.has_value() &&
      absl::Now() >= *entry.retention_deadline) {
    record_ttl_expiration();
  }
  LOG(INFO) << "stable_cache.evicted artifact_id=" << key.artifact_id << " size_bytes=" << entry.size_bytes
            << " reason=" << std::string(reason);
}

void StableDramCacheManager::set_spill_evictable_callback(SpillEvictableCallback callback) {
  absl::MutexLock lock(&mu_);
  spill_evictable_ = std::move(callback);
}

absl::StatusOr<replica::UnifiedMemoryAuthority::StableLease> StableDramCacheManager::acquire_stable_lease(
    const loading::ReplicaKey& key,
    const std::shared_ptr<replica::Replica>& replica) const {
  auto uma = replica->get_memory_manager().memory_authority();
  if (!uma) {
    return absl::FailedPreconditionError("stable cache admission requires UMA authority");
  }
  auto layout_or = uma->get_layout(key);
  if (!layout_or.ok()) {
    return layout_or.status();
  }
  const auto& layout = *layout_or;
  if (layout.artifact_chunk_bytes == 0) {
    return absl::FailedPreconditionError("stable cache admission requires artifact_chunk_bytes > 0");
  }
  const uint64_t chunk_count = (layout.artifact_bytes + layout.artifact_chunk_bytes - 1) / layout.artifact_chunk_bytes;
  if (chunk_count == 0) {
    return absl::FailedPreconditionError("stable cache admission requires non-empty artifact");
  }
  std::vector<uint32_t> chunk_ids;
  chunk_ids.reserve(static_cast<size_t>(chunk_count));
  for (uint32_t idx = 0; idx < chunk_count; ++idx) {
    chunk_ids.push_back(idx);
  }
  return uma->acquire_stable_lease(key, absl::MakeSpan(chunk_ids));
}

absl::Status StableDramCacheManager::evict_for_bytes(
    uint64_t required_bytes,
    const loading::ReplicaKey& exclude,
    absl::Time now,
    bool spill_only) {
  if (required_bytes == 0) {
    return absl::OkStatus();
  }
  const auto candidates = registry_->get_lru_instances();
  SpillEvictableCallback spill_evictable;
  if (spill_only) {
    absl::MutexLock lock(&mu_);
    spill_evictable = spill_evictable_;
  }
  uint64_t freed = 0;

  for (const auto& key : candidates) {
    if (!is_cpu_key(key) || key == exclude) {
      continue;
    }
    CacheEntry entry;
    {
      absl::MutexLock lock(&mu_);
      auto it = entries_.find(key);
      if (it == entries_.end()) {
        continue;
      }
      entry = it->second;
    }
    if (!is_entry_evictable(entry, now)) {
      continue;
    }
    if (spill_only && !is_spill_evictable(entry, spill_evictable)) {
      continue;
    }

    auto replica_or = registry_->find(key);
    if (!replica_or.ok()) {
      continue;
    }
    auto replica = replica_or.value();
    auto uma = replica->get_memory_manager().memory_authority();
    if (entry.stable_lease.has_value() && uma) {
      absl::Status st = uma->release_stable_lease(*entry.stable_lease);
      if (!st.ok()) {
        LOG(WARNING) << "stable_cache.release_lease_failed artifact_id=" << key.artifact_id << " status=" << st;
        continue;
      }
    }
    absl::Status release_status = replica->release_memory(common::memory::MemoryLocation::CPU);
    if (!release_status.ok()) {
      LOG(WARNING) << "stable_cache.release_memory_failed artifact_id=" << key.artifact_id
                   << " status=" << release_status;
    }

    {
      absl::MutexLock lock(&mu_);
      entries_.erase(key);
    }
    bytes_used_.fetch_sub(entry.stable_bytes);
    record_bytes_delta(-static_cast<int64_t>(entry.stable_bytes));
    record_eviction();
    if (entry.policy.retention_policy == StableRetentionPolicy::kTtl && entry.retention_deadline.has_value() &&
        now >= *entry.retention_deadline) {
      record_ttl_expiration();
    }
    freed += entry.stable_bytes;
    LOG(INFO) << "stable_cache.evicted artifact_id=" << key.artifact_id << " size_bytes=" << entry.size_bytes
              << " reason=admission";
    if (freed >= required_bytes) {
      return absl::OkStatus();
    }
  }

  return absl::ResourceExhaustedError("stable cache eviction could not free enough bytes");
}

bool StableDramCacheManager::is_entry_evictable(const CacheEntry& entry, absl::Time now) const {
  if (entry.policy.retention_policy == StableRetentionPolicy::kPinned) {
    return false;
  }
  if (entry.policy.retention_policy == StableRetentionPolicy::kTtl && entry.retention_deadline.has_value()) {
    return now >= *entry.retention_deadline;
  }
  return true;
}

bool StableDramCacheManager::is_spill_evictable(
    const CacheEntry& entry,
    const SpillEvictableCallback& spill_evictable) {
  if (!spill_evictable) {
    return false;
  }
  return spill_evictable(entry.key, entry.policy);
}

void StableDramCacheManager::record_bytes_delta(int64_t delta) const {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter(std::string(kMetricsScope));
    static auto counter = meter->CreateInt64UpDownCounter("tc_stable_cache_bytes_used");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    counter->Add(delta, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void StableDramCacheManager::record_eviction() const {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter(std::string(kMetricsScope));
    static auto counter = meter->CreateUInt64Counter("tc_stable_cache_evictions_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void StableDramCacheManager::record_hit() const {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter(std::string(kMetricsScope));
    static auto counter = meter->CreateUInt64Counter("tc_stable_cache_hits_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void StableDramCacheManager::record_miss() const {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter(std::string(kMetricsScope));
    static auto counter = meter->CreateUInt64Counter("tc_stable_cache_misses_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void StableDramCacheManager::record_ttl_expiration() const {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter(std::string(kMetricsScope));
    static auto counter = meter->CreateUInt64Counter("tc_stable_cache_ttl_expirations_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

} // namespace tensorcast::store::components
