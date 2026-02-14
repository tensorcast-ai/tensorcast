// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/components/stable_dram_cache_manager.h"

#include <chrono>
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/replica_registry.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/memory_tier_budget.h"
#include "core/store/replica/replica.h"
#include "gsl/pointers"

using tensorcast::DeviceType;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::MemoryTierBudget;
using tensorcast::store::components::ReplicaRegistry;
using tensorcast::store::components::StableDramCacheManager;
using tensorcast::store::components::StableDramCachePolicy;
using tensorcast::store::components::StableOverflowPolicy;
using tensorcast::store::components::StableRetentionPolicy;
using tensorcast::store::loading::InlineBufferSource;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::replica::MemoryState;
using tensorcast::store::replica::Replica;
using tensorcast::store::replica::ReplicaConfig;

namespace {

constexpr size_t kChunkBytes = 64;

std::shared_ptr<Replica> MakeCpuReplica(
    const std::string& artifact_id,
    uint64_t size_bytes,
    gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>> pool,
    gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>> runtime,
    const std::shared_ptr<MemoryTierBudget>& budget) {
  InlineBufferSource src{.data = nullptr, .size_bytes = size_bytes};
  ReplicaConfig cfg{
      .source = src,
      .artifact_identifier = artifact_id,
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool = pool,
      .async_runtime = runtime,
      .artifact_chunk_bytes = kChunkBytes,
      .expected_artifact_size = size_bytes,
  };
  auto replica_or = Replica::create(cfg);
  REQUIRE(replica_or.ok());
  auto replica = std::shared_ptr<Replica>(std::move(replica_or.value()));
  if (budget) {
    replica->get_memory_manager().set_memory_tier_budget(budget);
  }
  REQUIRE(replica->get_memory_manager().allocate_memory(MemoryLocation::CPU).ok());
  REQUIRE(replica->mark_loaded(MemoryLocation::CPU).ok());
  return replica;
}

StableDramCachePolicy MakePolicy(StableRetentionPolicy retention, StableOverflowPolicy overflow) {
  StableDramCachePolicy policy;
  policy.retention_policy = retention;
  policy.overflow_policy = overflow;
  policy.required = false;
  return policy;
}

} // namespace

TEST_CASE("StableDramCacheManager evicts best-effort entries on overflow", "[stable_cache]") {
  auto pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(1 << 20, 1 << 20);
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  auto budget = std::make_shared<MemoryTierBudget>(kChunkBytes, 0);
  ReplicaRegistry registry;

  StableDramCacheManager cache(
      StableDramCacheManager::Config{
          .registry = gsl::not_null<ReplicaRegistry*>{&registry},
          .memory_tier_budget = budget,
      });

  auto replica1 = MakeCpuReplica(
      "artifact-1",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey key1 = replica1->replica_key();
  REQUIRE(registry.emplace(key1, gsl::not_null<std::shared_ptr<Replica>>{replica1}).ok());

  StableDramCacheManager::AdmissionRequest request1;
  request1.key = key1;
  request1.replica = replica1;
  request1.size_bytes = kChunkBytes;
  request1.policy = MakePolicy(StableRetentionPolicy::kBestEffort, StableOverflowPolicy::kEvict);
  auto admit1 = cache.admit(request1);
  REQUIRE(admit1.ok());
  REQUIRE(admit1->admitted);

  auto replica2 = MakeCpuReplica(
      "artifact-2",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey key2 = replica2->replica_key();
  REQUIRE(registry.emplace(key2, gsl::not_null<std::shared_ptr<Replica>>{replica2}).ok());

  StableDramCacheManager::AdmissionRequest request2;
  request2.key = key2;
  request2.replica = replica2;
  request2.size_bytes = kChunkBytes;
  request2.policy = MakePolicy(StableRetentionPolicy::kBestEffort, StableOverflowPolicy::kEvict);
  auto admit2 = cache.admit(request2);
  REQUIRE(admit2.ok());
  REQUIRE(admit2->admitted);
  REQUIRE(replica1->get_memory_state(MemoryLocation::CPU) == MemoryState::UNALLOCATED);
  REQUIRE(replica2->get_memory_state(MemoryLocation::CPU) == MemoryState::LOADED);
}

TEST_CASE("StableDramCacheManager rejects admission when overflow policy is reject", "[stable_cache]") {
  auto pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(1 << 20, 1 << 20);
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  auto budget = std::make_shared<MemoryTierBudget>(kChunkBytes, 0);
  ReplicaRegistry registry;

  StableDramCacheManager cache(
      StableDramCacheManager::Config{
          .registry = gsl::not_null<ReplicaRegistry*>{&registry},
          .memory_tier_budget = budget,
      });

  auto replica1 = MakeCpuReplica(
      "artifact-reject-1",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey key1 = replica1->replica_key();
  REQUIRE(registry.emplace(key1, gsl::not_null<std::shared_ptr<Replica>>{replica1}).ok());

  StableDramCacheManager::AdmissionRequest request1;
  request1.key = key1;
  request1.replica = replica1;
  request1.size_bytes = kChunkBytes;
  request1.policy = MakePolicy(StableRetentionPolicy::kBestEffort, StableOverflowPolicy::kEvict);
  auto admit1 = cache.admit(request1);
  REQUIRE(admit1.ok());
  REQUIRE(admit1->admitted);

  auto replica2 = MakeCpuReplica(
      "artifact-reject-2",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey key2 = replica2->replica_key();
  REQUIRE(registry.emplace(key2, gsl::not_null<std::shared_ptr<Replica>>{replica2}).ok());

  StableDramCacheManager::AdmissionRequest request2;
  request2.key = key2;
  request2.replica = replica2;
  request2.size_bytes = kChunkBytes;
  request2.policy = MakePolicy(StableRetentionPolicy::kBestEffort, StableOverflowPolicy::kReject);
  auto admit2 = cache.admit(request2);
  REQUIRE_FALSE(admit2.ok());
  REQUIRE(absl::IsResourceExhausted(admit2.status()));
  REQUIRE(replica1->get_memory_state(MemoryLocation::CPU) == MemoryState::LOADED);
}

TEST_CASE("StableDramCacheManager honors TTL and pinned retention", "[stable_cache]") {
  auto pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(1 << 20, 1 << 20);
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  auto budget = std::make_shared<MemoryTierBudget>(kChunkBytes * 4, 0);
  ReplicaRegistry registry;

  StableDramCacheManager cache(
      StableDramCacheManager::Config{
          .registry = gsl::not_null<ReplicaRegistry*>{&registry},
          .memory_tier_budget = budget,
      });

  auto ttl_replica = MakeCpuReplica(
      "artifact-ttl",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey ttl_key = ttl_replica->replica_key();
  REQUIRE(registry.emplace(ttl_key, gsl::not_null<std::shared_ptr<Replica>>{ttl_replica}).ok());

  StableDramCacheManager::AdmissionRequest ttl_request;
  ttl_request.key = ttl_key;
  ttl_request.replica = ttl_replica;
  ttl_request.size_bytes = kChunkBytes;
  ttl_request.policy = MakePolicy(StableRetentionPolicy::kTtl, StableOverflowPolicy::kEvict);
  ttl_request.policy.retention_ttl = std::chrono::milliseconds(5000);
  auto ttl_admit = cache.admit(ttl_request);
  REQUIRE(ttl_admit.ok());
  REQUIRE(ttl_admit->admitted);

  const absl::Time now = absl::Now();
  REQUIRE_FALSE(cache.is_evictable(ttl_key, now));
  REQUIRE(cache.is_evictable(ttl_key, now + absl::Seconds(10)));

  auto pinned_replica = MakeCpuReplica(
      "artifact-pinned",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey pinned_key = pinned_replica->replica_key();
  REQUIRE(registry.emplace(pinned_key, gsl::not_null<std::shared_ptr<Replica>>{pinned_replica}).ok());

  StableDramCacheManager::AdmissionRequest pinned_request;
  pinned_request.key = pinned_key;
  pinned_request.replica = pinned_replica;
  pinned_request.size_bytes = kChunkBytes;
  pinned_request.policy = MakePolicy(StableRetentionPolicy::kPinned, StableOverflowPolicy::kEvict);
  auto pinned_admit = cache.admit(pinned_request);
  REQUIRE(pinned_admit.ok());
  REQUIRE(pinned_admit->admitted);
  REQUIRE_FALSE(cache.is_evictable(pinned_key, now + absl::Hours(1)));
}

TEST_CASE("StableDramCacheManager upgrades retention on re-admit", "[stable_cache]") {
  auto pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(1 << 20, 1 << 20);
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  auto budget = std::make_shared<MemoryTierBudget>(kChunkBytes * 4, 0);
  ReplicaRegistry registry;

  StableDramCacheManager cache(
      StableDramCacheManager::Config{
          .registry = gsl::not_null<ReplicaRegistry*>{&registry},
          .memory_tier_budget = budget,
      });

  auto replica = MakeCpuReplica(
      "artifact-upgrade",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey key = replica->replica_key();
  REQUIRE(registry.emplace(key, gsl::not_null<std::shared_ptr<Replica>>{replica}).ok());

  StableDramCacheManager::AdmissionRequest request;
  request.key = key;
  request.replica = replica;
  request.size_bytes = kChunkBytes;
  request.policy = MakePolicy(StableRetentionPolicy::kBestEffort, StableOverflowPolicy::kEvict);
  auto admit = cache.admit(request);
  REQUIRE(admit.ok());
  REQUIRE(admit->admitted);

  const absl::Time now = absl::Now();
  REQUIRE(cache.is_evictable(key, now));
  const uint64_t bytes_before = cache.bytes_used();

  StableDramCacheManager::AdmissionRequest upgrade = request;
  upgrade.policy = MakePolicy(StableRetentionPolicy::kPinned, StableOverflowPolicy::kEvict);
  auto admit_upgrade = cache.admit(upgrade);
  REQUIRE(admit_upgrade.ok());
  REQUIRE(admit_upgrade->admitted);
  REQUIRE_FALSE(cache.is_evictable(key, now + absl::Hours(1)));
  REQUIRE(cache.bytes_used() == bytes_before);
}

TEST_CASE("StableDramCacheManager rejects admission when required pinned entries fill budget", "[stable_cache]") {
  auto pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(1 << 20, 1 << 20);
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  auto budget = std::make_shared<MemoryTierBudget>(kChunkBytes, 0);
  ReplicaRegistry registry;

  StableDramCacheManager cache(
      StableDramCacheManager::Config{
          .registry = gsl::not_null<ReplicaRegistry*>{&registry},
          .memory_tier_budget = budget,
      });

  auto replica1 = MakeCpuReplica(
      "artifact-required-1",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey key1 = replica1->replica_key();
  REQUIRE(registry.emplace(key1, gsl::not_null<std::shared_ptr<Replica>>{replica1}).ok());

  StableDramCacheManager::AdmissionRequest request1;
  request1.key = key1;
  request1.replica = replica1;
  request1.size_bytes = kChunkBytes;
  request1.policy = MakePolicy(StableRetentionPolicy::kPinned, StableOverflowPolicy::kEvict);
  request1.policy.required = true;
  auto admit1 = cache.admit(request1);
  REQUIRE(admit1.ok());
  REQUIRE(admit1->admitted);

  auto replica2 = MakeCpuReplica(
      "artifact-required-2",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey key2 = replica2->replica_key();
  REQUIRE(registry.emplace(key2, gsl::not_null<std::shared_ptr<Replica>>{replica2}).ok());

  StableDramCacheManager::AdmissionRequest request2;
  request2.key = key2;
  request2.replica = replica2;
  request2.size_bytes = kChunkBytes;
  request2.policy = MakePolicy(StableRetentionPolicy::kPinned, StableOverflowPolicy::kEvict);
  request2.policy.required = true;
  auto admit2 = cache.admit(request2);
  REQUIRE_FALSE(admit2.ok());
  REQUIRE(absl::IsResourceExhausted(admit2.status()));
  REQUIRE(replica1->get_memory_state(MemoryLocation::CPU) == MemoryState::LOADED);
}

TEST_CASE("StableDramCacheManager spill fails closed when shared disk is unavailable", "[stable_cache]") {
  auto pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(1 << 20, 1 << 20);
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  auto budget = std::make_shared<MemoryTierBudget>(kChunkBytes, 0);
  ReplicaRegistry registry;

  StableDramCacheManager cache(
      StableDramCacheManager::Config{
          .registry = gsl::not_null<ReplicaRegistry*>{&registry},
          .memory_tier_budget = budget,
          .spill_guard = [](const ReplicaKey&) { return absl::FailedPreconditionError("shared disk unavailable"); },
          .spill_evictable = [](const ReplicaKey&, const StableDramCachePolicy&) { return true; },
      });

  auto replica1 = MakeCpuReplica(
      "artifact-spill-1",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey key1 = replica1->replica_key();
  REQUIRE(registry.emplace(key1, gsl::not_null<std::shared_ptr<Replica>>{replica1}).ok());

  StableDramCacheManager::AdmissionRequest request1;
  request1.key = key1;
  request1.replica = replica1;
  request1.size_bytes = kChunkBytes;
  request1.policy = MakePolicy(StableRetentionPolicy::kBestEffort, StableOverflowPolicy::kSpill);
  auto admit1 = cache.admit(request1);
  REQUIRE(admit1.ok());
  REQUIRE(admit1->admitted);

  auto replica2 = MakeCpuReplica(
      "artifact-spill-2",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey key2 = replica2->replica_key();
  REQUIRE(registry.emplace(key2, gsl::not_null<std::shared_ptr<Replica>>{replica2}).ok());

  StableDramCacheManager::AdmissionRequest request2;
  request2.key = key2;
  request2.replica = replica2;
  request2.size_bytes = kChunkBytes;
  request2.policy = MakePolicy(StableRetentionPolicy::kBestEffort, StableOverflowPolicy::kSpill);
  auto admit2 = cache.admit(request2);
  REQUIRE_FALSE(admit2.ok());
  REQUIRE(absl::IsResourceExhausted(admit2.status()));
  REQUIRE(replica1->get_memory_state(MemoryLocation::CPU) == MemoryState::LOADED);
}

TEST_CASE("StableDramCacheManager spill rejects when no durable entries exist", "[stable_cache]") {
  auto pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(1 << 20, 1 << 20);
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  auto budget = std::make_shared<MemoryTierBudget>(kChunkBytes, 0);
  ReplicaRegistry registry;

  StableDramCacheManager cache(
      StableDramCacheManager::Config{
          .registry = gsl::not_null<ReplicaRegistry*>{&registry},
          .memory_tier_budget = budget,
          .spill_guard = [](const ReplicaKey&) { return absl::OkStatus(); },
          .spill_evictable = [](const ReplicaKey&, const StableDramCachePolicy&) { return false; },
      });

  auto replica1 = MakeCpuReplica(
      "artifact-spill-nodurable-1",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey key1 = replica1->replica_key();
  REQUIRE(registry.emplace(key1, gsl::not_null<std::shared_ptr<Replica>>{replica1}).ok());

  StableDramCacheManager::AdmissionRequest request1;
  request1.key = key1;
  request1.replica = replica1;
  request1.size_bytes = kChunkBytes;
  request1.policy = MakePolicy(StableRetentionPolicy::kBestEffort, StableOverflowPolicy::kSpill);
  auto admit1 = cache.admit(request1);
  REQUIRE(admit1.ok());
  REQUIRE(admit1->admitted);

  auto replica2 = MakeCpuReplica(
      "artifact-spill-nodurable-2",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey key2 = replica2->replica_key();
  REQUIRE(registry.emplace(key2, gsl::not_null<std::shared_ptr<Replica>>{replica2}).ok());

  StableDramCacheManager::AdmissionRequest request2;
  request2.key = key2;
  request2.replica = replica2;
  request2.size_bytes = kChunkBytes;
  request2.policy = MakePolicy(StableRetentionPolicy::kBestEffort, StableOverflowPolicy::kSpill);
  auto admit2 = cache.admit(request2);
  REQUIRE_FALSE(admit2.ok());
  REQUIRE(absl::IsResourceExhausted(admit2.status()));
  REQUIRE(replica1->get_memory_state(MemoryLocation::CPU) == MemoryState::LOADED);
}

TEST_CASE("StableDramCacheManager spill evicts when shared disk is available", "[stable_cache]") {
  auto pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(1 << 20, 1 << 20);
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  auto budget = std::make_shared<MemoryTierBudget>(kChunkBytes, 0);
  ReplicaRegistry registry;

  StableDramCacheManager cache(
      StableDramCacheManager::Config{
          .registry = gsl::not_null<ReplicaRegistry*>{&registry},
          .memory_tier_budget = budget,
          .spill_guard = [](const ReplicaKey&) { return absl::OkStatus(); },
          .spill_evictable = [](const ReplicaKey& key,
                                const StableDramCachePolicy&) { return key.artifact_id == "artifact-spill-ok-1"; },
      });

  auto replica1 = MakeCpuReplica(
      "artifact-spill-ok-1",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey key1 = replica1->replica_key();
  REQUIRE(registry.emplace(key1, gsl::not_null<std::shared_ptr<Replica>>{replica1}).ok());

  StableDramCacheManager::AdmissionRequest request1;
  request1.key = key1;
  request1.replica = replica1;
  request1.size_bytes = kChunkBytes;
  request1.policy = MakePolicy(StableRetentionPolicy::kBestEffort, StableOverflowPolicy::kSpill);
  auto admit1 = cache.admit(request1);
  REQUIRE(admit1.ok());
  REQUIRE(admit1->admitted);

  auto replica2 = MakeCpuReplica(
      "artifact-spill-ok-2",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey key2 = replica2->replica_key();
  REQUIRE(registry.emplace(key2, gsl::not_null<std::shared_ptr<Replica>>{replica2}).ok());

  StableDramCacheManager::AdmissionRequest request2;
  request2.key = key2;
  request2.replica = replica2;
  request2.size_bytes = kChunkBytes;
  request2.policy = MakePolicy(StableRetentionPolicy::kBestEffort, StableOverflowPolicy::kSpill);
  auto admit2 = cache.admit(request2);
  REQUIRE(admit2.ok());
  REQUIRE(admit2->admitted);
  REQUIRE(replica1->get_memory_state(MemoryLocation::CPU) == MemoryState::UNALLOCATED);
  REQUIRE(replica2->get_memory_state(MemoryLocation::CPU) == MemoryState::LOADED);
}

TEST_CASE("StableDramCacheManager admits logical alias key for physical UMA replica", "[stable_cache]") {
  auto pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(1 << 20, 1 << 20);
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  auto budget = std::make_shared<MemoryTierBudget>(kChunkBytes * 2, 0);
  ReplicaRegistry registry;

  StableDramCacheManager cache(
      StableDramCacheManager::Config{
          .registry = gsl::not_null<ReplicaRegistry*>{&registry},
          .memory_tier_budget = budget,
      });

  auto replica = MakeCpuReplica(
      "mem_reg:alias",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey physical_key = replica->replica_key();

  ReplicaKey logical_key = physical_key;
  logical_key.artifact_id = "cgid:alias";
  REQUIRE(registry.emplace(logical_key, gsl::not_null<std::shared_ptr<Replica>>{replica}).ok());

  StableDramCacheManager::AdmissionRequest request;
  request.key = logical_key;
  request.replica = replica;
  request.size_bytes = kChunkBytes;
  request.policy = MakePolicy(StableRetentionPolicy::kPinned, StableOverflowPolicy::kEvict);
  auto admit = cache.admit(request);
  REQUIRE(admit.ok());
  REQUIRE(admit->admitted);
  REQUIRE(cache.bytes_used() > 0);

  REQUIRE_FALSE(cache.is_evictable(logical_key, absl::Now() + absl::Hours(1)));

  cache.on_replica_evicted(physical_key, "alias_test");
  REQUIRE(cache.bytes_used() == 0);
}

TEST_CASE("StableDramCacheManager evicts alias entry when LRU key is physical", "[stable_cache]") {
  auto pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(1 << 20, 1 << 20);
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  auto budget = std::make_shared<MemoryTierBudget>(kChunkBytes, 0);
  ReplicaRegistry registry;

  StableDramCacheManager cache(
      StableDramCacheManager::Config{
          .registry = gsl::not_null<ReplicaRegistry*>{&registry},
          .memory_tier_budget = budget,
      });

  auto replica = MakeCpuReplica(
      "mem_reg:evict-alias",
      kChunkBytes,
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime},
      budget);
  const ReplicaKey physical_key = replica->replica_key();

  ReplicaKey logical_key = physical_key;
  logical_key.artifact_id = "cgid:evict-alias";
  REQUIRE(registry.emplace(physical_key, gsl::not_null<std::shared_ptr<Replica>>{replica}).ok());
  REQUIRE(registry.emplace(logical_key, gsl::not_null<std::shared_ptr<Replica>>{replica}).ok());

  // Make physical alias the canonical LRU identity chosen by registry de-dup.
  auto touch_or = registry.find(physical_key);
  REQUIRE(touch_or.ok());

  StableDramCacheManager::AdmissionRequest request;
  request.key = logical_key;
  request.replica = replica;
  request.size_bytes = kChunkBytes;
  request.policy = MakePolicy(StableRetentionPolicy::kBestEffort, StableOverflowPolicy::kEvict);
  auto admit = cache.admit(request);
  REQUIRE(admit.ok());
  REQUIRE(admit->admitted);

  auto evict_status = cache.preempt_for_export(kChunkBytes, logical_key);
  REQUIRE(evict_status.ok());
  REQUIRE(cache.bytes_used() == 0);
  REQUIRE(replica->get_memory_state(MemoryLocation::CPU) == MemoryState::UNALLOCATED);
}
