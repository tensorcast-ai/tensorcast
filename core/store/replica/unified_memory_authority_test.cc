// Copyright (c) 2025-2026, TensorCast Team.

// Basic UMA unit test validating allocation and chunk mapping state.

#include "core/store/replica/unified_memory_authority.h"

#include <chrono>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/common/const/granularity.h"
#include "core/store/memory_tier_budget.h"

using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::replica::UnifiedMemoryAuthority;

TEST_CASE("UMA allocate + mappings + get_missing_chunks", "[uma]") {
  UnifiedMemoryAuthority uma(tensorcast::common::consts::kArtifactChunkDefault);

  ReplicaKey key{
      .artifact_id = std::string("uma_unit_test"),
      .view_id = std::nullopt,
      .device = {tensorcast::DeviceType::CPU, -1, ""},
      .replica = 0};

  // Allocate 2 chunks worth of CPU memory via UMA (through VS)
  auto st = uma.allocate(key, tensorcast::common::consts::kArtifactChunkDefault * 2);
  REQUIRE(st.ok());

  // Initially, CPU missing chunks for target GPU should be all chunks
  auto miss_gpu = uma.get_missing_chunks(key, MemoryLocation::GPU, /*device_id*/ 0);
  REQUIRE(miss_gpu.size() >= 2);

  // After a synthetic plan/commit, GPU should have fewer missing chunks
  std::vector<uint32_t> first_chunk{0};
  auto plan_or = uma.plan_load(key, MemoryLocation::GPU, /*device_id*/ 0, absl::MakeSpan(first_chunk));
  REQUIRE(plan_or.ok());
  auto cst = uma.commit(plan_or->session_id, MemoryLocation::GPU, absl::MakeSpan(first_chunk), /*device_id*/ 0);
  REQUIRE(cst.ok());
  auto miss_gpu2 = uma.get_missing_chunks(key, MemoryLocation::GPU, /*device_id*/ 0);
  REQUIRE(miss_gpu2.size() + 1 == miss_gpu.size());
}

TEST_CASE("UMA stable lease guards against preemptible marking", "[uma][stable]") {
  UnifiedMemoryAuthority uma(tensorcast::common::consts::kArtifactChunkDefault);

  ReplicaKey key{
      .artifact_id = std::string("uma_stable_test"),
      .view_id = std::nullopt,
      .device = {tensorcast::DeviceType::CPU, -1, ""},
      .replica = 0};

  const uint64_t bytes = tensorcast::common::consts::kArtifactChunkDefault * 2;
  REQUIRE(uma.allocate(key, bytes).ok());

  auto lease_or = uma.acquire_stable_lease(key, std::vector<uint32_t>{0});
  REQUIRE(lease_or.ok());
  auto state0 = uma.get_cpu_chunk_state(key, 0);
  REQUIRE(state0.ok());
  REQUIRE(*state0 == tensorcast::store::replica::ChunkState::STABLE);

  auto ledger_before = uma.get_ledger_version(key);
  REQUIRE(ledger_before.ok());

  REQUIRE(uma.mark_cpu_chunks_preemptible(key, 1.0F).ok());
  auto state0_after = uma.get_cpu_chunk_state(key, 0);
  auto state1_after = uma.get_cpu_chunk_state(key, 1);
  REQUIRE(state0_after.ok());
  REQUIRE(state1_after.ok());
  REQUIRE(*state0_after == tensorcast::store::replica::ChunkState::STABLE);
  REQUIRE(*state1_after == tensorcast::store::replica::ChunkState::PREEMPTIBLE);

  auto ledger_after = uma.get_ledger_version(key);
  REQUIRE(ledger_after.ok());
  REQUIRE(*ledger_after > *ledger_before);

  REQUIRE(uma.release_stable_lease(*lease_or).ok());
  auto released_state = uma.get_cpu_chunk_state(key, 0);
  REQUIRE(released_state.ok());
  REQUIRE(*released_state == tensorcast::store::replica::ChunkState::HOT);
}

TEST_CASE("UMA blocks GPU release while GPU export is active", "[uma][gpu_export]") {
  const size_t chunk_bytes = tensorcast::common::consts::kArtifactChunkDefault;
  UnifiedMemoryAuthority uma(chunk_bytes);

  ReplicaKey key{
      .artifact_id = std::string("uma_gpu_export_guard"),
      .view_id = std::nullopt,
      .device = {tensorcast::DeviceType::GPU, 0, ""},
      .replica = 0};

  REQUIRE(uma.allocate(key, chunk_bytes * 2).ok());

  auto export_on = uma.set_exported(key, MemoryLocation::GPU, std::vector<uint32_t>{0}, /*on=*/true);
  REQUIRE(export_on.ok());

  auto release_while_exported = uma.release_gpu_device(key, /*device_id=*/0, /*drop_allocation=*/true);
  REQUIRE_FALSE(release_while_exported.ok());
  REQUIRE(release_while_exported.code() == absl::StatusCode::kFailedPrecondition);

  auto export_off = uma.set_exported(key, MemoryLocation::GPU, std::vector<uint32_t>{0}, /*on=*/false);
  REQUIRE(export_off.ok());

  auto release_after_unexport = uma.release_gpu_device(key, /*device_id=*/0, /*drop_allocation=*/true);
  REQUIRE(release_after_unexport.ok());
}

TEST_CASE("UMA stable lease guards against preemptible marking (memfd)", "[uma][stable][memfd]") {
  constexpr size_t kChunkBytes = 1ULL << 20;
  UnifiedMemoryAuthority uma(kChunkBytes, UnifiedMemoryAuthority::Options{.cpu_shared_memory_enabled = true});

  ReplicaKey key{
      .artifact_id = std::string("uma_stable_memfd_test"),
      .view_id = std::nullopt,
      .device = {tensorcast::DeviceType::CPU, -1, ""},
      .replica = 0};

  const uint64_t bytes = static_cast<uint64_t>(kChunkBytes) * 2;
  REQUIRE(uma.allocate(key, bytes).ok());

  auto lease_or = uma.acquire_stable_lease(key, std::vector<uint32_t>{0});
  REQUIRE(lease_or.ok());
  auto state0 = uma.get_cpu_chunk_state(key, 0);
  REQUIRE(state0.ok());
  REQUIRE(*state0 == tensorcast::store::replica::ChunkState::STABLE);

  auto ledger_before = uma.get_ledger_version(key);
  REQUIRE(ledger_before.ok());

  REQUIRE(uma.mark_cpu_chunks_preemptible(key, 1.0F).ok());
  auto state0_after = uma.get_cpu_chunk_state(key, 0);
  auto state1_after = uma.get_cpu_chunk_state(key, 1);
  REQUIRE(state0_after.ok());
  REQUIRE(state1_after.ok());
  REQUIRE(*state0_after == tensorcast::store::replica::ChunkState::STABLE);
  REQUIRE(*state1_after == tensorcast::store::replica::ChunkState::PREEMPTIBLE);

  auto ledger_after = uma.get_ledger_version(key);
  REQUIRE(ledger_after.ok());
  REQUIRE(*ledger_after > *ledger_before);

  REQUIRE(uma.release_stable_lease(*lease_or).ok());
  auto released_state = uma.get_cpu_chunk_state(key, 0);
  REQUIRE(released_state.ok());
  REQUIRE(*released_state == tensorcast::store::replica::ChunkState::HOT);
}

TEST_CASE("UMA stable lease admission failure does not advance ledger", "[uma][stable]") {
  UnifiedMemoryAuthority uma(tensorcast::common::consts::kArtifactChunkDefault);
  auto budget =
      std::make_shared<tensorcast::store::MemoryTierBudget>(tensorcast::common::consts::kArtifactChunkDefault / 2, 0);
  uma.set_memory_tier_budget(budget);

  ReplicaKey key{
      .artifact_id = std::string("uma_stable_budget_fail"),
      .view_id = std::nullopt,
      .device = {tensorcast::DeviceType::CPU, -1, ""},
      .replica = 0};

  REQUIRE(uma.allocate(key, tensorcast::common::consts::kArtifactChunkDefault).ok());

  auto ledger_before = uma.get_ledger_version(key);
  REQUIRE(ledger_before.ok());

  auto lease_or = uma.acquire_stable_lease(key, std::vector<uint32_t>{0});
  REQUIRE_FALSE(lease_or.ok());

  auto ledger_after = uma.get_ledger_version(key);
  REQUIRE(ledger_after.ok());
  REQUIRE(*ledger_after == *ledger_before);

  auto chunk_state = uma.get_cpu_chunk_state(key, 0);
  REQUIRE(chunk_state.ok());
  REQUIRE(*chunk_state != tensorcast::store::replica::ChunkState::STABLE);
}

TEST_CASE("UMA preemptible chunks rehydrate and report telemetry", "[uma][preemptible]") {
  const size_t chunk_bytes = tensorcast::common::consts::kArtifactChunkDefault;
  UnifiedMemoryAuthority uma(chunk_bytes);
  auto budget = std::make_shared<tensorcast::store::MemoryTierBudget>(chunk_bytes * 2, chunk_bytes);
  uma.set_memory_tier_budget(budget);

  ReplicaKey key{
      .artifact_id = std::string("uma_preemptible_fault"),
      .view_id = std::nullopt,
      .device = {tensorcast::DeviceType::CPU, -1, ""},
      .replica = 0};

  REQUIRE(uma.allocate(key, chunk_bytes).ok());

  // Mark the chunk as loaded on CPU.
  auto plan_or = uma.plan_load(key, MemoryLocation::CPU, std::nullopt, std::nullopt);
  REQUIRE(plan_or.ok());
  REQUIRE_FALSE(plan_or->chunk_indices.empty());
  auto commit_st =
      uma.commit(plan_or->session_id, MemoryLocation::CPU, absl::MakeSpan(plan_or->chunk_indices), std::nullopt);
  REQUIRE(commit_st.ok());

  // Make the chunk preemptible and let UMA observe the eviction via mincore.
  REQUIRE(uma.mark_cpu_chunks_preemptible(key, 1.0F).ok());
  auto missing = uma.get_missing_chunks(key, MemoryLocation::CPU, std::nullopt);
  REQUIRE(missing.size() == plan_or->chunk_indices.size());

  // Rehydrate the missing chunk and ensure telemetry is recorded.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  auto plan_rehydrate = uma.plan_load(key, MemoryLocation::CPU, std::nullopt, absl::MakeSpan(missing));
  REQUIRE(plan_rehydrate.ok());
  auto commit_rehydrate = uma.commit(
      plan_rehydrate->session_id, MemoryLocation::CPU, absl::MakeSpan(plan_rehydrate->chunk_indices), std::nullopt);
  REQUIRE(commit_rehydrate.ok());

  auto snap = budget->snapshot();
  REQUIRE(snap.preemptible_marked_bytes == 0);
  REQUIRE(snap.faults_per_sec > 0.0);
  REQUIRE(snap.rehydrate_p99_ns > 0);
}
