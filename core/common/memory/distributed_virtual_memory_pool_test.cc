// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#define private public
#define protected public
#include "core/common/memory/distributed_virtual_memory_pool.h"
#undef private
#undef protected
#include "core/common/system_capabilities.h"
#include "core/store/replica/chunk_meta.h"

using namespace stepcast::memory;
using namespace stepcast::store;
using namespace stepcast::common;

namespace {
// Check if we're running in a CI environment with limited resources
bool is_ci_environment() {
  // Check common CI environment variables
  const char* ci_env = std::getenv("CI");
  const char* github_actions = std::getenv("GITHUB_ACTIONS");
  const char* use_fake_cuda = std::getenv("USE_FAKE_CUDA");

  return (ci_env && std::string(ci_env) == "true") || (github_actions && std::string(github_actions) == "true") ||
      (use_fake_cuda && std::string(use_fake_cuda) == "1");
}

// Get appropriate test allocation size based on environment
size_t get_large_test_size() {
  if (is_ci_environment()) {
    // Use 10GB in CI instead of 670GB
    return 10ULL * 1024 * 1024 * 1024;
  }
  // Use 670GB in local development
  return 670ULL * 1024 * 1024 * 1024;
}
} // namespace

TEST_CASE("DistributedVirtualMemoryPool basic operations", "[dvmp]") {
  DistributedVirtualMemoryPool dvmp(DistributedVirtualMemoryPool::kDefaultChunkSize);

  SECTION("Allocate and deallocate") {
    const size_t size_256mb = 256ULL * 1024 * 1024;
    auto region_or = dvmp.allocate("model1", size_256mb);
    REQUIRE(region_or.ok());
    auto region = *region_or;
    REQUIRE(region.cpu_base != nullptr);
    REQUIRE(region.bytes == size_256mb);

    // Duplicate allocation should fail
    auto dup_or = dvmp.allocate("model1", size_256mb);
    REQUIRE(!dup_or.ok());
    REQUIRE(dup_or.status().code() == absl::StatusCode::kAlreadyExists);
  }

  SECTION("Large allocation") {
    const size_t large_size = get_large_test_size();
    auto region_or = dvmp.allocate("large_artifact", large_size);

    if (!region_or.ok()) {
      // Large allocations may fail on resource-constrained environments (including some CI)
      WARN("Skipping large allocation test due to resource limits");
      return;
    }

    // Proceed only when allocation succeeded
    REQUIRE(region_or.ok());
    auto region = *region_or;
    REQUIRE(region.cpu_base != nullptr);
    REQUIRE(region.bytes == large_size);

    // Verify chunk count
    auto snapshot = dvmp.chunk_snapshot("large_artifact");
    REQUIRE(
        snapshot.size() ==
        (large_size + DistributedVirtualMemoryPool::kDefaultChunkSize - 1) /
            DistributedVirtualMemoryPool::kDefaultChunkSize);
  }

  SECTION("Chunk snapshot") {
    const size_t size_1gb = 1024ULL * 1024 * 1024;
    auto region_or = dvmp.allocate("model2", size_1gb);
    REQUIRE(region_or.ok());

    auto snapshot = dvmp.chunk_snapshot("model2");
    REQUIRE(snapshot.size() == 4); // 1GB = 4 * 256MB

    // All chunks should start as COLD
    for (const auto& meta : snapshot) {
      REQUIRE(meta.state.load() == ChunkState::COLD);
      REQUIRE(meta.last_touch_s.load() == 0);
    }

    // Non-existent replica should return empty span
    auto empty = dvmp.chunk_snapshot("nonexistent");
    REQUIRE(empty.empty());
  }
}

TEST_CASE("DistributedVirtualMemoryPool chunk locking", "[dvmp]") {
  DistributedVirtualMemoryPool dvmp(DistributedVirtualMemoryPool::kDefaultChunkSize);
  const size_t size_1gb = 1024ULL * 1024 * 1024;
  auto region_or = dvmp.allocate("lock_test", size_1gb);
  REQUIRE(region_or.ok());

  SECTION("Lock and unlock chunks") {
    std::vector<uint32_t> indices = {0, 1, 2};

    // Lock chunks
    auto lock_status = dvmp.lock_chunks("lock_test", indices);
    REQUIRE(lock_status.ok());

    // Verify state changed to LOCKED_TX
    auto snapshot = dvmp.chunk_snapshot("lock_test");
    for (uint32_t i : indices) {
      REQUIRE(snapshot[i].state.load() == ChunkState::LOCKED_TX);
    }
    REQUIRE(snapshot[3].state.load() == ChunkState::COLD); // Unlocked chunk

    // Try to lock already locked chunks should fail
    auto relock_status = dvmp.lock_chunks("lock_test", {0, 1});
    REQUIRE(!relock_status.ok());
    REQUIRE(relock_status.code() == absl::StatusCode::kResourceExhausted);

    // Unlock with copied_gpu=true
    auto unlock_status = dvmp.unlock_chunks("lock_test", indices, true);
    REQUIRE(unlock_status.ok());

    // Verify state changed to COPIED_GPU
    snapshot = dvmp.chunk_snapshot("lock_test");
    for (uint32_t i : indices) {
      REQUIRE(snapshot[i].state.load() == ChunkState::COPIED_GPU);
      REQUIRE(snapshot[i].last_touch_s.load() > 0);
    }
  }

  SECTION("Lock out of range chunks") {
    std::vector<uint32_t> indices = {0, 10}; // 10 is out of range
    auto lock_status = dvmp.lock_chunks("lock_test", indices);
    REQUIRE(!lock_status.ok());
    REQUIRE(lock_status.code() == absl::StatusCode::kOutOfRange);

    // Verify no chunks were locked (rollback)
    auto snapshot = dvmp.chunk_snapshot("lock_test");
    REQUIRE(snapshot[0].state.load() == ChunkState::COLD);
  }

  SECTION("Unlock not locked chunks") {
    auto unlock_status = dvmp.unlock_chunks("lock_test", {0}, false);
    REQUIRE(!unlock_status.ok());
    REQUIRE(unlock_status.code() == absl::StatusCode::kFailedPrecondition);
  }
}

TEST_CASE("DistributedVirtualMemoryPool eviction", "[dvmp]") {
  DistributedVirtualMemoryPool dvmp(DistributedVirtualMemoryPool::kDefaultChunkSize);
  const size_t size_1gb = 1024ULL * 1024 * 1024;
  auto region_or = dvmp.allocate("evict_test", size_1gb);
  REQUIRE(region_or.ok());

  SECTION("Evict tail bytes") {
    // Mark some chunks as different states
    REQUIRE(dvmp.lock_chunks("evict_test", {0}).ok());
    REQUIRE(dvmp.unlock_chunks("evict_test", {0}, true).ok()); // COPIED_GPU
    REQUIRE(dvmp.lock_chunks("evict_test", {1}).ok());
    REQUIRE(dvmp.unlock_chunks("evict_test", {1}, false).ok()); // HOT
    // Chunk 2,3 remain COLD

    // Evict 512MB (2 chunks) from tail
    size_t evicted = dvmp.evict_tail_bytes("evict_test", 512ULL * 1024 * 1024);
    REQUIRE(evicted == 512ULL * 1024 * 1024);

    // Verify eviction from tail (chunks 3 and 2)
    auto snapshot = dvmp.chunk_snapshot("evict_test");
    REQUIRE(snapshot[3].state.load() == ChunkState::EVICTED);
    REQUIRE(snapshot[2].state.load() == ChunkState::EVICTED);
    REQUIRE(snapshot[1].state.load() == ChunkState::HOT);
    REQUIRE(snapshot[0].state.load() == ChunkState::COPIED_GPU);
  }

  SECTION("Cannot evict locked chunks") {
    REQUIRE(dvmp.lock_chunks("evict_test", {3}).ok()); // Lock last chunk

    // Try to evict 256MB from tail (will skip locked chunk 3 and evict chunk 2)
    size_t evicted = dvmp.evict_tail_bytes("evict_test", 256ULL * 1024 * 1024);
    REQUIRE(evicted == 256ULL * 1024 * 1024); // Evicted chunk 2, skipped locked chunk 3

    auto snapshot = dvmp.chunk_snapshot("evict_test");
    REQUIRE(snapshot[3].state.load() == ChunkState::LOCKED_TX); // Still locked
    REQUIRE(snapshot[2].state.load() == ChunkState::EVICTED); // Evicted
  }

  SECTION("Evict more than available") {
    // Try to evict 2GB from 1GB replica
    size_t evicted = dvmp.evict_tail_bytes("evict_test", 2ULL * 1024 * 1024 * 1024);
    REQUIRE(evicted == size_1gb); // Should evict all 4 chunks

    auto snapshot = dvmp.chunk_snapshot("evict_test");
    for (const auto& meta : snapshot) {
      REQUIRE(meta.state.load() == ChunkState::EVICTED);
    }
  }
}

TEST_CASE("DistributedVirtualMemoryPool preemptible marking", "[dvmp]") {
  DistributedVirtualMemoryPool dvmp(DistributedVirtualMemoryPool::kDefaultChunkSize);
  const size_t size_1gb = 1024ULL * 1024 * 1024;
  auto region_or = dvmp.allocate("preempt_test", size_1gb);
  REQUIRE(region_or.ok());

  SECTION("Mark chunks as preemptible") {
    std::vector<uint32_t> indices = {0, 1};

    // Initially chunks are COLD
    auto snapshot = dvmp.chunk_snapshot("preempt_test");
    REQUIRE(snapshot[0].state.load() == ChunkState::COLD);
    REQUIRE(snapshot[1].state.load() == ChunkState::COLD);

    // Mark as preemptible
    auto status = dvmp.mark_preemptible("preempt_test", indices);
    REQUIRE(status.ok());

    // Verify state changed
    snapshot = dvmp.chunk_snapshot("preempt_test");
    REQUIRE(snapshot[0].state.load() == ChunkState::PREEMPTIBLE);
    REQUIRE(snapshot[1].state.load() == ChunkState::PREEMPTIBLE);
  }

  SECTION("Cannot mark non-HOT/COLD chunks as preemptible") {
    // Lock a chunk
    REQUIRE(dvmp.lock_chunks("preempt_test", {0}).ok());

    // Try to mark locked chunk as preemptible
    auto status = dvmp.mark_preemptible("preempt_test", {0});
    REQUIRE(status.ok()); // Operation succeeds but state doesn't change

    auto snapshot = dvmp.chunk_snapshot("preempt_test");
    REQUIRE(snapshot[0].state.load() == ChunkState::LOCKED_TX); // Still locked
  }
}

TEST_CASE("DistributedVirtualMemoryPool ensure resident", "[dvmp]") {
  DistributedVirtualMemoryPool dvmp(DistributedVirtualMemoryPool::kDefaultChunkSize);
  const size_t size_512mb = 512ULL * 1024 * 1024;
  auto region_or = dvmp.allocate("resident_test", size_512mb);
  REQUIRE(region_or.ok());

  SECTION("Check resident chunks") {
    // Initially all chunks are COLD (resident)
    auto status = dvmp.ensure_chunk_resident("resident_test", 0);
    REQUIRE(status.ok());

    status = dvmp.ensure_chunk_resident("resident_test", 1);
    REQUIRE(status.ok());
  }

  SECTION("Check evicted chunks") {
    // Evict first chunk
    dvmp.evict_tail_bytes("resident_test", DistributedVirtualMemoryPool::kDefaultChunkSize);

    // Check residency
    auto status = dvmp.ensure_chunk_resident("resident_test", 1); // Last chunk (evicted)
    REQUIRE(!status.ok());
    REQUIRE(status.code() == DistributedVirtualMemoryPool::kErrChunkRemote);

    status = dvmp.ensure_chunk_resident("resident_test", 0); // First chunk (not evicted)
    REQUIRE(status.ok());
  }

  SECTION("Out of range chunk") {
    auto status = dvmp.ensure_chunk_resident("resident_test", 10);
    REQUIRE(!status.ok());
    REQUIRE(status.code() == absl::StatusCode::kOutOfRange);
  }
}

TEST_CASE("DistributedVirtualMemoryPool concurrent operations", "[dvmp]") {
  DistributedVirtualMemoryPool dvmp(DistributedVirtualMemoryPool::kDefaultChunkSize);
  const size_t size_1gb = 1024ULL * 1024 * 1024;
  auto region_or = dvmp.allocate("concurrent_test", size_1gb);
  REQUIRE(region_or.ok());

  SECTION("Concurrent lock/unlock") {
    const int num_threads = 4;
    const int iterations = 100;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> chunk_dist(0, 3);

        for (int i = 0; i < iterations; ++i) {
          std::vector<uint32_t> chunks = {static_cast<uint32_t>(chunk_dist(gen))};

          auto lock_status = dvmp.lock_chunks("concurrent_test", chunks);
          if (lock_status.ok()) {
            // Simulate work
            std::this_thread::sleep_for(std::chrono::microseconds(10));

            auto unlock_status = dvmp.unlock_chunks("concurrent_test", chunks, i % 2 == 0);
            REQUIRE(unlock_status.ok());
            success_count++;
          }
        }
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    // Should have some successful operations
    REQUIRE(success_count > 0);

    // Verify final state consistency
    auto snapshot = dvmp.chunk_snapshot("concurrent_test");
    for (const auto& meta : snapshot) {
      auto state = meta.state.load();
      // Should be in a valid state (not LOCKED_TX)
      REQUIRE(state != ChunkState::LOCKED_TX);
    }
  }

  SECTION("Concurrent eviction and refresh") {
    std::thread evict_thread([&]() {
      for (int i = 0; i < 10; ++i) {
        dvmp.evict_tail_bytes("concurrent_test", DistributedVirtualMemoryPool::kDefaultChunkSize);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });

    std::thread refresh_thread([&]() {
      for (int i = 0; i < 20; ++i) {
        std::vector<uint32_t> chunks = {0, 1, 2, 3};
        dvmp.refresh_chunks("concurrent_test", chunks);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });

    evict_thread.join();
    refresh_thread.join();

    // System should still be in consistent state
    auto snapshot = dvmp.chunk_snapshot("concurrent_test");
    REQUIRE(snapshot.size() == 4);
  }
}

TEST_CASE("DistributedVirtualMemoryPool state transitions", "[dvmp]") {
  DistributedVirtualMemoryPool dvmp(DistributedVirtualMemoryPool::kDefaultChunkSize);
  const size_t size_512mb = 512ULL * 1024 * 1024;
  auto region_or = dvmp.allocate("state_test", size_512mb);
  REQUIRE(region_or.ok());

  SECTION("State machine transitions") {
    // COLD -> LOCKED_TX
    auto status = dvmp.lock_chunks("state_test", {0});
    REQUIRE(status.ok());
    auto snapshot = dvmp.chunk_snapshot("state_test");
    REQUIRE(snapshot[0].state.load() == ChunkState::LOCKED_TX);

    // LOCKED_TX -> HOT
    status = dvmp.unlock_chunks("state_test", {0}, false);
    REQUIRE(status.ok());
    snapshot = dvmp.chunk_snapshot("state_test");
    REQUIRE(snapshot[0].state.load() == ChunkState::HOT);

    // HOT -> PREEMPTIBLE
    status = dvmp.mark_preemptible("state_test", {0});
    REQUIRE(status.ok());
    snapshot = dvmp.chunk_snapshot("state_test");
    REQUIRE(snapshot[0].state.load() == ChunkState::PREEMPTIBLE);

    // PREEMPTIBLE -> LOCKED_TX
    status = dvmp.lock_chunks("state_test", {0});
    REQUIRE(status.ok());
    snapshot = dvmp.chunk_snapshot("state_test");
    REQUIRE(snapshot[0].state.load() == ChunkState::LOCKED_TX);

    // LOCKED_TX -> COPIED_GPU
    status = dvmp.unlock_chunks("state_test", {0}, true);
    REQUIRE(status.ok());
    snapshot = dvmp.chunk_snapshot("state_test");
    REQUIRE(snapshot[0].state.load() == ChunkState::COPIED_GPU);

    // COPIED_GPU -> EVICTED
    auto evicted = dvmp.evict_tail_bytes("state_test", 256ULL * 1024 * 1024);
    REQUIRE(evicted >= DistributedVirtualMemoryPool::kDefaultChunkSize);
    // Since eviction works from tail, check if any chunk is evicted
    bool has_evicted = false;
    snapshot = dvmp.chunk_snapshot("state_test");
    for (const auto& meta : snapshot) {
      if (meta.state.load() == ChunkState::EVICTED) {
        has_evicted = true;
        break;
      }
    }
    REQUIRE(has_evicted);
  }
}

TEST_CASE("DistributedVirtualMemoryPool error handling", "[dvmp]") {
  DistributedVirtualMemoryPool dvmp(DistributedVirtualMemoryPool::kDefaultChunkSize);

  SECTION("Operations on non-existent replica") {
    auto status = dvmp.lock_chunks("nonexistent", {0});
    REQUIRE(!status.ok());
    REQUIRE(status.code() == absl::StatusCode::kNotFound);

    status = dvmp.unlock_chunks("nonexistent", {0}, false);
    REQUIRE(!status.ok());
    REQUIRE(status.code() == absl::StatusCode::kNotFound);

    auto evicted = dvmp.evict_tail_bytes("nonexistent", 1024);
    REQUIRE(evicted == 0);

    status = dvmp.ensure_chunk_resident("nonexistent", 0);
    REQUIRE(!status.ok());
    REQUIRE(status.code() == absl::StatusCode::kNotFound);

    status = dvmp.mark_preemptible("nonexistent", {0});
    REQUIRE(!status.ok());
    REQUIRE(status.code() == absl::StatusCode::kNotFound);
  }

  SECTION("Empty indices") {
    const size_t size_256mb = 256ULL * 1024 * 1024;
    auto region_or = dvmp.allocate("empty_test", size_256mb);
    REQUIRE(region_or.ok());

    std::vector<uint32_t> empty_indices;
    auto status = dvmp.lock_chunks("empty_test", empty_indices);
    REQUIRE(status.ok()); // Empty operation succeeds

    status = dvmp.unlock_chunks("empty_test", empty_indices, false);
    REQUIRE(status.ok()); // Empty operation succeeds
  }
}

TEST_CASE("DVMP mlock refcount integrates locks and leases", "[dvmp][mlock]") {
  SystemCapabilities::instance().set_mlock_enabled(true);
  DistributedVirtualMemoryPool dvmp(4096); // 4 KiB chunks
  auto region_or = dvmp.allocate("mlock_refcnt", 8192);
  REQUIRE(region_or.ok());

  auto lease_or = dvmp.pin_range("mlock_refcnt", 0, 4096, "test");
  REQUIRE(lease_or.ok());
  DistributedVirtualMemoryPool::ChunkResidencyLease lease = std::move(*lease_or);

  auto info_sp_or = dvmp.get_artifact_info("mlock_refcnt");
  REQUIRE(info_sp_or.ok());
  const auto& info_sp = *info_sp_or;
  {
    std::lock_guard<std::mutex> guard(info_sp->artifact_mutex);
    REQUIRE(info_sp->pin_refcnt[0].load() == 1);
    REQUIRE(info_sp->mlock_refcnt[0].load() == 1);
  }

  REQUIRE(dvmp.lock_chunks("mlock_refcnt", {0}).ok());
  {
    std::lock_guard<std::mutex> guard(info_sp->artifact_mutex);
    REQUIRE(info_sp->mlock_refcnt[0].load() == 2);
  }

  REQUIRE(dvmp.unlock_chunks("mlock_refcnt", {0}, false).ok());
  {
    std::lock_guard<std::mutex> guard(info_sp->artifact_mutex);
    REQUIRE(info_sp->mlock_refcnt[0].load() == 1);
    REQUIRE(info_sp->pin_refcnt[0].load() == 1);
  }

  lease = {};
  {
    std::lock_guard<std::mutex> guard(info_sp->artifact_mutex);
    REQUIRE(info_sp->mlock_refcnt[0].load() == 0);
    REQUIRE(info_sp->pin_refcnt[0].load() == 0);
  }
}