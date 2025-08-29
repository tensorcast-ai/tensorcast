// Copyright (c) 2025, TensorCast Team.

#include "core/store/replica/replica_memory_coordinator.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/store/loading/loading_spec.h"

namespace tensorcast::store {
namespace {

// Mock implementation of DistributedVirtualMemoryPool for testing
class MockDistributedVirtualMemoryPool : public memory::DistributedVirtualMemoryPool {
 public:
  // === Configurable state ===
  VirtualRegion test_region_{};
  absl::Status allocate_status_ = absl::OkStatus();
  absl::Status lock_status_ = absl::OkStatus();
  absl::Status unlock_status_ = absl::OkStatus();
  absl::Status mark_preemptible_status_ = absl::OkStatus();
  std::vector<ChunkState> chunk_states_;

  // === Tracking ===
  std::vector<uint32_t> last_locked_chunks_;
  std::vector<uint32_t> last_unlocked_chunks_;
  std::vector<uint32_t> last_preemptible_chunks_;
  bool last_unlock_mark_preemptible_ = false;

  // === Overrides ===
  absl::StatusOr<VirtualRegion> allocate(std::string_view /*artifact_id*/, size_t bytes, int /*numa*/) override {
    if (!allocate_status_.ok()) {
      return allocate_status_;
    }
    if (test_region_.bytes == 0) {
      test_region_.bytes = bytes;
      test_region_.cpu_base = reinterpret_cast<void*>(0x1000000);
    }
    return test_region_;
  }

  absl::Status lock_chunks(std::string_view /*artifact_id*/, absl::Span<const uint32_t> idx) override {
    last_locked_chunks_.assign(idx.begin(), idx.end());
    return lock_status_;
  }

  absl::Status unlock_chunks(std::string_view /*artifact_id*/, absl::Span<const uint32_t> idx, bool copied_gpu)
      override {
    last_unlocked_chunks_.assign(idx.begin(), idx.end());
    last_unlock_mark_preemptible_ = copied_gpu;
    return unlock_status_;
  }

  absl::Status mark_preemptible(std::string_view /*artifact_id*/, absl::Span<const uint32_t> idx) override {
    last_preemptible_chunks_.assign(idx.begin(), idx.end());
    return mark_preemptible_status_;
  }

  absl::Span<const ChunkMeta> chunk_snapshot(std::string_view /*artifact_id*/) const noexcept override {
    static std::vector<ChunkMeta> meta_cache;
    std::vector<ChunkMeta> local(chunk_states_.size());
    for (size_t i = 0; i < chunk_states_.size(); ++i) {
      local[i].state.store(chunk_states_[i]);
    }
    meta_cache.swap(local);
    return absl::MakeConstSpan(meta_cache);
  }
};

TEST_CASE("ReplicaMemoryCoordinator allocation", "[unified_memory]") {
  auto mock_dvmp = std::make_shared<MockDistributedVirtualMemoryPool>();
  ReplicaMemoryCoordinator unified_memory(mock_dvmp);

  SECTION("successful allocation") {
    ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
    size_t size = size_t(1024) * 1024 * 1024; // 1GB

    REQUIRE(unified_memory.allocate(key, size).ok());

    // Verify chunk mappings were created
    auto mappings = unified_memory.get_chunk_mappings(key);
    size_t expected_chunks = (size + unified_memory.get_chunk_size() - 1) / unified_memory.get_chunk_size();
    REQUIRE(mappings.size() == expected_chunks);
  }

  SECTION("allocation failure") {
    ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
    size_t size = size_t(1024) * 1024 * 1024;

    mock_dvmp->allocate_status_ = absl::ResourceExhaustedError("Out of memory");

    auto status = unified_memory.allocate(key, size);
    REQUIRE(!status.ok());
    REQUIRE(status.code() == absl::StatusCode::kResourceExhausted);
  }
}

TEST_CASE("ReplicaMemoryCoordinator get missing chunks", "[unified_memory]") {
  auto mock_dvmp = std::make_shared<MockDistributedVirtualMemoryPool>();
  ReplicaMemoryCoordinator unified_memory(mock_dvmp);

  ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
  size_t size = size_t(768) * 1024 * 1024; // 768MB = 3 chunks of 256MB

  REQUIRE(unified_memory.allocate(key, size).ok());

  SECTION("missing chunks for GPU") {
    // Simulate some chunks already on GPU
    DeviceKey device_key{DeviceType::GPU, 0, ""};
    REQUIRE(
        unified_memory.update_chunk_states(key, MemoryLocation::GPU, {0, 2}, ChunkState::HOT, device_key.ordinal).ok());

    // Get missing chunks for GPU
    auto missing = unified_memory.get_missing_chunks(key, MemoryLocation::GPU, device_key.ordinal);
    REQUIRE(missing.size() == 1);
    REQUIRE(missing[0] == 1);
  }

  SECTION("missing chunks for CPU") {
    // Mock DVMP chunk states
    mock_dvmp->chunk_states_ = {ChunkState::HOT, ChunkState::EVICTED, ChunkState::HOT};

    auto missing = unified_memory.get_missing_chunks(key, MemoryLocation::PAGEABLE_CPU);
    REQUIRE(missing.size() == 1);
    REQUIRE(missing[0] == 1);
  }
}

TEST_CASE("ReplicaMemoryCoordinator lock chunks for transfer", "[unified_memory]") {
  auto mock_dvmp = std::make_shared<MockDistributedVirtualMemoryPool>();
  ReplicaMemoryCoordinator unified_memory(mock_dvmp);

  ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
  size_t size = size_t(512) * 1024 * 1024; // 512MB = 2 chunks

  REQUIRE(unified_memory.allocate(key, size).ok());

  SECTION("lock CPU chunks for GPU transfer") {
    // Setup initial states
    mock_dvmp->chunk_states_ = {ChunkState::HOT, ChunkState::HOT};

    // Lock chunks for CPU to GPU transfer
    std::vector<uint32_t> chunks_to_lock = {0, 1};
    REQUIRE(
        unified_memory.lock_chunks_for_transfer(key, MemoryLocation::PAGEABLE_CPU, MemoryLocation::GPU, chunks_to_lock)
            .ok());

    // Verify DVMP was called with correct chunks
    REQUIRE(mock_dvmp->last_locked_chunks_ == chunks_to_lock);
  }

  SECTION("lock failure handling") {
    mock_dvmp->lock_status_ = absl::InternalError("Lock failed");

    std::vector<uint32_t> chunks_to_lock = {0};
    auto status =
        unified_memory.lock_chunks_for_transfer(key, MemoryLocation::PAGEABLE_CPU, MemoryLocation::GPU, chunks_to_lock);

    REQUIRE(!status.ok());
    REQUIRE(status.code() == absl::StatusCode::kInternal);
  }
}

TEST_CASE("ReplicaMemoryCoordinator update chunk states", "[unified_memory]") {
  auto mock_dvmp = std::make_shared<MockDistributedVirtualMemoryPool>();
  ReplicaMemoryCoordinator unified_memory(mock_dvmp);

  ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
  size_t size = size_t(512) * 1024 * 1024;
  int device_id = 0;

  REQUIRE(unified_memory.allocate(key, size).ok());

  SECTION("update GPU chunk states") {
    // Update chunk states to COPIED_GPU
    std::vector<uint32_t> chunks = {0, 1};
    REQUIRE(
        unified_memory.update_chunk_states(key, MemoryLocation::GPU, chunks, ChunkState::COPIED_GPU, device_id).ok());

    // Verify GPU states
    auto mappings = unified_memory.get_chunk_mappings(key);
    REQUIRE(mappings[0].gpu_state.at(DeviceKey{DeviceType::GPU, device_id, ""}) == ChunkState::COPIED_GPU);
    REQUIRE(mappings[1].gpu_state.at(DeviceKey{DeviceType::GPU, device_id, ""}) == ChunkState::COPIED_GPU);
  }

  SECTION("update CPU chunk states") {
    std::vector<uint32_t> chunks = {1};
    REQUIRE(unified_memory.update_chunk_states(key, MemoryLocation::PAGEABLE_CPU, chunks, ChunkState::COLD).ok());

    auto mappings = unified_memory.get_chunk_mappings(key);
    REQUIRE(mappings[1].cpu_state == ChunkState::COLD);
  }
}

TEST_CASE("ReplicaMemoryCoordinator GPU allocation management", "[unified_memory]") {
  auto mock_dvmp = std::make_shared<MockDistributedVirtualMemoryPool>();
  ReplicaMemoryCoordinator unified_memory(mock_dvmp);

  ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
  size_t size = size_t(256) * 1024 * 1024;

  REQUIRE(unified_memory.allocate(key, size).ok());

  SECTION("lazy GPU allocation creation") {
    // First access should create allocation
    int device0 = 0;
    auto gpu_alloc0 = unified_memory.get_or_create_gpu_allocation(key, device0);
    REQUIRE(gpu_alloc0.ok());
    REQUIRE(gpu_alloc0.value() != nullptr);

    // Second access should return same allocation
    auto gpu_alloc0_2 = unified_memory.get_or_create_gpu_allocation(key, device0);
    REQUIRE(gpu_alloc0_2.ok());
    REQUIRE(gpu_alloc0.value() == gpu_alloc0_2.value());

    // Different device should create new allocation
    int device1 = 0;
    auto gpu_alloc1 = unified_memory.get_or_create_gpu_allocation(key, device1);
    REQUIRE(gpu_alloc1.ok());
    REQUIRE(gpu_alloc1.value() == gpu_alloc0.value());
  }
}

TEST_CASE("ReplicaMemoryCoordinator mark CPU chunks preemptible", "[unified_memory]") {
  auto mock_dvmp = std::make_shared<MockDistributedVirtualMemoryPool>();
  ReplicaMemoryCoordinator unified_memory(mock_dvmp);

  ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
  size_t size = size_t(1024) * 1024 * 1024; // 1GB = 4 chunks

  REQUIRE(unified_memory.allocate(key, size).ok());

  SECTION("mark 50% preemptible") {
    // Mock DVMP returning virtual memory for the replica
    mock_dvmp->test_region_.bytes = size;

    REQUIRE(unified_memory.mark_cpu_chunks_preemptible(key, 0.5F).ok());

    // Should mark first 2 of 4 chunks
    REQUIRE(mock_dvmp->last_preemptible_chunks_.size() == 2);
    REQUIRE(mock_dvmp->last_preemptible_chunks_[0] == 0);
    REQUIRE(mock_dvmp->last_preemptible_chunks_[1] == 1);
  }

  SECTION("mark all preemptible") {
    mock_dvmp->test_region_.bytes = size;

    REQUIRE(unified_memory.mark_cpu_chunks_preemptible(key, 1.0F).ok());

    // Should mark all 4 chunks
    REQUIRE(mock_dvmp->last_preemptible_chunks_.size() == 4);
  }

  SECTION("mark none preemptible") {
    mock_dvmp->test_region_.bytes = size;

    REQUIRE(unified_memory.mark_cpu_chunks_preemptible(key, 0.0F).ok());

    // Should mark no chunks
    REQUIRE(mock_dvmp->last_preemptible_chunks_.empty());
  }
}

TEST_CASE("ReplicaMemoryCoordinator chunk calculations", "[unified_memory]") {
  auto mock_dvmp = std::make_shared<MockDistributedVirtualMemoryPool>();
  ReplicaMemoryCoordinator unified_memory(mock_dvmp);

  // Test chunk size
  size_t chunk_size = unified_memory.get_chunk_size();
  REQUIRE(chunk_size == 256ULL * 1024 * 1024); // 256MB default

  SECTION("various artifact sizes") {
    struct TestCase {
      size_t artifact_size;
      size_t expected_chunks;
    };

    std::vector<TestCase> test_cases = {
        {size_t(100) * 1024 * 1024, 1}, // 100MB -> 1 chunk
        {size_t(256) * 1024 * 1024, 1}, // 256MB -> 1 chunk
        {size_t(257) * 1024 * 1024, 2}, // 257MB -> 2 chunks
        {size_t(512) * 1024 * 1024, 2}, // 512MB -> 2 chunks
        {size_t(1024) * 1024 * 1024, 4}, // 1GB -> 4 chunks
    };

    for (const auto& tc : test_cases) {
      DYNAMIC_SECTION("size " << tc.artifact_size) {
        ReplicaKey key{absl::StrCat("replica", tc.artifact_size), DeviceKey{DeviceType::GPU, 0, ""}, 0};

        // Reset mock for each test
        mock_dvmp->test_region_.bytes = 0;
        ReplicaMemoryCoordinator local_um(mock_dvmp);

        REQUIRE(local_um.allocate(key, tc.artifact_size).ok());
        auto mappings = local_um.get_chunk_mappings(key);
        REQUIRE(mappings.size() == tc.expected_chunks);
      }
    }
  }
}

TEST_CASE("ReplicaMemoryCoordinator multi-GPU state tracking", "[unified_memory]") {
  auto mock_dvmp = std::make_shared<MockDistributedVirtualMemoryPool>();
  ReplicaMemoryCoordinator unified_memory(mock_dvmp);

  ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
  size_t size = size_t(512) * 1024 * 1024;

  REQUIRE(unified_memory.allocate(key, size).ok());

  // Update states for multiple GPUs
  int gpu0 = 0;
  int gpu1 = 1;

  REQUIRE(unified_memory.update_chunk_states(key, MemoryLocation::GPU, {0}, ChunkState::HOT, gpu0).ok());
  REQUIRE(unified_memory.update_chunk_states(key, MemoryLocation::GPU, {0, 1}, ChunkState::HOT, gpu1).ok());

  // Check states
  auto mappings = unified_memory.get_chunk_mappings(key);
  REQUIRE(mappings[0].gpu_state.at(DeviceKey{DeviceType::GPU, gpu0, ""}) == ChunkState::HOT);
  REQUIRE(mappings[0].gpu_state.at(DeviceKey{DeviceType::GPU, gpu1, ""}) == ChunkState::HOT);
  REQUIRE(!mappings[1].gpu_state.contains(DeviceKey{DeviceType::GPU, gpu0, ""})); // Not set for gpu0
  REQUIRE(mappings[1].gpu_state.at(DeviceKey{DeviceType::GPU, gpu1, ""}) == ChunkState::HOT);
}

} // namespace
} // namespace tensorcast::store