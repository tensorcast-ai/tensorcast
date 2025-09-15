// Copyright (c) 2025, TensorCast Team.

#include "core/store/replica/unified_memory_authority.h"
// NOTE: renamed from *_compat_test to reflect final UMA plan/commit semantics.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <cstdlib>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/common/memory/virtual_address_space.h"
#include "core/store/loading/loading_spec.h"

namespace tensorcast::store::replica {
using ::tensorcast::DeviceType;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::DeviceKey;
using tensorcast::store::loading::ReplicaKey;

namespace {

// Mock implementation of VirtualAddressSpace for testing
class MockVirtualAddressSpace : public common::memory::VirtualAddressSpace {
 public:
  // === Configurable state ===
  VirtualRegion test_region_{};
  absl::Status allocate_status_ = absl::OkStatus();
  absl::Status mark_preemptible_status_ = absl::OkStatus();
  std::vector<ChunkState> chunk_states_;

  // === Tracking ===
  std::vector<uint32_t> last_preemptible_chunks_;

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

  absl::Status mark_preemptible(std::string_view /*artifact_id*/, absl::Span<const uint32_t> idx) override {
    last_preemptible_chunks_.assign(idx.begin(), idx.end());
    return mark_preemptible_status_;
  }

  absl::Span<const ChunkMeta> chunk_telemetry_snapshot(std::string_view /*artifact_id*/) const noexcept override {
    static std::vector<ChunkMeta> meta_cache;
    std::vector<ChunkMeta> local(chunk_states_.size());
    for (size_t i = 0; i < chunk_states_.size(); ++i) {
      local[i].state.store(chunk_states_[i]);
    }
    meta_cache.swap(local);
    return absl::MakeConstSpan(meta_cache);
  }
};

TEST_CASE("UnifiedMemoryAuthority allocation", "[unified_memory]") {
  auto mock_vs = std::make_shared<MockVirtualAddressSpace>();
  UnifiedMemoryAuthority unified_memory(mock_vs);

  SECTION("successful allocation") {
    ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
    size_t size = size_t(1024) * 1024 * 1024; // 1GB

    REQUIRE(unified_memory.allocate(key, size).ok());

    // Verify allocation layout matches size and chunk_size
    auto layout_or = unified_memory.get_layout(key);
    REQUIRE(layout_or.ok());
    auto layout = *layout_or;
    REQUIRE(layout.artifact_bytes == size);
    size_t expected_chunks = (size + layout.artifact_chunk_bytes - 1) / layout.artifact_chunk_bytes;
    REQUIRE(
        expected_chunks ==
        (size + unified_memory.get_artifact_chunk_bytes() - 1) / unified_memory.get_artifact_chunk_bytes());
  }

  SECTION("allocation failure") {
    ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
    size_t size = size_t(1024) * 1024 * 1024;

    mock_vs->allocate_status_ = absl::ResourceExhaustedError("Out of memory");

    auto status = unified_memory.allocate(key, size);
    REQUIRE(!status.ok());
    REQUIRE(status.code() == absl::StatusCode::kResourceExhausted);
  }
}

TEST_CASE("UnifiedMemoryAuthority get missing chunks", "[unified_memory]") {
  auto mock_vs = std::make_shared<MockVirtualAddressSpace>();
  UnifiedMemoryAuthority unified_memory(mock_vs);

  ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
  size_t size = size_t(768) * 1024 * 1024; // 768MB = 3 chunks of 256MB

  REQUIRE(unified_memory.allocate(key, size).ok());

  SECTION("missing chunks for GPU") {
    // Simulate some chunks already on GPU via plan/commit
    DeviceKey device_key{DeviceType::GPU, 0, ""};
    {
      std::vector<uint32_t> chunks{0, 2};
      auto plan_or = unified_memory.plan_load(key, MemoryLocation::GPU, device_key.ordinal, absl::MakeSpan(chunks));
      REQUIRE(plan_or.ok());
      auto cst =
          unified_memory.commit(plan_or->session_id, MemoryLocation::GPU, absl::MakeSpan(chunks), device_key.ordinal);
      REQUIRE(cst.ok());
    }

    // Get missing chunks for GPU
    auto missing = unified_memory.get_missing_chunks(key, MemoryLocation::GPU, device_key.ordinal);
    REQUIRE(missing.size() == 1);
    REQUIRE(missing[0] == 1);
  }

  // CPU missing chunks test removed: plan/commit path only marks HOT; EVICTED is set by policy.
}

TEST_CASE("UnifiedMemoryAuthority lock chunks for transfer", "[unified_memory]") {
  auto mock_vs = std::make_shared<MockVirtualAddressSpace>();
  UnifiedMemoryAuthority unified_memory(mock_vs);

  ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
  size_t size = size_t(512) * 1024 * 1024; // 512MB = 2 chunks

  REQUIRE(unified_memory.allocate(key, size).ok());

  SECTION("plan/commit failure handling (invalid index)") {
    // Commit with invalid chunk index should fail
    auto plan_or = unified_memory.plan_load(key, MemoryLocation::GPU, /*device_id*/ 0, absl::Span<const uint32_t>());
    REQUIRE(plan_or.ok());
    std::vector<uint32_t> bad{99};
    auto status = unified_memory.commit(plan_or->session_id, MemoryLocation::GPU, absl::MakeSpan(bad), /*device_id*/ 0);
    REQUIRE(!status.ok());
    REQUIRE(status.code() == absl::StatusCode::kOutOfRange);
  }
}

TEST_CASE("UnifiedMemoryAuthority update chunk states", "[unified_memory]") {
  auto mock_vs = std::make_shared<MockVirtualAddressSpace>();
  UnifiedMemoryAuthority unified_memory(mock_vs);

  ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
  size_t size = size_t(512) * 1024 * 1024;
  int device_id = 0;

  REQUIRE(unified_memory.allocate(key, size).ok());

  SECTION("update GPU chunk states") {
    // Update chunk states via plan/commit to COPIED_GPU
    std::vector<uint32_t> chunks = {0, 1};
    auto plan_or = unified_memory.plan_load(key, MemoryLocation::GPU, device_id, absl::MakeSpan(chunks));
    REQUIRE(plan_or.ok());
    REQUIRE(unified_memory.commit(plan_or->session_id, MemoryLocation::GPU, absl::MakeSpan(chunks), device_id).ok());

    // Verify GPU states
    auto missing = unified_memory.get_missing_chunks(key, MemoryLocation::GPU, device_id);
    REQUIRE(std::find(missing.begin(), missing.end(), 0) == missing.end());
    REQUIRE(std::find(missing.begin(), missing.end(), 1) == missing.end());
  }

  SECTION("update CPU chunk states") {
    // Mark CPU chunks HOT via plan/commit and verify state becomes HOT
    std::vector<uint32_t> chunks = {1};
    auto plan_or = unified_memory.plan_load(key, MemoryLocation::CPU, std::nullopt, absl::MakeSpan(chunks));
    REQUIRE(plan_or.ok());
    REQUIRE(unified_memory.commit(plan_or->session_id, MemoryLocation::CPU, absl::MakeSpan(chunks), std::nullopt).ok());

    auto st_or = unified_memory.get_cpu_chunk_state(key, 1);
    REQUIRE(st_or.ok());
    REQUIRE(*st_or == ChunkState::HOT);
  }
}

TEST_CASE("UnifiedMemoryAuthority GPU allocation management", "[unified_memory]") {
  auto mock_vs = std::make_shared<MockVirtualAddressSpace>();
  UnifiedMemoryAuthority unified_memory(mock_vs);

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

TEST_CASE("UnifiedMemoryAuthority mark CPU chunks preemptible", "[unified_memory]") {
  auto mock_vs = std::make_shared<MockVirtualAddressSpace>();
  UnifiedMemoryAuthority unified_memory(mock_vs);

  ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
  size_t size = size_t(1024) * 1024 * 1024; // 1GB = 4 chunks

  REQUIRE(unified_memory.allocate(key, size).ok());

  SECTION("mark 50% preemptible") {
    // Mock VS returning virtual memory for the replica
    mock_vs->test_region_.bytes = size;

    REQUIRE(unified_memory.mark_cpu_chunks_preemptible(key, 0.5F).ok());

    // Should mark first 2 of 4 chunks
    REQUIRE(mock_vs->last_preemptible_chunks_.size() == 2);
    REQUIRE(mock_vs->last_preemptible_chunks_[0] == 0);
    REQUIRE(mock_vs->last_preemptible_chunks_[1] == 1);
  }

  SECTION("mark all preemptible") {
    mock_vs->test_region_.bytes = size;

    REQUIRE(unified_memory.mark_cpu_chunks_preemptible(key, 1.0F).ok());

    // Should mark all 4 chunks
    REQUIRE(mock_vs->last_preemptible_chunks_.size() == 4);
  }

  SECTION("mark none preemptible") {
    mock_vs->test_region_.bytes = size;

    REQUIRE(unified_memory.mark_cpu_chunks_preemptible(key, 0.0F).ok());

    // Should mark no chunks
    REQUIRE(mock_vs->last_preemptible_chunks_.empty());
  }
}

TEST_CASE("UnifiedMemoryAuthority chunk calculations", "[unified_memory]") {
  auto mock_vs = std::make_shared<MockVirtualAddressSpace>();
  UnifiedMemoryAuthority unified_memory(mock_vs);

  // Test chunk size
  size_t chunk_size = unified_memory.get_artifact_chunk_bytes();
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
        mock_vs->test_region_.bytes = 0;
        UnifiedMemoryAuthority local_um(mock_vs);

        REQUIRE(local_um.allocate(key, tc.artifact_size).ok());
        auto layout_or = local_um.get_layout(key);
        REQUIRE(layout_or.ok());
        auto layout = *layout_or;
        size_t num_chunks = (layout.artifact_bytes + layout.artifact_chunk_bytes - 1) / layout.artifact_chunk_bytes;
        REQUIRE(num_chunks == tc.expected_chunks);
      }
    }
  }
}

TEST_CASE("UnifiedMemoryAuthority multi-GPU state tracking", "[unified_memory]") {
  auto mock_vs = std::make_shared<MockVirtualAddressSpace>();
  UnifiedMemoryAuthority unified_memory(mock_vs);

  ReplicaKey key{"model1", DeviceKey{DeviceType::GPU, 0, ""}, 0};
  size_t size = size_t(512) * 1024 * 1024;

  REQUIRE(unified_memory.allocate(key, size).ok());

  // Update states for multiple GPUs
  int gpu0 = 0;
  int gpu1 = 1;

  {
    std::vector<uint32_t> c0{0};
    auto p0 = unified_memory.plan_load(key, MemoryLocation::GPU, gpu0, absl::MakeSpan(c0));
    REQUIRE(p0.ok());
    REQUIRE(unified_memory.commit(p0->session_id, MemoryLocation::GPU, absl::MakeSpan(c0), gpu0).ok());
  }
  {
    std::vector<uint32_t> c1{0, 1};
    auto p1 = unified_memory.plan_load(key, MemoryLocation::GPU, gpu1, absl::MakeSpan(c1));
    REQUIRE(p1.ok());
    REQUIRE(unified_memory.commit(p1->session_id, MemoryLocation::GPU, absl::MakeSpan(c1), gpu1).ok());
  }

  // Check states
  auto missing_gpu0 = unified_memory.get_missing_chunks(key, MemoryLocation::GPU, gpu0);
  auto missing_gpu1 = unified_memory.get_missing_chunks(key, MemoryLocation::GPU, gpu1);
  REQUIRE(std::find(missing_gpu0.begin(), missing_gpu0.end(), 0) == missing_gpu0.end());
  REQUIRE(std::find(missing_gpu1.begin(), missing_gpu1.end(), 0) == missing_gpu1.end());
  REQUIRE(std::find(missing_gpu0.begin(), missing_gpu0.end(), 1) != missing_gpu0.end());
  REQUIRE(std::find(missing_gpu1.begin(), missing_gpu1.end(), 1) == missing_gpu1.end());
}

} // namespace
} // namespace tensorcast::store::replica
