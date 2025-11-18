// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/common/view_hash_utils.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/replica/replica.h"
#include "gsl/pointers"

namespace tensorcast::store {
namespace {

constexpr size_t kChunkBytes = 4ULL * 1024 * 1024;

class VectorSource final : public loader::SeekableSource {
 public:
  explicit VectorSource(const std::vector<uint8_t>& data) : data_(data) {}

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto bytes_or = read_at(cursor_, dst, max_bytes);
    if (!bytes_or.ok()) {
      return bytes_or;
    }
    cursor_ += *bytes_or;
    return bytes_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= data_.size() || bytes == 0) {
      return static_cast<size_t>(0);
    }
    const size_t available = static_cast<size_t>(std::min<uint64_t>(bytes, data_.size() - offset));
    std::memcpy(dst, data_.data() + offset, available);
    return available;
  }

 private:
  const std::vector<uint8_t>& data_;
  uint64_t cursor_{0};
};

std::shared_ptr<replica::Replica> MakeCpuReplica(
    const gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>& pool,
    uint64_t size_bytes) {
  loading::InlineBufferSource src{.data = nullptr, .size_bytes = size_bytes};
  if (size_bytes > 0) {
    auto storage = std::shared_ptr<uint8_t[]>(new uint8_t[size_bytes], std::default_delete<uint8_t[]>());
    for (uint64_t i = 0; i < size_bytes; ++i) {
      storage.get()[i] = static_cast<uint8_t>(i & 0xFF);
    }
    src.data = std::shared_ptr<const void>(storage, storage.get());
  }
  replica::ReplicaConfig cfg{
      .source = src,
      .artifact_identifier = "view-hash-test",
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool = pool,
      .artifact_chunk_bytes = kChunkBytes,
      .expected_artifact_size = size_bytes,
      .view_id = std::nullopt,
      .view_plan = std::nullopt,
      .transform_placement = loading::TransformPlacement::kServer};
  auto replica_or = replica::Replica::create(cfg);
  REQUIRE(replica_or.ok());
  return std::shared_ptr<replica::Replica>(std::move(replica_or.value()));
}

} // namespace

TEST_CASE("compute_view_data_hash returns null for empty view", "[view_hash_utils]") {
  auto pool = std::make_shared<common::memory::PinnedBufferPool>(16ULL << 20, 1ULL << 20);
  auto replica = MakeCpuReplica(gsl::not_null{pool}, 1024);
  auto hash = compute_view_data_hash(*replica, common::memory::MemoryLocation::CPU, 0, std::nullopt);
  REQUIRE_FALSE(hash.has_value());
}

TEST_CASE("compute_view_data_hash hashes CPU memory", "[view_hash_utils]") {
  auto pool = std::make_shared<common::memory::PinnedBufferPool>(16ULL << 20, 1ULL << 20);
  auto replica = MakeCpuReplica(gsl::not_null{pool}, 1024);
  replica->ensure_loaded_async(common::memory::MemoryLocation::CPU).wait();
  auto hash = compute_view_data_hash(*replica, common::memory::MemoryLocation::CPU, 1024, std::nullopt);
  REQUIRE(hash.has_value());
  auto hash_again = compute_view_data_hash(*replica, common::memory::MemoryLocation::CPU, 1024, std::nullopt);
  REQUIRE(hash_again.has_value());
  REQUIRE(*hash == *hash_again);
}

TEST_CASE("ViewHashComputer hashes replica views", "[view_hash_utils]") {
  auto pool = std::make_shared<common::memory::PinnedBufferPool>(16ULL << 20, 1ULL << 20);
  auto replica = MakeCpuReplica(gsl::not_null{pool}, 2048);
  replica->ensure_loaded_async(common::memory::MemoryLocation::CPU).wait();

  ViewHashComputer computer;
  auto hash = computer.hash_replica_view(*replica, common::memory::MemoryLocation::CPU, 2048, std::nullopt);
  REQUIRE(hash.has_value());
}

TEST_CASE("ViewHashComputer hashes view from seekable source", "[view_hash_utils]") {
  std::vector<uint8_t> backing(32);
  for (size_t i = 0; i < backing.size(); ++i) {
    backing[i] = static_cast<uint8_t>(i);
  }

  loader::ViewPlan plan;
  plan.is_identity = false;
  plan.view_size_bytes = 8;
  plan.selection.is_contiguous = true;
  plan.selection.num_ranges = 1;
  plan.selection.total_bytes = 8;
  plan.selection.requires_materialization = false;
  plan.selection.ranges.push_back(
      loader::SelectionPlan::Range{
          .kind = loader::SelectionPlan::Range::Kind::kData, .src_offset = 8, .dst_offset = 0, .length = 8});
  plan.transform.requires_materialization = false;

  VectorSource source(backing);
  ViewHashComputer computer;
  auto hash_or = computer.hash_view_from_source(source, plan);
  REQUIRE(hash_or.ok());

  std::vector<uint8_t> expected(backing.begin() + 8, backing.begin() + 16);
  auto expected_or =
      loader::compute_data_multihash_from_cpu_memory(gsl::not_null<const void*>{expected.data()}, expected.size());
  REQUIRE(expected_or.ok());
  CHECK(*hash_or == *expected_or);
}

} // namespace tensorcast::store
