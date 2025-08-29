// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/testing/common.h"

#include <filesystem>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/common/artifact_verification.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"

namespace fs = std::filesystem;
using namespace tensorcast::store;
using namespace tensorcast::tests;

TEST_CASE("Replica Verification System", "[replica][verification]") {
  const std::string artifact_id = "verification_artifact";
  const std::string artifact_dir = "verification_files";
  const std::string p0 = "tensor.data_0";
  const std::string p1 = "tensor.data_1";
  // Use page-aligned sizes to trigger DVMP mmap path
  const size_t page_size = 4096; // Common page size
  const size_t size0 = page_size * 3; // 12288 bytes
  const size_t size1 = page_size * 2; // 8192 bytes
  const size_t total_size = size0 + size1;

  fs::path base = fs::temp_directory_path() / "verification_test";
  if (fs::exists(base)) {
    fs::remove_all(base);
  }
  fs::create_directories(base / artifact_dir);

  fs::path path0 = base / artifact_dir / p0;
  fs::path path1 = base / artifact_dir / p1;
  REQUIRE(create_dummy_file(path0, size0, 'A'));
  REQUIRE(create_dummy_file(path1, size1, 'X'));

  // RFC-0007 metadata for standard partitions
  REQUIRE(write_rfc0007_descriptor_for_standard_artifact_dir(base / artifact_dir).ok());

  auto data0 = read_file_content(path0);
  auto data1 = read_file_content(path1);

  const size_t pool_total = 1024 * 1024;
  const size_t pool_chunk = 1024;
  auto pool = std::make_shared<PinnedMemoryPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  // Create DVMP
  auto dvmp = std::make_shared<::tensorcast::memory::DistributedVirtualMemoryPool>();
  // Use new DiskSource
  DiskSource disk_src;
  disk_src.path = base / artifact_dir;

  // Use aggregate initialization for ReplicaConfig
  // Set max_buffer_bytes to match the available pool size
  ReplicaConfig cfg{
      .source = disk_src,
      .artifact_identifier = artifact_id,
      .device_type = ::tensorcast::DeviceType::CPU,
      .local_device_id = 0,
      .pinned_memory_pool = pool,
      .dvmp = dvmp,
      .max_buffer_bytes = pool_total};

  auto mstatus = Replica::create(cfg);
  REQUIRE(mstatus.ok());
  auto replica = std::move(*mstatus);

  // Load to CPU
  auto load_fut = replica->ensure_loaded_async(MemoryLocation::PAGEABLE_CPU);
  REQUIRE(load_fut.valid());
  REQUIRE(replica->wait_until_loaded(MemoryLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());
  REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) == MemoryState::LOADED);

  SECTION("Generate verification info for CPU") {
    auto ver_status = replica->generate_verification_info(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(ver_status.ok());
    const ArtifactVerificationInfo info = ver_status.value();
    REQUIRE(info.artifact_size == total_size);
    REQUIRE(info.full_hash != 0);
    REQUIRE(!info.key_values.empty());
    REQUIRE(!info.segment_hashes.empty());
    REQUIRE(!info.sample_values.empty());
  }

  SECTION("Verify CPU data at different levels") {
    auto ver_status = replica->generate_verification_info(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(ver_status.ok());
    const ArtifactVerificationInfo info = ver_status.value();

    REQUIRE(replica->verify_key_points(MemoryLocation::PAGEABLE_CPU, info).ok());
    REQUIRE(replica->verify_artifact_data(MemoryLocation::PAGEABLE_CPU, info, VerificationLevel::SPARSE_SAMPLING).ok());
    REQUIRE(replica->verify_artifact_data(MemoryLocation::PAGEABLE_CPU, info, VerificationLevel::SEGMENT_HASHES).ok());
    REQUIRE(replica->verify_artifact_data(MemoryLocation::PAGEABLE_CPU, info, VerificationLevel::FULL_HASH).ok());
  }

  SECTION("Verification should fail on corrupted info") {
    auto ver_status = replica->generate_verification_info(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(ver_status.ok());
    ArtifactVerificationInfo corrupted = ver_status.value();
    corrupted.full_hash = 0xDEADBEEF;
    REQUIRE(!replica->verify_artifact_data(MemoryLocation::PAGEABLE_CPU, corrupted, VerificationLevel::FULL_HASH).ok());
  }

  // Teardown
  replica.reset();
  pool.reset();
  fs::remove_all(base);
}