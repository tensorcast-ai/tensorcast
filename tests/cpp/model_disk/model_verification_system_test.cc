// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "tests/cpp/common.h"

#include <filesystem>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/common/model_verification.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/model/model.h"
#include "core/store/model/model_config.h"

namespace fs = std::filesystem;
using namespace stepcast::store;
using namespace stepcast::tests;

TEST_CASE("Model Verification System", "[model][verification]") {
  const std::string model_id = "verification_model";
  const std::string model_subdir = "verification_files";
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
  fs::create_directories(base / model_subdir);

  fs::path path0 = base / model_subdir / p0;
  fs::path path1 = base / model_subdir / p1;
  REQUIRE(create_dummy_file(path0, size0, 'A'));
  REQUIRE(create_dummy_file(path1, size1, 'X'));

  auto data0 = read_file_content(path0);
  auto data1 = read_file_content(path1);

  const size_t pool_total = 1024 * 1024;
  const size_t pool_chunk = 1024;
  auto pool = std::make_shared<PinnedMemoryPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  // Create DVMP
  auto dvmp = std::make_shared<::stepcast::memory::DistributedVirtualMemoryPool>();

  // Use new DiskSource
  DiskSource disk_src;
  disk_src.path = base / model_subdir;

  // Use aggregate initialization for ModelConfig
  // Set max_buffer_bytes to match the available pool size
  ModelConfig cfg{
      .source = disk_src,
      .model_identifier = model_id,
      .device_type = ::stepcast::DeviceType::CPU,
      .local_device_id = 0,
      .pinned_memory_pool = pool,
      .dvmp = dvmp,
      .max_buffer_bytes = pool_total};

  auto mstatus = Model::create(cfg);
  REQUIRE(mstatus.ok());
  auto model = std::move(*mstatus);

  // Load to CPU
  auto load_fut = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
  REQUIRE(load_fut.valid());
  REQUIRE(model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());
  REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::LOADED);

  SECTION("Generate verification info for CPU") {
    auto ver_status = model->generate_verification_info(ModelLocation::PAGEABLE_CPU);
    REQUIRE(ver_status.ok());
    const ModelVerificationInfo info = ver_status.value();
    REQUIRE(info.model_size == total_size);
    REQUIRE(info.full_hash != 0);
    REQUIRE(!info.key_values.empty());
    REQUIRE(!info.segment_hashes.empty());
    REQUIRE(!info.sample_values.empty());
  }

  SECTION("Verify CPU data at different levels") {
    auto ver_status = model->generate_verification_info(ModelLocation::PAGEABLE_CPU);
    REQUIRE(ver_status.ok());
    const ModelVerificationInfo info = ver_status.value();

    REQUIRE(model->verify_key_points(ModelLocation::PAGEABLE_CPU, info).ok());
    REQUIRE(model->verify_model_data(ModelLocation::PAGEABLE_CPU, info, VerificationLevel::SPARSE_SAMPLING).ok());
    REQUIRE(model->verify_model_data(ModelLocation::PAGEABLE_CPU, info, VerificationLevel::SEGMENT_HASHES).ok());
    REQUIRE(model->verify_model_data(ModelLocation::PAGEABLE_CPU, info, VerificationLevel::FULL_HASH).ok());
  }

  SECTION("Verification should fail on corrupted info") {
    auto ver_status = model->generate_verification_info(ModelLocation::PAGEABLE_CPU);
    REQUIRE(ver_status.ok());
    ModelVerificationInfo corrupted = ver_status.value();
    corrupted.full_hash = 0xDEADBEEF;
    REQUIRE(!model->verify_model_data(ModelLocation::PAGEABLE_CPU, corrupted, VerificationLevel::FULL_HASH).ok());
  }

  // Teardown
  model.reset();
  pool.reset();
  fs::remove_all(base);
}