// Copyright (c) 2025, TensorCast Team.

#include "core/local/chunk/data_chunk.h"
#include "core/local/meta/local_manager.h"
#include "core/local/meta/replica.h"
#include "core/store/device_types.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace tensorcast::local::meta;
using namespace tensorcast::local::data;

// Macro to simplify parameterized tests with file size options
#define GENERATE_FILE_SIZE(file_size_options)    \
  const size_t file_size = GENERATE_COPY(values( \
      {(file_size_options)[0],                   \
       (file_size_options)[1],                   \
       (file_size_options)[2],                   \
       (file_size_options)[3],                   \
       (file_size_options)[4],                   \
       (file_size_options)[5],                   \
       (file_size_options)[6]}))

namespace {

tensorcast::store::DeviceKey MakeCpuDeviceKey() {
  tensorcast::store::DeviceKey key;
  key.type = tensorcast::DeviceType::CPU;
  key.ordinal = 0;
  key.uuid = "cpu0";
  return key;
}

size_t GetPageSize() {
  const int64_t page_size = ::sysconf(_SC_PAGESIZE);
  REQUIRE(page_size > 0);
  return static_cast<size_t>(page_size);
}

struct TempFileFixture {
  explicit TempFileFixture(size_t size_bytes) : data_(size_bytes) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);
    std::generate(data_.begin(), data_.end(), [&]() { return static_cast<std::uint8_t>(dist(gen)); });

    auto temp_dir = std::filesystem::temp_directory_path();
    path_ = temp_dir / ("tensorcast-localmanager-test-" + std::to_string(rd()) + ".bin");

    std::ofstream out(path_, std::ios::binary | std::ios::trunc);
    REQUIRE(out.is_open());
    out.write(reinterpret_cast<const char*>(data_.data()), static_cast<std::streamsize>(data_.size()));
    REQUIRE(out.good());
  }

  ~TempFileFixture() {
    if (path_.empty()) {
      return;
    }
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  const std::filesystem::path& file_path() const {
    return path_;
  }

  const std::vector<std::uint8_t>& data() const {
    return data_;
  }

 private:
  std::filesystem::path path_;
  std::vector<std::uint8_t> data_;
};

std::vector<size_t> GetFileSizeOptions(size_t page_size) {
  const size_t chunk_size = LocalManager::kLocalConfig.chunk_size;
  return {10, page_size, page_size * 4, chunk_size, chunk_size + 10, chunk_size + page_size, chunk_size * 100};
}

} // namespace

void InitializeDeviceList() {
  tensorcast::store::DeviceKey cpu_key;
  cpu_key.type = tensorcast::DeviceType::CPU;
  cpu_key.ordinal = 0;
  cpu_key.uuid = "cpu0";

  LocalManager::manual_set_devices({cpu_key});
}

TEST_CASE("LocalManager get_or_create_artifact creates artifact successfully", "[local_manager]") {
  InitializeDeviceList();
  auto artifact_or = LocalManager::get_or_create_artifact("test_artifact_1");
  REQUIRE(artifact_or.ok());
  auto* artifact = artifact_or.value();
  REQUIRE(artifact != nullptr);
  REQUIRE(artifact->get_artifact_id() == "test_artifact_1");
}

TEST_CASE("LocalManager get_or_create_artifact returns existing artifact for duplicate id", "[local_manager]") {
  InitializeDeviceList();
  auto artifact1_or = LocalManager::get_or_create_artifact("duplicate_id");
  REQUIRE(artifact1_or.ok());
  auto* artifact1 = artifact1_or.value();

  auto artifact2_or = LocalManager::get_or_create_artifact("duplicate_id");
  REQUIRE(artifact2_or.ok());
  auto* artifact2 = artifact2_or.value();

  // Should return the same artifact pointer
  REQUIRE(artifact1 == artifact2);
  REQUIRE(artifact2->get_artifact_id() == "duplicate_id");
}

TEST_CASE("LocalManager create_view creates view successfully", "[local_manager]") {
  InitializeDeviceList();
  auto artifact_or = LocalManager::get_or_create_artifact("test_artifact_2");
  REQUIRE(artifact_or.ok());
  auto* artifact = artifact_or.value();

  const size_t view_size = 10 * 1024 * 1024; // 10MB
  auto view_or = LocalManager::create_view(artifact, "test_view_1", View::ViewType::Vanilla, view_size, false);
  REQUIRE(view_or.ok());
  auto* view = view_or.value();
  REQUIRE(view != nullptr);
  REQUIRE(view->get_view_id() == "test_view_1");
}

TEST_CASE("LocalManager create_view returns error for duplicate view id", "[local_manager]") {
  InitializeDeviceList();
  auto artifact_or = LocalManager::get_or_create_artifact("test_artifact_3");
  REQUIRE(artifact_or.ok());
  auto* artifact = artifact_or.value();

  const size_t view_size = 10 * 1024 * 1024;
  auto view1_or = LocalManager::create_view(artifact, "duplicate_view", View::ViewType::Vanilla, view_size, false);
  REQUIRE(view1_or.ok());

  auto view2_or = LocalManager::create_view(artifact, "duplicate_view", View::ViewType::Vanilla, view_size, false);
  REQUIRE_FALSE(view2_or.ok());
  REQUIRE(view2_or.status().code() == absl::StatusCode::kAlreadyExists);
}

TEST_CASE("LocalManager create_view with chunks creates view with chunks", "[local_manager]") {
  InitializeDeviceList();
  auto artifact_or = LocalManager::get_or_create_artifact("test_artifact_4");
  REQUIRE(artifact_or.ok());
  auto* artifact = artifact_or.value();

  const size_t view_size = LocalManager::kLocalConfig.chunk_size * 3; // 3 chunks
  auto view_or = LocalManager::create_view(artifact, "test_view_chunks", View::ViewType::Vanilla, view_size, true);
  REQUIRE(view_or.ok());
  auto* view = view_or.value();
  REQUIRE(view != nullptr);
  REQUIRE(view->get_view_id() == "test_view_chunks");

  // Verify chunks are created at correct offsets
  const size_t chunk_size = LocalManager::kLocalConfig.chunk_size;
  REQUIRE(view->get_chunk_at(0) != nullptr);
  REQUIRE(view->get_chunk_at(chunk_size) != nullptr);
  REQUIRE(view->get_chunk_at(chunk_size * 2) != nullptr);
  REQUIRE(view->get_chunk_at(chunk_size * 3) == nullptr); // Beyond view size
}

TEST_CASE("LocalManager bind_replica_source binds CPU memory source", "[local_manager]") {
  InitializeDeviceList();
  auto artifact_or = LocalManager::get_or_create_artifact("test_artifact_5");
  REQUIRE(artifact_or.ok());
  auto* artifact = artifact_or.value();

  const size_t view_size = LocalManager::kLocalConfig.chunk_size * 2;
  auto view_or = LocalManager::create_view(artifact, "test_view_bind_cpu", View::ViewType::Vanilla, view_size, true);
  REQUIRE(view_or.ok());
  auto* view = view_or.value();

  auto device_key = MakeCpuDeviceKey();
  Replica replica("replica_1", view, device_key);

  // Allocate CPU memory aligned to page size
  const size_t page_size = GetPageSize();
  const size_t aligned_size = ((view_size + page_size - 1) / page_size) * page_size;
  void* cpu_mem = ::mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  REQUIRE(cpu_mem != MAP_FAILED);

  // Fill with test data
  std::vector<std::uint8_t> test_data(view_size);
  std::iota(test_data.begin(), test_data.end(), 0);
  std::memcpy(cpu_mem, test_data.data(), view_size);

  // Bind replica source
  auto status = LocalManager::bind_replica_source(replica, cpu_mem);
  REQUIRE(status.ok());

  // Verify loaders are registered
  size_t chunk_count = 0;
  for (auto it = replica.begin(); it != replica.end(); ++it) {
    auto* data_chunk = *it;
    REQUIRE(data_chunk != nullptr);
    chunk_count++;
  }
  REQUIRE(chunk_count == 2); // Should have 2 chunks

  // Cleanup
  ::munmap(cpu_mem, aligned_size);
}

TEST_CASE("LocalManager bind_replica_source binds file source", "[local_manager]") {
  InitializeDeviceList();
  const size_t page_size = GetPageSize();
  const auto file_size_options = GetFileSizeOptions(page_size);

  GENERATE_FILE_SIZE(file_size_options);

  TempFileFixture fixture(file_size);

  auto artifact_or = LocalManager::get_or_create_artifact("test_artifact_6_" + std::to_string(file_size));
  REQUIRE(artifact_or.ok());
  auto* artifact = artifact_or.value();

  const size_t view_size = file_size;
  auto view_or = LocalManager::create_view(
      artifact, "test_view_bind_file_" + std::to_string(file_size), View::ViewType::Vanilla, view_size, true);
  REQUIRE(view_or.ok());
  auto* view = view_or.value();

  auto device_key = MakeCpuDeviceKey();
  Replica replica("replica_2_" + std::to_string(file_size), view, device_key);

  // Bind replica source to file
  auto status = LocalManager::bind_replica_source(replica, fixture.file_path().string());
  REQUIRE(status.ok());

  // Verify loaders are registered
  size_t chunk_count = 0;
  for (auto it = replica.begin(); it != replica.end(); ++it) {
    auto* data_chunk = *it;
    REQUIRE(data_chunk != nullptr);
    chunk_count++;
  }
  REQUIRE(chunk_count > 0);
}

TEST_CASE("LocalManager load_replica loads replica successfully", "[local_manager]") {
  InitializeDeviceList();
  const size_t page_size = GetPageSize();
  const auto file_size_options = GetFileSizeOptions(page_size);

  GENERATE_FILE_SIZE(file_size_options);

  TempFileFixture fixture(file_size);

  auto artifact_or = LocalManager::get_or_create_artifact("test_artifact_7_" + std::to_string(file_size));
  REQUIRE(artifact_or.ok());
  auto* artifact = artifact_or.value();

  const size_t view_size = file_size;
  auto view_or = LocalManager::create_view(
      artifact, "test_view_load_" + std::to_string(file_size), View::ViewType::Vanilla, view_size, true);
  REQUIRE(view_or.ok());
  auto* view = view_or.value();

  auto device_key = MakeCpuDeviceKey();
  Replica replica("replica_3_" + std::to_string(file_size), view, device_key);

  // Bind file source
  auto bind_status = LocalManager::bind_replica_source(replica, fixture.file_path().string());
  REQUIRE(bind_status.ok());

  // Load replica
  auto handler_or = LocalManager::load_replica(replica);
  if (!handler_or.ok()) {
    std::cerr << "Failed to load replica: " << handler_or.status().ToString() << "\n";
    REQUIRE(false);
  }
  REQUIRE(handler_or.ok());
  auto handler = std::move(handler_or.value());
  REQUIRE(handler != nullptr);

  // Verify replica is loaded
  auto loaded_replica = handler->get_replica();
  REQUIRE(loaded_replica.get_view() == view);
  REQUIRE(loaded_replica.get_device_key().uuid == device_key.uuid);

  // Verify data chunks are loaded
  for (auto it = loaded_replica.begin(); it != loaded_replica.end(); ++it) {
    auto* data_chunk = *it;
    REQUIRE(data_chunk != nullptr);
    REQUIRE(data_chunk->is_loaded());
  }

  // Cleanup
  handler->release();
}

TEST_CASE("LocalManager load_replica verifies data correctness", "[local_manager]") {
  InitializeDeviceList();
  const size_t page_size = GetPageSize();
  const auto file_size_options = GetFileSizeOptions(page_size);

  GENERATE_FILE_SIZE(file_size_options);

  TempFileFixture fixture(file_size);

  auto artifact_or = LocalManager::get_or_create_artifact("test_artifact_8_" + std::to_string(file_size));
  REQUIRE(artifact_or.ok());
  auto* artifact = artifact_or.value();

  const size_t view_size = file_size;

  auto view_or = LocalManager::create_view(
      artifact, "test_view_verify_" + std::to_string(file_size), View::ViewType::Vanilla, view_size, true);
  REQUIRE(view_or.ok());
  auto* view = view_or.value();

  auto device_key = MakeCpuDeviceKey();
  Replica replica("replica_4_" + std::to_string(file_size), view, device_key);

  // Bind file source
  auto bind_status = LocalManager::bind_replica_source(replica, fixture.file_path().string());
  REQUIRE(bind_status.ok());

  // Load replica
  auto handler_or = LocalManager::load_replica(replica);
  REQUIRE(handler_or.ok());
  auto handler = std::move(handler_or.value());

  // Verify data matches file content
  auto loaded_replica = handler->get_replica();
  for (auto it = loaded_replica.begin(); it != loaded_replica.end(); ++it) {
    auto* data_chunk = *it;
    REQUIRE(data_chunk != nullptr);
    REQUIRE(data_chunk->is_loaded());

    void* base_addr = data_chunk->get_base_addr();
    size_t chunk_size = data_chunk->get_size();
    off_t file_offset = it.get_offset();

    REQUIRE(base_addr != nullptr);

    // For the last chunk, we may only have read partial data
    // Calculate the actual size to compare
    size_t actual_size = chunk_size;
    if (static_cast<size_t>(file_offset) + chunk_size > fixture.data().size()) {
      actual_size = fixture.data().size() - static_cast<size_t>(file_offset);
    }

    REQUIRE(actual_size > 0);
    REQUIRE(std::memcmp(base_addr, fixture.data().data() + file_offset, actual_size) == 0);
  }

  handler->release();
}

TEST_CASE("LocalManager full workflow: artifact -> view -> replica -> load", "[local_manager]") {
  InitializeDeviceList();
  const size_t page_size = GetPageSize();
  const auto file_size_options = GetFileSizeOptions(page_size);

  GENERATE_FILE_SIZE(file_size_options);

  TempFileFixture fixture(file_size);

  // Step 1: Create artifact
  auto artifact_or = LocalManager::get_or_create_artifact("workflow_artifact_" + std::to_string(file_size));
  REQUIRE(artifact_or.ok());
  auto* artifact = artifact_or.value();

  // Step 2: Create view with chunks
  const size_t view_size = file_size;
  auto view_or = LocalManager::create_view(
      artifact, "workflow_view_" + std::to_string(file_size), View::ViewType::Vanilla, view_size, true);
  REQUIRE(view_or.ok());
  auto* view = view_or.value();

  // Step 3: Create replica
  auto device_key = MakeCpuDeviceKey();
  Replica replica("workflow_replica_" + std::to_string(file_size), view, device_key);
  REQUIRE_FALSE(replica.empty());

  // Step 4: Bind file source
  auto bind_status = LocalManager::bind_replica_source(replica, fixture.file_path().string());
  REQUIRE(bind_status.ok());

  // Step 5: Load replica
  auto handler_or = LocalManager::load_replica(replica);
  REQUIRE(handler_or.ok());
  auto handler = std::move(handler_or.value());

  // Step 6: Verify loaded data
  auto loaded_replica = handler->get_replica();
  size_t chunk_count = 0;
  for (auto it = loaded_replica.begin(); it != loaded_replica.end(); ++it) {
    auto* data_chunk = *it;
    REQUIRE(data_chunk != nullptr);
    REQUIRE(data_chunk->is_loaded());
    chunk_count++;
  }
  REQUIRE(chunk_count > 0);

  handler->release();
}
