// Copyright (c) 2025, TensorCast Team.

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

// Define this before including catch.hpp
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/time/time.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/memory_location.h"
#include "core/store/components/communication_manager.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"
#include "core/testing/common.h"

namespace fs = std::filesystem;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::common::memory::PinnedBufferPool;
using tensorcast::store::ExportRegistration;
using tensorcast::store::components::CommunicationManager;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::replica::MemoryState;
using tensorcast::store::replica::Replica;
using tensorcast::store::replica::ReplicaConfig;

// Helper function to create a dummy file (copied from model_disk_test.cc)
bool create_dummy_file(const fs::path& path, size_t size) {
  std::ofstream outfile(path, std::ios::binary);
  if (!outfile) {
    return false;
  }
  std::vector<char> buffer(size, 'A'); // Fill with 'A's
  outfile.write(buffer.data(), size);
  return outfile.good();
}

// Helper function to check if CUDA devices are available
bool has_cuda_devices() {
  int device_count = 0;
  absl::Status status = tensorcast::cuda::get_device_count(&device_count);
  if (!status.ok()) {
    return false; // Error checking devices
  }
  return device_count > 0;
}

// Renamed Test Case
TEST_CASE("Replica Communication Memory Registration", "[replica][comm_registration][cpu][gpu]") {
  // --- Setup ---
  LOG(INFO) << "Setting up communication registration test...";

  // ------------------------------------------------------------------
  // Initialize a CommunicationManager for test and obtain its engine for
  // dependency injection to ReplicaConfig. Use ephemeral localhost port.
  // ------------------------------------------------------------------
  auto comm_mgr = std::make_shared<CommunicationManager>();
  absl::Status engine_status = comm_mgr->initialize("127.0.0.1", 16000);
  if (!engine_status.ok()) {
    FAIL("Failed to initialize communication manager: " << engine_status);
  }
  LOG(INFO) << "Test CommunicationManager initialized.";

  const std::string artifact_id_base = "test_comm_reg_artifact";
  const std::string artifact_dir_name = "comm_reg_artifact_files";
  const std::string partition_filename = "tensor.data_0";
  const size_t artifact_size = 1024UL * 512; // 512 KiB replica

  fs::path temp_dir_base = fs::temp_directory_path() / "artifact_tmp";
  fs::create_directories(temp_dir_base); // Create base dir once

  // Create memory pools (shared across sections)
  const size_t pool_total_size = 1024UL * 1024 * 32; // 32 MiB pool (increased size)
  const size_t pool_chunk_size = 64UL * 1024; // 64 KiB chunk size
  std::shared_ptr<PinnedBufferPool> pinned_pool = std::make_shared<PinnedBufferPool>(pool_total_size, pool_chunk_size);
  REQUIRE(pinned_pool != nullptr);
  auto async_runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  REQUIRE(async_runtime != nullptr);
  // --- Test GPU Registration --- (Requires CUDA device)
  SECTION("Load to GPU and Register for Communication") {
    // Skip section if no CUDA devices are available
    if (!has_cuda_devices()) {
      WARN("Skipping GPU Communication registration test: No CUDA devices found.");
      return;
    }

    LOG(INFO) << "Starting GPU Registration Section...";
    const std::string artifact_id = artifact_id_base + "_gpu";
    fs::path temp_dir = temp_dir_base / "gpu";
    fs::path artifact_dir = temp_dir / artifact_dir_name;
    fs::create_directories(artifact_dir);
    fs::path dummy_file_path = artifact_dir / partition_filename;
    REQUIRE(create_dummy_file(dummy_file_path, artifact_size));
    // RFC-0007 metadata for standard partitions
    REQUIRE(::tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

    int device_id = 0;
    int device_count;
    absl::Status status = tensorcast::cuda::get_device_count(&device_count);
    REQUIRE(status.ok());
    REQUIRE(device_count > 0);
    REQUIRE(device_id < device_count);

    DiskSource disk_src;
    disk_src.path = temp_dir / artifact_dir_name;

    ReplicaConfig config{
        .source = disk_src,
        .artifact_identifier = artifact_id,
        .device_type = ::tensorcast::DeviceType::GPU,
        .local_device_id = device_id,
        .pinned_buffer_pool = pinned_pool,
        .async_runtime = async_runtime,
        .expected_artifact_size = artifact_size,
        .p2p_comm_enabled = true};

    LOG(INFO) << "Creating Replica instance for GPU test...";
    absl::StatusOr<std::unique_ptr<Replica>> replica_or = Replica::create(config);
    INFO("Replica creation status: " << replica_or.status());
    REQUIRE(replica_or.ok());
    std::unique_ptr<Replica> replica = std::move(*replica_or);
    REQUIRE(replica != nullptr);
    REQUIRE(replica->artifact_id() == artifact_id);

    REQUIRE(replica->get_memory_state(MemoryLocation::GPU) <= MemoryState::UNALLOCATED);

    // Load to CPU first (as DiskLoader requires it)
    LOG(INFO) << "Loading replica to CPU (prerequisite for GPU load from disk)...";
    auto load_future = replica->ensure_loaded_async(MemoryLocation::CPU);
    REQUIRE(load_future.valid());
    absl::Status load_status = std::move(load_future).get(); // Wait for completion
    INFO("CPU load status: " << load_status);
    REQUIRE(load_status.ok());
    REQUIRE(replica->wait_until_loaded(MemoryLocation::CPU, absl::Seconds(10)).ok());
    REQUIRE(replica->get_memory_state(MemoryLocation::CPU) == MemoryState::LOADED);

    // Now load to GPU
    LOG(INFO) << "Loading replica to GPU...";
    auto gpu_future = replica->ensure_loaded_async(MemoryLocation::GPU);
    REQUIRE(gpu_future.valid());
    load_status = std::move(gpu_future).get(); // Wait for completion
    INFO("GPU load status: " << load_status);
    REQUIRE(load_status.ok());

    // Wait until fully loaded on GPU
    absl::Status wait_status = replica->wait_until_loaded(MemoryLocation::GPU, absl::Seconds(30));
    INFO("GPU load wait status: " << wait_status);
    REQUIRE(wait_status.ok());

    REQUIRE(replica->get_memory_state(MemoryLocation::GPU) == MemoryState::LOADED);
    auto data_ptrs_gpu = replica->get_data_pointer(MemoryLocation::GPU);
    REQUIRE_FALSE(data_ptrs_gpu.empty());
    void* gpu_data_ptr = data_ptrs_gpu[0];
    REQUIRE(gpu_data_ptr != nullptr);
    LOG(INFO) << "Replica loaded to GPU successfully. Pointer: " << gpu_data_ptr;

    // Register GPU memory for communication
    LOG(INFO) << "Registering GPU memory for communication...";
    absl::StatusOr<ExportRegistration> reg_info_status =
        replica->enable_remote_memory_access(MemoryLocation::GPU, comm_mgr->get_engine());
    INFO("Comm registration status (GPU): " << reg_info_status.status());
    REQUIRE(reg_info_status.ok()); // Expect registration to succeed
    const auto& reg_info = *reg_info_status;

    // Verify returned registration info
    REQUIRE(reg_info.artifact_size == artifact_size);
    REQUIRE(reg_info.location == MemoryLocation::GPU);
    REQUIRE(reg_info.device_id == device_id);
    REQUIRE(reg_info.buffer_addresses.size() == 1);
    REQUIRE(reg_info.buffer_sizes.size() == 1);
    REQUIRE(reg_info.remote_memory_keys.size() == 1);
    REQUIRE(reg_info.buffer_sizes[0] == artifact_size);
    REQUIRE(reinterpret_cast<void*>(reg_info.buffer_addresses[0]) == gpu_data_ptr);
    REQUIRE(reg_info.remote_memory_keys[0].find(artifact_id) != std::string::npos);
    REQUIRE(absl::StrContains(reg_info.remote_memory_keys[0], "GPU"));
    REQUIRE(absl::StrContains(reg_info.remote_memory_keys[0], "chunk_0"));

    LOG(INFO) << "GPU communication registration successful.";

    // Release memory
    LOG(INFO) << "Releasing GPU memory...";
    absl::Status release_status = replica->release_memory(MemoryLocation::GPU);
    REQUIRE(release_status.ok());
    REQUIRE(replica->get_memory_state(MemoryLocation::GPU) <= MemoryState::UNALLOCATED);
    REQUIRE(replica->get_data_pointer(MemoryLocation::GPU).empty());

    // Clean up section-specific directory
    std::error_code ec;
    fs::remove_all(temp_dir, ec);
    if (ec) {
      WARN("Failed to remove GPU test directory " << temp_dir << ": " << ec.message());
    }
    LOG(INFO) << "Finished GPU Registration Section.";
  }

  // --- Test CPU Registration --- (Always runs)
  SECTION("Load to CPU and Register for Communication") {
    LOG(INFO) << "Starting CPU Registration Section...";
    const std::string artifact_id = artifact_id_base + "_cpu";
    fs::path temp_dir = temp_dir_base / "cpu";
    fs::path artifact_dir = temp_dir / artifact_dir_name;
    fs::create_directories(artifact_dir);
    fs::path dummy_file_path = artifact_dir / partition_filename;
    REQUIRE(create_dummy_file(dummy_file_path, artifact_size));
    // RFC-0007 metadata for standard partitions
    REQUIRE(::tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

    // Note: No CUDA pool needed for CPU-only test
    int dummy_device_id = 0; // Still need a device ID for config

    DiskSource disk_src;
    disk_src.path = temp_dir / artifact_dir_name;

    ReplicaConfig config{
        .source = disk_src,
        .artifact_identifier = artifact_id,
        .device_type = ::tensorcast::DeviceType::CPU,
        .local_device_id = dummy_device_id,
        .pinned_buffer_pool = pinned_pool,
        .async_runtime = async_runtime,
        .expected_artifact_size = artifact_size,
        .p2p_comm_enabled = true};

    LOG(INFO) << "Creating Replica instance for CPU test...";
    absl::StatusOr<std::unique_ptr<Replica>> replica_or = Replica::create(config);
    INFO("Replica creation status: " << replica_or.status());
    REQUIRE(replica_or.ok());
    std::unique_ptr<Replica> replica = std::move(*replica_or);
    REQUIRE(replica != nullptr);
    REQUIRE(replica->artifact_id() == artifact_id);

    REQUIRE(replica->get_memory_state(MemoryLocation::CPU) <= MemoryState::UNALLOCATED);

    // Load to CPU
    LOG(INFO) << "Loading replica to CPU...";
    auto load_future = replica->ensure_loaded_async(MemoryLocation::CPU);
    REQUIRE(load_future.valid());
    const absl::Status load_status = std::move(load_future).get(); // Wait for completion
    INFO("CPU load status: " << load_status);
    REQUIRE(load_status.ok());

    // Wait until fully loaded on CPU
    absl::Status wait_status = replica->wait_until_loaded(MemoryLocation::CPU, absl::Seconds(10));
    INFO("CPU load wait status: " << wait_status);
    REQUIRE(wait_status.ok());

    REQUIRE(replica->get_memory_state(MemoryLocation::CPU) == MemoryState::LOADED);
    auto data_ptrs_cpu = replica->get_data_pointer(MemoryLocation::CPU);
    REQUIRE_FALSE(data_ptrs_cpu.empty());
    void* cpu_data_ptr = data_ptrs_cpu[0];
    REQUIRE(cpu_data_ptr != nullptr);
    LOG(INFO) << "Replica loaded to CPU successfully. Pointer (first chunk): " << cpu_data_ptr;

    // Register CPU memory for communication
    LOG(INFO) << "Registering CPU memory for communication...";
    absl::StatusOr<ExportRegistration> reg_info_status =
        replica->enable_remote_memory_access(MemoryLocation::CPU, comm_mgr->get_engine());
    INFO("Comm registration status (CPU): " << reg_info_status.status());
    REQUIRE(reg_info_status.ok()); // Expect registration to succeed
    const auto& reg_info = *reg_info_status;

    // Verify returned registration info
    REQUIRE(reg_info.artifact_size == artifact_size);
    REQUIRE(reg_info.location == MemoryLocation::CPU);
    REQUIRE(reg_info.device_id == -1); // CPU device ID is -1
    REQUIRE_FALSE(reg_info.buffer_addresses.empty());
    REQUIRE(reg_info.buffer_addresses.size() == reg_info.buffer_sizes.size());
    REQUIRE(reg_info.buffer_addresses.size() == reg_info.remote_memory_keys.size());

    // Check total size and individual keys
    size_t total_registered_size = 0;
    for (size_t i = 0; i < reg_info.buffer_addresses.size(); ++i) {
      total_registered_size += reg_info.buffer_sizes[i];
      REQUIRE(reg_info.remote_memory_keys[i].find(artifact_id) != std::string::npos);
      REQUIRE(absl::StrContains(reg_info.remote_memory_keys[i], "CPU"));
      REQUIRE(
          reg_info.remote_memory_keys[i].find(absl::StrFormat("chunk_%d", static_cast<int>(i))) != std::string::npos);
    }
    REQUIRE(total_registered_size == artifact_size);
    REQUIRE(reinterpret_cast<void*>(reg_info.buffer_addresses[0]) == cpu_data_ptr); // First chunk should match

    LOG(INFO) << "CPU communication registration successful.";

    // Release memory
    LOG(INFO) << "Releasing CPU memory...";
    absl::Status release_status = replica->release_memory(MemoryLocation::CPU);
    REQUIRE(release_status.ok());
    REQUIRE(replica->get_memory_state(MemoryLocation::CPU) <= MemoryState::UNALLOCATED);
    REQUIRE(replica->get_data_pointer(MemoryLocation::CPU).empty());

    // Clean up section-specific directory
    std::error_code ec;
    fs::remove_all(temp_dir, ec);
    if (ec) {
      WARN("Failed to remove CPU test directory " << temp_dir << ": " << ec.message());
    }
    LOG(INFO) << "Finished CPU Registration Section.";
  }

  // --- Teardown --- (Clean up base temporary directory)
  LOG(INFO) << "Cleaning up base temporary directory: " << temp_dir_base;
  std::error_code ec;
  fs::remove_all(temp_dir_base, ec);
  if (ec) {
    WARN("Failed to remove base temporary directory " << temp_dir_base << ": " << ec.message());
  }
}
