// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

// Define this before including catch.hpp
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

// #include "absl/log/check.h" // Avoid macro conflict with Catch2
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/time/time.h"
#include "core/common/cuda_api.h"
#include "core/store/components/communication_manager.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/model/model.h"
#include "core/store/model/model_config.h"
#include "core/store/model/model_location.h"

namespace fs = std::filesystem;
using namespace stepcast::store;

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
  absl::Status status = stepcast::cuda::get_device_count(&device_count);
  if (!status.ok()) {
    return false; // Error checking devices
  }
  return device_count > 0;
}

// Renamed Test Case
TEST_CASE("Model Communication Memory Registration", "[model][comm_registration][cpu][gpu]") {
  // --- Setup ---
  LOG(INFO) << "Setting up communication registration test...";

  // ------------------------------------------------------------------
  // Initialize a CommunicationManager for test and obtain its engine for
  // dependency injection to ModelConfig. Use ephemeral localhost port.
  // ------------------------------------------------------------------
  auto comm_mgr = std::make_shared<stepcast::store::CommunicationManager>();
  absl::Status engine_status = comm_mgr->initialize("127.0.0.1", 16000);
  if (!engine_status.ok()) {
    FAIL("Failed to initialize communication manager: " << engine_status);
  }
  LOG(INFO) << "Test CommunicationManager initialized.";

  const std::string model_id_base = "test_comm_reg_model";
  const std::string model_subdir_base = "comm_reg_model_files";
  const std::string partition_filename = "tensor.data_0";
  const size_t model_size = 1024UL * 512; // 512 KiB model

  fs::path temp_dir_base = fs::temp_directory_path() / "model_comm_reg_test";
  fs::create_directories(temp_dir_base); // Create base dir once

  // Create memory pools (shared across sections)
  const size_t pool_total_size = 1024UL * 1024 * 32; // 32 MiB pool (increased size)
  const size_t pool_chunk_size = 64UL * 1024; // 64 KiB chunk size
  std::shared_ptr<PinnedMemoryPool> pinned_pool = std::make_shared<PinnedMemoryPool>(pool_total_size, pool_chunk_size);
  REQUIRE(pinned_pool != nullptr);

  // --- Test GPU Registration --- (Requires CUDA device)
  SECTION("Load to GPU and Register for Communication") {
    // Skip section if no CUDA devices are available
    if (!has_cuda_devices()) {
      WARN("Skipping GPU Communication registration test: No CUDA devices found.");
      return;
    }

    LOG(INFO) << "Starting GPU Registration Section...";
    const std::string model_id = model_id_base + "_gpu";
    fs::path temp_dir = temp_dir_base / "gpu";
    fs::path model_data_path = temp_dir / model_subdir_base;
    fs::create_directories(model_data_path);
    fs::path dummy_file_path = model_data_path / partition_filename;
    REQUIRE(create_dummy_file(dummy_file_path, model_size));

    int device_id = 0;
    int device_count;
    absl::Status status = stepcast::cuda::get_device_count(&device_count);
    REQUIRE(status.ok());
    REQUIRE(device_count > 0);
    REQUIRE(device_id < device_count);

    ModelConfig config;
    config.model_identifier = model_id;

    // Use new DiskSource
    DiskSource disk_src;
    disk_src.path = temp_dir / model_subdir_base;
    config.source = disk_src;

    config.pinned_memory_pool = pinned_pool;
    config.local_device_id = device_id;
    config.p2p_comm_enabled = true; // Communicator engine will be injected where required

    LOG(INFO) << "Creating Model instance for GPU test...";
    absl::StatusOr<std::unique_ptr<Model>> model_status = Model::create(config);
    INFO("Model creation status: " << model_status.status());
    REQUIRE(model_status.ok());
    std::unique_ptr<Model> model = std::move(*model_status);
    REQUIRE(model != nullptr);
    REQUIRE(model->model_id() == model_id);

    REQUIRE(model->get_memory_state(ModelLocation::GPU) <= MemoryState::UNALLOCATED);

    // Load to CPU first (as DiskLoader requires it)
    LOG(INFO) << "Loading model to CPU (prerequisite for GPU load from disk)...";
    std::shared_future<absl::Status> load_future = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
    REQUIRE(load_future.valid());
    absl::Status load_status = load_future.get(); // Wait for completion
    INFO("CPU load status: " << load_status);
    REQUIRE(load_status.ok());
    REQUIRE(model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(10)).ok());
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::LOADED);

    // Now load to GPU
    LOG(INFO) << "Loading model to GPU...";
    load_future = model->ensure_loaded_async(ModelLocation::GPU);
    REQUIRE(load_future.valid());
    load_status = load_future.get(); // Wait for completion
    INFO("GPU load status: " << load_status);
    REQUIRE(load_status.ok());

    // Wait until fully loaded on GPU
    absl::Status wait_status = model->wait_until_loaded(ModelLocation::GPU, absl::Seconds(30));
    INFO("GPU load wait status: " << wait_status);
    REQUIRE(wait_status.ok());

    REQUIRE(model->get_memory_state(ModelLocation::GPU) == MemoryState::LOADED);
    auto data_ptrs_gpu = model->get_data_pointer(ModelLocation::GPU);
    REQUIRE_FALSE(data_ptrs_gpu.empty());
    void* gpu_data_ptr = data_ptrs_gpu[0];
    REQUIRE(gpu_data_ptr != nullptr);
    LOG(INFO) << "Model loaded to GPU successfully. Pointer: " << gpu_data_ptr;

    // Register GPU memory for communication
    LOG(INFO) << "Registering GPU memory for communication...";
    absl::StatusOr<CommRegistrationInfo> reg_info_status =
        model->enable_remote_memory_access(ModelLocation::GPU, *comm_mgr->get_shared_engine());
    INFO("Comm registration status (GPU): " << reg_info_status.status());
    REQUIRE(reg_info_status.ok()); // Expect registration to succeed
    const auto& reg_info = *reg_info_status;

    // Verify returned registration info
    REQUIRE(reg_info.model_size == model_size);
    REQUIRE(reg_info.location == ModelLocation::GPU);
    REQUIRE(reg_info.device_id == device_id);
    REQUIRE(reg_info.buffer_addresses.size() == 1);
    REQUIRE(reg_info.buffer_sizes.size() == 1);
    REQUIRE(reg_info.remote_memory_keys.size() == 1);
    REQUIRE(reg_info.buffer_sizes[0] == model_size);
    REQUIRE(reinterpret_cast<void*>(reg_info.buffer_addresses[0]) == gpu_data_ptr);
    REQUIRE(reg_info.remote_memory_keys[0].find(model_id) != std::string::npos);
    REQUIRE(absl::StrContains(reg_info.remote_memory_keys[0], "GPU"));
    REQUIRE(absl::StrContains(reg_info.remote_memory_keys[0], "chunk0"));

    LOG(INFO) << "GPU communication registration successful.";

    // Release memory
    LOG(INFO) << "Releasing GPU memory...";
    absl::Status release_status = model->release_memory(ModelLocation::GPU);
    REQUIRE(release_status.ok());
    REQUIRE(model->get_memory_state(ModelLocation::GPU) <= MemoryState::UNALLOCATED);
    REQUIRE(model->get_data_pointer(ModelLocation::GPU).empty());

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
    const std::string model_id = model_id_base + "_cpu";
    fs::path temp_dir = temp_dir_base / "cpu";
    fs::path model_data_path = temp_dir / model_subdir_base;
    fs::create_directories(model_data_path);
    fs::path dummy_file_path = model_data_path / partition_filename;
    REQUIRE(create_dummy_file(dummy_file_path, model_size));

    // Note: No CUDA pool needed for CPU-only test
    int dummy_device_id = 0; // Still need a device ID for config

    ModelConfig config;
    config.model_identifier = model_id;

    // Use new DiskSource
    DiskSource disk_src;
    disk_src.path = temp_dir / model_subdir_base;
    config.source = disk_src;

    config.pinned_memory_pool = pinned_pool;
    config.local_device_id = dummy_device_id;
    config.p2p_comm_enabled = true; // Communicator engine will be injected where required

    LOG(INFO) << "Creating Model instance for CPU test...";
    absl::StatusOr<std::unique_ptr<Model>> model_status = Model::create(config);
    INFO("Model creation status: " << model_status.status());
    REQUIRE(model_status.ok());
    std::unique_ptr<Model> model = std::move(*model_status);
    REQUIRE(model != nullptr);
    REQUIRE(model->model_id() == model_id);

    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) <= MemoryState::UNALLOCATED);

    // Load to CPU
    LOG(INFO) << "Loading model to CPU...";
    std::shared_future<absl::Status> load_future = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
    REQUIRE(load_future.valid());
    const absl::Status& load_status = load_future.get(); // Wait for completion
    INFO("CPU load status: " << load_status);
    REQUIRE(load_status.ok());

    // Wait until fully loaded on CPU
    absl::Status wait_status = model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(10));
    INFO("CPU load wait status: " << wait_status);
    REQUIRE(wait_status.ok());

    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::LOADED);
    auto data_ptrs_cpu = model->get_data_pointer(ModelLocation::PAGEABLE_CPU);
    REQUIRE_FALSE(data_ptrs_cpu.empty());
    void* cpu_data_ptr = data_ptrs_cpu[0];
    REQUIRE(cpu_data_ptr != nullptr);
    LOG(INFO) << "Model loaded to CPU successfully. Pointer (first chunk): " << cpu_data_ptr;

    // Register CPU memory for communication
    LOG(INFO) << "Registering CPU memory for communication...";
    absl::StatusOr<CommRegistrationInfo> reg_info_status =
        model->enable_remote_memory_access(ModelLocation::PAGEABLE_CPU, *comm_mgr->get_shared_engine());
    INFO("Comm registration status (CPU): " << reg_info_status.status());
    REQUIRE(reg_info_status.ok()); // Expect registration to succeed
    const auto& reg_info = *reg_info_status;

    // Verify returned registration info
    REQUIRE(reg_info.model_size == model_size);
    REQUIRE(reg_info.location == ModelLocation::PAGEABLE_CPU);
    REQUIRE(reg_info.device_id == 1); // Explicitly 1 for CPU
    REQUIRE_FALSE(reg_info.buffer_addresses.empty());
    REQUIRE(reg_info.buffer_addresses.size() == reg_info.buffer_sizes.size());
    REQUIRE(reg_info.buffer_addresses.size() == reg_info.remote_memory_keys.size());

    // Check total size and individual keys
    size_t total_registered_size = 0;
    for (size_t i = 0; i < reg_info.buffer_addresses.size(); ++i) {
      total_registered_size += reg_info.buffer_sizes[i];
      REQUIRE(reg_info.remote_memory_keys[i].find(model_id) != std::string::npos);
      REQUIRE(absl::StrContains(reg_info.remote_memory_keys[i], "CPU"));
      REQUIRE(reg_info.remote_memory_keys[i].find(absl::StrFormat("chunk_%d", i)) != std::string::npos);
    }
    REQUIRE(total_registered_size == model_size);
    REQUIRE(reinterpret_cast<void*>(reg_info.buffer_addresses[0]) == cpu_data_ptr); // First chunk should match

    LOG(INFO) << "CPU communication registration successful.";

    // Release memory
    LOG(INFO) << "Releasing CPU memory...";
    absl::Status release_status = model->release_memory(ModelLocation::PAGEABLE_CPU);
    REQUIRE(release_status.ok());
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) <= MemoryState::UNALLOCATED);
    REQUIRE(model->get_data_pointer(ModelLocation::PAGEABLE_CPU).empty());

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