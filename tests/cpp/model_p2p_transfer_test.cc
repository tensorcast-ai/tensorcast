// Copyright (c) 2025, StepCast Team. All rights reserved.

// Multi-process test for loading Model via P2P
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "core/common/cuda_api.h" // Use unified CUDA API

// #include "absl/log/check.h" // Avoid macro conflict with Catch2
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"

#include "core/common/model_verification.h" // Add verification support
#include "core/store/components/communication_manager.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/memory_types.h" // For MB definition
#include "core/store/model/model.h"
#include "core/store/model/model_config.h"
#include "core/store/model/model_location.h"

#include <catch2/catch_section_info.hpp>
#include <catch2/catch_test_macros.hpp>

namespace fs = std::filesystem;
using namespace stepcast::store;

// ---------------------------------------------------------------------------
// Test configuration structure
// ---------------------------------------------------------------------------
struct P2PTestConfig {
  std::string server_ip = "127.0.0.1";
  int server_port = 50061;
  std::string model_id = "p2p_transfer_model";
  int gpu_id = 0;
  size_t model_size_mb = 16;
  std::string allocation_mode = "pool";
  std::string register_location = "gpu";
  std::string server_register_location = "gpu";
  std::string client_target_location = "gpu";
};

// --- Shared Constants & Helpers ---
namespace {

const std::string MODEL_SUBDIR = "p2p_transfer_model_files";
const std::string PARTITION_FILENAME = "tensor.data_0";
const size_t POOL_SIZE_MB = 64;
const size_t POOL_CHUNK_SIZE_KB = 2048;

// Helper to create dummy file (similar to other tests)
bool create_dummy_file(const fs::path& path, size_t size) {
  std::ofstream outfile(path, std::ios::binary);
  if (!outfile) {
    LOG(ERROR) << "Failed to open file for writing: " << path;
    return false;
  }
  // Create slightly more varied data
  std::vector<char> buffer(size);
  for (size_t i = 0; i < size; ++i) {
    buffer[i] = static_cast<char>((i % 26) + 'A');
  }
  outfile.write(buffer.data(), size);
  if (!outfile.good()) {
    LOG(ERROR) << "Failed to write data to file: " << path;
    return false;
  }
  LOG(INFO) << "Created dummy file: " << path << " Size: " << size;
  return true;
}

// Helper to setup common resources (pools, temp dir)
struct TestResources {
  fs::path temp_dir;
  fs::path model_data_path;
  fs::path dummy_file_path;
  std::shared_ptr<PinnedMemoryPool> pinned_pool;
  size_t actual_model_size;
  bool is_cuda_available = false;
  int device_count = 0;
  size_t pinned_pool_chunk_size_bytes = 0; // Store the chunk size used

  bool setup(int gpu_id, size_t model_size_bytes) {
    actual_model_size = model_size_bytes;
    temp_dir = fs::temp_directory_path() / "model_p2p_transfer_test";
    model_data_path = temp_dir / MODEL_SUBDIR;
    dummy_file_path = model_data_path / PARTITION_FILENAME;

    std::error_code ec;
    fs::remove_all(temp_dir, ec); // Clean previous runs
    fs::create_directories(model_data_path, ec);
    if (ec) {
      LOG(ERROR) << "Failed to create directories: " << model_data_path << " Error: " << ec.message();
      return false;
    }

    // Create memory pools
    pinned_pool_chunk_size_bytes = POOL_CHUNK_SIZE_KB * 1024;
    pinned_pool = std::make_shared<PinnedMemoryPool>(POOL_SIZE_MB * MB, pinned_pool_chunk_size_bytes);
    if (!pinned_pool) {
      LOG(ERROR) << "Failed to create PinnedMemoryPool";
      return false;
    }
    LOG(INFO) << "PinnedMemoryPool created with chunk size: " << pinned_pool_chunk_size_bytes << " bytes.";

    // Check for CUDA devices and create pool if available
    absl::Status status = stepcast::cuda::get_device_count(&device_count);
    if (status.ok() && device_count > 0) {
      is_cuda_available = true;
      if (gpu_id >= device_count) {
        LOG(ERROR) << "Invalid GPU ID: " << gpu_id << " (Found " << device_count << " devices)";
        return false;
      }
      LOG(INFO) << "CUDA available. Found " << device_count << " devices. Using device " << gpu_id
                << ". CudaPool created.";
    } else {
      is_cuda_available = false;
      LOG(WARNING) << "No CUDA devices found or CUDA error (" << status.message()
                   << "). CUDA pool not created. GPU operations will fail.";
    }

    LOG(INFO) << "Test resources setup complete. Temp dir: " << temp_dir;
    return true;
  }

  void cleanup() {
    std::error_code ec;
    fs::remove_all(temp_dir, ec);
    if (ec) {
      LOG(WARNING) << "Failed to remove temporary directory " << temp_dir << ": " << ec.message();
    }
  }
};

// --- Server Implementation ---
class P2PTestServer {
 public:
  P2PTestServer(const P2PTestConfig& config) : config_(config), server_ready_(false) {}

  bool start() {
    server_thread_ = std::thread([this]() { run(); });

    // Wait for server to be ready
    auto start_time = std::chrono::steady_clock::now();
    while (!server_ready_ && std::chrono::steady_clock::now() - start_time < std::chrono::seconds(30)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return server_ready_;
  }

  void stop() {
    if (server_thread_.joinable()) {
      // In a real implementation, we'd have a proper shutdown mechanism
      server_thread_.detach();
    }
  }

  ~P2PTestServer() {
    stop();
  }

 private:
  void run() {
    std::string server_ip = "0.0.0.0"; // Listen on all interfaces
    int server_port = config_.server_port;
    std::string model_id = config_.model_id;
    int gpu_id = config_.gpu_id;
    size_t model_size = config_.model_size_mb * MB;
    std::string register_loc_str = config_.register_location;
    ModelLocation register_location;
    if (register_loc_str == "cpu") {
      register_location = ModelLocation::PAGEABLE_CPU;
    } else if (register_loc_str == "gpu") {
      register_location = ModelLocation::GPU;
    } else {
      LOG(ERROR) << "Invalid register_location value: " << register_loc_str << ". Use 'cpu' or 'gpu'.";
      return;
    }

    LOG(INFO) << "Starting P2P Test Server...";
    LOG(INFO) << " Server IP: " << server_ip;
    LOG(INFO) << " Server Port: " << server_port;
    LOG(INFO) << " Model ID: " << model_id;
    LOG(INFO) << " GPU ID: " << gpu_id;
    LOG(INFO) << " Model Size: " << model_size << " bytes";
    LOG(INFO) << " Register Location: " << register_loc_str;

    // Set environment variable for P2P server to listen on the correct port
    setenv("STEPCAST_COMM_LOCAL_PORT", std::to_string(server_port).c_str(), 1);

    // Initialize Global CommunicateEngine
    LOG(INFO) << "Initializing Global CommunicateEngine for Server on port " << server_port << "...";
    auto comm_mgr = std::make_shared<stepcast::store::CommunicationManager>();
    // The communication engine needs to listen on the same port that clients will connect to
    absl::Status engine_status = comm_mgr->initialize("0.0.0.0", server_port);
    if (!engine_status.ok()) {
      LOG(ERROR) << "Failed to initialize communication manager: " << engine_status.message();
      return;
    }
    LOG(INFO) << "Test CommunicationManager initialized on port " << server_port;

    auto shared_engine = comm_mgr->get_shared_engine();

    // Setup resources
    TestResources resources;
    if (!resources.setup(gpu_id, model_size)) {
      return;
    }

    // Check if GPU is required but unavailable
    if (register_location == ModelLocation::GPU && !resources.is_cuda_available) {
      LOG(ERROR) << "Cannot register GPU memory: CUDA devices are not available.";
      resources.cleanup();
      return;
    }

    // Create the dummy model file
    if (!create_dummy_file(resources.dummy_file_path, model_size)) {
      resources.cleanup();
      return;
    }

    // Configure Model
    ModelConfig model_config;
    model_config.model_identifier = model_id;

    // Use new DiskSource
    DiskSource disk_src;
    disk_src.path = resources.temp_dir / MODEL_SUBDIR;
    model_config.source = disk_src;

    model_config.pinned_memory_pool = resources.pinned_pool;
    model_config.local_device_id = gpu_id;
    model_config.p2p_comm_enabled = true;
    model_config.expected_model_size = model_size;

    absl::StatusOr<std::unique_ptr<Model>> model_status = Model::create(model_config);

    if (!model_status.ok()) {
      LOG(ERROR) << "Failed to create model: " << model_status.status();
      resources.cleanup();
      return;
    }
    std::unique_ptr<Model> model = std::move(*model_status);
    LOG(INFO) << "Model created successfully: " << model->model_id();

    // Load to CPU (always needed as source is disk)
    {
      LOG(INFO) << "Loading model to CPU...";
      auto load_future = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
      if (!load_future.get().ok()) {
        LOG(ERROR) << "Failed to initiate CPU load";
        resources.cleanup();
        return;
      }
      absl::Status wait_status = model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(60));
      if (!wait_status.ok()) {
        LOG(ERROR) << "Failed to load model to CPU: " << wait_status;
        resources.cleanup();
        return;
      }
      LOG(INFO) << "Model loaded to CPU.";
    }

    // Load to GPU only if registering GPU
    if (register_location == ModelLocation::GPU) {
      LOG(INFO) << "Loading model to GPU (as register_location is GPU)...";
      auto load_future = model->ensure_loaded_async(ModelLocation::GPU);
      if (!load_future.get().ok()) {
        LOG(ERROR) << "Failed to initiate GPU load";
        resources.cleanup();
        return;
      }
      absl::Status wait_status = model->wait_until_loaded(ModelLocation::GPU, absl::Seconds(60));
      if (!wait_status.ok()) {
        LOG(ERROR) << "Failed to load model to GPU: " << wait_status;
        resources.cleanup();
        return;
      }
      LOG(INFO) << "Model loaded to GPU.";
    }

    // Register Memory based on the flag
    LOG(INFO) << "Registering " << register_loc_str << " memory for communication...";
    absl::StatusOr<stepcast::store::CommRegistrationInfo> reg_status =
        model->enable_remote_memory_access(register_location, *shared_engine);

    if (!reg_status.ok()) {
      LOG(ERROR) << "Failed to register " << register_loc_str << " memory for communication: " << reg_status.status();
      resources.cleanup();
      return;
    }
    LOG(INFO) << register_loc_str << " memory registered for communication.";

    const auto info = reg_status.value();
    LOG(INFO) << "--- Comm Registration Info ---";
    LOG(INFO) << "  Model Size: " << info.model_size;
    LOG(INFO) << "  Location: " << register_loc_str;
    LOG(INFO) << "  Device ID: " << info.device_id;
    LOG(INFO) << "  Comm Dev Type: " << info.comm_dev_type;
    LOG(INFO) << "  Buffer Details (" << info.buffer_addresses.size() << " chunks):";
    size_t total_registered_size = 0;
    for (size_t i = 0; i < info.buffer_addresses.size(); ++i) {
      LOG(INFO) << "    Chunk " << i << ":";
      LOG(INFO) << "      Address: " << std::hex << info.buffer_addresses[i] << std::dec;
      LOG(INFO) << "      Size: " << info.buffer_sizes[i];
      LOG(INFO) << "      Key: " << info.remote_memory_keys[i];
      total_registered_size += info.buffer_sizes[i];
    }
    LOG(INFO) << "  Total Registered Size: " << total_registered_size;
    REQUIRE(total_registered_size == info.model_size);
    LOG(INFO) << "------------------------------";

    // Generate verification information for the registered memory
    LOG(INFO) << "Generating verification information for registered " << register_loc_str << " memory...";
    absl::StatusOr<stepcast::store::ModelVerificationInfo> verification_info_status =
        model->generate_verification_info(register_location);

    if (verification_info_status.ok()) {
      const auto& verification_info = verification_info_status.value();
      LOG(INFO) << "--- Model Verification Info ---";
      LOG(INFO) << "  Model Size: " << verification_info.model_size;
      LOG(INFO) << "  Full Hash: 0x" << std::hex << verification_info.full_hash << std::dec;
      LOG(INFO) << "  Key Values: [0x" << std::hex << verification_info.key_values[0] << ", 0x"
                << verification_info.key_values[1] << ", 0x" << verification_info.key_values[2] << std::dec << "]";
      LOG(INFO) << "  First 3 Segment Hashes: [0x" << std::hex << verification_info.segment_hashes[0] << ", 0x"
                << verification_info.segment_hashes[1] << ", 0x" << verification_info.segment_hashes[2] << std::dec
                << "]";
      LOG(INFO) << "Verification info generation completed successfully.";
      LOG(INFO) << "--------------------------------";
    } else {
      LOG(WARNING) << "Failed to generate verification information: " << verification_info_status.status();
      LOG(WARNING) << "Server will continue without verification support.";
    }

    // Signal that server is ready
    LOG(INFO) << "Server setup complete. Registered " << register_loc_str << " memory. Signaling ready...";
    server_ready_ = true;

    // Keep server running for a reasonable time to allow client to connect
    LOG(INFO) << "Server waiting for client connections on port " << server_port << "...";

    // Keep model alive for the duration of the test
    model_ = std::move(model);

    // Wait indefinitely (will be terminated by destructor)
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  P2PTestConfig config_;
  std::atomic<bool> server_ready_;
  std::thread server_thread_;
  std::unique_ptr<Model> model_;
};

// --- Client Implementation ---
class P2PTestClient {
 public:
  P2PTestClient(const P2PTestConfig& config) : config_(config) {}

  bool run() {
    return run_client_test() == 0;
  }

 private:
  [[nodiscard]] int run_client_test() const {
    std::string server_ip = config_.server_ip;
    int server_port = config_.server_port;
    std::string model_id = config_.model_id;
    uint32_t gpu_id = config_.gpu_id;
    size_t model_size = config_.model_size_mb * MB;
    std::string allocation_mode = config_.allocation_mode;
    std::string server_reg_loc_str = config_.server_register_location;
    std::string client_target_loc_str = config_.client_target_location;

    ModelLocation server_registered_location;
    if (server_reg_loc_str == "cpu") {
      server_registered_location = ModelLocation::PAGEABLE_CPU;
    } else if (server_reg_loc_str == "gpu") {
      server_registered_location = ModelLocation::GPU;
    } else {
      LOG(ERROR) << "Invalid server_register_location value: " << server_reg_loc_str << ". Use 'cpu' or 'gpu'.";
      return 1;
    }

    ModelLocation client_target_location;
    if (client_target_loc_str == "cpu") {
      client_target_location = ModelLocation::PAGEABLE_CPU;
    } else if (client_target_loc_str == "gpu") {
      client_target_location = ModelLocation::GPU;
    } else {
      LOG(ERROR) << "Invalid client_target_location value: " << client_target_loc_str << ". Use 'cpu' or 'gpu'.";
      return 1;
    }

    LOG(INFO) << "Starting Client Mode...";
    LOG(INFO) << " Server IP: " << server_ip;
    LOG(INFO) << " Server Port: " << server_port;
    LOG(INFO) << " Model ID: " << model_id;
    LOG(INFO) << " Client GPU ID: " << gpu_id;
    LOG(INFO) << " Expected Model Size: " << model_size << " bytes";
    LOG(INFO) << " Client Allocation Mode: " << allocation_mode;
    LOG(INFO) << " Server Registered Location: " << server_reg_loc_str;
    LOG(INFO) << " Client Target Location: " << client_target_loc_str;

    // --- Check for Unsupported Transfer Combinations based on flags ---
    if (server_registered_location == ModelLocation::GPU && client_target_location == ModelLocation::PAGEABLE_CPU) {
      LOG(ERROR) << "Unsupported test configuration requested: Cannot load from remote GPU directly to local CPU "
                 << "via P2P in this version. Aborting.";
      return 1;
    }
    if (server_registered_location == ModelLocation::PAGEABLE_CPU && client_target_location == ModelLocation::GPU) {
      LOG(ERROR)
          << "Unsupported test configuration requested: Cannot load from remote CPU chunks directly to local GPU "
          << "via P2P in this version. Aborting.";
      return 1;
    }
    LOG(INFO) << "Requested P2P transfer configuration (" << server_reg_loc_str << " -> " << client_target_loc_str
              << ") is supported.";

    // Setup client resources (pools, temp dir for potential staging/logging)
    TestResources resources;

    // Client doesn't need to listen, just connect
    // Create a communication manager without initializing a server
    LOG(INFO) << "Creating CommunicateEngine for Client (no server listening)...";
    auto client_comm_engine = std::make_shared<stepcast::communicator::CommunicateEngine>(false);
    // Initialize without binding to a port (client-only mode)
    // The engine will create outgoing connections as needed
    absl::Status client_engine_status = client_comm_engine->init("127.0.0.1", 0); // Port 0 means no server listening
    if (!client_engine_status.ok()) {
      LOG(ERROR) << "Failed to initialize client communication engine: " << client_engine_status.message();
      resources.cleanup();
      return 1;
    }
    const auto& shared_engine = client_comm_engine;
    if (!resources.setup(gpu_id, model_size)) {
      return 1;
    }

    // Check if GPU target is specified but unavailable
    if (client_target_location == ModelLocation::GPU && !resources.is_cuda_available) {
      LOG(ERROR) << "Cannot target GPU: CUDA devices are not available on the client.";
      resources.cleanup();
      return 1;
    }
    // Check if GPU allocation mode is 'borrow' but unavailable
    if (allocation_mode == "borrow" && client_target_location == ModelLocation::GPU && !resources.is_cuda_available) {
      LOG(ERROR) << "Cannot use allocation_mode 'borrow' for GPU target: CUDA devices are not available on the client.";
      resources.cleanup();
      return 1;
    }

    // Configure Model using P2P source
    // Predict the server's registration details
    // Assume server uses the same GPU ID flag if registering GPU memory
    // Use the PinnedMemoryPool chunk size from the client's resources (assuming it matches server's)
    size_t server_cpu_chunk_size = resources.pinned_pool_chunk_size_bytes; // Use actual chunk size from setup
    int server_gpu_id_used = config_.gpu_id; // Assume server uses the same gpu_id flag value for registration if GPU

    ModelConfig config;
    config.model_identifier = model_id + "_client"; // Use a different client-side ID
    config.pinned_memory_pool = resources.pinned_pool; // Required for P2P transfers
    config.local_device_id = gpu_id;
    config.p2p_comm_enabled = true;
    config.expected_model_size = model_size;
    P2PSource p2p_source;
    p2p_source.size_bytes = model_size;
    p2p_source.ip = server_ip;
    p2p_source.port = static_cast<uint16_t>(server_port);

    // Set the location of remote data
    p2p_source.location.type = server_registered_location;
    p2p_source.location.device_id = (server_registered_location == ModelLocation::GPU)
        ? server_gpu_id_used
        : 1; // Fixed device ID used for CPU registration in MemoryManager

    if (server_registered_location == ModelLocation::GPU) {
      // GPU registration involves a single buffer.
      p2p_source.memory_keys.push_back(absl::StrFormat("%s_GPU_dev%d_chunk0", model_id, server_gpu_id_used));
      p2p_source.buf_sizes.push_back(model_size);
    } else { // CPU registration involves potentially multiple chunks.
      size_t remaining_size = model_size;
      int chunk_index = 0;
      if (server_cpu_chunk_size == 0) {
        LOG(ERROR) << "Server CPU chunk size must be positive.";
        resources.cleanup();
        return 1;
      }
      while (remaining_size > 0) {
        size_t current_chunk_size = std::min(server_cpu_chunk_size, remaining_size);
        // Key format must match the one generated by MemoryManager::enable_remote_memory_access for CPU
        p2p_source.memory_keys.push_back(
            absl::StrFormat("%s_CPU_dev%d_chunk%d", model_id, p2p_source.location.device_id, chunk_index));
        p2p_source.buf_sizes.push_back(current_chunk_size);
        remaining_size -= current_chunk_size;
        chunk_index++;
      }
      // Sanity check
      uint64_t sum_of_buffer_sizes = std::accumulate(p2p_source.buf_sizes.begin(), p2p_source.buf_sizes.end(), 0ULL);
      if (sum_of_buffer_sizes != model_size) {
        LOG(ERROR) << "Sum of predicted CPU chunk sizes does not match total model size.";
        resources.cleanup();
        return 1;
      }
    }

    // Add verification configuration (Note: P2PSource only has enable_checksum, not full verification info)
    p2p_source.enable_checksum =
        false; // TODO: P2PSource doesn't have verification_info field, so we can't store the full info

    // Attach communicator engine to the P2P source (required by P2PLoader)
    p2p_source.comm_engine = shared_engine;
    config.source = p2p_source;

    // Log predicted source info for debugging
    if (const auto* p2p_src = std::get_if<P2PSource>(&config.source)) {
      LOG(INFO) << "--- Predicted P2P Source Config ---";
      LOG(INFO) << "  Model Size: " << p2p_src->size_bytes;
      LOG(INFO) << "  Remote Location: " << location_to_string(p2p_src->location.type);
      LOG(INFO) << "  Remote Address: " << p2p_src->ip;
      LOG(INFO) << "  Remote Port: " << p2p_src->port;
      LOG(INFO) << "  Remote Device ID (used for registration): " << p2p_src->location.device_id;
      LOG(INFO) << "  Remote Keys (" << p2p_src->memory_keys.size() << " chunks):";
      for (size_t i = 0; i < p2p_src->memory_keys.size(); ++i) {
        LOG(INFO) << "    Key[" << i << "]: " << p2p_src->memory_keys[i] << ", Size: " << p2p_src->buf_sizes[i];
      }
      LOG(INFO) << "------------------------------------";
    }

    // Create Model (this will internally create P2PLoader and attempt initialization)
    LOG(INFO) << "Creating Model instance with P2P source...";
    if (shared_engine == nullptr) {
      LOG(ERROR) << "Communication engine not initialized for client.";
      resources.cleanup();
      return 1;
    }

    absl::StatusOr<std::unique_ptr<Model>> model_status =
        Model::create(config); // Assuming Model::create wires up engine to MemoryManager
    if (!model_status.ok()) {
      LOG(ERROR) << "Failed to create model from P2P source: " << model_status.status();
      resources.cleanup();
      return 1;
    }
    std::unique_ptr<Model> model = std::move(*model_status);
    LOG(INFO) << "Model (P2P source) created successfully: " << model->model_id();

    // --- Handle GPU Memory Allocation (Pool vs Borrow) --- only if target is GPU
    void* borrowed_gpu_ptr = nullptr; // To manage cleanup if borrowed
    auto cleanup_borrowed_memory = [&]() {
      if (borrowed_gpu_ptr) {
        LOG(INFO) << "Freeing borrowed GPU memory: " << borrowed_gpu_ptr;
        absl::Status set_device_status = stepcast::cuda::set_device(gpu_id);
        if (!set_device_status.ok()) {
          LOG(ERROR) << "Failed to set CUDA device " << gpu_id
                     << " before freeing memory: " << set_device_status.message();
        }
        absl::Status free_status = stepcast::cuda::free(borrowed_gpu_ptr);
        if (!free_status.ok()) {
          LOG(ERROR) << "Failed to free borrowed GPU memory: " << free_status.message();
        }
        borrowed_gpu_ptr = nullptr;
      }
    };

    if (client_target_location == ModelLocation::GPU && allocation_mode == "borrow") {
      LOG(ERROR) << "Borrowing GPU memory is not supported in this version.";
      resources.cleanup();
      return 1;
    }
    if (client_target_location == ModelLocation::GPU && allocation_mode == "pool") {
      LOG(INFO) << "Using internal memory pool for GPU allocation.";
    } else if (client_target_location == ModelLocation::PAGEABLE_CPU) {
      LOG(INFO) << "Target location is CPU, using internal PinnedMemory pool.";
      // No specific allocation needed here, Model handles it via ensure_loaded_async
    }

    // --- Test Loading to Client Target Location via P2P ---
    LOG(INFO) << "Loading model to CLIENT " << client_target_loc_str << " via P2P from SERVER " << server_reg_loc_str
              << "...";
    auto load_future = model->ensure_loaded_async(client_target_location);
    absl::Status future_get_status = load_future.get(); // Check status immediately
    if (!future_get_status.ok()) {
      LOG(ERROR) << "ensure_loaded_async failed immediately for " << client_target_loc_str << ": " << future_get_status;
      cleanup_borrowed_memory();
      resources.cleanup();
      return 1;
    }

    // Wait for the actual loading to complete
    absl::Status wait_status = model->wait_until_loaded(client_target_location, absl::Seconds(60));

    if (!wait_status.ok()) {
      LOG(ERROR) << "Failed to load model to client " << client_target_loc_str << " via P2P from " << server_reg_loc_str
                 << ": " << wait_status;
      cleanup_borrowed_memory();
      resources.cleanup();
      return 1; // Loading failed, exit with error
    }

    LOG(INFO) << "Successfully loaded model to client " << client_target_loc_str << " via P2P from "
              << server_reg_loc_str << ".";

    // Verification
    if (model->get_memory_state(client_target_location) != MemoryState::LOADED) {
      LOG(ERROR) << "Model not in LOADED state after P2P transfer";
      cleanup_borrowed_memory();
      resources.cleanup();
      return 1;
    }
    std::vector<void*> data_ptrs = model->get_data_pointer(client_target_location);
    if (data_ptrs.empty()) {
      LOG(ERROR) << "get_data_pointer returned empty vector for loaded state.";
      cleanup_borrowed_memory();
      resources.cleanup();
      return 1;
    }
    if (data_ptrs[0] == nullptr) {
      LOG(ERROR) << "First data pointer is null for loaded state.";
      cleanup_borrowed_memory();
      resources.cleanup();
      return 1;
    }
    LOG(INFO) << client_target_loc_str << " Pointer(s) on client (" << data_ptrs.size() << " chunks): " << data_ptrs[0]
              << (data_ptrs.size() > 1 ? " ..." : "");

    // --- Data content verification (read first and last byte) ---
    // Always copy to host memory for reliable verification, regardless of client_target_location.
    std::vector<char> host_verification_buffer;
    bool verification_data_ready = false;

    if (model_size > 0) {
      LOG(INFO) << "Preparing host buffer for verification (size " << model_size << ")...";
      try {
        host_verification_buffer.resize(model_size);
      } catch (const std::bad_alloc& e) {
        LOG(ERROR) << "Failed to allocate host verification buffer of size " << model_size << ": " << e.what();
        // Cannot verify content if allocation fails
        host_verification_buffer.clear();
      }

      if (!host_verification_buffer.empty()) {
        if (client_target_location == ModelLocation::PAGEABLE_CPU) {
          LOG(INFO) << "Copying data from client CPU chunks to host verification buffer...";
          if (data_ptrs.empty()) {
            LOG(ERROR) << "No data pointers available";
            cleanup_borrowed_memory();
            resources.cleanup();
            return 1;
          }
          size_t chunk_size = model->get_memory_manager().get_cpu_chunk_size();
          if (chunk_size == 0) {
            LOG(ERROR) << "Invalid CPU chunk size";
            cleanup_borrowed_memory();
            resources.cleanup();
            return 1;
          }
          size_t bytes_copied = 0;
          char* dest_ptr = host_verification_buffer.data();
          for (size_t i = 0; i < data_ptrs.size(); ++i) {
            if (data_ptrs[i] == nullptr) {
              LOG(ERROR) << "Null data pointer at index " << i;
              cleanup_borrowed_memory();
              resources.cleanup();
              return 1;
            }
            size_t size_to_copy = (i == data_ptrs.size() - 1) ? (model_size - bytes_copied) : chunk_size;
            if (bytes_copied + size_to_copy > model_size) {
              LOG(ERROR) << "Buffer overflow: attempting to copy beyond model size";
              cleanup_borrowed_memory();
              resources.cleanup();
              return 1;
            }
            memcpy(dest_ptr + bytes_copied, data_ptrs[i], size_to_copy);
            bytes_copied += size_to_copy;
          }
          if (bytes_copied != model_size) {
            LOG(ERROR) << "Did not copy expected number of bytes from CPU chunks. Expected: " << model_size
                       << ", Got: " << bytes_copied;
            cleanup_borrowed_memory();
            resources.cleanup();
            return 1;
          }
          verification_data_ready = true;
          LOG(INFO) << "Successfully copied from " << data_ptrs.size() << " CPU chunks to host buffer.";

        } else { // client_target_location == ModelLocation::GPU
          LOG(INFO) << "Copying data from client GPU buffer to host verification buffer...";
          if (data_ptrs.size() != 1) {
            LOG(ERROR) << "GPU should have exactly one pointer, got " << data_ptrs.size();
            cleanup_borrowed_memory();
            resources.cleanup();
            return 1;
          }
          void* gpu_ptr = data_ptrs[0];
          if (gpu_ptr == nullptr) {
            LOG(ERROR) << "GPU pointer is null";
            cleanup_borrowed_memory();
            resources.cleanup();
            return 1;
          }

          absl::Status cuda_status = stepcast::cuda::set_device(gpu_id);
          if (!cuda_status.ok()) {
            LOG(ERROR) << "Failed to set CUDA device " << gpu_id
                       << " before verification copy: " << cuda_status.message();
          } else {
            cuda_status =
                stepcast::cuda::memcpy(host_verification_buffer.data(), gpu_ptr, model_size, cudaMemcpyDeviceToHost);
            if (!cuda_status.ok()) {
              LOG(ERROR) << "cudaMemcpy D->H failed for verification: " << cuda_status.message();
            } else {
              verification_data_ready = true;
              LOG(INFO) << "Successfully copied from GPU to host buffer.";
            }
          }
        }
      }
    } else {
      LOG(INFO) << "Model size is 0, skipping verification copy.";
      // Treat as ready, checks below will handle size 0 correctly.
      verification_data_ready = true;
    }

    // Perform the check on the host_verification_buffer
    char first_byte = 0;
    char last_byte = 0;
    char expected_first_byte = 0;
    char expected_last_byte = 0;

    if (model_size > 0) {
      expected_first_byte = 'A'; // Based on create_dummy_file
      expected_last_byte = static_cast<char>(((model_size - 1) % 26) + 'A'); // Based on create_dummy_file

      if (verification_data_ready) {
        first_byte = host_verification_buffer[0];
        last_byte = host_verification_buffer[model_size - 1];
        LOG(INFO) << "Verifying data content from host buffer... First byte: '" << first_byte << "' ("
                  << (int)first_byte << ") (Expected: '" << expected_first_byte << "'), Last byte: '" << last_byte
                  << "' (" << (int)last_byte << ") (Expected: '" << expected_last_byte << "')";
        if (first_byte != expected_first_byte) {
          LOG(ERROR) << "First byte mismatch";
          cleanup_borrowed_memory();
          resources.cleanup();
          return 1;
        }
        if (last_byte != expected_last_byte) {
          LOG(ERROR) << "Last byte mismatch";
          cleanup_borrowed_memory();
          resources.cleanup();
          return 1;
        }
        LOG(INFO) << "Data content verification passed.";
      } else {
        LOG(ERROR) << "Verification data was not successfully prepared. Skipping content checks.";
        cleanup_borrowed_memory();
        resources.cleanup();
        return 1;
      }
    } else {
      LOG(INFO) << "Skipping data content verification for zero-size model.";
      if (!verification_data_ready) {
        LOG(ERROR) << "Verification not ready for zero-size model";
        cleanup_borrowed_memory();
        resources.cleanup();
        return 1;
      }
    }

    // --- Model-level Verification Demo ---
    LOG(INFO) << "Testing model-level verification system...";
    auto client_verify_info = model->generate_verification_info(client_target_location);
    if (client_verify_info.ok()) {
      auto verify_status = model->verify_key_points(client_target_location, client_verify_info.value());
      LOG(INFO) << "Model verification: " << (verify_status.ok() ? "PASSED" : "FAILED");
    }

    // Release memory for the target location
    LOG(INFO) << "Releasing client " << client_target_loc_str << " memory...";
    absl::Status release_status = model->release_memory(client_target_location);
    if (!release_status.ok()) {
      LOG(ERROR) << "Failed to release client memory: " << release_status;
      cleanup_borrowed_memory();
      resources.cleanup();
      return 1;
    }
    if (model->get_memory_state(client_target_location) > MemoryState::UNALLOCATED) {
      LOG(ERROR) << "Memory state not properly released";
      cleanup_borrowed_memory();
      resources.cleanup();
      return 1;
    }
    LOG(INFO) << "Client " << client_target_loc_str << " memory released.";

    LOG(INFO) << "Client operations complete.";
    cleanup_borrowed_memory(); // Free memory if it was borrowed
    resources.cleanup();
    return 0;
  }

  P2PTestConfig config_;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Catch2 integration tests
// ---------------------------------------------------------------------------

TEST_CASE("Model P2P Transfer Integration Tests", "[model_p2p_transfer]") {
  // Initialize verbose logging for easier CI debugging
  absl::SetGlobalVLogLevel(1);
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);

  // Detect CUDA capability
  int device_count = 0;
  absl::Status cuda_status = stepcast::cuda::get_device_count(&device_count);
  bool has_cuda = (cuda_status.ok() && device_count > 0);

  SECTION("GPU to GPU transfer") {
    if (!has_cuda) {
      WARN("CUDA not available - skipping GPU to GPU test");
      return;
    }

    P2PTestConfig config;
    config.server_port = 50062;
    config.model_id = "p2p_gpu_to_gpu_test";
    config.gpu_id = 0;
    config.model_size_mb = 16;
    config.register_location = "gpu";
    config.server_register_location = "gpu";
    config.client_target_location = "gpu";

    P2PTestServer server(config);
    REQUIRE(server.start());

    // Give server time to fully initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    P2PTestClient client(config);
    REQUIRE(client.run());
  }

  SECTION("Small model transfer") {
    if (!has_cuda) {
      WARN("CUDA not available - skipping small model GPU test");
      return;
    }

    P2PTestConfig config;
    config.server_port = 50064;
    config.model_id = "p2p_small_model_test";
    config.gpu_id = 0;
    config.model_size_mb = 1; // 1MB model
    config.register_location = "gpu";
    config.server_register_location = "gpu";
    config.client_target_location = "gpu";

    P2PTestServer server(config);
    REQUIRE(server.start());

    // Give server time to fully initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    P2PTestClient client(config);
    REQUIRE(client.run());
  }

  SECTION("Large model transfer") {
    if (!has_cuda) {
      WARN("CUDA not available - skipping large model GPU test");
      return;
    }

    P2PTestConfig config;
    config.server_port = 50065;
    config.model_id = "p2p_large_model_test";
    config.gpu_id = 0;
    config.model_size_mb = 64; // 64MB model
    config.register_location = "gpu";
    config.server_register_location = "gpu";
    config.client_target_location = "gpu";

    P2PTestServer server(config);
    REQUIRE(server.start());

    // Give server time to fully initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    P2PTestClient client(config);
    REQUIRE(client.run());
  }
}