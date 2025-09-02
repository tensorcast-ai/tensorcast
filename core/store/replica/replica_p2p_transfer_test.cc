// Copyright (c) 2025, TensorCast Team.

// Multi-process test for loading Replica via P2P
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/common/cuda_api.h" // Use unified CUDA API
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/testing/common.h"
#include "core/testing/test_helpers.h"

// #include "absl/log/check.h" // Avoid macro conflict with Catch2
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"

#include "core/common/artifact_verification.h" // Add verification support
#include "core/common/memory/memory_location.h"
#include "core/store/components/communication_manager.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/memory_types.h" // For MB definition
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"

#include <catch2/catch_section_info.hpp>
#include <catch2/catch_test_macros.hpp>

namespace fs = std::filesystem;
using tensorcast::common::ArtifactVerificationInfo;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::common::memory::PinnedMemoryPool;
using tensorcast::store::MB;
using tensorcast::store::P2PSource;
using tensorcast::store::components::CommunicationManager;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::replica::MemoryState;
using tensorcast::store::replica::Replica;
using tensorcast::store::replica::ReplicaConfig;
using tensorcast::testing::create_dummy_file;
using tensorcast::testing::find_available_port;
using tensorcast::testing::is_cuda_available;
using tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir;

// ---------------------------------------------------------------------------
// Test configuration structure
// ---------------------------------------------------------------------------
struct P2PTestConfig {
  std::string server_ip = "127.0.0.1";
  int server_port = 50061;
  std::string artifact_id = "p2p_transfer_artifact";
  int gpu_id = 0;
  size_t artifact_size_mb = 16;
  std::string allocation_mode = "pool";
  std::string register_location = "gpu";
  std::string server_register_location = "gpu";
  std::string client_target_location = "gpu";
};

// --- Shared Constants & Helpers ---
namespace {

const std::string ARTIFACT_SUBDIR = "p2p_transfer_artifact_files";
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
  fs::path artifact_dir;
  fs::path dummy_file_path;
  std::shared_ptr<PinnedMemoryPool> pinned_pool;
  size_t actual_artifact_size;
  bool is_cuda_available = false;
  int device_count = 0;
  size_t pinned_pool_chunk_size_bytes = 0; // Store the chunk size used
  std::shared_ptr<tensorcast::common::memory::DistributedVirtualMemoryPool> dvmp;

  bool setup(int gpu_id, size_t artifact_size) {
    actual_artifact_size = artifact_size;
    temp_dir = fs::temp_directory_path() / "replica_p2p_transfer_test";
    artifact_dir = temp_dir / ARTIFACT_SUBDIR;
    dummy_file_path = artifact_dir / PARTITION_FILENAME;

    std::error_code ec;
    fs::remove_all(temp_dir, ec); // Clean previous runs
    fs::create_directories(artifact_dir, ec);
    if (ec) {
      LOG(ERROR) << "Failed to create directories: " << artifact_dir << " Error: " << ec.message();
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

    // Create DVMP and StreamingPinnedBuffer
    dvmp = std::make_shared<tensorcast::common::memory::DistributedVirtualMemoryPool>();
    if (!dvmp) {
      LOG(ERROR) << "Failed to create DistributedVirtualMemoryPool";
      return false;
    }

    // Check for CUDA devices and create pool if available
    absl::Status status = tensorcast::cuda::get_device_count(&device_count);
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
    std::string artifact_id = config_.artifact_id;
    int gpu_id = config_.gpu_id;
    size_t artifact_size = config_.artifact_size_mb * MB;
    std::string register_loc_str = config_.register_location;
    MemoryLocation register_location;
    if (register_loc_str == "cpu") {
      register_location = MemoryLocation::PAGEABLE_CPU;
    } else if (register_loc_str == "gpu") {
      register_location = MemoryLocation::GPU;
    } else {
      LOG(ERROR) << "Invalid register_location value: " << register_loc_str << ". Use 'cpu' or 'gpu'.";
      return;
    }

    LOG(INFO) << "Starting P2P Test Server...";
    LOG(INFO) << " Server IP: " << server_ip;
    LOG(INFO) << " Server Port: " << server_port;
    LOG(INFO) << " Replica ID: " << artifact_id;
    LOG(INFO) << " GPU ID: " << gpu_id;
    LOG(INFO) << " Replica Size: " << artifact_size << " bytes";
    LOG(INFO) << " Register Location: " << register_loc_str;

    // Set environment variable for P2P server to listen on the correct port
    setenv("TENSORCAST_COMM_LOCAL_PORT", std::to_string(server_port).c_str(), 1);

    // Initialize Global CommunicateEngine
    LOG(INFO) << "Initializing Global CommunicateEngine for Server on port " << server_port << "...";
    auto comm_mgr = std::make_shared<CommunicationManager>();
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
    if (!resources.setup(gpu_id, artifact_size)) {
      return;
    }

    // Check if GPU is required but unavailable
    if (register_location == MemoryLocation::GPU && !resources.is_cuda_available) {
      LOG(ERROR) << "Cannot register GPU memory: CUDA devices are not available.";
      resources.cleanup();
      return;
    }

    // Create the dummy replica file
    if (!create_dummy_file(resources.dummy_file_path, artifact_size)) {
      resources.cleanup();
      return;
    }
    // RFC-0007 metadata for standard partitions
    auto st_desc =
        ::tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(resources.temp_dir / ARTIFACT_SUBDIR);
    if (!st_desc.ok()) {
      LOG(ERROR) << "Failed to write descriptor/index: " << st_desc;
      resources.cleanup();
      return;
    }

    // Configure Replica
    DiskSource disk_src;
    disk_src.path = resources.temp_dir / ARTIFACT_SUBDIR;
    ReplicaConfig replica_config{
        .source = disk_src,
        .artifact_identifier = artifact_id,
        .device_type =
            (register_location == MemoryLocation::GPU) ? ::tensorcast::DeviceType::GPU : ::tensorcast::DeviceType::CPU,
        .local_device_id = (register_location == MemoryLocation::GPU) ? gpu_id : -1,
        .pinned_memory_pool = resources.pinned_pool,
        .dvmp = resources.dvmp,
        .expected_artifact_size = artifact_size,
        .p2p_comm_enabled = true};

    absl::StatusOr<std::unique_ptr<Replica>> created_replica = Replica::create(replica_config);

    if (!created_replica.ok()) {
      LOG(ERROR) << "Failed to create replica: " << created_replica.status();
      resources.cleanup();
      return;
    }
    std::unique_ptr<Replica> replica = std::move(*created_replica);
    LOG(INFO) << "Replica created successfully: " << replica->artifact_id();

    // Load to CPU (always needed as source is disk)
    {
      LOG(INFO) << "Loading replica to CPU...";
      auto load_future = replica->ensure_loaded_async(MemoryLocation::PAGEABLE_CPU);
      if (!load_future.get().ok()) {
        LOG(ERROR) << "Failed to initiate CPU load";
        resources.cleanup();
        return;
      }
      absl::Status wait_status = replica->wait_until_loaded(MemoryLocation::PAGEABLE_CPU, absl::Seconds(60));
      if (!wait_status.ok()) {
        LOG(ERROR) << "Failed to load replica to CPU: " << wait_status;
        resources.cleanup();
        return;
      }
      LOG(INFO) << "Replica loaded to CPU.";
    }

    // Load to GPU only if registering GPU
    if (register_location == MemoryLocation::GPU) {
      LOG(INFO) << "Loading replica to GPU (as register_location is GPU)...";
      auto load_future = replica->ensure_loaded_async(MemoryLocation::GPU);
      if (!load_future.get().ok()) {
        LOG(ERROR) << "Failed to initiate GPU load";
        resources.cleanup();
        return;
      }
      absl::Status wait_status = replica->wait_until_loaded(MemoryLocation::GPU, absl::Seconds(60));
      if (!wait_status.ok()) {
        LOG(ERROR) << "Failed to load replica to GPU: " << wait_status;
        resources.cleanup();
        return;
      }
      LOG(INFO) << "Replica loaded to GPU.";
    }

    // Register Memory based on the flag
    LOG(INFO) << "Registering " << register_loc_str << " memory for communication...";
    absl::StatusOr<tensorcast::store::CommRegistrationInfo> reg_status =
        replica->enable_remote_memory_access(register_location, *shared_engine);

    if (!reg_status.ok()) {
      LOG(ERROR) << "Failed to register " << register_loc_str << " memory for communication: " << reg_status.status();
      resources.cleanup();
      return;
    }
    LOG(INFO) << register_loc_str << " memory registered for communication.";

    const auto info = reg_status.value();
    LOG(INFO) << "--- Comm Registration Info ---";
    LOG(INFO) << "  Replica Size: " << info.artifact_size;
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
    REQUIRE(total_registered_size == info.artifact_size);
    LOG(INFO) << "------------------------------";

    // Generate verification information for the registered memory
    LOG(INFO) << "Generating verification information for registered " << register_loc_str << " memory...";
    absl::StatusOr<ArtifactVerificationInfo> verification_info_status =
        replica->generate_verification_info(register_location);

    if (verification_info_status.ok()) {
      const auto& verification_info = verification_info_status.value();
      LOG(INFO) << "--- Replica Verification Info ---";
      LOG(INFO) << "  Replica Size: " << verification_info.artifact_size;
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

    // Keep replica alive for the duration of the test
    replica_ = std::move(replica);

    // Wait indefinitely (will be terminated by destructor)
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  P2PTestConfig config_;
  std::atomic<bool> server_ready_;
  std::thread server_thread_;
  std::unique_ptr<Replica> replica_;
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
    std::string artifact_id = config_.artifact_id;
    uint32_t gpu_id = config_.gpu_id;
    size_t artifact_size = config_.artifact_size_mb * MB;
    std::string allocation_mode = config_.allocation_mode;
    std::string server_reg_loc_str = config_.server_register_location;
    std::string client_target_loc_str = config_.client_target_location;

    MemoryLocation server_registered_location;
    if (server_reg_loc_str == "cpu") {
      server_registered_location = MemoryLocation::PAGEABLE_CPU;
    } else if (server_reg_loc_str == "gpu") {
      server_registered_location = MemoryLocation::GPU;
    } else {
      LOG(ERROR) << "Invalid server_register_location value: " << server_reg_loc_str << ". Use 'cpu' or 'gpu'.";
      return 1;
    }

    MemoryLocation client_target_location;
    if (client_target_loc_str == "cpu") {
      client_target_location = MemoryLocation::PAGEABLE_CPU;
    } else if (client_target_loc_str == "gpu") {
      client_target_location = MemoryLocation::GPU;
    } else {
      LOG(ERROR) << "Invalid client_target_location value: " << client_target_loc_str << ". Use 'cpu' or 'gpu'.";
      return 1;
    }

    LOG(INFO) << "Starting Client Mode...";
    LOG(INFO) << " Server IP: " << server_ip;
    LOG(INFO) << " Server Port: " << server_port;
    LOG(INFO) << " Replica ID: " << artifact_id;
    LOG(INFO) << " Client GPU ID: " << gpu_id;
    LOG(INFO) << " Expected Replica Size: " << artifact_size << " bytes";
    LOG(INFO) << " Client Allocation Mode: " << allocation_mode;
    LOG(INFO) << " Server Registered Location: " << server_reg_loc_str;
    LOG(INFO) << " Client Target Location: " << client_target_loc_str;

    // --- Check for Unsupported Transfer Combinations based on flags ---
    if (server_registered_location == MemoryLocation::GPU && client_target_location == MemoryLocation::PAGEABLE_CPU) {
      LOG(ERROR) << "Unsupported test configuration requested: Cannot load from remote GPU directly to local CPU "
                 << "via P2P in this version. Aborting.";
      return 1;
    }
    if (server_registered_location == MemoryLocation::PAGEABLE_CPU && client_target_location == MemoryLocation::GPU) {
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
    tensorcast::communicator::CommunicatorConfig cfg;
    cfg.set_enable_rdma(false);
    auto client_comm_engine = std::make_shared<tensorcast::communicator::engine::CommunicateEngine>(cfg);
    // Bind to a dedicated, available client port to facilitate P2P connections
    int client_port = find_available_port(server_port + 1);
    if (client_port <= 0 || client_port == server_port) {
      client_port = server_port + 1; // fallback
    }
    absl::Status client_engine_status = client_comm_engine->init("127.0.0.1", static_cast<uint16_t>(client_port));
    if (!client_engine_status.ok()) {
      LOG(ERROR) << "Failed to initialize client communication engine: " << client_engine_status.message();
      resources.cleanup();
      return 1;
    }
    const auto& shared_engine = client_comm_engine;
    if (!resources.setup(gpu_id, artifact_size)) {
      return 1;
    }

    // Check if GPU target is specified but unavailable
    if (client_target_location == MemoryLocation::GPU && !resources.is_cuda_available) {
      LOG(ERROR) << "Cannot target GPU: CUDA devices are not available on the client.";
      resources.cleanup();
      return 1;
    }
    // Check if GPU allocation mode is 'borrow' but unavailable
    if (allocation_mode == "borrow" && client_target_location == MemoryLocation::GPU && !resources.is_cuda_available) {
      LOG(ERROR) << "Cannot use allocation_mode 'borrow' for GPU target: CUDA devices are not available on the client.";
      resources.cleanup();
      return 1;
    }

    // Configure Replica using P2P source
    // Predict the server's registration details (new unified key format)
    int server_gpu_id_used = config_.gpu_id; // server uses the same gpu_id flag value

    P2PSource p2p_source;
    p2p_source.size_bytes = artifact_size;
    p2p_source.ip = server_ip;
    p2p_source.port = static_cast<uint16_t>(server_port);

    // Set the location of remote data
    p2p_source.location.type = server_registered_location;
    p2p_source.location.device_id = (server_registered_location == MemoryLocation::GPU) ? server_gpu_id_used : -1;

    // With new export APIs, both CPU and GPU registrations coalesce to ranges.
    // For full-replica export, expect a single range with index 0.
    if (server_registered_location == MemoryLocation::GPU) {
      p2p_source.memory_keys.push_back(absl::StrFormat("%s_GPU_chunk_%d", artifact_id, 0));
      p2p_source.buf_sizes.push_back(artifact_size);
    } else {
      p2p_source.memory_keys.push_back(absl::StrFormat("%s_CPU_chunk_%d", artifact_id, 0));
      p2p_source.buf_sizes.push_back(artifact_size);
    }

    // Add verification configuration (Note: P2PSource only has enable_checksum, not full verification info)
    p2p_source.enable_checksum =
        false; // TODO: P2PSource doesn't have verification_info field, so we can't store the full info

    // Attach communicator engine to the P2P source (required by P2PLoader)
    p2p_source.comm_engine = shared_engine;

    // Build ReplicaConfig via aggregate initialization (avoid default construction)
    ReplicaConfig config{
        .source = p2p_source,
        .artifact_identifier = artifact_id + "_client",
        .device_type = (client_target_location == MemoryLocation::GPU) ? ::tensorcast::DeviceType::GPU
                                                                       : ::tensorcast::DeviceType::CPU,
        .local_device_id = (client_target_location == MemoryLocation::GPU) ? static_cast<int>(gpu_id) : -1,
        .pinned_memory_pool = resources.pinned_pool,
        .dvmp = resources.dvmp,
        .expected_artifact_size = artifact_size,
        .p2p_comm_enabled = true};

    // Log predicted source info for debugging
    if (const auto* p2p_src = std::get_if<P2PSource>(&config.source)) {
      LOG(INFO) << "--- Predicted P2P Source Config ---";
      LOG(INFO) << "  Replica Size: " << p2p_src->size_bytes;
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

    // Create Replica (this will internally create P2PLoader and attempt initialization)
    LOG(INFO) << "Creating Replica instance with P2P source...";
    if (shared_engine == nullptr) {
      LOG(ERROR) << "Communication engine not initialized for client.";
      resources.cleanup();
      return 1;
    }

    absl::StatusOr<std::unique_ptr<Replica>> created = Replica::create(config);
    if (!created.ok()) {
      LOG(ERROR) << "Failed to create replica from P2P source: " << created.status();
      resources.cleanup();
      return 1;
    }
    std::unique_ptr<Replica> replica = std::move(*created);
    LOG(INFO) << "Replica (P2P source) created successfully: " << replica->artifact_id();

    // --- Handle GPU Memory Allocation (Pool vs Borrow) --- only if target is GPU
    void* borrowed_gpu_ptr = nullptr; // To manage cleanup if borrowed
    auto cleanup_borrowed_memory = [&]() {
      if (borrowed_gpu_ptr) {
        LOG(INFO) << "Freeing borrowed GPU memory: " << borrowed_gpu_ptr;
        absl::Status set_device_status = tensorcast::cuda::set_device(gpu_id);
        if (!set_device_status.ok()) {
          LOG(ERROR) << "Failed to set CUDA device " << gpu_id
                     << " before freeing memory: " << set_device_status.message();
        }
        absl::Status free_status = tensorcast::cuda::free(borrowed_gpu_ptr);
        if (!free_status.ok()) {
          LOG(ERROR) << "Failed to free borrowed GPU memory: " << free_status.message();
        }
        borrowed_gpu_ptr = nullptr;
      }
    };

    if (client_target_location == MemoryLocation::GPU && allocation_mode == "borrow") {
      LOG(ERROR) << "Borrowing GPU memory is not supported in this version.";
      resources.cleanup();
      return 1;
    }
    if (client_target_location == MemoryLocation::GPU && allocation_mode == "pool") {
      LOG(INFO) << "Using internal memory pool for GPU allocation.";
    } else if (client_target_location == MemoryLocation::PAGEABLE_CPU) {
      LOG(INFO) << "Target location is CPU, using internal PinnedMemory pool.";
      // No specific allocation needed here, Replica handles it via ensure_loaded_async
    }

    // --- Test Loading to Client Target Location via P2P ---
    LOG(INFO) << "Loading replica to CLIENT " << client_target_loc_str << " via P2P from SERVER " << server_reg_loc_str
              << "...";
    auto load_future = replica->ensure_loaded_async(client_target_location);
    absl::Status future_get_status = load_future.get(); // Check status immediately
    if (!future_get_status.ok()) {
      LOG(ERROR) << "ensure_loaded_async failed immediately for " << client_target_loc_str << ": " << future_get_status;
      cleanup_borrowed_memory();
      resources.cleanup();
      return 1;
    }

    // Wait for the actual loading to complete
    absl::Status wait_status = replica->wait_until_loaded(client_target_location, absl::Seconds(60));

    if (!wait_status.ok()) {
      LOG(ERROR) << "Failed to load replica to client " << client_target_loc_str << " via P2P from "
                 << server_reg_loc_str << ": " << wait_status;
      cleanup_borrowed_memory();
      resources.cleanup();
      return 1; // Loading failed, exit with error
    }

    LOG(INFO) << "Successfully loaded replica to client " << client_target_loc_str << " via P2P from "
              << server_reg_loc_str << ".";

    // Verification
    if (replica->get_memory_state(client_target_location) != MemoryState::LOADED) {
      LOG(ERROR) << "Replica not in LOADED state after P2P transfer";
      cleanup_borrowed_memory();
      resources.cleanup();
      return 1;
    }
    std::vector<void*> data_ptrs = replica->get_data_pointer(client_target_location);
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

    if (artifact_size > 0) {
      LOG(INFO) << "Preparing host buffer for verification (size " << artifact_size << ")...";
      try {
        host_verification_buffer.resize(artifact_size);
      } catch (const std::bad_alloc& e) {
        LOG(ERROR) << "Failed to allocate host verification buffer of size " << artifact_size << ": " << e.what();
        // Cannot verify content if allocation fails
        host_verification_buffer.clear();
      }

      if (!host_verification_buffer.empty()) {
        if (client_target_location == MemoryLocation::PAGEABLE_CPU) {
          LOG(INFO) << "Copying data from client CPU buffer to host verification buffer...";
          if (data_ptrs.size() != 1 || data_ptrs[0] == nullptr) {
            LOG(ERROR) << "CPU should have exactly one valid pointer, got " << data_ptrs.size();
            cleanup_borrowed_memory();
            resources.cleanup();
            return 1;
          }
          std::memcpy(host_verification_buffer.data(), data_ptrs[0], artifact_size);
          verification_data_ready = true;
          LOG(INFO) << "Successfully copied from CPU to host buffer.";
        } else { // client_target_location == MemoryLocation::GPU
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

          absl::Status cuda_status = tensorcast::cuda::set_device(gpu_id);
          if (!cuda_status.ok()) {
            LOG(ERROR) << "Failed to set CUDA device " << gpu_id
                       << " before verification copy: " << cuda_status.message();
          } else {
            cuda_status = tensorcast::cuda::memcpy(
                host_verification_buffer.data(), gpu_ptr, artifact_size, cudaMemcpyDeviceToHost);
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
      LOG(INFO) << "Artifact size is 0, skipping verification copy.";
      // Treat as ready, checks below will handle size 0 correctly.
      verification_data_ready = true;
    }

    // Perform the check on the host_verification_buffer
    char first_byte = 0;
    char last_byte = 0;
    char expected_first_byte = 0;
    char expected_last_byte = 0;

    if (artifact_size > 0) {
      expected_first_byte = 'A'; // Based on create_dummy_file
      expected_last_byte = static_cast<char>(((artifact_size - 1) % 26) + 'A'); // Based on create_dummy_file

      if (verification_data_ready) {
        first_byte = host_verification_buffer[0];
        last_byte = host_verification_buffer[artifact_size - 1];
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
      LOG(INFO) << "Skipping data content verification for zero-size replica.";
      if (!verification_data_ready) {
        LOG(ERROR) << "Verification not ready for zero-size replica";
        cleanup_borrowed_memory();
        resources.cleanup();
        return 1;
      }
    }

    // --- Replica-level Verification Demo ---
    LOG(INFO) << "Testing replica-level verification system...";
    auto client_verify_info = replica->generate_verification_info(client_target_location);
    if (client_verify_info.ok()) {
      auto verify_status = replica->verify_key_points(client_target_location, client_verify_info.value());
      LOG(INFO) << "Replica verification: " << (verify_status.ok() ? "PASSED" : "FAILED");
    }

    // Release memory for the target location
    LOG(INFO) << "Releasing client " << client_target_loc_str << " memory...";
    absl::Status release_status = replica->release_memory(client_target_location);
    if (!release_status.ok()) {
      LOG(ERROR) << "Failed to release client memory: " << release_status;
      cleanup_borrowed_memory();
      resources.cleanup();
      return 1;
    }
    if (replica->get_memory_state(client_target_location) > MemoryState::UNALLOCATED) {
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

TEST_CASE("Replica P2P Transfer Integration Tests", "[replica_p2p_transfer]") {
  // Initialize verbose logging for easier CI debugging
  absl::SetGlobalVLogLevel(1);
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);

  // Detect CUDA capability
  int device_count = 0;
  absl::Status cuda_status = tensorcast::cuda::get_device_count(&device_count);
  bool has_cuda = (cuda_status.ok() && device_count > 0);

  SECTION("GPU to GPU transfer") {
    if (!has_cuda) {
      WARN("CUDA not available - skipping GPU to GPU test");
      return;
    }

    P2PTestConfig config;
    {
      int port = find_available_port(50060);
      REQUIRE(port > 0);
      config.server_port = port;
    }
    config.artifact_id = "p2p_gpu_to_gpu_test";
    config.gpu_id = 0;
    config.artifact_size_mb = 16;
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

  SECTION("Small replica transfer") {
    if (!has_cuda) {
      WARN("CUDA not available - skipping small replica GPU test");
      return;
    }

    P2PTestConfig config;
    {
      int port = find_available_port(50070);
      REQUIRE(port > 0);
      config.server_port = port;
    }
    config.artifact_id = "p2p_small_artifact_test";
    config.gpu_id = 0;
    config.artifact_size_mb = 1; // 1MB replica
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

  SECTION("Large replica transfer") {
    if (!has_cuda) {
      WARN("CUDA not available - skipping large replica GPU test");
      return;
    }

    P2PTestConfig config;
    {
      int port = find_available_port(50100);
      REQUIRE(port > 0);
      config.server_port = port;
    }
    config.artifact_id = "p2p_large_artifact_test";
    config.gpu_id = 0;
    config.artifact_size_mb = 64; // 64MB replica
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
