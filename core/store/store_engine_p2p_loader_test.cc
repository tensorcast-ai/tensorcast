// Copyright (c) 2025, StepCast Team. All rights reserved.

// New test for StoreEngine using P2PLoader over TCP.

#include "core/common/cuda_api.h"

#include <chrono>
#include <filesystem>
#include <system_error>
#include <vector>

#define private public
#define protected public
#include "core/store/store_engine.h" // must be included with macros to access internal P2P loader
#undef private
#undef protected

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"

#include "core/communicator/engine/engine.h"
#include "core/store/components/communication_manager.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/store_engine_options.h"
#include "core/testing/test_helpers.h"

namespace fs = std::filesystem;
using namespace stepcast::store;
using namespace stepcast::communicator;
using namespace stepcast::communicator::test;

TEST_CASE("StoreEngine P2P Loader TCP end-to-end", "[store_engine][p2p][tcp][gpu]") {
  // ---------------------------------------------------------------------------
  // 0. Environment pre-checks.
  // ---------------------------------------------------------------------------
  SKIP_IF_NO_CUDA();

  // RDMA is explicitly disabled via constructor parameter in CommunicateEngine and StoreEngine.

  const std::size_t artifact_size = 8 * 1024 * 1024; // 8 MiB

  // ---------------------------------------------------------------------------
  // 1. Spin up a standalone CommunicateEngine that owns the remote tensor.
  // ---------------------------------------------------------------------------
  int src_port = find_available_port(51000);
  REQUIRE(src_port > 0);

  auto src_engine = std::make_shared<CommunicateEngine>(false /* enable_rdma = false */);
  REQUIRE(src_engine->init("127.0.0.1", static_cast<uint16_t>(src_port)).ok());

  // Allocate GPU memory and fill with a deterministic pattern.
  void* src_gpu_ptr = nullptr;
  REQUIRE(stepcast::cuda::malloc(&src_gpu_ptr, artifact_size).ok());
  std::vector<uint8_t> src_pattern = create_test_pattern(artifact_size, /*seed=*/42);
  REQUIRE(stepcast::cuda::memcpy(src_gpu_ptr, src_pattern.data(), artifact_size, cudaMemcpyHostToDevice).ok());

  // Register the GPU buffer so that it can be fetched remotely.
  const char* kRemoteKey = "remote_model_weights";
  REQUIRE(src_engine
              ->register_tensor(
                  kRemoteKey,
                  reinterpret_cast<uint64_t>(src_gpu_ptr),
                  artifact_size,
                  COMMUNICATE_ENGINE_DEV_GPU,
                  /*dev_id=*/0,
                  /*async=*/false)
              .ok());

  // ---------------------------------------------------------------------------
  // 2. Instantiate the target StoreEngine with communication enabled.
  // ---------------------------------------------------------------------------
  // TransferService creates a per-session StreamingPinnedBuffer (16 chunks).
  // Store no longer pre-allocates a shared buffer, so 16 * chunk_size is sufficient.
  const std::size_t pool_size = 128 * 1024 * 1024; // 128 MiB pinned pool to satisfy 16 x 8 MiB chunks
  const std::size_t chunk_size = 8 * 1024 * 1024; // 8 MiB to match remote buffer size
  const int io_threads = 2;

  // storage_path is unused for remote loads; give a temporary directory.
  fs::path temp_root = fs::temp_directory_path() / "store_engine_p2p_test";
  fs::create_directories(temp_root);

  // ---------------------------------------------------------------------------
  // 2.a Create and initialise a CommunicationManager for the target store.
  // ---------------------------------------------------------------------------
  int comm_port = find_available_port(9090);
  REQUIRE(comm_port > 0);
  auto comm_manager = std::make_shared<stepcast::store::CommunicationManager>();
  REQUIRE(comm_manager->initialize("127.0.0.1", /*listen_port=*/comm_port, /*enable_rdma=*/false).ok());

  // ---------------------------------------------------------------------------
  // 2.b Construct StoreEngine using the new Options struct.
  // ---------------------------------------------------------------------------
  stepcast::store::StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = pool_size;
  opts.num_thread = io_threads;
  opts.chunk_size = chunk_size;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  opts.p2p_port = comm_port;
  opts.comm_manager = comm_manager;

  StoreEngine tgt_store(opts);

  // ---------------------------------------------------------------------------
  // 3. Build a LoadSpec describing the remote GPU source and local GPU target.
  // ---------------------------------------------------------------------------
  ReplicaLoadSpec spec;
  spec.identifier = "remote_artifact";

  // Configure P2P source
  P2PSource p2p_src;
  p2p_src.size_bytes = artifact_size;
  p2p_src.ip = "127.0.0.1";
  p2p_src.port = static_cast<uint16_t>(src_port);
  p2p_src.memory_keys = {kRemoteKey};
  p2p_src.buf_sizes = {artifact_size};
  p2p_src.location.type = MemoryLocation::GPU; // Remote data is on GPU
  p2p_src.location.device_id = 0;
  p2p_src.enable_checksum = true;
  spec.source = p2p_src;

  // Configure target
  ReplicaTarget target;
  target.location.type = MemoryLocation::GPU;
  target.location.device_id = 0;
  spec.target = target;

  // ---------------------------------------------------------------------------
  // 4. Issue P2P load via internal helper and wait for completion.
  // ---------------------------------------------------------------------------
  absl::StatusOr<ReplicaHandle> handle_or =
      tgt_store.ingest_from_p2p_internal("remote_artifact", p2p_src, target, spec.hints);
  REQUIRE(handle_or.ok());
  auto& handle = handle_or.value();
  REQUIRE(handle.ready_future.valid());
  REQUIRE(handle.ready_future.get().ok());

  // ---------------------------------------------------------------------------
  // 5. Validate that the replica resides on GPU and the contents match.
  // ---------------------------------------------------------------------------
  auto gpu_ptr_result = tgt_store.get_replica_gpu_ptr(handle.replica_key);
  REQUIRE(gpu_ptr_result.ok());
  REQUIRE(gpu_ptr_result.value() != 0);

  void* tgt_gpu_ptr = reinterpret_cast<void*>(gpu_ptr_result.value());
  std::vector<uint8_t> verify_buf(artifact_size);
  REQUIRE(stepcast::cuda::memcpy(verify_buf.data(), tgt_gpu_ptr, artifact_size, cudaMemcpyDeviceToHost).ok());
  REQUIRE(verify_pattern(verify_buf.data(), artifact_size, /*seed=*/42));

  // ---------------------------------------------------------------------------
  // 6. Clean-up GPU memory and stores.
  // ---------------------------------------------------------------------------
  REQUIRE(tgt_store.clear_mem() == 0);
  absl::Status free_status = stepcast::cuda::free(src_gpu_ptr);
  (void)free_status; // Ignore status in cleanup

  // Remove temp directory (best effort).
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}
