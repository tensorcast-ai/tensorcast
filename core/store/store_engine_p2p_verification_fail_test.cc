// Copyright (c) 2025, TensorCast Team.

#include "core/common/cuda_api.h"

#include <vector>

#define private public
#define protected public
#include "core/store/store_engine.h"
#undef private
#undef protected

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"

#include "core/common/artifact_verification.h"
#include "core/communicator/engine/engine.h"
#include "core/store/components/communication_manager.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/store_engine_options.h"
#include "core/testing/test_helpers.h"

using tensorcast::common::ArtifactVerifier;
using tensorcast::common::VerificationLevel;
using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
using tensorcast::communicator::engine::Communicator;
using tensorcast::communicator::v1::CommunicatorConfig;
using tensorcast::store::P2PSource;
using tensorcast::store::StoreEngine;
using tensorcast::store::components::CommunicationManager;
using tensorcast::store::loading::ReplicaHandle;
using tensorcast::store::loading::ReplicaLoadSpec;
using tensorcast::store::loading::ReplicaTarget;

namespace {

static std::vector<uint8_t> make_pattern(size_t n, uint8_t seed) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = static_cast<uint8_t>(seed + (i % 251));
  return v;
}

} // namespace

TEST_CASE("P2P receiver verification fails with mismatched verification_json", "[store_engine][p2p][verify][gpu]") {
  SKIP_IF_NO_CUDA();

  const std::size_t artifact_size = 4 * 1024 * 1024; // 4 MiB

  // 1) Start a minimal Communicator exposing a GPU tensor
  int src_port = tensorcast::testing::find_available_port(52000);
  REQUIRE(src_port > 0);

  CommunicatorConfig cfg;
  cfg.set_enable_rdma(false);
  auto src_engine = std::make_shared<Communicator>(cfg);
  REQUIRE(src_engine->init("127.0.0.1", static_cast<uint16_t>(src_port)).ok());

  void* src_gpu_ptr = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&src_gpu_ptr, artifact_size).ok());
  auto pattern = make_pattern(artifact_size, /*seed=*/42);
  REQUIRE(tensorcast::cuda::memcpy(src_gpu_ptr, pattern.data(), artifact_size, cudaMemcpyHostToDevice).ok());

  const char* kRemoteKey = "remote_model_weights_badverif";
  Communicator::RegisterTensorOptions reg_opts;
  reg_opts.register_mr = false;
  reg_opts.needs_staging = true;
  reg_opts.async = false;
  REQUIRE(src_engine
              ->register_tensor_ex(
                  kRemoteKey,
                  reinterpret_cast<uint64_t>(src_gpu_ptr),
                  artifact_size,
                  COMMUNICATE_ENGINE_DEV_GPU,
                  /*dev_id=*/0,
                  reg_opts)
              .ok());

  // 2) Build a mismatched verification_json (from a different CPU buffer)
  std::string bad_ver_json;
  {
    auto wrong = make_pattern(artifact_size, /*seed=*/7);
    std::vector<void*> ptrs{wrong.data()};
    std::vector<size_t> sizes{artifact_size};
    auto info_or =
        ArtifactVerifier::generate_verification_info(ptrs, sizes, /*device_id=*/-1, VerificationLevel::KEY_POINTS);
    REQUIRE(info_or.ok());
    bad_ver_json = info_or->to_json();
  }

  // 3) Prepare a target StoreEngine with communication enabled
  int comm_port = tensorcast::testing::find_available_port(9091);
  REQUIRE(comm_port > 0);
  auto comm_manager = std::make_shared<CommunicationManager>();
  REQUIRE(comm_manager->initialize("127.0.0.1", /*listen_port=*/comm_port, /*enable_rdma=*/false).ok());

  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (std::filesystem::temp_directory_path() / "store_engine_p2p_verify_fail").string();
  opts.memory_pool_size = 64ull * 1024 * 1024;
  opts.num_thread = 2;
  opts.chunk_size = 2ull * 1024 * 1024; // 2 MiB
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  opts.p2p_port = comm_port;
  opts.comm_manager = comm_manager;

  StoreEngine tgt_store(opts);

  // 4) Build P2P source with bad verification
  P2PSource p2p_src;
  p2p_src.size_bytes = artifact_size;
  p2p_src.ip = "127.0.0.1";
  p2p_src.port = static_cast<uint16_t>(src_port);
  p2p_src.memory_keys = {kRemoteKey};
  p2p_src.buf_sizes = {artifact_size};
  p2p_src.location.type = tensorcast::common::memory::MemoryLocation::GPU;
  p2p_src.location.device_id = 0;
  p2p_src.enable_checksum = true;
  p2p_src.verification_json = bad_ver_json;

  // 5) Attempt P2P ingest (GPU target) and expect DataLoss due to verification mismatch
  tensorcast::store::loading::ReplicaTarget target;
  target.location.type = tensorcast::common::memory::MemoryLocation::GPU;
  target.location.device_id = 0;

  auto handle_or = tgt_store.ingest_from_p2p_internal("badverify_artifact", p2p_src, target, {});
  REQUIRE(!handle_or.ok());
  REQUIRE(handle_or.status().code() == absl::StatusCode::kDataLoss);

  // Cleanup GPU memory
  (void)tensorcast::cuda::free(src_gpu_ptr);
}
