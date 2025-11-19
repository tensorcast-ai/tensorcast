// Copyright (c) 2025, TensorCast Team.

#include <filesystem>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"

#include "core/common/artifact_verification.h"
#include "core/common/cuda_api.h"
#include "core/communicator/engine/engine.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/runtime/pipeline/ingestion_pipeline.h"
#include "core/store/runtime/component_catalog.h"
#include "core/store/runtime/replica_runtime.h"
#include "core/store/runtime/runtime_event_hub.h"
#include "core/store/store_engine_options.h"
#include "core/testing/test_helpers.h"

namespace fs = std::filesystem;

using tensorcast::common::ArtifactVerifier;
using tensorcast::common::VerificationLevel;
using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
using tensorcast::communicator::engine::Communicator;
using tensorcast::store::P2PSource;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::components::CommunicationManager;
using tensorcast::store::loading::ReplicaTarget;
using tensorcast::store::runtime::ComponentCatalog;
using tensorcast::store::runtime::ReplicaRuntime;
using tensorcast::testing::find_available_port;
using tensorcast::testing::make_tcp_communicator_config;

namespace {

std::vector<uint8_t> MakePattern(size_t n, uint8_t seed) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = static_cast<uint8_t>(seed + (i % 251));
  }
  return v;
}

std::unique_ptr<tensorcast::store::materialization::runtime::pipeline::IngestionPipeline> MakePipeline(
    const StoreEngineOptions& opts,
    ComponentCatalog& catalog,
    ReplicaRuntime& runtime,
    tensorcast::store::runtime::RuntimeEventHub* event_hub) {
  tensorcast::store::materialization::runtime::pipeline::IngestionPipeline::Config cfg{
      .storage_path = fs::path(opts.storage_path),
      .num_threads = opts.num_thread,
      .artifact_chunk_bytes = opts.artifact_chunk_bytes,
      .pinned_memory_timeout = opts.pinned_memory_timeout,
      .engine_options = &opts,
      .replica_runtime = &runtime,
      .component_catalog = &catalog,
      .metadata_gateway = nullptr,
      .event_hub = event_hub,
  };
  return std::make_unique<tensorcast::store::materialization::runtime::pipeline::IngestionPipeline>(cfg);
}

} // namespace

TEST_CASE("P2P ingest fails when verification metadata mismatches", "[pipeline][p2p][verify][gpu]") {
  SKIP_IF_NO_CUDA();
  const std::size_t artifact_size = 4 * 1024 * 1024;
  const char* kRemoteKey = "remote_model_weights_badverif";

  int src_port = find_available_port(54000);
  REQUIRE(src_port > 0);
  auto cfg = make_tcp_communicator_config();
  auto src_engine = std::make_shared<Communicator>(cfg);
  REQUIRE(src_engine->init("127.0.0.1", static_cast<uint16_t>(src_port)).ok());

  void* src_gpu_ptr = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&src_gpu_ptr, artifact_size).ok());
  auto pattern = MakePattern(artifact_size, /*seed=*/42);
  REQUIRE(tensorcast::cuda::memcpy(src_gpu_ptr, pattern.data(), artifact_size, cudaMemcpyHostToDevice).ok());

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

  std::string bad_ver_json;
  {
    auto wrong = MakePattern(artifact_size, /*seed=*/7);
    std::vector<void*> ptrs{wrong.data()};
    std::vector<size_t> sizes{artifact_size};
    auto info_or =
        ArtifactVerifier::generate_verification_info(ptrs, sizes, /*device_id=*/-1, VerificationLevel::KEY_POINTS);
    REQUIRE(info_or.ok());
    bad_ver_json = info_or->to_json();
  }

  int comm_port = find_available_port(9091);
  REQUIRE(comm_port > 0);
  auto comm_manager = std::make_shared<CommunicationManager>();
  REQUIRE(comm_manager->initialize("127.0.0.1", /*listen_port=*/comm_port, /*enable_rdma=*/false).ok());

  StoreEngineOptions opts;
  opts.storage_path = (std::filesystem::temp_directory_path() / "pipeline_p2p_verify_fail").string();
  opts.memory_pool_size = 64ull * 1024 * 1024;
  opts.num_thread = 2;
  opts.tx_slice_bytes = 2ull * 1024 * 1024;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  opts.p2p_port = comm_port;
  opts.comm_manager = comm_manager;

  ComponentCatalog catalog(opts);
  CHECK_OK(catalog.start());
  tensorcast::store::runtime::RuntimeEventHub event_hub;
  ReplicaRuntime runtime(ReplicaRuntime::Config{.component_catalog = &catalog, .event_hub = &event_hub});
  auto pipeline = MakePipeline(opts, catalog, runtime, &event_hub);

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

  ReplicaTarget target;
  target.location.type = tensorcast::common::memory::MemoryLocation::GPU;
  target.location.device_id = 0;

  auto handle_or = pipeline->ingest_from_p2p("badverify_artifact", p2p_src, target, {});
  REQUIRE(!handle_or.ok());
  REQUIRE(handle_or.status().code() == absl::StatusCode::kDataLoss);

  runtime.clear_mem();
  catalog.shutdown();
  tensorcast::cuda::free(src_gpu_ptr).IgnoreError();
}
