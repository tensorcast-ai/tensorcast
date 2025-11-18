// Copyright (c) 2025, TensorCast Team.

#include <filesystem>
#include <system_error>
#include <vector>

#include "absl/log/check.h"
#include "catch2/catch_test_macros.hpp"
#include "core/common/artifact_verification.h"
#include "core/common/cuda_api.h"
#include "core/communicator/engine/engine.h"
#include "core/store/components/worker_identity.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/runtime/pipeline/ingestion_pipeline.h"
#include "core/store/runtime/component_catalog.h"
#include "core/store/runtime/replica_runtime.h"
#include "core/store/runtime/runtime_event_hub.h"
#include "core/store/store_engine_options.h"
#include "core/testing/test_helpers.h"

namespace fs = std::filesystem;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
using tensorcast::communicator::engine::Communicator;
using tensorcast::store::P2PSource;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::components::CommunicationManager;
using tensorcast::store::loading::ReplicaTarget;
using tensorcast::store::runtime::ComponentCatalog;
using tensorcast::store::runtime::ReplicaRuntime;
using tensorcast::testing::create_test_pattern;
using tensorcast::testing::find_available_port;
using tensorcast::testing::make_tcp_communicator_config;
using tensorcast::testing::verify_pattern;

namespace {

struct PipelineHarness {
  ComponentCatalog catalog;
  tensorcast::store::runtime::RuntimeEventHub event_hub;
  std::unique_ptr<ReplicaRuntime> replica_runtime;
  std::unique_ptr<tensorcast::store::materialization::runtime::pipeline::IngestionPipeline> pipeline;

  explicit PipelineHarness(const StoreEngineOptions& opts)
      : catalog(opts),
        replica_runtime(
            std::make_unique<ReplicaRuntime>(
                ReplicaRuntime::Config{.component_catalog = &catalog, .event_hub = &event_hub})) {
    CHECK_OK(catalog.start());
    tensorcast::store::materialization::runtime::pipeline::IngestionPipeline::Config cfg{
        .storage_path = fs::path(opts.storage_path),
        .num_threads = opts.num_thread,
        .artifact_chunk_bytes = opts.artifact_chunk_bytes,
        .pinned_memory_timeout = opts.pinned_memory_timeout,
        .engine_options = &opts,
        .replica_runtime = replica_runtime.get(),
        .component_catalog = &catalog,
        .metadata_gateway = nullptr,
        .event_hub = &event_hub,
    };
    pipeline = std::make_unique<tensorcast::store::materialization::runtime::pipeline::IngestionPipeline>(cfg);
  }

  ~PipelineHarness() {
    if (replica_runtime) {
      replica_runtime->clear_mem();
    }
    catalog.shutdown();
  }
};

} // namespace

TEST_CASE("IngestionPipeline P2P Loader TCP end-to-end", "[pipeline][p2p][gpu]") {
  SKIP_IF_NO_CUDA();

  const std::size_t artifact_size = 8 * 1024 * 1024;
  const char* kRemoteKey = "remote_model_weights";
  fs::path temp_root = fs::temp_directory_path() / "pipeline_p2p_test";
  fs::create_directories(temp_root);

  const int src_port = find_available_port(52000);
  REQUIRE(src_port > 0);
  auto cfg = make_tcp_communicator_config();
  auto src_engine = std::make_shared<Communicator>(cfg);
  REQUIRE(src_engine->init("127.0.0.1", static_cast<uint16_t>(src_port)).ok());

  void* src_gpu_ptr = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&src_gpu_ptr, artifact_size).ok());
  std::vector<uint8_t> src_pattern = create_test_pattern(artifact_size, /*seed=*/123);
  REQUIRE(tensorcast::cuda::memcpy(src_gpu_ptr, src_pattern.data(), artifact_size, cudaMemcpyHostToDevice).ok());

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

  std::string verification_json;
  {
    std::vector<void*> ptrs{src_gpu_ptr};
    std::vector<size_t> sizes{artifact_size};
    auto info_or = tensorcast::common::ArtifactVerifier::generate_verification_info(
        ptrs, sizes, /*device_id=*/0, tensorcast::common::VerificationLevel::KEY_POINTS);
    REQUIRE(info_or.ok());
    verification_json = info_or->to_json();
  }

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 128 * 1024 * 1024;
  opts.num_thread = 2;
  opts.tx_slice_bytes = artifact_size;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  opts.artifact_chunk_bytes = artifact_size;
  const int comm_port = find_available_port(9300);
  REQUIRE(comm_port > 0);
  auto comm_manager = std::make_shared<CommunicationManager>();
  REQUIRE(comm_manager->initialize("127.0.0.1", /*listen_port=*/comm_port, /*enable_rdma=*/false).ok());
  opts.comm_manager = comm_manager;
  PipelineHarness harness(opts);

  ReplicaTarget target;
  target.location.type = MemoryLocation::GPU;
  target.location.device_id = 0;

  P2PSource p2p_source;
  p2p_source.size_bytes = artifact_size;
  p2p_source.ip = "127.0.0.1";
  p2p_source.port = static_cast<uint16_t>(src_port);
  p2p_source.memory_keys = {kRemoteKey};
  p2p_source.buf_sizes = {artifact_size};
  p2p_source.location.type = MemoryLocation::GPU;
  p2p_source.location.device_id = 0;
  p2p_source.enable_checksum = true;
  p2p_source.verification_json = verification_json;

  auto handle_or = harness.pipeline->ingest_from_p2p("remote_artifact", p2p_source, target, {});
  REQUIRE(handle_or.ok());
  auto& handle = handle_or.value();
  REQUIRE(handle.ready_future.valid());
  REQUIRE(handle.ready_future.get().ok());

  auto gpu_ptr_result = harness.replica_runtime->get_replica_gpu_ptr(handle.replica_key);
  REQUIRE(gpu_ptr_result.ok());
  void* gpu_ptr = reinterpret_cast<void*>(gpu_ptr_result.value());
  std::vector<uint8_t> verify_buf(artifact_size);
  REQUIRE(tensorcast::cuda::memcpy(verify_buf.data(), gpu_ptr, artifact_size, cudaMemcpyDeviceToHost).ok());
  REQUIRE(verify_pattern(verify_buf.data(), artifact_size, /*seed=*/123));

  tensorcast::cuda::free(src_gpu_ptr).IgnoreError();
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}
