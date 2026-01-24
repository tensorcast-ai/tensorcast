// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <nlohmann/json.hpp>
#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>
#include "core/cuda/cuda_api.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "daemon/testing/cuda_ipc_spawn_helper.h"
#include "grpcpp/server_context.h"

namespace {

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = std::filesystem::temp_directory_path() / "tensorcast_daemon_lip_ttl_test";
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47003;
  opts.memory_pool_size = 64ULL << 20; // 64 MiB
  opts.tx_slice_bytes = 1ULL << 20; // 1 MiB
  opts.num_thread = 2;
  return opts;
}

std::unique_ptr<tensorcast::daemon::DaemonServiceHarness> make_harness(
    const std::shared_ptr<tensorcast::store::StoreEngine>& engine) {
  tensorcast::daemon::DaemonOptions options;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, options);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

} // namespace

TEST_CASE("LIP TTL expiry gates P2P source selection", "[daemon][lip][ttl][fakecuda]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness = make_harness(engine);
  auto& svc = harness->service();

  // Minimal canonical index
  nlohmann::json j = nlohmann::json::object();
  const std::string index_bytes = j.dump();

  const auto helper_path_or = tensorcast::daemon::testing::resolve_cuda_ipc_helper_path();
  REQUIRE(helper_path_or.ok());
  std::vector<tensorcast::daemon::testing::CudaIpcBufferSpec> buffers = {
      {.size_bytes = 1 * 1024 * 1024, .fill_byte = -1},
  };
  auto child_or = tensorcast::daemon::testing::CudaIpcChild::Spawn(*helper_path_or, 0, buffers);
  REQUIRE(child_or.ok());
  auto child = std::move(*child_or);
  REQUIRE(child.handle_bytes().size() == 1);
  const std::string& handle_bytes = child.handle_bytes().front();

  // Begin lease registration with small TTL and in_place=true
  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(1 * 1024 * 1024); // 1 MiB
  breq.set_ttl_ms(100); // 100 ms TTL
  breq.set_owner_pid(child.pid());
  auto* ti = breq.mutable_tensor_index_data();
  ti->set_data(index_bytes);
  ti->set_schema_version("v3");
  ti->set_encoding("json");
  breq.mutable_lease()->set_in_place(true);

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto st = svc.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(st.ok());

  // Allocate and feed a single segment covering the artifact
  REQUIRE(tensorcast::cuda::is_available());
  tensorcast::daemon::v2::FeedRegisterArtifactStreamRequest freq;
  freq.set_registration_id(bresp.registration_id());
  auto* ls = freq.mutable_lease_segments();
  auto* s = ls->add_segments();
  auto* storage = freq.add_storage_entries();
  storage->set_storage_id("s0");
  storage->set_device_id(0);
  storage->set_storage_length(1 * 1024 * 1024);
  storage->set_cuda_ipc_handle(handle_bytes);
  storage->set_mapping_base_offset(0);
  auto* alias = freq.add_tensor_aliases();
  alias->set_name("tensor");
  alias->set_storage_id("s0");
  alias->set_storage_offset(0);
  alias->set_logical_length(1 * 1024 * 1024);
  alias->add_shape(1 * 1024 * 1024);
  alias->add_stride(1);
  alias->set_dtype("torch.uint8");
  s->set_storage_id("s0");
  s->set_storage_offset(0);
  s->set_length(1 * 1024 * 1024);
  s->set_artifact_offset(0);
  st = svc.feed_register_artifact_stream_vector({freq});
  REQUIRE(st.ok());

  tensorcast::daemon::v2::CommitRegisteredArtifactRequest creq;
  creq.set_registration_id(bresp.registration_id());
  tensorcast::daemon::v2::CommitRegisteredArtifactResponse cresp;
  st = svc.CommitRegisteredArtifact(&ctx, &creq, &cresp);
  REQUIRE(st.ok());
  const std::string artifact_id = cresp.artifact_descriptor().artifact_id();

  // Wait past TTL to trigger expiry sweep
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  // Attempt to lock a chunk for P2P; should fail with DEADLINE_EXCEEDED due to expired lease
  tensorcast::daemon::v2::LockTransportChunksRequest lreq;
  lreq.set_artifact_id(artifact_id);
  lreq.add_chunk_indices(0);
  tensorcast::daemon::v2::LockTransportChunksResponse lresp;
  st = svc.LockTransportChunks(&ctx, &lreq, &lresp);
  // Allow either DEADLINE_EXCEEDED (explicit TTL gating) or NOT_FOUND (lease swept).
  REQUIRE((st.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED || st.error_code() == grpc::StatusCode::NOT_FOUND));
}
