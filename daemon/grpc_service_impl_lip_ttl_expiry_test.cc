// Copyright (c) 2025, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <nlohmann/json.hpp>
#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>
#include "core/common/cuda_api.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "grpcpp/server_context.h"

using tensorcast::daemon::StoreDaemonServiceImpl;

namespace {

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = std::filesystem::temp_directory_path() / "tensorcast_daemon_lip_ttl_test";
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 0;
  opts.memory_pool_size = 64ULL << 20; // 64 MiB
  opts.chunk_size = 1ULL << 20; // 1 MiB
  opts.num_thread = 2;
  return opts;
}

} // namespace

TEST_CASE("LIP TTL expiry gates P2P source selection", "[daemon][lip][ttl][fakecuda]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  StoreDaemonServiceImpl svc(engine);

  // Minimal canonical index
  nlohmann::json j = nlohmann::json::object();
  const std::string index_bytes = j.dump();

  // Begin lease registration with small TTL and in_place=true
  tensorcast::daemon::v1::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(1 * 1024 * 1024); // 1 MiB
  breq.set_ttl_ms(100); // 100 ms TTL
  breq.set_owner_pid(getpid());
  auto* ti = breq.mutable_tensor_index_data();
  ti->set_data(index_bytes);
  ti->set_schema_version("v2");
  ti->set_encoding("json");
  breq.mutable_lease()->set_in_place(true);

  grpc::ServerContext ctx;
  tensorcast::daemon::v1::BeginRegisterArtifactResponse bresp;
  auto st = svc.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(st.ok());

  // Allocate and feed a single segment covering the artifact
  REQUIRE(tensorcast::cuda::is_available());
  void* p = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&p, 1 * 1024 * 1024).ok());
  cudaIpcMemHandle_t h{};
  REQUIRE(tensorcast::cuda::get_ipc_mem_handle(&h, p).ok());

  tensorcast::daemon::v1::FeedRegisterArtifactStreamRequest freq;
  freq.set_registration_id(bresp.registration_id());
  auto* ls = freq.mutable_lease_segments();
  auto* s = ls->add_segments();
  s->set_device_id(0);
  s->set_length(1 * 1024 * 1024);
  s->set_base_addr(0);
  s->set_dst_offset(0);
  s->set_cuda_ipc_handle(std::string(reinterpret_cast<const char*>(&h), sizeof(h)));
  st = svc.feed_register_artifact_stream_vector({freq});
  REQUIRE(st.ok());

  tensorcast::daemon::v1::CommitRegisteredArtifactRequest creq;
  creq.set_registration_id(bresp.registration_id());
  tensorcast::daemon::v1::CommitRegisteredArtifactResponse cresp;
  st = svc.CommitRegisteredArtifact(&ctx, &creq, &cresp);
  REQUIRE(st.ok());
  const std::string artifact_id = cresp.artifact_descriptor().artifact_id();

  // Wait past TTL to trigger expiry sweep
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  // Attempt to lock a chunk for P2P; should fail with DEADLINE_EXCEEDED due to expired lease
  tensorcast::daemon::v1::LockTransportChunksRequest lreq;
  lreq.set_artifact_id(artifact_id);
  lreq.add_chunk_indices(0);
  tensorcast::daemon::v1::LockTransportChunksResponse lresp;
  st = svc.LockTransportChunks(&ctx, &lreq, &lresp);
  // Allow either DEADLINE_EXCEEDED (explicit TTL gating) or NOT_FOUND (lease swept).
  REQUIRE((st.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED || st.error_code() == grpc::StatusCode::NOT_FOUND));
}
