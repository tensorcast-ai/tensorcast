// Copyright (c) 2025, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <string>

#include <catch2/catch_test_macros.hpp>
#include "core/common/cuda_api.h"
#include "core/store/loader/source_hash.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "grpcpp/server_context.h"
#include "gsl/pointers"
#include "nlohmann/json.hpp"

using tensorcast::daemon::StoreDaemonServiceImpl;

namespace {

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = std::filesystem::temp_directory_path() / "tensorcast_daemon_lease_test";
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47002;
  opts.memory_pool_size = 64ull << 20; // 64 MiB
  opts.tx_slice_bytes = 1ull << 20; // 1 MiB
  opts.num_thread = 2;
  return opts;
}

} // namespace

TEST_CASE("Lease commit places segments by dst_offset and zeros PAD", "[daemon][lease][fakecuda]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  StoreDaemonServiceImpl svc(engine);

  // Build canonical index with two DATA regions and a PAD gap of 16 bytes
  // Regions: [0,16] and [32,16]; total_size = 48 (no trailing PAD)
  nlohmann::json j;
  j["a"] = {0, 16, std::vector<int>{16}, std::vector<int>{1}, "torch.uint8", 0};
  j["b"] = {32, 16, std::vector<int>{16}, std::vector<int>{1}, "torch.uint8", 0};
  const std::string index_bytes = j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);

  // Begin lease registration
  tensorcast::daemon::v1::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(48);
  breq.set_owner_pid(getpid());
  breq.mutable_tensor_index_data()->set_data(index_bytes);
  breq.mutable_tensor_index_data()->set_schema_version("v2");
  breq.mutable_tensor_index_data()->set_encoding("json");
  (void)breq.mutable_lease();

  grpc::ServerContext ctx;
  tensorcast::daemon::v1::BeginRegisterArtifactResponse bresp;
  auto st = svc.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(st.ok());
  REQUIRE(bresp.has_lease());

  // Allocate two device buffers, fill distinct patterns, export IPC handles
  REQUIRE(tensorcast::cuda::is_available());
  void* p1 = nullptr;
  void* p2 = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&p1, 16).ok());
  REQUIRE(tensorcast::cuda::malloc(&p2, 16).ok());
  REQUIRE(tensorcast::cuda::memset(p1, 0x11, 16).ok());
  REQUIRE(tensorcast::cuda::memset(p2, 0x22, 16).ok());
  cudaIpcMemHandle_t h1{};
  cudaIpcMemHandle_t h2{};
  REQUIRE(tensorcast::cuda::get_ipc_mem_handle(&h1, p1).ok());
  REQUIRE(tensorcast::cuda::get_ipc_mem_handle(&h2, p2).ok());

  // Feed in reverse order to validate order independence; include explicit dst_offset
  tensorcast::daemon::v1::FeedRegisterArtifactStreamRequest freq;
  freq.set_registration_id(bresp.registration_id());
  auto* ls = freq.mutable_lease_segments();
  auto* s2 = ls->add_segments();
  s2->set_device_id(0);
  s2->set_length(16);
  s2->set_base_addr(0);
  s2->set_cuda_ipc_handle(std::string(reinterpret_cast<const char*>(&h2), sizeof(cudaIpcMemHandle_t)));
  s2->set_dst_offset(32);
  auto* s1 = ls->add_segments();
  s1->set_device_id(0);
  s1->set_length(16);
  s1->set_base_addr(0);
  s1->set_cuda_ipc_handle(std::string(reinterpret_cast<const char*>(&h1), sizeof(cudaIpcMemHandle_t)));
  s1->set_dst_offset(0);

  // Use helper to feed streaming vector without spinning up gRPC server
  std::vector<tensorcast::daemon::v1::FeedRegisterArtifactStreamRequest> reqs;
  reqs.push_back(freq);
  st = svc.feed_register_artifact_stream_vector(reqs);
  REQUIRE(st.ok());

  // Commit
  tensorcast::daemon::v1::CommitRegisteredArtifactRequest creq;
  creq.set_registration_id(bresp.registration_id());
  tensorcast::daemon::v1::CommitRegisteredArtifactResponse cresp;
  st = svc.CommitRegisteredArtifact(&ctx, &creq, &cresp);
  INFO("Commit status: " << st.error_message());
  REQUIRE(st.ok());
  REQUIRE(cresp.has_artifact_descriptor());
  const auto& desc = cresp.artifact_descriptor();
  REQUIRE(desc.total_size() == 48);

  // Compute expected multihash from CPU memory: [0..15]=0x11, [16..31]=0, [32..47]=0x22
  std::array<uint8_t, 48> expected{};
  std::memset(expected.data(), 0, expected.size());
  std::memset(expected.data() + 0, 0x11, 16);
  std::memset(expected.data() + 32, 0x22, 16);
  auto mh_or = tensorcast::store::loader::compute_data_multihash_from_cpu_memory(
      gsl::not_null<const void*>{expected.data()}, expected.size());
  REQUIRE(mh_or.ok());
  REQUIRE(desc.data_multihash() == *mh_or);
}
