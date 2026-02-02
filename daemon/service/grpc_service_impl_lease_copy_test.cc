// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <string>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include "core/cuda/cuda_api.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "daemon/testing/cuda_ipc_spawn_helper.h"
#include "grpcpp/server_context.h"
#include "gsl/pointers"
#include "nlohmann/json.hpp"

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

TEST_CASE("Lease commit places segments by dst_offset and zeros PAD", "[daemon][lease][fakecuda]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness = make_harness(engine);
  auto& svc = harness->service();

  // Build canonical index with two DATA regions and a PAD gap of 16 bytes
  // Regions: [0,16] and [32,16]; total_size = 48 (no trailing PAD)
  nlohmann::json index_json;
  index_json["a"] = {0, 16, std::vector<int>{16}, std::vector<int>{1}, "torch.uint8", 0};
  index_json["b"] = {32, 16, std::vector<int>{16}, std::vector<int>{1}, "torch.uint8", 0};
  const std::string index_bytes = index_json.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);

  const auto helper_path_or = tensorcast::daemon::testing::resolve_cuda_ipc_helper_path();
  REQUIRE(helper_path_or.ok());
  std::vector<tensorcast::daemon::testing::CudaIpcBufferSpec> buffers = {
      {.size_bytes = 16, .fill_byte = 0x11},
      {.size_bytes = 16, .fill_byte = 0x22},
  };
  auto child_or = tensorcast::daemon::testing::CudaIpcChild::Spawn(*helper_path_or, 0, buffers);
  INFO("cuda_ipc_helper spawn status: " << child_or.status());
  REQUIRE(child_or.ok());
  auto child = std::move(*child_or);
  REQUIRE(child.handle_bytes().size() == 2);
  const std::string& handle_bytes_first = child.handle_bytes()[0];
  const std::string& handle_bytes_second = child.handle_bytes()[1];

  // Begin lease registration
  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(48);
  breq.set_owner_pid(child.pid());
  breq.mutable_tensor_index_data()->set_data(index_bytes);
  breq.mutable_tensor_index_data()->set_schema_version("v3");
  breq.mutable_tensor_index_data()->set_encoding("json");
  breq.mutable_lease()->set_in_place(true);

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto st = svc.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(st.ok());
  REQUIRE(bresp.has_lease());

  // Ensure CUDA is available in the environment.
  REQUIRE(tensorcast::cuda::is_available());

  // Feed in reverse order to validate order independence; include explicit dst_offset
  tensorcast::daemon::v2::FeedRegisterArtifactStreamRequest freq;
  freq.set_registration_id(bresp.registration_id());
  // Storage metadata (deduplicated table).
  auto* storage2 = freq.add_storage_entries();
  storage2->set_storage_id("s2");
  storage2->set_device_id(0);
  storage2->set_storage_length(16);
  storage2->set_cuda_ipc_handle(handle_bytes_second);
  storage2->set_mapping_base_offset(0);
  auto* storage1 = freq.add_storage_entries();
  storage1->set_storage_id("s1");
  storage1->set_device_id(0);
  storage1->set_storage_length(16);
  storage1->set_cuda_ipc_handle(handle_bytes_first);
  storage1->set_mapping_base_offset(0);
  // Tensor alias metadata (required for LIP commits).
  auto* alias_a = freq.add_tensor_aliases();
  alias_a->set_name("a");
  alias_a->set_storage_id("s1");
  alias_a->set_storage_offset(0);
  alias_a->set_logical_length(16);
  alias_a->add_shape(16);
  alias_a->add_stride(1);
  alias_a->set_dtype("torch.uint8");
  auto* alias_b = freq.add_tensor_aliases();
  alias_b->set_name("b");
  alias_b->set_storage_id("s2");
  alias_b->set_storage_offset(0);
  alias_b->set_logical_length(16);
  alias_b->add_shape(16);
  alias_b->add_stride(1);
  alias_b->set_dtype("torch.uint8");

  auto* ls = freq.mutable_lease_segments();
  auto* segment_second = ls->add_segments();
  segment_second->set_storage_id("s2");
  segment_second->set_storage_offset(0);
  segment_second->set_length(16);
  segment_second->set_artifact_offset(32);
  auto* segment_first = ls->add_segments();
  segment_first->set_storage_id("s1");
  segment_first->set_storage_offset(0);
  segment_first->set_length(16);
  segment_first->set_artifact_offset(0);

  // Use helper to feed streaming vector without spinning up gRPC server
  std::vector<tensorcast::daemon::v2::FeedRegisterArtifactStreamRequest> reqs;
  reqs.push_back(freq);
  st = svc.feed_register_artifact_stream_vector(reqs);
  REQUIRE(st.ok());

  // Commit
  tensorcast::daemon::v2::CommitRegisteredArtifactRequest creq;
  creq.set_registration_id(bresp.registration_id());
  tensorcast::daemon::v2::CommitRegisteredArtifactResponse cresp;
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
