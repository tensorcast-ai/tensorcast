// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include "core/cuda/cuda_api.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/recording_global_store_client.h"
#include "daemon/testing/cuda_ipc_spawn_helper.h"
#include "grpcpp/server_context.h"

namespace {

namespace fs = std::filesystem;

nlohmann::json make_tensor_entry(
    uint64_t offset,
    uint64_t size,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& stride,
    const std::string& dtype,
    uint64_t storage_offset = 0) {
  nlohmann::json entry = nlohmann::json::array();
  entry.push_back(offset);
  entry.push_back(size);
  nlohmann::json shape_json = nlohmann::json::array();
  for (int64_t dim : shape) {
    shape_json.push_back(dim);
  }
  entry.push_back(shape_json);
  nlohmann::json stride_json = nlohmann::json::array();
  for (int64_t s : stride) {
    stride_json.push_back(s);
  }
  entry.push_back(stride_json);
  entry.push_back(dtype);
  entry.push_back(storage_offset);
  return entry;
}

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = fs::temp_directory_path() / "tensorcast_daemon_lip_piece_assembly_test";
  fs::create_directories(opts.storage_path);
  opts.p2p_port = 0; // Let OS select an available port.
  opts.memory_pool_size = 64ULL << 20; // 64 MiB
  opts.tx_slice_bytes = 1ULL << 20; // 1 MiB
  opts.num_thread = 2;
  return opts;
}

std::unique_ptr<tensorcast::daemon::DaemonServiceHarness> make_harness(
    const std::shared_ptr<tensorcast::store::StoreEngine>& engine,
    const std::shared_ptr<tensorcast::store::components::IGlobalStoreClient>& gs_client) {
  tensorcast::daemon::DaemonOptions options;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, options, nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

void populate_view_infos(tensorcast::store::testing::RecordingGlobalStoreClient& gs_stub) {
  gs_stub.view_infos.clear();
  for (const auto& update : gs_stub.view_updates) {
    tensorcast::store::components::ViewInfo info;
    info.view_id = update.view_id;
    info.view_spec_json = update.view_spec_json;
    info.view_size_bytes = update.view_size_bytes;
    info.view_data_hash = update.view_data_hash;
    info.canonical_size_bytes = update.canonical_size_bytes;
    info.canonical_bytes_covered = update.canonical_bytes_covered;
    info.canonical_ranges = update.canonical_ranges;
    gs_stub.view_infos.push_back(info);
  }
}

} // namespace

TEST_CASE("LIP pieces can be registered and assembled via remote keys", "[daemon][lip][piece][assembly][fakecuda]") {
  if (!tensorcast::cuda::is_available()) {
    WARN("CUDA not available – skipping LIP piece assembly test.");
    return;
  }

  const auto opts = make_opts();
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(opts);
  auto gs_stub = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  engine->set_global_store_client_for_testing(gs_stub);
  auto harness = make_harness(engine, gs_stub);
  auto& svc = harness->service();

  const std::string assembly_id = "cgid:lip-piece-assembly";

  // Canonical index: single uint8 tensor of 8 bytes.
  nlohmann::json index = nlohmann::json::object();
  index["weights"] = make_tensor_entry(
      /*offset=*/0,
      /*size=*/8,
      /*shape=*/{8},
      /*stride=*/{1},
      /*dtype=*/"torch.uint8");
  const std::string index_json = index.dump();
  gs_stub->canonical_index_json = index_json;

  const auto helper_path_or = tensorcast::daemon::testing::resolve_cuda_ipc_helper_path();
  REQUIRE(helper_path_or.ok());
  std::vector<tensorcast::daemon::testing::CudaIpcBufferSpec> buffers = {
      {.size_bytes = 4, .fill_byte = 0x11},
      {.size_bytes = 4, .fill_byte = 0x22},
  };
  auto child_or = tensorcast::daemon::testing::CudaIpcChild::Spawn(*helper_path_or, 0, buffers);
  INFO("cuda_ipc_helper spawn status: " << child_or.status());
  REQUIRE(child_or.ok());
  auto child = std::move(*child_or);
  REQUIRE(child.handle_bytes().size() == 2);

  // Ensure local identity matches GS "remote" to trigger the local-routing fallback.
  const uint32_t listen_port = engine->get_shared_comm_manager()->listen_port();
  engine->set_worker_identity(
      /*worker_id=*/"worker-0",
      /*node_id=*/"node-0",
      /*node_address=*/"127.0.0.1",
      /*grpc_port=*/0,
      /*p2p_port=*/listen_port);
  harness->kernel().worker_identity_store().set_registered("worker-0", "node-0");
  gs_stub->remote_node_id = "node-0";
  gs_stub->remote_node_address = "127.0.0.1";
  gs_stub->remote_node_port = listen_port;
  gs_stub->allow_view_transport = true;
  gs_stub->replica_transport_not_found = true;

  auto register_piece = [&](int start, int length, const std::string& handle_bytes) {
    tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
    breq.set_device_id(0);
    breq.set_total_size(static_cast<uint64_t>(length));
    breq.set_ttl_ms(5000);
    breq.set_owner_pid(child.pid());
    breq.set_client_artifact_id(assembly_id);
    auto* ti = breq.mutable_tensor_index_data();
    ti->set_data(index_json);
    ti->set_schema_version("v3");
    ti->set_encoding("json");
    breq.mutable_lease()->set_in_place(true);

    auto* view = breq.mutable_view();
    view->set_canonical_size_bytes(8);
    view->set_registration_kind(tensorcast::daemon::v2::VIEW_REGISTRATION_KIND_PIECE);
    view->set_placement(tensorcast::daemon::v2::TRANSFORM_PLACEMENT_CLIENT);
    auto* spec = view->mutable_spec();
    auto& ops = (*spec->mutable_tensors())["weights"];
    auto* op = ops.add_ops();
    auto* narrow = op->mutable_narrow();
    narrow->set_dim(0);
    narrow->set_start(start);
    narrow->set_length(static_cast<uint64_t>(length));

    grpc::ServerContext ctx;
    tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
    auto st = svc.BeginRegisterArtifact(&ctx, &breq, &bresp);
    INFO("BeginRegisterArtifact status: " << st.error_message());
    REQUIRE(st.ok());

    tensorcast::daemon::v2::FeedRegisterArtifactStreamRequest freq;
    freq.set_registration_id(bresp.registration_id());
    auto* ls = freq.mutable_lease_segments();
    auto* seg = ls->add_segments();
    seg->set_storage_id("s0");
    seg->set_storage_offset(0);
    seg->set_artifact_offset(0);
    seg->set_length(static_cast<uint64_t>(length));
    auto* storage = freq.add_storage_entries();
    storage->set_storage_id("s0");
    storage->set_device_id(0);
    storage->set_storage_length(static_cast<uint64_t>(length));
    storage->set_cuda_ipc_handle(handle_bytes);
    storage->set_mapping_base_offset(0);

    st = svc.feed_register_artifact_stream_vector({freq});
    INFO("feed_register_artifact_stream_vector status: " << st.error_message());
    REQUIRE(st.ok());

    tensorcast::daemon::v2::CommitRegisteredArtifactRequest creq;
    creq.set_registration_id(bresp.registration_id());
    tensorcast::daemon::v2::CommitRegisteredArtifactResponse cresp;
    st = svc.CommitRegisteredArtifact(&ctx, &creq, &cresp);
    INFO("CommitRegisteredArtifact status: " << st.error_message());
    REQUIRE(st.ok());
    REQUIRE_FALSE(cresp.view_id().empty());
    REQUIRE(cresp.has_local_stable_tier());
    REQUIRE(cresp.local_stable_tier().status() == tensorcast::daemon::v2::LOCAL_STABLE_TIER_STATUS_SKIPPED);
    REQUIRE(cresp.local_stable_tier().message().find("view registrations") != std::string::npos);
  };

  register_piece(/*start=*/0, /*length=*/4, child.handle_bytes().front());
  register_piece(/*start=*/4, /*length=*/4, child.handle_bytes().back());

  populate_view_infos(*gs_stub);

  const tensorcast::store::DeviceKey gpu_device = tensorcast::store::DeviceRegistry::instance().gpu_key(0);
  tensorcast::store::loading::MaterializeHints hints;
  hints.artifact_id = assembly_id;
  auto handle_or =
      engine->materialize_replica(gpu_device, tensorcast::store::StoreEngine::MaterializeMode::AUTO, hints);
  INFO("materialize_replica status: " << handle_or.status());
  REQUIRE(handle_or.ok());
  const auto handle = std::move(*handle_or);
  REQUIRE(engine->wait_replica_ready(handle.replica_key) == 0);

  auto size_or = engine->get_replica_size(handle.replica_key);
  REQUIRE(size_or.ok());
  REQUIRE(*size_or == 8);

  auto ptr_or = engine->get_replica_gpu_ptr(handle.replica_key);
  REQUIRE(ptr_or.ok());
  std::vector<std::uint8_t> host(8, 0);
  auto copy_status =
      tensorcast::cuda::memcpy(host.data(), reinterpret_cast<void*>(*ptr_or), host.size(), cudaMemcpyDeviceToHost);
  REQUIRE(copy_status.ok());
  REQUIRE(tensorcast::cuda::device_synchronize().ok());

  if (tensorcast::cuda::is_fake()) {
    WARN("Skipping byte-level payload assertions on Fake CUDA backend; IPC/transport semantics are simulated.");
  } else {
    for (size_t i = 0; i < 4; ++i) {
      CHECK(host[i] == 0x11);
    }
    for (size_t i = 4; i < host.size(); ++i) {
      CHECK(host[i] == 0x22);
    }
  }

  REQUIRE(engine->clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(opts.storage_path, ec);
}
