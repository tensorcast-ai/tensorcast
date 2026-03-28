// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <unistd.h>
#include <filesystem>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include "absl/log/log.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/recording_global_store_client.h"
#include "grpcpp/server_context.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace {

static std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env)
    return std::filesystem::path(env);
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_cpp_test";
}

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = test_tmpdir();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47001;
  opts.memory_pool_size = 64ull << 20; // 64 MiB
  opts.tx_slice_bytes = 1ull << 20; // 1 MiB
  opts.num_thread = 2;
  return opts;
}

tensorcast::common::v1::ViewSpec make_narrow_view_spec(int start, int length) {
  tensorcast::common::v1::ViewSpec spec;
  tensorcast::common::v1::TensorViewOps ops;
  auto* narrow = ops.add_ops()->mutable_narrow();
  narrow->set_dim(0);
  narrow->set_start(start);
  narrow->set_length(length);
  (*spec.mutable_tensors())["weights"] = ops;
  return spec;
}

std::unique_ptr<tensorcast::daemon::DaemonServiceHarness> make_harness(
    const std::shared_ptr<tensorcast::store::StoreEngine>& engine) {
  tensorcast::daemon::DaemonOptions options;
  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, options, nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

struct HarnessWithGlobalStore {
  std::unique_ptr<tensorcast::daemon::DaemonServiceHarness> harness;
  std::shared_ptr<tensorcast::store::testing::RecordingGlobalStoreClient> global_store_client;
};

HarnessWithGlobalStore make_harness_with_global_store(const std::shared_ptr<tensorcast::store::StoreEngine>& engine) {
  tensorcast::daemon::DaemonOptions options;
  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, options, nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return {.harness = std::move(harness), .global_store_client = gs_client};
}

} // namespace

TEST_CASE("CommitRegisteredArtifact populates descriptor", "[daemon][registration]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness = make_harness(engine);
  auto& service = harness->service();

  // Begin registration with inline index data to exercise content addressing (v1 namespace)
  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(1 * 1024 * 1024);
  breq.set_owner_pid(getpid());
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data("{}");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto st = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  if (!st.ok()) {
    FAIL(std::string("BeginRegisterArtifact failed: ") + st.error_message());
  }
  REQUIRE(!bresp.registration_id().empty());
  REQUIRE(bresp.device_id() == 0);
  REQUIRE(bresp.total_size() == 1 * 1024 * 1024);
  REQUIRE(bresp.has_coalesced());
  REQUIRE(bresp.coalesced().daemon_ipc_handle().size() > 0);

  // Commit and validate descriptor fields are populated
  tensorcast::daemon::v2::CommitRegisteredArtifactRequest creq;
  creq.set_registration_id(bresp.registration_id());
  tensorcast::daemon::v2::CommitRegisteredArtifactResponse cresp;
  st = service.CommitRegisteredArtifact(&ctx, &creq, &cresp);
  REQUIRE(st.ok());
  // Descriptor presence and consistency (only field in new response)
  REQUIRE(cresp.has_artifact_descriptor());
  const auto& desc = cresp.artifact_descriptor();
  REQUIRE(desc.artifact_id().rfind("mi2:", 0) == 0); // starts with "mi2:"
  REQUIRE(!desc.index_multihash().empty());
  REQUIRE(!desc.data_multihash().empty());
  REQUIRE(desc.total_size() == 1 * 1024 * 1024);
  REQUIRE(desc.id_kind() == tensorcast::common::v1::ARTIFACT_ID_KIND_MI2);
}

TEST_CASE("CommitRegisteredArtifact degrades when warm local stable cannot be satisfied", "[daemon][registration]") {
  auto opts = make_opts();
  opts.memory_tier_config = tensorcast::store::MemoryTierConfig{.stable_bytes = 1};
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(std::move(opts));
  auto harness = make_harness(engine);
  auto& service = harness->service();

  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(32);
  breq.set_owner_pid(getpid());
  breq.mutable_policy()->set_profile(tensorcast::daemon::v2::POLICY_PROFILE_WARM);
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data(R"({"weights":[0,16,[2,2],[2,1],"torch.float32",0]})");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto st = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(st.ok());
  REQUIRE(!bresp.registration_id().empty());

  tensorcast::daemon::v2::CommitRegisteredArtifactRequest creq;
  creq.set_registration_id(bresp.registration_id());
  tensorcast::daemon::v2::CommitRegisteredArtifactResponse cresp;
  st = service.CommitRegisteredArtifact(&ctx, &creq, &cresp);
  REQUIRE(st.ok());
  REQUIRE(cresp.has_local_stable_tier());
  REQUIRE(cresp.local_stable_tier().status() == tensorcast::daemon::v2::LOCAL_STABLE_TIER_STATUS_DEGRADED);
  REQUIRE_FALSE(cresp.local_stable_tier().message().empty());
}

TEST_CASE("CommitRegisteredArtifact fails when pinned local stable cannot be satisfied", "[daemon][registration]") {
  auto opts = make_opts();
  opts.memory_tier_config = tensorcast::store::MemoryTierConfig{.stable_bytes = 1};
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(std::move(opts));
  auto harness = make_harness(engine);
  auto& service = harness->service();

  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(16);
  breq.set_owner_pid(getpid());
  breq.mutable_policy()->set_profile(tensorcast::daemon::v2::POLICY_PROFILE_PINNED);
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data(R"({"weights":[0,16,[2,2],[2,1],"torch.float32",0]})");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto st = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(st.ok());
  REQUIRE(!bresp.registration_id().empty());

  tensorcast::daemon::v2::CommitRegisteredArtifactRequest creq;
  creq.set_registration_id(bresp.registration_id());
  tensorcast::daemon::v2::CommitRegisteredArtifactResponse cresp;
  st = service.CommitRegisteredArtifact(&ctx, &creq, &cresp);
  REQUIRE_FALSE(st.ok());
  REQUIRE(st.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED);
}

TEST_CASE(
    "CommitRegisteredArtifact pinned fails when local stable tier hits banned GPU staging",
    "[daemon][registration][stable_budget]") {
  auto opts = make_opts();
  opts.memory_tier_config = tensorcast::store::MemoryTierConfig{.stable_bytes = 32};
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(std::move(opts));
  auto harness = make_harness(engine);
  auto& service = harness->service();

  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(24);
  breq.set_owner_pid(getpid());
  breq.mutable_policy()->set_profile(tensorcast::daemon::v2::POLICY_PROFILE_PINNED);
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data(R"({"weights":[0,24,[6],[1],"torch.float32",0]})");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto st = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(st.ok());
  REQUIRE(!bresp.registration_id().empty());

  tensorcast::daemon::v2::CommitRegisteredArtifactRequest creq;
  creq.set_registration_id(bresp.registration_id());
  tensorcast::daemon::v2::CommitRegisteredArtifactResponse cresp;
  st = service.CommitRegisteredArtifact(&ctx, &creq, &cresp);
  REQUIRE_FALSE(st.ok());
  REQUIRE(st.error_message().find("stage_on_gpu is disabled") != std::string::npos);
}

TEST_CASE("CommitRegisteredArtifact accepts CGID", "[daemon][registration]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness = make_harness(engine);
  auto& service = harness->service();

  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(512 * 1024);
  breq.set_owner_pid(getpid());
  breq.set_client_artifact_id("cgid:testcgid123");
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data("{}");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto st = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(st.ok());

  tensorcast::daemon::v2::CommitRegisteredArtifactRequest creq;
  creq.set_registration_id(bresp.registration_id());
  tensorcast::daemon::v2::CommitRegisteredArtifactResponse cresp;
  st = service.CommitRegisteredArtifact(&ctx, &creq, &cresp);
  REQUIRE(st.ok());
  REQUIRE(cresp.has_artifact_descriptor());
  const auto& desc = cresp.artifact_descriptor();
  REQUIRE(desc.artifact_id() == "cgid:testcgid123");
  REQUIRE_FALSE(desc.index_multihash().empty());
  REQUIRE(desc.data_multihash().empty());
  REQUIRE(desc.id_kind() == tensorcast::common::v1::ARTIFACT_ID_KIND_CGID);
}

TEST_CASE(
    "BeginRegisterArtifact enforces server transpose fallback when GPU unavailable",
    "[daemon][registration][view]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness = make_harness(engine);
  auto& service = harness->service();

  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(99); // invalid device id to trigger fallback guard
  breq.set_total_size(16);
  breq.set_owner_pid(getpid());
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data(R"({"weights":[0,16,[2,2],[2,1],"torch.float32",0]})");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  auto* view = breq.mutable_view();
  view->set_canonical_size_bytes(16);
  view->set_registration_kind(tensorcast::daemon::v2::VIEW_REGISTRATION_KIND_CANONICAL);
  view->set_placement(tensorcast::daemon::v2::TRANSFORM_PLACEMENT_SERVER);
  auto& tensors = *view->mutable_spec()->mutable_tensors();
  tensorcast::common::v1::TensorViewOps ops;
  auto* transpose = ops.add_ops()->mutable_transpose();
  transpose->set_dim0(0);
  transpose->set_dim1(1);
  tensors["weights"] = ops;

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto status = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(status.error_message().find("placement=CLIENT") != std::string::npos);
}

TEST_CASE("BeginRegisterArtifact rejects missing registration_kind", "[daemon][registration][view]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness = make_harness(engine);
  auto& service = harness->service();

  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(16);
  breq.set_owner_pid(getpid());
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data(R"({"weights":[0,32,[2,4],[4,1],"torch.float32",0]})");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  auto* view = breq.mutable_view();
  view->set_canonical_size_bytes(32);
  view->set_placement(tensorcast::daemon::v2::TRANSFORM_PLACEMENT_SERVER);
  view->mutable_spec()->CopyFrom(make_narrow_view_spec(0, 4));

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto status = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message().find("view.registration_kind must be specified") != std::string::npos);
}

TEST_CASE("BeginRegisterArtifact rejects full coverage transpose piece", "[daemon][registration][view]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness = make_harness(engine);
  auto& service = harness->service();

  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(32);
  breq.set_owner_pid(getpid());
  breq.set_client_artifact_id("cgid:piece-transpose");
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data(R"({"weights":[0,32,[2,4],[4,1],"torch.float32",0]})");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  auto* view = breq.mutable_view();
  view->set_canonical_size_bytes(32);
  view->set_placement(tensorcast::daemon::v2::TRANSFORM_PLACEMENT_SERVER);
  view->set_registration_kind(tensorcast::daemon::v2::VIEW_REGISTRATION_KIND_PIECE);
  tensorcast::common::v1::TensorViewOps ops;
  auto* transpose = ops.add_ops()->mutable_transpose();
  transpose->set_dim0(0);
  transpose->set_dim1(1);
  (*view->mutable_spec()->mutable_tensors())["weights"] = ops;

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto status = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message().find("must not fully cover canonical bytes") != std::string::npos);
}

TEST_CASE("BeginRegisterArtifact accepts piece lease-in-place", "[daemon][registration][view]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness = make_harness(engine);
  auto& service = harness->service();

  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(16);
  breq.set_owner_pid(getpid());
  breq.set_client_artifact_id("cgid:piece-lip");
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data(R"({"weights":[0,32,[8],[1],"torch.float32",0]})");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  auto* view = breq.mutable_view();
  view->set_canonical_size_bytes(32);
  view->set_placement(tensorcast::daemon::v2::TRANSFORM_PLACEMENT_SERVER);
  view->set_registration_kind(tensorcast::daemon::v2::VIEW_REGISTRATION_KIND_PIECE);
  view->mutable_spec()->CopyFrom(make_narrow_view_spec(0, 4));

  auto* lease = breq.mutable_lease();
  lease->set_in_place(true);

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto status = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(status.ok());
  REQUIRE(!bresp.registration_id().empty());
  REQUIRE(bresp.has_lease());
}

TEST_CASE("BeginRegisterArtifact rejects full coverage piece", "[daemon][registration][view]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness = make_harness(engine);
  auto& service = harness->service();

  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(32);
  breq.set_owner_pid(getpid());
  breq.set_client_artifact_id("cgid:piece-full");
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data(R"({"weights":[0,32,[8],[1],"torch.float32",0]})");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  auto* view = breq.mutable_view();
  view->set_canonical_size_bytes(32);
  view->set_placement(tensorcast::daemon::v2::TRANSFORM_PLACEMENT_SERVER);
  view->set_registration_kind(tensorcast::daemon::v2::VIEW_REGISTRATION_KIND_PIECE);
  view->mutable_spec()->CopyFrom(make_narrow_view_spec(0, 8));

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto status = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_CASE("BeginRegisterArtifact rejects piece begin when assembly is already sealed", "[daemon][registration][view]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness_with_gs = make_harness_with_global_store(engine);
  auto& service = harness_with_gs.harness->service();
  auto& gs_client = *harness_with_gs.global_store_client;

  tensorcast::store::components::ArtifactBinding binding;
  binding.from_artifact_id = "cgid:piece-sealed-begin";
  binding.to_artifact_id = "mi2:sealed";
  binding.kind = tensorcast::global_store::v1::ARTIFACT_BINDING_KIND_SEAL;
  gs_client.artifact_binding = std::move(binding);

  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(16);
  breq.set_owner_pid(getpid());
  breq.set_client_artifact_id("cgid:piece-sealed-begin");
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data(R"({"weights":[0,32,[8],[1],"torch.float32",0]})");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  auto* view = breq.mutable_view();
  view->set_canonical_size_bytes(32);
  view->set_placement(tensorcast::daemon::v2::TRANSFORM_PLACEMENT_SERVER);
  view->set_registration_kind(tensorcast::daemon::v2::VIEW_REGISTRATION_KIND_PIECE);
  view->mutable_spec()->CopyFrom(make_narrow_view_spec(0, 4));
  breq.mutable_lease()->set_in_place(true);

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto status = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(status.error_message().find("already sealed") != std::string::npos);
}

TEST_CASE(
    "CommitRegisteredArtifact rejects piece commit when assembly is already sealed",
    "[daemon][registration][view]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness_with_gs = make_harness_with_global_store(engine);
  auto& service = harness_with_gs.harness->service();
  auto& gs_client = *harness_with_gs.global_store_client;

  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(16);
  breq.set_owner_pid(getpid());
  breq.set_client_artifact_id("cgid:piece-sealed");
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data(R"({"weights":[0,32,[8],[1],"torch.float32",0]})");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  auto* view = breq.mutable_view();
  view->set_canonical_size_bytes(32);
  view->set_placement(tensorcast::daemon::v2::TRANSFORM_PLACEMENT_SERVER);
  view->set_registration_kind(tensorcast::daemon::v2::VIEW_REGISTRATION_KIND_PIECE);
  view->mutable_spec()->CopyFrom(make_narrow_view_spec(0, 4));
  breq.mutable_lease()->set_in_place(true);

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto status = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(status.ok());
  REQUIRE(!bresp.registration_id().empty());

  tensorcast::store::components::ArtifactBinding binding;
  binding.from_artifact_id = "cgid:piece-sealed";
  binding.to_artifact_id = "mi2:sealed";
  binding.kind = tensorcast::global_store::v1::ARTIFACT_BINDING_KIND_SEAL;
  gs_client.artifact_binding = std::move(binding);

  tensorcast::daemon::v2::CommitRegisteredArtifactRequest creq;
  creq.set_registration_id(bresp.registration_id());
  tensorcast::daemon::v2::CommitRegisteredArtifactResponse cresp;
  status = service.CommitRegisteredArtifact(&ctx, &creq, &cresp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(status.error_message().find("already sealed") != std::string::npos);
}
