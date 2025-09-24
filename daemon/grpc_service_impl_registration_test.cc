// Copyright (c) 2025, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <filesystem>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include "absl/log/log.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v1/store_daemon.grpc.pb.h"

using tensorcast::daemon::StoreDaemonServiceImpl;

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

} // namespace

TEST_CASE("CommitRegisteredArtifact populates descriptor", "[daemon][registration]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  StoreDaemonServiceImpl service(engine);

  // Begin registration with inline index data to exercise content addressing (v1 namespace)
  tensorcast::daemon::v1::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(1 * 1024 * 1024);
  breq.set_owner_pid(getpid());
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data("{}");
  idx->set_schema_version("v2");
  idx->set_encoding("json");

  grpc::ServerContext ctx;
  tensorcast::daemon::v1::BeginRegisterArtifactResponse bresp;
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
  tensorcast::daemon::v1::CommitRegisteredArtifactRequest creq;
  creq.set_registration_id(bresp.registration_id());
  tensorcast::daemon::v1::CommitRegisteredArtifactResponse cresp;
  st = service.CommitRegisteredArtifact(&ctx, &creq, &cresp);
  REQUIRE(st.ok());
  // Descriptor presence and consistency (only field in new response)
  REQUIRE(cresp.has_artifact_descriptor());
  const auto& desc = cresp.artifact_descriptor();
  REQUIRE(desc.artifact_id().rfind("mi2:", 0) == 0); // starts with "mi2:"
  REQUIRE(!desc.index_multihash().empty());
  REQUIRE(!desc.data_multihash().empty());
  REQUIRE(desc.total_size() == 1 * 1024 * 1024);
}
