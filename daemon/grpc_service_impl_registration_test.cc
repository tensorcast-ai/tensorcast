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
  opts.p2p_port = 0;
  opts.memory_pool_size = 64ull << 20; // 64 MiB
  opts.chunk_size = 1ull << 20; // 1 MiB
  opts.num_thread = 2;
  return opts;
}

} // namespace

TEST_CASE("CommitRegisteredArtifact populates descriptor", "[daemon][registration]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  StoreDaemonServiceImpl service(engine);

  // Begin registration with inline index data to exercise content addressing
  tensorcast::daemon::BeginRegisterArtifactRequest breq;
  breq.set_artifact_id("cpp_mem_artifact");
  breq.set_device_id(0);
  breq.set_total_size(1 * 1024 * 1024);
  breq.set_enable_p2p(false);
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data("{}");
  idx->set_schema_version("v2");
  idx->set_encoding("json");

  grpc::ServerContext ctx;
  tensorcast::daemon::BeginRegisterArtifactResponse bresp;
  auto st = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(st.ok());
  REQUIRE(!bresp.registration_id().empty());
  REQUIRE(bresp.device_id() == 0);
  REQUIRE(bresp.size() == 1 * 1024 * 1024);
  REQUIRE(bresp.daemon_ipc_handle().size() > 0);

  // Commit and validate descriptor fields are populated
  tensorcast::daemon::CommitRegisteredArtifactRequest creq;
  creq.set_registration_id(bresp.registration_id());
  tensorcast::daemon::CommitRegisteredArtifactResponse cresp;
  st = service.CommitRegisteredArtifact(&ctx, &creq, &cresp);
  REQUIRE(st.ok());
  REQUIRE(cresp.registration_id() == bresp.registration_id());
  REQUIRE(cresp.device_id() == 0);
  REQUIRE(cresp.size() == 1 * 1024 * 1024);
  REQUIRE(cresp.artifact_id().rfind("mi2:", 0) == 0); // starts with "mi2:"
  // Descriptor presence and consistency
  REQUIRE(cresp.has_descriptor_());
  const auto& desc = cresp.descriptor_();
  REQUIRE(desc.artifact_id() == cresp.artifact_id());
  REQUIRE(!desc.index_multihash().empty());
  REQUIRE(!desc.data_multihash().empty());
  REQUIRE(desc.total_size() == cresp.size());
}
