// Copyright (c) 2025, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <catch2/catch_test_macros.hpp>
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "grpcpp/server_context.h"

using tensorcast::daemon::StoreDaemonServiceImpl;

namespace {

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = std::filesystem::temp_directory_path() / "tensorcast_daemon_cpp_test";
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 0;
  opts.memory_pool_size = 64ull << 20;
  opts.tx_slice_bytes = 1ull << 20;
  opts.num_thread = 2;
  return opts;
}

} // namespace

TEST_CASE("KeepAlive/Revoke lifecycle no-ops", "[daemon][registration]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  StoreDaemonServiceImpl service(engine);

  // Begin coalesced registration
  grpc::ServerContext ctx;
  tensorcast::daemon::v1::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(1024 * 1024);
  breq.set_owner_pid(getpid());
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data("{}");
  idx->set_schema_version("v2");
  idx->set_encoding("json");
  tensorcast::daemon::v1::BeginRegisterArtifactResponse bresp;
  auto st = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(st.ok());

  // KeepAlive should return OK
  tensorcast::daemon::v1::KeepAliveRegisterArtifactRequest kreq;
  tensorcast::daemon::v1::KeepAliveRegisterArtifactResponse kresp;
  kreq.set_registration_id(bresp.registration_id());
  kreq.set_ttl_ms(2000);
  kreq.set_epoch(1);
  kreq.set_owner_pid(getpid());
  st = service.KeepAliveRegisterArtifact(&ctx, &kreq, &kresp);
  REQUIRE(st.ok());

  // Revoke should return OK
  tensorcast::daemon::v1::RevokeRegisteredArtifactRequest rreq;
  tensorcast::daemon::v1::RevokeRegisteredArtifactResponse rresp;
  rreq.set_registration_id(bresp.registration_id());
  rreq.set_reason("test");
  st = service.RevokeRegisteredArtifact(&ctx, &rreq, &rresp);
  REQUIRE(st.ok());
}
