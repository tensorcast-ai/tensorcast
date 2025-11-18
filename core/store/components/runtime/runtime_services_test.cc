// Copyright (c) 2025, TensorCast Team.

#include <chrono>
#include <memory>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"
#include "core/common/device_types.h"
#include "core/store/components/runtime/component_catalog.h"
#include "core/store/components/runtime/replica_service.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/replica_config.h"
#include "core/store/store_engine_options.h"
#include "core/testing/test_helpers.h"
#include "gsl/pointers"

using tensorcast::DeviceType;
using tensorcast::common::memory::PinnedBufferPool;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::components::runtime::ComponentCatalog;
using tensorcast::store::components::runtime::ReplicaService;
using tensorcast::store::loading::InlineBufferSource;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::replica::ReplicaConfig;

namespace {

StoreEngineOptions MakeTestOptions() {
  StoreEngineOptions opts;
  opts.storage_path = "";
  opts.memory_pool_size = 16ull * 1024 * 1024;
  opts.tx_slice_bytes = 512 * 1024;
  opts.artifact_chunk_bytes = opts.tx_slice_bytes * 2;
  opts.num_thread = 1;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  opts.p2p_listen_host = "127.0.0.1";
  opts.p2p_port = 0; // let the OS pick an ephemeral port during tests
  opts.enable_rdma = false;
  return opts;
}

TEST_CASE("ComponentCatalog start/stop preserves pinned pool", "[component_catalog]") {
  SKIP_IF_NO_CUDA();
  auto opts = MakeTestOptions();
  ComponentCatalog catalog(opts);

  auto pool_before = catalog.pinned_buffer_pool();
  REQUIRE(pool_before != nullptr);
  REQUIRE(pool_before->slice_bytes() == opts.tx_slice_bytes);

  CHECK_OK(catalog.start());
  auto pool_after = catalog.pinned_buffer_pool();
  REQUIRE(pool_after == pool_before);
  REQUIRE(catalog.communication_manager() != nullptr);
  REQUIRE(catalog.communication_manager()->is_enabled());
  REQUIRE(catalog.options().p2p_port != 0);

  catalog.shutdown();
  REQUIRE(catalog.communication_manager() == nullptr);
}

TEST_CASE("ReplicaService handles inline CPU replicas", "[replica_service]") {
  SKIP_IF_NO_CUDA();
  auto opts = MakeTestOptions();
  ComponentCatalog catalog(opts);
  CHECK_OK(catalog.start());
  ReplicaService service(&catalog);

  constexpr size_t kBufferBytes = 8 * 1024;
  auto backing = std::make_shared<std::vector<uint8_t>>(kBufferBytes, 0xAB);
  auto data_view = std::shared_ptr<const void>(backing, static_cast<const void*>(backing->data()));
  InlineBufferSource source{.data = data_view, .size_bytes = kBufferBytes};

  ReplicaConfig config{
      .source = source,
      .artifact_identifier = "inline_artifact",
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool = gsl::not_null<std::shared_ptr<PinnedBufferPool>>{catalog.pinned_buffer_pool()},
      .artifact_chunk_bytes = catalog.artifact_chunk_bytes(),
      .expected_artifact_size = kBufferBytes};
  config.max_buffer_bytes = kBufferBytes;
  config.pinned_memory_timeout = std::chrono::milliseconds(0);

  auto replica = service.get_or_create_replica("inline_artifact", config);
  REQUIRE(replica != nullptr);

  auto devices = service.get_resident_devices("inline_artifact");
  REQUIRE(devices.size() == 1);
  REQUIRE(devices[0].type == DeviceType::CPU);

  ReplicaKey key{
      .artifact_id = "inline_artifact",
      .view_id = std::nullopt,
      .device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      .replica = 0};
  REQUIRE(service.wait_replica_ready(key) == 0);

  REQUIRE(service.clear_mem() == 0);
  REQUIRE(service.get_resident_devices("inline_artifact").empty());
  catalog.shutdown();
}

} // namespace
