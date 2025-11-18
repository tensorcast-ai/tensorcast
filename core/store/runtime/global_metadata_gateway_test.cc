// Copyright (c) 2025, TensorCast Team.

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "catch2/catch_test_macros.hpp"
#include "core/common/memory/memory_location.h"
#include "core/store/components/global_store_client.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/replica_config.h"
#include "core/store/runtime/component_catalog.h"
#include "core/store/runtime/global_metadata_gateway.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/runtime/replica_runtime.h"
#include "core/store/runtime/runtime_event_hub.h"
#include "core/store/store_engine_options.h"
#include "core/testing/test_helpers.h"
#include "gsl/pointers"

using tensorcast::DeviceType;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::DeviceKey;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::components::ChunkLocationInfo;
using tensorcast::store::components::ChunkStateUpdate;
using tensorcast::store::components::IGlobalStoreClient;
using tensorcast::store::components::KeyMapping;
using tensorcast::store::components::RemoteReplicaInfo;
using tensorcast::store::components::VariantViewUpdate;
using tensorcast::store::loading::InlineBufferSource;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::replica::ReplicaConfig;
using tensorcast::store::runtime::ComponentCatalog;
using tensorcast::store::runtime::GlobalMetadataGateway;
using tensorcast::store::runtime::IngestionResultEvent;
using tensorcast::store::runtime::ReplicaRuntime;
using tensorcast::store::runtime::RuntimeEvent;
using tensorcast::store::runtime::RuntimeEventHub;
using tensorcast::store::runtime::RuntimeEventType;

namespace {

class FakeGlobalStoreClient : public IGlobalStoreClient {
 public:
  struct RegisterCall {
    std::string artifact_id;
    std::string worker_id;
    DeviceKey device;
    MemoryLocation location;
    uint64_t memory_size{0};
  };

  void set_connected(bool connected) {
    connected_ = connected;
  }

  const std::vector<RegisterCall>& register_calls() const {
    return register_calls_;
  }

  ~FakeGlobalStoreClient() override = default;

  absl::Status initialize() override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> register_worker(
      std::string_view,
      std::string_view,
      uint32_t,
      uint32_t,
      uint64_t,
      uint64_t,
      bool,
      std::string_view) override {
    return absl::UnimplementedError("register_worker not used");
  }

  absl::Status send_heartbeat(std::string_view, uint64_t, bool) override {
    return absl::UnimplementedError("send_heartbeat not used");
  }

  absl::StatusOr<tensorcast::global_store::v1::WorkerHeartbeatResponse> send_heartbeat_enhanced(
      std::string_view,
      uint64_t,
      bool,
      uint64_t,
      std::string_view,
      const std::vector<std::string>&,
      int64_t,
      tensorcast::global_store::v1::ConnectionStatus) override {
    return absl::UnimplementedError("send_heartbeat_enhanced not used");
  }

  absl::Status unregister_worker(std::string_view, bool) override {
    return absl::UnimplementedError("unregister_worker not used");
  }

  absl::StatusOr<std::string> register_replica(
      std::string_view artifact_id,
      std::string_view worker_id,
      const DeviceKey& device,
      MemoryLocation location,
      uint64_t memory_size,
      uint32_t) override {
    register_calls_.push_back(
        RegisterCall{
            .artifact_id = std::string(artifact_id),
            .worker_id = std::string(worker_id),
            .device = device,
            .location = location,
            .memory_size = memory_size,
        });
    return std::string("fake-replica");
  }

  absl::Status record_variant_residency(std::string_view, std::string_view, uint64_t, std::optional<std::string_view>)
      override {
    return absl::UnimplementedError("record_variant_residency not used");
  }

  absl::StatusOr<std::string> register_memory_replica(
      std::string_view,
      std::string_view,
      const DeviceKey&,
      uint64_t,
      std::string_view,
      const std::vector<std::string>&,
      const std::vector<uint64_t>&,
      const std::optional<std::string>&,
      std::string_view,
      std::string_view,
      uint32_t,
      const std::optional<std::string>&) override {
    return absl::UnimplementedError("register_memory_replica not used");
  }

  absl::Status unregister_replica(std::string_view, std::string_view) override {
    return absl::UnimplementedError("unregister_replica not used");
  }

  absl::Status unregister_replica_by_worker(
      std::string_view,
      std::string_view,
      std::optional<MemoryLocation>,
      std::optional<uint32_t>) override {
    return absl::UnimplementedError("unregister_replica_by_worker not used");
  }

  absl::StatusOr<tensorcast::store::components::TransportSession> request_replica_transport(
      std::string_view,
      std::string_view,
      std::string_view,
      uint32_t,
      const DeviceKey&,
      uint32_t) override {
    return absl::UnimplementedError("request_replica_transport not used");
  }

  absl::StatusOr<tensorcast::store::components::TransportSession> request_view_transport(
      std::string_view,
      std::string_view,
      std::string_view,
      std::string_view,
      uint32_t,
      const DeviceKey&,
      uint32_t) override {
    return absl::UnimplementedError("request_view_transport not used");
  }

  absl::Status complete_replica_transport(std::string_view) override {
    return absl::UnimplementedError("complete_replica_transport not used");
  }

  absl::StatusOr<std::vector<RemoteReplicaInfo>> get_artifact_replicas(std::string_view) override {
    return absl::UnimplementedError("get_artifact_replicas not used");
  }

  absl::StatusOr<std::vector<ChunkLocationInfo>> query_chunk_locations(std::string_view, const std::vector<uint32_t>&)
      override {
    return absl::UnimplementedError("query_chunk_locations not used");
  }

  absl::StatusOr<std::pair<uint64_t, std::string>> synchronize_worker_state(
      const tensorcast::global_store::v1::WorkerLocalState&,
      bool,
      std::vector<tensorcast::global_store::v1::StateChange>*) override {
    return absl::UnimplementedError("synchronize_worker_state not used");
  }

  absl::StatusOr<std::pair<uint64_t, std::string>> request_full_state_sync(
      std::string_view,
      uint64_t,
      std::vector<tensorcast::common::v1::ReplicaInfo>*) override {
    return absl::UnimplementedError("request_full_state_sync not used");
  }

  bool is_connected() const override {
    return connected_;
  }

  absl::Status batch_update_chunk_states(std::string_view, std::string_view, const std::vector<ChunkStateUpdate>&)
      override {
    return absl::UnimplementedError("batch_update_chunk_states not used");
  }

  absl::StatusOr<KeyMapping> resolve_key_mapping(std::string_view) override {
    return absl::UnimplementedError("resolve_key_mapping not used");
  }

  absl::StatusOr<std::string> get_artifact_index_by_id(std::string_view) override {
    return absl::UnimplementedError("get_artifact_index_by_id not used");
  }

  absl::Status upsert_key_mapping(std::string_view, std::string_view, std::string_view, absl::Duration) override {
    return absl::UnimplementedError("upsert_key_mapping not used");
  }

  absl::Status revoke_key_mapping(std::string_view) override {
    return absl::UnimplementedError("revoke_key_mapping not used");
  }

  void update_local_endpoint(std::string node_id, std::string node_address, uint32_t grpc_port, uint32_t p2p_port)
      override {
    last_endpoint_ = Endpoint{
        .node_id = std::move(node_id),
        .node_address = std::move(node_address),
        .grpc_port = grpc_port,
        .p2p_port = p2p_port,
    };
  }

  absl::Status update_artifact_view_state(const VariantViewUpdate&) override {
    return absl::UnimplementedError("update_artifact_view_state not used");
  }

 private:
  struct Endpoint {
    std::string node_id;
    std::string node_address;
    uint32_t grpc_port{0};
    uint32_t p2p_port{0};
  };

  bool connected_{false};
  std::vector<RegisterCall> register_calls_;
  Endpoint last_endpoint_;
};

ReplicaKey MakeCpuReplicaKey(std::string artifact_id) {
  DeviceKey device{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  ReplicaKey key{
      .artifact_id = std::move(artifact_id),
      .view_id = std::nullopt,
      .device = device,
      .replica = 0,
  };
  return key;
}

} // namespace

TEST_CASE("GlobalMetadataGateway registers replicas via RuntimeEventHub", "[runtime][metadata]") {
  SKIP_IF_NO_CUDA();
  StoreEngineOptions opts;
  opts.storage_path = "";
  opts.memory_pool_size = 32ull * 1024 * 1024;
  opts.tx_slice_bytes = 512 * 1024;
  opts.artifact_chunk_bytes = opts.tx_slice_bytes * 2;
  opts.num_thread = 1;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);

  ComponentCatalog catalog(opts);
  CHECK_OK(catalog.start());
  tensorcast::store::components::WorkerIdentity identity{
      .worker_id = "worker-test",
      .node_id = "node-A",
      .node_address = "127.0.0.1",
      .grpc_port = 50051,
      .p2p_port = 50052,
  };
  catalog.set_worker_identity(identity);

  auto fake_client = std::make_shared<FakeGlobalStoreClient>();
  fake_client->set_connected(true);
  catalog.set_global_store_client_for_testing(fake_client);

  RuntimeEventHub event_hub;
  ReplicaRuntime runtime(ReplicaRuntime::Config{.component_catalog = &catalog, .event_hub = &event_hub});
  GlobalMetadataGateway gateway(
      GlobalMetadataGateway::Config{
          .component_catalog = &catalog,
          .replica_runtime = &runtime,
          .event_hub = &event_hub,
      });

  constexpr size_t kBytes = 8 * 1024;
  auto backing = std::make_shared<std::vector<uint8_t>>(kBytes, 0x5A);
  auto data_view = std::shared_ptr<const void>(backing, static_cast<const void*>(backing->data()));
  InlineBufferSource source{.data = data_view, .size_bytes = kBytes};
  ReplicaConfig config{
      .source = source,
      .artifact_identifier = "ingress_artifact",
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool =
          gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{catalog.pinned_buffer_pool()},
      .artifact_chunk_bytes = catalog.artifact_chunk_bytes(),
      .expected_artifact_size = kBytes};
  config.max_buffer_bytes = kBytes;
  config.pinned_memory_timeout = std::chrono::milliseconds(0);

  auto replica = runtime.get_or_create_replica("ingress_artifact", config);
  REQUIRE(replica != nullptr);

  ReplicaKey key = MakeCpuReplicaKey("ingress_artifact");

  IngestionResultEvent event;
  event.source = tensorcast::store::runtime::IngestionSource::kDisk;
  event.artifact_id = "ingress_artifact";
  event.target_device = key.device;
  event.target_location = MemoryLocation::CPU;
  event.bytes_transferred = kBytes;
  event.duration_seconds = 0.25;
  event.status = absl::OkStatus();
  event.replica_key = key;
  event.publish_to_global_store = true;

  RuntimeEvent runtime_event;
  runtime_event.type = RuntimeEventType::kIngressCompleted;
  runtime_event.payload = event;
  event_hub.publish(runtime_event);

  REQUIRE(fake_client->register_calls().size() == 1);
  const auto& call = fake_client->register_calls().front();
  REQUIRE(call.artifact_id == "ingress_artifact");
  REQUIRE(call.worker_id == identity.worker_id);
  REQUIRE(call.device.type == DeviceType::CPU);
  REQUIRE(call.memory_size == kBytes);

  runtime.clear_mem();
  catalog.shutdown();
}
