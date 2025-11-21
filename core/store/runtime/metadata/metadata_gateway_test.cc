// Copyright (c) 2025, TensorCast Team.

#include <array>
#include <chrono>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "catch2/catch_test_macros.hpp"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/runtime/metadata/metadata_gateway.h"
#include "core/store/runtime/metadata/metadata_types.h"
#include "core/store/runtime/metadata/registration_backend.h"
#include "core/store/runtime/replica/replica_runtime.h"
#include "core/store/store_engine_options.h"
#include "core/testing/test_helpers.h"

using tensorcast::DeviceType;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::DeviceKey;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::loading::InlineBufferSource;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::replica::Replica;
using tensorcast::store::replica::ReplicaConfig;
using tensorcast::store::runtime::IngestionResultEvent;
using tensorcast::store::runtime::ReplicaRuntime;
using tensorcast::store::runtime::RuntimeContext;
using tensorcast::store::runtime::metadata::ArtifactRegistration;
using tensorcast::store::runtime::metadata::MetadataGateway;
using tensorcast::store::runtime::metadata::RegistrationBackend;
using tensorcast::store::runtime::metadata::RegistrationPublication;
using tensorcast::store::runtime::metadata::RegistrationPublisher;
using tensorcast::store::runtime::metadata::RegistrationResources;

namespace {

class RecordingPublisher final : public RegistrationPublisher {
 public:
  absl::Status publish_registration(const RegistrationPublication& publication) override {
    publications.push_back(publication);
    return absl::OkStatus();
  }

  absl::Status update_variant_view(const tensorcast::store::components::VariantViewUpdate&) override {
    ++variant_updates;
    return absl::OkStatus();
  }

  std::vector<RegistrationPublication> publications;
  int variant_updates{0};
};

struct RegistrationBackendHarness {
  RegistrationBackendHarness()
      : device_manager(std::make_unique<tensorcast::store::components::DeviceManager>()),
        replica_registry(std::make_unique<tensorcast::store::components::ReplicaRegistry>()),
        metrics_collector(std::make_unique<tensorcast::store::components::MetricsCollector>()),
        memory_pool(
            std::make_shared<tensorcast::common::memory::PinnedBufferPool>(8ULL * 1024 * 1024, 1ULL * 1024 * 1024)),
        resources{
            .device_manager = gsl::not_null<tensorcast::store::components::DeviceManager*>{device_manager.get()},
            .replica_registry = gsl::not_null<tensorcast::store::components::ReplicaRegistry*>{replica_registry.get()},
            .metrics_collector =
                gsl::not_null<tensorcast::store::components::MetricsCollector*>{metrics_collector.get()},
            .memory_pool = gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{memory_pool},
            .communication_manager = nullptr,
        } {
    REQUIRE(device_manager->initialize().ok());
  }

  std::unique_ptr<tensorcast::store::components::DeviceManager> device_manager;
  std::unique_ptr<tensorcast::store::components::ReplicaRegistry> replica_registry;
  std::unique_ptr<tensorcast::store::components::MetricsCollector> metrics_collector;
  std::shared_ptr<tensorcast::common::memory::PinnedBufferPool> memory_pool;
  RegistrationResources resources;
  RecordingPublisher publisher;

  RegistrationBackend make_backend() {
    auto factory = [](const tensorcast::store::replica::ReplicaConfig& config)
        -> absl::StatusOr<std::shared_ptr<tensorcast::store::replica::Replica>> {
      auto created_or = tensorcast::store::replica::Replica::create(config);
      if (!created_or.ok()) {
        return created_or.status();
      }
      return std::shared_ptr<tensorcast::store::replica::Replica>(std::move(created_or.value()));
    };
    return RegistrationBackend(resources, std::move(factory), 1ULL << 20, std::chrono::milliseconds(10), &publisher);
  }
};

ArtifactRegistration MakeRegistration(uint64_t total_bytes) {
  ArtifactRegistration reg;
  reg.artifact_id = "artifact-basic";
  reg.tensor_index_key = "index-basic";
  reg.tensor_index_data = "{}";
  reg.device_id = 0;
  reg.total_size_bytes = total_bytes;
  reg.schema_version = "v3";
  reg.encoding = "json";
  reg.enable_p2p = false;
  return reg;
}

StoreEngineOptions MakeTestOptions() {
  StoreEngineOptions opts;
  opts.storage_path.clear();
  opts.memory_pool_size = 32ull * 1024 * 1024;
  opts.tx_slice_bytes = 256 * 1024;
  opts.artifact_chunk_bytes = opts.tx_slice_bytes;
  opts.num_thread = 1;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  opts.p2p_listen_host = "127.0.0.1";
  opts.p2p_port = 0;
  opts.enable_rdma = false;
  opts.global_store_address.clear();
  return opts;
}

ReplicaConfig MakeInlineReplicaConfig(
    RuntimeContext& context,
    std::string artifact_id,
    DeviceType device_type,
    int device_id,
    const std::shared_ptr<const void>& view,
    size_t bytes) {
  ReplicaConfig config{
      .source = InlineBufferSource{.data = view, .size_bytes = bytes},
      .artifact_identifier = std::move(artifact_id),
      .device_type = device_type,
      .local_device_id = device_id,
      .pinned_buffer_pool =
          gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{context.pinned_buffer_pool()},
      .artifact_chunk_bytes = context.artifact_chunk_bytes(),
      .expected_artifact_size = bytes,
      .max_buffer_bytes = bytes,
      .pinned_memory_timeout = std::chrono::milliseconds(0),
  };
  return config;
}

ReplicaKey MakeReplicaKey(std::string artifact_id, DeviceKey device) {
  return ReplicaKey{
      .artifact_id = std::move(artifact_id),
      .view_id = std::nullopt,
      .device = device,
      .replica = 0,
  };
}

class TestGlobalStoreClient final : public tensorcast::store::components::IGlobalStoreClient {
 public:
  bool connected{true};
  absl::Status next_register_status{absl::OkStatus()};
  int register_calls{0};
  std::vector<std::string> registered_artifacts;

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
    return absl::UnimplementedError("register_worker not used in tests");
  }

  absl::Status send_heartbeat(std::string_view, uint64_t, bool) override {
    return absl::UnimplementedError("send_heartbeat not used in tests");
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
    return absl::UnimplementedError("send_heartbeat_enhanced not used in tests");
  }

  absl::Status unregister_worker(std::string_view, bool) override {
    return absl::UnimplementedError("unregister_worker not used in tests");
  }

  absl::StatusOr<std::string> register_replica(
      std::string_view artifact_id,
      std::string_view,
      const DeviceKey&,
      MemoryLocation,
      uint64_t,
      uint32_t) override {
    registered_artifacts.emplace_back(artifact_id);
    ++register_calls;
    if (!next_register_status.ok()) {
      return next_register_status;
    }
    return absl::StrCat("rep-", register_calls);
  }

  absl::Status record_variant_residency(std::string_view, std::string_view, uint64_t, std::optional<std::string_view>)
      override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> register_memory_replica(
      std::string_view artifact_id,
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
    registered_artifacts.emplace_back(artifact_id);
    ++register_calls;
    if (!next_register_status.ok()) {
      return next_register_status;
    }
    return absl::StrCat("mem-", register_calls);
  }

  absl::Status unregister_replica(std::string_view, std::string_view) override {
    return absl::OkStatus();
  }

  absl::Status unregister_replica_by_worker(
      std::string_view,
      std::string_view,
      std::optional<MemoryLocation>,
      std::optional<uint32_t>) override {
    return absl::OkStatus();
  }

  absl::Status update_artifact_view_state(const tensorcast::store::components::VariantViewUpdate&) override {
    return absl::OkStatus();
  }

  absl::StatusOr<tensorcast::store::components::TransportSession> request_replica_transport(
      std::string_view,
      std::string_view,
      std::string_view,
      uint32_t,
      const DeviceKey&,
      uint32_t) override {
    return absl::UnimplementedError("request_replica_transport not used in tests");
  }

  absl::StatusOr<tensorcast::store::components::TransportSession> request_view_transport(
      std::string_view,
      std::string_view,
      std::string_view,
      std::string_view,
      uint32_t,
      const DeviceKey&,
      uint32_t) override {
    return absl::UnimplementedError("request_view_transport not used in tests");
  }

  absl::Status complete_replica_transport(std::string_view) override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<tensorcast::store::components::RemoteReplicaInfo>> get_artifact_replicas(
      std::string_view) override {
    return absl::UnimplementedError("get_artifact_replicas not used in tests");
  }

  absl::StatusOr<std::vector<tensorcast::store::components::ChunkLocationInfo>> query_chunk_locations(
      std::string_view,
      const std::vector<uint32_t>&) override {
    return absl::UnimplementedError("query_chunk_locations not used in tests");
  }

  absl::StatusOr<std::pair<uint64_t, std::string>> synchronize_worker_state(
      const tensorcast::global_store::v1::WorkerLocalState&,
      bool,
      std::vector<tensorcast::global_store::v1::StateChange>*) override {
    return absl::UnimplementedError("synchronize_worker_state not used in tests");
  }

  absl::StatusOr<std::pair<uint64_t, std::string>> request_full_state_sync(
      std::string_view,
      uint64_t,
      std::vector<tensorcast::common::v1::ReplicaInfo>*) override {
    return absl::UnimplementedError("request_full_state_sync not used in tests");
  }

  bool is_connected() const override {
    return connected;
  }

  absl::Status batch_update_chunk_states(
      std::string_view,
      std::string_view,
      const std::vector<tensorcast::store::components::ChunkStateUpdate>&) override {
    return absl::UnimplementedError("batch_update_chunk_states not used in tests");
  }

  absl::StatusOr<tensorcast::store::components::KeyMapping> resolve_key_mapping(std::string_view) override {
    return absl::UnimplementedError("resolve_key_mapping not used in tests");
  }

  absl::StatusOr<std::string> get_artifact_index_by_id(std::string_view) override {
    return absl::UnimplementedError("get_artifact_index_by_id not used in tests");
  }

  absl::Status upsert_key_mapping(std::string_view, std::string_view, std::string_view, absl::Duration) override {
    return absl::UnimplementedError("upsert_key_mapping not used in tests");
  }

  absl::Status revoke_key_mapping(std::string_view) override {
    return absl::UnimplementedError("revoke_key_mapping not used in tests");
  }

  void update_local_endpoint(std::string, std::string, uint32_t, uint32_t) override {}
};

struct MetadataGatewayHarness {
  MetadataGatewayHarness()
      : options(MakeTestOptions()),
        context(options),
        replica_runtime(ReplicaRuntime::Config{.runtime_context = &context}) {
    REQUIRE(context.start().ok());
    MetadataGateway::Config cfg{
        .runtime_context = &context,
        .replica_runtime = &replica_runtime,
        .artifact_chunk_bytes = options.artifact_chunk_bytes,
        .pinned_memory_timeout = options.pinned_memory_timeout,
        .replica_factory = {},
    };
    gateway = std::make_unique<MetadataGateway>(cfg);
    client = std::make_shared<TestGlobalStoreClient>();
    gateway->set_client_override(client);
  }

  ~MetadataGatewayHarness() {
    context.shutdown();
  }

  StoreEngineOptions options;
  RuntimeContext context;
  ReplicaRuntime replica_runtime;
  std::unique_ptr<MetadataGateway> gateway;
  std::shared_ptr<TestGlobalStoreClient> client;
};

ReplicaKey CreateCpuReplica(RuntimeContext& context, ReplicaRuntime& runtime, const std::string& artifact_id) {
  constexpr size_t kBytes = 4096;
  auto backing = std::make_shared<std::vector<uint8_t>>(kBytes, 0xAB);
  auto view = std::shared_ptr<const void>(backing, static_cast<const void*>(backing->data()));
  auto config = MakeInlineReplicaConfig(context, artifact_id, DeviceType::CPU, -1, view, kBytes);
  auto replica = runtime.get_or_create_replica(artifact_id, config);
  REQUIRE(replica != nullptr);
  DeviceKey cpu_device{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  return MakeReplicaKey(artifact_id, cpu_device);
}

} // namespace

TEST_CASE("RegistrationBackend commits publish to MetadataGateway", "[registration_backend]") {
  SKIP_IF_NO_CUDA();

  RegistrationBackendHarness harness;
  auto backend = harness.make_backend();

  auto reg = MakeRegistration(/*total_bytes=*/32);
  auto begin_or = backend.begin(reg);
  REQUIRE(begin_or.ok());

  auto commit_or = backend.commit(begin_or->registration_id);
  REQUIRE(commit_or.ok());

  REQUIRE_FALSE(harness.publisher.publications.empty());
  const auto& publication = harness.publisher.publications.back();
  CHECK_FALSE(publication.artifact_id.empty());
  CHECK(publication.size_bytes == reg.total_size_bytes);
  CHECK(publication.device.type == DeviceType::GPU);
  CHECK(publication.device.ordinal == reg.device_id);
}

TEST_CASE("MetadataGateway deduplicates publish contexts", "[metadata_gateway][runtime]") {
  SKIP_IF_NO_CUDA();

  MetadataGatewayHarness harness;
  auto replica_key = CreateCpuReplica(harness.context, harness.replica_runtime, "artifact_ctx");

  IngestionResultEvent event;
  event.request_id = "req-1";
  event.source = tensorcast::store::runtime::IngestionSource::kDisk;
  event.materialize_mode = tensorcast::store::loading::MaterializeMode::LOAD_ONLY;
  event.artifact_id = replica_key.artifact_id;
  event.target_device = replica_key.device;
  event.target_location = MemoryLocation::CPU;
  event.bytes_transferred = 4096;
  event.duration_seconds = 0.01;
  event.status = absl::OkStatus();
  event.replica_key = replica_key;
  event.publish_to_global_store = true;
  event.publish_context_id = "ctx-1";

  harness.gateway->handle_ingestion_result(event);
  CHECK(harness.client->register_calls == 1);

  harness.gateway->handle_ingestion_result(event);
  CHECK(harness.client->register_calls == 1);

  auto dedup_status = harness.gateway->register_replica(replica_key, {}, event.publish_context_id);
  CHECK(dedup_status.ok());
  CHECK(harness.client->register_calls == 1);

  auto second_status = harness.gateway->register_replica(replica_key, {}, "ctx-2");
  CHECK(second_status.ok());
  CHECK(harness.client->register_calls == 2);
}
