// Copyright (c) 2025-2026, TensorCast Team.

#include <array>
#include <chrono>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "catch2/catch_test_macros.hpp"
#include "core/common/memory/host_memory.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/memory_tier_budget.h"
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
using tensorcast::common::memory::set_host_memory_available_override_for_testing;
using tensorcast::store::DeviceKey;
using tensorcast::store::MemoryTierBudget;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::loading::InlineBufferSource;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::replica::Replica;
using tensorcast::store::replica::ReplicaConfig;
using tensorcast::store::runtime::IngestionResultEvent;
using tensorcast::store::runtime::ReplicaPublishState;
using tensorcast::store::runtime::ReplicaRuntime;
using tensorcast::store::runtime::RuntimeContext;
using tensorcast::store::runtime::metadata::ArtifactRegistration;
using tensorcast::store::runtime::metadata::MetadataGateway;
using tensorcast::store::runtime::metadata::RegistrationBackend;
using tensorcast::store::runtime::metadata::RegistrationPlan;
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

  absl::Status update_view_state(const tensorcast::store::components::ViewStateUpdate&) override {
    ++view_updates;
    return absl::OkStatus();
  }

  std::vector<RegistrationPublication> publications;
  int view_updates{0};
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
            .async_runtime = std::make_shared<tensorcast::common::AsyncRuntime>(),
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
    return RegistrationBackend(
        resources,
        std::move(factory),
        1ULL << 20,
        std::chrono::milliseconds(10),
        /*streaming_buffer_chunks=*/16,
        &publisher);
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
      .async_runtime = gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{context.async_runtime()},
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

  void set_register_blocking(bool enabled) {
    absl::MutexLock lock(&register_mu_);
    block_register_ = enabled;
    register_started_ = false;
    allow_register_continue_ = !enabled;
    if (!enabled) {
      register_cv_.SignalAll();
    }
  }

  void wait_for_register_started() {
    absl::MutexLock lock(&register_mu_);
    while (!register_started_) {
      register_cv_.Wait(&register_mu_);
    }
  }

  void unblock_register() {
    absl::MutexLock lock(&register_mu_);
    allow_register_continue_ = true;
    register_cv_.SignalAll();
  }

  absl::Status initialize() override {
    return absl::OkStatus();
  }

  absl::StatusOr<tensorcast::store::components::WorkerRegistrationInfo> register_worker(
      std::string_view,
      std::string_view,
      uint32_t,
      uint32_t,
      uint64_t,
      uint64_t,
      bool,
      std::string_view,
      std::string_view,
      uint64_t) override {
    return absl::UnimplementedError("register_worker not used in tests");
  }

  absl::StatusOr<tensorcast::global_store::v1::WorkerHeartbeatResponse> send_heartbeat_enhanced(
      std::string_view,
      uint64_t,
      bool,
      uint64_t,
      std::string_view,
      const std::vector<std::string>&,
      int64_t,
      tensorcast::global_store::v1::ConnectionStatus,
      const tensorcast::store::components::RpcOptions&,
      std::string_view,
      uint64_t) override {
    return absl::UnimplementedError("send_heartbeat_enhanced not used in tests");
  }

  absl::Status unregister_worker(std::string_view, bool) override {
    return absl::UnimplementedError("unregister_worker not used in tests");
  }

  absl::StatusOr<std::vector<tensorcast::store::components::ActiveWorkerInfo>> list_active_workers(
      bool,
      uint64_t,
      const tensorcast::store::components::RpcOptions&) override {
    return absl::UnimplementedError("list_active_workers not used in tests");
  }

  absl::StatusOr<std::string> register_replica(
      std::string_view artifact_id,
      std::string_view,
      const DeviceKey&,
      MemoryLocation,
      uint64_t,
      uint32_t,
      std::optional<std::string_view>) override {
    {
      absl::MutexLock lock(&register_mu_);
      if (block_register_) {
        register_started_ = true;
        register_cv_.SignalAll();
        while (!allow_register_continue_) {
          register_cv_.Wait(&register_mu_);
        }
      }
    }
    registered_artifacts.emplace_back(artifact_id);
    ++register_calls;
    if (!next_register_status.ok()) {
      return next_register_status;
    }
    return absl::StrCat("rep-", register_calls);
  }

  absl::Status record_view_residency(std::string_view, std::string_view, uint64_t, std::optional<std::string_view>)
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
      const std::optional<std::string>&,
      std::optional<std::string_view>,
      const std::optional<tensorcast::common::v1::ArtifactDescriptor>&) override {
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

  absl::StatusOr<bool> mark_replica_unavailable(
      std::string_view,
      std::string_view,
      std::optional<std::string_view>,
      std::optional<std::string_view>) override {
    return true;
  }

  absl::StatusOr<tensorcast::store::components::ReplicaDrainStatus> wait_replica_drain(
      std::string_view,
      uint32_t,
      std::optional<std::string_view>) override {
    tensorcast::store::components::ReplicaDrainStatus out;
    out.drained = true;
    out.current_requests = 0;
    return out;
  }

  absl::Status update_artifact_view_state(const tensorcast::store::components::ViewStateUpdate&) override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<tensorcast::store::components::ViewInfo>> list_views(std::string_view) override {
    return absl::UnimplementedError("list_views not used in tests");
  }

  absl::StatusOr<tensorcast::global_store::v1::AssemblyLayoutBinding> get_assembly_layout_binding(
      std::string_view) override {
    return absl::UnimplementedError("get_assembly_layout_binding not used in tests");
  }

  absl::StatusOr<tensorcast::layout::v1::LayoutSpecRecord> get_layout_spec(std::string_view) override {
    return absl::UnimplementedError("get_layout_spec not used in tests");
  }

  absl::Status attach_layout_to_artifact(std::string_view, std::string_view) override {
    return absl::UnimplementedError("attach_layout_to_artifact not used in tests");
  }

  absl::StatusOr<std::vector<std::string>> list_artifact_layouts(std::string_view) override {
    return absl::UnimplementedError("list_artifact_layouts not used in tests");
  }

  absl::StatusOr<tensorcast::global_store::v1::WriteTensorProofCommitmentsResponse> write_tensor_proof_commitments(
      const tensorcast::global_store::v1::WriteTensorProofCommitmentsRequest&) override {
    return absl::UnimplementedError("write_tensor_proof_commitments not used in tests");
  }

  absl::StatusOr<tensorcast::global_store::v1::CheckProofCommitmentsMatchResponse> check_proof_commitments_match(
      const tensorcast::global_store::v1::CheckProofCommitmentsMatchRequest&) override {
    return absl::UnimplementedError("check_proof_commitments_match not used in tests");
  }

  absl::StatusOr<tensorcast::global_store::v1::AssemblyRuntimePolicy> get_assembly_runtime_policy(
      std::string_view) override {
    return absl::UnimplementedError("get_assembly_runtime_policy not used in tests");
  }

  absl::StatusOr<tensorcast::operation::v1::AcquireOperationLeaseResponse> acquire_operation_lease(
      const tensorcast::operation::v1::AcquireOperationLeaseRequest&) override {
    return absl::UnimplementedError("acquire_operation_lease not used in tests");
  }

  absl::StatusOr<tensorcast::operation::v1::KeepaliveOperationLeaseResponse> keepalive_operation_lease(
      const tensorcast::operation::v1::KeepaliveOperationLeaseRequest&) override {
    return absl::UnimplementedError("keepalive_operation_lease not used in tests");
  }

  absl::StatusOr<tensorcast::operation::v1::ReleaseOperationLeaseResponse> release_operation_lease(
      const tensorcast::operation::v1::ReleaseOperationLeaseRequest&) override {
    return absl::UnimplementedError("release_operation_lease not used in tests");
  }

  absl::StatusOr<tensorcast::operation::v1::GetOperationResponse> get_operation(
      const tensorcast::operation::v1::GetOperationRequest&) override {
    return absl::UnimplementedError("get_operation not used in tests");
  }

  absl::Status update_operation(const tensorcast::operation::v1::UpdateOperationRequest&) override {
    return absl::UnimplementedError("update_operation not used in tests");
  }

  absl::StatusOr<tensorcast::store::components::ArtifactBinding> get_artifact_binding(std::string_view) override {
    return absl::UnimplementedError("get_artifact_binding not used in tests");
  }

  absl::StatusOr<tensorcast::store::components::ArtifactBindingResult> upsert_artifact_binding(
      const tensorcast::store::components::ArtifactBinding&) override {
    return absl::UnimplementedError("upsert_artifact_binding not used in tests");
  }

  absl::StatusOr<tensorcast::store::components::TransportSession> request_replica_transport(
      std::string_view,
      std::string_view,
      std::string_view,
      uint32_t,
      const DeviceKey&,
      uint32_t,
      const std::optional<tensorcast::store::components::TransportSchedulingGroupHint>&,
      std::string_view,
      std::string_view) override {
    return absl::UnimplementedError("request_replica_transport not used in tests");
  }

  absl::StatusOr<tensorcast::store::components::TransportSession> request_view_transport(
      std::string_view,
      std::string_view,
      std::string_view,
      std::string_view,
      uint32_t,
      const DeviceKey&,
      uint32_t,
      const std::optional<tensorcast::store::components::TransportSchedulingGroupHint>&,
      std::string_view,
      std::string_view) override {
    return absl::UnimplementedError("request_view_transport not used in tests");
  }

  absl::Status complete_replica_transport(
      std::string_view,
      tensorcast::store::components::TransportCompletionOutcome,
      std::string_view) override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<tensorcast::store::components::RemoteReplicaInfo>> get_artifact_replicas(
      std::string_view,
      std::optional<std::string_view>) override {
    return absl::UnimplementedError("get_artifact_replicas not used in tests");
  }

  absl::StatusOr<std::vector<tensorcast::store::components::ChunkLocationInfo>> query_chunk_locations(
      std::string_view,
      const std::vector<uint32_t>&) override {
    return absl::UnimplementedError("query_chunk_locations not used in tests");
  }

  absl::StatusOr<tensorcast::store::components::StateSyncResult> reconcile_worker_state(
      std::string_view,
      std::string_view,
      const std::vector<tensorcast::common::v1::ReplicaInfo>&,
      bool,
      const tensorcast::store::components::StateSyncToken&,
      const tensorcast::store::components::RpcOptions&) override {
    return absl::UnimplementedError("reconcile_worker_state not used in tests");
  }

  absl::StatusOr<tensorcast::store::components::AcquireShardHomeLeaseResult> acquire_shard_home_lease(
      uint64_t,
      std::string_view,
      uint64_t,
      const tensorcast::store::components::RpcOptions&) override {
    return absl::UnimplementedError("acquire_shard_home_lease not used in tests");
  }

  absl::StatusOr<tensorcast::store::components::ShardHomeLeaseDescriptor> keepalive_shard_home_lease(
      std::string_view,
      uint64_t,
      const tensorcast::store::components::RpcOptions&) override {
    return absl::UnimplementedError("keepalive_shard_home_lease not used in tests");
  }

  absl::StatusOr<std::vector<tensorcast::store::components::ShardHomeLeaseKeepaliveOutcome>>
  batch_keepalive_shard_home_leases(
      const std::vector<tensorcast::store::components::ShardHomeLeaseKeepaliveInput>&,
      uint64_t,
      const tensorcast::store::components::RpcOptions&) override {
    return absl::UnimplementedError("batch_keepalive_shard_home_leases not used in tests");
  }

  absl::StatusOr<bool> release_shard_home_lease(std::string_view, const tensorcast::store::components::RpcOptions&)
      override {
    return absl::UnimplementedError("release_shard_home_lease not used in tests");
  }

  absl::StatusOr<tensorcast::store::components::ShardHomeRouteInfo> get_shard_home_lease(
      uint64_t,
      const tensorcast::store::components::RpcOptions&) override {
    return absl::UnimplementedError("get_shard_home_lease not used in tests");
  }

  absl::StatusOr<std::vector<tensorcast::store::components::ShardHomeRouteInfo>> batch_get_shard_home_leases(
      const std::vector<uint64_t>&,
      const tensorcast::store::components::RpcOptions&) override {
    return absl::UnimplementedError("batch_get_shard_home_leases not used in tests");
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

  absl::StatusOr<tensorcast::store::components::ViewMetadata> get_view_metadata(std::string_view, std::string_view)
      override {
    return absl::UnimplementedError("get_view_metadata not used in tests");
  }

  absl::Status upsert_key_mapping(std::string_view, std::string_view, absl::Duration) override {
    return absl::UnimplementedError("upsert_key_mapping not used in tests");
  }

  absl::StatusOr<std::string> get_cluster_id() override {
    return absl::UnimplementedError("get_cluster_id not used in tests");
  }

  absl::Status upsert_artifact_disk_location(
      std::string_view,
      std::string_view,
      std::string_view,
      tensorcast::global_store::v1::DiskLocationKind,
      bool) override {
    return absl::UnimplementedError("upsert_artifact_disk_location not used in tests");
  }

  absl::StatusOr<std::vector<tensorcast::store::components::ArtifactDiskLocation>> list_artifact_disk_locations(
      std::string_view,
      bool) override {
    return absl::UnimplementedError("list_artifact_disk_locations not used in tests");
  }

  absl::StatusOr<tensorcast::store::components::KeyMappingSwapResult> swap_key_mapping(
      std::string_view,
      std::string_view,
      std::optional<std::string_view>,
      std::optional<uint64_t>) override {
    return absl::UnimplementedError("swap_key_mapping not used in tests");
  }

  absl::Status revoke_key_mapping(std::string_view) override {
    return absl::UnimplementedError("revoke_key_mapping not used in tests");
  }

  absl::StatusOr<tensorcast::store::components::PlacementPlanResult> plan_placement(
      std::string_view,
      tensorcast::global_store::v1::PlacementPolicy,
      const std::vector<tensorcast::store::components::PlacementShardSpec>&,
      std::string_view) override {
    return absl::UnimplementedError("plan_placement not used in tests");
  }

  absl::Status report_persistence_status(const tensorcast::store::components::PersistenceReport&) override {
    return absl::UnimplementedError("report_persistence_status not used in tests");
  }

  void update_local_endpoint(std::string, std::string, uint32_t, uint32_t) override {}

 private:
  mutable absl::Mutex register_mu_;
  bool block_register_ ABSL_GUARDED_BY(register_mu_){false};
  bool register_started_ ABSL_GUARDED_BY(register_mu_){false};
  bool allow_register_continue_ ABSL_GUARDED_BY(register_mu_){false};
  absl::CondVar register_cv_;
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
        .promotion_manager = nullptr,
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
  REQUIRE(commit_or->verified_content_descriptor.has_value());
  REQUIRE(commit_or->verification_record.has_value());
  CHECK(
      commit_or->verified_content_descriptor->content_identity.semantic_layout_identity.kind ==
      tensorcast::store::runtime::ingestion::SemanticLayoutKind::kCanonicalIndexDigest);
  CHECK(
      commit_or->verified_content_descriptor->content_identity.semantic_layout_identity.value ==
      commit_or->index_multihash);
  CHECK(commit_or->verified_content_descriptor->content_identity.logical_size_bytes == reg.total_size_bytes);
  CHECK(commit_or->verified_content_descriptor->content_identity.digest_alg == "multihash");
  CHECK(commit_or->verified_content_descriptor->content_identity.digest_bytes == commit_or->data_multihash);
  CHECK(
      commit_or->verification_record->verification_method ==
      tensorcast::store::runtime::ingestion::VerificationMethod::kRegistrationCommit);

  REQUIRE_FALSE(harness.publisher.publications.empty());
  const auto& publication = harness.publisher.publications.back();
  CHECK_FALSE(publication.artifact_id.empty());
  CHECK(publication.size_bytes == reg.total_size_bytes);
  CHECK(publication.device.type == DeviceType::GPU);
  CHECK(publication.device.ordinal == reg.device_id);
}

TEST_CASE(
    "RegistrationBackend stable begin guard rejects when host memory is too low without reusable stable capacity",
    "[registration_backend][memory_guard]") {
  RegistrationBackendHarness harness;
  set_host_memory_available_override_for_testing(16ULL * 1024ULL * 1024ULL * 1024ULL);
  auto reset_override = absl::MakeCleanup([] { set_host_memory_available_override_for_testing(std::nullopt); });

  auto backend = harness.make_backend();
  auto reg = MakeRegistration(/*total_bytes=*/64ULL * 1024ULL * 1024ULL * 1024ULL);
  reg.plan = RegistrationPlan::kStableDram;
  reg.schema_version = "invalid-schema";

  auto begin_or = backend.begin(reg);
  REQUIRE_FALSE(begin_or.ok());
  CHECK(begin_or.status().code() == absl::StatusCode::kResourceExhausted);
}

TEST_CASE(
    "RegistrationBackend stable begin guard accounts for reusable stable capacity before checking host headroom",
    "[registration_backend][memory_guard]") {
  RegistrationBackendHarness harness;
  auto budget = std::make_shared<MemoryTierBudget>(
      /*stable_total_bytes=*/256ULL * 1024ULL * 1024ULL * 1024ULL,
      /*preemptible_total_bytes=*/0);
  REQUIRE(budget->try_acquire_stable(64ULL * 1024ULL * 1024ULL * 1024ULL).ok());
  harness.resources.memory_tier_budget = budget;

  set_host_memory_available_override_for_testing(16ULL * 1024ULL * 1024ULL * 1024ULL);
  auto reset_override = absl::MakeCleanup([] { set_host_memory_available_override_for_testing(std::nullopt); });

  auto backend = harness.make_backend();
  auto reg = MakeRegistration(/*total_bytes=*/64ULL * 1024ULL * 1024ULL * 1024ULL);
  reg.plan = RegistrationPlan::kStableDram;
  reg.schema_version = "invalid-schema";

  auto begin_or = backend.begin(reg);
  REQUIRE_FALSE(begin_or.ok());
  CHECK(begin_or.status().code() == absl::StatusCode::kInvalidArgument);
}

TEST_CASE("RegistrationBackend commit cleans pending mem_reg alias from ReplicaRegistry", "[registration_backend]") {
  SKIP_IF_NO_CUDA();

  RegistrationBackendHarness harness;
  auto backend = harness.make_backend();

  auto reg = MakeRegistration(/*total_bytes=*/32);
  reg.client_artifact_id = "cgid:registration-backend-alias-cleanup";
  auto begin_or = backend.begin(reg);
  REQUIRE(begin_or.ok());
  CHECK(harness.replica_registry->size() == 1);
  CHECK(harness.replica_registry->find_by_artifact(reg.artifact_id).size() == 1);

  auto commit_or = backend.commit(begin_or->registration_id);
  REQUIRE(commit_or.ok());

  CHECK(harness.replica_registry->size() == 1);
  CHECK(harness.replica_registry->find_by_artifact(reg.artifact_id).empty());
  CHECK(harness.replica_registry->find_by_artifact(commit_or->artifact_id).size() == 1);
}

TEST_CASE("RegistrationBackend abort cleans pending mem_reg alias from ReplicaRegistry", "[registration_backend]") {
  SKIP_IF_NO_CUDA();

  RegistrationBackendHarness harness;
  auto backend = harness.make_backend();

  auto reg = MakeRegistration(/*total_bytes=*/32);
  auto begin_or = backend.begin(reg);
  REQUIRE(begin_or.ok());
  CHECK(harness.replica_registry->size() == 1);

  auto abort_status = backend.abort(begin_or->registration_id);
  REQUIRE(abort_status.ok());
  CHECK(harness.replica_registry->size() == 0);
}

TEST_CASE(
    "RegistrationBackend stable_dram stage_on_gpu=false supports streamed CPU ingestion",
    "[registration_backend][stable_dram]") {
  RegistrationBackendHarness harness;
  auto backend = harness.make_backend();

  auto reg = MakeRegistration(/*total_bytes=*/16);
  reg.plan = RegistrationPlan::kStableDram;
  reg.stable_dram.stage_on_gpu = false;
  reg.stable_dram.release_gpu_on_commit = false;

  auto begin_or = backend.begin(reg);
  REQUIRE(begin_or.ok());

  auto gpu_ptr_or = backend.get_registration_gpu_ptr(begin_or->registration_id);
  REQUIRE_FALSE(gpu_ptr_or.ok());

  std::array<std::byte, 16> payload{};
  for (size_t index = 0; index < payload.size(); ++index) {
    payload[index] = static_cast<std::byte>(index + 1);
  }
  auto ingest_status =
      backend.ingest_registration_chunk(begin_or->registration_id, /*offset=*/0, absl::MakeConstSpan(payload));
  REQUIRE(ingest_status.ok());

  auto commit_or = backend.commit(begin_or->registration_id);
  REQUIRE(commit_or.ok());
  CHECK(commit_or->device.type == DeviceType::CPU);
  CHECK(commit_or->size_bytes == reg.total_size_bytes);
}

TEST_CASE(
    "RegistrationBackend stable_dram stage_on_gpu=false supports cpu_memfd written-range ingestion",
    "[registration_backend][stable_dram][cpu_memfd]") {
  RegistrationBackendHarness harness;
  harness.resources.cpu_shared_memory_enabled = true;
  auto backend = harness.make_backend();

  auto reg = MakeRegistration(/*total_bytes=*/16);
  reg.plan = RegistrationPlan::kStableDram;
  reg.stable_dram.stage_on_gpu = false;
  reg.stable_dram.release_gpu_on_commit = false;

  auto begin_or = backend.begin(reg);
  REQUIRE(begin_or.ok());

  auto memfd_or = backend.get_registration_cpu_memfd_info(begin_or->registration_id);
  REQUIRE(memfd_or.ok());
  CHECK(memfd_or->fd >= 0);
  CHECK(memfd_or->size_bytes >= reg.total_size_bytes);

  auto tail_status = backend.ingest_registration_written_range(begin_or->registration_id, /*offset=*/8, /*length=*/8);
  REQUIRE(tail_status.ok());

  auto head_status = backend.ingest_registration_written_range(begin_or->registration_id, /*offset=*/0, /*length=*/8);
  REQUIRE(head_status.ok());

  auto commit_or = backend.commit(begin_or->registration_id);
  REQUIRE(commit_or.ok());
  CHECK(commit_or->device.type == DeviceType::CPU);
}

TEST_CASE(
    "RegistrationBackend stable_dram written-range ingestion rejects overlap",
    "[registration_backend][stable_dram][cpu_memfd]") {
  RegistrationBackendHarness harness;
  harness.resources.cpu_shared_memory_enabled = true;
  auto backend = harness.make_backend();

  auto reg = MakeRegistration(/*total_bytes=*/16);
  reg.plan = RegistrationPlan::kStableDram;
  reg.stable_dram.stage_on_gpu = false;
  reg.stable_dram.release_gpu_on_commit = false;

  auto begin_or = backend.begin(reg);
  REQUIRE(begin_or.ok());

  auto first_status = backend.ingest_registration_written_range(begin_or->registration_id, /*offset=*/0, /*length=*/8);
  REQUIRE(first_status.ok());

  auto overlap_status =
      backend.ingest_registration_written_range(begin_or->registration_id, /*offset=*/4, /*length=*/8);
  REQUIRE_FALSE(overlap_status.ok());
  CHECK(overlap_status.code() == absl::StatusCode::kFailedPrecondition);
}

TEST_CASE(
    "RegistrationBackend stable_dram streamed ingestion propagates cpu_shared_memory_enabled to replicas",
    "[registration_backend][stable_dram][cpu_memfd]") {
  RegistrationBackendHarness harness;
  harness.resources.cpu_shared_memory_enabled = true;
  auto backend = harness.make_backend();

  auto reg = MakeRegistration(/*total_bytes=*/16);
  reg.plan = RegistrationPlan::kStableDram;
  reg.stable_dram.stage_on_gpu = false;
  reg.stable_dram.release_gpu_on_commit = false;

  auto begin_or = backend.begin(reg);
  REQUIRE(begin_or.ok());

  std::array<std::byte, 16> payload{};
  auto ingest_status =
      backend.ingest_registration_chunk(begin_or->registration_id, /*offset=*/0, absl::MakeConstSpan(payload));
  REQUIRE(ingest_status.ok());

  auto commit_or = backend.commit(begin_or->registration_id);
  REQUIRE(commit_or.ok());

  auto keys = harness.replica_registry->find_by_artifact(commit_or->artifact_id);
  REQUIRE(keys.size() == 1);
  auto replica_or = harness.replica_registry->find(keys.front());
  REQUIRE(replica_or.ok());

  const auto& allocation_key = replica_or.value()->replica_key();
  CHECK_FALSE(allocation_key.artifact_id.empty());
  auto uma = replica_or.value()->get_memory_manager().memory_authority();
  REQUIRE(uma != nullptr);
  auto region_or = uma->get_cpu_memfd_region(allocation_key);
  REQUIRE(region_or.ok());
  CHECK(region_or->fd >= 0);
  CHECK(region_or->size_bytes >= reg.total_size_bytes);
}

TEST_CASE(
    "RegistrationBackend stable_dram streamed ingestion accepts out-of-order chunks and enforces full coverage",
    "[registration_backend][stable_dram]") {
  RegistrationBackendHarness harness;
  auto backend = harness.make_backend();

  auto reg = MakeRegistration(/*total_bytes=*/16);
  reg.plan = RegistrationPlan::kStableDram;
  reg.stable_dram.stage_on_gpu = false;
  reg.stable_dram.release_gpu_on_commit = false;

  auto begin_or = backend.begin(reg);
  REQUIRE(begin_or.ok());

  std::array<std::byte, 8> chunk{};
  auto tail_status =
      backend.ingest_registration_chunk(begin_or->registration_id, /*offset=*/8, absl::MakeConstSpan(chunk));
  REQUIRE(tail_status.ok());

  auto premature_commit_or = backend.commit(begin_or->registration_id);
  REQUIRE_FALSE(premature_commit_or.ok());
  CHECK(premature_commit_or.status().code() == absl::StatusCode::kFailedPrecondition);

  auto head_status =
      backend.ingest_registration_chunk(begin_or->registration_id, /*offset=*/0, absl::MakeConstSpan(chunk));
  REQUIRE(head_status.ok());

  auto commit_or = backend.commit(begin_or->registration_id);
  REQUIRE(commit_or.ok());
}

TEST_CASE(
    "RegistrationBackend stable_dram streamed ingestion rejects overlapping chunks",
    "[registration_backend][stable_dram]") {
  RegistrationBackendHarness harness;
  auto backend = harness.make_backend();

  auto reg = MakeRegistration(/*total_bytes=*/16);
  reg.plan = RegistrationPlan::kStableDram;
  reg.stable_dram.stage_on_gpu = false;
  reg.stable_dram.release_gpu_on_commit = false;

  auto begin_or = backend.begin(reg);
  REQUIRE(begin_or.ok());

  std::array<std::byte, 8> chunk{};
  auto first_status =
      backend.ingest_registration_chunk(begin_or->registration_id, /*offset=*/0, absl::MakeConstSpan(chunk));
  REQUIRE(first_status.ok());

  auto overlap_status =
      backend.ingest_registration_chunk(begin_or->registration_id, /*offset=*/4, absl::MakeConstSpan(chunk));
  REQUIRE_FALSE(overlap_status.ok());
  CHECK(overlap_status.code() == absl::StatusCode::kFailedPrecondition);
}

TEST_CASE(
    "RegistrationBackend stable_dram streamed ingestion supports concurrent non-overlapping chunks",
    "[registration_backend][stable_dram]") {
  RegistrationBackendHarness harness;
  auto backend = harness.make_backend();

  constexpr uint64_t kChunkBytes = 4096;
  constexpr uint64_t kChunkCount = 4;
  auto reg = MakeRegistration(/*total_bytes=*/kChunkBytes * kChunkCount);
  reg.plan = RegistrationPlan::kStableDram;
  reg.stable_dram.stage_on_gpu = false;
  reg.stable_dram.release_gpu_on_commit = false;

  auto begin_or = backend.begin(reg);
  REQUIRE(begin_or.ok());

  std::array<std::array<std::byte, kChunkBytes>, kChunkCount> chunks{};
  for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
    for (size_t byte_index = 0; byte_index < chunks[chunk_index].size(); ++byte_index) {
      chunks[chunk_index][byte_index] = static_cast<std::byte>(chunk_index + 1);
    }
  }
  const std::array<uint64_t, kChunkCount> offsets = {kChunkBytes * 2, 0, kChunkBytes * 3, kChunkBytes};

  std::array<absl::Status, kChunkCount> ingest_status;
  std::vector<std::thread> workers;
  workers.reserve(kChunkCount);
  for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
    workers.emplace_back([&, chunk_index]() {
      ingest_status[chunk_index] = backend.ingest_registration_chunk(
          begin_or->registration_id, offsets[chunk_index], absl::MakeConstSpan(chunks[chunk_index]));
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (const auto& status : ingest_status) {
    REQUIRE(status.ok());
  }

  auto commit_or = backend.commit(begin_or->registration_id);
  REQUIRE(commit_or.ok());

  auto keys = harness.replica_registry->find_by_artifact(commit_or->artifact_id);
  REQUIRE(keys.size() == 1);
  auto replica_or = harness.replica_registry->find(keys.front());
  REQUIRE(replica_or.ok());
  const auto cpu_ptrs = replica_or.value()->get_memory_manager().get_pointer(MemoryLocation::CPU);
  REQUIRE(cpu_ptrs.size() == 1);
  REQUIRE(cpu_ptrs[0] != nullptr);

  const auto* bytes = static_cast<const std::byte*>(cpu_ptrs[0]);
  for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
    const uint64_t chunk_offset = offsets[chunk_index];
    for (size_t byte_index = 0; byte_index < chunks[chunk_index].size(); ++byte_index) {
      const auto actual = bytes[static_cast<size_t>(chunk_offset) + byte_index];
      const auto expected = chunks[chunk_index][byte_index];
      REQUIRE(actual == expected);
    }
  }
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

TEST_CASE("MetadataGateway suppresses concurrent publish context duplicates", "[metadata_gateway][runtime]") {
  SKIP_IF_NO_CUDA();

  MetadataGatewayHarness harness;
  auto replica_key = CreateCpuReplica(harness.context, harness.replica_runtime, "artifact_ctx_inflight");

  harness.client->set_register_blocking(true);

  absl::Status first_status = absl::UnknownError("not-run");
  std::thread first([&]() { first_status = harness.gateway->register_replica(replica_key, {}, "ctx-inflight"); });

  harness.client->wait_for_register_started();
  auto second_status = harness.gateway->register_replica(replica_key, {}, "ctx-inflight");
  CHECK(second_status.ok());

  harness.client->unblock_register();
  first.join();
  CHECK(first_status.ok());
  CHECK(harness.client->register_calls == 1);
}

TEST_CASE("MetadataGateway retries publish context after client reconnection", "[metadata_gateway][runtime]") {
  SKIP_IF_NO_CUDA();

  MetadataGatewayHarness harness;
  auto replica_key = CreateCpuReplica(harness.context, harness.replica_runtime, "artifact_ctx_retry_client");

  harness.client->connected = false;
  auto first_status = harness.gateway->register_replica(replica_key, {}, "ctx-retry-client");
  CHECK_FALSE(first_status.ok());
  CHECK(first_status.code() == absl::StatusCode::kFailedPrecondition);
  CHECK(harness.client->register_calls == 0);

  harness.client->connected = true;
  auto second_status = harness.gateway->register_replica(replica_key, {}, "ctx-retry-client");
  CHECK(second_status.ok());
  CHECK(harness.client->register_calls == 1);
}

TEST_CASE("MetadataGateway retries publish context after replica size lookup failure", "[metadata_gateway][runtime]") {
  SKIP_IF_NO_CUDA();

  MetadataGatewayHarness harness;
  DeviceKey cpu_device{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  auto missing_replica_key = MakeReplicaKey("artifact_ctx_retry_size", cpu_device);

  auto first_status = harness.gateway->register_replica(missing_replica_key, {}, "ctx-retry-size");
  CHECK_FALSE(first_status.ok());
  CHECK(first_status.code() == absl::StatusCode::kNotFound);
  CHECK(harness.client->register_calls == 0);

  auto replica_key = CreateCpuReplica(harness.context, harness.replica_runtime, "artifact_ctx_retry_size");
  auto second_status = harness.gateway->register_replica(replica_key, {}, "ctx-retry-size");
  CHECK(second_status.ok());
  CHECK(harness.client->register_calls == 1);
}

TEST_CASE(
    "MetadataGateway retries publish context after canonical artifact validation failure",
    "[metadata_gateway][runtime]") {
  SKIP_IF_NO_CUDA();

  MetadataGatewayHarness harness;
  auto replica_key = CreateCpuReplica(harness.context, harness.replica_runtime, "artifact_ctx_retry_canonical");

  auto first_status = harness.gateway->register_replica(replica_key, "artifact-non-canonical", "ctx-retry-canonical");
  CHECK_FALSE(first_status.ok());
  CHECK(first_status.code() == absl::StatusCode::kInvalidArgument);
  CHECK(harness.client->register_calls == 0);

  auto second_status =
      harness.gateway->register_replica(replica_key, "mi2:artifact_ctx_retry_canonical", "ctx-retry-canonical");
  CHECK(second_status.ok());
  CHECK(harness.client->register_calls == 1);
  CHECK_FALSE(harness.client->registered_artifacts.empty());
  CHECK(harness.client->registered_artifacts.back() == "mi2:artifact_ctx_retry_canonical");
}

TEST_CASE("MetadataGateway keeps publish pending on registration failure", "[metadata_gateway][runtime]") {
  SKIP_IF_NO_CUDA();

  MetadataGatewayHarness harness;
  auto replica_key = CreateCpuReplica(harness.context, harness.replica_runtime, "artifact_publish_state");

  IngestionResultEvent event;
  event.request_id = "req-2";
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
  event.publish_context_id = "ctx-publish-state";

  harness.replica_runtime.record_ingestion_result(event);
  CHECK(harness.replica_runtime.get_replica_publish_state(replica_key) == ReplicaPublishState::kPublishPending);

  harness.client->next_register_status = absl::UnavailableError("GS unavailable");
  harness.gateway->handle_ingestion_result(event);
  CHECK(harness.replica_runtime.get_replica_publish_state(replica_key) == ReplicaPublishState::kPublishPending);

  harness.client->next_register_status = absl::OkStatus();
  harness.gateway->handle_ingestion_result(event);
  CHECK(harness.replica_runtime.get_replica_publish_state(replica_key) == ReplicaPublishState::kPublished);
}
