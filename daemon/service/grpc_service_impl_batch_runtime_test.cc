// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "absl/log/globals.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "core/communicator/routing/adapter.h"
#include "core/communicator/routing/routing_context.h"
#include "core/communicator/topology/topology.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_ipc.h"
#include "core/store/components/endpoint_id.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/dataplane/contracts/inline_buffer_loader.h"
#include "core/store/replica/types/direct_write_grant.h"
#include "core/store/store_engine.h"
#include "core/store/testing/global_store_client_stub.h"
#include "core/store/testing/recording_global_store_client.h"
#include "core/testing/test_helpers.h"
#include "daemon/service/artifact_profile_registry.h"
#include "daemon/service/body_backing_manager.h"
#include "daemon/service/byte_artifact_body_handle.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/routed_authority_wire.h"
#include "daemon/state/worker_directory_cache.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace {

using tensorcast::communicator::routing::EndpointBinding;
using tensorcast::communicator::routing::PcieAdapter;
using tensorcast::communicator::routing::RoutingContext;
using tensorcast::communicator::topology::Endpoint;
using tensorcast::communicator::topology::EndpointKind;
using tensorcast::communicator::topology::EndpointType;
using tensorcast::communicator::topology::Link;
using tensorcast::communicator::topology::LinkType;
using tensorcast::communicator::topology::Pool;
using tensorcast::communicator::topology::PoolType;
using tensorcast::communicator::topology::Topology;
using tensorcast::daemon::DaemonOptions;
using tensorcast::daemon::DaemonServiceHarness;
using tensorcast::daemon::v2::BatchExistsRequest;
using tensorcast::daemon::v2::BatchExistsResponse;
using tensorcast::daemon::v2::BatchGetIntoRegionRequest;
using tensorcast::daemon::v2::BatchGetIntoRegionResponse;
using tensorcast::daemon::v2::BatchItemStatus;
using tensorcast::daemon::v2::BatchPutIfAbsentFromRegionRequest;
using tensorcast::daemon::v2::BatchPutIfAbsentFromRegionResponse;
using tensorcast::daemon::v2::DeviceType;
using tensorcast::daemon::v2::HomeBatchExistsRequest;
using tensorcast::daemon::v2::HomeBatchExistsResponse;
using tensorcast::daemon::v2::HomeBatchGetRequest;
using tensorcast::daemon::v2::HomeBatchGetResponse;
using tensorcast::daemon::v2::HomeBatchPutIfAbsentRequest;
using tensorcast::daemon::v2::HomeBatchPutIfAbsentResponse;
using tensorcast::daemon::v2::MaterializeReplicaRequest;
using tensorcast::daemon::v2::MaterializeReplicaResponse;
using tensorcast::daemon::v2::RouteAuthorityStageRequest;
using tensorcast::daemon::v2::RouteAuthorityStageResponse;

constexpr const char* kDaemonId = "daemon-batch-test";

tensorcast::store::DeviceKey cpu_device() {
  return tensorcast::store::DeviceKey{.type = tensorcast::DeviceType::CPU, .ordinal = -1, .uuid = ""};
}

std::size_t count_cpu_replicas(const tensorcast::store::StoreEngine& engine) {
  return engine.list_device_replicas(cpu_device()).size();
}

static tensorcast::store::StoreEngineOptions make_opts_basic() {
  tensorcast::store::StoreEngineOptions opts;
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  return opts;
}

static DaemonOptions make_daemon_options() {
  DaemonOptions opts;
  opts.storage_path = std::filesystem::temp_directory_path();
  opts.daemon_id = kDaemonId;
  opts.capability_tokens.active.version = 1;
  opts.capability_tokens.active.secret = "batch-runtime-secret";
  return opts;
}

static std::filesystem::path make_test_storage_root(std::string_view name) {
  const char* env = std::getenv("TEST_TMPDIR");
  const auto root = (env && *env) ? std::filesystem::path(env) / std::string(name)
                                  : std::filesystem::temp_directory_path() / std::string(name);
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  return root;
}

static tensorcast::daemon::v2::StorePolicy make_shared_disk_policy() {
  tensorcast::daemon::v2::StorePolicy policy;
  auto* must = policy.add_must();
  must->set_tier(tensorcast::daemon::v2::POLICY_TIER_SHARED_DISK);
  must->set_scope(tensorcast::daemon::v2::POLICY_SCOPE_ANY);
  must->set_min_replicas(1);
  return policy;
}

static std::unique_ptr<DaemonServiceHarness> make_harness(
    const std::shared_ptr<tensorcast::store::StoreEngine>& engine,
    const DaemonOptions& options,
    std::shared_ptr<tensorcast::store::components::IGlobalStoreClient> global_store_client = nullptr) {
  auto harness_or = DaemonServiceHarness::create(engine, options, nullptr, std::move(global_store_client));
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

class WorkerDirectoryTestGlobalStoreClient final : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  std::vector<tensorcast::store::components::ActiveWorkerInfo> active_workers;

  absl::StatusOr<std::vector<tensorcast::store::components::ActiveWorkerInfo>> list_active_workers(
      bool,
      uint64_t required_capability_flags,
      const tensorcast::store::components::RpcOptions&) override {
    std::vector<tensorcast::store::components::ActiveWorkerInfo> filtered;
    for (const auto& worker : active_workers) {
      if ((worker.capability_flags & required_capability_flags) != required_capability_flags) {
        continue;
      }
      filtered.push_back(worker);
    }
    return filtered;
  }
};

class CollectingLogSink final : public absl::LogSink {
 public:
  void Send(const absl::LogEntry& entry) override {
    absl::MutexLock lock(&mu_);
    messages_.push_back(std::string(entry.text_message()));
  }

  bool Contains(std::string_view needle) const {
    absl::MutexLock lock(&mu_);
    for (const auto& message : messages_) {
      if (message.find(needle) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  bool ContainsAll(std::initializer_list<std::string_view> needles) const {
    absl::MutexLock lock(&mu_);
    for (const auto needle : needles) {
      bool found = false;
      for (const auto& message : messages_) {
        if (message.find(needle) != std::string::npos) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
    return true;
  }

 private:
  mutable absl::Mutex mu_;
  std::vector<std::string> messages_ ABSL_GUARDED_BY(mu_);
};

class ScopedCollectingLogSink {
 public:
  explicit ScopedCollectingLogSink(CollectingLogSink& sink) : sink_(sink) {
    absl::AddLogSink(&sink_);
  }

  ~ScopedCollectingLogSink() {
    absl::RemoveLogSink(&sink_);
  }

  ScopedCollectingLogSink(const ScopedCollectingLogSink&) = delete;
  ScopedCollectingLogSink& operator=(const ScopedCollectingLogSink&) = delete;

 private:
  CollectingLogSink& sink_;
};

class ScopedVLogLevel {
 public:
  explicit ScopedVLogLevel(int level) : previous_level_(absl::SetGlobalVLogLevel(level)) {}

  ~ScopedVLogLevel() {
    absl::SetGlobalVLogLevel(previous_level_);
  }

  ScopedVLogLevel(const ScopedVLogLevel&) = delete;
  ScopedVLogLevel& operator=(const ScopedVLogLevel&) = delete;

 private:
  int previous_level_;
};

static Topology make_pcie_batch_payload_topology(
    std::string_view local_endpoint_id,
    std::string_view remote_endpoint_id) {
  std::vector<Pool> pools;
  pools.push_back(Pool{"cpu0", "cpu0", PoolType::kCpu});
  pools.push_back(Pool{"gpu0", "gpu0", PoolType::kGpu});
  pools.push_back(Pool{"gpu1", "gpu1", PoolType::kGpu});

  std::vector<Endpoint> endpoints;
  Endpoint local;
  local.id = std::string(local_endpoint_id);
  local.name = local.id;
  local.kind = EndpointKind::kClient;
  local.type = EndpointType::kPcie;
  local.pool_ids = {"cpu0", "gpu0"};
  endpoints.push_back(std::move(local));

  Endpoint remote;
  remote.id = std::string(remote_endpoint_id);
  remote.name = remote.id;
  remote.kind = EndpointKind::kClient;
  remote.type = EndpointType::kPcie;
  remote.pool_ids = {"cpu0", "gpu1"};
  endpoints.push_back(std::move(remote));

  std::vector<Link> links;
  Link link;
  link.id = absl::StrCat(local_endpoint_id, "_to_", remote_endpoint_id);
  link.name = link.id;
  link.type = LinkType::kP2P;
  link.src_endpoint_id = std::string(local_endpoint_id);
  link.dst_endpoint_id = std::string(remote_endpoint_id);
  links.push_back(std::move(link));

  auto topology_or = Topology::Build(
      std::move(pools),
      std::move(endpoints),
      std::move(links),
      {.require_endpoint_links = true, .require_connected = false});
  INFO(topology_or.status().message());
  REQUIRE(topology_or.ok());
  return std::move(*topology_or);
}

static void install_pcie_batch_payload_routing_context(
    const std::shared_ptr<tensorcast::store::components::CommunicationManager>& comm_manager,
    std::string_view local_endpoint_id,
    std::string_view remote_endpoint_id,
    std::string_view local_host,
    uint16_t local_port,
    std::string_view remote_host,
    uint16_t remote_port) {
  auto engine = comm_manager->get_shared_engine();
  auto routing_context = std::make_shared<RoutingContext>(
      RoutingContext::Options{}, engine, nullptr, std::make_shared<PcieAdapter>(engine));
  REQUIRE(routing_context->set_topology(make_pcie_batch_payload_topology(local_endpoint_id, remote_endpoint_id)).ok());
  REQUIRE(routing_context
              ->set_endpoint_bindings({
                  EndpointBinding{
                      .endpoint_id = std::string(local_endpoint_id),
                      .node_id = "node-pcie-direct",
                      .ip = std::string(local_host),
                      .port = local_port,
                  },
                  EndpointBinding{
                      .endpoint_id = std::string(remote_endpoint_id),
                      .node_id = "node-pcie-direct",
                      .ip = std::string(remote_host),
                      .port = remote_port,
                  },
              })
              .ok());
  comm_manager->set_routing_context(std::move(routing_context));
}

static tensorcast::communicator::v1::CommunicatorConfig make_single_node_communicator_config(
    int node_index,
    std::string_view nic_name) {
  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/false, /*buffers_per_flow=*/2);
  auto* simple = cfg.mutable_simple_numa();
  simple->set_enable(true);
  auto* node = simple->add_nodes();
  node->set_id(node_index);
  node->add_nics(std::string(nic_name));
  node->add_gpus(node_index);
  return cfg;
}

static std::shared_ptr<tensorcast::store::components::CommunicationManager> make_comm_manager_with_config(
    const tensorcast::communicator::v1::CommunicatorConfig& config) {
  auto comm_manager = std::make_shared<tensorcast::store::components::CommunicationManager>();
  auto status = comm_manager->initialize_with_config("127.0.0.1", /*listen_port=*/0, config);
  REQUIRE(status.ok());
  return comm_manager;
}

static void install_worker_directory_entry(
    DaemonServiceHarness& harness,
    std::string_view daemon_id,
    std::string_view worker_id,
    std::string_view node_id,
    std::string_view node_address,
    uint32_t grpc_port,
    uint32_t p2p_port) {
  harness.kernel().worker_directory_cache().update_local_entry(
      tensorcast::daemon::WorkerDirectoryCache::Entry{
          .daemon_id = std::string(daemon_id),
          .worker_id = std::string(worker_id),
          .node_id = std::string(node_id),
          .node_address = std::string(node_address),
          .grpc_port = grpc_port,
          .p2p_port = p2p_port,
          .address = absl::StrCat(node_address, ":", grpc_port),
          .capability_flags = 0,
      });
}

static void install_local_worker_directory_entry(DaemonServiceHarness& harness, std::string_view daemon_id) {
  install_worker_directory_entry(
      harness,
      daemon_id,
      /*worker_id=*/"worker-local",
      /*node_id=*/"node-local",
      /*node_address=*/"127.0.0.1",
      /*grpc_port=*/18080,
      /*p2p_port=*/19090);
}

static tensorcast::daemon::PersistenceTaskState advance_persistence_to_terminal(
    DaemonServiceHarness& harness,
    std::string_view task_id) {
  for (int i = 0; i < 100; ++i) {
    harness.kernel().persistence_manager()->advance_once_for_test();
    auto task = harness.kernel().persistence_manager()->get_by_task_id(task_id);
    REQUIRE(task.has_value());
    if (task->state == tensorcast::daemon::v2::PERSISTENCE_STATE_SUCCESS ||
        task->state == tensorcast::daemon::v2::PERSISTENCE_STATE_FAILED ||
        task->state == tensorcast::daemon::v2::PERSISTENCE_STATE_DEGRADED) {
      return *task;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  FAIL("persistence task did not reach terminal state");
  return tensorcast::daemon::PersistenceTaskState{};
}

std::string sha256_hex(std::string_view payload) {
  const auto digest = tensorcast::common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));
}

uint64_t shard_for_artifact(std::string_view artifact_id) {
  const auto digest = tensorcast::common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(artifact_id.data()), artifact_id.size()));
  uint64_t hash64 = 0;
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    hash64 |= static_cast<uint64_t>(digest[i]) << (8U * i);
  }
  return hash64 % 4096ULL;
}

std::string replace_final_b64u_suffix(std::string_view artifact_id, std::string_view suffix) {
  const std::size_t marker = artifact_id.rfind("~b64u.");
  REQUIRE(marker != std::string_view::npos);
  return absl::StrCat(artifact_id.substr(0, marker + 6), suffix);
}

std::string artifact_on_same_shard(std::string_view base_artifact_id, std::string_view seed) {
  const std::uint64_t target_shard = shard_for_artifact(base_artifact_id);
  for (int attempt = 0; attempt < 10000; ++attempt) {
    const std::string candidate = replace_final_b64u_suffix(base_artifact_id, absl::StrCat(seed, attempt));
    if (candidate != base_artifact_id && shard_for_artifact(candidate) == target_shard) {
      return candidate;
    }
  }
  FAIL("failed to find same-shard artifact id");
  return {};
}

std::string make_valid_byte_artifact_id(
    std::string_view namespace_name,
    std::string_view engine,
    std::string_view model_id,
    std::string_view model_version,
    std::string_view layout_id,
    std::string_view engine_key) {
  return absl::StrCat(
      "cgid:byte_artifact~",
      namespace_name,
      "~",
      engine,
      "~",
      tensorcast::common::encode_cgid_segment(model_id),
      "~",
      tensorcast::common::encode_cgid_segment(model_version),
      "~",
      layout_id,
      "~",
      tensorcast::common::encode_cgid_segment(engine_key));
}

std::string make_test_byte_artifact_id(std::string_view engine_key, std::string_view layout_id = "layout_v1") {
  return make_valid_byte_artifact_id("tenant", "engine", "batch-runtime-model", "v1", layout_id, engine_key);
}

TEST_CASE("ArtifactProfileRegistry exposes explicit family and authority traits", "[daemon][profile_registry]") {
  using Registry = tensorcast::daemon::ArtifactProfileRegistry;

  CHECK(Registry::classify_artifact_id("mi2:weights/example") == Registry::Profile::kOrdinaryArtifact);
  CHECK(
      Registry::traits_for_artifact_id("mi2:weights/example").authority_model ==
      Registry::AuthorityModel::kGlobalStoreBacked);
  CHECK(Registry::traits_for_artifact_id("mi2:weights/example").family == Registry::ArtifactFamily::kOrdinary);
  CHECK(Registry::classify_artifact_id("not-an-artifact-id") == Registry::Profile::kUnknown);
  CHECK_FALSE(
      Registry::runtime_for_profile(Registry::Profile::kOrdinaryArtifact)
          .validate_artifact_id_for_field("not-an-artifact-id", "artifact_id")
          .ok());

  const std::string byte_artifact_id = make_test_byte_artifact_id("registry-traits:blk-4");
  CHECK(Registry::classify_artifact_id(byte_artifact_id) == Registry::Profile::kByteArtifact);
  CHECK(
      Registry::traits_for_artifact_id(byte_artifact_id).authority_model == Registry::AuthorityModel::kRoutedHomeEpoch);
  CHECK(Registry::traits_for_artifact_id(byte_artifact_id).family == Registry::ArtifactFamily::kHighCardinality);
  CHECK(Registry::traits_for_artifact_id(byte_artifact_id).fixed_full_selection);
}

void set_invariant(
    tensorcast::daemon::v2::PutIfAbsentInvariant* invariant,
    std::string_view layout_id,
    std::string_view payload) {
  invariant->set_layout_id(std::string(layout_id));
  invariant->set_byte_length(payload.size());
  invariant->set_payload_digest_alg("sha256");
  invariant->set_payload_digest_hex(sha256_hex(payload));
}

struct RegisteredRegion {
  std::string region_id;
  std::string device_uuid;
  void* device_ptr{nullptr};
  int owner_pid{0};
  std::uint64_t size_bytes{0};
};

struct RegisteredHostSharedRegion {
  std::string region_id;
  void* base_ptr{nullptr};
  int owner_pid{0};
  std::uint64_t size_bytes{0};
};

RegisteredRegion register_test_region(
    tensorcast::daemon::DaemonServiceHarness& harness,
    int device_id,
    std::uint64_t size_bytes) {
  REQUIRE(tensorcast::cuda::set_device(device_id).ok());
  void* device_ptr = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&device_ptr, size_bytes).ok());

  cudaIpcMemHandle_t handle{};
  REQUIRE(tensorcast::cuda::get_ipc_mem_handle(&handle, device_ptr).ok());
  const auto handle_bytes = tensorcast::cuda::IpcHandleBytes::from_native(handle);

  auto device_key = tensorcast::store::DeviceRegistry::instance().gpu_key(device_id);
  if (device_key.uuid.empty()) {
    tensorcast::store::DeviceRegistry::instance().register_gpu(device_id, absl::StrCat("fake-device-", device_id));
    device_key = tensorcast::store::DeviceRegistry::instance().gpu_key(device_id);
  }

  tensorcast::daemon::IpcRegionRegistry::RegisterParams params;
  params.device_id = device_id;
  params.owner_pid = ::getpid();
  params.size_bytes = size_bytes;
  params.ttl_ms = 10'000;
  params.handle_bytes = std::string(handle_bytes.as_string_view());
  auto region_or = harness.kernel().region_registry().register_region(params);
  REQUIRE(region_or.ok());

  return RegisteredRegion{
      .region_id = region_or->region_id,
      .device_uuid = device_key.uuid,
      .device_ptr = device_ptr,
      .owner_pid = ::getpid(),
      .size_bytes = size_bytes,
  };
}

RegisteredHostSharedRegion register_test_host_shared_region(
    tensorcast::daemon::DaemonServiceHarness& harness,
    std::uint64_t size_bytes,
    tensorcast::daemon::IpcRegionRegistry::HostRegionClass host_region_class =
        tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kScratch) {
  tensorcast::daemon::IpcRegionRegistry::RegisterParams params;
  params.memory_kind = tensorcast::daemon::IpcRegionRegistry::MemoryKind::kHostShared;
  params.device_id = -1;
  params.owner_pid = ::getpid();
  params.size_bytes = size_bytes;
  params.ttl_ms = 10'000;
  params.daemon_managed = true;
  params.host_region_class = host_region_class;
  auto region_or = harness.kernel().region_registry().register_region(params);
  REQUIRE(region_or.ok());

  auto mapping_or =
      harness.kernel().region_registry().acquire_host_shared_local_mapping(region_or->region_id, ::getpid());
  REQUIRE(mapping_or.ok());
  void* base_ptr = mapping_or->base_ptr;
  REQUIRE(harness.kernel().region_registry().release(region_or->region_id).ok());

  return RegisteredHostSharedRegion{
      .region_id = region_or->region_id,
      .base_ptr = base_ptr,
      .owner_pid = ::getpid(),
      .size_bytes = size_bytes,
  };
}

void release_test_region(tensorcast::daemon::DaemonServiceHarness& harness, const RegisteredRegion& region) {
  auto unregister_or = harness.kernel().region_registry().unregister_region(
      region.region_id,
      region.owner_pid,
      /*force=*/true);
  REQUIRE(unregister_or.ok());
  REQUIRE(tensorcast::cuda::free(region.device_ptr).ok());
}

void release_test_host_shared_region(
    tensorcast::daemon::DaemonServiceHarness& harness,
    const RegisteredHostSharedRegion& region) {
  auto unregister_or = harness.kernel().region_registry().unregister_region(
      region.region_id,
      region.owner_pid,
      /*force=*/true);
  REQUIRE(unregister_or.ok());
}

void populate_single_region_layout(
    tensorcast::daemon::v2::TargetLayout* layout,
    const RegisteredRegion& region,
    std::string_view artifact_id,
    std::uint64_t byte_length,
    int device_id) {
  layout->Clear();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(device_id);
  storage->set_storage_length(byte_length);
  storage->set_vram_region_id(region.region_id);
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name(std::string(artifact_id));
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(byte_length);
}

void populate_single_host_shared_region_layout(
    tensorcast::daemon::v2::TargetLayout* layout,
    const RegisteredHostSharedRegion& region,
    std::string_view artifact_id,
    std::uint64_t byte_length,
    std::uint64_t storage_length) {
  layout->Clear();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(-1);
  storage->set_storage_length(storage_length);
  storage->set_mapping_base_offset(0);
  auto* region_ref = storage->mutable_region_ref();
  region_ref->set_region_id(region.region_id);
  region_ref->set_memory_kind(tensorcast::daemon::v2::REGION_MEMORY_KIND_HOST_SHARED);
  region_ref->set_device_id(-1);
  region_ref->set_size_bytes(region.size_bytes);

  auto* offset = layout->add_offsets();
  offset->set_name(std::string(artifact_id));
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(byte_length);
}

void populate_two_item_host_shared_region_layout(
    tensorcast::daemon::v2::TargetLayout* layout,
    const RegisteredHostSharedRegion& region,
    std::string_view artifact_id_a,
    std::uint64_t byte_length_a,
    std::string_view artifact_id_b,
    std::uint64_t byte_length_b) {
  layout->Clear();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(-1);
  storage->set_storage_length(byte_length_a + byte_length_b);
  storage->set_mapping_base_offset(0);
  auto* region_ref = storage->mutable_region_ref();
  region_ref->set_region_id(region.region_id);
  region_ref->set_memory_kind(tensorcast::daemon::v2::REGION_MEMORY_KIND_HOST_SHARED);
  region_ref->set_device_id(-1);
  region_ref->set_size_bytes(region.size_bytes);

  auto* offset_a = layout->add_offsets();
  offset_a->set_name(std::string(artifact_id_a));
  offset_a->set_storage_id("storage-0");
  offset_a->set_storage_offset(0);
  offset_a->set_logical_length(byte_length_a);

  auto* offset_b = layout->add_offsets();
  offset_b->set_name(std::string(artifact_id_b));
  offset_b->set_storage_id("storage-0");
  offset_b->set_storage_offset(byte_length_a);
  offset_b->set_logical_length(byte_length_b);
}

} // namespace

TEST_CASE("HomeBatch* put/get/exists support join and conflict", "[daemon][batch][home]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_valid_byte_artifact_id(
      "tenant", "sglang", "meta-llama/Llama-3.1-8B-Instruct", "default", "layout_v1", "request-host-shared-ok:blk-1");
  const std::string payload = "payload-bytes-v1";
  const uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  auto* item = put_req.add_items();
  item->set_artifact_id(artifact_id);
  item->set_inline_payload(payload);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  auto put_st = svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp);
  REQUIRE(put_st.ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(count_cpu_replicas(*engine) == 1);

  HomeBatchPutIfAbsentResponse join_resp;
  grpc::ServerContext join_ctx;
  auto join_st = svc.HomeBatchPutIfAbsent(&join_ctx, &put_req, &join_resp);
  REQUIRE(join_st.ok());
  REQUIRE(join_resp.outcomes_size() == 1);
  REQUIRE(join_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(count_cpu_replicas(*engine) == 1);

  HomeBatchPutIfAbsentRequest conflict_req = put_req;
  conflict_req.mutable_items(0)->mutable_invariant()->set_layout_id("layout_v2");
  HomeBatchPutIfAbsentResponse conflict_resp;
  grpc::ServerContext conflict_ctx;
  auto conflict_st = svc.HomeBatchPutIfAbsent(&conflict_ctx, &conflict_req, &conflict_resp);
  REQUIRE(conflict_st.ok());
  REQUIRE(conflict_resp.outcomes_size() == 1);
  CAPTURE(conflict_resp.outcomes(0).message());
  REQUIRE(conflict_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_FAILED_PRECONDITION);

  HomeBatchExistsRequest exists_req;
  exists_req.mutable_fence()->CopyFrom(put_req.fence());
  exists_req.add_artifact_ids(artifact_id);
  HomeBatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  auto exists_st = svc.HomeBatchExists(&exists_ctx, &exists_req, &exists_resp);
  REQUIRE(exists_st.ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  HomeBatchGetRequest get_req;
  get_req.mutable_fence()->CopyFrom(put_req.fence());
  get_req.add_artifact_ids(artifact_id);
  HomeBatchGetResponse get_resp;
  grpc::ServerContext get_ctx;
  auto get_st = svc.HomeBatchGet(&get_ctx, &get_req, &get_resp);
  REQUIRE(get_st.ok());
  REQUIRE(get_resp.items_size() == 1);
  REQUIRE(get_resp.items(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.items(0).inline_payload() == payload);
}

TEST_CASE("HomeBatch* enforces stale fence generation with redirect", "[daemon][batch][fence]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("stale-fence:blk-2");
  const std::string payload = "payload-bytes-v2";
  const uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  auto* item = put_req.add_items();
  item->set_artifact_id(artifact_id);
  item->set_inline_payload(payload);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  auto put_st = svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp);
  REQUIRE(put_st.ok());

  HomeBatchExistsRequest stale_req;
  stale_req.mutable_fence()->set_shard_id(shard_id);
  stale_req.mutable_fence()->set_lease_generation(2);
  stale_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  stale_req.mutable_fence()->set_routing_epoch(1);
  stale_req.add_artifact_ids(artifact_id);

  HomeBatchExistsResponse stale_resp;
  grpc::ServerContext stale_ctx;
  auto stale_st = svc.HomeBatchExists(&stale_ctx, &stale_req, &stale_resp);
  REQUIRE(stale_st.ok());
  REQUIRE(stale_resp.outcomes_size() == 1);
  REQUIRE(stale_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_FAILED_PRECONDITION);
}

TEST_CASE("HomeBatchExists rejects malformed byte artifact cgid", "[daemon][batch][validation]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~layout_only";
  const uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchExistsRequest exists_req;
  exists_req.mutable_fence()->set_shard_id(shard_id);
  exists_req.mutable_fence()->set_lease_generation(1);
  exists_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  exists_req.mutable_fence()->set_routing_epoch(1);
  exists_req.add_artifact_ids(artifact_id);

  HomeBatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  auto exists_st = svc.HomeBatchExists(&exists_ctx, &exists_req, &exists_resp);
  REQUIRE(exists_st.ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_INVALID_ARGUMENT);
}

TEST_CASE("Batch* front-door returns per-item outcomes and no UNIMPLEMENTED", "[daemon][batch][frontdoor]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id =
      make_valid_byte_artifact_id("tenant", "engine", "batch/frontdoor", "v1", "layout_v1", "frontdoor-put:blk-3");
  const std::string payload = "payload-bytes-v3";

  BatchPutIfAbsentFromRegionRequest put_req;
  {
    auto* valid = put_req.add_items();
    valid->mutable_selection()->set_artifact_id(artifact_id);
    valid->set_inline_payload(payload);
    set_invariant(valid->mutable_invariant(), "layout_v1", payload);
  }
  {
    auto* invalid = put_req.add_items();
    invalid->mutable_selection()->set_artifact_id("");
    invalid->set_inline_payload(payload);
    set_invariant(invalid->mutable_invariant(), "layout_v1", payload);
  }
  {
    auto* malformed = put_req.add_items();
    malformed->mutable_selection()->set_artifact_id("cgid:byte_artifact~tenant~engine~layout_only");
    malformed->set_inline_payload(payload);
    set_invariant(malformed->mutable_invariant(), "layout_v1", payload);
  }

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  auto put_st = svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp);
  REQUIRE(put_st.ok());
  REQUIRE(put_resp.outcomes_size() == 3);
  CAPTURE(put_resp.outcomes(0).message());
  CAPTURE(put_resp.outcomes(1).message());
  CAPTURE(put_resp.outcomes(2).message());
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(put_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_INVALID_ARGUMENT);
  REQUIRE(put_resp.outcomes(2).status() == BatchItemStatus::BATCH_ITEM_STATUS_INVALID_ARGUMENT);

  BatchExistsRequest exists_req;
  exists_req.add_selections()->set_artifact_id(artifact_id);
  BatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  auto exists_st = svc.BatchExists(&exists_ctx, &exists_req, &exists_resp);
  REQUIRE(exists_st.ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  auto region = register_test_region(*harness, /*device_id=*/0, payload.size());
  REQUIRE(tensorcast::cuda::memset(region.device_ptr, 0, payload.size()).ok());

  BatchGetIntoRegionRequest get_req;
  get_req.add_selections()->set_artifact_id(artifact_id);
  get_req.mutable_target_layout();
  populate_single_region_layout(
      get_req.mutable_target_layout(),
      region,
      artifact_id,
      payload.size(),
      /*device_id=*/0);
  get_req.set_pid(region.owner_pid);
  get_req.set_device_uuid(region.device_uuid);
  BatchGetIntoRegionResponse get_resp;
  grpc::ServerContext get_ctx;
  auto get_st = svc.BatchGetIntoRegion(&get_ctx, &get_req, &get_resp);
  REQUIRE(get_st.ok());
  REQUIRE(get_resp.outcomes_size() == 1);
  REQUIRE(get_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  std::vector<char> out(payload.size(), '\0');
  REQUIRE(tensorcast::cuda::memcpy(out.data(), region.device_ptr, out.size(), cudaMemcpyDeviceToHost).ok());
  REQUIRE(std::string(out.data(), out.size()) == payload);
  release_test_region(*harness, region);

  MaterializeReplicaRequest legacy_req;
  legacy_req.mutable_selection()->set_artifact_id("mi2:legacy:ok");
  legacy_req.set_target_device_type(DeviceType::DEVICE_TYPE_GPU);
  MaterializeReplicaResponse legacy_resp;
  grpc::ServerContext legacy_ctx;
  auto legacy_st = svc.MaterializeReplica(&legacy_ctx, &legacy_req, &legacy_resp);
  REQUIRE(legacy_st.error_code() != grpc::StatusCode::UNIMPLEMENTED);
}

TEST_CASE("Batch* front-door supports source_layout and payload_ref transport", "[daemon][batch][transport]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.batch_transport_protocol_version = 0;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id =
      make_valid_byte_artifact_id("tenant", "engine", "batch/transport", "v1", "layout_v1", "payload-ref-flow:blk-5");
  const std::string payload = "payload-ref-region-flow";

  auto source_region = register_test_region(*harness, /*device_id=*/0, payload.size());
  REQUIRE(
      tensorcast::cuda::memcpy(source_region.device_ptr, payload.data(), payload.size(), cudaMemcpyHostToDevice).ok());

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* item = put_req.add_items();
  item->mutable_selection()->set_artifact_id(artifact_id);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);
  populate_single_region_layout(
      put_req.mutable_source_layout(),
      source_region,
      artifact_id,
      payload.size(),
      /*device_id=*/0);
  put_req.set_pid(source_region.owner_pid);
  put_req.set_device_uuid(source_region.device_uuid);
  put_req.set_operation_id("op-local-payload-ref");

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  CAPTURE(put_resp.outcomes(0).message());
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  auto target_region = register_test_region(*harness, /*device_id=*/0, payload.size());
  REQUIRE(tensorcast::cuda::memset(target_region.device_ptr, 0, payload.size()).ok());

  BatchGetIntoRegionRequest get_req;
  get_req.add_selections()->set_artifact_id(artifact_id);
  populate_single_region_layout(
      get_req.mutable_target_layout(),
      target_region,
      artifact_id,
      payload.size(),
      /*device_id=*/0);
  get_req.set_pid(target_region.owner_pid);
  get_req.set_device_uuid(target_region.device_uuid);
  get_req.set_operation_id("op-local-payload-ref");

  BatchGetIntoRegionResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.BatchGetIntoRegion(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.outcomes_size() == 1);
  REQUIRE(get_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  std::vector<char> out(payload.size(), '\0');
  REQUIRE(tensorcast::cuda::memcpy(out.data(), target_region.device_ptr, out.size(), cudaMemcpyDeviceToHost).ok());
  REQUIRE(std::string(out.data(), out.size()) == payload);

  release_test_region(*harness, source_region);
  release_test_region(*harness, target_region);
}

TEST_CASE("BatchPutIfAbsentFromRegion supports HOST_SHARED source layouts", "[daemon][batch][host_shared][put]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_valid_byte_artifact_id(
      "tenant", "sglang", "meta-llama/Llama-3.1-8B-Instruct", "default", "layout_v1", "request-host-shared-ok:blk-1");
  const std::string payload = "host-shared-source-payload";

  auto source_region = register_test_host_shared_region(*harness, payload.size());
  std::memcpy(source_region.base_ptr, payload.data(), payload.size());

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* item = put_req.add_items();
  item->mutable_selection()->set_artifact_id(artifact_id);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);
  populate_single_host_shared_region_layout(
      put_req.mutable_source_layout(), source_region, artifact_id, payload.size(), payload.size());
  put_req.set_pid(source_region.owner_pid);
  put_req.set_operation_id("op-host-shared-put");

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  CAPTURE(put_resp.outcomes(0).message());
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(put_resp.outcomes(0).message() == "created");

  BatchPutIfAbsentFromRegionResponse join_resp;
  grpc::ServerContext join_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&join_ctx, &put_req, &join_resp).ok());
  REQUIRE(join_resp.outcomes_size() == 1);
  CAPTURE(join_resp.outcomes(0).message());
  REQUIRE(join_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(join_resp.outcomes(0).message() == "joined");

  auto target_region = register_test_host_shared_region(*harness, payload.size());
  std::memset(target_region.base_ptr, 0, payload.size());

  BatchGetIntoRegionRequest get_req;
  get_req.add_selections()->set_artifact_id(artifact_id);
  populate_single_host_shared_region_layout(
      get_req.mutable_target_layout(), target_region, artifact_id, payload.size(), payload.size());
  get_req.set_pid(target_region.owner_pid);
  get_req.set_operation_id("op-host-shared-put");

  BatchGetIntoRegionResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.BatchGetIntoRegion(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.outcomes_size() == 1);
  REQUIRE(get_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(std::memcmp(target_region.base_ptr, payload.data(), payload.size()) == 0);

  release_test_host_shared_region(*harness, target_region);
  release_test_host_shared_region(*harness, source_region);
}

TEST_CASE(
    "BatchPutIfAbsentFromRegion keeps partial outcomes when HOST_SHARED source layout is invalid",
    "[daemon][batch][host_shared][put]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string inline_artifact_id = make_valid_byte_artifact_id(
      "tenant",
      "sglang",
      "meta-llama/Llama-3.1-8B-Instruct",
      "default",
      "layout_v1",
      "request-host-shared-inline:blk-1");
  const std::string host_artifact_id = make_valid_byte_artifact_id(
      "tenant", "sglang", "meta-llama/Llama-3.1-8B-Instruct", "default", "layout_v1", "request-host-shared-host:blk-2");
  const std::string inline_payload = "inline-survives-invalid-host-layout";
  const std::string host_payload = "host-layout-invalid";

  auto source_region = register_test_host_shared_region(*harness, host_payload.size());
  std::memcpy(source_region.base_ptr, host_payload.data(), host_payload.size());

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* inline_item = put_req.add_items();
  inline_item->mutable_selection()->set_artifact_id(inline_artifact_id);
  inline_item->set_inline_payload(inline_payload);
  set_invariant(inline_item->mutable_invariant(), "layout_v1", inline_payload);

  auto* host_item = put_req.add_items();
  host_item->mutable_selection()->set_artifact_id(host_artifact_id);
  set_invariant(host_item->mutable_invariant(), "layout_v1", host_payload);

  populate_single_host_shared_region_layout(
      put_req.mutable_source_layout(), source_region, host_artifact_id, host_payload.size(), host_payload.size() - 1);
  put_req.set_pid(source_region.owner_pid);
  put_req.set_operation_id("op-host-shared-invalid-layout");

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 2);
  CAPTURE(put_resp.outcomes(0).message());
  CAPTURE(put_resp.outcomes(1).message());
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(put_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_FAILED_PRECONDITION);

  BatchExistsRequest exists_req;
  exists_req.add_selections()->set_artifact_id(inline_artifact_id);
  exists_req.add_selections()->set_artifact_id(host_artifact_id);

  BatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  REQUIRE(svc.BatchExists(&exists_ctx, &exists_req, &exists_resp).ok());
  REQUIRE(exists_resp.outcomes_size() == 2);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(exists_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_MISS);

  release_test_host_shared_region(*harness, source_region);
}

TEST_CASE(
    "BatchPutIfAbsentFromRegion rejects poisoned HOST_SHARED source regions",
    "[daemon][batch][host_shared][put]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_valid_byte_artifact_id(
      "tenant",
      "sglang",
      "meta-llama/Llama-3.1-8B-Instruct",
      "default",
      "layout_v1",
      "request-host-shared-poisoned:blk-1");
  const std::string payload = "host-shared-poisoned";

  auto source_region = register_test_host_shared_region(*harness, payload.size());
  std::memcpy(source_region.base_ptr, payload.data(), payload.size());
  REQUIRE(harness->kernel().region_registry().mark_poisoned(source_region.region_id).ok());

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* item = put_req.add_items();
  item->mutable_selection()->set_artifact_id(artifact_id);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);
  populate_single_host_shared_region_layout(
      put_req.mutable_source_layout(), source_region, artifact_id, payload.size(), payload.size());
  put_req.set_pid(source_region.owner_pid);
  put_req.set_operation_id("op-host-shared-poisoned");

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  CAPTURE(put_resp.outcomes(0).message());
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_FAILED_PRECONDITION);

  release_test_host_shared_region(*harness, source_region);
}

TEST_CASE("BatchGetIntoRegion supports HOST_SHARED target layouts", "[daemon][batch][host_shared][get]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_valid_byte_artifact_id(
      "tenant", "sglang", "meta-llama/Llama-3.1-8B-Instruct", "default", "layout_v1", "request-host-target-ok:blk-1");
  const std::string payload = "host-shared-target-payload";

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* put_item = put_req.add_items();
  put_item->mutable_selection()->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  auto target_region = register_test_host_shared_region(*harness, payload.size());
  std::memset(target_region.base_ptr, 0, payload.size());

  BatchGetIntoRegionRequest get_req;
  get_req.add_selections()->set_artifact_id(artifact_id);
  populate_single_host_shared_region_layout(
      get_req.mutable_target_layout(), target_region, artifact_id, payload.size(), payload.size());
  get_req.set_pid(target_region.owner_pid);
  get_req.set_operation_id("op-host-shared-get");

  BatchGetIntoRegionResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.BatchGetIntoRegion(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.outcomes_size() == 1);
  REQUIRE(get_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(std::memcmp(target_region.base_ptr, payload.data(), payload.size()) == 0);
  auto comm_manager = engine->get_shared_comm_manager();
  if (comm_manager->stable_local_backing_supported_for_test()) {
    bool active = false;
    for (int i = 0; i < 200; ++i) {
      if (comm_manager->stable_local_backing_active_for_test(target_region.region_id)) {
        active = true;
        break;
      }
      absl::SleepFor(absl::Milliseconds(10));
    }
    CHECK(active);
  }

  release_test_host_shared_region(*harness, target_region);
}

TEST_CASE("Batch* front-door echoes HOST_SHARED slot tokens in outcomes", "[daemon][batch][host_shared][slot_token]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_valid_byte_artifact_id(
      "tenant", "sglang", "meta-llama/Llama-3.1-8B-Instruct", "default", "layout_v1", "request-host-slot-token:blk-1");
  const std::string payload = "host-shared-slot-token-payload";
  constexpr std::uint64_t kSlotIndex = 23;
  constexpr std::uint64_t kSlotGeneration = 41;

  auto source_region = register_test_host_shared_region(
      *harness, payload.size(), tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator);
  std::memcpy(source_region.base_ptr, payload.data(), payload.size());

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* put_item = put_req.add_items();
  put_item->mutable_selection()->set_artifact_id(artifact_id);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);
  populate_single_host_shared_region_layout(
      put_req.mutable_source_layout(), source_region, artifact_id, payload.size(), payload.size());
  put_req.mutable_source_layout()->mutable_offsets(0)->set_slot_index(kSlotIndex);
  put_req.mutable_source_layout()->mutable_offsets(0)->set_slot_generation(kSlotGeneration);
  put_req.set_pid(source_region.owner_pid);
  put_req.set_operation_id("op-host-shared-slot-token-put");

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(put_resp.outcomes(0).has_slot_index());
  REQUIRE(put_resp.outcomes(0).has_slot_generation());
  REQUIRE(put_resp.outcomes(0).slot_index() == kSlotIndex);
  REQUIRE(put_resp.outcomes(0).slot_generation() == kSlotGeneration);

  auto target_region = register_test_host_shared_region(
      *harness, payload.size(), tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator);
  std::memset(target_region.base_ptr, 0, payload.size());

  BatchGetIntoRegionRequest get_req;
  get_req.add_selections()->set_artifact_id(artifact_id);
  populate_single_host_shared_region_layout(
      get_req.mutable_target_layout(), target_region, artifact_id, payload.size(), payload.size());
  get_req.mutable_target_layout()->mutable_offsets(0)->set_slot_index(kSlotIndex);
  get_req.mutable_target_layout()->mutable_offsets(0)->set_slot_generation(kSlotGeneration);
  get_req.set_pid(target_region.owner_pid);
  get_req.set_operation_id("op-host-shared-slot-token-get");

  BatchGetIntoRegionResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.BatchGetIntoRegion(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.outcomes_size() == 1);
  REQUIRE(get_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.outcomes(0).has_slot_index());
  REQUIRE(get_resp.outcomes(0).has_slot_generation());
  REQUIRE(get_resp.outcomes(0).slot_index() == kSlotIndex);
  REQUIRE(get_resp.outcomes(0).slot_generation() == kSlotGeneration);
  REQUIRE(std::memcmp(target_region.base_ptr, payload.data(), payload.size()) == 0);

  release_test_host_shared_region(*harness, target_region);
  release_test_host_shared_region(*harness, source_region);
}

TEST_CASE(
    "BatchPutIfAbsentFromRegion rejects allocator-backed HOST_SHARED source layouts without slot tokens",
    "[daemon][batch][host_shared][allocator][put]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_valid_byte_artifact_id(
      "tenant",
      "sglang",
      "meta-llama/Llama-3.1-8B-Instruct",
      "default",
      "layout_v1",
      "request-host-allocator-put:blk-1");
  const std::string payload = "allocator-source-without-slot-token";

  auto source_region = register_test_host_shared_region(
      *harness, payload.size(), tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator);
  std::memcpy(source_region.base_ptr, payload.data(), payload.size());

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* item = put_req.add_items();
  item->mutable_selection()->set_artifact_id(artifact_id);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);
  populate_single_host_shared_region_layout(
      put_req.mutable_source_layout(), source_region, artifact_id, payload.size(), payload.size());
  put_req.set_pid(source_region.owner_pid);
  put_req.set_operation_id("op-host-allocator-put-missing-slot-token");

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_FAILED_PRECONDITION);

  release_test_host_shared_region(*harness, source_region);
}

TEST_CASE(
    "BatchGetIntoRegion rejects allocator-backed HOST_SHARED target layouts without slot tokens",
    "[daemon][batch][host_shared][allocator][get]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_valid_byte_artifact_id(
      "tenant",
      "sglang",
      "meta-llama/Llama-3.1-8B-Instruct",
      "default",
      "layout_v1",
      "request-host-allocator-get:blk-1");
  const std::string payload = "allocator-target-without-slot-token";

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* put_item = put_req.add_items();
  put_item->mutable_selection()->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  auto target_region = register_test_host_shared_region(
      *harness, payload.size(), tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator);
  std::memset(target_region.base_ptr, 0, payload.size());

  BatchGetIntoRegionRequest get_req;
  get_req.add_selections()->set_artifact_id(artifact_id);
  populate_single_host_shared_region_layout(
      get_req.mutable_target_layout(), target_region, artifact_id, payload.size(), payload.size());
  get_req.set_pid(target_region.owner_pid);
  get_req.set_operation_id("op-host-allocator-get-missing-slot-token");

  BatchGetIntoRegionResponse get_resp;
  grpc::ServerContext get_ctx;
  const auto get_status = svc.BatchGetIntoRegion(&get_ctx, &get_req, &get_resp);
  REQUIRE_FALSE(get_status.ok());
  REQUIRE(get_status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);

  release_test_host_shared_region(*harness, target_region);
}

TEST_CASE(
    "BatchGetIntoRegion preserves partial-hit semantics on HOST_SHARED targets",
    "[daemon][batch][host_shared][get]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string hit_artifact_id = make_valid_byte_artifact_id(
      "tenant", "sglang", "meta-llama/Llama-3.1-8B-Instruct", "default", "layout_v1", "request-host-target-hit:blk-1");
  const std::string miss_artifact_id = make_valid_byte_artifact_id(
      "tenant", "sglang", "meta-llama/Llama-3.1-8B-Instruct", "default", "layout_v1", "request-host-target-miss:blk-2");
  const std::string hit_payload = "host-target-hit";
  const std::size_t total_bytes = hit_payload.size() + 16;

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* put_item = put_req.add_items();
  put_item->mutable_selection()->set_artifact_id(hit_artifact_id);
  put_item->set_inline_payload(hit_payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", hit_payload);

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  auto target_region = register_test_host_shared_region(*harness, total_bytes);
  std::memset(target_region.base_ptr, 0, total_bytes);

  BatchGetIntoRegionRequest get_req;
  get_req.add_selections()->set_artifact_id(hit_artifact_id);
  get_req.add_selections()->set_artifact_id(miss_artifact_id);
  populate_two_item_host_shared_region_layout(
      get_req.mutable_target_layout(),
      target_region,
      hit_artifact_id,
      hit_payload.size(),
      miss_artifact_id,
      total_bytes - hit_payload.size());
  get_req.set_pid(target_region.owner_pid);
  get_req.set_operation_id("op-host-shared-get-partial");

  BatchGetIntoRegionResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.BatchGetIntoRegion(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.outcomes_size() == 2);
  REQUIRE(get_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_MISS);
  REQUIRE(std::memcmp(target_region.base_ptr, hit_payload.data(), hit_payload.size()) == 0);
  const auto* miss_ptr = static_cast<const unsigned char*>(target_region.base_ptr) + hit_payload.size();
  for (std::size_t i = 0; i < total_bytes - hit_payload.size(); ++i) {
    REQUIRE(miss_ptr[i] == 0);
  }

  BatchGetIntoRegionRequest retry_req;
  retry_req.add_selections()->set_artifact_id(hit_artifact_id);
  populate_single_host_shared_region_layout(
      retry_req.mutable_target_layout(), target_region, hit_artifact_id, hit_payload.size(), hit_payload.size());
  retry_req.set_pid(target_region.owner_pid);
  retry_req.set_operation_id("op-host-shared-get-retry");

  BatchGetIntoRegionResponse retry_resp;
  grpc::ServerContext retry_ctx;
  REQUIRE(svc.BatchGetIntoRegion(&retry_ctx, &retry_req, &retry_resp).ok());
  REQUIRE(retry_resp.outcomes_size() == 1);
  REQUIRE(retry_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(std::memcmp(target_region.base_ptr, hit_payload.data(), hit_payload.size()) == 0);

  release_test_host_shared_region(*harness, target_region);
}

TEST_CASE(
    "BatchGetIntoRegion echoes allocator slot tokens on partial-hit HOST_SHARED outcomes",
    "[daemon][batch][host_shared][slot_token][partial]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string hit_artifact_id = make_valid_byte_artifact_id(
      "tenant",
      "sglang",
      "meta-llama/Llama-3.1-8B-Instruct",
      "default",
      "layout_v1",
      "request-host-target-slot-hit:blk-1");
  const std::string miss_artifact_id = make_valid_byte_artifact_id(
      "tenant",
      "sglang",
      "meta-llama/Llama-3.1-8B-Instruct",
      "default",
      "layout_v1",
      "request-host-target-slot-miss:blk-2");
  const std::string hit_payload = "host-target-slot-hit";
  const std::size_t total_bytes = hit_payload.size() + 32;
  constexpr std::uint64_t kHitSlotIndex = 17;
  constexpr std::uint64_t kHitSlotGeneration = 101;
  constexpr std::uint64_t kMissSlotIndex = 23;
  constexpr std::uint64_t kMissSlotGeneration = 205;

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* put_item = put_req.add_items();
  put_item->mutable_selection()->set_artifact_id(hit_artifact_id);
  put_item->set_inline_payload(hit_payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", hit_payload);

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  auto target_region = register_test_host_shared_region(
      *harness, total_bytes, tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator);
  std::memset(target_region.base_ptr, 0, total_bytes);

  BatchGetIntoRegionRequest get_req;
  get_req.add_selections()->set_artifact_id(hit_artifact_id);
  get_req.add_selections()->set_artifact_id(miss_artifact_id);
  populate_two_item_host_shared_region_layout(
      get_req.mutable_target_layout(),
      target_region,
      hit_artifact_id,
      hit_payload.size(),
      miss_artifact_id,
      total_bytes - hit_payload.size());
  get_req.mutable_target_layout()->mutable_offsets(0)->set_slot_index(kHitSlotIndex);
  get_req.mutable_target_layout()->mutable_offsets(0)->set_slot_generation(kHitSlotGeneration);
  get_req.mutable_target_layout()->mutable_offsets(1)->set_slot_index(kMissSlotIndex);
  get_req.mutable_target_layout()->mutable_offsets(1)->set_slot_generation(kMissSlotGeneration);
  get_req.set_pid(target_region.owner_pid);
  get_req.set_operation_id("op-host-shared-get-partial-slot-token");

  BatchGetIntoRegionResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.BatchGetIntoRegion(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.outcomes_size() == 2);
  REQUIRE(get_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.outcomes(0).has_slot_index());
  REQUIRE(get_resp.outcomes(0).has_slot_generation());
  REQUIRE(get_resp.outcomes(0).slot_index() == kHitSlotIndex);
  REQUIRE(get_resp.outcomes(0).slot_generation() == kHitSlotGeneration);
  REQUIRE(get_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_MISS);
  REQUIRE(get_resp.outcomes(1).has_slot_index());
  REQUIRE(get_resp.outcomes(1).has_slot_generation());
  REQUIRE(get_resp.outcomes(1).slot_index() == kMissSlotIndex);
  REQUIRE(get_resp.outcomes(1).slot_generation() == kMissSlotGeneration);
  REQUIRE(std::memcmp(target_region.base_ptr, hit_payload.data(), hit_payload.size()) == 0);
  const auto* miss_ptr = static_cast<const unsigned char*>(target_region.base_ptr) + hit_payload.size();
  for (std::size_t i = 0; i < total_bytes - hit_payload.size(); ++i) {
    REQUIRE(miss_ptr[i] == 0);
  }

  release_test_host_shared_region(*harness, target_region);
}

TEST_CASE(
    "BatchGetIntoRegion echoes allocator slot tokens on partial-failure HOST_SHARED outcomes",
    "[daemon][batch][host_shared][slot_token][failure]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string ok_artifact_id = make_valid_byte_artifact_id(
      "tenant",
      "sglang",
      "meta-llama/Llama-3.1-8B-Instruct",
      "default",
      "layout_v1",
      "request-host-target-slot-ok:blk-1");
  const std::string fail_artifact_id = make_valid_byte_artifact_id(
      "tenant",
      "sglang",
      "meta-llama/Llama-3.1-8B-Instruct",
      "default",
      "layout_v1",
      "request-host-target-slot-fail:blk-2");
  const std::string ok_payload = "host-target-slot-ok";
  const std::string fail_payload = "host-target-slot-fail";
  REQUIRE(fail_payload.size() > 1);
  constexpr std::uint64_t kOkSlotIndex = 29;
  constexpr std::uint64_t kOkSlotGeneration = 301;
  constexpr std::uint64_t kFailSlotIndex = 31;
  constexpr std::uint64_t kFailSlotGeneration = 407;

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* ok_put_item = put_req.add_items();
  ok_put_item->mutable_selection()->set_artifact_id(ok_artifact_id);
  ok_put_item->set_inline_payload(ok_payload);
  set_invariant(ok_put_item->mutable_invariant(), "layout_v1", ok_payload);
  auto* fail_put_item = put_req.add_items();
  fail_put_item->mutable_selection()->set_artifact_id(fail_artifact_id);
  fail_put_item->set_inline_payload(fail_payload);
  set_invariant(fail_put_item->mutable_invariant(), "layout_v1", fail_payload);

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 2);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(put_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  const std::size_t fail_target_length = fail_payload.size() - 1;
  auto target_region = register_test_host_shared_region(
      *harness,
      ok_payload.size() + fail_payload.size(),
      tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator);
  std::memset(target_region.base_ptr, 0, target_region.size_bytes);

  BatchGetIntoRegionRequest get_req;
  get_req.add_selections()->set_artifact_id(ok_artifact_id);
  get_req.add_selections()->set_artifact_id(fail_artifact_id);
  populate_two_item_host_shared_region_layout(
      get_req.mutable_target_layout(),
      target_region,
      ok_artifact_id,
      ok_payload.size(),
      fail_artifact_id,
      fail_target_length);
  get_req.mutable_target_layout()->mutable_offsets(0)->set_slot_index(kOkSlotIndex);
  get_req.mutable_target_layout()->mutable_offsets(0)->set_slot_generation(kOkSlotGeneration);
  get_req.mutable_target_layout()->mutable_offsets(1)->set_slot_index(kFailSlotIndex);
  get_req.mutable_target_layout()->mutable_offsets(1)->set_slot_generation(kFailSlotGeneration);
  get_req.set_pid(target_region.owner_pid);
  get_req.set_operation_id("op-host-shared-get-partial-failure-slot-token");

  BatchGetIntoRegionResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.BatchGetIntoRegion(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.outcomes_size() == 2);

  REQUIRE(get_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.outcomes(0).has_slot_index());
  REQUIRE(get_resp.outcomes(0).has_slot_generation());
  REQUIRE(get_resp.outcomes(0).slot_index() == kOkSlotIndex);
  REQUIRE(get_resp.outcomes(0).slot_generation() == kOkSlotGeneration);

  REQUIRE(get_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_FAILED_PRECONDITION);
  REQUIRE(get_resp.outcomes(1).has_slot_index());
  REQUIRE(get_resp.outcomes(1).has_slot_generation());
  REQUIRE(get_resp.outcomes(1).slot_index() == kFailSlotIndex);
  REQUIRE(get_resp.outcomes(1).slot_generation() == kFailSlotGeneration);
  REQUIRE(get_resp.outcomes(1).message().find("payload size does not match target layout length") != std::string::npos);

  REQUIRE(std::memcmp(target_region.base_ptr, ok_payload.data(), ok_payload.size()) == 0);
  const auto* fail_ptr = static_cast<const unsigned char*>(target_region.base_ptr) + ok_payload.size();
  for (std::size_t i = 0; i < fail_target_length; ++i) {
    REQUIRE(fail_ptr[i] == 0);
  }

  release_test_host_shared_region(*harness, target_region);
}

TEST_CASE(
    "BatchGetIntoRegion preserves requested slot_generation across allocator slot reuse",
    "[daemon][batch][host_shared][slot_token][generation]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_valid_byte_artifact_id(
      "tenant",
      "sglang",
      "meta-llama/Llama-3.1-8B-Instruct",
      "default",
      "layout_v1",
      "request-host-target-slot-generation:blk-1");
  const std::string payload = "host-target-slot-generation";
  constexpr std::uint64_t kSlotIndex = 43;
  constexpr std::uint64_t kFirstGeneration = 501;
  constexpr std::uint64_t kSecondGeneration = 502;

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* put_item = put_req.add_items();
  put_item->mutable_selection()->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  auto target_region = register_test_host_shared_region(
      *harness, payload.size(), tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator);

  const auto run_get = [&](std::uint64_t slot_generation, std::string_view operation_id) {
    std::memset(target_region.base_ptr, 0, payload.size());

    BatchGetIntoRegionRequest get_req;
    get_req.add_selections()->set_artifact_id(artifact_id);
    populate_single_host_shared_region_layout(
        get_req.mutable_target_layout(), target_region, artifact_id, payload.size(), payload.size());
    get_req.mutable_target_layout()->mutable_offsets(0)->set_slot_index(kSlotIndex);
    get_req.mutable_target_layout()->mutable_offsets(0)->set_slot_generation(slot_generation);
    get_req.set_pid(target_region.owner_pid);
    get_req.set_operation_id(std::string(operation_id));

    BatchGetIntoRegionResponse get_resp;
    grpc::ServerContext get_ctx;
    REQUIRE(svc.BatchGetIntoRegion(&get_ctx, &get_req, &get_resp).ok());
    REQUIRE(get_resp.outcomes_size() == 1);
    REQUIRE(get_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
    REQUIRE(get_resp.outcomes(0).has_slot_index());
    REQUIRE(get_resp.outcomes(0).has_slot_generation());
    REQUIRE(get_resp.outcomes(0).slot_index() == kSlotIndex);
    REQUIRE(get_resp.outcomes(0).slot_generation() == slot_generation);
    REQUIRE(std::memcmp(target_region.base_ptr, payload.data(), payload.size()) == 0);
  };

  run_get(kFirstGeneration, "op-host-shared-get-slot-generation-first");
  run_get(kSecondGeneration, "op-host-shared-get-slot-generation-second");

  release_test_host_shared_region(*harness, target_region);
}

TEST_CASE("BatchGetIntoRegion rejects invalid HOST_SHARED target bounds", "[daemon][batch][host_shared][get]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_valid_byte_artifact_id(
      "tenant",
      "sglang",
      "meta-llama/Llama-3.1-8B-Instruct",
      "default",
      "layout_v1",
      "request-host-target-bounds:blk-1");
  const std::string payload = "host-target-invalid-bounds";

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* put_item = put_req.add_items();
  put_item->mutable_selection()->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  auto target_region = register_test_host_shared_region(*harness, payload.size());
  std::memset(target_region.base_ptr, 0, payload.size());

  BatchGetIntoRegionRequest get_req;
  get_req.add_selections()->set_artifact_id(artifact_id);
  populate_single_host_shared_region_layout(
      get_req.mutable_target_layout(), target_region, artifact_id, payload.size(), payload.size() - 1);
  get_req.set_pid(target_region.owner_pid);
  get_req.set_operation_id("op-host-shared-get-bounds");

  BatchGetIntoRegionResponse get_resp;
  grpc::ServerContext get_ctx;
  const auto get_status = svc.BatchGetIntoRegion(&get_ctx, &get_req, &get_resp);
  REQUIRE_FALSE(get_status.ok());
  REQUIRE(get_status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);

  release_test_host_shared_region(*harness, target_region);
}

TEST_CASE("HomeBatchGet emits batch transport for large payloads", "[daemon][batch][batch_payload_ref][get]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.max_batch_payload_bytes = 1ULL << 20;
  options.byte_artifact_routing.payload_transport.max_batch_items = 8;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id_a = make_test_byte_artifact_id("batch-get-a:blk-5a");
  const std::string artifact_id_b = artifact_on_same_shard(artifact_id_a, "batchget");
  const std::string payload_a(64, 'a');
  const std::string payload_b(96, 'b');
  const std::uint64_t shard_id = shard_for_artifact(artifact_id_a);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-home-batch-get-pack");
  auto* item_a = put_req.add_items();
  item_a->set_artifact_id(artifact_id_a);
  item_a->set_inline_payload(payload_a);
  set_invariant(item_a->mutable_invariant(), "layout_v1", payload_a);
  auto* item_b = put_req.add_items();
  item_b->set_artifact_id(artifact_id_b);
  item_b->set_inline_payload(payload_b);
  set_invariant(item_b->mutable_invariant(), "layout_v1", payload_b);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 2);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(put_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  HomeBatchGetRequest get_req;
  get_req.mutable_fence()->CopyFrom(put_req.fence());
  get_req.set_operation_id("op-home-batch-get-pack");
  get_req.add_artifact_ids(artifact_id_a);
  get_req.add_artifact_ids(artifact_id_b);
  HomeBatchGetResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.HomeBatchGet(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.items_size() == 2);
  REQUIRE(get_resp.batch_transports_size() == 1);
  REQUIRE(get_resp.items(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.items(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.items(0).inline_payload().empty());
  REQUIRE(get_resp.items(1).inline_payload().empty());
  REQUIRE(get_resp.items(0).payload_ref().empty());
  REQUIRE(get_resp.items(1).payload_ref().empty());
  REQUIRE(get_resp.items(0).has_batch_payload_slice());
  REQUIRE(get_resp.items(1).has_batch_payload_slice());
  REQUIRE(get_resp.items(0).batch_payload_slice().transport_id() == "batch-transport-1");
  REQUIRE(get_resp.items(1).batch_payload_slice().transport_id() == "batch-transport-1");

  const auto& transport = get_resp.batch_transports(0);
  REQUIRE(transport.transport_id() == "batch-transport-1");
  REQUIRE(transport.has_grpc_chunk_ref());
  REQUIRE(transport.manifest().entries_size() == 2);
  REQUIRE(transport.manifest().entries(0).artifact_id() == artifact_id_a);
  REQUIRE(transport.manifest().entries(1).artifact_id() == artifact_id_b);
  REQUIRE(transport.manifest().entries(0).length() == payload_a.size());
  REQUIRE(transport.manifest().entries(1).length() == payload_b.size());
  REQUIRE(transport.manifest().total_size() == payload_a.size() + payload_b.size());

  const auto resolved_or = harness->kernel().payload_transport_broker().fetch_batch_payload_ref(
      harness->kernel().worker_directory_cache(),
      absl::Now(),
      absl::Seconds(1),
      kDaemonId,
      transport.grpc_chunk_ref().batch_payload_ref(),
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-home-batch-get-pack");
  REQUIRE(resolved_or.ok());
  REQUIRE_FALSE(resolved_or->remote);
  REQUIRE(resolved_or->payload != nullptr);
  REQUIRE(*resolved_or->payload == payload_a + payload_b);
}

TEST_CASE("HomeBatchPutIfAbsent accepts batch payload slices", "[daemon][batch][batch_payload_ref][put]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 1ULL << 20;
  options.byte_artifact_routing.payload_transport.max_batch_payload_bytes = 1ULL << 20;
  options.byte_artifact_routing.payload_transport.max_batch_items = 8;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id_a = make_test_byte_artifact_id("batch-put-a:blk-6a");
  const std::string artifact_id_b = artifact_on_same_shard(artifact_id_a, "batchput");
  const std::string payload_a = "batch-put-alpha";
  const std::string payload_b = "batch-put-beta-more-bytes";
  const std::string slab = payload_a + payload_b;
  const std::uint64_t shard_id = shard_for_artifact(artifact_id_a);

  tensorcast::daemon::v2::BatchPayloadManifest manifest;
  auto* entry_a = manifest.add_entries();
  entry_a->set_artifact_id(artifact_id_a);
  entry_a->set_offset(0);
  entry_a->set_length(payload_a.size());
  entry_a->set_digest_alg("sha256");
  entry_a->set_digest_hex(sha256_hex(payload_a));
  auto* entry_b = manifest.add_entries();
  entry_b->set_artifact_id(artifact_id_b);
  entry_b->set_offset(payload_a.size());
  entry_b->set_length(payload_b.size());
  entry_b->set_digest_alg("sha256");
  entry_b->set_digest_hex(sha256_hex(payload_b));
  manifest.set_total_size(slab.size());

  auto batch_payload_ref_or = harness->kernel().payload_transport_broker().issue_batch_payload_ref(
      manifest,
      std::make_shared<const std::string>(slab),
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
      "op-home-batch-put-pack");
  REQUIRE(batch_payload_ref_or.ok());

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-home-batch-put-pack");
  auto* transport = put_req.add_batch_transports();
  transport->set_transport_id("batch-transport-1");
  transport->mutable_manifest()->CopyFrom(manifest);
  transport->mutable_grpc_chunk_ref()->set_batch_payload_ref(*batch_payload_ref_or);
  transport->mutable_grpc_chunk_ref()->set_protocol_version(1);

  auto* item_a = put_req.add_items();
  item_a->set_artifact_id(artifact_id_a);
  set_invariant(item_a->mutable_invariant(), "layout_v1", payload_a);
  item_a->mutable_batch_payload_slice()->set_transport_id("batch-transport-1");
  item_a->mutable_batch_payload_slice()->set_offset(0);
  item_a->mutable_batch_payload_slice()->set_length(payload_a.size());

  auto* item_b = put_req.add_items();
  item_b->set_artifact_id(artifact_id_b);
  set_invariant(item_b->mutable_invariant(), "layout_v1", payload_b);
  item_b->mutable_batch_payload_slice()->set_transport_id("batch-transport-1");
  item_b->mutable_batch_payload_slice()->set_offset(payload_a.size());
  item_b->mutable_batch_payload_slice()->set_length(payload_b.size());

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 2);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(put_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  HomeBatchGetRequest get_req;
  get_req.mutable_fence()->CopyFrom(put_req.fence());
  get_req.add_artifact_ids(artifact_id_a);
  get_req.add_artifact_ids(artifact_id_b);
  HomeBatchGetResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.HomeBatchGet(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.items_size() == 2);
  REQUIRE(get_resp.items(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.items(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.items(0).inline_payload() == payload_a);
  REQUIRE(get_resp.items(1).inline_payload() == payload_b);
}

TEST_CASE(
    "HomeBatchPutIfAbsent consumes remote direct communicator slices without full-pack mirror",
    "[daemon][batch][batch_payload_ref][put][communicator]") {
  constexpr std::string_view kSourceDaemonId = "daemon-put-source-direct";
  constexpr std::string_view kHomeDaemonId = "daemon-put-home-direct";
  constexpr std::string_view kHost = "127.0.0.1";
  constexpr std::string_view kLocalEndpoint = "pcie-home-put-direct";
  constexpr std::string_view kRemoteEndpoint = "pcie-source-put-direct";

  auto source_opts = make_opts_basic();
  source_opts.comm_manager =
      make_comm_manager_with_config(make_single_node_communicator_config(/*node_index=*/0, /*nic_name=*/"eth0"));
  auto source_engine = std::make_shared<tensorcast::store::StoreEngine>(source_opts);

  auto home_opts = make_opts_basic();
  home_opts.comm_manager =
      make_comm_manager_with_config(make_single_node_communicator_config(/*node_index=*/1, /*nic_name=*/"eth1"));
  install_pcie_batch_payload_routing_context(
      home_opts.comm_manager,
      kLocalEndpoint,
      kRemoteEndpoint,
      kHost,
      home_opts.comm_manager->listen_port(),
      kHost,
      source_opts.comm_manager->listen_port());
  auto home_engine = std::make_shared<tensorcast::store::StoreEngine>(home_opts);

  auto source_daemon_options = make_daemon_options();
  source_daemon_options.daemon_id = std::string(kSourceDaemonId);
  source_daemon_options.storage_path = make_test_storage_root("batch-runtime-put-direct-source");
  auto source = make_harness(source_engine, source_daemon_options);

  auto home_daemon_options = make_daemon_options();
  home_daemon_options.daemon_id = std::string(kHomeDaemonId);
  home_daemon_options.storage_path = make_test_storage_root("batch-runtime-put-direct-home");
  auto home = make_harness(home_engine, home_daemon_options);

  const std::string artifact_id_a = make_test_byte_artifact_id("batch-put-direct-a:blk-6a");
  const std::string artifact_id_b = artifact_on_same_shard(artifact_id_a, "batchputdirect");
  const std::string payload_a = "remote-direct-put-alpha";
  const std::string payload_b = "remote-direct-put-beta-more-bytes";
  const std::string slab = payload_a + payload_b;
  const std::uint64_t shard_id = shard_for_artifact(artifact_id_a);

  tensorcast::daemon::v2::BatchPayloadManifest manifest;
  auto* entry_a = manifest.add_entries();
  entry_a->set_artifact_id(artifact_id_a);
  entry_a->set_offset(0);
  entry_a->set_length(payload_a.size());
  entry_a->set_digest_alg("sha256");
  entry_a->set_digest_hex(sha256_hex(payload_a));
  auto* entry_b = manifest.add_entries();
  entry_b->set_artifact_id(artifact_id_b);
  entry_b->set_offset(payload_a.size());
  entry_b->set_length(payload_b.size());
  entry_b->set_digest_alg("sha256");
  entry_b->set_digest_hex(sha256_hex(payload_b));
  manifest.set_total_size(slab.size());

  auto export_or = source->kernel().payload_transport_broker().issue_batch_payload_communicator_export(
      manifest,
      std::make_shared<const std::string>(slab),
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
      /*operation_id=*/"op-home-batch-put-direct-slice",
      absl::Now() + absl::Minutes(1),
      kHomeDaemonId);
  REQUIRE(export_or.ok());

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(std::string(kHomeDaemonId));
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-home-batch-put-direct-slice");
  auto* transport = put_req.add_batch_transports();
  transport->set_transport_id("transport-put-direct");
  transport->mutable_manifest()->CopyFrom(manifest);
  auto* communicator_source = transport->mutable_communicator_source();
  communicator_source->set_batch_payload_ref(export_or->batch_payload_ref);
  communicator_source->set_protocol_version(2);
  communicator_source->set_producer_daemon_id(std::string(kSourceDaemonId));
  communicator_source->set_consumer_daemon_id(std::string(kHomeDaemonId));
  communicator_source->set_producer_host(std::string(kHost));
  communicator_source->set_producer_port(source_opts.comm_manager->listen_port());
  communicator_source->set_remote_endpoint_id(std::string(kRemoteEndpoint));
  communicator_source->set_local_endpoint_id_hint(std::string(kLocalEndpoint));
  communicator_source->set_memory_location(tensorcast::daemon::v2::BATCH_PAYLOAD_MEMORY_LOCATION_HOST);
  communicator_source->set_total_payload_bytes(slab.size());
  for (const auto& memory_key : export_or->export_registration.remote_memory_keys) {
    communicator_source->add_remote_memory_keys(memory_key);
  }
  for (const auto buffer_size : export_or->export_registration.buffer_sizes) {
    communicator_source->add_buffer_sizes(buffer_size);
  }

  auto* item_a = put_req.add_items();
  item_a->set_artifact_id(artifact_id_a);
  set_invariant(item_a->mutable_invariant(), "layout_v1", payload_a);
  item_a->mutable_batch_payload_slice()->set_transport_id("transport-put-direct");
  item_a->mutable_batch_payload_slice()->set_offset(0);
  item_a->mutable_batch_payload_slice()->set_length(payload_a.size());

  auto* item_b = put_req.add_items();
  item_b->set_artifact_id(artifact_id_b);
  set_invariant(item_b->mutable_invariant(), "layout_v1", payload_b);
  item_b->mutable_batch_payload_slice()->set_transport_id("transport-put-direct");
  item_b->mutable_batch_payload_slice()->set_offset(payload_a.size());
  item_b->mutable_batch_payload_slice()->set_length(payload_b.size());

  CollectingLogSink sink;
  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  {
    ScopedVLogLevel scoped_vlog(/*level=*/2);
    ScopedCollectingLogSink scoped_sink(sink);
    REQUIRE(home->service().HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  }
  REQUIRE(put_resp.outcomes_size() == 2);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(put_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  CHECK(sink.Contains(
      "byte_artifact.home_batch_put_if_absent_transport_read_mode "
      "operation_id=op-home-batch-put-direct-slice"));
  CHECK(sink.Contains("read_mode=direct_remote_slice"));
  CHECK(sink.Contains("realization=source_slice_loader"));
  CHECK_FALSE(sink.Contains(
      "byte_artifact.home_batch_put_if_absent_transport_mirror "
      "operation_id=op-home-batch-put-direct-slice"));

  HomeBatchGetRequest get_req;
  get_req.mutable_fence()->CopyFrom(put_req.fence());
  get_req.add_artifact_ids(artifact_id_a);
  get_req.add_artifact_ids(artifact_id_b);
  HomeBatchGetResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(home->service().HomeBatchGet(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.items_size() == 2);
  REQUIRE(get_resp.items(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.items(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.items(0).inline_payload() == payload_a);
  REQUIRE(get_resp.items(1).inline_payload() == payload_b);
}

TEST_CASE(
    "HomeBatchPutIfAbsent keeps full-pack mirror for remote non-direct communicator sources",
    "[daemon][batch][batch_payload_ref][put][communicator]") {
  constexpr std::string_view kSourceDaemonId = "daemon-put-source-fallback";
  constexpr std::string_view kHomeDaemonId = "daemon-put-home-fallback";
  constexpr std::string_view kSourceHost = "127.0.0.1";

  auto source_opts = make_opts_basic();
  source_opts.comm_manager =
      make_comm_manager_with_config(make_single_node_communicator_config(/*node_index=*/0, /*nic_name=*/"eth0"));
  auto source_engine = std::make_shared<tensorcast::store::StoreEngine>(source_opts);

  auto home_opts = make_opts_basic();
  home_opts.comm_manager =
      make_comm_manager_with_config(make_single_node_communicator_config(/*node_index=*/1, /*nic_name=*/"eth1"));
  auto home_engine = std::make_shared<tensorcast::store::StoreEngine>(home_opts);

  auto source_daemon_options = make_daemon_options();
  source_daemon_options.daemon_id = std::string(kSourceDaemonId);
  source_daemon_options.storage_path = make_test_storage_root("batch-runtime-put-fallback-source");
  auto source = make_harness(source_engine, source_daemon_options);

  auto home_daemon_options = make_daemon_options();
  home_daemon_options.daemon_id = std::string(kHomeDaemonId);
  home_daemon_options.storage_path = make_test_storage_root("batch-runtime-put-fallback-home");
  auto home = make_harness(home_engine, home_daemon_options);

  const std::string artifact_id = make_test_byte_artifact_id("batch-put-fallback-a:blk-6a");
  const std::string payload = "remote-fallback-put-payload";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  tensorcast::daemon::v2::BatchPayloadManifest manifest;
  auto* entry = manifest.add_entries();
  entry->set_artifact_id(artifact_id);
  entry->set_offset(0);
  entry->set_length(payload.size());
  entry->set_digest_alg("sha256");
  entry->set_digest_hex(sha256_hex(payload));
  manifest.set_total_size(payload.size());

  auto export_or = source->kernel().payload_transport_broker().issue_batch_payload_communicator_export(
      manifest,
      std::make_shared<const std::string>(payload),
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
      /*operation_id=*/"op-home-batch-put-fallback-mirror",
      absl::Now() + absl::Minutes(1),
      kHomeDaemonId);
  REQUIRE(export_or.ok());

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(std::string(kHomeDaemonId));
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-home-batch-put-fallback-mirror");
  auto* transport = put_req.add_batch_transports();
  transport->set_transport_id("transport-put-fallback");
  transport->mutable_manifest()->CopyFrom(manifest);
  auto* communicator_source = transport->mutable_communicator_source();
  communicator_source->set_batch_payload_ref(export_or->batch_payload_ref);
  communicator_source->set_protocol_version(2);
  communicator_source->set_producer_daemon_id(std::string(kSourceDaemonId));
  communicator_source->set_consumer_daemon_id(std::string(kHomeDaemonId));
  communicator_source->set_producer_host(std::string(kSourceHost));
  communicator_source->set_producer_port(source_opts.comm_manager->listen_port());
  communicator_source->set_remote_endpoint_id(
      tensorcast::store::components::derive_endpoint_id(
          "node-put-source-fallback", tensorcast::common::memory::MemoryLocation::CPU, /*device_id=*/0));
  communicator_source->set_memory_location(tensorcast::daemon::v2::BATCH_PAYLOAD_MEMORY_LOCATION_HOST);
  communicator_source->set_total_payload_bytes(payload.size());
  for (const auto& memory_key : export_or->export_registration.remote_memory_keys) {
    communicator_source->add_remote_memory_keys(memory_key);
  }
  for (const auto buffer_size : export_or->export_registration.buffer_sizes) {
    communicator_source->add_buffer_sizes(buffer_size);
  }

  auto* item = put_req.add_items();
  item->set_artifact_id(artifact_id);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);
  item->mutable_batch_payload_slice()->set_transport_id("transport-put-fallback");
  item->mutable_batch_payload_slice()->set_offset(0);
  item->mutable_batch_payload_slice()->set_length(payload.size());

  CollectingLogSink sink;
  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  {
    ScopedVLogLevel scoped_vlog(/*level=*/2);
    ScopedCollectingLogSink scoped_sink(sink);
    REQUIRE(home->service().HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  }
  REQUIRE(put_resp.outcomes_size() == 1);
  INFO(put_resp.outcomes(0).message());
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  CHECK(sink.Contains(
      "byte_artifact.home_batch_put_if_absent_transport_mirror "
      "operation_id=op-home-batch-put-fallback-mirror"));
  CHECK(sink.Contains("realization=full_pack_mirror"));
  CHECK_FALSE(sink.Contains(
      "byte_artifact.home_batch_put_if_absent_transport_read_mode "
      "operation_id=op-home-batch-put-fallback-mirror"));
}

TEST_CASE(
    "HomeBatchGet emits segmented communicator transport without staged batch payload slabs",
    "[daemon][batch][batch_payload_ref][get][communicator]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.max_batch_payload_bytes = 1ULL << 20;
  options.byte_artifact_routing.payload_transport.max_batch_items = 8;
  auto harness = make_harness(engine, options);
  install_local_worker_directory_entry(*harness, kDaemonId);
  auto& svc = harness->service();

  const std::string artifact_id_a = make_test_byte_artifact_id("batch-get-segmented-a:blk-5b");
  const std::string artifact_id_b = artifact_on_same_shard(artifact_id_a, "batchgetsegmented");
  const std::string payload_a(64, 'a');
  const std::string payload_b(96, 'b');
  const std::uint64_t shard_id = shard_for_artifact(artifact_id_a);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-home-batch-get-segmented");
  auto* item_a = put_req.add_items();
  item_a->set_artifact_id(artifact_id_a);
  item_a->set_inline_payload(payload_a);
  set_invariant(item_a->mutable_invariant(), "layout_v1", payload_a);
  auto* item_b = put_req.add_items();
  item_b->set_artifact_id(artifact_id_b);
  item_b->set_inline_payload(payload_b);
  set_invariant(item_b->mutable_invariant(), "layout_v1", payload_b);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 2);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(put_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  HomeBatchGetRequest get_req;
  get_req.mutable_fence()->CopyFrom(put_req.fence());
  get_req.set_operation_id("op-home-batch-get-segmented");
  get_req.set_requester_daemon_id(kDaemonId);
  get_req.add_artifact_ids(artifact_id_a);
  get_req.add_artifact_ids(artifact_id_b);

  HomeBatchGetResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.HomeBatchGet(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.items_size() == 2);
  REQUIRE(get_resp.batch_transports_size() == 1);
  REQUIRE(get_resp.items(0).has_batch_payload_slice());
  REQUIRE(get_resp.items(1).has_batch_payload_slice());

  const auto& transport = get_resp.batch_transports(0);
  REQUIRE(transport.has_communicator_source());
  REQUIRE_FALSE(transport.has_grpc_chunk_ref());
  REQUIRE(transport.communicator_source().remote_memory_keys_size() == 2);
  REQUIRE(transport.communicator_source().buffer_sizes_size() == 2);
  REQUIRE(transport.communicator_source().buffer_sizes(0) == payload_a.size());
  REQUIRE(transport.communicator_source().buffer_sizes(1) == payload_b.size());
  REQUIRE(transport.communicator_source().total_payload_bytes() == payload_a.size() + payload_b.size());
  const std::string expected_cpu_endpoint = tensorcast::store::components::derive_endpoint_id(
      "node-local", tensorcast::common::memory::MemoryLocation::CPU, /*device_id=*/0);
  CHECK(transport.communicator_source().remote_endpoint_id() == expected_cpu_endpoint);
  CHECK(transport.communicator_source().local_endpoint_id_hint().empty());

  auto chunk_or = harness->kernel().payload_transport_broker().read_local_batch_payload_ref_chunk(
      transport.communicator_source().batch_payload_ref(),
      absl::Now(),
      /*offset=*/0,
      /*max_bytes=*/16,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-home-batch-get-segmented");
  REQUIRE_FALSE(chunk_or.ok());
  REQUIRE(chunk_or.status().code() == absl::StatusCode::kFailedPrecondition);

  auto source_or = harness->kernel().payload_transport_broker().open_batch_payload_communicator_source(
      harness->kernel().worker_directory_cache(),
      absl::Now(),
      absl::Seconds(1),
      kDaemonId,
      transport,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-home-batch-get-segmented");
  REQUIRE(source_or.ok());
  REQUIRE_FALSE(source_or->remote);
  REQUIRE(source_or->source != nullptr);

  std::string joined(payload_a.size() + payload_b.size(), '\0');
  auto read_or = source_or->source->read_at(/*offset=*/0, joined.data(), joined.size());
  REQUIRE(read_or.ok());
  REQUIRE(*read_or == joined.size());
  REQUIRE(joined == payload_a + payload_b);
  REQUIRE(source_or->source->supports_direct_write_at());

  std::vector<std::uint8_t> direct_write_buffer(joined.size() + 16, 0);
  tensorcast::store::DirectWriteGrant grant;
  grant.windows.push_back(
      tensorcast::store::DirectWriteGrant::Window{
          .va_offset = 11,
          .local_addr = reinterpret_cast<std::uint64_t>(direct_write_buffer.data()),
          .length = static_cast<std::uint64_t>(direct_write_buffer.size()),
      });
  const auto direct_expected = joined.substr(payload_a.size() - 4, 8);
  auto wrote_direct_or = source_or->source->read_into_at(
      /*src_offset=*/payload_a.size() - 4,
      /*dest_va_offset=*/13,
      /*bytes=*/direct_expected.size(),
      grant);
  REQUIRE(wrote_direct_or.ok());
  REQUIRE(*wrote_direct_or == direct_expected.size());
  REQUIRE(
      std::string(reinterpret_cast<const char*>(direct_write_buffer.data() + 2), direct_expected.size()) ==
      direct_expected);
}

TEST_CASE(
    "HomeBatchGet falls back to staged communicator transport when segmented policy is disabled",
    "[daemon][batch][batch_payload_ref][get][communicator]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.max_batch_payload_bytes = 1ULL << 20;
  options.byte_artifact_routing.payload_transport.max_batch_items = 8;
  options.byte_artifact_routing.payload_transport.segmented_communicator_export_enabled = false;
  auto harness = make_harness(engine, options);
  install_local_worker_directory_entry(*harness, kDaemonId);
  auto& svc = harness->service();

  const std::string artifact_id_a = make_test_byte_artifact_id("batch-get-staged-a:blk-5c");
  const std::string artifact_id_b = artifact_on_same_shard(artifact_id_a, "batchgetstaged");
  const std::string payload_a(64, 'c');
  const std::string payload_b(96, 'd');
  const std::uint64_t shard_id = shard_for_artifact(artifact_id_a);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-home-batch-get-staged");
  auto* item_a = put_req.add_items();
  item_a->set_artifact_id(artifact_id_a);
  item_a->set_inline_payload(payload_a);
  set_invariant(item_a->mutable_invariant(), "layout_v1", payload_a);
  auto* item_b = put_req.add_items();
  item_b->set_artifact_id(artifact_id_b);
  item_b->set_inline_payload(payload_b);
  set_invariant(item_b->mutable_invariant(), "layout_v1", payload_b);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 2);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(put_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  HomeBatchGetRequest get_req;
  get_req.mutable_fence()->CopyFrom(put_req.fence());
  get_req.set_operation_id("op-home-batch-get-staged");
  get_req.add_artifact_ids(artifact_id_a);
  get_req.add_artifact_ids(artifact_id_b);

  HomeBatchGetResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.HomeBatchGet(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.items_size() == 2);
  REQUIRE(get_resp.batch_transports_size() == 1);
  REQUIRE(get_resp.items(0).has_batch_payload_slice());
  REQUIRE(get_resp.items(1).has_batch_payload_slice());

  const auto& transport = get_resp.batch_transports(0);
  REQUIRE(transport.has_communicator_source());
  REQUIRE_FALSE(transport.has_grpc_chunk_ref());
  REQUIRE(transport.communicator_source().remote_memory_keys_size() == 1);
  REQUIRE(transport.communicator_source().buffer_sizes_size() == 1);
  REQUIRE(transport.communicator_source().total_payload_bytes() == payload_a.size() + payload_b.size());

  auto resolved_or = harness->kernel().payload_transport_broker().fetch_batch_payload_ref(
      harness->kernel().worker_directory_cache(),
      absl::Now(),
      absl::Seconds(1),
      kDaemonId,
      transport.communicator_source().batch_payload_ref(),
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-home-batch-get-staged");
  REQUIRE(resolved_or.ok());
  REQUIRE_FALSE(resolved_or->remote);
  REQUIRE(resolved_or->payload != nullptr);
  REQUIRE(*resolved_or->payload == payload_a + payload_b);
}

TEST_CASE(
    "OpenBatchPayloadCommunicatorSource leaves routing context unset for remote communicator sources",
    "[daemon][batch][batch_payload_ref][get][communicator][routing]") {
  constexpr std::string_view kHolderDaemonId = "daemon-holder";
  constexpr std::string_view kRequesterDaemonId = "daemon-requester";
  constexpr std::string_view kHolderNodeId = "node1";
  constexpr std::string_view kRequesterNodeId = "node0";
  constexpr std::string_view kHolderHost = "127.0.0.1";
  constexpr std::string_view kRequesterHost = "127.0.0.1";
  constexpr uint32_t kHolderGrpcPort = 18181;
  constexpr uint32_t kRequesterGrpcPort = 18182;

  auto holder_opts = make_opts_basic();
  holder_opts.comm_manager =
      make_comm_manager_with_config(make_single_node_communicator_config(/*node_index=*/1, /*nic_name=*/"eth1"));
  auto holder_engine = std::make_shared<tensorcast::store::StoreEngine>(holder_opts);

  auto requester_opts = make_opts_basic();
  requester_opts.comm_manager =
      make_comm_manager_with_config(make_single_node_communicator_config(/*node_index=*/0, /*nic_name=*/"eth0"));
  auto requester_engine = std::make_shared<tensorcast::store::StoreEngine>(requester_opts);

  auto global_store_client = std::make_shared<WorkerDirectoryTestGlobalStoreClient>();
  global_store_client->active_workers = {
      tensorcast::store::components::ActiveWorkerInfo{
          .worker_id = "worker-holder",
          .node_id = std::string(kHolderNodeId),
          .node_address = std::string(kHolderHost),
          .grpc_port = kHolderGrpcPort,
          .p2p_port = holder_opts.comm_manager->listen_port(),
          .accepting_new_requests = true,
          .daemon_id = std::string(kHolderDaemonId),
          .capability_flags = 0,
      },
      tensorcast::store::components::ActiveWorkerInfo{
          .worker_id = "worker-requester",
          .node_id = std::string(kRequesterNodeId),
          .node_address = std::string(kRequesterHost),
          .grpc_port = kRequesterGrpcPort,
          .p2p_port = requester_opts.comm_manager->listen_port(),
          .accepting_new_requests = true,
          .daemon_id = std::string(kRequesterDaemonId),
          .capability_flags = 0,
      },
  };

  auto holder_daemon_options = make_daemon_options();
  holder_daemon_options.daemon_id = std::string(kHolderDaemonId);
  holder_daemon_options.storage_path = make_test_storage_root("batch-runtime-routing-holder");
  auto holder = make_harness(holder_engine, holder_daemon_options, global_store_client);
  install_worker_directory_entry(
      *holder,
      kHolderDaemonId,
      /*worker_id=*/"worker-holder",
      kHolderNodeId,
      kHolderHost,
      kHolderGrpcPort,
      holder_opts.comm_manager->listen_port());

  auto requester_daemon_options = make_daemon_options();
  requester_daemon_options.daemon_id = std::string(kRequesterDaemonId);
  requester_daemon_options.storage_path = make_test_storage_root("batch-runtime-routing-requester");
  auto requester = make_harness(requester_engine, requester_daemon_options, global_store_client);
  install_worker_directory_entry(
      *requester,
      kRequesterDaemonId,
      /*worker_id=*/"worker-requester",
      kRequesterNodeId,
      kRequesterHost,
      kRequesterGrpcPort,
      requester_opts.comm_manager->listen_port());

  const std::string payload = "remote-communicator-source-payload";
  tensorcast::daemon::v2::BatchPayloadManifest manifest;
  manifest.set_total_size(payload.size());
  auto* entry = manifest.add_entries();
  entry->set_artifact_id("artifact-routing");
  entry->set_offset(0);
  entry->set_length(payload.size());

  auto export_or = holder->kernel().payload_transport_broker().issue_batch_payload_communicator_export(
      manifest,
      std::make_shared<const std::string>(payload),
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      /*operation_id=*/"op-remote-routing",
      absl::Now() + absl::Minutes(1),
      kRequesterDaemonId);
  REQUIRE(export_or.ok());

  tensorcast::daemon::v2::BatchPayloadTransport transport;
  transport.set_transport_id("transport-routing");
  transport.mutable_manifest()->CopyFrom(manifest);
  auto* communicator_source = transport.mutable_communicator_source();
  communicator_source->set_batch_payload_ref(export_or->batch_payload_ref);
  communicator_source->set_protocol_version(2);
  communicator_source->set_producer_daemon_id(std::string(kHolderDaemonId));
  communicator_source->set_consumer_daemon_id(std::string(kRequesterDaemonId));
  communicator_source->set_producer_host(std::string(kHolderHost));
  communicator_source->set_producer_port(holder_opts.comm_manager->listen_port());
  communicator_source->set_remote_endpoint_id(
      tensorcast::store::components::derive_endpoint_id(
          kHolderNodeId, tensorcast::common::memory::MemoryLocation::CPU, /*device_id=*/0));
  communicator_source->set_local_endpoint_id_hint(
      tensorcast::store::components::derive_endpoint_id(
          kRequesterNodeId, tensorcast::common::memory::MemoryLocation::CPU, /*device_id=*/0));
  communicator_source->set_memory_location(tensorcast::daemon::v2::BATCH_PAYLOAD_MEMORY_LOCATION_HOST);
  communicator_source->set_total_payload_bytes(payload.size());
  for (const auto& memory_key : export_or->export_registration.remote_memory_keys) {
    communicator_source->add_remote_memory_keys(memory_key);
  }
  for (const auto buffer_size : export_or->export_registration.buffer_sizes) {
    communicator_source->add_buffer_sizes(buffer_size);
  }

  auto source_or = requester->kernel().payload_transport_broker().open_batch_payload_communicator_source(
      requester->kernel().worker_directory_cache(),
      absl::Now(),
      absl::Seconds(2),
      kRequesterDaemonId,
      transport,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-remote-routing");
  REQUIRE(source_or.ok());
  REQUIRE(source_or->remote);
  REQUIRE(source_or->source != nullptr);

  auto routing_context = requester->kernel().engine().get_shared_comm_manager()->routing_context();
  CHECK(routing_context == nullptr);

  std::string received(payload.size(), '\0');
  auto read_or = source_or->source->read_at(/*offset=*/0, received.data(), received.size());
  REQUIRE(read_or.ok());
  CHECK(*read_or == received.size());
  CHECK(received == payload);
}

TEST_CASE("Local payload_ref resolves to reusable body capability", "[daemon][batch][payload_ref][reuse]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.batch_transport_protocol_version = 0;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("payload-ref-reuse:blk-6");
  const std::string payload = "payload-ref-reuse-body";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-reuse-body");
  auto* put_item = put_req.add_items();
  put_item->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  const auto entry = harness->kernel().byte_artifact_body_store().get(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE(entry.has_value());
  REQUIRE_FALSE(entry->backing_record.retained_body_handle.empty());
  REQUIRE(entry->authority_record.visible);
  REQUIRE(entry->authority_record.artifact_id == artifact_id);
  REQUIRE(entry->authority_record.retained_backing_identity.has_value());

  auto payload_ref_or = harness->kernel().payload_transport_broker().issue_payload_ref(
      artifact_id,
      entry->backing_record.retained_body_handle,
      entry->descriptor,
      entry->authority_record.retained_backing_identity,
      entry->backing_record.instance_generation,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
      "op-reuse-body");
  REQUIRE(payload_ref_or.ok());

  auto capability_or = harness->kernel().payload_transport_broker().resolve_payload_ref_capability(
      harness->kernel().worker_directory_cache(),
      *payload_ref_or,
      artifact_id,
      absl::Now(),
      absl::Seconds(1),
      kDaemonId,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
      "op-reuse-body");
  REQUIRE(capability_or.ok());
  REQUIRE(capability_or->body_capability.has_value());
  REQUIRE(capability_or->body_capability->local);
  REQUIRE(capability_or->body_capability->mode == tensorcast::daemon::BodyCapabilityResolutionMode::kLocalBodyHandle);
  REQUIRE_FALSE(capability_or->body_capability->body_handle.empty());
  REQUIRE(capability_or->body_capability->descriptor.payload_digest_hex == entry->descriptor.payload_digest_hex);
  REQUIRE(capability_or->serving_capability.capability_id == *payload_ref_or);
  REQUIRE(capability_or->serving_capability.local);
  REQUIRE(capability_or->serving_capability.mode == tensorcast::daemon::BodyCapabilityResolutionMode::kLocalBodyHandle);
  REQUIRE(capability_or->serving_capability.subject_kind == tensorcast::daemon::ServingCapabilitySubjectKind::kBacking);
  REQUIRE(
      capability_or->serving_capability.lifecycle_owner_ref.owner_kind ==
      tensorcast::daemon::LifecycleOwnerKind::kPayloadRefToken);
  REQUIRE(capability_or->serving_capability.backing_identity.has_value());
  REQUIRE(capability_or->serving_capability.backing_identity == entry->authority_record.retained_backing_identity);
  REQUIRE(capability_or->serving_capability.backing_instance_generation == entry->backing_record.instance_generation);
  REQUIRE(capability_or->backing_identity == entry->authority_record.retained_backing_identity);

  tensorcast::daemon::BodyBackingManager manager(*engine);
  auto reused_or = manager.try_reuse_body(
      tensorcast::daemon::BodyBackingManager::ReuseRequest{
          .artifact_id = artifact_id,
          .invariant = put_req.items(0).invariant(),
          .descriptor = capability_or->body_capability->descriptor,
          .body_handle = capability_or->body_capability->body_handle,
          .operation_id = "op-reuse-body",
          .access_class = tensorcast::daemon::BodyAccessClass::kHomeDefault,
      });
  REQUIRE(reused_or.ok());
  REQUIRE(reused_or->has_value());
  REQUIRE(
      (*reused_or)->body_handle.replica_handle().key() ==
      entry->backing_record.retained_body_handle.replica_handle().key());
  REQUIRE((*reused_or)->descriptor.physical_artifact_id == entry->descriptor.physical_artifact_id);
  REQUIRE((*reused_or)->backing_identity.physical_artifact_id == entry->descriptor.physical_artifact_id);
  REQUIRE((*reused_or)->backing_identity.replica_key == entry->backing_record.identity.replica_key);
}

TEST_CASE(
    "Local payload_ref derives live-backing generation from the body handle when the caller omits it",
    "[daemon][batch][payload_ref][generation]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.batch_transport_protocol_version = 0;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("derived-generation:blk-7");
  const std::string payload = "payload-ref-derived-generation";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-derived-generation");
  auto* put_item = put_req.add_items();
  put_item->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  const auto entry = harness->kernel().byte_artifact_body_store().get(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE(entry.has_value());
  REQUIRE(entry->authority_record.retained_backing_identity.has_value());
  REQUIRE(entry->backing_record.retained_body_handle.binding_generation() != 0);

  auto payload_ref_or = harness->kernel().payload_transport_broker().issue_payload_ref(
      artifact_id,
      entry->backing_record.retained_body_handle,
      entry->descriptor,
      entry->authority_record.retained_backing_identity,
      /*backing_instance_generation=*/0,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
      "op-derived-generation");
  REQUIRE(payload_ref_or.ok());

  auto capability_or = harness->kernel().payload_transport_broker().resolve_payload_ref_capability(
      harness->kernel().worker_directory_cache(),
      *payload_ref_or,
      artifact_id,
      absl::Now(),
      absl::Seconds(1),
      kDaemonId,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
      "op-derived-generation");
  REQUIRE(capability_or.ok());
  REQUIRE(capability_or->serving_capability.backing_instance_generation != 0);
  REQUIRE(
      capability_or->serving_capability.backing_instance_generation ==
      entry->backing_record.retained_body_handle.binding_generation());
  REQUIRE(capability_or->body_capability.has_value());
  REQUIRE(capability_or->body_capability->body_handle.binding_generation() != 0);
}

TEST_CASE(
    "GET payload_ref issued from a live body resolves to copied-payload source capability",
    "[daemon][batch][payload_ref][get_snapshot]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.batch_transport_protocol_version = 0;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("get-snapshot:blk-8");
  const std::string payload = "payload-ref-get-snapshot";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-get-snapshot");
  auto* put_item = put_req.add_items();
  put_item->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  const auto entry = harness->kernel().byte_artifact_body_store().get(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE(entry.has_value());

  auto payload_ref_or = harness->kernel().payload_transport_broker().issue_payload_ref(
      artifact_id,
      entry->backing_record.retained_body_handle,
      entry->descriptor,
      entry->authority_record.retained_backing_identity,
      entry->backing_record.instance_generation,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-get-snapshot");
  REQUIRE(payload_ref_or.ok());

  auto capability_or = harness->kernel().payload_transport_broker().resolve_payload_ref_capability(
      harness->kernel().worker_directory_cache(),
      *payload_ref_or,
      artifact_id,
      absl::Now(),
      absl::Seconds(1),
      kDaemonId,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-get-snapshot");
  REQUIRE(capability_or.ok());
  REQUIRE_FALSE(capability_or->body_capability.has_value());
  REQUIRE(capability_or->inline_payload);
  REQUIRE(*capability_or->inline_payload == payload);
  REQUIRE(
      capability_or->serving_capability.subject_kind ==
      tensorcast::daemon::ServingCapabilitySubjectKind::kCopiedPayload);
}

TEST_CASE("BodyHandle export capability reflects runtime export support", "[daemon][batch][body_handle][export]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.batch_transport_protocol_version = 0;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("body-export-capability:blk-0");
  const std::string payload = "body-export-capability-payload";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-body-export-capability");
  auto* put_item = put_req.add_items();
  put_item->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  const auto entry = harness->kernel().byte_artifact_body_store().get(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE(entry.has_value());
  REQUIRE_FALSE(entry->backing_record.retained_body_handle.empty());

  auto capability_or = entry->backing_record.retained_body_handle.inspect_export_capability();
  REQUIRE(capability_or.ok());
  REQUIRE(capability_or->memory_location == tensorcast::common::memory::MemoryLocation::CPU);
  REQUIRE(capability_or->local_loader_available);
  REQUIRE(capability_or->remote_source_eligible);
  REQUIRE(capability_or->supports_segmented_export);
}

TEST_CASE("BodyHandle acquire_export_view reuses live export lease", "[daemon][batch][body_handle][export]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.batch_transport_protocol_version = 0;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("body-export-view:blk-0");
  const std::string payload = "body-export-view-payload";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-body-export-view");
  auto* put_item = put_req.add_items();
  put_item->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  const auto entry = harness->kernel().byte_artifact_body_store().get(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE(entry.has_value());
  REQUIRE_FALSE(entry->backing_record.retained_body_handle.empty());
  REQUIRE(entry->authority_record.retained_backing_identity.has_value());

  const tensorcast::daemon::BodyExportRequest request{
      .preferred_location = tensorcast::common::memory::MemoryLocation::CPU,
      .require_remote_source = true,
      .allow_segmented_export = true,
  };
  auto view1_or = entry->backing_record.retained_body_handle.acquire_export_view(request);
  REQUIRE(view1_or.ok());
  auto view2_or = entry->backing_record.retained_body_handle.acquire_export_view(request);
  REQUIRE(view2_or.ok());

  const auto& view1 = *view1_or;
  const auto& view2 = *view2_or;
  REQUIRE(view1.backing_identity == *entry->authority_record.retained_backing_identity);
  REQUIRE(view1.binding_generation == entry->backing_record.retained_body_handle.binding_generation());
  REQUIRE(view1.memory_location == tensorcast::common::memory::MemoryLocation::CPU);
  REQUIRE(view1.communicator_export.has_value());
  REQUIRE(view1.keepalive != nullptr);
  REQUIRE(view1.communicator_export->buffer_sizes.size() == 1);
  REQUIRE(view1.communicator_export->buffer_sizes.front() == payload.size());
  REQUIRE(view1.communicator_export->remote_memory_keys.size() == 1);
  REQUIRE(view2.communicator_export.has_value());
  REQUIRE(view2.keepalive != nullptr);
  REQUIRE_FALSE(view1.keepalive.owner_before(view2.keepalive));
  REQUIRE_FALSE(view2.keepalive.owner_before(view1.keepalive));
  REQUIRE(view1.communicator_export->remote_memory_keys == view2.communicator_export->remote_memory_keys);

  auto local_only_view_or = entry->backing_record.retained_body_handle.acquire_export_view(
      tensorcast::daemon::BodyExportRequest{
          .preferred_location = tensorcast::common::memory::MemoryLocation::CPU,
          .require_remote_source = false,
          .allow_segmented_export = true,
      });
  REQUIRE(local_only_view_or.ok());
  REQUIRE_FALSE(local_only_view_or->communicator_export.has_value());
  REQUIRE(local_only_view_or->keepalive != nullptr);

  auto rejected_or = entry->backing_record.retained_body_handle.acquire_export_view(
      tensorcast::daemon::BodyExportRequest{
          .preferred_location = tensorcast::common::memory::MemoryLocation::CPU,
          .require_remote_source = true,
          .allow_segmented_export = false,
      });
  REQUIRE_FALSE(rejected_or.ok());
  REQUIRE(rejected_or.status().code() == absl::StatusCode::kFailedPrecondition);
}

TEST_CASE("ByteArtifactBodyStore retains publish prereg entries until TTL expiry", "[daemon][batch][publish_prereg]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.batch_transport_protocol_version = 0;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("publish-prereg:ttl");
  const std::string payload = "publish-prereg-ttl-payload";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-publish-prereg-ttl");
  auto* put_item = put_req.add_items();
  put_item->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  auto entry = harness->kernel().byte_artifact_body_store().get(
      artifact_id, shard_id, /*lease_generation=*/1, /*routing_epoch=*/1, absl::Now());
  REQUIRE(entry.has_value());

  const tensorcast::daemon::BodyExportRequest request{
      .preferred_location = tensorcast::common::memory::MemoryLocation::CPU,
      .require_remote_source = true,
      .allow_segmented_export = true,
  };
  auto export_view_or = entry->backing_record.retained_body_handle.acquire_export_view(request);
  REQUIRE(export_view_or.ok());
  REQUIRE(export_view_or->keepalive != nullptr);

  auto& body_store = harness->kernel().byte_artifact_body_store();
  REQUIRE(body_store.retain_publish_preregistered_export(
      entry->backing_record.identity,
      entry->backing_record.instance_generation,
      tensorcast::common::memory::MemoryLocation::CPU,
      export_view_or->keepalive,
      entry->descriptor.size_bytes,
      absl::Now(),
      absl::Milliseconds(50),
      /*max_live_entries=*/8,
      /*max_live_bytes=*/1ULL << 20));

  auto prereg = body_store.inspect_publish_preregistered_export(entry->backing_record.identity);
  REQUIRE(prereg.has_value());
  REQUIRE(prereg->instance_generation == entry->backing_record.instance_generation);
  REQUIRE(prereg->size_bytes == entry->descriptor.size_bytes);

  absl::SleepFor(absl::Milliseconds(80));
  body_store.run_maintenance_once();
  REQUIRE_FALSE(body_store.inspect_publish_preregistered_export(entry->backing_record.identity).has_value());
}

TEST_CASE(
    "ByteArtifactBodyStore clears publish prereg entries when backing is invalidated",
    "[daemon][batch][publish_prereg]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.batch_transport_protocol_version = 0;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("publish-prereg:invalidate");
  const std::string payload = "publish-prereg-invalidate-payload";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-publish-prereg-invalidate");
  auto* put_item = put_req.add_items();
  put_item->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  auto entry = harness->kernel().byte_artifact_body_store().get(
      artifact_id, shard_id, /*lease_generation=*/1, /*routing_epoch=*/1, absl::Now());
  REQUIRE(entry.has_value());

  const tensorcast::daemon::BodyExportRequest request{
      .preferred_location = tensorcast::common::memory::MemoryLocation::CPU,
      .require_remote_source = true,
      .allow_segmented_export = true,
  };
  auto export_view_or = entry->backing_record.retained_body_handle.acquire_export_view(request);
  REQUIRE(export_view_or.ok());
  REQUIRE(export_view_or->keepalive != nullptr);

  auto& body_store = harness->kernel().byte_artifact_body_store();
  REQUIRE(body_store.retain_publish_preregistered_export(
      entry->backing_record.identity,
      entry->backing_record.instance_generation,
      tensorcast::common::memory::MemoryLocation::CPU,
      export_view_or->keepalive,
      entry->descriptor.size_bytes,
      absl::Now(),
      absl::Seconds(5),
      /*max_live_entries=*/8,
      /*max_live_bytes=*/1ULL << 20));

  REQUIRE(body_store.inspect_publish_preregistered_export(entry->backing_record.identity).has_value());
  body_store.invalidate_artifact_visibility(artifact_id, absl::Now(), "test_publish_prereg_invalidate");
  REQUIRE_FALSE(body_store.inspect_publish_preregistered_export(entry->backing_record.identity).has_value());
}

TEST_CASE("GetServerConfig advertises segmented communicator export policy", "[daemon][batch][transport_config]") {
  auto enabled_engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto enabled_harness = make_harness(enabled_engine, make_daemon_options());

  tensorcast::daemon::v2::GetServerConfigRequest enabled_req;
  tensorcast::daemon::v2::GetServerConfigResponse enabled_resp;
  grpc::ServerContext enabled_ctx;
  REQUIRE(enabled_harness->service().GetServerConfig(&enabled_ctx, &enabled_req, &enabled_resp).ok());
  REQUIRE(enabled_resp.batch_payload_communicator_source_enabled());
  REQUIRE(enabled_resp.batch_payload_host_memory_export_enabled());
  REQUIRE(enabled_resp.batch_payload_segmented_communicator_export_enabled());

  auto disabled_engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto disabled_options = make_daemon_options();
  disabled_options.byte_artifact_routing.payload_transport.segmented_communicator_export_enabled = false;
  auto disabled_harness = make_harness(disabled_engine, disabled_options);

  tensorcast::daemon::v2::GetServerConfigRequest disabled_req;
  tensorcast::daemon::v2::GetServerConfigResponse disabled_resp;
  grpc::ServerContext disabled_ctx;
  REQUIRE(disabled_harness->service().GetServerConfig(&disabled_ctx, &disabled_req, &disabled_resp).ok());
  REQUIRE(disabled_resp.batch_payload_communicator_source_enabled());
  REQUIRE(disabled_resp.batch_payload_host_memory_export_enabled());
  REQUIRE_FALSE(disabled_resp.batch_payload_segmented_communicator_export_enabled());
}

TEST_CASE(
    "HomeBatchGet payload_ref capability is bounded by authority expiry",
    "[daemon][batch][payload_ref][expiry]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.batch_transport_protocol_version = 0;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("payload-ref-expiry:blk-6");
  const std::string payload = "payload-ref-expiry-body";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_ttl_ms(50);
  put_req.set_operation_id("op-payload-ref-expiry");
  auto* put_item = put_req.add_items();
  put_item->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  const auto entry = harness->kernel().byte_artifact_body_store().get(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE(entry.has_value());
  REQUIRE(entry->authority_record.retained_backing_identity.has_value());

  HomeBatchGetRequest get_req;
  get_req.mutable_fence()->CopyFrom(put_req.fence());
  get_req.add_artifact_ids(artifact_id);
  get_req.set_operation_id("op-payload-ref-expiry");

  HomeBatchGetResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.HomeBatchGet(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.items_size() == 1);
  REQUIRE(get_resp.items(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  auto metadata_or = [&]() -> absl::StatusOr<tensorcast::daemon::PayloadTransportBroker::BatchRefMetadata> {
    if (!get_resp.items(0).payload_ref().empty()) {
      auto payload_ref_or = harness->kernel().payload_transport_broker().inspect_payload_ref(
          get_resp.items(0).payload_ref(), absl::Now(), /*require_not_expired=*/true);
      if (!payload_ref_or.ok()) {
        return payload_ref_or.status();
      }
      tensorcast::daemon::PayloadTransportBroker::BatchRefMetadata metadata;
      metadata.expires_at = payload_ref_or->expires_at;
      return metadata;
    }
    REQUIRE(get_resp.items(0).has_batch_payload_slice());
    REQUIRE(get_resp.batch_transports_size() == 1);
    const auto& transport = get_resp.batch_transports(0);
    REQUIRE(transport.transport_id() == get_resp.items(0).batch_payload_slice().transport_id());
    REQUIRE(transport.has_grpc_chunk_ref());
    return harness->kernel().payload_transport_broker().inspect_batch_payload_ref(
        transport.grpc_chunk_ref().batch_payload_ref(), absl::Now(), /*require_not_expired=*/true);
  }();
  REQUIRE(metadata_or.ok());
  CHECK(metadata_or->expires_at <= entry->expires_at);
}

TEST_CASE("payload_ref front-door context preserves raw credential evidence", "[daemon][batch][payload_ref][context]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());

  const std::string artifact_id = make_test_byte_artifact_id("frontdoor-context:blk-0");
  const std::string payload = "payload-ref-context";

  auto payload_ref_or = harness->kernel().payload_transport_broker().issue_payload_ref(
      artifact_id, payload, tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET, "op-front-door-context");
  REQUIRE(payload_ref_or.ok());

  auto context_or = harness->kernel().payload_transport_broker().inspect_payload_ref_context(
      *payload_ref_or,
      artifact_id,
      absl::Now(),
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-front-door-context");
  REQUIRE(context_or.ok());
  CHECK(context_or->metadata.artifact_id == artifact_id);
  CHECK(context_or->front_door_context.parsed_credential.address.binding_key == context_or->metadata.payload_id);
  CHECK(
      context_or->front_door_context.parsed_credential.constraint_claims.digest_hex == context_or->metadata.digest_hex);
  REQUIRE(context_or->front_door_context.forwardable_evidence.has_value());
  CHECK(
      context_or->front_door_context.forwardable_evidence->evidence_kind ==
      tensorcast::daemon::CredentialEvidenceKind::kRawCredential);
  CHECK(
      context_or->front_door_context.forwardable_evidence->raw_credential_bytes ==
      std::optional<std::string>(*payload_ref_or));
  CHECK(context_or->front_door_context.local_observations.empty());
}

TEST_CASE(
    "payload_ref issuer routed request rejects missing evidence, projected evidence, and unsanitized local observations",
    "[daemon][batch][payload_ref][route_builder]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());

  const std::string artifact_id = make_test_byte_artifact_id("issuer-route:blk-4");
  const std::string payload = "payload-ref-route-builder";

  auto payload_ref_or = harness->kernel().payload_transport_broker().issue_payload_ref(
      artifact_id, payload, tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET, "op-route-builder");
  REQUIRE(payload_ref_or.ok());

  auto context_or = harness->kernel().payload_transport_broker().inspect_payload_ref_context(
      *payload_ref_or, artifact_id, absl::Now(), tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET, "op-route-builder");
  REQUIRE(context_or.ok());

  auto no_evidence_context = context_or->front_door_context;
  no_evidence_context.forwardable_evidence.reset();
  auto no_evidence_request_or = harness->kernel().payload_transport_broker().build_payload_ref_issuer_routed_request(
      context_or->metadata, no_evidence_context, "127.0.0.1:50051");
  REQUIRE_FALSE(no_evidence_request_or.ok());
  CHECK(no_evidence_request_or.status().message() == "payload_ref issuer route requires forwardable_evidence");

  auto projected_context = context_or->front_door_context;
  projected_context.forwardable_evidence = tensorcast::daemon::ForwardableCredentialEvidence{
      .evidence_kind = tensorcast::daemon::CredentialEvidenceKind::kIssuerVerifiableProjection,
      .canonical_projection =
          tensorcast::daemon::CanonicalCredentialProjection{
              .projection_kind = "payload_ref",
              .projection_version = "v1",
              .projection_bytes = "{}",
              .projection_digest = "digest",
              .issuer_binding = "issuer-binding",
          },
  };
  auto projected_request_or = harness->kernel().payload_transport_broker().build_payload_ref_issuer_routed_request(
      context_or->metadata, projected_context, "127.0.0.1:50051");
  REQUIRE_FALSE(projected_request_or.ok());
  CHECK(
      projected_request_or.status().message() ==
      "payload_ref issuer route currently supports only raw_credential issuer evidence; canonical projection is not "
      "supported");

  auto unsanitized_context = context_or->front_door_context;
  unsanitized_context.local_observations.observations.push_back(
      tensorcast::daemon::LocalObservation{.observation_kind = "peer_pid", .observation_payload = "1234"});
  auto unsanitized_request_or = harness->kernel().payload_transport_broker().build_payload_ref_issuer_routed_request(
      context_or->metadata, unsanitized_context, "127.0.0.1:50051");
  REQUIRE_FALSE(unsanitized_request_or.ok());
  CHECK(unsanitized_request_or.status().message() == "local observation is not routable: peer_pid");
}

TEST_CASE(
    "payload_ref issuer routed request translates local observations into explicit forwarded claims",
    "[daemon][batch][payload_ref][route_builder]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());

  const std::string artifact_id = make_test_byte_artifact_id("translate-evidence:blk-5");
  const std::string payload = "payload-ref-route-translation";

  auto payload_ref_or = harness->kernel().payload_transport_broker().issue_payload_ref(
      artifact_id, payload, tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET, "op-route-translation");
  REQUIRE(payload_ref_or.ok());

  auto context_or = harness->kernel().payload_transport_broker().inspect_payload_ref_context(
      *payload_ref_or,
      artifact_id,
      absl::Now(),
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-route-translation");
  REQUIRE(context_or.ok());

  auto translated_context = context_or->front_door_context;
  translated_context.local_observations.observations.push_back(
      tensorcast::daemon::LocalObservation{.observation_kind = "peer_pid", .observation_payload = "1234"});
  const tensorcast::daemon::LocalObservationRoutingRule translation_rule{
      .observation_kind = "peer_pid",
      .action = tensorcast::daemon::LocalObservationRoutingAction::kTranslateToForwardedClaim,
      .forwarded_claim_kind = "peer_pid_claim",
  };
  const auto translation_rules =
      absl::Span<const tensorcast::daemon::LocalObservationRoutingRule>(&translation_rule, 1);
  auto routed_request_or = harness->kernel().payload_transport_broker().build_payload_ref_issuer_routed_request(
      context_or->metadata, translated_context, "127.0.0.1:50051", translation_rules);
  REQUIRE(routed_request_or.ok());
  CHECK(
      routed_request_or->hop_auth_context.auth_class ==
      tensorcast::daemon::DaemonHopAuthClass::kDeploymentTrustedChannel);
  REQUIRE(routed_request_or->forwardable_evidence.has_value());
  REQUIRE(routed_request_or->portable_credential_envelope.has_value());
  CHECK(
      routed_request_or->portable_credential_envelope->payload_kind ==
      tensorcast::daemon::DelegationPayloadKind::kPortableCredential);
  REQUIRE(routed_request_or->forwardable_evidence_envelope.has_value());
  CHECK(
      routed_request_or->forwardable_evidence_envelope->delegation_class ==
      tensorcast::daemon::DelegationClass::kOwnerScopedSensitive);
  REQUIRE(routed_request_or->forwarded_claims.size() == 1);
  REQUIRE(routed_request_or->forwarded_claims_envelope.has_value());
  CHECK(
      routed_request_or->forwarded_claims_envelope->payload_kind ==
      tensorcast::daemon::DelegationPayloadKind::kForwardedClaim);
  CHECK(routed_request_or->forwarded_claims.front().claim_kind == "peer_pid_claim");
  CHECK(
      routed_request_or->forwarded_claims.front().provenance ==
      tensorcast::daemon::ForwardedClaimProvenance::kIngressLocal);
  CHECK(routed_request_or->forwarded_claims.front().audience_authority_ref.authority_id == kDaemonId);
  CHECK(
      routed_request_or->request_metadata.root_request_id ==
      absl::StrCat("payload-ref:", context_or->metadata.payload_id));
}

TEST_CASE("RouteAuthorityStage rejects authority mismatch at the receiving daemon", "[daemon][batch][route]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  RouteAuthorityStageRequest req;
  auto* routed_request = req.mutable_routed_request();
  auto* authority_ref = routed_request->mutable_authority_ref();
  authority_ref->set_authority_kind(tensorcast::daemon::v2::ROUTED_AUTHORITY_KIND_ISSUER_DAEMON);
  authority_ref->set_authority_id("issuer-daemon-a");
  routed_request->set_path_family("immediate_lowering");
  routed_request->set_stage_ref("issuer_validate");
  auto* portable_credential = routed_request->mutable_portable_credential();
  auto* address = portable_credential->mutable_address();
  address->set_route_principal_kind("issuer_daemon");
  address->set_route_principal_id("issuer-daemon-a");
  address->set_family("serve");
  address->set_binding_space("payload");
  address->set_binding_key_kind("payload_id");
  address->set_binding_key("payload-ref:test");
  address->set_subject_generation(1);
  portable_credential->set_front_door_kind("payload_ref");
  portable_credential->mutable_credential_expires_at()->set_seconds(
      absl::ToUnixSeconds(absl::Now() + absl::Minutes(1)));
  portable_credential->set_binding_mode("address_derived");
  portable_credential->mutable_portable_constraint_claims()->set_artifact_id("artifact-route");
  routed_request->mutable_request_metadata()->set_root_request_id("root-req-1");

  RouteAuthorityStageResponse resp;
  grpc::ServerContext ctx;
  REQUIRE(svc.RouteAuthorityStage(&ctx, &req, &resp).ok());
  CHECK(resp.status() == BatchItemStatus::BATCH_ITEM_STATUS_FAILED_PRECONDITION);
  CHECK(resp.message() == "requested authority does not match the receiving daemon");
  CHECK_FALSE(resp.has_owner_stage_reply());
}

TEST_CASE("RouteAuthorityStage rejects sender-forged hop auth elevation", "[daemon][batch][route][security]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  RouteAuthorityStageRequest req;
  auto* routed_request = req.mutable_routed_request();
  auto* authority_ref = routed_request->mutable_authority_ref();
  authority_ref->set_authority_kind(tensorcast::daemon::v2::ROUTED_AUTHORITY_KIND_ISSUER_DAEMON);
  authority_ref->set_authority_id("issuer-daemon-a");
  routed_request->set_path_family("immediate_lowering");
  routed_request->set_stage_ref("issuer_validate");
  auto* portable_credential = routed_request->mutable_portable_credential();
  auto* address = portable_credential->mutable_address();
  address->set_route_principal_kind("issuer_daemon");
  address->set_route_principal_id("issuer-daemon-a");
  address->set_family("serve");
  address->set_binding_space("payload");
  address->set_binding_key_kind("payload_id");
  address->set_binding_key("payload-ref:test");
  address->set_subject_generation(1);
  portable_credential->set_front_door_kind("payload_ref");
  portable_credential->mutable_credential_expires_at()->set_seconds(
      absl::ToUnixSeconds(absl::Now() + absl::Minutes(1)));
  portable_credential->set_binding_mode("address_derived");
  portable_credential->mutable_portable_constraint_claims()->set_artifact_id("artifact-route");
  routed_request->mutable_request_metadata()->set_root_request_id("root-req-2");
  routed_request->mutable_hop_auth_context()->set_auth_class(
      tensorcast::daemon::v2::ROUTED_DAEMON_HOP_AUTH_CLASS_DAEMON_MUTUAL_AUTH);
  routed_request->mutable_hop_auth_context()->set_authenticated_peer_daemon_id("issuer-daemon-a");

  RouteAuthorityStageResponse resp;
  grpc::ServerContext ctx;
  REQUIRE(svc.RouteAuthorityStage(&ctx, &req, &resp).ok());
  CHECK(resp.status() == BatchItemStatus::BATCH_ITEM_STATUS_FAILED_PRECONDITION);
  CHECK(resp.message() == "sender-reported hop auth exceeds transport-derived peer auth");
}

TEST_CASE("RouteAuthorityStage rejects undeclared path_family or stage_ref", "[daemon][batch][route][dispatch]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("route-declared:blk-9");
  const std::string payload = "declared-route-check";
  auto payload_ref_or = harness->kernel().payload_transport_broker().issue_payload_ref(
      artifact_id, payload, tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET, "op-declared-route");
  REQUIRE(payload_ref_or.ok());

  auto context_or = harness->kernel().payload_transport_broker().inspect_payload_ref_context(
      *payload_ref_or,
      artifact_id,
      absl::Now(),
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-declared-route");
  REQUIRE(context_or.ok());

  auto routed_request_or = harness->kernel().payload_transport_broker().build_payload_ref_issuer_routed_request(
      context_or->metadata, context_or->front_door_context, "127.0.0.1:50051");
  REQUIRE(routed_request_or.ok());

  SECTION("undeclared path_family") {
    routed_request_or->path_family = "unknown_family";
    REQUIRE(routed_request_or->portable_credential_envelope.has_value());
    routed_request_or->portable_credential_envelope->bound_path_family = routed_request_or->path_family;
  }

  SECTION("undeclared stage_ref") {
    routed_request_or->stage_ref = "unknown_stage";
  }

  routed_request_or->forwardable_evidence.reset();
  routed_request_or->forwardable_evidence_envelope.reset();

  RouteAuthorityStageRequest req;
  tensorcast::daemon::routed_authority_wire::populate_proto_routed_authority_request(
      *routed_request_or, req.mutable_routed_request());

  RouteAuthorityStageResponse resp;
  grpc::ServerContext ctx;
  REQUIRE(svc.RouteAuthorityStage(&ctx, &req, &resp).ok());
  CHECK(resp.status() == BatchItemStatus::BATCH_ITEM_STATUS_FAILED_PRECONDITION);
  CHECK(
      resp.message() ==
      absl::StrCat(
          "undeclared routed authority path/stage: ",
          req.routed_request().path_family(),
          "/",
          req.routed_request().stage_ref()));
  CHECK_FALSE(resp.has_owner_stage_reply());
}

TEST_CASE(
    "RouteAuthorityStage returns ready_for_lowering for local payload_ref issuer validation",
    "[daemon][batch][route][issuer]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("issuer-route:blk-1");
  const std::string payload = "issuer-routed-payload";
  auto payload_ref_or = harness->kernel().payload_transport_broker().issue_payload_ref(
      artifact_id, payload, tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET, "op-issuer-route");
  REQUIRE(payload_ref_or.ok());

  auto context_or = harness->kernel().payload_transport_broker().inspect_payload_ref_context(
      *payload_ref_or, artifact_id, absl::Now(), tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET, "op-issuer-route");
  REQUIRE(context_or.ok());
  REQUIRE(context_or->front_door_context.forwardable_evidence.has_value());

  auto routed_request_or = harness->kernel().payload_transport_broker().build_payload_ref_issuer_routed_request(
      context_or->metadata, context_or->front_door_context, "127.0.0.1:50051");
  REQUIRE(routed_request_or.ok());

  RouteAuthorityStageRequest req;
  tensorcast::daemon::routed_authority_wire::populate_proto_routed_authority_request(
      *routed_request_or, req.mutable_routed_request());

  RouteAuthorityStageResponse resp;
  grpc::ServerContext ctx;
  REQUIRE(svc.RouteAuthorityStage(&ctx, &req, &resp).ok());
  CHECK(resp.status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(resp.has_owner_stage_reply());
  CHECK(
      resp.owner_stage_reply().reply_kind() ==
      tensorcast::daemon::v2::ROUTED_OWNER_STAGE_REPLY_KIND_READY_FOR_LOWERING);
  CHECK(resp.owner_stage_reply().answered_by().authority_id() == kDaemonId);
  CHECK(resp.owner_stage_reply().path_family() == "immediate_lowering");
  CHECK(resp.owner_stage_reply().stage_ref() == "issuer_validate");
  CHECK_FALSE(resp.owner_stage_reply().resolved_source_capability().empty());
}

TEST_CASE(
    "Backing loss makes routed exists miss but preserves claim truth for conflict decisions",
    "[daemon][batch][visibility][claim_truth]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("claim-truth:blk-9");
  const std::string payload = "claim-truth-payload";
  const std::string conflicting_payload = "claim-truth-conflict";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  auto* item = put_req.add_items();
  item->set_artifact_id(artifact_id);
  item->set_inline_payload(payload);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  REQUIRE(engine->clear_mem() == 0);

  HomeBatchExistsRequest exists_req;
  exists_req.mutable_fence()->CopyFrom(put_req.fence());
  exists_req.add_artifact_ids(artifact_id);
  HomeBatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  REQUIRE(svc.HomeBatchExists(&exists_ctx, &exists_req, &exists_resp).ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_MISS);

  const auto authority_after_loss = harness->kernel().byte_artifact_body_store().inspect_authority(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE(authority_after_loss.has_value());
  REQUIRE(
      authority_after_loss->authority_record.claim_state == tensorcast::daemon::AuthorityClaimState::kClaimedInvisible);
  REQUIRE(authority_after_loss->authority_record.retained_backing_identity.has_value());
  const auto backing_before_restore = harness->kernel().byte_artifact_body_store().inspect_backing(
      *authority_after_loss->authority_record.retained_backing_identity);
  REQUIRE(backing_before_restore.has_value());
  REQUIRE(backing_before_restore->lifecycle_state == tensorcast::daemon::BackingLifecycleState::kInvalidated);

  HomeBatchPutIfAbsentRequest conflict_req;
  conflict_req.mutable_fence()->CopyFrom(put_req.fence());
  auto* conflict_item = conflict_req.add_items();
  conflict_item->set_artifact_id(artifact_id);
  conflict_item->set_inline_payload(conflicting_payload);
  set_invariant(conflict_item->mutable_invariant(), "layout_v1", conflicting_payload);

  HomeBatchPutIfAbsentResponse conflict_resp;
  grpc::ServerContext conflict_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&conflict_ctx, &conflict_req, &conflict_resp).ok());
  REQUIRE(conflict_resp.outcomes_size() == 1);
  REQUIRE(conflict_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_FAILED_PRECONDITION);

  HomeBatchPutIfAbsentRequest join_req;
  join_req.mutable_fence()->CopyFrom(put_req.fence());
  auto* join_item = join_req.add_items();
  join_item->set_artifact_id(artifact_id);
  join_item->set_inline_payload(payload);
  set_invariant(join_item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse join_resp;
  grpc::ServerContext join_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&join_ctx, &join_req, &join_resp).ok());
  REQUIRE(join_resp.outcomes_size() == 1);
  REQUIRE(join_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  HomeBatchExistsResponse exists_after_join_resp;
  grpc::ServerContext exists_after_join_ctx;
  REQUIRE(svc.HomeBatchExists(&exists_after_join_ctx, &exists_req, &exists_after_join_resp).ok());
  REQUIRE(exists_after_join_resp.outcomes_size() == 1);
  REQUIRE(exists_after_join_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  const auto restored_entry = harness->kernel().byte_artifact_body_store().get(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE(restored_entry.has_value());
  CHECK(restored_entry->backing_record.instance_generation == backing_before_restore->instance_generation + 1);
}

TEST_CASE(
    "Managed shared-disk policy path restores routed visibility after backing loss",
    "[daemon][batch][visibility][policy_backed_path]") {
  const auto storage_root = make_test_storage_root("byte_artifact_policy_backed_restore");
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.storage_path = storage_root;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  auto gs_client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* gs_client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(gs_client.get());
  harness->kernel().persistence_manager()->set_global_store_client(gs_client_ptr);

  const std::string artifact_id = make_test_byte_artifact_id("policy-restore:blk-12");
  const std::string payload = "policy-backed-restore-payload";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  auto* item = put_req.add_items();
  item->set_artifact_id(artifact_id);
  item->set_inline_payload(payload);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  tensorcast::daemon::v2::StartPersistenceRequest persist_req;
  persist_req.set_artifact_id(artifact_id);
  persist_req.mutable_policy()->CopyFrom(make_shared_disk_policy());
  tensorcast::daemon::v2::StartPersistenceResponse persist_resp;
  grpc::ServerContext persist_ctx;
  REQUIRE(svc.StartPersistence(&persist_ctx, &persist_req, &persist_resp).ok());
  REQUIRE_FALSE(persist_resp.task_id().empty());

  const auto persist_task = advance_persistence_to_terminal(*harness, persist_resp.task_id());
  REQUIRE(persist_task.state == tensorcast::daemon::v2::PERSISTENCE_STATE_SUCCESS);
  REQUIRE(persist_task.disk_location_registered);

  REQUIRE(engine->clear_mem() == 0);

  HomeBatchExistsRequest exists_req;
  exists_req.mutable_fence()->CopyFrom(put_req.fence());
  exists_req.add_artifact_ids(artifact_id);
  HomeBatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  REQUIRE(svc.HomeBatchExists(&exists_ctx, &exists_req, &exists_resp).ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  const auto policy_authority = harness->kernel().byte_artifact_body_store().inspect_authority(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE(policy_authority.has_value());
  REQUIRE(
      policy_authority->authority_record.visibility_kind ==
      tensorcast::daemon::AuthorityVisibilityKind::kPolicyBackedPath);
  REQUIRE(policy_authority->authority_record.policy_visibility_ref.has_value());
  REQUIRE(
      policy_authority->authority_record.policy_visibility_ref->path_kind ==
      tensorcast::daemon::PolicyVisibilityPathKind::kSharedDisk);
  REQUIRE(
      policy_authority->authority_record.policy_visibility_ref->verified_content_descriptor ==
      policy_authority->verified_content_descriptor);

  HomeBatchGetRequest get_req;
  get_req.mutable_fence()->CopyFrom(put_req.fence());
  get_req.add_artifact_ids(artifact_id);
  HomeBatchGetResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.HomeBatchGet(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.items_size() == 1);
  REQUIRE(get_resp.items(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.items(0).inline_payload() == payload);

  const auto restored_entry = harness->kernel().byte_artifact_body_store().get(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE(restored_entry.has_value());
  REQUIRE(
      restored_entry->authority_record.visibility_kind == tensorcast::daemon::AuthorityVisibilityKind::kReadyBacking);
  REQUIRE(restored_entry->backing_record.lifecycle_state == tensorcast::daemon::BackingLifecycleState::kActive);
}

TEST_CASE(
    "Older managed shared-disk proof remains actionable after a later failed persistence task",
    "[daemon][batch][visibility][policy_backed_path][source_truth]") {
  const auto storage_root = make_test_storage_root("byte_artifact_policy_backed_source_truth");
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.storage_path = storage_root;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  auto gs_client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* gs_client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(gs_client.get());
  harness->kernel().persistence_manager()->set_global_store_client(gs_client_ptr);

  const std::string artifact_id = make_test_byte_artifact_id("source-truth:blk-15");
  const std::string payload = "policy-backed-source-truth-payload";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  auto* item = put_req.add_items();
  item->set_artifact_id(artifact_id);
  item->set_inline_payload(payload);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  tensorcast::daemon::v2::StartPersistenceRequest first_req;
  first_req.set_artifact_id(artifact_id);
  first_req.mutable_policy()->CopyFrom(make_shared_disk_policy());
  tensorcast::daemon::v2::StartPersistenceResponse first_resp;
  grpc::ServerContext first_ctx;
  REQUIRE(svc.StartPersistence(&first_ctx, &first_req, &first_resp).ok());
  REQUIRE_FALSE(first_resp.task_id().empty());
  const auto first_task = advance_persistence_to_terminal(*harness, first_resp.task_id());
  REQUIRE(first_task.state == tensorcast::daemon::v2::PERSISTENCE_STATE_SUCCESS);

  harness->kernel().persistence_manager()->set_fail_shared_disk_for_test(true);
  tensorcast::daemon::v2::StartPersistenceRequest second_req;
  second_req.set_artifact_id(artifact_id);
  second_req.mutable_policy()->CopyFrom(make_shared_disk_policy());
  tensorcast::daemon::v2::StartPersistenceResponse second_resp;
  grpc::ServerContext second_ctx;
  REQUIRE(svc.StartPersistence(&second_ctx, &second_req, &second_resp).ok());
  REQUIRE_FALSE(second_resp.task_id().empty());
  const auto second_task = advance_persistence_to_terminal(*harness, second_resp.task_id());
  REQUIRE(second_task.state == tensorcast::daemon::v2::PERSISTENCE_STATE_FAILED);

  REQUIRE(engine->clear_mem() == 0);

  HomeBatchExistsRequest exists_req;
  exists_req.mutable_fence()->CopyFrom(put_req.fence());
  exists_req.add_artifact_ids(artifact_id);
  HomeBatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  REQUIRE(svc.HomeBatchExists(&exists_ctx, &exists_req, &exists_resp).ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  const auto authority = harness->kernel().byte_artifact_body_store().inspect_authority(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE(authority.has_value());
  REQUIRE(
      authority->authority_record.visibility_kind == tensorcast::daemon::AuthorityVisibilityKind::kPolicyBackedPath);
  REQUIRE(authority->authority_record.policy_visibility_ref.has_value());
  REQUIRE(authority->authority_record.policy_visibility_ref->control_ref == first_task.task_id);
  REQUIRE(authority->authority_record.policy_visibility_ref->control_ref != second_task.task_id);
}

TEST_CASE(
    "Policy-backed visibility is removed when managed shared-disk proof disappears",
    "[daemon][batch][visibility][policy_backed_path]") {
  const auto storage_root = make_test_storage_root("byte_artifact_policy_backed_missing");
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.storage_path = storage_root;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  auto gs_client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* gs_client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(gs_client.get());
  harness->kernel().persistence_manager()->set_global_store_client(gs_client_ptr);

  const std::string artifact_id = make_test_byte_artifact_id("policy-missing:blk-13");
  const std::string payload = "policy-backed-missing-payload";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  auto* item = put_req.add_items();
  item->set_artifact_id(artifact_id);
  item->set_inline_payload(payload);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  tensorcast::daemon::v2::StartPersistenceRequest persist_req;
  persist_req.set_artifact_id(artifact_id);
  persist_req.mutable_policy()->CopyFrom(make_shared_disk_policy());
  tensorcast::daemon::v2::StartPersistenceResponse persist_resp;
  grpc::ServerContext persist_ctx;
  REQUIRE(svc.StartPersistence(&persist_ctx, &persist_req, &persist_resp).ok());
  const auto persist_task = advance_persistence_to_terminal(*harness, persist_resp.task_id());
  REQUIRE(persist_task.state == tensorcast::daemon::v2::PERSISTENCE_STATE_SUCCESS);

  REQUIRE(engine->clear_mem() == 0);
  REQUIRE_FALSE(persist_task.disk_relative_path.empty());
  std::filesystem::remove_all(storage_root / persist_task.disk_relative_path);

  HomeBatchExistsRequest exists_req;
  exists_req.mutable_fence()->CopyFrom(put_req.fence());
  exists_req.add_artifact_ids(artifact_id);
  HomeBatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  REQUIRE(svc.HomeBatchExists(&exists_ctx, &exists_req, &exists_resp).ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_MISS);

  const auto authority = harness->kernel().byte_artifact_body_store().inspect_authority(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE(authority.has_value());
  REQUIRE(authority->authority_record.claim_state == tensorcast::daemon::AuthorityClaimState::kClaimedInvisible);
  REQUIRE_FALSE(authority->authority_record.policy_visibility_ref.has_value());
}

TEST_CASE(
    "Claim deletion is not resurrected by persisted managed shared-disk path",
    "[daemon][batch][visibility][policy_backed_path][claim_deleted]") {
  const auto storage_root = make_test_storage_root("byte_artifact_policy_backed_deleted");
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.storage_path = storage_root;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  auto gs_client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* gs_client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(gs_client.get());
  harness->kernel().persistence_manager()->set_global_store_client(gs_client_ptr);

  const std::string artifact_id = make_test_byte_artifact_id("policy-deleted:blk-14");
  const std::string payload = "policy-backed-deleted-payload";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_ttl_ms(100);
  auto* item = put_req.add_items();
  item->set_artifact_id(artifact_id);
  item->set_inline_payload(payload);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  tensorcast::daemon::v2::StartPersistenceRequest persist_req;
  persist_req.set_artifact_id(artifact_id);
  persist_req.mutable_policy()->CopyFrom(make_shared_disk_policy());
  tensorcast::daemon::v2::StartPersistenceResponse persist_resp;
  grpc::ServerContext persist_ctx;
  REQUIRE(svc.StartPersistence(&persist_ctx, &persist_req, &persist_resp).ok());
  const auto persist_task = advance_persistence_to_terminal(*harness, persist_resp.task_id());
  REQUIRE(persist_task.state == tensorcast::daemon::v2::PERSISTENCE_STATE_SUCCESS);

  absl::SleepFor(absl::Milliseconds(120));
  REQUIRE(engine->clear_mem() == 0);

  HomeBatchExistsRequest exists_req;
  exists_req.mutable_fence()->CopyFrom(put_req.fence());
  exists_req.add_artifact_ids(artifact_id);
  HomeBatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  REQUIRE(svc.HomeBatchExists(&exists_ctx, &exists_req, &exists_resp).ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_MISS);

  const auto authority = harness->kernel().byte_artifact_body_store().inspect_authority(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE_FALSE(authority.has_value());
}

TEST_CASE("TTL expiry deletes routed claim and allows fresh create", "[daemon][batch][ttl][recreate]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("ttl-recreate:blk-10");
  const std::string initial_payload = "ttl-recreate-initial";
  const std::string replacement_payload = "ttl-recreate-replacement";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_ttl_ms(1);
  auto* item = put_req.add_items();
  item->set_artifact_id(artifact_id);
  item->set_inline_payload(initial_payload);
  set_invariant(item->mutable_invariant(), "layout_v1", initial_payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  absl::SleepFor(absl::Milliseconds(5));

  HomeBatchExistsRequest exists_req;
  exists_req.mutable_fence()->CopyFrom(put_req.fence());
  exists_req.add_artifact_ids(artifact_id);
  HomeBatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  REQUIRE(svc.HomeBatchExists(&exists_ctx, &exists_req, &exists_resp).ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_MISS);

  HomeBatchPutIfAbsentRequest recreate_req;
  recreate_req.mutable_fence()->CopyFrom(put_req.fence());
  auto* recreate_item = recreate_req.add_items();
  recreate_item->set_artifact_id(artifact_id);
  recreate_item->set_inline_payload(replacement_payload);
  set_invariant(recreate_item->mutable_invariant(), "layout_v1", replacement_payload);

  HomeBatchPutIfAbsentResponse recreate_resp;
  grpc::ServerContext recreate_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&recreate_ctx, &recreate_req, &recreate_resp).ok());
  REQUIRE(recreate_resp.outcomes_size() == 1);
  REQUIRE(recreate_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
}

TEST_CASE(
    "Issued HomeBatchGet payload_ref survives immediate backing retirement pressure",
    "[daemon][batch][payload_ref][survives_retire]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.batch_transport_protocol_version = 0;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("survive-retire:blk-11");
  const std::string payload = "survive-retire-payload";
  const std::uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-survive-retire");
  auto* put_item = put_req.add_items();
  put_item->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  HomeBatchGetRequest get_req;
  get_req.mutable_fence()->CopyFrom(put_req.fence());
  get_req.add_artifact_ids(artifact_id);
  get_req.set_operation_id("op-survive-retire");

  HomeBatchGetResponse get_resp;
  grpc::ServerContext get_ctx;
  REQUIRE(svc.HomeBatchGet(&get_ctx, &get_req, &get_resp).ok());
  REQUIRE(get_resp.items_size() == 1);
  REQUIRE(get_resp.items(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  REQUIRE(engine->clear_mem() == 0);

  HomeBatchExistsRequest exists_req;
  exists_req.mutable_fence()->CopyFrom(put_req.fence());
  exists_req.add_artifact_ids(artifact_id);
  HomeBatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  REQUIRE(svc.HomeBatchExists(&exists_ctx, &exists_req, &exists_resp).ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_MISS);

  auto payload_or = [&]() -> absl::StatusOr<tensorcast::daemon::PayloadTransportBroker::ResolvedPayload> {
    if (!get_resp.items(0).payload_ref().empty()) {
      return harness->kernel().payload_transport_broker().resolve_local_payload_ref(
          get_resp.items(0).payload_ref(),
          artifact_id,
          absl::Now(),
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
          "op-survive-retire");
    }
    REQUIRE(get_resp.items(0).has_batch_payload_slice());
    REQUIRE(get_resp.batch_transports_size() == 1);
    const auto& transport = get_resp.batch_transports(0);
    REQUIRE(transport.transport_id() == get_resp.items(0).batch_payload_slice().transport_id());
    REQUIRE(transport.has_grpc_chunk_ref());
    auto resolved_or = harness->kernel().payload_transport_broker().fetch_batch_payload_ref(
        harness->kernel().worker_directory_cache(),
        absl::Now(),
        absl::Seconds(1),
        kDaemonId,
        transport.grpc_chunk_ref().batch_payload_ref(),
        tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
        "op-survive-retire");
    if (!resolved_or.ok()) {
      return resolved_or.status();
    }
    tensorcast::daemon::PayloadTransportBroker::ResolvedPayload payload;
    payload.metadata.expires_at = resolved_or->metadata.expires_at;
    payload.payload = resolved_or->payload != nullptr ? *resolved_or->payload : "";
    return payload;
  }();
  REQUIRE(payload_or.ok());
  REQUIRE(payload_or->payload == payload);
}

TEST_CASE("BodyBackingManager derives stable admission from shared policy flow", "[daemon][body_backing][policy]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  tensorcast::daemon::BodyBackingManager manager(*engine);

  const auto make_loader = [](const std::shared_ptr<const std::string>& payload) {
    return std::make_unique<tensorcast::store::InlineBufferLoader>(tensorcast::store::loading::InlineBufferSource{
        .data = std::shared_ptr<const void>(payload, static_cast<const void*>(payload->data())),
        .size_bytes = payload->size(),
    });
  };

  const std::string retained_artifact_id = make_test_byte_artifact_id("retained:blk-4");
  const auto retained_payload = std::make_shared<const std::string>("retained-home-payload");
  tensorcast::daemon::v2::PutIfAbsentInvariant retained_invariant;
  set_invariant(&retained_invariant, "layout_v1", *retained_payload);

  auto retained_or = manager.stage_body(
      tensorcast::daemon::BodyBackingManager::StageRequest{
          .artifact_id = retained_artifact_id,
          .invariant = retained_invariant,
          .loader = make_loader(retained_payload),
          .source_kind = tensorcast::store::loading::MaterializationSource::kLocalReplica,
          .operation_id = "op-retained",
          .access_class = tensorcast::daemon::BodyAccessClass::kHomeDefault,
          .route_role = tensorcast::daemon::BodyRouteRole::kHomeAuthority,
      });
  REQUIRE(retained_or.ok());
  CHECK(retained_or->verified_content_descriptor.content_identity.logical_size_bytes == retained_payload->size());
  CHECK(retained_or->backing_identity.physical_artifact_id == retained_or->descriptor.physical_artifact_id);
  CHECK(retained_or->backing_identity.replica_key.artifact_id == retained_or->descriptor.physical_artifact_id);
  CHECK_FALSE(retained_or->backing_identity.replica_key.view_id.has_value());
  CHECK(retained_or->backing_identity.replica_key.replica == 0);
  CHECK(retained_or->observation.stable_retention_state != tensorcast::daemon::BodyStableRetentionState::kNotRequested);
  REQUIRE(retained_or->body_handle.replica_handle().key().device.type == tensorcast::DeviceType::CPU);
  REQUIRE(retained_or->body_handle.retire().ok());

  const std::string transient_artifact_id = make_test_byte_artifact_id("transient:blk-4");
  const auto transient_payload = std::make_shared<const std::string>("transient-forward-payload");
  tensorcast::daemon::v2::PutIfAbsentInvariant transient_invariant;
  set_invariant(&transient_invariant, "layout_v1", *transient_payload);

  auto transient_or = manager.stage_body(
      tensorcast::daemon::BodyBackingManager::StageRequest{
          .artifact_id = transient_artifact_id,
          .invariant = transient_invariant,
          .loader = make_loader(transient_payload),
          .source_kind = tensorcast::store::loading::MaterializationSource::kLocalReplica,
          .operation_id = "op-transient",
          .access_class = tensorcast::daemon::BodyAccessClass::kTransientForward,
          .route_role = tensorcast::daemon::BodyRouteRole::kTransientForwarder,
      });
  REQUIRE(transient_or.ok());
  CHECK(
      transient_or->observation.stable_retention_state == tensorcast::daemon::BodyStableRetentionState::kNotRequested);
  REQUIRE(transient_or->body_handle.retire().ok());
}

TEST_CASE("BodyBackingManager fast CPU staging hashes during local byte ingress", "[daemon][body_backing][fast_cpu]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  tensorcast::daemon::BodyBackingManager manager(*engine);

  const std::string artifact_id = make_test_byte_artifact_id("fast-cpu:blk-4");
  const auto payload = std::make_shared<const std::string>("fast-cpu-body-payload");
  tensorcast::daemon::v2::PutIfAbsentInvariant invariant;
  set_invariant(&invariant, "layout_v1", *payload);

  auto staged_or = manager.stage_body_fast_cpu_verified(
      artifact_id,
      invariant,
      tensorcast::daemon::BodyBackingManager::LocalByteSpan{
          .owner = std::shared_ptr<const void>(payload, static_cast<const void*>(payload->data())),
          .data = reinterpret_cast<const std::uint8_t*>(payload->data()),
          .size_bytes = payload->size(),
      },
      tensorcast::store::loading::MaterializationSource::kP2P,
      "op-fast-cpu");
  REQUIRE(staged_or.ok());
  CHECK(staged_or->descriptor.payload_digest_alg == "sha256");
  CHECK(staged_or->descriptor.payload_digest_hex == invariant.payload_digest_hex());
  CHECK(staged_or->descriptor.size_bytes == payload->size());
  CHECK(
      staged_or->verification_record.verification_method ==
      tensorcast::store::runtime::ingestion::VerificationMethod::kSharedExecutorStreamDigest);
  auto read_back_or = staged_or->body_handle.read_all_bytes();
  REQUIRE(read_back_or.ok());
  CHECK(*read_back_or == *payload);
  REQUIRE(staged_or->body_handle.retire().ok());
}

class RecordingCompositeSource final : public tensorcast::store::loader::SeekableSource {
 public:
  explicit RecordingCompositeSource(std::string data) : data_(std::move(data)) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return data_.size();
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(cursor_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or.status();
    }
    cursor_ += *read_or;
    return *read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= data_.size() || bytes == 0) {
      return static_cast<size_t>(0);
    }
    const size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, data_.size() - offset));
    std::memcpy(dst, data_.data() + offset, to_copy);
    return to_copy;
  }

  [[nodiscard]] bool supports_direct_write_at() const override {
    return true;
  }

  [[nodiscard]] bool supports_batched_direct_write_at() const override {
    return true;
  }

  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const tensorcast::store::DirectWriteGrant& grant) override {
    if (src_offset > data_.size() || bytes > data_.size() - src_offset) {
      return absl::OutOfRangeError("source read exceeds source data");
    }
    size_t copied = 0;
    uint64_t cursor = dest_va_offset;
    while (copied < bytes) {
      bool matched = false;
      for (const auto& window : grant.windows) {
        if (cursor < window.va_offset || cursor >= window.va_offset + window.length) {
          continue;
        }
        const uint64_t window_offset = cursor - window.va_offset;
        const size_t take = static_cast<size_t>(std::min<uint64_t>(bytes - copied, window.length - window_offset));
        std::memcpy(
            reinterpret_cast<void*>(window.local_addr + window_offset), data_.data() + src_offset + copied, take);
        copied += take;
        cursor += take;
        matched = true;
        break;
      }
      if (!matched) {
        return absl::OutOfRangeError("direct write grant does not cover destination range");
      }
    }
    return copied;
  }

  absl::StatusOr<size_t> readv_into_at(
      absl::Span<const tensorcast::store::loader::DirectWriteOp> ops,
      const tensorcast::store::DirectWriteGrant& grant) override {
    ++readv_calls_;
    size_t total = 0;
    for (const auto& op : ops) {
      auto wrote_or = read_into_at(op.src_offset, op.dest_va_offset, op.bytes, grant);
      if (!wrote_or.ok()) {
        return wrote_or.status();
      }
      total += *wrote_or;
    }
    return total;
  }

  [[nodiscard]] int readv_calls() const {
    return readv_calls_;
  }

 private:
  std::string data_;
  uint64_t cursor_{0};
  int readv_calls_{0};
};

TEST_CASE(
    "BodyBackingManager composite staging writes one source into multiple final bodies",
    "[daemon][body_backing][composite]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  tensorcast::daemon::BodyBackingManager manager(*engine);

  const std::string payload_a = "composite-body-alpha";
  const std::string payload_b = "composite-body-beta-longer";
  const std::string prefix = "source-prefix:";
  const std::string gap = ":gap:";
  const std::string slab = prefix + payload_a + gap + payload_b;
  const std::uint64_t offset_a = prefix.size();
  const std::uint64_t offset_b = prefix.size() + payload_a.size() + gap.size();
  const auto source = std::make_shared<RecordingCompositeSource>(slab);

  const std::string artifact_id_a = make_test_byte_artifact_id("composite-stage-a:blk-4");
  const std::string artifact_id_b = make_test_byte_artifact_id("composite-stage-b:blk-4");
  tensorcast::daemon::v2::PutIfAbsentInvariant invariant_a;
  tensorcast::daemon::v2::PutIfAbsentInvariant invariant_b;
  set_invariant(&invariant_a, "layout_v1", payload_a);
  set_invariant(&invariant_b, "layout_v1", payload_b);
  invariant_a.set_verification_mode(tensorcast::daemon::v2::BYTE_ARTIFACT_VERIFICATION_MODE_LAYOUT_AND_SIZE_ONLY);
  invariant_b.set_verification_mode(tensorcast::daemon::v2::BYTE_ARTIFACT_VERIFICATION_MODE_LAYOUT_AND_SIZE_ONLY);

  auto staged_or = manager.stage_bodies_composite(
      tensorcast::daemon::BodyBackingManager::StageBodiesCompositeRequest{
          .source = source,
          .items =
              {
                  tensorcast::daemon::BodyBackingManager::CompositeStageItem{
                      .artifact_id = artifact_id_a,
                      .invariant = invariant_a,
                      .source_offset = offset_a,
                      .length = payload_a.size(),
                  },
                  tensorcast::daemon::BodyBackingManager::CompositeStageItem{
                      .artifact_id = artifact_id_b,
                      .invariant = invariant_b,
                      .source_offset = offset_b,
                      .length = payload_b.size(),
                  },
              },
          .source_kind = tensorcast::store::loading::MaterializationSource::kP2P,
          .operation_id = "op-body-composite-stage",
          .transport_id = "transport-body-composite-stage",
      });
  REQUIRE(staged_or.ok());
  REQUIRE(staged_or->staged_bodies.size() == 2);
  CHECK(source->readv_calls() > 0);
  CHECK(staged_or->materialize_result.direct_write_supported);

  auto read_a_or = staged_or->staged_bodies[0].body_handle.read_all_bytes();
  auto read_b_or = staged_or->staged_bodies[1].body_handle.read_all_bytes();
  REQUIRE(read_a_or.ok());
  REQUIRE(read_b_or.ok());
  CHECK(*read_a_or == payload_a);
  CHECK(*read_b_or == payload_b);
  CHECK(staged_or->staged_bodies[0].descriptor.payload_digest_alg.empty());
  CHECK(staged_or->staged_bodies[0].descriptor.payload_digest_hex.empty());
  CHECK(
      staged_or->staged_bodies[0].verification_record.verification_method ==
      tensorcast::store::runtime::ingestion::VerificationMethod::kLayoutAndSizeContract);

  REQUIRE(staged_or->staged_bodies[0].body_handle.retire().ok());
  REQUIRE(staged_or->staged_bodies[1].body_handle.retire().ok());
}

TEST_CASE("HomeBatchTouchTtl keeps immortal entries immortal", "[daemon][batch][ttl]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("ttl-immortal:blk-4");
  const std::string payload = "payload-immortal";
  const uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  auto* item = put_req.add_items();
  item->set_artifact_id(artifact_id);
  item->set_inline_payload(payload);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  tensorcast::daemon::v2::HomeBatchTouchTtlRequest touch_req;
  touch_req.mutable_fence()->CopyFrom(put_req.fence());
  touch_req.add_artifact_ids(artifact_id);
  touch_req.set_ttl_ms(1);
  tensorcast::daemon::v2::HomeBatchTouchTtlResponse touch_resp;
  grpc::ServerContext touch_ctx;
  REQUIRE(svc.HomeBatchTouchTtl(&touch_ctx, &touch_req, &touch_resp).ok());
  REQUIRE(touch_resp.outcomes_size() == 1);
  REQUIRE(touch_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  absl::SleepFor(absl::Milliseconds(5));

  HomeBatchExistsRequest exists_req;
  exists_req.mutable_fence()->CopyFrom(put_req.fence());
  exists_req.add_artifact_ids(artifact_id);
  HomeBatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  REQUIRE(svc.HomeBatchExists(&exists_ctx, &exists_req, &exists_resp).ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
}

TEST_CASE("HomeBatch expiry retires retained core replica", "[daemon][batch][ttl][cleanup]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = make_test_byte_artifact_id("ttl-cleanup:blk-7");
  const std::string payload = "payload-expire-cleanup";
  const uint64_t shard_id = shard_for_artifact(artifact_id);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_ttl_ms(1);
  auto* item = put_req.add_items();
  item->set_artifact_id(artifact_id);
  item->set_inline_payload(payload);
  set_invariant(item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(svc.HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(count_cpu_replicas(*engine) == 1);

  absl::SleepFor(absl::Milliseconds(5));

  HomeBatchExistsRequest exists_req;
  exists_req.mutable_fence()->CopyFrom(put_req.fence());
  exists_req.add_artifact_ids(artifact_id);
  HomeBatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  REQUIRE(svc.HomeBatchExists(&exists_ctx, &exists_req, &exists_resp).ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_MISS);
  REQUIRE(count_cpu_replicas(*engine) == 0);

  const auto entry = harness->kernel().byte_artifact_body_store().get(
      artifact_id,
      shard_id,
      /*lease_generation=*/1,
      /*routing_epoch=*/1,
      absl::Now());
  REQUIRE_FALSE(entry.has_value());
}
