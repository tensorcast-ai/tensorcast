// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <unistd.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_ipc.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "core/store/testing/global_store_client_stub.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace {

using tensorcast::daemon::DaemonOptions;
using tensorcast::daemon::DaemonServiceHarness;
using tensorcast::daemon::v2::BatchExistsRequest;
using tensorcast::daemon::v2::BatchExistsResponse;
using tensorcast::daemon::v2::BatchGetIntoRegionRequest;
using tensorcast::daemon::v2::BatchGetIntoRegionResponse;
using tensorcast::daemon::v2::BatchItemStatus;
using tensorcast::daemon::v2::BatchPutIfAbsentFromRegionRequest;
using tensorcast::daemon::v2::BatchPutIfAbsentFromRegionResponse;
using tensorcast::store::components::AcquireShardHomeLeaseResult;
using tensorcast::store::components::ActiveWorkerInfo;
using tensorcast::store::components::RpcOptions;
using tensorcast::store::components::ShardHomeLeaseDescriptor;
using tensorcast::store::components::ShardHomeRouteInfo;
using tensorcast::store::testing::GlobalStoreClientStub;

constexpr const char* kFrontDaemonId = "daemon-front-door";
constexpr const char* kHomeDaemonId = "daemon-home";
constexpr std::uint64_t kShardHomeEligibleFlag =
    (1ULL << tensorcast::global_store::v1::WORKER_CAPABILITY_FLAG_SHARD_HOME_ELIGIBLE);

tensorcast::store::DeviceKey cpu_device() {
  return tensorcast::store::DeviceKey{.type = tensorcast::DeviceType::CPU, .ordinal = -1, .uuid = ""};
}

std::size_t count_cpu_replicas(const tensorcast::store::StoreEngine& engine) {
  return engine.list_device_replicas(cpu_device()).size();
}

static tensorcast::store::StoreEngineOptions make_engine_opts(const std::filesystem::path& root) {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (root / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  return opts;
}

std::filesystem::path make_tmp_dir(std::string_view label) {
  const auto base = std::filesystem::temp_directory_path();
  const auto name = std::string("tensorcast_batch_redirect_") + std::string(label) + "_" + std::to_string(getpid());
  const auto path = base / name;
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

uint64_t shard_for_artifact(std::string_view artifact_id, uint64_t shard_count) {
  const auto digest = tensorcast::common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(artifact_id.data()), artifact_id.size()));
  uint64_t hash64 = 0;
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    hash64 |= static_cast<uint64_t>(digest[i]) << (8U * i);
  }
  return hash64 % shard_count;
}

struct RegisteredRegion {
  std::string region_id;
  std::string device_uuid;
  void* device_ptr{nullptr};
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

void release_test_region(tensorcast::daemon::DaemonServiceHarness& harness, const RegisteredRegion& region) {
  auto unregister_or = harness.kernel().region_registry().unregister_region(
      region.region_id,
      region.owner_pid,
      /*force=*/true);
  REQUIRE(unregister_or.ok());
  REQUIRE(tensorcast::cuda::free(region.device_ptr).ok());
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

std::string sha256_hex(std::string_view payload) {
  const auto digest = tensorcast::common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));
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

struct LeaseState {
  uint64_t shard_id{0};
  std::string holder_daemon_id;
  std::string lease_token;
  uint64_t lease_generation{0};
  absl::Time expires_at{absl::UnixEpoch()};
};

class InMemoryShardHomeGlobalStore final : public GlobalStoreClientStub {
 public:
  void upsert_worker(
      std::string daemon_id,
      std::string node_address,
      uint32_t grpc_port,
      uint64_t capability_flags = 0) {
    absl::MutexLock lock(&mu_);
    ActiveWorkerInfo worker;
    worker.worker_id = "worker-" + daemon_id;
    worker.node_id = "node-local";
    worker.node_address = std::move(node_address);
    worker.grpc_port = grpc_port;
    worker.p2p_port = 0;
    worker.accepting_new_requests = true;
    worker.daemon_id = std::move(daemon_id);
    worker.capability_flags = capability_flags;
    workers_[worker.daemon_id] = std::move(worker);
  }

  void seed_lease(uint64_t shard_id, std::string holder_daemon_id, uint64_t lease_generation) {
    absl::MutexLock lock(&mu_);
    LeaseState st;
    st.shard_id = shard_id;
    st.holder_daemon_id = std::move(holder_daemon_id);
    st.lease_token = "token_" + std::to_string(shard_id) + "_" + std::to_string(lease_generation);
    st.lease_generation = lease_generation;
    // Expires immediately unless refreshed via acquire/keepalive.
    st.expires_at = absl::UnixEpoch();
    leases_[shard_id] = std::move(st);
  }

  uint64_t current_generation(uint64_t shard_id) const {
    absl::MutexLock lock(&mu_);
    const auto it = leases_.find(shard_id);
    return it == leases_.end() ? 0 : it->second.lease_generation;
  }

  absl::StatusOr<std::vector<ActiveWorkerInfo>> list_active_workers(
      bool,
      uint64_t required_capability_flags,
      const RpcOptions&) override {
    absl::MutexLock lock(&mu_);
    std::vector<ActiveWorkerInfo> out;
    out.reserve(workers_.size());
    for (const auto& [_, worker] : workers_) {
      if (required_capability_flags != 0 &&
          (worker.capability_flags & required_capability_flags) != required_capability_flags) {
        continue;
      }
      out.push_back(worker);
    }
    return out;
  }

  absl::StatusOr<ShardHomeRouteInfo> get_shard_home_lease(uint64_t shard_id, const RpcOptions&) override {
    absl::MutexLock lock(&mu_);
    const auto it = leases_.find(shard_id);
    if (it == leases_.end() || it->second.holder_daemon_id.empty() || it->second.lease_generation == 0) {
      return absl::NotFoundError("shard lease not found");
    }
    return ShardHomeRouteInfo{
        .shard_id = it->second.shard_id,
        .holder_daemon_id = it->second.holder_daemon_id,
        .lease_generation = it->second.lease_generation,
        // Route freshness is bounded by staleness budgets; avoid coupling tests to wall-clock expiry.
        .expires_at = absl::UnixEpoch(),
    };
  }

  absl::StatusOr<std::vector<ShardHomeRouteInfo>> batch_get_shard_home_leases(
      const std::vector<uint64_t>& shard_ids,
      const RpcOptions&) override {
    std::vector<ShardHomeRouteInfo> out;
    out.reserve(shard_ids.size());
    for (const auto shard_id : shard_ids) {
      auto route_or = get_shard_home_lease(shard_id, RpcOptions{});
      if (route_or.ok()) {
        out.push_back(*route_or);
      }
    }
    return out;
  }

  absl::StatusOr<AcquireShardHomeLeaseResult> acquire_shard_home_lease(
      uint64_t shard_id,
      std::string_view holder_daemon_id,
      uint64_t ttl_ms,
      const RpcOptions&) override {
    absl::MutexLock lock(&mu_);
    const absl::Time now = absl::Now();
    const absl::Time new_expiry = now + absl::Milliseconds(ttl_ms);

    auto& st = leases_[shard_id];
    if (st.shard_id == 0) {
      st.shard_id = shard_id;
    }

    const bool active = st.lease_generation != 0 && !st.holder_daemon_id.empty() && !st.lease_token.empty() &&
        (st.expires_at == absl::UnixEpoch() || st.expires_at > now);

    AcquireShardHomeLeaseResult out;
    if (active && st.holder_daemon_id == holder_daemon_id) {
      st.expires_at = new_expiry;
      out.acquired = true;
      out.lease = ShardHomeLeaseDescriptor{
          .shard_id = st.shard_id,
          .holder_daemon_id = st.holder_daemon_id,
          .lease_token = st.lease_token,
          .lease_generation = st.lease_generation,
          .expires_at = st.expires_at,
      };
      return out;
    }

    if (!active) {
      st.holder_daemon_id = std::string(holder_daemon_id);
      st.lease_generation = (st.lease_generation == 0) ? 1 : (st.lease_generation + 1);
      st.lease_token = "token_" + std::to_string(shard_id) + "_" + std::to_string(st.lease_generation);
      st.expires_at = new_expiry;
      out.acquired = true;
      out.lease = ShardHomeLeaseDescriptor{
          .shard_id = st.shard_id,
          .holder_daemon_id = st.holder_daemon_id,
          .lease_token = st.lease_token,
          .lease_generation = st.lease_generation,
          .expires_at = st.expires_at,
      };
      return out;
    }

    // Conflict: disclose route but never the token.
    out.acquired = false;
    out.lease = ShardHomeLeaseDescriptor{
        .shard_id = st.shard_id,
        .holder_daemon_id = st.holder_daemon_id,
        .lease_token = "",
        .lease_generation = st.lease_generation,
        .expires_at = st.expires_at,
    };
    return out;
  }

  absl::StatusOr<ShardHomeLeaseDescriptor> keepalive_shard_home_lease(
      std::string_view lease_token,
      uint64_t ttl_ms,
      const RpcOptions&) override {
    absl::MutexLock lock(&mu_);
    const absl::Time now = absl::Now();
    for (auto& [_, st] : leases_) {
      if (st.lease_token != lease_token) {
        continue;
      }
      if (st.expires_at != absl::UnixEpoch() && st.expires_at <= now) {
        return absl::NotFoundError("lease expired");
      }
      st.expires_at = now + absl::Milliseconds(ttl_ms);
      return ShardHomeLeaseDescriptor{
          .shard_id = st.shard_id,
          .holder_daemon_id = st.holder_daemon_id,
          .lease_token = st.lease_token,
          .lease_generation = st.lease_generation,
          .expires_at = st.expires_at,
      };
    }
    return absl::NotFoundError("lease token not found");
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, ActiveWorkerInfo> workers_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<uint64_t, LeaseState> leases_ ABSL_GUARDED_BY(mu_);
};

struct GrpcDaemon {
  explicit GrpcDaemon(
      std::string daemon_id,
      const std::filesystem::path& root,
      const std::shared_ptr<tensorcast::store::StoreEngine>& engine,
      const std::shared_ptr<InMemoryShardHomeGlobalStore>& gs,
      std::chrono::milliseconds lease_ttl,
      std::chrono::milliseconds route_budget,
      std::chrono::milliseconds worker_budget,
      std::chrono::milliseconds keepalive_interval,
      bool shard_home_eligible,
      uint64_t routing_epoch,
      bool start_server,
      uint64_t inline_payload_threshold_bytes = (1ULL << 20)) {
    DaemonOptions opts;
    opts.storage_path = root;
    opts.daemon_id = std::move(daemon_id);
    opts.capability_tokens.active.version = 1;
    opts.capability_tokens.active.secret = "batch-redirect-secret";
    opts.byte_artifact_routing.lease_ttl = lease_ttl;
    opts.byte_artifact_routing.route_staleness_budget = route_budget;
    opts.byte_artifact_routing.worker_directory_staleness_budget = worker_budget;
    opts.byte_artifact_routing.keepalive_interval = keepalive_interval;
    opts.byte_artifact_routing.shard_home_eligible = shard_home_eligible;
    opts.byte_artifact_routing.routing_epoch = routing_epoch;
    opts.byte_artifact_routing.inline_payload_threshold_bytes = inline_payload_threshold_bytes;

    auto harness_or = DaemonServiceHarness::create(engine, opts, /*async_runtime=*/nullptr, gs);
    REQUIRE(harness_or.ok());
    harness = std::move(*harness_or);
    REQUIRE(harness->start().ok());

    if (start_server) {
      grpc::ServerBuilder builder;
      int selected_port = 0;
      builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
      builder.RegisterService(&harness->service());
      server = builder.BuildAndStart();
      REQUIRE(server != nullptr);
      REQUIRE(selected_port != 0);
      address = "127.0.0.1:" + std::to_string(selected_port);
    }
  }

  ~GrpcDaemon() {
    if (server) {
      server->Shutdown();
    }
  }

  std::unique_ptr<DaemonServiceHarness> harness;
  std::unique_ptr<grpc::Server> server;
  std::string address;
};

TEST_CASE("BatchExists retries on stale shard-home fence redirect (remote home)", "[daemon][batch][redirect][e2e]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front");
  const auto root_home = make_tmp_dir("home");
  auto engine_front = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_front));
  auto engine_home = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_home));

  const auto lease_ttl = std::chrono::milliseconds(50);
  const auto route_budget = std::chrono::seconds(10);
  const auto worker_budget = std::chrono::seconds(10);
  const auto keepalive_interval = std::chrono::hours(1);

  GrpcDaemon home(
      kHomeDaemonId,
      root_home,
      engine_home,
      gs,
      lease_ttl,
      route_budget,
      worker_budget,
      keepalive_interval,
      /*shard_home_eligible=*/true,
      /*routing_epoch=*/1,
      /*start_server=*/true);
  gs->upsert_worker(
      kHomeDaemonId,
      "127.0.0.1",
      static_cast<uint32_t>(std::stoi(home.address.substr(home.address.find(':') + 1))),
      kShardHomeEligibleFlag);

  GrpcDaemon front(
      kFrontDaemonId,
      root_front,
      engine_front,
      gs,
      lease_ttl,
      route_budget,
      worker_budget,
      keepalive_interval,
      /*shard_home_eligible=*/true,
      /*routing_epoch=*/1,
      /*start_server=*/false);
  gs->upsert_worker(kFrontDaemonId, "127.0.0.1", /*grpc_port=*/1, kShardHomeEligibleFlag);

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azE";
  const uint64_t shard_id = shard_for_artifact(artifact_id, /*shard_count=*/4096ULL);
  gs->seed_lease(shard_id, kHomeDaemonId, /*lease_generation=*/1);

  // First call: populate front-door route cache at generation=1 and verify MISS.
  {
    BatchExistsRequest req;
    req.add_selections()->set_artifact_id(artifact_id);
    BatchExistsResponse resp;
    grpc::ServerContext ctx;
    const auto st = front.harness->service().BatchExists(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.outcomes_size() == 1);
    REQUIRE(resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_MISS);
  }

  // Allow the home daemon's owned lease to expire so it must reacquire, bumping generation.
  std::this_thread::sleep_for(std::chrono::milliseconds(80));

  // Second call: front-door still uses cached generation=1, gets redirect from home, retries, and returns MISS.
  {
    BatchExistsRequest req;
    req.add_selections()->set_artifact_id(artifact_id);
    BatchExistsResponse resp;
    grpc::ServerContext ctx;
    const auto st = front.harness->service().BatchExists(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.outcomes_size() == 1);
    REQUIRE(resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_MISS);
  }

  REQUIRE(gs->current_generation(shard_id) >= 2);
}

TEST_CASE(
    "BatchExists does not acquire when daemon is not shard-home eligible",
    "[daemon][batch][redirect][eligibility]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_ineligible");
  auto engine_front = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_front));

  const auto lease_ttl = std::chrono::milliseconds(50);
  const auto route_budget = std::chrono::seconds(10);
  const auto worker_budget = std::chrono::seconds(10);
  const auto keepalive_interval = std::chrono::hours(1);

  GrpcDaemon front(
      kFrontDaemonId,
      root_front,
      engine_front,
      gs,
      lease_ttl,
      route_budget,
      worker_budget,
      keepalive_interval,
      /*shard_home_eligible=*/false,
      /*routing_epoch=*/1,
      /*start_server=*/false);
  gs->upsert_worker(kFrontDaemonId, "127.0.0.1", /*grpc_port=*/1, /*capability_flags=*/0);

  BatchExistsRequest req;
  req.add_selections()->set_artifact_id("cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azU");
  BatchExistsResponse resp;
  grpc::ServerContext ctx;
  const auto st = front.harness->service().BatchExists(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.outcomes_size() == 1);
  REQUIRE(resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_UNAVAILABLE);
}

TEST_CASE("BatchExists fails closed on routing epoch mismatch", "[daemon][batch][redirect][epoch]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_epoch");
  const auto root_home = make_tmp_dir("home_epoch");
  auto engine_front = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_front));
  auto engine_home = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_home));

  const auto lease_ttl = std::chrono::milliseconds(50);
  const auto route_budget = std::chrono::seconds(10);
  const auto worker_budget = std::chrono::seconds(10);
  const auto keepalive_interval = std::chrono::hours(1);

  GrpcDaemon home(
      kHomeDaemonId,
      root_home,
      engine_home,
      gs,
      lease_ttl,
      route_budget,
      worker_budget,
      keepalive_interval,
      /*shard_home_eligible=*/true,
      /*routing_epoch=*/2,
      /*start_server=*/true);
  gs->upsert_worker(
      kHomeDaemonId,
      "127.0.0.1",
      static_cast<uint32_t>(std::stoi(home.address.substr(home.address.find(':') + 1))),
      kShardHomeEligibleFlag);

  GrpcDaemon front(
      kFrontDaemonId,
      root_front,
      engine_front,
      gs,
      lease_ttl,
      route_budget,
      worker_budget,
      keepalive_interval,
      /*shard_home_eligible=*/true,
      /*routing_epoch=*/1,
      /*start_server=*/false);
  gs->upsert_worker(kFrontDaemonId, "127.0.0.1", /*grpc_port=*/1, kShardHomeEligibleFlag);

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azY";
  const uint64_t shard_id = shard_for_artifact(artifact_id, /*shard_count=*/4096ULL);
  gs->seed_lease(shard_id, kHomeDaemonId, /*lease_generation=*/1);

  BatchExistsRequest req;
  req.add_selections()->set_artifact_id(artifact_id);
  BatchExistsResponse resp;
  grpc::ServerContext ctx;
  const auto st = front.harness->service().BatchExists(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.outcomes_size() == 1);
  REQUIRE(resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_UNAVAILABLE);
}

TEST_CASE("Batch get/put transport payload_ref over remote home daemon", "[daemon][batch][redirect][transport]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_transport");
  const auto root_home = make_tmp_dir("home_transport");
  auto engine_front = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_front));
  auto engine_home = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_home));

  const auto lease_ttl = std::chrono::seconds(5);
  const auto route_budget = std::chrono::seconds(10);
  const auto worker_budget = std::chrono::seconds(10);
  const auto keepalive_interval = std::chrono::hours(1);

  GrpcDaemon home(
      kHomeDaemonId,
      root_home,
      engine_home,
      gs,
      lease_ttl,
      route_budget,
      worker_budget,
      keepalive_interval,
      /*shard_home_eligible=*/true,
      /*routing_epoch=*/1,
      /*start_server=*/true,
      /*inline_payload_threshold_bytes=*/8);
  gs->upsert_worker(
      kHomeDaemonId,
      "127.0.0.1",
      static_cast<uint32_t>(std::stoi(home.address.substr(home.address.find(':') + 1))),
      kShardHomeEligibleFlag);

  GrpcDaemon front(
      kFrontDaemonId,
      root_front,
      engine_front,
      gs,
      lease_ttl,
      route_budget,
      worker_budget,
      keepalive_interval,
      /*shard_home_eligible=*/true,
      /*routing_epoch=*/1,
      /*start_server=*/true,
      /*inline_payload_threshold_bytes=*/8);
  gs->upsert_worker(
      kFrontDaemonId,
      "127.0.0.1",
      static_cast<uint32_t>(std::stoi(front.address.substr(front.address.find(':') + 1))),
      kShardHomeEligibleFlag);

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azg";
  const std::string payload = "remote-payload-ref-transport";
  const uint64_t shard_id = shard_for_artifact(artifact_id, /*shard_count=*/4096ULL);
  gs->seed_lease(shard_id, kHomeDaemonId, /*lease_generation=*/1);

  auto source_region = register_test_region(*front.harness, /*device_id=*/0, payload.size());
  REQUIRE(
      tensorcast::cuda::memcpy(source_region.device_ptr, payload.data(), payload.size(), cudaMemcpyHostToDevice).ok());

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* put_item = put_req.add_items();
  put_item->mutable_selection()->set_artifact_id(artifact_id);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);
  populate_single_region_layout(
      put_req.mutable_source_layout(),
      source_region,
      artifact_id,
      payload.size(),
      /*device_id=*/0);
  put_req.set_pid(source_region.owner_pid);
  put_req.set_device_uuid(source_region.device_uuid);
  put_req.set_operation_id("op-remote-payload-ref");

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  const auto put_st = front.harness->service().BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp);
  REQUIRE(put_st.ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  auto target_region = register_test_region(*front.harness, /*device_id=*/0, payload.size());
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
  get_req.set_operation_id("op-remote-payload-ref");

  BatchGetIntoRegionResponse get_resp;
  grpc::ServerContext get_ctx;
  const auto get_st = front.harness->service().BatchGetIntoRegion(&get_ctx, &get_req, &get_resp);
  REQUIRE(get_st.ok());
  REQUIRE(get_resp.outcomes_size() == 1);
  REQUIRE(get_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  std::vector<char> out(payload.size(), '\0');
  REQUIRE(tensorcast::cuda::memcpy(out.data(), target_region.device_ptr, out.size(), cudaMemcpyDeviceToHost).ok());
  REQUIRE(std::string(out.data(), out.size()) == payload);

  release_test_region(*front.harness, source_region);
  release_test_region(*front.harness, target_region);
}

TEST_CASE(
    "BatchPutIfAbsentFromRegion retires transient staged replica when remote home is unreachable",
    "[daemon][batch][redirect][cleanup]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_failed_forward_cleanup");
  auto engine_front = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_front));

  const auto lease_ttl = std::chrono::seconds(5);
  const auto route_budget = std::chrono::seconds(1);
  const auto worker_budget = std::chrono::seconds(1);
  const auto keepalive_interval = std::chrono::hours(1);

  GrpcDaemon front(
      kFrontDaemonId,
      root_front,
      engine_front,
      gs,
      lease_ttl,
      route_budget,
      worker_budget,
      keepalive_interval,
      /*shard_home_eligible=*/true,
      /*routing_epoch=*/1,
      /*start_server=*/false,
      /*inline_payload_threshold_bytes=*/8);
  gs->upsert_worker(kFrontDaemonId, "127.0.0.1", /*grpc_port=*/1, kShardHomeEligibleFlag);

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azk";
  const std::string payload = "failed-forward-cleanup";
  const uint64_t shard_id = shard_for_artifact(artifact_id, /*shard_count=*/4096ULL);
  gs->seed_lease(shard_id, kHomeDaemonId, /*lease_generation=*/1);

  auto source_region = register_test_region(*front.harness, /*device_id=*/0, payload.size());
  REQUIRE(
      tensorcast::cuda::memcpy(source_region.device_ptr, payload.data(), payload.size(), cudaMemcpyHostToDevice).ok());

  REQUIRE(count_cpu_replicas(*engine_front) == 0);

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* put_item = put_req.add_items();
  put_item->mutable_selection()->set_artifact_id(artifact_id);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);
  populate_single_region_layout(
      put_req.mutable_source_layout(),
      source_region,
      artifact_id,
      payload.size(),
      /*device_id=*/0);
  put_req.set_pid(source_region.owner_pid);
  put_req.set_device_uuid(source_region.device_uuid);
  put_req.set_operation_id("op-forward-cleanup");

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  const auto put_st = front.harness->service().BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp);
  REQUIRE(put_st.ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_UNAVAILABLE);
  REQUIRE(count_cpu_replicas(*engine_front) == 0);

  release_test_region(*front.harness, source_region);
}

} // namespace
