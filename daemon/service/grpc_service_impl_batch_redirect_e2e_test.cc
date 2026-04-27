// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <unistd.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/capability_token.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_ipc.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "core/store/testing/global_store_client_stub.h"
#include "daemon/state/ipc_region_registry.h"
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
using tensorcast::daemon::v2::HomeBatchPutIfAbsentRequest;
using tensorcast::daemon::v2::HomeBatchPutIfAbsentResponse;
using tensorcast::store::components::AcquireShardHomeLeaseResult;
using tensorcast::store::components::ActiveWorkerInfo;
using tensorcast::store::components::RpcOptions;
using tensorcast::store::components::ShardHomeLeaseDescriptor;
using tensorcast::store::components::ShardHomeLeaseKeepaliveInput;
using tensorcast::store::components::ShardHomeLeaseKeepaliveOutcome;
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

static tensorcast::store::StoreEngineOptions make_engine_opts(
    const std::filesystem::path& root,
    uint16_t p2p_port = 0) {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (root / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_listen_host = "127.0.0.1";
  opts.p2p_port = p2p_port;
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

uint32_t grpc_port_from_address(std::string_view address) {
  return static_cast<uint32_t>(std::stoi(std::string(address.substr(address.find(':') + 1))));
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
  return make_valid_byte_artifact_id("tenant", "engine", "batch-redirect-model", "v1", layout_id, engine_key);
}

std::string replace_final_b64u_suffix(std::string_view artifact_id, std::string_view suffix) {
  const std::size_t marker = artifact_id.rfind("~b64u.");
  REQUIRE(marker != std::string_view::npos);
  return absl::StrCat(artifact_id.substr(0, marker + 6), suffix);
}

std::string artifact_on_same_shard(
    std::string_view base_artifact_id,
    std::string_view seed,
    uint64_t shard_count = 4096ULL) {
  const std::uint64_t target_shard = shard_for_artifact(base_artifact_id, shard_count);
  for (int attempt = 0; attempt < 10000; ++attempt) {
    const std::string candidate = replace_final_b64u_suffix(base_artifact_id, absl::StrCat(seed, attempt));
    if (candidate != base_artifact_id && shard_for_artifact(candidate, shard_count) == target_shard) {
      return candidate;
    }
  }
  FAIL("failed to find same-shard artifact id");
  return {};
}

uint64_t shard_home_hrw_score_for_test(uint64_t shard_id, std::string_view daemon_id) {
  const std::string key = absl::StrCat("byte-artifact-home:", shard_id, ":", daemon_id);
  const auto digest = tensorcast::common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(key.data()), key.size()));
  uint64_t score = 0;
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    score |= static_cast<uint64_t>(digest[i]) << (8U * i);
  }
  return score;
}

std::string expected_shard_home_owner_for_test(uint64_t shard_id, absl::Span<const std::string> daemon_ids) {
  std::string best_daemon_id;
  uint64_t best_score = 0;
  for (const auto& daemon_id : daemon_ids) {
    const auto score = shard_home_hrw_score_for_test(shard_id, daemon_id);
    if (best_daemon_id.empty() || score > best_score || (score == best_score && daemon_id < best_daemon_id)) {
      best_daemon_id = daemon_id;
      best_score = score;
    }
  }
  return best_daemon_id;
}

std::string find_artifact_id_for_expected_home(
    std::string_view expected_owner,
    absl::Span<const std::string> daemon_ids,
    uint64_t shard_count = 4096ULL) {
  for (uint64_t index = 0; index < 10000; ++index) {
    const std::string artifact_id = make_test_byte_artifact_id(absl::StrCat("expected-owner:", index));
    const uint64_t shard_id = shard_for_artifact(artifact_id, shard_count);
    if (expected_shard_home_owner_for_test(shard_id, daemon_ids) == expected_owner) {
      return artifact_id;
    }
  }
  FAIL("could not find test artifact with desired expected shard-home owner");
  return {};
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

void release_test_region(tensorcast::daemon::DaemonServiceHarness& harness, const RegisteredRegion& region) {
  auto unregister_or = harness.kernel().region_registry().unregister_region(
      region.region_id,
      region.owner_pid,
      /*force=*/true);
  REQUIRE(unregister_or.ok());
  REQUIRE(tensorcast::cuda::free(region.device_ptr).ok());
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

class CollectingLogSink : public absl::LogSink {
 public:
  void Send(const absl::LogEntry& entry) override {
    absl::MutexLock lock(&mu_);
    messages_.push_back(std::string(entry.text_message()));
  }

  bool Contains(absl::string_view needle) const {
    absl::MutexLock lock(&mu_);
    for (const auto& msg : messages_) {
      if (msg.find(needle) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

 private:
  mutable absl::Mutex mu_;
  std::vector<std::string> messages_ ABSL_GUARDED_BY(mu_);
};

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

void set_layout_only_invariant(
    tensorcast::daemon::v2::PutIfAbsentInvariant* invariant,
    std::string_view layout_id,
    std::string_view payload) {
  invariant->set_layout_id(std::string(layout_id));
  invariant->set_byte_length(payload.size());
  invariant->set_verification_mode(tensorcast::daemon::v2::BYTE_ARTIFACT_VERIFICATION_MODE_LAYOUT_AND_SIZE_ONLY);
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
      uint64_t capability_flags = 0,
      uint32_t p2p_port = 0,
      std::string node_id = "node-local") {
    absl::MutexLock lock(&mu_);
    ActiveWorkerInfo worker;
    worker.worker_id = "worker-" + daemon_id;
    worker.node_id = std::move(node_id);
    worker.node_address = std::move(node_address);
    worker.grpc_port = grpc_port;
    worker.p2p_port = p2p_port;
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

  absl::StatusOr<std::vector<ShardHomeLeaseKeepaliveOutcome>> batch_keepalive_shard_home_leases(
      const std::vector<ShardHomeLeaseKeepaliveInput>& leases,
      uint64_t ttl_ms,
      const RpcOptions&) override {
    std::vector<ShardHomeLeaseKeepaliveOutcome> out;
    out.reserve(leases.size());

    absl::MutexLock lock(&mu_);
    const absl::Time now = absl::Now();
    for (const auto& lease : leases) {
      ShardHomeLeaseKeepaliveOutcome outcome;
      outcome.shard_id = lease.shard_id;
      outcome.lease_generation = lease.lease_generation;
      outcome.lease_token = lease.lease_token;

      const auto it = leases_.find(lease.shard_id);
      if (it == leases_.end() || it->second.lease_token != lease.lease_token) {
        outcome.ok = false;
        outcome.message = "lease token not found";
        out.push_back(std::move(outcome));
        continue;
      }

      auto& st = it->second;
      if (st.expires_at != absl::UnixEpoch() && st.expires_at <= now) {
        outcome.ok = false;
        outcome.message = "lease expired";
        out.push_back(std::move(outcome));
        continue;
      }
      if (st.lease_generation != lease.lease_generation) {
        outcome.ok = false;
        outcome.message = "lease generation mismatch";
        outcome.lease = ShardHomeLeaseDescriptor{
            .shard_id = st.shard_id,
            .holder_daemon_id = st.holder_daemon_id,
            .lease_token = st.lease_token,
            .lease_generation = st.lease_generation,
            .expires_at = st.expires_at,
        };
        out.push_back(std::move(outcome));
        continue;
      }

      st.expires_at = now + absl::Milliseconds(ttl_ms);
      outcome.ok = true;
      outcome.lease = ShardHomeLeaseDescriptor{
          .shard_id = st.shard_id,
          .holder_daemon_id = st.holder_daemon_id,
          .lease_token = st.lease_token,
          .lease_generation = st.lease_generation,
          .expires_at = st.expires_at,
      };
      out.push_back(std::move(outcome));
    }
    return out;
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
      start_insecure_server();
    }
  }

  void start_insecure_server() {
    grpc::ServerBuilder builder;
    int selected_port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
    builder.RegisterService(&harness->service());
    server = builder.BuildAndStart();
    REQUIRE(server != nullptr);
    REQUIRE(selected_port != 0);
    address = "127.0.0.1:" + std::to_string(selected_port);
  }

  void shutdown_server() {
    if (server) {
      server->Shutdown();
      server.reset();
    }
    address.clear();
  }

  ~GrpcDaemon() {
    shutdown_server();
  }

  std::unique_ptr<DaemonServiceHarness> harness;
  std::unique_ptr<grpc::Server> server;
  std::string address;
};

class RouteAuthorityStageTestService final : public tensorcast::daemon::v2::StoreDaemonService::Service {
 public:
  using ReplyBuilder = std::function<void(
      const tensorcast::daemon::v2::RouteAuthorityStageRequest&,
      tensorcast::daemon::v2::RouteAuthorityStageResponse*)>;
  using FetchBuilder = std::function<void(
      const tensorcast::daemon::v2::FetchPayloadRefChunkRequest&,
      tensorcast::daemon::v2::FetchPayloadRefChunkResponse*)>;

  explicit RouteAuthorityStageTestService(ReplyBuilder reply_builder, FetchBuilder fetch_builder = {})
      : reply_builder_(std::move(reply_builder)), fetch_builder_(std::move(fetch_builder)) {}

  grpc::Status RouteAuthorityStage(
      grpc::ServerContext*,
      const tensorcast::daemon::v2::RouteAuthorityStageRequest* req,
      tensorcast::daemon::v2::RouteAuthorityStageResponse* resp) override {
    if (reply_builder_) {
      reply_builder_(*req, resp);
    } else {
      resp->set_status(tensorcast::daemon::v2::BATCH_ITEM_STATUS_OK);
      auto* reply = resp->mutable_owner_stage_reply();
      reply->mutable_answered_by()->CopyFrom(req->routed_request().authority_ref());
      reply->set_path_family(req->routed_request().path_family());
      reply->set_stage_ref(req->routed_request().stage_ref());
      reply->set_reply_kind(tensorcast::daemon::v2::ROUTED_OWNER_STAGE_REPLY_KIND_READY_FOR_LOWERING);
      reply->set_resolved_source_capability("{}");
    }
    return grpc::Status::OK;
  }

  grpc::Status FetchPayloadRefChunk(
      grpc::ServerContext*,
      const tensorcast::daemon::v2::FetchPayloadRefChunkRequest* req,
      tensorcast::daemon::v2::FetchPayloadRefChunkResponse* resp) override {
    if (fetch_builder_) {
      fetch_builder_(*req, resp);
    } else {
      resp->set_status(tensorcast::daemon::v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION);
      resp->set_message("fetch handler not configured");
    }
    return grpc::Status::OK;
  }

 private:
  ReplyBuilder reply_builder_;
  FetchBuilder fetch_builder_;
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

  const std::string artifact_id = make_test_byte_artifact_id("stale-redirect:blk-1");
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
    "BatchExists acquires unleased remote home via expected owner",
    "[daemon][batch][redirect][e2e][expected_owner]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_expected_owner_exists");
  const auto root_home = make_tmp_dir("home_expected_owner_exists");
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
      /*start_server=*/true);
  gs->upsert_worker(kHomeDaemonId, "127.0.0.1", grpc_port_from_address(home.address), kShardHomeEligibleFlag);

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

  const std::vector<std::string> daemon_ids{std::string(kFrontDaemonId), std::string(kHomeDaemonId)};
  const std::string artifact_id = find_artifact_id_for_expected_home(kHomeDaemonId, daemon_ids);
  const uint64_t shard_id = shard_for_artifact(artifact_id, /*shard_count=*/4096ULL);
  REQUIRE_FALSE(gs->get_shard_home_lease(shard_id, RpcOptions{}).ok());

  BatchExistsRequest req;
  req.add_selections()->set_artifact_id(artifact_id);
  BatchExistsResponse resp;
  grpc::ServerContext ctx;
  const auto st = front.harness->service().BatchExists(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.outcomes_size() == 1);
  REQUIRE(resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_MISS);

  auto route_or = gs->get_shard_home_lease(shard_id, RpcOptions{});
  REQUIRE(route_or.ok());
  REQUIRE(route_or->holder_daemon_id == kHomeDaemonId);
}

TEST_CASE(
    "BatchExists preserves published authority after home lease reacquire",
    "[daemon][batch][redirect][e2e][lease_reacquire]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_exists_after_reacquire");
  const auto root_home = make_tmp_dir("home_exists_after_reacquire");
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

  const std::string artifact_id = make_test_byte_artifact_id("lease-reacquire:blk-2");
  const std::string payload = "published-before-reacquire";
  const uint64_t shard_id = shard_for_artifact(artifact_id, /*shard_count=*/4096ULL);
  gs->seed_lease(shard_id, kHomeDaemonId, /*lease_generation=*/1);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kHomeDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  auto* put_item = put_req.add_items();
  put_item->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(home.harness->service().HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  {
    BatchExistsRequest req;
    req.add_selections()->set_artifact_id(artifact_id);
    BatchExistsResponse resp;
    grpc::ServerContext ctx;
    const auto st = front.harness->service().BatchExists(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.outcomes_size() == 1);
    REQUIRE(resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(80));

  {
    BatchExistsRequest req;
    req.add_selections()->set_artifact_id(artifact_id);
    BatchExistsResponse resp;
    grpc::ServerContext ctx;
    const auto st = front.harness->service().BatchExists(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.outcomes_size() == 1);
    REQUIRE(resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  }

  REQUIRE(gs->current_generation(shard_id) >= 2);
}

TEST_CASE(
    "BatchExists keeps shard-home lease alive while home daemon stays idle",
    "[daemon][batch][redirect][e2e][lease_keepalive]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_exists_with_keepalive");
  const auto root_home = make_tmp_dir("home_exists_with_keepalive");
  auto engine_front = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_front));
  auto engine_home = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_home));

  const auto lease_ttl = std::chrono::milliseconds(50);
  const auto route_budget = std::chrono::seconds(10);
  const auto worker_budget = std::chrono::seconds(10);
  const auto keepalive_interval = std::chrono::milliseconds(10);

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

  const std::string artifact_id = make_test_byte_artifact_id("keepalive-idle:blk-7");
  const std::string payload = "published-before-idle";
  const uint64_t shard_id = shard_for_artifact(artifact_id, /*shard_count=*/4096ULL);
  gs->seed_lease(shard_id, kHomeDaemonId, /*lease_generation=*/1);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kHomeDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  auto* put_item = put_req.add_items();
  put_item->set_artifact_id(artifact_id);
  put_item->set_inline_payload(payload);
  set_invariant(put_item->mutable_invariant(), "layout_v1", payload);

  HomeBatchPutIfAbsentResponse put_resp;
  grpc::ServerContext put_ctx;
  REQUIRE(home.harness->service().HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  std::this_thread::sleep_for(std::chrono::milliseconds(120));

  {
    BatchExistsRequest req;
    req.add_selections()->set_artifact_id(artifact_id);
    BatchExistsResponse resp;
    grpc::ServerContext ctx;
    const auto st = front.harness->service().BatchExists(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.outcomes_size() == 1);
    REQUIRE(resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  }

  REQUIRE(gs->current_generation(shard_id) == 1);
}

TEST_CASE(
    "ensure_home_lease keeps route-only shard-home leases alive before first authority entry",
    "[daemon][batch][redirect][e2e][lease_keepalive][route_only]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_home = make_tmp_dir("home_route_only_keepalive");
  auto engine_home = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_home));

  const auto lease_ttl = std::chrono::milliseconds(50);
  const auto route_budget = std::chrono::seconds(10);
  const auto worker_budget = std::chrono::seconds(10);
  const auto keepalive_interval = std::chrono::milliseconds(10);

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
      /*start_server=*/false);
  gs->upsert_worker(kHomeDaemonId, "127.0.0.1", /*grpc_port=*/1, kShardHomeEligibleFlag);

  constexpr uint64_t shard_id = 1729;
  auto& resolver = home.harness->kernel().byte_artifact_route_resolver();
  const auto route = resolver.resolve_route(shard_id, absl::Now());
  REQUIRE(route.ok);
  REQUIRE(route.holder_daemon_id == kHomeDaemonId);
  REQUIRE(route.lease_generation == 1);

  tensorcast::daemon::v2::RouteFence fence;
  fence.set_shard_id(shard_id);
  fence.set_lease_generation(route.lease_generation);
  fence.set_holder_daemon_id(kHomeDaemonId);
  fence.set_routing_epoch(1);

  std::this_thread::sleep_for(std::chrono::milliseconds(120));

  auto ensure_or = resolver.ensure_home_lease(fence, absl::Now());
  REQUIRE(ensure_or.ok());
  REQUIRE(ensure_or->kind == tensorcast::daemon::ByteArtifactRouteResolver::HomeLeaseDecision::Kind::kOwned);
  REQUIRE(ensure_or->lease_generation == 1);
  REQUIRE(gs->current_generation(shard_id) == 1);
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
  req.add_selections()->set_artifact_id(make_test_byte_artifact_id("ineligible:blk-5"));
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

  const std::string artifact_id = make_test_byte_artifact_id("epoch-mismatch:blk-6");
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

  const std::string artifact_id = make_test_byte_artifact_id("remote-transport:blk-8");
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
    "BatchPutIfAbsentFromRegion acquires unleased remote home via expected owner",
    "[daemon][batch][redirect][transport][expected_owner]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_expected_owner_put");
  const auto root_home = make_tmp_dir("home_expected_owner_put");
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
  gs->upsert_worker(kHomeDaemonId, "127.0.0.1", grpc_port_from_address(home.address), kShardHomeEligibleFlag);

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
  gs->upsert_worker(kFrontDaemonId, "127.0.0.1", grpc_port_from_address(front.address), kShardHomeEligibleFlag);

  const std::vector<std::string> daemon_ids{std::string(kFrontDaemonId), std::string(kHomeDaemonId)};
  const std::string artifact_id = find_artifact_id_for_expected_home(kHomeDaemonId, daemon_ids);
  const std::string payload = "remote-put-expected-owner";
  const uint64_t shard_id = shard_for_artifact(artifact_id, /*shard_count=*/4096ULL);
  REQUIRE_FALSE(gs->get_shard_home_lease(shard_id, RpcOptions{}).ok());

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
  put_req.set_operation_id("op-expected-owner-put");

  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  const auto put_st = front.harness->service().BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp);
  REQUIRE(put_st.ok());
  REQUIRE(put_resp.outcomes_size() == 1);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  auto route_or = gs->get_shard_home_lease(shard_id, RpcOptions{});
  REQUIRE(route_or.ok());
  REQUIRE(route_or->holder_daemon_id == kHomeDaemonId);
  REQUIRE(count_cpu_replicas(*engine_front) == 0);
  REQUIRE(count_cpu_replicas(*engine_home) == 1);

  BatchExistsRequest exists_req;
  exists_req.add_selections()->set_artifact_id(artifact_id);
  BatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  const auto exists_st = front.harness->service().BatchExists(&exists_ctx, &exists_req, &exists_resp);
  REQUIRE(exists_st.ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

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
  get_req.set_operation_id("op-expected-owner-get");

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
    "BatchGetIntoRegion prefers single-source composite materialization for remote communicator transports",
    "[daemon][batch][redirect][transport][communicator][host_shared]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_composite_batch_get");
  const auto root_home = make_tmp_dir("home_composite_batch_get");

  const auto lease_ttl = std::chrono::seconds(5);
  const auto route_budget = std::chrono::seconds(10);
  const auto worker_budget = std::chrono::seconds(10);
  const auto keepalive_interval = std::chrono::hours(1);
  constexpr uint16_t kHomeP2PPort = 47141;
  constexpr uint16_t kFrontP2PPort = 47142;
  auto engine_front = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_front, kFrontP2PPort));
  auto engine_home = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_home, kHomeP2PPort));

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
      kHomeDaemonId, "127.0.0.1", grpc_port_from_address(home.address), kShardHomeEligibleFlag, kHomeP2PPort);

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
      kFrontDaemonId, "127.0.0.1", grpc_port_from_address(front.address), kShardHomeEligibleFlag, kFrontP2PPort);

  const std::vector<std::string> daemon_ids{std::string(kFrontDaemonId), std::string(kHomeDaemonId)};
  const std::string artifact_id_a = find_artifact_id_for_expected_home(kHomeDaemonId, daemon_ids);
  const std::string artifact_id_b = artifact_on_same_shard(artifact_id_a, "composite-batch-get");
  const std::string payload_a(64, 'a');
  const std::string payload_b(96, 'b');
  const std::uint64_t shard_id = shard_for_artifact(artifact_id_a, /*shard_count=*/4096ULL);

  HomeBatchPutIfAbsentRequest put_req;
  put_req.mutable_fence()->set_shard_id(shard_id);
  put_req.mutable_fence()->set_lease_generation(1);
  put_req.mutable_fence()->set_holder_daemon_id(kHomeDaemonId);
  put_req.mutable_fence()->set_routing_epoch(1);
  put_req.set_operation_id("op-composite-batch-get");
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
  REQUIRE(home.harness->service().HomeBatchPutIfAbsent(&put_ctx, &put_req, &put_resp).ok());
  REQUIRE(put_resp.outcomes_size() == 2);
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(put_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);

  auto target_region = register_test_host_shared_region(*front.harness, payload_a.size() + payload_b.size());
  std::memset(target_region.base_ptr, 0, static_cast<std::size_t>(target_region.size_bytes));

  BatchGetIntoRegionRequest get_req;
  get_req.add_selections()->set_artifact_id(artifact_id_a);
  get_req.add_selections()->set_artifact_id(artifact_id_b);
  populate_two_item_host_shared_region_layout(
      get_req.mutable_target_layout(), target_region, artifact_id_a, payload_a.size(), artifact_id_b, payload_b.size());
  get_req.set_pid(target_region.owner_pid);
  get_req.set_operation_id("op-composite-batch-get");

  CollectingLogSink sink;
  absl::AddLogSink(&sink);
  BatchGetIntoRegionResponse get_resp;
  grpc::ServerContext get_ctx;
  const auto get_st = front.harness->service().BatchGetIntoRegion(&get_ctx, &get_req, &get_resp);
  absl::RemoveLogSink(&sink);

  REQUIRE(get_st.ok());
  REQUIRE(get_resp.outcomes_size() == 2);
  REQUIRE(get_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(std::string(static_cast<const char*>(target_region.base_ptr), payload_a.size()) == payload_a);
  REQUIRE(
      std::string(static_cast<const char*>(target_region.base_ptr) + payload_a.size(), payload_b.size()) == payload_b);
  if (sink.Contains("byte_artifact.batch_get_into_region_transport_materialize_mode")) {
    CHECK(sink.Contains("materialize_mode=single_source_composite"));
    CHECK(sink.Contains("batched_direct_write=true"));
  } else {
    CHECK(sink.Contains("byte_artifact.batch_get_into_region_transport_mirror"));
    CHECK(sink.Contains("read_mode=full_pack_mirror"));
  }

  release_test_host_shared_region(*front.harness, target_region);
}

TEST_CASE(
    "BatchPutIfAbsentFromRegion exports remote HOST_SHARED source layouts without staged slab",
    "[daemon][batch][redirect][transport][communicator][host_shared][put]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_put_segmented_region");
  const auto root_home = make_tmp_dir("home_put_segmented_region");

  const auto lease_ttl = std::chrono::seconds(5);
  const auto route_budget = std::chrono::seconds(10);
  const auto worker_budget = std::chrono::seconds(10);
  const auto keepalive_interval = std::chrono::hours(1);
  auto engine_front = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_front));
  auto engine_home = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_home));

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
  const uint16_t home_p2p_port = engine_home->get_shared_comm_manager()->listen_port();
  gs->upsert_worker(
      kHomeDaemonId, "127.0.0.1", grpc_port_from_address(home.address), kShardHomeEligibleFlag, home_p2p_port);

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
  const uint16_t front_p2p_port = engine_front->get_shared_comm_manager()->listen_port();
  gs->upsert_worker(
      kFrontDaemonId, "127.0.0.1", grpc_port_from_address(front.address), kShardHomeEligibleFlag, front_p2p_port);

  const std::vector<std::string> daemon_ids{std::string(kFrontDaemonId), std::string(kHomeDaemonId)};
  const std::string artifact_id_a = find_artifact_id_for_expected_home(kHomeDaemonId, daemon_ids);
  const std::string artifact_id_b = artifact_on_same_shard(artifact_id_a, "put-segmented-region");
  const std::string payload_a(64, 'x');
  const std::string payload_b(96, 'y');
  const std::string slab = payload_a + payload_b;
  const std::uint64_t shard_id = shard_for_artifact(artifact_id_a, /*shard_count=*/4096ULL);
  gs->seed_lease(shard_id, kHomeDaemonId, /*lease_generation=*/1);

  auto source_region = register_test_host_shared_region(
      *front.harness, slab.size(), tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator);
  std::memcpy(source_region.base_ptr, slab.data(), slab.size());

  BatchPutIfAbsentFromRegionRequest put_req;
  auto* put_item_a = put_req.add_items();
  put_item_a->mutable_selection()->set_artifact_id(artifact_id_a);
  set_layout_only_invariant(put_item_a->mutable_invariant(), "layout_v1", payload_a);
  auto* put_item_b = put_req.add_items();
  put_item_b->mutable_selection()->set_artifact_id(artifact_id_b);
  set_layout_only_invariant(put_item_b->mutable_invariant(), "layout_v1", payload_b);
  populate_two_item_host_shared_region_layout(
      put_req.mutable_source_layout(), source_region, artifact_id_a, payload_a.size(), artifact_id_b, payload_b.size());
  put_req.mutable_source_layout()->mutable_offsets(0)->set_slot_index(7);
  put_req.mutable_source_layout()->mutable_offsets(0)->set_slot_generation(17);
  put_req.mutable_source_layout()->mutable_offsets(1)->set_slot_index(8);
  put_req.mutable_source_layout()->mutable_offsets(1)->set_slot_generation(18);
  put_req.set_pid(source_region.owner_pid);
  put_req.set_operation_id("op-put-segmented-region");

  CollectingLogSink sink;
  absl::AddLogSink(&sink);
  BatchPutIfAbsentFromRegionResponse put_resp;
  grpc::ServerContext put_ctx;
  const auto put_st = front.harness->service().BatchPutIfAbsentFromRegion(&put_ctx, &put_req, &put_resp);
  absl::RemoveLogSink(&sink);

  REQUIRE(put_st.ok());
  REQUIRE(put_resp.outcomes_size() == 2);
  CAPTURE(put_resp.outcomes(0).message());
  CAPTURE(put_resp.outcomes(1).message());
  REQUIRE(put_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(put_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  CHECK(put_resp.outcomes(0).slot_index() == 7);
  CHECK(put_resp.outcomes(0).slot_generation() == 17);
  CHECK(put_resp.outcomes(1).slot_index() == 8);
  CHECK(put_resp.outcomes(1).slot_generation() == 18);
  CHECK(sink.Contains("mode=segmented_region_export"));
  CHECK(sink.Contains("source_realization_mode=source_layout_host_shared"));
  CHECK_FALSE(sink.Contains("mode=staged_slab"));

  auto target_region = register_test_host_shared_region(*front.harness, slab.size());
  std::memset(target_region.base_ptr, 0, static_cast<std::size_t>(target_region.size_bytes));

  BatchGetIntoRegionRequest get_req;
  get_req.add_selections()->set_artifact_id(artifact_id_a);
  get_req.add_selections()->set_artifact_id(artifact_id_b);
  populate_two_item_host_shared_region_layout(
      get_req.mutable_target_layout(), target_region, artifact_id_a, payload_a.size(), artifact_id_b, payload_b.size());
  get_req.set_pid(target_region.owner_pid);
  get_req.set_operation_id("op-put-segmented-region");

  BatchGetIntoRegionResponse get_resp;
  grpc::ServerContext get_ctx;
  const auto get_st = front.harness->service().BatchGetIntoRegion(&get_ctx, &get_req, &get_resp);
  REQUIRE(get_st.ok());
  REQUIRE(get_resp.outcomes_size() == 2);
  REQUIRE(get_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(get_resp.outcomes(1).status() == BatchItemStatus::BATCH_ITEM_STATUS_OK);
  REQUIRE(std::string(static_cast<const char*>(target_region.base_ptr), payload_a.size()) == payload_a);
  REQUIRE(
      std::string(static_cast<const char*>(target_region.base_ptr) + payload_a.size(), payload_b.size()) == payload_b);

  release_test_host_shared_region(*front.harness, source_region);
  release_test_host_shared_region(*front.harness, target_region);
}

TEST_CASE(
    "payload_ref remote issuer validation uses canonical routed owner rpc",
    "[daemon][batch][redirect][issuer_route]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_payload_ref_issuer_route");
  const auto root_home = make_tmp_dir("home_payload_ref_issuer_route");
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
      /*start_server=*/false,
      /*inline_payload_threshold_bytes=*/8);
  gs->upsert_worker(kFrontDaemonId, "127.0.0.1", /*grpc_port=*/1, kShardHomeEligibleFlag);

  const std::string artifact_id = make_test_byte_artifact_id("issuer-route-remote:blk-2");
  const std::string payload = "remote-issuer-route-payload";

  auto payload_ref_or = home.harness->kernel().payload_transport_broker().issue_payload_ref(
      artifact_id, payload, tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET, "op-remote-issuer-route");
  REQUIRE(payload_ref_or.ok());

  auto capability_or = front.harness->kernel().payload_transport_broker().resolve_payload_ref_capability(
      front.harness->kernel().worker_directory_cache(),
      *payload_ref_or,
      artifact_id,
      absl::Now(),
      absl::Seconds(10),
      kFrontDaemonId,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-remote-issuer-route");
  REQUIRE(capability_or.ok());
  CHECK(capability_or->serving_capability.mode == tensorcast::daemon::BodyCapabilityResolutionMode::kChunkRpcFallback);
  CHECK_FALSE(capability_or->serving_capability.local);
  CHECK(capability_or->payload_ref == *payload_ref_or);

  auto metadata_or = home.harness->kernel().payload_transport_broker().inspect_payload_ref(
      *payload_ref_or, absl::Now(), /*require_not_expired=*/true);
  REQUIRE(metadata_or.ok());
  REQUIRE(home.harness->kernel()
              .lifecycle_kernel()
              .release_capability(absl::StrCat("payload-ref:", metadata_or->payload_id))
              .ok());

  auto stale_capability_or = front.harness->kernel().payload_transport_broker().resolve_payload_ref_capability(
      front.harness->kernel().worker_directory_cache(),
      *payload_ref_or,
      artifact_id,
      absl::Now(),
      absl::Seconds(10),
      kFrontDaemonId,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-remote-issuer-route");
  REQUIRE_FALSE(stale_capability_or.ok());
  CHECK(stale_capability_or.status().code() == absl::StatusCode::kNotFound);
}

TEST_CASE(
    "payload_ref remote issuer route refreshes a stale daemon address before routing",
    "[daemon][batch][redirect][issuer_route][stale_route]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_payload_ref_refresh_route");
  const auto root_home = make_tmp_dir("home_payload_ref_refresh_route");
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
  gs->upsert_worker(kHomeDaemonId, "127.0.0.1", grpc_port_from_address(home.address), kShardHomeEligibleFlag);

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

  const std::string artifact_id = make_test_byte_artifact_id("route-refresh:blk-4");
  const std::string payload = "refresh-route-payload";
  auto payload_ref_or = home.harness->kernel().payload_transport_broker().issue_payload_ref(
      artifact_id, payload, tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET, "op-refresh-route");
  REQUIRE(payload_ref_or.ok());

  const auto cached_at = absl::Now();
  auto cached_address_or = front.harness->kernel().worker_directory_cache().resolve_daemon_address(
      kHomeDaemonId, cached_at, absl::Seconds(10));
  REQUIRE(cached_address_or.ok());
  REQUIRE(*cached_address_or == home.address);

  const std::string stale_address = home.address;
  home.shutdown_server();
  for (int attempt = 0; attempt < 5; ++attempt) {
    home.start_insecure_server();
    if (home.address != stale_address) {
      break;
    }
    home.shutdown_server();
  }
  REQUIRE(home.address != stale_address);
  gs->upsert_worker(kHomeDaemonId, "127.0.0.1", grpc_port_from_address(home.address), kShardHomeEligibleFlag);

  auto refreshed_capability_or = front.harness->kernel().payload_transport_broker().resolve_payload_ref_capability(
      front.harness->kernel().worker_directory_cache(),
      *payload_ref_or,
      artifact_id,
      cached_at + absl::Seconds(2),
      absl::Seconds(1),
      kFrontDaemonId,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-refresh-route");
  REQUIRE(refreshed_capability_or.ok());
  CHECK(refreshed_capability_or->payload_ref == *payload_ref_or);
  CHECK(
      refreshed_capability_or->serving_capability.mode ==
      tensorcast::daemon::BodyCapabilityResolutionMode::kChunkRpcFallback);
}

TEST_CASE(
    "payload_ref remote issuer route drops a stale daemon address when refresh no longer finds the issuer",
    "[daemon][batch][redirect][issuer_route][stale_route]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_payload_ref_missing_route");
  const auto root_home = make_tmp_dir("home_payload_ref_missing_route");
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
  gs->upsert_worker(kHomeDaemonId, "127.0.0.1", grpc_port_from_address(home.address), kShardHomeEligibleFlag);

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

  const std::string artifact_id = make_test_byte_artifact_id("route-missing:blk-5");
  const std::string payload = "missing-route-payload";
  auto payload_ref_or = home.harness->kernel().payload_transport_broker().issue_payload_ref(
      artifact_id, payload, tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET, "op-missing-route");
  REQUIRE(payload_ref_or.ok());

  const auto cached_at = absl::Now();
  auto cached_address_or = front.harness->kernel().worker_directory_cache().resolve_daemon_address(
      kHomeDaemonId, cached_at, absl::Seconds(10));
  REQUIRE(cached_address_or.ok());

  home.shutdown_server();
  gs->upsert_worker(kHomeDaemonId, "127.0.0.1", /*grpc_port=*/0, kShardHomeEligibleFlag);

  auto missing_route_or = front.harness->kernel().payload_transport_broker().resolve_payload_ref_capability(
      front.harness->kernel().worker_directory_cache(),
      *payload_ref_or,
      artifact_id,
      cached_at + absl::Seconds(2),
      absl::Seconds(1),
      kFrontDaemonId,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-missing-route");
  REQUIRE_FALSE(missing_route_or.ok());
  CHECK(missing_route_or.status().code() == absl::StatusCode::kNotFound);
  CHECK(missing_route_or.status().message() == "daemon endpoint not found");
}

TEST_CASE(
    "payload_ref remote issuer reply fails closed on answered_by mismatch",
    "[daemon][batch][redirect][issuer_route][reply_admission]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_payload_ref_reply_admission");
  auto engine_front = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_front));

  const auto lease_ttl = std::chrono::seconds(5);
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
      /*shard_home_eligible=*/true,
      /*routing_epoch=*/1,
      /*start_server=*/false,
      /*inline_payload_threshold_bytes=*/8);

  RouteAuthorityStageTestService fake_service([&](const auto& req, auto* resp) {
    resp->set_status(tensorcast::daemon::v2::BATCH_ITEM_STATUS_OK);
    auto* reply = resp->mutable_owner_stage_reply();
    reply->mutable_answered_by()->set_authority_kind(tensorcast::daemon::v2::ROUTED_AUTHORITY_KIND_ISSUER_DAEMON);
    reply->mutable_answered_by()->set_authority_id("daemon-wrong");
    reply->set_path_family(req.routed_request().path_family());
    reply->set_stage_ref(req.routed_request().stage_ref());
    reply->set_reply_kind(tensorcast::daemon::v2::ROUTED_OWNER_STAGE_REPLY_KIND_READY_FOR_LOWERING);
  });
  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
  builder.RegisterService(&fake_service);
  auto server = builder.BuildAndStart();
  REQUIRE(server != nullptr);
  REQUIRE(selected_port != 0);
  gs->upsert_worker(kHomeDaemonId, "127.0.0.1", static_cast<uint32_t>(selected_port), kShardHomeEligibleFlag);

  const std::string artifact_id = make_test_byte_artifact_id("reply-admission:blk-6");
  tensorcast::common::v1::PayloadRefScope scope;
  scope.set_payload_id("payload-reply-admission");
  scope.set_artifact_id(artifact_id);
  scope.set_payload_size(5);
  scope.set_digest_alg("sha256");
  scope.set_digest_hex("275876e34cf609db118f3d84b799a790");
  scope.set_direction(tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET);
  scope.set_operation_id("op-reply-admission");
  auto scope_or = tensorcast::common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  REQUIRE(scope_or.ok());
  auto payload_ref_or = front.harness->kernel().capability_tokens()->mint(
      kHomeDaemonId,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PAYLOAD_REF,
      *scope_or,
      static_cast<uint64_t>(absl::ToUnixMillis(absl::Now() + absl::Minutes(1))));
  REQUIRE(payload_ref_or.ok());

  auto capability_or = front.harness->kernel().payload_transport_broker().resolve_payload_ref_capability(
      front.harness->kernel().worker_directory_cache(),
      *payload_ref_or,
      artifact_id,
      absl::Now(),
      absl::Seconds(10),
      kFrontDaemonId,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-reply-admission");
  REQUIRE_FALSE(capability_or.ok());
  CHECK(capability_or.status().code() == absl::StatusCode::kFailedPrecondition);
  CHECK(capability_or.status().message() == "owner_stage_reply answered_by mismatch");

  server->Shutdown();
}

TEST_CASE(
    "payload_ref remote issuer request carries hop auth context and retry_later stays non-terminal",
    "[daemon][batch][redirect][issuer_route][retry]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_payload_ref_retry_later");
  auto engine_front = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_front));

  const auto lease_ttl = std::chrono::seconds(5);
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
      /*shard_home_eligible=*/true,
      /*routing_epoch=*/1,
      /*start_server=*/false,
      /*inline_payload_threshold_bytes=*/8);

  tensorcast::daemon::v2::RouteAuthorityStageRequest captured_request;
  absl::Mutex captured_mu;
  RouteAuthorityStageTestService fake_service([&](const auto& req, auto* resp) {
    {
      absl::MutexLock lock(&captured_mu);
      captured_request = req;
    }
    resp->set_status(tensorcast::daemon::v2::BATCH_ITEM_STATUS_OK);
    auto* reply = resp->mutable_owner_stage_reply();
    reply->mutable_answered_by()->CopyFrom(req.routed_request().authority_ref());
    reply->set_path_family(req.routed_request().path_family());
    reply->set_stage_ref(req.routed_request().stage_ref());
    reply->set_reply_kind(tensorcast::daemon::v2::ROUTED_OWNER_STAGE_REPLY_KIND_RETRY_LATER);
    auto* attachment = reply->mutable_attachment_ref();
    attachment->mutable_authority_ref()->CopyFrom(req.routed_request().authority_ref());
    attachment->set_attachment_kind("operation");
    attachment->set_attachment_id("op-retry-later");
  });
  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
  builder.RegisterService(&fake_service);
  auto server = builder.BuildAndStart();
  REQUIRE(server != nullptr);
  REQUIRE(selected_port != 0);
  gs->upsert_worker(kHomeDaemonId, "127.0.0.1", static_cast<uint32_t>(selected_port), kShardHomeEligibleFlag);

  const std::string artifact_id = make_test_byte_artifact_id("retry-later:blk-6");
  tensorcast::common::v1::PayloadRefScope scope;
  scope.set_payload_id("payload-retry-later");
  scope.set_artifact_id(artifact_id);
  scope.set_payload_size(5);
  scope.set_digest_alg("sha256");
  scope.set_digest_hex("275876e34cf609db118f3d84b799a790");
  scope.set_direction(tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET);
  scope.set_operation_id("op-retry-later");
  auto scope_or = tensorcast::common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  REQUIRE(scope_or.ok());
  auto payload_ref_or = front.harness->kernel().capability_tokens()->mint(
      kHomeDaemonId,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PAYLOAD_REF,
      *scope_or,
      static_cast<uint64_t>(absl::ToUnixMillis(absl::Now() + absl::Minutes(1))));
  REQUIRE(payload_ref_or.ok());

  auto capability_or = front.harness->kernel().payload_transport_broker().resolve_payload_ref_capability(
      front.harness->kernel().worker_directory_cache(),
      *payload_ref_or,
      artifact_id,
      absl::Now(),
      absl::Seconds(10),
      kFrontDaemonId,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-retry-later");
  REQUIRE_FALSE(capability_or.ok());
  CHECK(capability_or.status().code() == absl::StatusCode::kUnavailable);

  {
    absl::MutexLock lock(&captured_mu);
    REQUIRE(captured_request.has_routed_request());
    CHECK(captured_request.routed_request().has_hop_auth_context());
    CHECK(
        captured_request.routed_request().hop_auth_context().auth_class() ==
        tensorcast::daemon::v2::ROUTED_DAEMON_HOP_AUTH_CLASS_DEPLOYMENT_TRUSTED_CHANNEL);
    CHECK(captured_request.routed_request().has_portable_credential_envelope());
    CHECK(
        captured_request.routed_request().portable_credential_envelope().bound_root_request_id() ==
        "payload-ref:payload-retry-later");
    CHECK(
        captured_request.routed_request().portable_credential_envelope().payload_kind() ==
        tensorcast::daemon::v2::ROUTED_DELEGATION_PAYLOAD_KIND_PORTABLE_CREDENTIAL);
    CHECK(captured_request.routed_request().has_forwardable_evidence_envelope());
    CHECK(
        captured_request.routed_request().forwardable_evidence_envelope().payload_kind() ==
        tensorcast::daemon::v2::ROUTED_DELEGATION_PAYLOAD_KIND_FORWARDABLE_EVIDENCE);
    CHECK(captured_request.routed_request().forwarded_claims_size() == 0);
  }

  server->Shutdown();
}

TEST_CASE(
    "payload_ref remote issuer terminal reply surfaces terminal status",
    "[daemon][batch][redirect][issuer_route][terminal]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_payload_ref_terminal");
  auto engine_front = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_front));

  const auto lease_ttl = std::chrono::seconds(5);
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
      /*shard_home_eligible=*/true,
      /*routing_epoch=*/1,
      /*start_server=*/false,
      /*inline_payload_threshold_bytes=*/8);

  RouteAuthorityStageTestService fake_service([&](const auto& req, auto* resp) {
    resp->set_status(tensorcast::daemon::v2::BATCH_ITEM_STATUS_OK);
    auto* reply = resp->mutable_owner_stage_reply();
    reply->mutable_answered_by()->CopyFrom(req.routed_request().authority_ref());
    reply->set_path_family(req.routed_request().path_family());
    reply->set_stage_ref(req.routed_request().stage_ref());
    reply->set_reply_kind(tensorcast::daemon::v2::ROUTED_OWNER_STAGE_REPLY_KIND_TERMINAL);
    auto* terminal = reply->mutable_terminal_projection();
    terminal->set_projection_kind(tensorcast::daemon::v2::ROUTED_TERMINAL_PROJECTION_KIND_SEMANTIC_REJECT);
    terminal->set_status_code("permission_denied");
  });
  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
  builder.RegisterService(&fake_service);
  auto server = builder.BuildAndStart();
  REQUIRE(server != nullptr);
  REQUIRE(selected_port != 0);
  gs->upsert_worker(kHomeDaemonId, "127.0.0.1", static_cast<uint32_t>(selected_port), kShardHomeEligibleFlag);

  const std::string artifact_id = make_test_byte_artifact_id("terminal:blk-7");
  tensorcast::common::v1::PayloadRefScope scope;
  scope.set_payload_id("payload-terminal");
  scope.set_artifact_id(artifact_id);
  scope.set_payload_size(5);
  scope.set_digest_alg("sha256");
  scope.set_digest_hex("275876e34cf609db118f3d84b799a790");
  scope.set_direction(tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET);
  scope.set_operation_id("op-terminal");
  auto scope_or = tensorcast::common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  REQUIRE(scope_or.ok());
  auto payload_ref_or = front.harness->kernel().capability_tokens()->mint(
      kHomeDaemonId,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PAYLOAD_REF,
      *scope_or,
      static_cast<uint64_t>(absl::ToUnixMillis(absl::Now() + absl::Minutes(1))));
  REQUIRE(payload_ref_or.ok());

  auto capability_or = front.harness->kernel().payload_transport_broker().resolve_payload_ref_capability(
      front.harness->kernel().worker_directory_cache(),
      *payload_ref_or,
      artifact_id,
      absl::Now(),
      absl::Seconds(10),
      kFrontDaemonId,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-terminal");
  REQUIRE_FALSE(capability_or.ok());
  CHECK(capability_or.status().code() == absl::StatusCode::kPermissionDenied);

  server->Shutdown();
}

TEST_CASE(
    "payload_ref remote issuer stage failure does not fall back to chunk fetch for authority",
    "[daemon][batch][redirect][issuer_route][fail_closed]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_payload_ref_fail_closed");
  auto engine_front = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_front));

  const auto lease_ttl = std::chrono::seconds(5);
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
      /*shard_home_eligible=*/true,
      /*routing_epoch=*/1,
      /*start_server=*/false,
      /*inline_payload_threshold_bytes=*/8);

  absl::Mutex fetch_mu;
  std::size_t fetch_payload_ref_chunk_calls = 0;
  RouteAuthorityStageTestService fake_service(
      [&](const auto&, auto* resp) {
        resp->set_status(tensorcast::daemon::v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION);
        resp->set_message("issuer admission rejected");
      },
      [&](const auto&, auto* resp) {
        absl::MutexLock lock(&fetch_mu);
        ++fetch_payload_ref_chunk_calls;
        resp->set_status(tensorcast::daemon::v2::BATCH_ITEM_STATUS_OK);
        resp->set_total_size(5);
        resp->set_chunk("wrong");
        resp->set_eof(true);
      });
  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
  builder.RegisterService(&fake_service);
  auto server = builder.BuildAndStart();
  REQUIRE(server != nullptr);
  REQUIRE(selected_port != 0);
  gs->upsert_worker(kHomeDaemonId, "127.0.0.1", static_cast<uint32_t>(selected_port), kShardHomeEligibleFlag);

  const std::string artifact_id = make_test_byte_artifact_id("fail-closed:blk-8");
  tensorcast::common::v1::PayloadRefScope scope;
  scope.set_payload_id("payload-fail-closed");
  scope.set_artifact_id(artifact_id);
  scope.set_payload_size(5);
  scope.set_digest_alg("sha256");
  scope.set_digest_hex("275876e34cf609db118f3d84b799a790");
  scope.set_direction(tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET);
  scope.set_operation_id("op-fail-closed");
  auto scope_or = tensorcast::common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  REQUIRE(scope_or.ok());
  auto payload_ref_or = front.harness->kernel().capability_tokens()->mint(
      kHomeDaemonId,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PAYLOAD_REF,
      *scope_or,
      static_cast<uint64_t>(absl::ToUnixMillis(absl::Now() + absl::Minutes(1))));
  REQUIRE(payload_ref_or.ok());

  auto capability_or = front.harness->kernel().payload_transport_broker().resolve_payload_ref_capability(
      front.harness->kernel().worker_directory_cache(),
      *payload_ref_or,
      artifact_id,
      absl::Now(),
      absl::Seconds(10),
      kFrontDaemonId,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-fail-closed");
  REQUIRE_FALSE(capability_or.ok());
  CHECK(capability_or.status().code() == absl::StatusCode::kFailedPrecondition);
  CHECK(capability_or.status().message() == "issuer admission rejected");

  {
    absl::MutexLock lock(&fetch_mu);
    CHECK(fetch_payload_ref_chunk_calls == 0);
  }

  server->Shutdown();
}

TEST_CASE(
    "payload_ref remote issuer reply fails closed when continuity changes after a shape-valid reply",
    "[daemon][batch][redirect][issuer_route][continuity]") {
  auto gs = std::make_shared<InMemoryShardHomeGlobalStore>();
  gs->connected = true;

  const auto root_front = make_tmp_dir("front_payload_ref_continuity");
  auto engine_front = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root_front));

  const auto lease_ttl = std::chrono::seconds(5);
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
      /*shard_home_eligible=*/true,
      /*routing_epoch=*/1,
      /*start_server=*/false,
      /*inline_payload_threshold_bytes=*/8);

  RouteAuthorityStageTestService fake_service([&](const auto&, auto* resp) {
    gs->upsert_worker(kHomeDaemonId, "127.0.0.1", /*grpc_port=*/1, kShardHomeEligibleFlag);
    resp->set_status(tensorcast::daemon::v2::BATCH_ITEM_STATUS_OK);
    auto* reply = resp->mutable_owner_stage_reply();
    reply->mutable_answered_by()->set_authority_kind(tensorcast::daemon::v2::ROUTED_AUTHORITY_KIND_ISSUER_DAEMON);
    reply->mutable_answered_by()->set_authority_id(kHomeDaemonId);
    reply->set_path_family("immediate_lowering");
    reply->set_stage_ref("issuer_validate");
    reply->set_reply_kind(tensorcast::daemon::v2::ROUTED_OWNER_STAGE_REPLY_KIND_READY_FOR_LOWERING);
    reply->set_resolved_source_capability("{}");
  });
  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
  builder.RegisterService(&fake_service);
  auto server = builder.BuildAndStart();
  REQUIRE(server != nullptr);
  REQUIRE(selected_port != 0);
  gs->upsert_worker(kHomeDaemonId, "127.0.0.1", static_cast<uint32_t>(selected_port), kShardHomeEligibleFlag);

  const std::string artifact_id = make_test_byte_artifact_id("continuity-issuer:blk-3");
  tensorcast::common::v1::PayloadRefScope scope;
  scope.set_payload_id("payload-continuity");
  scope.set_artifact_id(artifact_id);
  scope.set_payload_size(5);
  scope.set_digest_alg("sha256");
  scope.set_digest_hex("275876e34cf609db118f3d84b799a790");
  scope.set_direction(tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET);
  scope.set_operation_id("op-continuity");
  auto scope_or = tensorcast::common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  REQUIRE(scope_or.ok());
  auto payload_ref_or = front.harness->kernel().capability_tokens()->mint(
      kHomeDaemonId,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PAYLOAD_REF,
      *scope_or,
      static_cast<uint64_t>(absl::ToUnixMillis(absl::Now() + absl::Minutes(1))));
  REQUIRE(payload_ref_or.ok());

  auto capability_or = front.harness->kernel().payload_transport_broker().resolve_payload_ref_capability(
      front.harness->kernel().worker_directory_cache(),
      *payload_ref_or,
      artifact_id,
      absl::Now(),
      absl::ZeroDuration(),
      kFrontDaemonId,
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-continuity");
  REQUIRE_FALSE(capability_or.ok());
  CHECK(capability_or.status().code() == absl::StatusCode::kUnavailable);

  server->Shutdown();
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

  const std::string artifact_id = make_test_byte_artifact_id("remote-home-unreachable:blk-9");
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
