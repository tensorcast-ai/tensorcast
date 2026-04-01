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
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_ipc.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/dataplane/contracts/inline_buffer_loader.h"
#include "core/store/store_engine.h"
#include "core/store/testing/recording_global_store_client.h"
#include "daemon/service/artifact_profile_registry.h"
#include "daemon/service/body_backing_manager.h"
#include "daemon/state/routed_authority_wire.h"
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
    const DaemonOptions& options) {
  auto harness_or = DaemonServiceHarness::create(engine, options);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
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

  const std::string byte_artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azQ";
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

} // namespace

TEST_CASE("HomeBatch* put/get/exists support join and conflict", "[daemon][batch][home]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azE";
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
  REQUIRE(conflict_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_FAILED_PRECONDITION);
  REQUIRE(count_cpu_replicas(*engine) == 1);

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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azI";
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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azM";
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
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azV";
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

TEST_CASE("HomeBatchGet emits batch transport for large payloads", "[daemon][batch][batch_payload_ref][get]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  options.byte_artifact_routing.payload_transport.max_batch_payload_bytes = 1ULL << 20;
  options.byte_artifact_routing.payload_transport.max_batch_items = 8;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id_a = "cgid:byte_artifact~tenant~engine~b64u.YmF0Y2gtZ2V0LWE~layout_v1~b64u.azVh";
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

  const std::string artifact_id_a = "cgid:byte_artifact~tenant~engine~b64u.YmF0Y2gtcHV0LWE~layout_v1~b64u.azZh";
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

TEST_CASE("Local payload_ref resolves to reusable body capability", "[daemon][batch][payload_ref][reuse]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azY";
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
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.Z2VuZXJhdGlvbg~layout_v1~b64u.azc";
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
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.Z2V0LXNuYXBzaG90~layout_v1~b64u.azg";
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

TEST_CASE(
    "HomeBatchGet payload_ref capability is bounded by authority expiry",
    "[daemon][batch][payload_ref][expiry]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto options = make_daemon_options();
  options.byte_artifact_routing.inline_payload_threshold_bytes = 8;
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.ZXhwaXJ5~layout_v1~b64u.azY";
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
  REQUIRE_FALSE(get_resp.items(0).payload_ref().empty());

  auto metadata_or = harness->kernel().payload_transport_broker().inspect_payload_ref(
      get_resp.items(0).payload_ref(), absl::Now(), /*require_not_expired=*/true);
  REQUIRE(metadata_or.ok());
  CHECK(metadata_or->expires_at <= entry->expires_at);
}

TEST_CASE("payload_ref front-door context preserves raw credential evidence", "[daemon][batch][payload_ref][context]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.Y29udGV4dA~layout_v1~b64u.azA";
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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.cm91dGU~layout_v1~b64u.azQ";
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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.dHJhbnNsYXRl~layout_v1~b64u.azU";
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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.cm91dGUtZGVjbGFyZWQ~layout_v1~b64u.azk";
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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.aXNzdWVyLXJvdXRl~layout_v1~b64u.azE";
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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.Y2xhaW0~layout_v1~b64u.azk";
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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.cG9saWN5LXJlc3RvcmU~layout_v1~b64u.azEy";
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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.c291cmNlLXRydXRo~layout_v1~b64u.azE1";
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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.cG9saWN5LW1pc3Npbmc~layout_v1~b64u.azEz";
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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.cG9saWN5LWRlbGV0ZWQ~layout_v1~b64u.azE0";
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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.dHRsLXJlY3JlYXRl~layout_v1~b64u.azEw";
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
  auto harness = make_harness(engine, options);
  auto& svc = harness->service();

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.c3Vydml2ZS1yZXRpcmU~layout_v1~b64u.azEx";
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
  REQUIRE_FALSE(get_resp.items(0).payload_ref().empty());

  REQUIRE(engine->clear_mem() == 0);

  HomeBatchExistsRequest exists_req;
  exists_req.mutable_fence()->CopyFrom(put_req.fence());
  exists_req.add_artifact_ids(artifact_id);
  HomeBatchExistsResponse exists_resp;
  grpc::ServerContext exists_ctx;
  REQUIRE(svc.HomeBatchExists(&exists_ctx, &exists_req, &exists_resp).ok());
  REQUIRE(exists_resp.outcomes_size() == 1);
  REQUIRE(exists_resp.outcomes(0).status() == BatchItemStatus::BATCH_ITEM_STATUS_MISS);

  auto payload_or = harness->kernel().payload_transport_broker().resolve_local_payload_ref(
      get_resp.items(0).payload_ref(),
      artifact_id,
      absl::Now(),
      tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
      "op-survive-retire");
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

  const std::string retained_artifact_id = "cgid:byte_artifact~tenant~engine~b64u.cmV0YWluZWQ~layout_v1~b64u.azQ";
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

  const std::string transient_artifact_id = "cgid:byte_artifact~tenant~engine~b64u.dHJhbnNpZW50~layout_v1~b64u.azQ";
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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.ZmFzdA~layout_v1~b64u.azQ";
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

TEST_CASE("HomeBatchTouchTtl keeps immortal entries immortal", "[daemon][batch][ttl]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azQ";
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

  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azc";
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
