// Copyright (c) 2025, TensorCast Team.

#include "core/store/components/registration/artifact_registration_manager.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/memory_location.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/replica/replica.h"
#include "gsl/pointers"
#include "nlohmann/json.hpp"

namespace tensorcast::store::components {
namespace {

using tensorcast::common::memory::PinnedBufferPool;
using tensorcast::store::loader::BidirectionalViewPlan;
using tensorcast::store::loader::NarrowOp;
using tensorcast::store::loader::TensorViewOps;
using tensorcast::store::loader::ViewOp;
using tensorcast::store::loader::ViewPlanner;
using tensorcast::store::loader::ViewSpec;

nlohmann::json TensorEntry(
    uint64_t offset,
    uint64_t size,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& stride,
    const std::string& dtype,
    uint64_t storage_offset) {
  nlohmann::json arr = nlohmann::json::array();
  arr.push_back(offset);
  arr.push_back(size);
  nlohmann::json j_shape = nlohmann::json::array();
  for (auto v : shape) {
    j_shape.push_back(v);
  }
  nlohmann::json j_stride = nlohmann::json::array();
  for (auto v : stride) {
    j_stride.push_back(v);
  }
  arr.push_back(j_shape);
  arr.push_back(j_stride);
  arr.push_back(dtype);
  arr.push_back(storage_offset);
  return arr;
}

std::string BuildCanonicalIndexJson(absl::string_view tensor_name, uint64_t element_count) {
  nlohmann::json index = nlohmann::json::object();
  const uint64_t total_bytes = element_count * sizeof(float);
  index[std::string(tensor_name)] =
      TensorEntry(/*offset=*/0, total_bytes, {static_cast<int64_t>(element_count)}, {1}, "torch.float32", 0);
  return index.dump();
}

ArtifactRegistration MakeBaseRegistration(
    std::string suffix,
    const std::string& canonical_index,
    uint64_t total_bytes) {
  ArtifactRegistration reg;
  reg.artifact_id = "artifact-" + suffix;
  reg.tensor_index_key = "index-" + suffix;
  reg.tensor_index_data = canonical_index;
  reg.device_id = 0;
  reg.total_size_bytes = total_bytes;
  reg.schema_version = "v3";
  reg.encoding = "json";
  reg.enable_p2p = false;
  return reg;
}

ViewSpec MakeNarrowSpec(absl::string_view tensor_name, int start, int length) {
  ViewSpec spec;
  TensorViewOps ops;
  ops.ops.push_back(ViewOp::Narrow(NarrowOp{.dim = 0, .start = start, .length = static_cast<uint64_t>(length)}));
  spec.tensors.emplace(std::string(tensor_name), ops);
  return spec;
}

class RegistrationManagerTestHarness {
 public:
  RegistrationManagerTestHarness()
      : device_manager_(std::make_unique<DeviceManager>()),
        replica_registry_(std::make_unique<ReplicaRegistry>()),
        metrics_collector_(std::make_unique<MetricsCollector>()),
        memory_pool_(std::make_shared<PinnedBufferPool>(kPoolBytes, kSliceBytes)) {
    auto status = device_manager_->initialize();
    REQUIRE(status.ok());

    RegistrationResources resources{
        .device_manager = gsl::not_null<DeviceManager*>{device_manager_.get()},
        .replica_registry = gsl::not_null<ReplicaRegistry*>{replica_registry_.get()},
        .metrics_collector = gsl::not_null<MetricsCollector*>{metrics_collector_.get()},
        .memory_pool = gsl::not_null<std::shared_ptr<PinnedBufferPool>>{memory_pool_},
        .communication_manager = nullptr,
        .global_store_client = nullptr};

    ReplicaFactory factory =
        [this](const replica::ReplicaConfig& config) -> absl::StatusOr<std::shared_ptr<replica::Replica>> {
      auto created_or = replica::Replica::create(config);
      if (!created_or.ok()) {
        return created_or.status();
      }
      auto shared = std::shared_ptr<replica::Replica>(std::move(created_or.value()));
      last_replica_ = shared;
      return shared;
    };

    manager_ = std::make_unique<ArtifactRegistrationManager>(
        std::move(resources), std::move(factory), kArtifactChunkBytes, std::chrono::milliseconds(10));
    manager_->set_worker_identity(
        WorkerIdentity{
            .worker_id = "worker-test",
            .node_id = "node-1",
            .node_address = "127.0.0.1",
            .grpc_port = 50051,
            .p2p_port = 50052});
  }

  ArtifactRegistrationManager& manager() {
    return *manager_;
  }

  void fill_last_replica_gpu(uint8_t pattern, size_t num_bytes) {
    REQUIRE(last_replica_);
    auto ptrs = last_replica_->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
    REQUIRE_FALSE(ptrs.empty());
    auto set_status = cuda::set_device(last_replica_->device().ordinal);
    REQUIRE(set_status.ok());
    auto status = cuda::memset(ptrs[0], pattern, num_bytes);
    REQUIRE(status.ok());
    auto sync_status = cuda::device_synchronize();
    REQUIRE(sync_status.ok());
  }

 private:
  static constexpr size_t kPoolBytes = 8ULL * 1024 * 1024;
  static constexpr size_t kSliceBytes = 1ULL * 1024 * 1024;
  static constexpr size_t kArtifactChunkBytes = 1ULL << 20;

  std::unique_ptr<DeviceManager> device_manager_;
  std::unique_ptr<ReplicaRegistry> replica_registry_;
  std::unique_ptr<MetricsCollector> metrics_collector_;
  std::shared_ptr<PinnedBufferPool> memory_pool_;
  std::unique_ptr<ArtifactRegistrationManager> manager_;
  std::shared_ptr<replica::Replica> last_replica_;
};

TEST_CASE("ArtifactRegistrationManager ingests server view and computes hashes", "[registration][view]") {
  RegistrationManagerTestHarness harness;
  const std::string canonical_json = BuildCanonicalIndexJson("weights", /*element_count=*/8);
  auto spec = MakeNarrowSpec("weights", /*start=*/2, /*length=*/4);
  auto plan_or = ViewPlanner::compute_bidirectional_view_plan(canonical_json, spec);
  REQUIRE(plan_or.ok());
  const BidirectionalViewPlan& plan = *plan_or;

  auto reg = MakeBaseRegistration("view", canonical_json, /*total_bytes=*/32);
  ViewRegistration view;
  view.view_id = "view-weights";
  view.spec = spec;
  view.placement = ViewPlacement::kServer;
  view.canonical_size_bytes = reg.total_size_bytes;
  view.allow_partial = true;
  reg.view = view;

  auto begin_or = harness.manager().begin(reg);
  REQUIRE(begin_or.ok());
  const auto begin = *begin_or;

  std::array<float, 4> payload = {1.0f, 2.0f, 3.0f, 4.0f};
  absl::Span<const std::byte> chunk{reinterpret_cast<const std::byte*>(payload.data()), payload.size() * sizeof(float)};
  auto ingest_status = harness.manager().ingest_view_chunk(begin.registration_id, /*view_offset=*/0, chunk);
  REQUIRE(ingest_status.ok());
  auto ingested_or = harness.manager().get_view_ingested_bytes(begin.registration_id);
  REQUIRE(ingested_or.ok());
  CHECK(*ingested_or == chunk.size());

  auto commit_or = harness.manager().commit(begin.registration_id);
  REQUIRE(commit_or.ok());
  const auto& commit = *commit_or;

  auto index_hash_or = common::compute_index_multihash(reg.tensor_index_data, reg.tensor_index_key);
  REQUIRE(index_hash_or.ok());

  std::vector<std::byte> canonical_bytes(reg.total_size_bytes, std::byte{0});
  const uint64_t canonical_offset = plan.write.chunks.front().canonical_offset;
  std::memcpy(canonical_bytes.data() + canonical_offset, payload.data(), chunk.size());
  auto data_hash_or = tensorcast::store::loader::compute_data_multihash_from_cpu_memory(
      gsl::not_null<const void*>{canonical_bytes.data()}, canonical_bytes.size());
  REQUIRE(data_hash_or.ok());

  CHECK(commit.registration_id == begin.registration_id);
  CHECK(commit.size_bytes == reg.total_size_bytes);
  CHECK(commit.index_multihash == *index_hash_or);
  CHECK(commit.data_multihash == *data_hash_or);
  REQUIRE(commit.view_id.has_value());
  CHECK(*commit.view_id == view.view_id);
  CHECK(commit.allow_partial);
  REQUIRE(commit.canonical_ranges.size() == 1);
  CHECK(commit.canonical_ranges.front().offset == canonical_offset);
  CHECK(commit.canonical_ranges.front().length == chunk.size());
  CHECK(commit.view_index_json.has_value());
  CHECK(commit.view_data_multihash.has_value());
}

TEST_CASE("ArtifactRegistrationManager enforces TTL and honors keep-alive", "[registration][ttl]") {
  RegistrationManagerTestHarness harness;
  const std::string canonical_json = BuildCanonicalIndexJson("ttl_tensor", /*element_count=*/4);

  auto ttl_reg = MakeBaseRegistration("ttl", canonical_json, /*total_bytes=*/16);
  ttl_reg.ttl_ms = 5;
  auto begin_or = harness.manager().begin(ttl_reg);
  REQUIRE(begin_or.ok());
  absl::SleepFor(absl::Milliseconds(25));
  auto expired_or = harness.manager().commit(begin_or->registration_id);
  REQUIRE_FALSE(expired_or.ok());
  CHECK(absl::IsDeadlineExceeded(expired_or.status()));

  auto keep_reg = MakeBaseRegistration("keep", canonical_json, /*total_bytes=*/16);
  keep_reg.ttl_ms = 10;
  auto keep_begin_or = harness.manager().begin(keep_reg);
  REQUIRE(keep_begin_or.ok());
  absl::SleepFor(absl::Milliseconds(5));
  auto keep_status = harness.manager().keep_alive(keep_begin_or->registration_id, /*ttl_ms=*/50);
  REQUIRE(keep_status.ok());
  absl::SleepFor(absl::Milliseconds(20));
  auto keep_commit_or = harness.manager().commit(keep_begin_or->registration_id);
  REQUIRE(keep_commit_or.ok());
}

TEST_CASE("ArtifactRegistrationManager reports existed for duplicate commits", "[registration][dedupe]") {
  RegistrationManagerTestHarness harness;
  const std::string canonical_json = BuildCanonicalIndexJson("dup_tensor", /*element_count=*/4);
  auto reg = MakeBaseRegistration("dedupe", canonical_json, /*total_bytes=*/16);

  auto begin1 = harness.manager().begin(reg);
  REQUIRE(begin1.ok());
  harness.fill_last_replica_gpu(/*pattern=*/0x3C, reg.total_size_bytes);
  auto commit1 = harness.manager().commit(begin1->registration_id);
  REQUIRE(commit1.ok());
  CHECK_FALSE(commit1->existed);

  auto begin2 = harness.manager().begin(reg);
  REQUIRE(begin2.ok());
  harness.fill_last_replica_gpu(/*pattern=*/0x3C, reg.total_size_bytes);
  auto commit2 = harness.manager().commit(begin2->registration_id);
  REQUIRE(commit2.ok());
  CHECK(commit2->existed);
  CHECK(commit2->artifact_id == commit1->artifact_id);
  CHECK(commit2->index_multihash == commit1->index_multihash);
  CHECK(commit2->data_multihash == commit1->data_multihash);
}

TEST_CASE("ArtifactRegistrationManager abort removes pending context", "[registration][abort]") {
  RegistrationManagerTestHarness harness;
  const std::string canonical_json = BuildCanonicalIndexJson("abort_tensor", /*element_count=*/2);
  auto reg = MakeBaseRegistration("abort", canonical_json, /*total_bytes=*/8);

  auto begin_or = harness.manager().begin(reg);
  REQUIRE(begin_or.ok());
  auto abort_status = harness.manager().abort(begin_or->registration_id);
  REQUIRE(abort_status.ok());

  auto commit_or = harness.manager().commit(begin_or->registration_id);
  REQUIRE_FALSE(commit_or.ok());
  CHECK(absl::IsNotFound(commit_or.status()));

  auto keep_status = harness.manager().keep_alive(begin_or->registration_id, /*ttl_ms=*/10);
  CHECK(absl::IsNotFound(keep_status));

  auto bytes_or = harness.manager().get_view_ingested_bytes(begin_or->registration_id);
  CHECK_FALSE(bytes_or.ok());
  CHECK(absl::IsNotFound(bytes_or.status()));
}

} // namespace
} // namespace tensorcast::store::components
