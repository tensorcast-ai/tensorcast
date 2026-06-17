// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/components/replica_registry.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"
#include "gsl/pointers"

using tensorcast::DeviceType;
using tensorcast::store::DeviceKey;
using tensorcast::store::components::ReplicaRegistry;
using tensorcast::store::loading::InlineBufferSource;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::replica::Replica;
using tensorcast::store::replica::ReplicaConfig;

namespace {

constexpr size_t kChunkBytes = 64;

std::shared_ptr<tensorcast::common::memory::PinnedBufferPool> MakeTestPinnedBufferPool() {
  tensorcast::common::memory::PinnedBufferPool::Options options;
  options.register_on_create = false;
  return std::make_shared<tensorcast::common::memory::PinnedBufferPool>(1 << 20, 1 << 20, std::move(options));
}

std::shared_ptr<Replica> MakeReplica(
    const std::string& artifact_id,
    gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>> pool,
    gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>> runtime) {
  InlineBufferSource src{.data = nullptr, .size_bytes = kChunkBytes};
  ReplicaConfig cfg{
      .source = src,
      .artifact_identifier = artifact_id,
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool = pool,
      .async_runtime = runtime,
      .artifact_chunk_bytes = kChunkBytes,
      .expected_artifact_size = kChunkBytes,
  };
  auto replica_or = Replica::create(cfg);
  REQUIRE(replica_or.ok());
  return std::shared_ptr<Replica>(std::move(replica_or.value()));
}

ReplicaKey MakeKey(
    std::string artifact_id,
    DeviceType device_type,
    int32_t ordinal,
    std::string uuid,
    uint32_t replica_ordinal) {
  return ReplicaKey{
      .artifact_id = std::move(artifact_id),
      .view_id = std::nullopt,
      .device = DeviceKey{.type = device_type, .ordinal = ordinal, .uuid = std::move(uuid)},
      .replica = replica_ordinal,
  };
}

std::vector<std::string> CanonicalizeKeys(std::vector<ReplicaKey> keys) {
  std::vector<std::string> rendered;
  rendered.reserve(keys.size());
  for (const auto& key : keys) {
    rendered.push_back(
        key.artifact_id + "|" + std::to_string(static_cast<int>(key.device.type)) + "|" +
        std::to_string(key.device.ordinal) + "|" + key.device.uuid + "|" + std::to_string(key.replica));
  }
  std::ranges::sort(rendered);
  return rendered;
}

void RequireKeySet(std::vector<ReplicaKey> actual, std::vector<ReplicaKey> expected) {
  REQUIRE(CanonicalizeKeys(std::move(actual)) == CanonicalizeKeys(std::move(expected)));
}

} // namespace

TEST_CASE("ReplicaRegistry erase keeps secondary indices consistent", "[replica_registry]") {
  auto pool = MakeTestPinnedBufferPool();
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  ReplicaRegistry registry;

  auto replica_a = MakeReplica(
      "artifact-a",
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime});
  auto replica_b = MakeReplica(
      "artifact-b",
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime});
  auto replica_c = MakeReplica(
      "artifact-c",
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime});
  auto replica_d = MakeReplica(
      "artifact-d",
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime});

  const ReplicaKey key_a0 = MakeKey("artifact-a", DeviceType::CPU, -1, "", 0);
  const ReplicaKey key_shared0 = MakeKey("artifact-shared", DeviceType::GPU, 0, "gpu-0", 0);
  const ReplicaKey key_c0 = MakeKey("artifact-c", DeviceType::GPU, 1, "gpu-1", 0);
  const ReplicaKey key_shared1 = MakeKey("artifact-shared", DeviceType::GPU, 0, "gpu-0", 1);

  REQUIRE(registry.emplace(key_a0, gsl::not_null<std::shared_ptr<Replica>>{replica_a}).ok());
  REQUIRE(registry.emplace(key_shared0, gsl::not_null<std::shared_ptr<Replica>>{replica_b}).ok());
  REQUIRE(registry.emplace(key_c0, gsl::not_null<std::shared_ptr<Replica>>{replica_c}).ok());
  REQUIRE(registry.emplace(key_shared1, gsl::not_null<std::shared_ptr<Replica>>{replica_d}).ok());

  SECTION("erase middle entry updates moved entry buckets") {
    auto removed = registry.erase(key_shared0);
    REQUIRE(removed.has_value());
    REQUIRE(removed->first == key_shared0);
    REQUIRE(registry.size() == 3);
    REQUIRE_FALSE(registry.find(key_shared0).ok());

    RequireKeySet(registry.find_by_artifact("artifact-shared"), {key_shared1});
    RequireKeySet(registry.find_by_artifact("artifact-a"), {key_a0});
    RequireKeySet(registry.find_by_artifact("artifact-c"), {key_c0});
    RequireKeySet(registry.find_by_device(key_shared0.device), {key_shared1});
    RequireKeySet(registry.find_by_device(key_a0.device), {key_a0});
    RequireKeySet(registry.find_by_device(key_c0.device), {key_c0});
    REQUIRE(registry.find(key_shared1).ok());
  }

  SECTION("erase head entry preserves artifact and device views") {
    auto removed = registry.erase(key_a0);
    REQUIRE(removed.has_value());
    REQUIRE(removed->first == key_a0);
    REQUIRE(registry.size() == 3);
    REQUIRE_FALSE(registry.find(key_a0).ok());

    RequireKeySet(registry.find_by_artifact("artifact-shared"), {key_shared0, key_shared1});
    RequireKeySet(registry.find_by_device(key_shared0.device), {key_shared0, key_shared1});
    RequireKeySet(registry.find_by_device(key_c0.device), {key_c0});
  }

  SECTION("erase tail entry and emplace again keep indices usable") {
    auto removed = registry.erase(key_shared1);
    REQUIRE(removed.has_value());
    REQUIRE(removed->first == key_shared1);
    REQUIRE(registry.size() == 3);
    REQUIRE_FALSE(registry.find(key_shared1).ok());

    const ReplicaKey key_shared2 = MakeKey("artifact-shared", DeviceType::GPU, 0, "gpu-0", 2);
    auto replica_e = MakeReplica(
        "artifact-e",
        gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
        gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime});
    REQUIRE(registry.emplace(key_shared2, gsl::not_null<std::shared_ptr<Replica>>{replica_e}).ok());

    RequireKeySet(registry.find_by_artifact("artifact-shared"), {key_shared0, key_shared2});
    RequireKeySet(registry.find_by_device(key_shared0.device), {key_shared0, key_shared2});
    REQUIRE(registry.find(key_shared2).ok());
  }
}

TEST_CASE("ReplicaRegistry LRU ordering is independent of entry compaction", "[replica_registry]") {
  auto pool = MakeTestPinnedBufferPool();
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  ReplicaRegistry registry;

  auto replica0 = MakeReplica(
      "artifact-0",
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime});
  auto replica1 = MakeReplica(
      "artifact-1",
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime});
  auto replica2 = MakeReplica(
      "artifact-2",
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime});
  auto replica3 = MakeReplica(
      "artifact-3",
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{runtime});

  const ReplicaKey key0 = MakeKey("artifact-0", DeviceType::CPU, -1, "", 0);
  const ReplicaKey key1 = MakeKey("artifact-1", DeviceType::GPU, 0, "gpu-0", 0);
  const ReplicaKey key2 = MakeKey("artifact-2", DeviceType::GPU, 1, "gpu-1", 0);
  const ReplicaKey key3 = MakeKey("artifact-3", DeviceType::GPU, 2, "gpu-2", 0);

  REQUIRE(registry.emplace(key0, gsl::not_null<std::shared_ptr<Replica>>{replica0}).ok());
  absl::SleepFor(absl::Milliseconds(2));
  REQUIRE(registry.emplace(key1, gsl::not_null<std::shared_ptr<Replica>>{replica1}).ok());
  absl::SleepFor(absl::Milliseconds(2));
  REQUIRE(registry.emplace(key2, gsl::not_null<std::shared_ptr<Replica>>{replica2}).ok());
  absl::SleepFor(absl::Milliseconds(2));
  REQUIRE(registry.emplace(key3, gsl::not_null<std::shared_ptr<Replica>>{replica3}).ok());

  absl::SleepFor(absl::Milliseconds(2));
  REQUIRE(registry.find(key2).ok());
  absl::SleepFor(absl::Milliseconds(2));
  REQUIRE(registry.find(key3).ok());

  auto removed = registry.erase(key1);
  REQUIRE(removed.has_value());

  const auto lru = registry.get_lru_instances();
  REQUIRE(lru.size() == 3);
  REQUIRE(lru[0] == key0);
  REQUIRE(lru[1] == key2);
  REQUIRE(lru[2] == key3);
}
