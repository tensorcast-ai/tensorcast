// Copyright (c) 2025, TensorCast Team.

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "catch2/catch_test_macros.hpp"
#include "core/store/runtime/artifact_ingress_manager.h"
#include "core/store/runtime/global_metadata_gateway.h"
#include "core/store/runtime/replica_runtime.h"
#include "core/store/runtime/runtime_env.h"
#include "core/store/runtime/runtime_event_hub.h"
#include "core/store/store_engine_options.h"
#include "core/testing/common.h"
#include "core/testing/test_helpers.h"

using tensorcast::DeviceType;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::components::ArtifactRegistration;
using tensorcast::store::runtime::ArtifactIngressManager;
using tensorcast::store::runtime::GlobalMetadataGateway;
using tensorcast::store::runtime::RegistrationEvent;
using tensorcast::store::runtime::ReplicaRuntime;
using tensorcast::store::runtime::RuntimeEnv;
using tensorcast::store::runtime::RuntimeEvent;
using tensorcast::store::runtime::RuntimeEventHub;
using tensorcast::store::runtime::RuntimeEventType;

namespace {

StoreEngineOptions MakeOptions(const std::filesystem::path& storage_root) {
  StoreEngineOptions opts;
  opts.storage_path = storage_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 64ULL * 1024;
  opts.artifact_chunk_bytes = opts.tx_slice_bytes;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  return opts;
}

ArtifactRegistration MakeRegistration(const std::string& artifact_id, uint64_t size_bytes) {
  ArtifactRegistration reg;
  reg.artifact_id = artifact_id;
  reg.tensor_index_key = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  reg.device_id = 0;
  reg.total_size_bytes = size_bytes;
  reg.enable_p2p = false;
  return reg;
}

void EnsureArtifactDirectory(const std::filesystem::path& storage_root, const std::string& artifact_id, size_t bytes) {
  auto artifact_dir = storage_root / artifact_id;
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_0", bytes));
}

} // namespace

TEST_CASE("ArtifactIngressManager publishes registration events", "[artifact_ingress_manager]") {
  SKIP_IF_NO_CUDA();

  const uint64_t kSizeBytes = 2ULL * 1024 * 1024;
  auto temp_root = std::filesystem::temp_directory_path() / "artifact_ingress_manager_event_test";
  std::filesystem::create_directories(temp_root);

  StoreEngineOptions opts = MakeOptions(temp_root);
  RuntimeEnv env(opts);
  CHECK_OK(env.Initialize());

  RuntimeEventHub& hub = env.event_hub();
  auto& catalog = env.component_catalog();
  ReplicaRuntime replica_runtime(ReplicaRuntime::Config{.component_catalog = &catalog, .event_hub = &hub});
  GlobalMetadataGateway metadata_gateway(
      GlobalMetadataGateway::Config{
          .component_catalog = &catalog, .replica_runtime = &replica_runtime, .event_hub = &hub});
  ArtifactIngressManager::Config config{
      .env = &env,
      .replica_runtime = &replica_runtime,
      .metadata_gateway = &metadata_gateway,
      .storage_path = opts.storage_path,
      .artifact_chunk_bytes = opts.artifact_chunk_bytes,
      .pinned_memory_timeout = opts.pinned_memory_timeout,
      .num_threads = opts.num_thread,
      .options = &opts,
  };
  ArtifactIngressManager manager(std::move(config));

  std::vector<RegistrationEvent> events;
  auto subscription = hub.subscribe([&events](const RuntimeEvent& event) {
    if (event.type == RuntimeEventType::kRegistrationCommitted ||
        event.type == RuntimeEventType::kRegistrationAborted) {
      events.push_back(std::get<RegistrationEvent>(event.payload));
    }
  });

  const std::string committed_artifact = "aim_event_commit";
  EnsureArtifactDirectory(temp_root, committed_artifact, static_cast<size_t>(kSizeBytes));
  auto begin_or = manager.begin_registration(MakeRegistration(committed_artifact, kSizeBytes));
  REQUIRE(begin_or.ok());

  auto commit_or = manager.commit_registration(begin_or->registration_id);
  REQUIRE(commit_or.ok());
  REQUIRE_FALSE(events.empty());
  const auto committed_event = events.back();
  CHECK(committed_event.committed);
  CHECK(committed_event.status.ok());
  CHECK(committed_event.registration_id == begin_or->registration_id);
  CHECK(committed_event.artifact_id == commit_or->artifact_id);
  CHECK(committed_event.device.type == DeviceType::GPU);
  CHECK(committed_event.device.ordinal == commit_or->device.ordinal);
  CHECK(committed_event.size_bytes == commit_or->size_bytes);

  const std::string aborted_artifact = "aim_event_abort";
  EnsureArtifactDirectory(temp_root, aborted_artifact, static_cast<size_t>(kSizeBytes));
  auto abort_begin_or = manager.begin_registration(MakeRegistration(aborted_artifact, kSizeBytes));
  REQUIRE(abort_begin_or.ok());

  auto abort_status = manager.abort_registration(abort_begin_or->registration_id);
  REQUIRE(abort_status.ok());

  REQUIRE(events.size() >= 2);
  const auto aborted_event = events.back();
  CHECK_FALSE(aborted_event.committed);
  CHECK(aborted_event.status.ok());
  CHECK(aborted_event.registration_id == abort_begin_or->registration_id);
  CHECK(aborted_event.artifact_id.empty());

  REQUIRE(replica_runtime.clear_mem() == 0);
  env.Shutdown();

  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}
