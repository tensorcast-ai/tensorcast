// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/ingestion/materialization_service.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include "absl/status/status.h"
#include "core/common/async_runtime.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/replica_registry.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/contracts/materialization_request.h"
#include "core/store/replica/replica.h"
#include "gsl/pointers"

using tensorcast::DeviceType;
using tensorcast::store::DeviceKey;
using tensorcast::store::components::DeviceManager;
using tensorcast::store::components::ReplicaRegistry;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::loading::InlineBufferSource;
using tensorcast::store::loading::MaterializationRequest;
using tensorcast::store::loading::MaterializeHints;
using tensorcast::store::loading::MaterializeMode;
using tensorcast::store::loading::ReplicaHandle;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::loading::ReplicaTarget;
using tensorcast::store::runtime::ingestion::MaterializationDeps;
using tensorcast::store::runtime::ingestion::MaterializationService;

namespace {

constexpr size_t kChunkBytes = 4ULL * 1024 * 1024;

DeviceKey MakeCpuKey() {
  return DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
}

DeviceKey MakeGpuKey(int ordinal) {
  return DeviceKey{.type = DeviceType::GPU, .ordinal = ordinal, .uuid = ""};
}

std::shared_ptr<tensorcast::common::ReadySignal<absl::Status>> MakeReadySignal(absl::Status status) {
  auto signal = std::make_shared<tensorcast::common::ReadySignal<absl::Status>>();
  signal->set_value(std::move(status));
  return signal;
}

ReplicaHandle MakeStubHandle(const ReplicaKey& key) {
  ReplicaHandle handle;
  handle.replica_key = key;
  handle.ready_signal = MakeReadySignal(absl::OkStatus());
  return handle;
}

struct TestHarness {
  TestHarness()
      : memory_pool(std::make_shared<tensorcast::common::memory::PinnedBufferPool>(1ULL << 20, 1ULL << 20)),
        async_runtime(std::make_shared<tensorcast::common::AsyncRuntime>()) {}

  MaterializationDeps BuildDeps() {
    MaterializationDeps deps(
        gsl::not_null<ReplicaRegistry*>{&registry},
        gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{memory_pool});
    deps.async_runtime = async_runtime;
    deps.artifact_chunk_bytes = kChunkBytes;
    deps.pinned_memory_timeout = std::chrono::milliseconds{0};
    deps.num_threads = 2;
    deps.run_auto = run_auto;
    deps.ingest_from_disk = ingest_from_disk;
    return deps;
  }

  ReplicaRegistry registry;
  std::shared_ptr<tensorcast::common::memory::PinnedBufferPool> memory_pool;
  std::shared_ptr<tensorcast::common::AsyncRuntime> async_runtime;
  std::function<absl::StatusOr<ReplicaHandle>(const MaterializationRequest&)> run_auto;
  std::function<absl::StatusOr<
      ReplicaHandle>(const std::string&, const DiskSource&, const ReplicaTarget&, const MaterializeHints&)>
      ingest_from_disk;
};

std::shared_ptr<tensorcast::store::replica::Replica> MakeCpuReplica(
    const std::string& artifact_id,
    gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>> pool,
    gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>> async_runtime) {
  InlineBufferSource src{.data = nullptr, .size_bytes = 16};
  tensorcast::store::replica::ReplicaConfig cfg{
      .source = src,
      .artifact_identifier = artifact_id,
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool = pool,
      .async_runtime = async_runtime,
      .artifact_chunk_bytes = kChunkBytes,
      .expected_artifact_size = 16};
  auto replica_or = tensorcast::store::replica::Replica::create(cfg);
  REQUIRE(replica_or.ok());
  return std::shared_ptr<tensorcast::store::replica::Replica>(std::move(replica_or.value()));
}

MaterializeHints MakeHints(std::string artifact_id) {
  MaterializeHints hints;
  hints.artifact_id = std::move(artifact_id);
  hints.max_buffer_bytes = 16ULL << 20;
  return hints;
}

} // namespace

TEST_CASE("MaterializationService reuses resident replicas", "[materialization_service]") {
  TestHarness harness;
  harness.run_auto = nullptr;
  harness.ingest_from_disk = [](const std::string&, const DiskSource&, const ReplicaTarget&, const MaterializeHints&) {
    return absl::UnimplementedError("ingest not used");
  };

  DeviceManager device_manager;
  MaterializeHints hints = MakeHints("cgid:artifact-resident");
  auto request_or = MaterializationRequest::Create(MakeCpuKey(), MaterializeMode::AUTO, hints, device_manager);
  REQUIRE(request_or.ok());
  const auto request = request_or.value();

  auto replica = MakeCpuReplica(
      request.canonical_artifact_id(),
      gsl::not_null<std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>>{harness.memory_pool},
      gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{harness.async_runtime});
  auto status = harness.registry.emplace(request.replica_key(), gsl::not_null{replica});
  REQUIRE(status.ok());

  MaterializationService service(harness.BuildDeps());
  auto result = service.execute(request);
  REQUIRE(result.ok());
  REQUIRE(result->replica_key.artifact_id == request.canonical_artifact_id());
}

TEST_CASE("MaterializationService COPY_ONLY fails without GPU sources", "[materialization_service]") {
  TestHarness harness;
  harness.ingest_from_disk = [](const std::string&, const DiskSource&, const ReplicaTarget&, const MaterializeHints&) {
    return absl::UnimplementedError("ingest not used");
  };

  DeviceManager device_manager;
  device_manager.set_num_gpus_for_testing(2);
  MaterializeHints hints = MakeHints("cgid:artifact-copy");
  auto request_or = MaterializationRequest::Create(MakeGpuKey(0), MaterializeMode::COPY_ONLY, hints, device_manager);
  REQUIRE(request_or.ok());

  MaterializationService service(harness.BuildDeps());
  auto result = service.execute(request_or.value());
  REQUIRE_FALSE(result.ok());
  REQUIRE(result.status().code() == absl::StatusCode::kFailedPrecondition);
}

TEST_CASE("MaterializationService proxies disk ingestion", "[materialization_service]") {
  TestHarness harness;
  DeviceManager device_manager;
  MaterializeHints hints = MakeHints("cgid:artifact-disk");
  DiskSource disk_source{.path = std::filesystem::path("/tmp/model"), .expected_size = std::nullopt};
  auto request_or =
      MaterializationRequest::Create(MakeCpuKey(), MaterializeMode::LOAD_ONLY, hints, device_manager, disk_source);
  REQUIRE(request_or.ok());
  const auto request = request_or.value();

  bool invoked = false;
  harness.ingest_from_disk = [&](const std::string& artifact,
                                 const DiskSource& source,
                                 const ReplicaTarget& target,
                                 const MaterializeHints&) -> absl::StatusOr<ReplicaHandle> {
    invoked = true;
    REQUIRE(artifact == request.canonical_artifact_id());
    REQUIRE(source.path == "/tmp/model");
    REQUIRE(target.location.type == tensorcast::common::memory::MemoryLocation::CPU);
    return MakeStubHandle(request.replica_key());
  };

  MaterializationService service(harness.BuildDeps());
  auto result = service.execute(request);
  REQUIRE(result.ok());
  REQUIRE(invoked);
}

TEST_CASE("MaterializationService AUTO uses injected orchestrator", "[materialization_service]") {
  TestHarness harness;
  auto ready_key =
      ReplicaKey{.artifact_id = "cgid:artifact-auto", .view_id = std::nullopt, .device = MakeCpuKey(), .replica = 0};
  harness.run_auto = [ready_key](const MaterializationRequest&) { return MakeStubHandle(ready_key); };
  harness.ingest_from_disk = [](const std::string&, const DiskSource&, const ReplicaTarget&, const MaterializeHints&) {
    return absl::UnimplementedError("ingest not used");
  };

  DeviceManager device_manager;
  MaterializeHints hints = MakeHints("cgid:artifact-auto");
  auto request_or = MaterializationRequest::Create(MakeCpuKey(), MaterializeMode::AUTO, hints, device_manager);
  REQUIRE(request_or.ok());

  MaterializationService service(harness.BuildDeps());
  auto result = service.execute(request_or.value());
  REQUIRE(result.ok());
  REQUIRE(result->replica_key.artifact_id == ready_key.artifact_id);

  harness.run_auto = [](const MaterializationRequest&) { return absl::AbortedError("downstream failure"); };
  MaterializationService failing_service(harness.BuildDeps());
  auto failing = failing_service.execute(request_or.value());
  REQUIRE_FALSE(failing.ok());
  REQUIRE(failing.status().code() == absl::StatusCode::kAborted);
}
