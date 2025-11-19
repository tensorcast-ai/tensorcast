// Copyright (c) 2025, TensorCast Team.

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "catch2/catch_test_macros.hpp"
#include "core/store/runtime/ingestion/ingestion_runtime.h"
#include "core/store/runtime/ingestion/testing/fake_ingestion_pipeline.h"
#include "core/store/runtime/ingestion/testing/recording_event_sink.h"
#include "core/store/runtime/ingestion/testing/scoped_ingestion_runtime_test_harness.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/store_engine_options.h"
#include "core/testing/common.h"
#include "core/testing/test_helpers.h"

using tensorcast::DeviceType;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::P2PSource;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::runtime::IngestionResultEvent;
using tensorcast::store::runtime::IngestionRuntime;
using tensorcast::store::runtime::IngestionRuntimeDependencies;
using tensorcast::store::runtime::IngestionSource;
using tensorcast::store::runtime::RuntimeContext;
using tensorcast::store::runtime::RuntimeEvent;
using tensorcast::store::runtime::RuntimeEventType;
namespace ingestion_testing = tensorcast::store::runtime::ingestion::testing;

namespace loading = tensorcast::store::loading;

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

struct CapturedIngestionEvent {
  RuntimeEventType type;
  IngestionResultEvent payload;
};

class RecordingCoordinator final : public tensorcast::store::runtime::ingestion::MaterializationCoordinator {
 public:
  explicit RecordingCoordinator(const Config& config)
      : tensorcast::store::runtime::ingestion::MaterializationCoordinator(config) {}

  absl::Status register_replica_with_global_store(
      const loading::ReplicaKey& key,
      std::string_view artifact_id_override,
      std::string_view publish_context_id) override {
    ++register_calls;
    last_key = key;
    last_publish_context_id = std::string(publish_context_id);
    last_artifact_override = std::string(artifact_id_override);
    return absl::OkStatus();
  }

  int register_calls{0};
  std::optional<loading::ReplicaKey> last_key;
  std::string last_publish_context_id;
  std::string last_artifact_override;
};

} // namespace

TEST_CASE("IngestionRuntime emits disk ingestion lifecycle events via injectable dependencies", "[ingestion_runtime]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "ingestion_runtime_disk_events";
  std::filesystem::create_directories(temp_root);

  StoreEngineOptions opts = MakeOptions(temp_root);
  ingestion_testing::ScopedIngestionRuntimeTestHarness harness(opts);
  CHECK_OK(harness.initialize());

  auto recording_sink = std::make_shared<ingestion_testing::RecordingEventSink>();
  auto dependencies = std::make_shared<IngestionRuntimeDependencies>();
  dependencies->event_sink_override = recording_sink;
  ingestion_testing::FakeIngestionPipeline* fake_pipeline = nullptr;
  dependencies->pipeline_factory = [&](const auto& cfg) {
    auto fake = std::make_unique<ingestion_testing::FakeIngestionPipeline>(cfg);
    fake_pipeline = fake.get();
    return fake;
  };

  IngestionRuntime manager(harness.make_runtime_config(dependencies));
  REQUIRE(fake_pipeline != nullptr);

  loading::DiskSource disk_source;
  disk_source.path = temp_root / "missing_artifact";

  loading::ReplicaTarget target;
  target.location.type = MemoryLocation::CPU;
  target.location.device_id = 0;

  loading::MaterializeHints hints;

  SECTION("disk ingestion failure publishes failure event") {
    static_cast<void>(recording_sink->drain());
    fake_pipeline->set_next_disk_result(absl::NotFoundError("disk_missing_stub"));
    auto handle_or = manager.ingest_from_disk("disk_missing", disk_source, target, hints);
    REQUIRE_FALSE(handle_or.ok());
    const absl::Status& failure_status = handle_or.status();

    auto events = recording_sink->drain();
    REQUIRE(events.size() == 2);
    const auto& started = events.front();
    const auto& failed = events.back();
    CHECK(started.type == RuntimeEventType::kIngestionStarted);
    CHECK(failed.type == RuntimeEventType::kIngestionFailed);
    CHECK(started.event.source == IngestionSource::kDisk);
    CHECK(failed.event.source == IngestionSource::kDisk);
    CHECK(started.event.request_id == failed.event.request_id);
    CHECK_FALSE(started.event.publish_context_id.empty());
    CHECK(started.event.publish_context_id == failed.event.publish_context_id);
    CHECK(started.event.artifact_id == "disk_missing");
    CHECK(failed.event.status.code() == failure_status.code());
    CHECK(fake_pipeline->disk_invocations().size() == 1);
  }

  SECTION("disk ingestion success publishes completion event") {
    static_cast<void>(recording_sink->drain());
    loading::ReplicaHandle handle;
    handle.replica_key.artifact_id = "disk_success";
    fake_pipeline->set_next_disk_result(std::move(handle));
    auto handle_or = manager.ingest_from_disk("disk_success", disk_source, target, hints);
    REQUIRE(handle_or.ok());

    auto events = recording_sink->drain();
    REQUIRE(events.size() == 2);
    const auto& started = events.front();
    const auto& completed = events.back();
    CHECK(started.type == RuntimeEventType::kIngestionStarted);
    CHECK(completed.type == RuntimeEventType::kIngestionCompleted);
    CHECK(completed.event.status.ok());
    CHECK(completed.event.request_id == started.event.request_id);
    CHECK(completed.event.artifact_id == "disk_success");
    CHECK(fake_pipeline->disk_invocations().size() == 1);
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "IngestionRuntime reuses publish context ids for synchronous registrations",
    "[ingestion_runtime][publish_context]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "ingestion_runtime_publish_context";
  std::filesystem::create_directories(temp_root);

  StoreEngineOptions opts = MakeOptions(temp_root);
  ingestion_testing::ScopedIngestionRuntimeTestHarness harness(opts);
  CHECK_OK(harness.initialize());

  auto recording_sink = std::make_shared<ingestion_testing::RecordingEventSink>();
  auto dependencies = std::make_shared<IngestionRuntimeDependencies>();
  dependencies->event_sink_override = recording_sink;

  ingestion_testing::FakeIngestionPipeline* fake_pipeline = nullptr;
  dependencies->pipeline_factory = [&](const auto& cfg) {
    auto fake = std::make_unique<ingestion_testing::FakeIngestionPipeline>(cfg);
    fake_pipeline = fake.get();
    return fake;
  };

  RecordingCoordinator* recording_coordinator = nullptr;
  dependencies->coordinator_factory = [&](const auto& cfg) {
    auto recorder = std::make_unique<RecordingCoordinator>(cfg);
    recording_coordinator = recorder.get();
    return recorder;
  };

  IngestionRuntime manager(harness.make_runtime_config(dependencies));
  REQUIRE(fake_pipeline != nullptr);
  REQUIRE(recording_coordinator != nullptr);

  loading::DiskSource disk_source;
  disk_source.path = temp_root / "artifact_publish_ctx";

  loading::ReplicaTarget target;
  target.location.type = MemoryLocation::CPU;
  target.location.device_id = 0;

  loading::MaterializeHints hints;

  loading::ReplicaHandle handle;
  handle.replica_key.artifact_id = "artifact_publish_ctx";
  handle.replica_key.device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  fake_pipeline->set_next_disk_result(std::move(handle));

  static_cast<void>(recording_sink->drain());
  auto handle_or = manager.ingest_from_disk("artifact_publish_ctx", disk_source, target, hints);
  REQUIRE(handle_or.ok());
  loading::ReplicaHandle ready_handle = std::move(handle_or.value());

  auto events = recording_sink->drain();
  REQUIRE(events.size() == 2);
  const auto completed_ctx = events.back().event.publish_context_id;
  CHECK_FALSE(completed_ctx.empty());

  auto reg_status = manager.register_replica_with_global_store(ready_handle.key(), {});
  CHECK(reg_status.ok());
  CHECK(recording_coordinator->register_calls == 1);
  REQUIRE(recording_coordinator->last_key.has_value());
  CHECK(recording_coordinator->last_key->artifact_id == ready_handle.key().artifact_id);
  CHECK(recording_coordinator->last_publish_context_id == completed_ctx);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("IngestionRuntime emits p2p ingestion lifecycle events via injectable dependencies", "[ingestion_runtime]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "ingestion_runtime_p2p_events";
  std::filesystem::create_directories(temp_root);

  StoreEngineOptions opts = MakeOptions(temp_root);
  ingestion_testing::ScopedIngestionRuntimeTestHarness harness(opts);
  CHECK_OK(harness.initialize());

  auto recording_sink = std::make_shared<ingestion_testing::RecordingEventSink>();
  auto dependencies = std::make_shared<IngestionRuntimeDependencies>();
  dependencies->event_sink_override = recording_sink;
  ingestion_testing::FakeIngestionPipeline* fake_pipeline = nullptr;
  dependencies->pipeline_factory = [&](const auto& cfg) {
    auto fake = std::make_unique<ingestion_testing::FakeIngestionPipeline>(cfg);
    fake_pipeline = fake.get();
    return fake;
  };

  IngestionRuntime manager(harness.make_runtime_config(dependencies));
  REQUIRE(fake_pipeline != nullptr);

  loading::ReplicaTarget target;
  target.location.type = MemoryLocation::GPU;
  target.location.device_id = 0;

  P2PSource p2p_source;
  p2p_source.size_bytes = 4096;
  p2p_source.ip = "127.0.0.1";
  p2p_source.port = 55000;
  p2p_source.memory_keys = {"missing_key"};
  p2p_source.buf_sizes = {4096};
  p2p_source.location.type = MemoryLocation::GPU;
  p2p_source.location.device_id = 0;

  loading::MaterializeHints hints;

  SECTION("p2p ingestion failure publishes failure event") {
    static_cast<void>(recording_sink->drain());
    fake_pipeline->set_next_p2p_result(absl::FailedPreconditionError("p2p_stub_failure"));
    auto handle_or = manager.ingest_from_p2p("p2p_missing", p2p_source, target, hints);
    REQUIRE_FALSE(handle_or.ok());
    const absl::Status& failure_status = handle_or.status();

    auto events = recording_sink->drain();
    REQUIRE(events.size() == 2);
    const auto& started = events.front();
    const auto& failed = events.back();
    CHECK(started.type == RuntimeEventType::kIngestionStarted);
    CHECK(failed.type == RuntimeEventType::kIngestionFailed);
    CHECK(started.event.source == IngestionSource::kP2P);
    CHECK(failed.event.source == IngestionSource::kP2P);
    CHECK(started.event.request_id == failed.event.request_id);
    CHECK(started.event.publish_context_id == failed.event.publish_context_id);
    CHECK(failed.event.status.code() == failure_status.code());
    CHECK(fake_pipeline->p2p_invocations().size() == 1);
  }

  SECTION("p2p ingestion success publishes completion event") {
    static_cast<void>(recording_sink->drain());
    loading::ReplicaHandle handle;
    handle.replica_key.artifact_id = "p2p_success";
    fake_pipeline->set_next_p2p_result(std::move(handle));
    auto handle_or = manager.ingest_from_p2p("p2p_success", p2p_source, target, hints);
    REQUIRE(handle_or.ok());

    auto events = recording_sink->drain();
    REQUIRE(events.size() == 2);
    const auto& started = events.front();
    const auto& completed = events.back();
    CHECK(started.type == RuntimeEventType::kIngestionStarted);
    CHECK(completed.type == RuntimeEventType::kIngestionCompleted);
    CHECK(completed.event.status.ok());
    CHECK(completed.event.request_id == started.event.request_id);
    CHECK(completed.event.artifact_id == "p2p_success");
    CHECK(fake_pipeline->p2p_invocations().size() == 1);
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}
