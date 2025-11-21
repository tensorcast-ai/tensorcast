// Copyright (c) 2025, TensorCast Team.

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "catch2/catch_test_macros.hpp"
#include "core/store/runtime/context/runtime_context_events.h"
#include "core/store/runtime/ingestion/ingestion_runtime.h"
#include "core/store/runtime/ingestion/testing/fake_ingestion_pipeline.h"
#include "core/store/runtime/ingestion/testing/scoped_ingestion_runtime_test_harness.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/store_engine_options.h"
#include "core/testing/common.h"
#include "core/testing/test_helpers.h"

using tensorcast::DeviceType;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::P2PSource;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::runtime::IngestionCompletedEvent;
using tensorcast::store::runtime::IngestionResultEvent;
using tensorcast::store::runtime::IngestionRuntime;
using tensorcast::store::runtime::IngestionRuntimeDependencies;
using tensorcast::store::runtime::IngestionSource;
using tensorcast::store::runtime::IngestionStartedEvent;
using tensorcast::store::runtime::MaterializationHooks;
using tensorcast::store::runtime::RuntimeContext;
using tensorcast::store::runtime::RuntimeContextEvents;
using tensorcast::store::runtime::RuntimeEvent;
using tensorcast::store::runtime::RuntimeEventType;
namespace ingestion_testing = tensorcast::store::runtime::ingestion::testing;

namespace loading = tensorcast::store::loading;

namespace {

class IngestionEventRecorder {
 public:
  explicit IngestionEventRecorder(RuntimeContext& context) {
    auto* hub = context.ingestion_event_hub();
    if (hub != nullptr) {
      started_subscription_ = hub->subscribe_started([this](const IngestionStartedEvent& event) {
        absl::MutexLock lock(&mu_);
        started_events_.push_back(event);
      });
      completed_subscription_ = hub->subscribe_completed([this](const IngestionCompletedEvent& event) {
        absl::MutexLock lock(&mu_);
        completed_events_.push_back(event);
      });
    }
  }

  std::vector<IngestionStartedEvent> drain_started() {
    absl::MutexLock lock(&mu_);
    std::vector<IngestionStartedEvent> events = std::move(started_events_);
    started_events_.clear();
    return events;
  }

  std::vector<IngestionCompletedEvent> drain_completed() {
    absl::MutexLock lock(&mu_);
    std::vector<IngestionCompletedEvent> events = std::move(completed_events_);
    completed_events_.clear();
    return events;
  }

 private:
  absl::Mutex mu_;
  std::vector<IngestionStartedEvent> started_events_ ABSL_GUARDED_BY(mu_);
  std::vector<IngestionCompletedEvent> completed_events_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<RuntimeContextEvents::Subscription> started_subscription_;
  std::unique_ptr<RuntimeContextEvents::Subscription> completed_subscription_;
};

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

} // namespace

TEST_CASE("IngestionRuntime emits disk ingestion lifecycle events via injectable dependencies", "[ingestion_runtime]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "ingestion_runtime_disk_events";
  std::filesystem::create_directories(temp_root);

  StoreEngineOptions opts = MakeOptions(temp_root);
  ingestion_testing::ScopedIngestionRuntimeTestHarness harness(opts);
  CHECK_OK(harness.initialize());

  auto dependencies = std::make_shared<IngestionRuntimeDependencies>();
  dependencies->hooks = std::make_shared<MaterializationHooks>();
  ingestion_testing::FakeIngestionPipeline* fake_pipeline = nullptr;
  dependencies->hooks->pipeline_factory = [&](const auto& cfg) {
    auto fake = std::make_unique<ingestion_testing::FakeIngestionPipeline>(cfg);
    fake_pipeline = fake.get();
    return fake;
  };

  IngestionEventRecorder recorder(harness.runtime_context());
  IngestionRuntime manager(harness.make_runtime_config(dependencies));
  REQUIRE(fake_pipeline != nullptr);

  loading::DiskSource disk_source;
  disk_source.path = temp_root / "missing_artifact";

  loading::ReplicaTarget target;
  target.location.type = MemoryLocation::CPU;
  target.location.device_id = 0;

  loading::MaterializeHints hints;

  SECTION("disk ingestion failure publishes failure event") {
    static_cast<void>(recorder.drain_started());
    static_cast<void>(recorder.drain_completed());
    fake_pipeline->set_next_disk_result(absl::NotFoundError("disk_missing_stub"));
    auto handle_or = manager.ingest_from_disk("disk_missing", disk_source, target, hints);
    REQUIRE_FALSE(handle_or.ok());
    const absl::Status& failure_status = handle_or.status();
    harness.runtime_context().drain_events();

    auto started_events = recorder.drain_started();
    auto completed_events = recorder.drain_completed();
    REQUIRE(started_events.size() == 1);
    REQUIRE(completed_events.size() == 1);
    const auto& started = started_events.front();
    const auto& failed = completed_events.front();
    CHECK(started.source == IngestionSource::kDisk);
    CHECK(failed.source == IngestionSource::kDisk);
    CHECK(started.request_id == failed.request_id);
    CHECK_FALSE(started.publish_context_id.empty());
    CHECK(started.publish_context_id == failed.publish_context_id);
    CHECK(started.artifact_id == "disk_missing");
    CHECK(failed.status.code() == failure_status.code());
    CHECK(fake_pipeline->disk_invocations().size() == 1);
  }

  SECTION("disk ingestion success publishes completion event") {
    static_cast<void>(recorder.drain_started());
    static_cast<void>(recorder.drain_completed());
    loading::ReplicaHandle handle;
    handle.replica_key.artifact_id = "disk_success";
    fake_pipeline->set_next_disk_result(std::move(handle));
    auto handle_or = manager.ingest_from_disk("disk_success", disk_source, target, hints);
    REQUIRE(handle_or.ok());
    harness.runtime_context().drain_events();

    auto started_events = recorder.drain_started();
    auto completed_events = recorder.drain_completed();
    REQUIRE(started_events.size() == 1);
    REQUIRE(completed_events.size() == 1);
    const auto& started = started_events.front();
    const auto& completed = completed_events.front();
    CHECK(completed.status.ok());
    CHECK(completed.request_id == started.request_id);
    CHECK(completed.artifact_id == "disk_success");
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

  auto dependencies = std::make_shared<IngestionRuntimeDependencies>();
  dependencies->hooks = std::make_shared<MaterializationHooks>();

  ingestion_testing::FakeIngestionPipeline* fake_pipeline = nullptr;
  dependencies->hooks->pipeline_factory = [&](const auto& cfg) {
    auto fake = std::make_unique<ingestion_testing::FakeIngestionPipeline>(cfg);
    fake_pipeline = fake.get();
    return fake;
  };

  struct RegisterCapture {
    int calls{0};
    std::optional<loading::ReplicaKey> manual_key;
    std::string manual_publish_context_id;
    std::string manual_artifact_override;
  };

  auto register_capture = std::make_shared<RegisterCapture>();
  dependencies->hooks->register_replica_override =
      [register_capture](
          const loading::ReplicaKey& key, std::string_view artifact_id_override, std::string_view publish_context_id) {
        ++register_capture->calls;
        register_capture->manual_key = key;
        register_capture->manual_publish_context_id = std::string(publish_context_id);
        register_capture->manual_artifact_override = std::string(artifact_id_override);
        return absl::OkStatus();
      };

  IngestionEventRecorder recorder(harness.runtime_context());
  IngestionRuntime manager(harness.make_runtime_config(dependencies));
  REQUIRE(fake_pipeline != nullptr);

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

  static_cast<void>(recorder.drain_started());
  static_cast<void>(recorder.drain_completed());
  auto handle_or = manager.ingest_from_disk("artifact_publish_ctx", disk_source, target, hints);
  REQUIRE(handle_or.ok());
  loading::ReplicaHandle ready_handle = std::move(handle_or.value());
  harness.runtime_context().drain_events();

  auto completed_events = recorder.drain_completed();
  REQUIRE(completed_events.size() == 1);
  const auto completed_ctx = completed_events.back().publish_context_id;
  CHECK_FALSE(completed_ctx.empty());

  auto reg_status = manager.register_replica_with_global_store(ready_handle.key(), {});
  CHECK(reg_status.ok());
  CHECK(register_capture->calls == 1);
  REQUIRE(register_capture->manual_key.has_value());
  CHECK(register_capture->manual_key->artifact_id == ready_handle.key().artifact_id);
  CHECK(register_capture->manual_publish_context_id == completed_ctx);

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

  auto dependencies = std::make_shared<IngestionRuntimeDependencies>();
  dependencies->hooks = std::make_shared<MaterializationHooks>();
  ingestion_testing::FakeIngestionPipeline* fake_pipeline = nullptr;
  dependencies->hooks->pipeline_factory = [&](const auto& cfg) {
    auto fake = std::make_unique<ingestion_testing::FakeIngestionPipeline>(cfg);
    fake_pipeline = fake.get();
    return fake;
  };

  IngestionEventRecorder recorder(harness.runtime_context());
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
    static_cast<void>(recorder.drain_started());
    static_cast<void>(recorder.drain_completed());
    fake_pipeline->set_next_p2p_result(absl::FailedPreconditionError("p2p_stub_failure"));
    auto handle_or = manager.ingest_from_p2p("p2p_missing", p2p_source, target, hints);
    REQUIRE_FALSE(handle_or.ok());
    const absl::Status& failure_status = handle_or.status();
    harness.runtime_context().drain_events();

    auto started_events = recorder.drain_started();
    auto completed_events = recorder.drain_completed();
    REQUIRE(started_events.size() == 1);
    REQUIRE(completed_events.size() == 1);
    const auto& started = started_events.front();
    const auto& failed = completed_events.front();
    CHECK(started.source == IngestionSource::kP2P);
    CHECK(failed.source == IngestionSource::kP2P);
    CHECK(started.request_id == failed.request_id);
    CHECK(started.publish_context_id == failed.publish_context_id);
    CHECK(failed.status.code() == failure_status.code());
    CHECK(fake_pipeline->p2p_invocations().size() == 1);
  }

  SECTION("p2p ingestion success publishes completion event") {
    static_cast<void>(recorder.drain_started());
    static_cast<void>(recorder.drain_completed());
    loading::ReplicaHandle handle;
    handle.replica_key.artifact_id = "p2p_success";
    fake_pipeline->set_next_p2p_result(std::move(handle));
    auto handle_or = manager.ingest_from_p2p("p2p_success", p2p_source, target, hints);
    REQUIRE(handle_or.ok());
    harness.runtime_context().drain_events();

    auto started_events = recorder.drain_started();
    auto completed_events = recorder.drain_completed();
    REQUIRE(started_events.size() == 1);
    REQUIRE(completed_events.size() == 1);
    const auto& started = started_events.front();
    const auto& completed = completed_events.front();
    CHECK(completed.status.ok());
    CHECK(completed.request_id == started.request_id);
    CHECK(completed.artifact_id == "p2p_success");
    CHECK(fake_pipeline->p2p_invocations().size() == 1);
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}
