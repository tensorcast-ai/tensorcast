// Copyright (c) 2025, TensorCast Team.

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmacro-redefined"
#endif
#include "catch2/catch_test_macros.hpp"
#include "core/store/runtime/context/runtime_context_events.h"
#include "core/store/runtime/ingestion/materialization_facade.h"
#include "core/store/runtime/ingestion/testing/fake_ingestion_pipeline.h"
#include "core/store/runtime/ingestion/testing/scoped_ingestion_runtime_test_harness.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/store_engine_options.h"
#include "core/testing/common.h"
#include "core/testing/test_helpers.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#include "gsl/pointers"

using tensorcast::DeviceType;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::P2PSource;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::runtime::IngestionCompletedEvent;
using tensorcast::store::runtime::IngestionResultEvent;
using tensorcast::store::runtime::IngestionStartedEvent;
using tensorcast::store::runtime::MaterializationHooks;
using tensorcast::store::runtime::RuntimeContext;
using tensorcast::store::runtime::RuntimeContextEvents;
using tensorcast::store::runtime::ingestion::MaterializationFacade;
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

struct FacadeHarness {
  ingestion_testing::ScopedIngestionRuntimeTestHarness harness;
  std::shared_ptr<MaterializationHooks> hooks;
  ingestion_testing::FakeIngestionPipeline* fake_pipeline{nullptr};
  std::unique_ptr<MaterializationFacade> facade;

  explicit FacadeHarness(const StoreEngineOptions& opts)
      : harness(opts), hooks(std::make_shared<MaterializationHooks>()) {}

  void initialize() {
    CHECK_OK(harness.initialize());
    hooks->pipeline_factory = [&](const auto& cfg) {
      auto fake = std::make_unique<ingestion_testing::FakeIngestionPipeline>(cfg);
      fake_pipeline = fake.get();
      return fake;
    };
    MaterializationFacade::Config cfg{
        .runtime_context = gsl::not_null<tensorcast::store::runtime::RuntimeContext*>{&harness.runtime_context()},
        .replica_runtime = gsl::not_null<tensorcast::store::runtime::ReplicaRuntime*>{&harness.replica_runtime()},
        .metadata_gateway =
            gsl::not_null<tensorcast::store::runtime::metadata::MetadataGateway*>{&harness.metadata_gateway()},
        .storage_path = std::filesystem::path(harness.options().storage_path),
        .artifact_chunk_bytes = harness.options().artifact_chunk_bytes,
        .pinned_memory_timeout = harness.options().pinned_memory_timeout,
        .num_threads = harness.options().num_thread,
        .options = &harness.options(),
        .hooks = hooks,
    };
    facade = std::make_unique<MaterializationFacade>(std::move(cfg));
    REQUIRE(fake_pipeline != nullptr);
  }

  void shutdown() {
    facade.reset();
    harness.shutdown();
  }

  RuntimeContext& runtime_context() {
    return harness.runtime_context();
  }
};

} // namespace

TEST_CASE("MaterializationFacade pipelines disk ingestion and publishes events", "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_disk";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();
  IngestionEventRecorder recorder(harness.runtime_context());

  loading::DiskSource disk_source;
  disk_source.path = temp_root / "missing_artifact";

  loading::ReplicaTarget target;
  target.location.type = MemoryLocation::CPU;
  target.location.device_id = 0;

  loading::MaterializeHints hints;

  SECTION("failure path publishes failed event") {
    static_cast<void>(recorder.drain_started());
    static_cast<void>(recorder.drain_completed());
    harness.fake_pipeline->set_next_disk_result(absl::NotFoundError("disk_missing_stub"));
    auto handle_or =
        harness.facade->ingest_from_disk("disk_missing", disk_source, target, hints, /*publish_to_global_store=*/true);
    REQUIRE_FALSE(handle_or.ok());
    harness.runtime_context().drain_events();

    auto started_events = recorder.drain_started();
    auto completed_events = recorder.drain_completed();
    REQUIRE(started_events.size() == 1);
    REQUIRE(completed_events.size() == 1);
    CHECK(started_events.front().request_id == completed_events.front().request_id);
    CHECK(completed_events.front().artifact_id == "disk_missing");
  }

  SECTION("success path publishes completed event and registers replica") {
    loading::ReplicaHandle handle;
    handle.replica_key.artifact_id = "disk_success";
    handle.replica_key.device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
    harness.fake_pipeline->set_next_disk_result(std::move(handle));

    static_cast<void>(recorder.drain_started());
    static_cast<void>(recorder.drain_completed());
    auto handle_or =
        harness.facade->ingest_from_disk("disk_success", disk_source, target, hints, /*publish_to_global_store=*/true);
    REQUIRE(handle_or.ok());
    harness.runtime_context().drain_events();

    auto completed_events = recorder.drain_completed();
    REQUIRE(completed_events.size() == 1);
    const auto& completed = completed_events.back();
    CHECK(completed.artifact_id == "disk_success");
    CHECK(completed.status.ok());
    CHECK_FALSE(completed.publish_context_id.empty());
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade registers stored publish contexts on demand", "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_register";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();
  IngestionEventRecorder recorder(harness.runtime_context());

  loading::DiskSource disk_source;
  disk_source.path = temp_root / "artifact_publish_ctx";

  loading::ReplicaTarget target;
  target.location.type = MemoryLocation::CPU;
  target.location.device_id = 0;

  loading::MaterializeHints hints;
  loading::ReplicaHandle handle;
  handle.replica_key.artifact_id = "artifact_publish_ctx";
  handle.replica_key.device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  harness.fake_pipeline->set_next_disk_result(std::move(handle));

  static_cast<void>(recorder.drain_started());
  static_cast<void>(recorder.drain_completed());
  auto handle_or = harness.facade->ingest_from_disk(
      "artifact_publish_ctx", disk_source, target, hints, /*publish_to_global_store=*/true);
  REQUIRE(handle_or.ok());
  harness.runtime_context().drain_events();
  loading::ReplicaHandle ready_handle = std::move(handle_or.value());

  auto completed_events = recorder.drain_completed();
  REQUIRE(completed_events.size() == 1);
  const auto publish_context = completed_events.back().publish_context_id;
  CHECK_FALSE(publish_context.empty());

  int register_calls = 0;
  harness.hooks->register_replica_override =
      [&](const loading::ReplicaKey& key, std::string_view artifact_override, std::string_view publish_ctx) {
        ++register_calls;
        CHECK(key.artifact_id == ready_handle.key().artifact_id);
        CHECK(publish_ctx == publish_context);
        CHECK(artifact_override.empty());
        return absl::OkStatus();
      };
  auto status = harness.facade->register_replica_with_global_store(ready_handle.key(), {});
  CHECK(status.ok());
  CHECK(register_calls == 1);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade handles p2p ingestion flows", "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_p2p";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();
  IngestionEventRecorder recorder(harness.runtime_context());

  loading::ReplicaTarget target;
  target.location.type = MemoryLocation::GPU;
  target.location.device_id = 0;

  P2PSource source;
  source.size_bytes = 4096;
  source.ip = "127.0.0.1";
  source.port = 55000;
  source.memory_keys = {"missing"};
  source.buf_sizes = {4096};
  source.location.type = MemoryLocation::GPU;
  source.location.device_id = 0;

  loading::MaterializeHints hints;

  SECTION("failure publishes failure event") {
    static_cast<void>(recorder.drain_started());
    static_cast<void>(recorder.drain_completed());
    harness.fake_pipeline->set_next_p2p_result(absl::FailedPreconditionError("p2p_stub_failure"));
    auto handle_or =
        harness.facade->ingest_from_p2p("p2p_missing", source, target, hints, /*publish_to_global_store=*/true);
    REQUIRE_FALSE(handle_or.ok());
    harness.runtime_context().drain_events();
    auto completed_events = recorder.drain_completed();
    REQUIRE(completed_events.size() == 1);
    CHECK(!completed_events.back().status.ok());
  }

  SECTION("success publishes completion event") {
    loading::ReplicaHandle handle;
    handle.replica_key.artifact_id = "p2p_success";
    harness.fake_pipeline->set_next_p2p_result(std::move(handle));
    static_cast<void>(recorder.drain_started());
    static_cast<void>(recorder.drain_completed());
    auto handle_or =
        harness.facade->ingest_from_p2p("p2p_success", source, target, hints, /*publish_to_global_store=*/true);
    REQUIRE(handle_or.ok());
    harness.runtime_context().drain_events();
    auto completed_events = recorder.drain_completed();
    REQUIRE(completed_events.size() == 1);
    CHECK(completed_events.back().artifact_id == "p2p_success");
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade materialize_replica reuses disk ingestion path", "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_materialize";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  loading::ReplicaTarget target;
  target.location.type = MemoryLocation::CPU;
  target.location.device_id = 0;

  loading::MaterializeHints hints;
  hints.disk_path = (temp_root / "artifact_auto").string();
  loading::ReplicaHandle handle;
  handle.replica_key.artifact_id = "artifact_auto";
  handle.replica_key.device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  harness.fake_pipeline->set_next_disk_result(std::move(handle));

  loading::MaterializeHints request_hints = hints;

  auto handle_or = harness.facade->materialize_replica(
      target.location.to_device_key(), loading::MaterializeMode::LOAD_ONLY, request_hints);
  REQUIRE(handle_or.ok());
  CHECK(handle_or->replica_key.artifact_id == "artifact_auto");

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}
