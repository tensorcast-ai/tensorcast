// Copyright (c) 2025-2026, TensorCast Team.

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/synchronization/mutex.h"
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmacro-redefined"
#endif
#include "catch2/catch_test_macros.hpp"
#include "core/common/artifact_hash.h"
#include "core/cuda/cuda_api.h"
#include "core/store/components/worker_identity.h"
#include "core/store/materialization/dataplane/contracts/inline_buffer_loader.h"
#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"
#include "core/store/materialization/dataplane/sources/memory_source.h"
#include "core/store/materialization/runtime/pipeline/ingestion_context.h"
#include "core/store/replica/replica.h"
#include "core/store/runtime/context/runtime_context_events.h"
#include "core/store/runtime/ingestion/artifact_lowering_plan.h"
#include "core/store/runtime/ingestion/materialization_facade.h"
#include "core/store/runtime/ingestion/testing/fake_ingestion_pipeline.h"
#include "core/store/runtime/ingestion/testing/scoped_ingestion_runtime_test_harness.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/recording_global_store_client.h"
#include "core/testing/common.h"
#include "core/testing/test_helpers.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#include "gsl/pointers"

using tensorcast::DeviceType;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::DeviceKey;
using tensorcast::store::P2PSource;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::components::WorkerIdentity;
using tensorcast::store::runtime::IngestionCompletedEvent;
using tensorcast::store::runtime::IngestionResultEvent;
using tensorcast::store::runtime::IngestionStartedEvent;
using tensorcast::store::runtime::MaterializationHooks;
using tensorcast::store::runtime::ReplicaRuntime;
using tensorcast::store::runtime::RuntimeContext;
using tensorcast::store::runtime::RuntimeContextEvents;
using tensorcast::store::runtime::ingestion::ArtifactLoweringPlan;
using tensorcast::store::runtime::ingestion::MaterializationFacade;
using tensorcast::store::runtime::ingestion::open_single_loader_sources;
using tensorcast::store::testing::RecordingGlobalStoreClient;
namespace ingestion_testing = tensorcast::store::runtime::ingestion::testing;
namespace pipeline_runtime = tensorcast::store::materialization::runtime::pipeline;
namespace view_contracts = tensorcast::store::materialization::view;

namespace loading = tensorcast::store::loading;

namespace tensorcast::store::runtime::ingestion {

class MaterializationFacadeTestPeer {
 public:
  static absl::StatusOr<loading::MaterializeIntoTargetResult> materialize_mapped_sources_into_target(
      MaterializationFacade& facade,
      const DeviceKey& target_device,
      const loading::IntoTargetLayout& target_layout,
      std::vector<std::shared_ptr<loader::SeekableSource>> sources,
      const tensorcast::store::loader::ByteRangeMap& mapping,
      const loading::MaterializeHints& hints,
      loading::MaterializationSource source_kind) {
    return facade.materialize_mapped_sources_into_target(
        target_device, target_layout, std::move(sources), mapping, hints, source_kind);
  }
};

} // namespace tensorcast::store::runtime::ingestion

namespace {

using PreparedSourceBoundExecutionPlan =
    tensorcast::store::runtime::ingestion::strategy::PreparedSourceBoundExecutionPlan;
using ResolvedMaterializationPlan = tensorcast::store::runtime::ingestion::strategy::ResolvedMaterializationPlan;
using SourceBoundLoweringArtifacts = tensorcast::store::runtime::ingestion::strategy::SourceBoundLoweringArtifacts;

template <typename PayloadT>
absl::StatusOr<std::vector<std::shared_ptr<tensorcast::store::loader::SeekableSource>>> open_inline_sources(
    std::shared_ptr<PayloadT> payload,
    std::uint64_t size_bytes,
    const tensorcast::store::loader::ByteRangeMap& mapping,
    std::string_view context) {
  return tensorcast::store::runtime::ingestion::open_single_loader_sources(
      std::make_unique<tensorcast::store::InlineBufferLoader>(
          loading::InlineBufferSource{.data = std::move(payload), .size_bytes = size_bytes}),
      mapping,
      context);
}

PreparedSourceBoundExecutionPlan make_prepared_source_bound_execution_plan(
    ResolvedMaterializationPlan resolved_plan,
    std::optional<tensorcast::store::loader::ByteRangeMap> executor_generic_data_map = std::nullopt,
    std::optional<tensorcast::store::loader::ByteRangeMap> collective_data_map = std::nullopt) {
  PreparedSourceBoundExecutionPlan prepared_execution;
  prepared_execution.resolved_plan = std::move(resolved_plan);
  if (executor_generic_data_map.has_value() || collective_data_map.has_value()) {
    prepared_execution.lowering_artifacts = SourceBoundLoweringArtifacts{
        .executor_generic_data_map = std::move(executor_generic_data_map),
        .collective_data_map = std::move(collective_data_map),
    };
  }
  return prepared_execution;
}

tensorcast::store::runtime::ingestion::strategy::SourceBoundStrategyPlan make_source_bound_strategy_plan(
    tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode mode,
    tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan lane_plan,
    tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary summary,
    tensorcast::store::runtime::ingestion::strategy::SourceBoundPolicy policy =
        tensorcast::store::runtime::ingestion::strategy::SourceBoundPolicy::kCollectiveFirst) {
  if (lane_plan.selection_reason.empty()) {
    lane_plan.selection_reason =
        std::string(tensorcast::store::runtime::ingestion::strategy::source_bound_execution_mode_name(mode));
  }
  if (!lane_plan.local_mapped_typed_selected &&
      lane_plan.selection_reason.find("local_mapped_typed") != std::string::npos) {
    lane_plan.local_mapped_typed_selected = true;
  }
  lane_plan.mode = mode;
  if (summary.execution_plan_kind.empty()) {
    summary.execution_plan_kind =
        std::string(tensorcast::store::runtime::ingestion::strategy::source_bound_execution_mode_name(mode));
  }
  return tensorcast::store::runtime::ingestion::strategy::SourceBoundStrategyPlan{
      .policy = policy,
      .lane_plan = std::move(lane_plan),
      .summary = std::move(summary),
  };
}

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

  ReplicaRuntime& replica_runtime() {
    return harness.replica_runtime();
  }

  const StoreEngineOptions& options() const {
    return harness.options();
  }
};

std::string sha256_hex(std::string_view payload) {
  const auto digest = tensorcast::common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  std::string hex =
      absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));
  absl::AsciiStrToLower(&hex);
  return hex;
}

std::filesystem::path make_temp_dir(const std::string& prefix) {
  auto base = std::filesystem::temp_directory_path();
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dist;
  auto dir = base / (prefix + "-" + std::to_string(dist(gen)));
  std::filesystem::create_directories(dir);
  return dir;
}

void write_u64_le(std::ofstream& out, uint64_t value) {
  unsigned char buf[8];
  for (int i = 0; i < 8; ++i) {
    buf[i] = static_cast<unsigned char>((value >> (8 * i)) & 0xFF);
  }
  out.write(reinterpret_cast<const char*>(buf), 8);
}

void create_safetensors_file(
    const std::filesystem::path& path,
    const std::string& header_json,
    const std::vector<unsigned char>& payload) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  REQUIRE(out.is_open());
  write_u64_le(out, static_cast<uint64_t>(header_json.size()));
  out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
  if (!payload.empty()) {
    out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
  }
}

pipeline_runtime::IngestionContext make_strategy_context(
    FacadeHarness& harness,
    const std::filesystem::path& artifact_path,
    loading::SourceLocalityHint source_locality,
    std::optional<std::string> source_sharing_domain) {
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  auto disk_context_or = tensorcast::store::loader::get_disk_artifact_context(artifact_path);
  if (!disk_context_or.ok()) {
    throw std::runtime_error(disk_context_or.status().ToString());
  }
  auto index_info_or = (*disk_context_or)->get_index_info(/*target_device_id=*/0);
  if (!index_info_or.ok()) {
    throw std::runtime_error(index_info_or.status().ToString());
  }

  pipeline_runtime::IngestionContext ctx{
      .source_type = pipeline_runtime::SourceType::kDisk,
      .request_id = "strategy-request",
      .publish_context_id = "strategy-publish",
      .materialize_mode = loading::MaterializeMode::LOAD_ONLY,
      .artifact_identifier = "cgid:strategy-artifact",
      .target =
          loading::ReplicaTarget{
              .location = {.type = MemoryLocation::GPU, .device_id = 0},
          },
      .hints =
          loading::MaterializeHints{
              .collective_load_group =
                  loading::CollectiveLoadGroupHint{.group_id = "strategy", .world_size = 2, .rank = 0},
              .source_locality = source_locality,
              .source_sharing_domain = std::move(source_sharing_domain),
              .artifact_id = "cgid:strategy-artifact",
              .variant = std::optional<view_contracts::VariantIdentity>(view_contracts::VariantIdentity{
                  .canonical_artifact_id = "cgid:strategy-artifact",
              }),
          },
      .target_device = {.type = DeviceType::GPU, .ordinal = 0, .uuid = ""},
      .target_location = MemoryLocation::GPU,
      .target_device_id = 0,
      .target_is_gpu = true,
      .storage_path = artifact_path,
      .artifact_chunk_bytes = harness.options().artifact_chunk_bytes,
      .tx_slice_bytes = harness.runtime_context().tx_slice_bytes(),
      .num_threads = harness.options().num_thread,
      .pinned_memory_timeout = harness.options().pinned_memory_timeout,
      .options = &harness.options(),
      .replica_runtime = &harness.replica_runtime(),
      .runtime_context = &harness.runtime_context(),
      .ordinary_disk_strategy_planner = {},
      .disk =
          pipeline_runtime::DiskSourceMetadata{
              .source = loading::DiskSource{.path = artifact_path, .expected_size = 64},
              .artifact_path = artifact_path,
              .descriptor_present = false,
              .is_safetensors = true,
              .source_index_json = index_info_or->canonical_index_json,
              .source_total_size_bytes = 64,
          },
      .verification =
          pipeline_runtime::VerificationState{
              .canonical_index_json = index_info_or->canonical_index_json,
              .logical_total_size = 64,
          },
      .start_time = std::chrono::steady_clock::now(),
      .publish_to_global_store = false,
  };
  return ctx;
}

absl::StatusOr<size_t> copy_into_direct_write_grant(
    absl::Span<const uint8_t> data,
    uint64_t src_offset,
    uint64_t dest_va_offset,
    size_t bytes,
    const tensorcast::store::DirectWriteGrant& grant) {
  size_t total = 0;
  uint64_t src_pos = src_offset;
  uint64_t dest_pos = dest_va_offset;
  while (total < bytes) {
    const tensorcast::store::DirectWriteGrant::Window* window = nullptr;
    for (const auto& candidate : grant.windows) {
      if (dest_pos >= candidate.va_offset && dest_pos < candidate.va_offset + candidate.length) {
        window = &candidate;
        break;
      }
    }
    if (window == nullptr) {
      return absl::InvalidArgumentError("no direct-write window for destination offset");
    }
    const uint64_t window_offset = dest_pos - window->va_offset;
    const size_t available = static_cast<size_t>(window->length - window_offset);
    const size_t step = std::min(bytes - total, available);
    if (src_pos > data.size() || step > data.size() - src_pos) {
      return absl::OutOfRangeError("source eof");
    }
    std::memcpy(reinterpret_cast<void*>(window->local_addr + window_offset), data.data() + src_pos, step);
    total += step;
    src_pos += step;
    dest_pos += step;
  }
  return total;
}

struct RecordingDirectSourceStats {
  size_t readv_calls{0};
  size_t read_into_calls{0};
  std::vector<size_t> batch_sizes;
  std::vector<bool> readv_window_has_stable_backing;
  std::vector<bool> read_into_window_has_stable_backing;
  std::vector<std::string> stable_backing_ids;
};

class RecordingDirectSeekableSource final : public tensorcast::store::loader::SeekableSource {
 public:
  RecordingDirectSeekableSource(
      std::vector<uint8_t> data,
      std::shared_ptr<RecordingDirectSourceStats> stats,
      bool batched_direct_write_supported,
      std::optional<absl::Status> readv_failure,
      bool partial_write_before_failure)
      : data_(std::move(data)),
        stats_(std::move(stats)),
        batched_direct_write_supported_(batched_direct_write_supported),
        readv_failure_(std::move(readv_failure)),
        partial_write_before_failure_(partial_write_before_failure) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return data_.size();
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(cursor_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or;
    }
    cursor_ += *read_or;
    return read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= data_.size()) {
      return static_cast<size_t>(0);
    }
    const size_t to_read = std::min<size_t>(bytes, data_.size() - static_cast<size_t>(offset));
    std::memcpy(dst, data_.data() + offset, to_read);
    return to_read;
  }

  [[nodiscard]] bool supports_direct_write_at() const override {
    return true;
  }

  [[nodiscard]] bool supports_batched_direct_write_at() const override {
    return batched_direct_write_supported_;
  }

  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const tensorcast::store::DirectWriteGrant& grant) override {
    ++stats_->read_into_calls;
    bool has_stable_backing = false;
    for (const auto& window : grant.windows) {
      if (window.stable_backing.has_value()) {
        has_stable_backing = true;
        stats_->stable_backing_ids.push_back(window.stable_backing->backing_id);
      }
    }
    stats_->read_into_window_has_stable_backing.push_back(has_stable_backing);
    return copy_into_direct_write_grant(data_, src_offset, dest_va_offset, bytes, grant);
  }

  absl::StatusOr<size_t> readv_into_at(
      absl::Span<const tensorcast::store::loader::DirectWriteOp> ops,
      const tensorcast::store::DirectWriteGrant& grant) override {
    ++stats_->readv_calls;
    stats_->batch_sizes.push_back(ops.size());
    bool has_stable_backing = false;
    for (const auto& window : grant.windows) {
      if (window.stable_backing.has_value()) {
        has_stable_backing = true;
        stats_->stable_backing_ids.push_back(window.stable_backing->backing_id);
      }
    }
    stats_->readv_window_has_stable_backing.push_back(has_stable_backing);
    if (readv_failure_.has_value()) {
      if (partial_write_before_failure_ && !ops.empty()) {
        auto first_write_or = copy_into_direct_write_grant(
            data_, ops.front().src_offset, ops.front().dest_va_offset, static_cast<size_t>(ops.front().bytes), grant);
        if (!first_write_or.ok()) {
          return first_write_or.status();
        }
      }
      return *readv_failure_;
    }
    size_t total = 0;
    for (const auto& op : ops) {
      auto wrote_or =
          copy_into_direct_write_grant(data_, op.src_offset, op.dest_va_offset, static_cast<size_t>(op.bytes), grant);
      if (!wrote_or.ok()) {
        return wrote_or.status();
      }
      total += *wrote_or;
    }
    return total;
  }

 private:
  std::vector<uint8_t> data_;
  std::shared_ptr<RecordingDirectSourceStats> stats_;
  bool batched_direct_write_supported_{true};
  uint64_t cursor_{0};
  std::optional<absl::Status> readv_failure_;
  bool partial_write_before_failure_{false};
};

class RecordingDirectLoader final : public tensorcast::store::IArtifactLoader {
 public:
  RecordingDirectLoader(
      std::vector<uint8_t> data,
      std::shared_ptr<RecordingDirectSourceStats> stats,
      bool batched_direct_write_supported = true)
      : data_(std::move(data)),
        stats_(std::move(stats)),
        batched_direct_write_supported_(batched_direct_write_supported) {}

  absl::Status initialize() override {
    initialized_ = true;
    return absl::OkStatus();
  }

  absl::StatusOr<uint64_t> get_artifact_size() override {
    if (!initialized_) {
      return absl::FailedPreconditionError("loader must be initialized before size lookup");
    }
    return data_.size();
  }

  absl::StatusOr<std::unique_ptr<tensorcast::store::loader::SeekableSource>> open_source() override {
    if (!initialized_) {
      return absl::FailedPreconditionError("loader must be initialized before open_source");
    }
    return std::make_unique<RecordingDirectSeekableSource>(
        data_, stats_, batched_direct_write_supported_, readv_failure_, partial_write_before_failure_);
  }

  void set_readv_failure(absl::Status status, bool partial_write_before_failure = false) {
    readv_failure_ = std::move(status);
    partial_write_before_failure_ = partial_write_before_failure;
  }

 private:
  std::vector<uint8_t> data_;
  std::shared_ptr<RecordingDirectSourceStats> stats_;
  bool batched_direct_write_supported_{true};
  bool initialized_{false};
  std::optional<absl::Status> readv_failure_;
  bool partial_write_before_failure_{false};
};

TEST_CASE(
    "MaterializationFacade internal mapped sources helper executes composite CPU sources into one target",
    "[materialization_facade]") {
  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_mapped_sources_helper";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  std::array<uint8_t, 4> source0 = {0x10, 0x11, 0x12, 0x13};
  std::array<uint8_t, 4> source1 = {0x20, 0x21, 0x22, 0x23};
  std::array<uint8_t, 10> target{};
  target.fill(0xCC);

  std::vector<std::shared_ptr<tensorcast::store::loader::SeekableSource>> sources;
  sources.emplace_back(
      std::make_shared<tensorcast::store::loader::CpuMemorySource>(
          gsl::not_null<const void*>{static_cast<const void*>(source0.data())}, source0.size()));
  sources.emplace_back(
      std::make_shared<tensorcast::store::loader::CpuMemorySource>(
          gsl::not_null<const void*>{static_cast<const void*>(source1.data())}, source1.size()));

  tensorcast::store::loader::ByteRangeMap mapping;
  mapping.total_bytes = target.size();
  mapping.num_sources = 2;
  mapping.segments = {
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = source0.size(),
          .src_offset = 0,
          .source_index = 0,
      },
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = source0.size(),
          .length = source1.size(),
          .src_offset = 0,
          .source_index = 1,
      },
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kPad,
          .dst_offset = source0.size() + source1.size(),
          .length = target.size() - source0.size() - source1.size(),
          .src_offset = 0,
          .source_index = 0,
      },
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{static_cast<void*>(target.data())}, .length = target.size()});
  target_layout.total_size = target.size();

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:composite-mapped-sources";

  auto result_or =
      tensorcast::store::runtime::ingestion::MaterializationFacadeTestPeer::materialize_mapped_sources_into_target(
          *harness.facade,
          DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
          target_layout,
          std::move(sources),
          mapping,
          hints,
          loading::MaterializationSource::kLocalReplica);
  REQUIRE(result_or.ok());
  CHECK(result_or->source == loading::MaterializationSource::kLocalReplica);
  CHECK(result_or->requested_bytes == target.size());
  CHECK(result_or->committed_bytes == target.size());
  CHECK(result_or->dominant_executor == "MappedSourcesTargetExecutor");
  CHECK(result_or->selection_reason == "mapped_sources_target");

  for (size_t index = 0; index < source0.size(); ++index) {
    CHECK(target[index] == source0[index]);
  }
  for (size_t index = 0; index < source1.size(); ++index) {
    CHECK(target[source0.size() + index] == source1[index]);
  }
  for (size_t index = source0.size() + source1.size(); index < target.size(); ++index) {
    CHECK(target[index] == 0);
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade mapped sources route through shared target executor", "[materialization_facade]") {
  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_mapped_sources_executor";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  const auto payload =
      std::make_shared<std::array<uint8_t, 6>>(std::array<uint8_t, 6>{0x31, 0x32, 0x33, 0x34, 0x35, 0x36});
  std::array<uint8_t, 6> target{};
  target.fill(0xEE);

  tensorcast::store::loader::ByteRangeMap mapping;
  mapping.total_bytes = target.size();
  mapping.num_sources = 1;
  mapping.segments = {
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = target.size(),
          .src_offset = 0,
          .source_index = 0,
      },
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{static_cast<void*>(target.data())}, .length = target.size()});
  target_layout.total_size = target.size();

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:mapped-sources-executor";

  auto sources_or = open_single_loader_sources(
      std::make_unique<tensorcast::store::InlineBufferLoader>(
          loading::InlineBufferSource{.data = payload, .size_bytes = payload->size()}),
      mapping,
      "mapped-sources-executor-test");
  REQUIRE(sources_or.ok());
  auto result_or = harness.facade->materialize_mapped_sources_into_target(
      DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      target_layout,
      std::move(*sources_or),
      mapping,
      hints,
      loading::MaterializationSource::kDisk);
  REQUIRE(result_or.ok());
  CHECK(result_or->source == loading::MaterializationSource::kDisk);
  CHECK(result_or->requested_bytes == target.size());
  CHECK(result_or->committed_bytes == target.size());
  CHECK(result_or->dominant_executor == "MappedSourcesTargetExecutor");
  CHECK(result_or->selection_reason == "mapped_sources_target");

  for (size_t index = 0; index < target.size(); ++index) {
    CHECK(target[index] == (*payload)[index]);
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade mapped sources integrate batched direct writes", "[materialization_facade]") {
  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_mapped_sources_direct_batch";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  auto stats = std::make_shared<RecordingDirectSourceStats>();
  std::vector<uint8_t> source_bytes{1, 2, 3, 4, 5, 6, 7, 8, 101, 102, 103, 104, 105, 106, 107, 108};
  std::array<uint8_t, 8> target{};
  target.fill(0xEE);

  tensorcast::store::loader::ByteRangeMap mapping;
  mapping.total_bytes = target.size();
  mapping.num_sources = 1;
  mapping.segments = {
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 4,
          .src_offset = 0,
          .source_index = 0,
      },
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 4,
          .length = 4,
          .src_offset = 8,
          .source_index = 0,
      },
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{static_cast<void*>(target.data())},
          .length = target.size(),
          .stable_backing =
              tensorcast::store::StableLocalBackingRef{
                  .kind = tensorcast::store::StableLocalBackingKind::kHostSharedRegion,
                  .backing_id = "region:test-batched-direct-write",
                  .backing_base_addr = reinterpret_cast<uint64_t>(target.data()),
                  .backing_bytes = target.size(),
                  .dev_type = tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
                  .dev_id = 0,
              },
      });
  target_layout.total_size = target.size();

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:mapped-sources-direct-batch";

  auto sources_or = open_single_loader_sources(
      std::make_unique<RecordingDirectLoader>(source_bytes, stats), mapping, "mapped-sources-direct-batch-test");
  REQUIRE(sources_or.ok());
  auto result_or = harness.facade->materialize_mapped_sources_into_target(
      DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      target_layout,
      std::move(*sources_or),
      mapping,
      hints,
      loading::MaterializationSource::kLocalReplica);
  REQUIRE(result_or.ok());
  CHECK(result_or->source == loading::MaterializationSource::kLocalReplica);
  CHECK(result_or->direct_write_supported);
  CHECK(result_or->dominant_executor == "MappedSourcesTargetExecutor");
  CHECK(result_or->selection_reason == "mapped_sources_target");
  REQUIRE(result_or->debug_stats.has_value());
  CHECK(result_or->debug_stats->produced_chunks == 1);
  CHECK(result_or->debug_stats->produced_bytes == target.size());
  CHECK(stats->readv_calls == 1);
  CHECK(stats->read_into_calls == 0);
  CHECK(stats->batch_sizes == std::vector<size_t>{2});
  CHECK(stats->readv_window_has_stable_backing == std::vector<bool>{true});
  CHECK(stats->stable_backing_ids == std::vector<std::string>{"region:test-batched-direct-write"});
  const std::array<uint8_t, 8> expected_target{1, 2, 3, 4, 101, 102, 103, 104};
  CHECK(target == expected_target);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade direct-write CPU target does not require pinned buffer availability",
    "[materialization_facade]") {
  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_direct_write_without_pinned_pool";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  auto stats = std::make_shared<RecordingDirectSourceStats>();
  std::vector<uint8_t> source_bytes{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<uint8_t, 8> target{};
  target.fill(0xEE);

  auto pool = harness.runtime_context().pinned_buffer_pool();
  REQUIRE(pool != nullptr);
  const size_t available_bytes = pool->get_available_size();
  REQUIRE(available_bytes > 0);
  std::vector<char*> held_buffers;
  REQUIRE(
      pool->allocate(
          available_bytes, held_buffers, std::chrono::milliseconds::zero(), "facade-direct-write-held-buffers") == 0);
  REQUIRE(pool->get_available_size() == 0);

  tensorcast::store::loader::ByteRangeMap mapping;
  mapping.total_bytes = target.size();
  mapping.num_sources = 1;
  mapping.segments = {
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 4,
          .src_offset = 0,
          .source_index = 0,
      },
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 4,
          .length = 4,
          .src_offset = 4,
          .source_index = 0,
      },
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{static_cast<void*>(target.data())},
          .length = target.size(),
      });
  target_layout.total_size = target.size();

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:direct-write-without-pinned-pool";
  hints.pinned_timeout = std::chrono::milliseconds(1);

  auto sources_or = open_single_loader_sources(
      std::make_unique<RecordingDirectLoader>(source_bytes, stats), mapping, "direct-write-without-pinned-pool-test");
  REQUIRE(sources_or.ok());
  auto result_or = harness.facade->materialize_mapped_sources_into_target(
      DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      target_layout,
      std::move(*sources_or),
      mapping,
      hints,
      loading::MaterializationSource::kLocalReplica);
  REQUIRE(result_or.ok());
  CHECK(result_or->direct_write_supported);
  REQUIRE(result_or->debug_stats.has_value());
  CHECK(result_or->debug_stats->produced_chunks == 1);
  CHECK(result_or->debug_stats->produced_bytes == target.size());
  CHECK(stats->readv_calls == 1);
  CHECK(stats->read_into_calls == 0);
  CHECK(pool->get_available_size() == 0);
  const std::array<uint8_t, 8> expected_target{1, 2, 3, 4, 5, 6, 7, 8};
  CHECK(target == expected_target);

  REQUIRE(pool->deallocate(held_buffers) == 0);
  CHECK(pool->get_available_size() == available_bytes);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade collapses single-source direct-write composite target ranges",
    "[materialization_facade]") {
  auto temp_root =
      std::filesystem::temp_directory_path() / "materialization_facade_single_source_direct_write_composite_target";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  auto stats = std::make_shared<RecordingDirectSourceStats>();
  std::vector<uint8_t> source_bytes(64);
  for (size_t i = 0; i < source_bytes.size(); ++i) {
    source_bytes[i] = static_cast<uint8_t>(i);
  }
  std::array<std::array<uint8_t, 4>, 16> target_segments{};
  for (auto& segment : target_segments) {
    segment.fill(0xEE);
  }

  tensorcast::store::loader::ByteRangeMap mapping;
  mapping.total_bytes = source_bytes.size();
  mapping.num_sources = 1;
  for (size_t i = 0; i < target_segments.size(); ++i) {
    mapping.segments.push_back(
        tensorcast::store::loader::ByteRangeSegment{
            .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
            .dst_offset = static_cast<uint64_t>(i * target_segments[i].size()),
            .length = target_segments[i].size(),
            .src_offset = static_cast<uint64_t>(i * target_segments[i].size()),
            .source_index = 0,
        });
  }

  loading::IntoTargetLayout target_layout;
  for (auto& segment : target_segments) {
    target_layout.storages.push_back(
        loading::IntoTargetStorage{
            .base_ptr = gsl::not_null<void*>{static_cast<void*>(segment.data())},
            .length = segment.size(),
            .stable_backing =
                tensorcast::store::StableLocalBackingRef{
                    .kind = tensorcast::store::StableLocalBackingKind::kHostSharedRegion,
                    .backing_id = "region:test-single-source-direct-write-composite-target",
                    .backing_base_addr = reinterpret_cast<uint64_t>(segment.data()),
                    .backing_bytes = segment.size(),
                    .dev_type = tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
                    .dev_id = 0,
                },
        });
  }
  target_layout.total_size = source_bytes.size();

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:single-source-direct-write-composite-target";

  auto sources_or = open_single_loader_sources(
      std::make_unique<RecordingDirectLoader>(source_bytes, stats),
      mapping,
      "single-source-direct-write-composite-target-test");
  REQUIRE(sources_or.ok());
  auto result_or = harness.facade->materialize_mapped_sources_into_target(
      DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      target_layout,
      std::move(*sources_or),
      mapping,
      hints,
      loading::MaterializationSource::kLocalReplica);
  REQUIRE(result_or.ok());
  CHECK(result_or->direct_write_supported);
  REQUIRE(result_or->debug_stats.has_value());
  CHECK(result_or->debug_stats->produced_chunks == 1);
  CHECK(result_or->debug_stats->produced_bytes == source_bytes.size());
  CHECK(stats->readv_calls == 1);
  CHECK(stats->read_into_calls == 0);
  CHECK(stats->batch_sizes == std::vector<size_t>{1});
  CHECK(stats->readv_window_has_stable_backing == std::vector<bool>{true});

  for (size_t i = 0; i < target_segments.size(); ++i) {
    const std::array<uint8_t, 4> expected{
        source_bytes[i * 4], source_bytes[i * 4 + 1], source_bytes[i * 4 + 2], source_bytes[i * 4 + 3]};
    CHECK(target_segments[i] == expected);
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade mapped sources preserve pre-issue direct-write fallback", "[materialization_facade]") {
  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_mapped_sources_direct_fallback";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  auto stats = std::make_shared<RecordingDirectSourceStats>();
  std::vector<uint8_t> source_bytes{11, 12, 13, 14, 15, 16, 17, 18, 201, 202, 203, 204, 205, 206, 207, 208};
  auto loader = std::make_unique<RecordingDirectLoader>(source_bytes, stats);
  loader->set_readv_failure(absl::UnimplementedError("vectored direct write unavailable"));
  std::array<uint8_t, 8> target{};
  target.fill(0xEE);

  tensorcast::store::loader::ByteRangeMap mapping;
  mapping.total_bytes = target.size();
  mapping.num_sources = 1;
  mapping.segments = {
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 4,
          .src_offset = 0,
          .source_index = 0,
      },
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 4,
          .length = 4,
          .src_offset = 8,
          .source_index = 0,
      },
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{static_cast<void*>(target.data())}, .length = target.size()});
  target_layout.total_size = target.size();

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:mapped-sources-direct-fallback";

  auto sources_or = open_single_loader_sources(std::move(loader), mapping, "mapped-sources-direct-fallback-test");
  REQUIRE(sources_or.ok());
  auto result_or = harness.facade->materialize_mapped_sources_into_target(
      DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      target_layout,
      std::move(*sources_or),
      mapping,
      hints,
      loading::MaterializationSource::kLocalReplica);
  REQUIRE(result_or.ok());
  CHECK(result_or->direct_write_supported);
  REQUIRE(result_or->debug_stats.has_value());
  CHECK(result_or->debug_stats->produced_chunks == 1);
  CHECK(result_or->debug_stats->produced_bytes == target.size());
  CHECK(stats->readv_calls == 1);
  CHECK(stats->read_into_calls == 2);
  CHECK(stats->batch_sizes == std::vector<size_t>{2});
  const std::array<uint8_t, 8> expected_target{11, 12, 13, 14, 201, 202, 203, 204};
  CHECK(target == expected_target);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade mapped sources reject missing source handles", "[materialization_facade]") {
  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_mapped_sources_missing_source";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  std::array<uint8_t, 8> target{};
  target.fill(0xAB);

  tensorcast::store::loader::ByteRangeMap mapping;
  mapping.total_bytes = target.size();
  mapping.num_sources = 2;
  mapping.segments = {
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 4,
          .src_offset = 0,
          .source_index = 0,
      },
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 4,
          .length = 4,
          .src_offset = 0,
          .source_index = 1,
      },
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{static_cast<void*>(target.data())}, .length = target.size()});
  target_layout.total_size = target.size();

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:mapped-sources-missing-source";

  const auto payload = std::make_shared<std::array<uint8_t, 8>>(std::array<uint8_t, 8>{1, 2, 3, 4, 5, 6, 7, 8});
  auto single_source_or = open_inline_sources(
      payload,
      payload->size(),
      loading::build_identity_byte_range_map(payload->size()),
      "mapped-sources-single-source-test");
  REQUIRE(single_source_or.ok());
  auto result_or = harness.facade->materialize_mapped_sources_into_target(
      DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      target_layout,
      std::move(*single_source_or),
      mapping,
      hints,
      loading::MaterializationSource::kDisk);
  REQUIRE_FALSE(result_or.ok());
  CHECK(absl::IsInvalidArgument(result_or.status()));
  CHECK(result_or.status().message().find("sources do not satisfy map.num_sources") != std::string::npos);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade ingest mapped sources preserves target id semantics", "[materialization_facade]") {
  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_ingest_mapped_sources_ids";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  const auto payload =
      std::make_shared<std::array<uint8_t, 6>>(std::array<uint8_t, 6>{0x41, 0x42, 0x43, 0x44, 0x45, 0x46});
  const std::string logical_artifact_id = "cgid:logical_ingest_mapped_sources";
  const std::string physical_artifact_id = "__tc_body__:ingest_mapped_sources";

  loading::ReplicaTarget target;
  target.location.type = MemoryLocation::CPU;
  target.location.device_id = -1;

  tensorcast::store::loader::ByteRangeMap mapping;
  mapping.total_bytes = payload->size();
  mapping.num_sources = 1;
  mapping.segments = {
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = payload->size(),
          .src_offset = 0,
          .source_index = 0,
      },
  };

  loading::MaterializeHints hints;
  hints.artifact_id = logical_artifact_id;

  auto sources_or = open_single_loader_sources(
      std::make_unique<tensorcast::store::InlineBufferLoader>(
          loading::InlineBufferSource{.data = payload, .size_bytes = payload->size()}),
      mapping,
      "ingest-mapped-sources-ids-test");
  REQUIRE(sources_or.ok());
  auto result_or = harness.facade->ingest_mapped_sources_into_replicas(
      {
          MaterializationFacade::MappedReplicaTarget{
              .logical_artifact_id = logical_artifact_id,
              .physical_artifact_id = physical_artifact_id,
              .target_device = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
              .target = target,
              .size_bytes = mapping.total_bytes,
          },
      },
      std::move(*sources_or),
      mapping,
      hints,
      loading::MaterializationSource::kDisk);
  REQUIRE(result_or.ok());
  REQUIRE(result_or->replica_handles.size() == 1);
  loading::ReplicaHandle handle = std::move(result_or->replica_handles.front());
  CHECK(handle.key().artifact_id == physical_artifact_id);
  CHECK(handle.source == loading::MaterializationSource::kDisk);
  CHECK(handle.cpu_state == tensorcast::store::replica::MemoryState::LOADED);
  CHECK(handle.wait_ready(std::chrono::milliseconds(100)).ok());

  auto replica_or = harness.replica_runtime().registry().find(handle.key());
  REQUIRE(replica_or.ok());
  CHECK(replica_or.value()->get_memory_state(MemoryLocation::CPU) == tensorcast::store::replica::MemoryState::LOADED);
  auto cpu_ptrs = replica_or.value()->get_data_pointer(MemoryLocation::CPU);
  REQUIRE(cpu_ptrs.size() == 1);
  auto* cpu_ptr = static_cast<const uint8_t*>(cpu_ptrs.front());
  REQUIRE(cpu_ptr != nullptr);
  for (size_t index = 0; index < payload->size(); ++index) {
    CHECK(cpu_ptr[index] == (*payload)[index]);
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade ingest mapped sources rejects missing source handles", "[materialization_facade]") {
  auto temp_root =
      std::filesystem::temp_directory_path() / "materialization_facade_ingest_mapped_sources_missing_source";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  const auto payload = std::make_shared<std::array<uint8_t, 8>>(std::array<uint8_t, 8>{1, 2, 3, 4, 5, 6, 7, 8});

  tensorcast::store::loader::ByteRangeMap mapping;
  mapping.total_bytes = payload->size();
  mapping.num_sources = 2;
  mapping.segments = {
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 4,
          .src_offset = 0,
          .source_index = 0,
      },
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 4,
          .length = 4,
          .src_offset = 0,
          .source_index = 1,
      },
  };

  loading::ReplicaTarget target;
  target.location.type = MemoryLocation::CPU;
  target.location.device_id = -1;

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:ingest-mapped-sources-missing-source";

  auto sources_or = open_inline_sources(
      payload,
      payload->size(),
      loading::build_identity_byte_range_map(payload->size()),
      "ingest-mapped-sources-single-source-test");
  REQUIRE(sources_or.ok());
  auto result_or = harness.facade->ingest_mapped_sources_into_replicas(
      {
          MaterializationFacade::MappedReplicaTarget{
              .logical_artifact_id = "cgid:logical_ingest_mapped_sources_missing_source",
              .physical_artifact_id = "__tc_body__:ingest_mapped_sources_missing_source",
              .target_device = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
              .target = target,
              .size_bytes = mapping.total_bytes,
          },
      },
      std::move(*sources_or),
      mapping,
      hints,
      loading::MaterializationSource::kDisk);
  REQUIRE_FALSE(result_or.ok());
  CHECK(absl::IsInvalidArgument(result_or.status()));
  CHECK(result_or.status().message().find("sources do not satisfy map.num_sources") != std::string::npos);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

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

TEST_CASE(
    "MaterializationFacade execute_artifact_lowering_plan returns verified content for replica staging",
    "[materialization_facade][artifact_lowering]") {
  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_artifact_lowering";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  const auto payload = std::make_shared<std::string>("body-bytes-for-verification");
  loading::ReplicaTarget target;
  target.location.type = MemoryLocation::CPU;
  target.location.device_id = -1;
  tensorcast::common::SelectionIdentity selection_identity{
      .artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azE",
      .logical_layout_hash = tensorcast::common::compute_byte_artifact_logical_layout_hash_bytes(),
      .selection_hash = tensorcast::common::compute_byte_artifact_selection_hash_bytes(),
  };
  auto plan_or = lower_to_artifact_plan(
      tensorcast::store::runtime::ingestion::LowerToArtifactPlanRequest{
          .identity =
              tensorcast::store::runtime::ingestion::ArtifactLoweringIdentity{
                  .logical_artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azE",
                  .physical_artifact_id = "__tc_body__:verified",
              },
          .target_device = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
          .source_loader = std::make_unique<tensorcast::store::InlineBufferLoader>(loading::InlineBufferSource{
              .data = std::shared_ptr<const void>(payload, static_cast<const void*>(payload->data())),
              .size_bytes = payload->size(),
          }),
          .selection_identity = selection_identity,
          .expected_size_bytes = payload->size(),
          .generation = 1,
          .source_kind = loading::MaterializationSource::kLocalReplica,
          .replica_target = target,
      });
  REQUIRE(plan_or.ok());
  ArtifactLoweringPlan plan = std::move(*plan_or);

  auto result_or = harness.facade->execute_artifact_lowering_plan(std::move(plan));
  REQUIRE(result_or.ok());
  REQUIRE(result_or->replica_handle.has_value());
  REQUIRE(result_or->selection_identity.has_value());
  REQUIRE(result_or->resolved_source_descriptor.has_value());
  REQUIRE(result_or->verified_content_descriptor.has_value());
  REQUIRE(result_or->verification_record.has_value());
  REQUIRE(result_or->backing_identity.has_value());
  CHECK(*result_or->selection_identity == selection_identity);
  CHECK(result_or->resolved_source_descriptor->exact_size_bytes == payload->size());
  CHECK(result_or->verified_content_descriptor->content_identity.logical_size_bytes == payload->size());
  CHECK(result_or->verified_content_descriptor->content_identity.digest_alg == "sha256");
  CHECK(
      tensorcast::store::runtime::ingestion::content_digest_bytes_to_hex(
          result_or->verified_content_descriptor->content_identity.digest_bytes) == sha256_hex(*payload));
  CHECK(result_or->verification_record->verified_at != absl::InfinitePast());
  CHECK(result_or->backing_identity->physical_artifact_id == "__tc_body__:verified");
  CHECK(result_or->backing_identity->replica_key.artifact_id == "__tc_body__:verified");
  CHECK_FALSE(result_or->backing_identity->replica_key.view_id.has_value());
  CHECK(result_or->backing_identity->replica_key.replica == 0);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "lower_to_artifact_plan enforces exact source size for identity-map staging",
    "[materialization_facade][artifact_lowering]") {
  const auto payload = std::make_shared<std::string>("size-check");
  loading::ReplicaTarget target;
  target.location.type = MemoryLocation::CPU;
  target.location.device_id = -1;

  auto plan_or = lower_to_artifact_plan(
      tensorcast::store::runtime::ingestion::LowerToArtifactPlanRequest{
          .identity =
              tensorcast::store::runtime::ingestion::ArtifactLoweringIdentity{
                  .logical_artifact_id = "cgid:byte_artifact~tenant~engine~b64u.c2l6ZQ~layout_v1~b64u.azEy",
                  .physical_artifact_id = "__tc_body__:size_check",
              },
          .target_device = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
          .source_loader = std::make_unique<tensorcast::store::InlineBufferLoader>(loading::InlineBufferSource{
              .data = std::shared_ptr<const void>(payload, static_cast<const void*>(payload->data())),
              .size_bytes = payload->size(),
          }),
          .expected_size_bytes = payload->size() + 1,
          .generation = 1,
          .source_kind = loading::MaterializationSource::kLocalReplica,
          .replica_target = target,
      });
  REQUIRE_FALSE(plan_or.ok());
  CHECK(absl::IsFailedPrecondition(plan_or.status()));
}

TEST_CASE("Shared content identity is independent of artifact id", "[materialization_facade][artifact_lowering]") {
  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_content_identity";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  const auto payload = std::make_shared<std::string>("same-content-different-artifact-id");
  loading::ReplicaTarget target;
  target.location.type = MemoryLocation::CPU;
  target.location.device_id = -1;

  const auto execute = [&](std::string_view logical_artifact_id, std::string_view physical_artifact_id) {
    auto plan_or = lower_to_artifact_plan(
        tensorcast::store::runtime::ingestion::LowerToArtifactPlanRequest{
            .identity =
                tensorcast::store::runtime::ingestion::ArtifactLoweringIdentity{
                    .logical_artifact_id = std::string(logical_artifact_id),
                    .physical_artifact_id = std::string(physical_artifact_id),
                },
            .target_device = DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
            .source_loader = std::make_unique<tensorcast::store::InlineBufferLoader>(loading::InlineBufferSource{
                .data = std::shared_ptr<const void>(payload, static_cast<const void*>(payload->data())),
                .size_bytes = payload->size(),
            }),
            .semantic_layout_identity =
                tensorcast::store::runtime::ingestion::SemanticLayoutIdentity{
                    .kind = tensorcast::store::runtime::ingestion::SemanticLayoutKind::kNamedLayoutId,
                    .value = "layout_v1",
                },
            .expected_size_bytes = payload->size(),
            .generation = 1,
            .source_kind = loading::MaterializationSource::kLocalReplica,
            .replica_target = target,
        });
    REQUIRE(plan_or.ok());
    return harness.facade->execute_artifact_lowering_plan(std::move(*plan_or));
  };

  auto first_or =
      execute("cgid:byte_artifact~tenant~engine~b64u.Zmlyc3Q~layout_v1~b64u.azEz", "__tc_body__:content_identity_a");
  auto second_or =
      execute("cgid:byte_artifact~tenant~engine~b64u.c2Vjb25k~layout_v1~b64u.azE0", "__tc_body__:content_identity_b");
  REQUIRE(first_or.ok());
  REQUIRE(second_or.ok());
  REQUIRE(first_or->verified_content_descriptor.has_value());
  REQUIRE(second_or->verified_content_descriptor.has_value());

  CHECK(
      first_or->verified_content_descriptor->content_identity.semantic_layout_identity ==
      second_or->verified_content_descriptor->content_identity.semantic_layout_identity);
  CHECK(
      first_or->verified_content_descriptor->content_identity.logical_size_bytes ==
      second_or->verified_content_descriptor->content_identity.logical_size_bytes);
  CHECK(
      first_or->verified_content_descriptor->content_identity.digest_alg ==
      second_or->verified_content_descriptor->content_identity.digest_alg);
  CHECK(
      first_or->verified_content_descriptor->content_identity.digest_bytes ==
      second_or->verified_content_descriptor->content_identity.digest_bytes);

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
    hints.export_policy = loading::ExportPolicy::kForce;
    loading::ReplicaHandle handle;
    handle.replica_key.artifact_id = "p2p_success";
    harness.fake_pipeline->set_next_p2p_result(std::move(handle));
    static_cast<void>(recorder.drain_started());
    static_cast<void>(recorder.drain_completed());
    auto handle_or =
        harness.facade->ingest_from_p2p("p2p_success", source, target, hints, /*publish_to_global_store=*/true);
    REQUIRE(handle_or.ok());
    harness.runtime_context().drain_events();
    auto started_events = recorder.drain_started();
    auto completed_events = recorder.drain_completed();
    REQUIRE(started_events.size() == 1);
    REQUIRE(completed_events.size() == 1);
    CHECK(started_events.back().export_policy == loading::ExportPolicy::kForce);
    CHECK(completed_events.back().artifact_id == "p2p_success");
    CHECK(completed_events.back().export_policy == loading::ExportPolicy::kForce);
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
  hints.artifact_id = "cgid:artifact_auto";
  loading::DiskSource disk_source{.path = temp_root / "artifact_auto", .expected_size = std::nullopt};
  loading::ReplicaHandle handle;
  handle.replica_key.artifact_id = hints.artifact_id;
  handle.replica_key.device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  harness.fake_pipeline->set_next_disk_result(std::move(handle));

  loading::MaterializeHints request_hints = hints;

  auto handle_or = harness.facade->materialize_replica(
      target.location.to_device_key(), loading::MaterializeMode::LOAD_ONLY, request_hints, disk_source);
  REQUIRE(handle_or.ok());
  CHECK(handle_or->replica_key.artifact_id == hints.artifact_id);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade strategy plan requires explicit shared-source proof for owner-file collective",
    "[materialization_facade][strategy_plan]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_strategy_unproven");
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[64],\"data_offsets\":[0,64]}}",
      std::vector<unsigned char>(64, 7));

  auto opts = MakeOptions(artifact_root);
  opts.materialization_strategy.enable_local_batched_disk_load = true;
  opts.materialization_strategy.enable_owner_file_collective = true;
  opts.materialization_strategy.owner_file_collective_min_dedup_saving_bytes = 1;

  FacadeHarness harness(opts);
  harness.initialize();

  auto ctx = make_strategy_context(harness, artifact_root, loading::SourceLocalityHint::kAuto, std::nullopt);
  auto plan_or = harness.facade->build_ordinary_disk_execution_strategy_plan_for_testing(ctx);
  REQUIRE(plan_or.ok());

  REQUIRE(
      plan_or->executor ==
      tensorcast::store::runtime::ingestion::strategy::ExecutionStrategyExecutor::kTensorBatchedLocal);
  REQUIRE(plan_or->selection_reason == "auto_prefers_local_batched");
  REQUIRE(plan_or->candidates.size() == 3);
  REQUIRE(plan_or->candidates[2].eligible == false);
  REQUIRE(plan_or->candidates[2].reason == "shared_source_unproven");

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade identity disk loads prefer local batched without an explicit variant",
    "[materialization_facade][strategy_plan]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_strategy_identity_no_variant");
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[64],\"data_offsets\":[0,64]}}",
      std::vector<unsigned char>(64, 13));

  auto opts = MakeOptions(artifact_root);
  opts.materialization_strategy.enable_local_batched_disk_load = true;
  opts.materialization_strategy.enable_owner_file_collective = false;

  FacadeHarness harness(opts);
  harness.initialize();

  auto ctx = make_strategy_context(harness, artifact_root, loading::SourceLocalityHint::kAuto, std::nullopt);
  ctx.hints.variant.reset();
  auto plan_or = harness.facade->build_ordinary_disk_execution_strategy_plan_for_testing(ctx);
  REQUIRE(plan_or.ok());

  REQUIRE(
      plan_or->executor ==
      tensorcast::store::runtime::ingestion::strategy::ExecutionStrategyExecutor::kTensorBatchedLocal);
  REQUIRE(plan_or->selection_reason == "auto_prefers_local_batched");

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade AUTO prefers local batched even for explicit shared source",
    "[materialization_facade][strategy_plan]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_strategy_shared");
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[64],\"data_offsets\":[0,64]}}",
      std::vector<unsigned char>(64, 11));

  auto opts = MakeOptions(artifact_root);
  opts.materialization_strategy.enable_local_batched_disk_load = true;
  opts.materialization_strategy.enable_owner_file_collective = true;
  opts.materialization_strategy.owner_file_collective_min_dedup_saving_bytes = 1;

  FacadeHarness harness(opts);
  harness.initialize();

  auto ctx = make_strategy_context(
      harness, artifact_root, loading::SourceLocalityHint::kSharedSource, std::string("shared-fs:test"));
  auto plan_or = harness.facade->build_ordinary_disk_execution_strategy_plan_for_testing(ctx);
  REQUIRE(plan_or.ok());

  REQUIRE(
      plan_or->executor ==
      tensorcast::store::runtime::ingestion::strategy::ExecutionStrategyExecutor::kTensorBatchedLocal);
  REQUIRE(plan_or->selection_reason == "auto_prefers_local_batched");
  REQUIRE(plan_or->candidates.size() == 3);
  REQUIRE(plan_or->candidates[2].eligible == true);
  REQUIRE(plan_or->candidates[2].reason == "eligible");

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade explicit owner-file preference selects collective for shared source",
    "[materialization_facade][strategy_plan]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_strategy_explicit_collective");
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[64],\"data_offsets\":[0,64]}}",
      std::vector<unsigned char>(64, 17));

  auto opts = MakeOptions(artifact_root);
  opts.materialization_strategy.enable_local_batched_disk_load = true;
  opts.materialization_strategy.enable_owner_file_collective = true;
  opts.materialization_strategy.executor_preference =
      tensorcast::store::StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kOwnerFileCollective;
  opts.materialization_strategy.owner_file_collective_min_dedup_saving_bytes = 1;

  FacadeHarness harness(opts);
  harness.initialize();

  auto ctx = make_strategy_context(
      harness, artifact_root, loading::SourceLocalityHint::kSharedSource, std::string("shared-fs:test"));
  auto plan_or = harness.facade->build_ordinary_disk_execution_strategy_plan_for_testing(ctx);
  REQUIRE(plan_or.ok());

  REQUIRE(
      plan_or->executor ==
      tensorcast::store::runtime::ingestion::strategy::ExecutionStrategyExecutor::kOwnerFileCollective);
  REQUIRE(plan_or->selection_reason == "executor_preference_owner_file_collective");

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade AUTO keeps dim1-amplified host-local workloads on exact generic execution",
    "[materialization_facade][strategy_plan]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_strategy_dim1_auto");
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[4,16],\"data_offsets\":[0,64]}}",
      std::vector<unsigned char>(64, 5));

  auto opts = MakeOptions(artifact_root);
  opts.materialization_strategy.enable_local_batched_disk_load = true;
  opts.materialization_strategy.enable_owner_file_collective = false;

  FacadeHarness harness(opts);
  harness.initialize();

  auto ctx = make_strategy_context(harness, artifact_root, loading::SourceLocalityHint::kHostLocal, std::nullopt);
  REQUIRE(ctx.hints.variant.has_value());
  ctx.hints.variant->view_id = "view:dim1";
  ctx.hints.variant->view_spec = view_contracts::ViewSpec{
      .tensors =
          {
              {
                  "tensor",
                  view_contracts::TensorViewOps{
                      .ops =
                          {
                              view_contracts::ViewOp::Narrow(
                                  view_contracts::NarrowOp{.dim = 1, .start = 0, .length = 4}),
                          },
                  },
              },
          },
  };
  ctx.resolved_view_plan = view_contracts::ViewPlan{
      .is_identity = false,
      .view_size_bytes = 16,
      .view_index_json = R"({"tensor":[0,16,[4,4],[4,1],"torch.uint8",0]})",
  };

  auto plan_or = harness.facade->build_ordinary_disk_execution_strategy_plan_for_testing(ctx);
  REQUIRE(plan_or.ok());

  REQUIRE(
      plan_or->executor ==
      tensorcast::store::runtime::ingestion::strategy::ExecutionStrategyExecutor::kGenericByteRange);
  REQUIRE(plan_or->candidates.size() == 3);
  CHECK(plan_or->candidates[1].eligible);
  CHECK(plan_or->candidates[1].reason == "eligible");
  CHECK(plan_or->selection_reason.find("local_source_amplification_exceeds_generic") != std::string::npos);

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade explicit tensor-aware-local preference still allows dim1-amplified workloads",
    "[materialization_facade][strategy_plan]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_strategy_dim1_force_local");
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[4,16],\"data_offsets\":[0,64]}}",
      std::vector<unsigned char>(64, 9));

  auto opts = MakeOptions(artifact_root);
  opts.materialization_strategy.enable_local_batched_disk_load = true;
  opts.materialization_strategy.enable_owner_file_collective = false;
  opts.materialization_strategy.executor_preference =
      StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kTensorAwareLocal;

  FacadeHarness harness(opts);
  harness.initialize();

  auto ctx = make_strategy_context(harness, artifact_root, loading::SourceLocalityHint::kHostLocal, std::nullopt);
  REQUIRE(ctx.hints.variant.has_value());
  ctx.hints.variant->view_id = "view:dim1";
  ctx.hints.variant->view_spec = view_contracts::ViewSpec{
      .tensors =
          {
              {
                  "tensor",
                  view_contracts::TensorViewOps{
                      .ops =
                          {
                              view_contracts::ViewOp::Narrow(
                                  view_contracts::NarrowOp{.dim = 1, .start = 0, .length = 4}),
                          },
                  },
              },
          },
  };
  ctx.resolved_view_plan = view_contracts::ViewPlan{
      .is_identity = false,
      .view_size_bytes = 16,
      .view_index_json = R"({"tensor":[0,16,[4,4],[4,1],"torch.uint8",0]})",
  };

  auto plan_or = harness.facade->build_ordinary_disk_execution_strategy_plan_for_testing(ctx);
  REQUIRE(plan_or.ok());

  REQUIRE(
      plan_or->executor ==
      tensorcast::store::runtime::ingestion::strategy::ExecutionStrategyExecutor::kTensorBatchedLocal);
  CHECK(plan_or->selection_reason == "executor_preference_tensor_aware_local");

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade AUTO falls back when Global Store route is stale", "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_auto_stale";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  WorkerIdentity identity;
  identity.worker_id = "worker-0";
  // Match RecordingGlobalStoreClient's default transport session so the route is treated as local.
  identity.node_id = "stub-node";
  identity.node_address = "127.0.0.1";
  identity.grpc_port = 9001;
  identity.p2p_port = 12345;
  harness.runtime_context().set_worker_identity(identity);

  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->allow_replica_transport = true;
  harness.runtime_context().set_global_store_client_for_testing(gs_client);

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:artifact_auto";
  hints.request_budget = std::chrono::milliseconds(1234);
  hints.transport_wait_timeout = std::chrono::milliseconds(1234);
  loading::DiskSource disk_source{.path = temp_root / "artifact_fallback", .expected_size = std::nullopt};

  loading::ReplicaHandle disk_handle;
  disk_handle.replica_key.artifact_id = hints.artifact_id;
  disk_handle.replica_key.device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  harness.fake_pipeline->set_next_disk_result(std::move(disk_handle));
  harness.fake_pipeline->set_next_p2p_result(absl::FailedPreconditionError("stale p2p route"));

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto handle_or =
      harness.facade->materialize_replica(target_device, loading::MaterializeMode::AUTO, hints, disk_source);
  REQUIRE(handle_or.ok());
  REQUIRE(harness.fake_pipeline->disk_invocations().size() == 1);
  REQUIRE_FALSE(harness.fake_pipeline->p2p_invocations().empty());
  REQUIRE(gs_client->replica_requests.size() >= 1);
  REQUIRE(gs_client->replica_request_wait_timeouts_ms.size() == gs_client->replica_requests.size());
  for (uint32_t timeout_ms : gs_client->replica_request_wait_timeouts_ms) {
    REQUIRE(timeout_ms <= 1234);
    REQUIRE(timeout_ms > 0);
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade AUTO treats same-node different daemon route as remote", "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_auto_same_node_remote";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  WorkerIdentity identity;
  identity.worker_id = "worker-0";
  identity.node_id = "stub-node";
  identity.node_address = "127.0.0.1";
  identity.grpc_port = 9001;
  identity.p2p_port = 22345;
  harness.runtime_context().set_worker_identity(identity);

  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->allow_replica_transport = true;
  gs_client->remote_node_id = "stub-node";
  gs_client->remote_node_address = "127.0.0.1";
  gs_client->remote_node_port = 12345;
  harness.runtime_context().set_global_store_client_for_testing(gs_client);

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:artifact_same_node_remote";

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  loading::ReplicaHandle p2p_handle;
  p2p_handle.replica_key.artifact_id = hints.artifact_id;
  p2p_handle.replica_key.device = target_device;
  harness.fake_pipeline->set_next_p2p_result(std::move(p2p_handle));

  auto handle_or = harness.facade->materialize_replica(target_device, loading::MaterializeMode::AUTO, hints);
  REQUIRE(handle_or.ok());
  CHECK(harness.fake_pipeline->disk_invocations().empty());
  CHECK(harness.fake_pipeline->p2p_invocations().size() == 1);
  CHECK(gs_client->replica_requests.size() == 1);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade AUTO uses disk when Global Store disconnected", "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_auto_no_gs";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->connected = false;
  gs_client->allow_replica_transport = true;
  harness.runtime_context().set_global_store_client_for_testing(gs_client);

  loading::MaterializeHints hints;
  hints.artifact_id = "mi2:mh:index:mh:data";
  hints.source_preference = loading::SourcePreference::kPreferDisk;
  hints.allow_disk = true;
  hints.allow_p2p = false;

  loading::DiskSource disk_source{.path = temp_root / "artifact_no_gs", .expected_size = std::nullopt};

  loading::ReplicaHandle disk_handle;
  disk_handle.replica_key.artifact_id = hints.artifact_id;
  disk_handle.replica_key.device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  harness.fake_pipeline->set_next_disk_result(std::move(disk_handle));

  DeviceKey target_device{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  auto handle_or =
      harness.facade->materialize_replica(target_device, loading::MaterializeMode::AUTO, hints, disk_source);
  REQUIRE(handle_or.ok());
  CHECK(harness.fake_pipeline->disk_invocations().size() == 1);
  CHECK(harness.fake_pipeline->p2p_invocations().empty());
  CHECK(gs_client->replica_requests.empty());

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade AUTO serves view from local canonical replica", "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_local_view";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->allow_view_transport = true;
  harness.runtime_context().set_global_store_client_for_testing(gs_client);

  const std::string artifact_id = "cgid:artifact_local_view";
  constexpr uint64_t kCanonicalSize = 64;
  loading::InlineBufferSource source{.data = nullptr, .size_bytes = kCanonicalSize};
  tensorcast::store::replica::ReplicaConfig cfg{
      .source = source,
      .artifact_identifier = artifact_id,
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool = harness.runtime_context().pinned_buffer_pool(),
      .async_runtime =
          gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{harness.runtime_context().async_runtime()},
      .artifact_chunk_bytes = harness.options().artifact_chunk_bytes,
      .expected_artifact_size = kCanonicalSize,
  };
  auto canonical_or = tensorcast::store::replica::Replica::create(cfg);
  REQUIRE(canonical_or.ok());
  auto canonical_replica = std::shared_ptr<tensorcast::store::replica::Replica>(std::move(canonical_or.value()));

  CHECK_OK(canonical_replica->get_memory_manager().allocate_memory(MemoryLocation::CPU));
  CHECK_OK(canonical_replica->mark_loaded(MemoryLocation::CPU));
  canonical_replica->set_ready_signal(MemoryLocation::CPU, absl::OkStatus());

  loading::ReplicaKey canonical_key{
      .artifact_id = artifact_id,
      .view_id = std::nullopt,
      .device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      .replica = 0,
  };
  CHECK_OK(harness.replica_runtime().registry().emplace(canonical_key, gsl::not_null{canonical_replica}));

  loading::MaterializeHints hints;
  hints.artifact_id = artifact_id;
  loading::VariantIdentity variant;
  variant.canonical_artifact_id = artifact_id;
  variant.view_id = "view:local";
  tensorcast::store::loader::ViewPlan view_plan;
  view_plan.is_identity = false;
  view_plan.view_size_bytes = 16;
  view_plan.view_index_json = R"({"tensor":[0,16]})";
  view_plan.selection.map.total_bytes = 16;
  view_plan.selection.map.num_sources = 1;
  view_plan.selection.map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 16,
          .src_offset = 8,
          .source_index = 0,
      });
  variant.cached_plan = view_plan;
  hints.variant = std::move(variant);

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto handle_or = harness.facade->materialize_replica(target_device, loading::MaterializeMode::AUTO, hints);
  REQUIRE(handle_or.ok());
  CHECK(handle_or->source == loading::MaterializationSource::kLocalReplica);
  CHECK(handle_or->replica_key.view_id.has_value());
  CHECK(*handle_or->replica_key.view_id == "view:local");
  CHECK(harness.fake_pipeline->disk_invocations().empty());
  CHECK(harness.fake_pipeline->p2p_invocations().empty());
  CHECK(gs_client->view_requests.empty());
  CHECK(gs_client->replica_requests.empty());

  auto view_or = harness.replica_runtime().registry().find(handle_or->replica_key);
  REQUIRE(view_or.ok());
  CHECK(view_or.value()->get_memory_state(MemoryLocation::GPU) == tensorcast::store::replica::MemoryState::LOADED);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade materialize_into_target prefers local canonical replica", "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  constexpr uint64_t kCanonicalSize = 16;
  const std::string artifact_id = "cgid:artifact_local_into_target";
  constexpr std::string_view kCanonicalIndexJson = R"({"tensor":[0,16,[16],[1],"torch.uint8",0]})";

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_local_into_target";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->allow_replica_transport = true;
  harness.runtime_context().set_global_store_client_for_testing(gs_client);

  loading::InlineBufferSource source{.data = nullptr, .size_bytes = kCanonicalSize};
  tensorcast::store::replica::ReplicaConfig cfg{
      .source = source,
      .artifact_identifier = artifact_id,
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool = harness.runtime_context().pinned_buffer_pool(),
      .async_runtime =
          gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{harness.runtime_context().async_runtime()},
      .artifact_chunk_bytes = harness.options().artifact_chunk_bytes,
      .expected_artifact_size = kCanonicalSize,
  };
  auto canonical_or = tensorcast::store::replica::Replica::create(cfg);
  REQUIRE(canonical_or.ok());
  auto canonical_replica = std::shared_ptr<tensorcast::store::replica::Replica>(std::move(canonical_or.value()));

  CHECK_OK(canonical_replica->get_memory_manager().allocate_memory(MemoryLocation::CPU));
  auto cpu_ptrs = canonical_replica->get_data_pointer(MemoryLocation::CPU);
  REQUIRE(cpu_ptrs.size() == 1);
  auto* cpu_ptr = static_cast<uint8_t*>(cpu_ptrs.front());
  REQUIRE(cpu_ptr != nullptr);
  for (uint8_t i = 0; i < kCanonicalSize; ++i) {
    cpu_ptr[i] = i;
  }
  CHECK_OK(canonical_replica->mark_loaded(MemoryLocation::CPU));
  canonical_replica->set_ready_signal(MemoryLocation::CPU, absl::OkStatus());

  loading::ReplicaKey canonical_key{
      .artifact_id = artifact_id,
      .view_id = std::nullopt,
      .device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      .replica = 0,
  };
  CHECK_OK(harness.replica_runtime().registry().emplace(canonical_key, gsl::not_null{canonical_replica}));

  void* gpu_buffer = nullptr;
  auto alloc_status = tensorcast::cuda::malloc(&gpu_buffer, kCanonicalSize);
  REQUIRE(alloc_status.ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{.base_ptr = gsl::not_null<void*>{gpu_buffer}, .length = kCanonicalSize});
  target_layout.total_size = kCanonicalSize;

  loading::MaterializeHints hints;
  hints.artifact_id = artifact_id;

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or = harness.facade->materialize_into_target(
      target_device, target_layout, kCanonicalIndexJson, /*generation=*/1, hints, std::nullopt);
  REQUIRE(result_or.ok());
  CHECK(result_or->source == loading::MaterializationSource::kLocalReplica);
  CHECK(gs_client->replica_requests.empty());

  std::array<uint8_t, kCanonicalSize> host_out{};
  auto copy_status = tensorcast::cuda::memcpy(host_out.data(), gpu_buffer, kCanonicalSize, cudaMemcpyDeviceToHost);
  REQUIRE(copy_status.ok());
  for (uint8_t i = 0; i < kCanonicalSize; ++i) {
    CHECK(host_out[i] == i);
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade mapped target respects typed local canonical override", "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  constexpr uint64_t kCanonicalSize = 16;
  const std::string artifact_id = "cgid:artifact_local_mapped_override";
  constexpr std::string_view kCanonicalIndexJson = R"({"tensor":[0,16,[16],[1],"torch.uint8",0]})";

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_local_mapped_override";
  std::filesystem::create_directories(temp_root);

  auto opts = MakeOptions(temp_root);
  opts.materialization_strategy.prefer_local_canonical_for_mapped = true;
  FacadeHarness harness(opts);
  harness.initialize();

  loading::InlineBufferSource source{.data = nullptr, .size_bytes = kCanonicalSize};
  tensorcast::store::replica::ReplicaConfig cfg{
      .source = source,
      .artifact_identifier = artifact_id,
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool = harness.runtime_context().pinned_buffer_pool(),
      .async_runtime =
          gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{harness.runtime_context().async_runtime()},
      .artifact_chunk_bytes = harness.options().artifact_chunk_bytes,
      .expected_artifact_size = kCanonicalSize,
      .materialization_strategy = harness.options().materialization_strategy,
  };
  auto canonical_or = tensorcast::store::replica::Replica::create(cfg);
  REQUIRE(canonical_or.ok());
  auto canonical_replica = std::shared_ptr<tensorcast::store::replica::Replica>(std::move(canonical_or.value()));

  CHECK_OK(canonical_replica->get_memory_manager().allocate_memory(MemoryLocation::CPU));
  auto cpu_ptrs = canonical_replica->get_data_pointer(MemoryLocation::CPU);
  REQUIRE(cpu_ptrs.size() == 1);
  auto* cpu_ptr = static_cast<uint8_t*>(cpu_ptrs.front());
  REQUIRE(cpu_ptr != nullptr);
  for (uint8_t i = 0; i < kCanonicalSize; ++i) {
    cpu_ptr[i] = static_cast<uint8_t>(kCanonicalSize - i);
  }
  CHECK_OK(canonical_replica->mark_loaded(MemoryLocation::CPU));
  canonical_replica->set_ready_signal(MemoryLocation::CPU, absl::OkStatus());

  loading::ReplicaKey canonical_key{
      .artifact_id = artifact_id,
      .view_id = std::nullopt,
      .device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      .replica = 0,
  };
  CHECK_OK(harness.replica_runtime().registry().emplace(canonical_key, gsl::not_null{canonical_replica}));

  void* gpu_buffer = nullptr;
  auto alloc_status = tensorcast::cuda::malloc(&gpu_buffer, kCanonicalSize);
  REQUIRE(alloc_status.ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{.base_ptr = gsl::not_null<void*>{gpu_buffer}, .length = kCanonicalSize});
  target_layout.total_size = kCanonicalSize;

  tensorcast::store::loader::ByteRangeMap mapping;
  mapping.total_bytes = kCanonicalSize;
  mapping.num_sources = 1;
  mapping.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = kCanonicalSize,
          .src_offset = 0,
          .source_index = 0,
      });

  loading::MaterializeHints hints;
  hints.artifact_id = artifact_id;
  loading::DiskMetadata disk_metadata;
  disk_metadata.source_index_json = std::string(kCanonicalIndexJson);
  hints.disk_metadata = disk_metadata;
  hints.allow_disk = false;
  hints.allow_p2p = false;

  ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = artifact_id;
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = kCanonicalIndexJson;
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan =
      tensorcast::store::materialization::contracts::RepresentationWorkPlan{.residual_fallback_map = mapping};
  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan, mapping);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kGenericOnly,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .generic_backend_map = mapping,
      },
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "generic_only",
          .planned_generic_residual_bytes = kCanonicalSize,
      });

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or =
      harness.facade->materialize_mapped_into_target(target_device, prepared_execution, hints, std::nullopt);
  REQUIRE(result_or.ok());
  CHECK(result_or->source == loading::MaterializationSource::kLocalReplica);

  std::array<uint8_t, kCanonicalSize> host_out{};
  auto copy_status = tensorcast::cuda::memcpy(host_out.data(), gpu_buffer, kCanonicalSize, cudaMemcpyDeviceToHost);
  REQUIRE(copy_status.ok());
  for (uint8_t i = 0; i < kCanonicalSize; ++i) {
    CHECK(host_out[i] == static_cast<uint8_t>(kCanonicalSize - i));
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade reports composite source-resolution gaps for mapped targets",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  constexpr uint64_t kCanonicalSize = 16;
  const std::string artifact_id = "cgid:artifact_local_mapped_composite_gap";
  constexpr std::string_view kCanonicalIndexJson = R"({"tensor":[0,16,[16],[1],"torch.uint8",0]})";

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_local_mapped_composite_gap";
  std::filesystem::create_directories(temp_root);

  auto opts = MakeOptions(temp_root);
  opts.materialization_strategy.prefer_local_canonical_for_mapped = true;
  FacadeHarness harness(opts);
  harness.initialize();

  loading::InlineBufferSource source{.data = nullptr, .size_bytes = kCanonicalSize};
  tensorcast::store::replica::ReplicaConfig cfg{
      .source = source,
      .artifact_identifier = artifact_id,
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool = harness.runtime_context().pinned_buffer_pool(),
      .async_runtime =
          gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{harness.runtime_context().async_runtime()},
      .artifact_chunk_bytes = harness.options().artifact_chunk_bytes,
      .expected_artifact_size = kCanonicalSize,
      .materialization_strategy = harness.options().materialization_strategy,
  };
  auto canonical_or = tensorcast::store::replica::Replica::create(cfg);
  REQUIRE(canonical_or.ok());
  auto canonical_replica = std::shared_ptr<tensorcast::store::replica::Replica>(std::move(canonical_or.value()));

  CHECK_OK(canonical_replica->get_memory_manager().allocate_memory(MemoryLocation::CPU));
  auto cpu_ptrs = canonical_replica->get_data_pointer(MemoryLocation::CPU);
  REQUIRE(cpu_ptrs.size() == 1);
  auto* cpu_ptr = static_cast<uint8_t*>(cpu_ptrs.front());
  REQUIRE(cpu_ptr != nullptr);
  for (uint8_t i = 0; i < kCanonicalSize; ++i) {
    cpu_ptr[i] = i;
  }
  CHECK_OK(canonical_replica->mark_loaded(MemoryLocation::CPU));
  canonical_replica->set_ready_signal(MemoryLocation::CPU, absl::OkStatus());

  loading::ReplicaKey canonical_key{
      .artifact_id = artifact_id,
      .view_id = std::nullopt,
      .device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      .replica = 0,
  };
  CHECK_OK(harness.replica_runtime().registry().emplace(canonical_key, gsl::not_null{canonical_replica}));

  void* gpu_buffer = nullptr;
  auto alloc_status = tensorcast::cuda::malloc(&gpu_buffer, kCanonicalSize);
  REQUIRE(alloc_status.ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{.base_ptr = gsl::not_null<void*>{gpu_buffer}, .length = kCanonicalSize});
  target_layout.total_size = kCanonicalSize;

  tensorcast::store::loader::ByteRangeMap mapping;
  mapping.total_bytes = kCanonicalSize;
  mapping.num_sources = 2;
  mapping.segments = {
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 8,
          .src_offset = 0,
          .source_index = 0,
      },
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 8,
          .length = 8,
          .src_offset = 0,
          .source_index = 1,
      },
  };

  loading::MaterializeHints hints;
  hints.artifact_id = artifact_id;
  hints.allow_disk = false;
  hints.allow_p2p = false;

  ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = artifact_id;
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = kCanonicalIndexJson;
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan =
      tensorcast::store::materialization::contracts::RepresentationWorkPlan{.residual_fallback_map = mapping};
  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan, mapping);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kGenericOnly,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .generic_backend_map = mapping,
      },
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "generic_only",
          .planned_generic_residual_bytes = kCanonicalSize,
      });

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or =
      harness.facade->materialize_mapped_into_target(target_device, prepared_execution, hints, std::nullopt);
  REQUIRE_FALSE(result_or.ok());
  CHECK(absl::IsFailedPrecondition(result_or.status()));
  CHECK(result_or.status().message().find("resolved 1 source(s) but execution maps require 2") != std::string::npos);
  CHECK(result_or.status().message().find("mapping.num_sources == 1") == std::string::npos);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade rejects mapped-target artifact drift against resolved plan",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{.base_ptr = gsl::not_null<void*>{reinterpret_cast<void*>(0x1)}, .length = 16});
  target_layout.total_size = 16;

  ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:artifact_authoritative";
  resolved_plan.generation = 7;
  resolved_plan.canonical_index_json = R"({"tensor":[0,16,[16],[1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  const auto mapping = tensorcast::store::loader::ByteRangeMap{
      .total_bytes = 16,
      .num_sources = 1,
      .segments = {tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 16,
          .src_offset = 0,
          .source_index = 0,
      }},
  };
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan =
      tensorcast::store::materialization::contracts::RepresentationWorkPlan{.residual_fallback_map = mapping};

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:artifact_drifted";
  hints.allow_disk = false;
  hints.allow_p2p = false;

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_authoritative_plan";
  std::filesystem::create_directories(temp_root);
  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or = harness.facade->materialize_mapped_into_target(
      target_device, make_prepared_source_bound_execution_plan(resolved_plan), hints, std::nullopt);
  REQUIRE_FALSE(result_or.ok());
  CHECK(absl::IsInvalidArgument(result_or.status()));
  CHECK(result_or.status().message().find("hints.artifact_id does not match resolved_plan") != std::string::npos);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade executes mapped const fill without source bytes", "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  constexpr uint64_t kTotalSize = 16;
  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_mapped_const_fill";
  std::filesystem::create_directories(temp_root);
  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  void* gpu_buffer = nullptr;
  auto alloc_status = tensorcast::cuda::malloc(&gpu_buffer, kTotalSize);
  REQUIRE(alloc_status.ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{.base_ptr = gsl::not_null<void*>{gpu_buffer}, .length = kTotalSize});
  target_layout.total_size = kTotalSize;

  ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:artifact_const_fill";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"tensor":[0,16,[8],[1],"torch.uint16",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan = tensorcast::store::materialization::contracts::RepresentationWorkPlan{
      .items =
          {
              tensorcast::store::materialization::contracts::RepresentationWorkItem{
                  .kind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kConstFill,
                  .dst_name = "tensor",
                  .dst_spec =
                      tensorcast::store::materialization::contracts::RepresentationTensorSpec{
                          .name = "tensor",
                          .shape = {8},
                          .stride = {1},
                          .dtype = "torch.uint16",
                          .logical_offset = 0,
                          .logical_length = 16,
                          .storage_offset = 0,
                          .element_size = 2,
                      },
                  .fill_rule =
                      tensorcast::store::materialization::contracts::FillRule{
                          .constant_value = {0x34, 0x12},
                      },
                  .committed_bytes = 16,
              },
          },
  };

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:artifact_const_fill";
  hints.allow_disk = false;
  hints.allow_p2p = false;

  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kLocalTypedOnly,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .local_typed_bytes = kTotalSize,
          .local_fill_bytes = kTotalSize,
      },
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "local_typed_only",
          .planned_local_typed_bytes = kTotalSize,
      });

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or =
      harness.facade->materialize_mapped_into_target(target_device, prepared_execution, hints, std::nullopt);
  REQUIRE(result_or.ok());

  std::array<uint8_t, kTotalSize> host_out{};
  auto copy_status = tensorcast::cuda::memcpy(host_out.data(), gpu_buffer, kTotalSize, cudaMemcpyDeviceToHost);
  REQUIRE(copy_status.ok());
  for (size_t index = 0; index < host_out.size(); index += 2) {
    CHECK(host_out[index] == 0x34);
    CHECK(host_out[index + 1] == 0x12);
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade executes mixed mapped source bytes and const fill without byte-range gaps",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  constexpr uint64_t kSourceSize = 8;
  constexpr uint64_t kTargetSize = 12;
  const std::string artifact_id = "cgid:artifact_mixed_copy_const_fill";
  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_mixed_copy_const_fill";
  std::filesystem::create_directories(temp_root);
  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  loading::InlineBufferSource source{.data = nullptr, .size_bytes = kSourceSize};
  tensorcast::store::replica::ReplicaConfig cfg{
      .source = source,
      .artifact_identifier = artifact_id,
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool = harness.runtime_context().pinned_buffer_pool(),
      .async_runtime =
          gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{harness.runtime_context().async_runtime()},
      .artifact_chunk_bytes = harness.options().artifact_chunk_bytes,
      .expected_artifact_size = kSourceSize,
      .materialization_strategy = harness.options().materialization_strategy,
  };
  auto canonical_or = tensorcast::store::replica::Replica::create(cfg);
  REQUIRE(canonical_or.ok());
  auto canonical_replica = std::shared_ptr<tensorcast::store::replica::Replica>(std::move(canonical_or.value()));
  CHECK_OK(canonical_replica->get_memory_manager().allocate_memory(MemoryLocation::CPU));
  auto cpu_ptrs = canonical_replica->get_data_pointer(MemoryLocation::CPU);
  REQUIRE(cpu_ptrs.size() == 1);
  auto* cpu_ptr = static_cast<uint8_t*>(cpu_ptrs.front());
  REQUIRE(cpu_ptr != nullptr);
  for (uint8_t i = 0; i < kSourceSize; ++i) {
    cpu_ptr[i] = static_cast<uint8_t>(0x10 + i);
  }
  CHECK_OK(canonical_replica->mark_loaded(MemoryLocation::CPU));
  canonical_replica->set_ready_signal(MemoryLocation::CPU, absl::OkStatus());
  loading::ReplicaKey canonical_key{
      .artifact_id = artifact_id,
      .view_id = std::nullopt,
      .device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      .replica = 0,
  };
  CHECK_OK(harness.replica_runtime().registry().emplace(canonical_key, gsl::not_null{canonical_replica}));

  void* gpu_buffer = nullptr;
  auto alloc_status = tensorcast::cuda::malloc(&gpu_buffer, kTargetSize);
  REQUIRE(alloc_status.ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };

  std::array<uint8_t, kTargetSize> initial{};
  initial.fill(0xAA);
  auto init_status = tensorcast::cuda::memcpy(gpu_buffer, initial.data(), kTargetSize, cudaMemcpyHostToDevice);
  REQUIRE(init_status.ok());

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{.base_ptr = gsl::not_null<void*>{gpu_buffer}, .length = kTargetSize});
  target_layout.total_size = kTargetSize;

  tensorcast::store::loader::ByteRangeMap mapping;
  mapping.total_bytes = kTargetSize;
  mapping.num_sources = 1;
  mapping.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = kSourceSize,
          .src_offset = 0,
          .source_index = 0,
      });

  ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = artifact_id;
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"prefix":[0,8,[8],[1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan = tensorcast::store::materialization::contracts::RepresentationWorkPlan{
      .items =
          {
              tensorcast::store::materialization::contracts::RepresentationWorkItem{
                  .kind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kConstFill,
                  .dst_name = "manifest",
                  .dst_spec =
                      tensorcast::store::materialization::contracts::RepresentationTensorSpec{
                          .name = "manifest",
                          .shape = {4},
                          .stride = {1},
                          .dtype = "torch.uint8",
                          .logical_offset = 8,
                          .logical_length = 4,
                          .storage_offset = 0,
                          .element_size = 1,
                      },
                  .fill_rule =
                      tensorcast::store::materialization::contracts::FillRule{
                          .constant_value = {0x01},
                      },
                  .committed_bytes = 4,
              },
          },
      .residual_fallback_map = mapping,
  };
  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan, mapping);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kGenericOnly,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .generic_backend_map = mapping,
          .local_typed_bytes = 4,
          .local_fill_bytes = 4,
      },
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "generic_only",
          .planned_local_typed_bytes = 4,
          .planned_generic_residual_bytes = kSourceSize,
      });

  loading::MaterializeHints hints;
  hints.artifact_id = artifact_id;
  hints.allow_disk = false;
  hints.allow_p2p = false;

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or =
      harness.facade->materialize_mapped_into_target(target_device, prepared_execution, hints, std::nullopt);
  REQUIRE(result_or.ok());
  CHECK(result_or->source == loading::MaterializationSource::kLocalReplica);

  std::array<uint8_t, kTargetSize> host_out{};
  auto copy_status = tensorcast::cuda::memcpy(host_out.data(), gpu_buffer, kTargetSize, cudaMemcpyDeviceToHost);
  REQUIRE(copy_status.ok());
  for (uint8_t i = 0; i < kSourceSize; ++i) {
    CHECK(host_out[i] == static_cast<uint8_t>(0x10 + i));
  }
  for (size_t index = kSourceSize; index < host_out.size(); ++index) {
    CHECK(host_out[index] == 0x01);
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade preserves source-ordered fast path for explicit generic source-bound plans",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_generic_source_ordered");
  std::vector<unsigned char> payload(8);
  for (size_t index = 0; index < payload.size(); ++index) {
    payload[index] = static_cast<unsigned char>(0x40 + index);
  }
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}",
      payload);

  FacadeHarness harness(MakeOptions(artifact_root));
  harness.initialize();

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, 4).ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };

  std::array<uint8_t, 4> initial{};
  initial.fill(0xCC);
  REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{gpu_buffer},
          .length = 4,
      });
  target_layout.total_size = 4;

  tensorcast::store::loader::ByteRangeMap full_map;
  full_map.total_bytes = 4;
  full_map.num_sources = 1;
  full_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 4,
          .src_offset = 0,
          .source_index = 0,
      });

  ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:artifact_generic_source_ordered";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"tensor":[0,4,[4],[1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan = tensorcast::store::materialization::contracts::RepresentationWorkPlan{
      .items =
          {
              tensorcast::store::materialization::contracts::RepresentationWorkItem{
                  .kind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kResidualByteRange,
                  .byte_range_map = full_map,
                  .committed_bytes = 4,
              },
          },
      .residual_fallback_map = full_map,
  };
  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan, full_map);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kGenericOnly,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .generic_backend_map = full_map,
      },
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "generic_only",
          .planned_generic_residual_bytes = 4,
      });

  loading::MaterializeHints hints;
  hints.artifact_id = resolved_plan.artifact_id;
  hints.allow_p2p = false;
  hints.allow_disk = true;
  loading::DiskMetadata disk_metadata;
  disk_metadata.source_index_json = R"({"tensor":[4,4,[4],[1],"torch.uint8",0]})";
  hints.disk_metadata = disk_metadata;

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or = harness.facade->materialize_mapped_into_target(
      target_device,
      prepared_execution,
      hints,
      loading::DiskSource{.path = artifact_root, .expected_size = std::nullopt});
  REQUIRE(result_or.ok());
  CHECK(result_or->source == loading::MaterializationSource::kDisk);
  CHECK(result_or->source_ordered);
  CHECK(result_or->dominant_executor == "SourceOrderedMappedTargetExecutor");
  CHECK(result_or->selection_reason == "generic_only");

  std::array<uint8_t, 4> host_out{};
  REQUIRE(tensorcast::cuda::memcpy(host_out.data(), gpu_buffer, host_out.size(), cudaMemcpyDeviceToHost).ok());
  for (size_t index = 0; index < host_out.size(); ++index) {
    CHECK(host_out[index] == payload[index + 4]);
  }

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade strict collective mapped request fails before generic fallback",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_collective_strict");
  std::vector<unsigned char> payload(8);
  for (size_t index = 0; index < payload.size(); ++index) {
    payload[index] = static_cast<unsigned char>(0x10 + index);
  }
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}",
      payload);

  FacadeHarness harness(MakeOptions(artifact_root));
  harness.initialize();
  harness.hooks->collective_mapped_target_load_override =
      [](const tensorcast::store::replica::CollectiveMappedTargetLoadRequest&,
         const std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>&,
         std::chrono::milliseconds,
         const tensorcast::store::replica::CollectiveMappedTargetLoadOptions&) {
        return tensorcast::store::replica::CollectiveMappedTargetLoadResult{
            .handled = false,
            .status = absl::OkStatus(),
        };
      };

  void* gpu_buffer = nullptr;
  auto alloc_status = tensorcast::cuda::malloc(&gpu_buffer, 8);
  REQUIRE(alloc_status.ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };
  std::array<uint8_t, 8> initial{};
  initial.fill(0xAA);
  REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{gpu_buffer},
          .length = 8,
      });
  target_layout.total_size = 8;

  tensorcast::store::loader::ByteRangeMap full_map;
  full_map.total_bytes = 8;
  full_map.num_sources = 1;
  full_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 8,
          .src_offset = 0,
          .source_index = 0,
      });

  ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:artifact_collective_strict";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"tensor":[0,8,[8],[1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan = tensorcast::store::materialization::contracts::RepresentationWorkPlan{};
  const auto source_bound_plan_summary =
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .planned_collective_candidate_bytes = 8,
          .planned_collective_admitted_bytes = 8,
          .collective_lane_eligible = true,
          .strict_pure_collective_eligible = true,
      };
  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan, full_map, full_map);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kPureCollective,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .collective_lane_map = full_map,
          .generic_backend_map = full_map,
          .require_collective_success = true,
      },
      source_bound_plan_summary,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundPolicy::kRequirePureCollective);

  loading::MaterializeHints hints;
  hints.artifact_id = resolved_plan.artifact_id;
  hints.collective_load_group = loading::CollectiveLoadGroupHint{
      .group_id = "strict-mapped",
      .world_size = 2,
      .rank = 0,
  };
  hints.require_collective_execution = true;

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or = harness.facade->materialize_mapped_into_target(
      target_device,
      prepared_execution,
      hints,
      loading::DiskSource{.path = artifact_root, .expected_size = std::nullopt});
  REQUIRE_FALSE(result_or.ok());
  REQUIRE(absl::IsFailedPrecondition(result_or.status()));
  CHECK(result_or.status().message().find("explicit source-bound strategy plan") != std::string::npos);
  CHECK(result_or.status().message().find("collective_executor_unhandled") != std::string::npos);

  std::array<uint8_t, 8> host_out{};
  REQUIRE(tensorcast::cuda::memcpy(host_out.data(), gpu_buffer, host_out.size(), cudaMemcpyDeviceToHost).ok());
  CHECK(host_out == initial);

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade surfaces collective execution metrics on handled mapped collective execution",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_collective_metrics");
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}",
      std::vector<unsigned char>(8, 0x5A));

  FacadeHarness harness(MakeOptions(artifact_root));
  harness.initialize();
  harness.hooks->collective_mapped_target_load_override =
      [](const tensorcast::store::replica::CollectiveMappedTargetLoadRequest&,
         const std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>&,
         std::chrono::milliseconds,
         const tensorcast::store::replica::CollectiveMappedTargetLoadOptions&) {
        return tensorcast::store::replica::CollectiveMappedTargetLoadResult{
            .handled = true,
            .status = absl::OkStatus(),
            .metrics =
                tensorcast::store::runtime::ingestion::strategy::CollectiveExecutionMetrics{
                    .unique_source_bytes = 64,
                    .peer_transfer_bytes = 32,
                    .peak_temporary_bytes = 128,
                    .batch_count = 2,
                    .dedup_saving_bytes = 16,
                },
        };
      };

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, 8).ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{gpu_buffer},
          .length = 8,
      });
  target_layout.total_size = 8;

  tensorcast::store::loader::ByteRangeMap full_map;
  full_map.total_bytes = 8;
  full_map.num_sources = 1;
  full_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 8,
          .src_offset = 0,
          .source_index = 0,
      });

  ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:artifact_collective_metrics";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"tensor":[0,8,[8],[1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan = tensorcast::store::materialization::contracts::RepresentationWorkPlan{};
  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan, full_map, full_map);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kPureCollective,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .collective_lane_map = full_map,
          .generic_backend_map = full_map,
          .require_collective_success = true,
          .selection_reason = "pure_collective",
      },
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "pure_collective",
          .planned_collective_candidate_bytes = 8,
          .planned_collective_admitted_bytes = 8,
          .collective_lane_eligible = true,
          .strict_pure_collective_eligible = true,
      });

  loading::MaterializeHints hints;
  hints.artifact_id = resolved_plan.artifact_id;
  hints.collective_load_group = loading::CollectiveLoadGroupHint{
      .group_id = "collective-metrics",
      .world_size = 2,
      .rank = 0,
  };
  hints.require_collective_execution = true;
  loading::DiskMetadata disk_metadata;
  disk_metadata.source_index_json = R"({"tensor":[0,8,[8],[1],"torch.uint8",0]})";
  hints.disk_metadata = disk_metadata;

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or = harness.facade->materialize_mapped_into_target(
      target_device,
      prepared_execution,
      hints,
      loading::DiskSource{.path = artifact_root, .expected_size = std::nullopt});
  REQUIRE(result_or.ok());
  CHECK(result_or->collective_handled);
  CHECK(result_or->actual_collective_committed_bytes == 8);
  CHECK(result_or->collective_unique_source_bytes == 64);
  CHECK(result_or->collective_peer_transfer_bytes == 32);
  CHECK(result_or->collective_peak_temporary_bytes == 128);
  CHECK(result_or->collective_batch_count == 2);
  CHECK(result_or->collective_dedup_saving_bytes == 16);

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade finalizes mapped collective lane with source layout remap",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_collective_source_layout");
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}",
      std::vector<unsigned char>(8, 3));

  FacadeHarness harness(MakeOptions(artifact_root));
  harness.initialize();

  std::optional<tensorcast::store::replica::CollectiveMappedTargetLoadRequest> captured_request;
  harness.hooks->collective_mapped_target_load_override =
      [&](const tensorcast::store::replica::CollectiveMappedTargetLoadRequest& request,
          const std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>&,
          std::chrono::milliseconds,
          const tensorcast::store::replica::CollectiveMappedTargetLoadOptions&) {
        captured_request = request;
        return tensorcast::store::replica::CollectiveMappedTargetLoadResult{
            .handled = false,
            .status = absl::OkStatus(),
        };
      };

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, 4).ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{gpu_buffer},
          .length = 4,
      });
  target_layout.total_size = 4;

  tensorcast::store::loader::ByteRangeMap full_map;
  full_map.total_bytes = 4;
  full_map.num_sources = 1;
  full_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 4,
          .src_offset = 0,
          .source_index = 0,
      });

  tensorcast::store::runtime::ingestion::strategy::ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:artifact_collective_source_layout";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"tensor":[0,4,[4],[1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan = tensorcast::store::materialization::contracts::RepresentationWorkPlan{};
  const auto source_bound_plan_summary =
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .planned_collective_candidate_bytes = 4,
          .planned_collective_admitted_bytes = 4,
          .collective_lane_eligible = true,
          .strict_pure_collective_eligible = true,
      };
  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan, full_map, full_map);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kPureCollective,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .collective_lane_map = full_map,
          .generic_backend_map = full_map,
          .require_collective_success = true,
      },
      source_bound_plan_summary,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundPolicy::kRequirePureCollective);

  loading::MaterializeHints hints;
  hints.artifact_id = resolved_plan.artifact_id;
  hints.collective_load_group = loading::CollectiveLoadGroupHint{
      .group_id = "source-layout-mapped",
      .world_size = 2,
      .rank = 0,
  };
  hints.require_collective_execution = true;
  loading::DiskMetadata disk_metadata;
  disk_metadata.source_index_json = R"({"tensor":[4,4,[4],[1],"torch.uint8",0]})";
  hints.disk_metadata = disk_metadata;

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or = harness.facade->materialize_mapped_into_target(
      target_device,
      prepared_execution,
      hints,
      loading::DiskSource{.path = artifact_root, .expected_size = std::nullopt});
  REQUIRE_FALSE(result_or.ok());
  REQUIRE(absl::IsFailedPrecondition(result_or.status()));
  CHECK(result_or.status().message().find("explicit source-bound strategy plan") != std::string::npos);
  REQUIRE(captured_request.has_value());
  REQUIRE(captured_request->collective_lane_map.segments.size() == 1);
  CHECK(captured_request->collective_lane_map.segments.front().src_offset == 4);
  CHECK(captured_request->collective_lane_map.segments.front().dst_offset == 0);
  CHECK(captured_request->collective_lane_map.segments.front().length == 4);

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade continues generic residual execution after mapped collective success",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_collective_residual");
  std::vector<unsigned char> payload(12);
  for (size_t index = 0; index < payload.size(); ++index) {
    payload[index] = static_cast<unsigned char>(0x10 + index);
  }
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[12],\"data_offsets\":[0,12]}}",
      payload);

  FacadeHarness harness(MakeOptions(artifact_root));
  harness.initialize();
  harness.hooks->collective_mapped_target_load_override =
      [](const tensorcast::store::replica::CollectiveMappedTargetLoadRequest& request,
         const std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>&,
         std::chrono::milliseconds,
         const tensorcast::store::replica::CollectiveMappedTargetLoadOptions&) {
        const std::array<uint8_t, 8> collective_bytes = {0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7};
        auto status = tensorcast::cuda::memcpy(
            request.target_layout.storages.front().base_ptr.get(),
            collective_bytes.data(),
            collective_bytes.size(),
            cudaMemcpyHostToDevice);
        if (!status.ok()) {
          return tensorcast::store::replica::CollectiveMappedTargetLoadResult{
              .handled = true,
              .status = status,
          };
        }
        return tensorcast::store::replica::CollectiveMappedTargetLoadResult{
            .handled = true,
            .status = absl::OkStatus(),
        };
      };

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, 16).ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };
  std::array<uint8_t, 16> initial{};
  initial.fill(0xCC);
  REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{gpu_buffer},
          .length = 16,
      });
  target_layout.total_size = 16;

  tensorcast::store::loader::ByteRangeMap executor_map;
  executor_map.total_bytes = 16;
  executor_map.num_sources = 1;
  executor_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 12,
          .src_offset = 0,
          .source_index = 0,
      });
  tensorcast::store::loader::ByteRangeMap collective_map;
  collective_map.total_bytes = 16;
  collective_map.num_sources = 1;
  collective_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 8,
          .src_offset = 0,
          .source_index = 0,
      });
  tensorcast::store::loader::ByteRangeMap residual_map;
  residual_map.total_bytes = 16;
  residual_map.num_sources = 1;
  residual_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 8,
          .length = 4,
          .src_offset = 8,
          .source_index = 0,
      });

  tensorcast::store::runtime::ingestion::strategy::ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:artifact_collective_residual";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"tensor":[0,12,[12],[1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan = tensorcast::store::materialization::contracts::RepresentationWorkPlan{
      .items =
          {
              tensorcast::store::materialization::contracts::RepresentationWorkItem{
                  .kind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kConstFill,
                  .dst_name = "tail",
                  .dst_spec =
                      tensorcast::store::materialization::contracts::RepresentationTensorSpec{
                          .name = "tail",
                          .shape = {4},
                          .stride = {1},
                          .dtype = "torch.uint8",
                          .logical_offset = 12,
                          .logical_length = 4,
                          .storage_offset = 0,
                          .element_size = 1,
                      },
                  .fill_rule =
                      tensorcast::store::materialization::contracts::FillRule{
                          .constant_value = {0xEE},
                      },
                  .committed_bytes = 4,
              },
          },
      .residual_fallback_map = residual_map,
  };
  const auto source_bound_plan_summary =
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "collective_first_mixed",
          .planned_collective_candidate_bytes = 8,
          .planned_collective_admitted_bytes = 8,
          .planned_local_typed_bytes = 4,
          .planned_generic_residual_bytes = 4,
          .collective_lane_eligible = true,
      };
  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan, executor_map, collective_map);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kCollectiveFirstMixed,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .collective_lane_map = collective_map,
          .generic_backend_map = executor_map,
          .true_residual_map = residual_map,
          .local_typed_bytes = 4,
          .local_fill_bytes = 4,
          .selection_reason = "collective_first_mixed",
      },
      source_bound_plan_summary);

  loading::MaterializeHints hints;
  hints.artifact_id = resolved_plan.artifact_id;
  hints.collective_load_group = loading::CollectiveLoadGroupHint{
      .group_id = "collective-residual",
      .world_size = 2,
      .rank = 0,
  };

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or = harness.facade->materialize_mapped_into_target(
      target_device,
      prepared_execution,
      hints,
      loading::DiskSource{.path = artifact_root, .expected_size = std::nullopt});
  REQUIRE(result_or.ok());
  CHECK(result_or->collective_handled);
  CHECK(result_or->actual_collective_committed_bytes == 8);
  CHECK(result_or->actual_generic_backend_bytes == 4);
  CHECK(result_or->actual_local_typed_bytes == 4);
  CHECK(result_or->fallback_bytes == 4);
  CHECK(result_or->committed_bytes == 16);
  CHECK(result_or->selection_reason == "collective_first_mixed");

  std::array<uint8_t, 16> host_out{};
  REQUIRE(tensorcast::cuda::memcpy(host_out.data(), gpu_buffer, host_out.size(), cudaMemcpyDeviceToHost).ok());
  for (size_t index = 0; index < 8; ++index) {
    CHECK(host_out[index] == static_cast<uint8_t>(0xA0 + index));
  }
  for (size_t index = 8; index < 12; ++index) {
    CHECK(host_out[index] == static_cast<uint8_t>(0x10 + index));
  }
  for (size_t index = 12; index < 16; ++index) {
    CHECK(host_out[index] == 0xEE);
  }

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade rejects collective-unhandled explicit mixed source-bound plan",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_collective_unhandled_mixed");
  std::vector<unsigned char> payload(12);
  for (size_t index = 0; index < payload.size(); ++index) {
    payload[index] = static_cast<unsigned char>(0x30 + index);
  }
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[12],\"data_offsets\":[0,12]}}",
      payload);

  FacadeHarness harness(MakeOptions(artifact_root));
  harness.initialize();
  harness.hooks->collective_mapped_target_load_override =
      [](const tensorcast::store::replica::CollectiveMappedTargetLoadRequest&,
         const std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>&,
         std::chrono::milliseconds,
         const tensorcast::store::replica::CollectiveMappedTargetLoadOptions&) {
        return tensorcast::store::replica::CollectiveMappedTargetLoadResult{
            .handled = false,
            .status = absl::OkStatus(),
        };
      };

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, 16).ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };
  std::array<uint8_t, 16> initial{};
  initial.fill(0xAB);
  REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{gpu_buffer},
          .length = 16,
      });
  target_layout.total_size = 16;

  tensorcast::store::loader::ByteRangeMap executor_map;
  executor_map.total_bytes = 16;
  executor_map.num_sources = 1;
  executor_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 8,
          .src_offset = 0,
          .source_index = 0,
      });
  executor_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 8,
          .length = 4,
          .src_offset = 8,
          .source_index = 0,
      });
  tensorcast::store::loader::ByteRangeMap collective_map;
  collective_map.total_bytes = 16;
  collective_map.num_sources = 1;
  collective_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 8,
          .src_offset = 0,
          .source_index = 0,
      });
  tensorcast::store::loader::ByteRangeMap residual_map;
  residual_map.total_bytes = 16;
  residual_map.num_sources = 1;
  residual_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 8,
          .length = 4,
          .src_offset = 8,
          .source_index = 0,
      });

  ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:artifact_collective_unhandled_mixed";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"tensor":[0,12,[12],[1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan = tensorcast::store::materialization::contracts::RepresentationWorkPlan{
      .items =
          {
              tensorcast::store::materialization::contracts::RepresentationWorkItem{
                  .kind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kConstFill,
                  .dst_name = "tail",
                  .dst_spec =
                      tensorcast::store::materialization::contracts::RepresentationTensorSpec{
                          .name = "tail",
                          .shape = {4},
                          .stride = {1},
                          .dtype = "torch.uint8",
                          .logical_offset = 12,
                          .logical_length = 4,
                          .storage_offset = 0,
                          .element_size = 1,
                      },
                  .fill_rule =
                      tensorcast::store::materialization::contracts::FillRule{
                          .constant_value = {0xEF},
                      },
                  .committed_bytes = 4,
              },
          },
      .residual_fallback_map = residual_map,
  };
  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan, executor_map, collective_map);
  prepared_execution.strategy_plan = tensorcast::store::runtime::ingestion::strategy::SourceBoundStrategyPlan{
      .policy = tensorcast::store::runtime::ingestion::strategy::SourceBoundPolicy::kCollectiveFirst,
      .lane_plan =
          tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
              .mode = tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kCollectiveFirstMixed,
              .collective_lane_map = collective_map,
              .generic_backend_map = executor_map,
              .true_residual_map = residual_map,
              .local_typed_bytes = 4,
              .local_fill_bytes = 4,
              .selection_reason = "collective_first_mixed",
          },
      .summary =
          tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
              .execution_plan_kind = "collective_first_mixed",
              .planned_collective_candidate_bytes = 8,
              .planned_collective_admitted_bytes = 8,
              .planned_local_typed_bytes = 4,
              .planned_generic_residual_bytes = 4,
              .collective_lane_eligible = true,
          },
  };

  loading::MaterializeHints hints;
  hints.artifact_id = resolved_plan.artifact_id;
  hints.collective_load_group = loading::CollectiveLoadGroupHint{
      .group_id = "collective-unhandled-mixed",
      .world_size = 2,
      .rank = 0,
  };

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or = harness.facade->materialize_mapped_into_target(
      target_device,
      prepared_execution,
      hints,
      loading::DiskSource{.path = artifact_root, .expected_size = std::nullopt});
  REQUIRE_FALSE(result_or.ok());
  REQUIRE(absl::IsFailedPrecondition(result_or.status()));
  CHECK(result_or.status().message().find("explicit source-bound strategy plan") != std::string::npos);
  CHECK(result_or.status().message().find("collective_executor_unhandled") != std::string::npos);

  std::array<uint8_t, 16> host_out{};
  REQUIRE(tensorcast::cuda::memcpy(host_out.data(), gpu_buffer, host_out.size(), cudaMemcpyDeviceToHost).ok());
  CHECK(host_out == initial);

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade executes planned local mapped rect2d work without generic backend bytes",
    "[materialization_facade][local_mapped]") {
  SKIP_IF_NO_CUDA();
  const char* cuda_backend = std::getenv("TENSORCAST_CUDA_BACKEND");
  if (cuda_backend != nullptr && std::string_view(cuda_backend) == "fake") {
    SKIP("rect2d local mapped commit report requires real CUDA because the executor uses cudaMemcpy2DAsync");
  }

  auto artifact_root = make_temp_dir("materialization_facade_local_mapped_rect2d");
  std::vector<unsigned char> payload(24);
  for (size_t index = 0; index < payload.size(); ++index) {
    payload[index] = static_cast<unsigned char>(0x50 + index);
  }
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"src\":{\"dtype\":\"U8\",\"shape\":[4,6],\"data_offsets\":[0,24]}}",
      payload);

  auto opts = MakeOptions(artifact_root);
  opts.materialization_strategy.allow_mixed_execution = true;
  FacadeHarness harness(opts);
  harness.initialize();

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, 24).ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };
  std::array<uint8_t, 24> initial{};
  initial.fill(0xEE);
  REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  loading::IntoTargetLayout target_layout{
      .storages =
          {
              loading::IntoTargetStorage{
                  .base_ptr = gsl::not_null<void*>{gpu_buffer},
                  .length = 24,
              },
          },
      .total_size = 24,
  };

  tensorcast::store::loader::ByteRangeMap executor_map;
  executor_map.total_bytes = 24;
  executor_map.num_sources = 1;
  executor_map.segments = {
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 1,
          .length = 3,
          .src_offset = 0,
          .source_index = 0,
      },
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 7,
          .length = 3,
          .src_offset = 0,
          .source_index = 0,
      },
  };

  tensorcast::store::runtime::ingestion::strategy::ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:local_mapped_rect2d";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"src":[0,24,[4,6],[6,1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  auto rect = [](int64_t row_begin, int64_t row_end, int64_t col_begin, int64_t col_end) {
    return tensorcast::store::materialization::contracts::TensorCoordinateSpec{
        .axes =
            {
                tensorcast::store::materialization::contracts::TensorAxisRange{
                    .dim = 0, .start = row_begin, .end = row_end},
                tensorcast::store::materialization::contracts::TensorAxisRange{
                    .dim = 1, .start = col_begin, .end = col_end},
            },
    };
  };
  const auto src_spec = tensorcast::store::materialization::contracts::RepresentationTensorSpec{
      .name = "src",
      .shape = {4, 6},
      .stride = {6, 1},
      .dtype = "torch.uint8",
      .logical_offset = 0,
      .logical_length = 24,
      .storage_offset = 0,
      .element_size = 1,
  };
  const auto dst_spec = tensorcast::store::materialization::contracts::RepresentationTensorSpec{
      .name = "dst",
      .shape = {4, 6},
      .stride = {6, 1},
      .dtype = "torch.uint8",
      .logical_offset = 0,
      .logical_length = 24,
      .storage_offset = 0,
      .element_size = 1,
  };
  tensorcast::store::materialization::contracts::RepresentationWorkItem item;
  item.kind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
  item.partition_kind = tensorcast::store::materialization::contracts::WorkPartitionKind::kUnknown;
  item.dst_name = "dst";
  item.dst_spec = dst_spec;
  item.committed_bytes = 6;
  item.sources.push_back(
      tensorcast::store::materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              tensorcast::store::materialization::contracts::SourceFragment{
                  .source_spec = src_spec,
                  .source_range = rect(1, 3, 2, 5),
                  .destination_range = rect(0, 2, 1, 4),
              },
      });
  resolved_plan.representation_work_plan =
      tensorcast::store::materialization::contracts::RepresentationWorkPlan{.items = {item}};

  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan, executor_map);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kLocalMappedTyped,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .generic_backend_map = executor_map,
          .deferred_typed_bytes = 6,
          .selection_reason = "local_mapped_typed",
      },
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "local_mapped_typed",
          .planned_non_admitted_typed_bytes = 6,
      });

  loading::MaterializeHints hints;
  hints.artifact_id = resolved_plan.artifact_id;
  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or = harness.facade->materialize_mapped_into_target(
      target_device,
      prepared_execution,
      hints,
      loading::DiskSource{.path = artifact_root, .expected_size = std::nullopt});
  REQUIRE(result_or.ok());
  CHECK(result_or->residual_bytes == 0);
  CHECK(result_or->fallback_bytes == 0);
  CHECK(result_or->actual_generic_backend_bytes == 0);
  CHECK(result_or->actual_local_typed_bytes == 6);
  CHECK(result_or->dominant_executor == "TensorMappedLocalExecutor");

  std::array<uint8_t, 24> actual{};
  REQUIRE(tensorcast::cuda::memcpy(actual.data(), gpu_buffer, actual.size(), cudaMemcpyDeviceToHost).ok());
  std::array<uint8_t, 24> expected = initial;
  expected[1] = payload[8];
  expected[2] = payload[9];
  expected[3] = payload[10];
  expected[7] = payload[14];
  expected[8] = payload[15];
  expected[9] = payload[16];
  CHECK(actual == expected);

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade continues generic residual after partial local mapped typed execution",
    "[materialization_facade][local_mapped]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_local_mapped_partial_residual");
  std::vector<unsigned char> payload(8);
  for (size_t index = 0; index < payload.size(); ++index) {
    payload[index] = static_cast<unsigned char>(0x60 + index);
  }
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"rep\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},"
      "\"src3d\":{\"dtype\":\"U8\",\"shape\":[2,2,1],\"data_offsets\":[4,8]}}",
      payload);

  auto opts = MakeOptions(artifact_root);
  opts.materialization_strategy.allow_mixed_execution = true;
  opts.materialization_strategy.enable_tensor_aware_mapped_executor = true;
  opts.materialization_strategy.enable_mapped_concat_jobs = true;
  opts.materialization_strategy.enable_mapped_concat_execution = true;
  FacadeHarness harness(opts);
  harness.initialize();

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, 8).ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };
  std::array<uint8_t, 8> initial{};
  initial.fill(0xEE);
  REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  loading::IntoTargetLayout target_layout{
      .storages =
          {
              loading::IntoTargetStorage{
                  .base_ptr = gsl::not_null<void*>{gpu_buffer},
                  .length = 8,
              },
          },
      .total_size = 8,
  };

  tensorcast::store::loader::ByteRangeMap executor_map;
  executor_map.total_bytes = 8;
  executor_map.num_sources = 1;
  executor_map.segments = {
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 4,
          .src_offset = 0,
          .source_index = 0,
      },
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 4,
          .length = 4,
          .src_offset = 4,
          .source_index = 0,
      },
  };

  tensorcast::store::runtime::ingestion::strategy::ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:local_mapped_partial_residual";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json =
      R"({"rep_dst":[0,4,[4],[1],"torch.uint8",0],"dst":[4,4,[2,2],[2,1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };

  const auto rep_src_spec = tensorcast::store::materialization::contracts::RepresentationTensorSpec{
      .name = "rep",
      .shape = {4},
      .stride = {1},
      .dtype = "torch.uint8",
      .logical_offset = 0,
      .logical_length = 4,
      .storage_offset = 0,
      .element_size = 1,
  };
  const auto rep_dst_spec = tensorcast::store::materialization::contracts::RepresentationTensorSpec{
      .name = "rep_dst",
      .shape = {4},
      .stride = {1},
      .dtype = "torch.uint8",
      .logical_offset = 0,
      .logical_length = 4,
      .storage_offset = 0,
      .element_size = 1,
  };
  tensorcast::store::materialization::contracts::RepresentationWorkItem replicated;
  replicated.kind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
  replicated.partition_kind = tensorcast::store::materialization::contracts::WorkPartitionKind::kReplicated;
  replicated.dst_name = "rep_dst";
  replicated.dst_spec = rep_dst_spec;
  replicated.committed_bytes = 4;
  replicated.sources.push_back(
      tensorcast::store::materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              tensorcast::store::materialization::contracts::SourceFragment{
                  .source_spec = rep_src_spec,
                  .source_range = tensorcast::store::materialization::contracts::TensorCoordinateSpec{},
                  .destination_range = tensorcast::store::materialization::contracts::TensorCoordinateSpec{},
              },
      });

  const auto src3d_spec = tensorcast::store::materialization::contracts::RepresentationTensorSpec{
      .name = "src3d",
      .shape = {2, 2, 1},
      .stride = {2, 1, 1},
      .dtype = "torch.uint8",
      .logical_offset = 4,
      .logical_length = 4,
      .storage_offset = 0,
      .element_size = 1,
  };
  const auto dst_spec = tensorcast::store::materialization::contracts::RepresentationTensorSpec{
      .name = "dst",
      .shape = {2, 2},
      .stride = {2, 1},
      .dtype = "torch.uint8",
      .logical_offset = 4,
      .logical_length = 4,
      .storage_offset = 0,
      .element_size = 1,
  };
  tensorcast::store::materialization::contracts::RepresentationWorkItem unsupported_typed;
  unsupported_typed.kind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kTensorCopy;
  unsupported_typed.partition_kind = tensorcast::store::materialization::contracts::WorkPartitionKind::kUnknown;
  unsupported_typed.dst_name = "dst";
  unsupported_typed.dst_spec = dst_spec;
  unsupported_typed.committed_bytes = 4;
  unsupported_typed.sources.push_back(
      tensorcast::store::materialization::contracts::RepresentationWorkSourceFragment{
          .fragment =
              tensorcast::store::materialization::contracts::SourceFragment{
                  .source_spec = src3d_spec,
                  .source_range =
                      tensorcast::store::materialization::contracts::TensorCoordinateSpec{
                          .axes =
                              {
                                  tensorcast::store::materialization::contracts::TensorAxisRange{
                                      .dim = 0, .start = 0, .end = 2},
                                  tensorcast::store::materialization::contracts::TensorAxisRange{
                                      .dim = 1, .start = 0, .end = 2},
                                  tensorcast::store::materialization::contracts::TensorAxisRange{
                                      .dim = 2, .start = 0, .end = 1},
                              },
                      },
                  .destination_range =
                      tensorcast::store::materialization::contracts::TensorCoordinateSpec{
                          .axes =
                              {
                                  tensorcast::store::materialization::contracts::TensorAxisRange{
                                      .dim = 0, .start = 0, .end = 2},
                                  tensorcast::store::materialization::contracts::TensorAxisRange{
                                      .dim = 1, .start = 0, .end = 2},
                              },
                      },
              },
      });
  resolved_plan.representation_work_plan = tensorcast::store::materialization::contracts::RepresentationWorkPlan{
      .items = {replicated, unsupported_typed},
  };

  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan, executor_map);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kLocalMappedTyped,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .generic_backend_map = executor_map,
          .deferred_typed_bytes = 4,
          .selection_reason = "local_mapped_typed",
      },
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "local_mapped_typed",
          .planned_non_admitted_typed_bytes = 4,
      });

  loading::MaterializeHints hints;
  hints.artifact_id = resolved_plan.artifact_id;
  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or = harness.facade->materialize_mapped_into_target(
      target_device,
      prepared_execution,
      hints,
      loading::DiskSource{.path = artifact_root, .expected_size = std::nullopt});
  REQUIRE(result_or.ok());
  CHECK(result_or->fallback_bytes == 4);
  CHECK(result_or->actual_local_typed_bytes == 4);
  CHECK(result_or->actual_generic_backend_bytes == 4);
  CHECK(result_or->committed_bytes == 8);
  CHECK(result_or->dominant_executor == "TensorMappedLocalExecutor+GenericByteRangeExecutor");

  std::array<uint8_t, 8> actual{};
  REQUIRE(tensorcast::cuda::memcpy(actual.data(), gpu_buffer, actual.size(), cudaMemcpyDeviceToHost).ok());
  for (size_t index = 0; index < actual.size(); ++index) {
    CHECK(actual[index] == payload[index]);
  }

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade does not let local source selection steal explicit collective lane ownership",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_collective_lane_ownership");
  std::vector<unsigned char> payload(8);
  for (size_t index = 0; index < payload.size(); ++index) {
    payload[index] = static_cast<unsigned char>(0x70 + index);
  }
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}",
      payload);

  FacadeHarness harness(MakeOptions(artifact_root));
  harness.initialize();
  harness.hooks->collective_mapped_target_load_override =
      [](const tensorcast::store::replica::CollectiveMappedTargetLoadRequest&,
         const std::shared_ptr<tensorcast::common::memory::PinnedBufferPool>&,
         std::chrono::milliseconds,
         const tensorcast::store::replica::CollectiveMappedTargetLoadOptions&) {
        return tensorcast::store::replica::CollectiveMappedTargetLoadResult{
            .handled = false,
            .status = absl::OkStatus(),
        };
      };

  loading::InlineBufferSource source{.data = nullptr, .size_bytes = 8};
  tensorcast::store::replica::ReplicaConfig cfg{
      .source = source,
      .artifact_identifier = "cgid:artifact_collective_lane_ownership",
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool = harness.runtime_context().pinned_buffer_pool(),
      .async_runtime =
          gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{harness.runtime_context().async_runtime()},
      .artifact_chunk_bytes = harness.options().artifact_chunk_bytes,
      .expected_artifact_size = 8,
      .materialization_strategy = harness.options().materialization_strategy,
  };
  auto local_or = tensorcast::store::replica::Replica::create(cfg);
  REQUIRE(local_or.ok());
  auto local_replica = std::shared_ptr<tensorcast::store::replica::Replica>(std::move(local_or.value()));
  CHECK_OK(local_replica->get_memory_manager().allocate_memory(MemoryLocation::CPU));
  auto cpu_ptrs = local_replica->get_data_pointer(MemoryLocation::CPU);
  REQUIRE(cpu_ptrs.size() == 1);
  auto* cpu_ptr = static_cast<uint8_t*>(cpu_ptrs.front());
  REQUIRE(cpu_ptr != nullptr);
  for (size_t index = 0; index < payload.size(); ++index) {
    cpu_ptr[index] = static_cast<uint8_t>(0x10 + index);
  }
  CHECK_OK(local_replica->mark_loaded(MemoryLocation::CPU));
  local_replica->set_ready_signal(MemoryLocation::CPU, absl::OkStatus());
  loading::ReplicaKey local_key{
      .artifact_id = "cgid:artifact_collective_lane_ownership",
      .view_id = std::nullopt,
      .device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      .replica = 0,
  };
  CHECK_OK(harness.replica_runtime().registry().emplace(local_key, gsl::not_null{local_replica}));

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, 8).ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };
  std::array<uint8_t, 8> initial{};
  initial.fill(0xBD);
  REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{gpu_buffer},
          .length = 8,
      });
  target_layout.total_size = 8;

  tensorcast::store::loader::ByteRangeMap full_map;
  full_map.total_bytes = 8;
  full_map.num_sources = 1;
  full_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 8,
          .src_offset = 0,
          .source_index = 0,
      });

  ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:artifact_collective_lane_ownership";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"tensor":[0,8,[8],[1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan = tensorcast::store::materialization::contracts::RepresentationWorkPlan{};
  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan, full_map, full_map);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kCollectiveFirstMixed,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .collective_lane_map = full_map,
          .generic_backend_map = full_map,
          .selection_reason = "collective_first_mixed",
      },
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "collective_first_mixed",
          .planned_collective_candidate_bytes = 8,
          .planned_collective_admitted_bytes = 8,
          .collective_lane_eligible = true,
      });

  loading::MaterializeHints hints;
  hints.artifact_id = resolved_plan.artifact_id;
  hints.collective_load_group = loading::CollectiveLoadGroupHint{
      .group_id = "collective-lane-ownership",
      .world_size = 2,
      .rank = 0,
  };

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or = harness.facade->materialize_mapped_into_target(
      target_device,
      prepared_execution,
      hints,
      loading::DiskSource{.path = artifact_root, .expected_size = std::nullopt});
  REQUIRE_FALSE(result_or.ok());
  REQUIRE(absl::IsFailedPrecondition(result_or.status()));
  CHECK(result_or.status().message().find("collective_executor_unhandled") != std::string::npos);

  std::array<uint8_t, 8> host_out{};
  REQUIRE(tensorcast::cuda::memcpy(host_out.data(), gpu_buffer, host_out.size(), cudaMemcpyDeviceToHost).ok());
  CHECK(host_out == initial);

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade rejects source-bound mapped execution without explicit executor fallback map",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_missing_executor_map");
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}",
      std::vector<unsigned char>(4, 7));

  FacadeHarness harness(MakeOptions(artifact_root));
  harness.initialize();

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, 4).ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{gpu_buffer},
          .length = 4,
      });
  target_layout.total_size = 4;

  tensorcast::store::loader::ByteRangeMap residual_map;
  residual_map.total_bytes = 4;
  residual_map.num_sources = 1;
  residual_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 4,
          .src_offset = 0,
          .source_index = 0,
      });

  tensorcast::store::runtime::ingestion::strategy::ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:artifact_missing_executor_map";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"tensor":[0,4,[4],[1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan = tensorcast::store::materialization::contracts::RepresentationWorkPlan{
      .items =
          {
              tensorcast::store::materialization::contracts::RepresentationWorkItem{
                  .kind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kResidualByteRange,
                  .byte_range_map = residual_map,
                  .committed_bytes = 4,
              },
          },
      .residual_fallback_map = residual_map,
  };

  loading::MaterializeHints hints;
  hints.artifact_id = resolved_plan.artifact_id;
  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kGenericOnly,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{},
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "generic_only",
          .planned_generic_residual_bytes = 4,
      });

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or = harness.facade->materialize_mapped_into_target(
      target_device,
      prepared_execution,
      hints,
      loading::DiskSource{.path = artifact_root, .expected_size = std::nullopt});
  REQUIRE_FALSE(result_or.ok());
  REQUIRE(absl::IsFailedPrecondition(result_or.status()));
  CHECK(result_or.status().message().find("executor fallback map") != std::string::npos);

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade rejects explicit rejected source-bound strategy plan before source setup",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_rejected_strategy_plan";
  std::filesystem::create_directories(temp_root);
  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, 4).ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };

  std::array<uint8_t, 4> initial{};
  initial.fill(0xD1);
  REQUIRE(tensorcast::cuda::memcpy(gpu_buffer, initial.data(), initial.size(), cudaMemcpyHostToDevice).ok());

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{gpu_buffer},
          .length = 4,
      });
  target_layout.total_size = 4;

  tensorcast::store::runtime::ingestion::strategy::ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:artifact_rejected_strategy_plan";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"tensor":[0,4,[4],[1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan = tensorcast::store::materialization::contracts::RepresentationWorkPlan{
      .items =
          {
              tensorcast::store::materialization::contracts::RepresentationWorkItem{
                  .kind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kResidualByteRange,
                  .committed_bytes = 4,
              },
          },
  };

  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kRejected,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .selection_reason = "reject",
          .reject_reason_buckets = {{"generic_backend_coverage_unproven", 4}},
      },
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "reject",
          .planned_generic_residual_bytes = 4,
          .planner_reject_reason_buckets = {{"generic_backend_coverage_unproven", 4}},
      });

  loading::MaterializeHints hints;
  hints.artifact_id = resolved_plan.artifact_id;

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or =
      harness.facade->materialize_mapped_into_target(target_device, prepared_execution, hints, std::nullopt);
  REQUIRE_FALSE(result_or.ok());
  REQUIRE(absl::IsFailedPrecondition(result_or.status()));
  CHECK(result_or.status().message().find("rejected request before execution setup") != std::string::npos);
  CHECK(result_or.status().message().find("generic_backend_coverage_unproven=4") != std::string::npos);

  std::array<uint8_t, 4> host_out{};
  REQUIRE(tensorcast::cuda::memcpy(host_out.data(), gpu_buffer, host_out.size(), cudaMemcpyDeviceToHost).ok());
  CHECK(host_out == initial);

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade rejects collective-aware mapped execution without source-bound strategy plan or collective lane map",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto artifact_root = make_temp_dir("materialization_facade_missing_collective_map");
  create_safetensors_file(
      artifact_root / "weights.safetensors",
      "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}",
      std::vector<unsigned char>(8, 9));

  FacadeHarness harness(MakeOptions(artifact_root));
  harness.initialize();

  void* gpu_buffer = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&gpu_buffer, 8).ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{gpu_buffer},
          .length = 8,
      });
  target_layout.total_size = 8;

  tensorcast::store::loader::ByteRangeMap executor_map;
  executor_map.total_bytes = 8;
  executor_map.num_sources = 1;
  executor_map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 8,
          .src_offset = 0,
          .source_index = 0,
      });

  tensorcast::store::runtime::ingestion::strategy::ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:artifact_missing_collective_map";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"tensor":[0,8,[8],[1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan = tensorcast::store::materialization::contracts::RepresentationWorkPlan{
      .items =
          {
              tensorcast::store::materialization::contracts::RepresentationWorkItem{
                  .kind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kTensorCopy,
                  .partition_kind = tensorcast::store::materialization::contracts::WorkPartitionKind::kDim0Partitioned,
                  .committed_bytes = 8,
              },
          },
  };
  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan, executor_map);

  loading::MaterializeHints hints;
  hints.artifact_id = resolved_plan.artifact_id;
  hints.collective_load_group = loading::CollectiveLoadGroupHint{
      .group_id = "missing-collective-map",
      .world_size = 2,
      .rank = 0,
  };

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto no_summary_or = harness.facade->materialize_mapped_into_target(
      target_device,
      prepared_execution,
      hints,
      loading::DiskSource{.path = artifact_root, .expected_size = std::nullopt});
  REQUIRE_FALSE(no_summary_or.ok());
  REQUIRE(absl::IsFailedPrecondition(no_summary_or.status()));
  CHECK(no_summary_or.status().message().find("source_bound_strategy_plan") != std::string::npos);

  const auto source_bound_plan_summary =
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "collective_first_mixed",
          .planned_collective_candidate_bytes = 8,
          .planned_collective_admitted_bytes = 8,
          .collective_lane_eligible = true,
      };
  auto missing_map_execution = make_prepared_source_bound_execution_plan(resolved_plan, executor_map);
  missing_map_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kCollectiveFirstMixed,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .generic_backend_map = executor_map,
          .selection_reason = "collective_first_mixed",
      },
      source_bound_plan_summary);
  auto missing_map_or = harness.facade->materialize_mapped_into_target(
      target_device,
      missing_map_execution,
      hints,
      loading::DiskSource{.path = artifact_root, .expected_size = std::nullopt});
  REQUIRE_FALSE(missing_map_or.ok());
  REQUIRE(absl::IsFailedPrecondition(missing_map_or.status()));
  CHECK(missing_map_or.status().message().find("collective compatibility map") != std::string::npos);

  harness.shutdown();
  tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(artifact_root, cleanup_ec);
}

TEST_CASE(
    "MaterializationFacade executes mapped partial const fill without touching uncovered bytes",
    "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  constexpr uint64_t kTotalSize = 16;
  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_mapped_partial_const_fill";
  std::filesystem::create_directories(temp_root);
  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  void* gpu_buffer = nullptr;
  auto alloc_status = tensorcast::cuda::malloc(&gpu_buffer, kTotalSize);
  REQUIRE(alloc_status.ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };

  std::array<uint8_t, kTotalSize> initial{};
  initial.fill(0xAA);
  auto init_status = tensorcast::cuda::memcpy(gpu_buffer, initial.data(), kTotalSize, cudaMemcpyHostToDevice);
  REQUIRE(init_status.ok());

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{.base_ptr = gsl::not_null<void*>{gpu_buffer}, .length = kTotalSize});
  target_layout.total_size = kTotalSize;

  tensorcast::store::runtime::ingestion::strategy::ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = "cgid:artifact_partial_const_fill";
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"tensor":[0,16,[8],[1],"torch.uint16",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan =
      tensorcast::store::materialization::contracts::RepresentationWorkPlan{
          .items =
              {
                  tensorcast::store::materialization::contracts::RepresentationWorkItem{
                      .kind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kConstFill,
                      .dst_name = "tensor",
                      .dst_spec =
                          tensorcast::store::materialization::contracts::RepresentationTensorSpec{
                              .name = "tensor",
                              .shape = {8},
                              .stride = {1},
                              .dtype = "torch.uint16",
                              .logical_offset = 0,
                              .logical_length = 16,
                              .storage_offset = 0,
                              .element_size = 2,
                          },
                      .fill_rule =
                          tensorcast::store::materialization::contracts::FillRule{
                              .constant_value = {0x34, 0x12},
                              .destination_range =
                                  tensorcast::store::materialization::contracts::TensorCoordinateSpec{
                                      .axes =
                                          {
                                              tensorcast::store::materialization::contracts::TensorAxisRange{
                                                  .dim = 0,
                                                  .start = 2,
                                                  .end = 6,
                                              },
                                          },
                                  },
                          },
                      .committed_bytes = 8,
                  },
              },
      };

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:artifact_partial_const_fill";
  hints.allow_disk = false;
  hints.allow_p2p = false;

  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kLocalTypedOnly,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .local_typed_bytes = 8,
          .local_fill_bytes = 8,
      },
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "local_typed_only",
          .planned_local_typed_bytes = 8,
      });

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or =
      harness.facade->materialize_mapped_into_target(target_device, prepared_execution, hints, std::nullopt);
  REQUIRE(result_or.ok());

  std::array<uint8_t, kTotalSize> host_out{};
  auto copy_status = tensorcast::cuda::memcpy(host_out.data(), gpu_buffer, kTotalSize, cudaMemcpyDeviceToHost);
  REQUIRE(copy_status.ok());
  for (size_t index = 0; index < host_out.size(); index += 2) {
    const size_t element = index / 2;
    if (element >= 2 && element < 6) {
      CHECK(host_out[index] == 0x34);
      CHECK(host_out[index + 1] == 0x12);
    } else {
      CHECK(host_out[index] == 0xAA);
      CHECK(host_out[index + 1] == 0xAA);
    }
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade executes mapped scalar broadcast fill from local source", "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  constexpr uint64_t kSourceSize = 4;
  constexpr uint64_t kTargetSize = 8;
  const std::string artifact_id = "cgid:artifact_scalar_fill";
  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_mapped_scalar_fill";
  std::filesystem::create_directories(temp_root);
  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  loading::InlineBufferSource source{.data = nullptr, .size_bytes = kSourceSize};
  tensorcast::store::replica::ReplicaConfig cfg{
      .source = source,
      .artifact_identifier = artifact_id,
      .device_type = DeviceType::CPU,
      .local_device_id = -1,
      .pinned_buffer_pool = harness.runtime_context().pinned_buffer_pool(),
      .async_runtime =
          gsl::not_null<std::shared_ptr<tensorcast::common::AsyncRuntime>>{harness.runtime_context().async_runtime()},
      .artifact_chunk_bytes = harness.options().artifact_chunk_bytes,
      .expected_artifact_size = kSourceSize,
      .materialization_strategy = harness.options().materialization_strategy,
  };
  auto canonical_or = tensorcast::store::replica::Replica::create(cfg);
  REQUIRE(canonical_or.ok());
  auto canonical_replica = std::shared_ptr<tensorcast::store::replica::Replica>(std::move(canonical_or.value()));
  CHECK_OK(canonical_replica->get_memory_manager().allocate_memory(MemoryLocation::CPU));
  auto cpu_ptrs = canonical_replica->get_data_pointer(MemoryLocation::CPU);
  REQUIRE(cpu_ptrs.size() == 1);
  auto* cpu_ptr = static_cast<uint8_t*>(cpu_ptrs.front());
  REQUIRE(cpu_ptr != nullptr);
  cpu_ptr[0] = 10;
  cpu_ptr[1] = 20;
  cpu_ptr[2] = 30;
  cpu_ptr[3] = 40;
  CHECK_OK(canonical_replica->mark_loaded(MemoryLocation::CPU));
  canonical_replica->set_ready_signal(MemoryLocation::CPU, absl::OkStatus());
  loading::ReplicaKey canonical_key{
      .artifact_id = artifact_id,
      .view_id = std::nullopt,
      .device = {.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      .replica = 0,
  };
  CHECK_OK(harness.replica_runtime().registry().emplace(canonical_key, gsl::not_null{canonical_replica}));

  void* gpu_buffer = nullptr;
  auto alloc_status = tensorcast::cuda::malloc(&gpu_buffer, kTargetSize);
  REQUIRE(alloc_status.ok());
  absl::Cleanup free_gpu = [&]() {
    auto st = tensorcast::cuda::free(gpu_buffer);
    (void)st;
  };

  loading::IntoTargetLayout target_layout;
  target_layout.storages.push_back(
      loading::IntoTargetStorage{.base_ptr = gsl::not_null<void*>{gpu_buffer}, .length = kTargetSize});
  target_layout.total_size = kTargetSize;

  tensorcast::store::runtime::ingestion::strategy::ResolvedMaterializationPlan resolved_plan;
  resolved_plan.artifact_id = artifact_id;
  resolved_plan.generation = 1;
  resolved_plan.canonical_index_json = R"({"scalar_src":[0,4,[4],[1],"torch.uint8",0]})";
  resolved_plan.target_layout = target_layout;
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  resolved_plan.representation_transform_contract =
      tensorcast::store::materialization::contracts::RepresentationTransformContract{
          .source_byte_space = byte_space,
          .target_representation =
              {.family = "ephemeral_into_target",
               .realization_kind =
                   tensorcast::store::materialization::contracts::RealizationKind::kEphemeralIntoTarget},
      };
  resolved_plan.representation_work_plan =
      tensorcast::store::materialization::contracts::RepresentationWorkPlan{
          .items =
              {
                  tensorcast::store::materialization::contracts::RepresentationWorkItem{
                      .kind =
                          tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kScalarBroadcastFill,
                      .dst_name = "filled",
                      .dst_spec =
                          tensorcast::store::materialization::contracts::RepresentationTensorSpec{
                              .name = "filled",
                              .shape = {8},
                              .stride = {1},
                              .dtype = "torch.uint8",
                              .logical_offset = 0,
                              .logical_length = 8,
                              .storage_offset = 0,
                              .element_size = 1,
                          },
                      .sources =
                          {
                              tensorcast::store::materialization::contracts::RepresentationWorkSourceFragment{
                                  .fragment =
                                      tensorcast::store::materialization::contracts::SourceFragment{
                                          .source_spec =
                                              tensorcast::store::materialization::contracts::RepresentationTensorSpec{
                                                  .name = "scalar_src",
                                                  .shape = {4},
                                                  .stride = {1},
                                                  .dtype = "torch.uint8",
                                                  .logical_offset = 0,
                                                  .logical_length = 4,
                                                  .storage_offset = 0,
                                                  .element_size = 1,
                                              },
                                          .source_range =
                                              tensorcast::store::materialization::contracts::TensorCoordinateSpec{
                                                  .axes =
                                                      {
                                                          tensorcast::store::materialization::contracts::TensorAxisRange{
                                                              .dim = 0,
                                                              .start = 2,
                                                              .end = 3,
                                                          },
                                                      },
                                              },
                                          .destination_range = {},
                                      },
                              },
                          },
                      .committed_bytes = 8,
                  },
              },
      };

  loading::MaterializeHints hints;
  hints.artifact_id = artifact_id;
  hints.allow_disk = false;
  hints.allow_p2p = false;

  auto prepared_execution = make_prepared_source_bound_execution_plan(resolved_plan);
  prepared_execution.strategy_plan = make_source_bound_strategy_plan(
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionMode::kLocalTypedOnly,
      tensorcast::store::runtime::ingestion::strategy::SourceBoundLanePlan{
          .local_typed_bytes = kTargetSize,
      },
      tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
          .execution_plan_kind = "local_typed_only",
          .planned_local_typed_bytes = kTargetSize,
      });

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  auto result_or =
      harness.facade->materialize_mapped_into_target(target_device, prepared_execution, hints, std::nullopt);
  REQUIRE(result_or.ok());

  std::array<uint8_t, kTargetSize> host_out{};
  auto copy_status = tensorcast::cuda::memcpy(host_out.data(), gpu_buffer, kTargetSize, cudaMemcpyDeviceToHost);
  REQUIRE(copy_status.ok());
  for (uint8_t value : host_out) {
    CHECK(value == 30);
  }

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}

TEST_CASE("MaterializationFacade AUTO view route falls back to canonical transport", "[materialization_facade]") {
  SKIP_IF_NO_CUDA();

  auto temp_root = std::filesystem::temp_directory_path() / "materialization_facade_view_route_fallback";
  std::filesystem::create_directories(temp_root);

  FacadeHarness harness(MakeOptions(temp_root));
  harness.initialize();

  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->allow_view_transport = false;
  gs_client->allow_replica_transport = true;
  harness.runtime_context().set_global_store_client_for_testing(gs_client);

  loading::MaterializeHints hints;
  hints.artifact_id = "cgid:artifact_view_route_fallback";
  hints.transport_wait_timeout = std::chrono::milliseconds(5000);
  loading::VariantIdentity variant;
  variant.canonical_artifact_id = hints.artifact_id;
  variant.view_id = "view:tp0";
  tensorcast::store::loader::ViewPlan view_plan;
  view_plan.is_identity = false;
  view_plan.view_size_bytes = 16;
  view_plan.view_index_json = R"({"tensor":[0,16]})";
  view_plan.selection.map.total_bytes = 16;
  view_plan.selection.map.num_sources = 1;
  view_plan.selection.map.segments.push_back(
      tensorcast::store::loader::ByteRangeSegment{
          .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 16,
          .src_offset = 0,
          .source_index = 0,
      });
  variant.cached_plan = view_plan;
  hints.variant = std::move(variant);

  DeviceKey target_device{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  loading::ReplicaHandle p2p_handle;
  p2p_handle.replica_key.artifact_id = hints.artifact_id;
  p2p_handle.replica_key.view_id = "view:tp0";
  p2p_handle.replica_key.device = target_device;
  harness.fake_pipeline->set_next_p2p_result(std::move(p2p_handle));

  auto handle_or = harness.facade->materialize_replica(target_device, loading::MaterializeMode::AUTO, hints);
  REQUIRE(handle_or.ok());
  CHECK(harness.fake_pipeline->disk_invocations().empty());
  CHECK(harness.fake_pipeline->p2p_invocations().size() == 1);
  CHECK(gs_client->view_requests.size() == 1);
  CHECK(gs_client->view_requests.front() == "view:tp0");
  CHECK(gs_client->replica_requests.size() == 1);
  REQUIRE(gs_client->view_request_wait_timeouts_ms.size() == 1);
  REQUIRE(gs_client->replica_request_wait_timeouts_ms.size() == 1);
  CHECK(gs_client->view_request_wait_timeouts_ms.front() > 0);
  CHECK(gs_client->view_request_wait_timeouts_ms.front() < gs_client->replica_request_wait_timeouts_ms.front());
  CHECK(gs_client->replica_request_wait_timeouts_ms.front() == 5000);

  const auto& invocation = harness.fake_pipeline->p2p_invocations().front();
  REQUIRE(invocation.hints.variant.has_value());
  REQUIRE(invocation.hints.variant->view_id.has_value());
  CHECK(*invocation.hints.variant->view_id == "view:tp0");

  harness.shutdown();
  std::error_code cleanup_ec;
  std::filesystem::remove_all(temp_root, cleanup_ec);
}
