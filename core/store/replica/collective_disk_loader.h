// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/common/memory/cuda_memory.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/contracts/view/view_plan.h"
#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"
#include "core/store/runtime/ingestion/materialization_strategy_types.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::replica {

struct SourceWindowCollectivePlan;

struct CollectiveDiskLoadRequest {
  loading::ReplicaKey replica_key;
  loading::CollectiveLoadGroupHint group;
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  materialization::contracts::RepresentationWorkPlan representation_work_plan;
  StoreEngineOptions::MaterializationStrategyConfig strategy_config;
  void* gpu_ptr{nullptr};
  int device_id{-1};
  std::shared_ptr<common::memory::GpuDeviceMemory> gpu_allocation;
};

struct CollectiveDiskLoadResult {
  bool handled{false};
  absl::Status status{absl::OkStatus()};
  runtime::ingestion::strategy::CollectiveExecutionMetrics metrics;
  std::string skip_reason;
};

struct CollectiveMappedTargetLoadOptions {
  uint64_t chunk_bytes{128ULL * 1024ULL * 1024ULL};
  uint32_t streaming_buffer_chunks{1};
  uint64_t merge_max_gap_bytes{256ULL * 1024ULL};
  uint64_t merge_max_amplification{4};
  StoreEngineOptions::MaterializationStrategyConfig strategy_config;
  bool enable_source_window_plan_cache{true};
};

struct CollectiveMappedTargetLoadRequest {
  std::string artifact_id;
  loading::CollectiveLoadGroupHint group;
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  materialization::contracts::RepresentationWorkPlan representation_work_plan;
  loader::ByteRangeMap collective_lane_map;
  loading::IntoTargetLayout target_layout;
  int device_id{-1};
};

struct CollectiveMappedTargetLoadResult {
  bool handled{false};
  absl::Status status{absl::OkStatus()};
  runtime::ingestion::strategy::CollectiveExecutionMetrics metrics;
  std::string skip_reason;
};

struct SourceWindowCollectiveMappedTargetLoadRequest {
  std::string artifact_id;
  loading::CollectiveLoadGroupHint group;
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  std::shared_ptr<const materialization::contracts::RepresentationWorkPlan> representation_work_plan_ref;
  materialization::contracts::RepresentationWorkPlan representation_work_plan;
  std::shared_ptr<const loading::IntoTargetLayout> target_layout_ref;
  loading::IntoTargetLayout target_layout;
  runtime::ingestion::strategy::SourceWindowCollectiveCandidateSummary candidate_summary;
  std::string source_index_digest;
  std::optional<loading::SourceWindowPreparedRealizationFacts> prepared_realization;
  int device_id{-1};
};

struct SourceWindowCollectiveMappedTargetLoadResult {
  bool handled{false};
  absl::Status status{absl::OkStatus()};
  runtime::ingestion::strategy::CollectiveExecutionMetrics metrics;
  std::string skip_reason;
  std::string plan_hash;
  bool plan_cache_hit{false};
};

struct SourceWindowCollectivePlanCacheStats {
  uint64_t hits{0};
  uint64_t misses{0};
  uint64_t entries{0};
};

struct SourceWindowRoutedProgramCachePrepareResult {
  bool prepared{false};
  absl::Status status{absl::OkStatus()};
  std::string skip_reason;
  std::string plan_hash;
  bool cache_hit{false};
  uint64_t runtime_chunk_count{0};
  uint64_t compiled_chunk_count{0};
  double program_build_sec{0.0};
  uint64_t program_build_threads{0};
};

struct LocalMappedTargetLoadRequest {
  std::string artifact_id;
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  materialization::contracts::RepresentationWorkPlan representation_work_plan;
  loader::ByteRangeMap data_lane_map;
  loading::IntoTargetLayout target_layout;
  StoreEngineOptions::MaterializationStrategyConfig strategy_config;
  int device_id{-1};
};

struct LocalMappedTargetLoadResult {
  bool handled{false};
  absl::Status status{absl::OkStatus()};
  runtime::ingestion::strategy::CollectiveExecutionMetrics metrics;
  loader::ByteRangeMap residual_data_map;
  uint64_t handled_bytes{0};
  std::string skip_reason;
};

struct LocalBatchedDiskLoadRequest {
  loading::ReplicaKey replica_key;
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  materialization::contracts::RepresentationWorkPlan representation_work_plan;
  StoreEngineOptions::MaterializationStrategyConfig strategy_config;
  void* gpu_ptr{nullptr};
  int device_id{-1};
  std::shared_ptr<common::memory::GpuDeviceMemory> gpu_allocation;
};

struct LocalBatchedDiskLoadResult {
  bool handled{false};
  absl::Status status{absl::OkStatus()};
  std::string skip_reason;
};

struct LocalBatchedPlanSummary {
  bool eligible{false};
  std::string reason;
  uint64_t requested_source_bytes{0};
  uint64_t unique_source_bytes{0};
  uint64_t peak_temporary_bytes{0};
  uint64_t batch_count{0};
  uint64_t dedup_saving_bytes{0};
  uint64_t direct_dedup_copy_bytes{0};
  size_t replicated_jobs{0};
  size_t dim0_jobs{0};
  size_t dim1_jobs{0};
};

struct LocalMappedSafetensorsAutoIoDecision {
  bool use_direct_aligned_edges{false};
  double page_cache_residency_ratio{-1.0};
  uint64_t buffered_probe_bytes{0};
  double buffered_probe_sec{-1.0};
  double buffered_probe_gib_per_sec{-1.0};
  bool direct_probe_attempted{false};
  bool direct_probe_supported{false};
  uint64_t direct_probe_bytes{0};
  int direct_probe_errno{0};
  std::string direct_probe_status;
  std::string reason;
};

absl::StatusOr<LocalBatchedPlanSummary> summarize_local_batched_disk_load(
    const materialization::contracts::RepresentationWorkPlan& representation_work_plan,
    const StoreEngineOptions::MaterializationStrategyConfig& strategy_config);

CollectiveDiskLoadResult try_collective_disk_load(
    const CollectiveDiskLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout);

CollectiveMappedTargetLoadResult try_collective_mapped_target_load(
    const CollectiveMappedTargetLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options);

SourceWindowCollectiveMappedTargetLoadResult try_source_window_collective_mapped_target_load(
    const SourceWindowCollectiveMappedTargetLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options);

LocalMappedTargetLoadResult try_local_mapped_target_load(
    const LocalMappedTargetLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options);

LocalBatchedDiskLoadResult try_local_batched_disk_load(
    const LocalBatchedDiskLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout);

absl::StatusOr<LocalMappedSafetensorsAutoIoDecision> choose_auto_local_mapped_safetensors_io_for_testing(
    absl::Span<const loader::SharedSafetensorsSegment> segments);

absl::Status warm_collective_clique_cache(const std::vector<int>& device_ids);

void clear_source_window_collective_plan_cache_for_testing();

SourceWindowCollectivePlanCacheStats source_window_collective_plan_cache_stats_for_testing();

void clear_source_window_routed_program_cache_for_testing();

SourceWindowCollectivePlanCacheStats source_window_routed_program_cache_stats_for_testing();

size_t source_window_compiled_routed_program_build_thread_count_for_testing(
    size_t chunk_count, uint32_t configured_thread_count);

absl::StatusOr<std::string> source_window_routed_program_cache_key_for_testing(
    std::string_view artifact_id,
    const SourceWindowCollectivePlan& plan,
    absl::Span<const SourceWindowCollectiveMappedTargetLoadRequest> requests,
    size_t configured_chunk_bytes,
    size_t max_collective_chunk_bytes,
    size_t max_stripe_bytes);

SourceWindowRoutedProgramCachePrepareResult prepare_source_window_routed_program_cache(
    std::string_view artifact_id,
    const SourceWindowCollectivePlan& plan,
    absl::Span<const SourceWindowCollectiveMappedTargetLoadRequest> requests,
    size_t configured_chunk_bytes,
    size_t max_collective_chunk_bytes,
    size_t max_stripe_bytes,
    uint32_t configured_build_threads);

SourceWindowRoutedProgramCachePrepareResult prepare_source_window_collective_routed_program_cache(
    absl::Span<const SourceWindowCollectiveMappedTargetLoadRequest> requests,
    const CollectiveMappedTargetLoadOptions& options);

SourceWindowRoutedProgramCachePrepareResult prepare_source_window_collective_plan_cache(
    absl::Span<const SourceWindowCollectiveMappedTargetLoadRequest> requests,
    const CollectiveMappedTargetLoadOptions& options);

SourceWindowRoutedProgramCachePrepareResult prepare_source_window_routed_program_cache_for_testing(
    std::string_view artifact_id,
    const SourceWindowCollectivePlan& plan,
    absl::Span<const SourceWindowCollectiveMappedTargetLoadRequest> requests,
    size_t configured_chunk_bytes,
    size_t max_collective_chunk_bytes,
    size_t max_stripe_bytes,
    uint32_t configured_build_threads = 0);

} // namespace tensorcast::store::replica
