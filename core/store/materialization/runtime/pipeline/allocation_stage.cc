// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/runtime/pipeline/allocation_stage.h"

#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/time/time.h"
#include "core/store/components/eviction_service.h"
#include "core/store/replica/memory_state.h"

namespace tensorcast::store::materialization::runtime::pipeline {

namespace {

bool status_indicates_gpu_memory_pressure(const absl::Status& status) {
  if (status.ok()) {
    return false;
  }
  if (absl::IsResourceExhausted(status)) {
    return true;
  }
  const std::string_view message = status.message();
  return absl::StrContains(message, "Resource exhausted") || absl::StrContains(message, "resource exhausted") ||
      absl::StrContains(message, "out of memory") || absl::StrContains(message, "Out of memory") ||
      absl::StrContains(message, "cudaErrorMemoryAllocation") || absl::StrContains(message, "cudaErrorOutOfMemory");
}

bool should_retry_gpu_load_after_wait_failure(const absl::Status& wait_status) {
  if (wait_status.ok()) {
    return false;
  }
  if (absl::IsDeadlineExceeded(wait_status)) {
    return false;
  }
  return status_indicates_gpu_memory_pressure(wait_status);
}

absl::StatusOr<replica::ReplicaConfig> build_replica_config(IngestionContext& ctx) {
  replica::ReplicaConfig config{
      .source = ctx.disk.source,
      .artifact_identifier = ctx.artifact_identifier,
      .device_type = ctx.target_is_gpu ? DeviceType::GPU : DeviceType::CPU,
      .local_device_id = ctx.target_is_gpu ? ctx.target_device_id : -1,
      .pinned_buffer_pool =
          gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{ctx.runtime_context->pinned_buffer_pool()},
      .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{ctx.runtime_context->async_runtime()},
      .artifact_chunk_bytes = ctx.artifact_chunk_bytes,
      .expected_artifact_size = std::nullopt};
  config.pinned_memory_timeout =
      ctx.hints.pinned_timeout.count() > 0 ? ctx.hints.pinned_timeout : ctx.pinned_memory_timeout;
  config.max_buffer_bytes = ctx.hints.max_buffer_bytes;
  if (ctx.options != nullptr) {
    config.streaming_buffer_chunks = std::max<size_t>(1, ctx.options->streaming_buffer_chunks);
    config.byte_mapping_config = ctx.options->byte_mapping;
    config.materialization_strategy = ctx.options->materialization_strategy;
    config.memory_tier_config = ctx.options->memory_tier_config;
  }
  if (ctx.source_type == SourceType::kDisk) {
    config.canonical_index_json = ctx.verification.canonical_index_json;
    config.source_index_json = ctx.disk.source_index_json;
  }
  if (ctx.source_type == SourceType::kDisk && ctx.target_is_gpu && ctx.ordinary_disk_strategy_planner) {
    auto plan_or = ctx.ordinary_disk_strategy_planner(ctx);
    if (!plan_or.ok()) {
      LOG(WARNING) << "ordinary disk strategy planning failed for artifact_id=" << ctx.artifact_identifier << ": "
                   << plan_or.status() << "; falling back to generic byte-range execution";
    } else {
      config.execution_strategy_plan = std::move(*plan_or);
    }
  }
  config.collective_load_group = ctx.hints.collective_load_group;
  config.variant_identity = ctx.hints.variant;
  config.view_id = ctx.hints.variant ? ctx.hints.variant->view_id : std::nullopt;
  config.view_plan = ctx.resolved_view_plan;
  config.transform_placement = ctx.hints.variant ? ctx.hints.variant->placement : loading::TransformPlacement::kServer;
  if (ctx.resolved_view_plan.has_value()) {
    config.expected_artifact_size = ctx.resolved_view_plan->view_size_bytes;
  } else if (
      ctx.source_type == SourceType::kDisk && ctx.disk.source_index_json.has_value() &&
      ctx.verification.logical_total_size > 0) {
    config.expected_artifact_size = ctx.verification.logical_total_size;
  } else if (ctx.source_type == SourceType::kDisk && ctx.disk.source.expected_size.has_value()) {
    config.expected_artifact_size = ctx.disk.source.expected_size;
  }

  if (ctx.source_type == SourceType::kP2P) {
    config.source = ctx.p2p.source;
    config.local_device_id = ctx.target_device_id;
    config.p2p_comm_enabled = true;
    config.source_is_view = ctx.p2p.source.source_is_view;
  }

  return config;
}

absl::Status retry_gpu_load_with_eviction(
    IngestionContext& ctx,
    const std::shared_ptr<replica::Replica>& replica,
    std::optional<int> gpu_device) {
  size_t required_bytes = 0;
  if (auto sz_or = replica->get_artifact_size(); sz_or.ok()) {
    required_bytes = *sz_or;
  }

  auto evict_status = components::evict_for_gpu(
      ctx.replica_runtime->registry(),
      ctx.replica_runtime->device_manager(),
      ctx.runtime_context->metrics_collector(),
      ctx.target_device_id,
      required_bytes);

  if (!evict_status.ok()) {
    LOG(WARNING) << "GPU eviction did not free enough memory: " << evict_status;
    return evict_status;
  }

  {
    absl::Status release_status = replica->release_memory(common::memory::MemoryLocation::GPU);
    if (!release_status.ok()) {
      LOG(WARNING) << "release_memory(GPU) failed during retry after eviction: " << release_status;
      return release_status;
    }
  }

  auto load_sf =
      replica->ensure_loaded_async(common::memory::MemoryLocation::GPU, ctx.num_threads, std::move(gpu_device));
  ctx.load_signal = replica->ready_signal_for(common::memory::MemoryLocation::GPU);
  auto wait_status = replica->get_memory_manager().wait_for_state(
      common::memory::MemoryLocation::GPU,
      replica::MemoryState::LOADED,
      ctx.hints.pinned_timeout.count() > 0 ? absl::Milliseconds(ctx.hints.pinned_timeout.count())
                                           : absl::InfiniteDuration());

  if (!wait_status.ok()) {
    LOG(WARNING) << "Retry after eviction still failed: " << wait_status;
    return wait_status;
  }

  absl::Status load_status = std::move(load_sf).get();
  if (!load_status.ok()) {
    return load_status;
  }

  return absl::OkStatus();
}

absl::Status retry_if_stale_ready_signal(
    IngestionContext& ctx,
    common::memory::MemoryLocation target_location,
    std::optional<int> gpu_device,
    folly::SemiFuture<absl::Status>* load_sf) {
  ctx.load_signal = ctx.replica->ready_signal_for(target_location);
  if (!ctx.load_signal || !ctx.load_signal->is_ready()) {
    return absl::OkStatus();
  }

  const absl::Status ready_status = std::move(ctx.load_signal->subscribe()).get();
  if (ready_status.ok()) {
    return absl::OkStatus();
  }
  if (!absl::IsFailedPrecondition(ready_status) && !absl::IsNotFound(ready_status)) {
    return ready_status;
  }

  if (target_location == common::memory::MemoryLocation::GPU) {
    const auto gpu_state = ctx.replica->get_memory_state(common::memory::MemoryLocation::GPU);
    const auto cpu_state = ctx.replica->get_memory_state(common::memory::MemoryLocation::CPU);
    if (gpu_state == replica::MemoryState::ALLOCATED && cpu_state != replica::MemoryState::LOADED) {
      LOG(WARNING) << "Detected stale GPU ready signal with no loaded CPU source; forcing GPU reload";
    }
  }

  absl::Status release_status = ctx.replica->release_memory(target_location);
  if (!release_status.ok() && !absl::IsNotFound(release_status)) {
    return release_status;
  }

  *load_sf = ctx.replica->ensure_loaded_async(target_location, ctx.num_threads, std::move(gpu_device));
  ctx.load_signal = ctx.replica->ready_signal_for(target_location);
  return absl::OkStatus();
}

} // namespace

absl::Status AllocationStage::allocate(IngestionContext& ctx) {
  auto config_or = build_replica_config(ctx);
  if (!config_or.ok()) {
    return config_or.status();
  }
  auto config = std::move(*config_or);

  ctx.replica = ctx.replica_runtime->get_or_create_replica(ctx.artifact_identifier, config);
  if (!ctx.replica) {
    return absl::InternalError("Failed to create replica");
  }

  auto target_location = ctx.target_location;
  std::optional<int> gpu_device = ctx.target_is_gpu ? std::optional<int>(ctx.target_device_id) : std::nullopt;
  auto load_sf = ctx.replica->ensure_loaded_async(target_location, ctx.num_threads, gpu_device);
  auto stale_retry_status = retry_if_stale_ready_signal(ctx, target_location, gpu_device, &load_sf);
  if (!stale_retry_status.ok()) {
    return stale_retry_status;
  }

  if (ctx.source_type == SourceType::kP2P) {
    absl::Status status = std::move(load_sf).get();
    if (!status.ok()) {
      if (absl::IsResourceExhausted(status)) {
        LOG(WARNING) << "Resource exhausted, attempting memory eviction for P2P load";
        auto evict_status = ctx.replica_runtime->try_evict_memory_for_replica(ctx.p2p.source.size_bytes);
        if (evict_status.ok()) {
          load_sf = ctx.replica->ensure_loaded_async(target_location, ctx.num_threads, gpu_device);
          stale_retry_status = retry_if_stale_ready_signal(ctx, target_location, gpu_device, &load_sf);
          if (!stale_retry_status.ok()) {
            status = stale_retry_status;
          } else {
            status = std::move(load_sf).get();
          }
        }
      }
      if (!status.ok()) {
        return status;
      }
    }
    return absl::OkStatus();
  }

  absl::Duration wait_duration = (ctx.hints.pinned_timeout.count() > 0)
      ? absl::Milliseconds(ctx.hints.pinned_timeout.count())
      : absl::InfiniteDuration();
  auto wait_status =
      ctx.replica->get_memory_manager().wait_for_state(target_location, replica::MemoryState::LOADED, wait_duration);
  if (!wait_status.ok() && ctx.target_is_gpu && should_retry_gpu_load_after_wait_failure(wait_status)) {
    auto retry_status = retry_gpu_load_with_eviction(ctx, ctx.replica, gpu_device);
    if (!retry_status.ok()) {
      return retry_status;
    }
    return absl::OkStatus();
  }

  if (!wait_status.ok()) {
    return wait_status;
  }

  absl::Status load_status = std::move(load_sf).get();
  if (!load_status.ok()) {
    return load_status;
  }

  return absl::OkStatus();
}

} // namespace tensorcast::store::materialization::runtime::pipeline
