// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/ingestion/materialization_service.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "core/common/artifact_identity.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/cuda/cuda_ipc.h"
#include "core/store/components/replica_registry.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/transfer_helpers.h"
#include "core/store/replica/unified_memory_authority.h"
#include "gsl/pointers"
#include "nlohmann/json.hpp"

namespace tensorcast::store::runtime::ingestion {

namespace {

using common::memory::MemoryLocation;
using loading::InlineBufferSource;
using loading::TransformPlacement;
using replica::MemoryState;

absl::Status copy_local_cpu_source_to_gpu_replica(
    const std::shared_ptr<replica::Replica>& src_replica,
    const std::shared_ptr<replica::Replica>& dst_replica,
    const loading::MaterializationRequest& request,
    const MaterializationDeps& deps,
    void* cpu_src_ptr,
    uint64_t expected_size) {
  if (cpu_src_ptr == nullptr) {
    return absl::FailedPreconditionError("CPU source pointer unavailable for local CPU copy");
  }
  if (expected_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return absl::OutOfRangeError("Artifact size exceeds host copy limits");
  }

  const auto current_gpu_state = dst_replica->get_memory_state(MemoryLocation::GPU);
  if (current_gpu_state == MemoryState::LOADED) {
    return absl::OkStatus();
  }

  // Reset stale GPU state and ready signal so this copy can publish a fresh completion signal.
  absl::Status release_status = dst_replica->release_memory(MemoryLocation::GPU);
  if (!release_status.ok() && !absl::IsNotFound(release_status)) {
    return release_status;
  }

  auto& memory_manager = dst_replica->get_memory_manager();
  absl::Status alloc_status = memory_manager.allocate_memory(MemoryLocation::GPU);
  if (!alloc_status.ok()) {
    return alloc_status;
  }
  const auto gpu_ptrs = memory_manager.get_pointer(MemoryLocation::GPU);
  if (gpu_ptrs.empty() || gpu_ptrs[0] == nullptr) {
    return absl::FailedPreconditionError("GPU pointer unavailable for local CPU copy");
  }

  const auto total_bytes = static_cast<size_t>(expected_size);
  if (total_bytes > 0) {
    const size_t slice_bytes = deps.memory_pool->slice_bytes();
    if (slice_bytes == 0) {
      return absl::FailedPreconditionError("Pinned buffer pool slice size is zero");
    }
    const size_t needed_chunks = (total_bytes + slice_bytes - 1) / slice_bytes;
    const size_t pool_capacity = deps.memory_pool->capacity_slices();
    const size_t max_chunks = std::max<size_t>(1, deps.streaming_buffer_chunks);
    const size_t num_chunks = std::max<size_t>(1, std::min({needed_chunks, max_chunks, pool_capacity}));
    auto streaming_buf =
        std::make_shared<common::memory::StreamingPinnedBuffer>(num_chunks, slice_bytes, deps.memory_pool);
    auto init_status = streaming_buf->initialize(deps.pinned_memory_timeout);
    if (!init_status.ok()) {
      return init_status;
    }
    auto copy_status = replica::perform_copy_cpu_to_gpu_streaming(
        src_replica->artifact_id(),
        static_cast<uint32_t>(request.target_device().ordinal),
        streaming_buf,
        gsl::not_null<void*>{gpu_ptrs[0]},
        total_bytes,
        gsl::not_null<void*>{cpu_src_ptr},
        src_replica->get_memory_manager().memory_authority(),
        src_replica->replica_key());
    if (!copy_status.ok()) {
      return copy_status;
    }
  }

  auto mark_status = dst_replica->mark_loaded(MemoryLocation::GPU);
  if (!mark_status.ok()) {
    return mark_status;
  }
  dst_replica->set_ready_signal(MemoryLocation::GPU, absl::OkStatus());
  return absl::OkStatus();
}

absl::Status validate_mi2_descriptor_matches_request(const loading::MaterializationRequest& request) {
  if (!absl::StartsWith(request.canonical_artifact_id(), "mi2:")) {
    return absl::OkStatus();
  }
  if (!request.has_disk_source()) {
    return absl::InvalidArgumentError(
        "Content-addressed load requires a disk source so the descriptor can be validated");
  }
  const std::filesystem::path descriptor_path = request.disk_source()->path / "artifact_descriptor.json";
  std::error_code ec;
  if (!std::filesystem::exists(descriptor_path, ec)) {
    return absl::FailedPreconditionError(
        absl::StrCat("artifact_descriptor.json required for content-addressed load: ", descriptor_path.string()));
  }
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("Failed to stat descriptor at ", descriptor_path.string()));
  }

  try {
    std::ifstream stream(descriptor_path);
    if (!stream.is_open()) {
      return absl::FailedPreconditionError(
          absl::StrCat("Cannot open artifact_descriptor.json at ", descriptor_path.string()));
    }
    nlohmann::json desc;
    stream >> desc;
    if (!desc.contains("artifact_id") || !desc["artifact_id"].is_string()) {
      return absl::FailedPreconditionError(
          absl::StrCat("artifact_descriptor.json missing artifact_id at ", descriptor_path.string()));
    }
    const std::string descriptor_id = desc["artifact_id"].get<std::string>();
    if (descriptor_id != request.canonical_artifact_id()) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "Descriptor artifact_id mismatch: expected ", request.canonical_artifact_id(), " got ", descriptor_id));
    }
  } catch (const std::exception& ex) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse artifact_descriptor.json at ", descriptor_path.string(), ": ", ex.what()));
  }
  return absl::OkStatus();
}

} // namespace

MaterializationService::MaterializationService(MaterializationDeps deps) : deps_(std::move(deps)) {
  ABSL_CHECK(deps_.async_runtime != nullptr) << "MaterializationDeps.async_runtime is required";
}

absl::StatusOr<ReplicaHandle> MaterializationService::execute(const MaterializationRequest& request) {
  auto existing_or = try_reuse_replica(request);
  if (existing_or.ok()) {
    return *existing_or;
  }
  if (!absl::IsNotFound(existing_or.status())) {
    return existing_or.status();
  }
  auto local_or = copy_from_local_cpu(request);
  if (local_or.ok()) {
    return *local_or;
  }
  if (!absl::IsNotFound(local_or.status())) {
    return local_or.status();
  }

  switch (request.mode()) {
    case MaterializeMode::COPY_ONLY:
      return copy_from_peer(request);
    case MaterializeMode::LOAD_ONLY:
      return load_from_disk(request);
    case MaterializeMode::AUTO:
      return run_auto(request);
  }

  return absl::InternalError("Invalid MaterializeMode");
}

absl::StatusOr<ReplicaHandle> MaterializationService::try_reuse_replica(const MaterializationRequest& request) const {
  auto existing_or = deps_.replica_registry->find(request.replica_key());
  if (!existing_or.ok()) {
    return existing_or.status();
  }

  const auto& replica = existing_or.value();
  std::optional<int> gpu_device =
      request.target_is_gpu() ? std::optional<int>(request.target_device().ordinal) : std::nullopt;
  (void)replica->ensure_loaded_async(request.target_location(), deps_.num_threads, gpu_device);
  if (request.target_is_gpu()) {
    const auto gpu_state = replica->get_memory_state(MemoryLocation::GPU);
    const auto cpu_state = replica->get_memory_state(MemoryLocation::CPU);
    if (gpu_state == MemoryState::ALLOCATED && cpu_state != MemoryState::LOADED) {
      return absl::NotFoundError("reuse replica has no loaded CPU source; falling back");
    }
  }
  auto ready_signal = replica->ready_signal_for(request.target_location());
  if (ready_signal && ready_signal->is_ready()) {
    const absl::Status load_status = std::move(ready_signal->subscribe()).get();
    if (!load_status.ok()) {
      // Reuse path is advisory. If we can detect an immediate load failure (e.g.
      // source CPU already evicted), downgrade to NotFound so execute() can fall
      // through to local CPU copy / orchestrator / disk fallback.
      if (absl::IsFailedPrecondition(load_status) || absl::IsNotFound(load_status)) {
        return absl::NotFoundError(load_status.message());
      }
      return load_status;
    }
  }
  return build_handle(request, replica, std::move(ready_signal), loading::MaterializationSource::kLocalReplica);
}

absl::StatusOr<ReplicaHandle> MaterializationService::copy_from_local_cpu(const MaterializationRequest& request) const {
  if (request.mode() != MaterializeMode::AUTO) {
    return absl::NotFoundError("Local CPU copy only applies to AUTO materialization");
  }
  if (!request.target_is_gpu()) {
    return absl::NotFoundError("Local CPU copy requires a GPU target");
  }
  if (request.canonical_artifact_id().empty()) {
    return absl::NotFoundError("Local CPU copy requires a canonical artifact id");
  }

  const auto candidates = deps_.replica_registry->find_by_artifact(request.canonical_artifact_id());
  for (const auto& cand_key : candidates) {
    if (cand_key.device.type != DeviceType::CPU) {
      continue;
    }
    if (cand_key.view_id != request.requested_view_id()) {
      continue;
    }

    auto src_or = deps_.replica_registry->find(cand_key);
    if (!src_or.ok()) {
      continue;
    }
    const auto& src_replica = src_or.value();
    if (src_replica->get_memory_state(MemoryLocation::CPU) != MemoryState::LOADED) {
      continue;
    }
    const auto cpu_ptrs = src_replica->get_data_pointer(MemoryLocation::CPU);
    if (cpu_ptrs.empty() || cpu_ptrs[0] == nullptr) {
      continue;
    }

    uint64_t expected_size = 0;
    if (auto sz_or = src_replica->get_artifact_size(); sz_or.ok()) {
      expected_size = *sz_or;
    } else {
      return sz_or.status();
    }

    auto existing_dst_or = deps_.replica_registry->find(request.replica_key());
    if (existing_dst_or.ok()) {
      const auto& existing_dst = existing_dst_or.value();
      auto copy_status =
          copy_local_cpu_source_to_gpu_replica(src_replica, existing_dst, request, deps_, cpu_ptrs[0], expected_size);
      if (!copy_status.ok()) {
        return copy_status;
      }
      return build_handle(
          request,
          existing_dst,
          existing_dst->ready_signal_for(request.target_location()),
          loading::MaterializationSource::kLocalReplica);
    }
    if (!absl::IsNotFound(existing_dst_or.status())) {
      return existing_dst_or.status();
    }

    replica::ReplicaConfig cfg = build_copy_replica_config(
        request, expected_size, src_replica, src_replica->get_memory_manager().memory_tier_config());

    auto dst_or = replica::Replica::create(cfg);
    if (!dst_or.ok()) {
      return dst_or.status();
    }
    auto dst_replica = std::shared_ptr<replica::Replica>(std::move(dst_or.value()));
    auto copy_status =
        copy_local_cpu_source_to_gpu_replica(src_replica, dst_replica, request, deps_, cpu_ptrs[0], expected_size);
    if (!copy_status.ok()) {
      return copy_status;
    }

    absl::Status emplace_status = deps_.replica_registry->emplace(request.replica_key(), gsl::not_null{dst_replica});
    if (absl::IsAlreadyExists(emplace_status)) {
      return reuse_existing_replica(request, loading::MaterializationSource::kLocalReplica);
    }
    if (!emplace_status.ok()) {
      return emplace_status;
    }
    return build_handle(
        request,
        dst_replica,
        dst_replica->ready_signal_for(request.target_location()),
        loading::MaterializationSource::kLocalReplica);
  }

  return absl::NotFoundError("No suitable CPU replica for local copy");
}

absl::StatusOr<ReplicaHandle> MaterializationService::copy_from_peer(const MaterializationRequest& request) const {
  if (!request.target_is_gpu()) {
    return absl::InvalidArgumentError("COPY_ONLY mode requires a GPU target device");
  }

  if (request.canonical_artifact_id().empty()) {
    return absl::InvalidArgumentError("COPY_ONLY requires hints.artifact_id (canonical artifact identifier)");
  }

  const auto candidates = deps_.replica_registry->find_by_artifact(request.canonical_artifact_id());
  for (const auto& cand_key : candidates) {
    if (cand_key.device.type != DeviceType::GPU) {
      continue;
    }
    if (cand_key.view_id != request.requested_view_id()) {
      continue;
    }

    auto src_or = deps_.replica_registry->find(cand_key);
    if (!src_or.ok()) {
      continue;
    }
    const auto& src_replica = src_or.value();
    if (src_replica->get_memory_state(MemoryLocation::GPU) != MemoryState::LOADED) {
      continue;
    }

    uint64_t expected_size = 0;
    if (auto sz_or = src_replica->get_artifact_size(); sz_or.ok()) {
      expected_size = *sz_or;
    } else {
      return sz_or.status();
    }

    replica::ReplicaConfig cfg = build_copy_replica_config(request, expected_size, src_replica, std::nullopt);

    auto dst_or = replica::Replica::create(cfg);
    if (!dst_or.ok()) {
      return dst_or.status();
    }
    auto dst_replica = std::shared_ptr<replica::Replica>(std::move(dst_or.value()));
    {
      absl::Status emplace_status = deps_.replica_registry->emplace(request.replica_key(), gsl::not_null{dst_replica});
      if (absl::IsAlreadyExists(emplace_status)) {
        VLOG(1) << "Replica already present for COPY_ONLY dst_key (will reuse existing instance): "
                << request.replica_key();
        return reuse_existing_replica(request, loading::MaterializationSource::kLocalReplica);
      }
      if (!emplace_status.ok()) {
        return emplace_status;
      }
    }

    absl::Status copy_st = dst_replica->copy_from(*src_replica);
    auto ready_signal = std::make_shared<common::ReadySignal<absl::Status>>();
    ready_signal->set_value(copy_st);
    return build_handle(request, dst_replica, std::move(ready_signal), loading::MaterializationSource::kLocalReplica);
  }

  return absl::FailedPreconditionError(
      absl::StrCat(
          "No suitable source instance for COPY_ONLY mode (artifact_id=",
          request.canonical_artifact_id(),
          ", view_id=",
          request.requested_view_id().has_value() ? *request.requested_view_id() : std::string("<none>"),
          ")"));
}

absl::StatusOr<ReplicaHandle> MaterializationService::load_from_disk(const MaterializationRequest& request) const {
  if (!request.hints().allow_disk) {
    return absl::InvalidArgumentError("source_policy disallows disk ingestion");
  }
  if (!request.has_disk_source()) {
    return absl::InvalidArgumentError("LOAD_ONLY materialize paths require a disk source for disk ingestion");
  }
  if (absl::StartsWith(request.canonical_artifact_id(), "mi2:")) {
    auto validate_st = validate_mi2_descriptor_matches_request(request);
    if (!validate_st.ok()) {
      return validate_st;
    }
  }

  DiskSource disk_src = *request.disk_source();
  // Content-addressed (mi2) loads require descriptor + index for verification; CGID/local ids can stream without it.
  disk_src.require_descriptor = tensorcast::common::is_mi2_artifact_id(request.canonical_artifact_id());

  ReplicaTarget target;
  target.location.type =
      request.target_is_gpu() ? common::memory::MemoryLocation::GPU : common::memory::MemoryLocation::CPU;
  target.location.device_id = request.target_device().ordinal;

  auto handle = deps_.ingest_from_disk(request.canonical_artifact_id(), disk_src, target, request.hints());
  if (handle.ok() && handle->source == loading::MaterializationSource::kUnspecified) {
    handle->source = loading::MaterializationSource::kDisk;
  }
  return handle;
}

absl::StatusOr<ReplicaHandle> MaterializationService::run_auto(const MaterializationRequest& request) const {
  const auto id_kind = tensorcast::common::infer_artifact_id_kind(request.canonical_artifact_id());
  if (id_kind == tensorcast::common::ArtifactIdKind::kMi2 || id_kind == tensorcast::common::ArtifactIdKind::kCgid) {
    auto id_kind_or = tensorcast::common::validate_and_get_artifact_id_kind(request.canonical_artifact_id());
    if (!id_kind_or.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat(
              "AUTO materialize_replica requires a canonical artifact_id starting with mi2: or cgid:. ",
              "Provide MaterializeHints.artifact_id or VariantIdentity.canonical_artifact_id. Details: ",
              id_kind_or.status().message()));
    }
  }
  const bool allow_disk = request.hints().allow_disk;
  if (deps_.run_auto) {
    auto orchestrated_or = deps_.run_auto(request);
    if (orchestrated_or.ok()) {
      return *orchestrated_or;
    }
    if (!allow_disk || !request.has_disk_source()) {
      return orchestrated_or.status();
    }
    if (deps_.run_auto_handles_disk_fallback) {
      LOG(WARNING) << "Materialize AUTO callback failed (callback owns disk fallback): " << orchestrated_or.status();
      return orchestrated_or.status();
    }
    LOG(WARNING) << "Materialize AUTO callback failed: " << orchestrated_or.status() << "; falling back to disk load";
  }

  if (allow_disk && request.has_disk_source()) {
    return load_from_disk(request);
  }
  if (!allow_disk && request.has_disk_source()) {
    return absl::FailedPreconditionError("source_policy disallows disk fallback");
  }

  return absl::FailedPreconditionError(
      "AUTO materialize_replica requires a canonical artifact identifier (mi2: or cgid:) with Global Store routing or an explicit disk source");
}

replica::ReplicaConfig MaterializationService::build_copy_replica_config(
    const MaterializationRequest& request,
    uint64_t expected_size,
    const std::shared_ptr<replica::Replica>& src_replica,
    std::optional<MemoryTierConfig> memory_tier_config) const {
  InlineBufferSource ib_source{.data = nullptr, .size_bytes = expected_size};
  replica::ReplicaConfig cfg{
      .source = ib_source,
      .artifact_identifier = request.canonical_artifact_id(),
      .device_type = DeviceType::GPU,
      .local_device_id = request.target_device().ordinal,
      .pinned_buffer_pool = deps_.memory_pool,
      .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{deps_.async_runtime},
      .artifact_chunk_bytes = deps_.artifact_chunk_bytes,
      .expected_artifact_size = expected_size,
      .view_plan = src_replica->view_plan(),
      .byte_mapping_config = deps_.byte_mapping_config,
      .memory_tier_config = std::move(memory_tier_config)};
  cfg.pinned_memory_timeout = deps_.pinned_memory_timeout;
  cfg.streaming_buffer_chunks = deps_.streaming_buffer_chunks;
  cfg.view_id = request.requested_view_id();
  cfg.transform_placement = request.hints().variant ? request.hints().variant->placement : TransformPlacement::kServer;
  return cfg;
}

absl::StatusOr<ReplicaHandle> MaterializationService::reuse_existing_replica(
    const MaterializationRequest& request,
    loading::MaterializationSource source) const {
  auto existing_or = deps_.replica_registry->find(request.replica_key());
  if (!existing_or.ok()) {
    return existing_or.status();
  }
  const auto& existing = existing_or.value();
  std::optional<int> gpu_device =
      request.target_is_gpu() ? std::optional<int>(request.target_device().ordinal) : std::nullopt;
  (void)existing->ensure_loaded_async(request.target_location(), deps_.num_threads, gpu_device);
  if (request.target_is_gpu()) {
    const auto gpu_state = existing->get_memory_state(MemoryLocation::GPU);
    const auto cpu_state = existing->get_memory_state(MemoryLocation::CPU);
    if (gpu_state == MemoryState::ALLOCATED && cpu_state != MemoryState::LOADED) {
      return absl::NotFoundError("reuse replica has no loaded CPU source; falling back");
    }
  }
  auto ready_signal = existing->ready_signal_for(request.target_location());
  if (ready_signal && ready_signal->is_ready()) {
    const absl::Status load_status = std::move(ready_signal->subscribe()).get();
    if (!load_status.ok()) {
      if (absl::IsFailedPrecondition(load_status) || absl::IsNotFound(load_status)) {
        return absl::NotFoundError(load_status.message());
      }
      return load_status;
    }
  }
  return build_handle(request, existing, std::move(ready_signal), source);
}

ReplicaHandle MaterializationService::build_handle(
    const MaterializationRequest& request,
    const std::shared_ptr<replica::Replica>& replica,
    std::shared_ptr<common::ReadySignal<absl::Status>> ready_signal,
    loading::MaterializationSource source) const {
  ReplicaHandle handle;
  handle.replica_key = request.replica_key();
  handle.ready_signal = std::move(ready_signal);
  handle.cpu_state = replica->get_memory_state(MemoryLocation::CPU);
  handle.gpu_state = replica->get_memory_state(MemoryLocation::GPU);
  handle.source = source;

  if (request.target_is_gpu()) {
    const auto gpu_ptrs = replica->get_data_pointer(MemoryLocation::GPU);
    handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

    auto ipc_or = replica->get_memory_manager().get_ipc_handle();
    if (ipc_or.ok()) {
      handle.cuda_ipc_handle = cuda::IpcHandleBytes::from_native(*ipc_or);
    }
  } else {
    auto uma = replica->get_memory_manager().memory_authority();
    if (uma) {
      const loading::ReplicaKey& allocation_key = replica->replica_key();
      auto region_or = uma->get_cpu_memfd_region(allocation_key);
      if (region_or.ok()) {
        handle.cpu_memfd_region = loading::CpuMemfdRegion{
            .fd = region_or->fd,
            .size_bytes = region_or->size_bytes,
            .offset_bytes = region_or->offset_bytes,
        };
      }
    }
  }

  const auto& view_plan = replica->view_plan();
  if (view_plan.has_value() && !view_plan->is_identity) {
    handle.view_index_json = view_plan->view_index_json;
    const uint64_t view_size = view_plan->view_size_bytes;
    if (request.hints().need_view_data_hash && view_size > 0 && deps_.view_hash_computer) {
      const bool target_is_gpu = request.target_is_gpu();
      const bool target_loaded =
          target_is_gpu ? handle.gpu_state == MemoryState::LOADED : handle.cpu_state == MemoryState::LOADED;
      std::optional<int> gpu_device =
          target_is_gpu ? std::optional<int>(request.target_device().ordinal) : std::nullopt;
      if (target_loaded) {
        auto hash = deps_.view_hash_computer->hash_replica_view(
            *replica, request.target_location(), view_size, std::move(gpu_device));
        if (hash.has_value()) {
          handle.view_data_hash = std::move(hash);
        }
      }
    }
  }

  return handle;
}

} // namespace tensorcast::store::runtime::ingestion
