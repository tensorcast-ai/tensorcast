// Copyright (c) 2025, TensorCast Team.

#include "core/store/runtime/ingestion/materialization_service.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "core/common/artifact_identity.h"
#include "core/store/components/replica_registry.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/replica.h"
#include "gsl/pointers"
#include "nlohmann/json.hpp"

namespace tensorcast::store::runtime::ingestion {

namespace {

using common::memory::MemoryLocation;
using loading::InlineBufferSource;
using loading::TransformPlacement;
using replica::MemoryState;

absl::Status validate_mi2_descriptor_matches_request(const loading::MaterializationRequest& request) {
  if (!absl::StartsWith(request.canonical_artifact_id(), "mi2:")) {
    return absl::OkStatus();
  }
  if (!request.has_disk_path()) {
    return absl::InvalidArgumentError(
        "Content-addressed load requires MaterializeHints.disk_path so the descriptor can be validated");
  }

  const std::filesystem::path descriptor_path =
      std::filesystem::path(request.hints().disk_path) / "artifact_descriptor.json";
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

MaterializationService::MaterializationService(MaterializationDeps deps) : deps_(std::move(deps)) {}

absl::StatusOr<ReplicaHandle> MaterializationService::execute(const MaterializationRequest& request) {
  auto existing_or = try_reuse_replica(request);
  if (existing_or.ok()) {
    return *existing_or;
  }
  if (!absl::IsNotFound(existing_or.status())) {
    return existing_or.status();
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
  auto fut = replica->ensure_loaded_async(request.target_location(), deps_.num_threads, gpu_device);
  return build_handle(request, replica, fut, loading::MaterializationSource::kLocalReplica);
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

    InlineBufferSource ib_source{.data = nullptr, .size_bytes = expected_size};
    replica::ReplicaConfig cfg{
        .source = ib_source,
        .artifact_identifier = request.canonical_artifact_id(),
        .device_type = DeviceType::GPU,
        .local_device_id = request.target_device().ordinal,
        .pinned_buffer_pool = deps_.memory_pool,
        .artifact_chunk_bytes = deps_.artifact_chunk_bytes,
        .expected_artifact_size = expected_size,
        .view_plan = src_replica->view_plan(),
        .memory_tier_config = std::nullopt};
    cfg.pinned_memory_timeout = deps_.pinned_memory_timeout;
    cfg.view_id = request.requested_view_id();
    cfg.transform_placement =
        request.hints().variant ? request.hints().variant->placement : TransformPlacement::kServer;

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
      } else if (!emplace_status.ok()) {
        return emplace_status;
      }
    }

    absl::Status copy_st = dst_replica->copy_from(*src_replica);
    std::promise<absl::Status> p;
    p.set_value(copy_st);
    return build_handle(request, dst_replica, p.get_future().share(), loading::MaterializationSource::kLocalReplica);
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
  if (!request.has_disk_path()) {
    return absl::InvalidArgumentError(
        "LOAD_ONLY materialize paths require MaterializeHints.disk_path for disk ingestion");
  }
  if (absl::StartsWith(request.canonical_artifact_id(), "mi2:")) {
    auto validate_st = validate_mi2_descriptor_matches_request(request);
    if (!validate_st.ok()) {
      return validate_st;
    }
  }

  DiskSource disk_src;
  disk_src.path = std::filesystem::path(request.hints().disk_path);
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
  if (deps_.run_auto) {
    auto orchestrated_or = deps_.run_auto(request);
    if (orchestrated_or.ok()) {
      return *orchestrated_or;
    }
    LOG(WARNING) << "Materialize AUTO callback failed: " << orchestrated_or.status() << "; falling back to disk load";
  }

  if (request.has_disk_path()) {
    return load_from_disk(request);
  }

  return absl::FailedPreconditionError(
      "AUTO materialize_replica requires a canonical artifact identifier (mi2: or cgid:) with Global Store routing or an explicit hints.disk_path");
}

ReplicaHandle MaterializationService::build_handle(
    const MaterializationRequest& request,
    const std::shared_ptr<replica::Replica>& replica,
    std::shared_future<absl::Status> ready_future,
    loading::MaterializationSource source) const {
  ReplicaHandle handle;
  handle.replica_key = request.replica_key();
  handle.ready_future = std::move(ready_future);
  handle.cpu_state = replica->get_memory_state(MemoryLocation::CPU);
  handle.gpu_state = replica->get_memory_state(MemoryLocation::GPU);
  handle.source = source;

  if (request.target_is_gpu()) {
    const auto gpu_ptrs = replica->get_data_pointer(MemoryLocation::GPU);
    handle.gpu_base_ptr = (!gpu_ptrs.empty() && gpu_ptrs[0] != nullptr) ? gpu_ptrs[0] : nullptr;

    auto ipc_or = replica->get_memory_manager().get_ipc_handle();
    if (ipc_or.ok()) {
      std::memcpy(handle.cuda_ipc_handle.bytes.data(), &(*ipc_or), sizeof(cudaIpcMemHandle_t));
    }
  }

  const auto& view_plan = replica->view_plan();
  if (view_plan.has_value() && !view_plan->is_identity) {
    handle.view_index_json = view_plan->view_index_json;
    const uint64_t view_size = view_plan->view_size_bytes;
    if (view_size > 0 && deps_.view_hash_computer) {
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
