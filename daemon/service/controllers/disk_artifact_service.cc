// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/disk_artifact_service.h"

#include <system_error>
#include <utility>

#include "absl/time/time.h"
#include "daemon/service/controllers/materialization_disk_resolve_utils.h"
#include "daemon/util/grpc_peer_utils.h"
#include "daemon/util/status_utils.h"

namespace tensorcast::daemon {

using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

using materialization_disk_resolve::record_disk_resolution_outcome;

} // namespace

DiskArtifactService::DiskArtifactService(Dep d) : d_(std::move(d)) {
  if (!d_.storage_path.empty()) {
    std::error_code ec;
    storage_path_ = std::filesystem::weakly_canonical(d_.storage_path, ec);
    if (ec) {
      ec.clear();
      storage_path_ = d_.storage_path.lexically_normal();
    }
  }
}

grpc::Status DiskArtifactService::resolve_artifact_from_disk(
    RpcContext& rctx,
    const v2::ResolveArtifactFromDiskRequest& req,
    v2::ResolveArtifactFromDiskResponse& resp) {
  auto& span = rctx.span();
  const bool verify_checksums = req.verify_checksums();
  if (req.disk_path().empty()) {
    record_disk_resolution_outcome("invalid_argument");
    return {StatusCode::INVALID_ARGUMENT, "disk_path is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    record_disk_resolution_outcome("unavailable");
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  const bool loopback_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  if (!loopback_peer) {
    record_disk_resolution_outcome("permission_denied");
    return {StatusCode::PERMISSION_DENIED, "ResolveArtifactFromDisk is local-only (loopback/UDS)"};
  }

  span->SetAttribute("tc.store.verify_checksums", verify_checksums);
  auto resolved_or =
      materialization_disk_resolve::resolve_artifact_from_disk(req.disk_path(), storage_path_, verify_checksums);
  if (!resolved_or.ok()) {
    return to_grpc_status(resolved_or.status());
  }
  const auto& resolved = *resolved_or;
  resp.set_disk_path(resolved.normalized_disk_path.string());
  resp.set_artifact_id(resolved.artifact_id);
  resp.set_canonical_index_bytes(resolved.canonical_index_json);
  resp.set_generation(resolved.generation);
  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.disk.path", resolved.normalized_disk_path.string());
    span->SetAttribute("tc.artifact.id", resolved.artifact_id);
  }
  span->SetAttribute("tc.artifact.generation", static_cast<int64_t>(resolved.generation));
  d_.disk_imports.upsert_import(
      resolved.artifact_id,
      LocalDiskImportCatalog::Entry{
          .normalized_disk_path = resolved.normalized_disk_path.string(),
          .descriptor_present = resolved.descriptor_present,
          .index_multihash = resolved.index_multihash,
          .data_multihash = resolved.data_multihash,
          .generation = resolved.generation,
          .created_at = absl::Now(),
      });
  rctx.mark_success();
  return grpc::Status::OK;
}

grpc::Status DiskArtifactService::get_artifact_index_by_id(
    RpcContext& rctx,
    const v2::GetArtifactIndexByIdRequest& req,
    v2::GetArtifactIndexByIdResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.artifact_id());

  if (req.artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  auto bytes_or = d_.engine.get_canonical_index_by_id(req.artifact_id());
  if (!bytes_or.ok()) {
    return to_grpc_status(bytes_or.status());
  }
  resp.set_tensor_index_data(*bytes_or);
  rctx.mark_success();
  return grpc::Status::OK;
}

} // namespace tensorcast::daemon
