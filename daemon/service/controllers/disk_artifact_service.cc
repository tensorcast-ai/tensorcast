// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/disk_artifact_service.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "daemon/util/grpc_peer_utils.h"
#include "daemon/util/path_utils.h"
#include "daemon/util/status_utils.h"

namespace tensorcast::daemon {

using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

constexpr double kDefaultImportCacheTtlSeconds = 600.0;
constexpr size_t kDefaultImportCacheMaxEntries = 1024;

std::optional<double> parse_positive_double_env(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return std::nullopt;
  }
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(raw, &end);
  if (end == raw || (end != nullptr && *end != '\0') || errno != 0 || !std::isfinite(value) || value < 0.0) {
    LOG(WARNING) << "Invalid " << name << "=" << raw << "; expected non-negative float";
    return std::nullopt;
  }
  return value;
}

std::optional<size_t> parse_positive_size_env(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return std::nullopt;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  if (end == raw || (end != nullptr && *end != '\0') || errno != 0 ||
      value > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
    LOG(WARNING) << "Invalid " << name << "=" << raw << "; expected non-negative integer";
    return std::nullopt;
  }
  return static_cast<size_t>(value);
}

v2::ImportArtifactPhase to_proto_phase(materialization_disk_resolve::ImportArtifactPhase phase) {
  using materialization_disk_resolve::ImportArtifactPhase;
  switch (phase) {
    case ImportArtifactPhase::kPrepare:
      return v2::IMPORT_ARTIFACT_PHASE_PREPARE;
    case ImportArtifactPhase::kScanSource:
      return v2::IMPORT_ARTIFACT_PHASE_SCAN_SOURCE;
    case ImportArtifactPhase::kReadHeaders:
      return v2::IMPORT_ARTIFACT_PHASE_READ_HEADERS;
    case ImportArtifactPhase::kBuildCanonicalIndex:
      return v2::IMPORT_ARTIFACT_PHASE_BUILD_CANONICAL_INDEX;
    case ImportArtifactPhase::kHashData:
      return v2::IMPORT_ARTIFACT_PHASE_HASH_DATA;
    case ImportArtifactPhase::kWriteRegistry:
      return v2::IMPORT_ARTIFACT_PHASE_WRITE_REGISTRY;
    case ImportArtifactPhase::kDone:
      return v2::IMPORT_ARTIFACT_PHASE_DONE;
    case ImportArtifactPhase::kError:
      return v2::IMPORT_ARTIFACT_PHASE_ERROR;
  }
  return v2::IMPORT_ARTIFACT_PHASE_UNSPECIFIED;
}

v2::ImportArtifactErrorCode to_proto_error_code(materialization_disk_resolve::ImportArtifactErrorCode code) {
  using materialization_disk_resolve::ImportArtifactErrorCode;
  switch (code) {
    case ImportArtifactErrorCode::kSourceNotFound:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_SOURCE_NOT_FOUND;
    case ImportArtifactErrorCode::kSourcePermissionDenied:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_SOURCE_PERMISSION_DENIED;
    case ImportArtifactErrorCode::kSourceFormatInvalid:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_SOURCE_FORMAT_INVALID;
    case ImportArtifactErrorCode::kSourceMutated:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_SOURCE_MUTATED;
    case ImportArtifactErrorCode::kRegistryIoFailure:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_REGISTRY_IO_FAILURE;
    case ImportArtifactErrorCode::kPolicyDeniedNonLocalPeer:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_POLICY_DENIED_NON_LOCAL_PEER;
    case ImportArtifactErrorCode::kUnspecified:
    default:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_UNSPECIFIED;
  }
}

double to_progress_percent(std::uint64_t processed_bytes, std::uint64_t total_bytes, bool done) {
  if (total_bytes == 0) {
    return done ? 100.0 : 0.0;
  }
  const double percent = (100.0 * static_cast<double>(processed_bytes)) / static_cast<double>(total_bytes);
  return std::clamp(percent, 0.0, 100.0);
}

void fill_import_response(
    const materialization_disk_resolve::ImportArtifactFromPathResult& imported,
    v2::ImportArtifactFromPathResponse& resp) {
  resp.set_artifact_id(imported.artifact_id);
  resp.set_canonical_index_bytes(imported.canonical_index_json);
  resp.set_generation(imported.generation);
  resp.set_import_state(v2::IMPORT_ARTIFACT_STATE_READY);
}

grpc::Status to_import_grpc_status(const absl::Status& status) {
  using materialization_disk_resolve::ImportArtifactErrorCode;
  const auto code = materialization_disk_resolve::classify_import_error(status);
  switch (code) {
    case ImportArtifactErrorCode::kSourceNotFound:
      return {StatusCode::NOT_FOUND, std::string(status.message())};
    case ImportArtifactErrorCode::kSourcePermissionDenied:
      return {StatusCode::PERMISSION_DENIED, std::string(status.message())};
    case ImportArtifactErrorCode::kSourceFormatInvalid:
      return {StatusCode::INVALID_ARGUMENT, std::string(status.message())};
    case ImportArtifactErrorCode::kSourceMutated:
      return {StatusCode::FAILED_PRECONDITION, std::string(status.message())};
    case ImportArtifactErrorCode::kRegistryIoFailure:
      return {StatusCode::UNAVAILABLE, std::string(status.message())};
    case ImportArtifactErrorCode::kPolicyDeniedNonLocalPeer:
      return {StatusCode::PERMISSION_DENIED, std::string(status.message())};
    case ImportArtifactErrorCode::kUnspecified:
    default:
      return to_grpc_status(status);
  }
}

materialization_disk_resolve::ImportArtifactErrorCode grpc_status_to_import_error(const grpc::Status& status) {
  using materialization_disk_resolve::ImportArtifactErrorCode;
  switch (status.error_code()) {
    case StatusCode::NOT_FOUND:
      return ImportArtifactErrorCode::kSourceNotFound;
    case StatusCode::PERMISSION_DENIED:
      return ImportArtifactErrorCode::kSourcePermissionDenied;
    case StatusCode::INVALID_ARGUMENT:
      return ImportArtifactErrorCode::kSourceFormatInvalid;
    case StatusCode::FAILED_PRECONDITION:
      return ImportArtifactErrorCode::kSourceMutated;
    case StatusCode::UNAVAILABLE:
      return ImportArtifactErrorCode::kRegistryIoFailure;
    default:
      return ImportArtifactErrorCode::kUnspecified;
  }
}

ArtifactSourceRegistry::FingerprintMap to_registry_fingerprints(
    const materialization_disk_resolve::SourceFingerprintMap& fingerprints) {
  ArtifactSourceRegistry::FingerprintMap out;
  out.reserve(fingerprints.size());
  for (const auto& [path, fp] : fingerprints) {
    out.insert_or_assign(
        path,
        ArtifactSourceRegistry::SourceFileFingerprint{
            .inode = fp.inode,
            .size = fp.size,
            .mtime_ns = fp.mtime_ns,
        });
  }
  return out;
}

} // namespace

DiskArtifactService::DiskArtifactService(Dep d)
    : d_(std::move(d)),
      import_cache_ttl_(import_cache_ttl_from_env()),
      import_cache_max_entries_(import_cache_max_entries_from_env()) {
  if (!d_.storage_path.empty()) {
    std::error_code ec;
    storage_path_ = std::filesystem::weakly_canonical(d_.storage_path, ec);
    if (ec) {
      ec.clear();
      storage_path_ = d_.storage_path.lexically_normal();
    }
  }
}

absl::Duration DiskArtifactService::import_cache_ttl_from_env() {
  const auto parsed = parse_positive_double_env("TENSORCAST_IMPORT_ARTIFACT_CACHE_TTL_SECONDS");
  return absl::Seconds(parsed.value_or(kDefaultImportCacheTtlSeconds));
}

size_t DiskArtifactService::import_cache_max_entries_from_env() {
  return parse_positive_size_env("TENSORCAST_IMPORT_ARTIFACT_CACHE_MAX_ENTRIES")
      .value_or(kDefaultImportCacheMaxEntries);
}

std::string DiskArtifactService::import_cache_key_for_path(
    const std::filesystem::path& normalized_path,
    bool verify_checksums) const {
  return absl::StrCat(normalized_path.string(), "|verify=", verify_checksums ? "1" : "0");
}

void DiskArtifactService::prune_expired_cache_locked(absl::Time now) {
  if (import_cache_ttl_ <= absl::ZeroDuration()) {
    import_cache_.clear();
    return;
  }
  for (auto it = import_cache_.begin(); it != import_cache_.end();) {
    if (now - it->second.cached_at > import_cache_ttl_) {
      auto erase_it = it;
      ++it;
      import_cache_.erase(erase_it);
    } else {
      ++it;
    }
  }
}

void DiskArtifactService::enforce_cache_capacity_locked() {
  if (import_cache_max_entries_ == 0) {
    import_cache_.clear();
    return;
  }
  while (import_cache_.size() > import_cache_max_entries_) {
    auto oldest_it = import_cache_.begin();
    for (auto it = import_cache_.begin(); it != import_cache_.end(); ++it) {
      if (it->second.cached_at < oldest_it->second.cached_at) {
        oldest_it = it;
      }
    }
    import_cache_.erase(oldest_it);
  }
}

absl::StatusOr<materialization_disk_resolve::ImportArtifactFromPathResult> DiskArtifactService::
    import_artifact_from_path_cached(
        const std::filesystem::path& normalized_path,
        bool verify_checksums,
        materialization_disk_resolve::ImportProgressCallback progress_cb) {
  const std::string key = import_cache_key_for_path(normalized_path, verify_checksums);

  std::shared_ptr<InflightImport> inflight;
  bool is_leader = false;
  {
    absl::MutexLock lock(&import_mu_);
    const absl::Time now = absl::Now();
    prune_expired_cache_locked(now);

    auto cache_it = import_cache_.find(key);
    if (cache_it != import_cache_.end()) {
      materialization_disk_resolve::record_disk_import_outcome("cache_hit");
      if (progress_cb) {
        materialization_disk_resolve::ImportProgressUpdate prep;
        prep.phase = materialization_disk_resolve::ImportArtifactPhase::kPrepare;
        prep.message = "served from daemon cache";
        progress_cb(prep);

        materialization_disk_resolve::ImportProgressUpdate done;
        done.phase = materialization_disk_resolve::ImportArtifactPhase::kDone;
        done.done = true;
        progress_cb(done);
      }
      return cache_it->second.imported;
    }

    auto [it, inserted] = inflight_imports_.try_emplace(key, std::make_shared<InflightImport>());
    inflight = it->second;
    is_leader = inserted;
  }

  if (!is_leader) {
    if (progress_cb) {
      materialization_disk_resolve::ImportProgressUpdate wait;
      wait.phase = materialization_disk_resolve::ImportArtifactPhase::kPrepare;
      wait.message = "waiting for inflight import";
      progress_cb(wait);
    }

    std::uint64_t seen_progress_version = 0;
    for (;;) {
      materialization_disk_resolve::ImportProgressUpdate progress_update;
      bool has_progress_update = false;
      bool done = false;
      absl::StatusOr<materialization_disk_resolve::ImportArtifactFromPathResult> imported_or(
          absl::UnknownError("inflight import incomplete"));
      {
        absl::MutexLock wait_lock(&inflight->mu);
        while (!inflight->done && inflight->progress_version <= seen_progress_version) {
          inflight->cv.Wait(&inflight->mu);
        }
        if (inflight->progress_version > seen_progress_version && inflight->has_progress) {
          seen_progress_version = inflight->progress_version;
          progress_update = inflight->latest_progress;
          has_progress_update = true;
        }
        if (inflight->done) {
          done = true;
          imported_or = inflight->imported;
        }
      }
      if (has_progress_update && progress_cb) {
        progress_cb(progress_update);
      }
      if (done) {
        return imported_or;
      }
    }
  }

  const auto publish_progress = [&](const materialization_disk_resolve::ImportProgressUpdate& update) {
    {
      absl::MutexLock progress_lock(&inflight->mu);
      inflight->latest_progress = update;
      inflight->has_progress = true;
      inflight->progress_version += 1;
      inflight->cv.SignalAll();
    }
    if (progress_cb) {
      progress_cb(update);
    }
  };

  auto imported_or = materialization_disk_resolve::import_artifact_from_path(
      normalized_path.string(), storage_path_, verify_checksums, publish_progress);

  bool terminal_emitted = false;
  {
    absl::MutexLock progress_lock(&inflight->mu);
    terminal_emitted = inflight->has_progress && (inflight->latest_progress.done || inflight->latest_progress.error);
  }
  if (!terminal_emitted) {
    materialization_disk_resolve::ImportProgressUpdate terminal;
    if (imported_or.ok()) {
      terminal.phase = materialization_disk_resolve::ImportArtifactPhase::kDone;
      terminal.done = true;
    } else {
      terminal.phase = materialization_disk_resolve::ImportArtifactPhase::kError;
      terminal.done = true;
      terminal.error = true;
      terminal.error_code = materialization_disk_resolve::classify_import_error(imported_or.status());
      terminal.message = std::string(imported_or.status().message());
    }
    publish_progress(terminal);
  }

  {
    absl::MutexLock wait_lock(&inflight->mu);
    inflight->imported = imported_or;
    inflight->done = true;
    inflight->cv.SignalAll();
  }

  {
    absl::MutexLock lock(&import_mu_);
    inflight_imports_.erase(key);
    if (imported_or.ok() && import_cache_ttl_ > absl::ZeroDuration()) {
      import_cache_.insert_or_assign(
          key,
          ImportCacheEntry{
              .imported = *imported_or,
              .cached_at = absl::Now(),
          });
      enforce_cache_capacity_locked();
    }
  }

  return imported_or;
}

grpc::Status DiskArtifactService::import_artifact_from_path(
    RpcContext& rctx,
    const v2::ImportArtifactFromPathRequest& req,
    v2::ImportArtifactFromPathResponse& resp) {
  auto& span = rctx.span();
  const bool verify_checksums = req.verify_checksums();
  if (req.path().empty()) {
    materialization_disk_resolve::record_disk_import_outcome("invalid_argument");
    return {StatusCode::INVALID_ARGUMENT, "path is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    materialization_disk_resolve::record_disk_import_outcome("unavailable");
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  const bool loopback_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  if (!loopback_peer) {
    materialization_disk_resolve::record_disk_import_outcome("permission_denied");
    return {StatusCode::PERMISSION_DENIED, "ImportArtifactFromPath is local-only (loopback/UDS)"};
  }

  auto normalized_or = normalize_disk_import_path(req.path(), storage_path_);
  if (!normalized_or.ok()) {
    materialization_disk_resolve::record_disk_import_outcome("invalid_argument");
    return to_grpc_status(normalized_or.status());
  }

  span->SetAttribute("tc.store.verify_checksums", verify_checksums);
  auto imported_or = import_artifact_from_path_cached(*normalized_or, verify_checksums);
  if (!imported_or.ok()) {
    return to_import_grpc_status(imported_or.status());
  }

  const auto& imported = *imported_or;
  fill_import_response(imported, resp);
  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.disk.path", imported.normalized_path.string());
    span->SetAttribute("tc.artifact.id", imported.artifact_id);
  }
  span->SetAttribute("tc.artifact.generation", static_cast<std::int64_t>(imported.generation));

  d_.source_registry.upsert_binding(
      imported.artifact_id,
      ArtifactSourceRegistry::Entry{
          .source_kind = ArtifactSourceRegistry::SourceKind::kLocalImport,
          .canonical_source_path = imported.normalized_path.string(),
          .source_disk_path = imported.normalized_path.string(),
          .descriptor_present = imported.descriptor_present,
          .index_multihash = imported.index_multihash,
          .data_multihash = imported.data_multihash,
          .generation = imported.generation,
          .file_fingerprints = to_registry_fingerprints(imported.file_fingerprints),
          .created_at = absl::Now(),
          .updated_at = absl::Now(),
      });

  rctx.mark_success();
  return grpc::Status::OK;
}

grpc::Status DiskArtifactService::import_artifact_from_path_stream(
    RpcContext& rctx,
    const v2::ImportArtifactFromPathRequest& req,
    grpc::ServerWriter<v2::ImportArtifactFromPathStreamEvent>& writer) {
  auto& span = rctx.span();
  const bool verify_checksums = req.verify_checksums();
  std::uint64_t seq = 0;
  bool stream_writable = true;

  const auto write_stream_event = [&](const materialization_disk_resolve::ImportProgressUpdate& update,
                                      const v2::ImportArtifactFromPathResponse* final_result = nullptr) {
    if (!stream_writable) {
      return;
    }
    v2::ImportArtifactFromPathStreamEvent event;
    event.set_seq(++seq);
    event.set_phase(to_proto_phase(update.phase));
    event.set_processed_bytes(update.processed_bytes);
    event.set_total_bytes(update.total_bytes);
    event.set_percent(to_progress_percent(update.processed_bytes, update.total_bytes, update.done));
    event.set_done(update.done);
    event.set_error(update.error);
    event.set_error_code(to_proto_error_code(update.error_code));
    if (!update.message.empty()) {
      event.set_message(update.message);
    }
    if (final_result != nullptr) {
      *event.mutable_result() = *final_result;
    }
    stream_writable = writer.Write(event);
  };

  const auto send_error_and_return = [&](const grpc::Status& status) {
    materialization_disk_resolve::ImportProgressUpdate update;
    update.phase = materialization_disk_resolve::ImportArtifactPhase::kError;
    update.done = true;
    update.error = true;
    update.error_code = grpc_status_to_import_error(status);
    update.message = status.error_message();
    write_stream_event(update);
    return status;
  };

  if (req.path().empty()) {
    materialization_disk_resolve::record_disk_import_outcome("invalid_argument");
    return send_error_and_return({StatusCode::INVALID_ARGUMENT, "path is required"});
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    materialization_disk_resolve::record_disk_import_outcome("unavailable");
    return send_error_and_return({StatusCode::UNAVAILABLE, "daemon is shutting down"});
  }

  const bool loopback_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  if (!loopback_peer) {
    materialization_disk_resolve::record_disk_import_outcome("permission_denied");
    materialization_disk_resolve::ImportProgressUpdate update;
    update.phase = materialization_disk_resolve::ImportArtifactPhase::kError;
    update.done = true;
    update.error = true;
    update.error_code = materialization_disk_resolve::ImportArtifactErrorCode::kPolicyDeniedNonLocalPeer;
    update.message = "ImportArtifactFromPath is local-only (loopback/UDS)";
    write_stream_event(update);
    return {StatusCode::PERMISSION_DENIED, update.message};
  }

  auto normalized_or = normalize_disk_import_path(req.path(), storage_path_);
  if (!normalized_or.ok()) {
    materialization_disk_resolve::record_disk_import_outcome("invalid_argument");
    return send_error_and_return(to_grpc_status(normalized_or.status()));
  }

  span->SetAttribute("tc.store.verify_checksums", verify_checksums);
  auto imported_or = import_artifact_from_path_cached(
      *normalized_or, verify_checksums, [&](const materialization_disk_resolve::ImportProgressUpdate& update) {
        if (update.done || update.error) {
          return;
        }
        write_stream_event(update);
      });
  if (!imported_or.ok()) {
    return send_error_and_return(to_import_grpc_status(imported_or.status()));
  }

  const auto& imported = *imported_or;
  v2::ImportArtifactFromPathResponse final_resp;
  fill_import_response(imported, final_resp);
  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.disk.path", imported.normalized_path.string());
    span->SetAttribute("tc.artifact.id", imported.artifact_id);
  }
  span->SetAttribute("tc.artifact.generation", static_cast<std::int64_t>(imported.generation));

  d_.source_registry.upsert_binding(
      imported.artifact_id,
      ArtifactSourceRegistry::Entry{
          .source_kind = ArtifactSourceRegistry::SourceKind::kLocalImport,
          .canonical_source_path = imported.normalized_path.string(),
          .source_disk_path = imported.normalized_path.string(),
          .descriptor_present = imported.descriptor_present,
          .index_multihash = imported.index_multihash,
          .data_multihash = imported.data_multihash,
          .generation = imported.generation,
          .file_fingerprints = to_registry_fingerprints(imported.file_fingerprints),
          .created_at = absl::Now(),
          .updated_at = absl::Now(),
      });

  materialization_disk_resolve::ImportProgressUpdate done;
  done.phase = materialization_disk_resolve::ImportArtifactPhase::kDone;
  done.done = true;
  write_stream_event(done, &final_resp);

  if (rctx.server_context().IsCancelled()) {
    return {StatusCode::CANCELLED, "client cancelled ImportArtifactFromPathStream"};
  }
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
