// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/service/controllers/materialization_disk_resolve_utils.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/artifact_source_registry.h"
#include "daemon/state/shutdown_signal.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class DiskArtifactService {
 public:
  struct Dep {
    store::StoreEngine& engine;
    ArtifactSourceRegistry& source_registry;
    ShutdownSignal& shutdown_signal;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    std::filesystem::path storage_path;
  };

  explicit DiskArtifactService(Dep d);

  grpc::Status import_artifact_from_path(
      RpcContext& rctx,
      const v2::ImportArtifactFromPathRequest& req,
      v2::ImportArtifactFromPathResponse& resp);

  grpc::Status resolve_public_disk_source(
      RpcContext& rctx,
      const v2::ResolvePublicDiskSourceRequest& req,
      v2::ResolvePublicDiskSourceResponse& resp);

  grpc::Status import_artifact_from_path_stream(
      RpcContext& rctx,
      const v2::ImportArtifactFromPathRequest& req,
      grpc::ServerWriter<v2::ImportArtifactFromPathStreamEvent>& writer);

  grpc::Status get_artifact_index_by_id(
      RpcContext& rctx,
      const v2::GetArtifactIndexByIdRequest& req,
      v2::GetArtifactIndexByIdResponse& resp);

 private:
  struct ImportCacheEntry {
    materialization_disk_resolve::ImportArtifactFromPathResult imported;
    absl::Time cached_at{absl::Now()};
  };

  struct InflightImport {
    absl::Mutex mu;
    absl::CondVar cv;
    std::uint64_t progress_version ABSL_GUARDED_BY(mu){0};
    bool has_progress ABSL_GUARDED_BY(mu){false};
    materialization_disk_resolve::ImportProgressUpdate latest_progress ABSL_GUARDED_BY(mu);
    bool done ABSL_GUARDED_BY(mu){false};
    absl::StatusOr<materialization_disk_resolve::ImportArtifactFromPathResult> imported ABSL_GUARDED_BY(mu);
  };

  static absl::Duration import_cache_ttl_from_env();
  static size_t import_cache_max_entries_from_env();

  [[nodiscard]] std::string import_cache_key_for_path(
      const std::filesystem::path& normalized_path,
      bool verify_checksums) const;

  void prune_expired_cache_locked(absl::Time now) ABSL_EXCLUSIVE_LOCKS_REQUIRED(import_mu_);
  void enforce_cache_capacity_locked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(import_mu_);

  absl::StatusOr<materialization_disk_resolve::ImportArtifactFromPathResult> import_artifact_from_path_cached(
      const std::filesystem::path& normalized_path,
      bool verify_checksums,
      materialization_disk_resolve::ImportProgressCallback progress_cb = {});

  absl::Status ensure_artifact_metadata_registered(
      const materialization_disk_resolve::ImportArtifactFromPathResult& imported) const;

  Dep d_;
  std::filesystem::path storage_path_;
  const absl::Duration import_cache_ttl_;
  const size_t import_cache_max_entries_;

  mutable absl::Mutex import_mu_;
  absl::flat_hash_map<std::string, ImportCacheEntry> import_cache_ ABSL_GUARDED_BY(import_mu_);
  absl::flat_hash_map<std::string, std::shared_ptr<InflightImport>> inflight_imports_ ABSL_GUARDED_BY(import_mu_);
};

} // namespace tensorcast::daemon
