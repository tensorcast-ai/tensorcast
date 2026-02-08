// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <filesystem>

#include "core/store/store_engine.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/local_disk_import_catalog.h"
#include "daemon/state/shutdown_signal.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class DiskArtifactService {
 public:
  struct Dep {
    store::StoreEngine& engine;
    LocalDiskImportCatalog& disk_imports;
    ShutdownSignal& shutdown_signal;
    std::filesystem::path storage_path;
  };

  explicit DiskArtifactService(Dep d);

  grpc::Status resolve_artifact_from_disk(
      RpcContext& rctx,
      const v2::ResolveArtifactFromDiskRequest& req,
      v2::ResolveArtifactFromDiskResponse& resp);

  grpc::Status get_artifact_index_by_id(
      RpcContext& rctx,
      const v2::GetArtifactIndexByIdRequest& req,
      v2::GetArtifactIndexByIdResponse& resp);

 private:
  Dep d_;
  std::filesystem::path storage_path_;
};

} // namespace tensorcast::daemon
