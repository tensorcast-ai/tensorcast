// Copyright (c) 2025-2026, TensorCast Team.

// TransportController: handles Lock/Unlock transport chunk operations

#pragma once

#include "core/store/store_engine.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/derived_view_export_manager.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/transport_lock_manager.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class TransportController {
 public:
  struct Dep {
    store::StoreEngine& engine;
    TransportLockManager& locks;
    LipManager& lip;
    DerivedViewExportManager* derived_view_exports{nullptr};
  };

  explicit TransportController(Dep d) : d_(d) {}

  grpc::Status lock(RpcContext& rctx, const v2::LockTransportChunksRequest& req, v2::LockTransportChunksResponse& resp);

  grpc::Status unlock(
      RpcContext& rctx,
      const v2::UnlockTransportChunksRequest& req,
      v2::UnlockTransportChunksResponse& resp);

  grpc::Status begin_replica_fetch(
      RpcContext& rctx,
      const v2::BeginReplicaFetchRequest& req,
      v2::BeginReplicaFetchResponse& resp);

  grpc::Status end_replica_fetch(
      RpcContext& rctx,
      const v2::EndReplicaFetchRequest& req,
      v2::EndReplicaFetchResponse& resp);

 private:
  Dep d_;
};

} // namespace tensorcast::daemon
