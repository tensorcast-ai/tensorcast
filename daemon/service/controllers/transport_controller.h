// Copyright (c) 2025, TensorCast Team.

// TransportController: handles Lock/Unlock transport chunk operations

#pragma once

#include "core/store/store_engine.h"
#include "daemon/lip_manager.h"
#include "daemon/rpc_context.h"
#include "daemon/transport_lock_manager.h"
#include "tensorcast/daemon/v1/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class TransportController {
 public:
  struct Dep {
    store::StoreEngine& engine;
    TransportLockManager& locks;
    LipManager& lip;
  };
  explicit TransportController(Dep d) : d_(d) {}

  grpc::Status lock(RpcContext& rctx, const v1::LockTransportChunksRequest& req, v1::LockTransportChunksResponse& resp);

  grpc::Status unlock(
      RpcContext& rctx,
      const v1::UnlockTransportChunksRequest& req,
      v1::UnlockTransportChunksResponse& resp);

 private:
  Dep d_;
};

} // namespace tensorcast::daemon
