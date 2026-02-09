// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include "core/store/store_engine.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/sessions_service.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class ReplicaLifecycleService {
 public:
  struct Dep {
    store::StoreEngine& engine;
    RefTracker& refs;
    SessionsService& sessions;
    SessionLifecycleManager* lifecycle{nullptr};
    DeviceResolver& devices;
  };

  explicit ReplicaLifecycleService(Dep d);

  grpc::Status confirm(RpcContext& rctx, const v2::ConfirmReplicaRequest& req, v2::ConfirmReplicaResponse& resp) const;

  grpc::Status unload(RpcContext& rctx, const v2::UnloadReplicaRequest& req, v2::UnloadReplicaResponse& resp);

  grpc::Status wait_verification(
      RpcContext& rctx,
      const v2::WaitReplicaVerificationRequest& req,
      v2::WaitReplicaVerificationResponse& resp);

 private:
  Dep d_;
};

} // namespace tensorcast::daemon
