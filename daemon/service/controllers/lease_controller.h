// Copyright (c) 2025-2026, TensorCast Team.

// LeaseController: handles placement lease and retention handle RPCs.

#pragma once

#include <string>
#include <utility>

#include "core/common/capability_token.h"
#include "core/store/store_engine.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/lifecycle_kernel.h"
#include "daemon/state/placement_lease_tokens.h"
#include "daemon/state/retention_registry.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/shutdown_signal.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class LeaseController {
 public:
  struct Dep {
    store::StoreEngine& engine;
    SessionLifecycleManager& lifecycle;
    LifecycleKernel& lifecycle_kernel;
    PlacementLeaseTokens& placement_lease_tokens;
    common::CapabilityTokenManager* capability_tokens{nullptr};
    RetentionRegistry* retention_registry{nullptr};
    std::string daemon_id;
    ShutdownSignal& shutdown_signal;
  };

  explicit LeaseController(Dep d) : d_(std::move(d)) {}

  grpc::Status create_placement_lease(
      RpcContext& rctx,
      const v2::CreatePlacementLeaseRequest& req,
      v2::CreatePlacementLeaseResponse& resp);

  grpc::Status renew_placement_lease(
      RpcContext& rctx,
      const v2::RenewPlacementLeaseRequest& req,
      v2::RenewPlacementLeaseResponse& resp);

  grpc::Status release_placement_lease(
      RpcContext& rctx,
      const v2::ReleasePlacementLeaseRequest& req,
      v2::ReleasePlacementLeaseResponse& resp);

  grpc::Status acquire_retention_handle(
      RpcContext& rctx,
      const v2::AcquireRetentionHandleRequest& req,
      v2::AcquireRetentionHandleResponse& resp);

  grpc::Status renew_retention_handle(
      RpcContext& rctx,
      const v2::RenewRetentionHandleRequest& req,
      v2::RenewRetentionHandleResponse& resp);

  grpc::Status release_retention_handle(
      RpcContext& rctx,
      const v2::ReleaseRetentionHandleRequest& req,
      v2::ReleaseRetentionHandleResponse& resp);

 private:
  Dep d_;
};

} // namespace tensorcast::daemon
