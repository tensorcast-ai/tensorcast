// Copyright (c) 2025-2026, TensorCast Team.

// RegistrationController: handles begin/feed/commit/abort/keep_alive/revoke

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <gsl/pointers>
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/registration_manager.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/worker_identity_store.h"
#include "grpcpp/grpcpp.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class RegistrationController {
 public:
  struct Dep {
    store::StoreEngine& engine;
    RegistrationManager& reg;
    LipManager& lip;
    RefTracker& refs;
    gsl::not_null<WorkerIdentityStore*> identity;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    gsl::not_null<SessionLifecycleManager*> lifecycle;
    IpcRegionRegistry& regions;
    uint32_t max_concurrency{4};
  };

  explicit RegistrationController(Dep d) : d_(std::move(d)) {}

  grpc::Status begin(
      RpcContext& rctx,
      const v2::BeginRegisterArtifactRequest& req,
      v2::BeginRegisterArtifactResponse& resp);

  grpc::Status feed_stream(
      RpcContext& rctx,
      ::grpc::ServerReader<v2::FeedRegisterArtifactStreamRequest>& reader,
      v2::FeedRegisterArtifactStreamResponse& resp);

  grpc::Status feed_vector(const std::vector<v2::FeedRegisterArtifactStreamRequest>& reqs);

  grpc::Status keep_alive(
      RpcContext& rctx,
      const v2::KeepAliveRegisterArtifactRequest& req,
      v2::KeepAliveRegisterArtifactResponse& resp);

  grpc::Status commit(
      RpcContext& rctx,
      const v2::CommitRegisteredArtifactRequest& req,
      v2::CommitRegisteredArtifactResponse& resp);

  grpc::Status abort(
      RpcContext& rctx,
      const v2::AbortRegisteredArtifactRequest& req,
      v2::AbortRegisteredArtifactResponse& resp);

  grpc::Status revoke(
      RpcContext& rctx,
      const v2::RevokeRegisteredArtifactRequest& req,
      v2::RevokeRegisteredArtifactResponse& resp);

 private:
  Dep d_;
};

} // namespace tensorcast::daemon
