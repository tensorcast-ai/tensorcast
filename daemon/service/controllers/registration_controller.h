// Copyright (c) 2025, TensorCast Team.

// RegistrationController: handles begin/feed/commit/abort/keep_alive/revoke

#pragma once

#include <vector>

#include "core/store/store_engine.h"
#include "daemon/lip_manager.h"
#include "daemon/registration_manager.h"
#include "daemon/rpc_context.h"
#include "grpcpp/grpcpp.h"
#include "tensorcast/daemon/v1/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class RegistrationController {
 public:
  struct Dep {
    store::StoreEngine& engine;
    RegistrationManager& reg;
    LipManager& lip;
  };
  explicit RegistrationController(Dep d) : d_(d) {}

  grpc::Status begin(
      RpcContext& rctx,
      const v1::BeginRegisterArtifactRequest& req,
      v1::BeginRegisterArtifactResponse& resp);

  grpc::Status feed_stream(
      RpcContext& rctx,
      ::grpc::ServerReader<v1::FeedRegisterArtifactStreamRequest>& reader,
      v1::FeedRegisterArtifactStreamResponse& resp);

  grpc::Status feed_vector(const std::vector<v1::FeedRegisterArtifactStreamRequest>& reqs);

  grpc::Status keep_alive(
      RpcContext& rctx,
      const v1::KeepAliveRegisterArtifactRequest& req,
      v1::KeepAliveRegisterArtifactResponse& resp);

  grpc::Status commit(
      RpcContext& rctx,
      const v1::CommitRegisteredArtifactRequest& req,
      v1::CommitRegisteredArtifactResponse& resp);

  grpc::Status abort(
      RpcContext& rctx,
      const v1::AbortRegisteredArtifactRequest& req,
      v1::AbortRegisteredArtifactResponse& resp);

  grpc::Status revoke(
      RpcContext& rctx,
      const v1::RevokeRegisteredArtifactRequest& req,
      v1::RevokeRegisteredArtifactResponse& resp);

 private:
  Dep d_;
};

} // namespace tensorcast::daemon
