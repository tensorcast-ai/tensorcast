// Copyright (c) 2025-2026, TensorCast Team.

// KeyMappingController: handles key-mapping RPCs backed by engine metadata.

#pragma once

#include "core/store/store_engine.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/shutdown_signal.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class KeyMappingController {
 public:
  struct Dep {
    store::StoreEngine& engine;
    ShutdownSignal& shutdown_signal;
  };

  explicit KeyMappingController(Dep d) : d_(d) {}

  grpc::Status publish_replica_key(
      RpcContext& rctx,
      const v2::PublishReplicaKeyRequest& req,
      v2::PublishReplicaKeyResponse& resp);

  grpc::Status resolve_key_mapping(
      RpcContext& rctx,
      const v2::ResolveKeyMappingRequest& req,
      v2::ResolveKeyMappingResponse& resp);

  grpc::Status swap_key_mapping(
      RpcContext& rctx,
      const v2::SwapKeyMappingRequest& req,
      v2::SwapKeyMappingResponse& resp);

 private:
  Dep d_;
};

} // namespace tensorcast::daemon
