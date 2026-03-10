// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <memory>

#include "absl/time/time.h"
#include "core/common/capability_token.h"
#include "core/store/components/global_store_client.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/target_write_registry.h"
#include "daemon/state/worker_identity_store.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class TargetPublishService {
 public:
  struct Dep {
    LipManager& lip_manager;
    DeviceResolver& devices;
    WorkerIdentityStore& identity;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    common::CapabilityTokenManager* capability_tokens{nullptr};
    uint32_t max_concurrency{4};
  };

  explicit TargetPublishService(Dep d);

  static absl::Duration target_write_token_ttl();

  TargetWriteRegistry::Record remember_target_write(TargetWriteRegistry::Record record);

  grpc::Status publish_target_replica(
      RpcContext& rctx,
      const v2::PublishTargetReplicaRequest& req,
      v2::PublishTargetReplicaResponse& resp);

 private:
  Dep d_;
  common::CapabilityTokenManager* capability_tokens_{nullptr};
  TargetWriteRegistry target_write_registry_;
};

} // namespace tensorcast::daemon
