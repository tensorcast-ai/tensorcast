// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <filesystem>
#include <memory>

#include "core/common/capability_token.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/service/controllers/target_publish_service.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/artifact_source_registry.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/target_write_registry.h"
#include "daemon/state/worker_identity_store.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class TargetMaterializationService {
 public:
  struct Dep {
    store::StoreEngine& engine;
    LipManager& lip_manager;
    DeviceResolver& devices;
    IpcRegionRegistry& regions;
    ArtifactSourceRegistry& disk_imports;
    ShutdownSignal& shutdown_signal;
    WorkerIdentityStore& identity;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    common::CapabilityTokenManager* capability_tokens{nullptr};
    uint32_t max_concurrency{4};
    bool external_target_verification_enabled{false};
    std::filesystem::path storage_path;
  };

  explicit TargetMaterializationService(Dep d);

  grpc::Status materialize_into_target(
      RpcContext& rctx,
      const v2::MaterializeIntoTargetRequest& req,
      v2::MaterializeIntoTargetResponse& resp);

  grpc::Status materialize_into_mapped_target(
      RpcContext& rctx,
      const v2::MaterializeIntoMappedTargetRequest& req,
      v2::MaterializeIntoTargetResponse& resp);

  grpc::Status publish_target_replica(
      RpcContext& rctx,
      const v2::PublishTargetReplicaRequest& req,
      v2::PublishTargetReplicaResponse& resp);

  TargetWriteRegistry::Record remember_target_write(TargetWriteRegistry::Record record);

  TargetWriteRegistry::Record insert_target_write_for_testing(TargetWriteRegistry::Record record);

 private:
  Dep d_;
  std::filesystem::path storage_path_;
  common::CapabilityTokenManager* capability_tokens_{nullptr};
  TargetPublishService target_publish_service_;
};

} // namespace tensorcast::daemon
