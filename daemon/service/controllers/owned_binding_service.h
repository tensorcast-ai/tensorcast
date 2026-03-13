// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <filesystem>
#include <memory>

#include "core/common/capability_token.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/service/controllers/target_materialization_service.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/artifact_source_registry.h"
#include "daemon/state/binding_registry.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/handle_lease_registry.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/worker_identity_store.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class OwnedBindingService {
 public:
  struct Dep {
    store::StoreEngine& engine;
    DeviceResolver& devices;
    ArtifactSourceRegistry& disk_imports;
    BindingRegistry& bindings;
    ShutdownSignal& shutdown_signal;
    WorkerIdentityStore& identity;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    HandleLeaseRegistry* handle_leases{nullptr};
    common::CapabilityTokenManager* capability_tokens{nullptr};
    TargetMaterializationService* target_materialization_service{nullptr};
    std::filesystem::path storage_path;
  };

  explicit OwnedBindingService(Dep d);

  grpc::Status create_owned_binding(
      RpcContext& rctx,
      const v2::CreateOwnedBindingRequest& req,
      v2::CreateOwnedBindingResponse& resp);

  grpc::Status refill_owned_binding(
      RpcContext& rctx,
      const v2::RefillOwnedBindingRequest& req,
      v2::RefillOwnedBindingResponse& resp);

  grpc::Status close_owned_binding(
      RpcContext& rctx,
      const v2::CloseOwnedBindingRequest& req,
      v2::CloseOwnedBindingResponse& resp);

 private:
  Dep d_;
};

} // namespace tensorcast::daemon
