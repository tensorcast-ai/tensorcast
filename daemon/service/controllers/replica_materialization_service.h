// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <filesystem>
#include <memory>

#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/artifact_source_registry.h"
#include "daemon/state/daemon_options.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/handle_lease_registry.h"
#include "daemon/state/lip_bridge.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/sessions_service.h"
#include "daemon/state/shutdown_signal.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class ReplicaMaterializationService {
 public:
  struct Dep {
    store::StoreEngine& engine;
    RefTracker& refs;
    SessionsService& sessions;
    LipBridge& lip;
    DeviceResolver& devices;
    ArtifactSourceRegistry& disk_imports;
    ShutdownSignal& shutdown_signal;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    SessionLifecycleManager* lifecycle{nullptr};
    HandleLeaseRegistry* handle_leases{nullptr};
    bool cpu_shared_memory_enabled{false};
    DaemonOptions::PostSealPolicy post_seal_policy{};
    std::filesystem::path storage_path;
  };

  explicit ReplicaMaterializationService(Dep d);

  grpc::Status materialize_replica(
      RpcContext& rctx,
      const v2::MaterializeReplicaRequest& req,
      v2::MaterializeReplicaResponse& resp);

 private:
  Dep d_;
  std::filesystem::path storage_path_;
};

} // namespace tensorcast::daemon
