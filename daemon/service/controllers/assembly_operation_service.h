// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "core/common/async_runtime.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/daemon_options.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/worker_identity_store.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class AssemblyOperationService {
 public:
  struct Dep {
    store::StoreEngine& engine;
    DeviceResolver& devices;
    ShutdownSignal& shutdown_signal;
    common::AsyncRuntime& async_runtime;
    WorkerIdentityStore& identity;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    DaemonOptions::PostSealPolicy post_seal_policy{};
  };

  explicit AssemblyOperationService(Dep d);

  grpc::Status seal_assembly(RpcContext& rctx, const v2::SealAssemblyRequest& req, v2::SealAssemblyResponse& resp);

  grpc::Status start_seal_assembly(
      RpcContext& rctx,
      const v2::StartSealAssemblyRequest& req,
      v2::StartSealAssemblyResponse& resp);

  grpc::Status get_operation(
      RpcContext& rctx,
      const tensorcast::operation::v1::GetOperationRequest& req,
      tensorcast::operation::v1::GetOperationResponse& resp);

  grpc::Status wait_operation(RpcContext& rctx, const v2::WaitOperationRequest& req, v2::WaitOperationResponse& resp);

 private:
  struct SealOperationTracker {
    absl::Mutex mu;
    absl::flat_hash_set<std::string> active_operations ABSL_GUARDED_BY(mu);
  };

  Dep d_;
  std::shared_ptr<SealOperationTracker> seal_operation_tracker_;
};

} // namespace tensorcast::daemon
