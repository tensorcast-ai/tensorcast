// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <atomic>
#include <filesystem>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "core/common/async_runtime.h"
#include "core/common/capability_token.h"
#include "core/store/components/global_store_client.h"
#include "core/store/runtime/ingestion/materialization_strategy_types.h"
#include "core/store/store_engine.h"
#include "daemon/service/controllers/target_materialization_service.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/artifact_source_registry.h"
#include "daemon/state/binding_registry.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/handle_lease_registry.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/registration_manager.h"
#include "daemon/state/session_lifecycle.h"
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
    common::AsyncRuntime& async_runtime;
    WorkerIdentityStore& identity;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    SessionLifecycleManager* lifecycle{nullptr};
    HandleLeaseRegistry* handle_leases{nullptr};
    RegistrationManager* registration_manager{nullptr};
    LipManager* lip_manager{nullptr};
    RefTracker* refs{nullptr};
    IpcRegionRegistry* regions{nullptr};
    uint32_t max_concurrency{4};
    common::CapabilityTokenManager* capability_tokens{nullptr};
    TargetMaterializationService* target_materialization_service{nullptr};
    std::filesystem::path storage_path;
  };

  explicit OwnedBindingService(Dep d);

  grpc::Status create_binding(RpcContext& rctx, const v2::CreateBindingRequest& req, v2::CreateBindingResponse& resp);

  grpc::Status create_owned_binding(
      RpcContext& rctx,
      const v2::CreateOwnedBindingRequest& req,
      v2::CreateOwnedBindingResponse& resp);

  grpc::Status commit_binding_artifact(
      RpcContext& rctx,
      const v2::CommitBindingArtifactRequest& req,
      v2::CommitBindingArtifactResponse& resp);

  grpc::Status begin_binding_update(
      RpcContext& rctx,
      const v2::BeginBindingUpdateRequest& req,
      v2::BeginBindingUpdateResponse& resp);

  grpc::Status submit_binding_contribution(
      RpcContext& rctx,
      const v2::SubmitBindingContributionRequest& req,
      v2::SubmitBindingContributionResponse& resp);

  grpc::Status seal_binding(RpcContext& rctx, const v2::SealBindingRequest& req, v2::SealBindingResponse& resp);

  grpc::Status promote_binding_current_value(
      RpcContext& rctx,
      const v2::PromoteBindingCurrentValueRequest& req,
      v2::PromoteBindingCurrentValueResponse& resp);

  grpc::Status refill_owned_binding(
      RpcContext& rctx,
      const v2::RefillOwnedBindingRequest& req,
      v2::RefillOwnedBindingResponse& resp);

  grpc::Status close_owned_binding(
      RpcContext& rctx,
      const v2::CloseOwnedBindingRequest& req,
      v2::CloseOwnedBindingResponse& resp);

 private:
  struct ContributionLeaseKeepaliveTracker {
    absl::Mutex mu;
    absl::flat_hash_map<std::string, std::shared_ptr<std::atomic<bool>>> stop_flags ABSL_GUARDED_BY(mu);
  };

  Dep d_;
  std::shared_ptr<ContributionLeaseKeepaliveTracker> contribution_keepalive_tracker_;
};

grpc::Status evaluate_strict_collective_preflight_for_testing(
    RpcContext* rctx,
    const store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary* plan_summary,
    v2::CollectivePolicy collective_policy);

store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary summarize_source_bound_plan_for_testing(
    const store::runtime::ingestion::strategy::ResolvedMaterializationPlan& resolved_plan,
    const std::optional<store::runtime::ingestion::strategy::SourceBoundLoweringArtifacts>& lowering_artifacts,
    const store::StoreEngineOptions::MaterializationStrategyConfig& strategy_config,
    const store::loading::ExecutionTopologyContext& execution_topology,
    v2::CollectivePolicy collective_policy,
    bool disk_source_available);

} // namespace tensorcast::daemon
