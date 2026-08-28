// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/common/capability_token.h"
#include "core/store/components/global_store_client.h"
#include "core/store/materialization/contracts/loading_spec.h"
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
  struct SourceBoundExecutionContext {
    bool loopback_peer{false};
    std::chrono::milliseconds request_budget{std::chrono::milliseconds(0)};
    std::function<absl::Status(const std::shared_ptr<BindingRegistry::Record>& record)> attach_ready_callback;
  };

  static SourceBoundExecutionContext source_bound_execution_context_from_server_context(
      const grpc::ServerContext& server_context);

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
    std::string daemon_id;
    std::string daemon_session_id;
    std::filesystem::path storage_path;
    std::function<absl::Status()> await_state_sync_barrier;
  };

  explicit OwnedBindingService(Dep d);

  grpc::Status create_binding(RpcContext& rctx, const v2::CreateBindingRequest& req, v2::CreateBindingResponse& resp);

  grpc::Status create_owned_binding(
      RpcContext& rctx,
      const v2::CreateOwnedBindingRequest& req,
      v2::CreateOwnedBindingResponse& resp);

  grpc::Status create_owned_binding_with_context(
      const SourceBoundExecutionContext& execution_context,
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

  grpc::Status freeze_binding_current_value(
      RpcContext& rctx,
      const v2::FreezeBindingCurrentValueRequest& req,
      v2::FreezeBindingCurrentValueResponse& resp);

  grpc::Status seal_binding(RpcContext& rctx, const v2::SealBindingRequest& req, v2::SealBindingResponse& resp);

  grpc::Status promote_binding_current_value(
      RpcContext& rctx,
      const v2::PromoteBindingCurrentValueRequest& req,
      v2::PromoteBindingCurrentValueResponse& resp);

  grpc::Status start_promote_binding_current_value(
      RpcContext& rctx,
      const v2::StartPromoteBindingCurrentValueRequest& req,
      v2::StartPromoteBindingCurrentValueResponse& resp);

  grpc::Status get_binding_promotion_status(
      RpcContext& rctx,
      const v2::GetBindingPromotionStatusRequest& req,
      v2::GetBindingPromotionStatusResponse& resp);

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

  struct PromotionJobRecord {
    v2::BindingPromotionStatus status;
  };

  grpc::Status promote_binding_current_value_impl(
      const v2::PromoteBindingCurrentValueRequest& req,
      v2::PromoteBindingCurrentValueResponse& resp);

  void run_async_promotion_job(std::string job_id, v2::PromoteBindingCurrentValueRequest req);

  grpc::Status create_owned_binding_impl(
      const SourceBoundExecutionContext& execution_context,
      RpcContext* rctx,
      const v2::CreateOwnedBindingRequest& req,
      v2::CreateOwnedBindingResponse& resp);

  void cancel_promotion_jobs_for_value(
      std::string_view binding_id,
      std::string_view binding_value_id,
      std::string_view reason);

  [[nodiscard]] std::string promotion_job_key(std::string_view binding_id, std::string_view binding_value_id) const;

  void fill_promotion_status_from_job(
      const std::shared_ptr<PromotionJobRecord>& job,
      v2::BindingPromotionStatus& status) const;

  Dep d_;
  std::shared_ptr<ContributionLeaseKeepaliveTracker> contribution_keepalive_tracker_;
  mutable absl::Mutex promotion_jobs_mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<PromotionJobRecord>> promotion_jobs_by_id_
      ABSL_GUARDED_BY(promotion_jobs_mu_);
  absl::flat_hash_map<std::string, std::string> promotion_job_ids_by_value_ ABSL_GUARDED_BY(promotion_jobs_mu_);
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

std::string mapped_execution_template_cache_key_for_testing(
    std::string_view plan_key,
    const std::optional<store::loading::DiskMetadata>& disk_metadata,
    v2::CollectivePolicy collective_policy,
    const store::StoreEngineOptions::MaterializationStrategyConfig& strategy_config,
    const store::loading::ExecutionTopologyContext& execution_topology,
    bool disk_source_available,
    bool include_runtime_group_id);

bool source_window_execution_template_uses_stable_runtime_group_for_testing(
    const store::StoreEngineOptions::MaterializationStrategyConfig& strategy_config,
    const std::optional<store::loading::DiskMetadata>& disk_metadata,
    v2::CollectivePolicy collective_policy,
    const store::loading::ExecutionTopologyContext& execution_topology,
    bool disk_source_available);

std::string source_window_prepared_realization_group_key_for_testing(
    std::string_view resolved_artifact_id,
    const tensorcast::common::v1::ArtifactSelection& selection,
    const v2::TargetLayout& target_layout,
    std::string_view target_index_json,
    std::string_view canonical_index_json,
    const std::optional<store::loading::DiskMetadata>& disk_metadata,
    v2::TransformPlacement placement,
    v2::CollectivePolicy collective_policy,
    const store::StoreEngineOptions::MaterializationStrategyConfig& strategy_config,
    const store::loading::ExecutionTopologyContext& execution_topology,
    bool disk_source_available);

std::string source_window_prepared_realization_member_key_for_testing(
    std::string_view group_key,
    std::string_view realization_plan_hash,
    const v2::TargetLayout& target_layout,
    const store::loading::ExecutionTopologyContext& execution_topology);

} // namespace tensorcast::daemon
