// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/common/capability_token.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/service/controllers/external_target_access_service.h"
#include "daemon/service/controllers/target_publish_service.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/artifact_source_registry.h"
#include "daemon/state/binding_registry.h"
#include "daemon/state/daemon_options.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lifecycle_kernel.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/target_publication_registry.h"
#include "daemon/state/worker_identity_store.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"
#include "tensorcast/operation/v1/operation.pb.h"

namespace tensorcast::daemon {

class TargetMaterializationService {
 public:
  struct Dep {
    store::StoreEngine& engine;
    LipManager& lip_manager;
    BindingRegistry& bindings;
    DeviceResolver& devices;
    IpcRegionRegistry& regions;
    ArtifactSourceRegistry& disk_imports;
    SessionLifecycleManager& lifecycle;
    LifecycleKernel& lifecycle_kernel;
    ShutdownSignal& shutdown_signal;
    common::AsyncRuntime& async_runtime;
    WorkerIdentityStore& identity;
    ExternalTargetAccessService& external_target_access_service;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    common::CapabilityTokenManager* capability_tokens{nullptr};
    uint32_t max_concurrency{4};
    bool external_target_verification_enabled{false};
    DaemonOptions::ProgressiveReplication progressive_replication{};
    std::string daemon_id;
    std::string daemon_session_id;
    std::filesystem::path storage_path;
    std::function<absl::Status()> await_state_sync_barrier;
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

  grpc::Status start_publish_target_replica(
      RpcContext& rctx,
      const v2::PublishTargetReplicaRequest& req,
      v2::StartPublishTargetReplicaResponse& resp);

  [[nodiscard]] absl::StatusOr<TargetPublishService::TargetPublicationFrontDoorContext>
  inspect_target_publication_context_for_testing(const v2::PublishTargetReplicaRequest& req, absl::Time now);

  [[nodiscard]] absl::StatusOr<RoutedAuthorityRequest> build_target_publication_workflow_routed_request_for_testing(
      const v2::PublishTargetReplicaRequest& req,
      absl::Time now) const;

  [[nodiscard]] absl::StatusOr<RoutedAuthorityRequest>
  build_target_publication_workflow_continuation_request_for_testing(
      const RoutedAuthorityRequest& routed_request,
      const OwnerStageReply& workflow_gate_reply) const;

  [[nodiscard]] absl::StatusOr<std::optional<OwnerStageReply>> maybe_route_authority_stage(
      const RoutedAuthorityRequest& routed_request,
      absl::Time now);

  [[nodiscard]] absl::StatusOr<TargetPublicationRegistry::Record> remember_target_publication(
      TargetPublicationRegistry::Record record);

  [[nodiscard]] absl::Status terminalize_target_publication(
      std::string_view publication_id,
      std::string_view reason,
      bool release_published_lifecycle_lease);

  [[nodiscard]] absl::StatusOr<TargetPublicationRegistry::Record> insert_target_publication_for_testing(
      TargetPublicationRegistry::Record record);

  [[nodiscard]] absl::Status admit_public_operation(
      const tensorcast::operation::v1::OperationRef& operation_ref,
      absl::Time now) const;

 private:
  Dep d_;
  std::filesystem::path storage_path_;
  common::CapabilityTokenManager* capability_tokens_{nullptr};
  TargetPublishService target_publish_service_;
};

} // namespace tensorcast::daemon
