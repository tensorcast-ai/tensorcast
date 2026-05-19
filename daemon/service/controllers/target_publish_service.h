// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string_view>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/common/capability_token.h"
#include "core/store/components/global_store_client.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/binding_registry.h"
#include "daemon/state/daemon_options.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/lifecycle_kernel.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/routed_authority_protocol.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/target_publication_registry.h"
#include "daemon/state/worker_identity_store.h"
#include "tensorcast/common/v1/capability_token.pb.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"
#include "tensorcast/operation/v1/operation.pb.h"

namespace tensorcast::daemon {

class TargetPublishService {
 public:
  struct Dep {
    LipManager& lip_manager;
    BindingRegistry& bindings;
    DeviceResolver& devices;
    WorkerIdentityStore& identity;
    SessionLifecycleManager& lifecycle;
    LifecycleKernel& lifecycle_kernel;
    common::AsyncRuntime& async_runtime;
    ShutdownSignal& shutdown_signal;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    common::CapabilityTokenManager* capability_tokens{nullptr};
    uint32_t max_concurrency{4};
    std::function<absl::Status()> await_state_sync_barrier;
    DaemonOptions::ProgressiveReplication progressive_replication{};
    std::string daemon_id;
    std::string daemon_session_id;
  };

  explicit TargetPublishService(Dep d);

  static absl::Duration binding_current_value_publication_token_ttl();

  static constexpr std::string_view public_operation_kind() {
    return "publish_target_replica";
  }

  struct TargetPublicationFrontDoorContext {
    TargetPublicationRegistry::Record record;
    tensorcast::common::v1::BindingCurrentValuePublicationScope scope;
    tensorcast::common::v1::ByteSpaceRef normalized_byte_space;
    FrontDoorCredentialContext front_door_context;
  };

  [[nodiscard]] absl::StatusOr<TargetPublicationRegistry::Record> remember_target_publication(
      TargetPublicationRegistry::Record record);

  [[nodiscard]] absl::Status terminalize_publication(
      std::string_view publication_id,
      std::string_view reason,
      bool release_published_lifecycle_lease);

  [[nodiscard]] absl::StatusOr<TargetPublicationFrontDoorContext> inspect_target_publication_context(
      const v2::PublishTargetReplicaRequest& req,
      absl::Time now) const;

  [[nodiscard]] absl::StatusOr<RoutedAuthorityRequest> build_target_publication_workflow_routed_request(
      const v2::PublishTargetReplicaRequest& req,
      absl::Time now) const;

  [[nodiscard]] absl::StatusOr<RoutedAuthorityRequest> build_target_publication_workflow_continuation_request(
      const RoutedAuthorityRequest& routed_request,
      const OwnerStageReply& workflow_gate_reply) const;

  [[nodiscard]] absl::StatusOr<std::optional<OwnerStageReply>> maybe_route_authority_stage(
      const RoutedAuthorityRequest& routed_request,
      absl::Time now);

  grpc::Status start_publish_target_replica(
      RpcContext& rctx,
      const v2::PublishTargetReplicaRequest& req,
      v2::StartPublishTargetReplicaResponse& resp);

  grpc::Status publish_target_replica(
      RpcContext& rctx,
      const v2::PublishTargetReplicaRequest& req,
      v2::PublishTargetReplicaResponse& resp);

  [[nodiscard]] absl::Status admit_public_operation(
      const tensorcast::operation::v1::OperationRef& operation_ref,
      absl::Time now) const;

 private:
  struct PublishOperationTracker {
    absl::Mutex mu;
    absl::flat_hash_set<std::string> active_operations ABSL_GUARDED_BY(mu);
  };

  Dep d_;
  common::CapabilityTokenManager* capability_tokens_{nullptr};
  std::shared_ptr<TargetPublicationRegistry> target_publication_registry_;
  std::shared_ptr<PublishOperationTracker> publish_operation_tracker_;
};

} // namespace tensorcast::daemon
