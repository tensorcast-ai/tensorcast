// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/service/body_backing_manager.h"
#include "daemon/service/byte_artifact_authority_service.h"
#include "daemon/service/byte_artifact_body_store.h"
#include "daemon/service/byte_artifact_route_resolver.h"
#include "daemon/service/controllers/external_target_access_service.h"
#include "daemon/service/payload_transport_broker.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/persistence_manager.h"
#include "daemon/state/worker_directory_cache.h"
#include "daemon/state/worker_identity_store.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class ByteArtifactController {
 public:
  struct Options {
    struct Routing {
      std::uint64_t shard_count{4096};
      std::uint64_t inline_payload_threshold_bytes{1ULL << 20};
      std::chrono::milliseconds route_staleness_budget{std::chrono::milliseconds(500)};
      std::chrono::milliseconds lease_ttl{std::chrono::seconds(5)};
      std::chrono::milliseconds keepalive_interval{std::chrono::seconds(1)};
      std::chrono::milliseconds worker_directory_staleness_budget{std::chrono::seconds(2)};
      std::uint64_t routing_epoch{1};
      bool shard_home_eligible{true};
    };

    Routing routing{};
    bool gateway_ingress_enabled{false};
  };

  struct Dep {
    ByteArtifactBodyStore& body_store;
    ByteArtifactRouteResolver& route_resolver;
    PayloadTransportBroker& payload_transport_broker;
    WorkerDirectoryCache& worker_directory_cache;
    ExternalTargetAccessService& external_target_access_service;
    WorkerIdentityStore& identity_store;
    store::StoreEngine& engine;
    PersistenceManager* persistence_manager{nullptr};
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
  };

  ByteArtifactController(Dep d, Options options);

  grpc::Status batch_exists(RpcContext& rctx, const v2::BatchExistsRequest& req, v2::BatchExistsResponse& resp);

  grpc::Status batch_get_into_region(
      RpcContext& rctx,
      const v2::BatchGetIntoRegionRequest& req,
      v2::BatchGetIntoRegionResponse& resp);

  grpc::Status batch_put_if_absent_from_region(
      RpcContext& rctx,
      const v2::BatchPutIfAbsentFromRegionRequest& req,
      v2::BatchPutIfAbsentFromRegionResponse& resp);

  grpc::Status batch_touch_ttl(RpcContext& rctx, const v2::BatchTouchTtlRequest& req, v2::BatchTouchTtlResponse& resp);

  grpc::Status home_batch_exists(
      RpcContext& rctx,
      const v2::HomeBatchExistsRequest& req,
      v2::HomeBatchExistsResponse& resp);

  grpc::Status home_batch_get(RpcContext& rctx, const v2::HomeBatchGetRequest& req, v2::HomeBatchGetResponse& resp);

  grpc::Status home_batch_put_if_absent(
      RpcContext& rctx,
      const v2::HomeBatchPutIfAbsentRequest& req,
      v2::HomeBatchPutIfAbsentResponse& resp);

  grpc::Status home_batch_touch_ttl(
      RpcContext& rctx,
      const v2::HomeBatchTouchTtlRequest& req,
      v2::HomeBatchTouchTtlResponse& resp);

 private:
  void reconcile_policy_visibility(
      const std::vector<std::string>& artifact_ids,
      const ByteArtifactAuthorityService::Context& context) const;

  [[nodiscard]] std::optional<PolicyVisibilityRef> resolve_policy_visibility_ref(
      const ByteArtifactBodyStore::AuthoritySnapshot& authority_snapshot) const;

  [[nodiscard]] absl::StatusOr<ResolvedSourceCapability> restore_backing_from_policy_visibility(
      ResolvedSourceCapability source_capability,
      const ByteArtifactAuthorityService::Context& context,
      std::string_view operation_id) const;

  Dep d_;
  ByteArtifactAuthorityService authority_service_;
  BodyBackingManager body_backing_manager_;
  Options options_;
};

} // namespace tensorcast::daemon
