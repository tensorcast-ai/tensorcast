// Copyright (c) 2025-2026, TensorCast Team.

// KeyMappingController: handles key-mapping RPCs backed by engine metadata.

#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <string_view>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/artifact_source_registry.h"
#include "daemon/state/shutdown_signal.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

class KeyMappingController {
 public:
  struct Dep {
    store::StoreEngine& engine;
    ShutdownSignal& shutdown_signal;
    ArtifactSourceRegistry* source_registry{nullptr};
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
  };

  explicit KeyMappingController(Dep d) : d_(d) {}

  grpc::Status publish_replica_key(
      RpcContext& rctx,
      const v2::PublishReplicaKeyRequest& req,
      v2::PublishReplicaKeyResponse& resp);

  grpc::Status resolve_key_mapping(
      RpcContext& rctx,
      const v2::ResolveKeyMappingRequest& req,
      v2::ResolveKeyMappingResponse& resp);

  grpc::Status swap_key_mapping(
      RpcContext& rctx,
      const v2::SwapKeyMappingRequest& req,
      v2::SwapKeyMappingResponse& resp);

 private:
  struct LocalCacheEntry {
    store::components::KeyMapping mapping;
    std::chrono::steady_clock::time_point expires_at{};
  };

  static constexpr uint32_t kLocalMutationCacheTtlSeconds = 30;

  [[nodiscard]] std::optional<store::components::KeyMapping> lookup_local_cache(std::string_view key) const;
  absl::Status maybe_register_local_import_artifact_metadata(
      const tensorcast::common::v1::ArtifactDescriptor& descriptor) const;
  void update_local_cache(
      std::string_view key,
      const store::components::KeyMapping& mapping,
      std::optional<uint32_t> ttl_override_seconds = std::nullopt);
  void erase_local_cache(std::string_view key);

  Dep d_;
  mutable absl::Mutex local_cache_mu_;
  mutable absl::flat_hash_map<std::string, LocalCacheEntry> local_cache_ ABSL_GUARDED_BY(local_cache_mu_);
};

} // namespace tensorcast::daemon
