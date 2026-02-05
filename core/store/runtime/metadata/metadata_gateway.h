// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/store/components/global_store_client.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/runtime/context/runtime_context_events.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/runtime/metadata/metadata_types.h"
#include "core/store/runtime/metadata/registration_backend.h"
#include "core/store/runtime/replica/replica_promotion_manager.h"
#include "core/store/runtime/replica/replica_runtime.h"
#include "gsl/pointers"

namespace tensorcast::store::runtime::metadata {

class MetadataGateway {
 public:
  struct Config {
    RuntimeContext* runtime_context;
    ReplicaRuntime* replica_runtime;
    runtime::ReplicaPromotionManager* promotion_manager;
    size_t artifact_chunk_bytes;
    std::chrono::milliseconds pinned_memory_timeout;
    ReplicaFactory replica_factory;
  };

  explicit MetadataGateway(Config config);
  ~MetadataGateway();

  MetadataGateway(const MetadataGateway&) = delete;
  MetadataGateway& operator=(const MetadataGateway&) = delete;

  [[nodiscard]] bool is_connected() const;
  void set_client_override(std::shared_ptr<components::IGlobalStoreClient> client);
  void refresh_override_endpoint();

  void handle_ingestion_result(const IngestionResultEvent& event);

  absl::Status register_replica(
      const loading::ReplicaKey& key,
      std::string_view artifact_id_override = {},
      std::string_view publish_context_id = {});
  absl::Status unregister_replica(std::string_view artifact_id, int device_id);

  absl::StatusOr<components::KeyMapping> resolve_key_mapping(std::string_view key) const;
  absl::StatusOr<std::string> get_canonical_index(std::string_view artifact_id) const;
  absl::StatusOr<components::ViewMetadata> get_view_metadata(std::string_view artifact_id, std::string_view view_id)
      const;
  absl::Status upsert_key_mapping(std::string_view key, std::string_view artifact_id, absl::Duration ttl);
  absl::StatusOr<components::KeyMappingSwapResult> swap_key_mapping(
      std::string_view key,
      std::string_view new_artifact_id,
      std::optional<std::string_view> expected_artifact_id,
      std::optional<uint64_t> expected_generation);
  absl::Status revoke_key_mapping(std::string_view key);

  // Registration workflow
  absl::StatusOr<RegistrationBeginResult> begin_registration(const ArtifactRegistration& reg);
  absl::StatusOr<RegistrationCommitResult> commit_registration(std::string_view registration_id);
  absl::Status abort_registration(std::string_view registration_id);
  absl::Status keep_alive_registration(std::string_view registration_id, uint32_t ttl_ms);
  absl::StatusOr<uint64_t> get_registration_gpu_ptr(std::string_view registration_id) const;
  absl::Status ingest_view_chunk(
      std::string_view registration_id,
      uint64_t view_offset,
      absl::Span<const std::byte> data);
  [[nodiscard]] absl::StatusOr<uint64_t> get_view_ingested_bytes(std::string_view registration_id) const;

 private:
  void publish_registration_event(
      RuntimeEventType type,
      std::string_view registration_id,
      const RegistrationCommitResult* result,
      const absl::Status& status) const;

  std::string worker_id() const;
  absl::StatusOr<std::shared_ptr<components::IGlobalStoreClient>> get_connected_client() const;
  RegistrationResources make_registration_resources() const;
  ReplicaFactory make_default_replica_factory() const;
  bool should_skip_publish_for_context(std::string_view publish_context_id, const loading::ReplicaKey& key) const;
  void record_publish_context_result(
      std::string_view publish_context_id,
      const loading::ReplicaKey& key,
      const absl::Status& status);
  void cleanup_publish_contexts_locked(absl::Time now) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(publish_context_mu_);

  struct PublishContextRecord {
    loading::ReplicaKey key;
    absl::Status status;
    absl::Time updated_at;
  };

  gsl::not_null<RuntimeContext*> runtime_context_;
  gsl::not_null<ReplicaRuntime*> replica_runtime_;
  runtime::ReplicaPromotionManager* promotion_manager_{nullptr};
  RuntimeContextEvents::Publisher event_publisher_;
  std::shared_ptr<components::IGlobalStoreClient> override_client_;
  std::unique_ptr<RegistrationPublisher> registration_publisher_;
  std::unique_ptr<RegistrationBackend> registration_backend_;
  size_t artifact_chunk_bytes_{0};
  std::chrono::milliseconds pinned_memory_timeout_{0};
  ReplicaFactory replica_factory_;
  mutable absl::Mutex publish_context_mu_;
  mutable absl::flat_hash_map<std::string, PublishContextRecord> publish_contexts_ ABSL_GUARDED_BY(publish_context_mu_);
  std::unique_ptr<RuntimeContextEvents::Subscription> ingestion_event_subscription_;
};

} // namespace tensorcast::store::runtime::metadata
