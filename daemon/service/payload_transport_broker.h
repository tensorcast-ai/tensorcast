// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/capability_token.h"
#include "core/store/communication_types.h"
#include "core/store/components/communication_manager.h"
#include "core/store/materialization/dataplane/contracts/loader.h"
#include "daemon/service/body_backing_types.h"
#include "daemon/service/byte_artifact_body_handle.h"
#include "daemon/state/daemon_options.h"
#include "daemon/state/lifecycle_kernel.h"
#include "daemon/state/session_lifecycle.h"
#include "grpcpp/security/credentials.h"
#include "tensorcast/common/v1/capability_token.pb.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

struct AuthorityRef;
struct PortableParsedCredential;
struct OwnerStageReply;
struct RoutedAuthorityRequest;

class WorkerDirectoryCache;

class PayloadTransportBroker {
 public:
  struct Options {
    absl::Duration ttl{absl::Minutes(5)};
    std::uint64_t max_chunk_bytes{1ULL << 20};
    absl::Duration fetch_deadline{absl::Seconds(5)};
    absl::Duration cleanup_interval{absl::Minutes(1)};
    std::uint64_t max_batch_payload_bytes{0};
    std::uint32_t max_batch_items{0};
    std::uint64_t max_batch_stage_bytes_per_peer{128ULL << 20};
    std::uint32_t batch_transport_protocol_version{2};
    bool communicator_source_enabled{true};
    bool host_memory_export_enabled{true};
    bool segmented_communicator_export_enabled{true};
    absl::Duration minimum_batch_transport_ttl{absl::Milliseconds(250)};
    absl::Duration transport_release_guard{absl::Seconds(1)};
    std::shared_ptr<store::components::CommunicationManager> comm_manager;
    std::function<std::string()> local_cpu_endpoint_id_provider;
    std::shared_ptr<grpc::ChannelCredentials> inter_daemon_channel_credentials;
    DaemonOptions::InterDaemonGrpcSecurity inter_daemon_grpc_security{};
  };

  struct RefMetadata {
    std::string issuer_daemon_id;
    std::string payload_id;
    std::string artifact_id;
    std::string digest_alg;
    std::string digest_hex;
    std::uint64_t payload_size{0};
    tensorcast::common::v1::PayloadRefDirection direction{tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED};
    std::string operation_id;
    absl::Time expires_at{absl::InfinitePast()};
  };

  struct ResolvedPayload {
    RefMetadata metadata;
    std::string payload;
  };

  struct BatchRefMetadata {
    std::string issuer_daemon_id;
    std::string transport_id;
    std::string manifest_digest_hex;
    std::string consumer_daemon_id;
    std::uint64_t payload_size{0};
    tensorcast::common::v1::PayloadRefDirection direction{tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED};
    std::string operation_id;
    absl::Time expires_at{absl::InfinitePast()};
  };

  struct ResolvedBatchPayload {
    BatchRefMetadata metadata;
    std::shared_ptr<const std::string> payload;
    bool remote{false};
  };

  struct BatchCommunicatorExport {
    BatchRefMetadata metadata;
    std::string batch_payload_ref;
    store::ExportRegistration export_registration;
    std::string registration_ownership;
    std::string mr_ownership;
    bool broker_owned_register{false};
  };

  struct BatchCommunicatorSourceSegment {
    BodyExportView export_view;
  };

  struct BatchCommunicatorRegionSourceSegment {
    const void* data{nullptr};
    std::uint64_t size_bytes{0};
    std::optional<store::StableLocalBackingRef> stable_backing;
    std::shared_ptr<void> stable_backing_keepalive;
    std::shared_ptr<void> keepalive;
  };

  struct BatchPayloadSource {
    BatchRefMetadata metadata;
    std::shared_ptr<store::loader::SeekableSource> source;
    bool remote{false};
  };

  struct PayloadLoader {
    RefMetadata metadata;
    std::unique_ptr<store::IArtifactLoader> loader;
    bool remote{false};
  };

  struct PayloadRefFrontDoorContext {
    RefMetadata metadata;
    FrontDoorCredentialContext front_door_context;
  };

  PayloadTransportBroker(
      std::string daemon_id,
      common::CapabilityTokenManager* capability_tokens,
      SessionLifecycleManager* lifecycle_manager,
      LifecycleKernel* lifecycle_kernel,
      Options options);

  [[nodiscard]] absl::StatusOr<std::string> issue_payload_ref(
      std::string_view artifact_id,
      std::string payload,
      tensorcast::common::v1::PayloadRefDirection direction,
      std::string_view operation_id = "",
      absl::Time capability_expires_at = absl::InfiniteFuture());

  [[nodiscard]] absl::StatusOr<std::string> issue_payload_ref(
      std::string_view artifact_id,
      std::shared_ptr<const std::string> payload,
      const BodyDescriptor& descriptor,
      tensorcast::common::v1::PayloadRefDirection direction,
      std::string_view operation_id = "",
      absl::Time capability_expires_at = absl::InfiniteFuture());

  [[nodiscard]] absl::StatusOr<std::string> issue_payload_ref(
      std::string_view artifact_id,
      std::shared_ptr<const std::string> payload,
      tensorcast::common::v1::PayloadRefDirection direction,
      std::string_view operation_id = "",
      absl::Time capability_expires_at = absl::InfiniteFuture());

  [[nodiscard]] absl::StatusOr<std::string> issue_payload_ref(
      std::string_view artifact_id,
      const BodyHandle& body_handle,
      const BodyDescriptor& descriptor,
      std::optional<store::runtime::ingestion::BackingIdentity> backing_identity,
      std::uint64_t backing_instance_generation,
      tensorcast::common::v1::PayloadRefDirection direction,
      std::string_view operation_id = "",
      absl::Time capability_expires_at = absl::InfiniteFuture());

  [[nodiscard]] absl::StatusOr<std::string> issue_batch_payload_ref(
      const v2::BatchPayloadManifest& manifest,
      std::shared_ptr<const std::string> payload,
      tensorcast::common::v1::PayloadRefDirection direction,
      std::string_view operation_id = "",
      absl::Time capability_expires_at = absl::InfiniteFuture(),
      std::string_view consumer_daemon_id = "");

  [[nodiscard]] absl::StatusOr<BatchCommunicatorExport> issue_batch_payload_communicator_export(
      const v2::BatchPayloadManifest& manifest,
      std::shared_ptr<const std::string> payload,
      tensorcast::common::v1::PayloadRefDirection direction,
      std::string_view operation_id = "",
      absl::Time capability_expires_at = absl::InfiniteFuture(),
      std::string_view consumer_daemon_id = "");

  [[nodiscard]] absl::StatusOr<BatchCommunicatorExport> issue_batch_payload_communicator_export(
      const v2::BatchPayloadManifest& manifest,
      absl::Span<const BatchCommunicatorSourceSegment> source_segments,
      tensorcast::common::v1::PayloadRefDirection direction,
      std::string_view operation_id = "",
      absl::Time capability_expires_at = absl::InfiniteFuture(),
      std::string_view consumer_daemon_id = "");

  [[nodiscard]] absl::StatusOr<BatchCommunicatorExport> issue_batch_payload_communicator_export(
      const v2::BatchPayloadManifest& manifest,
      absl::Span<const BatchCommunicatorRegionSourceSegment> source_segments,
      tensorcast::common::v1::PayloadRefDirection direction,
      std::string_view operation_id = "",
      absl::Time capability_expires_at = absl::InfiniteFuture(),
      std::string_view consumer_daemon_id = "");

  [[nodiscard]] absl::StatusOr<RefMetadata> inspect_payload_ref(
      std::string_view payload_ref,
      absl::Time now,
      bool require_not_expired) const;

  [[nodiscard]] absl::StatusOr<BatchRefMetadata> inspect_batch_payload_ref(
      std::string_view batch_payload_ref,
      absl::Time now,
      bool require_not_expired) const;

  [[nodiscard]] absl::StatusOr<PayloadRefFrontDoorContext> inspect_payload_ref_context(
      std::string_view payload_ref,
      std::string_view expected_artifact_id,
      absl::Time now,
      tensorcast::common::v1::PayloadRefDirection expected_direction =
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED,
      std::string_view expected_operation_id = "");

  [[nodiscard]] absl::StatusOr<ResolvedPayload> resolve_local_payload_ref(
      std::string_view payload_ref,
      std::string_view expected_artifact_id,
      absl::Time now,
      tensorcast::common::v1::PayloadRefDirection expected_direction =
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED,
      std::string_view expected_operation_id = "");

  [[nodiscard]] absl::StatusOr<ResolvedSourceCapability> resolve_payload_ref_capability(
      WorkerDirectoryCache& worker_directory_cache,
      std::string_view payload_ref,
      std::string_view expected_artifact_id,
      absl::Time now,
      absl::Duration worker_directory_staleness_budget,
      std::string_view local_daemon_id,
      tensorcast::common::v1::PayloadRefDirection expected_direction =
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED,
      std::string_view expected_operation_id = "");

  [[nodiscard]] absl::StatusOr<ResolvedPayload> fetch_payload_ref(
      WorkerDirectoryCache& worker_directory_cache,
      absl::Time now,
      absl::Duration worker_directory_staleness_budget,
      std::string_view local_daemon_id,
      std::string_view payload_ref,
      std::string_view expected_artifact_id,
      tensorcast::common::v1::PayloadRefDirection expected_direction =
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED,
      std::string_view expected_operation_id = "");

  [[nodiscard]] absl::StatusOr<PayloadLoader> open_payload_ref_loader(
      WorkerDirectoryCache& worker_directory_cache,
      absl::Time now,
      absl::Duration worker_directory_staleness_budget,
      std::string_view local_daemon_id,
      std::string_view payload_ref,
      std::string_view expected_artifact_id,
      tensorcast::common::v1::PayloadRefDirection expected_direction =
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED,
      std::string_view expected_operation_id = "");

  struct PayloadChunk {
    RefMetadata metadata;
    std::string chunk;
    bool eof{false};
  };

  [[nodiscard]] absl::StatusOr<PayloadChunk> read_local_payload_ref_chunk(
      std::string_view payload_ref,
      std::string_view expected_artifact_id,
      absl::Time now,
      std::uint64_t offset,
      std::uint64_t max_bytes,
      tensorcast::common::v1::PayloadRefDirection expected_direction =
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED,
      std::string_view expected_operation_id = "");

  struct BatchPayloadChunk {
    BatchRefMetadata metadata;
    std::string chunk;
    bool eof{false};
  };

  [[nodiscard]] absl::StatusOr<BatchPayloadChunk> read_local_batch_payload_ref_chunk(
      std::string_view batch_payload_ref,
      absl::Time now,
      std::uint64_t offset,
      std::uint64_t max_bytes,
      tensorcast::common::v1::PayloadRefDirection expected_direction =
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED,
      std::string_view expected_operation_id = "");

  [[nodiscard]] absl::StatusOr<ResolvedBatchPayload> fetch_batch_payload_ref(
      WorkerDirectoryCache& worker_directory_cache,
      absl::Time now,
      absl::Duration worker_directory_staleness_budget,
      std::string_view local_daemon_id,
      std::string_view batch_payload_ref,
      tensorcast::common::v1::PayloadRefDirection expected_direction =
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED,
      std::string_view expected_operation_id = "");

  [[nodiscard]] absl::StatusOr<BatchPayloadSource> open_batch_payload_communicator_source(
      WorkerDirectoryCache& worker_directory_cache,
      absl::Time now,
      absl::Duration worker_directory_staleness_budget,
      std::string_view local_daemon_id,
      const v2::BatchPayloadTransport& transport,
      tensorcast::common::v1::PayloadRefDirection expected_direction =
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED,
      std::string_view expected_operation_id = "");

  [[nodiscard]] absl::StatusOr<RoutedAuthorityRequest> build_payload_ref_issuer_routed_request(
      const RefMetadata& metadata,
      const FrontDoorCredentialContext& front_door_context,
      std::string_view route_address,
      absl::Span<const LocalObservationRoutingRule> local_observation_rules = {}) const;

  void prune(absl::Time now);

  [[nodiscard]] const std::string& daemon_id() const {
    return daemon_id_;
  }

  [[nodiscard]] std::uint64_t max_chunk_bytes() const {
    return options_.max_chunk_bytes;
  }

  [[nodiscard]] absl::Duration fetch_deadline() const {
    return options_.fetch_deadline;
  }

  [[nodiscard]] bool batch_transport_enabled() const {
    return options_.batch_transport_protocol_version != 0;
  }

  [[nodiscard]] bool batch_transport_communicator_enabled() const {
    return options_.batch_transport_protocol_version >= 2 && options_.communicator_source_enabled &&
        options_.host_memory_export_enabled && options_.comm_manager != nullptr && options_.comm_manager->is_enabled();
  }

  [[nodiscard]] bool batch_transport_segmented_communicator_export_enabled() const {
    return batch_transport_communicator_enabled() && options_.segmented_communicator_export_enabled;
  }

  [[nodiscard]] std::uint32_t batch_transport_protocol_version() const {
    return options_.batch_transport_protocol_version;
  }

  [[nodiscard]] std::uint64_t max_batch_payload_bytes() const {
    return options_.max_batch_payload_bytes;
  }

  [[nodiscard]] std::uint32_t max_batch_items() const {
    return options_.max_batch_items;
  }

  [[nodiscard]] absl::StatusOr<OwnerStageReply> route_authority_stage(
      const RoutedAuthorityRequest& routed_request,
      absl::Time now);

  [[nodiscard]] absl::StatusOr<std::optional<OwnerStageReply>> maybe_route_authority_stage(
      const RoutedAuthorityRequest& routed_request,
      absl::Time now);

 private:
  struct LocalResolvedPayload {
    RefMetadata metadata;
    FrontDoorCredentialContext front_door_context;
    std::shared_ptr<const std::string> payload;
    BodyHandle body_handle;
    BodyDescriptor descriptor;
    std::optional<store::runtime::ingestion::BackingIdentity> backing_identity;
    std::uint64_t backing_instance_generation{0};
  };

  struct Record {
    RefMetadata metadata;
    std::shared_ptr<const std::string> payload;
    BodyHandle body_handle;
    BodyDescriptor descriptor;
    std::optional<store::runtime::ingestion::BackingIdentity> backing_identity;
    std::uint64_t backing_instance_generation{0};
    SessionLifecycleManager::LeaseId lease_id{0};
  };

  struct BatchRecord {
    BatchRefMetadata metadata;
    v2::BatchPayloadManifest manifest;
    std::shared_ptr<const std::string> payload;
    std::optional<store::ExportRegistration> communicator_export;
    std::vector<std::shared_ptr<void>> communicator_export_keepalives;
    bool communicator_export_requires_unregister{false};
    SessionLifecycleManager::LeaseId lease_id{0};
  };

  struct LocalResolvedBatchPayload {
    BatchRefMetadata metadata;
    std::shared_ptr<const std::string> payload;
    std::optional<store::ExportRegistration> communicator_export;
    std::vector<std::shared_ptr<void>> communicator_export_keepalives;
  };

  struct BatchRefIssueResult {
    BatchRefMetadata metadata;
    std::string batch_payload_ref;
    SessionLifecycleManager::LeaseId lease_id{0};
  };

  [[nodiscard]] absl::StatusOr<LocalResolvedPayload> resolve_local_payload_ref_record(
      std::string_view payload_ref,
      std::string_view expected_artifact_id,
      absl::Time now,
      tensorcast::common::v1::PayloadRefDirection expected_direction,
      std::string_view expected_operation_id);
  [[nodiscard]] absl::StatusOr<AuthorityRef> derive_issuer_authority_ref(
      const FrontDoorCredentialContext& front_door_context) const;
  [[nodiscard]] absl::StatusOr<PortableParsedCredential> derive_payload_ref_portable_credential(
      const FrontDoorCredentialContext& front_door_context) const;
  [[nodiscard]] absl::StatusOr<OwnerStageReply> resolve_payload_ref_issuer_reply(
      std::string_view payload_ref,
      const LocalResolvedPayload& local_resolved_payload,
      bool remote_consumer);
  [[nodiscard]] absl::StatusOr<OwnerStageReply> route_payload_ref_issuer_stage(
      const RoutedAuthorityRequest& routed_request,
      absl::Time now);
  [[nodiscard]] absl::StatusOr<ResolvedSourceCapability> resolve_payload_ref_capability_from_reply(
      const OwnerStageReply& issuer_reply) const;
  [[nodiscard]] absl::StatusOr<ResolvedSourceCapability> resolve_remote_payload_ref_capability_via_issuer_route(
      WorkerDirectoryCache& worker_directory_cache,
      std::string_view payload_ref,
      const RefMetadata& metadata,
      const FrontDoorCredentialContext& front_door_context,
      absl::Time now,
      absl::Duration worker_directory_staleness_budget);
  [[nodiscard]] absl::StatusOr<LocalResolvedBatchPayload> resolve_local_batch_payload_ref_record(
      std::string_view batch_payload_ref,
      absl::Time now,
      tensorcast::common::v1::PayloadRefDirection expected_direction,
      std::string_view expected_operation_id);
  [[nodiscard]] absl::StatusOr<BatchRefIssueResult> issue_batch_payload_ref_record(
      const v2::BatchPayloadManifest& manifest,
      std::shared_ptr<const std::string> payload,
      tensorcast::common::v1::PayloadRefDirection direction,
      std::string_view operation_id,
      absl::Time capability_expires_at,
      std::string_view consumer_daemon_id);
  [[nodiscard]] FrontDoorCredentialContext build_payload_ref_front_door_context(
      const RefMetadata& metadata,
      std::string_view payload_ref,
      std::uint64_t subject_generation) const;
  [[nodiscard]] std::string mint_payload_id() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void prune_locked(absl::Time now) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::string daemon_id_;
  common::CapabilityTokenManager* capability_tokens_{nullptr};
  SessionLifecycleManager* lifecycle_manager_{nullptr};
  LifecycleKernel* lifecycle_kernel_{nullptr};
  Options options_;

  mutable absl::Mutex mu_;
  mutable absl::BitGen bitgen_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, Record> records_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, BatchRecord> batch_records_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
