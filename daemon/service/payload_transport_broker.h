// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/common/capability_token.h"
#include "core/store/materialization/dataplane/contracts/loader.h"
#include "daemon/service/body_backing_types.h"
#include "daemon/service/byte_artifact_body_handle.h"
#include "daemon/state/lifecycle_kernel.h"
#include "daemon/state/session_lifecycle.h"
#include "tensorcast/common/v1/capability_token.pb.h"

namespace tensorcast::daemon {

class WorkerDirectoryCache;

class PayloadTransportBroker {
 public:
  struct Options {
    absl::Duration ttl{absl::Minutes(5)};
    std::uint64_t max_chunk_bytes{1ULL << 20};
    absl::Duration fetch_deadline{absl::Seconds(5)};
    absl::Duration cleanup_interval{absl::Minutes(1)};
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

  struct PayloadLoader {
    RefMetadata metadata;
    std::unique_ptr<store::IArtifactLoader> loader;
    bool remote{false};
  };

  struct CapabilityResolution {
    RefMetadata metadata;
    ResolvedBodyCapability capability;
    ServingCapability serving_capability;
    std::shared_ptr<const std::string> payload;
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

  [[nodiscard]] absl::StatusOr<RefMetadata> inspect_payload_ref(
      std::string_view payload_ref,
      absl::Time now,
      bool require_not_expired) const;

  [[nodiscard]] absl::StatusOr<ResolvedPayload> resolve_local_payload_ref(
      std::string_view payload_ref,
      std::string_view expected_artifact_id,
      absl::Time now,
      tensorcast::common::v1::PayloadRefDirection expected_direction =
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED,
      std::string_view expected_operation_id = "");

  [[nodiscard]] absl::StatusOr<CapabilityResolution> resolve_payload_ref_capability(
      std::string_view payload_ref,
      std::string_view expected_artifact_id,
      absl::Time now,
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

  void prune(absl::Time now);

  [[nodiscard]] std::uint64_t max_chunk_bytes() const {
    return options_.max_chunk_bytes;
  }

  [[nodiscard]] absl::Duration fetch_deadline() const {
    return options_.fetch_deadline;
  }

 private:
  struct LocalResolvedPayload {
    RefMetadata metadata;
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

  [[nodiscard]] absl::StatusOr<LocalResolvedPayload> resolve_local_payload_ref_record(
      std::string_view payload_ref,
      std::string_view expected_artifact_id,
      absl::Time now,
      tensorcast::common::v1::PayloadRefDirection expected_direction,
      std::string_view expected_operation_id);
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
};

} // namespace tensorcast::daemon
