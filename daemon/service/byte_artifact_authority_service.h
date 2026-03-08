// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "daemon/service/body_backing_types.h"
#include "daemon/service/byte_artifact_body_handle.h"
#include "daemon/service/byte_artifact_body_store.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class ByteArtifactAuthorityService {
 public:
  struct Context {
    std::uint64_t shard_id{0};
    std::uint64_t lease_generation{0};
    std::uint64_t routing_epoch{1};
    std::uint64_t shard_count{0};
    absl::Time now{absl::UnixEpoch()};
  };

  struct GetResult {
    std::string artifact_id;
    v2::BatchItemStatus status{v2::BATCH_ITEM_STATUS_UNSPECIFIED};
    std::string message;
    BodyDescriptor descriptor;
    BodyHandle body_handle;
    AuthorityRecord authority_record;
    ServingCapability serving_capability;
  };

  struct PutItem {
    std::string artifact_id;
    v2::PutIfAbsentInvariant invariant;
    BodyDescriptor descriptor;
    store::runtime::ingestion::VerifiedContentDescriptor verified_content_descriptor;
    store::runtime::ingestion::VerificationRecord verification_record;
    BodyBackingObservation observation;
    BodyHandle body_handle;
  };

  explicit ByteArtifactAuthorityService(ByteArtifactBodyStore& body_store);

  [[nodiscard]] std::vector<v2::BatchItemOutcome> batch_exists(
      const std::vector<std::string>& artifact_ids,
      const Context& context) const;

  [[nodiscard]] std::vector<GetResult> batch_get(const std::vector<std::string>& artifact_ids, const Context& context)
      const;

  [[nodiscard]] std::vector<v2::BatchItemOutcome> batch_put_if_absent(
      const std::vector<PutItem>& items,
      const Context& context,
      const std::optional<std::uint64_t>& ttl_ms) const;

  [[nodiscard]] std::vector<v2::BatchItemOutcome> batch_touch_ttl(
      const std::vector<std::string>& artifact_ids,
      const Context& context,
      std::uint64_t ttl_ms) const;

 private:
  ByteArtifactBodyStore& body_store_;
};

} // namespace tensorcast::daemon
