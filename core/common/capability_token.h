// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "tensorcast/common/v1/capability_token.pb.h"

namespace tensorcast::common {

struct CapabilityTokenKey {
  uint32_t version{0};
  std::string secret;
};

struct CapabilityTokenConfig {
  CapabilityTokenKey active;
  std::vector<CapabilityTokenKey> previous;
};

class CapabilityTokenManager {
 public:
  explicit CapabilityTokenManager(CapabilityTokenConfig config);

  [[nodiscard]] bool configured() const;

  [[nodiscard]] absl::StatusOr<std::string> mint(
      std::string_view issuer_daemon_id,
      tensorcast::common::v1::CapabilityAudience audience,
      std::string_view scope_bytes,
      uint64_t expires_at_ms,
      const tensorcast::common::v1::QueueEpochFencing* queue_epoch = nullptr) const;

  [[nodiscard]] absl::StatusOr<tensorcast::common::v1::CapabilityTokenEnvelope> verify(
      std::string_view token_bytes,
      tensorcast::common::v1::CapabilityAudience expected_audience,
      std::string_view expected_issuer,
      absl::Time now,
      bool require_not_expired) const;

  [[nodiscard]] static bool looks_like_envelope(std::string_view token_bytes);

  [[nodiscard]] static absl::StatusOr<std::string> serialize_scope_deterministic(
      const google::protobuf::Message& message);

 private:
  [[nodiscard]] absl::StatusOr<std::string> compute_auth_tag(
      const tensorcast::common::v1::CapabilityTokenEnvelope& envelope,
      std::string_view secret) const;

  CapabilityTokenConfig config_;
};

} // namespace tensorcast::common
