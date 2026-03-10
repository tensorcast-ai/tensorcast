// Copyright (c) 2026, TensorCast Team.

#include "core/common/capability_token.h"

#include <cstring>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "openssl/hmac.h"
#include "openssl/sha.h"

namespace tensorcast::common {
namespace {

constexpr size_t kAuthTagBytes = SHA256_DIGEST_LENGTH;

absl::StatusOr<std::string> serialize_deterministic(const google::protobuf::Message& message) {
  const size_t size = message.ByteSizeLong();
  std::string out;
  out.resize(size);
  google::protobuf::io::ArrayOutputStream aos(out.data(), static_cast<int>(out.size()));
  google::protobuf::io::CodedOutputStream cos(&aos);
  cos.SetSerializationDeterministic(true);
  if (!message.SerializeToCodedStream(&cos)) {
    return absl::InternalError("failed to serialize protobuf deterministically");
  }
  return out;
}

} // namespace

CapabilityTokenManager::CapabilityTokenManager(CapabilityTokenConfig config) : config_(std::move(config)) {}

bool CapabilityTokenManager::configured() const {
  return config_.active.version != 0 && !config_.active.secret.empty();
}

absl::StatusOr<std::string> CapabilityTokenManager::serialize_scope_deterministic(
    const google::protobuf::Message& message) {
  return serialize_deterministic(message);
}

absl::StatusOr<std::string> CapabilityTokenManager::compute_auth_tag(
    const tensorcast::common::v1::CapabilityTokenEnvelope& envelope,
    std::string_view secret) const {
  if (secret.empty()) {
    return absl::InvalidArgumentError("capability token secret is empty");
  }
  tensorcast::common::v1::CapabilityTokenEnvelope unsigned_env = envelope;
  unsigned_env.clear_auth_tag();
  auto payload_or = serialize_deterministic(unsigned_env);
  if (!payload_or.ok()) {
    return payload_or.status();
  }
  unsigned int out_len = 0;
  unsigned char out[kAuthTagBytes];
  const unsigned char* result = HMAC(
      EVP_sha256(),
      secret.data(),
      static_cast<int>(secret.size()),
      reinterpret_cast<const unsigned char*>(payload_or->data()),
      payload_or->size(),
      out,
      &out_len);
  if (result == nullptr || out_len != kAuthTagBytes) {
    return absl::InternalError("failed to compute capability token auth tag");
  }
  return std::string(reinterpret_cast<const char*>(out), out_len);
}

absl::StatusOr<std::string> CapabilityTokenManager::mint(
    std::string_view issuer_daemon_id,
    tensorcast::common::v1::CapabilityAudience audience,
    std::string_view scope_bytes,
    uint64_t expires_at_ms,
    const tensorcast::common::v1::QueueEpochFencing* queue_epoch) const {
  if (!configured()) {
    return absl::FailedPreconditionError("capability token keys are not configured");
  }
  if (issuer_daemon_id.empty()) {
    return absl::InvalidArgumentError("issuer_daemon_id is required");
  }
  if (audience == tensorcast::common::v1::CAPABILITY_AUDIENCE_UNSPECIFIED) {
    return absl::InvalidArgumentError("capability token audience is required");
  }
  if (scope_bytes.empty()) {
    return absl::InvalidArgumentError("capability token scope is required");
  }
  if (expires_at_ms == 0) {
    return absl::InvalidArgumentError("capability token expires_at_ms is required");
  }

  tensorcast::common::v1::CapabilityTokenEnvelope envelope;
  envelope.set_token_version(config_.active.version);
  envelope.set_issuer_daemon_id(std::string(issuer_daemon_id));
  envelope.set_audience(audience);
  envelope.set_scope(std::string(scope_bytes));
  envelope.set_expires_at_ms(expires_at_ms);
  if (queue_epoch != nullptr) {
    *envelope.mutable_queue_epoch() = *queue_epoch;
  }

  auto auth_or = compute_auth_tag(envelope, config_.active.secret);
  if (!auth_or.ok()) {
    return auth_or.status();
  }
  envelope.set_auth_tag(*auth_or);

  return serialize_deterministic(envelope);
}

absl::StatusOr<tensorcast::common::v1::CapabilityTokenEnvelope> CapabilityTokenManager::verify(
    std::string_view token_bytes,
    tensorcast::common::v1::CapabilityAudience expected_audience,
    std::string_view expected_issuer,
    absl::Time now,
    bool require_not_expired) const {
  if (!configured()) {
    return absl::FailedPreconditionError("capability token keys are not configured");
  }
  tensorcast::common::v1::CapabilityTokenEnvelope envelope;
  if (!envelope.ParseFromArray(token_bytes.data(), static_cast<int>(token_bytes.size()))) {
    return absl::InvalidArgumentError("capability token parse failed");
  }
  if (envelope.token_version() == 0 || envelope.issuer_daemon_id().empty() ||
      envelope.audience() == tensorcast::common::v1::CAPABILITY_AUDIENCE_UNSPECIFIED || envelope.scope().empty() ||
      envelope.auth_tag().empty()) {
    return absl::InvalidArgumentError("capability token missing required fields");
  }
  if (!expected_issuer.empty() && envelope.issuer_daemon_id() != expected_issuer) {
    return absl::PermissionDeniedError("capability token issuer mismatch");
  }
  if (expected_audience != tensorcast::common::v1::CAPABILITY_AUDIENCE_UNSPECIFIED &&
      envelope.audience() != expected_audience) {
    return absl::PermissionDeniedError("capability token audience mismatch");
  }

  const uint64_t now_ms = static_cast<uint64_t>(absl::ToUnixMillis(now));
  if (require_not_expired && envelope.expires_at_ms() <= now_ms) {
    return absl::PermissionDeniedError("capability token expired");
  }

  std::string secret;
  if (envelope.token_version() == config_.active.version) {
    secret = config_.active.secret;
  } else {
    for (const auto& key : config_.previous) {
      if (key.version == envelope.token_version()) {
        secret = key.secret;
        break;
      }
    }
  }
  if (secret.empty()) {
    return absl::PermissionDeniedError("capability token key version not recognized");
  }

  auto auth_or = compute_auth_tag(envelope, secret);
  if (!auth_or.ok()) {
    return auth_or.status();
  }
  if (auth_or->size() != envelope.auth_tag().size() ||
      std::memcmp(auth_or->data(), envelope.auth_tag().data(), auth_or->size()) != 0) {
    return absl::PermissionDeniedError("capability token auth tag mismatch");
  }
  return envelope;
}

bool CapabilityTokenManager::looks_like_envelope(std::string_view token_bytes) {
  tensorcast::common::v1::CapabilityTokenEnvelope envelope;
  if (!envelope.ParseFromArray(token_bytes.data(), static_cast<int>(token_bytes.size()))) {
    return false;
  }
  if (envelope.token_version() == 0 || envelope.issuer_daemon_id().empty() ||
      envelope.audience() == tensorcast::common::v1::CAPABILITY_AUDIENCE_UNSPECIFIED || envelope.scope().empty() ||
      envelope.auth_tag().empty()) {
    return false;
  }
  return true;
}

} // namespace tensorcast::common
