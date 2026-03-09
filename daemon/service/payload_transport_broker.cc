// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/payload_transport_broker.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/dataplane/contracts/inline_buffer_loader.h"
#include "daemon/service/serving_lifecycle.h"
#include "daemon/state/worker_directory_cache.h"
#include "grpcpp/grpcpp.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"
#include "tensorcast/common/v1/capability_token.pb.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {

namespace {

std::string to_lower_copy(std::string_view value) {
  std::string out(value);
  absl::AsciiStrToLower(&out);
  return out;
}

std::string compute_sha256_hex(std::string_view payload) {
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  std::string hex =
      absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));
  absl::AsciiStrToLower(&hex);
  return hex;
}

absl::StatusOr<absl::Time> resolve_payload_ref_expiry(
    absl::Time now,
    absl::Duration default_ttl,
    absl::Time capability_expires_at) {
  absl::Time expires_at = now + default_ttl;
  if (capability_expires_at != absl::InfiniteFuture()) {
    expires_at = std::min(expires_at, capability_expires_at);
  }
  if (expires_at <= now) {
    return absl::FailedPreconditionError("payload_ref capability is already expired");
  }
  return expires_at;
}

const char* capability_mode_label(BodyCapabilityResolutionMode mode) {
  switch (mode) {
    case BodyCapabilityResolutionMode::kLocalBodyHandle:
      return "local_body_handle";
    case BodyCapabilityResolutionMode::kChunkRpcFallback:
      return "chunk_rpc_fallback";
    case BodyCapabilityResolutionMode::kLoader:
    default:
      return "loader";
  }
}

void record_payload_ref_resolution_metrics(BodyCapabilityResolutionMode mode, bool local) {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_body_capability_resolution_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("mode", opentelemetry::common::AttributeValue(std::string(capability_mode_label(mode))));
    attrs.emplace("local", opentelemetry::common::AttributeValue(local ? "true" : "false"));
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

absl::Status absl_status_from_batch_item_status(v2::BatchItemStatus status, std::string_view message) {
  switch (status) {
    case v2::BATCH_ITEM_STATUS_OK:
      return absl::OkStatus();
    case v2::BATCH_ITEM_STATUS_MISS:
      return absl::NotFoundError(std::string(message));
    case v2::BATCH_ITEM_STATUS_UNAVAILABLE:
      return absl::UnavailableError(std::string(message));
    case v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION:
      return absl::FailedPreconditionError(std::string(message));
    case v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT:
      return absl::InvalidArgumentError(std::string(message));
    case v2::BATCH_ITEM_STATUS_INTERNAL_ERROR:
    case v2::BATCH_ITEM_STATUS_UNSPECIFIED:
    default:
      return absl::InternalError(std::string(message));
  }
}

class SharedStringSource final : public store::loader::SeekableSource {
 public:
  explicit SharedStringSource(std::shared_ptr<const std::string> payload) : payload_(std::move(payload)) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return payload_ ? payload_->size() : 0;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(cursor_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or.status();
    }
    cursor_ += *read_or;
    return *read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (!payload_ || offset >= payload_->size() || bytes == 0) {
      return static_cast<size_t>(0);
    }
    const size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, payload_->size() - offset));
    std::memcpy(dst, payload_->data() + offset, to_copy);
    return to_copy;
  }

  [[nodiscard]] bool supports_direct_write_at() const override {
    return true;
  }

  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const store::DirectWriteGrant& grant) override {
    if (!payload_ || bytes == 0 || src_offset >= payload_->size()) {
      return static_cast<size_t>(0);
    }
    const size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, payload_->size() - src_offset));
    const store::DirectWriteGrant::Window* target = nullptr;
    for (const auto& window : grant.windows) {
      if (dest_va_offset >= window.va_offset && dest_va_offset + to_copy <= window.va_offset + window.length) {
        target = &window;
        break;
      }
    }
    if (target == nullptr) {
      return absl::InvalidArgumentError("No direct-write window covers requested payload range");
    }
    const uint64_t window_offset = dest_va_offset - target->va_offset;
    auto* dst = reinterpret_cast<void*>(target->local_addr + window_offset);
    std::memcpy(dst, payload_->data() + src_offset, to_copy);
    return to_copy;
  }

 private:
  std::shared_ptr<const std::string> payload_;
  uint64_t cursor_{0};
};

class RemotePayloadRefSource final : public store::loader::SeekableSource {
 public:
  struct Options {
    PayloadTransportBroker::RefMetadata metadata;
    std::string payload_ref;
    std::string artifact_id;
    std::string operation_id;
    std::string address;
    std::uint64_t max_chunk_bytes{1ULL << 20};
    absl::Duration fetch_deadline{absl::Seconds(5)};
  };

  explicit RemotePayloadRefSource(Options options)
      : options_(std::move(options)),
        channel_(grpc::CreateChannel(options_.address, grpc::InsecureChannelCredentials())),
        stub_(v2::StoreDaemonService::NewStub(channel_)) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return options_.metadata.payload_size;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(cursor_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or.status();
    }
    cursor_ += *read_or;
    return *read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= options_.metadata.payload_size || bytes == 0) {
      return static_cast<size_t>(0);
    }

    size_t copied = 0;
    const size_t target_bytes = static_cast<size_t>(std::min<uint64_t>(bytes, options_.metadata.payload_size - offset));
    auto* out = static_cast<char*>(dst);

    while (copied < target_bytes) {
      grpc::ClientContext client_ctx;
      client_ctx.set_deadline(
          std::chrono::system_clock::now() +
          std::chrono::milliseconds(absl::ToInt64Milliseconds(options_.fetch_deadline)));
      v2::FetchPayloadRefChunkRequest request;
      request.set_payload_ref(options_.payload_ref);
      request.set_artifact_id(options_.artifact_id);
      request.set_offset(offset + copied);
      request.set_max_bytes(std::min<std::uint64_t>(options_.max_chunk_bytes, target_bytes - copied));
      if (!options_.operation_id.empty()) {
        request.set_operation_id(options_.operation_id);
      }

      v2::FetchPayloadRefChunkResponse response;
      const auto rpc_status = stub_->FetchPayloadRefChunk(&client_ctx, request, &response);
      if (!rpc_status.ok()) {
        return absl::UnavailableError(rpc_status.error_message());
      }
      auto item_status = absl_status_from_batch_item_status(response.status(), response.message());
      if (!item_status.ok()) {
        return item_status;
      }
      if (response.total_size() != options_.metadata.payload_size) {
        return absl::DataLossError("payload_ref total_size mismatch");
      }
      if (response.chunk().empty()) {
        if (response.eof()) {
          break;
        }
        return absl::DataLossError("payload_ref fetch returned empty non-terminal chunk");
      }
      const size_t chunk_bytes = std::min(target_bytes - copied, response.chunk().size());
      std::memcpy(out + copied, response.chunk().data(), chunk_bytes);
      copied += chunk_bytes;
      if (response.eof()) {
        break;
      }
    }
    return copied;
  }

 private:
  Options options_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<v2::StoreDaemonService::Stub> stub_;
  uint64_t cursor_{0};
};

class PayloadRefLoader final : public store::IArtifactLoader {
 public:
  struct RemoteOptions {
    RemotePayloadRefSource::Options source;
  };

  explicit PayloadRefLoader(RemoteOptions options) : remote_(std::move(options)) {}

  absl::Status initialize() override {
    initialized_ = true;
    return absl::OkStatus();
  }

  absl::StatusOr<uint64_t> get_artifact_size() override {
    if (!initialized_) {
      return absl::FailedPreconditionError("PayloadRefLoader not initialized");
    }
    return remote_.source.metadata.payload_size;
  }

  absl::StatusOr<std::unique_ptr<store::loader::SeekableSource>> open_source() override {
    if (!initialized_) {
      return absl::FailedPreconditionError("PayloadRefLoader not initialized");
    }
    return std::unique_ptr<store::loader::SeekableSource>(std::make_unique<RemotePayloadRefSource>(remote_.source));
  }

 private:
  bool initialized_{false};
  RemoteOptions remote_;
};

absl::StatusOr<std::string> read_source_fully(store::loader::SeekableSource& source) {
  const uint64_t total_bytes = source.total_bytes();
  if (total_bytes == 0) {
    return absl::DataLossError("payload_ref fetch returned empty payload");
  }
  if (total_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return absl::OutOfRangeError("payload_ref size exceeds host limits");
  }
  std::string payload;
  payload.resize(static_cast<size_t>(total_bytes));
  size_t copied = 0;
  while (copied < payload.size()) {
    auto read_or = source.read_at(copied, payload.data() + copied, payload.size() - copied);
    if (!read_or.ok()) {
      return read_or.status();
    }
    if (*read_or == 0) {
      return absl::DataLossError("payload_ref fetch terminated before expected size");
    }
    copied += *read_or;
  }
  return payload;
}

} // namespace

PayloadTransportBroker::PayloadTransportBroker(
    std::string daemon_id,
    common::CapabilityTokenManager* capability_tokens,
    Options options)
    : daemon_id_(std::move(daemon_id)), capability_tokens_(capability_tokens), options_(std::move(options)) {}

absl::StatusOr<std::string> PayloadTransportBroker::issue_payload_ref(
    std::string_view artifact_id,
    std::string payload,
    tensorcast::common::v1::PayloadRefDirection direction,
    std::string_view operation_id,
    absl::Time capability_expires_at) {
  return issue_payload_ref(
      artifact_id,
      std::make_shared<const std::string>(std::move(payload)),
      direction,
      operation_id,
      capability_expires_at);
}

absl::StatusOr<std::string> PayloadTransportBroker::issue_payload_ref(
    std::string_view artifact_id,
    std::shared_ptr<const std::string> payload,
    const BodyDescriptor& descriptor,
    tensorcast::common::v1::PayloadRefDirection direction,
    std::string_view operation_id,
    absl::Time capability_expires_at) {
  if (!payload || payload->empty()) {
    return absl::InvalidArgumentError("payload is required for payload_ref issuance");
  }
  const BodyDescriptor normalized_descriptor = normalized_body_descriptor(descriptor);
  if (normalized_descriptor.size_bytes != payload->size()) {
    return absl::FailedPreconditionError("descriptor size does not match payload size");
  }
  if (normalized_descriptor.payload_digest_alg.empty() || normalized_descriptor.payload_digest_hex.empty()) {
    return absl::InvalidArgumentError("descriptor digest is required for payload_ref issuance");
  }
  if (daemon_id_.empty()) {
    return absl::FailedPreconditionError("daemon_id is required for payload_ref issuance");
  }
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required for payload_ref issuance");
  }
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return absl::FailedPreconditionError("capability tokens are required for payload_ref issuance");
  }
  if (direction == tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED) {
    return absl::InvalidArgumentError("payload_ref direction is required");
  }

  const absl::Time now = absl::Now();
  auto expires_at_or = resolve_payload_ref_expiry(now, options_.ttl, capability_expires_at);
  if (!expires_at_or.ok()) {
    return expires_at_or.status();
  }
  const absl::Time expires_at = *expires_at_or;
  const std::string payload_id = [&]() {
    absl::MutexLock lock(&mu_);
    prune_locked(now);
    return mint_payload_id();
  }();

  RefMetadata metadata;
  metadata.issuer_daemon_id = daemon_id_;
  metadata.payload_id = payload_id;
  metadata.artifact_id = std::string(artifact_id);
  metadata.payload_size = payload->size();
  metadata.digest_alg = normalized_descriptor.payload_digest_alg;
  metadata.digest_hex = normalized_descriptor.payload_digest_hex;
  metadata.direction = direction;
  metadata.operation_id = std::string(operation_id);
  metadata.expires_at = expires_at;

  {
    absl::MutexLock lock(&mu_);
    records_[payload_id] = Record{
        .metadata = metadata,
        .payload = std::move(payload),
        .body_handle = BodyHandle(),
        .descriptor = normalized_descriptor,
        .backing_identity = std::nullopt,
        .backing_instance_generation = 0,
    };
  }

  tensorcast::common::v1::PayloadRefScope scope;
  scope.set_payload_id(metadata.payload_id);
  scope.set_artifact_id(metadata.artifact_id);
  scope.set_payload_size(metadata.payload_size);
  scope.set_digest_alg(metadata.digest_alg);
  scope.set_digest_hex(metadata.digest_hex);
  scope.set_direction(direction);
  if (!metadata.operation_id.empty()) {
    scope.set_operation_id(metadata.operation_id);
  }
  auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  if (!scope_or.ok()) {
    return scope_or.status();
  }
  return capability_tokens_->mint(
      daemon_id_,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PAYLOAD_REF,
      *scope_or,
      static_cast<std::uint64_t>(absl::ToUnixMillis(expires_at)));
}

absl::StatusOr<std::string> PayloadTransportBroker::issue_payload_ref(
    std::string_view artifact_id,
    std::shared_ptr<const std::string> payload,
    tensorcast::common::v1::PayloadRefDirection direction,
    std::string_view operation_id,
    absl::Time capability_expires_at) {
  if (daemon_id_.empty()) {
    return absl::FailedPreconditionError("daemon_id is required for payload_ref issuance");
  }
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required for payload_ref issuance");
  }
  if (!payload || payload->empty()) {
    return absl::InvalidArgumentError("payload is required for payload_ref issuance");
  }
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return absl::FailedPreconditionError("capability tokens are required for payload_ref issuance");
  }
  if (direction == tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED) {
    return absl::InvalidArgumentError("payload_ref direction is required");
  }

  const absl::Time now = absl::Now();
  auto expires_at_or = resolve_payload_ref_expiry(now, options_.ttl, capability_expires_at);
  if (!expires_at_or.ok()) {
    return expires_at_or.status();
  }
  const absl::Time expires_at = *expires_at_or;
  const std::string payload_id = [&]() {
    absl::MutexLock lock(&mu_);
    prune_locked(now);
    return mint_payload_id();
  }();

  RefMetadata metadata;
  metadata.issuer_daemon_id = daemon_id_;
  metadata.payload_id = payload_id;
  metadata.artifact_id = std::string(artifact_id);
  metadata.payload_size = payload->size();
  metadata.digest_alg = "sha256";
  metadata.digest_hex = compute_sha256_hex(*payload);
  metadata.direction = direction;
  metadata.operation_id = std::string(operation_id);
  metadata.expires_at = expires_at;

  {
    absl::MutexLock lock(&mu_);
    records_[payload_id] = Record{
        .metadata = metadata,
        .payload = std::move(payload),
        .body_handle = BodyHandle(),
        .descriptor = BodyDescriptor(),
        .backing_identity = std::nullopt,
        .backing_instance_generation = 0,
    };
  }

  tensorcast::common::v1::PayloadRefScope scope;
  scope.set_payload_id(metadata.payload_id);
  scope.set_artifact_id(metadata.artifact_id);
  scope.set_payload_size(metadata.payload_size);
  scope.set_digest_alg(metadata.digest_alg);
  scope.set_digest_hex(metadata.digest_hex);
  scope.set_direction(direction);
  if (!metadata.operation_id.empty()) {
    scope.set_operation_id(metadata.operation_id);
  }
  auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  if (!scope_or.ok()) {
    return scope_or.status();
  }
  return capability_tokens_->mint(
      daemon_id_,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PAYLOAD_REF,
      *scope_or,
      static_cast<std::uint64_t>(absl::ToUnixMillis(expires_at)));
}

absl::StatusOr<std::string> PayloadTransportBroker::issue_payload_ref(
    std::string_view artifact_id,
    const BodyHandle& body_handle,
    const BodyDescriptor& descriptor,
    std::optional<store::runtime::ingestion::BackingIdentity> backing_identity,
    std::uint64_t backing_instance_generation,
    tensorcast::common::v1::PayloadRefDirection direction,
    std::string_view operation_id,
    absl::Time capability_expires_at) {
  if (daemon_id_.empty()) {
    return absl::FailedPreconditionError("daemon_id is required for payload_ref issuance");
  }
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required for payload_ref issuance");
  }
  if (body_handle.empty()) {
    return absl::InvalidArgumentError("body_handle is required for payload_ref issuance");
  }
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return absl::FailedPreconditionError("capability tokens are required for payload_ref issuance");
  }
  if (direction == tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED) {
    return absl::InvalidArgumentError("payload_ref direction is required");
  }
  if (descriptor.payload_digest_alg.empty() || descriptor.payload_digest_hex.empty()) {
    return absl::InvalidArgumentError("descriptor digest is required for payload_ref issuance");
  }
  if (backing_identity.has_value() &&
      !store::runtime::ingestion::backing_identity_matches_replica_key(*backing_identity)) {
    return absl::InvalidArgumentError("backing_identity must match replica_key.artifact_id");
  }
  const BodyDescriptor normalized_descriptor = normalized_body_descriptor(descriptor);
  if (!backing_identity.has_value()) {
    backing_identity = body_descriptor_to_backing_identity(normalized_descriptor, body_handle);
  }
  if (!backing_identity.has_value() ||
      !store::runtime::ingestion::backing_identity_matches_replica_key(*backing_identity)) {
    return absl::InvalidArgumentError("descriptor and body_handle do not identify a valid backing");
  }
  if (backing_instance_generation == 0) {
    backing_instance_generation = body_handle.binding_generation();
  }
  if (backing_instance_generation == 0) {
    return absl::InvalidArgumentError("backing_instance_generation is required for live-backing payload_ref issuance");
  }

  const absl::Time now = absl::Now();
  auto expires_at_or = resolve_payload_ref_expiry(now, options_.ttl, capability_expires_at);
  if (!expires_at_or.ok()) {
    return expires_at_or.status();
  }
  const absl::Time expires_at = *expires_at_or;
  const std::string payload_id = [&]() {
    absl::MutexLock lock(&mu_);
    prune_locked(now);
    return mint_payload_id();
  }();

  RefMetadata metadata;
  metadata.issuer_daemon_id = daemon_id_;
  metadata.payload_id = payload_id;
  metadata.artifact_id = std::string(artifact_id);
  metadata.payload_size = body_handle.size_bytes();
  metadata.digest_alg = descriptor.payload_digest_alg;
  metadata.digest_hex = descriptor.payload_digest_hex;
  metadata.direction = direction;
  metadata.operation_id = std::string(operation_id);
  metadata.expires_at = expires_at;
  if (metadata.payload_size == 0 || metadata.digest_alg.empty() || metadata.digest_hex.empty()) {
    return absl::InvalidArgumentError("body_handle metadata is incomplete for payload_ref issuance");
  }

  {
    absl::MutexLock lock(&mu_);
    records_[payload_id] = Record{
        .metadata = metadata,
        .payload = nullptr,
        .body_handle = body_handle,
        .descriptor = normalized_descriptor,
        .backing_identity = std::move(backing_identity),
        .backing_instance_generation = backing_instance_generation,
    };
  }

  tensorcast::common::v1::PayloadRefScope scope;
  scope.set_payload_id(metadata.payload_id);
  scope.set_artifact_id(metadata.artifact_id);
  scope.set_payload_size(metadata.payload_size);
  scope.set_digest_alg(metadata.digest_alg);
  scope.set_digest_hex(metadata.digest_hex);
  scope.set_direction(direction);
  if (!metadata.operation_id.empty()) {
    scope.set_operation_id(metadata.operation_id);
  }
  auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  if (!scope_or.ok()) {
    return scope_or.status();
  }
  return capability_tokens_->mint(
      daemon_id_,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PAYLOAD_REF,
      *scope_or,
      static_cast<std::uint64_t>(absl::ToUnixMillis(expires_at)));
}

absl::StatusOr<PayloadTransportBroker::RefMetadata> PayloadTransportBroker::inspect_payload_ref(
    std::string_view payload_ref,
    absl::Time now,
    bool require_not_expired) const {
  if (payload_ref.empty()) {
    return absl::InvalidArgumentError("payload_ref is required");
  }
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return absl::FailedPreconditionError("capability tokens are required for payload_ref transport");
  }
  if (!common::CapabilityTokenManager::looks_like_envelope(payload_ref)) {
    return absl::InvalidArgumentError("payload_ref format is invalid");
  }
  auto env_or = capability_tokens_->verify(
      payload_ref,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PAYLOAD_REF,
      /*expected_issuer=*/"",
      now,
      require_not_expired);
  if (!env_or.ok()) {
    return env_or.status();
  }

  tensorcast::common::v1::PayloadRefScope scope;
  if (!scope.ParseFromString(env_or->scope())) {
    return absl::InvalidArgumentError("payload_ref scope parse failed");
  }

  RefMetadata metadata;
  metadata.issuer_daemon_id = env_or->issuer_daemon_id();
  metadata.payload_id = scope.payload_id();
  metadata.artifact_id = scope.artifact_id();
  metadata.payload_size = scope.payload_size();
  metadata.digest_alg = to_lower_copy(scope.digest_alg());
  metadata.digest_hex = to_lower_copy(scope.digest_hex());
  metadata.direction = scope.direction();
  metadata.operation_id = scope.operation_id();
  metadata.expires_at = absl::FromUnixMillis(static_cast<std::int64_t>(env_or->expires_at_ms()));
  if (metadata.payload_id.empty() || metadata.artifact_id.empty() || metadata.payload_size == 0 ||
      metadata.digest_alg.empty() || metadata.digest_hex.empty() ||
      metadata.direction == tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED) {
    return absl::InvalidArgumentError("payload_ref scope missing required fields");
  }
  if (require_not_expired && metadata.expires_at <= now) {
    return absl::PermissionDeniedError("payload_ref expired");
  }
  return metadata;
}

absl::StatusOr<PayloadTransportBroker::ResolvedPayload> PayloadTransportBroker::resolve_local_payload_ref(
    std::string_view payload_ref,
    std::string_view expected_artifact_id,
    absl::Time now,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto resolved_or = resolve_local_payload_ref_record(
      payload_ref, expected_artifact_id, now, expected_direction, expected_operation_id);
  if (!resolved_or.ok()) {
    return resolved_or.status();
  }
  if (resolved_or->payload) {
    return ResolvedPayload{
        .metadata = resolved_or->metadata,
        .payload = *resolved_or->payload,
    };
  }
  auto payload_or = resolved_or->body_handle.read_all_bytes();
  if (!payload_or.ok()) {
    return payload_or.status();
  }
  return ResolvedPayload{
      .metadata = resolved_or->metadata,
      .payload = std::move(*payload_or),
  };
}

absl::StatusOr<PayloadTransportBroker::CapabilityResolution> PayloadTransportBroker::resolve_payload_ref_capability(
    std::string_view payload_ref,
    std::string_view expected_artifact_id,
    absl::Time now,
    std::string_view local_daemon_id,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto metadata_or = inspect_payload_ref(payload_ref, now, /*require_not_expired=*/true);
  if (!metadata_or.ok()) {
    return metadata_or.status();
  }
  if (expected_direction != tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED &&
      metadata_or->direction != expected_direction) {
    return absl::FailedPreconditionError("payload_ref direction mismatch");
  }
  if (!metadata_or->operation_id.empty()) {
    if (expected_operation_id.empty()) {
      return absl::InvalidArgumentError("operation_id is required for payload_ref");
    }
    if (metadata_or->operation_id != expected_operation_id) {
      return absl::FailedPreconditionError("payload_ref operation_id mismatch");
    }
  }

  if (metadata_or->issuer_daemon_id.empty() || metadata_or->issuer_daemon_id == local_daemon_id) {
    auto local_or = resolve_local_payload_ref_record(
        payload_ref, expected_artifact_id, now, expected_direction, expected_operation_id);
    if (!local_or.ok()) {
      return local_or.status();
    }
    CapabilityResolution resolution;
    resolution.metadata = local_or->metadata;
    resolution.payload = local_or->payload;
    if (!local_or->body_handle.empty()) {
      resolution.capability.mode = BodyCapabilityResolutionMode::kLocalBodyHandle;
      resolution.capability.local = true;
      resolution.capability.body_handle = local_or->body_handle;
      resolution.capability.descriptor = local_or->descriptor;
      auto backing_identity = local_or->backing_identity;
      if (!backing_identity.has_value()) {
        backing_identity = body_descriptor_to_backing_identity(local_or->descriptor, local_or->body_handle);
      }
      if (!backing_identity.has_value() ||
          !store::runtime::ingestion::backing_identity_matches_replica_key(*backing_identity)) {
        return absl::FailedPreconditionError("payload_ref backing_identity is missing or inconsistent");
      }
      std::uint64_t backing_instance_generation = local_or->backing_instance_generation;
      if (backing_instance_generation == 0) {
        backing_instance_generation = local_or->body_handle.binding_generation();
      }
      if (backing_instance_generation == 0) {
        return absl::FailedPreconditionError("payload_ref backing_instance_generation is missing");
      }
      auto capability_or = mint_serving_capability(
          MintServingCapabilityRequest{
              .capability_id = std::string(payload_ref),
              .expires_at = resolution.metadata.expires_at,
              .mode = resolution.capability.mode,
              .local = true,
              .subject_kind = ServingCapabilitySubjectKind::kBacking,
              .lifecycle_owner_ref =
                  LifecycleOwnerRef{
                      .owner_kind = LifecycleOwnerKind::kPayloadRefToken,
                      .owner_id = resolution.metadata.payload_id,
                  },
              .backing_identity = std::move(backing_identity),
              .backing_instance_generation = backing_instance_generation,
          });
      if (!capability_or.ok()) {
        return capability_or.status();
      }
      resolution.serving_capability = std::move(*capability_or);
      record_payload_ref_resolution_metrics(resolution.capability.mode, /*local=*/true);
      return resolution;
    }
    resolution.capability.mode = BodyCapabilityResolutionMode::kLoader;
    resolution.capability.local = true;
    auto capability_or = mint_serving_capability(
        MintServingCapabilityRequest{
            .capability_id = std::string(payload_ref),
            .expires_at = resolution.metadata.expires_at,
            .mode = resolution.capability.mode,
            .local = true,
            .subject_kind = ServingCapabilitySubjectKind::kCopiedPayload,
            .lifecycle_owner_ref =
                LifecycleOwnerRef{
                    .owner_kind = LifecycleOwnerKind::kPayloadRefToken,
                    .owner_id = resolution.metadata.payload_id,
                },
        });
    if (!capability_or.ok()) {
      return capability_or.status();
    }
    resolution.serving_capability = std::move(*capability_or);
    record_payload_ref_resolution_metrics(resolution.capability.mode, /*local=*/true);
    return resolution;
  }

  CapabilityResolution resolution;
  resolution.metadata = *metadata_or;
  resolution.capability.mode = BodyCapabilityResolutionMode::kChunkRpcFallback;
  resolution.capability.local = false;
  auto capability_or = mint_serving_capability(
      MintServingCapabilityRequest{
          .capability_id = std::string(payload_ref),
          .expires_at = resolution.metadata.expires_at,
          .mode = resolution.capability.mode,
          .local = false,
          .subject_kind = ServingCapabilitySubjectKind::kCopiedPayload,
          .lifecycle_owner_ref =
              LifecycleOwnerRef{
                  .owner_kind = LifecycleOwnerKind::kPayloadRefToken,
                  .owner_id = resolution.metadata.payload_id,
              },
      });
  if (!capability_or.ok()) {
    return capability_or.status();
  }
  resolution.serving_capability = std::move(*capability_or);
  record_payload_ref_resolution_metrics(resolution.capability.mode, /*local=*/false);
  return resolution;
}

absl::StatusOr<PayloadTransportBroker::LocalResolvedPayload> PayloadTransportBroker::resolve_local_payload_ref_record(
    std::string_view payload_ref,
    std::string_view expected_artifact_id,
    absl::Time now,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto metadata_or = inspect_payload_ref(payload_ref, now, /*require_not_expired=*/true);
  if (!metadata_or.ok()) {
    return metadata_or.status();
  }
  if (!expected_artifact_id.empty() && metadata_or->artifact_id.size() > 0 &&
      metadata_or->artifact_id != expected_artifact_id) {
    return absl::FailedPreconditionError("payload_ref artifact_id mismatch");
  }
  if (expected_direction != tensorcast::common::v1::PAYLOAD_REF_DIRECTION_UNSPECIFIED &&
      metadata_or->direction != expected_direction) {
    return absl::FailedPreconditionError("payload_ref direction mismatch");
  }
  if (!metadata_or->operation_id.empty()) {
    if (expected_operation_id.empty()) {
      return absl::InvalidArgumentError("operation_id is required for payload_ref");
    }
    if (metadata_or->operation_id != expected_operation_id) {
      return absl::FailedPreconditionError("payload_ref operation_id mismatch");
    }
  }

  absl::MutexLock lock(&mu_);
  prune_locked(now);
  auto it = records_.find(metadata_or->payload_id);
  if (it == records_.end()) {
    return absl::NotFoundError("payload_ref is no longer valid");
  }
  if (it->second.metadata.expires_at <= now) {
    if (!it->second.body_handle.empty() && it->second.body_handle.unique_owner()) {
      (void)it->second.body_handle.retire();
    }
    records_.erase(it);
    return absl::PermissionDeniedError("payload_ref expired");
  }
  if (!expected_artifact_id.empty() && it->second.metadata.artifact_id != expected_artifact_id) {
    return absl::FailedPreconditionError("payload_ref artifact_id mismatch");
  }
  if (metadata_or->artifact_id.size() > 0 && metadata_or->artifact_id != it->second.metadata.artifact_id) {
    return absl::FailedPreconditionError("payload_ref metadata mismatch");
  }
  if (metadata_or->payload_size != 0 && metadata_or->payload_size != it->second.metadata.payload_size) {
    return absl::FailedPreconditionError("payload_ref payload_size mismatch");
  }
  if (!metadata_or->digest_alg.empty() && metadata_or->digest_alg != it->second.metadata.digest_alg) {
    return absl::FailedPreconditionError("payload_ref digest_alg mismatch");
  }
  if (!metadata_or->digest_hex.empty() && metadata_or->digest_hex != it->second.metadata.digest_hex) {
    return absl::FailedPreconditionError("payload_ref digest_hex mismatch");
  }
  if (metadata_or->direction != it->second.metadata.direction) {
    return absl::FailedPreconditionError("payload_ref direction mismatch");
  }
  if (metadata_or->operation_id != it->second.metadata.operation_id) {
    return absl::FailedPreconditionError("payload_ref operation_id mismatch");
  }
  return LocalResolvedPayload{
      .metadata = it->second.metadata,
      .payload = it->second.payload,
      .body_handle = it->second.body_handle,
      .descriptor = it->second.descriptor,
      .backing_identity = it->second.backing_identity,
      .backing_instance_generation = it->second.backing_instance_generation,
  };
}

absl::StatusOr<PayloadTransportBroker::PayloadChunk> PayloadTransportBroker::read_local_payload_ref_chunk(
    std::string_view payload_ref,
    std::string_view expected_artifact_id,
    absl::Time now,
    std::uint64_t offset,
    std::uint64_t max_bytes,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto resolved_or = resolve_local_payload_ref_record(
      payload_ref, expected_artifact_id, now, expected_direction, expected_operation_id);
  if (!resolved_or.ok()) {
    return resolved_or.status();
  }
  if (offset > resolved_or->metadata.payload_size) {
    return absl::OutOfRangeError("payload_ref offset exceeds payload size");
  }
  const std::uint64_t remaining = resolved_or->metadata.payload_size - offset;
  const std::uint64_t chunk_bytes = std::min(max_bytes, remaining);
  std::string chunk;
  if (resolved_or->payload) {
    chunk.assign(resolved_or->payload->data() + offset, static_cast<std::size_t>(chunk_bytes));
  } else {
    auto chunk_or = resolved_or->body_handle.read_range(offset, static_cast<std::size_t>(chunk_bytes));
    if (!chunk_or.ok()) {
      return chunk_or.status();
    }
    chunk = std::move(*chunk_or);
  }
  return PayloadChunk{
      .metadata = resolved_or->metadata,
      .chunk = std::move(chunk),
      .eof = (offset + chunk_bytes) >= resolved_or->metadata.payload_size,
  };
}

absl::StatusOr<PayloadTransportBroker::ResolvedPayload> PayloadTransportBroker::fetch_payload_ref(
    WorkerDirectoryCache& worker_directory_cache,
    absl::Time now,
    absl::Duration worker_directory_staleness_budget,
    std::string_view local_daemon_id,
    std::string_view payload_ref,
    std::string_view expected_artifact_id,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto loader_or = open_payload_ref_loader(
      worker_directory_cache,
      now,
      worker_directory_staleness_budget,
      local_daemon_id,
      payload_ref,
      expected_artifact_id,
      expected_direction,
      expected_operation_id);
  if (!loader_or.ok()) {
    return loader_or.status();
  }
  auto init_status = loader_or->loader->initialize();
  if (!init_status.ok()) {
    return init_status;
  }
  auto source_or = loader_or->loader->open_source();
  if (!source_or.ok()) {
    return source_or.status();
  }
  auto payload_or = read_source_fully(**source_or);
  if (!payload_or.ok()) {
    return payload_or.status();
  }
  if (!loader_or->metadata.digest_alg.empty() && loader_or->metadata.digest_alg != "sha256") {
    return absl::FailedPreconditionError("payload_ref digest_alg mismatch");
  }
  if (!loader_or->metadata.digest_hex.empty() && loader_or->metadata.digest_hex != compute_sha256_hex(*payload_or)) {
    return absl::DataLossError("payload_ref digest_hex mismatch");
  }
  return ResolvedPayload{
      .metadata = loader_or->metadata,
      .payload = std::move(*payload_or),
  };
}

absl::StatusOr<PayloadTransportBroker::PayloadLoader> PayloadTransportBroker::open_payload_ref_loader(
    WorkerDirectoryCache& worker_directory_cache,
    absl::Time now,
    absl::Duration worker_directory_staleness_budget,
    std::string_view local_daemon_id,
    std::string_view payload_ref,
    std::string_view expected_artifact_id,
    tensorcast::common::v1::PayloadRefDirection expected_direction,
    std::string_view expected_operation_id) {
  auto resolution_or = resolve_payload_ref_capability(
      payload_ref, expected_artifact_id, now, local_daemon_id, expected_direction, expected_operation_id);
  if (!resolution_or.ok()) {
    return resolution_or.status();
  }
  if (resolution_or->capability.local) {
    if (resolution_or->capability.mode == BodyCapabilityResolutionMode::kLocalBodyHandle) {
      auto loader_or = resolution_or->capability.body_handle.make_loader();
      if (!loader_or.ok()) {
        return loader_or.status();
      }
      return PayloadLoader{
          .metadata = resolution_or->metadata,
          .loader = std::move(*loader_or),
          .remote = false,
      };
    }
    if (!resolution_or->payload) {
      return absl::FailedPreconditionError("payload_ref local record has no body_handle or payload");
    }
    return PayloadLoader{
        .metadata = resolution_or->metadata,
        .loader = std::make_unique<store::InlineBufferLoader>(store::loading::InlineBufferSource{
            .data = std::shared_ptr<const void>(
                resolution_or->payload, static_cast<const void*>(resolution_or->payload->data())),
            .size_bytes = resolution_or->payload->size(),
        }),
        .remote = false,
    };
  }

  auto address_or = worker_directory_cache.resolve_daemon_address(
      resolution_or->metadata.issuer_daemon_id, now, worker_directory_staleness_budget);
  if (!address_or.ok()) {
    return address_or.status();
  }
  return PayloadLoader{
      .metadata = resolution_or->metadata,
      .loader = std::make_unique<PayloadRefLoader>(PayloadRefLoader::RemoteOptions{
          .source =
              RemotePayloadRefSource::Options{
                  .metadata = resolution_or->metadata,
                  .payload_ref = std::string(payload_ref),
                  .artifact_id = std::string(expected_artifact_id),
                  .operation_id = std::string(expected_operation_id),
                  .address = *address_or,
                  .max_chunk_bytes = options_.max_chunk_bytes,
                  .fetch_deadline = options_.fetch_deadline,
              },
      }),
      .remote = true,
  };
}

void PayloadTransportBroker::prune(absl::Time now) {
  absl::MutexLock lock(&mu_);
  prune_locked(now);
}

void PayloadTransportBroker::prune_locked(absl::Time now) {
  std::vector<std::string> expired;
  expired.reserve(records_.size());
  for (const auto& [payload_id, record] : records_) {
    if (record.metadata.expires_at <= now) {
      expired.push_back(payload_id);
    }
  }
  for (const auto& payload_id : expired) {
    auto it = records_.find(payload_id);
    if (it == records_.end()) {
      continue;
    }
    if (!it->second.body_handle.empty() && it->second.body_handle.unique_owner()) {
      (void)it->second.body_handle.retire();
    }
    records_.erase(it);
  }
}

std::string PayloadTransportBroker::mint_payload_id() {
  for (;;) {
    std::string raw;
    raw.resize(16);
    for (char& byte : raw) {
      byte = static_cast<char>(absl::Uniform<std::uint32_t>(bitgen_, 0u, 256u));
    }
    const std::string payload_id = absl::BytesToHexString(raw);
    if (!records_.contains(payload_id)) {
      return payload_id;
    }
  }
}

} // namespace tensorcast::daemon
