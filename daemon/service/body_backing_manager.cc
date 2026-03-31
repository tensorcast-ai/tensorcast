// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/body_backing_manager.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/dataplane/contracts/inline_buffer_loader.h"
#include "core/store/runtime/ingestion/artifact_lowering_plan.h"
#include "openssl/sha.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

namespace {

using tensorcast::common::memory::MemoryLocation;

const char* access_class_label(BodyAccessClass access_class) {
  switch (access_class) {
    case BodyAccessClass::kLocalGpuHot:
      return "local_gpu_hot";
    case BodyAccessClass::kTransientForward:
      return "transient_forward";
    case BodyAccessClass::kSmallObject:
      return "small_object";
    case BodyAccessClass::kHomeDefault:
    default:
      return "home_default";
  }
}

const char* residency_label(BodyPreferredResidency residency) {
  return residency == BodyPreferredResidency::kGpu ? "gpu" : "cpu";
}

const char* retention_label(BodyRetentionIntent retention) {
  return retention == BodyRetentionIntent::kRetained ? "retained" : "ephemeral";
}

const char* stable_requirement_label(BodyStableRetentionRequirement requirement) {
  switch (requirement) {
    case BodyStableRetentionRequirement::kPreferStable:
      return "prefer";
    case BodyStableRetentionRequirement::kRequireStable:
      return "require";
    case BodyStableRetentionRequirement::kNone:
    default:
      return "none";
  }
}

void record_body_backing_metrics(
    std::string_view mode,
    BodyAccessClass access_class,
    const BodyBackingIntent& intent,
    std::string_view outcome) {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_body_backing_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("mode", opentelemetry::common::AttributeValue(std::string(mode)));
    attrs.emplace("access_class", opentelemetry::common::AttributeValue(std::string(access_class_label(access_class))));
    attrs.emplace(
        "preferred_residency",
        opentelemetry::common::AttributeValue(std::string(residency_label(intent.preferred_residency))));
    attrs.emplace(
        "retention_intent",
        opentelemetry::common::AttributeValue(std::string(retention_label(intent.retention_intent))));
    attrs.emplace(
        "stable_requirement",
        opentelemetry::common::AttributeValue(
            std::string(stable_requirement_label(intent.stable_retention_requirement))));
    attrs.emplace("outcome", opentelemetry::common::AttributeValue(std::string(outcome)));
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

std::string build_body_backing_artifact_id(std::string_view artifact_id, const v2::PutIfAbsentInvariant& invariant) {
  if (!verification_mode_requires_payload_digest(invariant_verification_mode(invariant))) {
    static std::atomic<std::uint64_t> logical_backing_seq{1};
    return absl::StrCat(
        "__tc_byte_body__:",
        artifact_id,
        ":",
        invariant.layout_id(),
        ":sz",
        invariant.byte_length(),
        ":logical:",
        absl::Hex(logical_backing_seq.fetch_add(1, std::memory_order_relaxed), absl::kZeroPad16));
  }
  std::string digest_hex = invariant.payload_digest_hex();
  absl::AsciiStrToLower(&digest_hex);
  return absl::StrCat("__tc_byte_body__:", artifact_id, ":", invariant.layout_id(), ":", digest_hex);
}

store::loading::MaterializeHints build_lowering_hints(std::string_view artifact_id, std::string_view operation_id) {
  store::loading::MaterializeHints hints;
  hints.artifact_id = std::string(artifact_id);
  if (!operation_id.empty()) {
    hints.transport_request_id = std::string(operation_id);
  }
  return hints;
}

store::DeviceKey resolve_target_device(const BodyBackingIntent& intent) {
  if (intent.preferred_residency == BodyPreferredResidency::kGpu) {
    return store::DeviceRegistry::instance().gpu_key(0);
  }
  return store::DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
}

store::loading::ReplicaTarget build_replica_target(const BodyBackingIntent& intent) {
  store::loading::ReplicaTarget target;
  if (intent.preferred_residency == BodyPreferredResidency::kGpu) {
    target.location.type = MemoryLocation::GPU;
    target.location.device_id = 0;
    return target;
  }
  target.location.type = MemoryLocation::CPU;
  target.location.device_id = -1;
  return target;
}

bool stable_retention_requested(
    const ResolvedStorePolicy& resolved_policy,
    const BodyBackingIntent& intent,
    const store::loading::ReplicaHandle& replica_handle) {
  if (replica_handle.key().device.type != DeviceType::CPU ||
      intent.stable_retention_requirement == BodyStableRetentionRequirement::kNone) {
    return false;
  }
  return stable_cache_policy_from_resolved(resolved_policy).has_value();
}

absl::StatusOr<BodyStableRetentionState> maybe_admit_stable_retention(
    store::StoreEngine& engine,
    const ResolvedStorePolicy& resolved_policy,
    const BodyBackingIntent& intent,
    const store::loading::ReplicaHandle& replica_handle) {
  if (!stable_retention_requested(resolved_policy, intent, replica_handle)) {
    return BodyStableRetentionState::kNotRequested;
  }

  const auto stable_policy_opt = stable_cache_policy_from_resolved(resolved_policy);
  if (!stable_policy_opt.has_value()) {
    return BodyStableRetentionState::kNotRequested;
  }

  auto admit_or = engine.admit_stable_cache_policy(replica_handle.key(), *stable_policy_opt);
  if (!admit_or.ok()) {
    if (stable_policy_opt->required) {
      return admit_or.status();
    }
    LOG(WARNING) << "body_backing: best-effort stable admission skipped for key=" << replica_handle.key() << ": "
                 << admit_or.status();
    return BodyStableRetentionState::kSkipped;
  }
  return admit_or->admitted ? BodyStableRetentionState::kHeld : BodyStableRetentionState::kSkipped;
}

BodyDescriptor make_body_descriptor(
    const store::runtime::ingestion::BackingIdentity& backing_identity,
    std::string_view layout_id,
    v2::ByteArtifactVerificationMode verification_mode,
    const store::runtime::ingestion::VerifiedContentDescriptor& verified_content_descriptor,
    const store::runtime::ingestion::VerificationRecord& verification_record) {
  BodyDescriptor descriptor;
  descriptor.physical_artifact_id = backing_identity.physical_artifact_id;
  descriptor.layout_id = std::string(layout_id);
  descriptor.size_bytes = verified_content_descriptor.content_identity.logical_size_bytes;
  descriptor.payload_digest_alg = normalize_body_digest_value(verified_content_descriptor.content_identity.digest_alg);
  descriptor.payload_digest_hex = normalize_body_digest_value(
      store::runtime::ingestion::content_digest_bytes_to_hex(
          verified_content_descriptor.content_identity.digest_bytes));
  descriptor.verification_mode = normalize_byte_artifact_verification_mode(verification_mode);
  descriptor.created_at = verification_record.verified_at;
  descriptor.verified_at = verification_record.verified_at;
  return descriptor;
}

bool invariant_matches_descriptor(const v2::PutIfAbsentInvariant& invariant, const BodyDescriptor& descriptor) {
  const auto verification_mode = invariant_verification_mode(invariant);
  if (verification_mode != descriptor.verification_mode || invariant.layout_id() != descriptor.layout_id ||
      invariant.byte_length() != descriptor.size_bytes) {
    return false;
  }
  if (!verification_mode_requires_payload_digest(verification_mode)) {
    return true;
  }
  return normalize_body_digest_value(invariant.payload_digest_alg()) == descriptor.payload_digest_alg &&
      normalize_body_digest_value(invariant.payload_digest_hex()) == descriptor.payload_digest_hex;
}

BodyBackingObservation make_observation(
    const BodyDescriptor& descriptor,
    const store::StoreEngine::ReplicaBackingObservation& core_observation,
    BodyStableRetentionState stable_retention_state,
    absl::Time now) {
  BodyBackingObservation observation;
  observation.physical_artifact_id = descriptor.physical_artifact_id;
  observation.memory_location = core_observation.memory_location;
  observation.size_bytes = core_observation.size_bytes;
  observation.cpu_memfd_available = core_observation.cpu_memfd_available;
  observation.cuda_ipc_available = core_observation.cuda_ipc_available;
  observation.communicator_export_state = core_observation.remote_access_enabled
      ? BodyCommunicatorExportState::kExported
      : BodyCommunicatorExportState::kNotExported;
  observation.stable_retention_state = stable_retention_state;
  observation.observed_at = now;
  return observation;
}

store::loading::InlineBufferSource make_inline_buffer_source(const BodyBackingManager::LocalByteSpan& source) {
  return store::loading::InlineBufferSource{
      .data = source.owner,
      .size_bytes = source.size_bytes,
  };
}

store::runtime::ingestion::VerifiedContentDescriptor build_invariant_verified_content_descriptor(
    const v2::PutIfAbsentInvariant& invariant) {
  if (!verification_mode_requires_payload_digest(invariant_verification_mode(invariant))) {
    return body_descriptor_to_verified_content_descriptor_with_layout(
        invariant.layout_id(), invariant.byte_length(), /*digest_alg=*/"", /*digest_hex=*/"");
  }
  return body_descriptor_to_verified_content_descriptor_with_layout(
      invariant.layout_id(), invariant.byte_length(), invariant.payload_digest_alg(), invariant.payload_digest_hex());
}

store::loader::ByteRangeMap build_identity_byte_range_map(std::uint64_t size_bytes) {
  store::loader::ByteRangeMap map;
  map.total_bytes = size_bytes;
  map.num_sources = 1;
  if (size_bytes > 0) {
    map.segments.push_back(
        store::loader::ByteRangeSegment{
            .kind = store::loader::ByteRangeSegment::Kind::kData,
            .dst_offset = 0,
            .length = size_bytes,
            .src_offset = 0,
            .source_index = 0,
        });
  }
  return map;
}

struct FinalizedStreamDigest {
  std::string digest_alg;
  std::string digest_hex;
  std::string digest_bytes;
  std::uint64_t streamed_bytes{0};
};

class FastCpuDigestState {
 public:
  explicit FastCpuDigestState(std::uint64_t total_size_bytes) : total_size_bytes_(total_size_bytes) {
    SHA256_Init(&ctx_);
  }

  [[nodiscard]] absl::Status update(std::uint64_t offset, const void* copied_bytes, std::size_t bytes) {
    absl::MutexLock lock(&mu_);
    if (!status_.ok()) {
      return status_;
    }
    if (finalized_) {
      return absl::FailedPreconditionError("stream digest already finalized");
    }
    if (offset != next_offset_) {
      status_ = absl::FailedPreconditionError(
          absl::StrCat("stream digest requires contiguous offsets: expected=", next_offset_, " actual=", offset));
      return status_;
    }
    if (bytes == 0) {
      return absl::OkStatus();
    }
    if (SHA256_Update(&ctx_, copied_bytes, bytes) != 1) {
      status_ = absl::InternalError("SHA256_Update failed during fast CPU body staging");
      return status_;
    }
    next_offset_ += bytes;
    return absl::OkStatus();
  }

  [[nodiscard]] absl::StatusOr<FinalizedStreamDigest> finalize() {
    absl::MutexLock lock(&mu_);
    if (!status_.ok()) {
      return status_;
    }
    if (!finalized_) {
      if (next_offset_ != total_size_bytes_) {
        return absl::FailedPreconditionError(
            absl::StrCat("stream digest incomplete: expected=", total_size_bytes_, " streamed=", next_offset_));
      }
      std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
      if (SHA256_Final(digest.data(), &ctx_) != 1) {
        return absl::InternalError("SHA256_Final failed during fast CPU body staging");
      }
      digest_bytes_.assign(reinterpret_cast<const char*>(digest.data()), digest.size());
      digest_hex_ = absl::BytesToHexString(digest_bytes_);
      absl::AsciiStrToLower(&digest_hex_);
      finalized_ = true;
    }
    return FinalizedStreamDigest{
        .digest_alg = "sha256",
        .digest_hex = digest_hex_,
        .digest_bytes = digest_bytes_,
        .streamed_bytes = next_offset_,
    };
  }

  [[nodiscard]] std::uint64_t streamed_bytes() const {
    absl::MutexLock lock(&mu_);
    return next_offset_;
  }

 private:
  const std::uint64_t total_size_bytes_;
  mutable absl::Mutex mu_;
  SHA256_CTX ctx_;
  std::uint64_t next_offset_ ABSL_GUARDED_BY(mu_){0};
  bool finalized_ ABSL_GUARDED_BY(mu_){false};
  absl::Status status_ ABSL_GUARDED_BY(mu_) = absl::OkStatus();
  std::string digest_hex_ ABSL_GUARDED_BY(mu_);
  std::string digest_bytes_ ABSL_GUARDED_BY(mu_);
};

class FastCpuVerifiedInlineLoader final : public store::IArtifactLoader {
 public:
  FastCpuVerifiedInlineLoader(
      BodyBackingManager::LocalByteSpan source,
      std::shared_ptr<FastCpuDigestState> digest_state)
      : source_(std::move(source)), digest_state_(std::move(digest_state)) {}

  absl::Status initialize() override {
    initialized_ = true;
    return absl::OkStatus();
  }

  absl::StatusOr<std::uint64_t> get_artifact_size() override {
    if (!initialized_) {
      return absl::FailedPreconditionError("FastCpuVerifiedInlineLoader not initialized");
    }
    if (!source_.owner || source_.data == nullptr || source_.size_bytes == 0) {
      return absl::FailedPreconditionError("FastCpuVerifiedInlineLoader requires local bytes");
    }
    return source_.size_bytes;
  }

  absl::StatusOr<std::unique_ptr<store::loader::SeekableSource>> open_source() override {
    if (!initialized_) {
      return absl::FailedPreconditionError("FastCpuVerifiedInlineLoader not initialized");
    }
    if (!source_.owner || source_.data == nullptr || source_.size_bytes == 0) {
      return absl::FailedPreconditionError("FastCpuVerifiedInlineLoader requires local bytes");
    }

    class Source final : public store::loader::SeekableSource {
     public:
      Source(BodyBackingManager::LocalByteSpan source, std::shared_ptr<FastCpuDigestState> digest_state)
          : source_(std::move(source)), digest_state_(std::move(digest_state)) {}

      [[nodiscard]] std::uint64_t total_bytes() const override {
        return source_.size_bytes;
      }

      absl::StatusOr<std::size_t> read(void* dst, std::size_t max_bytes) override {
        auto bytes_or = read_at(cursor_, dst, max_bytes);
        if (!bytes_or.ok()) {
          return bytes_or.status();
        }
        cursor_ += *bytes_or;
        return bytes_or;
      }

      absl::StatusOr<std::size_t> read_at(std::uint64_t offset, void* dst, std::size_t bytes) override {
        if (offset >= source_.size_bytes || bytes == 0) {
          return static_cast<std::size_t>(0);
        }
        const std::uint64_t remaining = source_.size_bytes - offset;
        const auto to_copy = static_cast<std::size_t>(std::min<std::uint64_t>(remaining, bytes));
        std::memcpy(dst, source_.data + offset, to_copy);
        if (digest_state_ != nullptr) {
          auto update_status = digest_state_->update(offset, dst, to_copy);
          if (!update_status.ok()) {
            return update_status;
          }
        }
        return to_copy;
      }

      [[nodiscard]] bool supports_direct_write_at() const override {
        return true;
      }

      absl::StatusOr<std::size_t> read_into_at(
          std::uint64_t src_offset,
          std::uint64_t dest_va_offset,
          std::size_t bytes,
          const store::DirectWriteGrant& grant) override {
        if (bytes == 0) {
          return static_cast<std::size_t>(0);
        }
        if (src_offset >= source_.size_bytes) {
          return static_cast<std::size_t>(0);
        }
        if (src_offset != dest_va_offset) {
          return absl::InvalidArgumentError("fast CPU verified source requires identity direct-write offsets");
        }
        const std::uint64_t remaining = source_.size_bytes - src_offset;
        const auto to_copy = static_cast<std::size_t>(std::min<std::uint64_t>(remaining, bytes));
        const store::DirectWriteGrant::Window* target_window = nullptr;
        for (const auto& window : grant.windows) {
          if (dest_va_offset >= window.va_offset && dest_va_offset + to_copy <= window.va_offset + window.length) {
            target_window = &window;
            break;
          }
        }
        if (target_window == nullptr) {
          return absl::InvalidArgumentError("No direct-write window covers requested fast CPU verified range");
        }
        const auto window_offset = dest_va_offset - target_window->va_offset;
        auto* dst = reinterpret_cast<void*>(target_window->local_addr + window_offset);
        std::memcpy(dst, source_.data + src_offset, to_copy);
        if (digest_state_ != nullptr) {
          auto update_status = digest_state_->update(src_offset, dst, to_copy);
          if (!update_status.ok()) {
            return update_status;
          }
        }
        return to_copy;
      }

     private:
      BodyBackingManager::LocalByteSpan source_;
      std::shared_ptr<FastCpuDigestState> digest_state_;
      std::uint64_t cursor_{0};
    };

    return std::unique_ptr<store::loader::SeekableSource>(std::make_unique<Source>(source_, digest_state_));
  }

 private:
  BodyBackingManager::LocalByteSpan source_;
  std::shared_ptr<FastCpuDigestState> digest_state_;
  bool initialized_{false};
};

store::runtime::ingestion::VerificationRecord build_stream_verification_record(
    std::string_view layout_id,
    absl::Time verified_at) {
  return store::runtime::ingestion::VerificationRecord{
      .verification_method = store::runtime::ingestion::VerificationMethod::kSharedExecutorStreamDigest,
      .verified_at = verified_at,
      .layout_proof_kind = store::runtime::ingestion::LayoutProofKind::kNamedLayoutId,
      .layout_proof_value = std::string(layout_id),
  };
}

store::runtime::ingestion::VerificationRecord build_layout_and_size_verification_record(
    std::string_view layout_id,
    absl::Time verified_at) {
  return store::runtime::ingestion::VerificationRecord{
      .verification_method = store::runtime::ingestion::VerificationMethod::kLayoutAndSizeContract,
      .verified_at = verified_at,
      .layout_proof_kind = store::runtime::ingestion::LayoutProofKind::kNamedLayoutId,
      .layout_proof_value = std::string(layout_id),
  };
}

absl::StatusOr<BodyBackingManager::StageResult> finalize_staged_replica(
    store::StoreEngine& engine,
    std::string_view metric_mode,
    BodyAccessClass access_class,
    const BodyBackingIntent& intent,
    const ResolvedStorePolicy& resolved_policy,
    std::string_view layout_id,
    v2::ByteArtifactVerificationMode verification_mode,
    store::loading::ReplicaHandle replica_handle,
    store::runtime::ingestion::BackingIdentity backing_identity,
    store::runtime::ingestion::VerifiedContentDescriptor verified_content_descriptor,
    store::runtime::ingestion::VerificationRecord verification_record,
    absl::Duration* stable_retention_elapsed = nullptr,
    absl::Duration* inspect_backing_elapsed = nullptr,
    absl::Duration* handle_create_elapsed = nullptr) {
  if (backing_identity.replica_key.artifact_id.empty()) {
    backing_identity.replica_key = replica_handle.key();
  }
  if (backing_identity.physical_artifact_id.empty()) {
    backing_identity.physical_artifact_id = replica_handle.key().artifact_id;
  }
  if (!store::runtime::ingestion::backing_identity_matches_replica_key(backing_identity)) {
    (void)engine.retire_replica_status(replica_handle.key());
    record_body_backing_metrics(metric_mode, access_class, intent, "backing_identity_mismatch");
    return absl::InternalError("body staging returned inconsistent backing identity");
  }

  verified_content_descriptor.content_identity.semantic_layout_identity.kind =
      store::runtime::ingestion::SemanticLayoutKind::kNamedLayoutId;
  verified_content_descriptor.content_identity.semantic_layout_identity.value = std::string(layout_id);
  verification_record.layout_proof_kind = store::runtime::ingestion::LayoutProofKind::kNamedLayoutId;
  verification_record.layout_proof_value = std::string(layout_id);
  const BodyDescriptor descriptor = make_body_descriptor(
      backing_identity, layout_id, verification_mode, verified_content_descriptor, verification_record);

  const absl::Time stable_retention_started_at = absl::Now();
  auto stable_state_or = maybe_admit_stable_retention(engine, resolved_policy, intent, replica_handle);
  const absl::Duration stable_elapsed = absl::Now() - stable_retention_started_at;
  if (stable_retention_elapsed != nullptr) {
    *stable_retention_elapsed = stable_elapsed;
  }
  if (!stable_state_or.ok() && intent.stable_retention_requirement == BodyStableRetentionRequirement::kRequireStable) {
    const auto retire_status = engine.retire_replica_status(replica_handle.key());
    if (!retire_status.ok()) {
      record_body_backing_metrics(metric_mode, access_class, intent, "retire_error");
      return retire_status;
    }
    record_body_backing_metrics(metric_mode, access_class, intent, "stable_required_error");
    return stable_state_or.status();
  }

  const absl::Time inspect_backing_started_at = absl::Now();
  auto core_observation_or = engine.inspect_replica_backing(replica_handle.key());
  const absl::Duration inspect_elapsed = absl::Now() - inspect_backing_started_at;
  if (inspect_backing_elapsed != nullptr) {
    *inspect_backing_elapsed = inspect_elapsed;
  }
  if (!core_observation_or.ok()) {
    (void)engine.retire_replica_status(replica_handle.key());
    record_body_backing_metrics(metric_mode, access_class, intent, "observe_error");
    return core_observation_or.status();
  }
  const BodyBackingObservation observation = make_observation(
      descriptor,
      *core_observation_or,
      stable_state_or.ok() ? *stable_state_or : BodyStableRetentionState::kSkipped,
      absl::Now());

  const auto replica_key = replica_handle.key();
  const absl::Time handle_create_started_at = absl::Now();
  auto body_handle_or = BodyHandle::create(engine, std::move(replica_handle));
  const absl::Duration handle_elapsed = absl::Now() - handle_create_started_at;
  if (handle_create_elapsed != nullptr) {
    *handle_create_elapsed = handle_elapsed;
  }
  if (!body_handle_or.ok()) {
    (void)engine.retire_replica_status(replica_key);
    record_body_backing_metrics(metric_mode, access_class, intent, "handle_error");
    return body_handle_or.status();
  }

  record_body_backing_metrics(metric_mode, access_class, intent, "ok");
  return BodyBackingManager::StageResult{
      .descriptor = descriptor,
      .observation = observation,
      .body_handle = std::move(*body_handle_or),
      .verified_content_descriptor = std::move(verified_content_descriptor),
      .verification_record = std::move(verification_record),
      .backing_identity = std::move(backing_identity),
  };
}

} // namespace

BodyBackingManager::BodyBackingManager(store::StoreEngine& engine) : engine_(engine) {}

absl::StatusOr<ResolvedStorePolicy> BodyBackingManager::resolve_body_store_policy(
    BodyAccessClass access_class,
    BodyRouteRole route_role,
    const std::optional<ResolvedStorePolicy>& resolved_store_policy) const {
  ResolvedStorePolicy resolved;
  if (resolved_store_policy.has_value()) {
    resolved = *resolved_store_policy;
  } else {
    auto default_policy_or = resolve_store_policy(nullptr);
    if (!default_policy_or.ok()) {
      return default_policy_or.status();
    }
    resolved = *default_policy_or;
  }
  const auto clear_local_stable = [&resolved]() {
    resolved.local_requirement = RequirementLevel::kNone;
    resolved.local_retention = store::components::StableRetentionPolicy::kBestEffort;
    resolved.local_ttl.reset();
  };

  if (route_role == BodyRouteRole::kTransientForwarder) {
    clear_local_stable();
  }
  switch (access_class) {
    case BodyAccessClass::kLocalGpuHot:
    case BodyAccessClass::kTransientForward:
    case BodyAccessClass::kSmallObject:
      clear_local_stable();
      break;
    case BodyAccessClass::kHomeDefault:
    default:
      break;
  }
  return resolved;
}

BodyPlacementContext BodyBackingManager::normalize_placement_context(
    BodyAccessClass access_class,
    BodyRouteRole route_role,
    std::uint64_t size_bytes) const {
  BodyPlacementContext context;
  context.route_role = route_role;
  context.size_bytes = size_bytes;
  context.expected_fanout = route_role == BodyRouteRole::kTransientForwarder ? 2 : 1;
  context.locality = route_role == BodyRouteRole::kTransientForwarder ? BodyConsumerLocality::kRemoteOrMixed
                                                                      : BodyConsumerLocality::kLocalOnly;
  switch (access_class) {
    case BodyAccessClass::kLocalGpuHot:
      context.access_pattern = BodyAccessPattern::kLocalGpuHot;
      context.locality = BodyConsumerLocality::kLocalOnly;
      break;
    case BodyAccessClass::kTransientForward:
      context.access_pattern = BodyAccessPattern::kTransientForward;
      context.locality = BodyConsumerLocality::kRemoteOrMixed;
      context.expected_fanout = std::max<std::uint32_t>(context.expected_fanout, 2);
      break;
    case BodyAccessClass::kSmallObject:
      context.access_pattern = BodyAccessPattern::kSmallObject;
      context.locality = BodyConsumerLocality::kLocalOnly;
      break;
    case BodyAccessClass::kHomeDefault:
    default:
      context.access_pattern = BodyAccessPattern::kDefault;
      break;
  }
  return context;
}

BodyBackingIntent BodyBackingManager::classify_intent(
    const BodyPlacementContext& context,
    const ResolvedStorePolicy& resolved_policy) const {
  const bool stable_requested = stable_cache_policy_from_resolved(resolved_policy).has_value();
  const auto stable_requirement = [&]() {
    if (!stable_requested) {
      return BodyStableRetentionRequirement::kNone;
    }
    return resolved_policy.local_requirement == RequirementLevel::kMust ? BodyStableRetentionRequirement::kRequireStable
                                                                        : BodyStableRetentionRequirement::kPreferStable;
  }();

  BodyBackingIntent intent;
  intent.preferred_residency = BodyPreferredResidency::kCpu;
  intent.retention_intent = BodyRetentionIntent::kRetained;
  intent.stable_retention_requirement = stable_requirement;
  intent.sharing_intent = context.locality == BodyConsumerLocality::kRemoteOrMixed
      ? BodySharingIntent::kRemoteShareable
      : BodySharingIntent::kLocalReadMostly;

  if (context.route_role == BodyRouteRole::kTransientForwarder ||
      context.access_pattern == BodyAccessPattern::kTransientForward) {
    intent.retention_intent = BodyRetentionIntent::kEphemeral;
    intent.stable_retention_requirement = BodyStableRetentionRequirement::kNone;
    intent.preferred_residency = BodyPreferredResidency::kCpu;
    intent.sharing_intent = BodySharingIntent::kRemoteShareable;
    return intent;
  }

  if (context.access_pattern == BodyAccessPattern::kLocalGpuHot) {
    intent.preferred_residency = BodyPreferredResidency::kGpu;
    intent.sharing_intent = BodySharingIntent::kLocalReadMostly;
    return intent;
  }

  if (context.access_pattern == BodyAccessPattern::kSmallObject) {
    intent.retention_intent = BodyRetentionIntent::kEphemeral;
    intent.stable_retention_requirement = BodyStableRetentionRequirement::kNone;
    intent.sharing_intent = BodySharingIntent::kPrivateLocal;
    return intent;
  }

  return intent;
}

absl::StatusOr<BodyBackingManager::StageResult> BodyBackingManager::stage_body(StageRequest request) const {
  const absl::Time total_started_at = absl::Now();
  absl::Duration resolve_policy_elapsed = absl::ZeroDuration();
  absl::Duration lower_plan_elapsed = absl::ZeroDuration();
  absl::Duration execute_plan_elapsed = absl::ZeroDuration();
  absl::Duration stable_retention_elapsed = absl::ZeroDuration();
  absl::Duration inspect_backing_elapsed = absl::ZeroDuration();
  absl::Duration handle_create_elapsed = absl::ZeroDuration();
  if (request.artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required for body staging");
  }
  if (request.invariant.layout_id().empty()) {
    return absl::InvalidArgumentError("invariant.layout_id is required for body staging");
  }
  if (request.loader == nullptr) {
    return absl::InvalidArgumentError("loader is required for body staging");
  }

  const absl::Time resolve_policy_started_at = absl::Now();
  auto resolved_policy_or =
      resolve_body_store_policy(request.access_class, request.route_role, request.resolved_store_policy);
  resolve_policy_elapsed = absl::Now() - resolve_policy_started_at;
  if (!resolved_policy_or.ok()) {
    return resolved_policy_or.status();
  }
  const BodyPlacementContext context =
      normalize_placement_context(request.access_class, request.route_role, request.invariant.byte_length());
  const BodyBackingIntent intent = classify_intent(context, *resolved_policy_or);
  const auto verification_mode = invariant_verification_mode(request.invariant);
  if (!verification_mode_requires_payload_digest(verification_mode)) {
    auto hints = build_lowering_hints(request.artifact_id, request.operation_id);
    const std::string physical_artifact_id = build_body_backing_artifact_id(request.artifact_id, request.invariant);
    const absl::Time execute_plan_started_at = absl::Now();
    auto replica_handle_or = engine_.ingestion_runtime().ingest_mapped_loader_into_replica(
        request.artifact_id,
        physical_artifact_id,
        resolve_target_device(intent),
        build_replica_target(intent),
        std::move(request.loader),
        build_identity_byte_range_map(request.invariant.byte_length()),
        hints,
        request.source_kind);
    execute_plan_elapsed = absl::Now() - execute_plan_started_at;
    if (!replica_handle_or.ok()) {
      record_body_backing_metrics("stage", request.access_class, intent, "fast_ingest_error");
      return replica_handle_or.status();
    }
    auto stage_result_or = finalize_staged_replica(
        engine_,
        "stage",
        request.access_class,
        intent,
        *resolved_policy_or,
        request.invariant.layout_id(),
        verification_mode,
        std::move(*replica_handle_or),
        store::runtime::ingestion::BackingIdentity{
            .physical_artifact_id = physical_artifact_id,
        },
        build_invariant_verified_content_descriptor(request.invariant),
        build_layout_and_size_verification_record(request.invariant.layout_id(), absl::Now()),
        &stable_retention_elapsed,
        &inspect_backing_elapsed,
        &handle_create_elapsed);
    if (!stage_result_or.ok()) {
      return stage_result_or.status();
    }
    const absl::Duration total_elapsed = absl::Now() - total_started_at;
    if (request.source_kind == store::loading::MaterializationSource::kP2P && total_elapsed >= absl::Milliseconds(1)) {
      LOG(INFO) << "body_backing.stage_body_summary"
                << " artifact_id=" << request.artifact_id << " operation_id=" << request.operation_id
                << " size_bytes=" << request.invariant.byte_length()
                << " source_kind=" << static_cast<int>(request.source_kind)
                << " verification_mode=" << byte_artifact_verification_mode_label(verification_mode)
                << " hash_bypassed=true"
                << " resolve_policy_ms=" << absl::ToDoubleMilliseconds(resolve_policy_elapsed)
                << " lower_plan_ms=" << absl::ToDoubleMilliseconds(lower_plan_elapsed)
                << " execute_plan_ms=" << absl::ToDoubleMilliseconds(execute_plan_elapsed)
                << " stable_retention_ms=" << absl::ToDoubleMilliseconds(stable_retention_elapsed)
                << " inspect_backing_ms=" << absl::ToDoubleMilliseconds(inspect_backing_elapsed)
                << " handle_create_ms=" << absl::ToDoubleMilliseconds(handle_create_elapsed)
                << " total_ms=" << absl::ToDoubleMilliseconds(total_elapsed);
    }
    return stage_result_or;
  }
  const absl::Time lower_plan_started_at = absl::Now();
  auto plan_or = store::runtime::ingestion::lower_to_artifact_plan(
      store::runtime::ingestion::LowerToArtifactPlanRequest{
          .identity =
              store::runtime::ingestion::ArtifactLoweringIdentity{
                  .logical_artifact_id = request.artifact_id,
                  .physical_artifact_id = build_body_backing_artifact_id(request.artifact_id, request.invariant),
                  .request_id = request.operation_id,
              },
          .target_device = resolve_target_device(intent),
          .source_loader = std::move(request.loader),
          .selection_identity =
              tensorcast::common::SelectionIdentity{
                  .artifact_id = request.artifact_id,
                  .logical_layout_hash = tensorcast::common::compute_byte_artifact_logical_layout_hash_bytes(),
                  .selection_hash = tensorcast::common::compute_byte_artifact_selection_hash_bytes(),
              },
          .semantic_layout_identity =
              store::runtime::ingestion::SemanticLayoutIdentity{
                  .kind = store::runtime::ingestion::SemanticLayoutKind::kNamedLayoutId,
                  .value = request.invariant.layout_id(),
              },
          .expected_size_bytes = request.invariant.byte_length(),
          .generation = 1,
          .hints = build_lowering_hints(request.artifact_id, request.operation_id),
          .source_kind = request.source_kind,
          .replica_target = build_replica_target(intent),
      });
  lower_plan_elapsed = absl::Now() - lower_plan_started_at;
  if (!plan_or.ok()) {
    record_body_backing_metrics("stage", request.access_class, intent, "lowering_error");
    return plan_or.status();
  }
  store::runtime::ingestion::ArtifactLoweringPlan plan = std::move(*plan_or);

  const std::string physical_artifact_id = plan.identity.physical_artifact_id;
  const absl::Time execute_plan_started_at = absl::Now();
  auto result_or = engine_.execute_artifact_lowering_plan(std::move(plan));
  execute_plan_elapsed = absl::Now() - execute_plan_started_at;
  if (!result_or.ok()) {
    record_body_backing_metrics("stage", request.access_class, intent, "error");
    return result_or.status();
  }
  if (!result_or->replica_handle.has_value()) {
    record_body_backing_metrics("stage", request.access_class, intent, "missing_replica");
    return absl::InternalError("body staging did not return a replica handle");
  }
  if (!result_or->verified_content_descriptor.has_value() || !result_or->verification_record.has_value() ||
      !result_or->backing_identity.has_value()) {
    record_body_backing_metrics("stage", request.access_class, intent, "missing_descriptor");
    return absl::InternalError("body staging did not return shared truth metadata");
  }

  auto stage_result_or = finalize_staged_replica(
      engine_,
      "stage",
      request.access_class,
      intent,
      *resolved_policy_or,
      request.invariant.layout_id(),
      verification_mode,
      std::move(*result_or->replica_handle),
      *result_or->backing_identity,
      *result_or->verified_content_descriptor,
      *result_or->verification_record,
      &stable_retention_elapsed,
      &inspect_backing_elapsed,
      &handle_create_elapsed);
  if (!stage_result_or.ok()) {
    return stage_result_or.status();
  }
  const absl::Duration total_elapsed = absl::Now() - total_started_at;
  if (request.source_kind == store::loading::MaterializationSource::kP2P && total_elapsed >= absl::Milliseconds(1)) {
    LOG(INFO) << "body_backing.stage_body_summary"
              << " artifact_id=" << request.artifact_id << " operation_id=" << request.operation_id
              << " size_bytes=" << request.invariant.byte_length()
              << " source_kind=" << static_cast<int>(request.source_kind)
              << " verification_mode=" << byte_artifact_verification_mode_label(verification_mode)
              << " hash_bypassed=false"
              << " resolve_policy_ms=" << absl::ToDoubleMilliseconds(resolve_policy_elapsed)
              << " lower_plan_ms=" << absl::ToDoubleMilliseconds(lower_plan_elapsed)
              << " execute_plan_ms=" << absl::ToDoubleMilliseconds(execute_plan_elapsed)
              << " stable_retention_ms=" << absl::ToDoubleMilliseconds(stable_retention_elapsed)
              << " inspect_backing_ms=" << absl::ToDoubleMilliseconds(inspect_backing_elapsed)
              << " handle_create_ms=" << absl::ToDoubleMilliseconds(handle_create_elapsed)
              << " total_ms=" << absl::ToDoubleMilliseconds(total_elapsed);
  }
  return stage_result_or;
}

absl::StatusOr<BodyBackingManager::StageResult> BodyBackingManager::stage_body_fast_cpu_verified(
    std::string artifact_id,
    const v2::PutIfAbsentInvariant& invariant,
    LocalByteSpan source,
    store::loading::MaterializationSource source_kind,
    std::string operation_id,
    BodyAccessClass access_class,
    BodyRouteRole route_role,
    std::optional<ResolvedStorePolicy> resolved_store_policy) const {
  const absl::Time total_started_at = absl::Now();
  absl::Duration resolve_policy_elapsed = absl::ZeroDuration();
  absl::Duration ingest_elapsed = absl::ZeroDuration();
  absl::Duration stable_retention_elapsed = absl::ZeroDuration();
  absl::Duration inspect_backing_elapsed = absl::ZeroDuration();
  absl::Duration handle_create_elapsed = absl::ZeroDuration();
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required for fast CPU body staging");
  }
  if (invariant.layout_id().empty()) {
    return absl::InvalidArgumentError("invariant.layout_id is required for fast CPU body staging");
  }
  if (!source.owner || source.data == nullptr || source.size_bytes == 0) {
    return absl::InvalidArgumentError("fast CPU body staging requires local source bytes");
  }
  if (source.size_bytes != invariant.byte_length()) {
    return absl::InvalidArgumentError("fast CPU body staging source size does not match invariant.byte_length");
  }

  const absl::Time resolve_policy_started_at = absl::Now();
  auto resolved_policy_or = resolve_body_store_policy(access_class, route_role, resolved_store_policy);
  resolve_policy_elapsed = absl::Now() - resolve_policy_started_at;
  if (!resolved_policy_or.ok()) {
    return resolved_policy_or.status();
  }
  const BodyPlacementContext context = normalize_placement_context(access_class, route_role, invariant.byte_length());
  const BodyBackingIntent intent = classify_intent(context, *resolved_policy_or);
  if (intent.preferred_residency != BodyPreferredResidency::kCpu) {
    return stage_body(
        StageRequest{
            .artifact_id = std::move(artifact_id),
            .invariant = invariant,
            .loader = std::make_unique<store::InlineBufferLoader>(make_inline_buffer_source(source)),
            .source_kind = source_kind,
            .operation_id = std::move(operation_id),
            .access_class = access_class,
            .route_role = route_role,
            .resolved_store_policy = std::move(resolved_store_policy),
        });
  }

  const auto verification_mode = invariant_verification_mode(invariant);
  std::shared_ptr<FastCpuDigestState> digest_state;
  if (verification_mode_requires_payload_digest(verification_mode)) {
    digest_state = std::make_shared<FastCpuDigestState>(source.size_bytes);
  }
  auto loader = std::make_unique<FastCpuVerifiedInlineLoader>(source, digest_state);
  auto hints = build_lowering_hints(artifact_id, operation_id);
  hints.pipeline_concurrency = 1;
  const std::string physical_artifact_id = build_body_backing_artifact_id(artifact_id, invariant);
  const absl::Time ingest_started_at = absl::Now();
  auto replica_handle_or = engine_.ingestion_runtime().ingest_mapped_loader_into_replica(
      artifact_id,
      physical_artifact_id,
      resolve_target_device(intent),
      build_replica_target(intent),
      std::move(loader),
      build_identity_byte_range_map(invariant.byte_length()),
      hints,
      source_kind);
  ingest_elapsed = absl::Now() - ingest_started_at;
  if (!replica_handle_or.ok()) {
    record_body_backing_metrics("stage", access_class, intent, "fast_ingest_error");
    return replica_handle_or.status();
  }

  absl::StatusOr<FinalizedStreamDigest> digest_or = absl::FailedPreconditionError("digest disabled");
  if (digest_state != nullptr) {
    digest_or = digest_state->finalize();
  }
  store::runtime::ingestion::VerifiedContentDescriptor verified_content_descriptor;
  store::runtime::ingestion::VerificationRecord verification_record;
  if (!verification_mode_requires_payload_digest(verification_mode)) {
    verified_content_descriptor = build_invariant_verified_content_descriptor(invariant);
    verification_record = build_layout_and_size_verification_record(invariant.layout_id(), absl::Now());
  } else if (digest_or.ok()) {
    verified_content_descriptor = body_descriptor_to_verified_content_descriptor_with_layout(
        invariant.layout_id(), invariant.byte_length(), digest_or->digest_alg, digest_or->digest_hex);
    verification_record = build_stream_verification_record(invariant.layout_id(), absl::Now());
  } else if (digest_state->streamed_bytes() == 0) {
    verified_content_descriptor = build_invariant_verified_content_descriptor(invariant);
    verification_record = build_stream_verification_record(invariant.layout_id(), absl::Now());
  } else {
    (void)engine_.retire_replica_status(replica_handle_or->key());
    record_body_backing_metrics("stage", access_class, intent, "fast_digest_error");
    return digest_or.status();
  }

  auto stage_result_or = finalize_staged_replica(
      engine_,
      "stage",
      access_class,
      intent,
      *resolved_policy_or,
      invariant.layout_id(),
      verification_mode,
      std::move(*replica_handle_or),
      store::runtime::ingestion::BackingIdentity{
          .physical_artifact_id = physical_artifact_id,
      },
      std::move(verified_content_descriptor),
      std::move(verification_record),
      &stable_retention_elapsed,
      &inspect_backing_elapsed,
      &handle_create_elapsed);
  if (!stage_result_or.ok()) {
    return stage_result_or.status();
  }

  const absl::Duration total_elapsed = absl::Now() - total_started_at;
  if (source_kind == store::loading::MaterializationSource::kP2P && total_elapsed >= absl::Milliseconds(1)) {
    LOG(INFO) << "body_backing.stage_body_fast_cpu_summary"
              << " artifact_id=" << artifact_id << " operation_id=" << operation_id
              << " size_bytes=" << invariant.byte_length() << " source_kind=" << static_cast<int>(source_kind)
              << " verification_mode=" << byte_artifact_verification_mode_label(verification_mode)
              << " hash_bypassed=" << (!verification_mode_requires_payload_digest(verification_mode))
              << " streamed_bytes=" << (digest_state != nullptr ? digest_state->streamed_bytes() : 0)
              << " resolve_policy_ms=" << absl::ToDoubleMilliseconds(resolve_policy_elapsed)
              << " ingest_ms=" << absl::ToDoubleMilliseconds(ingest_elapsed)
              << " stable_retention_ms=" << absl::ToDoubleMilliseconds(stable_retention_elapsed)
              << " inspect_backing_ms=" << absl::ToDoubleMilliseconds(inspect_backing_elapsed)
              << " handle_create_ms=" << absl::ToDoubleMilliseconds(handle_create_elapsed)
              << " total_ms=" << absl::ToDoubleMilliseconds(total_elapsed);
  }
  return stage_result_or;
}

absl::StatusOr<std::optional<BodyBackingManager::StageResult>> BodyBackingManager::try_reuse_body(
    ReuseRequest request) const {
  if (request.artifact_id.empty() || request.body_handle.empty()) {
    return std::nullopt;
  }

  auto resolved_policy_or =
      resolve_body_store_policy(request.access_class, request.route_role, request.resolved_store_policy);
  if (!resolved_policy_or.ok()) {
    return resolved_policy_or.status();
  }
  const BodyPlacementContext context =
      normalize_placement_context(request.access_class, request.route_role, request.descriptor.size_bytes);
  const BodyBackingIntent intent = classify_intent(context, *resolved_policy_or);
  if (intent.stable_retention_requirement == BodyStableRetentionRequirement::kRequireStable) {
    record_body_backing_metrics("reuse", request.access_class, intent, "skipped_required_stable");
    return std::nullopt;
  }

  const BodyDescriptor descriptor = normalized_body_descriptor(std::move(request.descriptor));
  if (!invariant_matches_descriptor(request.invariant, descriptor)) {
    record_body_backing_metrics("reuse", request.access_class, intent, "descriptor_mismatch");
    return absl::InvalidArgumentError("descriptor does not match requested invariant for body reuse");
  }
  if (verification_mode_requires_payload_digest(descriptor.verification_mode) &&
      descriptor.physical_artifact_id != build_body_backing_artifact_id(request.artifact_id, request.invariant)) {
    record_body_backing_metrics("reuse", request.access_class, intent, "skipped_identity_mismatch");
    return std::nullopt;
  }

  auto core_observation_or = engine_.inspect_replica_backing(request.body_handle.replica_handle().key());
  if (!core_observation_or.ok()) {
    record_body_backing_metrics("reuse", request.access_class, intent, "observe_error");
    return core_observation_or.status();
  }

  const BodyStableRetentionState stable_state =
      intent.stable_retention_requirement == BodyStableRetentionRequirement::kNone
      ? BodyStableRetentionState::kNotRequested
      : BodyStableRetentionState::kSkipped;
  record_body_backing_metrics("reuse", request.access_class, intent, "ok");
  return std::optional<StageResult>(StageResult{
      .descriptor = descriptor,
      .observation = make_observation(descriptor, *core_observation_or, stable_state, absl::Now()),
      .body_handle = request.body_handle,
      .verified_content_descriptor = body_descriptor_to_verified_content_descriptor(descriptor),
      .verification_record = body_descriptor_to_verification_record(descriptor),
      .backing_identity = body_descriptor_to_backing_identity(descriptor, request.body_handle),
  });
}

} // namespace tensorcast::daemon
