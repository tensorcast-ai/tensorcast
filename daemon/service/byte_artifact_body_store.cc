// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/byte_artifact_body_store.h"

#include <map>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

namespace {

std::string to_lower_copy(std::string_view value) {
  std::string out(value);
  absl::AsciiStrToLower(&out);
  return out;
}

bool invariant_matches_descriptor(const v2::PutIfAbsentInvariant& invariant, const BodyDescriptor& descriptor) {
  return invariant.layout_id() == descriptor.layout_id && invariant.byte_length() == descriptor.size_bytes &&
      to_lower_copy(invariant.payload_digest_alg()) == descriptor.payload_digest_alg &&
      to_lower_copy(invariant.payload_digest_hex()) == descriptor.payload_digest_hex;
}

void extend_expiry_monotonic(absl::Time new_expiry, absl::Time* expiry) {
  if (expiry == nullptr || *expiry == absl::InfiniteFuture()) {
    return;
  }
  if (new_expiry > *expiry) {
    *expiry = new_expiry;
  }
}

absl::Time resolve_expiry(absl::Time now, const std::optional<std::uint64_t>& ttl_ms) {
  if (!ttl_ms.has_value() || *ttl_ms == 0) {
    return absl::InfiniteFuture();
  }
  return now + absl::Milliseconds(*ttl_ms);
}

void record_body_store_retire_metrics(std::string_view reason) {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_body_store_retire_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("reason", opentelemetry::common::AttributeValue(std::string(reason)));
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void erase_byte_artifact_entry(std::string_view artifact_id, ByteArtifactRuntimeState* state, std::string_view reason)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  const std::string key(artifact_id);
  auto body_it = state->body_handles.find(key);
  if (body_it != state->body_handles.end()) {
    auto retire_status = body_it->second.retire();
    if (!retire_status.ok()) {
      LOG(WARNING) << "Failed to retire byte-artifact body handle for artifact_id=" << artifact_id << ": "
                   << retire_status;
    }
    record_body_store_retire_metrics(reason);
    state->body_handles.erase(body_it);
  }
  state->body_descriptors.erase(key);
  state->body_observations.erase(key);
  state->expires_at.erase(key);
  state->entry_shard_id.erase(key);
  state->entry_lease_generation.erase(key);
  state->entry_routing_epoch.erase(key);
}

bool has_visible_entry_locked(
    std::string_view artifact_id,
    const std::uint64_t shard_id,
    const std::uint64_t lease_generation,
    const std::uint64_t routing_epoch,
    absl::Time now,
    ByteArtifactRuntimeState* state) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  const std::string key(artifact_id);
  const auto body_it = state->body_handles.find(key);
  const auto descriptor_it = state->body_descriptors.find(key);
  const auto observation_it = state->body_observations.find(key);
  const auto expiry_it = state->expires_at.find(key);
  const auto shard_it = state->entry_shard_id.find(key);
  const auto generation_it = state->entry_lease_generation.find(key);
  const auto epoch_it = state->entry_routing_epoch.find(key);
  if (body_it == state->body_handles.end() || descriptor_it == state->body_descriptors.end() ||
      observation_it == state->body_observations.end() || expiry_it == state->expires_at.end() ||
      shard_it == state->entry_shard_id.end() || generation_it == state->entry_lease_generation.end() ||
      epoch_it == state->entry_routing_epoch.end()) {
    erase_byte_artifact_entry(artifact_id, state, "inconsistent");
    return false;
  }
  if (expiry_it->second != absl::InfiniteFuture() && now >= expiry_it->second) {
    erase_byte_artifact_entry(artifact_id, state, "expired");
    return false;
  }
  if (shard_it->second != shard_id || generation_it->second != lease_generation || epoch_it->second != routing_epoch) {
    return false;
  }
  return true;
}

} // namespace

ByteArtifactBodyStore::ByteArtifactBodyStore(ByteArtifactRuntimeState& state) : state_(state) {}

bool ByteArtifactBodyStore::exists(
    std::string_view artifact_id,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time now) {
  absl::MutexLock lock(&state_.mu);
  return has_visible_entry_locked(artifact_id, shard_id, lease_generation, routing_epoch, now, &state_);
}

std::optional<ByteArtifactBodyStore::EntrySnapshot> ByteArtifactBodyStore::get(
    std::string_view artifact_id,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time now) {
  absl::MutexLock lock(&state_.mu);
  if (!has_visible_entry_locked(artifact_id, shard_id, lease_generation, routing_epoch, now, &state_)) {
    return std::nullopt;
  }

  const std::string key(artifact_id);
  const auto body_it = state_.body_handles.find(key);
  const auto descriptor_it = state_.body_descriptors.find(key);
  const auto observation_it = state_.body_observations.find(key);
  const auto expiry_it = state_.expires_at.find(key);
  if (body_it == state_.body_handles.end() || descriptor_it == state_.body_descriptors.end() ||
      observation_it == state_.body_observations.end() || expiry_it == state_.expires_at.end()) {
    erase_byte_artifact_entry(artifact_id, &state_, "inconsistent");
    return std::nullopt;
  }
  return EntrySnapshot{
      .descriptor = descriptor_it->second,
      .body_handle = body_it->second,
      .expires_at = expiry_it->second,
  };
}

ByteArtifactBodyStore::PutResult ByteArtifactBodyStore::put_if_absent(
    std::string_view artifact_id,
    const v2::PutIfAbsentInvariant& invariant,
    const BodyDescriptor& descriptor,
    const BodyBackingObservation& observation,
    const BodyHandle& body_handle,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time now,
    const std::optional<std::uint64_t>& ttl_ms) {
  absl::MutexLock lock(&state_.mu);
  if (has_visible_entry_locked(artifact_id, shard_id, lease_generation, routing_epoch, now, &state_)) {
    const auto key_string = std::string(artifact_id);
    const auto existing_body_it = state_.body_handles.find(key_string);
    const auto descriptor_it = state_.body_descriptors.find(std::string(artifact_id));
    if (descriptor_it == state_.body_descriptors.end() ||
        !invariant_matches_descriptor(invariant, descriptor_it->second)) {
      auto retire_status = body_handle.retire();
      if (!retire_status.ok()) {
        LOG(WARNING) << "Failed to retire conflicting byte-artifact body handle for artifact_id=" << artifact_id << ": "
                     << retire_status;
      }
      record_body_store_retire_metrics("conflict");
      return PutResult{.outcome = PutOutcome::kConflict};
    }
    if (ttl_ms.has_value() && *ttl_ms > 0) {
      const absl::Time new_expiry = now + absl::Milliseconds(*ttl_ms);
      auto& expiry = state_.expires_at[std::string(artifact_id)];
      extend_expiry_monotonic(new_expiry, &expiry);
    }
    const bool shares_existing_backing = existing_body_it != state_.body_handles.end() &&
        existing_body_it->second.replica_handle().key() == body_handle.replica_handle().key();
    if (!shares_existing_backing) {
      auto retire_status = body_handle.retire();
      if (!retire_status.ok()) {
        LOG(WARNING) << "Failed to retire joined byte-artifact body handle for artifact_id=" << artifact_id << ": "
                     << retire_status;
      }
      record_body_store_retire_metrics("join_duplicate");
    }
    return PutResult{.outcome = PutOutcome::kJoined};
  }

  const std::string key(artifact_id);
  state_.body_handles[key] = body_handle;
  state_.body_descriptors[key] = normalized_body_descriptor(descriptor);
  state_.body_observations[key] = observation;
  state_.expires_at[key] = resolve_expiry(now, ttl_ms);
  state_.entry_shard_id[key] = shard_id;
  state_.entry_lease_generation[key] = lease_generation;
  state_.entry_routing_epoch[key] = routing_epoch;
  return PutResult{.outcome = PutOutcome::kCreated};
}

bool ByteArtifactBodyStore::touch_ttl(
    std::string_view artifact_id,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time now,
    std::uint64_t ttl_ms) {
  absl::MutexLock lock(&state_.mu);
  if (!has_visible_entry_locked(artifact_id, shard_id, lease_generation, routing_epoch, now, &state_)) {
    return false;
  }
  const absl::Time new_expiry = now + absl::Milliseconds(ttl_ms);
  auto& expiry = state_.expires_at[std::string(artifact_id)];
  extend_expiry_monotonic(new_expiry, &expiry);
  return true;
}

} // namespace tensorcast::daemon
