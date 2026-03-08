// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/byte_artifact_body_store.h"

#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

namespace {

using AuthorityEntry = ByteArtifactRuntimeState::AuthorityEntry;

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

std::string replica_key_token(const store::loading::ReplicaKey& key) {
  std::ostringstream stream;
  stream << key;
  return stream.str();
}

void remove_visible_replica_index_locked(
    std::string_view artifact_id,
    const AuthorityEntry& entry,
    ByteArtifactRuntimeState* state) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state == nullptr || entry.visible_body_handle.empty()) {
    return;
  }
  const std::string key = replica_key_token(entry.visible_body_handle.replica_handle().key());
  auto index_it = state->replica_visibility_index.find(key);
  if (index_it == state->replica_visibility_index.end()) {
    return;
  }
  index_it->second.erase(std::string(artifact_id));
  if (index_it->second.empty()) {
    state->replica_visibility_index.erase(index_it);
  }
}

void add_visible_replica_index_locked(
    std::string_view artifact_id,
    const AuthorityEntry& entry,
    ByteArtifactRuntimeState* state) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state == nullptr || entry.visible_body_handle.empty()) {
    return;
  }
  const std::string key = replica_key_token(entry.visible_body_handle.replica_handle().key());
  state->replica_visibility_index[key].insert(std::string(artifact_id));
}

void maybe_retire_visible_backing(AuthorityEntry* entry, std::string_view artifact_id, std::string_view reason) {
  if (entry == nullptr || entry->visible_body_handle.empty()) {
    return;
  }
  auto retire_status = entry->visible_body_handle.retire();
  if (!retire_status.ok()) {
    LOG(WARNING) << "Failed to retire byte-artifact body handle for artifact_id=" << artifact_id << ": "
                 << retire_status;
  }
  record_body_store_retire_metrics(reason);
}

void transition_to_invisible_locked(
    std::string_view artifact_id,
    AuthorityEntry* entry,
    ByteArtifactRuntimeState* state,
    std::string_view reason,
    bool retire_backing) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (entry == nullptr || state == nullptr) {
    return;
  }
  remove_visible_replica_index_locked(artifact_id, *entry, state);
  if (retire_backing) {
    maybe_retire_visible_backing(entry, artifact_id, reason);
  }
  entry->visible_body_handle = BodyHandle();
  entry->visibility_kind = AuthorityVisibilityKind::kNone;
  if (entry->claim_state != AuthorityClaimState::kClaimDeleted) {
    entry->claim_state = AuthorityClaimState::kClaimedInvisible;
  }
}

void erase_claim_locked(std::string_view artifact_id, ByteArtifactRuntimeState* state, std::string_view reason)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state == nullptr) {
    return;
  }
  const std::string key(artifact_id);
  auto entry_it = state->authority_entries.find(key);
  if (entry_it == state->authority_entries.end()) {
    return;
  }
  transition_to_invisible_locked(artifact_id, &entry_it->second, state, reason, /*retire_backing=*/true);
  state->authority_entries.erase(entry_it);
}

bool claim_matches_context_locked(
    std::string_view artifact_id,
    const AuthorityEntry& entry,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time now,
    ByteArtifactRuntimeState* state) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (entry.claim_state == AuthorityClaimState::kClaimDeleted || entry.claim_state == AuthorityClaimState::kUnclaimed) {
    return false;
  }
  if (entry.expires_at != absl::InfiniteFuture() && now >= entry.expires_at) {
    erase_claim_locked(artifact_id, state, "expired");
    return false;
  }
  if (entry.shard_id != shard_id || entry.lease_generation != lease_generation ||
      entry.routing_epoch != routing_epoch) {
    return false;
  }
  return true;
}

bool reconcile_visible_entry_locked(
    std::string_view artifact_id,
    AuthorityEntry* entry,
    ByteArtifactRuntimeState* state) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (entry == nullptr) {
    return false;
  }
  if (entry->visibility_kind != AuthorityVisibilityKind::kReadyBacking || entry->visible_body_handle.empty()) {
    entry->visibility_kind = AuthorityVisibilityKind::kNone;
    if (entry->claim_state != AuthorityClaimState::kClaimDeleted &&
        entry->claim_state != AuthorityClaimState::kUnclaimed) {
      entry->claim_state = AuthorityClaimState::kClaimedInvisible;
    }
    return false;
  }

  auto loader_or = entry->visible_body_handle.make_loader();
  if (!loader_or.ok()) {
    transition_to_invisible_locked(artifact_id, entry, state, "unreadable", /*retire_backing=*/false);
    return false;
  }
  auto init_status = (*loader_or)->initialize();
  if (!init_status.ok()) {
    transition_to_invisible_locked(artifact_id, entry, state, "unreadable", /*retire_backing=*/false);
    return false;
  }
  auto size_or = (*loader_or)->get_artifact_size();
  if (!size_or.ok() || *size_or != entry->claim_descriptor.size_bytes) {
    transition_to_invisible_locked(artifact_id, entry, state, "size_mismatch", /*retire_backing=*/false);
    return false;
  }

  entry->claim_state = AuthorityClaimState::kClaimedVisible;
  entry->visibility_kind = AuthorityVisibilityKind::kReadyBacking;
  return true;
}

AuthorityRecord build_authority_record(std::string_view artifact_id, const AuthorityEntry& entry) {
  return AuthorityRecord{
      .artifact_id = std::string(artifact_id),
      .shard_id = entry.shard_id,
      .lease_generation = entry.lease_generation,
      .routing_epoch = entry.routing_epoch,
      .expires_at = entry.expires_at,
      .visibility_kind = entry.visibility_kind,
      .claim_state = entry.claim_state,
      .visible = entry.claim_state == AuthorityClaimState::kClaimedVisible,
  };
}

std::optional<ByteArtifactBodyStore::EntrySnapshot> lookup_visible_entry_locked(
    std::string_view artifact_id,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time now,
    ByteArtifactRuntimeState* state) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state == nullptr) {
    return std::nullopt;
  }

  auto entry_it = state->authority_entries.find(std::string(artifact_id));
  if (entry_it == state->authority_entries.end()) {
    return std::nullopt;
  }
  auto& entry = entry_it->second;
  if (!claim_matches_context_locked(artifact_id, entry, shard_id, lease_generation, routing_epoch, now, state)) {
    return std::nullopt;
  }
  if (!reconcile_visible_entry_locked(artifact_id, &entry, state)) {
    return std::nullopt;
  }

  return ByteArtifactBodyStore::EntrySnapshot{
      .descriptor = entry.claim_descriptor,
      .verified_content_descriptor = entry.verified_content_descriptor,
      .verification_record = entry.verification_record,
      .body_handle = entry.visible_body_handle,
      .authority_record = build_authority_record(artifact_id, entry),
      .serving_capability = resolve_serving_capability(
          artifact_id, entry.expires_at, BodyCapabilityResolutionMode::kLocalBodyHandle, /*local=*/true),
      .expires_at = entry.expires_at,
  };
}

AuthorityEntry make_authority_entry(
    const BodyDescriptor& descriptor,
    const store::runtime::ingestion::VerifiedContentDescriptor& verified_content_descriptor,
    const store::runtime::ingestion::VerificationRecord& verification_record,
    const BodyBackingObservation& observation,
    const BodyHandle& body_handle,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time expires_at) {
  return AuthorityEntry{
      .claim_descriptor = normalized_body_descriptor(descriptor),
      .verified_content_descriptor = verified_content_descriptor,
      .verification_record = verification_record,
      .last_observation = observation,
      .visible_body_handle = body_handle,
      .expires_at = expires_at,
      .shard_id = shard_id,
      .lease_generation = lease_generation,
      .routing_epoch = routing_epoch,
      .visibility_kind = AuthorityVisibilityKind::kReadyBacking,
      .claim_state = AuthorityClaimState::kClaimedVisible,
  };
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
  return lookup_visible_entry_locked(artifact_id, shard_id, lease_generation, routing_epoch, now, &state_).has_value();
}

std::optional<ByteArtifactBodyStore::EntrySnapshot> ByteArtifactBodyStore::get(
    std::string_view artifact_id,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time now) {
  absl::MutexLock lock(&state_.mu);
  return lookup_visible_entry_locked(artifact_id, shard_id, lease_generation, routing_epoch, now, &state_);
}

ByteArtifactBodyStore::PutResult ByteArtifactBodyStore::put_if_absent(
    std::string_view artifact_id,
    const v2::PutIfAbsentInvariant& invariant,
    const BodyDescriptor& descriptor,
    const store::runtime::ingestion::VerifiedContentDescriptor& verified_content_descriptor,
    const store::runtime::ingestion::VerificationRecord& verification_record,
    const BodyBackingObservation& observation,
    const BodyHandle& body_handle,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time now,
    const std::optional<std::uint64_t>& ttl_ms) {
  absl::MutexLock lock(&state_.mu);
  auto entry_it = state_.authority_entries.find(std::string(artifact_id));
  if (entry_it != state_.authority_entries.end() &&
      claim_matches_context_locked(
          artifact_id, entry_it->second, shard_id, lease_generation, routing_epoch, now, &state_)) {
    auto& entry = entry_it->second;
    if (!invariant_matches_descriptor(invariant, entry.claim_descriptor)) {
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
      extend_expiry_monotonic(new_expiry, &entry.expires_at);
    }

    const bool shares_existing_backing = !entry.visible_body_handle.empty() &&
        entry.visible_body_handle.replica_handle().key() == body_handle.replica_handle().key();
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
  auto entry = make_authority_entry(
      descriptor,
      verified_content_descriptor,
      verification_record,
      observation,
      body_handle,
      shard_id,
      lease_generation,
      routing_epoch,
      resolve_expiry(now, ttl_ms));
  add_visible_replica_index_locked(key, entry, &state_);
  state_.authority_entries[key] = std::move(entry);
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
  auto entry_it = state_.authority_entries.find(std::string(artifact_id));
  if (entry_it == state_.authority_entries.end()) {
    return false;
  }
  if (!claim_matches_context_locked(
          artifact_id, entry_it->second, shard_id, lease_generation, routing_epoch, now, &state_)) {
    return false;
  }
  const absl::Time new_expiry = now + absl::Milliseconds(ttl_ms);
  extend_expiry_monotonic(new_expiry, &entry_it->second.expires_at);
  return true;
}

void ByteArtifactBodyStore::invalidate_artifact_visibility(
    std::string_view artifact_id,
    absl::Time now,
    std::string_view reason) {
  absl::MutexLock lock(&state_.mu);
  auto entry_it = state_.authority_entries.find(std::string(artifact_id));
  if (entry_it == state_.authority_entries.end()) {
    return;
  }
  if (entry_it->second.expires_at != absl::InfiniteFuture() && now >= entry_it->second.expires_at) {
    erase_claim_locked(artifact_id, &state_, "expired");
    return;
  }
  transition_to_invisible_locked(artifact_id, &entry_it->second, &state_, reason, /*retire_backing=*/false);
}

void ByteArtifactBodyStore::invalidate_replica_visibility(
    const store::loading::ReplicaKey& replica_key,
    absl::Time now,
    std::string_view reason) {
  absl::MutexLock lock(&state_.mu);
  const std::string token = replica_key_token(replica_key);
  const auto index_it = state_.replica_visibility_index.find(token);
  if (index_it == state_.replica_visibility_index.end()) {
    return;
  }
  const auto artifact_ids = std::vector<std::string>(index_it->second.begin(), index_it->second.end());
  for (const auto& artifact_id : artifact_ids) {
    auto entry_it = state_.authority_entries.find(artifact_id);
    if (entry_it == state_.authority_entries.end()) {
      continue;
    }
    if (entry_it->second.expires_at != absl::InfiniteFuture() && now >= entry_it->second.expires_at) {
      erase_claim_locked(artifact_id, &state_, "expired");
      continue;
    }
    transition_to_invisible_locked(artifact_id, &entry_it->second, &state_, reason, /*retire_backing=*/false);
  }
}

} // namespace tensorcast::daemon
