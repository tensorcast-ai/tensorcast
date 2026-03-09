// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/byte_artifact_body_store.h"

#include <map>
#include <optional>
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

bool content_matches_claim_descriptor(const BodyDescriptor& lhs, const BodyDescriptor& rhs) {
  return lhs.physical_artifact_id == rhs.physical_artifact_id && lhs.layout_id == rhs.layout_id &&
      lhs.size_bytes == rhs.size_bytes && lhs.payload_digest_alg == rhs.payload_digest_alg &&
      lhs.payload_digest_hex == rhs.payload_digest_hex;
}

bool authority_is_visible(const AuthorityEntry& entry) {
  return entry.claim_state == AuthorityClaimState::kClaimedVisible &&
      entry.visibility_kind != AuthorityVisibilityKind::kNone;
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

const char* backing_lifecycle_state_label(BackingLifecycleState state) {
  switch (state) {
    case BackingLifecycleState::kInvalidated:
      return "invalidated";
    case BackingLifecycleState::kSuperseded:
      return "superseded";
    case BackingLifecycleState::kDraining:
      return "draining";
    case BackingLifecycleState::kRetired:
      return "retired";
    case BackingLifecycleState::kActive:
    default:
      return "active";
  }
}

const char* policy_visibility_path_label(PolicyVisibilityPathKind kind) {
  switch (kind) {
    case PolicyVisibilityPathKind::kSharedDisk:
      return "shared_disk";
    case PolicyVisibilityPathKind::kRemoteStable:
      return "remote_stable";
    case PolicyVisibilityPathKind::kUnspecified:
    default:
      return "unspecified";
  }
}

void record_backing_lifecycle_transition_metrics(
    BackingLifecycleState from,
    BackingLifecycleState to,
    std::string_view reason) {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_backing_lifecycle_transition_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("from", opentelemetry::common::AttributeValue(std::string(backing_lifecycle_state_label(from))));
    attrs.emplace("to", opentelemetry::common::AttributeValue(std::string(backing_lifecycle_state_label(to))));
    attrs.emplace("reason", opentelemetry::common::AttributeValue(std::string(reason)));
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void record_policy_visibility_metrics(std::string_view operation, PolicyVisibilityPathKind kind) {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_policy_visibility_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("operation", opentelemetry::common::AttributeValue(std::string(operation)));
    attrs.emplace("path_kind", opentelemetry::common::AttributeValue(std::string(policy_visibility_path_label(kind))));
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void set_backing_lifecycle_state_locked(
    BackingRecord* record,
    BackingLifecycleState next_state,
    std::string_view reason) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (record == nullptr || record->lifecycle_state == next_state) {
    return;
  }
  const auto previous = record->lifecycle_state;
  record->lifecycle_state = next_state;
  record_backing_lifecycle_transition_metrics(previous, next_state, reason);
}

void add_backing_replica_index_locked(const BackingRecord& record, ByteArtifactRuntimeState* state)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state == nullptr) {
    return;
  }
  state->replica_visibility_index[record.identity.replica_key].insert(record.identity);
}

void remove_backing_replica_index_locked(const BackingRecord& record, ByteArtifactRuntimeState* state)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state == nullptr) {
    return;
  }
  auto index_it = state->replica_visibility_index.find(record.identity.replica_key);
  if (index_it == state->replica_visibility_index.end()) {
    return;
  }
  index_it->second.erase(record.identity);
  if (index_it->second.empty()) {
    state->replica_visibility_index.erase(index_it);
  }
}

void link_authority_to_backing_locked(
    std::string_view artifact_id,
    const store::runtime::ingestion::BackingIdentity& identity,
    ByteArtifactRuntimeState* state) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state == nullptr) {
    return;
  }
  state->backing_authority_index[identity].insert(std::string(artifact_id));
}

void unlink_authority_from_backing_locked(
    std::string_view artifact_id,
    const store::runtime::ingestion::BackingIdentity& identity,
    ByteArtifactRuntimeState* state) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state == nullptr) {
    return;
  }
  auto it = state->backing_authority_index.find(identity);
  if (it == state->backing_authority_index.end()) {
    return;
  }
  it->second.erase(std::string(artifact_id));
  if (it->second.empty()) {
    state->backing_authority_index.erase(it);
  }
}

bool backing_has_authority_refs_locked(
    const store::runtime::ingestion::BackingIdentity& identity,
    ByteArtifactRuntimeState* state) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state == nullptr) {
    return false;
  }
  auto it = state->backing_authority_index.find(identity);
  return it != state->backing_authority_index.end() && !it->second.empty();
}

void maybe_retire_backing_handle(const BodyHandle& body_handle, std::string_view reason) {
  if (body_handle.empty() || !body_handle.unique_owner()) {
    return;
  }
  auto retire_status = body_handle.retire();
  if (!retire_status.ok()) {
    LOG(WARNING) << "Failed to retire byte-artifact backing handle: " << retire_status;
    return;
  }
  record_body_store_retire_metrics(reason);
}

void maybe_prune_backing_locked(
    const store::runtime::ingestion::BackingIdentity& identity,
    ByteArtifactRuntimeState* state,
    std::string_view reason) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state == nullptr) {
    return;
  }
  auto backing_it = state->backing_entries.find(identity);
  if (backing_it == state->backing_entries.end()) {
    return;
  }
  if (backing_has_authority_refs_locked(identity, state)) {
    return;
  }
  if (backing_it->second.lifecycle_state == BackingLifecycleState::kActive ||
      backing_it->second.lifecycle_state == BackingLifecycleState::kInvalidated) {
    return;
  }
  maybe_retire_backing_handle(backing_it->second.retained_body_handle, reason);
  if (!backing_it->second.retained_body_handle.empty() && !backing_it->second.retained_body_handle.unique_owner()) {
    set_backing_lifecycle_state_locked(&backing_it->second, BackingLifecycleState::kDraining, reason);
    return;
  }
  set_backing_lifecycle_state_locked(&backing_it->second, BackingLifecycleState::kRetired, reason);
  remove_backing_replica_index_locked(backing_it->second, state);
  state->backing_entries.erase(backing_it);
}

void prune_orphan_backings_locked(ByteArtifactRuntimeState* state, std::string_view reason)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state == nullptr) {
    return;
  }
  std::vector<store::runtime::ingestion::BackingIdentity> candidates;
  candidates.reserve(state->backing_entries.size());
  for (const auto& [identity, record] : state->backing_entries) {
    (void)record;
    if (!backing_has_authority_refs_locked(identity, state)) {
      candidates.push_back(identity);
    }
  }
  for (const auto& identity : candidates) {
    maybe_prune_backing_locked(identity, state, reason);
  }
}

void mark_backing_invalidated_locked(
    const std::optional<store::runtime::ingestion::BackingIdentity>& identity,
    ByteArtifactRuntimeState* state) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (!identity.has_value() || state == nullptr) {
    return;
  }
  auto backing_it = state->backing_entries.find(*identity);
  if (backing_it == state->backing_entries.end()) {
    return;
  }
  set_backing_lifecycle_state_locked(&backing_it->second, BackingLifecycleState::kInvalidated, "backing_invalidated");
}

void transition_to_invisible_locked(AuthorityEntry* entry, ByteArtifactRuntimeState* state)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (entry == nullptr || state == nullptr) {
    return;
  }
  mark_backing_invalidated_locked(entry->retained_backing_identity, state);
  entry->visibility_kind = AuthorityVisibilityKind::kNone;
  if (entry->claim_state != AuthorityClaimState::kClaimDeleted &&
      entry->claim_state != AuthorityClaimState::kUnclaimed) {
    entry->claim_state = AuthorityClaimState::kClaimedInvisible;
  }
}

bool clear_policy_visibility_locked(AuthorityEntry* entry, std::string_view reason) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (entry == nullptr || !entry->policy_visibility_ref.has_value()) {
    return false;
  }
  record_policy_visibility_metrics("clear", entry->policy_visibility_ref->path_kind);
  entry->policy_visibility_ref.reset();
  entry->visibility_kind = AuthorityVisibilityKind::kNone;
  if (entry->claim_state != AuthorityClaimState::kClaimDeleted &&
      entry->claim_state != AuthorityClaimState::kUnclaimed) {
    entry->claim_state = AuthorityClaimState::kClaimedInvisible;
  }
  LOG(INFO) << "byte_artifact.policy_visibility.cleared reason=" << reason;
  return true;
}

bool install_policy_visibility_locked(AuthorityEntry* entry, const PolicyVisibilityRef& policy_visibility_ref)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (entry == nullptr) {
    return false;
  }
  const bool changed = !entry->policy_visibility_ref.has_value() ||
      *entry->policy_visibility_ref != policy_visibility_ref ||
      entry->visibility_kind != AuthorityVisibilityKind::kPolicyBackedPath ||
      entry->claim_state != AuthorityClaimState::kClaimedVisible;
  entry->policy_visibility_ref = policy_visibility_ref;
  entry->visibility_kind = AuthorityVisibilityKind::kPolicyBackedPath;
  entry->claim_state = AuthorityClaimState::kClaimedVisible;
  if (changed) {
    record_policy_visibility_metrics("install", policy_visibility_ref.path_kind);
    LOG(INFO) << "byte_artifact.policy_visibility.installed path_id=" << policy_visibility_ref.path_id
              << " control_ref=" << policy_visibility_ref.control_ref;
  }
  return changed;
}

void retire_authority_backing_locked(
    std::string_view artifact_id,
    AuthorityEntry* entry,
    ByteArtifactRuntimeState* state,
    std::string_view reason) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (entry == nullptr || state == nullptr || !entry->retained_backing_identity.has_value()) {
    return;
  }
  const auto identity = *entry->retained_backing_identity;
  auto backing_it = state->backing_entries.find(identity);
  if (backing_it != state->backing_entries.end()) {
    set_backing_lifecycle_state_locked(&backing_it->second, BackingLifecycleState::kDraining, reason);
  }
  unlink_authority_from_backing_locked(artifact_id, identity, state);
  maybe_prune_backing_locked(identity, state, reason);
  entry->retained_backing_identity.reset();
}

void erase_claim_locked(std::string_view artifact_id, ByteArtifactRuntimeState* state, std::string_view reason)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (state == nullptr) {
    return;
  }
  auto entry_it = state->authority_entries.find(std::string(artifact_id));
  if (entry_it == state->authority_entries.end()) {
    return;
  }
  entry_it->second.claim_state = AuthorityClaimState::kClaimDeleted;
  entry_it->second.visibility_kind = AuthorityVisibilityKind::kNone;
  entry_it->second.policy_visibility_ref.reset();
  retire_authority_backing_locked(artifact_id, &entry_it->second, state, reason);
  state->authority_entries.erase(entry_it);
  prune_orphan_backings_locked(state, reason);
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

bool reconcile_visible_entry_locked(AuthorityEntry* entry, ByteArtifactRuntimeState* state)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (entry == nullptr) {
    return false;
  }
  if (entry->visibility_kind != AuthorityVisibilityKind::kReadyBacking ||
      !entry->retained_backing_identity.has_value()) {
    transition_to_invisible_locked(entry, state);
    return false;
  }
  auto backing_it = state->backing_entries.find(*entry->retained_backing_identity);
  if (backing_it == state->backing_entries.end()) {
    transition_to_invisible_locked(entry, state);
    return false;
  }
  auto& backing = backing_it->second;
  if (backing.lifecycle_state != BackingLifecycleState::kActive) {
    transition_to_invisible_locked(entry, state);
    return false;
  }

  auto loader_or = backing.retained_body_handle.make_loader();
  if (!loader_or.ok()) {
    set_backing_lifecycle_state_locked(&backing, BackingLifecycleState::kInvalidated, "loader_open_failed");
    transition_to_invisible_locked(entry, state);
    return false;
  }
  auto init_status = (*loader_or)->initialize();
  if (!init_status.ok()) {
    set_backing_lifecycle_state_locked(&backing, BackingLifecycleState::kInvalidated, "loader_init_failed");
    transition_to_invisible_locked(entry, state);
    return false;
  }
  auto size_or = (*loader_or)->get_artifact_size();
  if (!size_or.ok() || *size_or != entry->claim_descriptor.size_bytes) {
    set_backing_lifecycle_state_locked(&backing, BackingLifecycleState::kInvalidated, "size_mismatch");
    transition_to_invisible_locked(entry, state);
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
      .retained_backing_identity = entry.retained_backing_identity,
      .policy_visibility_ref = entry.policy_visibility_ref,
      .visible = authority_is_visible(entry),
  };
}

ByteArtifactBodyStore::AuthoritySnapshot build_authority_snapshot(
    std::string_view artifact_id,
    const AuthorityEntry& entry) {
  return ByteArtifactBodyStore::AuthoritySnapshot{
      .descriptor = entry.claim_descriptor,
      .verified_content_descriptor = entry.verified_content_descriptor,
      .verification_record = entry.verification_record,
      .authority_record = build_authority_record(artifact_id, entry),
      .expires_at = entry.expires_at,
  };
}

std::optional<ByteArtifactBodyStore::AuthoritySnapshot> lookup_authority_locked(
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
  if (entry.policy_visibility_ref.has_value() && entry.policy_visibility_ref->expires_at != absl::InfiniteFuture() &&
      now >= entry.policy_visibility_ref->expires_at) {
    clear_policy_visibility_locked(&entry, "expired");
  }
  if (entry.visibility_kind == AuthorityVisibilityKind::kReadyBacking) {
    (void)reconcile_visible_entry_locked(&entry, state);
  }
  return build_authority_snapshot(artifact_id, entry);
}

std::optional<ByteArtifactBodyStore::EntrySnapshot> lookup_visible_entry_locked(
    std::string_view artifact_id,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time now,
    ByteArtifactRuntimeState* state) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  auto authority = lookup_authority_locked(artifact_id, shard_id, lease_generation, routing_epoch, now, state);
  if (!authority.has_value() || !authority->authority_record.visible ||
      authority->authority_record.visibility_kind != AuthorityVisibilityKind::kReadyBacking ||
      !authority->authority_record.retained_backing_identity.has_value()) {
    return std::nullopt;
  }
  auto backing_it = state->backing_entries.find(*authority->authority_record.retained_backing_identity);
  if (backing_it == state->backing_entries.end()) {
    return std::nullopt;
  }
  return ByteArtifactBodyStore::EntrySnapshot{
      .descriptor = authority->descriptor,
      .verified_content_descriptor = authority->verified_content_descriptor,
      .verification_record = authority->verification_record,
      .backing_record = backing_it->second,
      .authority_record = authority->authority_record,
      .expires_at = authority->expires_at,
  };
}

AuthorityEntry make_authority_entry(
    const BodyDescriptor& descriptor,
    const store::runtime::ingestion::VerifiedContentDescriptor& verified_content_descriptor,
    const store::runtime::ingestion::VerificationRecord& verification_record,
    const store::runtime::ingestion::BackingIdentity& backing_identity,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time expires_at) {
  return AuthorityEntry{
      .claim_descriptor = normalized_body_descriptor(descriptor),
      .verified_content_descriptor = verified_content_descriptor,
      .verification_record = verification_record,
      .retained_backing_identity = backing_identity,
      .expires_at = expires_at,
      .shard_id = shard_id,
      .lease_generation = lease_generation,
      .routing_epoch = routing_epoch,
      .visibility_kind = AuthorityVisibilityKind::kReadyBacking,
      .claim_state = AuthorityClaimState::kClaimedVisible,
  };
}

BackingRecord make_backing_record(
    const BodyDescriptor& descriptor,
    const store::runtime::ingestion::VerifiedContentDescriptor& verified_content_descriptor,
    const store::runtime::ingestion::VerificationRecord& verification_record,
    const store::runtime::ingestion::BackingIdentity& backing_identity,
    const BodyBackingObservation& observation,
    const BodyHandle& body_handle,
    std::uint64_t instance_generation,
    BackingLifecycleState lifecycle_state) {
  return BackingRecord{
      .identity = backing_identity,
      .instance_generation = instance_generation,
      .verified_content_descriptor = verified_content_descriptor,
      .verification_record = verification_record,
      .descriptor = normalized_body_descriptor(descriptor),
      .last_observation = observation,
      .retained_body_handle = body_handle,
      .lifecycle_state = lifecycle_state,
  };
}

void install_or_rebind_backing_locked(
    std::string_view artifact_id,
    AuthorityEntry* entry,
    const BodyDescriptor& descriptor,
    const store::runtime::ingestion::VerifiedContentDescriptor& verified_content_descriptor,
    const store::runtime::ingestion::VerificationRecord& verification_record,
    const store::runtime::ingestion::BackingIdentity& backing_identity,
    const BodyBackingObservation& observation,
    const BodyHandle& body_handle,
    ByteArtifactRuntimeState* state) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (entry == nullptr || state == nullptr) {
    return;
  }

  if (entry->retained_backing_identity.has_value() && *entry->retained_backing_identity != backing_identity) {
    const auto old_identity = *entry->retained_backing_identity;
    auto old_backing_it = state->backing_entries.find(old_identity);
    if (old_backing_it != state->backing_entries.end()) {
      set_backing_lifecycle_state_locked(
          &old_backing_it->second, BackingLifecycleState::kSuperseded, "rebind_new_identity");
    }
    unlink_authority_from_backing_locked(artifact_id, old_identity, state);
    if (old_backing_it != state->backing_entries.end() && !backing_has_authority_refs_locked(old_identity, state)) {
      set_backing_lifecycle_state_locked(
          &old_backing_it->second, BackingLifecycleState::kDraining, "authority_released");
    }
  }

  std::uint64_t generation = 1;
  auto backing_it = state->backing_entries.find(backing_identity);
  if (backing_it == state->backing_entries.end()) {
    auto [inserted_it, inserted] = state->backing_entries.emplace(
        backing_identity,
        make_backing_record(
            descriptor,
            verified_content_descriptor,
            verification_record,
            backing_identity,
            observation,
            body_handle,
            generation,
            BackingLifecycleState::kActive));
    (void)inserted;
    add_backing_replica_index_locked(inserted_it->second, state);
  } else {
    const bool old_handle_unique = backing_it->second.retained_body_handle.unique_owner();
    generation = backing_it->second.instance_generation + 1;
    BodyHandle old_handle = backing_it->second.retained_body_handle;
    const bool same_core_subject =
        !old_handle.empty() && old_handle.replica_handle().key() == body_handle.replica_handle().key();
    backing_it->second = make_backing_record(
        descriptor,
        verified_content_descriptor,
        verification_record,
        backing_identity,
        observation,
        body_handle,
        generation,
        BackingLifecycleState::kActive);
    if (!same_core_subject && old_handle_unique) {
      auto retire_status = old_handle.retire();
      if (!retire_status.ok()) {
        LOG(WARNING) << "Failed to retire rebound byte-artifact backing handle: " << retire_status;
      } else {
        record_body_store_retire_metrics("rebind");
      }
    }
  }

  link_authority_to_backing_locked(artifact_id, backing_identity, state);
  entry->claim_descriptor = normalized_body_descriptor(descriptor);
  entry->verified_content_descriptor = verified_content_descriptor;
  entry->verification_record = verification_record;
  entry->retained_backing_identity = backing_identity;
  (void)clear_policy_visibility_locked(entry, "ready_backing_rebound");
  entry->visibility_kind = AuthorityVisibilityKind::kReadyBacking;
  entry->claim_state = AuthorityClaimState::kClaimedVisible;
  prune_orphan_backings_locked(state, "rebind");
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
  auto authority = lookup_authority_locked(artifact_id, shard_id, lease_generation, routing_epoch, now, &state_);
  return authority.has_value() && authority->authority_record.visible;
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

std::optional<ByteArtifactBodyStore::AuthoritySnapshot> ByteArtifactBodyStore::inspect_authority(
    std::string_view artifact_id,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time now) {
  absl::MutexLock lock(&state_.mu);
  return lookup_authority_locked(artifact_id, shard_id, lease_generation, routing_epoch, now, &state_);
}

std::optional<BackingRecord> ByteArtifactBodyStore::inspect_backing(
    const store::runtime::ingestion::BackingIdentity& identity) const {
  absl::MutexLock lock(&state_.mu);
  auto it = state_.backing_entries.find(identity);
  if (it == state_.backing_entries.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<ByteArtifactBodyStore::PersistenceSourceSnapshot> ByteArtifactBodyStore::inspect_persistence_source(
    std::string_view artifact_id) const {
  absl::MutexLock lock(&state_.mu);
  auto authority_it = state_.authority_entries.find(std::string(artifact_id));
  if (authority_it == state_.authority_entries.end()) {
    return std::nullopt;
  }
  const auto& authority = authority_it->second;
  if (authority.claim_state != AuthorityClaimState::kClaimedVisible ||
      authority.visibility_kind != AuthorityVisibilityKind::kReadyBacking ||
      !authority.retained_backing_identity.has_value()) {
    return std::nullopt;
  }
  auto backing_it = state_.backing_entries.find(*authority.retained_backing_identity);
  if (backing_it == state_.backing_entries.end() ||
      backing_it->second.lifecycle_state != BackingLifecycleState::kActive) {
    return std::nullopt;
  }
  return PersistenceSourceSnapshot{
      .source_artifact_id = backing_it->second.identity.replica_key.artifact_id,
      .size_bytes = backing_it->second.descriptor.size_bytes,
      .verified_content_descriptor = authority.verified_content_descriptor,
  };
}

bool ByteArtifactBodyStore::install_policy_visibility(
    std::string_view artifact_id,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time now,
    const PolicyVisibilityRef& policy_visibility_ref) {
  absl::MutexLock lock(&state_.mu);
  auto entry_it = state_.authority_entries.find(std::string(artifact_id));
  if (entry_it == state_.authority_entries.end()) {
    return false;
  }
  if (!claim_matches_context_locked(
          artifact_id, entry_it->second, shard_id, lease_generation, routing_epoch, now, &state_)) {
    return false;
  }
  if (entry_it->second.claim_state == AuthorityClaimState::kClaimDeleted ||
      entry_it->second.claim_state == AuthorityClaimState::kUnclaimed) {
    return false;
  }
  return install_policy_visibility_locked(&entry_it->second, policy_visibility_ref);
}

bool ByteArtifactBodyStore::clear_policy_visibility(
    std::string_view artifact_id,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time now,
    std::string_view reason) {
  absl::MutexLock lock(&state_.mu);
  auto entry_it = state_.authority_entries.find(std::string(artifact_id));
  if (entry_it == state_.authority_entries.end()) {
    return false;
  }
  if (!claim_matches_context_locked(
          artifact_id, entry_it->second, shard_id, lease_generation, routing_epoch, now, &state_)) {
    return false;
  }
  return clear_policy_visibility_locked(&entry_it->second, reason);
}

ByteArtifactBodyStore::PutResult ByteArtifactBodyStore::put_if_absent(
    std::string_view artifact_id,
    const v2::PutIfAbsentInvariant& invariant,
    const BodyDescriptor& descriptor,
    const store::runtime::ingestion::VerifiedContentDescriptor& verified_content_descriptor,
    const store::runtime::ingestion::VerificationRecord& verification_record,
    const store::runtime::ingestion::BackingIdentity& backing_identity,
    const BodyBackingObservation& observation,
    const BodyHandle& body_handle,
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::uint64_t routing_epoch,
    absl::Time now,
    const std::optional<std::uint64_t>& ttl_ms) {
  if (!store::runtime::ingestion::backing_identity_matches_replica_key(backing_identity)) {
    maybe_retire_backing_handle(body_handle, "invalid_backing_identity");
    return PutResult{.outcome = PutOutcome::kConflict};
  }

  absl::MutexLock lock(&state_.mu);
  auto entry_it = state_.authority_entries.find(std::string(artifact_id));
  if (entry_it != state_.authority_entries.end() &&
      claim_matches_context_locked(
          artifact_id, entry_it->second, shard_id, lease_generation, routing_epoch, now, &state_)) {
    auto& entry = entry_it->second;
    const BodyDescriptor normalized_descriptor = normalized_body_descriptor(descriptor);
    if (!invariant_matches_descriptor(invariant, entry.claim_descriptor) ||
        !content_matches_claim_descriptor(normalized_descriptor, entry.claim_descriptor) ||
        verified_content_descriptor != entry.verified_content_descriptor) {
      maybe_retire_backing_handle(body_handle, "conflict");
      return PutResult{.outcome = PutOutcome::kConflict};
    }
    if (ttl_ms.has_value() && *ttl_ms > 0) {
      const absl::Time new_expiry = now + absl::Milliseconds(*ttl_ms);
      extend_expiry_monotonic(new_expiry, &entry.expires_at);
    }

    bool should_rebind = entry.visibility_kind != AuthorityVisibilityKind::kReadyBacking ||
        entry.claim_state != AuthorityClaimState::kClaimedVisible || !entry.retained_backing_identity.has_value();
    if (!should_rebind) {
      auto backing_it = state_.backing_entries.find(*entry.retained_backing_identity);
      should_rebind = backing_it == state_.backing_entries.end() ||
          backing_it->second.lifecycle_state != BackingLifecycleState::kActive;
    }
    if (should_rebind) {
      install_or_rebind_backing_locked(
          artifact_id,
          &entry,
          descriptor,
          verified_content_descriptor,
          verification_record,
          backing_identity,
          observation,
          body_handle,
          &state_);
      return PutResult{.outcome = PutOutcome::kJoined};
    }

    const bool shares_existing_backing =
        entry.retained_backing_identity.has_value() && *entry.retained_backing_identity == backing_identity;
    if (!shares_existing_backing) {
      maybe_retire_backing_handle(body_handle, "join_duplicate");
    }
    return PutResult{.outcome = PutOutcome::kJoined};
  }

  const std::string key(artifact_id);
  auto entry = make_authority_entry(
      descriptor,
      verified_content_descriptor,
      verification_record,
      backing_identity,
      shard_id,
      lease_generation,
      routing_epoch,
      resolve_expiry(now, ttl_ms));
  state_.authority_entries[key] = entry;
  install_or_rebind_backing_locked(
      artifact_id,
      &state_.authority_entries[key],
      descriptor,
      verified_content_descriptor,
      verification_record,
      backing_identity,
      observation,
      body_handle,
      &state_);
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
  transition_to_invisible_locked(&entry_it->second, &state_);
  prune_orphan_backings_locked(&state_, reason);
}

void ByteArtifactBodyStore::invalidate_replica_visibility(
    const store::loading::ReplicaKey& replica_key,
    absl::Time now,
    std::string_view reason) {
  absl::MutexLock lock(&state_.mu);
  const auto index_it = state_.replica_visibility_index.find(replica_key);
  if (index_it == state_.replica_visibility_index.end()) {
    return;
  }
  const auto backing_identities =
      std::vector<store::runtime::ingestion::BackingIdentity>(index_it->second.begin(), index_it->second.end());
  for (const auto& identity : backing_identities) {
    auto backing_it = state_.backing_entries.find(identity);
    if (backing_it != state_.backing_entries.end()) {
      set_backing_lifecycle_state_locked(&backing_it->second, BackingLifecycleState::kInvalidated, reason);
    }
    auto authority_it = state_.backing_authority_index.find(identity);
    if (authority_it == state_.backing_authority_index.end()) {
      continue;
    }
    const auto artifact_ids = std::vector<std::string>(authority_it->second.begin(), authority_it->second.end());
    for (const auto& artifact_id : artifact_ids) {
      auto entry_it = state_.authority_entries.find(artifact_id);
      if (entry_it == state_.authority_entries.end()) {
        continue;
      }
      if (entry_it->second.expires_at != absl::InfiniteFuture() && now >= entry_it->second.expires_at) {
        erase_claim_locked(artifact_id, &state_, "expired");
        continue;
      }
      transition_to_invisible_locked(&entry_it->second, &state_);
    }
  }
  prune_orphan_backings_locked(&state_, reason);
}

} // namespace tensorcast::daemon
