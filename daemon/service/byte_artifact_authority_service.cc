// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/byte_artifact_authority_service.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "daemon/service/artifact_profile_registry.h"

namespace tensorcast::daemon {

namespace {

const ArtifactProfileRuntime& byte_artifact_runtime() {
  return ArtifactProfileRegistry::runtime_for_profile(ArtifactProfileRegistry::Profile::kByteArtifact);
}

v2::BatchItemOutcome make_outcome(
    std::string_view artifact_id,
    v2::BatchItemStatus status,
    std::string_view message = "") {
  v2::BatchItemOutcome outcome;
  outcome.set_artifact_id(std::string(artifact_id));
  outcome.set_status(status);
  if (!message.empty()) {
    outcome.set_message(std::string(message));
  }
  return outcome;
}

absl::Status validate_artifact_for_context(
    std::string_view artifact_id,
    const ByteArtifactAuthorityService::Context& context) {
  auto artifact_id_st = byte_artifact_runtime().validate_artifact_id_for_field(artifact_id, "artifact_id");
  if (!artifact_id_st.ok()) {
    return artifact_id_st;
  }
  auto shard_or = byte_artifact_runtime().shard_id_for_artifact(artifact_id, context.shard_count);
  if (!shard_or.ok()) {
    return shard_or.status();
  }
  if (*shard_or != context.shard_id) {
    return absl::InvalidArgumentError("artifact_id does not belong to fence.shard_id");
  }
  return absl::OkStatus();
}

void retire_put_item_backing(const ByteArtifactAuthorityService::PutItem& item) {
  (void)item.body_handle.retire();
}

} // namespace

ByteArtifactAuthorityService::ByteArtifactAuthorityService(ByteArtifactBodyStore& body_store)
    : body_store_(body_store) {}

std::vector<v2::BatchItemOutcome> ByteArtifactAuthorityService::batch_exists(
    const std::vector<std::string>& artifact_ids,
    const Context& context) const {
  std::vector<v2::BatchItemOutcome> outcomes;
  outcomes.reserve(artifact_ids.size());
  for (const auto& artifact_id : artifact_ids) {
    const auto artifact_st = validate_artifact_for_context(artifact_id, context);
    if (!artifact_st.ok()) {
      outcomes.push_back(make_outcome(
          artifact_id,
          absl::IsInvalidArgument(artifact_st) ? v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT
                                               : v2::BATCH_ITEM_STATUS_INTERNAL_ERROR,
          artifact_st.message()));
      continue;
    }
    if (body_store_.exists(
            artifact_id, context.shard_id, context.lease_generation, context.routing_epoch, context.now)) {
      outcomes.push_back(make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_OK));
    } else {
      outcomes.push_back(make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_MISS));
    }
  }
  return outcomes;
}

std::vector<ByteArtifactAuthorityService::GetResult> ByteArtifactAuthorityService::batch_get(
    const std::vector<std::string>& artifact_ids,
    const Context& context) const {
  std::vector<GetResult> results;
  results.reserve(artifact_ids.size());
  for (const auto& artifact_id : artifact_ids) {
    const auto artifact_st = validate_artifact_for_context(artifact_id, context);
    if (!artifact_st.ok()) {
      results.push_back(
          GetResult{
              .artifact_id = artifact_id,
              .status = absl::IsInvalidArgument(artifact_st) ? v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT
                                                             : v2::BATCH_ITEM_STATUS_INTERNAL_ERROR,
              .message = std::string(artifact_st.message()),
          });
      continue;
    }

    auto entry =
        body_store_.get(artifact_id, context.shard_id, context.lease_generation, context.routing_epoch, context.now);
    if (!entry.has_value()) {
      results.push_back(
          GetResult{
              .artifact_id = artifact_id,
              .status = v2::BATCH_ITEM_STATUS_MISS,
          });
      continue;
    }

    results.push_back(
        GetResult{
            .artifact_id = artifact_id,
            .status = v2::BATCH_ITEM_STATUS_OK,
            .descriptor = entry->descriptor,
            .body_handle = entry->body_handle,
            .authority_record = entry->authority_record,
            .serving_capability = entry->serving_capability,
        });
  }
  return results;
}

std::vector<v2::BatchItemOutcome> ByteArtifactAuthorityService::batch_put_if_absent(
    const std::vector<PutItem>& items,
    const Context& context,
    const std::optional<std::uint64_t>& ttl_ms) const {
  std::vector<v2::BatchItemOutcome> outcomes;
  outcomes.reserve(items.size());
  for (const auto& item : items) {
    const auto artifact_st = validate_artifact_for_context(item.artifact_id, context);
    if (!artifact_st.ok()) {
      retire_put_item_backing(item);
      outcomes.push_back(make_outcome(
          item.artifact_id,
          absl::IsInvalidArgument(artifact_st) ? v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT
                                               : v2::BATCH_ITEM_STATUS_INTERNAL_ERROR,
          artifact_st.message()));
      continue;
    }

    const auto invariant_st =
        byte_artifact_runtime().validate_invariant_body_descriptor(item.invariant, item.descriptor);
    if (!invariant_st.ok()) {
      retire_put_item_backing(item);
      outcomes.push_back(
          make_outcome(item.artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, invariant_st.message()));
      continue;
    }

    const auto put_result = body_store_.put_if_absent(
        item.artifact_id,
        item.invariant,
        item.descriptor,
        item.verified_content_descriptor,
        item.verification_record,
        item.observation,
        item.body_handle,
        context.shard_id,
        context.lease_generation,
        context.routing_epoch,
        context.now,
        ttl_ms);
    switch (put_result.outcome) {
      case ByteArtifactBodyStore::PutOutcome::kCreated:
        outcomes.push_back(make_outcome(item.artifact_id, v2::BATCH_ITEM_STATUS_OK, "created"));
        break;
      case ByteArtifactBodyStore::PutOutcome::kJoined:
        outcomes.push_back(make_outcome(item.artifact_id, v2::BATCH_ITEM_STATUS_OK, "joined"));
        break;
      case ByteArtifactBodyStore::PutOutcome::kConflict:
        outcomes.push_back(make_outcome(
            item.artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, "put_if_absent invariant mismatch"));
        break;
    }
  }
  return outcomes;
}

std::vector<v2::BatchItemOutcome> ByteArtifactAuthorityService::batch_touch_ttl(
    const std::vector<std::string>& artifact_ids,
    const Context& context,
    std::uint64_t ttl_ms) const {
  std::vector<v2::BatchItemOutcome> outcomes;
  outcomes.reserve(artifact_ids.size());
  for (const auto& artifact_id : artifact_ids) {
    const auto artifact_st = validate_artifact_for_context(artifact_id, context);
    if (!artifact_st.ok()) {
      outcomes.push_back(make_outcome(
          artifact_id,
          absl::IsInvalidArgument(artifact_st) ? v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT
                                               : v2::BATCH_ITEM_STATUS_INTERNAL_ERROR,
          artifact_st.message()));
      continue;
    }

    if (body_store_.touch_ttl(
            artifact_id, context.shard_id, context.lease_generation, context.routing_epoch, context.now, ttl_ms)) {
      outcomes.push_back(make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_OK));
    } else {
      outcomes.push_back(make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_MISS));
    }
  }
  return outcomes;
}

} // namespace tensorcast::daemon
