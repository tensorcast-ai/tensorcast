// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/byte_artifact_controller.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "core/store/materialization/dataplane/contracts/inline_buffer_loader.h"
#include "core/store/materialization/dataplane/loaders/disk_loader.h"
#include "core/store/runtime/ingestion/artifact_lowering_plan.h"
#include "daemon/service/artifact_profile_registry.h"
#include "daemon/util/grpc_peer_utils.h"
#include "daemon/util/status_utils.h"
#include "grpcpp/grpcpp.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

const ArtifactProfileRuntime& byte_artifact_runtime() {
  return ArtifactProfileRegistry::runtime_for_profile(ArtifactProfileRegistry::Profile::kByteArtifact);
}

absl::Status validate_batch_selection(const tensorcast::common::v1::ArtifactSelection& selection) {
  return byte_artifact_runtime().validate_batch_selection(selection);
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

v2::HomeBatchGetItem make_home_get_item(
    std::string_view artifact_id,
    v2::BatchItemStatus status,
    std::string_view message = "") {
  v2::HomeBatchGetItem item;
  item.set_artifact_id(std::string(artifact_id));
  item.set_status(status);
  if (!message.empty()) {
    item.set_message(std::string(message));
  }
  return item;
}

v2::BatchItemStatus batch_item_status_from_absl_status(const absl::Status& status) {
  if (status.ok()) {
    return v2::BATCH_ITEM_STATUS_OK;
  }
  if (absl::IsInvalidArgument(status)) {
    return v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT;
  }
  if (absl::IsFailedPrecondition(status) || absl::IsPermissionDenied(status)) {
    return v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION;
  }
  if (absl::IsNotFound(status)) {
    return v2::BATCH_ITEM_STATUS_MISS;
  }
  if (absl::IsUnavailable(status) || absl::IsDeadlineExceeded(status)) {
    return v2::BATCH_ITEM_STATUS_UNAVAILABLE;
  }
  return v2::BATCH_ITEM_STATUS_INTERNAL_ERROR;
}

bool is_non_actionable_policy_path_error(const absl::Status& status) {
  return absl::IsNotFound(status) || absl::IsFailedPrecondition(status) || absl::IsDataLoss(status) ||
      absl::IsInvalidArgument(status);
}

PolicyVisibilityPathKind policy_visibility_path_kind_from_source(PersistenceManager::PolicySourceKind kind) {
  switch (kind) {
    case PersistenceManager::PolicySourceKind::kSharedDisk:
      return PolicyVisibilityPathKind::kSharedDisk;
    case PersistenceManager::PolicySourceKind::kUnspecified:
    default:
      return PolicyVisibilityPathKind::kUnspecified;
  }
}

class SeekableSourceLoader final : public store::IArtifactLoader {
 public:
  SeekableSourceLoader(std::shared_ptr<store::loader::SeekableSource> source, std::uint64_t size_bytes)
      : source_(std::move(source)), size_bytes_(size_bytes) {}

  absl::Status initialize() override {
    initialized_ = true;
    return absl::OkStatus();
  }

  absl::StatusOr<std::uint64_t> get_artifact_size() override {
    if (!initialized_) {
      return absl::FailedPreconditionError("SeekableSourceLoader not initialized");
    }
    return size_bytes_;
  }

  absl::StatusOr<std::unique_ptr<store::loader::SeekableSource>> open_source() override {
    if (!initialized_) {
      return absl::FailedPreconditionError("SeekableSourceLoader not initialized");
    }
    if (!source_) {
      return absl::FailedPreconditionError("SeekableSourceLoader requires source");
    }

    class SourceRef final : public store::loader::SeekableSource {
     public:
      explicit SourceRef(std::shared_ptr<store::loader::SeekableSource> source) : source_(std::move(source)) {}

      [[nodiscard]] uint64_t total_bytes() const override {
        return source_->total_bytes();
      }

      absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
        return source_->read(dst, max_bytes);
      }

      absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
        return source_->read_at(offset, dst, bytes);
      }

      [[nodiscard]] bool supports_direct_write_at() const override {
        return source_->supports_direct_write_at();
      }

      absl::StatusOr<size_t> read_into_at(
          uint64_t src_offset,
          uint64_t dest_va_offset,
          size_t bytes,
          const store::DirectWriteGrant& grant) override {
        return source_->read_into_at(src_offset, dest_va_offset, bytes, grant);
      }

     private:
      std::shared_ptr<store::loader::SeekableSource> source_;
    };

    return std::unique_ptr<store::loader::SeekableSource>(std::make_unique<SourceRef>(source_));
  }

 private:
  bool initialized_{false};
  std::shared_ptr<store::loader::SeekableSource> source_;
  std::uint64_t size_bytes_{0};
};

store::loading::MaterializeHints build_lowering_hints(std::string_view artifact_id, std::string_view operation_id) {
  store::loading::MaterializeHints hints;
  hints.artifact_id = std::string(artifact_id);
  if (!operation_id.empty()) {
    hints.transport_request_id = std::string(operation_id);
  }
  return hints;
}

absl::StatusOr<store::runtime::ingestion::ArtifactLoweringPlan> build_into_target_lowering_plan(
    std::string_view artifact_id,
    const store::DeviceKey& target_device,
    const store::loading::IntoTargetLayout& target_layout,
    std::unique_ptr<store::IArtifactLoader> loader,
    std::uint64_t payload_bytes,
    store::loading::MaterializationSource source_kind,
    std::string_view operation_id) {
  if (loader == nullptr) {
    return absl::InvalidArgumentError("into-target lowering requires loader");
  }
  return store::runtime::ingestion::lower_to_artifact_plan(
      store::runtime::ingestion::LowerToArtifactPlanRequest{
          .identity =
              store::runtime::ingestion::ArtifactLoweringIdentity{
                  .logical_artifact_id = std::string(artifact_id),
                  .request_id = std::string(operation_id),
              },
          .target_device = target_device,
          .source_loader = std::move(loader),
          .selection_identity =
              tensorcast::common::SelectionIdentity{
                  .artifact_id = std::string(artifact_id),
                  .logical_layout_hash = tensorcast::common::compute_byte_artifact_logical_layout_hash_bytes(),
                  .selection_hash = tensorcast::common::compute_byte_artifact_selection_hash_bytes(),
              },
          .expected_size_bytes = payload_bytes,
          .generation = 1,
          .hints = build_lowering_hints(artifact_id, operation_id),
          .source_kind = source_kind,
          .into_target = target_layout,
      });
}

} // namespace

ByteArtifactController::ByteArtifactController(Dep d, Options options)
    : d_(std::move(d)),
      authority_service_(d_.body_store),
      body_backing_manager_(d_.engine),
      options_(std::move(options)) {}

void ByteArtifactController::reconcile_policy_visibility(
    const std::vector<std::string>& artifact_ids,
    const ByteArtifactAuthorityService::Context& context) const {
  for (const auto& artifact_id : artifact_ids) {
    auto authority = d_.body_store.inspect_authority(
        artifact_id, context.shard_id, context.lease_generation, context.routing_epoch, context.now);
    if (!authority.has_value()) {
      continue;
    }
    if (authority->authority_record.claim_state == AuthorityClaimState::kClaimDeleted ||
        authority->authority_record.claim_state == AuthorityClaimState::kUnclaimed) {
      continue;
    }
    if (authority->authority_record.visibility_kind == AuthorityVisibilityKind::kReadyBacking &&
        authority->authority_record.claim_state == AuthorityClaimState::kClaimedVisible) {
      continue;
    }

    auto desired_ref = resolve_policy_visibility_ref(*authority);
    if (desired_ref.has_value()) {
      (void)d_.body_store.install_policy_visibility(
          artifact_id, context.shard_id, context.lease_generation, context.routing_epoch, context.now, *desired_ref);
      continue;
    }
    (void)d_.body_store.clear_policy_visibility(
        artifact_id,
        context.shard_id,
        context.lease_generation,
        context.routing_epoch,
        context.now,
        "policy_path_not_actionable");
  }
}

std::optional<PolicyVisibilityRef> ByteArtifactController::resolve_policy_visibility_ref(
    const ByteArtifactBodyStore::AuthoritySnapshot& authority_snapshot) const {
  if (d_.persistence_manager == nullptr) {
    return std::nullopt;
  }
  auto policy_source = d_.persistence_manager->resolve_policy_source(authority_snapshot.authority_record.artifact_id);
  if (!policy_source.has_value()) {
    return std::nullopt;
  }
  if (policy_source->path_kind != PersistenceManager::PolicySourceKind::kSharedDisk) {
    return std::nullopt;
  }
  if (!policy_source->verified_content_descriptor.has_value() ||
      *policy_source->verified_content_descriptor != authority_snapshot.verified_content_descriptor) {
    return std::nullopt;
  }
  return PolicyVisibilityRef{
      .path_id = policy_source->path_id,
      .path_kind = policy_visibility_path_kind_from_source(policy_source->path_kind),
      .verified_content_descriptor = authority_snapshot.verified_content_descriptor,
      .control_ref = policy_source->control_ref,
      .expires_at = authority_snapshot.expires_at,
  };
}

absl::StatusOr<ByteArtifactBodyStore::EntrySnapshot> ByteArtifactController::restore_backing_from_policy_visibility(
    std::string_view artifact_id,
    const BodyDescriptor& descriptor,
    const AuthorityRecord& authority_record,
    const ByteArtifactAuthorityService::Context& context,
    std::string_view operation_id) const {
  if (d_.persistence_manager == nullptr || !authority_record.policy_visibility_ref.has_value()) {
    return absl::FailedPreconditionError("policy-backed visibility requires persistence proof");
  }
  if (authority_record.policy_visibility_ref->path_kind != PolicyVisibilityPathKind::kSharedDisk) {
    return absl::FailedPreconditionError("unsupported policy-backed path kind");
  }
  auto policy_source =
      d_.persistence_manager->resolve_policy_source(artifact_id, authority_record.policy_visibility_ref->control_ref);
  if (!policy_source.has_value()) {
    (void)d_.body_store.clear_policy_visibility(
        artifact_id,
        context.shard_id,
        context.lease_generation,
        context.routing_epoch,
        context.now,
        "policy_path_not_found");
    return absl::NotFoundError("policy-backed shared-disk path is no longer actionable");
  }
  if (policy_source->path_kind != PersistenceManager::PolicySourceKind::kSharedDisk) {
    return absl::FailedPreconditionError("policy-backed source kind mismatch");
  }

  auto staged_body_or = body_backing_manager_.stage_body(
      BodyBackingManager::StageRequest{
          .artifact_id = std::string(artifact_id),
          .invariant = body_descriptor_to_invariant(descriptor),
          .loader = std::make_unique<store::DiskLoader>(store::loading::DiskSource{
              .path = policy_source->local_path,
              .expected_size = descriptor.size_bytes,
              .require_descriptor = true,
          }),
          .source_kind = store::loading::MaterializationSource::kDisk,
          .operation_id = std::string(operation_id),
          .access_class = BodyAccessClass::kHomeDefault,
          .route_role = BodyRouteRole::kHomeAuthority,
      });
  if (!staged_body_or.ok()) {
    if (is_non_actionable_policy_path_error(staged_body_or.status())) {
      (void)d_.body_store.clear_policy_visibility(
          artifact_id,
          context.shard_id,
          context.lease_generation,
          context.routing_epoch,
          context.now,
          "policy_materialization_failed");
    }
    return staged_body_or.status();
  }

  auto put_result = d_.body_store.put_if_absent(
      artifact_id,
      body_descriptor_to_invariant(descriptor),
      staged_body_or->descriptor,
      staged_body_or->verified_content_descriptor,
      staged_body_or->verification_record,
      staged_body_or->backing_identity,
      staged_body_or->observation,
      staged_body_or->body_handle,
      context.shard_id,
      context.lease_generation,
      context.routing_epoch,
      context.now,
      std::nullopt);
  if (put_result.outcome == ByteArtifactBodyStore::PutOutcome::kConflict) {
    (void)staged_body_or->body_handle.retire();
    return absl::FailedPreconditionError("policy-backed restore conflicted with current claim descriptor");
  }

  auto entry =
      d_.body_store.get(artifact_id, context.shard_id, context.lease_generation, context.routing_epoch, context.now);
  if (!entry.has_value()) {
    return absl::InternalError("policy-backed restore did not produce a visible backing");
  }
  return *entry;
}

grpc::Status ByteArtifactController::home_batch_exists(
    RpcContext& rctx,
    const v2::HomeBatchExistsRequest& req,
    v2::HomeBatchExistsResponse& resp) {
  const absl::Time now = absl::Now();
  auto home_lease_or = d_.route_resolver.ensure_home_lease(req.fence(), now);
  if (!home_lease_or.ok()) {
    return to_grpc_status(home_lease_or.status());
  }
  if (home_lease_or->kind != ByteArtifactRouteResolver::HomeLeaseDecision::Kind::kOwned) {
    if (home_lease_or->redirect.lease_generation() != 0 && !home_lease_or->redirect.holder_daemon_id().empty()) {
      resp.mutable_redirect()->CopyFrom(home_lease_or->redirect);
    }
    for (const auto& artifact_id : req.artifact_ids()) {
      auto* outcome = resp.add_outcomes();
      *outcome = make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, home_lease_or->message);
    }
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::string> artifact_ids;
  artifact_ids.reserve(req.artifact_ids_size());
  for (const auto& artifact_id : req.artifact_ids()) {
    artifact_ids.push_back(artifact_id);
  }
  ByteArtifactAuthorityService::Context authority_context{
      .shard_id = req.fence().shard_id(),
      .lease_generation = home_lease_or->lease_generation,
      .routing_epoch = options_.routing.routing_epoch,
      .shard_count = options_.routing.shard_count,
      .now = now,
  };
  reconcile_policy_visibility(artifact_ids, authority_context);
  for (auto outcome : authority_service_.batch_exists(artifact_ids, authority_context)) {
    *resp.add_outcomes() = std::move(outcome);
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::home_batch_get(
    RpcContext& rctx,
    const v2::HomeBatchGetRequest& req,
    v2::HomeBatchGetResponse& resp) {
  const absl::Time now = absl::Now();
  auto home_lease_or = d_.route_resolver.ensure_home_lease(req.fence(), now);
  if (!home_lease_or.ok()) {
    return to_grpc_status(home_lease_or.status());
  }
  if (home_lease_or->kind != ByteArtifactRouteResolver::HomeLeaseDecision::Kind::kOwned) {
    if (home_lease_or->redirect.lease_generation() != 0 && !home_lease_or->redirect.holder_daemon_id().empty()) {
      resp.mutable_redirect()->CopyFrom(home_lease_or->redirect);
    }
    for (const auto& artifact_id : req.artifact_ids()) {
      auto* item = resp.add_items();
      *item = make_home_get_item(artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, home_lease_or->message);
    }
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::string> artifact_ids;
  artifact_ids.reserve(req.artifact_ids_size());
  for (const auto& artifact_id : req.artifact_ids()) {
    artifact_ids.push_back(artifact_id);
  }
  ByteArtifactAuthorityService::Context authority_context{
      .shard_id = req.fence().shard_id(),
      .lease_generation = home_lease_or->lease_generation,
      .routing_epoch = options_.routing.routing_epoch,
      .shard_count = options_.routing.shard_count,
      .now = now,
  };
  reconcile_policy_visibility(artifact_ids, authority_context);
  for (auto& result : authority_service_.batch_get(artifact_ids, authority_context)) {
    auto* item = resp.add_items();
    item->set_artifact_id(result.artifact_id);
    item->set_status(result.status);
    if (!result.message.empty()) {
      item->set_message(result.message);
    }
    if (result.status != v2::BATCH_ITEM_STATUS_OK) {
      continue;
    }
    BodyHandle body_handle = result.body_handle;
    if (body_handle.empty() && result.authority_record.visibility_kind == AuthorityVisibilityKind::kPolicyBackedPath) {
      auto restored_or = restore_backing_from_policy_visibility(
          result.artifact_id,
          result.descriptor,
          result.authority_record,
          authority_context,
          req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view(""));
      if (!restored_or.ok()) {
        if (is_non_actionable_policy_path_error(restored_or.status())) {
          *item = make_home_get_item(result.artifact_id, v2::BATCH_ITEM_STATUS_MISS, "artifact is no longer visible");
        } else {
          *item = make_home_get_item(
              result.artifact_id,
              batch_item_status_from_absl_status(restored_or.status()),
              restored_or.status().message());
        }
        continue;
      }
      body_handle = restored_or->backing_record.retained_body_handle;
      result.descriptor = restored_or->descriptor;
    }
    auto payload_or = body_handle.read_all_bytes();
    if (!payload_or.ok()) {
      d_.body_store.invalidate_artifact_visibility(result.artifact_id, now, "serve_read_failed");
      *item = make_home_get_item(result.artifact_id, v2::BATCH_ITEM_STATUS_MISS, "artifact is no longer visible");
      continue;
    }
    auto payload = std::make_shared<const std::string>(std::move(*payload_or));
    if (payload->size() > options_.routing.inline_payload_threshold_bytes) {
      auto payload_ref_or = d_.payload_transport_broker.issue_payload_ref(
          result.artifact_id,
          payload,
          result.descriptor,
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
          req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view(""),
          result.serving_capability.expires_at);
      if (!payload_ref_or.ok()) {
        *item = make_home_get_item(
            result.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, payload_ref_or.status().message());
        continue;
      }
      item->set_payload_ref(*payload_ref_or);
      continue;
    }
    item->set_inline_payload(*payload);
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::home_batch_put_if_absent(
    RpcContext& rctx,
    const v2::HomeBatchPutIfAbsentRequest& req,
    v2::HomeBatchPutIfAbsentResponse& resp) {
  const absl::Time now = absl::Now();
  const std::optional<std::uint64_t> ttl_ms =
      req.has_ttl_ms() ? std::optional<std::uint64_t>(req.ttl_ms()) : std::nullopt;
  const std::string& local_daemon_id = d_.route_resolver.local_daemon_id();
  auto home_lease_or = d_.route_resolver.ensure_home_lease(req.fence(), now);
  if (!home_lease_or.ok()) {
    return to_grpc_status(home_lease_or.status());
  }
  if (home_lease_or->kind != ByteArtifactRouteResolver::HomeLeaseDecision::Kind::kOwned) {
    if (home_lease_or->redirect.lease_generation() != 0 && !home_lease_or->redirect.holder_daemon_id().empty()) {
      resp.mutable_redirect()->CopyFrom(home_lease_or->redirect);
    }
    for (const auto& item : req.items()) {
      auto* outcome = resp.add_outcomes();
      *outcome = make_outcome(item.artifact_id(), v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, home_lease_or->message);
    }
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::optional<v2::BatchItemOutcome>> deferred_outcomes(req.items_size());
  std::vector<ByteArtifactAuthorityService::PutItem> authority_items;
  std::vector<int> authority_item_indices;
  authority_items.reserve(req.items_size());
  authority_item_indices.reserve(req.items_size());

  for (int index = 0; index < req.items_size(); ++index) {
    const auto& item = req.items(index);
    const std::string artifact_id = item.artifact_id();
    if (!item.inline_payload().empty() && !item.payload_ref().empty()) {
      deferred_outcomes[index] = make_outcome(
          artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, "inline_payload and payload_ref are mutually exclusive");
      continue;
    }

    std::unique_ptr<store::IArtifactLoader> loader;
    store::loading::MaterializationSource source_kind = store::loading::MaterializationSource::kLocalReplica;
    std::optional<BodyBackingManager::StageResult> staged_body;
    if (!item.inline_payload().empty()) {
      auto payload = std::make_shared<std::string>(item.inline_payload());
      loader = std::make_unique<store::InlineBufferLoader>(store::loading::InlineBufferSource{
          .data = std::shared_ptr<const void>(payload, static_cast<const void*>(payload->data())),
          .size_bytes = payload->size(),
      });
    } else if (!item.payload_ref().empty()) {
      auto capability_or = d_.payload_transport_broker.resolve_payload_ref_capability(
          item.payload_ref(),
          artifact_id,
          now,
          local_daemon_id,
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
          req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view(""));
      if (!capability_or.ok()) {
        deferred_outcomes[index] = make_outcome(
            artifact_id,
            batch_item_status_from_absl_status(capability_or.status()),
            std::string(capability_or.status().message()));
        continue;
      }
      if (capability_or->capability.mode == BodyCapabilityResolutionMode::kLocalBodyHandle) {
        auto reused_or = body_backing_manager_.try_reuse_body(
            BodyBackingManager::ReuseRequest{
                .artifact_id = artifact_id,
                .invariant = item.invariant(),
                .descriptor = capability_or->capability.descriptor,
                .body_handle = capability_or->capability.body_handle,
                .operation_id = req.has_operation_id() ? req.operation_id() : "",
                .access_class = BodyAccessClass::kHomeDefault,
                .route_role = BodyRouteRole::kHomeAuthority,
            });
        if (!reused_or.ok()) {
          deferred_outcomes[index] = make_outcome(
              artifact_id,
              batch_item_status_from_absl_status(reused_or.status()),
              std::string(reused_or.status().message()));
          continue;
        }
        if (reused_or->has_value()) {
          staged_body = std::move(**reused_or);
        } else {
          auto loader_or = capability_or->capability.body_handle.make_loader();
          if (!loader_or.ok()) {
            deferred_outcomes[index] = make_outcome(
                artifact_id,
                batch_item_status_from_absl_status(loader_or.status()),
                std::string(loader_or.status().message()));
            continue;
          }
          loader = std::move(*loader_or);
        }
      } else if (capability_or->capability.local) {
        if (!capability_or->payload) {
          deferred_outcomes[index] = make_outcome(
              artifact_id,
              v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION,
              "payload_ref resolved locally without payload bytes");
          continue;
        }
        loader = std::make_unique<store::InlineBufferLoader>(store::loading::InlineBufferSource{
            .data = std::shared_ptr<const void>(
                capability_or->payload, static_cast<const void*>(capability_or->payload->data())),
            .size_bytes = capability_or->payload->size(),
        });
      } else {
        auto loader_or = d_.payload_transport_broker.open_payload_ref_loader(
            d_.worker_directory_cache,
            now,
            absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()),
            local_daemon_id,
            item.payload_ref(),
            artifact_id,
            tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
            req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view(""));
        if (!loader_or.ok()) {
          deferred_outcomes[index] = make_outcome(
              artifact_id,
              batch_item_status_from_absl_status(loader_or.status()),
              std::string(loader_or.status().message()));
          continue;
        }
        source_kind = loader_or->remote ? store::loading::MaterializationSource::kP2P
                                        : store::loading::MaterializationSource::kLocalReplica;
        loader = std::move(loader_or->loader);
      }
    } else {
      deferred_outcomes[index] = make_outcome(
          artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, "inline_payload or payload_ref is required");
      continue;
    }

    if (!staged_body.has_value()) {
      auto staged_body_or = body_backing_manager_.stage_body(
          BodyBackingManager::StageRequest{
              .artifact_id = artifact_id,
              .invariant = item.invariant(),
              .loader = std::move(loader),
              .source_kind = source_kind,
              .operation_id = req.has_operation_id() ? req.operation_id() : "",
              .access_class = BodyAccessClass::kHomeDefault,
              .route_role = BodyRouteRole::kHomeAuthority,
          });
      if (!staged_body_or.ok()) {
        deferred_outcomes[index] = make_outcome(
            artifact_id,
            batch_item_status_from_absl_status(staged_body_or.status()),
            std::string(staged_body_or.status().message()));
        continue;
      }
      staged_body = std::move(*staged_body_or);
    }

    const auto invariant_st =
        byte_artifact_runtime().validate_invariant_body_descriptor(item.invariant(), staged_body->descriptor);
    if (!invariant_st.ok()) {
      auto retire_status = staged_body->body_handle.retire();
      if (!retire_status.ok()) {
        deferred_outcomes[index] =
            make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, std::string(retire_status.message()));
        continue;
      }
      deferred_outcomes[index] =
          make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, std::string(invariant_st.message()));
      continue;
    }

    authority_items.push_back(
        ByteArtifactAuthorityService::PutItem{
            .artifact_id = artifact_id,
            .invariant = item.invariant(),
            .descriptor = staged_body->descriptor,
            .verified_content_descriptor = staged_body->verified_content_descriptor,
            .verification_record = staged_body->verification_record,
            .backing_identity = staged_body->backing_identity,
            .observation = staged_body->observation,
            .body_handle = staged_body->body_handle,
        });
    authority_item_indices.push_back(index);
  }

  const auto authority_outcomes = authority_service_.batch_put_if_absent(
      authority_items,
      ByteArtifactAuthorityService::Context{
          .shard_id = req.fence().shard_id(),
          .lease_generation = home_lease_or->lease_generation,
          .routing_epoch = options_.routing.routing_epoch,
          .shard_count = options_.routing.shard_count,
          .now = now,
      },
      ttl_ms);
  for (size_t index = 0; index < authority_outcomes.size(); ++index) {
    deferred_outcomes[authority_item_indices[index]] = authority_outcomes[index];
  }

  for (int index = 0; index < req.items_size(); ++index) {
    if (deferred_outcomes[index].has_value()) {
      *resp.add_outcomes() = std::move(*deferred_outcomes[index]);
      continue;
    }
    *resp.add_outcomes() =
        make_outcome(req.items(index).artifact_id(), v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "missing authority outcome");
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::home_batch_touch_ttl(
    RpcContext& rctx,
    const v2::HomeBatchTouchTtlRequest& req,
    v2::HomeBatchTouchTtlResponse& resp) {
  if (req.ttl_ms() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "ttl_ms must be > 0"};
  }
  const absl::Time now = absl::Now();
  auto home_lease_or = d_.route_resolver.ensure_home_lease(req.fence(), now);
  if (!home_lease_or.ok()) {
    return to_grpc_status(home_lease_or.status());
  }
  if (home_lease_or->kind != ByteArtifactRouteResolver::HomeLeaseDecision::Kind::kOwned) {
    if (home_lease_or->redirect.lease_generation() != 0 && !home_lease_or->redirect.holder_daemon_id().empty()) {
      resp.mutable_redirect()->CopyFrom(home_lease_or->redirect);
    }
    for (const auto& artifact_id : req.artifact_ids()) {
      auto* outcome = resp.add_outcomes();
      *outcome = make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, home_lease_or->message);
    }
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::string> artifact_ids;
  artifact_ids.reserve(req.artifact_ids_size());
  for (const auto& artifact_id : req.artifact_ids()) {
    artifact_ids.push_back(artifact_id);
  }
  for (auto outcome : authority_service_.batch_touch_ttl(
           artifact_ids,
           ByteArtifactAuthorityService::Context{
               .shard_id = req.fence().shard_id(),
               .lease_generation = home_lease_or->lease_generation,
               .routing_epoch = options_.routing.routing_epoch,
               .shard_count = options_.routing.shard_count,
               .now = now,
           },
           req.ttl_ms())) {
    *resp.add_outcomes() = std::move(outcome);
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::batch_exists(
    RpcContext& rctx,
    const v2::BatchExistsRequest& req,
    v2::BatchExistsResponse& resp) {
  const bool local_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  if (!local_peer && !options_.gateway_ingress_enabled) {
    return {StatusCode::PERMISSION_DENIED, "BatchExists requires gateway ingress on non-local peers"};
  }

  const absl::Time now = absl::Now();
  const std::string local_daemon_id = d_.route_resolver.local_daemon_id();

  struct ShardRequest {
    std::vector<std::string> artifact_ids;
    std::vector<int> outcome_indices;
  };

  absl::flat_hash_map<std::uint64_t, ShardRequest> shard_requests;
  shard_requests.reserve(static_cast<size_t>(req.selections_size()));

  for (int i = 0; i < req.selections_size(); ++i) {
    const auto& selection = req.selections(i);
    auto* outcome = resp.add_outcomes();
    outcome->set_artifact_id(selection.artifact_id());

    const auto selection_st = validate_batch_selection(selection);
    if (!selection_st.ok()) {
      *outcome = make_outcome(
          selection.artifact_id(), v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, std::string(selection_st.message()));
      continue;
    }
    auto shard_or =
        byte_artifact_runtime().shard_id_for_artifact(selection.artifact_id(), options_.routing.shard_count);
    if (!shard_or.ok()) {
      *outcome = make_outcome(
          selection.artifact_id(), v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, std::string(shard_or.status().message()));
      continue;
    }

    auto& entry = shard_requests[*shard_or];
    entry.artifact_ids.push_back(selection.artifact_id());
    entry.outcome_indices.push_back(i);
  }

  const bool local_only = (d_.global_store_client == nullptr);
  if (local_only) {
    for (const auto& [shard_id, batch] : shard_requests) {
      const auto outcomes = authority_service_.batch_exists(
          batch.artifact_ids,
          ByteArtifactAuthorityService::Context{
              .shard_id = shard_id,
              .lease_generation = 1,
              .routing_epoch = options_.routing.routing_epoch,
              .shard_count = options_.routing.shard_count,
              .now = now,
          });
      for (size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
        *resp.mutable_outcomes(batch.outcome_indices[idx]) = outcomes.at(idx);
      }
    }
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::uint64_t> shard_ids;
  shard_ids.reserve(shard_requests.size());
  for (const auto& [shard_id, /*batch*/ _] : shard_requests) {
    shard_ids.push_back(shard_id);
  }
  auto routes = d_.route_resolver.resolve_routes(absl::MakeSpan(shard_ids), now);

  std::vector<std::string> remote_daemon_ids;
  remote_daemon_ids.reserve(routes.size());
  for (const auto& [shard_id, route] : routes) {
    if (!route.ok || route.holder_daemon_id == local_daemon_id) {
      continue;
    }
    remote_daemon_ids.push_back(route.holder_daemon_id);
  }
  if (!remote_daemon_ids.empty()) {
    (void)d_.worker_directory_cache.warm_for_daemons(
        remote_daemon_ids, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
  }

  for (const auto& [shard_id, batch] : shard_requests) {
    const auto route_it = routes.find(shard_id);
    if (route_it == routes.end() || !route_it->second.ok) {
      const std::string message = (route_it == routes.end() || route_it->second.message.empty())
          ? "routing lease unavailable"
          : route_it->second.message;
      for (size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
        *resp.mutable_outcomes(batch.outcome_indices[idx]) =
            make_outcome(batch.artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, message);
      }
      continue;
    }

    std::uint64_t lease_generation = route_it->second.lease_generation;
    std::string holder_daemon_id = route_it->second.holder_daemon_id;

    for (int attempt = 0; attempt < 2; ++attempt) {
      v2::HomeBatchExistsResponse home_resp;
      grpc::Status home_status;

      if (holder_daemon_id == local_daemon_id) {
        v2::HomeBatchExistsRequest home_req;
        home_req.mutable_fence()->set_shard_id(shard_id);
        home_req.mutable_fence()->set_lease_generation(lease_generation);
        home_req.mutable_fence()->set_holder_daemon_id(local_daemon_id);
        home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
        for (const auto& artifact_id : batch.artifact_ids) {
          home_req.add_artifact_ids(artifact_id);
        }
        grpc::ServerContext home_ctx;
        RpcContext home_rctx{"HomeBatchExists", home_ctx, rctx.allow_high_card_attrs()};
        home_status = home_batch_exists(home_rctx, home_req, home_resp);
      } else {
        auto address_or = d_.worker_directory_cache.resolve_daemon_address(
            holder_daemon_id, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
        if (!address_or.ok()) {
          for (size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
            *resp.mutable_outcomes(batch.outcome_indices[idx]) = make_outcome(
                batch.artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, "home daemon address unavailable");
          }
          break;
        }
        auto channel = grpc::CreateChannel(*address_or, grpc::InsecureChannelCredentials());
        auto stub = v2::StoreDaemonService::NewStub(channel);
        grpc::ClientContext client_ctx;
        client_ctx.set_deadline(std::chrono::system_clock::now() + options_.routing.route_staleness_budget);
        v2::HomeBatchExistsRequest home_req;
        home_req.mutable_fence()->set_shard_id(shard_id);
        home_req.mutable_fence()->set_lease_generation(lease_generation);
        home_req.mutable_fence()->set_holder_daemon_id(holder_daemon_id);
        home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
        for (const auto& artifact_id : batch.artifact_ids) {
          home_req.add_artifact_ids(artifact_id);
        }
        home_status = stub->HomeBatchExists(&client_ctx, home_req, &home_resp);
      }

      if (!home_status.ok()) {
        for (size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
          *resp.mutable_outcomes(batch.outcome_indices[idx]) = make_outcome(
              batch.artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, std::string(home_status.error_message()));
        }
        break;
      }

      bool needs_redirect_retry = false;
      if (home_resp.has_redirect() && home_resp.redirect().shard_id() == shard_id &&
          home_resp.redirect().lease_generation() != 0 && !home_resp.redirect().holder_daemon_id().empty()) {
        for (const auto& outcome : home_resp.outcomes()) {
          if (outcome.status() == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
            needs_redirect_retry = true;
            break;
          }
        }
      }
      if (needs_redirect_retry && attempt == 0) {
        const auto& redirect = home_resp.redirect();
        const auto refreshed = d_.route_resolver.refresh_route_from_redirect(shard_id, redirect, now);
        if (!refreshed.ok) {
          for (size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
            *resp.mutable_outcomes(batch.outcome_indices[idx]) =
                make_outcome(batch.artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, refreshed.message);
          }
          break;
        }
        holder_daemon_id = refreshed.holder_daemon_id;
        lease_generation = refreshed.lease_generation;
        continue;
      }

      absl::flat_hash_map<std::string, int> index_by_artifact;
      index_by_artifact.reserve(batch.artifact_ids.size());
      for (size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
        index_by_artifact.emplace(batch.artifact_ids[idx], static_cast<int>(idx));
      }
      for (const auto& item : home_resp.outcomes()) {
        const auto idx_it = index_by_artifact.find(item.artifact_id());
        if (idx_it == index_by_artifact.end()) {
          continue;
        }
        v2::BatchItemStatus status = item.status();
        if (status == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
          status = v2::BATCH_ITEM_STATUS_UNAVAILABLE;
        }
        *resp.mutable_outcomes(batch.outcome_indices[static_cast<size_t>(idx_it->second)]) =
            make_outcome(item.artifact_id(), status, item.message());
      }
      break;
    }
  }

  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::batch_get_into_region(
    RpcContext& rctx,
    const v2::BatchGetIntoRegionRequest& req,
    v2::BatchGetIntoRegionResponse& resp) {
  auto local_peer_status =
      d_.external_target_access_service.ensure_local_region_peer(rctx.server_context().peer(), "BatchGetIntoRegion");
  if (!local_peer_status.ok()) {
    return to_grpc_status(local_peer_status);
  }
  const absl::Time now = absl::Now();
  const std::string local_daemon_id = d_.route_resolver.local_daemon_id();

  struct ShardRequest {
    std::vector<std::string> artifact_ids;
    std::vector<int> outcome_indices;
  };

  absl::flat_hash_map<std::uint64_t, ShardRequest> shard_requests;
  shard_requests.reserve(static_cast<std::size_t>(req.selections_size()));
  absl::flat_hash_map<std::string, std::uint64_t> target_layout_lengths;
  target_layout_lengths.reserve(static_cast<std::size_t>(req.selections_size()));

  for (int i = 0; i < req.selections_size(); ++i) {
    const auto& selection = req.selections(i);
    auto* outcome = resp.add_outcomes();
    outcome->set_artifact_id(selection.artifact_id());

    const auto selection_st = validate_batch_selection(selection);
    if (!selection_st.ok()) {
      *outcome = make_outcome(
          selection.artifact_id(), v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, std::string(selection_st.message()));
      continue;
    }
    auto shard_or =
        byte_artifact_runtime().shard_id_for_artifact(selection.artifact_id(), options_.routing.shard_count);
    if (!shard_or.ok()) {
      *outcome = make_outcome(
          selection.artifact_id(), v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, std::string(shard_or.status().message()));
      continue;
    }
    auto& entry = shard_requests[*shard_or];
    entry.artifact_ids.push_back(selection.artifact_id());
    entry.outcome_indices.push_back(i);
    target_layout_lengths.emplace(selection.artifact_id(), /*wildcard=*/0);
  }

  auto target_layout_or = d_.external_target_access_service.validate_local_source_layout(
      rctx.server_context().peer(),
      "BatchGetIntoRegion",
      req.target_layout(),
      req.pid(),
      req.device_uuid(),
      target_layout_lengths);
  if (!target_layout_or.ok()) {
    return to_grpc_status(target_layout_or.status());
  }
  auto target_layout = std::move(target_layout_or->layout);

  const auto apply_item = [&](int outcome_index, const std::string& artifact_id, const v2::HomeBatchGetItem* item) {
    auto* outcome = resp.mutable_outcomes(outcome_index);
    if (item == nullptr) {
      *outcome = make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "missing home item");
      return;
    }
    if (item->status() != v2::BATCH_ITEM_STATUS_OK) {
      *outcome = make_outcome(artifact_id, item->status(), item->message());
      return;
    }

    auto item_target_layout_or = target_layout.build_item_target_layout(artifact_id);
    if (!item_target_layout_or.ok()) {
      *outcome = make_outcome(
          artifact_id,
          batch_item_status_from_absl_status(item_target_layout_or.status()),
          std::string(item_target_layout_or.status().message()));
      return;
    }

    std::unique_ptr<store::IArtifactLoader> loader;
    store::loading::MaterializationSource source_kind = store::loading::MaterializationSource::kLocalReplica;
    uint64_t payload_bytes = 0;
    if (!item->inline_payload().empty()) {
      auto payload = std::make_shared<std::string>(item->inline_payload());
      auto payload_view = std::shared_ptr<const void>(payload, static_cast<const void*>(payload->data()));
      loader = std::make_unique<store::InlineBufferLoader>(store::loading::InlineBufferSource{
          .data = std::move(payload_view),
          .size_bytes = payload->size(),
      });
      payload_bytes = payload->size();
    } else if (!item->payload_ref().empty()) {
      auto loader_or = d_.payload_transport_broker.open_payload_ref_loader(
          d_.worker_directory_cache,
          now,
          absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()),
          local_daemon_id,
          item->payload_ref(),
          artifact_id,
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
          req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view(""));
      if (!loader_or.ok()) {
        *outcome = make_outcome(
            artifact_id,
            batch_item_status_from_absl_status(loader_or.status()),
            std::string(loader_or.status().message()));
        return;
      }
      source_kind = loader_or->remote ? store::loading::MaterializationSource::kP2P
                                      : store::loading::MaterializationSource::kLocalReplica;
      payload_bytes = loader_or->metadata.payload_size;
      loader = std::move(loader_or->loader);
    } else {
      *outcome = make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "home get returned no payload");
      return;
    }

    if (payload_bytes != item_target_layout_or->total_size) {
      *outcome = make_outcome(
          artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, "payload size does not match target layout length");
      return;
    }

    auto lowering_or = build_into_target_lowering_plan(
        artifact_id,
        store::DeviceRegistry::instance().gpu_key(target_layout.device_id()),
        *item_target_layout_or,
        std::move(loader),
        payload_bytes,
        source_kind,
        req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view(""));
    if (!lowering_or.ok()) {
      *outcome = make_outcome(
          artifact_id,
          batch_item_status_from_absl_status(lowering_or.status()),
          std::string(lowering_or.status().message()));
      return;
    }
    auto materialize_or = d_.engine.execute_artifact_lowering_plan(std::move(*lowering_or));
    if (!materialize_or.ok()) {
      *outcome = make_outcome(
          artifact_id,
          batch_item_status_from_absl_status(materialize_or.status()),
          std::string(materialize_or.status().message()));
      return;
    }
    *outcome = make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_OK);
  };

  const bool local_only = (d_.global_store_client == nullptr);
  if (local_only) {
    for (const auto& [shard_id, batch] : shard_requests) {
      if (batch.artifact_ids.empty()) {
        continue;
      }
      v2::HomeBatchGetRequest home_req;
      home_req.mutable_fence()->set_shard_id(shard_id);
      home_req.mutable_fence()->set_lease_generation(1);
      home_req.mutable_fence()->set_holder_daemon_id(local_daemon_id);
      home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
      if (req.has_operation_id()) {
        home_req.set_operation_id(req.operation_id());
      }
      for (const auto& artifact_id : batch.artifact_ids) {
        home_req.add_artifact_ids(artifact_id);
      }
      v2::HomeBatchGetResponse home_resp;
      grpc::ServerContext home_ctx;
      RpcContext home_rctx{"HomeBatchGet", home_ctx, rctx.allow_high_card_attrs()};
      const auto home_status = home_batch_get(home_rctx, home_req, home_resp);
      if (!home_status.ok()) {
        for (std::size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
          *resp.mutable_outcomes(batch.outcome_indices[idx]) =
              make_outcome(batch.artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, home_status.error_message());
        }
        continue;
      }
      absl::flat_hash_map<std::string, const v2::HomeBatchGetItem*> by_id;
      by_id.reserve(home_resp.items_size());
      for (const auto& item : home_resp.items()) {
        by_id.emplace(item.artifact_id(), &item);
      }
      for (std::size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
        const auto it = by_id.find(batch.artifact_ids[idx]);
        apply_item(batch.outcome_indices[idx], batch.artifact_ids[idx], it == by_id.end() ? nullptr : it->second);
      }
    }
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::uint64_t> shard_ids;
  shard_ids.reserve(shard_requests.size());
  for (const auto& [shard_id, /*batch*/ _] : shard_requests) {
    shard_ids.push_back(shard_id);
  }
  auto routes = d_.route_resolver.resolve_routes(absl::MakeSpan(shard_ids), now);

  std::vector<std::string> remote_daemon_ids;
  remote_daemon_ids.reserve(routes.size());
  for (const auto& [shard_id, route] : routes) {
    if (route.ok && route.holder_daemon_id != local_daemon_id) {
      remote_daemon_ids.push_back(route.holder_daemon_id);
    }
  }
  if (!remote_daemon_ids.empty()) {
    (void)d_.worker_directory_cache.warm_for_daemons(
        remote_daemon_ids, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
  }

  for (const auto& [shard_id, batch] : shard_requests) {
    const auto route_it = routes.find(shard_id);
    if (route_it == routes.end() || !route_it->second.ok) {
      const std::string message = (route_it == routes.end() || route_it->second.message.empty())
          ? "shard home route unavailable"
          : route_it->second.message;
      for (std::size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
        *resp.mutable_outcomes(batch.outcome_indices[idx]) =
            make_outcome(batch.artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, message);
      }
      continue;
    }

    std::uint64_t lease_generation = route_it->second.lease_generation;
    std::string holder_daemon_id = route_it->second.holder_daemon_id;

    for (int attempt = 0; attempt < 2; ++attempt) {
      v2::HomeBatchGetResponse home_resp;
      grpc::Status home_status;
      if (holder_daemon_id == local_daemon_id) {
        grpc::ServerContext home_ctx;
        v2::HomeBatchGetRequest home_req;
        home_req.mutable_fence()->set_shard_id(shard_id);
        home_req.mutable_fence()->set_lease_generation(lease_generation);
        home_req.mutable_fence()->set_holder_daemon_id(local_daemon_id);
        home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
        if (req.has_operation_id()) {
          home_req.set_operation_id(req.operation_id());
        }
        for (const auto& artifact_id : batch.artifact_ids) {
          home_req.add_artifact_ids(artifact_id);
        }
        RpcContext home_rctx{"HomeBatchGet", home_ctx, rctx.allow_high_card_attrs()};
        home_status = home_batch_get(home_rctx, home_req, home_resp);
      } else {
        auto address_or = d_.worker_directory_cache.resolve_daemon_address(
            holder_daemon_id, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
        if (!address_or.ok()) {
          for (std::size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
            *resp.mutable_outcomes(batch.outcome_indices[idx]) = make_outcome(
                batch.artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, "home daemon address unavailable");
          }
          break;
        }
        auto channel = grpc::CreateChannel(*address_or, grpc::InsecureChannelCredentials());
        auto stub = v2::StoreDaemonService::NewStub(channel);
        grpc::ClientContext client_ctx;
        client_ctx.set_deadline(std::chrono::system_clock::now() + options_.routing.route_staleness_budget);
        v2::HomeBatchGetRequest home_req;
        home_req.mutable_fence()->set_shard_id(shard_id);
        home_req.mutable_fence()->set_lease_generation(lease_generation);
        home_req.mutable_fence()->set_holder_daemon_id(holder_daemon_id);
        home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
        if (req.has_operation_id()) {
          home_req.set_operation_id(req.operation_id());
        }
        for (const auto& artifact_id : batch.artifact_ids) {
          home_req.add_artifact_ids(artifact_id);
        }
        home_status = stub->HomeBatchGet(&client_ctx, home_req, &home_resp);
      }

      if (!home_status.ok()) {
        for (std::size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
          *resp.mutable_outcomes(batch.outcome_indices[idx]) =
              make_outcome(batch.artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, home_status.error_message());
        }
        break;
      }

      bool needs_redirect_retry = false;
      if (home_resp.has_redirect() && home_resp.redirect().shard_id() == shard_id &&
          home_resp.redirect().lease_generation() != 0 && !home_resp.redirect().holder_daemon_id().empty()) {
        for (const auto& item : home_resp.items()) {
          if (item.status() == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
            needs_redirect_retry = true;
            break;
          }
        }
      }
      if (needs_redirect_retry && attempt == 0) {
        const auto& redirect = home_resp.redirect();
        const auto refreshed = d_.route_resolver.refresh_route_from_redirect(shard_id, redirect, now);
        if (!refreshed.ok) {
          for (std::size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
            *resp.mutable_outcomes(batch.outcome_indices[idx]) =
                make_outcome(batch.artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, refreshed.message);
          }
          break;
        }
        holder_daemon_id = refreshed.holder_daemon_id;
        lease_generation = refreshed.lease_generation;
        continue;
      }

      absl::flat_hash_map<std::string, const v2::HomeBatchGetItem*> by_id;
      by_id.reserve(home_resp.items_size());
      for (const auto& item : home_resp.items()) {
        by_id.emplace(item.artifact_id(), &item);
      }
      for (std::size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
        const auto it = by_id.find(batch.artifact_ids[idx]);
        apply_item(batch.outcome_indices[idx], batch.artifact_ids[idx], it == by_id.end() ? nullptr : it->second);
      }
      break;
    }
  }

  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::batch_put_if_absent_from_region(
    RpcContext& rctx,
    const v2::BatchPutIfAbsentFromRegionRequest& req,
    v2::BatchPutIfAbsentFromRegionResponse& resp) {
  auto local_peer_status = d_.external_target_access_service.ensure_local_region_peer(
      rctx.server_context().peer(), "BatchPutIfAbsentFromRegion");
  if (!local_peer_status.ok()) {
    return to_grpc_status(local_peer_status);
  }
  const absl::Time now = absl::Now();
  const std::optional<std::uint64_t> ttl_ms =
      req.has_ttl_ms() ? std::optional<std::uint64_t>(req.ttl_ms()) : std::nullopt;
  const std::string local_daemon_id = d_.route_resolver.local_daemon_id();
  const std::string operation_id = req.has_operation_id() ? req.operation_id() : "";

  struct PendingPut {
    std::string artifact_id;
    v2::PutIfAbsentInvariant invariant;
    std::optional<BodyHandle> body_handle;
    std::optional<BodyDescriptor> descriptor;
    std::optional<store::runtime::ingestion::VerifiedContentDescriptor> verified_content_descriptor;
    std::optional<store::runtime::ingestion::VerificationRecord> verification_record;
    std::optional<store::runtime::ingestion::BackingIdentity> backing_identity;
    std::optional<BodyBackingObservation> observation;
    std::shared_ptr<const std::string> inline_payload;
    std::string payload_ref;
    bool needs_source_layout{false};
    int outcome_index{0};
  };

  struct ShardPutBatch {
    std::vector<PendingPut> items;
  };

  absl::flat_hash_map<std::uint64_t, ShardPutBatch> shard_batches;
  shard_batches.reserve(static_cast<size_t>(req.items_size()));
  absl::flat_hash_map<std::string, std::uint64_t> source_layout_lengths;
  source_layout_lengths.reserve(static_cast<std::size_t>(req.items_size()));

  for (int i = 0; i < req.items_size(); ++i) {
    const auto& item = req.items(i);
    const std::string artifact_id = item.selection().artifact_id();
    auto* outcome = resp.add_outcomes();
    outcome->set_artifact_id(artifact_id);

    const auto selection_st = validate_batch_selection(item.selection());
    if (!selection_st.ok()) {
      *outcome = make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, std::string(selection_st.message()));
      continue;
    }

    auto shard_or = byte_artifact_runtime().shard_id_for_artifact(artifact_id, options_.routing.shard_count);
    if (!shard_or.ok()) {
      *outcome =
          make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, std::string(shard_or.status().message()));
      continue;
    }

    auto& batch = shard_batches[*shard_or];
    PendingPut pending{
        .artifact_id = artifact_id,
        .invariant = item.invariant(),
        .outcome_index = i,
    };
    if (!item.inline_payload().empty() && !item.payload_ref().empty()) {
      *outcome = make_outcome(
          artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, "inline_payload and payload_ref are mutually exclusive");
      continue;
    }
    if (!item.inline_payload().empty()) {
      pending.inline_payload = std::make_shared<const std::string>(item.inline_payload());
    } else if (!item.payload_ref().empty()) {
      pending.payload_ref = item.payload_ref();
    } else {
      pending.needs_source_layout = true;
      source_layout_lengths[artifact_id] = item.invariant().byte_length();
    }
    batch.items.push_back(std::move(pending));
  }

  std::optional<ByteArtifactRegionLayout> source_layout;
  if (!source_layout_lengths.empty()) {
    auto source_layout_or = d_.external_target_access_service.validate_local_source_layout(
        rctx.server_context().peer(),
        "BatchPutIfAbsentFromRegion",
        req.source_layout(),
        req.pid(),
        req.device_uuid(),
        source_layout_lengths);
    if (!source_layout_or.ok()) {
      const auto status = batch_item_status_from_absl_status(source_layout_or.status());
      for (auto& [_, batch] : shard_batches) {
        for (auto& pending : batch.items) {
          if (!pending.needs_source_layout) {
            continue;
          }
          *resp.mutable_outcomes(pending.outcome_index) =
              make_outcome(pending.artifact_id, status, std::string(source_layout_or.status().message()));
        }
      }
    } else {
      source_layout = std::move(source_layout_or->layout);
    }
  }

  const auto stage_pending_body = [&](PendingPut* pending, BodyAccessClass access_class) {
    if (pending == nullptr || pending->body_handle.has_value()) {
      return;
    }
    if (resp.outcomes(pending->outcome_index).status() != v2::BATCH_ITEM_STATUS_UNSPECIFIED) {
      return;
    }

    std::unique_ptr<store::IArtifactLoader> loader;
    store::loading::MaterializationSource source_kind{store::loading::MaterializationSource::kLocalReplica};
    std::optional<BodyBackingManager::StageResult> staged_body;
    if (pending->inline_payload) {
      loader = std::make_unique<store::InlineBufferLoader>(store::loading::InlineBufferSource{
          .data = std::shared_ptr<const void>(
              pending->inline_payload, static_cast<const void*>(pending->inline_payload->data())),
          .size_bytes = pending->inline_payload->size(),
      });
    } else if (!pending->payload_ref.empty()) {
      auto capability_or = d_.payload_transport_broker.resolve_payload_ref_capability(
          pending->payload_ref,
          pending->artifact_id,
          now,
          local_daemon_id,
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
          operation_id);
      if (!capability_or.ok()) {
        *resp.mutable_outcomes(pending->outcome_index) = make_outcome(
            pending->artifact_id,
            batch_item_status_from_absl_status(capability_or.status()),
            std::string(capability_or.status().message()));
        return;
      }
      const bool allow_reuse = access_class == BodyAccessClass::kHomeDefault;
      if (allow_reuse && capability_or->capability.mode == BodyCapabilityResolutionMode::kLocalBodyHandle) {
        auto reused_or = body_backing_manager_.try_reuse_body(
            BodyBackingManager::ReuseRequest{
                .artifact_id = pending->artifact_id,
                .invariant = pending->invariant,
                .descriptor = capability_or->capability.descriptor,
                .body_handle = capability_or->capability.body_handle,
                .operation_id = operation_id,
                .access_class = access_class,
                .route_role = access_class == BodyAccessClass::kTransientForward ? BodyRouteRole::kTransientForwarder
                                                                                 : BodyRouteRole::kHomeAuthority,
            });
        if (!reused_or.ok()) {
          *resp.mutable_outcomes(pending->outcome_index) = make_outcome(
              pending->artifact_id,
              batch_item_status_from_absl_status(reused_or.status()),
              std::string(reused_or.status().message()));
          return;
        }
        if (reused_or->has_value()) {
          staged_body = std::move(**reused_or);
        }
      }
      if (!staged_body.has_value()) {
        if (capability_or->capability.mode == BodyCapabilityResolutionMode::kLocalBodyHandle) {
          auto loader_or = capability_or->capability.body_handle.make_loader();
          if (!loader_or.ok()) {
            *resp.mutable_outcomes(pending->outcome_index) = make_outcome(
                pending->artifact_id,
                batch_item_status_from_absl_status(loader_or.status()),
                std::string(loader_or.status().message()));
            return;
          }
          loader = std::move(*loader_or);
        } else if (capability_or->capability.local) {
          if (!capability_or->payload) {
            *resp.mutable_outcomes(pending->outcome_index) = make_outcome(
                pending->artifact_id,
                v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION,
                "payload_ref resolved locally without payload bytes");
            return;
          }
          loader = std::make_unique<store::InlineBufferLoader>(store::loading::InlineBufferSource{
              .data = std::shared_ptr<const void>(
                  capability_or->payload, static_cast<const void*>(capability_or->payload->data())),
              .size_bytes = capability_or->payload->size(),
          });
        } else {
          auto loader_or = d_.payload_transport_broker.open_payload_ref_loader(
              d_.worker_directory_cache,
              now,
              absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()),
              local_daemon_id,
              pending->payload_ref,
              pending->artifact_id,
              tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
              operation_id);
          if (!loader_or.ok()) {
            *resp.mutable_outcomes(pending->outcome_index) = make_outcome(
                pending->artifact_id,
                batch_item_status_from_absl_status(loader_or.status()),
                std::string(loader_or.status().message()));
            return;
          }
          source_kind = loader_or->remote ? store::loading::MaterializationSource::kP2P
                                          : store::loading::MaterializationSource::kLocalReplica;
          loader = std::move(loader_or->loader);
        }
      }
    } else if (pending->needs_source_layout) {
      if (!source_layout.has_value()) {
        *resp.mutable_outcomes(pending->outcome_index) = make_outcome(
            pending->artifact_id,
            v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION,
            "source_layout validation did not produce a readable layout");
        return;
      }
      auto source_or = source_layout->open_item_source(pending->artifact_id);
      if (!source_or.ok()) {
        *resp.mutable_outcomes(pending->outcome_index) = make_outcome(
            pending->artifact_id,
            batch_item_status_from_absl_status(source_or.status()),
            std::string(source_or.status().message()));
        return;
      }
      loader = std::make_unique<SeekableSourceLoader>(*source_or, (*source_or)->total_bytes());
    } else {
      *resp.mutable_outcomes(pending->outcome_index) = make_outcome(
          pending->artifact_id,
          v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT,
          "inline_payload, payload_ref, or source_layout entry is required");
      return;
    }

    if (!staged_body.has_value()) {
      auto staged_body_or = body_backing_manager_.stage_body(
          BodyBackingManager::StageRequest{
              .artifact_id = pending->artifact_id,
              .invariant = pending->invariant,
              .loader = std::move(loader),
              .source_kind = source_kind,
              .operation_id = operation_id,
              .access_class = access_class,
              .route_role = access_class == BodyAccessClass::kTransientForward ? BodyRouteRole::kTransientForwarder
                                                                               : BodyRouteRole::kHomeAuthority,
          });
      if (!staged_body_or.ok()) {
        *resp.mutable_outcomes(pending->outcome_index) = make_outcome(
            pending->artifact_id,
            batch_item_status_from_absl_status(staged_body_or.status()),
            std::string(staged_body_or.status().message()));
        return;
      }
      staged_body = std::move(*staged_body_or);
    }

    auto invariant_st =
        byte_artifact_runtime().validate_invariant_body_descriptor(pending->invariant, staged_body->descriptor);
    if (!invariant_st.ok()) {
      auto retire_status = staged_body->body_handle.retire();
      if (!retire_status.ok()) {
        *resp.mutable_outcomes(pending->outcome_index) = make_outcome(
            pending->artifact_id,
            batch_item_status_from_absl_status(retire_status),
            std::string(retire_status.message()));
        return;
      }
      *resp.mutable_outcomes(pending->outcome_index) = make_outcome(
          pending->artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, std::string(invariant_st.message()));
      return;
    }

    pending->body_handle = staged_body->body_handle;
    pending->descriptor = staged_body->descriptor;
    pending->verified_content_descriptor = staged_body->verified_content_descriptor;
    pending->verification_record = staged_body->verification_record;
    pending->backing_identity = staged_body->backing_identity;
    pending->observation = staged_body->observation;
    pending->needs_source_layout = false;
  };

  const auto apply_local_home_put =
      [&](std::uint64_t shard_id, std::uint64_t lease_generation, const ShardPutBatch& batch) {
        v2::RouteFence fence;
        fence.set_shard_id(shard_id);
        fence.set_lease_generation(lease_generation);
        fence.set_holder_daemon_id(local_daemon_id);
        fence.set_routing_epoch(options_.routing.routing_epoch);
        auto home_lease_or = d_.route_resolver.ensure_home_lease(fence, now);
        if (!home_lease_or.ok()) {
          for (const auto& pending : batch.items) {
            if (!pending.body_handle.has_value()) {
              continue;
            }
            *resp.mutable_outcomes(pending.outcome_index) = make_outcome(
                pending.artifact_id,
                batch_item_status_from_absl_status(home_lease_or.status()),
                std::string(home_lease_or.status().message()));
          }
          return;
        }
        if (home_lease_or->kind != ByteArtifactRouteResolver::HomeLeaseDecision::Kind::kOwned) {
          for (const auto& pending : batch.items) {
            if (!pending.body_handle.has_value()) {
              continue;
            }
            *resp.mutable_outcomes(pending.outcome_index) =
                make_outcome(pending.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, home_lease_or->message);
          }
          return;
        }

        std::vector<ByteArtifactAuthorityService::PutItem> authority_items;
        std::vector<int> authority_item_indices;
        authority_items.reserve(batch.items.size());
        authority_item_indices.reserve(batch.items.size());
        for (const auto& pending : batch.items) {
          if (!pending.body_handle.has_value()) {
            continue;
          }
          if (!pending.descriptor.has_value() || !pending.verified_content_descriptor.has_value() ||
              !pending.verification_record.has_value() || !pending.backing_identity.has_value() ||
              !pending.observation.has_value()) {
            *resp.mutable_outcomes(pending.outcome_index) = make_outcome(
                pending.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "staged body metadata is missing");
            continue;
          }
          authority_items.push_back(
              ByteArtifactAuthorityService::PutItem{
                  .artifact_id = pending.artifact_id,
                  .invariant = pending.invariant,
                  .descriptor = *pending.descriptor,
                  .verified_content_descriptor = *pending.verified_content_descriptor,
                  .verification_record = *pending.verification_record,
                  .backing_identity = *pending.backing_identity,
                  .observation = *pending.observation,
                  .body_handle = *pending.body_handle,
              });
          authority_item_indices.push_back(pending.outcome_index);
        }
        const auto authority_outcomes = authority_service_.batch_put_if_absent(
            authority_items,
            ByteArtifactAuthorityService::Context{
                .shard_id = shard_id,
                .lease_generation = home_lease_or->lease_generation,
                .routing_epoch = options_.routing.routing_epoch,
                .shard_count = options_.routing.shard_count,
                .now = now,
            },
            ttl_ms);
        for (std::size_t index = 0; index < authority_outcomes.size(); ++index) {
          *resp.mutable_outcomes(authority_item_indices[index]) = authority_outcomes[index];
        }
      };

  const bool local_only = (d_.global_store_client == nullptr);
  if (local_only) {
    for (auto& [shard_id, batch] : shard_batches) {
      if (batch.items.empty()) {
        continue;
      }
      for (auto& pending : batch.items) {
        stage_pending_body(&pending, BodyAccessClass::kHomeDefault);
      }
      apply_local_home_put(shard_id, /*lease_generation=*/1, batch);
    }
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::uint64_t> shard_ids;
  shard_ids.reserve(shard_batches.size());
  for (const auto& [shard_id, /*batch*/ _] : shard_batches) {
    shard_ids.push_back(shard_id);
  }
  auto routes = d_.route_resolver.resolve_routes(absl::MakeSpan(shard_ids), now);

  std::vector<std::string> remote_daemon_ids;
  remote_daemon_ids.reserve(routes.size());
  for (const auto& [shard_id, route] : routes) {
    if (route.ok && route.holder_daemon_id != local_daemon_id) {
      remote_daemon_ids.push_back(route.holder_daemon_id);
    }
  }
  if (!remote_daemon_ids.empty()) {
    (void)d_.worker_directory_cache.warm_for_daemons(
        remote_daemon_ids, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
  }

  for (auto& [shard_id, batch] : shard_batches) {
    if (batch.items.empty()) {
      continue;
    }
    const auto route_it = routes.find(shard_id);
    if (route_it == routes.end() || !route_it->second.ok) {
      const std::string message = (route_it == routes.end() || route_it->second.message.empty())
          ? "routing lease unavailable"
          : route_it->second.message;
      for (const auto& pending : batch.items) {
        if (resp.outcomes(pending.outcome_index).status() != v2::BATCH_ITEM_STATUS_UNSPECIFIED) {
          continue;
        }
        *resp.mutable_outcomes(pending.outcome_index) =
            make_outcome(pending.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, message);
      }
      continue;
    }

    std::uint64_t lease_generation = route_it->second.lease_generation;
    std::string holder_daemon_id = route_it->second.holder_daemon_id;

    if (holder_daemon_id == local_daemon_id) {
      for (auto& pending : batch.items) {
        stage_pending_body(&pending, BodyAccessClass::kHomeDefault);
      }
      apply_local_home_put(shard_id, lease_generation, batch);
      continue;
    }

    for (auto& pending : batch.items) {
      stage_pending_body(&pending, BodyAccessClass::kTransientForward);
    }
    absl::flat_hash_map<std::string, std::string> payload_ref_by_artifact;
    payload_ref_by_artifact.reserve(batch.items.size());
    std::optional<std::string> payload_ref_error_message;
    for (const auto& pending : batch.items) {
      if (!pending.body_handle.has_value()) {
        continue;
      }
      if (!pending.descriptor.has_value()) {
        *resp.mutable_outcomes(pending.outcome_index) =
            make_outcome(pending.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "staged descriptor is missing");
        payload_ref_error_message = "staged descriptor is missing";
        break;
      }
      auto payload_ref_or = d_.payload_transport_broker.issue_payload_ref(
          pending.artifact_id,
          *pending.body_handle,
          *pending.descriptor,
          pending.backing_identity,
          pending.body_handle->binding_generation(),
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
          operation_id);
      if (!payload_ref_or.ok()) {
        *resp.mutable_outcomes(pending.outcome_index) = make_outcome(
            pending.artifact_id,
            batch_item_status_from_absl_status(payload_ref_or.status()),
            std::string(payload_ref_or.status().message()));
        payload_ref_error_message = std::string(payload_ref_or.status().message());
        break;
      }
      payload_ref_by_artifact.emplace(pending.artifact_id, *payload_ref_or);
    }
    if (payload_ref_error_message.has_value()) {
      for (const auto& pending : batch.items) {
        if (pending.body_handle.has_value() &&
            resp.outcomes(pending.outcome_index).status() == v2::BATCH_ITEM_STATUS_UNSPECIFIED) {
          *resp.mutable_outcomes(pending.outcome_index) =
              make_outcome(pending.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, *payload_ref_error_message);
        }
      }
    }
    if (payload_ref_by_artifact.empty()) {
      for (const auto& pending : batch.items) {
        if (pending.body_handle.has_value()) {
          (void)pending.body_handle->retire();
        }
      }
      continue;
    }

    for (int attempt = 0; attempt < 2; ++attempt) {
      v2::HomeBatchPutIfAbsentResponse home_resp;
      grpc::Status home_status;
      auto address_or = d_.worker_directory_cache.resolve_daemon_address(
          holder_daemon_id, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
      if (!address_or.ok()) {
        for (const auto& pending : batch.items) {
          if (!pending.body_handle.has_value()) {
            continue;
          }
          *resp.mutable_outcomes(pending.outcome_index) =
              make_outcome(pending.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, "home daemon address unavailable");
        }
        break;
      }
      auto channel = grpc::CreateChannel(*address_or, grpc::InsecureChannelCredentials());
      auto stub = v2::StoreDaemonService::NewStub(channel);
      grpc::ClientContext client_ctx;
      client_ctx.set_deadline(std::chrono::system_clock::now() + options_.routing.route_staleness_budget);
      v2::HomeBatchPutIfAbsentRequest home_req;
      home_req.mutable_fence()->set_shard_id(shard_id);
      home_req.mutable_fence()->set_lease_generation(lease_generation);
      home_req.mutable_fence()->set_holder_daemon_id(holder_daemon_id);
      home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
      if (ttl_ms.has_value()) {
        home_req.set_ttl_ms(*ttl_ms);
      }
      if (req.has_operation_id()) {
        home_req.set_operation_id(req.operation_id());
      }
      for (const auto& pending : batch.items) {
        if (!pending.body_handle.has_value()) {
          continue;
        }
        const auto payload_ref_it = payload_ref_by_artifact.find(pending.artifact_id);
        if (payload_ref_it == payload_ref_by_artifact.end()) {
          continue;
        }
        auto* dst = home_req.add_items();
        dst->set_artifact_id(pending.artifact_id);
        dst->mutable_invariant()->CopyFrom(pending.invariant);
        dst->set_payload_ref(payload_ref_it->second);
      }
      if (home_req.items_size() == 0) {
        break;
      }
      home_status = stub->HomeBatchPutIfAbsent(&client_ctx, home_req, &home_resp);

      if (!home_status.ok()) {
        for (const auto& pending : batch.items) {
          if (!pending.body_handle.has_value()) {
            continue;
          }
          *resp.mutable_outcomes(pending.outcome_index) =
              make_outcome(pending.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, home_status.error_message());
        }
        break;
      }

      bool needs_redirect_retry = false;
      if (home_resp.has_redirect() && home_resp.redirect().shard_id() == shard_id &&
          home_resp.redirect().lease_generation() != 0 && !home_resp.redirect().holder_daemon_id().empty()) {
        for (const auto& outcome : home_resp.outcomes()) {
          if (outcome.status() == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
            needs_redirect_retry = true;
            break;
          }
        }
      }
      if (needs_redirect_retry && attempt == 0) {
        const auto& redirect = home_resp.redirect();
        const auto refreshed = d_.route_resolver.refresh_route_from_redirect(shard_id, redirect, now);
        if (!refreshed.ok) {
          for (const auto& pending : batch.items) {
            if (!pending.body_handle.has_value()) {
              continue;
            }
            *resp.mutable_outcomes(pending.outcome_index) =
                make_outcome(pending.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, refreshed.message);
          }
          break;
        }
        holder_daemon_id = refreshed.holder_daemon_id;
        lease_generation = refreshed.lease_generation;
        continue;
      }

      absl::flat_hash_map<std::string, const v2::BatchItemOutcome*> by_id;
      by_id.reserve(home_resp.outcomes_size());
      for (const auto& out : home_resp.outcomes()) {
        by_id.emplace(out.artifact_id(), &out);
      }
      for (const auto& pending : batch.items) {
        if (!pending.body_handle.has_value()) {
          continue;
        }
        const auto out_it = by_id.find(pending.artifact_id);
        if (out_it == by_id.end()) {
          *resp.mutable_outcomes(pending.outcome_index) =
              make_outcome(pending.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "missing home outcome");
          continue;
        }
        v2::BatchItemStatus status = out_it->second->status();
        if (status == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
          status = v2::BATCH_ITEM_STATUS_UNAVAILABLE;
        }
        *resp.mutable_outcomes(pending.outcome_index) =
            make_outcome(pending.artifact_id, status, out_it->second->message());
      }
      break;
    }

    for (const auto& pending : batch.items) {
      if (pending.body_handle.has_value()) {
        (void)pending.body_handle->retire();
      }
    }
  }

  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::batch_touch_ttl(
    RpcContext& rctx,
    const v2::BatchTouchTtlRequest& req,
    v2::BatchTouchTtlResponse& resp) {
  const bool local_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  if (!local_peer && !options_.gateway_ingress_enabled) {
    return {StatusCode::PERMISSION_DENIED, "BatchTouchTtl requires gateway ingress on non-local peers"};
  }
  if (req.ttl_ms() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "ttl_ms must be > 0"};
  }

  const absl::Time now = absl::Now();
  const std::string local_daemon_id = d_.route_resolver.local_daemon_id();
  absl::flat_hash_map<std::uint64_t, std::vector<std::pair<std::string, int>>> shard_batches;
  shard_batches.reserve(static_cast<size_t>(req.artifact_ids_size()));

  for (int i = 0; i < req.artifact_ids_size(); ++i) {
    const auto& artifact_id = req.artifact_ids(i);
    auto* outcome = resp.add_outcomes();
    outcome->set_artifact_id(artifact_id);
    if (artifact_id.empty()) {
      *outcome = make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, "artifact_id is required");
      continue;
    }
    const auto artifact_id_st = byte_artifact_runtime().validate_artifact_id_for_field(artifact_id, "artifact_id");
    if (!artifact_id_st.ok()) {
      *outcome =
          make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, std::string(artifact_id_st.message()));
      continue;
    }
    auto shard_or = byte_artifact_runtime().shard_id_for_artifact(artifact_id, options_.routing.shard_count);
    if (!shard_or.ok()) {
      *outcome =
          make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, std::string(shard_or.status().message()));
      continue;
    }
    shard_batches[*shard_or].push_back({artifact_id, i});
  }

  const bool local_only = (d_.global_store_client == nullptr);
  if (local_only) {
    for (const auto& [shard_id, items] : shard_batches) {
      std::vector<std::string> artifact_ids;
      artifact_ids.reserve(items.size());
      for (const auto& [artifact_id, _] : items) {
        artifact_ids.push_back(artifact_id);
      }
      const auto outcomes = authority_service_.batch_touch_ttl(
          artifact_ids,
          ByteArtifactAuthorityService::Context{
              .shard_id = shard_id,
              .lease_generation = 1,
              .routing_epoch = options_.routing.routing_epoch,
              .shard_count = options_.routing.shard_count,
              .now = now,
          },
          req.ttl_ms());
      for (size_t idx = 0; idx < items.size(); ++idx) {
        *resp.mutable_outcomes(items[idx].second) = outcomes.at(idx);
      }
    }
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::uint64_t> shard_ids;
  shard_ids.reserve(shard_batches.size());
  for (const auto& [shard_id, /*items*/ _] : shard_batches) {
    shard_ids.push_back(shard_id);
  }
  auto routes = d_.route_resolver.resolve_routes(absl::MakeSpan(shard_ids), now);

  std::vector<std::string> remote_daemon_ids;
  remote_daemon_ids.reserve(routes.size());
  for (const auto& [shard_id, route] : routes) {
    if (route.ok && route.holder_daemon_id != local_daemon_id) {
      remote_daemon_ids.push_back(route.holder_daemon_id);
    }
  }
  if (!remote_daemon_ids.empty()) {
    (void)d_.worker_directory_cache.warm_for_daemons(
        remote_daemon_ids, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
  }

  for (const auto& [shard_id, items] : shard_batches) {
    const auto route_it = routes.find(shard_id);
    if (route_it == routes.end() || !route_it->second.ok) {
      const std::string message = (route_it == routes.end() || route_it->second.message.empty())
          ? "routing lease unavailable"
          : route_it->second.message;
      for (const auto& [artifact_id, idx] : items) {
        *resp.mutable_outcomes(idx) = make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, message);
      }
      continue;
    }

    std::uint64_t lease_generation = route_it->second.lease_generation;
    std::string holder_daemon_id = route_it->second.holder_daemon_id;

    for (int attempt = 0; attempt < 2; ++attempt) {
      v2::HomeBatchTouchTtlResponse home_resp;
      grpc::Status home_status;
      if (holder_daemon_id == local_daemon_id) {
        v2::HomeBatchTouchTtlRequest home_req;
        home_req.mutable_fence()->set_shard_id(shard_id);
        home_req.mutable_fence()->set_lease_generation(lease_generation);
        home_req.mutable_fence()->set_holder_daemon_id(local_daemon_id);
        home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
        home_req.set_ttl_ms(req.ttl_ms());
        for (const auto& [artifact_id, /*idx*/ _] : items) {
          home_req.add_artifact_ids(artifact_id);
        }
        grpc::ServerContext home_ctx;
        RpcContext home_rctx{"HomeBatchTouchTtl", home_ctx, rctx.allow_high_card_attrs()};
        home_status = home_batch_touch_ttl(home_rctx, home_req, home_resp);
      } else {
        auto address_or = d_.worker_directory_cache.resolve_daemon_address(
            holder_daemon_id, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
        if (!address_or.ok()) {
          for (const auto& [artifact_id, idx] : items) {
            *resp.mutable_outcomes(idx) =
                make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, "home daemon address unavailable");
          }
          break;
        }
        auto channel = grpc::CreateChannel(*address_or, grpc::InsecureChannelCredentials());
        auto stub = v2::StoreDaemonService::NewStub(channel);
        grpc::ClientContext client_ctx;
        client_ctx.set_deadline(std::chrono::system_clock::now() + options_.routing.route_staleness_budget);
        v2::HomeBatchTouchTtlRequest home_req;
        home_req.mutable_fence()->set_shard_id(shard_id);
        home_req.mutable_fence()->set_lease_generation(lease_generation);
        home_req.mutable_fence()->set_holder_daemon_id(holder_daemon_id);
        home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
        home_req.set_ttl_ms(req.ttl_ms());
        for (const auto& [artifact_id, /*idx*/ _] : items) {
          home_req.add_artifact_ids(artifact_id);
        }
        home_status = stub->HomeBatchTouchTtl(&client_ctx, home_req, &home_resp);
      }

      if (!home_status.ok()) {
        for (const auto& [artifact_id, idx] : items) {
          *resp.mutable_outcomes(idx) =
              make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, home_status.error_message());
        }
        break;
      }

      bool needs_redirect_retry = false;
      if (home_resp.has_redirect() && home_resp.redirect().shard_id() == shard_id &&
          home_resp.redirect().lease_generation() != 0 && !home_resp.redirect().holder_daemon_id().empty()) {
        for (const auto& outcome : home_resp.outcomes()) {
          if (outcome.status() == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
            needs_redirect_retry = true;
            break;
          }
        }
      }
      if (needs_redirect_retry && attempt == 0) {
        const auto& redirect = home_resp.redirect();
        const auto refreshed = d_.route_resolver.refresh_route_from_redirect(shard_id, redirect, now);
        if (!refreshed.ok) {
          for (const auto& [artifact_id, idx] : items) {
            *resp.mutable_outcomes(idx) =
                make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, refreshed.message);
          }
          break;
        }
        holder_daemon_id = refreshed.holder_daemon_id;
        lease_generation = refreshed.lease_generation;
        continue;
      }

      absl::flat_hash_map<std::string, const v2::BatchItemOutcome*> by_id;
      by_id.reserve(home_resp.outcomes_size());
      for (const auto& out : home_resp.outcomes()) {
        by_id.emplace(out.artifact_id(), &out);
      }
      for (const auto& [artifact_id, idx] : items) {
        const auto out_it = by_id.find(artifact_id);
        if (out_it == by_id.end()) {
          *resp.mutable_outcomes(idx) =
              make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "missing home outcome");
          continue;
        }
        v2::BatchItemStatus status = out_it->second->status();
        if (status == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
          status = v2::BATCH_ITEM_STATUS_UNAVAILABLE;
        }
        *resp.mutable_outcomes(idx) = make_outcome(artifact_id, status, out_it->second->message());
      }
      break;
    }
  }

  rctx.mark_success();
  return Status::OK;
}

} // namespace tensorcast::daemon
