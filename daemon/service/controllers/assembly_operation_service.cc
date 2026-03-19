// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/assembly_operation_service.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/sources/memory_source.h"
#include "daemon/service/controllers/assembly_coordination_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/service/controllers/materialization_post_seal_utils.h"
#include "daemon/util/status_utils.h"
#include "folly/futures/Future.h"
#include "google/protobuf/util/time_util.h"
#include "gsl/pointers"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
namespace coordination = assembly_coordination;
using materialization_policy::spec_includes_transpose;
using materialization_post_seal::check_post_seal_view_reuse_safe;
using materialization_post_seal::compute_view_meta_digest;
using status_utils::to_grpc_status;

namespace {
class OperationLeaseGuard {
 public:
  OperationLeaseGuard(
      std::shared_ptr<store::components::IGlobalStoreClient> client,
      std::string lease_token,
      std::string operation_id)
      : client_(std::move(client)), lease_token_(std::move(lease_token)), operation_id_(std::move(operation_id)) {}

  ~OperationLeaseGuard() {
    release();
  }

  void release() {
    if (released_ || client_ == nullptr || lease_token_.empty()) {
      released_ = true;
      return;
    }
    tensorcast::operation::v1::ReleaseOperationLeaseRequest release_req;
    release_req.set_lease_token(lease_token_);
    auto release_or = client_->release_operation_lease(release_req);
    if (!release_or.ok()) {
      LOG(WARNING) << "release_operation_lease failed for op=" << operation_id_ << ": " << release_or.status();
    }
    released_ = true;
  }

 private:
  std::shared_ptr<store::components::IGlobalStoreClient> client_;
  std::string lease_token_;
  std::string operation_id_;
  bool released_{false};
};

std::string compute_operation_id(std::string_view prefix, std::string_view assembly_id) {
  const std::string payload = absl::StrCat(prefix, ":", assembly_id);
  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  const std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(bytes);
  return tensorcast::common::multibase_multihash_sha256(digest);
}

std::string compute_assembly_attempt_operation_id(std::string_view attempt_id) {
  return compute_operation_id("assembly_attempt", attempt_id);
}

std::string compute_seal_operation_id(std::string_view assembly_id) {
  return compute_operation_id("seal_assembly", assembly_id);
}

std::string mint_random_hex_id(size_t bytes_len) {
  thread_local absl::BitGen bitgen;
  std::string raw;
  raw.resize(bytes_len);
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<char>(absl::Uniform<uint32_t>(bitgen, 0u, 256u));
  }
  return absl::BytesToHexString(raw);
}

std::string mint_assembly_attempt_id() {
  return absl::StrCat("cgid:assembly-attempt-", mint_random_hex_id(8));
}

std::string mint_assembly_workspace_id() {
  return absl::StrCat("cgid:assembly-workspace-", mint_random_hex_id(8));
}

std::string owner_id_for_operation(const WorkerIdentityStore& identity) {
  auto daemon_id = identity.daemon_id();
  if (!daemon_id.empty()) {
    return daemon_id;
  }
  auto worker_id = identity.worker_id();
  if (!worker_id.empty()) {
    return worker_id;
  }
  return "unknown";
}

tensorcast::operation::v1::OperationRef build_assembly_attempt_operation_ref(
    std::string_view operation_id,
    std::string_view attempt_id,
    std::string_view workspace_assembly_id,
    std::string_view attempt_intent_digest) {
  tensorcast::operation::v1::OperationRef ref;
  ref.set_operation_id(std::string(operation_id));
  ref.set_kind("assembly_attempt");
  ref.set_target_artifact_id(std::string(workspace_assembly_id));
  ref.set_authority_scope_kind("assembly_attempt");
  ref.set_authority_scope_id(std::string(attempt_id));
  ref.set_attachment_kind("assembly_attempt");
  ref.set_recovery_class("cluster_durable");
  if (!attempt_intent_digest.empty()) {
    ref.set_fencing_digest(std::string(attempt_intent_digest));
  }
  return ref;
}

tensorcast::operation::v1::OperationRef build_low_level_seal_operation_ref(
    std::string_view operation_id,
    std::string_view assembly_id) {
  tensorcast::operation::v1::OperationRef ref;
  ref.set_operation_id(std::string(operation_id));
  ref.set_kind("seal_assembly");
  ref.set_target_artifact_id(std::string(assembly_id));
  return ref;
}

google::protobuf::Any pack_operation_continuation_metadata(const tensorcast::operation::v1::OperationRef& ref) {
  tensorcast::operation::v1::OperationContinuationMetadata metadata;
  metadata.mutable_ref()->CopyFrom(ref);
  google::protobuf::Any any;
  any.PackFrom(metadata);
  return any;
}

v2::AssemblyCloseoutContract build_default_closeout_contract() {
  v2::AssemblyCloseoutContract contract;
  contract.set_kind(v2::ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY);
  return coordination::canonicalize_closeout_contract(contract);
}

absl::StatusOr<v2::AssemblyReadinessCut> capture_seal_readiness_snapshot(
    const std::shared_ptr<store::components::IGlobalStoreClient>& client,
    const v2::AssemblyAttemptRecord& record,
    uint64_t coordinator_generation) {
  if (!client || !client->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }

  v2::AssemblyReadinessCut readiness;
  readiness.set_attempt_id(record.attempt_id());
  readiness.set_attempt_intent_digest(record.intent().attempt_intent_digest());
  readiness.set_coordinator_generation(coordinator_generation);

  auto binding_or = client->get_assembly_layout_binding(record.workspace_assembly_id());
  if (!binding_or.ok()) {
    return binding_or.status();
  }
  if (binding_or->layout_id() != record.intent().layout_id()) {
    return absl::FailedPreconditionError("assembly attempt layout changed");
  }
  readiness.set_workspace_layout_binding_version(binding_or->binding_version());

  absl::flat_hash_set<std::string> added_view_ids;
  std::vector<std::string> required_view_ids;
  for (const auto& requirement : record.intent().requirements().inline_requirements()) {
    if (requirement.target().kind() != v2::ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW ||
        requirement.target().structural_view_id().empty()) {
      continue;
    }
    if (!added_view_ids.insert(requirement.target().structural_view_id()).second) {
      continue;
    }
    required_view_ids.push_back(requirement.target().structural_view_id());
  }

  auto accepted_or = client->list_assembly_slot_occupancies(
      record.attempt_id(),
      /*slot_id=*/std::nullopt,
      /*binding_id=*/std::nullopt,
      /*binding_value_id=*/std::nullopt,
      {"accepted"});
  if (!accepted_or.ok()) {
    return accepted_or.status();
  }

  auto active_identities_or = coordination::list_active_contributor_identities(client);
  if (!active_identities_or.ok()) {
    return active_identities_or.status();
  }
  const auto& active_identities = *active_identities_or;
  const absl::Time now = absl::Now();
  auto live_required_status = coordination::validate_live_required_slot_occupancies(
      record.intent().requirements(),
      record.intent().readiness_policy(),
      absl::MakeConstSpan(*accepted_or),
      active_identities,
      now);
  if (!live_required_status.ok()) {
    return live_required_status;
  }

  absl::flat_hash_map<std::string, store::components::AssemblySlotOccupancyInfo> live_by_slot;
  for (const auto& contribution : *accepted_or) {
    if (!coordination::slot_occupancy_is_live(contribution, now, &active_identities)) {
      continue;
    }
    if (!contribution.slot_id.empty()) {
      live_by_slot[contribution.slot_id] = contribution;
    }
  }

  for (const auto& requirement : record.intent().requirements().inline_requirements()) {
    auto it = live_by_slot.find(requirement.slot_id());
    if (it == live_by_slot.end()) {
      continue;
    }
    auto* out = readiness.add_live_slots();
    out->set_slot_id(requirement.slot_id());
    if (it->second.structural_view_id.has_value()) {
      out->set_structural_view_id(*it->second.structural_view_id);
    }
    out->set_binding_id(it->second.binding_id);
    out->set_binding_value_id(it->second.binding_value_id);
    out->set_lease_id(it->second.lease_id);
    out->set_lease_generation(it->second.lease_generation);
  }

  absl::flat_hash_map<std::string, store::components::ViewInfo> current_views;
  if (!required_view_ids.empty()) {
    auto views_or = client->list_views(record.workspace_assembly_id());
    if (!views_or.ok()) {
      return views_or.status();
    }
    current_views.reserve(views_or->size());
    for (auto& view : *views_or) {
      if (!view.view_id.empty()) {
        current_views.emplace(view.view_id, std::move(view));
      }
    }
  }

  for (const auto& required_view_id : required_view_ids) {
    auto it = current_views.find(required_view_id);
    if (it == current_views.end()) {
      return absl::FailedPreconditionError(
          absl::StrCat("required expected_view_id missing from current view set: ", required_view_id));
    }
    auto* out = readiness.add_views();
    out->set_structural_view_id(it->second.view_id);
    const auto digest = compute_view_meta_digest(it->second);
    out->set_meta_digest(digest.data(), static_cast<int>(digest.size()));
    out->set_view_spec_json(it->second.view_spec_json);
    out->set_view_size_bytes(it->second.view_size_bytes);
    if (it->second.view_data_hash.has_value()) {
      out->set_view_data_hash(*it->second.view_data_hash);
    }
    out->set_canonical_size_bytes(it->second.canonical_size_bytes);
    out->set_canonical_bytes_covered(it->second.canonical_bytes_covered);
    for (const auto& range : it->second.canonical_ranges) {
      auto* out_range = out->add_canonical_ranges();
      out_range->set_offset(range.offset);
      out_range->set_length(range.length);
    }
  }

  return readiness;
}

store::StoreEngine::SealAssemblyCutInput build_seal_cut_input(const v2::AssemblyReadinessCut& readiness_cut) {
  store::StoreEngine::SealAssemblyCutInput input;
  input.structural_views.reserve(static_cast<size_t>(readiness_cut.views_size()));
  for (const auto& cut_view : readiness_cut.views()) {
    store::components::ViewInfo view;
    view.view_id = cut_view.structural_view_id();
    view.view_spec_json = cut_view.view_spec_json();
    view.view_size_bytes = cut_view.view_size_bytes();
    if (!cut_view.view_data_hash().empty()) {
      view.view_data_hash = cut_view.view_data_hash();
    }
    view.canonical_size_bytes = cut_view.canonical_size_bytes();
    view.canonical_bytes_covered = cut_view.canonical_bytes_covered();
    view.canonical_ranges.reserve(static_cast<size_t>(cut_view.canonical_ranges_size()));
    for (const auto& range : cut_view.canonical_ranges()) {
      view.canonical_ranges.push_back(
          store::components::CanonicalRange{.offset = range.offset(), .length = range.length()});
    }
    input.structural_views.push_back(std::move(view));
  }
  input.canonical_full = std::any_of(
      readiness_cut.live_slots().begin(),
      readiness_cut.live_slots().end(),
      [](const v2::AssemblyReadinessCutSlot& slot) {
        return slot.slot_id() == coordination::kCanonicalFullContributionSlotKey;
      });
  return input;
}

bool retryable_status(const absl::Status& st) {
  return absl::IsUnavailable(st) || absl::IsDeadlineExceeded(st) || absl::IsAborted(st) || absl::IsInternal(st) ||
      absl::IsUnknown(st);
}

absl::Status ensure_local_readable_source_artifact(store::StoreEngine& engine, std::string_view artifact_id) {
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required");
  }
  const auto resident_devices = engine.get_resident_devices(artifact_id);
  const bool has_cpu_replica =
      std::any_of(resident_devices.begin(), resident_devices.end(), [](const store::DeviceKey& device) {
        return device.type == DeviceType::CPU;
      });
  if (has_cpu_replica) {
    return absl::OkStatus();
  }

  store::loading::MaterializeHints hints;
  hints.artifact_id = std::string(artifact_id);
  auto handle_or = engine.materialize_replica(
      store::DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      store::StoreEngine::MaterializeMode::AUTO,
      hints);
  if (!handle_or.ok()) {
    return handle_or.status();
  }
  return absl::OkStatus();
}

absl::Status finalize_assembly_slot_occupancies(
    const std::shared_ptr<store::components::IGlobalStoreClient>& client,
    std::string_view attempt_id,
    std::string_view target_state,
    const std::vector<std::string>& current_states) {
  if (!client || !client->is_connected()) {
    return absl::OkStatus();
  }
  auto rows_or = client->list_assembly_slot_occupancies(
      attempt_id,
      /*slot_id=*/std::nullopt,
      /*binding_id=*/std::nullopt,
      /*binding_value_id=*/std::nullopt,
      current_states);
  if (!rows_or.ok()) {
    return rows_or.status();
  }
  for (const auto& row : *rows_or) {
    auto update_or = client->update_assembly_slot_occupancy_state(
        row.attempt_id,
        row.slot_id,
        target_state,
        /*expected_lease_id=*/std::nullopt,
        /*expected_lease_generation=*/std::nullopt,
        /*lease_expires_at=*/std::nullopt,
        current_states);
    if (!update_or.ok() && !absl::IsNotFound(update_or.status())) {
      return update_or.status();
    }
  }
  return absl::OkStatus();
}

constexpr uint64_t kProofChunkBytesV1 = 4ULL * 1024 * 1024;

struct TensorInterval {
  std::string tensor_name;
  uint64_t offset{0};
  uint64_t size_bytes{0};
};

absl::StatusOr<std::vector<TensorInterval>> parse_tensor_intervals(std::string_view canonical_index_json) {
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("canonical_index_json must not be empty");
  }
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(canonical_index_json, nullptr, true);
  } catch (const std::exception& ex) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON: ", ex.what()));
  }
  if (!j.is_object()) {
    return absl::InvalidArgumentError("canonical index JSON must be an object");
  }

  std::vector<TensorInterval> out;
  out.reserve(j.size());
  for (auto it = j.begin(); it != j.end(); ++it) {
    const std::string tensor_name = it.key();
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() < 2) {
      continue;
    }
    TensorInterval interval;
    interval.tensor_name = tensor_name;
    interval.offset = arr[0].get<uint64_t>();
    interval.size_bytes = arr[1].get<uint64_t>();
    out.push_back(std::move(interval));
  }

  std::sort(
      out.begin(), out.end(), [](const TensorInterval& a, const TensorInterval& b) { return a.offset < b.offset; });
  return out;
}

absl::Status publish_immutable_key(store::StoreEngine* engine, std::string_view key, std::string_view artifact_id);

void populate_artifact_descriptor_from_seal_result(
    const store::SealAssemblyResult& seal_result,
    tensorcast::common::v1::ArtifactDescriptor* artifact) {
  ABSL_CHECK(artifact != nullptr);
  artifact->set_artifact_id(seal_result.sealed_artifact_id);
  artifact->set_id_kind(tensorcast::common::v1::ARTIFACT_ID_KIND_MI2);
  if (!seal_result.index_multihash.empty()) {
    artifact->set_index_multihash(seal_result.index_multihash);
  }
  if (!seal_result.data_multihash.empty()) {
    artifact->set_data_multihash(seal_result.data_multihash);
  }
  if (!seal_result.schema_version.empty()) {
    artifact->set_schema_version(seal_result.schema_version);
  }
  if (!seal_result.encoding.empty()) {
    artifact->set_encoding(seal_result.encoding);
  }
  if (seal_result.total_size > 0) {
    artifact->set_total_size(seal_result.total_size);
  }
}

absl::Status finalize_dependency_ready_closeout(
    store::StoreEngine* engine,
    const v2::AssemblyCloseoutContract& closeout_contract,
    const store::SealAssemblyResult& seal_result,
    v2::SealAssemblyResult* result_msg) {
  ABSL_CHECK(engine != nullptr);
  ABSL_CHECK(result_msg != nullptr);

  populate_artifact_descriptor_from_seal_result(seal_result, result_msg->mutable_artifact());

  const auto canonical = coordination::canonicalize_closeout_contract(closeout_contract);
  switch (canonical.kind()) {
    case v2::ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY:
      if (!canonical.serving_version_key().empty() || !canonical.serving_artifact_id().empty() ||
          !canonical.serving_manifest_ref().empty()) {
        return absl::InvalidArgumentError(
            "source_publish_only closeout contracts may not set serving_version_key, serving_artifact_id, or "
            "serving_manifest_ref");
      }
      if (!canonical.source_version_key().empty()) {
        auto publish_status =
            publish_immutable_key(engine, canonical.source_version_key(), seal_result.sealed_artifact_id);
        if (!publish_status.ok()) {
          return publish_status;
        }
        result_msg->set_source_version_key(canonical.source_version_key());
      }
      return absl::OkStatus();
    case v2::ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH:
    case v2::ASSEMBLY_CLOSEOUT_KIND_ROLLOUT_GATED_PUBLISH:
      return absl::UnimplementedError(
          "attempt closeout beyond source_publish_only requires typed child closeout contracts");
    case v2::ASSEMBLY_CLOSEOUT_KIND_UNSPECIFIED:
    default:
      return absl::InternalError("unexpected assembly closeout contract kind");
  }
}

absl::Status publish_immutable_key(store::StoreEngine* engine, std::string_view key, std::string_view artifact_id) {
  ABSL_CHECK(engine != nullptr);
  if (key.empty()) {
    return absl::InvalidArgumentError("immutable key must not be empty");
  }
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("immutable key publication requires artifact_id");
  }

  auto existing_or = engine->resolve_key_mapping(key);
  if (existing_or.ok()) {
    if (existing_or->artifact_id == artifact_id) {
      return absl::OkStatus();
    }
    return absl::AlreadyExistsError(
        absl::StrCat(
            "immutable key already points to a different artifact: key=",
            key,
            " current=",
            existing_or->artifact_id,
            " requested=",
            artifact_id));
  }
  if (!absl::IsNotFound(existing_or.status())) {
    return existing_or.status();
  }
  return engine->upsert_key_mapping(key, artifact_id);
}

} // namespace

AssemblyOperationService::AssemblyOperationService(Dep d)
    : d_(std::move(d)),
      seal_operation_tracker_(std::make_shared<SealOperationTracker>()),
      coordinator_keepalive_tracker_(std::make_shared<CoordinatorKeepaliveTracker>()) {}

grpc::Status AssemblyOperationService::start_assembly_attempt(
    RpcContext& rctx,
    const v2::StartAssemblyAttemptRequest& req,
    v2::StartAssemblyAttemptResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.layout.id", req.layout_id());

  if (req.layout_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "layout_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  if (!req.has_requirements()) {
    return {
        StatusCode::INVALID_ARGUMENT,
        "requirements are required; daemon no longer derives attempt requirements from layout expected_view_ids"};
  }

  const auto layout_or = d_.global_store_client->get_layout_spec(req.layout_id());
  if (!layout_or.ok()) {
    return to_grpc_status(layout_or.status());
  }

  v2::AssemblyRequirementSetRef requirements = coordination::canonicalize_requirement_set(req.requirements());
  if (requirements.inline_requirements_size() == 0) {
    return {
        StatusCode::INVALID_ARGUMENT,
        "requirements.inline_requirements must be non-empty; use a canonical_layout requirement for canonical-full "
        "attempts"};
  }
  if (auto requirement_status = coordination::validate_requirement_set(requirements); !requirement_status.ok()) {
    return to_grpc_status(requirement_status);
  }

  v2::AssemblyReadinessPolicy readiness_policy = req.has_readiness_policy()
      ? coordination::canonicalize_readiness_policy(req.readiness_policy())
      : coordination::canonicalize_readiness_policy(v2::AssemblyReadinessPolicy());

  v2::AssemblyCloseoutContract closeout_contract = req.has_closeout_contract()
      ? coordination::canonicalize_closeout_contract(req.closeout_contract())
      : build_default_closeout_contract();
  if (auto closeout_status = coordination::validate_dependency_ready_closeout_contract(closeout_contract);
      !closeout_status.ok()) {
    return to_grpc_status(closeout_status);
  }

  v2::AssemblyAttemptIntent intent;
  intent.set_layout_id(req.layout_id());
  *intent.mutable_requirements() = requirements;
  *intent.mutable_readiness_policy() = readiness_policy;
  *intent.mutable_closeout_contract() = closeout_contract;
  intent = coordination::canonicalize_attempt_intent(intent);

  const std::string attempt_id = mint_assembly_attempt_id();
  const std::string workspace_assembly_id = mint_assembly_workspace_id();
  auto binding_or = d_.global_store_client->update_assembly_layout_binding(
      workspace_assembly_id,
      req.layout_id(),
      /*expected_binding_version=*/0);
  if (!binding_or.ok()) {
    return to_grpc_status(binding_or.status());
  }

  v2::AssemblyAttemptRecord record;
  record.set_attempt_id(attempt_id);
  record.set_workspace_assembly_id(workspace_assembly_id);
  record.mutable_intent()->CopyFrom(intent);

  const std::string coordinator_operation_id = compute_assembly_attempt_operation_id(attempt_id);
  store::components::AssemblyAttemptRecordInfo attempt_info;
  attempt_info.attempt_id = attempt_id;
  attempt_info.workspace_assembly_id = workspace_assembly_id;
  attempt_info.layout_id = req.layout_id();
  attempt_info.attempt_intent_digest = intent.attempt_intent_digest();
  attempt_info.coordinator_operation_id = coordinator_operation_id;
  if (!record.SerializeToString(&attempt_info.attempt_record_proto)) {
    return {StatusCode::INTERNAL, "failed to serialize assembly attempt record"};
  }
  auto attempt_upsert_or = d_.global_store_client->upsert_assembly_attempt(attempt_info);
  if (!attempt_upsert_or.ok()) {
    return to_grpc_status(attempt_upsert_or.status());
  }

  const auto operation_ref = build_assembly_attempt_operation_ref(
      coordinator_operation_id, attempt_id, workspace_assembly_id, intent.attempt_intent_digest());

  tensorcast::operation::v1::AcquireOperationLeaseRequest lease_req;
  lease_req.set_operation_id(coordinator_operation_id);
  lease_req.set_kind("assembly_attempt");
  lease_req.set_target_artifact_id(workspace_assembly_id);
  lease_req.set_owner_id(owner_id_for_operation(d_.identity));
  lease_req.set_ttl_ms(0);
  auto lease_or = d_.global_store_client->acquire_operation_lease(lease_req);
  if (!lease_or.ok()) {
    return to_grpc_status(lease_or.status());
  }
  if (!lease_or->acquired()) {
    return {StatusCode::FAILED_PRECONDITION, "assembly attempt coordinator is already owned by another daemon"};
  }

  const google::protobuf::Any snapshot_any = pack_operation_continuation_metadata(operation_ref);
  tensorcast::operation::v1::UpdateOperationRequest pending;
  pending.set_operation_id(coordinator_operation_id);
  pending.set_lease_generation(lease_or->lease().lease_generation());
  auto* status = pending.mutable_status();
  status->set_state(tensorcast::operation::v1::OPERATION_STATE_PENDING);
  status->set_message("assembly attempt open");
  status->set_progress(0.0);
  *status->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
  pending.mutable_snapshot()->CopyFrom(snapshot_any);
  auto update_status = d_.global_store_client->update_operation(pending);
  if (!update_status.ok()) {
    tensorcast::operation::v1::ReleaseOperationLeaseRequest release_req;
    release_req.set_lease_token(lease_or->lease().lease_token());
    auto release_or = d_.global_store_client->release_operation_lease(release_req);
    if (!release_or.ok()) {
      LOG(WARNING) << "release_operation_lease failed after StartAssemblyAttempt update failure for op="
                   << coordinator_operation_id << ": " << release_or.status();
    }
    return to_grpc_status(update_status);
  }

  auto keepalive_stop = std::make_shared<std::atomic<bool>>(false);
  {
    absl::MutexLock lock(&coordinator_keepalive_tracker_->mu);
    coordinator_keepalive_tracker_->stop_flags[coordinator_operation_id] = keepalive_stop;
  }
  auto tracker = coordinator_keepalive_tracker_;
  auto client_sp = d_.global_store_client;
  auto executor = d_.async_runtime.blocking_executor();
  auto keepalive = std::make_shared<std::function<void()>>();
  std::weak_ptr<std::function<void()>> keepalive_weak = keepalive;
  const std::string lease_token = lease_or->lease().lease_token();
  *keepalive = [tracker,
                client_sp,
                keepalive_stop,
                executor,
                keepalive_weak,
                &timekeeper = d_.async_runtime.timekeeper(),
                coordinator_operation_id,
                lease_token]() mutable {
    if (keepalive_stop->load(std::memory_order_relaxed)) {
      absl::MutexLock lock(&tracker->mu);
      auto it = tracker->stop_flags.find(coordinator_operation_id);
      if (it != tracker->stop_flags.end() && it->second == keepalive_stop) {
        tracker->stop_flags.erase(it);
      }
      return;
    }
    timekeeper.after(std::chrono::milliseconds(5000))
        .via(executor)
        .thenValue([tracker, client_sp, keepalive_stop, keepalive_weak, coordinator_operation_id, lease_token](
                       folly::Unit) mutable {
          if (keepalive_stop->load(std::memory_order_relaxed)) {
            absl::MutexLock lock(&tracker->mu);
            auto it = tracker->stop_flags.find(coordinator_operation_id);
            if (it != tracker->stop_flags.end() && it->second == keepalive_stop) {
              tracker->stop_flags.erase(it);
            }
            return;
          }
          tensorcast::operation::v1::KeepaliveOperationLeaseRequest keepalive_req;
          keepalive_req.set_lease_token(lease_token);
          keepalive_req.set_ttl_ms(0);
          auto keepalive_or = client_sp->keepalive_operation_lease(keepalive_req);
          if (!keepalive_or.ok()) {
            LOG(WARNING) << "attempt coordinator keepalive failed for op=" << coordinator_operation_id << ": "
                         << keepalive_or.status();
          }
          auto next = keepalive_weak.lock();
          if (next != nullptr) {
            (*next)();
          }
        });
  };
  (*keepalive)();

  auto* attempt = resp.mutable_attempt();
  attempt->set_attempt_id(attempt_id);
  attempt->set_workspace_assembly_id(workspace_assembly_id);
  attempt->set_layout_id(req.layout_id());
  attempt->set_attempt_intent_digest(intent.attempt_intent_digest());
  attempt->set_coordinator_generation(lease_or->lease().lease_generation());
  attempt->mutable_coordinator_operation()->CopyFrom(operation_ref);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status AssemblyOperationService::seal_assembly_attempt(
    RpcContext& rctx,
    const v2::SealAssemblyAttemptRequest& req,
    v2::SealAssemblyAttemptResponse& resp) {
  if (req.attempt_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "attempt_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  auto attempt_or = d_.global_store_client->get_assembly_attempt(req.attempt_id());
  if (!attempt_or.ok()) {
    return to_grpc_status(attempt_or.status());
  }
  v2::AssemblyAttemptRecord record;
  if (!record.ParseFromString(attempt_or->attempt_record_proto)) {
    return {StatusCode::FAILED_PRECONDITION, "assembly attempt record is malformed"};
  }
  if (record.attempt_id() != req.attempt_id()) {
    return {StatusCode::FAILED_PRECONDITION, "assembly attempt changed"};
  }

  const std::string operation_id = compute_assembly_attempt_operation_id(record.attempt_id());
  auto* out_ref = resp.mutable_operation();
  out_ref->CopyFrom(build_assembly_attempt_operation_ref(
      operation_id, record.attempt_id(), record.workspace_assembly_id(), record.intent().attempt_intent_digest()));

  tensorcast::operation::v1::AcquireOperationLeaseRequest lease_req;
  lease_req.set_operation_id(operation_id);
  lease_req.set_kind("assembly_attempt");
  lease_req.set_target_artifact_id(record.workspace_assembly_id());
  lease_req.set_owner_id(owner_id_for_operation(d_.identity));
  lease_req.set_ttl_ms(0);

  auto lease_or = d_.global_store_client->acquire_operation_lease(lease_req);
  if (!lease_or.ok()) {
    if (absl::IsAlreadyExists(lease_or.status())) {
      rctx.mark_success();
      return Status::OK;
    }
    return to_grpc_status(lease_or.status());
  }
  if (!lease_or->acquired()) {
    rctx.mark_success();
    return Status::OK;
  }

  const auto lease = lease_or->lease();
  const uint64_t lease_generation = lease.lease_generation();
  const std::string lease_token = lease.lease_token();
  const std::string attempt_id = record.attempt_id();
  const std::string workspace_assembly_id = record.workspace_assembly_id();
  {
    absl::MutexLock lock(&coordinator_keepalive_tracker_->mu);
    auto it = coordinator_keepalive_tracker_->stop_flags.find(operation_id);
    if (it != coordinator_keepalive_tracker_->stop_flags.end()) {
      it->second->store(true, std::memory_order_relaxed);
      coordinator_keepalive_tracker_->stop_flags.erase(it);
    }
  }

  auto seal_tracker = seal_operation_tracker_;
  bool should_start = false;
  {
    absl::MutexLock lock(&seal_tracker->mu);
    should_start = seal_tracker->active_operations.insert(operation_id).second;
  }

  if (should_start) {
    auto client_sp = d_.global_store_client;
    auto executor = d_.async_runtime.blocking_executor();
    auto* async_runtime = &d_.async_runtime;
    auto* engine = &d_.engine;
    auto* devices = &d_.devices;
    auto* identity = &d_.identity;
    const DaemonOptions::PostSealPolicy post_seal_policy = d_.post_seal_policy;
    const google::protobuf::Any snapshot_any = pack_operation_continuation_metadata(*out_ref);
    executor->add(
        [seal_tracker,
         client_sp = std::move(client_sp),
         async_runtime,
         engine,
         devices,
         identity,
         post_seal_policy,
         operation_id,
         attempt_id,
         workspace_assembly_id,
         record,
         lease_generation,
         lease_token,
         snapshot_any]() mutable -> void {
          if (client_sp == nullptr) {
            return;
          }
          absl::Status final_status = absl::OkStatus();
          OperationLeaseGuard lease_guard(client_sp, lease_token, operation_id);
          auto cleanup = absl::MakeCleanup([seal_tracker, operation_id]() {
            absl::MutexLock lock(&seal_tracker->mu);
            seal_tracker->active_operations.erase(operation_id);
          });

          auto keepalive_stop = std::make_shared<std::atomic<bool>>(false);
          auto keepalive_exec = async_runtime->blocking_executor();
          auto keepalive = std::make_shared<std::function<void()>>();
          std::weak_ptr<std::function<void()>> keepalive_weak = keepalive;
          *keepalive = [client_sp,
                        keepalive_stop,
                        keepalive_exec,
                        keepalive_weak,
                        &timekeeper = async_runtime->timekeeper(),
                        lease_token]() mutable {
            if (keepalive_stop->load(std::memory_order_relaxed)) {
              return;
            }
            timekeeper.after(std::chrono::milliseconds(5000))
                .via(keepalive_exec)
                .thenValue([client_sp, keepalive_stop, lease_token, keepalive_weak](folly::Unit) mutable {
                  if (keepalive_stop->load(std::memory_order_relaxed)) {
                    return;
                  }
                  tensorcast::operation::v1::KeepaliveOperationLeaseRequest req;
                  req.set_lease_token(lease_token);
                  req.set_ttl_ms(0);
                  auto resp_or = client_sp->keepalive_operation_lease(req);
                  if (!resp_or.ok()) {
                    LOG(WARNING) << "keepalive_operation_lease failed for op=" << lease_token << ": "
                                 << resp_or.status();
                  }
                  auto next = keepalive_weak.lock();
                  if (next != nullptr) {
                    (*next)();
                  }
                });
          };
          (*keepalive)();

          tensorcast::operation::v1::UpdateOperationRequest running;
          running.set_operation_id(operation_id);
          running.set_lease_generation(lease_generation);
          auto* status = running.mutable_status();
          status->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);
          status->set_message("sealing");
          status->set_progress(0.0);
          *status->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
          running.mutable_snapshot()->CopyFrom(snapshot_any);
          final_status = client_sp->update_operation(running);
          if (!final_status.ok()) {
            LOG(WARNING) << "update_operation(RUNNING) failed for op=" << operation_id << ": " << final_status;
            keepalive_stop->store(true, std::memory_order_relaxed);
            return;
          }

          v2::AssemblyReadinessCut readiness_cut;
          std::optional<store::StoreEngine::SealAssemblyCutInput> seal_cut_input;
          if (final_status.ok()) {
            auto readiness_or = capture_seal_readiness_snapshot(client_sp, record, lease_generation);
            if (!readiness_or.ok()) {
              final_status = readiness_or.status();
            } else {
              readiness_cut = *readiness_or;
              store::components::AssemblyReadinessCutInfo readiness_info;
              readiness_info.attempt_id = attempt_id;
              if (!readiness_cut.SerializeToString(&readiness_info.readiness_cut_proto)) {
                final_status = absl::InternalError("failed to serialize readiness cut");
              } else {
                auto upsert_or = client_sp->upsert_assembly_readiness_cut(readiness_info);
                if (!upsert_or.ok()) {
                  final_status = upsert_or.status();
                }
              }
              if (final_status.ok()) {
                seal_cut_input = build_seal_cut_input(readiness_cut);
              }
            }
          }

          auto last_progress_ms = std::make_shared<std::atomic<int64_t>>(0);
          auto max_hashed = std::make_shared<std::atomic<uint64_t>>(0);
          auto enable_updates = std::make_shared<std::atomic<bool>>(true);
          store::runtime::ingestion::MaterializationFacade::SealProgressCallback progress_cb =
              [client_sp, operation_id, lease_generation, last_progress_ms, max_hashed, enable_updates](
                  uint64_t hashed_leaf_count, uint64_t total_hash_leaves) mutable {
                if (!enable_updates->load(std::memory_order_relaxed) || total_hash_leaves == 0) {
                  return;
                }
                const uint64_t prev_max = max_hashed->load(std::memory_order_relaxed);
                if (hashed_leaf_count <= prev_max && hashed_leaf_count != total_hash_leaves) {
                  return;
                }
                max_hashed->store(std::max(prev_max, hashed_leaf_count), std::memory_order_relaxed);
                const int64_t now_ms = absl::ToUnixMillis(absl::Now());
                const int64_t last_ms = last_progress_ms->load(std::memory_order_relaxed);
                if (hashed_leaf_count != total_hash_leaves && last_ms != 0 && now_ms - last_ms < 1000) {
                  return;
                }
                last_progress_ms->store(now_ms, std::memory_order_relaxed);
                tensorcast::operation::v1::UpdateOperationRequest update;
                update.set_operation_id(operation_id);
                update.set_lease_generation(lease_generation);
                auto* status = update.mutable_status();
                status->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);
                status->set_message(absl::StrCat("hashing ", hashed_leaf_count, "/", total_hash_leaves));
                status->set_progress(static_cast<double>(hashed_leaf_count) / static_cast<double>(total_hash_leaves));
                *status->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
                absl::Status st = client_sp->update_operation(update);
                if (!st.ok()) {
                  enable_updates->store(false, std::memory_order_relaxed);
                  LOG(WARNING) << "update_operation(progress) failed for op=" << operation_id << ": " << st;
                }
              };

          auto seal_or = final_status.ok() ? engine->seal_assembly_from_cut(
                                                 workspace_assembly_id,
                                                 *seal_cut_input,
                                                 /*publish_canonical=*/true,
                                                 std::move(progress_cb))
                                           : absl::StatusOr<store::SealAssemblyResult>(final_status);
          if (!seal_or.ok()) {
            final_status = seal_or.status();
          } else {
            const std::string sealed_artifact_id = seal_or->sealed_artifact_id;
            if (final_status.ok() && !record.intent().layout_id().empty()) {
              final_status = client_sp->attach_layout_to_artifact(sealed_artifact_id, record.intent().layout_id());
            }

            tensorcast::daemon::v2::SealAssemblyResult result_msg;
            if (final_status.ok()) {
              final_status = finalize_dependency_ready_closeout(
                  engine, record.intent().closeout_contract(), *seal_or, &result_msg);
            }

            if (final_status.ok()) {
              final_status = ensure_local_readable_source_artifact(*engine, sealed_artifact_id);
            }

            if (final_status.ok()) {
              auto finalize_status =
                  finalize_assembly_slot_occupancies(client_sp, attempt_id, "released", {"accepted"});
              if (!finalize_status.ok()) {
                LOG(WARNING) << "finalize assembly slot occupancies (released) failed for attempt=" << attempt_id
                             << ": " << finalize_status;
              }
            }

            tensorcast::operation::v1::UpdateOperationRequest success;
            success.set_operation_id(operation_id);
            success.set_lease_generation(lease_generation);
            auto* out = success.mutable_status();
            out->set_state(tensorcast::operation::v1::OPERATION_STATE_SUCCESS);
            out->set_message("sealed");
            out->set_progress(1.0);
            *out->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
            out->mutable_result()->PackFrom(result_msg);
            if (final_status.ok()) {
              final_status = client_sp->update_operation(success);
            }
          }

          if (!final_status.ok()) {
            auto finalize_status =
                finalize_assembly_slot_occupancies(client_sp, attempt_id, "aborted", {"accepted", "stale"});
            if (!finalize_status.ok()) {
              LOG(WARNING) << "finalize assembly slot occupancies (aborted) failed for attempt=" << attempt_id << ": "
                           << finalize_status;
            }
            tensorcast::operation::v1::UpdateOperationRequest failed;
            failed.set_operation_id(operation_id);
            failed.set_lease_generation(lease_generation);
            auto* out = failed.mutable_status();
            out->set_state(tensorcast::operation::v1::OPERATION_STATE_FAILED);
            out->set_message("seal failed");
            out->set_progress(0.0);
            *out->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
            auto* err = out->mutable_error();
            err->set_status_code(absl::StatusCodeToString(final_status.code()));
            err->set_message(std::string(final_status.message()));
            err->set_retryable(retryable_status(final_status));
            absl::Status update_st = client_sp->update_operation(failed);
            if (!update_st.ok()) {
              LOG(WARNING) << "update_operation(FAILED) failed for op=" << operation_id << ": " << update_st;
            }
          }

          keepalive_stop->store(true, std::memory_order_relaxed);
          lease_guard.release();
        });
  }

  rctx.mark_success();
  return Status::OK;
}

grpc::Status AssemblyOperationService::seal_assembly(
    RpcContext& rctx,
    const v2::SealAssemblyRequest& req,
    v2::SealAssemblyResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.assembly_id());

  if (req.assembly_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "assembly_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  auto result_or = d_.engine.seal_assembly(req.assembly_id(), req.publish_canonical());
  if (!result_or.ok()) {
    return to_grpc_status(result_or.status());
  }
  const auto& result = *result_or;
  resp.set_sealed_artifact_id(result.sealed_artifact_id);
  resp.set_already_sealed(result.already_sealed);
  populate_artifact_descriptor_from_seal_result(result, resp.mutable_descriptor_());
  rctx.mark_success();
  return Status::OK;
}

grpc::Status AssemblyOperationService::start_seal_assembly(
    RpcContext& rctx,
    const v2::StartSealAssemblyRequest& req,
    v2::StartSealAssemblyResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.assembly_id());

  if (req.assembly_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "assembly_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  const std::string operation_id = compute_seal_operation_id(req.assembly_id());
  auto* out_ref = resp.mutable_operation();
  out_ref->CopyFrom(build_low_level_seal_operation_ref(operation_id, req.assembly_id()));

  tensorcast::operation::v1::AcquireOperationLeaseRequest lease_req;
  lease_req.set_operation_id(operation_id);
  lease_req.set_kind("seal_assembly");
  lease_req.set_target_artifact_id(req.assembly_id());
  lease_req.set_owner_id(owner_id_for_operation(d_.identity));
  // Let Global Store apply defaults and clamp to limits.
  lease_req.set_ttl_ms(0);

  auto lease_or = d_.global_store_client->acquire_operation_lease(lease_req);
  if (!lease_or.ok()) {
    if (absl::IsAlreadyExists(lease_or.status())) {
      rctx.mark_success();
      return Status::OK;
    }
    return to_grpc_status(lease_or.status());
  }
  const auto& lease_resp = *lease_or;
  if (!lease_resp.acquired()) {
    rctx.mark_success();
    return Status::OK;
  }

  const auto lease = lease_resp.lease();
  const uint64_t lease_generation = lease.lease_generation();
  const std::string lease_token = lease.lease_token();
  const std::string assembly_id = req.assembly_id();
  {
    absl::MutexLock lock(&coordinator_keepalive_tracker_->mu);
    auto it = coordinator_keepalive_tracker_->stop_flags.find(operation_id);
    if (it != coordinator_keepalive_tracker_->stop_flags.end()) {
      it->second->store(true, std::memory_order_relaxed);
      coordinator_keepalive_tracker_->stop_flags.erase(it);
    }
  }

  auto seal_tracker = seal_operation_tracker_;
  bool should_start = false;
  {
    absl::MutexLock lock(&seal_tracker->mu);
    should_start = seal_tracker->active_operations.insert(operation_id).second;
  }

  if (should_start) {
    auto client_sp = d_.global_store_client;
    auto executor = d_.async_runtime.blocking_executor();
    auto* async_runtime = &d_.async_runtime;
    auto* engine = &d_.engine;
    auto* devices = &d_.devices;
    auto* identity = &d_.identity;
    const DaemonOptions::PostSealPolicy post_seal_policy = d_.post_seal_policy;
    const google::protobuf::Any snapshot_any = pack_operation_continuation_metadata(*out_ref);
    executor->add(
        [seal_tracker,
         client_sp = std::move(client_sp),
         async_runtime,
         engine,
         devices,
         identity,
         post_seal_policy,
         operation_id,
         assembly_id,
         layout_id = std::string(req.layout_id()),
         lease_generation,
         lease_token,
         snapshot_any]() mutable -> void {
          if (client_sp == nullptr) {
            return;
          }
          absl::Status final_status = absl::OkStatus();
          OperationLeaseGuard lease_guard(client_sp, lease_token, operation_id);
          auto cleanup = absl::MakeCleanup([seal_tracker, operation_id]() {
            absl::MutexLock lock(&seal_tracker->mu);
            seal_tracker->active_operations.erase(operation_id);
          });

          auto keepalive_stop = std::make_shared<std::atomic<bool>>(false);
          auto keepalive_exec = async_runtime->blocking_executor();
          auto keepalive = std::make_shared<std::function<void()>>();
          std::weak_ptr<std::function<void()>> keepalive_weak = keepalive;
          *keepalive = [client_sp,
                        keepalive_stop,
                        keepalive_exec,
                        keepalive_weak,
                        &timekeeper = async_runtime->timekeeper(),
                        lease_token]() mutable {
            if (keepalive_stop->load(std::memory_order_relaxed)) {
              return;
            }
            timekeeper.after(std::chrono::milliseconds(5000))
                .via(keepalive_exec)
                .thenValue([client_sp, keepalive_stop, lease_token, keepalive_weak](folly::Unit) mutable {
                  if (keepalive_stop->load(std::memory_order_relaxed)) {
                    return;
                  }
                  tensorcast::operation::v1::KeepaliveOperationLeaseRequest req;
                  req.set_lease_token(lease_token);
                  req.set_ttl_ms(0);
                  auto resp_or = client_sp->keepalive_operation_lease(req);
                  if (!resp_or.ok()) {
                    LOG(WARNING) << "keepalive_operation_lease failed for op=" << lease_token << ": "
                                 << resp_or.status();
                  }
                  auto next = keepalive_weak.lock();
                  if (next != nullptr) {
                    (*next)();
                  }
                });
          };
          (*keepalive)();

          tensorcast::operation::v1::UpdateOperationRequest running;
          running.set_operation_id(operation_id);
          running.set_lease_generation(lease_generation);
          auto* status = running.mutable_status();
          status->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);
          status->set_message("sealing");
          status->set_progress(0.0);
          *status->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
          running.mutable_snapshot()->CopyFrom(snapshot_any);
          final_status = client_sp->update_operation(running);
          if (!final_status.ok()) {
            LOG(WARNING) << "update_operation(RUNNING) failed for op=" << operation_id << ": " << final_status;
            keepalive_stop->store(true, std::memory_order_relaxed);
            return;
          }

          if (final_status.ok() && layout_id.empty()) {
            auto binding_or = client_sp->get_assembly_layout_binding(assembly_id);
            if (!binding_or.ok()) {
              final_status = binding_or.status();
            } else {
              layout_id = binding_or->layout_id();
            }
          }

          auto last_progress_ms = std::make_shared<std::atomic<int64_t>>(0);
          auto max_hashed = std::make_shared<std::atomic<uint64_t>>(0);
          auto enable_updates = std::make_shared<std::atomic<bool>>(true);
          store::runtime::ingestion::MaterializationFacade::SealProgressCallback progress_cb =
              [client_sp, operation_id, lease_generation, last_progress_ms, max_hashed, enable_updates](
                  uint64_t hashed_leaf_count, uint64_t total_hash_leaves) mutable {
                if (!enable_updates->load(std::memory_order_relaxed) || total_hash_leaves == 0) {
                  return;
                }
                const uint64_t prev_max = max_hashed->load(std::memory_order_relaxed);
                if (hashed_leaf_count <= prev_max && hashed_leaf_count != total_hash_leaves) {
                  return;
                }
                max_hashed->store(std::max(prev_max, hashed_leaf_count), std::memory_order_relaxed);

                const int64_t now_ms = absl::ToUnixMillis(absl::Now());
                const int64_t last_ms = last_progress_ms->load(std::memory_order_relaxed);
                if (hashed_leaf_count != total_hash_leaves && last_ms != 0 && now_ms - last_ms < 1000) {
                  return;
                }
                last_progress_ms->store(now_ms, std::memory_order_relaxed);

                tensorcast::operation::v1::UpdateOperationRequest update;
                update.set_operation_id(operation_id);
                update.set_lease_generation(lease_generation);
                auto* status = update.mutable_status();
                status->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);
                status->set_message(absl::StrCat("hashing ", hashed_leaf_count, "/", total_hash_leaves));
                status->set_progress(static_cast<double>(hashed_leaf_count) / static_cast<double>(total_hash_leaves));
                *status->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
                absl::Status st = client_sp->update_operation(update);
                if (!st.ok()) {
                  enable_updates->store(false, std::memory_order_relaxed);
                  LOG(WARNING) << "update_operation(progress) failed for op=" << operation_id << ": " << st;
                }
              };

          auto seal_or = final_status.ok()
              ? engine->seal_assembly(assembly_id, /*publish_canonical=*/true, std::move(progress_cb), nullptr)
              : absl::StatusOr<store::SealAssemblyResult>(final_status);
          if (!seal_or.ok()) {
            final_status = seal_or.status();
          } else {
            const std::string sealed_artifact_id = seal_or->sealed_artifact_id;
            if (final_status.ok() && !layout_id.empty()) {
              final_status = client_sp->attach_layout_to_artifact(sealed_artifact_id, layout_id);
            }

            std::optional<tensorcast::layout::v1::LayoutSpec> layout_spec_for_post_seal;
            if (final_status.ok() && !layout_id.empty()) {
              auto layout_or = client_sp->get_layout_spec(layout_id);
              if (!layout_or.ok()) {
                final_status = layout_or.status();
              } else {
                layout_spec_for_post_seal = layout_or->layout();
                const auto& layout_spec = *layout_spec_for_post_seal;
                const std::string proof_schema_version = layout_spec.proof_schema_version();
                absl::flat_hash_set<std::string> replicated_tensors;
                replicated_tensors.reserve(layout_spec.tensors_size());
                for (const auto& entry : layout_spec.tensors()) {
                  if (entry.second.overlap_mode() == tensorcast::layout::v1::OVERLAP_MODE_REPLICATE_EQUAL) {
                    replicated_tensors.insert(entry.first);
                  }
                }

                if (!replicated_tensors.empty()) {
                  if (proof_schema_version.empty()) {
                    final_status =
                        absl::FailedPreconditionError("proof_schema_version required for replicated tensors");
                  } else if (proof_schema_version != "v1") {
                    final_status = absl::UnimplementedError("unsupported proof_schema_version");
                  } else {
                    auto index_or = client_sp->get_artifact_index_by_id(sealed_artifact_id);
                    if (!index_or.ok()) {
                      final_status = index_or.status();
                    } else {
                      auto intervals_or = parse_tensor_intervals(*index_or);
                      if (!intervals_or.ok()) {
                        final_status = intervals_or.status();
                      } else {
                        auto resident_devices = engine->get_resident_devices(sealed_artifact_id);
                        auto gpu_it = std::find_if(
                            resident_devices.begin(), resident_devices.end(), [](const store::DeviceKey& d) {
                              return d.type == DeviceType::GPU;
                            });
                        if (gpu_it == resident_devices.end()) {
                          final_status =
                              absl::FailedPreconditionError("sealed artifact GPU replica unavailable for proofs");
                        } else {
                          store::loading::ReplicaKey replica_key;
                          replica_key.artifact_id = sealed_artifact_id;
                          replica_key.view_id = std::nullopt;
                          replica_key.device = *gpu_it;
                          replica_key.replica = 0;

                          auto size_or = engine->get_replica_size(replica_key);
                          auto ptr_or = engine->get_replica_gpu_ptr(replica_key);
                          if (!size_or.ok()) {
                            final_status = size_or.status();
                          } else if (!ptr_or.ok()) {
                            final_status = ptr_or.status();
                          } else {
                            store::loader::GpuMemorySource src(
                                gsl::not_null<void*>{reinterpret_cast<void*>(*ptr_or)},
                                /*device_id=*/gpu_it->ordinal,
                                *size_or);

                            std::vector<tensorcast::global_store::v1::TensorProofCommitmentWrite> writes;
                            for (const auto& interval : *intervals_or) {
                              if (interval.size_bytes == 0) {
                                continue;
                              }
                              if (!replicated_tensors.contains(interval.tensor_name)) {
                                continue;
                              }
                              const uint64_t expected_chunks =
                                  (interval.size_bytes + kProofChunkBytesV1 - 1) / kProofChunkBytesV1;
                              for (uint64_t chunk_idx = 0; chunk_idx < expected_chunks; ++chunk_idx) {
                                const uint64_t local_start = chunk_idx * kProofChunkBytesV1;
                                const uint64_t local_end =
                                    std::min<uint64_t>(interval.size_bytes, local_start + kProofChunkBytesV1);
                                if (local_end <= local_start) {
                                  continue;
                                }
                                const uint64_t abs_start = interval.offset + local_start;
                                const uint64_t read_len = local_end - local_start;
                                if (read_len > std::numeric_limits<size_t>::max()) {
                                  final_status = absl::OutOfRangeError("proof chunk exceeds host memory limits");
                                  break;
                                }
                                std::vector<uint8_t> buffer(static_cast<size_t>(read_len));
                                auto read_or = src.read_at(abs_start, buffer.data(), static_cast<size_t>(read_len));
                                if (!read_or.ok()) {
                                  final_status = read_or.status();
                                  break;
                                }
                                if (*read_or != buffer.size()) {
                                  final_status = absl::DataLossError("short read while computing MI2 proof digest");
                                  break;
                                }
                                std::vector<uint8_t> digest =
                                    tensorcast::common::sha256_digest_bytes(absl::MakeSpan(buffer));
                                if (digest.size() != 32) {
                                  final_status = absl::InternalError("sha256 digest size mismatch");
                                  break;
                                }
                                tensorcast::global_store::v1::TensorProofCommitmentWrite write;
                                write.set_tensor_name(interval.tensor_name);
                                write.set_proof_chunk_idx(chunk_idx);
                                write.set_digest(digest.data(), static_cast<int>(digest.size()));
                                writes.push_back(std::move(write));
                              }
                              if (!final_status.ok()) {
                                break;
                              }
                            }

                            if (final_status.ok() && !writes.empty()) {
                              constexpr size_t kBatchEntries = 1024;
                              for (size_t i = 0; i < writes.size(); i += kBatchEntries) {
                                tensorcast::global_store::v1::WriteTensorProofCommitmentsRequest write_req;
                                write_req.set_mi2_id(sealed_artifact_id);
                                write_req.set_proof_schema_version(proof_schema_version);
                                const size_t end = std::min(writes.size(), i + kBatchEntries);
                                for (size_t j = i; j < end; ++j) {
                                  *write_req.add_commitments() = writes[j];
                                }
                                auto write_resp_or = client_sp->write_tensor_proof_commitments(write_req);
                                if (!write_resp_or.ok()) {
                                  final_status = write_resp_or.status();
                                  break;
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }

            if (final_status.ok()) {
              const auto& policy = post_seal_policy;
              const bool allow_migration = policy.migrate_views;
              const bool allow_retire = policy.retire_pieces;
              if (allow_retire && !policy.migrate_views && !policy.reuse_views_if_safe) {
                LOG(WARNING) << "post-seal retire_pieces enabled without migrate_views or reuse_views_if_safe; "
                             << "view reads may fail after seal";
              }

              if (allow_migration) {
                auto views_or = client_sp->list_views(assembly_id);
                if (!views_or.ok()) {
                  LOG(WARNING) << "post-seal migrate_views list_views failed for assembly=" << assembly_id << ": "
                               << views_or.status();
                } else {
                  absl::flat_hash_set<std::string> expected_set;
                  if (layout_spec_for_post_seal.has_value()) {
                    const auto& expected = layout_spec_for_post_seal->expected_view_ids();
                    expected_set.reserve(static_cast<size_t>(expected.size()));
                    for (const auto& id : expected) {
                      if (!id.empty()) {
                        expected_set.insert(id);
                      }
                    }
                  }

                  if (engine->get_num_gpus() == 0) {
                    LOG(WARNING) << "post-seal migrate_views skipped: no GPU devices available";
                  } else {
                    for (const auto& view : *views_or) {
                      if (view.view_id.empty()) {
                        continue;
                      }
                      if (!expected_set.empty() && !expected_set.contains(view.view_id)) {
                        continue;
                      }
                      if (view.view_spec_json.empty()) {
                        LOG(WARNING) << "post-seal migrate_views skipping view_id=" << view.view_id
                                     << " (missing view_spec_json)";
                        continue;
                      }
                      if (view.view_size_bytes == 0) {
                        LOG(WARNING) << "post-seal migrate_views skipping view_id=" << view.view_id
                                     << " (view_size_bytes=0)";
                        continue;
                      }
                      if (policy.migrate_transpose_only) {
                        auto spec_or = store::view::parse_view_spec_json(view.view_spec_json);
                        if (!spec_or.ok()) {
                          LOG(WARNING) << "post-seal migrate_views skipping view_id=" << view.view_id
                                       << " (invalid view_spec_json): " << spec_or.status();
                          continue;
                        }
                        if (!spec_includes_transpose(*spec_or)) {
                          continue;
                        }
                      }

                      const store::DeviceKey target_device = devices->DefaultGpu();
                      auto handle_or = engine->materialize_view_from_assembly(
                          assembly_id,
                          sealed_artifact_id,
                          view.view_id,
                          view.view_spec_json,
                          target_device,
                          store::loading::TransformPlacement::kServer,
                          nullptr);
                      if (!handle_or.ok()) {
                        LOG(WARNING) << "post-seal migrate_views failed for view_id=" << view.view_id << ": "
                                     << handle_or.status();
                        continue;
                      }

                      auto publish_status = engine->register_replica_with_global_store(handle_or->replica_key, {});
                      if (!publish_status.ok() && !absl::IsAlreadyExists(publish_status)) {
                        LOG(WARNING) << "post-seal migrate_views register_replica failed for view_id=" << view.view_id
                                     << ": " << publish_status;
                      }

                      store::components::ViewStateUpdate update;
                      update.artifact_id = sealed_artifact_id;
                      update.view_id = view.view_id;
                      update.view_spec_json = view.view_spec_json;
                      update.view_size_bytes = view.view_size_bytes;
                      if (view.view_data_hash.has_value()) {
                        update.view_data_hash = view.view_data_hash;
                      }
                      update.mark_verified = view.verified_at.has_value();
                      update.canonical_size_bytes = view.canonical_size_bytes;
                      update.canonical_bytes_covered = view.canonical_bytes_covered;
                      update.canonical_ranges = view.canonical_ranges;
                      auto view_status = client_sp->update_artifact_view_state(update);
                      if (!view_status.ok()) {
                        LOG(WARNING) << "post-seal migrate_views update_view_state failed for view_id=" << view.view_id
                                     << ": " << view_status;
                      }
                    }
                  }
                }
              }

              if (allow_retire) {
                const std::string worker_id = identity->worker_id();
                if (!worker_id.empty()) {
                  auto unreg_status = client_sp->unregister_replica_by_worker(assembly_id, worker_id);
                  if (!unreg_status.ok()) {
                    LOG(WARNING) << "post-seal retire_pieces unregister_replica_by_worker failed for assembly="
                                 << assembly_id << ": " << unreg_status;
                  }
                } else {
                  LOG(WARNING) << "post-seal retire_pieces skipped unregister_replica_by_worker: worker_id unavailable";
                }

                std::vector<store::loading::ReplicaKey> to_unload;
                for (const auto& info : engine->get_all_replicas_info()) {
                  if (info.key.artifact_id == assembly_id) {
                    to_unload.push_back(info.key);
                  }
                }
                for (const auto& key : to_unload) {
                  auto unload_status = engine->unload_replica_status(key);
                  if (!unload_status.ok()) {
                    LOG(WARNING) << "post-seal retire_pieces unload_replica failed for key=" << key << ": "
                                 << unload_status;
                  }
                }
              }
            }

            tensorcast::daemon::v2::SealAssemblyResult result_msg;
            populate_artifact_descriptor_from_seal_result(*seal_or, result_msg.mutable_artifact());

            tensorcast::operation::v1::UpdateOperationRequest success;
            success.set_operation_id(operation_id);
            success.set_lease_generation(lease_generation);
            auto* out = success.mutable_status();
            out->set_state(tensorcast::operation::v1::OPERATION_STATE_SUCCESS);
            out->set_message("sealed");
            out->set_progress(1.0);
            *out->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
            out->mutable_result()->PackFrom(result_msg);
            if (final_status.ok()) {
              final_status = client_sp->update_operation(success);
            }
          }

          if (!final_status.ok()) {
            tensorcast::operation::v1::UpdateOperationRequest failed;
            failed.set_operation_id(operation_id);
            failed.set_lease_generation(lease_generation);
            auto* out = failed.mutable_status();
            out->set_state(tensorcast::operation::v1::OPERATION_STATE_FAILED);
            out->set_message("seal failed");
            out->set_progress(0.0);
            *out->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
            auto* err = out->mutable_error();
            err->set_status_code(absl::StatusCodeToString(final_status.code()));
            err->set_message(std::string(final_status.message()));
            err->set_retryable(retryable_status(final_status));
            absl::Status update_st = client_sp->update_operation(failed);
            if (!update_st.ok()) {
              LOG(WARNING) << "update_operation(FAILED) failed for op=" << operation_id << ": " << update_st;
            }
          }

          keepalive_stop->store(true, std::memory_order_relaxed);
          lease_guard.release();
        });
  }

  rctx.mark_success();
  return Status::OK;
}

grpc::Status AssemblyOperationService::get_operation(
    RpcContext& rctx,
    const tensorcast::operation::v1::GetOperationRequest& req,
    tensorcast::operation::v1::GetOperationResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.operation.id", req.operation_id());

  if (req.operation_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "operation_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  auto op_or = d_.global_store_client->get_operation(req);
  if (!op_or.ok()) {
    return to_grpc_status(op_or.status());
  }
  resp = std::move(*op_or);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status AssemblyOperationService::wait_operation(
    RpcContext& rctx,
    const v2::WaitOperationRequest& req,
    v2::WaitOperationResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.operation.id", req.operation_id());

  if (req.operation_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "operation_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  const uint64_t timeout_ms = req.timeout_ms();
  const absl::Time start = absl::Now();
  const absl::Time deadline = timeout_ms > 0 ? start + absl::Milliseconds(timeout_ms) : absl::InfiniteFuture();

  absl::Duration sleep = absl::Milliseconds(50);
  tensorcast::operation::v1::GetOperationRequest op_req;
  op_req.set_operation_id(req.operation_id());
  if (req.has_ref()) {
    op_req.mutable_ref()->CopyFrom(req.ref());
  }

  while (absl::Now() < deadline) {
    auto op_or = d_.global_store_client->get_operation(op_req);
    if (!op_or.ok()) {
      return to_grpc_status(op_or.status());
    }
    const auto state = op_or->status().state();
    resp.mutable_operation()->Swap(&(*op_or));
    if (state == tensorcast::operation::v1::OPERATION_STATE_SUCCESS ||
        state == tensorcast::operation::v1::OPERATION_STATE_FAILED ||
        state == tensorcast::operation::v1::OPERATION_STATE_CANCELLED) {
      rctx.mark_success();
      return Status::OK;
    }
    absl::SleepFor(sleep);
    sleep = std::min(sleep * 12 / 10, absl::Milliseconds(500));
  }

  auto op_or = d_.global_store_client->get_operation(op_req);
  if (!op_or.ok()) {
    return to_grpc_status(op_or.status());
  }
  resp.mutable_operation()->Swap(&(*op_or));
  rctx.mark_success();
  return Status::OK;
}

} // namespace tensorcast::daemon
