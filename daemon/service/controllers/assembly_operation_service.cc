// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/assembly_operation_service.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
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
#include "core/common/artifact_identity.h"
#include "core/common/trace/trace_macros.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/device_guard.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/sources/memory_source.h"
#include "daemon/service/controllers/assembly_closeout_identity_utils.h"
#include "daemon/service/controllers/assembly_coordination_utils.h"
#include "daemon/service/controllers/materialization_layout_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/service/controllers/materialization_post_seal_utils.h"
#include "daemon/service/controllers/serving_artifact_manifest_utils.h"
#include "daemon/state/lip_metadata_utils.h"
#include "daemon/util/status_utils.h"
#include "folly/futures/Future.h"
#include "google/protobuf/util/time_util.h"
#include "gsl/pointers"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
namespace coordination = assembly_coordination;
namespace closeout_identity = assembly_closeout_identity;
namespace pubv1 = tensorcast::publication::v1;
using materialization_layout::parse_canonical_index;
using materialization_layout::resolve_target_offsets;
namespace serving_manifest = serving_artifact_manifest;
using materialization_policy::build_execution_diagnostics;
using materialization_policy::HashExecutionDetails;
using materialization_policy::spec_includes_transpose;
using materialization_post_seal::check_post_seal_view_reuse_safe;
using materialization_post_seal::compute_view_meta_digest;
using status_utils::to_grpc_status;

namespace {
std::string_view binding_seal_identity_canonical_index_json(const BindingRegistry::Record& record) {
  if (record.sealed_commit_result.has_value() && !record.sealed_commit_result->canonical_index_json.empty()) {
    return record.sealed_commit_result->canonical_index_json;
  }
  if (!record.current_artifact_canonical_index_json.empty()) {
    return record.current_artifact_canonical_index_json;
  }
  return std::string_view();
}

std::string_view binding_tensor_canonical_index_json(const BindingRegistry::Record& record) {
  return record.target_index_json;
}

absl::Status validate_seal_identity_index_matches_binding_layout(
    std::string_view identity_index_json,
    std::string_view binding_index_json) {
  if (identity_index_json == binding_index_json) {
    return absl::OkStatus();
  }
  auto identity_or = parse_canonical_index(identity_index_json);
  if (!identity_or.ok()) {
    return identity_or.status();
  }
  auto binding_or = parse_canonical_index(binding_index_json);
  if (!binding_or.ok()) {
    return binding_or.status();
  }
  if (identity_or->logical_total_size != binding_or->logical_total_size) {
    return absl::FailedPreconditionError(
        "canonical_full binding layout index total size does not match source identity index");
  }
  if (identity_or->entries.size() != binding_or->entries.size()) {
    return absl::FailedPreconditionError(
        "canonical_full binding layout tensor set does not match source identity index");
  }
  for (const auto& [name, identity_entry] : identity_or->entries) {
    const auto binding_it = binding_or->entries.find(name);
    if (binding_it == binding_or->entries.end()) {
      return absl::FailedPreconditionError(absl::StrCat("canonical_full binding layout missing source tensor=", name));
    }
    const auto& binding_entry = binding_it->second;
    if (identity_entry.logical_offset != binding_entry.logical_offset ||
        identity_entry.logical_length != binding_entry.logical_length || identity_entry.dtype != binding_entry.dtype ||
        identity_entry.shape != binding_entry.shape || identity_entry.stride != binding_entry.stride) {
      return absl::FailedPreconditionError(
          absl::StrCat("canonical_full binding layout entry does not match source identity tensor=", name));
    }
  }
  return absl::OkStatus();
}

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

v2::AssemblyTargetRef publication_target_to_v2(const tensorcast::publication::v1::AssemblyTargetRef& source) {
  v2::AssemblyTargetRef target;
  switch (source.kind()) {
    case tensorcast::publication::v1::ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW:
      target.set_kind(v2::ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW);
      if (!source.structural_view_id().empty()) {
        target.set_structural_view_id(source.structural_view_id());
      }
      break;
    case tensorcast::publication::v1::ASSEMBLY_TARGET_KIND_CANONICAL_LAYOUT:
      target.set_kind(v2::ASSEMBLY_TARGET_KIND_CANONICAL_LAYOUT);
      break;
    default:
      break;
  }
  return target;
}

v2::AssemblyRequirementSetRef publication_requirements_to_v2(
    const tensorcast::publication::v1::AssemblyRequirementSetRef& source) {
  v2::AssemblyRequirementSetRef requirements;
  if (!source.requirements_digest().empty()) {
    requirements.set_requirements_digest(source.requirements_digest());
  }
  requirements.set_requirement_count(source.requirement_count());
  if (!source.carrier_form().empty()) {
    requirements.set_carrier_form(source.carrier_form());
  }
  for (const auto& item : source.inline_requirements()) {
    auto* requirement = requirements.add_inline_requirements();
    requirement->set_slot_id(item.slot_id());
    *requirement->mutable_target() = publication_target_to_v2(item.target());
    requirement->set_coverage_contract(item.coverage_contract());
  }
  return requirements;
}

v2::AssemblyReadinessPolicy publication_readiness_to_v2(
    const tensorcast::publication::v1::AssemblyReadinessPolicy& source) {
  v2::AssemblyReadinessPolicy readiness_policy;
  switch (source.contributor_liveness_mode()) {
    case tensorcast::publication::v1::ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_REQUIRE_LIVE_UNTIL_CUT:
      readiness_policy.set_contributor_liveness_mode(v2::ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_REQUIRE_LIVE_UNTIL_CUT);
      break;
    case tensorcast::publication::v1::ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_ALLOW_DURABLE_OCCUPANCY:
      readiness_policy.set_contributor_liveness_mode(v2::ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_ALLOW_DURABLE_OCCUPANCY);
      break;
    default:
      break;
  }
  return readiness_policy;
}

v2::AssemblyCloseoutContract publication_closeout_to_v2(
    const tensorcast::publication::v1::RepresentationPublishSpec& source) {
  v2::AssemblyCloseoutContract closeout_contract;
  closeout_contract.set_kind(v2::ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH);
  if (!source.source_version_key().empty()) {
    closeout_contract.set_source_version_key(source.source_version_key());
  }
  if (!source.serving_version_key().empty()) {
    closeout_contract.set_serving_version_key(source.serving_version_key());
  }
  const auto& contract = source.representation_publish_contract();
  if (contract.has_subject()) {
    closeout_contract.mutable_representation_publish_contract()->mutable_subject()->CopyFrom(contract.subject());
  }
  if (!contract.serving_manifest_ref().empty()) {
    closeout_contract.set_serving_manifest_ref(contract.serving_manifest_ref());
  }
  auto* typed = closeout_contract.mutable_representation_publish_contract();
  if (contract.has_subject() && !typed->has_subject()) {
    typed->mutable_subject()->CopyFrom(contract.subject());
  }
  if (!contract.serving_manifest_ref().empty()) {
    typed->set_serving_manifest_ref(contract.serving_manifest_ref());
  }
  if (!contract.representation_contract_hash().empty()) {
    typed->set_representation_contract_hash(contract.representation_contract_hash());
  }
  if (!contract.serving_build_digest().empty()) {
    typed->set_serving_build_digest(contract.serving_build_digest());
  }
  if (!contract.serving_build_digest_version().empty()) {
    typed->set_serving_build_digest_version(contract.serving_build_digest_version());
  }
  return closeout_contract;
}

absl::Status validate_representation_publish_admission_facts(
    const tensorcast::publication::v1::RepresentationPublishSpec& spec) {
  if (!spec.has_admission_facts()) {
    if (!spec.serving_manifest_bytes().empty()) {
      auto manifest_or = serving_manifest::parse_serving_manifest_payload(spec.serving_manifest_bytes());
      if (!manifest_or.ok()) {
        return manifest_or.status();
      }
      if (manifest_or->builder_mode == "binding_finalize") {
        return absl::FailedPreconditionError("binding_finalize representation_publish requires admission_facts");
      }
    }
    return absl::OkStatus();
  }
  if (spec.serving_manifest_bytes().empty()) {
    return absl::InvalidArgumentError("representation_publish admission_facts require serving_manifest_bytes");
  }

  auto manifest_or = serving_manifest::parse_serving_manifest_payload(spec.serving_manifest_bytes());
  if (!manifest_or.ok()) {
    return manifest_or.status();
  }

  const auto& admission = spec.admission_facts();
  if (admission.finalize_class() == tensorcast::publication::v1::FINALIZE_CLASS_UNSPECIFIED) {
    return absl::InvalidArgumentError("representation_publish admission_facts require finalize_class");
  }
  if (admission.support_level() == tensorcast::publication::v1::SERVING_SUPPORT_LEVEL_UNSPECIFIED) {
    return absl::InvalidArgumentError("representation_publish admission_facts require support_level");
  }
  if (admission.finalize_class() == tensorcast::publication::v1::FINALIZE_CLASS_UNKNOWN_BLOCKED) {
    return absl::FailedPreconditionError(
        "representation_publish admission_facts do not admit finalize_class=unknown_blocked");
  }
  if (admission.support_level() != tensorcast::publication::v1::SERVING_SUPPORT_LEVEL_BUILDER_PUBLICATION_READY &&
      admission.support_level() != tensorcast::publication::v1::SERVING_SUPPORT_LEVEL_RUNTIME_BIND_SWAP_READY) {
    return absl::FailedPreconditionError(
        "representation_publish admission_facts require support_level to admit builder publication");
  }
  if (!spec.serving_version_key().empty() &&
      admission.support_level() != tensorcast::publication::v1::SERVING_SUPPORT_LEVEL_RUNTIME_BIND_SWAP_READY) {
    return absl::FailedPreconditionError(
        "representation_publish admission_facts require support_level=runtime_bind_swap_ready for serving_version_key activation");
  }
  if (manifest_or->builder_mode == "pure_transform" &&
      admission.finalize_class() == tensorcast::publication::v1::FINALIZE_CLASS_REPRESENTATION_CHANGING) {
    return absl::FailedPreconditionError(
        "representation_publish admission_facts do not allow finalize_class=representation_changing with builder_mode=pure_transform");
  }
  if (manifest_or->builder_mode == "binding_finalize") {
    if (admission.finalize_class() != tensorcast::publication::v1::FINALIZE_CLASS_REPRESENTATION_CHANGING) {
      return absl::FailedPreconditionError(
          "representation_publish admission_facts require finalize_class=representation_changing with builder_mode=binding_finalize");
    }
    if (!admission.same_binding_fast_path_validated()) {
      return absl::FailedPreconditionError(
          "binding_finalize representation_publish requires same_binding_fast_path_validated=true");
    }
    if (!spec.representation_publish_contract().has_subject() ||
        spec.representation_publish_contract().subject().ref_case() !=
            tensorcast::publication::v1::ServingPublicationSubject::kBindingValue) {
      return absl::FailedPreconditionError(
          "binding_finalize representation_publish requires binding_value_ref subject");
    }
  }
  return absl::OkStatus();
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

absl::StatusOr<std::vector<store::StoreEngine::SealAssemblyCutInput::BoundCanonicalSpan>> build_bound_canonical_spans(
    const BindingRegistry::Record& record) {
  if (record.ownership != v2::BINDING_OWNERSHIP_DAEMON) {
    return absl::FailedPreconditionError("bound canonical source requires daemon-owned binding");
  }
  if (record.allocation == nullptr || record.allocation->get() == nullptr) {
    return absl::FailedPreconditionError("bound canonical source requires live binding allocation");
  }

  const std::string_view canonical_index_json = binding_tensor_canonical_index_json(record);
  auto index_or = parse_canonical_index(canonical_index_json);
  if (!index_or.ok()) {
    return index_or.status();
  }
  auto offsets_or = resolve_target_offsets(record.target_layout);
  if (!offsets_or.ok()) {
    return offsets_or.status();
  }

  absl::flat_hash_map<std::string, const v2::StorageEntry*> storages_by_id;
  storages_by_id.reserve(record.target_layout.storages_size());
  for (const auto& storage : record.target_layout.storages()) {
    storages_by_id.emplace(storage.storage_id(), &storage);
  }

  std::vector<store::StoreEngine::SealAssemblyCutInput::BoundCanonicalSpan> spans;
  spans.reserve(offsets_or->size());
  const uint64_t base_ptr = reinterpret_cast<uint64_t>(record.allocation->get());
  for (const auto& offset : *offsets_or) {
    const auto storage_it = storages_by_id.find(offset.storage_id);
    if (storage_it == storages_by_id.end()) {
      return absl::FailedPreconditionError(
          absl::StrCat("bound canonical source missing storage_id=", offset.storage_id));
    }
    const auto entry_it = index_or->entries.find(offset.name);
    if (entry_it == index_or->entries.end()) {
      return absl::FailedPreconditionError(absl::StrCat("bound canonical source missing tensor entry=", offset.name));
    }
    const auto& storage = *storage_it->second;
    const auto& entry = entry_it->second;
    if (entry.logical_length != offset.logical_length) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "bound canonical source logical length mismatch for tensor=",
              offset.name,
              " expected=",
              entry.logical_length,
              " actual=",
              offset.logical_length));
    }
    if (offset.storage_offset + offset.logical_length > storage.storage_length()) {
      return absl::FailedPreconditionError(
          absl::StrCat("bound canonical source exceeds storage for tensor=", offset.name));
    }
    spans.push_back(
        store::StoreEngine::SealAssemblyCutInput::BoundCanonicalSpan{
            .device_id = storage.device_id(),
            .base_ptr = base_ptr,
            .mapping_base_offset = storage.mapping_base_offset(),
            .storage_offset = offset.storage_offset,
            .logical_offset = entry.logical_offset,
            .logical_length = offset.logical_length,
        });
  }

  std::sort(spans.begin(), spans.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.logical_offset < rhs.logical_offset;
  });
  uint64_t cursor = 0;
  for (const auto& span : spans) {
    if (span.logical_length == 0) {
      return absl::FailedPreconditionError("bound canonical source contains zero-length span");
    }
    if (span.logical_offset != cursor) {
      return absl::FailedPreconditionError("bound canonical source coverage is not contiguous");
    }
    cursor += span.logical_length;
  }
  if (cursor != index_or->logical_total_size) {
    return absl::FailedPreconditionError("bound canonical source size does not match canonical index");
  }
  return spans;
}

absl::StatusOr<store::StoreEngine::SealAssemblyCutInput> build_seal_cut_input(
    BindingRegistry& bindings,
    const v2::AssemblyReadinessCut& readiness_cut,
    store::components::IGlobalStoreClient* global_store_client) {
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
  if (!input.canonical_full) {
    return input;
  }

  const auto slot_it = std::find_if(
      readiness_cut.live_slots().begin(),
      readiness_cut.live_slots().end(),
      [](const v2::AssemblyReadinessCutSlot& slot) {
        return slot.slot_id() == coordination::kCanonicalFullContributionSlotKey;
      });
  if (slot_it == readiness_cut.live_slots().end()) {
    return absl::FailedPreconditionError("canonical_full readiness cut is missing canonical slot");
  }
  if (slot_it->binding_id().empty() || slot_it->binding_value_id().empty()) {
    return absl::FailedPreconditionError("canonical_full slot is missing binding identity");
  }

  auto record_or = bindings.get(slot_it->binding_id());
  if (!record_or.ok()) {
    return record_or.status();
  }
  const auto& record = *record_or;
  std::string binding_index_json;
  std::string identity_index_json;
  {
    absl::MutexLock lock(&record->mu);
    if (record->closed) {
      return absl::FailedPreconditionError("canonical_full binding is closed");
    }
    if (record->current_binding_value_id != slot_it->binding_value_id()) {
      return absl::FailedPreconditionError("canonical_full binding value changed after readiness cut");
    }
    if (record->ownership == v2::BINDING_OWNERSHIP_DAEMON) {
      auto spans_or = build_bound_canonical_spans(*record);
      if (!spans_or.ok()) {
        return spans_or.status();
      }
      binding_index_json = std::string(binding_tensor_canonical_index_json(*record));
      const std::string_view local_identity_index_json = binding_seal_identity_canonical_index_json(*record);
      if (!local_identity_index_json.empty()) {
        identity_index_json = std::string(local_identity_index_json);
      }
      if (record->sealed_commit_result.has_value() && !record->sealed_commit_result->artifact_id.empty()) {
        input.canonical_artifact_id = record->sealed_commit_result->artifact_id;
      } else if (!record->current_artifact_id.empty()) {
        input.canonical_artifact_id = record->current_artifact_id;
      }
      input.bound_canonical_spans = std::move(*spans_or);
    }
  }
  if (!input.bound_canonical_spans.empty()) {
    if (identity_index_json.empty()) {
      if (input.canonical_artifact_id.empty()) {
        identity_index_json = binding_index_json;
      } else {
        if (global_store_client == nullptr || !global_store_client->is_connected()) {
          return absl::FailedPreconditionError(
              "canonical_full source identity index lookup requires Global Store client");
        }
        auto identity_index_or = global_store_client->get_artifact_index_by_id(input.canonical_artifact_id);
        if (!identity_index_or.ok()) {
          return identity_index_or.status();
        }
        identity_index_json = std::move(*identity_index_or);
      }
    }
    auto compatible_status =
        validate_seal_identity_index_matches_binding_layout(identity_index_json, binding_index_json);
    if (!compatible_status.ok()) {
      return compatible_status;
    }
    input.canonical_index_json = std::move(identity_index_json);
  }
  return input;
}

bool retryable_status(const absl::Status& st) {
  return absl::IsUnavailable(st) || absl::IsDeadlineExceeded(st) || absl::IsAborted(st) || absl::IsInternal(st) ||
      absl::IsUnknown(st);
}

absl::Status ensure_local_readable_source_artifact(store::StoreEngine& engine, std::string_view artifact_id) {
  SC_TRACE_SCOPE("assembly_attempt.ensure_local_readable_source_artifact");
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
  absl::StatusOr<store::loading::ReplicaHandle> handle_or = absl::UnknownError("uninitialized");
  {
    SC_TRACE_SCOPE("assembly_attempt.ensure_local_readable_source_artifact.materialize_cpu");
    handle_or = engine.materialize_replica(
        store::DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
        store::StoreEngine::MaterializeMode::AUTO,
        hints);
  }
  if (!handle_or.ok()) {
    return handle_or.status();
  }
  return absl::OkStatus();
}

absl::Status await_state_sync_barrier(const std::function<absl::Status()>& barrier) {
  if (!barrier) {
    return absl::OkStatus();
  }
  return barrier();
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

struct RepresentationPublishValidationResult {
  tensorcast::common::v1::ArtifactDescriptor serving_artifact;
  std::string serving_manifest_ref;
  std::string representation_contract_hash;
  std::string serving_build_digest;
  std::optional<v2::ExecutionDiagnostics> serving_execution_diagnostics;
};

struct BoundCloseoutSnapshot {
  std::string binding_id;
  std::string binding_layout_id;
  std::string binding_value_id;
  uint64_t seal_generation{0};
  int device_id{-1};
  int owner_pid{0};
  v2::TargetLayout target_layout;
  std::string target_index_json;
  cuda::IpcHandleBytes handle_bytes;
  common::memory::GpuDeviceMemory* allocation{nullptr};
  std::optional<CommitLeaseResult> sealed_commit_result;
  std::vector<store::runtime::ingestion::MaterializationFacade::SealAssemblyCutInput::BoundCanonicalSpan>
      bound_canonical_spans;
};

struct CloseoutOwnedStorageLayout {
  std::vector<LeaseSegMeta> publish_segments;
  std::vector<RegisterStorageMeta> publish_storages;
  uint64_t total_size{0};
};

absl::StatusOr<CloseoutOwnedStorageLayout> build_closeout_owned_storage_layout(
    const v2::TargetLayout& layout,
    int expected_device_id,
    const cuda::IpcHandleBytes& handle_bytes) {
  CloseoutOwnedStorageLayout result;
  const auto handle_view = handle_bytes.as_string_view();
  uint64_t cursor = 0;
  result.publish_segments.reserve(layout.storages_size());
  result.publish_storages.reserve(layout.storages_size());
  for (const auto& storage : layout.storages()) {
    if (storage.storage_id().empty()) {
      return absl::InvalidArgumentError("storage_id is required");
    }
    if (storage.storage_length() == 0) {
      return absl::InvalidArgumentError("storage_length must be non-zero");
    }
    if (storage.device_id() != expected_device_id) {
      return absl::InvalidArgumentError("storage.device_id does not match device_uuid");
    }

    RegisterStorageMeta meta;
    meta.storage_id = storage.storage_id();
    meta.device_id = storage.device_id();
    meta.handle_bytes = std::string(handle_view);
    meta.storage_length = storage.storage_length();
    meta.mapping_base_offset = cursor;
    result.publish_storages.push_back(std::move(meta));

    LeaseSegMeta seg;
    seg.storage_id = storage.storage_id();
    seg.storage_offset = 0;
    seg.artifact_offset = cursor;
    seg.length = storage.storage_length();
    result.publish_segments.push_back(std::move(seg));

    cursor += storage.storage_length();
  }
  if (cursor == 0) {
    return absl::InvalidArgumentError("target_layout storages must be non-empty");
  }
  result.total_size = cursor;
  return result;
}

absl::StatusOr<std::vector<RegisterTensorAliasMeta>> build_closeout_binding_tensor_aliases(
    const BindingRegistry::Record& record) {
  auto offsets_or = resolve_target_offsets(record.target_layout);
  if (!offsets_or.ok()) {
    return offsets_or.status();
  }
  const std::string_view canonical_index_json = binding_tensor_canonical_index_json(record);
  auto index_or = parse_canonical_index(canonical_index_json);
  if (!index_or.ok()) {
    return index_or.status();
  }

  std::vector<RegisterTensorAliasMeta> aliases;
  aliases.reserve(offsets_or->size());
  for (const auto& offset : *offsets_or) {
    auto entry_it = index_or->entries.find(offset.name);
    if (entry_it == index_or->entries.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("binding target_index_json missing entry for tensor=", offset.name));
    }
    const auto& entry = entry_it->second;
    if (entry.logical_length != offset.logical_length) {
      return absl::InvalidArgumentError(absl::StrCat("binding logical_length mismatch for tensor=", offset.name));
    }
    RegisterTensorAliasMeta alias;
    alias.name = offset.name;
    alias.storage_id = offset.storage_id;
    alias.storage_offset = offset.storage_offset;
    alias.logical_length = offset.logical_length;
    alias.shape = entry.shape;
    alias.stride = entry.stride;
    alias.dtype = entry.dtype;
    aliases.push_back(std::move(alias));
  }
  return aliases;
}

tensorcast::common::v1::ArtifactSelection build_closeout_bound_selection(
    std::string_view artifact_id,
    const v2::TargetLayout& layout,
    std::string_view target_index_json) {
  tensorcast::common::v1::ArtifactSelection selection;
  selection.set_artifact_id(std::string(artifact_id));
  if (!layout.view_id().empty()) {
    selection.set_view_id(layout.view_id());
  }
  if (!layout.logical_layout_hash().empty()) {
    selection.set_logical_layout_hash(layout.logical_layout_hash());
  } else {
    const bool needs_view_index = layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW;
    const std::string logical_layout_hash = common::compute_logical_layout_hash_bytes(
        absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(target_index_json.data()), target_index_json.size()),
        needs_view_index);
    selection.set_logical_layout_hash(logical_layout_hash);
  }
  const std::string selection_hash = common::compute_selection_hash_bytes(selection.view_id(), std::nullopt);
  selection.set_selection_hash(selection_hash);
  return selection;
}

absl::StatusOr<BoundCloseoutSnapshot> capture_bound_closeout_snapshot(
    BindingRegistry* bindings,
    const pubv1::BindingValueRef& binding_value) {
  if (bindings == nullptr) {
    return absl::FailedPreconditionError("binding closeout requires BindingRegistry");
  }
  auto record_or = bindings->get(binding_value.binding_id());
  if (!record_or.ok()) {
    return record_or.status();
  }
  const auto record = *record_or;

  BoundCloseoutSnapshot snapshot;
  {
    absl::MutexLock lock(&record->mu);
    if (record->closed) {
      return absl::FailedPreconditionError("binding publication subject is closed");
    }
    if (record->ownership != v2::BINDING_OWNERSHIP_DAEMON) {
      return absl::FailedPreconditionError("binding publication subject requires daemon-owned binding");
    }
    if (record->binding_layout_id != binding_value.binding_layout_id()) {
      return absl::FailedPreconditionError("binding publication subject layout changed");
    }
    if (record->current_binding_value_id != binding_value.binding_value_id()) {
      return absl::FailedPreconditionError("binding publication subject is no longer current");
    }
    if (record->seal_generation != binding_value.seal_generation()) {
      return absl::FailedPreconditionError("binding publication subject seal_generation changed");
    }
    auto spans_or = build_bound_canonical_spans(*record);
    if (!spans_or.ok()) {
      return spans_or.status();
    }
    snapshot.binding_id = record->binding_id;
    snapshot.binding_layout_id = record->binding_layout_id;
    snapshot.binding_value_id = record->current_binding_value_id;
    snapshot.seal_generation = record->seal_generation;
    snapshot.device_id = record->device_id;
    snapshot.owner_pid = record->owner_pid;
    snapshot.target_layout = record->target_layout;
    snapshot.target_index_json = std::string(binding_tensor_canonical_index_json(*record));
    snapshot.handle_bytes = record->handle_bytes;
    snapshot.allocation = record->allocation.get();
    snapshot.sealed_commit_result = record->sealed_commit_result;
    snapshot.bound_canonical_spans = std::move(*spans_or);
  }
  return snapshot;
}

absl::StatusOr<std::string> read_bound_canonical_bytes(
    absl::Span<const store::runtime::ingestion::MaterializationFacade::SealAssemblyCutInput::BoundCanonicalSpan> spans,
    uint64_t logical_offset,
    uint64_t logical_length) {
  std::string payload(logical_length, '\0');
  uint64_t cursor = logical_offset;
  size_t remaining = static_cast<size_t>(logical_length);
  char* out = payload.data();
  while (remaining > 0) {
    const store::runtime::ingestion::MaterializationFacade::SealAssemblyCutInput::BoundCanonicalSpan* span = nullptr;
    for (const auto& candidate : spans) {
      if (cursor >= candidate.logical_offset && cursor < candidate.logical_offset + candidate.logical_length) {
        span = &candidate;
        break;
      }
    }
    if (span == nullptr) {
      return absl::FailedPreconditionError("bound canonical source is missing requested byte coverage");
    }
    const uint64_t local_offset = cursor - span->logical_offset;
    const size_t available = static_cast<size_t>(span->logical_length - local_offset);
    const size_t take = std::min(remaining, available);
    cuda::CudaDeviceGuard guard(span->device_id);
    if (!guard.status().ok()) {
      return guard.status();
    }
    const auto* src = reinterpret_cast<const uint8_t*>(span->base_ptr) +
        static_cast<std::ptrdiff_t>(span->mapping_base_offset + span->storage_offset + local_offset);
    auto memcpy_status = cuda::memcpy(out, src, take, cudaMemcpyDeviceToHost);
    if (!memcpy_status.ok()) {
      return memcpy_status;
    }
    if (auto sync = cuda::device_synchronize(); !sync.ok()) {
      return sync;
    }
    out += take;
    cursor += take;
    remaining -= take;
  }
  return payload;
}

void populate_artifact_descriptor_from_commit_lease(
    const CommitLeaseResult& out,
    tensorcast::common::v1::ArtifactDescriptor* desc) {
  if (desc == nullptr) {
    return;
  }
  desc->set_artifact_id(out.artifact_id);
  if (!out.index_multihash.empty()) {
    desc->set_index_multihash(out.index_multihash);
  }
  if (!out.data_multihash.empty()) {
    desc->set_data_multihash(out.data_multihash);
  }
  if (!out.schema_version.empty()) {
    desc->set_schema_version(out.schema_version);
  }
  if (!out.encoding.empty()) {
    desc->set_encoding(out.encoding);
  }
  desc->set_total_size(out.total_size);
  desc->set_id_kind(
      out.id_kind == tensorcast::common::ArtifactIdKind::kCgid
          ? tensorcast::common::v1::ArtifactIdKind::ARTIFACT_ID_KIND_CGID
          : tensorcast::common::v1::ArtifactIdKind::ARTIFACT_ID_KIND_MI2);
}

HashExecutionDetails build_hash_execution_details(
    const CommitLeaseResult* hash_commit_result,
    uint32_t hash_rounds,
    v2::HashLocation hash_location,
    v2::IdentityMintStrategy identity_mint_strategy) {
  HashExecutionDetails details;
  details.hash_rounds = hash_rounds;
  details.hash_location = hash_location;
  details.identity_mint_strategy = identity_mint_strategy;
  if (hash_commit_result == nullptr) {
    return details;
  }
  switch (hash_commit_result->hash_info.backend) {
    case CommitLeaseHashBackend::kGpu:
      details.hash_backend = v2::HashBackend::HASH_BACKEND_GPU;
      break;
    case CommitLeaseHashBackend::kD2HCpu:
      details.hash_backend = v2::HashBackend::HASH_BACKEND_D2H_CPU;
      break;
    case CommitLeaseHashBackend::kCpu:
      details.hash_backend = v2::HashBackend::HASH_BACKEND_CPU;
      break;
    case CommitLeaseHashBackend::kNone:
    default:
      details.hash_backend = v2::HashBackend::HASH_BACKEND_NONE;
      break;
  }
  details.hash_bytes = hash_commit_result->hash_info.bytes;
  details.hash_wall_time_ms = hash_commit_result->hash_info.wall_time_ms;
  details.hash_identity_forming = hash_commit_result->hash_info.identity_forming;
  return details;
}

absl::StatusOr<RepresentationPublishValidationResult> validate_artifact_subject_closeout(
    store::StoreEngine* engine,
    const v2::RepresentationPublishContract& representation_publish) {
  if (!representation_publish.has_subject() ||
      representation_publish.subject().ref_case() != pubv1::ServingPublicationSubject::kServingArtifactId) {
    return absl::InvalidArgumentError(
        "artifact-backed representation_publish closeout requires serving_artifact_id subject");
  }
  const std::string artifact_id = representation_publish.subject().serving_artifact_id();
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact-backed representation_publish subject requires serving_artifact_id");
  }
  auto serving_artifact_or = engine->get_artifact_descriptor(artifact_id);
  if (!serving_artifact_or.ok()) {
    return serving_artifact_or.status();
  }
  auto canonical_index_or = engine->get_canonical_index_by_id(artifact_id);
  if (!canonical_index_or.ok()) {
    return canonical_index_or.status();
  }
  auto preflight_or = serving_manifest::preflight_serving_artifact(
      engine,
      serving_manifest::ServingArtifactPreflightRequest{
          .artifact_id = artifact_id,
          .canonical_index_json = *canonical_index_or,
          .serving_manifest_ref = representation_publish.serving_manifest_ref(),
          .expected_representation_contract_hash = representation_publish.representation_contract_hash(),
          .expected_serving_build_digest = representation_publish.serving_build_digest(),
          .expected_serving_build_digest_version =
              std::string(representation_publish.serving_build_digest_version()).empty()
              ? std::nullopt
              : std::optional<std::string>(representation_publish.serving_build_digest_version()),
          .require_manifest = true,
      });
  if (!preflight_or.ok()) {
    return preflight_or.status();
  }

  RepresentationPublishValidationResult result;
  result.serving_artifact = *serving_artifact_or;
  result.serving_manifest_ref = preflight_or->serving_manifest_ref;
  result.representation_contract_hash = preflight_or->representation_contract_hash;
  result.serving_build_digest = preflight_or->serving_build_digest;
  result.serving_execution_diagnostics = build_execution_diagnostics(
      /*result=*/nullptr,
      v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE,
      store::loading::ExecutionTopologyContext{},
      HashExecutionDetails{});
  return result;
}

absl::StatusOr<RepresentationPublishValidationResult> validate_representation_publish_closeout(
    store::StoreEngine* engine,
    BindingRegistry* bindings,
    LipManager* lip_manager,
    SessionLifecycleManager* lifecycle,
    const v2::AssemblyCloseoutContract& closeout_contract) {
  ABSL_CHECK(engine != nullptr);
  if (!closeout_contract.has_representation_publish_contract()) {
    return absl::InvalidArgumentError(
        "representation_publish closeout contracts require representation_publish_contract");
  }

  const auto& representation_publish = closeout_contract.representation_publish_contract();
  if (!representation_publish.has_subject()) {
    return absl::InvalidArgumentError(
        "representation_publish closeout contracts require a serving publication subject");
  }
  const auto subject = representation_publish.subject();
  if (subject.ref_case() == pubv1::ServingPublicationSubject::kServingArtifactId) {
    return validate_artifact_subject_closeout(engine, representation_publish);
  }
  if (subject.ref_case() != pubv1::ServingPublicationSubject::kBindingValue) {
    return absl::InvalidArgumentError("representation_publish subject is not set");
  }
  (void)engine;
  (void)lip_manager;
  (void)lifecycle;
  auto snapshot_or = capture_bound_closeout_snapshot(bindings, subject.binding_value());
  if (!snapshot_or.ok()) {
    return snapshot_or.status();
  }
  auto manifest_tensor_name_or =
      serving_manifest::parse_tensor_manifest_ref(representation_publish.serving_manifest_ref());
  if (!manifest_tensor_name_or.ok()) {
    return manifest_tensor_name_or.status();
  }
  auto index_or = parse_canonical_index(snapshot_or->target_index_json);
  if (!index_or.ok()) {
    return index_or.status();
  }
  const auto entry_it = index_or->entries.find(*manifest_tensor_name_or);
  if (entry_it == index_or->entries.end()) {
    return absl::DataLossError(
        absl::StrCat(
            "serving artifact is missing manifest tensor referenced by serving_manifest_ref: ",
            *manifest_tensor_name_or));
  }
  auto payload_or = read_bound_canonical_bytes(
      snapshot_or->bound_canonical_spans, entry_it->second.logical_offset, entry_it->second.logical_length);
  if (!payload_or.ok()) {
    return payload_or.status();
  }
  auto preflight_or = serving_manifest::preflight_serving_manifest_payload(
      serving_manifest::ServingManifestPayloadPreflightRequest{
          .canonical_index_json = snapshot_or->target_index_json,
          .manifest_payload = *payload_or,
          .serving_manifest_ref = representation_publish.serving_manifest_ref(),
          .expected_representation_contract_hash = representation_publish.representation_contract_hash(),
          .expected_serving_build_digest = representation_publish.serving_build_digest(),
          .expected_serving_build_digest_version =
              std::string(representation_publish.serving_build_digest_version()).empty()
              ? std::nullopt
              : std::optional<std::string>(representation_publish.serving_build_digest_version()),
          .require_manifest = true,
      });
  if (!preflight_or.ok()) {
    return preflight_or.status();
  }

  RepresentationPublishValidationResult result;
  result.serving_manifest_ref = preflight_or->serving_manifest_ref;
  result.representation_contract_hash = preflight_or->representation_contract_hash;
  result.serving_build_digest = preflight_or->serving_build_digest;
  return result;
}

absl::StatusOr<RepresentationPublishValidationResult> finalize_binding_subject_closeout(
    const std::function<absl::Status()>& state_sync_barrier,
    BindingRegistry* bindings,
    LipManager* lip_manager,
    SessionLifecycleManager* lifecycle,
    std::shared_ptr<store::components::IGlobalStoreClient> client,
    WorkerIdentityStore* identity,
    std::string_view layout_id,
    const v2::RepresentationPublishContract& representation_publish,
    RepresentationPublishValidationResult prevalidated) {
  if (!representation_publish.has_subject() ||
      representation_publish.subject().ref_case() != pubv1::ServingPublicationSubject::kBindingValue) {
    return absl::InvalidArgumentError("binding subject finalization requires binding_value subject");
  }
  if (lip_manager == nullptr) {
    return absl::FailedPreconditionError("binding subject finalization requires LipManager");
  }
  if (client == nullptr || !client->is_connected()) {
    return absl::FailedPreconditionError("binding subject finalization requires Global Store client");
  }
  auto snapshot_or = capture_bound_closeout_snapshot(bindings, representation_publish.subject().binding_value());
  if (!snapshot_or.ok()) {
    return snapshot_or.status();
  }
  const auto& snapshot = *snapshot_or;
  if (snapshot.allocation == nullptr) {
    return absl::FailedPreconditionError("binding publication subject allocation is unavailable");
  }

  auto record_or = bindings->get(snapshot.binding_id);
  if (!record_or.ok()) {
    return record_or.status();
  }
  auto storage_layout_or =
      build_closeout_owned_storage_layout(snapshot.target_layout, snapshot.device_id, snapshot.handle_bytes);
  if (!storage_layout_or.ok()) {
    return storage_layout_or.status();
  }
  auto aliases_or = build_closeout_binding_tensor_aliases(**record_or);
  if (!aliases_or.ok()) {
    return aliases_or.status();
  }
  auto stable_index_or = build_canonical_index_from_metadata(
      absl::MakeSpan(storage_layout_or->publish_segments),
      absl::MakeSpan(storage_layout_or->publish_storages),
      absl::MakeSpan(*aliases_or),
      snapshot.device_id);
  if (!stable_index_or.ok()) {
    return stable_index_or.status();
  }

  const std::string registration_id = absl::StrCat(
      "binding-publication-closeout:",
      snapshot.binding_id,
      ":",
      snapshot.binding_value_id,
      ":",
      snapshot.seal_generation);

  struct LipRollback {
    LipManager* lip{nullptr};
    std::string registration_id;
    bool active{true};

    ~LipRollback() {
      if (!active || lip == nullptr) {
        return;
      }
      auto st = lip->revoke_by_registration_id(registration_id);
      if (!st.ok()) {
        LOG(WARNING) << "binding closeout rollback failed for registration_id=" << registration_id << ": " << st;
      }
    }

    void release() {
      active = false;
    }
  } rollback{.lip = lip_manager, .registration_id = registration_id};

  const uint64_t epoch = static_cast<uint64_t>(absl::ToUnixMillis(absl::Now()));
  auto out_or = lip_manager->commit_lease_in_place(
      registration_id,
      snapshot.device_id,
      snapshot.owner_pid,
      /*ttl_ms=*/0,
      epoch,
      storage_layout_or->total_size,
      tensorcast::common::ArtifactIdKind::kMi2,
      /*client_artifact_id=*/"",
      *stable_index_or,
      /*index_key_hex=*/"",
      std::move(storage_layout_or->publish_segments),
      std::move(storage_layout_or->publish_storages),
      std::move(*aliases_or),
      snapshot.sealed_commit_result);
  if (!out_or.ok()) {
    return out_or.status();
  }
  if (snapshot.sealed_commit_result.has_value()) {
    auto reuse_status =
        closeout_identity::validate_reused_identity_matches_closeout_result(*snapshot.sealed_commit_result, *out_or);
    if (!reuse_status.ok()) {
      return reuse_status;
    }
  }
  auto registration_index_json_or =
      closeout_identity::resolve_registration_canonical_index_json(snapshot.sealed_commit_result, *out_or);
  if (!registration_index_json_or.ok()) {
    return registration_index_json_or.status();
  }
  const std::string& registration_index_json = *registration_index_json_or;
  auto registration_index_status =
      closeout_identity::validate_registration_canonical_index_matches_commit_result(registration_index_json, *out_or);
  if (!registration_index_status.ok()) {
    return registration_index_status;
  }

  auto routable_or = lip_manager->publish_committed_lease_routable(registration_id);
  if (!routable_or.ok()) {
    return routable_or.status();
  }

  std::string worker_id = identity != nullptr ? identity->worker_id() : std::string();
  if (worker_id.empty()) {
    return absl::UnavailableError(
        absl::StrCat(
            "worker identity unavailable while finalizing binding closeout for artifact_id=",
            out_or->artifact_id,
            "; routable replicas require completed worker lifecycle registration"));
  }

  const store::DeviceKey device = store::DeviceRegistry::instance().gpu_key(snapshot.device_id);
  tensorcast::common::v1::ArtifactDescriptor descriptor;
  populate_artifact_descriptor_from_commit_lease(*out_or, &descriptor);
  auto replica_id_or = client->register_memory_replica_idempotent(
      out_or->artifact_id,
      worker_id,
      device,
      out_or->total_size,
      out_or->index_multihash,
      routable_or->remote_memory_keys,
      routable_or->buffer_sizes,
      registration_index_json,
      out_or->encoding,
      out_or->schema_version,
      /*max_concurrency=*/1,
      out_or->verification_json.empty() ? std::nullopt : std::optional<std::string>(out_or->verification_json),
      /*view_id=*/std::nullopt,
      descriptor,
      registration_id);
  if (!replica_id_or.ok()) {
    return replica_id_or.status();
  }
  lip_manager->attach_replica_id(registration_id, *replica_id_or);

  auto barrier_status = await_state_sync_barrier(state_sync_barrier);
  if (!barrier_status.ok()) {
    return barrier_status;
  }

  if (!layout_id.empty()) {
    SC_TRACE_SCOPE("assembly_attempt.attach_layout_to_serving_artifact");
    auto attach_status = client->attach_layout_to_artifact(out_or->artifact_id, std::string(layout_id));
    if (!attach_status.ok()) {
      return attach_status;
    }
  }

  const auto selection =
      build_closeout_bound_selection(out_or->artifact_id, snapshot.target_layout, snapshot.target_index_json);
  {
    const auto record = *record_or;
    absl::MutexLock lock(&record->mu);
    if (record->current_binding_value_id != snapshot.binding_value_id) {
      return absl::FailedPreconditionError("binding publication subject changed during closeout finalization");
    }
    if (record->seal_generation != snapshot.seal_generation) {
      return absl::FailedPreconditionError(
          "binding publication subject generation changed during closeout finalization");
    }
    record->current_artifact_id = out_or->artifact_id;
    record->current_artifact_canonical_index_json = registration_index_json;
    record->current_selection = selection;
    record->target_publication_token.clear();
    record->state = v2::BINDING_STATE_READY_ARTIFACT;
    record->active_update_epoch.clear();
  }

  if (lifecycle != nullptr) {
    SessionLifecycleManager::CommitSubject subj{.artifact_id = out_or->artifact_id, .device_id = snapshot.device_id};
    auto lid_or = lifecycle->create_commit_lease(subj, snapshot.owner_pid);
    if (!lid_or.ok()) {
      LOG(WARNING) << "create_commit_lease failed during representation closeout finalization: artifact_id="
                   << out_or->artifact_id << " dev=" << snapshot.device_id << ": " << lid_or.status();
    }
  }

  populate_artifact_descriptor_from_commit_lease(*out_or, &prevalidated.serving_artifact);
  const CommitLeaseResult* hash_commit_result =
      snapshot.sealed_commit_result.has_value() ? &*snapshot.sealed_commit_result : &*out_or;
  prevalidated.serving_execution_diagnostics = build_execution_diagnostics(
      /*result=*/nullptr,
      v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE,
      store::loading::ExecutionTopologyContext{},
      build_hash_execution_details(
          hash_commit_result,
          snapshot.sealed_commit_result.has_value() ? 0U : 1U,
          snapshot.sealed_commit_result.has_value() ? v2::HashLocation::HASH_LOCATION_SEAL
                                                    : v2::HashLocation::HASH_LOCATION_BINDING_CLOSEOUT,
          snapshot.sealed_commit_result.has_value() ? v2::IdentityMintStrategy::IDENTITY_MINT_STRATEGY_SEAL_REUSE
                                                    : v2::IdentityMintStrategy::IDENTITY_MINT_STRATEGY_CLOSEOUT_MINT));
  rollback.release();
  return prevalidated;
}

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
    const std::function<absl::Status()>& state_sync_barrier,
    store::StoreEngine* engine,
    BindingRegistry* bindings,
    LipManager* lip_manager,
    SessionLifecycleManager* lifecycle,
    std::shared_ptr<store::components::IGlobalStoreClient> client,
    WorkerIdentityStore* identity,
    std::string_view layout_id,
    const v2::AssemblyCloseoutContract& closeout_contract,
    const store::SealAssemblyResult& seal_result,
    std::optional<RepresentationPublishValidationResult> prevalidated_representation_publish,
    v2::SealAssemblyResult* result_msg) {
  ABSL_CHECK(engine != nullptr);
  ABSL_CHECK(result_msg != nullptr);

  populate_artifact_descriptor_from_seal_result(seal_result, result_msg->mutable_artifact());

  const auto canonical = coordination::canonicalize_closeout_contract(closeout_contract);
  switch (canonical.kind()) {
    case v2::ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY:
      if (!canonical.serving_version_key().empty() || !canonical.serving_artifact_id().empty() ||
          !canonical.serving_manifest_ref().empty() || canonical.has_representation_publish_contract()) {
        return absl::InvalidArgumentError(
            "source_publish_only closeout contracts may not set serving_version_key, serving_artifact_id, "
            "serving_manifest_ref, or representation_publish_contract");
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
    case v2::ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH: {
      absl::StatusOr<RepresentationPublishValidationResult> representation_publish_or =
          prevalidated_representation_publish.has_value()
          ? absl::StatusOr<RepresentationPublishValidationResult>(*prevalidated_representation_publish)
          : validate_representation_publish_closeout(engine, bindings, lip_manager, lifecycle, canonical);
      const auto subject = canonical.representation_publish_contract().has_subject()
          ? canonical.representation_publish_contract().subject()
          : pubv1::ServingPublicationSubject{};
      if (representation_publish_or.ok() && subject.ref_case() == pubv1::ServingPublicationSubject::kBindingValue) {
        representation_publish_or = finalize_binding_subject_closeout(
            state_sync_barrier,
            bindings,
            lip_manager,
            lifecycle,
            std::move(client),
            identity,
            layout_id,
            canonical.representation_publish_contract(),
            std::move(*representation_publish_or));
      }
      if (!representation_publish_or.ok()) {
        return representation_publish_or.status();
      }
      if (!canonical.source_version_key().empty()) {
        auto publish_status =
            publish_immutable_key(engine, canonical.source_version_key(), seal_result.sealed_artifact_id);
        if (!publish_status.ok()) {
          return publish_status;
        }
        result_msg->set_source_version_key(canonical.source_version_key());
      }
      if (!canonical.serving_version_key().empty()) {
        auto publish_status = publish_immutable_key(
            engine, canonical.serving_version_key(), representation_publish_or->serving_artifact.artifact_id());
        if (!publish_status.ok()) {
          return publish_status;
        }
        result_msg->set_serving_version_key(canonical.serving_version_key());
      }
      result_msg->mutable_serving_artifact()->CopyFrom(representation_publish_or->serving_artifact);
      result_msg->set_representation_contract_hash(representation_publish_or->representation_contract_hash);
      result_msg->set_serving_manifest_ref(representation_publish_or->serving_manifest_ref);
      result_msg->set_serving_build_digest(representation_publish_or->serving_build_digest);
      if (representation_publish_or->serving_execution_diagnostics.has_value()) {
        result_msg->mutable_serving_execution_diagnostics()->CopyFrom(
            *representation_publish_or->serving_execution_diagnostics);
      }
      return absl::OkStatus();
    }
    case v2::ASSEMBLY_CLOSEOUT_KIND_ROLLOUT_GATED_PUBLISH:
      return absl::UnimplementedError(
          "attempt closeout beyond representation_publish requires typed child closeout contracts");
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
  const bool has_representation_publish_spec = req.has_representation_publish_spec();
  if (has_representation_publish_spec &&
      (!req.layout_id().empty() || req.has_requirements() || req.has_readiness_policy() ||
       req.has_closeout_contract())) {
    return {
        StatusCode::INVALID_ARGUMENT,
        "representation_publish_spec is mutually exclusive with layout_id, requirements, readiness_policy, and "
        "closeout_contract"};
  }

  const std::string resolved_layout_id =
      has_representation_publish_spec ? req.representation_publish_spec().layout_id() : req.layout_id();

  auto& span = rctx.span();
  span->SetAttribute("tc.layout.id", resolved_layout_id);

  if (resolved_layout_id.empty()) {
    return {StatusCode::INVALID_ARGUMENT, "layout_id is required"};
  }
  if (has_representation_publish_spec) {
    if (auto admission_status = validate_representation_publish_admission_facts(req.representation_publish_spec());
        !admission_status.ok()) {
      return to_grpc_status(admission_status);
    }
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  v2::AssemblyRequirementSetRef requested_requirements = has_representation_publish_spec
      ? publication_requirements_to_v2(req.representation_publish_spec().requirements())
      : req.requirements();
  if (requested_requirements.inline_requirements_size() == 0) {
    return {
        StatusCode::INVALID_ARGUMENT,
        "requirements are required; daemon no longer derives attempt requirements from layout expected_view_ids"};
  }

  const auto layout_or = d_.global_store_client->get_layout_spec(resolved_layout_id);
  if (!layout_or.ok()) {
    return to_grpc_status(layout_or.status());
  }

  v2::AssemblyRequirementSetRef requirements = coordination::canonicalize_requirement_set(requested_requirements);
  if (requirements.inline_requirements_size() == 0) {
    return {
        StatusCode::INVALID_ARGUMENT,
        "requirements.inline_requirements must be non-empty; use a canonical_layout requirement for canonical-full "
        "attempts"};
  }
  if (auto requirement_status = coordination::validate_requirement_set(requirements); !requirement_status.ok()) {
    return to_grpc_status(requirement_status);
  }

  const v2::AssemblyReadinessPolicy requested_readiness_policy = has_representation_publish_spec
      ? publication_readiness_to_v2(req.representation_publish_spec().readiness_policy())
      : (req.has_readiness_policy() ? req.readiness_policy() : v2::AssemblyReadinessPolicy());
  v2::AssemblyReadinessPolicy readiness_policy =
      coordination::canonicalize_readiness_policy(requested_readiness_policy);

  const v2::AssemblyCloseoutContract requested_closeout_contract = has_representation_publish_spec
      ? publication_closeout_to_v2(req.representation_publish_spec())
      : (req.has_closeout_contract() ? req.closeout_contract() : build_default_closeout_contract());
  v2::AssemblyCloseoutContract closeout_contract =
      coordination::canonicalize_closeout_contract(requested_closeout_contract);
  if (auto closeout_status = coordination::validate_dependency_ready_closeout_contract(closeout_contract);
      !closeout_status.ok()) {
    return to_grpc_status(closeout_status);
  }

  v2::AssemblyAttemptIntent intent;
  intent.set_layout_id(resolved_layout_id);
  *intent.mutable_requirements() = requirements;
  *intent.mutable_readiness_policy() = readiness_policy;
  *intent.mutable_closeout_contract() = closeout_contract;
  intent = coordination::canonicalize_attempt_intent(intent);

  const std::string attempt_id = mint_assembly_attempt_id();
  const std::string workspace_assembly_id = mint_assembly_workspace_id();
  auto binding_or = d_.global_store_client->update_assembly_layout_binding(
      workspace_assembly_id,
      resolved_layout_id,
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
  attempt_info.layout_id = resolved_layout_id;
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
    auto* bindings = &d_.bindings;
    auto* lip_manager = d_.lip_manager;
    auto* lifecycle = d_.lifecycle;
    auto* identity = &d_.identity;
    auto await_state_sync_barrier_fn = d_.await_state_sync_barrier;
    const google::protobuf::Any snapshot_any = pack_operation_continuation_metadata(*out_ref);
    executor->add(
        [seal_tracker,
         client_sp = std::move(client_sp),
         async_runtime,
         engine,
         bindings,
         lip_manager,
         lifecycle,
         identity,
         await_state_sync_barrier_fn,
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
          SC_TRACE_INIT_GUARD(operation_id, workspace_assembly_id, "assembly_attempt.seal_worker");
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
          {
            SC_TRACE_SCOPE("assembly_attempt.update_operation_running");
            final_status = client_sp->update_operation(running);
          }
          if (!final_status.ok()) {
            LOG(WARNING) << "update_operation(RUNNING) failed for op=" << operation_id << ": " << final_status;
            keepalive_stop->store(true, std::memory_order_relaxed);
            return;
          }

          v2::AssemblyReadinessCut readiness_cut;
          std::optional<store::StoreEngine::SealAssemblyCutInput> seal_cut_input;
          std::optional<RepresentationPublishValidationResult> prevalidated_representation_publish;
          const auto canonical_closeout_contract =
              coordination::canonicalize_closeout_contract(record.intent().closeout_contract());
          if (final_status.ok() &&
              canonical_closeout_contract.kind() == v2::ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH &&
              canonical_closeout_contract.has_representation_publish_contract()) {
            const auto& representation_publish = canonical_closeout_contract.representation_publish_contract();
            const auto subject = representation_publish.has_subject() ? representation_publish.subject()
                                                                      : pubv1::ServingPublicationSubject{};
            if (representation_publish.has_subject() &&
                subject.ref_case() == pubv1::ServingPublicationSubject::kBindingValue) {
              SC_TRACE_SCOPE("assembly_attempt.prevalidate_binding_subject_closeout");
              auto validation_or = validate_representation_publish_closeout(
                  engine, bindings, lip_manager, lifecycle, canonical_closeout_contract);
              if (!validation_or.ok()) {
                final_status = validation_or.status();
              } else {
                prevalidated_representation_publish = std::move(*validation_or);
              }
            }
          }

          if (final_status.ok()) {
            absl::StatusOr<v2::AssemblyReadinessCut> readiness_or = absl::UnknownError("uninitialized");
            {
              SC_TRACE_SCOPE("assembly_attempt.capture_readiness_snapshot");
              readiness_or = capture_seal_readiness_snapshot(client_sp, record, lease_generation);
            }
            if (!readiness_or.ok()) {
              final_status = readiness_or.status();
            } else {
              readiness_cut = *readiness_or;
              store::components::AssemblyReadinessCutInfo readiness_info;
              readiness_info.attempt_id = attempt_id;
              if (!readiness_cut.SerializeToString(&readiness_info.readiness_cut_proto)) {
                final_status = absl::InternalError("failed to serialize readiness cut");
              } else {
                absl::StatusOr<store::components::AssemblyReadinessCutInfo> upsert_or =
                    absl::UnknownError("uninitialized");
                {
                  SC_TRACE_SCOPE("assembly_attempt.upsert_readiness_cut");
                  upsert_or = client_sp->upsert_assembly_readiness_cut(readiness_info);
                }
                if (!upsert_or.ok()) {
                  final_status = upsert_or.status();
                }
              }
              if (final_status.ok()) {
                SC_TRACE_SCOPE("assembly_attempt.build_seal_cut_input");
                auto seal_cut_input_or = build_seal_cut_input(*bindings, readiness_cut, client_sp.get());
                if (!seal_cut_input_or.ok()) {
                  final_status = seal_cut_input_or.status();
                } else {
                  seal_cut_input = std::move(*seal_cut_input_or);
                }
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

          absl::StatusOr<store::SealAssemblyResult> seal_or = absl::UnknownError("uninitialized");
          if (final_status.ok()) {
            SC_TRACE_SCOPE("assembly_attempt.engine.seal_assembly_from_cut");
            seal_or = engine->seal_assembly_from_cut(
                workspace_assembly_id,
                *seal_cut_input,
                /*publish_canonical=*/!prevalidated_representation_publish.has_value(),
                std::move(progress_cb));
          } else {
            seal_or = absl::StatusOr<store::SealAssemblyResult>(final_status);
          }
          if (!seal_or.ok()) {
            final_status = seal_or.status();
          } else {
            const std::string sealed_artifact_id = seal_or->sealed_artifact_id;
            const bool binding_subject_closeout = prevalidated_representation_publish.has_value();
            if (final_status.ok() && !record.intent().layout_id().empty() && !binding_subject_closeout) {
              SC_TRACE_SCOPE("assembly_attempt.attach_layout_to_artifact");
              final_status = client_sp->attach_layout_to_artifact(sealed_artifact_id, record.intent().layout_id());
            }

            tensorcast::daemon::v2::SealAssemblyResult result_msg;
            if (final_status.ok()) {
              SC_TRACE_SCOPE("assembly_attempt.finalize_dependency_ready_closeout");
              final_status = finalize_dependency_ready_closeout(
                  await_state_sync_barrier_fn,
                  engine,
                  bindings,
                  lip_manager,
                  lifecycle,
                  client_sp,
                  identity,
                  record.intent().layout_id(),
                  record.intent().closeout_contract(),
                  *seal_or,
                  prevalidated_representation_publish,
                  &result_msg);
            }

            if (final_status.ok()) {
              // Publish workspace->seal only after closeout has succeeded so failed attempts
              // cannot be resolved later as if the seal were committed.
              SC_TRACE_SCOPE("assembly_attempt.publish_workspace_seal_binding");
              final_status = closeout_identity::publish_workspace_seal_binding_after_success(
                  client_sp,
                  workspace_assembly_id,
                  sealed_artifact_id,
                  /*binding_subject_closeout=*/prevalidated_representation_publish.has_value(),
                  /*reused_sealed_identity=*/seal_or->already_sealed);
            }

            if (final_status.ok() && !binding_subject_closeout) {
              SC_TRACE_SCOPE("assembly_attempt.ensure_local_readable_source");
              final_status = ensure_local_readable_source_artifact(*engine, sealed_artifact_id);
            }

            if (final_status.ok()) {
              absl::Status finalize_status = absl::UnknownError("uninitialized");
              {
                SC_TRACE_SCOPE("assembly_attempt.finalize_slot_occupancies_released");
                finalize_status = finalize_assembly_slot_occupancies(client_sp, attempt_id, "released", {"accepted"});
              }
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
              SC_TRACE_SCOPE("assembly_attempt.update_operation_success");
              final_status = client_sp->update_operation(success);
            }
          }

          if (!final_status.ok()) {
            absl::Status finalize_status = absl::UnknownError("uninitialized");
            {
              SC_TRACE_SCOPE("assembly_attempt.finalize_slot_occupancies_aborted");
              finalize_status =
                  finalize_assembly_slot_occupancies(client_sp, attempt_id, "aborted", {"accepted", "stale"});
            }
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
            absl::Status update_st = absl::UnknownError("uninitialized");
            {
              SC_TRACE_SCOPE("assembly_attempt.update_operation_failed");
              update_st = client_sp->update_operation(failed);
            }
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
