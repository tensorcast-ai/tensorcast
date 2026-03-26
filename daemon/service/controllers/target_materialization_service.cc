// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/target_materialization_service.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/selection_identity.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/runtime/ingestion/materialization_strategy_types.h"
#include "daemon/service/controllers/materialization_disk_resolve_utils.h"
#include "daemon/service/controllers/materialization_index_source_utils.h"
#include "daemon/service/controllers/materialization_layout_utils.h"
#include "daemon/service/controllers/materialization_payload_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/service/controllers/materialization_request_common_utils.h"
#include "daemon/service/controllers/materialization_target_plan_utils.h"
#include "daemon/service/controllers/materialization_target_storage_utils.h"
#include "daemon/service/controllers/serving_artifact_manifest_utils.h"
#include "daemon/util/deadline_utils.h"
#include "daemon/util/grpc_peer_utils.h"
#include "daemon/util/status_utils.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

using materialization_index_source::compute_target_layout_multihash;
using materialization_index_source::load_canonical_index_with_disk_fallback;
using materialization_index_source::load_descriptor_metadata;
using materialization_index_source::parse_mi2_data_multihash;
using materialization_index_source::TargetLayoutSpan;
using materialization_layout::resolve_target_offsets;
using materialization_payload::compute_generation_from_index;
using materialization_policy::resolve_source_policy;
using materialization_policy::resolve_transform_placement;
using materialization_policy::ResolvedSourcePolicy;
using materialization_policy::to_hint_preference;
using materialization_policy::validate_source_policy;
using materialization_request_common::resolve_artifact_binding;
using materialization_request_common::resolve_managed_disk_path;
using materialization_target_plan::build_mapped_target_materialization_plan;
using materialization_target_plan::build_resolved_mapped_materialization_plan;
using materialization_target_plan::build_target_materialization_plan;
using materialization_target_plan::MappedTargetMaterializationPlan;
using materialization_target_plan::TargetMaterializationPlan;
using store::loading::MaterializationSource;

std::string mint_publication_id() {
  thread_local absl::BitGen bitgen;
  std::string raw;
  raw.resize(16);
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<char>(absl::Uniform<uint32_t>(bitgen, 0u, 256u));
  }
  return absl::BytesToHexString(raw);
}

std::string compute_target_layout_hash(const v2::TargetLayout& layout) {
  std::string buffer;
  buffer.reserve(512);
  absl::StrAppend(
      &buffer,
      "lk:",
      static_cast<int>(layout.layout_kind()),
      "|ik:",
      static_cast<int>(layout.index_kind()),
      "|tk:",
      static_cast<int>(layout.tensor_spec_kind()),
      "|vid:",
      layout.view_id(),
      "|");
  buffer.append(layout.logical_layout_hash().data(), layout.logical_layout_hash().size());
  for (const auto& storage : layout.storages()) {
    absl::StrAppend(
        &buffer,
        "|s:",
        storage.storage_id(),
        ":",
        storage.device_id(),
        ":",
        storage.storage_length(),
        ":",
        storage.mapping_base_offset(),
        ":");
    if (storage.storage_source_case() == v2::StorageEntry::kVramRegionId) {
      absl::StrAppend(&buffer, "r:", storage.vram_region_id());
    } else if (storage.storage_source_case() == v2::StorageEntry::kCudaIpcHandle) {
      buffer.append("h:");
      buffer.append(storage.cuda_ipc_handle().data(), storage.cuda_ipc_handle().size());
    }
  }
  for (const auto& entry : layout.offsets()) {
    absl::StrAppend(
        &buffer,
        "|o:",
        entry.name(),
        ":",
        entry.storage_id(),
        ":",
        entry.storage_offset(),
        ":",
        entry.logical_length());
  }
  const std::vector<uint8_t> digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size()));
  return std::string(reinterpret_cast<const char*>(digest.data()), digest.size());
}

void record_materialize_into_target(
    std::string_view result,
    std::string_view reason,
    v2::MaterializationSource source) {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_store_materialize_into_target_total");
    if (counter) {
      counter->Add(
          1,
          {{"result", std::string(result)}, {"reason", std::string(reason)}, {"source", static_cast<int64_t>(source)}});
    }
  } catch (...) {
  }
}

void record_materialize_into_target_verification_enabled() {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_store_materialize_into_target_verification_enabled_total");
    if (counter) {
      counter->Add(1);
    }
  } catch (...) {
  }
}

void record_materialize_into_target_verification_skipped() {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_store_materialize_into_target_verification_skipped_total");
    if (counter) {
      counter->Add(1);
    }
  } catch (...) {
  }
}

v2::MaterializationSource to_proto_source(MaterializationSource source) {
  switch (source) {
    case MaterializationSource::kDisk:
      return v2::MaterializationSource::MATERIALIZATION_SOURCE_DISK;
    case MaterializationSource::kP2P:
      return v2::MaterializationSource::MATERIALIZATION_SOURCE_P2P;
    case MaterializationSource::kLocalReplica:
      return v2::MaterializationSource::MATERIALIZATION_SOURCE_LOCAL_REPLICA;
    case MaterializationSource::kUnspecified:
    default:
      return v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED;
  }
}

struct TargetMaterializationCommonContext {
  ResolvedSourcePolicy effective_policy;
  std::string resolved_artifact_id;
  bool gs_connected{false};
  std::optional<std::filesystem::path> normalized_disk_path;
};

std::chrono::milliseconds resolve_target_request_budget(const grpc::ServerContext& server_context) {
  using clock = std::chrono::system_clock;
  constexpr std::chrono::milliseconds kDefaultBudget{600000};
  constexpr std::chrono::milliseconds kHardCap{1800000};
  const auto grpc_deadline = server_context.deadline();
  if (grpc_deadline != clock::time_point::max()) {
    const auto now = clock::now();
    if (grpc_deadline <= now) {
      return std::chrono::milliseconds(0);
    }
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(grpc_deadline - now);
    if (remaining.count() <= 0) {
      return std::chrono::milliseconds(1);
    }
    return std::min(remaining, kHardCap);
  }
  return std::min(kDefaultBudget, kHardCap);
}

struct ParsedTransportHint {
  std::string request_id;
  std::optional<store::loading::TransportSchedulingGroupHint> scheduling_group;
  std::optional<store::loading::CollectiveLoadGroupHint> collective_load_group;
};

ParsedTransportHint parse_transport_hint_from_operation_id(std::string_view operation_id) {
  constexpr std::string_view kGroupMarker = "#tcg:";
  ParsedTransportHint parsed;
  if (operation_id.empty()) {
    return parsed;
  }
  const size_t marker_pos = operation_id.find(kGroupMarker);
  if (marker_pos == std::string_view::npos) {
    parsed.request_id = std::string(operation_id);
    return parsed;
  }

  parsed.request_id = std::string(operation_id.substr(0, marker_pos));
  const std::string_view metadata = operation_id.substr(marker_pos + kGroupMarker.size());

  std::string group_kind;
  std::string group_id;
  std::string part_id;
  std::string request_id_override;
  std::string collective_group_id;
  int group_total_parts = 0;
  int group_priority = 0;
  int collective_world_size = 0;
  int collective_rank = -1;
  uint64_t group_epoch = 0;

  for (const std::string_view item : absl::StrSplit(metadata, ';', absl::SkipEmpty())) {
    std::vector<std::string_view> kv = absl::StrSplit(item, absl::MaxSplits('=', 1));
    if (kv.size() != 2) {
      continue;
    }
    const std::string_view key = kv[0];
    const std::string_view value = kv[1];
    if (key == "kind") {
      group_kind = std::string(value);
      continue;
    }
    if (key == "gid") {
      group_id = std::string(value);
      continue;
    }
    if (key == "tot") {
      int parsed_total_parts = 0;
      if (absl::SimpleAtoi(value, &parsed_total_parts)) {
        group_total_parts = parsed_total_parts;
      }
      continue;
    }
    if (key == "part") {
      part_id = std::string(value);
      continue;
    }
    if (key == "pri") {
      int parsed_priority = 0;
      if (absl::SimpleAtoi(value, &parsed_priority)) {
        group_priority = parsed_priority;
      }
      continue;
    }
    if (key == "ep") {
      uint64_t parsed_epoch = 0;
      if (absl::SimpleAtoi(value, &parsed_epoch)) {
        group_epoch = parsed_epoch;
      }
      continue;
    }
    if (key == "rid") {
      request_id_override = std::string(value);
      continue;
    }
    if (key == "clid") {
      collective_group_id = std::string(value);
      continue;
    }
    if (key == "clws") {
      int parsed_world_size = 0;
      if (absl::SimpleAtoi(value, &parsed_world_size)) {
        collective_world_size = parsed_world_size;
      }
      continue;
    }
    if (key == "clrk") {
      int parsed_rank = -1;
      if (absl::SimpleAtoi(value, &parsed_rank)) {
        collective_rank = parsed_rank;
      }
      continue;
    }
  }

  if (!request_id_override.empty()) {
    parsed.request_id = std::move(request_id_override);
  }

  if (!group_kind.empty() && !group_id.empty() && !part_id.empty() && group_total_parts > 0) {
    store::loading::TransportSchedulingGroupHint group;
    group.group_kind = std::move(group_kind);
    group.group_id = std::move(group_id);
    group.total_parts = static_cast<uint32_t>(group_total_parts);
    group.part_id = std::move(part_id);
    group.priority = static_cast<uint32_t>(std::max(0, group_priority));
    group.epoch = group_epoch;
    parsed.scheduling_group = std::move(group);
  }
  if (!collective_group_id.empty() && collective_world_size > 1 && collective_rank >= 0 &&
      collective_rank < collective_world_size) {
    store::loading::CollectiveLoadGroupHint collective_group;
    collective_group.group_id = std::move(collective_group_id);
    collective_group.world_size = static_cast<uint32_t>(collective_world_size);
    collective_group.rank = static_cast<uint32_t>(collective_rank);
    parsed.collective_load_group = std::move(collective_group);
  }
  return parsed;
}

void apply_transport_hints_from_operation_id(const std::string& operation_id, store::loading::MaterializeHints* hints) {
  if (hints == nullptr || operation_id.empty()) {
    return;
  }
  ParsedTransportHint parsed = parse_transport_hint_from_operation_id(operation_id);
  if (!parsed.request_id.empty()) {
    hints->transport_request_id = parsed.request_id;
  }
  if (parsed.scheduling_group.has_value()) {
    hints->transport_scheduling_group = std::move(*parsed.scheduling_group);
  }
  if (parsed.collective_load_group.has_value()) {
    hints->collective_load_group = std::move(*parsed.collective_load_group);
  }
}

absl::StatusOr<std::optional<store::loading::DiskMetadata>> build_target_disk_metadata(
    const std::optional<std::filesystem::path>& normalized_disk_path,
    std::string_view resolved_artifact_id,
    int device_ordinal,
    ArtifactSourceRegistry& disk_imports) {
  std::optional<store::loading::DiskMetadata> disk_metadata;
  if (normalized_disk_path.has_value()) {
    auto descriptor_or = load_descriptor_metadata(*normalized_disk_path);
    if (!descriptor_or.ok()) {
      return descriptor_or.status();
    }
    auto index_or = store::loader::read_from_artifact_dir(*normalized_disk_path, device_ordinal);
    if (!index_or.ok()) {
      return index_or.status();
    }
    store::loading::DiskMetadata metadata;
    metadata.descriptor_present = descriptor_or->found;
    metadata.schema_version = descriptor_or->schema_version;
    metadata.index_multihash = descriptor_or->index_multihash;
    metadata.data_multihash = descriptor_or->data_multihash;
    metadata.canonical_index_json = index_or->canonical_index_json;
    if (index_or->source_index_json.has_value()) {
      metadata.source_index_json = index_or->source_index_json;
    }
    if (!index_or->index_multihash.empty()) {
      metadata.index_multihash = index_or->index_multihash;
    }
    if (index_or->total_size_bytes > 0) {
      metadata.logical_total_size = index_or->total_size_bytes;
    }
    if (index_or->source_total_size_bytes > 0) {
      metadata.source_total_size_bytes = index_or->source_total_size_bytes;
    }
    metadata.is_safetensors = index_or->is_safetensors;
    disk_metadata = std::move(metadata);
  }
  if (auto local_import = disk_imports.lookup_binding(resolved_artifact_id); local_import.has_value()) {
    if (!disk_metadata.has_value()) {
      disk_metadata = store::loading::DiskMetadata{};
    }
    auto& metadata = *disk_metadata;
    metadata.descriptor_present = metadata.descriptor_present || local_import->descriptor_present;
    if (!metadata.canonical_index_json.has_value() && !local_import->canonical_index_json.empty()) {
      metadata.canonical_index_json = local_import->canonical_index_json;
    }
    if (!metadata.source_index_json.has_value() && local_import->source_index_json.has_value()) {
      metadata.source_index_json = *local_import->source_index_json;
    }
    if (!metadata.index_multihash.has_value() && local_import->index_multihash.has_value()) {
      metadata.index_multihash = *local_import->index_multihash;
    }
    if (!metadata.data_multihash.has_value() && local_import->data_multihash.has_value()) {
      metadata.data_multihash = *local_import->data_multihash;
    }
  }
  return disk_metadata;
}

template <typename RequestT>
absl::StatusOr<TargetMaterializationCommonContext> prepare_target_materialization_common(
    const RequestT& req,
    std::string_view peer,
    std::string_view rpc_name,
    const std::shared_ptr<store::components::IGlobalStoreClient>& global_store_client,
    ArtifactSourceRegistry& disk_imports,
    const std::filesystem::path& storage_path) {
  TargetMaterializationCommonContext context;
  context.effective_policy =
      resolve_source_policy(req.has_source_policy() ? &req.source_policy() : nullptr, req.preference());
  const absl::Status policy_status = validate_source_policy(context.effective_policy);
  if (!policy_status.ok()) {
    record_materialize_into_target(
        "error", "policy_invalid", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return policy_status;
  }
  if (!is_loopback_grpc_peer(peer)) {
    record_materialize_into_target(
        "error", "non_loopback_peer", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::PermissionDeniedError(std::format("{} is local-only (loopback/UDS)", rpc_name));
  }

  if (!req.has_selection()) {
    record_materialize_into_target(
        "error", "missing_artifact_id", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::InvalidArgumentError(std::format("selection is required for {}", rpc_name));
  }
  if (req.selection().artifact_id().empty()) {
    record_materialize_into_target(
        "error", "missing_artifact_id", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::InvalidArgumentError(std::format("selection.artifact_id is required for {}", rpc_name));
  }

  context.resolved_artifact_id = req.selection().artifact_id();
  auto binding_or = resolve_artifact_binding(global_store_client, context.resolved_artifact_id);
  if (!binding_or.ok()) {
    record_materialize_into_target(
        "error", "binding_error", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return binding_or.status();
  }
  if (binding_or->has_value()) {
    context.resolved_artifact_id = binding_or->value();
  }
  context.gs_connected = global_store_client && global_store_client->is_connected();
  context.normalized_disk_path = resolve_managed_disk_path(
      global_store_client.get(), storage_path, context.resolved_artifact_id, context.effective_policy.allow_disk);
  if (!context.normalized_disk_path.has_value() && context.effective_policy.allow_disk) {
    auto entry = disk_imports.lookup_binding(context.resolved_artifact_id);
    if (entry.has_value()) {
      if (entry->source_kind == ArtifactSourceRegistry::SourceKind::kLocalImport) {
        materialization_disk_resolve::SourceFingerprintMap expected_fingerprints;
        expected_fingerprints.reserve(entry->file_fingerprints.size());
        for (const auto& [relative_path, fp] : entry->file_fingerprints) {
          expected_fingerprints.insert_or_assign(
              relative_path,
              materialization_disk_resolve::SourceFileFingerprint{
                  .inode = fp.inode,
                  .size = fp.size,
                  .mtime_ns = fp.mtime_ns,
              });
        }
        auto fingerprint_status = materialization_disk_resolve::validate_source_fingerprints(
            std::filesystem::path(entry->canonical_source_path), expected_fingerprints);
        if (!fingerprint_status.ok()) {
          return fingerprint_status;
        }
      }
      context.normalized_disk_path = std::filesystem::path(entry->canonical_source_path);
    }
  }
  if (!req.has_target_layout()) {
    record_materialize_into_target(
        "error", "layout_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::InvalidArgumentError("target_layout is required");
  }
  if (req.device_uuid().empty()) {
    record_materialize_into_target(
        "error", "device_uuid_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::InvalidArgumentError("device_uuid is required");
  }
  if (req.pid() <= 0) {
    record_materialize_into_target(
        "error", "owner_pid_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::InvalidArgumentError(std::format("pid is required for {}", rpc_name));
  }
  return context;
}

} // namespace

TargetMaterializationService::TargetMaterializationService(Dep d)
    : d_(std::move(d)),
      capability_tokens_(d_.capability_tokens),
      target_publish_service_(
          TargetPublishService::Dep{
              .lip_manager = d_.lip_manager,
              .devices = d_.devices,
              .identity = d_.identity,
              .lifecycle = d_.lifecycle,
              .lifecycle_kernel = d_.lifecycle_kernel,
              .async_runtime = d_.async_runtime,
              .shutdown_signal = d_.shutdown_signal,
              .global_store_client = d_.global_store_client,
              .capability_tokens = d_.capability_tokens,
              .max_concurrency = d_.max_concurrency,
          }) {
  if (!d_.storage_path.empty()) {
    std::error_code ec;
    storage_path_ = std::filesystem::weakly_canonical(d_.storage_path, ec);
    if (ec) {
      ec.clear();
      storage_path_ = d_.storage_path.lexically_normal();
    }
  }
}

absl::StatusOr<TargetPublicationRegistry::Record> TargetMaterializationService::insert_target_publication_for_testing(
    TargetPublicationRegistry::Record record) {
  return target_publish_service_.remember_target_publication(std::move(record));
}

absl::StatusOr<TargetPublishService::TargetPublicationFrontDoorContext> TargetMaterializationService::
    inspect_target_publication_context_for_testing(const v2::PublishTargetReplicaRequest& req, absl::Time now) {
  return target_publish_service_.inspect_target_publication_context(req, now);
}

absl::StatusOr<RoutedAuthorityRequest> TargetMaterializationService::
    build_target_publication_workflow_routed_request_for_testing(
        const v2::PublishTargetReplicaRequest& req,
        absl::Time now) const {
  return target_publish_service_.build_target_publication_workflow_routed_request(req, now);
}

absl::StatusOr<RoutedAuthorityRequest> TargetMaterializationService::
    build_target_publication_workflow_continuation_request_for_testing(
        const RoutedAuthorityRequest& routed_request,
        const OwnerStageReply& workflow_gate_reply) const {
  return target_publish_service_.build_target_publication_workflow_continuation_request(
      routed_request, workflow_gate_reply);
}

absl::StatusOr<std::optional<OwnerStageReply>> TargetMaterializationService::maybe_route_authority_stage(
    const RoutedAuthorityRequest& routed_request,
    absl::Time now) {
  return target_publish_service_.maybe_route_authority_stage(routed_request, now);
}

grpc::Status TargetMaterializationService::materialize_into_target(
    RpcContext& rctx,
    const v2::MaterializeIntoTargetRequest& req,
    v2::MaterializeIntoTargetResponse& resp) {
  if (d_.shutdown_signal.is_shutting_down()) {
    record_materialize_into_target(
        "error", "unavailable", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  auto common_or = prepare_target_materialization_common(
      req,
      rctx.server_context().peer(),
      "MaterializeIntoTarget",
      d_.global_store_client,
      d_.disk_imports,
      storage_path_);
  if (!common_or.ok()) {
    return to_grpc_status(common_or.status());
  }
  auto common = std::move(*common_or);
  ResolvedSourcePolicy effective_policy = std::move(common.effective_policy);
  std::string resolved_artifact_id = std::move(common.resolved_artifact_id);
  const bool gs_connected = common.gs_connected;
  std::optional<std::filesystem::path> normalized_disk_path = std::move(common.normalized_disk_path);

  const auto& layout = req.target_layout();
  if (layout.layout_kind() != v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED) {
    record_materialize_into_target(
        "error", "layout_kind_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Only LAYOUT_KIND_COALESCED_UNSPECIFIED is supported"};
  }
  if (layout.index_kind() != v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED &&
      layout.index_kind() != v2::TargetLayout::INDEX_KIND_VIEW) {
    record_materialize_into_target(
        "error", "index_kind_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Unsupported index_kind for MaterializeIntoTarget"};
  }
  if (layout.tensor_spec_kind() != v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS &&
      layout.tensor_spec_kind() != v2::TargetLayout::TENSOR_SPEC_KIND_ALIAS_UNSPECIFIED) {
    record_materialize_into_target(
        "error", "tensor_spec_kind_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Unsupported tensor_spec_kind for MaterializeIntoTarget"};
  }
  if (layout.storages_size() == 0) {
    record_materialize_into_target(
        "error", "storage_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "target_layout must include at least one storage entry"};
  }

  auto validated_target_or = d_.external_target_access_service.validate_local_target_layout(
      rctx.server_context().peer(), "MaterializeIntoTarget", layout, req.pid(), req.device_uuid());
  if (!validated_target_or.ok()) {
    record_materialize_into_target(
        "error", "target_access_invalid", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(validated_target_or.status());
  }
  auto validated_target = std::move(*validated_target_or);
  const auto device = validated_target.device;

  auto offsets_or = resolve_target_offsets(layout);
  if (!offsets_or.ok()) {
    record_materialize_into_target(
        "error", "offsets_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(offsets_or.status());
  }
  const auto& offsets = *offsets_or;
  if (offsets.empty()) {
    record_materialize_into_target(
        "error", "offsets_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "target_layout offsets are required"};
  }

  auto canonical_json_or = load_canonical_index_with_disk_fallback(
      d_.engine, resolved_artifact_id, normalized_disk_path, device.ordinal, gs_connected);
  if (!canonical_json_or.ok()) {
    record_materialize_into_target(
        "error", "index_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(canonical_json_or.status());
  }

  TargetMaterializationPlan plan;
  auto build_plan_status = build_target_materialization_plan(
      d_.engine,
      resolved_artifact_id,
      req,
      layout,
      offsets,
      std::move(*canonical_json_or),
      record_materialize_into_target,
      plan);
  if (!build_plan_status.ok()) {
    return build_plan_status;
  }

  const auto& resolved_selection = plan.resolved_selection;
  const bool has_subset = resolved_selection.tensor_names_size() > 0;
  auto& view_spec = plan.view_spec;
  auto& view_plan = plan.view_plan;
  auto& resolved_view_id = plan.resolved_view_id;
  auto& view_data_hash = plan.view_data_hash;
  const bool has_view_transform = plan.has_view_transform;
  const std::string& canonical_index_json = plan.canonical_index_json;
  const uint64_t logical_total_size = plan.logical_total_size;
  std::vector<RegisterStorageMeta> publish_storages = std::move(plan.publish_storages);
  std::vector<LeaseSegMeta> publish_segments = std::move(plan.publish_segments);

  if (d_.external_target_verification_enabled && resolved_view_id.has_value() && !view_data_hash.has_value()) {
    auto view_meta_or = d_.engine.get_view_metadata(resolved_artifact_id, *resolved_view_id);
    if (view_meta_or.ok()) {
      view_data_hash = view_meta_or->view_data_hash;
    } else {
      VLOG(1) << "MaterializeIntoTarget: view metadata unavailable for verification: " << view_meta_or.status();
    }
  }

  std::optional<std::string> expected_data_hash;
  bool verify_external_target = false;
  if (d_.external_target_verification_enabled) {
    if (layout.index_kind() == v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED && !has_view_transform &&
        !has_subset) {
      expected_data_hash = parse_mi2_data_multihash(resolved_artifact_id);
    } else if (resolved_view_id.has_value()) {
      expected_data_hash = view_data_hash;
    }
    if (expected_data_hash.has_value()) {
      record_materialize_into_target_verification_enabled();
      verify_external_target = true;
    } else {
      record_materialize_into_target_verification_skipped();
    }
  } else {
    record_materialize_into_target_verification_skipped();
  }

  if (rctx.server_context().IsCancelled()) {
    record_materialize_into_target("error", "cancelled", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::CANCELLED, "request cancelled before transfer"};
  }

  auto storage_lease = std::move(validated_target.storage_lease);

  std::vector<TargetLayoutSpan> verification_spans;
  if (verify_external_target) {
    verification_spans.reserve(storage_lease.storages().size());
    uint64_t cursor = 0;
    for (const auto& storage : storage_lease.storages()) {
      verification_spans.push_back(
          TargetLayoutSpan{
              .base_ptr = storage.base_ptr,
              .offset = cursor,
              .length = storage.length,
          });
      cursor += storage.length;
    }
  }

  std::optional<store::loading::DiskSource> disk_source;
  if (normalized_disk_path.has_value()) {
    disk_source = store::loading::DiskSource{
        .path = *normalized_disk_path,
        .expected_size = logical_total_size,
        .require_descriptor = true,
    };
  }
  auto disk_metadata_or =
      build_target_disk_metadata(normalized_disk_path, resolved_artifact_id, device.ordinal, d_.disk_imports);
  if (!disk_metadata_or.ok()) {
    return to_grpc_status(disk_metadata_or.status());
  }
  auto disk_metadata = std::move(*disk_metadata_or);

  store::loading::MaterializeHints hints;
  const std::chrono::milliseconds request_budget = resolve_target_request_budget(rctx.server_context());
  hints.request_budget = request_budget;
  hints.transport_wait_timeout = request_budget;
  hints.artifact_id = resolved_artifact_id;
  const std::string requester_worker_id = d_.identity.worker_id();
  if (!requester_worker_id.empty()) {
    hints.transport_requester_worker_id = requester_worker_id;
  }
  if (req.has_operation_id() && !req.operation_id().empty()) {
    apply_transport_hints_from_operation_id(req.operation_id(), &hints);
  }
  const bool prefer_direct_disk_for_source_layout =
      disk_source.has_value() && disk_metadata.has_value() && disk_metadata->source_index_json.has_value();
  hints.source_preference = prefer_direct_disk_for_source_layout ? store::loading::SourcePreference::kPreferDisk
                                                                 : to_hint_preference(effective_policy.preference);
  hints.allow_p2p = prefer_direct_disk_for_source_layout ? false : effective_policy.allow_p2p;
  hints.allow_disk = effective_policy.allow_disk;
  hints.verify = store::loading::MaterializeHints::Verify::NONE;
  // bind/swap target materialization should become reusable sources so
  // subsequent peers can diffuse fan-out instead of contending on the
  // original publisher. This is still policy-gated by daemon promotion
  // controls and GS routing limits.
  hints.export_policy = store::loading::ExportPolicy::kForce;
  if (disk_metadata.has_value()) {
    hints.disk_metadata = *disk_metadata;
  }
  if (disk_source.has_value()) {
    hints.source_mutation_policy = store::loading::SourceMutationPolicy::kReadOnly;
  }

  if (view_plan.has_value() && layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW) {
    store::loading::VariantIdentity variant;
    variant.canonical_artifact_id = resolved_artifact_id;
    if (resolved_view_id.has_value()) {
      variant.view_id = *resolved_view_id;
    }
    if (view_spec.has_value()) {
      variant.view_spec = view_spec;
    }
    variant.cached_plan = view_plan;
    variant.canonical_index_json = canonical_index_json;
    variant.placement = resolve_transform_placement(req.placement(), view_spec);
    hints.variant = std::move(variant);
  }

  const uint64_t generation = compute_generation_from_index(canonical_index_json);
  store::loading::IntoTargetLayout target_layout;
  target_layout.storages.assign(storage_lease.storages().begin(), storage_lease.storages().end());
  target_layout.total_size = logical_total_size;
  auto result_or =
      d_.engine.materialize_into_target(device, target_layout, canonical_index_json, generation, hints, disk_source);
  if (!result_or.ok()) {
    if (absl::IsDataLoss(result_or.status())) {
      for (const auto& region_id : storage_lease.acquired_region_ids()) {
        d_.regions.mark_poisoned(region_id).IgnoreError();
      }
      record_materialize_into_target(
          "error", "transfer_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    } else {
      record_materialize_into_target(
          "error", "transfer_error", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    }
    return to_grpc_status(result_or.status());
  }
  auto preflight_or = serving_artifact_manifest::preflight_serving_artifact(
      &d_.engine,
      serving_artifact_manifest::build_preflight_request(
          resolved_artifact_id,
          canonical_index_json,
          disk_source,
          req.has_serving_artifact_policy() ? &req.serving_artifact_policy() : nullptr));
  if (!preflight_or.ok()) {
    for (const auto& region_id : storage_lease.acquired_region_ids()) {
      d_.regions.mark_poisoned(region_id).IgnoreError();
    }
    record_materialize_into_target(
        "error", "serving_manifest_invalid", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(preflight_or.status());
  }

  if (verify_external_target) {
    auto actual_hash_or =
        compute_target_layout_multihash(std::move(verification_spans), logical_total_size, device.ordinal);
    if (!actual_hash_or.ok()) {
      for (const auto& region_id : storage_lease.acquired_region_ids()) {
        d_.regions.mark_poisoned(region_id).IgnoreError();
      }
      record_materialize_into_target(
          "error", "verification_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return to_grpc_status(
          absl::DataLossError(
              absl::StrCat("external target verification failed: ", actual_hash_or.status().message())));
    }
    if (expected_data_hash.has_value() && *expected_data_hash != *actual_hash_or) {
      for (const auto& region_id : storage_lease.acquired_region_ids()) {
        d_.regions.mark_poisoned(region_id).IgnoreError();
      }
      record_materialize_into_target(
          "error", "verification_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::DATA_LOSS, "external target verification failed: data hash mismatch"};
    }
  }

  resp.set_artifact_id(resolved_artifact_id);
  resp.set_status(v2::MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  resp.set_source(to_proto_source(result_or->source));
  resp.set_canonical_index_bytes(canonical_index_json);
  if (view_plan.has_value() && layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW) {
    resp.set_view_index_bytes(view_plan->view_index_json);
  }
  if (resolved_selection.tensor_names_size() > 0 || !resolved_selection.view_subset_hash().empty()) {
    auto* subset = resp.mutable_view_subset();
    if (!resolved_selection.view_subset_hash().empty()) {
      subset->set_subset_hash(resolved_selection.view_subset_hash());
    }
    for (const auto& name : resolved_selection.tensor_names()) {
      subset->add_tensor_names(name);
    }
  }
  resp.mutable_resolved_selection()->CopyFrom(resolved_selection);
  resp.set_generation(generation);
  if (capability_tokens_ != nullptr && capability_tokens_->configured()) {
    tensorcast::common::v1::ByteSpaceRef byte_space;
    if (!resolved_selection.view_id().empty()) {
      byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
      byte_space.set_id(resolved_selection.view_id());
    } else {
      byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
      byte_space.set_id("");
    }

    const std::string layout_hash = compute_target_layout_hash(layout);
    const std::string publication_id = mint_publication_id();
    const absl::Time expires_at = absl::Now() + TargetPublishService::target_publication_token_ttl();

    auto stable_index_or = store::loader::rebuild_stable_canonical_index(canonical_index_json, device.ordinal);
    if (!stable_index_or.ok()) {
      VLOG(1) << "MaterializeIntoTarget: failed to rebuild canonical index for target publication token: "
              << stable_index_or.status();
    } else {
      std::string stable_index_json = std::move(*stable_index_or);
      const auto digest = common::sha256_digest_bytes(
          absl::Span<const uint8_t>(
              reinterpret_cast<const uint8_t*>(stable_index_json.data()), stable_index_json.size()));
      std::string index_key_hex =
          absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));

      tensorcast::common::v1::TargetPublicationScope scope;
      scope.set_publication_id(publication_id);
      scope.mutable_selection()->CopyFrom(resolved_selection);
      scope.mutable_byte_space()->CopyFrom(byte_space);
      scope.set_device_uuid(req.device_uuid());
      scope.set_owner_pid(req.pid());
      scope.set_target_layout_hash(layout_hash);
      if (req.has_operation_id()) {
        scope.set_operation_id(req.operation_id());
      }

      auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
      if (scope_or.ok()) {
        const uint64_t expires_at_ms = static_cast<uint64_t>(absl::ToUnixMillis(expires_at));
        auto token_or = capability_tokens_->mint(
            d_.identity.daemon_id(),
            tensorcast::common::v1::CAPABILITY_AUDIENCE_TARGET_PUBLICATION,
            *scope_or,
            expires_at_ms);
        if (token_or.ok()) {
          TargetPublicationRegistry::Record record;
          record.publication_id = PublicationInstanceId{.value = publication_id};
          record.publication_subject_key =
              build_publication_subject_key(resolved_selection, byte_space, layout_hash, req.device_uuid());
          record.target_layout_hash = layout_hash;
          record.selection = resolved_selection;
          record.byte_space = byte_space;
          record.canonical_index_json = std::move(stable_index_json);
          record.index_key_hex = std::move(index_key_hex);
          record.device_uuid = req.device_uuid();
          record.owner_pid = req.pid();
          if (req.has_operation_id()) {
            record.request_operation_id = req.operation_id();
          }
          record.expires_at = expires_at;
          record.segments = std::move(publish_segments);
          record.storages = std::move(publish_storages);
          auto inserted_or = target_publish_service_.remember_target_publication(std::move(record));
          if (inserted_or.ok()) {
            resp.set_target_publication_token(*token_or);
          } else {
            VLOG(1) << "MaterializeIntoTarget: failed to register target publication lifecycle: "
                    << inserted_or.status();
          }
        } else {
          VLOG(1) << "MaterializeIntoTarget: failed to mint target_publication_token: " << token_or.status();
        }
      } else {
        VLOG(1) << "MaterializeIntoTarget: failed to serialize target_publication scope: " << scope_or.status();
      }
    }
  }
  record_materialize_into_target("ok", "ok", resp.source());
  rctx.mark_success();
  return Status::OK;
}

grpc::Status TargetMaterializationService::materialize_into_mapped_target(
    RpcContext& rctx,
    const v2::MaterializeIntoMappedTargetRequest& req,
    v2::MaterializeIntoTargetResponse& resp) {
  const auto total_start = std::chrono::steady_clock::now();
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs() && req.has_operation_id()) {
    span->SetAttribute("tc.operation.id", req.operation_id());
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    record_materialize_into_target(
        "error", "unavailable", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  auto common_or = prepare_target_materialization_common(
      req,
      rctx.server_context().peer(),
      "MaterializeIntoMappedTarget",
      d_.global_store_client,
      d_.disk_imports,
      storage_path_);
  const auto common_done = std::chrono::steady_clock::now();
  if (!common_or.ok()) {
    return to_grpc_status(common_or.status());
  }
  auto common = std::move(*common_or);
  ResolvedSourcePolicy effective_policy = std::move(common.effective_policy);
  std::string resolved_artifact_id = std::move(common.resolved_artifact_id);
  const bool gs_connected = common.gs_connected;
  std::optional<std::filesystem::path> normalized_disk_path = std::move(common.normalized_disk_path);

  if (!req.has_copy_plan() || req.copy_plan().entries_size() == 0) {
    record_materialize_into_target(
        "error", "mapping_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "copy_plan is required for mapped binding"};
  }
  if (req.copy_plan().version() != 1) {
    record_materialize_into_target(
        "error", "mapping_version", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "unsupported copy_plan version"};
  }

  const auto& layout = req.target_layout();
  if (layout.layout_kind() != v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED) {
    record_materialize_into_target(
        "error", "layout_kind_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Only LAYOUT_KIND_COALESCED_UNSPECIFIED is supported"};
  }
  if (layout.tensor_spec_kind() != v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS) {
    record_materialize_into_target(
        "error", "tensor_spec_kind_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Mapped binding requires TENSOR_SPEC_KIND_OFFSETS"};
  }
  if (layout.storages_size() == 0) {
    record_materialize_into_target(
        "error", "storage_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "target_layout must include at least one storage entry"};
  }

  auto validated_target_or = d_.external_target_access_service.validate_local_target_layout(
      rctx.server_context().peer(), "MaterializeIntoMappedTarget", layout, req.pid(), req.device_uuid());
  const auto target_validate_done = std::chrono::steady_clock::now();
  if (!validated_target_or.ok()) {
    record_materialize_into_target(
        "error", "target_access_invalid", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(validated_target_or.status());
  }
  auto validated_target = std::move(*validated_target_or);
  const auto device = validated_target.device;

  auto offsets_or = resolve_target_offsets(layout);
  const auto offsets_done = std::chrono::steady_clock::now();
  if (!offsets_or.ok()) {
    record_materialize_into_target(
        "error", "offsets_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(offsets_or.status());
  }
  const auto& offsets = *offsets_or;
  if (offsets.empty()) {
    record_materialize_into_target(
        "error", "offsets_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "target_layout offsets are required"};
  }

  auto canonical_json_or = load_canonical_index_with_disk_fallback(
      d_.engine, resolved_artifact_id, normalized_disk_path, device.ordinal, gs_connected);
  const auto canonical_done = std::chrono::steady_clock::now();
  if (!canonical_json_or.ok()) {
    record_materialize_into_target(
        "error", "index_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(canonical_json_or.status());
  }

  MappedTargetMaterializationPlan mapped_plan;
  auto build_mapped_plan_status = build_mapped_target_materialization_plan(
      d_.engine,
      req,
      resolved_artifact_id,
      offsets,
      std::move(*canonical_json_or),
      record_materialize_into_target,
      mapped_plan);
  const auto mapped_plan_done = std::chrono::steady_clock::now();
  if (!build_mapped_plan_status.ok()) {
    return build_mapped_plan_status;
  }
  const auto& resolved_selection = mapped_plan.resolved_selection;
  const uint64_t logical_total_size = mapped_plan.logical_total_size;
  const std::string& canonical_index_json = mapped_plan.canonical_index_json;
  auto& view_spec = mapped_plan.view_spec;
  auto& view_plan = mapped_plan.view_plan;
  auto publish_storages = std::move(mapped_plan.publish_storages);
  auto publish_segments = std::move(mapped_plan.publish_segments);

  auto storage_lease = std::move(validated_target.storage_lease);

  std::optional<store::loading::DiskSource> disk_source;
  if (normalized_disk_path.has_value()) {
    disk_source = store::loading::DiskSource{
        .path = *normalized_disk_path,
        .expected_size = logical_total_size,
        .require_descriptor = true,
    };
  }
  auto disk_metadata_or =
      build_target_disk_metadata(normalized_disk_path, resolved_artifact_id, device.ordinal, d_.disk_imports);
  const auto disk_metadata_done = std::chrono::steady_clock::now();
  if (!disk_metadata_or.ok()) {
    return to_grpc_status(disk_metadata_or.status());
  }
  auto disk_metadata = std::move(*disk_metadata_or);

  store::loading::MaterializeHints hints;
  const std::chrono::milliseconds request_budget = resolve_target_request_budget(rctx.server_context());
  hints.request_budget = request_budget;
  hints.transport_wait_timeout = request_budget;
  hints.artifact_id = resolved_artifact_id;
  const std::string requester_worker_id = d_.identity.worker_id();
  if (!requester_worker_id.empty()) {
    hints.transport_requester_worker_id = requester_worker_id;
  }
  if (req.has_operation_id() && !req.operation_id().empty()) {
    apply_transport_hints_from_operation_id(req.operation_id(), &hints);
  }
  const bool prefer_direct_disk_for_source_layout =
      disk_source.has_value() && disk_metadata.has_value() && disk_metadata->source_index_json.has_value();
  hints.source_preference = prefer_direct_disk_for_source_layout ? store::loading::SourcePreference::kPreferDisk
                                                                 : to_hint_preference(effective_policy.preference);
  hints.allow_p2p = prefer_direct_disk_for_source_layout ? false : effective_policy.allow_p2p;
  hints.allow_disk = effective_policy.allow_disk;
  hints.verify = store::loading::MaterializeHints::Verify::NONE;
  // bind/swap target materialization should become reusable sources so
  // subsequent peers can diffuse fan-out instead of contending on the
  // original publisher. This is still policy-gated by daemon promotion
  // controls and GS routing limits.
  hints.export_policy = store::loading::ExportPolicy::kForce;
  if (disk_metadata.has_value()) {
    hints.disk_metadata = *disk_metadata;
  }
  if (disk_source.has_value()) {
    hints.source_mutation_policy = store::loading::SourceMutationPolicy::kReadOnly;
  }
  if (view_plan.has_value() && view_plan->transform.requires_materialization) {
    return {StatusCode::INVALID_ARGUMENT, "mapped binding does not support view transforms"};
  }
  if (!resolved_selection.view_id().empty() || view_spec.has_value() || view_plan.has_value()) {
    store::loading::VariantIdentity variant;
    variant.canonical_artifact_id = resolved_artifact_id;
    if (!resolved_selection.view_id().empty()) {
      variant.view_id = resolved_selection.view_id();
    }
    if (view_spec.has_value()) {
      variant.view_spec = view_spec;
    }
    if (view_plan.has_value()) {
      variant.cached_plan = view_plan;
    }
    variant.canonical_index_json = canonical_index_json;
    variant.placement = resolve_transform_placement(req.placement(), view_spec);
    hints.variant = std::move(variant);
  }
  store::loading::IntoTargetLayout target_layout;
  target_layout.storages.assign(storage_lease.storages().begin(), storage_lease.storages().end());
  target_layout.total_size = logical_total_size;
  const uint64_t generation = compute_generation_from_index(canonical_index_json);
  auto resolved_plan_or = build_resolved_mapped_materialization_plan(
      resolved_artifact_id,
      generation,
      target_layout,
      materialization_target_plan::MappedTargetMaterializationPlan{
          .view_spec = view_spec,
          .view_plan = view_plan,
          .resolved_selection = resolved_selection,
          .representation = mapped_plan.representation,
          .canonical_index_json = canonical_index_json,
          .selected_index_json = {},
          .publish_storages = {},
          .publish_segments = {},
          .logical_total_size = logical_total_size,
      },
      hints.variant,
      disk_metadata.has_value() && disk_metadata->source_index_json.has_value()
          ? std::optional<std::string_view>(*disk_metadata->source_index_json)
          : std::nullopt);
  if (!resolved_plan_or.ok()) {
    return to_grpc_status(resolved_plan_or.status());
  }
  auto resolved_plan = std::move(*resolved_plan_or);
  const auto materialize_start = std::chrono::steady_clock::now();
  auto result_or = d_.engine.materialize_mapped_into_target(device, resolved_plan, hints, disk_source);
  const auto engine_done = std::chrono::steady_clock::now();
  if (!result_or.ok()) {
    LOG(ERROR) << "MaterializeIntoMappedTarget engine failure"
               << " artifact_id=" << resolved_artifact_id << " copy_entries=" << req.copy_plan().entries_size()
               << " dst_tensors=" << req.dst_tensors_size() << " storages=" << layout.storages_size()
               << " status=" << result_or.status();
    if (absl::IsDataLoss(result_or.status())) {
      for (const auto& region_id : storage_lease.acquired_region_ids()) {
        d_.regions.mark_poisoned(region_id).IgnoreError();
      }
      record_materialize_into_target(
          "error", "transfer_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    } else {
      record_materialize_into_target(
          "error", "transfer_error", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    }
    return to_grpc_status(result_or.status());
  }

  resp.set_artifact_id(resolved_artifact_id);
  resp.set_status(v2::MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  resp.set_source(to_proto_source(result_or->source));
  resp.set_canonical_index_bytes(canonical_index_json);
  if (view_plan.has_value() && !view_plan->is_identity) {
    resp.set_view_index_bytes(view_plan->view_index_json);
  }
  if (resolved_selection.tensor_names_size() > 0 || !resolved_selection.view_subset_hash().empty()) {
    auto* subset = resp.mutable_view_subset();
    if (!resolved_selection.view_subset_hash().empty()) {
      subset->set_subset_hash(resolved_selection.view_subset_hash());
    }
    for (const auto& name : resolved_selection.tensor_names()) {
      subset->add_tensor_names(name);
    }
  }
  resp.mutable_resolved_selection()->CopyFrom(resolved_selection);
  resp.set_generation(generation);
  if (capability_tokens_ != nullptr && capability_tokens_->configured()) {
    tensorcast::common::v1::ByteSpaceRef byte_space;
    if (!resolved_selection.view_id().empty()) {
      byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
      byte_space.set_id(resolved_selection.view_id());
    } else {
      byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
      byte_space.set_id("");
    }

    const std::string layout_hash = compute_target_layout_hash(layout);
    const std::string publication_id = mint_publication_id();
    const absl::Time expires_at = absl::Now() + TargetPublishService::target_publication_token_ttl();

    auto stable_index_or = store::loader::rebuild_stable_canonical_index(canonical_index_json, device.ordinal);
    if (!stable_index_or.ok()) {
      VLOG(1) << "MaterializeIntoMappedTarget: failed to rebuild canonical index for target publication token: "
              << stable_index_or.status();
    } else {
      std::string stable_index_json = std::move(*stable_index_or);
      const auto digest = common::sha256_digest_bytes(
          absl::Span<const uint8_t>(
              reinterpret_cast<const uint8_t*>(stable_index_json.data()), stable_index_json.size()));
      std::string index_key_hex =
          absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));

      tensorcast::common::v1::TargetPublicationScope scope;
      scope.set_publication_id(publication_id);
      scope.mutable_selection()->CopyFrom(resolved_selection);
      scope.mutable_byte_space()->CopyFrom(byte_space);
      scope.set_device_uuid(req.device_uuid());
      scope.set_owner_pid(req.pid());
      scope.set_target_layout_hash(layout_hash);
      if (req.has_operation_id()) {
        scope.set_operation_id(req.operation_id());
      }

      auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
      if (scope_or.ok()) {
        const uint64_t expires_at_ms = static_cast<uint64_t>(absl::ToUnixMillis(expires_at));
        auto token_or = capability_tokens_->mint(
            d_.identity.daemon_id(),
            tensorcast::common::v1::CAPABILITY_AUDIENCE_TARGET_PUBLICATION,
            *scope_or,
            expires_at_ms);
        if (token_or.ok()) {
          TargetPublicationRegistry::Record record;
          record.publication_id = PublicationInstanceId{.value = publication_id};
          record.publication_subject_key =
              build_publication_subject_key(resolved_selection, byte_space, layout_hash, req.device_uuid());
          record.target_layout_hash = layout_hash;
          record.selection = resolved_selection;
          record.byte_space = byte_space;
          record.canonical_index_json = std::move(stable_index_json);
          record.index_key_hex = std::move(index_key_hex);
          record.device_uuid = req.device_uuid();
          record.owner_pid = req.pid();
          if (req.has_operation_id()) {
            record.request_operation_id = req.operation_id();
          }
          record.expires_at = expires_at;
          record.segments = std::move(publish_segments);
          record.storages = std::move(publish_storages);
          auto inserted_or = target_publish_service_.remember_target_publication(std::move(record));
          if (inserted_or.ok()) {
            resp.set_target_publication_token(*token_or);
          } else {
            VLOG(1) << "MaterializeIntoMappedTarget: failed to register target publication lifecycle: "
                    << inserted_or.status();
          }
        } else {
          VLOG(1) << "MaterializeIntoMappedTarget: failed to mint target_publication_token: " << token_or.status();
        }
      } else {
        VLOG(1) << "MaterializeIntoMappedTarget: failed to serialize target_publication scope: " << scope_or.status();
      }
    }
  }
  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.mapped.entries", static_cast<int64_t>(req.copy_plan().entries_size()));
    span->SetAttribute("tc.mapped.bytes", static_cast<int64_t>(mapped_plan.representation.total_bytes_copied));
  }
  const double materialize_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - materialize_start).count();
  const double total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  LOG(INFO) << "MaterializeIntoMappedTarget controller timings"
            << " artifact_id=" << resolved_artifact_id << " copy_entries=" << req.copy_plan().entries_size()
            << " dst_tensors=" << req.dst_tensors_size() << " storages=" << layout.storages_size()
            << " common_sec=" << std::chrono::duration<double>(common_done - total_start).count()
            << " target_validate_sec=" << std::chrono::duration<double>(target_validate_done - common_done).count()
            << " offsets_sec=" << std::chrono::duration<double>(offsets_done - target_validate_done).count()
            << " canonical_sec=" << std::chrono::duration<double>(canonical_done - offsets_done).count()
            << " mapped_plan_sec=" << std::chrono::duration<double>(mapped_plan_done - canonical_done).count()
            << " disk_metadata_sec=" << std::chrono::duration<double>(disk_metadata_done - mapped_plan_done).count()
            << " engine_sec=" << std::chrono::duration<double>(engine_done - disk_metadata_done).count()
            << " materialize_sec=" << materialize_sec << " total_sec=" << total_sec;
  LOG(INFO) << "MaterializeIntoMappedTarget completed"
            << " artifact_id=" << resolved_artifact_id
            << " source_layout=" << (disk_metadata.has_value() && disk_metadata->source_index_json.has_value())
            << " collective=" << hints.collective_load_group.has_value()
            << " source=" << v2::MaterializationSource_Name(resp.source()) << " target_bytes=" << logical_total_size
            << " materialize_sec=" << materialize_sec << " total_sec=" << total_sec;
  record_materialize_into_target("ok", "ok", resp.source());
  rctx.mark_success();
  return Status::OK;
}

grpc::Status TargetMaterializationService::publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::PublishTargetReplicaResponse& resp) {
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  return target_publish_service_.publish_target_replica(rctx, req, resp);
}

grpc::Status TargetMaterializationService::start_publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::StartPublishTargetReplicaResponse& resp) {
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  return target_publish_service_.start_publish_target_replica(rctx, req, resp);
}

absl::StatusOr<TargetPublicationRegistry::Record> TargetMaterializationService::remember_target_publication(
    TargetPublicationRegistry::Record record) {
  return target_publish_service_.remember_target_publication(std::move(record));
}

absl::Status TargetMaterializationService::admit_public_operation(
    const tensorcast::operation::v1::OperationRef& operation_ref,
    absl::Time now) const {
  return target_publish_service_.admit_public_operation(operation_ref, now);
}

} // namespace tensorcast::daemon
