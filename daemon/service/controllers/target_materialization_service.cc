// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/target_materialization_service.h"

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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/selection_identity.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "daemon/service/controllers/materialization_index_source_utils.h"
#include "daemon/service/controllers/materialization_layout_utils.h"
#include "daemon/service/controllers/materialization_payload_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/service/controllers/materialization_request_common_utils.h"
#include "daemon/service/controllers/materialization_target_plan_utils.h"
#include "daemon/service/controllers/materialization_target_storage_utils.h"
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
using materialization_index_source::parse_mi2_data_multihash;
using materialization_index_source::TargetLayoutSpan;
using materialization_layout::resolve_target_offsets;
using materialization_payload::compute_generation_from_index;
using materialization_policy::build_view_spec_proto;
using materialization_policy::resolve_source_policy;
using materialization_policy::resolve_transform_placement;
using materialization_policy::ResolvedSourcePolicy;
using materialization_policy::to_hint_preference;
using materialization_policy::validate_source_policy;
using materialization_request_common::resolve_artifact_binding;
using materialization_request_common::resolve_managed_disk_path;
using materialization_target_plan::build_mapped_target_materialization_plan;
using materialization_target_plan::build_target_materialization_plan;
using materialization_target_plan::MappedTargetMaterializationPlan;
using materialization_target_plan::TargetMaterializationPlan;
using materialization_target_storage::acquire_error_reason;
using materialization_target_storage::AcquireTargetStoragesError;
using materialization_target_storage::TargetStorageLease;
using store::loading::MaterializationSource;

std::string mint_write_id() {
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

template <typename RequestT>
absl::StatusOr<TargetMaterializationCommonContext> prepare_target_materialization_common(
    const RequestT& req,
    std::string_view peer,
    std::string_view rpc_name,
    const std::shared_ptr<store::components::IGlobalStoreClient>& global_store_client,
    LocalDiskImportCatalog& disk_imports,
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

  const bool has_artifact_id = req.has_artifact_id() && !req.artifact_id().empty();
  const bool has_key = req.has_key() && !req.key().empty();
  if (has_key) {
    record_materialize_into_target(
        "error", "key_not_supported", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::InvalidArgumentError(std::format("key-based requests not supported for {}", rpc_name));
  }
  if (!has_artifact_id) {
    record_materialize_into_target(
        "error", "missing_artifact_id", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::InvalidArgumentError(std::format("artifact_id is required for {}", rpc_name));
  }

  context.resolved_artifact_id = req.artifact_id();
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
    auto entry = disk_imports.lookup_import(context.resolved_artifact_id);
    if (entry.has_value()) {
      context.normalized_disk_path = std::filesystem::path(entry->normalized_disk_path);
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
              .global_store_client = d_.global_store_client,
              .capability_tokens = d_.capability_tokens,
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

TargetWriteRegistry::Record TargetMaterializationService::insert_target_write_for_testing(
    TargetWriteRegistry::Record record) {
  return target_publish_service_.remember_target_write(std::move(record));
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

  const auto device = d_.devices.From(v2::DeviceType::DEVICE_TYPE_GPU, req.device_uuid(), std::nullopt);
  for (const auto& storage : layout.storages()) {
    if (storage.storage_source_case() != v2::StorageEntry::kVramRegionId) {
      record_materialize_into_target(
          "error", "storage_not_region", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "Target storage must reference a vram_region_id"};
    }
    if (storage.device_id() != device.ordinal) {
      record_materialize_into_target(
          "error", "device_uuid_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "storage.device_id does not match device_uuid"};
    }
  }

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
      d_.engine, req, layout, offsets, std::move(*canonical_json_or), record_materialize_into_target, plan);
  if (!build_plan_status.ok()) {
    return build_plan_status;
  }

  const auto& layout_names = plan.layout_names;
  const bool has_subset = plan.has_subset;
  auto& view_spec = plan.view_spec;
  auto& view_plan = plan.view_plan;
  auto& resolved_view_id = plan.resolved_view_id;
  auto& view_data_hash = plan.view_data_hash;
  const bool has_view_transform = plan.has_view_transform;
  const std::string& canonical_index_json = plan.canonical_index_json;
  const std::string& selected_index_json = plan.selected_index_json;
  const uint64_t logical_total_size = plan.logical_total_size;
  std::vector<RegisterStorageMeta> publish_storages = std::move(plan.publish_storages);
  std::vector<LeaseSegMeta> publish_segments = std::move(plan.publish_segments);
  std::string view_subset_hash = std::move(plan.view_subset_hash);
  std::optional<tensorcast::common::v1::ViewSpec> view_spec_proto;
  if (view_spec.has_value()) {
    view_spec_proto = build_view_spec_proto(*view_spec);
  }

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

  AcquireTargetStoragesError acquire_error = AcquireTargetStoragesError::kUnknown;
  auto storage_lease_or = TargetStorageLease::acquire(d_.regions, layout.storages(), req.pid(), &acquire_error);
  if (!storage_lease_or.ok()) {
    record_materialize_into_target(
        "error", acquire_error_reason(acquire_error), v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(storage_lease_or.status());
  }
  TargetStorageLease storage_lease = std::move(*storage_lease_or);

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

  store::loading::MaterializeHints hints;
  hints.artifact_id = resolved_artifact_id;
  hints.source_preference = to_hint_preference(effective_policy.preference);
  hints.allow_p2p = effective_policy.allow_p2p;
  hints.allow_disk = effective_policy.allow_disk;
  hints.verify = store::loading::MaterializeHints::Verify::NONE;

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
  if (!layout_names.empty()) {
    auto* subset = resp.mutable_view_subset();
    if (!view_subset_hash.empty()) {
      subset->set_subset_hash(view_subset_hash);
    }
    for (const auto& name : layout_names) {
      subset->add_tensor_names(name);
    }
  }
  resp.set_generation(generation);
  if (capability_tokens_ != nullptr && capability_tokens_->configured()) {
    const std::string view_id_value = resolved_view_id.value_or("");
    const bool needs_view_index = layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW;
    const std::string logical_layout_hash = common::compute_logical_layout_hash_bytes(
        absl::Span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(selected_index_json.data()), selected_index_json.size()),
        needs_view_index);
    std::optional<std::string_view> subset_hash_opt;
    if (!view_subset_hash.empty()) {
      subset_hash_opt = view_subset_hash;
    }
    const std::string selection_hash = common::compute_selection_hash_bytes(view_id_value, subset_hash_opt);

    tensorcast::common::v1::ArtifactSelection selection;
    selection.set_artifact_id(resolved_artifact_id);
    selection.set_view_id(view_id_value);
    selection.set_logical_layout_hash(logical_layout_hash);
    selection.set_selection_hash(selection_hash);
    if (!view_subset_hash.empty()) {
      selection.set_view_subset_hash(view_subset_hash);
    }
    for (const auto& name : req.tensor_names()) {
      selection.add_tensor_names(name);
    }
    if (view_spec_proto.has_value()) {
      selection.mutable_view_spec()->CopyFrom(*view_spec_proto);
    }

    tensorcast::common::v1::ByteSpaceRef byte_space;
    if (!view_id_value.empty()) {
      byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
      byte_space.set_id(view_id_value);
    } else {
      byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
      byte_space.set_id("");
    }

    const std::string layout_hash = compute_target_layout_hash(layout);
    const std::string write_id = mint_write_id();
    const absl::Time expires_at = absl::Now() + TargetPublishService::target_write_token_ttl();

    auto stable_index_or = store::loader::rebuild_stable_canonical_index(canonical_index_json, device.ordinal);
    if (!stable_index_or.ok()) {
      VLOG(1) << "MaterializeIntoTarget: failed to rebuild canonical index for target write token: "
              << stable_index_or.status();
    } else {
      std::string stable_index_json = std::move(*stable_index_or);
      const auto digest = common::sha256_digest_bytes(
          absl::Span<const uint8_t>(
              reinterpret_cast<const uint8_t*>(stable_index_json.data()), stable_index_json.size()));
      std::string index_key_hex =
          absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));

      tensorcast::common::v1::TargetWriteScope scope;
      scope.set_write_id(write_id);
      scope.mutable_selection()->CopyFrom(selection);
      scope.mutable_byte_space()->CopyFrom(byte_space);
      scope.set_device_uuid(req.device_uuid());
      scope.set_owner_pid(req.pid());
      scope.set_target_layout_hash(layout_hash);

      auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
      if (scope_or.ok()) {
        const uint64_t expires_at_ms = static_cast<uint64_t>(absl::ToUnixMillis(expires_at));
        auto token_or = capability_tokens_->mint(
            d_.identity.daemon_id(),
            tensorcast::common::v1::CAPABILITY_AUDIENCE_TARGET_WRITE,
            *scope_or,
            expires_at_ms);
        if (token_or.ok()) {
          TargetWriteRegistry::Record record;
          record.write_id = write_id;
          record.layout_key = layout_hash;
          record.target_layout_hash = layout_hash;
          record.selection = selection;
          record.byte_space = byte_space;
          record.canonical_index_json = std::move(stable_index_json);
          record.index_key_hex = std::move(index_key_hex);
          record.device_uuid = req.device_uuid();
          record.owner_pid = req.pid();
          if (req.has_operation_id()) {
            record.operation_id = req.operation_id();
          }
          record.expires_at = expires_at;
          record.segments = std::move(publish_segments);
          record.storages = std::move(publish_storages);
          auto inserted = target_publish_service_.remember_target_write(std::move(record));
          (void)inserted;
          resp.set_target_write_token(*token_or);
        } else {
          VLOG(1) << "MaterializeIntoTarget: failed to mint target_write_token: " << token_or.status();
        }
      } else {
        VLOG(1) << "MaterializeIntoTarget: failed to serialize target_write scope: " << scope_or.status();
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

  const auto device = d_.devices.From(v2::DeviceType::DEVICE_TYPE_GPU, req.device_uuid(), std::nullopt);
  for (const auto& storage : layout.storages()) {
    if (storage.storage_source_case() != v2::StorageEntry::kVramRegionId) {
      record_materialize_into_target(
          "error", "storage_not_region", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "Target storage must reference a vram_region_id"};
    }
    if (storage.device_id() != device.ordinal) {
      record_materialize_into_target(
          "error", "device_uuid_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "storage.device_id does not match device_uuid"};
    }
  }

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

  MappedTargetMaterializationPlan mapped_plan;
  auto build_mapped_plan_status = build_mapped_target_materialization_plan(
      d_.engine,
      req,
      resolved_artifact_id,
      offsets,
      std::move(*canonical_json_or),
      record_materialize_into_target,
      mapped_plan);
  if (!build_mapped_plan_status.ok()) {
    return build_mapped_plan_status;
  }
  const uint64_t logical_total_size = mapped_plan.logical_total_size;
  const std::string& canonical_index_json = mapped_plan.canonical_index_json;
  auto& view_spec = mapped_plan.view_spec;
  auto& view_plan = mapped_plan.view_plan;
  auto copy_plan = std::move(mapped_plan.copy_plan);

  AcquireTargetStoragesError acquire_error = AcquireTargetStoragesError::kUnknown;
  auto storage_lease_or = TargetStorageLease::acquire(d_.regions, layout.storages(), req.pid(), &acquire_error);
  if (!storage_lease_or.ok()) {
    record_materialize_into_target(
        "error", acquire_error_reason(acquire_error), v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(storage_lease_or.status());
  }
  TargetStorageLease storage_lease = std::move(*storage_lease_or);

  std::optional<store::loading::DiskSource> disk_source;
  if (normalized_disk_path.has_value()) {
    disk_source = store::loading::DiskSource{
        .path = *normalized_disk_path,
        .expected_size = logical_total_size,
        .require_descriptor = true,
    };
  }

  store::loading::MaterializeHints hints;
  hints.artifact_id = resolved_artifact_id;
  hints.source_preference = to_hint_preference(effective_policy.preference);
  hints.allow_p2p = effective_policy.allow_p2p;
  hints.allow_disk = effective_policy.allow_disk;
  hints.verify = store::loading::MaterializeHints::Verify::NONE;

  if (view_plan.has_value()) {
    if (view_plan->transform.requires_materialization) {
      return {StatusCode::INVALID_ARGUMENT, "mapped binding does not support view transforms"};
    }
    store::loading::VariantIdentity variant;
    variant.canonical_artifact_id = resolved_artifact_id;
    if (view_spec.has_value()) {
      variant.view_spec = view_spec;
    }
    variant.cached_plan = view_plan;
    variant.canonical_index_json = canonical_index_json;
    variant.placement = resolve_transform_placement(req.placement(), view_spec);
    hints.variant = std::move(variant);
  }

  store::loading::IntoTargetLayout target_layout;
  target_layout.storages.assign(storage_lease.storages().begin(), storage_lease.storages().end());
  target_layout.total_size = logical_total_size;

  const uint64_t generation = compute_generation_from_index(canonical_index_json);
  auto result_or = d_.engine.materialize_mapped_into_target(
      device, target_layout, copy_plan.map, canonical_index_json, generation, hints, disk_source);
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

  resp.set_artifact_id(resolved_artifact_id);
  resp.set_status(v2::MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  resp.set_source(to_proto_source(result_or->source));
  resp.set_canonical_index_bytes(canonical_index_json);
  if (view_plan.has_value() && !view_plan->is_identity) {
    resp.set_view_index_bytes(view_plan->view_index_json);
  }
  resp.set_generation(generation);
  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.mapped.entries", static_cast<int64_t>(req.copy_plan().entries_size()));
    span->SetAttribute("tc.mapped.bytes", static_cast<int64_t>(copy_plan.total_bytes_copied));
  }
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

} // namespace tensorcast::daemon
