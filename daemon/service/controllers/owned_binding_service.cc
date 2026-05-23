// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/owned_binding_service.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/runtime/ingestion/source_bound_strategy_planner.h"
#include "daemon/service/controllers/assembly_coordination_utils.h"
#include "daemon/service/controllers/materialization_disk_resolve_utils.h"
#include "daemon/service/controllers/materialization_index_source_utils.h"
#include "daemon/service/controllers/materialization_layout_utils.h"
#include "daemon/service/controllers/materialization_payload_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/service/controllers/materialization_request_common_utils.h"
#include "daemon/service/controllers/materialization_target_plan_utils.h"
#include "daemon/service/controllers/registration_controller.h"
#include "daemon/service/controllers/registration_storage_mapping_utils.h"
#include "daemon/service/controllers/serving_artifact_manifest_utils.h"
#include "daemon/util/grpc_peer_utils.h"
#include "daemon/util/path_utils.h"
#include "daemon/util/status_utils.h"

#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "core/common/selection_identity.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_ipc.h"
#include "core/cuda/device_guard.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/view_utils.h"
#include "folly/futures/Future.h"
#include "google/protobuf/message_lite.h"
#include "gsl/pointers"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

using materialization_index_source::load_canonical_index_with_disk_fallback;
using materialization_index_source::load_descriptor_metadata;
using materialization_layout::parse_canonical_index;
using materialization_layout::resolve_target_offsets;
using materialization_payload::build_descriptors_from_index;
using materialization_policy::apply_group_realization_begin_context_to_transport_context;
using materialization_policy::apply_operation_transport_context;
using materialization_policy::apply_request_context_to_hints;
using materialization_policy::begin_or_join_group_realization_if_enabled;
using materialization_policy::build_execution_diagnostics;
using materialization_policy::build_view_spec_proto;
using materialization_policy::collective_policy_requests_collective;
using materialization_policy::compute_view_id_from_spec;
using materialization_policy::GroupRealizationBeginContext;
using materialization_policy::GroupRealizationPreparedMemberContext;
using materialization_policy::HashExecutionDetails;
using materialization_policy::NormalizedMaterializationRequestContext;
using materialization_policy::OperationTransportContext;
using materialization_policy::report_group_realization_prepared_if_enabled;
using materialization_policy::resolve_collective_policy;
using materialization_policy::resolve_group_realization_transport_context;
using materialization_policy::resolve_materialization_request_context;
using materialization_policy::resolve_source_execution_topology;
using materialization_policy::resolve_transform_placement;
using materialization_policy::validate_group_realization_staged_publish_supported;
using materialization_request_common::resolve_artifact_and_disk_source;
using materialization_target_plan::build_binding_realization_materialization_plan;
using materialization_target_plan::build_mapped_target_materialization_plan;
using materialization_target_plan::build_resolved_mapped_materialization_plan;
using materialization_target_plan::build_target_materialization_plan;
using materialization_target_plan::MappedTargetMaterializationPlan;
using materialization_target_plan::TargetMaterializationPlan;
using store::loading::MaterializationSource;
namespace coordination = assembly_coordination;

constexpr size_t kBindingRealizationPlanCacheMaxEntries = 8;

std::string hash_cache_payload(std::string_view payload) {
  const std::vector<uint8_t> digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return common::multibase_multihash_sha256(digest);
}

void append_cache_field(std::string* payload, std::string_view value) {
  absl::StrAppend(payload, value.size(), ":");
  payload->append(value.data(), value.size());
  payload->push_back('|');
}

void append_cache_uint64(std::string* payload, uint64_t value) {
  absl::StrAppend(payload, value, "|");
}

std::optional<std::string_view> mi2_index_multihash(std::string_view artifact_id) {
  constexpr std::string_view kPrefix = "mi2:";
  if (!artifact_id.starts_with(kPrefix)) {
    return std::nullopt;
  }
  const std::string_view tail = artifact_id.substr(kPrefix.size());
  const size_t sep = tail.find(':');
  if (sep == std::string_view::npos || sep == 0 || sep + 1 >= tail.size()) {
    return std::nullopt;
  }
  return tail.substr(0, sep);
}

bool source_artifact_matches_target_index(std::string_view artifact_id, std::string_view target_index_json) {
  if (target_index_json.empty()) {
    return false;
  }
  const std::optional<std::string_view> source_index_multihash = mi2_index_multihash(artifact_id);
  if (!source_index_multihash.has_value()) {
    return false;
  }
  auto target_index_multihash_or =
      common::compute_index_multihash(std::optional<std::string>(std::string(target_index_json)), "");
  if (!target_index_multihash_or.ok()) {
    LOG(WARNING) << "Failed to compute binding target index multihash for direct refill: "
                 << target_index_multihash_or.status();
    return false;
  }
  return *source_index_multihash == *target_index_multihash_or;
}

bool selection_declares_view_or_subset(const tensorcast::common::v1::ArtifactSelection& selection) {
  return !selection.view_id().empty() || selection.has_view_spec() || !selection.view_subset_hash().empty() ||
      selection.tensor_names_size() > 0;
}

v2::TargetLayout build_direct_refill_target_plan_layout(
    const v2::TargetLayout& target_layout,
    const tensorcast::common::v1::ArtifactSelection& selection,
    bool source_artifact_index_matches_target) {
  v2::TargetLayout layout = target_layout;
  if (!source_artifact_index_matches_target || selection_declares_view_or_subset(selection)) {
    layout.set_index_kind(v2::TargetLayout::INDEX_KIND_VIEW);
    if (!selection.view_id().empty()) {
      layout.set_view_id(selection.view_id());
    } else {
      layout.clear_view_id();
    }
  }
  return layout;
}

std::string serialize_proto_for_cache_key(const google::protobuf::MessageLite& proto) {
  std::string out;
  proto.SerializeToString(&out);
  return out;
}

std::string compute_target_layout_geometry_hash(const v2::TargetLayout& layout) {
  std::string payload;
  absl::StrAppend(
      &payload,
      "layout_kind=",
      static_cast<int>(layout.layout_kind()),
      "|index_kind=",
      static_cast<int>(layout.index_kind()),
      "|tensor_spec_kind=",
      static_cast<int>(layout.tensor_spec_kind()),
      "|view_id=");
  append_cache_field(&payload, layout.view_id());
  append_cache_field(&payload, layout.logical_layout_hash());
  for (const auto& storage : layout.storages()) {
    append_cache_field(&payload, storage.storage_id());
    append_cache_uint64(&payload, static_cast<uint64_t>(storage.device_id()));
    append_cache_uint64(&payload, storage.storage_length());
    append_cache_uint64(&payload, storage.mapping_base_offset());
  }
  for (const auto& entry : layout.offsets()) {
    append_cache_field(&payload, entry.name());
    append_cache_field(&payload, entry.storage_id());
    append_cache_uint64(&payload, entry.storage_offset());
    append_cache_uint64(&payload, entry.logical_length());
  }
  return hash_cache_payload(payload);
}

std::string binding_realization_plan_cache_key(
    std::string_view resolved_artifact_id,
    const tensorcast::common::v1::ArtifactSelection& selection,
    const v2::BindingRealizationPlan& realization_plan,
    const v2::TargetLayout& target_layout,
    std::string_view target_index_json,
    std::string_view canonical_index_json,
    v2::TransformPlacement placement) {
  std::string payload;
  append_cache_field(&payload, "binding-realization-plan-v1");
  append_cache_field(&payload, resolved_artifact_id);
  append_cache_field(&payload, serialize_proto_for_cache_key(selection));
  append_cache_field(&payload, serialize_proto_for_cache_key(realization_plan));
  append_cache_field(&payload, compute_target_layout_geometry_hash(target_layout));
  append_cache_field(&payload, target_index_json);
  append_cache_field(&payload, canonical_index_json);
  append_cache_uint64(&payload, static_cast<uint64_t>(placement));
  return hash_cache_payload(payload);
}

void append_materialization_strategy_cache_key(
    std::string* payload,
    const store::StoreEngineOptions::MaterializationStrategyConfig& config) {
  append_cache_uint64(payload, config.enable_tensor_aware_mapped_executor ? 1 : 0);
  append_cache_uint64(payload, config.enable_local_batched_disk_load ? 1 : 0);
  append_cache_uint64(payload, config.enable_owner_file_collective ? 1 : 0);
  append_cache_uint64(payload, config.allow_mixed_execution ? 1 : 0);
  append_cache_uint64(payload, config.prefer_local_canonical_for_mapped ? 1 : 0);
  append_cache_uint64(payload, config.allow_source_ordered_for_mapped ? 1 : 0);
  append_cache_uint64(payload, config.enable_mapped_dim0_tensor_jobs ? 1 : 0);
  append_cache_uint64(payload, config.enable_mapped_dim1_tensor_jobs ? 1 : 0);
  append_cache_uint64(payload, config.enable_mapped_concat_jobs ? 1 : 0);
  append_cache_uint64(payload, config.enable_mapped_concat_execution ? 1 : 0);
  append_cache_uint64(payload, config.enable_mapped_single_range_concat_jobs ? 1 : 0);
  append_cache_uint64(payload, config.enable_mapped_multirange_concat_jobs ? 1 : 0);
  append_cache_uint64(payload, config.sync_after_single_range_concat_job ? 1 : 0);
  append_cache_uint64(payload, config.use_dedicated_single_range_concat_stream ? 1 : 0);
  append_cache_uint64(payload, static_cast<uint64_t>(config.executor_preference));
  append_cache_uint64(payload, config.direct_write_batch_bytes);
  append_cache_uint64(payload, config.direct_write_batch_ops);
  append_cache_uint64(payload, config.owner_file_collective_peak_bytes_budget);
  append_cache_uint64(payload, config.owner_file_collective_batch_bytes);
  append_cache_uint64(payload, config.owner_file_collective_dim1_staging_bytes);
  append_cache_uint64(payload, config.owner_file_collective_max_inflight_batches);
  append_cache_uint64(payload, config.owner_file_collective_shared_fs_only ? 1 : 0);
  append_cache_field(payload, absl::StrCat(config.owner_file_collective_max_owner_skew_ratio));
  append_cache_uint64(payload, config.owner_file_collective_min_dedup_saving_bytes);
  append_cache_uint64(payload, static_cast<uint64_t>(config.owner_file_collective_group_assemble_timeout.count()));
  append_cache_uint64(payload, config.owner_file_collective_allow_mixed_residual ? 1 : 0);
  append_cache_uint64(payload, static_cast<uint64_t>(config.local_mapped_safetensors_io_mode));
}

void append_execution_topology_cache_key(
    std::string* payload,
    const store::loading::ExecutionTopologyContext& topology) {
  append_cache_uint64(payload, static_cast<uint64_t>(topology.source_locality));
  append_cache_field(payload, topology.source_sharing_domain.value_or(""));
  append_cache_uint64(payload, topology.collective_load_group.has_value() ? 1 : 0);
  if (topology.collective_load_group.has_value()) {
    append_cache_field(payload, topology.collective_load_group->group_id);
    append_cache_uint64(payload, topology.collective_load_group->world_size);
    append_cache_uint64(payload, topology.collective_load_group->rank);
  }
}

std::string mapped_execution_template_cache_key(
    std::string_view plan_key,
    const std::optional<store::loading::DiskMetadata>& disk_metadata,
    v2::CollectivePolicy collective_policy,
    const store::StoreEngineOptions::MaterializationStrategyConfig& strategy_config,
    const store::loading::ExecutionTopologyContext& topology,
    bool disk_source_available) {
  std::string payload;
  append_cache_field(&payload, "mapped-execution-template-v1");
  append_cache_field(&payload, plan_key);
  append_cache_field(
      &payload,
      disk_metadata.has_value() && disk_metadata->source_index_json.has_value()
          ? std::string_view(*disk_metadata->source_index_json)
          : std::string_view());
  append_cache_uint64(&payload, static_cast<uint64_t>(collective_policy));
  append_cache_uint64(&payload, disk_source_available ? 1 : 0);
  append_cache_uint64(&payload, disk_metadata.has_value() && disk_metadata->is_safetensors.value_or(false) ? 1 : 0);
  append_materialization_strategy_cache_key(&payload, strategy_config);
  append_execution_topology_cache_key(&payload, topology);
  return hash_cache_payload(payload);
}

struct CachedMappedExecutionTemplate {
  std::optional<store::materialization::contracts::RepresentationWorkPlan> representation_work_plan;
  std::optional<store::runtime::ingestion::strategy::SourceBoundStrategyPlan> strategy_plan;
};

struct BindingRealizationPlanCacheState {
  absl::Mutex mu;
  absl::flat_hash_map<std::string, MappedTargetMaterializationPlan> mapped_plans ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::string, CachedMappedExecutionTemplate> execution_templates ABSL_GUARDED_BY(mu);
  std::vector<std::string> mapped_plan_order ABSL_GUARDED_BY(mu);
  std::vector<std::string> execution_template_order ABSL_GUARDED_BY(mu);
};

BindingRealizationPlanCacheState& binding_realization_plan_cache() {
  static auto* state = new BindingRealizationPlanCacheState();
  return *state;
}

template <typename Value>
void put_bounded_binding_realization_cache_entry(
    absl::flat_hash_map<std::string, Value>* map,
    std::vector<std::string>* insertion_order,
    std::string key,
    Value value) {
  if (map == nullptr || insertion_order == nullptr || key.empty()) {
    return;
  }
  if (!map->contains(key)) {
    if (map->size() >= kBindingRealizationPlanCacheMaxEntries && !insertion_order->empty()) {
      map->erase(insertion_order->front());
      insertion_order->erase(insertion_order->begin());
    }
    insertion_order->push_back(key);
  }
  (*map)[std::move(key)] = std::move(value);
}

std::chrono::milliseconds resolve_owner_request_budget(const grpc::ServerContext& server_context) {
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

std::string mint_random_id(size_t bytes_len) {
  thread_local absl::BitGen bitgen;
  std::string raw;
  raw.resize(bytes_len);
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<char>(absl::Uniform<uint32_t>(bitgen, 0u, 256u));
  }
  return absl::BytesToHexString(raw);
}

std::string mint_binding_id() {
  return mint_random_id(16);
}

std::string mint_publication_id() {
  return mint_random_id(16);
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

bool selection_requires_tensor_aware_metadata(const tensorcast::common::v1::ArtifactSelection& selection) {
  return selection.has_view_spec() || !selection.view_id().empty() || selection.tensor_names_size() > 0 ||
      !selection.view_subset_hash().empty();
}

bool is_byte_only_disk_metadata(const std::optional<store::loading::DiskMetadata>& disk_metadata) {
  return disk_metadata.has_value() && disk_metadata->tensor_aware.has_value() && !*disk_metadata->tensor_aware;
}

std::optional<bool> public_disk_source_is_safetensors(const v2::PublicDiskSourceHandle& public_disk_source) {
  switch (public_disk_source.format_kind()) {
    case v2::DISK_SOURCE_FORMAT_KIND_PARTITIONED:
      return false;
    case v2::DISK_SOURCE_FORMAT_KIND_SAFETENSORS:
      return true;
    case v2::DISK_SOURCE_FORMAT_KIND_UNSPECIFIED:
    default:
      return std::nullopt;
  }
}

std::optional<bool> registry_source_is_safetensors(const ArtifactSourceRegistry::Entry& entry) {
  switch (entry.source_format_kind) {
    case ArtifactSourceRegistry::SourceFormatKind::kPartitioned:
      return false;
    case ArtifactSourceRegistry::SourceFormatKind::kSafetensors:
      return true;
    case ArtifactSourceRegistry::SourceFormatKind::kUnspecified:
    default:
      return std::nullopt;
  }
}

std::optional<bool> public_disk_source_is_tensor_aware(const v2::PublicDiskSourceHandle& public_disk_source) {
  switch (public_disk_source.metadata_capability()) {
    case v2::DISK_METADATA_CAPABILITY_BYTE_ONLY:
      return false;
    case v2::DISK_METADATA_CAPABILITY_TENSOR_AWARE:
      return true;
    case v2::DISK_METADATA_CAPABILITY_UNSPECIFIED:
    default:
      return std::nullopt;
  }
}

absl::StatusOr<std::optional<store::loading::DiskMetadata>> build_binding_disk_metadata(
    const std::optional<std::filesystem::path>& normalized_disk_path,
    std::string_view resolved_artifact_id,
    int device_ordinal,
    ArtifactSourceRegistry& disk_imports,
    const v2::PublicDiskSourceHandle* public_disk_source,
    const std::optional<ArtifactSourceRegistry::Entry>& resolved_source_entry) {
  (void)device_ordinal;
  std::optional<store::loading::DiskMetadata> disk_metadata;
  std::optional<ArtifactSourceRegistry::Entry> local_import = resolved_source_entry;
  if (!local_import.has_value()) {
    local_import = disk_imports.lookup_binding(resolved_artifact_id);
  }
  if (public_disk_source != nullptr && !public_disk_source->canonical_index_bytes().empty() &&
      local_import.has_value() &&
      local_import->source_kind == ArtifactSourceRegistry::SourceKind::kMountedSourceArtifact) {
    if (local_import->canonical_index_json.empty()) {
      return absl::FailedPreconditionError("public_disk_source requires daemon-resolved canonical index");
    }
    if (public_disk_source->canonical_index_bytes() != local_import->canonical_index_json) {
      return absl::FailedPreconditionError(
          "public_disk_source.canonical_index_bytes does not match daemon-resolved canonical index");
    }
    if (!public_disk_source->source_index_bytes().empty() &&
        (!local_import->source_index_json.has_value() ||
         public_disk_source->source_index_bytes() != *local_import->source_index_json)) {
      return absl::FailedPreconditionError(
          "public_disk_source.source_index_bytes does not match daemon-resolved source index");
    }
    const std::optional<bool> public_is_safetensors = public_disk_source_is_safetensors(*public_disk_source);
    const std::optional<bool> registry_is_safetensors = registry_source_is_safetensors(*local_import);
    if (public_is_safetensors.has_value() && registry_is_safetensors.has_value() &&
        *public_is_safetensors != *registry_is_safetensors) {
      return absl::FailedPreconditionError(
          "public_disk_source.format_kind does not match daemon-resolved mounted-source format");
    }
    const std::optional<bool> public_tensor_aware = public_disk_source_is_tensor_aware(*public_disk_source);
    if (public_tensor_aware.has_value() && *public_tensor_aware != local_import->tensor_aware_metadata) {
      return absl::FailedPreconditionError(
          "public_disk_source.metadata_capability does not match daemon-resolved mounted-source metadata");
    }
    store::loading::DiskMetadata metadata;
    metadata.descriptor_present = local_import->descriptor_present;
    metadata.canonical_index_json = public_disk_source->canonical_index_bytes();
    if (!public_disk_source->source_index_bytes().empty()) {
      metadata.source_index_json = public_disk_source->source_index_bytes();
    } else if (local_import->source_index_json.has_value()) {
      metadata.source_index_json = *local_import->source_index_json;
    }
    if (local_import->index_multihash.has_value()) {
      metadata.index_multihash = *local_import->index_multihash;
    }
    if (local_import->data_multihash.has_value()) {
      metadata.data_multihash = *local_import->data_multihash;
    }
    if (public_disk_source->exact_size_bytes() > 0) {
      metadata.logical_total_size = public_disk_source->exact_size_bytes();
      metadata.source_total_size_bytes = public_disk_source->exact_size_bytes();
    }
    if (public_is_safetensors.has_value()) {
      metadata.is_safetensors = *public_is_safetensors;
    } else if (registry_is_safetensors.has_value()) {
      metadata.is_safetensors = *registry_is_safetensors;
    }
    if (public_tensor_aware.has_value()) {
      metadata.tensor_aware = *public_tensor_aware;
    } else {
      metadata.tensor_aware = local_import->tensor_aware_metadata;
    }
    if (metadata.tensor_aware.value_or(false) && !metadata.is_safetensors.has_value()) {
      return absl::FailedPreconditionError(
          "public_disk_source tensor-aware mounted-source metadata requires a daemon-resolved source format");
    }
    disk_metadata = std::move(metadata);
  }
  if (normalized_disk_path.has_value()) {
    if (!disk_metadata.has_value()) {
      auto descriptor_or = load_descriptor_metadata(*normalized_disk_path);
      if (!descriptor_or.ok()) {
        return descriptor_or.status();
      }
      auto mounted_metadata_or = materialization_disk_resolve::build_mounted_source_metadata(*normalized_disk_path);
      if (!mounted_metadata_or.ok()) {
        return mounted_metadata_or.status();
      }
      store::loading::DiskMetadata metadata;
      metadata.descriptor_present = descriptor_or->found;
      metadata.schema_version = descriptor_or->schema_version;
      metadata.index_multihash = descriptor_or->index_multihash;
      metadata.data_multihash = descriptor_or->data_multihash;
      metadata.canonical_index_json = mounted_metadata_or->index_info.canonical_index_json;
      if (mounted_metadata_or->index_info.source_index_json.has_value()) {
        metadata.source_index_json = mounted_metadata_or->index_info.source_index_json;
      }
      if (!mounted_metadata_or->canonical_index_multihash.empty()) {
        metadata.index_multihash = mounted_metadata_or->canonical_index_multihash;
      }
      if (mounted_metadata_or->exact_size_bytes > 0) {
        metadata.logical_total_size = mounted_metadata_or->exact_size_bytes;
      }
      if (mounted_metadata_or->index_info.source_total_size_bytes > 0) {
        metadata.source_total_size_bytes = mounted_metadata_or->index_info.source_total_size_bytes;
      } else if (mounted_metadata_or->exact_size_bytes > 0) {
        metadata.source_total_size_bytes = mounted_metadata_or->exact_size_bytes;
      }
      metadata.is_safetensors =
          mounted_metadata_or->format_kind == materialization_disk_resolve::MountedSourceFormatKind::kSafetensors;
      metadata.tensor_aware = mounted_metadata_or->metadata_capability ==
          materialization_disk_resolve::MountedSourceMetadataCapability::kTensorAware;
      disk_metadata = std::move(metadata);
    }
  }
  if (local_import.has_value()) {
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
    if (!metadata.tensor_aware.has_value()) {
      metadata.tensor_aware = local_import->tensor_aware_metadata;
    }
  }
  return disk_metadata;
}

struct OwnedStorageLayout {
  store::loading::IntoTargetLayout into_target;
  std::vector<LeaseSegMeta> publish_segments;
  std::vector<RegisterStorageMeta> publish_storages;
  uint64_t total_size{0};
};

absl::StatusOr<OwnedStorageLayout> build_owned_storage_layout(
    const v2::TargetLayout& layout,
    int expected_device_id,
    gsl::not_null<void*> allocation_base_ptr,
    const cuda::IpcHandleBytes& handle_bytes) {
  OwnedStorageLayout result;
  const auto handle_view = handle_bytes.as_string_view();
  uint64_t cursor = 0;
  result.into_target.storages.reserve(layout.storages_size());
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
    if (storage.storage_source_case() != v2::StorageEntry::STORAGE_SOURCE_NOT_SET) {
      return absl::InvalidArgumentError("owner binding target_layout storages must not include storage sources");
    }

    auto* base_ptr = static_cast<uint8_t*>(allocation_base_ptr.get()) + cursor;
    result.into_target.storages.push_back(
        store::loading::IntoTargetStorage{
            .base_ptr = gsl::not_null<void*>{base_ptr},
            .length = storage.storage_length(),
        });

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
  result.into_target.total_size = cursor;
  result.total_size = cursor;
  if (cursor == 0) {
    return absl::InvalidArgumentError("target_layout storages must be non-empty");
  }
  return result;
}

absl::StatusOr<OwnedStorageLayout> build_client_binding_publication_storage_layout(
    const v2::TargetLayout& layout,
    int expected_device_id) {
  OwnedStorageLayout result;
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
    meta.storage_length = storage.storage_length();
    meta.mapping_base_offset = storage.mapping_base_offset();
    switch (storage.storage_source_case()) {
      case v2::StorageEntry::kVramRegionId:
        meta.region_id = storage.vram_region_id();
        break;
      case v2::StorageEntry::kRegionRef:
        if (storage.region_ref().memory_kind() != v2::REGION_MEMORY_KIND_VRAM) {
          return absl::InvalidArgumentError("client binding publication only supports VRAM target regions");
        }
        meta.region_id = storage.region_ref().region_id();
        break;
      case v2::StorageEntry::kCudaIpcHandle:
        meta.handle_bytes = storage.cuda_ipc_handle();
        break;
      case v2::StorageEntry::STORAGE_SOURCE_NOT_SET:
      default:
        return absl::InvalidArgumentError("client binding target storage must reference a publishable source");
    }
    if (meta.handle_bytes.empty() == meta.region_id.empty()) {
      return absl::InvalidArgumentError("client binding storage must specify exactly one publishable source");
    }
    result.publish_storages.push_back(std::move(meta));

    LeaseSegMeta seg;
    seg.storage_id = storage.storage_id();
    seg.storage_offset = 0;
    seg.artifact_offset = cursor;
    seg.length = storage.storage_length();
    result.publish_segments.push_back(std::move(seg));

    if (storage.storage_length() > std::numeric_limits<uint64_t>::max() - cursor) {
      return absl::InvalidArgumentError("storage_length sum overflow");
    }
    cursor += storage.storage_length();
  }
  result.into_target.total_size = cursor;
  result.total_size = cursor;
  if (cursor == 0) {
    return absl::InvalidArgumentError("target_layout storages must be non-empty");
  }
  return result;
}

struct ContributionStorageLayout {
  std::vector<LeaseSegMeta> segments;
  std::vector<RegisterStorageMeta> storages;
  uint64_t total_size{0};
};

struct PreparedContributionSegment {
  int device_id{0};
  cuda::IpcMapping* mapping{nullptr};
  uint64_t base_offset{0};
  uint64_t artifact_offset{0};
  uint64_t length{0};
};

struct OpenedContributionLayout {
  std::vector<PreparedContributionSegment> segments;
  absl::flat_hash_map<std::string, std::unique_ptr<cuda::IpcMapping>> mapping_cache;
  std::unique_ptr<RegionPinGuard> region_pin;
};

struct PieceContributionViewSpec {
  tensorcast::common::v1::ViewSpec spec;
  std::vector<std::string> tensor_names;
  std::string view_id;
  uint64_t canonical_size_bytes{0};
};

struct BindingContributionRegistrationPlan {
  v2::BeginRegisterArtifactRequest begin_request;
  ContributionStorageLayout storage_layout;
  std::string slot_id;
  std::optional<std::string> structural_view_id;
};

constexpr uint64_t kContributionFeedChunkBytes = 4ULL * 1024ULL * 1024ULL;

absl::StatusOr<ContributionStorageLayout> build_contribution_storage_layout(const BindingRegistry::Record& record) {
  ContributionStorageLayout result;
  result.segments.reserve(record.target_layout.storages_size());
  result.storages.reserve(record.target_layout.storages_size());

  uint64_t cursor = 0;
  const auto handle_view = record.handle_bytes.as_string_view();
  for (const auto& storage : record.target_layout.storages()) {
    if (storage.storage_id().empty()) {
      return absl::InvalidArgumentError("storage_id is required");
    }
    if (storage.storage_length() == 0) {
      return absl::InvalidArgumentError("storage_length must be non-zero");
    }
    if (storage.device_id() != record.device_id) {
      return absl::InvalidArgumentError("storage.device_id does not match binding device");
    }

    RegisterStorageMeta meta;
    meta.storage_id = storage.storage_id();
    meta.device_id = storage.device_id();
    meta.storage_length = storage.storage_length();
    if (storage.storage_source_case() == v2::StorageEntry::kVramRegionId) {
      meta.region_id = storage.vram_region_id();
      meta.mapping_base_offset = storage.mapping_base_offset();
    } else if (storage.storage_source_case() == v2::StorageEntry::kCudaIpcHandle) {
      meta.handle_bytes = storage.cuda_ipc_handle();
      meta.mapping_base_offset = storage.mapping_base_offset();
    } else {
      if (!handle_view.empty()) {
        meta.handle_bytes = std::string(handle_view);
        meta.mapping_base_offset = cursor;
      } else {
        return absl::FailedPreconditionError(
            "binding target_layout storage is missing a readable source for contribution lowering");
      }
    }
    result.storages.push_back(std::move(meta));

    LeaseSegMeta segment;
    segment.storage_id = storage.storage_id();
    segment.storage_offset = 0;
    segment.artifact_offset = cursor;
    segment.length = storage.storage_length();
    result.segments.push_back(std::move(segment));
    cursor += storage.storage_length();
  }

  if (cursor == 0) {
    return absl::InvalidArgumentError("binding target_layout storages must be non-empty");
  }
  result.total_size = cursor;
  return result;
}

absl::StatusOr<PieceContributionViewSpec> build_subset_identity_piece_view_spec(
    std::string_view canonical_index_json,
    const google::protobuf::RepeatedPtrField<std::string>& tensor_names) {
  auto index_table_or = materialization_layout::parse_canonical_index(canonical_index_json);
  if (!index_table_or.ok()) {
    return index_table_or.status();
  }
  PieceContributionViewSpec result;
  result.tensor_names.reserve(tensor_names.size());
  result.canonical_size_bytes = index_table_or->logical_total_size;
  for (const auto& tensor_name : tensor_names) {
    const auto it = index_table_or->entries.find(tensor_name);
    if (it == index_table_or->entries.end()) {
      return absl::InvalidArgumentError(absl::StrCat("binding contribution references unknown tensor=", tensor_name));
    }
    if (it->second.shape.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("binding subset contribution requires shape metadata for tensor=", tensor_name));
    }
    auto* ops = (*result.spec.mutable_tensors())[tensor_name].add_ops();
    ops->mutable_narrow()->set_dim(0);
    ops->mutable_narrow()->set_start(0);
    ops->mutable_narrow()->set_length(static_cast<uint64_t>(it->second.shape.front()));
    result.tensor_names.push_back(tensor_name);
  }
  auto view_id_or = compute_view_id_from_spec(result.spec, canonical_index_json);
  if (!view_id_or.ok()) {
    return view_id_or.status();
  }
  result.view_id = *view_id_or;
  return result;
}

absl::StatusOr<PieceContributionViewSpec> resolve_piece_contribution_view_spec(
    store::StoreEngine& engine,
    std::string_view resolved_artifact_id,
    std::string_view canonical_index_json,
    const tensorcast::common::v1::ArtifactSelection& selection,
    std::string_view requested_view_id) {
  PieceContributionViewSpec result;
  auto index_table_or = materialization_layout::parse_canonical_index(canonical_index_json);
  if (!index_table_or.ok()) {
    return index_table_or.status();
  }
  result.canonical_size_bytes = index_table_or->logical_total_size;
  for (const auto& name : selection.tensor_names()) {
    if (!name.empty()) {
      result.tensor_names.push_back(name);
    }
  }

  if (selection.has_view_spec()) {
    result.spec = selection.view_spec();
  } else if (!selection.view_id().empty()) {
    auto metadata_or = engine.get_view_metadata(std::string(resolved_artifact_id), selection.view_id());
    if (!metadata_or.ok()) {
      return metadata_or.status();
    }
    auto parsed_or = store::view::parse_view_selection_json(metadata_or->view_spec_json);
    if (!parsed_or.ok()) {
      return parsed_or.status();
    }
    result.spec = build_view_spec_proto(parsed_or->spec);
    if (result.tensor_names.empty()) {
      result.tensor_names.assign(parsed_or->tensor_names.begin(), parsed_or->tensor_names.end());
    }
  } else if (selection.tensor_names_size() > 0) {
    auto subset_or = build_subset_identity_piece_view_spec(canonical_index_json, selection.tensor_names());
    if (!subset_or.ok()) {
      return subset_or.status();
    }
    return *subset_or;
  } else {
    return absl::InvalidArgumentError(
        "piece contribution requires binding selection metadata with view_spec, view_id, or tensor_names");
  }

  auto view_id_or = compute_view_id_from_spec(result.spec, canonical_index_json);
  if (!view_id_or.ok()) {
    return view_id_or.status();
  }
  result.view_id = *view_id_or;
  const std::string expected_view_id = requested_view_id.empty() ? selection.view_id() : std::string(requested_view_id);
  if (!expected_view_id.empty() && expected_view_id != result.view_id) {
    return absl::InvalidArgumentError("binding contribution view_id does not match binding selection");
  }
  return result;
}

absl::StatusOr<BindingContributionRegistrationPlan> build_binding_contribution_registration_plan(
    const OwnedBindingService::Dep& dep,
    const grpc::ServerContext& server_context,
    const BindingRegistry::Record& record,
    const v2::SubmitBindingContributionRequest& req) {
  auto storage_layout_or = build_contribution_storage_layout(record);
  if (!storage_layout_or.ok()) {
    return storage_layout_or.status();
  }

  BindingContributionRegistrationPlan plan;
  plan.storage_layout = std::move(*storage_layout_or);
  plan.begin_request.set_device_id(record.device_id);
  plan.begin_request.set_total_size(plan.storage_layout.total_size);
  plan.begin_request.set_owner_pid(record.owner_pid);
  plan.begin_request.set_client_artifact_id(req.workspace_assembly_id());
  plan.begin_request.mutable_tensor_index_data()->set_schema_version("v3");
  plan.begin_request.mutable_tensor_index_data()->set_encoding("json");
  plan.begin_request.mutable_coalesced();

  if (req.contribution_kind() == v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL) {
    plan.begin_request.mutable_tensor_index_data()->set_data(record.target_index_json);
    plan.slot_id = std::string(coordination::kCanonicalFullContributionSlotKey);
    return plan;
  }

  const auto& selection =
      !record.current_selection.artifact_id().empty() ? record.current_selection : record.source_selection;
  if (selection.artifact_id().empty()) {
    return absl::FailedPreconditionError("piece contribution requires artifact-backed binding selection metadata");
  }

  const bool loopback_peer = is_loopback_grpc_peer(server_context.peer());
  auto resolution_or = resolve_artifact_and_disk_source(
      dep.global_store_client,
      &dep.disk_imports,
      dep.storage_path,
      selection.artifact_id(),
      /*allow_disk=*/true,
      /*allow_local_import_fallback=*/true,
      loopback_peer);
  if (!resolution_or.ok()) {
    return resolution_or.status();
  }
  auto resolution = std::move(*resolution_or);
  if (resolution.local_import.has_value() && !resolution.local_import->tensor_aware_metadata) {
    return absl::InvalidArgumentError("piece contribution view planning requires tensor-aware mounted-source metadata");
  }
  auto canonical_index_or = load_canonical_index_with_disk_fallback(
      dep.engine,
      resolution.resolved_artifact_id,
      resolution.normalized_disk_path,
      record.device_id,
      resolution.gs_connected);
  if (!canonical_index_or.ok()) {
    return canonical_index_or.status();
  }

  auto piece_or = resolve_piece_contribution_view_spec(
      dep.engine, resolution.resolved_artifact_id, *canonical_index_or, selection, req.view_id());
  if (!piece_or.ok()) {
    return piece_or.status();
  }

  plan.begin_request.mutable_tensor_index_data()->set_data(*canonical_index_or);
  auto* view = plan.begin_request.mutable_view();
  view->mutable_spec()->CopyFrom(piece_or->spec);
  view->set_placement(v2::TRANSFORM_PLACEMENT_SERVER);
  view->set_canonical_size_bytes(piece_or->canonical_size_bytes);
  view->set_registration_kind(v2::VIEW_REGISTRATION_KIND_PIECE);
  view->set_view_id(piece_or->view_id);
  for (const auto& tensor_name : piece_or->tensor_names) {
    view->add_tensor_names(tensor_name);
  }
  plan.slot_id = piece_or->view_id;
  plan.structural_view_id = piece_or->view_id;
  return plan;
}

absl::StatusOr<OpenedContributionLayout> prepare_contribution_segments(
    const ContributionStorageLayout& layout,
    IpcRegionRegistry& regions,
    int owner_pid) {
  OpenedContributionLayout opened;
  absl::flat_hash_map<std::string, const RegisterStorageMeta*> storage_by_id;
  storage_by_id.reserve(layout.storages.size());
  for (const auto& storage : layout.storages) {
    storage_by_id.emplace(storage.storage_id, &storage);
  }

  opened.region_pin = std::make_unique<RegionPinGuard>(regions);
  opened.mapping_cache.reserve(layout.storages.size());

  opened.segments.reserve(layout.segments.size());
  for (const auto& segment : layout.segments) {
    const auto it = storage_by_id.find(segment.storage_id);
    if (it == storage_by_id.end()) {
      return absl::InvalidArgumentError("binding contribution segment references unknown storage_id");
    }
    auto mapping_or =
        get_or_open_mapping_for_storage(*it->second, opened.mapping_cache, *opened.region_pin, regions, owner_pid);
    if (!mapping_or.ok()) {
      return mapping_or.status();
    }
    PreparedContributionSegment prepared_segment;
    prepared_segment.device_id = it->second->device_id;
    prepared_segment.mapping = *mapping_or;
    prepared_segment.base_offset = it->second->mapping_base_offset + segment.storage_offset;
    prepared_segment.artifact_offset = segment.artifact_offset;
    prepared_segment.length = segment.length;
    opened.segments.push_back(prepared_segment);
  }
  std::sort(
      opened.segments.begin(),
      opened.segments.end(),
      [](const PreparedContributionSegment& lhs, const PreparedContributionSegment& rhs) {
        return lhs.artifact_offset < rhs.artifact_offset;
      });
  return opened;
}

absl::Status read_contribution_bytes(
    absl::Span<const PreparedContributionSegment> segments,
    uint64_t artifact_offset,
    absl::Span<uint8_t> dst) {
  uint64_t cursor = artifact_offset;
  size_t remaining = dst.size();
  uint8_t* out = dst.data();
  while (remaining > 0) {
    const PreparedContributionSegment* segment = nullptr;
    for (const auto& candidate : segments) {
      if (cursor >= candidate.artifact_offset && cursor < candidate.artifact_offset + candidate.length) {
        segment = &candidate;
        break;
      }
    }
    if (segment == nullptr || segment->mapping == nullptr) {
      return absl::FailedPreconditionError("binding contribution storage layout is missing byte coverage");
    }
    const uint64_t local_offset = cursor - segment->artifact_offset;
    const size_t available = static_cast<size_t>(segment->length - local_offset);
    const size_t take = std::min(remaining, available);
    cuda::CudaDeviceGuard guard(segment->device_id);
    if (!guard.status().ok()) {
      return guard.status();
    }
    auto memcpy_status = cuda::memcpy(
        out,
        static_cast<const uint8_t*>(segment->mapping->get()) +
            static_cast<std::ptrdiff_t>(segment->base_offset + local_offset),
        take,
        cudaMemcpyDeviceToHost);
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
  return absl::OkStatus();
}

absl::Status copy_contribution_segments_to_device_buffer(
    absl::Span<const PreparedContributionSegment> segments,
    void* dst_base_ptr,
    int device_id) {
  if (dst_base_ptr == nullptr) {
    return absl::InvalidArgumentError("destination registration buffer is null");
  }
  cuda::CudaDeviceGuard guard(device_id);
  if (!guard.status().ok()) {
    return guard.status();
  }
  for (const auto& segment : segments) {
    if (segment.mapping == nullptr || segment.length == 0) {
      continue;
    }
    auto* dst = static_cast<uint8_t*>(dst_base_ptr) + static_cast<std::ptrdiff_t>(segment.artifact_offset);
    auto* src = static_cast<const uint8_t*>(segment.mapping->get()) + static_cast<std::ptrdiff_t>(segment.base_offset);
    auto memcpy_status = cuda::memcpy(dst, src, static_cast<size_t>(segment.length), cudaMemcpyDeviceToDevice);
    if (!memcpy_status.ok()) {
      return memcpy_status;
    }
  }
  return cuda::device_synchronize();
}

grpc::Status abort_registration_best_effort(
    RegistrationController& controller,
    grpc::ServerContext& server_context,
    std::string_view registration_id) {
  if (registration_id.empty()) {
    return Status::OK;
  }
  v2::AbortRegisteredArtifactRequest abort_req;
  abort_req.set_registration_id(std::string(registration_id));
  v2::AbortRegisteredArtifactResponse abort_resp;
  RpcContext abort_rctx{
      "AbortRegisterArtifact.InternalBindingContribution", server_context, /*allow_high_card_attrs=*/true};
  return controller.abort(abort_rctx, abort_req, abort_resp);
}

grpc::Status commit_binding_contribution_registration(
    const OwnedBindingService::Dep& dep,
    grpc::ServerContext& server_context,
    const BindingRegistry::Record& record,
    const BindingContributionRegistrationPlan& plan,
    std::optional<std::string>& committed_view_id) {
  if (dep.registration_manager == nullptr || dep.lip_manager == nullptr || dep.refs == nullptr ||
      dep.lifecycle == nullptr || dep.regions == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "binding contribution registration dependencies are unavailable"};
  }

  RegistrationController controller(
      RegistrationController::Dep{
          .engine = dep.engine,
          .reg = *dep.registration_manager,
          .lip = *dep.lip_manager,
          .refs = *dep.refs,
          .identity = &dep.identity,
          .global_store_client = dep.global_store_client,
          .lifecycle = dep.lifecycle,
          .handle_leases = dep.handle_leases,
          .regions = *dep.regions,
          .max_concurrency = dep.max_concurrency,
          .await_state_sync_barrier = dep.await_state_sync_barrier,
      });

  v2::BeginRegisterArtifactResponse begin_resp;
  RpcContext begin_rctx{
      "BeginRegisterArtifact.InternalBindingContribution", server_context, /*allow_high_card_attrs=*/true};
  auto begin_status = controller.begin(begin_rctx, plan.begin_request, begin_resp);
  if (!begin_status.ok()) {
    return begin_status;
  }

  const std::string registration_id = begin_resp.registration_id();
  auto prepared_layout_or = prepare_contribution_segments(plan.storage_layout, *dep.regions, record.owner_pid);
  if (!prepared_layout_or.ok()) {
    (void)abort_registration_best_effort(controller, server_context, registration_id);
    return to_grpc_status(prepared_layout_or.status());
  }

  if (plan.structural_view_id.has_value()) {
    std::vector<uint8_t> chunk_buffer;
    chunk_buffer.resize(
        static_cast<size_t>(std::min<uint64_t>(kContributionFeedChunkBytes, plan.storage_layout.total_size)));
    for (uint64_t offset = 0; offset < plan.storage_layout.total_size;
         offset += static_cast<uint64_t>(chunk_buffer.size())) {
      const size_t chunk_bytes =
          static_cast<size_t>(std::min<uint64_t>(chunk_buffer.size(), plan.storage_layout.total_size - offset));
      auto read_status = read_contribution_bytes(
          prepared_layout_or->segments, offset, absl::MakeSpan(chunk_buffer).subspan(0, chunk_bytes));
      if (!read_status.ok()) {
        (void)abort_registration_best_effort(controller, server_context, registration_id);
        return to_grpc_status(read_status);
      }
      v2::FeedRegisterArtifactStreamRequest feed_req;
      feed_req.set_registration_id(registration_id);
      feed_req.mutable_view_chunk()->set_view_offset(offset);
      feed_req.mutable_view_chunk()->set_data(chunk_buffer.data(), static_cast<int>(chunk_bytes));
      const auto feed_status = controller.feed_vector({std::move(feed_req)});
      if (!feed_status.ok()) {
        (void)abort_registration_best_effort(controller, server_context, registration_id);
        return feed_status;
      }
    }
  } else {
    auto gpu_ptr_or = dep.engine.get_registration_gpu_ptr(registration_id);
    if (!gpu_ptr_or.ok()) {
      (void)abort_registration_best_effort(controller, server_context, registration_id);
      return to_grpc_status(gpu_ptr_or.status());
    }
    auto copy_status = copy_contribution_segments_to_device_buffer(
        prepared_layout_or->segments, reinterpret_cast<void*>(static_cast<uintptr_t>(*gpu_ptr_or)), record.device_id);
    if (!copy_status.ok()) {
      (void)abort_registration_best_effort(controller, server_context, registration_id);
      return to_grpc_status(copy_status);
    }
  }

  v2::CommitRegisteredArtifactRequest commit_req;
  commit_req.set_registration_id(registration_id);
  v2::CommitRegisteredArtifactResponse commit_resp;
  RpcContext commit_rctx{
      "CommitRegisteredArtifact.InternalBindingContribution", server_context, /*allow_high_card_attrs=*/true};
  const auto commit_status = controller.commit(commit_rctx, commit_req, commit_resp);
  if (!commit_status.ok()) {
    (void)abort_registration_best_effort(controller, server_context, registration_id);
    return commit_status;
  }
  if (plan.structural_view_id.has_value()) {
    if (commit_resp.view_id().empty()) {
      return {StatusCode::DATA_LOSS, "piece contribution registration returned empty view_id"};
    }
    committed_view_id = commit_resp.view_id();
  } else {
    committed_view_id.reset();
  }
  return Status::OK;
}

tensorcast::common::v1::ByteSpaceRef byte_space_from_selection(
    const tensorcast::common::v1::ArtifactSelection& selection) {
  tensorcast::common::v1::ByteSpaceRef byte_space;
  if (!selection.view_id().empty()) {
    byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
    byte_space.set_id(selection.view_id());
  } else {
    byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
    byte_space.set_id("");
  }
  return byte_space;
}

tensorcast::common::v1::ByteSpaceRef canonical_byte_space_ref() {
  tensorcast::common::v1::ByteSpaceRef byte_space;
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  byte_space.set_id("");
  return byte_space;
}

bool selection_declares_tensor_subset(const tensorcast::common::v1::ArtifactSelection& selection) {
  return selection.tensor_names_size() > 0 || !selection.view_subset_hash().empty();
}

tensorcast::common::v1::ArtifactSelection build_canonical_current_value_publication_selection(
    std::string_view artifact_id,
    std::string_view canonical_index_json) {
  tensorcast::common::v1::ArtifactSelection selection;
  selection.set_artifact_id(std::string(artifact_id));
  selection.set_logical_layout_hash(
      common::compute_logical_layout_hash_bytes(
          absl::Span<const uint8_t>(
              reinterpret_cast<const uint8_t*>(canonical_index_json.data()), canonical_index_json.size()),
          /*needs_view_index=*/false));
  selection.set_selection_hash(common::compute_selection_hash_bytes("", std::nullopt));
  return selection;
}

tensorcast::common::v1::ArtifactSelection build_mapped_bound_selection(
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

bool artifact_selection_equal(
    const tensorcast::common::v1::ArtifactSelection& lhs,
    const tensorcast::common::v1::ArtifactSelection& rhs) {
  if (lhs.artifact_id() != rhs.artifact_id() || lhs.view_id() != rhs.view_id() ||
      lhs.logical_layout_hash() != rhs.logical_layout_hash() || lhs.selection_hash() != rhs.selection_hash() ||
      lhs.view_subset_hash() != rhs.view_subset_hash() || lhs.tensor_names_size() != rhs.tensor_names_size()) {
    return false;
  }
  for (int i = 0; i < lhs.tensor_names_size(); ++i) {
    if (lhs.tensor_names(i) != rhs.tensor_names(i)) {
      return false;
    }
  }
  return true;
}

absl::Status ensure_binding_current_value_not_published(const BindingRegistry::Record& record) {
  if (!record.active_published_current.has_value()) {
    return absl::OkStatus();
  }
  const auto& published = *record.active_published_current;
  if (published.binding_id == record.binding_id && published.binding_value_id == record.current_binding_value_id &&
      published.seal_generation == record.seal_generation) {
    return absl::FailedPreconditionError("binding current value is already published; call retire() first");
  }
  return absl::OkStatus();
}

absl::Status validate_publication_snapshot_unchanged(
    const BindingRegistry::Record& record,
    std::string_view expected_binding_value_id,
    uint64_t expected_seal_generation,
    std::string_view expected_artifact_id,
    const tensorcast::common::v1::ArtifactSelection& expected_selection) {
  if (record.closed) {
    return absl::FailedPreconditionError("binding is closed");
  }
  if (record.retired) {
    return absl::FailedPreconditionError("binding is retired");
  }
  if (record.state != v2::BINDING_STATE_READY_ARTIFACT) {
    return absl::FailedPreconditionError("binding current value is not artifact-backed");
  }
  if (record.current_binding_value_id != expected_binding_value_id ||
      record.seal_generation != expected_seal_generation || record.current_artifact_id != expected_artifact_id ||
      !artifact_selection_equal(record.current_selection, expected_selection)) {
    return absl::FailedPreconditionError("binding current value changed while minting publication token");
  }
  return ensure_binding_current_value_not_published(record);
}

std::string daemon_id_for_binding_publication(
    const WorkerIdentityStore& identity,
    std::string_view configured_daemon_id,
    const BindingRegistry::Record& record) {
  if (!record.daemon_id.empty()) {
    return record.daemon_id;
  }
  std::string daemon_id = identity.daemon_id();
  if (!daemon_id.empty()) {
    return daemon_id;
  }
  return std::string(configured_daemon_id);
}

absl::StatusOr<std::string> maybe_mint_binding_current_value_publication_token(
    common::CapabilityTokenManager* capability_tokens,
    WorkerIdentityStore& identity,
    TargetMaterializationService* target_materialization_service,
    const std::shared_ptr<BindingRegistry::Record>& record,
    std::string_view configured_daemon_id,
    std::string_view daemon_session_id,
    std::string_view operation_id,
    std::vector<LeaseSegMeta> publish_segments,
    std::vector<RegisterStorageMeta> publish_storages) {
  if (capability_tokens == nullptr || !capability_tokens->configured()) {
    return std::string();
  }
  if (target_materialization_service == nullptr) {
    return absl::FailedPreconditionError("target materialization service is unavailable");
  }
  if (record == nullptr) {
    return absl::InvalidArgumentError("binding record is required");
  }

  std::string binding_id;
  std::string binding_layout_id;
  std::string binding_value_id;
  std::string artifact_id;
  std::string canonical_index_json;
  std::string layout_hash;
  std::string device_uuid;
  std::string daemon_id;
  std::string effective_daemon_session_id;
  int device_id = -1;
  int owner_pid = 0;
  uint64_t seal_generation = 0;
  tensorcast::common::v1::ArtifactSelection current_selection;
  {
    absl::MutexLock lock(&record->mu);
    auto snapshot_status = validate_publication_snapshot_unchanged(
        *record,
        record->current_binding_value_id,
        record->seal_generation,
        record->current_artifact_id,
        record->current_selection);
    if (!snapshot_status.ok()) {
      return snapshot_status;
    }
    if (record->current_binding_value_id.empty() || record->current_artifact_id.empty()) {
      return absl::FailedPreconditionError("binding current value is not publishable");
    }
    binding_id = record->binding_id;
    binding_layout_id = record->binding_layout_id;
    binding_value_id = record->current_binding_value_id;
    artifact_id = record->current_artifact_id;
    canonical_index_json = record->current_artifact_canonical_index_json;
    layout_hash = record->target_layout_hash.empty() ? compute_target_layout_hash(record->target_layout)
                                                     : record->target_layout_hash;
    device_uuid = record->device_uuid;
    daemon_id = daemon_id_for_binding_publication(identity, configured_daemon_id, *record);
    effective_daemon_session_id =
        !record->daemon_session_id.empty() ? record->daemon_session_id : std::string(daemon_session_id);
    device_id = record->device_id;
    owner_pid = record->owner_pid;
    seal_generation = record->seal_generation;
    current_selection = record->current_selection;
  }
  if (daemon_id.empty()) {
    return absl::FailedPreconditionError("daemon_id is required for binding current value publication");
  }
  if (effective_daemon_session_id.empty()) {
    return absl::FailedPreconditionError("daemon_session_id is required for binding current value publication");
  }
  if (canonical_index_json.empty()) {
    return absl::FailedPreconditionError("binding current value canonical index is unavailable");
  }

  auto stable_index_or = store::loader::rebuild_stable_canonical_index(canonical_index_json, device_id);
  if (!stable_index_or.ok()) {
    return stable_index_or.status();
  }
  std::string stable_index_json = std::move(*stable_index_or);

  tensorcast::common::v1::ArtifactSelection publication_selection = current_selection;
  auto byte_space = byte_space_from_selection(current_selection);
  if (!selection_declares_tensor_subset(current_selection)) {
    publication_selection = build_canonical_current_value_publication_selection(artifact_id, stable_index_json);
    byte_space = canonical_byte_space_ref();
  }

  const std::string publication_id = mint_publication_id();
  const absl::Time expires_at = absl::Now() + TargetPublishService::binding_current_value_publication_token_ttl();
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(stable_index_json.data()), stable_index_json.size()));
  std::string index_key_hex =
      absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));

  tensorcast::common::v1::BindingCurrentValuePublicationScope scope;
  scope.set_publication_id(publication_id);
  scope.mutable_selection()->CopyFrom(publication_selection);
  scope.mutable_byte_space()->CopyFrom(byte_space);
  scope.set_device_uuid(device_uuid);
  scope.set_owner_pid(owner_pid);
  scope.set_target_layout_hash(layout_hash);
  if (!operation_id.empty()) {
    scope.set_operation_id(std::string(operation_id));
  }
  scope.set_binding_id(binding_id);
  scope.set_binding_layout_id(binding_layout_id);
  scope.set_binding_value_id(binding_value_id);
  scope.set_seal_generation(seal_generation);
  scope.set_daemon_id(daemon_id);
  scope.set_daemon_session_id(effective_daemon_session_id);

  auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  if (!scope_or.ok()) {
    return scope_or.status();
  }
  const uint64_t expires_at_ms = static_cast<uint64_t>(absl::ToUnixMillis(expires_at));
  auto token_or = capability_tokens->mint(
      daemon_id,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_BINDING_CURRENT_VALUE_PUBLICATION,
      *scope_or,
      expires_at_ms);
  if (!token_or.ok()) {
    return token_or.status();
  }

  TargetPublicationRegistry::Record publication_record;
  publication_record.publication_id = PublicationInstanceId{.value = publication_id};
  publication_record.publication_subject_key =
      build_publication_subject_key(publication_selection, byte_space, layout_hash, device_uuid);
  publication_record.target_layout_hash = layout_hash;
  publication_record.selection = publication_selection;
  publication_record.byte_space = byte_space;
  publication_record.canonical_index_json = std::move(stable_index_json);
  publication_record.index_key_hex = std::move(index_key_hex);
  publication_record.device_uuid = device_uuid;
  publication_record.owner_pid = owner_pid;
  publication_record.daemon_id = daemon_id;
  publication_record.daemon_session_id = effective_daemon_session_id;
  publication_record.binding_id = binding_id;
  publication_record.binding_layout_id = binding_layout_id;
  publication_record.binding_value_id = binding_value_id;
  publication_record.seal_generation = seal_generation;
  publication_record.request_operation_id = std::string(operation_id);
  publication_record.expires_at = expires_at;
  publication_record.segments = std::move(publish_segments);
  publication_record.storages = std::move(publish_storages);
  auto remembered_or = target_materialization_service->remember_target_publication(std::move(publication_record));
  if (!remembered_or.ok()) {
    return remembered_or.status();
  }
  absl::Status snapshot_status = absl::OkStatus();
  {
    absl::MutexLock lock(&record->mu);
    snapshot_status = validate_publication_snapshot_unchanged(
        *record, binding_value_id, seal_generation, artifact_id, current_selection);
    if (snapshot_status.ok() &&
        (record->binding_id != binding_id || record->binding_layout_id != binding_layout_id ||
         record->device_uuid != device_uuid || record->owner_pid != owner_pid ||
         daemon_id_for_binding_publication(identity, configured_daemon_id, *record) != daemon_id ||
         (!record->daemon_session_id.empty() && record->daemon_session_id != effective_daemon_session_id))) {
      snapshot_status = absl::FailedPreconditionError("binding identity changed while minting publication token");
    }
    if (snapshot_status.ok()) {
      record->binding_current_value_publication_token = *token_or;
    }
  }
  if (!snapshot_status.ok()) {
    auto cleanup_status = target_materialization_service->terminalize_target_publication(
        publication_id, "binding_current_value_changed_during_token_mint", /*release_published_lifecycle_lease=*/false);
    if (!cleanup_status.ok() && !absl::IsNotFound(cleanup_status)) {
      LOG(WARNING) << "failed to clean up stale binding current value publication " << publication_id << ": "
                   << cleanup_status;
    }
    return snapshot_status;
  }
  return *token_or;
}

std::string mint_binding_value_id() {
  return mint_random_id(16);
}

constexpr std::string_view kCollectiveFailureClassMetadataKey = "tc.collective_failure_class";

std::string collective_failure_class_name(v2::CollectiveFailureClass failure_class) {
  switch (failure_class) {
    case v2::CollectiveFailureClass::COLLECTIVE_FAILURE_CLASS_NOT_ELIGIBLE:
      return "not_eligible";
    case v2::CollectiveFailureClass::COLLECTIVE_FAILURE_CLASS_EXECUTION_FAILED:
      return "execution_failed";
    case v2::CollectiveFailureClass::COLLECTIVE_FAILURE_CLASS_UNSPECIFIED:
    default:
      return "unspecified";
  }
}

void attach_collective_failure_metadata(RpcContext* rctx, v2::CollectiveFailureClass failure_class) {
  if (rctx == nullptr) {
    return;
  }
  const std::string failure_class_value = collective_failure_class_name(failure_class);
  if (failure_class_value == "unspecified") {
    return;
  }
  rctx->server_context().AddTrailingMetadata(std::string(kCollectiveFailureClassMetadataKey), failure_class_value);
}

grpc::Status make_collective_failure_status(
    RpcContext* rctx,
    grpc::StatusCode code,
    std::string_view message,
    v2::CollectiveFailureClass failure_class) {
  attach_collective_failure_metadata(rctx, failure_class);
  return {code, std::string(message)};
}

bool is_collective_execution_failure(const absl::Status& status) {
  return absl::StrContains(status.message(), "collective execution failed:");
}

std::string format_reject_reason_buckets(
    const absl::flat_hash_map<std::string, uint64_t>& planner_reject_reason_buckets) {
  if (planner_reject_reason_buckets.empty()) {
    return "{}";
  }
  std::vector<std::pair<std::string, uint64_t>> ordered(
      planner_reject_reason_buckets.begin(), planner_reject_reason_buckets.end());
  std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  std::vector<std::string> rendered;
  rendered.reserve(ordered.size());
  for (const auto& [reason, bytes] : ordered) {
    rendered.push_back(absl::StrCat(reason, "=", bytes));
  }
  return absl::StrCat("{", absl::StrJoin(rendered, ","), "}");
}

std::string append_pure_collective_blockers(
    std::string_view message,
    const store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary& summary) {
  absl::flat_hash_map<std::string, uint64_t> pure_collective_blockers;
  if (summary.planned_non_admitted_typed_bytes > 0) {
    pure_collective_blockers["typed_work_not_collective_admitted"] = summary.planned_non_admitted_typed_bytes;
  }
  if (summary.planned_local_typed_bytes > 0) {
    pure_collective_blockers["local_typed_work_present"] = summary.planned_local_typed_bytes;
  }
  if (summary.planned_generic_residual_bytes > 0) {
    pure_collective_blockers["true_generic_residual_present"] = summary.planned_generic_residual_bytes;
  }
  return absl::StrCat(
      message,
      " (planned_local_typed_bytes=",
      summary.planned_local_typed_bytes,
      ", planned_non_admitted_typed_bytes=",
      summary.planned_non_admitted_typed_bytes,
      ", planned_generic_residual_bytes=",
      summary.planned_generic_residual_bytes,
      ", pure_collective_blockers=",
      format_reject_reason_buckets(pure_collective_blockers),
      ", planner_reject_reason_buckets=",
      format_reject_reason_buckets(summary.planner_reject_reason_buckets),
      ")");
}

store::runtime::ingestion::strategy::SourceBoundPolicy normalize_source_bound_policy(
    v2::CollectivePolicy collective_policy) {
  using SourceBoundPolicy = store::runtime::ingestion::strategy::SourceBoundPolicy;
  switch (collective_policy) {
    case v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE:
      return SourceBoundPolicy::kRequirePureCollective;
    case v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE:
      return SourceBoundPolicy::kDisableCollective;
    case v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST:
    default:
      return SourceBoundPolicy::kCollectiveFirst;
  }
}

store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary summarize_source_bound_plan(
    const store::runtime::ingestion::strategy::ResolvedMaterializationPlan& resolved_plan,
    const std::optional<store::runtime::ingestion::strategy::SourceBoundLoweringArtifacts>& lowering_artifacts,
    const store::StoreEngineOptions::MaterializationStrategyConfig& strategy_config,
    const store::loading::ExecutionTopologyContext& execution_topology,
    v2::CollectivePolicy collective_policy,
    store::runtime::ingestion::strategy::SourceBoundSourceFacts source_facts) {
  auto strategy_plan_or = store::runtime::ingestion::strategy::build_source_bound_execution_strategy_plan(
      resolved_plan,
      lowering_artifacts,
      normalize_source_bound_policy(collective_policy),
      strategy_config,
      execution_topology,
      source_facts);
  if (!strategy_plan_or.ok()) {
    LOG(ERROR) << "source-bound strategy planning failed: " << strategy_plan_or.status();
    store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary summary;
    summary.planner_version = "source_bound_collective_first.v4";
    summary.execution_plan_kind = collective_policy == v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE
        ? "pure_collective"
        : "generic_only";
    return summary;
  }
  return strategy_plan_or->summary;
}

absl::StatusOr<std::vector<RegisterTensorAliasMeta>> build_binding_tensor_aliases(
    const BindingRegistry::Record& record) {
  auto offsets_or = resolve_target_offsets(record.target_layout);
  if (!offsets_or.ok()) {
    return offsets_or.status();
  }
  auto index_or = parse_canonical_index(record.target_index_json);
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

void populate_artifact_descriptor(const CommitLeaseResult& out, tensorcast::common::v1::ArtifactDescriptor* desc) {
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

v2::ExecutionDiagnostics build_binding_closeout_diagnostics(
    v2::IdentityMintStrategy identity_mint_strategy,
    const CommitLeaseResult* hash_commit_result = nullptr,
    const store::loading::MaterializeIntoTargetResult* source_execution = nullptr) {
  HashExecutionDetails hash_details;
  switch (identity_mint_strategy) {
    case v2::IdentityMintStrategy::IDENTITY_MINT_STRATEGY_SEAL_MINT:
      hash_details.hash_rounds = 1U;
      hash_details.hash_location = v2::HashLocation::HASH_LOCATION_SEAL;
      break;
    case v2::IdentityMintStrategy::IDENTITY_MINT_STRATEGY_SEAL_REUSE:
      hash_details.hash_rounds = 0U;
      hash_details.hash_location = v2::HashLocation::HASH_LOCATION_SEAL;
      break;
    case v2::IdentityMintStrategy::IDENTITY_MINT_STRATEGY_CLOSEOUT_MINT:
      hash_details.hash_rounds = 1U;
      hash_details.hash_location = v2::HashLocation::HASH_LOCATION_BINDING_CLOSEOUT;
      break;
    case v2::IdentityMintStrategy::IDENTITY_MINT_STRATEGY_NOT_APPLICABLE:
    case v2::IdentityMintStrategy::IDENTITY_MINT_STRATEGY_UNSPECIFIED:
    default:
      hash_details.hash_rounds = 0U;
      hash_details.hash_location = v2::HashLocation::HASH_LOCATION_NONE;
      break;
  }
  hash_details.identity_mint_strategy = identity_mint_strategy;
  if (hash_commit_result != nullptr) {
    switch (hash_commit_result->hash_info.backend) {
      case CommitLeaseHashBackend::kGpu:
        hash_details.hash_backend = v2::HashBackend::HASH_BACKEND_GPU;
        break;
      case CommitLeaseHashBackend::kD2HCpu:
        hash_details.hash_backend = v2::HashBackend::HASH_BACKEND_D2H_CPU;
        break;
      case CommitLeaseHashBackend::kCpu:
        hash_details.hash_backend = v2::HashBackend::HASH_BACKEND_CPU;
        break;
      case CommitLeaseHashBackend::kNone:
      default:
        hash_details.hash_backend = v2::HashBackend::HASH_BACKEND_NONE;
        break;
    }
    hash_details.hash_bytes = hash_commit_result->hash_info.bytes;
    hash_details.hash_wall_time_ms = hash_commit_result->hash_info.wall_time_ms;
    hash_details.hash_identity_forming = hash_commit_result->hash_info.identity_forming;
  }
  return build_execution_diagnostics(
      source_execution,
      v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE,
      store::loading::ExecutionTopologyContext{},
      hash_details);
}

v2::SourceBoundPlanDiagnostics build_source_bound_plan_diagnostics(
    const std::optional<store::runtime::ingestion::strategy::PreparedSourceBoundExecutionPlan>&
        prepared_execution_plan) {
  v2::SourceBoundPlanDiagnostics diagnostics;
  if (!prepared_execution_plan.has_value() || !prepared_execution_plan->strategy_plan.has_value()) {
    return diagnostics;
  }
  const auto* summary = &prepared_execution_plan->strategy_plan->summary;
  diagnostics.set_execution_plan_kind(summary->execution_plan_kind);
  diagnostics.set_planned_collective_candidate_bytes(summary->planned_collective_candidate_bytes);
  diagnostics.set_planned_collective_admitted_bytes(summary->planned_collective_admitted_bytes);
  diagnostics.set_planned_local_typed_bytes(summary->planned_local_typed_bytes);
  diagnostics.set_planned_non_admitted_typed_bytes(summary->planned_non_admitted_typed_bytes);
  diagnostics.set_planned_generic_residual_bytes(summary->planned_generic_residual_bytes);
  diagnostics.set_collective_lowered_bytes(summary->collective_lowered_bytes);
  diagnostics.set_planner_version(summary->planner_version);
  diagnostics.set_plan_hash(summary->plan_hash);
  diagnostics.set_estimated_collective_peak_temporary_bytes(summary->estimated_collective_peak_temporary_bytes);
  diagnostics.set_estimated_collective_batch_bytes(summary->estimated_collective_batch_bytes);
  diagnostics.set_estimated_collective_dedup_saving_bytes(summary->estimated_collective_dedup_saving_bytes);
  for (const auto& [reason, bytes] : summary->planner_reject_reason_buckets) {
    (*diagnostics.mutable_planner_reject_reason_buckets())[reason] = bytes;
  }
  return diagnostics;
}

void fill_binding_value(const BindingRegistry::Record& record, v2::BindingValue& value) {
  value.set_binding_id(record.binding_id);
  value.set_binding_layout_id(record.binding_layout_id);
  value.set_binding_value_id(record.current_binding_value_id);
  value.set_seal_generation(record.seal_generation);
  if (!record.current_artifact_id.empty()) {
    value.set_source_artifact_id(record.current_artifact_id);
  }
  if (!record.current_selection.artifact_id().empty()) {
    value.mutable_selection()->CopyFrom(record.current_selection);
  }
  value.set_is_artifact_backed(!record.current_artifact_id.empty());
  value.set_verification_state(record.verification_state);
  value.set_verification_job_id(record.verification_job_id);
  value.set_source_artifact_ref(record.source_artifact_ref);
  value.set_local_serving_ref(record.local_serving_ref);
  if (!record.serving_artifact_id.empty()) {
    value.set_serving_artifact_id(record.serving_artifact_id);
  }
  if (!record.verification_failure_reason.empty()) {
    value.set_verification_failure_reason(record.verification_failure_reason);
  }
}

void fill_staged_binding_value(
    const BindingRegistry::Record& record,
    const BindingRegistry::StagedBindingValue& staged,
    v2::BindingValue& value) {
  value.set_binding_id(record.binding_id);
  value.set_binding_layout_id(record.binding_layout_id);
  value.set_binding_value_id(staged.binding_value_id);
  value.set_seal_generation(staged.expected_previous_seal_generation);
  if (!staged.artifact_id.empty()) {
    value.set_source_artifact_id(staged.artifact_id);
  }
  if (!staged.selection.artifact_id().empty()) {
    value.mutable_selection()->CopyFrom(staged.selection);
  }
  value.set_is_artifact_backed(!staged.artifact_id.empty());
  value.set_verification_state(staged.verification_state);
  if (!staged.verification_failure_reason.empty()) {
    value.set_verification_failure_reason(staged.verification_failure_reason);
  }
}

void fill_group_realization_acquire_ref(
    const BindingRegistry::StagedBindingValue& staged,
    v2::GroupRealizationAcquireRef& acquire) {
  acquire.set_transaction_id(staged.transaction_id);
  acquire.set_version_set_id(staged.version_set_id);
  acquire.set_part_id(staged.part_id);
  acquire.set_staging_token(staged.staging_token);
}

void clear_verification_metadata(BindingRegistry::Record* record) {
  record->verification_state = v2::BINDING_VALUE_VERIFICATION_STATE_UNSPECIFIED;
  record->verification_job_id.clear();
  record->source_artifact_ref.clear();
  record->local_serving_ref.clear();
  record->serving_artifact_id.clear();
  record->verification_failure_reason.clear();
}

void set_artifact_verification_metadata(BindingRegistry::Record* record, std::string_view artifact_id) {
  clear_verification_metadata(record);
  if (artifact_id.empty()) {
    return;
  }
  record->serving_artifact_id = std::string(artifact_id);
  record->verification_state = common::infer_artifact_id_kind(artifact_id) == tensorcast::common::ArtifactIdKind::kMi2
      ? v2::BINDING_VALUE_VERIFICATION_STATE_VERIFIED
      : v2::BINDING_VALUE_VERIFICATION_STATE_LOCAL_ONLY;
}

void set_local_ready_verification_metadata(
    BindingRegistry::Record* record,
    v2::BindingValueVerificationState state,
    std::string_view source_artifact_ref,
    std::string_view serving_artifact_id = std::string_view()) {
  record->verification_state = state;
  record->source_artifact_ref = std::string(source_artifact_ref);
  record->local_serving_ref = absl::StrCat("binding-local:", record->binding_id, ":", record->current_binding_value_id);
  record->serving_artifact_id = std::string(serving_artifact_id);
  record->verification_failure_reason.clear();
  if (record->verification_state == v2::BINDING_VALUE_VERIFICATION_STATE_VERIFIED) {
    record->verification_job_id.clear();
  }
}

void mark_ready_artifact(
    BindingRegistry::Record* record,
    std::string_view artifact_id,
    const tensorcast::common::v1::ArtifactSelection& selection,
    std::string_view canonical_index_json = std::string_view()) {
  record->current_artifact_id = std::string(artifact_id);
  record->current_artifact_canonical_index_json = std::string(canonical_index_json);
  record->current_selection = selection;
  record->binding_current_value_publication_token.clear();
  record->state = v2::BINDING_STATE_READY_ARTIFACT;
  record->active_update_epoch.clear();
  record->current_binding_value_id = mint_binding_value_id();
  record->seal_generation += 1;
  record->sealed_commit_result.reset();
  record->ready_at = absl::Now();
  set_artifact_verification_metadata(record, artifact_id);
}

void mark_allocated(BindingRegistry::Record* record) {
  record->current_artifact_id.clear();
  record->current_artifact_canonical_index_json.clear();
  record->current_selection.Clear();
  record->binding_current_value_publication_token.clear();
  record->current_binding_value_id.clear();
  record->state = v2::BINDING_STATE_ALLOCATED;
  record->active_update_epoch.clear();
  record->sealed_commit_result.reset();
  clear_verification_metadata(record);
}

void mark_mutable(BindingRegistry::Record* record, std::string_view update_epoch) {
  record->current_artifact_id.clear();
  record->current_artifact_canonical_index_json.clear();
  record->current_selection.Clear();
  record->binding_current_value_publication_token.clear();
  record->current_binding_value_id.clear();
  record->state = v2::BINDING_STATE_MUTABLE;
  record->active_update_epoch = std::string(update_epoch);
  record->sealed_commit_result.reset();
  clear_verification_metadata(record);
}

void mark_ready_local(BindingRegistry::Record* record) {
  record->current_artifact_id.clear();
  record->current_artifact_canonical_index_json.clear();
  record->current_selection.Clear();
  record->binding_current_value_publication_token.clear();
  record->state = v2::BINDING_STATE_READY_LOCAL;
  record->active_update_epoch.clear();
  record->current_binding_value_id = mint_binding_value_id();
  record->seal_generation += 1;
  record->sealed_commit_result.reset();
  record->ready_at = absl::Now();
  clear_verification_metadata(record);
}

void mark_dirty(BindingRegistry::Record* record) {
  record->current_artifact_id.clear();
  record->current_artifact_canonical_index_json.clear();
  record->current_selection.Clear();
  record->binding_current_value_publication_token.clear();
  record->current_binding_value_id.clear();
  record->state = v2::BINDING_STATE_DIRTY;
  record->active_update_epoch.clear();
  record->sealed_commit_result.reset();
  clear_verification_metadata(record);
}

std::string next_update_epoch(BindingRegistry::Record* record) {
  record->update_epoch_counter += 1;
  return absl::StrCat("bue:", record->binding_id, ":", record->update_epoch_counter);
}

store::loading::ReplicaKey contribution_lease_subject(std::string_view binding_id, std::string_view binding_value_id) {
  return store::loading::ReplicaKey{
      .artifact_id = absl::StrCat("binding-contribution:", binding_id, ":", binding_value_id),
      .view_id = std::nullopt,
      .device = store::DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
      .replica = 0,
  };
}

std::string daemon_identity_for_contribution(const WorkerIdentityStore& identity) {
  auto daemon_id = identity.daemon_id();
  if (!daemon_id.empty()) {
    return daemon_id;
  }
  auto worker_id = identity.worker_id();
  if (!worker_id.empty()) {
    return worker_id;
  }
  return "unknown-daemon";
}

absl::Status ensure_no_live_binding_contributions(
    const std::shared_ptr<store::components::IGlobalStoreClient>& client,
    std::string_view binding_id,
    std::string_view binding_value_id) {
  if (binding_id.empty() || binding_value_id.empty()) {
    return absl::OkStatus();
  }
  if (!client || !client->is_connected()) {
    return absl::OkStatus();
  }
  auto contributions_or = client->list_assembly_slot_occupancies(
      /*attempt_id=*/std::nullopt,
      /*slot_id=*/std::nullopt,
      binding_id,
      binding_value_id,
      {"accepted"});
  if (!contributions_or.ok()) {
    return contributions_or.status();
  }
  if (contributions_or->empty()) {
    return absl::OkStatus();
  }
  auto active_identities_or = coordination::list_active_contributor_identities(client);
  if (!active_identities_or.ok()) {
    return active_identities_or.status();
  }
  const auto& active_identities = *active_identities_or;
  const absl::Time now = absl::Now();
  std::vector<std::string> attempts;
  attempts.reserve(contributions_or->size());
  for (const auto& contribution : *contributions_or) {
    if (!coordination::slot_occupancy_is_live(contribution, now, &active_identities)) {
      continue;
    }
    attempts.push_back(contribution.attempt_id);
  }
  if (attempts.empty()) {
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(
      absl::StrCat("binding value has live assembly contributions: ", absl::StrJoin(attempts, ",")));
}

v2::MaterializeIntoTargetRequest build_unmapped_request(
    const tensorcast::common::v1::ArtifactSelection& source_selection,
    std::string_view artifact_id,
    const v2::TargetLayout& target_layout,
    int pid,
    std::string_view device_uuid,
    const v2::SourcePolicy* source_policy,
    std::optional<std::string_view> operation_id,
    v2::TransformPlacement placement) {
  v2::MaterializeIntoTargetRequest request;
  auto selection = source_selection;
  selection.set_artifact_id(std::string(artifact_id));
  request.mutable_selection()->CopyFrom(selection);
  request.mutable_target_layout()->CopyFrom(target_layout);
  request.set_pid(pid);
  request.set_device_uuid(std::string(device_uuid));
  if (source_policy != nullptr) {
    request.mutable_source_policy()->CopyFrom(*source_policy);
  }
  if (operation_id.has_value() && !operation_id->empty()) {
    request.set_operation_id(std::string(*operation_id));
  }
  request.set_placement(placement);
  return request;
}

v2::MaterializeIntoMappedTargetRequest build_mapped_request(
    const BindingRegistry::Record& record,
    std::string_view artifact_id,
    const v2::SourcePolicy* source_policy,
    std::optional<std::string_view> operation_id,
    v2::TransformPlacement placement) {
  v2::MaterializeIntoMappedTargetRequest request;
  auto selection = record.source_selection;
  selection.set_artifact_id(std::string(artifact_id));
  request.mutable_selection()->CopyFrom(selection);
  request.mutable_target_layout()->CopyFrom(record.target_layout);
  request.set_pid(record.owner_pid);
  request.set_device_uuid(record.device_uuid);
  if (source_policy != nullptr) {
    request.mutable_source_policy()->CopyFrom(*source_policy);
  }
  if (operation_id.has_value() && !operation_id->empty()) {
    request.set_operation_id(std::string(*operation_id));
  }
  request.set_placement(placement);
  request.mutable_copy_plan()->CopyFrom(record.copy_plan);
  for (const auto& spec : record.dst_tensors) {
    request.add_dst_tensors()->CopyFrom(spec);
  }
  return request;
}

struct SourceBoundMaterializationRequest {
  bool mapped{false};
  int owner_pid{0};
  std::string_view device_uuid;
  const tensorcast::common::v1::ArtifactSelection* source_selection{nullptr};
  const v2::PublicDiskSourceHandle* public_disk_source{nullptr};
  const v2::TargetLayout* target_layout{nullptr};
  std::string_view target_index_json;
  const v2::BindingRealizationPlan* realization_plan{nullptr};
  const v2::CopyPlan* copy_plan{nullptr};
  const std::vector<v2::MappedTensorSpec>* dst_tensors{nullptr};
  const v2::SourcePolicy* source_policy{nullptr};
  const v2::SourceExecutionTopology* execution_topology{nullptr};
  v2::CollectivePolicy collective_policy{v2::COLLECTIVE_POLICY_UNSPECIFIED};
  std::optional<std::string_view> operation_id;
  const v2::GroupRealizationOptions* group_realization{nullptr};
  bool staged_publish_supported{false};
  v2::TransformPlacement placement{v2::TRANSFORM_PLACEMENT_UNSPECIFIED};
};

struct PreparedSourceBoundPlan {
  std::string resolved_artifact_id;
  tensorcast::common::v1::ArtifactSelection current_selection;
  std::string canonical_index_json;
  uint64_t logical_total_size{0};
  v2::TransformPlacement requested_placement{v2::TRANSFORM_PLACEMENT_UNSPECIFIED};
  NormalizedMaterializationRequestContext request_context;
  OperationTransportContext transport_context;
  v2::CollectivePolicy collective_policy{v2::COLLECTIVE_POLICY_UNSPECIFIED};
  std::optional<store::loading::DiskSource> disk_source;
  std::optional<store::loading::DiskMetadata> disk_metadata;
  std::optional<MappedTargetMaterializationPlan> mapped_plan;
  std::optional<std::string> mapped_plan_cache_key;
  bool mapped_plan_cache_hit{false};
  std::optional<TargetMaterializationPlan> unmapped_plan;
  std::optional<GroupRealizationBeginContext> group_begin_context;
  bool execution_only_mutable{false};
};

struct PreparedSourceBoundExecution {
  store::loading::MaterializeHints hints;
  std::optional<store::runtime::ingestion::strategy::PreparedSourceBoundExecutionPlan> prepared_execution_plan;
};

std::string resolve_source_artifact_id(
    const tensorcast::common::v1::ArtifactSelection* source_selection,
    const v2::PublicDiskSourceHandle* public_disk_source) {
  if (public_disk_source != nullptr && !public_disk_source->artifact_id().empty()) {
    return public_disk_source->artifact_id();
  }
  if (source_selection != nullptr && !source_selection->artifact_id().empty()) {
    return source_selection->artifact_id();
  }
  if (public_disk_source == nullptr) {
    return std::string();
  }
  return std::string();
}

grpc::Status validate_public_disk_source_hints(
    const v2::PublicDiskSourceHandle& public_disk_source,
    const materialization_request_common::ArtifactResolution& resolution) {
  if (public_disk_source.canonical_index_bytes().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "public_disk_source.canonical_index_bytes is required"};
  }
  if (!public_disk_source.artifact_id().empty() &&
      public_disk_source.artifact_id() != resolution.resolved_artifact_id) {
    return {StatusCode::FAILED_PRECONDITION, "public_disk_source.artifact_id does not match daemon-resolved artifact"};
  }
  if (!public_disk_source.path().empty() && resolution.normalized_disk_path.has_value() &&
      public_disk_source.path() != resolution.normalized_disk_path->string()) {
    return {StatusCode::FAILED_PRECONDITION, "public_disk_source.path does not match daemon-resolved disk path"};
  }
  if (!resolution.local_import.has_value() ||
      resolution.local_import->source_kind != ArtifactSourceRegistry::SourceKind::kMountedSourceArtifact) {
    return {StatusCode::FAILED_PRECONDITION, "public_disk_source requires daemon-resolved mounted-source metadata"};
  }
  const auto& entry = *resolution.local_import;
  if (entry.canonical_index_json.empty()) {
    return {StatusCode::FAILED_PRECONDITION, "public_disk_source requires daemon-resolved canonical index"};
  }
  if (public_disk_source.canonical_index_bytes() != entry.canonical_index_json) {
    return {
        StatusCode::FAILED_PRECONDITION,
        "public_disk_source.canonical_index_bytes does not match daemon-resolved canonical index",
    };
  }
  if (!public_disk_source.source_index_bytes().empty()) {
    if (!entry.source_index_json.has_value() || public_disk_source.source_index_bytes() != *entry.source_index_json) {
      return {
          StatusCode::FAILED_PRECONDITION,
          "public_disk_source.source_index_bytes does not match daemon-resolved source index",
      };
    }
  }
  const std::optional<bool> public_is_safetensors = public_disk_source_is_safetensors(public_disk_source);
  const std::optional<bool> registry_is_safetensors = registry_source_is_safetensors(entry);
  if (public_is_safetensors.has_value() && registry_is_safetensors.has_value() &&
      *public_is_safetensors != *registry_is_safetensors) {
    return {
        StatusCode::FAILED_PRECONDITION,
        "public_disk_source.format_kind does not match daemon-resolved mounted-source format",
    };
  }
  const std::optional<bool> public_tensor_aware = public_disk_source_is_tensor_aware(public_disk_source);
  if (public_tensor_aware.has_value() && *public_tensor_aware != entry.tensor_aware_metadata) {
    return {
        StatusCode::FAILED_PRECONDITION,
        "public_disk_source.metadata_capability does not match daemon-resolved mounted-source metadata",
    };
  }
  return grpc::Status::OK;
}

grpc::Status prepare_source_bound_plan(
    const OwnedBindingService::Dep& d,
    RpcContext& rctx,
    const store::DeviceKey& device,
    const SourceBoundMaterializationRequest& request,
    PreparedSourceBoundPlan& out) {
  const auto profile_start = std::chrono::steady_clock::now();
  auto elapsed_sec = [](auto start, auto end) { return std::chrono::duration<double>(end - start).count(); };
  double policy_sec = 0.0;
  double resolve_sec = 0.0;
  double disk_metadata_sec = 0.0;
  double canonical_index_sec = 0.0;
  double target_offsets_sec = 0.0;
  double target_plan_sec = 0.0;
  const bool group_realization_enabled = request.group_realization != nullptr && request.group_realization->enabled();
  if ((request.source_selection == nullptr && request.public_disk_source == nullptr && !group_realization_enabled) ||
      request.target_layout == nullptr) {
    return {StatusCode::INVALID_ARGUMENT, "source-bound materialization request is incomplete"};
  }
  if (request.realization_plan != nullptr && request.realization_plan->entries_size() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "realization_plan.entries must be non-empty"};
  }
  if (request.realization_plan != nullptr && request.copy_plan != nullptr) {
    return {StatusCode::INVALID_ARGUMENT, "realization_plan and copy_plan are mutually exclusive"};
  }
  if (request.mapped && request.realization_plan == nullptr &&
      (request.copy_plan == nullptr || request.dst_tensors == nullptr)) {
    return {StatusCode::INVALID_ARGUMENT, "mapped source-bound materialization request is incomplete"};
  }

  out = PreparedSourceBoundPlan{};
  out.requested_placement = request.placement;
  const std::string_view operation_id =
      request.operation_id.has_value() && !request.operation_id->empty() ? *request.operation_id : std::string_view();
  auto transport_context_or = resolve_group_realization_transport_context(operation_id, request.group_realization);
  if (!transport_context_or.ok()) {
    return to_grpc_status(transport_context_or.status());
  }
  auto staged_publish_status =
      validate_group_realization_staged_publish_supported(request.group_realization, request.staged_publish_supported);
  if (!staged_publish_status.ok()) {
    return to_grpc_status(staged_publish_status);
  }
  out.transport_context = std::move(*transport_context_or);
  auto execution_topology_or = resolve_source_execution_topology(request.execution_topology);
  if (!execution_topology_or.ok()) {
    return to_grpc_status(execution_topology_or.status());
  }
  auto request_context_or = resolve_materialization_request_context(request.source_policy, *execution_topology_or);
  if (!request_context_or.ok()) {
    return to_grpc_status(request_context_or.status());
  }
  out.request_context = std::move(*request_context_or);
  auto collective_policy_or =
      resolve_collective_policy(request.collective_policy, out.request_context.execution_topology);
  if (!collective_policy_or.ok()) {
    return to_grpc_status(collective_policy_or.status());
  }
  out.collective_policy = *collective_policy_or;
  auto policy_done = std::chrono::steady_clock::now();
  policy_sec = elapsed_sec(profile_start, policy_done);

  const bool loopback_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  tensorcast::common::v1::ArtifactSelection effective_source_selection;
  if (request.source_selection != nullptr) {
    effective_source_selection = *request.source_selection;
  }
  std::string daemon_id = d.identity.daemon_id();
  if (daemon_id.empty()) {
    daemon_id = d.daemon_id;
  }
  auto begin_context_or = begin_or_join_group_realization_if_enabled(
      d.global_store_client, request.group_realization, daemon_id, d.daemon_session_id, d.identity.worker_id());
  if (!begin_context_or.ok()) {
    return to_grpc_status(begin_context_or.status());
  }
  const bool has_frozen_group_selection = begin_context_or->has_value();
  if (has_frozen_group_selection) {
    effective_source_selection = (*begin_context_or)->part_selection;
    out.group_begin_context = **begin_context_or;
    apply_group_realization_begin_context_to_transport_context(**begin_context_or, &out.transport_context);
  }
  materialization_request_common::ArtifactResolution resolution;
  const auto resolve_start = std::chrono::steady_clock::now();
  if (request.public_disk_source != nullptr) {
    if (!loopback_peer) {
      return {StatusCode::PERMISSION_DENIED, "public_disk_source is local-only (loopback/UDS)"};
    }
    if (has_frozen_group_selection && !request.public_disk_source->artifact_id().empty() &&
        request.public_disk_source->artifact_id() != effective_source_selection.artifact_id()) {
      return {StatusCode::FAILED_PRECONDITION, "public_disk_source.artifact_id conflicts with group frozen selection"};
    }
    const std::string resolved_artifact_id =
        resolve_source_artifact_id(&effective_source_selection, request.public_disk_source);
    if (resolved_artifact_id.empty()) {
      return {StatusCode::INVALID_ARGUMENT, "public_disk_source.artifact_id is required"};
    }
    auto resolution_or = resolve_artifact_and_disk_source(
        d.global_store_client,
        &d.disk_imports,
        d.storage_path,
        resolved_artifact_id,
        out.request_context.retrieval_policy.allow_disk,
        /*allow_local_import_fallback=*/true,
        /*loopback_peer=*/true,
        /*disk_expected_size=*/std::nullopt,
        /*lightweight_msa1_validation=*/true);
    if (!resolution_or.ok()) {
      return to_grpc_status(resolution_or.status());
    }
    resolution = std::move(*resolution_or);
    effective_source_selection.set_artifact_id(resolution.resolved_artifact_id);
  } else {
    if (effective_source_selection.artifact_id().empty()) {
      return {StatusCode::INVALID_ARGUMENT, "source_selection.artifact_id is required"};
    }
    auto resolution_or = resolve_artifact_and_disk_source(
        d.global_store_client,
        &d.disk_imports,
        d.storage_path,
        effective_source_selection.artifact_id(),
        out.request_context.retrieval_policy.allow_disk,
        /*allow_local_import_fallback=*/true,
        loopback_peer);
    if (!resolution_or.ok()) {
      return to_grpc_status(resolution_or.status());
    }
    resolution = std::move(*resolution_or);
    effective_source_selection.set_artifact_id(resolution.resolved_artifact_id);
  }
  const auto resolve_done = std::chrono::steady_clock::now();
  resolve_sec = elapsed_sec(resolve_start, resolve_done);
  out.resolved_artifact_id = resolution.resolved_artifact_id;
  out.disk_source = resolution.disk_source;
  if (request.public_disk_source != nullptr && request.public_disk_source->canonical_index_bytes().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "public_disk_source.canonical_index_bytes is required"};
  }
  if (request.public_disk_source != nullptr) {
    auto hint_status = validate_public_disk_source_hints(*request.public_disk_source, resolution);
    if (!hint_status.ok()) {
      return hint_status;
    }
  }
  const auto disk_metadata_start = std::chrono::steady_clock::now();
  auto disk_metadata_or = build_binding_disk_metadata(
      resolution.normalized_disk_path,
      out.resolved_artifact_id,
      device.ordinal,
      d.disk_imports,
      request.public_disk_source,
      resolution.local_import);
  if (!disk_metadata_or.ok()) {
    return to_grpc_status(disk_metadata_or.status());
  }
  out.disk_metadata = std::move(*disk_metadata_or);
  const auto disk_metadata_done = std::chrono::steady_clock::now();
  disk_metadata_sec = elapsed_sec(disk_metadata_start, disk_metadata_done);
  if (is_byte_only_disk_metadata(out.disk_metadata)) {
    if (request.realization_plan != nullptr) {
      return {StatusCode::INVALID_ARGUMENT, "binding realization requires tensor-aware mounted-source metadata"};
    }
    if (selection_requires_tensor_aware_metadata(effective_source_selection)) {
      return {StatusCode::INVALID_ARGUMENT, "selection/view requires tensor-aware mounted-source metadata"};
    }
  }

  const auto canonical_index_start = std::chrono::steady_clock::now();
  absl::StatusOr<std::string> canonical_json_or =
      request.public_disk_source != nullptr && !request.public_disk_source->canonical_index_bytes().empty()
      ? absl::StatusOr<std::string>(request.public_disk_source->canonical_index_bytes())
      : load_canonical_index_with_disk_fallback(
            d.engine,
            out.resolved_artifact_id,
            resolution.normalized_disk_path,
            device.ordinal,
            resolution.gs_connected);
  if (!canonical_json_or.ok()) {
    return to_grpc_status(canonical_json_or.status());
  }
  const auto canonical_index_done = std::chrono::steady_clock::now();
  canonical_index_sec = elapsed_sec(canonical_index_start, canonical_index_done);
  std::string canonical_index_json = std::move(*canonical_json_or);

  const auto target_offsets_start = std::chrono::steady_clock::now();
  auto offsets_or = resolve_target_offsets(*request.target_layout);
  if (!offsets_or.ok()) {
    return to_grpc_status(offsets_or.status());
  }
  const auto target_offsets_done = std::chrono::steady_clock::now();
  target_offsets_sec = elapsed_sec(target_offsets_start, target_offsets_done);

  const auto target_plan_start = std::chrono::steady_clock::now();
  const bool mapped_without_realization = request.mapped && request.realization_plan == nullptr;
  const bool source_artifact_is_mi2 = mi2_index_multihash(out.resolved_artifact_id).has_value();
  const bool source_artifact_index_matches_target = mapped_without_realization &&
      source_artifact_matches_target_index(out.resolved_artifact_id, request.target_index_json);
  bool mapped_direct_byte_space = false;
  bool mapped_direct_plan_attempted = false;
  if (mapped_without_realization && source_artifact_is_mi2) {
    mapped_direct_plan_attempted = true;
    const v2::TargetLayout direct_target_layout = build_direct_refill_target_plan_layout(
        *request.target_layout, effective_source_selection, source_artifact_index_matches_target);
    v2::MaterializeIntoTargetRequest unmapped_request = build_unmapped_request(
        effective_source_selection,
        out.resolved_artifact_id,
        direct_target_layout,
        request.owner_pid,
        request.device_uuid,
        request.source_policy,
        request.operation_id,
        request.placement);
    TargetMaterializationPlan unmapped_plan;
    auto plan_status = build_target_materialization_plan(
        d.engine,
        out.resolved_artifact_id,
        unmapped_request,
        direct_target_layout,
        *offsets_or,
        std::string(canonical_index_json),
        /*record_result=*/nullptr,
        unmapped_plan);
    if (plan_status.ok()) {
      mapped_direct_byte_space = true;
      out.logical_total_size = unmapped_plan.logical_total_size;
      out.current_selection = unmapped_plan.resolved_selection;
      out.canonical_index_json = unmapped_plan.canonical_index_json;
      out.unmapped_plan = std::move(unmapped_plan);
    } else {
      LOG(INFO) << "tc_profile prepare_source_bound_plan direct_target_plan_rejected"
                << " artifact_id=" << out.resolved_artifact_id << " target_device=" << device.ordinal
                << " status_code=" << plan_status.error_code() << " message=" << plan_status.error_message();
    }
  }
  if (request.realization_plan != nullptr) {
    MappedTargetMaterializationPlan mapped_plan;
    const std::string cache_key = binding_realization_plan_cache_key(
        out.resolved_artifact_id,
        effective_source_selection,
        *request.realization_plan,
        *request.target_layout,
        request.target_index_json,
        canonical_index_json,
        request.placement);
    bool plan_cache_hit = false;
    {
      auto& cache = binding_realization_plan_cache();
      absl::MutexLock lock(&cache.mu);
      auto it = cache.mapped_plans.find(cache_key);
      if (it != cache.mapped_plans.end()) {
        mapped_plan = it->second;
        plan_cache_hit = true;
      }
    }
    if (!plan_cache_hit) {
      auto plan_status = build_binding_realization_materialization_plan(
          d.engine,
          effective_source_selection,
          *request.realization_plan,
          out.resolved_artifact_id,
          *request.target_layout,
          request.target_index_json,
          *offsets_or,
          std::string(canonical_index_json),
          /*record_result=*/nullptr,
          mapped_plan);
      if (!plan_status.ok()) {
        return plan_status;
      }
      auto& cache = binding_realization_plan_cache();
      absl::MutexLock lock(&cache.mu);
      put_bounded_binding_realization_cache_entry(
          &cache.mapped_plans, &cache.mapped_plan_order, cache_key, mapped_plan);
    }
    out.logical_total_size = mapped_plan.logical_total_size;
    out.current_selection.Clear();
    out.canonical_index_json = mapped_plan.canonical_index_json;
    out.mapped_plan = std::move(mapped_plan);
    out.mapped_plan_cache_key = cache_key;
    out.mapped_plan_cache_hit = plan_cache_hit;
    out.execution_only_mutable = true;
  } else if (request.mapped && !mapped_direct_byte_space) {
    BindingRegistry::Record replay_record;
    replay_record.owner_pid = request.owner_pid;
    replay_record.device_uuid = std::string(request.device_uuid);
    replay_record.source_selection = effective_source_selection;
    replay_record.target_layout = *request.target_layout;
    replay_record.copy_plan = *request.copy_plan;
    replay_record.dst_tensors = *request.dst_tensors;
    v2::MaterializeIntoMappedTargetRequest mapped_request = build_mapped_request(
        replay_record, out.resolved_artifact_id, request.source_policy, request.operation_id, request.placement);
    MappedTargetMaterializationPlan mapped_plan;
    auto plan_status = build_mapped_target_materialization_plan(
        d.engine,
        mapped_request,
        out.resolved_artifact_id,
        *offsets_or,
        std::move(canonical_index_json),
        /*record_result=*/nullptr,
        mapped_plan);
    if (!plan_status.ok()) {
      return plan_status;
    }
    out.logical_total_size = mapped_plan.logical_total_size;
    out.current_selection =
        build_mapped_bound_selection(out.resolved_artifact_id, *request.target_layout, request.target_index_json);
    out.canonical_index_json = mapped_plan.canonical_index_json;
    out.mapped_plan = std::move(mapped_plan);
  } else if (!mapped_direct_byte_space) {
    v2::MaterializeIntoTargetRequest unmapped_request = build_unmapped_request(
        effective_source_selection,
        out.resolved_artifact_id,
        *request.target_layout,
        request.owner_pid,
        request.device_uuid,
        request.source_policy,
        request.operation_id,
        request.placement);
    TargetMaterializationPlan unmapped_plan;
    auto plan_status = build_target_materialization_plan(
        d.engine,
        out.resolved_artifact_id,
        unmapped_request,
        *request.target_layout,
        *offsets_or,
        std::move(canonical_index_json),
        /*record_result=*/nullptr,
        unmapped_plan);
    if (!plan_status.ok()) {
      return plan_status;
    }
    out.logical_total_size = unmapped_plan.logical_total_size;
    out.current_selection = unmapped_plan.resolved_selection;
    out.canonical_index_json = unmapped_plan.canonical_index_json;
    out.unmapped_plan = std::move(unmapped_plan);
  }
  const auto target_plan_done = std::chrono::steady_clock::now();
  target_plan_sec = elapsed_sec(target_plan_start, target_plan_done);
  LOG(INFO) << "tc_profile prepare_source_bound_plan timings"
            << " artifact_id=" << out.resolved_artifact_id << " target_device=" << device.ordinal
            << " mapped=" << request.mapped << " realization_plan=" << (request.realization_plan != nullptr)
            << " mapped_direct_byte_space=" << mapped_direct_byte_space
            << " mapped_direct_plan_attempted=" << mapped_direct_plan_attempted
            << " source_artifact_mi2=" << source_artifact_is_mi2
            << " source_artifact_index_matches_target=" << source_artifact_index_matches_target
            << " public_disk_source=" << (request.public_disk_source != nullptr) << " metadata_fast_public="
            << (request.public_disk_source != nullptr && !request.public_disk_source->canonical_index_bytes().empty() &&
                resolution.local_import.has_value())
            << " mapped_plan_cache_hit=" << (out.mapped_plan_cache_hit ? 1 : 0) << " policy_sec=" << policy_sec
            << " resolve_sec=" << resolve_sec << " disk_metadata_sec=" << disk_metadata_sec
            << " canonical_index_sec=" << canonical_index_sec << " target_offsets_sec=" << target_offsets_sec
            << " target_plan_sec=" << target_plan_sec << " total_sec=" << elapsed_sec(profile_start, target_plan_done);
  return Status::OK;
}

grpc::Status prepare_source_bound_execution(
    const OwnedBindingService::Dep& d,
    RpcContext& rctx,
    const v2::TargetLayout& target_layout,
    const OwnedStorageLayout& storage_layout,
    const PreparedSourceBoundPlan& prepared_plan,
    PreparedSourceBoundExecution& out) {
  const auto profile_start = std::chrono::steady_clock::now();
  auto elapsed_sec = [](auto start, auto end) { return std::chrono::duration<double>(end - start).count(); };
  out = PreparedSourceBoundExecution{};

  const std::chrono::milliseconds request_budget = resolve_owner_request_budget(rctx.server_context());
  out.hints.request_budget = request_budget;
  out.hints.transport_wait_timeout = request_budget;
  out.hints.artifact_id = prepared_plan.resolved_artifact_id;
  const std::string requester_worker_id = d.identity.worker_id();
  if (!requester_worker_id.empty()) {
    out.hints.transport_requester_worker_id = requester_worker_id;
  }
  apply_operation_transport_context(prepared_plan.transport_context, &out.hints);
  apply_request_context_to_hints(prepared_plan.request_context, &out.hints);
  const bool prefer_direct_disk_for_source_layout = prepared_plan.disk_source.has_value() &&
      prepared_plan.disk_metadata.has_value() && prepared_plan.disk_metadata->source_index_json.has_value();
  if (prefer_direct_disk_for_source_layout) {
    out.hints.set_retrieval_policy(
        store::loading::RetrievalPolicy{
            .preference = store::loading::SourcePreference::kPreferDisk,
            .allow_p2p = false,
            .allow_disk = prepared_plan.request_context.retrieval_policy.allow_disk,
        });
  }
  out.hints.verify = store::loading::MaterializeHints::Verify::NONE;
  out.hints.export_policy = store::loading::ExportPolicy::kForce;
  if (prepared_plan.disk_metadata.has_value()) {
    out.hints.disk_metadata = *prepared_plan.disk_metadata;
  }
  if (prepared_plan.disk_source.has_value()) {
    out.hints.source_mutation_policy = store::loading::SourceMutationPolicy::kReadOnly;
  }
  out.hints.require_collective_execution =
      prepared_plan.collective_policy == v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE;

  if (prepared_plan.mapped_plan.has_value()) {
    const auto& mapped_plan = *prepared_plan.mapped_plan;
    if (mapped_plan.view_plan.has_value() && mapped_plan.view_plan->transform.requires_materialization) {
      return {StatusCode::INVALID_ARGUMENT, "mapped binding does not support view transforms"};
    }
    if (!mapped_plan.resolved_selection.view_id().empty() || mapped_plan.view_spec.has_value() ||
        mapped_plan.view_plan.has_value()) {
      store::loading::VariantIdentity variant;
      variant.canonical_artifact_id = prepared_plan.resolved_artifact_id;
      if (!mapped_plan.resolved_selection.view_id().empty()) {
        variant.view_id = mapped_plan.resolved_selection.view_id();
      }
      if (mapped_plan.view_spec.has_value()) {
        variant.view_spec = mapped_plan.view_spec;
      }
      if (mapped_plan.view_plan.has_value()) {
        variant.cached_plan = mapped_plan.view_plan;
      }
      variant.canonical_index_json = mapped_plan.canonical_index_json;
      variant.placement = resolve_transform_placement(prepared_plan.requested_placement, mapped_plan.view_spec);
      out.hints.variant = std::move(variant);
    }

    const uint64_t generation =
        materialization_payload::compute_generation_from_index(mapped_plan.canonical_index_json);
    std::optional<std::string> execution_template_cache_key;
    std::optional<CachedMappedExecutionTemplate> cached_template;
    if (prepared_plan.mapped_plan_cache_key.has_value()) {
      execution_template_cache_key = mapped_execution_template_cache_key(
          *prepared_plan.mapped_plan_cache_key,
          prepared_plan.disk_metadata,
          prepared_plan.collective_policy,
          d.engine.options().materialization_strategy,
          prepared_plan.request_context.execution_topology,
          prepared_plan.disk_source.has_value());
      auto& cache = binding_realization_plan_cache();
      absl::MutexLock lock(&cache.mu);
      auto it = cache.execution_templates.find(*execution_template_cache_key);
      if (it != cache.execution_templates.end()) {
        cached_template = it->second;
      }
    }
    if (cached_template.has_value()) {
      store::runtime::ingestion::strategy::PreparedSourceBoundExecutionPlan prepared_execution_plan;
      prepared_execution_plan.resolved_plan.artifact_id = prepared_plan.resolved_artifact_id;
      prepared_execution_plan.resolved_plan.generation = generation;
      prepared_execution_plan.resolved_plan.variant = out.hints.variant;
      prepared_execution_plan.resolved_plan.canonical_index_json = mapped_plan.canonical_index_json;
      prepared_execution_plan.resolved_plan.target_layout = storage_layout.into_target;
      prepared_execution_plan.resolved_plan.representation_transform_contract =
          mapped_plan.representation.transform_contract;
      prepared_execution_plan.resolved_plan.representation_work_plan = cached_template->representation_work_plan;
      prepared_execution_plan.strategy_plan = cached_template->strategy_plan;
      out.prepared_execution_plan = std::move(prepared_execution_plan);
      const auto done = std::chrono::steady_clock::now();
      LOG(INFO) << "tc_profile prepare_source_bound_execution timings"
                << " artifact_id=" << prepared_plan.resolved_artifact_id << " mapped=1"
                << " target_total_size=" << storage_layout.total_size << " execution_template_cache_hit=1"
                << " build_resolved_plan_sec=0 strategy_plan_sec=0"
                << " total_sec=" << elapsed_sec(profile_start, done);
      return Status::OK;
    }

    const auto build_resolved_start = std::chrono::steady_clock::now();
    auto prepared_execution_plan_or = build_resolved_mapped_materialization_plan(
        prepared_plan.resolved_artifact_id,
        generation,
        storage_layout.into_target,
        mapped_plan,
        out.hints.variant,
        prepared_plan.disk_metadata.has_value() && prepared_plan.disk_metadata->source_index_json.has_value()
            ? std::optional<std::string_view>(*prepared_plan.disk_metadata->source_index_json)
            : std::nullopt);
    if (!prepared_execution_plan_or.ok()) {
      return to_grpc_status(prepared_execution_plan_or.status());
    }
    const auto build_resolved_done = std::chrono::steady_clock::now();
    const auto strategy_start = std::chrono::steady_clock::now();
    auto strategy_plan_or = store::runtime::ingestion::strategy::build_source_bound_execution_strategy_plan(
        prepared_execution_plan_or->resolved_plan,
        prepared_execution_plan_or->lowering_artifacts,
        normalize_source_bound_policy(prepared_plan.collective_policy),
        d.engine.options().materialization_strategy,
        prepared_plan.request_context.execution_topology,
        store::runtime::ingestion::strategy::SourceBoundSourceFacts{
            .disk_source_available = prepared_plan.disk_source.has_value(),
            .disk_source_is_safetensors =
                prepared_plan.disk_metadata.has_value() && prepared_plan.disk_metadata->is_safetensors.value_or(false),
        });
    if (!strategy_plan_or.ok()) {
      return to_grpc_status(strategy_plan_or.status());
    }
    const auto strategy_done = std::chrono::steady_clock::now();
    prepared_execution_plan_or->strategy_plan = *strategy_plan_or;
    if (execution_template_cache_key.has_value()) {
      CachedMappedExecutionTemplate template_entry{
          .representation_work_plan = prepared_execution_plan_or->resolved_plan.representation_work_plan,
          .strategy_plan = *strategy_plan_or,
      };
      auto& cache = binding_realization_plan_cache();
      absl::MutexLock lock(&cache.mu);
      put_bounded_binding_realization_cache_entry(
          &cache.execution_templates, &cache.execution_template_order, *execution_template_cache_key, template_entry);
    }
    prepared_execution_plan_or->lowering_artifacts.reset();
    out.prepared_execution_plan = std::move(*prepared_execution_plan_or);
    LOG(INFO) << "tc_profile prepare_source_bound_execution timings"
              << " artifact_id=" << prepared_plan.resolved_artifact_id << " mapped=1"
              << " target_total_size=" << storage_layout.total_size << " execution_template_cache_hit=0"
              << " build_resolved_plan_sec=" << elapsed_sec(build_resolved_start, build_resolved_done)
              << " strategy_plan_sec=" << elapsed_sec(strategy_start, strategy_done)
              << " total_sec=" << elapsed_sec(profile_start, strategy_done);
    return Status::OK;
  }

  if (!prepared_plan.unmapped_plan.has_value()) {
    return {StatusCode::FAILED_PRECONDITION, "source-bound materialization plan is missing execution state"};
  }
  const auto& unmapped_plan = *prepared_plan.unmapped_plan;
  if (unmapped_plan.view_plan.has_value()) {
    store::loading::VariantIdentity variant;
    variant.canonical_artifact_id = prepared_plan.resolved_artifact_id;
    if (unmapped_plan.resolved_view_id.has_value()) {
      variant.view_id = *unmapped_plan.resolved_view_id;
    }
    if (unmapped_plan.view_spec.has_value()) {
      variant.view_spec = unmapped_plan.view_spec;
    }
    variant.cached_plan = unmapped_plan.view_plan;
    variant.canonical_index_json = unmapped_plan.canonical_index_json;
    variant.placement = resolve_transform_placement(prepared_plan.requested_placement, unmapped_plan.view_spec);
    out.hints.variant = std::move(variant);
  }
  return Status::OK;
}

} // namespace

store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary summarize_source_bound_plan_for_testing(
    const store::runtime::ingestion::strategy::ResolvedMaterializationPlan& resolved_plan,
    const std::optional<store::runtime::ingestion::strategy::SourceBoundLoweringArtifacts>& lowering_artifacts,
    const store::StoreEngineOptions::MaterializationStrategyConfig& strategy_config,
    const store::loading::ExecutionTopologyContext& execution_topology,
    v2::CollectivePolicy collective_policy,
    bool disk_source_available) {
  return summarize_source_bound_plan(
      resolved_plan,
      lowering_artifacts,
      strategy_config,
      execution_topology,
      collective_policy,
      store::runtime::ingestion::strategy::SourceBoundSourceFacts{
          .disk_source_available = disk_source_available,
          .disk_source_is_safetensors = disk_source_available,
      });
}

grpc::Status evaluate_strict_collective_preflight_for_testing(
    RpcContext* rctx,
    const store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary* plan_summary,
    v2::CollectivePolicy collective_policy) {
  if (collective_policy != v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE || plan_summary == nullptr ||
      plan_summary->strict_pure_collective_eligible) {
    return Status::OK;
  }
  return make_collective_failure_status(
      rctx,
      StatusCode::FAILED_PRECONDITION,
      append_pure_collective_blockers(
          "collective requested but the source-bound plan is not pure-collective eligible", *plan_summary),
      v2::CollectiveFailureClass::COLLECTIVE_FAILURE_CLASS_NOT_ELIGIBLE);
}

OwnedBindingService::OwnedBindingService(Dep d)
    : d_(std::move(d)), contribution_keepalive_tracker_(std::make_shared<ContributionLeaseKeepaliveTracker>()) {}

std::string OwnedBindingService::promotion_job_key(std::string_view binding_id, std::string_view binding_value_id)
    const {
  return absl::StrCat(binding_id, ":", binding_value_id);
}

void OwnedBindingService::fill_promotion_status_from_job(
    const std::shared_ptr<PromotionJobRecord>& job,
    v2::BindingPromotionStatus& status) const {
  if (job == nullptr) {
    return;
  }
  status.CopyFrom(job->status);
}

void OwnedBindingService::cancel_promotion_jobs_for_value(
    std::string_view binding_id,
    std::string_view binding_value_id,
    std::string_view reason) {
  if (binding_id.empty() || binding_value_id.empty()) {
    return;
  }
  absl::MutexLock lock(&promotion_jobs_mu_);
  const std::string key = promotion_job_key(binding_id, binding_value_id);
  const auto id_it = promotion_job_ids_by_value_.find(key);
  if (id_it == promotion_job_ids_by_value_.end()) {
    return;
  }
  const auto job_it = promotion_jobs_by_id_.find(id_it->second);
  if (job_it == promotion_jobs_by_id_.end()) {
    promotion_job_ids_by_value_.erase(id_it);
    return;
  }
  const auto current_state = job_it->second->status.state();
  if (current_state == v2::BINDING_PROMOTION_JOB_STATE_SUCCEEDED ||
      current_state == v2::BINDING_PROMOTION_JOB_STATE_FAILED ||
      current_state == v2::BINDING_PROMOTION_JOB_STATE_CANCELED) {
    return;
  }
  job_it->second->status.set_state(v2::BINDING_PROMOTION_JOB_STATE_CANCELED);
  job_it->second->status.set_failure_reason(std::string(reason));
}

grpc::Status OwnedBindingService::create_binding(
    RpcContext& rctx,
    const v2::CreateBindingRequest& req,
    v2::CreateBindingResponse& resp) {
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req.ownership() == v2::BINDING_OWNERSHIP_UNSPECIFIED) {
    return {StatusCode::INVALID_ARGUMENT, "ownership is required"};
  }
  if (req.binding_layout_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_layout_id is required"};
  }
  if (req.target_layout().storages_size() == 0 || req.target_layout().offsets_size() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "target_layout storages and offsets are required"};
  }
  if (req.target_index_bytes().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "target_index_bytes is required"};
  }
  if (req.binding_layout_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_layout_id is required"};
  }
  if (req.pid() <= 0) {
    return {StatusCode::INVALID_ARGUMENT, "pid is required"};
  }
  const bool mapped = req.copy_plan().entries_size() > 0 || req.dst_tensors_size() > 0;
  if (req.copy_plan().entries_size() > 0 && req.dst_tensors_size() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "mapped binding requires dst_tensors"};
  }
  if (req.has_initial_selection() && req.initial_selection().artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "initial_selection.artifact_id is required when provided"};
  }
  if (!req.has_initial_selection() && req.has_source_artifact_id()) {
    return {StatusCode::INVALID_ARGUMENT, "source_artifact_id requires initial_selection"};
  }
  if (req.ownership() == v2::BINDING_OWNERSHIP_DAEMON && req.has_initial_selection()) {
    return {StatusCode::INVALID_ARGUMENT, "daemon-owned CreateBinding does not accept initial_selection"};
  }

  const auto device = d_.devices.From(v2::DeviceType::DEVICE_TYPE_GPU, req.device_uuid(), std::nullopt);
  if (device.type != DeviceType::GPU || device.ordinal < 0) {
    return {StatusCode::INVALID_ARGUMENT, "binding requires a CUDA device"};
  }

  google::protobuf::RepeatedPtrField<std::string> no_tensor_filter;
  auto descriptors_or = build_descriptors_from_index(
      std::string_view(req.target_index_bytes().data(), req.target_index_bytes().size()),
      no_tensor_filter,
      req.device_uuid());
  if (!descriptors_or.ok()) {
    return to_grpc_status(descriptors_or.status());
  }

  std::unique_ptr<common::memory::GpuDeviceMemory> allocation;
  cuda::IpcHandleBytes handle_bytes;
  if (req.ownership() == v2::BINDING_OWNERSHIP_DAEMON) {
    if (d_.handle_leases == nullptr) {
      return {StatusCode::FAILED_PRECONDITION, "local handle plane is disabled (no handle leases)"};
    }
    auto offsets_or = resolve_target_offsets(req.target_layout());
    if (!offsets_or.ok()) {
      return to_grpc_status(offsets_or.status());
    }
    uint64_t logical_total_size = 0;
    for (const auto& offset : *offsets_or) {
      const uint64_t end = offset.storage_offset + offset.logical_length;
      if (end > logical_total_size) {
        logical_total_size = end;
      }
    }
    if (logical_total_size == 0) {
      return {StatusCode::INVALID_ARGUMENT, "target_layout logical size must be non-zero"};
    }
    allocation = std::make_unique<common::memory::GpuDeviceMemory>();
    if (auto allocate_status = allocation->allocate(logical_total_size, device.ordinal); !allocate_status.ok()) {
      return to_grpc_status(allocate_status);
    }
    handle_bytes = cuda::IpcHandleBytes::from_native(allocation->get_handle());
    if (!handle_bytes.is_valid()) {
      return {StatusCode::FAILED_PRECONDITION, "failed to export CUDA IPC handle for binding"};
    }
    auto storage_layout_or = build_owned_storage_layout(
        req.target_layout(), device.ordinal, gsl::not_null<void*>{allocation->get()}, handle_bytes);
    if (!storage_layout_or.ok()) {
      return to_grpc_status(storage_layout_or.status());
    }
    (void)storage_layout_or;
  }

  auto record = std::make_shared<BindingRegistry::Record>();
  record->binding_id = mint_binding_id();
  record->binding_layout_id = req.binding_layout_id();
  record->owner_pid = req.pid();
  record->creator_pid = req.pid();
  record->daemon_id = !d_.identity.daemon_id().empty() ? d_.identity.daemon_id() : d_.daemon_id;
  record->daemon_session_id = d_.daemon_session_id;
  record->created_at = absl::Now();
  record->reserved_at = record->created_at;
  record->device_id = device.ordinal;
  record->device_uuid = req.device_uuid();
  record->ownership = req.ownership();
  record->state = v2::BINDING_STATE_ALLOCATED;
  record->mapped = mapped;
  record->closed = false;
  record->export_refs = req.ownership() == v2::BINDING_OWNERSHIP_DAEMON ? 1 : 0;
  record->allocation = std::move(allocation);
  record->handle_bytes = handle_bytes;
  record->target_layout = req.target_layout();
  record->target_index_json = std::string(req.target_index_bytes().data(), req.target_index_bytes().size());
  record->target_layout_hash = compute_target_layout_hash(req.target_layout());
  record->payloads.assign(descriptors_or->descriptors.begin(), descriptors_or->descriptors.end());
  if (mapped) {
    record->copy_plan = req.copy_plan();
    record->dst_tensors.assign(req.dst_tensors().begin(), req.dst_tensors().end());
  }
  if (req.has_initial_selection()) {
    record->source_selection = req.initial_selection();
    std::string_view canonical_index_json;
    if (req.target_layout().index_kind() == v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED &&
        !selection_declares_view_or_subset(req.initial_selection())) {
      canonical_index_json = std::string_view(req.target_index_bytes().data(), req.target_index_bytes().size());
    }
    mark_ready_artifact(
        record.get(),
        req.has_source_artifact_id() ? req.source_artifact_id() : req.initial_selection().artifact_id(),
        req.initial_selection(),
        canonical_index_json);
  } else {
    mark_allocated(record.get());
  }

  if (auto insert_status = d_.bindings.insert(record); !insert_status.ok()) {
    return to_grpc_status(insert_status);
  }

  if (req.ownership() == v2::BINDING_OWNERSHIP_CLIENT && req.has_initial_selection() &&
      !record->current_artifact_canonical_index_json.empty()) {
    auto storage_layout_or = build_client_binding_publication_storage_layout(req.target_layout(), device.ordinal);
    if (!storage_layout_or.ok()) {
      const bool closed = d_.bindings.close_control(record->binding_id);
      if (!closed) {
        LOG(WARNING) << "binding cleanup could not find binding_id=" << record->binding_id;
      }
      return to_grpc_status(storage_layout_or.status());
    }
    auto token_or = maybe_mint_binding_current_value_publication_token(
        d_.capability_tokens,
        d_.identity,
        d_.target_materialization_service,
        record,
        d_.daemon_id,
        d_.daemon_session_id,
        std::string_view(),
        std::move(storage_layout_or->publish_segments),
        std::move(storage_layout_or->publish_storages));
    if (!token_or.ok()) {
      const bool closed = d_.bindings.close_control(record->binding_id);
      if (!closed) {
        LOG(WARNING) << "binding cleanup could not find binding_id=" << record->binding_id;
      }
      return to_grpc_status(token_or.status());
    }
    if (!token_or->empty()) {
      resp.set_binding_current_value_publication_token(*token_or);
    }
  }

  if (req.ownership() == v2::BINDING_OWNERSHIP_DAEMON) {
    auto token_or = d_.handle_leases->mint_external_cuda_lease(
        req.pid(),
        [registry = &d_.bindings, binding_id = record->binding_id]() { registry->release_export_ref(binding_id); });
    if (!token_or.ok()) {
      const bool closed = d_.bindings.close_control(record->binding_id);
      if (!closed) {
        LOG(WARNING) << "binding cleanup could not find binding_id=" << record->binding_id;
      }
      d_.bindings.release_export_ref(record->binding_id);
      return to_grpc_status(token_or.status());
    }
    resp.mutable_mem_handle()->set_cuda_ipc_handle(
        handle_bytes.as_string_view().data(), handle_bytes.as_string_view().size());
    resp.mutable_mem_handle()->set_lease_token(*token_or);
    for (auto& descriptor : descriptors_or->descriptors) {
      *resp.add_payloads() = std::move(descriptor);
    }
  }

  resp.set_binding_id(record->binding_id);
  resp.set_target_index_bytes(req.target_index_bytes());
  resp.set_state(record->state);
  if (!record->current_binding_value_id.empty()) {
    fill_binding_value(*record, *resp.mutable_current_value());
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status OwnedBindingService::create_owned_binding(
    RpcContext& rctx,
    const v2::CreateOwnedBindingRequest& req,
    v2::CreateOwnedBindingResponse& resp) {
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!req.has_source_selection()) {
    return {StatusCode::INVALID_ARGUMENT, "source_selection is required"};
  }
  if (req.source_selection().artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "source_selection.artifact_id is required"};
  }
  if (req.target_layout().storages_size() == 0 || req.target_layout().offsets_size() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "target_layout storages and offsets are required"};
  }
  if (req.target_index_bytes().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "target_index_bytes is required"};
  }
  if (req.binding_layout_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_layout_id is required"};
  }
  if (req.pid() <= 0) {
    return {StatusCode::INVALID_ARGUMENT, "pid is required"};
  }
  if (d_.handle_leases == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "local handle plane is disabled (no handle leases)"};
  }
  const bool mapped = req.copy_plan().entries_size() > 0 || req.dst_tensors_size() > 0;
  if (req.copy_plan().entries_size() > 0 && req.dst_tensors_size() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "mapped owner binding requires dst_tensors"};
  }

  const auto device = d_.devices.From(v2::DeviceType::DEVICE_TYPE_GPU, req.device_uuid(), std::nullopt);
  if (device.type != DeviceType::GPU || device.ordinal < 0) {
    return {StatusCode::INVALID_ARGUMENT, "bind() requires a CUDA device"};
  }

  std::vector<v2::MappedTensorSpec> mapped_dst_tensors(req.dst_tensors().begin(), req.dst_tensors().end());
  PreparedSourceBoundPlan prepared_plan;
  auto prepare_status = prepare_source_bound_plan(
      d_,
      rctx,
      device,
      SourceBoundMaterializationRequest{
          .mapped = mapped,
          .owner_pid = req.pid(),
          .device_uuid = req.device_uuid(),
          .source_selection = &req.source_selection(),
          .target_layout = &req.target_layout(),
          .target_index_json = std::string_view(req.target_index_bytes().data(), req.target_index_bytes().size()),
          .copy_plan = mapped ? &req.copy_plan() : nullptr,
          .dst_tensors = mapped ? &mapped_dst_tensors : nullptr,
          .source_policy = req.has_source_policy() ? &req.source_policy() : nullptr,
          .operation_id = req.has_operation_id() ? std::optional<std::string_view>(req.operation_id()) : std::nullopt,
          .group_realization = req.has_group_realization() ? &req.group_realization() : nullptr,
          .staged_publish_supported = true,
          .placement = req.placement(),
      },
      prepared_plan);
  if (!prepare_status.ok()) {
    return prepare_status;
  }
  const bool create_staged_value = req.has_group_realization() && req.group_realization().enabled() &&
      req.group_realization().require_staged_publish();
  if (create_staged_value && !prepared_plan.group_begin_context.has_value()) {
    return {StatusCode::FAILED_PRECONDITION, "group realization begin context is required for staged binding"};
  }
  std::string group_daemon_id = d_.identity.daemon_id();
  if (group_daemon_id.empty()) {
    group_daemon_id = d_.daemon_id;
  }

  auto allocation = std::make_unique<common::memory::GpuDeviceMemory>();
  if (auto allocate_status = allocation->allocate(prepared_plan.logical_total_size, device.ordinal);
      !allocate_status.ok()) {
    return to_grpc_status(allocate_status);
  }
  const cuda::IpcHandleBytes handle_bytes = cuda::IpcHandleBytes::from_native(allocation->get_handle());
  if (!handle_bytes.is_valid()) {
    return {StatusCode::FAILED_PRECONDITION, "failed to export CUDA IPC handle for owner binding"};
  }

  auto storage_layout_or = build_owned_storage_layout(
      req.target_layout(), device.ordinal, gsl::not_null<void*>{allocation->get()}, handle_bytes);
  if (!storage_layout_or.ok()) {
    return to_grpc_status(storage_layout_or.status());
  }
  OwnedStorageLayout storage_layout = std::move(*storage_layout_or);

  PreparedSourceBoundExecution prepared_execution;
  auto execution_status =
      prepare_source_bound_execution(d_, rctx, req.target_layout(), storage_layout, prepared_plan, prepared_execution);
  if (!execution_status.ok()) {
    return execution_status;
  }

  absl::StatusOr<store::loading::MaterializeIntoTargetResult> result_or = mapped
      ? d_.engine.materialize_mapped_into_target(
            device, *prepared_execution.prepared_execution_plan, prepared_execution.hints, prepared_plan.disk_source)
      : d_.engine.materialize_into_target(
            device,
            storage_layout.into_target,
            prepared_plan.canonical_index_json,
            materialization_payload::compute_generation_from_index(prepared_plan.canonical_index_json),
            prepared_execution.hints,
            prepared_plan.disk_source);
  if (!result_or.ok()) {
    return to_grpc_status(result_or.status());
  }
  auto preflight_or = serving_artifact_manifest::preflight_serving_artifact(
      &d_.engine,
      serving_artifact_manifest::build_preflight_request(
          prepared_plan.resolved_artifact_id,
          prepared_plan.canonical_index_json,
          prepared_plan.disk_source,
          prepared_plan.disk_metadata,
          req.has_serving_artifact_policy() ? &req.serving_artifact_policy() : nullptr));
  if (!preflight_or.ok()) {
    return to_grpc_status(preflight_or.status());
  }

  google::protobuf::RepeatedPtrField<std::string> no_tensor_filter;
  auto descriptors_or = build_descriptors_from_index(
      std::string_view(req.target_index_bytes().data(), req.target_index_bytes().size()),
      no_tensor_filter,
      req.device_uuid());
  if (!descriptors_or.ok()) {
    return to_grpc_status(descriptors_or.status());
  }

  const std::string target_layout_hash = compute_target_layout_hash(req.target_layout());
  std::shared_ptr<common::memory::GpuDeviceMemory> staged_allocation;
  if (create_staged_value) {
    staged_allocation = std::shared_ptr<common::memory::GpuDeviceMemory>(std::move(allocation));
  }
  std::string binding_id;
  std::shared_ptr<BindingRegistry::Record> record;
  absl::Status insert_status = absl::UnknownError("uninitialized");
  for (int attempt = 0; attempt < 4; ++attempt) {
    binding_id = mint_binding_id();
    record = std::make_shared<BindingRegistry::Record>();
    record->binding_id = binding_id;
    record->binding_layout_id = req.binding_layout_id();
    record->owner_pid = req.pid();
    record->creator_pid = req.pid();
    record->daemon_id = !d_.identity.daemon_id().empty() ? d_.identity.daemon_id() : d_.daemon_id;
    record->daemon_session_id = d_.daemon_session_id;
    record->created_at = absl::Now();
    record->reserved_at = record->created_at;
    record->device_id = device.ordinal;
    record->device_uuid = req.device_uuid();
    record->ownership = v2::BINDING_OWNERSHIP_DAEMON;
    record->mapped = mapped;
    record->closed = false;
    record->export_refs = 1;
    if (!create_staged_value) {
      record->allocation = std::move(allocation);
    }
    record->handle_bytes = handle_bytes;
    record->source_selection = req.source_selection();
    record->source_selection.set_artifact_id(prepared_plan.resolved_artifact_id);
    record->target_layout = req.target_layout();
    record->target_index_json = std::string(req.target_index_bytes().data(), req.target_index_bytes().size());
    record->target_layout_hash = target_layout_hash;
    if (create_staged_value) {
      record->state = v2::BINDING_STATE_READY_LOCAL;
      record->active_update_epoch.clear();
      record->ready_at = absl::Now();
      clear_verification_metadata(record.get());
    } else {
      mark_ready_artifact(
          record.get(),
          prepared_plan.resolved_artifact_id,
          prepared_plan.current_selection,
          prepared_plan.canonical_index_json);
    }
    if (mapped) {
      record->copy_plan = req.copy_plan();
      record->dst_tensors.assign(req.dst_tensors().begin(), req.dst_tensors().end());
    }
    record->payloads.assign(descriptors_or->descriptors.begin(), descriptors_or->descriptors.end());
    insert_status = d_.bindings.insert(record);
    if (insert_status.ok()) {
      break;
    }
    if (!absl::IsAlreadyExists(insert_status)) {
      return to_grpc_status(insert_status);
    }
  }
  if (!insert_status.ok()) {
    return to_grpc_status(insert_status);
  }

  std::optional<BindingRegistry::StagedBindingValue> created_staged;
  if (create_staged_value) {
    BindingRegistry::StagedBindingValue staged;
    staged.transaction_id = prepared_plan.group_begin_context->transaction_id;
    staged.version_set_id = prepared_plan.group_begin_context->version_set.version_set_id();
    staged.part_id = prepared_plan.group_begin_context->part_id;
    staged.binding_value_id = mint_binding_value_id();
    staged.staging_token = absl::StrCat("stage:", binding_id, ":", staged.binding_value_id);
    staged.staging_epoch = 1;
    staged.daemon_id = group_daemon_id;
    staged.daemon_session_id = d_.daemon_session_id;
    staged.materialization_attempt_id = req.has_operation_id() && !req.operation_id().empty()
        ? req.operation_id()
        : absl::StrCat("create:", binding_id);
    staged.allocation = staged_allocation;
    staged.handle_bytes = handle_bytes;
    staged.selection = prepared_plan.current_selection;
    staged.artifact_id = prepared_plan.resolved_artifact_id;
    staged.canonical_index_json = prepared_plan.canonical_index_json;
    staged.target_layout = req.target_layout();
    staged.target_index_json = std::string(req.target_index_bytes().data(), req.target_index_bytes().size());
    staged.target_layout_hash = target_layout_hash;
    staged.payloads.assign(descriptors_or->descriptors.begin(), descriptors_or->descriptors.end());
    staged.logical_total_size = prepared_plan.logical_total_size;
    staged.expected_previous_seal_generation = record->seal_generation;
    staged.verification_state = v2::BINDING_VALUE_VERIFICATION_STATE_LOCAL_ONLY;
    staged.created_at = absl::Now();
    auto staged_status = d_.bindings.insert_staged_value(binding_id, staged, staged.created_at);
    if (!staged_status.ok()) {
      const bool closed = d_.bindings.close_control(binding_id);
      if (!closed) {
        LOG(WARNING) << "staged owner binding cleanup could not find binding_id=" << binding_id;
      }
      d_.bindings.release_export_ref(binding_id);
      return to_grpc_status(staged_status);
    }
    created_staged = std::move(staged);
  }

  auto token_or = d_.handle_leases->mint_external_cuda_lease(
      req.pid(), [registry = &d_.bindings, binding_id]() { registry->release_export_ref(binding_id); });
  if (!token_or.ok()) {
    const bool closed = d_.bindings.close_control(binding_id);
    if (!closed) {
      LOG(WARNING) << "owner binding cleanup could not find binding_id=" << binding_id;
    }
    d_.bindings.release_export_ref(binding_id);
    return to_grpc_status(token_or.status());
  }

  absl::StatusOr<std::string> binding_current_value_publication_token_or = std::string();
  if (!create_staged_value) {
    binding_current_value_publication_token_or = maybe_mint_binding_current_value_publication_token(
        d_.capability_tokens,
        d_.identity,
        d_.target_materialization_service,
        record,
        d_.daemon_id,
        d_.daemon_session_id,
        req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view(),
        std::move(storage_layout.publish_segments),
        std::move(storage_layout.publish_storages));
  }
  if (!binding_current_value_publication_token_or.ok()) {
    auto release_status = d_.handle_leases->release(*token_or);
    if (!release_status.ok()) {
      LOG(WARNING) << "failed to release owner binding bootstrap lease after publication token failure: "
                   << release_status;
    }
    const bool closed = d_.bindings.close_control(binding_id);
    if (!closed) {
      LOG(WARNING) << "owner binding cleanup after publication token failure could not find binding_id=" << binding_id;
    }
    return to_grpc_status(binding_current_value_publication_token_or.status());
  }

  if (create_staged_value) {
    GroupRealizationPreparedMemberContext prepared_member{
        .binding_id = binding_id,
        .binding_value_id = created_staged->binding_value_id,
        .staging_token = created_staged->staging_token,
        .staging_epoch = created_staged->staging_epoch,
        .expected_previous_seal_generation = created_staged->expected_previous_seal_generation,
        .materialization_attempt_id = created_staged->materialization_attempt_id,
        .source_replica_id = created_staged->source_replica_id,
        .source_export_generation = created_staged->source_export_generation,
        .child_transport_request_id = prepared_plan.transport_context.transport_request_id,
    };
    auto report_or = report_group_realization_prepared_if_enabled(
        d_.global_store_client,
        req.has_group_realization() ? &req.group_realization() : nullptr,
        &*prepared_plan.group_begin_context,
        prepared_member,
        group_daemon_id,
        d_.daemon_session_id,
        d_.identity.worker_id());
    if (!report_or.ok()) {
      const auto remove_status =
          d_.bindings.remove_staged_value(binding_id, created_staged->binding_value_id, "prepared_report_failed");
      if (!remove_status.ok()) {
        LOG(WARNING) << "failed to remove staged binding after prepared report failure: " << remove_status;
      }
      auto release_status = d_.handle_leases->release(*token_or);
      if (!release_status.ok()) {
        LOG(WARNING) << "failed to release staged binding bootstrap lease after prepared report failure: "
                     << release_status;
      }
      const bool closed = d_.bindings.close_control(binding_id);
      if (!closed) {
        LOG(WARNING) << "failed to close staged binding after prepared report failure binding_id=" << binding_id;
      }
      return to_grpc_status(report_or.status());
    }
  }

  resp.set_binding_id(binding_id);
  resp.set_artifact_id(prepared_plan.resolved_artifact_id);
  resp.mutable_mem_handle()->set_cuda_ipc_handle(
      handle_bytes.as_string_view().data(), handle_bytes.as_string_view().size());
  resp.mutable_mem_handle()->set_lease_token(*token_or);
  resp.set_target_index_bytes(req.target_index_bytes());
  for (auto& descriptor : descriptors_or->descriptors) {
    *resp.add_payloads() = std::move(descriptor);
  }
  if (!binding_current_value_publication_token_or->empty()) {
    resp.set_binding_current_value_publication_token(*binding_current_value_publication_token_or);
  }
  resp.mutable_resolved_selection()->CopyFrom(prepared_plan.current_selection);
  resp.set_source(to_proto_source(result_or->source));
  if (!result_or->selected_source_replica_id.empty()) {
    resp.set_selected_source_replica_id(result_or->selected_source_replica_id);
  }
  if (!result_or->selected_source_transport_id.empty()) {
    resp.set_selected_source_transport_id(result_or->selected_source_transport_id);
  }
  resp.set_state(create_staged_value ? v2::BINDING_STATE_READY_LOCAL : v2::BINDING_STATE_READY_ARTIFACT);
  if (create_staged_value) {
    resp.set_created_staged_value(true);
    fill_staged_binding_value(*record, *created_staged, *resp.mutable_staged_value());
    fill_group_realization_acquire_ref(*created_staged, *resp.mutable_group_realization_acquire());
  } else {
    fill_binding_value(*record, *resp.mutable_current_value());
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status OwnedBindingService::commit_binding_artifact(
    RpcContext& rctx,
    const v2::CommitBindingArtifactRequest& req,
    v2::CommitBindingArtifactResponse& resp) {
  if (req.binding_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_id is required"};
  }
  if (!req.has_selection() || req.selection().artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "selection.artifact_id is required"};
  }
  auto record_or = d_.bindings.get(req.binding_id());
  if (!record_or.ok()) {
    return to_grpc_status(record_or.status());
  }
  const auto record = *record_or;
  std::string current_binding_value_id;
  {
    absl::MutexLock lock(&record->mu);
    if (record->closed) {
      return {StatusCode::FAILED_PRECONDITION, "binding is closed"};
    }
    if (record->state == v2::BINDING_STATE_MUTABLE) {
      return {StatusCode::FAILED_PRECONDITION, "binding is mutable"};
    }
    if (record->state == v2::BINDING_STATE_DIRTY) {
      return {StatusCode::FAILED_PRECONDITION, "binding is dirty"};
    }
    if (auto guard_status = ensure_binding_current_value_not_published(*record); !guard_status.ok()) {
      return to_grpc_status(guard_status);
    }
    current_binding_value_id = record->current_binding_value_id;
  }
  if (auto contribution_status =
          ensure_no_live_binding_contributions(d_.global_store_client, record->binding_id, current_binding_value_id);
      !contribution_status.ok()) {
    return to_grpc_status(contribution_status);
  }
  cancel_promotion_jobs_for_value(
      record->binding_id, current_binding_value_id, "binding current value replaced by artifact commit");
  {
    absl::MutexLock lock(&record->mu);
    if (record->closed) {
      return {StatusCode::FAILED_PRECONDITION, "binding is closed"};
    }
    if (record->state == v2::BINDING_STATE_MUTABLE) {
      return {StatusCode::FAILED_PRECONDITION, "binding is mutable"};
    }
    if (record->state == v2::BINDING_STATE_DIRTY) {
      return {StatusCode::FAILED_PRECONDITION, "binding is dirty"};
    }
    if (!current_binding_value_id.empty() && record->current_binding_value_id != current_binding_value_id) {
      return {StatusCode::FAILED_PRECONDITION, "binding current value changed during artifact commit"};
    }
    if (auto guard_status = ensure_binding_current_value_not_published(*record); !guard_status.ok()) {
      return to_grpc_status(guard_status);
    }
    mark_ready_artifact(
        record.get(),
        req.has_source_artifact_id() ? req.source_artifact_id() : req.selection().artifact_id(),
        req.selection());
    resp.set_state(record->state);
    fill_binding_value(*record, *resp.mutable_current_value());
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status OwnedBindingService::begin_binding_update(
    RpcContext& rctx,
    const v2::BeginBindingUpdateRequest& req,
    v2::BeginBindingUpdateResponse& resp) {
  if (req.binding_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_id is required"};
  }
  auto record_or = d_.bindings.get(req.binding_id());
  if (!record_or.ok()) {
    return to_grpc_status(record_or.status());
  }
  const auto record = *record_or;
  std::string current_binding_value_id;
  {
    absl::MutexLock lock(&record->mu);
    if (record->closed) {
      return {StatusCode::FAILED_PRECONDITION, "binding is closed"};
    }
    if (record->state == v2::BINDING_STATE_DIRTY) {
      return {StatusCode::FAILED_PRECONDITION, "binding is dirty"};
    }
    if (record->state == v2::BINDING_STATE_MUTABLE) {
      return {StatusCode::FAILED_PRECONDITION, "binding is already mutable"};
    }
    if (auto guard_status = ensure_binding_current_value_not_published(*record); !guard_status.ok()) {
      return to_grpc_status(guard_status);
    }
    current_binding_value_id = record->current_binding_value_id;
  }
  if (auto contribution_status =
          ensure_no_live_binding_contributions(d_.global_store_client, record->binding_id, current_binding_value_id);
      !contribution_status.ok()) {
    return to_grpc_status(contribution_status);
  }
  cancel_promotion_jobs_for_value(record->binding_id, current_binding_value_id, "binding update started");
  {
    absl::MutexLock lock(&record->mu);
    if (record->closed) {
      return {StatusCode::FAILED_PRECONDITION, "binding is closed"};
    }
    if (record->state == v2::BINDING_STATE_DIRTY) {
      return {StatusCode::FAILED_PRECONDITION, "binding is dirty"};
    }
    if (record->state == v2::BINDING_STATE_MUTABLE) {
      return {StatusCode::FAILED_PRECONDITION, "binding is already mutable"};
    }
    if (!current_binding_value_id.empty() && record->current_binding_value_id != current_binding_value_id) {
      return {StatusCode::FAILED_PRECONDITION, "binding current value changed during update preparation"};
    }
    if (auto guard_status = ensure_binding_current_value_not_published(*record); !guard_status.ok()) {
      return to_grpc_status(guard_status);
    }
    const std::string update_epoch = next_update_epoch(record.get());
    mark_mutable(record.get(), update_epoch);
    resp.set_update_epoch(update_epoch);
    resp.set_state(record->state);
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status OwnedBindingService::submit_binding_contribution(
    RpcContext& rctx,
    const v2::SubmitBindingContributionRequest& req,
    v2::SubmitBindingContributionResponse& resp) {
  if (req.attempt_id().empty() || req.workspace_assembly_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "attempt_id and workspace_assembly_id are required"};
  }
  if (req.binding_id().empty() || req.binding_value_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_id and binding_value_id are required"};
  }
  if (req.coverage_plan_hash().empty() || req.coordinator_operation_id().empty() || req.coordinator_generation() == 0) {
    return {
        StatusCode::INVALID_ARGUMENT,
        "coverage_plan_hash, coordinator_operation_id, and coordinator_generation are required"};
  }
  if (req.contribution_kind() == v2::BINDING_CONTRIBUTION_KIND_UNSPECIFIED) {
    return {StatusCode::INVALID_ARGUMENT, "contribution_kind is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }
  if (d_.lifecycle == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "session lifecycle is unavailable"};
  }

  auto binding_record_or = d_.bindings.get(req.binding_id());
  if (!binding_record_or.ok()) {
    return to_grpc_status(binding_record_or.status());
  }
  const auto binding_record = *binding_record_or;

  int owner_pid = 0;
  v2::BindingOwnership binding_ownership = v2::BINDING_OWNERSHIP_UNSPECIFIED;
  BindingRegistry::Record contribution_record;
  {
    absl::MutexLock lock(&binding_record->mu);
    if (binding_record->closed) {
      return {StatusCode::FAILED_PRECONDITION, "binding is closed"};
    }
    if (binding_record->state == v2::BINDING_STATE_DIRTY || binding_record->state == v2::BINDING_STATE_MUTABLE) {
      return {StatusCode::FAILED_PRECONDITION, "binding is not sealed"};
    }
    if (binding_record->current_binding_value_id.empty()) {
      return {StatusCode::FAILED_PRECONDITION, "binding has no current sealed value"};
    }
    if (binding_record->current_binding_value_id != req.binding_value_id()) {
      return {StatusCode::FAILED_PRECONDITION, "binding_value_id is no longer current"};
    }
    owner_pid = binding_record->owner_pid;
    binding_ownership = binding_record->ownership;
    contribution_record.owner_pid = binding_record->owner_pid;
    contribution_record.device_id = binding_record->device_id;
    contribution_record.device_uuid = binding_record->device_uuid;
    contribution_record.target_layout = binding_record->target_layout;
    contribution_record.target_index_json = binding_record->target_index_json;
    contribution_record.current_artifact_id = binding_record->current_artifact_id;
    contribution_record.current_artifact_canonical_index_json = binding_record->current_artifact_canonical_index_json;
    contribution_record.current_selection = binding_record->current_selection;
    contribution_record.source_selection = binding_record->source_selection;
    contribution_record.handle_bytes = binding_record->handle_bytes;
  }

  auto attempt_or = d_.global_store_client->get_assembly_attempt(req.attempt_id());
  if (!attempt_or.ok()) {
    return to_grpc_status(attempt_or.status());
  }
  v2::AssemblyAttemptRecord attempt_record;
  if (!attempt_record.ParseFromString(attempt_or->attempt_record_proto)) {
    return {StatusCode::FAILED_PRECONDITION, "assembly attempt record is malformed"};
  }
  if (attempt_record.workspace_assembly_id() != req.workspace_assembly_id()) {
    return {StatusCode::FAILED_PRECONDITION, "assembly attempt workspace changed"};
  }
  if (!req.attempt_intent_digest().empty() &&
      attempt_record.intent().attempt_intent_digest() != req.attempt_intent_digest()) {
    return {StatusCode::FAILED_PRECONDITION, "assembly attempt intent changed"};
  }

  auto binding_or = d_.global_store_client->get_assembly_layout_binding(req.workspace_assembly_id());
  if (!binding_or.ok()) {
    return to_grpc_status(binding_or.status());
  }
  if (binding_or->layout_id() != attempt_record.intent().layout_id()) {
    return {StatusCode::FAILED_PRECONDITION, "assembly layout binding changed"};
  }

  tensorcast::operation::v1::GetOperationRequest get_operation_req;
  get_operation_req.set_operation_id(req.coordinator_operation_id());
  auto operation_or = d_.global_store_client->get_operation(get_operation_req);
  if (!operation_or.ok()) {
    return to_grpc_status(operation_or.status());
  }
  if (operation_or->lease_generation() != req.coordinator_generation()) {
    return {StatusCode::FAILED_PRECONDITION, "coordinator generation changed"};
  }
  if (!coordination::operation_lease_is_live(*operation_or)) {
    return {StatusCode::FAILED_PRECONDITION, "coordinator lease is no longer live"};
  }
  if (!coordination::operation_allows_contributions(*operation_or)) {
    return {StatusCode::FAILED_PRECONDITION, "assembly attempt is no longer accepting contributions"};
  }

  auto registration_plan_or =
      build_binding_contribution_registration_plan(d_, rctx.server_context(), contribution_record, req);
  if (!registration_plan_or.ok()) {
    return to_grpc_status(registration_plan_or.status());
  }
  BindingContributionRegistrationPlan registration_plan = std::move(*registration_plan_or);

  const auto contract_status = coordination::validate_binding_requirement_entry(
      attempt_record.intent().requirements(),
      req.contribution_kind(),
      registration_plan.structural_view_id.has_value() ? std::string_view(*registration_plan.structural_view_id)
                                                       : std::string_view());
  if (!contract_status.ok()) {
    return to_grpc_status(contract_status);
  }
  const bool canonical_bound_fast_path = req.contribution_kind() == v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL &&
      binding_ownership == v2::BINDING_OWNERSHIP_DAEMON;
  std::optional<std::string> committed_view_id;
  const std::string slot_id = registration_plan.slot_id;
  auto active_identities_or = coordination::list_active_contributor_identities(d_.global_store_client);
  if (!active_identities_or.ok()) {
    return to_grpc_status(active_identities_or.status());
  }
  const auto& active_identities = *active_identities_or;
  const absl::Time now = absl::Now();

  auto existing_or = d_.global_store_client->get_assembly_slot_occupancy(req.attempt_id(), slot_id);
  if (existing_or.ok()) {
    const bool existing_live = coordination::slot_occupancy_is_live(*existing_or, now, &active_identities);
    const bool same_coordinator_attempt = existing_or->coordinator_operation_id == req.coordinator_operation_id() &&
        existing_or->coordinator_generation == req.coordinator_generation();
    if (existing_or->binding_id == req.binding_id() && existing_or->binding_value_id == req.binding_value_id() &&
        existing_live) {
      resp.set_accepted(true);
      resp.set_already_exists(true);
      resp.set_lease_id(existing_or->lease_id);
      resp.set_lease_generation(existing_or->lease_generation);
      resp.set_state(existing_or->state);
      resp.set_slot_id(existing_or->slot_id);
      rctx.mark_success();
      return Status::OK;
    }
    if (existing_live && !same_coordinator_attempt) {
      return {StatusCode::FAILED_PRECONDITION, "assembly contribution slot is already occupied by a live contributor"};
    }
  } else if (!absl::IsNotFound(existing_or.status())) {
    return to_grpc_status(existing_or.status());
  }

  if (!canonical_bound_fast_path) {
    const auto registration_status = commit_binding_contribution_registration(
        d_, rctx.server_context(), contribution_record, registration_plan, committed_view_id);
    if (!registration_status.ok()) {
      return registration_status;
    }
    if (registration_plan.structural_view_id.has_value() && committed_view_id.has_value() &&
        committed_view_id != registration_plan.structural_view_id) {
      return {StatusCode::FAILED_PRECONDITION, "piece contribution structural lowering changed during registration"};
    }
  }

  auto lease_id_holder = std::make_shared<std::string>();
  auto contribution_stop = std::make_shared<std::atomic<bool>>(false);
  auto contribution_client = d_.global_store_client;
  const std::string attempt_id = req.attempt_id();
  auto lease_or = d_.lifecycle->create_use_lease(
      contribution_lease_subject(req.binding_id(), req.binding_value_id()),
      owner_pid,
      {[contribution_client, attempt_id, slot_id, lease_id_holder, contribution_stop]() -> absl::Status {
        contribution_stop->store(true, std::memory_order_relaxed);
        if (!contribution_client || !contribution_client->is_connected() || lease_id_holder->empty()) {
          return absl::OkStatus();
        }
        auto state_or = contribution_client->update_assembly_slot_occupancy_state(
            attempt_id,
            slot_id,
            "stale",
            *lease_id_holder,
            /*expected_lease_generation=*/1,
            /*lease_expires_at=*/std::nullopt,
            {"accepted"});
        if (!state_or.ok() && !absl::IsNotFound(state_or.status())) {
          return state_or.status();
        }
        return absl::OkStatus();
      }});
  if (!lease_or.ok()) {
    return to_grpc_status(lease_or.status());
  }
  *lease_id_holder = absl::StrCat(*lease_or);

  store::components::AssemblySlotOccupancyInfo contribution;
  contribution.attempt_id = attempt_id;
  contribution.slot_id = slot_id;
  if (registration_plan.structural_view_id.has_value()) {
    contribution.structural_view_id =
        committed_view_id.has_value() ? *committed_view_id : *registration_plan.structural_view_id;
  }
  contribution.binding_id = req.binding_id();
  contribution.binding_value_id = req.binding_value_id();
  contribution.coverage_plan_hash = req.coverage_plan_hash();
  contribution.contributor_daemon_id = daemon_identity_for_contribution(d_.identity);
  contribution.coordinator_operation_id = req.coordinator_operation_id();
  contribution.coordinator_generation = req.coordinator_generation();
  contribution.lease_id = *lease_id_holder;
  contribution.lease_generation = 1;
  contribution.lease_expires_at = absl::Now() + coordination::kContributionLeaseTtl;
  contribution.state = "accepted";
  auto upsert_or = d_.global_store_client->upsert_assembly_slot_occupancy(contribution);
  if (!upsert_or.ok()) {
    if (absl::IsFailedPrecondition(upsert_or.status())) {
      auto current_or = d_.global_store_client->get_assembly_slot_occupancy(req.attempt_id(), slot_id);
      if (current_or.ok() && current_or->binding_id == req.binding_id() &&
          current_or->binding_value_id == req.binding_value_id() &&
          coordination::slot_occupancy_is_live(*current_or, now, &active_identities)) {
        d_.lifecycle->release_lease(*lease_or);
        resp.set_accepted(true);
        resp.set_already_exists(true);
        resp.set_lease_id(current_or->lease_id);
        resp.set_lease_generation(current_or->lease_generation);
        resp.set_state(current_or->state);
        resp.set_slot_id(current_or->slot_id);
        rctx.mark_success();
        return Status::OK;
      }
    }
    d_.lifecycle->release_lease(*lease_or);
    return to_grpc_status(upsert_or.status());
  }

  {
    absl::MutexLock lock(&contribution_keepalive_tracker_->mu);
    contribution_keepalive_tracker_->stop_flags[contribution.lease_id] = contribution_stop;
  }
  auto tracker = contribution_keepalive_tracker_;
  auto executor = d_.async_runtime.blocking_executor();
  const std::string lease_id = contribution.lease_id;
  const uint64_t lease_generation = contribution.lease_generation;
  auto keepalive = std::make_shared<std::function<void()>>();
  std::weak_ptr<std::function<void()>> keepalive_weak = keepalive;
  *keepalive = [tracker,
                contribution_client,
                contribution_stop,
                executor,
                keepalive_weak,
                &timekeeper = d_.async_runtime.timekeeper(),
                attempt_id,
                slot_id,
                lease_id,
                lease_generation]() mutable {
    if (contribution_stop->load(std::memory_order_relaxed)) {
      absl::MutexLock lock(&tracker->mu);
      auto it = tracker->stop_flags.find(lease_id);
      if (it != tracker->stop_flags.end() && it->second == contribution_stop) {
        tracker->stop_flags.erase(it);
      }
      return;
    }
    timekeeper
        .after(std::chrono::milliseconds(absl::ToInt64Milliseconds(coordination::kContributionLeaseRefreshInterval)))
        .via(executor)
        .thenValue([tracker,
                    contribution_client,
                    contribution_stop,
                    keepalive_weak,
                    attempt_id,
                    slot_id,
                    lease_id,
                    lease_generation](folly::Unit) mutable {
          if (contribution_stop->load(std::memory_order_relaxed)) {
            absl::MutexLock lock(&tracker->mu);
            auto it = tracker->stop_flags.find(lease_id);
            if (it != tracker->stop_flags.end() && it->second == contribution_stop) {
              tracker->stop_flags.erase(it);
            }
            return;
          }
          auto refresh_or = contribution_client->update_assembly_slot_occupancy_state(
              attempt_id,
              slot_id,
              "accepted",
              lease_id,
              lease_generation,
              absl::Now() + coordination::kContributionLeaseTtl,
              {"accepted"});
          if (!refresh_or.ok()) {
            if (!absl::IsNotFound(refresh_or.status())) {
              LOG(WARNING) << "assembly slot occupancy keepalive failed for attempt=" << attempt_id
                           << " slot_id=" << slot_id << ": " << refresh_or.status();
              auto next = keepalive_weak.lock();
              if (next != nullptr) {
                (*next)();
              }
              return;
            }
            contribution_stop->store(true, std::memory_order_relaxed);
          }
          auto next = keepalive_weak.lock();
          if (next != nullptr) {
            (*next)();
          }
        });
  };
  (*keepalive)();

  if (existing_or.ok() && existing_or->contributor_daemon_id == contribution.contributor_daemon_id &&
      existing_or->lease_id != contribution.lease_id) {
    uint64_t old_lease_id = 0;
    if (absl::SimpleAtoi(existing_or->lease_id, &old_lease_id)) {
      d_.lifecycle->release_lease(old_lease_id);
    }
  }

  resp.set_accepted(true);
  resp.set_already_exists(false);
  resp.set_lease_id(upsert_or->lease_id);
  resp.set_lease_generation(upsert_or->lease_generation);
  resp.set_state(upsert_or->state);
  resp.set_slot_id(upsert_or->slot_id);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status OwnedBindingService::freeze_binding_current_value(
    RpcContext& rctx,
    const v2::FreezeBindingCurrentValueRequest& req,
    v2::FreezeBindingCurrentValueResponse& resp) {
  if (req.binding_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_id is required"};
  }
  if (req.update_epoch().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "update_epoch is required"};
  }
  auto record_or = d_.bindings.get(req.binding_id());
  if (!record_or.ok()) {
    return to_grpc_status(record_or.status());
  }
  const auto record = *record_or;

  int device_id = -1;
  v2::TargetLayout target_layout;
  std::string source_artifact_ref;
  common::memory::GpuDeviceMemory* allocation = nullptr;
  cuda::IpcHandleBytes handle_bytes;
  {
    absl::MutexLock lock(&record->mu);
    if (record->closed) {
      return {StatusCode::FAILED_PRECONDITION, "binding is closed"};
    }
    if (record->state != v2::BINDING_STATE_MUTABLE) {
      return {StatusCode::FAILED_PRECONDITION, "binding is not mutable"};
    }
    if (record->active_update_epoch != req.update_epoch()) {
      mark_dirty(record.get());
      return {StatusCode::FAILED_PRECONDITION, "update_epoch mismatch"};
    }
    device_id = record->device_id;
    target_layout = record->target_layout;
    source_artifact_ref =
        req.source_artifact_ref().empty() ? record->source_selection.artifact_id() : req.source_artifact_ref();
    allocation = record->allocation.get();
    handle_bytes = record->handle_bytes;
  }
  if (allocation == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "binding allocation is unavailable"};
  }
  auto storage_layout_or =
      build_owned_storage_layout(target_layout, device_id, gsl::not_null<void*>{allocation->get()}, handle_bytes);
  if (!storage_layout_or.ok()) {
    return to_grpc_status(storage_layout_or.status());
  }
  auto aliases_or = build_binding_tensor_aliases(*record);
  if (!aliases_or.ok()) {
    return to_grpc_status(aliases_or.status());
  }

  {
    absl::MutexLock lock(&record->mu);
    if (record->closed) {
      return {StatusCode::FAILED_PRECONDITION, "binding is closed"};
    }
    if (record->state != v2::BINDING_STATE_MUTABLE) {
      return {StatusCode::FAILED_PRECONDITION, "binding is not mutable"};
    }
    if (record->active_update_epoch != req.update_epoch()) {
      mark_dirty(record.get());
      return {StatusCode::FAILED_PRECONDITION, "update_epoch mismatch"};
    }
    if (auto guard_status = ensure_binding_current_value_not_published(*record); !guard_status.ok()) {
      return to_grpc_status(guard_status);
    }
    mark_ready_local(record.get());
    set_local_ready_verification_metadata(
        record.get(), v2::BINDING_VALUE_VERIFICATION_STATE_PENDING, source_artifact_ref);
    resp.set_state(record->state);
    fill_binding_value(*record, *resp.mutable_current_value());
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status OwnedBindingService::seal_binding(
    RpcContext& rctx,
    const v2::SealBindingRequest& req,
    v2::SealBindingResponse& resp) {
  if (req.binding_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_id is required"};
  }
  if (req.update_epoch().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "update_epoch is required"};
  }
  auto record_or = d_.bindings.get(req.binding_id());
  if (!record_or.ok()) {
    return to_grpc_status(record_or.status());
  }
  const auto record = *record_or;
  if (d_.lip_manager == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "LipManager is unavailable"};
  }
  int device_id = -1;
  int owner_pid = 0;
  v2::TargetLayout target_layout;
  std::string target_index_json;
  std::string source_artifact_id;
  common::memory::GpuDeviceMemory* allocation = nullptr;
  cuda::IpcHandleBytes handle_bytes;
  {
    absl::MutexLock lock(&record->mu);
    if (record->closed) {
      return {StatusCode::FAILED_PRECONDITION, "binding is closed"};
    }
    if (record->state != v2::BINDING_STATE_MUTABLE) {
      return {StatusCode::FAILED_PRECONDITION, "binding is not mutable"};
    }
    if (record->active_update_epoch != req.update_epoch()) {
      mark_dirty(record.get());
      return {StatusCode::FAILED_PRECONDITION, "update_epoch mismatch"};
    }
    device_id = record->device_id;
    owner_pid = record->owner_pid;
    target_layout = record->target_layout;
    target_index_json = record->target_index_json;
    source_artifact_id = record->source_selection.artifact_id();
    allocation = record->allocation.get();
    handle_bytes = record->handle_bytes;
  }
  if (allocation == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "binding allocation is unavailable"};
  }
  auto storage_layout_or =
      build_owned_storage_layout(target_layout, device_id, gsl::not_null<void*>{allocation->get()}, handle_bytes);
  if (!storage_layout_or.ok()) {
    return to_grpc_status(storage_layout_or.status());
  }
  auto aliases_or = build_binding_tensor_aliases(*record);
  if (!aliases_or.ok()) {
    return to_grpc_status(aliases_or.status());
  }
  auto sealed_commit_or = d_.lip_manager->build_commit_lease_result(
      device_id,
      owner_pid,
      storage_layout_or->total_size,
      tensorcast::common::ArtifactIdKind::kMi2,
      /*client_artifact_id=*/"",
      target_index_json,
      /*index_key_hex=*/"",
      absl::MakeSpan(storage_layout_or->publish_segments),
      absl::MakeSpan(storage_layout_or->publish_storages),
      absl::MakeSpan(*aliases_or),
      std::nullopt,
      LipManager::BuildCommitLeaseOptions{
          .direct_gpu_hash_ptr = gsl::not_null<void*>{allocation->get()},
          .require_gpu_identity_hash = true,
      });
  if (!sealed_commit_or.ok()) {
    return to_grpc_status(sealed_commit_or.status());
  }
  if (common::is_msa1_artifact_id(source_artifact_id)) {
    const bool promotion_noted =
        d_.disk_imports.note_promoted_artifact_for_binding(source_artifact_id, sealed_commit_or->artifact_id);
    (void)promotion_noted;
  }
  {
    absl::MutexLock lock(&record->mu);
    if (record->closed) {
      return {StatusCode::FAILED_PRECONDITION, "binding is closed"};
    }
    if (record->state != v2::BINDING_STATE_MUTABLE) {
      return {StatusCode::FAILED_PRECONDITION, "binding is not mutable"};
    }
    if (record->active_update_epoch != req.update_epoch()) {
      mark_dirty(record.get());
      return {StatusCode::FAILED_PRECONDITION, "update_epoch mismatch"};
    }
    if (auto guard_status = ensure_binding_current_value_not_published(*record); !guard_status.ok()) {
      return to_grpc_status(guard_status);
    }
    mark_ready_local(record.get());
    record->sealed_commit_result = *sealed_commit_or;
    set_local_ready_verification_metadata(
        record.get(), v2::BINDING_VALUE_VERIFICATION_STATE_VERIFIED, source_artifact_id, sealed_commit_or->artifact_id);
    resp.set_state(record->state);
    fill_binding_value(*record, *resp.mutable_current_value());
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status OwnedBindingService::promote_binding_current_value_impl(
    const v2::PromoteBindingCurrentValueRequest& req,
    v2::PromoteBindingCurrentValueResponse& resp) {
  if (req.binding_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_id is required"};
  }
  if (req.binding_value_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_value_id is required"};
  }
  auto record_or = d_.bindings.get(req.binding_id());
  if (!record_or.ok()) {
    return to_grpc_status(record_or.status());
  }
  const auto record = *record_or;

  std::string current_binding_value_id;
  std::string current_artifact_id;
  std::string current_artifact_canonical_index_json;
  int device_id = -1;
  int owner_pid = 0;
  v2::TargetLayout target_layout;
  std::string target_index_json;
  cuda::IpcHandleBytes handle_bytes;
  common::memory::GpuDeviceMemory* allocation = nullptr;
  std::optional<CommitLeaseResult> sealed_commit_result;
  {
    absl::MutexLock lock(&record->mu);
    if (record->closed) {
      return {StatusCode::FAILED_PRECONDITION, "binding is closed"};
    }
    if (record->state == v2::BINDING_STATE_MUTABLE) {
      return {StatusCode::FAILED_PRECONDITION, "binding is mutable"};
    }
    if (record->state == v2::BINDING_STATE_DIRTY) {
      return {StatusCode::FAILED_PRECONDITION, "binding is dirty"};
    }
    if (record->current_binding_value_id.empty()) {
      return {StatusCode::FAILED_PRECONDITION, "binding has no current value"};
    }
    if (record->current_binding_value_id != req.binding_value_id()) {
      return {StatusCode::FAILED_PRECONDITION, "binding current value changed"};
    }
    current_binding_value_id = record->current_binding_value_id;
    current_artifact_id = record->current_artifact_id;
    current_artifact_canonical_index_json = record->current_artifact_canonical_index_json;
    device_id = record->device_id;
    owner_pid = record->owner_pid;
    target_layout = record->target_layout;
    target_index_json = record->target_index_json;
    handle_bytes = record->handle_bytes;
    allocation = record->allocation.get();
    sealed_commit_result = record->sealed_commit_result;
  }

  if (auto contribution_status =
          ensure_no_live_binding_contributions(d_.global_store_client, record->binding_id, current_binding_value_id);
      !contribution_status.ok()) {
    return to_grpc_status(contribution_status);
  }

  if (!current_artifact_id.empty() &&
      common::infer_artifact_id_kind(current_artifact_id) == tensorcast::common::ArtifactIdKind::kMi2) {
    auto desc_or = d_.engine.get_artifact_descriptor(current_artifact_id);
    if (!desc_or.ok()) {
      return to_grpc_status(desc_or.status());
    }
    std::string publication_token;
    {
      absl::MutexLock lock(&record->mu);
      if (record->current_binding_value_id != req.binding_value_id()) {
        return {StatusCode::FAILED_PRECONDITION, "binding current value changed during promotion"};
      }
      record->verification_state = v2::BINDING_VALUE_VERIFICATION_STATE_VERIFIED;
      record->serving_artifact_id = current_artifact_id;
      record->verification_failure_reason.clear();
      publication_token = record->binding_current_value_publication_token;
      resp.set_state(record->state);
      fill_binding_value(*record, *resp.mutable_current_value());
    }
    if (publication_token.empty() && allocation != nullptr && !current_artifact_canonical_index_json.empty()) {
      auto storage_layout_or =
          build_owned_storage_layout(target_layout, device_id, gsl::not_null<void*>{allocation->get()}, handle_bytes);
      if (!storage_layout_or.ok()) {
        return to_grpc_status(storage_layout_or.status());
      }
      auto token_or = maybe_mint_binding_current_value_publication_token(
          d_.capability_tokens,
          d_.identity,
          d_.target_materialization_service,
          record,
          d_.daemon_id,
          d_.daemon_session_id,
          std::string_view(),
          std::move(storage_layout_or->publish_segments),
          std::move(storage_layout_or->publish_storages));
      if (!token_or.ok()) {
        return to_grpc_status(token_or.status());
      }
      publication_token = *token_or;
    }
    if (!publication_token.empty()) {
      resp.set_binding_current_value_publication_token(publication_token);
    }
    resp.mutable_artifact_descriptor()->CopyFrom(*desc_or);
    resp.mutable_execution_diagnostics()->CopyFrom(
        build_binding_closeout_diagnostics(v2::IdentityMintStrategy::IDENTITY_MINT_STRATEGY_NOT_APPLICABLE));
    resp.set_existed(true);
    return Status::OK;
  }

  if (allocation == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "binding allocation is unavailable"};
  }
  if (d_.lip_manager == nullptr) {
    return {StatusCode::FAILED_PRECONDITION, "LipManager is unavailable"};
  }
  auto storage_layout_or =
      build_owned_storage_layout(target_layout, device_id, gsl::not_null<void*>{allocation->get()}, handle_bytes);
  if (!storage_layout_or.ok()) {
    return to_grpc_status(storage_layout_or.status());
  }
  std::vector<LeaseSegMeta> publication_segments = storage_layout_or->publish_segments;
  std::vector<RegisterStorageMeta> publication_storages = storage_layout_or->publish_storages;
  auto aliases_or = build_binding_tensor_aliases(*record);
  if (!aliases_or.ok()) {
    return to_grpc_status(aliases_or.status());
  }

  const std::string registration_id = absl::StrCat("binding-promote:", record->binding_id, ":", req.binding_value_id());
  const uint64_t epoch = static_cast<uint64_t>(absl::ToUnixMillis(absl::Now()));
  auto out_or = d_.lip_manager->commit_lease_in_place(
      registration_id,
      device_id,
      owner_pid,
      /*ttl_ms=*/0,
      epoch,
      storage_layout_or->total_size,
      tensorcast::common::ArtifactIdKind::kMi2,
      /*client_artifact_id=*/"",
      target_index_json,
      /*index_key_hex=*/"",
      std::move(storage_layout_or->publish_segments),
      std::move(storage_layout_or->publish_storages),
      std::move(*aliases_or),
      sealed_commit_result,
      LipManager::BuildCommitLeaseOptions{
          .direct_gpu_hash_ptr = gsl::not_null<void*>{allocation->get()},
          .require_gpu_identity_hash = true,
      });
  if (!out_or.ok()) {
    return to_grpc_status(out_or.status());
  }
  if (common::is_msa1_artifact_id(current_artifact_id)) {
    const bool promotion_noted =
        d_.disk_imports.note_promoted_artifact_for_binding(current_artifact_id, out_or->artifact_id);
    (void)promotion_noted;
  }

  const auto selection = build_mapped_bound_selection(out_or->artifact_id, target_layout, target_index_json);
  {
    absl::MutexLock lock(&record->mu);
    if (record->current_binding_value_id != req.binding_value_id()) {
      return {StatusCode::FAILED_PRECONDITION, "binding current value changed during promotion"};
    }
    if (auto guard_status = ensure_binding_current_value_not_published(*record); !guard_status.ok()) {
      return to_grpc_status(guard_status);
    }
    record->current_artifact_id = out_or->artifact_id;
    record->current_artifact_canonical_index_json = out_or->canonical_index_json;
    record->current_selection = selection;
    record->binding_current_value_publication_token.clear();
    record->state = v2::BINDING_STATE_READY_ARTIFACT;
    record->active_update_epoch.clear();
    record->sealed_commit_result = *out_or;
    record->verification_state = v2::BINDING_VALUE_VERIFICATION_STATE_VERIFIED;
    record->verification_failure_reason.clear();
    record->serving_artifact_id = out_or->artifact_id;
    resp.set_state(record->state);
    fill_binding_value(*record, *resp.mutable_current_value());
  }

  auto publication_token_or = maybe_mint_binding_current_value_publication_token(
      d_.capability_tokens,
      d_.identity,
      d_.target_materialization_service,
      record,
      d_.daemon_id,
      d_.daemon_session_id,
      std::string_view(),
      std::move(publication_segments),
      std::move(publication_storages));
  if (!publication_token_or.ok()) {
    return to_grpc_status(publication_token_or.status());
  }
  if (!publication_token_or->empty()) {
    resp.set_binding_current_value_publication_token(*publication_token_or);
  }

  if (d_.lifecycle != nullptr) {
    SessionLifecycleManager::CommitSubject subj{.artifact_id = out_or->artifact_id, .device_id = device_id};
    auto lid_or = d_.lifecycle->create_commit_lease(subj, owner_pid);
    if (!lid_or.ok()) {
      LOG(WARNING) << "create_commit_lease failed during binding promotion: artifact_id=" << out_or->artifact_id
                   << " dev=" << device_id << ": " << lid_or.status();
    }
  }

  populate_artifact_descriptor(*out_or, resp.mutable_artifact_descriptor());
  const CommitLeaseResult* hash_commit_result = sealed_commit_result.has_value() ? &*sealed_commit_result : &*out_or;
  resp.mutable_execution_diagnostics()->CopyFrom(build_binding_closeout_diagnostics(
      sealed_commit_result.has_value() ? v2::IdentityMintStrategy::IDENTITY_MINT_STRATEGY_SEAL_REUSE
                                       : v2::IdentityMintStrategy::IDENTITY_MINT_STRATEGY_CLOSEOUT_MINT,
      hash_commit_result));
  resp.set_existed(false);
  return Status::OK;
}

grpc::Status OwnedBindingService::promote_binding_current_value(
    RpcContext& rctx,
    const v2::PromoteBindingCurrentValueRequest& req,
    v2::PromoteBindingCurrentValueResponse& resp) {
  auto status = promote_binding_current_value_impl(req, resp);
  if (status.ok()) {
    rctx.mark_success();
  }
  return status;
}

void OwnedBindingService::run_async_promotion_job(std::string job_id, v2::PromoteBindingCurrentValueRequest req) {
  {
    absl::MutexLock lock(&promotion_jobs_mu_);
    const auto it = promotion_jobs_by_id_.find(job_id);
    if (it == promotion_jobs_by_id_.end()) {
      return;
    }
    if (it->second->status.state() == v2::BINDING_PROMOTION_JOB_STATE_CANCELED) {
      return;
    }
    it->second->status.set_state(v2::BINDING_PROMOTION_JOB_STATE_RUNNING);
  }

  v2::PromoteBindingCurrentValueResponse promote_resp;
  const grpc::Status status = promote_binding_current_value_impl(req, promote_resp);

  absl::MutexLock lock(&promotion_jobs_mu_);
  const auto it = promotion_jobs_by_id_.find(job_id);
  if (it == promotion_jobs_by_id_.end()) {
    return;
  }
  auto& job_status = it->second->status;
  if (job_status.state() == v2::BINDING_PROMOTION_JOB_STATE_CANCELED) {
    return;
  }
  if (!status.ok()) {
    job_status.set_state(v2::BINDING_PROMOTION_JOB_STATE_FAILED);
    job_status.set_failure_reason(status.error_message());
    auto record_or = d_.bindings.get(req.binding_id());
    if (record_or.ok()) {
      const auto record = *record_or;
      absl::MutexLock record_lock(&record->mu);
      if (record->current_binding_value_id == req.binding_value_id()) {
        record->verification_state = v2::BINDING_VALUE_VERIFICATION_STATE_FAILED;
        record->verification_job_id = job_id;
        record->verification_failure_reason = status.error_message();
      }
    }
    LOG(WARNING) << "async binding promotion failed binding_id=" << req.binding_id()
                 << " binding_value_id=" << req.binding_value_id() << ": " << status.error_message();
    return;
  }
  job_status.set_state(v2::BINDING_PROMOTION_JOB_STATE_SUCCEEDED);
  job_status.set_existed(promote_resp.existed());
  job_status.mutable_current_value()->CopyFrom(promote_resp.current_value());
  if (!promote_resp.binding_current_value_publication_token().empty()) {
    job_status.set_binding_current_value_publication_token(promote_resp.binding_current_value_publication_token());
  }
  if (promote_resp.has_artifact_descriptor()) {
    job_status.mutable_artifact_descriptor()->CopyFrom(promote_resp.artifact_descriptor());
    job_status.set_serving_artifact_id(promote_resp.artifact_descriptor().artifact_id());
  }
}

grpc::Status OwnedBindingService::start_promote_binding_current_value(
    RpcContext& rctx,
    const v2::StartPromoteBindingCurrentValueRequest& req,
    v2::StartPromoteBindingCurrentValueResponse& resp) {
  if (req.binding_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_id is required"};
  }
  if (req.binding_value_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_value_id is required"};
  }
  auto record_or = d_.bindings.get(req.binding_id());
  if (!record_or.ok()) {
    return to_grpc_status(record_or.status());
  }
  const auto record = *record_or;
  const std::string key = promotion_job_key(req.binding_id(), req.binding_value_id());
  {
    absl::MutexLock lock(&promotion_jobs_mu_);
    const auto id_it = promotion_job_ids_by_value_.find(key);
    if (id_it != promotion_job_ids_by_value_.end()) {
      const auto job_it = promotion_jobs_by_id_.find(id_it->second);
      if (job_it != promotion_jobs_by_id_.end()) {
        fill_promotion_status_from_job(job_it->second, *resp.mutable_status());
        rctx.mark_success();
        return Status::OK;
      }
      promotion_job_ids_by_value_.erase(id_it);
    }
  }

  std::string source_artifact_ref;
  std::string local_serving_ref;
  std::string binding_current_value_publication_token;
  v2::BindingValue current_value;
  {
    absl::MutexLock lock(&record->mu);
    if (record->closed) {
      return {StatusCode::FAILED_PRECONDITION, "binding is closed"};
    }
    if (record->state == v2::BINDING_STATE_MUTABLE) {
      return {StatusCode::FAILED_PRECONDITION, "binding is mutable"};
    }
    if (record->state == v2::BINDING_STATE_DIRTY) {
      return {StatusCode::FAILED_PRECONDITION, "binding is dirty"};
    }
    if (record->current_binding_value_id.empty()) {
      return {StatusCode::FAILED_PRECONDITION, "binding has no current value"};
    }
    if (record->current_binding_value_id != req.binding_value_id()) {
      return {StatusCode::FAILED_PRECONDITION, "binding current value changed"};
    }
    source_artifact_ref = record->source_artifact_ref;
    local_serving_ref = record->local_serving_ref;
    binding_current_value_publication_token = record->binding_current_value_publication_token;
    fill_binding_value(*record, current_value);
  }

  if (current_value.verification_state() == v2::BINDING_VALUE_VERIFICATION_STATE_VERIFIED &&
      current_value.has_serving_artifact_id()) {
    auto desc_or = d_.engine.get_artifact_descriptor(current_value.serving_artifact_id());
    if (!desc_or.ok()) {
      return to_grpc_status(desc_or.status());
    }
    auto* status = resp.mutable_status();
    const std::string job_id = absl::StrCat("bpj:", req.binding_id(), ":", req.binding_value_id(), ":verified");
    status->set_verification_job_id(job_id);
    status->set_binding_id(req.binding_id());
    status->set_binding_value_id(req.binding_value_id());
    status->set_state(v2::BINDING_PROMOTION_JOB_STATE_SUCCEEDED);
    status->mutable_current_value()->CopyFrom(current_value);
    status->set_serving_artifact_id(current_value.serving_artifact_id());
    status->mutable_artifact_descriptor()->CopyFrom(*desc_or);
    status->set_existed(true);
    if (!binding_current_value_publication_token.empty()) {
      status->set_binding_current_value_publication_token(binding_current_value_publication_token);
    }
    rctx.mark_success();
    return Status::OK;
  }

  const std::string job_id = absl::StrCat("bpj:", mint_random_id(12));
  auto job = std::make_shared<PromotionJobRecord>();
  job->status.set_verification_job_id(job_id);
  job->status.set_binding_id(req.binding_id());
  job->status.set_binding_value_id(req.binding_value_id());
  job->status.set_state(v2::BINDING_PROMOTION_JOB_STATE_PENDING);
  job->status.mutable_current_value()->CopyFrom(current_value);
  {
    absl::MutexLock lock(&promotion_jobs_mu_);
    promotion_jobs_by_id_.insert_or_assign(job_id, job);
    promotion_job_ids_by_value_.insert_or_assign(key, job_id);
  }
  {
    absl::MutexLock lock(&record->mu);
    if (record->current_binding_value_id == req.binding_value_id()) {
      record->verification_state = v2::BINDING_VALUE_VERIFICATION_STATE_PENDING;
      record->verification_job_id = job_id;
      record->source_artifact_ref = source_artifact_ref;
      record->local_serving_ref = local_serving_ref;
      record->verification_failure_reason.clear();
      fill_binding_value(*record, *job->status.mutable_current_value());
    }
  }
  v2::PromoteBindingCurrentValueRequest promote_req;
  promote_req.set_binding_id(req.binding_id());
  promote_req.set_binding_value_id(req.binding_value_id());
  d_.async_runtime.blocking_executor()->add(
      [this, job_id, promote_req]() mutable { run_async_promotion_job(job_id, std::move(promote_req)); });
  {
    absl::MutexLock lock(&promotion_jobs_mu_);
    fill_promotion_status_from_job(job, *resp.mutable_status());
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status OwnedBindingService::get_binding_promotion_status(
    RpcContext& rctx,
    const v2::GetBindingPromotionStatusRequest& req,
    v2::GetBindingPromotionStatusResponse& resp) {
  {
    absl::MutexLock lock(&promotion_jobs_mu_);
    if (!req.verification_job_id().empty()) {
      const auto it = promotion_jobs_by_id_.find(req.verification_job_id());
      if (it != promotion_jobs_by_id_.end()) {
        fill_promotion_status_from_job(it->second, *resp.mutable_status());
        rctx.mark_success();
        return Status::OK;
      }
    } else if (!req.binding_id().empty() && !req.binding_value_id().empty()) {
      const auto key = promotion_job_key(req.binding_id(), req.binding_value_id());
      const auto id_it = promotion_job_ids_by_value_.find(key);
      if (id_it != promotion_job_ids_by_value_.end()) {
        const auto job_it = promotion_jobs_by_id_.find(id_it->second);
        if (job_it != promotion_jobs_by_id_.end()) {
          fill_promotion_status_from_job(job_it->second, *resp.mutable_status());
          rctx.mark_success();
          return Status::OK;
        }
      }
    }
  }
  return {StatusCode::NOT_FOUND, "binding promotion job not found"};
}

grpc::Status OwnedBindingService::refill_owned_binding(
    RpcContext& rctx,
    const v2::RefillOwnedBindingRequest& req,
    v2::RefillOwnedBindingResponse& resp) {
  const auto profile_start = std::chrono::steady_clock::now();
  auto elapsed_sec = [](auto start, auto end) { return std::chrono::duration<double>(end - start).count(); };
  double prepare_plan_sec = 0.0;
  double storage_layout_sec = 0.0;
  double prepare_execution_sec = 0.0;
  double strict_preflight_sec = 0.0;
  double materialize_sec = 0.0;
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req.binding_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_id is required"};
  }
  if (req.artifact_id().empty() && !req.has_public_disk_source()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id or public_disk_source is required"};
  }

  auto record_or = d_.bindings.get(req.binding_id());
  if (!record_or.ok()) {
    return to_grpc_status(record_or.status());
  }
  const auto record = *record_or;

  int owner_pid = 0;
  int device_id = -1;
  std::string device_uuid;
  bool mapped = false;
  v2::TargetLayout target_layout;
  std::string target_index_json;
  tensorcast::common::v1::ArtifactSelection source_selection;
  v2::CopyPlan copy_plan;
  std::vector<v2::MappedTensorSpec> dst_tensors;
  std::string current_binding_value_id;
  {
    absl::MutexLock lock(&record->mu);
    if (record->closed) {
      return {StatusCode::FAILED_PRECONDITION, "binding is closed"};
    }
    if (record->state == v2::BINDING_STATE_MUTABLE) {
      return {StatusCode::FAILED_PRECONDITION, "binding is mutable"};
    }
    if (auto guard_status = ensure_binding_current_value_not_published(*record); !guard_status.ok()) {
      return to_grpc_status(guard_status);
    }
    owner_pid = record->owner_pid;
    device_id = record->device_id;
    device_uuid = record->device_uuid;
    mapped = record->mapped;
    target_layout = record->target_layout;
    target_index_json = record->target_index_json;
    if (req.has_source_selection()) {
      source_selection = req.source_selection();
    } else {
      source_selection = record->source_selection;
    }
    copy_plan = record->copy_plan;
    dst_tensors = record->dst_tensors;
    current_binding_value_id = record->current_binding_value_id;
  }
  source_selection.set_artifact_id(req.artifact_id());
  if (auto contribution_status =
          ensure_no_live_binding_contributions(d_.global_store_client, record->binding_id, current_binding_value_id);
      !contribution_status.ok()) {
    return to_grpc_status(contribution_status);
  }

  const auto device = d_.devices.From(v2::DeviceType::DEVICE_TYPE_GPU, device_uuid, std::nullopt);
  const bool use_realization_plan = req.has_realization_plan() && req.realization_plan().entries_size() > 0;
  const bool use_mapped_materialization = mapped || use_realization_plan;
  PreparedSourceBoundPlan prepared_plan;
  const auto prepare_plan_start = std::chrono::steady_clock::now();
  auto prepare_status = prepare_source_bound_plan(
      d_,
      rctx,
      device,
      SourceBoundMaterializationRequest{
          .mapped = use_mapped_materialization,
          .owner_pid = owner_pid,
          .device_uuid = device_uuid,
          .source_selection =
              (req.has_public_disk_source() && !req.has_source_selection() ? nullptr : &source_selection),
          .public_disk_source = req.has_public_disk_source() ? &req.public_disk_source() : nullptr,
          .target_layout = &target_layout,
          .target_index_json = target_index_json,
          .realization_plan = use_realization_plan ? &req.realization_plan() : nullptr,
          .copy_plan = (mapped && !use_realization_plan) ? &copy_plan : nullptr,
          .dst_tensors = use_mapped_materialization ? &dst_tensors : nullptr,
          .source_policy = req.has_source_policy() ? &req.source_policy() : nullptr,
          .execution_topology = req.has_execution_topology() ? &req.execution_topology() : nullptr,
          .collective_policy = req.collective_policy(),
          .operation_id = req.has_operation_id() ? std::optional<std::string_view>(req.operation_id()) : std::nullopt,
          .group_realization = req.has_group_realization() ? &req.group_realization() : nullptr,
          .placement = req.placement(),
      },
      prepared_plan);
  if (!prepare_status.ok()) {
    return prepare_status;
  }
  const auto prepare_plan_done = std::chrono::steady_clock::now();
  prepare_plan_sec = elapsed_sec(prepare_plan_start, prepare_plan_done);

  OwnedStorageLayout storage_layout;
  const auto storage_layout_start = std::chrono::steady_clock::now();
  {
    absl::MutexLock lock(&record->mu);
    auto storage_layout_or = build_owned_storage_layout(
        target_layout, device_id, gsl::not_null<void*>{record->allocation->get()}, record->handle_bytes);
    if (!storage_layout_or.ok()) {
      return to_grpc_status(storage_layout_or.status());
    }
    storage_layout = std::move(*storage_layout_or);
  }
  const auto storage_layout_done = std::chrono::steady_clock::now();
  storage_layout_sec = elapsed_sec(storage_layout_start, storage_layout_done);

  PreparedSourceBoundExecution prepared_execution;
  const auto prepare_execution_start = std::chrono::steady_clock::now();
  auto execution_status =
      prepare_source_bound_execution(d_, rctx, target_layout, storage_layout, prepared_plan, prepared_execution);
  if (!execution_status.ok()) {
    return execution_status;
  }
  const auto prepare_execution_done = std::chrono::steady_clock::now();
  prepare_execution_sec = elapsed_sec(prepare_execution_start, prepare_execution_done);
  const auto strict_preflight_start = std::chrono::steady_clock::now();
  if (auto strict_preflight_status = evaluate_strict_collective_preflight_for_testing(
          &rctx,
          prepared_execution.prepared_execution_plan.has_value() &&
                  prepared_execution.prepared_execution_plan->strategy_plan.has_value()
              ? &prepared_execution.prepared_execution_plan->strategy_plan->summary
              : nullptr,
          prepared_plan.collective_policy);
      !strict_preflight_status.ok()) {
    return strict_preflight_status;
  }
  const auto strict_preflight_done = std::chrono::steady_clock::now();
  strict_preflight_sec = elapsed_sec(strict_preflight_start, strict_preflight_done);
  const bool effective_mapped_materialization = prepared_plan.mapped_plan.has_value();

  const auto materialize_start = std::chrono::steady_clock::now();
  absl::StatusOr<store::loading::MaterializeIntoTargetResult> result_or = effective_mapped_materialization
      ? d_.engine.materialize_mapped_into_target(
            device, *prepared_execution.prepared_execution_plan, prepared_execution.hints, prepared_plan.disk_source)
      : d_.engine.materialize_into_target(
            device,
            storage_layout.into_target,
            prepared_plan.canonical_index_json,
            materialization_payload::compute_generation_from_index(prepared_plan.canonical_index_json),
            prepared_execution.hints,
            prepared_plan.disk_source);
  const auto materialize_done = std::chrono::steady_clock::now();
  materialize_sec = elapsed_sec(materialize_start, materialize_done);
  if (!result_or.ok()) {
    if (prepared_plan.collective_policy == v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE &&
        absl::IsFailedPrecondition(result_or.status()) && !is_collective_execution_failure(result_or.status())) {
      return make_collective_failure_status(
          &rctx,
          StatusCode::FAILED_PRECONDITION,
          result_or.status().message(),
          v2::CollectiveFailureClass::COLLECTIVE_FAILURE_CLASS_NOT_ELIGIBLE);
    }
    {
      absl::MutexLock lock(&record->mu);
      mark_dirty(record.get());
    }
    if (collective_policy_requests_collective(prepared_plan.collective_policy) &&
        is_collective_execution_failure(result_or.status())) {
      return make_collective_failure_status(
          &rctx,
          StatusCode::FAILED_PRECONDITION,
          result_or.status().message(),
          v2::CollectiveFailureClass::COLLECTIVE_FAILURE_CLASS_EXECUTION_FAILED);
    }
    return to_grpc_status(result_or.status());
  }
  const auto execution_diagnostics = build_execution_diagnostics(
      &*result_or,
      prepared_plan.collective_policy,
      prepared_plan.request_context.execution_topology,
      HashExecutionDetails{});
  const auto source_bound_plan_diagnostics =
      build_source_bound_plan_diagnostics(prepared_execution.prepared_execution_plan);
  if (collective_policy_requests_collective(prepared_plan.collective_policy) &&
      execution_diagnostics.collective_requested() && !execution_diagnostics.collective_used() &&
      prepared_plan.collective_policy == v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE) {
    {
      absl::MutexLock lock(&record->mu);
      mark_dirty(record.get());
    }
    return make_collective_failure_status(
        &rctx,
        StatusCode::FAILED_PRECONDITION,
        "collective requested but the source-bound path was not eligible",
        v2::CollectiveFailureClass::COLLECTIVE_FAILURE_CLASS_NOT_ELIGIBLE);
  }
  if (prepared_plan.execution_only_mutable) {
    std::string update_epoch;
    cancel_promotion_jobs_for_value(
        record->binding_id, current_binding_value_id, "binding refill entered mutable execution-only state");
    {
      absl::MutexLock lock(&record->mu);
      if (record->closed) {
        mark_dirty(record.get());
        return {StatusCode::FAILED_PRECONDITION, "binding is closed"};
      }
      if (!current_binding_value_id.empty() && record->current_binding_value_id != current_binding_value_id) {
        return {StatusCode::FAILED_PRECONDITION, "binding current value changed during refill"};
      }
      if (auto guard_status = ensure_binding_current_value_not_published(*record); !guard_status.ok()) {
        return to_grpc_status(guard_status);
      }
      update_epoch = next_update_epoch(record.get());
      mark_mutable(record.get(), update_epoch);
    }
    resp.set_artifact_id(prepared_plan.resolved_artifact_id);
    resp.set_source(to_proto_source(result_or->source));
    if (!result_or->selected_source_replica_id.empty()) {
      resp.set_selected_source_replica_id(result_or->selected_source_replica_id);
    }
    if (!result_or->selected_source_transport_id.empty()) {
      resp.set_selected_source_transport_id(result_or->selected_source_transport_id);
    }
    resp.set_state(v2::BINDING_STATE_MUTABLE);
    resp.set_update_epoch(update_epoch);
    resp.mutable_execution_diagnostics()->CopyFrom(execution_diagnostics);
    resp.mutable_source_bound_plan_diagnostics()->CopyFrom(source_bound_plan_diagnostics);
    const auto done = std::chrono::steady_clock::now();
    LOG(INFO) << "tc_profile refill_owned_binding timings"
              << " binding_id=" << req.binding_id() << " artifact_id=" << prepared_plan.resolved_artifact_id
              << " target_device=" << device.ordinal << " mapped=" << mapped
              << " effective_mapped=" << effective_mapped_materialization
              << " realization_plan=" << use_realization_plan << " execution_only_mutable=1"
              << " prepare_plan_sec=" << prepare_plan_sec << " storage_layout_sec=" << storage_layout_sec
              << " prepare_execution_sec=" << prepare_execution_sec << " strict_preflight_sec=" << strict_preflight_sec
              << " materialize_sec=" << materialize_sec
              << " post_materialize_sec=" << elapsed_sec(materialize_done, done)
              << " total_sec=" << elapsed_sec(profile_start, done);
    rctx.mark_success();
    return Status::OK;
  }
  auto preflight_or = serving_artifact_manifest::preflight_serving_artifact(
      &d_.engine,
      serving_artifact_manifest::build_preflight_request(
          prepared_plan.resolved_artifact_id,
          prepared_plan.canonical_index_json,
          prepared_plan.disk_source,
          prepared_plan.disk_metadata,
          req.has_serving_artifact_policy() ? &req.serving_artifact_policy() : nullptr));
  if (!preflight_or.ok()) {
    {
      absl::MutexLock lock(&record->mu);
      mark_dirty(record.get());
    }
    return to_grpc_status(preflight_or.status());
  }

  {
    absl::MutexLock lock(&record->mu);
    if (!current_binding_value_id.empty() && record->current_binding_value_id != current_binding_value_id) {
      return {StatusCode::FAILED_PRECONDITION, "binding current value changed during refill"};
    }
    if (auto guard_status = ensure_binding_current_value_not_published(*record); !guard_status.ok()) {
      return to_grpc_status(guard_status);
    }
    mark_ready_artifact(
        record.get(),
        prepared_plan.resolved_artifact_id,
        prepared_plan.current_selection,
        prepared_plan.canonical_index_json);
  }

  auto binding_current_value_publication_token_or = maybe_mint_binding_current_value_publication_token(
      d_.capability_tokens,
      d_.identity,
      d_.target_materialization_service,
      record,
      d_.daemon_id,
      d_.daemon_session_id,
      req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view(),
      std::move(storage_layout.publish_segments),
      std::move(storage_layout.publish_storages));
  if (!binding_current_value_publication_token_or.ok()) {
    return to_grpc_status(binding_current_value_publication_token_or.status());
  }

  resp.set_artifact_id(prepared_plan.resolved_artifact_id);
  resp.set_source(to_proto_source(result_or->source));
  if (!result_or->selected_source_replica_id.empty()) {
    resp.set_selected_source_replica_id(result_or->selected_source_replica_id);
  }
  if (!result_or->selected_source_transport_id.empty()) {
    resp.set_selected_source_transport_id(result_or->selected_source_transport_id);
  }
  if (!binding_current_value_publication_token_or->empty()) {
    resp.set_binding_current_value_publication_token(*binding_current_value_publication_token_or);
  }
  resp.mutable_resolved_selection()->CopyFrom(prepared_plan.current_selection);
  resp.set_state(v2::BINDING_STATE_READY_ARTIFACT);
  fill_binding_value(*record, *resp.mutable_current_value());
  resp.mutable_execution_diagnostics()->CopyFrom(execution_diagnostics);
  resp.mutable_source_bound_plan_diagnostics()->CopyFrom(source_bound_plan_diagnostics);
  const auto done = std::chrono::steady_clock::now();
  LOG(INFO) << "tc_profile refill_owned_binding timings"
            << " binding_id=" << req.binding_id() << " artifact_id=" << prepared_plan.resolved_artifact_id
            << " target_device=" << device.ordinal << " mapped=" << mapped
            << " effective_mapped=" << effective_mapped_materialization << " realization_plan=" << use_realization_plan
            << " execution_only_mutable=0"
            << " prepare_plan_sec=" << prepare_plan_sec << " storage_layout_sec=" << storage_layout_sec
            << " prepare_execution_sec=" << prepare_execution_sec << " strict_preflight_sec=" << strict_preflight_sec
            << " materialize_sec=" << materialize_sec << " post_materialize_sec=" << elapsed_sec(materialize_done, done)
            << " total_sec=" << elapsed_sec(profile_start, done);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status OwnedBindingService::close_owned_binding(
    RpcContext& rctx,
    const v2::CloseOwnedBindingRequest& req,
    v2::CloseOwnedBindingResponse& resp) {
  if (req.binding_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_id is required"};
  }
  resp.set_closed(d_.bindings.close_control(req.binding_id()));
  rctx.mark_success();
  return Status::OK;
}

} // namespace tensorcast::daemon
