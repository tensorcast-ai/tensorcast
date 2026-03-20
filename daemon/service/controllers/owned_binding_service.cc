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
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "daemon/service/controllers/assembly_coordination_utils.h"
#include "daemon/service/controllers/materialization_index_source_utils.h"
#include "daemon/service/controllers/materialization_layout_utils.h"
#include "daemon/service/controllers/materialization_payload_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/service/controllers/materialization_request_common_utils.h"
#include "daemon/service/controllers/materialization_target_plan_utils.h"
#include "daemon/service/controllers/registration_controller.h"
#include "daemon/service/controllers/registration_storage_mapping_utils.h"
#include "daemon/util/grpc_peer_utils.h"
#include "daemon/util/status_utils.h"

#include "core/common/artifact_hash.h"
#include "core/common/selection_identity.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_ipc.h"
#include "core/cuda/device_guard.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/view_utils.h"
#include "folly/futures/Future.h"
#include "gsl/pointers"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

using materialization_index_source::load_canonical_index_with_disk_fallback;
using materialization_index_source::load_descriptor_metadata;
using materialization_layout::resolve_target_offsets;
using materialization_payload::build_descriptors_from_index;
using materialization_policy::build_view_spec_proto;
using materialization_policy::compute_view_id_from_spec;
using materialization_policy::resolve_source_policy;
using materialization_policy::resolve_transform_placement;
using materialization_policy::to_hint_preference;
using materialization_policy::validate_source_policy;
using materialization_request_common::resolve_artifact_and_disk_source;
using materialization_target_plan::build_mapped_target_materialization_plan;
using materialization_target_plan::build_target_materialization_plan;
using materialization_target_plan::MappedTargetMaterializationPlan;
using materialization_target_plan::TargetMaterializationPlan;
using store::loading::MaterializationSource;
namespace coordination = assembly_coordination;

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

absl::StatusOr<std::optional<store::loading::DiskMetadata>> build_binding_disk_metadata(
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

absl::StatusOr<std::string> maybe_mint_target_publication_token(
    common::CapabilityTokenManager* capability_tokens,
    WorkerIdentityStore& identity,
    TargetMaterializationService* target_materialization_service,
    const v2::TargetLayout& layout,
    const tensorcast::common::v1::ArtifactSelection& current_selection,
    std::string_view canonical_index_json,
    int device_id,
    std::string_view device_uuid,
    int owner_pid,
    std::string_view operation_id,
    std::vector<LeaseSegMeta> publish_segments,
    std::vector<RegisterStorageMeta> publish_storages) {
  if (capability_tokens == nullptr || !capability_tokens->configured()) {
    return std::string();
  }
  if (target_materialization_service == nullptr) {
    return absl::FailedPreconditionError("target materialization service is unavailable");
  }

  const std::string layout_hash = compute_target_layout_hash(layout);
  const std::string publication_id = mint_publication_id();
  const auto byte_space = byte_space_from_selection(current_selection);
  const absl::Time expires_at = absl::Now() + TargetPublishService::target_publication_token_ttl();

  auto stable_index_or = store::loader::rebuild_stable_canonical_index(std::string(canonical_index_json), device_id);
  if (!stable_index_or.ok()) {
    return stable_index_or.status();
  }
  std::string stable_index_json = std::move(*stable_index_or);
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(stable_index_json.data()), stable_index_json.size()));
  std::string index_key_hex =
      absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));

  tensorcast::common::v1::TargetPublicationScope scope;
  scope.set_publication_id(publication_id);
  scope.mutable_selection()->CopyFrom(current_selection);
  scope.mutable_byte_space()->CopyFrom(byte_space);
  scope.set_device_uuid(std::string(device_uuid));
  scope.set_owner_pid(owner_pid);
  scope.set_target_layout_hash(layout_hash);
  if (!operation_id.empty()) {
    scope.set_operation_id(std::string(operation_id));
  }

  auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  if (!scope_or.ok()) {
    return scope_or.status();
  }
  const uint64_t expires_at_ms = static_cast<uint64_t>(absl::ToUnixMillis(expires_at));
  auto token_or = capability_tokens->mint(
      identity.daemon_id(), tensorcast::common::v1::CAPABILITY_AUDIENCE_TARGET_PUBLICATION, *scope_or, expires_at_ms);
  if (!token_or.ok()) {
    return token_or.status();
  }

  TargetPublicationRegistry::Record record;
  record.publication_id = PublicationInstanceId{.value = publication_id};
  record.publication_subject_key =
      build_publication_subject_key(current_selection, byte_space, layout_hash, device_uuid);
  record.target_layout_hash = layout_hash;
  record.selection = current_selection;
  record.byte_space = byte_space;
  record.canonical_index_json = std::move(stable_index_json);
  record.index_key_hex = std::move(index_key_hex);
  record.device_uuid = std::string(device_uuid);
  record.owner_pid = owner_pid;
  record.request_operation_id = std::string(operation_id);
  record.expires_at = expires_at;
  record.segments = std::move(publish_segments);
  record.storages = std::move(publish_storages);
  auto remembered_or = target_materialization_service->remember_target_publication(std::move(record));
  if (!remembered_or.ok()) {
    return remembered_or.status();
  }
  return *token_or;
}

std::string mint_binding_value_id() {
  return mint_random_id(16);
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
}

void mark_ready_artifact(
    BindingRegistry::Record* record,
    std::string_view artifact_id,
    const tensorcast::common::v1::ArtifactSelection& selection,
    std::string_view target_publication_token) {
  record->current_artifact_id = std::string(artifact_id);
  record->current_selection = selection;
  record->target_publication_token = std::string(target_publication_token);
  record->state = v2::BINDING_STATE_READY_ARTIFACT;
  record->active_update_epoch.clear();
  record->current_binding_value_id = mint_binding_value_id();
  record->seal_generation += 1;
}

void mark_allocated(BindingRegistry::Record* record) {
  record->current_artifact_id.clear();
  record->current_selection.Clear();
  record->target_publication_token.clear();
  record->current_binding_value_id.clear();
  record->state = v2::BINDING_STATE_ALLOCATED;
  record->active_update_epoch.clear();
}

void mark_mutable(BindingRegistry::Record* record, std::string_view update_epoch) {
  record->current_artifact_id.clear();
  record->current_selection.Clear();
  record->target_publication_token.clear();
  record->current_binding_value_id.clear();
  record->state = v2::BINDING_STATE_MUTABLE;
  record->active_update_epoch = std::string(update_epoch);
}

void mark_ready_local(BindingRegistry::Record* record) {
  record->current_artifact_id.clear();
  record->current_selection.Clear();
  record->target_publication_token.clear();
  record->state = v2::BINDING_STATE_READY_LOCAL;
  record->active_update_epoch.clear();
  record->current_binding_value_id = mint_binding_value_id();
  record->seal_generation += 1;
}

void mark_dirty(BindingRegistry::Record* record) {
  record->current_artifact_id.clear();
  record->current_selection.Clear();
  record->target_publication_token.clear();
  record->current_binding_value_id.clear();
  record->state = v2::BINDING_STATE_DIRTY;
  record->active_update_epoch.clear();
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
    v2::SourcePreference preference,
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
  request.set_preference(preference);
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
    v2::SourcePreference preference,
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
  request.set_preference(preference);
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

} // namespace

OwnedBindingService::OwnedBindingService(Dep d)
    : d_(std::move(d)), contribution_keepalive_tracker_(std::make_shared<ContributionLeaseKeepaliveTracker>()) {}

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
  if (mapped && (req.copy_plan().entries_size() == 0 || req.dst_tensors_size() == 0)) {
    return {StatusCode::INVALID_ARGUMENT, "mapped binding requires copy_plan and dst_tensors"};
  }
  if (!mapped && (req.copy_plan().entries_size() > 0 || req.dst_tensors_size() > 0)) {
    return {StatusCode::INVALID_ARGUMENT, "unmapped binding must not include copy_plan or dst_tensors"};
  }
  if (req.has_initial_selection() && req.initial_selection().artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "initial_selection.artifact_id is required when provided"};
  }
  if (!req.has_initial_selection() && req.has_source_artifact_id()) {
    return {StatusCode::INVALID_ARGUMENT, "source_artifact_id requires initial_selection"};
  }
  if (!req.has_initial_selection() && !req.target_publication_token().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "target_publication_token requires initial_selection"};
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
  if (mapped) {
    record->copy_plan = req.copy_plan();
    record->dst_tensors.assign(req.dst_tensors().begin(), req.dst_tensors().end());
  }
  if (req.has_initial_selection()) {
    record->source_selection = req.initial_selection();
    mark_ready_artifact(
        record.get(),
        req.has_source_artifact_id() ? req.source_artifact_id() : req.initial_selection().artifact_id(),
        req.initial_selection(),
        std::string_view(req.target_publication_token().data(), req.target_publication_token().size()));
  } else {
    mark_allocated(record.get());
  }

  if (auto insert_status = d_.bindings.insert(record); !insert_status.ok()) {
    return to_grpc_status(insert_status);
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
  if (mapped && (req.copy_plan().entries_size() == 0 || req.dst_tensors_size() == 0)) {
    return {StatusCode::INVALID_ARGUMENT, "mapped owner binding requires copy_plan and dst_tensors"};
  }
  if (!mapped && (req.copy_plan().entries_size() > 0 || req.dst_tensors_size() > 0)) {
    return {StatusCode::INVALID_ARGUMENT, "unmapped owner binding must not include copy_plan or dst_tensors"};
  }

  const auto device = d_.devices.From(v2::DeviceType::DEVICE_TYPE_GPU, req.device_uuid(), std::nullopt);
  if (device.type != DeviceType::GPU || device.ordinal < 0) {
    return {StatusCode::INVALID_ARGUMENT, "bind() requires a CUDA device"};
  }

  auto effective_policy =
      resolve_source_policy(req.has_source_policy() ? &req.source_policy() : nullptr, req.preference());
  if (auto policy_status = validate_source_policy(effective_policy); !policy_status.ok()) {
    return to_grpc_status(policy_status);
  }

  const bool loopback_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  auto resolution_or = resolve_artifact_and_disk_source(
      d_.global_store_client,
      &d_.disk_imports,
      d_.storage_path,
      req.source_selection().artifact_id(),
      effective_policy.allow_disk,
      /*allow_local_import_fallback=*/true,
      loopback_peer);
  if (!resolution_or.ok()) {
    return to_grpc_status(resolution_or.status());
  }
  auto resolution = std::move(*resolution_or);

  auto canonical_json_or = load_canonical_index_with_disk_fallback(
      d_.engine,
      resolution.resolved_artifact_id,
      resolution.normalized_disk_path,
      device.ordinal,
      resolution.gs_connected);
  if (!canonical_json_or.ok()) {
    return to_grpc_status(canonical_json_or.status());
  }

  auto offsets_or = resolve_target_offsets(req.target_layout());
  if (!offsets_or.ok()) {
    return to_grpc_status(offsets_or.status());
  }

  TargetMaterializationPlan unmapped_plan;
  MappedTargetMaterializationPlan mapped_plan;
  tensorcast::common::v1::ArtifactSelection current_selection;
  uint64_t logical_total_size = 0;
  v2::TransformPlacement placement = req.placement();
  std::optional<store::loading::DiskSource> disk_source = resolution.disk_source;

  if (mapped) {
    BindingRegistry::Record replay_record;
    replay_record.owner_pid = req.pid();
    replay_record.device_uuid = req.device_uuid();
    replay_record.source_selection = req.source_selection();
    replay_record.target_layout = req.target_layout();
    replay_record.copy_plan = req.copy_plan();
    replay_record.dst_tensors.assign(req.dst_tensors().begin(), req.dst_tensors().end());
    v2::MaterializeIntoMappedTargetRequest request = build_mapped_request(
        replay_record,
        resolution.resolved_artifact_id,
        effective_policy.preference,
        req.has_source_policy() ? &req.source_policy() : nullptr,
        req.has_operation_id() ? std::optional<std::string_view>(req.operation_id()) : std::nullopt,
        placement);
    auto plan_status = build_mapped_target_materialization_plan(
        d_.engine,
        request,
        resolution.resolved_artifact_id,
        *offsets_or,
        std::move(*canonical_json_or),
        /*record_result=*/nullptr,
        mapped_plan);
    if (!plan_status.ok()) {
      return plan_status;
    }
    logical_total_size = mapped_plan.logical_total_size;
    current_selection = build_mapped_bound_selection(
        resolution.resolved_artifact_id,
        req.target_layout(),
        std::string_view(req.target_index_bytes().data(), req.target_index_bytes().size()));
  } else {
    v2::MaterializeIntoTargetRequest request = build_unmapped_request(
        req.source_selection(),
        resolution.resolved_artifact_id,
        req.target_layout(),
        req.pid(),
        req.device_uuid(),
        effective_policy.preference,
        req.has_source_policy() ? &req.source_policy() : nullptr,
        req.has_operation_id() ? std::optional<std::string_view>(req.operation_id()) : std::nullopt,
        placement);
    auto plan_status = build_target_materialization_plan(
        d_.engine,
        resolution.resolved_artifact_id,
        request,
        req.target_layout(),
        *offsets_or,
        std::move(*canonical_json_or),
        /*record_result=*/nullptr,
        unmapped_plan);
    if (!plan_status.ok()) {
      return plan_status;
    }
    logical_total_size = unmapped_plan.logical_total_size;
    current_selection = unmapped_plan.resolved_selection;
  }

  auto allocation = std::make_unique<common::memory::GpuDeviceMemory>();
  if (auto allocate_status = allocation->allocate(logical_total_size, device.ordinal); !allocate_status.ok()) {
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

  store::loading::MaterializeHints hints;
  const std::chrono::milliseconds request_budget = resolve_owner_request_budget(rctx.server_context());
  hints.request_budget = request_budget;
  hints.transport_wait_timeout = request_budget;
  hints.artifact_id = resolution.resolved_artifact_id;
  const std::string requester_worker_id = d_.identity.worker_id();
  if (!requester_worker_id.empty()) {
    hints.transport_requester_worker_id = requester_worker_id;
  }
  auto disk_metadata_or = build_binding_disk_metadata(
      resolution.normalized_disk_path, resolution.resolved_artifact_id, device.ordinal, d_.disk_imports);
  if (!disk_metadata_or.ok()) {
    return to_grpc_status(disk_metadata_or.status());
  }
  auto disk_metadata = std::move(*disk_metadata_or);
  const bool prefer_direct_disk_for_source_layout =
      disk_source.has_value() && disk_metadata.has_value() && disk_metadata->source_index_json.has_value();
  hints.source_preference = prefer_direct_disk_for_source_layout ? store::loading::SourcePreference::kPreferDisk
                                                                 : to_hint_preference(effective_policy.preference);
  hints.allow_p2p = prefer_direct_disk_for_source_layout ? false : effective_policy.allow_p2p;
  hints.allow_disk = effective_policy.allow_disk;
  hints.verify = store::loading::MaterializeHints::Verify::NONE;
  hints.export_policy = store::loading::ExportPolicy::kForce;
  if (disk_metadata.has_value()) {
    hints.disk_metadata = *disk_metadata;
  }
  if (disk_source.has_value()) {
    hints.source_mutation_policy = store::loading::SourceMutationPolicy::kReadOnly;
  }
  if (mapped) {
    if (mapped_plan.view_plan.has_value() && mapped_plan.view_plan->transform.requires_materialization) {
      return {StatusCode::INVALID_ARGUMENT, "mapped binding does not support view transforms"};
    }
    if (!mapped_plan.resolved_selection.view_id().empty() || mapped_plan.view_spec.has_value() ||
        mapped_plan.view_plan.has_value()) {
      store::loading::VariantIdentity variant;
      variant.canonical_artifact_id = resolution.resolved_artifact_id;
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
      variant.placement = resolve_transform_placement(req.placement(), mapped_plan.view_spec);
      hints.variant = std::move(variant);
    }
  } else if (
      unmapped_plan.view_plan.has_value() && req.target_layout().index_kind() == v2::TargetLayout::INDEX_KIND_VIEW) {
    store::loading::VariantIdentity variant;
    variant.canonical_artifact_id = resolution.resolved_artifact_id;
    if (unmapped_plan.resolved_view_id.has_value()) {
      variant.view_id = *unmapped_plan.resolved_view_id;
    }
    if (unmapped_plan.view_spec.has_value()) {
      variant.view_spec = unmapped_plan.view_spec;
    }
    variant.cached_plan = unmapped_plan.view_plan;
    variant.canonical_index_json = unmapped_plan.canonical_index_json;
    variant.placement = resolve_transform_placement(req.placement(), unmapped_plan.view_spec);
    hints.variant = std::move(variant);
  }

  absl::StatusOr<store::loading::MaterializeIntoTargetResult> result_or = mapped
      ? d_.engine.materialize_mapped_into_target(
            device,
            storage_layout.into_target,
            mapped_plan.copy_plan.map,
            mapped_plan.canonical_index_json,
            materialization_payload::compute_generation_from_index(mapped_plan.canonical_index_json),
            hints,
            disk_source)
      : d_.engine.materialize_into_target(
            device,
            storage_layout.into_target,
            unmapped_plan.canonical_index_json,
            materialization_payload::compute_generation_from_index(unmapped_plan.canonical_index_json),
            hints,
            disk_source);
  if (!result_or.ok()) {
    return to_grpc_status(result_or.status());
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
  std::string binding_id;
  std::shared_ptr<BindingRegistry::Record> record;
  absl::Status insert_status = absl::UnknownError("uninitialized");
  for (int attempt = 0; attempt < 4; ++attempt) {
    binding_id = mint_binding_id();
    record = std::make_shared<BindingRegistry::Record>();
    record->binding_id = binding_id;
    record->binding_layout_id = req.binding_layout_id();
    record->owner_pid = req.pid();
    record->device_id = device.ordinal;
    record->device_uuid = req.device_uuid();
    record->ownership = v2::BINDING_OWNERSHIP_DAEMON;
    record->mapped = mapped;
    record->closed = false;
    record->export_refs = 1;
    record->allocation = std::move(allocation);
    record->handle_bytes = handle_bytes;
    record->source_selection = req.source_selection();
    record->source_selection.set_artifact_id(resolution.resolved_artifact_id);
    record->target_layout = req.target_layout();
    record->target_index_json = std::string(req.target_index_bytes().data(), req.target_index_bytes().size());
    record->target_layout_hash = target_layout_hash;
    mark_ready_artifact(record.get(), resolution.resolved_artifact_id, current_selection, std::string());
    if (mapped) {
      record->copy_plan = req.copy_plan();
      record->dst_tensors.assign(req.dst_tensors().begin(), req.dst_tensors().end());
    }
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

  auto target_publication_token_or = maybe_mint_target_publication_token(
      d_.capability_tokens,
      d_.identity,
      d_.target_materialization_service,
      req.target_layout(),
      current_selection,
      mapped ? mapped_plan.canonical_index_json : unmapped_plan.canonical_index_json,
      device.ordinal,
      req.device_uuid(),
      req.pid(),
      req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view(),
      std::move(storage_layout.publish_segments),
      std::move(storage_layout.publish_storages));
  if (target_publication_token_or.ok()) {
    absl::MutexLock lock(&record->mu);
    record->target_publication_token = *target_publication_token_or;
  }

  resp.set_binding_id(binding_id);
  resp.set_artifact_id(resolution.resolved_artifact_id);
  resp.mutable_mem_handle()->set_cuda_ipc_handle(
      handle_bytes.as_string_view().data(), handle_bytes.as_string_view().size());
  resp.mutable_mem_handle()->set_lease_token(*token_or);
  resp.set_target_index_bytes(req.target_index_bytes());
  for (auto& descriptor : descriptors_or->descriptors) {
    *resp.add_payloads() = std::move(descriptor);
  }
  if (target_publication_token_or.ok() && !target_publication_token_or->empty()) {
    resp.set_target_publication_token(*target_publication_token_or);
  }
  resp.mutable_resolved_selection()->CopyFrom(current_selection);
  resp.set_source(to_proto_source(result_or->source));
  resp.set_state(v2::BINDING_STATE_READY_ARTIFACT);
  fill_binding_value(*record, *resp.mutable_current_value());
  {
    absl::MutexLock lock(&record->mu);
    if (target_publication_token_or.ok()) {
      record->target_publication_token = *target_publication_token_or;
    }
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
    current_binding_value_id = record->current_binding_value_id;
  }
  if (auto contribution_status =
          ensure_no_live_binding_contributions(d_.global_store_client, record->binding_id, current_binding_value_id);
      !contribution_status.ok()) {
    return to_grpc_status(contribution_status);
  }
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
    mark_ready_artifact(
        record.get(),
        req.has_source_artifact_id() ? req.source_artifact_id() : req.selection().artifact_id(),
        req.selection(),
        std::string_view(req.target_publication_token().data(), req.target_publication_token().size()));
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
    current_binding_value_id = record->current_binding_value_id;
  }
  if (auto contribution_status =
          ensure_no_live_binding_contributions(d_.global_store_client, record->binding_id, current_binding_value_id);
      !contribution_status.ok()) {
    return to_grpc_status(contribution_status);
  }
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
    contribution_record.owner_pid = binding_record->owner_pid;
    contribution_record.device_id = binding_record->device_id;
    contribution_record.device_uuid = binding_record->device_uuid;
    contribution_record.target_layout = binding_record->target_layout;
    contribution_record.target_index_json = binding_record->target_index_json;
    contribution_record.current_artifact_id = binding_record->current_artifact_id;
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

  std::optional<std::string> committed_view_id;
  const auto registration_status = commit_binding_contribution_registration(
      d_, rctx.server_context(), contribution_record, registration_plan, committed_view_id);
  if (!registration_status.ok()) {
    return registration_status;
  }
  if (registration_plan.structural_view_id.has_value() && committed_view_id.has_value() &&
      committed_view_id != registration_plan.structural_view_id) {
    return {StatusCode::FAILED_PRECONDITION, "piece contribution structural lowering changed during registration"};
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
    mark_ready_local(record.get());
    resp.set_state(record->state);
    fill_binding_value(*record, *resp.mutable_current_value());
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status OwnedBindingService::refill_owned_binding(
    RpcContext& rctx,
    const v2::RefillOwnedBindingRequest& req,
    v2::RefillOwnedBindingResponse& resp) {
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req.binding_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_id is required"};
  }
  if (req.artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
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
    owner_pid = record->owner_pid;
    device_id = record->device_id;
    device_uuid = record->device_uuid;
    mapped = record->mapped;
    target_layout = record->target_layout;
    target_index_json = record->target_index_json;
    source_selection = record->source_selection;
    copy_plan = record->copy_plan;
    dst_tensors = record->dst_tensors;
    current_binding_value_id = record->current_binding_value_id;
  }
  if (auto contribution_status =
          ensure_no_live_binding_contributions(d_.global_store_client, record->binding_id, current_binding_value_id);
      !contribution_status.ok()) {
    return to_grpc_status(contribution_status);
  }

  auto effective_policy =
      resolve_source_policy(req.has_source_policy() ? &req.source_policy() : nullptr, req.preference());
  if (auto policy_status = validate_source_policy(effective_policy); !policy_status.ok()) {
    return to_grpc_status(policy_status);
  }

  const auto device = d_.devices.From(v2::DeviceType::DEVICE_TYPE_GPU, device_uuid, std::nullopt);
  const bool loopback_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  auto resolution_or = resolve_artifact_and_disk_source(
      d_.global_store_client,
      &d_.disk_imports,
      d_.storage_path,
      req.artifact_id(),
      effective_policy.allow_disk,
      /*allow_local_import_fallback=*/true,
      loopback_peer);
  if (!resolution_or.ok()) {
    return to_grpc_status(resolution_or.status());
  }
  auto resolution = std::move(*resolution_or);

  auto canonical_json_or = load_canonical_index_with_disk_fallback(
      d_.engine,
      resolution.resolved_artifact_id,
      resolution.normalized_disk_path,
      device.ordinal,
      resolution.gs_connected);
  if (!canonical_json_or.ok()) {
    return to_grpc_status(canonical_json_or.status());
  }

  auto offsets_or = resolve_target_offsets(target_layout);
  if (!offsets_or.ok()) {
    return to_grpc_status(offsets_or.status());
  }

  tensorcast::common::v1::ArtifactSelection current_selection;
  MappedTargetMaterializationPlan mapped_plan;
  TargetMaterializationPlan unmapped_plan;
  std::optional<store::loading::DiskSource> disk_source = resolution.disk_source;
  if (mapped) {
    BindingRegistry::Record replay_record;
    replay_record.owner_pid = owner_pid;
    replay_record.device_uuid = device_uuid;
    replay_record.source_selection = source_selection;
    replay_record.target_layout = target_layout;
    replay_record.copy_plan = copy_plan;
    replay_record.dst_tensors = dst_tensors;
    v2::MaterializeIntoMappedTargetRequest request = build_mapped_request(
        replay_record,
        resolution.resolved_artifact_id,
        effective_policy.preference,
        req.has_source_policy() ? &req.source_policy() : nullptr,
        req.has_operation_id() ? std::optional<std::string_view>(req.operation_id()) : std::nullopt,
        req.placement());
    auto plan_status = build_mapped_target_materialization_plan(
        d_.engine,
        request,
        resolution.resolved_artifact_id,
        *offsets_or,
        std::move(*canonical_json_or),
        /*record_result=*/nullptr,
        mapped_plan);
    if (!plan_status.ok()) {
      return plan_status;
    }
    current_selection = build_mapped_bound_selection(
        resolution.resolved_artifact_id, target_layout, std::string_view(target_index_json));
  } else {
    v2::MaterializeIntoTargetRequest request = build_unmapped_request(
        source_selection,
        resolution.resolved_artifact_id,
        target_layout,
        owner_pid,
        device_uuid,
        effective_policy.preference,
        req.has_source_policy() ? &req.source_policy() : nullptr,
        req.has_operation_id() ? std::optional<std::string_view>(req.operation_id()) : std::nullopt,
        req.placement());
    auto plan_status = build_target_materialization_plan(
        d_.engine,
        resolution.resolved_artifact_id,
        request,
        target_layout,
        *offsets_or,
        std::move(*canonical_json_or),
        /*record_result=*/nullptr,
        unmapped_plan);
    if (!plan_status.ok()) {
      return plan_status;
    }
    current_selection = unmapped_plan.resolved_selection;
  }

  OwnedStorageLayout storage_layout;
  {
    absl::MutexLock lock(&record->mu);
    auto storage_layout_or = build_owned_storage_layout(
        target_layout, device_id, gsl::not_null<void*>{record->allocation->get()}, record->handle_bytes);
    if (!storage_layout_or.ok()) {
      return to_grpc_status(storage_layout_or.status());
    }
    storage_layout = std::move(*storage_layout_or);
  }

  store::loading::MaterializeHints hints;
  const std::chrono::milliseconds request_budget = resolve_owner_request_budget(rctx.server_context());
  hints.request_budget = request_budget;
  hints.transport_wait_timeout = request_budget;
  hints.artifact_id = resolution.resolved_artifact_id;
  const std::string requester_worker_id = d_.identity.worker_id();
  if (!requester_worker_id.empty()) {
    hints.transport_requester_worker_id = requester_worker_id;
  }
  auto disk_metadata_or = build_binding_disk_metadata(
      resolution.normalized_disk_path, resolution.resolved_artifact_id, device.ordinal, d_.disk_imports);
  if (!disk_metadata_or.ok()) {
    return to_grpc_status(disk_metadata_or.status());
  }
  auto disk_metadata = std::move(*disk_metadata_or);
  const bool prefer_direct_disk_for_source_layout =
      disk_source.has_value() && disk_metadata.has_value() && disk_metadata->source_index_json.has_value();
  hints.source_preference = prefer_direct_disk_for_source_layout ? store::loading::SourcePreference::kPreferDisk
                                                                 : to_hint_preference(effective_policy.preference);
  hints.allow_p2p = prefer_direct_disk_for_source_layout ? false : effective_policy.allow_p2p;
  hints.allow_disk = effective_policy.allow_disk;
  hints.verify = store::loading::MaterializeHints::Verify::NONE;
  hints.export_policy = store::loading::ExportPolicy::kForce;
  if (disk_metadata.has_value()) {
    hints.disk_metadata = *disk_metadata;
  }
  if (disk_source.has_value()) {
    hints.source_mutation_policy = store::loading::SourceMutationPolicy::kReadOnly;
  }
  if (mapped) {
    if (mapped_plan.view_plan.has_value() && mapped_plan.view_plan->transform.requires_materialization) {
      return {StatusCode::INVALID_ARGUMENT, "mapped binding does not support view transforms"};
    }
    if (!mapped_plan.resolved_selection.view_id().empty() || mapped_plan.view_spec.has_value() ||
        mapped_plan.view_plan.has_value()) {
      store::loading::VariantIdentity variant;
      variant.canonical_artifact_id = resolution.resolved_artifact_id;
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
      variant.placement = resolve_transform_placement(req.placement(), mapped_plan.view_spec);
      hints.variant = std::move(variant);
    }
  } else if (unmapped_plan.view_plan.has_value() && target_layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW) {
    store::loading::VariantIdentity variant;
    variant.canonical_artifact_id = resolution.resolved_artifact_id;
    if (unmapped_plan.resolved_view_id.has_value()) {
      variant.view_id = *unmapped_plan.resolved_view_id;
    }
    if (unmapped_plan.view_spec.has_value()) {
      variant.view_spec = unmapped_plan.view_spec;
    }
    variant.cached_plan = unmapped_plan.view_plan;
    variant.canonical_index_json = unmapped_plan.canonical_index_json;
    variant.placement = resolve_transform_placement(req.placement(), unmapped_plan.view_spec);
    hints.variant = std::move(variant);
  }

  absl::StatusOr<store::loading::MaterializeIntoTargetResult> result_or = mapped
      ? d_.engine.materialize_mapped_into_target(
            device,
            storage_layout.into_target,
            mapped_plan.copy_plan.map,
            mapped_plan.canonical_index_json,
            materialization_payload::compute_generation_from_index(mapped_plan.canonical_index_json),
            hints,
            disk_source)
      : d_.engine.materialize_into_target(
            device,
            storage_layout.into_target,
            unmapped_plan.canonical_index_json,
            materialization_payload::compute_generation_from_index(unmapped_plan.canonical_index_json),
            hints,
            disk_source);
  if (!result_or.ok()) {
    {
      absl::MutexLock lock(&record->mu);
      mark_dirty(record.get());
    }
    return to_grpc_status(result_or.status());
  }

  auto target_publication_token_or = maybe_mint_target_publication_token(
      d_.capability_tokens,
      d_.identity,
      d_.target_materialization_service,
      target_layout,
      current_selection,
      mapped ? mapped_plan.canonical_index_json : unmapped_plan.canonical_index_json,
      device.ordinal,
      device_uuid,
      owner_pid,
      req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view(),
      std::move(storage_layout.publish_segments),
      std::move(storage_layout.publish_storages));

  {
    absl::MutexLock lock(&record->mu);
    mark_ready_artifact(
        record.get(),
        resolution.resolved_artifact_id,
        current_selection,
        target_publication_token_or.ok() ? *target_publication_token_or : std::string());
  }

  resp.set_artifact_id(resolution.resolved_artifact_id);
  resp.set_source(to_proto_source(result_or->source));
  if (target_publication_token_or.ok() && !target_publication_token_or->empty()) {
    resp.set_target_publication_token(*target_publication_token_or);
  }
  resp.mutable_resolved_selection()->CopyFrom(current_selection);
  resp.set_state(v2::BINDING_STATE_READY_ARTIFACT);
  fill_binding_value(*record, *resp.mutable_current_value());
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
