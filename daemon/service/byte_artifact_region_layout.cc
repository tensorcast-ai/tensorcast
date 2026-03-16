// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/byte_artifact_region_layout.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "core/store/device_registry.h"
#include "core/store/device_types.h"
#include "core/store/materialization/dataplane/sources/memory_source.h"
#include "gsl/pointers"

namespace tensorcast::daemon {

namespace {

using materialization_target_storage::AcquireTargetStoragesError;
using materialization_target_storage::TargetStorageLease;

absl::Status validate_layout_device_uuid(const v2::TargetLayout& layout, std::string_view device_uuid) {
  if (device_uuid.empty()) {
    return absl::InvalidArgumentError("device_uuid is required");
  }
  std::optional<int32_t> device_id;
  for (const auto& storage : layout.storages()) {
    if (storage.device_id() < 0) {
      return absl::InvalidArgumentError("storage.device_id must be >= 0");
    }
    if (!device_id.has_value()) {
      device_id = storage.device_id();
    } else if (*device_id != storage.device_id()) {
      return absl::InvalidArgumentError("byte artifact region layout must use one device_id");
    }
    const auto key = store::DeviceRegistry::instance().gpu_key(storage.device_id());
    if (!key.uuid.empty() && key.uuid != device_uuid) {
      return absl::InvalidArgumentError("storage.device_id does not match device_uuid");
    }
  }
  return absl::OkStatus();
}

} // namespace

absl::StatusOr<ByteArtifactRegionLayout> ByteArtifactRegionLayout::acquire(
    IpcRegionRegistry& registry,
    const v2::TargetLayout& layout,
    int owner_pid,
    std::string_view device_uuid,
    const absl::flat_hash_map<std::string, std::uint64_t>& expected_lengths) {
  if (layout.layout_kind() != v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED) {
    return absl::InvalidArgumentError("target_layout.layout_kind must be coalesced");
  }
  if (layout.index_kind() != v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED) {
    return absl::InvalidArgumentError("target_layout.index_kind must be canonical");
  }
  if (layout.tensor_spec_kind() != v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS) {
    return absl::InvalidArgumentError("target_layout.tensor_spec_kind must be offsets");
  }
  if (!layout.view_id().empty()) {
    return absl::InvalidArgumentError("target_layout.view_id is not supported for byte artifacts");
  }
  if (layout.aliases_size() != 0) {
    return absl::InvalidArgumentError("target_layout.aliases are not supported for byte artifacts");
  }
  if (layout.storages_size() == 0) {
    return absl::InvalidArgumentError("target_layout must include at least one storage");
  }
  for (const auto& storage : layout.storages()) {
    if (storage.storage_source_case() != v2::StorageEntry::kVramRegionId) {
      return absl::InvalidArgumentError("target_layout storages must use vram_region_id");
    }
  }
  auto device_uuid_st = validate_layout_device_uuid(layout, device_uuid);
  if (!device_uuid_st.ok()) {
    return device_uuid_st;
  }

  AcquireTargetStoragesError acquire_error = AcquireTargetStoragesError::kUnknown;
  auto lease_or = TargetStorageLease::acquire(registry, layout.storages(), owner_pid, &acquire_error);
  if (!lease_or.ok()) {
    return lease_or.status();
  }

  ByteArtifactRegionLayout result;
  result.storage_lease_ = std::move(*lease_or);
  result.storages_.reserve(layout.storages_size());
  result.device_id_ = layout.storages(0).device_id();

  absl::flat_hash_map<std::string, std::size_t> storage_indices;
  storage_indices.reserve(layout.storages_size());
  std::uint64_t logical_cursor = 0;
  for (int i = 0; i < layout.storages_size(); ++i) {
    const auto& storage = layout.storages(i);
    result.storages_.push_back(
        StorageRange{
            .storage_id = storage.storage_id(),
            .logical_base = logical_cursor,
            .length = storage.storage_length(),
            .base_ptr = result.storage_lease_.storages().at(static_cast<std::size_t>(i)).base_ptr.get(),
        });
    storage_indices.emplace(storage.storage_id(), static_cast<std::size_t>(i));
    logical_cursor += storage.storage_length();
  }

  if (layout.offsets_size() == 0) {
    if (expected_lengths.size() != 1) {
      return absl::InvalidArgumentError(
          "target_layout.offsets are required for multi-item byte artifact region access");
    }
    if (result.storages_.size() != 1) {
      return absl::InvalidArgumentError("target_layout.offsets are required when multiple storages are used");
    }
    const auto& [artifact_id, expected_length] = *expected_lengths.begin();
    if (expected_length != 0 && expected_length != logical_cursor) {
      return absl::InvalidArgumentError("target_layout total size does not match expected byte length");
    }
    result.items_.emplace(
        artifact_id,
        ItemRange{
            .storage_index = 0,
            .logical_offset = 0,
            .logical_length = logical_cursor,
            .storage_local_offset = 0,
        });
    return result;
  }

  if (layout.offsets_size() != static_cast<int>(expected_lengths.size())) {
    return absl::InvalidArgumentError("target_layout.offsets must cover every requested byte artifact");
  }

  result.items_.reserve(layout.offsets_size());
  for (const auto& entry : layout.offsets()) {
    if (entry.name().empty()) {
      return absl::InvalidArgumentError("target_layout.offsets include empty name");
    }
    const auto expected_it = expected_lengths.find(entry.name());
    if (expected_it == expected_lengths.end()) {
      return absl::InvalidArgumentError("target_layout.offsets include unknown artifact_id");
    }
    const auto storage_it = storage_indices.find(entry.storage_id());
    if (storage_it == storage_indices.end()) {
      return absl::InvalidArgumentError("target_layout.offsets reference unknown storage_id");
    }
    const auto& storage = result.storages_.at(storage_it->second);
    if (entry.storage_offset() < storage.logical_base) {
      return absl::InvalidArgumentError("target_layout.offset.storage_offset is out of bounds");
    }
    if (entry.logical_length() == 0) {
      return absl::InvalidArgumentError("target_layout.offset.logical_length must be > 0");
    }
    if (expected_it->second != 0 && expected_it->second != entry.logical_length()) {
      return absl::InvalidArgumentError("target_layout.offset.logical_length does not match expected byte length");
    }
    const std::uint64_t storage_local_offset = entry.storage_offset() - storage.logical_base;
    if (storage_local_offset + entry.logical_length() > storage.length) {
      return absl::InvalidArgumentError("target_layout.offset exceeds storage bounds");
    }
    auto [_, inserted] = result.items_.emplace(
        entry.name(),
        ItemRange{
            .storage_index = storage_it->second,
            .logical_offset = entry.storage_offset(),
            .logical_length = entry.logical_length(),
            .storage_local_offset = storage_local_offset,
        });
    if (!inserted) {
      return absl::InvalidArgumentError("target_layout.offsets include duplicate artifact_id");
    }
  }
  return result;
}

bool ByteArtifactRegionLayout::contains(std::string_view artifact_id) const {
  return items_.contains(std::string(artifact_id));
}

std::uint64_t ByteArtifactRegionLayout::expected_length(std::string_view artifact_id) const {
  const auto it = items_.find(std::string(artifact_id));
  if (it == items_.end()) {
    return 0;
  }
  return it->second.logical_length;
}

int ByteArtifactRegionLayout::device_id() const {
  return device_id_;
}

absl::StatusOr<store::loading::IntoTargetLayout> ByteArtifactRegionLayout::build_item_target_layout(
    std::string_view artifact_id) const {
  const auto it = items_.find(std::string(artifact_id));
  if (it == items_.end()) {
    return absl::NotFoundError("artifact_id is not mapped in region layout");
  }
  const auto& range = it->second;
  const auto& storage = storages_.at(range.storage_index);
  store::loading::IntoTargetLayout layout;
  layout.total_size = range.logical_length;
  layout.storages.push_back(
      store::loading::IntoTargetStorage{
          .base_ptr = gsl::not_null<void*>{static_cast<std::uint8_t*>(storage.base_ptr) + range.storage_local_offset},
          .length = range.logical_length,
      });
  return layout;
}

absl::StatusOr<std::shared_ptr<store::loader::SeekableSource>> ByteArtifactRegionLayout::open_item_source(
    std::string_view artifact_id) const {
  const auto it = items_.find(std::string(artifact_id));
  if (it == items_.end()) {
    return absl::NotFoundError("artifact_id is not mapped in region layout");
  }
  const auto& range = it->second;
  const auto& storage = storages_.at(range.storage_index);
  return std::shared_ptr<store::loader::SeekableSource>(std::make_shared<store::loader::GpuMemorySource>(
      gsl::not_null<void*>{static_cast<std::uint8_t*>(storage.base_ptr) + range.storage_local_offset},
      device_id_,
      range.logical_length));
}

} // namespace tensorcast::daemon
