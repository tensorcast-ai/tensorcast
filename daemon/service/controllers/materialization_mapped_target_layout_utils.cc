// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_mapped_target_layout_utils.h"

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"

namespace tensorcast::daemon::materialization_mapped_target_layout {

namespace {

using materialization_layout::dtype_element_size;
using materialization_layout::product_dims;
using materialization_mapped_copy_plan::is_contiguous;
using materialization_mapped_copy_plan::MappedTensorSpec;

void set_reason(ValidationErrorReason* reason, ValidationErrorReason value) {
  if (reason != nullptr) {
    *reason = value;
  }
}

} // namespace

std::string_view validation_error_reason(ValidationErrorReason reason) {
  switch (reason) {
    case ValidationErrorReason::kTensorNameMissing:
      return "tensor_name_missing";
    case ValidationErrorReason::kTensorNameDuplicate:
      return "tensor_name_duplicate";
    case ValidationErrorReason::kLayoutMismatch:
      return "layout_mismatch";
    case ValidationErrorReason::kTensorNameMismatch:
      return "tensor_name_mismatch";
    case ValidationErrorReason::kStorageIdMissing:
      return "storage_id_missing";
    case ValidationErrorReason::kStorageIdMismatch:
      return "storage_id_mismatch";
    case ValidationErrorReason::kStorageLengthMismatch:
      return "storage_length_mismatch";
    case ValidationErrorReason::kOffsetMismatch:
      return "offset_mismatch";
    case ValidationErrorReason::kUnknown:
    default:
      return "transfer_error";
  }
}

absl::StatusOr<ValidatedMappedTargetLayout> validate_mapped_target_layout(
    const v2::MaterializeIntoMappedTargetRequest& req,
    const std::vector<materialization_layout::TargetOffsetEntry>& offsets,
    ValidationErrorReason* reason) {
  set_reason(reason, ValidationErrorReason::kUnknown);

  absl::flat_hash_map<std::string, materialization_layout::TargetOffsetEntry> offsets_by_name;
  offsets_by_name.reserve(offsets.size());
  for (const auto& entry : offsets) {
    if (entry.name.empty()) {
      set_reason(reason, ValidationErrorReason::kTensorNameMissing);
      return absl::InvalidArgumentError("target_layout includes empty tensor name");
    }
    if (offsets_by_name.contains(entry.name)) {
      set_reason(reason, ValidationErrorReason::kTensorNameDuplicate);
      return absl::InvalidArgumentError("target_layout includes duplicate tensor name");
    }
    offsets_by_name.emplace(entry.name, entry);
  }

  ValidatedMappedTargetLayout result;
  result.dst_specs.reserve(req.dst_tensors_size());
  for (const auto& spec : req.dst_tensors()) {
    if (spec.name().empty()) {
      set_reason(reason, ValidationErrorReason::kTensorNameMissing);
      return absl::InvalidArgumentError("dst_tensors include empty tensor name");
    }
    if (result.dst_specs.contains(spec.name())) {
      set_reason(reason, ValidationErrorReason::kTensorNameDuplicate);
      return absl::InvalidArgumentError("dst_tensors include duplicate tensor name");
    }
    if (spec.shape_size() != spec.stride_size()) {
      set_reason(reason, ValidationErrorReason::kLayoutMismatch);
      return absl::InvalidArgumentError("dst_tensors shape/stride size mismatch");
    }

    auto elem_or = dtype_element_size(spec.dtype());
    if (!elem_or.ok()) {
      set_reason(reason, ValidationErrorReason::kLayoutMismatch);
      return elem_or.status();
    }

    std::vector<int64_t> shape;
    shape.reserve(spec.shape_size());
    for (const auto& dim : spec.shape()) {
      shape.push_back(dim);
    }
    std::vector<int64_t> stride;
    stride.reserve(spec.stride_size());
    for (const auto& dim : spec.stride()) {
      stride.push_back(dim);
    }
    if (!is_contiguous(shape, stride)) {
      set_reason(reason, ValidationErrorReason::kLayoutMismatch);
      return absl::InvalidArgumentError("dst_tensors must be contiguous");
    }
    if (spec.storage_offset() != 0) {
      set_reason(reason, ValidationErrorReason::kLayoutMismatch);
      return absl::InvalidArgumentError("dst_tensors storage_offset must be 0");
    }

    uint64_t expected_bytes = *elem_or;
    if (!shape.empty()) {
      auto count_or = product_dims(shape);
      if (!count_or.ok()) {
        set_reason(reason, ValidationErrorReason::kLayoutMismatch);
        return count_or.status();
      }
      expected_bytes = (*count_or) * (*elem_or);
    }
    if (expected_bytes != spec.logical_length()) {
      set_reason(reason, ValidationErrorReason::kLayoutMismatch);
      return absl::InvalidArgumentError("dst_tensors logical_length mismatch");
    }

    result.dst_specs.emplace(
        spec.name(),
        MappedTensorSpec{
            .shape = std::move(shape),
            .stride = std::move(stride),
            .dtype = spec.dtype(),
            .storage_offset = spec.storage_offset(),
            .logical_length = spec.logical_length(),
            .element_size = *elem_or,
        });
  }

  if (result.dst_specs.empty()) {
    set_reason(reason, ValidationErrorReason::kLayoutMismatch);
    return absl::InvalidArgumentError("dst_tensors must be non-empty");
  }

  if (offsets_by_name.size() != result.dst_specs.size()) {
    set_reason(reason, ValidationErrorReason::kTensorNameMismatch);
    return absl::InvalidArgumentError("dst_tensors must match target_layout offsets");
  }
  for (const auto& [name, _] : result.dst_specs) {
    if (!offsets_by_name.contains(name)) {
      set_reason(reason, ValidationErrorReason::kTensorNameMismatch);
      return absl::InvalidArgumentError("dst_tensors must match target_layout offsets");
    }
  }

  absl::flat_hash_set<std::string> mapped_dst_names;
  mapped_dst_names.reserve(req.copy_plan().entries_size());
  for (const auto& entry : req.copy_plan().entries()) {
    if (entry.dst_name().empty()) {
      set_reason(reason, ValidationErrorReason::kTensorNameMissing);
      return absl::InvalidArgumentError("copy_plan entry missing dst_name");
    }
    mapped_dst_names.insert(entry.dst_name());
  }
  if (mapped_dst_names.size() != result.dst_specs.size()) {
    set_reason(reason, ValidationErrorReason::kTensorNameMismatch);
    return absl::InvalidArgumentError("copy_plan must cover every dst tensor");
  }

  struct StorageRange {
    uint64_t base_offset{0};
    uint64_t length{0};
  };

  absl::flat_hash_map<std::string, StorageRange> storage_ranges;
  storage_ranges.reserve(req.target_layout().storages_size());
  uint64_t range_cursor = 0;
  for (const auto& storage : req.target_layout().storages()) {
    if (storage.storage_id().empty()) {
      set_reason(reason, ValidationErrorReason::kStorageIdMissing);
      return absl::InvalidArgumentError("storage_id is required for each storage entry");
    }
    if (storage.storage_length() == 0) {
      set_reason(reason, ValidationErrorReason::kStorageLengthMismatch);
      return absl::InvalidArgumentError("storage_length must be non-zero");
    }
    if (storage_ranges.contains(storage.storage_id())) {
      set_reason(reason, ValidationErrorReason::kStorageIdMismatch);
      return absl::InvalidArgumentError("storage_id must be unique in target_layout");
    }
    storage_ranges.emplace(
        storage.storage_id(), StorageRange{.base_offset = range_cursor, .length = storage.storage_length()});
    if (storage.storage_length() > std::numeric_limits<uint64_t>::max() - range_cursor) {
      set_reason(reason, ValidationErrorReason::kStorageLengthMismatch);
      return absl::InvalidArgumentError("storage_length sum overflow");
    }
    range_cursor += storage.storage_length();
  }
  result.logical_total_size = range_cursor;

  result.dst_base_offsets.reserve(offsets_by_name.size());
  for (const auto& [name, entry] : offsets_by_name) {
    const auto range_it = storage_ranges.find(entry.storage_id);
    if (range_it == storage_ranges.end()) {
      set_reason(reason, ValidationErrorReason::kStorageIdMismatch);
      return absl::InvalidArgumentError("target_layout references unknown storage_id");
    }
    const auto& range = range_it->second;
    if (entry.storage_offset + entry.logical_length > range.length) {
      set_reason(reason, ValidationErrorReason::kOffsetMismatch);
      return absl::InvalidArgumentError("target_layout exceeds storage bounds");
    }
    const auto& spec = result.dst_specs.at(name);
    if (entry.logical_length != spec.logical_length) {
      set_reason(reason, ValidationErrorReason::kLayoutMismatch);
      return absl::InvalidArgumentError("target_layout logical_length mismatch");
    }
    if (entry.storage_offset != spec.storage_offset * spec.element_size) {
      set_reason(reason, ValidationErrorReason::kLayoutMismatch);
      return absl::InvalidArgumentError("target_layout storage_offset mismatch");
    }
    result.dst_base_offsets.emplace(name, range.base_offset + entry.storage_offset);
  }

  return result;
}

} // namespace tensorcast::daemon::materialization_mapped_target_layout
