// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "daemon/service/controllers/materialization_layout_utils.h"
#include "daemon/service/controllers/representation_layout_types.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::materialization_mapped_target_layout {

enum class ValidationErrorReason {
  kTensorNameMissing,
  kTensorNameDuplicate,
  kLayoutMismatch,
  kTensorNameMismatch,
  kStorageIdMissing,
  kStorageIdMismatch,
  kStorageLengthMismatch,
  kOffsetMismatch,
  kUnknown,
};

std::string_view validation_error_reason(ValidationErrorReason reason);

struct ValidatedMappedTargetLayout {
  absl::flat_hash_map<std::string, representation_layout::TensorLayoutSpec> dst_specs;
  absl::flat_hash_map<std::string, uint64_t> dst_base_offsets;
  uint64_t logical_total_size{0};
};

absl::StatusOr<ValidatedMappedTargetLayout> validate_mapped_target_layout(
    const v2::MaterializeIntoMappedTargetRequest& req,
    const std::vector<materialization_layout::TargetOffsetEntry>& offsets,
    ValidationErrorReason* reason);

} // namespace tensorcast::daemon::materialization_mapped_target_layout
