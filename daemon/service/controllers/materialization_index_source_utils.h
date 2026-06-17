// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/store_engine.h"
#include "gsl/pointers"

namespace tensorcast::daemon::materialization_index_source {

struct DescriptorMetadata {
  bool found{false};
  std::optional<std::string> schema_version;
  std::optional<std::string> artifact_id;
  std::optional<std::string> index_multihash;
  std::optional<std::string> data_multihash;
};

struct TargetLayoutSpan {
  gsl::not_null<void*> base_ptr;
  uint64_t offset{0};
  uint64_t length{0};
};

enum class CanonicalIndexAuthority : std::uint8_t {
  kEngineMetadata = 1,
  kDiskSource = 2,
};

absl::Status ensure_tensor_index_present(const std::filesystem::path& artifact_dir);

CanonicalIndexAuthority canonical_index_authority_for_resolution(bool gs_connected, bool has_local_import);

absl::StatusOr<std::string> load_canonical_index_from_authority(
    store::StoreEngine& engine,
    std::string_view resolved_artifact_id,
    const std::optional<std::filesystem::path>& normalized_disk_path,
    int device_ordinal,
    CanonicalIndexAuthority authority);

std::optional<std::string> parse_mi2_data_multihash(std::string_view artifact_id);

absl::StatusOr<std::string> compute_target_layout_multihash(
    std::vector<TargetLayoutSpan> spans,
    uint64_t total_size,
    int device_id);

absl::StatusOr<DescriptorMetadata> load_descriptor_metadata(const std::filesystem::path& artifact_dir);

absl::Status validate_descriptor_against_index(
    const DescriptorMetadata& descriptor,
    const store::loader::IndexInfo& index_info,
    bool verify_checksums);

} // namespace tensorcast::daemon::materialization_index_source
