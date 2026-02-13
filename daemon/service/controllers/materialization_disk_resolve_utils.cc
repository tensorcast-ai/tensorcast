// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_disk_resolve_utils.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "daemon/service/controllers/materialization_index_source_utils.h"
#include "daemon/service/controllers/materialization_payload_utils.h"
#include "daemon/util/atomic_file_utils.h"
#include "daemon/util/path_utils.h"
#include "nlohmann/json.hpp"
#include "opentelemetry/metrics/provider.h"

#include "core/common/artifact_hash.h"
#include "core/store/materialization/dataplane/metadata/disk_dir_hash.h"

namespace tensorcast::daemon::materialization_disk_resolve {

namespace {

using materialization_index_source::DescriptorMetadata;
using materialization_index_source::ensure_tensor_index_present;
using materialization_index_source::load_descriptor_metadata;
using materialization_index_source::maybe_backfill_tensor_index;
using materialization_index_source::validate_descriptor_against_index;
using materialization_payload::compute_generation_from_index;

absl::Status write_artifact_descriptor(
    const std::filesystem::path& artifact_dir,
    std::string_view artifact_id,
    std::string_view index_multihash,
    std::string_view data_multihash,
    std::optional<uint64_t> total_size,
    std::optional<std::string_view> schema_version) {
  nlohmann::json desc;
  desc["artifact_id"] = std::string(artifact_id);
  desc["index_multihash"] = std::string(index_multihash);
  desc["data_multihash"] = std::string(data_multihash);
  desc["schema_version"] = schema_version.has_value() ? std::string(*schema_version) : "v3";
  desc["encoding"] = "json";
  if (total_size.has_value() && *total_size > 0) {
    desc["total_size"] = *total_size;
  }
  const auto descriptor_path = artifact_dir / "artifact_descriptor.json";
  return atomic_file_utils::write_file_atomically(descriptor_path, desc.dump(2));
}

} // namespace

void record_disk_path_denied() {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_store_disk_path_denied_total");
    if (counter) {
      counter->Add(1);
    }
  } catch (...) {
  }
}

void record_disk_resolution_outcome(std::string_view outcome) {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_store_disk_path_resolve_total");
    if (counter) {
      counter->Add(1, {{"outcome", std::string(outcome)}});
    }
  } catch (...) {
  }
}

absl::StatusOr<ResolveArtifactFromDiskResult> resolve_artifact_from_disk(
    std::string_view disk_path,
    const std::filesystem::path& storage_path,
    bool verify_checksums) {
  auto normalized_or = normalize_disk_import_path(disk_path, storage_path);
  if (!normalized_or.ok()) {
    record_disk_path_denied();
    record_disk_resolution_outcome("invalid_argument");
    return normalized_or.status();
  }
  const auto& normalized = *normalized_or;

  auto descriptor_or = load_descriptor_metadata(normalized);
  if (!descriptor_or.ok()) {
    record_disk_resolution_outcome("invalid_descriptor");
    return descriptor_or.status();
  }
  DescriptorMetadata descriptor = *descriptor_or;

  auto index_presence_status = ensure_tensor_index_present(normalized);
  if (!index_presence_status.ok()) {
    record_disk_resolution_outcome("not_found");
    return index_presence_status;
  }

  auto index_or = store::loader::read_from_artifact_dir(normalized, /*target_device_id=*/0);
  if (!index_or.ok()) {
    record_disk_resolution_outcome("not_found");
    return index_or.status();
  }

  const bool has_descriptor = descriptor.found;
  const bool is_safetensors = index_or->is_safetensors;
  if (verify_checksums && has_descriptor) {
    auto validation_status = validate_descriptor_against_index(descriptor, *index_or, /*verify_checksums=*/true);
    if (!validation_status.ok()) {
      record_disk_resolution_outcome("checksum_failed");
      return validation_status;
    }
  } else if (verify_checksums && !has_descriptor) {
    LOG(WARNING) << "verify_checksums requested but artifact_descriptor.json missing at " << normalized.string()
                 << "; skipping descriptor validation";
  }

  std::string index_multihash = index_or->index_multihash;
  if (index_multihash.empty()) {
    auto index_mh_or = common::compute_index_multihash(
        std::optional<std::string>(index_or->canonical_index_json), /*index_key_hex=*/"");
    if (!index_mh_or.ok()) {
      record_disk_resolution_outcome("invalid_descriptor");
      return index_mh_or.status();
    }
    index_multihash = *index_mh_or;
  }

  auto data_mh_or = store::loader::compute_data_multihash_from_disk_dir(normalized.string());
  if (!data_mh_or.ok()) {
    record_disk_resolution_outcome("invalid_descriptor");
    return data_mh_or.status();
  }
  const std::string data_multihash = *data_mh_or;
  const std::string artifact_id = absl::StrCat("mi2:", index_multihash, ":", data_multihash);

  const bool artifact_id_mismatch = descriptor.artifact_id.has_value() && *descriptor.artifact_id != artifact_id;
  if (artifact_id_mismatch && verify_checksums) {
    record_disk_resolution_outcome("checksum_failed");
    return absl::FailedPreconditionError("artifact_id mismatch between resolved descriptor and computed identity");
  }
  if (artifact_id_mismatch && !verify_checksums) {
    LOG(WARNING) << "artifact_id mismatch between descriptor and computed identity at " << normalized.string()
                 << "; verify_checksums=false, proceeding with computed identity";
  }

  const bool data_multihash_mismatch =
      descriptor.data_multihash.has_value() && *descriptor.data_multihash != data_multihash;
  if (data_multihash_mismatch && verify_checksums) {
    record_disk_resolution_outcome("checksum_failed");
    return absl::FailedPreconditionError("data_multihash mismatch for disk artifact");
  }
  if (data_multihash_mismatch && !verify_checksums) {
    LOG(WARNING) << "data_multihash mismatch between descriptor and disk bytes at " << normalized.string()
                 << "; verify_checksums=false, proceeding with computed multihash";
  }

  const bool missing_descriptor_fields = !descriptor.found || !descriptor.artifact_id.has_value() ||
      !descriptor.index_multihash.has_value() || !descriptor.data_multihash.has_value();
  const bool stale_descriptor_fields = artifact_id_mismatch ||
      (descriptor.index_multihash.has_value() && *descriptor.index_multihash != index_multihash) ||
      data_multihash_mismatch;
  const bool should_backfill_descriptor = missing_descriptor_fields || stale_descriptor_fields;
  bool descriptor_present = descriptor.found;
  if (should_backfill_descriptor) {
    const auto total_size =
        index_or->total_size_bytes > 0 ? std::optional<uint64_t>(index_or->total_size_bytes) : std::nullopt;
    const std::optional<std::string_view> schema_version = descriptor.schema_version.has_value()
        ? std::optional<std::string_view>(*descriptor.schema_version)
        : std::nullopt;
    auto write_status =
        write_artifact_descriptor(normalized, artifact_id, index_multihash, data_multihash, total_size, schema_version);
    if (!write_status.ok()) {
      LOG(WARNING) << "Failed to backfill artifact_descriptor.json at " << normalized.string() << ": " << write_status;
    } else {
      descriptor_present = true;
    }
  }

  if (is_safetensors) {
    maybe_backfill_tensor_index(normalized, index_or->canonical_index_json);
  }

  ResolveArtifactFromDiskResult result;
  result.normalized_disk_path = normalized;
  result.artifact_id = artifact_id;
  result.canonical_index_json = index_or->canonical_index_json;
  result.index_multihash = index_multihash;
  result.data_multihash = data_multihash;
  result.generation = compute_generation_from_index(index_or->canonical_index_json);
  result.descriptor_present = descriptor_present;
  record_disk_resolution_outcome("ok");
  return result;
}

} // namespace tensorcast::daemon::materialization_disk_resolve
