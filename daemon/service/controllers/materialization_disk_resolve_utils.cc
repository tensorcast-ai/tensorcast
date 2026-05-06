// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_disk_resolve_utils.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <sys/stat.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "daemon/service/controllers/materialization_index_source_utils.h"
#include "daemon/service/controllers/materialization_payload_utils.h"
#include "daemon/util/path_utils.h"
#include "opentelemetry/metrics/provider.h"

#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"
#include "core/store/materialization/dataplane/metadata/disk_dir_hash.h"
#include "core/store/materialization/dataplane/verification/verification_utils.h"

namespace tensorcast::daemon::materialization_disk_resolve {

namespace {

using materialization_index_source::DescriptorMetadata;
using materialization_index_source::load_descriptor_metadata;
using materialization_index_source::validate_descriptor_against_index;
using materialization_payload::compute_generation_from_index;
using store::loader::DiskArtifactContext;
using store::loader::IndexInfo;

constexpr std::string_view kSourceMutatedPrefix = "SOURCE_MUTATED:";

absl::Status normalize_disk_context_status(const absl::Status& status, std::string_view path) {
  if (status.ok()) {
    return status;
  }
  if (absl::IsNotFound(status)) {
    const std::string message(status.message());
    if (message.find("No replica partition files found") != std::string::npos) {
      return absl::InvalidArgumentError(
          absl::StrCat(
              "SOURCE_FORMAT_INVALID: source must include top-level tensor.data*, or *.safetensors files: ", path));
    }
    return absl::NotFoundError(absl::StrCat("SOURCE_NOT_FOUND: source path not found: ", path));
  }
  if (absl::IsFailedPrecondition(status)) {
    return absl::InvalidArgumentError(absl::StrCat("SOURCE_FORMAT_INVALID: ", status.message()));
  }
  return status;
}

std::string_view format_kind_token(MountedSourceFormatKind kind) {
  switch (kind) {
    case MountedSourceFormatKind::kPartitioned:
      return "partitioned";
    case MountedSourceFormatKind::kSafetensors:
      return "safetensors";
    case MountedSourceFormatKind::kUnspecified:
    default:
      return "unspecified";
  }
}

absl::Status validate_attestation_policy(
    const MountedSourceAttestationPolicy& policy,
    const MountedSourceMetadata& metadata) {
  if (!policy.lightweight_attestation_enabled) {
    return absl::FailedPreconditionError("mounted-source lightweight attestation is disabled by daemon policy");
  }
  if (metadata.format_kind == MountedSourceFormatKind::kPartitioned && !policy.allow_partitioned) {
    return absl::PermissionDeniedError("mounted-source policy rejects partitioned sources");
  }
  if (metadata.format_kind == MountedSourceFormatKind::kSafetensors && !policy.allow_safetensors) {
    return absl::PermissionDeniedError("mounted-source policy rejects safetensors sources");
  }
  if (metadata.metadata_capability == MountedSourceMetadataCapability::kTensorAware && !policy.allow_tensor_aware) {
    return absl::PermissionDeniedError("mounted-source policy rejects tensor-aware metadata");
  }
  if (metadata.metadata_capability == MountedSourceMetadataCapability::kByteOnly && !policy.allow_byte_only) {
    return absl::PermissionDeniedError("mounted-source policy rejects byte-only metadata");
  }
  return absl::OkStatus();
}

absl::StatusOr<SourceFileFingerprint> stat_file_fingerprint(const std::filesystem::path& path) {
  struct stat st{};
  if (::stat(path.c_str(), &st) != 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("stat failed for ", path.string()));
  }
  SourceFileFingerprint fp;
  fp.inode = static_cast<std::uint64_t>(st.st_ino);
  fp.size = static_cast<std::uint64_t>(st.st_size);
#if defined(__linux__)
  fp.mtime_ns =
      static_cast<std::int64_t>(st.st_mtim.tv_sec) * 1000000000LL + static_cast<std::int64_t>(st.st_mtim.tv_nsec);
#else
  fp.mtime_ns = static_cast<std::int64_t>(st.st_mtime) * 1000000000LL;
#endif
  return fp;
}

struct SourceScanResult {
  SourceFingerprintMap fingerprints;
  std::uint64_t total_bytes{0};
};

absl::StatusOr<SourceScanResult> scan_source(const std::filesystem::path& root) {
  auto context_or = store::loader::get_disk_artifact_context(root);
  if (!context_or.ok()) {
    return normalize_disk_context_status(context_or.status(), root.string());
  }

  SourceScanResult out;
  out.total_bytes = (*context_or)->total_size();
  for (const auto& path : (*context_or)->partition_paths()) {
    auto fp_or = stat_file_fingerprint(path);
    if (!fp_or.ok()) {
      return fp_or.status();
    }
    const auto rel = path.lexically_relative(root).string();
    out.fingerprints.insert_or_assign(rel, *fp_or);
  }
  return out;
}

absl::StatusOr<MountedSourceMetadata> build_mounted_source_metadata_from_context(
    const std::filesystem::path& normalized_path,
    const std::shared_ptr<const DiskArtifactContext>& context) {
  MountedSourceMetadata metadata;
  metadata.normalized_path = normalized_path;
  metadata.format_kind =
      context->is_safetensors() ? MountedSourceFormatKind::kSafetensors : MountedSourceFormatKind::kPartitioned;

  if (!context->is_safetensors() && !context->tensor_index_json_present() && !context->tensor_index_cbor_present()) {
    metadata.metadata_capability = MountedSourceMetadataCapability::kByteOnly;
    metadata.exact_size_bytes = context->total_size();
    metadata.index_info.canonical_index_json =
        store::loading::build_synthetic_payload_canonical_index_json(metadata.exact_size_bytes);
    metadata.index_info.total_size_bytes = metadata.exact_size_bytes;
    metadata.index_info.is_safetensors = false;
    metadata.index_info.source_total_size_bytes = metadata.exact_size_bytes;
  } else {
    auto index_or = context->get_index_info(/*target_device_id=*/0);
    if (!index_or.ok()) {
      return index_or.status();
    }
    metadata.index_info = *index_or;
    metadata.metadata_capability = MountedSourceMetadataCapability::kTensorAware;
  }

  metadata.canonical_index_multihash = metadata.index_info.index_multihash;
  if (metadata.canonical_index_multihash.empty()) {
    auto index_mh_or = common::compute_index_multihash(
        std::optional<std::string>(metadata.index_info.canonical_index_json), /*index_key_hex=*/"");
    if (!index_mh_or.ok()) {
      return index_mh_or.status();
    }
    metadata.canonical_index_multihash = *index_mh_or;
  }
  if (metadata.index_info.source_index_json.has_value()) {
    auto source_index_mh_or =
        common::compute_index_multihash(metadata.index_info.source_index_json, /*index_key_hex=*/"");
    if (!source_index_mh_or.ok()) {
      return source_index_mh_or.status();
    }
    metadata.source_index_multihash = *source_index_mh_or;
  }

  if (metadata.exact_size_bytes == 0) {
    metadata.exact_size_bytes = metadata.index_info.total_size_bytes;
    if (metadata.exact_size_bytes == 0 && metadata.index_info.source_total_size_bytes > 0) {
      metadata.exact_size_bytes = metadata.index_info.source_total_size_bytes;
    }
    if (metadata.exact_size_bytes == 0) {
      metadata.exact_size_bytes = context->total_size();
    }
  }
  metadata.generation = compute_generation_from_index(metadata.index_info.canonical_index_json);

  auto scan_or = scan_source(normalized_path);
  if (!scan_or.ok()) {
    return scan_or.status();
  }
  metadata.file_fingerprints = std::move(scan_or->fingerprints);
  return metadata;
}

std::string build_mounted_snapshot_digest(const MountedSourceMetadata& metadata, std::string_view policy_id) {
  std::vector<std::pair<std::string, SourceFileFingerprint>> ordered_fingerprints;
  ordered_fingerprints.reserve(metadata.file_fingerprints.size());
  for (const auto& entry : metadata.file_fingerprints) {
    ordered_fingerprints.push_back(entry);
  }
  std::ranges::sort(ordered_fingerprints, [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  std::string payload;
  absl::StrAppend(
      &payload,
      metadata.normalized_path.string(),
      "\npolicy=",
      policy_id,
      "\nformat=",
      format_kind_token(metadata.format_kind),
      "\ncapability=",
      static_cast<int>(metadata.metadata_capability),
      "\nindex=",
      metadata.canonical_index_multihash,
      "\nsource_index=",
      metadata.source_index_multihash.value_or(""),
      "\nsize=",
      metadata.exact_size_bytes);
  for (const auto& [relative_path, fingerprint] : ordered_fingerprints) {
    absl::StrAppend(
        &payload,
        "\nfile=",
        relative_path,
        "|inode=",
        fingerprint.inode,
        "|size=",
        fingerprint.size,
        "|mtime_ns=",
        fingerprint.mtime_ns);
  }

  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));
}

std::string mint_msa1_artifact_id(
    std::string_view daemon_session_token,
    std::string_view policy_id,
    const MountedSourceMetadata& metadata) {
  return absl::StrCat(
      common::kMsa1Prefix,
      daemon_session_token,
      "~",
      policy_id,
      "~",
      format_kind_token(metadata.format_kind),
      "~",
      build_mounted_snapshot_digest(metadata, policy_id));
}

std::optional<std::string> maybe_trusted_descriptor_hint(
    const DescriptorMetadata& descriptor,
    const IndexInfo& index_info,
    bool verify_checksums,
    std::string_view canonical_index_multihash) {
  if (!descriptor.found) {
    return std::nullopt;
  }

  if (verify_checksums) {
    auto status = validate_descriptor_against_index(descriptor, index_info, /*verify_checksums=*/true);
    if (!status.ok()) {
      return std::nullopt;
    }
  }

  if (!descriptor.index_multihash.has_value() || descriptor.index_multihash->empty() ||
      !descriptor.data_multihash.has_value() || descriptor.data_multihash->empty() ||
      !descriptor.artifact_id.has_value() || descriptor.artifact_id->empty()) {
    return std::nullopt;
  }
  if (*descriptor.index_multihash != canonical_index_multihash) {
    return std::nullopt;
  }
  const std::string expected_artifact_id =
      absl::StrCat(common::kMi2Prefix, *descriptor.index_multihash, ":", *descriptor.data_multihash);
  if (*descriptor.artifact_id != expected_artifact_id) {
    return std::nullopt;
  }
  return *descriptor.artifact_id;
}

ImportArtifactErrorCode status_to_error_code(const absl::Status& status) {
  if (status.ok()) {
    return ImportArtifactErrorCode::kUnspecified;
  }
  const std::string message(status.message());
  if (absl::StartsWith(message, kSourceMutatedPrefix)) {
    return ImportArtifactErrorCode::kSourceMutated;
  }
  if (absl::IsNotFound(status)) {
    return ImportArtifactErrorCode::kSourceNotFound;
  }
  if (absl::IsPermissionDenied(status)) {
    return ImportArtifactErrorCode::kSourcePermissionDenied;
  }
  if (absl::IsUnavailable(status)) {
    return ImportArtifactErrorCode::kRegistryIoFailure;
  }
  if (absl::IsFailedPrecondition(status)) {
    return ImportArtifactErrorCode::kSourceFormatInvalid;
  }
  return ImportArtifactErrorCode::kSourceFormatInvalid;
}

void record_metric(std::string_view name, std::string_view outcome) {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_store_import_artifact_total");
    if (counter) {
      counter->Add(1, {{"metric", std::string(name)}, {"outcome", std::string(outcome)}});
    }
  } catch (...) {
  }
}

absl::Status descriptor_consistency_check(
    const DescriptorMetadata& descriptor,
    const store::loader::IndexInfo& index,
    bool verify_checksums,
    std::string_view index_multihash,
    std::string_view data_multihash,
    std::string_view artifact_id) {
  if (!descriptor.found) {
    return absl::OkStatus();
  }
  if (verify_checksums) {
    auto st = validate_descriptor_against_index(descriptor, index, /*verify_checksums=*/true);
    if (!st.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("SOURCE_FORMAT_INVALID: descriptor/index mismatch: ", st.message()));
    }
  }
  if (descriptor.index_multihash.has_value() && !descriptor.index_multihash->empty() &&
      *descriptor.index_multihash != index_multihash) {
    return absl::InvalidArgumentError("SOURCE_FORMAT_INVALID: index_multihash mismatch");
  }
  if (descriptor.data_multihash.has_value() && !descriptor.data_multihash->empty() &&
      *descriptor.data_multihash != data_multihash) {
    return absl::InvalidArgumentError("SOURCE_FORMAT_INVALID: data_multihash mismatch");
  }
  if (descriptor.artifact_id.has_value() && !descriptor.artifact_id->empty() &&
      *descriptor.artifact_id != artifact_id) {
    return absl::InvalidArgumentError("SOURCE_FORMAT_INVALID: artifact_id mismatch");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<std::pair<std::string, std::string>>> try_trust_existing_descriptor(
    const DescriptorMetadata& descriptor,
    const store::loader::IndexInfo& index,
    bool verify_checksums,
    std::string_view index_multihash) {
  if (!descriptor.found) {
    return std::optional<std::pair<std::string, std::string>>{};
  }
  if (!descriptor.index_multihash.has_value() || descriptor.index_multihash->empty() ||
      !descriptor.data_multihash.has_value() || descriptor.data_multihash->empty() ||
      !descriptor.artifact_id.has_value() || descriptor.artifact_id->empty()) {
    return std::optional<std::pair<std::string, std::string>>{};
  }

  if (verify_checksums) {
    auto st = validate_descriptor_against_index(descriptor, index, /*verify_checksums=*/true);
    if (!st.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("SOURCE_FORMAT_INVALID: descriptor/index mismatch: ", st.message()));
    }
  }

  if (*descriptor.index_multihash != index_multihash) {
    return std::optional<std::pair<std::string, std::string>>{};
  }

  const std::string expected_artifact_id =
      absl::StrCat("mi2:", *descriptor.index_multihash, ":", *descriptor.data_multihash);
  if (*descriptor.artifact_id != expected_artifact_id) {
    return std::optional<std::pair<std::string, std::string>>{};
  }

  return std::optional<std::pair<std::string, std::string>>(
      std::make_pair(*descriptor.artifact_id, *descriptor.data_multihash));
}

} // namespace

void record_disk_path_denied() {
  record_metric("path_denied", "denied");
}

void record_disk_import_outcome(std::string_view outcome) {
  record_metric("import", outcome);
}

void record_mounted_source_validation_outcome(std::string_view outcome) {
  record_metric("mounted_source_validation", outcome);
}

ImportArtifactErrorCode classify_import_error(const absl::Status& status) {
  return status_to_error_code(status);
}

absl::StatusOr<MountedSourceMetadata> build_mounted_source_metadata(const std::filesystem::path& normalized_path) {
  auto context_or = store::loader::get_disk_artifact_context(normalized_path);
  if (!context_or.ok()) {
    return normalize_disk_context_status(context_or.status(), normalized_path.string());
  }
  return build_mounted_source_metadata_from_context(normalized_path, *context_or);
}

absl::StatusOr<ResolveMountedSourceResult> resolve_mounted_source_artifact(
    const std::filesystem::path& normalized_path,
    bool verify_checksums,
    std::string_view daemon_session_token,
    const MountedSourceAttestationPolicy& policy) {
  auto metadata_or = build_mounted_source_metadata(normalized_path);
  if (!metadata_or.ok()) {
    return metadata_or.status();
  }
  auto policy_status = validate_attestation_policy(policy, *metadata_or);
  if (!policy_status.ok()) {
    return policy_status;
  }
  auto descriptor_or = load_descriptor_metadata(normalized_path);
  if (!descriptor_or.ok()) {
    return descriptor_or.status();
  }
  if (verify_checksums && descriptor_or->found) {
    auto descriptor_status =
        validate_descriptor_against_index(*descriptor_or, metadata_or->index_info, /*verify_checksums=*/true);
    if (!descriptor_status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("SOURCE_FORMAT_INVALID: descriptor/index mismatch: ", descriptor_status.message()));
    }
  }

  ResolveMountedSourceResult result;
  result.normalized_path = normalized_path;
  result.canonical_index_json = metadata_or->index_info.canonical_index_json;
  result.source_index_json = metadata_or->index_info.source_index_json;
  result.canonical_index_multihash = metadata_or->canonical_index_multihash;
  result.source_index_multihash = metadata_or->source_index_multihash;
  result.generation = metadata_or->generation;
  result.descriptor_present = descriptor_or->found;
  result.file_fingerprints = metadata_or->file_fingerprints;
  result.format_kind = metadata_or->format_kind;
  result.metadata_capability = metadata_or->metadata_capability;
  result.validation_mode = policy.validation_mode;
  result.policy_id = policy.policy_id;
  result.snapshot_digest = build_mounted_snapshot_digest(*metadata_or, policy.policy_id);
  result.exact_size_bytes = metadata_or->exact_size_bytes;
  if (policy.descriptor_reuse_mode == MountedSourceDescriptorReuseMode::kTrustedHintOnly) {
    result.trusted_content_artifact_id = maybe_trusted_descriptor_hint(
        *descriptor_or, metadata_or->index_info, verify_checksums, result.canonical_index_multihash);
  }
  result.resolution_strategy = result.trusted_content_artifact_id.has_value()
      ? MountedSourceResolutionStrategy::kAttestedWithTrustedDescriptorHint
      : MountedSourceResolutionStrategy::kAttestedOnly;
  result.artifact_id = mint_msa1_artifact_id(daemon_session_token, policy.policy_id, *metadata_or);
  return result;
}

absl::StatusOr<ImportArtifactFromPathResult> import_artifact_from_path(
    std::string_view path,
    const std::filesystem::path& storage_path,
    bool verify_checksums,
    ImportProgressCallback progress_cb) {
  const auto emit_progress = [&](ImportArtifactPhase phase,
                                 std::uint64_t processed_bytes = 0,
                                 std::uint64_t total_bytes = 0,
                                 bool done = false,
                                 bool error = false,
                                 ImportArtifactErrorCode error_code = ImportArtifactErrorCode::kUnspecified,
                                 std::string message = std::string()) {
    if (!progress_cb) {
      return;
    }
    ImportProgressUpdate update;
    update.phase = phase;
    update.processed_bytes = processed_bytes;
    update.total_bytes = total_bytes;
    update.done = done;
    update.error = error;
    update.error_code = error_code;
    update.message = std::move(message);
    progress_cb(update);
  };

  const auto emit_error = [&](const absl::Status& status) {
    emit_progress(
        ImportArtifactPhase::kError,
        0,
        0,
        /*done=*/true,
        /*error=*/true,
        status_to_error_code(status),
        std::string(status.message()));
  };

  emit_progress(ImportArtifactPhase::kPrepare);
  auto normalized_or = normalize_disk_import_path(path, storage_path);
  if (!normalized_or.ok()) {
    record_disk_path_denied();
    record_disk_import_outcome("invalid_path");
    emit_error(normalized_or.status());
    return normalized_or.status();
  }
  const auto normalized_path = *normalized_or;

  emit_progress(ImportArtifactPhase::kScanSource);
  auto scan_or = scan_source(normalized_path);
  if (!scan_or.ok()) {
    record_disk_import_outcome("scan_failed");
    emit_error(scan_or.status());
    return scan_or.status();
  }

  emit_progress(ImportArtifactPhase::kReadHeaders);
  auto descriptor_or = load_descriptor_metadata(normalized_path);
  if (!descriptor_or.ok()) {
    record_disk_import_outcome("descriptor_failed");
    emit_error(descriptor_or.status());
    return descriptor_or.status();
  }
  const DescriptorMetadata descriptor = *descriptor_or;

  emit_progress(ImportArtifactPhase::kBuildCanonicalIndex);
  auto metadata_or = build_mounted_source_metadata(normalized_path);
  if (!metadata_or.ok()) {
    record_disk_import_outcome("index_failed");
    emit_error(metadata_or.status());
    return metadata_or.status();
  }
  const auto& index = metadata_or->index_info;

  std::string index_multihash = metadata_or->canonical_index_multihash;
  if (index_multihash.empty()) {
    auto index_mh_or =
        common::compute_index_multihash(std::optional<std::string>(index.canonical_index_json), /*index_key_hex=*/"");
    if (!index_mh_or.ok()) {
      record_disk_import_outcome("index_hash_failed");
      emit_error(index_mh_or.status());
      return index_mh_or.status();
    }
    index_multihash = *index_mh_or;
  }

  std::string artifact_id;
  std::string data_multihash;
  bool descriptor_present = descriptor.found;

  auto trusted_descriptor_or = try_trust_existing_descriptor(descriptor, index, verify_checksums, index_multihash);
  if (!trusted_descriptor_or.ok()) {
    record_disk_import_outcome("descriptor_mismatch");
    emit_error(trusted_descriptor_or.status());
    return trusted_descriptor_or.status();
  }

  if (trusted_descriptor_or->has_value()) {
    artifact_id = trusted_descriptor_or->value().first;
    data_multihash = trusted_descriptor_or->value().second;
  } else {
    emit_progress(ImportArtifactPhase::kHashData);
    auto data_mh_or = store::loader::compute_data_multihash_from_disk_dir(
        normalized_path.string(), [&](std::uint64_t processed_bytes, std::uint64_t total_bytes) {
          emit_progress(ImportArtifactPhase::kHashData, processed_bytes, total_bytes);
        });
    if (!data_mh_or.ok()) {
      record_disk_import_outcome("data_hash_failed");
      emit_error(data_mh_or.status());
      return data_mh_or.status();
    }
    data_multihash = *data_mh_or;
    artifact_id = absl::StrCat("mi2:", index_multihash, ":", data_multihash);

    auto consistency =
        descriptor_consistency_check(descriptor, index, verify_checksums, index_multihash, data_multihash, artifact_id);
    if (!consistency.ok()) {
      record_disk_import_outcome("descriptor_mismatch");
      emit_error(consistency);
      return consistency;
    }

    uint64_t total_size = index.total_size_bytes;
    if (total_size == 0 && index.source_total_size_bytes > 0) {
      total_size = index.source_total_size_bytes;
    }
    auto write_status = store::loader::verification::write_descriptor_if_absent(
        normalized_path, index_multihash, data_multihash, total_size, "json");
    if (!write_status.ok()) {
      LOG(WARNING) << "Failed to backfill artifact_descriptor.json for import path '" << normalized_path.string()
                   << "': " << write_status;
    } else {
      descriptor_present = true;
    }
  }

  emit_progress(ImportArtifactPhase::kWriteRegistry);

  ImportArtifactFromPathResult result;
  result.normalized_path = normalized_path;
  result.artifact_id = artifact_id;
  result.canonical_index_json = index.canonical_index_json;
  result.source_index_json = index.source_index_json;
  result.index_multihash = std::move(index_multihash);
  result.data_multihash = data_multihash;
  result.generation = metadata_or->generation;
  result.descriptor_present = descriptor_present;
  result.format_kind = metadata_or->format_kind;
  result.metadata_capability = metadata_or->metadata_capability;
  result.exact_size_bytes = metadata_or->exact_size_bytes;
  result.file_fingerprints = metadata_or->file_fingerprints;

  record_disk_import_outcome("ok");
  emit_progress(ImportArtifactPhase::kDone, scan_or->total_bytes, scan_or->total_bytes, /*done=*/true);
  return result;
}

absl::Status validate_source_fingerprints(
    const std::filesystem::path& source_root,
    const SourceFingerprintMap& expected_fingerprints) {
  auto current_or = scan_source(source_root);
  if (!current_or.ok()) {
    if (absl::IsNotFound(current_or.status())) {
      return absl::FailedPreconditionError(
          absl::StrCat(kSourceMutatedPrefix, " source path missing: ", source_root.string()));
    }
    if (absl::IsPermissionDenied(current_or.status())) {
      return current_or.status();
    }
    return absl::FailedPreconditionError(
        absl::StrCat(kSourceMutatedPrefix, " cannot read source path: ", current_or.status().message()));
  }

  const auto& current = current_or->fingerprints;
  if (current.size() != expected_fingerprints.size()) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            kSourceMutatedPrefix,
            " file count changed (expected=",
            expected_fingerprints.size(),
            ", actual=",
            current.size(),
            ")"));
  }
  for (const auto& [relative_path, expected] : expected_fingerprints) {
    auto it = current.find(relative_path);
    if (it == current.end()) {
      return absl::FailedPreconditionError(absl::StrCat(kSourceMutatedPrefix, " file removed: ", relative_path));
    }
    if (!(it->second == expected)) {
      return absl::FailedPreconditionError(
          absl::StrCat(kSourceMutatedPrefix, " file fingerprint changed: ", relative_path));
    }
  }
  return absl::OkStatus();
}

absl::Status validate_mounted_source_snapshot(
    const std::filesystem::path& source_root,
    std::string_view expected_canonical_index_json,
    const std::optional<std::string>& expected_source_index_json,
    const SourceFingerprintMap& expected_fingerprints) {
  auto metadata_or = build_mounted_source_metadata(source_root);
  if (!metadata_or.ok()) {
    if (absl::IsNotFound(metadata_or.status())) {
      return absl::FailedPreconditionError(
          absl::StrCat(kSourceMutatedPrefix, " source path missing: ", source_root.string()));
    }
    if (absl::IsPermissionDenied(metadata_or.status())) {
      return metadata_or.status();
    }
    return absl::FailedPreconditionError(
        absl::StrCat(
            kSourceMutatedPrefix, " cannot rebuild mounted-source metadata: ", metadata_or.status().message()));
  }
  if (metadata_or->index_info.canonical_index_json != expected_canonical_index_json) {
    return absl::FailedPreconditionError(
        absl::StrCat(kSourceMutatedPrefix, " canonical index changed for ", source_root.string()));
  }
  if (metadata_or->index_info.source_index_json != expected_source_index_json) {
    return absl::FailedPreconditionError(
        absl::StrCat(kSourceMutatedPrefix, " source index changed for ", source_root.string()));
  }
  return validate_source_fingerprints(source_root, expected_fingerprints);
}

} // namespace tensorcast::daemon::materialization_disk_resolve
