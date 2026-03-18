// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_disk_resolve_utils.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <sys/stat.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "daemon/service/controllers/materialization_index_source_utils.h"
#include "daemon/service/controllers/materialization_payload_utils.h"
#include "daemon/util/path_utils.h"
#include "opentelemetry/metrics/provider.h"

#include "core/common/artifact_hash.h"
#include "core/store/materialization/dataplane/metadata/disk_dir_hash.h"
#include "core/store/materialization/dataplane/verification/verification_utils.h"

namespace tensorcast::daemon::materialization_disk_resolve {

namespace {

using materialization_index_source::DescriptorMetadata;
using materialization_index_source::load_descriptor_metadata;
using materialization_index_source::validate_descriptor_against_index;
using materialization_payload::compute_generation_from_index;

constexpr std::string_view kSourceMutatedPrefix = "SOURCE_MUTATED:";

bool is_supported_source_file(std::string_view filename) {
  return filename == "tensor.data" || absl::StartsWith(filename, "tensor.data_") ||
      absl::EndsWith(filename, ".safetensors");
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
  bool has_supported_layout{false};
};

absl::StatusOr<SourceScanResult> scan_source(const std::filesystem::path& root) {
  std::error_code ec;
  const bool exists = std::filesystem::exists(root, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("Failed to access source path: ", root.string()));
  }
  if (!exists) {
    return absl::NotFoundError(absl::StrCat("SOURCE_NOT_FOUND: source path not found: ", root.string()));
  }
  const bool is_dir = std::filesystem::is_directory(root, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("Failed to stat source path: ", root.string()));
  }
  if (!is_dir) {
    return absl::InvalidArgumentError(
        absl::StrCat("SOURCE_FORMAT_INVALID: source is not a directory: ", root.string()));
  }

  SourceScanResult out;
  std::filesystem::recursive_directory_iterator it(root, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("Failed to enumerate source directory: ", root.string()));
  }
  const std::filesystem::recursive_directory_iterator end;
  for (; it != end; it.increment(ec)) {
    if (ec) {
      return absl::ErrnoToStatus(ec.value(), absl::StrCat("Failed to traverse source directory: ", root.string()));
    }
    const auto& entry = *it;
    if (!entry.is_regular_file(ec)) {
      if (ec) {
        return absl::ErrnoToStatus(ec.value(), absl::StrCat("Failed to inspect source entry: ", entry.path().string()));
      }
      continue;
    }
    auto fp_or = stat_file_fingerprint(entry.path());
    if (!fp_or.ok()) {
      return fp_or.status();
    }
    const auto rel = entry.path().lexically_relative(root).string();
    out.total_bytes += fp_or->size;
    out.fingerprints.insert_or_assign(rel, *fp_or);
    if (is_supported_source_file(entry.path().filename().string())) {
      out.has_supported_layout = true;
    }
  }

  if (!out.has_supported_layout) {
    return absl::InvalidArgumentError(
        absl::StrCat(
            "SOURCE_FORMAT_INVALID: source must include at least one *.safetensors or tensor.data* file: ",
            root.string()));
  }
  return out;
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

ImportArtifactErrorCode classify_import_error(const absl::Status& status) {
  return status_to_error_code(status);
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
  auto index_or = store::loader::read_from_artifact_dir(normalized_path, /*target_device_id=*/0);
  if (!index_or.ok()) {
    record_disk_import_outcome("index_failed");
    emit_error(index_or.status());
    return index_or.status();
  }
  auto index = *index_or;

  std::string index_multihash = index.index_multihash;
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

  auto final_scan_or = scan_source(normalized_path);
  if (!final_scan_or.ok()) {
    return final_scan_or.status();
  }

  ImportArtifactFromPathResult result;
  result.normalized_path = normalized_path;
  result.artifact_id = artifact_id;
  result.canonical_index_json = index.canonical_index_json;
  result.source_index_json = index.source_index_json;
  result.index_multihash = std::move(index_multihash);
  result.data_multihash = data_multihash;
  result.generation = compute_generation_from_index(index.canonical_index_json);
  result.descriptor_present = descriptor_present;
  result.file_fingerprints = std::move(final_scan_or->fingerprints);

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

} // namespace tensorcast::daemon::materialization_disk_resolve
