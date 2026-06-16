// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/disk_artifact_service.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "daemon/service/controllers/materialization_request_common_utils.h"
#include "daemon/util/grpc_peer_utils.h"
#include "daemon/util/path_utils.h"
#include "daemon/util/status_utils.h"

namespace tensorcast::daemon {

using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

constexpr double kDefaultImportCacheTtlSeconds = 600.0;
constexpr size_t kDefaultImportCacheMaxEntries = 1024;

using PublicDiskSourcePolicy = DaemonOptions::PublicDiskSourcePolicy;
using TrustedRootPolicy = PublicDiskSourcePolicy::TrustedRootPolicy;

std::string mint_random_hex_token(size_t bytes_len) {
  absl::BitGen gen;
  std::string bytes;
  bytes.resize(bytes_len);
  for (size_t i = 0; i < bytes_len; ++i) {
    bytes[i] = static_cast<char>(absl::Uniform<unsigned int>(gen, 0, 256));
  }
  return absl::BytesToHexString(bytes);
}

std::optional<double> parse_positive_double_env(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return std::nullopt;
  }
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(raw, &end);
  if (end == raw || (end != nullptr && *end != '\0') || errno != 0 || !std::isfinite(value) || value < 0.0) {
    LOG(WARNING) << "Invalid " << name << "=" << raw << "; expected non-negative float";
    return std::nullopt;
  }
  return value;
}

std::optional<size_t> parse_positive_size_env(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return std::nullopt;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  if (end == raw || (end != nullptr && *end != '\0') || errno != 0 ||
      value > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
    LOG(WARNING) << "Invalid " << name << "=" << raw << "; expected non-negative integer";
    return std::nullopt;
  }
  return static_cast<size_t>(value);
}

bool path_has_prefix(const std::filesystem::path& path, const std::filesystem::path& prefix) {
  auto path_it = path.begin();
  for (auto prefix_it = prefix.begin(); prefix_it != prefix.end(); ++prefix_it, ++path_it) {
    if (path_it == path.end() || *path_it != *prefix_it) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<std::filesystem::path> weakly_canonical_or_normalized(const std::filesystem::path& input) {
  std::error_code ec;
  auto normalized = std::filesystem::weakly_canonical(input, ec);
  if (!ec) {
    return normalized;
  }
  normalized = input.lexically_normal();
  if (normalized.empty()) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("Failed to canonicalize path: ", input.string()));
  }
  return normalized;
}

std::string default_public_disk_source_policy_id(const std::filesystem::path& root_path) {
  if (root_path.empty()) {
    return "trusted_absolute_local_path";
  }
  const std::string root = root_path.string();
  const std::vector<uint8_t> digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(root.data()), root.size()));
  const std::string hex =
      absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));
  return absl::StrCat("trusted_storage_root_", hex.substr(0, 16));
}

bool format_allowed(
    const TrustedRootPolicy& policy,
    materialization_disk_resolve::MountedSourceFormatKind format_kind) {
  if (policy.allowed_formats.empty()) {
    return true;
  }
  const auto expected = format_kind == materialization_disk_resolve::MountedSourceFormatKind::kPartitioned
      ? PublicDiskSourcePolicy::Format::kPartitioned
      : PublicDiskSourcePolicy::Format::kSafetensors;
  return std::find(policy.allowed_formats.begin(), policy.allowed_formats.end(), expected) !=
      policy.allowed_formats.end();
}

bool metadata_capability_allowed(
    const TrustedRootPolicy& policy,
    materialization_disk_resolve::MountedSourceMetadataCapability metadata_capability) {
  if (policy.allowed_metadata_capabilities.empty()) {
    return true;
  }
  const auto expected =
      metadata_capability == materialization_disk_resolve::MountedSourceMetadataCapability::kTensorAware
      ? PublicDiskSourcePolicy::MetadataCapability::kTensorAware
      : PublicDiskSourcePolicy::MetadataCapability::kByteOnly;
  return std::find(
             policy.allowed_metadata_capabilities.begin(), policy.allowed_metadata_capabilities.end(), expected) !=
      policy.allowed_metadata_capabilities.end();
}

materialization_disk_resolve::MountedSourceAttestationPolicy build_attestation_policy(const TrustedRootPolicy& policy) {
  materialization_disk_resolve::MountedSourceAttestationPolicy out;
  out.policy_id = policy.policy_id;
  out.allow_partitioned = format_allowed(policy, materialization_disk_resolve::MountedSourceFormatKind::kPartitioned);
  out.allow_safetensors = format_allowed(policy, materialization_disk_resolve::MountedSourceFormatKind::kSafetensors);
  out.allow_tensor_aware =
      metadata_capability_allowed(policy, materialization_disk_resolve::MountedSourceMetadataCapability::kTensorAware);
  out.allow_byte_only =
      metadata_capability_allowed(policy, materialization_disk_resolve::MountedSourceMetadataCapability::kByteOnly);
  out.descriptor_reuse_mode = policy.descriptor_reuse_mode == PublicDiskSourcePolicy::DescriptorReuseMode::kDisabled
      ? materialization_disk_resolve::MountedSourceDescriptorReuseMode::kDisabled
      : materialization_disk_resolve::MountedSourceDescriptorReuseMode::kTrustedHintOnly;
  out.validation_mode = materialization_disk_resolve::MountedSourceValidationMode::kValidateBeforeRead;
  out.lightweight_attestation_enabled = policy.lightweight_attestation_enabled;
  return out;
}

std::string mounted_source_policy_cache_key(
    const materialization_disk_resolve::MountedSourceAttestationPolicy& policy) {
  return absl::StrCat(
      policy.policy_id,
      "|partitioned=",
      policy.allow_partitioned ? "1" : "0",
      "|safetensors=",
      policy.allow_safetensors ? "1" : "0",
      "|tensor_aware=",
      policy.allow_tensor_aware ? "1" : "0",
      "|byte_only=",
      policy.allow_byte_only ? "1" : "0",
      "|descriptor_reuse=",
      static_cast<int>(policy.descriptor_reuse_mode),
      "|validation=",
      static_cast<int>(policy.validation_mode),
      "|lightweight=",
      policy.lightweight_attestation_enabled ? "1" : "0");
}

absl::Status descriptor_presence_matches(
    const std::filesystem::path& normalized_path,
    bool expected_descriptor_present) {
  std::error_code ec;
  const bool descriptor_present = std::filesystem::exists(normalized_path / "artifact_descriptor.json", ec);
  if (ec) {
    return absl::ErrnoToStatus(
        ec.value(), absl::StrCat("SOURCE_MUTATED: cannot stat artifact_descriptor.json: ", normalized_path.string()));
  }
  if (descriptor_present != expected_descriptor_present) {
    return absl::FailedPreconditionError("SOURCE_MUTATED: artifact_descriptor.json presence changed");
  }
  return absl::OkStatus();
}

absl::Status validate_descriptorless_cached_mounted_source(
    const std::filesystem::path& normalized_path,
    const materialization_disk_resolve::ResolveMountedSourceResult& resolved) {
  if (resolved.descriptor_present) {
    return absl::FailedPreconditionError("descriptor-backed mounted source is not resolve-cache eligible");
  }
  if (resolved.normalized_path != normalized_path) {
    return absl::FailedPreconditionError("mounted-source resolve cache path mismatch");
  }
  auto descriptor_status = descriptor_presence_matches(normalized_path, /*expected_descriptor_present=*/false);
  if (!descriptor_status.ok()) {
    return descriptor_status;
  }
  return materialization_disk_resolve::validate_source_fingerprints(normalized_path, resolved.file_fingerprints);
}

struct ResolvedDiskPathPolicy {
  std::filesystem::path normalized_path;
  TrustedRootPolicy policy;
};

absl::StatusOr<ResolvedDiskPathPolicy> resolve_disk_path_policy(
    std::string_view request_path,
    const PublicDiskSourcePolicy& policy) {
  const std::filesystem::path input(request_path);
  if (request_path.empty()) {
    return absl::InvalidArgumentError("path is required");
  }

  if (!input.is_absolute()) {
    if (policy.trusted_root_policies.empty()) {
      return absl::InvalidArgumentError("relative disk path requires a configured trusted root policy");
    }
    if (policy.trusted_root_policies.size() != 1) {
      return absl::InvalidArgumentError(
          "relative disk path is ambiguous across multiple trusted root policies; use an absolute path");
    }
    const auto& trusted_root = policy.trusted_root_policies.front();
    auto normalized_or = normalize_disk_path(request_path, trusted_root.root_path);
    if (!normalized_or.ok()) {
      return normalized_or.status();
    }
    return ResolvedDiskPathPolicy{
        .normalized_path = *normalized_or,
        .policy = trusted_root,
    };
  }

  auto normalized_abs_or = weakly_canonical_or_normalized(input);
  if (!normalized_abs_or.ok()) {
    return normalized_abs_or.status();
  }

  const TrustedRootPolicy* best_match = nullptr;
  std::size_t best_depth = 0;
  for (const auto& trusted_root : policy.trusted_root_policies) {
    if (trusted_root.root_path.empty()) {
      continue;
    }
    if (!path_has_prefix(*normalized_abs_or, trusted_root.root_path)) {
      continue;
    }
    const std::size_t depth =
        static_cast<std::size_t>(std::distance(trusted_root.root_path.begin(), trusted_root.root_path.end()));
    if (best_match == nullptr || depth > best_depth) {
      best_match = &trusted_root;
      best_depth = depth;
    }
  }

  if (best_match != nullptr) {
    auto normalized_or = normalize_disk_path(request_path, best_match->root_path);
    if (!normalized_or.ok()) {
      return normalized_or.status();
    }
    return ResolvedDiskPathPolicy{
        .normalized_path = *normalized_or,
        .policy = *best_match,
    };
  }

  return absl::PermissionDeniedError(absl::StrCat("disk path is outside configured trusted roots: ", request_path));
}

std::string artifact_id_kind_attr(std::string_view artifact_id) {
  switch (common::infer_artifact_id_kind(artifact_id)) {
    case common::ArtifactIdKind::kMi2:
      return "mi2";
    case common::ArtifactIdKind::kCgid:
      return "cgid";
    case common::ArtifactIdKind::kMsa1:
      return "msa1";
    case common::ArtifactIdKind::kUnspecified:
    default:
      return "unknown";
  }
}

std::string trusted_content_hint_kind_attr(const std::optional<std::string>& artifact_id) {
  if (!artifact_id.has_value() || artifact_id->empty()) {
    return "none";
  }
  return artifact_id_kind_attr(*artifact_id);
}

std::string metadata_capability_attr(materialization_disk_resolve::MountedSourceMetadataCapability capability) {
  switch (capability) {
    case materialization_disk_resolve::MountedSourceMetadataCapability::kTensorAware:
      return "tensor_aware";
    case materialization_disk_resolve::MountedSourceMetadataCapability::kByteOnly:
      return "byte_only";
    case materialization_disk_resolve::MountedSourceMetadataCapability::kUnspecified:
    default:
      return "unspecified";
  }
}

std::string resolution_strategy_attr(materialization_disk_resolve::MountedSourceResolutionStrategy strategy) {
  switch (strategy) {
    case materialization_disk_resolve::MountedSourceResolutionStrategy::kAttestedOnly:
      return "attested_only";
    case materialization_disk_resolve::MountedSourceResolutionStrategy::kAttestedWithTrustedDescriptorHint:
      return "attested_with_trusted_descriptor_hint";
    case materialization_disk_resolve::MountedSourceResolutionStrategy::kUnspecified:
    default:
      return "unspecified";
  }
}

std::string validation_mode_attr(materialization_disk_resolve::MountedSourceValidationMode mode) {
  switch (mode) {
    case materialization_disk_resolve::MountedSourceValidationMode::kValidateBeforeRead:
      return "validate_before_read";
    case materialization_disk_resolve::MountedSourceValidationMode::kUnspecified:
    default:
      return "unspecified";
  }
}

v2::ImportArtifactPhase to_proto_phase(materialization_disk_resolve::ImportArtifactPhase phase) {
  using materialization_disk_resolve::ImportArtifactPhase;
  switch (phase) {
    case ImportArtifactPhase::kPrepare:
      return v2::IMPORT_ARTIFACT_PHASE_PREPARE;
    case ImportArtifactPhase::kScanSource:
      return v2::IMPORT_ARTIFACT_PHASE_SCAN_SOURCE;
    case ImportArtifactPhase::kReadHeaders:
      return v2::IMPORT_ARTIFACT_PHASE_READ_HEADERS;
    case ImportArtifactPhase::kBuildCanonicalIndex:
      return v2::IMPORT_ARTIFACT_PHASE_BUILD_CANONICAL_INDEX;
    case ImportArtifactPhase::kHashData:
      return v2::IMPORT_ARTIFACT_PHASE_HASH_DATA;
    case ImportArtifactPhase::kWriteRegistry:
      return v2::IMPORT_ARTIFACT_PHASE_WRITE_REGISTRY;
    case ImportArtifactPhase::kDone:
      return v2::IMPORT_ARTIFACT_PHASE_DONE;
    case ImportArtifactPhase::kError:
      return v2::IMPORT_ARTIFACT_PHASE_ERROR;
  }
  return v2::IMPORT_ARTIFACT_PHASE_UNSPECIFIED;
}

v2::ImportArtifactErrorCode to_proto_error_code(materialization_disk_resolve::ImportArtifactErrorCode code) {
  using materialization_disk_resolve::ImportArtifactErrorCode;
  switch (code) {
    case ImportArtifactErrorCode::kSourceNotFound:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_SOURCE_NOT_FOUND;
    case ImportArtifactErrorCode::kSourcePermissionDenied:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_SOURCE_PERMISSION_DENIED;
    case ImportArtifactErrorCode::kSourceFormatInvalid:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_SOURCE_FORMAT_INVALID;
    case ImportArtifactErrorCode::kSourceMutated:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_SOURCE_MUTATED;
    case ImportArtifactErrorCode::kRegistryIoFailure:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_REGISTRY_IO_FAILURE;
    case ImportArtifactErrorCode::kPolicyDeniedNonLocalPeer:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_POLICY_DENIED_NON_LOCAL_PEER;
    case ImportArtifactErrorCode::kUnspecified:
    default:
      return v2::IMPORT_ARTIFACT_ERROR_CODE_UNSPECIFIED;
  }
}

double to_progress_percent(std::uint64_t processed_bytes, std::uint64_t total_bytes, bool done) {
  if (total_bytes == 0) {
    return done ? 100.0 : 0.0;
  }
  const double percent = (100.0 * static_cast<double>(processed_bytes)) / static_cast<double>(total_bytes);
  return std::clamp(percent, 0.0, 100.0);
}

void fill_import_response(
    const materialization_disk_resolve::ImportArtifactFromPathResult& imported,
    v2::ImportArtifactFromPathResponse& resp) {
  resp.set_artifact_id(imported.artifact_id);
  resp.set_canonical_index_bytes(imported.canonical_index_json);
  resp.set_generation(imported.generation);
  resp.set_import_state(v2::IMPORT_ARTIFACT_STATE_READY);
}

void fill_promote_mounted_source_response(
    const materialization_disk_resolve::ImportArtifactFromPathResult& imported,
    std::string_view source_artifact_id,
    v2::PromoteMountedSourceArtifactResponse& resp) {
  resp.set_artifact_id(imported.artifact_id);
  resp.set_canonical_index_bytes(imported.canonical_index_json);
  resp.set_generation(imported.generation);
  resp.set_import_state(v2::IMPORT_ARTIFACT_STATE_READY);
  resp.set_source_artifact_id(std::string(source_artifact_id));
}

void fill_public_disk_source_response(
    const materialization_disk_resolve::ResolveMountedSourceResult& resolved,
    bool verify_checksums,
    v2::ResolvePublicDiskSourceResponse& resp) {
  auto* source = resp.mutable_source();
  source->set_path(resolved.normalized_path.string());
  source->set_canonical_index_bytes(resolved.canonical_index_json);
  source->set_artifact_id(resolved.artifact_id);
  source->set_generation(resolved.generation);
  source->set_verify_checksums(verify_checksums);
  if (resolved.trusted_content_artifact_id.has_value()) {
    source->set_trusted_content_artifact_id(*resolved.trusted_content_artifact_id);
  }
  if (resolved.source_index_json.has_value()) {
    source->set_source_index_bytes(*resolved.source_index_json);
  }
  switch (resolved.format_kind) {
    case materialization_disk_resolve::MountedSourceFormatKind::kPartitioned:
      source->set_format_kind(v2::DISK_SOURCE_FORMAT_KIND_PARTITIONED);
      break;
    case materialization_disk_resolve::MountedSourceFormatKind::kSafetensors:
      source->set_format_kind(v2::DISK_SOURCE_FORMAT_KIND_SAFETENSORS);
      break;
    case materialization_disk_resolve::MountedSourceFormatKind::kUnspecified:
    default:
      source->set_format_kind(v2::DISK_SOURCE_FORMAT_KIND_UNSPECIFIED);
      break;
  }
  switch (resolved.metadata_capability) {
    case materialization_disk_resolve::MountedSourceMetadataCapability::kTensorAware:
      source->set_metadata_capability(v2::DISK_METADATA_CAPABILITY_TENSOR_AWARE);
      break;
    case materialization_disk_resolve::MountedSourceMetadataCapability::kByteOnly:
      source->set_metadata_capability(v2::DISK_METADATA_CAPABILITY_BYTE_ONLY);
      break;
    case materialization_disk_resolve::MountedSourceMetadataCapability::kUnspecified:
    default:
      source->set_metadata_capability(v2::DISK_METADATA_CAPABILITY_UNSPECIFIED);
      break;
  }
  switch (resolved.resolution_strategy) {
    case materialization_disk_resolve::MountedSourceResolutionStrategy::kAttestedOnly:
      source->set_resolution_strategy(v2::DISK_RESOLUTION_STRATEGY_ATTESTED_ONLY);
      break;
    case materialization_disk_resolve::MountedSourceResolutionStrategy::kAttestedWithTrustedDescriptorHint:
      source->set_resolution_strategy(v2::DISK_RESOLUTION_STRATEGY_ATTESTED_WITH_TRUSTED_DESCRIPTOR_HINT);
      break;
    case materialization_disk_resolve::MountedSourceResolutionStrategy::kUnspecified:
    default:
      source->set_resolution_strategy(v2::DISK_RESOLUTION_STRATEGY_UNSPECIFIED);
      break;
  }
  switch (resolved.validation_mode) {
    case materialization_disk_resolve::MountedSourceValidationMode::kValidateBeforeRead:
      source->set_validation_mode(v2::DISK_VALIDATION_MODE_VALIDATE_BEFORE_READ);
      break;
    case materialization_disk_resolve::MountedSourceValidationMode::kUnspecified:
    default:
      source->set_validation_mode(v2::DISK_VALIDATION_MODE_UNSPECIFIED);
      break;
  }
  source->set_policy_id(resolved.policy_id);
  source->set_exact_size_bytes(resolved.exact_size_bytes);
}

grpc::Status to_import_grpc_status(const absl::Status& status) {
  using materialization_disk_resolve::ImportArtifactErrorCode;
  const auto code = materialization_disk_resolve::classify_import_error(status);
  switch (code) {
    case ImportArtifactErrorCode::kSourceNotFound:
      return {StatusCode::NOT_FOUND, std::string(status.message())};
    case ImportArtifactErrorCode::kSourcePermissionDenied:
      return {StatusCode::PERMISSION_DENIED, std::string(status.message())};
    case ImportArtifactErrorCode::kSourceFormatInvalid:
      return {StatusCode::INVALID_ARGUMENT, std::string(status.message())};
    case ImportArtifactErrorCode::kSourceMutated:
      return {StatusCode::FAILED_PRECONDITION, std::string(status.message())};
    case ImportArtifactErrorCode::kRegistryIoFailure:
      return {StatusCode::UNAVAILABLE, std::string(status.message())};
    case ImportArtifactErrorCode::kPolicyDeniedNonLocalPeer:
      return {StatusCode::PERMISSION_DENIED, std::string(status.message())};
    case ImportArtifactErrorCode::kUnspecified:
    default:
      return to_grpc_status(status);
  }
}

materialization_disk_resolve::ImportArtifactErrorCode grpc_status_to_import_error(const grpc::Status& status) {
  using materialization_disk_resolve::ImportArtifactErrorCode;
  switch (status.error_code()) {
    case StatusCode::NOT_FOUND:
      return ImportArtifactErrorCode::kSourceNotFound;
    case StatusCode::PERMISSION_DENIED:
      return ImportArtifactErrorCode::kSourcePermissionDenied;
    case StatusCode::INVALID_ARGUMENT:
      return ImportArtifactErrorCode::kSourceFormatInvalid;
    case StatusCode::FAILED_PRECONDITION:
      return ImportArtifactErrorCode::kSourceMutated;
    case StatusCode::UNAVAILABLE:
      return ImportArtifactErrorCode::kRegistryIoFailure;
    default:
      return ImportArtifactErrorCode::kUnspecified;
  }
}

ArtifactSourceRegistry::FingerprintMap to_registry_fingerprints(
    const materialization_disk_resolve::SourceFingerprintMap& fingerprints) {
  ArtifactSourceRegistry::FingerprintMap out;
  out.reserve(fingerprints.size());
  for (const auto& [path, fp] : fingerprints) {
    out.insert_or_assign(
        path,
        ArtifactSourceRegistry::SourceFileFingerprint{
            .inode = fp.inode,
            .size = fp.size,
            .mtime_ns = fp.mtime_ns,
        });
  }
  return out;
}

ArtifactSourceRegistry::SourceFormatKind to_registry_source_format(
    materialization_disk_resolve::MountedSourceFormatKind format_kind) {
  switch (format_kind) {
    case materialization_disk_resolve::MountedSourceFormatKind::kPartitioned:
      return ArtifactSourceRegistry::SourceFormatKind::kPartitioned;
    case materialization_disk_resolve::MountedSourceFormatKind::kSafetensors:
      return ArtifactSourceRegistry::SourceFormatKind::kSafetensors;
    case materialization_disk_resolve::MountedSourceFormatKind::kUnspecified:
    default:
      return ArtifactSourceRegistry::SourceFormatKind::kUnspecified;
  }
}

void upsert_local_import_binding(
    ArtifactSourceRegistry& source_registry,
    const materialization_disk_resolve::ImportArtifactFromPathResult& imported) {
  source_registry.upsert_binding(
      imported.artifact_id,
      ArtifactSourceRegistry::Entry{
          .source_kind = ArtifactSourceRegistry::SourceKind::kLocalImport,
          .canonical_source_path = imported.normalized_path.string(),
          .canonical_index_json = imported.canonical_index_json,
          .source_index_json = imported.source_index_json,
          .source_disk_path = imported.normalized_path.string(),
          .source_format_kind = to_registry_source_format(imported.format_kind),
          .descriptor_present = imported.descriptor_present,
          .index_multihash = imported.index_multihash,
          .data_multihash = imported.data_multihash,
          .generation = imported.generation,
          .tensor_aware_metadata = imported.metadata_capability ==
              materialization_disk_resolve::MountedSourceMetadataCapability::kTensorAware,
          .file_fingerprints = to_registry_fingerprints(imported.file_fingerprints),
          .created_at = absl::Now(),
          .updated_at = absl::Now(),
      });
}

} // namespace

DiskArtifactService::DiskArtifactService(Dep d)
    : d_(std::move(d)),
      daemon_session_token_(mint_random_hex_token(8)),
      import_cache_ttl_(import_cache_ttl_from_env()),
      import_cache_max_entries_(import_cache_max_entries_from_env()) {
  if (!d_.storage_path.empty()) {
    std::error_code ec;
    storage_path_ = std::filesystem::weakly_canonical(d_.storage_path, ec);
    if (ec) {
      ec.clear();
      storage_path_ = d_.storage_path.lexically_normal();
    }
  }
  if (d_.public_disk_source_policy.trusted_root_policies.empty()) {
    if (!storage_path_.empty()) {
      d_.public_disk_source_policy.trusted_root_policies.push_back(
          TrustedRootPolicy{
              .policy_id = mounted_source_policy_id(),
              .root_path = storage_path_,
          });
    }
  }
  for (auto& trusted_root : d_.public_disk_source_policy.trusted_root_policies) {
    if (!trusted_root.root_path.empty()) {
      auto normalized_or = weakly_canonical_or_normalized(trusted_root.root_path);
      if (normalized_or.ok()) {
        trusted_root.root_path = *normalized_or;
      }
    }
    if (trusted_root.policy_id.empty()) {
      trusted_root.policy_id = default_public_disk_source_policy_id(trusted_root.root_path);
    }
  }
}

absl::Status DiskArtifactService::ensure_artifact_metadata_registered(
    const materialization_disk_resolve::ImportArtifactFromPathResult& imported) const {
  if (d_.global_store_client == nullptr || !d_.global_store_client->is_connected()) {
    return absl::OkStatus();
  }
  if (imported.artifact_id.empty() || imported.index_multihash.empty() || imported.canonical_index_json.empty()) {
    return absl::FailedPreconditionError("imported artifact metadata is incomplete");
  }

  tensorcast::common::v1::ArtifactDescriptor descriptor;
  descriptor.set_artifact_id(imported.artifact_id);
  descriptor.set_index_multihash(imported.index_multihash);
  descriptor.set_data_multihash(imported.data_multihash);
  descriptor.set_schema_version("v3");
  descriptor.set_encoding("json");
  descriptor.set_id_kind(tensorcast::common::v1::ArtifactIdKind::ARTIFACT_ID_KIND_MI2);
  return d_.global_store_client->upsert_artifact_metadata(descriptor, imported.canonical_index_json);
}

absl::Duration DiskArtifactService::import_cache_ttl_from_env() {
  const auto parsed = parse_positive_double_env("TENSORCAST_IMPORT_ARTIFACT_CACHE_TTL_SECONDS");
  return absl::Seconds(parsed.value_or(kDefaultImportCacheTtlSeconds));
}

size_t DiskArtifactService::import_cache_max_entries_from_env() {
  return parse_positive_size_env("TENSORCAST_IMPORT_ARTIFACT_CACHE_MAX_ENTRIES")
      .value_or(kDefaultImportCacheMaxEntries);
}

std::string DiskArtifactService::import_cache_key_for_path(
    const std::filesystem::path& normalized_path,
    bool verify_checksums) const {
  return absl::StrCat(normalized_path.string(), "|verify=", verify_checksums ? "1" : "0");
}

std::string DiskArtifactService::mounted_source_resolve_cache_key_for_path(
    const std::filesystem::path& normalized_path,
    bool verify_checksums,
    std::string_view policy_cache_key) const {
  return absl::StrCat(normalized_path.string(), "|verify=", verify_checksums ? "1" : "0", "|policy=", policy_cache_key);
}

std::string DiskArtifactService::mounted_source_policy_id() const {
  if (storage_path_.empty()) {
    return "trusted_absolute_local_path";
  }
  const std::string storage_root = storage_path_.string();
  const std::vector<uint8_t> digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(storage_root.data()), storage_root.size()));
  const std::string hex =
      absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));
  return absl::StrCat("trusted_storage_root_", hex.substr(0, 16));
}

void DiskArtifactService::prune_expired_cache_locked(absl::Time now) {
  if (import_cache_ttl_ <= absl::ZeroDuration()) {
    import_cache_.clear();
    return;
  }
  for (auto it = import_cache_.begin(); it != import_cache_.end();) {
    if (now - it->second.cached_at > import_cache_ttl_) {
      auto erase_it = it;
      ++it;
      import_cache_.erase(erase_it);
    } else {
      ++it;
    }
  }
}

void DiskArtifactService::enforce_cache_capacity_locked() {
  if (import_cache_max_entries_ == 0) {
    import_cache_.clear();
    return;
  }
  while (import_cache_.size() > import_cache_max_entries_) {
    auto oldest_it = import_cache_.begin();
    for (auto it = import_cache_.begin(); it != import_cache_.end(); ++it) {
      if (it->second.cached_at < oldest_it->second.cached_at) {
        oldest_it = it;
      }
    }
    import_cache_.erase(oldest_it);
  }
}

void DiskArtifactService::prune_expired_mounted_source_resolve_cache_locked(absl::Time now) {
  if (import_cache_ttl_ <= absl::ZeroDuration()) {
    mounted_source_resolve_cache_.clear();
    return;
  }
  for (auto it = mounted_source_resolve_cache_.begin(); it != mounted_source_resolve_cache_.end();) {
    if (now - it->second.cached_at > import_cache_ttl_) {
      auto erase_it = it;
      ++it;
      mounted_source_resolve_cache_.erase(erase_it);
    } else {
      ++it;
    }
  }
}

void DiskArtifactService::enforce_mounted_source_resolve_cache_capacity_locked() {
  if (import_cache_max_entries_ == 0) {
    mounted_source_resolve_cache_.clear();
    return;
  }
  while (mounted_source_resolve_cache_.size() > import_cache_max_entries_) {
    auto oldest_it = mounted_source_resolve_cache_.begin();
    for (auto it = mounted_source_resolve_cache_.begin(); it != mounted_source_resolve_cache_.end(); ++it) {
      if (it->second.cached_at < oldest_it->second.cached_at) {
        oldest_it = it;
      }
    }
    mounted_source_resolve_cache_.erase(oldest_it);
  }
}

absl::StatusOr<materialization_disk_resolve::ImportArtifactFromPathResult> DiskArtifactService::
    import_artifact_from_path_cached(
        const std::filesystem::path& normalized_path,
        bool verify_checksums,
        materialization_disk_resolve::ImportProgressCallback progress_cb) {
  const std::string key = import_cache_key_for_path(normalized_path, verify_checksums);

  std::shared_ptr<InflightImport> inflight;
  bool is_leader = false;
  {
    absl::MutexLock lock(&import_mu_);
    const absl::Time now = absl::Now();
    prune_expired_cache_locked(now);

    auto cache_it = import_cache_.find(key);
    if (cache_it != import_cache_.end()) {
      materialization_disk_resolve::record_disk_import_outcome("cache_hit");
      if (progress_cb) {
        materialization_disk_resolve::ImportProgressUpdate prep;
        prep.phase = materialization_disk_resolve::ImportArtifactPhase::kPrepare;
        prep.message = "served from daemon cache";
        progress_cb(prep);

        materialization_disk_resolve::ImportProgressUpdate done;
        done.phase = materialization_disk_resolve::ImportArtifactPhase::kDone;
        done.done = true;
        progress_cb(done);
      }
      return cache_it->second.imported;
    }

    auto [it, inserted] = inflight_imports_.try_emplace(key, std::make_shared<InflightImport>());
    inflight = it->second;
    is_leader = inserted;
  }

  if (!is_leader) {
    if (progress_cb) {
      materialization_disk_resolve::ImportProgressUpdate wait;
      wait.phase = materialization_disk_resolve::ImportArtifactPhase::kPrepare;
      wait.message = "waiting for inflight import";
      progress_cb(wait);
    }

    std::uint64_t seen_progress_version = 0;
    for (;;) {
      materialization_disk_resolve::ImportProgressUpdate progress_update;
      bool has_progress_update = false;
      bool done = false;
      absl::StatusOr<materialization_disk_resolve::ImportArtifactFromPathResult> imported_or(
          absl::UnknownError("inflight import incomplete"));
      {
        absl::MutexLock wait_lock(&inflight->mu);
        while (!inflight->done && inflight->progress_version <= seen_progress_version) {
          inflight->cv.Wait(&inflight->mu);
        }
        if (inflight->progress_version > seen_progress_version && inflight->has_progress) {
          seen_progress_version = inflight->progress_version;
          progress_update = inflight->latest_progress;
          has_progress_update = true;
        }
        if (inflight->done) {
          done = true;
          imported_or = inflight->imported;
        }
      }
      if (has_progress_update && progress_cb) {
        progress_cb(progress_update);
      }
      if (done) {
        return imported_or;
      }
    }
  }

  const auto publish_progress = [&](const materialization_disk_resolve::ImportProgressUpdate& update) {
    {
      absl::MutexLock progress_lock(&inflight->mu);
      inflight->latest_progress = update;
      inflight->has_progress = true;
      inflight->progress_version += 1;
      inflight->cv.SignalAll();
    }
    if (progress_cb) {
      progress_cb(update);
    }
  };

  auto imported_or = materialization_disk_resolve::import_artifact_from_path(
      normalized_path.string(), std::filesystem::path(), verify_checksums, publish_progress);

  bool terminal_emitted = false;
  {
    absl::MutexLock progress_lock(&inflight->mu);
    terminal_emitted = inflight->has_progress && (inflight->latest_progress.done || inflight->latest_progress.error);
  }
  if (!terminal_emitted) {
    materialization_disk_resolve::ImportProgressUpdate terminal;
    if (imported_or.ok()) {
      terminal.phase = materialization_disk_resolve::ImportArtifactPhase::kDone;
      terminal.done = true;
    } else {
      terminal.phase = materialization_disk_resolve::ImportArtifactPhase::kError;
      terminal.done = true;
      terminal.error = true;
      terminal.error_code = materialization_disk_resolve::classify_import_error(imported_or.status());
      terminal.message = std::string(imported_or.status().message());
    }
    publish_progress(terminal);
  }

  {
    absl::MutexLock wait_lock(&inflight->mu);
    inflight->imported = imported_or;
    inflight->done = true;
    inflight->cv.SignalAll();
  }

  {
    absl::MutexLock lock(&import_mu_);
    inflight_imports_.erase(key);
    if (imported_or.ok() && import_cache_ttl_ > absl::ZeroDuration()) {
      import_cache_.insert_or_assign(
          key,
          ImportCacheEntry{
              .imported = *imported_or,
              .cached_at = absl::Now(),
          });
      enforce_cache_capacity_locked();
    }
  }

  return imported_or;
}

absl::StatusOr<materialization_disk_resolve::ResolveMountedSourceResult> DiskArtifactService::
    resolve_mounted_source_artifact_cached(
        const std::filesystem::path& normalized_path,
        bool verify_checksums,
        const materialization_disk_resolve::MountedSourceAttestationPolicy& policy) {
  const std::string key = mounted_source_resolve_cache_key_for_path(
      normalized_path, verify_checksums, mounted_source_policy_cache_key(policy));

  for (;;) {
    std::optional<materialization_disk_resolve::ResolveMountedSourceResult> cached;
    std::shared_ptr<InflightMountedSourceResolve> inflight;
    bool is_leader = false;
    {
      absl::MutexLock lock(&mounted_source_resolve_mu_);
      const absl::Time now = absl::Now();
      prune_expired_mounted_source_resolve_cache_locked(now);

      auto cache_it = mounted_source_resolve_cache_.find(key);
      if (cache_it != mounted_source_resolve_cache_.end()) {
        cached = cache_it->second.resolved;
      } else {
        auto [it, inserted] =
            inflight_mounted_source_resolves_.try_emplace(key, std::make_shared<InflightMountedSourceResolve>());
        inflight = it->second;
        is_leader = inserted;
      }
    }

    if (cached.has_value()) {
      auto validation_status = validate_descriptorless_cached_mounted_source(normalized_path, *cached);
      if (validation_status.ok()) {
        materialization_disk_resolve::record_disk_import_outcome("mounted_source_resolve_cache_hit");
        return *cached;
      }
      materialization_disk_resolve::record_disk_import_outcome("mounted_source_resolve_cache_stale");
      {
        absl::MutexLock lock(&mounted_source_resolve_mu_);
        mounted_source_resolve_cache_.erase(key);
      }
      const bool erased = d_.source_registry.erase_binding(cached->artifact_id);
      (void)erased;
      continue;
    }

    if (!is_leader) {
      materialization_disk_resolve::record_disk_import_outcome("mounted_source_resolve_inflight_join");
      absl::StatusOr<materialization_disk_resolve::ResolveMountedSourceResult> resolved_or(
          absl::UnknownError("inflight mounted-source resolve incomplete"));
      {
        absl::MutexLock wait_lock(&inflight->mu);
        while (!inflight->done) {
          inflight->cv.Wait(&inflight->mu);
        }
        resolved_or = inflight->resolved;
      }
      return resolved_or;
    }

    materialization_disk_resolve::record_disk_import_outcome("mounted_source_resolve_cache_miss");
    auto resolved_or = materialization_disk_resolve::resolve_mounted_source_artifact(
        normalized_path, verify_checksums, daemon_session_token_, policy);

    {
      absl::MutexLock wait_lock(&inflight->mu);
      inflight->resolved = resolved_or;
      inflight->done = true;
      inflight->cv.SignalAll();
    }

    {
      absl::MutexLock lock(&mounted_source_resolve_mu_);
      inflight_mounted_source_resolves_.erase(key);
      if (resolved_or.ok() && !resolved_or->descriptor_present && import_cache_ttl_ > absl::ZeroDuration()) {
        mounted_source_resolve_cache_.insert_or_assign(
            key,
            MountedSourceResolveCacheEntry{
                .resolved = *resolved_or,
                .cached_at = absl::Now(),
            });
        enforce_mounted_source_resolve_cache_capacity_locked();
      }
    }

    return resolved_or;
  }
}

grpc::Status DiskArtifactService::import_artifact_from_path(
    RpcContext& rctx,
    const v2::ImportArtifactFromPathRequest& req,
    v2::ImportArtifactFromPathResponse& resp) {
  auto& span = rctx.span();
  const bool verify_checksums = req.verify_checksums();
  if (req.path().empty()) {
    materialization_disk_resolve::record_disk_import_outcome("invalid_argument");
    return {StatusCode::INVALID_ARGUMENT, "path is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    materialization_disk_resolve::record_disk_import_outcome("unavailable");
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  const bool loopback_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  if (!loopback_peer) {
    materialization_disk_resolve::record_disk_import_outcome("permission_denied");
    return {StatusCode::PERMISSION_DENIED, "ImportArtifactFromPath is local-only (loopback/UDS)"};
  }

  auto path_policy_or = resolve_disk_path_policy(req.path(), d_.public_disk_source_policy);
  if (!path_policy_or.ok()) {
    materialization_disk_resolve::record_disk_import_outcome("invalid_argument");
    return to_grpc_status(path_policy_or.status());
  }

  span->SetAttribute("tc.store.verify_checksums", verify_checksums);
  auto imported_or = import_artifact_from_path_cached(path_policy_or->normalized_path, verify_checksums);
  if (!imported_or.ok()) {
    return to_import_grpc_status(imported_or.status());
  }

  const auto& imported = *imported_or;
  auto metadata_status = ensure_artifact_metadata_registered(imported);
  if (!metadata_status.ok()) {
    return to_grpc_status(metadata_status);
  }
  const size_t promotions_noted = d_.source_registry.note_promoted_artifact_for_source(
      imported.normalized_path.string(), to_registry_fingerprints(imported.file_fingerprints), imported.artifact_id);
  (void)promotions_noted;
  fill_import_response(imported, resp);
  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.disk.path", imported.normalized_path.string());
    span->SetAttribute("tc.artifact.id", imported.artifact_id);
  }
  span->SetAttribute("tc.artifact.generation", static_cast<std::int64_t>(imported.generation));

  upsert_local_import_binding(d_.source_registry, imported);

  rctx.mark_success();
  return grpc::Status::OK;
}

grpc::Status DiskArtifactService::resolve_public_disk_source(
    RpcContext& rctx,
    const v2::ResolvePublicDiskSourceRequest& req,
    v2::ResolvePublicDiskSourceResponse& resp) {
  auto& span = rctx.span();
  const bool verify_checksums = req.verify_checksums();
  if (req.path().empty()) {
    materialization_disk_resolve::record_disk_import_outcome("invalid_argument");
    return {StatusCode::INVALID_ARGUMENT, "path is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    materialization_disk_resolve::record_disk_import_outcome("unavailable");
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  const bool loopback_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  if (!loopback_peer) {
    materialization_disk_resolve::record_disk_import_outcome("permission_denied");
    return {StatusCode::PERMISSION_DENIED, "ResolvePublicDiskSource is local-only (loopback/UDS)"};
  }

  auto path_policy_or = resolve_disk_path_policy(req.path(), d_.public_disk_source_policy);
  if (!path_policy_or.ok()) {
    materialization_disk_resolve::record_disk_import_outcome("invalid_argument");
    return to_grpc_status(path_policy_or.status());
  }

  span->SetAttribute("tc.store.verify_checksums", verify_checksums);
  span->SetAttribute("tc.disk.hash_data_skipped", true);
  const auto attestation_policy = build_attestation_policy(path_policy_or->policy);
  auto resolved_or =
      resolve_mounted_source_artifact_cached(path_policy_or->normalized_path, verify_checksums, attestation_policy);
  if (!resolved_or.ok()) {
    return to_import_grpc_status(resolved_or.status());
  }
  auto existing_entry = d_.source_registry.lookup_binding(resolved_or->artifact_id);
  if (existing_entry.has_value() && existing_entry->promoted_content_artifact_id.has_value() &&
      !resolved_or->trusted_content_artifact_id.has_value()) {
    resolved_or->trusted_content_artifact_id = existing_entry->promoted_content_artifact_id;
    resolved_or->resolution_strategy =
        materialization_disk_resolve::MountedSourceResolutionStrategy::kAttestedWithTrustedDescriptorHint;
  }

  fill_public_disk_source_response(*resolved_or, verify_checksums, resp);
  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.disk.path", resolved_or->normalized_path.string());
    span->SetAttribute("tc.artifact.id", resolved_or->artifact_id);
    span->SetAttribute("tc.disk.policy_id", resolved_or->policy_id);
  }
  span->SetAttribute("tc.artifact.generation", static_cast<std::int64_t>(resolved_or->generation));
  span->SetAttribute("tc.artifact.id_kind", artifact_id_kind_attr(resolved_or->artifact_id));
  span->SetAttribute(
      "tc.artifact.trusted_content_hint_kind",
      trusted_content_hint_kind_attr(resolved_or->trusted_content_artifact_id));
  span->SetAttribute("tc.disk.metadata_capability", metadata_capability_attr(resolved_or->metadata_capability));
  span->SetAttribute("tc.disk.resolution_strategy", resolution_strategy_attr(resolved_or->resolution_strategy));
  span->SetAttribute("tc.disk.validation_mode", validation_mode_attr(resolved_or->validation_mode));
  d_.source_registry.upsert_binding(
      resolved_or->artifact_id,
      ArtifactSourceRegistry::Entry{
          .source_kind = ArtifactSourceRegistry::SourceKind::kMountedSourceArtifact,
          .canonical_source_path = resolved_or->normalized_path.string(),
          .canonical_index_json = resolved_or->canonical_index_json,
          .source_index_json = resolved_or->source_index_json,
          .source_disk_path = resolved_or->normalized_path.string(),
          .source_format_kind = to_registry_source_format(resolved_or->format_kind),
          .descriptor_present = resolved_or->descriptor_present,
          .index_multihash = resolved_or->canonical_index_multihash,
          .data_multihash = std::nullopt,
          .trusted_content_artifact_id = resolved_or->trusted_content_artifact_id,
          .promoted_content_artifact_id =
              existing_entry.has_value() ? existing_entry->promoted_content_artifact_id : std::nullopt,
          .policy_id = resolved_or->policy_id,
          .snapshot_digest = resolved_or->snapshot_digest,
          .generation = resolved_or->generation,
          .tensor_aware_metadata = resolved_or->metadata_capability ==
              materialization_disk_resolve::MountedSourceMetadataCapability::kTensorAware,
          .validate_before_read = resolved_or->validation_mode ==
              materialization_disk_resolve::MountedSourceValidationMode::kValidateBeforeRead,
          .file_fingerprints = to_registry_fingerprints(resolved_or->file_fingerprints),
          .created_at = absl::Now(),
          .updated_at = absl::Now(),
      });
  rctx.mark_success();
  return grpc::Status::OK;
}

grpc::Status DiskArtifactService::promote_mounted_source_artifact(
    RpcContext& rctx,
    const v2::PromoteMountedSourceArtifactRequest& req,
    v2::PromoteMountedSourceArtifactResponse& resp) {
  auto& span = rctx.span();
  const bool verify_checksums = req.verify_checksums();
  if (req.artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  if (!common::is_msa1_artifact_id(req.artifact_id())) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id must be an msa1 mounted-source artifact id"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  const bool loopback_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  if (!loopback_peer) {
    return {StatusCode::PERMISSION_DENIED, "PromoteMountedSourceArtifact is local-only (loopback/UDS)"};
  }

  auto resolution_or = materialization_request_common::resolve_artifact_and_disk_source(
      d_.global_store_client,
      &d_.source_registry,
      storage_path_,
      req.artifact_id(),
      /*allow_disk=*/true,
      /*allow_local_import_disk_source=*/true,
      /*loopback_peer=*/true);
  if (!resolution_or.ok()) {
    return to_import_grpc_status(resolution_or.status());
  }
  if (!resolution_or->normalized_disk_path.has_value()) {
    return {StatusCode::FAILED_PRECONDITION, "mounted-source artifact has no local disk path in this daemon session"};
  }

  span->SetAttribute("tc.store.verify_checksums", verify_checksums);
  span->SetAttribute("tc.artifact.id", req.artifact_id());
  span->SetAttribute("tc.artifact.id_kind", artifact_id_kind_attr(req.artifact_id()));
  auto imported_or = import_artifact_from_path_cached(*resolution_or->normalized_disk_path, verify_checksums);
  if (!imported_or.ok()) {
    return to_import_grpc_status(imported_or.status());
  }

  const auto& imported = *imported_or;
  auto metadata_status = ensure_artifact_metadata_registered(imported);
  if (!metadata_status.ok()) {
    return to_grpc_status(metadata_status);
  }
  const size_t promotions_noted = d_.source_registry.note_promoted_artifact_for_source(
      imported.normalized_path.string(),
      to_registry_fingerprints(imported.file_fingerprints),
      imported.artifact_id,
      ArtifactSourceRegistry::PromotionOrigin::kMountedVerify);
  (void)promotions_noted;
  upsert_local_import_binding(d_.source_registry, imported);
  fill_promote_mounted_source_response(imported, req.artifact_id(), resp);
  span->SetAttribute("tc.artifact.promoted_id", imported.artifact_id);
  span->SetAttribute("tc.artifact.promoted_id_kind", artifact_id_kind_attr(imported.artifact_id));
  rctx.mark_success();
  return grpc::Status::OK;
}

grpc::Status DiskArtifactService::import_artifact_from_path_stream(
    RpcContext& rctx,
    const v2::ImportArtifactFromPathRequest& req,
    grpc::ServerWriter<v2::ImportArtifactFromPathStreamEvent>& writer) {
  auto& span = rctx.span();
  const bool verify_checksums = req.verify_checksums();
  std::uint64_t seq = 0;
  bool stream_writable = true;

  const auto write_stream_event = [&](const materialization_disk_resolve::ImportProgressUpdate& update,
                                      const v2::ImportArtifactFromPathResponse* final_result = nullptr) {
    if (!stream_writable) {
      return;
    }
    v2::ImportArtifactFromPathStreamEvent event;
    event.set_seq(++seq);
    event.set_phase(to_proto_phase(update.phase));
    event.set_processed_bytes(update.processed_bytes);
    event.set_total_bytes(update.total_bytes);
    event.set_percent(to_progress_percent(update.processed_bytes, update.total_bytes, update.done));
    event.set_done(update.done);
    event.set_error(update.error);
    event.set_error_code(to_proto_error_code(update.error_code));
    if (!update.message.empty()) {
      event.set_message(update.message);
    }
    if (final_result != nullptr) {
      *event.mutable_result() = *final_result;
    }
    stream_writable = writer.Write(event);
  };

  const auto send_error_and_return = [&](const grpc::Status& status) {
    materialization_disk_resolve::ImportProgressUpdate update;
    update.phase = materialization_disk_resolve::ImportArtifactPhase::kError;
    update.done = true;
    update.error = true;
    update.error_code = grpc_status_to_import_error(status);
    update.message = status.error_message();
    write_stream_event(update);
    return status;
  };

  if (req.path().empty()) {
    materialization_disk_resolve::record_disk_import_outcome("invalid_argument");
    return send_error_and_return({StatusCode::INVALID_ARGUMENT, "path is required"});
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    materialization_disk_resolve::record_disk_import_outcome("unavailable");
    return send_error_and_return({StatusCode::UNAVAILABLE, "daemon is shutting down"});
  }

  const bool loopback_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  if (!loopback_peer) {
    materialization_disk_resolve::record_disk_import_outcome("permission_denied");
    materialization_disk_resolve::ImportProgressUpdate update;
    update.phase = materialization_disk_resolve::ImportArtifactPhase::kError;
    update.done = true;
    update.error = true;
    update.error_code = materialization_disk_resolve::ImportArtifactErrorCode::kPolicyDeniedNonLocalPeer;
    update.message = "ImportArtifactFromPath is local-only (loopback/UDS)";
    write_stream_event(update);
    return {StatusCode::PERMISSION_DENIED, update.message};
  }

  auto path_policy_or = resolve_disk_path_policy(req.path(), d_.public_disk_source_policy);
  if (!path_policy_or.ok()) {
    materialization_disk_resolve::record_disk_import_outcome("invalid_argument");
    return send_error_and_return(to_grpc_status(path_policy_or.status()));
  }

  span->SetAttribute("tc.store.verify_checksums", verify_checksums);
  auto imported_or = import_artifact_from_path_cached(
      path_policy_or->normalized_path,
      verify_checksums,
      [&](const materialization_disk_resolve::ImportProgressUpdate& update) {
        if (update.done || update.error) {
          return;
        }
        write_stream_event(update);
      });
  if (!imported_or.ok()) {
    return send_error_and_return(to_import_grpc_status(imported_or.status()));
  }

  const auto& imported = *imported_or;
  auto metadata_status = ensure_artifact_metadata_registered(imported);
  if (!metadata_status.ok()) {
    return send_error_and_return(to_grpc_status(metadata_status));
  }
  const size_t promotions_noted = d_.source_registry.note_promoted_artifact_for_source(
      imported.normalized_path.string(), to_registry_fingerprints(imported.file_fingerprints), imported.artifact_id);
  (void)promotions_noted;
  v2::ImportArtifactFromPathResponse final_resp;
  fill_import_response(imported, final_resp);
  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.disk.path", imported.normalized_path.string());
    span->SetAttribute("tc.artifact.id", imported.artifact_id);
  }
  span->SetAttribute("tc.artifact.generation", static_cast<std::int64_t>(imported.generation));

  upsert_local_import_binding(d_.source_registry, imported);

  materialization_disk_resolve::ImportProgressUpdate done;
  done.phase = materialization_disk_resolve::ImportArtifactPhase::kDone;
  done.done = true;
  write_stream_event(done, &final_resp);

  if (rctx.server_context().IsCancelled()) {
    return {StatusCode::CANCELLED, "client cancelled ImportArtifactFromPathStream"};
  }
  rctx.mark_success();
  return grpc::Status::OK;
}

grpc::Status DiskArtifactService::get_artifact_index_by_id(
    RpcContext& rctx,
    const v2::GetArtifactIndexByIdRequest& req,
    v2::GetArtifactIndexByIdResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.artifact_id());

  if (req.artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  auto local_entry = d_.source_registry.lookup_binding(req.artifact_id());
  if (common::is_msa1_artifact_id(req.artifact_id()) && !local_entry.has_value()) {
    materialization_disk_resolve::record_mounted_source_validation_outcome("stale_session");
    return {StatusCode::FAILED_PRECONDITION, "mounted-source artifact_id is not valid in this daemon session"};
  }
  if (local_entry.has_value() &&
      local_entry->source_kind == ArtifactSourceRegistry::SourceKind::kMountedSourceArtifact &&
      local_entry->validate_before_read) {
    materialization_disk_resolve::SourceFingerprintMap expected_fingerprints;
    expected_fingerprints.reserve(local_entry->file_fingerprints.size());
    for (const auto& [relative_path, fingerprint] : local_entry->file_fingerprints) {
      expected_fingerprints.insert_or_assign(
          relative_path,
          materialization_disk_resolve::SourceFileFingerprint{
              .inode = fingerprint.inode,
              .size = fingerprint.size,
              .mtime_ns = fingerprint.mtime_ns,
          });
    }
    auto validation_status = materialization_disk_resolve::validate_mounted_source_snapshot(
        std::filesystem::path(local_entry->canonical_source_path),
        local_entry->canonical_index_json,
        local_entry->source_index_json,
        expected_fingerprints);
    if (!validation_status.ok()) {
      materialization_disk_resolve::record_mounted_source_validation_outcome("mismatch");
      const bool erased = d_.source_registry.erase_binding(req.artifact_id());
      (void)erased;
      return to_import_grpc_status(validation_status);
    }
    materialization_disk_resolve::record_mounted_source_validation_outcome("ok");
    resp.set_tensor_index_data(local_entry->canonical_index_json);
    rctx.mark_success();
    return grpc::Status::OK;
  }
  if (local_entry.has_value() && !local_entry->canonical_index_json.empty()) {
    resp.set_tensor_index_data(local_entry->canonical_index_json);
    rctx.mark_success();
    return grpc::Status::OK;
  }

  auto bytes_or = d_.engine.get_canonical_index_by_id(req.artifact_id());
  if (!bytes_or.ok()) {
    return to_grpc_status(bytes_or.status());
  }
  resp.set_tensor_index_data(*bytes_or);
  rctx.mark_success();
  return grpc::Status::OK;
}

} // namespace tensorcast::daemon
