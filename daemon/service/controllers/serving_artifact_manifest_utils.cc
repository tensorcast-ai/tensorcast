// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/serving_artifact_manifest_utils.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/selection_identity.h"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon::serving_artifact_manifest {
namespace {

using nlohmann::json;

constexpr std::string_view kTensorSchemaHashVersion = "tensorcast.representation.tensor_schema.v1";

struct CanonicalTensorEntry {
  std::string name;
  uint64_t segment_offset{0};
  uint64_t segment_size{0};
  uint64_t tensor_nbytes{0};
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::string dtype;
  uint64_t storage_offset{0};
};

struct ParsedCanonicalIndex {
  std::vector<CanonicalTensorEntry> tensors;
  uint64_t total_size{0};
};

absl::StatusOr<uint64_t> infer_dtype_element_size(std::string_view dtype) {
  static const std::pair<std::string_view, uint64_t> kMap[] = {
      {"torch.float16", 2},
      {"torch.bfloat16", 2},
      {"torch.float8_e4m3fn", 1},
      {"torch.float8_e5m2", 1},
      {"torch.float32", 4},
      {"torch.float64", 8},
      {"torch.int8", 1},
      {"torch.uint8", 1},
      {"torch.int16", 2},
      {"torch.int32", 4},
      {"torch.int64", 8},
      {"torch.bool", 1},
      {"torch.float", 4},
      {"torch.double", 8},
  };
  for (const auto& [candidate, bytes] : kMap) {
    if (candidate == dtype) {
      return bytes;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat("unsupported dtype in canonical index: ", dtype));
}

absl::StatusOr<uint64_t> infer_tensor_nbytes(const std::vector<int64_t>& shape, uint64_t element_size) {
  uint64_t elements = 1;
  for (const int64_t dim : shape) {
    if (dim < 0) {
      return absl::InvalidArgumentError("canonical index shape dimensions must be non-negative");
    }
    const uint64_t u_dim = static_cast<uint64_t>(dim);
    if (u_dim != 0 && elements > std::numeric_limits<uint64_t>::max() / u_dim) {
      return absl::InvalidArgumentError("canonical index tensor element count overflows uint64");
    }
    elements *= u_dim;
  }
  if (element_size != 0 && elements > std::numeric_limits<uint64_t>::max() / element_size) {
    return absl::InvalidArgumentError("canonical index tensor byte size overflows uint64");
  }
  return elements * element_size;
}

std::string hash_serialized_payload(std::string_view version, const std::string& serialized) {
  std::string payload;
  payload.reserve(version.size() + 1 + serialized.size());
  payload.append(version.data(), version.size());
  payload.push_back('\n');
  payload.append(serialized);
  const std::vector<std::uint8_t> digest = tensorcast::common::sha256_digest_bytes(
      absl::Span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size()));
  return tensorcast::common::multibase_multihash_sha256(digest);
}

absl::StatusOr<ParsedCanonicalIndex> parse_canonical_index_json(std::string_view canonical_index_json) {
  json root;
  try {
    root = json::parse(canonical_index_json, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("failed to parse canonical index JSON: ", e.what()));
  }
  if (!root.is_object()) {
    return absl::InvalidArgumentError("canonical index JSON must be an object");
  }

  ParsedCanonicalIndex parsed;
  parsed.tensors.reserve(root.size());
  try {
    for (auto it = root.begin(); it != root.end(); ++it) {
      if (!it.value().is_array() || it.value().size() < 5) {
        return absl::InvalidArgumentError(absl::StrCat("canonical index entry must be an array for tensor=", it.key()));
      }
      CanonicalTensorEntry entry;
      entry.name = it.key();
      entry.segment_offset = it.value().at(0).get<uint64_t>();
      entry.segment_size = it.value().at(1).get<uint64_t>();
      for (const auto& dim : it.value().at(2)) {
        entry.shape.push_back(dim.get<int64_t>());
      }
      for (const auto& dim : it.value().at(3)) {
        entry.stride.push_back(dim.get<int64_t>());
      }
      entry.dtype = it.value().at(4).get<std::string>();
      if (it.value().size() >= 6 && !it.value().at(5).is_null()) {
        entry.storage_offset = it.value().at(5).get<uint64_t>();
      }
      auto element_size_or = infer_dtype_element_size(entry.dtype);
      if (!element_size_or.ok()) {
        return element_size_or.status();
      }
      auto tensor_nbytes_or = infer_tensor_nbytes(entry.shape, *element_size_or);
      if (!tensor_nbytes_or.ok()) {
        return tensor_nbytes_or.status();
      }
      entry.tensor_nbytes = *tensor_nbytes_or;
      parsed.total_size = std::max(parsed.total_size, entry.segment_offset + entry.segment_size);
      parsed.tensors.push_back(std::move(entry));
    }
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("failed to decode canonical index JSON: ", e.what()));
  }
  return parsed;
}

absl::StatusOr<std::string> compute_tensor_schema_hash(
    const std::vector<CanonicalTensorEntry>& tensors,
    std::string_view manifest_tensor_name) {
  std::vector<CanonicalTensorEntry> canonical_tensors;
  canonical_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    if (tensor.name == manifest_tensor_name) {
      continue;
    }
    canonical_tensors.push_back(tensor);
  }
  std::sort(
      canonical_tensors.begin(),
      canonical_tensors.end(),
      [](const CanonicalTensorEntry& lhs, const CanonicalTensorEntry& rhs) { return lhs.name < rhs.name; });

  json tensors_json = json::array();
  for (const auto& tensor : canonical_tensors) {
    auto element_size_or = infer_dtype_element_size(tensor.dtype);
    if (!element_size_or.ok()) {
      return element_size_or.status();
    }
    tensors_json.push_back(
        json{
            {"name", tensor.name},
            {"dtype", tensor.dtype},
            {"shape", tensor.shape},
            {"stride", tensor.stride},
            {"element_size", *element_size_or},
        });
  }
  const std::string serialized = json{{"tensors", std::move(tensors_json)}}.dump(-1, ' ', true);
  return hash_serialized_payload(kTensorSchemaHashVersion, serialized);
}

absl::StatusOr<const CanonicalTensorEntry*> find_manifest_tensor(
    const std::vector<CanonicalTensorEntry>& tensors,
    std::string_view manifest_tensor_name) {
  for (const auto& tensor : tensors) {
    if (tensor.name == manifest_tensor_name) {
      if (tensor.tensor_nbytes == 0) {
        return absl::DataLossError("serving manifest tensor must not be empty");
      }
      if (tensor.dtype != "torch.uint8") {
        return absl::FailedPreconditionError("serving manifest tensor must use dtype torch.uint8");
      }
      if (tensor.shape.size() != 1 || static_cast<uint64_t>(tensor.shape.front()) != tensor.tensor_nbytes) {
        return absl::FailedPreconditionError("serving manifest tensor must be a 1-D uint8 tensor sized to its payload");
      }
      if (tensor.stride.size() != 1 || tensor.stride.front() != 1) {
        return absl::FailedPreconditionError("serving manifest tensor must be contiguous");
      }
      return &tensor;
    }
  }
  return absl::NotFoundError("serving manifest tensor not found");
}

absl::StatusOr<std::string> read_disk_range(
    const std::filesystem::path& artifact_dir,
    uint64_t offset,
    uint64_t length) {
  if (length > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return absl::InvalidArgumentError("serving manifest tensor payload is too large to read into memory");
  }
  std::string payload(static_cast<size_t>(length), '\0');
  uint64_t remaining = length;
  uint64_t logical_offset = offset;
  uint64_t copied = 0;

  auto read_from_file = [&](const std::filesystem::path& path, uint64_t file_offset, uint64_t bytes) -> absl::Status {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
      return absl::NotFoundError(absl::StrCat("serving manifest disk carrier not readable: ", path.string()));
    }
    if (file_offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
      return absl::InvalidArgumentError("serving manifest disk carrier offset exceeds stream range");
    }
    in.seekg(static_cast<std::streamoff>(file_offset), std::ios::beg);
    if (!in.good()) {
      return absl::DataLossError("failed to seek serving manifest disk carrier");
    }
    in.read(payload.data() + copied, static_cast<std::streamsize>(bytes));
    if (in.gcount() != static_cast<std::streamsize>(bytes)) {
      return absl::DataLossError("serving manifest disk carrier is truncated");
    }
    return absl::OkStatus();
  };

  const auto single_part = artifact_dir / "tensor.data";
  std::error_code ec;
  if (std::filesystem::exists(single_part, ec) && !ec) {
    const uint64_t file_size = std::filesystem::file_size(single_part, ec);
    if (ec || offset > file_size || length > file_size - offset) {
      return absl::DataLossError("serving manifest disk carrier exceeds tensor.data");
    }
    auto st = read_from_file(single_part, offset, length);
    if (!st.ok()) {
      return st;
    }
    return payload;
  }

  for (uint64_t shard_idx = 0; remaining > 0; ++shard_idx) {
    const auto part_path = artifact_dir / absl::StrCat("tensor.data_", shard_idx);
    if (!std::filesystem::exists(part_path, ec) || ec) {
      return absl::NotFoundError("serving manifest disk carrier shard not found");
    }
    const uint64_t file_size = std::filesystem::file_size(part_path, ec);
    if (ec) {
      return absl::ErrnoToStatus(ec.value(), "failed to stat serving manifest disk carrier shard");
    }
    if (logical_offset >= file_size) {
      logical_offset -= file_size;
      continue;
    }
    const uint64_t bytes = std::min<uint64_t>(remaining, file_size - logical_offset);
    auto st = read_from_file(part_path, logical_offset, bytes);
    if (!st.ok()) {
      return st;
    }
    copied += bytes;
    remaining -= bytes;
    logical_offset = 0;
  }
  return payload;
}

absl::StatusOr<std::string> read_manifest_tensor_payload(
    store::StoreEngine* engine,
    std::string_view artifact_id,
    std::string_view canonical_index_json,
    const CanonicalTensorEntry& manifest_tensor,
    uint64_t total_size,
    const std::optional<store::loading::DiskSource>& disk_source,
    const std::optional<store::loading::DiskMetadata>& disk_metadata) {
  if (manifest_tensor.storage_offset > std::numeric_limits<uint64_t>::max() - manifest_tensor.segment_offset) {
    return absl::DataLossError("serving manifest tensor offset overflows");
  }
  const uint64_t payload_offset = manifest_tensor.segment_offset + manifest_tensor.storage_offset;
  if (manifest_tensor.tensor_nbytes > std::numeric_limits<uint64_t>::max() - payload_offset) {
    return absl::DataLossError("serving manifest tensor range overflows");
  }
  if (payload_offset + manifest_tensor.tensor_nbytes > total_size) {
    return absl::DataLossError("serving manifest tensor exceeds the serving artifact size");
  }
  const bool known_partitioned_disk_source = disk_source.has_value() && disk_metadata.has_value() &&
      disk_metadata->is_safetensors.has_value() && !*disk_metadata->is_safetensors;
  if (known_partitioned_disk_source) {
    return read_disk_range(disk_source->path, payload_offset, manifest_tensor.tensor_nbytes);
  }

  store::loading::MaterializeHints hints;
  hints.artifact_id = std::string(artifact_id);
  std::vector<std::string> tensor_names{manifest_tensor.name};
  auto view_plan_or =
      store::StoreEngine::compute_view_plan(canonical_index_json, store::loader::ViewSpec{}, tensor_names);
  if (!view_plan_or.ok()) {
    return view_plan_or.status();
  }
  store::loading::VariantIdentity variant;
  variant.canonical_artifact_id = std::string(artifact_id);
  variant.canonical_index_json = std::string(canonical_index_json);
  variant.cached_plan = *view_plan_or;
  const std::string subset_hash = tensorcast::common::compute_view_subset_hash_bytes(absl::MakeSpan(tensor_names));
  if (!subset_hash.empty()) {
    variant.view_id = absl::StrCat("subset:", absl::BytesToHexString(subset_hash));
  }
  hints.variant = std::move(variant);
  if (disk_metadata.has_value()) {
    hints.disk_metadata = *disk_metadata;
  }
  if (disk_source.has_value()) {
    hints.source_mutation_policy = store::loading::SourceMutationPolicy::kReadOnly;
  }
  const store::DeviceKey cpu_device{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  auto handle_or =
      engine->materialize_replica(cpu_device, store::StoreEngine::MaterializeMode::AUTO, hints, disk_source);
  if (!handle_or.ok()) {
    return handle_or.status();
  }
  auto loader_or =
      engine->open_local_replica_loader(handle_or->replica_key, tensorcast::common::memory::MemoryLocation::CPU);
  if (!loader_or.ok()) {
    return loader_or.status();
  }
  auto init_status = (*loader_or)->initialize();
  if (!init_status.ok()) {
    return init_status;
  }
  auto source_or = (*loader_or)->open_source();
  if (!source_or.ok()) {
    return source_or.status();
  }
  std::string payload(static_cast<size_t>(manifest_tensor.tensor_nbytes), '\0');
  auto read_or = (*source_or)->read_at(0, payload.data(), payload.size());
  if (!read_or.ok()) {
    return read_or.status();
  }
  if (*read_or != payload.size()) {
    return absl::DataLossError("serving manifest compact view returned a short read");
  }
  return payload;
}

absl::Status validate_manifest_record(
    const ServingArtifactManifestRecord& manifest,
    std::string_view actual_serving_manifest_ref,
    bool require_self_describing_manifest,
    uint64_t canonical_tensor_count,
    std::string_view computed_tensor_schema_hash,
    const std::optional<std::string>& expected_representation_contract_hash,
    const std::optional<std::string>& expected_serving_build_digest,
    const std::optional<std::string>& expected_serving_build_digest_version,
    const std::optional<std::string>& expected_topology_admission_digest);

absl::StatusOr<ServingArtifactPreflightResult> preflight_manifest_payload_common(
    std::string_view canonical_index_json,
    std::string_view manifest_payload,
    const std::optional<std::string>& serving_manifest_ref_override,
    const std::optional<std::string>& expected_representation_contract_hash,
    const std::optional<std::string>& expected_serving_build_digest,
    const std::optional<std::string>& expected_serving_build_digest_version,
    const std::optional<std::string>& expected_topology_admission_digest,
    bool require_manifest) {
  auto parsed_index_or = parse_canonical_index_json(canonical_index_json);
  if (!parsed_index_or.ok()) {
    return parsed_index_or.status();
  }

  const std::string serving_manifest_ref =
      serving_manifest_ref_override.value_or(std::string(kPhase1ServingManifestRef));
  auto manifest_tensor_name_or = parse_tensor_manifest_ref(serving_manifest_ref);
  if (!manifest_tensor_name_or.ok()) {
    return manifest_tensor_name_or.status();
  }
  const std::string& manifest_tensor_name = *manifest_tensor_name_or;

  auto manifest_tensor_or = find_manifest_tensor(parsed_index_or->tensors, manifest_tensor_name);
  if (!manifest_tensor_or.ok()) {
    if (absl::IsNotFound(manifest_tensor_or.status()) && !require_manifest) {
      return ServingArtifactPreflightResult{};
    }
    if (absl::IsNotFound(manifest_tensor_or.status())) {
      return absl::DataLossError(
          absl::StrCat(
              "serving artifact is missing manifest tensor referenced by serving_manifest_ref: ",
              manifest_tensor_name));
    }
    return manifest_tensor_or.status();
  }

  auto tensor_schema_hash_or = compute_tensor_schema_hash(parsed_index_or->tensors, manifest_tensor_name);
  if (!tensor_schema_hash_or.ok()) {
    return tensor_schema_hash_or.status();
  }

  uint64_t canonical_tensor_count = 0;
  for (const auto& tensor : parsed_index_or->tensors) {
    if (tensor.name != manifest_tensor_name) {
      ++canonical_tensor_count;
    }
  }

  auto manifest_or = parse_serving_manifest_payload(manifest_payload);
  if (!manifest_or.ok()) {
    return manifest_or.status();
  }
  auto validate_status = validate_manifest_record(
      *manifest_or,
      serving_manifest_ref,
      require_manifest,
      canonical_tensor_count,
      *tensor_schema_hash_or,
      expected_representation_contract_hash,
      expected_serving_build_digest,
      expected_serving_build_digest_version,
      expected_topology_admission_digest);
  if (!validate_status.ok()) {
    return validate_status;
  }

  ServingArtifactPreflightResult result;
  result.serving_manifest_present = true;
  result.serving_manifest_ref = serving_manifest_ref;
  result.representation_contract_hash = manifest_or->representation_contract_hash;
  result.serving_build_digest = manifest_or->serving_build_digest;
  result.tensor_schema_hash = *tensor_schema_hash_or;
  result.canonical_tensor_count = canonical_tensor_count;
  result.manifest = std::move(*manifest_or);
  return result;
}

absl::Status validate_manifest_record(
    const ServingArtifactManifestRecord& manifest,
    std::string_view actual_serving_manifest_ref,
    bool require_self_describing_manifest,
    uint64_t canonical_tensor_count,
    std::string_view computed_tensor_schema_hash,
    const std::optional<std::string>& expected_representation_contract_hash,
    const std::optional<std::string>& expected_serving_build_digest,
    const std::optional<std::string>& expected_serving_build_digest_version,
    const std::optional<std::string>& expected_topology_admission_digest) {
  if (manifest.schema_version != 1) {
    return absl::FailedPreconditionError("serving artifact manifest schema_version must be 1 in phase 1");
  }
  if (manifest.artifact_kind != "serving") {
    return absl::FailedPreconditionError("serving artifact manifest artifact_kind must be 'serving'");
  }
  if (manifest.framework_name.empty() || manifest.adapter_version.empty() || manifest.serving_abi_version.empty()) {
    return absl::FailedPreconditionError(
        "serving artifact manifest requires framework_name, adapter_version, and serving_abi_version");
  }
  if (manifest.representation_contract_hash.empty()) {
    return absl::FailedPreconditionError("serving artifact manifest requires representation_contract_hash");
  }
  if (manifest.serving_build_digest.empty()) {
    return absl::FailedPreconditionError("serving artifact manifest requires serving_build_digest");
  }
  const std::string resolved_serving_build_digest_version =
      manifest.serving_build_digest_version.value_or(std::string(kPhase1ServingBuildDigestVersion));
  if (resolved_serving_build_digest_version.empty()) {
    return absl::FailedPreconditionError("serving artifact manifest requires serving_build_digest_version");
  }
  if (resolved_serving_build_digest_version != kPhase1ServingBuildDigestVersion) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            "unsupported serving artifact manifest serving_build_digest_version: ",
            resolved_serving_build_digest_version));
  }
  if (manifest.tensor_schema_hash.empty()) {
    return absl::FailedPreconditionError("serving artifact manifest requires tensor_schema_hash");
  }
  if (manifest.builder_mode.empty() || manifest.build_pipeline_version.empty()) {
    return absl::FailedPreconditionError("serving artifact manifest requires builder_mode and build_pipeline_version");
  }
  if (manifest.logical_topology_json.has_value()) {
    try {
      (void)json::parse(*manifest.logical_topology_json, nullptr, true);
    } catch (const std::exception& e) {
      return absl::InvalidArgumentError(
          absl::StrCat("serving artifact manifest logical_topology_json is invalid: ", e.what()));
    }
  }
  if (manifest.serving_manifest_ref.has_value()) {
    auto manifest_ref_or = parse_tensor_manifest_ref(*manifest.serving_manifest_ref);
    if (!manifest_ref_or.ok()) {
      return manifest_ref_or.status();
    }
    if (*manifest.serving_manifest_ref != actual_serving_manifest_ref) {
      return absl::FailedPreconditionError(
          "serving artifact manifest serving_manifest_ref does not match the resolved carrier");
    }
  } else if (require_self_describing_manifest) {
    return absl::FailedPreconditionError(
        "serving artifact manifest requires serving_manifest_ref in strict-serving mode");
  }
  if (manifest.canonical_tensor_count != canonical_tensor_count) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            "serving artifact manifest canonical_tensor_count mismatch: manifest=",
            manifest.canonical_tensor_count,
            " index=",
            canonical_tensor_count));
  }
  if (manifest.tensor_schema_hash != computed_tensor_schema_hash) {
    return absl::FailedPreconditionError("serving artifact manifest tensor_schema_hash does not match canonical index");
  }
  if (expected_representation_contract_hash.has_value() &&
      manifest.representation_contract_hash != *expected_representation_contract_hash) {
    return absl::FailedPreconditionError(
        "serving artifact manifest representation_contract_hash does not match the expected lineage");
  }
  if (expected_serving_build_digest.has_value() && manifest.serving_build_digest != *expected_serving_build_digest) {
    return absl::FailedPreconditionError(
        "serving artifact manifest serving_build_digest does not match the expected lineage");
  }
  if (expected_serving_build_digest_version.has_value() &&
      resolved_serving_build_digest_version != *expected_serving_build_digest_version) {
    return absl::FailedPreconditionError(
        "serving artifact manifest serving_build_digest_version does not match the expected lineage");
  }
  if (expected_topology_admission_digest.has_value() &&
      manifest.topology_admission_digest.value_or("") != *expected_topology_admission_digest) {
    return absl::FailedPreconditionError(
        "serving artifact manifest topology_admission_digest does not match the expected topology");
  }
  return absl::OkStatus();
}

} // namespace

absl::StatusOr<ServingArtifactManifestRecord> parse_serving_manifest_payload(std::string_view payload) {
  json root;
  try {
    root = json::parse(payload, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("failed to parse serving artifact manifest: ", e.what()));
  }
  if (!root.is_object()) {
    return absl::InvalidArgumentError("serving artifact manifest must be a JSON object");
  }

  ServingArtifactManifestRecord manifest;
  try {
    manifest.schema_version = root.at("schema_version").get<int64_t>();
    manifest.artifact_kind = root.at("artifact_kind").get<std::string>();
    manifest.framework_name = root.at("framework_name").get<std::string>();
    manifest.adapter_version = root.at("adapter_version").get<std::string>();
    manifest.serving_abi_version = root.at("serving_abi_version").get<std::string>();
    manifest.representation_contract_hash = root.at("representation_contract_hash").get<std::string>();
    manifest.serving_build_digest = root.at("serving_build_digest").get<std::string>();
    manifest.tensor_schema_hash = root.at("tensor_schema_hash").get<std::string>();
    manifest.canonical_tensor_count = root.at("canonical_tensor_count").get<uint64_t>();
    manifest.builder_mode = root.at("builder_mode").get<std::string>();
    manifest.build_pipeline_version = root.at("build_pipeline_version").get<std::string>();
    if (root.contains("serving_build_digest_version") && !root.at("serving_build_digest_version").is_null()) {
      manifest.serving_build_digest_version = root.at("serving_build_digest_version").get<std::string>();
    }
    if (root.contains("serving_manifest_ref") && !root.at("serving_manifest_ref").is_null()) {
      manifest.serving_manifest_ref = root.at("serving_manifest_ref").get<std::string>();
    }
    if (root.contains("source_artifact_ref") && !root.at("source_artifact_ref").is_null()) {
      manifest.source_artifact_ref = root.at("source_artifact_ref").get<std::string>();
    }
    if (root.contains("logical_topology_json") && !root.at("logical_topology_json").is_null()) {
      manifest.logical_topology_json = root.at("logical_topology_json").get<std::string>();
    }
    if (root.contains("topology_admission_digest") && !root.at("topology_admission_digest").is_null()) {
      manifest.topology_admission_digest = root.at("topology_admission_digest").get<std::string>();
    }
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(
        absl::StrCat("serving artifact manifest missing or malformed fields: ", e.what()));
  }
  return manifest;
}

absl::StatusOr<std::string> parse_tensor_manifest_ref(std::string_view serving_manifest_ref) {
  constexpr std::string_view kTensorRefPrefix = "tensor:";
  if (serving_manifest_ref.substr(0, kTensorRefPrefix.size()) != kTensorRefPrefix) {
    return absl::UnimplementedError("only tensor:<name> serving_manifest_ref carriers are supported in phase 1");
  }
  const std::string tensor_name = std::string(serving_manifest_ref.substr(kTensorRefPrefix.size()));
  if (tensor_name.empty()) {
    return absl::InvalidArgumentError("serving_manifest_ref tensor carrier requires a tensor name");
  }
  return tensor_name;
}

ServingArtifactPreflightRequest build_preflight_request(
    std::string artifact_id,
    std::string canonical_index_json,
    std::optional<store::loading::DiskSource> disk_source,
    std::optional<store::loading::DiskMetadata> disk_metadata,
    const v2::ServingArtifactRuntimePolicy* runtime_policy) {
  ServingArtifactPreflightRequest request{
      .artifact_id = std::move(artifact_id),
      .canonical_index_json = std::move(canonical_index_json),
      .disk_source = std::move(disk_source),
      .disk_metadata = std::move(disk_metadata),
  };
  if (runtime_policy == nullptr) {
    return request;
  }
  if (!runtime_policy->serving_manifest_ref().empty()) {
    request.serving_manifest_ref = runtime_policy->serving_manifest_ref();
  }
  if (!runtime_policy->expected_representation_contract_hash().empty()) {
    request.expected_representation_contract_hash = runtime_policy->expected_representation_contract_hash();
  }
  if (!runtime_policy->expected_serving_build_digest().empty()) {
    request.expected_serving_build_digest = runtime_policy->expected_serving_build_digest();
  }
  if (!runtime_policy->expected_topology_admission_digest().empty()) {
    request.expected_topology_admission_digest = runtime_policy->expected_topology_admission_digest();
  }
  request.require_manifest = runtime_policy->require_manifest() || request.serving_manifest_ref.has_value() ||
      request.expected_representation_contract_hash.has_value() || request.expected_serving_build_digest.has_value() ||
      request.expected_serving_build_digest_version.has_value() ||
      request.expected_topology_admission_digest.has_value();
  return request;
}

absl::StatusOr<ServingArtifactPreflightResult> preflight_serving_artifact(
    store::StoreEngine* engine,
    const ServingArtifactPreflightRequest& request) {
  if (engine == nullptr) {
    return absl::InvalidArgumentError("preflight_serving_artifact requires StoreEngine");
  }
  if (request.artifact_id.empty()) {
    return absl::InvalidArgumentError("preflight_serving_artifact requires artifact_id");
  }
  if (request.canonical_index_json.empty()) {
    return absl::InvalidArgumentError("preflight_serving_artifact requires canonical_index_json");
  }
  const std::string serving_manifest_ref =
      request.serving_manifest_ref.value_or(std::string(kPhase1ServingManifestRef));
  auto manifest_tensor_name_or = parse_tensor_manifest_ref(serving_manifest_ref);
  if (!manifest_tensor_name_or.ok()) {
    return manifest_tensor_name_or.status();
  }
  if (!request.require_manifest && !request.serving_manifest_ref.has_value() &&
      std::string_view(request.canonical_index_json).find(kPhase1ServingManifestTensorName) == std::string_view::npos) {
    return ServingArtifactPreflightResult{};
  }
  auto parsed_index_or = parse_canonical_index_json(request.canonical_index_json);
  if (!parsed_index_or.ok()) {
    return parsed_index_or.status();
  }
  auto manifest_tensor_or = find_manifest_tensor(parsed_index_or->tensors, *manifest_tensor_name_or);
  if (!manifest_tensor_or.ok()) {
    if (absl::IsNotFound(manifest_tensor_or.status()) && !request.require_manifest) {
      return ServingArtifactPreflightResult{};
    }
    if (absl::IsNotFound(manifest_tensor_or.status())) {
      return absl::DataLossError(
          absl::StrCat(
              "serving artifact is missing manifest tensor referenced by serving_manifest_ref: ",
              *manifest_tensor_name_or));
    }
    return manifest_tensor_or.status();
  }
  auto payload_or = read_manifest_tensor_payload(
      engine,
      request.artifact_id,
      request.canonical_index_json,
      **manifest_tensor_or,
      parsed_index_or->total_size,
      request.disk_source,
      request.disk_metadata);
  if (!payload_or.ok()) {
    return payload_or.status();
  }
  return preflight_manifest_payload_common(
      request.canonical_index_json,
      *payload_or,
      request.serving_manifest_ref,
      request.expected_representation_contract_hash,
      request.expected_serving_build_digest,
      request.expected_serving_build_digest_version,
      request.expected_topology_admission_digest,
      request.require_manifest);
}

absl::StatusOr<ServingArtifactPreflightResult> preflight_serving_manifest_payload(
    const ServingManifestPayloadPreflightRequest& request) {
  if (request.canonical_index_json.empty()) {
    return absl::InvalidArgumentError("preflight_serving_manifest_payload requires canonical_index_json");
  }
  if (request.manifest_payload.empty() && request.require_manifest) {
    return absl::InvalidArgumentError("preflight_serving_manifest_payload requires manifest_payload");
  }
  return preflight_manifest_payload_common(
      request.canonical_index_json,
      request.manifest_payload,
      request.serving_manifest_ref,
      request.expected_representation_contract_hash,
      request.expected_serving_build_digest,
      request.expected_serving_build_digest_version,
      request.expected_topology_admission_digest,
      request.require_manifest);
}

} // namespace tensorcast::daemon::serving_artifact_manifest
