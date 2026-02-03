// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/metadata/index_reader.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <system_error>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "core/store/materialization/dataplane/metadata/safetensors_util.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::loader {
namespace {

uint64_t compute_total_size_bytes(std::string_view index_json) {
  if (index_json.empty()) {
    return 0;
  }
  try {
    const nlohmann::json idx = nlohmann::json::parse(index_json, nullptr, true);
    uint64_t total_size = 0;
    for (auto it = idx.begin(); it != idx.end(); ++it) {
      const auto& arr = it.value();
      if (!arr.is_array() || arr.size() < 2) {
        continue;
      }
      const uint64_t offset = arr[0].get<uint64_t>();
      const uint64_t size = arr[1].get<uint64_t>();
      total_size = std::max<uint64_t>(total_size, offset + size);
    }
    return total_size;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to parse canonical index JSON for total_size: " << e.what();
    return 0;
  }
}

absl::StatusOr<std::string> load_tensor_index_json(const std::filesystem::path& tensor_index_path) {
  std::ifstream f(tensor_index_path);
  if (!f.is_open()) {
    return absl::NotFoundError(absl::StrCat("tensor_index.json not found at ", tensor_index_path.string()));
  }
  try {
    nlohmann::json j;
    f >> j;
    return j.dump();
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to read/parse tensor_index.json: " << e.what();
    return absl::InternalError("Failed to parse tensor_index.json");
  }
}

absl::StatusOr<std::string> load_tensor_index_cbor(const std::filesystem::path& tensor_index_path) {
  std::ifstream f(tensor_index_path, std::ios::binary);
  if (!f.is_open()) {
    return absl::NotFoundError(absl::StrCat("tensor_index.cbor not found at ", tensor_index_path.string()));
  }
  std::vector<std::uint8_t> bytes;
  bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  if (bytes.empty()) {
    return absl::InternalError("tensor_index.cbor is empty");
  }
  try {
    nlohmann::json j = nlohmann::json::from_cbor(bytes);
    return j.dump();
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to read/parse tensor_index.cbor: " << e.what();
    return absl::InternalError("Failed to parse tensor_index.cbor");
  }
}

IndexInfo make_index_info(
    std::string canonical_json,
    bool is_safetensors,
    std::optional<std::string_view> existing_index_multihash,
    std::optional<std::string> source_json = std::nullopt) {
  IndexInfo info;
  info.is_safetensors = is_safetensors;
  info.total_size_bytes = compute_total_size_bytes(canonical_json);
  info.canonical_index_json = std::move(canonical_json);
  if (source_json.has_value() && !source_json->empty()) {
    info.source_total_size_bytes = compute_total_size_bytes(*source_json);
    info.source_index_json = std::move(source_json);
  }
  if (existing_index_multihash.has_value() && !existing_index_multihash->empty()) {
    info.index_multihash = std::string(*existing_index_multihash);
  } else {
    auto index_mh_or = common::compute_index_multihash(
        std::optional<std::string>(info.canonical_index_json), /*canonical_index_key=*/"");
    if (!index_mh_or.ok()) {
      LOG(WARNING) << "Index multihash computation failed: " << index_mh_or.status();
      info.index_multihash.clear();
    } else {
      info.index_multihash = std::move(index_mh_or).value();
    }
  }
  return info;
}

std::vector<std::filesystem::path> collect_safetensors(const std::filesystem::path& artifact_path) {
  std::vector<std::filesystem::path> files;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(artifact_path, ec)) {
    if (ec) {
      LOG(WARNING) << "Failed to enumerate artifact directory '" << artifact_path.string() << "': " << ec.message();
      break;
    }
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string name = entry.path().filename().string();
    if (absl::EndsWith(name, ".safetensors")) {
      files.push_back(entry.path());
    }
  }
  return files;
}

} // namespace

absl::StatusOr<IndexInfo> canonicalize_from_raw_json(std::string raw_json, int target_device_id) {
  if (raw_json.empty()) {
    return absl::NotFoundError("canonical index JSON is empty");
  }
  auto rebuilt_or = loader::rebuild_stable_canonical_index(raw_json, target_device_id);
  if (!rebuilt_or.ok()) {
    return rebuilt_or.status();
  }
  return make_index_info(std::move(rebuilt_or).value(), /*is_safetensors=*/false, std::nullopt);
}

absl::StatusOr<IndexInfo> build_from_safetensors(
    const std::vector<std::filesystem::path>& safetensor_files,
    std::optional<std::string_view> existing_index_multihash) {
  if (safetensor_files.empty()) {
    return absl::NotFoundError("No safetensors files provided");
  }
  auto source_bytes_or = loader::BuildSourceIndexFromSafetensors(safetensor_files);
  if (!source_bytes_or.ok()) {
    LOG(WARNING) << "Failed to build source index from safetensors: " << source_bytes_or.status();
    return source_bytes_or.status();
  }
  auto canonical_bytes_or =
      loader::build_coalesced_canonical_index_from_source_index_json(*source_bytes_or, /*align_bytes=*/8);
  if (!canonical_bytes_or.ok()) {
    LOG(WARNING) << "Failed to build canonical index from safetensors: " << canonical_bytes_or.status();
    return canonical_bytes_or.status();
  }
  return make_index_info(
      std::move(canonical_bytes_or).value(),
      /*is_safetensors=*/true,
      existing_index_multihash,
      std::move(source_bytes_or).value());
}

absl::StatusOr<IndexInfo> read_from_artifact_dir(const std::filesystem::path& artifact_path, int target_device_id) {
  std::error_code ec;
  const bool exists = std::filesystem::exists(artifact_path, ec);
  if (ec) {
    return absl::ErrnoToStatus(
        ec.value(), absl::StrCat("Failed to access artifact directory '", artifact_path.string(), "'"));
  }
  if (!exists) {
    return absl::NotFoundError(absl::StrCat("Artifact directory not found: ", artifact_path.string()));
  }
  const bool is_dir = std::filesystem::is_directory(artifact_path, ec);
  if (ec) {
    return absl::ErrnoToStatus(
        ec.value(), absl::StrCat("Failed to stat artifact directory '", artifact_path.string(), "'"));
  }
  if (!is_dir) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected artifact path to be a directory: ", artifact_path.string()));
  }

  const auto index_json_path = artifact_path / "tensor_index.json";
  const auto index_cbor_path = artifact_path / "tensor_index.cbor";
  std::error_code index_ec;
  const bool has_json = std::filesystem::exists(index_json_path, index_ec);
  if (index_ec) {
    return absl::ErrnoToStatus(
        index_ec.value(), absl::StrCat("Failed to stat tensor_index.json at ", index_json_path.string()));
  }
  if (has_json) {
    auto raw_or = load_tensor_index_json(index_json_path);
    if (!raw_or.ok()) {
      return raw_or.status();
    }
    return canonicalize_from_raw_json(std::move(raw_or).value(), target_device_id);
  }

  const bool has_cbor = std::filesystem::exists(index_cbor_path, index_ec);
  if (index_ec) {
    return absl::ErrnoToStatus(
        index_ec.value(), absl::StrCat("Failed to stat tensor_index.cbor at ", index_cbor_path.string()));
  }
  if (has_cbor) {
    auto raw_or = load_tensor_index_cbor(index_cbor_path);
    if (!raw_or.ok()) {
      return raw_or.status();
    }
    return canonicalize_from_raw_json(std::move(raw_or).value(), target_device_id);
  }

  auto files = collect_safetensors(artifact_path);
  if (!files.empty()) {
    return build_from_safetensors(files, std::nullopt);
  }

  return absl::NotFoundError(
      absl::StrCat(
          "tensor_index.json/tensor_index.cbor not found and no .safetensors files in artifact directory: ",
          artifact_path.string()));
}

} // namespace tensorcast::store::loader
