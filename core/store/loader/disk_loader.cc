// Copyright (c) 2025, TensorCast Team.

#include "core/store/loader/disk_loader.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include <nlohmann/json.hpp>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "core/common/artifact_verification.h"
#include "core/store/loader/file_partition_source.h"
#include "core/store/loader/multi_safetensors_source.h"
#include "core/store/loader/safetensors_source.h"
#include "core/store/loader/safetensors_util.h"

namespace tensorcast::store {

// Helper to combine paths safely
std::filesystem::path safe_path_join(const std::filesystem::path& base, const std::filesystem::path& sub) {
  if (base.empty()) {
    if (absl::StrContains(sub.string(), "..")) {
      LOG(FATAL) << "Invalid path: contains '..' - " << sub.string();
    }
    return sub;
  }

  if (sub.is_absolute() || absl::StrContains(sub.string(), "..")) {
    LOG(FATAL) << "Invalid subdirectory path: " << sub.string();
  }
  return base / sub;
}

// Helper to load verification info from disk if available
absl::StatusOr<ArtifactVerificationInfo> load_verification_info_from_disk(const std::filesystem::path& artifact_dir) {
  std::filesystem::path verification_path = artifact_dir / "verification.json";
  if (!std::filesystem::exists(verification_path)) {
    return absl::NotFoundError("Verification file not found");
  }

  try {
    std::ifstream file(verification_path);
    if (!file.is_open()) {
      return absl::InternalError(absl::StrFormat("Failed to open verification file: %s", verification_path.string()));
    }

    std::string json_content;
    file.seekg(0, std::ios::end);
    json_content.reserve(file.tellg());
    file.seekg(0, std::ios::beg);
    json_content.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    if (json_content.empty()) {
      return absl::InvalidArgumentError("Verification file is empty");
    }

    absl::StatusOr<ArtifactVerificationInfo> result = ArtifactVerificationInfo::from_json(json_content);
    if (!result.ok()) {
      return absl::InvalidArgumentError(
          absl::StrFormat(
              "Failed to parse verification file %s: %s", verification_path.string(), result.status().message()));
    }

    LOG(INFO) << "Successfully loaded verification info from " << verification_path.string()
              << " (artifact_size: " << result->artifact_size << ", full_hash: 0x" << std::hex << result->full_hash
              << std::dec << ")";

    return result.value();
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrFormat("Exception while loading verification file %s: %s", verification_path.string(), e.what()));
  }
}

DiskLoader::DiskLoader(DiskSource source) : source_(std::move(source)), artifact_size_(0), initialized_(false) {}

absl::Status DiskLoader::initialize() {
  absl::MutexLock lock(&mutex_);

  if (initialized_) {
    return absl::OkStatus();
  }

  // Use the configured path directly as the replica directory
  std::filesystem::path artifact_dir = source_.path;

  // Check if the directory exists
  if (!std::filesystem::exists(artifact_dir)) {
    return absl::NotFoundError(absl::StrFormat("Replica directory not found: %s", artifact_dir.string()));
  }

  if (!std::filesystem::is_directory(artifact_dir)) {
    return absl::InvalidArgumentError(absl::StrFormat("Path is not a directory: %s", artifact_dir.string()));
  }

  // Find all partition files
  partition_paths_.clear();
  partition_sizes_.clear();
  artifact_size_ = 0;

  // First scan and categorize to avoid double-counting when both single-file
  // (tensor.data) and multi-part (tensor.data_*) exist. Prefer multi-part if present.
  bool has_single_file = false;
  std::filesystem::path single_file_path;
  std::vector<std::filesystem::path> multipart_paths;

  for (const auto& entry : std::filesystem::directory_iterator(artifact_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (filename == "tensor.data") {
      has_single_file = true;
      single_file_path = entry.path();
    } else if (filename.starts_with("tensor.data_")) {
      multipart_paths.push_back(entry.path());
    }
  }

  if (!multipart_paths.empty()) {
    // Use only multi-part files
    for (const auto& p : multipart_paths) {
      partition_paths_.push_back(p);
      size_t file_size = std::filesystem::file_size(p);
      partition_sizes_.push_back(file_size);
      artifact_size_ += file_size;
    }
  } else if (has_single_file) {
    // Fallback to single-file format
    partition_paths_.push_back(single_file_path);
    size_t file_size = std::filesystem::file_size(single_file_path);
    partition_sizes_.push_back(file_size);
    artifact_size_ += file_size;
  }

  // If no partitions were detected with the supported patterns, probe for .safetensors files
  if (partition_paths_.empty()) {
    std::vector<std::filesystem::path> safetensors_paths;
    for (const auto& entry : std::filesystem::directory_iterator(artifact_dir)) {
      if (entry.is_regular_file()) {
        const std::string filename = entry.path().filename().string();
        const std::string ext = ".safetensors";
        if (filename.size() >= ext.size() && filename.rfind(ext) == filename.size() - ext.size()) {
          safetensors_paths.push_back(entry.path());
        }
      }
    }
    if (safetensors_paths.empty()) {
      return absl::NotFoundError(
          absl::StrFormat("No replica partition files found in: %s (also no .safetensors)", artifact_dir.string()));
    }
    std::sort(safetensors_paths.begin(), safetensors_paths.end(), [](const auto& a, const auto& b) {
      return a.filename() < b.filename();
    });
    for (const auto& p : safetensors_paths) {
      int fd = ::open(p.c_str(), O_RDONLY);
      if (fd < 0) {
        return absl::NotFoundError(absl::StrFormat("Failed to open %s: %s", p.string(), std::strerror(errno)));
      }
      // Use the shared utility function to parse the header
      auto header_info = loader::ParseSafetensorsHeader(fd);
      ::close(fd);
      if (!header_info.ok()) {
        return header_info.status();
      }
      partition_paths_.push_back(p);
      partition_sizes_.push_back(static_cast<size_t>(header_info->data_size));
      artifact_size_ += header_info->data_size;
    }
  }

  // Sort partitions by name to ensure consistent ordering
  std::vector<std::pair<std::filesystem::path, size_t>> path_size_pairs;
  path_size_pairs.reserve(partition_paths_.size());
  for (size_t i = 0; i < partition_paths_.size(); ++i) {
    path_size_pairs.emplace_back(partition_paths_[i], partition_sizes_[i]);
  }

  std::ranges::sort(
      path_size_pairs, [](const auto& a, const auto& b) { return a.first.filename() < b.first.filename(); });

  partition_paths_.clear();
  partition_sizes_.clear();
  for (const auto& [path, size] : path_size_pairs) {
    partition_paths_.push_back(path);
    partition_sizes_.push_back(size);
  }

  // RFC-0007: For standard partition format, require descriptor and index presence
  // (artifact_descriptor.json + tensor_index.(json|cbor)). Safetensors is exempt (may be backfilled later).
  if (!partition_paths_.empty()) {
    const auto first_name = partition_paths_[0].filename().string();
    const std::string st_ext = ".safetensors";
    const bool is_safetensors =
        first_name.size() >= st_ext.size() && first_name.rfind(st_ext) == first_name.size() - st_ext.size();
    if (!is_safetensors) {
      const auto descriptor_path = artifact_dir / "artifact_descriptor.json";
      const auto index_json_path = artifact_dir / "tensor_index.json";
      const auto index_cbor_path = artifact_dir / "tensor_index.cbor";

      if (!std::filesystem::exists(descriptor_path) ||
          (!std::filesystem::exists(index_json_path) && !std::filesystem::exists(index_cbor_path))) {
        return absl::FailedPreconditionError(
            "ARTIFACT_DESCRIPTOR_REQUIRED: missing artifact_descriptor.json or tensor_index.(json|cbor)");
      }

      // Basic validation of descriptor JSON using nlohmann/json
      try {
        std::ifstream f(descriptor_path);
        if (!f.is_open()) {
          return absl::FailedPreconditionError("ARTIFACT_DESCRIPTOR_REQUIRED: cannot open artifact_descriptor.json");
        }
        nlohmann::json j;
        f >> j;
        if (!j.contains("artifact_id") || !j.contains("index_multihash") || !j.contains("data_multihash")) {
          return absl::InvalidArgumentError(
              "Invalid artifact_descriptor.json: missing required fields (artifact_id, index_multihash, data_multihash)");
        }
        if (!j["artifact_id"].is_string() || !j["index_multihash"].is_string() || !j["data_multihash"].is_string()) {
          return absl::InvalidArgumentError("Invalid artifact_descriptor.json: fields must be strings");
        }
        const std::string artifact_id = j["artifact_id"].get<std::string>();
        if (artifact_id.rfind("mi2:", 0) != 0) {
          return absl::InvalidArgumentError("Invalid artifact_descriptor.json: artifact_id must start with 'mi2:'");
        }
      } catch (const std::exception& e) {
        return absl::InvalidArgumentError(absl::StrFormat("Failed to parse artifact_descriptor.json: %s", e.what()));
      }
    }
  }

  // Try to load verification info
  auto verification_result = load_verification_info_from_disk(artifact_dir);
  if (verification_result.ok()) {
    // Store verification info if needed
    // verification_info_ = verification_result.value();
  }

  LOG(INFO) << "DiskLoader initialized: " << partition_paths_.size() << " partitions, total size: " << artifact_size_
            << " bytes";
  initialized_ = true;

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<loader::SeekableSource>> DiskLoader::open_source() {
  // Ensure initialized
  if (!initialized_) {
    auto st = initialize();
    if (!st.ok()) {
      return st;
    }
  }
  // If paths end with .safetensors, return safetensors sources
  {
    absl::MutexLock lock(&mutex_);
    if (!partition_paths_.empty()) {
      const auto first_name = partition_paths_[0].filename().string();
      const std::string ext = ".safetensors";
      if (first_name.size() >= ext.size() && first_name.rfind(ext) == first_name.size() - ext.size()) {
        if (partition_paths_.size() == 1) {
          return std::make_unique<loader::SafetensorsSource>(partition_paths_[0]);
        }
        return std::make_unique<loader::MultiSafetensorsSource>(partition_paths_);
      }
    }
  }
  loader::FilePartitionSource::Options source_opts;
  {
    absl::MutexLock lock(&mutex_);
    source_opts.partition_paths = partition_paths_;
    source_opts.partition_sizes = partition_sizes_;
    source_opts.total_size = artifact_size_;
    // chunk size is determined by MemoryManager's pinned pool; a default here is fine
    source_opts.chunk_size = 128 * 1024 * 1024;
    source_opts.use_direct_io = (artifact_size_ > 5ULL * 1024 * 1024 * 1024);
  }
  return std::make_unique<loader::FilePartitionSource>(std::move(source_opts));
}

absl::StatusOr<uint64_t> DiskLoader::get_artifact_size() {
  absl::MutexLock lock(&mutex_);
  if (!initialized_) {
    return absl::FailedPreconditionError("DiskLoader not initialized");
  }
  return artifact_size_;
}

absl::StatusOr<ArtifactVerificationInfo> DiskLoader::get_verification_info() const {
  absl::MutexLock lock(&mutex_);
  if (!initialized_) {
    return absl::FailedPreconditionError("DiskLoader not initialized");
  }
  // Return placeholder for now
  return absl::NotFoundError("No verification info available");
}

} // namespace tensorcast::store
