// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/disk_loader.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "core/common/model_verification.h"
#include "core/store/loader/file_partition_source.h"
#include "core/store/loader/multi_safetensors_source.h"
#include "core/store/loader/safetensors_source.h"
#include "core/store/loader/safetensors_util.h"

namespace stepcast::store {

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
absl::StatusOr<ModelVerificationInfo> load_verification_info_from_disk(const std::filesystem::path& model_dir) {
  std::filesystem::path verification_path = model_dir / "verification.json";
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

    absl::StatusOr<ModelVerificationInfo> result = ModelVerificationInfo::from_json(json_content);
    if (!result.ok()) {
      return absl::InvalidArgumentError(
          absl::StrFormat(
              "Failed to parse verification file %s: %s", verification_path.string(), result.status().message()));
    }

    LOG(INFO) << "Successfully loaded verification info from " << verification_path.string()
              << " (model_size: " << result->model_size << ", full_hash: 0x" << std::hex << result->full_hash
              << std::dec << ")";

    return result.value();
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrFormat("Exception while loading verification file %s: %s", verification_path.string(), e.what()));
  }
}

DiskLoader::DiskLoader(DiskSource source) : source_(std::move(source)), model_size_(0), initialized_(false) {}

absl::Status DiskLoader::initialize() {
  absl::MutexLock lock(&mutex_);

  if (initialized_) {
    return absl::OkStatus();
  }

  // Use the configured path directly as the model directory
  std::filesystem::path model_dir = source_.path;

  // Check if the directory exists
  if (!std::filesystem::exists(model_dir)) {
    return absl::NotFoundError(absl::StrFormat("Model directory not found: %s", model_dir.string()));
  }

  if (!std::filesystem::is_directory(model_dir)) {
    return absl::InvalidArgumentError(absl::StrFormat("Path is not a directory: %s", model_dir.string()));
  }

  // Find all partition files
  partition_paths_.clear();
  partition_sizes_.clear();
  model_size_ = 0;

  // First scan and categorize to avoid double-counting when both single-file
  // (tensor.data) and multi-part (tensor.data_*) exist. Prefer multi-part if present.
  bool has_single_file = false;
  std::filesystem::path single_file_path;
  std::vector<std::filesystem::path> multipart_paths;

  for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
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
      model_size_ += file_size;
    }
  } else if (has_single_file) {
    // Fallback to single-file format
    partition_paths_.push_back(single_file_path);
    size_t file_size = std::filesystem::file_size(single_file_path);
    partition_sizes_.push_back(file_size);
    model_size_ += file_size;
  }

  // If no partitions were detected with the supported patterns, probe for .safetensors files
  if (partition_paths_.empty()) {
    std::vector<std::filesystem::path> safetensors_paths;
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
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
          absl::StrFormat("No model partition files found in: %s (also no .safetensors)", model_dir.string()));
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
      model_size_ += header_info->data_size;
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

  // Try to load verification info
  auto verification_result = load_verification_info_from_disk(model_dir);
  if (verification_result.ok()) {
    // Store verification info if needed
    // verification_info_ = verification_result.value();
  }

  LOG(INFO) << "DiskLoader initialized: " << partition_paths_.size() << " partitions, total size: " << model_size_
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
    source_opts.total_size = model_size_;
    // chunk size is determined by MemoryManager's pinned pool; a default here is fine
    source_opts.chunk_size = 128 * 1024 * 1024;
    source_opts.use_direct_io = (model_size_ > 5ULL * 1024 * 1024 * 1024);
  }
  return std::make_unique<loader::FilePartitionSource>(std::move(source_opts));
}

absl::StatusOr<uint64_t> DiskLoader::get_model_size() {
  absl::MutexLock lock(&mutex_);
  if (!initialized_) {
    return absl::FailedPreconditionError("DiskLoader not initialized");
  }
  return model_size_;
}

absl::StatusOr<ModelVerificationInfo> DiskLoader::get_verification_info() const {
  absl::MutexLock lock(&mutex_);
  if (!initialized_) {
    return absl::FailedPreconditionError("DiskLoader not initialized");
  }
  // Return placeholder for now
  return absl::NotFoundError("No verification info available");
}

} // namespace stepcast::store
