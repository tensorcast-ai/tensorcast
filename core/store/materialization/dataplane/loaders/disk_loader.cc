// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/loaders/disk_loader.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include <nlohmann/json.hpp>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "core/common/artifact_verification.h"
#include "core/store/materialization/dataplane/metadata/safetensors_util.h"
#include "core/store/materialization/dataplane/sources/file_partition_source.h"
#include "core/store/materialization/dataplane/sources/multi_safetensors_source.h"
#include "core/store/materialization/dataplane/sources/safetensors_source.h"

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
absl::StatusOr<tensorcast::common::ArtifactVerificationInfo> load_verification_info_from_disk(
    const std::filesystem::path& artifact_dir) {
  std::filesystem::path verification_path = artifact_dir / "verification.json";
  if (!std::filesystem::exists(verification_path)) {
    return absl::NotFoundError("Verification file not found");
  }

  try {
    std::ifstream file(verification_path);
    if (!file.is_open()) {
      return absl::InternalError(absl::StrCat("Failed to open verification file: ", verification_path.string()));
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

    absl::StatusOr<tensorcast::common::ArtifactVerificationInfo> result =
        tensorcast::common::ArtifactVerificationInfo::from_json(json_content);
    if (!result.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat(
              "Failed to parse verification file ", verification_path.string(), ": ", result.status().message()));
    }

    LOG(INFO) << "Successfully loaded verification info from " << verification_path.string()
              << " (artifact_size: " << result->artifact_size << ", full_hash: 0x" << std::hex << result->full_hash
              << std::dec << ")";

    return result.value();
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("Exception while loading verification file ", verification_path.string(), ": ", e.what()));
  }
}

DiskLoader::DiskLoader(loading::DiskSource source)
    : source_(std::move(source)), artifact_size_(0), initialized_(false) {}

absl::Status DiskLoader::initialize() {
  absl::MutexLock lock(&mutex_);

  if (initialized_) {
    return absl::OkStatus();
  }

  // Use the configured path directly as the replica directory
  std::filesystem::path artifact_dir =
      source_.path.is_absolute() ? source_.path : std::filesystem::absolute(source_.path);
  artifact_dir = artifact_dir.lexically_normal();
  auto context_or = loader::get_disk_artifact_context(artifact_dir);
  if (!context_or.ok()) {
    return context_or.status();
  }
  shared_context_ = *context_or;
  source_.path = artifact_dir;
  partition_paths_ = shared_context_->partition_paths();
  partition_sizes_ = shared_context_->partition_sizes();
  artifact_size_ = shared_context_->total_size();

  // RFC-0007: For standard partition format, require descriptor and index presence
  // (artifact_descriptor.json + tensor_index.(json|cbor)). Safetensors is exempt (may be backfilled later).
  if (!partition_paths_.empty()) {
    const auto first_name = partition_paths_[0].filename().string();
    const std::string st_ext = ".safetensors";
    const bool is_safetensors = first_name.ends_with(st_ext);
    const bool enforce_descriptor = source_.require_descriptor && !is_safetensors;
    if (enforce_descriptor) {
      const auto descriptor_path = artifact_dir / "artifact_descriptor.json";
      const auto index_json_path = artifact_dir / "tensor_index.json";
      const auto index_cbor_path = artifact_dir / "tensor_index.cbor";

      if (!shared_context_->descriptor_present() ||
          (!shared_context_->tensor_index_json_present() && !shared_context_->tensor_index_cbor_present())) {
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
        if (!absl::StartsWith(artifact_id, "mi2:")) {
          return absl::InvalidArgumentError("Invalid artifact_descriptor.json: artifact_id must start with 'mi2:'");
        }
      } catch (const std::exception& e) {
        return absl::InvalidArgumentError(absl::StrCat("Failed to parse artifact_descriptor.json: ", e.what()));
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
  std::shared_ptr<const loader::DiskArtifactContext> shared_context;
  {
    absl::MutexLock lock(&mutex_);
    shared_context = shared_context_;
    if (!partition_paths_.empty()) {
      const auto first_name = partition_paths_[0].filename().string();
      const std::string ext = ".safetensors";
      if (first_name.ends_with(ext)) {
        if (shared_context && !shared_context->safetensors_segments().empty()) {
          if (shared_context->safetensors_segments().size() == 1) {
            return std::make_unique<loader::SafetensorsSource>(shared_context->safetensors_segments()[0]);
          }
          return std::make_unique<loader::MultiSafetensorsSource>(shared_context->safetensors_segments());
        }
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
    // chunk size is determined by ReplicaLoadController's pinned pool; a default here is fine
    source_opts.io_batch_bytes = 128 * 1024 * 1024;
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

absl::StatusOr<tensorcast::common::ArtifactVerificationInfo> DiskLoader::get_verification_info() const {
  absl::MutexLock lock(&mutex_);
  if (!initialized_) {
    return absl::FailedPreconditionError("DiskLoader not initialized");
  }
  // Return placeholder for now
  return absl::NotFoundError("No verification info available");
}

} // namespace tensorcast::store
