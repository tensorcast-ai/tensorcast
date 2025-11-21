// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/runtime/pipeline/source_adapter.h"

#include <filesystem>
#include <fstream>
#include <system_error>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::materialization::runtime::pipeline {

namespace {

absl::Status validate_descriptor_schema(const DiskSourceMetadata& metadata) {
  if (metadata.descriptor_present && !metadata.schema_version.has_value()) {
    return absl::FailedPreconditionError(
        "artifact_descriptor.json missing schema_version; canonical index v3 required");
  }
  if (metadata.schema_version.has_value() && *metadata.schema_version != "v3") {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported artifact descriptor schema_version='", *metadata.schema_version, "'; v3 required"));
  }
  return absl::OkStatus();
}

} // namespace

absl::Status DiskSourceAdapter::prepare(const loading::DiskSource& source, IngestionContext& ctx) {
  loading::DiskSource resolved_source = source;
  std::filesystem::path artifact_path;
  if (source.path.is_absolute()) {
    artifact_path = source.path;
  } else if (!ctx.storage_path.empty()) {
    artifact_path = ctx.storage_path / source.path;
  } else {
    artifact_path = std::filesystem::absolute(source.path);
  }
  artifact_path = artifact_path.lexically_normal();
  resolved_source.path = artifact_path;

  std::error_code fs_error;
  const bool artifact_exists = std::filesystem::exists(artifact_path, fs_error);
  if (fs_error) {
    return absl::ErrnoToStatus(
        fs_error.value(), absl::StrCat("Failed to access artifact directory '", artifact_path.string(), "'"));
  }
  if (!artifact_exists) {
    return absl::NotFoundError(absl::StrCat("Artifact directory not found: ", artifact_path.string()));
  }
  const bool is_artifact_dir = std::filesystem::is_directory(artifact_path, fs_error);
  if (fs_error) {
    return absl::ErrnoToStatus(
        fs_error.value(), absl::StrCat("Failed to stat artifact directory '", artifact_path.string(), "'"));
  }
  if (!is_artifact_dir) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected artifact path to be a directory: ", artifact_path.string()));
  }

  bool is_safetensors = false;
  std::error_code dir_error;
  std::filesystem::directory_iterator dir_it(artifact_path, dir_error);
  if (dir_error) {
    return absl::ErrnoToStatus(
        dir_error.value(), absl::StrCat("Failed to enumerate artifact directory '", artifact_path.string(), "'"));
  }
  const std::filesystem::directory_iterator dir_end;
  while (dir_it != dir_end) {
    std::error_code entry_error;
    const auto& entry = *dir_it;
    const bool is_regular_file = entry.is_regular_file(entry_error);
    if (entry_error) {
      return absl::ErrnoToStatus(
          entry_error.value(),
          absl::StrCat("Failed to inspect artifact directory entry '", entry.path().string(), "'"));
    }
    if (is_regular_file) {
      const auto name = entry.path().filename().string();
      if (name.ends_with(".safetensors")) {
        is_safetensors = true;
        break;
      }
    }
    dir_it.increment(dir_error);
    if (dir_error) {
      return absl::ErrnoToStatus(
          dir_error.value(), absl::StrCat("Failed to traverse artifact directory '", artifact_path.string(), "'"));
    }
  }

  DiskSourceMetadata metadata;
  metadata.source = resolved_source;
  metadata.artifact_path = artifact_path;
  metadata.is_safetensors = is_safetensors;

  const auto descriptor_path = artifact_path / "artifact_descriptor.json";
  std::error_code descriptor_error;
  const bool descriptor_exists = std::filesystem::exists(descriptor_path, descriptor_error);
  if (descriptor_error) {
    return absl::ErrnoToStatus(
        descriptor_error.value(),
        absl::StrCat("Failed to access artifact descriptor '", descriptor_path.string(), "'"));
  }
  if (descriptor_exists) {
    metadata.descriptor_present = true;
    std::ifstream descriptor_stream(descriptor_path);
    if (descriptor_stream.is_open()) {
      try {
        nlohmann::json descriptor_json;
        descriptor_stream >> descriptor_json;
        if (descriptor_json.contains("schema_version") && descriptor_json["schema_version"].is_string()) {
          metadata.schema_version = descriptor_json["schema_version"].get<std::string>();
        }
        if (descriptor_json.contains("index_multihash") && descriptor_json["index_multihash"].is_string()) {
          metadata.existing_index_multihash = descriptor_json["index_multihash"].get<std::string>();
        }
        if (descriptor_json.contains("data_multihash") && descriptor_json["data_multihash"].is_string()) {
          metadata.existing_data_multihash = descriptor_json["data_multihash"].get<std::string>();
        }
      } catch (const std::exception& ex) {
        LOG(WARNING) << "Ignoring malformed artifact_descriptor.json: " << ex.what();
      }
    }
  }

  absl::Status schema_status = validate_descriptor_schema(metadata);
  if (!schema_status.ok()) {
    return schema_status;
  }

  ctx.disk = std::move(metadata);
  return absl::OkStatus();
}

absl::Status P2PSourceAdapter::prepare(const P2PSource& source, IngestionContext& ctx) {
  auto comm_manager = ctx.runtime_context->communication_manager();
  if (!comm_manager || !comm_manager->is_enabled()) {
    return absl::FailedPreconditionError("Communication not enabled");
  }

  P2PSource normalized = source;
  normalized.comm_engine =
      gsl::not_null<std::shared_ptr<communicator::engine::Communicator>>{comm_manager->get_shared_engine()};
  normalized.fallback_disk_dir = ctx.options->p2p_fallback_disk_dir;

  ctx.p2p.source = std::move(normalized);
  return absl::OkStatus();
}

} // namespace tensorcast::store::materialization::runtime::pipeline
