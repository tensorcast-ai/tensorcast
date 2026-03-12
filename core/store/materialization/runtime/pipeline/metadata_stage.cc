// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/runtime/pipeline/metadata_stage.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/materialization/dataplane/metadata/safetensors_util.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::materialization::runtime::pipeline {

namespace {

absl::StatusOr<std::string> fetch_canonical_index(
    tensorcast::store::runtime::RuntimeContext& catalog,
    std::string_view artifact_id) {
  auto client = catalog.global_store_client();
  if (!client || !client->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  return client->get_artifact_index_by_id(artifact_id);
}

void update_logical_size(uint64_t candidate, VerificationState& verification) {
  verification.logical_total_size = std::max<uint64_t>(verification.logical_total_size, candidate);
}

std::optional<uint64_t> compute_total_size_from_index(std::string_view index_json) {
  if (index_json.empty()) {
    return std::nullopt;
  }
  try {
    nlohmann::json parsed = nlohmann::json::parse(index_json, nullptr, true);
    if (!parsed.is_object()) {
      return std::nullopt;
    }
    uint64_t total_size = 0;
    for (auto it = parsed.begin(); it != parsed.end(); ++it) {
      const auto& arr = it.value();
      if (!arr.is_array() || arr.size() < 2) {
        continue;
      }
      uint64_t offset = arr[0].get<uint64_t>();
      uint64_t size = arr[1].get<uint64_t>();
      total_size = std::max<uint64_t>(total_size, offset + size);
    }
    if (total_size == 0) {
      return std::nullopt;
    }
    return total_size;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

absl::Status process_disk_canonical_index(IngestionContext& ctx) {
  auto& verification = ctx.verification;
  const auto& disk = ctx.disk;

  if (!verification.canonical_index_json.has_value()) {
    auto artifact_ctx_or = loader::get_disk_artifact_context(disk.artifact_path);
    if (!artifact_ctx_or.ok()) {
      return artifact_ctx_or.status();
    }
    auto index_info_or = (*artifact_ctx_or)->get_index_info(ctx.target_device_id);
    if (index_info_or.ok()) {
      const loader::IndexInfo& info = *index_info_or;
      verification.canonical_index_json = info.canonical_index_json;
      if (!info.index_multihash.empty()) {
        verification.computed_index_multihash = info.index_multihash;
      }
      if (info.total_size_bytes > 0) {
        update_logical_size(info.total_size_bytes, verification);
      }
      if (info.is_safetensors) {
        ctx.disk.is_safetensors = true;
      }
      if (info.source_index_json.has_value()) {
        ctx.disk.source_index_json = info.source_index_json;
      }
      if (info.source_total_size_bytes > 0) {
        ctx.disk.source_total_size_bytes = info.source_total_size_bytes;
      }
    } else if (!absl::IsNotFound(index_info_or.status())) {
      LOG(WARNING) << "Failed to resolve canonical index for '" << disk.artifact_path.string()
                   << "': " << index_info_or.status();
    }
  } else {
    auto info_or = loader::canonicalize_from_raw_json(*verification.canonical_index_json, ctx.target_device_id);
    if (info_or.ok()) {
      const loader::IndexInfo& info = *info_or;
      if (!info.index_multihash.empty()) {
        verification.computed_index_multihash = info.index_multihash;
      }
      if (info.total_size_bytes > 0) {
        update_logical_size(info.total_size_bytes, verification);
      }
    } else if (!absl::IsNotFound(info_or.status())) {
      LOG(WARNING) << "Failed to canonicalize index JSON: " << info_or.status();
    }
  }

  if (!verification.computed_index_multihash.has_value() && disk.existing_index_multihash.has_value()) {
    verification.computed_index_multihash = disk.existing_index_multihash;
  }

  return absl::OkStatus();
}

absl::Status process_variant_hints(IngestionContext& ctx) {
  if (!ctx.hints.variant.has_value()) {
    return absl::OkStatus();
  }

  const auto& variant = *ctx.hints.variant;
  if (variant.cached_plan.has_value()) {
    ctx.resolved_view_plan = variant.cached_plan;
    if (ctx.resolved_view_plan->view_size_bytes > 0) {
      update_logical_size(ctx.resolved_view_plan->view_size_bytes, ctx.verification);
    }
    return absl::OkStatus();
  }

  if (variant.view_spec.has_value()) {
    if (!ctx.verification.canonical_index_json.has_value()) {
      return absl::FailedPreconditionError(
          "Variant view requested but canonical index bytes are unavailable for planning");
    }
    auto plan_or = loader::ViewPlanner::compute_view_plan(*ctx.verification.canonical_index_json, *variant.view_spec);
    if (!plan_or.ok()) {
      return plan_or.status();
    }
    ctx.resolved_view_plan = std::move(*plan_or);
    if (ctx.resolved_view_plan->view_size_bytes > 0) {
      update_logical_size(ctx.resolved_view_plan->view_size_bytes, ctx.verification);
    }
  }
  return absl::OkStatus();
}

absl::Status build_canonical_index_for_safetensors(IngestionContext& ctx) {
  if (!ctx.disk.is_safetensors) {
    return absl::OkStatus();
  }

  auto& verification = ctx.verification;
  const auto& artifact_path = ctx.disk.artifact_path;

  std::vector<std::filesystem::path> safetensor_files;
  for (const auto& entry : std::filesystem::directory_iterator(artifact_path)) {
    if (entry.is_regular_file()) {
      const auto name = entry.path().filename().string();
      if (name.ends_with(".safetensors")) {
        safetensor_files.push_back(entry.path());
      }
    }
  }

  if (safetensor_files.empty()) {
    return absl::OkStatus();
  }

  auto source_bytes_or = loader::BuildSourceIndexFromSafetensors(safetensor_files);
  if (!source_bytes_or.ok()) {
    return source_bytes_or.status();
  }
  auto canonical_bytes_or =
      loader::build_coalesced_canonical_index_from_source_index_json(*source_bytes_or, /*align_bytes=*/8);
  if (!canonical_bytes_or.ok()) {
    return canonical_bytes_or.status();
  }

  verification.canonical_index_json = canonical_bytes_or.value();
  ctx.disk.source_index_json = source_bytes_or.value();
  if (auto total_or = compute_total_size_from_index(*source_bytes_or); total_or.has_value()) {
    ctx.disk.source_total_size_bytes = *total_or;
  }
  if (!verification.computed_index_multihash.has_value()) {
    auto index_mh_or =
        common::compute_index_multihash(std::optional<std::string>(*verification.canonical_index_json), "");
    if (index_mh_or.ok()) {
      verification.computed_index_multihash = *index_mh_or;
    } else {
      LOG(WARNING) << "Index multihash computation failed: " << index_mh_or.status();
    }
  }

  try {
    nlohmann::json parsed = nlohmann::json::parse(*verification.canonical_index_json, nullptr, true);
    for (auto it = parsed.begin(); it != parsed.end(); ++it) {
      const auto& arr = it.value();
      if (!arr.is_array() || arr.size() < 2) {
        continue;
      }
      uint64_t offset = arr[0].get<uint64_t>();
      uint64_t size = arr[1].get<uint64_t>();
      update_logical_size(offset + size, verification);
    }
  } catch (const std::exception& ex) {
    LOG(WARNING) << "Failed to parse canonical index for total_size: " << ex.what();
  }
  return absl::OkStatus();
}

absl::Status build_canonical_index_from_tensor_index(IngestionContext& ctx) {
  const auto index_json_path = ctx.disk.artifact_path / "tensor_index.json";
  if (!std::filesystem::exists(index_json_path)) {
    return absl::OkStatus();
  }

  std::string raw_json;
  try {
    std::ifstream stream(index_json_path);
    if (!stream.is_open()) {
      return absl::OkStatus();
    }
    nlohmann::json parsed;
    stream >> parsed;
    raw_json = parsed.dump();
  } catch (const std::exception& ex) {
    LOG(WARNING) << "Failed to read/parse tensor_index.json: " << ex.what();
    return absl::OkStatus();
  }
  if (raw_json.empty()) {
    return absl::OkStatus();
  }

  auto rebuilt_or = loader::rebuild_stable_canonical_index(raw_json, ctx.target_device_id);
  const std::string& canonical_json = rebuilt_or.ok() ? *rebuilt_or : raw_json;
  ctx.verification.canonical_index_json = canonical_json;
  auto index_mh_or = common::compute_index_multihash(std::optional<std::string>(canonical_json), "");
  if (index_mh_or.ok()) {
    ctx.verification.computed_index_multihash = *index_mh_or;
  } else {
    LOG(WARNING) << "Index multihash computation failed: " << index_mh_or.status();
  }
  try {
    nlohmann::json parsed = nlohmann::json::parse(canonical_json);
    for (auto it = parsed.begin(); it != parsed.end(); ++it) {
      const auto& arr = it.value();
      if (!arr.is_array() || arr.size() < 2) {
        continue;
      }
      uint64_t offset = arr[0].get<uint64_t>();
      uint64_t size = arr[1].get<uint64_t>();
      update_logical_size(offset + size, ctx.verification);
    }
  } catch (const std::exception& ex) {
    LOG(WARNING) << "Failed to parse canonical index JSON for total_size: " << ex.what();
  }
  return absl::OkStatus();
}

} // namespace

absl::Status MetadataStage::process(IngestionContext& ctx) {
  if (ctx.hints.variant.has_value() && ctx.hints.variant->canonical_index_json.has_value()) {
    ctx.verification.canonical_index_json = ctx.hints.variant->canonical_index_json;
  }
  if (!ctx.verification.canonical_index_json.has_value() && ctx.hints.disk_metadata.has_value()) {
    const auto& disk_metadata = *ctx.hints.disk_metadata;
    if (disk_metadata.canonical_index_json.has_value()) {
      ctx.verification.canonical_index_json = disk_metadata.canonical_index_json;
    }
  }
  if (ctx.hints.disk_metadata.has_value()) {
    const auto& disk_metadata = *ctx.hints.disk_metadata;
    if (!ctx.verification.computed_index_multihash.has_value() && disk_metadata.index_multihash.has_value()) {
      ctx.verification.computed_index_multihash = disk_metadata.index_multihash;
    }
    if (disk_metadata.logical_total_size.has_value()) {
      update_logical_size(*disk_metadata.logical_total_size, ctx.verification);
    }
    if (disk_metadata.source_index_json.has_value()) {
      ctx.disk.source_index_json = disk_metadata.source_index_json;
    }
    if (disk_metadata.source_total_size_bytes.has_value()) {
      ctx.disk.source_total_size_bytes = disk_metadata.source_total_size_bytes;
    }
  }

  if (ctx.source_type == SourceType::kDisk) {
    absl::Status disk_status = process_disk_canonical_index(ctx);
    if (!disk_status.ok()) {
      return disk_status;
    }
    if (!ctx.verification.canonical_index_json.has_value()) {
      if (ctx.disk.is_safetensors) {
        absl::Status safetensor_status = build_canonical_index_for_safetensors(ctx);
        if (!safetensor_status.ok()) {
          return safetensor_status;
        }
      } else {
        absl::Status rebuild_status = build_canonical_index_from_tensor_index(ctx);
        if (!rebuild_status.ok()) {
          return rebuild_status;
        }
      }
    }
  } else {
    if (!ctx.verification.canonical_index_json.has_value() && ctx.hints.variant.has_value()) {
      auto idx_or = fetch_canonical_index(*ctx.runtime_context, ctx.artifact_identifier);
      if (!idx_or.ok()) {
        return idx_or.status();
      }
      ctx.verification.canonical_index_json = std::move(idx_or).value();
    }
  }

  return process_variant_hints(ctx);
}

} // namespace tensorcast::store::materialization::runtime::pipeline
