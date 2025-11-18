// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/runtime/pipeline/verification_stage.h"

#include <filesystem>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_verification.h"
#include "core/store/materialization/common/view_hash_utils.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/verification/verification_utils.h"

namespace tensorcast::store::materialization::runtime::pipeline {

namespace {

void maybe_compute_view_hash(IngestionContext& ctx) {
  if (!ctx.resolved_view_plan.has_value() || ctx.resolved_view_plan->is_identity) {
    return;
  }
  auto catalog = ctx.component_catalog;
  if (catalog == nullptr) {
    return;
  }
  auto hasher = catalog->view_hash_computer();
  if (!hasher) {
    return;
  }
  if (!ctx.replica) {
    return;
  }
  auto hash = hasher->hash_replica_view(
      *ctx.replica,
      ctx.target_location,
      ctx.resolved_view_plan->view_size_bytes,
      ctx.target_is_gpu ? std::optional<int>(ctx.target_device_id) : std::nullopt);
  if (hash.has_value()) {
    ctx.verification.view_data_hash = std::move(hash);
  }
}

absl::Status compute_full_digest_if_needed(IngestionContext& ctx) {
  auto& replica = ctx.replica;
  auto& verification = ctx.verification;

  const bool force_full_digest = ctx.options->force_full_digest_on_load || ctx.disk.is_safetensors;
  if (ctx.hints.verify != loading::MaterializeHints::Verify::FULL_DIGEST && !force_full_digest) {
    return absl::OkStatus();
  }

  uint64_t verify_size = 0;
  if (auto sz_or = replica->get_artifact_size(); sz_or.ok()) {
    verify_size = *sz_or;
  }
  if (verify_size == 0) {
    return absl::OkStatus();
  }

  if (ctx.target_location == common::memory::MemoryLocation::GPU) {
    auto view_or = replica->get_memory_manager().get_gpu_allocation_view();
    if (view_or.ok() && view_or->base_ptr != nullptr) {
      [[maybe_unused]] auto keep_allocation = view_or->allocation;
      auto data_mh_or = tensorcast::common::compute_data_multihash_from_gpu(
          gsl::not_null<void*>{view_or->base_ptr}, verify_size, ctx.target_device_id);
      if (data_mh_or.ok()) {
        verification.computed_data_multihash = *data_mh_or;
      } else {
        LOG(WARNING) << "Data multihash computation (GPU) failed: " << data_mh_or.status();
      }
    }
  } else {
    const auto cpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::CPU);
    if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr) {
      auto data_mh_or =
          loader::compute_data_multihash_from_cpu_memory(gsl::not_null<const void*>{cpu_ptrs[0]}, verify_size);
      if (data_mh_or.ok()) {
        verification.computed_data_multihash = *data_mh_or;
      } else {
        LOG(WARNING) << "Data multihash computation (CPU) failed: " << data_mh_or.status();
      }
    }
  }

  return absl::OkStatus();
}

absl::Status verify_descriptor_consistency(const IngestionContext& ctx) {
  if (!ctx.disk.descriptor_present || !ctx.disk.existing_data_multihash.has_value()) {
    return absl::OkStatus();
  }
  if (!ctx.verification.computed_data_multihash.has_value()) {
    return absl::OkStatus();
  }
  if (*ctx.disk.existing_data_multihash != *ctx.verification.computed_data_multihash) {
    return absl::DataLossError("ARTIFACT_ID_MISMATCH: data_multihash does not match loaded data");
  }
  return absl::OkStatus();
}

absl::Status ensure_descriptor_for_safetensors(IngestionContext& ctx) {
  if (!ctx.disk.is_safetensors) {
    return absl::OkStatus();
  }
  const auto descriptor_path = ctx.disk.artifact_path / "artifact_descriptor.json";
  if (std::filesystem::exists(descriptor_path)) {
    return absl::OkStatus();
  }

  if (!ctx.verification.computed_index_multihash.has_value() || !ctx.verification.computed_data_multihash.has_value()) {
    return absl::FailedPreconditionError("Cannot write descriptor without index/data multihash");
  }

  uint64_t total_bytes = ctx.verification.logical_total_size;
  if (total_bytes == 0) {
    if (auto sz_or = ctx.replica->get_artifact_size(); sz_or.ok()) {
      total_bytes = *sz_or;
    }
  }

  return loader::verification::write_descriptor_if_absent(
      ctx.disk.artifact_path,
      *ctx.verification.computed_index_multihash,
      *ctx.verification.computed_data_multihash,
      total_bytes,
      "json");
}

absl::Status handle_variant_verification(IngestionContext& ctx) {
  if (!ctx.hints.variant.has_value()) {
    return absl::OkStatus();
  }
  if (!ctx.hints.variant->view_id.has_value()) {
    return absl::OkStatus();
  }

  const auto descriptor_path = ctx.disk.artifact_path / "artifact_descriptor.json";
  const bool allow_verification_metadata =
      ctx.hints.variant->view_id.has_value() && !ctx.hints.variant->view_id->empty();

  if (!std::filesystem::exists(descriptor_path) || !allow_verification_metadata) {
    return absl::OkStatus();
  }

  auto expected_byte_space_id = *ctx.hints.variant->view_id;
  loader::verification::MemoryView verification_view;
  verification_view.location = ctx.target_location;
  verification_view.size_bytes = ctx.resolved_view_plan.has_value() ? ctx.resolved_view_plan->view_size_bytes : 0;
  verification_view.gpu_device_id = ctx.target_is_gpu ? std::optional<int>(ctx.target_device_id) : std::nullopt;
  const auto data_ptrs = ctx.replica->get_data_pointer(ctx.target_location);
  if (data_ptrs.empty() || data_ptrs[0] == nullptr) {
    return absl::OkStatus();
  }
  verification_view.base_ptr = data_ptrs[0];

  absl::Status verification_status = loader::verification::reuse_or_generate_verification_json(
      ctx.disk.artifact_path, expected_byte_space_id, verification_view);
  if (!verification_status.ok()) {
    return verification_status;
  }
  return absl::OkStatus();
}

absl::Status handle_canonical_verification(IngestionContext& ctx) {
  if (ctx.hints.variant.has_value()) {
    return absl::OkStatus();
  }
  const auto descriptor_path = ctx.disk.artifact_path / "artifact_descriptor.json";
  if (!std::filesystem::exists(descriptor_path)) {
    return absl::OkStatus();
  }
  if (!ctx.replica) {
    return absl::OkStatus();
  }

  loader::verification::MemoryView verification_view;
  verification_view.location = ctx.target_location;
  verification_view.gpu_device_id = ctx.target_is_gpu ? std::optional<int>(ctx.target_device_id) : std::nullopt;

  uint64_t view_size = ctx.verification.logical_total_size;
  if (view_size == 0) {
    if (auto sz_or = ctx.replica->get_artifact_size(); sz_or.ok()) {
      view_size = *sz_or;
    }
  }
  verification_view.size_bytes = view_size;

  const auto data_ptrs = ctx.replica->get_data_pointer(ctx.target_location);
  if (data_ptrs.empty() || data_ptrs[0] == nullptr || verification_view.size_bytes == 0) {
    return absl::OkStatus();
  }

  verification_view.base_ptr = data_ptrs[0];
  return loader::verification::reuse_or_generate_verification_json(ctx.disk.artifact_path, "", verification_view);
}

absl::Status verify_disk(IngestionContext& ctx) {
  maybe_compute_view_hash(ctx);

  absl::Status status = compute_full_digest_if_needed(ctx);
  if (!status.ok()) {
    return status;
  }
  status = verify_descriptor_consistency(ctx);
  if (!status.ok()) {
    return status;
  }
  status = handle_variant_verification(ctx);
  if (!status.ok()) {
    return status;
  }
  status = handle_canonical_verification(ctx);
  if (!status.ok()) {
    return status;
  }
  return ensure_descriptor_for_safetensors(ctx);
}

absl::Status verify_p2p(IngestionContext& ctx) {
  const auto& source = ctx.p2p.source;
  if (!source.verification_json.empty()) {
    auto info_or = common::ArtifactVerificationInfo::from_json(source.verification_json);
    if (!info_or.ok()) {
      return absl::DataLossError("verification_json parse failed");
    }
    auto verify_status = ctx.replica->verify_key_points(ctx.target_location, *info_or);
    if (!verify_status.ok()) {
      return absl::DataLossError(std::string(verify_status.message()));
    }
  }

  maybe_compute_view_hash(ctx);

  return absl::OkStatus();
}

} // namespace

absl::Status VerificationStage::verify(IngestionContext& ctx) {
  if (ctx.source_type == SourceType::kDisk) {
    return verify_disk(ctx);
  }
  return verify_p2p(ctx);
}

} // namespace tensorcast::store::materialization::runtime::pipeline
