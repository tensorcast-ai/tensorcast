// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/runtime/pipeline/verification_stage.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_verification.h"
#include "core/store/materialization/common/view_hash_utils.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/verification/verification_utils.h"

namespace tensorcast::store::materialization::runtime::pipeline {

namespace {

bool source_mutation_allowed(const IngestionContext& ctx) {
  return ctx.hints.source_mutation_policy != loading::SourceMutationPolicy::kReadOnly;
}

void maybe_compute_view_hash(IngestionContext& ctx) {
  if (!ctx.resolved_view_plan.has_value() || ctx.resolved_view_plan->is_identity) {
    return;
  }
  auto context = ctx.runtime_context;
  if (context == nullptr) {
    return;
  }
  auto hasher = context->view_hash_computer();
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

  const FullDigestDecision digest_decision = resolve_full_digest_decision(ctx);
  if (!digest_decision.should_compute) {
    if (digest_decision.trusted_existing_data_multihash && ctx.disk.is_safetensors) {
      VLOG(1) << "Skipping safetensors full digest for artifact_id=" << ctx.artifact_identifier
              << " (trusted existing data_multihash under read-only source policy)";
    }
    return absl::OkStatus();
  }

  uint64_t verify_size = 0;
  if (auto sz_or = replica->get_artifact_size(); sz_or.ok()) {
    verify_size = *sz_or;
  }
  if (verify_size == 0) {
    return absl::OkStatus();
  }

  const absl::Time digest_start = absl::Now();
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
      int last_logged_bucket = -1;
      auto progress_cb = [&](uint64_t hashed_leaf_count, uint64_t total_hash_leaves) {
        if (total_hash_leaves == 0) {
          return;
        }
        const int bucket = static_cast<int>((hashed_leaf_count * 10) / total_hash_leaves);
        if (bucket == last_logged_bucket) {
          return;
        }
        if (bucket == 0 && hashed_leaf_count != total_hash_leaves) {
          return;
        }
        last_logged_bucket = bucket;

        const uint64_t scaled = total_hash_leaves == 0 ? 0 : (hashed_leaf_count * verify_size) / total_hash_leaves;
        const uint64_t processed_bytes = std::min<uint64_t>(verify_size, scaled);
        const uint64_t percent = verify_size == 0 ? 0 : (processed_bytes * 100) / verify_size;
        LOG(INFO) << "full_digest_progress artifact_id=" << ctx.artifact_identifier
                  << " location=cpu processed_bytes=" << processed_bytes << "/" << verify_size
                  << " percent=" << percent;
      };
      auto data_mh_or = loader::compute_data_multihash_from_cpu_memory(
          gsl::not_null<const void*>{cpu_ptrs[0]}, verify_size, /*leaf_chunk_bytes=*/4ULL * 1024 * 1024, progress_cb);
      if (data_mh_or.ok()) {
        verification.computed_data_multihash = *data_mh_or;
      } else {
        LOG(WARNING) << "Data multihash computation (CPU) failed: " << data_mh_or.status();
      }
    }
  }

  LOG(INFO) << "full_digest_complete artifact_id=" << ctx.artifact_identifier
            << " location=" << (ctx.target_location == common::memory::MemoryLocation::GPU ? "gpu" : "cpu")
            << " duration=" << absl::FormatDuration(absl::Now() - digest_start);

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
  if (!source_mutation_allowed(ctx)) {
    return absl::OkStatus();
  }
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
  if (!source_mutation_allowed(ctx)) {
    return absl::OkStatus();
  }
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
  std::shared_ptr<common::memory::GpuDeviceMemory> gpu_allocation_guard;
  if (ctx.target_is_gpu) {
    // Verification outlives the controller lock. Keep the allocation alive
    // while a concurrent unload removes it from the replica/UMA ledgers.
    auto view_or = ctx.replica->get_memory_manager().get_gpu_allocation_view();
    if (!view_or.ok()) {
      // Preserve immediate-unload semantics: if release won the race there is
      // no resident view left to verify, and materialization remains tolerant.
      return absl::OkStatus();
    }
    gpu_allocation_guard = view_or->allocation;
    verification_view.base_ptr = view_or->base_ptr;
  } else {
    const auto data_ptrs = ctx.replica->get_data_pointer(ctx.target_location);
    if (data_ptrs.empty() || data_ptrs[0] == nullptr) {
      return absl::OkStatus();
    }
    verification_view.base_ptr = data_ptrs[0];
  }

  absl::Status verification_status = loader::verification::reuse_or_generate_verification_json(
      ctx.disk.artifact_path, expected_byte_space_id, verification_view);
  (void)gpu_allocation_guard;
  if (!verification_status.ok()) {
    return verification_status;
  }
  return absl::OkStatus();
}

absl::Status handle_canonical_verification(IngestionContext& ctx) {
  if (!source_mutation_allowed(ctx)) {
    return absl::OkStatus();
  }
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

  if (verification_view.size_bytes == 0) {
    return absl::OkStatus();
  }

  std::shared_ptr<common::memory::GpuDeviceMemory> gpu_allocation_guard;
  if (ctx.target_is_gpu) {
    // Do not hand a bare GPU pointer to the verifier: unload_replica() may
    // otherwise free the allocation before the verifier finishes reading it.
    auto view_or = ctx.replica->get_memory_manager().get_gpu_allocation_view();
    if (!view_or.ok()) {
      // Preserve immediate-unload semantics: if release won the race there is
      // no resident view left to verify, and materialization remains tolerant.
      return absl::OkStatus();
    }
    gpu_allocation_guard = view_or->allocation;
    verification_view.base_ptr = view_or->base_ptr;
  } else {
    const auto data_ptrs = ctx.replica->get_data_pointer(ctx.target_location);
    if (data_ptrs.empty() || data_ptrs[0] == nullptr) {
      return absl::OkStatus();
    }
    verification_view.base_ptr = data_ptrs[0];
  }

  auto status =
      loader::verification::reuse_or_generate_verification_json(ctx.disk.artifact_path, "", verification_view);
  (void)gpu_allocation_guard;
  return status;
}

absl::Status verify_disk(IngestionContext& ctx) {
  if (should_skip_disk_verification(ctx)) {
    return absl::OkStatus();
  }

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
  const bool is_view_request = ctx.resolved_view_plan.has_value() && !ctx.resolved_view_plan->is_identity &&
      ctx.resolved_view_plan->view_size_bytes > 0;
  const uint64_t expected_view_size = is_view_request ? ctx.resolved_view_plan->view_size_bytes : 0;
  if (!source.verification_json.empty()) {
    auto info_or = common::ArtifactVerificationInfo::from_json(source.verification_json);
    if (!info_or.ok()) {
      return absl::DataLossError("verification_json parse failed");
    }
    if (is_view_request && info_or->artifact_size != expected_view_size) {
      LOG(INFO) << "verify_p2p: skipping incompatible canonical verification metadata for view materialization"
                << " artifact_id=" << ctx.artifact_identifier << " expected_view_size=" << expected_view_size
                << " verification_size=" << info_or->artifact_size << " view_id="
                << (ctx.hints.variant.has_value() && ctx.hints.variant->view_id.has_value()
                        ? *ctx.hints.variant->view_id
                        : std::string("<unknown>"))
                << " need_view_data_hash=" << ctx.hints.need_view_data_hash;
    } else {
      auto verify_status = ctx.replica->verify_key_points(ctx.target_location, *info_or);
      if (!verify_status.ok()) {
        return absl::DataLossError(std::string(verify_status.message()));
      }
    }
  }

  maybe_compute_view_hash(ctx);

  return absl::OkStatus();
}

} // namespace

FullDigestDecision resolve_full_digest_decision(const IngestionContext& ctx) {
  FullDigestDecision decision;
  decision.forced_by_hint = ctx.hints.verify == loading::MaterializeHints::Verify::FULL_DIGEST;
  decision.forced_by_engine_option = ctx.options != nullptr && ctx.options->force_full_digest_on_load;

  const bool has_existing_data_multihash =
      ctx.disk.existing_data_multihash.has_value() && !ctx.disk.existing_data_multihash->empty();
  decision.trusted_existing_data_multihash =
      has_existing_data_multihash && ctx.hints.source_mutation_policy == loading::SourceMutationPolicy::kReadOnly;
  decision.forced_by_safetensors = ctx.disk.is_safetensors && !decision.trusted_existing_data_multihash;
  decision.should_compute =
      decision.forced_by_hint || decision.forced_by_engine_option || decision.forced_by_safetensors;
  return decision;
}

bool should_skip_disk_verification(const IngestionContext& ctx) {
  if (ctx.source_type != SourceType::kDisk) {
    return false;
  }
  if (source_mutation_allowed(ctx)) {
    return false;
  }
  if (ctx.hints.export_policy != loading::ExportPolicy::kNever) {
    return false;
  }
  return ctx.hints.verify != loading::MaterializeHints::Verify::FULL_DIGEST;
}

absl::Status VerificationStage::verify(IngestionContext& ctx) {
  if (ctx.source_type == SourceType::kDisk) {
    return verify_disk(ctx);
  }
  return verify_p2p(ctx);
}

} // namespace tensorcast::store::materialization::runtime::pipeline
