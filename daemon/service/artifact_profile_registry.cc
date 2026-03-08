// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/artifact_profile_registry.h"

#include <cstdint>
#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "core/common/selection_identity.h"

namespace tensorcast::daemon {

namespace {

std::string to_lower_copy(std::string_view value) {
  std::string out(value);
  absl::AsciiStrToLower(&out);
  return out;
}

class ByteArtifactProfileRuntime final : public ArtifactProfileRuntime {
 public:
  [[nodiscard]] absl::Status validate_artifact_id_for_field(std::string_view artifact_id, std::string_view field_name)
      const override {
    if (artifact_id.empty()) {
      return absl::InvalidArgumentError(absl::StrCat(field_name, " is required"));
    }
    if (!common::is_byte_artifact_id(artifact_id)) {
      return absl::InvalidArgumentError(absl::StrCat(field_name, " must be a byte artifact CGID"));
    }
    const auto validation_st = common::validate_byte_artifact_cgid(artifact_id);
    if (!validation_st.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat(field_name, " must be a valid byte artifact CGID: ", validation_st.message()));
    }
    return absl::OkStatus();
  }

  [[nodiscard]] absl::Status validate_batch_selection(
      const tensorcast::common::v1::ArtifactSelection& selection) const override {
    const auto artifact_id_st = validate_artifact_id_for_field(selection.artifact_id(), "selection.artifact_id");
    if (!artifact_id_st.ok()) {
      return artifact_id_st;
    }
    if (!selection.view_id().empty() || selection.has_view_spec()) {
      return absl::InvalidArgumentError("byte artifact selection does not support view transforms");
    }
    if (!selection.view_subset_hash().empty()) {
      return absl::InvalidArgumentError("byte artifact selection does not support view_subset_hash");
    }
    if (selection.tensor_names_size() > 0) {
      return absl::InvalidArgumentError("byte artifact selection supports full selection only");
    }

    const std::string expected_layout_hash = common::compute_byte_artifact_logical_layout_hash_bytes();
    const std::string expected_selection_hash = common::compute_byte_artifact_selection_hash_bytes();
    if (!selection.logical_layout_hash().empty() && selection.logical_layout_hash() != expected_layout_hash) {
      return absl::InvalidArgumentError("selection.logical_layout_hash does not match byte artifact profile");
    }
    if (!selection.selection_hash().empty() && selection.selection_hash() != expected_selection_hash) {
      return absl::InvalidArgumentError("selection.selection_hash does not match byte artifact profile");
    }
    return absl::OkStatus();
  }

  [[nodiscard]] absl::StatusOr<tensorcast::common::v1::ArtifactSelection> build_normalized_selection(
      std::string_view artifact_id) const override {
    const auto artifact_id_st = validate_artifact_id_for_field(artifact_id, "artifact_id");
    if (!artifact_id_st.ok()) {
      return artifact_id_st;
    }

    tensorcast::common::v1::ArtifactSelection selection;
    selection.set_artifact_id(std::string(artifact_id));
    selection.set_view_id("");
    selection.set_logical_layout_hash(common::compute_byte_artifact_logical_layout_hash_bytes());
    selection.set_selection_hash(common::compute_byte_artifact_selection_hash_bytes());
    return selection;
  }

  [[nodiscard]] absl::StatusOr<std::uint64_t> shard_id_for_artifact(
      std::string_view artifact_id,
      std::uint64_t shard_count) const override {
    if (shard_count == 0) {
      return absl::InvalidArgumentError("shard_count must be > 0");
    }
    const auto digest = common::sha256_digest_bytes(
        absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(artifact_id.data()), artifact_id.size()));
    if (digest.size() < sizeof(std::uint64_t)) {
      return absl::InternalError("sha256 digest is too short");
    }
    std::uint64_t hash64 = 0;
    for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i) {
      hash64 |= static_cast<std::uint64_t>(digest[i]) << (8U * i);
    }
    return hash64 % shard_count;
  }

  [[nodiscard]] absl::Status validate_invariant_body_descriptor(
      const v2::PutIfAbsentInvariant& invariant,
      const BodyDescriptor& descriptor) const override {
    if (descriptor.physical_artifact_id.empty()) {
      return absl::InvalidArgumentError("descriptor.physical_artifact_id is required");
    }
    if (invariant.layout_id().empty()) {
      return absl::InvalidArgumentError("invariant.layout_id is required");
    }
    if (invariant.byte_length() != descriptor.size_bytes) {
      return absl::InvalidArgumentError("invariant.byte_length does not match payload size");
    }
    const std::string digest_alg = to_lower_copy(invariant.payload_digest_alg());
    if (digest_alg != "sha256") {
      return absl::InvalidArgumentError("invariant.payload_digest_alg must be sha256");
    }
    if (descriptor.layout_id != invariant.layout_id()) {
      return absl::InvalidArgumentError("invariant.layout_id does not match payload layout");
    }
    if (descriptor.payload_digest_alg != digest_alg) {
      return absl::InvalidArgumentError("descriptor.digest_alg does not match invariant");
    }
    if (to_lower_copy(invariant.payload_digest_hex()) != descriptor.payload_digest_hex) {
      return absl::InvalidArgumentError("invariant.payload_digest_hex does not match payload digest");
    }
    return absl::OkStatus();
  }
};

class UnknownArtifactProfileRuntime final : public ArtifactProfileRuntime {
 public:
  [[nodiscard]] absl::Status validate_artifact_id_for_field(
      std::string_view /*artifact_id*/,
      std::string_view field_name) const override {
    return absl::InvalidArgumentError(absl::StrCat(field_name, " does not resolve to a supported artifact profile"));
  }

  [[nodiscard]] absl::Status validate_batch_selection(
      const tensorcast::common::v1::ArtifactSelection& selection) const override {
    return absl::InvalidArgumentError(
        absl::StrCat(
            "selection.artifact_id does not resolve to a supported artifact profile: ", selection.artifact_id()));
  }

  [[nodiscard]] absl::StatusOr<tensorcast::common::v1::ArtifactSelection> build_normalized_selection(
      std::string_view /*artifact_id*/) const override {
    return absl::InvalidArgumentError("unsupported artifact profile");
  }

  [[nodiscard]] absl::StatusOr<std::uint64_t> shard_id_for_artifact(
      std::string_view /*artifact_id*/,
      std::uint64_t /*shard_count*/) const override {
    return absl::InvalidArgumentError("unsupported artifact profile");
  }

  [[nodiscard]] absl::Status validate_invariant_body_descriptor(
      const v2::PutIfAbsentInvariant& /*invariant*/,
      const BodyDescriptor& /*descriptor*/) const override {
    return absl::InvalidArgumentError("unsupported artifact profile");
  }
};

const ByteArtifactProfileRuntime& byte_artifact_profile_runtime() {
  static const auto* runtime = new ByteArtifactProfileRuntime();
  return *runtime;
}

const UnknownArtifactProfileRuntime& unknown_artifact_profile_runtime() {
  static const auto* runtime = new UnknownArtifactProfileRuntime();
  return *runtime;
}

} // namespace

const ArtifactProfileRuntime& ArtifactProfileRegistry::runtime_for_profile(Profile profile) {
  switch (profile) {
    case Profile::kByteArtifact:
      return byte_artifact_profile_runtime();
    case Profile::kUnknown:
    default:
      return unknown_artifact_profile_runtime();
  }
}

const ArtifactProfileRuntime& ArtifactProfileRegistry::runtime_for_artifact_id(std::string_view artifact_id) {
  return runtime_for_profile(classify_artifact_id(artifact_id));
}

ArtifactProfileRegistry::Profile ArtifactProfileRegistry::classify_artifact_id(std::string_view artifact_id) {
  if (common::is_byte_artifact_id(artifact_id)) {
    return Profile::kByteArtifact;
  }
  return Profile::kUnknown;
}

} // namespace tensorcast::daemon
