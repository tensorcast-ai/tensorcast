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
    if (descriptor.layout_id != invariant.layout_id()) {
      return absl::InvalidArgumentError("invariant.layout_id does not match payload layout");
    }
    const auto verification_mode = invariant_verification_mode(invariant);
    if (descriptor.verification_mode != verification_mode) {
      return absl::InvalidArgumentError("descriptor.verification_mode does not match invariant");
    }
    const bool digest_pair_present = !invariant.payload_digest_alg().empty() || !invariant.payload_digest_hex().empty();
    if (!invariant.payload_digest_alg().empty() != !invariant.payload_digest_hex().empty()) {
      return absl::InvalidArgumentError(
          "invariant.payload_digest_alg and invariant.payload_digest_hex must both be set");
    }
    if (!verification_mode_requires_payload_digest(verification_mode)) {
      if (!digest_pair_present) {
        return absl::OkStatus();
      }
      if (to_lower_copy(invariant.payload_digest_alg()) != "sha256") {
        return absl::InvalidArgumentError("layout-and-size-only advisory payload_digest_alg must be sha256");
      }
      return absl::OkStatus();
    }
    const std::string digest_alg = to_lower_copy(invariant.payload_digest_alg());
    if (digest_alg != "sha256") {
      return absl::InvalidArgumentError("invariant.payload_digest_alg must be sha256");
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

class OrdinaryArtifactProfileRuntime final : public ArtifactProfileRuntime {
 public:
  [[nodiscard]] absl::Status validate_artifact_id_for_field(std::string_view artifact_id, std::string_view field_name)
      const override {
    if (artifact_id.empty()) {
      return absl::InvalidArgumentError(absl::StrCat(field_name, " is required"));
    }
    const auto artifact_kind_or = common::validate_and_get_artifact_id_kind(artifact_id);
    if (!artifact_kind_or.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat(field_name, " is not a valid artifact id: ", artifact_kind_or.status().message()));
    }
    if (*artifact_kind_or == common::ArtifactIdKind::kCgid) {
      const auto validation_st = common::validate_client_generated_id(artifact_id);
      if (!validation_st.ok()) {
        return absl::InvalidArgumentError(
            absl::StrCat(field_name, " is not a valid client-generated artifact id: ", validation_st.message()));
      }
    }
    return absl::OkStatus();
  }

  [[nodiscard]] absl::Status validate_batch_selection(
      const tensorcast::common::v1::ArtifactSelection& selection) const override {
    return validate_artifact_id_for_field(selection.artifact_id(), "selection.artifact_id");
  }

  [[nodiscard]] absl::StatusOr<tensorcast::common::v1::ArtifactSelection> build_normalized_selection(
      std::string_view artifact_id) const override {
    const auto artifact_id_st = validate_artifact_id_for_field(artifact_id, "artifact_id");
    if (!artifact_id_st.ok()) {
      return artifact_id_st;
    }
    tensorcast::common::v1::ArtifactSelection selection;
    selection.set_artifact_id(std::string(artifact_id));
    return selection;
  }

  [[nodiscard]] absl::StatusOr<std::uint64_t> shard_id_for_artifact(
      std::string_view /*artifact_id*/,
      std::uint64_t /*shard_count*/) const override {
    return absl::FailedPreconditionError("ordinary artifacts do not use routed shard authority");
  }

  [[nodiscard]] absl::Status validate_invariant_body_descriptor(
      const v2::PutIfAbsentInvariant& /*invariant*/,
      const BodyDescriptor& /*descriptor*/) const override {
    return absl::FailedPreconditionError("ordinary artifacts do not use byte-artifact body invariants");
  }
};

class MountedSourceArtifactProfileRuntime final : public ArtifactProfileRuntime {
 public:
  [[nodiscard]] absl::Status validate_artifact_id_for_field(std::string_view artifact_id, std::string_view field_name)
      const override {
    if (artifact_id.empty()) {
      return absl::InvalidArgumentError(absl::StrCat(field_name, " is required"));
    }
    if (!common::is_msa1_artifact_id(artifact_id)) {
      return absl::InvalidArgumentError(absl::StrCat(field_name, " must be an msa1 mounted-source artifact id"));
    }
    return absl::OkStatus();
  }

  [[nodiscard]] absl::Status validate_batch_selection(
      const tensorcast::common::v1::ArtifactSelection& selection) const override {
    return validate_artifact_id_for_field(selection.artifact_id(), "selection.artifact_id");
  }

  [[nodiscard]] absl::StatusOr<tensorcast::common::v1::ArtifactSelection> build_normalized_selection(
      std::string_view artifact_id) const override {
    const auto artifact_id_st = validate_artifact_id_for_field(artifact_id, "artifact_id");
    if (!artifact_id_st.ok()) {
      return artifact_id_st;
    }
    tensorcast::common::v1::ArtifactSelection selection;
    selection.set_artifact_id(std::string(artifact_id));
    return selection;
  }

  [[nodiscard]] absl::StatusOr<std::uint64_t> shard_id_for_artifact(
      std::string_view /*artifact_id*/,
      std::uint64_t /*shard_count*/) const override {
    return absl::FailedPreconditionError("mounted-source artifacts do not use routed shard authority");
  }

  [[nodiscard]] absl::Status validate_invariant_body_descriptor(
      const v2::PutIfAbsentInvariant& /*invariant*/,
      const BodyDescriptor& /*descriptor*/) const override {
    return absl::FailedPreconditionError("mounted-source artifacts do not use byte-artifact body invariants");
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

const OrdinaryArtifactProfileRuntime& ordinary_artifact_profile_runtime() {
  static const auto* runtime = new OrdinaryArtifactProfileRuntime();
  return *runtime;
}

const MountedSourceArtifactProfileRuntime& mounted_source_artifact_profile_runtime() {
  static const auto* runtime = new MountedSourceArtifactProfileRuntime();
  return *runtime;
}

const UnknownArtifactProfileRuntime& unknown_artifact_profile_runtime() {
  static const auto* runtime = new UnknownArtifactProfileRuntime();
  return *runtime;
}

const ArtifactProfileRegistry::ProfileTraits& ordinary_profile_traits() {
  static const auto* traits = new ArtifactProfileRegistry::ProfileTraits{
      .profile = ArtifactProfileRegistry::Profile::kOrdinaryArtifact,
      .profile_name = "ordinary_artifact",
      .family = ArtifactProfileRegistry::ArtifactFamily::kOrdinary,
      .authority_model = ArtifactProfileRegistry::AuthorityModel::kGlobalStoreBacked,
      .fixed_full_selection = false,
  };
  return *traits;
}

const ArtifactProfileRegistry::ProfileTraits& byte_artifact_profile_traits() {
  static const auto* traits = new ArtifactProfileRegistry::ProfileTraits{
      .profile = ArtifactProfileRegistry::Profile::kByteArtifact,
      .profile_name = "byte_artifact",
      .family = ArtifactProfileRegistry::ArtifactFamily::kHighCardinality,
      .authority_model = ArtifactProfileRegistry::AuthorityModel::kRoutedHomeEpoch,
      .fixed_full_selection = true,
  };
  return *traits;
}

const ArtifactProfileRegistry::ProfileTraits& mounted_source_artifact_profile_traits() {
  static const auto* traits = new ArtifactProfileRegistry::ProfileTraits{
      .profile = ArtifactProfileRegistry::Profile::kMountedSourceArtifact,
      .profile_name = "mounted_source_artifact",
      .family = ArtifactProfileRegistry::ArtifactFamily::kOrdinary,
      .authority_model = ArtifactProfileRegistry::AuthorityModel::kDaemonSessionLocal,
      .fixed_full_selection = false,
  };
  return *traits;
}

const ArtifactProfileRegistry::ProfileTraits& unknown_profile_traits() {
  static const auto* traits = new ArtifactProfileRegistry::ProfileTraits{
      .profile = ArtifactProfileRegistry::Profile::kUnknown,
      .profile_name = "unknown",
      .family = ArtifactProfileRegistry::ArtifactFamily::kOrdinary,
      .authority_model = ArtifactProfileRegistry::AuthorityModel::kUnknown,
      .fixed_full_selection = false,
  };
  return *traits;
}

} // namespace

const ArtifactProfileRuntime& ArtifactProfileRegistry::runtime_for_profile(Profile profile) {
  switch (profile) {
    case Profile::kOrdinaryArtifact:
      return ordinary_artifact_profile_runtime();
    case Profile::kByteArtifact:
      return byte_artifact_profile_runtime();
    case Profile::kMountedSourceArtifact:
      return mounted_source_artifact_profile_runtime();
    case Profile::kUnknown:
    default:
      return unknown_artifact_profile_runtime();
  }
}

const ArtifactProfileRuntime& ArtifactProfileRegistry::runtime_for_artifact_id(std::string_view artifact_id) {
  return runtime_for_profile(classify_artifact_id(artifact_id));
}

const ArtifactProfileRegistry::ProfileTraits& ArtifactProfileRegistry::traits_for_profile(Profile profile) {
  switch (profile) {
    case Profile::kOrdinaryArtifact:
      return ordinary_profile_traits();
    case Profile::kByteArtifact:
      return byte_artifact_profile_traits();
    case Profile::kMountedSourceArtifact:
      return mounted_source_artifact_profile_traits();
    case Profile::kUnknown:
    default:
      return unknown_profile_traits();
  }
}

const ArtifactProfileRegistry::ProfileTraits& ArtifactProfileRegistry::traits_for_artifact_id(
    std::string_view artifact_id) {
  return traits_for_profile(classify_artifact_id(artifact_id));
}

ArtifactProfileRegistry::Profile ArtifactProfileRegistry::classify_artifact_id(std::string_view artifact_id) {
  if (artifact_id.empty()) {
    return Profile::kUnknown;
  }
  if (common::is_byte_artifact_id(artifact_id)) {
    return Profile::kByteArtifact;
  }
  switch (common::infer_artifact_id_kind(artifact_id)) {
    case common::ArtifactIdKind::kMi2:
    case common::ArtifactIdKind::kCgid:
      return Profile::kOrdinaryArtifact;
    case common::ArtifactIdKind::kMsa1:
      return Profile::kMountedSourceArtifact;
    case common::ArtifactIdKind::kUnspecified:
    default:
      return Profile::kUnknown;
  }
}

} // namespace tensorcast::daemon
