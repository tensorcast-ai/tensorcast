// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "daemon/service/body_backing_types.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class ArtifactProfileRuntime {
 public:
  virtual ~ArtifactProfileRuntime() = default;

  [[nodiscard]] virtual absl::Status validate_artifact_id_for_field(
      std::string_view artifact_id,
      std::string_view field_name) const = 0;

  [[nodiscard]] virtual absl::Status validate_batch_selection(
      const tensorcast::common::v1::ArtifactSelection& selection) const = 0;

  [[nodiscard]] virtual absl::StatusOr<tensorcast::common::v1::ArtifactSelection> build_normalized_selection(
      std::string_view artifact_id) const = 0;

  [[nodiscard]] virtual absl::StatusOr<std::uint64_t> shard_id_for_artifact(
      std::string_view artifact_id,
      std::uint64_t shard_count) const = 0;

  [[nodiscard]] virtual absl::Status validate_invariant_body_descriptor(
      const v2::PutIfAbsentInvariant& invariant,
      const BodyDescriptor& descriptor) const = 0;
};

class ArtifactProfileRegistry {
 public:
  enum class Profile {
    kUnknown = 0,
    kByteArtifact = 1,
  };

  [[nodiscard]] static const ArtifactProfileRuntime& runtime_for_profile(Profile profile);
  [[nodiscard]] static const ArtifactProfileRuntime& runtime_for_artifact_id(std::string_view artifact_id);

  [[nodiscard]] static Profile classify_artifact_id(std::string_view artifact_id);
};

} // namespace tensorcast::daemon
