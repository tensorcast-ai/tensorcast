// Copyright (c) 2025, TensorCast Team.

#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace tensorcast::common {

// Enumeration describing supported artifact identity prefixes.
enum class ArtifactIdKind {
  kUnspecified = 0,
  kMi2 = 1,
  kCgid = 2,
};

constexpr absl::string_view kMi2Prefix = "mi2:";
constexpr absl::string_view kCgidPrefix = "cgid:";

// Returns true when artifact_id begins with the mi2 prefix.
bool is_mi2_artifact_id(absl::string_view artifact_id);

// Returns true when artifact_id begins with the CGID prefix.
bool is_cgid_artifact_id(absl::string_view artifact_id);

// Validate a client-generated artifact identity (cgid prefix and grammar).
absl::Status validate_client_generated_id(absl::string_view artifact_id);

// Validate a generic artifact identity (mi2 or cgid) and return its kind.
absl::StatusOr<ArtifactIdKind> validate_and_get_artifact_id_kind(absl::string_view artifact_id);

// Lightweight classification helper; does not validate grammar.
ArtifactIdKind infer_artifact_id_kind(absl::string_view artifact_id);

} // namespace tensorcast::common
