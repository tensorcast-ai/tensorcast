// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace tensorcast::common {

// Enumeration describing supported artifact identity prefixes.
enum class ArtifactIdKind {
  kUnspecified = 0,
  kMi2 = 1,
  kCgid = 2,
  kMsa1 = 3,
};

constexpr absl::string_view kMi2Prefix = "mi2:";
constexpr absl::string_view kCgidPrefix = "cgid:";
constexpr absl::string_view kMsa1Prefix = "msa1:";
constexpr absl::string_view kByteArtifactCgidNamespace = "byte_artifact";

// Returns true when artifact_id begins with the mi2 prefix.
bool is_mi2_artifact_id(absl::string_view artifact_id);

// Returns true when artifact_id begins with the CGID prefix.
bool is_cgid_artifact_id(absl::string_view artifact_id);

// Returns true when artifact_id begins with the MSA1 prefix.
bool is_msa1_artifact_id(absl::string_view artifact_id);

// Validate a client-generated artifact identity (cgid prefix and grammar).
absl::Status validate_client_generated_id(absl::string_view artifact_id);

struct ByteArtifactCgidParts {
  std::string namespace_name;
  std::string engine;
  std::string model_id_enc;
  std::string model_version_enc;
  std::string layout_id;
  std::string engine_key_enc;
};

// Returns true when artifact_id is in the byte-artifact profile namespace.
bool is_byte_artifact_id(absl::string_view artifact_id);

// Parse a byte-artifact profile CGID.
absl::StatusOr<ByteArtifactCgidParts> parse_byte_artifact_cgid(absl::string_view artifact_id);

// Validate that artifact_id is a syntactically valid byte-artifact profile CGID.
absl::Status validate_byte_artifact_cgid(absl::string_view artifact_id);

// Encode/decode delimiter-safe CGID segments as b64u.<base64url_nopad>.
std::string encode_cgid_segment(absl::string_view raw_bytes);
absl::StatusOr<std::string> decode_cgid_segment(absl::string_view encoded_segment);

// Validate a generic artifact identity (mi2 or cgid) and return its kind.
absl::StatusOr<ArtifactIdKind> validate_and_get_artifact_id_kind(absl::string_view artifact_id);

// Lightweight classification helper; does not validate grammar.
ArtifactIdKind infer_artifact_id_kind(absl::string_view artifact_id);

} // namespace tensorcast::common
