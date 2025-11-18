// Copyright (c) 2025, TensorCast Team.

#include "core/common/artifact_identity.h"

#include <cctype>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"

namespace tensorcast::common {
namespace {

constexpr int kCgidMinLength = 8;
constexpr int kCgidMaxLength = 200;

bool is_ascii(absl::string_view value) {
  for (unsigned char c : value) {
    if (c > 0x7F) {
      return false;
    }
  }
  return true;
}

bool is_cgid_body_char(char c) {
  if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
    return true;
  }
  switch (c) {
    case '-':
    case '.':
    case '_':
    case '~':
      return true;
    default:
      return false;
  }
}

absl::Status validate_cgid_grammar(absl::string_view artifact_id) {
  if (!is_cgid_artifact_id(artifact_id)) {
    return absl::InvalidArgumentError("cgid must start with \"cgid:\" prefix");
  }
  if (!is_ascii(artifact_id)) {
    return absl::InvalidArgumentError("cgid must be ASCII");
  }
  const int length = static_cast<int>(artifact_id.size());
  if (length < kCgidMinLength || length > kCgidMaxLength) {
    return absl::InvalidArgumentError("cgid length must be between 8 and 200 characters");
  }
  const absl::string_view suffix = artifact_id.substr(kCgidPrefix.size());
  if (suffix.empty()) {
    return absl::InvalidArgumentError("cgid must include identifier segment after prefix");
  }
  for (char c : suffix) {
    if (!is_cgid_body_char(c)) {
      return absl::InvalidArgumentError("cgid contains invalid characters");
    }
  }
  return absl::OkStatus();
}

} // namespace

bool is_mi2_artifact_id(absl::string_view artifact_id) {
  return absl::StartsWith(artifact_id, kMi2Prefix);
}

bool is_cgid_artifact_id(absl::string_view artifact_id) {
  return absl::StartsWith(artifact_id, kCgidPrefix);
}

absl::Status validate_client_generated_id(absl::string_view artifact_id) {
  return validate_cgid_grammar(artifact_id);
}

ArtifactIdKind infer_artifact_id_kind(absl::string_view artifact_id) {
  if (is_mi2_artifact_id(artifact_id)) {
    return ArtifactIdKind::kMi2;
  }
  if (is_cgid_artifact_id(artifact_id)) {
    return ArtifactIdKind::kCgid;
  }
  return ArtifactIdKind::kUnspecified;
}

absl::StatusOr<ArtifactIdKind> validate_and_get_artifact_id_kind(absl::string_view artifact_id) {
  const ArtifactIdKind kind = infer_artifact_id_kind(artifact_id);
  switch (kind) {
    case ArtifactIdKind::kMi2:
      return kind;
    case ArtifactIdKind::kCgid: {
      auto st = validate_cgid_grammar(artifact_id);
      if (!st.ok()) {
        return st;
      }
      return kind;
    }
    case ArtifactIdKind::kUnspecified:
    default:
      return absl::InvalidArgumentError(R"(artifact_id must start with "mi2:" or "cgid:" prefix)");
  }
}

} // namespace tensorcast::common
