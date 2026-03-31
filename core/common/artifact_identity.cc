// Copyright (c) 2025-2026, TensorCast Team.

#include "core/common/artifact_identity.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

namespace tensorcast::common {
namespace {

constexpr int kCgidMinLength = 8;
constexpr int kCgidMaxLength = 200;
constexpr int kByteArtifactSegmentCount = 7;
constexpr absl::string_view kB64uPrefix = "b64u.";

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

bool is_base64url_char(char c) {
  if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
    return true;
  }
  return c == '-' || c == '_';
}

bool is_delimiter_safe_segment(absl::string_view segment) {
  return !segment.empty() && segment.find('|') == absl::string_view::npos &&
      segment.find('\n') == absl::string_view::npos && segment.find('\r') == absl::string_view::npos &&
      segment.find('~') == absl::string_view::npos;
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

absl::StatusOr<std::vector<absl::string_view>> split_byte_artifact_segments(absl::string_view artifact_id) {
  auto validation = validate_cgid_grammar(artifact_id);
  if (!validation.ok()) {
    return validation;
  }
  const absl::string_view suffix = artifact_id.substr(kCgidPrefix.size());
  std::vector<absl::string_view> segments = absl::StrSplit(suffix, '~');
  if (segments.size() != kByteArtifactSegmentCount || segments[0] != kByteArtifactCgidNamespace) {
    return absl::InvalidArgumentError(
        "byte artifact cgid must match "
        "\"cgid:byte_artifact~<namespace>~<engine>~<model_id_enc>~"
        "<model_version_enc>~<layout_id>~<engine_key_enc>\"");
  }
  for (const auto segment : segments) {
    if (!is_delimiter_safe_segment(segment)) {
      return absl::InvalidArgumentError("byte artifact cgid segments contain forbidden delimiters");
    }
  }
  return segments;
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

bool is_byte_artifact_id(absl::string_view artifact_id) {
  if (!is_cgid_artifact_id(artifact_id)) {
    return false;
  }
  const absl::string_view suffix = artifact_id.substr(kCgidPrefix.size());
  return absl::StartsWith(suffix, absl::StrCat(kByteArtifactCgidNamespace, "~"));
}

absl::StatusOr<ByteArtifactCgidParts> parse_byte_artifact_cgid(absl::string_view artifact_id) {
  auto segments_or = split_byte_artifact_segments(artifact_id);
  if (!segments_or.ok()) {
    return segments_or.status();
  }
  const auto& segments = *segments_or;
  ByteArtifactCgidParts out;
  out.namespace_name = std::string(segments[1]);
  out.engine = std::string(segments[2]);
  out.model_id_enc = std::string(segments[3]);
  out.model_version_enc = std::string(segments[4]);
  out.layout_id = std::string(segments[5]);
  out.engine_key_enc = std::string(segments[6]);
  return out;
}

absl::Status validate_byte_artifact_cgid(absl::string_view artifact_id) {
  auto parsed_or = parse_byte_artifact_cgid(artifact_id);
  if (!parsed_or.ok()) {
    return parsed_or.status();
  }
  return absl::OkStatus();
}

std::string encode_cgid_segment(absl::string_view raw_bytes) {
  std::string encoded = absl::WebSafeBase64Escape(raw_bytes);
  while (!encoded.empty() && encoded.back() == '=') {
    encoded.pop_back();
  }
  return absl::StrCat(kB64uPrefix, encoded);
}

absl::StatusOr<std::string> decode_cgid_segment(absl::string_view encoded_segment) {
  if (!absl::StartsWith(encoded_segment, kB64uPrefix)) {
    return absl::InvalidArgumentError("cgid segment must start with \"b64u.\"");
  }
  std::string payload(encoded_segment.substr(kB64uPrefix.size()));
  if (payload.empty()) {
    return absl::InvalidArgumentError("cgid segment payload is empty");
  }
  for (const char c : payload) {
    if (!is_base64url_char(c)) {
      return absl::InvalidArgumentError("cgid segment payload contains invalid base64url characters");
    }
  }
  const int remainder = static_cast<int>(payload.size() % 4);
  if (remainder != 0) {
    payload.append(static_cast<size_t>(4 - remainder), '=');
  }
  std::string decoded;
  if (!absl::WebSafeBase64Unescape(payload, &decoded)) {
    return absl::InvalidArgumentError("cgid segment payload is not valid base64url");
  }
  return decoded;
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
