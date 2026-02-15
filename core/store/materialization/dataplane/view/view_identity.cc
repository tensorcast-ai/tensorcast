// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/dataplane/view/view_identity.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"

namespace tensorcast::store::loader {
namespace {

using materialization::view::ViewOp;

template <typename T>
void append_le(std::string& out, T value) {
  static_assert(std::is_integral_v<T>, "append_le requires integral type");
  using Unsigned = std::make_unsigned_t<T>;
  Unsigned bits = static_cast<Unsigned>(value);
  for (size_t i = 0; i < sizeof(Unsigned); ++i) {
    out.push_back(static_cast<char>((bits >> (i * 8)) & static_cast<Unsigned>(0xFF)));
  }
}

void append_string(std::string& out, std::string_view value) {
  append_le<std::uint32_t>(out, static_cast<std::uint32_t>(value.size()));
  out.append(value.data(), value.size());
}

} // namespace

std::string canonicalize_view_spec_for_identity(const ViewSpec& spec) {
  std::string payload;
  payload.reserve(32 + spec.tensors.size() * 64);

  constexpr std::uint32_t kEncodingVersion = 1;
  append_le<std::uint32_t>(payload, kEncodingVersion);
  append_le<std::uint32_t>(payload, static_cast<std::uint32_t>(spec.tensors.size()));

  for (const auto& [tensor_name, tensor_ops] : spec.tensors) {
    append_string(payload, tensor_name);
    append_le<std::uint32_t>(payload, static_cast<std::uint32_t>(tensor_ops.ops.size()));
    for (const auto& op : tensor_ops.ops) {
      switch (op.kind) {
        case ViewOp::Kind::kNarrow:
          payload.push_back(static_cast<char>(0x00));
          append_le<std::int32_t>(payload, op.narrow.dim);
          append_le<std::int64_t>(payload, op.narrow.start);
          append_le<std::uint64_t>(payload, op.narrow.length);
          break;
        case ViewOp::Kind::kTranspose:
          payload.push_back(static_cast<char>(0x01));
          append_le<std::int32_t>(payload, op.transpose.dim0);
          append_le<std::int32_t>(payload, op.transpose.dim1);
          break;
      }
    }
  }
  return payload;
}

absl::StatusOr<std::string> compute_view_id_from_spec(const ViewSpec& spec, std::string_view canonical_index_json) {
  if (spec.tensors.empty()) {
    return absl::InvalidArgumentError("view spec must be non-empty for view_id computation");
  }
  auto index_mh_or = common::compute_index_multihash(std::optional<std::string>(canonical_index_json), "");
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }

  const std::string spec_bytes = canonicalize_view_spec_for_identity(spec);
  std::vector<uint8_t> buffer;
  buffer.reserve(spec_bytes.size() + index_mh_or->size());
  buffer.insert(buffer.end(), spec_bytes.begin(), spec_bytes.end());
  buffer.insert(buffer.end(), index_mh_or->begin(), index_mh_or->end());
  const std::vector<uint8_t> digest = common::sha256_digest_bytes(absl::Span<const uint8_t>(buffer));
  return common::multibase_multihash_sha256(digest);
}

} // namespace tensorcast::store::loader
