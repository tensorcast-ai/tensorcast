// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_post_seal_utils.h"

#include <algorithm>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

#include "core/common/artifact_hash.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::daemon::materialization_post_seal {

std::vector<uint8_t> compute_view_meta_digest(const store::components::ViewInfo& view) {
  std::vector<store::components::CanonicalRange> ranges = view.canonical_ranges;
  std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
    if (a.offset != b.offset) {
      return a.offset < b.offset;
    }
    return a.length < b.length;
  });

  std::string payload;
  payload.reserve(256 + ranges.size() * 32);
  absl::StrAppend(&payload, "view_id=", view.view_id, ";");
  absl::StrAppend(&payload, "view_data_hash=", view.view_data_hash.value_or(""), ";");
  absl::StrAppend(&payload, "view_size_bytes=", view.view_size_bytes, ";");
  absl::StrAppend(&payload, "canonical_size_bytes=", view.canonical_size_bytes, ";");
  absl::StrAppend(&payload, "canonical_bytes_covered=", view.canonical_bytes_covered, ";");
  for (const auto& range : ranges) {
    absl::StrAppend(&payload, range.offset, ":", range.length, ";");
  }

  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  return tensorcast::common::sha256_digest_bytes(bytes);
}

absl::StatusOr<bool> check_post_seal_view_reuse_safe(
    store::components::IGlobalStoreClient& client,
    std::string_view assembly_id,
    std::string_view mi2_id) {
  if (assembly_id.empty() || mi2_id.empty()) {
    return absl::InvalidArgumentError("check_post_seal_view_reuse_safe requires assembly_id and mi2_id");
  }

  auto layouts_or = client.list_artifact_layouts(mi2_id);
  if (!layouts_or.ok()) {
    return layouts_or.status();
  }
  if (layouts_or->empty()) {
    return true;
  }

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> tensors_by_schema;
  for (const auto& layout_id : *layouts_or) {
    if (layout_id.empty()) {
      continue;
    }
    auto spec_or = client.get_layout_spec(layout_id);
    if (!spec_or.ok()) {
      return spec_or.status();
    }
    const auto& layout_spec = spec_or->layout();
    const std::string schema_version = layout_spec.proof_schema_version();
    for (const auto& entry : layout_spec.tensors()) {
      if (entry.second.overlap_mode() == tensorcast::layout::v1::OVERLAP_MODE_REPLICATE_EQUAL) {
        if (schema_version.empty()) {
          return absl::FailedPreconditionError("proof_schema_version required for replicated tensors");
        }
        tensors_by_schema[schema_version].insert(entry.first);
      }
    }
  }

  if (tensors_by_schema.empty()) {
    return true;
  }

  for (const auto& [schema_version, tensors] : tensors_by_schema) {
    if (tensors.empty()) {
      continue;
    }
    tensorcast::global_store::v1::CheckProofCommitmentsMatchRequest req;
    req.set_assembly_id(std::string(assembly_id));
    req.set_mi2_id(std::string(mi2_id));
    req.set_proof_schema_version(schema_version);
    for (const auto& name : tensors) {
      if (!name.empty()) {
        req.add_tensor_names(name);
      }
    }
    auto resp_or = client.check_proof_commitments_match(req);
    if (!resp_or.ok()) {
      return resp_or.status();
    }
    if (!resp_or->match()) {
      return false;
    }
  }
  return true;
}

} // namespace tensorcast::daemon::materialization_post_seal
