// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/lip_metadata_utils.h"

#include <algorithm>
#include <unordered_map>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"

namespace tensorcast::daemon {

absl::StatusOr<std::string> build_canonical_index_from_metadata(
    absl::Span<const LeaseSegMeta> segments,
    absl::Span<const RegisterStorageMeta> storages,
    absl::Span<const RegisterTensorAliasMeta> aliases,
    int device_id) {
  if (storages.empty() || aliases.empty()) {
    return std::string{};
  }

  absl::flat_hash_map<std::string, uint64_t> storage_to_dst;
  storage_to_dst.reserve(segments.size());
  for (const auto& seg : segments) {
    auto [it, inserted] = storage_to_dst.emplace(seg.storage_id, seg.artifact_offset);
    if (!inserted && it->second != seg.artifact_offset) {
      return absl::InvalidArgumentError(absl::StrCat("conflicting artifact_offset for storage_id=", seg.storage_id));
    }
  }

  absl::flat_hash_map<std::string, const RegisterStorageMeta*> storage_by_id;
  storage_by_id.reserve(storages.size());
  for (const auto& storage : storages) {
    storage_by_id.emplace(storage.storage_id, &storage);
  }

  absl::flat_hash_map<std::string, const RegisterTensorAliasMeta*> alias_by_name;
  alias_by_name.reserve(aliases.size());
  for (const auto& alias : aliases) {
    if (!alias_by_name.emplace(alias.name, &alias).second) {
      return absl::InvalidArgumentError(absl::StrCat("duplicate tensor alias name: ", alias.name));
    }
  }

  std::vector<std::string> ordered_names;
  ordered_names.reserve(aliases.size());
  std::unordered_map<std::string, uint64_t> offsets;
  offsets.reserve(aliases.size());
  std::unordered_map<std::string, uint64_t> storage_sizes;
  storage_sizes.reserve(aliases.size());
  std::unordered_map<std::string, store::loader::CanonicalTensorMeta> metas;
  metas.reserve(aliases.size());

  for (const auto& [name, alias_ptr] : alias_by_name) {
    const RegisterTensorAliasMeta& alias = *alias_ptr;
    auto storage_it = storage_by_id.find(alias.storage_id);
    if (storage_it == storage_by_id.end()) {
      return absl::InvalidArgumentError(absl::StrCat("missing storage entry for alias storage_id=", alias.storage_id));
    }
    const RegisterStorageMeta* storage_meta = storage_it->second;
    std::optional<uint64_t> base_dst;
    if (auto dst_it = storage_to_dst.find(alias.storage_id); dst_it != storage_to_dst.end()) {
      base_dst = dst_it->second;
    }
    if (!base_dst.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat("missing segment entry for storage_id=", alias.storage_id));
    }
    if (alias.storage_offset + alias.logical_length > storage_meta->storage_length) {
      return absl::OutOfRangeError(
          absl::StrCat(
              "alias ",
              alias.name,
              " exceeds storage bounds (offset=",
              alias.storage_offset,
              ", length=",
              alias.logical_length,
              ", storage_length=",
              storage_meta->storage_length,
              ")"));
    }

    ordered_names.push_back(name);
    // Emit storage-level destination offset for every alias to keep canonical
    // index bytes stable across LIP and disk registrations.
    offsets.emplace(name, *base_dst);
    storage_sizes.emplace(name, storage_meta->storage_length);
    store::loader::CanonicalTensorMeta meta;
    meta.shape = alias.shape;
    meta.stride = alias.stride;
    meta.dtype = alias.dtype;
    meta.storage_offset = alias.storage_offset;
    metas.emplace(name, std::move(meta));
  }

  std::sort(ordered_names.begin(), ordered_names.end());
  return store::loader::build_canonical_index_json(ordered_names, offsets, storage_sizes, metas);
}

} // namespace tensorcast::daemon
