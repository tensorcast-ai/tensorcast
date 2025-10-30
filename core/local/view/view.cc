// Copyright (c) 2025, TensorCast Team.

#include "core/local/view/view.h"
#include <memory>
#include "absl/log/absl_check.h"
#include "absl/strings/str_format.h"

namespace tensorcast::local::meta {

std::shared_ptr<Chunk> View::get_chunk_at(off_t offset) const {
  auto it = chunks_map_.find(offset);
  if (it != chunks_map_.end()) {
    return it->second;
  }
  return nullptr;
}

absl::Status View::bind_chunk_at(off_t offset, std::shared_ptr<Chunk> chunk, bool force_replace) {
  if (!force_replace) {
    auto chunks_it = chunks_map_.find(offset);
    auto offset_it = chunk->offset_in_view_.find(view_id_);

    // Case 1: both have value, reject with error
    if (chunks_it != chunks_map_.end() && offset_it != chunk->offset_in_view_.end()) {
      return absl::FailedPreconditionError(
          absl::StrFormat(
              "Chunk already bound at offset %lld in view '%s'; must remove first",
              static_cast<long long>(offset),
              view_id_.c_str()));
    }

    // Case 2: inconsistent state - only one present, should not happen
    if ((chunks_it != chunks_map_.end()) != (offset_it != chunk->offset_in_view_.end())) {
      ABSL_CHECK(false) << "Inconsistent chunk/view mapping detected at offset " << offset << " for view '" << view_id_
                        << "'";
      return absl::FailedPreconditionError(
          absl::StrFormat(
              "Inconsistent mapping: chunks_map_ and offset_in_view_ out of sync at offset %lld view '%s'",
              static_cast<long long>(offset),
              view_id_.c_str()));
    }
  }

  // Case 3: neither present or force_replace, OK to insert
  chunks_map_.emplace(offset, chunk);
  chunk->offset_in_view_.emplace(view_id_, offset);
  return absl::OkStatus();
}

absl::Status View::remove_chunk_at(off_t offset) {
  auto it = chunks_map_.find(offset);
  if (it != chunks_map_.end()) {
    auto chunk = it->second;
    chunks_map_.erase(it);
    chunk->offset_in_view_.erase(view_id_);
    return absl::OkStatus();
  }
  return absl::NotFoundError(
      absl::StrFormat("Chunk not found at offset %lld in view '%s'", static_cast<long long>(offset), view_id_.c_str()));
}

std::string View::get_view_id() const {
  return view_id_;
}

} // namespace tensorcast::local::meta