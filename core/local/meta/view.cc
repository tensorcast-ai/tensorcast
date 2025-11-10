// Copyright (c) 2025, TensorCast Team.

#include "core/local/meta/view.h"
#include <cassert>
#include <memory>
#include "absl/log/absl_check.h"
#include "absl/strings/str_format.h"
#include "core/local/meta/local_manager.h"

namespace tensorcast::local::meta {

View::View(
    std::string view_id,
    std::string view_spec_json,
    std::size_t view_size,
    std::string view_data_hash,
    Artifact* artifact,
    ViewType view_type,
    const std::vector<std::shared_ptr<Chunk>>& chunks)
    : view_id_(std::move(view_id)),
      view_spec_json_(std::move(view_spec_json)),
      view_size_(view_size),
      view_data_hash_(std::move(view_data_hash)),
      view_type_(view_type),
      artifact_(artifact) {
  int chunk_num = view_size_ / LocalManager::kLocalConfig.chunk_size;
  // assert(chunks.size() == chunk_num || (chunks.empty() && view_type_ == ViewType::Vanilla));
  if (!chunks.empty()) {
    assert(chunks.size() == chunk_num);
    for (int i = 0; i < chunk_num; i++) {
      auto s = bind_chunk_at(i * LocalManager::kLocalConfig.chunk_size, chunks[i]);
      assert(s.ok());
    }
  }
  // else if (view_type_ == ViewType::Vanilla) {
  //   for (off_t i = 0; i < view_size_; i += LocalManager::kLocalConfig.chunk_size) {
  //     auto chunk = std::make_shared<Chunk>(LocalManager::kLocalConfig.chunk_size, artifact_);
  //     auto s = bind_chunk_at(i, chunk);
  //     assert(s.ok());
  //   }
  // }
  // else {
  //   assert(false);
  // }
}

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
    assert((chunks_it != chunks_map_.end()) == (offset_it != chunk->offset_in_view_.end()));
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