// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "absl/status/status.h"

#include "core/local/chunk/chunk.h"

namespace tensorcast::local::meta {

class Artifact;

class View {
 public:
  View(
      std::string view_id,
      std::string view_spec_json,
      std::size_t view_size,
      std::string view_data_hash,
      Artifact* artifact)
      : view_id_(std::move(view_id)),
        view_spec_json_(std::move(view_spec_json)),
        view_size_(view_size),
        view_data_hash_(std::move(view_data_hash)),
        artifact_(artifact) {}

  ~View() = default;

  std::shared_ptr<Chunk> get_chunk_at(off_t offset) const;
  absl::Status bind_chunk_at(off_t offset, std::shared_ptr<Chunk> chunk, bool force_replace = false);
  absl::Status remove_chunk_at(off_t offset);
  std::string get_view_id() const;

 private:
  std::string view_id_;
  std::string view_spec_json_;
  std::size_t view_size_;
  std::string view_data_hash_;
  std::unordered_map<off_t, std::shared_ptr<Chunk>> chunks_map_;
  Artifact* artifact_;
};
} // namespace tensorcast::local::meta