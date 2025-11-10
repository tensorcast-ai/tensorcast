// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"

#include "core/local/chunk/chunk.h"

namespace tensorcast::local::meta {

class Artifact;
class Replica;

class View {
 public:
  // using ChunksMap = std::map<off_t, std::shared_ptr<Chunk>>;
  enum class ViewType : uint8_t { Vanilla, Slice, Transpose, Others = 255 };
  View(
      std::string view_id,
      std::string view_spec_json,
      std::size_t view_size,
      std::string view_data_hash,
      Artifact* artifact,
      ViewType view_type,
      const std::vector<std::shared_ptr<Chunk>>& chunks);

  View(
      std::string view_id,
      std::string view_spec_json,
      std::size_t view_size,
      std::string view_data_hash,
      // ViewType view_type,
      Artifact* artifact)
      : View(
            std::move(view_id),
            std::move(view_spec_json),
            view_size,
            std::move(view_data_hash),
            artifact,
            ViewType::Vanilla,
            std::vector<std::shared_ptr<Chunk>>{}) {};

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
  ViewType view_type_;
  std::map<off_t, std::shared_ptr<Chunk>> chunks_map_;
  Artifact* artifact_;

  friend class Replica;
};
} // namespace tensorcast::local::meta