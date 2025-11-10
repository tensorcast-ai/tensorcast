// Copyright (c) 2025, TensorCast Team.

#include <memory>
#include <string>
#include <vector>

#include "core/local/chunk/chunk.h"
#include "core/local/meta/view.h"

namespace tensorcast::local::meta {

class Artifact {
 public:
  explicit Artifact(const std::string& artifact_id) : artifact_id_(artifact_id) {}

  ~Artifact() = default;

  std::string get_artifact_id() const;

  View* get_view_by_id(const std::string& view_id) const;

  void add_view(std::unique_ptr<View> view);

  // TODO: from_view_id / from_file
  absl::Status create_vanilla_view(const std::string& view_id);

 private:
  std::string artifact_id_;
  std::vector<std::shared_ptr<Chunk>> chunks_;
  // chunks that are not bound to any view
  std::vector<std::shared_ptr<Chunk>> orphan_chunks_;
  // view id to view obj
  std::map<std::string, std::unique_ptr<View>> views_;
};
} // namespace tensorcast::local::meta