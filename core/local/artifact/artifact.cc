// Copyright (c) 2025, TensorCast Team.

#include "core/local/artifact/artifact.h"
#include <string>

namespace tensorcast::local::meta {

std::string Artifact::get_artifact_id() const {
  return artifact_id_;
}

View* Artifact::get_view_by_id(const std::string& view_id) const {
  auto it = views_.find(view_id);
  if (it == views_.end()) {
    return nullptr;
  }
  return it->second.get();
}
} // namespace tensorcast::local::meta