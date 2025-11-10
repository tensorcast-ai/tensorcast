// Copyright (c) 2025, TensorCast Team.

#include "core/local/meta/artifact.h"
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

void Artifact::add_view(std::unique_ptr<View> view) {
  if (view == nullptr) {
    return;
  }
  std::string view_id = view->get_view_id();
  views_.emplace(view_id, std::move(view));
}
} // namespace tensorcast::local::meta