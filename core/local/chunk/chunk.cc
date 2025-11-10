// Copyright (c) 2025, TensorCast Team.

#include "core/local/chunk/chunk.h"
#include <cassert>

#include "core/local/chunk/data_chunk.h"
#include "core/local/meta/artifact.h"

namespace tensorcast::local::meta {

void Chunk::generate_data_chunks(const std::vector<store::DeviceKey>& device_keys) {
  for (const auto& key : device_keys) {
    std::unique_ptr<data::DataChunk> dchunk;
    if (key.type == DeviceType::CPU) {
      dchunk = std::make_unique<data::CPUDataChunk>(this, key);
    }
    // TODO: implement GPU data chunk
    // else if (key.type == DeviceType::GPU) {
    //   dchunk = std::make_unique<data::GPUDataChunk>(this, key);
    // }
    else {
      assert(false);
      throw std::invalid_argument("Invalid device type");
    }
    dev_data_chunks_.emplace(key, std::move(dchunk));
  }
}

data::DataChunk* Chunk::get_data_chunk(const store::DeviceKey& device_key) const {
  auto it = dev_data_chunks_.find(device_key);
  if (it == dev_data_chunks_.end()) {
    return nullptr;
  }
  return it->second.get();
}

std::string Chunk::to_string() const {
  return absl::StrFormat("Chunk(ptr: %p, size: %zu, artifact: %s)", this, size_, get_artifact()->get_artifact_id());
}

off_t Chunk::get_offset_in_view(const View* view) const {
  auto it = offset_in_view_.find(view->get_view_id());
  if (it == offset_in_view_.end()) {
    return -1;
  }
  return it->second;
}

std::vector<View*> Chunk::get_holder_views() const {
  std::vector<View*> holder_views;
  for (const auto& pair : offset_in_view_) {
    auto* view = artifact_->get_view_by_id(pair.first);
    if (view != nullptr) {
      holder_views.push_back(view);
    }
  }
  return holder_views;
}

} // namespace tensorcast::local::meta
