// Copyright (c) 2025, TensorCast Team.

#include "core/local/chunk/chunk.h"

#include "core/local/chunk/data_chunk.h"

namespace tensorcast::local::chunk {

void Chunk::generate_data_chunks(const std::vector<store::DeviceKey>& device_keys) {
  for (const auto& key : device_keys) {
    auto chunk = std::make_unique<CPUDataChunk>(this);
    dev_data_chunks_.emplace(key, std::move(chunk));
  }
}

DataChunk* Chunk::get_data_chunk(const store::DeviceKey& device_key) const {
  auto it = dev_data_chunks_.find(device_key);
  if (it == dev_data_chunks_.end()) {
    return nullptr;
  }
  return it->second.get();
}

} // namespace tensorcast::local::chunk
