// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "absl/hash/hash.h"
#include "core/store/replica/chunk_meta.h"

namespace tensorcast::store {

struct ChunkLocation {
  std::string node_id; ///< Node UUID or hostname
  ChunkState state{ChunkState::COLD};
  uint32_t last_touch_s{0}; ///< Last heartbeat from that replica
  int device_id{-1}; ///< GPU device id, -1 means CPU
};

using ChunkKey = std::pair<std::string, uint32_t>; // <artifact_id, chunk_idx>

struct ChunkKeyHash {
  size_t operator()(const ChunkKey& k) const noexcept {
    return absl::Hash<ChunkKey>{}(k);
  }
};

// Directory that tracks all replicas of every chunk across the cluster.
using ChunkDirectory = std::unordered_map<ChunkKey, std::vector<ChunkLocation>, ChunkKeyHash>;

} // namespace tensorcast::store