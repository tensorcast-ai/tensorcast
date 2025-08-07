// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/unified_memory_sink.h"

#include "absl/log/log.h"
#include "core/store/model/chunk_meta.h"

namespace stepcast::store::loader {

UnifiedMemorySink::UnifiedMemorySink(Options options) : options_(std::move(options)) {
  if (!options_.inner_sink) {
    LOG(ERROR) << "Inner sink is null";
  }
  if (!options_.memory_manager) {
    LOG(ERROR) << "Memory manager is null";
  }
}

absl::Status UnifiedMemorySink::write(const void* src, size_t bytes) {
  if (!options_.inner_sink) {
    return absl::InvalidArgumentError("Inner sink is null");
  }

  // Delegate actual write to inner sink
  auto status = options_.inner_sink->write(src, bytes);
  if (status.ok()) {
    bytes_written_ += bytes;
  }
  return status;
}

absl::Status UnifiedMemorySink::write_at(uint64_t offset, const void* src, size_t bytes) {
  if (!options_.inner_sink) {
    return absl::InvalidArgumentError("Inner sink is null");
  }
  // Try positioned sink path
  if (auto* ps = dynamic_cast<PositionedSink*>(options_.inner_sink.get())) {
    auto st = ps->write_at(offset, src, bytes);
    if (st.ok()) {
      bytes_written_ += bytes;
    }
    return st;
  }
  // Fallback: sequential write (less correct for out-of-order ranges)
  LOG(WARNING) << "UnifiedMemorySink: inner sink does not implement PositionedSink; falling back to sequential write";
  return write(src, bytes);
}

absl::Status UnifiedMemorySink::close() {
  if (!options_.inner_sink) {
    return absl::InvalidArgumentError("Inner sink is null");
  }

  // Close inner sink first
  auto close_status = options_.inner_sink->close();
  if (!close_status.ok()) {
    return close_status;
  }

  // Update chunk states if successful
  if (!states_updated_) {
    auto update_status = update_chunk_states();
    if (!update_status.ok()) {
      LOG(ERROR) << "Failed to update chunk states: " << update_status;
      return update_status;
    }
    states_updated_ = true;
  }

  VLOG(2) << "UnifiedMemorySink closed. Bytes written: " << bytes_written_;
  return absl::OkStatus();
}

absl::Status UnifiedMemorySink::update_chunk_states() {
  if (!options_.memory_manager) {
    return absl::InvalidArgumentError("Memory manager is null");
  }

  // Unified memory is an optional component that may not be initialised in some
  // execution paths (e.g.
  // unit tests that only exercise basic Disk→CPU loading).  In such cases we
  // can safely skip the chunk-state update because the higher-level logic does
  // not depend on it.  Treat the absence of UnifiedModelMemory as a no-op
  // rather than a hard error so that loaders remain usable without the full
  // unified-memory stack.

  auto unified_memory = options_.memory_manager->get_unified_memory();
  if (!unified_memory) {
    LOG(WARNING) << "UnifiedMemorySink: unified memory not initialised – skipping chunk state update";
    return absl::OkStatus();
  }

  // Determine new chunk state based on target location
  store::ChunkState new_state;
  if (options_.target_location == store::ModelLocation::GPU) {
    new_state = store::ChunkState::COPIED_GPU;
  } else {
    new_state = store::ChunkState::HOT;
  }

  // Update chunk states
  if (options_.chunk_indices.has_value()) {
    // Partial load - update specific chunks
    const auto& chunks = options_.chunk_indices.value();
    VLOG(2) << "Updating " << chunks.size() << " chunks to state " << static_cast<int>(new_state);

    return unified_memory->update_chunk_states(
        options_.memory_manager->instance_key(), options_.target_location, chunks, new_state, options_.device_id);
  }
  // Full load - update all chunks
  auto chunk_span = options_.memory_manager->chunk_snapshot();
  size_t num_chunks = chunk_span.size();
  std::vector<uint32_t> all_chunks;
  all_chunks.reserve(num_chunks);
  for (uint32_t i = 0; i < num_chunks; ++i) {
    all_chunks.push_back(i);
  }

  VLOG(2) << "Updating all " << num_chunks << " chunks to state " << static_cast<int>(new_state);

  return unified_memory->update_chunk_states(
      options_.memory_manager->instance_key(), options_.target_location, all_chunks, new_state, options_.device_id);
}

} // namespace stepcast::store::loader
