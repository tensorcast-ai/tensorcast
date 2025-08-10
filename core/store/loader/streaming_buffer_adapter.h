// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <memory>

#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/store/loader/buffer_pool.h"

namespace stepcast::store::loader {

// Adapter to make StreamingPinnedBuffer implement the BufferPool interface
class StreamingBufferAdapter : public BufferPool {
 public:
  explicit StreamingBufferAdapter(std::shared_ptr<StreamingPinnedBuffer> buffer);
  ~StreamingBufferAdapter() override = default;

  [[nodiscard]] size_t chunk_size() const override;

  [[nodiscard]] int capacity() const override;

  absl::StatusOr<int> get_free_chunk() override;

  void return_chunk(int slot_id) override;

  absl::Status mark_chunk_ready(int slot_id, uint64_t global_chunk_idx, size_t valid_bytes) override;

  absl::StatusOr<ReadyChunk> get_ready_chunk() override;

  void signal_production_complete() override;

  void* get_chunk_data_ptr(int slot_id) override;

  // Get the underlying buffer for direct access if needed
  StreamingPinnedBuffer* get_buffer() {
    return buffer_.get();
  }

 private:
  std::shared_ptr<StreamingPinnedBuffer> buffer_;
};

} // namespace stepcast::store::loader