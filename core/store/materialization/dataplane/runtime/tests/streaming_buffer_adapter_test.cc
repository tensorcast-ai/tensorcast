// Copyright (c) 2026, TensorCast Team.

#include <chrono>
#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "core/common/memory/pinned_buffer_pool.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/store/materialization/dataplane/runtime/streaming_buffer_adapter.h"

using tensorcast::common::memory::PinnedBufferPool;
using tensorcast::common::memory::StreamingPinnedBuffer;
using tensorcast::store::loader::StreamingBufferAdapter;

TEST_CASE("StreamingBufferAdapter returns producer-owned slots safely", "[streaming_buffer_adapter][regression]") {
  constexpr size_t kSliceBytes = 4096;
  auto pool = std::make_shared<PinnedBufferPool>(kSliceBytes, kSliceBytes, "adapter_test_pool");
  auto buffer = std::make_shared<StreamingPinnedBuffer>(1, kSliceBytes, pool);
  REQUIRE(buffer->initialize(std::chrono::milliseconds(100)).ok());

  StreamingBufferAdapter adapter(buffer);
  auto slot_or = adapter.get_free_chunk();
  REQUIRE(slot_or.ok());
  const int slot_id = *slot_or;

  // Producer-side cleanup paths call BufferPool::return_chunk; adapter must
  // recover producer-owned slots instead of leaking pinned slices.
  adapter.return_chunk(slot_id);

  auto recycled_or = buffer->try_get_free_chunk();
  REQUIRE(recycled_or.ok());
  REQUIRE(*recycled_or == slot_id);
  REQUIRE(buffer->abort_producer_slot(*recycled_or).ok());
  REQUIRE(buffer->release().ok());
}
