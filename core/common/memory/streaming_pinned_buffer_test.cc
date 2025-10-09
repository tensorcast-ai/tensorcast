// Copyright (c) 2025, TensorCast Team.

#include <chrono>

#include <catch2/catch_test_macros.hpp>

#include "core/common/memory/pinned_buffer_pool.h"
#include "core/common/memory/streaming_pinned_buffer.h"

namespace tensorcast::common::memory {

TEST_CASE("StreamingPinnedBuffer detects invalid slot reuse", "[streaming_pinned_buffer]") {
  constexpr size_t kChunkBytes = 4096;
  constexpr size_t kNumChunks = 4;
  auto pool = std::make_shared<PinnedBufferPool>(kChunkBytes * kNumChunks, kChunkBytes);
  StreamingPinnedBuffer buffer(kNumChunks, kChunkBytes, pool);
  REQUIRE(buffer.initialize(std::chrono::milliseconds(10)).ok());

  auto slot_or = buffer.get_free_chunk();
  REQUIRE(slot_or.ok());
  const int slot_id = *slot_or;
  REQUIRE(buffer.mark_chunk_ready(slot_id, /*global_chunk_id=*/1, kChunkBytes).ok());

  auto ready_or = buffer.get_ready_chunk();
  REQUIRE(ready_or.ok());
  REQUIRE(ready_or->slot_id == slot_id);

  // Attempting to mark the chunk ready again before returning must fail.
  auto invalid_ready = buffer.mark_chunk_ready(slot_id, /*global_chunk_id=*/2, kChunkBytes);
  REQUIRE_FALSE(invalid_ready.ok());

  // Returning the chunk transitions it back to free.
  REQUIRE(buffer.return_chunk(slot_id).ok());

  // Reacquire and reuse succeeds (slot ordering is implementation-defined).
  auto slot_two = buffer.get_free_chunk();
  REQUIRE(slot_two.ok());
  REQUIRE(buffer.mark_chunk_ready(*slot_two, /*global_chunk_id=*/3, kChunkBytes).ok());
  auto ready_two = buffer.get_ready_chunk();
  REQUIRE(ready_two.ok());
  REQUIRE(buffer.return_chunk(ready_two->slot_id).ok());

  REQUIRE(buffer.release().ok());
}

TEST_CASE("StreamingPinnedBuffer return_chunk enforces consumer ownership", "[streaming_pinned_buffer]") {
  constexpr size_t kChunkBytes = 2048;
  constexpr size_t kNumChunks = 2;
  auto pool = std::make_shared<PinnedBufferPool>(kChunkBytes * kNumChunks, kChunkBytes);
  StreamingPinnedBuffer buffer(kNumChunks, kChunkBytes, pool);
  REQUIRE(buffer.initialize(std::chrono::milliseconds(0)).ok());

  auto slot_or = buffer.get_free_chunk();
  REQUIRE(slot_or.ok());
  const int slot = *slot_or;

  // Returning without flowing through mark/get should fail.
  auto premature_return = buffer.return_chunk(slot);
  REQUIRE_FALSE(premature_return.ok());

  REQUIRE(buffer.mark_chunk_ready(slot, /*global_chunk_id=*/5, kChunkBytes).ok());
  auto ready_or = buffer.get_ready_chunk();
  REQUIRE(ready_or.ok());
  REQUIRE(buffer.return_chunk(ready_or->slot_id).ok());

  REQUIRE(buffer.release().ok());
}

TEST_CASE("StreamingPinnedBuffer promote and abort helpers work", "[streaming_pinned_buffer]") {
  constexpr size_t kChunkBytes = 1024;
  constexpr size_t kNumChunks = 2;
  auto pool = std::make_shared<PinnedBufferPool>(kChunkBytes * kNumChunks, kChunkBytes);
  StreamingPinnedBuffer buffer(kNumChunks, kChunkBytes, pool);
  REQUIRE(buffer.initialize(std::chrono::milliseconds(0)).ok());

  auto slot_or = buffer.get_free_chunk();
  REQUIRE(slot_or.ok());
  const int slot_id = *slot_or;
  REQUIRE(
      buffer.promote_producer_slot_to_consumer(slot_id, /*global_chunk_id=*/0, /*bytes_in_chunk=*/kChunkBytes).ok());
  REQUIRE(buffer.return_chunk(slot_id).ok());

  auto second_slot = buffer.get_free_chunk();
  REQUIRE(second_slot.ok());
  REQUIRE(buffer.abort_producer_slot(*second_slot).ok());

  auto final_slot = buffer.get_free_chunk();
  REQUIRE(final_slot.ok());
  REQUIRE(buffer.mark_chunk_ready(*final_slot, /*global_chunk_id=*/1, /*bytes_in_chunk=*/kChunkBytes / 2).ok());
  auto ready_chunk = buffer.get_ready_chunk();
  REQUIRE(ready_chunk.ok());
  REQUIRE(buffer.return_chunk(ready_chunk->slot_id).ok());

  REQUIRE(buffer.release().ok());
}

} // namespace tensorcast::common::memory
