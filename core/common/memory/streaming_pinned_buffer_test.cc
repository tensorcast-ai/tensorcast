// Copyright (c) 2025-2026, TensorCast Team.

#include <atomic>
#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
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

// Producer/consumer decoupling: a slow consumer must never wedge a producer that
// keeps acquiring slots. With far more items than slots, the producer acquires
// via the non-blocking try_get_free_chunk (polling on Unavailable) while a slow
// consumer cycles slots, so the whole stream drains without deadlock. This is
// the StreamingPinnedBuffer property the MTCP recv path relies on: the network
// producer hands staged slots to a single H2D consumer and is never forced into
// a hard block that a stalled consumer cannot break.
TEST_CASE("StreamingPinnedBuffer slow consumer does not block producer", "[streaming_pinned_buffer]") {
  constexpr size_t kChunkBytes = 4096;
  constexpr size_t kNumChunks = 4;
  constexpr size_t kTotalItems = 64; // >> slots, so the producer must wait on the consumer
  auto pool = std::make_shared<PinnedBufferPool>(kChunkBytes * kNumChunks, kChunkBytes);
  StreamingPinnedBuffer buffer(kNumChunks, kChunkBytes, pool);
  REQUIRE(buffer.initialize(std::chrono::milliseconds(0)).ok());

  std::atomic<size_t> produced{0};
  std::atomic<size_t> consumed{0};
  std::atomic<bool> saw_backpressure{false};

  std::thread producer([&] {
    for (size_t i = 0; i < kTotalItems; ++i) {
      int slot = -1;
      // Non-blocking acquire with poll -- never a hard block the consumer can't break.
      while (true) {
        auto slot_or = buffer.try_get_free_chunk();
        if (slot_or.ok()) {
          slot = *slot_or;
          break;
        }
        // The only expected failure while draining is "no free slot right now".
        REQUIRE(absl::IsUnavailable(slot_or.status()));
        saw_backpressure.store(true, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
      REQUIRE(buffer.mark_chunk_ready(slot, /*global_chunk_id=*/i, kChunkBytes).ok());
      produced.fetch_add(1, std::memory_order_relaxed);
    }
    buffer.signal_production_complete();
  });

  std::thread consumer([&] {
    while (true) {
      auto ready_or = buffer.get_ready_chunk();
      if (!ready_or.ok()) {
        // OutOfRange once production is complete and the queue is drained.
        REQUIRE(absl::IsOutOfRange(ready_or.status()));
        break;
      }
      // Simulate a slow H2D copy so the producer is forced to wait on free slots.
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      REQUIRE(buffer.return_chunk(ready_or->slot_id).ok());
      consumed.fetch_add(1, std::memory_order_relaxed);
    }
  });

  producer.join();
  consumer.join();

  REQUIRE(produced.load() == kTotalItems);
  REQUIRE(consumed.load() == kTotalItems);
  // The test is only meaningful if the producer actually hit slot exhaustion and
  // recovered via the non-blocking path rather than deadlocking.
  REQUIRE(saw_backpressure.load());
  REQUIRE(buffer.release().ok());
}

} // namespace tensorcast::common::memory
