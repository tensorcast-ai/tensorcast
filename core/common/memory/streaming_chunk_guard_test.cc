// Copyright (c) 2025, TensorCast Team.

#include <chrono>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"

#include "core/common/memory/pinned_buffer_pool.h"
#include "core/common/memory/streaming_chunk_guard.h"
#include "core/common/memory/streaming_pinned_buffer.h"

namespace tensorcast::common::memory {
namespace {

constexpr size_t kChunkSize = 512;
constexpr size_t kNumChunks = 8;

std::shared_ptr<StreamingPinnedBuffer> MakeBuffer(std::shared_ptr<PinnedBufferPool> pool) {
  auto buffer = std::make_shared<StreamingPinnedBuffer>(kNumChunks, kChunkSize, std::move(pool));
  REQUIRE(buffer->initialize(std::chrono::milliseconds::zero()).ok());
  return buffer;
}

TEST_CASE("StreamingChunkGuard returns producer slot on scope exit", "[streaming_chunk_guard]") {
  auto pool = std::make_shared<PinnedBufferPool>(kChunkSize * kNumChunks, kChunkSize);
  auto streaming = MakeBuffer(pool);

  {
    StreamingChunkGuard guard(streaming);
    auto host_ptr_or = guard.acquire();
    REQUIRE(host_ptr_or.ok());
    REQUIRE(*host_ptr_or != nullptr);
    // No promotion or release; guard destructor should abort and recycle.
  }

  StreamingChunkGuard guard(streaming);
  auto host_ptr_or = guard.acquire();
  REQUIRE(host_ptr_or.ok());
  REQUIRE(*host_ptr_or != nullptr);
  REQUIRE(guard.promote_to_consumer(/*global_chunk_id=*/0, /*bytes_in_chunk=*/kChunkSize).ok());
  const int slot_id = guard.release_for_async();
  REQUIRE(slot_id >= 0);
  REQUIRE(streaming->return_chunk(slot_id).ok());

  REQUIRE(streaming->release().ok());
}

TEST_CASE("StreamingChunkGuard TryAcquire reports availability", "[streaming_chunk_guard]") {
  auto pool = std::make_shared<PinnedBufferPool>(kChunkSize, kChunkSize);
  auto streaming = std::make_shared<StreamingPinnedBuffer>(/*num_chunks=*/1, kChunkSize, pool);
  REQUIRE(streaming->initialize(std::chrono::milliseconds::zero()).ok());

  {
    StreamingChunkGuard guard(streaming);
    REQUIRE(guard.acquire().ok());

    StreamingChunkGuard contender(streaming);
    auto try_result = contender.try_acquire();
    REQUIRE_FALSE(try_result.ok());
    REQUIRE(try_result.status().code() == absl::StatusCode::kUnavailable);
  }

  {
    StreamingChunkGuard guard(streaming);
    auto ptr_or = guard.try_acquire();
    REQUIRE(ptr_or.ok());
    REQUIRE(*ptr_or != nullptr);
  }

  REQUIRE(streaming->release().ok());
}

} // namespace
} // namespace tensorcast::common::memory
