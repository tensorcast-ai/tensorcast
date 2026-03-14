// Copyright (c) 2025-2026, TensorCast Team.

#include "core/common/memory/pinned_buffer_pool.h"

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/cuda/cuda_api.h"

namespace tensorcast::common::memory {

TEST_CASE("PinnedBufferPool deferred host registration blocks allocation until ready", "[pinned_buffer_pool][fake]") {
  if (!cuda::is_fake()) {
    return;
  }

  PinnedBufferPool::Options options{
      .name = "deferred_test_pool",
      .register_on_create = false,
  };
  PinnedBufferPool pool(2ULL * 1024 * 1024, 1ULL * 1024 * 1024, options);
  std::vector<char*> buffers;

  REQUIRE_FALSE(pool.is_host_registered());
  REQUIRE(pool.allocate(1ULL * 1024 * 1024, buffers) != 0);

  REQUIRE(pool.register_host_memory().ok());
  REQUIRE(pool.is_host_registered());
  REQUIRE(pool.allocate(1ULL * 1024 * 1024, buffers) == 0);
  REQUIRE(pool.deallocate(buffers) == 0);
}

} // namespace tensorcast::common::memory
