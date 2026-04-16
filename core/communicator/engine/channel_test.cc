// Copyright (c) 2025-2026, TensorCast Team.

#include "catch2/catch_test_macros.hpp"

#include <atomic>
#include <thread>

#include "core/communicator/base/constants.h"
#include "core/communicator/engine/channel.h"

namespace tensorcast::communicator::engine {

TEST_CASE("Channel caches RDMA endpoints by device pair", "[communicator][channel]") {
  Channel channel(
      /*control=*/nullptr,
      base::CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/16);

  auto first = channel.ensure_rdma_endpoint("mlx5_0", "mlx5_1");
  REQUIRE(first != nullptr);

  auto second = channel.ensure_rdma_endpoint("mlx5_0", "mlx5_1");
  REQUIRE(second != nullptr);
  CHECK(second == first);

  auto third = channel.ensure_rdma_endpoint("mlx5_2", "mlx5_3");
  REQUIRE(third != nullptr);
  CHECK(third != first);

  CHECK(channel.get_rdma_endpoint("mlx5_0", "mlx5_1") == first);
  CHECK(channel.get_rdma_endpoint("mlx5_2", "mlx5_3") == third);
}

TEST_CASE("Channel close is safe under concurrent slot reads", "[communicator][channel]") {
  Channel channel(
      /*control=*/nullptr,
      base::CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/16);
  channel.set_gpu_slot_handle(std::make_shared<int>(7));

  std::atomic<bool> stop{false};
  std::thread reader([&]() {
    while (!stop.load(std::memory_order_acquire)) {
      (void)channel.has_gpu_slot();
    }
  });

  channel.close();
  stop.store(true, std::memory_order_release);
  reader.join();

  CHECK_FALSE(channel.has_gpu_slot());
}

} // namespace tensorcast::communicator::engine
