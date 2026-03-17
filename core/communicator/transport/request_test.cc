// Copyright (c) 2025-2026, TensorCast Team.

#include <atomic>
#include <mutex>
#include <thread>
#include <tuple>

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"

#include "core/communicator/base/constants.h"
#include "core/communicator/transport/request.h"

using tensorcast::communicator::transport::PartitionTensor;
using tensorcast::communicator::transport::ReadRequest;

TEST_CASE("ReadRequest emits window ACKs as segments complete", "[request]") {
  auto local = std::make_shared<PartitionTensor>(
      "key",
      /*addr*/ 0,
      /*bytes*/ 4096,
      tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
      nullptr);

  ReadRequest req("key", "127.0.0.1", 12345, local, /*remote_offset*/ 0, /*request_id=*/1);

  std::vector<std::tuple<uint32_t, std::vector<uint64_t>, bool>> acks;
  req.set_ack_sender([&acks](uint32_t window_seq, const std::vector<uint64_t>& offsets, bool final_window) {
    acks.emplace_back(window_seq, offsets, final_window);
  });

  req.enqueue_window_ack(0, {0, 64}, /*final_window=*/false);
  req.enqueue_window_ack(1, {128}, /*final_window=*/true);
  req.note_rdma_window(2, /*final_window=*/false);
  req.note_rdma_window(1, /*final_window=*/true);

  REQUIRE_FALSE(req.mark_completion_and_is_done());
  REQUIRE(acks.empty());

  REQUIRE_FALSE(req.mark_completion_and_is_done());
  REQUIRE(acks.size() == 1);
  REQUIRE(std::get<0>(acks[0]) == 0);
  REQUIRE(std::get<1>(acks[0]) == std::vector<uint64_t>({0, 64}));
  REQUIRE(std::get<2>(acks[0]) == false);

  REQUIRE(req.mark_completion_and_is_done());
  REQUIRE(acks.size() == 2);
  REQUIRE(std::get<0>(acks[1]) == 1);
  REQUIRE(std::get<1>(acks[1]) == std::vector<uint64_t>({128}));
  REQUIRE(std::get<2>(acks[1]) == true);
}

TEST_CASE("ReadRequest reports byte progress and completion status", "[request]") {
  auto local = std::make_shared<PartitionTensor>(
      "progress",
      /*addr*/ 0,
      /*bytes*/ 1024,
      tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
      nullptr);

  ReadRequest req("progress", "127.0.0.1", 12345, local, /*remote_offset*/ 0, /*request_id=*/2);

  uint64_t last_done = 0;
  uint64_t total = 0;
  int progress_events = 0;
  bool completed = false;

  req.set_progress_callbacks(
      [&](uint64_t done, uint64_t total_bytes) {
        last_done = done;
        total = total_bytes;
        ++progress_events;
      },
      [&](const absl::Status& status) { completed = status.ok(); });

  req.enqueue_completion_bytes(256);
  req.enqueue_completion_bytes(128);
  req.note_rdma_window(2, /*final_window=*/true);

  REQUIRE_FALSE(req.mark_completion_and_is_done());
  REQUIRE(req.mark_completion_and_is_done());
  REQUIRE(progress_events == 2);
  REQUIRE(last_done == 384);
  REQUIRE(total == 1024);

  req.set_result(absl::OkStatus());
  REQUIRE(completed);
}

TEST_CASE("ReadRequest completion path is concurrency-safe", "[request][concurrency]") {
  auto local = std::make_shared<PartitionTensor>(
      "concurrent",
      /*addr*/ 0,
      /*bytes*/ 2048,
      tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
      nullptr);

  ReadRequest req("concurrent", "127.0.0.1", 12345, local, /*remote_offset=*/0, /*request_id=*/3);

  std::mutex ack_mu;
  std::vector<std::tuple<uint32_t, std::vector<uint64_t>, bool>> acks;
  std::atomic<int> progress_events{0};
  std::atomic<int> completion_true_count{0};
  std::atomic<uint64_t> max_done{0};
  std::atomic<uint64_t> observed_total{0};

  req.set_ack_sender([&](uint32_t window_seq, const std::vector<uint64_t>& offsets, bool final_window) {
    std::lock_guard<std::mutex> lock(ack_mu);
    acks.emplace_back(window_seq, offsets, final_window);
  });
  req.set_progress_callbacks(
      [&](uint64_t done, uint64_t total_bytes) {
        observed_total.store(total_bytes, std::memory_order_relaxed);
        uint64_t current = max_done.load(std::memory_order_relaxed);
        while (done > current && !max_done.compare_exchange_weak(current, done, std::memory_order_relaxed)) {
        }
        progress_events.fetch_add(1);
      },
      [&](const absl::Status& status) {
        if (status.ok()) {
          completion_true_count.fetch_add(1);
        }
      });

  req.enqueue_window_ack(0, {0, 64, 128, 192}, /*final_window=*/false);
  req.enqueue_window_ack(1, {256, 320, 384, 448}, /*final_window=*/true);
  for (int i = 0; i < 8; ++i) {
    req.enqueue_completion_bytes(64);
  }
  req.note_rdma_window(4, /*final_window=*/false);
  req.note_rdma_window(4, /*final_window=*/true);

  std::atomic<int> done_true_count{0};
  std::vector<std::thread> workers;
  workers.reserve(8);
  for (int i = 0; i < 8; ++i) {
    workers.emplace_back([&]() {
      if (req.mark_completion_and_is_done()) {
        done_true_count.fetch_add(1);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }

  REQUIRE(done_true_count.load() == 1);
  REQUIRE(progress_events.load() == 8);
  REQUIRE(max_done.load(std::memory_order_relaxed) == 512);
  REQUIRE(observed_total.load(std::memory_order_relaxed) == 2048);

  {
    std::lock_guard<std::mutex> lock(ack_mu);
    REQUIRE(acks.size() == 2);
    REQUIRE(std::get<0>(acks[0]) == 0);
    REQUIRE(std::get<1>(acks[0]) == std::vector<uint64_t>({0, 64, 128, 192}));
    REQUIRE_FALSE(std::get<2>(acks[0]));
    REQUIRE(std::get<0>(acks[1]) == 1);
    REQUIRE(std::get<1>(acks[1]) == std::vector<uint64_t>({256, 320, 384, 448}));
    REQUIRE(std::get<2>(acks[1]));
  }

  auto future = req.get_future();

  std::vector<std::thread> setters;
  setters.reserve(8);
  for (int i = 0; i < 8; ++i) {
    setters.emplace_back([&]() { req.set_result(absl::OkStatus()); });
  }
  for (auto& setter : setters) {
    setter.join();
  }

  auto result = future.get();
  REQUIRE(result.status.ok());
  REQUIRE(completion_true_count.load() == 1);
}
