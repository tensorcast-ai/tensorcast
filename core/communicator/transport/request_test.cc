// Copyright (c) 2025-2026, TensorCast Team.

#include <atomic>
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

  ReadRequest req("key", "127.0.0.1", 12345, local, /*remote_offset*/ 0);

  std::vector<std::tuple<uint32_t, std::vector<uint64_t>, bool>> acks;
  req.set_ack_sender([&acks](uint32_t window_seq, const std::vector<uint64_t>& offsets, bool final_window) {
    acks.emplace_back(window_seq, offsets, final_window);
  });

  req.enqueue_window_ack(0, {0, 64}, /*final_window=*/false);
  req.enqueue_window_ack(1, {128}, /*final_window=*/true);
  req.add_expected_completions(2);
  req.add_expected_completions(1);

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

  ReadRequest req("progress", "127.0.0.1", 12345, local, /*remote_offset*/ 0);

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
  req.add_expected_completions(2);

  REQUIRE_FALSE(req.mark_completion_and_is_done());
  REQUIRE(req.mark_completion_and_is_done());
  REQUIRE(progress_events == 2);
  REQUIRE(last_done == 384);
  REQUIRE(total == 1024);

  req.set_result(absl::OkStatus());
  REQUIRE(completed);
}
