// Copyright (c) 2025-2026, TensorCast Team.
//
// Unit tests for MTcpRecvRequestState, the per-request completion accounting
// that replaced the serializing aggregate future.get() on the MTCP recv path.
//
// These pin down the "issuing token" invariant that decouples the network
// producer (recv_loop) from the H2D consumer: a request finalizes exactly once,
// only after every issued sub-chunk has reported *and* the producer has sealed,
// regardless of the order those events occur across threads. That is the
// recv-side deadlock fix expressed at the accounting layer -- the producer never
// has to block waiting for completions, so a slow/stalled consumer cannot wedge
// socket draining.

#include <atomic>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/communicator/transport/mtcp_transport.h"

namespace tensorcast::communicator::transport {
namespace {

// A request with no sub-chunks finalizes successfully once sealed (the issuing
// token alone brings the count to zero).
TEST_CASE("MTcpRecvRequestState seal with no chunks finalizes ok", "[mtcp_recv_state]") {
  MTcpRecvRequestState state(nullptr);
  REQUIRE_FALSE(state.finalized());
  state.seal();
  REQUIRE(state.finalized());
  REQUIRE_FALSE(state.result_failed());
}

// Completion-before-seal must not finalize early: while the producer is still
// issuing (not yet sealed), the request stays open even if its only chunk is
// already done.
TEST_CASE("MTcpRecvRequestState chunk completes before seal", "[mtcp_recv_state]") {
  MTcpRecvRequestState state(nullptr);
  state.add_chunk();
  state.on_chunk_done(misc::SUCCESS, 4096);
  REQUIRE_FALSE(state.finalized());
  state.seal();
  REQUIRE(state.finalized());
  REQUIRE_FALSE(state.result_failed());
}

// Seal-before-completion: the last outstanding chunk is the finalizer.
TEST_CASE("MTcpRecvRequestState seal before chunk completes", "[mtcp_recv_state]") {
  MTcpRecvRequestState state(nullptr);
  state.add_chunk();
  state.seal();
  REQUIRE_FALSE(state.finalized());
  state.on_chunk_done(misc::SUCCESS, 4096);
  REQUIRE(state.finalized());
  REQUIRE_FALSE(state.result_failed());
}

// Multiple chunks: not finalized until every chunk and the seal are accounted.
TEST_CASE("MTcpRecvRequestState multi-chunk finalizes only when all accounted", "[mtcp_recv_state]") {
  MTcpRecvRequestState state(nullptr);
  constexpr int kChunks = 4;
  for (int i = 0; i < kChunks; ++i) {
    state.add_chunk();
  }
  state.seal();
  for (int i = 0; i < kChunks - 1; ++i) {
    state.on_chunk_done(misc::SUCCESS, 1024);
    REQUIRE_FALSE(state.finalized());
  }
  state.on_chunk_done(misc::SUCCESS, 1024);
  REQUIRE(state.finalized());
  REQUIRE_FALSE(state.result_failed());
}

// A single failing sub-chunk marks the whole request failed.
TEST_CASE("MTcpRecvRequestState propagates chunk failure", "[mtcp_recv_state]") {
  MTcpRecvRequestState state(nullptr);
  state.add_chunk();
  state.add_chunk();
  state.on_chunk_done(misc::SUCCESS, 2048);
  state.on_chunk_done(misc::TRANSPORT_FAILED, 0);
  state.seal();
  REQUIRE(state.finalized());
  REQUIRE(state.result_failed());
}

// mark_failed flags the request independently of any chunk completion (mirrors a
// producer-side slot-acquire / invalid-lane failure during issue).
TEST_CASE("MTcpRecvRequestState mark_failed is honored", "[mtcp_recv_state]") {
  MTcpRecvRequestState state(nullptr);
  state.add_chunk();
  state.mark_failed();
  state.on_chunk_done(misc::SUCCESS, 2048);
  state.seal();
  REQUIRE(state.finalized());
  REQUIRE(state.result_failed());
}

// Concurrency: many chunks complete on independent threads while the producer
// seals concurrently. The request must always finalize and report the correct
// status. This is the property that lets the recv producer hand sub-chunks to
// lanes / the H2D consumer without ever blocking on their completion.
TEST_CASE("MTcpRecvRequestState concurrent completion finalizes once", "[mtcp_recv_state]") {
  constexpr int kIterations = 2000;
  constexpr int kChunks = 8;
  for (int iter = 0; iter < kIterations; ++iter) {
    const bool inject_failure = (iter % 3 == 0);
    const int failing_chunk = iter % kChunks;
    MTcpRecvRequestState state(nullptr);
    for (int i = 0; i < kChunks; ++i) {
      state.add_chunk();
    }

    std::vector<std::thread> threads;
    threads.reserve(kChunks + 1);
    std::atomic<bool> go{false};
    for (int i = 0; i < kChunks; ++i) {
      threads.emplace_back([&state, &go, i, inject_failure, failing_chunk] {
        while (!go.load(std::memory_order_acquire)) {
        }
        const misc::result_t st = (inject_failure && i == failing_chunk) ? misc::TRANSPORT_FAILED : misc::SUCCESS;
        state.on_chunk_done(st, st == misc::SUCCESS ? 1024 : 0);
      });
    }
    // Producer seals concurrently with the completions.
    threads.emplace_back([&state, &go] {
      while (!go.load(std::memory_order_acquire)) {
      }
      state.seal();
    });

    go.store(true, std::memory_order_release);
    for (auto& t : threads) {
      t.join();
    }

    REQUIRE(state.finalized());
    REQUIRE(state.result_failed() == inject_failure);
  }
}

} // namespace
} // namespace tensorcast::communicator::transport
