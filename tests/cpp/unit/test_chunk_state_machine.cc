// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <map>
#include <set>
#include <thread>
#include <vector>

#include "core/store/model/chunk_meta.h"

using namespace stepcast::store;

// Helper to test state transitions
class ChunkStateMachine {
 public:
  ChunkStateMachine() : meta_{} {
    meta_.state.store(ChunkState::COLD, std::memory_order_relaxed);
    meta_.last_touch_s.store(0, std::memory_order_relaxed);
  }

  bool try_transition(ChunkState from, ChunkState to) {
    ChunkState expected = from;
    return meta_.state.compare_exchange_strong(expected, to, std::memory_order_acq_rel);
  }

  ChunkState get_state() const {
    return meta_.state.load(std::memory_order_acquire);
  }

  void set_state(ChunkState state) {
    meta_.state.store(state, std::memory_order_release);
  }

  const ChunkMeta& get_meta() const {
    return meta_;
  }

 private:
  ChunkMeta meta_;
};

TEST_CASE("ChunkState basic properties", "[chunk_state]") {
  SECTION("ChunkState enum values") {
    // Verify distinct values
    std::set<int> values;
    values.insert(static_cast<int>(ChunkState::HOT));
    values.insert(static_cast<int>(ChunkState::LOCKED_TX));
    values.insert(static_cast<int>(ChunkState::COPIED_GPU));
    values.insert(static_cast<int>(ChunkState::COLD));
    values.insert(static_cast<int>(ChunkState::EVICTED));
    values.insert(static_cast<int>(ChunkState::PREEMPTIBLE));

    REQUIRE(values.size() == 6); // All values are distinct
  }

  SECTION("ChunkMeta initialization") {
    ChunkMeta meta{};
    REQUIRE(meta.state.load() == ChunkState::COLD); // Default initialization
    REQUIRE(meta.last_touch_s.load() == 0);
  }

  SECTION("ChunkMeta atomic operations") {
    ChunkMeta meta{};
    meta.state.store(ChunkState::COLD, std::memory_order_relaxed);
    REQUIRE(meta.state.load(std::memory_order_relaxed) == ChunkState::COLD);

    // Test compare_exchange
    ChunkState expected = ChunkState::COLD;
    bool success = meta.state.compare_exchange_strong(expected, ChunkState::HOT);
    REQUIRE(success);
    REQUIRE(meta.state.load() == ChunkState::HOT);

    // Failed compare_exchange
    expected = ChunkState::COLD; // Wrong expected value
    success = meta.state.compare_exchange_strong(expected, ChunkState::EVICTED);
    REQUIRE(!success);
    REQUIRE(expected == ChunkState::HOT); // Updated to actual value
    REQUIRE(meta.state.load() == ChunkState::HOT); // Unchanged
  }
}

TEST_CASE("ChunkState valid transitions", "[chunk_state]") {
  ChunkStateMachine sm;

  SECTION("COLD state transitions") {
    sm.set_state(ChunkState::COLD);

    // Valid transitions from COLD
    REQUIRE(sm.try_transition(ChunkState::COLD, ChunkState::LOCKED_TX));
    sm.set_state(ChunkState::COLD);
    REQUIRE(sm.try_transition(ChunkState::COLD, ChunkState::EVICTED));
    sm.set_state(ChunkState::COLD);
    REQUIRE(sm.try_transition(ChunkState::COLD, ChunkState::PREEMPTIBLE));
  }

  SECTION("HOT state transitions") {
    sm.set_state(ChunkState::HOT);

    // Valid transitions from HOT
    REQUIRE(sm.try_transition(ChunkState::HOT, ChunkState::LOCKED_TX));
    sm.set_state(ChunkState::HOT);
    REQUIRE(sm.try_transition(ChunkState::HOT, ChunkState::COLD));
    sm.set_state(ChunkState::HOT);
    REQUIRE(sm.try_transition(ChunkState::HOT, ChunkState::EVICTED));
    sm.set_state(ChunkState::HOT);
    REQUIRE(sm.try_transition(ChunkState::HOT, ChunkState::PREEMPTIBLE));
  }

  SECTION("LOCKED_TX state transitions") {
    sm.set_state(ChunkState::LOCKED_TX);

    // Valid transitions from LOCKED_TX
    REQUIRE(sm.try_transition(ChunkState::LOCKED_TX, ChunkState::HOT));
    sm.set_state(ChunkState::LOCKED_TX);
    REQUIRE(sm.try_transition(ChunkState::LOCKED_TX, ChunkState::COPIED_GPU));
  }

  SECTION("COPIED_GPU state transitions") {
    sm.set_state(ChunkState::COPIED_GPU);

    // Valid transitions from COPIED_GPU
    REQUIRE(sm.try_transition(ChunkState::COPIED_GPU, ChunkState::EVICTED));
    sm.set_state(ChunkState::COPIED_GPU);
    REQUIRE(sm.try_transition(ChunkState::COPIED_GPU, ChunkState::LOCKED_TX)); // For re-transfer
  }

  SECTION("EVICTED state transitions") {
    sm.set_state(ChunkState::EVICTED);

    // Valid transitions from EVICTED
    REQUIRE(sm.try_transition(ChunkState::EVICTED, ChunkState::HOT)); // Page fault recovery
  }

  SECTION("PREEMPTIBLE state transitions") {
    sm.set_state(ChunkState::PREEMPTIBLE);

    // Valid transitions from PREEMPTIBLE
    REQUIRE(sm.try_transition(ChunkState::PREEMPTIBLE, ChunkState::LOCKED_TX));
    sm.set_state(ChunkState::PREEMPTIBLE);
    REQUIRE(sm.try_transition(ChunkState::PREEMPTIBLE, ChunkState::EVICTED));
  }
}

TEST_CASE("ChunkState complete lifecycle", "[chunk_state]") {
  ChunkStateMachine sm;

  SECTION("Typical GPU loading lifecycle") {
    // Start with COLD (initial state)
    sm.set_state(ChunkState::COLD);
    REQUIRE(sm.get_state() == ChunkState::COLD);

    // Lock for transfer
    REQUIRE(sm.try_transition(ChunkState::COLD, ChunkState::LOCKED_TX));
    REQUIRE(sm.get_state() == ChunkState::LOCKED_TX);

    // Complete GPU transfer
    REQUIRE(sm.try_transition(ChunkState::LOCKED_TX, ChunkState::COPIED_GPU));
    REQUIRE(sm.get_state() == ChunkState::COPIED_GPU);

    // Evict after GPU copy
    REQUIRE(sm.try_transition(ChunkState::COPIED_GPU, ChunkState::EVICTED));
    REQUIRE(sm.get_state() == ChunkState::EVICTED);
  }

  SECTION("CPU-only lifecycle with preemption") {
    // Start with HOT (loaded to CPU)
    sm.set_state(ChunkState::HOT);
    REQUIRE(sm.get_state() == ChunkState::HOT);

    // Mark as preemptible
    REQUIRE(sm.try_transition(ChunkState::HOT, ChunkState::PREEMPTIBLE));
    REQUIRE(sm.get_state() == ChunkState::PREEMPTIBLE);

    // System evicts under memory pressure
    REQUIRE(sm.try_transition(ChunkState::PREEMPTIBLE, ChunkState::EVICTED));
    REQUIRE(sm.get_state() == ChunkState::EVICTED);

    // Page fault brings it back
    REQUIRE(sm.try_transition(ChunkState::EVICTED, ChunkState::HOT));
    REQUIRE(sm.get_state() == ChunkState::HOT);
  }

  SECTION("Failed transfer recovery") {
    // Start transfer
    sm.set_state(ChunkState::HOT);
    REQUIRE(sm.try_transition(ChunkState::HOT, ChunkState::LOCKED_TX));

    // Transfer fails, revert to COLD
    REQUIRE(sm.try_transition(ChunkState::LOCKED_TX, ChunkState::HOT));
    REQUIRE(sm.get_state() == ChunkState::HOT);
  }
}

TEST_CASE("ChunkState concurrent state transitions", "[chunk_state]") {
  SECTION("Concurrent lock attempts") {
    ChunkMeta meta{};
    meta.state.store(ChunkState::HOT, std::memory_order_release);

    const int num_threads = 8;
    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back([&]() {
        ChunkState expected = ChunkState::HOT;
        if (meta.state.compare_exchange_strong(expected, ChunkState::LOCKED_TX, std::memory_order_acq_rel)) {
          success_count++;
        }
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    // Exactly one thread should succeed
    REQUIRE(success_count == 1);
    REQUIRE(meta.state.load() == ChunkState::LOCKED_TX);
  }

  SECTION("Lock-unlock cycle stress test") {
    ChunkMeta meta{};
    meta.state.store(ChunkState::HOT, std::memory_order_release);

    const int num_threads = 4;
    const int iterations = 1000;
    std::atomic<int> lock_count{0};
    std::atomic<int> unlock_count{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back([&]() {
        for (int j = 0; j < iterations; ++j) {
          // Try to lock
          ChunkState expected = meta.state.load(std::memory_order_acquire);
          while (expected == ChunkState::HOT || expected == ChunkState::COLD) {
            if (meta.state.compare_exchange_weak(expected, ChunkState::LOCKED_TX, std::memory_order_acq_rel)) {
              lock_count++;

              // Simulate work
              std::this_thread::yield();

              // Unlock
              ChunkState locked = ChunkState::LOCKED_TX;
              bool unlocked = meta.state.compare_exchange_strong(locked, ChunkState::HOT, std::memory_order_acq_rel);
              REQUIRE(unlocked);
              unlock_count++;
              break;
            }
          }
        }
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    // All locks should be matched by unlocks
    REQUIRE(lock_count == unlock_count);
    REQUIRE(lock_count > 0);

    // Final state should not be LOCKED_TX
    auto final_state = meta.state.load();
    REQUIRE(final_state != ChunkState::LOCKED_TX);
  }
}

TEST_CASE("ChunkState edge cases", "[chunk_state]") {
  SECTION("Double lock protection") {
    ChunkStateMachine sm;
    sm.set_state(ChunkState::HOT);

    // First lock succeeds
    REQUIRE(sm.try_transition(ChunkState::HOT, ChunkState::LOCKED_TX));

    // Second lock fails
    REQUIRE(!sm.try_transition(ChunkState::HOT, ChunkState::LOCKED_TX));
    REQUIRE(sm.get_state() == ChunkState::LOCKED_TX);
  }

  SECTION("Invalid unlock") {
    ChunkStateMachine sm;
    sm.set_state(ChunkState::HOT);

    // Try to unlock non-locked chunk
    REQUIRE(!sm.try_transition(ChunkState::LOCKED_TX, ChunkState::HOT));
    REQUIRE(sm.get_state() == ChunkState::HOT);
  }

  SECTION("Evicted chunk operations") {
    ChunkStateMachine sm;
    sm.set_state(ChunkState::EVICTED);

    // Cannot lock evicted chunk directly
    REQUIRE(!sm.try_transition(ChunkState::EVICTED, ChunkState::LOCKED_TX));

    // Must recover to HOT first
    REQUIRE(sm.try_transition(ChunkState::EVICTED, ChunkState::HOT));
    REQUIRE(sm.try_transition(ChunkState::HOT, ChunkState::LOCKED_TX));
  }
}

TEST_CASE("ChunkMeta timestamp tracking", "[chunk_state]") {
  SECTION("Timestamp updates") {
    ChunkMeta meta{};
    REQUIRE(meta.last_touch_s.load() == 0);

    // Simulate touch
    uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    meta.last_touch_s.store(now, std::memory_order_relaxed);
    REQUIRE(meta.last_touch_s.load() == now);

    // Concurrent timestamp updates
    const int num_threads = 4;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back([&]() {
        for (int j = 0; j < 100; ++j) {
          uint64_t timestamp = static_cast<uint64_t>(std::time(nullptr));
          meta.last_touch_s.store(timestamp, std::memory_order_relaxed);
          std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    // Final timestamp should be recent
    uint64_t final_time = meta.last_touch_s.load();
    REQUIRE(final_time >= now);
    REQUIRE(final_time <= static_cast<uint64_t>(std::time(nullptr)));
  }
}

TEST_CASE("ChunkState transition validation matrix", "[chunk_state]") {
  // Define valid transitions based on the state machine design
  std::map<ChunkState, std::set<ChunkState>> valid_transitions = {
      {ChunkState::HOT, {ChunkState::LOCKED_TX, ChunkState::COLD, ChunkState::EVICTED, ChunkState::PREEMPTIBLE}},
      {ChunkState::LOCKED_TX, {ChunkState::HOT, ChunkState::COPIED_GPU}},
      {ChunkState::COPIED_GPU, {ChunkState::EVICTED, ChunkState::LOCKED_TX}},
      {ChunkState::COLD, {ChunkState::LOCKED_TX, ChunkState::EVICTED, ChunkState::PREEMPTIBLE}},
      {ChunkState::EVICTED, {ChunkState::HOT}},
      {ChunkState::PREEMPTIBLE, {ChunkState::LOCKED_TX, ChunkState::EVICTED}}};

  // Test all possible transitions
  std::vector<ChunkState> all_states = {
      ChunkState::HOT,
      ChunkState::LOCKED_TX,
      ChunkState::COPIED_GPU,
      ChunkState::COLD,
      ChunkState::EVICTED,
      ChunkState::PREEMPTIBLE};

  for (ChunkState from : all_states) {
    for (ChunkState to : all_states) {
      ChunkStateMachine sm;
      sm.set_state(from);

      bool should_succeed = valid_transitions[from].count(to) > 0;
      bool did_succeed = sm.try_transition(from, to);

      INFO("Testing transition from " << static_cast<int>(from) << " to " << static_cast<int>(to));
      if (from == to) {
        // Self-transition always succeeds with CAS
        REQUIRE(did_succeed);
      } else {
        REQUIRE(did_succeed == should_succeed);
      }
    }
  }
}