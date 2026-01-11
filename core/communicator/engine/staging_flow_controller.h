// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "core/communicator/engine/memory_stager.h"

struct ibv_mr;

namespace tensorcast::communicator::engine {

// Transport identifier used for metrics/logging.
enum class StageTransport {
  kUnknown = 0,
  kRdma,
  kMtcp,
};

// Channel-scoped ledger that accounts for staging credit (segments).
class FlowCreditLedger {
 public:
  class Lease {
   public:
    Lease() = default;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&& other) noexcept;
    ~Lease();

    [[nodiscard]] int granted_segments() const {
      return granted_;
    }

    // Marks |segments| from this lease as consumed. Consumed credit is expected
    // to be returned explicitly (e.g. when a StageLease releases). Any
    // unconsumed credit is automatically returned to the ledger when the lease
    // is destroyed.
    void mark_consumed(int segments = 1);

    // Explicitly release all unused credit back to the ledger. Safe to call
    // multiple times; subsequent calls are no-ops.
    void release_unused();

   private:
    friend class FlowCreditLedger;
    Lease(FlowCreditLedger* ledger, int granted_segments);

    void cleanup();

    FlowCreditLedger* ledger_ = nullptr;
    int granted_ = 0;
    int consumed_ = 0;
  };

  explicit FlowCreditLedger(int total_credit);

  FlowCreditLedger(const FlowCreditLedger&) = delete;
  FlowCreditLedger& operator=(const FlowCreditLedger&) = delete;

  // Attempts to acquire |requested_segments| from the ledger. The call blocks
  // until enough credit becomes available. Returns a Lease representing the
  // granted segments.
  absl::StatusOr<Lease> acquire(int requested_segments);

  // Attempts to acquire up to |requested_segments| immediately without
  // blocking. If insufficient credit is available the call returns an
  // Unavailable status. The returned Lease may contain fewer than the
  // requested segments to match the currently available credit.
  absl::StatusOr<Lease> try_acquire(int requested_segments);

  // Returns |segments| back to the ledger and wakes the next waiter if the
  // head of the queue can now be satisfied.
  void release(int segments);

  [[nodiscard]] int total_credit() const;
  [[nodiscard]] int outstanding_credit() const;
  [[nodiscard]] int available_credit() const;

 private:
  struct Waiter {
    int requested = 0;
    bool ready = false;
    absl::CondVar cv;
  };

  void wake_waiters_locked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const int total_credit_;

  mutable absl::Mutex mu_;
  int outstanding_ ABSL_GUARDED_BY(mu_) = 0;
  std::deque<std::shared_ptr<Waiter>> waiters_ ABSL_GUARDED_BY(mu_);
};

// StageLease is an RAII wrapper that releases staged buffers, deregisters MRs,
// and returns credit to the ledger when released. Copies share ownership of the
// underlying state, making it safe to hand leases across threads.
class StageLease {
 public:
  StageLease() = default;

  struct Metadata {
    StageTransport transport = StageTransport::kUnknown;
    std::string request_key;
    uint32_t window_seq = 0;
    uint32_t segment_idx = 0;
    uint64_t offset = 0;
    size_t bytes = 0;
    bool zero_copy = false;
  };

  StageLease(
      std::shared_ptr<MemoryStager> stager,
      FlowCreditLedger* ledger,
      void* exposed_ptr,
      size_t bytes,
      ibv_mr* mr,
      bool deregister_mr,
      Metadata metadata,
      std::function<void()> extra_release_hook = nullptr);

  StageLease(const StageLease&) = default;
  StageLease& operator=(const StageLease&) = default;
  StageLease(StageLease&&) noexcept = default;
  StageLease& operator=(StageLease&&) noexcept = default;

  [[nodiscard]] bool valid() const {
    return static_cast<bool>(state_);
  }

  [[nodiscard]] bool released() const;
  void release();

  [[nodiscard]] void* exposed_ptr() const;
  [[nodiscard]] size_t bytes() const;
  [[nodiscard]] ibv_mr* mr() const;
  [[nodiscard]] const Metadata& metadata() const;
  [[nodiscard]] absl::Time acquired_at() const;
  void set_metadata(Metadata metadata);

 private:
  struct State {
    State(
        std::shared_ptr<MemoryStager> stager,
        FlowCreditLedger* ledger,
        void* exposed_ptr,
        size_t bytes,
        ibv_mr* mr,
        bool deregister_mr,
        Metadata metadata,
        std::function<void()> extra_release_hook);

    void do_release();

    std::shared_ptr<MemoryStager> stager;
    FlowCreditLedger* ledger;
    void* exposed_ptr;
    size_t bytes;
    ibv_mr* mr;
    bool deregister_mr;
    Metadata metadata;
    std::function<void()> extra_release_hook;
    absl::Time acquired_at;

    mutable absl::Mutex mu;
    bool released ABSL_GUARDED_BY(mu) = false;
  };

  std::shared_ptr<State> state_;
};

struct StageLeaseKey {
  std::string request_key;
  uint32_t window_seq = 0;
  uint32_t segment_idx = 0;
};

struct StageLeaseKeyHash {
  size_t operator()(const StageLeaseKey& key) const;
};

inline bool operator==(const StageLeaseKey& lhs, const StageLeaseKey& rhs) {
  return lhs.window_seq == rhs.window_seq && lhs.segment_idx == rhs.segment_idx && lhs.request_key == rhs.request_key;
}

// Registry tracking active StageLeases, used by ACK handlers and TTL reapers.
class StageLeaseRegistry {
 public:
  StageLeaseRegistry() = default;

  void put(const StageLeaseKey& key, StageLease lease, absl::Time now = absl::Now());
  absl::StatusOr<StageLease> take(const StageLeaseKey& key);

  struct RegistryEntry {
    StageLeaseKey key;
    StageLease lease;
    absl::Time inserted_at;
  };

  std::vector<RegistryEntry> snapshot_expired(absl::Duration ttl) const;
  size_t size() const;

 private:
  struct Entry {
    StageLease lease;
    absl::Time inserted_at;
  };

  mutable absl::Mutex mu_;
  absl::flat_hash_map<StageLeaseKey, Entry, StageLeaseKeyHash> entries_ ABSL_GUARDED_BY(mu_);
};

// Request-scoped helper that stages data in windows bounded by available
// credit. The staging function is injected, making the helper independent from
// specific transport logic and simplifying testing.
class StagingWindow {
 public:
  struct Segment {
    StageLease lease;
    uint64_t offset = 0;
    uint32_t bytes = 0;
    uint32_t segment_idx = 0;
  };

  struct Window {
    uint32_t window_seq = 0;
    std::vector<Segment> segments;
    bool more_segments = false;
    int granted_credit = 0;
  };

  using StageFn = std::function<absl::StatusOr<StageLease>(uint64_t offset, uint32_t bytes, uint32_t segment_idx)>;

  StagingWindow(
      FlowCreditLedger& ledger,
      StageFn stage_fn,
      uint64_t total_bytes,
      uint64_t chunk_size,
      uint64_t initial_offset,
      uint32_t max_window_segments = 0);

  // Returns the next window worth of staged segments. When all bytes have been
  // staged the call returns OutOfRange status.
  absl::StatusOr<Window> stage_next();

  [[nodiscard]] bool done() const {
    return remaining_bytes_ == 0;
  }

  [[nodiscard]] uint32_t next_window_seq() const {
    return next_window_seq_;
  }

 private:
  FlowCreditLedger& ledger_;
  StageFn stage_fn_;
  const uint64_t chunk_size_;
  const uint32_t max_window_segments_;

  uint64_t remaining_bytes_;
  uint64_t current_offset_;
  uint32_t next_window_seq_ = 0;
};

} // namespace tensorcast::communicator::engine
