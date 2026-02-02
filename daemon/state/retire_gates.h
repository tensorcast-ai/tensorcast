// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>

#include "core/store/materialization/contracts/loading_spec.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/transport_lock_manager.h"

namespace tensorcast::daemon {

struct RetireGateSnapshot {
  size_t ref_count{0};
  size_t use_count{0};
  size_t placement_pins{0};
  bool has_transport_lock{false};

  bool ready() const {
    return ref_count == 0 && use_count == 0 && placement_pins == 0 && !has_transport_lock;
  }
};

class RetireGates {
 public:
  RetireGates(RefTracker& refs, SessionLifecycleManager& lifecycle, TransportLockManager& locks)
      : refs_(refs), lifecycle_(lifecycle), locks_(locks) {}

  RetireGateSnapshot snapshot_for(const store::loading::ReplicaKey& key) const;

  size_t ref_count_for(const store::loading::ReplicaKey& key) const;
  size_t use_count_for(const store::loading::ReplicaKey& key) const;
  size_t placement_pin_count_for(const store::loading::ReplicaKey& key) const;
  bool has_transport_lock_for(const store::loading::ReplicaKey& key) const;

 private:
  RefTracker& refs_;
  SessionLifecycleManager& lifecycle_;
  TransportLockManager& locks_;
};

} // namespace tensorcast::daemon
