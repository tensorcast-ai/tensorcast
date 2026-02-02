// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/retire_gates.h"

namespace tensorcast::daemon {

RetireGateSnapshot RetireGates::snapshot_for(const store::loading::ReplicaKey& key) const {
  RetireGateSnapshot snapshot;
  snapshot.ref_count = refs_.ref_count(key);
  snapshot.use_count = lifecycle_.use_count_for(key);
  snapshot.placement_pins = lifecycle_.placement_pin_count_for(key);
  snapshot.has_transport_lock = locks_.has_lock_for_key(key);
  return snapshot;
}

size_t RetireGates::ref_count_for(const store::loading::ReplicaKey& key) const {
  return refs_.ref_count(key);
}

size_t RetireGates::use_count_for(const store::loading::ReplicaKey& key) const {
  return lifecycle_.use_count_for(key);
}

size_t RetireGates::placement_pin_count_for(const store::loading::ReplicaKey& key) const {
  return lifecycle_.placement_pin_count_for(key);
}

bool RetireGates::has_transport_lock_for(const store::loading::ReplicaKey& key) const {
  return locks_.has_lock_for_key(key);
}

} // namespace tensorcast::daemon
