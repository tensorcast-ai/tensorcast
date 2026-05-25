// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include "absl/status/statusor.h"
#include "core/store/device_types.h"
#include "core/store/replica/replica_config.h"

namespace tensorcast::store::replica {

// Resolves a ReplicaConfig into the canonical local DeviceKey used by core
// replica state. The device_type field is authoritative: GPU replicas must
// carry an explicit local_device_id, while CPU replicas must use -1.
absl::StatusOr<DeviceKey> resolve_replica_config_device_key(const ReplicaConfig& config);

// Canonicalizes a replica DeviceKey without selecting a default GPU. GPU keys
// may be completed from a registered UUID, but an ordinal-less GPU key remains
// invalid so upstream validation can report the contract violation.
DeviceKey normalize_replica_device_key(DeviceKey key);

} // namespace tensorcast::store::replica
