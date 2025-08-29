// Copyright (c) 2025, TensorCast Team.

/*
 *  DeviceRegistry – Centralised mapping between physical devices and DeviceKey.
 *
 *  Provides helper utilities to obtain a canonical DeviceKey for a given GPU
 *  ordinal as well as normalising user-supplied keys.  The initial version is
 *  intentionally lightweight – advanced features (dynamic hot-plug, remote
 *  discovery) can be layered on later without breaking existing callers.
 */
#pragma once

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"

#include "core/store/device_types.h"

namespace tensorcast::store {

class DeviceRegistry {
 public:
  // Returns the global singleton instance.  Thread-safe and lazy-initialised.
  static DeviceRegistry& instance();

  // Registers a mapping from GPU |ordinal| → |uuid|.  If the mapping already
  // exists, it is overwritten to reflect the most recent information.
  void register_gpu(int ordinal, std::string uuid);

  // Returns a canonical DeviceKey for the specified GPU |ordinal|.  If a UUID
  // has been registered for the ordinal, it is included.  Otherwise the UUID
  // field is left empty (legacy behaviour).
  [[nodiscard]] DeviceKey gpu_key(int ordinal) const;

  // Canonicalises an arbitrary DeviceKey.  For GPU keys the UUID/ordinal pair
  // is completed based on the registry state.  CPU/REMOTE keys are returned
  // unchanged.
  [[nodiscard]] DeviceKey normalize(const DeviceKey& key) const;

 private:
  DeviceRegistry() = default;

  // Bidirectional mappings so we can canonicalise by either field.
  absl::flat_hash_map<int, std::string> ordinal_to_uuid_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<std::string, int> uuid_to_ordinal_ ABSL_GUARDED_BY(mutex_);

  mutable absl::Mutex mutex_;
};

} // namespace tensorcast::store