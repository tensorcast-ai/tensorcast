// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tensorcast::daemon {

// IpcRegionRegistry tracks CUDA IPC memory regions that can be reused across
// Lease-In-Place registrations. Regions are owned by a single process (PID) and
// have a TTL to allow automatic cleanup when clients crash or forget to
// unregister.
class IpcRegionRegistry {
 public:
  struct Options {
    size_t capacity = 4096;
    absl::Duration max_ttl = absl::Minutes(10);
  };

  struct RegisterParams {
    int device_id = -1;
    int owner_pid = -1;
    uint64_t size_bytes = 0;
    uint32_t ttl_ms = 0;
    std::string session_id;
    std::string region_name;
    std::string handle_bytes;
  };

  struct RegionDescriptor {
    std::string region_id;
    int device_id = 0;
    int owner_pid = 0;
    uint64_t size_bytes = 0;
    uint32_t ttl_ms = 0;
    std::string session_id;
    std::string region_name;
    absl::Time expires_at = absl::InfiniteFuture();
  };

  explicit IpcRegionRegistry(Options opts);

  // Register a new CUDA IPC region and return its descriptor.
  absl::StatusOr<RegionDescriptor> register_region(const RegisterParams& params);

  // Unregister a region. Returns true when the region was removed.
  absl::StatusOr<bool> unregister_region(const std::string& region_id, int owner_pid, bool force);

  // Retrieve metadata for a region. Returns NOT_FOUND when missing.
  absl::StatusOr<RegionDescriptor> describe(const std::string& region_id) const;

  // Refresh the TTL for a region. Returns false when region not found.
  bool refresh_ttl(const std::string& region_id, uint32_t ttl_ms);

  // Remove regions whose expiry is before |now| and return their descriptors.
  std::vector<RegionDescriptor> sweep_expired(absl::Time now);

  // Retrieve the raw CUDA handle bytes for a region.
  absl::StatusOr<std::string> get_handle_bytes(const std::string& region_id) const;

  // Acquire a region for active use, incrementing its reference count.
  absl::StatusOr<RegionDescriptor> acquire(const std::string& region_id, int owner_pid);

  // Release a previously acquired region.
  absl::Status release(const std::string& region_id);

 private:
  struct RegionRecord {
    RegionDescriptor desc;
    std::string handle_bytes;
    uint64_t refcount = 0;
    absl::Time inserted_at = absl::Now();
  };

  std::string mint_region_id_locked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const Options opts_;
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, RegionRecord> regions_ ABSL_GUARDED_BY(mu_);
  absl::BitGen bitgen_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
