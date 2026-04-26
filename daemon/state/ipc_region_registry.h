// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
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

// IpcRegionRegistry tracks local process-visible regions that can be reused
// across Lease-In-Place registrations and region-backed batch byte-artifact IO.
// Regions are owned by a single process (PID) and optionally have a TTL for
// automatic cleanup. When ttl_ms==0, the region does not time-expire and relies
// on explicit unregister + PID-exit cleanup.
class IpcRegionRegistry {
 public:
  enum class MemoryKind {
    kVram = 0,
    kHostShared = 1,
  };

  enum class HostRegionClass {
    kNone = 0,
    kScratch = 1,
    kAllocator = 2,
  };

  struct Options {
    size_t capacity = 4096;
    absl::Duration max_ttl = absl::Minutes(10);
  };

  struct RegisterParams {
    MemoryKind memory_kind = MemoryKind::kVram;
    int device_id = -1;
    int owner_pid = -1;
    uint64_t size_bytes = 0;
    uint32_t ttl_ms = 0;
    std::string session_id;
    std::string region_name;
    bool daemon_managed = false;
    HostRegionClass host_region_class = HostRegionClass::kNone;
    std::string attachment_bytes;
    // Legacy alias for VRAM CUDA IPC callers.
    std::string handle_bytes;
  };

  struct RegionDescriptor {
    std::string region_id;
    MemoryKind memory_kind = MemoryKind::kVram;
    int device_id = -1;
    int owner_pid = 0;
    uint64_t size_bytes = 0;
    uint32_t ttl_ms = 0;
    std::string session_id;
    std::string region_name;
    bool daemon_managed = false;
    HostRegionClass host_region_class = HostRegionClass::kNone;
    std::string attach_token;
    absl::Time expires_at = absl::InfiniteFuture();
    bool poisoned = false;
  };

  using RegionCleanupCallback = std::function<void(const RegionDescriptor&)>;

  struct HostSharedAttachment {
    RegionDescriptor region;
    std::string attach_token;
    int fd = -1;
    uint64_t offset_bytes = 0;
    uint64_t size_bytes = 0;
  };

  struct HostSharedLocalMapping {
    RegionDescriptor region;
    void* base_ptr = nullptr;
    uint64_t size_bytes = 0;
  };

  explicit IpcRegionRegistry(Options opts);

  void set_pre_cleanup_callback(RegionCleanupCallback callback);

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

  // Best-effort cleanup for when the owning process exits. Removes all regions
  // whose owner_pid matches |owner_pid| and returns their descriptors.
  std::vector<RegionDescriptor> handle_pid_exit(int owner_pid);

  // Retrieve the opaque attachment bytes for a region.
  absl::StatusOr<std::string> get_attachment_bytes(const std::string& region_id) const;

  // Retrieve the raw CUDA handle bytes for a VRAM region.
  absl::StatusOr<std::string> get_handle_bytes(const std::string& region_id) const;

  // Acquire a region for active use, incrementing its reference count.
  absl::StatusOr<RegionDescriptor> acquire(const std::string& region_id, int owner_pid);

  // Release a previously acquired region.
  absl::Status release(const std::string& region_id);

  // Mark a region as poisoned to prevent further writes.
  absl::Status mark_poisoned(const std::string& region_id);

  // Check if a region is poisoned.
  bool is_poisoned(const std::string& region_id) const;

  // Acquire a daemon-managed HOST_SHARED slab by opaque attach token.
  absl::StatusOr<HostSharedAttachment> acquire_host_shared_attachment(const std::string& attach_token, int owner_pid);

  // Release a previously acquired HOST_SHARED attachment token.
  absl::Status release_host_shared_attachment(const std::string& attach_token, int owner_pid);

  // Acquire a daemon-local mapping for a daemon-managed HOST_SHARED region.
  absl::StatusOr<HostSharedLocalMapping> acquire_host_shared_local_mapping(const std::string& region_id, int owner_pid);

 private:
  struct DaemonManagedHostSharedSlab {
    int fd = -1;
    void* base = nullptr;
    uint64_t size_bytes = 0;
  };

  struct RegionRecord {
    RegionDescriptor desc;
    std::string attachment_bytes;
    // Total live holds for this region across acquire()/release(),
    // host-shared attachment acquires, and daemon-local mappings.
    uint64_t refcount = 0;
    // Sub-count for daemon-managed HOST_SHARED attachment acquires served by
    // acquire_host_shared_attachment(). This lets release_host_shared_attachment()
    // reject stray or duplicate releases instead of consuming unrelated region
    // holds such as active local mappings.
    uint64_t host_shared_attachment_refcount = 0;
    absl::Time inserted_at = absl::Now();
    bool poisoned = false;
    bool retiring = false;
    std::unique_ptr<DaemonManagedHostSharedSlab> daemon_host_shared_slab;
  };

  std::string mint_region_id_locked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  std::string mint_attach_token_locked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void cleanup_region_record_locked(RegionRecord* rec) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const Options opts_;
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, RegionRecord> regions_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::string> attach_tokens_ ABSL_GUARDED_BY(mu_);
  RegionCleanupCallback pre_cleanup_callback_ ABSL_GUARDED_BY(mu_);
  absl::BitGen bitgen_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
