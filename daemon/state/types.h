// Copyright (c) 2025-2026, TensorCast Team.

// Shared daemon-local strong types and hashing helpers

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "daemon/state/ipc_region_registry.h"

#include "absl/container/flat_hash_map.h"
#include "core/common/artifact_identity.h"
#include "core/cuda/cuda_ipc.h"

namespace tensorcast::daemon {

struct ArtifactDeviceKey {
  std::string artifact_id;
  int device_id{0};
  bool operator==(const ArtifactDeviceKey&) const = default;
};

template <typename H>
H AbslHashValue(H h, const ArtifactDeviceKey& k) {
  return H::combine(std::move(h), k.artifact_id, k.device_id);
}

// Lease segment metadata used for LIP registration/commit
struct LeaseSegMeta {
  // Physical storage backing this segment (must match RegisterStorageMeta.storage_id).
  std::string storage_id;
  // Byte offset within the referenced storage window where this segment begins.
  uint64_t storage_offset{0};
  // Logical byte offset within the artifact where this segment begins.
  uint64_t artifact_offset{0};
  uint64_t length{0};
};

struct RegisterStorageMeta {
  std::string storage_id;
  int device_id{0};
  std::string handle_bytes;
  uint64_t storage_length{0};
  std::string region_id;
  // Byte offset from the mapped base pointer (allocation base for handle_bytes, region base for region_id)
  // to the start of this storage window.
  uint64_t mapping_base_offset{0};

  [[nodiscard]] bool has_region() const {
    return !region_id.empty();
  }

  [[nodiscard]] bool has_handle() const {
    return !handle_bytes.empty();
  }
};

struct RegisterTensorAliasMeta {
  std::string name;
  std::string storage_id;
  uint64_t storage_offset{0};
  uint64_t logical_length{0};
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::string dtype;
};

// LIP Registry entry (post-Commit leases)
struct LipLeaseEntry {
  std::string registration_id; // original registration id for keepalive/revoke
  std::string artifact_id;
  std::string client_artifact_id;
  tensorcast::common::ArtifactIdKind id_kind{tensorcast::common::ArtifactIdKind::kMi2};
  int device_id{0};
  int owner_pid{0};
  uint32_t ttl_ms{0};
  std::chrono::time_point<std::chrono::steady_clock> expiry;
  uint64_t epoch{0};
  uint64_t total_size{0};
  std::string index_data; // canonical JSON (for verification hashing if needed)
  std::vector<LeaseSegMeta> segments; // mapped via cuda IPC when used
  std::vector<RegisterStorageMeta> storages; // deduplicated storage entries
  std::vector<RegisterTensorAliasMeta> aliases; // logical tensor metadata
  std::string verification_json; // optional stored verification metadata (JSON)
};

// Temporary export record for LIP-based staged transport
struct LipExportRecord {
  std::string artifact_id;
  int device_id{0};
  std::vector<cuda::IpcMapping> opened_maps; // RAII CUDA IPC mappings held until unlock
  std::vector<std::string> tensor_keys; // registered keys to unregister
  // Region lease ownership for region-backed storages used by this export.
  // These references are acquired explicitly at staging time and must be
  // released when the staged export is released.
  IpcRegionRegistry* region_registry{nullptr};
  absl::flat_hash_map<std::string, uint32_t> held_region_refs; // region_id -> refcount
};

// Result of committing a LIP in-place registration: descriptor fields and
// optional verification JSON payload for quick client checks.
struct CommitLeaseResult {
  std::string artifact_id;
  std::string index_multihash;
  std::string data_multihash;
  std::string schema_version; // e.g., "v3"
  std::string encoding; // e.g., "json"
  uint64_t total_size{0};
  std::string verification_json; // optional JSON payload
  tensorcast::common::ArtifactIdKind id_kind{tensorcast::common::ArtifactIdKind::kMi2};
};

} // namespace tensorcast::daemon
