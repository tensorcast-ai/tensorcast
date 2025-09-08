// Copyright (c) 2025, TensorCast Team.

// Shared daemon-local strong types and hashing helpers

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "daemon/cuda_ipc_raii.h"

#include "absl/container/flat_hash_map.h"

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
  int device_id{0};
  std::string handle_bytes; // raw cudaIpcMemHandle_t bytes
  uint64_t base_offset{0}; // offset within mapped handle
  uint64_t length{0};
  uint64_t dst_offset{0}; // destination offset in coalesced buffer
};

// LIP Registry entry (post-Commit leases)
struct LipLeaseEntry {
  std::string registration_id; // original registration id for keepalive/revoke
  std::string artifact_id;
  int device_id{0};
  int owner_pid{0};
  uint32_t ttl_ms{0};
  std::chrono::time_point<std::chrono::steady_clock> expiry;
  uint64_t epoch{0};
  uint64_t total_size{0};
  std::string index_data; // canonical JSON (for verification hashing if needed)
  std::vector<LeaseSegMeta> segments; // mapped via cuda IPC when used
  std::string verification_json; // optional stored verification metadata (JSON)
};

// Temporary export record for LIP-based staged transport
struct LipExportRecord {
  std::string artifact_id;
  int device_id{0};
  std::vector<CudaIpcMapping> opened_maps; // RAII CUDA IPC mappings held until unlock
  std::vector<std::string> tensor_keys; // registered keys to unregister
};

// Result of committing a LIP in-place registration: descriptor fields and
// optional verification JSON payload for quick client checks.
struct CommitLeaseResult {
  std::string artifact_id;
  std::string index_multihash;
  std::string data_multihash;
  std::string schema_version; // e.g., "v2"
  std::string encoding; // e.g., "json"
  uint64_t total_size{0};
  std::string verification_json; // optional JSON payload
};

} // namespace tensorcast::daemon
