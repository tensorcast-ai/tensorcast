// Copyright (c) 2025, TensorCast Team.

// LipManager: encapsulates LIP-related helpers for the daemon.
// Phase 1: provide LIP -> coalesced copy using RAII CUDA IPC mappings.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "core/store/store_engine.h"
#include "daemon/types.h"

namespace tensorcast::daemon {

class LipManager {
 public:
  explicit LipManager(std::shared_ptr<store::StoreEngine> engine) : engine_(std::move(engine)) {}

  // Copy from LIP segments on source GPU(s) into a freshly allocated
  // coalesced destination buffer on target_device_id. Returns CUDA IPC handle bytes.
  absl::StatusOr<std::vector<uint8_t>> copy_to_new_coalesced(
      int target_device_id,
      const std::string& canonical_index_json,
      uint64_t total_size,
      absl::Span<const LeaseSegMeta> segments,
      absl::Span<const RegisterStorageMeta> storages);

  // Register staged export for the given LIP lease over specified chunk indices.
  // Returns an opaque lock token which can be used to release the export via
  // release_staged_export(). This does not lock VS chunks; it registers GPU
  // ranges for transport and records CUDA IPC mappings for later cleanup.
  [[nodiscard]] absl::StatusOr<std::string> create_staged_export(
      const LipLeaseEntry& lip,
      absl::Span<const uint32_t> chunk_indices,
      store::StoreEngine& engine);

  // Release a staged export by lock token: unregister tensor keys and close
  // CUDA IPC mappings.
  [[nodiscard]] absl::Status release_staged_export(const std::string& token, store::StoreEngine& engine);

  // LIP registry operations (moved from service)
  void put_lease(const std::string& registration_id, const ArtifactDeviceKey& key, LipLeaseEntry entry);
  absl::Status keepalive_lease(const std::string& registration_id, int owner_pid, uint64_t epoch, uint32_t ttl_ms);
  absl::Status revoke_by_registration_id(const std::string& registration_id);
  std::optional<LipLeaseEntry> find_active_by_artifact_id(const std::string& artifact_id) const;
  bool has_active_on_device(const std::string& artifact_id, int device_id) const;
  void sweep_expired_and_dead_pids();

  // Remove the commit lease for (artifact, device) when the owner pid matches.
  // Best-effort: returns true on removal, false if not found or owner mismatch.
  bool revoke_commit_lease_if_owner_matches(const std::string& artifact_id, int device_id, int owner_pid);

  // Commit a LIP (lease in-place) registration into a persistent lease entry,
  // computing the artifact descriptor using index/data multihashes streamed
  // from the leased GPU segments. Stores the lease for keepalive/revoke and
  // returns descriptor fields and an optional verification JSON.
  [[nodiscard]] absl::StatusOr<CommitLeaseResult> commit_lease_in_place(
      const std::string& registration_id,
      int device_id,
      int owner_pid,
      uint32_t ttl_ms,
      uint64_t epoch,
      uint64_t total_size,
      tensorcast::common::ArtifactIdKind id_kind,
      const std::string& client_artifact_id,
      const std::string& index_data, // canonical index JSON (may be empty)
      const std::string& index_key_hex, // precomputed index sha256 hex (may be empty)
      std::vector<LeaseSegMeta>&& segments,
      std::vector<RegisterStorageMeta>&& storages,
      std::vector<RegisterTensorAliasMeta>&& aliases);

 private:
  std::shared_ptr<store::StoreEngine> engine_;

  absl::Mutex exp_mu_;
  absl::flat_hash_map<std::string, LipExportRecord> exports_ ABSL_GUARDED_BY(exp_mu_);

  // Internal LIP lease registry
  mutable absl::Mutex mu_;
  absl::flat_hash_map<ArtifactDeviceKey, LipLeaseEntry> leases_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, ArtifactDeviceKey> reg_to_key_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
