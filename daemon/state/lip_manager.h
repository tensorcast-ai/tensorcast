// Copyright (c) 2025-2026, TensorCast Team.

// LipManager: encapsulates LIP-related helpers for the daemon.
// Phase 1: provide LIP -> coalesced copy using RAII CUDA IPC mappings.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/store/store_engine.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/types.h"
#include "gsl/pointers"

namespace tensorcast::daemon {

class LipManager {
 public:
  struct BuildCommitLeaseOptions {
    std::optional<gsl::not_null<void*>> direct_gpu_hash_ptr;
    bool require_gpu_identity_hash;
  };

  struct RoutableLeaseResult {
    ArtifactDeviceKey key;
    std::vector<std::string> remote_memory_keys;
    std::vector<uint64_t> buffer_sizes;
  };

  LipManager(std::shared_ptr<store::StoreEngine> engine, IpcRegionRegistry* regions)
      : engine_(std::move(engine)), region_registry_(regions) {}

  void set_global_store_client(std::shared_ptr<store::components::IGlobalStoreClient> client) {
    global_store_client_ = std::move(client);
  }

  // Copy from LIP segments on source GPU(s) into a freshly allocated
  // coalesced destination buffer on target_device_id. Returns CUDA IPC handle bytes.
  absl::StatusOr<std::vector<uint8_t>> copy_to_new_coalesced(
      int target_device_id,
      const std::string& canonical_index_json,
      uint64_t total_size,
      absl::Span<const LeaseSegMeta> segments,
      absl::Span<const RegisterStorageMeta> storages,
      int owner_pid = 0);

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
  std::optional<LipLeaseEntry> find_active_by_artifact_id(
      std::string_view artifact_id,
      std::optional<std::string_view> view_id = std::nullopt) const;
  std::optional<LipLeaseEntry> find_active_by_key(const ArtifactDeviceKey& key) const;
  std::vector<LipLeaseEntry> list_active_by_artifact_id(
      std::string_view artifact_id,
      std::optional<std::string_view> view_id = std::nullopt) const;
  std::optional<ArtifactDeviceKey> find_key_by_registration_id(std::string_view registration_id) const;
  std::optional<std::string> find_replica_id(const ArtifactDeviceKey& key) const;
  bool has_active_on_device(
      std::string_view artifact_id,
      int device_id,
      std::optional<std::string_view> view_id = std::nullopt) const;
  void sweep_expired_and_dead_pids();

  // Remove the commit lease for (artifact, device) when the owner pid matches.
  // Best-effort: returns true on removal, false if not found or owner mismatch.
  bool revoke_commit_lease_if_owner_matches(
      std::string_view artifact_id,
      int device_id,
      int owner_pid,
      std::optional<std::string_view> view_id = std::nullopt);

  // Extend TTL for an active lease by artifact_id and optionally bump TTL on
  // any region-backed storages referenced by the lease. Returns OK if lease
  // found and updated, NOT_FOUND if no active lease, or another status.
  absl::Status extend_ttl_for_artifact(
      std::string_view artifact_id,
      uint32_t extend_ttl_ms,
      std::optional<std::string_view> view_id = std::nullopt);

  // Quiesce a LIP lease to block new staged exports.
  void quiesce_lease(const ArtifactDeviceKey& key);

  // Wait for staged exports to drain for a given lease until deadline.
  // Returns true if drained, false on timeout.
  bool wait_exports_drained(const ArtifactDeviceKey& key, absl::Time deadline);

  // Commit a LIP (lease in-place) registration into a persistent lease entry,
  // computing the artifact descriptor using index/data multihashes streamed
  // from the leased GPU segments. Stores the lease for keepalive/revoke and
  // returns descriptor fields and an optional verification JSON.
  [[nodiscard]] absl::StatusOr<CommitLeaseResult> build_commit_lease_result(
      int device_id,
      int owner_pid,
      uint64_t total_size,
      tensorcast::common::ArtifactIdKind id_kind,
      const std::string& client_artifact_id,
      const std::string& index_data,
      const std::string& index_key_hex,
      absl::Span<const LeaseSegMeta> segments,
      absl::Span<const RegisterStorageMeta> storages,
      absl::Span<const RegisterTensorAliasMeta> aliases,
      const std::optional<CommitLeaseResult>& identity_override = std::nullopt,
      BuildCommitLeaseOptions options = {});

  // Commit a LIP (lease in-place) registration into a persistent lease entry,
  // optionally reusing a pre-minted identity instead of hashing the same bytes
  // a second time.
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
      std::vector<RegisterTensorAliasMeta>&& aliases,
      const std::optional<CommitLeaseResult>& identity_override = std::nullopt,
      BuildCommitLeaseOptions options = {});

  // Commit a view-scoped routable LIP lease (used for piece registrations).
  // This registers long-lived communicator keys over the leased view ByteSpace
  // and stores lease state for keepalive/revoke/TTL.
  [[nodiscard]] absl::StatusOr<RoutableLeaseResult> commit_routable_view_lease_in_place(
      const std::string& registration_id,
      std::string_view artifact_id,
      std::string_view view_id,
      int device_id,
      int owner_pid,
      uint32_t ttl_ms,
      uint64_t epoch,
      uint64_t total_size,
      std::vector<LeaseSegMeta>&& segments,
      std::vector<RegisterStorageMeta>&& storages);

  // Export an already committed lease as a routable replica without creating a
  // second lease entry. Used by full-artifact binding-subject publication.
  [[nodiscard]] absl::StatusOr<RoutableLeaseResult> publish_committed_lease_routable(std::string_view registration_id);

  // Attach a Global Store replica_id to a committed LIP lease for best-effort cleanup.
  void attach_replica_id(const std::string& registration_id, std::string replica_id);

 private:
  std::shared_ptr<store::StoreEngine> engine_;
  IpcRegionRegistry* region_registry_;
  std::shared_ptr<store::components::IGlobalStoreClient> global_store_client_;

  absl::Mutex exp_mu_;
  absl::flat_hash_map<std::string, LipExportRecord> exports_ ABSL_GUARDED_BY(exp_mu_);

  mutable absl::Mutex routable_mu_;
  absl::flat_hash_map<ArtifactDeviceKey, LipExportRecord> routable_exports_ ABSL_GUARDED_BY(routable_mu_);
  absl::flat_hash_map<ArtifactDeviceKey, std::string> routable_replica_ids_ ABSL_GUARDED_BY(routable_mu_);

  // Internal LIP lease registry
  mutable absl::Mutex mu_;
  absl::flat_hash_map<ArtifactDeviceKey, LipLeaseEntry> leases_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, ArtifactDeviceKey> reg_to_key_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_set<ArtifactDeviceKey> quiesced_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
