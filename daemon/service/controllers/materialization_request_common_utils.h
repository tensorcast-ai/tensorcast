// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/components/global_store_client.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "daemon/state/artifact_source_registry.h"
#include "daemon/state/handle_lease_registry.h"
#include "daemon/state/lip_bridge.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/sessions_service.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::materialization_request_common {

struct LeaseContext {
  bool loopback_peer{false};
  bool no_lease{false};
  int32_t effective_pid{0};
};

absl::StatusOr<LeaseContext> validate_and_compute_lease_context(
    std::string_view peer,
    v2::LeaseMode lease_mode,
    bool wait_for_completion,
    int32_t request_pid,
    bool cpu_target,
    bool cpu_shared_memory_enabled,
    bool handle_leases_available);

struct ArtifactResolution {
  std::string resolved_artifact_id;
  std::optional<std::string> fallback_artifact_id;
  std::optional<std::string> bound_artifact_id;
  bool gs_connected{false};
  std::optional<std::filesystem::path> normalized_disk_path;
  std::optional<ArtifactSourceRegistry::Entry> local_import;
  std::optional<store::loading::DiskSource> disk_source;
};

absl::StatusOr<std::optional<std::string>> resolve_artifact_binding(
    const std::shared_ptr<store::components::IGlobalStoreClient>& client,
    std::string_view artifact_id);

std::optional<std::filesystem::path> resolve_managed_disk_path(
    store::components::IGlobalStoreClient* client,
    const std::filesystem::path& storage_root,
    std::string_view artifact_id,
    bool allow_disk);

absl::StatusOr<std::filesystem::path> wait_for_local_managed_disk_path(
    store::components::IGlobalStoreClient* client,
    const std::filesystem::path& storage_root,
    std::string_view artifact_id,
    std::chrono::milliseconds wait_budget,
    const grpc::ServerContext& ctx);

absl::StatusOr<ArtifactResolution> resolve_artifact_and_disk_source(
    const std::shared_ptr<store::components::IGlobalStoreClient>& global_store_client,
    ArtifactSourceRegistry* source_registry,
    const std::filesystem::path& storage_path,
    std::string artifact_id,
    bool allow_disk,
    bool allow_local_import_fallback,
    bool loopback_peer,
    std::optional<uint64_t> disk_expected_size = std::nullopt);

using MaterializeAttemptFn =
    std::function<absl::StatusOr<store::loading::ReplicaHandle>(const std::optional<store::loading::DiskSource>&)>;
using PrepareRetryDiskSourceFn =
    std::function<absl::StatusOr<std::optional<store::loading::DiskSource>>(const std::filesystem::path&)>;

absl::StatusOr<store::loading::ReplicaHandle> materialize_with_shared_disk_retry(
    const absl::Status& initial_status,
    store::components::IGlobalStoreClient* global_store_client,
    const std::filesystem::path& storage_path,
    std::string_view resolved_artifact_id,
    int wait_for_shared_disk_ms,
    bool allow_disk,
    const grpc::ServerContext& server_context,
    std::optional<std::filesystem::path>& normalized_disk_path,
    const MaterializeAttemptFn& materialize_retry_once,
    const PrepareRetryDiskSourceFn& prepare_retry_disk_source);

struct LipFastPathRequest {
  std::string artifact_id;
  int target_device_id{0};
  std::string replica_uuid;
  int32_t effective_pid{0};
  bool allow_pid_ref{false};
  std::string lease_log_context;
};

struct LipFastPathResult {
  bool satisfied{false};
  std::optional<store::loading::ReplicaKey> replica_key;
};

absl::StatusOr<LipFastPathResult> try_satisfy_lip_fast_path(
    LipBridge& lip,
    SessionsService& sessions,
    RefTracker& refs,
    HandleLeaseRegistry* handle_leases,
    SessionLifecycleManager* lifecycle,
    const LipFastPathRequest& request,
    v2::MemCopyHandle& out_handle,
    const std::function<void()>& on_lease_create_failed);

} // namespace tensorcast::daemon::materialization_request_common
