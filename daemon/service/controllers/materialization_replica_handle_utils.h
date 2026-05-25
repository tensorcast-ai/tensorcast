// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/common/ready_signal.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/store_engine.h"
#include "daemon/state/handle_lease_registry.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/sessions_service.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::materialization_replica_handle {

absl::Status register_session_and_refs(
    SessionsService& sessions,
    RefTracker& refs,
    const store::loading::ReplicaKey& replica_key,
    std::shared_ptr<tensorcast::common::ReadySignal<absl::Status>> ready_signal,
    const std::string& replica_uuid,
    int32_t pid,
    bool allow_pid_ref);

absl::StatusOr<std::vector<uint32_t>> build_export_chunks_for_replica(
    store::StoreEngine& engine,
    const store::loading::ReplicaKey& key,
    std::optional<uint64_t> size_bytes_override = std::nullopt);

absl::Status bind_replica_handle_for_response(
    store::StoreEngine& engine,
    SessionsService& sessions,
    RefTracker& refs,
    SessionLifecycleManager* lifecycle,
    HandleLeaseRegistry* handle_leases,
    const store::loading::ReplicaHandle& handle,
    std::string_view replica_uuid,
    int32_t effective_pid,
    bool allow_pid_ref,
    std::string_view planned_export_kind,
    std::string_view lease_log_context,
    const std::function<void()>& on_lease_create_failed,
    v2::MemCopyHandle& out_mem_handle);

absl::Status attach_cuda_lease_for_replica_key(
    const store::loading::ReplicaKey& replica_key,
    int32_t effective_pid,
    HandleLeaseRegistry* handle_leases,
    SessionLifecycleManager* lifecycle,
    std::string_view lease_log_context,
    const std::function<void()>& on_lease_create_failed,
    v2::MemCopyHandle& out_mem_handle);

} // namespace tensorcast::daemon::materialization_replica_handle
