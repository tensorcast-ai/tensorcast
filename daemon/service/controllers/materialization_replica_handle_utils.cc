// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_replica_handle_utils.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"

namespace tensorcast::daemon::materialization_replica_handle {

absl::Status register_session_and_refs(
    SessionsService& sessions,
    RefTracker& refs,
    const store::loading::ReplicaKey& replica_key,
    std::shared_ptr<tensorcast::common::ReadySignal<absl::Status>> ready_signal,
    const std::string& replica_uuid,
    int32_t pid,
    bool allow_pid_ref) {
  if (!replica_uuid.empty()) {
    auto st = sessions.put_with_verification(replica_uuid, replica_key, std::move(ready_signal));
    if (!st.ok()) {
      return st;
    }
  }
  if (!allow_pid_ref || pid <= 0) {
    return absl::OkStatus();
  }
  refs.add_ref(replica_key, pid);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<uint32_t>> build_export_chunks_for_replica(
    store::StoreEngine& engine,
    const store::loading::ReplicaKey& key,
    std::optional<uint64_t> size_bytes_override) {
  uint64_t size_bytes = 0;
  if (size_bytes_override.has_value()) {
    size_bytes = *size_bytes_override;
  } else {
    auto size_or = engine.get_replica_size(key);
    if (!size_or.ok()) {
      return size_or.status();
    }
    size_bytes = *size_or;
  }
  const uint64_t chunk_bytes = static_cast<uint64_t>(engine.get_artifact_chunk_bytes());
  if (chunk_bytes == 0) {
    return absl::FailedPreconditionError("artifact_chunk_bytes is zero");
  }
  const uint64_t num_chunks = (size_bytes + chunk_bytes - 1) / chunk_bytes;
  if (num_chunks == 0) {
    return absl::InvalidArgumentError("replica size is zero");
  }
  if (num_chunks > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return absl::InvalidArgumentError("replica has too many chunks");
  }
  std::vector<uint32_t> chunks;
  chunks.reserve(static_cast<size_t>(num_chunks));
  for (uint32_t i = 0; i < static_cast<uint32_t>(num_chunks); ++i) {
    chunks.push_back(i);
  }
  return chunks;
}

absl::Status attach_cuda_lease_for_replica_key(
    const store::loading::ReplicaKey& replica_key,
    int32_t effective_pid,
    HandleLeaseRegistry* handle_leases,
    SessionLifecycleManager* lifecycle,
    std::string_view lease_log_context,
    const std::function<void()>& on_lease_create_failed,
    v2::MemCopyHandle& out_mem_handle) {
  bool lease_created = false;
  if (handle_leases != nullptr && effective_pid > 0) {
    auto token_or = handle_leases->mint_cuda_ipc_lease(replica_key, effective_pid);
    if (token_or.ok()) {
      out_mem_handle.set_lease_token(*token_or);
      lease_created = true;
    } else {
      LOG(WARNING) << "mint_cuda_ipc_lease failed (" << lease_log_context << "): key=" << replica_key
                   << " pid=" << effective_pid << ": " << token_or.status();
      if (on_lease_create_failed) {
        on_lease_create_failed();
      }
    }
  }

  if (!lease_created && lifecycle != nullptr && effective_pid > 0) {
    auto lid_or = lifecycle->create_use_lease(replica_key, effective_pid);
    if (!lid_or.ok()) {
      LOG(WARNING) << "create_use_lease failed (" << lease_log_context << "): key=" << replica_key
                   << " pid=" << effective_pid << ": " << lid_or.status();
      if (on_lease_create_failed) {
        on_lease_create_failed();
      }
    }
  }

  return absl::OkStatus();
}

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
    v2::MemCopyHandle& out_mem_handle) {
  auto session_status = register_session_and_refs(
      sessions, refs, handle.replica_key, handle.ready_signal, std::string(replica_uuid), effective_pid, allow_pid_ref);
  if (!session_status.ok()) {
    return session_status;
  }
  auto rollback = absl::MakeCleanup([&]() {
    if (!replica_uuid.empty()) {
      (void)sessions.erase(std::string(replica_uuid));
    }
    if (allow_pid_ref && effective_pid > 0) {
      refs.drop_ref(handle.replica_key, effective_pid);
    }
    (void)engine.unload_replica(handle.replica_key);
  });

  if (planned_export_kind == "none") {
    std::move(rollback).Cancel();
    return absl::OkStatus();
  }

  if (planned_export_kind == "cpu_memfd_lease") {
    if (!handle.cpu_memfd_region.has_value()) {
      return absl::FailedPreconditionError("CPU memfd handle unavailable for replica");
    }
    if (handle_leases == nullptr) {
      return absl::FailedPreconditionError("local handle plane is disabled (no handle leases)");
    }

    const auto& region = *handle.cpu_memfd_region;
    auto chunks_or = build_export_chunks_for_replica(engine, handle.replica_key);
    if (!chunks_or.ok()) {
      chunks_or = build_export_chunks_for_replica(engine, handle.replica_key, region.size_bytes);
    }
    if (!chunks_or.ok()) {
      return chunks_or.status();
    }

    HandleLeaseRegistry::CpuMemfdDescriptor memfd_desc{
        .fd = region.fd,
        .size_bytes = region.size_bytes,
        .offset_bytes = region.offset_bytes,
    };
    auto token_or = handle_leases->mint_cpu_memfd_lease(handle.replica_key, effective_pid, memfd_desc, *chunks_or);
    if (!token_or.ok()) {
      return token_or.status();
    }

    auto* cpu = out_mem_handle.mutable_cpu_memfd();
    cpu->set_size_bytes(region.size_bytes);
    cpu->set_offset_bytes(region.offset_bytes);
    out_mem_handle.set_lease_token(*token_or);
    std::move(rollback).Cancel();
    return absl::OkStatus();
  }

  if (planned_export_kind != "cuda_ipc_lease") {
    return absl::InvalidArgumentError("unsupported replica handle export kind");
  }

  if (!handle.cuda_ipc_handle.is_valid()) {
    return absl::FailedPreconditionError("CUDA IPC handle unavailable for replica");
  }
  auto handle_view = handle.cuda_ipc_handle.as_string_view();
  out_mem_handle.set_cuda_ipc_handle(handle_view.data(), handle_view.size());

  auto lease_status = attach_cuda_lease_for_replica_key(
      handle.replica_key,
      effective_pid,
      handle_leases,
      lifecycle,
      lease_log_context,
      on_lease_create_failed,
      out_mem_handle);
  if (!lease_status.ok()) {
    return lease_status;
  }

  std::move(rollback).Cancel();
  return absl::OkStatus();
}

} // namespace tensorcast::daemon::materialization_replica_handle
