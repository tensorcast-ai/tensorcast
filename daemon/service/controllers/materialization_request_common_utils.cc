// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_request_common_utils.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "daemon/service/controllers/materialization_disk_resolve_utils.h"
#include "daemon/service/controllers/materialization_replica_handle_utils.h"
#include "daemon/util/deadline_utils.h"
#include "daemon/util/grpc_peer_utils.h"
#include "daemon/util/path_utils.h"
#include "opentelemetry/metrics/provider.h"

#include "core/common/artifact_identity.h"

namespace tensorcast::daemon::materialization_request_common {

namespace {

using materialization_disk_resolve::record_disk_import_outcome;
using materialization_disk_resolve::record_disk_path_denied;
using materialization_replica_handle::attach_cuda_lease_for_replica_key;
using materialization_replica_handle::register_session_and_refs;

void record_wait_for_shared_disk(std::string_view outcome, absl::Duration waited) {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_store_wait_for_shared_disk_total");
    static auto hist = meter->CreateDoubleHistogram("tc_store_wait_for_shared_disk_seconds");
    if (counter) {
      counter->Add(1, {{"outcome", std::string(outcome)}});
    }
    if (hist) {
      hist->Record(
          absl::ToDoubleSeconds(waited), {{"outcome", std::string(outcome)}}, opentelemetry::context::Context{});
    }
  } catch (...) {
  }
}

} // namespace

absl::StatusOr<std::optional<std::string>> resolve_artifact_binding(
    const std::shared_ptr<store::components::IGlobalStoreClient>& client,
    std::string_view artifact_id) {
  if (!client || !client->is_connected()) {
    return std::nullopt;
  }
  if (common::infer_artifact_id_kind(artifact_id) != common::ArtifactIdKind::kCgid) {
    return std::nullopt;
  }
  auto binding_or = client->get_artifact_binding(artifact_id);
  if (binding_or.ok()) {
    return binding_or->to_artifact_id;
  }
  if (absl::IsNotFound(binding_or.status())) {
    return std::nullopt;
  }
  return binding_or.status();
}

std::optional<std::filesystem::path> resolve_managed_disk_path(
    store::components::IGlobalStoreClient* client,
    const std::filesystem::path& storage_root,
    std::string_view artifact_id,
    bool allow_disk) {
  if (!allow_disk) {
    record_disk_import_outcome("disabled");
    return std::nullopt;
  }
  if (artifact_id.empty()) {
    record_disk_import_outcome("missing_artifact_id");
    return std::nullopt;
  }
  if (client == nullptr) {
    record_disk_import_outcome("no_client");
    return std::nullopt;
  }
  if (storage_root.empty()) {
    record_disk_import_outcome("no_storage_root");
    return std::nullopt;
  }
  auto cluster_or = client->get_cluster_id();
  if (!cluster_or.ok() || cluster_or->empty()) {
    record_disk_import_outcome("cluster_id_missing");
    return std::nullopt;
  }
  auto locations_or = client->list_artifact_disk_locations(artifact_id);
  if (!locations_or.ok()) {
    record_disk_import_outcome("not_found");
    return std::nullopt;
  }
  const std::string& cluster_id = *cluster_or;
  std::optional<store::components::ArtifactDiskLocation> selected;
  for (const auto& loc : *locations_or) {
    if (loc.cluster_id != cluster_id) {
      continue;
    }
    if (loc.kind == tensorcast::global_store::v1::DISK_LOCATION_KIND_MANAGED) {
      selected = loc;
      break;
    }
    if (!selected.has_value()) {
      selected = loc;
    }
  }
  if (!selected.has_value()) {
    record_disk_import_outcome("cluster_mismatch");
    return std::nullopt;
  }
  auto normalized_or = normalize_disk_path(selected->relative_path, storage_root);
  if (!normalized_or.ok()) {
    record_disk_path_denied();
    record_disk_import_outcome("invalid_path");
    LOG(WARNING) << "managed disk path rejected for artifact_id=" << artifact_id << ": " << normalized_or.status();
    return std::nullopt;
  }
  record_disk_import_outcome("ok");
  return *normalized_or;
}

absl::StatusOr<std::filesystem::path> wait_for_local_managed_disk_path(
    store::components::IGlobalStoreClient* client,
    const std::filesystem::path& storage_root,
    std::string_view artifact_id,
    std::chrono::milliseconds wait_budget,
    const grpc::ServerContext& ctx) {
  if (wait_budget.count() <= 0) {
    return absl::InvalidArgumentError("wait_budget must be > 0");
  }
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required");
  }
  if (client == nullptr || !client->is_connected()) {
    return absl::FailedPreconditionError("GlobalStoreClient not connected");
  }
  if (storage_root.empty()) {
    return absl::FailedPreconditionError("storage_path is required for shared-disk wait");
  }

  auto cluster_or = client->get_cluster_id();
  if (!cluster_or.ok() || cluster_or->empty()) {
    return absl::FailedPreconditionError(absl::StrCat("cluster_id unavailable: ", cluster_or.status().message()));
  }
  const std::string cluster_id = *cluster_or;

  const auto effective_budget = ClampToDeadline(ctx, wait_budget, wait_budget);
  if (effective_budget.count() <= 0) {
    record_wait_for_shared_disk("deadline_exceeded", absl::ZeroDuration());
    return absl::DeadlineExceededError("wait_for_shared_disk budget exhausted (RPC deadline)");
  }
  const absl::Time start = absl::Now();
  const absl::Time deadline = start + absl::Milliseconds(effective_budget.count());

  absl::Duration backoff = absl::Milliseconds(25);
  constexpr absl::Duration kMaxBackoff = absl::Seconds(1);
  absl::BitGen bitgen;

  while (absl::Now() < deadline) {
    if (ctx.IsCancelled()) {
      record_wait_for_shared_disk("cancelled", absl::Now() - start);
      return absl::CancelledError("RPC cancelled while waiting for shared-disk readiness");
    }

    auto locations_or = client->list_artifact_disk_locations(artifact_id);
    if (locations_or.ok()) {
      for (const auto& loc : *locations_or) {
        if (loc.cluster_id != cluster_id) {
          continue;
        }
        if (loc.kind != tensorcast::global_store::v1::DISK_LOCATION_KIND_MANAGED) {
          continue;
        }
        auto normalized_or = normalize_disk_path(loc.relative_path, storage_root);
        if (!normalized_or.ok()) {
          record_disk_path_denied();
          record_wait_for_shared_disk("invalid_path", absl::Now() - start);
          return normalized_or.status();
        }
        record_wait_for_shared_disk("ready", absl::Now() - start);
        return *normalized_or;
      }
    } else if (!absl::IsNotFound(locations_or.status())) {
      record_wait_for_shared_disk("gs_error", absl::Now() - start);
      return locations_or.status();
    }

    absl::Duration sleep_for = backoff;
    const double jitter = absl::Uniform<double>(bitgen, 0.5, 1.5);
    const auto sleep_ms =
        static_cast<int64_t>(std::max<double>(1.0, static_cast<double>(absl::ToInt64Milliseconds(sleep_for)) * jitter));
    sleep_for = absl::Milliseconds(sleep_ms);
    const absl::Duration remaining = deadline - absl::Now();
    if (remaining <= absl::ZeroDuration()) {
      break;
    }
    if (sleep_for > remaining) {
      sleep_for = remaining;
    }
    absl::SleepFor(sleep_for);
    backoff = std::min(backoff * 2, kMaxBackoff);
  }

  record_wait_for_shared_disk("deadline_exceeded", absl::Now() - start);
  return absl::DeadlineExceededError("managed shared-disk location not ready before deadline");
}

absl::StatusOr<LeaseContext> validate_and_compute_lease_context(
    std::string_view peer,
    v2::LeaseMode lease_mode,
    bool wait_for_completion,
    int32_t request_pid,
    bool cpu_target,
    bool cpu_shared_memory_enabled,
    bool handle_leases_available) {
  LeaseContext lease_context;
  lease_context.loopback_peer = is_loopback_grpc_peer(peer);
  lease_context.no_lease = lease_mode == v2::LeaseMode::LEASE_MODE_NO_LEASE;
  lease_context.effective_pid = (lease_context.loopback_peer && !lease_context.no_lease) ? request_pid : 0;

  if (wait_for_completion && !lease_context.loopback_peer) {
    return absl::PermissionDeniedError("wait_for_completion materialization is local-only (loopback/UDS)");
  }
  if (lease_context.no_lease && wait_for_completion) {
    return absl::InvalidArgumentError("lease_mode=NO_LEASE requires wait_for_completion=false");
  }
  if (!cpu_target) {
    return lease_context;
  }
  if (lease_context.no_lease) {
    // CPU prefetch / daemon-owned DRAM materialization intentionally avoids
    // issuing local handle leases. In that mode there is no client-owned CPU
    // handle to bind to a PID, so loopback-only CPU eligibility is sufficient.
    return lease_context;
  }
  if (!lease_context.loopback_peer) {
    return absl::PermissionDeniedError("CPU shared-memory materialization is local-only");
  }
  if (lease_context.effective_pid <= 0) {
    return absl::InvalidArgumentError("pid is required for CPU handle leases");
  }
  if (!cpu_shared_memory_enabled) {
    return absl::FailedPreconditionError("cpu_shared_memory is disabled");
  }
  if (!handle_leases_available) {
    return absl::FailedPreconditionError("local handle plane is disabled (no handle leases)");
  }
  return lease_context;
}

absl::StatusOr<ArtifactResolution> resolve_artifact_and_disk_source(
    const std::shared_ptr<store::components::IGlobalStoreClient>& global_store_client,
    ArtifactSourceRegistry* source_registry,
    const std::filesystem::path& storage_path,
    std::string artifact_id,
    bool allow_disk,
    bool allow_local_import_fallback,
    bool loopback_peer,
    std::optional<uint64_t> disk_expected_size) {
  ArtifactResolution resolution;
  resolution.resolved_artifact_id = std::move(artifact_id);

  auto binding_or = resolve_artifact_binding(global_store_client, resolution.resolved_artifact_id);
  if (!binding_or.ok()) {
    return binding_or.status();
  }
  if (binding_or->has_value()) {
    resolution.fallback_artifact_id = resolution.resolved_artifact_id;
    resolution.bound_artifact_id = binding_or->value();
    resolution.resolved_artifact_id = binding_or->value();
  }

  resolution.gs_connected = global_store_client && global_store_client->is_connected();
  resolution.normalized_disk_path =
      resolve_managed_disk_path(global_store_client.get(), storage_path, resolution.resolved_artifact_id, allow_disk);
  if (resolution.normalized_disk_path.has_value() && source_registry != nullptr) {
    source_registry->upsert_binding(
        resolution.resolved_artifact_id,
        ArtifactSourceRegistry::Entry{
            .source_kind = ArtifactSourceRegistry::SourceKind::kManagedSharedDisk,
            .canonical_source_path = resolution.normalized_disk_path->string(),
            .source_disk_path = resolution.normalized_disk_path->string(),
            .created_at = absl::Now(),
            .updated_at = absl::Now(),
        });
  }
  // Managed shared-disk remains preferred. If unavailable, local imports can still provide
  // a deterministic disk source even when Global Store is connected.
  if (!resolution.normalized_disk_path.has_value() && allow_disk && allow_local_import_fallback &&
      source_registry != nullptr) {
    auto entry = source_registry->lookup_binding(resolution.resolved_artifact_id);
    if (entry.has_value()) {
      if (!loopback_peer) {
        return absl::PermissionDeniedError("standalone disk materialization is local-only (loopback/UDS)");
      }
      if (entry->source_kind != ArtifactSourceRegistry::SourceKind::kLocalImport) {
        return absl::FailedPreconditionError(
            "SOURCE_MUTATED: unexpected source binding type for local import fallback");
      }
      materialization_disk_resolve::SourceFingerprintMap expected_fingerprints;
      expected_fingerprints.reserve(entry->file_fingerprints.size());
      for (const auto& [relative_path, fp] : entry->file_fingerprints) {
        expected_fingerprints.insert_or_assign(
            relative_path,
            materialization_disk_resolve::SourceFileFingerprint{
                .inode = fp.inode,
                .size = fp.size,
                .mtime_ns = fp.mtime_ns,
            });
      }
      auto fingerprint_status = materialization_disk_resolve::validate_source_fingerprints(
          std::filesystem::path(entry->canonical_source_path), expected_fingerprints);
      if (!fingerprint_status.ok()) {
        return fingerprint_status;
      }
      resolution.normalized_disk_path = std::filesystem::path(entry->canonical_source_path);
      resolution.local_import = std::move(entry);
    }
  }
  if (resolution.normalized_disk_path.has_value()) {
    resolution.disk_source = store::loading::DiskSource{
        .path = *resolution.normalized_disk_path,
        .expected_size = disk_expected_size,
        .require_descriptor = true,
    };
  }
  return resolution;
}

absl::StatusOr<store::loading::ReplicaHandle> materialize_with_shared_disk_retry(
    const absl::Status& initial_status,
    store::components::IGlobalStoreClient* global_store_client,
    const std::filesystem::path& storage_path,
    std::string_view resolved_artifact_id,
    const materialization_policy::NormalizedMaterializationRequestContext& request_context,
    const grpc::ServerContext& server_context,
    std::optional<std::filesystem::path>& normalized_disk_path,
    const MaterializeAttemptFn& materialize_retry_once,
    const PrepareRetryDiskSourceFn& prepare_retry_disk_source) {
  if (request_context.wait_for_shared_disk_ms <= 0 || !request_context.retrieval_policy.allow_disk) {
    return initial_status;
  }

  auto wait_or = wait_for_local_managed_disk_path(
      global_store_client,
      storage_path,
      resolved_artifact_id,
      std::chrono::milliseconds(request_context.wait_for_shared_disk_ms),
      server_context);
  if (!wait_or.ok()) {
    if (absl::IsUnavailable(wait_or.status())) {
      return initial_status;
    }
    return wait_or.status();
  }

  normalized_disk_path = std::move(*wait_or);
  auto retry_disk_source_or = prepare_retry_disk_source(*normalized_disk_path);
  if (!retry_disk_source_or.ok()) {
    return retry_disk_source_or.status();
  }
  return materialize_retry_once(*retry_disk_source_or);
}

absl::StatusOr<LipFastPathResult> try_satisfy_lip_fast_path(
    LipBridge& lip,
    SessionsService& sessions,
    RefTracker& refs,
    HandleLeaseRegistry* handle_leases,
    SessionLifecycleManager* lifecycle,
    const LipFastPathRequest& request,
    v2::MemCopyHandle& out_handle,
    const std::function<void()>& on_lease_create_failed) {
  LipFastPathResult outcome;
  absl::Status session_status = absl::OkStatus();
  auto satisfied = lip.try_satisfy_from_lip(
      request.artifact_id,
      request.target_device_id,
      [&](const store::loading::ReplicaKey& replica_key) {
        outcome.replica_key = replica_key;
        session_status = register_session_and_refs(
            sessions, refs, replica_key, nullptr, request.replica_uuid, request.effective_pid, request.allow_pid_ref);
      },
      &out_handle);
  if (!session_status.ok()) {
    return session_status;
  }
  if (!satisfied.ok()) {
    return satisfied.status();
  }
  if (!*satisfied) {
    return outcome;
  }
  outcome.satisfied = true;
  if (outcome.replica_key.has_value()) {
    auto lease_status = attach_cuda_lease_for_replica_key(
        *outcome.replica_key,
        request.effective_pid,
        handle_leases,
        lifecycle,
        request.lease_log_context,
        on_lease_create_failed,
        out_handle);
    if (!lease_status.ok()) {
      return lease_status;
    }
  }
  return outcome;
}

} // namespace tensorcast::daemon::materialization_request_common
