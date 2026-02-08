// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/artifact_retire_utils.h"

#include <algorithm>
#include <filesystem>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "daemon/util/path_utils.h"

namespace tensorcast::daemon::artifact_retire {

namespace {

uint32_t remaining_timeout_ms(absl::Time deadline, uint32_t timeout_ms) {
  if (timeout_ms == 0) {
    return 0;
  }
  const absl::Duration remaining = deadline - absl::Now();
  if (remaining <= absl::ZeroDuration()) {
    return 0;
  }
  const int64_t remaining_ms = absl::ToInt64Milliseconds(remaining);
  const int64_t clamped_ms = std::min<int64_t>(remaining_ms, timeout_ms);
  return static_cast<uint32_t>(clamped_ms);
}

bool is_relative_path_under_prefix(
    const std::filesystem::path& relative_path,
    const std::filesystem::path& expected_prefix) {
  if (!std::filesystem::path{relative_path}.has_relative_path() || relative_path.is_absolute()) {
    return false;
  }
  auto rel_it = relative_path.begin();
  for (auto prefix_it = expected_prefix.begin(); prefix_it != expected_prefix.end(); ++prefix_it, ++rel_it) {
    if (rel_it == relative_path.end() || *rel_it != *prefix_it) {
      return false;
    }
  }
  return true;
}

} // namespace

absl::StatusOr<DrainOutcome> retire_replica_with_drain(
    LipManager* lip_manager,
    store::components::IGlobalStoreClient* global_store_client,
    std::string_view artifact_id,
    const ArtifactDeviceKey& key,
    bool wait_for_drain,
    uint32_t timeout_ms,
    std::optional<std::string_view> operation_id) {
  DrainOutcome outcome{.drained = !wait_for_drain, .replica_id = std::nullopt};
  const absl::Time deadline = absl::Now() + absl::Milliseconds(timeout_ms);

  outcome.replica_id = lip_manager->find_replica_id(key);
  if (outcome.replica_id.has_value() && !outcome.replica_id->empty()) {
    if (global_store_client == nullptr || !global_store_client->is_connected()) {
      return absl::FailedPreconditionError("Global Store client unavailable");
    }
    auto mark_or = global_store_client->mark_replica_unavailable(
        artifact_id,
        *outcome.replica_id,
        /*reason=*/"retire",
        operation_id);
    if (!mark_or.ok() && !absl::IsNotFound(mark_or.status())) {
      return mark_or.status();
    }
    if (wait_for_drain) {
      const uint32_t remaining_ms = remaining_timeout_ms(deadline, timeout_ms);
      auto drain_or = global_store_client->wait_replica_drain(*outcome.replica_id, remaining_ms, operation_id);
      if (!drain_or.ok() && !absl::IsNotFound(drain_or.status())) {
        return drain_or.status();
      }
      if (drain_or.ok() && !drain_or->drained) {
        return absl::DeadlineExceededError("drain timed out; artifact remains quiesced");
      }
    }
  }

  if (wait_for_drain) {
    const bool drained = lip_manager->wait_exports_drained(key, deadline);
    if (!drained) {
      return absl::DeadlineExceededError("drain timed out; artifact remains quiesced");
    }
    outcome.drained = true;
  }
  return outcome;
}

void append_deregister_message(v2::DeregisterArtifactResponse* resp, std::string_view msg) {
  if (resp == nullptr || msg.empty()) {
    return;
  }
  if (!resp->message().empty()) {
    resp->set_message(absl::StrCat(resp->message(), "; ", msg));
    return;
  }
  resp->set_message(std::string(msg));
}

void purge_managed_shared_disk_artifact(
    store::components::IGlobalStoreClient* global_store_client,
    std::string_view artifact_id,
    const std::filesystem::path& storage_path,
    v2::DeregisterArtifactResponse* resp) {
  if (global_store_client == nullptr || !global_store_client->is_connected()) {
    append_deregister_message(resp, "shared disk purge skipped: Global Store client unavailable");
    return;
  }

  auto cluster_or = global_store_client->get_cluster_id();
  if (!cluster_or.ok() || cluster_or->empty()) {
    append_deregister_message(resp, "shared disk purge skipped: cluster_id unavailable");
    return;
  }

  const std::string& cluster_id = *cluster_or;
  auto locations_or =
      global_store_client->list_artifact_disk_locations(std::string(artifact_id), /*include_deleted=*/true);
  if (!locations_or.ok()) {
    append_deregister_message(resp, "shared disk purge skipped: disk locations unavailable");
    return;
  }

  const std::filesystem::path expected_prefix = std::filesystem::path("clusters") / cluster_id / "objects";
  std::vector<std::string> managed_paths;
  managed_paths.reserve(locations_or->size());
  for (const auto& location : *locations_or) {
    if (location.cluster_id != cluster_id) {
      continue;
    }
    if (location.kind != tensorcast::global_store::v1::DISK_LOCATION_KIND_MANAGED) {
      continue;
    }
    if (location.relative_path.empty()) {
      continue;
    }
    const std::filesystem::path relative_path(location.relative_path);
    if (!is_relative_path_under_prefix(relative_path, expected_prefix)) {
      continue;
    }
    managed_paths.push_back(location.relative_path);
  }

  for (const auto& relative_path : managed_paths) {
    // Tombstone first so disk fallback stops advertising the path.
    absl::Status st = global_store_client->upsert_artifact_disk_location(
        std::string(artifact_id),
        cluster_id,
        relative_path,
        tensorcast::global_store::v1::DISK_LOCATION_KIND_MANAGED,
        /*is_deleted=*/true);
    if (!st.ok()) {
      append_deregister_message(resp, absl::StrCat("shared disk tombstone failed: ", st.message()));
    }
  }

  if (storage_path.empty()) {
    append_deregister_message(resp, "shared disk purge skipped: storage_path missing");
    return;
  }

  for (const auto& relative_path : managed_paths) {
    auto normalized_or = normalize_disk_path(relative_path, storage_path);
    if (!normalized_or.ok()) {
      append_deregister_message(
          resp, absl::StrCat("shared disk purge path rejected: ", normalized_or.status().message()));
      continue;
    }
    std::error_code error_code;
    std::filesystem::remove_all(*normalized_or, error_code);
    if (error_code) {
      append_deregister_message(resp, absl::StrCat("shared disk purge failed: ", error_code.message()));
    }
  }
}

} // namespace tensorcast::daemon::artifact_retire
