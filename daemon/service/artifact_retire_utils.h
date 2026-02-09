// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "core/store/components/global_store_client.h"
#include "daemon/state/lip_manager.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::artifact_retire {

struct DrainOutcome {
  bool drained{true};
  std::optional<std::string> replica_id;
};

absl::StatusOr<DrainOutcome> retire_replica_with_drain(
    LipManager* lip_manager,
    store::components::IGlobalStoreClient* global_store_client,
    std::string_view artifact_id,
    const ArtifactDeviceKey& key,
    bool wait_for_drain,
    uint32_t timeout_ms,
    std::optional<std::string_view> operation_id);

void append_deregister_message(v2::DeregisterArtifactResponse* resp, std::string_view msg);

void purge_managed_shared_disk_artifact(
    store::components::IGlobalStoreClient* global_store_client,
    std::string_view artifact_id,
    const std::filesystem::path& storage_path,
    v2::DeregisterArtifactResponse* resp);

} // namespace tensorcast::daemon::artifact_retire
