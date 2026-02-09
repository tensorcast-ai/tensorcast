// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <functional>
#include <string_view>

#include "absl/status/status.h"
#include "core/common/artifact_identity.h"
#include "core/store/store_engine.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/registration_manager.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class LocalStableTierService {
 public:
  struct Dep {
    store::StoreEngine& engine;
    LipManager& lip;
    IpcRegionRegistry& regions;
  };

  explicit LocalStableTierService(Dep d);

  absl::Status apply_local_stable_tier(
      const RegistrationManager::RegMeta& meta,
      std::string_view artifact_id,
      tensorcast::common::ArtifactIdKind id_kind,
      uint64_t total_size,
      v2::LocalStableTierResult& local_stable,
      const std::function<void()>& on_must_failure_cleanup) const;

 private:
  Dep d_;
};

} // namespace tensorcast::daemon
