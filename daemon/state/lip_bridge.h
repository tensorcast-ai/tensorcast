// Copyright (c) 2025-2026, TensorCast Team.

// LipBridge: thin adapter around LipManager for cross-device LIP consumption

#pragma once

#include <functional>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "daemon/state/lip_manager.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class LipBridge {
 public:
  explicit LipBridge(LipManager& lip) : lip_(lip) {}

  // Try to satisfy a materialization request from an active LIP on another device.
  // Returns:
  //  - StatusOr<bool>: ok + true if satisfied from LIP, ok + false if no active LIP
  //  - error status if same-device LIP or copy fails
  absl::StatusOr<bool> try_satisfy_from_lip(
      absl::string_view artifact_id,
      int target_device_id,
      const std::function<void(const store::loading::ReplicaKey&)>& on_ready,
      v2::MemCopyHandle* out_handle);

  [[nodiscard]] bool has_active_on_device(
      absl::string_view artifact_id,
      int device_id,
      std::optional<absl::string_view> view_id = std::nullopt) const;

 private:
  LipManager& lip_;
};

} // namespace tensorcast::daemon
