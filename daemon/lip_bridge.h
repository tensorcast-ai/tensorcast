// Copyright (c) 2025, TensorCast Team.

// LipBridge: thin adapter around LipManager for cross-device LIP consumption

#pragma once

#include <functional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "core/store/loading/loading_spec.h"
#include "daemon/lip_manager.h"
#include "tensorcast/daemon/v1/store_daemon.pb.h"

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
      v1::MemCopyHandle* out_handle);

 private:
  LipManager& lip_;
};

} // namespace tensorcast::daemon
