// Copyright (c) 2025-2026, TensorCast Team.

// DeviceResolver: unify device selection from request parameters

#pragma once

#include <optional>

#include "absl/strings/string_view.h"
#include "core/store/device_registry.h"
#include "core/store/device_types.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class DeviceResolver {
 public:
  explicit DeviceResolver(store::DeviceRegistry& reg) : reg_(reg) {}

  store::DeviceKey DefaultGpu() const {
    return reg_.gpu_key(0);
  }

  // Resolve a device from type + uuid + optional ordinal hint (for GPU)
  store::DeviceKey From(
      v2::DeviceType type,
      absl::string_view device_uuid,
      std::optional<int> ordinal_hint = std::nullopt) const {
    using store::DeviceKey;
    if (!device_uuid.empty()) {
      DeviceKey key{.type = DeviceType::GPU, .ordinal = 0, .uuid = std::string(device_uuid)};
      return reg_.normalize(key);
    }
    if (type == v2::DeviceType::DEVICE_TYPE_CPU) {
      return DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
    }
    if (type == v2::DeviceType::DEVICE_TYPE_GPU && ordinal_hint.has_value() && *ordinal_hint >= 0) {
      return reg_.gpu_key(*ordinal_hint);
    }
    // Treat DISK as ingest-to-default GPU for legacy parity; default GPU otherwise.
    return DefaultGpu();
  }

 private:
  store::DeviceRegistry& reg_;
};

} // namespace tensorcast::daemon
