// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <algorithm>
#include <array>
#include <string>
#include "absl/hash/hash.h"
#include "core/common/device_types.h"
#include "core/common/memory/memory_location.h"

namespace stepcast::store {

// ══════════════════════════════════════════════════════════════════════════
// Unified Device Representation
// ══════════════════════════════════════════════════════════════════════════

/**
 * @brief A logical device identifier that stays stable across process restarts.
 *
 * The pair (ordinal, uuid) allows us to reference the same physical GPU even
 * when the ordinal changes (e.g. due to driver upgrades). For CPU we fix the
 * ordinal to -1 and leave uuid empty. For REMOTE we use the remote node id in
 * uuid and ordinal >= 0 for intra-node ranking.
 */
struct DeviceKey {
  DeviceType type{::stepcast::DeviceType::CPU};
  int32_t ordinal{-1};
  std::string uuid;

  // Equality is structural – all three fields must match.
  bool operator==(const DeviceKey&) const = default;

  // String representation for logging / debugging.
  [[nodiscard]] std::string to_string() const {
    const char* type_str = "REMOTE";
    if (type == ::stepcast::DeviceType::CPU) {
      type_str = "CPU";
    } else if (type == ::stepcast::DeviceType::GPU) {
      type_str = "GPU";
    }
    return std::string("DeviceKey{") + type_str + ":" + std::to_string(ordinal) + ":" + uuid + "}";
  }
};

// Hash functor for DeviceKey so it can be used as flat_hash_map key directly.
struct DeviceKeyHash {
  size_t operator()(const DeviceKey& k) const {
    return absl::HashOf(static_cast<int>(k.type), k.ordinal, k.uuid);
  }
};

/**
 * @brief Location - Triple describing the medium and device where data resides
 */
struct Location {
  MemoryLocation type = MemoryLocation::NONE;
  int32_t device_id = -1;
  std::string device_uuid;

  // Convert to DeviceKey for unified device handling
  [[nodiscard]] DeviceKey to_device_key() const {
    DeviceKey key;
    if (type == MemoryLocation::GPU) {
      key.type = ::stepcast::DeviceType::GPU;
      key.ordinal = device_id;
      key.uuid = device_uuid;
    } else if (type == MemoryLocation::PAGEABLE_CPU || type == MemoryLocation::DISK) {
      key.type = ::stepcast::DeviceType::CPU;
      key.ordinal = -1;
    }
    return key;
  }
};

/**
 * @brief CUDA IPC Handle wrapper for type safety
 */
struct CudaIpcHandle {
  static constexpr size_t kHandleSize = 64;
  std::array<char, kHandleSize> bytes{{}};

  // Convert to/from string for compatibility
  [[nodiscard]] std::string to_string() const {
    return std::string(bytes.data(), kHandleSize);
  }

  static CudaIpcHandle from_string(const std::string& str) {
    CudaIpcHandle handle;
    if (str.size() == kHandleSize) {
      std::copy(str.begin(), str.end(), handle.bytes.begin());
    }
    return handle;
  }

  [[nodiscard]] bool is_valid() const {
    return std::any_of(bytes.begin(), bytes.end(), [](char c) { return c != 0; });
  }
};

} // namespace stepcast::store