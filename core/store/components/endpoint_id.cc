// Copyright (c) 2026, TensorCast Team.

#include "core/store/components/endpoint_id.h"

#include <algorithm>

#include "absl/strings/str_cat.h"

namespace tensorcast::store::components {

std::string derive_endpoint_id(
    std::string_view node_id,
    common::memory::MemoryLocation memory_type,
    int device_id) {
  if (node_id.empty()) {
    return {};
  }
  const std::string_view dev_type =
      memory_type == common::memory::MemoryLocation::GPU ? "gpu" : "cpu";
  const int endpoint_dev_id =
      memory_type == common::memory::MemoryLocation::GPU ? std::max(0, device_id) : 0;
  return absl::StrCat(node_id, "/dev/", dev_type, "/", endpoint_dev_id);
}

std::string derive_endpoint_id(const WorkerIdentity& local_identity, const DeviceKey& device) {
  const common::memory::MemoryLocation memory_type =
      device.type == DeviceType::GPU ? common::memory::MemoryLocation::GPU : common::memory::MemoryLocation::CPU;
  const int device_id = device.type == DeviceType::GPU ? device.ordinal : 0;
  return derive_endpoint_id(local_identity.node_id, memory_type, device_id);
}

} // namespace tensorcast::store::components
