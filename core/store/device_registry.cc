// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/device_registry.h"

namespace stepcast::store {

DeviceRegistry& DeviceRegistry::instance() {
  static auto* inst = new DeviceRegistry();
  return *inst;
}

void DeviceRegistry::register_gpu(int ordinal, std::string uuid) {
  absl::MutexLock lock(&mutex_);
  ordinal_to_uuid_[ordinal] = uuid;
  if (!uuid.empty()) {
    uuid_to_ordinal_[uuid] = ordinal;
  }
}

DeviceKey DeviceRegistry::gpu_key(int ordinal) const {
  absl::MutexLock lock(&mutex_);
  DeviceKey key{::stepcast::DeviceType::GPU, ordinal, ""};
  auto it = ordinal_to_uuid_.find(ordinal);
  if (it != ordinal_to_uuid_.end()) {
    key.uuid = it->second;
  }
  return key;
}

DeviceKey DeviceRegistry::normalize(const DeviceKey& in) const {
  if (in.type != ::stepcast::DeviceType::GPU) {
    return in; // CPU / REMOTE are already canonical.
  }

  absl::MutexLock lock(&mutex_);

  // Case 1: uuid is known, but ordinal might be missing or outdated.
  if (!in.uuid.empty()) {
    auto ord_it = uuid_to_ordinal_.find(in.uuid);
    if (ord_it != uuid_to_ordinal_.end()) {
      return DeviceKey{::stepcast::DeviceType::GPU, ord_it->second, in.uuid};
    }
    // Unknown uuid – fall back to provided fields.
    return in;
  }

  // Case 2: uuid empty – try to fill it.
  auto uuid_it = ordinal_to_uuid_.find(in.ordinal);
  if (uuid_it != ordinal_to_uuid_.end()) {
    return DeviceKey{::stepcast::DeviceType::GPU, in.ordinal, uuid_it->second};
  }

  // No mapping – return original key.
  return in;
}

} // namespace stepcast::store