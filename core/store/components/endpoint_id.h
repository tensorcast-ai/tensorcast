// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <string>
#include <string_view>

#include "core/common/memory/memory_location.h"
#include "core/store/components/worker_identity.h"
#include "core/store/device_types.h"

namespace tensorcast::store::components {

// Canonical endpoint id format: "<node_id>/dev/<gpu|cpu>/<dev_id>".
std::string derive_endpoint_id(std::string_view node_id, common::memory::MemoryLocation memory_type, int device_id);

std::string derive_endpoint_id(const WorkerIdentity& local_identity, const DeviceKey& device);

} // namespace tensorcast::store::components
