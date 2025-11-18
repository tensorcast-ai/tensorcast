// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "core/store/materialization/contracts/loading_spec.h"

namespace tensorcast::common::memory {
class PinnedBufferPool;
} // namespace tensorcast::common::memory

namespace tensorcast::store::components {

class DeviceManager;
class MetricsCollector;
class ReplicaRegistry;

absl::Status evict_for_gpu(
    ReplicaRegistry& registry,
    DeviceManager& device_manager,
    MetricsCollector& metrics,
    int device_id,
    size_t required_bytes);

absl::Status evict_for_cpu(
    ReplicaRegistry& registry,
    tensorcast::common::memory::PinnedBufferPool& memory_pool,
    MetricsCollector& metrics,
    size_t required_pinned_pool_bytes);

namespace detail {

absl::Status evict_core(
    absl::FunctionRef<std::vector<loading::ReplicaKey>()> get_candidates,
    absl::FunctionRef<absl::Status(const loading::ReplicaKey&)> try_release,
    absl::FunctionRef<void()> record_eviction,
    absl::FunctionRef<bool()> has_freed_enough,
    absl::FunctionRef<void(const loading::ReplicaKey&)> on_evicted = [](const loading::ReplicaKey&) {});

} // namespace detail

} // namespace tensorcast::store::components
