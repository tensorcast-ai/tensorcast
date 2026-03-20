// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "core/common/memory/cuda_memory.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"

namespace tensorcast::store::replica {

struct CollectiveDiskLoadRequest {
  loading::ReplicaKey replica_key;
  loading::CollectiveLoadGroupHint group;
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  std::string source_index_json;
  std::string view_index_json;
  std::optional<loading::VariantIdentity> variant_identity;
  void* gpu_ptr{nullptr};
  int device_id{-1};
  std::shared_ptr<common::memory::GpuDeviceMemory> gpu_allocation;
};

struct CollectiveDiskLoadResult {
  bool handled{false};
  absl::Status status{absl::OkStatus()};
};

struct CollectiveMappedTargetLoadOptions {
  uint64_t chunk_bytes{128ULL * 1024ULL * 1024ULL};
  uint64_t merge_max_gap_bytes{256ULL * 1024ULL};
  uint64_t merge_max_amplification{4};
};

struct CollectiveMappedTargetLoadRequest {
  std::string artifact_id;
  loading::CollectiveLoadGroupHint group;
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  loader::ByteRangeMap map;
  loading::IntoTargetLayout target_layout;
  int device_id{-1};
};

struct CollectiveMappedTargetLoadResult {
  bool handled{false};
  absl::Status status{absl::OkStatus()};
};

CollectiveDiskLoadResult try_collective_disk_load(
    const CollectiveDiskLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout);

CollectiveMappedTargetLoadResult try_collective_mapped_target_load(
    const CollectiveMappedTargetLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options);

absl::Status warm_collective_clique_cache(const std::vector<int>& device_ids);

} // namespace tensorcast::store::replica
