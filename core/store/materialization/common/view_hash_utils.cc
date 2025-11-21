// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/common/view_hash_utils.h"

#include <limits>
#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/common/memory/cuda_memory.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/verification/verification_utils.h"
#include "core/store/materialization/dataplane/view/view_plan_source.h"
#include "core/store/materialization/dataplane/view/view_transform_executor.h"
#include "gsl/pointers"

namespace tensorcast::store {

namespace {

constexpr size_t kDefaultLeafChunkBytes = 4ULL * 1024 * 1024;

size_t NormalizeLeafBytes(size_t value) {
  return value == 0 ? kDefaultLeafChunkBytes : value;
}

} // namespace

ViewHashComputer::ViewHashComputer() : default_leaf_chunk_bytes_(kDefaultLeafChunkBytes) {}

ViewHashComputer::ViewHashComputer(ViewHashConfig config)
    : default_leaf_chunk_bytes_(NormalizeLeafBytes(config.default_leaf_chunk_bytes)) {}

std::optional<std::string> ViewHashComputer::hash_replica_view(
    replica::Replica& replica,
    common::memory::MemoryLocation location,
    uint64_t view_size_bytes,
    std::optional<int> gpu_device_id) const {
  if (view_size_bytes == 0) {
    return std::nullopt;
  }

  loader::verification::MemoryView mem_view;
  mem_view.location = location;
  mem_view.size_bytes = view_size_bytes;
  mem_view.gpu_device_id = gpu_device_id;
  std::shared_ptr<common::memory::GpuDeviceMemory> gpu_guard;

  if (location == common::memory::MemoryLocation::GPU) {
    if (!gpu_device_id.has_value()) {
      return std::nullopt;
    }
    auto view_or = replica.get_memory_manager().get_gpu_allocation_view();
    if (!view_or.ok() || view_or->base_ptr == nullptr) {
      VLOG(1) << "hash_replica_view: GPU allocation view unavailable";
      return std::nullopt;
    }
    gpu_guard = view_or->allocation;
    mem_view.base_ptr = view_or->base_ptr;
  } else {
    const auto cpu_ptrs = replica.get_memory_manager().get_pointer(common::memory::MemoryLocation::CPU);
    if (cpu_ptrs.empty() || cpu_ptrs[0] == nullptr) {
      VLOG(1) << "hash_replica_view: CPU memory unavailable for view hash";
      return std::nullopt;
    }
    mem_view.base_ptr = const_cast<void*>(cpu_ptrs[0]);
  }

  auto hash_or = loader::verification::compute_data_multihash(mem_view);
  if (!hash_or.ok()) {
    if (!absl::IsNotFound(hash_or.status())) {
      LOG(WARNING) << "compute_data_multihash (view) failed: " << hash_or.status();
    }
    return std::nullopt;
  }
  (void)gpu_guard;
  return *hash_or;
}

absl::StatusOr<std::string> ViewHashComputer::hash_view_from_source(
    loader::SeekableSource& base_source,
    const loader::ViewPlan& plan,
    size_t leaf_chunk_bytes) const {
  if (plan.selection.total_bytes == 0) {
    return absl::InvalidArgumentError("view plan contains no data to hash");
  }
  if (leaf_chunk_bytes == 0) {
    return absl::InvalidArgumentError("leaf_chunk_bytes must be > 0");
  }
  if (plan.selection.total_bytes > std::numeric_limits<size_t>::max()) {
    return absl::OutOfRangeError("view plan exceeds host memory limits");
  }

  loader::ViewPlanSource view_source(gsl::not_null<loader::SeekableSource*>{&base_source}, plan.selection);
  const size_t total_bytes = static_cast<size_t>(plan.selection.total_bytes);

  if (!plan.transform.requires_materialization && plan.transform.tensors.empty()) {
    return loader::compute_data_multihash_from_seekable_source(
        view_source, plan.selection.total_bytes, leaf_chunk_bytes);
  }

  std::vector<uint8_t> staging(total_bytes);
  auto read_or = view_source.read_at(0, staging.data(), staging.size());
  if (!read_or.ok()) {
    return read_or.status();
  }
  if (*read_or != staging.size()) {
    return absl::DataLossError("short read while materializing view for hashing");
  }

  absl::Status transform_status =
      loader::execute_transform(plan.transform, common::memory::MemoryLocation::CPU, staging.data(), /*device_id=*/-1);
  if (!transform_status.ok()) {
    return transform_status;
  }

  return loader::compute_data_multihash_from_cpu_memory(
      gsl::not_null<const void*>{staging.data()}, plan.selection.total_bytes, leaf_chunk_bytes);
}

absl::StatusOr<std::string> ViewHashComputer::hash_view_from_source(
    loader::SeekableSource& base_source,
    const loader::ViewPlan& plan) const {
  return hash_view_from_source(base_source, plan, default_leaf_chunk_bytes_);
}

std::optional<std::string> compute_view_data_hash(
    replica::Replica& replica,
    common::memory::MemoryLocation location,
    uint64_t view_size_bytes,
    std::optional<int> gpu_device_id) {
  static ViewHashComputer kDefaultHasher;
  return kDefaultHasher.hash_replica_view(replica, location, view_size_bytes, std::move(gpu_device_id));
}

} // namespace tensorcast::store
