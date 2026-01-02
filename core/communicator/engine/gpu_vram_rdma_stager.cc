// Copyright (c) 2025-2026, TensorCast Team.

#include "core/communicator/engine/gpu_vram_rdma_stager.h"

#include <algorithm>
#include <cstdint>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::communicator::engine {

GpuVramStagingPool::GpuVramStagingPool(int device_id, size_t pool_bytes, size_t slice_bytes)
    : device_id_(device_id), pool_bytes_(pool_bytes), slice_bytes_(slice_bytes) {}

GpuVramStagingPool::~GpuVramStagingPool() {
  auto status = release_pool();
  if (!status.ok()) {
    LOG(WARNING) << "Failed to release VRAM staging pool: " << status;
  }
}

absl::Status GpuVramStagingPool::initialize() {
  if (initialized_) {
    return absl::FailedPreconditionError("GpuVramStagingPool already initialized");
  }
  if (slice_bytes_ == 0) {
    return absl::InvalidArgumentError("GpuVramStagingPool slice_bytes must be > 0");
  }
  if (pool_bytes_ < slice_bytes_) {
    return absl::InvalidArgumentError("GpuVramStagingPool pool_bytes must be >= slice_bytes");
  }
  num_slices_ = pool_bytes_ / slice_bytes_;
  if (num_slices_ == 0) {
    return absl::InvalidArgumentError("GpuVramStagingPool has zero usable slices");
  }
  if (pool_bytes_ % slice_bytes_ != 0) {
    LOG(WARNING) << "GpuVramStagingPool pool_bytes not aligned to slice_bytes; truncating remainder."
                 << " pool_bytes=" << pool_bytes_ << " slice_bytes=" << slice_bytes_;
  }

  int device_id = std::max(device_id_, 0);
  auto device_status = cuda::set_device(device_id);
  if (!device_status.ok()) {
    return device_status;
  }

  auto alloc_status = cuda::malloc(&base_ptr_, pool_bytes_);
  if (!alloc_status.ok()) {
    return alloc_status;
  }

  {
    absl::MutexLock lock(&mu_);
    free_indices_.reserve(num_slices_);
    in_use_.assign(num_slices_, false);
    for (size_t i = 0; i < num_slices_; ++i) {
      free_indices_.push_back(static_cast<int>(i));
    }
    initialized_ = true;
  }

  return absl::OkStatus();
}

absl::StatusOr<GpuVramStagingPool::Slice> GpuVramStagingPool::acquire_slice() {
  absl::MutexLock lock(&mu_);
  if (!initialized_) {
    return absl::FailedPreconditionError("GpuVramStagingPool not initialized");
  }
  if (free_indices_.empty()) {
    return absl::ResourceExhaustedError(
        absl::StrCat(
            "GpuVramStagingPool exhausted: device_id=",
            device_id_,
            " slice_bytes=",
            slice_bytes_,
            " pool_bytes=",
            pool_bytes_));
  }
  int index = free_indices_.back();
  free_indices_.pop_back();
  in_use_.at(static_cast<size_t>(index)) = true;
  auto* base = static_cast<uint8_t*>(base_ptr_);
  void* ptr = static_cast<void*>(base + (static_cast<size_t>(index) * slice_bytes_));
  return Slice{.ptr = ptr, .bytes = slice_bytes_, .index = index};
}

absl::Status GpuVramStagingPool::release_slice(int index) {
  absl::MutexLock lock(&mu_);
  if (!initialized_) {
    return absl::FailedPreconditionError("GpuVramStagingPool not initialized");
  }
  if (index < 0 || static_cast<size_t>(index) >= num_slices_) {
    return absl::InvalidArgumentError("GpuVramStagingPool slice index out of range");
  }
  if (!in_use_.at(static_cast<size_t>(index))) {
    return absl::NotFoundError("GpuVramStagingPool slice not in use");
  }
  in_use_.at(static_cast<size_t>(index)) = false;
  free_indices_.push_back(index);
  return absl::OkStatus();
}

bool GpuVramStagingPool::contains_ptr(gsl::not_null<void*> ptr) const {
  if (base_ptr_ == nullptr || pool_bytes_ == 0) {
    return false;
  }
  const auto base = reinterpret_cast<std::uintptr_t>(base_ptr_);
  const auto end = base + pool_bytes_;
  const auto probe = reinterpret_cast<std::uintptr_t>(ptr.get());
  return probe >= base && probe < end;
}

std::optional<MemoryStager::MrSlab> GpuVramStagingPool::mr_slab() const {
  if (base_ptr_ == nullptr || pool_bytes_ == 0) {
    return std::nullopt;
  }
  return MemoryStager::MrSlab{gsl::not_null<void*>{base_ptr_}, pool_bytes_};
}

absl::Status GpuVramStagingPool::release_pool() {
  if (base_ptr_ == nullptr) {
    return absl::OkStatus();
  }
  int device_id = std::max(device_id_, 0);
  auto device_status = cuda::set_device(device_id);
  if (!device_status.ok()) {
    return device_status;
  }
  auto free_status = cuda::free(base_ptr_);
  if (!free_status.ok()) {
    return free_status;
  }
  base_ptr_ = nullptr;
  {
    absl::MutexLock lock(&mu_);
    free_indices_.clear();
    in_use_.clear();
    initialized_ = false;
  }
  return absl::OkStatus();
}

GpuVramRdmaStager::GpuVramRdmaStager(std::shared_ptr<GpuVramStagingPool> pool) : pool_(std::move(pool)) {
  if (!pool_) {
    LOG(FATAL) << "GpuVramRdmaStager requires a non-null pool";
  }
}

absl::StatusOr<void*> GpuVramRdmaStager::stage(
    const std::shared_ptr<transport::PartitionTensor>& tensor,
    uint64_t offset,
    uint64_t bytes,
    StageMode /*mode*/) {
  if (bytes == 0 || bytes > pool_->slice_bytes()) {
    return absl::InvalidArgumentError("GpuVramRdmaStager: bytes must be within (0, slice_bytes]");
  }
  if (offset + bytes > tensor->get_bytes()) {
    return absl::InvalidArgumentError("GpuVramRdmaStager: staging beyond tensor bounds");
  }
  if (tensor->get_mem_type() != base::COMMUNICATE_ENGINE_DEV_GPU) {
    return absl::InvalidArgumentError("GpuVramRdmaStager requires GPU tensors");
  }

  auto slice_or = pool_->acquire_slice();
  if (!slice_or.ok()) {
    return slice_or.status();
  }
  auto slice = slice_or.value();

  const int tensor_device_id = tensor->get_device_id();
  const int pool_device_id = pool_->device_id();
  if (tensor_device_id >= 0 && pool_device_id >= 0 && tensor_device_id != pool_device_id) {
    auto release_status = pool_->release_slice(slice.index);
    if (!release_status.ok()) {
      LOG(WARNING) << "GpuVramRdmaStager failed to release slice after device mismatch: " << release_status;
    }
    return absl::InvalidArgumentError(
        absl::StrCat(
            "GpuVramRdmaStager tensor device id ",
            tensor_device_id,
            " does not match pool device id ",
            pool_device_id));
  }
  int device_id = tensor_device_id >= 0 ? tensor_device_id : pool_device_id;
  device_id = std::max(device_id, 0);
  auto guard = cuda::set_device(device_id);
  if (!guard.ok()) {
    auto release_status = pool_->release_slice(slice.index);
    if (!release_status.ok()) {
      LOG(WARNING) << "GpuVramRdmaStager failed to release slice after device error: " << release_status;
    }
    return guard;
  }

  void* src = reinterpret_cast<void*>(tensor->get_uint64_addr() + offset);
  auto copy_status = cuda::memcpy(slice.ptr, src, bytes, cudaMemcpyDeviceToDevice);
  if (!copy_status.ok()) {
    auto release_status = pool_->release_slice(slice.index);
    if (!release_status.ok()) {
      LOG(WARNING) << "GpuVramRdmaStager failed to release slice after copy error: " << release_status;
    }
    return copy_status;
  }

  {
    absl::MutexLock lock(&mu_);
    staged_slices_[slice.ptr] = slice.index;
  }

  return slice.ptr;
}

absl::Status GpuVramRdmaStager::release_staged_buffer(gsl::not_null<void*> exposed_ptr) {
  int index = -1;
  {
    absl::MutexLock lock(&mu_);
    auto it = staged_slices_.find(exposed_ptr.get());
    if (it == staged_slices_.end()) {
      return absl::NotFoundError("GpuVramRdmaStager: exposed ptr not found");
    }
    index = it->second;
    staged_slices_.erase(it);
  }
  return pool_->release_slice(index);
}

size_t GpuVramRdmaStager::get_chunk_size() const {
  return pool_->slice_bytes();
}

size_t GpuVramRdmaStager::get_num_buffers() const {
  return pool_->num_slices();
}

std::optional<MemoryStager::MrSlab> GpuVramRdmaStager::mr_slab_for_ptr(gsl::not_null<void*> exposed_ptr) const {
  if (!pool_->contains_ptr(exposed_ptr)) {
    return std::nullopt;
  }
  return pool_->mr_slab();
}

} // namespace tensorcast::communicator::engine
