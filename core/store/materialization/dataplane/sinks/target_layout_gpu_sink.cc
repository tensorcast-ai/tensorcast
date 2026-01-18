// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/sinks/target_layout_gpu_sink.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace tensorcast::store::loader {

namespace {

constexpr size_t kInvalidIndex = std::numeric_limits<size_t>::max();

} // namespace

TargetLayoutGpuSink::TargetLayoutGpuSink(Options options) {
  storage_states_.reserve(options.storages.size());
  uint64_t cursor = 0;
  for (const auto& storage : options.storages) {
    StorageState state;
    state.base_offset = cursor;
    state.length = storage.length;
    GpuMemorySink::Options sink_opts{
        .gpu_base_ptr = storage.gpu_base_ptr,
        .total_size = storage.length,
        .chunk_size = options.chunk_size,
        .device_id = options.device_id,
        .gpu_sched_enabled = options.gpu_sched_enabled,
        .gpu_sched_limit_bytes = options.gpu_sched_limit_bytes,
        .gpu_sched_limit_copies = options.gpu_sched_limit_copies,
    };
    state.sink = std::make_unique<GpuMemorySink>(std::move(sink_opts));
    storage_states_.push_back(std::move(state));
    if (storage.length > std::numeric_limits<uint64_t>::max() - cursor) {
      total_size_ = 0;
      overall_status_ = absl::OutOfRangeError("target layout size overflow");
      return;
    }
    cursor += storage.length;
  }
  total_size_ = cursor;
}

TargetLayoutGpuSink::~TargetLayoutGpuSink() = default;

size_t TargetLayoutGpuSink::locate_storage(uint64_t offset, size_t bytes) const {
  if (storage_states_.empty()) {
    return kInvalidIndex;
  }
  if (bytes == 0) {
    return 0;
  }
  if (offset > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(bytes)) {
    return kInvalidIndex;
  }
  const uint64_t end = offset + static_cast<uint64_t>(bytes);
  for (size_t idx = 0; idx < storage_states_.size(); ++idx) {
    const auto& state = storage_states_[idx];
    const uint64_t state_end = state.base_offset + state.length;
    if (offset >= state.base_offset && end <= state_end) {
      return idx;
    }
  }
  return kInvalidIndex;
}

absl::Status TargetLayoutGpuSink::write(const void* src, size_t bytes) {
  if (!overall_status_.ok()) {
    return overall_status_;
  }
  auto st = write_at(current_offset_, src, bytes);
  if (st.ok()) {
    current_offset_ += bytes;
  }
  return st;
}

absl::Status TargetLayoutGpuSink::write_at(uint64_t offset, const void* src, size_t bytes) {
  if (!overall_status_.ok()) {
    return overall_status_;
  }
  if (bytes == 0) {
    return absl::OkStatus();
  }
  if (offset > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(bytes)) {
    overall_status_ = absl::OutOfRangeError("write_at offset overflows target layout");
    return overall_status_;
  }
  if (offset + static_cast<uint64_t>(bytes) > total_size_) {
    overall_status_ = absl::InvalidArgumentError("write_at exceeds target layout size");
    return overall_status_;
  }
  const size_t idx = locate_storage(offset, bytes);
  if (idx == kInvalidIndex) {
    overall_status_ = absl::InvalidArgumentError("write_at spans multiple target storages");
    return overall_status_;
  }
  StorageState& state = storage_states_[idx];
  const uint64_t local_offset = offset - state.base_offset;
  absl::Status st = state.sink->write_at(local_offset, src, bytes);
  if (!st.ok()) {
    overall_status_ = st;
  }
  return st;
}

absl::StatusOr<common::CopyHandle> TargetLayoutGpuSink::write_at_async(
    uint64_t offset,
    const void* src,
    size_t bytes,
    const AsyncWriteOptions& options) {
  if (!overall_status_.ok()) {
    return overall_status_;
  }
  if (bytes == 0) {
    return common::CopyHandle{};
  }
  if (offset > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(bytes)) {
    overall_status_ = absl::OutOfRangeError("write_at_async offset overflows target layout");
    return overall_status_;
  }
  if (offset + static_cast<uint64_t>(bytes) > total_size_) {
    overall_status_ = absl::InvalidArgumentError("write_at_async exceeds target layout size");
    return overall_status_;
  }
  const size_t idx = locate_storage(offset, bytes);
  if (idx == kInvalidIndex) {
    overall_status_ = absl::InvalidArgumentError("write_at_async spans multiple target storages");
    return overall_status_;
  }
  StorageState& state = storage_states_[idx];
  const uint64_t local_offset = offset - state.base_offset;
  auto handle_or = state.sink->write_at_async(local_offset, src, bytes, options);
  if (!handle_or.ok()) {
    overall_status_ = handle_or.status();
  }
  return handle_or;
}

absl::Status TargetLayoutGpuSink::close() {
  if (!overall_status_.ok()) {
    return overall_status_;
  }
  for (auto& state : storage_states_) {
    absl::Status st = state.sink->close();
    if (!st.ok()) {
      overall_status_ = st;
      return overall_status_;
    }
  }
  return absl::OkStatus();
}

} // namespace tensorcast::store::loader
