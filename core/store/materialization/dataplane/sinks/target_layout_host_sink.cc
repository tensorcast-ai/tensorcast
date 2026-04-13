// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/sinks/target_layout_host_sink.h"

#include <cstring>
#include <limits>
#include <utility>

namespace tensorcast::store::loader {

namespace {

constexpr size_t kInvalidIndex = std::numeric_limits<size_t>::max();

} // namespace

TargetLayoutHostSink::TargetLayoutHostSink(Options options) : keepalive_(std::move(options.keepalive)) {
  storage_states_.reserve(options.storages.size());
  uint64_t cursor = 0;
  for (const auto& storage : options.storages) {
    storage_states_.push_back(
        StorageState{
            .base_offset = cursor,
            .length = storage.length,
            .base_ptr = storage.base_ptr,
        });
    if (storage.length > std::numeric_limits<uint64_t>::max() - cursor) {
      total_size_ = 0;
      overall_status_ = absl::OutOfRangeError("target layout size overflow");
      return;
    }
    cursor += storage.length;
  }
  total_size_ = cursor;
}

size_t TargetLayoutHostSink::find_storage_index(uint64_t offset) const {
  for (size_t idx = 0; idx < storage_states_.size(); ++idx) {
    const auto& state = storage_states_[idx];
    if (offset >= state.base_offset && offset < state.base_offset + state.length) {
      return idx;
    }
  }
  return kInvalidIndex;
}

absl::Status TargetLayoutHostSink::write(const void* src, size_t bytes) {
  if (!overall_status_.ok()) {
    return overall_status_;
  }
  auto status = write_at(current_offset_, src, bytes);
  if (status.ok()) {
    current_offset_ += bytes;
  }
  return status;
}

absl::Status TargetLayoutHostSink::write_at(uint64_t offset, const void* src, size_t bytes) {
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

  const auto* src_bytes = static_cast<const std::byte*>(src);
  uint64_t cursor = offset;
  size_t copied = 0;
  size_t remaining = bytes;
  while (remaining > 0) {
    const size_t idx = find_storage_index(cursor);
    if (idx == kInvalidIndex) {
      overall_status_ = absl::InvalidArgumentError("write_at targets an unmapped storage range");
      return overall_status_;
    }
    const auto& state = storage_states_[idx];
    const uint64_t local_offset = cursor - state.base_offset;
    const size_t available = static_cast<size_t>(state.length - local_offset);
    const size_t take = std::min(remaining, available);
    std::memcpy(static_cast<std::byte*>(state.base_ptr.get()) + local_offset, src_bytes + copied, take);
    cursor += take;
    copied += take;
    remaining -= take;
  }
  return absl::OkStatus();
}

absl::StatusOr<DirectWriteGrant> TargetLayoutHostSink::plan_direct_write(absl::Span<const VaRange> ranges) {
  if (!overall_status_.ok()) {
    return overall_status_;
  }
  DirectWriteGrant grant;
  grant.keepalive = keepalive_;
  for (const auto& range : ranges) {
    if (range.length == 0) {
      continue;
    }
    if (range.offset > std::numeric_limits<uint64_t>::max() - range.length) {
      return absl::OutOfRangeError("plan_direct_write offset overflows target layout");
    }
    if (range.offset + range.length > total_size_) {
      return absl::OutOfRangeError("plan_direct_write exceeds target layout size");
    }
    uint64_t cursor = range.offset;
    uint64_t remaining = range.length;
    while (remaining > 0) {
      const size_t idx = find_storage_index(cursor);
      if (idx == kInvalidIndex) {
        return absl::InvalidArgumentError("plan_direct_write targets an unmapped storage range");
      }
      const auto& state = storage_states_[idx];
      const uint64_t local_offset = cursor - state.base_offset;
      const uint64_t available = state.length - local_offset;
      const uint64_t take = std::min(remaining, available);
      grant.windows.push_back(
          DirectWriteGrant::Window{
              .va_offset = cursor,
              .local_addr = reinterpret_cast<uint64_t>(static_cast<std::byte*>(state.base_ptr.get()) + local_offset),
              .length = take,
          });
      cursor += take;
      remaining -= take;
    }
  }
  return grant;
}

} // namespace tensorcast::store::loader
