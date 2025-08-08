// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/remote_key_source.h"

#include <algorithm>
#include <cstring>

#include "absl/log/log.h"
#include "core/communicator/engine/engine.h"

namespace stepcast::store::loader {

RemoteKeySource::RemoteKeySource(Options options) : options_(std::move(options)) {
  if (!options_.comm_engine) {
    LOG(ERROR) << "CommunicateEngine is null";
  }
  if (options_.memory_keys.size() != options_.buffer_sizes.size()) {
    LOG(ERROR) << "Memory keys and buffer sizes mismatch";
  }
}

absl::StatusOr<size_t> RemoteKeySource::read(void* dst, size_t max_bytes) {
  if (!options_.comm_engine) {
    return absl::InvalidArgumentError("CommunicateEngine is null");
  }

  if (total_bytes_read_ >= options_.total_size) {
    return 0; // EOF
  }

  if (current_key_index_ >= options_.memory_keys.size()) {
    return 0; // No more keys
  }

  size_t bytes_to_read = std::min(max_bytes, static_cast<size_t>(options_.total_size - total_bytes_read_));
  size_t bytes_read = 0;
  char* dst_ptr = static_cast<char*>(dst);

  while (bytes_read < bytes_to_read && current_key_index_ < options_.memory_keys.size()) {
    const auto& key = options_.memory_keys[current_key_index_];
    const auto key_size = options_.buffer_sizes[current_key_index_];

    size_t remaining_in_key = key_size - current_key_offset_;
    size_t to_read = std::min(bytes_to_read - bytes_read, remaining_in_key);

    // Use CommunicateEngine to read from remote peer directly into dst_ptr.
    auto future = options_.comm_engine->read_tensor(
        key,
        reinterpret_cast<uint64_t>(dst_ptr + bytes_read),
        static_cast<uint64_t>(to_read),
        0 /* CPU dev type */,
        -1,
        options_.ip,
        options_.port,
        static_cast<uint64_t>(current_key_offset_));

    auto result = future.get();
    if (!result.status.ok()) {
      LOG(ERROR) << "Failed to read from remote key " << key << " at offset " << current_key_offset_ << " : "
                 << result.status.message();
      return result.status;
    }

    bytes_read += to_read;
    current_key_offset_ += to_read;
    total_bytes_read_ += to_read;

    // Move to next key if current one is exhausted
    if (current_key_offset_ >= key_size) {
      current_key_index_++;
      current_key_offset_ = 0;
    }
  }

  VLOG(3) << "Read " << bytes_read << " bytes from remote. "
          << "Total read: " << total_bytes_read_ << "/" << options_.total_size;

  return bytes_read;
}

absl::StatusOr<size_t> RemoteKeySource::read_at(uint64_t offset, void* dst, size_t bytes) {
  if (!options_.comm_engine) {
    return absl::InvalidArgumentError("CommunicateEngine is null");
  }

  if (offset >= options_.total_size) {
    return 0; // Offset beyond EOF
  }

  size_t bytes_to_read = std::min(bytes, static_cast<size_t>(options_.total_size - offset));
  size_t bytes_read = 0;
  char* dst_ptr = static_cast<char*>(dst);

  // Determine starting key index and local offset within that key (stateless)
  size_t key_index = 0;
  size_t key_offset = 0;
  {
    uint64_t running_total = 0;
    for (size_t i = 0; i < options_.buffer_sizes.size(); ++i) {
      if (offset < running_total + options_.buffer_sizes[i]) {
        key_index = i;
        key_offset = static_cast<size_t>(offset - running_total);
        break;
      }
      running_total += options_.buffer_sizes[i];
    }
  }

  while (bytes_read < bytes_to_read && key_index < options_.memory_keys.size()) {
    const auto& key = options_.memory_keys[key_index];
    const auto key_size = options_.buffer_sizes[key_index];

    size_t remaining_in_key = key_size - key_offset;
    size_t to_read = std::min(bytes_to_read - bytes_read, remaining_in_key);

    auto future = options_.comm_engine->read_tensor(
        key,
        reinterpret_cast<uint64_t>(dst_ptr + bytes_read),
        static_cast<uint64_t>(to_read),
        0 /* CPU dev type */,
        -1,
        options_.ip,
        options_.port,
        static_cast<uint64_t>(key_offset));

    auto result = future.get();
    if (!result.status.ok()) {
      return result.status;
    }

    bytes_read += to_read;
    key_offset += to_read;
    if (key_offset >= key_size) {
      key_index++;
      key_offset = 0;
    }
  }

  return bytes_read;
}

bool RemoteKeySource::supports_direct_write() const {
  return options_.comm_engine && options_.comm_engine->is_rdma_enabled();
}

absl::StatusOr<size_t> RemoteKeySource::read_into(
    uint64_t dest_va_offset,
    size_t bytes,
    const stepcast::store::DirectWriteToken& token) {
  if (!options_.comm_engine) {
    return absl::InvalidArgumentError("CommunicateEngine is null");
  }
  if (dest_va_offset >= options_.total_size) {
    return 0; // Beyond EOF
  }

  // Helper: find token segment for a given VA offset
  auto find_segment = [&](uint64_t va_off) -> const DirectWriteToken::Segment* {
    for (const auto& seg : token.segments) {
      if (va_off >= seg.va_offset && va_off < seg.va_offset + seg.length) {
        return &seg;
      }
    }
    return nullptr;
  };

  size_t to_read_total = std::min<uint64_t>(bytes, options_.total_size - dest_va_offset);
  size_t bytes_done = 0;
  uint64_t src_global_offset = dest_va_offset; // assume linear global mapping

  while (bytes_done < to_read_total) {
    const uint64_t cur_va = dest_va_offset + bytes_done;
    const auto* seg = find_segment(cur_va);
    if (seg == nullptr) {
      return absl::InvalidArgumentError("DirectWriteToken lacks segment for requested VA range");
    }
    const uint64_t seg_off_in_bytes = cur_va - seg->va_offset;
    const uint64_t local_addr = seg->local_addr + seg_off_in_bytes;
    const uint64_t seg_bytes_left = seg->length - seg_off_in_bytes;

    // Determine starting key and offset for src_global_offset
    size_t key_index = 0;
    size_t key_offset = 0;
    {
      uint64_t running_total = 0;
      for (size_t i = 0; i < options_.buffer_sizes.size(); ++i) {
        if (src_global_offset < running_total + options_.buffer_sizes[i]) {
          key_index = i;
          key_offset = static_cast<size_t>(src_global_offset - running_total);
          break;
        }
        running_total += options_.buffer_sizes[i];
      }
    }

    const auto& key = options_.memory_keys[key_index];
    const auto key_size = options_.buffer_sizes[key_index];
    const size_t remaining_in_key = key_size - key_offset;
    const size_t step = static_cast<size_t>(
        std::min<uint64_t>(to_read_total - bytes_done, std::min<uint64_t>(remaining_in_key, seg_bytes_left)));

    auto future = options_.comm_engine->read_tensor(
        key,
        local_addr,
        static_cast<uint64_t>(step),
        0 /* CPU */,
        -1,
        options_.ip,
        options_.port,
        static_cast<uint64_t>(key_offset));
    auto result = future.get();
    if (!result.status.ok()) {
      return result.status;
    }

    bytes_done += step;
    src_global_offset += step;
  }
  return bytes_done;
}

} // namespace stepcast::store::loader
