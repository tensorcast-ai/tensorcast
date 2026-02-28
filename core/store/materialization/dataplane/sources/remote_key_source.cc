// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/dataplane/sources/remote_key_source.h"

#include <algorithm>
#include <cstring>
#include <exception>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/communicator/engine/engine.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"

namespace tensorcast::store::loader {

namespace {

bool can_use_routed_read(const RemoteKeySource::Options& options, bool routed_path_enabled) {
  return routed_path_enabled && options.routing_context != nullptr &&
      !options.local_endpoint_id.empty() && !options.remote_endpoint_id.empty();
}

communicator::transport::read_result_t read_direct(
    const RemoteKeySource::Options& options,
    const std::string& key,
    uint64_t local_addr,
    uint64_t bytes,
    uint64_t remote_offset,
    int dev_type,
    int dev_id) {
  auto future = options.comm_engine->read_tensor(
      key,
      local_addr,
      bytes,
      dev_type,
      dev_id,
      options.ip,
      options.port,
      remote_offset);
  return future.get();
}

absl::StatusOr<communicator::transport::read_result_t> read_routed(
    const RemoteKeySource::Options& options,
    const std::string& key,
    uint64_t local_addr,
    uint64_t bytes,
    uint64_t remote_offset,
    int dev_type,
    int dev_id) {
  auto communicator_or =
      options.routing_context->get_communicator(options.local_endpoint_id, options.remote_endpoint_id);
  if (!communicator_or.ok()) {
    return communicator_or.status();
  }

  communicator::routing::ReadRequest request{
      .tensor_key = key,
      .addr = local_addr,
      .bytes = bytes,
      .dev_type = dev_type,
      .dev_id = dev_id,
      .remote_offset = remote_offset,
  };
  try {
    auto future = communicator_or.value()->read_tensor(request);
    auto result = future.get();
    if (!result.status.ok()) {
      return result.status;
    }
    return result;
  } catch (const std::exception& ex) {
    return absl::InternalError(absl::StrCat("routed read future failed: ", ex.what()));
  }
}

communicator::transport::read_result_t read_with_strict_fallback(
    const RemoteKeySource::Options& options,
    const std::string& key,
    uint64_t local_addr,
    uint64_t bytes,
    uint64_t remote_offset,
    int dev_type,
    int dev_id,
    bool* routed_path_enabled) {
  if (can_use_routed_read(options, *routed_path_enabled)) {
    auto routed_or = read_routed(options, key, local_addr, bytes, remote_offset, dev_type, dev_id);
    if (routed_or.ok()) {
      return *routed_or;
    }
    *routed_path_enabled = false;
    LOG(WARNING) << "Routed read failed for src_endpoint=" << options.local_endpoint_id
                 << " dst_endpoint=" << options.remote_endpoint_id << ", fallback to direct ip/port " << options.ip
                 << ":" << options.port << " reason=" << routed_or.status();
  }

  return read_direct(options, key, local_addr, bytes, remote_offset, dev_type, dev_id);
}

} // namespace

RemoteKeySource::RemoteKeySource(Options options) : options_(std::move(options)) {
  if (options_.memory_keys.size() != options_.buffer_sizes.size()) {
    LOG(ERROR) << "Memory keys and buffer sizes mismatch";
  }
}

absl::StatusOr<size_t> RemoteKeySource::read(void* dst, size_t max_bytes) {
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

    namespace otel = opentelemetry;
    auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.store");
    auto evt_span = tracer->StartSpan("P2P/ChunkRecv");
    otel::trace::Scope evt_scope(evt_span);
    evt_span->SetAttribute("event", "p2p_chunk_recv");
    evt_span->SetAttribute("tc.conn.index", static_cast<int64_t>(current_key_index_));
    evt_span->SetAttribute("tc.offset", static_cast<int64_t>(current_key_offset_));
    evt_span->SetAttribute("tc.size.bytes", static_cast<int64_t>(to_read));
    evt_span->SetAttribute("tc.remote.key", key);

    auto result = read_with_strict_fallback(
        options_,
        key,
        reinterpret_cast<uint64_t>(dst_ptr + bytes_read),
        static_cast<uint64_t>(to_read),
        static_cast<uint64_t>(current_key_offset_),
        communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
        -1,
        &routed_path_enabled_);
    if (!result.status.ok()) {
      evt_span->SetAttribute("error", true);
      evt_span->AddEvent("recv_error", {{"message", result.status.message()}});
      evt_span->End();
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

    evt_span->End();
  }

  VLOG(3) << "Read " << bytes_read << " bytes from remote. "
          << "Total read: " << total_bytes_read_ << "/" << options_.total_size;

  return bytes_read;
}

absl::StatusOr<size_t> RemoteKeySource::read_at(uint64_t offset, void* dst, size_t bytes) {
  if (offset >= options_.total_size) {
    return 0; // Offset beyond EOF
  }

  // Validate key metadata to prevent out-of-range access.
  if (options_.memory_keys.empty() || options_.buffer_sizes.empty()) {
    return absl::InvalidArgumentError("RemoteKeySource has no memory keys/buffer sizes configured");
  }
  if (options_.memory_keys.size() != options_.buffer_sizes.size()) {
    return absl::InvalidArgumentError("Memory keys and buffer sizes size mismatch");
  }

  size_t bytes_to_read = std::min(bytes, static_cast<size_t>(options_.total_size - offset));
  size_t bytes_read = 0;
  char* dst_ptr = static_cast<char*>(dst);

  // Determine starting key index and local offset within that key (stateless)
  size_t key_index = 0;
  size_t key_offset = 0;
  bool found_segment = false;
  {
    uint64_t running_total = 0;
    for (size_t i = 0; i < options_.buffer_sizes.size(); ++i) {
      const uint64_t segment_end = running_total + options_.buffer_sizes[i];
      if (offset < segment_end) {
        key_index = i;
        key_offset = static_cast<size_t>(offset - running_total);
        found_segment = true;
        break;
      }
      running_total = segment_end;
    }
  }

  if (!found_segment) {
    return absl::InternalError("Failed to map offset to a valid key segment");
  }
  if (key_index >= options_.memory_keys.size() || key_index >= options_.buffer_sizes.size()) {
    return absl::InternalError("Computed key_index is out of range");
  }
  if (key_offset >= options_.buffer_sizes[key_index]) {
    return absl::InternalError("Computed key_offset exceeds key segment size");
  }

  while (bytes_read < bytes_to_read && key_index < options_.memory_keys.size()) {
    const auto& key = options_.memory_keys[key_index];
    const auto key_size = options_.buffer_sizes[key_index];

    size_t remaining_in_key = key_size - key_offset;
    size_t to_read = std::min(bytes_to_read - bytes_read, remaining_in_key);

    namespace otel = opentelemetry;
    auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.store");
    auto evt_span = tracer->StartSpan("P2P/ChunkRecv");
    otel::trace::Scope evt_scope(evt_span);
    evt_span->SetAttribute("event", "p2p_chunk_recv");
    evt_span->SetAttribute("tc.conn.index", static_cast<int64_t>(key_index));
    evt_span->SetAttribute("tc.offset", static_cast<int64_t>(key_offset));
    evt_span->SetAttribute("tc.size.bytes", static_cast<int64_t>(to_read));
    evt_span->SetAttribute("tc.remote.key", key);

    auto result = read_with_strict_fallback(
        options_,
        key,
        reinterpret_cast<uint64_t>(dst_ptr + bytes_read),
        static_cast<uint64_t>(to_read),
        static_cast<uint64_t>(key_offset),
        communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
        -1,
        &routed_path_enabled_);
    if (!result.status.ok()) {
      evt_span->SetAttribute("error", true);
      evt_span->AddEvent("recv_error", {{"message", result.status.message()}});
      evt_span->End();
      return result.status;
    }

    bytes_read += to_read;
    key_offset += to_read;
    if (key_offset >= key_size) {
      key_index++;
      key_offset = 0;
    }

    evt_span->End();
  }

  return bytes_read;
}

bool RemoteKeySource::supports_direct_write() const {
  return options_.comm_engine->is_rdma_enabled();
}

absl::StatusOr<size_t> RemoteKeySource::read_into(
    uint64_t dest_va_offset,
    size_t bytes,
    const DirectWriteGrant& grant) {
  if (dest_va_offset >= options_.total_size) {
    return 0; // Beyond EOF
  }

  // Helper: find grant window for a given VA offset
  auto find_window = [&](uint64_t va_off) -> const DirectWriteGrant::Window* {
    for (const auto& win : grant.windows) {
      if (va_off >= win.va_offset && va_off < win.va_offset + win.length) {
        return &win;
      }
    }
    return nullptr;
  };

  size_t to_read_total = std::min<uint64_t>(bytes, options_.total_size - dest_va_offset);
  size_t bytes_done = 0;
  uint64_t src_global_offset = dest_va_offset; // assume linear global mapping

  while (bytes_done < to_read_total) {
    const uint64_t cur_va = dest_va_offset + bytes_done;
    const auto* win = find_window(cur_va);
    if (win == nullptr) {
      return absl::InvalidArgumentError("DirectWriteGrant lacks window for requested VA range");
    }
    const uint64_t seg_off_in_bytes = cur_va - win->va_offset;
    const uint64_t local_addr = win->local_addr + seg_off_in_bytes;
    const uint64_t seg_bytes_left = win->length - seg_off_in_bytes;

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

    // Span per direct-write segment
    namespace otel = opentelemetry;
    auto tracer = otel::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.store");
    auto evt_span = tracer->StartSpan("P2P/DirectWrite");
    otel::trace::Scope evt_scope(evt_span);
    evt_span->SetAttribute("event", "p2p_chunk_recv");
    evt_span->SetAttribute("tc.offset", static_cast<int64_t>(src_global_offset));
    evt_span->SetAttribute("tc.size.bytes", static_cast<int64_t>(step));
    evt_span->SetAttribute("tc.remote.key", key);

    auto result = read_with_strict_fallback(
        options_,
        key,
        local_addr,
        static_cast<uint64_t>(step),
        static_cast<uint64_t>(key_offset),
        communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
        -1,
        &routed_path_enabled_);
    if (!result.status.ok()) {
      evt_span->SetAttribute("error", true);
      evt_span->AddEvent("recv_error", {{"message", result.status.message()}});
      evt_span->End();
      return result.status;
    }

    bytes_done += step;
    src_global_offset += step;

    evt_span->End();
  }
  return bytes_done;
}

} // namespace tensorcast::store::loader
