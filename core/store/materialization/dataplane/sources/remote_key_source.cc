// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/remote_key_source.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/communicator/engine/engine.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"

namespace tensorcast::store::loader {

namespace {

constexpr uint64_t kSlowQueueCostUs = 1'000'000; // 1s
constexpr uint64_t kSlowReadCostUs = 5'000'000; // 5s

bool should_log_cost_breakdown(const communicator::transport::read_result_t& result) {
  return result.request_cost >= kSlowQueueCostUs || result.rdma_queue_cost >= kSlowQueueCostUs ||
      result.read_cost >= kSlowReadCostUs || !result.status.ok();
}

std::string format_cost_breakdown(const communicator::transport::read_result_t& result) {
  return absl::StrCat(
      "request_cost_us=",
      result.request_cost,
      " read_cost_us=",
      result.read_cost,
      " rdma_queue_cost_us=",
      result.rdma_queue_cost,
      " rdma_regmr_cost_us=",
      result.rdma_regmr_cost);
}

} // namespace

RemoteKeySource::RemoteKeySource(Options options) : options_(std::move(options)) {
  if (options_.memory_keys.size() != options_.buffer_sizes.size()) {
    LOG(ERROR) << "Memory keys and buffer sizes mismatch";
  }
}

std::chrono::milliseconds RemoteKeySource::remaining_request_budget() const {
  if (options_.request_budget.count() <= 0) {
    return std::chrono::milliseconds(0);
  }
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - request_start_);
  if (elapsed >= options_.request_budget) {
    return std::chrono::milliseconds(0);
  }
  return options_.request_budget - elapsed;
}

void RemoteKeySource::abort_timed_out_channel(std::string_view key, uint64_t remote_offset, size_t bytes) const {
  absl::Status close_status = options_.comm_engine->close_connection(options_.ip, options_.port);
  if (!close_status.ok()) {
    VLOG(1) << "RemoteKeySource timeout cleanup could not close channel"
            << " artifact_id=" << options_.artifact_id << " key=" << key << " peer=" << options_.ip << ":"
            << options_.port << " remote_offset=" << remote_offset << " bytes=" << bytes << " status=" << close_status;
    return;
  }
  LOG(WARNING) << "RemoteKeySource closed channel after request budget timeout"
               << " artifact_id=" << options_.artifact_id << " key=" << key << " peer=" << options_.ip << ":"
               << options_.port << " remote_offset=" << remote_offset << " bytes=" << bytes;
}

absl::StatusOr<communicator::transport::read_result_t> RemoteKeySource::await_read_result(
    communicator::transport::future_read_result_t& future,
    std::string_view key,
    uint64_t remote_offset,
    size_t bytes) {
  const std::chrono::milliseconds wait_slice =
      options_.wait_slice.count() > 0 ? options_.wait_slice : std::chrono::seconds(5);
  const std::chrono::milliseconds stalled_log_interval =
      options_.stalled_log_interval.count() > 0 ? options_.stalled_log_interval : std::chrono::seconds(30);
  const auto wait_start = std::chrono::steady_clock::now();

  while (true) {
    std::chrono::milliseconds wait_budget = wait_slice;
    if (options_.request_budget.count() > 0) {
      const std::chrono::milliseconds remaining = remaining_request_budget();
      if (remaining.count() <= 0) {
        const auto waited_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - wait_start)
                .count();
        abort_timed_out_channel(key, remote_offset, bytes);
        return absl::DeadlineExceededError(
            absl::StrCat(
                "RemoteKeySource read timed out: artifact_id=",
                options_.artifact_id,
                " key=",
                key,
                " remote_offset=",
                remote_offset,
                " bytes=",
                bytes,
                " waited_ms=",
                waited_ms,
                " request_budget_ms=",
                options_.request_budget.count()));
      }
      wait_budget = std::min(wait_budget, remaining);
    }

    const std::future_status status = future.wait_for(wait_budget);
    if (status == std::future_status::ready) {
      auto result = future.get();
      if (should_log_cost_breakdown(result)) {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_cost_log_ >= stalled_log_interval) {
          last_cost_log_ = now;
          const auto waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - wait_start).count();
          LOG(WARNING) << "RemoteKeySource read completed with slow stage or error"
                       << " artifact_id=" << options_.artifact_id << " key=" << key
                       << " remote_offset=" << remote_offset << " bytes=" << bytes << " waited_ms=" << waited_ms
                       << " remaining_budget_ms=" << remaining_request_budget().count() << " "
                       << format_cost_breakdown(result) << " status=" << result.status;
        }
      }
      return result;
    }
    if (status == std::future_status::deferred) {
      return future.get();
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - last_stalled_log_ >= stalled_log_interval) {
      last_stalled_log_ = now;
      const auto waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - wait_start).count();
      const auto remaining_ms = remaining_request_budget().count();
      LOG(WARNING) << "RemoteKeySource read still pending"
                   << " artifact_id=" << options_.artifact_id << " key=" << key << " remote_offset=" << remote_offset
                   << " bytes=" << bytes << " waited_ms=" << waited_ms << " remaining_budget_ms=" << remaining_ms;
    }
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

    // Use Communicator to read from remote peer directly into dst_ptr.
    auto future = options_.comm_engine->read_tensor(
        key,
        reinterpret_cast<uint64_t>(dst_ptr + bytes_read),
        static_cast<uint64_t>(to_read),
        0 /* CPU dev type */,
        -1,
        options_.ip,
        options_.port,
        static_cast<uint64_t>(current_key_offset_));

    auto result_or = await_read_result(future, key, current_key_offset_, to_read);
    if (!result_or.ok()) {
      evt_span->SetAttribute("error", true);
      evt_span->AddEvent("recv_error", {{"message", std::string(result_or.status().message())}});
      evt_span->End();
      LOG(ERROR) << "Failed to read from remote key " << key << " at offset " << current_key_offset_ << " : "
                 << result_or.status();
      return result_or.status();
    }
    auto result = *result_or;
    if (!result.status.ok()) {
      evt_span->SetAttribute("error", true);
      evt_span->AddEvent("recv_error", {{"message", result.status.message()}});
      evt_span->End();
      LOG(ERROR) << "Failed to read from remote key " << key << " at offset " << current_key_offset_ << " : "
                 << result.status.message() << " " << format_cost_breakdown(result);
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

  if (bytes_read != bytes_to_read) {
    return absl::DataLossError("RemoteKeySource short read before expected EOF");
  }
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

    auto future = options_.comm_engine->read_tensor(
        key,
        reinterpret_cast<uint64_t>(dst_ptr + bytes_read),
        static_cast<uint64_t>(to_read),
        0 /* CPU dev type */,
        -1,
        options_.ip,
        options_.port,
        static_cast<uint64_t>(key_offset));

    auto result_or = await_read_result(future, key, key_offset, to_read);
    if (!result_or.ok()) {
      evt_span->SetAttribute("error", true);
      evt_span->AddEvent("recv_error", {{"message", std::string(result_or.status().message())}});
      evt_span->End();
      return result_or.status();
    }
    auto result = *result_or;
    if (!result.status.ok()) {
      evt_span->SetAttribute("error", true);
      evt_span->AddEvent("recv_error", {{"message", result.status.message()}});
      evt_span->End();
      LOG(ERROR) << "read_at failed for key=" << key << " key_offset=" << key_offset << " key_size=" << key_size
                 << " total_size=" << options_.total_size << " bytes=" << to_read << " "
                 << format_cost_breakdown(result) << " status=" << result.status;
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

  if (bytes_read != bytes_to_read) {
    return absl::DataLossError("RemoteKeySource short read before expected EOF");
  }
  return bytes_read;
}

bool RemoteKeySource::supports_direct_write_at() const {
  return options_.comm_engine->is_rdma_enabled();
}

absl::StatusOr<size_t> RemoteKeySource::read_into_at(
    uint64_t src_offset,
    uint64_t dest_va_offset,
    size_t bytes,
    const DirectWriteGrant& grant) {
  if (src_offset >= options_.total_size) {
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

  size_t to_read_total = std::min<uint64_t>(bytes, options_.total_size - src_offset);
  size_t bytes_done = 0;
  uint64_t src_global_offset = src_offset;

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

    auto future = options_.comm_engine->read_tensor(
        key,
        local_addr,
        static_cast<uint64_t>(step),
        0 /* CPU */,
        -1,
        options_.ip,
        options_.port,
        static_cast<uint64_t>(key_offset));
    auto result_or = await_read_result(future, key, key_offset, step);
    if (!result_or.ok()) {
      evt_span->SetAttribute("error", true);
      evt_span->AddEvent("recv_error", {{"message", std::string(result_or.status().message())}});
      evt_span->End();
      return result_or.status();
    }
    auto result = *result_or;
    if (!result.status.ok()) {
      evt_span->SetAttribute("error", true);
      evt_span->AddEvent("recv_error", {{"message", result.status.message()}});
      evt_span->End();
      LOG(ERROR) << "read_into_at failed for key=" << key << " key_offset=" << key_offset << " bytes=" << step << " "
                 << format_cost_breakdown(result) << " status=" << result.status;
      return result.status;
    }

    bytes_done += step;
    src_global_offset += step;

    evt_span->End();
  }
  return bytes_done;
}

} // namespace tensorcast::store::loader
