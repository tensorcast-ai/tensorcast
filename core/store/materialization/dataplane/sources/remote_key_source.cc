// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/remote_key_source.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <future>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/communicator/engine/engine.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"

namespace tensorcast::store::loader {

communicator::routing::ConnectionProtocol normalize_direct_write_read_plan_protocol(
    communicator::routing::ConnectionProtocol protocol,
    const communicator::routing::EndpointBinding& local_binding,
    const communicator::routing::EndpointBinding& remote_binding,
    bool rdma_enabled) {
  const bool cross_node = !local_binding.node_id.empty() && !remote_binding.node_id.empty() &&
      local_binding.node_id != remote_binding.node_id;
  if (protocol == communicator::routing::ConnectionProtocol::kAuto && cross_node && rdma_enabled) {
    return communicator::routing::ConnectionProtocol::kRdma;
  }
  return protocol;
}

namespace {

bool can_use_routed_read(const RemoteKeySource::Options& options, bool routed_path_enabled) {
  return routed_path_enabled && options.routing_context != nullptr && !options.local_endpoint_id.empty() &&
      !options.remote_endpoint_id.empty();
}

struct ReadAttempt {
  communicator::transport::future_read_result_t future;
  bool used_routed = false;
};

bool is_local_direct_protocol(communicator::routing::ConnectionProtocol protocol) {
  return protocol == communicator::routing::ConnectionProtocol::kNvlink ||
      protocol == communicator::routing::ConnectionProtocol::kPcie;
}

bool is_pre_issue_direct_write_capability_miss(const absl::Status& status) {
  return absl::IsUnimplemented(status) || absl::IsFailedPrecondition(status);
}

communicator::transport::future_read_result_t read_direct(
    const RemoteKeySource::Options& options,
    const std::string& key,
    uint64_t local_addr,
    uint64_t bytes,
    uint64_t remote_offset,
    int dev_type,
    int dev_id) {
  auto future = options.comm_engine->read_tensor(
      key, local_addr, bytes, dev_type, dev_id, options.ip, options.port, remote_offset);
  return future;
}

absl::StatusOr<communicator::transport::future_read_result_t> read_routed(
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
    return communicator_or.value()->read_tensor(request);
  } catch (const std::exception& ex) {
    return absl::InternalError(absl::StrCat("routed read future failed: ", ex.what()));
  }
}

absl::StatusOr<communicator::transport::future_read_result_t> read_routed_with_communicator(
    const std::shared_ptr<communicator::routing::RoutingContext::Communicator>& communicator,
    const std::string& key,
    uint64_t local_addr,
    uint64_t bytes,
    uint64_t remote_offset,
    int dev_type,
    int dev_id) {
  if (communicator == nullptr) {
    return absl::FailedPreconditionError("routed direct-write communicator is null");
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
    return communicator->read_tensor(request);
  } catch (const std::exception& ex) {
    return absl::InternalError(absl::StrCat("routed direct-write future failed: ", ex.what()));
  }
}

absl::StatusOr<communicator::transport::future_read_result_t> submit_routed_read_plan(
    const std::shared_ptr<communicator::routing::RoutingContext::Communicator>& communicator,
    const communicator::routing::ReadPlan& plan) {
  if (communicator == nullptr) {
    return absl::FailedPreconditionError("routed read-plan communicator is null");
  }
  try {
    return communicator->read_plan(plan);
  } catch (const std::exception& ex) {
    return absl::InternalError(absl::StrCat("routed read-plan future failed: ", ex.what()));
  }
}

void log_routed_fallback(const RemoteKeySource::Options& options, const absl::Status& status) {
  LOG(WARNING) << "Routed read failed for src_endpoint=" << options.local_endpoint_id
               << " dst_endpoint=" << options.remote_endpoint_id << ", fallback to direct ip/port " << options.ip << ":"
               << options.port << " reason=" << status;
}

void log_read_plan_direct_write_fallback(
    const RemoteKeySource::Options& options,
    communicator::routing::ConnectionProtocol protocol,
    const absl::Status& status) {
  LOG(INFO) << "RemoteKeySource routed ReadPlan direct-write fallback to per-op read_tensor"
            << " authority=" << (options.authority_id.empty() ? "<unset>" : options.authority_id)
            << " artifact=" << (options.artifact_id.empty() ? "<unset>" : options.artifact_id)
            << " route=" << options.local_endpoint_id << "->" << options.remote_endpoint_id
            << " protocol=" << communicator::routing::to_string(protocol) << " status=" << status;
}

ReadAttempt begin_read_with_strict_fallback(
    const RemoteKeySource::Options& options,
    const std::string& key,
    uint64_t local_addr,
    uint64_t bytes,
    uint64_t remote_offset,
    int dev_type,
    int dev_id,
    bool* routed_path_enabled) {
  if (can_use_routed_read(options, *routed_path_enabled)) {
    auto routed_future_or = read_routed(options, key, local_addr, bytes, remote_offset, dev_type, dev_id);
    if (routed_future_or.ok()) {
      return ReadAttempt{.future = std::move(*routed_future_or), .used_routed = true};
    }
    *routed_path_enabled = false;
    log_routed_fallback(options, routed_future_or.status());
  }

  return ReadAttempt{
      .future = read_direct(options, key, local_addr, bytes, remote_offset, dev_type, dev_id),
      .used_routed = false,
  };
}

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

absl::Status RemoteKeySource::validate_key_layout() const {
  if (options_.memory_keys.empty() || options_.buffer_sizes.empty()) {
    return absl::InvalidArgumentError("RemoteKeySource has no memory keys/buffer sizes configured");
  }
  if (options_.memory_keys.size() != options_.buffer_sizes.size()) {
    return absl::InvalidArgumentError("Memory keys and buffer sizes size mismatch");
  }
  return absl::OkStatus();
}

absl::StatusOr<RemoteKeySource::LocatedKeySegment> RemoteKeySource::locate_key_segment(uint64_t offset) const {
  auto layout_status = validate_key_layout();
  if (!layout_status.ok()) {
    return layout_status;
  }
  if (offset >= options_.total_size) {
    return absl::OutOfRangeError("RemoteKeySource offset beyond EOF");
  }

  uint64_t running_total = 0;
  for (size_t i = 0; i < options_.buffer_sizes.size(); ++i) {
    const uint64_t segment_end = running_total + options_.buffer_sizes[i];
    if (offset < segment_end) {
      const size_t key_offset = static_cast<size_t>(offset - running_total);
      if (key_offset >= options_.buffer_sizes[i]) {
        return absl::InternalError("Computed key_offset exceeds key segment size");
      }
      return LocatedKeySegment{
          .key_index = i,
          .key_offset = key_offset,
      };
    }
    running_total = segment_end;
  }

  return absl::InternalError("Failed to map offset to a valid key segment");
}

absl::StatusOr<RemoteKeySource::FrozenDirectWriteContext> RemoteKeySource::freeze_direct_write_context() const {
  auto layout_status = validate_key_layout();
  if (!layout_status.ok()) {
    return layout_status;
  }

  if (can_use_routed_read(options_, /*routed_path_enabled=*/true)) {
    auto communicator_or =
        options_.routing_context->get_communicator(options_.local_endpoint_id, options_.remote_endpoint_id);
    if (!communicator_or.ok()) {
      return absl::FailedPreconditionError(
          absl::StrCat("RemoteKeySource direct-write route freeze failed: ", communicator_or.status().message()));
    }
    auto channel_or = communicator_or.value()->primary_channel();
    if (!channel_or.ok()) {
      return absl::FailedPreconditionError(
          absl::StrCat("RemoteKeySource direct-write channel freeze failed: ", channel_or.status().message()));
    }
    if (channel_or.value() == nullptr) {
      return absl::FailedPreconditionError("RemoteKeySource direct-write route freeze returned null channel");
    }
    if (channel_or.value()->hop_count() != 1) {
      return absl::UnimplementedError("RemoteKeySource direct-write requires a single-hop routed channel");
    }
    const std::shared_ptr<communicator::routing::Connection>& hop = channel_or.value()->hops().front();
    if (hop == nullptr) {
      return absl::FailedPreconditionError("RemoteKeySource direct-write route freeze returned null hop");
    }
    const communicator::routing::ConnectionProtocol protocol = normalize_direct_write_read_plan_protocol(
        hop->protocol(), hop->local_binding(), hop->remote_binding(), options_.comm_engine->is_rdma_enabled());
    if (!options_.comm_engine->is_rdma_enabled() && !is_local_direct_protocol(protocol)) {
      return absl::UnimplementedError("RemoteKeySource direct write requires RDMA or a same-node local direct route");
    }
    const int16_t rail_id =
        hop->local_binding().rail_id >= 0 ? hop->local_binding().rail_id : hop->remote_binding().rail_id;
    return FrozenDirectWriteContext{
        .mode = DirectWriteMode::kRouted,
        .routed_communicator = std::move(*communicator_or),
        .route_context =
            communicator::routing::ReadRouteContext{
                .local_endpoint_id = options_.local_endpoint_id,
                .remote_endpoint_id = options_.remote_endpoint_id,
                .protocol = protocol,
                .rail_id = rail_id,
            },
    };
  }

  if (!options_.comm_engine->is_rdma_enabled()) {
    return absl::UnimplementedError("RemoteKeySource direct write requires an RDMA-enabled communicator");
  }
  return FrozenDirectWriteContext{
      .mode = DirectWriteMode::kDirect,
      .routed_communicator = nullptr,
      .route_context =
          communicator::routing::ReadRouteContext{
              .local_endpoint_id = options_.local_endpoint_id,
              .remote_endpoint_id = options_.remote_endpoint_id,
              .protocol = communicator::routing::ConnectionProtocol::kRdma,
              .rail_id = -1,
          },
  };
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

std::string RemoteKeySource::resolved_authority_id() const {
  if (!options_.authority_id.empty()) {
    return options_.authority_id;
  }
  if (!options_.artifact_id.empty()) {
    return options_.artifact_id;
  }
  if (!options_.remote_endpoint_id.empty()) {
    return absl::StrCat("endpoint:", options_.remote_endpoint_id);
  }
  return "remote_key_source";
}

absl::StatusOr<communicator::routing::ReadPlan> RemoteKeySource::build_routed_read_plan(
    absl::Span<const DirectWriteOp> ops,
    const DirectWriteGrant& grant,
    const FrozenDirectWriteContext& context) const {
  if (context.route_context.protocol != communicator::routing::ConnectionProtocol::kRdma ||
      context.route_context.local_endpoint_id.empty() || context.route_context.remote_endpoint_id.empty()) {
    return absl::FailedPreconditionError("RemoteKeySource direct-write ReadPlan requires RDMA route metadata");
  }

  communicator::routing::ReadPlan plan;
  plan.local_regions.reserve(grant.windows.size());
  for (const auto& window : grant.windows) {
    if (window.length == 0) {
      return absl::InvalidArgumentError("RemoteKeySource direct-write grant contains an empty window");
    }
    plan.local_regions.push_back(
        communicator::routing::LocalRegion{
            .addr = window.local_addr,
            .bytes = window.length,
            .dev_type = communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
            .dev_id = 0,
        });
  }

  const std::string authority_id = resolved_authority_id();
  for (const auto& op : ops) {
    if (op.bytes == 0) {
      continue;
    }
    if (op.src_offset >= options_.total_size) {
      return absl::OutOfRangeError("RemoteKeySource direct-write op starts beyond EOF");
    }
    if (op.bytes > options_.total_size - op.src_offset) {
      return absl::OutOfRangeError("RemoteKeySource direct-write op exceeds EOF");
    }

    uint64_t bytes_done = 0;
    uint64_t src_global_offset = op.src_offset;
    while (bytes_done < op.bytes) {
      const uint64_t current_va = op.dest_va_offset + bytes_done;
      size_t window_index = grant.windows.size();
      for (size_t index = 0; index < grant.windows.size(); ++index) {
        const auto& window = grant.windows[index];
        if (current_va >= window.va_offset && current_va < window.va_offset + window.length) {
          window_index = index;
          break;
        }
      }
      if (window_index >= grant.windows.size()) {
        return absl::InvalidArgumentError("DirectWriteGrant lacks window coverage for requested VA range");
      }

      const auto& window = grant.windows[window_index];
      const uint64_t local_region_offset = current_va - window.va_offset;
      const uint64_t remaining_window_bytes = window.length - local_region_offset;
      auto located_or = locate_key_segment(src_global_offset);
      if (!located_or.ok()) {
        return located_or.status();
      }

      const size_t key_index = located_or->key_index;
      const size_t key_offset = located_or->key_offset;
      const uint64_t remaining_key_bytes = options_.buffer_sizes[key_index] - key_offset;
      const uint64_t step =
          std::min<uint64_t>(op.bytes - bytes_done, std::min(remaining_window_bytes, remaining_key_bytes));
      if (step == 0) {
        return absl::InternalError("RemoteKeySource direct-write ReadPlan lowering produced an empty step");
      }

      const uint32_t source_slice_index = static_cast<uint32_t>(plan.source_slices.size());
      plan.source_slices.push_back(
          communicator::routing::SourceSlice{
              .authority_id = authority_id,
              .route = context.route_context,
              .tensor_key = options_.memory_keys[key_index],
              .remote_offset = static_cast<uint64_t>(key_offset),
              .bytes = step,
          });
      plan.slices.push_back(
          communicator::routing::ReadPlanSlice{
              .source_slice_index = source_slice_index,
              .local_region_index = static_cast<uint32_t>(window_index),
              .local_region_offset = local_region_offset,
              .bytes = step,
          });

      bytes_done += step;
      src_global_offset += step;
    }
  }

  const absl::Status plan_status = communicator::routing::validate_read_plan(plan);
  if (!plan_status.ok()) {
    return plan_status;
  }
  return plan;
}

absl::StatusOr<size_t> RemoteKeySource::execute_direct_write_batch_fallback(
    absl::Span<const DirectWriteOp> ops,
    const DirectWriteGrant& grant,
    const FrozenDirectWriteContext& context) {
  size_t total_bytes = 0;
  for (const auto& op : ops) {
    if (op.bytes == 0) {
      continue;
    }
    auto wrote_or = execute_direct_write_op(op, grant, context);
    if (!wrote_or.ok()) {
      return wrote_or.status();
    }
    if (*wrote_or != op.bytes) {
      return absl::DataLossError("RemoteKeySource direct-write batch short write");
    }
    total_bytes += *wrote_or;
  }
  return total_bytes;
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

absl::StatusOr<communicator::transport::read_result_t> RemoteKeySource::read_with_strict_fallback(
    const std::string& key,
    uint64_t local_addr,
    size_t bytes,
    uint64_t remote_offset,
    int dev_type,
    int dev_id) {
  auto attempt = begin_read_with_strict_fallback(
      options_, key, local_addr, static_cast<uint64_t>(bytes), remote_offset, dev_type, dev_id, &routed_path_enabled_);
  auto result_or = await_read_result(attempt.future, key, remote_offset, bytes);
  if (!attempt.used_routed) {
    return result_or;
  }
  if (result_or.ok() && result_or->status.ok()) {
    return result_or;
  }

  routed_path_enabled_ = false;
  const absl::Status routed_status = result_or.ok() ? result_or->status : result_or.status();
  log_routed_fallback(options_, routed_status);

  auto direct_future =
      read_direct(options_, key, local_addr, static_cast<uint64_t>(bytes), remote_offset, dev_type, dev_id);
  return await_read_result(direct_future, key, remote_offset, bytes);
}

absl::StatusOr<communicator::transport::read_result_t> RemoteKeySource::read_with_frozen_mode(
    const FrozenDirectWriteContext& context,
    const std::string& key,
    uint64_t local_addr,
    size_t bytes,
    uint64_t remote_offset,
    int dev_type,
    int dev_id) {
  if (context.mode == DirectWriteMode::kRouted) {
    auto routed_future_or = read_routed_with_communicator(
        context.routed_communicator, key, local_addr, static_cast<uint64_t>(bytes), remote_offset, dev_type, dev_id);
    if (!routed_future_or.ok()) {
      return routed_future_or.status();
    }
    return await_read_result(*routed_future_or, key, remote_offset, bytes);
  }

  auto direct_future =
      read_direct(options_, key, local_addr, static_cast<uint64_t>(bytes), remote_offset, dev_type, dev_id);
  return await_read_result(direct_future, key, remote_offset, bytes);
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

    auto result_or = read_with_strict_fallback(
        key,
        reinterpret_cast<uint64_t>(dst_ptr + bytes_read),
        to_read,
        static_cast<uint64_t>(current_key_offset_),
        communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
        -1);
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
  auto layout_status = validate_key_layout();
  if (!layout_status.ok()) {
    return layout_status;
  }

  size_t bytes_to_read = std::min(bytes, static_cast<size_t>(options_.total_size - offset));
  size_t bytes_read = 0;
  char* dst_ptr = static_cast<char*>(dst);

  // Determine starting key index and local offset within that key (stateless)
  auto located_or = locate_key_segment(offset);
  if (!located_or.ok()) {
    return located_or.status();
  }
  size_t key_index = located_or->key_index;
  size_t key_offset = located_or->key_offset;

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

    auto result_or = read_with_strict_fallback(
        key,
        reinterpret_cast<uint64_t>(dst_ptr + bytes_read),
        to_read,
        static_cast<uint64_t>(key_offset),
        communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
        -1);
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
  if (options_.comm_engine->is_rdma_enabled()) {
    return true;
  }
  if (!can_use_routed_read(options_, /*routed_path_enabled=*/true)) {
    return false;
  }
  auto communicator_or =
      options_.routing_context->get_communicator(options_.local_endpoint_id, options_.remote_endpoint_id);
  if (!communicator_or.ok()) {
    return false;
  }
  auto channel_or = communicator_or.value()->primary_channel();
  if (!channel_or.ok() || channel_or.value() == nullptr || channel_or.value()->hop_count() != 1 ||
      channel_or.value()->hops().empty() || channel_or.value()->hops().front() == nullptr) {
    return false;
  }
  return is_local_direct_protocol(channel_or.value()->hops().front()->protocol());
}

bool RemoteKeySource::supports_batched_direct_write_at() const {
  auto frozen_or = freeze_direct_write_context();
  if (!frozen_or.ok()) {
    return false;
  }
  return frozen_or->mode == DirectWriteMode::kRouted && frozen_or->routed_communicator != nullptr &&
      frozen_or->route_context.protocol == communicator::routing::ConnectionProtocol::kRdma &&
      !frozen_or->route_context.local_endpoint_id.empty() && !frozen_or->route_context.remote_endpoint_id.empty();
}

absl::StatusOr<size_t> RemoteKeySource::execute_direct_write_op(
    const DirectWriteOp& op,
    const DirectWriteGrant& grant,
    const FrozenDirectWriteContext& context) {
  if (op.src_offset >= options_.total_size || op.bytes == 0) {
    return static_cast<size_t>(0);
  }

  auto find_window = [&](uint64_t va_off) -> const DirectWriteGrant::Window* {
    for (const auto& win : grant.windows) {
      if (va_off >= win.va_offset && va_off < win.va_offset + win.length) {
        return &win;
      }
    }
    return nullptr;
  };

  size_t to_read_total = static_cast<size_t>(std::min<uint64_t>(op.bytes, options_.total_size - op.src_offset));
  size_t bytes_done = 0;
  uint64_t src_global_offset = op.src_offset;

  while (bytes_done < to_read_total) {
    const uint64_t cur_va = op.dest_va_offset + bytes_done;
    const auto* win = find_window(cur_va);
    if (win == nullptr) {
      return absl::InvalidArgumentError("DirectWriteGrant lacks window for requested VA range");
    }
    const uint64_t seg_off_in_bytes = cur_va - win->va_offset;
    const uint64_t local_addr = win->local_addr + seg_off_in_bytes;
    const uint64_t seg_bytes_left = win->length - seg_off_in_bytes;

    auto located_or = locate_key_segment(src_global_offset);
    if (!located_or.ok()) {
      return located_or.status();
    }
    const size_t key_index = located_or->key_index;
    const size_t key_offset = located_or->key_offset;

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

    auto result_or = read_with_frozen_mode(
        context,
        key,
        local_addr,
        step,
        static_cast<uint64_t>(key_offset),
        communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
        -1);
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

absl::StatusOr<size_t> RemoteKeySource::read_into_at(
    uint64_t src_offset,
    uint64_t dest_va_offset,
    size_t bytes,
    const DirectWriteGrant& grant) {
  auto frozen_or = freeze_direct_write_context();
  if (!frozen_or.ok()) {
    return frozen_or.status();
  }
  return execute_direct_write_op(
      DirectWriteOp{
          .src_offset = src_offset,
          .dest_va_offset = dest_va_offset,
          .bytes = static_cast<uint64_t>(bytes),
      },
      grant,
      *frozen_or);
}

absl::StatusOr<size_t> RemoteKeySource::readv_into_at(
    absl::Span<const DirectWriteOp> ops,
    const DirectWriteGrant& grant) {
  if (ops.empty()) {
    return static_cast<size_t>(0);
  }

  auto frozen_or = freeze_direct_write_context();
  if (!frozen_or.ok()) {
    return frozen_or.status();
  }

  if (frozen_or->route_context.protocol != communicator::routing::ConnectionProtocol::kRdma ||
      frozen_or->route_context.local_endpoint_id.empty() || frozen_or->route_context.remote_endpoint_id.empty()) {
    LOG(INFO) << "RemoteKeySource batched direct-write falling back to per-op scalar direct-write"
              << " authority=" << (options_.authority_id.empty() ? "<unset>" : options_.authority_id)
              << " artifact=" << (options_.artifact_id.empty() ? "<unset>" : options_.artifact_id)
              << " route=" << options_.local_endpoint_id << "->" << options_.remote_endpoint_id
              << " mode=" << (frozen_or->mode == DirectWriteMode::kRouted ? "routed" : "direct")
              << " protocol=" << communicator::routing::to_string(frozen_or->route_context.protocol)
              << " routed_communicator=" << (frozen_or->routed_communicator != nullptr) << " op_count=" << ops.size();
    return execute_direct_write_batch_fallback(ops, grant, *frozen_or);
  }

  auto plan_or = build_routed_read_plan(ops, grant, *frozen_or);
  if (!plan_or.ok()) {
    return plan_or.status();
  }
  if (plan_or->slices.empty()) {
    return static_cast<size_t>(0);
  }

  size_t total_bytes = 0;
  for (const auto& slice : plan_or->slices) {
    total_bytes += slice.bytes;
  }

  VLOG(2) << "RemoteKeySource direct-write ReadPlan issue"
          << " authority=" << (options_.authority_id.empty() ? "<unset>" : options_.authority_id)
          << " artifact=" << (options_.artifact_id.empty() ? "<unset>" : options_.artifact_id)
          << " route=" << frozen_or->route_context.local_endpoint_id << "->"
          << frozen_or->route_context.remote_endpoint_id
          << " mode=" << (frozen_or->mode == DirectWriteMode::kRouted ? "routed" : "direct")
          << " protocol=" << communicator::routing::to_string(frozen_or->route_context.protocol)
          << " op_count=" << ops.size() << " source_slices=" << plan_or->source_slices.size()
          << " read_slices=" << plan_or->slices.size() << " local_regions=" << plan_or->local_regions.size()
          << " bytes=" << total_bytes;

  absl::StatusOr<communicator::transport::future_read_result_t> future_or =
      frozen_or->mode == DirectWriteMode::kRouted && frozen_or->routed_communicator != nullptr
      ? submit_routed_read_plan(frozen_or->routed_communicator, *plan_or)
      : absl::StatusOr<communicator::transport::future_read_result_t>{
            options_.comm_engine->read_plan(*plan_or, options_.ip, options_.port)};
  if (!future_or.ok()) {
    if (is_pre_issue_direct_write_capability_miss(future_or.status())) {
      log_read_plan_direct_write_fallback(options_, frozen_or->route_context.protocol, future_or.status());
      return execute_direct_write_batch_fallback(ops, grant, *frozen_or);
    }
    return future_or.status();
  }

  communicator::transport::future_read_result_t future = std::move(*future_or);
  const std::future_status initial_status = future.wait_for(std::chrono::milliseconds(0));
  if (initial_status == std::future_status::ready) {
    auto result = future.get();
    if (is_pre_issue_direct_write_capability_miss(result.status)) {
      log_read_plan_direct_write_fallback(options_, frozen_or->route_context.protocol, result.status);
      return execute_direct_write_batch_fallback(ops, grant, *frozen_or);
    }
    if (!result.status.ok()) {
      return result.status;
    }
    return total_bytes;
  }

  auto result_or = await_read_result(
      future, plan_or->source_slices.front().tensor_key, plan_or->source_slices.front().remote_offset, total_bytes);
  if (!result_or.ok()) {
    return result_or.status();
  }
  if (!result_or->status.ok()) {
    return result_or->status;
  }
  return total_bytes;
}

} // namespace tensorcast::store::loader
