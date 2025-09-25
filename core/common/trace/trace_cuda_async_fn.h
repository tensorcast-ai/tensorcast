// Copyright (c) 2025, TensorCast Team.

// All rights reserved.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "absl/status/status.h"
#include "core/common/cuda_api.h"
#include "core/common/trace/trace_manager.h"

namespace tensorcast::common::trace {
namespace detail {

// Internal payload passed to the CUDA host callback.
struct CudaTracePayload {
  std::string artifact_id;
  std::string request_id;
  TraceManager::SpanId span_id;
  std::function<void(absl::Status)> on_complete;
  cudaStream_t stream{nullptr};
  int device_id{-1};
};

inline void sc_schedule_trace_host_cb(cudaStream_t stream, CudaTracePayload* payload) {
  payload->stream = stream;
  auto status = tensorcast::cuda::launch_host_func(
      stream,
      [](void* user_data) {
        auto* p = static_cast<CudaTracePayload*>(user_data);
        TraceManager::instance().end_span(p->artifact_id, p->request_id, p->span_id);
        if (p->on_complete) {
          p->on_complete(absl::OkStatus());
        }
        delete p;
      },
      payload);
  if (!status.ok()) {
    // Fallback: synchronize the stream on a detached thread before completing.
    std::unique_ptr<CudaTracePayload> holder(payload);
    std::thread(
        [](std::unique_ptr<CudaTracePayload> payload_up) {
          if (!payload_up) {
            return;
          }
          auto* p = payload_up.get();
          if (p->device_id >= 0) {
            (void)tensorcast::cuda::set_device(p->device_id);
          }
          absl::Status sync_status = tensorcast::cuda::stream_synchronize(p->stream);
          TraceManager::instance().end_span(p->artifact_id, p->request_id, p->span_id);
          if (p->on_complete) {
            p->on_complete(sync_status);
          }
        },
        std::move(holder))
        .detach();
  }
}

} // namespace detail

// ---------------------------------------------------------------------------
// trace_cuda_async
// ---------------------------------------------------------------------------
// Helper that executes a CUDA asynchronous operation (op) on the given stream,
// starts a Trace span (stage) when called, and automatically ends the span once
// the operation and all prior work in the stream completes.  An optional
// on_complete callback is run inside the same host callback right after the
// span ends, typically for resource cleanup.
//
// Usage:
//   SC_RETURN_IF_ERROR(trace_cuda_async("h2d_copy", stream,
//       [&]{ return tensorcast::cuda::memcpy_async(dst, src, n, cudaMemcpyHostToDevice, stream); },
//       [&](absl::Status st){ pool->return_slot(slot); }));
// ---------------------------------------------------------------------------
template <typename Op, typename Done = std::function<void(absl::Status)>>
inline absl::Status trace_cuda_async(
    const std::string& stage,
    cudaStream_t stream,
    Op&& op,
    Done&& on_complete = Done{}) {
  const std::string& artifact_id = TraceManager::current_artifact_id();
  const std::string& request_id = TraceManager::current_request_id();
  // Pass the CUDA stream to begin_span for better Chrome Trace visualization
  auto span_id = TraceManager::instance().begin_span(artifact_id, request_id, stage, static_cast<void*>(stream));

  // Execute the user-supplied CUDA operation.
  absl::Status status = std::forward<Op>(op)();
  if (!status.ok()) {
    TraceManager::instance().end_span(artifact_id, request_id, span_id);
    return status;
  }

  // Schedule host callback to end the span when the stream finishes work.
  auto* payload = new detail::CudaTracePayload{artifact_id, request_id, span_id, std::forward<Done>(on_complete)};
  int current_device = -1;
  if (auto dev_status = tensorcast::cuda::get_device(&current_device); dev_status.ok()) {
    payload->device_id = current_device;
  }
  detail::sc_schedule_trace_host_cb(stream, payload);

  return absl::OkStatus();
}

} // namespace tensorcast::common::trace
