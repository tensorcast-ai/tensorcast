// Copyright (c) 2025, StepCast Team. All rights reserved.

// All rights reserved.
#pragma once

#include <functional>
#include <string>
#include <utility>

#include "core/common/trace/trace_manager.h"

namespace stepcast::store {

// ---------------------------------------------------------------------------
// Helper for propagating request-id & model-id across std::async / thread pools
// ---------------------------------------------------------------------------

// Wrap a callable so that it automatically restores the current request_id
// and model_id in the new thread before executing the underlying functor.
//
// Example:
//   auto fut = std::async(std::launch::async,
//       with_trace_ctx([&]{ ... }));
//
// Usually users call it via the SC_TRACE_ASYNC macro defined in trace_macros.h.
template <typename Fn>
auto with_trace_ctx(Fn&& fn) {
  std::string rid = TraceManager::current_request_id();
  std::string mid = TraceManager::current_model_id();
  return [rid = std::move(rid), mid = std::move(mid), fn = std::forward<Fn>(fn)](
             auto&&... args) mutable -> decltype(auto) {
    TraceManager::RequestIdGuard _rid_guard(rid);
    TraceManager::ModelIdGuard _mid_guard(mid);
    return std::invoke(std::forward<decltype(fn)>(fn), std::forward<decltype(args)>(args)...);
  };
}

} // namespace stepcast::store