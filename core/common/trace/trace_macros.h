// Copyright (c) 2025, StepCast Team. All rights reserved.

// All rights reserved.
#pragma once

// NOLINTBEGIN(unused-includes)
#include <future>
#include "core/common/trace/trace_ctx.h"
#include "core/common/trace/trace_manager.h"
#include "core/common/trace/trace_scope.h"
// NOLINTEND(unused-includes)

// ---------------------------------------------------------------------------
// 1. Low-level helpers (not intended to be used directly)
// ---------------------------------------------------------------------------

#define _SC_TRACE_SCOPE_IMPL_EX(model_id, request_id, stage)                               \
  ::stepcast::store::TraceManager::RequestIdGuard _trace_req_guard_##__LINE__(request_id); \
  ::stepcast::store::TraceManager::ModelIdGuard _trace_mid_guard_##__LINE__(model_id);     \
  ::stepcast::store::TraceScope _trace_scope_##__LINE__ {                                  \
    model_id, stage                                                                        \
  }

#define _SC_TRACE_SCOPE_IMPL_AUTO(stage)                                       \
  ::stepcast::store::TraceManager::RequestIdGuard _trace_req_guard_##__LINE__( \
      ::stepcast::store::TraceManager::current_request_id());                  \
  ::stepcast::store::TraceManager::ModelIdGuard _trace_mid_guard_##__LINE__(   \
      ::stepcast::store::TraceManager::current_model_id());                    \
  ::stepcast::store::TraceScope _trace_scope_##__LINE__ {                      \
    ::stepcast::store::TraceManager::current_model_id(), stage                 \
  }

// ---------------------------------------------------------------------------
// 2. Public macros
// ---------------------------------------------------------------------------

// Simplest form: only stage name is required – model_id and request_id are
// inferred from thread-local context that should be set via the guard macros.
#define SC_TRACE_SCOPE(stage) _SC_TRACE_SCOPE_IMPL_AUTO(stage)

// Extended form when caller needs to explicitly override request_id and/or
// model_id (e.g. in cross-process scenarios). Kept for backward-compat.
#define SC_TRACE_SCOPE_EX(model_id, request_id, stage) _SC_TRACE_SCOPE_IMPL_EX(model_id, request_id, stage)

// ---------------------------------------------------------------------------
// 3. Convenience guard macros
// ---------------------------------------------------------------------------

// Set both request_id and model_id along with summary guard and initial trace scope.
// This should only be used in load_model_from_remote/disk entry points.
#define SC_TRACE_INIT_GUARD(request_id, model_id, stage)                                   \
  ::stepcast::store::TraceManager::RequestIdGuard _trace_req_guard_##__LINE__(request_id); \
  ::stepcast::store::TraceManager::ModelIdGuard _trace_mid_guard_##__LINE__(model_id);     \
  ::stepcast::store::TraceSummaryGuard _trace_summary_guard_##__LINE__(model_id);          \
  ::stepcast::store::TraceScope _trace_scope_##__LINE__ {                                  \
    model_id, stage                                                                        \
  }

// Spawn std::async task that automatically propagates current request-id / model-id.
// Users must #include "core/common/trace/trace_ctx.h" when using this macro.
#define SC_TRACE_ASYNC(policy, ...) std::async(policy, ::stepcast::store::with_trace_ctx(__VA_ARGS__))