// Copyright (c) 2025, TensorCast Team.

// All rights reserved.
#pragma once

// NOLINTBEGIN(unused-includes)
#include <future>
#include "core/common/otel/trace_scope_bridge.h"
#include "core/common/trace/trace_ctx.h"
#include "core/common/trace/trace_manager.h"
#include "core/common/trace/trace_scope.h"
//
// NOLINTEND(unused-includes)

// ---------------------------------------------------------------------------
// 1. Low-level helpers (not intended to be used directly)
// ---------------------------------------------------------------------------

#define _SC_TRACE_SCOPE_IMPL_EX(artifact_id, request_id, stage)                                \
  ::tensorcast::store::TraceManager::RequestIdGuard _trace_req_guard_##__LINE__(request_id);   \
  ::tensorcast::store::TraceManager::ArtifactIdGuard _trace_mid_guard_##__LINE__(artifact_id); \
  ::tensorcast::store::TraceScope _trace_scope_##__LINE__{artifact_id, stage};                 \
  ::tensorcast::obs::TraceScopeBridge _otel_trace_scope_bridge_##__LINE__ {                    \
    artifact_id, stage                                                                         \
  }

#define _SC_TRACE_SCOPE_IMPL_AUTO(stage)                                          \
  ::tensorcast::store::TraceManager::RequestIdGuard _trace_req_guard_##__LINE__(  \
      ::tensorcast::store::TraceManager::current_request_id());                   \
  ::tensorcast::store::TraceManager::ArtifactIdGuard _trace_mid_guard_##__LINE__( \
      ::tensorcast::store::TraceManager::current_artifact_id());                  \
  ::tensorcast::store::TraceScope _trace_scope_##__LINE__{                        \
      ::tensorcast::store::TraceManager::current_artifact_id(), stage};           \
  ::tensorcast::obs::TraceScopeBridge _otel_trace_scope_bridge_##__LINE__ {       \
    ::tensorcast::store::TraceManager::current_artifact_id(), stage               \
  }

// ---------------------------------------------------------------------------
// 2. Public macros
// ---------------------------------------------------------------------------

// Simplest form: only stage name is required – artifact_id and request_id are
// inferred from thread-local context that should be set via the guard macros.
#define SC_TRACE_SCOPE(stage) _SC_TRACE_SCOPE_IMPL_AUTO(stage)

// ---------------------------------------------------------------------------
// 3. Convenience guard macros
// ---------------------------------------------------------------------------

// Set both request_id and artifact_id along with summary guard and initial trace scope.
// This should only be used in load_model_from_remote/disk entry points.
#define SC_TRACE_INIT_GUARD(request_id, artifact_id, stage)                                    \
  ::tensorcast::store::TraceManager::RequestIdGuard _trace_req_guard_##__LINE__(request_id);   \
  ::tensorcast::store::TraceManager::ArtifactIdGuard _trace_mid_guard_##__LINE__(artifact_id); \
  ::tensorcast::store::TraceSummaryGuard _trace_summary_guard_##__LINE__(artifact_id);         \
  ::tensorcast::store::TraceScope _trace_scope_##__LINE__{artifact_id, stage};                 \
  ::tensorcast::obs::TraceScopeBridge _otel_trace_scope_bridge_##__LINE__ {                    \
    artifact_id, stage                                                                         \
  }

// Spawn std::async task that automatically propagates current request-id / replica-id.
// Users must #include "core/common/trace/trace_ctx.h" when using this macro.
#define SC_TRACE_ASYNC(policy, ...) std::async(policy, ::tensorcast::store::with_trace_ctx(__VA_ARGS__))
