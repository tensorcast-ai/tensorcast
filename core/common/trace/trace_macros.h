// Copyright (c) 2025, TensorCast Team.

// All rights reserved.
#pragma once

#include <memory>

// NOLINTBEGIN(unused-includes)
#include "core/common/otel/trace_scope_bridge.h"
#include "core/common/trace/trace_manager.h"
#include "core/common/trace/trace_request_data.h"
#include "core/common/trace/trace_scope.h"
#include "folly/io/async/Request.h"
// NOLINTEND(unused-includes)

// ---------------------------------------------------------------------------
// 1. Low-level helpers (not intended to be used directly)
// ---------------------------------------------------------------------------

#define _SC_TRACE_SCOPE_IMPL_EX(artifact_id, request_id, stage)                          \
  common::trace::TraceManager::RequestIdGuard _trace_req_guard_##__LINE__(request_id);   \
  common::trace::TraceManager::ArtifactIdGuard _trace_mid_guard_##__LINE__(artifact_id); \
  common::trace::TraceScope _trace_scope_##__LINE__{artifact_id, stage};                 \
  common::otel::TraceScopeBridge _otel_trace_scope_bridge_##__LINE__ {                   \
    artifact_id, stage                                                                   \
  }

#define _SC_TRACE_SCOPE_IMPL_AUTO(stage)                                                                        \
  common::trace::TraceManager::RequestIdGuard _trace_req_guard_##__LINE__(                                      \
      common::trace::TraceManager::current_request_id());                                                       \
  common::trace::TraceManager::ArtifactIdGuard _trace_mid_guard_##__LINE__(                                     \
      common::trace::TraceManager::current_artifact_id());                                                      \
  common::trace::TraceScope _trace_scope_##__LINE__{common::trace::TraceManager::current_artifact_id(), stage}; \
  common::otel::TraceScopeBridge _otel_trace_scope_bridge_##__LINE__ {                                          \
    common::trace::TraceManager::current_artifact_id(), stage                                                   \
  }

#define _SC_TRACE_REQUEST_CONTEXT_IMPL(request_id, artifact_id)                                                        \
  auto _trace_ids_##__LINE__ = std::make_shared<common::trace::TraceIds>(request_id, artifact_id);                     \
  folly::ShallowCopyRequestContextScopeGuard _trace_reqctx_guard_##__LINE__(                                           \
      folly::RequestDataItem{                                                                                          \
          common::trace::kTraceIdsToken, std::make_unique<common::trace::TraceIdsRequestData>(_trace_ids_##__LINE__)}, \
      folly::RequestDataItem{                                                                                          \
          common::trace::kTraceRequestDataToken,                                                                       \
          std::make_unique<common::trace::TraceRequestData>(_trace_ids_##__LINE__)}) /**/

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
#define SC_TRACE_INIT_GUARD(request_id, artifact_id, stage)                              \
  _SC_TRACE_REQUEST_CONTEXT_IMPL(request_id, artifact_id);                               \
  common::trace::TraceManager::RequestIdGuard _trace_req_guard_##__LINE__(request_id);   \
  common::trace::TraceManager::ArtifactIdGuard _trace_mid_guard_##__LINE__(artifact_id); \
  common::trace::TraceSummaryGuard _trace_summary_guard_##__LINE__(artifact_id);         \
  common::trace::TraceScope _trace_scope_##__LINE__{artifact_id, stage};                 \
  common::otel::TraceScopeBridge _otel_trace_scope_bridge_##__LINE__ {                   \
    artifact_id, stage                                                                   \
  }
