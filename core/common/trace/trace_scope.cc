// Copyright (c) 2025, TensorCast Team.

// All rights reserved.

#include "core/common/trace/trace_scope.h"

#include "core/common/trace/trace_manager.h"

namespace tensorcast::store {

TraceScope::TraceScope(const std::string& replica, const std::string& stage)
    : artifact_id_(replica), request_id_(TraceManager::current_request_id()) {
  id_ = TraceManager::instance().begin_span(artifact_id_, request_id_, stage);
}

TraceScope::TraceScope(TraceScope&& other) noexcept
    : artifact_id_(std::move(other.artifact_id_)), request_id_(std::move(other.request_id_)), id_(other.id_) {
  other.id_ = kInvalidSpan;
}

TraceScope& TraceScope::operator=(TraceScope&& other) noexcept {
  if (this != &other) {
    Finish();
    artifact_id_ = std::move(other.artifact_id_);
    request_id_ = std::move(other.request_id_);
    id_ = other.id_;
    other.id_ = kInvalidSpan;
  }
  return *this;
}

TraceScope::~TraceScope() {
  Finish();
}

void TraceScope::Finish() {
  if (id_ != kInvalidSpan) {
    TraceManager::instance().end_span(artifact_id_, request_id_, id_);
    id_ = kInvalidSpan;
  }
}

} // namespace tensorcast::store