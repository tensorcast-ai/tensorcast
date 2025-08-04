// Copyright (c) 2025, StepCast Team. All rights reserved.

// All rights reserved.

#include "core/common/trace/trace_scope.h"

#include "core/common/trace/trace_manager.h"

namespace stepcast::store {

TraceScope::TraceScope(const std::string& model, const std::string& stage)
    : model_(model), request_id_(TraceManager::current_request_id()) {
  id_ = TraceManager::instance().begin_span(model_, request_id_, stage);
}

TraceScope::TraceScope(TraceScope&& other) noexcept
    : model_(std::move(other.model_)), request_id_(std::move(other.request_id_)), id_(other.id_) {
  other.id_ = kInvalidSpan;
}

TraceScope& TraceScope::operator=(TraceScope&& other) noexcept {
  if (this != &other) {
    Finish();
    model_ = std::move(other.model_);
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
    TraceManager::instance().end_span(model_, request_id_, id_);
    id_ = kInvalidSpan;
  }
}

} // namespace stepcast::store