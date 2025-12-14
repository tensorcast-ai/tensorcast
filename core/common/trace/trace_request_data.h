// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "folly/io/async/Request.h"

#include "core/common/trace/trace_manager.h"

namespace tensorcast::common::trace {

inline const folly::RequestToken kTraceIdsToken{"tc.trace_ids"};
inline const folly::RequestToken kTraceRequestDataToken{"tc.trace.request_data"};
inline const folly::RequestToken kRpcMethodToken{"rpc.method"};

class TraceIds {
 public:
  TraceIds() = default;

  TraceIds(std::string request_id, std::string artifact_id)
      : request_id_(std::move(request_id)), artifact_id_(std::move(artifact_id)) {}

  void set_request_id(std::string request_id) {
    absl::MutexLock lock(&mu_);
    request_id_ = std::move(request_id);
  }

  void set_artifact_id(std::string artifact_id) {
    absl::MutexLock lock(&mu_);
    artifact_id_ = std::move(artifact_id);
  }

  [[nodiscard]] std::string request_id() const {
    absl::MutexLock lock(&mu_);
    return request_id_;
  }

  [[nodiscard]] std::string artifact_id() const {
    absl::MutexLock lock(&mu_);
    return artifact_id_;
  }

 private:
  mutable absl::Mutex mu_;
  std::string request_id_ ABSL_GUARDED_BY(mu_);
  std::string artifact_id_ ABSL_GUARDED_BY(mu_);
};

class TraceIdsRequestData final : public folly::RequestData {
 public:
  explicit TraceIdsRequestData(std::shared_ptr<TraceIds> ids) : ids_(std::move(ids)) {}

  bool hasCallback() override {
    return false;
  }

  [[nodiscard]] const std::shared_ptr<TraceIds>& ids() const {
    return ids_;
  }

 private:
  std::shared_ptr<TraceIds> ids_;
};

class TraceRequestData final : public folly::RequestData {
 public:
  explicit TraceRequestData(std::shared_ptr<TraceIds> ids) : ids_(std::move(ids)) {}

  bool hasCallback() override {
    return true;
  }

  void onSet() override {
    tls_stack().push_back(
        PreviousIds{
            .request_id = TraceManager::current_request_id(),
            .artifact_id = TraceManager::current_artifact_id(),
        });
    if (!ids_) {
      TraceManager::set_current_request_id("");
      TraceManager::set_current_artifact_id("");
      return;
    }
    TraceManager::set_current_request_id(ids_->request_id());
    TraceManager::set_current_artifact_id(ids_->artifact_id());
  }

  void onUnset() override {
    auto& stack = tls_stack();
    if (stack.empty()) {
      TraceManager::set_current_request_id("");
      TraceManager::set_current_artifact_id("");
      return;
    }
    PreviousIds prev = std::move(stack.back());
    stack.pop_back();
    TraceManager::set_current_request_id(prev.request_id);
    TraceManager::set_current_artifact_id(prev.artifact_id);
  }

 private:
  struct PreviousIds {
    std::string request_id;
    std::string artifact_id;
  };

  static std::vector<PreviousIds>& tls_stack() {
    static thread_local std::vector<PreviousIds> stack;
    return stack;
  }

  std::shared_ptr<TraceIds> ids_;
};

[[nodiscard]] inline std::shared_ptr<TraceIds> get_trace_ids_from_request_context() {
  folly::RequestContext* ctx = folly::RequestContext::try_get();
  if (ctx == nullptr) {
    return nullptr;
  }
  auto* data = ctx->getContextData(kTraceIdsToken);
  auto* typed = dynamic_cast<TraceIdsRequestData*>(data);
  if (typed == nullptr) {
    return nullptr;
  }
  return typed->ids();
}

} // namespace tensorcast::common::trace
