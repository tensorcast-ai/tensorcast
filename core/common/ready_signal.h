// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <utility>

#include "folly/futures/SharedPromise.h"

namespace tensorcast::common {

// ReadySignal<T> is a producer-owned completion signal that can be subscribed
// to multiple times. It is the foundation for completion-driven state
// progression (fan-out) without polling.
template <typename T>
class ReadySignal {
 public:
  ReadySignal() = default;

  [[nodiscard]] folly::SemiFuture<T> subscribe() const {
    return promise_.getSemiFuture();
  }

  [[nodiscard]] bool is_ready() const {
    return promise_.isFulfilled();
  }

  void set_value(T value) {
    promise_.setValue(std::move(value));
  }

  void set_exception(folly::exception_wrapper ew) {
    promise_.setException(std::move(ew));
  }

 private:
  mutable folly::SharedPromise<T> promise_;
};

} // namespace tensorcast::common
