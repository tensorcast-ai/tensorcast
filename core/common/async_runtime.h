// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "folly/Executor.h"
#include "folly/futures/Future.h"

namespace tensorcast::common {

// AsyncRuntime: injectable, drainable async runtime for C++ components.
// Owns executors used for CPU-bound work, blocking work, and serial state
// progression. All executors capture the current folly::RequestContext at
// submission time and restore it on execution to preserve observability
// context across threads.
class AsyncRuntime {
 public:
  struct Options {
    size_t cpu_threads{0};
    size_t blocking_threads{0};
    std::string thread_name_prefix{"tensorcast"};
  };

  AsyncRuntime();
  explicit AsyncRuntime(Options opts);
  ~AsyncRuntime();

  AsyncRuntime(const AsyncRuntime&) = delete;
  AsyncRuntime& operator=(const AsyncRuntime&) = delete;
  AsyncRuntime(AsyncRuntime&&) = delete;
  AsyncRuntime& operator=(AsyncRuntime&&) = delete;

  [[nodiscard]] folly::Executor::KeepAlive<> cpu_executor() const;
  [[nodiscard]] folly::Executor::KeepAlive<> blocking_executor() const;
  [[nodiscard]] folly::Executor::KeepAlive<> serial_executor() const;

  [[nodiscard]] folly::Timekeeper& timekeeper() const;

  void shutdown();
  [[nodiscard]] absl::Status drain(absl::Time deadline);

  [[nodiscard]] bool is_shutting_down() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace tensorcast::common
