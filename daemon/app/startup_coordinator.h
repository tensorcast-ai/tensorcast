// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

namespace tensorcast::daemon {

class StartupCoordinator {
 public:
  enum class Phase {
    kListening,
    kReady,
    kFailed,
  };

  StartupCoordinator() = default;

  void begin_startup(std::string message);
  void mark_ready();
  void mark_failed(absl::Status status);

  [[nodiscard]] bool is_ready() const;
  [[nodiscard]] Phase current_phase() const;
  [[nodiscard]] absl::Status startup_barrier_status() const;

 private:
  enum class State {
    kReady,
    kStarting,
    kFailed,
  };

  mutable absl::Mutex mu_;
  State state_ ABSL_GUARDED_BY(mu_) = State::kReady;
  std::string startup_message_ ABSL_GUARDED_BY(mu_);
  absl::Status failure_status_ ABSL_GUARDED_BY(mu_) = absl::OkStatus();
};

} // namespace tensorcast::daemon
