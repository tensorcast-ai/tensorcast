// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/app/startup_coordinator.h"

#include <utility>

#include "absl/strings/str_cat.h"

namespace tensorcast::daemon {

void StartupCoordinator::begin_startup(std::string message) {
  absl::MutexLock lock(&mu_);
  state_ = State::kStarting;
  startup_message_ = std::move(message);
  failure_status_ = absl::OkStatus();
}

void StartupCoordinator::mark_ready() {
  absl::MutexLock lock(&mu_);
  state_ = State::kReady;
  startup_message_.clear();
  failure_status_ = absl::OkStatus();
}

void StartupCoordinator::mark_failed(absl::Status status) {
  absl::MutexLock lock(&mu_);
  state_ = State::kFailed;
  failure_status_ = std::move(status);
}

bool StartupCoordinator::is_ready() const {
  absl::MutexLock lock(&mu_);
  return state_ == State::kReady;
}

StartupCoordinator::Phase StartupCoordinator::current_phase() const {
  absl::MutexLock lock(&mu_);
  switch (state_) {
    case State::kReady:
      return Phase::kReady;
    case State::kStarting:
      return Phase::kListening;
    case State::kFailed:
      return Phase::kFailed;
  }
  return Phase::kFailed;
}

absl::Status StartupCoordinator::startup_barrier_status() const {
  absl::MutexLock lock(&mu_);
  switch (state_) {
    case State::kReady:
      return absl::OkStatus();
    case State::kStarting:
      return absl::UnavailableError(startup_message_.empty() ? "daemon startup still in progress" : startup_message_);
    case State::kFailed:
      if (failure_status_.ok()) {
        return absl::InternalError("daemon startup failed");
      }
      return failure_status_;
  }
  return absl::InternalError("unknown daemon startup state");
}

} // namespace tensorcast::daemon
