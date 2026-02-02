// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <atomic>

namespace tensorcast::daemon {

class ShutdownSignal {
 public:
  // Returns true if this call initiated shutdown.
  bool begin_shutdown() {
    bool expected = false;
    return shutting_down_.compare_exchange_strong(expected, true);
  }

  bool is_shutting_down() const {
    return shutting_down_.load();
  }

 private:
  std::atomic<bool> shutting_down_{false};
};

} // namespace tensorcast::daemon
