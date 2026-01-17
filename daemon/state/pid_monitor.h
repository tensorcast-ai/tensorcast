// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <sys/types.h>

namespace tensorcast::daemon {

// Lightweight PID liveness monitor.
// Uses pidfd + epoll when available; falls back to periodic /proc polling otherwise.
class PidMonitor final {
 public:
  using ExitCallback = std::function<void(pid_t)>;

  // poll_interval controls the fallback polling cadence when pidfd is
  // unavailable, and also the epoll wait timeout when pidfd is available.
  explicit PidMonitor(ExitCallback cb, std::chrono::milliseconds poll_interval = std::chrono::milliseconds(1000));
  ~PidMonitor();

  void start();
  void stop();

  // Start watching a PID; safe to call repeatedly.
  void watch(pid_t pid);

  // Stop watching a PID. Best-effort: removes any pidfd from epoll and closes it.
  void unwatch(pid_t pid);

  bool using_pidfd() const;

  bool is_watching_for_test(pid_t pid);

 private:
  void run_loop_();
  void run_epoll_();
  void run_poll_();
  void poll_missing_pidfds_once_();

  static constexpr uint64_t kEpollTagMask = 0xFFFF000000000000ULL;
  static constexpr uint64_t kEpollTagWake = 0xFFFF000000000000ULL;
  static constexpr uint64_t kEpollTagPid = 0xEEEE000000000000ULL;
  static constexpr uint64_t kEpollValueMask = 0x0000FFFFFFFFFFFFULL;

  ExitCallback cb_;
  std::chrono::milliseconds poll_interval_;
  std::atomic<bool> running_{false};
  std::atomic<bool> use_pidfd_{false};
  std::thread th_;
  std::mutex mu_;
  std::unordered_map<pid_t, int> pidfds_;
  std::unordered_set<pid_t> watched_;
  int epoll_fd_{-1};
  int event_fd_{-1};
};

} // namespace tensorcast::daemon
