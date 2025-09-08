// Copyright (c) 2025, TensorCast Team.

// BackgroundScheduler: a simple condition-variable driven scheduler for daemon sweepers

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace tensorcast::daemon {

enum class TaskKind : std::uint8_t { kSessionTTL, kLockTTL, kPidWatch, kVerification, kEviction };

class BackgroundScheduler {
 public:
  using Clock = std::chrono::steady_clock;
  using TaskFn = std::function<void()>;

  struct TaskSpec {
    TaskKind kind;
    std::chrono::milliseconds interval;
    TaskFn fn;
  };

  BackgroundScheduler() = default;
  ~BackgroundScheduler() {
    stop();
  }

  void add_task(TaskKind kind, std::chrono::milliseconds interval, TaskFn fn) {
    std::lock_guard<std::mutex> g(mu_);
    Task t;
    t.kind = kind;
    t.interval = interval;
    t.fn = std::move(fn);
    t.next_due = Clock::now() + interval;
    tasks_.push_back(std::move(t));
  }

  void start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true))
      return;
    th_ = std::thread([this]() { this->run_loop(); });
  }

  void stop() {
    if (!running_.load())
      return;
    running_.store(false);
    {
      std::lock_guard<std::mutex> g(mu_);
      // no-op
    }
    cv_.notify_all();
    if (th_.joinable())
      th_.join();
  }

  void notify(TaskKind kind) {
    std::lock_guard<std::mutex> g(mu_);
    for (auto& t : tasks_) {
      if (t.kind == kind)
        t.pending = true;
    }
    cv_.notify_one();
  }

 private:
  struct Task {
    TaskKind kind;
    std::chrono::milliseconds interval{0};
    TaskFn fn;
    Clock::time_point next_due{Clock::now()};
    bool pending{false};
  };

  void run_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    while (running_.load()) {
      auto now = Clock::now();
      Clock::time_point next_wake = now + std::chrono::hours(24);
      // Determine which tasks to run
      std::vector<size_t> to_run;
      to_run.reserve(tasks_.size());
      for (size_t i = 0; i < tasks_.size(); ++i) {
        auto& t = tasks_[i];
        if (t.pending || now >= t.next_due) {
          to_run.push_back(i);
          // schedule next run
          t.pending = false;
          t.next_due = now + t.interval;
        }
        if (t.next_due < next_wake)
          next_wake = t.next_due;
      }

      // Run outside the lock
      lk.unlock();
      for (size_t idx : to_run) {
        // Guard against exceptions to keep the loop alive
        try {
          tasks_[idx].fn();
        } catch (...) {
          // swallow
        }
      }
      lk.lock();

      // Wait until the next due time or a notification
      if (!running_.load())
        break;
      now = Clock::now();
      if (next_wake > now) {
        cv_.wait_until(lk, next_wake, [this]() { return !running_.load(); });
      }
    }
  }

  std::atomic<bool> running_{false};
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<Task> tasks_;
  std::thread th_;
};

} // namespace tensorcast::daemon
