// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/pid_monitor.h"

#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <utility>

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "daemon/common/safe_sys.h"

namespace tensorcast::daemon {

PidMonitor::PidMonitor(ExitCallback cb, std::chrono::milliseconds poll_interval)
    : cb_(std::move(cb)), poll_interval_(poll_interval) {}

PidMonitor::~PidMonitor() {
  stop();
}

void PidMonitor::start() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true))
    return;
  // Try to create epoll and eventfd; if either fails, use polling fallback
  epoll_fd_ = ::epoll_create1(EPOLL_CLOEXEC);
  event_fd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (epoll_fd_ >= 0 && event_fd_ >= 0) {
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.u64 = kEpollTagWake;
    {
      auto _tc_st = sys::safe_epoll_add(epoll_fd_, event_fd_, &ev);
      LOG_IF(WARNING, !_tc_st.ok()) << "PidMonitor: epoll add eventfd: " << _tc_st;
    }
    use_pidfd_.store(true);
  } else {
    // cleanup partially created
    if (epoll_fd_ >= 0) {
      ::close(epoll_fd_);
      epoll_fd_ = -1;
    }
    if (event_fd_ >= 0) {
      ::close(event_fd_);
      event_fd_ = -1;
    }
    use_pidfd_.store(false);
  }
  if (use_pidfd_.load()) {
    VLOG(1) << "PidMonitor: pidfd enabled";
  } else {
    LOG(WARNING) << "PidMonitor: pidfd unavailable; falling back to /proc polling";
  }
  th_ = std::thread([this]() { this->run_loop_(); });
}

void PidMonitor::stop() {
  if (!running_.load())
    return;
  running_.store(false);
  // Wake thread
  if (event_fd_ >= 0) {
    auto _tc_st = sys::safe_eventfd_write(event_fd_, /*v=*/1);
    LOG_IF(WARNING, !_tc_st.ok()) << "PidMonitor: wake on stop: " << _tc_st;
  }
  if (th_.joinable())
    th_.join();
  // Close fds
  if (epoll_fd_ >= 0) {
    ::close(epoll_fd_);
    epoll_fd_ = -1;
  }
  if (event_fd_ >= 0) {
    ::close(event_fd_);
    event_fd_ = -1;
  }
  // Close all pidfds
  {
    std::lock_guard<std::mutex> g(mu_);
    for (auto& kv : pidfds_) {
      if (kv.second >= 0)
        ::close(kv.second);
    }
    pidfds_.clear();
    watched_.clear();
  }
}

void PidMonitor::watch(pid_t pid) {
  if (pid <= 0)
    return;
  std::lock_guard<std::mutex> g(mu_);
  if (!watched_.insert(pid).second)
    return; // already watching
  if (use_pidfd_.load()) {
    int pfd = static_cast<int>(::syscall(SYS_pidfd_open, pid, 0));
    if (pfd >= 0) {
      pidfds_[pid] = pfd;
      epoll_event ev{};
      ev.events = EPOLLIN | EPOLLRDHUP | EPOLLHUP;
      ev.data.u64 = (kEpollTagPid | static_cast<uint64_t>(pid));
      {
        auto _tc_st = sys::safe_epoll_add(epoll_fd_, pfd, &ev);
        LOG_IF(WARNING, !_tc_st.ok()) << "PidMonitor: epoll add pidfd: " << _tc_st;
      }
    } else {
      const int err = errno;
      VLOG(1) << "PidMonitor: pidfd_open failed for pid=" << pid << ": "
              << absl::ErrnoToStatus(err, "pidfd_open failed");
      // pidfd unavailable for this pid (maybe it already died); leave in watched_ for poll fallback
    }
  }
  // Wake the epoll loop so fallback polling notices newly watched PIDs promptly
  if (event_fd_ >= 0) {
    auto _tc_st = sys::safe_eventfd_write(event_fd_, /*v=*/1);
    LOG_IF(WARNING, !_tc_st.ok()) << "PidMonitor: wake on watch: " << _tc_st;
  }
}

void PidMonitor::unwatch(pid_t pid) {
  if (pid <= 0)
    return;
  int pfd = -1;
  {
    std::lock_guard<std::mutex> g(mu_);
    watched_.erase(pid);
    auto it = pidfds_.find(pid);
    if (it != pidfds_.end()) {
      pfd = it->second;
      pidfds_.erase(it);
    }
  }
  if (use_pidfd_.load() && epoll_fd_ >= 0 && pfd >= 0) {
    {
      auto _tc_st = sys::safe_epoll_del(epoll_fd_, pfd);
      LOG_IF(WARNING, !_tc_st.ok()) << "PidMonitor: epoll del pidfd: " << _tc_st;
    }
    ::close(pfd);
  }
  // Wake the loop to apply changes promptly
  if (event_fd_ >= 0) {
    auto _tc_st = sys::safe_eventfd_write(event_fd_, /*v=*/1);
    LOG_IF(WARNING, !_tc_st.ok()) << "PidMonitor: wake on unwatch: " << _tc_st;
  }
}

bool PidMonitor::using_pidfd() const {
  return use_pidfd_.load();
}

bool PidMonitor::is_watching_for_test(pid_t pid) {
  std::lock_guard<std::mutex> g(mu_);
  return watched_.find(pid) != watched_.end();
}

void PidMonitor::run_loop_() {
  if (use_pidfd_.load()) {
    run_epoll_();
  } else {
    run_poll_();
  }
}

void PidMonitor::run_epoll_() {
  constexpr int kMaxEvents = 32;
  std::array<epoll_event, kMaxEvents> events{};
  while (running_.load()) {
    int timeout_ms = static_cast<int>(poll_interval_.count());
    if (timeout_ms <= 0)
      timeout_ms = 1;
    int n = ::epoll_wait(epoll_fd_, events.data(), kMaxEvents, timeout_ms);
    if (n < 0)
      continue;
    for (int i = 0; i < n; ++i) {
      const auto& ev = events[static_cast<size_t>(i)];
      const uint64_t data = ev.data.u64;
      if ((data & kEpollTagMask) == kEpollTagWake) {
        // drain eventfd
        uint64_t tmp;
        {
          auto _tc_st = sys::safe_eventfd_read(event_fd_, &tmp);
          LOG_IF(WARNING, !_tc_st.ok()) << "PidMonitor: drain eventfd: " << _tc_st;
        }
        continue;
      }
      if ((data & kEpollTagMask) != kEpollTagPid) {
        // Unknown tag; ignore
        continue;
      }
      pid_t pid = static_cast<pid_t>(data & kEpollValueMask);
      // Cleanup pidfd and mapping
      int pfd = -1;
      {
        std::lock_guard<std::mutex> g(mu_);
        auto it = pidfds_.find(pid);
        if (it != pidfds_.end()) {
          pfd = it->second;
          pidfds_.erase(it);
        }
        watched_.erase(pid);
      }
      if (pfd >= 0)
        ::close(pfd);
      // Notify
      if (cb_)
        cb_(pid);
    }
    // Poll fallback for PIDs without pidfd
    poll_missing_pidfds_once_();
  }
}

void PidMonitor::run_poll_() {
  while (running_.load()) {
    std::vector<pid_t> to_drop;
    {
      std::lock_guard<std::mutex> g(mu_);
      to_drop.reserve(watched_.size());
      for (pid_t pid : watched_) {
        if (::kill(pid, 0) == -1 && errno == ESRCH) {
          to_drop.push_back(pid);
        }
      }
      for (pid_t pid : to_drop) {
        watched_.erase(pid);
      }
    }
    for (pid_t pid : to_drop) {
      if (cb_)
        cb_(pid);
    }
    // Sleep for configured interval
    std::this_thread::sleep_for(poll_interval_);
  }
}

void PidMonitor::poll_missing_pidfds_once_() {
  if (!running_.load())
    return;
  std::vector<pid_t> to_drop;
  {
    std::lock_guard<std::mutex> g(mu_);
    to_drop.reserve(watched_.size());
    for (pid_t pid : watched_) {
      if (pidfds_.find(pid) != pidfds_.end())
        continue; // handled by epoll via pidfd
      if (::kill(pid, 0) == -1 && errno == ESRCH) {
        to_drop.push_back(pid);
      }
    }
    for (pid_t pid : to_drop) {
      watched_.erase(pid);
    }
  }
  for (pid_t pid : to_drop) {
    if (cb_)
      cb_(pid);
  }
}

} // namespace tensorcast::daemon
