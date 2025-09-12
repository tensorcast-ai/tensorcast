// Copyright (c) 2025, TensorCast Team.

// Lightweight wrappers around common syscalls with errno→absl::Status mapping.
// Intended for daemon internals to avoid ad-hoc (void) casts and to centralize
// handling of expected errno cases.

#pragma once

#include <cstdint>

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <cerrno>

#include "absl/status/status.h"

namespace tensorcast::daemon::sys {

inline absl::Status safe_epoll_add(int epoll_fd, int fd, epoll_event* ev) {
  if (::epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, ev) < 0) {
    const int e = errno;
    if (e == EEXIST)
      return absl::OkStatus();
    return absl::ErrnoToStatus(e, "epoll_ctl(ADD) failed");
  }
  return absl::OkStatus();
}

inline absl::Status safe_epoll_del(int epoll_fd, int fd) {
  if (::epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, nullptr) < 0) {
    const int e = errno;
    if (e == ENOENT)
      return absl::OkStatus();
    return absl::ErrnoToStatus(e, "epoll_ctl(DEL) failed");
  }
  return absl::OkStatus();
}

inline absl::Status safe_eventfd_write(int event_fd, uint64_t v) {
  ssize_t n = ::write(event_fd, &v, sizeof(v));
  if (n != static_cast<ssize_t>(sizeof(v))) {
    const int e = errno;
    return absl::ErrnoToStatus(e, "eventfd write failed");
  }
  return absl::OkStatus();
}

inline absl::Status safe_eventfd_read(int event_fd, uint64_t* out) {
  uint64_t tmp = 0;
  ssize_t n = ::read(event_fd, &tmp, sizeof(tmp));
  if (n != static_cast<ssize_t>(sizeof(tmp))) {
    const int e = errno;
    return absl::ErrnoToStatus(e, "eventfd read failed");
  }
  if (out)
    *out = tmp;
  return absl::OkStatus();
}

} // namespace tensorcast::daemon::sys
