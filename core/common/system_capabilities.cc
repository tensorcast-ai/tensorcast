// Copyright (c) 2025, TensorCast Team.

#include "core/common/system_capabilities.h"

#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstring>

#include "absl/log/log.h"

namespace tensorcast::common {

namespace {
// Try a tiny probe on anonymous page to check MADV flag acceptance.
bool probe_madv(int advice) noexcept {
  int64_t page_bytes = sysconf(_SC_PAGESIZE);
  void* p =
      ::mmap(nullptr, static_cast<size_t>(page_bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    return false;
  }
  int rc = ::madvise(p, static_cast<size_t>(page_bytes), advice);
  int saved = errno;
  ::munmap(p, static_cast<size_t>(page_bytes));
  if (rc == 0) {
    return true;
  }
  // EINVAL indicates unsupported advice on this kernel
  if (rc != 0 && saved == EINVAL) {
    return false;
  }
  // Other errors are treated as unsupported for safety.
  return false;
}

bool can_mlock_some_bytes() noexcept {
  // Check RLIMIT_MEMLOCK
  struct rlimit lim;
  if (getrlimit(RLIMIT_MEMLOCK, &lim) != 0) {
    PLOG(WARNING) << "getrlimit(RLIMIT_MEMLOCK) failed";
    return false;
  }
  if (lim.rlim_cur == 0) {
    return false;
  }
  // Try locking a single page to verify permission, then unlock.
  int64_t page_bytes = sysconf(_SC_PAGESIZE);
  void* p =
      ::mmap(nullptr, static_cast<size_t>(page_bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    return false;
  }
  if (::mlock(p, static_cast<size_t>(page_bytes)) != 0) {
    int e = errno;
    ::munmap(p, static_cast<size_t>(page_bytes));
    // EPERM/ENOMEM mean unavailable or quota too small → treat as not enabled.
    if (e == EPERM || e == ENOMEM) {
      return false;
    }
    return false;
  }
  if (::munlock(p, static_cast<size_t>(page_bytes)) != 0) {
    PLOG(WARNING) << "munlock probe unlock failed";
  }
  ::munmap(p, static_cast<size_t>(page_bytes));
  return true;
}
} // namespace

SystemCapabilities& SystemCapabilities::instance() {
  static SystemCapabilities inst;
  return inst;
}

SystemCapabilities::SystemCapabilities() {
  detect_mlock_();
  detect_madv_();
}

void SystemCapabilities::detect_mlock_() noexcept {
  bool ok = can_mlock_some_bytes();
  mlock_enabled_.store(ok, std::memory_order_release);
  LOG(INFO) << "SystemCapabilities: mlock " << (ok ? "enabled" : "disabled") << "; RLIMIT-based probe";
}

void SystemCapabilities::detect_madv_() noexcept {
  bool willneed = probe_madv(MADV_WILLNEED);
  bool free_ok = probe_madv(MADV_FREE);
#ifndef MADV_PAGEOUT
  bool pageout_ok = false;
#else
  bool pageout_ok = probe_madv(MADV_PAGEOUT);
#endif
  madv_willneed_available_.store(willneed, std::memory_order_release);
  madv_free_available_.store(free_ok, std::memory_order_release);
  madv_pageout_available_.store(pageout_ok, std::memory_order_release);
  LOG(INFO) << "SystemCapabilities: MADV_WILLNEED=" << willneed << ", MADV_FREE=" << free_ok
            << ", MADV_PAGEOUT=" << pageout_ok;
}

} // namespace tensorcast::common
