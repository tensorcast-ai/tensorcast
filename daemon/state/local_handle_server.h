// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <atomic>
#include <string>
#include <thread>

#include "absl/status/status.h"
#include "daemon/state/handle_lease_registry.h"

namespace tensorcast::daemon {

// LocalHandleServer is a local-only Unix domain socket service used for:
//  - CPU memfd FD handoff (SCM_RIGHTS)
//  - Handle lease release (lease_token)
//
// The server enforces uid==daemon uid via SO_PEERCRED and creates the socket
// file with mode 0600.
class LocalHandleServer final {
 public:
  struct Options {
    std::string socket_path;
    bool cpu_shared_memory_enabled{false};
  };

  LocalHandleServer(Options opts, HandleLeaseRegistry& leases);
  ~LocalHandleServer();

  LocalHandleServer(const LocalHandleServer&) = delete;
  LocalHandleServer& operator=(const LocalHandleServer&) = delete;

  [[nodiscard]] absl::Status start();
  void stop();

 private:
  enum class OpCode : uint8_t { kGetCpuMemfdFd = 1, kReleaseHandle = 2 };
  enum class RespCode : uint8_t {
    kOk = 0,
    kNotFound = 1,
    kFailedPrecondition = 2,
    kPermissionDenied = 3,
    kInternal = 4,
  };

  [[nodiscard]] absl::Status bind_and_listen_();
  void run_loop_();

  [[nodiscard]] static absl::Status validate_socket_parent_dir_(const std::string& path);
  [[nodiscard]] static absl::Status remove_stale_socket_if_safe_(const std::string& path);

  [[nodiscard]] static absl::Status read_exact_(int fd, void* buf, size_t n);
  [[nodiscard]] static absl::Status write_exact_(int fd, const void* buf, size_t n);

  [[nodiscard]] static RespCode map_status_(const absl::Status& st);
  [[nodiscard]] absl::Status handle_conn_(int conn_fd);

  Options opts_;
  HandleLeaseRegistry* leases_;

  std::atomic<bool> running_{false};
  int listen_fd_{-1};
  std::thread th_;
};

} // namespace tensorcast::daemon
