// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <sys/types.h>

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tensorcast::daemon::testing {

struct CudaIpcBufferSpec {
  size_t size_bytes = 0;
  int fill_byte = -1;
};

absl::StatusOr<std::string> resolve_cuda_ipc_helper_path();

class CudaIpcChild {
 public:
  CudaIpcChild() = default;
  CudaIpcChild(const CudaIpcChild&) = delete;
  CudaIpcChild& operator=(const CudaIpcChild&) = delete;
  CudaIpcChild(CudaIpcChild&& other) noexcept;
  CudaIpcChild& operator=(CudaIpcChild&& other) noexcept;
  ~CudaIpcChild();

  static absl::StatusOr<CudaIpcChild> Spawn(
      std::string helper_path,
      int device_id,
      const std::vector<CudaIpcBufferSpec>& buffers);

  absl::Status Shutdown();

  pid_t pid() const {
    return pid_;
  }

  const std::vector<std::string>& handle_bytes() const {
    return handle_bytes_;
  }

 private:
  void reset_no_wait();

  pid_t pid_{-1};
  int stdin_fd_{-1};
  int stdout_fd_{-1};
  bool closed_{true};
  std::vector<std::string> handle_bytes_;
};

} // namespace tensorcast::daemon::testing
