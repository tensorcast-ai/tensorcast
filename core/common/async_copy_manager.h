// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "core/common/cuda_api.h"
#include "core/common/trace/trace_cuda_async_fn.h"

namespace tensorcast::common {

// Lightweight region descriptors kept close to RFC-0018 API shape.
struct HostRegion {
  const void* base{nullptr};
  size_t length{0};
  bool pinned{false}; // Informational only in this minimal implementation.
};

struct DeviceRegion {
  int device_id{-1};
  void* dev_ptr{nullptr};
  size_t length{0};
};

struct CopyCallbacks {
  std::function<void()> on_copy_done; // Fired on host callback when copy completes
};

struct CopyOptions {
  const char* tracing_stage{nullptr};
  CopyCallbacks callbacks{};
};

class CopyHandle {
 public:
  CopyHandle() = default;
  ~CopyHandle();
  CopyHandle(CopyHandle&&) noexcept = default;
  CopyHandle& operator=(CopyHandle&&) noexcept = default;
  CopyHandle(const CopyHandle&) = delete;
  CopyHandle& operator=(const CopyHandle&) = delete;

  absl::Status wait(absl::Duration timeout = absl::InfiniteDuration()) const;
  bool ok() const;

 private:
  struct Impl {
    mutable absl::Mutex mu;
    bool done ABSL_GUARDED_BY(mu) = false;
    absl::Status status ABSL_GUARDED_BY(mu) = absl::OkStatus();
    absl::CondVar cv;
  };
  std::shared_ptr<Impl> p_ = std::make_shared<Impl>();

  friend class AsyncCopyManager;
};

// A minimal ACM that wraps a single cudaMemcpyAsync in a traced host callback
// and returns a CopyHandle that becomes ready when the callback fires.
class AsyncCopyManager {
 public:
  static AsyncCopyManager& instance();

  absl::StatusOr<CopyHandle> submit_h2d(
      const HostRegion& src,
      const DeviceRegion& dst,
      cudaStream_t stream = nullptr,
      const CopyOptions& opts = {});

  absl::StatusOr<CopyHandle> submit_d2h(
      const DeviceRegion& src,
      const HostRegion& dst,
      cudaStream_t stream = nullptr,
      const CopyOptions& opts = {});

  // Minimal D2D support: same-device only for this phase.
  absl::StatusOr<CopyHandle> submit_d2d(
      const DeviceRegion& src,
      const DeviceRegion& dst,
      cudaStream_t stream = nullptr,
      const CopyOptions& opts = {});

  absl::StatusOr<CopyHandle> submit_h2h(const HostRegion& src, const HostRegion& dst, const CopyOptions& opts = {});

 private:
  AsyncCopyManager() = default;
  ~AsyncCopyManager() = default;
};

} // namespace tensorcast::common
