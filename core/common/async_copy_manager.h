// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <queue>
#include <thread>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "absl/container/flat_hash_map.h"

#include "core/common/async_runtime.h"
#include "core/common/cuda_api.h"

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
  // Fired asynchronously on a CPU worker thread after the CUDA work for this
  // handle has completed. The callback runs outside of cudaLaunchHostFunc so it
  // must still avoid invoking CUDA APIs directly; use AsyncCopyManager helpers
  // (e.g., submit another copy) instead of calling CUDA Runtime/Driver calls.
  // Status conveys the completion state reported by CUDA.
  std::function<void(absl::Status)> on_copy_done;
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
  [[nodiscard]] bool ok() const;

 private:
  absl::Status resolve_status_() const;

  struct Impl {
    mutable absl::Mutex mu;
    bool done ABSL_GUARDED_BY(mu) = false;
    absl::Status status ABSL_GUARDED_BY(mu) = absl::OkStatus();
    absl::CondVar cv;
    int device_id ABSL_GUARDED_BY(mu) = -1;
    mutable bool needs_device_check ABSL_GUARDED_BY(mu) = false;
  };

  std::shared_ptr<Impl> p_ = std::make_shared<Impl>();

  friend class AsyncCopyManager;
};

// A minimal ACM that wraps a single cudaMemcpyAsync in a traced host callback
// and returns a CopyHandle that becomes ready when the callback fires.
class AsyncCopyManager {
 public:
  static AsyncCopyManager& instance();

  // Optional: bind copy callbacks and trace fallbacks to an injected runtime.
  // When set, host callbacks will schedule user callbacks onto
  // AsyncRuntime::cpu_executor and stream-synchronization fallback work onto
  // AsyncRuntime::blocking_executor. If not set, AsyncCopyManager falls back to
  // its internal callback worker thread.
  void set_async_runtime(std::weak_ptr<AsyncRuntime> async_runtime);

  absl::StatusOr<CopyHandle> submit_h2d(const HostRegion& src, const DeviceRegion& dst, const CopyOptions& opts = {});

  absl::StatusOr<CopyHandle> submit_d2h(const DeviceRegion& src, const HostRegion& dst, const CopyOptions& opts = {});

  // Minimal D2D support: same-device only for this phase.
  absl::StatusOr<CopyHandle> submit_d2d(const DeviceRegion& src, const DeviceRegion& dst, const CopyOptions& opts = {});

  // Dispatches the memcpy on the callback worker so completion behaves like
  // other async submissions, enabling CPU-only tests to exercise pump logic.
  absl::StatusOr<CopyHandle> submit_h2h(const HostRegion& src, const HostRegion& dst, const CopyOptions& opts = {});

  // Synchronize the per-device H2D stream to guarantee that all outstanding
  // host→device copies have completed. If no stream has been created for the
  // device, this returns OK immediately.
  absl::Status synchronize_h2d_stream(int device_id);

  // Destroy all lazily-created per-device streams. Safe to call multiple times.
  void shutdown();

 private:
  AsyncCopyManager();
  ~AsyncCopyManager();

  void shutdown_impl_(bool destroy_streams);

  // Internal helpers to lazily create and cache per-device non-blocking streams
  absl::StatusOr<cudaStream_t> get_h2d_stream_(int device_id);
  absl::StatusOr<cudaStream_t> get_d2h_stream_(int device_id);
  absl::StatusOr<cudaStream_t> get_d2d_stream_(int device_id);
  std::shared_ptr<AsyncRuntime> get_async_runtime_() const;
  void enqueue_callback_(std::function<void()> cb);
  void callback_loop_();

  absl::Mutex mu_;
  absl::flat_hash_map<int, cudaStream_t> h2d_streams_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<int, cudaStream_t> d2h_streams_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<int, cudaStream_t> d2d_streams_ ABSL_GUARDED_BY(mu_);

  absl::Mutex callback_mu_;
  absl::CondVar callback_cv_;
  std::queue<std::function<void()>> callback_queue_ ABSL_GUARDED_BY(callback_mu_);
  bool callback_shutdown_ ABSL_GUARDED_BY(callback_mu_) = false;
  bool callback_thread_started_ ABSL_GUARDED_BY(callback_mu_) = false;
  std::thread callback_thread_;

  mutable absl::Mutex runtime_mu_;
  std::weak_ptr<AsyncRuntime> async_runtime_ ABSL_GUARDED_BY(runtime_mu_);
};

} // namespace tensorcast::common
