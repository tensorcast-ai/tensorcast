// Copyright (c) 2025, TensorCast Team.

#include "core/common/async_copy_manager.h"
#include "absl/container/flat_hash_map.h"

namespace tensorcast::common {

AsyncCopyManager& AsyncCopyManager::instance() {
  static AsyncCopyManager inst;
  return inst;
}

CopyHandle::~CopyHandle() = default;

absl::Status CopyHandle::wait(absl::Duration timeout) const {
  absl::MutexLock lock(&p_->mu);
  if (!p_->done) {
    if (timeout == absl::InfiniteDuration()) {
      p_->cv.Wait(&p_->mu);
    } else {
      if (!p_->cv.WaitWithTimeout(&p_->mu, timeout)) {
        return absl::DeadlineExceededError("copy wait timeout");
      }
    }
  }
  return p_->status;
}

bool CopyHandle::ok() const {
  absl::MutexLock lock(&p_->mu);
  return p_->done && p_->status.ok();
}

namespace {

inline const char* stage_or(const CopyOptions& opts, const char* def_stage) {
  return opts.tracing_stage ? opts.tracing_stage : def_stage;
}

} // namespace

absl::StatusOr<CopyHandle> AsyncCopyManager::submit_h2d(
    const HostRegion& src,
    const DeviceRegion& dst,
    cudaStream_t stream,
    const CopyOptions& opts) {
  if (src.base == nullptr || dst.dev_ptr == nullptr || src.length == 0 || dst.length == 0) {
    return absl::InvalidArgumentError("invalid region(s) for submit_h2d");
  }
  if (src.length != dst.length) {
    return absl::InvalidArgumentError("length mismatch in submit_h2d");
  }

  CopyHandle handle;
  auto p = handle.p_;

  // Wrap CUDA call in trace helper; schedule host callback to finalize handle
  const size_t bytes = src.length;
  void* dst_ptr = dst.dev_ptr;
  const void* src_ptr = src.base;
  // Resolve stream if not provided (per-device H2D)
  cudaStream_t stream_to_use = stream;
  if (stream_to_use == nullptr && dst.device_id >= 0) {
    static absl::Mutex mu(absl::kConstInit);
    static absl::flat_hash_map<int, cudaStream_t> h2d_streams ABSL_GUARDED_BY(mu);
    {
      absl::MutexLock lock(&mu);
      auto it = h2d_streams.find(dst.device_id);
      if (it == h2d_streams.end()) {
        cudaStream_t s = nullptr;
        auto st = cuda::set_device(dst.device_id);
        if (!st.ok())
          return st;
        st = cuda::stream_create_with_flags(&s, /*flags=*/1 /*cudaStreamNonBlocking*/);
        if (!st.ok())
          return st;
        it = h2d_streams.emplace(dst.device_id, s).first;
      }
      stream_to_use = it->second;
    }
  }
  auto status = tensorcast::common::trace::trace_cuda_async(
      stage_or(opts, "H2D/Copy"),
      stream_to_use,
      [&]() { return cuda::memcpy_async(dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice, stream_to_use); },
      [p, cb = opts.callbacks.on_copy_done]() {
        if (cb)
          cb();
        absl::MutexLock lock(&p->mu);
        p->status = absl::OkStatus();
        p->done = true;
        p->cv.SignalAll();
      });

  if (!status.ok()) {
    absl::MutexLock lock(&p->mu);
    p->status = status;
    p->done = true;
    p->cv.SignalAll();
    return status;
  }

  return handle;
}

absl::StatusOr<CopyHandle> AsyncCopyManager::submit_d2d(
    const DeviceRegion& src,
    const DeviceRegion& dst,
    cudaStream_t stream,
    const CopyOptions& opts) {
  if (src.dev_ptr == nullptr || dst.dev_ptr == nullptr || src.length == 0 || dst.length == 0) {
    return absl::InvalidArgumentError("invalid region(s) for submit_d2d");
  }
  if (src.length != dst.length) {
    return absl::InvalidArgumentError("length mismatch in submit_d2d");
  }
  if (src.device_id != dst.device_id) {
    return absl::UnimplementedError("cross-device D2D not yet implemented in ACM");
  }

  CopyHandle handle;
  auto p = handle.p_;

  const size_t bytes = src.length;
  const void* src_ptr = src.dev_ptr;
  void* dst_ptr = dst.dev_ptr;
  // Resolve stream if not provided (per-device D2D)
  cudaStream_t stream_to_use = stream;
  if (stream_to_use == nullptr && dst.device_id >= 0) {
    static absl::Mutex mu(absl::kConstInit);
    static absl::flat_hash_map<int, cudaStream_t> d2d_streams ABSL_GUARDED_BY(mu);
    {
      absl::MutexLock lock(&mu);
      auto it = d2d_streams.find(dst.device_id);
      if (it == d2d_streams.end()) {
        cudaStream_t s = nullptr;
        auto st = cuda::set_device(dst.device_id);
        if (!st.ok())
          return st;
        st = cuda::stream_create_with_flags(&s, /*flags=*/1 /*cudaStreamNonBlocking*/);
        if (!st.ok())
          return st;
        it = d2d_streams.emplace(dst.device_id, s).first;
      }
      stream_to_use = it->second;
    }
  }
  auto status = tensorcast::common::trace::trace_cuda_async(
      stage_or(opts, "D2D/Copy"),
      stream_to_use,
      [&]() { return cuda::memcpy_async(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice, stream_to_use); },
      [p, cb = opts.callbacks.on_copy_done]() {
        if (cb)
          cb();
        absl::MutexLock lock(&p->mu);
        p->status = absl::OkStatus();
        p->done = true;
        p->cv.SignalAll();
      });

  if (!status.ok()) {
    absl::MutexLock lock(&p->mu);
    p->status = status;
    p->done = true;
    p->cv.SignalAll();
    return status;
  }

  return handle;
}

absl::StatusOr<CopyHandle> AsyncCopyManager::submit_d2h(
    const DeviceRegion& src,
    const HostRegion& dst,
    cudaStream_t stream,
    const CopyOptions& opts) {
  if (src.dev_ptr == nullptr || dst.base == nullptr || src.length == 0 || dst.length == 0) {
    return absl::InvalidArgumentError("invalid region(s) for submit_d2h");
  }
  if (src.length != dst.length) {
    return absl::InvalidArgumentError("length mismatch in submit_d2h");
  }

  CopyHandle handle;
  auto p = handle.p_;

  const size_t bytes = src.length;
  const void* src_ptr = src.dev_ptr;
  void* dst_ptr = const_cast<void*>(dst.base);
  // Resolve stream if not provided (per-device D2H)
  cudaStream_t stream_to_use = stream;
  if (stream_to_use == nullptr && src.device_id >= 0) {
    static absl::Mutex mu(absl::kConstInit);
    static absl::flat_hash_map<int, cudaStream_t> d2h_streams ABSL_GUARDED_BY(mu);
    {
      absl::MutexLock lock(&mu);
      auto it = d2h_streams.find(src.device_id);
      if (it == d2h_streams.end()) {
        cudaStream_t s = nullptr;
        auto st = cuda::set_device(src.device_id);
        if (!st.ok())
          return st;
        st = cuda::stream_create_with_flags(&s, /*flags=*/1 /*cudaStreamNonBlocking*/);
        if (!st.ok())
          return st;
        it = d2h_streams.emplace(src.device_id, s).first;
      }
      stream_to_use = it->second;
    }
  }
  auto status = tensorcast::common::trace::trace_cuda_async(
      stage_or(opts, "D2H/Copy"),
      stream_to_use,
      [&]() { return cuda::memcpy_async(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost, stream_to_use); },
      [p, cb = opts.callbacks.on_copy_done]() {
        if (cb)
          cb();
        absl::MutexLock lock(&p->mu);
        p->status = absl::OkStatus();
        p->done = true;
        p->cv.SignalAll();
      });

  if (!status.ok()) {
    absl::MutexLock lock(&p->mu);
    p->status = status;
    p->done = true;
    p->cv.SignalAll();
    return status;
  }

  return handle;
}

absl::StatusOr<CopyHandle> AsyncCopyManager::submit_h2h(
    const HostRegion& src,
    const HostRegion& dst,
    const CopyOptions& opts) {
  if (src.base == nullptr || dst.base == nullptr || src.length == 0 || dst.length == 0) {
    return absl::InvalidArgumentError("invalid region(s) for submit_h2h");
  }
  if (src.length != dst.length) {
    return absl::InvalidArgumentError("length mismatch in submit_h2h");
  }

  CopyHandle handle;
  auto p = handle.p_;
  // Synchronous memcpy followed by immediate callback and handle completion.
  std::memcpy(const_cast<void*>(dst.base), src.base, src.length);
  if (opts.callbacks.on_copy_done) {
    opts.callbacks.on_copy_done();
  }
  {
    absl::MutexLock lock(&p->mu);
    p->status = absl::OkStatus();
    p->done = true;
    p->cv.SignalAll();
  }
  return handle;
}

} // namespace tensorcast::common
