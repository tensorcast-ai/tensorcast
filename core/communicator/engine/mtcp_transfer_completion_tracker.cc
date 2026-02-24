// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/engine/mtcp_transfer_completion_tracker.h"

#include "absl/log/log.h"

namespace tensorcast::communicator::engine {

MtcpTransferCompletionTracker::MtcpTransferCompletionTracker(CompletionCallback callback)
    : callback_(std::move(callback)) {}

void MtcpTransferCompletionTracker::add_pending_segments(int segment_count) {
  if (segment_count <= 0) {
    return;
  }
  pending_segments_.fetch_add(segment_count, std::memory_order_acq_rel);
}

void MtcpTransferCompletionTracker::mark_final_window_enqueued() {
  final_window_enqueued_.store(true, std::memory_order_release);
  maybe_complete();
}

void MtcpTransferCompletionTracker::mark_segment_finished(bool success) {
  if (!success) {
    segment_failure_.store(true, std::memory_order_release);
  }

  const int previous = pending_segments_.fetch_sub(1, std::memory_order_acq_rel);
  if (previous <= 0) {
    pending_segments_.store(0, std::memory_order_release);
    LOG(WARNING) << "MtcpTransferCompletionTracker pending segment underflow";
  }

  maybe_complete();
}

void MtcpTransferCompletionTracker::fail_fast(const absl::Status& status) {
  complete_once(status.ok() ? absl::InternalError("MTCP transfer fail_fast requires non-OK status") : status);
}

int MtcpTransferCompletionTracker::pending_segments() const {
  return pending_segments_.load(std::memory_order_acquire);
}

void MtcpTransferCompletionTracker::maybe_complete() {
  if (!final_window_enqueued_.load(std::memory_order_acquire)) {
    return;
  }
  if (pending_segments_.load(std::memory_order_acquire) != 0) {
    return;
  }
  if (segment_failure_.load(std::memory_order_acquire)) {
    complete_once(absl::UnavailableError("MTCP send completed with segment failures"));
    return;
  }
  complete_once(absl::OkStatus());
}

void MtcpTransferCompletionTracker::complete_once(const absl::Status& status) {
  if (completed_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  if (callback_) {
    callback_(status);
  }
}

} // namespace tensorcast::communicator::engine
