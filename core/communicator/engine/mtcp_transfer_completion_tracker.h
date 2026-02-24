// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <atomic>
#include <functional>

#include "absl/status/status.h"

namespace tensorcast::communicator::engine {

class MtcpTransferCompletionTracker {
 public:
  using CompletionCallback = std::function<void(const absl::Status&)>;

  explicit MtcpTransferCompletionTracker(CompletionCallback callback);

  void add_pending_segments(int segment_count);
  void mark_final_window_enqueued();
  void mark_segment_finished(bool success);
  void fail_fast(const absl::Status& status);

  [[nodiscard]] int pending_segments() const;

 private:
  void maybe_complete();
  void complete_once(const absl::Status& status);

  CompletionCallback callback_;
  std::atomic<int> pending_segments_{0};
  std::atomic<bool> final_window_enqueued_{false};
  std::atomic<bool> segment_failure_{false};
  std::atomic<bool> completed_{false};
};

} // namespace tensorcast::communicator::engine
