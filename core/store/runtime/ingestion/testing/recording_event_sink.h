// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "core/store/runtime/ingestion/ingestion_runtime.h"

namespace tensorcast::store::runtime::ingestion::testing {

class RecordingEventSink final : public IngestionEventSink {
 public:
  struct RecordedEvent {
    RuntimeEventType type;
    IngestionResultEvent event;
  };

  void publish(RuntimeEventType type, IngestionResultEvent event) override {
    absl::MutexLock lock(&mu_);
    events_.push_back(RecordedEvent{.type = type, .event = std::move(event)});
  }

  [[nodiscard]] std::vector<RecordedEvent> snapshot() const {
    absl::MutexLock lock(&mu_);
    return events_;
  }

  [[nodiscard]] std::vector<RecordedEvent> drain() {
    absl::MutexLock lock(&mu_);
    auto copy = events_;
    events_.clear();
    return copy;
  }

  [[nodiscard]] size_t size() const {
    absl::MutexLock lock(&mu_);
    return events_.size();
  }

 private:
  mutable absl::Mutex mu_;
  std::vector<RecordedEvent> events_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::store::runtime::ingestion::testing
