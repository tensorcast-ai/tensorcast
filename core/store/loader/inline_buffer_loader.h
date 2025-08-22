// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/store/loader/loader.h"
#include "core/store/loading/loading_spec.h"

namespace stepcast::store {

/**
 * @brief Minimal loader for InlineBufferSource.
 *
 * This loader exists to support memory-only flows such as RFC-0006
 * registration, where we need a Replica instance with a known size but have
 * no real data source to stream from. It reports the provided size and
 * does not implement open_source().
 */
class InlineBufferLoader : public IArtifactLoader {
 public:
  explicit InlineBufferLoader(InlineBufferSource source) : source_(std::move(source)) {}

  ~InlineBufferLoader() override = default;

  InlineBufferLoader(const InlineBufferLoader&) = delete;
  InlineBufferLoader& operator=(const InlineBufferLoader&) = delete;
  InlineBufferLoader(InlineBufferLoader&&) = delete;
  InlineBufferLoader& operator=(InlineBufferLoader&&) = delete;

  absl::Status initialize() override {
    absl::MutexLock lock(&mutex_);
    initialized_ = true;
    return absl::OkStatus();
  }

  absl::StatusOr<uint64_t> get_artifact_size() override {
    absl::MutexLock lock(&mutex_);
    if (!initialized_) {
      return absl::FailedPreconditionError("InlineBufferLoader not initialized");
    }
    if (source_.size_bytes == 0) {
      return absl::FailedPreconditionError("InlineBufferSource size_bytes is 0");
    }
    return source_.size_bytes;
  }

  absl::StatusOr<std::unique_ptr<loader::SeekableSource>> open_source() override {
    // This loader is size-only; no streaming source is provided.
    return absl::UnimplementedError("InlineBufferLoader does not provide a streaming source");
  }

 private:
  InlineBufferSource source_;
  bool initialized_ = false;
  absl::Mutex mutex_;
};

} // namespace stepcast::store
