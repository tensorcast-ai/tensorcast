// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/contracts/loader.h"

namespace tensorcast::store {

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
  explicit InlineBufferLoader(loading::InlineBufferSource source) : source_(std::move(source)) {}

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
    absl::MutexLock lock(&mutex_);
    if (!initialized_) {
      return absl::FailedPreconditionError("InlineBufferLoader not initialized");
    }
    if (!source_.data) {
      return absl::FailedPreconditionError("InlineBufferSource does not contain inline data");
    }
    if (source_.size_bytes == 0) {
      return absl::FailedPreconditionError("InlineBufferSource size_bytes is 0");
    }

    class InlineSeekableSource final : public loader::SeekableSource {
     public:
      InlineSeekableSource(std::shared_ptr<const void> data, uint64_t size)
          : data_(std::move(data)), size_(size), cursor_(0) {}

      [[nodiscard]] uint64_t total_bytes() const override {
        return size_;
      }

      absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
        auto bytes_or = read_at(cursor_, dst, max_bytes);
        if (!bytes_or.ok()) {
          return bytes_or.status();
        }
        cursor_ += bytes_or.value();
        return bytes_or;
      }

      absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
        if (offset >= size_ || bytes == 0) {
          return static_cast<size_t>(0);
        }
        const uint64_t remaining = size_ - offset;
        const auto to_copy = static_cast<size_t>(std::min<uint64_t>(remaining, bytes));
        const auto* src = static_cast<const std::byte*>(data_.get()) + offset;
        std::memcpy(dst, src, to_copy);
        return to_copy;
      }

      [[nodiscard]] bool supports_direct_write_at() const override {
        return true;
      }

      absl::StatusOr<size_t> read_into_at(
          uint64_t src_offset,
          uint64_t dest_va_offset,
          size_t bytes,
          const DirectWriteGrant& grant) override {
        if (bytes == 0) {
          return static_cast<size_t>(0);
        }
        if (src_offset >= size_) {
          return static_cast<size_t>(0);
        }
        const uint64_t remaining = size_ - src_offset;
        const size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, remaining));
        const DirectWriteGrant::Window* target = nullptr;
        for (const auto& window : grant.windows) {
          if (dest_va_offset >= window.va_offset && dest_va_offset + to_copy <= window.va_offset + window.length) {
            target = &window;
            break;
          }
        }
        if (target == nullptr) {
          return absl::InvalidArgumentError("No direct-write window covers requested inline range");
        }
        auto window_offset = dest_va_offset - target->va_offset;
        auto* dst = reinterpret_cast<void*>(target->local_addr + window_offset);
        const auto* src = static_cast<const std::byte*>(data_.get()) + src_offset;
        std::memcpy(dst, src, to_copy);
        return to_copy;
      }

     private:
      std::shared_ptr<const void> data_;
      uint64_t size_;
      uint64_t cursor_;
    };

    return std::unique_ptr<loader::SeekableSource>(
        std::make_unique<InlineSeekableSource>(source_.data, source_.size_bytes));
  }

 private:
  loading::InlineBufferSource source_;
  bool initialized_ = false;
  absl::Mutex mutex_;
};

} // namespace tensorcast::store
