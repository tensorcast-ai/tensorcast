// Copyright (c) 2025, TensorCast Team.

#include "core/store/loader/mux_seekable_source.h"

#include "absl/log/log.h"
#include "core/common/metrics/metric_objects.h"

namespace tensorcast::store::loader {

MuxSeekableSource::MuxSeekableSource(std::shared_ptr<SeekableSource> primary, std::shared_ptr<SeekableSource> fallback)
    : primary_(std::move(primary)), fallback_(std::move(fallback)) {
  if (!primary_) {
    LOG(ERROR) << "MuxSeekableSource: primary source is null";
  }
  if (!fallback_) {
    LOG(WARNING) << "MuxSeekableSource: fallback source is null; no fallback will be possible";
  }
}

absl::StatusOr<size_t> MuxSeekableSource::read(void* dst, size_t max_bytes) {
  // Implement in terms of read_at with current_offset_
  auto st = read_at(current_offset_, dst, max_bytes);
  if (!st.ok()) {
    return st;
  }
  current_offset_ += *st;
  return st;
}

absl::StatusOr<size_t> MuxSeekableSource::read_at(uint64_t offset, void* dst, size_t bytes) {
  size_t total_read = 0;
  char* ptr = static_cast<char*>(dst);

  if (bytes == 0) {
    return static_cast<size_t>(0);
  }

  // Try primary for the whole request
  if (primary_) {
    auto st = primary_->read_at(offset, ptr, bytes);
    if (st.ok()) {
      total_read = *st;
    } else {
      VLOG(1) << "MuxSeekableSource: primary read_at failed: " << st.status();
      // Metrics: record fallback due to primary error
      try {
        static const metrics::Counter kFallbackChunks("fallback_chunks_total");
        kFallbackChunks.with_labels({{"reason", "primary_error"}}).inc();
      } catch (...) {
      }
    }
  }

  // If short or failed and fallback exists, complete the remainder
  if (fallback_ && total_read < bytes) {
    size_t remain = bytes - total_read;
    auto fst = fallback_->read_at(offset + total_read, ptr + total_read, remain);
    if (!fst.ok()) {
      // Log fallback failure for debugging
      LOG(WARNING) << "MuxSeekableSource: fallback read_at failed after primary delivered " << total_read << " of "
                   << bytes << " bytes. Fallback error: " << fst.status();

      // If both primary and fallback failed to deliver any data, return fallback error.
      if (total_read == 0) {
        return fst.status();
      }
      // Partial success from primary; propagate bytes read so far.
      return total_read;
    }
    total_read += *fst;
    if (total_read == bytes) {
      // Metrics: record fallback due to short read (primary delivered fewer bytes than requested)
      try {
        static const metrics::Counter kFallbackChunks("fallback_chunks_total");
        kFallbackChunks.with_labels({{"reason", "short_read"}}).inc();
      } catch (...) {
      }
    }
  }

  return total_read;
}

} // namespace tensorcast::store::loader
