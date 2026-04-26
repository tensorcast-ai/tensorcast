// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/mux_seekable_source.h"

#include <algorithm>
#include <map>
#include "absl/log/log.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::loader {

namespace {

absl::StatusOr<SeekableSource*> freeze_direct_branch(
    const std::shared_ptr<SeekableSource>& primary,
    const std::shared_ptr<SeekableSource>& fallback) {
  if (primary != nullptr && primary->supports_direct_write_at()) {
    return primary.get();
  }
  if (fallback != nullptr && fallback->supports_direct_write_at()) {
    return fallback.get();
  }
  return absl::UnimplementedError("direct write not supported by mux sources");
}

absl::StatusOr<SeekableSource*> freeze_batched_direct_branch(
    const std::shared_ptr<SeekableSource>& primary,
    const std::shared_ptr<SeekableSource>& fallback) {
  if (primary != nullptr && primary->supports_batched_direct_write_at()) {
    return primary.get();
  }
  if (fallback != nullptr && fallback->supports_batched_direct_write_at()) {
    return fallback.get();
  }
  return absl::UnimplementedError("batched direct write not supported by mux sources");
}

} // namespace

MuxSeekableSource::MuxSeekableSource(
    gsl::not_null<std::shared_ptr<SeekableSource>> primary,
    gsl::not_null<std::shared_ptr<SeekableSource>> fallback)
    : primary_(std::move(primary)), fallback_(std::move(fallback)) {}

uint64_t MuxSeekableSource::total_bytes() const {
  return std::max(primary_->total_bytes(), fallback_->total_bytes());
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
  const uint64_t total = total_bytes();
  if (offset >= total || bytes == 0) {
    return static_cast<size_t>(0);
  }
  const size_t bytes_to_read = static_cast<size_t>(std::min<uint64_t>(bytes, total - offset));
  size_t total_read = 0;
  char* ptr = static_cast<char*>(dst);

  // Try primary for the whole request
  {
    auto st = primary_->read_at(offset, ptr, bytes_to_read);
    if (st.ok()) {
      total_read = *st;
    } else {
      VLOG(1) << "MuxSeekableSource: primary read_at failed: " << st.status();
      // Metrics: record fallback due to primary error
      try {
        static auto meter =
            opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
        static auto counter = meter->CreateDoubleCounter("tc_fallback_chunks_total");
        std::map<std::string, opentelemetry::common::AttributeValue> attrs;
        attrs.emplace("reason", opentelemetry::common::AttributeValue(std::string("primary_error")));
        counter->Add(1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
      } catch (...) {
        VLOG(1) << "metrics counter tc_fallback_chunks_total unavailable";
      }
    }
  }

  // If short or failed and fallback exists, complete the remainder
  if (total_read < bytes_to_read) {
    size_t remain = bytes_to_read - total_read;
    auto fst = fallback_->read_at(offset + total_read, ptr + total_read, remain);
    if (!fst.ok()) {
      // Log fallback failure for debugging
      LOG(WARNING) << "MuxSeekableSource: fallback read_at failed after primary delivered " << total_read << " of "
                   << bytes_to_read << " bytes. Fallback error: " << fst.status();

      // If both primary and fallback failed to deliver any data, return fallback error.
      if (total_read == 0) {
        return fst.status();
      }
      // Partial success from primary; propagate bytes read so far.
      return total_read;
    }
    total_read += *fst;
    if (total_read == bytes_to_read) {
      // Metrics: record fallback due to short read (primary delivered fewer bytes than requested)
      try {
        static auto meter =
            opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
        static auto counter = meter->CreateDoubleCounter("tc_fallback_chunks_total");
        std::map<std::string, opentelemetry::common::AttributeValue> attrs;
        attrs.emplace("reason", opentelemetry::common::AttributeValue(std::string("short_read")));
        counter->Add(1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
      } catch (...) {
        VLOG(1) << "metrics counter tc_fallback_chunks_total unavailable";
      }
    }
  }

  if (total_read != bytes_to_read) {
    return absl::DataLossError("MuxSeekableSource short read before expected EOF");
  }
  return total_read;
}

bool MuxSeekableSource::supports_direct_write_at() const {
  return primary_->supports_direct_write_at() || fallback_->supports_direct_write_at();
}

bool MuxSeekableSource::supports_batched_direct_write_at() const {
  return primary_->supports_batched_direct_write_at() || fallback_->supports_batched_direct_write_at();
}

absl::StatusOr<size_t> MuxSeekableSource::read_into_at(
    uint64_t src_offset,
    uint64_t dest_va_offset,
    size_t bytes,
    const DirectWriteGrant& grant) {
  auto selected_or = freeze_direct_branch(primary_, fallback_);
  if (!selected_or.ok()) {
    return selected_or.status();
  }
  return (*selected_or)->read_into_at(src_offset, dest_va_offset, bytes, grant);
}

absl::StatusOr<size_t> MuxSeekableSource::readv_into_at(
    absl::Span<const DirectWriteOp> ops,
    const DirectWriteGrant& grant) {
  if (ops.empty()) {
    return static_cast<size_t>(0);
  }
  auto selected_or = freeze_batched_direct_branch(primary_, fallback_);
  if (!selected_or.ok()) {
    return selected_or.status();
  }
  return (*selected_or)->readv_into_at(ops, grant);
}

} // namespace tensorcast::store::loader
