// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/dataplane/sources/mux_seekable_source.h"

#include <map>
#include "absl/log/log.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::loader {

MuxSeekableSource::MuxSeekableSource(
    gsl::not_null<std::shared_ptr<SeekableSource>> primary,
    gsl::not_null<std::shared_ptr<SeekableSource>> fallback)
    : primary_(std::move(primary)), fallback_(std::move(fallback)) {}

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
  {
    auto st = primary_->read_at(offset, ptr, bytes);
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
  if (total_read < bytes) {
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

  return total_read;
}

} // namespace tensorcast::store::loader
