// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/source_window_scheduler.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::loader {
namespace {

struct WindowSlice {
  uint64_t src_offset{0};
  uint64_t dst_offset{0};
  uint64_t length{0};
};

struct SourceWindow {
  uint32_t source_index{0};
  uint64_t src_offset{0};
  uint64_t length{0};
  uint64_t payload_bytes{0};
  std::vector<WindowSlice> slices;
};

struct SourceSegment {
  uint32_t source_index{0};
  uint64_t src_offset{0};
  uint64_t dst_offset{0};
  uint64_t length{0};
};

struct MetricsHandles {
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> meter;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> base_read_calls;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> base_read_bytes;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> output_bytes;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> pad_bytes;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> amplification_ratio;
};

MetricsHandles init_metrics_handles() {
  MetricsHandles handles;
  handles.meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  handles.base_read_calls = handles.meter->CreateDoubleCounter("tc_byte_range_base_read_calls_total");
  handles.base_read_bytes = handles.meter->CreateDoubleCounter("tc_byte_range_base_read_bytes_total");
  handles.output_bytes = handles.meter->CreateDoubleCounter("tc_byte_range_output_bytes_total");
  handles.pad_bytes = handles.meter->CreateDoubleCounter("tc_byte_range_pad_bytes_total");
  handles.amplification_ratio = handles.meter->CreateDoubleHistogram("tc_byte_range_amplification_ratio");
  return handles;
}

MetricsHandles& metrics_handles() {
  static MetricsHandles handles = init_metrics_handles();
  return handles;
}

void record_counter(
    const opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>>& counter,
    std::string_view path,
    uint64_t value) {
  if (value == 0) {
    return;
  }
  try {
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("path", opentelemetry::common::AttributeValue(std::string(path)));
    counter->Add(
        static_cast<double>(value),
        opentelemetry::common::KeyValueIterableView(attrs),
        opentelemetry::context::Context{});
  } catch (...) {
    VLOG(2) << "metrics counter tc_byte_range_* unavailable";
  }
}

void record_histogram(
    const opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>>& histogram,
    std::string_view path,
    double value) {
  try {
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("path", opentelemetry::common::AttributeValue(std::string(path)));
    histogram->Record(value, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
    VLOG(2) << "metrics histogram tc_byte_range_amplification_ratio unavailable";
  }
}

bool amplification_within_limit(uint64_t window_bytes, uint64_t payload_bytes, uint32_t max_amplification) {
  if (payload_bytes == 0 || max_amplification == 0) {
    return false;
  }
  if (payload_bytes > std::numeric_limits<uint64_t>::max() / max_amplification) {
    return false;
  }
  return window_bytes <= static_cast<uint64_t>(max_amplification) * payload_bytes;
}

absl::Status validate_sources(
    absl::Span<const SourceSegment> segments,
    absl::Span<const std::shared_ptr<SeekableSource>> sources) {
  for (const auto& seg : segments) {
    if (seg.length == 0) {
      continue;
    }
    if (seg.source_index >= sources.size()) {
      return absl::InvalidArgumentError("source index out of range for source-ordered scheduler");
    }
    const uint64_t total = sources[seg.source_index]->total_bytes();
    if (seg.src_offset > total || seg.length > total - seg.src_offset) {
      return absl::InvalidArgumentError("source-ordered scheduler segment out of bounds");
    }
  }
  return absl::OkStatus();
}

std::vector<SourceWindow> split_windows(std::vector<SourceWindow> windows, uint64_t cap_bytes) {
  if (cap_bytes == 0) {
    return windows;
  }
  std::vector<SourceWindow> out;
  for (const auto& win : windows) {
    if (win.length <= cap_bytes) {
      out.push_back(win);
      continue;
    }
    const uint64_t win_start = win.src_offset;
    const uint64_t win_end = win.src_offset + win.length;
    for (uint64_t chunk_start = win_start; chunk_start < win_end; chunk_start += cap_bytes) {
      const uint64_t chunk_end = std::min<uint64_t>(chunk_start + cap_bytes, win_end);
      SourceWindow chunk;
      chunk.source_index = win.source_index;
      chunk.src_offset = chunk_start;
      chunk.length = chunk_end - chunk_start;
      chunk.payload_bytes = 0;
      for (const auto& slice : win.slices) {
        const uint64_t slice_end = slice.src_offset + slice.length;
        const uint64_t overlap_start = std::max<uint64_t>(slice.src_offset, chunk_start);
        const uint64_t overlap_end = std::min<uint64_t>(slice_end, chunk_end);
        if (overlap_end <= overlap_start) {
          continue;
        }
        WindowSlice chunk_slice;
        chunk_slice.src_offset = overlap_start;
        chunk_slice.dst_offset = slice.dst_offset + (overlap_start - slice.src_offset);
        chunk_slice.length = overlap_end - overlap_start;
        chunk.payload_bytes += chunk_slice.length;
        chunk.slices.push_back(chunk_slice);
      }
      if (chunk.payload_bytes > 0) {
        out.push_back(std::move(chunk));
      }
    }
  }
  return out;
}

} // namespace

SourceWindowScheduler::SourceWindowScheduler(Options options) : options_(std::move(options)) {
  if (options_.prefetch_depth == 0) {
    options_.prefetch_depth = 1;
  }
}

absl::Status SourceWindowScheduler::Execute(
    const ByteRangeMap& map,
    absl::Span<const std::shared_ptr<SeekableSource>> sources,
    PositionedSink& sink,
    std::shared_ptr<common::memory::PinnedBufferPool> pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    bool use_pinned_buffers) {
  auto normalized_or = normalize_byte_range_map(map);
  if (!normalized_or.ok()) {
    return normalized_or.status();
  }
  const auto& normalized = *normalized_or;

  std::vector<SourceSegment> data_segments;
  std::vector<ByteRangeSegment> pad_segments;
  data_segments.reserve(normalized.segments.size());
  pad_segments.reserve(normalized.segments.size());
  for (const auto& seg : normalized.segments) {
    if (seg.length == 0) {
      continue;
    }
    if (seg.kind == ByteRangeSegment::Kind::kPad) {
      pad_segments.push_back(seg);
      continue;
    }
    data_segments.push_back(
        SourceSegment{
            .source_index = seg.source_index,
            .src_offset = seg.src_offset,
            .dst_offset = seg.dst_offset,
            .length = seg.length,
        });
  }

  std::sort(data_segments.begin(), data_segments.end(), [](const SourceSegment& a, const SourceSegment& b) {
    if (a.source_index != b.source_index) {
      return a.source_index < b.source_index;
    }
    return a.src_offset < b.src_offset;
  });

  if (auto st = validate_sources(data_segments, sources); !st.ok()) {
    return st;
  }

  std::vector<SourceWindow> windows;
  SourceWindow current;
  bool has_current = false;
  uint64_t current_end = 0;

  auto flush_current = [&]() {
    if (!has_current) {
      return;
    }
    windows.push_back(std::move(current));
    current = SourceWindow{};
    has_current = false;
    current_end = 0;
  };

  for (const auto& seg : data_segments) {
    if (!has_current || seg.source_index != current.source_index) {
      flush_current();
      has_current = true;
      current.source_index = seg.source_index;
      current.src_offset = seg.src_offset;
      current.length = seg.length;
      current.payload_bytes = seg.length;
      current.slices.clear();
      current.slices.push_back(
          WindowSlice{
              .src_offset = seg.src_offset,
              .dst_offset = seg.dst_offset,
              .length = seg.length,
          });
      current_end = seg.src_offset + seg.length;
      continue;
    }

    if (seg.src_offset < current_end) {
      return absl::InvalidArgumentError("source-ordered scheduler found overlapping segments");
    }

    const uint64_t gap = seg.src_offset - current_end;
    const uint64_t candidate_end = seg.src_offset + seg.length;
    const uint64_t candidate_payload = current.payload_bytes + seg.length;
    const uint64_t candidate_len = candidate_end - current.src_offset;
    const bool gap_ok = gap <= options_.merge_max_gap_bytes;
    const bool amp_ok = amplification_within_limit(candidate_len, candidate_payload, options_.merge_max_amplification);

    if (gap_ok && amp_ok) {
      current.length = candidate_len;
      current.payload_bytes = candidate_payload;
      current.slices.push_back(
          WindowSlice{
              .src_offset = seg.src_offset,
              .dst_offset = seg.dst_offset,
              .length = seg.length,
          });
      current_end = candidate_end;
    } else {
      flush_current();
      has_current = true;
      current.source_index = seg.source_index;
      current.src_offset = seg.src_offset;
      current.length = seg.length;
      current.payload_bytes = seg.length;
      current.slices.clear();
      current.slices.push_back(
          WindowSlice{
              .src_offset = seg.src_offset,
              .dst_offset = seg.dst_offset,
              .length = seg.length,
          });
      current_end = seg.src_offset + seg.length;
    }
  }
  flush_current();

  windows = split_windows(std::move(windows), options_.window_cap_bytes);

  uint64_t base_read_calls = 0;
  uint64_t base_read_bytes = 0;
  uint64_t output_bytes = 0;
  uint64_t pad_bytes = 0;

  const uint64_t zero_chunk_bytes = options_.window_cap_bytes > 0 ? options_.window_cap_bytes : (1ULL << 20);
  std::vector<uint8_t> zero_buf(static_cast<size_t>(zero_chunk_bytes), 0);
  for (const auto& pad : pad_segments) {
    uint64_t remaining = pad.length;
    uint64_t cursor = pad.dst_offset;
    while (remaining > 0) {
      const uint64_t chunk = std::min<uint64_t>(remaining, zero_chunk_bytes);
      auto st = sink.write_at(cursor, zero_buf.data(), static_cast<size_t>(chunk));
      if (!st.ok()) {
        return st;
      }
      pad_bytes += chunk;
      output_bytes += chunk;
      remaining -= chunk;
      cursor += chunk;
    }
  }

  auto* async_sink = dynamic_cast<AsyncPositionedSink*>(&sink);
  const bool use_async = async_sink != nullptr;

  if (use_pinned_buffers) {
    if (!pinned_pool) {
      return absl::FailedPreconditionError("pinned_pool is required for pinned source-ordered scheduling");
    }
    if (options_.window_cap_bytes == 0) {
      return absl::FailedPreconditionError("window_cap_bytes must be > 0 for pinned source-ordered scheduling");
    }
    const size_t pinned_slice_bytes = pinned_pool->slice_bytes();
    if (pinned_slice_bytes == 0) {
      return absl::FailedPreconditionError("pinned_pool slice_bytes is zero");
    }
    if (options_.window_cap_bytes > pinned_slice_bytes) {
      return absl::FailedPreconditionError("window_cap_bytes exceeds pinned_pool slice_bytes");
    }
    auto spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
        std::max<size_t>(1, options_.prefetch_depth), pinned_slice_bytes, pinned_pool);
    auto init_status = spb->initialize(
        pinned_timeout,
        absl::StrCat(
            "path=",
            options_.path,
            " source_ordered=true prefetch_depth=",
            options_.prefetch_depth,
            " window_cap_bytes=",
            options_.window_cap_bytes));
    if (!init_status.ok()) {
      return init_status;
    }

    struct InFlight {
      int slot_id;
      std::vector<common::CopyHandle> handles;
    };

    std::deque<InFlight> inflight;
    bool spb_released = false;

    auto cleanup_spb = [&]() {
      if (spb_released) {
        return;
      }
      while (!inflight.empty()) {
        InFlight entry = std::move(inflight.front());
        inflight.pop_front();
        for (auto& handle : entry.handles) {
          auto st = handle.wait();
          if (!st.ok()) {
            LOG(WARNING) << "SourceWindowScheduler cleanup: async handle wait failed for slot " << entry.slot_id << ": "
                         << st;
          }
        }
        auto ret_status = spb->abort_producer_slot(entry.slot_id);
        if (!ret_status.ok()) {
          LOG(WARNING) << "SourceWindowScheduler cleanup: abort_producer_slot failed for slot " << entry.slot_id << ": "
                       << ret_status;
        }
      }
      auto reset_status = spb->reset_for_new_production();
      if (!reset_status.ok()) {
        LOG(WARNING) << "SourceWindowScheduler cleanup: reset_for_new_production failed: " << reset_status;
      }
      auto release_status = spb->release();
      if (!release_status.ok()) {
        LOG(WARNING) << "SourceWindowScheduler cleanup: release failed: " << release_status;
      }
    };
    absl::Cleanup spb_cleanup = [&]() { cleanup_spb(); };

    auto wait_inflight = [&]() -> absl::Status {
      if (inflight.empty()) {
        return absl::OkStatus();
      }
      InFlight entry = std::move(inflight.front());
      inflight.pop_front();
      absl::Status result = absl::OkStatus();
      for (auto& handle : entry.handles) {
        absl::Status st = handle.wait();
        if (!st.ok() && result.ok()) {
          result = st;
        }
      }
      auto ret_status = spb->abort_producer_slot(entry.slot_id);
      if (!ret_status.ok()) {
        return ret_status;
      }
      return result;
    };

    for (const auto& window : windows) {
      auto slot_or = spb->get_free_chunk();
      if (!slot_or.ok()) {
        return slot_or.status();
      }
      const int slot_id = *slot_or;
      char* buf = spb->get_chunk_ptr(slot_id);
      if (buf == nullptr) {
        return absl::InternalError("StreamingPinnedBuffer returned null buffer");
      }

      base_read_calls += 1;
      auto read_or = sources[window.source_index]->read_at(window.src_offset, buf, static_cast<size_t>(window.length));
      if (!read_or.ok()) {
        (void)spb->abort_producer_slot(slot_id);
        return read_or.status();
      }
      if (*read_or != window.length) {
        (void)spb->abort_producer_slot(slot_id);
        return absl::DataLossError("short read while executing source-ordered window");
      }
      base_read_bytes += *read_or;

      InFlight inflight_entry{slot_id, {}};
      for (const auto& slice : window.slices) {
        const uint64_t offset_in_window = slice.src_offset - window.src_offset;
        const void* src_ptr = buf + offset_in_window;
        if (use_async) {
          auto handle_or = async_sink->write_at_async(slice.dst_offset, src_ptr, static_cast<size_t>(slice.length));
          if (!handle_or.ok()) {
            for (auto& handle : inflight_entry.handles) {
              (void)handle.wait();
            }
            (void)spb->abort_producer_slot(slot_id);
            return handle_or.status();
          }
          inflight_entry.handles.push_back(std::move(*handle_or));
        } else {
          auto st = sink.write_at(slice.dst_offset, src_ptr, static_cast<size_t>(slice.length));
          if (!st.ok()) {
            (void)spb->abort_producer_slot(slot_id);
            return st;
          }
        }
        output_bytes += slice.length;
      }

      if (use_async) {
        inflight.push_back(std::move(inflight_entry));
        if (inflight.size() >= options_.prefetch_depth) {
          auto wait_status = wait_inflight();
          if (!wait_status.ok()) {
            return wait_status;
          }
        }
      } else {
        auto ret_status = spb->abort_producer_slot(slot_id);
        if (!ret_status.ok()) {
          return ret_status;
        }
      }
    }

    while (!inflight.empty()) {
      auto wait_status = wait_inflight();
      if (!wait_status.ok()) {
        return wait_status;
      }
    }

    auto release_status = spb->release();
    if (!release_status.ok()) {
      LOG(WARNING) << "SourceWindowScheduler: initial StreamingPinnedBuffer release failed: " << release_status
                   << "; forcing reset_for_new_production() and retrying release";
      auto reset_status = spb->reset_for_new_production();
      if (!reset_status.ok()) {
        return reset_status;
      }
      release_status = spb->release();
      if (!release_status.ok()) {
        return release_status;
      }
    }
    spb_released = true;
  } else {
    for (const auto& window : windows) {
      std::vector<uint8_t> buffer(static_cast<size_t>(window.length), 0);
      base_read_calls += 1;
      auto read_or =
          sources[window.source_index]->read_at(window.src_offset, buffer.data(), static_cast<size_t>(window.length));
      if (!read_or.ok()) {
        return read_or.status();
      }
      if (*read_or != window.length) {
        return absl::DataLossError("short read while executing source-ordered window");
      }
      base_read_bytes += *read_or;

      for (const auto& slice : window.slices) {
        const uint64_t offset_in_window = slice.src_offset - window.src_offset;
        const void* src_ptr = buffer.data() + offset_in_window;
        auto st = sink.write_at(slice.dst_offset, src_ptr, static_cast<size_t>(slice.length));
        if (!st.ok()) {
          return st;
        }
        output_bytes += slice.length;
      }
    }
  }

  const double amplification =
      output_bytes == 0 ? 0.0 : static_cast<double>(base_read_bytes) / static_cast<double>(output_bytes);

  auto& handles = metrics_handles();
  record_counter(handles.base_read_calls, options_.path, base_read_calls);
  record_counter(handles.base_read_bytes, options_.path, base_read_bytes);
  record_counter(handles.output_bytes, options_.path, output_bytes);
  record_counter(handles.pad_bytes, options_.path, pad_bytes);
  record_histogram(handles.amplification_ratio, options_.path, amplification);

  return absl::OkStatus();
}

} // namespace tensorcast::store::loader
