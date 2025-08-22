// Copyright (c) 2025, StepCast Team. All rights reserved.

// All rights reserved.

#include "core/common/trace/trace_manager.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "nlohmann/json.hpp"

namespace stepcast::store {

// ----------------------------------------------------------------------
// Singleton accessor
// ----------------------------------------------------------------------

TraceManager& TraceManager::instance() {
  // Meyers' singleton: thread-safe in C++11 and later.
  static TraceManager instance;
  return instance;
}

// Thread-local storage for current request id.
thread_local std::string TraceManager::tls_request_id_;
thread_local std::string TraceManager::tls_artifact_id_;

// ---------------- Request-id helpers ------------------------------------

void TraceManager::set_current_request_id(const std::string& request_id) {
  tls_request_id_ = request_id;
}

const std::string& TraceManager::current_request_id() {
  return tls_request_id_;
}

TraceManager::RequestIdGuard::RequestIdGuard(const std::string& request_id) {
  previous_id_ = TraceManager::current_request_id();
  TraceManager::set_current_request_id(request_id);
}

TraceManager::RequestIdGuard::~RequestIdGuard() {
  TraceManager::set_current_request_id(previous_id_);
}

// ---------------- Internal helpers --------------------------------------

std::string TraceManager::make_key(const std::string& artifact_id, const std::string& request_id) {
  if (request_id.empty()) {
    return artifact_id; // Legacy key w/o request id.
  }
  return absl::StrCat(artifact_id, "|", request_id);
}

std::shared_ptr<ReplicaTrace> TraceManager::get_or_create_trace_internal(const std::string& key) {
  absl::MutexLock lock(&global_mutex_);
  auto it = traces_.find(key);
  if (it != traces_.end()) {
    return it->second;
  }
  auto ptr = std::make_shared<ReplicaTrace>();
  traces_.emplace(key, ptr);

  // Record insertion order for eviction.
  insertion_order_.push_back(key);
  if (insertion_order_.size() > kMaxTraces) {
    // Copy the key before popping to avoid dangling reference.
    std::string oldest_key = insertion_order_.front();
    insertion_order_.pop_front();
    traces_.erase(oldest_key);
  }

  return ptr;
}

// ---------------- Span management ---------------------------------------

TraceManager::SpanId TraceManager::begin_span(
    const std::string& artifact_id,
    const std::string& request_id,
    const std::string& stage) {
  return begin_span(artifact_id, request_id, stage, nullptr);
}

TraceManager::SpanId TraceManager::begin_span(
    const std::string& artifact_id,
    const std::string& request_id,
    const std::string& stage,
    void* cuda_stream) {
  const std::string key = make_key(artifact_id, request_id);
  auto trace = get_or_create_trace_internal(key);
  absl::MutexLock lock(&trace->m);
  Span span{stage, absl::Now(), absl::ZeroDuration(), std::this_thread::get_id(), cuda_stream};
  trace->spans.push_back(span);
  return trace->spans.size() - 1;
}

void TraceManager::end_span(const std::string& artifact_id, const std::string& request_id, SpanId span_id) {
  const std::string key = make_key(artifact_id, request_id);
  auto trace = get_or_create_trace_internal(key);
  absl::MutexLock lock(&trace->m);
  if (span_id >= trace->spans.size()) {
    LOG(WARNING) << "TraceManager::end_span invalid span id " << span_id;
    return;
  }
  Span& span = trace->spans[static_cast<size_t>(span_id)];
  if (span.duration == absl::ZeroDuration()) {
    span.duration = absl::Now() - span.t_start;
  }
}

void TraceManager::dump_summary(const std::string& artifact_id, const std::string& request_id, std::ostream& os) {
  const std::string key = make_key(artifact_id, request_id);
  auto trace = get_or_create_trace_internal(key);
  absl::MutexLock lock(&trace->m);

  if (trace->spans.empty()) {
    os << "No trace data for replica: " << artifact_id << " (request_id=" << request_id << ")\n";
    return;
  }

  struct Aggregate {
    int calls = 0;
    absl::Duration total = absl::ZeroDuration();
  };

  absl::flat_hash_map<std::string, Aggregate> agg;
  for (const auto& span : trace->spans) {
    if (span.duration == absl::ZeroDuration()) {
      continue;
    }
    auto& a = agg[span.stage];
    a.calls++;
    a.total += span.duration;
  }

  // Column widths
  size_t stage_w = 24;
  for (const auto& [stage, _] : agg) {
    stage_w = std::max(stage_w, stage.size() + 2);
  }
  const int calls_w = 10;
  const int total_w = 12;
  const int avg_w = 11;

  os << "[TRACE] " << artifact_id << " (req=" << request_id << ") SUMMARY\n";
  os << std::left << std::setw(static_cast<int>(stage_w)) << "stage" << std::right << std::setw(calls_w) << "calls"
     << std::setw(total_w) << "total(ms)" << std::setw(avg_w) << "avg(ms)" << "\n";
  os << std::string(stage_w + calls_w + total_w + avg_w, '-') << "\n";

  // Track the earliest start time and latest end time across all completed spans
  absl::Time earliest_start = absl::InfiniteFuture();
  absl::Time latest_end = absl::InfinitePast();

  for (const auto& [stage, a] : agg) {
    double total_ms = absl::ToDoubleMilliseconds(a.total);
    double avg_ms = total_ms / a.calls;
    os << std::left << std::setw(static_cast<int>(stage_w)) << stage << std::right << std::setw(calls_w) << a.calls
       << std::setw(total_w) << std::fixed << std::setprecision(1) << total_ms << std::setw(avg_w) << std::fixed
       << std::setprecision(1) << avg_ms << "\n";
  }

  // A second pass to determine the true TOTAL duration (outermost span)
  for (const auto& span : trace->spans) {
    if (span.duration == absl::ZeroDuration()) {
      continue; // Skip unfinished spans
    }
    earliest_start = std::min(earliest_start, span.t_start);
    latest_end = std::max(latest_end, span.t_start + span.duration);
  }

  absl::Duration total_all = absl::ZeroDuration();
  if (latest_end > earliest_start) {
    total_all = latest_end - earliest_start;
  }

  os << std::string(stage_w + calls_w + total_w + avg_w, '-') << "\n";
  os << std::left << std::setw(static_cast<int>(stage_w + calls_w)) << "TOTAL" << std::right << std::setw(total_w)
     << std::fixed << std::setprecision(0) << absl::ToDoubleMilliseconds(total_all) << " ms\n";
}

std::string TraceManager::generate_chrome_trace(const std::string& artifact_id, const std::string& request_id) {
  const std::string key = make_key(artifact_id, request_id);
  auto trace = get_or_create_trace_internal(key);
  absl::MutexLock lock(&trace->m);

  if (trace->spans.empty()) {
    return "[]";
  }

  // Find the earliest start time to use as the reference point
  absl::Time reference_time = absl::InfiniteFuture();
  for (const auto& span : trace->spans) {
    if (span.duration != absl::ZeroDuration()) {
      reference_time = std::min(reference_time, span.t_start);
    }
  }

  if (reference_time == absl::InfiniteFuture()) {
    return "[]";
  }

  nlohmann::json events = nlohmann::json::array();

  for (const auto& span : trace->spans) {
    if (span.duration == absl::ZeroDuration()) {
      continue; // Skip unfinished spans
    }

    // Convert times to microseconds relative to reference time
    double ts_us = absl::ToDoubleMicroseconds(span.t_start - reference_time);
    double dur_us = absl::ToDoubleMicroseconds(span.duration);

    // Determine thread ID for visualization:
    // - If this is a CUDA operation (cuda_stream != nullptr), use stream pointer
    // - Otherwise use the actual thread ID
    std::string tid_str;
    if (span.cuda_stream != nullptr) {
      // Use CUDA stream pointer as "thread" ID for better visualization
      std::ostringstream stream_id;
      stream_id << "cuda_stream_" << span.cuda_stream;
      tid_str = stream_id.str();
    } else {
      // Regular CPU thread ID
      std::ostringstream tid_stream;
      tid_stream << span.tid;
      tid_str = tid_stream.str();
    }

    nlohmann::json event;
    event["name"] = span.stage;
    event["cat"] = "artifact"; // Category
    event["ph"] = "X"; // Complete event
    event["ts"] = ts_us;
    event["dur"] = dur_us;
    event["tid"] = tid_str;
    event["pid"] = 1; // Process ID (fixed)

    // Add additional args for context
    event["args"] = {{"artifact_id", artifact_id}, {"request_id", request_id}};

    // Add CUDA stream info if available
    if (span.cuda_stream != nullptr) {
      std::ostringstream stream_ptr;
      stream_ptr << span.cuda_stream;
      event["args"]["cuda_stream"] = stream_ptr.str();
    }

    events.push_back(event);
  }

  return events.dump(2); // Pretty print with 2-space indent
}

void TraceManager::clear_trace(const std::string& artifact_id, const std::string& request_id) {
  const std::string key = make_key(artifact_id, request_id);
  absl::MutexLock lock(&global_mutex_);
  traces_.erase(key);
  // Keep insertion_order_ consistent by removing the key if present.
  auto it = std::find(insertion_order_.begin(), insertion_order_.end(), key);
  if (it != insertion_order_.end()) {
    insertion_order_.erase(it);
  }
}

} // namespace stepcast::store