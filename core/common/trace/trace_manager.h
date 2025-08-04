// Copyright (c) 2025, StepCast Team. All rights reserved.

// All rights reserved.
//
// Trace framework for measuring model loading pipeline latency.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace stepcast::store {

// A single timing span for a particular stage.
struct Span {
  std::string stage; // Stage name.
  absl::Time t_start; // Start timestamp.
  absl::Duration duration; // Duration (populated in end_span).
  std::thread::id tid; // Thread id that created the span.

  // CUDA stream pointer (nullptr if not a CUDA operation).
  // We store the raw pointer value for identification purposes only.
  void* cuda_stream = nullptr;
};

struct ModelTrace {
  // All spans recorded for the model.
  std::vector<Span> spans ABSL_GUARDED_BY(m);
  // Mutex to protect access to the spans vector.
  absl::Mutex m;
};

// TraceManager is a thread-safe singleton that manages spans for all models.
class TraceManager {
 public:
  using SpanId = uint64_t;

  // Access the global instance.
  static TraceManager& instance();

  // ---------------------------------------------------------------------
  // Request-scoped helpers ------------------------------------------------
  // ---------------------------------------------------------------------

  // Sets the thread-local request id for all subsequent TraceScope
  // instances created on the same thread.  Use the RequestIdGuard helper
  // for RAII style management instead of calling this directly.
  static void set_current_request_id(const std::string& request_id);

  // Returns the current thread-local request id (may be empty when no
  // request context is active).
  static const std::string& current_request_id();

  // RAII helper that sets the current request id for the lifetime of the
  // guard on the current thread.
  class RequestIdGuard {
   public:
    explicit RequestIdGuard(const std::string& request_id);
    RequestIdGuard(const RequestIdGuard&) = delete;
    RequestIdGuard& operator=(const RequestIdGuard&) = delete;
    ~RequestIdGuard();

   private:
    std::string previous_id_;
  };

  // ---------------------------------------------------------------------
  // Span management with request id --------------------------------------
  // ---------------------------------------------------------------------

  // Begin/end span variants that carry an explicit request id so that
  // multiple concurrent loads of the same model can be distinguished.
  SpanId begin_span(const std::string& model_id, const std::string& request_id, const std::string& stage);

  // Begin span with CUDA stream information for better visualization.
  SpanId begin_span(
      const std::string& model_id,
      const std::string& request_id,
      const std::string& stage,
      void* cuda_stream);

  void end_span(const std::string& model_id, const std::string& request_id, SpanId span_id);

  // Dump a human-readable summary for a particular (model_id, request_id)
  // pair.
  void dump_summary(const std::string& model_id, const std::string& request_id, std::ostream& os);

  // Generate Chrome Trace format JSON for a particular (model_id, request_id) pair.
  // Returns the JSON string that can be loaded in chrome://tracing.
  std::string generate_chrome_trace(const std::string& model_id, const std::string& request_id);

  // Erase all trace data for the specified (model_id, request_id).  This
  // helps keep memory usage bounded in long-running processes.
  void clear_trace(const std::string& model_id, const std::string& request_id);

  // Back-compat overloads that operate without a request id (treated as
  // an empty string internally).
  SpanId begin_span(const std::string& model_id, const std::string& stage) {
    return begin_span(model_id, /*request_id=*/"", stage);
  }
  void end_span(const std::string& model_id, SpanId span_id) {
    end_span(model_id, /*request_id=*/"", span_id);
  }
  void dump_summary(const std::string& model_id, std::ostream& os) {
    dump_summary(model_id, /*request_id=*/"", os);
  }

  // Backward-compat helper: retrieves trace using legacy key (model_id only).
  std::shared_ptr<ModelTrace> get_or_create_model_trace(const std::string& model_id);

  // -----------------------------------------------------------------
  // Model-id helpers (mirrors request-id helpers)
  // -----------------------------------------------------------------
  static void set_current_model_id(const std::string& model_id) {
    tls_model_id_ = model_id;
  }
  static const std::string& current_model_id() {
    return tls_model_id_;
  }

  class ModelIdGuard {
   public:
    explicit ModelIdGuard(const std::string& model_id) : previous_id_(TraceManager::current_model_id()) {
      TraceManager::set_current_model_id(model_id);
    }
    ~ModelIdGuard() {
      TraceManager::set_current_model_id(previous_id_);
    }

    ModelIdGuard(const ModelIdGuard&) = delete;
    ModelIdGuard& operator=(const ModelIdGuard&) = delete;

   private:
    std::string previous_id_;
  };

 private:
  TraceManager() = default;

  // Helper that returns (model_id|request_id) combined key for the map.
  static std::string make_key(const std::string& model_id, const std::string& request_id);

  std::shared_ptr<ModelTrace> get_or_create_model_trace_internal(const std::string& key);

  // Thread-local request id (empty when no context active).
  static thread_local std::string tls_request_id_;

  // Thread-local current model identifier used for implicit propagation.
  static thread_local std::string tls_model_id_;

  // Protects access to the traces_ map itself. Individual ModelTrace objects
  // have their own mutex for fine-grained locking.
  absl::Mutex global_mutex_;
  absl::flat_hash_map<std::string, std::shared_ptr<ModelTrace>> traces_ ABSL_GUARDED_BY(global_mutex_);

  // Max number of trace entries kept in memory.
  static constexpr size_t kMaxTraces = 256;

  // Maintain insertion order for simple FIFO eviction.
  std::deque<std::string> insertion_order_ ABSL_GUARDED_BY(global_mutex_);
};

// ---------------------------------------------------------------------
// TraceSummaryGuard: RAII summary dumper --------------------------------
// ---------------------------------------------------------------------

class TraceSummaryGuard {
 public:
  explicit TraceSummaryGuard(
      const std::string& model_id,
      const std::string& request_id = TraceManager::current_request_id())
      : model_id_(model_id), request_id_(request_id) {}
  ~TraceSummaryGuard() {
    std::ostringstream oss;
    TraceManager::instance().dump_summary(model_id_, request_id_, oss);
    if (!oss.str().empty()) {
      LOG(INFO) << "[TraceSummary] " << oss.str();
    }

    // Check if Chrome Trace output is enabled via environment variable
    const char* trace_output_dir = std::getenv("SC_TRACE_OUTPUT_DIR");
    if (trace_output_dir != nullptr && trace_output_dir[0] != '\0') {
      try {
        std::string chrome_trace_json = TraceManager::instance().generate_chrome_trace(model_id_, request_id_);

        // Generate filename: model_id+request_id.json
        std::string filename = model_id_;
        if (!request_id_.empty()) {
          filename += "+" + request_id_;
        }

        // Clean filename: replace path separators and other special chars with underscore
        for (char& c : filename) {
          if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' ||
              c == '|') {
            c = '_';
          }
        }

        filename += ".json";

        // Build full path
        std::string full_path = std::string(trace_output_dir) + "/" + filename;

        // Write to file
        std::ofstream trace_file(full_path);
        if (trace_file.is_open()) {
          trace_file << chrome_trace_json;
          trace_file.close();
          LOG(INFO) << "[TraceSummary] Chrome trace saved to: " << full_path;
        } else {
          LOG(WARNING) << "[TraceSummary] Failed to open file for writing: " << full_path;
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "[TraceSummary] Failed to save Chrome trace: " << e.what();
      }
    }

    // Clear the trace data for this (model_id, request_id) pair to avoid
    // holding unbounded history once the summary has been emitted.
    TraceManager::instance().clear_trace(model_id_, request_id_);
  }

  TraceSummaryGuard(const TraceSummaryGuard&) = delete;
  TraceSummaryGuard& operator=(const TraceSummaryGuard&) = delete;

 private:
  std::string model_id_;
  std::string request_id_;
};

} // namespace stepcast::store