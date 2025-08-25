// Copyright (c) 2025, StepCast Team. All rights reserved.

// All rights reserved.
//
// Trace framework for measuring replica loading pipeline latency.
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

struct ReplicaTrace {
  // All spans recorded for the replica.
  std::vector<Span> spans ABSL_GUARDED_BY(m);
  // Mutex to protect access to the spans vector.
  absl::Mutex m;
};

// TraceManager is a thread-safe singleton that manages spans for all artifacts/replicas.
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
  // multiple concurrent loads of the same replica can be distinguished.
  SpanId begin_span(const std::string& artifact_id, const std::string& request_id, const std::string& stage);

  // Begin span with CUDA stream information for better visualization.
  SpanId begin_span(
      const std::string& artifact_id,
      const std::string& request_id,
      const std::string& stage,
      void* cuda_stream);

  void end_span(const std::string& artifact_id, const std::string& request_id, SpanId span_id);

  // Dump a human-readable summary for a particular (artifact_id, request_id)
  // pair.
  void dump_summary(const std::string& artifact_id, const std::string& request_id, std::ostream& os);

  // Generate Chrome Trace format JSON for a particular (artifact_id, request_id) pair.
  // Returns the JSON string that can be loaded in chrome://tracing.
  std::string generate_chrome_trace(const std::string& artifact_id, const std::string& request_id);

  // Erase all trace data for the specified (artifact_id, request_id).  This
  // helps keep memory usage bounded in long-running processes.
  void clear_trace(const std::string& artifact_id, const std::string& request_id);

  // Helper: retrieves trace using key (artifact_id only).
  std::shared_ptr<ReplicaTrace> get_or_create_artifact_trace(const std::string& artifact_id);

  // -----------------------------------------------------------------
  // artifact-id helpers (mirrors request-id helpers)
  // -----------------------------------------------------------------
  static void set_current_artifact_id(const std::string& artifact_id) {
    tls_artifact_id_ = artifact_id;
  }
  static const std::string& current_artifact_id() {
    return tls_artifact_id_;
  }

  class ArtifactIdGuard {
   public:
    explicit ArtifactIdGuard(const std::string& artifact_id) : previous_id_(TraceManager::current_artifact_id()) {
      TraceManager::set_current_artifact_id(artifact_id);
    }
    ~ArtifactIdGuard() {
      TraceManager::set_current_artifact_id(previous_id_);
    }

    ArtifactIdGuard(const ArtifactIdGuard&) = delete;
    ArtifactIdGuard& operator=(const ArtifactIdGuard&) = delete;

   private:
    std::string previous_id_;
  };

 private:
  TraceManager() = default;

  // Helper that returns (artifact_id|request_id) combined key for the map.
  static std::string make_key(const std::string& artifact_id, const std::string& request_id);

  std::shared_ptr<ReplicaTrace> get_or_create_trace_internal(const std::string& key);

  // Thread-local request id (empty when no context active).
  static thread_local std::string tls_request_id_;

  // Thread-local current replica identifier used for implicit propagation.
  static thread_local std::string tls_artifact_id_;

  // Protects access to the traces_ map itself. Individual ReplicaTrace objects
  // have their own mutex for fine-grained locking.
  absl::Mutex global_mutex_;
  absl::flat_hash_map<std::string, std::shared_ptr<ReplicaTrace>> traces_ ABSL_GUARDED_BY(global_mutex_);

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
      const std::string& artifact_id,
      const std::string& request_id = TraceManager::current_request_id())
      : artifact_id_(artifact_id), request_id_(request_id) {}
  ~TraceSummaryGuard() {
    std::ostringstream oss;
    TraceManager::instance().dump_summary(artifact_id_, request_id_, oss);
    if (!oss.str().empty()) {
      LOG(INFO) << "[TraceSummary] " << oss.str();
    }

    // Check if Chrome Trace output is enabled via environment variable
    const char* trace_output_dir = std::getenv("SC_TRACE_OUTPUT_DIR");
    if (trace_output_dir != nullptr && trace_output_dir[0] != '\0') {
      try {
        std::string chrome_trace_json = TraceManager::instance().generate_chrome_trace(artifact_id_, request_id_);

        // Generate filename: artifact_id+request_id.json
        std::string filename = artifact_id_;
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

    // Clear the trace data for this (artifact_id, request_id) pair to avoid
    // holding unbounded history once the summary has been emitted.
    TraceManager::instance().clear_trace(artifact_id_, request_id_);
  }

  TraceSummaryGuard(const TraceSummaryGuard&) = delete;
  TraceSummaryGuard& operator=(const TraceSummaryGuard&) = delete;

 private:
  std::string artifact_id_;
  std::string request_id_;
};

} // namespace stepcast::store