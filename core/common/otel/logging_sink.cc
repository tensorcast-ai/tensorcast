// Copyright (c) 2025, TensorCast Team.

#include "core/common/otel/logging_sink.h"

#include <atomic>
#include <cctype>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "absl/log/log.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/strings/str_format.h"

#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/trace/context.h"

namespace tensorcast::common::otel {

namespace {

inline bool truthy(const char* v) {
  if (!v)
    return false;
  std::string s(v);
  for (auto& c : s)
    c = static_cast<char>(std::tolower(c));
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

class OtelLogSink : public absl::LogSink {
 public:
  explicit OtelLogSink(std::string path) : path_(std::move(path)) {
    file_.open(path_, std::ios::out | std::ios::app);
    if (!file_.good()) {
      // Fall back to stderr via LOG if file open fails
      LOG(WARNING) << "OtelLogSink failed to open file: " << path_;
    }
  }

  ~OtelLogSink() override {
    std::lock_guard<std::mutex> _lk(mu_);
    if (file_.is_open()) {
      file_.flush();
    }
    file_.close();
  }

  void Send(const absl::LogEntry& entry) override {
    std::string trace_hex("-");
    std::string span_hex("-");

    try {
      namespace otel = opentelemetry;
      auto ctx = otel::context::RuntimeContext::GetCurrent();
      auto span = otel::trace::GetSpan(ctx);
      // Span may be a non-recording no-op; guard for context validity if available
      // The API provides Span::GetContext() which exposes TraceId/SpanId.
      if (span) {
        auto sc = span->GetContext();
        // Convert to lowercase base16 (32/16 chars) using OTel nostd::span
        char tbuf[32]{};
        char sbuf[16]{};
        opentelemetry::nostd::span<char, 32> tspan{tbuf, sizeof(tbuf)};
        opentelemetry::nostd::span<char, 16> sspan{sbuf, sizeof(sbuf)};
        sc.trace_id().ToLowerBase16(tspan);
        sc.span_id().ToLowerBase16(sspan);
        trace_hex.assign(tbuf, sizeof(tbuf));
        span_hex.assign(sbuf, sizeof(sbuf));
      }
    } catch (...) {
      // Best-effort only
    }

    // Format: time severity thread file:line trace_id=... span_id=... message
    // Timestamp as unix nanos to ease parsing
    const auto ts = absl::FormatTime(entry.timestamp());
    const char sev = [s = entry.log_severity()]() {
      switch (s) {
        case absl::LogSeverity::kInfo:
          return 'I';
        case absl::LogSeverity::kWarning:
          return 'W';
        case absl::LogSeverity::kError:
          return 'E';
        case absl::LogSeverity::kFatal:
          return 'F';
      }
      return 'I';
    }();

    std::string line = absl::StrFormat(
        "%s %c tid=%d %s:%d trace_id=%s span_id=%s %s\n",
        ts,
        sev,
        static_cast<int>(entry.tid()),
        entry.source_basename(),
        entry.source_line(),
        trace_hex,
        span_hex,
        entry.text_message());

    std::lock_guard<std::mutex> _lk(mu_);
    if (file_.is_open()) {
      file_ << line;
      file_.flush();
    }
  }

 private:
  std::mutex mu_;
  std::string path_;
  std::ofstream file_;
};

std::atomic<bool> g_installed{false};
std::unique_ptr<OtelLogSink> g_sink;

} // namespace

void install_otel_log_sink_from_env() {
  if (g_installed.load(std::memory_order_acquire)) {
    return;
  }

  const bool enabled = truthy(std::getenv("TC_LOG_OTEL_CONTEXT_ENABLED")) ||
      (std::getenv("TC_LOG_OTEL_CONTEXT_ENABLED") == nullptr); // default on
  if (!enabled) {
    return;
  }

  const char* path = std::getenv("TC_LOG_SINK_FILE");
  if (!path || std::string(path).empty()) {
    // No file target; do not install sink
    return;
  }

  auto sink = std::make_unique<OtelLogSink>(std::string(path));
  absl::AddLogSink(sink.get());
  g_sink = std::move(sink);
  g_installed.store(true, std::memory_order_release);
  LOG(INFO) << "Installed OTel log sink (file): " << path;
}

void remove_otel_log_sink() {
  if (!g_installed.load(std::memory_order_acquire)) {
    return;
  }
  if (g_sink) {
    absl::RemoveLogSink(g_sink.get());
    g_sink.reset();
  }
  g_installed.store(false, std::memory_order_release);
}

} // namespace tensorcast::common::otel
