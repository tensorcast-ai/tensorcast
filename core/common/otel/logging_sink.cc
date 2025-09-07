// Copyright (c) 2025, TensorCast Team.

#include "core/common/otel/logging_sink.h"

#include <atomic>
#include <cctype>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/strings/str_cat.h"

#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/trace/context.h"
#include "tensorcast/config/v1/common.pb.h"

namespace tensorcast::common::otel {
using tensorcast::config::v1::Observability_Logging;

namespace {

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
    std::lock_guard<std::mutex> lk(mu_);
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
      VLOG(1) << "OTel context unavailable; skipping trace fields"; // Best-effort only
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

    std::string line = absl::StrCat(
        ts,
        " ",
        absl::string_view(&sev, 1),
        " tid=",
        static_cast<int>(entry.tid()),
        " ",
        entry.source_basename(),
        ":",
        entry.source_line(),
        " trace_id=",
        trace_hex,
        " span_id=",
        span_hex,
        " ",
        entry.text_message(),
        "\n");

    std::lock_guard<std::mutex> lk(mu_);
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

// Env-driven sink installer removed in final scheme

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

void install_otel_log_sink_from_config(const Observability_Logging& log_cfg) {
  if (g_installed.load(std::memory_order_acquire)) {
    return;
  }
  const bool enabled = log_cfg.otel_context_enabled();
  if (!enabled) {
    return;
  }
  const std::string& path = log_cfg.sink_file();
  if (path.empty()) {
    return; // no file target
  }
  auto sink = std::make_unique<OtelLogSink>(path);
  absl::AddLogSink(sink.get());
  g_sink = std::move(sink);
  g_installed.store(true, std::memory_order_release);
  LOG(INFO) << "Installed OTel log sink (file): " << path;
}

namespace {
class PlainFileLogSink : public absl::LogSink {
 public:
  explicit PlainFileLogSink(std::string path) : path_(std::move(path)) {
    file_.open(path_, std::ios::out | std::ios::app);
    if (!file_.good()) {
      LOG(WARNING) << "PlainFileLogSink failed to open file: " << path_;
    }
  }
  ~PlainFileLogSink() override {
    std::lock_guard<std::mutex> lk(mu_);
    if (file_.is_open())
      file_.flush();
    file_.close();
  }
  void Send(const absl::LogEntry& entry) override {
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
    std::string line = absl::StrCat(
        ts,
        " ",
        absl::string_view(&sev, 1),
        " tid=",
        static_cast<int>(entry.tid()),
        " ",
        entry.source_basename(),
        ":",
        entry.source_line(),
        " ",
        entry.text_message(),
        "\n");
    std::lock_guard<std::mutex> lk(mu_);
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

std::atomic<bool> g_plain_installed{false};
std::unique_ptr<PlainFileLogSink> g_plain_sink;
} // namespace

void apply_absl_log_level_from_config(const Observability_Logging& log_cfg) {
  absl::InitializeLog();
  absl::LogSeverityAtLeast min_level = absl::LogSeverityAtLeast::kInfo;
  switch (log_cfg.level()) {
    case tensorcast::config::v1::Observability::LOG_LEVEL_DEBUG: // DEBUG via VLOG
    case tensorcast::config::v1::Observability::LOG_LEVEL_INFO:
      min_level = absl::LogSeverityAtLeast::kInfo;
      break;
    case tensorcast::config::v1::Observability::LOG_LEVEL_WARN:
      min_level = absl::LogSeverityAtLeast::kWarning;
      break;
    case tensorcast::config::v1::Observability::LOG_LEVEL_ERROR:
      min_level = absl::LogSeverityAtLeast::kError;
      break;
    case tensorcast::config::v1::Observability::LOG_LEVEL_UNSPECIFIED:
    default:
      min_level = absl::LogSeverityAtLeast::kInfo;
      break;
  }
  absl::SetStderrThreshold(min_level);
  absl::SetMinLogLevel(min_level);
  if (log_cfg.vlog_level() > 0) {
    absl::SetVLogLevel("*", log_cfg.vlog_level());
  }
}

void install_plain_log_sink_from_config(const Observability_Logging& log_cfg) {
  if (g_plain_installed.load(std::memory_order_acquire))
    return;
  const std::string& path = log_cfg.file();
  if (path.empty())
    return;
  auto sink = std::make_unique<PlainFileLogSink>(path);
  absl::AddLogSink(sink.get());
  g_plain_sink = std::move(sink);
  g_plain_installed.store(true, std::memory_order_release);
  LOG(INFO) << "Installed plain logfile sink: " << path;
}

} // namespace tensorcast::common::otel
