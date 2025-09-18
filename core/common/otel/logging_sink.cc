// Copyright (c) 2025, TensorCast Team.

#include "core/common/otel/logging_sink.h"

#include <cctype>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"

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

std::mutex g_otel_sink_mu;
std::unique_ptr<OtelLogSink> g_sink;
std::string g_sink_path;

} // namespace

// Env-driven sink installer removed in final scheme

void remove_otel_log_sink() {
  std::lock_guard<std::mutex> lk(g_otel_sink_mu);
  if (!g_sink) {
    return;
  }
  absl::RemoveLogSink(g_sink.get());
  g_sink.reset();
  g_sink_path.clear();
}

void install_otel_log_sink_from_config(const Observability_Logging& log_cfg) {
  const std::string& path = log_cfg.sink_file();
  const bool enable_sink = log_cfg.otel_context_enabled() && !path.empty();

  std::lock_guard<std::mutex> lk(g_otel_sink_mu);

  if (!enable_sink) {
    if (g_sink) {
      absl::RemoveLogSink(g_sink.get());
      g_sink.reset();
      g_sink_path.clear();
      LOG(INFO) << "Removed OTel log sink";
    }
    return;
  }

  if (g_sink && g_sink_path == path) {
    return;
  }

  if (g_sink) {
    absl::RemoveLogSink(g_sink.get());
    g_sink.reset();
    g_sink_path.clear();
  }

  auto sink = std::make_unique<OtelLogSink>(path);
  absl::AddLogSink(sink.get());
  g_sink_path = path;
  g_sink = std::move(sink);
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

std::mutex g_plain_sink_mu;
std::unique_ptr<PlainFileLogSink> g_plain_sink;
std::string g_plain_path;
} // namespace

void install_plain_log_sink_from_config(const Observability_Logging& log_cfg) {
  const std::string& path = log_cfg.file();
  std::lock_guard<std::mutex> lk(g_plain_sink_mu);

  if (path.empty()) {
    if (g_plain_sink) {
      absl::RemoveLogSink(g_plain_sink.get());
      g_plain_sink.reset();
      g_plain_path.clear();
      LOG(INFO) << "Removed plain logfile sink";
    }
    return;
  }

  if (g_plain_sink && g_plain_path == path) {
    return;
  }

  if (g_plain_sink) {
    absl::RemoveLogSink(g_plain_sink.get());
    g_plain_sink.reset();
    g_plain_path.clear();
  }

  auto sink = std::make_unique<PlainFileLogSink>(path);
  absl::AddLogSink(sink.get());
  g_plain_path = path;
  g_plain_sink = std::move(sink);
  LOG(INFO) << "Installed plain logfile sink: " << path;
}

} // namespace tensorcast::common::otel
