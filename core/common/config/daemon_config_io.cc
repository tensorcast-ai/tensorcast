// Copyright (c) 2025, TensorCast Team.

#include "core/common/config/daemon_config_io.h"

#include <cctype>
#include <charconv>
#include <cerrno>
#include <fstream>
#include <optional>
#include <sstream>
#include <cstdlib>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "google/protobuf/util/json_util.h"
#include "nlohmann/json.hpp"
#include "yaml-cpp/yaml.h"

#include "core/communicator/config_io.h"

namespace tcfg = tensorcast::config::v1;

namespace tensorcast::common::config {

namespace {

// Convert a YAML::Node tree into nlohmann::json for uniform JsonStringToMessage parsing.
nlohmann::json yaml_node_to_json(const YAML::Node& node) {
  using nlohmann::json;
  switch (node.Type()) {
    case YAML::NodeType::Null:
      return json(nullptr);
    case YAML::NodeType::Scalar: {
      const std::string s = node.as<std::string>();
      if (s == "true" || s == "True")
        return json(true);
      if (s == "false" || s == "False")
        return json(false);
      // Only treat as number if the entire scalar parses as an integer or a floating value.
      // This avoids mis-parsing strings like IPv4 addresses (e.g., "127.0.0.1").
      if (!s.empty() && (std::isdigit(s[0]) || s[0] == '-' || s[0] == '+')) {
        // Try integer via from_chars with full consumption
        int64_t iv = 0;
        const char* begin = s.data();
        const char* end = s.data() + s.size();
        auto [ptr, ec] = std::from_chars(begin, end, iv);
        if (ec == std::errc() && ptr == end) {
          return json(iv);
        }
        // Try double via strtod with full consumption
        char* endptr = nullptr;
        errno = 0;
        double dv = std::strtod(s.c_str(), &endptr);
        if (errno == 0 && endptr == s.c_str() + s.size()) {
          return json(dv);
        }
      }
      return json(s);
    }
    case YAML::NodeType::Sequence: {
      nlohmann::json arr = nlohmann::json::array();
      for (const auto& it : node)
        arr.push_back(yaml_node_to_json(it));
      return arr;
    }
    case YAML::NodeType::Map: {
      nlohmann::json obj = nlohmann::json::object();
      for (const auto& it : node)
        obj[it.first.as<std::string>()] = yaml_node_to_json(it.second);
      return obj;
    }
    case YAML::NodeType::Undefined:
    default:
      return nlohmann::json(nullptr);
  }
}

inline bool has_ext(const std::string& path, const char* ext) {
  return absl::EndsWithIgnoreCase(path, ext);
}

std::optional<uint64_t> parse_size_bytes(const std::string& s) {
  // Accept plain integer or integer with KB/MB/GB/TB suffix (case-insensitive)
  std::string t = absl::AsciiStrToLower(absl::StripAsciiWhitespace(s));
  // remove optional underscores and spaces
  t = absl::StrReplaceAll(t, {{"_", ""}, {" ", ""}});
  uint64_t mul = 1;
  if (absl::EndsWith(t, "kb")) {
    mul = 1024ULL;
    t.resize(t.size() - 2);
  } else if (absl::EndsWith(t, "mb")) {
    mul = 1024ULL * 1024ULL;
    t.resize(t.size() - 2);
  } else if (absl::EndsWith(t, "gb")) {
    mul = 1024ULL * 1024ULL * 1024ULL;
    t.resize(t.size() - 2);
  } else if (absl::EndsWith(t, "tb")) {
    mul = 1024ULL * 1024ULL * 1024ULL * 1024ULL;
    t.resize(t.size() - 2);
  }
  // numeric?
  uint64_t base = 0;
  auto [ptr, ec] = std::from_chars(t.data(), t.data() + t.size(), base);
  if (ec != std::errc() || ptr != t.data() + t.size())
    return std::nullopt;
  return base * mul;
}

// Map convenient strings to fULLy-qualified enum names expected by Protobuf JSON.
void normalize_enum_aliases(nlohmann::json& root) {
  // logging.level: INFO/WARN/ERROR/DEBUG
  if (root.contains("observability") && root["observability"].is_object()) {
    auto& obs = root["observability"];
    if (obs.contains("logging") && obs["logging"].is_object()) {
      auto& log = obs["logging"];
      if (log.contains("level") && log["level"].is_string()) {
        std::string lvl = log["level"].get<std::string>();
        std::string upper = absl::AsciiStrToUpper(absl::StripAsciiWhitespace(lvl));
        if (upper == "DEBUG")
          log["level"] = "LOG_LEVEL_DEBUG";
        else if (upper == "INFO")
          log["level"] = "LOG_LEVEL_INFO";
        else if (upper == "WARN" || upper == "WARNING")
          log["level"] = "LOG_LEVEL_WARN";
        else if (upper == "ERROR")
          log["level"] = "LOG_LEVEL_ERROR";
      }
    }
    if (obs.contains("otel") && obs["otel"].is_object()) {
      auto& otel = obs["otel"];
      if (otel.contains("exporter_protocol") && otel["exporter_protocol"].is_string()) {
        std::string p = absl::AsciiStrToLower(otel["exporter_protocol"].get<std::string>());
        if (p == "grpc")
          otel["exporter_protocol"] = "O_TEL_PROTOCOL_GRPC";
        else if (p == "http/protobuf" || p == "http_protobuf")
          otel["exporter_protocol"] = "O_TEL_PROTOCOL_HTTP_PROTOBUF";
      }
    }
  }
  // compatibility.verification_timeout_status: ok|deadline
  if (root.contains("compatibility") && root["compatibility"].is_object()) {
    auto& c = root["compatibility"];
    if (c.contains("verification_timeout_status") && c["verification_timeout_status"].is_string()) {
      std::string v = absl::AsciiStrToLower(c["verification_timeout_status"].get<std::string>());
      if (v == "ok")
        c["verification_timeout_status"] = "VERIFICATION_TIMEOUT_STATUS_OK";
      else if (v == "deadline")
        c["verification_timeout_status"] = "VERIFICATION_TIMEOUT_STATUS_DEADLINE";
    }
  }
}

// Convert size-like string fields to numeric bytes in-place on JSON tree
void normalize_size_fields(nlohmann::json& root) {
  auto to_bytes = [](nlohmann::json& n) {
    if (n.is_string()) {
      if (auto v = parse_size_bytes(n.get<std::string>()); v.has_value()) {
        n = *v;
      }
    }
  };

  // engine.* bytes
  if (root.contains("engine") && root["engine"].is_object()) {
    auto& e = root["engine"];
    if (e.contains("mem_pool_size_bytes"))
      to_bytes(e["mem_pool_size_bytes"]);
    if (e.contains("chunk_bytes"))
      to_bytes(e["chunk_bytes"]);
    if (e.contains("dvmp_chunk_size_bytes"))
      to_bytes(e["dvmp_chunk_size_bytes"]);
  }
  // checkpoint.streaming.* bytes
  if (root.contains("checkpoint") && root["checkpoint"].is_object()) {
    auto& cp = root["checkpoint"];
    if (cp.contains("streaming") && cp["streaming"].is_object()) {
      auto& st = cp["streaming"];
      if (st.contains("io_chunk_bytes"))
        to_bytes(st["io_chunk_bytes"]);
      if (st.contains("pinned_pool_bytes"))
        to_bytes(st["pinned_pool_bytes"]);
    }
  }
}

} // namespace

void normalize_defaults(tcfg::DaemonConfig* cfg) {
  if (!cfg)
    return;

  // Engine defaults
  auto* e = cfg->mutable_engine();
  if (e->mem_pool_size_bytes() == 0)
    e->set_mem_pool_size_bytes(8ULL * 1024 * 1024 * 1024);
  if (e->chunk_bytes() == 0)
    e->set_chunk_bytes(256ULL * 1024 * 1024);
  if (e->dvmp_chunk_size_bytes() == 0)
    e->set_dvmp_chunk_size_bytes(256ULL * 1024 * 1024);
  if (e->streaming_buffer_max_concurrent_sessions() == 0)
    e->set_streaming_buffer_max_concurrent_sessions(1);

  // Lifecycle defaults (durations left at 0s unless specified)
  auto* lf = cfg->mutable_lifecycle();
  if (lf->gpu_memory_limit_fraction() <= 0.0)
    lf->set_gpu_memory_limit_fraction(0.75);
  if (!lf->has_eviction_loop_interval()) {
    auto* d = lf->mutable_eviction_loop_interval();
    d->set_seconds(1);
    d->set_nanos(0);
  }

  // Observability defaults
  auto* obs = cfg->mutable_observability();
  auto* log = obs->mutable_logging();
  if (log->level() == tcfg::Observability::LOG_LEVEL_UNSPECIFIED)
    log->set_level(tcfg::Observability::LOG_LEVEL_INFO);

  // Communicator defaults via existing helper
  tensorcast::communicator::normalize_defaults(cfg->mutable_communicator());
}

absl::StatusOr<tcfg::DaemonConfig> load_daemon_config_from_file(const std::string& path) {
  tcfg::DaemonConfig cfg;

  std::ifstream in(path);
  if (!in.good()) {
    return absl::NotFoundError(absl::StrCat("Config file not found: ", path));
  }

  std::ostringstream oss;
  oss << in.rdbuf();
  const std::string content = oss.str();

  nlohmann::json root_json;
  try {
    if (has_ext(path, ".yaml") || has_ext(path, ".yml")) {
      YAML::Node root = YAML::Load(content);
      root_json = yaml_node_to_json(root);
    } else {
      root_json = nlohmann::json::parse(content);
    }
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Parse error: ", e.what()));
  }

  // Normalize before protobuf parsing
  normalize_enum_aliases(root_json);
  normalize_size_fields(root_json);

  const std::string json_text = root_json.dump();

  google::protobuf::util::JsonParseOptions opts;
  opts.ignore_unknown_fields = false; // strict
  auto status = google::protobuf::util::JsonStringToMessage(json_text, &cfg, opts);
  if (!status.ok()) {
    return absl::InvalidArgumentError(absl::StrCat("JSON->Proto parse error: ", status.message()));
  }

  // Apply numeric/time defaults and communicator defaults
  normalize_defaults(&cfg);
  return cfg;
}

} // namespace tensorcast::common::config
