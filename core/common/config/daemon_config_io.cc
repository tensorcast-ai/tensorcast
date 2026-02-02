// Copyright (c) 2025-2026, TensorCast Team.

#include "core/common/config/daemon_config_io.h"

#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "google/protobuf/util/json_util.h"
#include "nlohmann/json.hpp"
#include "yaml-cpp/yaml.h"

#include "core/communicator/config_io.h"

namespace tcfg = tensorcast::config::v1;

namespace tensorcast::common::config {

namespace {

// Convert a YAML::Node tree into nlohmann::json for uniform JsonStringToMessage parsing.
nlohmann::json yaml_node_to_json(const YAML::Node& node, std::string_view key = {}) {
  using nlohmann::json;
  switch (node.Type()) {
    case YAML::NodeType::Null:
      return json(nullptr);
    case YAML::NodeType::Scalar: {
      const std::string s = node.as<std::string>();
      if (key == "sampler_arg") {
        return json(s);
      }
      if (s == "true" || s == "True")
        return json(true);
      if (s == "false" || s == "False")
        return json(false);
      // Only treat as number if the entire scalar parses as an integer or a floating value.
      // This avoids misparsing strings like IPv4 addresses (e.g., "127.0.0.1").
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
      for (const auto& it : node) {
        const std::string k = it.first.as<std::string>();
        obj[k] = yaml_node_to_json(it.second, k);
      }
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

std::optional<std::string> parse_duration_proto(const std::string& s) {
  // Accept plain integer or number with ms/s/m/h suffix and return canonical
  // protobuf JSON duration string (e.g., "0.5s").
  std::string t = absl::AsciiStrToLower(absl::StripAsciiWhitespace(s));
  t = absl::StrReplaceAll(t, {{"_", ""}, {" ", ""}});
  double mul = 1.0;
  if (absl::EndsWith(t, "ms")) {
    mul = 1.0 / 1000.0;
    t.resize(t.size() - 2);
  } else if (absl::EndsWith(t, "s")) {
    mul = 1.0;
    t.resize(t.size() - 1);
  } else if (absl::EndsWith(t, "m")) {
    mul = 60.0;
    t.resize(t.size() - 1);
  } else if (absl::EndsWith(t, "h")) {
    mul = 3600.0;
    t.resize(t.size() - 1);
  } else {
    return std::nullopt;
  }
  errno = 0;
  char* endptr = nullptr;
  double base = std::strtod(t.c_str(), &endptr);
  if (errno != 0 || endptr != t.c_str() + t.size())
    return std::nullopt;
  double seconds = base * mul;
  std::string formatted = absl::StrFormat("%.9f", seconds);
  while (formatted.find('.') != std::string::npos && formatted.back() == '0')
    formatted.pop_back();
  if (!formatted.empty() && formatted.back() == '.')
    formatted.pop_back();
  return absl::StrCat(formatted, "s");
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

  // engine.* bytes (final canonical names only)
  if (root.contains("engine") && root["engine"].is_object()) {
    auto& e = root["engine"];
    if (e.contains("artifact_chunk_bytes"))
      to_bytes(e["artifact_chunk_bytes"]);
    if (e.contains("memory_tiers") && e["memory_tiers"].is_object()) {
      auto& mt = e["memory_tiers"];
      if (mt.contains("stable_bytes"))
        to_bytes(mt["stable_bytes"]);
      if (mt.contains("preemptible_limit_bytes"))
        to_bytes(mt["preemptible_limit_bytes"]);
    }
  }

  if (root.contains("pinned_memory") && root["pinned_memory"].is_object()) {
    auto& pm = root["pinned_memory"];
    if (pm.contains("classes") && pm["classes"].is_array()) {
      for (auto& cls : pm["classes"]) {
        if (!cls.is_object())
          continue;
        if (cls.contains("slice_bytes"))
          to_bytes(cls["slice_bytes"]);
        if (cls.contains("pool_bytes"))
          to_bytes(cls["pool_bytes"]);
      }
    }
  }
}

// Convert duration-like string fields (e.g., "500ms", "2m") into canonical
// protobuf JSON duration strings before parsing.
void normalize_duration_fields(nlohmann::json& root) {
  auto to_duration = [](nlohmann::json& n) {
    if (n.is_string()) {
      if (auto v = parse_duration_proto(n.get<std::string>()); v.has_value()) {
        n = *v;
      }
    }
  };

  if (root.contains("pinned_memory") && root["pinned_memory"].is_object()) {
    auto& pm = root["pinned_memory"];
    if (pm.contains("allocation_timeout"))
      to_duration(pm["allocation_timeout"]);
  }

  if (root.contains("server") && root["server"].is_object()) {
    auto& srv = root["server"];
    if (srv.contains("grpc") && srv["grpc"].is_object()) {
      auto& g = srv["grpc"];
      if (g.contains("keepalive_time"))
        to_duration(g["keepalive_time"]);
      if (g.contains("keepalive_timeout"))
        to_duration(g["keepalive_timeout"]);
      if (g.contains("max_connection_idle"))
        to_duration(g["max_connection_idle"]);
      if (g.contains("max_connection_age"))
        to_duration(g["max_connection_age"]);
    }
  }

  if (root.contains("lifecycle") && root["lifecycle"].is_object()) {
    auto& lf = root["lifecycle"];
    const char* fields[] = {
        "eviction_check_interval",
        "proc_check_interval",
        "sessions_sweep_interval",
        "locks_sweep_interval",
        "verification_sweep_interval",
        "sessions_ttl",
        "locks_ttl",
        "eviction_loop_interval"};
    for (const char* f : fields) {
      if (lf.contains(f))
        to_duration(lf[f]);
    }
    if (lf.contains("handle_leases") && lf["handle_leases"].is_object()) {
      auto& hl = lf["handle_leases"];
      if (hl.contains("ttl")) {
        to_duration(hl["ttl"]);
      }
    }
  }

  if (root.contains("high_availability") && root["high_availability"].is_object()) {
    auto& ha = root["high_availability"];
    const char* ha_fields[] = {
        "heartbeat_interval",
        "periodic_sync_interval",
        "registration_retry_delay",
        "heartbeat_rpc_timeout",
        "state_sync_rpc_timeout",
        "full_sync_rpc_timeout"};
    for (const char* f : ha_fields) {
      if (ha.contains(f))
        to_duration(ha[f]);
    }
  }

  if (root.contains("retention_handles") && root["retention_handles"].is_object()) {
    auto& rh = root["retention_handles"];
    if (rh.contains("default_ttl"))
      to_duration(rh["default_ttl"]);
    if (rh.contains("max_ttl"))
      to_duration(rh["max_ttl"]);
  }
}

} // namespace

void normalize_defaults(tcfg::DaemonConfig* cfg) {
  if (!cfg)
    return;

  // Engine defaults
  auto* e = cfg->mutable_engine();
  if (e->artifact_chunk_bytes() == 0)
    e->set_artifact_chunk_bytes(256ULL * 1024 * 1024);
  if (e->streaming_buffer_chunks() == 0)
    e->set_streaming_buffer_chunks(16);
  if (e->has_memory_tiers()) {
    auto* mt = e->mutable_memory_tiers();
    if (mt->preemptible_low_watermark_ratio() <= 0.0) {
      mt->set_preemptible_low_watermark_ratio(0.4);
    }
  }
  if (!e->has_byte_mapping()) {
    e->mutable_byte_mapping();
  }
  auto* bm = e->mutable_byte_mapping();
  if (!bm->has_enable_strided_execution()) {
    bm->set_enable_strided_execution(true);
  }
  if (!bm->has_enable_direct_write_at()) {
    bm->set_enable_direct_write_at(true);
  }
  if (bm->program_cache_entries() == 0) {
    bm->set_program_cache_entries(256);
  }
  if (bm->strided_run_min_ranges() == 0) {
    bm->set_strided_run_min_ranges(128);
  }
  if (bm->strided_min_row_len_bytes() == 0) {
    bm->set_strided_min_row_len_bytes(4096);
  }
  if (bm->strided_max_amplification() == 0) {
    bm->set_strided_max_amplification(8);
  }
  if (bm->strided_block_target_bytes() == 0) {
    bm->set_strided_block_target_bytes(16ULL * 1024 * 1024);
  }
  if (bm->strided_block_max_bytes() == 0) {
    bm->set_strided_block_max_bytes(64ULL * 1024 * 1024);
  }

  if (cfg->has_pinned_memory()) {
    auto* pm = cfg->mutable_pinned_memory();
    if (!pm->has_allocation_timeout()) {
      auto* d = pm->mutable_allocation_timeout();
      d->set_seconds(30);
      d->set_nanos(0);
    }
  }

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

  // Retention handle defaults (durations)
  auto* rh = cfg->mutable_retention_handles();
  if (!rh->has_default_ttl()) {
    auto* d = rh->mutable_default_ttl();
    d->set_seconds(600);
    d->set_nanos(0);
  }
  if (!rh->has_max_ttl()) {
    auto* d = rh->mutable_max_ttl();
    d->set_seconds(24 * 60 * 60);
    d->set_nanos(0);
  }

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
  normalize_duration_fields(root_json);

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

absl::StatusOr<tcfg::DaemonConfig> load_daemon_config_from_text(const std::string& content) {
  tcfg::DaemonConfig cfg;

  if (content.empty()) {
    return absl::InvalidArgumentError("Inline config text is empty");
  }

  nlohmann::json root_json;
  // Try YAML first, then fall back to JSON
  try {
    YAML::Node root = YAML::Load(content);
    if (!root.IsNull()) {
      root_json = yaml_node_to_json(root);
    } else {
      // Empty YAML treated as empty object
      root_json = nlohmann::json::object();
    }
  } catch (const std::exception&) {
    try {
      root_json = nlohmann::json::parse(content);
    } catch (const std::exception& e) {
      return absl::InvalidArgumentError(absl::StrCat("Parse error (YAML/JSON): ", e.what()));
    }
  }

  // Normalize before protobuf parsing
  normalize_enum_aliases(root_json);
  normalize_size_fields(root_json);
  normalize_duration_fields(root_json);

  const std::string json_text = root_json.dump();

  google::protobuf::util::JsonParseOptions opts;
  opts.ignore_unknown_fields = false; // strict
  auto status = google::protobuf::util::JsonStringToMessage(json_text, &cfg, opts);
  if (!status.ok()) {
    return absl::InvalidArgumentError(absl::StrCat("JSON->Proto parse error: ", status.message()));
  }

  normalize_defaults(&cfg);
  return cfg;
}

} // namespace tensorcast::common::config
