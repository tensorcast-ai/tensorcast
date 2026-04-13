// Copyright (c) 2025-2026, TensorCast Team.

#include "core/common/config/daemon_config_io.h"

#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdint>
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

constexpr uint64_t kDefaultCpuSharedMemoryStableBytes = 64ULL * 1024 * 1024;

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
  if (root.contains("promotion") && root["promotion"].is_object()) {
    auto& p = root["promotion"];
    if (p.contains("policy") && p["policy"].is_string()) {
      std::string v = absl::AsciiStrToLower(p["policy"].get<std::string>());
      if (v == "never")
        p["policy"] = "PROMOTION_POLICY_NEVER";
      else if (v == "on_materialize" || v == "materialize")
        p["policy"] = "PROMOTION_POLICY_ON_MATERIALIZE";
      else if (v == "on_hotness" || v == "hotness")
        p["policy"] = "PROMOTION_POLICY_ON_HOTNESS";
      else if (v == "on_policy" || v == "policy")
        p["policy"] = "PROMOTION_POLICY_ON_POLICY";
    }
  }
}

void normalize_capability_defaults(nlohmann::json& root) {
  if (!root.contains("engine") || !root["engine"].is_object()) {
    return;
  }
  auto& engine = root["engine"];
  if (!engine.contains("cpu_shared_memory")) {
    engine["cpu_shared_memory"] = nlohmann::json::object({{"enabled", true}});
    return;
  }
  auto& cpu_shared = engine["cpu_shared_memory"];
  if (cpu_shared.is_object() && !cpu_shared.contains("enabled")) {
    cpu_shared["enabled"] = true;
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

  if (root.contains("byte_artifact_routing") && root["byte_artifact_routing"].is_object()) {
    auto& byte_artifact_routing = root["byte_artifact_routing"];
    if (byte_artifact_routing.contains("inline_payload_threshold_bytes")) {
      to_bytes(byte_artifact_routing["inline_payload_threshold_bytes"]);
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
      if (hl.contains("ttl"))
        to_duration(hl["ttl"]);
    }
  }

  if (root.contains("promotion") && root["promotion"].is_object()) {
    auto& promo = root["promotion"];
    if (promo.contains("demotion_drain_timeout"))
      to_duration(promo["demotion_drain_timeout"]);
  }

  if (root.contains("high_availability") && root["high_availability"].is_object()) {
    auto& ha = root["high_availability"];
    const char* ha_fields[] = {
        "heartbeat_interval",
        "periodic_sync_interval",
        "registration_retry_delay",
        "heartbeat_rpc_timeout",
        "state_sync_rpc_timeout"};
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

  if (root.contains("byte_artifact_routing") && root["byte_artifact_routing"].is_object()) {
    auto& byte_artifact_routing = root["byte_artifact_routing"];
    const char* fields[] = {
        "route_staleness_budget",
        "lease_ttl",
        "keepalive_interval",
        "worker_directory_staleness_budget",
    };
    for (const char* f : fields) {
      if (byte_artifact_routing.contains(f)) {
        to_duration(byte_artifact_routing[f]);
      }
    }
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
  if (!e->has_cpu_shared_memory()) {
    e->mutable_cpu_shared_memory()->set_enabled(true);
  }
  if (e->cpu_shared_memory().enabled()) {
    auto* mt = e->mutable_memory_tiers();
    if (mt->stable_bytes() == 0) {
      mt->set_stable_bytes(kDefaultCpuSharedMemoryStableBytes);
    }
  }
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
  if (!bm->has_disk_source_ordered_read()) {
    bm->set_disk_source_ordered_read(true);
  }
  if (bm->disk_source_merge_max_gap_bytes() == 0) {
    bm->set_disk_source_merge_max_gap_bytes(256ULL * 1024);
  }
  if (bm->disk_source_merge_max_amplification() == 0) {
    bm->set_disk_source_merge_max_amplification(4);
  }
  if (bm->disk_source_prefetch_depth() == 0) {
    bm->set_disk_source_prefetch_depth(2);
  }

  const bool missing_materialization_strategy = !e->has_materialization_strategy();
  if (missing_materialization_strategy) {
    auto* defaults = e->mutable_materialization_strategy();
    defaults->set_enable_local_batched_disk_load(true);
    defaults->set_enable_owner_file_collective(false);
  }
  auto* ms = e->mutable_materialization_strategy();
  if (!ms->has_enable_local_batched_disk_load()) {
    ms->set_enable_local_batched_disk_load(true);
  }
  if (!ms->has_enable_owner_file_collective()) {
    ms->set_enable_owner_file_collective(false);
  }
  if (!ms->has_enable_tensor_aware_mapped_executor()) {
    ms->set_enable_tensor_aware_mapped_executor(true);
  }
  if (!ms->has_allow_mixed_execution()) {
    ms->set_allow_mixed_execution(true);
  }
  if (!ms->has_allow_source_ordered_for_mapped()) {
    ms->set_allow_source_ordered_for_mapped(true);
  }
  if (!ms->has_enable_mapped_dim0_tensor_jobs()) {
    ms->set_enable_mapped_dim0_tensor_jobs(true);
  }
  if (!ms->has_enable_mapped_dim1_tensor_jobs()) {
    ms->set_enable_mapped_dim1_tensor_jobs(true);
  }
  if (!ms->has_enable_mapped_concat_jobs()) {
    ms->set_enable_mapped_concat_jobs(true);
  }
  if (!ms->has_enable_mapped_concat_execution()) {
    ms->set_enable_mapped_concat_execution(true);
  }
  if (!ms->has_enable_mapped_single_range_concat_jobs()) {
    ms->set_enable_mapped_single_range_concat_jobs(true);
  }
  if (!ms->has_enable_mapped_multirange_concat_jobs()) {
    ms->set_enable_mapped_multirange_concat_jobs(true);
  }
  if (ms->executor_preference() == tcfg::Engine::MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_UNSPECIFIED) {
    ms->set_executor_preference(tcfg::Engine::MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_AUTO);
  }
  if (ms->diagnostics_verbosity() == tcfg::Engine::MATERIALIZATION_STRATEGY_DIAGNOSTICS_VERBOSITY_UNSPECIFIED) {
    ms->set_diagnostics_verbosity(tcfg::Engine::MATERIALIZATION_STRATEGY_DIAGNOSTICS_VERBOSITY_BASIC);
  }
  if (ms->owner_file_collective_peak_bytes_budget() == 0) {
    ms->set_owner_file_collective_peak_bytes_budget(8ULL * 1024ULL * 1024ULL * 1024ULL);
  }
  if (ms->owner_file_collective_batch_bytes() == 0) {
    ms->set_owner_file_collective_batch_bytes(512ULL * 1024ULL * 1024ULL);
  }
  if (ms->owner_file_collective_dim1_staging_bytes() == 0) {
    ms->set_owner_file_collective_dim1_staging_bytes(256ULL * 1024ULL * 1024ULL);
  }
  if (ms->owner_file_collective_max_inflight_batches() == 0) {
    ms->set_owner_file_collective_max_inflight_batches(1);
  }
  if (!ms->has_owner_file_collective_shared_fs_only()) {
    ms->set_owner_file_collective_shared_fs_only(true);
  }
  if (ms->owner_file_collective_max_owner_skew_ratio() <= 0.0) {
    ms->set_owner_file_collective_max_owner_skew_ratio(1.5);
  }
  if (ms->owner_file_collective_min_dedup_saving_bytes() == 0) {
    ms->set_owner_file_collective_min_dedup_saving_bytes(64ULL * 1024ULL * 1024ULL);
  }
  if (!ms->has_owner_file_collective_group_assemble_timeout()) {
    auto* d = ms->mutable_owner_file_collective_group_assemble_timeout();
    d->set_seconds(2);
    d->set_nanos(0);
  }
  if (!ms->has_owner_file_collective_allow_mixed_residual()) {
    ms->set_owner_file_collective_allow_mixed_residual(false);
  }
  if (ms->owner_file_collective_planner_cache_entries() == 0) {
    ms->set_owner_file_collective_planner_cache_entries(256);
  }

  if (cfg->has_pinned_memory()) {
    auto* pm = cfg->mutable_pinned_memory();
    if (!pm->has_allocation_timeout()) {
      auto* d = pm->mutable_allocation_timeout();
      d->set_seconds(30);
      d->set_nanos(0);
    }
  }

  auto* promo = cfg->mutable_promotion();
  if (promo->max_concurrency() == 0) {
    promo->set_max_concurrency(4);
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

  // Byte artifact routed home defaults.
  auto* byte_artifact_routing = cfg->mutable_byte_artifact_routing();
  if (byte_artifact_routing->shard_count() == 0) {
    byte_artifact_routing->set_shard_count(4096);
  }
  if (byte_artifact_routing->inline_payload_threshold_bytes() == 0) {
    byte_artifact_routing->set_inline_payload_threshold_bytes(1ULL << 20);
  }
  if (!byte_artifact_routing->has_route_staleness_budget()) {
    auto* d = byte_artifact_routing->mutable_route_staleness_budget();
    d->set_seconds(0);
    d->set_nanos(500 * 1000 * 1000);
  }
  if (!byte_artifact_routing->has_lease_ttl()) {
    auto* d = byte_artifact_routing->mutable_lease_ttl();
    d->set_seconds(5);
    d->set_nanos(0);
  }
  if (!byte_artifact_routing->has_keepalive_interval()) {
    auto* d = byte_artifact_routing->mutable_keepalive_interval();
    d->set_seconds(1);
    d->set_nanos(0);
  }
  if (!byte_artifact_routing->has_worker_directory_staleness_budget()) {
    auto* d = byte_artifact_routing->mutable_worker_directory_staleness_budget();
    d->set_seconds(2);
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
  normalize_capability_defaults(root_json);
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
  normalize_capability_defaults(root_json);
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
