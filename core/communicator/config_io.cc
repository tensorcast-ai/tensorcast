// Copyright (c) 2025-2026, TensorCast Team.

#include "core/communicator/config_io.h"

#include <cctype>
#include <fstream>
#include <sstream>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/util/json_util.h"
#include "nlohmann/json.hpp"

#include "yaml-cpp/yaml.h"

namespace tc = tensorcast::communicator::v1;

namespace tensorcast::communicator {

namespace {
// Convert a YAML::Node tree into nlohmann::json for uniform JsonStringToMessage parsing.
nlohmann::json YamlNodeToJson(const YAML::Node& node) {
  using nlohmann::json;
  switch (node.Type()) {
    case YAML::NodeType::Null:
      return json(nullptr);
    case YAML::NodeType::Scalar: {
      const std::string s = node.as<std::string>();
      // Try to parse booleans
      if (s == "true" || s == "True")
        return json(true);
      if (s == "false" || s == "False")
        return json(false);
      // Try to parse integer
      bool numeric = !s.empty() && (std::isdigit(s[0]) || s[0] == '-' || s[0] == '+');
      if (numeric) {
        // Try int64 first, fallback to as string on exception
        try {
          long long v = std::stoll(s);
          return json(v);
        } catch (...) {
          // not an int
        }
        try {
          double d = std::stod(s);
          return json(d);
        } catch (...) {
          // not a double
        }
      }
      return json(s);
    }
    case YAML::NodeType::Sequence: {
      json arr = json::array();
      for (const auto& it : node) {
        arr.push_back(YamlNodeToJson(it));
      }
      return arr;
    }
    case YAML::NodeType::Map: {
      json obj = json::object();
      for (const auto& it : node) {
        obj[it.first.as<std::string>()] = YamlNodeToJson(it.second);
      }
      return obj;
    }
    case YAML::NodeType::Undefined:
    default:
      return json(nullptr);
  }
}

static inline bool HasExt(const std::string& path, const char* ext) {
  return absl::EndsWithIgnoreCase(path, ext);
}

} // namespace

void normalize_defaults(tc::CommunicatorConfig* cfg) {
  if (!cfg)
    return;

  // Stager defaults
  auto* st = cfg->mutable_stager();
  if (st->buffers_per_flow() <= 0)
    st->set_buffers_per_flow(4);

  // RDMA defaults
  auto* rd = cfg->mutable_rdma();
  if (rd->outstanding_wr() <= 0)
    rd->set_outstanding_wr(64);
  if (rd->ack_ttl_ms() <= 0)
    rd->set_ack_ttl_ms(30000);
  if (rd->traffic_class() == 0)
    rd->set_traffic_class(186);
  if (rd->qp_timeout() <= 0)
    rd->set_qp_timeout(20);
  if (rd->qp_retry() <= 0)
    rd->set_qp_retry(7);

  // Transport defaults
  auto* tr = cfg->mutable_transport();
  if (tr->tcp_conn_count() <= 0)
    tr->set_tcp_conn_count(8);
  if (tr->tcp_tos() < 0)
    tr->set_tcp_tos(0);
  if (tr->connect_timeout_sec() <= 0)
    tr->set_connect_timeout_sec(10);
  if (!tr->has_so_reuseport())
    tr->set_so_reuseport(false);

  // Affinity defaults
  cfg->mutable_affinity();

  // NUMA defaults
  cfg->mutable_simple_numa();
}

absl::StatusOr<tc::CommunicatorConfig> LoadCommunicatorConfigFromFile(const std::string& path) {
  tc::CommunicatorConfig cfg;

  std::ifstream in(path);
  if (!in.good()) {
    return absl::NotFoundError(absl::StrCat("Config file not found: ", path));
  }

  std::ostringstream oss;
  oss << in.rdbuf();
  const std::string content = oss.str();

  std::string json_text;
  nlohmann::json root_json;
  if (HasExt(path, ".yaml") || HasExt(path, ".yml")) {
    try {
      YAML::Node root = YAML::Load(content);
      root_json = YamlNodeToJson(root);
      json_text = root_json.dump();
    } catch (const std::exception& e) {
      return absl::InvalidArgumentError(absl::StrCat("YAML parse error: ", e.what()));
    }
  } else {
    // Assume JSON
    try {
      root_json = nlohmann::json::parse(content);
      json_text = root_json.dump();
    } catch (const std::exception& e) {
      return absl::InvalidArgumentError(absl::StrCat("JSON parse error: ", e.what()));
    }
  }

  auto status = google::protobuf::util::JsonStringToMessage(json_text, &cfg);
  if (!status.ok()) {
    return absl::InvalidArgumentError(absl::StrCat("JSON->Proto parse error: ", status.message()));
  }

  // Handle boolean defaults that require presence checks on the original JSON
  if (!(root_json.contains("stager") && root_json["stager"].contains("stage_cpu_for_rdma"))) {
    cfg.mutable_stager()->set_stage_cpu_for_rdma(true);
  }

  normalize_defaults(&cfg);
  return cfg;
}

} // namespace tensorcast::communicator
