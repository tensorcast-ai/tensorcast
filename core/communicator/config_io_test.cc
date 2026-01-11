// Copyright (c) 2025-2026, TensorCast Team.

#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "absl/strings/str_cat.h"
#include "core/communicator/config_io.h"

using tensorcast::communicator::LoadCommunicatorConfigFromFile;
using tensorcast::communicator::normalize_defaults;
using tensorcast::communicator::v1::CommunicatorConfig;

static std::string write_temp_file(const std::string& content, const std::string& suffix) {
  // Write into the test's working directory
  std::string path = absl::StrCat("comm_cfg_", suffix);
  std::ofstream ofs(path);
  ofs << content;
  ofs.close();
  return path;
}

TEST_CASE("config_io YAML parse + defaults", "[communicator][config]") {
  const char* yaml = R"YAML(
enable_rdma: false
stager:
  buffers_per_flow: 2
rdma:
  outstanding_wr: 32
transport:
  tcp_conn_count: 4
)YAML";
  const std::string path = write_temp_file(yaml, "yaml.yaml");
  auto cfg_or = LoadCommunicatorConfigFromFile(path);
  REQUIRE(cfg_or.ok());
  CommunicatorConfig cfg = cfg_or.value();

  // Presence-based defaults applied
  REQUIRE(cfg.stager().stage_cpu_for_rdma() == true);
  // Provided values retained
  REQUIRE(cfg.stager().buffers_per_flow() == 2);
  REQUIRE(cfg.rdma().outstanding_wr() == 32);
  // Numeric defaults applied
  REQUIRE(cfg.transport().tcp_conn_count() == 4);
  REQUIRE(cfg.transport().connect_timeout_sec() == 10);
  REQUIRE(cfg.transport().so_reuseport() == true);
}

TEST_CASE("config_io JSON parse + defaults", "[communicator][config]") {
  const char* json = R"JSON({
    "enable_rdma": true,
    "rdma": {"qp_retry": 5}
  })JSON";
  const std::string path = write_temp_file(json, "json.json");
  auto cfg_or = LoadCommunicatorConfigFromFile(path);
  REQUIRE(cfg_or.ok());
  CommunicatorConfig cfg = cfg_or.value();

  REQUIRE(cfg.enable_rdma() == true);
  REQUIRE(cfg.rdma().qp_retry() == 5);
  // Defaults
  REQUIRE(cfg.stager().buffers_per_flow() == 4);
  REQUIRE(cfg.transport().tcp_tos() == 0);
  REQUIRE(cfg.transport().so_reuseport() == true);
}
