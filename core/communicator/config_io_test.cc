// Copyright (c) 2025, TensorCast Team.

#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "absl/strings/str_cat.h"
#include "core/communicator/config_io.h"

using tensorcast::communicator::CommunicatorConfig;
using tensorcast::communicator::LoadCommunicatorConfigFromFile;
using tensorcast::communicator::NormalizeDefaults;

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
  stage_chunk_mb_cpu: 8
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
  REQUIRE(cfg.stager().stage_chunk_mb_cpu() == 8);
  REQUIRE(cfg.rdma().outstanding_wr() == 32);
  // Numeric defaults applied
  REQUIRE(cfg.stager().stage_chunk_mb_gpu() == 16);
  REQUIRE(cfg.pool().pool_size_bytes() == 8ULL * 1024 * 1024 * 1024);
  REQUIRE(cfg.transport().tcp_conn_count() == 4);
  REQUIRE(cfg.transport().connect_timeout_sec() == 10);
}

TEST_CASE("config_io JSON parse + defaults", "[communicator][config]") {
  const char* json = R"JSON({
    "enable_rdma": true,
    "rdma": {"qp_retry": 5},
    "pool": {"chunk_bytes": 33554432}
  })JSON";
  const std::string path = write_temp_file(json, "json.json");
  auto cfg_or = LoadCommunicatorConfigFromFile(path);
  REQUIRE(cfg_or.ok());
  CommunicatorConfig cfg = cfg_or.value();

  REQUIRE(cfg.enable_rdma() == true);
  REQUIRE(cfg.rdma().qp_retry() == 5);
  // Defaults
  REQUIRE(cfg.pool().pool_size_bytes() == 8ULL * 1024 * 1024 * 1024);
  REQUIRE(cfg.transport().tcp_tos() == 0);
}
