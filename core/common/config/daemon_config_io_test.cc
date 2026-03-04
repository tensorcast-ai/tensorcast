// Copyright (c) 2026, TensorCast Team.

#include "core/common/config/daemon_config_io.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

namespace tensorcast::common::config {
namespace {

TEST_CASE("DaemonConfig loader normalizes handle_leases.ttl durations", "[config]") {
  const std::string yaml = R"YAML(
server:
  listen: {host: "127.0.0.1", port: 50052}
  p2p_listen: {host: "127.0.0.1", port: 65090}
  storage_path: "/tmp"
  num_threads: 2
engine:
  artifact_chunk_bytes: 1MB
  streaming_buffer_chunks: 4
  cpu_shared_memory:
    enabled: true
  memory_tiers:
    enable_preemptible: false
    stable_bytes: 64MB
pinned_memory:
  allocation_timeout: 30s
  classes: []
lifecycle:
  handle_leases:
    local_handle_socket_path: "/tmp/tensorcast_local_handle.sock"
    ttl: 10m
)YAML";

  auto cfg_or = load_daemon_config_from_text(yaml);
  REQUIRE(cfg_or.ok());
  const auto& cfg = *cfg_or;

  REQUIRE(cfg.has_lifecycle());
  REQUIRE(cfg.lifecycle().has_handle_leases());
  REQUIRE(cfg.lifecycle().handle_leases().has_ttl());
  REQUIRE(cfg.lifecycle().handle_leases().ttl().seconds() == 600);
  REQUIRE(cfg.lifecycle().handle_leases().ttl().nanos() == 0);
}

TEST_CASE("DaemonConfig defaults enable cpu shared memory when omitted", "[config]") {
  const std::string yaml = R"YAML(
server:
  listen: {host: "127.0.0.1", port: 50052}
  p2p_listen: {host: "127.0.0.1", port: 65090}
  storage_path: "/tmp"
  num_threads: 2
pinned_memory:
  allocation_timeout: 30s
  classes: []
)YAML";

  auto cfg_or = load_daemon_config_from_text(yaml);
  REQUIRE(cfg_or.ok());
  const auto& cfg = *cfg_or;

  REQUIRE(cfg.has_engine());
  REQUIRE(cfg.engine().has_cpu_shared_memory());
  REQUIRE(cfg.engine().cpu_shared_memory().enabled());
  REQUIRE(cfg.engine().has_memory_tiers());
  REQUIRE(cfg.engine().memory_tiers().stable_bytes() == 64ULL * 1024 * 1024);
}

TEST_CASE("DaemonConfig defaults enable cpu shared memory when object is empty", "[config]") {
  const std::string yaml = R"YAML(
server:
  listen: {host: "127.0.0.1", port: 50052}
  p2p_listen: {host: "127.0.0.1", port: 65090}
  storage_path: "/tmp"
  num_threads: 2
engine:
  cpu_shared_memory: {}
pinned_memory:
  allocation_timeout: 30s
  classes: []
)YAML";

  auto cfg_or = load_daemon_config_from_text(yaml);
  REQUIRE(cfg_or.ok());
  const auto& cfg = *cfg_or;

  REQUIRE(cfg.engine().has_cpu_shared_memory());
  REQUIRE(cfg.engine().cpu_shared_memory().enabled());
  REQUIRE(cfg.engine().has_memory_tiers());
  REQUIRE(cfg.engine().memory_tiers().stable_bytes() == 64ULL * 1024 * 1024);
}

TEST_CASE("DaemonConfig keeps explicit cpu shared memory disablement", "[config]") {
  const std::string yaml = R"YAML(
server:
  listen: {host: "127.0.0.1", port: 50052}
  p2p_listen: {host: "127.0.0.1", port: 65090}
  storage_path: "/tmp"
  num_threads: 2
engine:
  cpu_shared_memory:
    enabled: false
pinned_memory:
  allocation_timeout: 30s
  classes: []
)YAML";

  auto cfg_or = load_daemon_config_from_text(yaml);
  REQUIRE(cfg_or.ok());
  const auto& cfg = *cfg_or;

  REQUIRE(cfg.engine().has_cpu_shared_memory());
  REQUIRE_FALSE(cfg.engine().cpu_shared_memory().enabled());
}

} // namespace
} // namespace tensorcast::common::config
