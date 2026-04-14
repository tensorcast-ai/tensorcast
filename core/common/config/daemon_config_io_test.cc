// Copyright (c) 2026, TensorCast Team.

#include "core/common/config/daemon_config_io.h"
#include "tensorcast/config/v1/daemon_config.pb.h"

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

TEST_CASE("DaemonConfig materialization strategy defaults and explicit overrides", "[config]") {
  const std::string yaml = R"YAML(
server:
  listen: {host: "127.0.0.1", port: 50052}
  p2p_listen: {host: "127.0.0.1", port: 65090}
  storage_path: "/tmp"
  num_threads: 2
engine:
  materialization_strategy:
    enable_tensor_aware_mapped_executor: false
    allow_mixed_execution: false
    prefer_local_canonical_for_mapped: true
    allow_source_ordered_for_mapped: false
    enable_mapped_dim0_tensor_jobs: false
    enable_mapped_dim1_tensor_jobs: false
    enable_mapped_concat_jobs: false
    enable_mapped_concat_execution: false
    enable_mapped_single_range_concat_jobs: false
    enable_mapped_multirange_concat_jobs: false
    enable_local_batched_disk_load: true
    enable_owner_file_collective: true
    enable_topology_guided_transfer: true
    topology_guided_mode: TOPOLOGY_GUIDED_MODE_PREFER_GUIDED
    enable_remote_bootstrap_local_fanout: true
    executor_preference: MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_OWNER_FILE_COLLECTIVE
    diagnostics_verbosity: MATERIALIZATION_STRATEGY_DIAGNOSTICS_VERBOSITY_VERBOSE
pinned_memory:
  allocation_timeout: 30s
  classes: []
)YAML";

  auto cfg_or = load_daemon_config_from_text(yaml);
  REQUIRE(cfg_or.ok());
  const auto& strategy = cfg_or->engine().materialization_strategy();

  REQUIRE(cfg_or->engine().has_materialization_strategy());
  REQUIRE(strategy.has_enable_tensor_aware_mapped_executor());
  REQUIRE_FALSE(strategy.enable_tensor_aware_mapped_executor());
  REQUIRE(strategy.has_allow_mixed_execution());
  REQUIRE_FALSE(strategy.allow_mixed_execution());
  REQUIRE(strategy.prefer_local_canonical_for_mapped());
  REQUIRE(strategy.has_allow_source_ordered_for_mapped());
  REQUIRE_FALSE(strategy.allow_source_ordered_for_mapped());
  REQUIRE(strategy.has_enable_mapped_dim0_tensor_jobs());
  REQUIRE_FALSE(strategy.enable_mapped_dim0_tensor_jobs());
  REQUIRE(strategy.has_enable_mapped_dim1_tensor_jobs());
  REQUIRE_FALSE(strategy.enable_mapped_dim1_tensor_jobs());
  REQUIRE(strategy.has_enable_mapped_concat_jobs());
  REQUIRE_FALSE(strategy.enable_mapped_concat_jobs());
  REQUIRE(strategy.has_enable_mapped_concat_execution());
  REQUIRE_FALSE(strategy.enable_mapped_concat_execution());
  REQUIRE(strategy.has_enable_mapped_single_range_concat_jobs());
  REQUIRE_FALSE(strategy.enable_mapped_single_range_concat_jobs());
  REQUIRE(strategy.has_enable_mapped_multirange_concat_jobs());
  REQUIRE_FALSE(strategy.enable_mapped_multirange_concat_jobs());
  REQUIRE(strategy.has_enable_local_batched_disk_load());
  REQUIRE(strategy.enable_local_batched_disk_load());
  REQUIRE(strategy.has_enable_owner_file_collective());
  REQUIRE(strategy.enable_owner_file_collective());
  REQUIRE(
      strategy.executor_preference() ==
      tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_OWNER_FILE_COLLECTIVE);
  REQUIRE(
      strategy.diagnostics_verbosity() ==
      tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_DIAGNOSTICS_VERBOSITY_VERBOSE);
  REQUIRE(strategy.has_enable_topology_guided_transfer());
  REQUIRE(strategy.enable_topology_guided_transfer());
  REQUIRE(
      strategy.topology_guided_mode() ==
      tensorcast::config::v1::Engine::MaterializationStrategy::TOPOLOGY_GUIDED_MODE_PREFER_GUIDED);
  REQUIRE(strategy.has_enable_remote_bootstrap_local_fanout());
  REQUIRE(strategy.enable_remote_bootstrap_local_fanout());
}

TEST_CASE("DaemonConfig missing materialization strategy gets 0108 defaults", "[config]") {
  const std::string yaml = R"YAML(
server:
  listen: {host: "127.0.0.1", port: 50052}
  p2p_listen: {host: "127.0.0.1", port: 65090}
  storage_path: "/tmp"
  num_threads: 2
engine:
  artifact_chunk_bytes: 256MB
pinned_memory:
  allocation_timeout: 30s
  classes: []
)YAML";

  auto cfg_or = load_daemon_config_from_text(yaml);
  REQUIRE(cfg_or.ok());
  const auto& strategy = cfg_or->engine().materialization_strategy();

  REQUIRE(cfg_or->engine().has_materialization_strategy());
  REQUIRE(strategy.has_enable_tensor_aware_mapped_executor());
  REQUIRE(strategy.enable_tensor_aware_mapped_executor());
  REQUIRE(strategy.has_enable_local_batched_disk_load());
  REQUIRE(strategy.enable_local_batched_disk_load());
  REQUIRE(strategy.has_enable_owner_file_collective());
  REQUIRE_FALSE(strategy.enable_owner_file_collective());
  REQUIRE(strategy.has_allow_mixed_execution());
  REQUIRE(strategy.allow_mixed_execution());
  REQUIRE(
      strategy.executor_preference() ==
      tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_AUTO);
  REQUIRE(strategy.has_enable_topology_guided_transfer());
  REQUIRE_FALSE(strategy.enable_topology_guided_transfer());
  REQUIRE(
      strategy.topology_guided_mode() ==
      tensorcast::config::v1::Engine::MaterializationStrategy::TOPOLOGY_GUIDED_MODE_DISABLED);
  REQUIRE(strategy.has_enable_remote_bootstrap_local_fanout());
  REQUIRE_FALSE(strategy.enable_remote_bootstrap_local_fanout());
}

TEST_CASE("DaemonConfig partial materialization strategy preserves 0108 defaults", "[config]") {
  const std::string yaml = R"YAML(
server:
  listen: {host: "127.0.0.1", port: 50052}
  p2p_listen: {host: "127.0.0.1", port: 65090}
  storage_path: "/tmp"
  num_threads: 2
engine:
  materialization_strategy:
    executor_preference: MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_TENSOR_AWARE_LOCAL
pinned_memory:
  allocation_timeout: 30s
  classes: []
)YAML";

  auto cfg_or = load_daemon_config_from_text(yaml);
  REQUIRE(cfg_or.ok());
  const auto& strategy = cfg_or->engine().materialization_strategy();

  REQUIRE(strategy.has_enable_local_batched_disk_load());
  REQUIRE(strategy.enable_local_batched_disk_load());
  REQUIRE(strategy.has_enable_owner_file_collective());
  REQUIRE_FALSE(strategy.enable_owner_file_collective());
  REQUIRE(strategy.has_enable_topology_guided_transfer());
  REQUIRE_FALSE(strategy.enable_topology_guided_transfer());
  REQUIRE(
      strategy.topology_guided_mode() ==
      tensorcast::config::v1::Engine::MaterializationStrategy::TOPOLOGY_GUIDED_MODE_DISABLED);
  REQUIRE(strategy.has_enable_remote_bootstrap_local_fanout());
  REQUIRE_FALSE(strategy.enable_remote_bootstrap_local_fanout());
  REQUIRE(
      strategy.executor_preference() ==
      tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_TENSOR_AWARE_LOCAL);
}

} // namespace
} // namespace tensorcast::common::config
