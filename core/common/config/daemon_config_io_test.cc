// Copyright (c) 2026, TensorCast Team.

#include "core/common/config/daemon_config_io.h"

#include <catch2/catch_approx.hpp>
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

TEST_CASE("DaemonConfig defaults public disk source policy from storage_path", "[config]") {
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

  REQUIRE(cfg.has_public_disk_source());
  REQUIRE(cfg.public_disk_source().trusted_root_policies_size() == 1);
  REQUIRE(cfg.public_disk_source().trusted_root_policies(0).root_path() == "/tmp");
  REQUIRE(
      cfg.public_disk_source().trusted_root_policies(0).descriptor_reuse_mode() ==
      tensorcast::config::v1::DaemonConfig::PUBLIC_DISK_SOURCE_DESCRIPTOR_REUSE_MODE_TRUSTED_HINT_ONLY);
  REQUIRE(
      cfg.public_disk_source().trusted_root_policies(0).validation_mode() ==
      tensorcast::config::v1::DaemonConfig::PUBLIC_DISK_SOURCE_VALIDATION_MODE_VALIDATE_BEFORE_READ);
  REQUIRE(cfg.public_disk_source().trusted_root_policies(0).lightweight_attestation_enabled());
}

TEST_CASE("DaemonConfig rejects overlapping public disk source roots", "[config]") {
  const std::string yaml = R"YAML(
server:
  listen: {host: "127.0.0.1", port: 50052}
  p2p_listen: {host: "127.0.0.1", port: 65090}
  storage_path: "/tmp"
  num_threads: 2
public_disk_source:
  trusted_root_policies:
    - policy_id: root-a
      root_path: /tmp/models
      validation_mode: PUBLIC_DISK_SOURCE_VALIDATION_MODE_VALIDATE_BEFORE_READ
      lightweight_attestation_enabled: true
    - policy_id: root-b
      root_path: /tmp/models/subdir
      validation_mode: PUBLIC_DISK_SOURCE_VALIDATION_MODE_VALIDATE_BEFORE_READ
      lightweight_attestation_enabled: true
pinned_memory:
  allocation_timeout: 30s
  classes: []
)YAML";

  auto cfg_or = load_daemon_config_from_text(yaml);
  REQUIRE_FALSE(cfg_or.ok());
  REQUIRE(cfg_or.status().message().find("must not overlap") != std::string::npos);
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
    enable_source_window_collective: true
    enable_source_window_plan_cache: true
    enable_source_window_batched_scatter_kernel: true
    enable_source_window_compiled_routed_program: true
    source_window_compiled_program_build_threads: 16
    enable_source_window_scatter_cuda_graph: true
    executor_preference: MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_OWNER_FILE_COLLECTIVE
    diagnostics_verbosity: MATERIALIZATION_STRATEGY_DIAGNOSTICS_VERBOSITY_VERBOSE
    owner_file_collective_peak_bytes_budget: 17179869184
    owner_file_collective_batch_bytes: 1073741824
    owner_file_collective_dim1_staging_bytes: 536870912
    owner_file_collective_max_inflight_batches: 2
    owner_file_collective_shared_fs_only: false
    owner_file_collective_max_owner_skew_ratio: 1.25
    owner_file_collective_min_dedup_saving_bytes: 134217728
    owner_file_collective_group_assemble_timeout: 5s
    owner_file_collective_allow_mixed_residual: true
    owner_file_collective_planner_cache_entries: 1024
    source_window_collective_selection_mode: MATERIALIZATION_STRATEGY_SOURCE_WINDOW_COLLECTIVE_SELECTION_MODE_STRICT
    source_window_collective_window_bytes: 268435456
    source_window_collective_max_gap_bytes: 131072
    source_window_collective_max_window_amplification_x1000: 1750
    source_window_collective_max_plan_read_amplification_x1000: 1100
    source_window_collective_max_scatter_ops_per_window: 2048
    source_window_collective_peak_bytes_budget: 2147483648
    source_window_collective_min_rank_read_saving_bytes: 67108864
    source_window_collective_max_peer_to_read_ratio_x1000: 4000
    source_window_collective_min_routed_peer_saving_bytes: 33554432
    source_window_collective_distribution_mode: MATERIALIZATION_STRATEGY_SOURCE_WINDOW_COLLECTIVE_DISTRIBUTION_MODE_FULL_WINDOW_ALL_GATHER
    source_window_collective_allow_mixed_residual: true
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
  REQUIRE(strategy.has_enable_source_window_collective());
  REQUIRE(strategy.enable_source_window_collective());
  REQUIRE(strategy.has_enable_source_window_plan_cache());
  REQUIRE(strategy.enable_source_window_plan_cache());
  REQUIRE(strategy.has_enable_source_window_batched_scatter_kernel());
  REQUIRE(strategy.enable_source_window_batched_scatter_kernel());
  REQUIRE(strategy.has_enable_source_window_compiled_routed_program());
  REQUIRE(strategy.enable_source_window_compiled_routed_program());
  REQUIRE(strategy.source_window_compiled_program_build_threads() == 16);
  REQUIRE(strategy.has_enable_source_window_scatter_cuda_graph());
  REQUIRE(strategy.enable_source_window_scatter_cuda_graph());
  REQUIRE(
      strategy.executor_preference() ==
      tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_OWNER_FILE_COLLECTIVE);
  REQUIRE(
      strategy.diagnostics_verbosity() ==
      tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_DIAGNOSTICS_VERBOSITY_VERBOSE);
  REQUIRE(strategy.owner_file_collective_peak_bytes_budget() == 17179869184ULL);
  REQUIRE(strategy.owner_file_collective_batch_bytes() == 1073741824ULL);
  REQUIRE(strategy.owner_file_collective_dim1_staging_bytes() == 536870912ULL);
  REQUIRE(strategy.owner_file_collective_max_inflight_batches() == 2);
  REQUIRE(strategy.has_owner_file_collective_shared_fs_only());
  REQUIRE_FALSE(strategy.owner_file_collective_shared_fs_only());
  REQUIRE(strategy.owner_file_collective_max_owner_skew_ratio() == Catch::Approx(1.25));
  REQUIRE(strategy.owner_file_collective_min_dedup_saving_bytes() == 134217728ULL);
  REQUIRE(strategy.has_owner_file_collective_group_assemble_timeout());
  REQUIRE(strategy.owner_file_collective_group_assemble_timeout().seconds() == 5);
  REQUIRE(strategy.has_owner_file_collective_allow_mixed_residual());
  REQUIRE(strategy.owner_file_collective_allow_mixed_residual());
  REQUIRE(strategy.owner_file_collective_planner_cache_entries() == 1024);
  REQUIRE(
      strategy.source_window_collective_selection_mode() ==
      tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_SOURCE_WINDOW_COLLECTIVE_SELECTION_MODE_STRICT);
  REQUIRE(strategy.source_window_collective_window_bytes() == 268435456ULL);
  REQUIRE(strategy.source_window_collective_max_gap_bytes() == 131072ULL);
  REQUIRE(strategy.source_window_collective_max_window_amplification_x1000() == 1750);
  REQUIRE(strategy.source_window_collective_max_plan_read_amplification_x1000() == 1100);
  REQUIRE(strategy.source_window_collective_max_scatter_ops_per_window() == 2048);
  REQUIRE(strategy.source_window_collective_peak_bytes_budget() == 2147483648ULL);
  REQUIRE(strategy.source_window_collective_min_rank_read_saving_bytes() == 67108864ULL);
  REQUIRE(strategy.source_window_collective_max_peer_to_read_ratio_x1000() == 4000);
  REQUIRE(strategy.source_window_collective_min_routed_peer_saving_bytes() == 33554432ULL);
  REQUIRE(
      strategy.source_window_collective_distribution_mode() ==
      tensorcast::config::v1::Engine::
          MATERIALIZATION_STRATEGY_SOURCE_WINDOW_COLLECTIVE_DISTRIBUTION_MODE_FULL_WINDOW_ALL_GATHER);
  REQUIRE(strategy.has_source_window_collective_allow_mixed_residual());
  REQUIRE(strategy.source_window_collective_allow_mixed_residual());
}

TEST_CASE("DaemonConfig missing materialization strategy gets auto strategy defaults", "[config]") {
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
  REQUIRE(strategy.has_enable_source_window_collective());
  REQUIRE(strategy.enable_source_window_collective());
  REQUIRE(strategy.has_enable_source_window_plan_cache());
  REQUIRE(strategy.enable_source_window_plan_cache());
  REQUIRE(strategy.has_enable_source_window_batched_scatter_kernel());
  REQUIRE(strategy.enable_source_window_batched_scatter_kernel());
  REQUIRE(strategy.has_enable_source_window_compiled_routed_program());
  REQUIRE(strategy.enable_source_window_compiled_routed_program());
  REQUIRE(strategy.source_window_compiled_program_build_threads() == 0);
  REQUIRE(strategy.has_enable_source_window_scatter_cuda_graph());
  REQUIRE_FALSE(strategy.enable_source_window_scatter_cuda_graph());
  REQUIRE(strategy.has_allow_mixed_execution());
  REQUIRE(strategy.allow_mixed_execution());
  REQUIRE(strategy.owner_file_collective_peak_bytes_budget() == 8ULL * 1024ULL * 1024ULL * 1024ULL);
  REQUIRE(strategy.owner_file_collective_batch_bytes() == 512ULL * 1024ULL * 1024ULL);
  REQUIRE(strategy.owner_file_collective_dim1_staging_bytes() == 256ULL * 1024ULL * 1024ULL);
  REQUIRE(strategy.owner_file_collective_max_inflight_batches() == 1);
  REQUIRE(strategy.has_owner_file_collective_shared_fs_only());
  REQUIRE(strategy.owner_file_collective_shared_fs_only());
  REQUIRE(strategy.owner_file_collective_max_owner_skew_ratio() == Catch::Approx(1.5));
  REQUIRE(strategy.owner_file_collective_min_dedup_saving_bytes() == 512ULL * 1024ULL * 1024ULL);
  REQUIRE(strategy.has_owner_file_collective_group_assemble_timeout());
  REQUIRE(strategy.owner_file_collective_group_assemble_timeout().seconds() == 15);
  REQUIRE(strategy.has_owner_file_collective_allow_mixed_residual());
  REQUIRE_FALSE(strategy.owner_file_collective_allow_mixed_residual());
  REQUIRE(strategy.owner_file_collective_planner_cache_entries() == 256);
  REQUIRE(
      strategy.source_window_collective_selection_mode() ==
      tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_SOURCE_WINDOW_COLLECTIVE_SELECTION_MODE_AUTO);
  REQUIRE(strategy.source_window_collective_window_bytes() == 512ULL * 1024ULL * 1024ULL);
  REQUIRE(strategy.source_window_collective_max_gap_bytes() == 256ULL * 1024ULL);
  REQUIRE(strategy.source_window_collective_max_window_amplification_x1000() == 2000);
  REQUIRE(strategy.source_window_collective_max_plan_read_amplification_x1000() == 1200);
  REQUIRE(strategy.source_window_collective_max_scatter_ops_per_window() == 4096);
  REQUIRE(strategy.source_window_collective_peak_bytes_budget() == 4ULL * 1024ULL * 1024ULL * 1024ULL);
  REQUIRE(strategy.source_window_collective_min_rank_read_saving_bytes() == 512ULL * 1024ULL * 1024ULL);
  REQUIRE(strategy.source_window_collective_max_peer_to_read_ratio_x1000() == 8000);
  REQUIRE(strategy.source_window_collective_min_routed_peer_saving_bytes() == 64ULL * 1024ULL * 1024ULL);
  REQUIRE(
      strategy.source_window_collective_distribution_mode() ==
      tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_SOURCE_WINDOW_COLLECTIVE_DISTRIBUTION_MODE_AUTO);
  REQUIRE(strategy.has_source_window_collective_allow_mixed_residual());
  REQUIRE_FALSE(strategy.source_window_collective_allow_mixed_residual());
  REQUIRE(
      strategy.executor_preference() ==
      tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_AUTO);
}

TEST_CASE("DaemonConfig partial materialization strategy preserves auto strategy defaults", "[config]") {
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
  REQUIRE(strategy.has_enable_source_window_collective());
  REQUIRE(strategy.enable_source_window_collective());
  REQUIRE(strategy.has_enable_source_window_plan_cache());
  REQUIRE(strategy.enable_source_window_plan_cache());
  REQUIRE(strategy.has_enable_source_window_batched_scatter_kernel());
  REQUIRE(strategy.enable_source_window_batched_scatter_kernel());
  REQUIRE(strategy.has_enable_source_window_compiled_routed_program());
  REQUIRE(strategy.enable_source_window_compiled_routed_program());
  REQUIRE(strategy.source_window_compiled_program_build_threads() == 0);
  REQUIRE(strategy.has_enable_source_window_scatter_cuda_graph());
  REQUIRE_FALSE(strategy.enable_source_window_scatter_cuda_graph());
  REQUIRE(strategy.has_owner_file_collective_shared_fs_only());
  REQUIRE(strategy.owner_file_collective_shared_fs_only());
  REQUIRE(strategy.owner_file_collective_batch_bytes() == 512ULL * 1024ULL * 1024ULL);
  REQUIRE(strategy.owner_file_collective_planner_cache_entries() == 256);
  REQUIRE(
      strategy.source_window_collective_selection_mode() ==
      tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_SOURCE_WINDOW_COLLECTIVE_SELECTION_MODE_AUTO);
  REQUIRE(strategy.source_window_collective_window_bytes() == 512ULL * 1024ULL * 1024ULL);
  REQUIRE(
      strategy.executor_preference() ==
      tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_TENSOR_AWARE_LOCAL);
}

} // namespace
} // namespace tensorcast::common::config
