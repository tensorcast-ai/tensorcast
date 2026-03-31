// Copyright (c) 2025-2026, TensorCast Team.

// StatusController: handles get_server_config / get_worker_status / get_detailed_status / get_loaded_replicas_v2

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "daemon/app/startup_coordinator.h"
#include "daemon/service/controllers/status_assembler.h"
#include "daemon/service/replica_listing.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/instance_execution_directory_cache.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/worker_directory_cache.h"
#include "daemon/state/worker_identity_store.h"
#include "daemon/util/status_utils.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

class StatusController {
 public:
  struct Dep {
    store::StoreEngine& engine;
    RefTracker& refs;
    ShutdownSignal& shutdown_signal;
    WorkerIdentityStore& identity;
    std::chrono::steady_clock::time_point start_time;
    std::string local_handle_socket_path;
    bool cpu_shared_memory_enabled{true};
    std::shared_ptr<StartupCoordinator> startup_coordinator;
    WorkerDirectoryCache& worker_directory_cache;
    InstanceExecutionDirectoryCache& instance_execution_directory_cache;
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client;
    absl::Duration directory_staleness_budget{absl::Seconds(2)};
  };

  explicit StatusController(Dep d) : d_(std::move(d)) {}

  grpc::Status get_server_config(RpcContext& rctx, v2::GetServerConfigResponse& resp) {
    auto& e = d_.engine;
    (void)rctx;
    // Contract version 2 includes truthful same-binding identity diagnostics
    // (`seal_mint` vs `seal_reuse` vs `not_applicable`) and structured
    // collective failure-class surfacing through SDK-visible errors.
    constexpr uint32_t kSourceBoundContractVersion = 2;
    constexpr uint64_t kSourceBoundCapabilityFlags =
        static_cast<uint64_t>(
            v2::SourceBoundCapabilityFlag::SOURCE_BOUND_CAPABILITY_FLAG_FIRST_CLASS_COLLECTIVE_INGRESS) |
        static_cast<uint64_t>(v2::SourceBoundCapabilityFlag::SOURCE_BOUND_CAPABILITY_FLAG_TYPED_EXECUTION_DIAGNOSTICS) |
        static_cast<uint64_t>(v2::SourceBoundCapabilityFlag::SOURCE_BOUND_CAPABILITY_FLAG_SINGLE_MINT_BINDING_CLOSEOUT);
    resp.set_mem_pool_size(static_cast<int64_t>(e.get_mem_pool_size()));
    // Canonical fields (no gRPC frame size surfaced)
    resp.set_artifact_chunk_bytes(static_cast<uint64_t>(e.get_artifact_chunk_bytes()));
    resp.set_tx_slice_bytes(static_cast<uint64_t>(e.get_tx_slice_bytes()));
    resp.set_local_handle_socket_path(d_.local_handle_socket_path);
    resp.set_cpu_shared_memory_enabled(d_.cpu_shared_memory_enabled);
    resp.set_startup_phase(startup_phase_proto());
    resp.set_source_bound_capability_flags(kSourceBoundCapabilityFlags);
    resp.set_source_bound_contract_version(kSourceBoundContractVersion);
    rctx.mark_success();
    return grpc::Status::OK;
  }

  grpc::Status get_worker_status(RpcContext& rctx, v2::GetWorkerStatusResponse& resp) const {
    const int64_t as_of_ms = absl::ToUnixMillis(absl::Now());
    const uint64_t cache_epoch = worker_status_cache_epoch_.fetch_add(1, std::memory_order_relaxed) + 1;
    resp.set_is_registered(d_.identity.is_registered());
    resp.set_is_healthy(true);
    resp.set_is_shutting_down(d_.shutdown_signal.is_shutting_down());
    resp.set_mem_pool_total_size(d_.engine.get_mem_pool_size());
    resp.set_mem_pool_available_size(d_.engine.get_available_memory());
    resp.set_uptime_seconds(uptime().count());
    resp.set_worker_id(d_.identity.is_registered() ? d_.identity.worker_id() : "");
    resp.set_daemon_id(d_.identity.daemon_id());
    resp.set_as_of_ms(as_of_ms);
    resp.set_staleness_ms(0);
    resp.set_cache_epoch(cache_epoch);
    resp.set_freshness_state("current");
    // Optional metrics for status snapshots
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto upt_hist = meter->CreateDoubleHistogram("tc_status_worker_uptime_seconds");
      static auto reg_counter = meter->CreateDoubleCounter("tc_status_worker_registered_total");
      upt_hist->Record(static_cast<double>(resp.uptime_seconds()), opentelemetry::context::Context{});
      if (resp.is_registered())
        reg_counter->Add(1.0);
    } catch (...) {
    }
    rctx.mark_success();
    return grpc::Status::OK;
  }

  grpc::Status list_directory_workers(
      RpcContext& rctx,
      const v2::ListDirectoryWorkersRequest& req,
      v2::ListDirectoryWorkersResponse& resp) const {
    const absl::Duration staleness_budget = sanitize_staleness_budget(
        req.has_max_staleness_ms() ? std::optional<uint64_t>(req.max_staleness_ms()) : std::nullopt);
    if (is_local_only()) {
      auto* route = resp.add_workers();
      route->set_daemon_id(d_.identity.daemon_id());
      if (d_.identity.is_registered()) {
        route->set_worker_id(d_.identity.worker_id());
      }
      fill_directory_freshness(local_directory_freshness(local_worker_directory_cache_epoch_), resp);
      rctx.mark_success();
      return grpc::Status::OK;
    }
    if (!is_global_store_backed()) {
      return authority_unavailable_status("ListDirectoryWorkers");
    }

    const absl::Time now = absl::Now();
    auto snapshot_or = d_.worker_directory_cache.list_workers(
        req.include_unavailable(), req.required_capability_flags(), now, staleness_budget);
    if (!snapshot_or.ok()) {
      return status_utils::to_grpc_status(snapshot_or.status());
    }
    for (const auto& entry : snapshot_or->entries) {
      auto* route = resp.add_workers();
      route->set_daemon_id(entry.daemon_id);
      route->set_worker_id(entry.worker_id);
      route->set_daemon_address(entry.address);
      route->set_capability_flags(entry.capability_flags);
    }
    fill_directory_freshness(snapshot_directory_freshness(*snapshot_or, now, "GLOBAL_STORE_BACKED"), resp);
    rctx.mark_success();
    return grpc::Status::OK;
  }

  grpc::Status list_directory_instances(
      RpcContext& rctx,
      const v2::ListDirectoryInstancesRequest& req,
      v2::ListDirectoryInstancesResponse& resp) const {
    const absl::Duration staleness_budget = sanitize_staleness_budget(
        req.has_max_staleness_ms() ? std::optional<uint64_t>(req.max_staleness_ms()) : std::nullopt);
    if (is_local_only()) {
      fill_directory_freshness(local_directory_freshness(local_instance_directory_cache_epoch_), resp);
      rctx.mark_success();
      return grpc::Status::OK;
    }
    if (!is_global_store_backed()) {
      return authority_unavailable_status("ListDirectoryInstances");
    }

    const absl::Time now = absl::Now();
    auto snapshot_or = d_.instance_execution_directory_cache.list_instances(
        req.include_unavailable(), req.required_capability_flags(), now, staleness_budget);
    if (!snapshot_or.ok()) {
      return status_utils::to_grpc_status(snapshot_or.status());
    }
    for (const auto& entry : snapshot_or->entries) {
      auto* route = resp.add_instances();
      route->set_instance_id(entry.instance_id);
      route->set_daemon_id(entry.daemon_id);
      route->set_execution_host_kind(entry.execution_host_kind);
      route->set_execution_endpoint(entry.execution_endpoint);
      route->set_engine(entry.engine);
      route->set_capability_flags(entry.capability_flags);
    }
    fill_directory_freshness(snapshot_directory_freshness(*snapshot_or, now, "GLOBAL_STORE_BACKED"), resp);
    rctx.mark_success();
    return grpc::Status::OK;
  }

  grpc::Status resolve_instance_execution(
      RpcContext& rctx,
      const v2::ResolveInstanceExecutionRequest& req,
      v2::ResolveInstanceExecutionResponse& resp) const {
    if (req.instance_id().empty()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "instance_id is required");
    }
    const absl::Duration staleness_budget = sanitize_staleness_budget(
        req.has_max_staleness_ms() ? std::optional<uint64_t>(req.max_staleness_ms()) : std::nullopt);
    if (is_local_only()) {
      return grpc::Status(grpc::StatusCode::NOT_FOUND, "instance execution route not found in LOCAL_ONLY authority");
    }
    if (!is_global_store_backed()) {
      return authority_unavailable_status("ResolveInstanceExecution");
    }

    const absl::Time now = absl::Now();
    auto route_or =
        d_.instance_execution_directory_cache.resolve_instance_execution(req.instance_id(), now, staleness_budget);
    if (!route_or.ok()) {
      return status_utils::to_grpc_status(route_or.status());
    }
    resp.mutable_route()->set_instance_id(route_or->instance_id);
    resp.mutable_route()->set_daemon_id(route_or->daemon_id);
    resp.mutable_route()->set_execution_host_kind(route_or->execution_host_kind);
    resp.mutable_route()->set_execution_endpoint(route_or->execution_endpoint);
    resp.mutable_route()->set_engine(route_or->engine);
    resp.mutable_route()->set_capability_flags(route_or->capability_flags);
    auto snapshot_or = d_.instance_execution_directory_cache.list_instances(
        /*include_unavailable=*/false,
        /*required_capability_flags=*/0,
        now,
        staleness_budget);
    if (!snapshot_or.ok()) {
      return status_utils::to_grpc_status(snapshot_or.status());
    }
    fill_directory_freshness(snapshot_directory_freshness(*snapshot_or, now, "GLOBAL_STORE_BACKED"), resp);
    rctx.mark_success();
    return grpc::Status::OK;
  }

  grpc::Status get_detailed_status(RpcContext& rctx, v2::GetDetailedStatusResponse& resp) {
    resp.set_is_registered(d_.identity.is_registered());
    resp.set_is_healthy(true);
    resp.set_is_shutting_down(d_.shutdown_signal.is_shutting_down());
    resp.set_uptime_seconds(uptime().count());
    resp.set_worker_id(d_.identity.is_registered() ? d_.identity.worker_id() : "");
    StatusAssembler::FillDetailedStatus(d_.engine, d_.refs, resp);
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto repl_hist = meter->CreateDoubleHistogram("tc_status_total_replicas");
      static auto bytes_hist = meter->CreateDoubleHistogram("tc_status_total_bytes");
      repl_hist->Record(static_cast<double>(resp.total_replicas_loaded()), opentelemetry::context::Context{});
      bytes_hist->Record(static_cast<double>(resp.total_artifact_size_bytes()), opentelemetry::context::Context{});
    } catch (...) {
    }
    rctx.mark_success();
    return grpc::Status::OK;
  }

  grpc::Status get_loaded_replicas_v2(
      RpcContext& rctx,
      const v2::GetLoadedReplicasV2Request& req,
      v2::GetLoadedReplicasV2Response& resp,
      bool use_cursor_pagination) {
    if (rctx.allow_high_card_attrs()) {
      if (req.has_artifact_id_filter())
        rctx.span()->SetAttribute("tc.artifact.filter", req.artifact_id_filter());
    }
    listing::FillLoadedReplicasV2(d_.engine, d_.refs, req, resp, use_cursor_pagination);
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto page_counter = meter->CreateDoubleCounter("tc_status_list_pages_total");
      static auto size_hist = meter->CreateDoubleHistogram("tc_status_list_page_size");
      const double page_size = static_cast<double>(resp.replicas_size());
      page_counter->Add(1.0);
      size_hist->Record(page_size, opentelemetry::context::Context{});
    } catch (...) {
    }
    rctx.mark_success();
    return grpc::Status::OK;
  }

 private:
  struct DirectoryFreshness {
    int64_t as_of_ms{0};
    uint64_t staleness_ms{0};
    uint64_t cache_epoch{0};
    std::string freshness_state{"current"};
    std::string authority_mode{"unknown"};
  };

  Dep d_;
  mutable std::atomic<uint64_t> worker_status_cache_epoch_{0};
  mutable std::atomic<uint64_t> local_worker_directory_cache_epoch_{0};
  mutable std::atomic<uint64_t> local_instance_directory_cache_epoch_{0};

  [[nodiscard]] bool is_local_only() const {
    return d_.global_store_client == nullptr;
  }

  [[nodiscard]] bool is_global_store_backed() const {
    return d_.global_store_client != nullptr && d_.global_store_client->is_connected();
  }

  [[nodiscard]] grpc::Status authority_unavailable_status(std::string_view operation) const {
    return grpc::Status(
        grpc::StatusCode::UNAVAILABLE, std::string(operation) + " requires a connected Global Store authority");
  }

  [[nodiscard]] absl::Duration sanitize_staleness_budget(std::optional<uint64_t> request_ms) const {
    const uint64_t raw_ms = request_ms.has_value()
        ? *request_ms
        : static_cast<uint64_t>(std::max<int64_t>(1, absl::ToInt64Milliseconds(d_.directory_staleness_budget)));
    return absl::Milliseconds(std::max<uint64_t>(1, raw_ms));
  }

  [[nodiscard]] static DirectoryFreshness snapshot_directory_freshness(
      const auto& snapshot,
      absl::Time now,
      std::string authority_mode) {
    DirectoryFreshness freshness;
    freshness.as_of_ms = absl::ToUnixMillis(snapshot.refreshed_at);
    freshness.staleness_ms =
        static_cast<uint64_t>(std::max<int64_t>(0, absl::ToInt64Milliseconds(now - snapshot.refreshed_at)));
    freshness.cache_epoch = snapshot.cache_epoch;
    freshness.freshness_state = "current";
    freshness.authority_mode = std::move(authority_mode);
    return freshness;
  }

  [[nodiscard]] static DirectoryFreshness local_directory_freshness(std::atomic<uint64_t>& epoch) {
    DirectoryFreshness freshness;
    freshness.as_of_ms = absl::ToUnixMillis(absl::Now());
    freshness.staleness_ms = 0;
    freshness.cache_epoch = epoch.fetch_add(1, std::memory_order_relaxed) + 1;
    freshness.freshness_state = "current";
    freshness.authority_mode = "LOCAL_ONLY";
    return freshness;
  }

  template <typename Response>
  static void fill_directory_freshness(const DirectoryFreshness& freshness, Response& resp) {
    resp.set_as_of_ms(freshness.as_of_ms);
    resp.set_staleness_ms(freshness.staleness_ms);
    resp.set_cache_epoch(freshness.cache_epoch);
    resp.set_freshness_state(freshness.freshness_state);
    resp.set_authority_mode(freshness.authority_mode);
  }

  v2::DaemonStartupPhase startup_phase_proto() const {
    if (!d_.startup_coordinator) {
      return v2::DaemonStartupPhase::DAEMON_STARTUP_PHASE_READY;
    }
    switch (d_.startup_coordinator->current_phase()) {
      case StartupCoordinator::Phase::kListening:
        return v2::DaemonStartupPhase::DAEMON_STARTUP_PHASE_LISTENING;
      case StartupCoordinator::Phase::kReady:
        return v2::DaemonStartupPhase::DAEMON_STARTUP_PHASE_READY;
      case StartupCoordinator::Phase::kFailed:
        return v2::DaemonStartupPhase::DAEMON_STARTUP_PHASE_FAILED;
    }
    return v2::DaemonStartupPhase::DAEMON_STARTUP_PHASE_FAILED;
  }

  std::chrono::seconds uptime() const {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - d_.start_time);
  }
};

} // namespace tensorcast::daemon
