// Copyright (c) 2025-2026, TensorCast Team.

// StatusController: handles get_server_config / get_worker_status / get_detailed_status / get_loaded_replicas_v2

#pragma once

#include <chrono>
#include <memory>
#include <string>

#include "core/store/store_engine.h"
#include "daemon/app/startup_coordinator.h"
#include "daemon/service/controllers/status_assembler.h"
#include "daemon/service/replica_listing.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/worker_identity_store.h"
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
  };

  explicit StatusController(Dep d) : d_(std::move(d)) {}

  grpc::Status get_server_config(RpcContext& rctx, v2::GetServerConfigResponse& resp) {
    auto& e = d_.engine;
    (void)rctx;
    resp.set_mem_pool_size(static_cast<int64_t>(e.get_mem_pool_size()));
    // Canonical fields (no gRPC frame size surfaced)
    resp.set_artifact_chunk_bytes(static_cast<uint64_t>(e.get_artifact_chunk_bytes()));
    resp.set_tx_slice_bytes(static_cast<uint64_t>(e.get_tx_slice_bytes()));
    resp.set_local_handle_socket_path(d_.local_handle_socket_path);
    resp.set_cpu_shared_memory_enabled(d_.cpu_shared_memory_enabled);
    resp.set_startup_phase(startup_phase_proto());
    rctx.mark_success();
    return grpc::Status::OK;
  }

  grpc::Status get_worker_status(RpcContext& rctx, v2::GetWorkerStatusResponse& resp) const {
    resp.set_is_registered(d_.identity.is_registered());
    resp.set_is_healthy(true);
    resp.set_is_shutting_down(d_.shutdown_signal.is_shutting_down());
    resp.set_mem_pool_total_size(d_.engine.get_mem_pool_size());
    resp.set_mem_pool_available_size(d_.engine.get_available_memory());
    resp.set_uptime_seconds(uptime().count());
    resp.set_worker_id(d_.identity.is_registered() ? d_.identity.worker_id() : "");
    resp.set_daemon_id(d_.identity.daemon_id());
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
  Dep d_;

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
