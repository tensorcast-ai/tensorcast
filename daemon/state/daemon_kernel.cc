// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/daemon_kernel.h"

#include <algorithm>
#include <utility>

#include "absl/log/log.h"
#include "absl/time/time.h"
#include "daemon/state/pid_monitor.h"
#include "daemon/state/sweep_tasks.h"
#include "daemon/util/grpc_daemon_transport.h"

namespace tensorcast::daemon {

DaemonKernel::DaemonKernel(
    std::shared_ptr<store::StoreEngine> engine,
    std::shared_ptr<common::AsyncRuntime> async_runtime,
    DaemonOptions options,
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client)
    : engine_(std::move(engine)),
      async_runtime_(async_runtime ? std::move(async_runtime) : std::make_shared<common::AsyncRuntime>()),
      options_(std::move(options)),
      sessions_(options_.sessions_ttl),
      locks_(options_.locks_ttl),
      region_registry_(
          std::make_unique<IpcRegionRegistry>(IpcRegionRegistry::Options{
              .capacity = options_.max_vram_regions,
              .max_ttl = options_.max_region_ttl,
          })),
      lip_mgr_(std::make_unique<LipManager>(engine_, region_registry_.get())),
      scheduler_(std::make_unique<BackgroundScheduler>()),
      verif_tracker_(std::make_unique<VerificationTracker>()),
      reg_mgr_(std::make_unique<RegistrationManager>()),
      devices_(store::DeviceRegistry::instance()) {
  verif_tracker_->set_serial_executor(async_runtime_->serial_executor());

  lifecycle_mgr_ = std::make_shared<SessionLifecycleManager>(sessions_, refs_, *lip_mgr_, *engine_);
  lifecycle_kernel_ =
      std::make_unique<LifecycleKernel>(options_.daemon_id.empty() ? std::string("daemon-local") : options_.daemon_id);
  pid_monitor_ = std::make_unique<PidMonitor>(
      [this](pid_t pid) {
        if (this->lifecycle_mgr_) {
          this->lifecycle_mgr_->handle_pid_exit(pid);
        }
        if (this->region_registry_) {
          (void)this->region_registry_->handle_pid_exit(static_cast<int>(pid));
        }
        if (this->handle_leases_) {
          this->handle_leases_->handle_pid_exit(pid);
        }
        if (this->binding_registry_) {
          this->binding_registry_->handle_pid_exit(static_cast<int>(pid));
        }
      },
      std::chrono::duration_cast<std::chrono::milliseconds>(options_.proc_check_interval));
  lifecycle_mgr_->attach_pid_monitor(pid_monitor_.get());

  configure_scheduler_tasks_();

  sessions_svc_ = std::make_unique<SessionsService>(
      sessions_, *verif_tracker_, scheduler_.get(), lifecycle_mgr_.get(), absl::Seconds(options_.sessions_ttl.count()));

  lip_bridge_ = std::make_unique<LipBridge>(*lip_mgr_);
  binding_registry_ = std::make_unique<BindingRegistry>();

  if (!options_.local_handle_socket_path.empty()) {
    HandleLeaseRegistry::Options hl_opts;
    hl_opts.capacity = 4096;
    if (options_.handle_lease_ttl.has_value()) {
      const auto ttl_ms = *options_.handle_lease_ttl;
      hl_opts.ttl = absl::Milliseconds(static_cast<int64_t>(ttl_ms.count()));
    }
    hl_opts.max_mints_per_second = options_.handle_lease_max_mints_per_second;
    handle_leases_ = std::make_unique<HandleLeaseRegistry>(hl_opts, *engine_, *lifecycle_mgr_, *lifecycle_kernel_);
  }

  placement_lease_tokens_ = std::make_unique<PlacementLeaseTokens>(PlacementLeaseTokens::Options{});

  if (options_.capability_tokens.active.version != 0 && !options_.capability_tokens.active.secret.empty()) {
    capability_tokens_ = std::make_unique<common::CapabilityTokenManager>(options_.capability_tokens);
  }

  if (options_.retention_handles.enabled) {
    RetentionRegistry::Options retention_opts;
    retention_opts.enabled = options_.retention_handles.enabled;
    retention_opts.default_ttl =
        absl::Milliseconds(static_cast<int64_t>(options_.retention_handles.default_ttl.count()));
    retention_opts.max_ttl = absl::Milliseconds(static_cast<int64_t>(options_.retention_handles.max_ttl.count()));
    retention_registry_ = std::make_unique<RetentionRegistry>(
        retention_opts,
        make_store_engine_retention_backend(*engine_),
        *lifecycle_mgr_,
        *lifecycle_kernel_,
        capability_tokens_.get(),
        options_.daemon_id);
  }

  persistence_mgr_ = std::make_unique<PersistenceManager>(
      scheduler_.get(),
      lip_mgr_.get(),
      engine_.get(),
      async_runtime_,
      engine_->get_artifact_chunk_bytes(),
      std::chrono::milliseconds(500),
      options_.persistence_log_path);
  if (persistence_mgr_) {
    persistence_mgr_->set_storage_path(options_.storage_path);
    persistence_mgr_->set_max_concurrency(options_.max_concurrency);
  }
  engine_->set_stable_cache_spill_evictable([this](const auto& key, const auto& policy) {
    if (!persistence_mgr_) {
      return false;
    }
    return persistence_mgr_->is_spill_evictable(
        key.artifact_id, policy.require_shared_disk_for_spill, policy.require_remote_stable_for_spill);
  });

  identity_store_ = std::make_unique<WorkerIdentityStore>(persistence_mgr_.get());
  identity_store_->set_daemon_id(options_.daemon_id);
  byte_artifact_runtime_state_ = std::make_unique<ByteArtifactRuntimeState>();
  byte_artifact_body_store_ = std::make_unique<ByteArtifactBodyStore>(*byte_artifact_runtime_state_);
  if (persistence_mgr_) {
    persistence_mgr_->set_external_source_resolver(
        [this](std::string_view artifact_id) -> absl::StatusOr<PersistenceManager::PersistenceSource> {
          if (!byte_artifact_body_store_) {
            return absl::NotFoundError("byte_artifact_body_store_unavailable");
          }
          auto source_snapshot = byte_artifact_body_store_->inspect_persistence_source(artifact_id);
          if (!source_snapshot.has_value()) {
            return absl::NotFoundError("byte_artifact_persistence_source_not_found");
          }
          return PersistenceManager::PersistenceSource{
              .artifact_id = std::string(artifact_id),
              .source_artifact_id = source_snapshot->source_artifact_id,
              .total_size_bytes = source_snapshot->size_bytes,
              .verified_content_descriptor = source_snapshot->verified_content_descriptor,
          };
        });
  }
  runtime_event_subscription_ = engine_->subscribe_to_runtime_events([this](const store::runtime::RuntimeEvent& event) {
    if (event.type != store::runtime::RuntimeEventType::kReplicaEvicted) {
      return;
    }
    const auto* payload = std::get_if<store::runtime::ReplicaLifecycleEvent>(&event.payload);
    if (payload == nullptr || !byte_artifact_body_store_) {
      return;
    }
    byte_artifact_body_store_->invalidate_replica_visibility(payload->key, absl::Now(), "runtime_evicted");
  });
  worker_directory_cache_ = std::make_unique<WorkerDirectoryCache>(global_store_client);
  inter_daemon_channel_credentials_ = make_inter_daemon_channel_credentials(options_.inter_daemon_grpc_security);
  const std::string local_daemon_id = options_.daemon_id.empty() ? std::string("daemon-local") : options_.daemon_id;
  byte_artifact_route_resolver_ = std::make_unique<ByteArtifactRouteResolver>(
      *byte_artifact_runtime_state_,
      global_store_client,
      local_daemon_id,
      ByteArtifactRouteResolver::Options{
          .route_staleness_budget = absl::Milliseconds(
              std::max<int64_t>(
                  1, static_cast<int64_t>(options_.byte_artifact_routing.route_staleness_budget.count()))),
          .route_refresh_timeout = absl::Milliseconds(
              std::max<int64_t>(
                  1, static_cast<int64_t>(options_.byte_artifact_routing.route_staleness_budget.count()))),
          .lease_ttl = absl::Milliseconds(
              std::max<int64_t>(1, static_cast<int64_t>(options_.byte_artifact_routing.lease_ttl.count()))),
          .keepalive_interval = absl::Milliseconds(
              std::max<int64_t>(1, static_cast<int64_t>(options_.byte_artifact_routing.keepalive_interval.count()))),
          .routing_epoch = options_.byte_artifact_routing.routing_epoch,
          .shard_home_eligible = options_.byte_artifact_routing.shard_home_eligible,
      });
  payload_transport_broker_ = std::make_unique<PayloadTransportBroker>(
      local_daemon_id,
      capability_tokens_.get(),
      lifecycle_mgr_.get(),
      lifecycle_kernel_.get(),
      PayloadTransportBroker::Options{
          .ttl = options_.byte_artifact_routing.payload_transport.ref_ttl,
          .max_chunk_bytes = options_.byte_artifact_routing.payload_transport.max_chunk_bytes,
          .fetch_deadline = options_.byte_artifact_routing.payload_transport.fetch_deadline,
          .cleanup_interval = options_.byte_artifact_routing.payload_transport.cleanup_interval,
          .inter_daemon_channel_credentials = inter_daemon_channel_credentials_,
          .inter_daemon_grpc_security = options_.inter_daemon_grpc_security,
      });
  retire_gates_ = std::make_unique<RetireGates>(refs_, *lifecycle_mgr_, locks_);
}

DaemonKernel::~DaemonKernel() {
  runtime_event_subscription_.reset();
  stop();
  if (engine_) {
    engine_->set_stable_cache_spill_evictable({});
  }
}

void DaemonKernel::start() {
  bool expected = false;
  if (!started_.compare_exchange_strong(expected, true)) {
    return;
  }
  if (pid_monitor_) {
    pid_monitor_->start();
  }
  if (scheduler_) {
    scheduler_->start();
  }
}

void DaemonKernel::stop() {
  if (!started_.load()) {
    return;
  }
  started_.store(false);
  runtime_event_subscription_.reset();
  if (scheduler_) {
    scheduler_->stop();
  }
  if (pid_monitor_) {
    pid_monitor_->stop();
  }
}

void DaemonKernel::begin_shutdown() {
  shutdown_signal_.begin_shutdown();
  if (async_runtime_) {
    async_runtime_->shutdown();
  }
}

absl::Status DaemonKernel::drain_async_runtime(absl::Time deadline) const {
  if (!async_runtime_) {
    return absl::OkStatus();
  }
  return async_runtime_->drain(deadline);
}

void DaemonKernel::configure_scheduler_tasks_() {
  using std::chrono::milliseconds;

  auto* sched_ptr = scheduler_.get();

  // Session lifecycle: unified task for sessions TTL, PID liveness, join TTL
  auto lifecycle_task = std::make_shared<SessionLifecycleTask>(*lifecycle_mgr_);
  lifecycle_mgr_->set_schedule_hook([this, sched_ptr](absl::Time when) {
    absl::Duration delta;
    if (when == absl::InfiniteFuture()) {
      delta = absl::Milliseconds(
          std::chrono::duration_cast<std::chrono::milliseconds>(this->options_.sessions_sweep_interval).count());
    } else {
      delta = when - absl::Now();
    }
    if (delta < absl::Milliseconds(1)) {
      delta = absl::Milliseconds(1);
    }
    auto next = BackgroundScheduler::Clock::now() + std::chrono::milliseconds(absl::ToInt64Milliseconds(delta));
    if (sched_ptr) {
      sched_ptr->set_next_due(TaskKind::kSessionLifecycle, next);
    }
  });
  scheduler_->add_task(TaskKind::kSessionLifecycle, std::chrono::milliseconds(0), [lifecycle_task, this, sched_ptr]() {
    lifecycle_task->poll();
    if (sched_ptr) {
      absl::Time next_deadline = this->lifecycle_mgr_->next_deadline();
      absl::Duration delta;
      if (next_deadline == absl::InfiniteFuture()) {
        delta = absl::Milliseconds(
            std::chrono::duration_cast<std::chrono::milliseconds>(this->options_.sessions_sweep_interval).count());
      } else {
        delta = next_deadline - absl::Now();
      }
      if (delta < absl::Milliseconds(1)) {
        delta = absl::Milliseconds(1);
      }
      auto when = BackgroundScheduler::Clock::now() + std::chrono::milliseconds(absl::ToInt64Milliseconds(delta));
      sched_ptr->set_next_due(TaskKind::kSessionLifecycle, when);
    }
  });

  // Lock TTL
  {
    auto t = std::make_shared<LockTtlTask>(locks_, *engine_);
    scheduler_->add_task(
        TaskKind::kLockTTL, std::chrono::duration_cast<milliseconds>(options_.locks_sweep_interval), [t]() {
          t->run_once();
        });
  }

  // Region registry sweep
  {
    auto t = std::make_shared<RegionRegistrySweepTask>(*region_registry_);
    scheduler_->add_task(
        TaskKind::kRegionRegistry, std::chrono::duration_cast<milliseconds>(options_.region_sweep_interval), [t]() {
          t->run_once();
        });
  }

  // Verification
  {
    auto t = std::make_shared<VerificationTask>(*verif_tracker_);
    scheduler_->add_task(
        TaskKind::kVerification, std::chrono::duration_cast<milliseconds>(options_.verification_sweep_interval), [t]() {
          t->run_once();
        });
  }

  // Optional eviction
  if (options_.enable_periodic_eviction) {
    const double limit = options_.gpu_memory_limit_fraction;
    auto t = std::make_shared<EvictionTask>(*engine_, refs_, lifecycle_mgr_.get(), limit);
    scheduler_->add_task(
        TaskKind::kEviction, std::chrono::duration_cast<milliseconds>(options_.eviction_check_interval), [t]() {
          t->run_once();
        });
  }
}

} // namespace tensorcast::daemon
