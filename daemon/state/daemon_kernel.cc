// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/daemon_kernel.h"

#include <utility>

#include "absl/log/log.h"
#include "absl/time/time.h"
#include "daemon/state/pid_monitor.h"
#include "daemon/state/sweep_tasks.h"

namespace tensorcast::daemon {

DaemonKernel::DaemonKernel(
    std::shared_ptr<store::StoreEngine> engine,
    std::shared_ptr<common::AsyncRuntime> async_runtime,
    DaemonOptions options)
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
  derived_view_export_mgr_ = std::make_unique<DerivedViewExportManager>(
      *engine_,
      *lifecycle_mgr_,
      DerivedViewExportManager::Options{
          .ttl = absl::Milliseconds(static_cast<int64_t>(options_.derived_view_exports.ttl.count())),
          .retry_retire_ttl =
              absl::Milliseconds(static_cast<int64_t>(options_.derived_view_exports.retry_retire_ttl.count())),
      });
  pid_monitor_ = std::make_unique<PidMonitor>(
      [this](pid_t pid) {
        if (this->lifecycle_mgr_) {
          this->lifecycle_mgr_->handle_pid_exit(pid);
        }
        if (this->region_registry_) {
          (void)this->region_registry_->handle_pid_exit(static_cast<int>(pid));
        }
      },
      std::chrono::duration_cast<std::chrono::milliseconds>(options_.proc_check_interval));
  lifecycle_mgr_->attach_pid_monitor(pid_monitor_.get());

  configure_scheduler_tasks_();

  sessions_svc_ = std::make_unique<SessionsService>(
      sessions_, *verif_tracker_, scheduler_.get(), lifecycle_mgr_.get(), absl::Seconds(options_.sessions_ttl.count()));

  lip_bridge_ = std::make_unique<LipBridge>(*lip_mgr_);

  if (!options_.local_handle_socket_path.empty()) {
    HandleLeaseRegistry::Options hl_opts;
    hl_opts.capacity = 4096;
    if (options_.handle_lease_ttl.has_value()) {
      const auto ttl_ms = *options_.handle_lease_ttl;
      hl_opts.ttl = absl::Milliseconds(static_cast<int64_t>(ttl_ms.count()));
    }
    hl_opts.max_mints_per_second = options_.handle_lease_max_mints_per_second;
    handle_leases_ = std::make_unique<HandleLeaseRegistry>(hl_opts, *engine_, *lifecycle_mgr_);
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
  retire_gates_ = std::make_unique<RetireGates>(refs_, *lifecycle_mgr_, locks_);
}

DaemonKernel::~DaemonKernel() {
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
    auto t = std::make_shared<LockTtlTask>(locks_, *engine_, derived_view_export_mgr_.get());
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
