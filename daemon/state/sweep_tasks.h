// Copyright (c) 2025-2026, TensorCast Team.

// Sweep tasks: objectify background lambdas into testable task classes

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "core/store/components/global_store_client.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "daemon/state/binding_registry.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/session_lifecycle.h"
#include "daemon/state/transport_lock_manager.h"
#include "daemon/state/verification_tracker.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

namespace global_store = tensorcast::global_store::v1;

class IBackgroundTask {
 public:
  virtual ~IBackgroundTask() = default;
  virtual void run_once() = 0;
  [[nodiscard]] virtual std::string name() const = 0;
};

// Session TTL handling moved under unified SessionLifecycleTask

class LockTtlTask final : public IBackgroundTask {
 public:
  LockTtlTask(TransportLockManager& locks, store::StoreEngine& engine) : locks_(locks), engine_(engine) {}

  void run_once() override {
    for (const auto& tok : locks_.tokens()) {
      auto expired = locks_.remove_if_expired(tok);
      if (expired.has_value()) {
        // UMA final: no engine-level unlock; TTL sweep clears daemon bookkeeping only.
        VLOG(1) << "LockTtlTask: expired transport lock cleared for artifact_id=" << expired->key.artifact_id;
      }
    }
  }

  [[nodiscard]] std::string name() const override {
    return "LockTtlTask";
  }

 private:
  TransportLockManager& locks_;
  [[maybe_unused]] store::StoreEngine& engine_;
};

class RegionRegistrySweepTask final : public IBackgroundTask {
 public:
  explicit RegionRegistrySweepTask(IpcRegionRegistry& registry) : registry_(registry) {}

  void run_once() override {
    auto expired = registry_.sweep_expired(absl::Now());
    if (expired.empty()) {
      return;
    }
    for (const auto& d : expired) {
      VLOG(1) << "RegionRegistrySweepTask: expired region id=" << d.region_id << " owner_pid=" << d.owner_pid
              << " device=" << d.device_id;
    }
  }

  [[nodiscard]] std::string name() const override {
    return "RegionRegistrySweepTask";
  }

 private:
  IpcRegionRegistry& registry_;
};

class BindingRetentionSweepTask final : public IBackgroundTask {
 public:
  explicit BindingRetentionSweepTask(
      BindingRegistry& registry,
      std::shared_ptr<store::components::IGlobalStoreClient> global_store_client = nullptr)
      : registry_(registry), global_store_client_(std::move(global_store_client)) {}

  void run_once() override {
    const absl::Time now = absl::Now();
    const size_t retired = registry_.sweep_retention(now);
    const size_t staged_expired = registry_.sweep_staged_values(now);
    const size_t staged_terminal = sweep_terminal_group_realizations_();
    if (retired == 0 && staged_expired == 0 && staged_terminal == 0) {
      return;
    }
    VLOG(1) << "BindingRetentionSweepTask: retired retained binding count=" << retired
            << " expired_staged_value_count=" << staged_expired
            << " terminal_group_staged_value_count=" << staged_terminal;
  }

  [[nodiscard]] std::string name() const override {
    return "BindingRetentionSweepTask";
  }

 private:
  [[nodiscard]] size_t sweep_terminal_group_realizations_() {
    if (global_store_client_ == nullptr) {
      return 0;
    }

    constexpr size_t kMaxTransactionChecksPerSweep = 64;
    size_t removed = 0;
    for (const auto& transaction_id : registry_.staged_transaction_ids(kMaxTransactionChecksPerSweep)) {
      global_store::GetGroupRealizationRequest request;
      request.set_transaction_id(transaction_id);
      store::components::RpcOptions options;
      options.timeout = absl::Seconds(1);
      options.max_retries = 0;
      auto response_or = global_store_client_->get_group_realization(request, options);
      if (!response_or.ok()) {
        VLOG(1) << "BindingRetentionSweepTask: group realization status check failed"
                << " transaction_id=" << transaction_id << " status=" << response_or.status();
        continue;
      }
      const auto status = response_or->status();
      const auto state = response_or->state();
      if (status == global_store::STATUS_NOT_FOUND) {
        removed += registry_.remove_staged_values_for_transaction(
            transaction_id, "group_realization_not_found", /*max_to_remove=*/0);
        continue;
      }
      if (status != global_store::STATUS_OK) {
        VLOG(1) << "BindingRetentionSweepTask: group realization status check returned non-ok"
                << " transaction_id=" << transaction_id << " status=" << status;
        continue;
      }
      if (state == global_store::GROUP_REALIZATION_STATE_ABORTED ||
          state == global_store::GROUP_REALIZATION_STATE_EXPIRED) {
        removed += registry_.remove_staged_values_for_transaction(
            transaction_id, "group_realization_terminal", /*max_to_remove=*/0);
      }
    }
    return removed;
  }

  BindingRegistry& registry_;
  std::shared_ptr<store::components::IGlobalStoreClient> global_store_client_;
};

class VerificationTask final : public IBackgroundTask {
 public:
  explicit VerificationTask(VerificationTracker& tracker) : tracker_(tracker) {}

  void run_once() override {
    tracker_.prune();
  }

  [[nodiscard]] std::string name() const override {
    return "VerificationTask";
  }

 private:
  VerificationTracker& tracker_;
};

// PID watch handling moved under unified SessionLifecycleTask

class EvictionTask final : public IBackgroundTask {
 public:
  EvictionTask(store::StoreEngine& engine, RefTracker& refs, SessionLifecycleManager* lifecycle, double limit)
      : engine_(engine), refs_(refs), lifecycle_(lifecycle), limit_(limit) {}

  void run_once() override {
    const int num_gpus = engine_.get_num_gpus();
    for (int dev = 0; dev < num_gpus; ++dev) {
      auto tot_or = engine_.get_device_total_memory(dev);
      auto free_or = engine_.get_device_free_memory(dev);
      if (!tot_or.ok() || !free_or.ok())
        continue;
      const auto total = static_cast<double>(*tot_or);
      const auto used = static_cast<double>(*tot_or - *free_or);
      if (total <= 0.0)
        continue;
      double ratio = used / total;
      if (ratio <= limit_)
        continue;

      struct Cand {
        store::loading::ReplicaKey key;
        std::chrono::time_point<std::chrono::system_clock> last_access;
        size_t size;
      };

      std::vector<Cand> cands;
      for (const auto& info : engine_.get_all_replicas_info()) {
        if (info.gpu_state == common::memory::MemoryLocation::NONE)
          continue;
        if (info.gpu_device_id != dev)
          continue;
        const store::loading::ReplicaKey& key = info.key;
        // Evict only when no active use or placement pins
        if (refs_.ref_count(key) > 0)
          continue;
        if (lifecycle_ && (lifecycle_->use_count_for(key) > 0 || lifecycle_->placement_pin_count_for(key) > 0))
          continue;
        cands.push_back(
            Cand{.key = key, .last_access = info.last_access_time, .size = static_cast<size_t>(info.size_bytes)});
      }
      std::ranges::sort(cands, [](const Cand& a, const Cand& b) { return a.last_access < b.last_access; });
      for (const auto& c : cands) {
        if (ratio <= limit_)
          break;
        int rc = engine_.unload_replica(c.key);
        if (rc != 0) {
          LOG(WARNING) << "EvictionTask: unload_replica failed rc=" << rc << " artifact_id=" << c.key.artifact_id
                       << " device_ord=" << c.key.device.ordinal;
          try {
            static auto meter =
                opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
            static auto ctr = meter->CreateDoubleCounter("tc_unload_failed_total");
            ctr->Add(1.0);
          } catch (...) {
          }
        }
        auto f2 = engine_.get_device_free_memory(dev);
        if (!f2.ok())
          break;
        const auto used2 = static_cast<double>(*tot_or - *f2);
        ratio = used2 / total;
      }
    }
  }

  std::string name() const override {
    return "EvictionTask";
  }

 private:
  store::StoreEngine& engine_;
  RefTracker& refs_;
  SessionLifecycleManager* lifecycle_{nullptr};
  double limit_;
};

// Registration join TTL handling moved under unified SessionLifecycleTask

} // namespace tensorcast::daemon
