// Copyright (c) 2025, TensorCast Team.

// Sweep tasks: objectify background lambdas into testable task classes

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <unistd.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "daemon/ipc_region_registry.h"
#include "daemon/ref_tracker.h"
#include "daemon/session_lifecycle.h"
#include "daemon/transport_lock_manager.h"
#include "daemon/verification_tracker.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

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
        store::loading::ReplicaKey key{
            .artifact_id = info.artifact_id, .device = store::DeviceRegistry::instance().gpu_key(dev), .replica = 0};
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
