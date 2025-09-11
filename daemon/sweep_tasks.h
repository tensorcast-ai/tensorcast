// Copyright (c) 2025, TensorCast Team.

// Sweep tasks: objectify background lambdas into testable task classes

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <vector>

#include <unistd.h>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "daemon/lip_manager.h"
#include "daemon/ref_tracker.h"
#include "daemon/registration_manager.h"
#include "daemon/replica_session_manager.h"
#include "daemon/transport_lock_manager.h"
#include "daemon/verification_tracker.h"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon {

class IBackgroundTask {
 public:
  virtual ~IBackgroundTask() = default;
  virtual void run_once() = 0;
  [[nodiscard]] virtual std::string name() const = 0;
};

class SessionTtlTask final : public IBackgroundTask {
 public:
  explicit SessionTtlTask(ReplicaSessionManager& sessions) : sessions_(sessions) {}
  void run_once() override {
    for (const auto& k : sessions_.keys()) {
      sessions_.remove_if_expired(k);
    }
  }
  [[nodiscard]] std::string name() const override {
    return "SessionTtlTask";
  }

 private:
  ReplicaSessionManager& sessions_;
};

class LockTtlTask final : public IBackgroundTask {
 public:
  LockTtlTask(TransportLockManager& locks, store::StoreEngine& engine) : locks_(locks), engine_(engine) {}
  void run_once() override {
    for (const auto& tok : locks_.tokens()) {
      auto expired = locks_.remove_if_expired(tok);
      if (expired.has_value()) {
        (void)engine_.unlock_chunks(expired->key, absl::MakeSpan(expired->chunk_indices), /*copied_gpu=*/false);
      }
    }
  }
  [[nodiscard]] std::string name() const override {
    return "LockTtlTask";
  }

 private:
  TransportLockManager& locks_;
  store::StoreEngine& engine_;
};

// Shared struct used by verification auto-registration queue
struct AutoRegTask {
  store::loading::ReplicaKey key;
  std::string disk_path;
  std::shared_future<absl::Status> ready;
};

class VerificationTask final : public IBackgroundTask {
 public:
  VerificationTask(
      VerificationTracker& tracker,
      store::StoreEngine& engine,
      absl::Mutex* mu,
      std::deque<AutoRegTask>* queue)
      : tracker_(tracker), engine_(engine), mu_(mu), queue_(queue) {}

  void run_once() override {
    tracker_.drain_ready_and_update();
    std::vector<AutoRegTask> reg_ready;
    {
      absl::MutexLock l(mu_);
      for (auto it = queue_->begin(); it != queue_->end();) {
        if (it->ready.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
          reg_ready.push_back(*it);
          it = queue_->erase(it);
        } else {
          ++it;
        }
      }
    }
    for (auto& task : reg_ready) {
      (void)task.ready.get();
      std::string mi2_id;
      try {
        std::filesystem::path desc_path = std::filesystem::path(task.disk_path) / "artifact_descriptor.json";
        if (std::filesystem::exists(desc_path)) {
          std::ifstream f(desc_path);
          if (f.is_open()) {
            nlohmann::json j;
            f >> j;
            if (j.contains("artifact_id") && j["artifact_id"].is_string()) {
              mi2_id = j["artifact_id"].get<std::string>();
            } else if (
                j.contains("index_multihash") && j.contains("data_multihash") && j["index_multihash"].is_string() &&
                j["data_multihash"].is_string()) {
              mi2_id = absl::StrCat(
                  "mi2:", j["index_multihash"].get<std::string>(), ":", j["data_multihash"].get<std::string>());
            }
          }
        }
      } catch (...) {
        VLOG(1) << "descriptor parse error (ignored)";
      }
      auto st = engine_.register_replica_with_global_store(task.key, mi2_id);
      if (!st.ok()) {
        VLOG(1) << "Auto-register disk load failed: " << st;
      }
    }
    tracker_.prune();
  }
  [[nodiscard]] std::string name() const override {
    return "VerificationTask";
  }

 private:
  VerificationTracker& tracker_;
  store::StoreEngine& engine_;
  absl::Mutex* mu_;
  std::deque<AutoRegTask>* queue_ ABSL_GUARDED_BY(*mu_);
};

class PidWatchTask final : public IBackgroundTask {
 public:
  PidWatchTask(RefTracker& refs, LipManager& lip) : refs_(refs), lip_(lip) {}
  void run_once() override {
    auto keys = refs_.keys();
    for (const auto& key : keys) {
      auto plist = refs_.pids(key);
      for (int32_t pid : plist) {
        std::string proc_path = absl::StrCat("/proc/", pid);
        if (::access(proc_path.c_str(), F_OK) != 0) {
          refs_.drop_ref(key, pid);
        }
      }
    }
    lip_.sweep_expired_and_dead_pids();
  }
  std::string name() const override {
    return "PidWatchTask";
  }

 private:
  RefTracker& refs_;
  LipManager& lip_;
};

class EvictionTask final : public IBackgroundTask {
 public:
  EvictionTask(store::StoreEngine& engine, RefTracker& refs, double limit)
      : engine_(engine), refs_(refs), limit_(limit) {}

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
        if (refs_.ref_count(key) > 0 || refs_.keep_for_global(key))
          continue;
        cands.push_back(
            Cand{.key = key, .last_access = info.last_access_time, .size = static_cast<size_t>(info.size_bytes)});
      }
      std::ranges::sort(cands, [](const Cand& a, const Cand& b) { return a.last_access < b.last_access; });
      for (const auto& c : cands) {
        if (ratio <= limit_)
          break;
        (void)engine_.unload_replica(c.key);
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
  double limit_;
};

// TTL sweeper for duplicate-join registrations: when a joined registration's
// TTL expires, drop the lightweight reference that was added at commit time.
class RegJoinTtlTask final : public IBackgroundTask {
 public:
  RegJoinTtlTask(RegistrationManager& reg, RefTracker& refs) : reg_(reg), refs_(refs) {}
  void run_once() override {
    auto ids = reg_.keys();
    for (const auto& id : ids) {
      auto meta_opt = reg_.get_meta(id);
      if (!meta_opt.has_value())
        continue;
      const auto& m = *meta_opt;
      if (!m.joined_existing)
        continue;
      if (reg_.expire_if_ttl_elapsed(id)) {
        store::DeviceKey dev_key{
            .type = (m.plan == RegistrationManager::RegPlan::DVMP ? DeviceType::CPU : DeviceType::GPU),
            .ordinal = (m.plan == RegistrationManager::RegPlan::DVMP ? -1 : m.device_id),
            .uuid = ""};
        store::loading::ReplicaKey key{.artifact_id = m.artifact_id_mi2, .device = dev_key, .replica = 0};
        refs_.drop_ref(key, m.owner_pid);
      }
    }
  }
  [[nodiscard]] std::string name() const override {
    return "RegJoinTtlTask";
  }

 private:
  RegistrationManager& reg_;
  RefTracker& refs_;
};

} // namespace tensorcast::daemon
