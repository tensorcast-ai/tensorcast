// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <atomic>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

namespace tensorcast::daemon {

class PersistenceManager;

class WorkerIdentityStore {
 public:
  explicit WorkerIdentityStore(PersistenceManager* persistence_mgr = nullptr) : persistence_mgr_(persistence_mgr) {}

  void set_registered(std::string worker_id) {
    set_registered(std::move(worker_id), /*node_id=*/"");
  }

  void set_registered(std::string worker_id, std::string node_id);

  bool is_registered() const {
    return is_registered_.load();
  }

  std::string worker_id() const;
  std::string node_id() const;

 private:
  PersistenceManager* persistence_mgr_; // Not owned.
  std::atomic<bool> is_registered_{false};
  mutable absl::Mutex mu_;
  std::string worker_id_ ABSL_GUARDED_BY(mu_);
  std::string node_id_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
