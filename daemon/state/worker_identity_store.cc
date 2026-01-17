// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/worker_identity_store.h"

#include "daemon/state/persistence_manager.h"

namespace tensorcast::daemon {

void WorkerIdentityStore::set_registered(std::string worker_id, std::string node_id) {
  std::string node_id_copy;
  {
    absl::MutexLock lock(&mu_);
    worker_id_ = std::move(worker_id);
    node_id_ = std::move(node_id);
    node_id_copy = node_id_;
  }
  if (persistence_mgr_) {
    persistence_mgr_->set_local_node_id(node_id_copy);
  }
  is_registered_.store(true);
}

std::string WorkerIdentityStore::worker_id() const {
  absl::MutexLock lock(&mu_);
  return worker_id_;
}

std::string WorkerIdentityStore::node_id() const {
  absl::MutexLock lock(&mu_);
  return node_id_;
}

} // namespace tensorcast::daemon
