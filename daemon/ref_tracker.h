// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstdint>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "core/store/loading/loading_spec.h" // ReplicaKey

namespace stepcast::daemon {

class RefTracker {
 public:
  void add_ref(const stepcast::store::ReplicaKey& key, int32_t pid) {
    absl::MutexLock l(&mu_);
    auto& set = refs_[key];
    set.insert(pid);
  }

  void drop_ref(const stepcast::store::ReplicaKey& key, int32_t pid) {
    absl::MutexLock l(&mu_);
    auto it = refs_.find(key);
    if (it == refs_.end()) {
      return;
    }
    it->second.erase(pid);
    if (it->second.empty()) {
      refs_.erase(it);
    }
  }

  size_t ref_count(const stepcast::store::ReplicaKey& key) const {
    absl::MutexLock l(&mu_);
    auto it = refs_.find(key);
    return it == refs_.end() ? 0 : it->second.size();
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<stepcast::store::ReplicaKey, std::unordered_set<int32_t>, stepcast::store::ReplicaKeyHash> refs_
      ABSL_GUARDED_BY(mu_);
};

} // namespace stepcast::daemon
