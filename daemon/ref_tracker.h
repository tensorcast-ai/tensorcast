// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "core/store/loading/loading_spec.h" // ReplicaKey

namespace tensorcast::daemon {

class RefTracker {
 public:
  struct Entry {
    std::unordered_set<int32_t> pids;
    bool keep_for_global{false};
  };

  void add_ref(const store::loading::ReplicaKey& key, int32_t pid, bool keep_for_global = false) {
    absl::MutexLock l(&mu_);
    auto& e = refs_[key];
    e.pids.insert(pid);
    // Once set, keep_for_global remains true for the lifetime of the replica unless all refs drop.
    e.keep_for_global = e.keep_for_global || keep_for_global;
  }

  void drop_ref(const store::loading::ReplicaKey& key, int32_t pid) {
    absl::MutexLock l(&mu_);
    auto it = refs_.find(key);
    if (it == refs_.end()) {
      return;
    }
    it->second.pids.erase(pid);
    if (it->second.pids.empty()) {
      refs_.erase(it);
    }
  }

  size_t ref_count(const store::loading::ReplicaKey& key) const {
    absl::MutexLock l(&mu_);
    auto it = refs_.find(key);
    return it == refs_.end() ? 0 : it->second.pids.size();
  }

  std::vector<int32_t> pids(const store::loading::ReplicaKey& key) const {
    absl::MutexLock l(&mu_);
    std::vector<int32_t> out;
    auto it = refs_.find(key);
    if (it == refs_.end())
      return out;
    out.reserve(it->second.pids.size());
    for (int32_t p : it->second.pids)
      out.push_back(p);
    return out;
  }

  bool keep_for_global(const store::loading::ReplicaKey& key) const {
    absl::MutexLock l(&mu_);
    auto it = refs_.find(key);
    return it == refs_.end() ? false : it->second.keep_for_global;
  }

  std::vector<store::loading::ReplicaKey> keys() const {
    absl::MutexLock l(&mu_);
    std::vector<store::loading::ReplicaKey> out;
    out.reserve(refs_.size());
    for (const auto& kv : refs_)
      out.push_back(kv.first);
    return out;
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<store::loading::ReplicaKey, Entry, store::loading::ReplicaKeyHash> refs_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
