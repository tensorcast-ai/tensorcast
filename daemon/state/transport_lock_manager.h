// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "core/store/materialization/contracts/loading_spec.h" // ReplicaKey

namespace tensorcast::daemon {

struct LockEntry {
  store::loading::ReplicaKey key;
  std::vector<uint32_t> chunk_indices;
  std::chrono::steady_clock::time_point expiry;
};

class TransportLockManager {
 public:
  explicit TransportLockManager(std::chrono::seconds ttl) : ttl_(ttl) {}

  std::string mint_token() {
    // Simple 128-bit random hex token
    std::uniform_int_distribution<uint64_t> dist;
    uint64_t a = dist(rng_);
    uint64_t b = dist(rng_);
    return absl::StrFormat("%016x%016x", a, b);
  }

  void put(const std::string& token, const store::loading::ReplicaKey& key, std::vector<uint32_t> indices) {
    absl::MutexLock l(&mu_);
    locks_[token] = LockEntry{key, std::move(indices), now() + ttl_};
  }

  std::optional<LockEntry> get(const std::string& token) {
    absl::MutexLock l(&mu_);
    auto it = locks_.find(token);
    if (it == locks_.end())
      return std::nullopt;
    if (expired(it->second)) {
      locks_.erase(it);
      return std::nullopt;
    }
    return it->second;
  }

  bool erase(const std::string& token) {
    absl::MutexLock l(&mu_);
    return locks_.erase(token) > 0;
  }

  std::optional<LockEntry> take(const std::string& token) {
    absl::MutexLock l(&mu_);
    auto it = locks_.find(token);
    if (it == locks_.end()) {
      return std::nullopt;
    }
    LockEntry entry = it->second;
    locks_.erase(it);
    return entry;
  }

  bool has_lock_for_key(const store::loading::ReplicaKey& key) const {
    absl::MutexLock l(&mu_);
    for (const auto& entry : locks_) {
      if (expired(entry.second)) {
        continue;
      }
      if (entry.second.key == key) {
        return true;
      }
    }
    return false;
  }

  // Enumerate tokens for sweeping
  std::vector<std::string> tokens() {
    absl::MutexLock l(&mu_);
    std::vector<std::string> out;
    out.reserve(locks_.size());
    for (const auto& kv : locks_)
      out.push_back(kv.first);
    return out;
  }

  // Remove token if expired; returns optional removed entry
  std::optional<LockEntry> remove_if_expired(const std::string& token) {
    absl::MutexLock l(&mu_);
    auto it = locks_.find(token);
    if (it == locks_.end())
      return std::nullopt;
    if (expired(it->second)) {
      LockEntry e = it->second;
      locks_.erase(it);
      return e;
    }
    return std::nullopt;
  }

 private:
  static std::chrono::steady_clock::time_point now() {
    return std::chrono::steady_clock::now();
  }

  static bool expired(const LockEntry& e) {
    return now() >= e.expiry;
  }

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, LockEntry> locks_ ABSL_GUARDED_BY(mu_);
  std::chrono::seconds ttl_;
  std::mt19937_64 rng_{std::random_device{}()};
};

} // namespace tensorcast::daemon
