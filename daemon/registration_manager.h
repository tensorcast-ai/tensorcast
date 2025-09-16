// Copyright (c) 2025, TensorCast Team.

// RegistrationManager: encapsulates registration metadata and lease segments
// for Begin/Feed/KeepAlive/Commit lifecycle. Keeps gRPC service thin.

#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "core/store/store_engine.h"
#include "daemon/types.h"

namespace tensorcast::daemon {

class RegistrationManager {
 public:
  enum class RegPlan : uint8_t { COALESCED = 0, LEASE = 1 };

  struct RegMeta {
    RegPlan plan{RegPlan::COALESCED};
    std::chrono::time_point<std::chrono::steady_clock> expiry;
    uint32_t ttl_ms{0};
    uint64_t epoch{0};
    uint64_t total_size{0};
    int device_id{0};
    int owner_pid{0};
    bool lease_in_place{false};
    std::string index_key_hex;
    std::string index_data;
    // Join semantics: set when Commit detects an existing replica and the
    // controller joined a lightweight reference for this registration.
    bool joined_existing{false};
    // The content-addressed mi2 id associated with the joined replica.
    std::string artifact_id_mi2;
    // Lease ID for TTL UseLease created on commit when joining existing GPU replica.
    uint64_t use_lease_id{0};
  };

  RegistrationManager() = default;

  void set_meta(const std::string& reg_id, const RegMeta& meta) {
    absl::MutexLock l(&mu_);
    reg_meta_[reg_id] = meta;
  }

  bool has_meta(const std::string& reg_id) const {
    absl::MutexLock l(&mu_);
    return reg_meta_.contains(reg_id);
  }

  std::optional<RegMeta> get_meta(const std::string& reg_id) const {
    absl::MutexLock l(&mu_);
    auto it = reg_meta_.find(reg_id);
    if (it == reg_meta_.end())
      return std::nullopt;
    return it->second;
  }

  void erase_meta(const std::string& reg_id) {
    absl::MutexLock l(&mu_);
    reg_meta_.erase(reg_id);
  }

  void append_lease_segments(const std::string& reg_id, std::vector<LeaseSegMeta>&& segs) {
    absl::MutexLock l(&mu_);
    auto& v = reg_leases_[reg_id];
    for (auto& s : segs)
      v.push_back(std::move(s));
  }

  std::vector<LeaseSegMeta> get_lease_segments(const std::string& reg_id) const {
    absl::MutexLock l(&mu_);
    auto it = reg_leases_.find(reg_id);
    if (it == reg_leases_.end())
      return {};
    return it->second;
  }

  void erase_lease_segments(const std::string& reg_id) {
    absl::MutexLock l(&mu_);
    reg_leases_.erase(reg_id);
  }

  // Returns true if meta existed and was expired (and erased)
  bool expire_if_ttl_elapsed(const std::string& reg_id) {
    absl::MutexLock l(&mu_);
    auto it = reg_meta_.find(reg_id);
    if (it == reg_meta_.end())
      return false;
    const auto now = std::chrono::steady_clock::now();
    if (it->second.expiry.time_since_epoch().count() > 0 && now > it->second.expiry) {
      reg_meta_.erase(it);
      reg_leases_.erase(reg_id);
      return true;
    }
    return false;
  }

  // Refresh TTL in meta if ttl_ms>0; return ttl to propagate to engine (0 if none)
  uint32_t extend_if_has_ttl(const std::string& reg_id) {
    absl::MutexLock l(&mu_);
    auto it = reg_meta_.find(reg_id);
    if (it == reg_meta_.end())
      return 0;
    if (it->second.ttl_ms > 0) {
      it->second.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(it->second.ttl_ms);
      return it->second.ttl_ms;
    }
    return 0;
  }

  // Pre-commit keepalive for registrations: validates owner, updates epoch/TTL, and extends engine TTL.
  absl::Status keepalive_precommit(
      const std::string& reg_id,
      int owner_pid,
      uint64_t epoch,
      uint32_t ttl_ms,
      store::StoreEngine& engine) {
    uint32_t effective_extend_ms = 0;
    {
      absl::MutexLock l(&mu_);
      auto it = reg_meta_.find(reg_id);
      if (it == reg_meta_.end())
        return absl::NotFoundError("registration_id not found");
      if (owner_pid != it->second.owner_pid)
        return absl::PermissionDeniedError("owner_pid mismatch");
      it->second.epoch = epoch;
      if (ttl_ms > 0) {
        it->second.ttl_ms = ttl_ms;
        it->second.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(ttl_ms);
        effective_extend_ms = ttl_ms;
      }
    }
    if (effective_extend_ms > 0) {
      absl::Status _st = engine.keep_alive_registered_artifact(reg_id, effective_extend_ms);
      if (!_st.ok()) {
        LOG(WARNING) << "keep_alive_registered_artifact failed in keepalive_precommit: reg_id=" << reg_id
                     << " ms=" << effective_extend_ms << ": " << _st;
      }
    }
    return absl::OkStatus();
  }

  void erase_all_for(const std::string& reg_id) {
    absl::MutexLock l(&mu_);
    reg_meta_.erase(reg_id);
    reg_leases_.erase(reg_id);
  }

  // Enumerate registration ids (for TTL sweeping and tests)
  std::vector<std::string> keys() const {
    absl::MutexLock l(&mu_);
    std::vector<std::string> out;
    out.reserve(reg_meta_.size());
    for (const auto& kv : reg_meta_)
      out.push_back(kv.first);
    return out;
  }

 private:
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, RegMeta> reg_meta_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::vector<LeaseSegMeta>> reg_leases_ ABSL_GUARDED_BY(mu_);
};

// Stream operator for RegPlan to enable readable logging
inline std::ostream& operator<<(std::ostream& os, RegistrationManager::RegPlan plan) {
  switch (plan) {
    case RegistrationManager::RegPlan::COALESCED:
      os << "coalesced";
      break;
    case RegistrationManager::RegPlan::LEASE:
      os << "lease";
      break;
    default:
      os << "unknown(" << static_cast<int>(plan) << ")";
      break;
  }
  return os;
}

} // namespace tensorcast::daemon
