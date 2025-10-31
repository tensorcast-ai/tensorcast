// Copyright (c) 2025, TensorCast Team.

#include "daemon/ipc_region_registry.h"

#include <algorithm>
#include <limits>

#include "absl/log/log.h"

namespace tensorcast::daemon {

namespace {

uint32_t ClampTtlMs(uint32_t requested, absl::Duration max_ttl) {
  if (requested == 0) {
    return 0;
  }
  const absl::Duration requested_duration = absl::Milliseconds(requested);
  if (max_ttl <= absl::ZeroDuration()) {
    return requested;
  }
  const absl::Duration clamped = std::min(requested_duration, max_ttl);
  const int64_t ms = absl::ToInt64Milliseconds(clamped);
  if (ms <= 0) {
    return 0;
  }
  return static_cast<uint32_t>(std::min<int64_t>(ms, std::numeric_limits<uint32_t>::max()));
}

} // namespace

IpcRegionRegistry::IpcRegionRegistry(Options opts) : opts_(std::move(opts)) {}

absl::StatusOr<IpcRegionRegistry::RegionDescriptor> IpcRegionRegistry::Register(const RegisterParams& params) {
  if (params.owner_pid <= 0) {
    return absl::InvalidArgumentError("owner_pid must be > 0");
  }
  if (params.device_id < 0) {
    return absl::InvalidArgumentError("device_id must be >= 0");
  }
  if (params.size_bytes == 0) {
    return absl::InvalidArgumentError("size_bytes must be > 0");
  }
  if (params.ttl_ms == 0) {
    return absl::InvalidArgumentError("ttl_ms must be > 0");
  }
  if (params.handle_bytes.empty()) {
    return absl::InvalidArgumentError("cuda_ipc_handle must not be empty");
  }

  absl::MutexLock lock(&mu_);
  if (regions_.size() >= opts_.capacity) {
    return absl::ResourceExhaustedError("region registry capacity reached");
  }

  RegionRecord rec;
  rec.desc.region_id = MintRegionIdLocked();
  rec.desc.device_id = params.device_id;
  rec.desc.owner_pid = params.owner_pid;
  rec.desc.size_bytes = params.size_bytes;
  const uint32_t ttl_ms = ClampTtlMs(params.ttl_ms, opts_.max_ttl);
  if (ttl_ms == 0) {
    return absl::InvalidArgumentError("effective ttl_ms underflowed");
  }
  rec.desc.ttl_ms = ttl_ms;
  rec.desc.session_id = params.session_id;
  rec.desc.region_name = params.region_name;
  rec.desc.expires_at = absl::Now() + absl::Milliseconds(ttl_ms);
  rec.handle_bytes = params.handle_bytes;
  rec.refcount = 0;
  rec.inserted_at = absl::Now();

  auto [it, inserted] = regions_.emplace(rec.desc.region_id, std::move(rec));
  if (!inserted) {
    return absl::InternalError("failed to store region");
  }
  return it->second.desc;
}

absl::StatusOr<bool> IpcRegionRegistry::Unregister(const std::string& region_id, int owner_pid, bool force) {
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return absl::NotFoundError("region_id not found");
  }
  if (owner_pid != it->second.desc.owner_pid) {
    return absl::PermissionDeniedError("owner_pid mismatch");
  }
  if (!force && it->second.refcount > 0) {
    return absl::FailedPreconditionError("region has active references");
  }
  regions_.erase(it);
  return true;
}

absl::StatusOr<IpcRegionRegistry::RegionDescriptor> IpcRegionRegistry::Describe(const std::string& region_id) const {
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return absl::NotFoundError("region_id not found");
  }
  return it->second.desc;
}

bool IpcRegionRegistry::RefreshTtl(const std::string& region_id, uint32_t ttl_ms) {
  if (ttl_ms == 0) {
    return false;
  }
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return false;
  }
  const uint32_t effective = ClampTtlMs(ttl_ms, opts_.max_ttl);
  if (effective == 0) {
    return false;
  }
  it->second.desc.ttl_ms = effective;
  it->second.desc.expires_at = absl::Now() + absl::Milliseconds(effective);
  return true;
}

std::vector<IpcRegionRegistry::RegionDescriptor> IpcRegionRegistry::SweepExpired(absl::Time now) {
  std::vector<RegionDescriptor> expired;
  absl::MutexLock lock(&mu_);
  for (auto it = regions_.begin(); it != regions_.end();) {
    if (it->second.desc.expires_at <= now) {
      if (it->second.refcount > 0) {
        it->second.desc.expires_at = absl::Now() + absl::Milliseconds(it->second.desc.ttl_ms);
        ++it;
        continue;
      }
      expired.push_back(it->second.desc);
      auto to_erase = it;
      ++it;
      regions_.erase(to_erase);
    } else {
      ++it;
    }
  }
  return expired;
}

absl::StatusOr<std::string> IpcRegionRegistry::GetHandleBytes(const std::string& region_id) const {
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return absl::NotFoundError("region_id not found");
  }
  return it->second.handle_bytes;
}

absl::StatusOr<IpcRegionRegistry::RegionDescriptor> IpcRegionRegistry::Acquire(
    const std::string& region_id,
    int owner_pid) {
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return absl::NotFoundError("region_id not found");
  }
  if (owner_pid != 0 && owner_pid != it->second.desc.owner_pid) {
    return absl::PermissionDeniedError("owner_pid mismatch");
  }
  ++it->second.refcount;
  it->second.desc.expires_at = absl::Now() + absl::Milliseconds(it->second.desc.ttl_ms);
  return it->second.desc;
}

absl::Status IpcRegionRegistry::Release(const std::string& region_id) {
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return absl::NotFoundError("region_id not found");
  }
  if (it->second.refcount == 0) {
    return absl::FailedPreconditionError("region refcount underflow");
  }
  --it->second.refcount;
  if (it->second.refcount == 0) {
    it->second.desc.expires_at = absl::Now() + absl::Milliseconds(it->second.desc.ttl_ms);
  }
  return absl::OkStatus();
}

std::string IpcRegionRegistry::MintRegionIdLocked() {
  // 128 bits of randomness, rendered as hex.
  for (;;) {
    uint64_t hi = absl::Uniform<uint64_t>(bitgen_);
    uint64_t lo = absl::Uniform<uint64_t>(bitgen_);
    std::string id = absl::StrFormat("region:%016x%016x", hi, lo);
    if (!regions_.contains(id)) {
      return id;
    }
  }
}

} // namespace tensorcast::daemon
