// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/ipc_region_registry.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>

#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

namespace tensorcast::daemon {

namespace {

struct TemporaryHostSharedSlab {
  int fd = -1;
  void* base = nullptr;
  uint64_t size_bytes = 0;
};

void cleanup_temporary_host_shared_slab(TemporaryHostSharedSlab* slab) {
  if (slab == nullptr) {
    return;
  }
  if (slab->base != nullptr && slab->size_bytes != 0) {
    (void)::munmap(slab->base, static_cast<size_t>(slab->size_bytes));
  }
  if (slab->fd >= 0) {
    (void)::close(slab->fd);
  }
  slab->fd = -1;
  slab->base = nullptr;
  slab->size_bytes = 0;
}

absl::StatusOr<TemporaryHostSharedSlab> create_temporary_host_shared_slab(uint64_t size_bytes) {
  if (size_bytes == 0) {
    return absl::InvalidArgumentError("HOST_SHARED slab size must be > 0");
  }
  TemporaryHostSharedSlab slab;
  slab.size_bytes = size_bytes;

  const int fd = static_cast<int>(::syscall(SYS_memfd_create, "tensorcast_region", MFD_CLOEXEC | MFD_ALLOW_SEALING));
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, "memfd_create failed for HOST_SHARED region");
  }
  slab.fd = fd;
  if (::ftruncate(fd, static_cast<off_t>(size_bytes)) != 0) {
    const int err = errno;
    cleanup_temporary_host_shared_slab(&slab);
    return absl::ErrnoToStatus(err, "ftruncate failed for HOST_SHARED region");
  }

  void* base = ::mmap(nullptr, static_cast<size_t>(size_bytes), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (base == MAP_FAILED) {
    const int err = errno;
    slab.base = nullptr;
    cleanup_temporary_host_shared_slab(&slab);
    return absl::ErrnoToStatus(err, "mmap(MAP_SHARED) failed for HOST_SHARED region");
  }
  slab.base = base;
  return slab;
}

uint32_t clamp_ttl_ms(uint32_t requested, absl::Duration max_ttl) {
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

void IpcRegionRegistry::set_pre_cleanup_callback(RegionCleanupCallback callback) {
  absl::MutexLock lock(&mu_);
  pre_cleanup_callback_ = std::move(callback);
}

absl::StatusOr<IpcRegionRegistry::RegionDescriptor> IpcRegionRegistry::register_region(const RegisterParams& params) {
  if (params.owner_pid <= 0) {
    return absl::InvalidArgumentError("owner_pid must be > 0");
  }
  if (params.size_bytes == 0) {
    return absl::InvalidArgumentError("size_bytes must be > 0");
  }

  std::string attachment_bytes = params.attachment_bytes.empty() ? params.handle_bytes : params.attachment_bytes;
  TemporaryHostSharedSlab temporary_host_shared_slab;

  switch (params.memory_kind) {
    case MemoryKind::kVram:
      if (params.device_id < 0) {
        return absl::InvalidArgumentError("device_id must be >= 0");
      }
      if (attachment_bytes.empty()) {
        return absl::InvalidArgumentError("cuda_ipc_handle must not be empty");
      }
      break;
    case MemoryKind::kHostShared:
      if (params.device_id != -1) {
        return absl::InvalidArgumentError("HOST_SHARED regions must use device_id=-1");
      }
      if (params.daemon_managed) {
        if (!attachment_bytes.empty()) {
          return absl::InvalidArgumentError("daemon-managed HOST_SHARED regions must not provide attachment metadata");
        }
        auto slab_or = create_temporary_host_shared_slab(params.size_bytes);
        if (!slab_or.ok()) {
          return slab_or.status();
        }
        temporary_host_shared_slab = *slab_or;
      } else if (attachment_bytes.empty()) {
        return absl::InvalidArgumentError("HOST_SHARED regions require attachment metadata unless daemon_managed=true");
      }
      break;
  }

  absl::MutexLock lock(&mu_);
  if (regions_.size() >= opts_.capacity) {
    cleanup_temporary_host_shared_slab(&temporary_host_shared_slab);
    return absl::ResourceExhaustedError("region registry capacity reached");
  }

  RegionRecord rec;
  rec.desc.region_id = mint_region_id_locked();
  rec.desc.memory_kind = params.memory_kind;
  rec.desc.device_id = params.device_id;
  rec.desc.owner_pid = params.owner_pid;
  rec.desc.size_bytes = params.size_bytes;
  if (params.ttl_ms == 0) {
    rec.desc.ttl_ms = 0;
    rec.desc.expires_at = absl::InfiniteFuture();
  } else {
    const uint32_t ttl_ms = clamp_ttl_ms(params.ttl_ms, opts_.max_ttl);
    if (ttl_ms == 0) {
      cleanup_temporary_host_shared_slab(&temporary_host_shared_slab);
      return absl::InvalidArgumentError("effective ttl_ms underflowed");
    }
    rec.desc.ttl_ms = ttl_ms;
    rec.desc.expires_at = absl::Now() + absl::Milliseconds(ttl_ms);
  }
  rec.desc.session_id = params.session_id;
  rec.desc.region_name = params.region_name;
  rec.desc.daemon_managed = params.daemon_managed;
  rec.desc.host_region_class = params.host_region_class;
  rec.desc.poisoned = false;

  if (params.memory_kind == MemoryKind::kHostShared && params.daemon_managed) {
    rec.desc.attach_token = mint_attach_token_locked();
    rec.attachment_bytes = rec.desc.attach_token;
    auto daemon_slab = std::make_unique<DaemonManagedHostSharedSlab>();
    daemon_slab->fd = temporary_host_shared_slab.fd;
    daemon_slab->base = temporary_host_shared_slab.base;
    daemon_slab->size_bytes = temporary_host_shared_slab.size_bytes;
    rec.daemon_host_shared_slab = std::move(daemon_slab);
    temporary_host_shared_slab.fd = -1;
    temporary_host_shared_slab.base = nullptr;
    temporary_host_shared_slab.size_bytes = 0;
  } else {
    rec.desc.attach_token = (params.memory_kind == MemoryKind::kHostShared) ? attachment_bytes : std::string{};
    rec.attachment_bytes = std::move(attachment_bytes);
  }
  rec.refcount = 0;
  rec.inserted_at = absl::Now();
  rec.poisoned = false;
  rec.retiring = false;

  auto [it, inserted] = regions_.emplace(rec.desc.region_id, std::move(rec));
  if (!inserted) {
    cleanup_temporary_host_shared_slab(&temporary_host_shared_slab);
    return absl::InternalError("failed to store region");
  }
  if (it->second.desc.memory_kind == MemoryKind::kHostShared && !it->second.desc.attach_token.empty()) {
    attach_tokens_[it->second.desc.attach_token] = it->second.desc.region_id;
  }
  return it->second.desc;
}

absl::StatusOr<bool> IpcRegionRegistry::unregister_region(const std::string& region_id, int owner_pid, bool force) {
  RegionDescriptor desc;
  RegionCleanupCallback callback;
  {
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
    it->second.retiring = true;
    desc = it->second.desc;
    desc.poisoned = it->second.poisoned;
    callback = pre_cleanup_callback_;
  }
  if (callback) {
    callback(desc);
  }
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return true;
  }
  cleanup_region_record_locked(&it->second);
  regions_.erase(it);
  return true;
}

absl::StatusOr<IpcRegionRegistry::RegionDescriptor> IpcRegionRegistry::describe(const std::string& region_id) const {
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return absl::NotFoundError("region_id not found");
  }
  RegionDescriptor desc = it->second.desc;
  desc.poisoned = it->second.poisoned;
  return desc;
}

bool IpcRegionRegistry::refresh_ttl(const std::string& region_id, uint32_t ttl_ms) {
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return false;
  }
  if (it->second.desc.ttl_ms == 0) {
    it->second.desc.expires_at = absl::InfiniteFuture();
    return true;
  }
  if (ttl_ms == 0) {
    return false;
  }
  const uint32_t effective = clamp_ttl_ms(ttl_ms, opts_.max_ttl);
  if (effective == 0) {
    return false;
  }
  it->second.desc.ttl_ms = effective;
  it->second.desc.expires_at = absl::Now() + absl::Milliseconds(effective);
  return true;
}

std::vector<IpcRegionRegistry::RegionDescriptor> IpcRegionRegistry::sweep_expired(absl::Time now) {
  std::vector<RegionDescriptor> expired;
  std::vector<RegionDescriptor> pending_cleanup;
  RegionCleanupCallback callback;
  {
    absl::MutexLock lock(&mu_);
    callback = pre_cleanup_callback_;
    for (auto& [region_id, record] : regions_) {
      if (record.desc.expires_at > now) {
        continue;
      }
      if (record.refcount > 0) {
        if (record.desc.ttl_ms == 0) {
          record.desc.expires_at = absl::InfiniteFuture();
        } else {
          record.desc.expires_at = absl::Now() + absl::Milliseconds(record.desc.ttl_ms);
        }
        continue;
      }
      record.retiring = true;
      auto desc = record.desc;
      desc.poisoned = record.poisoned;
      pending_cleanup.push_back(std::move(desc));
    }
  }
  for (const auto& desc : pending_cleanup) {
    if (callback) {
      callback(desc);
    }
    absl::MutexLock lock(&mu_);
    auto it = regions_.find(desc.region_id);
    if (it == regions_.end()) {
      continue;
    }
    expired.push_back(it->second.desc);
    cleanup_region_record_locked(&it->second);
    regions_.erase(it);
  }
  return expired;
}

std::vector<IpcRegionRegistry::RegionDescriptor> IpcRegionRegistry::handle_pid_exit(int owner_pid) {
  std::vector<RegionDescriptor> removed;
  if (owner_pid <= 0) {
    return removed;
  }
  std::vector<RegionDescriptor> pending_cleanup;
  RegionCleanupCallback callback;
  {
    absl::MutexLock lock(&mu_);
    callback = pre_cleanup_callback_;
    for (auto& [region_id, record] : regions_) {
      if (record.desc.owner_pid != owner_pid) {
        continue;
      }
      record.retiring = true;
      auto desc = record.desc;
      desc.poisoned = record.poisoned;
      pending_cleanup.push_back(std::move(desc));
    }
  }
  for (const auto& desc : pending_cleanup) {
    if (callback) {
      callback(desc);
    }
    absl::MutexLock lock(&mu_);
    auto it = regions_.find(desc.region_id);
    if (it == regions_.end()) {
      continue;
    }
    removed.push_back(it->second.desc);
    cleanup_region_record_locked(&it->second);
    regions_.erase(it);
  }
  return removed;
}

absl::StatusOr<std::string> IpcRegionRegistry::get_attachment_bytes(const std::string& region_id) const {
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return absl::NotFoundError("region_id not found");
  }
  return it->second.attachment_bytes;
}

absl::StatusOr<std::string> IpcRegionRegistry::get_handle_bytes(const std::string& region_id) const {
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return absl::NotFoundError("region_id not found");
  }
  if (it->second.desc.memory_kind != MemoryKind::kVram) {
    return absl::FailedPreconditionError("region is not VRAM-backed");
  }
  return it->second.attachment_bytes;
}

absl::StatusOr<IpcRegionRegistry::RegionDescriptor> IpcRegionRegistry::acquire(
    const std::string& region_id,
    int owner_pid) {
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return absl::NotFoundError("region_id not found");
  }
  if (it->second.poisoned) {
    return absl::FailedPreconditionError("region is poisoned");
  }
  if (it->second.retiring) {
    return absl::FailedPreconditionError("region is retiring");
  }
  if (owner_pid != 0 && owner_pid != it->second.desc.owner_pid) {
    return absl::PermissionDeniedError("owner_pid mismatch");
  }
  ++it->second.refcount;
  if (it->second.desc.ttl_ms == 0) {
    it->second.desc.expires_at = absl::InfiniteFuture();
  } else {
    it->second.desc.expires_at = absl::Now() + absl::Milliseconds(it->second.desc.ttl_ms);
  }
  it->second.desc.poisoned = it->second.poisoned;
  return it->second.desc;
}

absl::Status IpcRegionRegistry::release(const std::string& region_id) {
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
    if (it->second.desc.ttl_ms == 0) {
      it->second.desc.expires_at = absl::InfiniteFuture();
    } else {
      it->second.desc.expires_at = absl::Now() + absl::Milliseconds(it->second.desc.ttl_ms);
    }
  }
  return absl::OkStatus();
}

absl::Status IpcRegionRegistry::mark_poisoned(const std::string& region_id) {
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return absl::NotFoundError("region_id not found");
  }
  it->second.poisoned = true;
  it->second.desc.poisoned = true;
  return absl::OkStatus();
}

bool IpcRegionRegistry::is_poisoned(const std::string& region_id) const {
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return false;
  }
  return it->second.poisoned;
}

absl::StatusOr<IpcRegionRegistry::HostSharedAttachment> IpcRegionRegistry::acquire_host_shared_attachment(
    const std::string& attach_token,
    int owner_pid) {
  absl::MutexLock lock(&mu_);
  auto token_it = attach_tokens_.find(attach_token);
  if (token_it == attach_tokens_.end()) {
    return absl::NotFoundError("attach_token not found");
  }
  auto region_it = regions_.find(token_it->second);
  if (region_it == regions_.end()) {
    return absl::NotFoundError("region_id not found");
  }
  RegionRecord& rec = region_it->second;
  if (rec.desc.memory_kind != MemoryKind::kHostShared) {
    return absl::FailedPreconditionError("attach_token does not reference HOST_SHARED region");
  }
  if (!rec.desc.daemon_managed || rec.daemon_host_shared_slab == nullptr) {
    return absl::FailedPreconditionError("HOST_SHARED region is not daemon-managed");
  }
  if (rec.poisoned) {
    return absl::FailedPreconditionError("region is poisoned");
  }
  if (rec.retiring) {
    return absl::FailedPreconditionError("region is retiring");
  }
  if (owner_pid <= 0 || owner_pid != rec.desc.owner_pid) {
    return absl::PermissionDeniedError("owner_pid mismatch");
  }
  ++rec.refcount;
  ++rec.host_shared_attachment_refcount;
  if (rec.desc.ttl_ms == 0) {
    rec.desc.expires_at = absl::InfiniteFuture();
  } else {
    rec.desc.expires_at = absl::Now() + absl::Milliseconds(rec.desc.ttl_ms);
  }
  rec.desc.poisoned = rec.poisoned;
  return HostSharedAttachment{
      .region = rec.desc,
      .attach_token = rec.desc.attach_token,
      .fd = rec.daemon_host_shared_slab->fd,
      .offset_bytes = 0,
      .size_bytes = rec.daemon_host_shared_slab->size_bytes,
  };
}

absl::Status IpcRegionRegistry::release_host_shared_attachment(const std::string& attach_token, int owner_pid) {
  absl::MutexLock lock(&mu_);
  auto token_it = attach_tokens_.find(attach_token);
  if (token_it == attach_tokens_.end()) {
    return absl::NotFoundError("attach_token not found");
  }
  auto region_it = regions_.find(token_it->second);
  if (region_it == regions_.end()) {
    return absl::NotFoundError("region_id not found");
  }
  RegionRecord& rec = region_it->second;
  if (rec.desc.memory_kind != MemoryKind::kHostShared) {
    return absl::FailedPreconditionError("attach_token does not reference HOST_SHARED region");
  }
  if (owner_pid <= 0 || owner_pid != rec.desc.owner_pid) {
    return absl::PermissionDeniedError("owner_pid mismatch");
  }
  if (rec.host_shared_attachment_refcount == 0) {
    return absl::FailedPreconditionError("HOST_SHARED attachment was not acquired");
  }
  if (rec.refcount == 0) {
    return absl::InternalError("HOST_SHARED attachment bookkeeping is inconsistent");
  }
  --rec.host_shared_attachment_refcount;
  --rec.refcount;
  if (rec.refcount == 0) {
    if (rec.desc.ttl_ms == 0) {
      rec.desc.expires_at = absl::InfiniteFuture();
    } else {
      rec.desc.expires_at = absl::Now() + absl::Milliseconds(rec.desc.ttl_ms);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<IpcRegionRegistry::HostSharedLocalMapping> IpcRegionRegistry::acquire_host_shared_local_mapping(
    const std::string& region_id,
    int owner_pid) {
  absl::MutexLock lock(&mu_);
  auto it = regions_.find(region_id);
  if (it == regions_.end()) {
    return absl::NotFoundError("region_id not found");
  }
  RegionRecord& rec = it->second;
  if (rec.poisoned) {
    return absl::FailedPreconditionError("region is poisoned");
  }
  if (rec.retiring) {
    return absl::FailedPreconditionError("region is retiring");
  }
  if (owner_pid != 0 && owner_pid != rec.desc.owner_pid) {
    return absl::PermissionDeniedError("owner_pid mismatch");
  }
  if (rec.desc.memory_kind != MemoryKind::kHostShared) {
    return absl::FailedPreconditionError("region is not HOST_SHARED");
  }
  if (!rec.desc.daemon_managed || rec.daemon_host_shared_slab == nullptr ||
      rec.daemon_host_shared_slab->base == nullptr) {
    return absl::FailedPreconditionError("HOST_SHARED region is not daemon-managed");
  }
  ++rec.refcount;
  if (rec.desc.ttl_ms == 0) {
    rec.desc.expires_at = absl::InfiniteFuture();
  } else {
    rec.desc.expires_at = absl::Now() + absl::Milliseconds(rec.desc.ttl_ms);
  }
  rec.desc.poisoned = rec.poisoned;
  return HostSharedLocalMapping{
      .region = rec.desc,
      .base_ptr = rec.daemon_host_shared_slab->base,
      .size_bytes = rec.daemon_host_shared_slab->size_bytes,
  };
}

std::string IpcRegionRegistry::mint_region_id_locked() {
  for (;;) {
    const uint64_t hi = absl::Uniform<uint64_t>(bitgen_);
    const uint64_t lo = absl::Uniform<uint64_t>(bitgen_);
    const std::string id = absl::StrFormat("region:%016x%016x", hi, lo);
    if (!regions_.contains(id)) {
      return id;
    }
  }
}

std::string IpcRegionRegistry::mint_attach_token_locked() {
  for (;;) {
    const uint64_t hi = absl::Uniform<uint64_t>(bitgen_);
    const uint64_t lo = absl::Uniform<uint64_t>(bitgen_);
    const std::string token = absl::StrFormat("region-attach:%016x%016x", hi, lo);
    if (!attach_tokens_.contains(token)) {
      return token;
    }
  }
}

void IpcRegionRegistry::cleanup_region_record_locked(RegionRecord* rec) {
  if (rec == nullptr) {
    return;
  }
  if (!rec->desc.attach_token.empty()) {
    attach_tokens_.erase(rec->desc.attach_token);
  }
  if (rec->daemon_host_shared_slab != nullptr) {
    if (rec->daemon_host_shared_slab->base != nullptr && rec->daemon_host_shared_slab->size_bytes != 0) {
      (void)::munmap(rec->daemon_host_shared_slab->base, static_cast<size_t>(rec->daemon_host_shared_slab->size_bytes));
    }
    if (rec->daemon_host_shared_slab->fd >= 0) {
      (void)::close(rec->daemon_host_shared_slab->fd);
    }
    rec->daemon_host_shared_slab.reset();
  }
}

} // namespace tensorcast::daemon
