// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/handle_lease_registry.h"

#include <cerrno>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#include "absl/log/log.h"
#include "absl/random/distributions.h"
#include "absl/strings/str_cat.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {
namespace {

constexpr size_t kLeaseTokenBytes = 32;
constexpr uint64_t kFdWarnHeadroom = 128;
constexpr uint64_t kFdFailHeadroom = 32;

std::string export_capability_id(std::string_view lease_token) {
  return absl::StrCat("export:", lease_token);
}

std::string export_subject_id(const store::loading::ReplicaKey& key) {
  return absl::StrCat(
      "backing:",
      key.artifact_id,
      "|",
      key.view_id.value_or(""),
      "|",
      static_cast<int>(key.device.type),
      "|",
      key.device.ordinal,
      "|",
      key.replica);
}

absl::Status errno_to_status_for_fd_ops(int err, const char* what) {
  if (err == EMFILE || err == ENFILE) {
    return absl::ResourceExhaustedError(absl::StrCat(what, ": file descriptor limit reached (RLIMIT_NOFILE)"));
  }
  return absl::ErrnoToStatus(err, what);
}

absl::StatusOr<uint64_t> count_open_fds() {
  DIR* d = ::opendir("/proc/self/fd");
  if (d == nullptr) {
    return absl::ErrnoToStatus(errno, "opendir(/proc/self/fd) failed");
  }
  uint64_t count = 0;
  for (;;) {
    errno = 0;
    const dirent* ent = ::readdir(d);
    if (ent == nullptr) {
      if (errno != 0) {
        const int err = errno;
        ::closedir(d);
        return absl::ErrnoToStatus(err, "readdir(/proc/self/fd) failed");
      }
      break;
    }
    // Skip "." and "..".
    if (ent->d_name[0] == '.') {
      if (ent->d_name[1] == '\0') {
        continue;
      }
      if (ent->d_name[1] == '.' && ent->d_name[2] == '\0') {
        continue;
      }
    }
    count++;
  }
  ::closedir(d);
  return count;
}

absl::Status ensure_fd_headroom(const char* what) {
  rlimit rl{};
  if (::getrlimit(RLIMIT_NOFILE, &rl) != 0) {
    return absl::OkStatus();
  }
  if (rl.rlim_cur == RLIM_INFINITY) {
    return absl::OkStatus();
  }
  const uint64_t limit = static_cast<uint64_t>(rl.rlim_cur);
  const auto open_or = count_open_fds();
  if (!open_or.ok()) {
    return absl::OkStatus();
  }
  const uint64_t open = *open_or;
  const uint64_t headroom = (limit > open) ? (limit - open) : 0;
  if (headroom < kFdFailHeadroom) {
    return absl::ResourceExhaustedError(
        absl::StrCat(
            what,
            ": refusing to allocate new FDs (RLIMIT_NOFILE headroom too low): open_fds=",
            open,
            " limit=",
            limit,
            " headroom=",
            headroom,
            " min_headroom=",
            kFdFailHeadroom));
  }
  if (headroom < kFdWarnHeadroom) {
    LOG(WARNING) << what << ": RLIMIT_NOFILE headroom low: open_fds=" << open << " limit=" << limit
                 << " headroom=" << headroom;
  }
  return absl::OkStatus();
}

absl::StatusOr<int> dup_fd_cloexec(int fd) {
  if (fd < 0) {
    return absl::InvalidArgumentError("fd must be >= 0");
  }
#ifdef F_DUPFD_CLOEXEC
  int duped = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
  if (duped < 0) {
    return errno_to_status_for_fd_ops(errno, "fcntl(F_DUPFD_CLOEXEC) failed");
  }
  return duped;
#else
  int duped = ::dup(fd);
  if (duped < 0) {
    return errno_to_status_for_fd_ops(errno, "dup failed");
  }
  int flags = ::fcntl(duped, F_GETFD);
  if (flags < 0) {
    const int err = errno;
    ::close(duped);
    return errno_to_status_for_fd_ops(err, "fcntl(F_GETFD) failed");
  }
  if (::fcntl(duped, F_SETFD, flags | FD_CLOEXEC) < 0) {
    const int err = errno;
    ::close(duped);
    return errno_to_status_for_fd_ops(err, "fcntl(F_SETFD, FD_CLOEXEC) failed");
  }
  return duped;
#endif
}

absl::StatusOr<std::vector<uint32_t>> copy_chunks(absl::Span<const uint32_t> chunks) {
  if (chunks.empty()) {
    return absl::InvalidArgumentError("exported_chunks must be non-empty");
  }
  std::vector<uint32_t> out;
  out.reserve(chunks.size());
  for (uint32_t c : chunks) {
    out.push_back(c);
  }
  return out;
}

absl::StatusOr<std::vector<uint32_t>> build_export_chunks_for_replica(
    store::StoreEngine* engine,
    const store::loading::ReplicaKey& key) {
  if (engine == nullptr) {
    return absl::FailedPreconditionError("engine is unavailable");
  }
  auto size_or = engine->get_replica_size(key);
  if (!size_or.ok()) {
    return size_or.status();
  }
  const uint64_t size_bytes = *size_or;
  const uint64_t chunk_bytes = static_cast<uint64_t>(engine->get_artifact_chunk_bytes());
  if (chunk_bytes == 0) {
    return absl::FailedPreconditionError("artifact_chunk_bytes is zero");
  }
  const uint64_t num_chunks = (size_bytes + chunk_bytes - 1) / chunk_bytes;
  if (num_chunks == 0) {
    return absl::InvalidArgumentError("replica size is zero");
  }
  if (num_chunks > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return absl::InvalidArgumentError("replica has too many chunks");
  }
  std::vector<uint32_t> chunks;
  chunks.reserve(static_cast<size_t>(num_chunks));
  for (uint32_t i = 0; i < static_cast<uint32_t>(num_chunks); ++i) {
    chunks.push_back(i);
  }
  return chunks;
}

} // namespace

HandleLeaseRegistry::HandleLeaseRegistry(
    Options opts,
    store::StoreEngine& engine,
    SessionLifecycleManager& lifecycle,
    LifecycleKernel& lifecycle_kernel)
    : opts_(std::move(opts)), engine_(&engine), lifecycle_(&lifecycle), lifecycle_kernel_(&lifecycle_kernel) {
  meter_ = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  active_leases_gauge_ = meter_->CreateDoubleObservableGauge("tc_handle_leases_active_gauge");
  active_leases_gauge_->AddCallback(&HandleLeaseRegistry::active_leases_gauge_callback, this);
  cpu_exports_gauge_ = meter_->CreateDoubleObservableGauge("tc_handle_cpu_exports_active_gauge");
  cpu_exports_gauge_->AddCallback(&HandleLeaseRegistry::cpu_exports_gauge_callback, this);
}

void HandleLeaseRegistry::active_leases_gauge_callback(
    opentelemetry::metrics::ObserverResult result,
    void* state) noexcept {
  auto* self = static_cast<HandleLeaseRegistry*>(state);
  if (self == nullptr) {
    return;
  }
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }

  double cuda_ipc_count = 0.0;
  double cpu_memfd_count = 0.0;
  double external_count = 0.0;
  {
    absl::MutexLock lock(&self->mu_);
    for (const auto& [token, rec] : self->leases_) {
      (void)token;
      if (rec.kind == HandleKind::kCudaIpc) {
        cuda_ipc_count += 1.0;
      } else if (rec.kind == HandleKind::kCpuMemfd) {
        cpu_memfd_count += 1.0;
      } else if (rec.kind == HandleKind::kExternal) {
        external_count += 1.0;
      }
    }
  }

  obs->Observe(cuda_ipc_count, {{"kind", opentelemetry::common::AttributeValue("cuda_ipc")}});
  obs->Observe(cpu_memfd_count, {{"kind", opentelemetry::common::AttributeValue("cpu_memfd")}});
  obs->Observe(external_count, {{"kind", opentelemetry::common::AttributeValue("external_cuda")}});
}

void HandleLeaseRegistry::cpu_exports_gauge_callback(
    opentelemetry::metrics::ObserverResult result,
    void* state) noexcept {
  auto* self = static_cast<HandleLeaseRegistry*>(state);
  if (self == nullptr) {
    return;
  }
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  double count = 0.0;
  {
    absl::MutexLock lock(&self->mu_);
    count = static_cast<double>(self->cpu_exports_.size());
  }
  obs->Observe(count);
}

std::string HandleLeaseRegistry::mint_token_locked() {
  std::string token;
  token.resize(kLeaseTokenBytes);
  for (;;) {
    for (size_t i = 0; i < kLeaseTokenBytes; ++i) {
      token[i] = static_cast<char>(absl::Uniform<uint32_t>(bitgen_, 0u, 256u));
    }
    if (leases_.find(token) == leases_.end()) {
      return token;
    }
  }
}

absl::Status HandleLeaseRegistry::maybe_rate_limit_mint_locked(absl::Time now) {
  if (opts_.max_mints_per_second == 0) {
    return absl::OkStatus();
  }
  if (mint_window_start_ == absl::UnixEpoch() || now < mint_window_start_ ||
      now - mint_window_start_ >= absl::Seconds(1)) {
    mint_window_start_ = now;
    mint_window_count_ = 0;
  }
  if (mint_window_count_ >= opts_.max_mints_per_second) {
    return absl::ResourceExhaustedError("handle lease mint rate limit exceeded");
  }
  mint_window_count_ += 1;
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<HandleLeaseRegistry::CpuExportState>> HandleLeaseRegistry::acquire_cpu_export_state_(
    const store::loading::ReplicaKey& key,
    CpuMemfdDescriptor memfd,
    absl::Span<const uint32_t> exported_chunks) {
  if (memfd.fd < 0) {
    return absl::InvalidArgumentError("memfd.fd must be >= 0");
  }
  if (memfd.size_bytes == 0) {
    return absl::InvalidArgumentError("memfd.size_bytes must be > 0");
  }
  if (engine_ == nullptr) {
    return absl::FailedPreconditionError("engine is unavailable");
  }

  std::shared_ptr<CpuExportState> state;
  bool init_needed = false;
  {
    absl::MutexLock lock(&mu_);
    auto it = cpu_exports_.find(key);
    if (it != cpu_exports_.end()) {
      state = it->second;
      state->refcount++;
    } else {
      if (cpu_exports_.size() >= opts_.capacity) {
        return absl::ResourceExhaustedError("CPU export registry capacity exceeded");
      }
      state = std::make_shared<CpuExportState>();
      state->ready = false;
      state->status = absl::OkStatus();
      state->refcount = 1;
      state->memfd = CpuMemfdDescriptor{.fd = -1, .size_bytes = memfd.size_bytes, .offset_bytes = memfd.offset_bytes};
      auto chunks_or = copy_chunks(exported_chunks);
      if (!chunks_or.ok()) {
        return chunks_or.status();
      }
      state->chunks = std::move(*chunks_or);
      cpu_exports_.emplace(key, state);
      init_needed = true;
    }
  }

  if (!init_needed) {
    absl::Status st = absl::OkStatus();
    {
      absl::MutexLock lock(&mu_);
      while (!state->ready) {
        state->cv.Wait(&mu_);
      }
      if (state->status.ok()) {
        return state;
      }
      st = state->status;
    }
    release_cpu_export_state_(key, state);
    return st;
  }

  // Initialize export outside lock: duplicate fd, acquire stable lease, set exported.
  absl::Status init_status = absl::OkStatus();
  std::optional<store::replica::UnifiedMemoryAuthority::StableLease> stable_lease;
  std::shared_ptr<void> keepalive;
  int duped_fd = -1;

  if (auto st = ensure_fd_headroom("HandleLeaseRegistry CPU export"); !st.ok()) {
    init_status = st;
  }

  if (init_status.ok()) {
    auto dup_or = dup_fd_cloexec(memfd.fd);
    if (!dup_or.ok()) {
      init_status = dup_or.status();
    } else {
      duped_fd = *dup_or;
    }
  }

  if (init_status.ok()) {
    auto stable_or = engine_->acquire_replica_stable_lease(key, exported_chunks);
    if (!stable_or.ok()) {
      init_status = stable_or.status();
    } else {
      stable_lease = std::move(*stable_or);
      auto export_or =
          engine_->set_replica_exported(key, common::memory::MemoryLocation::CPU, exported_chunks, /*on=*/true);
      if (!export_or.ok()) {
        auto rel = engine_->release_replica_stable_lease(*stable_lease);
        LOG_IF(WARNING, !rel.ok()) << "release_replica_stable_lease failed after set_exported failure: " << rel;
        stable_lease.reset();
        init_status = export_or.status();
      } else {
        keepalive = std::move(export_or->keepalive);
      }
    }
  }

  {
    absl::MutexLock lock(&mu_);
    // State may have been removed (e.g., shutdown); treat as aborted.
    auto it = cpu_exports_.find(key);
    if (it == cpu_exports_.end() || it->second.get() != state.get()) {
      if (duped_fd >= 0) {
        ::close(duped_fd);
      }
      if (stable_lease.has_value()) {
        auto rel = engine_->release_replica_stable_lease(*stable_lease);
        LOG_IF(WARNING, !rel.ok()) << "release_replica_stable_lease failed: " << rel;
      }
      return absl::AbortedError("CPU export state disappeared during initialization");
    }
    // Install results into state.
    state->status = init_status;
    if (init_status.ok()) {
      state->memfd.fd = duped_fd;
      state->stable_lease = std::move(stable_lease);
      state->export_keepalive = std::move(keepalive);
    } else {
      if (duped_fd >= 0) {
        ::close(duped_fd);
      }
    }
    state->ready = true;
    state->cv.SignalAll();
  }

  if (!init_status.ok()) {
    release_cpu_export_state_(key, state);
    return init_status;
  }

  return state;
}

void HandleLeaseRegistry::release_cpu_export_state_(
    const store::loading::ReplicaKey& key,
    const std::shared_ptr<CpuExportState>& state) {
  if (!state) {
    return;
  }

  std::optional<store::replica::UnifiedMemoryAuthority::StableLease> stable_lease;
  std::vector<uint32_t> chunks;
  int fd_to_close = -1;
  bool do_cleanup = false;

  {
    absl::MutexLock lock(&mu_);
    auto it = cpu_exports_.find(key);
    if (it == cpu_exports_.end()) {
      return;
    }
    if (it->second.get() != state.get()) {
      return;
    }
    if (state->refcount == 0) {
      return;
    }
    state->refcount--;
    if (state->refcount != 0) {
      return;
    }
    // Remove from registry and cleanup outside lock.
    stable_lease = state->stable_lease;
    chunks = state->chunks;
    fd_to_close = state->memfd.fd;
    cpu_exports_.erase(it);
    do_cleanup = true;
  }

  if (!do_cleanup || engine_ == nullptr) {
    if (fd_to_close >= 0) {
      ::close(fd_to_close);
    }
    return;
  }

  if (!chunks.empty()) {
    auto st = engine_->set_replica_exported(key, common::memory::MemoryLocation::CPU, chunks, /*on=*/false);
    LOG_IF(WARNING, !st.ok()) << "set_replica_exported(off) failed: " << st.status();
  }
  if (stable_lease.has_value()) {
    auto st = engine_->release_replica_stable_lease(*stable_lease);
    LOG_IF(WARNING, !st.ok()) << "release_replica_stable_lease failed: " << st;
  }
  if (fd_to_close >= 0) {
    ::close(fd_to_close);
  }
}

absl::StatusOr<std::shared_ptr<HandleLeaseRegistry::GpuExportState>> HandleLeaseRegistry::acquire_gpu_export_state_(
    const store::loading::ReplicaKey& key) {
  if (engine_ == nullptr) {
    return absl::FailedPreconditionError("engine is unavailable");
  }

  std::shared_ptr<GpuExportState> state;
  bool init_needed = false;
  {
    absl::MutexLock lock(&mu_);
    auto it = gpu_exports_.find(key);
    if (it != gpu_exports_.end()) {
      state = it->second;
      state->refcount++;
    } else {
      if (gpu_exports_.size() >= opts_.capacity) {
        return absl::ResourceExhaustedError("GPU export registry capacity exceeded");
      }
      state = std::make_shared<GpuExportState>();
      state->ready = false;
      state->status = absl::OkStatus();
      state->refcount = 1;
      gpu_exports_.emplace(key, state);
      init_needed = true;
    }
  }

  if (!init_needed) {
    absl::Status st = absl::OkStatus();
    {
      absl::MutexLock lock(&mu_);
      while (!state->ready) {
        state->cv.Wait(&mu_);
      }
      if (state->status.ok()) {
        return state;
      }
      st = state->status;
    }
    release_gpu_export_state_(key, state);
    return st;
  }

  absl::Status init_status = absl::OkStatus();
  std::vector<uint32_t> chunks;
  if (init_status.ok()) {
    auto chunks_or = build_export_chunks_for_replica(engine_, key);
    if (!chunks_or.ok()) {
      init_status = chunks_or.status();
    } else {
      chunks = std::move(*chunks_or);
    }
  }
  if (init_status.ok()) {
    auto export_or = engine_->set_replica_exported(key, common::memory::MemoryLocation::GPU, chunks, /*on=*/true);
    if (!export_or.ok()) {
      init_status = export_or.status();
    }
  }

  {
    absl::MutexLock lock(&mu_);
    auto it = gpu_exports_.find(key);
    if (it == gpu_exports_.end() || it->second.get() != state.get()) {
      if (init_status.ok()) {
        auto st = engine_->set_replica_exported(key, common::memory::MemoryLocation::GPU, chunks, /*on=*/false);
        LOG_IF(WARNING, !st.ok()) << "set_replica_exported(off) failed after state disappearance: " << st.status();
      }
      return absl::AbortedError("GPU export state disappeared during initialization");
    }
    state->status = init_status;
    if (init_status.ok()) {
      state->chunks = std::move(chunks);
    }
    state->ready = true;
    state->cv.SignalAll();
  }

  if (!init_status.ok()) {
    release_gpu_export_state_(key, state);
    return init_status;
  }

  return state;
}

void HandleLeaseRegistry::release_gpu_export_state_(
    const store::loading::ReplicaKey& key,
    const std::shared_ptr<GpuExportState>& state) {
  if (!state) {
    return;
  }

  std::vector<uint32_t> chunks;
  bool do_cleanup = false;
  {
    absl::MutexLock lock(&mu_);
    auto it = gpu_exports_.find(key);
    if (it == gpu_exports_.end()) {
      return;
    }
    if (it->second.get() != state.get()) {
      return;
    }
    if (state->refcount == 0) {
      return;
    }
    state->refcount--;
    if (state->refcount != 0) {
      return;
    }
    chunks = state->chunks;
    gpu_exports_.erase(it);
    do_cleanup = true;
  }

  if (!do_cleanup || engine_ == nullptr || chunks.empty()) {
    return;
  }
  auto st = engine_->set_replica_exported(key, common::memory::MemoryLocation::GPU, chunks, /*on=*/false);
  if (!st.ok() && st.status().code() != absl::StatusCode::kNotFound) {
    LOG(WARNING) << "set_replica_exported(off) failed for GPU export: " << st.status();
  }
}

absl::StatusOr<std::string> HandleLeaseRegistry::mint_cuda_ipc_lease(const store::loading::ReplicaKey& key, pid_t pid) {
  if (pid <= 0) {
    return absl::InvalidArgumentError("pid must be > 0");
  }
  if (lifecycle_ == nullptr) {
    return absl::FailedPreconditionError("lifecycle manager is unavailable");
  }

  auto gpu_state_or = acquire_gpu_export_state_(key);
  if (!gpu_state_or.ok()) {
    return gpu_state_or.status();
  }
  auto gpu_state = *gpu_state_or;

  const absl::Duration ttl = opts_.ttl;
  const bool ttl_enabled = ttl > absl::ZeroDuration();
  std::string token;
  absl::Status mint_status = absl::OkStatus();
  {
    absl::MutexLock lock(&mu_);
    if (leases_.size() >= opts_.capacity) {
      mint_status = absl::ResourceExhaustedError("handle lease registry capacity exceeded");
    } else {
      mint_status = maybe_rate_limit_mint_locked(absl::Now());
    }
    if (mint_status.ok()) {
      token = mint_token_locked();
      leases_[token] = LeaseRecord{
          .kind = HandleKind::kCudaIpc,
          .key = key,
          .lease_id = 0,
          .cpu_export_state = {},
          .gpu_export_state = gpu_state};
    }
  }
  if (!mint_status.ok()) {
    release_gpu_export_state_(key, gpu_state);
    return mint_status;
  }

  auto cleanup = [this, token, key, gpu_state]() -> absl::Status {
    {
      absl::MutexLock lock(&mu_);
      leases_.erase(token);
    }
    if (lifecycle_kernel_ != nullptr) {
      auto st = lifecycle_kernel_->release_capability(export_capability_id(token));
      LOG_IF(WARNING, !st.ok()) << "export cleanup failed to release lifecycle capability: " << st;
    }
    release_gpu_export_state_(key, gpu_state);
    return absl::OkStatus();
  };

  auto id_or = ttl_enabled ? lifecycle_->create_ttl_use_lease(key, pid, ttl, {std::move(cleanup)})
                           : lifecycle_->create_use_lease(key, pid, {std::move(cleanup)});
  if (!id_or.ok()) {
    {
      absl::MutexLock lock(&mu_);
      leases_.erase(token);
    }
    release_gpu_export_state_(key, gpu_state);
    return id_or.status();
  }

  {
    absl::MutexLock lock(&mu_);
    auto it = leases_.find(token);
    if (it == leases_.end()) {
      // Lease retired before registration; do not hand out a dangling token.
      lifecycle_->release_lease(*id_or);
      release_gpu_export_state_(key, gpu_state);
      return absl::AbortedError("lease retired before token registration");
    }
    it->second.lease_id = *id_or;
  }

  if (lifecycle_kernel_ != nullptr) {
    const absl::Time issued_at = absl::Now();
    LifecycleSubjectRecord subject;
    subject.subject_id = export_subject_id(key);
    subject.epochs.subject_generation = *id_or;
    subject.subject_kind = LifecycleSubjectKind::kBacking;
    subject.created_at = issued_at;
    subject.last_observed_at = issued_at;
    subject.artifact_id = key.artifact_id;
    subject.semantic_ref_id = subject.subject_id;
    auto mint_or = lifecycle_kernel_->mint_capability(
        MintCapabilityRequest{
            .subject = subject,
            .address =
                CapabilityBindingAddress{
                    .route_principal = make_issuer_route_principal(lifecycle_kernel_->issuer_daemon_id()),
                    .family = LifecycleCapabilityFamily::kExport,
                    .binding_space = LifecycleBindingSpace::kExportHandle,
                    .binding_key_kind = BindingKeyKind::kOpaqueLocalToken,
                    .binding_key = token,
                    .epochs = subject.epochs,
                    .binding_id = token,
                },
            .front_door_kind = LifecycleFrontDoorKind::kLocalCudaIpcExport,
            .capability_id = export_capability_id(token),
            .lease_id = *id_or,
            .capability_expires_at = ttl_enabled ? issued_at + ttl : absl::InfiniteFuture(),
            .carriage_kind = CredentialCarriageKind::kOpaqueLocalCompat,
            .binding_mode = LifecycleBindingMode::kBindingRecord,
            .constraint_claims =
                ConstraintClaims{
                    .artifact_id = key.artifact_id,
                    .local_only = true,
                },
            .credential_expires_at = std::optional<absl::Time>(ttl_enabled ? issued_at + ttl : absl::InfiniteFuture()),
            .binding_id = token,
            .local_only = true,
            .workflow_gate = WorkflowGateKind::kNone,
        });
    if (!mint_or.ok()) {
      lifecycle_->release_lease(*id_or);
      return mint_or.status();
    }
  }

  return token;
}

absl::StatusOr<std::string> HandleLeaseRegistry::mint_cpu_memfd_lease(
    const store::loading::ReplicaKey& key,
    pid_t pid,
    CpuMemfdDescriptor memfd,
    absl::Span<const uint32_t> exported_chunks) {
  if (pid <= 0) {
    return absl::InvalidArgumentError("pid must be > 0");
  }
  if (lifecycle_ == nullptr) {
    return absl::FailedPreconditionError("lifecycle manager is unavailable");
  }

  auto state_or = acquire_cpu_export_state_(key, memfd, exported_chunks);
  if (!state_or.ok()) {
    return state_or.status();
  }
  auto state = *state_or;

  const absl::Duration ttl = opts_.ttl;
  const bool ttl_enabled = ttl > absl::ZeroDuration();
  std::string token;
  absl::Status mint_status = absl::OkStatus();
  {
    absl::MutexLock lock(&mu_);
    if (leases_.size() >= opts_.capacity) {
      mint_status = absl::ResourceExhaustedError("handle lease registry capacity exceeded");
    } else {
      mint_status = maybe_rate_limit_mint_locked(absl::Now());
    }
    if (mint_status.ok()) {
      token = mint_token_locked();
      leases_[token] = LeaseRecord{.kind = HandleKind::kCpuMemfd, .key = key, .lease_id = 0, .cpu_export_state = state};
    }
  }
  if (!mint_status.ok()) {
    release_cpu_export_state_(key, state);
    return mint_status;
  }

  auto cleanup = [this, token, key, state]() -> absl::Status {
    {
      absl::MutexLock lock(&mu_);
      leases_.erase(token);
    }
    if (lifecycle_kernel_ != nullptr) {
      auto st = lifecycle_kernel_->release_capability(export_capability_id(token));
      LOG_IF(WARNING, !st.ok()) << "export cleanup failed to release lifecycle capability: " << st;
    }
    release_cpu_export_state_(key, state);
    return absl::OkStatus();
  };

  auto id_or = ttl_enabled ? lifecycle_->create_ttl_use_lease(key, pid, ttl, {std::move(cleanup)})
                           : lifecycle_->create_use_lease(key, pid, {std::move(cleanup)});
  if (!id_or.ok()) {
    {
      absl::MutexLock lock(&mu_);
      leases_.erase(token);
    }
    release_cpu_export_state_(key, state);
    return id_or.status();
  }

  {
    absl::MutexLock lock(&mu_);
    auto it = leases_.find(token);
    if (it == leases_.end()) {
      lifecycle_->release_lease(*id_or);
      release_cpu_export_state_(key, state);
      return absl::AbortedError("lease retired before token registration");
    }
    it->second.lease_id = *id_or;
  }

  if (lifecycle_kernel_ != nullptr) {
    const absl::Time issued_at = absl::Now();
    LifecycleSubjectRecord subject;
    subject.subject_id = export_subject_id(key);
    subject.epochs.subject_generation = *id_or;
    subject.subject_kind = LifecycleSubjectKind::kBacking;
    subject.created_at = issued_at;
    subject.last_observed_at = issued_at;
    subject.artifact_id = key.artifact_id;
    subject.semantic_ref_id = subject.subject_id;
    auto mint_or = lifecycle_kernel_->mint_capability(
        MintCapabilityRequest{
            .subject = subject,
            .address =
                CapabilityBindingAddress{
                    .route_principal = make_issuer_route_principal(lifecycle_kernel_->issuer_daemon_id()),
                    .family = LifecycleCapabilityFamily::kExport,
                    .binding_space = LifecycleBindingSpace::kExportHandle,
                    .binding_key_kind = BindingKeyKind::kOpaqueLocalToken,
                    .binding_key = token,
                    .epochs = subject.epochs,
                    .binding_id = token,
                },
            .front_door_kind = LifecycleFrontDoorKind::kLocalCpuMemfdExport,
            .capability_id = export_capability_id(token),
            .lease_id = *id_or,
            .capability_expires_at = ttl_enabled ? issued_at + ttl : absl::InfiniteFuture(),
            .carriage_kind = CredentialCarriageKind::kOpaqueLocalCompat,
            .binding_mode = LifecycleBindingMode::kBindingRecord,
            .constraint_claims =
                ConstraintClaims{
                    .artifact_id = key.artifact_id,
                    .local_only = true,
                },
            .credential_expires_at = std::optional<absl::Time>(ttl_enabled ? issued_at + ttl : absl::InfiniteFuture()),
            .binding_id = token,
            .local_only = true,
            .workflow_gate = WorkflowGateKind::kNone,
        });
    if (!mint_or.ok()) {
      lifecycle_->release_lease(*id_or);
      return mint_or.status();
    }
  }

  return token;
}

absl::StatusOr<std::string> HandleLeaseRegistry::mint_external_cuda_lease(pid_t pid, std::function<void()> cleanup) {
  if (pid <= 0) {
    return absl::InvalidArgumentError("pid must be > 0");
  }
  if (!cleanup) {
    return absl::InvalidArgumentError("cleanup is required");
  }
  if (lifecycle_ == nullptr) {
    return absl::FailedPreconditionError("lifecycle manager is unavailable");
  }

  std::string token;
  {
    absl::MutexLock lock(&mu_);
    if (leases_.size() >= opts_.capacity) {
      return absl::ResourceExhaustedError("handle lease registry capacity exceeded");
    }
    auto mint_status = maybe_rate_limit_mint_locked(absl::Now());
    if (!mint_status.ok()) {
      return mint_status;
    }
    token = mint_token_locked();
    leases_[token] = LeaseRecord{
        .kind = HandleKind::kExternal,
        .key = {},
        .lease_id = 0,
        .cpu_export_state = {},
        .gpu_export_state = {},
        .external_owner_pid = pid,
        .external_cleanup = std::move(cleanup),
    };
  }
  lifecycle_->watch_pid(pid);
  return token;
}

absl::Status HandleLeaseRegistry::release(const std::string& lease_token) {
  SessionLifecycleManager::LeaseId id = 0;
  pid_t external_owner_pid = 0;
  std::function<void()> external_cleanup;
  {
    absl::MutexLock lock(&mu_);
    auto it = leases_.find(lease_token);
    if (it == leases_.end()) {
      return absl::NotFoundError("lease_token not found");
    }
    if (it->second.kind == HandleKind::kExternal) {
      external_owner_pid = it->second.external_owner_pid;
      external_cleanup = std::move(it->second.external_cleanup);
      leases_.erase(it);
    } else {
      id = it->second.lease_id;
    }
  }
  if (external_cleanup) {
    external_cleanup();
    if (lifecycle_ != nullptr && external_owner_pid > 0) {
      lifecycle_->unwatch_pid(external_owner_pid);
    }
    return absl::OkStatus();
  }
  lifecycle_->release_lease(id);
  return absl::OkStatus();
}

void HandleLeaseRegistry::handle_pid_exit(pid_t pid) {
  if (pid <= 0) {
    return;
  }
  std::vector<std::function<void()>> cleanups;
  {
    absl::MutexLock lock(&mu_);
    for (auto it = leases_.begin(); it != leases_.end();) {
      if (it->second.kind != HandleKind::kExternal || it->second.external_owner_pid != pid) {
        ++it;
        continue;
      }
      if (it->second.external_cleanup) {
        cleanups.push_back(std::move(it->second.external_cleanup));
      }
      auto erase_it = it;
      ++it;
      leases_.erase(erase_it);
    }
  }
  for (auto& cleanup : cleanups) {
    if (cleanup) {
      cleanup();
    }
  }
}

absl::StatusOr<HandleLeaseRegistry::CpuMemfdDescriptor> HandleLeaseRegistry::get_cpu_memfd_descriptor(
    const std::string& lease_token) const {
  absl::MutexLock lock(&mu_);
  auto it = leases_.find(lease_token);
  if (it == leases_.end()) {
    return absl::NotFoundError("lease_token not found");
  }
  if (it->second.kind != HandleKind::kCpuMemfd) {
    return absl::FailedPreconditionError("lease_token is not a CPU memfd lease");
  }
  const auto& state = it->second.cpu_export_state;
  if (!state) {
    return absl::FailedPreconditionError("cpu export state is unavailable for lease_token");
  }
  if (!state->ready) {
    return absl::FailedPreconditionError("cpu export state is not ready");
  }
  if (!state->status.ok()) {
    return state->status;
  }
  if (state->memfd.fd < 0) {
    return absl::FailedPreconditionError("cpu export memfd is invalid");
  }
  return state->memfd;
}

absl::StatusOr<ParsedCredential> HandleLeaseRegistry::build_parsed_credential(
    const std::string& lease_token,
    LifecycleFrontDoorKind front_door_kind,
    absl::Time now) const {
  (void)now;
  absl::MutexLock lock(&mu_);
  auto it = leases_.find(lease_token);
  if (it == leases_.end()) {
    return absl::NotFoundError("lease_token not found");
  }
  if (front_door_kind == LifecycleFrontDoorKind::kLocalCpuMemfdExport && it->second.kind != HandleKind::kCpuMemfd) {
    return absl::FailedPreconditionError("lease_token is not a CPU memfd export token");
  }
  if (front_door_kind == LifecycleFrontDoorKind::kLocalCudaIpcExport && it->second.kind != HandleKind::kCudaIpc) {
    return absl::FailedPreconditionError("lease_token is not a CUDA IPC export token");
  }
  if (lifecycle_kernel_ == nullptr) {
    return absl::FailedPreconditionError("lifecycle kernel is unavailable");
  }
  return ParsedCredential{
      .address =
          CapabilityBindingAddress{
              .route_principal = make_issuer_route_principal(lifecycle_kernel_->issuer_daemon_id()),
              .family = LifecycleCapabilityFamily::kExport,
              .binding_space = LifecycleBindingSpace::kExportHandle,
              .binding_key_kind = BindingKeyKind::kOpaqueLocalToken,
              .binding_key = lease_token,
              .epochs =
                  LifecycleEpochs{
                      .subject_generation = it->second.lease_id == 0 ? 1 : it->second.lease_id,
                  },
              .binding_id = lease_token,
          },
      .front_door_kind = front_door_kind,
      .credential_expires_at = absl::InfiniteFuture(),
      .carriage_kind = CredentialCarriageKind::kOpaqueLocalCompat,
      .binding_mode = LifecycleBindingMode::kBindingRecord,
      .constraint_claims =
          ConstraintClaims{
              .artifact_id = it->second.key.artifact_id,
              .local_only = true,
          },
  };
}

size_t HandleLeaseRegistry::size() const {
  absl::MutexLock lock(&mu_);
  return leases_.size();
}

} // namespace tensorcast::daemon
