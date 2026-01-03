// Copyright (c) 2025-2026, TensorCast Team.

#include "core/communicator/engine/staging_flow_controller.h"

#include <algorithm>

#include "absl/log/log.h"
#include "absl/time/clock.h"

#include "gsl/pointers"

#include "core/communicator/misc/ibv_wrap.h"

namespace tensorcast::communicator::engine {

namespace {

// Helper to deregister an MR safely while logging failures via PLOG.
void DeregisterMr(ibv_mr* mr) {
  if (mr == nullptr) {
    return;
  }
  const int rc = misc::wrap_ibv_dereg_mr(mr);
  if (rc != 0) {
    PLOG(ERROR) << "Failed to deregister MR";
  }
}

} // namespace

// ===== FlowCreditLedger::Lease =====

FlowCreditLedger::Lease::Lease(FlowCreditLedger* ledger, int granted_segments)
    : ledger_(ledger), granted_(granted_segments) {}

FlowCreditLedger::Lease::Lease(Lease&& other) noexcept {
  ledger_ = other.ledger_;
  granted_ = other.granted_;
  consumed_ = other.consumed_;
  other.ledger_ = nullptr;
  other.granted_ = 0;
  other.consumed_ = 0;
}

FlowCreditLedger::Lease& FlowCreditLedger::Lease::operator=(Lease&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  cleanup();
  ledger_ = other.ledger_;
  granted_ = other.granted_;
  consumed_ = other.consumed_;
  other.ledger_ = nullptr;
  other.granted_ = 0;
  other.consumed_ = 0;
  return *this;
}

FlowCreditLedger::Lease::~Lease() {
  cleanup();
}

void FlowCreditLedger::Lease::cleanup() {
  if (ledger_ == nullptr) {
    return;
  }
  const int unused = granted_ - consumed_;
  if (unused > 0) {
    ledger_->release(unused);
  }
  ledger_ = nullptr;
  granted_ = 0;
  consumed_ = 0;
}

void FlowCreditLedger::Lease::mark_consumed(int segments) {
  if (ledger_ == nullptr) {
    return;
  }
  consumed_ += segments;
  if (consumed_ > granted_) {
    LOG(WARNING) << "FlowCreditLease consumed more credit (" << consumed_ << ") than granted (" << granted_ << ")";
  }
}

void FlowCreditLedger::Lease::release_unused() {
  cleanup();
}

// ===== FlowCreditLedger =====

FlowCreditLedger::FlowCreditLedger(int total_credit) : total_credit_(total_credit) {
  if (total_credit_ <= 0) {
    LOG(FATAL) << "FlowCreditLedger requires positive credit";
  }
}

absl::StatusOr<FlowCreditLedger::Lease> FlowCreditLedger::acquire(int requested_segments) {
  if (requested_segments <= 0) {
    return absl::InvalidArgumentError("requested_segments must be > 0");
  }
  if (requested_segments > total_credit_) {
    return absl::InvalidArgumentError("requested_segments exceeds ledger capacity");
  }

  auto waiter = std::make_shared<Waiter>();
  waiter->requested = requested_segments;

  absl::MutexLock lk(&mu_);
  const bool immediate = waiters_.empty() && (total_credit_ - outstanding_ >= requested_segments);
  if (!immediate) {
    waiters_.push_back(waiter);
    while (!waiter->ready) {
      waiter->cv.Wait(&mu_);
    }
  }

  outstanding_ += requested_segments;
  wake_waiters_locked();
  return Lease(this, requested_segments);
}

absl::StatusOr<FlowCreditLedger::Lease> FlowCreditLedger::try_acquire(int requested_segments) {
  if (requested_segments <= 0) {
    return absl::InvalidArgumentError("requested_segments must be > 0");
  }
  if (requested_segments > total_credit_) {
    return absl::InvalidArgumentError("requested_segments exceeds ledger capacity");
  }

  absl::MutexLock lk(&mu_);
  const int available = total_credit_ - outstanding_;
  if (available <= 0) {
    return absl::UnavailableError("no staging credit available");
  }

  const int grant = std::min(requested_segments, available);
  outstanding_ += grant;
  // No waiters are involved in the non-blocking path, but wake any queued
  // waiters if the newly updated outstanding value permits progress.
  wake_waiters_locked();
  return Lease(this, grant);
}

void FlowCreditLedger::release(int segments) {
  if (segments <= 0) {
    return;
  }
  absl::MutexLock lk(&mu_);
  outstanding_ -= segments;
  if (outstanding_ < 0) {
    LOG(WARNING) << "FlowCreditLedger outstanding credit dropped below zero: " << outstanding_;
    outstanding_ = 0;
  }
  wake_waiters_locked();
}

int FlowCreditLedger::total_credit() const {
  return total_credit_;
}

int FlowCreditLedger::outstanding_credit() const {
  absl::MutexLock lk(&mu_);
  return outstanding_;
}

int FlowCreditLedger::available_credit() const {
  absl::MutexLock lk(&mu_);
  return total_credit_ - outstanding_;
}

void FlowCreditLedger::wake_waiters_locked() {
  // Wake waiters in FIFO order while sufficient credit exists.
  int available = total_credit_ - outstanding_;
  while (!waiters_.empty()) {
    const auto& waiter = waiters_.front();
    if (waiter->requested <= available) {
      waiter->ready = true;
      waiter->cv.Signal();
      available -= waiter->requested;
      waiters_.pop_front();
    } else {
      break;
    }
  }
}

// ===== StageLease =====

StageLease::State::State(
    std::shared_ptr<MemoryStager> stager,
    FlowCreditLedger* ledger,
    void* exposed_ptr,
    size_t bytes,
    ibv_mr* mr,
    bool deregister_mr,
    Metadata metadata,
    std::function<void()> extra_release_hook)
    : stager(std::move(stager)),
      ledger(ledger),
      exposed_ptr(exposed_ptr),
      bytes(bytes),
      mr(mr),
      deregister_mr(deregister_mr),
      metadata(std::move(metadata)),
      extra_release_hook(std::move(extra_release_hook)),
      acquired_at(absl::Now()) {}

void StageLease::State::do_release() {
  if (ledger != nullptr) {
    ledger->release(1);
  }
  if (mr != nullptr && deregister_mr) {
    DeregisterMr(mr);
  }
  if (stager != nullptr && exposed_ptr != nullptr) {
    auto status = stager->release_staged_buffer(gsl::not_null<void*>{exposed_ptr});
    if (!status.ok()) {
      LOG(WARNING) << "Failed to release staged buffer: " << status;
    }
  }
  if (extra_release_hook) {
    extra_release_hook();
  }
}

StageLease::StageLease(
    std::shared_ptr<MemoryStager> stager,
    FlowCreditLedger* ledger,
    void* exposed_ptr,
    size_t bytes,
    ibv_mr* mr,
    bool deregister_mr,
    Metadata metadata,
    std::function<void()> extra_release_hook)
    : state_(
          std::make_shared<State>(
              std::move(stager),
              ledger,
              exposed_ptr,
              bytes,
              mr,
              deregister_mr,
              std::move(metadata),
              std::move(extra_release_hook))) {}

bool StageLease::released() const {
  if (!state_) {
    return true;
  }
  absl::MutexLock lk(&state_->mu);
  return state_->released;
}

void StageLease::release() {
  if (!state_) {
    return;
  }
  bool should_release = false;
  {
    absl::MutexLock lk(&state_->mu);
    if (!state_->released) {
      state_->released = true;
      should_release = true;
    }
  }
  if (should_release) {
    state_->do_release();
  }
}

void* StageLease::exposed_ptr() const {
  return state_ ? state_->exposed_ptr : nullptr;
}

size_t StageLease::bytes() const {
  return state_ ? state_->bytes : 0;
}

ibv_mr* StageLease::mr() const {
  return state_ ? state_->mr : nullptr;
}

const StageLease::Metadata& StageLease::metadata() const {
  static const Metadata kEmpty;
  return state_ ? state_->metadata : kEmpty;
}

absl::Time StageLease::acquired_at() const {
  return state_ ? state_->acquired_at : absl::Time();
}

void StageLease::set_metadata(Metadata metadata) {
  if (!state_) {
    return;
  }
  absl::MutexLock lk(&state_->mu);
  state_->metadata = std::move(metadata);
}

// ===== StageLeaseRegistry =====

size_t StageLeaseKeyHash::operator()(const StageLeaseKey& key) const {
  size_t h = std::hash<std::string>{}(key.request_key);
  h ^= std::hash<uint32_t>{}(key.window_seq + 0x9e3779b9 + (h << 6) + (h >> 2));
  h ^= std::hash<uint32_t>{}(key.segment_idx + 0x9e3779b9 + (h << 6) + (h >> 2));
  return h;
}

void StageLeaseRegistry::put(const StageLeaseKey& key, StageLease lease, absl::Time now) {
  absl::MutexLock lk(&mu_);
  entries_[key] = Entry{.lease = std::move(lease), .inserted_at = now};
}

absl::StatusOr<StageLease> StageLeaseRegistry::take(const StageLeaseKey& key) {
  absl::MutexLock lk(&mu_);
  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return absl::NotFoundError("StageLease not found");
  }
  StageLease lease = std::move(it->second.lease);
  entries_.erase(it);
  return lease;
}

std::vector<StageLeaseRegistry::RegistryEntry> StageLeaseRegistry::snapshot_expired(absl::Duration ttl) const {
  const absl::Time cutoff = absl::Now() - ttl;
  std::vector<RegistryEntry> out;
  absl::MutexLock lk(&mu_);
  for (const auto& kv : entries_) {
    if (kv.second.inserted_at <= cutoff) {
      out.push_back(RegistryEntry{.key = kv.first, .lease = kv.second.lease, .inserted_at = kv.second.inserted_at});
    }
  }
  return out;
}

size_t StageLeaseRegistry::size() const {
  absl::MutexLock lk(&mu_);
  return entries_.size();
}

// ===== StagingWindow =====

namespace {

uint32_t CeilDivUint64(uint64_t num, uint64_t denom) {
  if (denom == 0) {
    return 0;
  }
  return static_cast<uint32_t>((num + denom - 1) / denom);
}

} // namespace

StagingWindow::StagingWindow(
    FlowCreditLedger& ledger,
    StageFn stage_fn,
    uint64_t total_bytes,
    uint64_t chunk_size,
    uint64_t initial_offset,
    uint32_t max_window_segments)
    : ledger_(ledger),
      stage_fn_(std::move(stage_fn)),
      chunk_size_(chunk_size),
      max_window_segments_(max_window_segments),
      remaining_bytes_(total_bytes),
      current_offset_(initial_offset) {
  if (chunk_size_ == 0) {
    LOG(FATAL) << "StagingWindow requires positive chunk size";
  }
}

absl::StatusOr<StagingWindow::Window> StagingWindow::stage_next() {
  if (remaining_bytes_ == 0) {
    return absl::OutOfRangeError("staging complete");
  }

  const uint32_t segments_remaining = CeilDivUint64(remaining_bytes_, chunk_size_);
  const uint32_t per_window_cap = max_window_segments_ == 0 ? segments_remaining : max_window_segments_;
  const uint32_t requested_segments = std::min(segments_remaining, per_window_cap);

  if (requested_segments == 0) {
    return absl::InternalError("staging requested zero segments");
  }

  auto lease_or = ledger_.try_acquire(static_cast<int>(requested_segments));
  if (!lease_or.ok()) {
    return lease_or.status();
  }
  FlowCreditLedger::Lease credit_lease = std::move(lease_or.value());
  if (credit_lease.granted_segments() <= 0) {
    return absl::UnavailableError("ledger granted no segments");
  }

  Window window;
  const uint32_t window_seq = next_window_seq_;
  window.window_seq = window_seq;
  window.granted_credit = 0;
  window.segments.reserve(static_cast<size_t>(credit_lease.granted_segments()));

  uint32_t segment_idx = 0;
  for (; segment_idx < static_cast<uint32_t>(credit_lease.granted_segments()) && remaining_bytes_ > 0; ++segment_idx) {
    const uint32_t bytes = static_cast<uint32_t>(std::min<uint64_t>(chunk_size_, remaining_bytes_));
    auto lease_status = stage_fn_(current_offset_, bytes, segment_idx);
    if (!lease_status.ok()) {
      const absl::Status status = lease_status.status();
      credit_lease.release_unused();
      if (absl::IsResourceExhausted(status) || absl::IsUnavailable(status)) {
        if (window.segments.empty()) {
          return status;
        }
        window.more_segments = true;
        next_window_seq_ = window_seq + 1;
        window.granted_credit = static_cast<int>(window.segments.size());
        return window;
      }
      for (auto& seg : window.segments) {
        seg.lease.release();
      }
      window.segments.clear();
      return status;
    }
    StageLease lease = std::move(lease_status.value());
    credit_lease.mark_consumed();

    window.segments.push_back(
        Segment{.lease = std::move(lease), .offset = current_offset_, .bytes = bytes, .segment_idx = segment_idx});

    current_offset_ += bytes;
    remaining_bytes_ -= bytes;
  }

  window.more_segments = remaining_bytes_ > 0;
  next_window_seq_ = window_seq + 1;
  window.granted_credit = static_cast<int>(window.segments.size());
  return window;
}

} // namespace tensorcast::communicator::engine
