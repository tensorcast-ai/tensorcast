
// Copyright (c) 2025-2026, TensorCast Team.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <format>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "core/common/system_capabilities.h"
#include "core/communicator/engine/channel.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/engine/gpu_vram_rdma_stager.h"
#include "core/communicator/engine/host_pinned_cpu_stager.h"
#include "core/communicator/engine/host_pinned_gpu_stager.h"
#include "core/communicator/engine/message.h"
#include "core/communicator/engine/mtcp_transfer_completion_tracker.h"
#include "core/communicator/engine/protocol.h"
#include "core/communicator/engine/staging_flow_controller.h"
#include "core/communicator/misc/utils.h"
#include "core/communicator/routing/types.h"
#include "core/communicator/transport/rdma_context.h"
#include "core/cuda/cuda_api.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::communicator::engine {

using base::CHANNEL_MTCP;
using base::CHANNEL_RDMA;
using base::COMMUNICATE_ENGINE_DEV_CPU;
using base::COMMUNICATE_ENGINE_DEV_GPU;
using misc::get_us;
using misc::INTERNAL_ERROR;
using misc::SUCCESS;
using tensorcast::store::StableLocalBackingKind;
using tensorcast::store::StableLocalBackingRef;
using transport::future_read_result_t;
using transport::net_dev_t;
using transport::PartitionTensor;
using transport::RdmaContext;
using transport::read_request_t;
using transport::tcp_transport_t;

struct RdmaReadPlanSourceSlice {
  uint32_t source_slice_index = 0;
  std::string tensor_key;
  uint64_t remote_offset = 0;
  uint64_t bytes = 0;
  std::shared_ptr<PartitionTensor> tensor;
  std::shared_ptr<MemoryStager> stager;
  net_dev_t dev;
  StagingWindow::StageFn stage_fn;
  uint64_t chunk_size = 0;
  bool zero_copy = false;
};

struct RdmaSourceStageProfile {
  uint64_t window_count = 0;
  uint64_t segment_count = 0;
  uint64_t stage_calls = 0;
  uint64_t staged_bytes = 0;
  uint64_t cpu_stage_calls = 0;
  uint64_t cpu_stage_bytes = 0;
  uint64_t stage_total_us = 0;
  uint64_t cpu_pool_allocate_us = 0;
  uint64_t cpu_lease_acquire_us = 0;
  uint64_t cpu_memcpy_us = 0;
  uint64_t mr_us = 0;
  uint64_t mr_cache_hits = 0;
  uint64_t mr_cache_misses = 0;
  uint64_t max_cpu_memcpy_us = 0;
  uint64_t max_mr_us = 0;
};

absl::Mutex& lazy_direct_source_mr_mu() {
  static auto* mu = new absl::Mutex();
  return *mu;
}

absl::Status ensure_tensor_registered_on_dev(const std::shared_ptr<PartitionTensor>& tensor, const net_dev_t& dev) {
  if (tensor == nullptr || dev == nullptr) {
    return absl::InvalidArgumentError("tensor and dev are required for lazy MR registration");
  }
  if (tensor->has_registered_mr(dev)) {
    return absl::OkStatus();
  }

  absl::MutexLock lock(&lazy_direct_source_mr_mu());
  if (tensor->has_registered_mr(dev)) {
    return absl::OkStatus();
  }
  if (tensor->get_dev_by_rail(dev->get_rail_id()) == nullptr) {
    tensor->add_dev(dev);
  }
  if (!tensor->is_registered(dev)) {
    const auto reg_status = dev->reg_async(tensor);
    if (reg_status != misc::SUCCESS) {
      return absl::InternalError(absl::StrCat("failed to lazily register source tensor MR on ", dev->get_name()));
    }
  }
  tensor->wait_mr_ready(dev);
  if (!tensor->has_registered_mr(dev)) {
    return absl::InternalError(
        absl::StrCat("lazy source tensor MR unavailable on ", dev->get_name(), " after registration"));
  }
  return absl::OkStatus();
}

struct RdmaReadSession {
  enum class Mode {
    kLegacyTensor = 0,
    kReadPlan = 1,
  };

  Mode mode = Mode::kLegacyTensor;
  ProtoReadRequest request;
  ProtoReadPlanRequestHeader plan_request;
  std::string tensor_key;
  std::string request_key;
  std::string transfer_id;
  std::shared_ptr<PartitionTensor> tensor;
  std::shared_ptr<MemoryStager> stager;
  net_dev_t dev;
  tcp_transport_t control_transport;
  std::shared_ptr<void> read_guard;
  std::unique_ptr<FlowCreditLedger> direct_ledger;
  std::unique_ptr<StagingWindow> window;
  bool zero_copy = false;
  bool direct_source_response = false;
  uint32_t read_plan_window_segment_limit = 0;
  std::shared_ptr<RdmaSourceStageProfile> source_stage_profile;
  std::chrono::steady_clock::time_point created_at = std::chrono::steady_clock::now();
  uint64_t first_stage_start_us = 0;
  uint64_t first_window_send_us = 0;
  uint64_t first_window_segment_count = 0;
  uint64_t first_window_stage_calls = 0;
  uint64_t first_window_staged_bytes = 0;
  uint64_t first_window_cpu_stage_bytes = 0;
  uint64_t first_window_stage_total_us = 0;
  uint64_t first_window_cpu_memcpy_us = 0;
  uint64_t first_window_mr_us = 0;
  uint64_t first_window_mr_cache_hits = 0;
  uint64_t first_window_mr_cache_misses = 0;
  std::vector<RdmaReadPlanSourceSlice> plan_source_slices;
  size_t next_source_slice = 0;
  uint64_t next_source_slice_offset = 0;
  uint32_t next_window_seq = 0;
};

struct Communicator::GpuChannelLease {
  explicit GpuChannelLease(Communicator* owner) : owner(owner) {}

  ~GpuChannelLease() {
    if (owner != nullptr) {
      owner->release_gpu_channel_slot();
    }
  }

  Communicator* owner;
};

struct Communicator::TensorReadLease {
  TensorReadLease(Communicator* owner, std::string key) : owner(owner), tensor_key(std::move(key)) {}

  ~TensorReadLease() {
    if (owner != nullptr) {
      owner->release_tensor_read_lease(tensor_key);
    }
  }

  Communicator* owner;
  std::string tensor_key;
};

struct Communicator::StableLocalBackingState : public std::enable_shared_from_this<StableLocalBackingState> {
  struct ChunkRegistration {
    uint64_t chunk_index = 0;
    uint64_t base_addr = 0;
    uint64_t bytes = 0;
    struct ibv_mr* mr = nullptr;
  };

  struct ChunkEntry {
    enum class State {
      kEmpty,
      kRegistering,
      kReady,
      kFailed,
    };

    absl::Mutex mu;
    absl::CondVar cv;
    State state ABSL_GUARDED_BY(mu) = State::kEmpty;
    ChunkRegistration registration ABSL_GUARDED_BY(mu);
    absl::Status last_error ABSL_GUARDED_BY(mu) = absl::OkStatus();
  };

  struct RailRegistration {
    int16_t rail_id = -1;
    std::string nic_name;
    net_dev_t dev;
    absl::Mutex mu;
    absl::flat_hash_map<uint64_t, std::shared_ptr<ChunkEntry>> chunks ABSL_GUARDED_BY(mu);
  };

  struct ChunkHandle {
    int16_t rail_id = -1;
    std::string nic_name;
    struct ibv_mr* mr = nullptr;
    uint64_t chunk_index = 0;
    bool cache_hit = false;
    bool waited_on_inflight = false;
    bool registered_now = false;
  };

  struct ResolvedChunk {
    uint64_t chunk_index = 0;
    uint64_t chunk_base_addr = 0;
    uint64_t chunk_bytes = 0;
    uint64_t requested_chunk_bytes = 0;
  };

  struct Use {
    explicit Use(std::shared_ptr<StableLocalBackingState> owner) : owner(std::move(owner)) {}

    ~Use() {
      if (owner == nullptr) {
        return;
      }
      absl::MutexLock lock(&owner->lifecycle_mu);
      CHECK(owner->inflight_uses > 0);
      --owner->inflight_uses;
      if (owner->retiring && owner->inflight_uses == 0) {
        owner->drained_cv.SignalAll();
      }
    }

    std::shared_ptr<StableLocalBackingState> owner;
  };

  ~StableLocalBackingState() {
    std::vector<std::shared_ptr<RailRegistration>> rail_states;
    {
      absl::MutexLock lock(&rails_mu);
      rail_states.reserve(rails.size());
      for (const auto& [rail_id, registration] : rails) {
        (void)rail_id;
        rail_states.push_back(registration);
      }
    }
    for (const auto& rail : rail_states) {
      if (rail == nullptr) {
        continue;
      }
      std::vector<std::pair<uint64_t, std::shared_ptr<ChunkEntry>>> entries;
      {
        absl::MutexLock lock(&rail->mu);
        entries.reserve(rail->chunks.size());
        for (const auto& [chunk_index, entry] : rail->chunks) {
          entries.emplace_back(chunk_index, entry);
        }
      }
      for (const auto& [chunk_index, entry] : entries) {
        if (entry == nullptr) {
          continue;
        }

        struct ibv_mr* mr = nullptr;
        {
          absl::MutexLock entry_lock(&entry->mu);
          if (entry->state == ChunkEntry::State::kReady) {
            mr = entry->registration.mr;
          }
        }

        if (mr == nullptr) {
          continue;
        }
        const int rc = misc::wrap_ibv_dereg_mr(mr);
        if (rc != misc::SUCCESS) {
          LOG(WARNING) << "stable_local_backing.dereg_failed"
                       << " backing_id=" << backing.backing_id << " rail_id=" << rail->rail_id
                       << " nic=" << rail->nic_name << " chunk_index=" << chunk_index;
        }
      }
    }
  }

  std::shared_ptr<void> acquire_use() {
    absl::MutexLock lock(&lifecycle_mu);
    if (retiring) {
      return nullptr;
    }
    ++inflight_uses;
    return std::make_shared<Use>(shared_from_this());
  }

  void begin_retire_and_wait() {
    absl::MutexLock lock(&lifecycle_mu);
    retiring = true;
    while (inflight_uses > 0) {
      drained_cv.Wait(&lifecycle_mu);
    }
  }

  absl::Status merge_activation_backing(const StableLocalBackingRef& candidate, std::shared_ptr<void> keepalive) {
    absl::MutexLock lock(&lifecycle_mu);
    if (candidate.kind != backing.kind || candidate.backing_id != backing.backing_id ||
        candidate.backing_base_addr != backing.backing_base_addr || candidate.backing_bytes != backing.backing_bytes ||
        candidate.dev_type != backing.dev_type || candidate.dev_id != backing.dev_id) {
      return absl::FailedPreconditionError("stable local backing id already exists with different metadata");
    }
    if (backing.slot_bytes == 0 && candidate.slot_bytes > 0) {
      backing.slot_bytes = candidate.slot_bytes;
    } else if (candidate.slot_bytes > 0 && backing.slot_bytes > 0 && backing.slot_bytes != candidate.slot_bytes) {
      return absl::FailedPreconditionError("stable local backing slot_bytes mismatch");
    }
    if (keepalive != nullptr) {
      activation_keepalive = std::move(keepalive);
    }
    return absl::OkStatus();
  }

  absl::StatusOr<uint64_t> compute_requested_chunk_bytes(uint64_t requested_slot_bytes, uint32_t requested_chunk_slots)
      const {
    if (requested_slot_bytes == 0 || requested_chunk_slots == 0) {
      return absl::InvalidArgumentError("stable local backing chunk registration requires slot geometry");
    }
    if (requested_slot_bytes > std::numeric_limits<uint64_t>::max() / requested_chunk_slots) {
      return absl::OutOfRangeError("stable local backing chunk geometry overflows");
    }
    return requested_slot_bytes * requested_chunk_slots;
  }

  absl::Status validate_geometry_locked(
      uint64_t requested_slot_bytes,
      uint32_t requested_chunk_slots,
      uint64_t requested_chunk_bytes) ABSL_EXCLUSIVE_LOCKS_REQUIRED(lifecycle_mu) {
    if (registration_slot_bytes == 0) {
      registration_slot_bytes = requested_slot_bytes;
      registration_chunk_slots = requested_chunk_slots;
      registration_chunk_bytes = requested_chunk_bytes;
      return absl::OkStatus();
    }
    if (registration_slot_bytes != requested_slot_bytes || registration_chunk_slots != requested_chunk_slots ||
        registration_chunk_bytes != requested_chunk_bytes) {
      return absl::FailedPreconditionError("stable local backing chunk geometry mismatch");
    }
    return absl::OkStatus();
  }

  absl::StatusOr<ResolvedChunk> resolve_chunk_for_region(
      uint64_t requested_slot_bytes,
      uint32_t requested_chunk_slots,
      uint64_t local_addr,
      uint64_t local_bytes) const {
    if (local_addr < backing.backing_base_addr) {
      return absl::InvalidArgumentError("stable local backing local_addr precedes backing base");
    }
    const auto requested_chunk_bytes_or = compute_requested_chunk_bytes(requested_slot_bytes, requested_chunk_slots);
    if (!requested_chunk_bytes_or.ok()) {
      return requested_chunk_bytes_or.status();
    }
    const uint64_t requested_chunk_bytes = *requested_chunk_bytes_or;
    const uint64_t region_offset = local_addr - backing.backing_base_addr;
    if (region_offset > std::numeric_limits<uint64_t>::max() - local_bytes ||
        region_offset + local_bytes > backing.backing_bytes) {
      return absl::InvalidArgumentError("stable local backing local region exceeds backing bounds");
    }
    if (region_offset % requested_slot_bytes != 0) {
      return absl::InvalidArgumentError("stable local backing local region is not slot-aligned");
    }
    if (local_bytes == 0 || local_bytes > requested_slot_bytes) {
      return absl::InvalidArgumentError("stable local backing local region size is incompatible with slot_bytes");
    }
    const uint64_t slot_index = region_offset / requested_slot_bytes;
    const uint64_t chunk_index = slot_index / requested_chunk_slots;
    if (chunk_index > std::numeric_limits<uint64_t>::max() / requested_chunk_bytes) {
      return absl::OutOfRangeError("stable local backing chunk offset overflows");
    }
    const uint64_t chunk_base_offset = chunk_index * requested_chunk_bytes;
    if (chunk_base_offset >= backing.backing_bytes) {
      return absl::OutOfRangeError("stable local backing chunk base exceeds backing bytes");
    }
    const uint64_t chunk_base_addr = backing.backing_base_addr + chunk_base_offset;
    const uint64_t chunk_bytes = std::min<uint64_t>(requested_chunk_bytes, backing.backing_bytes - chunk_base_offset);
    if (region_offset + local_bytes > chunk_base_offset + chunk_bytes) {
      return absl::InvalidArgumentError("stable local backing local region crosses chunk boundary");
    }
    return ResolvedChunk{
        .chunk_index = chunk_index,
        .chunk_base_addr = chunk_base_addr,
        .chunk_bytes = chunk_bytes,
        .requested_chunk_bytes = requested_chunk_bytes,
    };
  }

  absl::StatusOr<ResolvedChunk> resolve_chunk_by_index(
      uint64_t requested_slot_bytes,
      uint32_t requested_chunk_slots,
      uint64_t chunk_index) const {
    const auto requested_chunk_bytes_or = compute_requested_chunk_bytes(requested_slot_bytes, requested_chunk_slots);
    if (!requested_chunk_bytes_or.ok()) {
      return requested_chunk_bytes_or.status();
    }
    const uint64_t requested_chunk_bytes = *requested_chunk_bytes_or;
    if (backing.backing_bytes == 0) {
      return absl::InvalidArgumentError("stable local backing requires non-zero backing_bytes");
    }
    if (chunk_index > std::numeric_limits<uint64_t>::max() / requested_chunk_bytes) {
      return absl::OutOfRangeError("stable local backing chunk offset overflows");
    }
    const uint64_t chunk_base_offset = chunk_index * requested_chunk_bytes;
    if (chunk_base_offset >= backing.backing_bytes) {
      return absl::OutOfRangeError("stable local backing chunk base exceeds backing bytes");
    }
    return ResolvedChunk{
        .chunk_index = chunk_index,
        .chunk_base_addr = backing.backing_base_addr + chunk_base_offset,
        .chunk_bytes = std::min<uint64_t>(requested_chunk_bytes, backing.backing_bytes - chunk_base_offset),
        .requested_chunk_bytes = requested_chunk_bytes,
    };
  }

  absl::Status ensure_geometry(
      uint64_t requested_slot_bytes,
      uint32_t requested_chunk_slots,
      uint64_t requested_chunk_bytes) {
    absl::MutexLock lock(&lifecycle_mu);
    if (retiring) {
      return absl::FailedPreconditionError("stable local backing is retiring");
    }
    return validate_geometry_locked(requested_slot_bytes, requested_chunk_slots, requested_chunk_bytes);
  }

  std::shared_ptr<RailRegistration> get_or_create_rail(net_dev_t dev, int16_t rail_id) {
    absl::MutexLock lock(&rails_mu);
    auto it = rails.find(rail_id);
    if (it != rails.end()) {
      return it->second;
    }
    auto rail = std::make_shared<RailRegistration>();
    rail->rail_id = rail_id;
    rail->nic_name = dev->get_name();
    rail->dev = dev;
    rails.emplace(rail_id, rail);
    return rail;
  }

  std::shared_ptr<ChunkEntry> get_or_create_chunk_entry(
      const std::shared_ptr<RailRegistration>& rail,
      uint64_t chunk_index) {
    absl::MutexLock lock(&rail->mu);
    auto it = rail->chunks.find(chunk_index);
    if (it != rail->chunks.end()) {
      return it->second;
    }
    auto entry = std::make_shared<ChunkEntry>();
    rail->chunks.emplace(chunk_index, entry);
    return entry;
  }

  absl::StatusOr<ChunkHandle> ensure_resolved_chunk(
      net_dev_t dev,
      int16_t rail_id,
      uint64_t requested_slot_bytes,
      uint32_t requested_chunk_slots,
      const ResolvedChunk& resolved_chunk) {
    if (dev == nullptr) {
      return absl::InvalidArgumentError("stable local backing chunk registration requires a net_dev");
    }

    auto geometry_status =
        ensure_geometry(requested_slot_bytes, requested_chunk_slots, resolved_chunk.requested_chunk_bytes);
    if (!geometry_status.ok()) {
      return geometry_status;
    }
    auto rail = get_or_create_rail(dev, rail_id);
    auto entry = get_or_create_chunk_entry(rail, resolved_chunk.chunk_index);

    bool waited_on_inflight = false;
    while (true) {
      {
        absl::MutexLock entry_lock(&entry->mu);
        if (entry->state == ChunkEntry::State::kReady) {
          return ChunkHandle{
              .rail_id = rail->rail_id,
              .nic_name = rail->nic_name,
              .mr = entry->registration.mr,
              .chunk_index = resolved_chunk.chunk_index,
              .cache_hit = !waited_on_inflight,
              .waited_on_inflight = waited_on_inflight,
              .registered_now = false,
          };
        }
        if (entry->state == ChunkEntry::State::kRegistering) {
          waited_on_inflight = true;
          entry->cv.Wait(&entry->mu);
          continue;
        }
        entry->state = ChunkEntry::State::kRegistering;
        entry->last_error = absl::OkStatus();
      }

      constexpr int kAccess = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;
      struct ibv_mr* mr = nullptr;
      auto status = dev->reg_mr(
          &mr,
          reinterpret_cast<void*>(resolved_chunk.chunk_base_addr),
          static_cast<size_t>(resolved_chunk.chunk_bytes),
          kAccess);

      absl::Status register_status = absl::OkStatus();
      if (status != SUCCESS || mr == nullptr) {
        register_status = absl::InternalError("failed to register stable local backing chunk MR");
      }

      {
        absl::MutexLock entry_lock(&entry->mu);
        if (register_status.ok()) {
          entry->registration = ChunkRegistration{
              .chunk_index = resolved_chunk.chunk_index,
              .base_addr = resolved_chunk.chunk_base_addr,
              .bytes = resolved_chunk.chunk_bytes,
              .mr = mr,
          };
          entry->state = ChunkEntry::State::kReady;
        } else {
          if (mr != nullptr) {
            const int rc = misc::wrap_ibv_dereg_mr(mr);
            if (rc != misc::SUCCESS) {
              LOG(WARNING) << "stable_local_backing.chunk_reg_cleanup_failed"
                           << " backing_id=" << backing.backing_id << " rail_id=" << rail_id
                           << " nic=" << rail->nic_name << " chunk_index=" << resolved_chunk.chunk_index;
            }
          }
          entry->last_error = register_status;
          entry->state = ChunkEntry::State::kFailed;
        }
        entry->cv.SignalAll();
        if (!register_status.ok()) {
          return register_status;
        }
        return ChunkHandle{
            .rail_id = rail->rail_id,
            .nic_name = rail->nic_name,
            .mr = entry->registration.mr,
            .chunk_index = resolved_chunk.chunk_index,
            .cache_hit = false,
            .waited_on_inflight = waited_on_inflight,
            .registered_now = true,
        };
      }
    }
  }

  absl::StatusOr<ChunkHandle> ensure_chunk(
      net_dev_t dev,
      int16_t rail_id,
      uint64_t requested_slot_bytes,
      uint32_t requested_chunk_slots,
      uint64_t local_addr,
      uint64_t local_bytes) {
    auto resolved_chunk_or =
        resolve_chunk_for_region(requested_slot_bytes, requested_chunk_slots, local_addr, local_bytes);
    if (!resolved_chunk_or.ok()) {
      return resolved_chunk_or.status();
    }
    return ensure_resolved_chunk(dev, rail_id, requested_slot_bytes, requested_chunk_slots, *resolved_chunk_or);
  }

  bool try_mark_prewarm_requested(size_t rail_count, uint64_t chunk_count_per_rail) {
    absl::MutexLock lock(&lifecycle_mu);
    if (retiring || prewarm_requested) {
      return false;
    }
    prewarm_requested = true;
    prewarm_complete = rail_count == 0;
    prewarm_rail_count = rail_count;
    prewarm_chunk_count_per_rail = chunk_count_per_rail;
    prewarm_jobs_total = rail_count;
    prewarm_jobs_completed = 0;
    prewarm_jobs_failed = 0;
    if (prewarm_complete) {
      prewarm_cv.SignalAll();
    }
    return true;
  }

  void note_prewarm_job_finished(bool success) {
    absl::MutexLock lock(&lifecycle_mu);
    if (!prewarm_requested) {
      return;
    }
    if (success) {
      ++prewarm_jobs_completed;
    } else {
      ++prewarm_jobs_failed;
    }
    if (!prewarm_complete && prewarm_jobs_completed + prewarm_jobs_failed >= prewarm_jobs_total) {
      prewarm_complete = true;
      prewarm_cv.SignalAll();
    }
  }

  bool prewarm_requested_enabled() const {
    absl::MutexLock lock(&lifecycle_mu);
    return prewarm_requested;
  }

  bool prewarm_complete_for_test() const {
    absl::MutexLock lock(&lifecycle_mu);
    return prewarm_complete;
  }

  bool wait_for_prewarm_completion(absl::Duration timeout) const {
    absl::MutexLock lock(&lifecycle_mu);
    if (!prewarm_requested || prewarm_complete) {
      return true;
    }
    const absl::Time deadline = absl::Now() + timeout;
    while (!prewarm_complete) {
      if (absl::Now() >= deadline) {
        break;
      }
      prewarm_cv.WaitWithDeadline(&lifecycle_mu, deadline);
    }
    return prewarm_complete;
  }

  std::string rail_chunk_counts_debug_string() const {
    std::vector<std::shared_ptr<RailRegistration>> rail_states;
    {
      absl::MutexLock lock(&rails_mu);
      rail_states.reserve(rails.size());
      for (const auto& [rail_id, registration] : rails) {
        (void)rail_id;
        rail_states.push_back(registration);
      }
    }
    std::ostringstream oss;
    bool first = true;
    for (const auto& registration : rail_states) {
      if (registration == nullptr) {
        continue;
      }
      if (!first) {
        oss << ",";
      }
      first = false;
      oss << registration->rail_id << ":" << chunk_count_for_rail(registration->rail_id);
    }
    return oss.str();
  }

  size_t chunk_count_for_rail(int16_t rail_id) const {
    std::shared_ptr<RailRegistration> rail;
    {
      absl::MutexLock lock(&rails_mu);
      auto it = rails.find(rail_id);
      if (it == rails.end()) {
        return 0;
      }
      rail = it->second;
    }
    if (rail == nullptr) {
      return 0;
    }
    std::vector<std::shared_ptr<ChunkEntry>> entries;
    {
      absl::MutexLock lock(&rail->mu);
      entries.reserve(rail->chunks.size());
      for (const auto& [chunk_index, entry] : rail->chunks) {
        (void)chunk_index;
        entries.push_back(entry);
      }
    }
    size_t ready = 0;
    for (const auto& entry : entries) {
      if (entry == nullptr) {
        continue;
      }
      absl::MutexLock entry_lock(&entry->mu);
      if (entry->state == ChunkEntry::State::kReady) {
        ++ready;
      }
    }
    return ready;
  }

  size_t prewarm_rail_count_for_test() const {
    absl::MutexLock lock(&lifecycle_mu);
    return prewarm_rail_count;
  }

  uint64_t prewarm_chunk_count_per_rail_for_test() const {
    absl::MutexLock lock(&lifecycle_mu);
    return prewarm_chunk_count_per_rail;
  }

  std::pair<size_t, size_t> prewarm_job_counters_for_test() const {
    absl::MutexLock lock(&lifecycle_mu);
    return {prewarm_jobs_completed, prewarm_jobs_failed};
  }

  StableLocalBackingRef backing;
  std::shared_ptr<void> activation_keepalive;
  mutable absl::Mutex lifecycle_mu;
  mutable absl::Mutex rails_mu;
  absl::flat_hash_map<int16_t, std::shared_ptr<RailRegistration>> rails ABSL_GUARDED_BY(rails_mu);
  uint64_t registration_slot_bytes ABSL_GUARDED_BY(lifecycle_mu) = 0;
  uint32_t registration_chunk_slots ABSL_GUARDED_BY(lifecycle_mu) = 0;
  uint64_t registration_chunk_bytes ABSL_GUARDED_BY(lifecycle_mu) = 0;
  int inflight_uses ABSL_GUARDED_BY(lifecycle_mu) = 0;
  bool retiring ABSL_GUARDED_BY(lifecycle_mu) = false;
  bool prewarm_requested ABSL_GUARDED_BY(lifecycle_mu) = false;
  bool prewarm_complete ABSL_GUARDED_BY(lifecycle_mu) = false;
  size_t prewarm_rail_count ABSL_GUARDED_BY(lifecycle_mu) = 0;
  uint64_t prewarm_chunk_count_per_rail ABSL_GUARDED_BY(lifecycle_mu) = 0;
  size_t prewarm_jobs_total ABSL_GUARDED_BY(lifecycle_mu) = 0;
  size_t prewarm_jobs_completed ABSL_GUARDED_BY(lifecycle_mu) = 0;
  size_t prewarm_jobs_failed ABSL_GUARDED_BY(lifecycle_mu) = 0;
  absl::CondVar drained_cv;
  mutable absl::CondVar prewarm_cv;
};

struct Communicator::TransferProgressState {
  std::string transfer_id;
  std::string request_key;
  std::string peer;
  std::string side;
  std::string transport;
  uint64_t total_bytes = 0;
  absl::Time start_time = absl::Now();
  std::atomic<uint64_t> bytes_completed{0};
  std::atomic<uint64_t> last_logged_bytes{0};
  std::atomic<int64_t> next_log_ms{0};
  std::atomic<int64_t> last_log_ms{0};
  std::atomic<bool> finished{false};
};

namespace {

future_read_result_t make_failed_read_future(absl::Status status, std::string tensor_key = {}) {
  std::promise<transport::read_result_t> promise;
  auto future = promise.get_future();
  transport::read_result_t result;
  result.status = std::move(status);
  result.tensor_key = std::move(tensor_key);
  promise.set_value(std::move(result));
  return future;
}

bool add_overflows(uint64_t lhs, uint64_t rhs) {
  return lhs > std::numeric_limits<uint64_t>::max() - rhs;
}

absl::StatusOr<std::vector<transport::RdmaTransport::RdmaReadSeg>> BuildPreparedPlanRdmaSegments(
    const transport::PreparedReadPlan& prepared,
    const ProtoReadPlanResponseExHeader& header,
    const ProtoReadPlanResponseExSeg* segments) {
  std::vector<transport::RdmaTransport::RdmaReadSeg> rdma_segments;
  rdma_segments.reserve(header.num_segments);
  uint64_t total_bytes = 0;
  uint64_t local_addr_min = std::numeric_limits<uint64_t>::max();
  uint64_t local_addr_max_end = 0;
  uint64_t remote_addr_min = std::numeric_limits<uint64_t>::max();
  uint64_t remote_addr_max_end = 0;
  uint64_t mr_relative_min = std::numeric_limits<uint64_t>::max();
  uint64_t mr_relative_max_end = 0;
  uint64_t mr_length_min = std::numeric_limits<uint64_t>::max();
  uint64_t mr_length_max = 0;
  std::vector<uint32_t> local_lkeys;
  std::vector<uint64_t> mr_bases;
  for (uint32_t i = 0; i < header.num_segments; ++i) {
    const auto& seg = segments[i];
    if (seg.source_slice_index >= prepared.logical_plan.source_slices.size()) {
      return absl::InvalidArgumentError("read_plan response references invalid source slice index");
    }
    if (seg.source_slice_index >= prepared.placements_by_source_slice.size()) {
      return absl::InvalidArgumentError("prepared read_plan missing source slice placement");
    }
    const auto& placements = prepared.placements_by_source_slice[seg.source_slice_index];
    if (placements.empty()) {
      return absl::FailedPreconditionError("prepared read_plan source slice has no placements");
    }
    const auto& source_slice = prepared.logical_plan.source_slices[seg.source_slice_index];
    if (seg.source_slice_offset > source_slice.bytes || seg.bytes > source_slice.bytes - seg.source_slice_offset) {
      return absl::OutOfRangeError("read_plan response segment exceeds source slice bounds");
    }
    const uint64_t segment_begin = seg.source_slice_offset;
    const uint64_t segment_end = seg.source_slice_offset + seg.bytes;
    uint64_t covered_until = segment_begin;
    for (const auto& placement : placements) {
      if (placement.local_region_index >= prepared.local_regions.size()) {
        return absl::InvalidArgumentError("prepared read_plan placement references invalid local region");
      }
      if (add_overflows(placement.source_slice_offset, placement.bytes)) {
        return absl::FailedPreconditionError("prepared read_plan placement source range overflows");
      }
      const uint64_t placement_begin = placement.source_slice_offset;
      const uint64_t placement_end = placement.source_slice_offset + placement.bytes;
      if (placement_end <= segment_begin) {
        continue;
      }
      if (placement_begin >= segment_end) {
        break;
      }

      const uint64_t overlap_begin = std::max<uint64_t>(segment_begin, placement_begin);
      const uint64_t overlap_end = std::min<uint64_t>(segment_end, placement_end);
      if (overlap_begin >= overlap_end) {
        continue;
      }
      if (overlap_begin != covered_until) {
        return absl::FailedPreconditionError("read_plan response segment is not fully covered by prepared placements");
      }

      const auto& region = prepared.local_regions[placement.local_region_index];
      struct ibv_mr* mr = region.mr;
      if (mr == nullptr) {
        if (region.tensor == nullptr) {
          return absl::FailedPreconditionError("prepared read_plan local region missing registered tensor");
        }
        auto dev = region.tensor->get_dev_by_rail(region.rail_id);
        if (dev == nullptr) {
          dev = region.tensor->get_dev();
        }
        if (dev == nullptr) {
          return absl::FailedPreconditionError("prepared read_plan local region missing RDMA device");
        }
        region.tensor->wait_mr_ready(dev);
        mr = region.tensor->get_mr(dev);
      }
      if (mr == nullptr) {
        return absl::FailedPreconditionError("prepared read_plan local region MR not ready");
      }

      const uint64_t local_offset = overlap_begin - placement_begin;
      const uint64_t remote_offset = overlap_begin - segment_begin;
      if (add_overflows(placement.local_region_offset, local_offset) ||
          add_overflows(region.logical_region.addr, placement.local_region_offset + local_offset) ||
          add_overflows(seg.addr, remote_offset)) {
        return absl::FailedPreconditionError("prepared read_plan segment address computation overflow");
      }

      const uint32_t length = static_cast<uint32_t>(overlap_end - overlap_begin);
      const uint64_t local_addr = region.logical_region.addr + placement.local_region_offset + local_offset;
      const uint64_t remote_addr = seg.addr + remote_offset;
      if (add_overflows(local_addr, length) || add_overflows(remote_addr, length)) {
        return absl::FailedPreconditionError("prepared read_plan segment end address overflow");
      }
      const uint64_t local_end = local_addr + length;
      const uint64_t remote_end = remote_addr + length;
      const uint64_t mr_base = reinterpret_cast<uint64_t>(mr->addr);
      if (local_addr < mr_base) {
        return absl::FailedPreconditionError("prepared read_plan local addr precedes MR base");
      }
      const uint64_t mr_relative = local_addr - mr_base;
      if (add_overflows(mr_relative, length)) {
        return absl::FailedPreconditionError("prepared read_plan MR-relative address overflow");
      }
      const uint64_t mr_relative_end = mr_relative + length;

      total_bytes += length;
      local_addr_min = std::min(local_addr_min, local_addr);
      local_addr_max_end = std::max(local_addr_max_end, local_end);
      remote_addr_min = std::min(remote_addr_min, remote_addr);
      remote_addr_max_end = std::max(remote_addr_max_end, remote_end);
      mr_relative_min = std::min(mr_relative_min, mr_relative);
      mr_relative_max_end = std::max(mr_relative_max_end, mr_relative_end);
      mr_length_min = std::min<uint64_t>(mr_length_min, mr->length);
      mr_length_max = std::max<uint64_t>(mr_length_max, mr->length);
      local_lkeys.push_back(mr->lkey);
      mr_bases.push_back(mr_base);

      rdma_segments.push_back(
          transport::RdmaTransport::RdmaReadSeg{
              .local_addr = local_addr,
              .local_lkey = mr->lkey,
              .length = length,
              .remote_addr = remote_addr,
              .rkey = seg.rkey,
              .window_seq = header.window_seq,
              .segment_idx = i,
          });
      covered_until = overlap_end;
    }
    if (covered_until != segment_end) {
      return absl::FailedPreconditionError("read_plan response segment exceeds prepared placement coverage");
    }
  }
  if (!rdma_segments.empty()) {
    std::sort(local_lkeys.begin(), local_lkeys.end());
    local_lkeys.erase(std::unique(local_lkeys.begin(), local_lkeys.end()), local_lkeys.end());
    std::sort(mr_bases.begin(), mr_bases.end());
    mr_bases.erase(std::unique(mr_bases.begin(), mr_bases.end()), mr_bases.end());
    VLOG(2) << "communicator.read_plan_segments"
            << " request_id=" << header.request_id << " window_seq=" << header.window_seq
            << " local_registration_mode=" << prepared.local_registration_mode
            << " stable_backing_id=" << prepared.stable_backing_id << " response_segments=" << header.num_segments
            << " rdma_segments=" << rdma_segments.size() << " total_bytes=" << total_bytes
            << " local_span_bytes=" << (local_addr_max_end - local_addr_min)
            << " remote_span_bytes=" << (remote_addr_max_end - remote_addr_min)
            << " unique_lkeys=" << local_lkeys.size() << " unique_mr_bases=" << mr_bases.size()
            << " mr_length_min=" << mr_length_min << " mr_length_max=" << mr_length_max
            << " mr_relative_min=" << mr_relative_min << " mr_relative_max_end=" << mr_relative_max_end
            << " mr_relative_span_bytes=" << (mr_relative_max_end - mr_relative_min) << " rail_id=" << prepared.rail_id
            << " local_nic=" << prepared.local_nic;
  }
  return rdma_segments;
}

struct RdmaDriveResult {
  absl::Status status = absl::OkStatus();
  bool made_progress = false;
  bool completed = false;
};

// TODO: Is rdma really need this application layer windows control?
enum class DirectFallbackReason {
  kNone = 0,
  kNotGpu,
  kNeedsStaging,
  kMrUnavailable,
  kOutOfRange,
};

const char* DirectFallbackReasonToString(DirectFallbackReason reason) {
  switch (reason) {
    case DirectFallbackReason::kNone:
      return "none";
    case DirectFallbackReason::kNotGpu:
      return "non_gpu";
    case DirectFallbackReason::kNeedsStaging:
      return "requires_staging";
    case DirectFallbackReason::kMrUnavailable:
      return "mr_unavailable";
    case DirectFallbackReason::kOutOfRange:
      return "out_of_range";
  }
  return "unknown";
}

uint32_t ReadFailedReasonFromStatus(const absl::Status& status) {
  if (absl::IsNotFound(status)) {
    return TENSORCAST_READ_FAILED_NO_TENSOR;
  }
  if (absl::IsOutOfRange(status)) {
    return TENSORCAST_READ_FAILED_OVERFLOW;
  }
  if (absl::IsResourceExhausted(status) || absl::IsUnavailable(status)) {
    return TENSORCAST_READ_FAILED_RESOURCE_EXHAUSTED;
  }
  if (absl::IsFailedPrecondition(status)) {
    return TENSORCAST_READ_FAILED_DIRECT_RDMA_REQUIRED;
  }
  return TENSORCAST_READ_FAILED_MEM_MISMATCH;
}

v1::RdmaConfig::StagedRdmaBackend NormalizeStagedBackend(const v1::CommunicatorConfig& config) {
  auto backend = config.rdma().staging_backend();
  if (backend == v1::RdmaConfig::STAGED_RDMA_BACKEND_UNSPECIFIED) {
    return v1::RdmaConfig::STAGED_RDMA_BACKEND_HOST_PINNED;
  }
  return backend;
}

const char* StagedBackendToString(v1::RdmaConfig::StagedRdmaBackend backend) {
  switch (backend) {
    case v1::RdmaConfig::STAGED_RDMA_BACKEND_HOST_PINNED:
      return "host_pinned";
    case v1::RdmaConfig::STAGED_RDMA_BACKEND_GPU_VRAM:
      return "gpu_vram";
    case v1::RdmaConfig::STAGED_RDMA_BACKEND_UNSPECIFIED:
    case v1::RdmaConfig_StagedRdmaBackend_RdmaConfig_StagedRdmaBackend_INT_MIN_SENTINEL_DO_NOT_USE_:
    case v1::RdmaConfig_StagedRdmaBackend_RdmaConfig_StagedRdmaBackend_INT_MAX_SENTINEL_DO_NOT_USE_:
      break;
  }
  return "host_pinned";
}

void record_staged_backend_metrics(v1::RdmaConfig::StagedRdmaBackend backend, bool tensor_on_cpu, uint64_t bytes);
void record_mr_register_metrics(const char* location, const char* reason, bool success);

} // namespace

StagingWindow::StageFn MakeStageFunction(
    const std::shared_ptr<PartitionTensor>& tensor,
    FlowCreditLedger* ledger,
    const std::shared_ptr<MemoryStager>& stager,
    const net_dev_t& dev,
    MrCache* mr_cache,
    std::string tensor_key,
    std::string request_key,
    v1::RdmaConfig::StagedRdmaBackend staged_backend,
    bool use_direct,
    ibv_mr* direct_mr,
    std::shared_ptr<RdmaSourceStageProfile> source_stage_profile) {
  if (use_direct) {
    const uint64_t base_addr = tensor->get_uint64_addr();
    const uint64_t tensor_bytes = tensor->get_bytes();
    return [tensor, ledger, request_key = std::move(request_key), base_addr, tensor_bytes, direct_mr](
               uint64_t offset, uint32_t bytes, uint32_t /*segment_idx*/) -> absl::StatusOr<StageLease> {
      if (offset + bytes > tensor_bytes) {
        return absl::OutOfRangeError("direct RDMA stage exceeded tensor bounds");
      }
      void* device_ptr = reinterpret_cast<void*>(base_addr + offset);
      StageLease::Metadata metadata;
      metadata.transport = StageTransport::kRdma;
      metadata.request_key = request_key;
      metadata.offset = offset;
      metadata.bytes = bytes;
      metadata.zero_copy = true;
      return StageLease(
          /*stager=*/nullptr,
          ledger,
          device_ptr,
          bytes,
          direct_mr,
          /*deregister_mr=*/false,
          metadata);
    };
  }

  if (!stager) {
    return [tensor_key = std::move(tensor_key)](
               uint64_t /*offset*/, uint32_t /*bytes*/, uint32_t /*segment_idx*/) -> absl::StatusOr<StageLease> {
      return absl::FailedPreconditionError(absl::StrCat("no staging backend available for tensor=", tensor_key));
    };
  }

  auto fallback_stager = stager;
  return [fallback_stager,
          tensor,
          dev,
          ledger,
          mr_cache,
          request_key = std::move(request_key),
          staged_backend,
          source_stage_profile = std::move(source_stage_profile)](
             uint64_t offset, uint32_t bytes, uint32_t /*segment_idx*/) -> absl::StatusOr<StageLease> {
    constexpr int kAccess = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;
    auto to_duration_us = [](std::chrono::steady_clock::time_point start,
                             std::chrono::steady_clock::time_point end) -> uint64_t {
      return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    };
    const auto stage_started_at = std::chrono::steady_clock::now();
    auto staged_or = fallback_stager->stage(tensor, offset, bytes, MemoryStager::StageMode::kTry);
    const auto stage_finished_at = std::chrono::steady_clock::now();
    if (!staged_or.ok()) {
      return staged_or.status();
    }
    void* exposed_ptr = *staged_or;
    ibv_mr* staged_mr = nullptr;
    bool deregister_mr = false;

    gsl::not_null<void*> exposed_ptr_nn{exposed_ptr};
    std::optional<HostPinnedCpuStager::StageStats> cpu_stage_stats;
    if (auto* cpu_stager = dynamic_cast<HostPinnedCpuStager*>(fallback_stager.get()); cpu_stager != nullptr) {
      cpu_stage_stats = cpu_stager->stage_stats_for_ptr(exposed_ptr_nn);
    }
    const char* mr_location = StagedBackendToString(staged_backend);
    const auto mr_started_at = std::chrono::steady_clock::now();
    if (mr_cache) {
      const auto slab = NormalizeMrRegion(*fallback_stager, exposed_ptr_nn, bytes);
      gsl::not_null<void*> mr_base = slab.base;
      size_t mr_bytes = slab.bytes;

      auto mr_result = mr_cache->get_or_register(dev->get_pd(), mr_base, mr_bytes, kAccess);
      const auto mr_finished_at = std::chrono::steady_clock::now();
      staged_mr = mr_result.mr;
      if (staged_mr == nullptr) {
        record_mr_register_metrics(mr_location, "cache_register_failed", false);
        auto release_status = fallback_stager->release_staged_buffer(exposed_ptr_nn);
        if (!release_status.ok()) {
          LOG(WARNING) << "Failed to release staged buffer after MR cache failure: " << release_status;
        }
        return absl::InternalError("failed to register MR via cache");
      }
      if (source_stage_profile != nullptr) {
        source_stage_profile->stage_calls += 1;
        source_stage_profile->staged_bytes += bytes;
        source_stage_profile->stage_total_us += to_duration_us(stage_started_at, stage_finished_at);
        source_stage_profile->mr_us += to_duration_us(mr_started_at, mr_finished_at);
        if (mr_result.registered) {
          source_stage_profile->mr_cache_misses += 1;
        } else {
          source_stage_profile->mr_cache_hits += 1;
        }
        source_stage_profile->max_mr_us =
            std::max<uint64_t>(source_stage_profile->max_mr_us, to_duration_us(mr_started_at, mr_finished_at));
        if (cpu_stage_stats.has_value()) {
          source_stage_profile->cpu_stage_calls += 1;
          source_stage_profile->cpu_stage_bytes += cpu_stage_stats->requested_bytes;
          source_stage_profile->cpu_pool_allocate_us += cpu_stage_stats->pool_allocate_us;
          source_stage_profile->cpu_lease_acquire_us += cpu_stage_stats->lease_acquire_us;
          source_stage_profile->cpu_memcpy_us += cpu_stage_stats->memcpy_us;
          source_stage_profile->max_cpu_memcpy_us =
              std::max<uint64_t>(source_stage_profile->max_cpu_memcpy_us, cpu_stage_stats->memcpy_us);
        }
      }
      if (mr_result.registered) {
        record_mr_register_metrics(mr_location, nullptr, true);
      }
    } else {
      if (dev->reg_mr(&staged_mr, exposed_ptr, bytes, kAccess) != SUCCESS) {
        record_mr_register_metrics(mr_location, "direct_register_failed", false);
        auto release_status = fallback_stager->release_staged_buffer(exposed_ptr_nn);
        if (!release_status.ok()) {
          LOG(WARNING) << "Failed to release staged buffer after MR registration failure: " << release_status;
        }
        return absl::InternalError("failed to register staged MR");
      }
      const auto mr_finished_at = std::chrono::steady_clock::now();
      if (source_stage_profile != nullptr) {
        source_stage_profile->stage_calls += 1;
        source_stage_profile->staged_bytes += bytes;
        source_stage_profile->stage_total_us += to_duration_us(stage_started_at, stage_finished_at);
        source_stage_profile->mr_us += to_duration_us(mr_started_at, mr_finished_at);
        source_stage_profile->mr_cache_misses += 1;
        source_stage_profile->max_mr_us =
            std::max<uint64_t>(source_stage_profile->max_mr_us, to_duration_us(mr_started_at, mr_finished_at));
        if (cpu_stage_stats.has_value()) {
          source_stage_profile->cpu_stage_calls += 1;
          source_stage_profile->cpu_stage_bytes += cpu_stage_stats->requested_bytes;
          source_stage_profile->cpu_pool_allocate_us += cpu_stage_stats->pool_allocate_us;
          source_stage_profile->cpu_lease_acquire_us += cpu_stage_stats->lease_acquire_us;
          source_stage_profile->cpu_memcpy_us += cpu_stage_stats->memcpy_us;
          source_stage_profile->max_cpu_memcpy_us =
              std::max<uint64_t>(source_stage_profile->max_cpu_memcpy_us, cpu_stage_stats->memcpy_us);
        }
      }
      record_mr_register_metrics(mr_location, nullptr, true);
      deregister_mr = true;
    }

    record_staged_backend_metrics(staged_backend, tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_CPU, bytes);

    StageLease::Metadata metadata;
    metadata.transport = StageTransport::kRdma;
    metadata.request_key = request_key;
    metadata.offset = offset;
    metadata.bytes = bytes;
    metadata.zero_copy = false;

    return StageLease(fallback_stager, ledger, exposed_ptr, bytes, staged_mr, deregister_mr, metadata);
  };
}

namespace {

opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> g_comm_meter;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_direct_segments_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_direct_bytes_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_direct_fallback_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> g_rdma_direct_window_bytes_hist;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_staged_backend_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_staged_bytes_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_mr_register_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_mr_register_failures_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_pinned_rdma_prereg_failures_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> g_pinned_rdma_prereg_latency_ms_hist;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> g_pinned_rdma_prereg_bytes_gauge;
std::atomic<double> g_pinned_rdma_prereg_bytes_last{0.0};
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_vram_rdma_prereg_failures_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> g_vram_rdma_prereg_latency_ms_hist;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> g_vram_rdma_prereg_bytes_gauge;
std::atomic<double> g_vram_rdma_prereg_bytes_last{0.0};

absl::once_flag g_comm_meter_once;
absl::once_flag g_rdma_direct_window_metrics_once;
absl::once_flag g_rdma_direct_fallback_metric_once;
absl::once_flag g_rdma_staged_metrics_once;
absl::once_flag g_rdma_mr_register_metrics_once;
absl::once_flag g_pinned_rdma_prereg_metrics_once;
absl::once_flag g_vram_rdma_prereg_metrics_once;

inline void ensure_communicator_meter() {
  absl::call_once(g_comm_meter_once, []() {
    g_comm_meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.communicator", "1.0.0");
  });
}

inline void ensure_rdma_direct_window_metrics() {
  ensure_communicator_meter();
  absl::call_once(g_rdma_direct_window_metrics_once, []() {
    g_rdma_direct_segments_total = g_comm_meter->CreateDoubleCounter("tc_rdma_direct_segments_total");
    g_rdma_direct_bytes_total = g_comm_meter->CreateDoubleCounter("tc_rdma_direct_bytes_total");
    g_rdma_direct_window_bytes_hist = g_comm_meter->CreateDoubleHistogram("tc_rdma_direct_window_bytes");
  });
}

inline void ensure_rdma_direct_fallback_metric() {
  ensure_communicator_meter();
  absl::call_once(g_rdma_direct_fallback_metric_once, []() {
    g_rdma_direct_fallback_total = g_comm_meter->CreateDoubleCounter("tc_rdma_direct_fallback_total");
  });
}

inline void ensure_rdma_staged_metrics() {
  ensure_communicator_meter();
  absl::call_once(g_rdma_staged_metrics_once, []() {
    g_rdma_staged_backend_total = g_comm_meter->CreateDoubleCounter("tc_rdma_staged_backend_total");
    g_rdma_staged_bytes_total = g_comm_meter->CreateDoubleCounter("tc_rdma_staged_bytes_total");
  });
}

inline void ensure_rdma_mr_register_metrics() {
  ensure_communicator_meter();
  absl::call_once(g_rdma_mr_register_metrics_once, []() {
    g_rdma_mr_register_total = g_comm_meter->CreateDoubleCounter("tc_rdma_mr_register_total");
    g_rdma_mr_register_failures_total = g_comm_meter->CreateDoubleCounter("tc_rdma_mr_register_failures_total");
  });
}

void pinned_rdma_prereg_bytes_callback(opentelemetry::metrics::ObserverResult result, void* /*state*/) noexcept {
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  obs->Observe(g_pinned_rdma_prereg_bytes_last.load());
}

inline void ensure_pinned_rdma_prereg_metrics() {
  ensure_communicator_meter();
  absl::call_once(g_pinned_rdma_prereg_metrics_once, []() {
    g_pinned_rdma_prereg_failures_total = g_comm_meter->CreateDoubleCounter("tc_pinned_rdma_prereg_failures_total");
    g_pinned_rdma_prereg_latency_ms_hist = g_comm_meter->CreateDoubleHistogram("tc_pinned_rdma_prereg_latency_ms");
    g_pinned_rdma_prereg_bytes_gauge = g_comm_meter->CreateDoubleObservableGauge("tc_pinned_rdma_prereg_bytes");
    g_pinned_rdma_prereg_bytes_gauge->AddCallback(&pinned_rdma_prereg_bytes_callback, nullptr);
  });
}

void vram_rdma_prereg_bytes_callback(opentelemetry::metrics::ObserverResult result, void* /*state*/) noexcept {
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  obs->Observe(g_vram_rdma_prereg_bytes_last.load());
}

inline void ensure_vram_rdma_prereg_metrics() {
  ensure_communicator_meter();
  absl::call_once(g_vram_rdma_prereg_metrics_once, []() {
    g_vram_rdma_prereg_failures_total = g_comm_meter->CreateDoubleCounter("tc_vram_rdma_prereg_failures_total");
    g_vram_rdma_prereg_latency_ms_hist = g_comm_meter->CreateDoubleHistogram("tc_vram_rdma_prereg_latency_ms");
    g_vram_rdma_prereg_bytes_gauge = g_comm_meter->CreateDoubleObservableGauge("tc_vram_rdma_prereg_bytes");
    g_vram_rdma_prereg_bytes_gauge->AddCallback(&vram_rdma_prereg_bytes_callback, nullptr);
  });
}

void record_direct_window_metrics(int device_id, uint64_t segments, uint64_t bytes) {
  if (segments == 0 && bytes == 0) {
    return;
  }
  ensure_rdma_direct_window_metrics();
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("device_id", opentelemetry::common::AttributeValue(device_id));
  auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
  auto ctx = opentelemetry::context::Context{};
  if (segments > 0) {
    g_rdma_direct_segments_total->Add(static_cast<double>(segments), attr_view, ctx);
  }
  if (bytes > 0) {
    const double bytes_double = static_cast<double>(bytes);
    g_rdma_direct_bytes_total->Add(bytes_double, attr_view, ctx);
    g_rdma_direct_window_bytes_hist->Record(bytes_double, attr_view, ctx);
  }
}

void record_staged_backend_metrics(v1::RdmaConfig::StagedRdmaBackend backend, bool tensor_on_cpu, uint64_t bytes) {
  ensure_rdma_staged_metrics();
  if (!g_rdma_staged_backend_total || !g_rdma_staged_bytes_total) {
    return;
  }
  const char* backend_label = StagedBackendToString(backend);
  const char* mem_label = tensor_on_cpu ? "cpu" : "gpu";
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("backend", backend_label);
  attrs.emplace("mem", mem_label);
  auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
  g_rdma_staged_backend_total->Add(1.0, attr_view, opentelemetry::context::Context{});
  g_rdma_staged_bytes_total->Add(static_cast<double>(bytes), attr_view, opentelemetry::context::Context{});
}

void record_mr_register_metrics(const char* location, const char* reason, bool success) {
  ensure_rdma_mr_register_metrics();
  if (!g_rdma_mr_register_total || !g_rdma_mr_register_failures_total) {
    return;
  }
  if (success) {
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("method", "ibv_reg_mr");
    attrs.emplace("location", location);
    auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
    g_rdma_mr_register_total->Add(1.0, attr_view, opentelemetry::context::Context{});
    return;
  }
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("method", "ibv_reg_mr");
  attrs.emplace("reason", reason);
  auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
  g_rdma_mr_register_failures_total->Add(1.0, attr_view, opentelemetry::context::Context{});
}

void record_direct_fallback_metric(DirectFallbackReason reason) {
  if (reason == DirectFallbackReason::kNone) {
    return;
  }
  ensure_rdma_direct_fallback_metric();
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("reason", opentelemetry::common::AttributeValue(DirectFallbackReasonToString(reason)));
  auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
  g_rdma_direct_fallback_total->Add(1.0, attr_view, opentelemetry::context::Context{});
}

constexpr uint64_t kTransferProgressMinBytes = 64ULL * 1024 * 1024;
constexpr int64_t kTransferProgressLogIntervalMs = 1000;
constexpr int kTransferProgressBarWidth = 18;
constexpr double kBytesPerGiB = static_cast<double>(1ULL << 30);
constexpr absl::Duration kUnregisterTensorDrainTimeout = absl::Minutes(10);
constexpr absl::Duration kUnregisterTensorDrainPollInterval = absl::Seconds(1);
constexpr absl::Duration kUnregisterTensorDrainLogInterval = absl::Seconds(5);

std::string truncate_token(std::string_view token, size_t max_chars) {
  if (token.size() <= max_chars) {
    return std::string(token);
  }
  if (max_chars <= 3) {
    return std::string(token.substr(0, max_chars));
  }
  return std::string(token.substr(0, max_chars - 3)) + "...";
}

std::string build_progress_bar(uint64_t done, uint64_t total) {
  if (total == 0) {
    return std::string(kTransferProgressBarWidth, '#');
  }
  const double ratio = std::clamp(static_cast<double>(done) / static_cast<double>(total), 0.0, 1.0);
  const int filled = static_cast<int>(ratio * static_cast<double>(kTransferProgressBarWidth));
  std::string bar(static_cast<size_t>(filled), '#');
  bar.append(static_cast<size_t>(kTransferProgressBarWidth - filled), '-');
  return bar;
}

double bytes_to_gib(uint64_t bytes) {
  return static_cast<double>(bytes) / kBytesPerGiB;
}

uint64_t compute_direct_chunk_bytes(uint64_t total_bytes, uint64_t base_chunk, uint32_t base_window, int qp_count) {
  if (total_bytes == 0) {
    return std::max<uint64_t>(1, base_chunk);
  }
  constexpr uint64_t kMinDirectChunkBytes = 1ULL << 20;
  const uint32_t target_segments =
      std::max<uint32_t>(16, std::max<uint32_t>(1, base_window) * static_cast<uint32_t>(std::max(1, qp_count)));
  uint64_t desired_chunk = (total_bytes + target_segments - 1) / target_segments;
  desired_chunk = std::max<uint64_t>(std::min<uint64_t>(total_bytes, kMinDirectChunkBytes), desired_chunk);
  if (base_chunk > 0) {
    desired_chunk = std::min<uint64_t>(desired_chunk, base_chunk);
  }
  return std::max<uint64_t>(1, std::min<uint64_t>(desired_chunk, total_bytes));
}

uint32_t compute_direct_window_segments(uint64_t total_bytes, uint64_t chunk_size, uint32_t base_window, int qp_count) {
  if (chunk_size == 0) {
    return base_window;
  }
  const uint64_t total_segments = (total_bytes + chunk_size - 1) / chunk_size;
  if (total_segments == 0) {
    return std::max<uint32_t>(1, base_window);
  }
  const uint32_t scaled_window = std::max<uint32_t>(1, base_window) * static_cast<uint32_t>(std::max(1, qp_count));
  return static_cast<uint32_t>(std::min<uint64_t>(total_segments, scaled_window));
}

uint64_t compute_read_plan_direct_chunk_bytes(uint64_t source_bytes) {
  if (source_bytes == 0) {
    return 1;
  }
  return std::min<uint64_t>(source_bytes, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
}

uint32_t compute_read_plan_direct_window_segment_limit(uint64_t total_segments) {
  if (total_segments == 0) {
    return 1;
  }
  const uint64_t max_segments_by_payload =
      (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - sizeof(ProtoReadPlanResponseExHeader)) /
      sizeof(ProtoReadPlanResponseExSeg);
  const uint64_t max_segments =
      std::min<uint64_t>(max_segments_by_payload, static_cast<uint64_t>(std::numeric_limits<int>::max()));
  return static_cast<uint32_t>(std::min<uint64_t>(total_segments, max_segments));
}

void log_rdma_source_stage_summary(const RdmaReadSession& session, const absl::Status& status) {
  if (session.source_stage_profile == nullptr) {
    return;
  }
  const auto& profile = *session.source_stage_profile;
  if (profile.stage_calls == 0 && profile.window_count == 0) {
    return;
  }
  const char* mode = session.mode == RdmaReadSession::Mode::kReadPlan ? "read_plan" : "legacy_tensor";
  const double stage_total_ms = static_cast<double>(profile.stage_total_us) / 1000.0;
  const double alloc_ms = static_cast<double>(profile.cpu_pool_allocate_us) / 1000.0;
  const double lease_ms = static_cast<double>(profile.cpu_lease_acquire_us) / 1000.0;
  const double copy_ms = static_cast<double>(profile.cpu_memcpy_us) / 1000.0;
  const double mr_ms = static_cast<double>(profile.mr_us) / 1000.0;
  const double source_prep_ms = stage_total_ms + mr_ms;
  const double copy_share_of_stage_pct = profile.stage_total_us == 0
      ? 0.0
      : 100.0 * static_cast<double>(profile.cpu_memcpy_us) / static_cast<double>(profile.stage_total_us);
  const double copy_share_of_source_prep_pct = source_prep_ms <= 0.0 ? 0.0 : 100.0 * copy_ms / source_prep_ms;
  const double first_stage_start_ms = static_cast<double>(session.first_stage_start_us) / 1000.0;
  const double first_window_send_ms = static_cast<double>(session.first_window_send_us) / 1000.0;
  const double first_window_stage_total_ms = static_cast<double>(session.first_window_stage_total_us) / 1000.0;
  const double first_window_cpu_memcpy_ms = static_cast<double>(session.first_window_cpu_memcpy_us) / 1000.0;
  const double first_window_mr_ms = static_cast<double>(session.first_window_mr_us) / 1000.0;
  const double first_window_other_before_send_ms =
      std::max(0.0, first_window_send_ms - first_stage_start_ms - first_window_stage_total_ms - first_window_mr_ms);
  std::ostringstream log;
  log << "rdma_source_stage_summary"
      << " mode=" << mode << " request=" << session.request_key;
  if (session.mode == RdmaReadSession::Mode::kReadPlan) {
    log << " request_id=" << session.plan_request.request_id
        << " direct_source_response=" << (session.direct_source_response ? "yes" : "no")
        << " read_plan_window_segment_limit=" << session.read_plan_window_segment_limit;
  }
  log << " tensor_key=" << (session.tensor_key.empty() ? "<read_plan>" : session.tensor_key) << " status=" << status
      << " windows=" << profile.window_count << " segments=" << profile.segment_count
      << " source_slices=" << session.plan_source_slices.size() << " stage_calls=" << profile.stage_calls
      << " staged_bytes=" << profile.staged_bytes << " cpu_stage_calls=" << profile.cpu_stage_calls
      << " cpu_stage_bytes=" << profile.cpu_stage_bytes << " stage_total_ms=" << stage_total_ms
      << " cpu_pool_allocate_ms=" << alloc_ms << " cpu_lease_acquire_ms=" << lease_ms << " cpu_memcpy_ms=" << copy_ms
      << " mr_ms=" << mr_ms << " source_prep_ms=" << source_prep_ms
      << " copy_share_of_stage_pct=" << copy_share_of_stage_pct
      << " copy_share_of_source_prep_pct=" << copy_share_of_source_prep_pct
      << " mr_cache_hits=" << profile.mr_cache_hits << " mr_cache_misses=" << profile.mr_cache_misses
      << " max_cpu_memcpy_ms=" << (static_cast<double>(profile.max_cpu_memcpy_us) / 1000.0)
      << " max_mr_ms=" << (static_cast<double>(profile.max_mr_us) / 1000.0)
      << " first_stage_start_ms=" << first_stage_start_ms << " first_window_send_ms=" << first_window_send_ms
      << " first_window_segments=" << session.first_window_segment_count
      << " first_window_stage_calls=" << session.first_window_stage_calls
      << " first_window_staged_bytes=" << session.first_window_staged_bytes
      << " first_window_cpu_stage_bytes=" << session.first_window_cpu_stage_bytes
      << " first_window_stage_total_ms=" << first_window_stage_total_ms
      << " first_window_cpu_memcpy_ms=" << first_window_cpu_memcpy_ms << " first_window_mr_ms=" << first_window_mr_ms
      << " first_window_other_before_send_ms=" << first_window_other_before_send_ms
      << " first_window_mr_cache_hits=" << session.first_window_mr_cache_hits
      << " first_window_mr_cache_misses=" << session.first_window_mr_cache_misses;
  if (session.mode == RdmaReadSession::Mode::kReadPlan) {
    LOG(INFO) << log.str();
  } else {
    VLOG(2) << log.str();
  }
}

RdmaDriveResult DriveRdmaSession(Channel::FlowState& flow_state, RdmaReadSession& session) {
  RdmaDriveResult result;
  while (true) {
    auto window_or = session.window->stage_next();
    if (!window_or.ok()) {
      if (absl::IsOutOfRange(window_or.status())) {
        result.completed = true;
        result.status = absl::OkStatus();
        return result;
      }
      result.status = window_or.status();
      return result;
    }

    auto staged_window = std::move(window_or).value();
    if (staged_window.segments.empty()) {
      // No staged segments implies no credit; propagate as unavailable for the caller to retry later.
      result.status = absl::UnavailableError("staging produced no segments");
      return result;
    }

    uint64_t zero_copy_segments = 0;
    uint64_t zero_copy_bytes = 0;
    for (const auto& segment : staged_window.segments) {
      if (segment.lease.metadata().zero_copy) {
        ++zero_copy_segments;
        zero_copy_bytes += segment.bytes;
      }
    }
    const bool all_zero_copy = zero_copy_segments == staged_window.segments.size();
    if (zero_copy_segments > 0) {
      const int device_id = session.tensor ? session.tensor->get_device_id() : -1;
      record_direct_window_metrics(device_id, zero_copy_segments, zero_copy_bytes);
    }
    if (session.source_stage_profile != nullptr) {
      session.source_stage_profile->window_count += 1;
      session.source_stage_profile->segment_count += static_cast<uint64_t>(staged_window.segments.size());
    }

    result.made_progress = true;
    LOG(INFO) << "[staging_credit] request=" << session.request_key
              << " transport=rdma window=" << staged_window.window_seq << " granted=" << staged_window.granted_credit
              << " more=" << (staged_window.more_segments ? "yes" : "no")
              << " outstanding=" << flow_state.ledger.outstanding_credit();

    const uint32_t seg_count = static_cast<uint32_t>(staged_window.segments.size());
    auto rsp = std::make_shared<EngineMessage>(
        ENGINE_OP_READ_RESPONSE_EX,
        static_cast<uint32_t>(sizeof(ProtoReadResponseExHeader) + seg_count * sizeof(ProtoReadResponseExSeg)));

    auto* hdr = rsp->get_payload<ProtoReadResponseExHeader>();
    memcpy(hdr->tensor_key, session.request.tensor_key, kMaxTensorNameLen);
    hdr->transport_type = ENGINE_TRANSPORT_RDMA;
    hdr->staged = all_zero_copy ? 0 : 1;
    misc::STRNCPY(hdr->nic_name, session.dev ? session.dev->get_name().c_str() : "", kMaxDevName);
    hdr->request_offset = session.request.offset;
    hdr->request_id = session.request.request_id;
    hdr->zero_copy = session.zero_copy ? 1 : 0;
    hdr->rail_id = session.dev ? session.dev->get_rail_id() : 0;
    hdr->num_segments = seg_count;
    hdr->window_seq = staged_window.window_seq;
    hdr->credit_granted = static_cast<uint32_t>(staged_window.granted_credit);
    hdr->request_offset = session.request.offset;
    hdr->more_segments = staged_window.more_segments ? 1 : 0;

    std::vector<StageLeaseKey> inserted_keys;
    inserted_keys.reserve(seg_count);

    for (uint32_t i = 0; i < seg_count; ++i) {
      auto& segment = staged_window.segments[i];
      auto* seg_pl = reinterpret_cast<ProtoReadResponseExSeg*>(
          reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader) + i * sizeof(ProtoReadResponseExSeg));

      seg_pl->addr = reinterpret_cast<uint64_t>(segment.lease.exposed_ptr());
      seg_pl->offset = segment.offset;
      seg_pl->bytes = segment.bytes;
      seg_pl->rkey = segment.lease.mr() ? segment.lease.mr()->rkey : 0;

      StageLease::Metadata metadata = segment.lease.metadata();
      metadata.window_seq = staged_window.window_seq;
      metadata.segment_idx = segment.segment_idx;
      metadata.offset = segment.offset;
      metadata.bytes = segment.bytes;
      segment.lease.set_metadata(metadata);

      StageLeaseKey key{
          .request_key = metadata.request_key,
          .window_seq = metadata.window_seq,
          .segment_idx = metadata.segment_idx,
      };
      if (hdr->staged) {
        flow_state.registry.put(key, segment.lease);
        inserted_keys.push_back(key);
      }
    }

    misc::result_t send_res = session.control_transport->send(rsp);
    if (send_res != misc::SUCCESS) {
      LOG(ERROR) << "Failed to send RDMA READ_RESPONSE_EX window: res=" << send_res;
      if (hdr->staged) {
        for (const auto& key : inserted_keys) {
          auto lease_or = flow_state.registry.take(key);
          if (lease_or.ok()) {
            lease_or->release();
          }
        }
      } else {
        for (auto& segment : staged_window.segments) {
          segment.lease.release();
        }
      }
      result.status = absl::InternalError("failed to send RDMA window");
      return result;
    }

    if (!hdr->staged) {
      for (auto& segment : staged_window.segments) {
        segment.lease.release();
      }
    }

    if (!staged_window.more_segments) {
      result.completed = true;
      result.status = absl::OkStatus();
      return result;
    }
  }
}

RdmaDriveResult DriveRdmaPlanSession(Channel::FlowState& flow_state, RdmaReadSession& session) {
  struct WindowSegment {
    StageLease lease;
    uint32_t source_slice_index = 0;
    uint64_t source_slice_offset = 0;
    uint32_t bytes = 0;
    uint32_t segment_idx = 0;
  };

  RdmaDriveResult result;
  auto elapsed_us = [&](std::chrono::steady_clock::time_point end) -> uint64_t {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - session.created_at).count());
  };
  while (true) {
    if (session.next_source_slice >= session.plan_source_slices.size()) {
      result.completed = true;
      result.status = absl::OkStatus();
      return result;
    }

    FlowCreditLedger* ledger = session.direct_source_response ? session.direct_ledger.get() : &flow_state.ledger;
    CHECK(ledger != nullptr);
    const uint32_t requested_segments = session.direct_source_response
        ? std::max<uint32_t>(1, session.read_plan_window_segment_limit)
        : std::max<uint32_t>(1, flow_state.max_window_segments);
    auto credit_or = ledger->try_acquire(static_cast<int>(requested_segments));
    if (!credit_or.ok()) {
      result.status = credit_or.status();
      return result;
    }
    FlowCreditLedger::Lease credit_lease = std::move(credit_or.value());
    if (credit_lease.granted_segments() <= 0) {
      result.status = absl::UnavailableError("ledger granted no segments for read_plan");
      return result;
    }

    std::vector<WindowSegment> window_segments;
    window_segments.reserve(static_cast<size_t>(credit_lease.granted_segments()));
    bool all_zero_copy = true;
    bool early_window_exit = false;
    const uint32_t window_seq = session.next_window_seq++;
    const bool capture_first_window = session.first_window_send_us == 0;
    uint64_t first_window_stage_calls_before = 0;
    uint64_t first_window_staged_bytes_before = 0;
    uint64_t first_window_cpu_stage_bytes_before = 0;
    uint64_t first_window_stage_total_us_before = 0;
    uint64_t first_window_cpu_memcpy_us_before = 0;
    uint64_t first_window_mr_us_before = 0;
    uint64_t first_window_mr_cache_hits_before = 0;
    uint64_t first_window_mr_cache_misses_before = 0;
    if (capture_first_window && session.source_stage_profile != nullptr) {
      first_window_stage_calls_before = session.source_stage_profile->stage_calls;
      first_window_staged_bytes_before = session.source_stage_profile->staged_bytes;
      first_window_cpu_stage_bytes_before = session.source_stage_profile->cpu_stage_bytes;
      first_window_stage_total_us_before = session.source_stage_profile->stage_total_us;
      first_window_cpu_memcpy_us_before = session.source_stage_profile->cpu_memcpy_us;
      first_window_mr_us_before = session.source_stage_profile->mr_us;
      first_window_mr_cache_hits_before = session.source_stage_profile->mr_cache_hits;
      first_window_mr_cache_misses_before = session.source_stage_profile->mr_cache_misses;
    }

    for (uint32_t segment_idx = 0; segment_idx < static_cast<uint32_t>(credit_lease.granted_segments());
         ++segment_idx) {
      if (session.next_source_slice >= session.plan_source_slices.size()) {
        break;
      }
      auto& source_slice = session.plan_source_slices[session.next_source_slice];
      const uint64_t source_slice_offset = session.next_source_slice_offset;
      const uint64_t remaining_slice_bytes = source_slice.bytes - source_slice_offset;
      const uint32_t bytes = static_cast<uint32_t>(std::min<uint64_t>(source_slice.chunk_size, remaining_slice_bytes));
      if (session.first_stage_start_us == 0) {
        session.first_stage_start_us = elapsed_us(std::chrono::steady_clock::now());
      }
      auto lease_or = source_slice.stage_fn(source_slice.remote_offset + source_slice_offset, bytes, segment_idx);
      if (!lease_or.ok()) {
        const absl::Status status = lease_or.status();
        credit_lease.release_unused();
        if ((absl::IsResourceExhausted(status) || absl::IsUnavailable(status)) && !window_segments.empty()) {
          early_window_exit = true;
          break;
        }
        for (auto& segment : window_segments) {
          segment.lease.release();
        }
        result.status = status;
        return result;
      }

      StageLease lease = std::move(lease_or.value());
      all_zero_copy = all_zero_copy && lease.metadata().zero_copy;
      credit_lease.mark_consumed();
      window_segments.push_back(
          WindowSegment{
              .lease = std::move(lease),
              .source_slice_index = source_slice.source_slice_index,
              .source_slice_offset = source_slice_offset,
              .bytes = bytes,
              .segment_idx = segment_idx,
          });

      session.next_source_slice_offset += bytes;
      if (session.next_source_slice_offset == source_slice.bytes) {
        session.next_source_slice += 1;
        session.next_source_slice_offset = 0;
      }
    }

    if (window_segments.empty()) {
      result.status = absl::UnavailableError("read_plan staging produced no segments");
      return result;
    }

    const bool more_segments = early_window_exit || session.next_source_slice < session.plan_source_slices.size();
    result.made_progress = true;
    if (session.source_stage_profile != nullptr) {
      session.source_stage_profile->window_count += 1;
      session.source_stage_profile->segment_count += static_cast<uint64_t>(window_segments.size());
    }
    VLOG(2) << "rdma_plan_window"
            << " request=" << session.request_key
            << " direct_source_response=" << (session.direct_source_response ? "yes" : "no")
            << " window_seq=" << window_seq << " granted=" << credit_lease.granted_segments()
            << " segments=" << window_segments.size() << " more=" << (more_segments ? "yes" : "no")
            << " outstanding=" << ledger->outstanding_credit();
    const uint32_t seg_count = static_cast<uint32_t>(window_segments.size());
    auto rsp = std::make_shared<EngineMessage>(
        ENGINE_OP_READ_PLAN_RESPONSE_EX,
        static_cast<uint32_t>(sizeof(ProtoReadPlanResponseExHeader) + seg_count * sizeof(ProtoReadPlanResponseExSeg)));
    auto* hdr = rsp->get_payload<ProtoReadPlanResponseExHeader>();
    hdr->transport_type = ENGINE_TRANSPORT_RDMA;
    hdr->staged = all_zero_copy ? 0 : 1;
    misc::STRNCPY(hdr->nic_name, session.dev ? session.dev->get_name().c_str() : "", kMaxDevName);
    hdr->request_id = session.plan_request.request_id;
    hdr->num_segments = seg_count;
    hdr->window_seq = window_seq;
    hdr->credit_granted = static_cast<uint32_t>(window_segments.size());
    hdr->more_segments = more_segments ? 1 : 0;
    hdr->zero_copy = all_zero_copy ? 1 : 0;
    hdr->rail_id = session.dev ? session.dev->get_rail_id() : 0;

    std::vector<StageLeaseKey> inserted_keys;
    inserted_keys.reserve(seg_count);
    for (uint32_t i = 0; i < seg_count; ++i) {
      auto& segment = window_segments[i];
      auto* seg_pl = reinterpret_cast<ProtoReadPlanResponseExSeg*>(
          reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanResponseExHeader) +
          i * sizeof(ProtoReadPlanResponseExSeg));
      seg_pl->source_slice_index = segment.source_slice_index;
      seg_pl->source_slice_offset = segment.source_slice_offset;
      seg_pl->addr = reinterpret_cast<uint64_t>(segment.lease.exposed_ptr());
      seg_pl->bytes = segment.bytes;
      seg_pl->rkey = segment.lease.mr() ? segment.lease.mr()->rkey : 0;

      StageLease::Metadata metadata = segment.lease.metadata();
      metadata.window_seq = window_seq;
      metadata.segment_idx = segment.segment_idx;
      metadata.offset = segment.source_slice_offset;
      metadata.bytes = segment.bytes;
      segment.lease.set_metadata(metadata);

      StageLeaseKey key{
          .request_key = metadata.request_key,
          .window_seq = metadata.window_seq,
          .segment_idx = metadata.segment_idx,
      };
      if (hdr->staged) {
        flow_state.registry.put(key, segment.lease);
        inserted_keys.push_back(key);
      }
    }

    if (session.control_transport->send(rsp) != misc::SUCCESS) {
      if (hdr->staged) {
        for (const auto& key : inserted_keys) {
          auto lease_or = flow_state.registry.take(key);
          if (lease_or.ok()) {
            lease_or->release();
          }
        }
      } else {
        for (auto& segment : window_segments) {
          segment.lease.release();
        }
      }
      result.status = absl::InternalError("failed to send read_plan response window");
      return result;
    }

    if (capture_first_window) {
      session.first_window_send_us = elapsed_us(std::chrono::steady_clock::now());
      session.first_window_segment_count = static_cast<uint64_t>(window_segments.size());
      if (session.source_stage_profile != nullptr) {
        session.first_window_stage_calls = session.source_stage_profile->stage_calls - first_window_stage_calls_before;
        session.first_window_staged_bytes =
            session.source_stage_profile->staged_bytes - first_window_staged_bytes_before;
        session.first_window_cpu_stage_bytes =
            session.source_stage_profile->cpu_stage_bytes - first_window_cpu_stage_bytes_before;
        session.first_window_stage_total_us =
            session.source_stage_profile->stage_total_us - first_window_stage_total_us_before;
        session.first_window_cpu_memcpy_us =
            session.source_stage_profile->cpu_memcpy_us - first_window_cpu_memcpy_us_before;
        session.first_window_mr_us = session.source_stage_profile->mr_us - first_window_mr_us_before;
        session.first_window_mr_cache_hits =
            session.source_stage_profile->mr_cache_hits - first_window_mr_cache_hits_before;
        session.first_window_mr_cache_misses =
            session.source_stage_profile->mr_cache_misses - first_window_mr_cache_misses_before;
      }
    }

    if (!hdr->staged) {
      for (auto& segment : window_segments) {
        segment.lease.release();
      }
    }

    if (!more_segments) {
      result.completed = true;
      result.status = absl::OkStatus();
      return result;
    }
  }
}

absl::Duration compute_handshake_backoff(int failure_count) {
  if (failure_count <= 0) {
    return absl::Milliseconds(50);
  }
  const int capped = std::min(failure_count, 5);
  return absl::Milliseconds(50 * (1 << capped));
}

std::vector<Channel::PendingRdmaRead> drain_pending_reads_for_generation(
    const std::shared_ptr<Channel::RdmaEndpoint>& endpoint,
    uint64_t generation) {
  std::vector<Channel::PendingRdmaRead> drained;
  absl::MutexLock lock(&endpoint->mu);
  for (auto it = endpoint->pending_reads.begin(); it != endpoint->pending_reads.end();) {
    if (generation == 0 || it->generation == generation) {
      drained.push_back(std::move(*it));
      it = endpoint->pending_reads.erase(it);
    } else {
      ++it;
    }
  }
  return drained;
}

void log_handshake_transition(
    const std::string& local_dev,
    const std::string& peer_dev,
    Channel::HandshakeState from,
    Channel::HandshakeState to,
    uint64_t generation,
    size_t queue_depth) {
  auto state_to_string = [](Channel::HandshakeState state) {
    switch (state) {
      case Channel::HandshakeState::kIdle:
        return "idle";
      case Channel::HandshakeState::kConnectRequested:
        return "connecting";
      case Channel::HandshakeState::kReady:
        return "ready";
      case Channel::HandshakeState::kFailed:
        return "failed";
    }
    return "unknown";
  };

  LOG(INFO) << "[rdma_handshake] dev=" << local_dev << " peer=" << peer_dev << " state=" << state_to_string(from)
            << " -> " << state_to_string(to) << " gen=" << generation << " pending=" << queue_depth;
}

} // namespace

absl::StatusOr<std::shared_ptr<void>> Communicator::acquire_tensor_read_lease(const std::string& tensor_key) {
  if (tensor_key.empty()) {
    return absl::InvalidArgumentError("tensor key is empty");
  }
  absl::MutexLock lock(&tensor_read_mu_);
  auto& state_ptr = tensor_read_states_[tensor_key];
  if (state_ptr == nullptr) {
    state_ptr = std::make_unique<TensorReadState>();
  }
  if (state_ptr->retiring) {
    return absl::UnavailableError(std::format("tensor {} is retiring", tensor_key));
  }
  state_ptr->inflight += 1;
  auto lease = std::make_shared<TensorReadLease>(this, tensor_key);
  return std::static_pointer_cast<void>(lease);
}

void Communicator::release_tensor_read_lease(const std::string& tensor_key) {
  absl::MutexLock lock(&tensor_read_mu_);
  auto it = tensor_read_states_.find(tensor_key);
  if (it == tensor_read_states_.end() || it->second == nullptr) {
    LOG(WARNING) << "[Communicator] Tensor read lease release on unknown key=" << tensor_key;
    return;
  }
  auto& state = *(it->second);
  if (state.inflight <= 0) {
    LOG(WARNING) << "[Communicator] Tensor read lease underflow key=" << tensor_key;
    state.inflight = 0;
  } else {
    state.inflight -= 1;
  }
  if (state.inflight == 0) {
    state.drained_cv.SignalAll();
    if (!state.retiring) {
      tensor_read_states_.erase(it);
    }
  }
}

absl::Status Communicator::wait_for_tensor_reads_to_drain(const std::string& tensor_key, absl::Duration timeout) {
  const absl::Time start = absl::Now();
  const absl::Time deadline = start + timeout;

  absl::MutexLock lock(&tensor_read_mu_);
  auto& state_ptr = tensor_read_states_[tensor_key];
  if (state_ptr == nullptr) {
    state_ptr = std::make_unique<TensorReadState>();
  }
  state_ptr->retiring = true;

  absl::Time last_log = absl::InfinitePast();
  while (state_ptr->inflight > 0) {
    const absl::Time now = absl::Now();
    if (now >= deadline) {
      return absl::DeadlineExceededError(
          std::format(
              "tensor {} still has {} in-flight source reads after {} ms",
              tensor_key,
              state_ptr->inflight,
              absl::ToInt64Milliseconds(timeout)));
    }
    if (last_log == absl::InfinitePast() || now - last_log >= kUnregisterTensorDrainLogInterval) {
      LOG(INFO) << "[unregister_tensor] waiting for in-flight source reads key=" << tensor_key
                << " inflight=" << state_ptr->inflight << " elapsed_ms=" << absl::ToInt64Milliseconds(now - start)
                << " timeout_ms=" << absl::ToInt64Milliseconds(timeout);
      last_log = now;
    }
    const absl::Time wake_deadline = std::min(deadline, now + kUnregisterTensorDrainPollInterval);
    (void)state_ptr->drained_cv.WaitWithDeadline(&tensor_read_mu_, wake_deadline);
  }

  return absl::OkStatus();
}

std::shared_ptr<Communicator::TransferProgressState> Communicator::create_transfer_progress_state(
    std::string transfer_id,
    std::string request_key,
    std::string peer,
    std::string side,
    std::string transport,
    uint64_t total_bytes) {
  if (total_bytes < kTransferProgressMinBytes) {
    return nullptr;
  }

  auto state = std::make_shared<TransferProgressState>();
  state->transfer_id = std::move(transfer_id);
  state->request_key = std::move(request_key);
  state->peer = std::move(peer);
  state->side = std::move(side);
  state->transport = std::move(transport);
  state->total_bytes = total_bytes;
  state->start_time = absl::Now();
  const int64_t now_ms = absl::ToUnixMillis(state->start_time);
  state->next_log_ms.store(now_ms + kTransferProgressLogIntervalMs, std::memory_order_relaxed);
  state->last_log_ms.store(now_ms, std::memory_order_relaxed);

  LOG(INFO) << std::format(
      "[xfer_progress] side={} transport={} state=start peer={} request={} total_gib={:.3f}",
      state->side,
      state->transport,
      truncate_token(state->peer, 64),
      truncate_token(state->request_key, 80),
      bytes_to_gib(state->total_bytes));

  return state;
}

uint64_t Communicator::add_transfer_progress_bytes(
    const std::shared_ptr<TransferProgressState>& state,
    uint64_t bytes) {
  if (!state || bytes == 0 || state->finished.load(std::memory_order_relaxed)) {
    return state ? std::min<uint64_t>(state->bytes_completed.load(std::memory_order_relaxed), state->total_bytes) : 0;
  }

  const uint64_t done_raw = state->bytes_completed.fetch_add(bytes, std::memory_order_relaxed) + bytes;
  const uint64_t done = std::min<uint64_t>(done_raw, state->total_bytes);
  if (done >= state->total_bytes) {
    return done;
  }

  const absl::Time now = absl::Now();
  const int64_t now_ms = absl::ToUnixMillis(now);
  int64_t next_log_ms = state->next_log_ms.load(std::memory_order_relaxed);
  bool should_log = false;
  while (now_ms >= next_log_ms) {
    if (state->next_log_ms.compare_exchange_weak(
            next_log_ms, now_ms + kTransferProgressLogIntervalMs, std::memory_order_relaxed)) {
      should_log = true;
      break;
    }
  }
  if (!should_log) {
    return done;
  }

  const uint64_t prev_bytes = state->last_logged_bytes.exchange(done, std::memory_order_relaxed);
  const int64_t prev_ms = state->last_log_ms.exchange(now_ms, std::memory_order_relaxed);
  const double elapsed_sec = std::max(1e-6, absl::ToDoubleSeconds(now - state->start_time));
  const double avg_gibps = bytes_to_gib(done) / elapsed_sec;
  double inst_gibps = avg_gibps;
  if (prev_ms > 0 && now_ms > prev_ms && done >= prev_bytes) {
    const double delta_sec = static_cast<double>(now_ms - prev_ms) / 1000.0;
    if (delta_sec > 0.0) {
      inst_gibps = bytes_to_gib(done - prev_bytes) / delta_sec;
    }
  }

  const double progress_percent =
      state->total_bytes > 0 ? static_cast<double>(done) * 100.0 / static_cast<double>(state->total_bytes) : 100.0;
  LOG(INFO) << std::format(
      "[xfer_progress] side={} transport={} state=progress peer={} request={} bar=[{}] {:5.1f}% "
      "done_gib={:.3f}/{:.3f} rate_inst_gibps={:.3f} rate_avg_gibps={:.3f}",
      state->side,
      state->transport,
      truncate_token(state->peer, 64),
      truncate_token(state->request_key, 80),
      build_progress_bar(done, state->total_bytes),
      progress_percent,
      bytes_to_gib(done),
      bytes_to_gib(state->total_bytes),
      inst_gibps,
      avg_gibps);
  return done;
}

void Communicator::finish_transfer_progress(
    const std::shared_ptr<TransferProgressState>& state,
    const absl::Status& status) {
  if (!state) {
    return;
  }
  if (state->finished.exchange(true, std::memory_order_acq_rel)) {
    return;
  }

  const absl::Time now = absl::Now();
  const int64_t now_ms = absl::ToUnixMillis(now);
  const uint64_t done = std::min<uint64_t>(state->bytes_completed.load(std::memory_order_relaxed), state->total_bytes);
  const uint64_t prev_bytes = state->last_logged_bytes.exchange(done, std::memory_order_relaxed);
  const int64_t prev_ms = state->last_log_ms.exchange(now_ms, std::memory_order_relaxed);
  const double elapsed_sec = std::max(1e-6, absl::ToDoubleSeconds(now - state->start_time));
  const double avg_gibps = bytes_to_gib(done) / elapsed_sec;
  double inst_gibps = avg_gibps;
  if (prev_ms > 0 && now_ms > prev_ms && done >= prev_bytes) {
    const double delta_sec = static_cast<double>(now_ms - prev_ms) / 1000.0;
    if (delta_sec > 0.0) {
      inst_gibps = bytes_to_gib(done - prev_bytes) / delta_sec;
    }
  }
  const double progress_percent =
      state->total_bytes > 0 ? static_cast<double>(done) * 100.0 / static_cast<double>(state->total_bytes) : 100.0;
  const std::string phase = status.ok() ? "done" : "failed";
  const std::string status_text = status.ok() ? std::string() : truncate_token(status.message(), 120);
  const std::string line = std::format(
      "[xfer_progress] side={} transport={} state={} peer={} request={} bar=[{}] {:5.1f}% "
      "done_gib={:.3f}/{:.3f} rate_inst_gibps={:.3f} rate_avg_gibps={:.3f}{}",
      state->side,
      state->transport,
      phase,
      truncate_token(state->peer, 64),
      truncate_token(state->request_key, 80),
      build_progress_bar(done, state->total_bytes),
      progress_percent,
      bytes_to_gib(done),
      bytes_to_gib(state->total_bytes),
      inst_gibps,
      avg_gibps,
      status.ok() ? "" : std::format(" error={}", status_text));

  if (status.ok()) {
    LOG(INFO) << line;
  } else {
    LOG(WARNING) << line;
  }
}

std::string Communicator::make_transfer_id(std::string_view request_key, std::string_view peer) {
  std::string id;
  id.reserve(request_key.size() + peer.size() + 1);
  id.append(request_key);
  id.push_back('@');
  id.append(peer);
  return id;
}

std::shared_ptr<Communicator::TransferProgressState> Communicator::register_source_transfer_progress(
    std::string request_key,
    std::string peer,
    std::string transport,
    uint64_t total_bytes,
    std::shared_ptr<void> read_guard) {
  const std::string transfer_id = make_transfer_id(request_key, peer);
  auto state = create_transfer_progress_state(
      transfer_id, std::move(request_key), std::move(peer), "source", std::move(transport), total_bytes);
  if (!state) {
    if (read_guard) {
      absl::MutexLock lock(&source_transfer_progress_mu_);
      source_transfer_read_guards_[transfer_id] = std::move(read_guard);
    }
    return nullptr;
  }

  std::shared_ptr<TransferProgressState> replaced;
  std::shared_ptr<void> replaced_guard;
  {
    absl::MutexLock lock(&source_transfer_progress_mu_);
    if (read_guard) {
      auto guard_it = source_transfer_read_guards_.find(transfer_id);
      if (guard_it != source_transfer_read_guards_.end()) {
        replaced_guard = std::move(guard_it->second);
        source_transfer_read_guards_.erase(guard_it);
      }
      source_transfer_read_guards_[transfer_id] = std::move(read_guard);
    }
    auto it = source_transfer_progress_.find(transfer_id);
    if (it != source_transfer_progress_.end()) {
      replaced = it->second;
      it->second = state;
    } else {
      source_transfer_progress_.emplace(transfer_id, state);
    }
  }
  if (replaced) {
    finish_transfer_progress(replaced, absl::AbortedError("replaced by newer source transfer"));
  }
  replaced_guard.reset();
  return state;
}

std::shared_ptr<Communicator::TransferProgressState> Communicator::lookup_source_transfer_progress(
    const std::string& transfer_id) const {
  absl::MutexLock lock(&source_transfer_progress_mu_);
  auto it = source_transfer_progress_.find(transfer_id);
  if (it == source_transfer_progress_.end()) {
    return nullptr;
  }
  return it->second;
}

void Communicator::finish_source_transfer_progress(const std::string& transfer_id, const absl::Status& status) {
  std::shared_ptr<TransferProgressState> state;
  std::shared_ptr<void> read_guard;
  {
    absl::MutexLock lock(&source_transfer_progress_mu_);
    auto guard_it = source_transfer_read_guards_.find(transfer_id);
    if (guard_it != source_transfer_read_guards_.end()) {
      read_guard = std::move(guard_it->second);
      source_transfer_read_guards_.erase(guard_it);
    }
    auto it = source_transfer_progress_.find(transfer_id);
    if (it != source_transfer_progress_.end()) {
      state = it->second;
      source_transfer_progress_.erase(it);
    }
  }
  if (state) {
    finish_transfer_progress(state, status);
  }
  read_guard.reset();
}

Communicator::Communicator(const v1::CommunicatorConfig& config, uint32_t channel_expire_sec)
    : Communicator(
          config,
          [&config]() {
            constexpr size_t kDefaultGpuSliceBytes = 16ULL * 1024 * 1024;
            constexpr size_t kDefaultCpuSliceBytes = 4ULL * 1024 * 1024;
            PinnedStagingPools pools;
            const size_t num_buffers = static_cast<size_t>(std::max(1, config.stager().buffers_per_flow()));
            int conn = config.transport().tcp_conn_count();
            if (conn <= 0) {
              conn = base::kMTcpConnCount;
            }
            const size_t conn_count = static_cast<size_t>(std::max(2, conn));
            const size_t required_gpu_slices = num_buffers + (num_buffers * conn_count);
            pools.gpu_pool = std::make_shared<common::memory::PinnedBufferPool>(
                required_gpu_slices * kDefaultGpuSliceBytes, kDefaultGpuSliceBytes);
            pools.cpu_pool = std::make_shared<common::memory::PinnedBufferPool>(
                num_buffers * kDefaultCpuSliceBytes, kDefaultCpuSliceBytes);
            pools.preregister_gpu = config.enable_rdma();
            pools.preregister_cpu = config.enable_rdma();
            return pools;
          }(),
          channel_expire_sec) {}

Communicator::Communicator(const v1::CommunicatorConfig& config, PinnedStagingPools pools, uint32_t channel_expire_sec)
    : stop_(false),
      inited_(false),
      server_context_(new transport::TcpContext()),
      client_context_(new transport::TcpContext()),
      enable_rdma_(config.enable_rdma()),
      mtcp_conn_count_(config.transport().tcp_conn_count()),
      ack_ttl_ms_(config.rdma().ack_ttl_ms()),
      config_(config),
      channel_expire_(channel_expire_sec) {
  common::SystemCapabilities::instance().record_rdma_available(enable_rdma_);
  request_thread_ = std::thread([this]() { this->do_read_request_loop(); });
  gc_thread_ = std::thread([this]() { this->do_channel_gc_loop(); });
  // Apply typed config to TCP contexts
  server_context_->set_connect_timeout(config_.transport().connect_timeout_sec());
  client_context_->set_connect_timeout(config_.transport().connect_timeout_sec());
  bool so_reuseport_enabled = true;
  if (config_.transport().has_so_reuseport()) {
    so_reuseport_enabled = config_.transport().so_reuseport();
  }
  server_context_->set_so_reuseport(so_reuseport_enabled);
  client_context_->set_so_reuseport(so_reuseport_enabled);

  // No default residency provider required; staging policy no longer consults UMA bridges.

  // Staging resources are sized from the pinned-memory authority. The communicator config
  // controls fan-out and transport behavior only.
  if (!pools.gpu_pool) {
    LOG(FATAL) << "Communicator requires a non-null pinned gpu_pool";
  }
  gpu_memory_pool_ = std::move(pools.gpu_pool);
  cpu_memory_pool_ = pools.cpu_pool ? std::move(pools.cpu_pool) : gpu_memory_pool_;
  preregister_gpu_pool_ = pools.preregister_gpu;
  preregister_cpu_pool_ = pools.preregister_cpu;
  staging_wait_timeout_ = pools.staging_wait_timeout;
  if (staging_wait_timeout_ <= absl::ZeroDuration()) {
    LOG(WARNING) << "Communicator staging_wait_timeout is <= 0; defaulting to 30s";
    staging_wait_timeout_ = absl::Seconds(30);
  }
  const auto staging_backend = NormalizeStagedBackend(config_);
  if (staging_backend == v1::RdmaConfig::STAGED_RDMA_BACKEND_GPU_VRAM && !enable_rdma_) {
    LOG(FATAL) << "rdma.staging_backend=STAGED_RDMA_BACKEND_GPU_VRAM requires enable_rdma=true";
  }
  use_gpu_vram_staging_ = enable_rdma_ && staging_backend == v1::RdmaConfig::STAGED_RDMA_BACKEND_GPU_VRAM;
  if (use_gpu_vram_staging_) {
    const uint64_t pool_bytes = config_.rdma().vram_pool_bytes_per_gpu();
    const uint64_t slice_bytes = config_.rdma().vram_slice_bytes();
    if (pool_bytes == 0 || slice_bytes == 0) {
      LOG(FATAL) << "rdma.vram_pool_bytes_per_gpu and rdma.vram_slice_bytes must be > 0 when "
                    "STAGED_RDMA_BACKEND_GPU_VRAM staging is set";
    }
    int device_count = 0;
    auto count_status = cuda::get_device_count(&device_count);
    if (!count_status.ok()) {
      LOG(FATAL) << "Failed to query CUDA device count for GPU VRAM staging: " << count_status;
    }
    if (device_count <= 0) {
      LOG(FATAL) << "GPU VRAM staging requires at least one CUDA device";
    }
    for (int device_id = 0; device_id < device_count; ++device_id) {
      auto pool = std::make_shared<GpuVramStagingPool>(
          device_id, static_cast<size_t>(pool_bytes), static_cast<size_t>(slice_bytes));
      auto init_status = pool->initialize();
      if (!init_status.ok()) {
        LOG(FATAL) << "Failed to initialize GPU VRAM staging pool for device=" << device_id << ": " << init_status;
      }
      gpu_vram_pools_[device_id] = pool;
      gpu_vram_stagers_[device_id] = std::make_shared<GpuVramRdmaStager>(pool);
    }
  }

  if (config_.stager().buffers_per_flow() <= 0) {
    LOG(FATAL) << "stager.buffers_per_flow must be > 0";
  }
  const size_t gpu_chunk_size = gpu_memory_pool_->slice_bytes();
  const size_t cpu_chunk_size = cpu_memory_pool_->slice_bytes();
  if (gpu_chunk_size == 0 || cpu_chunk_size == 0) {
    LOG(FATAL) << "Pinned staging pool slice_bytes must be > 0";
  }

  const size_t num_buffers = static_cast<size_t>(config_.stager().buffers_per_flow());
  buffers_per_flow_ = static_cast<int>(num_buffers);
  direct_rdma_chunk_bytes_ = gpu_chunk_size;
  if (direct_rdma_chunk_bytes_ == 0) {
    LOG(FATAL) << "direct_rdma_chunk_bytes must be > 0";
  }
  const uint32_t configured_max_window = config_.stager().max_window_segments();
  if (configured_max_window == 0) {
    max_window_segments_ = static_cast<uint32_t>(num_buffers);
  } else {
    if (configured_max_window > num_buffers) {
      LOG(WARNING) << "max_window_segments=" << configured_max_window << " exceeds buffers_per_flow=" << num_buffers
                   << "; clamping to buffers_per_flow";
    }
    max_window_segments_ = std::min(configured_max_window, static_cast<uint32_t>(num_buffers));
  }
  int configured_conn = config_.transport().tcp_conn_count();
  if (configured_conn <= 0) {
    configured_conn = base::kMTcpConnCount;
  }
  configured_conn = std::max(2, configured_conn);
  mtcp_conn_count_ = configured_conn;
  stable_local_mr_reuse_chunk_slots_ = std::max<uint32_t>(1, config_.rdma().stable_local_mr_reuse_chunk_slots());
  stable_local_mr_reuse_async_prewarm_enabled_ = false;
  stable_local_mr_reuse_prewarm_workers_ = 0;
  if (config_.rdma().has_stable_local_mr_reuse_prewarm_workers()) {
    stable_local_mr_reuse_prewarm_workers_ = config_.rdma().stable_local_mr_reuse_prewarm_workers();
    stable_local_mr_reuse_async_prewarm_enabled_ = stable_local_mr_reuse_prewarm_workers_ > 0;
  } else if (config_.rdma().has_stable_local_mr_reuse_eager_prereg_all_rails()) {
    stable_local_mr_reuse_async_prewarm_enabled_ = config_.rdma().stable_local_mr_reuse_eager_prereg_all_rails();
    stable_local_mr_reuse_prewarm_workers_ = stable_local_mr_reuse_async_prewarm_enabled_ ? 1 : 0;
  }
  if (config_.rdma().has_stable_local_mr_reuse_prewarm_workers() &&
      config_.rdma().has_stable_local_mr_reuse_eager_prereg_all_rails()) {
    const bool alias_enabled = config_.rdma().stable_local_mr_reuse_eager_prereg_all_rails();
    const bool workers_enabled = config_.rdma().stable_local_mr_reuse_prewarm_workers() > 0;
    if (alias_enabled != workers_enabled) {
      LOG(WARNING) << "communicator.rdma.stable_local_mr_reuse_eager_prereg_all_rails is deprecated and ignored "
                   << "because communicator.rdma.stable_local_mr_reuse_prewarm_workers is set";
    }
  }
  const auto mtcp_conn_budget = static_cast<size_t>(configured_conn);
  const size_t recv_num_buffers = num_buffers * mtcp_conn_budget;
  const size_t computed_pool_buffers = num_buffers + recv_num_buffers;
  const size_t required_gpu_slices = computed_pool_buffers;
  const size_t capacity_gpu_slices = gpu_memory_pool_->capacity_slices();
  if (capacity_gpu_slices < required_gpu_slices) {
    LOG(FATAL) << "Pinned staging pool (comm_gpu) too small: capacity_slices=" << capacity_gpu_slices
               << " required_slices=" << required_gpu_slices << " slice_bytes=" << gpu_chunk_size
               << " (buffers_per_flow=" << num_buffers << " tcp_conn_count=" << config_.transport().tcp_conn_count()
               << ")";
  }
  gpu_memory_stager_ = std::make_shared<HostPinnedGpuStager>(gpu_chunk_size, num_buffers, gpu_memory_pool_);

  const size_t total_gpu_slices = capacity_gpu_slices;
  if (total_gpu_slices == 0) {
    LOG(FATAL) << "GPU pinned buffer pool initialized with zero slices";
  }
  const size_t stager_reserve = num_buffers;
  size_t available_gpu_slices = 0;
  if (total_gpu_slices > stager_reserve) {
    available_gpu_slices = total_gpu_slices - stager_reserve;
  }
  size_t computed_gpu_channel_limit = 0;
  if (buffers_per_flow_ > 0) {
    computed_gpu_channel_limit = available_gpu_slices / static_cast<size_t>(buffers_per_flow_);
  }

  const uint32_t configured_gpu_channels = config_.stager().expected_gpu_channels();
  if (configured_gpu_channels > 0) {
    if (configured_gpu_channels > computed_gpu_channel_limit) {
      LOG(FATAL) << "expected_gpu_channels=" << configured_gpu_channels
                 << " exceeds staging capacity. gpu_pool_slices=" << total_gpu_slices << " reserve=" << stager_reserve
                 << " buffers_per_flow=" << buffers_per_flow_ << " computed_limit=" << computed_gpu_channel_limit
                 << ". Increase pinned_memory.classes[name=comm_gpu].pool_bytes or reduce expected_gpu_channels.";
    }
    max_gpu_channels_ = static_cast<int>(configured_gpu_channels);
    enforce_gpu_channel_limit_ = max_gpu_channels_ > 0;
  } else {
    max_gpu_channels_ = static_cast<int>(computed_gpu_channel_limit);
    enforce_gpu_channel_limit_ = max_gpu_channels_ > 0;
  }

  if (enforce_gpu_channel_limit_) {
    LOG(INFO) << "[Communicator] GPU staging pool allows up to " << max_gpu_channels_
              << " concurrent MTCP transports before additional channels are rejected";
  } else {
    LOG(WARNING) << "[Communicator] GPU staging pool has insufficient headroom to compute MTCP channel limit;"
                 << " concurrent GPU reads may block waiting for staging buffers.";
  }

  const size_t required_cpu_slices = num_buffers;
  if (cpu_memory_pool_ && cpu_memory_pool_.get() != gpu_memory_pool_.get()) {
    const size_t available_cpu_slices = cpu_memory_pool_->capacity_slices();
    if (available_cpu_slices < required_cpu_slices) {
      LOG(FATAL) << "Pinned staging pool (comm_cpu) too small: capacity_slices=" << available_cpu_slices
                 << " required_slices=" << required_cpu_slices << " slice_bytes=" << cpu_chunk_size
                 << " (buffers_per_flow=" << num_buffers << ")";
    }
  }
  auto dram_pool = cpu_memory_pool_ ? cpu_memory_pool_ : gpu_memory_pool_;
  memory_stager_ = std::make_shared<HostPinnedCpuStager>(
      gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{dram_pool}, /*num_buffers_hint=*/num_buffers);
  if (auto ds = std::dynamic_pointer_cast<HostPinnedCpuStager>(memory_stager_)) {
    ds->set_lease_provider(HostPinnedCpuStager::make_noop_lease_provider());
  }

  if (enable_rdma_) {
    rdma_context_ = std::make_shared<RdmaContext>();
    meta_mr_cache_ = std::make_unique<MrCache>();
    // Apply typed RDMA QP tuning
    rdma_context_->set_qp_params(
        config_.rdma().traffic_class(), config_.rdma().qp_timeout(), config_.rdma().qp_retry());
    rdma_context_->set_outstanding_wr(config_.rdma().outstanding_wr());
    // Apply multi-QP configuration
    rdma_context_->set_multi_qp_config(config_.rdma().qp_count(), config_.rdma().bonding_balance());
    int access = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;

    if (config_.simple_numa().enable()) {
      for (const auto& node : config_.simple_numa().nodes()) {
        auto cpu_stager = std::make_shared<HostPinnedCpuStager>(
            gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{dram_pool},
            /*num_buffers_hint=*/num_buffers);
        if (auto ds = std::dynamic_pointer_cast<HostPinnedCpuStager>(cpu_stager)) {
          ds->set_lease_provider(HostPinnedCpuStager::make_noop_lease_provider());
        }
        auto gpu_mem_stager = std::make_shared<HostPinnedGpuStager>(gpu_chunk_size, num_buffers, gpu_memory_pool_);
        // Map GPU ids
        for (int gid : node.gpus()) {
          gpu_mem_stagers_[gid] = gpu_mem_stager;
        }
        // Map NIC names
        for (const auto& nic : node.nics()) {
          nic_cpu_stagers_[nic] = cpu_stager;
        }
      }
    }

    if (use_gpu_vram_staging_) {
      ensure_vram_rdma_prereg_metrics();
      const auto prereg_start = std::chrono::steady_clock::now();
      uint64_t prereg_bytes = 0;
      for (const auto& entry : gpu_vram_pools_) {
        prereg_bytes += entry.second->pool_bytes();
      }
      g_vram_rdma_prereg_bytes_last.store(static_cast<double>(prereg_bytes));

      uint64_t failures = 0;
      for (const auto& dev : rdma_context_->list_devs()) {
        for (const auto& entry : gpu_vram_pools_) {
          auto slab = entry.second->mr_slab();
          if (!slab.has_value()) {
            ++failures;
            LOG(WARNING) << "Failed to preregister VRAM MR for device=" << entry.first << ": missing slab";
            record_mr_register_metrics("gpu_vram", "preregister_failed", false);
            continue;
          }
          auto result = meta_mr_cache_->get_or_register(dev->get_pd(), slab->base, slab->bytes, access);
          if (result.mr == nullptr) {
            ++failures;
            LOG(WARNING) << "Failed to preregister VRAM MR for slab " << static_cast<void*>(slab->base.get())
                         << " bytes=" << slab->bytes << " on PD";
            record_mr_register_metrics("gpu_vram", "preregister_failed", false);
            continue;
          }
          if (result.registered) {
            record_mr_register_metrics("gpu_vram", nullptr, true);
          }
        }
      }
      const auto prereg_end = std::chrono::steady_clock::now();
      const double prereg_ms =
          std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(prereg_end - prereg_start).count();
      if (g_vram_rdma_prereg_latency_ms_hist) {
        std::map<std::string, opentelemetry::common::AttributeValue> attrs;
        auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
        g_vram_rdma_prereg_latency_ms_hist->Record(prereg_ms, attr_view, opentelemetry::context::Context{});
      }
      if (failures > 0 && g_vram_rdma_prereg_failures_total) {
        std::map<std::string, opentelemetry::common::AttributeValue> attrs;
        auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
        g_vram_rdma_prereg_failures_total->Add(
            static_cast<double>(failures), attr_view, opentelemetry::context::Context{});
      }
      if (failures > 0) {
        LOG(FATAL) << "GPU VRAM staging preregistration failed for " << failures << " slabs";
      }
    }

    // Preregister MRs for selected pools (slab-level).
    ensure_pinned_rdma_prereg_metrics();
    const auto prereg_start = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<common::memory::PinnedBufferPool>> pools;
    if (preregister_gpu_pool_) {
      pools.push_back(gpu_memory_pool_);
    }
    if (preregister_cpu_pool_ && cpu_memory_pool_ && cpu_memory_pool_.get() != gpu_memory_pool_.get()) {
      pools.push_back(cpu_memory_pool_);
    }
    uint64_t prereg_bytes = 0;
    for (const auto& pool : pools) {
      for (const auto& slab : pool->list_slabs()) {
        prereg_bytes += slab.bytes;
      }
    }
    g_pinned_rdma_prereg_bytes_last.store(static_cast<double>(prereg_bytes));

    uint64_t failures = 0;
    for (const auto& dev : rdma_context_->list_devs()) {
      for (auto& pool : pools) {
        for (const auto& slab : pool->list_slabs()) {
          auto result = meta_mr_cache_->get_or_register(dev->get_pd(), slab.base.get(), slab.bytes, access);
          if (result.mr == nullptr) {
            ++failures;
            LOG(WARNING) << "Failed to preregister MR for slab " << static_cast<void*>(slab.base.get())
                         << " bytes=" << slab.bytes << " on PD";
            record_mr_register_metrics("host_pinned", "preregister_failed", false);
          } else if (result.registered) {
            record_mr_register_metrics("host_pinned", nullptr, true);
          }
        }
      }
    }
    const auto prereg_end = std::chrono::steady_clock::now();
    const double prereg_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(prereg_end - prereg_start).count();
    if (g_pinned_rdma_prereg_latency_ms_hist) {
      std::map<std::string, opentelemetry::common::AttributeValue> attrs;
      auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
      g_pinned_rdma_prereg_latency_ms_hist->Record(prereg_ms, attr_view, opentelemetry::context::Context{});
    }
    if (failures > 0 && g_pinned_rdma_prereg_failures_total) {
      std::map<std::string, opentelemetry::common::AttributeValue> attrs;
      auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
      g_pinned_rdma_prereg_failures_total->Add(
          static_cast<double>(failures), attr_view, opentelemetry::context::Context{});
    }

    handshake_retry_thread_ = std::thread([this]() { this->handshake_retry_loop(); });
    handshake_retry_thread_started_ = true;

    for (uint32_t worker_index = 0; worker_index < stable_local_mr_reuse_prewarm_workers_; ++worker_index) {
      stable_local_prewarm_threads_.emplace_back([this]() { this->stable_local_prewarm_loop(); });
    }
  }

  mtcp_staging_thread_ = std::thread([this]() { this->mtcp_staging_loop(); });
}

Communicator::~Communicator() {
  store_.clear();
  stop_.store(true);
  request_queue_.stop();
  handshake_retry_stop_.store(true);
  handshake_retry_cv_.SignalAll();
  stable_local_prewarm_queue_.stop();
  mtcp_staging_queue_.notify();
  if (handshake_retry_thread_started_ && handshake_retry_thread_.joinable()) {
    handshake_retry_thread_.join();
  }
  for (auto& thread : stable_local_prewarm_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  if (mtcp_staging_thread_.joinable()) {
    mtcp_staging_thread_.join();
  }
  mtcp_staging_queue_.stop();
  if (request_thread_.joinable()) {
    request_thread_.join();
  }
  if (gc_thread_.joinable()) {
    gc_thread_.join();
  }

  for (auto& channel : channels_.pairs()) {
    channel.second->close();
  }

  pending_requests_.clear();
  {
    absl::MutexLock lock(&stable_local_backings_mu_);
    stable_local_backings_.clear();
  }
}

absl::Status Communicator::activate_stable_local_backing(
    const StableLocalBackingRef& backing,
    std::shared_ptr<void> keepalive) {
  if (!enable_rdma_ || rdma_context_ == nullptr) {
    return absl::OkStatus();
  }
  if (backing.kind != StableLocalBackingKind::kHostSharedRegion) {
    return absl::InvalidArgumentError("stable local backing activation only supports HOST_SHARED regions");
  }
  if (backing.backing_id.empty() || backing.backing_base_addr == 0 || backing.backing_bytes == 0) {
    return absl::InvalidArgumentError("stable local backing activation requires id, base address, and bytes");
  }
  if (backing.dev_type != COMMUNICATE_ENGINE_DEV_CPU) {
    return absl::InvalidArgumentError("stable local backing activation only supports CPU backing in the first cut");
  }

  std::shared_ptr<StableLocalBackingState> state;
  bool created = false;
  {
    absl::MutexLock lock(&stable_local_backings_mu_);
    auto it = stable_local_backings_.find(backing.backing_id);
    if (it != stable_local_backings_.end()) {
      if (it->second == nullptr) {
        return absl::FailedPreconditionError("stable local backing exists with null state");
      }
      state = it->second;
    } else {
      state = std::make_shared<StableLocalBackingState>();
      state->backing = backing;
      state->activation_keepalive = std::move(keepalive);
      stable_local_backings_.emplace(backing.backing_id, state);
      created = true;
    }
  }

  if (!created) {
    auto status = state->merge_activation_backing(backing, std::move(keepalive));
    if (!status.ok()) {
      return status;
    }
    VLOG(2) << "stable_local_backing.activate_merge"
            << " backing_id=" << backing.backing_id << " backing_bytes=" << backing.backing_bytes
            << " slot_bytes=" << backing.slot_bytes;
  } else {
    VLOG(2) << "stable_local_backing.activate"
            << " backing_id=" << backing.backing_id << " backing_base_addr=0x" << std::hex << backing.backing_base_addr
            << std::dec << " backing_bytes=" << backing.backing_bytes << " slot_bytes=" << backing.slot_bytes;
  }

  auto prewarm_status = schedule_stable_local_backing_prewarm(state);
  if (!prewarm_status.ok()) {
    if (created) {
      absl::MutexLock lock(&stable_local_backings_mu_);
      auto it = stable_local_backings_.find(backing.backing_id);
      if (it != stable_local_backings_.end() && it->second == state) {
        stable_local_backings_.erase(it);
      }
    }
    return prewarm_status;
  }
  return absl::OkStatus();
}

absl::Status Communicator::deactivate_stable_local_backing(std::string_view backing_id) {
  std::shared_ptr<StableLocalBackingState> state;
  {
    absl::MutexLock lock(&stable_local_backings_mu_);
    auto it = stable_local_backings_.find(std::string(backing_id));
    if (it == stable_local_backings_.end()) {
      return absl::OkStatus();
    }
    state = std::move(it->second);
    stable_local_backings_.erase(it);
  }

  VLOG(2) << "stable_local_backing.deactivate_begin"
          << " backing_id=" << backing_id;
  if (state != nullptr) {
    state->begin_retire_and_wait();
  }
  VLOG(2) << "stable_local_backing.deactivate_done" << " backing_id=" << backing_id;
  return absl::OkStatus();
}

bool Communicator::stable_local_backing_supported_for_test() const {
  return enable_rdma_ && rdma_context_ != nullptr;
}

bool Communicator::stable_local_backing_active_for_test(std::string_view backing_id) const {
  absl::MutexLock lock(&stable_local_backings_mu_);
  auto it = stable_local_backings_.find(std::string(backing_id));
  return it != stable_local_backings_.end() && it->second != nullptr;
}

size_t Communicator::stable_local_backing_chunk_count_for_test(std::string_view backing_id, int16_t rail_id) const {
  absl::MutexLock lock(&stable_local_backings_mu_);
  auto it = stable_local_backings_.find(std::string(backing_id));
  if (it == stable_local_backings_.end() || it->second == nullptr) {
    return 0;
  }
  return it->second->chunk_count_for_rail(rail_id);
}

bool Communicator::stable_local_backing_prewarm_complete_for_test(std::string_view backing_id) const {
  absl::MutexLock lock(&stable_local_backings_mu_);
  auto it = stable_local_backings_.find(std::string(backing_id));
  if (it == stable_local_backings_.end() || it->second == nullptr) {
    return false;
  }
  return it->second->prewarm_complete_for_test();
}

bool Communicator::wait_for_stable_local_backing_prewarm_for_test(std::string_view backing_id, absl::Duration timeout)
    const {
  std::shared_ptr<StableLocalBackingState> state;
  {
    absl::MutexLock lock(&stable_local_backings_mu_);
    auto it = stable_local_backings_.find(std::string(backing_id));
    if (it == stable_local_backings_.end() || it->second == nullptr) {
      return false;
    }
    state = it->second;
  }
  return state->wait_for_prewarm_completion(timeout);
}

std::vector<net_dev_t> Communicator::collect_primary_visible_rdma_rail_devs() const {
  std::vector<net_dev_t> rail_devs;
  if (!enable_rdma_ || rdma_context_ == nullptr) {
    return rail_devs;
  }
  absl::flat_hash_set<int16_t> seen_rails;
  rail_devs.reserve(rdma_context_->list_devs().size());
  for (const auto& dev : rdma_context_->list_devs()) {
    if (dev == nullptr) {
      continue;
    }
    if (!seen_rails.insert(dev->get_rail_id()).second) {
      continue;
    }
    auto primary = rdma_context_->get_dev_by_rail(dev->get_rail_id());
    if (primary != nullptr) {
      rail_devs.push_back(primary);
    }
  }
  return rail_devs;
}

absl::Status Communicator::schedule_stable_local_backing_prewarm(
    const std::shared_ptr<StableLocalBackingState>& state) {
  if (state == nullptr || !stable_local_mr_reuse_async_prewarm_enabled_ ||
      stable_local_mr_reuse_prewarm_workers_ == 0) {
    return absl::OkStatus();
  }
  if (state->backing.slot_bytes == 0) {
    LOG(WARNING) << "stable_local_backing.activate_prewarm_skipped"
                 << " backing_id=" << state->backing.backing_id << " reason=missing_slot_bytes";
    return absl::OkStatus();
  }
  const auto rail_devs = collect_primary_visible_rdma_rail_devs();
  if (rail_devs.empty()) {
    VLOG(2) << "stable_local_backing.activate_prewarm_skipped"
            << " backing_id=" << state->backing.backing_id << " reason=no_visible_rdma_rails";
    return absl::OkStatus();
  }
  const uint32_t chunk_slots = std::max<uint32_t>(1, stable_local_mr_reuse_chunk_slots_);
  const auto chunk_bytes_or = state->compute_requested_chunk_bytes(state->backing.slot_bytes, chunk_slots);
  if (!chunk_bytes_or.ok()) {
    return chunk_bytes_or.status();
  }
  const uint64_t chunk_bytes = *chunk_bytes_or;
  const uint64_t chunk_count = 1 + ((state->backing.backing_bytes - 1) / chunk_bytes);
  if (!state->try_mark_prewarm_requested(rail_devs.size(), chunk_count)) {
    return absl::OkStatus();
  }

  LOG(INFO) << "stable_local_backing.activate_prewarm_enqueued"
            << " backing_id=" << state->backing.backing_id << " backing_bytes=" << state->backing.backing_bytes
            << " slot_bytes=" << state->backing.slot_bytes << " chunk_slots=" << chunk_slots
            << " chunk_bytes=" << chunk_bytes << " chunk_count_per_rail=" << chunk_count
            << " visible_rail_count=" << rail_devs.size()
            << " prewarm_workers=" << stable_local_mr_reuse_prewarm_workers_;
  for (const auto& dev : rail_devs) {
    StableLocalPrewarmTask task;
    task.backing_state = state;
    task.backing_id = state->backing.backing_id;
    task.dev = dev;
    task.rail_id = dev->get_rail_id();
    task.slot_bytes = state->backing.slot_bytes;
    task.chunk_slots = chunk_slots;
    task.chunk_count = chunk_count;
    if (stable_local_prewarm_queue_.push(task) != misc::SUCCESS) {
      state->note_prewarm_job_finished(false);
      LOG(WARNING) << "stable_local_backing.activate_prewarm_enqueue_failed"
                   << " backing_id=" << state->backing.backing_id << " rail_id=" << task.rail_id;
    }
  }
  return absl::OkStatus();
}

void Communicator::stable_local_prewarm_loop() {
  while (!stop_.load()) {
    StableLocalPrewarmTask task = stable_local_prewarm_queue_.pop(true, 1000);
    if (task.backing_id.empty() && task.backing_state.expired()) {
      continue;
    }
    if (task.backing_state.expired()) {
      continue;
    }
    process_stable_local_prewarm_task(std::move(task));
  }

  while (true) {
    StableLocalPrewarmTask task = stable_local_prewarm_queue_.pop(false);
    if (task.backing_id.empty() && task.backing_state.expired()) {
      break;
    }
    if (task.backing_state.expired()) {
      continue;
    }
    process_stable_local_prewarm_task(std::move(task));
  }
}

void Communicator::process_stable_local_prewarm_task(StableLocalPrewarmTask task) {
  auto state = task.backing_state.lock();
  if (state == nullptr) {
    return;
  }

  auto use = state->acquire_use();
  if (use == nullptr) {
    state->note_prewarm_job_finished(false);
    return;
  }

  const absl::Time started_at = absl::Now();
  bool success = true;
  uint64_t cache_hits = 0;
  uint64_t lazy_regs = 0;
  for (uint64_t chunk_index = 0; chunk_index < task.chunk_count; ++chunk_index) {
    if (stop_.load(std::memory_order_relaxed)) {
      success = false;
      break;
    }
    auto resolved_chunk_or = state->resolve_chunk_by_index(task.slot_bytes, task.chunk_slots, chunk_index);
    if (!resolved_chunk_or.ok()) {
      success = false;
      LOG(WARNING) << "stable_local_backing.prewarm_chunk_resolve_failed"
                   << " backing_id=" << task.backing_id << " rail_id=" << task.rail_id << " chunk_index=" << chunk_index
                   << " status=" << resolved_chunk_or.status();
      break;
    }
    auto chunk_or =
        state->ensure_resolved_chunk(task.dev, task.rail_id, task.slot_bytes, task.chunk_slots, *resolved_chunk_or);
    if (!chunk_or.ok()) {
      success = false;
      LOG(WARNING) << "stable_local_backing.prewarm_chunk_register_failed"
                   << " backing_id=" << task.backing_id << " rail_id=" << task.rail_id << " chunk_index=" << chunk_index
                   << " status=" << chunk_or.status();
      break;
    }
    if (chunk_or->cache_hit) {
      ++cache_hits;
    }
    if (chunk_or->registered_now) {
      ++lazy_regs;
    }
  }

  state->note_prewarm_job_finished(success);
  VLOG(2) << "stable_local_backing.prewarm_job_done"
          << " backing_id=" << task.backing_id << " rail_id=" << task.rail_id << " chunk_count=" << task.chunk_count
          << " ready_chunks=" << state->chunk_count_for_rail(task.rail_id) << " cache_hits=" << cache_hits
          << " lazy_regs=" << lazy_regs << " success=" << success
          << " prewarm_ms=" << absl::ToDoubleMilliseconds(absl::Now() - started_at);
}

void Communicator::mtcp_staging_loop() {
  while (!stop_.load()) {
    MtcpReadTask task = mtcp_staging_queue_.pop(true, 1000);
    if (!task.channel) {
      continue;
    }
    process_mtcp_read_task(std::move(task));
  }

  while (true) {
    MtcpReadTask task = mtcp_staging_queue_.pop(false);
    if (!task.channel) {
      break;
    }
    fail_mtcp_read_task(task, absl::CancelledError("communicator shutting down"));
  }
}

void Communicator::fail_mtcp_read_task(const MtcpReadTask& task, absl::Status status) {
  if (status.ok()) {
    return;
  }

  const std::string tensor_key(reinterpret_cast<const char*>(task.request.tensor_key));
  const std::string request_key =
      transport::get_request_instance_key(tensor_key, task.request.offset, task.request.request_id);
  const std::string peer = task.control_transport ? task.control_transport->get_remote_url() : std::string();
  this->finish_source_transfer_progress(make_transfer_id(request_key, peer), status);

  if (task.control_transport) {
    auto fail_msg = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
    auto* payload = fail_msg->get_payload<ProtoReadFailed>();
    memcpy(payload->tensor_key, task.request.tensor_key, kMaxTensorNameLen);
    payload->offset = task.request.offset;
    payload->request_id = task.request.request_id;
    payload->reason = TENSORCAST_READ_FAILED_MEM_MISMATCH;
    misc::result_t send_res = task.control_transport->send(fail_msg);
    if (send_res != misc::SUCCESS) {
      LOG(WARNING) << "Failed to send READ_FAILED after staging failure: key=" << tensor_key << " res=" << send_res;
    }
  }

  LOG(WARNING) << "MTCP staging task failed for key=" << tensor_key << ": " << status;
}

absl::StatusOr<std::shared_ptr<void>> Communicator::acquire_gpu_channel_slot() {
  if (!enforce_gpu_channel_limit_ || max_gpu_channels_ <= 0) {
    return std::shared_ptr<void>{};
  }

  int current = active_gpu_channels_.load(std::memory_order_relaxed);
  while (true) {
    if (current >= max_gpu_channels_) {
      return absl::ResourceExhaustedError(
          absl::StrFormat(
              "GPU staging pool capacity exceeded: %d active MTCP channels, limit=%d (buffers_per_flow=%d)",
              current,
              max_gpu_channels_,
              buffers_per_flow_));
    }
    if (active_gpu_channels_.compare_exchange_weak(
            current, current + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      auto lease = std::shared_ptr<GpuChannelLease>(new GpuChannelLease(this));
      return std::static_pointer_cast<void>(lease);
    }
  }
}

void Communicator::release_gpu_channel_slot() {
  int previous = active_gpu_channels_.fetch_sub(1, std::memory_order_acq_rel);
  if (previous <= 0) {
    active_gpu_channels_.store(0, std::memory_order_relaxed);
    LOG(WARNING) << "[Communicator] GPU channel slot release underflow";
  }
}

void Communicator::process_mtcp_read_task(MtcpReadTask task) {
  const std::string tensor_key_for_progress(reinterpret_cast<const char*>(task.request.tensor_key));
  const std::string request_key_for_progress =
      transport::get_request_instance_key(tensor_key_for_progress, task.request.offset, task.request.request_id);
  const std::string peer_for_progress =
      task.control_transport ? task.control_transport->get_remote_url() : std::string();
  const std::string transfer_id = make_transfer_id(request_key_for_progress, peer_for_progress);

  if (!task.channel || !task.tensor) {
    finish_source_transfer_progress(transfer_id, absl::FailedPreconditionError("invalid MTCP staging task"));
    if (task.channel) {
      task.channel->mtcp_request_finished();
    }
    fail_mtcp_read_task(task, absl::FailedPreconditionError("invalid MTCP staging task"));
    return;
  }

  auto release_called = std::make_shared<std::atomic<bool>>(false);
  auto release_once = [channel = task.channel, release_called]() {
    if (!release_called->exchange(true, std::memory_order_acq_rel)) {
      channel->mtcp_request_finished();
    }
  };

  auto transfer_tracker = std::make_shared<MtcpTransferCompletionTracker>(
      [this, release_cb = release_once, transfer_id](const absl::Status& status) {
        release_cb();
        finish_source_transfer_progress(transfer_id, status);
      });

  auto flow_state = task.channel->flow_state();
  if (!flow_state) {
    const absl::Status status = absl::InternalError("channel missing flow state");
    transfer_tracker->fail_fast(status);
    fail_mtcp_read_task(task, status);
    return;
  }

  auto transport = task.channel->get_mtcp();
  if (transport == nullptr) {
    const absl::Status status = absl::InternalError("missing MTCP transport");
    transfer_tracker->fail_fast(status);
    fail_mtcp_read_task(task, status);
    return;
  }

  const ProtoReadRequest& request = task.request;
  const std::string tensor_key = tensor_key_for_progress;
  const uint64_t total_bytes = request.bytes;
  const uint64_t start_offset = request.offset;
  const std::string request_key = request_key_for_progress;

  std::shared_ptr<MemoryStager> stager = task.stager;
  if (!stager) {
    const bool needs_gpu_staging =
        task.tensor->needs_staging() || task.tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_GPU;
    if (needs_gpu_staging) {
      stager = get_gpu_mem_stager_for_id(task.tensor->get_device_id());
      if (!stager) {
        stager = gpu_memory_stager_;
      }
    } else {
      stager = memory_stager_;
    }
  }
  if (!stager) {
    const absl::Status status = absl::FailedPreconditionError("no staging backend available for MTCP tensor");
    transfer_tracker->fail_fast(status);
    fail_mtcp_read_task(task, status);
    return;
  }

  const uint64_t chunk_size = task.stage_chunk_bytes > 0
      ? task.stage_chunk_bytes
      : (stager->get_chunk_size() > 0 ? stager->get_chunk_size() : total_bytes);

  auto stage_fn = [&](uint64_t offset, uint32_t bytes, uint32_t segment_idx) -> absl::StatusOr<StageLease> {
    // Do not block inside stage(); bounded waiting and retry is handled by the
    // MTCP staging loop via backoff + retry_deadline.
    auto staged_or = stager->stage(task.tensor, offset, bytes, MemoryStager::StageMode::kTry);
    if (!staged_or.ok()) {
      return staged_or.status();
    }
    void* exposed_ptr = *staged_or;

    StageLease::Metadata metadata;
    metadata.transport = StageTransport::kMtcp;
    metadata.request_key = request_key;
    metadata.offset = offset;
    metadata.bytes = bytes;
    metadata.segment_idx = segment_idx;

    return StageLease(
        stager,
        &flow_state->ledger,
        exposed_ptr,
        bytes,
        /*mr=*/nullptr,
        /*deregister_mr=*/false,
        metadata);
  };

  StagingWindow window(
      flow_state->ledger, stage_fn, total_bytes, chunk_size, start_offset, flow_state->max_window_segments);

  absl::Time retry_deadline = absl::Now() + staging_wait_timeout_;
  absl::Duration backoff = absl::Milliseconds(1);
  constexpr absl::Duration kMaxBackoff = absl::Milliseconds(50);
  absl::Time last_warning = absl::InfinitePast();
  bool final_window_enqueued = false;

  while (true) {
    auto window_or = window.stage_next();
    if (!window_or.ok()) {
      if (absl::IsOutOfRange(window_or.status())) {
        break;
      }

      if (absl::IsUnavailable(window_or.status()) || absl::IsResourceExhausted(window_or.status())) {
        const absl::Time now = absl::Now();
        if (last_warning == absl::InfinitePast() || now - last_warning >= absl::Seconds(1)) {
          LOG(WARNING) << "[staging_credit] request=" << request_key
                       << " transport=mtcp waiting for staging credit outstanding="
                       << flow_state->ledger.outstanding_credit() << "/" << flow_state->ledger.total_credit();
          last_warning = now;
        }

        if (now >= retry_deadline) {
          LOG(ERROR) << "MTCP staging credit wait exceeded deadline for request=" << request_key;
          const absl::Status status = absl::ResourceExhaustedError("MTCP staging credit wait timed out");
          transfer_tracker->fail_fast(status);
          fail_mtcp_read_task(task, status);
          return;
        }

        absl::SleepFor(backoff);
        backoff = std::min(backoff * 2, kMaxBackoff);
        continue;
      }

      LOG(ERROR) << "Failed to stage MTCP window: " << window_or.status();
      transfer_tracker->fail_fast(window_or.status());
      fail_mtcp_read_task(task, window_or.status());
      return;
    }

    backoff = absl::Milliseconds(1);

    auto staged_window = std::move(window_or).value();
    transport::MTcpTransport::StageSendWindow send_window;
    send_window.request_key = request_key;
    send_window.window_seq = staged_window.window_seq;
    send_window.final_window = !staged_window.more_segments;
    send_window.total_bytes = total_bytes;
    send_window.stage_unit_bytes = chunk_size;
    send_window.segments.reserve(staged_window.segments.size());

    transfer_tracker->add_pending_segments(static_cast<int>(staged_window.segments.size()));

    VLOG(1) << "[staging_credit] request=" << request_key << " transport=mtcp window=" << staged_window.window_seq
            << " granted=" << staged_window.granted_credit << " more=" << (staged_window.more_segments ? "yes" : "no")
            << " outstanding=" << flow_state->ledger.outstanding_credit();

    for (auto& segment : staged_window.segments) {
      StageLease lease = std::move(segment.lease);
      StageLease::Metadata metadata = lease.metadata();
      metadata.window_seq = staged_window.window_seq;
      metadata.segment_idx = segment.segment_idx;
      metadata.offset = segment.offset;
      metadata.bytes = segment.bytes;
      lease.set_metadata(metadata);

      StageLeaseKey key{
          .request_key = metadata.request_key,
          .window_seq = metadata.window_seq,
          .segment_idx = metadata.segment_idx,
      };

      flow_state->registry.put(key, lease);

      transport::MTcpTransport::StageSendSegment send_segment;
      send_segment.data = lease.exposed_ptr();
      send_segment.bytes = metadata.bytes;
      send_segment.metadata = metadata;

      send_segment.on_complete = [this,
                                  flow_state_ref = flow_state,
                                  key,
                                  metadata,
                                  transfer_id,
                                  tracker = transfer_tracker,
                                  lease = std::move(lease)](misc::result_t status) mutable {
        if (flow_state_ref) {
          auto lease_or = flow_state_ref->registry.take(key);
          if (lease_or.ok()) {
            lease_or->release();
          } else {
            VLOG(1) << "[MTCP] StageLease missing during release: key=" << key.request_key
                    << " window=" << key.window_seq << " segment=" << key.segment_idx;
          }
        }
        lease.release();

        const bool ok = (status == misc::SUCCESS);
        if (ok) {
          auto progress = lookup_source_transfer_progress(transfer_id);
          if (progress) {
            add_transfer_progress_bytes(progress, metadata.bytes);
          }
        } else {
          LOG(WARNING) << "[MTCP] StageLease send failure request=" << metadata.request_key
                       << " window=" << metadata.window_seq << " segment=" << metadata.segment_idx
                       << " status=" << status;
        }

        tracker->mark_segment_finished(ok);
      };

      send_window.segments.push_back(std::move(send_segment));
    }

    const bool is_final_window = send_window.final_window;
    transport->enqueue_stage_window(std::move(send_window));

    if (is_final_window) {
      final_window_enqueued = true;
      transfer_tracker->mark_final_window_enqueued();
    }
  }

  if (!final_window_enqueued) {
    transfer_tracker->mark_final_window_enqueued();
  }
}

void Communicator::set_dram_lease_provider(const std::shared_ptr<HostPinnedCpuStager::LeaseProvider>& provider) {
  if (!memory_stager_)
    return;
  if (auto ds = std::dynamic_pointer_cast<HostPinnedCpuStager>(memory_stager_)) {
    ds->set_lease_provider(provider);
  }
  // Also propagate to NUMA CPU stagers if present
  for (auto& kv : nic_cpu_stagers_) {
    if (auto ds2 = std::dynamic_pointer_cast<HostPinnedCpuStager>(kv.second)) {
      ds2->set_lease_provider(provider);
    }
  }
}

future_read_result_t Communicator::read_tensor_local(
    const std::string& key,
    uint64_t addr,
    uint64_t bytes,
    int dev_type,
    int dev_id,
    uint64_t remote_offset) {
  std::promise<transport::read_result_t> promise;
  auto future = promise.get_future();
  transport::read_result_t result;
  result.tensor_key = key;

  auto local_tensor = store_.get_tensor(key);
  if (local_tensor == nullptr) {
    result.status = absl::NotFoundError(absl::StrCat("local tensor not found: key=", key));
    promise.set_value(std::move(result));
    return future;
  }

  const uint64_t tensor_bytes = local_tensor->get_bytes();
  if (remote_offset > tensor_bytes || bytes > tensor_bytes - remote_offset) {
    result.status = absl::OutOfRangeError(
        absl::StrCat(
            "local read range out of bounds: key=",
            key,
            " offset=",
            remote_offset,
            " bytes=",
            bytes,
            " tensor_bytes=",
            tensor_bytes));
    promise.set_value(std::move(result));
    return future;
  }
  if (bytes == 0) {
    result.status = absl::OkStatus();
    promise.set_value(std::move(result));
    return future;
  }
  if (addr == 0) {
    result.status = absl::InvalidArgumentError("local read destination address must be non-zero");
    promise.set_value(std::move(result));
    return future;
  }
  if (dev_type != COMMUNICATE_ENGINE_DEV_CPU && dev_type != COMMUNICATE_ENGINE_DEV_GPU) {
    result.status = absl::InvalidArgumentError(absl::StrCat("unsupported destination device type: ", dev_type));
    promise.set_value(std::move(result));
    return future;
  }

  const int src_dev_type = local_tensor->get_mem_type();
  const int src_dev_id = local_tensor->get_device_id();
  const uint64_t src_addr = local_tensor->get_uint64_addr();
  if (src_addr == 0) {
    result.status = absl::FailedPreconditionError(absl::StrCat("local tensor address is null: key=", key));
    promise.set_value(std::move(result));
    return future;
  }

  const void* src_ptr = reinterpret_cast<const void*>(src_addr + remote_offset);
  void* dst_ptr = reinterpret_cast<void*>(addr);

  auto wrap_cuda_status = [&key](const char* op, absl::Status status) -> absl::Status {
    if (status.ok()) {
      return status;
    }
    return absl::Status(
        status.code(), absl::StrCat("local tensor copy ", op, " failed for key=", key, ": ", status.message()));
  };

  absl::Status copy_status = absl::OkStatus();
  if (src_dev_type == COMMUNICATE_ENGINE_DEV_CPU && dev_type == COMMUNICATE_ENGINE_DEV_CPU) {
    std::memcpy(dst_ptr, src_ptr, bytes);
  } else if (src_dev_type == COMMUNICATE_ENGINE_DEV_CPU && dev_type == COMMUNICATE_ENGINE_DEV_GPU) {
    if (dev_id < 0) {
      copy_status = absl::InvalidArgumentError("destination GPU device id must be non-negative");
    } else {
      copy_status = wrap_cuda_status("set_device(dst_gpu)", cuda::set_device(dev_id));
      if (copy_status.ok()) {
        copy_status = wrap_cuda_status("cpu_to_gpu", cuda::memcpy(dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice));
      }
    }
  } else if (src_dev_type == COMMUNICATE_ENGINE_DEV_GPU && dev_type == COMMUNICATE_ENGINE_DEV_CPU) {
    if (src_dev_id < 0) {
      copy_status = absl::InvalidArgumentError("source tensor has invalid GPU device id");
    } else {
      copy_status = wrap_cuda_status("set_device(src_gpu)", cuda::set_device(src_dev_id));
      if (copy_status.ok()) {
        copy_status = wrap_cuda_status("gpu_to_cpu", cuda::memcpy(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost));
      }
    }
  } else if (src_dev_type == COMMUNICATE_ENGINE_DEV_GPU && dev_type == COMMUNICATE_ENGINE_DEV_GPU) {
    if (src_dev_id < 0 || dev_id < 0) {
      copy_status = absl::InvalidArgumentError("source/destination GPU device id must be non-negative");
    } else if (src_dev_id == dev_id) {
      copy_status = wrap_cuda_status("set_device(d2d)", cuda::set_device(dev_id));
      if (copy_status.ok()) {
        copy_status =
            wrap_cuda_status("gpu_to_gpu_same_device", cuda::memcpy(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice));
      }
    } else {
      int can_access = 0;
      copy_status =
          wrap_cuda_status("device_can_access_peer", cuda::device_can_access_peer(&can_access, dev_id, src_dev_id));
      if (copy_status.ok() && can_access == 0) {
        copy_status = absl::FailedPreconditionError(
            absl::StrCat("peer access unavailable between dst_device=", dev_id, " and src_device=", src_dev_id));
      }
      if (copy_status.ok()) {
        copy_status = wrap_cuda_status("enable_peer_access", cuda::enable_peer_access(dev_id, src_dev_id));
      }
      if (copy_status.ok()) {
        copy_status =
            wrap_cuda_status("memcpy_peer_async", cuda::memcpy_peer_async(dst_ptr, dev_id, src_ptr, src_dev_id, bytes));
      }
      if (copy_status.ok()) {
        copy_status = wrap_cuda_status("device_synchronize", cuda::device_synchronize());
      }
    }
  } else {
    copy_status = absl::InvalidArgumentError(
        absl::StrCat("unsupported local copy matrix: src_dev_type=", src_dev_type, " dst_dev_type=", dev_type));
  }

  result.status = copy_status;
  promise.set_value(std::move(result));
  return future;
}

absl::Status Communicator::init(const std::string& ip, uint16_t port, int conn_count) {
  inited_.store(true);
  if (server_context_->open(ip, port, [this](tcp_transport_t t) { return this->on_new_client(t); }) != SUCCESS) {
    return absl::InternalError("failed to open server " + ip + ":" + std::to_string(port));
  }
  if (conn_count > 0) {
    mtcp_conn_count_ = conn_count;
  }
  return absl::OkStatus();
}

std::vector<Communicator::VisibleRdmaDeviceInfo> Communicator::visible_rdma_devices() const {
  std::vector<VisibleRdmaDeviceInfo> devices;
  if (rdma_context_ == nullptr) {
    return devices;
  }
  devices.reserve(rdma_context_->list_devs().size());
  for (const auto& dev : rdma_context_->list_devs()) {
    if (dev == nullptr) {
      continue;
    }
    devices.push_back(
        VisibleRdmaDeviceInfo{
            .name = dev->get_name(),
            .rail_id = dev->get_rail_id(),
        });
  }
  return devices;
}

uint16_t Communicator::listening_port() const {
  if (!server_context_) {
    return 0;
  }
  return server_context_->listening_port();
}

future_read_result_t Communicator::read_plan(
    const routing::ReadPlan& plan,
    const std::string& dst_ip,
    uint16_t dst_port) {
  const std::string tensor_key =
      plan.source_slices.empty() ? std::string("read_plan") : plan.source_slices.front().tensor_key;
  if (!inited_.load()) {
    return make_failed_read_future(
        absl::FailedPreconditionError("failed to read plan through un-initiated engine"), tensor_key);
  }
  if (!enable_rdma_) {
    return make_failed_read_future(
        absl::UnimplementedError(
            std::format("communicator read_plan requires RDMA transport for peer {}:{}", dst_ip, dst_port)),
        tensor_key);
  }
  if (plan.source_slices.empty() || plan.slices.empty() || plan.local_regions.empty()) {
    return make_failed_read_future(
        absl::InvalidArgumentError("read_plan requires non-empty sources, slices, and regions"), tensor_key);
  }

  const routing::ReadRouteContext& route = plan.source_slices.front().route;
  if (route.protocol != routing::ConnectionProtocol::kRdma) {
    return make_failed_read_future(
        absl::UnimplementedError(
            std::format("communicator read_plan only supports routed RDMA today for peer {}:{}", dst_ip, dst_port)),
        tensor_key);
  }

  const uint64_t request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  auto prepared_or = prepare_read_plan(plan, request_id, tensor_key);
  if (!prepared_or.ok()) {
    return make_failed_read_future(prepared_or.status(), tensor_key);
  }
  auto prepared = std::move(*prepared_or);

  if (!plan.transport_request_id.empty()) {
    LOG(INFO) << "communicator.read_plan_issue"
              << " request_id=" << request_id << " transport_request_id=" << plan.transport_request_id
              << " tensor_key=" << tensor_key << " peer=" << dst_ip << ":" << dst_port
              << " source_slices=" << plan.source_slices.size() << " read_slices=" << plan.slices.size()
              << " local_regions=" << plan.local_regions.size() << " total_bytes=" << prepared->total_bytes
              << " local_registration_mode=" << prepared->local_registration_mode
              << " stable_backing_chunk_count=" << prepared->stable_backing_chunk_count
              << " stable_backing_chunk_cache_hits=" << prepared->stable_backing_chunk_cache_hits
              << " stable_backing_chunk_cache_misses=" << prepared->stable_backing_chunk_cache_misses
              << " stable_backing_chunk_waits=" << prepared->stable_backing_chunk_waits
              << " stable_backing_chunk_lazy_registrations=" << prepared->stable_backing_chunk_lazy_registrations
              << " stable_backing_prewarm_requested=" << prepared->stable_backing_prewarm_requested
              << " stable_backing_prewarm_complete=" << prepared->stable_backing_prewarm_complete
              << " rail_id=" << prepared->rail_id << " local_nic=" << prepared->local_nic;
  }

  auto req =
      std::make_shared<transport::ReadRequest>(tensor_key, dst_ip, dst_port, prepared, request_id, prepared->rail_id);
  req->status_.transport_is_rdma = true;
  req->status_.local_nic = prepared->local_nic;
  req->status_.local_rail_id = prepared->rail_id;
  const std::string peer = std::format("{}:{}", dst_ip, dst_port);
  auto target_progress = create_transfer_progress_state(
      make_transfer_id(req->get_key(), peer), req->get_key(), peer, "target", "rdma", req->total_bytes());
  if (target_progress) {
    auto last_done = std::make_shared<std::atomic<uint64_t>>(0);
    req->set_progress_callbacks(
        [state = target_progress, last_done](uint64_t done, uint64_t /*total*/) {
          const uint64_t prev = last_done->exchange(done, std::memory_order_relaxed);
          if (done > prev) {
            add_transfer_progress_bytes(state, done - prev);
          }
        },
        [this, req_key = req->get_key(), state = std::move(target_progress)](const absl::Status& status) {
          pending_requests_.erase_if_present(req_key);
          finish_transfer_progress(state, status);
        });
  } else {
    req->set_progress_callbacks(
        {}, [this, req_key = req->get_key()](const absl::Status&) { pending_requests_.erase_if_present(req_key); });
  }

  request_queue_.push(req);
  return req->get_future();
}

absl::StatusOr<std::shared_ptr<transport::PreparedReadPlan>> Communicator::prepare_read_plan(
    const routing::ReadPlan& plan,
    uint64_t request_id,
    std::string_view tensor_key) {
  if (plan.source_slices.empty() || plan.slices.empty() || plan.local_regions.empty()) {
    return absl::InvalidArgumentError("read_plan requires non-empty sources, slices, and regions");
  }

  const routing::ReadRouteContext& route = plan.source_slices.front().route;
  if (route.protocol != routing::ConnectionProtocol::kRdma) {
    return absl::UnimplementedError("communicator prepare_read_plan only supports routed RDMA today");
  }

  const absl::Time prepare_started_at = absl::Now();
  const std::string tensor_key_owned(tensor_key);
  auto prepared = std::make_shared<transport::PreparedReadPlan>();
  prepared->logical_plan = plan;
  prepared->remote_endpoint_id = route.remote_endpoint_id;
  prepared->protocol = route.protocol;
  prepared->rail_id = route.rail_id;
  prepared->placements_by_source_slice.resize(plan.source_slices.size());

  absl::Duration rail_resolve_elapsed = absl::ZeroDuration();
  absl::Duration stable_backing_lookup_elapsed = absl::ZeroDuration();
  absl::Duration stable_backing_acquire_elapsed = absl::ZeroDuration();
  absl::Duration local_region_reg_elapsed = absl::ZeroDuration();
  size_t local_regions_reused = 0;
  size_t local_regions_registered = 0;
  uint64_t total_bytes = 0;
  for (const auto& slice : plan.slices) {
    if (slice.source_slice_index >= plan.source_slices.size()) {
      return absl::InvalidArgumentError("read_plan slice references invalid source slice index");
    }
    if (slice.local_region_index >= plan.local_regions.size()) {
      return absl::InvalidArgumentError("read_plan slice references invalid local region index");
    }
    const auto& source_slice = plan.source_slices[slice.source_slice_index];
    if (slice.bytes == 0 || slice.bytes > source_slice.bytes) {
      return absl::InvalidArgumentError("read_plan slice has invalid byte count");
    }
    prepared->placements_by_source_slice[slice.source_slice_index].push_back(
        transport::PreparedSourcePlacement{
            .local_region_index = slice.local_region_index,
            .local_region_offset = slice.local_region_offset,
            .source_slice_offset = slice.source_slice_offset,
            .bytes = slice.bytes,
        });
    if (slice.bytes > std::numeric_limits<uint64_t>::max() - total_bytes) {
      return absl::InvalidArgumentError("read_plan total byte count overflow");
    }
    total_bytes += slice.bytes;
  }
  for (size_t source_index = 0; source_index < prepared->placements_by_source_slice.size(); ++source_index) {
    auto& placements = prepared->placements_by_source_slice[source_index];
    if (placements.empty()) {
      return absl::FailedPreconditionError(
          "read_plan execution requires each source slice to have prepared placements");
    }
    std::sort(
        placements.begin(),
        placements.end(),
        [](const transport::PreparedSourcePlacement& lhs, const transport::PreparedSourcePlacement& rhs) {
          if (lhs.source_slice_offset != rhs.source_slice_offset) {
            return lhs.source_slice_offset < rhs.source_slice_offset;
          }
          return lhs.local_region_index < rhs.local_region_index;
        });
    uint64_t expected_offset = 0;
    for (const auto& placement : placements) {
      if (placement.source_slice_offset != expected_offset) {
        return absl::FailedPreconditionError(
            "read_plan placements must exactly cover each source slice without gaps or overlap");
      }
      if (placement.bytes > std::numeric_limits<uint64_t>::max() - expected_offset) {
        return absl::InvalidArgumentError("read_plan placement coverage overflows source slice range");
      }
      expected_offset += placement.bytes;
    }
    if (expected_offset != plan.source_slices[source_index].bytes) {
      return absl::FailedPreconditionError("read_plan placements must exactly cover the full source slice");
    }
  }

  net_dev_t selected_net_dev = nullptr;
  const absl::Time local_region_device_resolve_started_at = absl::Now();
  for (const auto& region : plan.local_regions) {
    if (region.addr == 0 || region.bytes == 0) {
      return absl::InvalidArgumentError("read_plan local region must have non-zero addr and bytes");
    }
    const absl::Time net_dev_started_at = absl::Now();
    auto net_dev = get_net_dev(region.dev_type, region.dev_id, tensor_key_owned, route.rail_id);
    rail_resolve_elapsed += absl::Now() - net_dev_started_at;
    if (net_dev == nullptr) {
      return absl::InternalError("failed to select RDMA device for read_plan local region");
    }
    if (selected_net_dev == nullptr) {
      selected_net_dev = net_dev;
      prepared->local_nic = net_dev->get_name();
      prepared->rail_id = net_dev->get_rail_id();
      continue;
    }
    if (prepared->local_nic != net_dev->get_name() || prepared->rail_id != net_dev->get_rail_id()) {
      return absl::FailedPreconditionError(
          "read_plan local regions must resolve to one RDMA device/rail in the first cut");
    }
  }
  const absl::Duration local_region_device_resolve_elapsed = absl::Now() - local_region_device_resolve_started_at;

  bool stable_local_mr_reuse_enabled = true;
  if (config_.rdma().has_enable_stable_local_mr_reuse()) {
    stable_local_mr_reuse_enabled = config_.rdma().enable_stable_local_mr_reuse();
  }
  const uint32_t chunk_slots = std::max<uint32_t>(1, stable_local_mr_reuse_chunk_slots_);
  std::shared_ptr<StableLocalBackingState> stable_backing_state;
  std::shared_ptr<void> stable_backing_use;
  bool stable_backing_prewarm_requested = false;
  bool stable_backing_prewarm_complete = false;
  const absl::Time stable_backing_prepare_started_at = absl::Now();
  if (!plan.local_regions.empty()) {
    if (!stable_local_mr_reuse_enabled) {
      prepared->local_registration_fallback_reason = "stable_backing_reuse_disabled";
    } else if (!plan.local_regions.front().stable_backing.has_value()) {
      prepared->local_registration_fallback_reason = "stable_backing_missing";
    } else {
      const auto& first_backing = *plan.local_regions.front().stable_backing;
      bool consistent_backing = first_backing.kind == StableLocalBackingKind::kHostSharedRegion &&
          first_backing.dev_type == COMMUNICATE_ENGINE_DEV_CPU && first_backing.slot_bytes > 0;
      for (size_t i = 0; i < plan.local_regions.size() && consistent_backing; ++i) {
        consistent_backing =
            plan.local_regions[i].stable_backing.has_value() && *plan.local_regions[i].stable_backing == first_backing;
      }
      if (!consistent_backing) {
        prepared->local_registration_fallback_reason = "stable_backing_prepare_failure";
      } else if (selected_net_dev == nullptr) {
        return absl::InternalError("read_plan stable local backing path requires a selected RDMA device");
      } else if (first_backing.slot_bytes > std::numeric_limits<uint64_t>::max() / chunk_slots) {
        prepared->local_registration_fallback_reason = "stable_backing_prepare_failure";
      } else {
        const absl::Time stable_backing_lookup_started_at = absl::Now();
        {
          absl::MutexLock lock(&stable_local_backings_mu_);
          auto it = stable_local_backings_.find(first_backing.backing_id);
          if (it != stable_local_backings_.end()) {
            stable_backing_state = it->second;
          }
        }
        stable_backing_lookup_elapsed += absl::Now() - stable_backing_lookup_started_at;
        if (stable_backing_state == nullptr) {
          prepared->local_registration_fallback_reason = "stable_backing_missing";
        } else {
          stable_backing_prewarm_requested = stable_backing_state->prewarm_requested_enabled();
          stable_backing_prewarm_complete = stable_backing_state->prewarm_complete_for_test();
          const absl::Time stable_backing_acquire_started_at = absl::Now();
          stable_backing_use = stable_backing_state->acquire_use();
          stable_backing_acquire_elapsed += absl::Now() - stable_backing_acquire_started_at;
          if (stable_backing_use == nullptr) {
            prepared->local_registration_fallback_reason = "stable_backing_prepare_failure";
            stable_backing_state.reset();
          } else {
            prepared->stable_backing_id = first_backing.backing_id;
            prepared->stable_backing_chunk_bytes = first_backing.slot_bytes * chunk_slots;
            absl::flat_hash_set<uint64_t> chunk_indices;
            std::vector<transport::PreparedLocalRegion> stable_regions;
            stable_regions.reserve(plan.local_regions.size());
            uint32_t chunk_cache_hits = 0;
            uint32_t chunk_cache_misses = 0;
            uint32_t chunk_waits = 0;
            uint32_t chunk_lazy_registrations = 0;
            for (const auto& region : plan.local_regions) {
              auto chunk_or = stable_backing_state->ensure_chunk(
                  selected_net_dev,
                  prepared->rail_id,
                  first_backing.slot_bytes,
                  chunk_slots,
                  region.addr,
                  region.bytes);
              if (!chunk_or.ok()) {
                VLOG(2) << "communicator.read_plan_prepare_chunk_failed"
                        << " request_id=" << request_id << " tensor_key=" << tensor_key_owned
                        << " backing_id=" << first_backing.backing_id << " rail_id=" << prepared->rail_id
                        << " status=" << chunk_or.status();
                prepared->local_registration_fallback_reason = "stable_backing_prepare_failure";
                prepared->stable_backing_id.clear();
                prepared->stable_backing_chunk_bytes = 0;
                stable_backing_use.reset();
                stable_backing_state.reset();
                stable_regions.clear();
                chunk_indices.clear();
                chunk_cache_hits = 0;
                chunk_cache_misses = 0;
                chunk_waits = 0;
                chunk_lazy_registrations = 0;
                break;
              }
              const auto& chunk = *chunk_or;
              chunk_indices.insert(chunk.chunk_index);
              if (chunk.cache_hit) {
                ++chunk_cache_hits;
              } else {
                ++chunk_cache_misses;
              }
              if (chunk.waited_on_inflight) {
                ++chunk_waits;
              }
              if (chunk.registered_now) {
                ++chunk_lazy_registrations;
              }
              stable_regions.push_back(
                  transport::PreparedLocalRegion{
                      .logical_region = region,
                      .rail_id = chunk.rail_id,
                      .nic_name = chunk.nic_name,
                      .tensor = nullptr,
                      .mr = chunk.mr,
                      .keepalive = stable_backing_use,
                  });
            }
            if (stable_backing_use != nullptr && stable_regions.size() == plan.local_regions.size()) {
              prepared->local_registration_mode = "stable_backing_reuse";
              prepared->local_registration_fallback_reason.clear();
              prepared->local_regions = std::move(stable_regions);
              prepared->stable_backing_chunk_count = static_cast<uint32_t>(chunk_indices.size());
              prepared->stable_backing_chunk_cache_hits = chunk_cache_hits;
              prepared->stable_backing_chunk_cache_misses = chunk_cache_misses;
              prepared->stable_backing_chunk_waits = chunk_waits;
              prepared->stable_backing_chunk_lazy_registrations = chunk_lazy_registrations;
              prepared->stable_backing_prewarm_requested = stable_backing_prewarm_requested;
              prepared->stable_backing_prewarm_complete = stable_backing_prewarm_complete;
              local_regions_reused = prepared->local_regions.size();
            }
          }
        }
      }
    }
  }
  const absl::Duration stable_backing_prepare_elapsed = absl::Now() - stable_backing_prepare_started_at;
  {
    std::ostringstream log;
    log << "communicator.read_plan_prepare"
        << " request_id=" << request_id << " tensor_key=" << tensor_key_owned
        << " transport_request_id=" << (plan.transport_request_id.empty() ? "<unset>" : plan.transport_request_id)
        << " local_registration_mode=" << prepared->local_registration_mode
        << " fallback_reason=" << prepared->local_registration_fallback_reason
        << " stable_backing_id=" << prepared->stable_backing_id
        << " stable_backing_prewarm_requested=" << stable_backing_prewarm_requested
        << " stable_backing_prewarm_complete=" << stable_backing_prewarm_complete
        << " stable_backing_chunk_bytes=" << prepared->stable_backing_chunk_bytes
        << " stable_backing_chunk_count=" << prepared->stable_backing_chunk_count
        << " stable_backing_chunk_cache_hits=" << prepared->stable_backing_chunk_cache_hits
        << " stable_backing_chunk_cache_misses=" << prepared->stable_backing_chunk_cache_misses
        << " stable_backing_chunk_waits=" << prepared->stable_backing_chunk_waits
        << " stable_backing_chunk_lazy_registrations=" << prepared->stable_backing_chunk_lazy_registrations
        << " rail_id=" << prepared->rail_id << " local_regions=" << plan.local_regions.size()
        << " stable_local_mr_reuse_enabled=" << stable_local_mr_reuse_enabled
        << " device_resolve_ms=" << absl::ToDoubleMilliseconds(local_region_device_resolve_elapsed)
        << " stable_backing_prepare_ms=" << absl::ToDoubleMilliseconds(stable_backing_prepare_elapsed)
        << " stable_backing_lookup_ms=" << absl::ToDoubleMilliseconds(stable_backing_lookup_elapsed)
        << " stable_backing_acquire_ms=" << absl::ToDoubleMilliseconds(stable_backing_acquire_elapsed);
    if (!plan.transport_request_id.empty()) {
      LOG(INFO) << log.str();
    } else {
      VLOG(2) << log.str();
    }
  }

  const absl::Time local_region_prepare_started_at = absl::Now();
  if (prepared->local_registration_mode != "stable_backing_reuse") {
    prepared->local_regions.clear();
    prepared->local_regions.reserve(plan.local_regions.size());
    for (size_t region_index = 0; region_index < plan.local_regions.size(); ++region_index) {
      const auto& region = plan.local_regions[region_index];
      const absl::Time net_dev_started_at = absl::Now();
      auto net_dev = get_net_dev(region.dev_type, region.dev_id, tensor_key_owned, route.rail_id);
      rail_resolve_elapsed += absl::Now() - net_dev_started_at;
      if (net_dev == nullptr) {
        return absl::InternalError("failed to select RDMA device for read_plan local region");
      }
      auto local_tensor = std::make_shared<PartitionTensor>(
          std::format("read_plan:{}:{}", request_id, region_index),
          region.addr,
          region.bytes,
          region.dev_type,
          net_dev);
      local_tensor->set_read_unready();
      if (region.dev_type == COMMUNICATE_ENGINE_DEV_GPU) {
        local_tensor->set_device_id(region.dev_id);
      }
      if (local_tensor->get_dev_by_rail(net_dev->get_rail_id()) == nullptr) {
        local_tensor->add_dev(net_dev);
      }
      if (!local_tensor->is_registered(net_dev)) {
        const absl::Time reg_started_at = absl::Now();
        net_dev->reg_async(local_tensor);
        local_tensor->wait_mr_ready(net_dev);
        local_region_reg_elapsed += absl::Now() - reg_started_at;
        if (local_tensor->get_mr(net_dev) == nullptr) {
          return absl::InternalError("failed to register read_plan local region MR");
        }
      }
      local_tensor->set_read_ready();
      prepared->local_regions.push_back(
          transport::PreparedLocalRegion{
              .logical_region = region,
              .rail_id = net_dev->get_rail_id(),
              .nic_name = net_dev->get_name(),
              .tensor = local_tensor,
          });
      ++local_regions_registered;
    }
  }
  const absl::Duration local_region_prepare_elapsed = absl::Now() - local_region_prepare_started_at;
  {
    std::ostringstream log;
    log << "communicator.read_plan_prepare_complete"
        << " request_id=" << request_id << " tensor_key=" << tensor_key_owned
        << " transport_request_id=" << (plan.transport_request_id.empty() ? "<unset>" : plan.transport_request_id)
        << " local_registration_mode=" << prepared->local_registration_mode
        << " fallback_reason=" << prepared->local_registration_fallback_reason
        << " local_regions=" << prepared->local_regions.size() << " local_regions_reused=" << local_regions_reused
        << " local_regions_registered=" << local_regions_registered
        << " stable_backing_chunk_count=" << prepared->stable_backing_chunk_count
        << " stable_backing_chunk_cache_hits=" << prepared->stable_backing_chunk_cache_hits
        << " stable_backing_chunk_cache_misses=" << prepared->stable_backing_chunk_cache_misses
        << " stable_backing_chunk_waits=" << prepared->stable_backing_chunk_waits
        << " stable_backing_chunk_lazy_registrations=" << prepared->stable_backing_chunk_lazy_registrations
        << " stable_backing_prewarm_requested=" << prepared->stable_backing_prewarm_requested
        << " stable_backing_prewarm_complete=" << prepared->stable_backing_prewarm_complete
        << " rail_resolve_ms=" << absl::ToDoubleMilliseconds(rail_resolve_elapsed)
        << " local_region_prepare_ms=" << absl::ToDoubleMilliseconds(local_region_prepare_elapsed)
        << " local_region_reg_ms=" << absl::ToDoubleMilliseconds(local_region_reg_elapsed)
        << " total_prepare_ms=" << absl::ToDoubleMilliseconds(absl::Now() - prepare_started_at);
    if (!plan.transport_request_id.empty()) {
      LOG(INFO) << log.str();
    } else {
      VLOG(2) << log.str();
    }
  }
  prepared->total_bytes = total_bytes;
  return prepared;
}

future_read_result_t Communicator::read_tensor(
    const std::string& key,
    uint64_t addr,
    uint64_t bytes,
    int dev_type,
    int dev_id, // gpu_id
    const std::string& dst_ip,
    uint16_t dst_port,
    uint64_t remote_offset) {
  if (!inited_.load()) {
    LOG(ERROR) << "failed to read a tensor with a un-inited engine";
    return transport::ReadRequest::get_read_result_future("failed to read tensor through un-initiated engine");
  }
  net_dev_t net_dev = nullptr;
  if (enable_rdma_) {
    // with rail
    net_dev = get_net_dev(dev_type, dev_id, key);
    if (net_dev == nullptr) {
      return transport::ReadRequest::get_read_result_future("failed to get net dev for the rdma connection");
    }
  } else if (COMMUNICATE_ENGINE_DEV_GPU == dev_type && !gpu_memory_stager_) {
    return transport::ReadRequest::get_read_result_future(
        "failed to read GPU tensor with tcp: GPU stager not initialized");
  }

  LOG(INFO) << "read tensor:"
            << " dst=" << dst_ip << ":" << dst_port << ", key=" << key << " ,offset=" << remote_offset
            << ", net_dev=" << (net_dev == nullptr ? "none" : net_dev->get_name());

  // Request-side destination buffers must stay request-scoped. Reusing the
  // shared source tensor store here causes same-process loopback reads to
  // overwrite the advertised source tensor metadata with the destination
  // window metadata for the in-flight request.
  auto local_tensor = std::make_shared<PartitionTensor>(key, addr, bytes, dev_type, net_dev);
  local_tensor->set_read_unready();
  if (dev_type == COMMUNICATE_ENGINE_DEV_GPU) {
    local_tensor->set_device_id(dev_id);
  }

  if (enable_rdma_ && net_dev != nullptr) {
    local_tensor->set_read_unready();
    if (local_tensor->get_dev_by_rail(net_dev->get_rail_id()) == nullptr) {
      local_tensor->add_dev(net_dev);
    }
    if (!local_tensor->is_registered(net_dev)) {
      // Request destination buffers are caller-owned and request-scoped. They
      // do not have the stable slab lifetime required by meta_mr_cache_, so
      // each read request must register its own MR and retire it with the
      // request-local PartitionTensor.
      net_dev->reg_async(local_tensor);
    }
  }
  local_tensor->set_read_ready();

  auto req = std::make_shared<transport::ReadRequest>(
      key,
      dst_ip,
      dst_port,
      local_tensor,
      remote_offset,
      next_request_id_.fetch_add(1, std::memory_order_relaxed),
      net_dev != nullptr ? net_dev->get_rail_id() : -1);
  const std::string req_key = req->get_key();
  req->status_.transport_is_rdma = enable_rdma_;
  if (net_dev != nullptr) {
    req->status_.local_nic = net_dev->get_name();
    req->status_.local_rail_id = net_dev->get_rail_id();
  }
  const std::string peer = std::format("{}:{}", dst_ip, dst_port);
  auto target_progress = create_transfer_progress_state(
      make_transfer_id(req->get_key(), peer), req->get_key(), peer, "target", enable_rdma_ ? "rdma" : "mtcp", bytes);
  if (target_progress) {
    auto last_done = std::make_shared<std::atomic<uint64_t>>(0);
    req->set_progress_callbacks(
        [state = target_progress, last_done](uint64_t done, uint64_t /*total*/) {
          const uint64_t prev = last_done->exchange(done, std::memory_order_relaxed);
          if (done > prev) {
            add_transfer_progress_bytes(state, done - prev);
          }
        },
        [this, req_key, state = std::move(target_progress)](const absl::Status& status) {
          pending_requests_.erase_if_present(req_key);
          finish_transfer_progress(state, status);
        });
  } else {
    req->set_progress_callbacks(
        {}, [this, req_key](const absl::Status&) { pending_requests_.erase_if_present(req_key); });
  }
  LOG(INFO) << "[read_tensor] Creating request: key=" << key << " dst=" << dst_ip << ":" << dst_port
            << " req_key=" << req_key << " req_rail=" << req->get_rail_id();
  request_queue_.push(req);
  LOG(INFO) << "[read_tensor] Request pushed to queue successfully for key=" << key;
  return req->get_future();
}

absl::Status Communicator::register_tensor_ex(
    const std::string& tensor_key,
    uint64_t addr,
    uint64_t bytes,
    int dev_type,
    int dev_id,
    const RegisterTensorOptions& opts) {
  const absl::Time total_started_at = absl::Now();
  // Check for zero-size tensor
  if (bytes == 0) {
    return absl::InvalidArgumentError("Cannot register zero-size tensor");
  }
  if (opts.direct_rdma_required && !opts.direct_rdma_enabled) {
    return absl::InvalidArgumentError("direct_rdma_required requires direct_rdma_enabled=true");
  }

  net_dev_t net_dev = nullptr;
  absl::Duration net_dev_select_elapsed = absl::ZeroDuration();
  if (enable_rdma_) {
    const absl::Time net_dev_select_started_at = absl::Now();
    net_dev = get_net_dev(dev_type, dev_id, tensor_key);
    net_dev_select_elapsed = absl::Now() - net_dev_select_started_at;
    if (net_dev == nullptr) {
      return absl::InternalError("failed to get net dev");
    }
  }

  // Note: In TCP mode, GPU tensors are now supported with staging
  if (COMMUNICATE_ENGINE_DEV_GPU == dev_type) {
    if (dev_id < 0 || dev_id >= 16) {
      return absl::InternalError("failed to register tensor on a invalid gpu");
    }
  }

  VLOG(1) << "register tensor:"
          << " key=" << tensor_key << ", addr=" << addr << ", bytes=" << bytes << ", gpu=" << dev_id
          << ", net_dev=" << (net_dev == nullptr ? "none" : net_dev->get_name());

  auto tensor = std::make_shared<PartitionTensor>(tensor_key, addr, bytes, dev_type, net_dev);
  tensor->set_read_ready();

  std::size_t visible_dev_count = 0;
  absl::Duration visible_dev_attach_elapsed = absl::ZeroDuration();
  if (enable_rdma_ && rdma_context_ != nullptr && dev_type == COMMUNICATE_ENGINE_DEV_CPU && opts.direct_rdma_enabled) {
    const absl::Time visible_dev_attach_started_at = absl::Now();
    for (const auto& visible_dev : rdma_context_->list_devs()) {
      tensor->add_dev(visible_dev);
      ++visible_dev_count;
    }
    visible_dev_attach_elapsed = absl::Now() - visible_dev_attach_started_at;
  }

  // Set device ID for GPU tensors
  if (dev_type == COMMUNICATE_ENGINE_DEV_GPU) {
    tensor->set_device_id(dev_id);
  }

  // Mark tensors that need staging if requested by policy
  if (opts.needs_staging) {
    tensor->set_needs_staging(true);
  }
  if (opts.direct_rdma_enabled) {
    tensor->set_direct_rdma_enabled(true);
  }
  if (opts.direct_rdma_required) {
    tensor->set_direct_rdma_required(true);
  }

  absl::Duration reg_async_elapsed = absl::ZeroDuration();
  absl::Duration wait_mr_elapsed = absl::ZeroDuration();
  if (enable_rdma_ && opts.register_mr) {
    const absl::Time reg_async_started_at = absl::Now();
    net_dev->reg_async(tensor);
    reg_async_elapsed = absl::Now() - reg_async_started_at;
    if (!opts.async) {
      const absl::Time wait_mr_started_at = absl::Now();
      if (tensor->get_mr(net_dev) == nullptr) {
        return absl::InternalError("failed to register mr");
      }
      wait_mr_elapsed = absl::Now() - wait_mr_started_at;
    }
  }

  const absl::Time store_register_started_at = absl::Now();
  store_.register_tensor(tensor);
  const absl::Duration store_register_elapsed = absl::Now() - store_register_started_at;
  {
    absl::MutexLock lock(&tensor_read_mu_);
    auto it = tensor_read_states_.find(tensor_key);
    if (it != tensor_read_states_.end() && it->second != nullptr) {
      it->second->retiring = false;
      if (it->second->inflight == 0) {
        tensor_read_states_.erase(it);
      }
    }
  }
  if (opts.direct_rdma_enabled || opts.register_mr) {
    VLOG(2) << "communicator.register_tensor_ex"
            << " key=" << tensor_key << " mem_type=" << dev_type << " dev_id=" << dev_id << " bytes=" << bytes
            << " selected_net_dev=" << (net_dev == nullptr ? "none" : net_dev->get_name())
            << " net_dev_select_ms=" << absl::ToDoubleMilliseconds(net_dev_select_elapsed)
            << " visible_dev_count=" << visible_dev_count
            << " visible_dev_attach_ms=" << absl::ToDoubleMilliseconds(visible_dev_attach_elapsed)
            << " register_mr=" << opts.register_mr << " direct_rdma_enabled=" << opts.direct_rdma_enabled
            << " direct_rdma_required=" << opts.direct_rdma_required
            << " reg_async_ms=" << absl::ToDoubleMilliseconds(reg_async_elapsed)
            << " wait_mr_ms=" << absl::ToDoubleMilliseconds(wait_mr_elapsed)
            << " store_register_ms=" << absl::ToDoubleMilliseconds(store_register_elapsed)
            << " total_ms=" << absl::ToDoubleMilliseconds(absl::Now() - total_started_at);
  }
  return absl::OkStatus();
}

absl::Status Communicator::handle_rdma_read_request(
    const channel_t& channel,
    const tcp_transport_t& control_transport,
    ProtoReadRequest& request,
    const std::shared_ptr<PartitionTensor>& tensor,
    std::shared_ptr<void> read_guard) {
  if (!enable_rdma_) {
    return absl::FailedPreconditionError("RDMA transport disabled");
  }

  auto flow_state = channel->flow_state();
  if (!flow_state) {
    return absl::InternalError("channel missing flow state");
  }

  tensor->wait_read_ready();
  auto dev = tensor->get_dev();
  if (dev == nullptr) {
    return absl::InternalError("tensor missing RDMA device");
  }
  // The request rail is the initiator's local rail, not an instruction for the
  // target to rebind its own tensor onto the same rail number. For cross-host
  // GPU/NIC mapping experiments the target must keep its local preferred NIC.
  request.rail_id = dev->get_rail_id();

  const bool tensor_on_cpu = tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_CPU;
  const int device_id = tensor->get_device_id();

  std::shared_ptr<MemoryStager> stager;
  if (tensor_on_cpu) {
    stager = get_cpu_stager_for_nic(dev->get_name());
  } else if (use_gpu_vram_staging_) {
    stager = get_gpu_vram_stager_for_id(device_id);
  } else {
    stager = get_gpu_mem_stager_for_id(device_id);
  }

  if (!stager) {
    if (tensor_on_cpu) {
      stager = memory_stager_;
    } else if (use_gpu_vram_staging_) {
      // VRAM staging is RDMA-only and depends on a per-GPU pool. If the tensor reports an
      // unexpected device id, fall back to host-pinned GPU staging so we never end up with
      // a nullptr stager later in the RDMA staging path.
      LOG_FIRST_N(WARNING, 1) << "GPU VRAM staging enabled but no VRAM stager for device_id=" << device_id
                              << "; falling back to host-pinned staging";
      stager = get_gpu_mem_stager_for_id(device_id);
      if (!stager) {
        stager = gpu_memory_stager_;
      }
    } else {
      stager = gpu_memory_stager_;
    }
  }

  const uint64_t total_bytes = request.bytes;
  const uint64_t start_offset = request.offset;
  const std::string tensor_key(reinterpret_cast<const char*>(request.tensor_key));
  const std::string request_key = transport::get_request_instance_key(tensor_key, start_offset, request.request_id);
  const std::string peer = control_transport ? control_transport->get_remote_url() : std::string();
  const std::string transfer_id = make_transfer_id(request_key, peer);
  (void)register_source_transfer_progress(request_key, peer, "rdma", total_bytes, std::move(read_guard));
  FlowCreditLedger* ledger_ptr = &flow_state->ledger;
  MrCache* mr_cache_ptr = meta_mr_cache_.get();

  const bool direct_requested = tensor->direct_rdma_enabled();
  bool use_direct = false;
  ibv_mr* direct_mr = nullptr;
  DirectFallbackReason fallback_reason = DirectFallbackReason::kNone;

  if (direct_requested) {
    if (tensor_on_cpu || tensor->get_mem_type() != COMMUNICATE_ENGINE_DEV_GPU) {
      fallback_reason = DirectFallbackReason::kNotGpu;
    } else if (tensor->needs_staging()) {
      fallback_reason = DirectFallbackReason::kNeedsStaging;
    } else {
      const uint64_t tensor_bytes = tensor->get_bytes();
      if (start_offset > tensor_bytes || total_bytes > (tensor_bytes - start_offset)) {
        fallback_reason = DirectFallbackReason::kOutOfRange;
      } else {
        tensor->wait_mr_ready(dev);
        direct_mr = tensor->get_mr(dev);
        if (direct_mr == nullptr || !tensor->has_registered_mr(dev)) {
          fallback_reason = DirectFallbackReason::kMrUnavailable;
        } else {
          use_direct = true;
        }
      }
    }
  }

  if (!use_direct && !stager) {
    finish_source_transfer_progress(transfer_id, absl::FailedPreconditionError("no staging backend available"));
    return absl::FailedPreconditionError("no staging backend available for tensor");
  }
  if (direct_requested && !use_direct && fallback_reason != DirectFallbackReason::kNone) {
    LOG_FIRST_N(WARNING, 1) << "RDMA zero-copy fallback for tensor=" << tensor_key
                            << " reason=" << DirectFallbackReasonToString(fallback_reason);
    record_direct_fallback_metric(fallback_reason);
    if (tensor->direct_rdma_required()) {
      const absl::Status direct_required_status = absl::FailedPreconditionError(
          absl::StrCat(
              "direct RDMA required for tensor=",
              tensor_key,
              " but unavailable: reason=",
              DirectFallbackReasonToString(fallback_reason)));
      finish_source_transfer_progress(transfer_id, direct_required_status);
      return direct_required_status;
    }
  }

  const uint64_t chunk_size = use_direct
      ? compute_direct_chunk_bytes(
            total_bytes, direct_rdma_chunk_bytes_, flow_state->max_window_segments, config_.rdma().qp_count())
      : (stager && stager->get_chunk_size() > 0 ? stager->get_chunk_size() : request.bytes);

  const bool using_gpu_vram_stager =
      !tensor_on_cpu && use_gpu_vram_staging_ && stager && stager == get_gpu_vram_stager_for_id(device_id);
  const v1::RdmaConfig::StagedRdmaBackend staged_backend = using_gpu_vram_stager
      ? v1::RdmaConfig::STAGED_RDMA_BACKEND_GPU_VRAM
      : v1::RdmaConfig::STAGED_RDMA_BACKEND_HOST_PINNED;
  auto session = std::make_shared<RdmaReadSession>();
  session->request = request;
  session->tensor_key = tensor_key;
  session->request_key = request_key;
  session->tensor = tensor;
  session->stager = stager;
  session->dev = dev;
  session->control_transport = control_transport;
  session->transfer_id = transfer_id;
  session->zero_copy = use_direct;
  session->source_stage_profile = std::make_shared<RdmaSourceStageProfile>();

  uint32_t window_segments = flow_state->max_window_segments;
  if (use_direct) {
    window_segments = compute_direct_window_segments(
        total_bytes, chunk_size, flow_state->max_window_segments, config_.rdma().qp_count());
    session->direct_ledger = std::make_unique<FlowCreditLedger>(static_cast<int>(window_segments));
    ledger_ptr = session->direct_ledger.get();
  }

  auto stage_fn = MakeStageFunction(
      tensor,
      ledger_ptr,
      stager,
      dev,
      mr_cache_ptr,
      tensor_key,
      request_key,
      staged_backend,
      use_direct,
      direct_mr,
      session->source_stage_profile);
  session->window =
      std::make_unique<StagingWindow>(*ledger_ptr, stage_fn, total_bytes, chunk_size, start_offset, window_segments);

  size_t pending_reads = 0;
  {
    absl::MutexLock lock(&flow_state->rdma_pending_reads_mu);
    flow_state->rdma_pending_reads.push_back(session);
    pending_reads = flow_state->rdma_pending_reads.size();
  }
  LOG(INFO) << "[rdma_session] queued request=" << request_key << " zero_copy=" << use_direct
            << " pending_reads=" << pending_reads << " outstanding_credit=" << ledger_ptr->outstanding_credit()
            << " window_segments=" << window_segments;

  auto status = resume_rdma_reads(channel);
  {
    absl::MutexLock lock(&flow_state->rdma_pending_reads_mu);
    pending_reads = flow_state->rdma_pending_reads.size();
  }
  LOG(INFO) << "[rdma_session] resume status request=" << request_key << " status=" << status
            << " pending_reads=" << pending_reads << " outstanding_credit=" << flow_state->ledger.outstanding_credit();
  if (!status.ok()) {
    finish_source_transfer_progress(transfer_id, status);
    return status;
  }

  return absl::OkStatus();
}

absl::Status Communicator::resume_rdma_reads(const channel_t& channel) {
  auto flow_state = channel->flow_state();
  if (!flow_state) {
    return absl::OkStatus();
  }
  bool expected = false;
  if (!flow_state->rdma_refill_in_progress.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    flow_state->rdma_refill_requested.store(true, std::memory_order_release);
    size_t pending_reads = 0;
    {
      absl::MutexLock lock(&flow_state->rdma_pending_reads_mu);
      pending_reads = flow_state->rdma_pending_reads.size();
    }
    LOG(INFO) << "[rdma_resume] refill already in progress; request another pass pending_reads=" << pending_reads
              << " outstanding_credit=" << flow_state->ledger.outstanding_credit();
    return absl::OkStatus();
  }

  absl::Status first_error = absl::OkStatus();

  while (true) {
    size_t pending_reads = 0;
    {
      absl::MutexLock lock(&flow_state->rdma_pending_reads_mu);
      pending_reads = flow_state->rdma_pending_reads.size();
    }
    LOG(INFO) << "[rdma_resume] pass begin pending_reads=" << pending_reads
              << " outstanding_credit=" << flow_state->ledger.outstanding_credit();
    flow_state->rdma_refill_requested.store(false, std::memory_order_release);

    while (true) {
      std::shared_ptr<RdmaReadSession> session;
      {
        absl::MutexLock lock(&flow_state->rdma_pending_reads_mu);
        if (flow_state->rdma_pending_reads.empty()) {
          break;
        }
        session = flow_state->rdma_pending_reads.front();
        pending_reads = flow_state->rdma_pending_reads.size();
      }
      auto result = session->mode == RdmaReadSession::Mode::kReadPlan ? DriveRdmaPlanSession(*flow_state, *session)
                                                                      : DriveRdmaSession(*flow_state, *session);
      LOG(INFO) << "[rdma_resume] drove request=" << session->request_key << " status=" << result.status
                << " completed=" << result.completed << " made_progress=" << result.made_progress
                << " pending_reads=" << pending_reads
                << " outstanding_credit=" << flow_state->ledger.outstanding_credit();

      if (!result.status.ok()) {
        if (absl::IsResourceExhausted(result.status) || absl::IsUnavailable(result.status)) {
          if (!result.made_progress) {
            absl::MutexLock lock(&flow_state->rdma_pending_reads_mu);
            if (flow_state->rdma_pending_reads.size() > 1 && flow_state->rdma_pending_reads.front() == session) {
              auto queued = flow_state->rdma_pending_reads.front();
              flow_state->rdma_pending_reads.pop_front();
              flow_state->rdma_pending_reads.push_back(std::move(queued));
            }
          }
          break;
        }

        LOG(ERROR) << "Failed to service RDMA read request=" << session->request_key << " status=" << result.status;

        misc::result_t send_res = misc::FAILED;
        if (session->mode == RdmaReadSession::Mode::kReadPlan) {
          auto rsp = EngineMessage::make_message<ProtoReadPlanFailed>(ENGINE_OP_READ_PLAN_FAILED);
          auto* payload = rsp->get_payload<ProtoReadPlanFailed>();
          payload->request_id = session->plan_request.request_id;
          payload->reason = TENSORCAST_READ_FAILED_MEM_MISMATCH;
          send_res = session->control_transport->send(rsp);
        } else {
          auto rsp = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
          auto* payload = rsp->get_payload<ProtoReadFailed>();
          memcpy(payload->tensor_key, session->request.tensor_key, kMaxTensorNameLen);
          payload->offset = session->request.offset;
          payload->request_id = session->request.request_id;
          payload->reason = TENSORCAST_READ_FAILED_MEM_MISMATCH;
          send_res = session->control_transport->send(rsp);
        }
        if (send_res != misc::SUCCESS) {
          LOG(WARNING) << "Failed to send READ_FAILED after staging failure: " << send_res;
        }

        {
          absl::MutexLock lock(&flow_state->rdma_pending_reads_mu);
          if (!flow_state->rdma_pending_reads.empty() && flow_state->rdma_pending_reads.front() == session) {
            flow_state->rdma_pending_reads.pop_front();
          }
        }
        log_rdma_source_stage_summary(*session, result.status);
        if (!session->transfer_id.empty()) {
          this->finish_source_transfer_progress(session->transfer_id, result.status);
        }
        if (first_error.ok()) {
          first_error = result.status;
        }
        continue;
      }

      if (result.completed) {
        {
          absl::MutexLock lock(&flow_state->rdma_pending_reads_mu);
          if (!flow_state->rdma_pending_reads.empty() && flow_state->rdma_pending_reads.front() == session) {
            flow_state->rdma_pending_reads.pop_front();
          }
        }
        log_rdma_source_stage_summary(*session, result.status);
        continue;
      }

      if (result.made_progress) {
        // Wait for RDMA ACKs to return credit before continuing.
        break;
      }

      break; // Defensive: no progress and no status; avoid tight loop.
    }

    flow_state->rdma_refill_in_progress.store(false, std::memory_order_release);
    {
      absl::MutexLock lock(&flow_state->rdma_pending_reads_mu);
      pending_reads = flow_state->rdma_pending_reads.size();
    }
    const bool requested_again = flow_state->rdma_refill_requested.load(std::memory_order_acquire);
    LOG(INFO) << "[rdma_resume] pass end pending_reads=" << pending_reads << " requested_again=" << requested_again
              << " outstanding_credit=" << flow_state->ledger.outstanding_credit();
    if (!requested_again || pending_reads == 0 || !first_error.ok()) {
      break;
    }

    expected = false;
    if (!flow_state->rdma_refill_in_progress.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      break;
    }
  }

  return first_error;
}

absl::Status Communicator::handle_mtcp_read_request(
    const channel_t& channel,
    const tcp_transport_t& control_transport,
    const ProtoReadRequest& request,
    const std::shared_ptr<PartitionTensor>& tensor,
    std::shared_ptr<void> read_guard) {
  const std::string tensor_key(reinterpret_cast<const char*>(request.tensor_key));
  MtcpReadTask task;
  task.channel = channel;
  task.control_transport = control_transport;
  task.request = request;
  task.tensor = tensor;
  task.read_guard = std::move(read_guard);

  auto flow_state = channel->flow_state();
  if (!flow_state) {
    auto status = absl::InternalError("channel missing flow state");
    fail_mtcp_read_task(task, status);
    return status;
  }

  auto transport = channel->get_mtcp();
  if (transport == nullptr) {
    auto slot_or = acquire_gpu_channel_slot();
    if (!slot_or.ok()) {
      auto fail_msg = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
      auto* payload = fail_msg->get_payload<ProtoReadFailed>();
      memcpy(payload->tensor_key, request.tensor_key, kMaxTensorNameLen);
      payload->offset = request.offset;
      payload->request_id = request.request_id;
      payload->reason = absl::IsResourceExhausted(slot_or.status()) ? TENSORCAST_READ_FAILED_RESOURCE_EXHAUSTED
                                                                    : TENSORCAST_READ_FAILED_MEM_MISMATCH;
      misc::result_t send_res = control_transport->send(fail_msg);
      if (send_res != misc::SUCCESS) {
        LOG(WARNING) << "Failed to send READ_FAILED after GPU channel limit hit: key=" << tensor_key
                     << " res=" << send_res;
      }
      return slot_or.status();
    }
    std::shared_ptr<void> gpu_slot_handle = std::move(slot_or).value();
    transport = std::make_shared<transport::MTcpTransport>(
        mtcp_conn_count_,
        gsl::not_null<std::shared_ptr<MemoryStager>>{memory_stager_},
        gsl::not_null<std::shared_ptr<MemoryStager>>{gpu_memory_stager_},
        gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{gpu_memory_pool_},
        buffers_per_flow_);
    channel->set_transport(transport);
    channel->set_gpu_slot_handle(std::move(gpu_slot_handle));
  }
  transport->set_tcp_tos(config_.transport().tcp_tos());

  const bool needs_gpu_staging =
      task.tensor->needs_staging() || task.tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_GPU;
  std::shared_ptr<MemoryStager> stager;
  if (needs_gpu_staging) {
    stager = get_gpu_mem_stager_for_id(task.tensor->get_device_id());
    if (!stager) {
      stager = gpu_memory_stager_;
    }
  } else {
    stager = memory_stager_;
  }
  if (!stager) {
    auto status = absl::FailedPreconditionError("no staging backend available for MTCP tensor");
    fail_mtcp_read_task(task, status);
    return status;
  }
  const uint64_t stage_chunk_bytes = stager->get_chunk_size() > 0 ? stager->get_chunk_size() : request.bytes;
  task.stager = stager;
  task.stage_chunk_bytes = stage_chunk_bytes;

  channel->mtcp_request_started();
  bool request_handed_off = false;
  absl::Cleanup mtcp_request_guard = [&]() {
    if (!request_handed_off) {
      channel->mtcp_request_finished();
    }
  };

  auto rsp = std::make_shared<EngineMessage>(
      ENGINE_OP_READ_RESPONSE_EX,
      static_cast<uint32_t>(sizeof(ProtoReadResponseExHeader) + sizeof(ProtoReadResponseExSeg)));
  auto* hdr = rsp->get_payload<ProtoReadResponseExHeader>();
  memcpy(hdr->tensor_key, request.tensor_key, kMaxTensorNameLen);
  hdr->transport_type = ENGINE_TRANSPORT_MTCP;
  hdr->staged = 0;
  misc::STRCPY(hdr->nic_name, "");
  hdr->request_offset = request.offset;
  hdr->request_id = request.request_id;
  hdr->num_segments = 1;
  hdr->window_seq = 0;
  const uint64_t stage_chunk_hint = task.stage_chunk_bytes > 0 ? task.stage_chunk_bytes : request.bytes;
  hdr->credit_granted = static_cast<uint32_t>(
      std::min<uint64_t>(stage_chunk_hint, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
  hdr->request_offset = request.offset;
  hdr->more_segments = 0;
  absl::Status shutdown_status = absl::CancelledError("communicator shutting down");
  if (stop_.load(std::memory_order_relaxed)) {
    fail_mtcp_read_task(task, shutdown_status);
    return shutdown_status;
  }
  auto* s0 =
      reinterpret_cast<ProtoReadResponseExSeg*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader));
  s0->addr = 0;
  s0->rkey = 0;
  s0->bytes = static_cast<uint32_t>(request.bytes);
  s0->offset = request.offset;

  misc::result_t send_res = control_transport->send(rsp);
  if (send_res != misc::SUCCESS) {
    return absl::InternalError("failed to send READ_RESPONSE_EX for MTCP");
  }

  if (stop_.load(std::memory_order_relaxed)) {
    fail_mtcp_read_task(task, shutdown_status);
    return shutdown_status;
  }

  const std::string request_key =
      communicator::transport::get_request_instance_key(tensor_key, request.offset, request.request_id);
  const std::string peer = control_transport ? control_transport->get_remote_url() : std::string();
  const std::string transfer_id = make_transfer_id(request_key, peer);
  (void)register_source_transfer_progress(request_key, peer, "mtcp", request.bytes, std::move(task.read_guard));

  if (mtcp_staging_queue_.push(task) != misc::SUCCESS) {
    auto status = absl::InternalError("failed to enqueue MTCP staging task");
    finish_source_transfer_progress(transfer_id, status);
    fail_mtcp_read_task(task, status);
    return status;
  }

  request_handed_off = true;
  return absl::OkStatus();
}

absl::Status Communicator::unregister_tensor(const std::string& tensor_key) {
  const absl::Status drain_status = wait_for_tensor_reads_to_drain(tensor_key, kUnregisterTensorDrainTimeout);
  if (!drain_status.ok()) {
    LOG(ERROR) << "[unregister_tensor] timed out draining in-flight source reads for key=" << tensor_key
               << " status=" << drain_status;
    return drain_status;
  }

  if (store_.get_tensor(tensor_key) == nullptr) {
    VLOG(1) << "[unregister_tensor] key not found, treating as idempotent OK: " << tensor_key;
  } else {
    store_.unregister_tensor(tensor_key);
  }

  {
    absl::MutexLock lock(&tensor_read_mu_);
    auto it = tensor_read_states_.find(tensor_key);
    if (it != tensor_read_states_.end() && it->second != nullptr) {
      it->second->retiring = false;
      if (it->second->inflight == 0) {
        tensor_read_states_.erase(it);
      }
    }
  }
  return absl::OkStatus();
}

misc::result_t Communicator::on_new_client(const tcp_transport_t& t) {
  LOG(INFO) << "[on_new_client] New client connection from " << t->get_remote_url() << " fd=" << t->get_fd();
  auto channel =
      std::make_shared<Channel>(t, enable_rdma_ ? CHANNEL_RDMA : CHANNEL_MTCP, buffers_per_flow_, max_window_segments_);
  channels_.put(t->get_remote_url(), channel);
  t->set_recv_func([this](const tcp_transport_t& t) -> misc::result_t {
    auto channel = this->channels_.get(t->get_remote_url());
    if (channel == nullptr) {
      LOG(WARNING) << "failed to process recv message due to nil channel: " << t->get_remote_url();
      return INTERNAL_ERROR;
    }
    ProtoHeader header = {};
    COMM_CHECK(t->recv<ProtoHeader>(&header));
    auto msg = std::make_shared<EngineMessage>(&header);
    COMM_CHECK(t->recv(msg->get_payload<uint8_t>(), msg->get_payload_size()));
    return this->on_receive_request(channel, t, msg);
  });
  t->set_close_func([this](const tcp_transport_t& t) {
    const std::string url_key = t->get_remote_url();
    LOG(INFO) << "[on_new_client] Client connection closed: " << url_key;
    auto channel = channels_.get(url_key);
    if (channel && channel->get_control().get() == t.get()) {
      channel->close();
      if (!channels_.erase_if(url_key, channel)) {
        VLOG(1) << "[on_new_client] Channel already removed or replaced for " << url_key;
      }
    } else {
      VLOG(1) << "[on_new_client] Channel mismatch or missing for " << url_key;
    }
    return misc::SUCCESS;
  });
  return misc::SUCCESS;
}

absl::StatusOr<channel_t> Communicator::do_create_channel(const std::string& ip, uint16_t port) {
  absl::MutexLock lock(&create_channel_mu_);

  // Fast-path: if another thread has already created the channel, reuse it
  std::stringstream url_ss;
  url_ss << ip << ":" << port;
  const std::string url_key = url_ss.str();

  LOG(INFO) << "[do_create_channel] Attempting to create channel for " << url_key;

  if (channels_.exist(url_key)) {
    LOG(INFO) << "[do_create_channel] Channel already exists for " << url_key << ", reusing";
    return channels_.get(url_key);
  }

  LOG(INFO) << "create a channel: dst=" << ip << ":" << port;
  auto t = client_context_->connect(ip, port);
  if (!t.ok()) {
    LOG(WARNING) << "failed to connect peer " << ip << ":" << port;
    return absl::InternalError(t.status().message());
  }

  auto transport = *t;
  auto channel = std::make_shared<Channel>(
      transport, enable_rdma_ ? CHANNEL_RDMA : CHANNEL_MTCP, buffers_per_flow_, max_window_segments_);

  VLOG(1) << "[Communicator] Control channel connected: local=" << server_context_->get_local_ip() << ":" << port
          << " remote=" << ip << ":" << port << " fd=" << transport->get_fd();

  transport->set_recv_func([this](const tcp_transport_t& t) {
    ProtoHeader header = {};
    auto channel = this->channels_.get(t->get_remote_url());
    COMM_CHECK(t->recv<ProtoHeader>(&header));
    auto msg = std::make_shared<EngineMessage>(&header);
    COMM_CHECK(t->recv(msg->get_payload<uint8_t>(), msg->get_payload_size()));
    return this->on_receive_response(channel, t, msg);
  });
  transport->set_close_func([this, transport_ptr = transport.get()](const tcp_transport_t& t) {
    const std::string url_key = t->get_remote_url();
    LOG(INFO) << "[do_create_channel] TCP connection closed for " << url_key << ", transport ptr: " << t.get() << " vs "
              << transport_ptr;
    auto channel = channels_.get(url_key);
    if (channel && channel->get_control().get() == t.get()) {
      // Only remove the channel if this is the actual control connection
      LOG(INFO) << "[do_create_channel] This is the control connection, removing channel";
      channel->close();
      if (!channels_.erase_if(url_key, channel)) {
        VLOG(1) << "[do_create_channel] Channel already removed or replaced for " << url_key;
      }
    } else {
      LOG(INFO) << "[do_create_channel] This is not the control connection, keeping channel";
    }
    return SUCCESS;
  });
  if (channel_expire_ > 0) {
    channel->record_expire(channel_expire_);
  }
  // Only insert if still absent to avoid clobbering an existing active channel
  if (!channels_.exist(url_key)) {
    channels_.put(url_key, channel);
    LOG(INFO) << "[do_create_channel] Channel created and stored for " << url_key;
  } else {
    // Another thread beat us – use that channel, close the one we just created
    LOG(INFO) << "[do_create_channel] Another thread already created channel for " << url_key
              << ", closing duplicate transport (fd=" << transport->get_fd() << ")";
    channel_t existing = channels_.get(url_key);
    // Close just the transport, not the channel
    transport->close();
    return existing;
  }

  VLOG(1) << "[Communicator] Channel stored: " << transport->get_remote_url();
  return channel;
}

void Communicator::do_read_request_loop() {
  while (!stop_.load()) {
    auto req = request_queue_.pop(true);
    if (stop_.load()) {
      break;
    }
    if (req == nullptr) {
      continue;
    }

    auto channel = channels_.get(req->get_dst_url());
    if (channel == nullptr) {
      VLOG(1) << "[do_read_request_loop] No existing channel for " << req->get_dst_url() << ", creating new channel";
      auto status = do_create_channel(req->dst_ip_, req->dst_port_);
      if (!status.ok()) {
        LOG(WARNING) << "failed to create channel " << req->dst_ip_ << ":" << req->dst_port_;
        req->set_result(absl::InternalError(status.status().message()));
        continue;
      }
      channel = *status;
    } else {
      VLOG(1) << "[do_read_request_loop] Using existing channel for " << req->get_dst_url();
    }

    auto transport = channel->get_control();
    if (transport == nullptr) {
      req->set_result(absl::InternalError("failed to get transport control"));
      LOG(WARNING) << "failed to get control transport " << req->dst_ip_ << ":" << req->dst_port_;
      continue;
    }

    std::shared_ptr<EngineMessage> msg;
    if (req->is_read_plan()) {
      auto prepared = req->get_prepared_read_plan();
      if (prepared == nullptr) {
        req->set_result(absl::InternalError("read_plan request missing prepared plan"));
        continue;
      }
      if (prepared->logical_plan.source_slices.size() > std::numeric_limits<uint32_t>::max()) {
        req->set_result(absl::InvalidArgumentError("read_plan source slice count exceeds wire limit"));
        continue;
      }
      const uint32_t payload_size = static_cast<uint32_t>(
          sizeof(ProtoReadPlanRequestHeader) +
          prepared->logical_plan.source_slices.size() * sizeof(ProtoReadPlanSourceSlice));
      msg = std::make_shared<EngineMessage>(ENGINE_OP_READ_PLAN_REQUEST, payload_size);
      auto* header = msg->get_payload<ProtoReadPlanRequestHeader>();
      header->transport_type = ENGINE_TRANSPORT_RDMA;
      header->rail_id = req->get_rail_id();
      header->request_id = req->request_id();
      header->num_source_slices = static_cast<uint32_t>(prepared->logical_plan.source_slices.size());
      auto* source_payload = reinterpret_cast<ProtoReadPlanSourceSlice*>(
          reinterpret_cast<uint8_t*>(header) + sizeof(ProtoReadPlanRequestHeader));
      for (size_t index = 0; index < prepared->logical_plan.source_slices.size(); ++index) {
        const auto& source_slice = prepared->logical_plan.source_slices[index];
        misc::STRNCPY(source_payload[index].tensor_key, source_slice.tensor_key, kMaxTensorNameLen);
        source_payload[index].source_slice_index = static_cast<uint32_t>(index);
        source_payload[index].remote_offset = source_slice.remote_offset;
        source_payload[index].bytes = source_slice.bytes;
      }
      VLOG(1) << "[do_read_request_loop] Sending READ_PLAN_REQUEST: key=" << req->get_key() << " to "
              << req->get_dst_url() << " source_slices=" << prepared->logical_plan.source_slices.size();
    } else {
      msg = EngineMessage::make_message<ProtoReadRequest>(ENGINE_OP_READ_REQUEST);
      auto* request = msg->get_payload<ProtoReadRequest>();
      misc::STRNCPY(request->tensor_key, req->tensor_key_, kMaxTensorNameLen);

      request->transport_type = enable_rdma_ ? ENGINE_TRANSPORT_RDMA : ENGINE_TRANSPORT_MTCP;
      request->offset = req->remote_offset_;
      request->bytes = req->get_local_tensor()->get_bytes();
      request->request_id = req->request_id();
      request->rail_id = req->get_rail_id();

      VLOG(1) << "[do_read_request_loop] Sending READ_REQUEST: key=" << req->tensor_key_ << " to " << req->get_dst_url()
              << " transport_type=" << (request->transport_type == ENGINE_TRANSPORT_MTCP ? "MTCP" : "RDMA");
    }

    const std::string req_key = req->get_key();
    auto existing = pending_requests_.get(req_key);
    if (existing != nullptr) {
      if (!existing->is_result_set()) {
        LOG(ERROR) << "[do_read_request_loop] duplicate in-flight READ_REQUEST key=" << req_key;
        req->set_result(absl::AlreadyExistsError("duplicate in-flight read request key"));
        continue;
      }
      LOG(WARNING) << "[do_read_request_loop] replacing stale completed pending request key=" << req_key;
      pending_requests_.erase_if(req_key, existing);
    }

    std::weak_ptr<transport::ReadRequest> weak_req = req;
    req->set_on_result([this, req_key, weak_req]() {
      auto locked = weak_req.lock();
      if (locked == nullptr) {
        return;
      }
      pending_requests_.erase_if(req_key, locked);
    });

    // Put into pending BEFORE send to prevent response racing ahead of insertion
    pending_requests_.put(req_key, req);

    if (transport->send(msg) == SUCCESS) {
      LOG(INFO) << "[do_read_request_loop] READ_REQUEST sent successfully, pending: " << req_key;
    } else {
      // Rollback pending on failure
      pending_requests_.del(req_key);
      LOG(ERROR) << "[do_read_request_loop] Failed to send READ_REQUEST for key=" << req->tensor_key_ << " to "
                 << req->get_dst_url();
      req->set_result(absl::InternalError("failed to send request"));
    }

    if (channel_expire_ > 0) {
      channel->record_expire(channel_expire_);
    }
  }
}

misc::result_t Communicator::on_receive_request(
    const channel_t& channel,
    const tcp_transport_t& t,
    const engine_message_t& msg) {
  static std::atomic<int> server_requests_received(0);
  auto send_read_plan_failed = [&](uint64_t request_id, uint32_t reason) {
    auto rsp = EngineMessage::make_message<ProtoReadPlanFailed>(ENGINE_OP_READ_PLAN_FAILED);
    auto* payload = rsp->get_payload<ProtoReadPlanFailed>();
    payload->request_id = request_id;
    payload->reason = reason;
    return t->send(rsp);
  };

  switch (msg->get_op()) {
    case ENGINE_OP_RDMA_CONNECT_REQUEST: {
      auto* req = msg->get_payload<ProtoRdmaConnectRequest>();
      auto local_dev_name = std::string(req->dst_dev_name);
      auto peer_dev_name = std::string(req->src_dev_name);
      LOG(INFO) << "recv rdma connect from " << t->get_remote_url() << ": net_dev=" << local_dev_name;

      CHECK(rdma_context_ != nullptr) << "rdma context is not initialized";
      auto transport = rdma_context_->create_transport(local_dev_name);

      if (transport != nullptr && transport->connect(&req->qp_info) == misc::SUCCESS) {
        channel->set_transport(local_dev_name, peer_dev_name, transport);
        auto rsp = EngineMessage::make_message<ProtoRdmaConnectResponse>(ENGINE_OP_RDMA_CONNECT_RESPONSE);
        auto* payload = rsp->get_payload<ProtoRdmaConnectResponse>();
        COMM_CHECK(transport->get_local_info(&payload->qp_info));

        memcpy(payload->src_dev_name, req->src_dev_name, kMaxDevName);
        memcpy(payload->dst_dev_name, req->dst_dev_name, kMaxDevName);
        COMM_CHECK(t->send(rsp));
      } else {
        if (transport == nullptr) {
          LOG(WARNING) << "failed to create rdma transport from " << t->get_remote_url()
                       << ": net_dev=" << local_dev_name;
        } else {
          LOG(WARNING) << "failed to rdma connect from " << t->get_remote_url() << ": net_dev=" << local_dev_name;
        }
        auto rsp = EngineMessage::make_message<ProtoRdmaConnectFailed>(ENGINE_OP_RDMA_CONNECT_FAILED);
        auto* payload = rsp->get_payload<ProtoRdmaConnectFailed>();
        memcpy(payload->src_dev_name, req->src_dev_name, kMaxDevName);
        memcpy(payload->dst_dev_name, req->dst_dev_name, kMaxDevName);
        COMM_CHECK(t->send(rsp));
      }
      break;
    }
    case ENGINE_OP_MTCP_CONNECT_REQUEST: {
      auto* req = msg->get_payload<ProtoMtcpConnectRequest>();
      LOG(INFO) << "recv mtcp connect from " << t->get_remote_url();

      auto transport = channel->get_mtcp();
      transport->set_conn_count(std::min(mtcp_conn_count_, req->conn_count));

      std::string ip = server_context_->get_local_ip();
      uint16_t port = 0;
      if (transport->listen(ip, &port) == misc::SUCCESS) {
        auto rsp = EngineMessage::make_message<ProtoMtcpConnectResponse>(ENGINE_OP_MTCP_CONNECT_RESPONSE);
        auto* payload = rsp->get_payload<ProtoMtcpConnectResponse>();
        payload->conn_count = std::min(mtcp_conn_count_, req->conn_count);
        payload->port = port;
        auto ip_addr = inet_addr(ip.c_str());
        payload->ip = ntohl(ip_addr);
        COMM_CHECK(t->send(rsp));
      } else {
        LOG(WARNING) << "failed to create mtcp transport: source=" << t->get_remote_url();
        auto rsp = EngineMessage::make_message<ProtoMtcpConnectFailed>(ENGINE_OP_MTCP_CONNECT_FAILED);
        auto* payload = rsp->get_payload<ProtoMtcpConnectFailed>();
        payload->ip = inet_addr(ip.c_str());
        COMM_CHECK(t->send(rsp));
      }
      break;
    }
    case ENGINE_OP_READ_REQUEST: {
      auto* req = msg->get_payload<ProtoReadRequest>();
      auto tensor_key = std::string(req->tensor_key);

      int request_num = ++server_requests_received;
      LOG(INFO) << "[on_receive_request] Server received READ_REQUEST #" << request_num << " from "
                << t->get_remote_url() << ": key=" << tensor_key
                << ", transport=" << (req->transport_type == ENGINE_TRANSPORT_MTCP ? "mtcp" : "rdma")
                << ", offset=" << req->offset << ", size=" << req->bytes;

      LOG(INFO) << "read request from " << t->get_remote_url() << ": key=" << tensor_key
                << ", transport=" << (req->transport_type == ENGINE_TRANSPORT_MTCP ? "mtcp" : "rdma")
                << ", offset=" << req->offset << ", size=" << req->bytes;

      auto tensor = store_.get_tensor(tensor_key);
      if (tensor == nullptr) {
        auto rsp = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
        auto* payload = rsp->get_payload<ProtoReadFailed>();
        memcpy(payload->tensor_key, req->tensor_key, 512);
        payload->offset = req->offset;
        payload->request_id = req->request_id;
        payload->reason = TENSORCAST_READ_FAILED_NO_TENSOR;
        COMM_CHECK(t->send(rsp));
      } else if (req->offset + req->bytes > tensor->get_bytes()) {
        LOG(ERROR) << "[on_receive_request] READ_REQUEST overflow: key=" << tensor_key
                   << " tensor_bytes=" << tensor->get_bytes() << " offset=" << req->offset << " size=" << req->bytes;
        auto rsp = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
        auto* payload = rsp->get_payload<ProtoReadFailed>();
        memcpy(payload->tensor_key, req->tensor_key, 512);
        payload->offset = req->offset;
        payload->request_id = req->request_id;
        payload->reason = TENSORCAST_READ_FAILED_OVERFLOW;
        COMM_CHECK(t->send(rsp));
      } else {
        auto read_guard_or = acquire_tensor_read_lease(tensor_key);
        if (!read_guard_or.ok()) {
          auto rsp = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
          auto* payload = rsp->get_payload<ProtoReadFailed>();
          memcpy(payload->tensor_key, req->tensor_key, 512);
          payload->offset = req->offset;
          payload->request_id = req->request_id;
          payload->reason = TENSORCAST_READ_FAILED_NO_TENSOR;
          COMM_CHECK(t->send(rsp));
          LOG(WARNING) << "Read request rejected for retiring tensor key=" << tensor_key
                       << " peer=" << t->get_remote_url() << " status=" << read_guard_or.status();
          break;
        }

        std::shared_ptr<void> read_guard = std::move(*read_guard_or);
        // Build response depending on transport type
        if (enable_rdma_ && req->transport_type == ENGINE_TRANSPORT_RDMA) {
          auto status = handle_rdma_read_request(channel, t, *req, tensor, std::move(read_guard));
          if (!status.ok()) {
            auto failure = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
            auto* payload = failure->get_payload<ProtoReadFailed>();
            memcpy(payload->tensor_key, req->tensor_key, kMaxTensorNameLen);
            payload->offset = req->offset;
            payload->request_id = req->request_id;
            payload->reason = ReadFailedReasonFromStatus(status);
            COMM_CHECK(t->send(failure));
            LOG(WARNING) << "RDMA read request failed: " << status;
            break;
          }
        } else {
          auto status = handle_mtcp_read_request(channel, t, *req, tensor, std::move(read_guard));
          if (!status.ok()) {
            auto failure = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
            auto* payload = failure->get_payload<ProtoReadFailed>();
            memcpy(payload->tensor_key, req->tensor_key, kMaxTensorNameLen);
            payload->offset = req->offset;
            payload->request_id = req->request_id;
            payload->reason = ReadFailedReasonFromStatus(status);
            COMM_CHECK(t->send(failure));
            LOG(WARNING) << "MTCP read request failed: " << status;
            break;
          }
        }
      }
      break;
    }
    case ENGINE_OP_READ_PLAN_REQUEST: {
      auto* req = msg->get_payload<ProtoReadPlanRequestHeader>();
      if (msg->get_payload_size() < sizeof(ProtoReadPlanRequestHeader)) {
        LOG(WARNING) << "READ_PLAN_REQUEST payload too small";
        break;
      }
      const size_t expected_size = sizeof(ProtoReadPlanRequestHeader) +
          static_cast<size_t>(req->num_source_slices) * sizeof(ProtoReadPlanSourceSlice);
      if (msg->get_payload_size() < expected_size) {
        LOG(WARNING) << "READ_PLAN_REQUEST payload truncated: expected=" << expected_size
                     << " actual=" << msg->get_payload_size();
        send_read_plan_failed(req->request_id, TENSORCAST_READ_FAILED_MEM_MISMATCH);
        break;
      }
      if (!enable_rdma_ || req->transport_type != ENGINE_TRANSPORT_RDMA) {
        send_read_plan_failed(req->request_id, TENSORCAST_READ_FAILED_MEM_MISMATCH);
        break;
      }
      auto flow_state = channel->flow_state();
      if (!flow_state) {
        send_read_plan_failed(req->request_id, TENSORCAST_READ_FAILED_MEM_MISMATCH);
        break;
      }

      auto* source_payload = reinterpret_cast<ProtoReadPlanSourceSlice*>(
          reinterpret_cast<uint8_t*>(req) + sizeof(ProtoReadPlanRequestHeader));
      auto read_guards = std::make_shared<std::vector<std::shared_ptr<void>>>();
      auto session = std::make_shared<RdmaReadSession>();
      session->mode = RdmaReadSession::Mode::kReadPlan;
      session->plan_request = *req;
      session->request_key = transport::get_read_plan_request_key(req->request_id);
      session->control_transport = t;
      session->source_stage_profile = std::make_shared<RdmaSourceStageProfile>();

      struct PlannedSourceSlice {
        uint32_t source_slice_index = 0;
        std::string tensor_key;
        uint64_t remote_offset = 0;
        uint64_t bytes = 0;
        std::shared_ptr<PartitionTensor> tensor;
        std::shared_ptr<MemoryStager> stager;
        net_dev_t dev;
        v1::RdmaConfig::StagedRdmaBackend staged_backend = v1::RdmaConfig::STAGED_RDMA_BACKEND_HOST_PINNED;
        bool direct_eligible = false;
        ibv_mr* direct_mr = nullptr;
      };

      std::vector<PlannedSourceSlice> resolved_slices;
      resolved_slices.reserve(req->num_source_slices);
      uint64_t total_bytes = 0;
      bool failed = false;
      absl::Status failure_status = absl::OkStatus();

      for (uint32_t index = 0; index < req->num_source_slices; ++index) {
        const auto& source = source_payload[index];
        const std::string tensor_key = reinterpret_cast<const char*>(source.tensor_key);
        auto tensor = store_.get_tensor(tensor_key);
        if (tensor == nullptr) {
          failed = true;
          failure_status = absl::NotFoundError(absl::StrCat("read_plan tensor not found: ", tensor_key));
          break;
        }
        if (source.remote_offset > tensor->get_bytes() || source.bytes > tensor->get_bytes() - source.remote_offset) {
          failed = true;
          failure_status = absl::OutOfRangeError(absl::StrCat("read_plan slice out of range: ", tensor_key));
          break;
        }
        auto read_guard_or = acquire_tensor_read_lease(tensor_key);
        if (!read_guard_or.ok()) {
          failed = true;
          failure_status = read_guard_or.status();
          break;
        }
        read_guards->push_back(*read_guard_or);

        tensor->wait_read_ready();
        auto dev = req->rail_id >= 0 ? tensor->get_dev_by_rail(req->rail_id) : nullptr;
        if (dev == nullptr) {
          // The requester encodes its own local rail in READ_PLAN_REQUEST.
          // Source-side tensors must keep their local preferred NIC; otherwise
          // routed read-plan becomes stricter than legacy READ_REQUEST and
          // spuriously falls back to scalar reads when rail ids differ across
          // hosts.
          dev = tensor->get_dev();
          if (dev != nullptr && req->rail_id >= 0) {
            VLOG(2) << "READ_PLAN_REQUEST rail fallback tensor=" << tensor_key << " requested_rail=" << req->rail_id
                    << " selected_rail=" << dev->get_rail_id() << " nic=" << dev->get_name();
          }
        }
        const bool tensor_on_cpu = tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_CPU;
        if (tensor_on_cpu && session->dev != nullptr &&
            (dev == nullptr || session->dev->get_name() != dev->get_name() ||
             session->dev->get_rail_id() != dev->get_rail_id())) {
          VLOG(2) << "READ_PLAN_REQUEST cpu source rebinding tensor=" << tensor_key
                  << " selected_rail=" << (dev != nullptr ? dev->get_rail_id() : -1)
                  << " selected_nic=" << (dev != nullptr ? dev->get_name() : "<none>")
                  << " session_rail=" << session->dev->get_rail_id() << " session_nic=" << session->dev->get_name();
          dev = session->dev;
        }
        if (dev == nullptr) {
          failed = true;
          failure_status =
              absl::FailedPreconditionError(absl::StrCat("read_plan tensor missing RDMA device: ", tensor_key));
          break;
        }
        if (session->dev == nullptr) {
          session->dev = dev;
        } else if (session->dev->get_name() != dev->get_name() || session->dev->get_rail_id() != dev->get_rail_id()) {
          failed = true;
          failure_status =
              absl::FailedPreconditionError("read_plan source slices must resolve to one RDMA device/rail");
          break;
        }

        const int device_id = tensor->get_device_id();
        std::shared_ptr<MemoryStager> stager;
        if (tensor_on_cpu) {
          stager = get_cpu_stager_for_nic(dev->get_name());
        } else if (use_gpu_vram_staging_) {
          stager = get_gpu_vram_stager_for_id(device_id);
        } else {
          stager = get_gpu_mem_stager_for_id(device_id);
        }
        if (!stager) {
          if (tensor_on_cpu) {
            stager = memory_stager_;
          } else if (use_gpu_vram_staging_) {
            stager = get_gpu_mem_stager_for_id(device_id);
            if (!stager) {
              stager = gpu_memory_stager_;
            }
          } else {
            stager = gpu_memory_stager_;
          }
        }

        const bool direct_requested = tensor->direct_rdma_enabled();
        bool direct_eligible = false;
        ibv_mr* direct_mr = nullptr;
        DirectFallbackReason fallback_reason = DirectFallbackReason::kNone;
        if (direct_requested) {
          const bool direct_mem_supported = tensor_on_cpu || tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_GPU;
          if (!direct_mem_supported) {
            fallback_reason = DirectFallbackReason::kNotGpu;
          } else if (tensor->needs_staging()) {
            fallback_reason = DirectFallbackReason::kNeedsStaging;
          } else {
            if (tensor_on_cpu) {
              auto ensure_status = ensure_tensor_registered_on_dev(tensor, dev);
              if (!ensure_status.ok()) {
                fallback_reason = DirectFallbackReason::kMrUnavailable;
              } else {
                direct_mr = tensor->get_mr(dev);
                direct_eligible = direct_mr != nullptr && tensor->has_registered_mr(dev);
                if (!direct_eligible) {
                  fallback_reason = DirectFallbackReason::kMrUnavailable;
                }
              }
            } else {
              tensor->wait_mr_ready(dev);
              direct_mr = tensor->get_mr(dev);
              if (direct_mr == nullptr || !tensor->has_registered_mr(dev)) {
                fallback_reason = DirectFallbackReason::kMrUnavailable;
              } else {
                direct_eligible = true;
              }
            }
          }
          if (!direct_eligible && tensor->direct_rdma_required()) {
            failed = true;
            failure_status = absl::FailedPreconditionError(
                absl::StrCat("read_plan direct RDMA required but unavailable: ", tensor_key));
            break;
          }
          if (!direct_eligible && fallback_reason != DirectFallbackReason::kNone) {
            record_direct_fallback_metric(fallback_reason);
          }
        }
        if (!direct_eligible && !stager) {
          failed = true;
          failure_status = absl::FailedPreconditionError("no staging backend available for read_plan source slice");
          break;
        }

        const bool using_gpu_vram_stager =
            !tensor_on_cpu && use_gpu_vram_staging_ && stager && stager == get_gpu_vram_stager_for_id(device_id);
        const v1::RdmaConfig::StagedRdmaBackend staged_backend = using_gpu_vram_stager
            ? v1::RdmaConfig::STAGED_RDMA_BACKEND_GPU_VRAM
            : v1::RdmaConfig::STAGED_RDMA_BACKEND_HOST_PINNED;
        resolved_slices.push_back(
            PlannedSourceSlice{
                .source_slice_index = source.source_slice_index,
                .tensor_key = tensor_key,
                .remote_offset = source.remote_offset,
                .bytes = source.bytes,
                .tensor = tensor,
                .stager = stager,
                .dev = dev,
                .staged_backend = staged_backend,
                .direct_eligible = direct_eligible,
                .direct_mr = direct_mr,
            });
        if (source.bytes > std::numeric_limits<uint64_t>::max() - total_bytes) {
          failed = true;
          failure_status = absl::InvalidArgumentError("read_plan source byte count overflow");
          break;
        }
        total_bytes += source.bytes;
      }

      if (!failed) {
        bool direct_source_response = !resolved_slices.empty();
        uint64_t direct_source_segment_count = 0;
        for (const auto& source : resolved_slices) {
          if (source.tensor->get_mem_type() != COMMUNICATE_ENGINE_DEV_CPU || !source.direct_eligible) {
            direct_source_response = false;
            break;
          }
          const uint64_t chunk_size = compute_read_plan_direct_chunk_bytes(source.bytes);
          const uint64_t source_segments = (source.bytes + chunk_size - 1) / chunk_size;
          if (source_segments > std::numeric_limits<uint64_t>::max() - direct_source_segment_count) {
            failed = true;
            failure_status = absl::InvalidArgumentError("read_plan direct source segment count overflow");
            break;
          }
          direct_source_segment_count += source_segments;
        }
        if (!failed) {
          if (direct_source_response) {
            session->direct_source_response = true;
            session->read_plan_window_segment_limit =
                compute_read_plan_direct_window_segment_limit(direct_source_segment_count);
            session->direct_ledger =
                std::make_unique<FlowCreditLedger>(static_cast<int>(session->read_plan_window_segment_limit));
          }
          FlowCreditLedger* plan_ledger =
              session->direct_source_response ? session->direct_ledger.get() : &flow_state->ledger;
          CHECK(plan_ledger != nullptr);
          for (const auto& source : resolved_slices) {
            const bool use_direct = session->direct_source_response
                ? source.direct_eligible
                : (source.tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_GPU && source.direct_eligible);
            const uint64_t chunk_size = use_direct
                ? (session->direct_source_response ? compute_read_plan_direct_chunk_bytes(source.bytes)
                                                   : compute_direct_chunk_bytes(
                                                         source.bytes,
                                                         direct_rdma_chunk_bytes_,
                                                         flow_state->max_window_segments,
                                                         config_.rdma().qp_count()))
                : (source.stager && source.stager->get_chunk_size() > 0 ? source.stager->get_chunk_size()
                                                                        : source.bytes);
            session->plan_source_slices.push_back(
                RdmaReadPlanSourceSlice{
                    .source_slice_index = source.source_slice_index,
                    .tensor_key = source.tensor_key,
                    .remote_offset = source.remote_offset,
                    .bytes = source.bytes,
                    .tensor = source.tensor,
                    .stager = source.stager,
                    .dev = source.dev,
                    .stage_fn = MakeStageFunction(
                        source.tensor,
                        plan_ledger,
                        source.stager,
                        source.dev,
                        meta_mr_cache_.get(),
                        source.tensor_key,
                        session->request_key,
                        source.staged_backend,
                        use_direct,
                        source.direct_mr,
                        session->source_stage_profile),
                    .chunk_size = chunk_size,
                    .zero_copy = use_direct,
                });
          }
        }
      }

      if (failed) {
        send_read_plan_failed(req->request_id, ReadFailedReasonFromStatus(failure_status));
        LOG(WARNING) << "READ_PLAN_REQUEST rejected: " << failure_status;
        break;
      }

      session->read_guard = read_guards;
      const std::string peer = t ? t->get_remote_url() : std::string();
      session->transfer_id = make_transfer_id(session->request_key, peer);
      (void)register_source_transfer_progress(session->request_key, peer, "rdma", total_bytes, session->read_guard);

      size_t pending_reads = 0;
      {
        absl::MutexLock lock(&flow_state->rdma_pending_reads_mu);
        flow_state->rdma_pending_reads.push_back(session);
        pending_reads = flow_state->rdma_pending_reads.size();
      }
      LOG(INFO) << "[rdma_plan_session] queued request=" << session->request_key
                << " source_slices=" << session->plan_source_slices.size() << " pending_reads=" << pending_reads;

      auto status = resume_rdma_reads(channel);
      if (!status.ok()) {
        finish_source_transfer_progress(session->transfer_id, status);
        send_read_plan_failed(req->request_id, ReadFailedReasonFromStatus(status));
        LOG(WARNING) << "READ_PLAN_REQUEST failed to resume session: " << status;
      }
      break;
    }
    case ENGINE_OP_RDMA_READ_DONE_EX: {
      auto* hdr = msg->get_payload<ProtoRdmaReadDoneExHeader>();
      const std::string tensor_key = reinterpret_cast<char*>(hdr->tensor_key);
      const std::string peer = t ? t->get_remote_url() : std::string();
      auto flow_state = channel->flow_state();
      if (!flow_state) {
        LOG(WARNING) << "RDMA_READ_DONE_EX without channel flow state";
        break;
      }
      for (uint32_t i = 0; i < hdr->num_segments; ++i) {
        StageLeaseKey key{
            .request_key = transport::get_request_instance_key(tensor_key, hdr->request_offset, hdr->request_id),
            .window_seq = hdr->window_seq,
            .segment_idx = i,
        };
        auto lease_or = flow_state->registry.take(key);
        if (!lease_or.ok()) {
          LOG(WARNING) << "RDMA_READ_DONE_EX for unknown lease: key=" << key.request_key
                       << " window=" << hdr->window_seq << " segment=" << i;
          continue;
        }
        const uint64_t bytes = lease_or->bytes();
        const std::string transfer_id = make_transfer_id(key.request_key, peer);
        auto progress = lookup_source_transfer_progress(transfer_id);
        if (progress && bytes > 0) {
          const uint64_t done = add_transfer_progress_bytes(progress, bytes);
          if (done >= progress->total_bytes) {
            finish_source_transfer_progress(transfer_id, absl::OkStatus());
          }
        }
        lease_or->release();
      }
      auto resume_status = resume_rdma_reads(channel);
      if (!resume_status.ok()) {
        LOG(WARNING) << "Failed to resume RDMA staging after ACK: " << resume_status;
      }
      break;
    }
    case ENGINE_OP_RDMA_READ_PLAN_DONE_EX: {
      auto* hdr = msg->get_payload<ProtoRdmaReadPlanDoneExHeader>();
      const std::string peer = t ? t->get_remote_url() : std::string();
      auto flow_state = channel->flow_state();
      if (!flow_state) {
        LOG(WARNING) << "RDMA_READ_PLAN_DONE_EX without channel flow state";
        break;
      }
      const std::string request_key = transport::get_read_plan_request_key(hdr->request_id);
      for (uint32_t i = 0; i < hdr->num_segments; ++i) {
        StageLeaseKey key{
            .request_key = request_key,
            .window_seq = hdr->window_seq,
            .segment_idx = i,
        };
        auto lease_or = flow_state->registry.take(key);
        if (!lease_or.ok()) {
          LOG(WARNING) << "RDMA_READ_PLAN_DONE_EX for unknown lease: key=" << key.request_key
                       << " window=" << hdr->window_seq << " segment=" << i;
          continue;
        }
        const uint64_t bytes = lease_or->bytes();
        const std::string transfer_id = make_transfer_id(request_key, peer);
        auto progress = lookup_source_transfer_progress(transfer_id);
        if (progress && bytes > 0) {
          const uint64_t done = add_transfer_progress_bytes(progress, bytes);
          if (done >= progress->total_bytes) {
            finish_source_transfer_progress(transfer_id, absl::OkStatus());
          }
        }
        lease_or->release();
      }
      auto resume_status = resume_rdma_reads(channel);
      if (!resume_status.ok()) {
        LOG(WARNING) << "Failed to resume RDMA read_plan staging after ACK: " << resume_status;
      }
      break;
    }
    default:
      LOG(WARNING) << "failed to process request: " << msg->get_op();
      return misc::FAILED;
  }
  return misc::SUCCESS;
}

misc::result_t Communicator::on_receive_response(
    const channel_t& channel,
    const tcp_transport_t& t,
    const engine_message_t& msg) {
  LOG(INFO) << "[on_receive_response] Received response op=" << msg->get_op() << " from " << t->get_remote_url();

  auto handle_rdma_connect_failure =
      [&](const std::string& local_dev_name, const std::string& peer_dev_name, const char* failure_reason) {
        LOG(ERROR) << "[on_receive_response] RDMA_CONNECT_FAILED: local=" << local_dev_name << " peer=" << peer_dev_name
                   << " reason=" << failure_reason;

        auto endpoint = channel->get_rdma_endpoint(local_dev_name, peer_dev_name);
        if (endpoint == nullptr) {
          return;
        }

        uint64_t generation = 0;
        Channel::HandshakeState from_state = Channel::HandshakeState::kIdle;
        {
          absl::MutexLock lock(&endpoint->mu);
          generation = endpoint->generation;
          from_state = endpoint->state;
        }

        auto failed_reads = drain_pending_reads_for_generation(endpoint, generation);
        const absl::Status status = absl::UnavailableError(failure_reason);
        for (auto& pending : failed_reads) {
          pending_requests_.erase_if_present(pending.request->get_key());
          pending.request->set_result(status);
        }

        {
          absl::MutexLock lock(&endpoint->mu);
          log_handshake_transition(
              local_dev_name,
              peer_dev_name,
              from_state,
              Channel::HandshakeState::kFailed,
              endpoint->generation,
              endpoint->pending_reads.size());
          endpoint->state = Channel::HandshakeState::kFailed;
          endpoint->transport.reset();
          endpoint->failure_count += 1;
          endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
          endpoint->retry_scheduled = false;
        }
      };
  auto handle_target_rdma_reads = [&](const std::string& req_key,
                                      const std::shared_ptr<transport::ReadRequest>& read_request,
                                      const std::string& peer_dev_name,
                                      std::vector<transport::RdmaTransport::RdmaReadSeg> rdma_segs) {
    CHECK(rdma_context_ != nullptr) << "rdma context is not initialized";

    std::string local_dev_name;
    if (read_request->is_read_plan()) {
      auto prepared = read_request->get_prepared_read_plan();
      if (prepared == nullptr || prepared->local_nic.empty()) {
        read_request->set_result(absl::FailedPreconditionError("read_plan request missing prepared local nic"));
        pending_requests_.erase_if_present(req_key);
        return;
      }
      local_dev_name = prepared->local_nic;
    } else {
      auto tensor = read_request->get_local_tensor();
      if (tensor == nullptr || tensor->get_dev() == nullptr) {
        read_request->set_result(absl::FailedPreconditionError("read request missing local tensor device metadata"));
        pending_requests_.erase_if_present(req_key);
        return;
      }
      local_dev_name = tensor->get_dev()->get_name();
    }

    auto endpoint = channel->ensure_rdma_endpoint(local_dev_name, peer_dev_name);
    auto now = absl::Now();
    transport::rdma_transport_t transport_to_use;
    transport::rdma_transport_t prepared_transport;
    std::shared_ptr<EngineMessage> connect_request_msg;
    bool issue_now = false;
    bool handshake_started = false;
    bool queued_current = false;
    bool deferred_for_backoff = false;
    bool schedule_retry = false;
    absl::Duration backoff_remaining = absl::ZeroDuration();
    absl::Status immediate_failure = absl::OkStatus();
    uint64_t generation = 0;

    while (true) {
      endpoint->mu.Lock();
      auto state = endpoint->state;
      if (state == Channel::HandshakeState::kReady) {
        transport_to_use = endpoint->transport;
        if (transport_to_use == nullptr || !transport_to_use->ready()) {
          log_handshake_transition(
              local_dev_name,
              peer_dev_name,
              state,
              Channel::HandshakeState::kIdle,
              endpoint->generation,
              endpoint->pending_reads.size());
          endpoint->state = Channel::HandshakeState::kIdle;
          endpoint->transport.reset();
          endpoint->mu.Unlock();
          continue;
        }
        generation = endpoint->generation;
        endpoint->mu.Unlock();
        issue_now = true;
        break;
      }

      if (state == Channel::HandshakeState::kConnectRequested) {
        generation = endpoint->generation;
        endpoint->pending_reads.push_back(
            Channel::PendingRdmaRead{
                .request = read_request,
                .segments = std::move(rdma_segs),
                .enqueued_at = now,
                .generation = generation,
            });
        queued_current = true;
        endpoint->mu.Unlock();
        break;
      }

      if (state == Channel::HandshakeState::kIdle || state == Channel::HandshakeState::kFailed) {
        const bool can_retry = state == Channel::HandshakeState::kIdle || now >= endpoint->next_retry_at;
        if (!can_retry) {
          generation = endpoint->generation;
          endpoint->pending_reads.push_back(
              Channel::PendingRdmaRead{
                  .request = read_request,
                  .segments = std::move(rdma_segs),
                  .enqueued_at = now,
                  .generation = generation,
              });
          queued_current = true;
          deferred_for_backoff = true;
          backoff_remaining = endpoint->next_retry_at - now;
          if (!endpoint->retry_scheduled) {
            endpoint->retry_scheduled = true;
            schedule_retry = true;
          }
          endpoint->mu.Unlock();
          break;
        }

        if (prepared_transport == nullptr) {
          endpoint->mu.Unlock();
          prepared_transport = rdma_context_->create_transport(local_dev_name);
          if (prepared_transport == nullptr) {
            immediate_failure = absl::InternalError("failed to allocate RDMA transport");
            break;
          }
          connect_request_msg = EngineMessage::make_message<ProtoRdmaConnectRequest>(ENGINE_OP_RDMA_CONNECT_REQUEST);
          auto* payload = connect_request_msg->get_payload<ProtoRdmaConnectRequest>();
          misc::result_t info_res = prepared_transport->get_local_info(&payload->qp_info);
          if (info_res != misc::SUCCESS) {
            immediate_failure = absl::InternalError("failed to prepare RDMA connect info");
            prepared_transport.reset();
            connect_request_msg.reset();
            break;
          }
          misc::STRNCPY(payload->src_dev_name, local_dev_name, kMaxDevName);
          misc::STRNCPY(payload->dst_dev_name, peer_dev_name, kMaxDevName);
          continue;
        }

        Channel::HandshakeState from_state = state;
        endpoint->transport = prepared_transport;
        endpoint->generation += 1;
        generation = endpoint->generation;
        endpoint->state = Channel::HandshakeState::kConnectRequested;
        endpoint->failure_count = 0;
        endpoint->next_retry_at = absl::InfinitePast();
        endpoint->retry_scheduled = false;
        endpoint->pending_reads.push_back(
            Channel::PendingRdmaRead{
                .request = read_request,
                .segments = std::move(rdma_segs),
                .enqueued_at = now,
                .generation = generation,
            });
        queued_current = true;
        handshake_started = true;
        const size_t queue_depth = endpoint->pending_reads.size();
        endpoint->mu.Unlock();
        log_handshake_transition(
            local_dev_name,
            peer_dev_name,
            from_state,
            Channel::HandshakeState::kConnectRequested,
            generation,
            queue_depth);
        break;
      }

      endpoint->mu.Unlock();
      immediate_failure = absl::InternalError("unexpected RDMA handshake state");
      break;
    }

    if (!immediate_failure.ok()) {
      read_request->set_result(immediate_failure);
      pending_requests_.erase_if_present(req_key);
      return;
    }

    if (issue_now) {
      auto res = transport_to_use->read_multi(read_request, rdma_segs);
      if (res != misc::SUCCESS) {
        read_request->set_result(absl::UnavailableError("rdma read_multi failed before completion"));
        pending_requests_.erase_if_present(req_key);
      }
      return;
    }

    if (handshake_started) {
      auto send_res = t->send(connect_request_msg);
      if (send_res != misc::SUCCESS) {
        const absl::Status send_error = absl::UnavailableError("failed to send RDMA connect request to peer");
        auto failed_reads = drain_pending_reads_for_generation(endpoint, generation);
        for (auto& pending : failed_reads) {
          pending_requests_.erase_if_present(pending.request->get_key());
          pending.request->set_result(send_error);
        }
        {
          absl::MutexLock lock(&endpoint->mu);
          log_handshake_transition(
              local_dev_name,
              peer_dev_name,
              Channel::HandshakeState::kConnectRequested,
              Channel::HandshakeState::kFailed,
              endpoint->generation,
              endpoint->pending_reads.size());
          endpoint->state = Channel::HandshakeState::kFailed;
          endpoint->transport.reset();
          endpoint->failure_count += 1;
          endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
          endpoint->retry_scheduled = false;
        }
      }
      return;
    }

    if (queued_current && deferred_for_backoff && schedule_retry) {
      if (backoff_remaining <= absl::ZeroDuration()) {
        backoff_remaining = absl::Milliseconds(1);
      }
      schedule_handshake_retry(channel, local_dev_name, peer_dev_name, backoff_remaining);
    }
  };

  switch (msg->get_op()) {
    case ENGINE_OP_RDMA_CONNECT_RESPONSE: {
      if (msg->get_payload_size() == sizeof(ProtoRdmaConnectFailed)) {
        auto* failed = msg->get_payload<ProtoRdmaConnectFailed>();
        std::string peer_dev_name = reinterpret_cast<char*>(failed->dst_dev_name);
        std::string local_dev_name = reinterpret_cast<char*>(failed->src_dev_name);
        LOG(WARNING) << "[on_receive_response] Received RDMA_CONNECT_RESPONSE with failed payload; "
                        "treating as connect failure";
        handle_rdma_connect_failure(local_dev_name, peer_dev_name, "remote RDMA connect failed");
        break;
      }
      if (msg->get_payload_size() < sizeof(ProtoRdmaConnectResponse)) {
        LOG(ERROR) << "[on_receive_response] RDMA_CONNECT_RESPONSE payload too small: got=" << msg->get_payload_size()
                   << " expected=" << sizeof(ProtoRdmaConnectResponse);
        break;
      }

      LOG(INFO) << "get rdma response from " << t->get_remote_url();

      auto* req = msg->get_payload<ProtoRdmaConnectResponse>();
      std::string local_dev_name = reinterpret_cast<char*>(req->src_dev_name);
      std::string peer_dev_name = reinterpret_cast<char*>(req->dst_dev_name);
      auto endpoint = channel->get_rdma_endpoint(local_dev_name, peer_dev_name);
      if (endpoint == nullptr) {
        LOG(WARNING) << "[rdma_handshake] received connect response for unknown endpoint: local_dev=" << local_dev_name
                     << " peer_dev=" << peer_dev_name;
        break;
      }

      transport::rdma_transport_t transport;
      uint64_t generation = 0;
      Channel::HandshakeState from_state = Channel::HandshakeState::kConnectRequested;
      bool already_ready = false;
      {
        absl::MutexLock lock(&endpoint->mu);
        if (endpoint->state == Channel::HandshakeState::kConnectRequested) {
          transport = endpoint->transport;
          generation = endpoint->generation;
          from_state = Channel::HandshakeState::kConnectRequested;
        } else if (endpoint->state == Channel::HandshakeState::kReady && !endpoint->pending_reads.empty()) {
          transport = endpoint->transport;
          generation = endpoint->generation;
          from_state = Channel::HandshakeState::kReady;
          already_ready = true;
        } else {
          LOG(INFO) << "[rdma_handshake] ignoring late connect response: local_dev=" << local_dev_name
                    << " peer_dev=" << peer_dev_name;
          break;
        }
      }

      if (transport == nullptr) {
        LOG(WARNING) << "[rdma_handshake] connect response missing transport: local_dev=" << local_dev_name
                     << " peer_dev=" << peer_dev_name;
        auto failed_reads = drain_pending_reads_for_generation(endpoint, generation);
        const absl::Status status = absl::UnavailableError("rdma transport missing while processing connect response");
        for (auto& pending : failed_reads) {
          pending_requests_.erase_if_present(pending.request->get_key());
          pending.request->set_result(status);
        }
        {
          absl::MutexLock lock(&endpoint->mu);
          log_handshake_transition(
              local_dev_name,
              peer_dev_name,
              Channel::HandshakeState::kConnectRequested,
              Channel::HandshakeState::kFailed,
              endpoint->generation,
              endpoint->pending_reads.size());
          endpoint->state = Channel::HandshakeState::kFailed;
          endpoint->failure_count += 1;
          endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
          endpoint->retry_scheduled = false;
        }
        break;
      }

      if (!already_ready) {
        misc::result_t connect_res = transport->connect(&req->qp_info);
        if (connect_res != misc::SUCCESS) {
          LOG(WARNING) << "[rdma_handshake] transport connect failed: local_dev=" << local_dev_name
                       << " peer_dev=" << peer_dev_name << " res=" << connect_res;
          auto failed_reads = drain_pending_reads_for_generation(endpoint, generation);
          const absl::Status status = absl::UnavailableError("remote RDMA connect failed");
          for (auto& pending : failed_reads) {
            pending_requests_.erase_if_present(pending.request->get_key());
            pending.request->set_result(status);
          }
          {
            absl::MutexLock lock(&endpoint->mu);
            log_handshake_transition(
                local_dev_name,
                peer_dev_name,
                Channel::HandshakeState::kConnectRequested,
                Channel::HandshakeState::kFailed,
                endpoint->generation,
                endpoint->pending_reads.size());
            endpoint->state = Channel::HandshakeState::kFailed;
            endpoint->transport.reset();
            endpoint->failure_count += 1;
            endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
            endpoint->retry_scheduled = false;
          }
          break;
        }
      }

      size_t queued = 0;
      {
        absl::MutexLock lock(&endpoint->mu);
        queued = endpoint->pending_reads.size();
        endpoint->state = Channel::HandshakeState::kReady;
        endpoint->failure_count = 0;
        endpoint->next_retry_at = absl::InfinitePast();
        endpoint->retry_scheduled = false;
      }
      log_handshake_transition(
          local_dev_name, peer_dev_name, from_state, Channel::HandshakeState::kReady, generation, queued);

      auto ready_reads = drain_pending_reads_for_generation(endpoint, generation);
      for (auto& pending : ready_reads) {
        if (pending.enqueued_at != absl::InfinitePast()) {
          const auto wait_us = absl::ToInt64Microseconds(absl::Now() - pending.enqueued_at);
          if (wait_us > 0) {
            if (pending.request->rdma_profile_enabled()) {
              pending.request->note_rdma_handshake_queue_wait_us(static_cast<uint64_t>(wait_us));
            }
          }
        }
        auto res = transport->read_multi(pending.request, pending.segments);
        if (res != misc::SUCCESS) {
          pending_requests_.erase_if_present(pending.request->get_key());
          pending.request->set_result(absl::UnavailableError("rdma read_multi failed after handshake"));
        }
      }
      break;
    }
    case ENGINE_OP_MTCP_CONNECT_RESPONSE: {
      LOG(INFO) << "get mtcp response from " << t->get_remote_url();

      auto* rsp = msg->get_payload<ProtoMtcpConnectResponse>();
      struct in_addr sin_addr = {};
      sin_addr.s_addr = htonl(rsp->ip);
      auto transport = channel->get_mtcp();
      auto* ip = inet_ntoa(sin_addr);
      LOG(INFO) << "[on_receive_response] MTCP_CONNECT_RESPONSE: connecting to " << ip << ":" << rsp->port
                << " with conn_count=" << rsp->conn_count;
      const int negotiated_conn = std::max(2, static_cast<int>(rsp->conn_count));
      transport->set_conn_count(negotiated_conn);
      COMM_CHECK(transport->connect(ip, rsp->port, negotiated_conn));
      LOG(INFO) << "mtcp connect done " << ip << ":" << rsp->port << " " << negotiated_conn;
      break;
    }
    // case ENGINE_OP_READ_RESPONSE: (legacy) removed
    case ENGINE_OP_READ_RESPONSE_EX: {
      auto* hdr = msg->get_payload<ProtoReadResponseExHeader>();
      std::string tensor_key = reinterpret_cast<char*>(hdr->tensor_key);
      std::string peer_dev_name = reinterpret_cast<char*>(hdr->nic_name);

      auto* seg0 = reinterpret_cast<ProtoReadResponseExSeg*>(
          reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader));
      LOG(INFO) << "[on_receive_response] READ_RESPONSE_EX: key=" << tensor_key << " segs=" << hdr->num_segments
                << " transport=" << (hdr->transport_type == ENGINE_TRANSPORT_MTCP ? "MTCP" : "RDMA")
                << " offset=" << seg0->offset << " bytes=" << seg0->bytes << " stage_hint=" << hdr->credit_granted;
      auto req_key = transport::get_request_instance_key(tensor_key, hdr->request_offset, hdr->request_id);
      auto read_request = pending_requests_.get(req_key);
      if (read_request == nullptr) {
        LOG(ERROR) << "[on_receive_response] READ_RESPONSE_EX: pending request not found for " << req_key;
        break;
      }
      read_request->status_.transport_is_rdma = hdr->transport_type == ENGINE_TRANSPORT_RDMA;
      read_request->status_.remote_nic = peer_dev_name;
      read_request->status_.remote_rail_id = hdr->rail_id;
      read_request->status_.rdma_staged_response = hdr->staged != 0;
      read_request->status_.rdma_zero_copy_response = hdr->zero_copy != 0;
      read_request->record_request_response();

      if (enable_rdma_ && hdr->transport_type == ENGINE_TRANSPORT_RDMA) {
        if (read_request->rdma_profile_enabled()) {
          read_request->note_rdma_response_window(hdr->num_segments);
        }
        auto tensor = read_request->get_local_tensor();
        auto dev = tensor != nullptr ? tensor->get_dev() : nullptr;
        std::vector<transport::RdmaTransport::RdmaReadSeg> rdma_segs;
        rdma_segs.reserve(hdr->num_segments);
        std::vector<uint64_t> ack_offsets;
        ack_offsets.reserve(hdr->num_segments);
        const uint64_t base_off = read_request->remote_offset_;
        for (uint32_t i = 0; i < hdr->num_segments; ++i) {
          auto* s = reinterpret_cast<ProtoReadResponseExSeg*>(
              reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader) + i * sizeof(ProtoReadResponseExSeg));
          transport::RdmaTransport::RdmaReadSeg seg{};
          seg.remote_addr = s->addr;
          seg.rkey = s->rkey;
          seg.length = s->bytes;
          seg.local_addr = tensor->get_uint64_addr() + (s->offset - base_off);
          seg.window_seq = hdr->window_seq;
          seg.segment_idx = i;
          rdma_segs.emplace_back(seg);
          ack_offsets.emplace_back(s->offset);
        }

        read_request->note_rdma_window(static_cast<int>(rdma_segs.size()), hdr->more_segments == 0);

        if (hdr->staged) {
          auto ctrl = channel->get_control();
          const std::string staged_key = tensor_key;
          const uint64_t request_offset = read_request->remote_offset_;
          const uint64_t request_id = read_request->request_id();
          std::weak_ptr<transport::ReadRequest> weak_read_request(read_request);
          read_request->set_ack_sender([ctrl, staged_key, request_offset, request_id, weak_read_request](
                                           const transport::ReadRequest::PendingAckWindow& window) {
            if (auto ack_request = weak_read_request.lock(); ack_request != nullptr) {
              if (ack_request->rdma_profile_enabled()) {
                ack_request->note_rdma_ack_window(window.num_segments);
              }
            }
            auto ack = std::make_shared<EngineMessage>(
                ENGINE_OP_RDMA_READ_DONE_EX,
                static_cast<uint32_t>(
                    sizeof(ProtoRdmaReadDoneExHeader) + window.offsets.size() * sizeof(ProtoRdmaReadDoneExSeg)));
            auto* h = ack->get_payload<ProtoRdmaReadDoneExHeader>();
            misc::STRNCPY(h->tensor_key, staged_key, kMaxTensorNameLen);
            h->request_offset = request_offset;
            h->request_id = request_id;
            h->num_segments = static_cast<uint32_t>(window.offsets.size());
            h->window_seq = window.window_seq;
            h->final_window = window.final_window ? 1 : 0;
            for (size_t i = 0; i < window.offsets.size(); ++i) {
              auto* seg_ack = reinterpret_cast<ProtoRdmaReadDoneExSeg*>(
                  reinterpret_cast<uint8_t*>(h) + sizeof(ProtoRdmaReadDoneExHeader) +
                  i * sizeof(ProtoRdmaReadDoneExSeg));
              seg_ack->offset = window.offsets[i];
            }
            CHECK_WARN(ctrl->send(ack), "ack send failed");
          });

          read_request->enqueue_window_ack(hdr->window_seq, std::move(ack_offsets), hdr->more_segments == 0);
        }
        if (dev == nullptr) {
          read_request->set_result(absl::FailedPreconditionError("local tensor missing device metadata"));
          pending_requests_.erase_if_present(req_key);
          break;
        }
        handle_target_rdma_reads(req_key, read_request, peer_dev_name, std::move(rdma_segs));
      } else if (hdr->transport_type == ENGINE_TRANSPORT_MTCP) {
        // MTCP path using EX header (no segments)
        auto transport = channel->get_mtcp();
        if (transport == nullptr) {
          const int requested_conn = std::max(2, mtcp_conn_count_);
          transport = std::make_shared<transport::MTcpTransport>(
              requested_conn,
              gsl::not_null<std::shared_ptr<MemoryStager>>{memory_stager_},
              gsl::not_null<std::shared_ptr<MemoryStager>>{gpu_memory_stager_},
              gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{gpu_memory_pool_},
              buffers_per_flow_);
          LOG(INFO) << "[on_receive_response] Sending MTCP_CONNECT_REQUEST for " << tensor_key;
          auto req = EngineMessage::make_message<ProtoMtcpConnectRequest>(ENGINE_OP_MTCP_CONNECT_REQUEST);
          auto* payload = req->get_payload<ProtoMtcpConnectRequest>();
          payload->conn_count = requested_conn;
          COMM_CHECK(t->send(req));
          channel->set_transport(transport);
        }
        transport->set_tcp_tos(config_.transport().tcp_tos());
        if (hdr->credit_granted > 0) {
          read_request->set_mtcp_stage_unit_hint_bytes(static_cast<uint64_t>(hdr->credit_granted));
        }
        CHECK_WARN(transport->recv(read_request), "failed to recv via mtcp");
        // Remove pending entry now; completion is tracked in request future
        pending_requests_.del(req_key);
      } else {
        LOG(ERROR) << "[on_receive_response] READ_RESPONSE_EX unsupported transport type";
        read_request->set_result(absl::InternalError("READ_RESPONSE_EX unsupported transport type"));
        pending_requests_.erase_if_present(req_key);
      }
      break;
    }
    case ENGINE_OP_READ_PLAN_RESPONSE_EX: {
      auto* hdr = msg->get_payload<ProtoReadPlanResponseExHeader>();
      std::string peer_dev_name = reinterpret_cast<char*>(hdr->nic_name);
      auto req_key = transport::get_read_plan_request_key(hdr->request_id);
      auto read_request = pending_requests_.get(req_key);
      if (read_request == nullptr) {
        LOG(ERROR) << "[on_receive_response] READ_PLAN_RESPONSE_EX: pending request not found for " << req_key;
        break;
      }
      read_request->status_.transport_is_rdma = hdr->transport_type == ENGINE_TRANSPORT_RDMA;
      read_request->status_.remote_nic = peer_dev_name;
      read_request->status_.remote_rail_id = hdr->rail_id;
      read_request->status_.rdma_staged_response = hdr->staged != 0;
      read_request->status_.rdma_zero_copy_response = hdr->zero_copy != 0;
      read_request->record_request_response();

      if (enable_rdma_ && hdr->transport_type == ENGINE_TRANSPORT_RDMA) {
        if (read_request->rdma_profile_enabled()) {
          read_request->note_rdma_response_window(hdr->num_segments);
        }
        auto* segments = reinterpret_cast<ProtoReadPlanResponseExSeg*>(
            reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanResponseExHeader));
        auto prepared = read_request->get_prepared_read_plan();
        if (prepared == nullptr) {
          read_request->set_result(absl::FailedPreconditionError("read_plan response missing prepared plan"));
          pending_requests_.erase_if_present(req_key);
          break;
        }
        auto rdma_segs_or = BuildPreparedPlanRdmaSegments(*prepared, *hdr, segments);
        if (!rdma_segs_or.ok()) {
          read_request->set_result(rdma_segs_or.status());
          pending_requests_.erase_if_present(req_key);
          break;
        }
        auto rdma_segs = std::move(rdma_segs_or.value());
        read_request->note_rdma_window(static_cast<int>(rdma_segs.size()), hdr->more_segments == 0);

        if (hdr->staged) {
          auto ctrl = channel->get_control();
          const uint64_t request_id = read_request->request_id();
          std::weak_ptr<transport::ReadRequest> weak_read_request(read_request);
          read_request->set_ack_sender(
              [ctrl, request_id, weak_read_request](const transport::ReadRequest::PendingAckWindow& window) {
                if (auto ack_request = weak_read_request.lock(); ack_request != nullptr) {
                  if (ack_request->rdma_profile_enabled()) {
                    ack_request->note_rdma_ack_window(window.num_segments);
                  }
                }
                auto ack = EngineMessage::make_message<ProtoRdmaReadPlanDoneExHeader>(ENGINE_OP_RDMA_READ_PLAN_DONE_EX);
                auto* payload = ack->get_payload<ProtoRdmaReadPlanDoneExHeader>();
                payload->request_id = request_id;
                payload->num_segments = window.num_segments;
                payload->window_seq = window.window_seq;
                payload->final_window = window.final_window ? 1 : 0;
                CHECK_WARN(ctrl->send(ack), "read_plan ack send failed");
              });
          read_request->enqueue_plan_window_ack(
              hdr->window_seq, hdr->num_segments, static_cast<uint32_t>(rdma_segs.size()), hdr->more_segments == 0);
        }

        handle_target_rdma_reads(req_key, read_request, peer_dev_name, std::move(rdma_segs));
      } else {
        read_request->set_result(absl::InternalError("READ_PLAN_RESPONSE_EX unsupported transport type"));
        pending_requests_.erase_if_present(req_key);
      }
      break;
    }
    case ENGINE_OP_RDMA_CONNECT_FAILED: {
      if (msg->get_payload_size() < sizeof(ProtoRdmaConnectFailed)) {
        LOG(ERROR) << "[on_receive_response] RDMA_CONNECT_FAILED payload too small: got=" << msg->get_payload_size()
                   << " expected=" << sizeof(ProtoRdmaConnectFailed);
        break;
      }

      auto* req = msg->get_payload<ProtoRdmaConnectFailed>();
      std::string peer_dev_name = reinterpret_cast<char*>(req->dst_dev_name);
      std::string local_dev_name = reinterpret_cast<char*>(req->src_dev_name);
      handle_rdma_connect_failure(local_dev_name, peer_dev_name, "remote RDMA connect failed");
      break;
    }
    case ENGINE_OP_READ_FAILED: {
      auto* rsp = msg->get_payload<ProtoReadFailed>();
      auto tensor_key = std::string(reinterpret_cast<char*>(rsp->tensor_key));
      auto req_key = transport::get_request_instance_key(tensor_key, rsp->offset, rsp->request_id);

      LOG(ERROR) << "[on_receive_response] READ_FAILED: key=" << tensor_key << " offset=" << rsp->offset
                 << " reason=" << rsp->reason;

      auto read_request = pending_requests_.get(req_key);
      if (read_request == nullptr) {
        LOG(WARNING) << "failed to get read response: key=" << tensor_key;
        break;
      }
      pending_requests_.del(req_key);
      absl::Status failure_status = absl::InternalError("failed to read from peer");
      switch (rsp->reason) {
        case TENSORCAST_READ_FAILED_NO_TENSOR:
          failure_status = absl::NotFoundError(absl::StrCat("tensor not found: ", tensor_key));
          break;
        case TENSORCAST_READ_FAILED_OVERFLOW:
          failure_status =
              absl::OutOfRangeError(absl::StrCat("read overflow for tensor ", tensor_key, " offset=", rsp->offset));
          break;
        case TENSORCAST_READ_FAILED_RESOURCE_EXHAUSTED:
          failure_status = absl::ResourceExhaustedError("GPU staging pool capacity exceeded");
          break;
        case TENSORCAST_READ_FAILED_DIRECT_RDMA_REQUIRED:
          failure_status = absl::FailedPreconditionError("direct RDMA required but unavailable on peer");
          break;
        case TENSORCAST_READ_FAILED_MEM_MISMATCH:
        default:
          break;
      }
      read_request->set_result(failure_status);
      break;
    }
    case ENGINE_OP_READ_PLAN_FAILED: {
      auto* rsp = msg->get_payload<ProtoReadPlanFailed>();
      const auto req_key = transport::get_read_plan_request_key(rsp->request_id);
      auto read_request = pending_requests_.get(req_key);
      if (read_request == nullptr) {
        LOG(WARNING) << "failed to get read_plan response for request_id=" << rsp->request_id;
        break;
      }
      pending_requests_.del(req_key);
      absl::Status failure_status = absl::InternalError("failed to read plan from peer");
      switch (rsp->reason) {
        case TENSORCAST_READ_FAILED_NO_TENSOR:
          failure_status = absl::NotFoundError("read_plan source tensor not found on peer");
          break;
        case TENSORCAST_READ_FAILED_OVERFLOW:
          failure_status = absl::OutOfRangeError("read_plan source slice overflow on peer");
          break;
        case TENSORCAST_READ_FAILED_RESOURCE_EXHAUSTED:
          failure_status = absl::ResourceExhaustedError("read_plan source staging capacity exceeded on peer");
          break;
        case TENSORCAST_READ_FAILED_DIRECT_RDMA_REQUIRED:
          failure_status = absl::FailedPreconditionError("read_plan direct RDMA required but unavailable on peer");
          break;
        case TENSORCAST_READ_FAILED_MEM_MISMATCH:
        default:
          failure_status = absl::InternalError("read_plan request failed on peer");
          break;
      }
      read_request->set_result(failure_status);
      break;
    }
    default:
      LOG(WARNING) << "failed to process response: " << msg->get_op();
  }

  return misc::SUCCESS;
}

void Communicator::schedule_handshake_retry(
    const channel_t& channel,
    const std::string& local_dev_name,
    const std::string& peer_dev_name,
    absl::Duration delay) {
  if (!enable_rdma_ || !handshake_retry_thread_started_ || handshake_retry_stop_.load(std::memory_order_relaxed)) {
    return;
  }
  if (delay <= absl::ZeroDuration()) {
    delay = absl::Milliseconds(1);
  }

  HandshakeRetryTask task;
  task.resume_at = absl::Now() + delay;
  task.channel = channel;
  task.local_dev_name = local_dev_name;
  task.peer_dev_name = peer_dev_name;

  {
    absl::MutexLock lock(&handshake_retry_mu_);
    handshake_retry_queue_.push(std::move(task));
  }
  handshake_retry_cv_.Signal();
}

void Communicator::handshake_retry_loop() {
  while (!handshake_retry_stop_.load(std::memory_order_relaxed)) {
    HandshakeRetryTask task;
    bool has_task = false;
    {
      absl::MutexLock lock(&handshake_retry_mu_);
      while (!handshake_retry_stop_.load(std::memory_order_relaxed) && handshake_retry_queue_.empty()) {
        handshake_retry_cv_.Wait(&handshake_retry_mu_);
      }
      if (handshake_retry_stop_.load(std::memory_order_relaxed)) {
        break;
      }
      auto now = absl::Now();
      const auto& next = handshake_retry_queue_.top();
      if (next.resume_at > now) {
        handshake_retry_cv_.WaitWithDeadline(&handshake_retry_mu_, next.resume_at);
        continue;
      }
      task = handshake_retry_queue_.top();
      handshake_retry_queue_.pop();
      has_task = true;
    }

    if (!has_task) {
      continue;
    }

    process_handshake_retry_task(task.channel, task.local_dev_name, task.peer_dev_name);
  }
}

void Communicator::process_handshake_retry_task(
    const std::weak_ptr<Channel>& channel_weak,
    const std::string& local_dev_name,
    const std::string& peer_dev_name) {
  if (handshake_retry_stop_.load(std::memory_order_relaxed)) {
    return;
  }

  auto channel = channel_weak.lock();
  if (!channel) {
    return;
  }

  auto endpoint = channel->get_rdma_endpoint(local_dev_name, peer_dev_name);
  if (endpoint == nullptr) {
    return;
  }

  start_pending_rdma_handshake(channel, endpoint, local_dev_name, peer_dev_name);
}

void Communicator::start_pending_rdma_handshake(
    const channel_t& channel,
    const std::shared_ptr<Channel::RdmaEndpoint>& endpoint,
    const std::string& local_dev_name,
    const std::string& peer_dev_name) {
  if (!enable_rdma_ || handshake_retry_stop_.load(std::memory_order_relaxed)) {
    return;
  }

  auto control = channel->get_control();
  if (control == nullptr) {
    return;
  }

  transport::rdma_transport_t prepared_transport;
  std::shared_ptr<EngineMessage> connect_request_msg;

  while (!handshake_retry_stop_.load(std::memory_order_relaxed)) {
    endpoint->mu.Lock();
    endpoint->retry_scheduled = false;
    const auto state = endpoint->state;
    const bool has_pending = !endpoint->pending_reads.empty();
    const absl::Time now = absl::Now();

    if (!has_pending) {
      endpoint->mu.Unlock();
      return;
    }

    if (state == Channel::HandshakeState::kConnectRequested || state == Channel::HandshakeState::kReady) {
      endpoint->mu.Unlock();
      return;
    }

    if (state == Channel::HandshakeState::kFailed && now < endpoint->next_retry_at) {
      const absl::Duration delay = endpoint->next_retry_at - now;
      endpoint->retry_scheduled = true;
      endpoint->mu.Unlock();
      schedule_handshake_retry(channel, local_dev_name, peer_dev_name, delay);
      return;
    }

    if (prepared_transport == nullptr) {
      endpoint->mu.Unlock();
      prepared_transport = rdma_context_->create_transport(local_dev_name);
      if (prepared_transport == nullptr) {
        const absl::Status status = absl::InternalError("failed to allocate RDMA transport");
        auto failed_reads = drain_pending_reads_for_generation(endpoint, 0);
        for (auto& pending : failed_reads) {
          pending_requests_.erase_if_present(pending.request->get_key());
          pending.request->set_result(status);
        }
        {
          absl::MutexLock lock(&endpoint->mu);
          endpoint->state = Channel::HandshakeState::kFailed;
          endpoint->transport.reset();
          endpoint->failure_count += 1;
          endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
          endpoint->retry_scheduled = false;
        }
        return;
      }

      connect_request_msg = EngineMessage::make_message<ProtoRdmaConnectRequest>(ENGINE_OP_RDMA_CONNECT_REQUEST);
      auto* payload = connect_request_msg->get_payload<ProtoRdmaConnectRequest>();
      misc::result_t info_res = prepared_transport->get_local_info(&payload->qp_info);
      if (info_res != misc::SUCCESS) {
        prepared_transport.reset();
        connect_request_msg.reset();
        const absl::Status status = absl::InternalError("failed to prepare RDMA connect info");
        auto failed_reads = drain_pending_reads_for_generation(endpoint, 0);
        for (auto& pending : failed_reads) {
          pending_requests_.erase_if_present(pending.request->get_key());
          pending.request->set_result(status);
        }
        {
          absl::MutexLock lock(&endpoint->mu);
          endpoint->state = Channel::HandshakeState::kFailed;
          endpoint->transport.reset();
          endpoint->failure_count += 1;
          endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
          endpoint->retry_scheduled = false;
        }
        return;
      }

      misc::STRNCPY(payload->src_dev_name, local_dev_name, kMaxDevName);
      misc::STRNCPY(payload->dst_dev_name, peer_dev_name, kMaxDevName);
      continue;
    }

    Channel::HandshakeState from_state = state;
    endpoint->transport = prepared_transport;
    endpoint->generation += 1;
    uint64_t generation = endpoint->generation;
    endpoint->state = Channel::HandshakeState::kConnectRequested;
    endpoint->failure_count = 0;
    endpoint->next_retry_at = absl::InfinitePast();
    const size_t queue_depth = endpoint->pending_reads.size();
    for (auto& pending : endpoint->pending_reads) {
      pending.generation = generation;
    }
    endpoint->mu.Unlock();

    log_handshake_transition(
        local_dev_name, peer_dev_name, from_state, Channel::HandshakeState::kConnectRequested, generation, queue_depth);

    auto send_res = control->send(connect_request_msg);
    if (send_res != misc::SUCCESS) {
      LOG(WARNING) << "[rdma_handshake] failed to send connect request: local_dev=" << local_dev_name
                   << " peer_dev=" << peer_dev_name << " res=" << send_res;
      const absl::Status send_error = absl::UnavailableError("failed to send RDMA connect request to peer");
      auto failed_reads = drain_pending_reads_for_generation(endpoint, generation);
      for (auto& pending : failed_reads) {
        pending_requests_.erase_if_present(pending.request->get_key());
        pending.request->set_result(send_error);
      }
      {
        absl::MutexLock lock(&endpoint->mu);
        log_handshake_transition(
            local_dev_name,
            peer_dev_name,
            Channel::HandshakeState::kConnectRequested,
            Channel::HandshakeState::kFailed,
            endpoint->generation,
            endpoint->pending_reads.size());
        endpoint->state = Channel::HandshakeState::kFailed;
        endpoint->transport.reset();
        endpoint->failure_count += 1;
        endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
        endpoint->retry_scheduled = false;
      }
    }
    return;
  }
}

net_dev_t Communicator::get_net_dev(int dev_type, int dev_id, const std::string& key, int rail_id) {
  net_dev_t net_dev(nullptr);
  if (enable_rdma_) {
    CHECK(rdma_context_ != nullptr) << "rdma context is not initialized";

    net_dev = rdma_context_->get_best_dev(dev_type, dev_id, rail_id, key);

    if (net_dev == nullptr) {
      LOG(WARNING) << "failed to select RDMA device (dev_type=" << dev_type
                   << ") — ensure CommunicatorConfig specifies device mapping";
      return nullptr;
    }
  }
  return net_dev;
}

absl::Status Communicator::close_connection(const std::string& dst_ip, uint16_t dst_port) {
  std::stringstream url;
  url << dst_ip << ":" << dst_port;
  auto channel = channels_.get(url.str());
  if (channel == nullptr) {
    return absl::InternalError("could not find the connection");
  }
  if (!channels_.erase_if(url.str(), channel)) {
    VLOG(1) << "[close_connection] Channel already removed or replaced for " << url.str();
  }
  channel->close();
  return absl::OkStatus();
}

void Communicator::do_channel_gc_loop() {
  while (!stop_.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    auto pairs = channels_.pairs();
    auto now = get_us() / 1000000;
    for (auto& p : pairs) {
      if (p.second->is_expired(now)) {
        LOG(INFO) << "channel gc " << p.first;
        if (!channels_.erase_if(p.first, p.second)) {
          VLOG(1) << "[channel gc] Channel already removed or replaced for " << p.first;
        }
        p.second->close();
      }
    }

    pairs.clear();

    const uint64_t ttl_ms = ack_ttl_ms_ ? ack_ttl_ms_ : 30000;
    if (ttl_ms > 0) {
      auto channels_copy = channels_.pairs();
      const absl::Duration ttl = absl::Milliseconds(static_cast<int64_t>(ttl_ms));
      for (auto& entry : channels_copy) {
        auto flow_state = entry.second->flow_state();
        if (!flow_state) {
          continue;
        }
        auto expired = flow_state->registry.snapshot_expired(ttl);
        bool resumed = false;
        for (const auto& stale : expired) {
          auto lease_or = flow_state->registry.take(stale.key);
          if (!lease_or.ok()) {
            continue;
          }
          auto metadata = lease_or->metadata();
          LOG(WARNING) << "[staging_credit] Reaping lease request=" << metadata.request_key
                       << " window=" << metadata.window_seq << " segment=" << metadata.segment_idx
                       << " bytes=" << metadata.bytes;
          lease_or->release();
          finish_source_transfer_progress(
              make_transfer_id(metadata.request_key, entry.first),
              absl::DeadlineExceededError("source staged lease reaped before ACK"));
          resumed = true;
        }
        if (resumed) {
          auto resume_status = resume_rdma_reads(entry.second);
          if (!resume_status.ok()) {
            LOG(WARNING) << "Failed to resume RDMA staging after lease reap: " << resume_status;
          }
        }
      }
    }
  }
}

std::shared_ptr<MemoryStager> Communicator::get_cpu_stager_for_nic(const std::string& nic_name) const {
  auto it = nic_cpu_stagers_.find(nic_name);
  if (it != nic_cpu_stagers_.end())
    return it->second;
  return nullptr;
}

std::shared_ptr<MemoryStager> Communicator::get_gpu_mem_stager_for_id(int gpu_id) const {
  auto it = gpu_mem_stagers_.find(gpu_id);
  if (it != gpu_mem_stagers_.end())
    return it->second;
  return nullptr;
}

std::shared_ptr<MemoryStager> Communicator::get_gpu_vram_stager_for_id(int gpu_id) const {
  auto it = gpu_vram_stagers_.find(gpu_id);
  if (it != gpu_vram_stagers_.end())
    return it->second;
  return nullptr;
}

} // namespace tensorcast::communicator::engine
