// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/metadata/registration_backend.h"

#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_verification.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/cuda_memory.h"
#include "core/common/trace/trace_macros.h"
#include "core/store/components/eviction_service.h"
#include "core/store/components/stable_dram_cache_manager.h"
#include "core/store/device_registry.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/sources/segment_plan_source.h"
#include "core/store/materialization/dataplane/verification/verification_utils.h"
#include "core/store/materialization/dataplane/view/view_ingest_executor.h"
#include "core/store/materialization/dataplane/view/view_plan_source.h"
#include "core/store/view_utils.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::store::runtime::metadata {

namespace {

uint64_t sum_view_write_bytes(const loader::ViewWritePlan& write_plan) {
  uint64_t total = 0;
  for (const auto& chunk : write_plan.chunks) {
    total += chunk.length;
  }
  return total;
}

std::vector<CanonicalRange> canonical_ranges_from_write_plan(const loader::ViewWritePlan& write_plan) {
  std::vector<CanonicalRange> ranges;
  ranges.reserve(write_plan.chunks.size());
  for (const auto& chunk : write_plan.chunks) {
    CanonicalRange range;
    range.offset = chunk.canonical_offset;
    range.length = chunk.length;
    ranges.push_back(range);
  }
  std::sort(ranges.begin(), ranges.end(), [](const CanonicalRange& a, const CanonicalRange& b) {
    return a.offset < b.offset;
  });
  std::vector<CanonicalRange> merged;
  merged.reserve(ranges.size());
  for (const auto& range : ranges) {
    if (range.length == 0) {
      continue;
    }
    if (merged.empty()) {
      merged.push_back(range);
      continue;
    }
    auto& last = merged.back();
    const uint64_t last_end = last.offset + last.length;
    if (range.offset <= last_end) {
      const uint64_t new_end = std::max(last_end, range.offset + range.length);
      last.length = new_end - last.offset;
    } else {
      merged.push_back(range);
    }
  }
  return merged;
}

std::string make_registration_id() {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;
  return absl::StrCat("reg_", absl::ToUnixNanos(absl::Now()), "_", getpid(), "_", dis(gen));
}

absl::StatusOr<std::pair<std::string, std::string>> parse_mi2_multihashes(std::string_view artifact_id) {
  constexpr std::string_view kPrefix = "mi2:";
  if (!artifact_id.starts_with(kPrefix)) {
    return absl::InvalidArgumentError("artifact_id_override must start with \"mi2:\"");
  }
  const size_t index_begin = kPrefix.size();
  const size_t sep = artifact_id.find(':', index_begin);
  if (sep == std::string_view::npos) {
    return absl::InvalidArgumentError("artifact_id_override must be of form mi2:<index_multihash>:<data_multihash>");
  }
  const std::string_view index_mh = artifact_id.substr(index_begin, sep - index_begin);
  const std::string_view data_mh = artifact_id.substr(sep + 1);
  if (index_mh.empty() || data_mh.empty()) {
    return absl::InvalidArgumentError("artifact_id_override must include index and data multihash components");
  }
  return std::make_pair(std::string(index_mh), std::string(data_mh));
}

} // namespace

struct RegistrationBackend::PendingRegistrationContext {
  enum class Plan : uint8_t { kCoalesced = 0, kStableDram = 1 };

  std::string registration_id;
  std::string artifact_id;
  std::string client_artifact_id;
  std::optional<std::string> artifact_id_override;
  int device_id{0};
  uint64_t size_bytes{0};
  std::string tensor_index_key;
  std::optional<std::string> tensor_index_data;
  std::string schema_version;
  std::string encoding;
  bool enable_p2p{true};
  common::ArtifactIdKind id_kind{common::ArtifactIdKind::kMi2};
  std::shared_ptr<replica::Replica> replica;
  void* gpu_ptr{nullptr};
  cudaIpcMemHandle_t ipc_handle{};
  std::unique_ptr<common::memory::GpuDeviceMemory> staging_gpu;
  StableDramOptions stable_dram;
  std::optional<components::StableDramCachePolicy> stable_cache_policy;
  std::chrono::steady_clock::time_point expiry_time;
  std::chrono::steady_clock::time_point begin_time;
  Plan plan{Plan::kCoalesced};

  struct ViewState {
    ViewRegistration options;
    loader::BidirectionalViewPlan plan;
    std::unique_ptr<loader::ViewIngestExecutor> executor;
    uint64_t expected_view_bytes{0};
    uint64_t ingested_bytes{0};
    bool finalized{false};
  };

  std::unique_ptr<ViewState> view_state;
};

RegistrationBackend::RegistrationBackend(
    RegistrationResources resources,
    ReplicaFactory replica_factory,
    size_t artifact_chunk_bytes,
    std::chrono::milliseconds pinned_memory_timeout,
    size_t streaming_buffer_chunks,
    RegistrationPublisher* publisher)
    : device_manager_(resources.device_manager),
      replica_registry_(resources.replica_registry),
      metrics_collector_(resources.metrics_collector),
      memory_pool_(resources.memory_pool),
      communication_manager_(std::move(resources.communication_manager)),
      stable_cache_manager_(std::move(resources.stable_cache_manager)),
      async_runtime_(std::move(resources.async_runtime)),
      memory_tier_budget_(std::move(resources.memory_tier_budget)),
      memory_tier_config_(resources.memory_tier_config),
      replica_factory_(std::move(replica_factory)),
      artifact_chunk_bytes_(artifact_chunk_bytes),
      pinned_memory_timeout_(pinned_memory_timeout),
      streaming_buffer_chunks_(std::max<size_t>(1, streaming_buffer_chunks)),
      publisher_(publisher) {
  ABSL_CHECK(replica_factory_) << "ReplicaFactory must be provided";
  ABSL_CHECK(async_runtime_ != nullptr) << "RegistrationResources.async_runtime is required";
}

absl::StatusOr<RegistrationBeginResult> RegistrationBackend::begin(const ArtifactRegistration& reg) {
  std::string trace_request_id = reg.tensor_index_key;
  if (trace_request_id.empty()) {
    trace_request_id = reg.artifact_id;
  }
  if (trace_request_id.empty()) {
    trace_request_id = "registration";
  }
  std::string trace_artifact_id = reg.artifact_id.empty() ? trace_request_id : reg.artifact_id;
  SC_TRACE_INIT_GUARD(trace_request_id, trace_artifact_id, "registration_begin");

  if (reg.total_size_bytes == 0) {
    return absl::InvalidArgumentError("total_size_bytes must be > 0");
  }
  const bool stable_dram = reg.plan == RegistrationPlan::kStableDram;
  if (reg.device_id < 0) {
    return absl::InvalidArgumentError("device_id must be >= 0");
  }
  if (stable_dram && !reg.stable_dram.stage_on_gpu) {
    return absl::UnimplementedError("stable_dram stage_on_gpu=false is not implemented");
  }
  if (!reg.schema_version.empty() && reg.schema_version != "v3") {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported ArtifactRegistration.schema_version='", reg.schema_version, "'; expected 'v3'"));
  }
  if (reg.tensor_index_key.empty() && !reg.tensor_index_data.has_value()) {
    return absl::InvalidArgumentError("tensor index key or data must be provided");
  }
  if (reg.client_artifact_id.has_value() && reg.artifact_id_override.has_value()) {
    return absl::InvalidArgumentError("artifact_id_override cannot be set when client_artifact_id is provided");
  }
  if (reg.artifact_id_override.has_value()) {
    if (reg.artifact_id_override->empty()) {
      return absl::InvalidArgumentError("artifact_id_override must be non-empty when provided");
    }
    auto parsed_or = parse_mi2_multihashes(*reg.artifact_id_override);
    if (!parsed_or.ok()) {
      return parsed_or.status();
    }
  }

  const uint64_t canonical_size = (reg.view.has_value() && reg.view->canonical_size_bytes != 0)
      ? reg.view->canonical_size_bytes
      : reg.total_size_bytes;

  std::optional<loader::BidirectionalViewPlan> view_plan;
  uint64_t expected_view_bytes = 0;
  std::vector<CanonicalRange> canonical_ranges;
  if (reg.view.has_value()) {
    const auto& view_opts = *reg.view;
    if (!reg.tensor_index_data.has_value() || reg.tensor_index_data->empty()) {
      return absl::InvalidArgumentError("view registration requires inline canonical index data");
    }
    if (view_opts.placement == ViewPlacement::kUnspecified) {
      return absl::InvalidArgumentError("view registration requires explicit placement");
    }
    if (view_opts.canonical_size_bytes != 0 && view_opts.canonical_size_bytes != reg.total_size_bytes) {
      return absl::InvalidArgumentError("view.canonical_size_bytes must match total_size_bytes");
    }
    auto plan_or = loader::ViewPlanner::compute_bidirectional_view_plan(*reg.tensor_index_data, view_opts.spec);
    if (!plan_or.ok()) {
      return plan_or.status();
    }
    view_plan = std::move(*plan_or);
    expected_view_bytes = sum_view_write_bytes(view_plan->write);
    canonical_ranges = canonical_ranges_from_write_plan(view_plan->write);
    uint64_t covered_bytes = 0;
    for (const auto& range : canonical_ranges) {
      covered_bytes += range.length;
    }
    if (!view_opts.allow_partial && covered_bytes != canonical_size) {
      return absl::InvalidArgumentError(
          "view registration does not fully cover canonical bytes; set allow_partial=true to permit partial coverage");
    }
    if (covered_bytes > canonical_size) {
      return absl::InvalidArgumentError("view registration exceeds canonical byte space");
    }
    if (view_opts.placement == ViewPlacement::kServer && view_plan->inverse_transform.requires_materialization) {
      auto info_or = device_manager_->get_gpu_info(reg.device_id);
      if (!info_or.ok()) {
        return absl::FailedPreconditionError(
            absl::StrCat(
                "SERVER placement for view registration requires GPU transpose support on device ",
                reg.device_id,
                "; retry with placement=CLIENT (",
                info_or.status().message(),
                ")"));
      }
    }
  }

  loading::InlineBufferSource ib_source{.data = nullptr, .size_bytes = reg.total_size_bytes};
  replica::ReplicaConfig cfg{
      .source = ib_source,
      .artifact_identifier = reg.artifact_id,
      .device_type = stable_dram ? DeviceType::CPU : DeviceType::GPU,
      .local_device_id = stable_dram ? -1 : reg.device_id,
      .pinned_buffer_pool = memory_pool_,
      .async_runtime = gsl::not_null<std::shared_ptr<common::AsyncRuntime>>{async_runtime_},
      .artifact_chunk_bytes = artifact_chunk_bytes_,
      .expected_artifact_size = reg.total_size_bytes};
  cfg.pinned_memory_timeout = pinned_memory_timeout_;
  cfg.streaming_buffer_chunks = streaming_buffer_chunks_;
  if (memory_tier_config_.has_value()) {
    cfg.memory_tier_config = memory_tier_config_;
  }

  const bool needs_gpu_staging = !stable_dram || reg.stable_dram.stage_on_gpu;
  if (needs_gpu_staging) {
    if (auto free_or = device_manager_->get_free_memory(reg.device_id); free_or.ok()) {
      size_t free_bytes = free_or.value();
      if (reg.total_size_bytes > free_bytes) {
        auto evict_status = components::evict_for_gpu(
            *replica_registry_,
            *device_manager_,
            *metrics_collector_,
            reg.device_id,
            reg.total_size_bytes - free_bytes);
        if (!evict_status.ok()) {
          return absl::ResourceExhaustedError(
              absl::StrCat(
                  "Insufficient GPU memory available. Requested: ",
                  reg.total_size_bytes,
                  " bytes, Free: ",
                  free_bytes,
                  ". ",
                  evict_status.message()));
        }
      }
    }
  }

  auto replica_or = replica_factory_(cfg);
  if (!replica_or.ok()) {
    return replica_or.status();
  }
  auto replica = std::move(replica_or.value());

  if (memory_tier_budget_) {
    replica->get_memory_manager().set_memory_tier_budget(memory_tier_budget_);
  }

  void* base_ptr = nullptr;
  cudaIpcMemHandle_t ipc_handle{};
  std::unique_ptr<common::memory::GpuDeviceMemory> staging_gpu;
  if (stable_dram) {
    absl::Status alloc_status = replica->get_memory_manager().allocate_memory(common::memory::MemoryLocation::CPU);
    if (!alloc_status.ok()) {
      return alloc_status;
    }
    if (reg.stable_dram.stage_on_gpu) {
      staging_gpu = std::make_unique<common::memory::GpuDeviceMemory>();
      absl::Status staging_status = staging_gpu->allocate(reg.total_size_bytes, reg.device_id);
      if (!staging_status.ok()) {
        return staging_status;
      }
      base_ptr = staging_gpu->get();
      if (base_ptr == nullptr) {
        return absl::InternalError("Stable DRAM staging pointer is null");
      }
      auto ipc_status = cuda::get_ipc_mem_handle(&ipc_handle, base_ptr);
      if (!ipc_status.ok()) {
        return ipc_status;
      }
    }
  } else {
    absl::Status alloc_status = replica->get_memory_manager().allocate_memory(common::memory::MemoryLocation::GPU);
    if (!alloc_status.ok()) {
      return alloc_status;
    }

    auto ipc_or = replica->get_memory_manager().get_ipc_handle();
    if (!ipc_or.ok()) {
      return ipc_or.status();
    }
    ipc_handle = *ipc_or;
    const auto gpu_ptrs = replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
    base_ptr = (!gpu_ptrs.empty() ? gpu_ptrs[0] : nullptr);
  }

  DeviceKey dev_key = stable_dram ? DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""}
                                  : DeviceRegistry::instance().gpu_key(reg.device_id);
  loading::ReplicaKey inst_key{.artifact_id = reg.artifact_id, .device = dev_key, .replica = 0};
  {
    absl::Status emplace_status =
        replica_registry_->emplace(inst_key, gsl::not_null<std::shared_ptr<replica::Replica>>{replica});
    if (absl::IsAlreadyExists(emplace_status)) {
      VLOG(1) << "Pending registry already had instance for key=" << inst_key.artifact_id;
    } else if (!emplace_status.ok()) {
      return emplace_status;
    }
  }

  auto entry = std::make_shared<PendingRegistrationContext>();
  entry->registration_id = make_registration_id();
  entry->artifact_id = reg.artifact_id;
  if (reg.client_artifact_id.has_value()) {
    entry->client_artifact_id = *reg.client_artifact_id;
    entry->id_kind = common::ArtifactIdKind::kCgid;
  } else {
    entry->client_artifact_id.clear();
    entry->id_kind = common::ArtifactIdKind::kMi2;
  }
  entry->artifact_id_override = reg.artifact_id_override;
  entry->device_id = reg.device_id;
  entry->size_bytes = reg.total_size_bytes;
  entry->tensor_index_key = reg.tensor_index_key;
  entry->tensor_index_data = reg.tensor_index_data;
  entry->schema_version = reg.schema_version;
  entry->encoding = reg.encoding;
  entry->enable_p2p = reg.enable_p2p;
  if (reg.ttl_ms > 0) {
    entry->expiry_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(reg.ttl_ms);
  }
  entry->replica = replica;
  entry->gpu_ptr = base_ptr;
  entry->ipc_handle = ipc_handle;
  entry->staging_gpu = std::move(staging_gpu);
  entry->stable_dram = reg.stable_dram;
  entry->stable_cache_policy = reg.stable_cache_policy;
  entry->plan =
      stable_dram ? PendingRegistrationContext::Plan::kStableDram : PendingRegistrationContext::Plan::kCoalesced;
  entry->begin_time = std::chrono::steady_clock::now();

  if (view_plan.has_value()) {
    entry->view_state = std::make_unique<PendingRegistrationContext::ViewState>();
    entry->view_state->options = *reg.view;
    entry->view_state->options.canonical_size_bytes = canonical_size;
    entry->view_state->options.canonical_ranges = canonical_ranges;
    entry->view_state->plan = *view_plan;
    entry->view_state->expected_view_bytes = expected_view_bytes;
    entry->view_state->ingested_bytes = 0;
    if (entry->view_state->options.placement == ViewPlacement::kServer) {
      entry->view_state->executor = std::make_unique<loader::ViewIngestExecutor>(
          entry->view_state->plan.write, entry->view_state->plan.inverse_transform);
    }
  }

  size_t pending_size_after = 0;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_regs_.emplace(entry->registration_id, entry);
    pending_size_after = pending_regs_.size();
  }
  record_pending_gauge(pending_size_after);

  RegistrationBeginResult out;
  out.registration_id = entry->registration_id;
  out.device_id = reg.device_id;
  out.size_bytes = reg.total_size_bytes;
  std::memcpy(out.cuda_ipc_handle_bytes.data(), &ipc_handle, sizeof(cudaIpcMemHandle_t));
  return out;
}

absl::StatusOr<RegistrationCommitResult> RegistrationBackend::commit(std::string_view registration_id) {
  std::string request_id(registration_id);
  if (request_id.empty()) {
    request_id = "registration";
  }
  std::shared_ptr<PendingRegistrationContext> entry;
  std::shared_ptr<PendingRegistrationContext> expired_entry;
  size_t pending_size_after_expire = 0;
  std::string trace_artifact_id = request_id;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_regs_.find(std::string(registration_id));
    if (it == pending_regs_.end()) {
      return absl::NotFoundError("registration_id not found");
    }
    if (!it->second->artifact_id.empty()) {
      trace_artifact_id = it->second->artifact_id;
    }
    const bool has_ttl = it->second->expiry_time.time_since_epoch().count() > 0;
    if (has_ttl && std::chrono::steady_clock::now() > it->second->expiry_time) {
      expired_entry = it->second;
      pending_regs_.erase(it);
      pending_size_after_expire = pending_regs_.size();
    } else {
      entry = it->second;
    }
  }

  SC_TRACE_INIT_GUARD(request_id, trace_artifact_id, "registration_commit");

  if (expired_entry) {
    record_pending_gauge(pending_size_after_expire);
    record_commit_latency(*expired_entry, "expired");
    const auto location = expired_entry->plan == PendingRegistrationContext::Plan::kStableDram
        ? common::memory::MemoryLocation::CPU
        : common::memory::MemoryLocation::GPU;
    release_replica_memory(expired_entry->replica, location);
    return absl::DeadlineExceededError("registration expired (TTL)");
  }
  if (!entry) {
    return absl::InternalError("pending registration entry missing after TTL check");
  }

  if (entry->view_state && entry->view_state->options.placement == ViewPlacement::kServer) {
    auto& view_state = *entry->view_state;
    if (!view_state.executor) {
      return absl::FailedPreconditionError("view executor missing for server placement registration");
    }
    if (!view_state.executor->is_complete()) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "view bytes incomplete; expected ",
              view_state.expected_view_bytes,
              " received ",
              view_state.executor->ingested_bytes()));
    }
    auto finalize_status =
        view_state.executor->finalize(common::memory::MemoryLocation::GPU, entry->gpu_ptr, entry->device_id);
    if (!finalize_status.ok()) {
      return finalize_status;
    }
    view_state.finalized = true;
  }

  if (entry->view_state && entry->view_state->options.allow_partial) {
    const auto& ranges = entry->view_state->options.canonical_ranges;
    uint64_t covered_bytes = 0;
    for (const auto& r : ranges) {
      covered_bytes += r.length;
    }
    if (covered_bytes < entry->size_bytes) {
      void* gpu_ptr = entry->gpu_ptr;
      if (gpu_ptr == nullptr) {
        const auto ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
        gpu_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
      }
      if (!gpu_ptr) {
        return absl::FailedPreconditionError("GPU pointer is null; cannot zero uncovered regions");
      }
      auto dev_status = cuda::set_device(entry->device_id);
      if (!dev_status.ok()) {
        return dev_status;
      }
      uint64_t cursor = 0;
      for (const auto& r : ranges) {
        if (r.offset > cursor) {
          auto ms = cuda::memset(static_cast<uint8_t*>(gpu_ptr) + cursor, 0, static_cast<size_t>(r.offset - cursor));
          if (!ms.ok()) {
            return ms;
          }
        }
        const uint64_t r_end = r.offset + r.length;
        cursor = std::max(r_end, cursor);
      }
      if (cursor < entry->size_bytes) {
        auto ms =
            cuda::memset(static_cast<uint8_t*>(gpu_ptr) + cursor, 0, static_cast<size_t>(entry->size_bytes - cursor));
        if (!ms.ok()) {
          return ms;
        }
      }
      auto sync = cuda::device_synchronize();
      if (!sync.ok()) {
        return sync;
      }
    }
  }

  if (entry->plan == PendingRegistrationContext::Plan::kStableDram) {
    if (entry->stable_dram.stage_on_gpu) {
      if (entry->gpu_ptr == nullptr) {
        return absl::FailedPreconditionError("Stable DRAM staging pointer is null");
      }
      const auto cpu_ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::CPU);
      if (cpu_ptrs.empty() || cpu_ptrs[0] == nullptr) {
        return absl::FailedPreconditionError("CPU pointer unavailable for stable DRAM commit");
      }
      auto dev_status = cuda::set_device(entry->device_id);
      if (!dev_status.ok()) {
        return dev_status;
      }
      auto copy_status =
          cuda::memcpy(cpu_ptrs[0], entry->gpu_ptr, static_cast<size_t>(entry->size_bytes), cudaMemcpyDeviceToHost);
      if (!copy_status.ok()) {
        return copy_status;
      }
      auto sync_status = cuda::device_synchronize();
      if (!sync_status.ok()) {
        return sync_status;
      }
    }
  }

  std::string index_multihash;
  std::string data_multihash;
  std::vector<loader::SegmentPiece> segment_plan;
  bool segment_plan_ready = false;

  const bool has_artifact_id_override =
      entry->artifact_id_override.has_value() && !entry->artifact_id_override->empty();
  if (has_artifact_id_override) {
    if (!entry->client_artifact_id.empty()) {
      return absl::InvalidArgumentError("artifact_id_override cannot be set when client_artifact_id is present");
    }
    auto parsed_or = parse_mi2_multihashes(*entry->artifact_id_override);
    if (!parsed_or.ok()) {
      return parsed_or.status();
    }
    index_multihash = parsed_or->first;
    data_multihash = parsed_or->second;
    entry->artifact_id = *entry->artifact_id_override;
    entry->id_kind = common::ArtifactIdKind::kMi2;
  } else if (entry->id_kind == common::ArtifactIdKind::kMi2 || entry->client_artifact_id.empty()) {
    absl::StatusOr<std::string> index_mh_or =
        common::compute_index_multihash(entry->tensor_index_data, entry->tensor_index_key);
    if (!index_mh_or.ok()) {
      return index_mh_or.status();
    }
    index_multihash = *index_mh_or;

    if (!entry->schema_version.empty() && entry->schema_version != "v3") {
      return absl::FailedPreconditionError(
          absl::StrCat("Pending registration schema_version must be 'v3'; found '", entry->schema_version, "'"));
    }

    absl::StatusOr<std::string> data_mh_or;
    if (entry->plan == PendingRegistrationContext::Plan::kStableDram) {
      if (!entry->replica) {
        return absl::FailedPreconditionError("CPU registration missing backing replica");
      }
      const auto cpu_ptrs = entry->replica->get_data_pointer(common::memory::MemoryLocation::CPU);
      if (cpu_ptrs.empty() || cpu_ptrs[0] == nullptr) {
        return absl::FailedPreconditionError("CPU data pointer unavailable for hashing");
      }
      gsl::not_null<const void*> cpu_base{static_cast<const void*>(cpu_ptrs[0])};
      auto mh_or = loader::compute_data_multihash_from_cpu_memory(cpu_base, entry->size_bytes);
      if (!mh_or.ok()) {
        return mh_or.status();
      }
      data_mh_or = *mh_or;
    } else {
      void* gpu_ptr = entry->gpu_ptr;
      if (gpu_ptr == nullptr) {
        const auto ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
        gpu_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
      }
      if (!gpu_ptr) {
        return absl::FailedPreconditionError("GPU pointer is null; cannot hash GPU data");
      }
      if (entry->tensor_index_data.has_value() && !entry->tensor_index_data->empty() && entry->encoding == "json") {
        auto plan_or = loader::build_segment_plan_from_canonical_index_json(
            *entry->tensor_index_data, entry->size_bytes, /*align_bytes=*/8);
        if (plan_or.ok()) {
          segment_plan = std::move(*plan_or);
          segment_plan_ready = true;
        }
      }
      if (segment_plan_ready) {
        auto mh_or = loader::compute_data_multihash_from_gpu_plan(
            gsl::not_null<void*>{gpu_ptr}, entry->device_id, absl::MakeSpan(segment_plan), entry->size_bytes);
        if (!mh_or.ok()) {
          return mh_or.status();
        }
        data_mh_or = *mh_or;
      } else {
        auto mh_or = common::compute_data_multihash_from_gpu(gpu_ptr, entry->size_bytes, entry->device_id);
        if (!mh_or.ok()) {
          return mh_or.status();
        }
        data_mh_or = *mh_or;
      }
    }
    if (!data_mh_or.ok()) {
      return data_mh_or.status();
    }
    data_multihash = *data_mh_or;
    entry->artifact_id = absl::StrCat("mi2:", index_multihash, ":", data_multihash);
    entry->id_kind = common::ArtifactIdKind::kMi2;
  } else {
    entry->artifact_id = entry->client_artifact_id;
    entry->id_kind = common::ArtifactIdKind::kCgid;
    index_multihash.clear();
    data_multihash.clear();
  }

  DeviceKey dev_key = entry->plan == PendingRegistrationContext::Plan::kStableDram
      ? DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""}
      : DeviceRegistry::instance().gpu_key(entry->device_id);
  loading::ReplicaKey mi2_key{.artifact_id = entry->artifact_id, .device = dev_key, .replica = 0};
  const bool allow_idempotent = entry->view_state == nullptr && entry->id_kind == common::ArtifactIdKind::kMi2;
  bool reuse_existing = false;
  if (allow_idempotent) {
    if (auto existing_or = replica_registry_->find(mi2_key); existing_or.ok()) {
      reuse_existing = true;
    }
  }
  if (!reuse_existing) {
    absl::Status emplace_status =
        replica_registry_->emplace(mi2_key, gsl::not_null<std::shared_ptr<replica::Replica>>{entry->replica});
    if (absl::IsAlreadyExists(emplace_status)) {
      if (allow_idempotent) {
        reuse_existing = true;
      } else {
        VLOG(1) << "mi2 mapping already present for artifact_id=" << entry->artifact_id;
      }
    } else if (!emplace_status.ok()) {
      return emplace_status;
    }
  }

  if (reuse_existing) {
    const auto location = entry->plan == PendingRegistrationContext::Plan::kStableDram
        ? common::memory::MemoryLocation::CPU
        : common::memory::MemoryLocation::GPU;
    release_replica_memory(entry->replica, location);
    size_t pending_size_after = 0;
    erase_pending(registration_id, &pending_size_after);
    record_pending_gauge(pending_size_after);

    RegistrationCommitResult result;
    result.registration_id = std::string(registration_id);
    result.artifact_id = entry->artifact_id;
    result.device_id = entry->device_id;
    result.device = dev_key;
    result.size_bytes = entry->size_bytes;
    result.existed = true;
    result.index_multihash = index_multihash;
    result.data_multihash = data_multihash;
    result.schema_version = entry->schema_version;
    result.encoding = entry->encoding;
    result.id_kind = entry->id_kind;
    record_commit_latency(*entry, "existed");
    return result;
  }

  if (entry->plan == PendingRegistrationContext::Plan::kStableDram && entry->stable_cache_policy.has_value() &&
      stable_cache_manager_) {
    components::StableDramCacheManager::AdmissionRequest admit_request;
    admit_request.key = mi2_key;
    admit_request.replica = entry->replica;
    admit_request.size_bytes = entry->size_bytes;
    admit_request.policy = *entry->stable_cache_policy;
    auto admit_or = stable_cache_manager_->admit(admit_request);
    if (!admit_or.ok()) {
      release_replica_memory(entry->replica, common::memory::MemoryLocation::CPU);
      return admit_or.status();
    }
  }

  std::vector<std::string> remote_keys;
  std::vector<uint64_t> buffer_sizes;
  if (entry->enable_p2p && communication_manager_ && communication_manager_->is_enabled()) {
    const auto location = entry->plan == PendingRegistrationContext::Plan::kStableDram
        ? common::memory::MemoryLocation::CPU
        : common::memory::MemoryLocation::GPU;
    auto reg_info_or = entry->replica->enable_remote_memory_access(location, communication_manager_->get_engine());
    if (!reg_info_or.ok()) {
      return reg_info_or.status();
    }
    remote_keys = reg_info_or->remote_memory_keys;
    buffer_sizes.reserve(reg_info_or->buffer_sizes.size());
    for (const auto& sz : reg_info_or->buffer_sizes) {
      buffer_sizes.push_back(static_cast<uint64_t>(sz));
    }
  }

  DeviceKey device = entry->plan == PendingRegistrationContext::Plan::kStableDram
      ? DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""}
      : DeviceRegistry::instance().gpu_key(entry->device_id);
  std::optional<std::string> verification_json;
  if (!remote_keys.empty()) {
    const auto location = entry->plan == PendingRegistrationContext::Plan::kStableDram
        ? common::memory::MemoryLocation::CPU
        : common::memory::MemoryLocation::GPU;
    std::vector<void*> data_ptrs = entry->replica->get_memory_manager().get_pointer(location);
    if (!data_ptrs.empty() && data_ptrs[0] != nullptr) {
      std::vector<size_t> data_sizes{static_cast<size_t>(entry->size_bytes)};
      const int verify_device = (location == common::memory::MemoryLocation::GPU) ? entry->device_id : -1;
      auto info_or = common::ArtifactVerifier::generate_verification_info(
          data_ptrs, data_sizes, verify_device, common::VerificationLevel::KEY_POINTS);
      if (info_or.ok()) {
        verification_json = info_or->to_json();
      }
    }
  }

  if (publisher_) {
    RegistrationPublication publication{
        .artifact_id = entry->artifact_id,
        .device = device,
        .size_bytes = entry->size_bytes,
        .tensor_index_key = entry->tensor_index_key,
        .remote_memory_keys = remote_keys,
        .buffer_sizes = buffer_sizes,
        .tensor_index_data = entry->tensor_index_data,
        .encoding = entry->encoding,
        .schema_version = entry->schema_version,
        .verification_json = verification_json};
    absl::Status registration_status = publisher_->publish_registration(publication);
    if (!registration_status.ok()) {
      // GlobalStore not connected - skip publication in standalone mode.
      // Local registration remains valid; artifact is usable on this node.
      if (absl::IsFailedPrecondition(registration_status)) {
        LOG(INFO) << "Skipping GlobalStore publication (not connected): " << registration_status.message();
      } else {
        return registration_status;
      }
    }
  }

  std::optional<std::string> view_data_hash;
  std::optional<std::string> view_spec_json;
  std::vector<global_store::v1::LeafWrite> leaf_writes;
  std::vector<uint64_t> canonical_leaf_indices;
  std::optional<size_t> leaf_chunk_bytes;
  if (entry->view_state) {
    if (!segment_plan_ready && entry->tensor_index_data.has_value() && !entry->tensor_index_data->empty() &&
        entry->encoding == "json") {
      auto plan_or = loader::build_segment_plan_from_canonical_index_json(
          *entry->tensor_index_data, entry->size_bytes, /*align_bytes=*/8);
      if (plan_or.ok()) {
        segment_plan = std::move(*plan_or);
        segment_plan_ready = true;
      } else {
        LOG(WARNING) << "Failed to rebuild segment plan for view hash: " << plan_or.status();
      }
    }
    view_spec_json = view::build_view_spec_json(entry->view_state->options.spec);
    leaf_chunk_bytes = artifact_chunk_bytes_ == 0 ? static_cast<size_t>(4ULL * 1024 * 1024) : artifact_chunk_bytes_;

    if (segment_plan_ready && entry->view_state->plan.forward.selection.total_bytes > 0 &&
        leaf_chunk_bytes.has_value()) {
      void* gpu_ptr = entry->gpu_ptr;
      if (gpu_ptr == nullptr) {
        const auto ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
        gpu_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
      }
      if (gpu_ptr != nullptr) {
        loader::LinearizedGpuPlanSource canonical_source(
            gsl::not_null<void*>{gpu_ptr}, entry->device_id, absl::MakeSpan(segment_plan), entry->size_bytes);
        loader::ViewPlanSource view_source(
            gsl::not_null<loader::SeekableSource*>{&canonical_source}, entry->view_state->plan.forward.selection);
        auto view_hash_or = loader::verification::compute_view_tree_hash_and_leaves(
            view_source, entry->view_state->plan.forward.selection.total_bytes, *leaf_chunk_bytes);
        if (view_hash_or.ok()) {
          view_data_hash = view_hash_or->multihash;
          const auto& digests = view_hash_or->leaf_digests;
          leaf_writes.reserve(leaf_writes.size() + digests.size());
          for (size_t idx = 0; idx < digests.size(); ++idx) {
            global_store::v1::LeafWrite leaf;
            leaf.set_space_kind(tensorcast::global_store::v1::BYTE_SPACE_KIND_VARIANT);
            leaf.set_space_id(entry->view_state->options.view_id);
            leaf.set_leaf_idx(static_cast<uint64_t>(idx));
            const auto& digest = digests[idx];
            leaf.set_digest(digest.data(), static_cast<int>(digest.size()));
            leaf_writes.push_back(std::move(leaf));
          }
        } else {
          LOG(WARNING) << "ComputeTreeHashAndLeaves (view) failed: " << view_hash_or.status();
        }
      }
    }
    canonical_leaf_indices = view::compute_fully_covered_canonical_leaf_indices(
        entry->view_state->options.canonical_ranges, leaf_chunk_bytes.value_or(0));
  }

  if (entry->plan == PendingRegistrationContext::Plan::kStableDram && entry->stable_dram.stage_on_gpu &&
      entry->stable_dram.release_gpu_on_commit && entry->staging_gpu) {
    entry->staging_gpu.reset();
    entry->gpu_ptr = nullptr;
  }

  if (entry->id_kind == common::ArtifactIdKind::kMi2 && entry->view_state && leaf_chunk_bytes.has_value() &&
      !canonical_leaf_indices.empty() && !index_multihash.empty()) {
    loader::verification::MemoryView canonical_view;
    canonical_view.size_bytes = entry->size_bytes;
    if (entry->plan == PendingRegistrationContext::Plan::kStableDram) {
      canonical_view.location = common::memory::MemoryLocation::CPU;
      const auto cpu_ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::CPU);
      if (!cpu_ptrs.empty() && cpu_ptrs[0] != nullptr) {
        canonical_view.base_ptr = const_cast<void*>(cpu_ptrs[0]);
      }
    } else {
      canonical_view.location = common::memory::MemoryLocation::GPU;
      canonical_view.gpu_device_id = entry->device_id;
      void* gpu_ptr = entry->gpu_ptr;
      if (gpu_ptr == nullptr) {
        const auto ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
        gpu_ptr = (!ptrs.empty() ? ptrs[0] : nullptr);
      }
      canonical_view.base_ptr = gpu_ptr;
    }

    auto canonical_digest_or = loader::verification::compute_canonical_leaf_digests(
        canonical_view, absl::Span<const uint64_t>(canonical_leaf_indices), *leaf_chunk_bytes);
    if (!canonical_digest_or.ok()) {
      LOG(WARNING) << "Failed to compute canonical leaf digests for view registration: "
                   << canonical_digest_or.status();
    } else if (canonical_digest_or->size() != canonical_leaf_indices.size()) {
      LOG(WARNING) << "Canonical leaf digest count mismatch: expected " << canonical_leaf_indices.size() << " got "
                   << canonical_digest_or->size();
    } else {
      leaf_writes.reserve(leaf_writes.size() + canonical_leaf_indices.size());
      for (size_t i = 0; i < canonical_leaf_indices.size(); ++i) {
        global_store::v1::LeafWrite leaf;
        leaf.set_space_kind(tensorcast::global_store::v1::BYTE_SPACE_KIND_CANONICAL);
        leaf.set_space_id(index_multihash);
        leaf.set_leaf_idx(canonical_leaf_indices[i]);
        const auto& digest = (*canonical_digest_or)[i];
        leaf.set_digest(digest.data(), static_cast<int>(digest.size()));
        leaf_writes.push_back(std::move(leaf));
      }
    }
  }

  size_t pending_size_after = 0;
  erase_pending(registration_id, &pending_size_after);
  record_pending_gauge(pending_size_after);

  RegistrationCommitResult result;
  result.registration_id = std::string(registration_id);
  result.artifact_id = entry->artifact_id;
  result.device_id = entry->device_id;
  result.device = device;
  result.size_bytes = entry->size_bytes;
  result.existed = false;
  result.index_multihash = index_multihash;
  result.data_multihash = data_multihash;
  result.schema_version = entry->schema_version;
  result.encoding = entry->encoding;
  result.id_kind = entry->id_kind;
  uint64_t covered_bytes = 0;
  if (entry->view_state) {
    if (!entry->view_state->options.view_id.empty()) {
      result.view_id = entry->view_state->options.view_id;
    }
    if (view_data_hash.has_value()) {
      result.view_data_multihash = view_data_hash;
    }
    if (!entry->view_state->plan.forward.view_index_json.empty()) {
      result.view_index_json = entry->view_state->plan.forward.view_index_json;
    }
    result.canonical_ranges = entry->view_state->options.canonical_ranges;
    result.allow_partial = entry->view_state->options.allow_partial;
    for (const auto& range : result.canonical_ranges) {
      covered_bytes += range.length;
    }
  }

  if (entry->view_state && !entry->view_state->options.view_id.empty() && publisher_) {
    components::VariantViewUpdate update;
    update.artifact_id = entry->artifact_id;
    update.view_id = entry->view_state->options.view_id;
    update.view_spec_json = view_spec_json.value_or(view::build_view_spec_json(entry->view_state->options.spec));
    update.view_size_bytes = entry->view_state->plan.forward.view_size_bytes;
    update.view_data_hash = view_data_hash;
    update.mark_verified = true;
    update.canonical_size_bytes = entry->size_bytes;
    update.canonical_bytes_covered = covered_bytes;
    update.leaf_writes = std::move(leaf_writes);
    absl::Status update_status = publisher_->update_variant_view(update);
    if (!update_status.ok()) {
      LOG(WARNING) << "UpdateArtifactViewState failed for artifact " << entry->artifact_id
                   << " view_id=" << entry->view_state->options.view_id << ": " << update_status;
    }
  }
  {
    const auto location = entry->plan == PendingRegistrationContext::Plan::kStableDram
        ? common::memory::MemoryLocation::CPU
        : common::memory::MemoryLocation::GPU;
    auto mark_status = entry->replica->mark_loaded(location);
    if (!mark_status.ok()) {
      return mark_status;
    }
    entry->replica->set_ready_signal(location, absl::OkStatus());
  }
  record_commit_latency(*entry, "ok");
  return result;
}

absl::Status RegistrationBackend::ingest_view_chunk(
    std::string_view registration_id,
    uint64_t view_offset,
    absl::Span<const std::byte> data) {
  if (data.empty()) {
    return absl::OkStatus();
  }
  auto entry = lookup_pending(registration_id);
  if (!entry) {
    return absl::NotFoundError("registration_id not found");
  }
  if (!entry->view_state) {
    return absl::FailedPreconditionError("registration is not view-enabled");
  }
  auto& view_state = *entry->view_state;
  if (view_state.options.placement != ViewPlacement::kServer) {
    return absl::FailedPreconditionError("view chunk ingestion is only valid for SERVER placement");
  }
  if (!view_state.executor) {
    return absl::FailedPreconditionError("view executor missing for SERVER placement ingestion");
  }
  void* canonical_base = entry->gpu_ptr;
  if (canonical_base == nullptr) {
    const auto ptrs = entry->replica->get_memory_manager().get_pointer(common::memory::MemoryLocation::GPU);
    canonical_base = (!ptrs.empty() ? ptrs[0] : nullptr);
    if (canonical_base == nullptr) {
      return absl::FailedPreconditionError("GPU pointer unavailable for view ingestion");
    }
    entry->gpu_ptr = canonical_base;
  }
  auto status = view_state.executor->ingest_chunk(
      view_offset, data, common::memory::MemoryLocation::GPU, canonical_base, entry->device_id);
  if (!status.ok()) {
    return status;
  }
  view_state.ingested_bytes = view_state.executor->ingested_bytes();
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> RegistrationBackend::get_view_ingested_bytes(std::string_view registration_id) const {
  auto entry = lookup_pending(registration_id);
  if (!entry) {
    return absl::NotFoundError("registration_id not found");
  }
  if (!entry->view_state) {
    return absl::FailedPreconditionError("registration has no view state");
  }
  if (entry->view_state->executor) {
    return entry->view_state->executor->ingested_bytes();
  }
  return entry->view_state->expected_view_bytes;
}

absl::StatusOr<uint64_t> RegistrationBackend::get_registration_gpu_ptr(std::string_view registration_id) const {
  auto entry = lookup_pending(registration_id);
  if (!entry) {
    return absl::NotFoundError("registration_id not found");
  }
  if (entry->gpu_ptr == nullptr) {
    return absl::FailedPreconditionError("registration has no GPU pointer");
  }
  return reinterpret_cast<uint64_t>(entry->gpu_ptr);
}

absl::Status RegistrationBackend::abort(std::string_view registration_id) {
  size_t pending_size_after = 0;
  auto entry = erase_pending(registration_id, &pending_size_after);
  if (!entry) {
    return absl::NotFoundError("registration_id not found");
  }
  record_pending_gauge(pending_size_after);
  record_commit_latency(*entry, "aborted");
  const auto location = entry->plan == PendingRegistrationContext::Plan::kStableDram
      ? common::memory::MemoryLocation::CPU
      : common::memory::MemoryLocation::GPU;
  release_replica_memory(entry->replica, location);
  return absl::OkStatus();
}

absl::Status RegistrationBackend::keep_alive(std::string_view registration_id, uint32_t ttl_ms) {
  if (ttl_ms == 0) {
    return absl::OkStatus();
  }
  std::lock_guard<std::mutex> lock(pending_mutex_);
  auto it = pending_regs_.find(std::string(registration_id));
  if (it == pending_regs_.end()) {
    return absl::NotFoundError("registration_id not found");
  }
  it->second->expiry_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(ttl_ms);
  return absl::OkStatus();
}

std::shared_ptr<RegistrationBackend::PendingRegistrationContext> RegistrationBackend::erase_pending(
    std::string_view registration_id,
    size_t* pending_size_after) {
  std::shared_ptr<PendingRegistrationContext> entry;
  size_t new_size = 0;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_regs_.find(std::string(registration_id));
    if (it == pending_regs_.end()) {
      new_size = pending_regs_.size();
    } else {
      entry = it->second;
      pending_regs_.erase(it);
      new_size = pending_regs_.size();
    }
  }
  if (pending_size_after != nullptr) {
    *pending_size_after = new_size;
  }
  return entry;
}

std::shared_ptr<RegistrationBackend::PendingRegistrationContext> RegistrationBackend::lookup_pending(
    std::string_view registration_id) const {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  auto it = pending_regs_.find(std::string(registration_id));
  if (it == pending_regs_.end()) {
    return nullptr;
  }
  return it->second;
}

void RegistrationBackend::release_replica_memory(
    const std::shared_ptr<replica::Replica>& replica,
    common::memory::MemoryLocation location) {
  if (!replica) {
    return;
  }
  absl::Status st = replica->release_memory(location);
  if (!st.ok()) {
    VLOG(1) << "release_memory(" << static_cast<int>(location) << ") failed: " << st;
  }
}

void RegistrationBackend::record_pending_gauge(size_t pending_count) const {
  metrics_collector_->record_registration_pending(pending_count);
}

void RegistrationBackend::record_commit_latency(const PendingRegistrationContext& ctx, std::string_view status) const {
  if (ctx.begin_time.time_since_epoch().count() == 0) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  const double duration_seconds = std::chrono::duration<double>(now - ctx.begin_time).count();
  metrics_collector_->record_registration_commit(duration_seconds, status);
}

} // namespace tensorcast::store::runtime::metadata
