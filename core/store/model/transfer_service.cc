// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/model/transfer_service.h"

#include <algorithm>
#include <utility>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "core/store/loader/dvmp_region_sink.h"
#include "core/store/loader/file_partition_source.h"
#include "core/store/loader/gpu_memory_sink.h"
#include "core/store/loader/multi_safetensors_source.h"
#include "core/store/loader/pump.h"
#include "core/store/loader/safetensors_source.h"
#include "core/store/loader/streaming_buffer_adapter.h"
#include "core/store/model/transfer_helpers.h"

namespace stepcast::store {

namespace {
// Per-GPU (device_id) concurrency limiter: at most 1 active session per GPU.
ABSL_CONST_INIT absl::Mutex g_gpu_limit_mu(absl::kConstInit);
struct GpuGate {
  bool active{false};
  absl::CondVar cv;
};
std::unordered_map<int, GpuGate> g_gpu_gates ABSL_GUARDED_BY(g_gpu_limit_mu);
} // namespace

TransferService::TransferService(
    const gsl::not_null<std::shared_ptr<PinnedMemoryPool>>& pinned_pool,
    const gsl::not_null<std::shared_ptr<memory::DistributedVirtualMemoryPool>>& dvmp,
    const gsl::not_null<std::shared_ptr<ModelMemoryCoordinator>>& uma,
    InstanceKey instance_key,
    Config cfg)
    : pinned_pool_(pinned_pool),
      dvmp_(dvmp),
      uma_(uma),
      instance_key_(std::move(instance_key)),
      cfg_(cfg),
      spb_(std::make_shared<StreamingPinnedBuffer>(/*num_chunks=*/16, pinned_pool_->chunk_size(), pinned_pool_)) {}

size_t TransferService::get_pool_chunk_size() const {
  return pinned_pool_->chunk_size();
}

// ScopedGpuPermit implementation
TransferService::ScopedGpuPermit::ScopedGpuPermit(int device_id) : device_id_(device_id) {
  absl::MutexLock lock(&g_gpu_limit_mu);
  GpuGate& gate = g_gpu_gates[device_id_];
  while (gate.active) {
    gate.cv.Wait(&g_gpu_limit_mu);
  }
  gate.active = true;
}

TransferService::ScopedGpuPermit::~ScopedGpuPermit() {
  absl::MutexLock lock(&g_gpu_limit_mu);
  auto it = g_gpu_gates.find(device_id_);
  if (it != g_gpu_gates.end()) {
    it->second.active = false;
    it->second.cv.Signal();
  }
}

absl::Status TransferService::copy_cpu_to_gpu_streaming(
    uint32_t device_id,
    cudaStream_t stream,
    void* gpu_ptr,
    size_t total_bytes) {
  // Validate parameters first
  if (!gpu_ptr) {
    return absl::InvalidArgumentError("GPU pointer is null");
  }
  if (total_bytes == 0) {
    return absl::InvalidArgumentError("Total bytes must be greater than 0");
  }

  void* dvmp_base = uma_->get_cpu_base_ptr(instance_key_);
  if (!dvmp_base) {
    return absl::FailedPreconditionError("DVMP base not available via UMA");
  }
  // Acquire per-GPU permit (1 active session per GPU)
  ScopedGpuPermit permit(static_cast<int>(device_id));
  // Create a per-session streaming buffer backed by the shared pinned pool
  auto session_spb = std::make_shared<StreamingPinnedBuffer>(
      /*num_chunks=*/16, get_pool_chunk_size(), pinned_pool_);
  auto init_status = session_spb->initialize(cfg_.pinned_memory_timeout);
  if (!init_status.ok()) {
    return init_status;
  }
  return perform_copy_cpu_to_gpu_streaming(
      instance_key_.model_id,
      device_id,
      session_spb,
      gpu_ptr,
      total_bytes,
      stream,
      dvmp_base,
      dvmp_,
      uma_,
      instance_key_);
}

absl::Status TransferService::copy_gpu_to_cpu_streaming(
    uint32_t device_id,
    cudaStream_t stream,
    void* gpu_ptr,
    size_t total_bytes) {
  // Validate parameters first
  if (!gpu_ptr) {
    return absl::InvalidArgumentError("GPU pointer is null");
  }
  if (total_bytes == 0) {
    return absl::InvalidArgumentError("Total bytes must be greater than 0");
  }

  auto spb = get_streaming_buffer();
  if (!spb) {
    return absl::FailedPreconditionError("Streaming buffer not available");
  }
  void* dvmp_base = uma_->get_cpu_base_ptr(instance_key_);
  if (!dvmp_base) {
    return absl::FailedPreconditionError("DVMP base not available via UMA");
  }
  return perform_copy_gpu_to_cpu_streaming(
      instance_key_.model_id, device_id, spb, gpu_ptr, total_bytes, stream, dvmp_base, dvmp_);
}

std::unique_ptr<loader::PositionedSink> TransferService::build_sink_(
    ModelLocation target_location,
    void* gpu_ptr,
    int device_id) {
  if (target_location == ModelLocation::GPU) {
    auto sink = std::make_unique<loader::GPUMemorySink>(loader::GPUMemorySink::Options{
        .gpu_base_ptr = gpu_ptr,
        .total_size = uma_->get_model_size(instance_key_).ok() ? *uma_->get_model_size(instance_key_) : 0,
        .chunk_size = get_pool_chunk_size(),
        .device_id = device_id});
    return sink;
  }
  // CPU sink writes into DVMP region via PositionedSink
  auto region_or = dvmp_->open(instance_key_.model_id);
  if (!region_or.ok()) {
    return nullptr;
  }
  loader::DVMPRegionSink::Options opts;
  opts.region = *region_or;
  opts.total_size = uma_->get_model_size(instance_key_).ok() ? *uma_->get_model_size(instance_key_) : 0;
  opts.plan_direct_write_fn =
      [uma = uma_, key = instance_key_](absl::Span<const VaRange> ranges) -> absl::StatusOr<DirectWriteToken> {
    return uma->create_direct_write_token(key, ranges);
  };
  auto sink = std::make_unique<loader::DVMPRegionSink>(std::move(opts));
  return sink;
}

std::vector<std::pair<uint64_t, size_t>> TransferService::build_ranges_(
    std::optional<absl::Span<const uint32_t>> chunk_indices,
    size_t chunk_size,
    uint64_t total_bytes) {
  std::vector<std::pair<uint64_t, size_t>> ranges;
  if (!chunk_indices.has_value() || chunk_indices->empty()) {
    ranges.emplace_back(0ULL, total_bytes);
    return ranges;
  }

  // Remove duplicates and sort in one pass
  std::vector<uint32_t> sorted(chunk_indices->begin(), chunk_indices->end());
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  uint32_t run_start = sorted.front();
  uint32_t prev = run_start;
  for (size_t i = 1; i < sorted.size(); ++i) {
    const uint32_t idx = sorted[i];
    if (idx == prev + 1) {
      prev = idx;
      continue;
    }
    const uint64_t off = static_cast<uint64_t>(run_start) * chunk_size;
    const uint64_t end = static_cast<uint64_t>(prev + 1) * chunk_size;
    const uint64_t len64 = (end > total_bytes) ? (total_bytes - off) : (end - off);
    const size_t len = static_cast<size_t>(len64);
    ranges.emplace_back(off, len);
    run_start = prev = idx;
  }
  const uint64_t off = static_cast<uint64_t>(run_start) * chunk_size;
  const uint64_t end = static_cast<uint64_t>(prev + 1) * chunk_size;
  const uint64_t len64 = (end > total_bytes) ? (total_bytes - off) : (end - off);
  const auto len = static_cast<size_t>(len64);
  ranges.emplace_back(off, len);
  return ranges;
}

absl::Status TransferService::load_from_source(
    std::unique_ptr<loader::SeekableSource>& source,
    ModelLocation target_location,
    int concurrency,
    std::optional<absl::Span<const uint32_t>> chunk_indices,
    void* gpu_ptr_or_null,
    int device_id) {
  // Validate parameters
  if (!source) {
    return absl::InvalidArgumentError("Source is null");
  }
  if (concurrency <= 0) {
    return absl::InvalidArgumentError("Concurrency must be positive");
  }
  if (target_location == ModelLocation::GPU && !gpu_ptr_or_null) {
    return absl::InvalidArgumentError("GPU pointer required for GPU target location");
  }

  const size_t chunk_size = get_pool_chunk_size();
  uint64_t total_size = 0;
  auto sz = uma_->get_model_size(instance_key_);
  if (sz.ok()) {
    total_size = *sz;
  }
  // Fallback: use source total size when UMA doesn't know
  if (total_size == 0) {
    if (auto* fps = dynamic_cast<loader::FilePartitionSource*>(source.get())) {
      total_size = fps->total_size();
    }
    if (total_size == 0) {
      if (auto* ss = dynamic_cast<loader::SafetensorsSource*>(source.get())) {
        total_size = ss->total_size();
      } else if (auto* ms = dynamic_cast<loader::MultiSafetensorsSource*>(source.get())) {
        total_size = ms->total_size();
      }
    }
  }

  // Acquire per-GPU permit (1 active session per GPU)
  ScopedGpuPermit permit(device_id);

  // Create a per-session streaming buffer backed by the shared pinned pool
  auto session_spb = std::make_shared<StreamingPinnedBuffer>(
      /*num_chunks=*/16, get_pool_chunk_size(), pinned_pool_);
  auto init_status = session_spb->initialize(cfg_.pinned_memory_timeout);
  if (!init_status.ok()) {
    return init_status;
  }

  loader::StreamingBufferAdapter adapter(session_spb);
  auto sink = build_sink_(target_location, gpu_ptr_or_null, device_id);
  if (!sink) {
    return absl::FailedPreconditionError("Failed to construct sink for target location");
  }

  auto ranges = build_ranges_(chunk_indices, chunk_size, total_size);
  absl::Status pump_status = loader::pump_ranges(*source, *sink, adapter, absl::MakeSpan(ranges), concurrency);
  if (!pump_status.ok()) {
    return pump_status;
  }
  (void)sink->close();
  return absl::OkStatus();
}

} // namespace stepcast::store