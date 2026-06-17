// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/replica/collective_disk_loader.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nccl.h>
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "core/common/async_runtime.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/error_handling.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/runtime/pump.h"
#include "core/store/materialization/dataplane/runtime/streaming_buffer_adapter.h"
#include "core/store/materialization/dataplane/sinks/gpu_memory_sink.h"
#include "core/store/materialization/dataplane/sinks/target_layout_gpu_sink.h"
#include "core/store/materialization/dataplane/sources/multi_safetensors_source.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::replica {

namespace {

#define TC_RETURN_IF_ERROR(expr)     \
  do {                               \
    ::absl::Status _status = (expr); \
    if (!_status.ok()) {             \
      return _status;                \
    }                                \
  } while (false)

#define TC_ASSIGN_OR_RETURN(lhs, expr)   \
  do {                                   \
    auto _status_or = (expr);            \
    if (!_status_or.ok()) {              \
      return _status_or.status();        \
    }                                    \
    lhs = std::move(_status_or).value(); \
  } while (false)

constexpr std::chrono::milliseconds kGroupAssembleTimeout{15000};
// The mapped collective path can emit millions of tiny peer pieces. Keeping the
// NCCL group cap too low forces thousands of group flushes and full
// synchronize_all() barriers. 8192 materially reduces barrier count while
// remaining a bounded batch size.
constexpr size_t kMaxMappedPeerPairsPerNcclGroup = 8192;

using StrategyConfig = StoreEngineOptions::MaterializationStrategyConfig;
using RepresentationWorkItem = materialization::contracts::RepresentationWorkItem;
using RepresentationWorkItemKind = materialization::contracts::RepresentationWorkItemKind;
using RepresentationWorkPlan = materialization::contracts::RepresentationWorkPlan;
using RepresentationWorkSourceFragment = materialization::contracts::RepresentationWorkSourceFragment;
using RepresentationTensorBinding = materialization::contracts::RepresentationTensorBinding;
using RepresentationTensorSpec = materialization::contracts::RepresentationTensorSpec;
using TensorByteSpan = materialization::contracts::TensorByteSpan;
using RepresentationTransformContract = materialization::contracts::RepresentationTransformContract;
using SourceFragment = materialization::contracts::SourceFragment;
using TensorAxisRange = materialization::contracts::TensorAxisRange;
using TensorCoordinateSpec = materialization::contracts::TensorCoordinateSpec;
using WorkPartitionKind = materialization::contracts::WorkPartitionKind;

bool enable_mapped_tensor_job_fast_path(const StrategyConfig& strategy) {
  return strategy.enable_tensor_aware_mapped_executor && strategy.allow_mixed_execution;
}

bool enable_mapped_dim0_tensor_jobs(const StrategyConfig& strategy) {
  return strategy.enable_mapped_dim0_tensor_jobs;
}

bool enable_mapped_dim1_tensor_jobs(const StrategyConfig& strategy) {
  return strategy.enable_mapped_dim1_tensor_jobs;
}

bool enable_mapped_concat_jobs(const StrategyConfig& strategy) {
  return strategy.enable_mapped_concat_jobs;
}

bool enable_mapped_concat_execution(const StrategyConfig& strategy) {
  return strategy.enable_mapped_concat_execution;
}

bool allow_source_ordered_for_mapped(const StrategyConfig& strategy) {
  return strategy.allow_source_ordered_for_mapped;
}

bool enable_mapped_multirange_concat_jobs(const StrategyConfig& strategy) {
  return strategy.enable_mapped_multirange_concat_jobs;
}

bool enable_mapped_single_range_concat_jobs(const StrategyConfig& strategy) {
  return strategy.enable_mapped_single_range_concat_jobs;
}

bool enable_collective_owner_file_strategy(const StrategyConfig& strategy) {
  return strategy.enable_owner_file_collective;
}

bool enable_local_batched_disk_load(const StrategyConfig& strategy) {
  return strategy.enable_local_batched_disk_load;
}

bool verbose_mapped_concat_diagnostics(const StrategyConfig& strategy) {
  return strategy.diagnostics_verbosity == StrategyConfig::DiagnosticsVerbosity::kVerbose;
}

bool sync_after_single_range_concat_job(const StrategyConfig& strategy) {
  return strategy.sync_after_single_range_concat_job;
}

bool use_dedicated_single_range_concat_stream(const StrategyConfig& strategy) {
  return strategy.use_dedicated_single_range_concat_stream;
}

std::chrono::milliseconds collective_group_assemble_timeout(const StrategyConfig& strategy) {
  return strategy.owner_file_collective_group_assemble_timeout.count() > 0
      ? strategy.owner_file_collective_group_assemble_timeout
      : kGroupAssembleTimeout;
}

LocalBatchedDiskLoadResult local_batched_fallback(
    std::string_view artifact_id,
    std::string_view reason,
    std::optional<absl::Status> detail = std::nullopt) {
  LOG(WARNING) << "LOCAL_BATCHED_DISK_LOAD_FALLBACK artifact_id=" << artifact_id << " reason=" << reason
               << (detail.has_value() ? absl::StrCat(" status=", detail->ToString()) : "");
  return {.handled = false, .status = absl::OkStatus(), .skip_reason = std::string(reason)};
}

absl::Mutex g_peer_copy_mu;
absl::flat_hash_map<uint64_t, bool> g_peer_copy_capable ABSL_GUARDED_BY(g_peer_copy_mu);

bool ensure_peer_copy_capable(int src_device, int dst_device) {
  if (src_device == dst_device) {
    return true;
  }
  const uint64_t key =
      (static_cast<uint64_t>(static_cast<uint32_t>(src_device)) << 32) | static_cast<uint32_t>(dst_device);
  {
    absl::MutexLock lock(&g_peer_copy_mu);
    auto it = g_peer_copy_capable.find(key);
    if (it != g_peer_copy_capable.end()) {
      return it->second;
    }
  }

  int can_access = 0;
  bool capable = false;
  if (tensorcast::cuda::device_can_access_peer(&can_access, dst_device, src_device).ok() && can_access != 0) {
    const absl::Status enable_dst = tensorcast::cuda::enable_peer_access(dst_device, src_device);
    if (!enable_dst.ok()) {
      VLOG(1) << "enable_peer_access(dst,src) failed: " << enable_dst;
    }
    const absl::Status enable_src = tensorcast::cuda::enable_peer_access(src_device, dst_device);
    if (!enable_src.ok()) {
      VLOG(1) << "enable_peer_access(src,dst) failed: " << enable_src;
    }
    capable = true;
  }

  {
    absl::MutexLock lock(&g_peer_copy_mu);
    g_peer_copy_capable[key] = capable;
  }
  return capable;
}

absl::Status nccl_status(ncclResult_t rc, std::string_view what) {
  if (rc == ncclSuccess) {
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrCat(what, ": ", ncclGetErrorString(rc)));
}

class NcclClique {
 public:
  struct RankCtx {
    int device_id{-1};
    cudaStream_t stream{nullptr};
    ncclComm_t comm{nullptr};
  };

  ~NcclClique() {
    for (auto& rank : ranks_) {
      if (rank.stream != nullptr) {
        (void)tensorcast::cuda::set_device(rank.device_id);
        (void)tensorcast::cuda::stream_destroy(rank.stream);
      }
      if (rank.comm != nullptr) {
        (void)ncclCommDestroy(rank.comm);
      }
    }
  }

  static absl::StatusOr<std::shared_ptr<NcclClique>> create(const std::vector<int>& device_ids) {
    if (device_ids.size() <= 1) {
      return absl::InvalidArgumentError("NcclClique requires at least 2 devices");
    }
    auto clique = std::shared_ptr<NcclClique>(new NcclClique());
    clique->ranks_.resize(device_ids.size());
    for (size_t i = 0; i < device_ids.size(); ++i) {
      clique->ranks_[i].device_id = device_ids[i];
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_ids[i]));
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&clique->ranks_[i].stream, cudaStreamNonBlocking));
    }
    std::vector<ncclComm_t> comms(device_ids.size(), nullptr);
    TC_RETURN_IF_ERROR(nccl_status(
        ncclCommInitAll(comms.data(), static_cast<int>(device_ids.size()), device_ids.data()), "ncclCommInitAll"));
    for (size_t i = 0; i < comms.size(); ++i) {
      clique->ranks_[i].comm = comms[i];
    }
    return clique;
  }

  int world_size() const {
    return static_cast<int>(ranks_.size());
  }

  int device_id(int rank) const {
    return ranks_[static_cast<size_t>(rank)].device_id;
  }

  cudaStream_t stream(int rank) const {
    return ranks_[static_cast<size_t>(rank)].stream;
  }

  absl::Status wait_stream_on_event(int rank, cudaEvent_t event) {
    return tensorcast::cuda::stream_wait_event(stream(rank), event);
  }

  absl::Status group_start() {
    return nccl_status(ncclGroupStart(), "ncclGroupStart");
  }

  absl::Status group_end() {
    return nccl_status(ncclGroupEnd(), "ncclGroupEnd");
  }

  absl::Status broadcast_u8(int rank, const void* send, void* recv, size_t bytes, int root_rank) {
    return nccl_status(
        ncclBroadcast(send, recv, bytes, ncclUint8, root_rank, ranks_[static_cast<size_t>(rank)].comm, stream(rank)),
        "ncclBroadcast");
  }

  absl::Status send_u8(int rank, const void* src, size_t bytes, int peer_rank) {
    return nccl_status(
        ncclSend(src, bytes, ncclUint8, peer_rank, ranks_[static_cast<size_t>(rank)].comm, stream(rank)), "ncclSend");
  }

  absl::Status recv_u8(int rank, void* dst, size_t bytes, int peer_rank) {
    return nccl_status(
        ncclRecv(dst, bytes, ncclUint8, peer_rank, ranks_[static_cast<size_t>(rank)].comm, stream(rank)), "ncclRecv");
  }

  absl::Status synchronize_all() {
    for (const auto& rank : ranks_) {
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(rank.device_id));
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(rank.stream));
    }
    return absl::OkStatus();
  }

  absl::Mutex& use_mutex() {
    return use_mu_;
  }

 private:
  NcclClique() = default;

  std::vector<RankCtx> ranks_;
  absl::Mutex use_mu_;
};

struct TensorMeta {
  uint64_t offset{0};
  uint64_t size_bytes{0};
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::string dtype;
  uint64_t storage_offset{0};
  uint64_t elem_size{0};
};

absl::StatusOr<uint64_t> source_base_offset_bytes(const TensorMeta& source) {
  if (source.storage_offset > 0 && source.elem_size > std::numeric_limits<uint64_t>::max() / source.storage_offset) {
    return absl::OutOfRangeError("source storage_offset byte conversion overflows");
  }
  const uint64_t storage_offset_bytes = source.storage_offset * source.elem_size;
  if (source.offset > std::numeric_limits<uint64_t>::max() - storage_offset_bytes) {
    return absl::OutOfRangeError("source base offset overflows");
  }
  return source.offset + storage_offset_bytes;
}

struct RankTensorSlice {
  enum class Kind : uint8_t { kFull = 0, kDim0 = 1, kDim1 = 2, kRect2D = 3 };

  uint64_t dst_offset{0};
  uint64_t dst_size_bytes{0};
  uint64_t dst_row_stride_bytes{0};
  Kind kind{Kind::kFull};
  int64_t start{0};
  uint64_t length{0};
  uint64_t row_start{0};
  uint64_t row_count{0};
  uint64_t src_col_start{0};
  uint64_t col_count{0};
};

struct TensorJob {
  enum class Distribution : uint8_t { kReplicated = 0, kDim0Partitioned = 1, kDim1Partitioned = 2 };

  std::string name;
  TensorMeta source;
  Distribution distribution{Distribution::kReplicated};
  std::vector<RankTensorSlice> slices;
};

struct ParsedParticipant {
  loading::ReplicaKey replica_key;
  int rank{-1};
  int device_id{-1};
  void* gpu_ptr{nullptr};
  std::shared_ptr<common::memory::GpuDeviceMemory> gpu_allocation;
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  RepresentationWorkPlan representation_work_plan;
};

struct GroupState {
  explicit GroupState(uint32_t size) : world_size(size), participants(size) {}

  const uint32_t world_size;
  absl::Mutex mu;
  absl::CondVar cv;
  std::vector<std::optional<ParsedParticipant>> participants ABSL_GUARDED_BY(mu);
  uint32_t joined ABSL_GUARDED_BY(mu){0};
  bool launching ABSL_GUARDED_BY(mu){false};
  bool complete ABSL_GUARDED_BY(mu){false};
  absl::Status status ABSL_GUARDED_BY(mu){absl::OkStatus()};
  runtime::ingestion::strategy::CollectiveExecutionMetrics metrics ABSL_GUARDED_BY(mu);
};

struct TargetStorageSpan {
  uint64_t base_offset{0};
  uint64_t length{0};
  gsl::not_null<std::uint8_t*> base_ptr{reinterpret_cast<std::uint8_t*>(1)};
};

struct MappedSegmentRef {
  int rank{-1};
  uint64_t src_offset{0};
  uint64_t dst_offset{0};
  uint64_t length{0};
};

struct ByteRange {
  uint64_t begin{0};
  uint64_t end{0};
};

struct MappedSourceWindow {
  uint64_t start{0};
  uint64_t end{0};
  uint64_t covered_bytes{0};
  std::vector<size_t> segment_indices;
};

struct ChunkReadPlan {
  uint64_t start{0};
  uint64_t end{0};
  size_t length{0};
  const MappedSourceWindow* window{nullptr};
};

struct ParsedMappedParticipant {
  std::string artifact_id;
  int rank{-1};
  int device_id{-1};
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  RepresentationWorkPlan work_plan;
  loader::ByteRangeMap collective_lane_map;
  loading::IntoTargetLayout target_layout;
  std::vector<TargetStorageSpan> storage_spans;
};

struct MappedTensorJobRuntime {
  TensorJob job;
  std::vector<ParsedParticipant> destinations;
};

struct MappedTensorJobBuildResult {
  std::vector<MappedTensorJobRuntime> jobs;
  std::vector<std::vector<ByteRange>> handled_dst_ranges_by_rank;
  uint64_t handled_source_bytes{0};
  uint64_t handled_root_dst_bytes{0};
};

struct ConcatBlockPieceRuntime {
  uint64_t block_offset{0};
  uint64_t length{0};
  std::uint8_t* dst_ptr{nullptr};
};

struct MappedConcatFragmentRuntime {
  TensorMeta source;
  int64_t src_start{0};
  int64_t src_end{0};
  std::vector<int64_t> src_starts_by_rank;
  std::vector<int64_t> src_ends_by_rank;
  uint64_t prefix_count{0};
  uint64_t src_block_bytes{0};
  uint64_t dst_block_offset_bytes{0};
  uint64_t dst_block_stride_bytes{0};
  uint64_t dst_block_bytes{0};
  std::vector<uint64_t> dst_logical_begins_by_rank;
  std::vector<void*> dst_ptrs;
  std::vector<std::vector<std::vector<ConcatBlockPieceRuntime>>> dst_block_pieces_by_rank;
};

struct MappedConcatJobRuntime {
  std::string name;
  uint64_t dst_base_offset{0};
  uint64_t dst_size_bytes{0};
  uint64_t prefix_count{0};
  std::vector<ParsedParticipant> destinations;
  std::vector<MappedConcatFragmentRuntime> fragments;
};

struct MappedConcatJobBuildResult {
  std::vector<MappedConcatJobRuntime> jobs;
  std::vector<std::vector<ByteRange>> handled_dst_ranges_by_rank;
  uint64_t handled_source_bytes{0};
  uint64_t handled_root_dst_bytes{0};
};

struct LocalMappedTargetExecutionResult {
  runtime::ingestion::strategy::CollectiveExecutionMetrics metrics;
  loader::ByteRangeMap residual_data_map;
  uint64_t handled_bytes{0};
};

struct LocalMappedTensorExecutionStats {
  uint64_t replicated_jobs{0};
  uint64_t dim0_jobs{0};
  uint64_t dim1_jobs{0};
  uint64_t rect2d_jobs{0};
  size_t tasks{0};
  uint64_t read_bytes{0};
  uint64_t dst_bytes{0};
  double exec_sec{0.0};
};

struct LocalMappedTensorTask {
  enum class Kind : std::uint8_t {
    kContiguous = 0,
    kRect2D = 1,
  };

  Kind kind{Kind::kContiguous};
  uint64_t source_offset{0};
  uint64_t read_bytes{0};
  std::uint8_t* dst_ptr{nullptr};
  uint64_t dst_bytes{0};
  uint64_t src_col_offset_bytes{0};
  uint64_t src_pitch_bytes{0};
  uint64_t dst_pitch_bytes{0};
  uint64_t rows{0};
};

struct LocalMappedTensorTaskBuildResult {
  std::vector<LocalMappedTensorTask> tasks;
  LocalMappedTensorExecutionStats stats;
};

struct LocalMappedConcatTask {
  uint64_t source_offset{0};
  uint64_t dst_logical_offset{0};
  uint64_t block_bytes{0};
  uint64_t block_count{0};
  uint64_t dst_block_stride_bytes{0};
  std::uint8_t* direct_dst_ptr{nullptr};

  [[nodiscard]] uint64_t total_bytes() const {
    return block_bytes * block_count;
  }
};

struct LocalMappedConcatTaskBuildResult {
  std::vector<LocalMappedConcatTask> tasks;
  uint64_t source_bytes{0};
  uint64_t direct_bytes{0};
  uint64_t layout_bytes{0};
  uint64_t strided_bytes{0};
  size_t direct_tasks{0};
  size_t layout_tasks{0};
  size_t strided_tasks{0};
};

struct LocalMappedConcatExecutionStats {
  size_t tasks{0};
  size_t direct_tasks{0};
  size_t layout_tasks{0};
  size_t strided_tasks{0};
  uint64_t read_bytes{0};
  uint64_t direct_bytes{0};
  uint64_t layout_bytes{0};
  uint64_t strided_bytes{0};
  double exec_sec{0.0};
};

struct LocalMappedSourceOrderedExecutionStats {
  LocalMappedTensorExecutionStats tensor;
  LocalMappedConcatExecutionStats concat;
  size_t tasks{0};
  double exec_sec{0.0};
  double tensor_copy_sec{0.0};
  double concat_copy_sec{0.0};
};

struct Dim1PackWorkspace {
  std::vector<std::unique_ptr<common::memory::GpuDeviceMemory>> pack_buffers;
  size_t capacity_bytes{0};
  cudaStream_t pack_stream{nullptr};
};

struct OwnerRankWorkspace {
  std::unique_ptr<common::memory::GpuDeviceMemory> stage_buffer;
  size_t stage_capacity_bytes{0};
  cudaStream_t h2d_stream{nullptr};
  cudaEvent_t ready_event{nullptr};
  int device_id{-1};
  Dim1PackWorkspace dim1_pack_workspace;

  ~OwnerRankWorkspace() {
    if (ready_event != nullptr && device_id >= 0) {
      const absl::Status set_st = tensorcast::cuda::set_device(device_id);
      if (!set_st.ok()) {
        LOG(WARNING) << "failed to set device before destroying owner ready event: " << set_st;
      } else {
        const absl::Status destroy_st = tensorcast::cuda::event_destroy(ready_event);
        if (!destroy_st.ok()) {
          LOG(WARNING) << "failed to destroy owner ready event: " << destroy_st;
        }
      }
    }
    if (h2d_stream != nullptr && device_id >= 0) {
      const absl::Status set_st = tensorcast::cuda::set_device(device_id);
      if (!set_st.ok()) {
        LOG(WARNING) << "failed to set device before destroying owner H2D stream: " << set_st;
      } else {
        const absl::Status destroy_st = tensorcast::cuda::stream_destroy(h2d_stream);
        if (!destroy_st.ok()) {
          LOG(WARNING) << "failed to destroy owner H2D stream: " << destroy_st;
        }
      }
    }
    if (dim1_pack_workspace.pack_stream != nullptr && device_id >= 0) {
      const absl::Status set_st = tensorcast::cuda::set_device(device_id);
      if (!set_st.ok()) {
        LOG(WARNING) << "failed to set device before destroying owner dim1 pack stream: " << set_st;
      } else {
        const absl::Status destroy_st = tensorcast::cuda::stream_destroy(dim1_pack_workspace.pack_stream);
        if (!destroy_st.ok()) {
          LOG(WARNING) << "failed to destroy owner dim1 pack stream: " << destroy_st;
        }
      }
    }
  }
};

struct RemoteStageWorkspace {
  std::vector<std::unique_ptr<common::memory::GpuDeviceMemory>> buffers;
  std::vector<cudaEvent_t> done_events;
  std::vector<int> done_event_devices;
  size_t capacity_bytes{0};
  int root_device_id{-1};

  ~RemoteStageWorkspace() {
    for (size_t idx = 0; idx < done_events.size(); ++idx) {
      if (done_events[idx] == nullptr) {
        continue;
      }
      if (idx < done_event_devices.size() && done_event_devices[idx] >= 0) {
        const absl::Status set_st = tensorcast::cuda::set_device(done_event_devices[idx]);
        if (!set_st.ok()) {
          LOG(WARNING) << "failed to set device before destroying concat done event: " << set_st;
        }
      }
      const absl::Status destroy_st = tensorcast::cuda::event_destroy(done_events[idx]);
      if (!destroy_st.ok()) {
        LOG(WARNING) << "failed to destroy concat done event: " << destroy_st;
      }
    }
  }
};

struct MappedGroupState {
  explicit MappedGroupState(uint32_t size) : world_size(size), participants(size) {}

  const uint32_t world_size;
  absl::Mutex mu;
  absl::CondVar cv;
  std::vector<std::optional<ParsedMappedParticipant>> participants ABSL_GUARDED_BY(mu);
  uint32_t joined ABSL_GUARDED_BY(mu){0};
  bool launching ABSL_GUARDED_BY(mu){false};
  bool complete ABSL_GUARDED_BY(mu){false};
  absl::Status status ABSL_GUARDED_BY(mu){absl::OkStatus()};
  runtime::ingestion::strategy::CollectiveExecutionMetrics metrics ABSL_GUARDED_BY(mu);
};

struct FileSegment {
  uint64_t base_offset{0};
  uint64_t data_size{0};
  size_t source_file_index{0};
  int owner_rank{0};
};

enum class OwnerBatchOpKind : uint8_t {
  kReplicated = 0,
  kDim0 = 1,
  kDim1 = 2,
};

struct OwnerPlannedJob {
  size_t job_index{0};
  size_t file_index{0};
  int owner_rank{0};
  OwnerBatchOpKind op_kind{OwnerBatchOpKind::kReplicated};
  uint64_t source_begin{0};
  uint64_t source_end{0};
  uint64_t source_bytes{0};
  uint64_t peer_transfer_bytes{0};
  uint64_t peak_temporary_bytes{0};
};

struct OwnerBatchPlan {
  int owner_rank{0};
  uint64_t batch_bytes{0};
  uint64_t peak_temporary_bytes{0};
  uint64_t peer_transfer_bytes{0};
  uint64_t first_source_offset{0};
  std::vector<size_t> planned_job_indices;
};

struct OwnerCollectivePlan {
  std::vector<OwnerPlannedJob> planned_jobs;
  std::vector<OwnerBatchPlan> batches;
  std::vector<uint64_t> owner_bytes_by_rank;
  uint64_t unique_source_bytes{0};
  uint64_t peer_transfer_bytes{0};
  uint64_t dedup_saving_bytes{0};
  double owner_skew_ratio{1.0};
  bool used_segment_split{false};
};

struct SegmentCopy {
  uint64_t src_offset{0};
  uint64_t dst_offset{0};
  size_t bytes{0};
};

struct LocalDedupCopy {
  uint64_t src_dst_offset{0};
  uint64_t dst_offset{0};
  size_t bytes{0};
};

struct LocalBatchedExecutionPlan {
  std::vector<TensorJob> jobs;
  std::vector<SegmentCopy> direct_segments;
  std::vector<LocalDedupCopy> direct_dedup_copies;
  std::vector<TensorJob> dim1_jobs;
  LocalBatchedPlanSummary summary;
};

ABSL_CONST_INIT absl::Mutex g_group_mu(absl::kConstInit);
absl::flat_hash_map<std::string, std::shared_ptr<GroupState>> g_groups ABSL_GUARDED_BY(g_group_mu);
ABSL_CONST_INIT absl::Mutex g_mapped_group_mu(absl::kConstInit);
absl::flat_hash_map<std::string, std::shared_ptr<MappedGroupState>> g_mapped_groups ABSL_GUARDED_BY(g_mapped_group_mu);
ABSL_CONST_INIT absl::Mutex g_clique_mu(absl::kConstInit);
absl::flat_hash_map<std::string, std::shared_ptr<NcclClique>> g_clique_cache ABSL_GUARDED_BY(g_clique_mu);

std::string clique_cache_key(const std::vector<int>& device_ids) {
  std::string key;
  for (size_t i = 0; i < device_ids.size(); ++i) {
    if (i > 0) {
      key.push_back(',');
    }
    absl::StrAppend(&key, device_ids[i]);
  }
  return key;
}

absl::StatusOr<std::shared_ptr<NcclClique>> get_or_create_cached_clique(
    const std::vector<int>& device_ids,
    bool* cache_hit) {
  const std::string key = clique_cache_key(device_ids);
  {
    absl::MutexLock lock(&g_clique_mu);
    auto it = g_clique_cache.find(key);
    if (it != g_clique_cache.end()) {
      if (cache_hit != nullptr) {
        *cache_hit = true;
      }
      return it->second;
    }
  }

  auto clique_or = NcclClique::create(device_ids);
  if (!clique_or.ok()) {
    return clique_or.status();
  }

  {
    absl::MutexLock lock(&g_clique_mu);
    auto [it, inserted] = g_clique_cache.try_emplace(key, *clique_or);
    if (cache_hit != nullptr) {
      *cache_hit = !inserted;
    }
    return it->second;
  }
}

std::optional<uint64_t> dtype_element_size(std::string_view dtype) {
  static const absl::flat_hash_map<std::string_view, uint64_t> kMap = {
      {"torch.float16", 2},
      {"torch.bfloat16", 2},
      {"torch.float8_e4m3fn", 1},
      {"torch.float8_e5m2", 1},
      {"torch.float32", 4},
      {"torch.float64", 8},
      {"torch.int8", 1},
      {"torch.uint8", 1},
      {"torch.int16", 2},
      {"torch.int32", 4},
      {"torch.int64", 8},
      {"torch.bool", 1},
      {"torch.float", 4},
      {"torch.double", 8},
  };
  auto it = kMap.find(dtype);
  if (it == kMap.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool is_row_major_contiguous(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride) {
  if (shape.size() != stride.size()) {
    return false;
  }
  int64_t acc = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    if (stride[static_cast<size_t>(i)] != acc) {
      return false;
    }
    acc *= shape[static_cast<size_t>(i)];
  }
  return true;
}

absl::StatusOr<absl::flat_hash_map<std::string, TensorMeta>> parse_index_json(std::string_view index_json) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(index_json, nullptr, true);
  } catch (const std::exception& ex) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse index json: ", ex.what()));
  }
  if (!j.is_object()) {
    return absl::InvalidArgumentError("index json must be an object");
  }
  absl::flat_hash_map<std::string, TensorMeta> out;
  out.reserve(j.size());
  for (auto it = j.begin(); it != j.end(); ++it) {
    const auto& value = it.value();
    if (!value.is_array() || value.size() < 6) {
      return absl::InvalidArgumentError("index entry must be [offset,size,shape,stride,dtype,storage_offset]");
    }
    TensorMeta meta;
    meta.offset = value[0].get<uint64_t>();
    meta.size_bytes = value[1].get<uint64_t>();
    for (const auto& dim : value[2]) {
      meta.shape.push_back(dim.get<int64_t>());
    }
    for (const auto& dim : value[3]) {
      meta.stride.push_back(dim.get<int64_t>());
    }
    meta.dtype = value[4].get<std::string>();
    meta.storage_offset = value[5].get<uint64_t>();
    auto elem_size = dtype_element_size(meta.dtype);
    if (!elem_size.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat("unsupported dtype in index json: ", meta.dtype));
    }
    meta.elem_size = *elem_size;
    out.emplace(it.key(), std::move(meta));
  }
  return out;
}

absl::StatusOr<RankTensorSlice> build_rank_tensor_slice(
    const TensorMeta& source,
    const TensorMeta& view_meta,
    const std::optional<materialization::view::TensorViewOps>& view_ops) {
  RankTensorSlice slice;
  slice.dst_offset = view_meta.offset;
  slice.dst_size_bytes = view_meta.size_bytes;
  if (!view_ops.has_value() || view_ops->ops.empty()) {
    slice.kind = RankTensorSlice::Kind::kFull;
    slice.start = 0;
    slice.length = 0;
    return slice;
  }
  if (view_ops->ops.size() != 1 || view_ops->ops[0].kind != materialization::view::ViewOp::Kind::kNarrow) {
    return absl::UnimplementedError("only a single narrow op is supported in collective disk load");
  }
  const auto& narrow = view_ops->ops[0].narrow;
  int64_t dim_extent = 0;
  if (narrow.dim < 0 || narrow.dim >= static_cast<int32_t>(source.shape.size())) {
    return absl::InvalidArgumentError("narrow dim out of range in collective disk load");
  }
  dim_extent = source.shape[static_cast<size_t>(narrow.dim)];
  int64_t normalized_start = narrow.start;
  if (normalized_start < 0) {
    normalized_start += dim_extent;
  }
  if (normalized_start < 0 || normalized_start >= dim_extent) {
    return absl::InvalidArgumentError("narrow start out of range in collective disk load");
  }
  slice.start = normalized_start;
  slice.length = narrow.length;
  if (narrow.dim == 0) {
    slice.kind = RankTensorSlice::Kind::kDim0;
  } else if (narrow.dim == 1 && source.shape.size() == 2 && is_row_major_contiguous(source.shape, source.stride)) {
    slice.kind = RankTensorSlice::Kind::kDim1;
  } else {
    return absl::UnimplementedError("collective disk load only supports dim0 or 2D dim1 narrow");
  }
  return slice;
}

TensorJob::Distribution to_tensor_job_distribution(WorkPartitionKind distribution);

absl::StatusOr<std::vector<TensorJob>> build_tensor_jobs(const std::vector<ParsedParticipant>& participants) {
  if (participants.empty()) {
    return std::vector<TensorJob>{};
  }

  std::vector<RepresentationWorkPlan> work_plans;
  work_plans.reserve(participants.size());
  for (const auto& participant : participants) {
    work_plans.push_back(participant.representation_work_plan);
  }

  std::vector<const RepresentationWorkItem*> ordered_items;
  ordered_items.reserve(work_plans.front().items.size());
  for (const auto& item : work_plans.front().items) {
    if (item.kind != RepresentationWorkItemKind::kTensorCopy || item.sources.size() != 1) {
      continue;
    }
    ordered_items.push_back(&item);
  }
  std::sort(
      ordered_items.begin(),
      ordered_items.end(),
      [](const RepresentationWorkItem* lhs, const RepresentationWorkItem* rhs) {
        const uint64_t lhs_offset = lhs->sources.front().fragment.source_spec.logical_offset;
        const uint64_t rhs_offset = rhs->sources.front().fragment.source_spec.logical_offset;
        if (lhs_offset != rhs_offset) {
          return lhs_offset < rhs_offset;
        }
        return lhs->dst_name < rhs->dst_name;
      });

  std::vector<absl::flat_hash_map<std::string, const RepresentationWorkItem*>> jobs_by_rank;
  jobs_by_rank.reserve(work_plans.size());
  for (const auto& work_plan : work_plans) {
    absl::flat_hash_map<std::string, const RepresentationWorkItem*> by_name;
    by_name.reserve(work_plan.items.size());
    for (const auto& item : work_plan.items) {
      if (item.kind == RepresentationWorkItemKind::kTensorCopy) {
        by_name.emplace(item.dst_name, &item);
      }
    }
    jobs_by_rank.push_back(std::move(by_name));
  }

  std::vector<TensorJob> jobs;
  jobs.reserve(ordered_items.size());
  for (const auto* first : ordered_items) {
    if (first == nullptr || first->sources.size() != 1) {
      continue;
    }
    TensorJob job;
    job.name = first->dst_name;
    const auto& first_source = first->sources.front().fragment.source_spec;
    job.source = TensorMeta{
        .offset = first_source.logical_offset,
        .size_bytes = first_source.logical_length,
        .shape = first_source.shape,
        .stride = first_source.stride,
        .dtype = first_source.dtype,
        .storage_offset = first_source.storage_offset,
        .elem_size = first_source.element_size,
    };
    job.distribution = to_tensor_job_distribution(first->partition_kind);
    job.slices.resize(participants.size());

    bool any_dim0 = false;
    bool any_dim1 = false;
    bool any_full = false;
    for (size_t idx = 0; idx < participants.size(); ++idx) {
      auto it = jobs_by_rank[idx].find(first->dst_name);
      if (it == jobs_by_rank[idx].end() || it->second == nullptr || it->second->sources.size() != 1) {
        return absl::FailedPreconditionError(
            absl::StrCat("collective disk load requires the same tensor work set: ", first->dst_name));
      }
      const auto& item = *it->second;
      if (item.partition_kind != first->partition_kind) {
        return absl::UnimplementedError(
            absl::StrCat("mixed tensor work partitions are unsupported: ", first->dst_name));
      }
      const auto& item_source = item.sources.front().fragment.source_spec;
      if (item_source != first_source) {
        return absl::UnimplementedError(absl::StrCat("mixed tensor work sources are unsupported: ", first->dst_name));
      }
      RankTensorSlice slice;
      slice.dst_offset = item.dst_spec.logical_offset;
      slice.dst_size_bytes = item.dst_spec.logical_length;
      const auto& fragment = item.sources.front().fragment;
      const auto& source_axes = fragment.source_range.axes;
      switch (item.partition_kind) {
        case WorkPartitionKind::kReplicated:
          slice.kind = RankTensorSlice::Kind::kFull;
          break;
        case WorkPartitionKind::kDim0Partitioned:
          if (source_axes.size() != 1) {
            return absl::InvalidArgumentError("dim0 work item missing source axis");
          }
          slice.kind = RankTensorSlice::Kind::kDim0;
          slice.start = source_axes.front().start;
          slice.length = static_cast<uint64_t>(source_axes.front().end - source_axes.front().start);
          break;
        case WorkPartitionKind::kDim1Partitioned:
          if (source_axes.size() != 1) {
            return absl::InvalidArgumentError("dim1 work item missing source axis");
          }
          slice.kind = RankTensorSlice::Kind::kDim1;
          slice.start = source_axes.front().start;
          slice.length = static_cast<uint64_t>(source_axes.front().end - source_axes.front().start);
          break;
        case WorkPartitionKind::kUnknown:
          return absl::UnimplementedError(absl::StrCat("unknown tensor work partition for ", first->dst_name));
      }
      job.slices[idx] = slice;
      any_full = any_full || slice.kind == RankTensorSlice::Kind::kFull;
      any_dim0 = any_dim0 || slice.kind == RankTensorSlice::Kind::kDim0;
      any_dim1 = any_dim1 || slice.kind == RankTensorSlice::Kind::kDim1;
    }

    if (any_dim0 && (any_dim1 || any_full)) {
      return absl::UnimplementedError(absl::StrCat("mixed slice kinds for tensor are unsupported: ", first->dst_name));
    }
    if (any_dim1 && any_full) {
      return absl::UnimplementedError(absl::StrCat("mixed dim1/full slices are unsupported: ", first->dst_name));
    }
    jobs.push_back(std::move(job));
  }
  return jobs;
}

struct PinnedBorrow {
  std::shared_ptr<common::memory::PinnedBufferPool> pool;
  std::vector<char*> buffers;

  ~PinnedBorrow() {
    if (pool != nullptr && !buffers.empty()) {
      (void)pool->deallocate(buffers);
    }
  }
};

absl::Status read_exact(loader::MultiSafetensorsSource& source, uint64_t offset, void* dst, size_t bytes) {
  size_t got = 0;
  TC_ASSIGN_OR_RETURN(got, source.read_at(offset, dst, bytes));
  if (got != bytes) {
    return absl::OutOfRangeError(absl::StrCat("short read: got=", got, " want=", bytes, " offset=", offset));
  }
  return absl::OkStatus();
}

absl::Status read_exact(loader::SeekableSource& source, uint64_t offset, void* dst, size_t bytes) {
  size_t got = 0;
  TC_ASSIGN_OR_RETURN(got, source.read_at(offset, dst, bytes));
  if (got != bytes) {
    return absl::OutOfRangeError(absl::StrCat("short read: got=", got, " want=", bytes, " offset=", offset));
  }
  return absl::OkStatus();
}

absl::StatusOr<size_t> pread_fully(int fd, uint64_t offset, void* dst, size_t bytes) {
  size_t total = 0;
  auto* ptr = static_cast<char*>(dst);
  while (total < bytes) {
    const ssize_t got = ::pread(fd, ptr + total, bytes - total, static_cast<off_t>(offset + total));
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      return absl::ErrnoToStatus(errno, "pread failed");
    }
    if (got == 0) {
      break;
    }
    total += static_cast<size_t>(got);
  }
  return total;
}

uint64_t total_source_bytes(absl::Span<const loader::SharedSafetensorsSegment> segments) {
  uint64_t total = 0;
  for (const auto& segment : segments) {
    total = std::max<uint64_t>(total, segment.base_offset + segment.data_size);
  }
  return total;
}

absl::StatusOr<std::vector<FileSegment>> compute_file_segments(
    absl::Span<const loader::SharedSafetensorsSegment> segments,
    int world_size) {
  if (segments.empty()) {
    return absl::InvalidArgumentError("compute_file_segments requires non-empty segments");
  }
  if (world_size <= 0) {
    return absl::InvalidArgumentError("compute_file_segments requires world_size > 0");
  }
  std::vector<FileSegment> out;
  out.reserve(segments.size());
  for (size_t idx = 0; idx < segments.size(); ++idx) {
    const auto& segment = segments[idx];
    out.push_back(
        FileSegment{
            .base_offset = segment.base_offset,
            .data_size = segment.data_size,
            .source_file_index = idx,
            .owner_rank = static_cast<int>(idx % static_cast<size_t>(world_size)),
        });
  }
  std::sort(
      out.begin(), out.end(), [](const FileSegment& a, const FileSegment& b) { return a.base_offset < b.base_offset; });
  return out;
}

absl::StatusOr<size_t> find_file_index_for_offset(const std::vector<FileSegment>& segments, uint64_t offset) {
  for (size_t idx = 0; idx < segments.size(); ++idx) {
    const auto& segment = segments[idx];
    const uint64_t end = segment.base_offset + segment.data_size;
    if (offset >= segment.base_offset && offset < end) {
      return idx;
    }
  }
  return absl::OutOfRangeError("offset not mapped to a safetensors file");
}

uint64_t effective_owner_batch_bytes(const StrategyConfig& strategy_config, uint64_t fallback_bytes) {
  const uint64_t configured_bytes = strategy_config.owner_file_collective_batch_bytes > 0
      ? strategy_config.owner_file_collective_batch_bytes
      : fallback_bytes;
  return std::max<uint64_t>(1, configured_bytes);
}

uint64_t effective_owner_dim1_staging_bytes(const StrategyConfig& strategy_config, uint64_t fallback_bytes) {
  const uint64_t configured_bytes = strategy_config.owner_file_collective_dim1_staging_bytes > 0
      ? strategy_config.owner_file_collective_dim1_staging_bytes
      : fallback_bytes;
  return std::max<uint64_t>(1, configured_bytes);
}

absl::StatusOr<std::pair<uint64_t, uint64_t>> owner_job_source_interval(const TensorJob& job) {
  uint64_t source_begin = 0;
  TC_ASSIGN_OR_RETURN(source_begin, source_base_offset_bytes(job.source));
  if (source_begin > std::numeric_limits<uint64_t>::max() - job.source.size_bytes) {
    return absl::OutOfRangeError("owner collective source interval overflows");
  }
  return std::pair<uint64_t, uint64_t>{source_begin, source_begin + job.source.size_bytes};
}

uint64_t owner_job_peer_transfer_bytes(const TensorJob& job, int owner_rank) {
  uint64_t peer_transfer_bytes = 0;
  for (size_t rank = 0; rank < job.slices.size(); ++rank) {
    if (static_cast<int>(rank) == owner_rank) {
      continue;
    }
    peer_transfer_bytes += job.slices[rank].dst_size_bytes;
  }
  return peer_transfer_bytes;
}

uint64_t owner_job_naive_source_bytes(const TensorJob& job) {
  switch (job.distribution) {
    case TensorJob::Distribution::kReplicated:
      return job.source.size_bytes * static_cast<uint64_t>(job.slices.size());
    case TensorJob::Distribution::kDim1Partitioned: {
      uint64_t participating_ranks = 0;
      for (const auto& slice : job.slices) {
        if (slice.dst_size_bytes > 0) {
          participating_ranks += 1;
        }
      }
      return job.source.size_bytes * std::max<uint64_t>(1, participating_ranks);
    }
    case TensorJob::Distribution::kDim0Partitioned:
    default: {
      uint64_t total = 0;
      for (const auto& slice : job.slices) {
        total += slice.dst_size_bytes;
      }
      return total;
    }
  }
}

OwnerBatchOpKind owner_batch_op_kind_for_job(const TensorJob& job) {
  switch (job.distribution) {
    case TensorJob::Distribution::kDim0Partitioned:
      return OwnerBatchOpKind::kDim0;
    case TensorJob::Distribution::kDim1Partitioned:
      return OwnerBatchOpKind::kDim1;
    case TensorJob::Distribution::kReplicated:
    default:
      return OwnerBatchOpKind::kReplicated;
  }
}

uint64_t estimate_owner_job_peak_temporary_bytes(
    const TensorJob& job,
    uint64_t direct_batch_bytes,
    uint64_t dim1_staging_bytes) {
  switch (job.distribution) {
    case TensorJob::Distribution::kDim1Partitioned:
      return std::min<uint64_t>(job.source.size_bytes, dim1_staging_bytes);
    case TensorJob::Distribution::kDim0Partitioned:
    case TensorJob::Distribution::kReplicated:
    default:
      return std::min<uint64_t>(job.source.size_bytes, direct_batch_bytes);
  }
}

double compute_owner_skew_ratio(const std::vector<uint64_t>& owner_bytes_by_rank) {
  if (owner_bytes_by_rank.empty()) {
    return 1.0;
  }
  uint64_t total_bytes = 0;
  uint64_t max_bytes = 0;
  for (uint64_t bytes : owner_bytes_by_rank) {
    total_bytes += bytes;
    max_bytes = std::max(max_bytes, bytes);
  }
  if (total_bytes == 0) {
    return 1.0;
  }
  const double average_bytes =
      static_cast<double>(total_bytes) / static_cast<double>(std::max<size_t>(1, owner_bytes_by_rank.size()));
  if (average_bytes <= 0.0) {
    return 1.0;
  }
  return static_cast<double>(max_bytes) / average_bytes;
}

absl::StatusOr<OwnerCollectivePlan> build_owner_file_collective_plan(
    const std::vector<TensorJob>& jobs,
    absl::Span<const loader::SharedSafetensorsSegment> segments,
    int world_size,
    const StrategyConfig& strategy_config,
    uint64_t stage_chunk_bytes) {
  if (jobs.empty()) {
    return OwnerCollectivePlan{};
  }
  if (segments.empty()) {
    return absl::InvalidArgumentError("owner collective planning requires safetensors segments");
  }
  if (world_size <= 1) {
    return absl::InvalidArgumentError("owner collective planning requires world_size > 1");
  }

  std::vector<FileSegment> file_segments;
  TC_ASSIGN_OR_RETURN(file_segments, compute_file_segments(segments, world_size));
  OwnerCollectivePlan plan;
  plan.owner_bytes_by_rank.resize(static_cast<size_t>(world_size), 0);
  plan.planned_jobs.resize(jobs.size());

  std::vector<std::vector<size_t>> jobs_by_file(file_segments.size());
  std::vector<uint64_t> source_bytes_by_file(file_segments.size(), 0);

  const uint64_t batch_bytes = effective_owner_batch_bytes(strategy_config, stage_chunk_bytes);
  const uint64_t dim1_staging_bytes = effective_owner_dim1_staging_bytes(strategy_config, batch_bytes);
  uint64_t naive_source_bytes = 0;
  for (size_t job_idx = 0; job_idx < jobs.size(); ++job_idx) {
    const auto& job = jobs[job_idx];
    std::pair<uint64_t, uint64_t> source_interval;
    TC_ASSIGN_OR_RETURN(source_interval, owner_job_source_interval(job));
    size_t file_idx = 0;
    TC_ASSIGN_OR_RETURN(file_idx, find_file_index_for_offset(file_segments, source_interval.first));
    const auto& file_segment = file_segments[file_idx];
    if (source_interval.second > file_segment.base_offset + file_segment.data_size) {
      return absl::UnimplementedError(
          absl::StrCat("owner collective does not support multi-file tensor source spans: ", job.name));
    }

    auto& planned_job = plan.planned_jobs[job_idx];
    planned_job.job_index = job_idx;
    planned_job.file_index = file_idx;
    planned_job.op_kind = owner_batch_op_kind_for_job(job);
    planned_job.source_begin = source_interval.first;
    planned_job.source_end = source_interval.second;
    planned_job.source_bytes = source_interval.second - source_interval.first;
    planned_job.peer_transfer_bytes = 0;
    planned_job.peak_temporary_bytes = estimate_owner_job_peak_temporary_bytes(job, batch_bytes, dim1_staging_bytes);

    jobs_by_file[file_idx].push_back(job_idx);
    source_bytes_by_file[file_idx] += planned_job.source_bytes;
    plan.unique_source_bytes += planned_job.source_bytes;
    naive_source_bytes += owner_job_naive_source_bytes(job);
  }
  plan.dedup_saving_bytes =
      naive_source_bytes > plan.unique_source_bytes ? naive_source_bytes - plan.unique_source_bytes : 0;

  std::vector<size_t> file_order;
  file_order.reserve(file_segments.size());
  for (size_t file_idx = 0; file_idx < file_segments.size(); ++file_idx) {
    if (source_bytes_by_file[file_idx] == 0) {
      continue;
    }
    file_order.push_back(file_idx);
  }
  std::sort(file_order.begin(), file_order.end(), [&](size_t lhs, size_t rhs) {
    if (source_bytes_by_file[lhs] != source_bytes_by_file[rhs]) {
      return source_bytes_by_file[lhs] > source_bytes_by_file[rhs];
    }
    return file_segments[lhs].base_offset < file_segments[rhs].base_offset;
  });

  auto assign_jobs_greedily = [&](const std::vector<size_t>& job_order) {
    std::fill(plan.owner_bytes_by_rank.begin(), plan.owner_bytes_by_rank.end(), 0);
    for (size_t planned_idx = 0; planned_idx < plan.planned_jobs.size(); ++planned_idx) {
      plan.planned_jobs[planned_idx].owner_rank = 0;
    }
    for (size_t job_idx : job_order) {
      size_t chosen_rank = 0;
      uint64_t chosen_bytes = plan.owner_bytes_by_rank[0];
      for (size_t rank = 1; rank < plan.owner_bytes_by_rank.size(); ++rank) {
        if (plan.owner_bytes_by_rank[rank] < chosen_bytes) {
          chosen_rank = rank;
          chosen_bytes = plan.owner_bytes_by_rank[rank];
        }
      }
      plan.planned_jobs[job_idx].owner_rank = static_cast<int>(chosen_rank);
      plan.owner_bytes_by_rank[chosen_rank] += plan.planned_jobs[job_idx].source_bytes;
    }
  };

  {
    std::fill(plan.owner_bytes_by_rank.begin(), plan.owner_bytes_by_rank.end(), 0);
    for (size_t file_idx : file_order) {
      size_t chosen_rank = 0;
      uint64_t chosen_bytes = plan.owner_bytes_by_rank[0];
      for (size_t rank = 1; rank < plan.owner_bytes_by_rank.size(); ++rank) {
        if (plan.owner_bytes_by_rank[rank] < chosen_bytes) {
          chosen_rank = rank;
          chosen_bytes = plan.owner_bytes_by_rank[rank];
        }
      }
      for (size_t job_idx : jobs_by_file[file_idx]) {
        plan.planned_jobs[job_idx].owner_rank = static_cast<int>(chosen_rank);
      }
      plan.owner_bytes_by_rank[chosen_rank] += source_bytes_by_file[file_idx];
    }
  }

  plan.owner_skew_ratio = compute_owner_skew_ratio(plan.owner_bytes_by_rank);
  const bool skew_exceeds_threshold = strategy_config.owner_file_collective_max_owner_skew_ratio > 0.0 &&
      plan.owner_skew_ratio > strategy_config.owner_file_collective_max_owner_skew_ratio;
  if (skew_exceeds_threshold) {
    std::vector<size_t> job_order;
    job_order.reserve(plan.planned_jobs.size());
    for (size_t job_idx = 0; job_idx < plan.planned_jobs.size(); ++job_idx) {
      job_order.push_back(job_idx);
    }
    std::sort(job_order.begin(), job_order.end(), [&](size_t lhs, size_t rhs) {
      const auto& lhs_job = plan.planned_jobs[lhs];
      const auto& rhs_job = plan.planned_jobs[rhs];
      if (lhs_job.source_bytes != rhs_job.source_bytes) {
        return lhs_job.source_bytes > rhs_job.source_bytes;
      }
      if (lhs_job.file_index != rhs_job.file_index) {
        return lhs_job.file_index < rhs_job.file_index;
      }
      return lhs_job.source_begin < rhs_job.source_begin;
    });
    assign_jobs_greedily(job_order);
    plan.owner_skew_ratio = compute_owner_skew_ratio(plan.owner_bytes_by_rank);
    plan.used_segment_split = true;
  }

  for (auto& planned_job : plan.planned_jobs) {
    planned_job.peer_transfer_bytes =
        owner_job_peer_transfer_bytes(jobs[planned_job.job_index], planned_job.owner_rank);
    plan.peer_transfer_bytes += planned_job.peer_transfer_bytes;
  }

  std::vector<std::vector<size_t>> jobs_by_owner(plan.owner_bytes_by_rank.size());
  for (size_t planned_idx = 0; planned_idx < plan.planned_jobs.size(); ++planned_idx) {
    jobs_by_owner[static_cast<size_t>(plan.planned_jobs[planned_idx].owner_rank)].push_back(planned_idx);
  }

  for (size_t rank = 0; rank < jobs_by_owner.size(); ++rank) {
    auto& owner_jobs = jobs_by_owner[rank];
    if (owner_jobs.empty()) {
      continue;
    }
    std::sort(owner_jobs.begin(), owner_jobs.end(), [&](size_t lhs, size_t rhs) {
      const auto& lhs_job = plan.planned_jobs[lhs];
      const auto& rhs_job = plan.planned_jobs[rhs];
      if (lhs_job.file_index != rhs_job.file_index) {
        return lhs_job.file_index < rhs_job.file_index;
      }
      return lhs_job.source_begin < rhs_job.source_begin;
    });

    OwnerBatchPlan current_batch;
    current_batch.owner_rank = static_cast<int>(rank);
    current_batch.first_source_offset = std::numeric_limits<uint64_t>::max();
    for (size_t planned_idx : owner_jobs) {
      const auto& planned_job = plan.planned_jobs[planned_idx];
      if (!current_batch.planned_job_indices.empty() &&
          current_batch.batch_bytes + planned_job.source_bytes > batch_bytes) {
        plan.batches.push_back(std::move(current_batch));
        current_batch = OwnerBatchPlan{
            .owner_rank = static_cast<int>(rank),
            .first_source_offset = std::numeric_limits<uint64_t>::max(),
        };
      }
      current_batch.batch_bytes += planned_job.source_bytes;
      current_batch.peer_transfer_bytes += planned_job.peer_transfer_bytes;
      current_batch.peak_temporary_bytes =
          std::max(current_batch.peak_temporary_bytes, planned_job.peak_temporary_bytes);
      current_batch.first_source_offset = std::min(current_batch.first_source_offset, planned_job.source_begin);
      current_batch.planned_job_indices.push_back(planned_idx);
    }
    if (!current_batch.planned_job_indices.empty()) {
      plan.batches.push_back(std::move(current_batch));
    }
  }

  std::sort(plan.batches.begin(), plan.batches.end(), [](const OwnerBatchPlan& lhs, const OwnerBatchPlan& rhs) {
    if (lhs.first_source_offset != rhs.first_source_offset) {
      return lhs.first_source_offset < rhs.first_source_offset;
    }
    return lhs.owner_rank < rhs.owner_rank;
  });
  return plan;
}

runtime::ingestion::strategy::CollectiveExecutionMetrics owner_collective_metrics(const OwnerCollectivePlan& plan) {
  runtime::ingestion::strategy::CollectiveExecutionMetrics metrics;
  metrics.unique_source_bytes = plan.unique_source_bytes;
  metrics.peer_transfer_bytes = plan.peer_transfer_bytes;
  metrics.batch_count = plan.batches.size();
  metrics.dedup_saving_bytes = plan.dedup_saving_bytes;
  for (const auto& batch : plan.batches) {
    metrics.peak_temporary_bytes = std::max(metrics.peak_temporary_bytes, batch.peak_temporary_bytes);
  }
  return metrics;
}

absl::Status ensure_owner_rank_workspace(OwnerRankWorkspace* workspace, int device_id, size_t stage_bytes) {
  if (workspace == nullptr) {
    return absl::InvalidArgumentError("owner rank workspace is required");
  }
  if (workspace->device_id != device_id) {
    workspace->stage_buffer.reset();
    workspace->stage_capacity_bytes = 0;
    workspace->dim1_pack_workspace.pack_buffers.clear();
    workspace->dim1_pack_workspace.capacity_bytes = 0;
    if (workspace->ready_event != nullptr && workspace->device_id >= 0) {
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(workspace->device_id));
      TC_RETURN_IF_ERROR(tensorcast::cuda::event_destroy(workspace->ready_event));
      workspace->ready_event = nullptr;
    }
    if (workspace->h2d_stream != nullptr && workspace->device_id >= 0) {
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(workspace->device_id));
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_destroy(workspace->h2d_stream));
      workspace->h2d_stream = nullptr;
    }
    if (workspace->dim1_pack_workspace.pack_stream != nullptr && workspace->device_id >= 0) {
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(workspace->device_id));
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_destroy(workspace->dim1_pack_workspace.pack_stream));
      workspace->dim1_pack_workspace.pack_stream = nullptr;
    }
    workspace->device_id = device_id;
  }

  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_id));
  if (workspace->stage_buffer == nullptr || workspace->stage_capacity_bytes < stage_bytes) {
    workspace->stage_buffer = std::make_unique<common::memory::GpuDeviceMemory>();
    TC_RETURN_IF_ERROR(workspace->stage_buffer->allocate(stage_bytes, device_id));
    workspace->stage_capacity_bytes = stage_bytes;
  }
  if (workspace->h2d_stream == nullptr) {
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&workspace->h2d_stream, cudaStreamNonBlocking));
  }
  if (workspace->ready_event == nullptr) {
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_create_with_flags(&workspace->ready_event, cudaEventDisableTiming));
  }
  return absl::OkStatus();
}

std::vector<loader::Range> split_even_ranges(uint64_t base, uint64_t total_bytes, int parts) {
  if (total_bytes == 0) {
    return {};
  }
  const int effective_parts = std::max(1, parts);
  std::vector<loader::Range> ranges;
  ranges.reserve(static_cast<size_t>(effective_parts));
  const uint64_t base_chunk = total_bytes / static_cast<uint64_t>(effective_parts);
  const uint64_t remainder = total_bytes % static_cast<uint64_t>(effective_parts);
  uint64_t cursor = base;
  for (int idx = 0; idx < effective_parts; ++idx) {
    const uint64_t length = base_chunk + (static_cast<uint64_t>(idx) < remainder ? 1 : 0);
    if (length == 0) {
      continue;
    }
    ranges.emplace_back(cursor, static_cast<size_t>(length));
    cursor += length;
  }
  return ranges;
}

std::vector<std::pair<uint64_t, size_t>> merge_adjacent_ranges(std::vector<std::pair<uint64_t, size_t>> ranges) {
  if (ranges.empty()) {
    return ranges;
  }
  std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  std::vector<std::pair<uint64_t, size_t>> merged;
  merged.reserve(ranges.size());
  uint64_t current_offset = ranges.front().first;
  uint64_t current_end = current_offset + ranges.front().second;
  for (size_t idx = 1; idx < ranges.size(); ++idx) {
    const auto& [offset, length] = ranges[idx];
    const uint64_t end = offset + length;
    if (offset <= current_end) {
      current_end = std::max(current_end, end);
      continue;
    }
    merged.push_back({current_offset, static_cast<size_t>(current_end - current_offset)});
    current_offset = offset;
    current_end = end;
  }
  merged.push_back({current_offset, static_cast<size_t>(current_end - current_offset)});
  return merged;
}

std::vector<SegmentCopy> merge_adjacent_segments_by_src(std::vector<SegmentCopy> segments) {
  segments.erase(
      std::remove_if(segments.begin(), segments.end(), [](const SegmentCopy& segment) { return segment.bytes == 0; }),
      segments.end());
  std::sort(segments.begin(), segments.end(), [](const SegmentCopy& lhs, const SegmentCopy& rhs) {
    if (lhs.src_offset != rhs.src_offset) {
      return lhs.src_offset < rhs.src_offset;
    }
    if (lhs.dst_offset != rhs.dst_offset) {
      return lhs.dst_offset < rhs.dst_offset;
    }
    return lhs.bytes < rhs.bytes;
  });

  std::vector<SegmentCopy> merged;
  merged.reserve(segments.size());
  for (const auto& segment : segments) {
    if (merged.empty()) {
      merged.push_back(segment);
      continue;
    }
    auto& previous = merged.back();
    const uint64_t previous_src_end = previous.src_offset + static_cast<uint64_t>(previous.bytes);
    const uint64_t previous_dst_end = previous.dst_offset + static_cast<uint64_t>(previous.bytes);
    if (previous_src_end == segment.src_offset && previous_dst_end == segment.dst_offset) {
      previous.bytes += segment.bytes;
      continue;
    }
    merged.push_back(segment);
  }
  return merged;
}

absl::StatusOr<std::vector<loader::Range>> build_pump_ranges_for_copy(
    const std::vector<SegmentCopy>& segments,
    int io_threads) {
  std::vector<std::pair<uint64_t, size_t>> dst_ranges;
  dst_ranges.reserve(segments.size());
  for (const auto& segment : segments) {
    if (segment.bytes == 0) {
      continue;
    }
    dst_ranges.push_back({segment.dst_offset, segment.bytes});
  }
  dst_ranges = merge_adjacent_ranges(std::move(dst_ranges));
  if (dst_ranges.size() == 1 && io_threads > 1) {
    const auto [offset, length] = dst_ranges.front();
    return split_even_ranges(offset, length, io_threads);
  }
  std::vector<loader::Range> out;
  out.reserve(dst_ranges.size());
  for (const auto& [offset, length] : dst_ranges) {
    out.emplace_back(offset, length);
  }
  return out;
}

absl::StatusOr<uint64_t> contiguous_dim0_slice_bytes(
    const std::vector<int64_t>& shape,
    uint64_t elem_size,
    int64_t start,
    int64_t end) {
  if (end <= start) {
    return absl::InvalidArgumentError("contiguous_dim0_slice_bytes requires end > start");
  }
  if (shape.empty()) {
    return elem_size;
  }
  uint64_t tail = 1;
  for (size_t idx = 1; idx < shape.size(); ++idx) {
    tail *= static_cast<uint64_t>(shape[idx]);
  }
  return static_cast<uint64_t>(end - start) * tail * elem_size;
}

class PreadMultiSafetensorsSource final : public loader::SeekableSource {
 public:
  explicit PreadMultiSafetensorsSource(std::vector<loader::SharedSafetensorsSegment> segments)
      : segments_(std::move(segments)), total_bytes_(total_source_bytes(segments_)) {
    std::sort(
        segments_.begin(), segments_.end(), [](const auto& a, const auto& b) { return a.base_offset < b.base_offset; });
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    absl::MutexLock lock(&offset_mu_);
    auto read_or = read_at(current_offset_, dst, max_bytes);
    if (read_or.ok()) {
      current_offset_ += *read_or;
    }
    return read_or;
  }

  [[nodiscard]] uint64_t total_bytes() const override {
    return total_bytes_;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= total_bytes_) {
      return static_cast<size_t>(0);
    }
    size_t to_read = static_cast<size_t>(std::min<uint64_t>(bytes, total_bytes_ - offset));
    size_t total = 0;
    auto* out = static_cast<char*>(dst);
    uint64_t cursor = offset;
    while (total < to_read) {
      auto it = std::upper_bound(
          segments_.begin(),
          segments_.end(),
          cursor,
          [](uint64_t value, const loader::SharedSafetensorsSegment& segment) {
            return value < segment.base_offset + segment.data_size;
          });
      if (it == segments_.end()) {
        break;
      }
      const auto& segment = *it;
      if (cursor < segment.base_offset) {
        return absl::InternalError("pread source contains an uncovered gap");
      }
      if (segment.file == nullptr) {
        return absl::FailedPreconditionError("shared safetensors segment is missing a file handle");
      }
      const uint64_t within = cursor - segment.base_offset;
      const size_t available = static_cast<size_t>(segment.data_size - within);
      const size_t step = std::min(to_read - total, available);
      auto got_or = pread_fully(segment.file->fd(), segment.data_start + within, out + total, step);
      if (!got_or.ok()) {
        return got_or.status();
      }
      if (*got_or != step) {
        return absl::OutOfRangeError(
            absl::StrCat("short read in PreadMultiSafetensorsSource: got=", *got_or, " want=", step));
      }
      total += step;
      cursor += step;
    }
    if (total != to_read) {
      return absl::OutOfRangeError(
          absl::StrCat("short logical read in PreadMultiSafetensorsSource: got=", total, " want=", to_read));
    }
    return total;
  }

 private:
  std::vector<loader::SharedSafetensorsSegment> segments_;
  uint64_t total_bytes_{0};
  absl::Mutex offset_mu_;
  uint64_t current_offset_ ABSL_GUARDED_BY(offset_mu_){0};
};

constexpr uint64_t kLocalMappedDirectIoAlignment = 512;
constexpr uint64_t kLocalMappedDirectIoMinSourceBytes = 1ULL << 30;
constexpr uint64_t kLocalMappedPageCacheSamplePagesPerFile = 4096;
constexpr double kLocalMappedBufferedPageCacheResidencyThreshold = 0.75;
constexpr uint64_t kLocalMappedBufferedProbeMaxBytes = 256ULL << 20;
constexpr uint64_t kLocalMappedBufferedProbeChunkBytes = 16ULL << 20;
constexpr double kLocalMappedBufferedHotProbeMinGiBPerSec = 3.0;

struct LocalMappedSafetensorsAutoIoDecision {
  bool use_direct_aligned_edges{false};
  double page_cache_residency_ratio{-1.0};
  uint64_t buffered_probe_bytes{0};
  double buffered_probe_sec{-1.0};
  double buffered_probe_gib_per_sec{-1.0};
  std::string reason;
};

struct LocalMappedBufferedProbeResult {
  uint64_t bytes{0};
  double sec{0.0};
  double gib_per_sec{0.0};
};

struct LocalMappedBufferedProbeRead {
  const loader::SharedSafetensorsSegment* segment{nullptr};
  uint64_t file_offset{0};
  size_t bytes{0};
};

uint64_t align_down_u64(uint64_t value, uint64_t alignment) {
  return value - (value % alignment);
}

uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

bool is_aligned_address(const void* ptr, uint64_t alignment) {
  return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

bool is_direct_friendly_fs_type(uint64_t fs_type) {
  // Linux magic values for local filesystems where O_DIRECT is expected to be
  // stable for large regular files. Unknown, network, and memory filesystems
  // intentionally stay on the buffered path in auto mode.
  constexpr uint64_t kExtSuperMagic = 0xEF53;
  constexpr uint64_t kXfsSuperMagic = 0x58465342;
  constexpr uint64_t kBtrfsSuperMagic = 0x9123683E;
  constexpr uint64_t kF2fsSuperMagic = 0xF2F52010;
  return fs_type == kExtSuperMagic || fs_type == kXfsSuperMagic || fs_type == kBtrfsSuperMagic ||
      fs_type == kF2fsSuperMagic;
}

absl::StatusOr<double> estimate_file_page_cache_residency(const std::filesystem::path& path, uint64_t file_size) {
  if (file_size == 0) {
    return 1.0;
  }
  const long page_size_long = ::sysconf(_SC_PAGESIZE);
  if (page_size_long <= 0) {
    return absl::FailedPreconditionError("sysconf(_SC_PAGESIZE) failed while estimating page-cache residency");
  }
  const uint64_t page_size = static_cast<uint64_t>(page_size_long);
  const uint64_t page_count = (file_size + page_size - 1) / page_size;
  const uint64_t samples =
      std::max<uint64_t>(1, std::min<uint64_t>(page_count, kLocalMappedPageCacheSamplePagesPerFile));

  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("open failed while estimating page-cache residency: ", path.string()));
  }
  absl::Cleanup close_fd = [fd] { ::close(fd); };

  void* mapping = ::mmap(nullptr, static_cast<size_t>(file_size), PROT_NONE, MAP_PRIVATE, fd, 0);
  if (mapping == MAP_FAILED) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("mmap failed while estimating page-cache residency: ", path.string()));
  }
  absl::Cleanup unmap_file = [mapping, file_size] { ::munmap(mapping, static_cast<size_t>(file_size)); };

  uint64_t resident = 0;
  auto* base = static_cast<char*>(mapping);
  for (uint64_t i = 0; i < samples; ++i) {
    const uint64_t page_index = std::min<uint64_t>((i * page_count) / samples, page_count - 1);
    unsigned char vec = 0;
    if (::mincore(base + page_index * page_size, static_cast<size_t>(page_size), &vec) != 0) {
      return absl::ErrnoToStatus(
          errno, absl::StrCat("mincore failed while estimating page-cache residency: ", path.string()));
    }
    if ((vec & 0x1U) != 0) {
      ++resident;
    }
  }
  return static_cast<double>(resident) / static_cast<double>(samples);
}

absl::StatusOr<double> estimate_source_page_cache_residency(
    absl::Span<const loader::SharedSafetensorsSegment> segments) {
  absl::flat_hash_map<std::string, uint64_t> files;
  for (const auto& segment : segments) {
    if (segment.path.empty()) {
      continue;
    }
    auto [it, inserted] = files.emplace(segment.path.string(), 0);
    if (inserted) {
      struct stat st{};
      if (::stat(segment.path.c_str(), &st) != 0) {
        return absl::ErrnoToStatus(
            errno, absl::StrCat("stat failed while estimating page-cache residency: ", segment.path.string()));
      }
      if (st.st_size < 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("negative file size while estimating page-cache residency: ", segment.path.string()));
      }
      it->second = static_cast<uint64_t>(st.st_size);
    }
  }
  if (files.empty()) {
    return 0.0;
  }

  double weighted_residency = 0.0;
  uint64_t total_bytes = 0;
  for (const auto& [path, file_size] : files) {
    double residency = 0.0;
    TC_ASSIGN_OR_RETURN(residency, estimate_file_page_cache_residency(std::filesystem::path(path), file_size));
    weighted_residency += residency * static_cast<double>(file_size);
    total_bytes += file_size;
  }
  if (total_bytes == 0) {
    return 1.0;
  }
  return weighted_residency / static_cast<double>(total_bytes);
}

std::vector<LocalMappedBufferedProbeRead> plan_buffered_probe_reads(
    absl::Span<const loader::SharedSafetensorsSegment> segments,
    uint64_t total_bytes) {
  std::vector<const loader::SharedSafetensorsSegment*> sorted_segments;
  sorted_segments.reserve(segments.size());
  for (const auto& segment : segments) {
    if (segment.data_size > 0) {
      sorted_segments.push_back(&segment);
    }
  }
  std::sort(sorted_segments.begin(), sorted_segments.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->base_offset < rhs->base_offset;
  });
  if (sorted_segments.empty() || total_bytes == 0) {
    return {};
  }

  const uint64_t target_bytes = std::min<uint64_t>(total_bytes, kLocalMappedBufferedProbeMaxBytes);
  const uint64_t sample_count = std::max<uint64_t>(
      1, (target_bytes + kLocalMappedBufferedProbeChunkBytes - 1) / kLocalMappedBufferedProbeChunkBytes);

  std::vector<LocalMappedBufferedProbeRead> reads;
  reads.reserve(static_cast<size_t>(sample_count));
  uint64_t planned_bytes = 0;
  for (uint64_t i = 0; i < sample_count && planned_bytes < target_bytes; ++i) {
    const uint64_t logical_offset = std::min<uint64_t>((i * total_bytes) / sample_count, total_bytes - 1);
    auto it = std::upper_bound(
        sorted_segments.begin(),
        sorted_segments.end(),
        logical_offset,
        [](uint64_t value, const loader::SharedSafetensorsSegment* segment) {
          return value < segment->base_offset + segment->data_size;
        });
    if (it == sorted_segments.end()) {
      continue;
    }
    const auto* segment = *it;
    if (logical_offset < segment->base_offset) {
      continue;
    }
    const uint64_t within_segment = logical_offset - segment->base_offset;
    const uint64_t available = segment->data_size - within_segment;
    const uint64_t remaining = target_bytes - planned_bytes;
    const size_t bytes =
        static_cast<size_t>(std::min<uint64_t>({kLocalMappedBufferedProbeChunkBytes, available, remaining}));
    if (bytes == 0) {
      continue;
    }
    reads.push_back(
        LocalMappedBufferedProbeRead{
            .segment = segment,
            .file_offset = segment->data_start + within_segment,
            .bytes = bytes,
        });
    planned_bytes += bytes;
  }
  return reads;
}

absl::StatusOr<LocalMappedBufferedProbeResult> probe_buffered_source_throughput(
    absl::Span<const loader::SharedSafetensorsSegment> segments,
    uint64_t total_bytes) {
  auto reads = plan_buffered_probe_reads(segments, total_bytes);
  if (reads.empty()) {
    return LocalMappedBufferedProbeResult{};
  }

  std::vector<char> buffer(static_cast<size_t>(kLocalMappedBufferedProbeChunkBytes));
  uint64_t read_bytes = 0;
  const auto start = std::chrono::steady_clock::now();
  for (const auto& read : reads) {
    if (read.segment == nullptr || read.segment->file == nullptr) {
      return absl::FailedPreconditionError("buffered source probe is missing a file handle");
    }
    size_t got = 0;
    TC_ASSIGN_OR_RETURN(got, pread_fully(read.segment->file->fd(), read.file_offset, buffer.data(), read.bytes));
    if (got != read.bytes) {
      return absl::OutOfRangeError(absl::StrCat("short buffered source probe read: got=", got, " want=", read.bytes));
    }
    read_bytes += static_cast<uint64_t>(got);
  }
  const auto end = std::chrono::steady_clock::now();
  const double sec = std::chrono::duration<double>(end - start).count();
  const double safe_sec = std::max(sec, 1e-9);
  const double gib_per_sec = (static_cast<double>(read_bytes) / static_cast<double>(1ULL << 30)) / safe_sec;
  return LocalMappedBufferedProbeResult{
      .bytes = read_bytes,
      .sec = sec,
      .gib_per_sec = gib_per_sec,
  };
}

absl::StatusOr<LocalMappedSafetensorsAutoIoDecision> choose_auto_local_mapped_safetensors_io(
    absl::Span<const loader::SharedSafetensorsSegment> segments) {
  const uint64_t total_bytes = total_source_bytes(segments);
  if (total_bytes < kLocalMappedDirectIoMinSourceBytes) {
    return LocalMappedSafetensorsAutoIoDecision{
        .use_direct_aligned_edges = false,
        .reason = "source_below_direct_threshold",
    };
  }
  for (const auto& segment : segments) {
    if (segment.path.empty()) {
      return LocalMappedSafetensorsAutoIoDecision{
          .use_direct_aligned_edges = false,
          .reason = "segment_path_missing",
      };
    }
    struct stat st{};
    if (::stat(segment.path.c_str(), &st) != 0) {
      return absl::ErrnoToStatus(errno, absl::StrCat("stat failed for safetensors source: ", segment.path.string()));
    }
    if (!S_ISREG(st.st_mode)) {
      return LocalMappedSafetensorsAutoIoDecision{
          .use_direct_aligned_edges = false,
          .reason = "source_not_regular_file",
      };
    }
    struct statfs fs{};
    if (::statfs(segment.path.c_str(), &fs) != 0) {
      return absl::ErrnoToStatus(errno, absl::StrCat("statfs failed for safetensors source: ", segment.path.string()));
    }
    if (!is_direct_friendly_fs_type(static_cast<uint64_t>(fs.f_type))) {
      return LocalMappedSafetensorsAutoIoDecision{
          .use_direct_aligned_edges = false,
          .reason = "filesystem_not_direct_friendly",
      };
    }
  }
  double page_cache_residency = 0.0;
  TC_ASSIGN_OR_RETURN(page_cache_residency, estimate_source_page_cache_residency(segments));
  if (page_cache_residency >= kLocalMappedBufferedPageCacheResidencyThreshold) {
    LocalMappedBufferedProbeResult probe;
    TC_ASSIGN_OR_RETURN(probe, probe_buffered_source_throughput(segments, total_bytes));
    if (probe.gib_per_sec >= kLocalMappedBufferedHotProbeMinGiBPerSec) {
      return LocalMappedSafetensorsAutoIoDecision{
          .use_direct_aligned_edges = false,
          .page_cache_residency_ratio = page_cache_residency,
          .buffered_probe_bytes = probe.bytes,
          .buffered_probe_sec = probe.sec,
          .buffered_probe_gib_per_sec = probe.gib_per_sec,
          .reason = "page_cache_hot_probe_fast",
      };
    }
    return LocalMappedSafetensorsAutoIoDecision{
        .use_direct_aligned_edges = true,
        .page_cache_residency_ratio = page_cache_residency,
        .buffered_probe_bytes = probe.bytes,
        .buffered_probe_sec = probe.sec,
        .buffered_probe_gib_per_sec = probe.gib_per_sec,
        .reason = "page_cache_resident_probe_slow",
    };
  }
  return LocalMappedSafetensorsAutoIoDecision{
      .use_direct_aligned_edges = true,
      .page_cache_residency_ratio = page_cache_residency,
      .reason = "page_cache_cold_or_partial",
  };
}

class DirectAlignedSafetensorsSource final : public loader::SeekableSource {
 public:
  explicit DirectAlignedSafetensorsSource(std::vector<loader::SharedSafetensorsSegment> segments)
      : total_bytes_(total_source_bytes(segments)) {
    segments_.reserve(segments.size());
    for (auto& segment : segments) {
      segments_.push_back(Segment{.segment = std::move(segment)});
    }
    std::sort(segments_.begin(), segments_.end(), [](const auto& a, const auto& b) {
      return a.segment.base_offset < b.segment.base_offset;
    });
  }

  ~DirectAlignedSafetensorsSource() override {
    for (auto& segment : segments_) {
      if (segment.direct_fd >= 0) {
        ::close(segment.direct_fd);
        segment.direct_fd = -1;
      }
    }
  }

  DirectAlignedSafetensorsSource(const DirectAlignedSafetensorsSource&) = delete;
  DirectAlignedSafetensorsSource& operator=(const DirectAlignedSafetensorsSource&) = delete;

  absl::Status initialize() {
    for (auto& segment : segments_) {
      if (segment.segment.path.empty()) {
        return absl::FailedPreconditionError("direct safetensors source requires segment paths");
      }
      const int fd = ::open(segment.segment.path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
      if (fd < 0) {
        return absl::ErrnoToStatus(
            errno, absl::StrCat("O_DIRECT open failed for safetensors source: ", segment.segment.path.string()));
      }
      struct stat st{};
      if (::fstat(fd, &st) != 0) {
        const int err = errno;
        ::close(fd);
        return absl::ErrnoToStatus(
            err, absl::StrCat("fstat failed for safetensors source: ", segment.segment.path.string()));
      }
      if (st.st_size < 0) {
        ::close(fd);
        return absl::InvalidArgumentError(
            absl::StrCat("negative file size for safetensors source: ", segment.segment.path.string()));
      }
      segment.direct_fd = fd;
      segment.file_size = static_cast<uint64_t>(st.st_size);
      segment.direct_file_floor = align_down_u64(segment.file_size, kLocalMappedDirectIoAlignment);
    }
    return absl::OkStatus();
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    absl::MutexLock lock(&offset_mu_);
    auto read_or = read_at(current_offset_, dst, max_bytes);
    if (read_or.ok()) {
      current_offset_ += *read_or;
    }
    return read_or;
  }

  [[nodiscard]] uint64_t total_bytes() const override {
    return total_bytes_;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= total_bytes_) {
      return static_cast<size_t>(0);
    }
    const size_t to_read = static_cast<size_t>(std::min<uint64_t>(bytes, total_bytes_ - offset));
    size_t total = 0;
    auto* out = static_cast<char*>(dst);
    uint64_t cursor = offset;
    while (total < to_read) {
      Segment* segment = find_segment_for_offset(cursor);
      if (segment == nullptr) {
        return absl::InternalError("direct safetensors source contains an uncovered gap");
      }
      if (segment->segment.file == nullptr) {
        return absl::FailedPreconditionError("direct safetensors source is missing a buffered file handle");
      }
      if (segment->direct_fd < 0) {
        return absl::FailedPreconditionError("direct safetensors source was not initialized");
      }
      const uint64_t within = cursor - segment->segment.base_offset;
      const size_t available = static_cast<size_t>(segment->segment.data_size - within);
      const size_t step = std::min(to_read - total, available);
      TC_RETURN_IF_ERROR(read_file_range(*segment, segment->segment.data_start + within, out + total, step));
      total += step;
      cursor += step;
    }
    if (total != to_read) {
      return absl::OutOfRangeError(
          absl::StrCat("short logical read in DirectAlignedSafetensorsSource: got=", total, " want=", to_read));
    }
    return total;
  }

 private:
  struct Segment {
    loader::SharedSafetensorsSegment segment;
    int direct_fd{-1};
    uint64_t file_size{0};
    uint64_t direct_file_floor{0};
  };

  class AlignedScratch {
   public:
    ~AlignedScratch() {
      std::free(ptr_);
    }

    AlignedScratch(const AlignedScratch&) = delete;
    AlignedScratch& operator=(const AlignedScratch&) = delete;
    AlignedScratch() = default;

    absl::Status ensure(size_t bytes) {
      if (bytes <= capacity_) {
        return absl::OkStatus();
      }
      void* next = nullptr;
      const int rc = ::posix_memalign(&next, kLocalMappedDirectIoAlignment, bytes);
      if (rc != 0) {
        return absl::ErrnoToStatus(rc, "posix_memalign failed for O_DIRECT scratch");
      }
      std::free(ptr_);
      ptr_ = next;
      capacity_ = bytes;
      return absl::OkStatus();
    }

    void* data() {
      return ptr_;
    }

   private:
    void* ptr_{nullptr};
    size_t capacity_{0};
  };

  static AlignedScratch& thread_scratch() {
    thread_local AlignedScratch scratch;
    return scratch;
  }

  Segment* find_segment_for_offset(uint64_t offset) {
    auto it = std::upper_bound(segments_.begin(), segments_.end(), offset, [](uint64_t value, const Segment& segment) {
      return value < segment.segment.base_offset;
    });
    if (it == segments_.begin()) {
      return nullptr;
    }
    --it;
    const uint64_t end = it->segment.base_offset + it->segment.data_size;
    if (offset >= end) {
      return nullptr;
    }
    return &*it;
  }

  absl::Status read_file_range(Segment& segment, uint64_t file_offset, char* dst, size_t bytes) {
    if (bytes == 0) {
      return absl::OkStatus();
    }
    const uint64_t begin = file_offset;
    const uint64_t end = file_offset + static_cast<uint64_t>(bytes);
    const bool direct_to_dst = (begin % kLocalMappedDirectIoAlignment) == 0 &&
        (static_cast<uint64_t>(bytes) % kLocalMappedDirectIoAlignment) == 0 && end <= segment.direct_file_floor &&
        is_aligned_address(dst, kLocalMappedDirectIoAlignment);
    if (direct_to_dst) {
      size_t got = 0;
      TC_ASSIGN_OR_RETURN(got, pread_fully(segment.direct_fd, begin, dst, bytes));
      if (got != bytes) {
        return absl::OutOfRangeError(
            absl::StrCat("short O_DIRECT read: got=", got, " want=", bytes, " offset=", begin));
      }
      return absl::OkStatus();
    }

    const uint64_t direct_begin = align_down_u64(begin, kLocalMappedDirectIoAlignment);
    const uint64_t direct_end =
        std::min<uint64_t>(align_up_u64(end, kLocalMappedDirectIoAlignment), segment.direct_file_floor);
    if (direct_end > direct_begin) {
      const size_t direct_bytes = static_cast<size_t>(direct_end - direct_begin);
      auto& scratch = thread_scratch();
      TC_RETURN_IF_ERROR(scratch.ensure(direct_bytes));
      size_t got = 0;
      TC_ASSIGN_OR_RETURN(got, pread_fully(segment.direct_fd, direct_begin, scratch.data(), direct_bytes));
      if (got != direct_bytes) {
        return absl::OutOfRangeError(
            absl::StrCat("short O_DIRECT scratch read: got=", got, " want=", direct_bytes, " offset=", direct_begin));
      }
      const uint64_t useful_begin = std::max(begin, direct_begin);
      const uint64_t useful_end = std::min(end, direct_end);
      if (useful_end > useful_begin) {
        std::memcpy(
            dst + static_cast<size_t>(useful_begin - begin),
            static_cast<const char*>(scratch.data()) + static_cast<size_t>(useful_begin - direct_begin),
            static_cast<size_t>(useful_end - useful_begin));
      }
    }

    if (end > direct_end) {
      const uint64_t tail_begin = std::max(begin, direct_end);
      const size_t tail_bytes = static_cast<size_t>(end - tail_begin);
      size_t got = 0;
      TC_ASSIGN_OR_RETURN(
          got,
          pread_fully(
              segment.segment.file->fd(), tail_begin, dst + static_cast<size_t>(tail_begin - begin), tail_bytes));
      if (got != tail_bytes) {
        return absl::OutOfRangeError(
            absl::StrCat("short buffered EOF-edge read: got=", got, " want=", tail_bytes, " offset=", tail_begin));
      }
    }
    return absl::OkStatus();
  }

  std::vector<Segment> segments_;
  uint64_t total_bytes_{0};
  absl::Mutex offset_mu_;
  uint64_t current_offset_ ABSL_GUARDED_BY(offset_mu_){0};
};

absl::StatusOr<std::unique_ptr<loader::SeekableSource>> make_local_mapped_safetensors_source(
    absl::Span<const loader::SharedSafetensorsSegment> segments,
    const StrategyConfig& strategy) {
  using IoMode = StrategyConfig::LocalMappedSafetensorsIoMode;
  IoMode mode = strategy.local_mapped_safetensors_io_mode;
  if (mode == IoMode::kAutoByFilesystem) {
    LocalMappedSafetensorsAutoIoDecision decision;
    TC_ASSIGN_OR_RETURN(decision, choose_auto_local_mapped_safetensors_io(segments));
    mode = decision.use_direct_aligned_edges ? IoMode::kDirectAlignedEdges : IoMode::kBuffered;
    LOG(INFO) << "local_mapped_safetensors_auto source_bytes=" << total_source_bytes(segments)
              << " page_cache_residency_ratio=" << decision.page_cache_residency_ratio
              << " buffered_probe_bytes=" << decision.buffered_probe_bytes
              << " buffered_probe_sec=" << decision.buffered_probe_sec
              << " buffered_probe_gib_per_sec=" << decision.buffered_probe_gib_per_sec
              << " decision=" << (decision.use_direct_aligned_edges ? "direct_aligned_edges" : "buffered")
              << " reason=" << decision.reason;
  }

  std::vector<loader::SharedSafetensorsSegment> owned_segments(segments.begin(), segments.end());
  if (mode == IoMode::kBuffered) {
    LOG(INFO) << "local_mapped_safetensors_io_mode=buffered source_bytes=" << total_source_bytes(segments);
    std::unique_ptr<loader::SeekableSource> source =
        std::make_unique<PreadMultiSafetensorsSource>(std::move(owned_segments));
    return source;
  }

  auto source = std::make_unique<DirectAlignedSafetensorsSource>(std::move(owned_segments));
  TC_RETURN_IF_ERROR(source->initialize());
  LOG(INFO) << "local_mapped_safetensors_io_mode=direct_aligned_edges source_bytes=" << total_source_bytes(segments)
            << " direct_alignment=" << kLocalMappedDirectIoAlignment;
  std::unique_ptr<loader::SeekableSource> base_source = std::move(source);
  return base_source;
}

class RemappedSource final : public loader::SeekableSource {
 public:
  struct Segment {
    uint64_t dst_offset{0};
    uint64_t src_offset{0};
    uint64_t end_offset{0};
  };

  RemappedSource(gsl::not_null<loader::SeekableSource*> backing, std::vector<Segment> segments)
      : backing_(backing), segments_(std::move(segments)) {
    std::sort(segments_.begin(), segments_.end(), [](const Segment& lhs, const Segment& rhs) {
      return lhs.dst_offset < rhs.dst_offset;
    });
    if (!segments_.empty()) {
      total_bytes_ = segments_.back().end_offset;
    }
  }

  [[nodiscard]] uint64_t total_bytes() const override {
    return total_bytes_;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    absl::MutexLock lock(&offset_mu_);
    auto got_or = read_at(current_offset_, dst, max_bytes);
    if (got_or.ok()) {
      current_offset_ += static_cast<uint64_t>(*got_or);
    }
    return got_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (bytes == 0) {
      return 0;
    }
    if (offset >= total_bytes_) {
      return static_cast<size_t>(0);
    }
    const Segment* segment = find_segment_(offset);
    if (segment == nullptr) {
      return absl::OutOfRangeError("RemappedSource: offset not mapped");
    }
    const uint64_t available = segment->end_offset - offset;
    const size_t to_read = static_cast<size_t>(std::min<uint64_t>(bytes, available));
    const uint64_t delta = segment->src_offset - segment->dst_offset;
    return backing_->read_at(offset + delta, dst, to_read);
  }

 private:
  const Segment* find_segment_(uint64_t offset) const {
    if (segments_.empty()) {
      return nullptr;
    }
    size_t lo = 0;
    size_t hi = segments_.size();
    while (lo < hi) {
      const size_t mid = lo + (hi - lo) / 2;
      const Segment& segment = segments_[mid];
      if (offset < segment.dst_offset) {
        hi = mid;
      } else if (offset >= segment.end_offset) {
        lo = mid + 1;
      } else {
        return &segment;
      }
    }
    return nullptr;
  }

  gsl::not_null<loader::SeekableSource*> backing_;
  std::vector<Segment> segments_;
  uint64_t total_bytes_{0};
  absl::Mutex offset_mu_;
  uint64_t current_offset_ ABSL_GUARDED_BY(offset_mu_){0};
};

class ChunkPrefetcher {
 public:
  explicit ChunkPrefetcher(loader::SeekableSource* source) : source_(source), worker_([this] { run(); }) {}

  ~ChunkPrefetcher() {
    shutdown();
  }

  ChunkPrefetcher(const ChunkPrefetcher&) = delete;
  ChunkPrefetcher& operator=(const ChunkPrefetcher&) = delete;

  absl::Status start(uint64_t offset, size_t length, char* dst) {
    absl::MutexLock lock(&mu_);
    while (job_pending_ || job_ready_) {
      cv_.Wait(&mu_);
    }
    if (stopped_) {
      return absl::FailedPreconditionError("chunk prefetcher has been stopped");
    }
    job_offset_ = offset;
    job_length_ = length;
    job_dst_ = dst;
    job_status_ = absl::UnknownError("chunk prefetch not completed");
    job_read_sec_ = 0.0;
    job_pending_ = true;
    cv_.SignalAll();
    return absl::OkStatus();
  }

  absl::Status wait(double* read_sec) {
    absl::MutexLock lock(&mu_);
    while (!job_ready_ && !stopped_) {
      cv_.Wait(&mu_);
    }
    if (!job_ready_) {
      return absl::CancelledError("chunk prefetcher stopped before job completion");
    }
    if (read_sec != nullptr) {
      *read_sec = job_read_sec_;
    }
    absl::Status status = job_status_;
    job_ready_ = false;
    cv_.SignalAll();
    return status;
  }

  void shutdown() {
    {
      absl::MutexLock lock(&mu_);
      if (stopped_) {
        return;
      }
      stopped_ = true;
      cv_.SignalAll();
    }
    if (worker_.joinable()) {
      worker_.join();
    }
  }

 private:
  void run() {
    while (true) {
      uint64_t offset = 0;
      size_t length = 0;
      char* dst = nullptr;
      {
        absl::MutexLock lock(&mu_);
        while (!job_pending_ && !stopped_) {
          cv_.Wait(&mu_);
        }
        if (stopped_) {
          return;
        }
        offset = job_offset_;
        length = job_length_;
        dst = job_dst_;
        job_pending_ = false;
      }

      const auto start = std::chrono::steady_clock::now();
      const absl::Status status = read_exact(*source_, offset, dst, length);
      const double read_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

      {
        absl::MutexLock lock(&mu_);
        job_status_ = status;
        job_read_sec_ = read_sec;
        job_ready_ = true;
        cv_.SignalAll();
      }
    }
  }

  loader::SeekableSource* source_{nullptr};
  absl::Mutex mu_;
  absl::CondVar cv_;
  bool stopped_ ABSL_GUARDED_BY(mu_){false};
  bool job_pending_ ABSL_GUARDED_BY(mu_){false};
  bool job_ready_ ABSL_GUARDED_BY(mu_){false};
  uint64_t job_offset_ ABSL_GUARDED_BY(mu_){0};
  size_t job_length_ ABSL_GUARDED_BY(mu_){0};
  char* job_dst_ ABSL_GUARDED_BY(mu_){nullptr};
  absl::Status job_status_ ABSL_GUARDED_BY(mu_){absl::OkStatus()};
  double job_read_sec_ ABSL_GUARDED_BY(mu_){0.0};
  std::thread worker_;
};

common::AsyncRuntime& whole_source_load_runtime() {
  static common::AsyncRuntime runtime(
      common::AsyncRuntime::Options{
          .cpu_threads = 1,
          .blocking_threads = 8,
          .thread_name_prefix = "tc-whole-src",
      });
  return runtime;
}

std::vector<loader::Range> split_even_ranges(uint64_t total_bytes, int parts) {
  if (total_bytes == 0) {
    return {};
  }
  const int effective_parts = std::max(1, parts);
  std::vector<loader::Range> ranges;
  ranges.reserve(static_cast<size_t>(effective_parts));
  const uint64_t base_chunk = total_bytes / static_cast<uint64_t>(effective_parts);
  const uint64_t remainder = total_bytes % static_cast<uint64_t>(effective_parts);
  uint64_t cursor = 0;
  for (int idx = 0; idx < effective_parts; ++idx) {
    uint64_t length = base_chunk + (static_cast<uint64_t>(idx) < remainder ? 1 : 0);
    if (length == 0) {
      continue;
    }
    ranges.emplace_back(cursor, static_cast<size_t>(length));
    cursor += length;
  }
  return ranges;
}

absl::StatusOr<std::vector<TargetStorageSpan>> build_target_storage_spans(
    const loading::IntoTargetLayout& target_layout) {
  std::vector<TargetStorageSpan> spans;
  spans.reserve(target_layout.storages.size());
  uint64_t cursor = 0;
  for (const auto& storage : target_layout.storages) {
    if (storage.length == 0) {
      return absl::InvalidArgumentError("mapped collective target storage length must be non-zero");
    }
    spans.push_back(
        TargetStorageSpan{
            .base_offset = cursor,
            .length = storage.length,
            .base_ptr = gsl::not_null<std::uint8_t*>{static_cast<std::uint8_t*>(storage.base_ptr.get())},
        });
    if (storage.length > std::numeric_limits<uint64_t>::max() - cursor) {
      return absl::OutOfRangeError("mapped collective target layout length overflow");
    }
    cursor += storage.length;
  }
  if (target_layout.total_size != 0 && target_layout.total_size != cursor) {
    return absl::InvalidArgumentError("mapped collective target total_size mismatch");
  }
  return spans;
}

TensorJob::Distribution to_tensor_job_distribution(WorkPartitionKind distribution) {
  switch (distribution) {
    case WorkPartitionKind::kReplicated:
      return TensorJob::Distribution::kReplicated;
    case WorkPartitionKind::kDim0Partitioned:
      return TensorJob::Distribution::kDim0Partitioned;
    case WorkPartitionKind::kDim1Partitioned:
      return TensorJob::Distribution::kDim1Partitioned;
    case WorkPartitionKind::kUnknown:
      return TensorJob::Distribution::kReplicated;
  }
  return TensorJob::Distribution::kReplicated;
}

void merge_byte_ranges(std::vector<ByteRange>* ranges) {
  if (ranges == nullptr || ranges->empty()) {
    return;
  }
  std::sort(ranges->begin(), ranges->end(), [](const ByteRange& a, const ByteRange& b) { return a.begin < b.begin; });
  size_t out = 0;
  for (size_t idx = 1; idx < ranges->size(); ++idx) {
    if ((*ranges)[idx].begin <= (*ranges)[out].end) {
      (*ranges)[out].end = std::max((*ranges)[out].end, (*ranges)[idx].end);
      continue;
    }
    ++out;
    (*ranges)[out] = (*ranges)[idx];
  }
  ranges->resize(out + 1);
}

std::vector<ByteRange> data_ranges_from_lane_map(const loader::ByteRangeMap& map) {
  std::vector<ByteRange> ranges;
  ranges.reserve(map.segments.size());
  for (const auto& segment : map.segments) {
    if (segment.kind != loader::ByteRangeSegment::Kind::kData || segment.length == 0) {
      continue;
    }
    ranges.push_back(
        ByteRange{
            .begin = segment.dst_offset,
            .end = segment.dst_offset + segment.length,
        });
  }
  merge_byte_ranges(&ranges);
  return ranges;
}

std::vector<std::vector<ByteRange>> data_ranges_by_participant(
    const std::vector<ParsedMappedParticipant>& participants) {
  std::vector<std::vector<ByteRange>> ranges;
  ranges.reserve(participants.size());
  for (const auto& participant : participants) {
    ranges.push_back(data_ranges_from_lane_map(participant.collective_lane_map));
  }
  return ranges;
}

bool byte_range_is_fully_covered(const std::vector<ByteRange>& ranges, uint64_t begin, uint64_t end) {
  if (end <= begin) {
    return true;
  }
  uint64_t cursor = begin;
  for (const auto& range : ranges) {
    if (range.end <= cursor) {
      continue;
    }
    if (range.begin > cursor) {
      return false;
    }
    cursor = std::max<uint64_t>(cursor, range.end);
    if (cursor >= end) {
      return true;
    }
  }
  return false;
}

bool byte_range_overlaps_any(const std::vector<ByteRange>& ranges, uint64_t begin, uint64_t end) {
  if (end <= begin) {
    return false;
  }
  for (const auto& range : ranges) {
    if (range.end <= begin) {
      continue;
    }
    if (range.begin >= end) {
      return false;
    }
    return true;
  }
  return false;
}

uint64_t byte_ranges_covered_bytes(absl::Span<const ByteRange> ranges) {
  uint64_t total = 0;
  for (const auto& range : ranges) {
    if (range.end > range.begin) {
      total += range.end - range.begin;
    }
  }
  return total;
}

uint64_t byte_ranges_overlap_bytes(absl::Span<const ByteRange> lhs, absl::Span<const ByteRange> rhs) {
  uint64_t total = 0;
  size_t lhs_index = 0;
  size_t rhs_index = 0;
  while (lhs_index < lhs.size() && rhs_index < rhs.size()) {
    const auto& left = lhs[lhs_index];
    const auto& right = rhs[rhs_index];
    const uint64_t begin = std::max(left.begin, right.begin);
    const uint64_t end = std::min(left.end, right.end);
    if (end > begin) {
      total += end - begin;
    }
    if (left.end <= right.end) {
      ++lhs_index;
    } else {
      ++rhs_index;
    }
  }
  return total;
}

std::vector<ByteRange> subtract_byte_ranges(
    absl::Span<const ByteRange> whole_ranges,
    absl::Span<const ByteRange> removed_ranges) {
  std::vector<ByteRange> residual;
  size_t removed_index = 0;
  for (const auto& whole : whole_ranges) {
    uint64_t cursor = whole.begin;
    while (removed_index < removed_ranges.size() && removed_ranges[removed_index].end <= cursor) {
      ++removed_index;
    }
    size_t current_removed = removed_index;
    while (current_removed < removed_ranges.size()) {
      const auto& removed = removed_ranges[current_removed];
      if (removed.begin >= whole.end) {
        break;
      }
      if (removed.begin > cursor) {
        residual.push_back(ByteRange{.begin = cursor, .end = std::min<uint64_t>(removed.begin, whole.end)});
      }
      cursor = std::max<uint64_t>(cursor, removed.end);
      if (cursor >= whole.end) {
        break;
      }
      ++current_removed;
    }
    if (cursor < whole.end) {
      residual.push_back(ByteRange{.begin = cursor, .end = whole.end});
    }
    removed_index = current_removed;
  }
  return residual;
}

std::optional<TensorAxisRange> single_axis_range(const TensorCoordinateSpec& spec) {
  if (spec.selects_scalar || spec.axes.size() != 1) {
    return std::nullopt;
  }
  return spec.axes.front();
}

bool mapped_tensor_job_sources_match(const RepresentationWorkItem& lhs, const RepresentationWorkItem& rhs) {
  return lhs.partition_kind == rhs.partition_kind && lhs.dst_spec == rhs.dst_spec && lhs.sources == rhs.sources;
}

struct MappedExpertDim0Pattern {
  TensorAxisRange source_axis;
  TensorAxisRange dst_expert_axis;
  TensorAxisRange dst_value_axis;
};

struct MappedExpertDim0GroupKey {
  std::string dst_name;
  std::string ckpt_name;
  int dst_value_dim{-1};
  int64_t dst_value_start{0};
  int64_t dst_value_end{0};

  bool operator==(const MappedExpertDim0GroupKey&) const = default;
};

struct MappedExpertDim0GroupKeyHash {
  size_t operator()(const MappedExpertDim0GroupKey& key) const {
    return absl::HashOf(key.dst_name, key.ckpt_name, key.dst_value_dim, key.dst_value_start, key.dst_value_end);
  }
};

std::optional<TensorAxisRange> find_axis_by_dim(const TensorCoordinateSpec& spec, int dim) {
  if (spec.selects_scalar) {
    return std::nullopt;
  }
  for (const auto& axis : spec.axes) {
    if (axis.dim == dim) {
      return axis;
    }
  }
  return std::nullopt;
}

std::optional<MappedExpertDim0Pattern> detect_mapped_expert_dim0_pattern(const RepresentationWorkItem& item) {
  if ((item.kind != RepresentationWorkItemKind::kTensorCopy &&
       item.kind != RepresentationWorkItemKind::kExpertDim0Concat) ||
      item.sources.size() != 1 || item.partition_kind != WorkPartitionKind::kUnknown) {
    return std::nullopt;
  }
  const auto& fragment = item.sources.front().fragment;
  const auto source_axis = single_axis_range(fragment.source_range);
  if (!source_axis.has_value() || source_axis->dim != 0 || source_axis->end <= source_axis->start) {
    return std::nullopt;
  }
  if (fragment.destination_range.selects_scalar || fragment.destination_range.axes.size() != 2) {
    return std::nullopt;
  }
  const auto dst_expert_axis = find_axis_by_dim(fragment.destination_range, source_axis->dim);
  if (!dst_expert_axis.has_value() || dst_expert_axis->end <= dst_expert_axis->start) {
    return std::nullopt;
  }
  if ((dst_expert_axis->end - dst_expert_axis->start) != (source_axis->end - source_axis->start)) {
    return std::nullopt;
  }
  std::optional<TensorAxisRange> dst_value_axis;
  for (const auto& axis : fragment.destination_range.axes) {
    if (axis.dim == dst_expert_axis->dim) {
      continue;
    }
    dst_value_axis = axis;
    break;
  }
  if (!dst_value_axis.has_value() || dst_value_axis->end <= dst_value_axis->start) {
    return std::nullopt;
  }
  return MappedExpertDim0Pattern{
      .source_axis = *source_axis,
      .dst_expert_axis = *dst_expert_axis,
      .dst_value_axis = *dst_value_axis,
  };
}

MappedExpertDim0GroupKey build_mapped_expert_dim0_group_key(
    const RepresentationWorkItem& item,
    const MappedExpertDim0Pattern& pattern) {
  return MappedExpertDim0GroupKey{
      .dst_name = item.dst_name,
      .ckpt_name = item.sources.front().fragment.source_spec.name,
      .dst_value_dim = pattern.dst_value_axis.dim,
      .dst_value_start = pattern.dst_value_axis.start,
      .dst_value_end = pattern.dst_value_axis.end,
  };
}

std::uint8_t* find_tensor_base_ptr(
    const ParsedMappedParticipant& participant,
    uint64_t logical_offset,
    uint64_t length_bytes);

absl::StatusOr<std::vector<std::vector<ConcatBlockPieceRuntime>>> resolve_concat_block_pieces(
    const ParsedMappedParticipant& participant,
    uint64_t logical_begin,
    uint64_t dst_block_bytes,
    uint64_t dst_block_stride_bytes,
    uint64_t prefix_count);

absl::StatusOr<uint64_t> product_dims_from(const std::vector<int64_t>& shape, size_t start_dim) {
  uint64_t out = 1;
  for (size_t dim = start_dim; dim < shape.size(); ++dim) {
    if (shape[dim] <= 0) {
      return absl::InvalidArgumentError("tensor shape must be positive");
    }
    out *= static_cast<uint64_t>(shape[dim]);
  }
  return out;
}

absl::StatusOr<MappedTensorJobBuildResult> build_mapped_tensor_jobs(
    const std::vector<ParsedMappedParticipant>& participants) {
  MappedTensorJobBuildResult result;
  if (participants.empty()) {
    return result;
  }
  const auto allowed_data_ranges_by_rank = data_ranges_by_participant(participants);
  std::vector<absl::flat_hash_map<std::string, const RepresentationWorkItem*>> jobs_by_rank;
  jobs_by_rank.reserve(participants.size());
  for (const auto& participant : participants) {
    absl::flat_hash_map<std::string, const RepresentationWorkItem*> by_name;
    by_name.reserve(participant.work_plan.items.size());
    for (const auto& item : participant.work_plan.items) {
      if (item.kind != RepresentationWorkItemKind::kTensorCopy) {
        continue;
      }
      by_name.emplace(item.dst_name, &item);
    }
    jobs_by_rank.push_back(std::move(by_name));
  }

  std::vector<const RepresentationWorkItem*> ordered_hints;
  ordered_hints.reserve(participants.front().work_plan.items.size());
  for (const auto& item : participants.front().work_plan.items) {
    if (item.kind != RepresentationWorkItemKind::kTensorCopy) {
      continue;
    }
    ordered_hints.push_back(&item);
  }
  std::sort(ordered_hints.begin(), ordered_hints.end(), [](const auto* a, const auto* b) {
    const uint64_t lhs_offset = (!a->sources.empty()) ? a->sources.front().fragment.source_spec.logical_offset
                                                      : std::numeric_limits<uint64_t>::max();
    const uint64_t rhs_offset = (!b->sources.empty()) ? b->sources.front().fragment.source_spec.logical_offset
                                                      : std::numeric_limits<uint64_t>::max();
    if (lhs_offset != rhs_offset) {
      return lhs_offset < rhs_offset;
    }
    if (a->dst_name != b->dst_name) {
      return a->dst_name < b->dst_name;
    }
    const std::string& lhs_name = (!a->sources.empty()) ? a->sources.front().fragment.source_spec.name : a->dst_name;
    const std::string& rhs_name = (!b->sources.empty()) ? b->sources.front().fragment.source_spec.name : b->dst_name;
    return lhs_name < rhs_name;
  });

  result.handled_dst_ranges_by_rank.resize(participants.size());
  for (const auto* first : ordered_hints) {
    if (first == nullptr || first->dst_name.empty() || first->sources.size() != 1) {
      continue;
    }
    const auto& first_source = first->sources.front();
    const auto first_src_axis = single_axis_range(first_source.fragment.source_range);
    const auto first_dst_axis = single_axis_range(first_source.fragment.destination_range);
    if (first->partition_kind == WorkPartitionKind::kUnknown || first->dst_spec.logical_length == 0 ||
        first_source.fragment.source_spec.element_size == 0) {
      continue;
    }
    if (first->partition_kind == WorkPartitionKind::kReplicated) {
      continue;
    }
    if (!is_row_major_contiguous(first_source.fragment.source_spec.shape, first_source.fragment.source_spec.stride)) {
      continue;
    }
    if ((first->partition_kind == WorkPartitionKind::kDim0Partitioned ||
         first->partition_kind == WorkPartitionKind::kDim1Partitioned) &&
        (!first_src_axis.has_value() || !first_dst_axis.has_value() || first_dst_axis->start != 0)) {
      continue;
    }
    if (first->partition_kind == WorkPartitionKind::kDim1Partitioned &&
        (first_source.fragment.source_spec.shape.size() != 2 || first->dst_spec.shape.size() != 2)) {
      continue;
    }

    MappedTensorJobRuntime runtime_job;
    runtime_job.job.name = first->dst_name;
    runtime_job.job.source = TensorMeta{
        .offset = first_source.fragment.source_spec.logical_offset,
        .size_bytes = first_source.fragment.source_spec.logical_length,
        .shape = first_source.fragment.source_spec.shape,
        .stride = first_source.fragment.source_spec.stride,
        .dtype = first_source.fragment.source_spec.dtype,
        .storage_offset = first_source.fragment.source_spec.storage_offset,
        .elem_size = first_source.fragment.source_spec.element_size,
    };
    runtime_job.job.distribution = to_tensor_job_distribution(first->partition_kind);
    runtime_job.job.slices.resize(participants.size());
    runtime_job.destinations.resize(participants.size());

    bool compatible = true;
    for (size_t rank = 0; rank < participants.size(); ++rank) {
      auto it = jobs_by_rank[rank].find(first->dst_name);
      if (it == jobs_by_rank[rank].end() || it->second == nullptr) {
        compatible = false;
        break;
      }
      const auto& hint = *it->second;
      if (!mapped_tensor_job_sources_match(*first, hint) || hint.sources.size() != 1 ||
          hint.dst_spec.logical_length == 0) {
        compatible = false;
        break;
      }
      if (!byte_range_is_fully_covered(
              allowed_data_ranges_by_rank[rank],
              hint.dst_spec.logical_offset,
              hint.dst_spec.logical_offset + hint.dst_spec.logical_length)) {
        compatible = false;
        break;
      }
      const auto& hint_source = hint.sources.front();
      const auto hint_src_axis = single_axis_range(hint_source.fragment.source_range);
      std::uint8_t* dst_base_ptr = nullptr;
      for (const auto& span : participants[rank].storage_spans) {
        const uint64_t span_end = span.base_offset + span.length;
        const uint64_t tensor_end = hint.dst_spec.logical_offset + hint.dst_spec.logical_length;
        if (hint.dst_spec.logical_offset >= span.base_offset && tensor_end <= span_end) {
          dst_base_ptr = span.base_ptr.get() + (hint.dst_spec.logical_offset - span.base_offset);
          break;
        }
      }
      if (dst_base_ptr == nullptr) {
        compatible = false;
        break;
      }
      RankTensorSlice slice;
      slice.dst_offset = 0;
      slice.dst_size_bytes = hint.dst_spec.logical_length;
      switch (hint.partition_kind) {
        case WorkPartitionKind::kReplicated:
          slice.kind = RankTensorSlice::Kind::kFull;
          slice.start = 0;
          slice.length = 0;
          break;
        case WorkPartitionKind::kDim0Partitioned:
          if (!hint_src_axis.has_value() || hint_src_axis->end <= hint_src_axis->start) {
            compatible = false;
            break;
          }
          slice.kind = RankTensorSlice::Kind::kDim0;
          slice.start = hint_src_axis->start;
          slice.length = static_cast<uint64_t>(hint_src_axis->end - hint_src_axis->start);
          break;
        case WorkPartitionKind::kDim1Partitioned:
          if (!hint_src_axis.has_value() || hint_src_axis->end <= hint_src_axis->start) {
            compatible = false;
            break;
          }
          slice.kind = RankTensorSlice::Kind::kDim1;
          slice.start = hint_src_axis->start;
          slice.length = static_cast<uint64_t>(hint_src_axis->end - hint_src_axis->start);
          break;
        case WorkPartitionKind::kUnknown:
          compatible = false;
          break;
      }
      if (!compatible) {
        break;
      }
      runtime_job.job.slices[rank] = slice;
      runtime_job.destinations[rank].rank = participants[rank].rank;
      runtime_job.destinations[rank].device_id = participants[rank].device_id;
      runtime_job.destinations[rank].gpu_ptr = dst_base_ptr;
    }
    if (!compatible) {
      continue;
    }

    result.handled_source_bytes += runtime_job.job.source.size_bytes;
    result.handled_root_dst_bytes += runtime_job.job.slices.front().dst_size_bytes;
    for (size_t rank = 0; rank < participants.size(); ++rank) {
      const auto& hint = *jobs_by_rank[rank].at(first->dst_name);
      result.handled_dst_ranges_by_rank[rank].push_back(
          ByteRange{
              .begin = hint.dst_spec.logical_offset,
              .end = hint.dst_spec.logical_offset + hint.dst_spec.logical_length,
          });
    }
    result.jobs.push_back(std::move(runtime_job));
  }

  for (auto& ranges : result.handled_dst_ranges_by_rank) {
    merge_byte_ranges(&ranges);
  }
  return result;
}

absl::StatusOr<std::uint8_t*> find_mapped_destination_base_ptr(
    const ParsedMappedParticipant& participant,
    const RepresentationTensorSpec& dst_spec) {
  if (dst_spec.logical_length == 0) {
    return absl::InvalidArgumentError("mapped tensor destination has empty logical length");
  }
  for (const auto& span : participant.storage_spans) {
    const uint64_t span_end = span.base_offset + span.length;
    const uint64_t tensor_end = dst_spec.logical_offset + dst_spec.logical_length;
    if (dst_spec.logical_offset >= span.base_offset && tensor_end <= span_end) {
      return span.base_ptr.get() + (dst_spec.logical_offset - span.base_offset);
    }
  }
  return absl::InvalidArgumentError("mapped tensor destination is not contained in target storage spans");
}

TensorMeta tensor_meta_from_spec(const RepresentationTensorSpec& spec) {
  return TensorMeta{
      .offset = spec.logical_offset,
      .size_bytes = spec.logical_length,
      .shape = spec.shape,
      .stride = spec.stride,
      .dtype = spec.dtype,
      .storage_offset = spec.storage_offset,
      .elem_size = spec.element_size,
  };
}

bool destination_spans_are_covered(const std::vector<ByteRange>& ranges, absl::Span<const TensorByteSpan> spans) {
  for (const auto& span : spans) {
    if (!byte_range_is_fully_covered(ranges, span.offset, span.offset + span.length)) {
      return false;
    }
  }
  return true;
}

bool destination_spans_overlap(const std::vector<ByteRange>& ranges, absl::Span<const TensorByteSpan> spans) {
  for (const auto& span : spans) {
    if (byte_range_overlaps_any(ranges, span.offset, span.offset + span.length)) {
      return true;
    }
  }
  return false;
}

void append_destination_spans_as_ranges(absl::Span<const TensorByteSpan> spans, std::vector<ByteRange>* ranges) {
  for (const auto& span : spans) {
    if (span.length == 0) {
      continue;
    }
    ranges->push_back(ByteRange{.begin = span.offset, .end = span.offset + span.length});
  }
}

uint64_t tensor_byte_span_total_bytes(absl::Span<const TensorByteSpan> spans) {
  uint64_t total = 0;
  for (const auto& span : spans) {
    total += span.length;
  }
  return total;
}

absl::StatusOr<uint64_t> checked_mul_u64(uint64_t lhs, uint64_t rhs, std::string_view label) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    return absl::OutOfRangeError(absl::StrCat(label, " overflows uint64_t"));
  }
  return lhs * rhs;
}

absl::StatusOr<uint64_t> checked_add_u64(uint64_t lhs, uint64_t rhs, std::string_view label) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    return absl::OutOfRangeError(absl::StrCat(label, " overflows uint64_t"));
  }
  return lhs + rhs;
}

uint64_t elapsed_ns_since(std::chrono::steady_clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

double seconds_from_ns(uint64_t ns) {
  return static_cast<double>(ns) / 1'000'000'000.0;
}

absl::StatusOr<TensorAxisRange> coordinate_axis_range_or_full(
    const TensorCoordinateSpec& range,
    const RepresentationTensorSpec& tensor,
    int32_t dim,
    std::string_view role) {
  if (range.selects_scalar) {
    return absl::InvalidArgumentError(absl::StrCat(role, " scalar coordinates are unsupported for rect2d copy"));
  }
  if (tensor.shape.size() != 2 || dim < 0 || dim >= static_cast<int32_t>(tensor.shape.size())) {
    return absl::InvalidArgumentError(absl::StrCat(role, " rect2d copy requires a 2D tensor"));
  }
  for (const auto& axis : range.axes) {
    if (axis.dim < 0 || axis.dim >= static_cast<int32_t>(tensor.shape.size())) {
      return absl::InvalidArgumentError(absl::StrCat(role, " rect2d coordinate dim out of bounds"));
    }
    if (axis.dim != dim) {
      continue;
    }
    if (axis.start < 0 || axis.end <= axis.start || axis.end > tensor.shape[static_cast<size_t>(dim)]) {
      return absl::InvalidArgumentError(absl::StrCat(role, " rect2d coordinate range is invalid"));
    }
    return axis;
  }
  return TensorAxisRange{
      .dim = dim,
      .start = 0,
      .end = tensor.shape[static_cast<size_t>(dim)],
  };
}

absl::StatusOr<MappedTensorJobRuntime> build_local_partial_dim0_tensor_job(
    const ParsedMappedParticipant& participant,
    const RepresentationWorkItem& item,
    absl::Span<const TensorByteSpan> dst_spans) {
  if (dst_spans.size() != 1 || item.sources.size() != 1) {
    return absl::InvalidArgumentError("local partial dim0 tensor job requires a single contiguous destination span");
  }
  const auto& fragment = item.sources.front().fragment;
  const auto src_axis = single_axis_range(fragment.source_range);
  const auto dst_axis = single_axis_range(fragment.destination_range);
  if (!src_axis.has_value() || !dst_axis.has_value() || src_axis->dim != 0 || dst_axis->dim != 0 ||
      src_axis->end <= src_axis->start || dst_axis->end <= dst_axis->start ||
      (src_axis->end - src_axis->start) != (dst_axis->end - dst_axis->start)) {
    return absl::InvalidArgumentError("local partial dim0 tensor job requires matching dim0 source/destination axes");
  }
  if (!is_row_major_contiguous(fragment.source_spec.shape, fragment.source_spec.stride) ||
      !is_row_major_contiguous(item.dst_spec.shape, item.dst_spec.stride)) {
    return absl::InvalidArgumentError("local partial dim0 tensor job requires contiguous source and destination");
  }
  auto src_bytes_or = contiguous_dim0_slice_bytes(
      fragment.source_spec.shape, fragment.source_spec.element_size, src_axis->start, src_axis->end);
  if (!src_bytes_or.ok()) {
    return src_bytes_or.status();
  }
  auto dst_bytes_or =
      contiguous_dim0_slice_bytes(item.dst_spec.shape, item.dst_spec.element_size, dst_axis->start, dst_axis->end);
  if (!dst_bytes_or.ok()) {
    return dst_bytes_or.status();
  }
  if (*src_bytes_or != *dst_bytes_or || dst_spans.front().length != *dst_bytes_or) {
    return absl::InvalidArgumentError("local partial dim0 tensor job source/destination byte sizes differ");
  }
  auto dst_ptr_or = find_mapped_destination_base_ptr(participant, item.dst_spec);
  if (!dst_ptr_or.ok()) {
    return dst_ptr_or.status();
  }

  MappedTensorJobRuntime runtime_job;
  runtime_job.job.name = item.dst_name;
  runtime_job.job.source = tensor_meta_from_spec(fragment.source_spec);
  runtime_job.job.distribution = TensorJob::Distribution::kDim0Partitioned;
  runtime_job.job.slices.push_back(
      RankTensorSlice{
          .dst_offset = dst_spans.front().offset - item.dst_spec.logical_offset,
          .dst_size_bytes = *dst_bytes_or,
          .kind = RankTensorSlice::Kind::kDim0,
          .start = src_axis->start,
          .length = static_cast<uint64_t>(src_axis->end - src_axis->start),
      });
  runtime_job.destinations.push_back(
      ParsedParticipant{
          .rank = participant.rank,
          .device_id = participant.device_id,
          .gpu_ptr = *dst_ptr_or,
      });
  return runtime_job;
}

absl::StatusOr<MappedTensorJobRuntime> build_local_partial_replicated_tensor_job(
    const ParsedMappedParticipant& participant,
    const RepresentationWorkItem& item,
    absl::Span<const TensorByteSpan> dst_spans) {
  if (dst_spans.size() != 1 || item.sources.size() != 1) {
    return absl::InvalidArgumentError(
        "local partial replicated tensor job requires one contiguous source and destination span");
  }
  const auto& fragment = item.sources.front().fragment;
  if (fragment.source_spec.element_size == 0 || fragment.source_spec.element_size != item.dst_spec.element_size ||
      fragment.source_spec.dtype != item.dst_spec.dtype) {
    return absl::InvalidArgumentError("local partial replicated tensor job has incompatible dtypes");
  }
  auto source_spans_or =
      materialization::contracts::build_coordinate_byte_spans(fragment.source_spec, fragment.source_range);
  if (!source_spans_or.ok()) {
    return source_spans_or.status();
  }
  if (source_spans_or->size() != 1 || source_spans_or->front().length != dst_spans.front().length) {
    return absl::InvalidArgumentError("local partial replicated tensor job requires matching contiguous byte spans");
  }
  auto dst_ptr_or = find_mapped_destination_base_ptr(participant, item.dst_spec);
  if (!dst_ptr_or.ok()) {
    return dst_ptr_or.status();
  }

  MappedTensorJobRuntime runtime_job;
  runtime_job.job.name = item.dst_name;
  runtime_job.job.source = tensor_meta_from_spec(fragment.source_spec);
  runtime_job.job.source.offset = source_spans_or->front().offset;
  runtime_job.job.source.size_bytes = source_spans_or->front().length;
  runtime_job.job.source.storage_offset = 0;
  runtime_job.job.distribution = TensorJob::Distribution::kReplicated;
  runtime_job.job.slices.push_back(
      RankTensorSlice{
          .dst_offset = dst_spans.front().offset - item.dst_spec.logical_offset,
          .dst_size_bytes = dst_spans.front().length,
          .kind = RankTensorSlice::Kind::kFull,
      });
  runtime_job.destinations.push_back(
      ParsedParticipant{
          .rank = participant.rank,
          .device_id = participant.device_id,
          .gpu_ptr = *dst_ptr_or,
      });
  return runtime_job;
}

absl::StatusOr<MappedTensorJobRuntime> build_local_partial_dim1_tensor_job(
    const ParsedMappedParticipant& participant,
    const RepresentationWorkItem& item,
    absl::Span<const TensorByteSpan> dst_spans) {
  if (item.sources.size() != 1) {
    return absl::InvalidArgumentError("local partial dim1 tensor job requires one source");
  }
  const auto& fragment = item.sources.front().fragment;
  const auto src_axis = single_axis_range(fragment.source_range);
  const auto dst_axis = single_axis_range(fragment.destination_range);
  if (!src_axis.has_value() || !dst_axis.has_value() || src_axis->dim != 1 || dst_axis->dim != 1 ||
      src_axis->end <= src_axis->start || dst_axis->end <= dst_axis->start ||
      (src_axis->end - src_axis->start) != (dst_axis->end - dst_axis->start)) {
    return absl::InvalidArgumentError("local partial dim1 tensor job requires matching dim1 source/destination axes");
  }
  if (fragment.source_spec.shape.size() != 2 || item.dst_spec.shape.size() != 2 ||
      fragment.source_spec.shape[0] != item.dst_spec.shape[0]) {
    return absl::InvalidArgumentError("local partial dim1 tensor job requires matching 2D row counts");
  }
  if (!is_row_major_contiguous(fragment.source_spec.shape, fragment.source_spec.stride) ||
      !is_row_major_contiguous(item.dst_spec.shape, item.dst_spec.stride)) {
    return absl::InvalidArgumentError("local partial dim1 tensor job requires contiguous source and destination");
  }
  if (fragment.source_spec.element_size == 0 || fragment.source_spec.element_size != item.dst_spec.element_size) {
    return absl::InvalidArgumentError("local partial dim1 tensor job has invalid element sizes");
  }
  const uint64_t rows = static_cast<uint64_t>(fragment.source_spec.shape[0]);
  const uint64_t selected_cols = static_cast<uint64_t>(src_axis->end - src_axis->start);
  const uint64_t selected_bytes = rows * selected_cols * fragment.source_spec.element_size;
  uint64_t dst_span_bytes = 0;
  for (const auto& span : dst_spans) {
    dst_span_bytes += span.length;
  }
  if (selected_bytes == 0 || dst_span_bytes != selected_bytes) {
    return absl::InvalidArgumentError("local partial dim1 tensor job source/destination byte sizes differ");
  }
  auto dst_ptr_or = find_mapped_destination_base_ptr(participant, item.dst_spec);
  if (!dst_ptr_or.ok()) {
    return dst_ptr_or.status();
  }

  MappedTensorJobRuntime runtime_job;
  runtime_job.job.name = item.dst_name;
  runtime_job.job.source = tensor_meta_from_spec(fragment.source_spec);
  runtime_job.job.distribution = TensorJob::Distribution::kDim1Partitioned;
  runtime_job.job.slices.push_back(
      RankTensorSlice{
          .dst_offset = static_cast<uint64_t>(dst_axis->start) * item.dst_spec.element_size,
          .dst_size_bytes = selected_bytes,
          .dst_row_stride_bytes = static_cast<uint64_t>(item.dst_spec.shape[1]) * item.dst_spec.element_size,
          .kind = RankTensorSlice::Kind::kDim1,
          .start = src_axis->start,
          .length = selected_cols,
      });
  runtime_job.destinations.push_back(
      ParsedParticipant{
          .rank = participant.rank,
          .device_id = participant.device_id,
          .gpu_ptr = *dst_ptr_or,
      });
  return runtime_job;
}

absl::StatusOr<MappedTensorJobRuntime> build_local_partial_rect2d_tensor_job(
    const ParsedMappedParticipant& participant,
    const RepresentationWorkItem& item,
    absl::Span<const TensorByteSpan> dst_spans) {
  if (item.sources.size() != 1) {
    return absl::InvalidArgumentError("local partial rect2d tensor job requires one source");
  }
  const auto& fragment = item.sources.front().fragment;
  if (fragment.source_spec.shape.size() != 2 || item.dst_spec.shape.size() != 2) {
    return absl::InvalidArgumentError("local partial rect2d tensor job requires 2D source and destination tensors");
  }
  if (fragment.source_spec.shape[0] <= 0 || fragment.source_spec.shape[1] <= 0 || item.dst_spec.shape[0] <= 0 ||
      item.dst_spec.shape[1] <= 0) {
    return absl::InvalidArgumentError("local partial rect2d tensor job requires positive tensor dimensions");
  }
  if (!is_row_major_contiguous(fragment.source_spec.shape, fragment.source_spec.stride) ||
      !is_row_major_contiguous(item.dst_spec.shape, item.dst_spec.stride)) {
    return absl::InvalidArgumentError("local partial rect2d tensor job requires contiguous source and destination");
  }
  if (fragment.source_spec.element_size == 0 || fragment.source_spec.element_size != item.dst_spec.element_size ||
      fragment.source_spec.dtype != item.dst_spec.dtype) {
    return absl::InvalidArgumentError("local partial rect2d tensor job has incompatible dtypes");
  }

  auto src_rows_or = coordinate_axis_range_or_full(fragment.source_range, fragment.source_spec, /*dim=*/0, "source");
  if (!src_rows_or.ok()) {
    return src_rows_or.status();
  }
  auto src_cols_or = coordinate_axis_range_or_full(fragment.source_range, fragment.source_spec, /*dim=*/1, "source");
  if (!src_cols_or.ok()) {
    return src_cols_or.status();
  }
  auto dst_rows_or = coordinate_axis_range_or_full(fragment.destination_range, item.dst_spec, /*dim=*/0, "destination");
  if (!dst_rows_or.ok()) {
    return dst_rows_or.status();
  }
  auto dst_cols_or = coordinate_axis_range_or_full(fragment.destination_range, item.dst_spec, /*dim=*/1, "destination");
  if (!dst_cols_or.ok()) {
    return dst_cols_or.status();
  }

  const uint64_t row_count = static_cast<uint64_t>(src_rows_or->end - src_rows_or->start);
  const uint64_t col_count = static_cast<uint64_t>(src_cols_or->end - src_cols_or->start);
  if (row_count == 0 || col_count == 0 || row_count != static_cast<uint64_t>(dst_rows_or->end - dst_rows_or->start) ||
      col_count != static_cast<uint64_t>(dst_cols_or->end - dst_cols_or->start)) {
    return absl::InvalidArgumentError("local partial rect2d tensor job requires matching source/destination extents");
  }

  auto row_elements_or = checked_mul_u64(row_count, col_count, "local partial rect2d selected elements");
  if (!row_elements_or.ok()) {
    return row_elements_or.status();
  }
  auto selected_bytes_or =
      checked_mul_u64(*row_elements_or, fragment.source_spec.element_size, "local partial rect2d selected bytes");
  if (!selected_bytes_or.ok()) {
    return selected_bytes_or.status();
  }
  const uint64_t selected_bytes = *selected_bytes_or;
  if (selected_bytes == 0 || tensor_byte_span_total_bytes(dst_spans) != selected_bytes) {
    return absl::InvalidArgumentError("local partial rect2d tensor job source/destination byte sizes differ");
  }

  auto dst_row_stride_bytes_or = checked_mul_u64(
      static_cast<uint64_t>(item.dst_spec.shape[1]),
      item.dst_spec.element_size,
      "local partial rect2d destination row stride");
  if (!dst_row_stride_bytes_or.ok()) {
    return dst_row_stride_bytes_or.status();
  }
  auto dst_row_offset_or = checked_mul_u64(
      static_cast<uint64_t>(dst_rows_or->start),
      *dst_row_stride_bytes_or,
      "local partial rect2d destination row offset");
  if (!dst_row_offset_or.ok()) {
    return dst_row_offset_or.status();
  }
  auto dst_col_offset_or = checked_mul_u64(
      static_cast<uint64_t>(dst_cols_or->start),
      item.dst_spec.element_size,
      "local partial rect2d destination column offset");
  if (!dst_col_offset_or.ok()) {
    return dst_col_offset_or.status();
  }
  if (*dst_col_offset_or > std::numeric_limits<uint64_t>::max() - *dst_row_offset_or) {
    return absl::OutOfRangeError("local partial rect2d destination offset overflows");
  }
  auto dst_ptr_or = find_mapped_destination_base_ptr(participant, item.dst_spec);
  if (!dst_ptr_or.ok()) {
    return dst_ptr_or.status();
  }

  MappedTensorJobRuntime runtime_job;
  runtime_job.job.name = item.dst_name;
  runtime_job.job.source = tensor_meta_from_spec(fragment.source_spec);
  runtime_job.job.distribution = TensorJob::Distribution::kDim1Partitioned;
  runtime_job.job.slices.push_back(
      RankTensorSlice{
          .dst_offset = *dst_row_offset_or + *dst_col_offset_or,
          .dst_size_bytes = selected_bytes,
          .dst_row_stride_bytes = *dst_row_stride_bytes_or,
          .kind = RankTensorSlice::Kind::kRect2D,
          .start = 0,
          .length = 0,
          .row_start = static_cast<uint64_t>(src_rows_or->start),
          .row_count = row_count,
          .src_col_start = static_cast<uint64_t>(src_cols_or->start),
          .col_count = col_count,
      });
  runtime_job.destinations.push_back(
      ParsedParticipant{
          .rank = participant.rank,
          .device_id = participant.device_id,
          .gpu_ptr = *dst_ptr_or,
      });
  return runtime_job;
}

absl::StatusOr<MappedTensorJobBuildResult> build_local_mapped_partial_tensor_jobs(
    const ParsedMappedParticipant& participant,
    absl::Span<const ByteRange> initially_handled_ranges) {
  MappedTensorJobBuildResult result;
  result.handled_dst_ranges_by_rank.resize(1);
  std::vector<ByteRange> lane_ranges = data_ranges_from_lane_map(participant.collective_lane_map);
  std::vector<ByteRange> handled_ranges(initially_handled_ranges.begin(), initially_handled_ranges.end());
  merge_byte_ranges(&handled_ranges);

  size_t considered = 0;
  size_t accepted_replicated = 0;
  size_t accepted_dim0 = 0;
  size_t accepted_dim1 = 0;
  size_t accepted_rect2d = 0;
  size_t skipped_lane = 0;
  size_t skipped_already = 0;
  size_t skipped_overlap = 0;
  size_t skipped_unsupported = 0;
  uint64_t accepted_bytes = 0;
  uint64_t skipped_lane_bytes = 0;
  uint64_t skipped_already_bytes = 0;
  uint64_t skipped_overlap_bytes = 0;
  uint64_t skipped_unsupported_bytes = 0;

  for (const auto& item : participant.work_plan.items) {
    if (item.kind != RepresentationWorkItemKind::kTensorCopy || item.sources.size() != 1) {
      continue;
    }
    considered += 1;
    auto dst_spans_or = materialization::contracts::build_coordinate_byte_spans(
        item.dst_spec, item.sources.front().fragment.destination_range);
    if (!dst_spans_or.ok()) {
      return dst_spans_or.status();
    }
    const auto& dst_spans = *dst_spans_or;
    const uint64_t dst_span_bytes = tensor_byte_span_total_bytes(absl::MakeSpan(dst_spans));
    if (!destination_spans_are_covered(lane_ranges, absl::MakeSpan(dst_spans))) {
      skipped_lane += 1;
      skipped_lane_bytes += dst_span_bytes;
      continue;
    }
    if (destination_spans_are_covered(handled_ranges, absl::MakeSpan(dst_spans))) {
      skipped_already += 1;
      skipped_already_bytes += dst_span_bytes;
      continue;
    }
    if (destination_spans_overlap(handled_ranges, absl::MakeSpan(dst_spans))) {
      skipped_overlap += 1;
      skipped_overlap_bytes += dst_span_bytes;
      continue;
    }

    absl::StatusOr<MappedTensorJobRuntime> runtime_job_or =
        absl::UnimplementedError("unsupported local partial tensor work item");
    enum class AcceptedKind : uint8_t { kNone = 0, kReplicated = 1, kDim0 = 2, kDim1 = 3, kRect2D = 4 };
    AcceptedKind accepted_kind = AcceptedKind::kNone;
    if (item.partition_kind == WorkPartitionKind::kReplicated) {
      runtime_job_or = build_local_partial_replicated_tensor_job(participant, item, absl::MakeSpan(dst_spans));
      if (runtime_job_or.ok()) {
        accepted_kind = AcceptedKind::kReplicated;
      }
    } else if (item.partition_kind == WorkPartitionKind::kDim0Partitioned) {
      runtime_job_or = build_local_partial_dim0_tensor_job(participant, item, absl::MakeSpan(dst_spans));
      if (runtime_job_or.ok()) {
        accepted_kind = AcceptedKind::kDim0;
      }
    } else if (item.partition_kind == WorkPartitionKind::kDim1Partitioned) {
      runtime_job_or = build_local_partial_dim1_tensor_job(participant, item, absl::MakeSpan(dst_spans));
      if (runtime_job_or.ok()) {
        accepted_kind = AcceptedKind::kDim1;
      }
    }
    if (!runtime_job_or.ok() &&
        (absl::IsInvalidArgument(runtime_job_or.status()) || absl::IsUnimplemented(runtime_job_or.status()))) {
      runtime_job_or = build_local_partial_rect2d_tensor_job(participant, item, absl::MakeSpan(dst_spans));
      if (runtime_job_or.ok()) {
        accepted_kind = AcceptedKind::kRect2D;
      }
    }
    if (!runtime_job_or.ok()) {
      if (absl::IsInvalidArgument(runtime_job_or.status()) || absl::IsUnimplemented(runtime_job_or.status())) {
        skipped_unsupported += 1;
        skipped_unsupported_bytes += dst_span_bytes;
        continue;
      }
      return runtime_job_or.status();
    }

    switch (accepted_kind) {
      case AcceptedKind::kReplicated:
        accepted_replicated += 1;
        break;
      case AcceptedKind::kDim0:
        accepted_dim0 += 1;
        break;
      case AcceptedKind::kDim1:
        accepted_dim1 += 1;
        break;
      case AcceptedKind::kRect2D:
        accepted_rect2d += 1;
        break;
      case AcceptedKind::kNone:
        return absl::InternalError("local partial tensor job accepted without a kind");
    }
    accepted_bytes += runtime_job_or->job.slices.front().dst_size_bytes;
    append_destination_spans_as_ranges(absl::MakeSpan(dst_spans), &result.handled_dst_ranges_by_rank.front());
    append_destination_spans_as_ranges(absl::MakeSpan(dst_spans), &handled_ranges);
    merge_byte_ranges(&handled_ranges);
    result.jobs.push_back(std::move(*runtime_job_or));
  }

  merge_byte_ranges(&result.handled_dst_ranges_by_rank.front());
  LOG(INFO) << "local_mapped_partial_tensor_job_summary"
            << " considered=" << considered << " accepted_replicated=" << accepted_replicated
            << " accepted_dim0=" << accepted_dim0 << " accepted_dim1=" << accepted_dim1
            << " accepted_rect2d=" << accepted_rect2d << " accepted_bytes=" << accepted_bytes
            << " skipped_lane=" << skipped_lane << " skipped_lane_bytes=" << skipped_lane_bytes
            << " skipped_already=" << skipped_already << " skipped_already_bytes=" << skipped_already_bytes
            << " skipped_overlap=" << skipped_overlap << " skipped_overlap_bytes=" << skipped_overlap_bytes
            << " skipped_unsupported=" << skipped_unsupported
            << " skipped_unsupported_bytes=" << skipped_unsupported_bytes;
  result.handled_source_bytes = accepted_bytes;
  result.handled_root_dst_bytes = accepted_bytes;
  return result;
}

absl::StatusOr<MappedConcatJobBuildResult> build_mapped_expert_dim0_concat_jobs(
    const std::vector<ParsedMappedParticipant>& participants,
    const std::vector<std::vector<ByteRange>>& allowed_data_ranges_by_rank) {
  MappedConcatJobBuildResult result;
  result.handled_dst_ranges_by_rank.resize(participants.size());
  if (participants.empty()) {
    return result;
  }

  std::vector<std::unordered_map<
      MappedExpertDim0GroupKey,
      std::vector<const RepresentationWorkItem*>,
      MappedExpertDim0GroupKeyHash>>
      expert_items_by_rank;
  expert_items_by_rank.reserve(participants.size());
  std::vector<MappedExpertDim0GroupKey> ordered_expert_keys;
  std::unordered_set<MappedExpertDim0GroupKey, MappedExpertDim0GroupKeyHash> seen_expert_keys;
  for (size_t rank = 0; rank < participants.size(); ++rank) {
    std::unordered_map<
        MappedExpertDim0GroupKey,
        std::vector<const RepresentationWorkItem*>,
        MappedExpertDim0GroupKeyHash>
        by_key;
    by_key.reserve(participants[rank].work_plan.items.size());
    for (const auto& item : participants[rank].work_plan.items) {
      const auto pattern = detect_mapped_expert_dim0_pattern(item);
      if (!pattern.has_value()) {
        continue;
      }
      const auto key = build_mapped_expert_dim0_group_key(item, *pattern);
      by_key[key].push_back(&item);
      if (rank == 0 && seen_expert_keys.insert(key).second) {
        ordered_expert_keys.push_back(key);
      }
    }
    expert_items_by_rank.push_back(std::move(by_key));
  }

  size_t accepted_jobs = 0;
  for (const auto& key : ordered_expert_keys) {
    const auto first_rank_it = expert_items_by_rank.front().find(key);
    if (first_rank_it == expert_items_by_rank.front().end() || first_rank_it->second.empty()) {
      continue;
    }
    const auto* first = first_rank_it->second.front();
    if (first == nullptr || first->sources.size() != 1 ||
        !is_row_major_contiguous(first->dst_spec.shape, first->dst_spec.stride)) {
      continue;
    }
    const auto first_pattern = detect_mapped_expert_dim0_pattern(*first);
    if (!first_pattern.has_value() || first_pattern->dst_expert_axis.dim != 0 ||
        first_pattern->dst_value_axis.dim != 1 ||
        (first_pattern->source_axis.end - first_pattern->source_axis.start) != 1 ||
        (first_pattern->dst_expert_axis.end - first_pattern->dst_expert_axis.start) != 1) {
      continue;
    }
    const auto& first_source_spec = first->sources.front().fragment.source_spec;
    if (!is_row_major_contiguous(first_source_spec.shape, first_source_spec.stride) ||
        first->dst_spec.shape.size() < 2) {
      continue;
    }

    const auto source_block_bytes_or =
        contiguous_dim0_slice_bytes(first_source_spec.shape, first_source_spec.element_size, 0, 1);
    if (!source_block_bytes_or.ok()) {
      return source_block_bytes_or.status();
    }
    const auto dst_expert_stride_bytes_or =
        contiguous_dim0_slice_bytes(first->dst_spec.shape, first->dst_spec.element_size, 0, 1);
    if (!dst_expert_stride_bytes_or.ok()) {
      return dst_expert_stride_bytes_or.status();
    }
    const auto dst_tail_elems_or = product_dims_from(first->dst_spec.shape, /*start_dim=*/2);
    if (!dst_tail_elems_or.ok()) {
      return dst_tail_elems_or.status();
    }
    const uint64_t source_block_bytes = *source_block_bytes_or;
    const uint64_t dst_expert_stride_bytes = *dst_expert_stride_bytes_or;
    const uint64_t dst_value_tail_bytes = (*dst_tail_elems_or) * first->dst_spec.element_size;
    const uint64_t dst_block_bytes =
        static_cast<uint64_t>(first_pattern->dst_value_axis.end - first_pattern->dst_value_axis.start) *
        dst_value_tail_bytes;
    const uint64_t dst_value_offset_bytes =
        static_cast<uint64_t>(first_pattern->dst_value_axis.start) * dst_value_tail_bytes;
    if (dst_block_bytes != source_block_bytes) {
      continue;
    }

    MappedConcatJobRuntime job;
    job.name = absl::StrCat(first->dst_name, "::", key.ckpt_name, "::expert_dim0_concat");
    job.prefix_count = first_rank_it->second.size();
    job.destinations.resize(participants.size());

    MappedConcatFragmentRuntime fragment;
    fragment.source = TensorMeta{
        .offset = first_source_spec.logical_offset,
        .size_bytes = first_source_spec.logical_length,
        .shape = first_source_spec.shape,
        .stride = first_source_spec.stride,
        .dtype = first_source_spec.dtype,
        .storage_offset = first_source_spec.storage_offset,
        .elem_size = first_source_spec.element_size,
    };
    fragment.prefix_count = first_rank_it->second.size();
    fragment.src_block_bytes = source_block_bytes;
    fragment.dst_block_offset_bytes = 0;
    fragment.dst_block_stride_bytes = dst_expert_stride_bytes;
    fragment.dst_block_bytes = dst_block_bytes;
    fragment.dst_logical_begins_by_rank.resize(participants.size(), 0);
    fragment.dst_ptrs.resize(participants.size(), nullptr);
    fragment.dst_block_pieces_by_rank.resize(participants.size());
    fragment.src_starts_by_rank.resize(participants.size(), 0);
    fragment.src_ends_by_rank.resize(participants.size(), 0);

    int64_t global_src_start = std::numeric_limits<int64_t>::max();
    int64_t global_src_end = std::numeric_limits<int64_t>::min();
    bool compatible = true;
    std::vector<std::vector<ByteRange>> pending_ranges_by_rank(participants.size());
    for (size_t rank = 0; rank < participants.size(); ++rank) {
      auto rank_it = expert_items_by_rank[rank].find(key);
      if (rank_it == expert_items_by_rank[rank].end() || rank_it->second.empty() ||
          rank_it->second.size() != first_rank_it->second.size()) {
        compatible = false;
        break;
      }
      auto rank_items = rank_it->second;
      std::sort(rank_items.begin(), rank_items.end(), [](const auto* lhs, const auto* rhs) {
        if (lhs == nullptr || rhs == nullptr) {
          return lhs < rhs;
        }
        const auto lhs_pattern = detect_mapped_expert_dim0_pattern(*lhs);
        const auto rhs_pattern = detect_mapped_expert_dim0_pattern(*rhs);
        if (!lhs_pattern.has_value() || !rhs_pattern.has_value()) {
          return lhs < rhs;
        }
        if (lhs_pattern->dst_expert_axis.start != rhs_pattern->dst_expert_axis.start) {
          return lhs_pattern->dst_expert_axis.start < rhs_pattern->dst_expert_axis.start;
        }
        return lhs_pattern->source_axis.start < rhs_pattern->source_axis.start;
      });

      const auto* first_rank_item = rank_items.front();
      const auto first_rank_pattern =
          first_rank_item != nullptr ? detect_mapped_expert_dim0_pattern(*first_rank_item) : std::nullopt;
      if (first_rank_item == nullptr || !first_rank_pattern.has_value()) {
        compatible = false;
        break;
      }
      if (!is_row_major_contiguous(first_rank_item->dst_spec.shape, first_rank_item->dst_spec.stride) ||
          first_rank_item->dst_spec.shape != first->dst_spec.shape ||
          first_rank_item->dst_spec.stride != first->dst_spec.stride ||
          first_rank_item->dst_spec.dtype != first->dst_spec.dtype ||
          first_rank_item->dst_spec.element_size != first->dst_spec.element_size) {
        compatible = false;
        break;
      }

      const uint64_t rank_logical_begin = first_rank_item->dst_spec.logical_offset +
          static_cast<uint64_t>(first_rank_pattern->dst_expert_axis.start) * dst_expert_stride_bytes +
          dst_value_offset_bytes;
      int64_t expected_source_cursor = first_rank_pattern->source_axis.start;
      int64_t expected_dst_expert_cursor = first_rank_pattern->dst_expert_axis.start;
      int64_t source_start = first_rank_pattern->source_axis.start;
      int64_t source_end = first_rank_pattern->source_axis.start;
      for (size_t item_index = 0; item_index < rank_items.size(); ++item_index) {
        const auto* item = rank_items[item_index];
        const auto pattern = item != nullptr ? detect_mapped_expert_dim0_pattern(*item) : std::nullopt;
        if (item == nullptr || !pattern.has_value() || item->sources.size() != 1 ||
            item->sources.front().fragment.source_spec != first_source_spec ||
            pattern->dst_expert_axis !=
                TensorAxisRange{.dim = 0, .start = expected_dst_expert_cursor, .end = expected_dst_expert_cursor + 1} ||
            pattern->dst_value_axis != first_pattern->dst_value_axis ||
            pattern->source_axis.start != expected_source_cursor ||
            pattern->source_axis.end != expected_source_cursor + 1) {
          compatible = false;
          break;
        }
        const uint64_t actual_logical_begin = item->dst_spec.logical_offset +
            static_cast<uint64_t>(pattern->dst_expert_axis.start) * dst_expert_stride_bytes + dst_value_offset_bytes;
        const uint64_t expected_logical_begin = rank_logical_begin + item_index * dst_expert_stride_bytes;
        if (actual_logical_begin != expected_logical_begin) {
          compatible = false;
          break;
        }
        pending_ranges_by_rank[rank].push_back(
            ByteRange{
                .begin = actual_logical_begin,
                .end = actual_logical_begin + dst_block_bytes,
            });
        if (!byte_range_is_fully_covered(
                allowed_data_ranges_by_rank[rank], actual_logical_begin, actual_logical_begin + dst_block_bytes)) {
          compatible = false;
          break;
        }
        expected_source_cursor = pattern->source_axis.end;
        expected_dst_expert_cursor = pattern->dst_expert_axis.end;
        source_end = pattern->source_axis.end;
      }
      if (!compatible || source_end <= source_start) {
        compatible = false;
        break;
      }

      fragment.dst_logical_begins_by_rank[rank] = rank_logical_begin;
      auto* dst_base_ptr = find_tensor_base_ptr(
          participants[rank], rank_logical_begin, dst_block_bytes + (rank_items.size() - 1) * dst_expert_stride_bytes);
      if (dst_base_ptr != nullptr) {
        fragment.dst_ptrs[rank] = dst_base_ptr;
      } else {
        auto pieces_or = resolve_concat_block_pieces(
            participants[rank], rank_logical_begin, dst_block_bytes, dst_expert_stride_bytes, rank_items.size());
        if (!pieces_or.ok()) {
          compatible = false;
          break;
        }
        fragment.dst_block_pieces_by_rank[rank] = std::move(*pieces_or);
      }
      fragment.src_starts_by_rank[rank] = source_start;
      fragment.src_ends_by_rank[rank] = source_end;
      global_src_start = std::min(global_src_start, source_start);
      global_src_end = std::max(global_src_end, source_end);
      job.destinations[rank].rank = participants[rank].rank;
      job.destinations[rank].device_id = participants[rank].device_id;
    }

    if (!compatible || global_src_end <= global_src_start) {
      continue;
    }

    job.fragments.push_back(std::move(fragment));
    for (size_t rank = 0; rank < participants.size(); ++rank) {
      result.handled_dst_ranges_by_rank[rank].insert(
          result.handled_dst_ranges_by_rank[rank].end(),
          pending_ranges_by_rank[rank].begin(),
          pending_ranges_by_rank[rank].end());
    }
    result.handled_source_bytes += static_cast<uint64_t>(global_src_end - global_src_start) * source_block_bytes;
    result.handled_root_dst_bytes += static_cast<uint64_t>(pending_ranges_by_rank.front().size()) * dst_block_bytes;
    result.jobs.push_back(std::move(job));
    accepted_jobs += 1;
  }

  LOG(INFO) << "mapped_expert_dim0_concat_job_summary"
            << " requested=" << ordered_expert_keys.size() << " accepted=" << accepted_jobs
            << " handled_source_bytes=" << result.handled_source_bytes
            << " handled_root_dst_bytes=" << result.handled_root_dst_bytes;
  for (auto& ranges : result.handled_dst_ranges_by_rank) {
    merge_byte_ranges(&ranges);
  }
  return result;
}

std::uint8_t* find_tensor_base_ptr(
    const ParsedMappedParticipant& participant,
    uint64_t logical_offset,
    uint64_t length_bytes) {
  for (const auto& span : participant.storage_spans) {
    const uint64_t span_end = span.base_offset + span.length;
    const uint64_t tensor_end = logical_offset + length_bytes;
    if (logical_offset >= span.base_offset && tensor_end <= span_end) {
      return span.base_ptr.get() + (logical_offset - span.base_offset);
    }
  }
  return nullptr;
}

absl::StatusOr<std::vector<std::vector<ConcatBlockPieceRuntime>>> resolve_concat_block_pieces(
    const ParsedMappedParticipant& participant,
    uint64_t logical_begin,
    uint64_t dst_block_bytes,
    uint64_t dst_block_stride_bytes,
    uint64_t prefix_count) {
  std::vector<std::vector<ConcatBlockPieceRuntime>> block_pieces(prefix_count);
  for (uint64_t block_idx = 0; block_idx < prefix_count; ++block_idx) {
    const uint64_t block_begin = logical_begin + block_idx * dst_block_stride_bytes;
    const uint64_t block_end = block_begin + dst_block_bytes;
    uint64_t cursor = block_begin;
    for (const auto& span : participant.storage_spans) {
      const uint64_t span_begin = span.base_offset;
      const uint64_t span_end = span.base_offset + span.length;
      if (span_end <= cursor) {
        continue;
      }
      if (span_begin >= block_end) {
        break;
      }
      if (span_begin > cursor) {
        return absl::FailedPreconditionError("concat block target span has uncovered gap");
      }
      const uint64_t piece_begin = cursor;
      const uint64_t piece_end = std::min<uint64_t>(block_end, span_end);
      block_pieces[static_cast<size_t>(block_idx)].push_back(
          ConcatBlockPieceRuntime{
              .block_offset = piece_begin - block_begin,
              .length = piece_end - piece_begin,
              .dst_ptr = span.base_ptr.get() + (piece_begin - span_begin),
          });
      cursor = piece_end;
      if (cursor == block_end) {
        break;
      }
    }
    if (cursor != block_end) {
      return absl::FailedPreconditionError("concat block target span extends beyond storage coverage");
    }
  }
  return block_pieces;
}

absl::StatusOr<MappedConcatJobBuildResult> build_mapped_concat_jobs(
    const std::vector<ParsedMappedParticipant>& participants,
    const StrategyConfig& strategy) {
  MappedConcatJobBuildResult result;
  result.handled_dst_ranges_by_rank.resize(participants.size());
  if (participants.empty()) {
    return result;
  }
  const auto allowed_data_ranges_by_rank = data_ranges_by_participant(participants);

  std::vector<absl::flat_hash_map<std::string, const RepresentationWorkItem*>> jobs_by_rank;
  jobs_by_rank.reserve(participants.size());
  for (const auto& participant : participants) {
    absl::flat_hash_map<std::string, const RepresentationWorkItem*> by_name;
    by_name.reserve(participant.work_plan.items.size());
    for (const auto& item : participant.work_plan.items) {
      if (item.kind != RepresentationWorkItemKind::kConcatAssemble) {
        continue;
      }
      by_name.emplace(item.dst_name, &item);
    }
    jobs_by_rank.push_back(std::move(by_name));
  }

  std::vector<const RepresentationWorkItem*> concat_items;
  concat_items.reserve(participants.front().work_plan.items.size());
  for (const auto& item : participants.front().work_plan.items) {
    if (item.kind == RepresentationWorkItemKind::kConcatAssemble) {
      concat_items.push_back(&item);
    }
  }

  for (const auto* hint : concat_items) {
    if (hint == nullptr || hint->sources.empty() || hint->dst_spec.logical_length == 0) {
      LOG(INFO) << "mapped_concat_job_skip"
                << " dst_name=" << (hint != nullptr ? hint->dst_name : std::string("<null>")) << " reason=empty_hint";
      continue;
    }
    const uint64_t prefix_count = hint->sources.front().prefix_count;
    if (prefix_count == 0) {
      continue;
    }
    if (prefix_count > 1 && !enable_mapped_multirange_concat_jobs(strategy)) {
      continue;
    }
    if (prefix_count == 1 && !enable_mapped_single_range_concat_jobs(strategy)) {
      continue;
    }
    MappedConcatJobRuntime job;
    job.name = hint->dst_name;
    job.dst_base_offset = hint->dst_spec.logical_offset;
    job.dst_size_bytes = hint->dst_spec.logical_length;
    job.prefix_count = prefix_count;
    job.destinations.resize(participants.size());
    job.fragments.reserve(hint->sources.size());

    bool compatible = true;
    std::vector<const RepresentationWorkItem*> rank_hints(participants.size(), nullptr);
    for (size_t rank = 0; rank < participants.size(); ++rank) {
      auto it = jobs_by_rank[rank].find(hint->dst_name);
      if (it == jobs_by_rank[rank].end() || it->second == nullptr) {
        if (verbose_mapped_concat_diagnostics(strategy)) {
          LOG(INFO) << "mapped_concat_job_skip"
                    << " dst_name=" << hint->dst_name << " reason=missing_rank_hint"
                    << " rank=" << rank;
        }
        compatible = false;
        break;
      }
      rank_hints[rank] = it->second;
      const uint64_t rank_prefix_count = it->second->sources.empty() ? 0 : it->second->sources.front().prefix_count;
      if (it->second->sources.size() != hint->sources.size() ||
          it->second->dst_spec.logical_length != hint->dst_spec.logical_length || rank_prefix_count != prefix_count) {
        if (verbose_mapped_concat_diagnostics(strategy)) {
          LOG(INFO) << "mapped_concat_job_skip"
                    << " dst_name=" << hint->dst_name << " reason=rank_hint_shape_mismatch"
                    << " rank=" << rank << " expected_sources=" << hint->sources.size()
                    << " actual_sources=" << it->second->sources.size()
                    << " expected_dst_size_bytes=" << hint->dst_spec.logical_length
                    << " actual_dst_size_bytes=" << it->second->dst_spec.logical_length
                    << " expected_prefix_count=" << prefix_count << " actual_prefix_count=" << rank_prefix_count;
        }
        compatible = false;
        break;
      }
      if (!byte_range_is_fully_covered(
              allowed_data_ranges_by_rank[rank],
              it->second->dst_spec.logical_offset,
              it->second->dst_spec.logical_offset + it->second->dst_spec.logical_length)) {
        compatible = false;
        break;
      }
      job.destinations[rank].rank = participants[rank].rank;
      job.destinations[rank].device_id = participants[rank].device_id;
    }
    if (!compatible) {
      continue;
    }

    for (size_t source_idx = 0; source_idx < hint->sources.size(); ++source_idx) {
      const auto& source = hint->sources[source_idx];
      const auto src_axis = single_axis_range(source.fragment.source_range);
      if (!src_axis.has_value() || src_axis->dim != 0 || source.prefix_count == 0 || source.dst_block_bytes == 0 ||
          src_axis->end <= src_axis->start) {
        LOG(INFO) << "mapped_concat_job_skip"
                  << " dst_name=" << hint->dst_name << " reason=invalid_source_fragment"
                  << " src_name=" << source.fragment.source_spec.name;
        compatible = false;
        break;
      }
      auto src_block_bytes_or = contiguous_dim0_slice_bytes(
          source.fragment.source_spec.shape, source.fragment.source_spec.element_size, src_axis->start, src_axis->end);
      if (!src_block_bytes_or.ok()) {
        return src_block_bytes_or.status();
      }
      if (source.dst_block_bytes != *src_block_bytes_or || source.dst_block_stride_bytes < source.dst_block_bytes) {
        if (verbose_mapped_concat_diagnostics(strategy)) {
          LOG(INFO) << "mapped_concat_job_skip"
                    << " dst_name=" << hint->dst_name << " reason=unsupported_block_geometry"
                    << " src_name=" << source.fragment.source_spec.name << " src_block_bytes=" << *src_block_bytes_or
                    << " dst_block_bytes=" << source.dst_block_bytes
                    << " dst_block_stride_bytes=" << source.dst_block_stride_bytes;
        }
        compatible = false;
        break;
      }
      std::vector<void*> dst_ptrs(participants.size(), nullptr);
      std::vector<uint64_t> dst_logical_begins_by_rank(participants.size(), 0);
      std::vector<std::vector<std::vector<ConcatBlockPieceRuntime>>> dst_block_pieces_by_rank(participants.size());
      std::vector<int64_t> src_starts_by_rank(participants.size(), 0);
      std::vector<int64_t> src_ends_by_rank(participants.size(), 0);
      for (size_t rank = 0; rank < participants.size(); ++rank) {
        const auto& rank_source = rank_hints[rank]->sources[source_idx];
        const auto rank_src_axis = single_axis_range(rank_source.fragment.source_range);
        if (!rank_src_axis.has_value() || rank_src_axis->dim != 0 ||
            rank_source.fragment.source_spec.name != source.fragment.source_spec.name ||
            rank_source.fragment.source_spec.shape != source.fragment.source_spec.shape ||
            rank_source.fragment.source_spec.stride != source.fragment.source_spec.stride ||
            rank_source.fragment.source_spec.dtype != source.fragment.source_spec.dtype ||
            rank_source.fragment.source_spec.element_size != source.fragment.source_spec.element_size ||
            rank_source.dst_block_offset_bytes != source.dst_block_offset_bytes ||
            rank_source.dst_block_stride_bytes != source.dst_block_stride_bytes ||
            rank_source.dst_block_bytes != source.dst_block_bytes || rank_source.prefix_count != source.prefix_count ||
            rank_src_axis->end <= rank_src_axis->start) {
          if (verbose_mapped_concat_diagnostics(strategy)) {
            LOG(INFO) << "mapped_concat_job_skip"
                      << " dst_name=" << hint->dst_name << " reason=rank_source_mismatch"
                      << " rank=" << rank << " src_idx=" << source_idx
                      << " src_name=" << source.fragment.source_spec.name
                      << " rank_src_name=" << rank_source.fragment.source_spec.name << " src_start=" << src_axis->start
                      << " src_end=" << src_axis->end << " rank_src_start=" << rank_src_axis->start
                      << " rank_src_end=" << rank_src_axis->end
                      << " dst_block_offset_bytes=" << source.dst_block_offset_bytes
                      << " rank_dst_block_offset_bytes=" << rank_source.dst_block_offset_bytes
                      << " dst_block_stride_bytes=" << source.dst_block_stride_bytes
                      << " rank_dst_block_stride_bytes=" << rank_source.dst_block_stride_bytes
                      << " dst_block_bytes=" << source.dst_block_bytes
                      << " rank_dst_block_bytes=" << rank_source.dst_block_bytes
                      << " prefix_count=" << source.prefix_count << " rank_prefix_count=" << rank_source.prefix_count;
          }
          compatible = false;
          break;
        }
        const int64_t src_block_rows = rank_src_axis->end - rank_src_axis->start;
        if (src_block_rows <= 0) {
          compatible = false;
          break;
        }
        src_starts_by_rank[rank] = rank_src_axis->start;
        src_ends_by_rank[rank] = rank_src_axis->start + static_cast<int64_t>(rank_source.prefix_count) * src_block_rows;
        const uint64_t logical_begin = hint->dst_spec.logical_offset + rank_source.dst_block_offset_bytes;
        const uint64_t logical_span =
            source.dst_block_bytes + (source.prefix_count - 1) * source.dst_block_stride_bytes;
        dst_logical_begins_by_rank[rank] = logical_begin;
        auto* dst_base_ptr = find_tensor_base_ptr(participants[rank], logical_begin, logical_span);
        if (dst_base_ptr != nullptr) {
          dst_ptrs[rank] = dst_base_ptr;
          continue;
        }
        auto pieces_or = resolve_concat_block_pieces(
            participants[rank],
            logical_begin,
            source.dst_block_bytes,
            source.dst_block_stride_bytes,
            source.prefix_count);
        if (!pieces_or.ok()) {
          LOG(INFO) << "mapped_concat_job_skip"
                    << " dst_name=" << hint->dst_name << " reason=fragment_target_not_single_storage"
                    << " rank=" << rank << " src_name=" << source.fragment.source_spec.name
                    << " status=" << pieces_or.status();
          compatible = false;
          break;
        }
        dst_block_pieces_by_rank[rank] = std::move(*pieces_or);
      }
      if (!compatible) {
        break;
      }
      job.fragments.push_back(
          MappedConcatFragmentRuntime{
              .source =
                  TensorMeta{
                      .offset = source.fragment.source_spec.logical_offset,
                      .size_bytes = source.fragment.source_spec.logical_length,
                      .shape = source.fragment.source_spec.shape,
                      .stride = source.fragment.source_spec.stride,
                      .dtype = source.fragment.source_spec.dtype,
                      .storage_offset = source.fragment.source_spec.storage_offset,
                      .elem_size = source.fragment.source_spec.element_size,
                  },
              .src_start = src_axis->start,
              .src_end = src_axis->end,
              .src_starts_by_rank = std::move(src_starts_by_rank),
              .src_ends_by_rank = std::move(src_ends_by_rank),
              .prefix_count = source.prefix_count,
              .src_block_bytes = *src_block_bytes_or,
              .dst_block_offset_bytes = source.dst_block_offset_bytes,
              .dst_block_stride_bytes = source.dst_block_stride_bytes,
              .dst_block_bytes = source.dst_block_bytes,
              .dst_logical_begins_by_rank = std::move(dst_logical_begins_by_rank),
              .dst_ptrs = std::move(dst_ptrs),
              .dst_block_pieces_by_rank = std::move(dst_block_pieces_by_rank),
          });
      result.handled_source_bytes += source.prefix_count * (*src_block_bytes_or);
    }
    if (!compatible) {
      continue;
    }
    for (size_t rank = 0; rank < participants.size(); ++rank) {
      result.handled_dst_ranges_by_rank[rank].push_back(
          ByteRange{
              .begin = hint->dst_spec.logical_offset,
              .end = hint->dst_spec.logical_offset + hint->dst_spec.logical_length,
          });
    }
    result.handled_root_dst_bytes += hint->dst_spec.logical_length;
    result.jobs.push_back(std::move(job));
  }

  LOG(INFO) << "mapped_concat_job_summary"
            << " requested=" << concat_items.size() << " accepted=" << result.jobs.size()
            << " handled_source_bytes=" << result.handled_source_bytes
            << " handled_root_dst_bytes=" << result.handled_root_dst_bytes;

  auto expert_concat_jobs_or = build_mapped_expert_dim0_concat_jobs(participants, allowed_data_ranges_by_rank);
  if (!expert_concat_jobs_or.ok()) {
    return expert_concat_jobs_or.status();
  }
  auto expert_concat_jobs = std::move(*expert_concat_jobs_or);
  result.handled_source_bytes += expert_concat_jobs.handled_source_bytes;
  result.handled_root_dst_bytes += expert_concat_jobs.handled_root_dst_bytes;
  result.jobs.insert(
      result.jobs.end(),
      std::make_move_iterator(expert_concat_jobs.jobs.begin()),
      std::make_move_iterator(expert_concat_jobs.jobs.end()));
  for (size_t rank = 0; rank < participants.size(); ++rank) {
    result.handled_dst_ranges_by_rank[rank].insert(
        result.handled_dst_ranges_by_rank[rank].end(),
        expert_concat_jobs.handled_dst_ranges_by_rank[rank].begin(),
        expert_concat_jobs.handled_dst_ranges_by_rank[rank].end());
  }

  for (auto& ranges : result.handled_dst_ranges_by_rank) {
    merge_byte_ranges(&ranges);
  }
  return result;
}

absl::StatusOr<std::vector<MappedSegmentRef>> build_mapped_segment_refs(
    const std::vector<ParsedMappedParticipant>& participants,
    const std::vector<std::vector<ByteRange>>& handled_dst_ranges_by_rank) {
  std::vector<MappedSegmentRef> segments;
  for (size_t participant_index = 0; participant_index < participants.size(); ++participant_index) {
    const auto& participant = participants[participant_index];
    if (participant.collective_lane_map.num_sources != 1) {
      return absl::InvalidArgumentError("mapped collective load requires mapping.num_sources == 1");
    }
    const std::vector<ByteRange> empty_ranges;
    const auto& handled_ranges = participant_index < handled_dst_ranges_by_rank.size()
        ? handled_dst_ranges_by_rank[participant_index]
        : empty_ranges;
    std::vector<loader::ByteRangeSegment> lane_segments;
    lane_segments.reserve(participant.collective_lane_map.segments.size());
    for (const auto& segment : participant.collective_lane_map.segments) {
      if (segment.kind != loader::ByteRangeSegment::Kind::kData) {
        return absl::InvalidArgumentError("mapped collective load requires data-only collective lane map");
      }
      if (segment.source_index != 0) {
        return absl::InvalidArgumentError("mapped collective load requires source_index == 0");
      }
      if (segment.length == 0) {
        continue;
      }
      lane_segments.push_back(segment);
    }
    std::sort(lane_segments.begin(), lane_segments.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.dst_offset != rhs.dst_offset) {
        return lhs.dst_offset < rhs.dst_offset;
      }
      if (lhs.src_offset != rhs.src_offset) {
        return lhs.src_offset < rhs.src_offset;
      }
      return lhs.length < rhs.length;
    });
    const auto lane_ranges = data_ranges_from_lane_map(participant.collective_lane_map);
    const auto residual_ranges = subtract_byte_ranges(absl::MakeSpan(lane_ranges), absl::MakeSpan(handled_ranges));
    size_t lane_index = 0;
    for (const auto& residual : residual_ranges) {
      uint64_t cursor = residual.begin;
      while (lane_index < lane_segments.size() &&
             lane_segments[lane_index].dst_offset + lane_segments[lane_index].length <= cursor) {
        ++lane_index;
      }
      while (cursor < residual.end) {
        if (lane_index >= lane_segments.size()) {
          return absl::InvalidArgumentError("mapped residual range extends beyond lane map");
        }
        const auto& segment = lane_segments[lane_index];
        const uint64_t segment_end = segment.dst_offset + segment.length;
        if (segment.dst_offset > cursor || segment_end <= cursor) {
          return absl::InvalidArgumentError("mapped residual range is not covered by lane map");
        }
        const uint64_t take_end = std::min<uint64_t>(residual.end, segment_end);
        segments.push_back(
            MappedSegmentRef{
                .rank = participant.rank,
                .src_offset = segment.src_offset + (cursor - segment.dst_offset),
                .dst_offset = cursor,
                .length = take_end - cursor,
            });
        cursor = take_end;
        if (cursor >= segment_end) {
          ++lane_index;
        }
      }
    }
  }
  std::sort(segments.begin(), segments.end(), [](const MappedSegmentRef& a, const MappedSegmentRef& b) {
    if (a.src_offset != b.src_offset) {
      return a.src_offset < b.src_offset;
    }
    if (a.length != b.length) {
      return a.length < b.length;
    }
    if (a.rank != b.rank) {
      return a.rank < b.rank;
    }
    return a.dst_offset < b.dst_offset;
  });
  return segments;
}

loader::ByteRangeMap build_data_map_from_segment_refs(
    absl::Span<const MappedSegmentRef> segment_refs,
    uint64_t total_bytes) {
  loader::ByteRangeMap map;
  map.total_bytes = total_bytes;
  map.num_sources = 1;
  map.segments.reserve(segment_refs.size());
  for (const auto& segment : segment_refs) {
    if (segment.length == 0) {
      continue;
    }
    map.segments.push_back(
        loader::ByteRangeSegment{
            .kind = loader::ByteRangeSegment::Kind::kData,
            .dst_offset = segment.dst_offset,
            .length = segment.length,
            .src_offset = segment.src_offset,
            .source_index = 0,
        });
  }
  return map;
}

uint64_t mapped_segment_ref_covered_bytes(absl::Span<const MappedSegmentRef> segment_refs) {
  uint64_t total = 0;
  for (const auto& segment : segment_refs) {
    total += segment.length;
  }
  return total;
}

absl::StatusOr<std::vector<MappedSourceWindow>> build_mapped_source_windows(
    const std::vector<MappedSegmentRef>& segments,
    const CollectiveMappedTargetLoadOptions& options) {
  std::vector<MappedSourceWindow> windows;
  if (segments.empty()) {
    return windows;
  }
  const uint64_t merge_gap = options.merge_max_gap_bytes;
  const uint64_t max_amp = std::max<uint64_t>(1, options.merge_max_amplification);
  for (size_t idx = 0; idx < segments.size(); ++idx) {
    const auto& seg = segments[idx];
    const uint64_t seg_end = seg.src_offset + seg.length;
    if (windows.empty()) {
      windows.push_back(
          MappedSourceWindow{
              .start = seg.src_offset,
              .end = seg_end,
              .covered_bytes = seg.length,
              .segment_indices = {idx},
          });
      continue;
    }
    auto& window = windows.back();
    const uint64_t gap = seg.src_offset > window.end ? (seg.src_offset - window.end) : 0;
    const uint64_t new_end = std::max<uint64_t>(window.end, seg_end);
    const uint64_t new_span = new_end - window.start;
    const uint64_t added_unique = seg_end > window.end ? (seg_end - std::max<uint64_t>(window.end, seg.src_offset)) : 0;
    const uint64_t new_covered = window.covered_bytes + added_unique;
    if (gap <= merge_gap && new_span <= new_covered * max_amp) {
      window.end = new_end;
      window.covered_bytes = new_covered;
      window.segment_indices.push_back(idx);
      continue;
    }
    windows.push_back(
        MappedSourceWindow{
            .start = seg.src_offset,
            .end = seg_end,
            .covered_bytes = seg.length,
            .segment_indices = {idx},
        });
  }
  return windows;
}

struct TargetPiece {
  gsl::not_null<std::uint8_t*> dst_ptr{reinterpret_cast<std::uint8_t*>(1)};
  uint64_t length{0};
  uint64_t src_offset{0};
};

struct CopyPiece {
  const std::uint8_t* src_ptr{nullptr};
  std::uint8_t* dst_ptr{nullptr};
  uint64_t length{0};
};

bool can_merge_copy_piece(const CopyPiece& prev, const CopyPiece& next) {
  return prev.length > 0 && next.length > 0 && prev.src_ptr != nullptr && prev.dst_ptr != nullptr &&
      next.src_ptr != nullptr && next.dst_ptr != nullptr && prev.src_ptr + prev.length == next.src_ptr &&
      prev.dst_ptr + prev.length == next.dst_ptr;
}

void append_merged_copy_piece(std::vector<CopyPiece>& pieces, CopyPiece piece) {
  if (piece.length == 0 || piece.src_ptr == nullptr || piece.dst_ptr == nullptr) {
    return;
  }
  if (!pieces.empty() && can_merge_copy_piece(pieces.back(), piece)) {
    pieces.back().length += piece.length;
    return;
  }
  pieces.push_back(piece);
}

absl::StatusOr<std::vector<TargetPiece>> resolve_target_pieces(
    const ParsedMappedParticipant& participant,
    uint64_t logical_offset,
    uint64_t length) {
  if (length == 0) {
    return std::vector<TargetPiece>{};
  }
  if (logical_offset > std::numeric_limits<uint64_t>::max() - length) {
    return absl::OutOfRangeError("mapped collective target logical range overflows");
  }
  const uint64_t logical_end = logical_offset + length;
  std::vector<TargetPiece> pieces;
  uint64_t cursor = logical_offset;
  while (cursor < logical_end) {
    bool found = false;
    for (const auto& span : participant.storage_spans) {
      const uint64_t span_end = span.base_offset + span.length;
      if (cursor < span.base_offset || cursor >= span_end) {
        continue;
      }
      const uint64_t take = std::min<uint64_t>(logical_end, span_end) - cursor;
      const uint64_t span_local_offset = cursor - span.base_offset;
      pieces.push_back(
          TargetPiece{
              .dst_ptr = gsl::not_null<std::uint8_t*>{span.base_ptr.get() + span_local_offset},
              .length = take,
              .src_offset = cursor - logical_offset,
          });
      cursor += take;
      found = true;
      break;
    }
    if (!found) {
      return absl::InvalidArgumentError("mapped collective target logical range is outside storage layout");
    }
  }
  return pieces;
}

absl::Status execute_replicated_tensor(
    const TensorJob& job,
    const std::vector<ParsedParticipant>& participants,
    loader::SeekableSource& source,
    NcclClique& clique,
    char* host_buffer,
    size_t host_buffer_bytes,
    void* root_stage_ptr,
    cudaStream_t h2d_stream,
    cudaEvent_t ready_event,
    int root_rank) {
  const auto source_base_offset_or = source_base_offset_bytes(job.source);
  if (!source_base_offset_or.ok()) {
    return source_base_offset_or.status();
  }
  const uint64_t source_base_offset = *source_base_offset_or;
  uint64_t copied = 0;
  while (copied < job.source.size_bytes) {
    const size_t chunk_bytes = static_cast<size_t>(
        std::min<uint64_t>(job.source.size_bytes - copied, static_cast<uint64_t>(host_buffer_bytes)));
    TC_RETURN_IF_ERROR(read_exact(source, source_base_offset + copied, host_buffer, chunk_bytes));
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
    TC_RETURN_IF_ERROR(
        tensorcast::cuda::memcpy_async(root_stage_ptr, host_buffer, chunk_bytes, cudaMemcpyHostToDevice, h2d_stream));
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(ready_event, h2d_stream));
    TC_RETURN_IF_ERROR(clique.wait_stream_on_event(root_rank, ready_event));
    TC_RETURN_IF_ERROR(clique.group_start());
    for (size_t idx = 0; idx < participants.size(); ++idx) {
      auto* dst_ptr = static_cast<uint8_t*>(participants[idx].gpu_ptr) + job.slices[idx].dst_offset + copied;
      const void* send_ptr = (static_cast<int>(idx) == root_rank) ? root_stage_ptr : dst_ptr;
      TC_RETURN_IF_ERROR(clique.broadcast_u8(static_cast<int>(idx), send_ptr, dst_ptr, chunk_bytes, root_rank));
    }
    TC_RETURN_IF_ERROR(clique.group_end());
    TC_RETURN_IF_ERROR(clique.synchronize_all());
    copied += static_cast<uint64_t>(chunk_bytes);
  }
  return absl::OkStatus();
}

absl::Status execute_dim0_tensor(
    const TensorJob& job,
    const std::vector<ParsedParticipant>& participants,
    loader::SeekableSource& source,
    NcclClique& clique,
    char* host_buffer,
    size_t host_buffer_bytes,
    void* root_stage_ptr,
    cudaStream_t h2d_stream,
    cudaEvent_t ready_event,
    int root_rank) {
  const auto source_base_offset_or = source_base_offset_bytes(job.source);
  if (!source_base_offset_or.ok()) {
    return source_base_offset_or.status();
  }
  const uint64_t source_base_offset = *source_base_offset_or;
  double read_sec = 0.0;
  double h2d_sec = 0.0;
  double wait_ready_sec = 0.0;
  double issue_sec = 0.0;
  double sync_sec = 0.0;
  double root_d2d_sec = 0.0;
  size_t chunk_count = 0;
  uint64_t peer_transfer_bytes = 0;
  const auto total_start = std::chrono::steady_clock::now();
  uint64_t per_row_bytes = job.source.elem_size;
  for (size_t dim = 1; dim < job.source.shape.size(); ++dim) {
    per_row_bytes *= static_cast<uint64_t>(job.source.shape[dim]);
  }
  const uint64_t full_bytes = job.source.size_bytes;
  uint64_t copied = 0;
  while (copied < full_bytes) {
    chunk_count += 1;
    const size_t chunk_bytes =
        static_cast<size_t>(std::min<uint64_t>(full_bytes - copied, static_cast<uint64_t>(host_buffer_bytes)));
    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(read_exact(source, source_base_offset + copied, host_buffer, chunk_bytes));
      read_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(
          tensorcast::cuda::memcpy_async(root_stage_ptr, host_buffer, chunk_bytes, cudaMemcpyHostToDevice, h2d_stream));
      TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(ready_event, h2d_stream));
      h2d_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(clique.wait_stream_on_event(root_rank, ready_event));
      wait_ready_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }

    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(clique.group_start());
      for (size_t idx = 0; idx < participants.size(); ++idx) {
        const auto& slice = job.slices[idx];
        const uint64_t slice_begin = static_cast<uint64_t>(slice.start) * per_row_bytes;
        const uint64_t slice_end = slice_begin + slice.dst_size_bytes;
        const uint64_t chunk_begin = copied;
        const uint64_t chunk_end = copied + chunk_bytes;
        const uint64_t overlap_begin = std::max<uint64_t>(slice_begin, chunk_begin);
        const uint64_t overlap_end = std::min<uint64_t>(slice_end, chunk_end);
        if (overlap_end <= overlap_begin) {
          continue;
        }
        const size_t overlap_bytes = static_cast<size_t>(overlap_end - overlap_begin);
        const uint64_t src_off = overlap_begin - chunk_begin;
        const uint64_t dst_off = slice.dst_offset + (overlap_begin - slice_begin);
        auto* dst_ptr = static_cast<uint8_t*>(participants[idx].gpu_ptr) + dst_off;
        const auto* src_ptr = static_cast<const uint8_t*>(root_stage_ptr) + src_off;
        if (static_cast<int>(idx) == root_rank) {
          const auto local_copy_start = std::chrono::steady_clock::now();
          TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[idx].device_id));
          TC_RETURN_IF_ERROR(
              tensorcast::cuda::memcpy_async(
                  dst_ptr, src_ptr, overlap_bytes, cudaMemcpyDeviceToDevice, clique.stream(root_rank)));
          root_d2d_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - local_copy_start).count();
        } else {
          peer_transfer_bytes += overlap_bytes;
          TC_RETURN_IF_ERROR(clique.send_u8(root_rank, src_ptr, overlap_bytes, static_cast<int>(idx)));
          TC_RETURN_IF_ERROR(clique.recv_u8(static_cast<int>(idx), dst_ptr, overlap_bytes, root_rank));
        }
      }
      TC_RETURN_IF_ERROR(clique.group_end());
      issue_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(clique.synchronize_all());
      sync_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    copied += static_cast<uint64_t>(chunk_bytes);
  }
  const auto total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  LOG(INFO) << "collective_dim0_job name=" << job.name << " bytes=" << full_bytes << " chunks=" << chunk_count
            << " read=" << read_sec << "s h2d=" << h2d_sec << "s wait_ready=" << wait_ready_sec
            << "s issue=" << issue_sec << "s sync=" << sync_sec << "s root_d2d=" << root_d2d_sec
            << "s peer_transfer_bytes=" << peer_transfer_bytes << " total=" << total_sec << "s";
  return absl::OkStatus();
}

absl::Status execute_dim1_tensor(
    const TensorJob& job,
    const std::vector<ParsedParticipant>& participants,
    loader::SeekableSource& source,
    NcclClique& clique,
    char* host_buffer,
    size_t host_buffer_bytes,
    void* root_stage_ptr,
    cudaStream_t h2d_stream,
    cudaEvent_t ready_event,
    int root_rank,
    Dim1PackWorkspace* workspace = nullptr) {
  const auto source_base_offset_or = source_base_offset_bytes(job.source);
  if (!source_base_offset_or.ok()) {
    return source_base_offset_or.status();
  }
  const uint64_t source_base_offset = *source_base_offset_or;
  double pack_alloc_sec = 0.0;
  double read_sec = 0.0;
  double h2d_sec = 0.0;
  double pack_sec = 0.0;
  double issue_sec = 0.0;
  double sync_sec = 0.0;
  size_t chunk_count = 0;
  uint64_t peer_transfer_bytes = 0;
  const auto total_start = std::chrono::steady_clock::now();
  if (job.source.shape.size() != 2) {
    return absl::UnimplementedError("dim1 collective tensor must be 2D");
  }
  const uint64_t rows = static_cast<uint64_t>(job.source.shape[0]);
  const uint64_t cols = static_cast<uint64_t>(job.source.shape[1]);
  const uint64_t row_bytes = cols * job.source.elem_size;
  if (row_bytes == 0) {
    return absl::InvalidArgumentError("dim1 collective tensor has zero row_bytes");
  }
  const uint64_t rows_per_chunk = std::max<uint64_t>(1, static_cast<uint64_t>(host_buffer_bytes) / row_bytes);
  Dim1PackWorkspace local_workspace;
  Dim1PackWorkspace& pack_workspace = workspace != nullptr ? *workspace : local_workspace;
  if (pack_workspace.pack_buffers.size() < participants.size()) {
    pack_workspace.pack_buffers.resize(participants.size());
  }
  size_t required_capacity_bytes = 0;
  for (size_t idx = 0; idx < participants.size(); ++idx) {
    if (static_cast<int>(idx) == root_rank) {
      continue;
    }
    const uint64_t col_bytes = job.slices[idx].dst_size_bytes / std::max<uint64_t>(1, rows);
    if (col_bytes == 0) {
      continue;
    }
    required_capacity_bytes = std::max(required_capacity_bytes, static_cast<size_t>(rows_per_chunk * col_bytes));
  }
  if (required_capacity_bytes > pack_workspace.capacity_bytes) {
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
    const auto alloc_start = std::chrono::steady_clock::now();
    for (size_t idx = 0; idx < participants.size(); ++idx) {
      if (static_cast<int>(idx) == root_rank) {
        continue;
      }
      const uint64_t col_bytes = job.slices[idx].dst_size_bytes / std::max<uint64_t>(1, rows);
      if (col_bytes == 0) {
        continue;
      }
      pack_workspace.pack_buffers[idx] = std::make_unique<common::memory::GpuDeviceMemory>();
      TC_RETURN_IF_ERROR(pack_workspace.pack_buffers[idx]->allocate(
          required_capacity_bytes, participants[static_cast<size_t>(root_rank)].device_id));
    }
    pack_workspace.capacity_bytes = required_capacity_bytes;
    pack_alloc_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - alloc_start).count();
  }

  if (pack_workspace.pack_stream == nullptr) {
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&pack_workspace.pack_stream, cudaStreamNonBlocking));
  }
  cudaStream_t pack_stream = pack_workspace.pack_stream;

  for (uint64_t row = 0; row < rows; row += rows_per_chunk) {
    chunk_count += 1;
    const uint64_t chunk_rows = std::min<uint64_t>(rows_per_chunk, rows - row);
    const uint64_t chunk_bytes = chunk_rows * row_bytes;
    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(
          read_exact(source, source_base_offset + row * row_bytes, host_buffer, static_cast<size_t>(chunk_bytes)));
      read_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(
          tensorcast::cuda::memcpy_async(
              root_stage_ptr, host_buffer, static_cast<size_t>(chunk_bytes), cudaMemcpyHostToDevice, h2d_stream));
      TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(ready_event, h2d_stream));
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_wait_event(pack_stream, ready_event));
      h2d_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }

    {
      const auto step_start = std::chrono::steady_clock::now();
      for (size_t idx = 0; idx < participants.size(); ++idx) {
        const auto& slice = job.slices[idx];
        const uint64_t col_bytes = slice.dst_size_bytes / std::max<uint64_t>(1, rows);
        const uint64_t dst_pitch_bytes = slice.dst_row_stride_bytes == 0 ? col_bytes : slice.dst_row_stride_bytes;
        const uint64_t src_col_bytes = static_cast<uint64_t>(slice.start) * job.source.elem_size;
        auto* dst_ptr = static_cast<uint8_t*>(participants[idx].gpu_ptr) + slice.dst_offset + row * dst_pitch_bytes;
        const auto* src_ptr = static_cast<const uint8_t*>(root_stage_ptr) + src_col_bytes;
        if (static_cast<int>(idx) == root_rank) {
          SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
              dst_ptr,
              static_cast<size_t>(dst_pitch_bytes),
              src_ptr,
              static_cast<size_t>(row_bytes),
              static_cast<size_t>(col_bytes),
              static_cast<size_t>(chunk_rows),
              cudaMemcpyDeviceToDevice,
              pack_stream));
          continue;
        }
        auto* pack_ptr = static_cast<uint8_t*>(pack_workspace.pack_buffers[idx]->get());
        SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
            pack_ptr,
            static_cast<size_t>(col_bytes),
            src_ptr,
            static_cast<size_t>(row_bytes),
            static_cast<size_t>(col_bytes),
            static_cast<size_t>(chunk_rows),
            cudaMemcpyDeviceToDevice,
            pack_stream));
      }
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(pack_stream));
      pack_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(clique.group_start());
      for (size_t idx = 0; idx < participants.size(); ++idx) {
        if (static_cast<int>(idx) == root_rank) {
          continue;
        }
        const auto& slice = job.slices[idx];
        const uint64_t col_bytes = slice.dst_size_bytes / std::max<uint64_t>(1, rows);
        const uint64_t dst_pitch_bytes = slice.dst_row_stride_bytes == 0 ? col_bytes : slice.dst_row_stride_bytes;
        const size_t send_bytes = static_cast<size_t>(chunk_rows * col_bytes);
        auto* dst_ptr = static_cast<uint8_t*>(participants[idx].gpu_ptr) + slice.dst_offset + row * dst_pitch_bytes;
        if (dst_pitch_bytes != col_bytes && chunk_rows > 1) {
          return absl::UnimplementedError("collective dim1 peer receive does not support strided destinations");
        }
        peer_transfer_bytes += send_bytes;
        TC_RETURN_IF_ERROR(
            clique.send_u8(root_rank, pack_workspace.pack_buffers[idx]->get(), send_bytes, static_cast<int>(idx)));
        TC_RETURN_IF_ERROR(clique.recv_u8(static_cast<int>(idx), dst_ptr, send_bytes, root_rank));
      }
      TC_RETURN_IF_ERROR(clique.group_end());
      issue_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(clique.synchronize_all());
      sync_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
  }

  if (workspace == nullptr && pack_stream != nullptr) {
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_destroy(pack_stream));
  }
  const auto total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  LOG(INFO) << "collective_dim1_job name=" << job.name << " bytes=" << job.source.size_bytes << " rows=" << rows
            << " rows_per_chunk=" << rows_per_chunk << " chunks=" << chunk_count << " pack_alloc=" << pack_alloc_sec
            << "s read=" << read_sec << "s h2d=" << h2d_sec << "s pack=" << pack_sec << "s issue=" << issue_sec
            << "s sync=" << sync_sec << "s peer_transfer_bytes=" << peer_transfer_bytes << " total=" << total_sec
            << "s";
  return absl::OkStatus();
}

uint64_t total_segment_bytes(const std::vector<SegmentCopy>& segments) {
  uint64_t total = 0;
  for (const auto& segment : segments) {
    total += segment.bytes;
  }
  return total;
}

absl::StatusOr<LocalBatchedExecutionPlan> build_local_batched_execution_plan(const std::vector<TensorJob>& jobs) {
  LocalBatchedExecutionPlan plan;
  plan.jobs = jobs;
  plan.summary.eligible = true;
  plan.summary.reason = "eligible";
  plan.direct_segments.reserve(jobs.size());
  plan.direct_dedup_copies.reserve(jobs.size());
  plan.dim1_jobs.reserve(jobs.size());

  absl::flat_hash_map<std::string, SegmentCopy> primary_reads_by_key;
  primary_reads_by_key.reserve(jobs.size());

  for (const auto& job : jobs) {
    const auto& slice = job.slices.front();
    if (slice.dst_size_bytes == 0) {
      continue;
    }
    switch (job.distribution) {
      case TensorJob::Distribution::kReplicated: {
        const std::string key = absl::StrCat(job.source.offset, ":", slice.dst_size_bytes);
        auto [it, inserted] = primary_reads_by_key.try_emplace(
            key,
            SegmentCopy{
                .src_offset = job.source.offset,
                .dst_offset = slice.dst_offset,
                .bytes = static_cast<size_t>(slice.dst_size_bytes),
            });
        if (inserted) {
          plan.direct_segments.push_back(it->second);
        } else if (it->second.dst_offset != slice.dst_offset) {
          plan.direct_dedup_copies.push_back(
              LocalDedupCopy{
                  .src_dst_offset = it->second.dst_offset,
                  .dst_offset = slice.dst_offset,
                  .bytes = static_cast<size_t>(slice.dst_size_bytes),
              });
        }
        plan.summary.replicated_jobs += 1;
        plan.summary.requested_source_bytes += slice.dst_size_bytes;
        break;
      }
      case TensorJob::Distribution::kDim0Partitioned: {
        uint64_t per_row_bytes = job.source.elem_size;
        for (size_t dim = 1; dim < job.source.shape.size(); ++dim) {
          per_row_bytes *= static_cast<uint64_t>(job.source.shape[dim]);
        }
        const uint64_t src_offset = job.source.offset + static_cast<uint64_t>(slice.start) * per_row_bytes;
        const std::string key = absl::StrCat(src_offset, ":", slice.dst_size_bytes);
        auto [it, inserted] = primary_reads_by_key.try_emplace(
            key,
            SegmentCopy{
                .src_offset = src_offset,
                .dst_offset = slice.dst_offset,
                .bytes = static_cast<size_t>(slice.dst_size_bytes),
            });
        if (inserted) {
          plan.direct_segments.push_back(it->second);
        } else if (it->second.dst_offset != slice.dst_offset) {
          plan.direct_dedup_copies.push_back(
              LocalDedupCopy{
                  .src_dst_offset = it->second.dst_offset,
                  .dst_offset = slice.dst_offset,
                  .bytes = static_cast<size_t>(slice.dst_size_bytes),
              });
        }
        plan.summary.dim0_jobs += 1;
        plan.summary.requested_source_bytes += slice.dst_size_bytes;
        break;
      }
      case TensorJob::Distribution::kDim1Partitioned:
        plan.dim1_jobs.push_back(job);
        plan.summary.dim1_jobs += 1;
        plan.summary.requested_source_bytes += job.source.size_bytes;
        break;
    }
  }

  plan.direct_segments = merge_adjacent_segments_by_src(std::move(plan.direct_segments));
  plan.summary.unique_source_bytes = total_segment_bytes(plan.direct_segments);
  for (const auto& job : plan.dim1_jobs) {
    plan.summary.unique_source_bytes += job.source.size_bytes;
  }
  for (const auto& copy : plan.direct_dedup_copies) {
    plan.summary.direct_dedup_copy_bytes += copy.bytes;
  }
  plan.summary.dedup_saving_bytes = plan.summary.requested_source_bytes > plan.summary.unique_source_bytes
      ? plan.summary.requested_source_bytes - plan.summary.unique_source_bytes
      : 0;
  const uint64_t direct_peak_bytes = 0;
  const uint64_t dim1_peak_bytes = plan.dim1_jobs.empty() ? 0 : 1024ULL * 1024ULL * 1024ULL;
  plan.summary.peak_temporary_bytes = std::max<uint64_t>(direct_peak_bytes, dim1_peak_bytes);
  plan.summary.batch_count =
      static_cast<uint64_t>(plan.direct_segments.size()) + static_cast<uint64_t>(plan.dim1_jobs.size());
  if (plan.direct_segments.empty() && plan.dim1_jobs.empty()) {
    plan.summary.eligible = false;
    plan.summary.reason = "no_supported_tensor_jobs";
  }
  return plan;
}

absl::Status execute_local_dedup_copies(absl::Span<const LocalDedupCopy> copies, void* gpu_ptr, int device_id) {
  if (copies.empty()) {
    return absl::OkStatus();
  }
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_id));
  cudaStream_t d2d_stream = nullptr;
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&d2d_stream, cudaStreamNonBlocking));
  for (const auto& copy : copies) {
    auto* src_ptr = static_cast<std::uint8_t*>(gpu_ptr) + copy.src_dst_offset;
    auto* dst_ptr = static_cast<std::uint8_t*>(gpu_ptr) + copy.dst_offset;
    TC_RETURN_IF_ERROR(
        tensorcast::cuda::memcpy_async(dst_ptr, src_ptr, copy.bytes, cudaMemcpyDeviceToDevice, d2d_stream));
  }
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(d2d_stream));
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_destroy(d2d_stream));
  return absl::OkStatus();
}

absl::Status execute_local_dim1_jobs(
    const std::vector<TensorJob>& jobs,
    loader::SeekableSource& source,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    void* gpu_ptr,
    int device_id) {
  if (jobs.empty()) {
    return absl::OkStatus();
  }
  const size_t io_threads = 4;
  const size_t host_chunk_bytes = pinned_pool->slice_bytes();
  const uint64_t staging_bytes = 1024ull * 1024ull * 1024ull;
  auto staging = std::make_unique<common::memory::GpuDeviceMemory>();
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_id));
  TC_RETURN_IF_ERROR(staging->allocate(static_cast<size_t>(staging_bytes), device_id));
  cudaStream_t pack_stream = nullptr;
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&pack_stream, cudaStreamNonBlocking));

  auto session_spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
      /*num_chunks=*/std::max<size_t>(2, 8), host_chunk_bytes, pinned_pool);
  TC_RETURN_IF_ERROR(session_spb->initialize(pinned_timeout, absl::StrCat("local_batched_dim1 device=", device_id)));
  loader::StreamingBufferAdapter adapter(session_spb);

  for (const auto& job : jobs) {
    if (job.distribution != TensorJob::Distribution::kDim1Partitioned) {
      continue;
    }
    const auto& slice = job.slices.front();
    if (job.source.shape.size() != 2 || slice.dst_size_bytes == 0) {
      return absl::UnimplementedError("local batched dim1 load requires non-empty 2D tensors");
    }
    const uint64_t rows = static_cast<uint64_t>(job.source.shape[0]);
    const uint64_t cols = static_cast<uint64_t>(job.source.shape[1]);
    const uint64_t row_bytes = cols * job.source.elem_size;
    const uint64_t col_bytes = slice.dst_size_bytes / std::max<uint64_t>(1, rows);
    const uint64_t src_col_bytes = static_cast<uint64_t>(slice.start) * job.source.elem_size;
    const uint64_t rows_per_chunk = std::max<uint64_t>(1, staging_bytes / std::max<uint64_t>(1, row_bytes));

    for (uint64_t row = 0; row < rows; row += rows_per_chunk) {
      const uint64_t chunk_rows = std::min<uint64_t>(rows_per_chunk, rows - row);
      const uint64_t chunk_bytes = chunk_rows * row_bytes;
      std::vector<RemappedSource::Segment> remap;
      remap.push_back(
          RemappedSource::Segment{
              .dst_offset = 0,
              .src_offset = job.source.offset + row * row_bytes,
              .end_offset = chunk_bytes,
          });
      RemappedSource remapped(gsl::not_null<loader::SeekableSource*>{&source}, std::move(remap));
      loader::GpuMemorySink sink(
          loader::GpuMemorySink::Options{
              .gpu_base_ptr = gsl::not_null<void*>{staging->get()},
              .total_size = 0,
              .chunk_size = host_chunk_bytes,
              .device_id = device_id,
              .allocation = nullptr,
              .gpu_sched_enabled = true,
              .gpu_sched_limit_bytes = loader::DEFAULT_GPU_SCHED_LIMIT_BYTES,
              .gpu_sched_limit_copies = loader::DEFAULT_GPU_SCHED_LIMIT_COPIES,
          });
      TC_RETURN_IF_ERROR(adapter.get_buffer()->reset_for_new_production());
      auto ranges = split_even_ranges(/*base=*/0, chunk_bytes, io_threads);
      TC_RETURN_IF_ERROR(
          loader::pump_ranges(
              remapped, sink, adapter, ranges, io_threads, whole_source_load_runtime().blocking_executor()));
      TC_RETURN_IF_ERROR(sink.close());
      auto* dst_ptr = static_cast<std::uint8_t*>(gpu_ptr) + slice.dst_offset + row * col_bytes;
      auto* src_ptr = static_cast<std::uint8_t*>(staging->get()) + src_col_bytes;
      SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
          dst_ptr,
          static_cast<size_t>(col_bytes),
          src_ptr,
          static_cast<size_t>(row_bytes),
          static_cast<size_t>(col_bytes),
          static_cast<size_t>(chunk_rows),
          cudaMemcpyDeviceToDevice,
          pack_stream));
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(pack_stream));
    }
  }
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_destroy(pack_stream));
  return absl::OkStatus();
}

LocalBatchedDiskLoadResult try_local_batched_disk_load_impl(
    const LocalBatchedDiskLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout) {
  if (!enable_local_batched_disk_load(request.strategy_config)) {
    return {.handled = false, .status = absl::OkStatus(), .skip_reason = "strategy_disabled"};
  }
  if (request.disk_context == nullptr || request.representation_work_plan.items.empty() || request.gpu_ptr == nullptr ||
      request.device_id < 0) {
    return local_batched_fallback(request.replica_key.artifact_id, "missing_prerequisites");
  }

  ParsedParticipant participant{
      .replica_key = request.replica_key,
      .rank = 0,
      .device_id = request.device_id,
      .gpu_ptr = request.gpu_ptr,
      .gpu_allocation = request.gpu_allocation,
      .disk_context = request.disk_context,
      .representation_work_plan = request.representation_work_plan,
  };
  auto jobs_or = build_tensor_jobs(std::vector<ParsedParticipant>{participant});
  if (!jobs_or.ok()) {
    // Local batched disk load is a best-effort executor. Shape/view patterns it
    // cannot represent must fall back to the generic byte-range path rather
    // than shrinking mapped-view correctness.
    if (absl::IsUnimplemented(jobs_or.status())) {
      return local_batched_fallback(request.replica_key.artifact_id, "unsupported_tensor_jobs", jobs_or.status());
    }
    return {.handled = true, .status = jobs_or.status()};
  }

  auto local_plan_or = build_local_batched_execution_plan(*jobs_or);
  if (!local_plan_or.ok()) {
    return {.handled = true, .status = local_plan_or.status()};
  }
  if (!local_plan_or->summary.eligible) {
    return local_batched_fallback(request.replica_key.artifact_id, local_plan_or->summary.reason);
  }

  loader::MultiSafetensorsSource backing_source(request.disk_context->safetensors_segments());

  const auto total_start = std::chrono::steady_clock::now();
  double direct_sec = 0.0;
  if (!local_plan_or->direct_segments.empty()) {
    std::vector<RemappedSource::Segment> remap;
    remap.reserve(local_plan_or->direct_segments.size());
    for (const auto& segment : local_plan_or->direct_segments) {
      remap.push_back(
          RemappedSource::Segment{
              .dst_offset = segment.dst_offset,
              .src_offset = segment.src_offset,
              .end_offset = segment.dst_offset + segment.bytes,
          });
    }
    RemappedSource remapped(gsl::not_null<loader::SeekableSource*>{&backing_source}, std::move(remap));
    auto session_spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
        /*num_chunks=*/std::max<size_t>(2, 8), pinned_pool->slice_bytes(), pinned_pool);
    {
      const absl::Status init_status = session_spb->initialize(
          pinned_timeout, absl::StrCat("local_batched_direct artifact_id=", request.replica_key.artifact_id));
      if (!init_status.ok()) {
        return {.handled = true, .status = init_status};
      }
    }
    loader::StreamingBufferAdapter adapter(session_spb);
    loader::GpuMemorySink sink(
        loader::GpuMemorySink::Options{
            .gpu_base_ptr = gsl::not_null<void*>{request.gpu_ptr},
            .total_size = 0,
            .chunk_size = pinned_pool->slice_bytes(),
            .device_id = request.device_id,
            .allocation = request.gpu_allocation,
            .gpu_sched_enabled = true,
            .gpu_sched_limit_bytes = loader::DEFAULT_GPU_SCHED_LIMIT_BYTES,
            .gpu_sched_limit_copies = loader::DEFAULT_GPU_SCHED_LIMIT_COPIES,
        });
    auto ranges_or = build_pump_ranges_for_copy(local_plan_or->direct_segments, /*io_threads=*/4);
    if (!ranges_or.ok()) {
      return {.handled = true, .status = ranges_or.status()};
    }
    const auto direct_start = std::chrono::steady_clock::now();
    {
      const absl::Status pump_status = loader::pump_ranges(
          remapped, sink, adapter, *ranges_or, /*concurrency=*/4, whole_source_load_runtime().blocking_executor());
      if (!pump_status.ok()) {
        return {.handled = true, .status = pump_status};
      }
    }
    {
      const absl::Status close_status = sink.close();
      if (!close_status.ok()) {
        return {.handled = true, .status = close_status};
      }
    }
    const absl::Status dedup_status =
        execute_local_dedup_copies(local_plan_or->direct_dedup_copies, request.gpu_ptr, request.device_id);
    if (!dedup_status.ok()) {
      return {.handled = true, .status = dedup_status};
    }
    direct_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - direct_start).count();
  }

  const auto dim1_start = std::chrono::steady_clock::now();
  {
    const absl::Status dim1_status = execute_local_dim1_jobs(
        local_plan_or->dim1_jobs, backing_source, pinned_pool, pinned_timeout, request.gpu_ptr, request.device_id);
    if (!dim1_status.ok()) {
      if (absl::IsUnimplemented(dim1_status)) {
        return local_batched_fallback(request.replica_key.artifact_id, "unsupported_dim1_executor", dim1_status);
      }
      return {.handled = true, .status = dim1_status};
    }
  }
  const double dim1_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - dim1_start).count();

  const double total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  LOG(INFO) << "local_batched_disk_load timings: artifact_id=" << request.replica_key.artifact_id
            << " direct_segments=" << local_plan_or->direct_segments.size()
            << " direct_dedup_copies=" << local_plan_or->direct_dedup_copies.size()
            << " replicated_jobs=" << local_plan_or->summary.replicated_jobs
            << " dim0_jobs=" << local_plan_or->summary.dim0_jobs << " dim1_jobs=" << local_plan_or->summary.dim1_jobs
            << " requested_source_bytes=" << local_plan_or->summary.requested_source_bytes
            << " unique_source_bytes=" << local_plan_or->summary.unique_source_bytes
            << " dedup_saving_bytes=" << local_plan_or->summary.dedup_saving_bytes
            << " direct_dedup_copy_bytes=" << local_plan_or->summary.direct_dedup_copy_bytes
            << " direct_sec=" << direct_sec << " dim1_sec=" << dim1_sec << " total=" << total_sec;
  return {.handled = true, .status = absl::OkStatus()};
}

absl::Status execute_concat_dim0_job(
    const MappedConcatJobRuntime& job,
    loader::SeekableSource& source,
    NcclClique& clique,
    char* host_buffer,
    size_t host_buffer_bytes,
    void* root_stage_ptr,
    cudaStream_t h2d_stream,
    cudaEvent_t ready_event,
    int root_rank,
    const StrategyConfig& strategy_config,
    RemoteStageWorkspace* workspace = nullptr) {
  RemoteStageWorkspace local_workspace;
  RemoteStageWorkspace& remote_workspace = workspace != nullptr ? *workspace : local_workspace;
  const int root_device_id = job.destinations[static_cast<size_t>(root_rank)].device_id;
  remote_workspace.root_device_id = root_device_id;
  if (remote_workspace.buffers.empty()) {
    remote_workspace.buffers.resize(job.destinations.size());
  }
  if (remote_workspace.buffers.size() < job.destinations.size()) {
    remote_workspace.buffers.resize(job.destinations.size());
  }
  if (remote_workspace.done_events.size() < job.destinations.size()) {
    remote_workspace.done_events.resize(job.destinations.size(), nullptr);
  }
  if (remote_workspace.done_event_devices.size() < job.destinations.size()) {
    remote_workspace.done_event_devices.resize(job.destinations.size(), -1);
  }

  auto ensure_remote_stage = [&](size_t rank, size_t bytes) -> absl::Status {
    if (static_cast<int>(rank) == root_rank || bytes == 0) {
      return absl::OkStatus();
    }
    auto& buffers = remote_workspace.buffers;
    if (buffers[rank] != nullptr && buffers[rank]->size() >= bytes) {
      return absl::OkStatus();
    }
    buffers[rank] = std::make_unique<common::memory::GpuDeviceMemory>();
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(job.destinations[rank].device_id));
    TC_RETURN_IF_ERROR(buffers[rank]->allocate(bytes, job.destinations[rank].device_id));
    remote_workspace.capacity_bytes = std::max(remote_workspace.capacity_bytes, bytes);
    return absl::OkStatus();
  };
  auto ensure_done_event = [&](size_t rank) -> absl::Status {
    const int device_id = job.destinations[rank].device_id;
    if (remote_workspace.done_events[rank] != nullptr && remote_workspace.done_event_devices[rank] == device_id) {
      return absl::OkStatus();
    }
    if (remote_workspace.done_events[rank] != nullptr) {
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(remote_workspace.done_event_devices[rank]));
      TC_RETURN_IF_ERROR(tensorcast::cuda::event_destroy(remote_workspace.done_events[rank]));
      remote_workspace.done_events[rank] = nullptr;
      remote_workspace.done_event_devices[rank] = -1;
    }
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_id));
    TC_RETURN_IF_ERROR(
        tensorcast::cuda::event_create_with_flags(&remote_workspace.done_events[rank], cudaEventDisableTiming));
    remote_workspace.done_event_devices[rank] = device_id;
    return absl::OkStatus();
  };

  const auto total_start = std::chrono::steady_clock::now();
  double read_sec = 0.0;
  double h2d_sec = 0.0;
  double issue_sec = 0.0;
  double sync_sec = 0.0;
  double root_d2d_sec = 0.0;
  uint64_t peer_transfer_bytes = 0;
  size_t chunk_count = 0;
  bool peer_chunk_inflight = false;
  cudaStream_t single_range_h2d_stream = nullptr;
  cudaEvent_t single_range_ready_event = nullptr;

  auto ensure_single_range_h2d = [&]() -> absl::Status {
    if (!use_dedicated_single_range_concat_stream(strategy_config)) {
      return absl::OkStatus();
    }
    if (single_range_h2d_stream != nullptr && single_range_ready_event != nullptr) {
      return absl::OkStatus();
    }
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(root_device_id));
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&single_range_h2d_stream, cudaStreamNonBlocking));
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_create_with_flags(&single_range_ready_event, cudaEventDisableTiming));
    return absl::OkStatus();
  };

  for (const auto& fragment : job.fragments) {
    if (!fragment.src_starts_by_rank.empty()) {
      if (fragment.prefix_count == 1) {
        TC_RETURN_IF_ERROR(ensure_single_range_h2d());
        TensorJob lowered_job;
        lowered_job.name = absl::StrCat(job.name, "::", fragment.source.dtype, "::concat_dim0");
        lowered_job.source = fragment.source;
        lowered_job.distribution = TensorJob::Distribution::kDim0Partitioned;
        lowered_job.slices.resize(job.destinations.size());
        std::vector<ParsedParticipant> lowered_destinations = job.destinations;
        for (size_t rank = 0; rank < job.destinations.size(); ++rank) {
          if (fragment.dst_ptrs[rank] == nullptr) {
            return absl::FailedPreconditionError("concat lowered tensor job requires direct destination pointers");
          }
          lowered_destinations[rank].gpu_ptr = fragment.dst_ptrs[rank];
          const int64_t src_start = fragment.src_starts_by_rank[rank];
          const int64_t src_end = fragment.src_ends_by_rank[rank];
          if (src_end <= src_start) {
            return absl::InvalidArgumentError("concat lowered tensor job has empty rank slice");
          }
          lowered_job.slices[rank] = RankTensorSlice{
              .dst_offset = 0,
              .dst_size_bytes = fragment.dst_block_bytes,
              .kind = RankTensorSlice::Kind::kDim0,
              .start = src_start,
              .length = static_cast<uint64_t>(src_end - src_start),
          };
        }
        TC_RETURN_IF_ERROR(execute_dim0_tensor(
            lowered_job,
            lowered_destinations,
            source,
            clique,
            host_buffer,
            host_buffer_bytes,
            root_stage_ptr,
            single_range_h2d_stream != nullptr ? single_range_h2d_stream : h2d_stream,
            single_range_ready_event != nullptr ? single_range_ready_event : ready_event,
            root_rank));
        if (sync_after_single_range_concat_job(strategy_config)) {
          TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(root_device_id));
          TC_RETURN_IF_ERROR(
              tensorcast::cuda::stream_synchronize(
                  single_range_h2d_stream != nullptr ? single_range_h2d_stream : h2d_stream));
          TC_RETURN_IF_ERROR(clique.synchronize_all());
        }
        continue;
      }

      const auto source_base_offset_or = source_base_offset_bytes(fragment.source);
      if (!source_base_offset_or.ok()) {
        return source_base_offset_or.status();
      }
      const uint64_t source_base_offset = *source_base_offset_or;
      const uint64_t block_bytes = fragment.src_block_bytes;
      if (block_bytes == 0) {
        continue;
      }
      int64_t global_src_start = std::numeric_limits<int64_t>::max();
      int64_t global_src_end = std::numeric_limits<int64_t>::min();
      for (size_t rank = 0; rank < job.destinations.size(); ++rank) {
        global_src_start = std::min(global_src_start, fragment.src_starts_by_rank[rank]);
        global_src_end = std::max(global_src_end, fragment.src_ends_by_rank[rank]);
      }
      if (global_src_end <= global_src_start) {
        return absl::InvalidArgumentError("concat fragment has empty global source range");
      }
      const size_t rows_per_chunk = std::max<size_t>(1, host_buffer_bytes / std::max<uint64_t>(1, block_bytes));
      for (int64_t row = global_src_start; row < global_src_end; row += static_cast<int64_t>(rows_per_chunk)) {
        const uint64_t chunk_rows =
            static_cast<uint64_t>(std::min<int64_t>(global_src_end - row, static_cast<int64_t>(rows_per_chunk)));
        const size_t chunk_bytes = static_cast<size_t>(chunk_rows * block_bytes);
        chunk_count += 1;

        {
          const auto step_start = std::chrono::steady_clock::now();
          TC_RETURN_IF_ERROR(read_exact(
              source, source_base_offset + static_cast<uint64_t>(row) * block_bytes, host_buffer, chunk_bytes));
          read_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
        }

        TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(root_device_id));
        {
          const auto step_start = std::chrono::steady_clock::now();
          if (peer_chunk_inflight) {
            TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(root_device_id));
            for (size_t rank = 0; rank < job.destinations.size(); ++rank) {
              if (remote_workspace.done_events[rank] == nullptr) {
                continue;
              }
              TC_RETURN_IF_ERROR(tensorcast::cuda::stream_wait_event(h2d_stream, remote_workspace.done_events[rank]));
            }
            peer_chunk_inflight = false;
          }
          TC_RETURN_IF_ERROR(
              tensorcast::cuda::memcpy_async(
                  root_stage_ptr, host_buffer, chunk_bytes, cudaMemcpyHostToDevice, h2d_stream));
          TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(ready_event, h2d_stream));
          h2d_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
        }

        bool can_use_peer_copy = true;
        for (size_t rank = 0; rank < job.destinations.size(); ++rank) {
          if (static_cast<int>(rank) == root_rank) {
            continue;
          }
          if (!ensure_peer_copy_capable(root_device_id, job.destinations[rank].device_id)) {
            can_use_peer_copy = false;
            break;
          }
        }

        auto issue_rank_rows =
            [&](size_t rank, const std::uint8_t* src_ptr, size_t bytes, uint64_t dst_block_index) -> absl::Status {
          const size_t rows = bytes / static_cast<size_t>(block_bytes);
          if (rows == 0) {
            return absl::OkStatus();
          }
          TC_RETURN_IF_ERROR(clique.wait_stream_on_event(static_cast<int>(rank), ready_event));
          const bool rank_piecewise = !fragment.dst_block_pieces_by_rank[rank].empty();
          auto* dst_base_ptr = static_cast<std::uint8_t*>(fragment.dst_ptrs[rank]);
          auto* dst_ptr = rank_piecewise ? nullptr : (dst_base_ptr + dst_block_index * fragment.dst_block_stride_bytes);
          if (static_cast<int>(rank) == root_rank) {
            const auto local_copy_start = std::chrono::steady_clock::now();
            if (rank_piecewise) {
              for (size_t local_block = 0; local_block < rows; ++local_block) {
                const auto& pieces =
                    fragment.dst_block_pieces_by_rank[rank][static_cast<size_t>(dst_block_index + local_block)];
                const auto* src_block_ptr = src_ptr + local_block * block_bytes;
                for (const auto& piece : pieces) {
                  TC_RETURN_IF_ERROR(
                      tensorcast::cuda::memcpy_async(
                          piece.dst_ptr,
                          src_block_ptr + piece.block_offset,
                          static_cast<size_t>(piece.length),
                          cudaMemcpyDeviceToDevice,
                          clique.stream(root_rank)));
                }
              }
            } else {
              SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
                  dst_ptr,
                  static_cast<size_t>(fragment.dst_block_stride_bytes),
                  src_ptr,
                  static_cast<size_t>(block_bytes),
                  static_cast<size_t>(block_bytes),
                  rows,
                  cudaMemcpyDeviceToDevice,
                  clique.stream(root_rank)));
            }
            root_d2d_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - local_copy_start).count();
            return absl::OkStatus();
          }
          if (can_use_peer_copy) {
            TC_RETURN_IF_ERROR(ensure_done_event(rank));
            TC_RETURN_IF_ERROR(ensure_remote_stage(rank, bytes));
            const auto issue_start = std::chrono::steady_clock::now();
            peer_transfer_bytes += bytes;
            auto* recv_ptr = remote_workspace.buffers[rank]->get();
            TC_RETURN_IF_ERROR(
                tensorcast::cuda::memcpy_peer_async(
                    recv_ptr,
                    job.destinations[rank].device_id,
                    src_ptr,
                    root_device_id,
                    bytes,
                    clique.stream(static_cast<int>(rank))));
            if (rank_piecewise) {
              for (size_t local_block = 0; local_block < rows; ++local_block) {
                const auto& pieces =
                    fragment.dst_block_pieces_by_rank[rank][static_cast<size_t>(dst_block_index + local_block)];
                const auto* src_block_ptr = static_cast<std::uint8_t*>(recv_ptr) + local_block * block_bytes;
                for (const auto& piece : pieces) {
                  TC_RETURN_IF_ERROR(
                      tensorcast::cuda::memcpy_async(
                          piece.dst_ptr,
                          src_block_ptr + piece.block_offset,
                          static_cast<size_t>(piece.length),
                          cudaMemcpyDeviceToDevice,
                          clique.stream(static_cast<int>(rank))));
                }
              }
            } else {
              SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
                  dst_ptr,
                  static_cast<size_t>(fragment.dst_block_stride_bytes),
                  recv_ptr,
                  static_cast<size_t>(block_bytes),
                  static_cast<size_t>(block_bytes),
                  rows,
                  cudaMemcpyDeviceToDevice,
                  clique.stream(static_cast<int>(rank))));
            }
            issue_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - issue_start).count();
            TC_RETURN_IF_ERROR(
                tensorcast::cuda::event_record(
                    remote_workspace.done_events[rank], clique.stream(static_cast<int>(rank))));
            peer_chunk_inflight = true;
            return absl::OkStatus();
          }
          return absl::UnimplementedError("concat strided fallback without peer-copy is unsupported");
        };

        const auto* src_chunk_ptr = static_cast<const std::uint8_t*>(root_stage_ptr);
        for (size_t rank = 0; rank < job.destinations.size(); ++rank) {
          const int64_t rank_start = fragment.src_starts_by_rank[rank];
          const int64_t rank_end = fragment.src_ends_by_rank[rank];
          const int64_t overlap_start = std::max<int64_t>(row, rank_start);
          const int64_t overlap_end = std::min<int64_t>(row + static_cast<int64_t>(chunk_rows), rank_end);
          if (overlap_end <= overlap_start) {
            continue;
          }
          const uint64_t local_row_offset = static_cast<uint64_t>(overlap_start - row);
          const uint64_t dst_block_index = static_cast<uint64_t>(overlap_start - rank_start);
          const size_t bytes = static_cast<size_t>(static_cast<uint64_t>(overlap_end - overlap_start) * block_bytes);
          TC_RETURN_IF_ERROR(
              issue_rank_rows(rank, src_chunk_ptr + local_row_offset * block_bytes, bytes, dst_block_index));
        }
      }
      continue;
    }
    const auto source_base_offset_or = source_base_offset_bytes(fragment.source);
    if (!source_base_offset_or.ok()) {
      return source_base_offset_or.status();
    }
    const uint64_t source_base_offset = *source_base_offset_or;
    if (fragment.prefix_count == 0 || fragment.src_block_bytes == 0 || fragment.dst_block_bytes == 0) {
      continue;
    }
    const uint64_t dim0_len = static_cast<uint64_t>(fragment.src_end - fragment.src_start);
    if (dim0_len == 0) {
      return absl::InvalidArgumentError("concat_dim0 fragment has empty source range");
    }
    const uint64_t dim0_unit_bytes = fragment.src_block_bytes / dim0_len;
    const uint64_t first_src_offset = source_base_offset + static_cast<uint64_t>(fragment.src_start) * dim0_unit_bytes;
    const size_t blocks_per_chunk =
        std::max<size_t>(1, host_buffer_bytes / std::max<uint64_t>(1, fragment.src_block_bytes));
    for (uint64_t block_start = 0; block_start < fragment.prefix_count; block_start += blocks_per_chunk) {
      const uint64_t chunk_blocks = std::min<uint64_t>(fragment.prefix_count - block_start, blocks_per_chunk);
      const size_t chunk_bytes = static_cast<size_t>(chunk_blocks * fragment.src_block_bytes);
      chunk_count += 1;

      {
        const auto step_start = std::chrono::steady_clock::now();
        TC_RETURN_IF_ERROR(
            read_exact(source, first_src_offset + block_start * fragment.src_block_bytes, host_buffer, chunk_bytes));
        read_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
      }

      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(root_device_id));
      {
        const auto step_start = std::chrono::steady_clock::now();
        if (peer_chunk_inflight) {
          TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(root_device_id));
          for (size_t rank = 0; rank < job.destinations.size(); ++rank) {
            if (remote_workspace.done_events[rank] == nullptr) {
              continue;
            }
            TC_RETURN_IF_ERROR(tensorcast::cuda::stream_wait_event(h2d_stream, remote_workspace.done_events[rank]));
          }
          peer_chunk_inflight = false;
        }
        TC_RETURN_IF_ERROR(
            tensorcast::cuda::memcpy_async(
                root_stage_ptr, host_buffer, chunk_bytes, cudaMemcpyHostToDevice, h2d_stream));
        TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(ready_event, h2d_stream));
        h2d_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
      }

      const bool dst_is_strided = fragment.dst_block_stride_bytes != fragment.dst_block_bytes;
      for (size_t rank = 0; rank < job.destinations.size(); ++rank) {
        const bool rank_piecewise = !fragment.dst_block_pieces_by_rank[rank].empty();
        const bool needs_intermediate = dst_is_strided || rank_piecewise;
        if (static_cast<int>(rank) == root_rank && !needs_intermediate) {
          continue;
        }
        TC_RETURN_IF_ERROR(ensure_remote_stage(rank, chunk_bytes));
      }

      bool can_use_peer_copy = true;
      for (size_t rank = 0; rank < job.destinations.size(); ++rank) {
        if (static_cast<int>(rank) == root_rank) {
          continue;
        }
        if (!ensure_peer_copy_capable(root_device_id, job.destinations[rank].device_id)) {
          can_use_peer_copy = false;
          break;
        }
      }

      if (can_use_peer_copy) {
        const auto* send_ptr = static_cast<const std::uint8_t*>(root_stage_ptr);
        for (size_t rank = 0; rank < job.destinations.size(); ++rank) {
          if (static_cast<int>(rank) != root_rank) {
            TC_RETURN_IF_ERROR(ensure_done_event(rank));
          }
          TC_RETURN_IF_ERROR(clique.wait_stream_on_event(static_cast<int>(rank), ready_event));
        }
        for (size_t rank = 0; rank < job.destinations.size(); ++rank) {
          auto* dst_base_ptr = static_cast<std::uint8_t*>(fragment.dst_ptrs[rank]);
          const uint64_t dst_chunk_offset = block_start * fragment.dst_block_stride_bytes;
          const bool rank_piecewise = !fragment.dst_block_pieces_by_rank[rank].empty();
          if (static_cast<int>(rank) == root_rank) {
            if (rank_piecewise) {
              TC_RETURN_IF_ERROR(ensure_done_event(rank));
              const auto local_copy_start = std::chrono::steady_clock::now();
              for (uint64_t local_block = 0; local_block < chunk_blocks; ++local_block) {
                const auto& pieces =
                    fragment.dst_block_pieces_by_rank[rank][static_cast<size_t>(block_start + local_block)];
                const auto* src_block_ptr = send_ptr + local_block * fragment.dst_block_bytes;
                for (const auto& piece : pieces) {
                  TC_RETURN_IF_ERROR(
                      tensorcast::cuda::memcpy_async(
                          piece.dst_ptr,
                          src_block_ptr + piece.block_offset,
                          static_cast<size_t>(piece.length),
                          cudaMemcpyDeviceToDevice,
                          clique.stream(root_rank)));
                }
              }
              root_d2d_sec +=
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - local_copy_start).count();
              TC_RETURN_IF_ERROR(
                  tensorcast::cuda::event_record(remote_workspace.done_events[rank], clique.stream(root_rank)));
            } else if (dst_is_strided) {
              TC_RETURN_IF_ERROR(ensure_done_event(rank));
              const auto local_copy_start = std::chrono::steady_clock::now();
              SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
                  dst_base_ptr + dst_chunk_offset,
                  static_cast<size_t>(fragment.dst_block_stride_bytes),
                  send_ptr,
                  static_cast<size_t>(fragment.dst_block_bytes),
                  static_cast<size_t>(fragment.dst_block_bytes),
                  static_cast<size_t>(chunk_blocks),
                  cudaMemcpyDeviceToDevice,
                  clique.stream(root_rank)));
              root_d2d_sec +=
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - local_copy_start).count();
              TC_RETURN_IF_ERROR(
                  tensorcast::cuda::event_record(remote_workspace.done_events[rank], clique.stream(root_rank)));
            } else {
              const auto local_copy_start = std::chrono::steady_clock::now();
              TC_RETURN_IF_ERROR(
                  tensorcast::cuda::memcpy_async(
                      dst_base_ptr + dst_chunk_offset,
                      send_ptr,
                      chunk_bytes,
                      cudaMemcpyDeviceToDevice,
                      clique.stream(root_rank)));
              root_d2d_sec +=
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - local_copy_start).count();
            }
            continue;
          }

          const auto issue_start = std::chrono::steady_clock::now();
          peer_transfer_bytes += chunk_bytes;
          if (rank_piecewise) {
            auto* recv_ptr = remote_workspace.buffers[rank]->get();
            TC_RETURN_IF_ERROR(
                tensorcast::cuda::memcpy_peer_async(
                    recv_ptr,
                    job.destinations[rank].device_id,
                    send_ptr,
                    root_device_id,
                    chunk_bytes,
                    clique.stream(static_cast<int>(rank))));
            for (uint64_t local_block = 0; local_block < chunk_blocks; ++local_block) {
              const auto& pieces =
                  fragment.dst_block_pieces_by_rank[rank][static_cast<size_t>(block_start + local_block)];
              const auto* src_block_ptr =
                  static_cast<const std::uint8_t*>(recv_ptr) + local_block * fragment.dst_block_bytes;
              for (const auto& piece : pieces) {
                TC_RETURN_IF_ERROR(
                    tensorcast::cuda::memcpy_async(
                        piece.dst_ptr,
                        src_block_ptr + piece.block_offset,
                        static_cast<size_t>(piece.length),
                        cudaMemcpyDeviceToDevice,
                        clique.stream(static_cast<int>(rank))));
              }
            }
          } else if (dst_is_strided) {
            auto* recv_ptr = remote_workspace.buffers[rank]->get();
            TC_RETURN_IF_ERROR(
                tensorcast::cuda::memcpy_peer_async(
                    recv_ptr,
                    job.destinations[rank].device_id,
                    send_ptr,
                    root_device_id,
                    chunk_bytes,
                    clique.stream(static_cast<int>(rank))));
            SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
                dst_base_ptr + dst_chunk_offset,
                static_cast<size_t>(fragment.dst_block_stride_bytes),
                recv_ptr,
                static_cast<size_t>(fragment.dst_block_bytes),
                static_cast<size_t>(fragment.dst_block_bytes),
                static_cast<size_t>(chunk_blocks),
                cudaMemcpyDeviceToDevice,
                clique.stream(static_cast<int>(rank))));
          } else {
            TC_RETURN_IF_ERROR(
                tensorcast::cuda::memcpy_peer_async(
                    dst_base_ptr + dst_chunk_offset,
                    job.destinations[rank].device_id,
                    send_ptr,
                    root_device_id,
                    chunk_bytes,
                    clique.stream(static_cast<int>(rank))));
          }
          issue_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - issue_start).count();
          TC_RETURN_IF_ERROR(
              tensorcast::cuda::event_record(
                  remote_workspace.done_events[rank], clique.stream(static_cast<int>(rank))));
        }
        peer_chunk_inflight = true;
      } else {
        TC_RETURN_IF_ERROR(clique.group_start());
        {
          const auto* send_ptr = static_cast<const std::uint8_t*>(root_stage_ptr);
          TC_RETURN_IF_ERROR(clique.wait_stream_on_event(root_rank, ready_event));
          for (size_t rank = 0; rank < job.destinations.size(); ++rank) {
            auto* dst_base_ptr = static_cast<std::uint8_t*>(fragment.dst_ptrs[rank]);
            const uint64_t dst_chunk_offset = block_start * fragment.dst_block_stride_bytes;
            const bool rank_piecewise = !fragment.dst_block_pieces_by_rank[rank].empty();
            if (static_cast<int>(rank) == root_rank) {
              if (rank_piecewise) {
                const auto local_copy_start = std::chrono::steady_clock::now();
                for (uint64_t local_block = 0; local_block < chunk_blocks; ++local_block) {
                  const auto& pieces =
                      fragment.dst_block_pieces_by_rank[rank][static_cast<size_t>(block_start + local_block)];
                  const auto* src_block_ptr = send_ptr + local_block * fragment.dst_block_bytes;
                  for (const auto& piece : pieces) {
                    TC_RETURN_IF_ERROR(
                        tensorcast::cuda::memcpy_async(
                            piece.dst_ptr,
                            src_block_ptr + piece.block_offset,
                            static_cast<size_t>(piece.length),
                            cudaMemcpyDeviceToDevice,
                            clique.stream(root_rank)));
                  }
                }
                root_d2d_sec +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - local_copy_start).count();
              } else if (dst_is_strided) {
                const auto local_copy_start = std::chrono::steady_clock::now();
                SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
                    dst_base_ptr + dst_chunk_offset,
                    static_cast<size_t>(fragment.dst_block_stride_bytes),
                    send_ptr,
                    static_cast<size_t>(fragment.dst_block_bytes),
                    static_cast<size_t>(fragment.dst_block_bytes),
                    static_cast<size_t>(chunk_blocks),
                    cudaMemcpyDeviceToDevice,
                    clique.stream(root_rank)));
                root_d2d_sec +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - local_copy_start).count();
              } else {
                const auto local_copy_start = std::chrono::steady_clock::now();
                TC_RETURN_IF_ERROR(
                    tensorcast::cuda::memcpy_async(
                        dst_base_ptr + dst_chunk_offset,
                        send_ptr,
                        chunk_bytes,
                        cudaMemcpyDeviceToDevice,
                        clique.stream(root_rank)));
                root_d2d_sec +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - local_copy_start).count();
              }
              continue;
            }
            auto* recv_ptr = (dst_is_strided || rank_piecewise) ? remote_workspace.buffers[rank]->get()
                                                                : (dst_base_ptr + dst_chunk_offset);
            const auto issue_start = std::chrono::steady_clock::now();
            peer_transfer_bytes += chunk_bytes;
            TC_RETURN_IF_ERROR(clique.send_u8(root_rank, send_ptr, chunk_bytes, static_cast<int>(rank)));
            TC_RETURN_IF_ERROR(clique.recv_u8(static_cast<int>(rank), recv_ptr, chunk_bytes, root_rank));
            issue_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - issue_start).count();
          }
        }
        TC_RETURN_IF_ERROR(clique.group_end());

        bool any_piecewise_remote = false;
        for (size_t rank = 0; rank < job.destinations.size(); ++rank) {
          if (static_cast<int>(rank) == root_rank) {
            continue;
          }
          any_piecewise_remote = any_piecewise_remote || !fragment.dst_block_pieces_by_rank[rank].empty();
        }
        if (dst_is_strided || any_piecewise_remote) {
          for (size_t rank = 0; rank < job.destinations.size(); ++rank) {
            if (static_cast<int>(rank) == root_rank) {
              continue;
            }
            auto* dst_base_ptr = static_cast<std::uint8_t*>(fragment.dst_ptrs[rank]);
            const uint64_t dst_chunk_offset = block_start * fragment.dst_block_stride_bytes;
            const bool rank_piecewise = !fragment.dst_block_pieces_by_rank[rank].empty();
            if (rank_piecewise) {
              for (uint64_t local_block = 0; local_block < chunk_blocks; ++local_block) {
                const auto& pieces =
                    fragment.dst_block_pieces_by_rank[rank][static_cast<size_t>(block_start + local_block)];
                const auto* src_block_ptr = static_cast<const std::uint8_t*>(remote_workspace.buffers[rank]->get()) +
                    local_block * fragment.dst_block_bytes;
                for (const auto& piece : pieces) {
                  TC_RETURN_IF_ERROR(
                      tensorcast::cuda::memcpy_async(
                          piece.dst_ptr,
                          src_block_ptr + piece.block_offset,
                          static_cast<size_t>(piece.length),
                          cudaMemcpyDeviceToDevice,
                          clique.stream(static_cast<int>(rank))));
                }
              }
            } else {
              SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
                  dst_base_ptr + dst_chunk_offset,
                  static_cast<size_t>(fragment.dst_block_stride_bytes),
                  remote_workspace.buffers[rank]->get(),
                  static_cast<size_t>(fragment.dst_block_bytes),
                  static_cast<size_t>(fragment.dst_block_bytes),
                  static_cast<size_t>(chunk_blocks),
                  cudaMemcpyDeviceToDevice,
                  clique.stream(static_cast<int>(rank))));
            }
          }
        }
        const auto sync_start = std::chrono::steady_clock::now();
        TC_RETURN_IF_ERROR(clique.synchronize_all());
        sync_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - sync_start).count();
      }
    }
  }
  if (peer_chunk_inflight) {
    const auto sync_start = std::chrono::steady_clock::now();
    TC_RETURN_IF_ERROR(clique.synchronize_all());
    sync_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - sync_start).count();
  }

  const auto total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  if (single_range_ready_event != nullptr) {
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(root_device_id));
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_destroy(single_range_ready_event));
  }
  if (single_range_h2d_stream != nullptr) {
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(root_device_id));
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_destroy(single_range_h2d_stream));
  }
  LOG(INFO) << "collective_concat_dim0_job name=" << job.name << " fragments=" << job.fragments.size()
            << " prefix_count=" << job.prefix_count << " chunk_count=" << chunk_count << " read=" << read_sec
            << "s h2d=" << h2d_sec << "s issue=" << issue_sec << "s sync=" << sync_sec << "s root_d2d=" << root_d2d_sec
            << "s peer_transfer_bytes=" << peer_transfer_bytes << " total=" << total_sec << "s";
  return absl::OkStatus();
}

absl::StatusOr<runtime::ingestion::strategy::CollectiveExecutionMetrics> execute_group_collective(
    const std::vector<ParsedParticipant>& participants,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const StrategyConfig& strategy_config) {
  const auto total_start = std::chrono::steady_clock::now();
  if (participants.empty()) {
    return absl::InvalidArgumentError("collective disk load participants are empty");
  }
  runtime::ingestion::strategy::CollectiveExecutionMetrics collective_metrics;
  std::vector<int> device_ids;
  device_ids.reserve(participants.size());
  for (const auto& participant : participants) {
    device_ids.push_back(participant.device_id);
  }
  const auto clique_start = std::chrono::steady_clock::now();
  bool clique_cache_hit = false;
  auto clique_or = get_or_create_cached_clique(device_ids, &clique_cache_hit);
  if (!clique_or.ok()) {
    return clique_or.status();
  }
  auto clique = *clique_or;
  const auto clique_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - clique_start).count();
  absl::MutexLock clique_use_lock(&clique->use_mutex());
  const size_t chunk_bytes = pinned_pool->slice_bytes();

  const int root_rank = 0;
  std::unique_ptr<common::memory::GpuDeviceMemory> root_stage;
  root_stage = std::make_unique<common::memory::GpuDeviceMemory>();
  const auto root_alloc_start = std::chrono::steady_clock::now();
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
  TC_RETURN_IF_ERROR(root_stage->allocate(chunk_bytes, participants[static_cast<size_t>(root_rank)].device_id));
  const auto root_alloc_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - root_alloc_start).count();

  cudaStream_t h2d_stream = nullptr;
  cudaEvent_t ready_event = nullptr;
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&h2d_stream, cudaStreamNonBlocking));
  TC_RETURN_IF_ERROR(tensorcast::cuda::event_create_with_flags(&ready_event, cudaEventDisableTiming));

  loader::MultiSafetensorsSource source(participants.front().disk_context->safetensors_segments());
  const auto jobs_start = std::chrono::steady_clock::now();
  auto jobs_or = build_tensor_jobs(participants);
  if (!jobs_or.ok()) {
    return jobs_or.status();
  }
  const auto jobs_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - jobs_start).count();
  const bool owner_file_enabled = enable_collective_owner_file_strategy(strategy_config);
  std::optional<OwnerCollectivePlan> owner_collective_plan;
  double owner_plan_sec = 0.0;
  size_t owner_batch_count = 0;
  size_t owner_replicated_jobs = 0;
  size_t owner_dim0_jobs = 0;
  size_t owner_dim1_jobs = 0;
  double owner_exec_sec = 0.0;
  uint64_t owner_peer_transfer_bytes = 0;
  if (owner_file_enabled) {
    const auto owner_plan_start = std::chrono::steady_clock::now();
    auto owner_plan_or = build_owner_file_collective_plan(
        *jobs_or,
        participants.front().disk_context->safetensors_segments(),
        clique->world_size(),
        strategy_config,
        chunk_bytes);
    owner_plan_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - owner_plan_start).count();
    if (owner_plan_or.ok()) {
      owner_collective_plan = std::move(*owner_plan_or);
      owner_batch_count = owner_collective_plan->batches.size();
      collective_metrics = owner_collective_metrics(*owner_collective_plan);
      LOG(INFO) << "collective_owner_file_batched plan"
                << " artifact_id=" << participants.front().replica_key.artifact_id
                << " batches=" << owner_collective_plan->batches.size()
                << " unique_source_bytes=" << owner_collective_plan->unique_source_bytes
                << " peer_transfer_bytes=" << owner_collective_plan->peer_transfer_bytes
                << " owner_skew_ratio=" << owner_collective_plan->owner_skew_ratio
                << " segment_split=" << (owner_collective_plan->used_segment_split ? 1 : 0);
    } else {
      return owner_plan_or.status();
    }
  }
  if (participants.front().disk_context->safetensors_segments().empty()) {
    return absl::FailedPreconditionError("collective disk load requires non-empty safetensors segments");
  }

  PinnedBorrow host_pool;
  host_pool.pool = pinned_pool;
  double pinned_alloc_sec = 0.0;
  char* host_buffer = nullptr;
  {
    const auto pinned_alloc_start = std::chrono::steady_clock::now();
    const std::string request_context =
        absl::StrCat("collective_disk_load artifact_id=", participants.front().replica_key.artifact_id);
    if (pinned_pool->allocate(pinned_pool->slice_bytes() * 2, host_pool.buffers, pinned_timeout, request_context) !=
            0 ||
        host_pool.buffers.empty()) {
      host_pool.buffers.clear();
      if (pinned_pool->allocate(pinned_pool->slice_bytes(), host_pool.buffers, pinned_timeout, request_context) != 0 ||
          host_pool.buffers.empty()) {
        return absl::ResourceExhaustedError("failed to allocate pinned buffer for collective disk load");
      }
    }
    pinned_alloc_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - pinned_alloc_start).count();
    host_buffer = host_pool.buffers.front();
  }
  size_t replicated_jobs = 0;
  size_t dim0_jobs = 0;
  size_t dim1_jobs = 0;
  uint64_t replicated_source_bytes = 0;
  uint64_t dim0_source_bytes = 0;
  uint64_t dim1_source_bytes = 0;
  double replicated_sec = 0.0;
  double dim0_sec = 0.0;
  double dim1_sec = 0.0;
  std::vector<bool> handled_by_owner_file(jobs_or->size(), false);
  if (owner_collective_plan.has_value()) {
    const uint64_t owner_direct_chunk_bytes =
        std::min<uint64_t>(chunk_bytes, effective_owner_batch_bytes(strategy_config, chunk_bytes));
    const uint64_t owner_dim1_chunk_bytes = std::min<uint64_t>(
        owner_direct_chunk_bytes, effective_owner_dim1_staging_bytes(strategy_config, owner_direct_chunk_bytes));
    const size_t owner_stage_bytes = static_cast<size_t>(std::max(owner_direct_chunk_bytes, owner_dim1_chunk_bytes));
    std::vector<OwnerRankWorkspace> owner_workspaces(participants.size());
    for (const auto& batch : owner_collective_plan->batches) {
      auto& workspace = owner_workspaces[static_cast<size_t>(batch.owner_rank)];
      TC_RETURN_IF_ERROR(ensure_owner_rank_workspace(
          &workspace, participants[static_cast<size_t>(batch.owner_rank)].device_id, owner_stage_bytes));
      for (size_t planned_idx : batch.planned_job_indices) {
        const auto& planned_job = owner_collective_plan->planned_jobs[planned_idx];
        const auto& job = (*jobs_or)[planned_job.job_index];
        const auto owner_start = std::chrono::steady_clock::now();
        switch (job.distribution) {
          case TensorJob::Distribution::kReplicated:
            TC_RETURN_IF_ERROR(execute_replicated_tensor(
                job,
                participants,
                source,
                *clique,
                host_buffer,
                static_cast<size_t>(owner_direct_chunk_bytes),
                workspace.stage_buffer->get(),
                workspace.h2d_stream,
                workspace.ready_event,
                batch.owner_rank));
            owner_replicated_jobs += 1;
            break;
          case TensorJob::Distribution::kDim0Partitioned:
            TC_RETURN_IF_ERROR(execute_dim0_tensor(
                job,
                participants,
                source,
                *clique,
                host_buffer,
                static_cast<size_t>(owner_direct_chunk_bytes),
                workspace.stage_buffer->get(),
                workspace.h2d_stream,
                workspace.ready_event,
                batch.owner_rank));
            owner_dim0_jobs += 1;
            break;
          case TensorJob::Distribution::kDim1Partitioned:
            TC_RETURN_IF_ERROR(execute_dim1_tensor(
                job,
                participants,
                source,
                *clique,
                host_buffer,
                static_cast<size_t>(owner_dim1_chunk_bytes),
                workspace.stage_buffer->get(),
                workspace.h2d_stream,
                workspace.ready_event,
                batch.owner_rank,
                &workspace.dim1_pack_workspace));
            owner_dim1_jobs += 1;
            break;
        }
        owner_exec_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - owner_start).count();
        owner_peer_transfer_bytes += planned_job.peer_transfer_bytes;
        handled_by_owner_file[planned_job.job_index] = true;
      }
    }
  }
  for (size_t job_idx = 0; job_idx < jobs_or->size(); ++job_idx) {
    if (handled_by_owner_file[job_idx]) {
      continue;
    }
    const auto& job = (*jobs_or)[job_idx];
    const auto job_start = std::chrono::steady_clock::now();
    switch (job.distribution) {
      case TensorJob::Distribution::kReplicated:
        TC_RETURN_IF_ERROR(execute_replicated_tensor(
            job,
            participants,
            source,
            *clique,
            host_buffer,
            chunk_bytes,
            root_stage->get(),
            h2d_stream,
            ready_event,
            root_rank));
        replicated_jobs += 1;
        replicated_source_bytes += job.source.size_bytes;
        replicated_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - job_start).count();
        break;
      case TensorJob::Distribution::kDim0Partitioned:
        TC_RETURN_IF_ERROR(execute_dim0_tensor(
            job,
            participants,
            source,
            *clique,
            host_buffer,
            chunk_bytes,
            root_stage->get(),
            h2d_stream,
            ready_event,
            root_rank));
        dim0_jobs += 1;
        dim0_source_bytes += job.source.size_bytes;
        dim0_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - job_start).count();
        break;
      case TensorJob::Distribution::kDim1Partitioned:
        TC_RETURN_IF_ERROR(execute_dim1_tensor(
            job,
            participants,
            source,
            *clique,
            host_buffer,
            chunk_bytes,
            root_stage->get(),
            h2d_stream,
            ready_event,
            root_rank));
        dim1_jobs += 1;
        dim1_source_bytes += job.source.size_bytes;
        dim1_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - job_start).count();
        break;
    }
  }

  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
  TC_RETURN_IF_ERROR(tensorcast::cuda::event_destroy(ready_event));
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_destroy(h2d_stream));
  const auto total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  const double replicated_gib_s = replicated_sec > 0.0
      ? (static_cast<double>(replicated_source_bytes) / static_cast<double>(1ULL << 30)) / replicated_sec
      : 0.0;
  const double dim0_gib_s =
      dim0_sec > 0.0 ? (static_cast<double>(dim0_source_bytes) / static_cast<double>(1ULL << 30)) / dim0_sec : 0.0;
  const double dim1_gib_s =
      dim1_sec > 0.0 ? (static_cast<double>(dim1_source_bytes) / static_cast<double>(1ULL << 30)) / dim1_sec : 0.0;
  LOG(INFO) << "collective_disk_load timings: clique_init=" << clique_sec << "s"
            << " clique_cache_hit=" << (clique_cache_hit ? 1 : 0) << " pinned_alloc=" << pinned_alloc_sec << "s"
            << " root_alloc=" << root_alloc_sec << "s"
            << " owner_file_enabled=" << (owner_file_enabled ? 1 : 0) << " owner_plan=" << owner_plan_sec << "s"
            << " owner_batches=" << owner_batch_count << " owner_replicated_jobs=" << owner_replicated_jobs
            << " owner_dim0_jobs=" << owner_dim0_jobs << " owner_dim1_jobs=" << owner_dim1_jobs
            << " owner_exec=" << owner_exec_sec << "s"
            << " owner_peer_transfer_bytes=" << owner_peer_transfer_bytes << " owner_unique_source_bytes="
            << (owner_collective_plan.has_value() ? owner_collective_plan->unique_source_bytes : 0)
            << " owner_skew_ratio="
            << (owner_collective_plan.has_value() ? owner_collective_plan->owner_skew_ratio : 1.0)
            << " owner_segment_split="
            << (owner_collective_plan.has_value() && owner_collective_plan->used_segment_split ? 1 : 0)
            << " build_jobs=" << jobs_sec << "s"
            << " replicated_jobs=" << replicated_jobs << " replicated_source_bytes=" << replicated_source_bytes
            << " replicated_exec=" << replicated_sec << "s"
            << " replicated_gib_s=" << replicated_gib_s << " dim0_jobs=" << dim0_jobs
            << " dim0_source_bytes=" << dim0_source_bytes << " dim0_exec=" << dim0_sec << "s"
            << " dim0_gib_s=" << dim0_gib_s << " dim1_jobs=" << dim1_jobs << " dim1_source_bytes=" << dim1_source_bytes
            << " dim1_exec=" << dim1_sec << "s"
            << " dim1_gib_s=" << dim1_gib_s << " chunk_bytes=" << chunk_bytes << " total=" << total_sec << "s";
  if (!owner_collective_plan.has_value()) {
    collective_metrics.unique_source_bytes = replicated_source_bytes + dim0_source_bytes + dim1_source_bytes;
    collective_metrics.peer_transfer_bytes = owner_peer_transfer_bytes;
    collective_metrics.peak_temporary_bytes = chunk_bytes;
    collective_metrics.batch_count = replicated_jobs + dim0_jobs + dim1_jobs;
    collective_metrics.dedup_saving_bytes = 0;
  }
  return collective_metrics;
}

absl::StatusOr<runtime::ingestion::strategy::CollectiveExecutionMetrics> execute_group_collective_mapped(
    const std::vector<ParsedMappedParticipant>& participants,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options) {
  const auto total_start = std::chrono::steady_clock::now();
  if (participants.empty()) {
    return absl::InvalidArgumentError("mapped collective load participants are empty");
  }
  runtime::ingestion::strategy::CollectiveExecutionMetrics collective_metrics;
  if (pinned_pool == nullptr) {
    return absl::InvalidArgumentError("mapped collective load requires a pinned pool");
  }
  if (participants.front().disk_context == nullptr ||
      participants.front().disk_context->safetensors_segments().empty()) {
    return absl::InvalidArgumentError("mapped collective load requires safetensors segments");
  }

  std::vector<int> device_ids;
  device_ids.reserve(participants.size());
  for (const auto& participant : participants) {
    device_ids.push_back(participant.device_id);
  }
  const auto clique_start = std::chrono::steady_clock::now();
  bool clique_cache_hit = false;
  auto clique_or = get_or_create_cached_clique(device_ids, &clique_cache_hit);
  if (!clique_or.ok()) {
    return clique_or.status();
  }
  auto clique = *clique_or;
  const auto clique_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - clique_start).count();
  absl::MutexLock clique_use_lock(&clique->use_mutex());

  const size_t chunk_bytes = pinned_pool->slice_bytes();
  const int root_rank = 0;
  const int root_device_id = participants[static_cast<size_t>(root_rank)].device_id;

  auto root_stage = std::make_unique<common::memory::GpuDeviceMemory>();
  const auto root_alloc_start = std::chrono::steady_clock::now();
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(root_device_id));
  TC_RETURN_IF_ERROR(root_stage->allocate(chunk_bytes, root_device_id));
  const auto root_alloc_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - root_alloc_start).count();

  cudaStream_t h2d_stream = nullptr;
  cudaEvent_t ready_event = nullptr;
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&h2d_stream, cudaStreamNonBlocking));
  TC_RETURN_IF_ERROR(tensorcast::cuda::event_create_with_flags(&ready_event, cudaEventDisableTiming));

  PinnedBorrow host_pool;
  host_pool.pool = pinned_pool;
  const auto pinned_alloc_start = std::chrono::steady_clock::now();
  const std::string request_context =
      absl::StrCat("collective_mapped_target artifact_id=", participants.front().artifact_id);
  if (pinned_pool->allocate(chunk_bytes, host_pool.buffers, pinned_timeout, request_context) != 0 ||
      host_pool.buffers.empty()) {
    return absl::ResourceExhaustedError("failed to allocate pinned buffer for mapped collective load");
  }
  char* host_buffer = host_pool.buffers.front();
  const auto pinned_alloc_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - pinned_alloc_start).count();
  PinnedBorrow prefetch_pool;
  prefetch_pool.pool = pinned_pool;
  if (pinned_pool->allocate(chunk_bytes, prefetch_pool.buffers, pinned_timeout, request_context) != 0 ||
      prefetch_pool.buffers.empty()) {
    return absl::ResourceExhaustedError("failed to allocate second pinned buffer for mapped collective load");
  }
  char* prefetch_buffer = prefetch_pool.buffers.front();

  std::unique_ptr<loader::SeekableSource> source_owner =
      std::make_unique<PreadMultiSafetensorsSource>(std::vector<loader::SharedSafetensorsSegment>(
          participants.front().disk_context->safetensors_segments().begin(),
          participants.front().disk_context->safetensors_segments().end()));
  loader::SeekableSource& source = *source_owner;

  MappedTensorJobBuildResult tensor_job_build;
  MappedConcatJobBuildResult concat_job_build;
  Dim1PackWorkspace dim1_pack_workspace;
  RemoteStageWorkspace concat_remote_workspace;
  double tensor_job_build_sec = 0.0;
  double tensor_job_exec_sec = 0.0;
  size_t tensor_job_count = 0;
  double concat_job_build_sec = 0.0;
  double concat_job_exec_sec = 0.0;
  size_t concat_job_count = 0;
  if (enable_mapped_tensor_job_fast_path(options.strategy_config)) {
    const bool run_dim0_tensor_jobs = enable_mapped_dim0_tensor_jobs(options.strategy_config);
    const bool run_dim1_tensor_jobs = enable_mapped_dim1_tensor_jobs(options.strategy_config);
    const bool run_concat_jobs = enable_mapped_concat_jobs(options.strategy_config);
    const auto tensor_job_build_start = std::chrono::steady_clock::now();
    auto tensor_job_build_or = build_mapped_tensor_jobs(participants);
    if (!tensor_job_build_or.ok()) {
      return tensor_job_build_or.status();
    }
    tensor_job_build = std::move(*tensor_job_build_or);
    tensor_job_build_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tensor_job_build_start).count();
    tensor_job_count = tensor_job_build.jobs.size();
    for (const auto& runtime_job : tensor_job_build.jobs) {
      const auto job_start = std::chrono::steady_clock::now();
      switch (runtime_job.job.distribution) {
        case TensorJob::Distribution::kReplicated:
          TC_RETURN_IF_ERROR(execute_replicated_tensor(
              runtime_job.job,
              runtime_job.destinations,
              source,
              *clique,
              host_buffer,
              chunk_bytes,
              root_stage->get(),
              h2d_stream,
              ready_event,
              root_rank));
          break;
        case TensorJob::Distribution::kDim0Partitioned:
          if (!run_dim0_tensor_jobs) {
            continue;
          }
          TC_RETURN_IF_ERROR(execute_dim0_tensor(
              runtime_job.job,
              runtime_job.destinations,
              source,
              *clique,
              host_buffer,
              chunk_bytes,
              root_stage->get(),
              h2d_stream,
              ready_event,
              root_rank));
          break;
        case TensorJob::Distribution::kDim1Partitioned:
          if (!run_dim1_tensor_jobs) {
            continue;
          }
          TC_RETURN_IF_ERROR(execute_dim1_tensor(
              runtime_job.job,
              runtime_job.destinations,
              source,
              *clique,
              host_buffer,
              chunk_bytes,
              root_stage->get(),
              h2d_stream,
              ready_event,
              root_rank,
              &dim1_pack_workspace));
          break;
      }
      tensor_job_exec_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - job_start).count();
    }

    const auto concat_job_build_start = std::chrono::steady_clock::now();
    auto concat_job_build_or = build_mapped_concat_jobs(participants, options.strategy_config);
    if (!concat_job_build_or.ok()) {
      return concat_job_build_or.status();
    }
    concat_job_build = std::move(*concat_job_build_or);
    concat_job_build_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - concat_job_build_start).count();
    concat_job_count = concat_job_build.jobs.size();
    if (run_concat_jobs && enable_mapped_concat_execution(options.strategy_config)) {
      for (const auto& job : concat_job_build.jobs) {
        const auto job_start = std::chrono::steady_clock::now();
        TC_RETURN_IF_ERROR(execute_concat_dim0_job(
            job,
            source,
            *clique,
            host_buffer,
            chunk_bytes,
            root_stage->get(),
            h2d_stream,
            ready_event,
            root_rank,
            options.strategy_config,
            &concat_remote_workspace));
        concat_job_exec_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - job_start).count();
      }
    }
  }

  std::vector<std::vector<ByteRange>> handled_dst_ranges_by_rank(participants.size());
  for (size_t rank = 0; rank < participants.size(); ++rank) {
    handled_dst_ranges_by_rank[rank] = tensor_job_build.handled_dst_ranges_by_rank[rank];
    handled_dst_ranges_by_rank[rank].insert(
        handled_dst_ranges_by_rank[rank].end(),
        concat_job_build.handled_dst_ranges_by_rank[rank].begin(),
        concat_job_build.handled_dst_ranges_by_rank[rank].end());
    merge_byte_ranges(&handled_dst_ranges_by_rank[rank]);
  }

  const auto segments_start = std::chrono::steady_clock::now();
  auto segment_refs_or = build_mapped_segment_refs(participants, handled_dst_ranges_by_rank);
  if (!segment_refs_or.ok()) {
    return segment_refs_or.status();
  }
  auto windows_or = build_mapped_source_windows(*segment_refs_or, options);
  if (!windows_or.ok()) {
    return windows_or.status();
  }
  const auto segments_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - segments_start).count();
  LOG(INFO) << "collective_mapped_target prepared"
            << " artifact_id=" << participants.front().artifact_id << " segments=" << segment_refs_or->size()
            << " windows=" << windows_or->size() << " chunk_bytes=" << chunk_bytes
            << " tensor_jobs=" << tensor_job_count
            << " tensor_job_source_bytes=" << tensor_job_build.handled_source_bytes
            << " tensor_job_root_dst_bytes=" << tensor_job_build.handled_root_dst_bytes
            << " tensor_job_build_sec=" << tensor_job_build_sec << " tensor_job_exec_sec=" << tensor_job_exec_sec
            << " concat_jobs=" << concat_job_count
            << " concat_job_source_bytes=" << concat_job_build.handled_source_bytes
            << " concat_job_root_dst_bytes=" << concat_job_build.handled_root_dst_bytes
            << " concat_job_build_sec=" << concat_job_build_sec << " concat_job_exec_sec=" << concat_job_exec_sec
            << " build_segments_sec=" << segments_sec;
  std::vector<ChunkReadPlan> chunk_plans;
  chunk_plans.reserve(2048);
  for (const auto& window : *windows_or) {
    uint64_t chunk_start = window.start;
    while (chunk_start < window.end) {
      const uint64_t chunk_end = std::min<uint64_t>(window.end, chunk_start + chunk_bytes);
      chunk_plans.push_back(
          ChunkReadPlan{
              .start = chunk_start,
              .end = chunk_end,
              .length = static_cast<size_t>(chunk_end - chunk_start),
              .window = &window,
          });
      chunk_start = chunk_end;
    }
  }
  if (chunk_plans.empty()) {
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(root_device_id));
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_destroy(ready_event));
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_destroy(h2d_stream));
    const auto total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
    LOG(INFO) << "collective_mapped_target timings: clique_init=" << clique_sec << "s"
              << " clique_cache_hit=" << (clique_cache_hit ? 1 : 0) << " pinned_alloc=" << pinned_alloc_sec << "s"
              << " root_alloc=" << root_alloc_sec << "s"
              << " tensor_jobs=" << tensor_job_count
              << " tensor_job_source_bytes=" << tensor_job_build.handled_source_bytes
              << " tensor_job_root_dst_bytes=" << tensor_job_build.handled_root_dst_bytes
              << " tensor_job_build=" << tensor_job_build_sec << "s"
              << " tensor_job_exec=" << tensor_job_exec_sec << "s"
              << " concat_jobs=" << concat_job_count
              << " concat_job_source_bytes=" << concat_job_build.handled_source_bytes
              << " concat_job_root_dst_bytes=" << concat_job_build.handled_root_dst_bytes
              << " concat_job_build=" << concat_job_build_sec << "s"
              << " concat_job_exec=" << concat_job_exec_sec << "s"
              << " build_segments=" << segments_sec << "s"
              << " windows=0 segments=0"
              << " total=" << total_sec << "s";
    collective_metrics.unique_source_bytes =
        tensor_job_build.handled_source_bytes + concat_job_build.handled_source_bytes;
    collective_metrics.peer_transfer_bytes = 0;
    collective_metrics.peak_temporary_bytes = chunk_bytes;
    collective_metrics.batch_count = tensor_job_count + concat_job_count;
    collective_metrics.dedup_saving_bytes = 0;
    return collective_metrics;
  }
  ChunkPrefetcher prefetcher(&source);
  TC_RETURN_IF_ERROR(prefetcher.start(chunk_plans.front().start, chunk_plans.front().length, host_buffer));

  double read_sec = 0.0;
  double h2d_sec = 0.0;
  double wait_ready_sec = 0.0;
  double issue_sec = 0.0;
  double sync_sec = 0.0;
  double root_d2d_sec = 0.0;
  uint64_t bytes_read = 0;
  uint64_t peer_transfer_bytes = 0;
  size_t chunk_count = 0;
  uint64_t raw_piece_count = 0;
  uint64_t merged_piece_count = 0;
  uint64_t remote_group_count = 0;
  size_t max_peer_piece_count = 0;
  size_t max_group_pairs = 0;
  bool first_chunk_logged = false;

  const auto& segment_refs = *segment_refs_or;
  for (size_t chunk_index = 0; chunk_index < chunk_plans.size(); ++chunk_index) {
    const ChunkReadPlan& plan = chunk_plans[chunk_index];
    char* active_host_buffer = (chunk_index % 2 == 0) ? host_buffer : prefetch_buffer;
    char* next_host_buffer = (chunk_index % 2 == 0) ? prefetch_buffer : host_buffer;
    {
      double chunk_read_sec = 0.0;
      TC_RETURN_IF_ERROR(prefetcher.wait(&chunk_read_sec));
      read_sec += chunk_read_sec;
    }
    if (chunk_index + 1 < chunk_plans.size()) {
      const ChunkReadPlan& next_plan = chunk_plans[chunk_index + 1];
      TC_RETURN_IF_ERROR(prefetcher.start(next_plan.start, next_plan.length, next_host_buffer));
    }

    chunk_count += 1;
    const uint64_t chunk_start = plan.start;
    const uint64_t chunk_end = plan.end;
    const size_t chunk_len = plan.length;
    const MappedSourceWindow& window = *plan.window;
    bytes_read += chunk_len;
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(root_device_id));
    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(
          tensorcast::cuda::memcpy_async(
              root_stage->get(), active_host_buffer, chunk_len, cudaMemcpyHostToDevice, h2d_stream));
      TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(ready_event, h2d_stream));
      h2d_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(clique->wait_stream_on_event(root_rank, ready_event));
      wait_ready_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    {
      std::vector<CopyPiece> local_pieces;
      std::vector<std::vector<CopyPiece>> remote_pieces(participants.size());
      uint64_t chunk_raw_pieces = 0;
      double chunk_sync_sec = 0.0;
      for (size_t segment_index : window.segment_indices) {
        const auto& segment = segment_refs[segment_index];
        const uint64_t segment_end = segment.src_offset + segment.length;
        const uint64_t overlap_begin = std::max<uint64_t>(segment.src_offset, chunk_start);
        const uint64_t overlap_end = std::min<uint64_t>(segment_end, chunk_end);
        if (overlap_end <= overlap_begin) {
          continue;
        }
        const uint64_t overlap_len = overlap_end - overlap_begin;
        const uint64_t src_chunk_offset = overlap_begin - chunk_start;
        const uint64_t dst_logical_offset = segment.dst_offset + (overlap_begin - segment.src_offset);
        const auto& participant = participants[static_cast<size_t>(segment.rank)];
        auto pieces_or = resolve_target_pieces(participant, dst_logical_offset, overlap_len);
        if (!pieces_or.ok()) {
          return pieces_or.status();
        }
        for (const auto& piece : *pieces_or) {
          const auto* src_ptr =
              static_cast<const std::uint8_t*>(root_stage->get()) + src_chunk_offset + piece.src_offset;
          CopyPiece copy_piece{
              .src_ptr = src_ptr,
              .dst_ptr = piece.dst_ptr.get(),
              .length = piece.length,
          };
          chunk_raw_pieces += 1;
          if (segment.rank == root_rank) {
            append_merged_copy_piece(local_pieces, copy_piece);
          } else {
            append_merged_copy_piece(remote_pieces[static_cast<size_t>(segment.rank)], copy_piece);
          }
        }
      }
      raw_piece_count += chunk_raw_pieces;
      size_t chunk_merged_pieces = local_pieces.size();
      for (size_t rank = 0; rank < remote_pieces.size(); ++rank) {
        if (static_cast<int>(rank) == root_rank) {
          continue;
        }
        chunk_merged_pieces += remote_pieces[rank].size();
        max_peer_piece_count = std::max(max_peer_piece_count, remote_pieces[rank].size());
      }
      merged_piece_count += static_cast<uint64_t>(chunk_merged_pieces);

      for (const auto& piece : local_pieces) {
        const auto local_copy_start = std::chrono::steady_clock::now();
        TC_RETURN_IF_ERROR(
            tensorcast::cuda::memcpy_async(
                piece.dst_ptr, piece.src_ptr, piece.length, cudaMemcpyDeviceToDevice, clique->stream(root_rank)));
        root_d2d_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - local_copy_start).count();
      }

      size_t current_group_pairs = 0;
      bool group_open = false;
      size_t chunk_remote_groups = 0;
      auto flush_remote_group = [&]() -> absl::Status {
        if (!group_open) {
          return absl::OkStatus();
        }
        TC_RETURN_IF_ERROR(clique->group_end());
        chunk_remote_groups += 1;
        remote_group_count += 1;
        max_group_pairs = std::max(max_group_pairs, current_group_pairs);
        current_group_pairs = 0;
        group_open = false;
        const auto sync_start = std::chrono::steady_clock::now();
        TC_RETURN_IF_ERROR(clique->synchronize_all());
        chunk_sync_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - sync_start).count();
        return absl::OkStatus();
      };

      for (size_t rank = 0; rank < remote_pieces.size(); ++rank) {
        if (static_cast<int>(rank) == root_rank) {
          continue;
        }
        for (const auto& piece : remote_pieces[rank]) {
          if (!group_open) {
            TC_RETURN_IF_ERROR(clique->group_start());
            group_open = true;
          }
          if (current_group_pairs >= kMaxMappedPeerPairsPerNcclGroup) {
            TC_RETURN_IF_ERROR(flush_remote_group());
            TC_RETURN_IF_ERROR(clique->group_start());
            group_open = true;
          }
          const auto issue_start = std::chrono::steady_clock::now();
          peer_transfer_bytes += piece.length;
          TC_RETURN_IF_ERROR(clique->send_u8(root_rank, piece.src_ptr, piece.length, static_cast<int>(rank)));
          TC_RETURN_IF_ERROR(clique->recv_u8(static_cast<int>(rank), piece.dst_ptr, piece.length, root_rank));
          issue_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - issue_start).count();
          current_group_pairs += 1;
        }
      }
      if (group_open) {
        TC_RETURN_IF_ERROR(flush_remote_group());
      } else {
        const auto sync_start = std::chrono::steady_clock::now();
        TC_RETURN_IF_ERROR(clique->synchronize_all());
        chunk_sync_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - sync_start).count();
      }
      if (!first_chunk_logged) {
        size_t chunk_remote_merged_pieces = chunk_merged_pieces - local_pieces.size();
        LOG(INFO) << "collective_mapped_target first_chunk_stats"
                  << " artifact_id=" << participants.front().artifact_id << " raw_pieces=" << chunk_raw_pieces
                  << " merged_local_pieces=" << local_pieces.size()
                  << " merged_remote_pieces=" << chunk_remote_merged_pieces << " remote_groups=" << chunk_remote_groups
                  << " max_peer_pieces=" << ([&]() {
                       size_t value = 0;
                       for (size_t rank = 0; rank < remote_pieces.size(); ++rank) {
                         if (static_cast<int>(rank) == root_rank) {
                           continue;
                         }
                         value = std::max(value, remote_pieces[rank].size());
                       }
                       return value;
                     })();
        first_chunk_logged = true;
      }
      sync_sec += chunk_sync_sec;
    }
  }

  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(root_device_id));
  TC_RETURN_IF_ERROR(tensorcast::cuda::event_destroy(ready_event));
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_destroy(h2d_stream));
  if (dim1_pack_workspace.pack_stream != nullptr) {
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_destroy(dim1_pack_workspace.pack_stream));
  }

  const auto total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  LOG(INFO) << "collective_mapped_target timings: clique_init=" << clique_sec << "s"
            << " clique_cache_hit=" << (clique_cache_hit ? 1 : 0) << " pinned_alloc=" << pinned_alloc_sec << "s"
            << " root_alloc=" << root_alloc_sec << "s"
            << " build_segments=" << segments_sec << "s"
            << " windows=" << windows_or->size() << " segments=" << segment_refs.size()
            << " chunk_bytes=" << chunk_bytes << " chunks=" << chunk_count << " bytes_read=" << bytes_read
            << " peer_transfer_bytes=" << peer_transfer_bytes << " raw_pieces=" << raw_piece_count
            << " merged_pieces=" << merged_piece_count << " remote_groups=" << remote_group_count
            << " max_peer_pieces=" << max_peer_piece_count << " max_group_pairs=" << max_group_pairs
            << " read=" << read_sec << "s"
            << " h2d=" << h2d_sec << "s"
            << " wait_ready=" << wait_ready_sec << "s"
            << " issue=" << issue_sec << "s"
            << " sync=" << sync_sec << "s"
            << " root_d2d=" << root_d2d_sec << "s"
            << " total=" << total_sec << "s";
  uint64_t tensor_job_naive_source_bytes = 0;
  for (const auto& runtime_job : tensor_job_build.jobs) {
    tensor_job_naive_source_bytes += owner_job_naive_source_bytes(runtime_job.job);
  }
  collective_metrics.unique_source_bytes =
      tensor_job_build.handled_source_bytes + concat_job_build.handled_source_bytes + bytes_read;
  collective_metrics.peer_transfer_bytes = peer_transfer_bytes;
  collective_metrics.peak_temporary_bytes =
      std::max<uint64_t>(chunk_bytes, std::max<uint64_t>(root_stage->size(), dim1_pack_workspace.capacity_bytes));
  collective_metrics.batch_count = chunk_count + tensor_job_count + concat_job_count;
  collective_metrics.dedup_saving_bytes = tensor_job_naive_source_bytes > tensor_job_build.handled_source_bytes
      ? tensor_job_naive_source_bytes - tensor_job_build.handled_source_bytes
      : 0;
  return collective_metrics;
}

absl::Status wait_copy_handles(std::vector<common::CopyHandle>* handles) {
  for (const auto& handle : *handles) {
    TC_RETURN_IF_ERROR(handle.wait());
  }
  handles->clear();
  return absl::OkStatus();
}

absl::Status submit_target_layout_write(
    const ParsedMappedParticipant& participant,
    loader::TargetLayoutGpuSink& sink,
    uint64_t logical_offset,
    const void* src,
    uint64_t length,
    std::vector<common::CopyHandle>* handles) {
  if (length == 0) {
    return absl::OkStatus();
  }
  auto pieces_or = resolve_target_pieces(participant, logical_offset, length);
  if (!pieces_or.ok()) {
    return pieces_or.status();
  }
  const auto* src_bytes = reinterpret_cast<const std::uint8_t*>(src);
  for (const auto& piece : *pieces_or) {
    if (piece.length > std::numeric_limits<size_t>::max()) {
      return absl::OutOfRangeError("local mapped target write exceeds host size_t limit");
    }
    auto handle_or = sink.write_at_async(
        logical_offset + piece.src_offset,
        src_bytes + piece.src_offset,
        static_cast<size_t>(piece.length),
        loader::AsyncPositionedSink::AsyncWriteOptions{});
    if (!handle_or.ok()) {
      return handle_or.status();
    }
    handles->push_back(std::move(*handle_or));
  }
  return absl::OkStatus();
}

absl::Status append_local_mapped_concat_task(LocalMappedConcatTask task, LocalMappedConcatTaskBuildResult* result) {
  if (result == nullptr) {
    return absl::InvalidArgumentError("local mapped concat task result is null");
  }
  if (task.block_bytes == 0 || task.block_count == 0) {
    return absl::OkStatus();
  }
  if (task.block_count > std::numeric_limits<uint64_t>::max() / task.block_bytes) {
    return absl::OutOfRangeError("local mapped concat task byte count overflow");
  }
  const uint64_t bytes = task.total_bytes();
  if (bytes > std::numeric_limits<size_t>::max()) {
    return absl::OutOfRangeError("local mapped concat task exceeds host size_t limit");
  }
  result->source_bytes += bytes;
  if (task.direct_dst_ptr != nullptr) {
    result->direct_tasks += 1;
    result->direct_bytes += bytes;
  } else {
    result->layout_tasks += 1;
    result->layout_bytes += bytes;
  }
  if (task.block_count > 1 && task.dst_block_stride_bytes != task.block_bytes) {
    result->strided_tasks += 1;
    result->strided_bytes += bytes;
  }
  result->tasks.push_back(task);
  return absl::OkStatus();
}

absl::Status append_local_mapped_concat_job_tasks(
    const ParsedMappedParticipant& participant,
    const MappedConcatJobRuntime& job,
    size_t host_buffer_bytes,
    LocalMappedConcatTaskBuildResult* result) {
  if (job.destinations.size() != 1) {
    return absl::InvalidArgumentError("local mapped concat task build requires exactly one destination");
  }
  for (const auto& fragment : job.fragments) {
    const auto source_base_offset_or = source_base_offset_bytes(fragment.source);
    if (!source_base_offset_or.ok()) {
      return source_base_offset_or.status();
    }
    const uint64_t source_base_offset = *source_base_offset_or;
    if (fragment.src_block_bytes == 0 || fragment.dst_block_bytes == 0) {
      continue;
    }
    if (fragment.dst_logical_begins_by_rank.empty()) {
      return absl::FailedPreconditionError("local mapped concat requires destination logical offsets");
    }

    int64_t src_start = fragment.src_start;
    int64_t src_end = fragment.src_end;
    if (!fragment.src_starts_by_rank.empty()) {
      if (fragment.src_ends_by_rank.empty()) {
        return absl::InvalidArgumentError("local mapped concat has mismatched source rank ranges");
      }
      src_start = fragment.src_starts_by_rank.front();
      src_end = fragment.src_ends_by_rank.front();
    }
    if (src_start < 0 || src_end <= src_start) {
      return absl::InvalidArgumentError("local mapped concat fragment has invalid source range");
    }

    const uint64_t dst_logical_begin = fragment.dst_logical_begins_by_rank.front();
    std::uint8_t* direct_dst_base = nullptr;
    if (!fragment.dst_ptrs.empty()) {
      direct_dst_base = static_cast<std::uint8_t*>(fragment.dst_ptrs.front());
    }

    if (fragment.prefix_count == 1) {
      const uint64_t source_rows = static_cast<uint64_t>(src_end - src_start);
      if (source_rows == 0 || fragment.src_block_bytes % source_rows != 0) {
        return absl::InvalidArgumentError("local mapped single-range concat has invalid dim0 source block");
      }
      const uint64_t source_row_bytes = fragment.src_block_bytes / source_rows;
      uint64_t copied = 0;
      while (copied < fragment.src_block_bytes) {
        const uint64_t task_bytes =
            std::min<uint64_t>(fragment.src_block_bytes - copied, static_cast<uint64_t>(host_buffer_bytes));
        TC_RETURN_IF_ERROR(append_local_mapped_concat_task(
            LocalMappedConcatTask{
                .source_offset = source_base_offset + static_cast<uint64_t>(src_start) * source_row_bytes + copied,
                .dst_logical_offset = dst_logical_begin + copied,
                .block_bytes = task_bytes,
                .block_count = 1,
                .dst_block_stride_bytes = task_bytes,
                .direct_dst_ptr = direct_dst_base != nullptr ? direct_dst_base + copied : nullptr,
            },
            result));
        copied += task_bytes;
      }
      continue;
    }

    const uint64_t source_rows = static_cast<uint64_t>(src_end - src_start);
    if (fragment.prefix_count == 0 || source_rows % fragment.prefix_count != 0) {
      return absl::InvalidArgumentError("local mapped multi-range concat has invalid source block geometry");
    }
    const uint64_t source_rows_per_block = source_rows / fragment.prefix_count;
    if (source_rows_per_block == 0 || fragment.src_block_bytes % source_rows_per_block != 0) {
      return absl::InvalidArgumentError("local mapped multi-range concat has invalid per-row source block size");
    }
    if (fragment.src_block_bytes != fragment.dst_block_bytes) {
      return absl::InvalidArgumentError("local mapped multi-range concat source/destination block mismatch");
    }
    if (fragment.src_block_bytes > host_buffer_bytes) {
      return absl::FailedPreconditionError("local mapped multi-range concat block exceeds pinned buffer size");
    }
    const uint64_t source_row_bytes = fragment.src_block_bytes / source_rows_per_block;
    const uint64_t blocks_per_task =
        std::max<uint64_t>(1, static_cast<uint64_t>(host_buffer_bytes) / fragment.src_block_bytes);
    for (uint64_t block = 0; block < fragment.prefix_count; block += blocks_per_task) {
      const uint64_t task_blocks = std::min<uint64_t>(fragment.prefix_count - block, blocks_per_task);
      TC_RETURN_IF_ERROR(append_local_mapped_concat_task(
          LocalMappedConcatTask{
              .source_offset = source_base_offset +
                  (static_cast<uint64_t>(src_start) + block * source_rows_per_block) * source_row_bytes,
              .dst_logical_offset = dst_logical_begin + block * fragment.dst_block_stride_bytes,
              .block_bytes = fragment.dst_block_bytes,
              .block_count = task_blocks,
              .dst_block_stride_bytes = fragment.dst_block_stride_bytes,
              .direct_dst_ptr =
                  direct_dst_base != nullptr ? direct_dst_base + block * fragment.dst_block_stride_bytes : nullptr,
          },
          result));
    }
  }
  (void)participant;
  return absl::OkStatus();
}

absl::StatusOr<LocalMappedConcatTaskBuildResult> build_local_mapped_concat_tasks(
    const ParsedMappedParticipant& participant,
    const std::vector<MappedConcatJobRuntime>& jobs,
    size_t host_buffer_bytes) {
  LocalMappedConcatTaskBuildResult result;
  for (const auto& job : jobs) {
    TC_RETURN_IF_ERROR(append_local_mapped_concat_job_tasks(participant, job, host_buffer_bytes, &result));
  }
  std::sort(result.tasks.begin(), result.tasks.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.source_offset != rhs.source_offset) {
      return lhs.source_offset < rhs.source_offset;
    }
    if (lhs.dst_logical_offset != rhs.dst_logical_offset) {
      return lhs.dst_logical_offset < rhs.dst_logical_offset;
    }
    return lhs.total_bytes() < rhs.total_bytes();
  });
  return result;
}

absl::Status execute_local_mapped_concat_task_copy(
    const ParsedMappedParticipant& participant,
    const LocalMappedConcatTask& task,
    const void* host_data,
    loader::TargetLayoutGpuSink& sink,
    cudaStream_t stream) {
  const uint64_t total_bytes = task.total_bytes();
  if (total_bytes == 0) {
    return absl::OkStatus();
  }
  if (total_bytes > std::numeric_limits<size_t>::max() || task.block_bytes > std::numeric_limits<size_t>::max() ||
      task.block_count > std::numeric_limits<size_t>::max() ||
      task.dst_block_stride_bytes > std::numeric_limits<size_t>::max()) {
    return absl::OutOfRangeError("local mapped concat task exceeds host size_t limit");
  }

  if (task.direct_dst_ptr != nullptr) {
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participant.device_id));
    if (task.block_count == 1 || task.dst_block_stride_bytes == task.block_bytes) {
      TC_RETURN_IF_ERROR(
          tensorcast::cuda::memcpy_async(
              task.direct_dst_ptr, host_data, static_cast<size_t>(total_bytes), cudaMemcpyHostToDevice, stream));
    } else {
      SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
          task.direct_dst_ptr,
          static_cast<size_t>(task.dst_block_stride_bytes),
          host_data,
          static_cast<size_t>(task.block_bytes),
          static_cast<size_t>(task.block_bytes),
          static_cast<size_t>(task.block_count),
          cudaMemcpyHostToDevice,
          stream));
    }
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(stream));
    return absl::OkStatus();
  }

  std::vector<common::CopyHandle> copy_handles;
  if (task.block_count == 1 || task.dst_block_stride_bytes == task.block_bytes) {
    TC_RETURN_IF_ERROR(
        submit_target_layout_write(participant, sink, task.dst_logical_offset, host_data, total_bytes, &copy_handles));
    return wait_copy_handles(&copy_handles);
  }

  const auto* host_bytes = static_cast<const std::uint8_t*>(host_data);
  for (uint64_t block = 0; block < task.block_count; ++block) {
    TC_RETURN_IF_ERROR(submit_target_layout_write(
        participant,
        sink,
        task.dst_logical_offset + block * task.dst_block_stride_bytes,
        host_bytes + block * task.block_bytes,
        task.block_bytes,
        &copy_handles));
  }
  return wait_copy_handles(&copy_handles);
}

absl::StatusOr<LocalMappedConcatExecutionStats> execute_local_mapped_concat_jobs_streaming(
    const ParsedMappedParticipant& participant,
    const std::vector<MappedConcatJobRuntime>& jobs,
    loader::SeekableSource& source,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    size_t host_buffer_bytes,
    size_t streaming_buffer_chunks,
    loader::TargetLayoutGpuSink& sink) {
  LocalMappedConcatExecutionStats stats;
  const auto total_start = std::chrono::steady_clock::now();
  if (jobs.empty()) {
    return stats;
  }
  if (pinned_pool == nullptr) {
    return absl::InvalidArgumentError("local mapped concat streaming requires a pinned pool");
  }
  const auto task_build_start = std::chrono::steady_clock::now();
  auto task_build_or = build_local_mapped_concat_tasks(participant, jobs, host_buffer_bytes);
  if (!task_build_or.ok()) {
    return task_build_or.status();
  }
  auto task_build = std::move(*task_build_or);
  const double task_build_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - task_build_start).count();
  if (task_build.tasks.empty()) {
    return stats;
  }
  const size_t concurrency = std::max<size_t>(1, streaming_buffer_chunks);
  auto session_spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
      /*num_chunks=*/concurrency, host_buffer_bytes, pinned_pool);
  TC_RETURN_IF_ERROR(session_spb->initialize(
      pinned_timeout, absl::StrCat("local_mapped_concat artifact_id=", participant.artifact_id)));

  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participant.device_id));
  cudaStream_t stream = nullptr;
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&stream, cudaStreamNonBlocking));
  absl::Cleanup destroy_stream = [&stream]() {
    if (stream != nullptr) {
      (void)tensorcast::cuda::stream_destroy(stream);
    }
  };

  absl::Mutex status_mu;
  absl::Status first_status;
  std::atomic<bool> stop{false};
  std::atomic<size_t> next_task{0};
  std::atomic<uint64_t> producer_get_free_ns{0};
  std::atomic<uint64_t> producer_read_ns{0};
  std::atomic<uint64_t> producer_mark_ready_ns{0};
  std::atomic<uint64_t> producer_tasks{0};
  std::atomic<uint64_t> producer_bytes{0};
  auto producers_done = std::make_shared<absl::BlockingCounter>(static_cast<int>(concurrency));
  auto producers_remaining = std::make_shared<std::atomic<size_t>>(concurrency);

  auto record_failure = [&](absl::Status status) {
    if (status.ok()) {
      return;
    }
    {
      absl::MutexLock lock(&status_mu);
      if (first_status.ok()) {
        first_status = std::move(status);
      }
    }
    stop.store(true, std::memory_order_release);
  };

  auto executor = whole_source_load_runtime().blocking_executor();
  for (size_t worker = 0; worker < concurrency; ++worker) {
    executor->add([&, producers_done, producers_remaining]() {
      while (!stop.load(std::memory_order_acquire)) {
        const size_t task_index = next_task.fetch_add(1, std::memory_order_acq_rel);
        if (task_index >= task_build.tasks.size()) {
          break;
        }
        const auto& task = task_build.tasks[task_index];
        const auto get_free_start = std::chrono::steady_clock::now();
        auto slot_or = session_spb->get_free_chunk();
        producer_get_free_ns.fetch_add(elapsed_ns_since(get_free_start), std::memory_order_relaxed);
        if (!slot_or.ok()) {
          if (!stop.load(std::memory_order_acquire)) {
            record_failure(slot_or.status());
          }
          break;
        }
        const int slot_id = *slot_or;
        if (stop.load(std::memory_order_acquire)) {
          (void)session_spb->abort_producer_slot(slot_id);
          break;
        }
        char* host_ptr = session_spb->get_chunk_ptr(slot_id);
        if (host_ptr == nullptr) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(absl::InternalError("local mapped concat streaming got a null pinned chunk"));
          break;
        }
        const size_t bytes = static_cast<size_t>(task.total_bytes());
        const auto read_start = std::chrono::steady_clock::now();
        auto got_or = source.read_at(task.source_offset, host_ptr, bytes);
        producer_read_ns.fetch_add(elapsed_ns_since(read_start), std::memory_order_relaxed);
        if (!got_or.ok()) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(got_or.status());
          break;
        }
        if (*got_or != bytes) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(
              absl::OutOfRangeError(
                  absl::StrCat("short read in local mapped concat streaming: got=", *got_or, " want=", bytes)));
          break;
        }
        const auto mark_ready_start = std::chrono::steady_clock::now();
        const absl::Status ready_status = session_spb->mark_chunk_ready(slot_id, task_index, bytes);
        producer_mark_ready_ns.fetch_add(elapsed_ns_since(mark_ready_start), std::memory_order_relaxed);
        if (!ready_status.ok()) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(ready_status);
          break;
        }
        producer_tasks.fetch_add(1, std::memory_order_relaxed);
        producer_bytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed);
      }
      if (producers_remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
        session_spb->signal_production_complete();
      }
      producers_done->DecrementCount();
    });
  }

  uint64_t consumer_get_ready_ns = 0;
  uint64_t consumer_copy_ns = 0;
  uint64_t consumer_return_ns = 0;
  uint64_t consumer_tasks = 0;
  uint64_t consumer_bytes = 0;
  while (true) {
    const auto get_ready_start = std::chrono::steady_clock::now();
    auto chunk_or = session_spb->get_ready_chunk();
    consumer_get_ready_ns += elapsed_ns_since(get_ready_start);
    if (!chunk_or.ok()) {
      if (!absl::IsOutOfRange(chunk_or.status()) && !absl::IsUnavailable(chunk_or.status())) {
        record_failure(chunk_or.status());
        session_spb->signal_production_complete();
      }
      break;
    }
    auto chunk = *chunk_or;
    absl::Cleanup return_chunk = [&]() {
      const auto return_start = std::chrono::steady_clock::now();
      const absl::Status return_status = session_spb->return_chunk(chunk.slot_id);
      consumer_return_ns += elapsed_ns_since(return_start);
      if (!return_status.ok()) {
        record_failure(return_status);
      }
    };

    if (chunk.global_chunk_id >= task_build.tasks.size()) {
      record_failure(absl::InternalError("local mapped concat streaming chunk id out of range"));
      continue;
    }
    const auto& task = task_build.tasks[chunk.global_chunk_id];
    if (chunk.bytes_in_chunk != task.total_bytes()) {
      record_failure(absl::InternalError("local mapped concat streaming chunk size mismatch"));
      continue;
    }
    if (stop.load(std::memory_order_acquire)) {
      continue;
    }
    const auto copy_start = std::chrono::steady_clock::now();
    absl::Status copy_status = execute_local_mapped_concat_task_copy(participant, task, chunk.data_ptr, sink, stream);
    consumer_copy_ns += elapsed_ns_since(copy_start);
    consumer_tasks += 1;
    consumer_bytes += static_cast<uint64_t>(chunk.bytes_in_chunk);
    if (!copy_status.ok()) {
      record_failure(std::move(copy_status));
    }
  }

  producers_done->Wait();
  {
    absl::MutexLock lock(&status_mu);
    if (!first_status.ok()) {
      return first_status;
    }
  }

  stats.tasks = task_build.tasks.size();
  stats.direct_tasks = task_build.direct_tasks;
  stats.layout_tasks = task_build.layout_tasks;
  stats.strided_tasks = task_build.strided_tasks;
  stats.read_bytes = task_build.source_bytes;
  stats.direct_bytes = task_build.direct_bytes;
  stats.layout_bytes = task_build.layout_bytes;
  stats.strided_bytes = task_build.strided_bytes;
  stats.exec_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  LOG(INFO) << "local_mapped_concat_streaming_summary"
            << " artifact_id=" << participant.artifact_id << " jobs=" << jobs.size() << " tasks=" << stats.tasks
            << " direct_tasks=" << stats.direct_tasks << " layout_tasks=" << stats.layout_tasks
            << " strided_tasks=" << stats.strided_tasks << " read_bytes=" << stats.read_bytes
            << " direct_bytes=" << stats.direct_bytes << " layout_bytes=" << stats.layout_bytes
            << " strided_bytes=" << stats.strided_bytes << " chunk_bytes=" << host_buffer_bytes
            << " streaming_buffer_chunks=" << concurrency << " task_build_sec=" << task_build_sec
            << " producer_tasks=" << producer_tasks.load(std::memory_order_relaxed)
            << " producer_bytes=" << producer_bytes.load(std::memory_order_relaxed)
            << " producer_get_free_sec=" << seconds_from_ns(producer_get_free_ns.load(std::memory_order_relaxed))
            << " producer_read_sec=" << seconds_from_ns(producer_read_ns.load(std::memory_order_relaxed))
            << " producer_mark_ready_sec=" << seconds_from_ns(producer_mark_ready_ns.load(std::memory_order_relaxed))
            << " consumer_tasks=" << consumer_tasks << " consumer_bytes=" << consumer_bytes
            << " consumer_get_ready_sec=" << seconds_from_ns(consumer_get_ready_ns)
            << " consumer_copy_sec=" << seconds_from_ns(consumer_copy_ns)
            << " consumer_return_sec=" << seconds_from_ns(consumer_return_ns) << " exec=" << stats.exec_sec << "s";
  return stats;
}

absl::Status validate_local_mapped_tensor_job_common(
    const MappedTensorJobRuntime& job,
    int device_id,
    RankTensorSlice::Kind expected_kind,
    std::string_view role) {
  if (job.destinations.size() != 1 || job.job.slices.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(role, " requires one destination and one slice"));
  }
  const auto& destination = job.destinations.front();
  if (destination.gpu_ptr == nullptr || destination.device_id != device_id) {
    return absl::InvalidArgumentError(absl::StrCat(role, " destination does not match target device"));
  }
  const auto& slice = job.job.slices.front();
  if (slice.kind != expected_kind || slice.dst_size_bytes == 0) {
    return absl::InvalidArgumentError(absl::StrCat(role, " has invalid or empty destination slice"));
  }
  if (job.job.source.elem_size == 0) {
    return absl::InvalidArgumentError(absl::StrCat(role, " has zero element size"));
  }
  return source_base_offset_bytes(job.job.source).status();
}

absl::Status validate_local_mapped_replicated_tensor_job_admission(const MappedTensorJobRuntime& job, int device_id) {
  TC_RETURN_IF_ERROR(validate_local_mapped_tensor_job_common(
      job, device_id, RankTensorSlice::Kind::kFull, "local mapped replicated tensor job"));
  const auto& slice = job.job.slices.front();
  if (job.job.source.size_bytes != 0 && slice.dst_size_bytes > job.job.source.size_bytes) {
    return absl::InvalidArgumentError("local mapped replicated tensor job destination exceeds source tensor bytes");
  }
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> local_mapped_2d_row_bytes(const TensorMeta& source, std::string_view role) {
  if (source.shape.size() != 2 || source.shape[0] <= 0 || source.shape[1] <= 0 || source.elem_size == 0) {
    return absl::InvalidArgumentError(absl::StrCat(role, " requires a non-empty 2D source tensor"));
  }
  return checked_mul_u64(static_cast<uint64_t>(source.shape[1]), source.elem_size, absl::StrCat(role, " row bytes"));
}

absl::Status validate_local_mapped_dim0_tensor_job_admission(
    const MappedTensorJobRuntime& job,
    int device_id,
    size_t host_buffer_bytes) {
  if (host_buffer_bytes == 0) {
    return absl::InvalidArgumentError("local mapped dim0 tensor executor requires a non-empty host buffer");
  }
  TC_RETURN_IF_ERROR(validate_local_mapped_tensor_job_common(
      job, device_id, RankTensorSlice::Kind::kDim0, "local mapped dim0 tensor job"));
  const auto& slice = job.job.slices.front();
  if (slice.start < 0 || slice.length == 0 || job.job.source.shape.empty()) {
    return absl::InvalidArgumentError("local mapped dim0 tensor job has invalid dim0 geometry");
  }
  return absl::OkStatus();
}

absl::Status validate_local_mapped_dim1_tensor_job_admission(
    const MappedTensorJobRuntime& job,
    int device_id,
    size_t host_buffer_bytes) {
  if (host_buffer_bytes == 0) {
    return absl::InvalidArgumentError("local mapped dim1 tensor executor requires a non-empty host buffer");
  }
  TC_RETURN_IF_ERROR(validate_local_mapped_tensor_job_common(
      job, device_id, RankTensorSlice::Kind::kDim1, "local mapped dim1 tensor job"));
  const auto row_bytes_or = local_mapped_2d_row_bytes(job.job.source, "local mapped dim1 tensor job");
  if (!row_bytes_or.ok()) {
    return row_bytes_or.status();
  }
  const uint64_t rows = static_cast<uint64_t>(job.job.source.shape[0]);
  const auto& slice = job.job.slices.front();
  if (slice.start < 0 || slice.dst_size_bytes % rows != 0) {
    return absl::InvalidArgumentError("local mapped dim1 tensor job has invalid row geometry");
  }
  if (*row_bytes_or > host_buffer_bytes) {
    return absl::FailedPreconditionError("local mapped dim1 tensor row exceeds pinned buffer size");
  }
  return absl::OkStatus();
}

absl::Status validate_local_mapped_rect2d_tensor_job_admission(
    const MappedTensorJobRuntime& job,
    int device_id,
    size_t host_buffer_bytes) {
  if (host_buffer_bytes == 0) {
    return absl::InvalidArgumentError("local mapped rect2d tensor executor requires a non-empty host buffer");
  }
  TC_RETURN_IF_ERROR(validate_local_mapped_tensor_job_common(
      job, device_id, RankTensorSlice::Kind::kRect2D, "local mapped rect2d tensor job"));
  const auto row_bytes_or = local_mapped_2d_row_bytes(job.job.source, "local mapped rect2d tensor job");
  if (!row_bytes_or.ok()) {
    return row_bytes_or.status();
  }
  const uint64_t rows = static_cast<uint64_t>(job.job.source.shape[0]);
  const uint64_t cols = static_cast<uint64_t>(job.job.source.shape[1]);
  const auto& slice = job.job.slices.front();
  if (slice.row_count == 0 || slice.col_count == 0 || slice.row_start > rows ||
      slice.row_count > rows - slice.row_start || slice.src_col_start > cols ||
      slice.col_count > cols - slice.src_col_start) {
    return absl::InvalidArgumentError("local mapped rect2d tensor job has invalid source geometry");
  }
  const auto col_bytes_or =
      checked_mul_u64(slice.col_count, job.job.source.elem_size, "local mapped rect2d tensor selected bytes");
  if (!col_bytes_or.ok()) {
    return col_bytes_or.status();
  }
  const auto expected_dst_bytes_or =
      checked_mul_u64(slice.row_count, *col_bytes_or, "local mapped rect2d tensor destination bytes");
  if (!expected_dst_bytes_or.ok()) {
    return expected_dst_bytes_or.status();
  }
  if (*expected_dst_bytes_or != slice.dst_size_bytes) {
    return absl::InvalidArgumentError("local mapped rect2d tensor job destination byte size mismatch");
  }
  if (*row_bytes_or > host_buffer_bytes) {
    return absl::FailedPreconditionError("local mapped rect2d tensor row exceeds pinned buffer size");
  }
  return absl::OkStatus();
}

absl::Status validate_local_mapped_tensor_jobs_admission(
    const std::vector<MappedTensorJobRuntime>& jobs,
    int device_id,
    size_t host_buffer_bytes) {
  for (const auto& job : jobs) {
    if (job.job.slices.size() != 1) {
      return absl::InvalidArgumentError("local mapped tensor admission requires one slice");
    }
    const auto& slice = job.job.slices.front();
    if (slice.kind == RankTensorSlice::Kind::kRect2D) {
      TC_RETURN_IF_ERROR(validate_local_mapped_rect2d_tensor_job_admission(job, device_id, host_buffer_bytes));
      continue;
    }
    switch (job.job.distribution) {
      case TensorJob::Distribution::kDim0Partitioned:
        TC_RETURN_IF_ERROR(validate_local_mapped_dim0_tensor_job_admission(job, device_id, host_buffer_bytes));
        break;
      case TensorJob::Distribution::kDim1Partitioned:
        TC_RETURN_IF_ERROR(validate_local_mapped_dim1_tensor_job_admission(job, device_id, host_buffer_bytes));
        break;
      case TensorJob::Distribution::kReplicated:
        TC_RETURN_IF_ERROR(validate_local_mapped_replicated_tensor_job_admission(job, device_id));
        break;
    }
  }
  return absl::OkStatus();
}

absl::Status validate_local_mapped_concat_fragment_admission(
    const MappedConcatFragmentRuntime& fragment,
    size_t host_buffer_bytes) {
  if (host_buffer_bytes == 0) {
    return absl::InvalidArgumentError("local mapped concat executor requires a non-empty host buffer");
  }
  TC_RETURN_IF_ERROR(source_base_offset_bytes(fragment.source).status());
  if (fragment.src_block_bytes == 0 || fragment.dst_block_bytes == 0) {
    return absl::OkStatus();
  }
  if (fragment.dst_logical_begins_by_rank.empty()) {
    return absl::InvalidArgumentError("local mapped concat requires destination logical offsets");
  }
  if (fragment.dst_block_stride_bytes < fragment.dst_block_bytes) {
    return absl::InvalidArgumentError("local mapped concat destination block stride is smaller than block bytes");
  }
  int64_t src_start = fragment.src_start;
  int64_t src_end = fragment.src_end;
  if (!fragment.src_starts_by_rank.empty()) {
    if (fragment.src_ends_by_rank.empty()) {
      return absl::InvalidArgumentError("local mapped concat has mismatched source rank ranges");
    }
    src_start = fragment.src_starts_by_rank.front();
    src_end = fragment.src_ends_by_rank.front();
  }
  if (src_start < 0 || src_end <= src_start) {
    return absl::InvalidArgumentError("local mapped concat fragment has invalid source range");
  }
  const uint64_t source_rows = static_cast<uint64_t>(src_end - src_start);
  if (fragment.prefix_count == 1) {
    if (fragment.src_block_bytes % source_rows != 0) {
      return absl::InvalidArgumentError("local mapped single-range concat has invalid dim0 source block");
    }
    return absl::OkStatus();
  }
  if (fragment.prefix_count == 0) {
    return absl::InvalidArgumentError("local mapped multi-range concat has zero prefix count");
  }
  if (fragment.src_block_bytes != fragment.dst_block_bytes) {
    return absl::InvalidArgumentError(
        "local mapped multi-range concat requires matching source and destination blocks");
  }
  if (source_rows % fragment.prefix_count != 0) {
    return absl::InvalidArgumentError("local mapped multi-range concat source rows are not divisible by prefix count");
  }
  const uint64_t source_rows_per_block = source_rows / fragment.prefix_count;
  if (source_rows_per_block == 0 || fragment.src_block_bytes % source_rows_per_block != 0) {
    return absl::InvalidArgumentError("local mapped multi-range concat has invalid per-block source geometry");
  }
  if (fragment.src_block_bytes > host_buffer_bytes) {
    return absl::FailedPreconditionError("local mapped multi-range concat block exceeds pinned buffer size");
  }
  return absl::OkStatus();
}

absl::Status validate_local_mapped_concat_jobs_admission(
    const std::vector<MappedConcatJobRuntime>& jobs,
    size_t host_buffer_bytes) {
  for (const auto& job : jobs) {
    if (job.destinations.size() != 1) {
      return absl::InvalidArgumentError("local mapped concat admission requires exactly one destination");
    }
    for (const auto& fragment : job.fragments) {
      TC_RETURN_IF_ERROR(validate_local_mapped_concat_fragment_admission(fragment, host_buffer_bytes));
    }
  }
  return absl::OkStatus();
}

absl::Status append_local_mapped_tensor_task(LocalMappedTensorTask task, std::vector<LocalMappedTensorTask>* tasks) {
  if (tasks == nullptr) {
    return absl::InvalidArgumentError("local mapped tensor task append requires task storage");
  }
  if (task.dst_ptr == nullptr || task.read_bytes == 0 || task.dst_bytes == 0) {
    return absl::InvalidArgumentError("local mapped tensor task has an empty source or destination");
  }
  if (task.read_bytes > std::numeric_limits<size_t>::max() || task.dst_bytes > std::numeric_limits<size_t>::max()) {
    return absl::OutOfRangeError("local mapped tensor task exceeds host size_t limit");
  }
  switch (task.kind) {
    case LocalMappedTensorTask::Kind::kContiguous:
      if (task.read_bytes != task.dst_bytes) {
        return absl::InvalidArgumentError("local mapped contiguous tensor task size mismatch");
      }
      break;
    case LocalMappedTensorTask::Kind::kRect2D: {
      if (task.rows == 0 || task.src_pitch_bytes == 0 || task.dst_pitch_bytes == 0) {
        return absl::InvalidArgumentError("local mapped rect2d tensor task has invalid pitch geometry");
      }
      if (task.dst_bytes % task.rows != 0) {
        return absl::InvalidArgumentError("local mapped rect2d tensor task has fractional row width");
      }
      const uint64_t width_bytes = task.dst_bytes / task.rows;
      auto expected_read_bytes_or =
          checked_mul_u64(task.rows, task.src_pitch_bytes, "local mapped rect2d tensor task read bytes");
      if (!expected_read_bytes_or.ok()) {
        return expected_read_bytes_or.status();
      }
      if (*expected_read_bytes_or != task.read_bytes) {
        return absl::InvalidArgumentError("local mapped rect2d tensor task read byte size mismatch");
      }
      auto src_end_or =
          checked_add_u64(task.src_col_offset_bytes, width_bytes, "local mapped rect2d tensor task source row end");
      if (!src_end_or.ok()) {
        return src_end_or.status();
      }
      if (*src_end_or > task.src_pitch_bytes || task.dst_pitch_bytes < width_bytes) {
        return absl::InvalidArgumentError("local mapped rect2d tensor task has invalid source or destination pitch");
      }
      break;
    }
  }
  tasks->push_back(task);
  return absl::OkStatus();
}

absl::Status append_local_mapped_replicated_tensor_tasks(
    const MappedTensorJobRuntime& job,
    size_t host_buffer_bytes,
    LocalMappedTensorTaskBuildResult* result) {
  if (job.destinations.size() != 1 || job.job.slices.size() != 1) {
    return absl::InvalidArgumentError("local mapped replicated tensor job requires exactly one destination");
  }
  const auto source_base_offset_or = source_base_offset_bytes(job.job.source);
  if (!source_base_offset_or.ok()) {
    return source_base_offset_or.status();
  }
  const auto& slice = job.job.slices.front();
  if (slice.kind != RankTensorSlice::Kind::kFull || slice.dst_size_bytes == 0) {
    return absl::InvalidArgumentError("local mapped replicated tensor job requires a non-empty full slice");
  }
  if (host_buffer_bytes == 0) {
    return absl::InvalidArgumentError("local mapped tensor executor requires a non-empty host buffer");
  }

  const uint64_t source_begin = *source_base_offset_or;
  auto* dst_base = static_cast<std::uint8_t*>(job.destinations.front().gpu_ptr) + slice.dst_offset;
  uint64_t copied = 0;
  while (copied < slice.dst_size_bytes) {
    const size_t chunk_bytes =
        static_cast<size_t>(std::min<uint64_t>(slice.dst_size_bytes - copied, host_buffer_bytes));
    auto task_source_or = checked_add_u64(source_begin, copied, "local mapped replicated tensor task source offset");
    if (!task_source_or.ok()) {
      return task_source_or.status();
    }
    TC_RETURN_IF_ERROR(append_local_mapped_tensor_task(
        LocalMappedTensorTask{
            .kind = LocalMappedTensorTask::Kind::kContiguous,
            .source_offset = *task_source_or,
            .read_bytes = static_cast<uint64_t>(chunk_bytes),
            .dst_ptr = dst_base + copied,
            .dst_bytes = static_cast<uint64_t>(chunk_bytes),
        },
        &result->tasks));
    copied += static_cast<uint64_t>(chunk_bytes);
  }

  result->stats.replicated_jobs += 1;
  result->stats.read_bytes += slice.dst_size_bytes;
  result->stats.dst_bytes += slice.dst_size_bytes;
  return absl::OkStatus();
}

absl::Status append_local_mapped_dim0_tensor_tasks(
    const MappedTensorJobRuntime& job,
    size_t host_buffer_bytes,
    LocalMappedTensorTaskBuildResult* result) {
  if (job.destinations.size() != 1 || job.job.slices.size() != 1) {
    return absl::InvalidArgumentError("local mapped dim0 tensor job requires exactly one destination");
  }
  const auto source_base_offset_or = source_base_offset_bytes(job.job.source);
  if (!source_base_offset_or.ok()) {
    return source_base_offset_or.status();
  }
  const auto& slice = job.job.slices.front();
  if (slice.kind != RankTensorSlice::Kind::kDim0 || slice.dst_size_bytes == 0) {
    return absl::InvalidArgumentError("local mapped dim0 tensor job requires a non-empty dim0 slice");
  }
  if (host_buffer_bytes == 0) {
    return absl::InvalidArgumentError("local mapped tensor executor requires a non-empty host buffer");
  }

  uint64_t per_row_bytes = job.job.source.elem_size;
  for (size_t dim = 1; dim < job.job.source.shape.size(); ++dim) {
    auto per_row_bytes_or = checked_mul_u64(
        per_row_bytes, static_cast<uint64_t>(job.job.source.shape[dim]), "local mapped dim0 tensor per-row bytes");
    if (!per_row_bytes_or.ok()) {
      return per_row_bytes_or.status();
    }
    per_row_bytes = *per_row_bytes_or;
  }
  auto source_rel_begin_or =
      checked_mul_u64(static_cast<uint64_t>(slice.start), per_row_bytes, "local mapped dim0 tensor source begin");
  if (!source_rel_begin_or.ok()) {
    return source_rel_begin_or.status();
  }
  auto source_begin_or =
      checked_add_u64(*source_base_offset_or, *source_rel_begin_or, "local mapped dim0 tensor source begin");
  if (!source_begin_or.ok()) {
    return source_begin_or.status();
  }
  const uint64_t source_begin = *source_begin_or;
  auto* dst_base = static_cast<std::uint8_t*>(job.destinations.front().gpu_ptr) + slice.dst_offset;

  uint64_t copied = 0;
  while (copied < slice.dst_size_bytes) {
    const size_t chunk_bytes =
        static_cast<size_t>(std::min<uint64_t>(slice.dst_size_bytes - copied, host_buffer_bytes));
    auto task_source_or = checked_add_u64(source_begin, copied, "local mapped dim0 tensor task source offset");
    if (!task_source_or.ok()) {
      return task_source_or.status();
    }
    TC_RETURN_IF_ERROR(append_local_mapped_tensor_task(
        LocalMappedTensorTask{
            .kind = LocalMappedTensorTask::Kind::kContiguous,
            .source_offset = *task_source_or,
            .read_bytes = static_cast<uint64_t>(chunk_bytes),
            .dst_ptr = dst_base + copied,
            .dst_bytes = static_cast<uint64_t>(chunk_bytes),
        },
        &result->tasks));
    copied += static_cast<uint64_t>(chunk_bytes);
  }

  result->stats.dim0_jobs += 1;
  result->stats.read_bytes += slice.dst_size_bytes;
  result->stats.dst_bytes += slice.dst_size_bytes;
  return absl::OkStatus();
}

absl::Status append_local_mapped_dim1_tensor_tasks(
    const MappedTensorJobRuntime& job,
    size_t host_buffer_bytes,
    LocalMappedTensorTaskBuildResult* result) {
  if (job.destinations.size() != 1 || job.job.slices.size() != 1) {
    return absl::InvalidArgumentError("local mapped dim1 tensor job requires exactly one destination");
  }
  const auto source_base_offset_or = source_base_offset_bytes(job.job.source);
  if (!source_base_offset_or.ok()) {
    return source_base_offset_or.status();
  }
  const auto& slice = job.job.slices.front();
  if (slice.kind != RankTensorSlice::Kind::kDim1 || slice.dst_size_bytes == 0) {
    return absl::InvalidArgumentError("local mapped dim1 tensor job requires a non-empty dim1 slice");
  }
  if (job.job.source.shape.size() != 2) {
    return absl::UnimplementedError("local mapped dim1 tensor job requires a 2D source tensor");
  }

  const uint64_t rows = static_cast<uint64_t>(job.job.source.shape[0]);
  const uint64_t cols = static_cast<uint64_t>(job.job.source.shape[1]);
  auto row_bytes_or = checked_mul_u64(cols, job.job.source.elem_size, "local mapped dim1 tensor row bytes");
  if (!row_bytes_or.ok()) {
    return row_bytes_or.status();
  }
  const uint64_t row_bytes = *row_bytes_or;
  if (rows == 0 || row_bytes == 0 || slice.dst_size_bytes % rows != 0) {
    return absl::InvalidArgumentError("local mapped dim1 tensor job has invalid row geometry");
  }
  if (row_bytes > host_buffer_bytes) {
    return absl::ResourceExhaustedError("local mapped dim1 tensor row exceeds pinned buffer size");
  }

  const uint64_t col_bytes = slice.dst_size_bytes / rows;
  const uint64_t src_col_bytes = static_cast<uint64_t>(slice.start) * job.job.source.elem_size;
  const uint64_t dst_pitch_bytes = slice.dst_row_stride_bytes == 0 ? col_bytes : slice.dst_row_stride_bytes;
  const uint64_t rows_per_chunk = std::max<uint64_t>(1, static_cast<uint64_t>(host_buffer_bytes) / row_bytes);
  auto* dst_base = static_cast<std::uint8_t*>(job.destinations.front().gpu_ptr) + slice.dst_offset;

  for (uint64_t row = 0; row < rows; row += rows_per_chunk) {
    const uint64_t chunk_rows = std::min<uint64_t>(rows_per_chunk, rows - row);
    auto chunk_bytes_or = checked_mul_u64(chunk_rows, row_bytes, "local mapped dim1 tensor task read bytes");
    if (!chunk_bytes_or.ok()) {
      return chunk_bytes_or.status();
    }
    auto source_row_offset_or = checked_mul_u64(row, row_bytes, "local mapped dim1 tensor task source row offset");
    if (!source_row_offset_or.ok()) {
      return source_row_offset_or.status();
    }
    auto task_source_or =
        checked_add_u64(*source_base_offset_or, *source_row_offset_or, "local mapped dim1 tensor task source offset");
    if (!task_source_or.ok()) {
      return task_source_or.status();
    }
    auto dst_row_offset_or = checked_mul_u64(row, dst_pitch_bytes, "local mapped dim1 tensor task destination offset");
    if (!dst_row_offset_or.ok()) {
      return dst_row_offset_or.status();
    }
    auto dst_bytes_or = checked_mul_u64(chunk_rows, col_bytes, "local mapped dim1 tensor task destination bytes");
    if (!dst_bytes_or.ok()) {
      return dst_bytes_or.status();
    }
    TC_RETURN_IF_ERROR(append_local_mapped_tensor_task(
        LocalMappedTensorTask{
            .kind = LocalMappedTensorTask::Kind::kRect2D,
            .source_offset = *task_source_or,
            .read_bytes = *chunk_bytes_or,
            .dst_ptr = dst_base + *dst_row_offset_or,
            .dst_bytes = *dst_bytes_or,
            .src_col_offset_bytes = src_col_bytes,
            .src_pitch_bytes = row_bytes,
            .dst_pitch_bytes = dst_pitch_bytes,
            .rows = chunk_rows,
        },
        &result->tasks));
  }

  result->stats.dim1_jobs += 1;
  result->stats.read_bytes += rows * row_bytes;
  result->stats.dst_bytes += slice.dst_size_bytes;
  return absl::OkStatus();
}

absl::Status append_local_mapped_rect2d_tensor_tasks(
    const MappedTensorJobRuntime& job,
    size_t host_buffer_bytes,
    LocalMappedTensorTaskBuildResult* result) {
  if (job.destinations.size() != 1 || job.job.slices.size() != 1) {
    return absl::InvalidArgumentError("local mapped rect2d tensor job requires exactly one destination");
  }
  const auto source_base_offset_or = source_base_offset_bytes(job.job.source);
  if (!source_base_offset_or.ok()) {
    return source_base_offset_or.status();
  }
  const auto& slice = job.job.slices.front();
  if (slice.kind != RankTensorSlice::Kind::kRect2D || slice.dst_size_bytes == 0) {
    return absl::InvalidArgumentError("local mapped rect2d tensor job requires a non-empty rect2d slice");
  }
  if (job.job.source.shape.size() != 2 || job.job.source.elem_size == 0 || job.job.source.shape[0] <= 0 ||
      job.job.source.shape[1] <= 0) {
    return absl::InvalidArgumentError("local mapped rect2d tensor job requires a 2D source tensor");
  }

  const uint64_t rows = static_cast<uint64_t>(job.job.source.shape[0]);
  const uint64_t cols = static_cast<uint64_t>(job.job.source.shape[1]);
  auto row_bytes_or = checked_mul_u64(cols, job.job.source.elem_size, "local mapped rect2d tensor row bytes");
  if (!row_bytes_or.ok()) {
    return row_bytes_or.status();
  }
  const uint64_t row_bytes = *row_bytes_or;
  auto col_bytes_or =
      checked_mul_u64(slice.col_count, job.job.source.elem_size, "local mapped rect2d tensor selected bytes");
  if (!col_bytes_or.ok()) {
    return col_bytes_or.status();
  }
  const uint64_t col_bytes = *col_bytes_or;
  if (rows == 0 || cols == 0 || row_bytes == 0 || col_bytes == 0 || slice.row_count == 0 || slice.row_start > rows ||
      slice.row_count > rows - slice.row_start || slice.src_col_start > cols ||
      slice.col_count > cols - slice.src_col_start) {
    return absl::InvalidArgumentError("local mapped rect2d tensor job has invalid source geometry");
  }
  auto expected_dst_bytes_or =
      checked_mul_u64(slice.row_count, col_bytes, "local mapped rect2d tensor destination bytes");
  if (!expected_dst_bytes_or.ok()) {
    return expected_dst_bytes_or.status();
  }
  if (*expected_dst_bytes_or != slice.dst_size_bytes) {
    return absl::InvalidArgumentError("local mapped rect2d tensor job destination byte size mismatch");
  }
  if (row_bytes > host_buffer_bytes) {
    return absl::ResourceExhaustedError("local mapped rect2d tensor row exceeds pinned buffer size");
  }

  const uint64_t src_col_bytes = slice.src_col_start * job.job.source.elem_size;
  const uint64_t dst_pitch_bytes = slice.dst_row_stride_bytes == 0 ? col_bytes : slice.dst_row_stride_bytes;
  const uint64_t rows_per_chunk = std::max<uint64_t>(1, static_cast<uint64_t>(host_buffer_bytes) / row_bytes);
  auto* dst_base = static_cast<std::uint8_t*>(job.destinations.front().gpu_ptr) + slice.dst_offset;

  for (uint64_t row = 0; row < slice.row_count; row += rows_per_chunk) {
    const uint64_t chunk_rows = std::min<uint64_t>(rows_per_chunk, slice.row_count - row);
    auto chunk_bytes_or = checked_mul_u64(chunk_rows, row_bytes, "local mapped rect2d tensor task read bytes");
    if (!chunk_bytes_or.ok()) {
      return chunk_bytes_or.status();
    }
    auto source_row_or = checked_add_u64(slice.row_start, row, "local mapped rect2d tensor task source row");
    if (!source_row_or.ok()) {
      return source_row_or.status();
    }
    auto source_row_offset_or =
        checked_mul_u64(*source_row_or, row_bytes, "local mapped rect2d tensor task source row offset");
    if (!source_row_offset_or.ok()) {
      return source_row_offset_or.status();
    }
    auto task_source_or =
        checked_add_u64(*source_base_offset_or, *source_row_offset_or, "local mapped rect2d tensor task source offset");
    if (!task_source_or.ok()) {
      return task_source_or.status();
    }
    auto dst_row_offset_or =
        checked_mul_u64(row, dst_pitch_bytes, "local mapped rect2d tensor task destination offset");
    if (!dst_row_offset_or.ok()) {
      return dst_row_offset_or.status();
    }
    auto dst_bytes_or = checked_mul_u64(chunk_rows, col_bytes, "local mapped rect2d tensor task destination bytes");
    if (!dst_bytes_or.ok()) {
      return dst_bytes_or.status();
    }
    TC_RETURN_IF_ERROR(append_local_mapped_tensor_task(
        LocalMappedTensorTask{
            .kind = LocalMappedTensorTask::Kind::kRect2D,
            .source_offset = *task_source_or,
            .read_bytes = *chunk_bytes_or,
            .dst_ptr = dst_base + *dst_row_offset_or,
            .dst_bytes = *dst_bytes_or,
            .src_col_offset_bytes = src_col_bytes,
            .src_pitch_bytes = row_bytes,
            .dst_pitch_bytes = dst_pitch_bytes,
            .rows = chunk_rows,
        },
        &result->tasks));
  }

  auto full_read_bytes_or = checked_mul_u64(slice.row_count, row_bytes, "local mapped rect2d tensor total read bytes");
  if (!full_read_bytes_or.ok()) {
    return full_read_bytes_or.status();
  }
  result->stats.rect2d_jobs += 1;
  result->stats.read_bytes += *full_read_bytes_or;
  result->stats.dst_bytes += slice.dst_size_bytes;
  return absl::OkStatus();
}

absl::StatusOr<LocalMappedTensorTaskBuildResult> build_local_mapped_tensor_tasks(
    const std::vector<MappedTensorJobRuntime>& jobs,
    int device_id,
    size_t host_buffer_bytes) {
  LocalMappedTensorTaskBuildResult result;
  for (const auto& job : jobs) {
    if (job.destinations.size() != 1 || job.job.slices.size() != 1) {
      return absl::InvalidArgumentError("local mapped tensor job requires one destination and one slice");
    }
    const auto& destination = job.destinations.front();
    if (destination.gpu_ptr == nullptr || destination.device_id != device_id) {
      return absl::InvalidArgumentError("local mapped tensor job destination does not match target device");
    }
    const auto& slice = job.job.slices.front();
    if (slice.kind == RankTensorSlice::Kind::kRect2D) {
      TC_RETURN_IF_ERROR(append_local_mapped_rect2d_tensor_tasks(job, host_buffer_bytes, &result));
      continue;
    }
    switch (job.job.distribution) {
      case TensorJob::Distribution::kReplicated:
        TC_RETURN_IF_ERROR(append_local_mapped_replicated_tensor_tasks(job, host_buffer_bytes, &result));
        break;
      case TensorJob::Distribution::kDim0Partitioned:
        TC_RETURN_IF_ERROR(append_local_mapped_dim0_tensor_tasks(job, host_buffer_bytes, &result));
        break;
      case TensorJob::Distribution::kDim1Partitioned:
        TC_RETURN_IF_ERROR(append_local_mapped_dim1_tensor_tasks(job, host_buffer_bytes, &result));
        break;
    }
    if (slice.dst_size_bytes == 0) {
      return absl::InvalidArgumentError("local mapped tensor job has empty destination slice");
    }
  }
  std::stable_sort(
      result.tasks.begin(), result.tasks.end(), [](const LocalMappedTensorTask& lhs, const LocalMappedTensorTask& rhs) {
        return lhs.source_offset < rhs.source_offset;
      });
  result.stats.tasks = result.tasks.size();
  return result;
}

absl::Status execute_local_mapped_tensor_task_copy(
    const LocalMappedTensorTask& task,
    const void* host_data,
    cudaStream_t stream) {
  if (host_data == nullptr) {
    return absl::InvalidArgumentError("local mapped tensor task copy got a null host buffer");
  }
  switch (task.kind) {
    case LocalMappedTensorTask::Kind::kContiguous:
      TC_RETURN_IF_ERROR(
          tensorcast::cuda::memcpy_async(
              task.dst_ptr, host_data, static_cast<size_t>(task.dst_bytes), cudaMemcpyHostToDevice, stream));
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(stream));
      return absl::OkStatus();
    case LocalMappedTensorTask::Kind::kRect2D: {
      if (task.rows == 0 || task.dst_bytes % task.rows != 0) {
        return absl::InvalidArgumentError("local mapped rect2d tensor task has invalid row geometry");
      }
      const uint64_t width_bytes = task.dst_bytes / task.rows;
      if (width_bytes > std::numeric_limits<size_t>::max() ||
          task.src_pitch_bytes > std::numeric_limits<size_t>::max() ||
          task.dst_pitch_bytes > std::numeric_limits<size_t>::max() || task.rows > std::numeric_limits<size_t>::max()) {
        return absl::OutOfRangeError("local mapped rect2d tensor task exceeds host size_t limit");
      }
      const auto* host_bytes = static_cast<const std::uint8_t*>(host_data);
      SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
          task.dst_ptr,
          static_cast<size_t>(task.dst_pitch_bytes),
          host_bytes + task.src_col_offset_bytes,
          static_cast<size_t>(task.src_pitch_bytes),
          static_cast<size_t>(width_bytes),
          static_cast<size_t>(task.rows),
          cudaMemcpyHostToDevice,
          stream));
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(stream));
      return absl::OkStatus();
    }
  }
  return absl::InternalError("unknown local mapped tensor task kind");
}

absl::StatusOr<LocalMappedTensorExecutionStats> execute_local_mapped_tensor_jobs(
    const std::vector<MappedTensorJobRuntime>& jobs,
    int device_id,
    loader::SeekableSource& source,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    size_t host_buffer_bytes,
    size_t streaming_buffer_chunks,
    const std::string& artifact_id) {
  LocalMappedTensorExecutionStats stats;
  const auto total_start = std::chrono::steady_clock::now();
  if (jobs.empty()) {
    return stats;
  }
  if (pinned_pool == nullptr) {
    return absl::InvalidArgumentError("local mapped tensor streaming requires a pinned pool");
  }

  const auto task_build_start = std::chrono::steady_clock::now();
  auto task_build_or = build_local_mapped_tensor_tasks(jobs, device_id, host_buffer_bytes);
  if (!task_build_or.ok()) {
    return task_build_or.status();
  }
  auto task_build = std::move(*task_build_or);
  const double task_build_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - task_build_start).count();
  if (task_build.tasks.empty()) {
    return task_build.stats;
  }
  const size_t concurrency = std::max<size_t>(1, streaming_buffer_chunks);
  auto session_spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
      /*num_chunks=*/concurrency, host_buffer_bytes, pinned_pool);
  TC_RETURN_IF_ERROR(
      session_spb->initialize(pinned_timeout, absl::StrCat("local_mapped_tensor artifact_id=", artifact_id)));

  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_id));
  cudaStream_t stream = nullptr;
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&stream, cudaStreamNonBlocking));
  absl::Cleanup destroy_stream = [&stream]() {
    if (stream != nullptr) {
      (void)tensorcast::cuda::stream_destroy(stream);
    }
  };

  absl::Mutex status_mu;
  absl::Status first_status;
  std::atomic<bool> stop{false};
  std::atomic<size_t> next_task{0};
  std::atomic<uint64_t> producer_get_free_ns{0};
  std::atomic<uint64_t> producer_read_ns{0};
  std::atomic<uint64_t> producer_mark_ready_ns{0};
  std::atomic<uint64_t> producer_tasks{0};
  std::atomic<uint64_t> producer_bytes{0};
  auto producers_done = std::make_shared<absl::BlockingCounter>(static_cast<int>(concurrency));
  auto producers_remaining = std::make_shared<std::atomic<size_t>>(concurrency);

  auto record_failure = [&](absl::Status status) {
    if (status.ok()) {
      return;
    }
    {
      absl::MutexLock lock(&status_mu);
      if (first_status.ok()) {
        first_status = std::move(status);
      }
    }
    stop.store(true, std::memory_order_release);
  };

  auto executor = whole_source_load_runtime().blocking_executor();
  for (size_t worker = 0; worker < concurrency; ++worker) {
    executor->add([&, producers_done, producers_remaining]() {
      while (!stop.load(std::memory_order_acquire)) {
        const size_t task_index = next_task.fetch_add(1, std::memory_order_acq_rel);
        if (task_index >= task_build.tasks.size()) {
          break;
        }
        const auto& task = task_build.tasks[task_index];
        const auto get_free_start = std::chrono::steady_clock::now();
        auto slot_or = session_spb->get_free_chunk();
        producer_get_free_ns.fetch_add(elapsed_ns_since(get_free_start), std::memory_order_relaxed);
        if (!slot_or.ok()) {
          if (!stop.load(std::memory_order_acquire)) {
            record_failure(slot_or.status());
          }
          break;
        }
        const int slot_id = *slot_or;
        if (stop.load(std::memory_order_acquire)) {
          (void)session_spb->abort_producer_slot(slot_id);
          break;
        }
        char* host_ptr = session_spb->get_chunk_ptr(slot_id);
        if (host_ptr == nullptr) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(absl::InternalError("local mapped tensor streaming got a null pinned chunk"));
          break;
        }
        const size_t bytes = static_cast<size_t>(task.read_bytes);
        const auto read_start = std::chrono::steady_clock::now();
        auto got_or = source.read_at(task.source_offset, host_ptr, bytes);
        producer_read_ns.fetch_add(elapsed_ns_since(read_start), std::memory_order_relaxed);
        if (!got_or.ok()) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(got_or.status());
          break;
        }
        if (*got_or != bytes) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(
              absl::OutOfRangeError(
                  absl::StrCat("short read in local mapped tensor streaming: got=", *got_or, " want=", bytes)));
          break;
        }
        const auto mark_ready_start = std::chrono::steady_clock::now();
        const absl::Status ready_status = session_spb->mark_chunk_ready(slot_id, task_index, bytes);
        producer_mark_ready_ns.fetch_add(elapsed_ns_since(mark_ready_start), std::memory_order_relaxed);
        if (!ready_status.ok()) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(ready_status);
          break;
        }
        producer_tasks.fetch_add(1, std::memory_order_relaxed);
        producer_bytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed);
      }
      if (producers_remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
        session_spb->signal_production_complete();
      }
      producers_done->DecrementCount();
    });
  }

  uint64_t consumer_get_ready_ns = 0;
  uint64_t consumer_copy_ns = 0;
  uint64_t consumer_return_ns = 0;
  uint64_t consumer_tasks = 0;
  uint64_t consumer_bytes = 0;
  while (true) {
    const auto get_ready_start = std::chrono::steady_clock::now();
    auto chunk_or = session_spb->get_ready_chunk();
    consumer_get_ready_ns += elapsed_ns_since(get_ready_start);
    if (!chunk_or.ok()) {
      if (!absl::IsOutOfRange(chunk_or.status()) && !absl::IsUnavailable(chunk_or.status())) {
        record_failure(chunk_or.status());
        session_spb->signal_production_complete();
      }
      break;
    }
    auto chunk = *chunk_or;
    absl::Cleanup return_chunk = [&]() {
      const auto return_start = std::chrono::steady_clock::now();
      const absl::Status return_status = session_spb->return_chunk(chunk.slot_id);
      consumer_return_ns += elapsed_ns_since(return_start);
      if (!return_status.ok()) {
        record_failure(return_status);
      }
    };

    if (chunk.global_chunk_id >= task_build.tasks.size()) {
      record_failure(absl::InternalError("local mapped tensor streaming chunk id out of range"));
      continue;
    }
    const auto& task = task_build.tasks[chunk.global_chunk_id];
    if (chunk.bytes_in_chunk != task.read_bytes) {
      record_failure(absl::InternalError("local mapped tensor streaming chunk size mismatch"));
      continue;
    }
    if (stop.load(std::memory_order_acquire)) {
      continue;
    }
    const auto copy_start = std::chrono::steady_clock::now();
    absl::Status copy_status = execute_local_mapped_tensor_task_copy(task, chunk.data_ptr, stream);
    consumer_copy_ns += elapsed_ns_since(copy_start);
    consumer_tasks += 1;
    consumer_bytes += static_cast<uint64_t>(chunk.bytes_in_chunk);
    if (!copy_status.ok()) {
      record_failure(std::move(copy_status));
    }
  }

  producers_done->Wait();
  {
    absl::MutexLock lock(&status_mu);
    if (!first_status.ok()) {
      return first_status;
    }
  }

  stats = task_build.stats;
  stats.exec_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  LOG(INFO) << "local_mapped_tensor_streaming_summary"
            << " artifact_id=" << artifact_id << " jobs=" << jobs.size() << " tasks=" << stats.tasks
            << " replicated_jobs=" << stats.replicated_jobs << " dim0_jobs=" << stats.dim0_jobs
            << " dim1_jobs=" << stats.dim1_jobs << " rect2d_jobs=" << stats.rect2d_jobs
            << " read_bytes=" << stats.read_bytes << " dst_bytes=" << stats.dst_bytes
            << " chunk_bytes=" << host_buffer_bytes << " streaming_buffer_chunks=" << concurrency
            << " task_build_sec=" << task_build_sec
            << " producer_tasks=" << producer_tasks.load(std::memory_order_relaxed)
            << " producer_bytes=" << producer_bytes.load(std::memory_order_relaxed)
            << " producer_get_free_sec=" << seconds_from_ns(producer_get_free_ns.load(std::memory_order_relaxed))
            << " producer_read_sec=" << seconds_from_ns(producer_read_ns.load(std::memory_order_relaxed))
            << " producer_mark_ready_sec=" << seconds_from_ns(producer_mark_ready_ns.load(std::memory_order_relaxed))
            << " consumer_tasks=" << consumer_tasks << " consumer_bytes=" << consumer_bytes
            << " consumer_get_ready_sec=" << seconds_from_ns(consumer_get_ready_ns)
            << " consumer_copy_sec=" << seconds_from_ns(consumer_copy_ns)
            << " consumer_return_sec=" << seconds_from_ns(consumer_return_ns) << " exec=" << stats.exec_sec << "s";
  return stats;
}

absl::StatusOr<LocalMappedSourceOrderedExecutionStats> execute_local_mapped_source_ordered_tasks(
    const ParsedMappedParticipant& participant,
    const LocalMappedTensorTaskBuildResult& tensor_task_build,
    const LocalMappedConcatTaskBuildResult& concat_task_build,
    loader::SeekableSource& source,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    size_t host_buffer_bytes,
    size_t streaming_buffer_chunks,
    loader::TargetLayoutGpuSink& sink) {
  struct SourceOrderedTask {
    enum class Kind : std::uint8_t {
      kTensor = 0,
      kConcat = 1,
    };
    Kind kind{Kind::kTensor};
    size_t index{0};
    uint64_t source_offset{0};
    uint64_t read_bytes{0};
  };

  LocalMappedSourceOrderedExecutionStats stats;
  stats.tensor = tensor_task_build.stats;
  stats.concat.tasks = concat_task_build.tasks.size();
  stats.concat.direct_tasks = concat_task_build.direct_tasks;
  stats.concat.layout_tasks = concat_task_build.layout_tasks;
  stats.concat.strided_tasks = concat_task_build.strided_tasks;
  stats.concat.read_bytes = concat_task_build.source_bytes;
  stats.concat.direct_bytes = concat_task_build.direct_bytes;
  stats.concat.layout_bytes = concat_task_build.layout_bytes;
  stats.concat.strided_bytes = concat_task_build.strided_bytes;

  const auto total_start = std::chrono::steady_clock::now();
  if (tensor_task_build.tasks.empty() && concat_task_build.tasks.empty()) {
    return stats;
  }
  if (pinned_pool == nullptr) {
    return absl::InvalidArgumentError("local mapped source-ordered streaming requires a pinned pool");
  }

  std::vector<SourceOrderedTask> tasks;
  tasks.reserve(tensor_task_build.tasks.size() + concat_task_build.tasks.size());
  for (size_t idx = 0; idx < tensor_task_build.tasks.size(); ++idx) {
    const auto& task = tensor_task_build.tasks[idx];
    tasks.push_back(
        SourceOrderedTask{
            .kind = SourceOrderedTask::Kind::kTensor,
            .index = idx,
            .source_offset = task.source_offset,
            .read_bytes = task.read_bytes,
        });
  }
  for (size_t idx = 0; idx < concat_task_build.tasks.size(); ++idx) {
    const auto& task = concat_task_build.tasks[idx];
    tasks.push_back(
        SourceOrderedTask{
            .kind = SourceOrderedTask::Kind::kConcat,
            .index = idx,
            .source_offset = task.source_offset,
            .read_bytes = task.total_bytes(),
        });
  }
  std::stable_sort(tasks.begin(), tasks.end(), [](const SourceOrderedTask& lhs, const SourceOrderedTask& rhs) {
    if (lhs.source_offset != rhs.source_offset) {
      return lhs.source_offset < rhs.source_offset;
    }
    if (lhs.kind != rhs.kind) {
      return static_cast<std::uint8_t>(lhs.kind) < static_cast<std::uint8_t>(rhs.kind);
    }
    return lhs.index < rhs.index;
  });

  const size_t concurrency = std::max<size_t>(1, streaming_buffer_chunks);
  auto session_spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
      /*num_chunks=*/concurrency, host_buffer_bytes, pinned_pool);
  TC_RETURN_IF_ERROR(session_spb->initialize(
      pinned_timeout, absl::StrCat("local_mapped_source_ordered artifact_id=", participant.artifact_id)));

  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participant.device_id));
  cudaStream_t stream = nullptr;
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&stream, cudaStreamNonBlocking));
  absl::Cleanup destroy_stream = [&stream]() {
    if (stream != nullptr) {
      (void)tensorcast::cuda::stream_destroy(stream);
    }
  };

  absl::Mutex status_mu;
  absl::Status first_status;
  std::atomic<bool> stop{false};
  std::atomic<size_t> next_task{0};
  std::atomic<uint64_t> producer_get_free_ns{0};
  std::atomic<uint64_t> producer_read_ns{0};
  std::atomic<uint64_t> producer_mark_ready_ns{0};
  std::atomic<uint64_t> producer_tasks{0};
  std::atomic<uint64_t> producer_bytes{0};
  auto producers_done = std::make_shared<absl::BlockingCounter>(static_cast<int>(concurrency));
  auto producers_remaining = std::make_shared<std::atomic<size_t>>(concurrency);

  auto record_failure = [&](absl::Status status) {
    if (status.ok()) {
      return;
    }
    {
      absl::MutexLock lock(&status_mu);
      if (first_status.ok()) {
        first_status = std::move(status);
      }
    }
    stop.store(true, std::memory_order_release);
  };

  auto executor = whole_source_load_runtime().blocking_executor();
  for (size_t worker = 0; worker < concurrency; ++worker) {
    executor->add([&, producers_done, producers_remaining]() {
      while (!stop.load(std::memory_order_acquire)) {
        const size_t task_index = next_task.fetch_add(1, std::memory_order_acq_rel);
        if (task_index >= tasks.size()) {
          break;
        }
        const auto& task = tasks[task_index];
        const auto get_free_start = std::chrono::steady_clock::now();
        auto slot_or = session_spb->get_free_chunk();
        producer_get_free_ns.fetch_add(elapsed_ns_since(get_free_start), std::memory_order_relaxed);
        if (!slot_or.ok()) {
          if (!stop.load(std::memory_order_acquire)) {
            record_failure(slot_or.status());
          }
          break;
        }
        const int slot_id = *slot_or;
        if (stop.load(std::memory_order_acquire)) {
          (void)session_spb->abort_producer_slot(slot_id);
          break;
        }
        char* host_ptr = session_spb->get_chunk_ptr(slot_id);
        if (host_ptr == nullptr) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(absl::InternalError("local mapped source-ordered streaming got a null pinned chunk"));
          break;
        }
        if (task.read_bytes > std::numeric_limits<size_t>::max()) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(absl::OutOfRangeError("local mapped source-ordered task exceeds host size_t limit"));
          break;
        }
        const size_t bytes = static_cast<size_t>(task.read_bytes);
        const auto read_start = std::chrono::steady_clock::now();
        auto got_or = source.read_at(task.source_offset, host_ptr, bytes);
        producer_read_ns.fetch_add(elapsed_ns_since(read_start), std::memory_order_relaxed);
        if (!got_or.ok()) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(got_or.status());
          break;
        }
        if (*got_or != bytes) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(
              absl::OutOfRangeError(
                  absl::StrCat("short read in local mapped source-ordered streaming: got=", *got_or, " want=", bytes)));
          break;
        }
        const auto mark_ready_start = std::chrono::steady_clock::now();
        const absl::Status ready_status = session_spb->mark_chunk_ready(slot_id, task_index, bytes);
        producer_mark_ready_ns.fetch_add(elapsed_ns_since(mark_ready_start), std::memory_order_relaxed);
        if (!ready_status.ok()) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(ready_status);
          break;
        }
        producer_tasks.fetch_add(1, std::memory_order_relaxed);
        producer_bytes.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed);
      }
      if (producers_remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
        session_spb->signal_production_complete();
      }
      producers_done->DecrementCount();
    });
  }

  uint64_t consumer_get_ready_ns = 0;
  uint64_t consumer_copy_ns = 0;
  uint64_t consumer_return_ns = 0;
  uint64_t consumer_tasks = 0;
  uint64_t consumer_tensor_tasks = 0;
  uint64_t consumer_concat_tasks = 0;
  uint64_t consumer_bytes = 0;
  uint64_t tensor_copy_ns = 0;
  uint64_t concat_copy_ns = 0;
  while (true) {
    const auto get_ready_start = std::chrono::steady_clock::now();
    auto chunk_or = session_spb->get_ready_chunk();
    consumer_get_ready_ns += elapsed_ns_since(get_ready_start);
    if (!chunk_or.ok()) {
      if (!absl::IsOutOfRange(chunk_or.status()) && !absl::IsUnavailable(chunk_or.status())) {
        record_failure(chunk_or.status());
        session_spb->signal_production_complete();
      }
      break;
    }
    auto chunk = *chunk_or;
    absl::Cleanup return_chunk = [&]() {
      const auto return_start = std::chrono::steady_clock::now();
      const absl::Status return_status = session_spb->return_chunk(chunk.slot_id);
      consumer_return_ns += elapsed_ns_since(return_start);
      if (!return_status.ok()) {
        record_failure(return_status);
      }
    };

    if (chunk.global_chunk_id >= tasks.size()) {
      record_failure(absl::InternalError("local mapped source-ordered chunk id out of range"));
      continue;
    }
    const auto& task = tasks[chunk.global_chunk_id];
    if (chunk.bytes_in_chunk != task.read_bytes) {
      record_failure(absl::InternalError("local mapped source-ordered chunk size mismatch"));
      continue;
    }
    if (stop.load(std::memory_order_acquire)) {
      continue;
    }

    const auto copy_start = std::chrono::steady_clock::now();
    absl::Status copy_status;
    if (task.kind == SourceOrderedTask::Kind::kTensor) {
      if (task.index >= tensor_task_build.tasks.size()) {
        copy_status = absl::InternalError("local mapped source-ordered tensor task index out of range");
      } else {
        copy_status =
            execute_local_mapped_tensor_task_copy(tensor_task_build.tasks[task.index], chunk.data_ptr, stream);
      }
      const uint64_t elapsed = elapsed_ns_since(copy_start);
      tensor_copy_ns += elapsed;
      consumer_copy_ns += elapsed;
      consumer_tensor_tasks += 1;
    } else {
      if (task.index >= concat_task_build.tasks.size()) {
        copy_status = absl::InternalError("local mapped source-ordered concat task index out of range");
      } else {
        copy_status = execute_local_mapped_concat_task_copy(
            participant, concat_task_build.tasks[task.index], chunk.data_ptr, sink, stream);
      }
      const uint64_t elapsed = elapsed_ns_since(copy_start);
      concat_copy_ns += elapsed;
      consumer_copy_ns += elapsed;
      consumer_concat_tasks += 1;
    }
    consumer_tasks += 1;
    consumer_bytes += static_cast<uint64_t>(chunk.bytes_in_chunk);
    if (!copy_status.ok()) {
      record_failure(std::move(copy_status));
    }
  }

  producers_done->Wait();
  {
    absl::MutexLock lock(&status_mu);
    if (!first_status.ok()) {
      return first_status;
    }
  }

  stats.tasks = tasks.size();
  stats.exec_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  stats.tensor_copy_sec = seconds_from_ns(tensor_copy_ns);
  stats.concat_copy_sec = seconds_from_ns(concat_copy_ns);
  stats.tensor.exec_sec = stats.tensor_copy_sec;
  stats.concat.exec_sec = stats.concat_copy_sec;
  LOG(INFO) << "local_mapped_source_ordered_streaming_summary"
            << " artifact_id=" << participant.artifact_id << " tasks=" << stats.tasks
            << " tensor_tasks=" << tensor_task_build.tasks.size() << " concat_tasks=" << concat_task_build.tasks.size()
            << " tensor_read_bytes=" << stats.tensor.read_bytes << " tensor_dst_bytes=" << stats.tensor.dst_bytes
            << " concat_read_bytes=" << stats.concat.read_bytes << " concat_direct_bytes=" << stats.concat.direct_bytes
            << " concat_strided_bytes=" << stats.concat.strided_bytes << " chunk_bytes=" << host_buffer_bytes
            << " streaming_buffer_chunks=" << concurrency
            << " producer_tasks=" << producer_tasks.load(std::memory_order_relaxed)
            << " producer_bytes=" << producer_bytes.load(std::memory_order_relaxed)
            << " producer_get_free_sec=" << seconds_from_ns(producer_get_free_ns.load(std::memory_order_relaxed))
            << " producer_read_sec=" << seconds_from_ns(producer_read_ns.load(std::memory_order_relaxed))
            << " producer_mark_ready_sec=" << seconds_from_ns(producer_mark_ready_ns.load(std::memory_order_relaxed))
            << " consumer_tasks=" << consumer_tasks << " consumer_tensor_tasks=" << consumer_tensor_tasks
            << " consumer_concat_tasks=" << consumer_concat_tasks << " consumer_bytes=" << consumer_bytes
            << " consumer_get_ready_sec=" << seconds_from_ns(consumer_get_ready_ns)
            << " consumer_copy_sec=" << seconds_from_ns(consumer_copy_ns)
            << " tensor_copy_sec=" << stats.tensor_copy_sec << " concat_copy_sec=" << stats.concat_copy_sec
            << " consumer_return_sec=" << seconds_from_ns(consumer_return_ns) << " exec=" << stats.exec_sec << "s";
  return stats;
}

absl::StatusOr<LocalMappedTargetExecutionResult> execute_local_mapped_target(
    ParsedMappedParticipant participant,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options) {
  if (pinned_pool == nullptr) {
    return absl::InvalidArgumentError("local mapped target load requires a pinned pool");
  }
  if (participant.disk_context == nullptr || participant.disk_context->safetensors_segments().empty()) {
    return absl::InvalidArgumentError("local mapped target load requires safetensors segments");
  }
  if (!enable_mapped_tensor_job_fast_path(options.strategy_config) ||
      !enable_mapped_concat_jobs(options.strategy_config) || !enable_mapped_concat_execution(options.strategy_config)) {
    return absl::FailedPreconditionError("local mapped target tensor-aware concat executor is disabled");
  }

  const auto total_start = std::chrono::steady_clock::now();
  const size_t chunk_bytes = pinned_pool->slice_bytes();

  std::unique_ptr<loader::SeekableSource> source_owner;
  TC_ASSIGN_OR_RETURN(
      source_owner,
      make_local_mapped_safetensors_source(participant.disk_context->safetensors_segments(), options.strategy_config));
  loader::SeekableSource& source = *source_owner;

  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participant.device_id));

  std::vector<loader::TargetStorage> target_storages;
  target_storages.reserve(participant.target_layout.storages.size());
  for (const auto& storage : participant.target_layout.storages) {
    target_storages.push_back(loader::TargetStorage{storage.base_ptr, storage.length});
  }
  loader::TargetLayoutGpuSink target_sink(
      loader::TargetLayoutGpuSink::Options{
          .storages = std::move(target_storages),
          .chunk_size = chunk_bytes,
          .device_id = participant.device_id,
      });

  std::vector<ParsedMappedParticipant> participants = {participant};
  const auto tensor_build_start = std::chrono::steady_clock::now();
  auto tensor_job_build_or = build_mapped_tensor_jobs(participants);
  if (!tensor_job_build_or.ok()) {
    return tensor_job_build_or.status();
  }
  auto tensor_job_build = std::move(*tensor_job_build_or);
  const double tensor_build_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - tensor_build_start).count();

  const auto concat_build_start = std::chrono::steady_clock::now();
  auto concat_job_build_or = build_mapped_concat_jobs(participants, options.strategy_config);
  if (!concat_job_build_or.ok()) {
    return concat_job_build_or.status();
  }
  auto concat_job_build = std::move(*concat_job_build_or);
  const double concat_build_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - concat_build_start).count();

  std::vector<std::vector<ByteRange>> initially_handled_dst_ranges_by_rank =
      tensor_job_build.handled_dst_ranges_by_rank;
  if (initially_handled_dst_ranges_by_rank.size() < concat_job_build.handled_dst_ranges_by_rank.size()) {
    initially_handled_dst_ranges_by_rank.resize(concat_job_build.handled_dst_ranges_by_rank.size());
  }
  for (size_t rank = 0; rank < concat_job_build.handled_dst_ranges_by_rank.size(); ++rank) {
    initially_handled_dst_ranges_by_rank[rank].insert(
        initially_handled_dst_ranges_by_rank[rank].end(),
        concat_job_build.handled_dst_ranges_by_rank[rank].begin(),
        concat_job_build.handled_dst_ranges_by_rank[rank].end());
    merge_byte_ranges(&initially_handled_dst_ranges_by_rank[rank]);
  }

  MappedTensorJobBuildResult partial_tensor_job_build;
  {
    absl::Span<const ByteRange> initially_handled = initially_handled_dst_ranges_by_rank.empty()
        ? absl::Span<const ByteRange>()
        : absl::MakeSpan(initially_handled_dst_ranges_by_rank.front());
    auto partial_tensor_job_build_or = build_local_mapped_partial_tensor_jobs(participant, initially_handled);
    if (!partial_tensor_job_build_or.ok()) {
      return partial_tensor_job_build_or.status();
    }
    partial_tensor_job_build = std::move(*partial_tensor_job_build_or);
  }

  std::vector<MappedTensorJobRuntime> local_tensor_jobs = tensor_job_build.jobs;
  local_tensor_jobs.insert(
      local_tensor_jobs.end(),
      std::make_move_iterator(partial_tensor_job_build.jobs.begin()),
      std::make_move_iterator(partial_tensor_job_build.jobs.end()));
  if (local_tensor_jobs.empty() && concat_job_build.jobs.empty()) {
    return absl::FailedPreconditionError("local mapped target found no tensor-aware jobs");
  }

  std::vector<std::vector<ByteRange>> handled_dst_ranges_by_rank = initially_handled_dst_ranges_by_rank;
  if (handled_dst_ranges_by_rank.size() < partial_tensor_job_build.handled_dst_ranges_by_rank.size()) {
    handled_dst_ranges_by_rank.resize(partial_tensor_job_build.handled_dst_ranges_by_rank.size());
  }
  for (size_t rank = 0; rank < partial_tensor_job_build.handled_dst_ranges_by_rank.size(); ++rank) {
    handled_dst_ranges_by_rank[rank].insert(
        handled_dst_ranges_by_rank[rank].end(),
        partial_tensor_job_build.handled_dst_ranges_by_rank[rank].begin(),
        partial_tensor_job_build.handled_dst_ranges_by_rank[rank].end());
    merge_byte_ranges(&handled_dst_ranges_by_rank[rank]);
  }
  const auto lane_ranges = data_ranges_from_lane_map(participant.collective_lane_map);
  const uint64_t lane_range_bytes = byte_ranges_covered_bytes(absl::MakeSpan(lane_ranges));
  const uint64_t handled_range_bytes = handled_dst_ranges_by_rank.empty()
      ? 0
      : byte_ranges_covered_bytes(absl::MakeSpan(handled_dst_ranges_by_rank.front()));
  const uint64_t handled_overlap_bytes = handled_dst_ranges_by_rank.empty()
      ? 0
      : byte_ranges_overlap_bytes(absl::MakeSpan(lane_ranges), absl::MakeSpan(handled_dst_ranges_by_rank.front()));

  auto segment_refs_or = build_mapped_segment_refs(participants, handled_dst_ranges_by_rank);
  if (!segment_refs_or.ok()) {
    return segment_refs_or.status();
  }
  loader::ByteRangeMap residual_data_map =
      build_data_map_from_segment_refs(absl::MakeSpan(*segment_refs_or), participant.collective_lane_map.total_bytes);
  const uint64_t residual_bytes = mapped_segment_ref_covered_bytes(absl::MakeSpan(*segment_refs_or));
  const uint64_t planned_handled_bytes = tensor_job_build.handled_root_dst_bytes +
      partial_tensor_job_build.handled_root_dst_bytes + concat_job_build.handled_root_dst_bytes;
  if (residual_bytes > 0 && !options.strategy_config.allow_mixed_execution) {
    return absl::FailedPreconditionError(
        "local mapped target produced generic residual but mixed execution is disabled");
  }

  TC_RETURN_IF_ERROR(
      validate_local_mapped_tensor_jobs_admission(local_tensor_jobs, participant.device_id, chunk_bytes));
  TC_RETURN_IF_ERROR(validate_local_mapped_concat_jobs_admission(concat_job_build.jobs, chunk_bytes));

  const size_t streaming_buffer_chunks = std::max<size_t>(1, options.streaming_buffer_chunks);
  LocalMappedTensorExecutionStats tensor_stats;
  LocalMappedConcatExecutionStats concat_stats;
  bool source_ordered_local_mapped = false;
  double source_ordered_exec_sec = 0.0;
  if (allow_source_ordered_for_mapped(options.strategy_config) && !local_tensor_jobs.empty() &&
      !concat_job_build.jobs.empty()) {
    const auto tensor_task_build_start = std::chrono::steady_clock::now();
    auto tensor_task_build_or = build_local_mapped_tensor_tasks(local_tensor_jobs, participant.device_id, chunk_bytes);
    if (!tensor_task_build_or.ok()) {
      return tensor_task_build_or.status();
    }
    auto tensor_task_build = std::move(*tensor_task_build_or);
    const double tensor_task_build_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tensor_task_build_start).count();

    const auto concat_task_build_start = std::chrono::steady_clock::now();
    auto concat_task_build_or = build_local_mapped_concat_tasks(participant, concat_job_build.jobs, chunk_bytes);
    if (!concat_task_build_or.ok()) {
      return concat_task_build_or.status();
    }
    auto concat_task_build = std::move(*concat_task_build_or);
    const double concat_task_build_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - concat_task_build_start).count();

    auto combined_stats_or = execute_local_mapped_source_ordered_tasks(
        participant,
        tensor_task_build,
        concat_task_build,
        source,
        pinned_pool,
        pinned_timeout,
        chunk_bytes,
        streaming_buffer_chunks,
        target_sink);
    if (!combined_stats_or.ok()) {
      return combined_stats_or.status();
    }
    auto combined_stats = *combined_stats_or;
    tensor_stats = combined_stats.tensor;
    concat_stats = combined_stats.concat;
    source_ordered_local_mapped = true;
    source_ordered_exec_sec = combined_stats.exec_sec;
    LOG(INFO) << "local_mapped_source_ordered_task_build"
              << " artifact_id=" << participant.artifact_id << " tensor_task_build_sec=" << tensor_task_build_sec
              << " concat_task_build_sec=" << concat_task_build_sec
              << " tensor_tasks=" << tensor_task_build.tasks.size()
              << " concat_tasks=" << concat_task_build.tasks.size();
  } else {
    auto tensor_stats_or = execute_local_mapped_tensor_jobs(
        local_tensor_jobs,
        participant.device_id,
        source,
        pinned_pool,
        pinned_timeout,
        chunk_bytes,
        streaming_buffer_chunks,
        participant.artifact_id);
    if (!tensor_stats_or.ok()) {
      return tensor_stats_or.status();
    }
    tensor_stats = *tensor_stats_or;

    if (!concat_job_build.jobs.empty()) {
      auto concat_stats_or = execute_local_mapped_concat_jobs_streaming(
          participant,
          concat_job_build.jobs,
          source,
          pinned_pool,
          pinned_timeout,
          chunk_bytes,
          streaming_buffer_chunks,
          target_sink);
      if (!concat_stats_or.ok()) {
        return concat_stats_or.status();
      }
      concat_stats = *concat_stats_or;
    }
  }

  const uint64_t unique_source_bytes = tensor_stats.read_bytes + concat_job_build.handled_source_bytes;
  const uint64_t concat_peak_temporary_bytes = chunk_bytes * static_cast<uint64_t>(streaming_buffer_chunks);

  const auto total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  LOG(INFO) << "local_mapped_target timings"
            << " artifact_id=" << participant.artifact_id << " tensor_jobs=" << local_tensor_jobs.size()
            << " base_tensor_jobs=" << tensor_job_build.jobs.size()
            << " partial_tensor_jobs=" << partial_tensor_job_build.jobs.size()
            << " tensor_replicated_jobs=" << tensor_stats.replicated_jobs
            << " tensor_dim0_jobs=" << tensor_stats.dim0_jobs << " tensor_dim1_jobs=" << tensor_stats.dim1_jobs
            << " tensor_rect2d_jobs=" << tensor_stats.rect2d_jobs << " tensor_read_bytes=" << tensor_stats.read_bytes
            << " tensor_dst_bytes=" << tensor_stats.dst_bytes << " tensor_job_build=" << tensor_build_sec << "s"
            << " tensor_job_exec=" << tensor_stats.exec_sec << "s"
            << " tensor_streaming_tasks=" << tensor_stats.tasks << " concat_jobs=" << concat_job_build.jobs.size()
            << " concat_job_source_bytes=" << concat_job_build.handled_source_bytes
            << " concat_job_root_dst_bytes=" << concat_job_build.handled_root_dst_bytes
            << " concat_job_build=" << concat_build_sec << "s"
            << " concat_job_exec=" << concat_stats.exec_sec << "s"
            << " source_ordered_local_mapped=" << (source_ordered_local_mapped ? "true" : "false")
            << " source_ordered_exec=" << source_ordered_exec_sec << "s"
            << " concat_streaming_tasks=" << concat_stats.tasks << " concat_direct_tasks=" << concat_stats.direct_tasks
            << " concat_layout_tasks=" << concat_stats.layout_tasks
            << " concat_strided_tasks=" << concat_stats.strided_tasks << " lane_range_bytes=" << lane_range_bytes
            << " handled_range_bytes=" << handled_range_bytes << " handled_overlap_bytes=" << handled_overlap_bytes
            << " residual_segments=" << segment_refs_or->size() << " residual_bytes=" << residual_bytes
            << " total=" << total_sec << "s";
  return LocalMappedTargetExecutionResult{
      .metrics =
          runtime::ingestion::strategy::CollectiveExecutionMetrics{
              .unique_source_bytes = unique_source_bytes,
              .peer_transfer_bytes = 0,
              .peak_temporary_bytes = std::max<uint64_t>(chunk_bytes, concat_peak_temporary_bytes),
              .batch_count = static_cast<uint64_t>(local_tensor_jobs.size() + concat_stats.tasks),
              .dedup_saving_bytes = 0,
          },
      .residual_data_map = std::move(residual_data_map),
      .handled_bytes = planned_handled_bytes,
  };
}

CollectiveDiskLoadResult wait_for_group_and_maybe_execute(
    const CollectiveDiskLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout) {
  auto parsed = ParsedParticipant{
      .replica_key = request.replica_key,
      .rank = static_cast<int>(request.group.rank),
      .device_id = request.device_id,
      .gpu_ptr = request.gpu_ptr,
      .gpu_allocation = request.gpu_allocation,
      .disk_context = request.disk_context,
      .representation_work_plan = request.representation_work_plan,
  };
  std::shared_ptr<GroupState> state;
  {
    absl::MutexLock lock(&g_group_mu);
    auto& slot = g_groups[request.group.group_id];
    if (slot == nullptr) {
      slot = std::make_shared<GroupState>(request.group.world_size);
    }
    state = slot;
  }

  bool leader = false;
  bool erase_empty_group = false;
  bool timed_out = false;
  {
    absl::MutexLock lock(&state->mu);
    if (request.group.rank >= state->world_size) {
      return {.handled = false, .status = absl::InvalidArgumentError("collective rank out of range")};
    }
    auto& slot = state->participants[request.group.rank];
    if (!slot.has_value()) {
      slot = parsed;
      state->joined += 1;
      LOG(INFO) << "collective_disk_load join group_id=" << request.group.group_id << " rank=" << request.group.rank
                << " joined=" << state->joined << "/" << state->world_size;
    }
    if (state->joined == state->world_size) {
      state->launching = true;
      leader = true;
      state->cv.SignalAll();
      LOG(INFO) << "collective_disk_load launching group_id=" << request.group.group_id
                << " world_size=" << state->world_size;
    } else {
      const absl::Time deadline =
          absl::Now() + absl::Milliseconds(collective_group_assemble_timeout(request.strategy_config).count());
      while (!state->launching && !state->complete && absl::Now() < deadline) {
        state->cv.WaitWithDeadline(&state->mu, deadline);
      }
      if (!state->launching && !state->complete) {
        if (slot.has_value()) {
          slot.reset();
          state->joined -= 1;
        }
        timed_out = true;
        erase_empty_group = state->joined == 0;
        LOG(INFO) << "collective_disk_load timeout group_id=" << request.group.group_id
                  << " rank=" << request.group.rank << " remaining=" << state->joined;
      }
    }
  }
  if (erase_empty_group) {
    absl::MutexLock group_lock(&g_group_mu);
    g_groups.erase(request.group.group_id);
    return {.handled = false, .status = absl::OkStatus()};
  }
  if (timed_out) {
    return {.handled = false, .status = absl::OkStatus()};
  }

  if (leader) {
    std::vector<ParsedParticipant> participants;
    participants.reserve(state->world_size);
    {
      absl::MutexLock lock(&state->mu);
      for (const auto& participant : state->participants) {
        if (!participant.has_value()) {
          state->status = absl::FailedPreconditionError("collective disk load group is incomplete");
          state->complete = true;
          state->cv.SignalAll();
          return {.handled = true, .status = state->status};
        }
        participants.push_back(*participant);
      }
    }
    std::sort(participants.begin(), participants.end(), [](const ParsedParticipant& a, const ParsedParticipant& b) {
      return a.rank < b.rank;
    });
    auto exec_metrics_or = execute_group_collective(participants, pinned_pool, pinned_timeout, request.strategy_config);
    const absl::Status exec_status = exec_metrics_or.ok() ? absl::OkStatus() : exec_metrics_or.status();
    LOG(INFO) << "collective_disk_load finished group_id=" << request.group.group_id << " status=" << exec_status;
    {
      absl::MutexLock lock(&state->mu);
      state->status = exec_status;
      state->metrics =
          exec_metrics_or.ok() ? *exec_metrics_or : runtime::ingestion::strategy::CollectiveExecutionMetrics{};
      state->complete = true;
      state->cv.SignalAll();
    }
    {
      absl::MutexLock lock(&g_group_mu);
      g_groups.erase(request.group.group_id);
    }
    return {
        .handled = true,
        .status = exec_status,
        .metrics = exec_metrics_or.ok() ? *exec_metrics_or : runtime::ingestion::strategy::CollectiveExecutionMetrics{},
    };
  }

  {
    absl::MutexLock lock(&state->mu);
    while (!state->complete) {
      state->cv.Wait(&state->mu);
    }
    return {.handled = true, .status = state->status, .metrics = state->metrics};
  }
}

CollectiveMappedTargetLoadResult wait_for_mapped_group_and_maybe_execute(
    const CollectiveMappedTargetLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options) {
  auto storage_spans_or = build_target_storage_spans(request.target_layout);
  if (!storage_spans_or.ok()) {
    return {
        .handled = false,
        .status = storage_spans_or.status(),
        .skip_reason = "invalid_target_storage_layout",
    };
  }
  auto parsed = ParsedMappedParticipant{
      .artifact_id = request.artifact_id,
      .rank = static_cast<int>(request.group.rank),
      .device_id = request.device_id,
      .disk_context = request.disk_context,
      .work_plan = request.representation_work_plan,
      .collective_lane_map = request.collective_lane_map,
      .target_layout = request.target_layout,
      .storage_spans = std::move(*storage_spans_or),
  };
  std::shared_ptr<MappedGroupState> state;
  {
    absl::MutexLock lock(&g_mapped_group_mu);
    auto& slot = g_mapped_groups[request.group.group_id];
    if (slot == nullptr) {
      slot = std::make_shared<MappedGroupState>(request.group.world_size);
    }
    state = slot;
  }

  bool leader = false;
  bool erase_empty_group = false;
  bool timed_out = false;
  {
    absl::MutexLock lock(&state->mu);
    if (request.group.rank >= state->world_size) {
      return {
          .handled = false,
          .status = absl::InvalidArgumentError("mapped collective rank out of range"),
          .skip_reason = "collective_rank_out_of_range",
      };
    }
    auto& slot = state->participants[request.group.rank];
    if (!slot.has_value()) {
      slot = std::move(parsed);
      state->joined += 1;
      LOG(INFO) << "collective_mapped_target join group_id=" << request.group.group_id << " rank=" << request.group.rank
                << " joined=" << state->joined << "/" << state->world_size;
    }
    if (state->joined == state->world_size) {
      state->launching = true;
      leader = true;
      state->cv.SignalAll();
      LOG(INFO) << "collective_mapped_target launching group_id=" << request.group.group_id
                << " world_size=" << state->world_size;
    } else {
      const absl::Time deadline =
          absl::Now() + absl::Milliseconds(collective_group_assemble_timeout(options.strategy_config).count());
      while (!state->launching && !state->complete && absl::Now() < deadline) {
        state->cv.WaitWithDeadline(&state->mu, deadline);
      }
      if (!state->launching && !state->complete) {
        if (slot.has_value()) {
          slot.reset();
          state->joined -= 1;
        }
        timed_out = true;
        erase_empty_group = state->joined == 0;
        LOG(INFO) << "collective_mapped_target timeout group_id=" << request.group.group_id
                  << " rank=" << request.group.rank << " remaining=" << state->joined;
      }
    }
  }
  if (erase_empty_group) {
    absl::MutexLock group_lock(&g_mapped_group_mu);
    g_mapped_groups.erase(request.group.group_id);
    return {.handled = false, .status = absl::OkStatus(), .skip_reason = "group_assemble_timeout"};
  }
  if (timed_out) {
    return {.handled = false, .status = absl::OkStatus(), .skip_reason = "group_assemble_timeout"};
  }

  if (leader) {
    std::vector<ParsedMappedParticipant> participants;
    participants.reserve(state->world_size);
    {
      absl::MutexLock lock(&state->mu);
      for (const auto& participant : state->participants) {
        if (!participant.has_value()) {
          state->status = absl::FailedPreconditionError("mapped collective group is incomplete");
          state->complete = true;
          state->cv.SignalAll();
          return {.handled = true, .status = state->status};
        }
        participants.push_back(*participant);
      }
    }
    std::sort(
        participants.begin(),
        participants.end(),
        [](const ParsedMappedParticipant& a, const ParsedMappedParticipant& b) { return a.rank < b.rank; });
    auto exec_metrics_or = execute_group_collective_mapped(participants, pinned_pool, pinned_timeout, options);
    const absl::Status exec_status = exec_metrics_or.ok() ? absl::OkStatus() : exec_metrics_or.status();
    LOG(INFO) << "collective_mapped_target finished group_id=" << request.group.group_id << " status=" << exec_status;
    {
      absl::MutexLock lock(&state->mu);
      state->status = exec_status;
      state->metrics =
          exec_metrics_or.ok() ? *exec_metrics_or : runtime::ingestion::strategy::CollectiveExecutionMetrics{};
      state->complete = true;
      state->cv.SignalAll();
    }
    {
      absl::MutexLock lock(&g_mapped_group_mu);
      g_mapped_groups.erase(request.group.group_id);
    }
    return {
        .handled = true,
        .status = exec_status,
        .metrics = exec_metrics_or.ok() ? *exec_metrics_or : runtime::ingestion::strategy::CollectiveExecutionMetrics{},
    };
  }

  {
    absl::MutexLock lock(&state->mu);
    while (!state->complete) {
      state->cv.Wait(&state->mu);
    }
    return {.handled = true, .status = state->status, .metrics = state->metrics};
  }
}

} // namespace

absl::StatusOr<LocalBatchedPlanSummary> summarize_local_batched_disk_load(
    const materialization::contracts::RepresentationWorkPlan& representation_work_plan,
    const StrategyConfig& strategy_config) {
  if (!enable_local_batched_disk_load(strategy_config)) {
    return LocalBatchedPlanSummary{
        .eligible = false,
        .reason = "strategy_disabled",
    };
  }
  ParsedParticipant participant{
      .replica_key = loading::ReplicaKey{.artifact_id = "local-batched-summary"},
      .rank = 0,
      .device_id = 0,
      .gpu_ptr = nullptr,
      .gpu_allocation = nullptr,
      .disk_context = nullptr,
      .representation_work_plan = representation_work_plan,
  };
  auto jobs_or = build_tensor_jobs(std::vector<ParsedParticipant>{participant});
  if (!jobs_or.ok()) {
    if (absl::IsUnimplemented(jobs_or.status())) {
      return LocalBatchedPlanSummary{
          .eligible = false,
          .reason = "unsupported_tensor_jobs",
      };
    }
    return jobs_or.status();
  }
  auto plan_or = build_local_batched_execution_plan(*jobs_or);
  if (!plan_or.ok()) {
    return plan_or.status();
  }
  return plan_or->summary;
}

LocalBatchedDiskLoadResult try_local_batched_disk_load(
    const LocalBatchedDiskLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout) {
  return try_local_batched_disk_load_impl(request, pinned_pool, pinned_timeout);
}

CollectiveDiskLoadResult try_collective_disk_load(
    const CollectiveDiskLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout) {
  if (!enable_collective_owner_file_strategy(request.strategy_config)) {
    return {.handled = false, .status = absl::OkStatus(), .skip_reason = "strategy_disabled"};
  }
  if (request.group.world_size <= 1 || request.gpu_ptr == nullptr || request.device_id < 0 ||
      request.disk_context == nullptr || request.representation_work_plan.items.empty() ||
      request.gpu_allocation == nullptr) {
    LOG(INFO) << "collective_disk_load skipped group_id=" << request.group.group_id << " reason=request_incomplete"
              << " world_size=" << request.group.world_size << " gpu_ptr=" << (request.gpu_ptr != nullptr)
              << " device_id=" << request.device_id << " disk_context=" << (request.disk_context != nullptr)
              << " work_items=" << request.representation_work_plan.items.size()
              << " gpu_allocation=" << (request.gpu_allocation != nullptr);
    return {.handled = false, .status = absl::OkStatus(), .skip_reason = "request_incomplete"};
  }
  if (!request.disk_context->is_safetensors()) {
    LOG(INFO) << "collective_disk_load skipped group_id=" << request.group.group_id << " reason=non_safetensors_source";
    return {.handled = false, .status = absl::OkStatus(), .skip_reason = "non_safetensors_source"};
  }
  return wait_for_group_and_maybe_execute(request, pinned_pool, pinned_timeout);
}

CollectiveMappedTargetLoadResult try_collective_mapped_target_load(
    const CollectiveMappedTargetLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options) {
  if (request.group.world_size <= 1 || request.device_id < 0 || request.disk_context == nullptr ||
      request.target_layout.storages.empty() ||
      (request.representation_work_plan.items.empty() && request.collective_lane_map.total_bytes == 0) ||
      request.artifact_id.empty()) {
    LOG(INFO) << "collective_mapped_target skipped group_id=" << request.group.group_id << " reason=request_incomplete"
              << " world_size=" << request.group.world_size << " device_id=" << request.device_id
              << " disk_context=" << (request.disk_context != nullptr)
              << " storages=" << request.target_layout.storages.size()
              << " map_bytes=" << request.collective_lane_map.total_bytes
              << " artifact_id=" << (!request.artifact_id.empty());
    return {.handled = false, .status = absl::OkStatus(), .skip_reason = "request_incomplete"};
  }
  if (!request.disk_context->is_safetensors()) {
    LOG(INFO) << "collective_mapped_target skipped group_id=" << request.group.group_id
              << " reason=non_safetensors_source";
    return {.handled = false, .status = absl::OkStatus(), .skip_reason = "non_safetensors_source"};
  }
  return wait_for_mapped_group_and_maybe_execute(request, pinned_pool, pinned_timeout, options);
}

LocalMappedTargetLoadResult try_local_mapped_target_load(
    const LocalMappedTargetLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options) {
  if (!enable_mapped_tensor_job_fast_path(request.strategy_config)) {
    return {.handled = false, .status = absl::OkStatus(), .skip_reason = "strategy_disabled"};
  }
  if (request.device_id < 0 || request.disk_context == nullptr || request.target_layout.storages.empty() ||
      request.representation_work_plan.items.empty() || request.data_lane_map.total_bytes == 0 ||
      request.artifact_id.empty()) {
    return {.handled = false, .status = absl::OkStatus(), .skip_reason = "request_incomplete"};
  }
  if (!request.disk_context->is_safetensors()) {
    return {.handled = false, .status = absl::OkStatus(), .skip_reason = "non_safetensors_source"};
  }
  auto storage_spans_or = build_target_storage_spans(request.target_layout);
  if (!storage_spans_or.ok()) {
    return {.handled = false, .status = storage_spans_or.status(), .skip_reason = "invalid_target_storage_layout"};
  }
  ParsedMappedParticipant participant{
      .artifact_id = request.artifact_id,
      .rank = 0,
      .device_id = request.device_id,
      .disk_context = request.disk_context,
      .work_plan = request.representation_work_plan,
      .collective_lane_map = request.data_lane_map,
      .target_layout = request.target_layout,
      .storage_spans = std::move(*storage_spans_or),
  };
  auto result_or = execute_local_mapped_target(std::move(participant), pinned_pool, pinned_timeout, options);
  if (!result_or.ok()) {
    return {.handled = true, .status = result_or.status()};
  }
  return {
      .handled = true,
      .status = absl::OkStatus(),
      .metrics = result_or->metrics,
      .residual_data_map = std::move(result_or->residual_data_map),
      .handled_bytes = result_or->handled_bytes,
  };
}

absl::Status warm_collective_clique_cache(const std::vector<int>& device_ids) {
  if (device_ids.size() <= 1) {
    return absl::OkStatus();
  }
  bool cache_hit = false;
  auto clique_or = get_or_create_cached_clique(device_ids, &cache_hit);
  if (!clique_or.ok()) {
    return clique_or.status();
  }
  LOG(INFO) << "collective_disk_load clique prewarm complete device_ids=" << clique_cache_key(device_ids)
            << " cache_hit=" << (cache_hit ? 1 : 0);
  return absl::OkStatus();
}

} // namespace tensorcast::store::replica
