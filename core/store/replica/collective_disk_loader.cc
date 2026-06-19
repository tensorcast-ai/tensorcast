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
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
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
#include "core/common/artifact_hash.h"
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
#include "core/store/replica/source_window_batched_scatter_kernel.h"
#include "core/store/replica/source_window_collective_plan.h"
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
using FillRule = materialization::contracts::FillRule;
using TensorAxisRange = materialization::contracts::TensorAxisRange;
using TensorCoordinateSpec = materialization::contracts::TensorCoordinateSpec;
using WorkPartitionKind = materialization::contracts::WorkPartitionKind;

size_t compiled_routed_program_build_thread_count(size_t chunk_count, uint32_t configured_thread_count) {
  if (chunk_count < 2) {
    return 1;
  }
  if (configured_thread_count > 0) {
    if (configured_thread_count <= 1) {
      return 1;
    }
    return std::min<size_t>(static_cast<size_t>(configured_thread_count), chunk_count);
  }
  if (chunk_count < 32) {
    return 1;
  }
  const size_t hw = std::max<size_t>(1, std::thread::hardware_concurrency());
  return std::min<size_t>({chunk_count, size_t{16}, std::max<size_t>(2, hw / 4)});
}

struct SourceWindowRuntimeScatterGraph {
  int device_id{-1};
  cudaGraph_t graph{nullptr};
  cudaGraphExec_t exec{nullptr};
};

void destroy_source_window_runtime_scatter_graph(SourceWindowRuntimeScatterGraph* graph) {
  if (graph == nullptr) {
    return;
  }
  if (graph->device_id >= 0) {
    const absl::Status set_device_status = tensorcast::cuda::set_device(graph->device_id);
    if (!set_device_status.ok()) {
      LOG(WARNING) << "source-window scatter cuda graph set_device failed during cleanup: " << set_device_status;
    }
  }
  if (graph->exec != nullptr) {
    const absl::Status destroy_status =
        tensorcast::cuda::cuda_as_status(cudaGraphExecDestroy(graph->exec), "cudaGraphExecDestroy");
    if (!destroy_status.ok()) {
      LOG(WARNING) << "source-window scatter cuda graph exec destroy failed: " << destroy_status;
    }
    graph->exec = nullptr;
  }
  if (graph->graph != nullptr) {
    const absl::Status destroy_status =
        tensorcast::cuda::cuda_as_status(cudaGraphDestroy(graph->graph), "cudaGraphDestroy");
    if (!destroy_status.ok()) {
      LOG(WARNING) << "source-window scatter cuda graph destroy failed: " << destroy_status;
    }
    graph->graph = nullptr;
  }
}

void destroy_source_window_runtime_scatter_graphs(std::vector<SourceWindowRuntimeScatterGraph>* graphs) {
  if (graphs == nullptr) {
    return;
  }
  for (auto& graph : *graphs) {
    destroy_source_window_runtime_scatter_graph(&graph);
  }
  graphs->clear();
}

absl::StatusOr<SourceWindowRuntimeScatterGraph> build_source_window_runtime_scatter_graph(
    absl::Span<const SourceWindowBatchedScatterDescriptor> descriptors,
    int device_id) {
  if (descriptors.empty()) {
    return absl::InvalidArgumentError("source-window scatter cuda graph requires descriptors");
  }
  SourceWindowRuntimeScatterGraph result{.device_id = device_id};
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_id));
  absl::Status status = tensorcast::cuda::cuda_as_status(
      cudaGraphCreate(&result.graph, 0), "cudaGraphCreate(source-window scatter cuda graph)");
  if (!status.ok()) {
    return status;
  }
  for (const auto& desc : descriptors) {
    auto* dst_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(desc.dst_ptr));
    auto* src_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(desc.src_ptr));
    cudaGraphNode_t node = nullptr;
    if (desc.row_count > 1) {
      cudaMemcpy3DParms params{};
      params.srcPtr = make_cudaPitchedPtr(
          src_ptr,
          static_cast<size_t>(desc.source_stride_bytes),
          static_cast<size_t>(desc.row_bytes),
          static_cast<size_t>(desc.row_count));
      params.dstPtr = make_cudaPitchedPtr(
          dst_ptr,
          static_cast<size_t>(desc.target_stride_bytes),
          static_cast<size_t>(desc.row_bytes),
          static_cast<size_t>(desc.row_count));
      params.extent = make_cudaExtent(
          static_cast<size_t>(desc.row_bytes), static_cast<size_t>(desc.row_count), static_cast<size_t>(1));
      params.kind = cudaMemcpyDeviceToDevice;
      status = tensorcast::cuda::cuda_as_status(
          cudaGraphAddMemcpyNode(&node, result.graph, nullptr, 0, &params),
          "cudaGraphAddMemcpyNode(source-window scatter cuda graph)");
    } else {
      status = tensorcast::cuda::cuda_as_status(
          cudaGraphAddMemcpyNode1D(
              &node,
              result.graph,
              nullptr,
              0,
              dst_ptr,
              src_ptr,
              static_cast<size_t>(desc.row_bytes),
              cudaMemcpyDeviceToDevice),
          "cudaGraphAddMemcpyNode1D(source-window scatter cuda graph)");
    }
    if (!status.ok()) {
      destroy_source_window_runtime_scatter_graph(&result);
      return status;
    }
  }
  status = tensorcast::cuda::cuda_as_status(
      cudaGraphInstantiate(&result.exec, result.graph, nullptr, nullptr, 0),
      "cudaGraphInstantiate(source-window scatter cuda graph)");
  if (!status.ok()) {
    destroy_source_window_runtime_scatter_graph(&result);
    return status;
  }
  return result;
}

absl::Status source_window_runtime_scatter_graph_node_count(cudaGraph_t graph, uint64_t* node_count) {
  if (node_count == nullptr) {
    return absl::InvalidArgumentError("source-window scatter cuda graph node_count output is null");
  }
  size_t graph_node_count = 0;
  TC_RETURN_IF_ERROR(
      tensorcast::cuda::cuda_as_status(
          cudaGraphGetNodes(graph, nullptr, &graph_node_count), "cudaGraphGetNodes(source-window scatter cuda graph)"));
  *node_count = static_cast<uint64_t>(graph_node_count);
  return absl::OkStatus();
}

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

bool verbose_materialization_strategy_diagnostics(const StrategyConfig& strategy) {
  return strategy.diagnostics_verbosity == StrategyConfig::DiagnosticsVerbosity::kVerbose;
}

bool verbose_mapped_concat_diagnostics(const StrategyConfig& strategy) {
  return verbose_materialization_strategy_diagnostics(strategy);
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

  absl::Status all_gather_u8(int rank, const void* send, void* recv, size_t bytes) {
    return nccl_status(
        ncclAllGather(send, recv, bytes, ncclUint8, ranks_[static_cast<size_t>(rank)].comm, stream(rank)),
        "ncclAllGather");
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
    kRect2DPacked = 2,
  };

  Kind kind{Kind::kContiguous};
  uint64_t source_offset{0};
  uint64_t read_bytes{0};
  std::uint8_t* dst_ptr{nullptr};
  uint64_t dst_bytes{0};
  uint64_t src_col_offset_bytes{0};
  uint64_t src_pitch_bytes{0};
  uint64_t source_row_stride_bytes{0};
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

struct SourceWindowMappedParticipant {
  std::string artifact_id;
  int rank{-1};
  int device_id{-1};
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  std::shared_ptr<const RepresentationWorkPlan> work_plan;
  std::shared_ptr<const loading::IntoTargetLayout> target_layout;
  std::vector<TargetStorageSpan> storage_spans;
  runtime::ingestion::strategy::SourceWindowCollectiveCandidateSummary candidate_summary;
  std::string source_index_digest;
  std::optional<loading::SourceWindowPreparedRealizationFacts> prepared_realization;
};

struct SourceWindowMappedGroupState {
  explicit SourceWindowMappedGroupState(uint32_t size) : world_size(size), participants(size) {}

  const uint32_t world_size;
  absl::Mutex mu;
  absl::CondVar cv;
  std::vector<std::optional<SourceWindowMappedParticipant>> participants ABSL_GUARDED_BY(mu);
  uint32_t joined ABSL_GUARDED_BY(mu){0};
  bool launching ABSL_GUARDED_BY(mu){false};
  bool complete ABSL_GUARDED_BY(mu){false};
  SourceWindowCollectiveMappedTargetLoadResult result ABSL_GUARDED_BY(mu);
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
ABSL_CONST_INIT absl::Mutex g_source_window_mapped_group_mu(absl::kConstInit);
absl::flat_hash_map<std::string, std::shared_ptr<SourceWindowMappedGroupState>> g_source_window_mapped_groups
    ABSL_GUARDED_BY(g_source_window_mapped_group_mu);
ABSL_CONST_INIT absl::Mutex g_clique_mu(absl::kConstInit);
absl::flat_hash_map<std::string, std::shared_ptr<NcclClique>> g_clique_cache ABSL_GUARDED_BY(g_clique_mu);

constexpr size_t kSourceWindowCollectivePlanCacheMaxEntries = 8;
constexpr size_t kSourceWindowRoutedProgramCacheMaxEntries = 8;

struct SourceWindowCollectivePlanCacheState {
  absl::Mutex mu;
  absl::flat_hash_map<std::string, SourceWindowCollectivePlan> plans ABSL_GUARDED_BY(mu);
  std::vector<std::string> insertion_order ABSL_GUARDED_BY(mu);
  std::unordered_set<std::string> inflight ABSL_GUARDED_BY(mu);
  absl::CondVar cv;
  uint64_t hits ABSL_GUARDED_BY(mu){0};
  uint64_t misses ABSL_GUARDED_BY(mu){0};
  uint64_t waits ABSL_GUARDED_BY(mu){0};
};

struct RoutedTargetRef {
  uint32_t rank{0};
  uint32_t storage_index{0};
  uint64_t logical_offset{0};
};

struct RoutedDescriptorTemplate {
  size_t rank{0};
  uint64_t src_offset{0};
  RoutedTargetRef target;
  uint64_t row_bytes{0};
  uint64_t row_count{1};
  uint64_t source_stride_bytes{0};
  uint64_t target_stride_bytes{0};
};

struct RoutedPackDescriptorTemplate {
  size_t producer{0};
  size_t consumer{0};
  uint64_t src_offset{0};
  uint64_t pack_offset{0};
  uint64_t row_bytes{0};
  uint64_t row_count{1};
  uint64_t source_stride_bytes{0};
  uint64_t target_stride_bytes{0};
};

struct RoutedPackedRemotePieceTemplate {
  size_t producer{0};
  size_t consumer{0};
  uint64_t pack_offset{0};
  RoutedTargetRef target;
  uint64_t length{0};
};

struct RoutedDirectRemotePieceTemplate {
  size_t producer{0};
  size_t consumer{0};
  uint64_t src_offset{0};
  RoutedTargetRef target;
  uint64_t length{0};
};

struct RoutedPackedTransferTemplate {
  size_t producer{0};
  size_t consumer{0};
  uint64_t bytes{0};
};

struct SourceWindowRoutedChunkProgram {
  bool compiled{false};
  std::vector<std::vector<RoutedDescriptorTemplate>> local_descriptors_by_rank;
  std::vector<std::vector<RoutedPackDescriptorTemplate>> pack_descriptors_by_producer;
  std::vector<RoutedPackedRemotePieceTemplate> packed_remote_pieces;
  std::vector<RoutedDirectRemotePieceTemplate> direct_remote_pieces;
  std::vector<RoutedPackedTransferTemplate> packed_transfers;
  uint64_t target_storage_fast_path_pieces{0};
  uint64_t target_storage_fast_path_bytes{0};
  uint64_t local_2d_pieces{0};
  uint64_t local_pieces{0};
  uint64_t pack_ops{0};
  uint64_t deferred_2d_pack_ops{0};
  uint64_t packed_remote_piece_count{0};
  uint64_t direct_remote_piece_count{0};
};

struct SourceWindowRoutedProgramCacheEntry {
  std::vector<SourceWindowRoutedChunkProgram> programs;
};

struct SourceWindowRoutedProgramCacheState {
  absl::Mutex mu;
  absl::flat_hash_map<std::string, SourceWindowRoutedProgramCacheEntry> entries ABSL_GUARDED_BY(mu);
  std::vector<std::string> insertion_order ABSL_GUARDED_BY(mu);
  std::unordered_set<std::string> inflight ABSL_GUARDED_BY(mu);
  absl::CondVar cv;
  uint64_t hits ABSL_GUARDED_BY(mu){0};
  uint64_t misses ABSL_GUARDED_BY(mu){0};
  uint64_t waits ABSL_GUARDED_BY(mu){0};
};

struct SourceWindowRoutedProgramCacheAcquireResult {
  bool cache_hit{false};
  bool reserved_build{false};
  bool waited{false};
  bool size_mismatch{false};
  double wait_sec{0.0};
  std::vector<SourceWindowRoutedChunkProgram> programs;
};

struct SourceWindowRuntimeChunk {
  const SourceWindowCollectiveWindow* window{nullptr};
  std::vector<const SourceWindowCollectiveConsumerSpan*> consumer_spans;
  SourceWindowRoutedChunkProgram routed_program;
  uint64_t chunk_start{0};
  uint64_t chunk_end{0};
  size_t chunk_len{0};
  size_t stripe_bytes{0};
  size_t gathered_bytes{0};
};

SourceWindowCollectivePlanCacheState& source_window_collective_plan_cache() {
  static auto* cache = new SourceWindowCollectivePlanCacheState();
  return *cache;
}

SourceWindowRoutedProgramCacheState& source_window_routed_program_cache() {
  static auto* cache = new SourceWindowRoutedProgramCacheState();
  return *cache;
}

const RepresentationWorkPlan& source_window_member_work_plan(const SourceWindowCollectiveMemberInput& member) {
  if (member.work_plan_ref != nullptr) {
    return *member.work_plan_ref;
  }
  return member.work_plan;
}

const loading::IntoTargetLayout& source_window_member_target_layout(const SourceWindowCollectiveMemberInput& member) {
  if (member.target_layout_ref != nullptr) {
    return *member.target_layout_ref;
  }
  return member.target_layout;
}

void append_key_u8(std::string* payload, uint8_t value) {
  payload->push_back(static_cast<char>(value));
}

void append_key_u32(std::string* payload, uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    payload->push_back(static_cast<char>((value >> shift) & 0xffU));
  }
}

void append_key_u64(std::string* payload, uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    payload->push_back(static_cast<char>((value >> shift) & 0xffULL));
  }
}

void append_key_i32(std::string* payload, int32_t value) {
  append_key_u32(payload, static_cast<uint32_t>(value));
}

void append_key_i64(std::string* payload, int64_t value) {
  append_key_u64(payload, static_cast<uint64_t>(value));
}

void append_key_bool(std::string* payload, bool value) {
  append_key_u8(payload, value ? 1 : 0);
}

void append_key_size(std::string* payload, size_t value) {
  append_key_u64(payload, static_cast<uint64_t>(value));
}

void append_key_string(std::string* payload, std::string_view value) {
  append_key_size(payload, value.size());
  payload->append(value.data(), value.size());
}

void append_key_bytes(std::string* payload, absl::Span<const uint8_t> values) {
  append_key_size(payload, values.size());
  if (!values.empty()) {
    payload->append(reinterpret_cast<const char*>(values.data()), values.size());
  }
}

void append_int64_vector_to_key(std::string* payload, absl::Span<const int64_t> values) {
  append_key_size(payload, values.size());
  for (const auto value : values) {
    append_key_i64(payload, value);
  }
}

void append_tensor_coordinate_to_key(std::string* payload, const TensorCoordinateSpec& coordinate) {
  append_key_bool(payload, coordinate.selects_scalar);
  append_key_size(payload, coordinate.axes.size());
  for (const auto& axis : coordinate.axes) {
    append_key_i32(payload, axis.dim);
    append_key_i64(payload, axis.start);
    append_key_i64(payload, axis.end);
  }
}

void append_tensor_spec_to_key(std::string* payload, const RepresentationTensorSpec& spec) {
  append_key_string(payload, spec.name);
  append_key_string(payload, spec.dtype);
  append_key_u64(payload, spec.logical_offset);
  append_key_u64(payload, spec.logical_length);
  append_key_u64(payload, spec.storage_offset);
  append_key_u64(payload, spec.element_size);
  append_int64_vector_to_key(payload, absl::MakeConstSpan(spec.shape));
  append_int64_vector_to_key(payload, absl::MakeConstSpan(spec.stride));
}

void append_fill_rule_to_key(std::string* payload, const FillRule& fill_rule) {
  append_key_bytes(payload, absl::MakeConstSpan(fill_rule.constant_value));
  append_tensor_coordinate_to_key(payload, fill_rule.destination_range);
}

void append_source_fragment_to_key(std::string* payload, const RepresentationWorkSourceFragment& source) {
  append_tensor_spec_to_key(payload, source.fragment.source_spec);
  append_tensor_coordinate_to_key(payload, source.fragment.source_range);
  append_tensor_coordinate_to_key(payload, source.fragment.destination_range);
  append_key_u8(payload, static_cast<uint8_t>(source.fragment.role));
  append_key_u64(payload, source.prefix_count);
  append_key_u64(payload, source.dst_block_offset_bytes);
  append_key_u64(payload, source.dst_block_stride_bytes);
  append_key_u64(payload, source.dst_block_bytes);
}

void append_work_item_to_key(std::string* payload, const RepresentationWorkItem& item) {
  append_key_u8(payload, static_cast<uint8_t>(item.kind));
  append_key_u8(payload, static_cast<uint8_t>(item.partition_kind));
  append_key_string(payload, item.dst_name);
  append_key_u64(payload, item.committed_bytes);
  append_tensor_spec_to_key(payload, item.dst_spec);
  append_key_size(payload, item.sources.size());
  for (const auto& source : item.sources) {
    append_source_fragment_to_key(payload, source);
  }
  append_key_bool(payload, item.fill_rule.has_value());
  if (item.fill_rule.has_value()) {
    append_fill_rule_to_key(payload, *item.fill_rule);
  }
}

void append_work_plan_to_key(std::string* payload, const RepresentationWorkPlan& work_plan) {
  append_key_u64(payload, work_plan.committed_bytes);
  append_key_size(payload, work_plan.items.size());
  for (const auto& item : work_plan.items) {
    append_work_item_to_key(payload, item);
  }
  // Source-window collective planning consumes work item geometry and residual
  // coverage totals, not the local mapped byte-range-map segment expansion.
  // Including those segments makes the strict cache key O(local-mapped plan)
  // without adding source-window identity.
  append_key_u64(payload, work_plan.residual_fallback_map.total_bytes);
}

void append_source_window_config_to_key(std::string* payload, const SourceWindowCollectiveConfig& config) {
  absl::StrAppend(
      payload,
      "{enabled=",
      config.enabled,
      ",selection=",
      runtime::ingestion::strategy::source_window_collective_selection_mode_name(config.selection_mode),
      ",window_bytes=",
      config.window_bytes,
      ",max_gap_bytes=",
      config.max_gap_bytes,
      ",max_window_amplification_x1000=",
      config.max_window_amplification_x1000,
      ",max_plan_read_amplification_x1000=",
      config.max_plan_read_amplification_x1000,
      ",max_scatter_ops_per_window=",
      config.max_scatter_ops_per_window,
      ",peak_bytes_budget=",
      config.peak_bytes_budget,
      ",min_rank_read_saving_bytes=",
      config.min_rank_read_saving_bytes,
      ",max_peer_to_read_ratio_x1000=",
      config.max_peer_to_read_ratio_x1000,
      ",min_routed_peer_saving_bytes=",
      config.min_routed_peer_saving_bytes,
      ",distribution=",
      runtime::ingestion::strategy::source_window_collective_distribution_mode_name(config.distribution_mode),
      ",allow_mixed_residual=",
      config.allow_mixed_residual,
      "}");
}

std::string source_window_collective_plan_cache_key(
    std::string_view artifact_id,
    const SourceWindowCollectiveGroupInput& input) {
  size_t item_count = 0;
  size_t source_count = 0;
  for (const auto& member : input.members) {
    const auto& work_plan = source_window_member_work_plan(member);
    item_count += work_plan.items.size();
    for (const auto& item : work_plan.items) {
      source_count += item.sources.size();
    }
  }

  std::string payload;
  payload.reserve(1024 + input.members.size() * 256 + item_count * 384 + source_count * 256);
  absl::StrAppend(
      &payload, "source_window_collective_group_plan_cache_v1|artifact_id=", artifact_id, "|artifact_path=");
  if (input.disk_context != nullptr) {
    absl::StrAppend(&payload, input.disk_context->artifact_path().generic_string());
  }
  absl::StrAppend(
      &payload, "|source_index_digest=", input.source_index_digest, "|world_size=", input.group.world_size, "|config=");
  append_source_window_config_to_key(&payload, input.config);
  absl::StrAppend(&payload, "|members=", input.members.size());
  for (const auto& member : input.members) {
    const auto& target_layout = source_window_member_target_layout(member);
    const auto& work_plan = source_window_member_work_plan(member);
    absl::StrAppend(
        &payload,
        "|member=",
        member.rank,
        ",device=",
        member.device_id,
        ",target_total=",
        target_layout.total_size,
        ",storages=",
        target_layout.storages.size());
    for (const auto& storage : target_layout.storages) {
      absl::StrAppend(&payload, "/storage_length=", storage.length);
    }
    absl::StrAppend(&payload, ",storage_spans=", member.storage_spans.size());
    for (const auto& storage_span : member.storage_spans) {
      absl::StrAppend(&payload, "/span=", storage_span.base_offset, ":", storage_span.length);
    }
    absl::StrAppend(&payload, ",work_plan=");
    append_work_plan_to_key(&payload, work_plan);
  }
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return common::multibase_multihash_sha256(digest);
}

std::optional<std::string> source_window_collective_prepared_plan_cache_key(
    std::string_view artifact_id,
    const SourceWindowCollectiveGroupInput& input) {
  if (input.members.empty()) {
    return std::nullopt;
  }

  struct PreparedMemberKeyPart {
    uint32_t rank{0};
    int device_id{-1};
    std::string group_key;
    std::string member_key;
    std::string realization_plan_hash;
    std::string target_layout_template_hash;
    std::string target_index_hash;
    uint64_t target_total_size{0};
    std::vector<uint64_t> storage_lengths;
    std::vector<std::pair<uint64_t, uint64_t>> storage_spans;
  };

  std::vector<PreparedMemberKeyPart> member_key_parts;
  member_key_parts.reserve(input.members.size());
  std::unordered_set<std::string> member_keys;
  for (const auto& member : input.members) {
    if (!member.prepared_realization.has_value()) {
      return std::nullopt;
    }
    const auto& facts = *member.prepared_realization;
    if (facts.group_key.empty() || facts.member_key.empty() || facts.realization_plan_hash.empty() ||
        facts.target_layout_template_hash.empty() || facts.target_index_hash.empty()) {
      return std::nullopt;
    }
    if (!member_keys.insert(facts.member_key).second) {
      return std::nullopt;
    }

    const auto& target_layout = source_window_member_target_layout(member);
    PreparedMemberKeyPart part{
        .rank = member.rank,
        .device_id = member.device_id,
        .group_key = facts.group_key,
        .member_key = facts.member_key,
        .realization_plan_hash = facts.realization_plan_hash,
        .target_layout_template_hash = facts.target_layout_template_hash,
        .target_index_hash = facts.target_index_hash,
        .target_total_size = target_layout.total_size,
    };
    part.storage_lengths.reserve(target_layout.storages.size());
    for (const auto& storage : target_layout.storages) {
      part.storage_lengths.push_back(storage.length);
    }
    part.storage_spans.reserve(member.storage_spans.size());
    for (const auto& storage_span : member.storage_spans) {
      part.storage_spans.emplace_back(storage_span.base_offset, storage_span.length);
    }
    member_key_parts.push_back(std::move(part));
  }

  std::sort(
      member_key_parts.begin(),
      member_key_parts.end(),
      [](const PreparedMemberKeyPart& lhs, const PreparedMemberKeyPart& rhs) {
        return std::tie(lhs.rank, lhs.device_id, lhs.member_key) < std::tie(rhs.rank, rhs.device_id, rhs.member_key);
      });

  std::string payload;
  payload.reserve(1024 + member_key_parts.size() * 512);
  absl::StrAppend(
      &payload, "source_window_collective_group_prepared_plan_cache_v1|artifact_id=", artifact_id, "|artifact_path=");
  if (input.disk_context != nullptr) {
    absl::StrAppend(&payload, input.disk_context->artifact_path().generic_string());
  }
  absl::StrAppend(
      &payload, "|source_index_digest=", input.source_index_digest, "|world_size=", input.group.world_size, "|config=");
  append_source_window_config_to_key(&payload, input.config);
  absl::StrAppend(&payload, "|members=", member_key_parts.size());
  for (const auto& part : member_key_parts) {
    absl::StrAppend(
        &payload,
        "|member=",
        part.rank,
        ",device=",
        part.device_id,
        ",group_key=",
        part.group_key,
        ",member_key=",
        part.member_key,
        ",realization_plan_hash=",
        part.realization_plan_hash,
        ",target_layout_template_hash=",
        part.target_layout_template_hash,
        ",target_index_hash=",
        part.target_index_hash,
        ",target_total=",
        part.target_total_size,
        ",storages=",
        part.storage_lengths.size());
    for (const auto length : part.storage_lengths) {
      absl::StrAppend(&payload, "/storage_length=", length);
    }
    absl::StrAppend(&payload, ",storage_spans=", part.storage_spans.size());
    for (const auto& [base_offset, length] : part.storage_spans) {
      absl::StrAppend(&payload, "/span=", base_offset, ":", length);
    }
  }

  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return common::multibase_multihash_sha256(digest);
}

bool lookup_source_window_collective_plan_cache(
    const std::string& key,
    const loading::CollectiveLoadGroupHint& group,
    SourceWindowCollectivePlan* plan) {
  auto& cache = source_window_collective_plan_cache();
  absl::MutexLock lock(&cache.mu);
  for (;;) {
    auto it = cache.plans.find(key);
    if (it != cache.plans.end()) {
      cache.hits += 1;
      if (plan != nullptr) {
        *plan = it->second;
        plan->group = group;
      }
      return true;
    }

    if (cache.inflight.find(key) == cache.inflight.end()) {
      cache.inflight.insert(key);
      cache.misses += 1;
      return false;
    }

    cache.waits += 1;
    cache.cv.Wait(&cache.mu);
  }
}

void abandon_source_window_collective_plan_cache_build(const std::string& key) {
  auto& cache = source_window_collective_plan_cache();
  absl::MutexLock lock(&cache.mu);
  cache.inflight.erase(key);
  cache.cv.SignalAll();
}

void store_source_window_collective_plan_cache(const std::string& key, const SourceWindowCollectivePlan& plan) {
  auto& cache = source_window_collective_plan_cache();
  absl::MutexLock lock(&cache.mu);
  auto existing = cache.plans.find(key);
  if (existing != cache.plans.end()) {
    existing->second = plan;
    cache.inflight.erase(key);
    cache.cv.SignalAll();
    return;
  }
  while (cache.plans.size() >= kSourceWindowCollectivePlanCacheMaxEntries && !cache.insertion_order.empty()) {
    cache.plans.erase(cache.insertion_order.front());
    cache.insertion_order.erase(cache.insertion_order.begin());
  }
  cache.plans.emplace(key, plan);
  cache.insertion_order.push_back(key);
  cache.inflight.erase(key);
  cache.cv.SignalAll();
}

void clear_source_window_collective_plan_cache() {
  auto& cache = source_window_collective_plan_cache();
  absl::MutexLock lock(&cache.mu);
  cache.plans.clear();
  cache.insertion_order.clear();
  cache.inflight.clear();
  cache.hits = 0;
  cache.misses = 0;
  cache.waits = 0;
  cache.cv.SignalAll();
}

SourceWindowCollectivePlanCacheStats source_window_collective_plan_cache_stats_snapshot() {
  auto& cache = source_window_collective_plan_cache();
  absl::MutexLock lock(&cache.mu);
  return SourceWindowCollectivePlanCacheStats{
      .hits = cache.hits,
      .misses = cache.misses,
      .entries = cache.plans.size(),
  };
}

std::optional<std::string> source_window_routed_program_cache_key(
    std::string_view artifact_id,
    const SourceWindowCollectivePlan& plan,
    absl::Span<const SourceWindowMappedParticipant> participants,
    size_t configured_chunk_bytes,
    size_t max_collective_chunk_bytes,
    size_t max_stripe_bytes) {
  if (artifact_id.empty() || plan.plan_hash.empty() || participants.empty()) {
    return std::nullopt;
  }

  std::string payload;
  payload.reserve(1024 + participants.size() * 256);
  absl::StrAppend(&payload, "source_window_routed_program_cache_v2|artifact_id=");
  append_key_string(&payload, artifact_id);
  absl::StrAppend(&payload, "|artifact_path=");
  if (participants.front().disk_context != nullptr) {
    append_key_string(&payload, participants.front().disk_context->artifact_path().generic_string());
  } else {
    append_key_string(&payload, "");
  }
  absl::StrAppend(&payload, "|plan_hash=");
  append_key_string(&payload, plan.plan_hash);
  append_key_size(&payload, participants.size());
  append_key_size(&payload, configured_chunk_bytes);
  append_key_size(&payload, max_collective_chunk_bytes);
  append_key_size(&payload, max_stripe_bytes);
  append_key_size(&payload, plan.windows.size());

  bool can_use_prepared_identity = true;
  std::string prepared_group_key;
  for (const auto& participant : participants) {
    if (!participant.prepared_realization.has_value()) {
      can_use_prepared_identity = false;
      break;
    }
    const auto& facts = *participant.prepared_realization;
    if (facts.group_key.empty() || facts.member_key.empty() || facts.realization_plan_hash.empty() ||
        facts.target_layout_template_hash.empty() || facts.target_index_hash.empty()) {
      can_use_prepared_identity = false;
      break;
    }
    if (prepared_group_key.empty()) {
      prepared_group_key = facts.group_key;
    } else if (prepared_group_key != facts.group_key) {
      can_use_prepared_identity = false;
      break;
    }
  }

  append_key_bool(&payload, can_use_prepared_identity);
  if (can_use_prepared_identity) {
    append_key_string(&payload, prepared_group_key);
  }

  std::vector<const SourceWindowMappedParticipant*> sorted_participants;
  sorted_participants.reserve(participants.size());
  for (const auto& participant : participants) {
    sorted_participants.push_back(&participant);
  }
  std::sort(
      sorted_participants.begin(),
      sorted_participants.end(),
      [](const SourceWindowMappedParticipant* lhs, const SourceWindowMappedParticipant* rhs) {
        return std::tie(lhs->rank, lhs->device_id) < std::tie(rhs->rank, rhs->device_id);
      });

  for (const auto* participant : sorted_participants) {
    append_key_i32(&payload, participant->rank);
    append_key_i32(&payload, participant->device_id);
    append_key_string(&payload, participant->source_index_digest);
    if (can_use_prepared_identity) {
      const auto& facts = *participant->prepared_realization;
      append_key_string(&payload, facts.member_key);
      append_key_string(&payload, facts.realization_plan_hash);
      append_key_string(&payload, facts.target_layout_template_hash);
      append_key_string(&payload, facts.target_index_hash);
    }
    append_key_size(&payload, participant->storage_spans.size());
    for (const auto& storage_span : participant->storage_spans) {
      append_key_u64(&payload, storage_span.base_offset);
      append_key_u64(&payload, storage_span.length);
    }
  }

  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return common::multibase_multihash_sha256(digest);
}

void erase_source_window_routed_program_cache_entry_locked(
    SourceWindowRoutedProgramCacheState& cache,
    const std::string& key) ABSL_EXCLUSIVE_LOCKS_REQUIRED(cache.mu) {
  cache.entries.erase(key);
  const auto order_it = std::find(cache.insertion_order.begin(), cache.insertion_order.end(), key);
  if (order_it != cache.insertion_order.end()) {
    cache.insertion_order.erase(order_it);
  }
}

void store_source_window_routed_program_cache_locked(
    SourceWindowRoutedProgramCacheState& cache,
    const std::string& key,
    std::vector<SourceWindowRoutedChunkProgram> programs) ABSL_EXCLUSIVE_LOCKS_REQUIRED(cache.mu) {
  auto existing = cache.entries.find(key);
  if (existing != cache.entries.end()) {
    existing->second.programs = std::move(programs);
    return;
  }
  while (cache.entries.size() >= kSourceWindowRoutedProgramCacheMaxEntries && !cache.insertion_order.empty()) {
    cache.entries.erase(cache.insertion_order.front());
    cache.insertion_order.erase(cache.insertion_order.begin());
  }
  cache.entries.emplace(key, SourceWindowRoutedProgramCacheEntry{.programs = std::move(programs)});
  cache.insertion_order.push_back(key);
}

SourceWindowRoutedProgramCacheAcquireResult acquire_source_window_routed_program_cache(
    const std::string& key,
    size_t expected_program_count) {
  SourceWindowRoutedProgramCacheAcquireResult result;
  auto& cache = source_window_routed_program_cache();
  const auto wait_start = std::chrono::steady_clock::now();
  absl::MutexLock lock(&cache.mu);
  for (;;) {
    auto it = cache.entries.find(key);
    if (it != cache.entries.end()) {
      if (expected_program_count > 0 && it->second.programs.size() != expected_program_count) {
        result.size_mismatch = true;
        erase_source_window_routed_program_cache_entry_locked(cache, key);
        cache.misses += 1;
        cache.inflight.insert(key);
        result.reserved_build = true;
        if (result.waited) {
          result.wait_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start).count();
        }
        return result;
      }
      cache.hits += 1;
      result.cache_hit = true;
      result.programs = it->second.programs;
      if (result.waited) {
        result.wait_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start).count();
      }
      return result;
    }

    if (cache.inflight.find(key) == cache.inflight.end()) {
      cache.misses += 1;
      cache.inflight.insert(key);
      result.reserved_build = true;
      if (result.waited) {
        result.wait_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start).count();
      }
      return result;
    }

    result.waited = true;
    cache.waits += 1;
    cache.cv.Wait(&cache.mu);
  }
}

void complete_source_window_routed_program_cache_build(
    const std::string& key,
    std::vector<SourceWindowRoutedChunkProgram> programs) {
  auto& cache = source_window_routed_program_cache();
  absl::MutexLock lock(&cache.mu);
  store_source_window_routed_program_cache_locked(cache, key, std::move(programs));
  cache.inflight.erase(key);
  cache.cv.SignalAll();
}

void abandon_source_window_routed_program_cache_build(const std::string& key) {
  auto& cache = source_window_routed_program_cache();
  absl::MutexLock lock(&cache.mu);
  cache.inflight.erase(key);
  cache.cv.SignalAll();
}

void store_source_window_routed_program_cache(
    const std::string& key,
    std::vector<SourceWindowRoutedChunkProgram> programs) {
  complete_source_window_routed_program_cache_build(key, std::move(programs));
}

void clear_source_window_routed_program_cache() {
  auto& cache = source_window_routed_program_cache();
  absl::MutexLock lock(&cache.mu);
  cache.entries.clear();
  cache.insertion_order.clear();
  cache.inflight.clear();
  cache.hits = 0;
  cache.misses = 0;
  cache.waits = 0;
  cache.cv.SignalAll();
}

SourceWindowCollectivePlanCacheStats source_window_routed_program_cache_stats_snapshot() {
  auto& cache = source_window_routed_program_cache();
  absl::MutexLock lock(&cache.mu);
  return SourceWindowCollectivePlanCacheStats{
      .hits = cache.hits,
      .misses = cache.misses,
      .entries = cache.entries.size(),
  };
}

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

struct SourceWindowDeviceMemorySnapshot {
  int device_id{-1};
  bool ok{false};
  size_t free_bytes{0};
  size_t total_bytes{0};
  std::string error;
};

std::vector<SourceWindowDeviceMemorySnapshot> capture_source_window_device_memory(
    const std::vector<int>& device_ids) {
  std::vector<SourceWindowDeviceMemorySnapshot> snapshots;
  snapshots.reserve(device_ids.size());
  for (int device_id : device_ids) {
    SourceWindowDeviceMemorySnapshot snapshot;
    snapshot.device_id = device_id;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const absl::Status status = tensorcast::cuda::get_memory_info(&free_bytes, &total_bytes, device_id);
    if (status.ok()) {
      snapshot.ok = true;
      snapshot.free_bytes = free_bytes;
      snapshot.total_bytes = total_bytes;
    } else {
      snapshot.error = status.ToString();
    }
    snapshots.push_back(std::move(snapshot));
  }
  return snapshots;
}

std::string join_source_window_memory_bool_field(
    const std::vector<SourceWindowDeviceMemorySnapshot>& snapshots) {
  std::string out = "[";
  for (size_t i = 0; i < snapshots.size(); ++i) {
    if (i != 0) {
      absl::StrAppend(&out, ",");
    }
    absl::StrAppend(&out, snapshots[i].ok ? 1 : 0);
  }
  absl::StrAppend(&out, "]");
  return out;
}

std::string join_source_window_memory_size_field(
    const std::vector<SourceWindowDeviceMemorySnapshot>& snapshots,
    bool total_bytes) {
  std::string out = "[";
  for (size_t i = 0; i < snapshots.size(); ++i) {
    if (i != 0) {
      absl::StrAppend(&out, ",");
    }
    if (snapshots[i].ok) {
      absl::StrAppend(&out, total_bytes ? snapshots[i].total_bytes : snapshots[i].free_bytes);
    } else {
      absl::StrAppend(&out, -1);
    }
  }
  absl::StrAppend(&out, "]");
  return out;
}

std::string join_source_window_memory_delta_field(
    const std::vector<SourceWindowDeviceMemorySnapshot>& snapshots,
    const std::vector<SourceWindowDeviceMemorySnapshot>* baseline) {
  std::string out = "[";
  for (size_t i = 0; i < snapshots.size(); ++i) {
    if (i != 0) {
      absl::StrAppend(&out, ",");
    }
    if (baseline != nullptr && i < baseline->size() && snapshots[i].ok && (*baseline)[i].ok &&
        snapshots[i].device_id == (*baseline)[i].device_id) {
      absl::StrAppend(
          &out,
          static_cast<long long>(snapshots[i].free_bytes) -
              static_cast<long long>((*baseline)[i].free_bytes));
    } else {
      absl::StrAppend(&out, "null");
    }
  }
  absl::StrAppend(&out, "]");
  return out;
}

std::string join_source_window_memory_error_field(
    const std::vector<SourceWindowDeviceMemorySnapshot>& snapshots) {
  std::string out = "[";
  for (size_t i = 0; i < snapshots.size(); ++i) {
    if (i != 0) {
      absl::StrAppend(&out, "|");
    }
    if (snapshots[i].ok) {
      absl::StrAppend(&out, "ok");
    } else {
      absl::StrAppend(&out, snapshots[i].error);
    }
  }
  absl::StrAppend(&out, "]");
  return out;
}

void log_source_window_collective_memory(
    absl::string_view stage,
    absl::string_view artifact_id,
    absl::string_view group_id,
    const std::vector<SourceWindowDeviceMemorySnapshot>& snapshots,
    const std::vector<SourceWindowDeviceMemorySnapshot>* baseline = nullptr) {
  LOG(INFO) << "source_window_collective_memory"
            << " stage=" << stage
            << " artifact_id=" << artifact_id
            << " group_id=" << group_id
            << " ok=" << join_source_window_memory_bool_field(snapshots)
            << " free_bytes=" << join_source_window_memory_size_field(snapshots, /*total_bytes=*/false)
            << " total_bytes=" << join_source_window_memory_size_field(snapshots, /*total_bytes=*/true)
            << " free_delta_from_entry_bytes=" << join_source_window_memory_delta_field(snapshots, baseline)
            << " errors=" << join_source_window_memory_error_field(snapshots);
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

void try_evict_cached_clique_if_idle(
    const std::vector<int>& device_ids,
    const std::weak_ptr<NcclClique>& expected_clique,
    absl::string_view reason) {
  const std::string key = clique_cache_key(device_ids);
  bool found = false;
  bool same_entry = false;
  bool erased = false;
  long use_count = 0;
  std::owner_less<void> owner_less;
  {
    absl::MutexLock lock(&g_clique_mu);
    auto it = g_clique_cache.find(key);
    if (it != g_clique_cache.end()) {
      found = true;
      same_entry =
          !owner_less(expected_clique, it->second) && !owner_less(it->second, expected_clique);
      use_count = it->second.use_count();
      if (same_entry && use_count == 1) {
        g_clique_cache.erase(it);
        erased = true;
      }
    }
  }
  LOG(INFO) << "collective_disk_load clique cache idle evict device_ids=" << key
            << " found=" << (found ? 1 : 0)
            << " same_entry=" << (same_entry ? 1 : 0)
            << " use_count=" << use_count
            << " erased=" << (erased ? 1 : 0)
            << " reason=" << reason;
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

struct LocalMappedDirectProbeResult {
  bool attempted{false};
  bool supported{false};
  uint64_t bytes{0};
  int error_number{0};
  std::string status;
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
  // may still be probed below before auto falls back to the buffered path.
  constexpr uint64_t kExtSuperMagic = 0xEF53;
  constexpr uint64_t kXfsSuperMagic = 0x58465342;
  constexpr uint64_t kBtrfsSuperMagic = 0x9123683E;
  constexpr uint64_t kF2fsSuperMagic = 0xF2F52010;
  return fs_type == kExtSuperMagic || fs_type == kXfsSuperMagic || fs_type == kBtrfsSuperMagic ||
      fs_type == kF2fsSuperMagic;
}

bool is_memory_fs_type(uint64_t fs_type) {
  constexpr uint64_t kTmpfsSuperMagic = 0x01021994;
  constexpr uint64_t kRamfsSuperMagic = 0x858458F6;
  constexpr uint64_t kHugetlbfsSuperMagic = 0x958458F6;
  return fs_type == kTmpfsSuperMagic || fs_type == kRamfsSuperMagic || fs_type == kHugetlbfsSuperMagic;
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

LocalMappedDirectProbeResult probe_direct_source_capability(
    absl::Span<const loader::SharedSafetensorsSegment> segments) {
  for (const auto& segment : segments) {
    if (segment.path.empty() || segment.data_size == 0) {
      continue;
    }
    struct stat st{};
    if (::stat(segment.path.c_str(), &st) != 0) {
      return LocalMappedDirectProbeResult{
          .attempted = true,
          .supported = false,
          .error_number = errno,
          .status = absl::StrCat("stat_failed:", segment.path.string()),
      };
    }
    if (st.st_size <= 0) {
      continue;
    }
    const uint64_t file_size = static_cast<uint64_t>(st.st_size);
    const uint64_t data_begin = segment.data_start;
    const uint64_t data_end = segment.data_start + segment.data_size;
    const uint64_t probe_begin = align_up_u64(data_begin, kLocalMappedDirectIoAlignment);
    const uint64_t probe_limit =
        std::min<uint64_t>(align_down_u64(data_end, kLocalMappedDirectIoAlignment),
                           align_down_u64(file_size, kLocalMappedDirectIoAlignment));
    if (probe_begin + kLocalMappedDirectIoAlignment > probe_limit) {
      continue;
    }

    const int fd = ::open(segment.path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd < 0) {
      return LocalMappedDirectProbeResult{
          .attempted = true,
          .supported = false,
          .error_number = errno,
          .status = absl::StrCat("open_failed:", segment.path.string()),
      };
    }
    absl::Cleanup close_fd = [fd] { ::close(fd); };

    void* buffer = nullptr;
    const int rc = ::posix_memalign(&buffer, kLocalMappedDirectIoAlignment, kLocalMappedDirectIoAlignment);
    if (rc != 0) {
      return LocalMappedDirectProbeResult{
          .attempted = true,
          .supported = false,
          .error_number = rc,
          .status = "posix_memalign_failed",
      };
    }
    absl::Cleanup free_buffer = [buffer] { std::free(buffer); };

    auto got_or = pread_fully(fd, probe_begin, buffer, static_cast<size_t>(kLocalMappedDirectIoAlignment));
    if (!got_or.ok()) {
      int err = 0;
      if (absl::IsInternal(got_or.status()) || absl::IsInvalidArgument(got_or.status()) ||
          absl::IsFailedPrecondition(got_or.status()) || absl::IsUnavailable(got_or.status())) {
        err = errno;
      }
      return LocalMappedDirectProbeResult{
          .attempted = true,
          .supported = false,
          .error_number = err,
          .status = got_or.status().ToString(),
      };
    }
    if (*got_or != kLocalMappedDirectIoAlignment) {
      return LocalMappedDirectProbeResult{
          .attempted = true,
          .supported = false,
          .status = absl::StrCat("short_read:", *got_or),
      };
    }
    return LocalMappedDirectProbeResult{
        .attempted = true,
        .supported = true,
        .bytes = kLocalMappedDirectIoAlignment,
        .status = "ok",
    };
  }
  return LocalMappedDirectProbeResult{
      .attempted = false,
      .supported = false,
      .status = "no_aligned_probe_range",
  };
}

LocalMappedSafetensorsAutoIoDecision decision_with_direct_probe(
    bool use_direct,
    double page_cache_residency,
    const LocalMappedDirectProbeResult& probe,
    std::string reason) {
  return LocalMappedSafetensorsAutoIoDecision{
      .use_direct_aligned_edges = use_direct,
      .page_cache_residency_ratio = page_cache_residency,
      .direct_probe_attempted = probe.attempted,
      .direct_probe_supported = probe.supported,
      .direct_probe_bytes = probe.bytes,
      .direct_probe_errno = probe.error_number,
      .direct_probe_status = probe.status,
      .reason = std::move(reason),
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
  bool all_direct_friendly = true;
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
    const auto fs_type = static_cast<uint64_t>(fs.f_type);
    if (is_memory_fs_type(fs_type)) {
      return LocalMappedSafetensorsAutoIoDecision{
          .use_direct_aligned_edges = false,
          .reason = "filesystem_memory_buffered",
      };
    }
    all_direct_friendly = all_direct_friendly && is_direct_friendly_fs_type(fs_type);
  }
  double page_cache_residency = 0.0;
  auto page_cache_residency_or = estimate_source_page_cache_residency(segments);
  if (!page_cache_residency_or.ok()) {
    if (all_direct_friendly) {
      return page_cache_residency_or.status();
    }
    const auto probe = probe_direct_source_capability(segments);
    if (probe.supported) {
      return decision_with_direct_probe(
          true, -1.0, probe, "page_cache_residency_unavailable_direct_probe_supported");
    }
    return decision_with_direct_probe(
        false, -1.0, probe, "page_cache_residency_unavailable_direct_probe_unsupported_buffered");
  }
  page_cache_residency = *page_cache_residency_or;
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
    if (!all_direct_friendly) {
      const auto direct_probe = probe_direct_source_capability(segments);
      if (!direct_probe.supported) {
        return LocalMappedSafetensorsAutoIoDecision{
            .use_direct_aligned_edges = false,
            .page_cache_residency_ratio = page_cache_residency,
            .buffered_probe_bytes = probe.bytes,
            .buffered_probe_sec = probe.sec,
            .buffered_probe_gib_per_sec = probe.gib_per_sec,
            .direct_probe_attempted = direct_probe.attempted,
            .direct_probe_supported = direct_probe.supported,
            .direct_probe_bytes = direct_probe.bytes,
            .direct_probe_errno = direct_probe.error_number,
            .direct_probe_status = direct_probe.status,
            .reason = "page_cache_resident_probe_slow_direct_probe_unsupported_buffered",
        };
      }
      return LocalMappedSafetensorsAutoIoDecision{
          .use_direct_aligned_edges = true,
          .page_cache_residency_ratio = page_cache_residency,
          .buffered_probe_bytes = probe.bytes,
          .buffered_probe_sec = probe.sec,
          .buffered_probe_gib_per_sec = probe.gib_per_sec,
          .direct_probe_attempted = direct_probe.attempted,
          .direct_probe_supported = direct_probe.supported,
          .direct_probe_bytes = direct_probe.bytes,
          .direct_probe_errno = direct_probe.error_number,
          .direct_probe_status = direct_probe.status,
          .reason = "page_cache_resident_probe_slow_direct_probe_supported",
      };
    }
    return LocalMappedSafetensorsAutoIoDecision{
        .use_direct_aligned_edges = true,
        .page_cache_residency_ratio = page_cache_residency,
        .buffered_probe_bytes = probe.bytes,
        .buffered_probe_sec = probe.sec,
        .buffered_probe_gib_per_sec = probe.gib_per_sec,
        .reason = "page_cache_resident_probe_slow_direct",
    };
  }
  if (!all_direct_friendly) {
    const auto direct_probe = probe_direct_source_capability(segments);
    if (direct_probe.supported) {
      return decision_with_direct_probe(true, page_cache_residency, direct_probe, "direct_probe_cold_or_partial_direct");
    }
    return decision_with_direct_probe(
        false, page_cache_residency, direct_probe, "direct_probe_unsupported_buffered");
  }
  return LocalMappedSafetensorsAutoIoDecision{
      .use_direct_aligned_edges = true,
      .page_cache_residency_ratio = page_cache_residency,
      .reason = "page_cache_cold_or_partial_direct",
  };
}

void log_local_mapped_safetensors_auto_decision(
    absl::Span<const loader::SharedSafetensorsSegment> segments,
    const LocalMappedSafetensorsAutoIoDecision& decision) {
  LOG(INFO) << "local_mapped_safetensors_auto source_bytes=" << total_source_bytes(segments)
            << " page_cache_residency_ratio=" << decision.page_cache_residency_ratio
            << " buffered_probe_bytes=" << decision.buffered_probe_bytes
            << " buffered_probe_sec=" << decision.buffered_probe_sec
            << " buffered_probe_gib_per_sec=" << decision.buffered_probe_gib_per_sec
            << " direct_probe_attempted=" << decision.direct_probe_attempted
            << " direct_probe_supported=" << decision.direct_probe_supported
            << " direct_probe_bytes=" << decision.direct_probe_bytes
            << " direct_probe_errno=" << decision.direct_probe_errno
            << " direct_probe_status=" << decision.direct_probe_status
            << " decision=" << (decision.use_direct_aligned_edges ? "direct_aligned_edges" : "buffered")
            << " reason=" << decision.reason;
}

absl::StatusOr<StrategyConfig> resolve_local_mapped_safetensors_auto_strategy(
    absl::Span<const loader::SharedSafetensorsSegment> segments,
    const StrategyConfig& strategy,
    bool log_decision) {
  using IoMode = StrategyConfig::LocalMappedSafetensorsIoMode;
  if (strategy.local_mapped_safetensors_io_mode != IoMode::kAutoByFilesystem) {
    return strategy;
  }
  LocalMappedSafetensorsAutoIoDecision decision;
  TC_ASSIGN_OR_RETURN(decision, choose_auto_local_mapped_safetensors_io(segments));
  if (log_decision) {
    log_local_mapped_safetensors_auto_decision(segments, decision);
  }
  StrategyConfig resolved = strategy;
  resolved.local_mapped_safetensors_io_mode =
      decision.use_direct_aligned_edges ? IoMode::kDirectAlignedEdges : IoMode::kBuffered;
  return resolved;
}

class DirectAlignedSafetensorsSource final : public loader::SeekableSource {
 public:
  enum class PinnedWindowFallbackReason : std::uint8_t {
    kNone,
    kUnalignedHostBuffer,
    kOutsideSegment,
    kCrossSegment,
    kFileEdge,
    kCapacity,
  };

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

  absl::StatusOr<size_t> read_at_for_pinned_window(
      uint64_t offset,
      char* dst,
      size_t read_bytes,
      size_t h2d_bytes,
      size_t dst_capacity,
      PinnedWindowFallbackReason* fallback_reason = nullptr) {
    if (fallback_reason != nullptr) {
      *fallback_reason = PinnedWindowFallbackReason::kNone;
    }
    auto unimplemented = [&](PinnedWindowFallbackReason reason, std::string message) -> absl::StatusOr<size_t> {
      if (fallback_reason != nullptr) {
        *fallback_reason = reason;
      }
      return absl::UnimplementedError(std::move(message));
    };
    if (read_bytes == 0) {
      return static_cast<size_t>(0);
    }
    if (!is_aligned_address(dst, kLocalMappedDirectIoAlignment)) {
      return unimplemented(
          PinnedWindowFallbackReason::kUnalignedHostBuffer,
          "source-window direct pinned read requires an aligned host buffer");
    }
    if (offset >= total_bytes_ || read_bytes > total_bytes_ - offset || read_bytes > h2d_bytes) {
      return absl::OutOfRangeError("source-window direct pinned read exceeds source or target bounds");
    }
    Segment* segment = find_segment_for_offset(offset);
    if (segment == nullptr) {
      return unimplemented(
          PinnedWindowFallbackReason::kOutsideSegment,
          "source-window direct pinned read starts outside a safetensors segment");
    }
    const uint64_t segment_end = segment->segment.base_offset + segment->segment.data_size;
    if (read_bytes > segment_end - offset) {
      return unimplemented(
          PinnedWindowFallbackReason::kCrossSegment,
          "source-window direct pinned read crosses safetensors file segments");
    }
    if (segment->direct_fd < 0) {
      return absl::FailedPreconditionError("direct safetensors source was not initialized");
    }
    const uint64_t file_begin = segment->segment.data_start + (offset - segment->segment.base_offset);
    const uint64_t file_end = file_begin + static_cast<uint64_t>(read_bytes);
    const uint64_t direct_begin = align_down_u64(file_begin, kLocalMappedDirectIoAlignment);
    const uint64_t direct_end = align_up_u64(file_end, kLocalMappedDirectIoAlignment);
    if (direct_end > segment->direct_file_floor) {
      return unimplemented(
          PinnedWindowFallbackReason::kFileEdge, "source-window direct pinned read reaches an unaligned file edge");
    }
    const size_t prefix_bytes = static_cast<size_t>(file_begin - direct_begin);
    const size_t direct_bytes = static_cast<size_t>(direct_end - direct_begin);
    if (direct_bytes > dst_capacity || prefix_bytes > dst_capacity || h2d_bytes > dst_capacity - prefix_bytes) {
      return unimplemented(
          PinnedWindowFallbackReason::kCapacity,
          "source-window direct pinned read does not fit in the host staging buffer");
    }
    size_t got = 0;
    TC_ASSIGN_OR_RETURN(got, pread_fully(segment->direct_fd, direct_begin, dst, direct_bytes));
    if (got != direct_bytes) {
      return absl::OutOfRangeError(
          absl::StrCat(
              "short source-window O_DIRECT pinned read: got=", got, " want=", direct_bytes, " offset=", direct_begin));
    }
    if (read_bytes < h2d_bytes) {
      std::memset(dst + prefix_bytes + read_bytes, 0, h2d_bytes - read_bytes);
    }
    return prefix_bytes;
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
  StrategyConfig resolved_strategy;
  TC_ASSIGN_OR_RETURN(
      resolved_strategy, resolve_local_mapped_safetensors_auto_strategy(segments, strategy, /*log_decision=*/true));
  IoMode mode = resolved_strategy.local_mapped_safetensors_io_mode;

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
  auto it = std::upper_bound(
      ranges.begin(), ranges.end(), begin, [](uint64_t value, const ByteRange& range) { return value < range.begin; });
  if (it != ranges.begin()) {
    --it;
  }
  uint64_t cursor = begin;
  for (; it != ranges.end(); ++it) {
    if (it->end <= cursor) {
      continue;
    }
    if (it->begin > cursor) {
      return false;
    }
    cursor = std::max<uint64_t>(cursor, it->end);
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
  auto it = std::upper_bound(
      ranges.begin(), ranges.end(), begin, [](uint64_t value, const ByteRange& range) { return value < range.begin; });
  if (it != ranges.begin()) {
    const auto& previous = *(it - 1);
    if (previous.end > begin) {
      return true;
    }
  }
  if (it != ranges.end() && it->begin < end) {
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

void insert_byte_range_sorted_merged(ByteRange incoming, std::vector<ByteRange>* ranges) {
  if (ranges == nullptr || incoming.end <= incoming.begin) {
    return;
  }
  auto it =
      std::lower_bound(ranges->begin(), ranges->end(), incoming.begin, [](const ByteRange& range, uint64_t value) {
        return range.end < value;
      });
  while (it != ranges->end() && it->begin <= incoming.end) {
    incoming.begin = std::min(incoming.begin, it->begin);
    incoming.end = std::max(incoming.end, it->end);
    it = ranges->erase(it);
  }
  ranges->insert(it, incoming);
}

void insert_destination_spans_as_sorted_ranges(absl::Span<const TensorByteSpan> spans, std::vector<ByteRange>* ranges) {
  for (const auto& span : spans) {
    if (span.length == 0) {
      continue;
    }
    insert_byte_range_sorted_merged(ByteRange{.begin = span.offset, .end = span.offset + span.length}, ranges);
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

absl::StatusOr<MappedTensorJobRuntime> build_local_partial_dim1_compact_tensor_job(
    const ParsedMappedParticipant& participant,
    const RepresentationWorkItem& item,
    absl::Span<const TensorByteSpan> dst_spans) {
  if (dst_spans.size() != 1 || item.sources.size() != 1) {
    return absl::InvalidArgumentError("local partial dim1 compact tensor job requires one compact destination span");
  }
  const auto& fragment = item.sources.front().fragment;
  if (fragment.source_spec.shape.size() != 2 ||
      !is_row_major_contiguous(fragment.source_spec.shape, fragment.source_spec.stride) ||
      !is_row_major_contiguous(item.dst_spec.shape, item.dst_spec.stride)) {
    return absl::InvalidArgumentError(
        "local partial dim1 compact tensor job requires contiguous 2D source and contiguous destination");
  }
  if (fragment.source_spec.element_size == 0 || fragment.source_spec.element_size != item.dst_spec.element_size ||
      fragment.source_spec.dtype != item.dst_spec.dtype) {
    return absl::InvalidArgumentError("local partial dim1 compact tensor job has incompatible dtypes");
  }

  auto src_rows_or = coordinate_axis_range_or_full(fragment.source_range, fragment.source_spec, /*dim=*/0, "source");
  if (!src_rows_or.ok()) {
    return src_rows_or.status();
  }
  auto src_cols_or = coordinate_axis_range_or_full(fragment.source_range, fragment.source_spec, /*dim=*/1, "source");
  if (!src_cols_or.ok()) {
    return src_cols_or.status();
  }
  const uint64_t row_count = static_cast<uint64_t>(src_rows_or->end - src_rows_or->start);
  const uint64_t col_count = static_cast<uint64_t>(src_cols_or->end - src_cols_or->start);
  if (row_count == 0 || col_count == 0) {
    return absl::InvalidArgumentError("local partial dim1 compact tensor job requires a non-empty source slice");
  }
  auto selected_elements_or = checked_mul_u64(row_count, col_count, "local partial dim1 compact selected elements");
  if (!selected_elements_or.ok()) {
    return selected_elements_or.status();
  }
  auto selected_bytes_or =
      checked_mul_u64(*selected_elements_or, fragment.source_spec.element_size, "local partial dim1 compact bytes");
  if (!selected_bytes_or.ok()) {
    return selected_bytes_or.status();
  }
  if (*selected_bytes_or == 0 || dst_spans.front().length != *selected_bytes_or) {
    return absl::InvalidArgumentError("local partial dim1 compact tensor job source/destination byte sizes differ");
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
          .dst_offset = dst_spans.front().offset - item.dst_spec.logical_offset,
          .dst_size_bytes = dst_spans.front().length,
          .dst_row_stride_bytes = 0,
          .kind = RankTensorSlice::Kind::kRect2D,
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
    } else if (item.partition_kind == WorkPartitionKind::kUnknown) {
      runtime_job_or = build_local_partial_replicated_tensor_job(participant, item, absl::MakeSpan(dst_spans));
      if (runtime_job_or.ok()) {
        accepted_kind = AcceptedKind::kReplicated;
      }
      if (!runtime_job_or.ok() && absl::IsInvalidArgument(runtime_job_or.status())) {
        runtime_job_or = build_local_partial_dim1_compact_tensor_job(participant, item, absl::MakeSpan(dst_spans));
        if (runtime_job_or.ok()) {
          accepted_kind = AcceptedKind::kRect2D;
        }
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
    insert_destination_spans_as_sorted_ranges(absl::MakeSpan(dst_spans), &handled_ranges);
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

struct TargetPieceGeometry {
  uint64_t length{0};
  uint64_t src_offset{0};
};

struct CopyPiece {
  const std::uint8_t* src_ptr{nullptr};
  std::uint8_t* dst_ptr{nullptr};
  uint64_t length{0};
};

struct PackedRemotePiece {
  uint64_t pack_offset{0};
  std::uint8_t* dst_ptr{nullptr};
  uint64_t length{0};
};

absl::StatusOr<TargetPieceGeometry> resolve_target_piece_geometry_by_storage_index(
    absl::Span<const TargetStorageSpan> storage_spans,
    uint32_t storage_index,
    uint64_t logical_offset,
    uint64_t length);

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

bool can_merge_packed_remote_piece(const PackedRemotePiece& prev, const PackedRemotePiece& next) {
  return prev.length > 0 && next.length > 0 && prev.dst_ptr != nullptr && next.dst_ptr != nullptr &&
      prev.pack_offset + prev.length == next.pack_offset && prev.dst_ptr + prev.length == next.dst_ptr;
}

void append_merged_packed_remote_piece(std::vector<PackedRemotePiece>& pieces, PackedRemotePiece piece) {
  if (piece.length == 0 || piece.dst_ptr == nullptr) {
    return;
  }
  if (!pieces.empty() && can_merge_packed_remote_piece(pieces.back(), piece)) {
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

absl::StatusOr<TargetPiece> resolve_target_piece_by_storage_index(
    const ParsedMappedParticipant& participant,
    uint32_t storage_index,
    uint64_t logical_offset,
    uint64_t length) {
  auto geometry_or =
      resolve_target_piece_geometry_by_storage_index(participant.storage_spans, storage_index, logical_offset, length);
  if (!geometry_or.ok()) {
    return geometry_or.status();
  }
  const auto& geometry = *geometry_or;
  if (length == 0) {
    return TargetPiece{
        .dst_ptr =
            gsl::not_null<std::uint8_t*>{
                participant.storage_spans.empty() ? reinterpret_cast<std::uint8_t*>(1)
                                                  : participant.storage_spans.front().base_ptr.get()},
        .length = geometry.length,
        .src_offset = geometry.src_offset,
    };
  }
  const auto& span = participant.storage_spans[storage_index];
  return TargetPiece{
      .dst_ptr = gsl::not_null<std::uint8_t*>{span.base_ptr.get() + (logical_offset - span.base_offset)},
      .length = geometry.length,
      .src_offset = geometry.src_offset,
  };
}

absl::StatusOr<TargetPieceGeometry> resolve_target_piece_geometry_by_storage_index(
    absl::Span<const TargetStorageSpan> storage_spans,
    uint32_t storage_index,
    uint64_t logical_offset,
    uint64_t length) {
  if (length == 0) {
    return TargetPieceGeometry{
        .length = 0,
        .src_offset = 0,
    };
  }
  if (logical_offset > std::numeric_limits<uint64_t>::max() - length) {
    return absl::OutOfRangeError("source-window target logical range overflows");
  }
  if (storage_index >= storage_spans.size()) {
    return absl::InvalidArgumentError("source-window target storage index is outside target layout");
  }
  const auto& span = storage_spans[storage_index];
  if (span.length > std::numeric_limits<uint64_t>::max() - span.base_offset) {
    return absl::OutOfRangeError("source-window target storage span overflows");
  }
  const uint64_t logical_end = logical_offset + length;
  const uint64_t span_end = span.base_offset + span.length;
  if (logical_offset < span.base_offset || logical_end > span_end) {
    return absl::InvalidArgumentError("source-window target logical range is outside planned storage span");
  }
  return TargetPieceGeometry{
      .length = length,
      .src_offset = 0,
  };
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

bool local_mapped_rect2d_uses_packed_rows(uint64_t row_bytes, uint64_t col_bytes, bool enable_packed_rect2d_row_reads) {
  return enable_packed_rect2d_row_reads && col_bytes < row_bytes;
}

uint64_t local_mapped_rect2d_host_row_bytes(
    uint64_t row_bytes,
    uint64_t col_bytes,
    bool enable_packed_rect2d_row_reads) {
  if (local_mapped_rect2d_uses_packed_rows(row_bytes, col_bytes, enable_packed_rect2d_row_reads)) {
    return col_bytes;
  }
  return row_bytes;
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
    size_t host_buffer_bytes,
    bool enable_packed_rect2d_row_reads) {
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
  const uint64_t host_row_bytes =
      local_mapped_rect2d_host_row_bytes(*row_bytes_or, *col_bytes_or, enable_packed_rect2d_row_reads);
  if (host_row_bytes > host_buffer_bytes) {
    return absl::FailedPreconditionError("local mapped rect2d tensor row exceeds pinned buffer size");
  }
  return absl::OkStatus();
}

absl::Status validate_local_mapped_tensor_jobs_admission(
    const std::vector<MappedTensorJobRuntime>& jobs,
    int device_id,
    size_t host_buffer_bytes,
    bool enable_packed_rect2d_row_reads) {
  for (const auto& job : jobs) {
    if (job.job.slices.size() != 1) {
      return absl::InvalidArgumentError("local mapped tensor admission requires one slice");
    }
    const auto& slice = job.job.slices.front();
    if (slice.kind == RankTensorSlice::Kind::kRect2D) {
      TC_RETURN_IF_ERROR(validate_local_mapped_rect2d_tensor_job_admission(
          job, device_id, host_buffer_bytes, enable_packed_rect2d_row_reads));
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
    case LocalMappedTensorTask::Kind::kRect2D:
    case LocalMappedTensorTask::Kind::kRect2DPacked: {
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
      if (task.kind == LocalMappedTensorTask::Kind::kRect2DPacked && task.source_row_stride_bytes < width_bytes) {
        return absl::InvalidArgumentError("local mapped packed rect2d tensor task has invalid source row stride");
      }
      break;
    }
  }
  tasks->push_back(task);
  return absl::OkStatus();
}

absl::StatusOr<size_t> read_local_mapped_tensor_task_into_buffer(
    loader::SeekableSource& source,
    const LocalMappedTensorTask& task,
    char* host_ptr) {
  if (host_ptr == nullptr) {
    return absl::InvalidArgumentError("local mapped tensor task read got a null host buffer");
  }
  if (task.read_bytes > std::numeric_limits<size_t>::max()) {
    return absl::OutOfRangeError("local mapped tensor task exceeds host size_t limit");
  }
  if (task.kind != LocalMappedTensorTask::Kind::kRect2DPacked) {
    return source.read_at(task.source_offset, host_ptr, static_cast<size_t>(task.read_bytes));
  }
  if (task.rows == 0 || task.src_pitch_bytes == 0 || task.source_row_stride_bytes == 0 ||
      task.read_bytes != task.rows * task.src_pitch_bytes) {
    return absl::InvalidArgumentError("local mapped packed rect2d tensor task has invalid read geometry");
  }
  size_t total = 0;
  for (uint64_t row = 0; row < task.rows; ++row) {
    auto row_source_delta_or = checked_mul_u64(row, task.source_row_stride_bytes, "packed rect2d source row delta");
    if (!row_source_delta_or.ok()) {
      return row_source_delta_or.status();
    }
    auto source_offset_or =
        checked_add_u64(task.source_offset, *row_source_delta_or, "packed rect2d source row offset");
    if (!source_offset_or.ok()) {
      return source_offset_or.status();
    }
    char* row_ptr = host_ptr + row * task.src_pitch_bytes;
    auto got_or = source.read_at(*source_offset_or, row_ptr, static_cast<size_t>(task.src_pitch_bytes));
    if (!got_or.ok()) {
      return got_or.status();
    }
    if (*got_or != static_cast<size_t>(task.src_pitch_bytes)) {
      return absl::OutOfRangeError(
          absl::StrCat(
              "short read in local mapped packed rect2d tensor task: got=", *got_or, " want=", task.src_pitch_bytes));
    }
    total += *got_or;
  }
  return total;
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
    bool enable_packed_rect2d_row_reads,
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
  const bool use_packed_rows =
      local_mapped_rect2d_uses_packed_rows(row_bytes, col_bytes, enable_packed_rect2d_row_reads);
  const uint64_t host_row_bytes = use_packed_rows ? col_bytes : row_bytes;
  if (host_row_bytes > host_buffer_bytes) {
    return absl::ResourceExhaustedError("local mapped rect2d tensor row exceeds pinned buffer size");
  }

  const uint64_t src_col_bytes = slice.src_col_start * job.job.source.elem_size;
  const uint64_t dst_pitch_bytes = slice.dst_row_stride_bytes == 0 ? col_bytes : slice.dst_row_stride_bytes;
  const uint64_t rows_per_chunk = std::max<uint64_t>(1, static_cast<uint64_t>(host_buffer_bytes) / host_row_bytes);
  auto* dst_base = static_cast<std::uint8_t*>(job.destinations.front().gpu_ptr) + slice.dst_offset;

  for (uint64_t row = 0; row < slice.row_count; row += rows_per_chunk) {
    const uint64_t chunk_rows = std::min<uint64_t>(rows_per_chunk, slice.row_count - row);
    auto chunk_bytes_or = checked_mul_u64(chunk_rows, host_row_bytes, "local mapped rect2d tensor task read bytes");
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
    if (task_source_or.ok() && use_packed_rows) {
      task_source_or =
          checked_add_u64(*task_source_or, src_col_bytes, "local mapped packed rect2d tensor task source offset");
    }
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
            .kind = use_packed_rows ? LocalMappedTensorTask::Kind::kRect2DPacked : LocalMappedTensorTask::Kind::kRect2D,
            .source_offset = *task_source_or,
            .read_bytes = *chunk_bytes_or,
            .dst_ptr = dst_base + *dst_row_offset_or,
            .dst_bytes = *dst_bytes_or,
            .src_col_offset_bytes = use_packed_rows ? 0 : src_col_bytes,
            .src_pitch_bytes = host_row_bytes,
            .source_row_stride_bytes = row_bytes,
            .dst_pitch_bytes = dst_pitch_bytes,
            .rows = chunk_rows,
        },
        &result->tasks));
  }

  auto full_read_bytes_or =
      checked_mul_u64(slice.row_count, host_row_bytes, "local mapped rect2d tensor total read bytes");
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
    size_t host_buffer_bytes,
    bool enable_packed_rect2d_row_reads) {
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
      TC_RETURN_IF_ERROR(
          append_local_mapped_rect2d_tensor_tasks(job, host_buffer_bytes, enable_packed_rect2d_row_reads, &result));
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
    case LocalMappedTensorTask::Kind::kRect2D:
    case LocalMappedTensorTask::Kind::kRect2DPacked: {
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
    bool enable_packed_rect2d_row_reads,
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
  auto task_build_or =
      build_local_mapped_tensor_tasks(jobs, device_id, host_buffer_bytes, enable_packed_rect2d_row_reads);
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
        const auto read_start = std::chrono::steady_clock::now();
        auto got_or = read_local_mapped_tensor_task_into_buffer(source, task, host_ptr);
        producer_read_ns.fetch_add(elapsed_ns_since(read_start), std::memory_order_relaxed);
        if (!got_or.ok()) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(got_or.status());
          break;
        }
        if (*got_or != static_cast<size_t>(task.read_bytes)) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(
              absl::OutOfRangeError(
                  absl::StrCat(
                      "short read in local mapped tensor streaming: got=", *got_or, " want=", task.read_bytes)));
          break;
        }
        const auto mark_ready_start = std::chrono::steady_clock::now();
        const absl::Status ready_status =
            session_spb->mark_chunk_ready(slot_id, task_index, static_cast<size_t>(task.read_bytes));
        producer_mark_ready_ns.fetch_add(elapsed_ns_since(mark_ready_start), std::memory_order_relaxed);
        if (!ready_status.ok()) {
          (void)session_spb->abort_producer_slot(slot_id);
          record_failure(ready_status);
          break;
        }
        producer_tasks.fetch_add(1, std::memory_order_relaxed);
        producer_bytes.fetch_add(task.read_bytes, std::memory_order_relaxed);
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
        absl::StatusOr<size_t> got_or;
        if (task.kind == SourceOrderedTask::Kind::kTensor) {
          if (task.index >= tensor_task_build.tasks.size()) {
            got_or = absl::InternalError("local mapped source-ordered tensor task index out of range");
          } else {
            got_or = read_local_mapped_tensor_task_into_buffer(source, tensor_task_build.tasks[task.index], host_ptr);
          }
        } else {
          got_or = source.read_at(task.source_offset, host_ptr, bytes);
        }
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
  double partial_tensor_build_sec = 0.0;
  {
    absl::Span<const ByteRange> initially_handled = initially_handled_dst_ranges_by_rank.empty()
        ? absl::Span<const ByteRange>()
        : absl::MakeSpan(initially_handled_dst_ranges_by_rank.front());
    const auto partial_tensor_build_start = std::chrono::steady_clock::now();
    auto partial_tensor_job_build_or = build_local_mapped_partial_tensor_jobs(participant, initially_handled);
    if (!partial_tensor_job_build_or.ok()) {
      return partial_tensor_job_build_or.status();
    }
    partial_tensor_build_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - partial_tensor_build_start).count();
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

  TC_RETURN_IF_ERROR(validate_local_mapped_tensor_jobs_admission(
      local_tensor_jobs, participant.device_id, chunk_bytes, options.strategy_config.enable_packed_rect2d_row_reads));
  TC_RETURN_IF_ERROR(validate_local_mapped_concat_jobs_admission(concat_job_build.jobs, chunk_bytes));

  const size_t streaming_buffer_chunks = std::max<size_t>(1, options.streaming_buffer_chunks);
  LocalMappedTensorExecutionStats tensor_stats;
  LocalMappedConcatExecutionStats concat_stats;
  bool source_ordered_local_mapped = false;
  double source_ordered_exec_sec = 0.0;
  if (allow_source_ordered_for_mapped(options.strategy_config) && !local_tensor_jobs.empty() &&
      !concat_job_build.jobs.empty()) {
    const auto tensor_task_build_start = std::chrono::steady_clock::now();
    auto tensor_task_build_or = build_local_mapped_tensor_tasks(
        local_tensor_jobs, participant.device_id, chunk_bytes, options.strategy_config.enable_packed_rect2d_row_reads);
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
        options.strategy_config.enable_packed_rect2d_row_reads,
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
            << " partial_tensor_job_build=" << partial_tensor_build_sec << "s"
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

runtime::ingestion::strategy::CollectiveExecutionMetrics source_window_metrics_from_summary(
    const runtime::ingestion::strategy::SourceWindowCollectiveCandidateSummary& summary) {
  runtime::ingestion::strategy::CollectiveExecutionMetrics metrics;
  metrics.unique_source_bytes = summary.source_window_group_disk_read_bytes;
  metrics.peer_transfer_bytes = summary.source_window_peer_transfer_bytes;
  metrics.peak_temporary_bytes = summary.source_window_rank_read_bytes_max;
  metrics.batch_count = summary.source_window_window_count;
  metrics.dedup_saving_bytes = 0;
  metrics.source_window_group_disk_read_bytes = summary.source_window_group_disk_read_bytes;
  metrics.source_window_rank_read_bytes_max = summary.source_window_rank_read_bytes_max;
  metrics.source_window_local_rank_read_bytes_max = summary.source_window_local_rank_read_bytes_max;
  metrics.source_window_rank_read_saving_bytes = summary.source_window_rank_read_saving_bytes;
  metrics.source_window_unique_payload_bytes = summary.source_window_unique_payload_bytes;
  metrics.source_window_target_write_bytes = summary.source_window_target_write_bytes;
  metrics.source_window_peer_transfer_bytes = summary.source_window_peer_transfer_bytes;
  metrics.source_window_peer_useful_bytes = summary.source_window_peer_useful_bytes;
  metrics.source_window_peer_waste_bytes = summary.source_window_peer_waste_bytes;
  metrics.source_window_scatter_op_count = summary.source_window_scatter_op_count;
  metrics.source_window_window_count = summary.source_window_window_count;
  metrics.source_window_read_amplification_x1000 = summary.source_window_read_amplification_x1000;
  metrics.source_window_distribution_mode = std::string(
      runtime::ingestion::strategy::source_window_collective_distribution_mode_name(summary.distribution_mode));
  return metrics;
}

bool source_window_strict_mode(const SourceWindowCollectiveConfig& config) {
  return config.selection_mode == runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kStrict;
}

SourceWindowCollectiveMappedTargetLoadResult source_window_unhandled_or_strict_failure(
    const SourceWindowCollectiveConfig& config,
    std::string reason,
    absl::Status status = absl::OkStatus()) {
  if (source_window_strict_mode(config)) {
    return {
        .handled = true,
        .status = status.ok() ? absl::FailedPreconditionError(reason) : std::move(status),
        .skip_reason = std::move(reason),
    };
  }
  return {
      .handled = false,
      .status = std::move(status),
      .skip_reason = std::move(reason),
  };
}

std::vector<::tensorcast::store::replica::TargetStorageSpan> to_source_window_target_storage_spans(
    absl::Span<const TargetStorageSpan> storage_spans) {
  std::vector<::tensorcast::store::replica::TargetStorageSpan> projected;
  projected.reserve(storage_spans.size());
  for (const auto& span : storage_spans) {
    projected.push_back(
        ::tensorcast::store::replica::TargetStorageSpan{
            .base_offset = span.base_offset,
            .length = span.length,
            .base_ptr = span.base_ptr.get(),
        });
  }
  return projected;
}

absl::StatusOr<SourceWindowCollectiveGroupInput> build_source_window_collective_group_input(
    const loading::CollectiveLoadGroupHint& group,
    const std::vector<SourceWindowMappedParticipant>& participants,
    const CollectiveMappedTargetLoadOptions& options) {
  if (participants.empty()) {
    return absl::FailedPreconditionError("source-window group has no participants");
  }
  std::string artifact_id = participants.front().artifact_id;
  std::string source_index_digest = participants.front().source_index_digest;
  std::shared_ptr<const loader::DiskArtifactContext> disk_context = participants.front().disk_context;
  std::vector<SourceWindowCollectiveMemberInput> members;
  members.reserve(participants.size());
  for (const auto& participant : participants) {
    if (participant.artifact_id != artifact_id) {
      return absl::FailedPreconditionError("source-window artifact mismatch across participants");
    }
    if (participant.disk_context == nullptr) {
      return absl::FailedPreconditionError("source-window disk context missing");
    }
    if (disk_context != nullptr && participant.disk_context->artifact_path() != disk_context->artifact_path()) {
      return absl::FailedPreconditionError("source-window disk context mismatch across participants");
    }
    if (participant.source_index_digest != source_index_digest) {
      return absl::FailedPreconditionError("source-window source index digest mismatch");
    }
    if (participant.work_plan == nullptr || participant.target_layout == nullptr) {
      return absl::FailedPreconditionError("source-window participant missing work plan or target layout");
    }
    members.push_back(
        SourceWindowCollectiveMemberInput{
            .rank = static_cast<uint32_t>(participant.rank),
            .device_id = participant.device_id,
            .work_plan_ref = participant.work_plan,
            .target_layout_ref = participant.target_layout,
            .storage_spans = to_source_window_target_storage_spans(participant.storage_spans),
            .prepared_realization = participant.prepared_realization,
        });
  }
  return SourceWindowCollectiveGroupInput{
      .group = group,
      .disk_context = std::move(disk_context),
      .source_index_digest = std::move(source_index_digest),
      .members = std::move(members),
      .config = source_window_collective_config_from_strategy(options.strategy_config),
  };
}

SourceWindowCollectiveMappedTargetLoadResult source_window_group_rejected_result(
    const SourceWindowCollectiveConfig& config,
    std::string reason) {
  const std::string skip_reason = absl::StrCat("source_window_group_rejected:", reason);
  if (source_window_strict_mode(config)) {
    return {
        .handled = true,
        .status = absl::FailedPreconditionError(skip_reason),
        .skip_reason = skip_reason,
    };
  }
  return {
      .handled = false,
      .status = absl::OkStatus(),
      .skip_reason = skip_reason,
  };
}

SourceWindowCollectiveMappedTargetLoadResult source_window_group_rejected_result(
    const SourceWindowCollectiveConfig& config,
    const SourceWindowCollectivePlan& plan,
    std::string reason) {
  SourceWindowCollectiveMappedTargetLoadResult result = source_window_group_rejected_result(config, std::move(reason));
  result.metrics = source_window_metrics_from_summary(plan.summary);
  result.plan_hash = plan.plan_hash;
  return result;
}

SourceWindowCollectiveMappedTargetLoadResult source_window_runtime_unavailable_result(
    const SourceWindowCollectiveConfig& config,
    const SourceWindowCollectivePlan& plan,
    std::string reason) {
  SourceWindowCollectiveMappedTargetLoadResult result;
  result.metrics = source_window_metrics_from_summary(plan.summary);
  result.plan_hash = plan.plan_hash;
  result.skip_reason = absl::StrCat("source_window_runtime_unavailable:", reason);
  if (source_window_strict_mode(config)) {
    result.handled = true;
    result.status = absl::FailedPreconditionError(result.skip_reason);
    return result;
  }
  result.handled = false;
  result.status = absl::OkStatus();
  return result;
}

SourceWindowCollectiveMappedTargetLoadResult source_window_runtime_failure_result(
    const SourceWindowCollectivePlan& plan,
    absl::Status status) {
  SourceWindowCollectiveMappedTargetLoadResult result;
  result.handled = true;
  result.status = std::move(status);
  result.metrics = source_window_metrics_from_summary(plan.summary);
  result.plan_hash = plan.plan_hash;
  result.skip_reason = absl::StrCat("source_window_runtime_failed:", result.status.message());
  return result;
}

SourceWindowCollectiveMappedTargetLoadResult source_window_runtime_success_result(
    const SourceWindowCollectivePlan& plan,
    runtime::ingestion::strategy::CollectiveExecutionMetrics metrics) {
  SourceWindowCollectiveMappedTargetLoadResult result;
  result.handled = true;
  result.status = absl::OkStatus();
  result.metrics = std::move(metrics);
  result.plan_hash = plan.plan_hash;
  return result;
}

const RepresentationWorkPlan& source_window_request_work_plan(
    const SourceWindowCollectiveMappedTargetLoadRequest& request) {
  if (request.representation_work_plan_ref != nullptr) {
    return *request.representation_work_plan_ref;
  }
  return request.representation_work_plan;
}

std::shared_ptr<const RepresentationWorkPlan> source_window_request_work_plan_ref(
    const SourceWindowCollectiveMappedTargetLoadRequest& request) {
  if (request.representation_work_plan_ref != nullptr) {
    return request.representation_work_plan_ref;
  }
  return std::make_shared<const RepresentationWorkPlan>(request.representation_work_plan);
}

const loading::IntoTargetLayout& source_window_request_target_layout(
    const SourceWindowCollectiveMappedTargetLoadRequest& request) {
  if (request.target_layout_ref != nullptr) {
    return *request.target_layout_ref;
  }
  return request.target_layout;
}

std::shared_ptr<const loading::IntoTargetLayout> source_window_request_target_layout_ref(
    const SourceWindowCollectiveMappedTargetLoadRequest& request) {
  if (request.target_layout_ref != nullptr) {
    return request.target_layout_ref;
  }
  return std::make_shared<const loading::IntoTargetLayout>(request.target_layout);
}

struct SourceWindowPreparedRealizationFactStats {
  uint64_t member_count{0};
  uint64_t group_key_unique{0};
  uint64_t member_key_unique{0};
  uint64_t realization_plan_hash_unique{0};
  uint64_t target_layout_template_hash_unique{0};
  uint64_t target_index_hash_unique{0};
};

SourceWindowPreparedRealizationFactStats source_window_prepared_realization_fact_stats(
    const SourceWindowCollectiveGroupInput& input) {
  std::unordered_set<std::string> group_keys;
  std::unordered_set<std::string> member_keys;
  std::unordered_set<std::string> realization_plan_hashes;
  std::unordered_set<std::string> target_layout_template_hashes;
  std::unordered_set<std::string> target_index_hashes;
  uint64_t member_count = 0;
  for (const auto& member : input.members) {
    if (!member.prepared_realization.has_value()) {
      continue;
    }
    member_count++;
    const auto& facts = *member.prepared_realization;
    if (!facts.group_key.empty()) {
      group_keys.insert(facts.group_key);
    }
    if (!facts.member_key.empty()) {
      member_keys.insert(facts.member_key);
    }
    if (!facts.realization_plan_hash.empty()) {
      realization_plan_hashes.insert(facts.realization_plan_hash);
    }
    if (!facts.target_layout_template_hash.empty()) {
      target_layout_template_hashes.insert(facts.target_layout_template_hash);
    }
    if (!facts.target_index_hash.empty()) {
      target_index_hashes.insert(facts.target_index_hash);
    }
  }
  return SourceWindowPreparedRealizationFactStats{
      .member_count = member_count,
      .group_key_unique = group_keys.size(),
      .member_key_unique = member_keys.size(),
      .realization_plan_hash_unique = realization_plan_hashes.size(),
      .target_layout_template_hash_unique = target_layout_template_hashes.size(),
      .target_index_hash_unique = target_index_hashes.size(),
  };
}

std::vector<ParsedMappedParticipant> source_window_parsed_mapped_participants(
    const std::vector<SourceWindowMappedParticipant>& participants) {
  std::vector<ParsedMappedParticipant> parsed;
  parsed.reserve(participants.size());
  for (const auto& participant : participants) {
    // Source-window scatter only needs resolved target storage spans. Avoid
    // copying the large per-rank RepresentationWorkPlan into the runtime
    // participant view; the group plan has already consumed it.
    parsed.push_back(
        ParsedMappedParticipant{
            .artifact_id = participant.artifact_id,
            .rank = participant.rank,
            .device_id = participant.device_id,
            .disk_context = participant.disk_context,
            .storage_spans = participant.storage_spans,
        });
  }
  return parsed;
}

bool source_window_consumer_span_overlaps_chunk(
    const SourceWindowCollectiveConsumerSpan& span,
    uint64_t chunk_start,
    uint64_t chunk_end) {
  uint64_t span_begin = span.source_window_start;
  uint64_t span_end = span.source_window_end;
  if (span_end <= span_begin) {
    span_begin = span.source_offset;
    span_end = span.source_offset + span.length;
  }
  return span_end > chunk_start && span_begin < chunk_end;
}

bool source_window_window_uses_local_only(const SourceWindowCollectiveWindow& window) {
  return window.distribution_mode == runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kLocalOnly;
}

struct SourceWindowRuntimeChunkSizing {
  size_t stripe_buffer_bytes{0};
  size_t max_collective_chunk_bytes{0};
  size_t max_stripe_bytes{0};
};

absl::StatusOr<SourceWindowRuntimeChunkSizing> source_window_runtime_chunk_sizing(
    size_t stripe_buffer_bytes,
    size_t world_size,
    runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode distribution_mode) {
  if (stripe_buffer_bytes == 0) {
    return absl::InvalidArgumentError("source-window runtime stripe buffer bytes must be non-zero");
  }
  if (world_size == 0) {
    return absl::InvalidArgumentError("source-window runtime world size must be non-zero");
  }
  const bool consumer_routed =
      distribution_mode == runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted;
  if (consumer_routed) {
    if (stripe_buffer_bytes > std::numeric_limits<size_t>::max() / world_size) {
      return absl::InvalidArgumentError("source-window runtime consumer-routed chunk sizing overflows");
    }
    return SourceWindowRuntimeChunkSizing{
        .stripe_buffer_bytes = stripe_buffer_bytes,
        .max_collective_chunk_bytes = stripe_buffer_bytes * world_size,
        .max_stripe_bytes = stripe_buffer_bytes,
    };
  }

  const size_t max_collective_chunk_bytes = (stripe_buffer_bytes / world_size) * world_size;
  if (max_collective_chunk_bytes == 0) {
    return absl::InvalidArgumentError("source-window runtime chunk bytes must be at least world_size");
  }
  return SourceWindowRuntimeChunkSizing{
      .stripe_buffer_bytes = stripe_buffer_bytes,
      .max_collective_chunk_bytes = max_collective_chunk_bytes,
      .max_stripe_bytes = max_collective_chunk_bytes / world_size,
  };
}

absl::StatusOr<std::vector<SourceWindowRuntimeChunk>> build_source_window_runtime_chunks(
    const SourceWindowCollectivePlan& plan,
    size_t world_size,
    size_t configured_chunk_bytes,
    size_t max_collective_chunk_bytes,
    uint64_t* unfiltered_consumer_span_refs,
    uint64_t* prefiltered_consumer_span_refs) {
  std::vector<SourceWindowRuntimeChunk> runtime_chunks;
  if (unfiltered_consumer_span_refs != nullptr) {
    *unfiltered_consumer_span_refs = 0;
  }
  if (prefiltered_consumer_span_refs != nullptr) {
    *prefiltered_consumer_span_refs = 0;
  }
  for (const auto& window : plan.windows) {
    if (window.owner_rank >= world_size) {
      return absl::InvalidArgumentError("source-window owner rank out of bounds");
    }
    if (window.end < window.start) {
      return absl::InvalidArgumentError("source-window has an invalid interval");
    }
    const bool local_only_window = source_window_window_uses_local_only(window);
    const size_t max_runtime_chunk_bytes = local_only_window ? configured_chunk_bytes : max_collective_chunk_bytes;
    uint64_t chunk_start = window.start;
    while (chunk_start < window.end) {
      const uint64_t chunk_end =
          std::min<uint64_t>(window.end, chunk_start + static_cast<uint64_t>(max_runtime_chunk_bytes));
      const size_t chunk_len = static_cast<size_t>(chunk_end - chunk_start);
      const size_t stripe_bytes = local_only_window ? chunk_len : (chunk_len + world_size - 1) / world_size;
      const size_t gathered_bytes = local_only_window ? chunk_len : stripe_bytes * world_size;
      const size_t max_stripe_bytes = local_only_window ? configured_chunk_bytes : max_collective_chunk_bytes / world_size;
      if (stripe_bytes == 0 || stripe_bytes > max_stripe_bytes || gathered_bytes > max_collective_chunk_bytes) {
        return absl::InternalError("source-window collective stripe sizing exceeded staging buffers");
      }
      std::vector<const SourceWindowCollectiveConsumerSpan*> chunk_consumer_spans;
      chunk_consumer_spans.reserve(window.consumer_spans.size());
      for (const auto& span : window.consumer_spans) {
        if (source_window_consumer_span_overlaps_chunk(span, chunk_start, chunk_end)) {
          chunk_consumer_spans.push_back(&span);
        }
      }
      if (unfiltered_consumer_span_refs != nullptr) {
        *unfiltered_consumer_span_refs += window.consumer_spans.size();
      }
      if (prefiltered_consumer_span_refs != nullptr) {
        *prefiltered_consumer_span_refs += chunk_consumer_spans.size();
      }
      runtime_chunks.push_back(
          SourceWindowRuntimeChunk{
              .window = &window,
              .consumer_spans = std::move(chunk_consumer_spans),
              .chunk_start = chunk_start,
              .chunk_end = chunk_end,
              .chunk_len = chunk_len,
              .stripe_bytes = stripe_bytes,
              .gathered_bytes = gathered_bytes,
          });
      chunk_start = chunk_end;
    }
  }
  return runtime_chunks;
}

absl::StatusOr<SourceWindowRoutedChunkProgram> build_source_window_routed_chunk_program(
    const SourceWindowRuntimeChunk& chunk,
    absl::Span<const ParsedMappedParticipant> mapped_participants,
    size_t world_size,
    size_t max_stripe_bytes) {
  SourceWindowRoutedChunkProgram program;
  program.compiled = true;
  program.local_descriptors_by_rank.resize(world_size);
  program.pack_descriptors_by_producer.resize(world_size);
  std::vector<std::vector<uint64_t>> pack_offsets(world_size, std::vector<uint64_t>(world_size, 0));

  auto producer_for_source_offset = [&](uint64_t source_offset) -> absl::StatusOr<size_t> {
    if (source_offset < chunk.chunk_start || source_offset >= chunk.chunk_end) {
      return absl::OutOfRangeError("source-window routed source offset is outside chunk");
    }
    const uint64_t chunk_offset = source_offset - chunk.chunk_start;
    size_t producer = static_cast<size_t>(chunk_offset / chunk.stripe_bytes);
    if (producer >= world_size) {
      producer = world_size - 1;
    }
    return producer;
  };

  auto append_local_descriptor = [&](RoutedDescriptorTemplate descriptor) {
    if (descriptor.row_bytes == 0 || descriptor.row_count == 0) {
      return;
    }
    auto& descriptors = program.local_descriptors_by_rank[descriptor.rank];
    if (!descriptors.empty()) {
      auto& previous = descriptors.back();
      if (previous.row_count == 1 && descriptor.row_count == 1 && previous.rank == descriptor.rank &&
          previous.src_offset + previous.row_bytes == descriptor.src_offset &&
          previous.target.rank == descriptor.target.rank &&
          previous.target.storage_index == descriptor.target.storage_index &&
          previous.target.logical_offset + previous.row_bytes == descriptor.target.logical_offset) {
        previous.row_bytes += descriptor.row_bytes;
        previous.source_stride_bytes = previous.row_bytes;
        previous.target_stride_bytes = previous.row_bytes;
        return;
      }
    }
    descriptors.push_back(descriptor);
  };

  auto append_packed_remote_piece = [&](RoutedPackedRemotePieceTemplate piece) {
    if (piece.length == 0) {
      return;
    }
    if (!program.packed_remote_pieces.empty()) {
      auto& previous = program.packed_remote_pieces.back();
      if (previous.producer == piece.producer && previous.consumer == piece.consumer &&
          previous.pack_offset + previous.length == piece.pack_offset && previous.target.rank == piece.target.rank &&
          previous.target.storage_index == piece.target.storage_index &&
          previous.target.logical_offset + previous.length == piece.target.logical_offset) {
        previous.length += piece.length;
        return;
      }
    }
    program.packed_remote_pieces.push_back(piece);
  };

  auto append_direct_remote_piece = [&](RoutedDirectRemotePieceTemplate piece) {
    if (piece.length == 0) {
      return;
    }
    if (!program.direct_remote_pieces.empty()) {
      auto& previous = program.direct_remote_pieces.back();
      if (previous.producer == piece.producer && previous.consumer == piece.consumer &&
          previous.src_offset + previous.length == piece.src_offset && previous.target.rank == piece.target.rank &&
          previous.target.storage_index == piece.target.storage_index &&
          previous.target.logical_offset + previous.length == piece.target.logical_offset) {
        previous.length += piece.length;
        return;
      }
    }
    program.direct_remote_pieces.push_back(piece);
  };

  auto append_packed_linear_piece = [&](size_t producer,
                                        uint32_t consumer_rank,
                                        uint64_t src_offset,
                                        uint32_t storage_index,
                                        uint64_t target_logical_offset,
                                        uint64_t length) -> absl::StatusOr<bool> {
    if (producer == consumer_rank || length == 0 || length > std::numeric_limits<size_t>::max()) {
      return false;
    }
    uint64_t& pack_offset = pack_offsets[producer][consumer_rank];
    if (pack_offset > max_stripe_bytes || length > max_stripe_bytes - pack_offset) {
      return false;
    }
    program.pack_descriptors_by_producer[producer].push_back(
        RoutedPackDescriptorTemplate{
            .producer = producer,
            .consumer = consumer_rank,
            .src_offset = src_offset,
            .pack_offset = pack_offset,
            .row_bytes = length,
            .row_count = 1,
            .source_stride_bytes = length,
            .target_stride_bytes = length,
        });
    append_packed_remote_piece(
        RoutedPackedRemotePieceTemplate{
            .producer = producer,
            .consumer = consumer_rank,
            .pack_offset = pack_offset,
            .target =
                RoutedTargetRef{
                    .rank = consumer_rank,
                    .storage_index = storage_index,
                    .logical_offset = target_logical_offset,
                },
            .length = length,
        });
    pack_offset += length;
    program.pack_ops += 1;
    return true;
  };

  auto append_routed_linear_piece = [&](uint32_t consumer_rank,
                                        uint32_t storage_index,
                                        uint64_t source_begin,
                                        uint64_t target_logical_offset,
                                        uint64_t length) -> absl::Status {
    if (consumer_rank >= mapped_participants.size()) {
      return absl::InvalidArgumentError("source-window consumer rank out of bounds");
    }
    auto piece_or = resolve_target_piece_geometry_by_storage_index(
        absl::MakeConstSpan(mapped_participants[consumer_rank].storage_spans),
        storage_index,
        target_logical_offset,
        length);
    if (!piece_or.ok()) {
      return piece_or.status();
    }
    const auto& piece = *piece_or;
    program.target_storage_fast_path_pieces += 1;
    program.target_storage_fast_path_bytes += piece.length;
    uint64_t remaining = piece.length;
    uint64_t source_cursor = source_begin + piece.src_offset;
    uint64_t target_piece_offset = 0;
    while (remaining > 0) {
      auto producer_or = producer_for_source_offset(source_cursor);
      if (!producer_or.ok()) {
        return producer_or.status();
      }
      const size_t producer = *producer_or;
      const uint64_t producer_stripe_begin = static_cast<uint64_t>(producer) * chunk.stripe_bytes;
      const uint64_t producer_offset = source_cursor - chunk.chunk_start - producer_stripe_begin;
      const uint64_t producer_available = chunk.stripe_bytes - producer_offset;
      const uint64_t take =
          std::min<uint64_t>(remaining, std::min<uint64_t>(producer_available, chunk.chunk_end - source_cursor));
      if (take == 0) {
        return absl::InternalError("source-window routed linear split made no progress");
      }
      if (take > std::numeric_limits<size_t>::max()) {
        return absl::OutOfRangeError("source-window routed piece exceeds size_t limits");
      }
      const uint64_t target_cursor = target_logical_offset + target_piece_offset;
      if (producer == consumer_rank) {
        append_local_descriptor(
            RoutedDescriptorTemplate{
                .rank = consumer_rank,
                .src_offset = producer_offset,
                .target =
                    RoutedTargetRef{
                        .rank = consumer_rank,
                        .storage_index = storage_index,
                        .logical_offset = target_cursor,
                    },
                .row_bytes = take,
                .row_count = 1,
                .source_stride_bytes = take,
                .target_stride_bytes = take,
            });
        program.local_pieces += 1;
      } else {
        constexpr uint64_t kRoutedPackSmallPieceBytes = 1ULL << 20;
        const bool pair_already_packing = pack_offsets[producer][consumer_rank] > 0;
        auto packed_or = (pair_already_packing || take <= kRoutedPackSmallPieceBytes)
            ? append_packed_linear_piece(producer, consumer_rank, producer_offset, storage_index, target_cursor, take)
            : absl::StatusOr<bool>(false);
        if (!packed_or.ok()) {
          return packed_or.status();
        }
        if (!*packed_or) {
          append_direct_remote_piece(
              RoutedDirectRemotePieceTemplate{
                  .producer = producer,
                  .consumer = consumer_rank,
                  .src_offset = producer_offset,
                  .target =
                      RoutedTargetRef{
                          .rank = consumer_rank,
                          .storage_index = storage_index,
                          .logical_offset = target_cursor,
                      },
                  .length = take,
              });
        }
      }
      remaining -= take;
      source_cursor += take;
      target_piece_offset += take;
    }
    return absl::OkStatus();
  };

  auto append_remote_packed_2d = [&](size_t producer,
                                     uint32_t consumer_rank,
                                     uint32_t storage_index,
                                     uint64_t src_offset,
                                     uint64_t source_pitch,
                                     uint64_t dst_logical_offset,
                                     uint64_t width,
                                     uint64_t rows,
                                     uint64_t target_pitch) -> absl::StatusOr<bool> {
    if (producer == consumer_rank || rows <= 1 || width == 0 || target_pitch != width) {
      return false;
    }
    if (rows > std::numeric_limits<uint64_t>::max() / width) {
      return absl::OutOfRangeError("source-window routed packed bytes overflow");
    }
    const uint64_t packed_bytes = rows * width;
    if (packed_bytes == 0 || packed_bytes > max_stripe_bytes || packed_bytes > std::numeric_limits<size_t>::max() ||
        width > std::numeric_limits<size_t>::max() || source_pitch > std::numeric_limits<size_t>::max() ||
        rows > std::numeric_limits<size_t>::max()) {
      return false;
    }
    auto piece_or = resolve_target_piece_geometry_by_storage_index(
        absl::MakeConstSpan(mapped_participants[consumer_rank].storage_spans),
        storage_index,
        dst_logical_offset,
        packed_bytes);
    if (!piece_or.ok()) {
      return piece_or.status();
    }
    uint64_t& pack_offset = pack_offsets[producer][consumer_rank];
    if (pack_offset > max_stripe_bytes || packed_bytes > max_stripe_bytes - pack_offset) {
      return false;
    }
    program.pack_descriptors_by_producer[producer].push_back(
        RoutedPackDescriptorTemplate{
            .producer = producer,
            .consumer = consumer_rank,
            .src_offset = src_offset,
            .pack_offset = pack_offset,
            .row_bytes = width,
            .row_count = rows,
            .source_stride_bytes = source_pitch,
            .target_stride_bytes = width,
        });
    append_packed_remote_piece(
        RoutedPackedRemotePieceTemplate{
            .producer = producer,
            .consumer = consumer_rank,
            .pack_offset = pack_offset,
            .target =
                RoutedTargetRef{
                    .rank = consumer_rank,
                    .storage_index = storage_index,
                    .logical_offset = dst_logical_offset,
                },
            .length = packed_bytes,
        });
    program.target_storage_fast_path_pieces += 1;
    program.target_storage_fast_path_bytes += packed_bytes;
    pack_offset += packed_bytes;
    program.pack_ops += 1;
    program.deferred_2d_pack_ops += 1;
    return true;
  };

  for (const auto* span_ptr : chunk.consumer_spans) {
    const auto& span = *span_ptr;
    if (span.rank >= mapped_participants.size()) {
      return absl::InvalidArgumentError("source-window consumer rank out of bounds");
    }
    if (span.row_count > 1 && span.row_bytes > 0 && span.source_stride_bytes > 0 && span.target_stride_bytes > 0) {
      if (span.source_stride_bytes < span.row_bytes || span.target_stride_bytes < span.row_bytes) {
        return absl::InvalidArgumentError("source-window 2D routed span has invalid row strides");
      }
      uint64_t row = 0;
      if (chunk.chunk_start > span.source_offset) {
        row = std::min<uint64_t>(span.row_count, (chunk.chunk_start - span.source_offset) / span.source_stride_bytes);
        while (row < span.row_count &&
               span.source_offset + row * span.source_stride_bytes + span.row_bytes <= chunk.chunk_start) {
          ++row;
        }
      }
      while (row < span.row_count) {
        const uint64_t row_source_begin = span.source_offset + row * span.source_stride_bytes;
        if (row_source_begin >= chunk.chunk_end) {
          break;
        }
        const uint64_t row_source_end = row_source_begin + span.row_bytes;
        const uint64_t overlap_begin = std::max<uint64_t>(row_source_begin, chunk.chunk_start);
        const uint64_t overlap_end = std::min<uint64_t>(row_source_end, chunk.chunk_end);
        if (overlap_end <= overlap_begin) {
          ++row;
          continue;
        }
        const uint64_t copy_col_offset = overlap_begin - row_source_begin;
        const uint64_t copy_width = overlap_end - overlap_begin;
        uint64_t rows_in_run = 1;
        while (row + rows_in_run < span.row_count) {
          const uint64_t next_row_begin = span.source_offset + (row + rows_in_run) * span.source_stride_bytes;
          if (next_row_begin >= chunk.chunk_end) {
            break;
          }
          const uint64_t next_overlap_begin = std::max<uint64_t>(next_row_begin, chunk.chunk_start);
          const uint64_t next_overlap_end = std::min<uint64_t>(next_row_begin + span.row_bytes, chunk.chunk_end);
          if (next_overlap_end <= next_overlap_begin || next_overlap_begin - next_row_begin != copy_col_offset ||
              next_overlap_end - next_overlap_begin != copy_width) {
            break;
          }
          ++rows_in_run;
        }

        uint64_t run_row = 0;
        while (run_row < rows_in_run) {
          const uint64_t first_source =
              span.source_offset + (row + run_row) * span.source_stride_bytes + copy_col_offset;
          auto producer_or = producer_for_source_offset(first_source);
          if (!producer_or.ok()) {
            return producer_or.status();
          }
          const size_t producer = *producer_or;
          const uint64_t producer_stripe_begin = static_cast<uint64_t>(producer) * chunk.stripe_bytes;
          const uint64_t producer_offset = first_source - chunk.chunk_start - producer_stripe_begin;
          if (producer_offset + copy_width > chunk.stripe_bytes) {
            const uint64_t dst_logical_offset =
                span.target_offset + (row + run_row) * span.target_stride_bytes + copy_col_offset;
            TC_RETURN_IF_ERROR(append_routed_linear_piece(
                span.rank, span.storage_index, first_source, dst_logical_offset, copy_width));
            run_row += 1;
            continue;
          }

          uint64_t grouped_rows = 1;
          while (run_row + grouped_rows < rows_in_run) {
            const uint64_t next_source =
                span.source_offset + (row + run_row + grouped_rows) * span.source_stride_bytes + copy_col_offset;
            auto next_producer_or = producer_for_source_offset(next_source);
            if (!next_producer_or.ok() || *next_producer_or != producer) {
              break;
            }
            const uint64_t next_producer_offset = next_source - chunk.chunk_start - producer_stripe_begin;
            if (next_producer_offset + copy_width > chunk.stripe_bytes) {
              break;
            }
            ++grouped_rows;
          }

          const uint64_t dst_logical_offset =
              span.target_offset + (row + run_row) * span.target_stride_bytes + copy_col_offset;
          bool copied_as_group = false;
          if (producer == span.rank && grouped_rows > 1) {
            const uint64_t target_envelope_bytes = (grouped_rows - 1) * span.target_stride_bytes + copy_width;
            auto piece_or = resolve_target_piece_geometry_by_storage_index(
                absl::MakeConstSpan(mapped_participants[span.rank].storage_spans),
                span.storage_index,
                dst_logical_offset,
                target_envelope_bytes);
            if (!piece_or.ok()) {
              return piece_or.status();
            }
            if (copy_width <= std::numeric_limits<size_t>::max() &&
                span.source_stride_bytes <= std::numeric_limits<size_t>::max() &&
                span.target_stride_bytes <= std::numeric_limits<size_t>::max() &&
                grouped_rows <= std::numeric_limits<size_t>::max()) {
              append_local_descriptor(
                  RoutedDescriptorTemplate{
                      .rank = span.rank,
                      .src_offset = producer_offset,
                      .target =
                          RoutedTargetRef{
                              .rank = span.rank,
                              .storage_index = span.storage_index,
                              .logical_offset = dst_logical_offset,
                          },
                      .row_bytes = copy_width,
                      .row_count = grouped_rows,
                      .source_stride_bytes = span.source_stride_bytes,
                      .target_stride_bytes = span.target_stride_bytes,
                  });
              program.target_storage_fast_path_pieces += 1;
              program.target_storage_fast_path_bytes += copy_width * grouped_rows;
              program.local_2d_pieces += 1;
              program.local_pieces += 1;
              copied_as_group = true;
            }
          } else if (producer != span.rank && grouped_rows > 1) {
            auto packed_or = append_remote_packed_2d(
                producer,
                span.rank,
                span.storage_index,
                producer_offset,
                span.source_stride_bytes,
                dst_logical_offset,
                copy_width,
                grouped_rows,
                span.target_stride_bytes);
            if (!packed_or.ok()) {
              return packed_or.status();
            }
            copied_as_group = *packed_or;
          }
          if (!copied_as_group) {
            for (uint64_t offset_row = 0; offset_row < grouped_rows; ++offset_row) {
              const uint64_t row_source =
                  span.source_offset + (row + run_row + offset_row) * span.source_stride_bytes + copy_col_offset;
              const uint64_t row_target =
                  span.target_offset + (row + run_row + offset_row) * span.target_stride_bytes + copy_col_offset;
              TC_RETURN_IF_ERROR(
                  append_routed_linear_piece(span.rank, span.storage_index, row_source, row_target, copy_width));
            }
          }
          run_row += grouped_rows;
        }
        row += rows_in_run;
      }
    } else {
      const uint64_t span_end = span.source_offset + span.length;
      const uint64_t overlap_begin = std::max<uint64_t>(span.source_offset, chunk.chunk_start);
      const uint64_t overlap_end = std::min<uint64_t>(span_end, chunk.chunk_end);
      if (overlap_end <= overlap_begin) {
        continue;
      }
      const uint64_t dst_logical_offset = span.target_offset + (overlap_begin - span.source_offset);
      TC_RETURN_IF_ERROR(append_routed_linear_piece(
          span.rank, span.storage_index, overlap_begin, dst_logical_offset, overlap_end - overlap_begin));
    }
  }

  for (size_t producer = 0; producer < pack_offsets.size(); ++producer) {
    for (size_t consumer = 0; consumer < pack_offsets[producer].size(); ++consumer) {
      if (producer != consumer && pack_offsets[producer][consumer] > 0) {
        program.packed_transfers.push_back(
            RoutedPackedTransferTemplate{
                .producer = producer,
                .consumer = consumer,
                .bytes = pack_offsets[producer][consumer],
            });
      }
    }
  }
  program.packed_remote_piece_count = program.packed_remote_pieces.size();
  program.direct_remote_piece_count = program.direct_remote_pieces.size();
  return program;
}

absl::StatusOr<std::vector<SourceWindowRoutedChunkProgram>> build_source_window_routed_programs(
    absl::Span<const SourceWindowRuntimeChunk> runtime_chunks,
    absl::Span<const ParsedMappedParticipant> mapped_participants,
    size_t world_size,
    size_t max_stripe_bytes,
    uint32_t configured_build_threads,
    double* build_sec) {
  const auto build_start = std::chrono::steady_clock::now();
  std::vector<SourceWindowRoutedChunkProgram> programs(runtime_chunks.size());
  auto build_one = [&](size_t idx) -> absl::Status {
    const auto& runtime_chunk = runtime_chunks[idx];
    if (runtime_chunk.window == nullptr ||
        runtime_chunk.window->distribution_mode !=
            runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted) {
      return absl::OkStatus();
    }
    auto program_or =
        build_source_window_routed_chunk_program(runtime_chunk, mapped_participants, world_size, max_stripe_bytes);
    if (!program_or.ok()) {
      return program_or.status();
    }
    programs[idx] = std::move(*program_or);
    return absl::OkStatus();
  };

  const size_t worker_count =
      compiled_routed_program_build_thread_count(runtime_chunks.size(), configured_build_threads);
  if (worker_count <= 1) {
    for (size_t idx = 0; idx < runtime_chunks.size(); ++idx) {
      TC_RETURN_IF_ERROR(build_one(idx));
    }
  } else {
    std::atomic<size_t> next_idx{0};
    std::atomic<bool> failed{false};
    std::mutex status_mu;
    absl::Status first_status = absl::OkStatus();
    auto worker = [&]() {
      while (!failed.load(std::memory_order_acquire)) {
        const size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
        if (idx >= runtime_chunks.size()) {
          break;
        }
        absl::Status status = build_one(idx);
        if (!status.ok()) {
          {
            std::lock_guard<std::mutex> lock(status_mu);
            if (first_status.ok()) {
              first_status = std::move(status);
            }
          }
          failed.store(true, std::memory_order_release);
          break;
        }
      }
    };
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (size_t idx = 0; idx < worker_count; ++idx) {
      workers.emplace_back(worker);
    }
    for (auto& thread : workers) {
      thread.join();
    }
    if (!first_status.ok()) {
      return first_status;
    }
  }
  if (build_sec != nullptr) {
    *build_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();
  }
  return programs;
}

absl::StatusOr<runtime::ingestion::strategy::CollectiveExecutionMetrics> execute_source_window_collective_mapped(
    const SourceWindowCollectivePlan& plan,
    const std::vector<SourceWindowMappedParticipant>& participants,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options) {
  const auto total_start = std::chrono::steady_clock::now();
  if (participants.empty()) {
    return absl::InvalidArgumentError("source-window collective participants are empty");
  }
  double logged_total_sec = 0.0;
  bool logged_success = false;
  absl::Cleanup exit_profile = [&]() {
    const double exit_total_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
    LOG(INFO) << "tc_profile source_window_collective_mapped_target exit"
              << " artifact_id=" << participants.front().artifact_id
              << " group_id=" << plan.group.group_id
              << " logged_success=" << (logged_success ? 1 : 0)
              << " logged_total_sec=" << logged_total_sec
              << " exit_total_sec=" << exit_total_sec
              << " post_log_teardown_sec=" << (logged_success ? exit_total_sec - logged_total_sec : 0.0);
  };
  if (pinned_pool == nullptr) {
    return absl::FailedPreconditionError("pinned_pool_missing");
  }
  const bool use_full_window_all_gather = plan.distribution_mode ==
      runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kFullWindowAllGather;
  const bool use_consumer_routed =
      plan.distribution_mode == runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted;
  const bool use_local_only =
      plan.distribution_mode == runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kLocalOnly;
  const bool use_hybrid_window =
      plan.distribution_mode == runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kHybridWindow;
  if (!use_full_window_all_gather && !use_consumer_routed && !use_local_only && !use_hybrid_window) {
    return absl::UnimplementedError(
        "source-window runtime only supports full_window_all_gather, consumer_routed, hybrid_window, and local_only");
  }
  if (participants.front().disk_context == nullptr ||
      participants.front().disk_context->safetensors_segments().empty()) {
    return absl::InvalidArgumentError("source-window runtime requires safetensors segments");
  }
  if (pinned_pool->slice_bytes() == 0) {
    return absl::InvalidArgumentError("source-window runtime requires a non-empty pinned slice");
  }
  const size_t configured_chunk_bytes = static_cast<size_t>(std::min<uint64_t>(
      options.chunk_bytes == 0 ? pinned_pool->slice_bytes() : options.chunk_bytes, pinned_pool->slice_bytes()));
  if (configured_chunk_bytes == 0) {
    return absl::InvalidArgumentError("source-window runtime chunk bytes must be non-zero");
  }

  std::vector<int> device_ids;
  device_ids.reserve(participants.size());
  for (const auto& participant : participants) {
    if (participant.rank < 0 || static_cast<size_t>(participant.rank) >= participants.size()) {
      return absl::InvalidArgumentError("source-window participant rank out of bounds");
    }
    device_ids.push_back(participant.device_id);
  }
  const auto memory_entry = capture_source_window_device_memory(device_ids);
  log_source_window_collective_memory(
      "entry",
      participants.front().artifact_id,
      plan.group.group_id,
      memory_entry,
      &memory_entry);
  const auto clique_start = std::chrono::steady_clock::now();
  bool clique_cache_hit = false;
  auto clique_or = get_or_create_cached_clique(device_ids, &clique_cache_hit);
  if (!clique_or.ok()) {
    return clique_or.status();
  }
  auto clique = std::move(clique_or).value();
  std::weak_ptr<NcclClique> clique_identity = clique;
  bool evict_clique_on_exit = false;
  absl::Cleanup clique_cache_cleanup = [&clique, &device_ids, &clique_identity, &evict_clique_on_exit]() {
    if (!evict_clique_on_exit) {
      return;
    }
    // Drop this call's handle first; an idle cache entry then has use_count() == 1.
    clique.reset();
    try_evict_cached_clique_if_idle(device_ids, clique_identity, "source_window_collective_complete");
  };
  const auto clique_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - clique_start).count();
  log_source_window_collective_memory(
      "after_clique",
      participants.front().artifact_id,
      plan.group.group_id,
      capture_source_window_device_memory(device_ids),
      &memory_entry);
  absl::MutexLock clique_use_lock(&clique->use_mutex());

  PinnedBorrow host_pool;
  host_pool.pool = pinned_pool;
  const auto pinned_alloc_start = std::chrono::steady_clock::now();
  const std::string request_context =
      absl::StrCat("source_window_collective artifact_id=", participants.front().artifact_id);
  const size_t requested_pipeline_slots = std::max<size_t>(1, static_cast<size_t>(options.streaming_buffer_chunks));
  // Source-window keeps the GPU side ordered on the clique streams, but host
  // reads can safely run several chunks ahead as long as each chunk owns a
  // distinct pinned slot until its H2D copy has been issued and synchronized by
  // the downstream collective/scatter step.
  constexpr size_t kMaxSourceWindowPipelineSlots = 8;
  const size_t max_pipeline_slots = std::min<size_t>(requested_pipeline_slots, kMaxSourceWindowPipelineSlots);
  size_t active_pipeline_slots = 1;
  bool parallel_host_buffers = false;
  for (size_t slot_count = max_pipeline_slots; slot_count >= 1; --slot_count) {
    const size_t requested_parallel_host_bytes = configured_chunk_bytes * participants.size() * slot_count;
    if (pinned_pool->allocate(requested_parallel_host_bytes, host_pool.buffers, pinned_timeout, request_context) == 0 &&
        host_pool.buffers.size() >= participants.size() * slot_count) {
      parallel_host_buffers = true;
      active_pipeline_slots = slot_count;
      break;
    }
    if (!host_pool.buffers.empty()) {
      (void)pinned_pool->deallocate(host_pool.buffers);
      host_pool.buffers.clear();
    }
    if (slot_count == 1) {
      break;
    }
  }
  if (!parallel_host_buffers) {
    if (!host_pool.buffers.empty()) {
      (void)pinned_pool->deallocate(host_pool.buffers);
      host_pool.buffers.clear();
    }
    parallel_host_buffers = false;
    if (pinned_pool->allocate(configured_chunk_bytes, host_pool.buffers, pinned_timeout, request_context) != 0 ||
        host_pool.buffers.empty()) {
      return absl::ResourceExhaustedError("failed to allocate pinned buffer for source-window collective load");
    }
  }
  const auto pinned_alloc_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - pinned_alloc_start).count();
  log_source_window_collective_memory(
      "after_pinned_alloc",
      participants.front().artifact_id,
      plan.group.group_id,
      capture_source_window_device_memory(device_ids),
      &memory_entry);

  const size_t world_size = participants.size();
  auto chunk_sizing_or = source_window_runtime_chunk_sizing(configured_chunk_bytes, world_size, plan.distribution_mode);
  if (!chunk_sizing_or.ok()) {
    return chunk_sizing_or.status();
  }
  const auto chunk_sizing = *chunk_sizing_or;
  const size_t max_collective_chunk_bytes = chunk_sizing.max_collective_chunk_bytes;
  const size_t max_stripe_bytes = chunk_sizing.max_stripe_bytes;
  const bool has_consumer_routed_windows =
      std::any_of(plan.windows.begin(), plan.windows.end(), [](const SourceWindowCollectiveWindow& window) {
        return window.distribution_mode ==
            runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted;
      });

  log_source_window_collective_memory(
      "before_gpu_stage_alloc",
      participants.front().artifact_id,
      plan.group.group_id,
      capture_source_window_device_memory(device_ids),
      &memory_entry);
  const auto stage_alloc_start = std::chrono::steady_clock::now();
  std::vector<std::unique_ptr<common::memory::GpuDeviceMemory>> rank_send_stages;
  std::vector<std::unique_ptr<common::memory::GpuDeviceMemory>> rank_stages;
  std::vector<std::unique_ptr<common::memory::GpuDeviceMemory>> rank_batched_scatter_descriptors;

  struct BatchedScatterHostDescriptorSlot {
    SourceWindowBatchedScatterDescriptor* ptr{nullptr};
    cudaEvent_t ready_event{nullptr};
    int device_id{-1};
    bool in_flight{false};
  };

  std::vector<std::vector<BatchedScatterHostDescriptorSlot>> rank_batched_scatter_host_descriptor_slots;
  std::vector<size_t> rank_next_batched_scatter_host_descriptor_slot;
  double batched_scatter_descriptor_final_drain_sec = 0.0;
  std::vector<std::vector<std::unique_ptr<common::memory::GpuDeviceMemory>>> route_pack_stages;
  std::vector<std::vector<std::unique_ptr<common::memory::GpuDeviceMemory>>> route_recv_stages;
  rank_send_stages.reserve(participants.size());
  rank_stages.reserve(participants.size());
  rank_batched_scatter_descriptors.resize(participants.size());
  rank_batched_scatter_host_descriptor_slots.resize(participants.size());
  rank_next_batched_scatter_host_descriptor_slot.resize(participants.size(), 0);
  auto drain_batched_scatter_host_descriptor_slots = [&]() {
    const auto drain_start = std::chrono::steady_clock::now();
    for (auto& slots : rank_batched_scatter_host_descriptor_slots) {
      for (auto& slot : slots) {
        if (slot.ready_event != nullptr && slot.in_flight) {
          const absl::Status wait_status = tensorcast::cuda::event_synchronize(slot.ready_event);
          if (!wait_status.ok()) {
            LOG(WARNING) << "source-window batched scatter host descriptor slot synchronize failed: " << wait_status;
          }
          slot.in_flight = false;
        }
      }
    }
    batched_scatter_descriptor_final_drain_sec +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - drain_start).count();
  };
  auto cleanup_batched_scatter_host_descriptors = absl::Cleanup([&]() {
    drain_batched_scatter_host_descriptor_slots();
    for (auto& slots : rank_batched_scatter_host_descriptor_slots) {
      for (auto& slot : slots) {
        if (slot.ready_event != nullptr) {
          if (slot.device_id >= 0) {
            const absl::Status set_device_status = tensorcast::cuda::set_device(slot.device_id);
            if (!set_device_status.ok()) {
              LOG(WARNING) << "source-window batched scatter host descriptor slot set_device failed: "
                           << set_device_status;
            }
          }
          const absl::Status destroy_status = tensorcast::cuda::event_destroy(slot.ready_event);
          if (!destroy_status.ok()) {
            LOG(WARNING) << "source-window batched scatter host descriptor event destroy failed: " << destroy_status;
          }
          slot.ready_event = nullptr;
        }
        if (slot.ptr != nullptr) {
          const absl::Status free_status = tensorcast::cuda::free_host(slot.ptr);
          if (!free_status.ok()) {
            LOG(WARNING) << "source-window batched scatter host descriptor buffer free failed: " << free_status;
          }
          slot.ptr = nullptr;
        }
      }
    }
  });
  for (const auto& participant : participants) {
    auto send_stage = std::make_unique<common::memory::GpuDeviceMemory>();
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participant.device_id));
    TC_RETURN_IF_ERROR(send_stage->allocate(max_stripe_bytes, participant.device_id));
    rank_send_stages.push_back(std::move(send_stage));
    if (use_full_window_all_gather || use_local_only || use_hybrid_window) {
      auto stage = std::make_unique<common::memory::GpuDeviceMemory>();
      TC_RETURN_IF_ERROR(stage->allocate(configured_chunk_bytes, participant.device_id));
      rank_stages.push_back(std::move(stage));
    }
  }
  constexpr size_t kSourceWindowBatchedScatterDescriptorCapacity = 4096;
  constexpr size_t kSourceWindowBatchedScatterMinDescriptors = 2;
  const size_t batched_scatter_descriptor_buffer_bytes =
      kSourceWindowBatchedScatterDescriptorCapacity * source_window_batched_scatter_descriptor_bytes();
  const bool scatter_cuda_graph_enabled =
      options.strategy_config.enable_source_window_scatter_cuda_graph && !tensorcast::cuda::is_fake();
  const bool compiled_routed_program_enabled = options.strategy_config.enable_source_window_compiled_routed_program;
  const bool batched_scatter_kernel_requested = options.strategy_config.enable_source_window_batched_scatter_kernel;
  const bool batched_scatter_kernel_enabled = batched_scatter_kernel_requested;
  bool batched_scatter_descriptor_buffers_ready = batched_scatter_kernel_enabled && !participants.empty();
  constexpr size_t kSourceWindowBatchedScatterHostDescriptorSlots = 16;
  if (batched_scatter_descriptor_buffers_ready) {
    for (const auto& participant : participants) {
      const size_t rank = static_cast<size_t>(participant.rank);
      auto descriptors = std::make_unique<common::memory::GpuDeviceMemory>();
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participant.device_id));
      const absl::Status allocate_status =
          descriptors->allocate(batched_scatter_descriptor_buffer_bytes, participant.device_id);
      if (!allocate_status.ok()) {
        LOG(WARNING) << "source-window batched scatter descriptor buffer allocation failed on device "
                     << participant.device_id << "; falling back to cudaMemcpy scatter path: " << allocate_status;
        batched_scatter_descriptor_buffers_ready = false;
        for (auto& rank_descriptors : rank_batched_scatter_descriptors) {
          rank_descriptors.reset();
        }
        break;
      }
      rank_batched_scatter_descriptors[rank] = std::move(descriptors);
      auto& slots = rank_batched_scatter_host_descriptor_slots[rank];
      slots.resize(kSourceWindowBatchedScatterHostDescriptorSlots);
      for (auto& slot : slots) {
        slot.device_id = participant.device_id;
        void* host_descriptors = nullptr;
        const absl::Status host_allocate_status =
            tensorcast::cuda::malloc_host(&host_descriptors, batched_scatter_descriptor_buffer_bytes);
        if (!host_allocate_status.ok()) {
          LOG(WARNING) << "source-window batched scatter pinned host descriptor buffer allocation failed; "
                       << "falling back to cudaMemcpy scatter path: " << host_allocate_status;
          batched_scatter_descriptor_buffers_ready = false;
          for (auto& rank_descriptors : rank_batched_scatter_descriptors) {
            rank_descriptors.reset();
          }
          break;
        }
        slot.ptr = static_cast<SourceWindowBatchedScatterDescriptor*>(host_descriptors);
        const absl::Status event_status =
            tensorcast::cuda::event_create_with_flags(&slot.ready_event, cudaEventDisableTiming);
        if (!event_status.ok()) {
          LOG(WARNING) << "source-window batched scatter host descriptor event allocation failed; "
                       << "falling back to cudaMemcpy scatter path: " << event_status;
          batched_scatter_descriptor_buffers_ready = false;
          for (auto& rank_descriptors : rank_batched_scatter_descriptors) {
            rank_descriptors.reset();
          }
          break;
        }
      }
      if (!batched_scatter_descriptor_buffers_ready) {
        break;
      }
    }
  }
  if (has_consumer_routed_windows) {
    route_pack_stages.resize(participants.size());
    route_recv_stages.resize(participants.size());
    for (size_t producer = 0; producer < participants.size(); ++producer) {
      route_pack_stages[producer].resize(participants.size());
      route_recv_stages[producer].resize(participants.size());
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[producer].device_id));
      for (size_t consumer = 0; consumer < participants.size(); ++consumer) {
        if (producer == consumer) {
          continue;
        }
        auto pack_stage = std::make_unique<common::memory::GpuDeviceMemory>();
        TC_RETURN_IF_ERROR(pack_stage->allocate(max_stripe_bytes, participants[producer].device_id));
        route_pack_stages[producer][consumer] = std::move(pack_stage);
        auto recv_stage = std::make_unique<common::memory::GpuDeviceMemory>();
        TC_RETURN_IF_ERROR(recv_stage->allocate(max_stripe_bytes, participants[consumer].device_id));
        route_recv_stages[producer][consumer] = std::move(recv_stage);
      }
    }
  }
  const auto stage_alloc_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_alloc_start).count();
  log_source_window_collective_memory(
      "after_gpu_stage_alloc",
      participants.front().artifact_id,
      plan.group.group_id,
      capture_source_window_device_memory(device_ids),
      &memory_entry);

  std::vector<std::unique_ptr<loader::SeekableSource>> rank_sources;
  rank_sources.reserve(participants.size());
  const auto source_segments = absl::MakeSpan(participants.front().disk_context->safetensors_segments());
  StrategyConfig source_strategy;
  TC_ASSIGN_OR_RETURN(
      source_strategy,
      resolve_local_mapped_safetensors_auto_strategy(source_segments, options.strategy_config, /*log_decision=*/true));
  for (size_t rank = 0; rank < participants.size(); ++rank) {
    auto source_or = make_local_mapped_safetensors_source(source_segments, source_strategy);
    if (!source_or.ok()) {
      return source_or.status();
    }
    rank_sources.push_back(std::move(*source_or));
  }
  const std::vector<ParsedMappedParticipant> mapped_participants =
      source_window_parsed_mapped_participants(participants);

  uint64_t cuda_device_switches = 0;
  int current_cuda_device = -1;
  auto set_device_cached = [&](int device_id) -> absl::Status {
    if (current_cuda_device == device_id) {
      return absl::OkStatus();
    }
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_id));
    current_cuda_device = device_id;
    cuda_device_switches += 1;
    return absl::OkStatus();
  };
  auto synchronize_clique_all = [&]() -> absl::Status {
    absl::Status status = clique->synchronize_all();
    // NcclClique::synchronize_all() switches devices internally, so the local
    // cached device state is no longer authoritative afterwards.
    current_cuda_device = -1;
    return status;
  };

  double read_sec = 0.0;
  double read_job_sec = 0.0;
  double h2d_sec = 0.0;
  double collective_sec = 0.0;
  double collective_sync_sec = 0.0;
  double scatter_issue_sec = 0.0;
  double scatter_sync_sec = 0.0;
  uint64_t bytes_read = 0;
  uint64_t actual_peer_transfer_bytes = 0;
  uint64_t actual_scatter_ops = 0;
  uint64_t routed_pack_ops = 0;
  uint64_t routed_deferred_2d_pack_ops = 0;
  uint64_t routed_packed_pairs = 0;
  uint64_t routed_local_2d_pieces = 0;
  uint64_t routed_local_pieces = 0;
  uint64_t routed_remote_pieces = 0;
  uint64_t target_storage_fast_path_pieces = 0;
  uint64_t target_storage_fast_path_bytes = 0;
  uint64_t batched_scatter_kernel_launches = 0;
  uint64_t batched_scatter_kernel_descriptors = 0;
  uint64_t batched_scatter_fallback_launches = 0;
  uint64_t batched_scatter_fallback_descriptors = 0;
  uint64_t batched_scatter_capacity_fallback_chunks = 0;
  uint64_t batched_routed_pack_kernel_launches = 0;
  uint64_t batched_routed_pack_kernel_descriptors = 0;
  uint64_t batched_routed_pack_fallback_launches = 0;
  uint64_t batched_routed_pack_fallback_descriptors = 0;
  double batched_scatter_descriptor_build_sec = 0.0;
  double batched_scatter_descriptor_host_copy_sec = 0.0;
  double batched_scatter_descriptor_slot_wait_sec = 0.0;
  double batched_scatter_kernel_submit_sec = 0.0;
  double batched_scatter_fallback_submit_sec = 0.0;
  double routed_span_plan_sec = 0.0;
  double routed_pack_descriptor_build_sec = 0.0;
  double routed_pack_issue_sec = 0.0;
  double routed_local_descriptor_build_sec = 0.0;
  double routed_local_issue_sec = 0.0;
  double routed_remote_descriptor_build_sec = 0.0;
  double routed_remote_scatter_issue_sec = 0.0;
  double routed_compiled_program_build_sec = 0.0;
  size_t routed_compiled_program_build_threads = 0;
  double routed_compiled_program_key_sec = 0.0;
  double routed_compiled_program_lookup_sec = 0.0;
  double routed_compiled_program_wait_sec = 0.0;
  double routed_compiled_program_cache_store_sec = 0.0;
  bool routed_compiled_program_cache_eligible = false;
  bool routed_compiled_program_cache_hit = false;
  bool routed_compiled_program_cache_waited = false;
  bool routed_compiled_program_cache_size_mismatch = false;
  uint64_t routed_compiled_program_chunks = 0;
  uint64_t routed_compiled_program_local_descriptors = 0;
  uint64_t routed_compiled_program_pack_descriptors = 0;
  uint64_t routed_compiled_program_packed_remote_pieces = 0;
  uint64_t routed_compiled_program_direct_remote_pieces = 0;
  uint64_t scatter_cuda_graph_launches = 0;
  uint64_t scatter_cuda_graph_descriptors = 0;
  uint64_t scatter_cuda_graph_nodes = 0;
  uint64_t scatter_cuda_graph_fallback_chunks = 0;
  double scatter_cuda_graph_build_sec = 0.0;
  std::atomic<uint64_t> direct_pinned_read_attempts{0};
  std::atomic<uint64_t> direct_pinned_read_successes{0};
  std::atomic<uint64_t> direct_pinned_read_fallbacks{0};
  std::atomic<uint64_t> direct_pinned_read_success_bytes{0};
  std::atomic<uint64_t> direct_pinned_read_fallback_bytes{0};
  std::atomic<uint64_t> direct_pinned_fallback_unaligned_host{0};
  std::atomic<uint64_t> direct_pinned_fallback_outside_segment{0};
  std::atomic<uint64_t> direct_pinned_fallback_cross_segment{0};
  std::atomic<uint64_t> direct_pinned_fallback_file_edge{0};
  std::atomic<uint64_t> direct_pinned_fallback_capacity{0};
  size_t chunk_count = 0;

  auto chunk_uses_local_only = [&](const SourceWindowRuntimeChunk& chunk) {
    return chunk.window != nullptr && source_window_window_uses_local_only(*chunk.window);
  };
  auto chunk_uses_consumer_routed = [&](const SourceWindowRuntimeChunk& chunk) {
    return chunk.window != nullptr &&
        chunk.window->distribution_mode ==
        runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted;
  };

  std::vector<SourceWindowRuntimeChunk> runtime_chunks;
  uint64_t runtime_chunk_unfiltered_consumer_span_refs = 0;
  uint64_t runtime_chunk_prefiltered_consumer_span_refs = 0;
  TC_ASSIGN_OR_RETURN(
      runtime_chunks,
      build_source_window_runtime_chunks(
          plan,
          world_size,
          configured_chunk_bytes,
          max_collective_chunk_bytes,
          &runtime_chunk_unfiltered_consumer_span_refs,
          &runtime_chunk_prefiltered_consumer_span_refs));

  auto record_routed_program_stats = [&](const SourceWindowRoutedChunkProgram& program) {
    if (!program.compiled) {
      return;
    }
    routed_compiled_program_chunks += 1;
    for (const auto& descriptors : program.local_descriptors_by_rank) {
      routed_compiled_program_local_descriptors += descriptors.size();
    }
    for (const auto& descriptors : program.pack_descriptors_by_producer) {
      routed_compiled_program_pack_descriptors += descriptors.size();
    }
    routed_compiled_program_packed_remote_pieces += program.packed_remote_pieces.size();
    routed_compiled_program_direct_remote_pieces += program.direct_remote_pieces.size();
  };

  if (compiled_routed_program_enabled && has_consumer_routed_windows) {
    std::optional<std::string> routed_program_cache_key;
    {
      const auto key_start = std::chrono::steady_clock::now();
      routed_program_cache_key = source_window_routed_program_cache_key(
          participants.front().artifact_id,
          plan,
          absl::MakeConstSpan(participants),
          configured_chunk_bytes,
          max_collective_chunk_bytes,
          max_stripe_bytes);
      routed_compiled_program_key_sec =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - key_start).count();
    }
    routed_compiled_program_cache_eligible = routed_program_cache_key.has_value();

    std::vector<SourceWindowRoutedChunkProgram> cached_programs;
    bool routed_program_cache_build_reserved = false;
    bool routed_program_cache_build_completed = false;
    if (routed_compiled_program_cache_eligible) {
      const auto lookup_start = std::chrono::steady_clock::now();
      auto acquire_result =
          acquire_source_window_routed_program_cache(*routed_program_cache_key, runtime_chunks.size());
      routed_compiled_program_lookup_sec =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - lookup_start).count();
      routed_compiled_program_cache_hit = acquire_result.cache_hit;
      routed_compiled_program_cache_waited = acquire_result.waited;
      routed_compiled_program_wait_sec = acquire_result.wait_sec;
      routed_compiled_program_cache_size_mismatch = acquire_result.size_mismatch;
      routed_program_cache_build_reserved = acquire_result.reserved_build;
      cached_programs = std::move(acquire_result.programs);
    }
    auto cache_build_cleanup = absl::Cleanup([&]() {
      if (routed_compiled_program_cache_eligible && routed_program_cache_build_reserved &&
          !routed_program_cache_build_completed) {
        abandon_source_window_routed_program_cache_build(*routed_program_cache_key);
      }
    });

    if (!routed_compiled_program_cache_hit) {
      routed_compiled_program_build_threads = compiled_routed_program_build_thread_count(
          runtime_chunks.size(), options.strategy_config.source_window_compiled_program_build_threads);
      auto programs_or = build_source_window_routed_programs(
          absl::MakeConstSpan(runtime_chunks),
          absl::MakeConstSpan(mapped_participants),
          world_size,
          max_stripe_bytes,
          options.strategy_config.source_window_compiled_program_build_threads,
          &routed_compiled_program_build_sec);
      if (!programs_or.ok()) {
        return programs_or.status();
      }
      std::vector<SourceWindowRoutedChunkProgram> programs = std::move(*programs_or);
      if (routed_compiled_program_cache_eligible) {
        const auto store_start = std::chrono::steady_clock::now();
        if (routed_program_cache_build_reserved) {
          complete_source_window_routed_program_cache_build(*routed_program_cache_key, programs);
          routed_program_cache_build_completed = true;
        } else {
          store_source_window_routed_program_cache(*routed_program_cache_key, programs);
        }
        routed_compiled_program_cache_store_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - store_start).count();
      }
      for (size_t idx = 0; idx < runtime_chunks.size(); ++idx) {
        runtime_chunks[idx].routed_program = std::move(programs[idx]);
      }
    } else {
      for (size_t idx = 0; idx < runtime_chunks.size(); ++idx) {
        runtime_chunks[idx].routed_program = std::move(cached_programs[idx]);
      }
    }

    for (const auto& runtime_chunk : runtime_chunks) {
      record_routed_program_stats(runtime_chunk.routed_program);
    }
  }

  struct SourceWindowReadAheadResult {
    std::vector<absl::Status> statuses;
    double read_sec{0.0};
    double read_job_sec{0.0};
  };

  auto host_buffer_for_slot = [&](size_t slot, size_t rank) -> char* {
    return host_pool.buffers[slot * world_size + rank];
  };
  std::vector<size_t> host_h2d_offsets(active_pipeline_slots * world_size, 0);
  auto host_h2d_offset_for_slot = [&](size_t slot, size_t rank) -> size_t& {
    return host_h2d_offsets[slot * world_size + rank];
  };
  std::vector<std::atomic<uint64_t>> rank_read_bytes(world_size);
  std::vector<std::atomic<uint64_t>> rank_read_calls(world_size);
  std::vector<std::atomic<uint64_t>> rank_read_ns(world_size);
  std::vector<std::atomic<uint64_t>> rank_zero_fill_bytes(world_size);
  std::vector<std::atomic<uint64_t>> rank_zero_fill_calls(world_size);
  for (size_t rank = 0; rank < world_size; ++rank) {
    rank_read_bytes[rank].store(0, std::memory_order_relaxed);
    rank_read_calls[rank].store(0, std::memory_order_relaxed);
    rank_read_ns[rank].store(0, std::memory_order_relaxed);
    rank_zero_fill_bytes[rank].store(0, std::memory_order_relaxed);
    rank_zero_fill_calls[rank].store(0, std::memory_order_relaxed);
  }

  struct SourceWindowReaderSlotState {
    std::mutex mu;
    std::condition_variable cv;
    const SourceWindowRuntimeChunk* chunk{nullptr};
    size_t chunk_index{0};
    uint64_t generation{0};
    size_t remaining{0};
    bool stop{false};
    bool done{true};
    std::vector<absl::Status> statuses;
    std::chrono::steady_clock::time_point launch_time;
    double job_sec{0.0};
  };

  std::vector<std::unique_ptr<SourceWindowReaderSlotState>> reader_slots;
  reader_slots.reserve(active_pipeline_slots);
  for (size_t slot = 0; slot < active_pipeline_slots; ++slot) {
    auto state = std::make_unique<SourceWindowReaderSlotState>();
    state->statuses.resize(world_size);
    reader_slots.push_back(std::move(state));
  }
  std::vector<std::thread> reader_threads;
  reader_threads.reserve(active_pipeline_slots * world_size);
  for (size_t slot = 0; slot < active_pipeline_slots; ++slot) {
    for (size_t rank = 0; rank < world_size; ++rank) {
      reader_threads.emplace_back([&, slot, rank]() {
        auto& slot_state = *reader_slots[slot];
        uint64_t seen_generation = 0;
        while (true) {
          const SourceWindowRuntimeChunk* chunk = nullptr;
          {
            std::unique_lock<std::mutex> lock(slot_state.mu);
            slot_state.cv.wait(lock, [&]() { return slot_state.stop || slot_state.generation != seen_generation; });
            if (slot_state.stop) {
              return;
            }
            seen_generation = slot_state.generation;
            chunk = slot_state.chunk;
          }

          absl::Status status = absl::OkStatus();
          size_t h2d_offset = 0;
          if (chunk == nullptr) {
            status = absl::InternalError("source-window reader received empty chunk");
          } else {
            char* host_buffer = host_buffer_for_slot(slot, rank);
            const bool local_only_chunk = chunk_uses_local_only(*chunk);
            const bool rank_reads_chunk = !local_only_chunk || rank == chunk->window->owner_rank;
            const uint64_t stripe_start = local_only_chunk
                ? chunk->chunk_start
                : chunk->chunk_start + static_cast<uint64_t>(rank * chunk->stripe_bytes);
            const size_t read_len = rank_reads_chunk && stripe_start < chunk->chunk_end
                ? static_cast<size_t>(std::min<uint64_t>(chunk->stripe_bytes, chunk->chunk_end - stripe_start))
                : 0;
            if (read_len > 0) {
              const auto rank_read_start = std::chrono::steady_clock::now();
              auto* direct_source = dynamic_cast<DirectAlignedSafetensorsSource*>(rank_sources[rank].get());
              if (direct_source != nullptr) {
                direct_pinned_read_attempts.fetch_add(1, std::memory_order_relaxed);
                DirectAlignedSafetensorsSource::PinnedWindowFallbackReason fallback_reason =
                    DirectAlignedSafetensorsSource::PinnedWindowFallbackReason::kNone;
                auto h2d_offset_or = direct_source->read_at_for_pinned_window(
                    stripe_start, host_buffer, read_len, chunk->stripe_bytes, configured_chunk_bytes, &fallback_reason);
                if (h2d_offset_or.ok()) {
                  h2d_offset = *h2d_offset_or;
                  direct_pinned_read_successes.fetch_add(1, std::memory_order_relaxed);
                  direct_pinned_read_success_bytes.fetch_add(read_len, std::memory_order_relaxed);
                } else if (absl::IsUnimplemented(h2d_offset_or.status())) {
                  direct_pinned_read_fallbacks.fetch_add(1, std::memory_order_relaxed);
                  direct_pinned_read_fallback_bytes.fetch_add(read_len, std::memory_order_relaxed);
                  switch (fallback_reason) {
                    case DirectAlignedSafetensorsSource::PinnedWindowFallbackReason::kUnalignedHostBuffer:
                      direct_pinned_fallback_unaligned_host.fetch_add(1, std::memory_order_relaxed);
                      break;
                    case DirectAlignedSafetensorsSource::PinnedWindowFallbackReason::kOutsideSegment:
                      direct_pinned_fallback_outside_segment.fetch_add(1, std::memory_order_relaxed);
                      break;
                    case DirectAlignedSafetensorsSource::PinnedWindowFallbackReason::kCrossSegment:
                      direct_pinned_fallback_cross_segment.fetch_add(1, std::memory_order_relaxed);
                      break;
                    case DirectAlignedSafetensorsSource::PinnedWindowFallbackReason::kFileEdge:
                      direct_pinned_fallback_file_edge.fetch_add(1, std::memory_order_relaxed);
                      break;
                    case DirectAlignedSafetensorsSource::PinnedWindowFallbackReason::kCapacity:
                      direct_pinned_fallback_capacity.fetch_add(1, std::memory_order_relaxed);
                      break;
                    case DirectAlignedSafetensorsSource::PinnedWindowFallbackReason::kNone:
                      break;
                  }
                  status = read_exact(*rank_sources[rank], stripe_start, host_buffer, read_len);
                  if (status.ok() && read_len < chunk->stripe_bytes) {
                    std::memset(host_buffer + read_len, 0, chunk->stripe_bytes - read_len);
                    rank_zero_fill_bytes[rank].fetch_add(chunk->stripe_bytes - read_len, std::memory_order_relaxed);
                    rank_zero_fill_calls[rank].fetch_add(1, std::memory_order_relaxed);
                  }
                } else {
                  status = h2d_offset_or.status();
                }
              } else {
                status = read_exact(*rank_sources[rank], stripe_start, host_buffer, read_len);
                if (status.ok() && read_len < chunk->stripe_bytes) {
                  std::memset(host_buffer + read_len, 0, chunk->stripe_bytes - read_len);
                  rank_zero_fill_bytes[rank].fetch_add(chunk->stripe_bytes - read_len, std::memory_order_relaxed);
                  rank_zero_fill_calls[rank].fetch_add(1, std::memory_order_relaxed);
                }
              }
              const auto rank_read_elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                    std::chrono::steady_clock::now() - rank_read_start)
                                                    .count();
              rank_read_ns[rank].fetch_add(static_cast<uint64_t>(rank_read_elapsed_ns), std::memory_order_relaxed);
              rank_read_bytes[rank].fetch_add(read_len, std::memory_order_relaxed);
              rank_read_calls[rank].fetch_add(1, std::memory_order_relaxed);
            } else {
              std::memset(host_buffer, 0, chunk->stripe_bytes);
              rank_zero_fill_bytes[rank].fetch_add(chunk->stripe_bytes, std::memory_order_relaxed);
              rank_zero_fill_calls[rank].fetch_add(1, std::memory_order_relaxed);
            }
          }

          {
            std::lock_guard<std::mutex> lock(slot_state.mu);
            slot_state.statuses[rank] = std::move(status);
            host_h2d_offset_for_slot(slot, rank) = h2d_offset;
            if (slot_state.remaining > 0) {
              slot_state.remaining -= 1;
            }
            if (slot_state.remaining == 0) {
              slot_state.done = true;
              slot_state.job_sec =
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - slot_state.launch_time).count();
              slot_state.cv.notify_all();
            }
          }
        }
      });
    }
  }
  auto stop_reader_threads = absl::Cleanup([&]() {
    for (auto& slot_state_ptr : reader_slots) {
      auto& slot_state = *slot_state_ptr;
      {
        std::lock_guard<std::mutex> lock(slot_state.mu);
        slot_state.stop = true;
        slot_state.generation += 1;
      }
      slot_state.cv.notify_all();
    }
    for (auto& thread : reader_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  });

  auto launch_parallel_chunk_read = [&](size_t chunk_index) -> absl::Status {
    const size_t slot = chunk_index % active_pipeline_slots;
    auto& slot_state = *reader_slots[slot];
    {
      std::lock_guard<std::mutex> lock(slot_state.mu);
      if (!slot_state.done || slot_state.remaining != 0) {
        return absl::FailedPreconditionError("source-window read-ahead slot already has an in-flight job");
      }
      slot_state.chunk = &runtime_chunks[chunk_index];
      slot_state.chunk_index = chunk_index;
      slot_state.remaining = world_size;
      slot_state.done = false;
      slot_state.job_sec = 0.0;
      slot_state.launch_time = std::chrono::steady_clock::now();
      std::fill(slot_state.statuses.begin(), slot_state.statuses.end(), absl::OkStatus());
      slot_state.generation += 1;
    }
    slot_state.cv.notify_all();
    return absl::OkStatus();
  };

  auto wait_parallel_chunk_read = [&](size_t chunk_index) -> SourceWindowReadAheadResult {
    const size_t slot = chunk_index % active_pipeline_slots;
    auto& slot_state = *reader_slots[slot];
    const auto wait_start = std::chrono::steady_clock::now();
    SourceWindowReadAheadResult result;
    {
      std::unique_lock<std::mutex> lock(slot_state.mu);
      slot_state.cv.wait(lock, [&]() { return slot_state.done && slot_state.chunk_index == chunk_index; });
      result.statuses = slot_state.statuses;
      result.read_job_sec = slot_state.job_sec;
    }
    result.read_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start).count();
    return result;
  };

  auto issue_h2d_from_slot = [&](const SourceWindowRuntimeChunk& chunk, size_t slot) -> absl::Status {
    const auto step_start = std::chrono::steady_clock::now();
    for (size_t rank = 0; rank < world_size; ++rank) {
      TC_RETURN_IF_ERROR(set_device_cached(participants[rank].device_id));
      const size_t h2d_offset = host_h2d_offset_for_slot(slot, rank);
      TC_RETURN_IF_ERROR(
          tensorcast::cuda::memcpy_async(
              rank_send_stages[rank]->get(),
              host_buffer_for_slot(slot, rank) + h2d_offset,
              chunk.stripe_bytes,
              cudaMemcpyHostToDevice,
              clique->stream(static_cast<int>(rank))));
    }
    h2d_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    return absl::OkStatus();
  };

  bool batched_scatter_kernel_disabled = false;
  bool batched_scatter_kernel_failure_logged = false;
  bool scatter_cuda_graph_disabled = false;
  bool scatter_cuda_graph_failure_logged = false;

  auto acquire_batched_scatter_host_descriptor_slot =
      [&](size_t rank) -> absl::StatusOr<BatchedScatterHostDescriptorSlot*> {
    if (rank >= rank_batched_scatter_host_descriptor_slots.size()) {
      return absl::InvalidArgumentError("source-window batched scatter rank out of bounds");
    }
    auto& slots = rank_batched_scatter_host_descriptor_slots[rank];
    if (slots.empty()) {
      return absl::FailedPreconditionError("source-window batched scatter host descriptor slots unavailable");
    }
    size_t& next_slot = rank_next_batched_scatter_host_descriptor_slot[rank];
    auto& slot = slots[next_slot % slots.size()];
    next_slot += 1;
    if (slot.ptr == nullptr || slot.ready_event == nullptr) {
      return absl::FailedPreconditionError("source-window batched scatter host descriptor slot is incomplete");
    }
    if (slot.in_flight) {
      const auto wait_start = std::chrono::steady_clock::now();
      bool ready = false;
      TC_RETURN_IF_ERROR(tensorcast::cuda::event_query(slot.ready_event, &ready));
      if (!ready) {
        TC_RETURN_IF_ERROR(tensorcast::cuda::event_synchronize(slot.ready_event));
      }
      batched_scatter_descriptor_slot_wait_sec +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start).count();
      slot.in_flight = false;
    }
    return &slot;
  };

  auto issue_scatter_descriptors_with_memcpy = [&](size_t rank,
                                                   absl::Span<const SourceWindowBatchedScatterDescriptor> descriptors,
                                                   bool count_as_routed_pack) -> absl::Status {
    if (descriptors.empty()) {
      return absl::OkStatus();
    }
    const auto fallback_start = std::chrono::steady_clock::now();
    TC_RETURN_IF_ERROR(set_device_cached(participants[rank].device_id));
    cudaStream_t stream = clique->stream(static_cast<int>(rank));
    for (const auto& desc : descriptors) {
      auto* dst_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(desc.dst_ptr));
      const auto* src_ptr = reinterpret_cast<const void*>(static_cast<uintptr_t>(desc.src_ptr));
      if (desc.row_count > 1) {
        SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
            dst_ptr,
            static_cast<size_t>(desc.target_stride_bytes),
            src_ptr,
            static_cast<size_t>(desc.source_stride_bytes),
            static_cast<size_t>(desc.row_bytes),
            static_cast<size_t>(desc.row_count),
            cudaMemcpyDeviceToDevice,
            stream));
      } else {
        TC_RETURN_IF_ERROR(
            tensorcast::cuda::memcpy_async(
                dst_ptr, src_ptr, static_cast<size_t>(desc.row_bytes), cudaMemcpyDeviceToDevice, stream));
      }
      batched_scatter_fallback_launches += 1;
    }
    batched_scatter_fallback_descriptors += descriptors.size();
    if (count_as_routed_pack) {
      batched_routed_pack_fallback_launches += descriptors.size();
      batched_routed_pack_fallback_descriptors += descriptors.size();
    }
    batched_scatter_fallback_submit_sec +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - fallback_start).count();
    return absl::OkStatus();
  };

  auto issue_scatter_descriptors_with_batched_kernel_or_memcpy =
      [&](size_t rank,
          absl::Span<const SourceWindowBatchedScatterDescriptor> descriptors,
          bool count_as_routed_pack = false) -> absl::Status {
    if (descriptors.empty()) {
      return absl::OkStatus();
    }
    const bool descriptor_buffer_ready = batched_scatter_descriptor_buffers_ready &&
        rank < rank_batched_scatter_descriptors.size() && rank_batched_scatter_descriptors[rank] != nullptr &&
        rank < rank_batched_scatter_host_descriptor_slots.size() &&
        !rank_batched_scatter_host_descriptor_slots[rank].empty();
    const bool use_batched_kernel = descriptor_buffer_ready && !batched_scatter_kernel_disabled &&
        descriptors.size() >= kSourceWindowBatchedScatterMinDescriptors &&
        descriptors.size() <= kSourceWindowBatchedScatterDescriptorCapacity;
    if (use_batched_kernel) {
      auto slot_or = acquire_batched_scatter_host_descriptor_slot(rank);
      if (!slot_or.ok()) {
        batched_scatter_kernel_disabled = true;
        if (!batched_scatter_kernel_failure_logged) {
          LOG(WARNING) << "source-window batched scatter host descriptor slot unavailable; "
                       << "falling back to cudaMemcpy scatter path: " << slot_or.status();
          batched_scatter_kernel_failure_logged = true;
        }
        return issue_scatter_descriptors_with_memcpy(rank, descriptors, count_as_routed_pack);
      }
      auto* slot = *slot_or;
      auto* pinned_descriptors = slot->ptr;
      const auto host_copy_start = std::chrono::steady_clock::now();
      std::memcpy(
          pinned_descriptors,
          descriptors.data(),
          descriptors.size() * source_window_batched_scatter_descriptor_bytes());
      batched_scatter_descriptor_host_copy_sec +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - host_copy_start).count();
      TC_RETURN_IF_ERROR(set_device_cached(participants[rank].device_id));
      const auto kernel_submit_start = std::chrono::steady_clock::now();
      absl::Status launch_status = launch_source_window_batched_scatter(
          absl::MakeSpan(pinned_descriptors, descriptors.size()),
          rank_batched_scatter_descriptors[rank]->get(),
          rank_batched_scatter_descriptors[rank]->size(),
          participants[rank].device_id,
          clique->stream(static_cast<int>(rank)));
      batched_scatter_kernel_submit_sec +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - kernel_submit_start).count();
      if (launch_status.ok()) {
        TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(slot->ready_event, clique->stream(static_cast<int>(rank))));
        slot->in_flight = true;
        batched_scatter_kernel_launches += 1;
        batched_scatter_kernel_descriptors += descriptors.size();
        if (count_as_routed_pack) {
          batched_routed_pack_kernel_launches += 1;
          batched_routed_pack_kernel_descriptors += descriptors.size();
        }
        return absl::OkStatus();
      }
      batched_scatter_kernel_disabled = true;
      if (slot->ready_event != nullptr) {
        const absl::Status record_status =
            tensorcast::cuda::event_record(slot->ready_event, clique->stream(static_cast<int>(rank)));
        if (record_status.ok()) {
          slot->in_flight = true;
        } else {
          const absl::Status sync_status = tensorcast::cuda::stream_synchronize(clique->stream(static_cast<int>(rank)));
          if (!sync_status.ok()) {
            LOG(WARNING) << "source-window batched scatter fallback stream synchronize failed: " << sync_status;
          }
          slot->in_flight = false;
        }
      }
      if (!batched_scatter_kernel_failure_logged) {
        LOG(WARNING) << "source-window batched scatter kernel failed; falling back to cudaMemcpy scatter path: "
                     << launch_status;
        batched_scatter_kernel_failure_logged = true;
      }
    } else if (descriptors.size() > kSourceWindowBatchedScatterDescriptorCapacity) {
      batched_scatter_capacity_fallback_chunks += 1;
    }
    return issue_scatter_descriptors_with_memcpy(rank, descriptors, count_as_routed_pack);
  };

  auto scatter_rank_stages_chunk = [&](const SourceWindowRuntimeChunk& chunk) -> absl::Status {
    const auto step_start = std::chrono::steady_clock::now();
    std::vector<std::vector<SourceWindowBatchedScatterDescriptor>> descriptors_by_rank(world_size);
    auto append_scatter_descriptor = [&](uint32_t rank,
                                         uint32_t storage_index,
                                         const std::uint8_t* source_ptr,
                                         uint64_t target_logical_offset,
                                         uint64_t row_bytes,
                                         uint64_t row_count,
                                         uint64_t source_stride_bytes,
                                         uint64_t target_stride_bytes) -> absl::Status {
      if (rank >= mapped_participants.size()) {
        return absl::InvalidArgumentError("source-window consumer rank out of bounds");
      }
      if (source_ptr == nullptr || row_bytes == 0 || row_count == 0) {
        return absl::OkStatus();
      }
      if (row_count > 1) {
        if (source_stride_bytes < row_bytes || target_stride_bytes < row_bytes) {
          return absl::InvalidArgumentError("source-window scatter span has invalid row strides");
        }
        if (row_count - 1 > (std::numeric_limits<uint64_t>::max() - row_bytes) / target_stride_bytes) {
          return absl::OutOfRangeError("source-window 2D scatter target envelope overflows");
        }
      }
      const uint64_t target_envelope_bytes =
          row_count <= 1 ? row_bytes : (row_count - 1) * target_stride_bytes + row_bytes;
      if (row_count > std::numeric_limits<uint64_t>::max() / row_bytes) {
        return absl::OutOfRangeError("source-window scatter byte count overflows");
      }
      if (row_bytes > std::numeric_limits<size_t>::max() || source_stride_bytes > std::numeric_limits<size_t>::max() ||
          target_stride_bytes > std::numeric_limits<size_t>::max() || row_count > std::numeric_limits<size_t>::max() ||
          target_envelope_bytes > std::numeric_limits<size_t>::max()) {
        return absl::OutOfRangeError("source-window scatter descriptor exceeds CUDA size_t limits");
      }
      auto piece_or = resolve_target_piece_by_storage_index(
          mapped_participants[rank], storage_index, target_logical_offset, target_envelope_bytes);
      if (!piece_or.ok()) {
        return piece_or.status();
      }
      const auto& piece = *piece_or;
      descriptors_by_rank[rank].push_back(
          SourceWindowBatchedScatterDescriptor{
              .src_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(source_ptr + piece.src_offset)),
              .dst_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(piece.dst_ptr.get())),
              .row_bytes = row_bytes,
              .row_count = row_count,
              .source_stride_bytes = source_stride_bytes,
              .target_stride_bytes = target_stride_bytes,
          });
      actual_scatter_ops += 1;
      target_storage_fast_path_pieces += 1;
      target_storage_fast_path_bytes += row_bytes * row_count;
      return absl::OkStatus();
    };

    for (const auto* span_ptr : chunk.consumer_spans) {
      const auto& span = *span_ptr;
      if (span.rank >= mapped_participants.size()) {
        return absl::InvalidArgumentError("source-window consumer rank out of bounds");
      }
      if (span.row_count > 1 && span.row_bytes > 0 && span.source_stride_bytes > 0 && span.target_stride_bytes > 0) {
        if (span.source_stride_bytes < span.row_bytes || span.target_stride_bytes < span.row_bytes) {
          return absl::InvalidArgumentError("source-window 2D scatter span has invalid row strides");
        }
        uint64_t row = 0;
        if (chunk.chunk_start > span.source_offset) {
          row = std::min<uint64_t>(span.row_count, (chunk.chunk_start - span.source_offset) / span.source_stride_bytes);
          while (row < span.row_count &&
                 span.source_offset + row * span.source_stride_bytes + span.row_bytes <= chunk.chunk_start) {
            ++row;
          }
        }
        while (row < span.row_count) {
          const uint64_t row_source_begin = span.source_offset + row * span.source_stride_bytes;
          if (row_source_begin >= chunk.chunk_end) {
            break;
          }
          const uint64_t row_source_end = row_source_begin + span.row_bytes;
          const uint64_t overlap_begin = std::max<uint64_t>(row_source_begin, chunk.chunk_start);
          const uint64_t overlap_end = std::min<uint64_t>(row_source_end, chunk.chunk_end);
          if (overlap_end <= overlap_begin) {
            ++row;
            continue;
          }
          const uint64_t copy_col_offset = overlap_begin - row_source_begin;
          const uint64_t copy_width = overlap_end - overlap_begin;
          uint64_t rows_in_run = 1;
          while (row + rows_in_run < span.row_count) {
            const uint64_t next_row_begin = span.source_offset + (row + rows_in_run) * span.source_stride_bytes;
            if (next_row_begin >= chunk.chunk_end) {
              break;
            }
            const uint64_t next_overlap_begin = std::max<uint64_t>(next_row_begin, chunk.chunk_start);
            const uint64_t next_overlap_end = std::min<uint64_t>(next_row_begin + span.row_bytes, chunk.chunk_end);
            if (next_overlap_end <= next_overlap_begin || next_overlap_begin - next_row_begin != copy_col_offset ||
                next_overlap_end - next_overlap_begin != copy_width) {
              break;
            }
            ++rows_in_run;
          }

          const uint64_t src_chunk_offset = overlap_begin - chunk.chunk_start;
          const uint64_t dst_logical_offset = span.target_offset + row * span.target_stride_bytes + copy_col_offset;
          const auto* source_ptr = static_cast<const std::uint8_t*>(rank_stages[span.rank]->get()) + src_chunk_offset;
          TC_RETURN_IF_ERROR(append_scatter_descriptor(
              span.rank,
              span.storage_index,
              source_ptr,
              dst_logical_offset,
              copy_width,
              rows_in_run,
              span.source_stride_bytes,
              span.target_stride_bytes));
          row += rows_in_run;
        }
      } else {
        const uint64_t span_end = span.source_offset + span.length;
        const uint64_t overlap_begin = std::max<uint64_t>(span.source_offset, chunk.chunk_start);
        const uint64_t overlap_end = std::min<uint64_t>(span_end, chunk.chunk_end);
        if (overlap_end <= overlap_begin) {
          continue;
        }
        const uint64_t overlap_len = overlap_end - overlap_begin;
        const uint64_t src_chunk_offset = overlap_begin - chunk.chunk_start;
        const uint64_t dst_logical_offset = span.target_offset + (overlap_begin - span.source_offset);
        const auto* stage_base = static_cast<const std::uint8_t*>(rank_stages[span.rank]->get()) + src_chunk_offset;
        TC_RETURN_IF_ERROR(append_scatter_descriptor(
            span.rank, span.storage_index, stage_base, dst_logical_offset, overlap_len, 1, overlap_len, overlap_len));
      }
    }
    batched_scatter_descriptor_build_sec +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();

    std::vector<SourceWindowRuntimeScatterGraph> in_flight_scatter_graphs;
    auto cleanup_in_flight_scatter_graphs = absl::Cleanup([&]() {
      if (!in_flight_scatter_graphs.empty()) {
        const absl::Status sync_status = synchronize_clique_all();
        if (!sync_status.ok()) {
          LOG(WARNING) << "source-window scatter cuda graph cleanup synchronize failed: " << sync_status;
        }
        destroy_source_window_runtime_scatter_graphs(&in_flight_scatter_graphs);
      }
    });
    auto issue_scatter_descriptors_with_cuda_graph =
        [&](size_t rank, absl::Span<const SourceWindowBatchedScatterDescriptor> descriptors) -> absl::Status {
      if (descriptors.empty()) {
        return absl::OkStatus();
      }
      TC_RETURN_IF_ERROR(set_device_cached(participants[rank].device_id));
      uint64_t node_count = 0;
      const auto build_start = std::chrono::steady_clock::now();
      auto graph_or = build_source_window_runtime_scatter_graph(descriptors, participants[rank].device_id);
      scatter_cuda_graph_build_sec +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();
      if (!graph_or.ok()) {
        return graph_or.status();
      }
      SourceWindowRuntimeScatterGraph graph = std::move(*graph_or);
      absl::Status node_count_status = source_window_runtime_scatter_graph_node_count(graph.graph, &node_count);
      if (!node_count_status.ok()) {
        destroy_source_window_runtime_scatter_graph(&graph);
        return node_count_status;
      }
      absl::Status launch_status = tensorcast::cuda::cuda_as_status(
          cudaGraphLaunch(graph.exec, clique->stream(static_cast<int>(rank))),
          "cudaGraphLaunch(source-window scatter cuda graph)");
      if (!launch_status.ok()) {
        destroy_source_window_runtime_scatter_graph(&graph);
        return launch_status;
      }
      scatter_cuda_graph_launches += 1;
      scatter_cuda_graph_descriptors += descriptors.size();
      scatter_cuda_graph_nodes += node_count;
      in_flight_scatter_graphs.push_back(std::move(graph));
      return absl::OkStatus();
    };

    for (size_t rank = 0; rank < descriptors_by_rank.size(); ++rank) {
      const auto& descriptors = descriptors_by_rank[rank];
      if (descriptors.empty()) {
        continue;
      }
      const bool use_cuda_graph = scatter_cuda_graph_enabled && !scatter_cuda_graph_disabled &&
          descriptors.size() >= kSourceWindowBatchedScatterMinDescriptors;
      if (use_cuda_graph) {
        absl::Status launch_status = issue_scatter_descriptors_with_cuda_graph(rank, absl::MakeSpan(descriptors));
        if (launch_status.ok()) {
          continue;
        }
        scatter_cuda_graph_disabled = true;
        scatter_cuda_graph_fallback_chunks += 1;
        if (!scatter_cuda_graph_failure_logged) {
          LOG(WARNING) << "source-window scatter cuda graph failed; falling back to cudaMemcpy scatter path: "
                       << launch_status;
          scatter_cuda_graph_failure_logged = true;
        }
      }
      TC_RETURN_IF_ERROR(issue_scatter_descriptors_with_batched_kernel_or_memcpy(rank, absl::MakeSpan(descriptors)));
    }
    scatter_issue_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    {
      const auto sync_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(synchronize_clique_all());
      scatter_sync_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - sync_start).count();
    }
    destroy_source_window_runtime_scatter_graphs(&in_flight_scatter_graphs);
    return absl::OkStatus();
  };

  auto issue_local_only_chunk_from_parallel_slot =
      [&](const SourceWindowRuntimeChunk& chunk, size_t slot, bool issue_h2d_from_host_slot) -> absl::Status {
    if (chunk.window->owner_rank >= world_size) {
      return absl::InvalidArgumentError("source-window local-only owner rank out of bounds");
    }
    const size_t owner = chunk.window->owner_rank;
    if (issue_h2d_from_host_slot) {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(set_device_cached(participants[owner].device_id));
      const size_t h2d_offset = host_h2d_offset_for_slot(slot, owner);
      TC_RETURN_IF_ERROR(
          tensorcast::cuda::memcpy_async(
              rank_stages[owner]->get(),
              host_buffer_for_slot(slot, owner) + h2d_offset,
              chunk.chunk_len,
              cudaMemcpyHostToDevice,
              clique->stream(static_cast<int>(owner))));
      h2d_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    for (const auto* span_ptr : chunk.consumer_spans) {
      const auto& span = *span_ptr;
      if (span.rank != owner) {
        return absl::FailedPreconditionError("source-window local-only runtime found a remote consumer");
      }
    }
    return scatter_rank_stages_chunk(chunk);
  };

  auto issue_gpu_chunk_from_parallel_slot =
      [&](const SourceWindowRuntimeChunk& chunk, size_t slot, bool issue_h2d_from_host_slot) -> absl::Status {
    if (issue_h2d_from_host_slot) {
      TC_RETURN_IF_ERROR(issue_h2d_from_slot(chunk, slot));
    }

    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(clique->group_start());
      for (size_t rank = 0; rank < world_size; ++rank) {
        TC_RETURN_IF_ERROR(clique->all_gather_u8(
            static_cast<int>(rank), rank_send_stages[rank]->get(), rank_stages[rank]->get(), chunk.stripe_bytes));
      }
      TC_RETURN_IF_ERROR(clique->group_end());
      collective_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
      if (world_size > 1) {
        actual_peer_transfer_bytes += chunk.gathered_bytes * static_cast<uint64_t>(world_size - 1);
      }
    }

    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(synchronize_clique_all());
      collective_sync_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }

    return scatter_rank_stages_chunk(chunk);
  };

  auto issue_consumer_routed_chunk_from_parallel_slot =
      [&](const SourceWindowRuntimeChunk& chunk, size_t slot, bool issue_h2d_from_host_slot) -> absl::Status {
    if (issue_h2d_from_host_slot) {
      TC_RETURN_IF_ERROR(issue_h2d_from_slot(chunk, slot));
    }

    if (chunk.routed_program.compiled) {
      const auto& program = chunk.routed_program;
      target_storage_fast_path_pieces += program.target_storage_fast_path_pieces;
      target_storage_fast_path_bytes += program.target_storage_fast_path_bytes;
      auto descriptor_target_envelope_bytes =
          [](const RoutedDescriptorTemplate& descriptor) -> absl::StatusOr<uint64_t> {
        if (descriptor.row_bytes == 0 || descriptor.row_count == 0) {
          return uint64_t{0};
        }
        const uint64_t rows_minus_one = descriptor.row_count - 1;
        if (rows_minus_one > 0 &&
            descriptor.target_stride_bytes >
                (std::numeric_limits<uint64_t>::max() - descriptor.row_bytes) / rows_minus_one) {
          return absl::OutOfRangeError("source-window routed descriptor target envelope overflows");
        }
        return rows_minus_one * descriptor.target_stride_bytes + descriptor.row_bytes;
      };
      auto resolve_routed_target_ptr = [&](const RoutedTargetRef& target,
                                           uint64_t length) -> absl::StatusOr<std::uint8_t*> {
        if (target.rank >= mapped_participants.size()) {
          return absl::InvalidArgumentError("source-window routed target rank out of bounds");
        }
        auto piece_or = resolve_target_piece_by_storage_index(
            mapped_participants[target.rank], target.storage_index, target.logical_offset, length);
        if (!piece_or.ok()) {
          return piece_or.status();
        }
        return piece_or->dst_ptr.get();
      };

      {
        const auto step_start = std::chrono::steady_clock::now();
        for (size_t rank = 0; rank < program.local_descriptors_by_rank.size(); ++rank) {
          const auto& descriptor_templates = program.local_descriptors_by_rank[rank];
          if (descriptor_templates.empty()) {
            continue;
          }
          const auto descriptor_build_start = std::chrono::steady_clock::now();
          const auto* source_base = static_cast<const std::uint8_t*>(rank_send_stages[rank]->get());
          std::vector<SourceWindowBatchedScatterDescriptor> descriptors;
          descriptors.reserve(descriptor_templates.size());
          for (const auto& descriptor_template : descriptor_templates) {
            auto target_bytes_or = descriptor_target_envelope_bytes(descriptor_template);
            if (!target_bytes_or.ok()) {
              return target_bytes_or.status();
            }
            auto dst_ptr_or = resolve_routed_target_ptr(descriptor_template.target, *target_bytes_or);
            if (!dst_ptr_or.ok()) {
              return dst_ptr_or.status();
            }
            descriptors.push_back(
                SourceWindowBatchedScatterDescriptor{
                    .src_ptr = static_cast<uint64_t>(
                        reinterpret_cast<uintptr_t>(source_base + descriptor_template.src_offset)),
                    .dst_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(*dst_ptr_or)),
                    .row_bytes = descriptor_template.row_bytes,
                    .row_count = descriptor_template.row_count,
                    .source_stride_bytes = descriptor_template.source_stride_bytes,
                    .target_stride_bytes = descriptor_template.target_stride_bytes,
                });
          }
          routed_local_descriptor_build_sec +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() - descriptor_build_start).count();
          const auto local_issue_start = std::chrono::steady_clock::now();
          TC_RETURN_IF_ERROR(
              issue_scatter_descriptors_with_batched_kernel_or_memcpy(rank, absl::MakeSpan(descriptors)));
          routed_local_issue_sec +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() - local_issue_start).count();
          actual_scatter_ops += descriptors.size();
          routed_local_pieces += descriptors.size();
        }
        routed_local_2d_pieces += program.local_2d_pieces;

        for (size_t producer = 0; producer < program.pack_descriptors_by_producer.size(); ++producer) {
          const auto& descriptor_templates = program.pack_descriptors_by_producer[producer];
          if (descriptor_templates.empty()) {
            continue;
          }
          const auto descriptor_build_start = std::chrono::steady_clock::now();
          const auto* source_base = static_cast<const std::uint8_t*>(rank_send_stages[producer]->get());
          std::vector<SourceWindowBatchedScatterDescriptor> descriptors;
          descriptors.reserve(descriptor_templates.size());
          for (const auto& descriptor_template : descriptor_templates) {
            auto* pack_stage = route_pack_stages[producer][descriptor_template.consumer].get();
            if (pack_stage == nullptr) {
              return absl::FailedPreconditionError("source-window routed compiled pack is missing staging buffer");
            }
            auto* pack_base = static_cast<std::uint8_t*>(pack_stage->get());
            descriptors.push_back(
                SourceWindowBatchedScatterDescriptor{
                    .src_ptr = static_cast<uint64_t>(
                        reinterpret_cast<uintptr_t>(source_base + descriptor_template.src_offset)),
                    .dst_ptr =
                        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pack_base + descriptor_template.pack_offset)),
                    .row_bytes = descriptor_template.row_bytes,
                    .row_count = descriptor_template.row_count,
                    .source_stride_bytes = descriptor_template.source_stride_bytes,
                    .target_stride_bytes = descriptor_template.target_stride_bytes,
                });
          }
          routed_pack_descriptor_build_sec +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() - descriptor_build_start).count();
          const auto pack_issue_start = std::chrono::steady_clock::now();
          TC_RETURN_IF_ERROR(issue_scatter_descriptors_with_batched_kernel_or_memcpy(
              producer,
              absl::MakeSpan(descriptors),
              /*count_as_routed_pack=*/true));
          routed_pack_issue_sec +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() - pack_issue_start).count();
        }
        routed_pack_ops += program.pack_ops;
        routed_deferred_2d_pack_ops += program.deferred_2d_pack_ops;
        scatter_issue_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
      }

      {
        const auto step_start = std::chrono::steady_clock::now();
        bool group_open = false;
        size_t current_group_pairs = 0;
        auto ensure_group_pair_capacity = [&]() -> absl::Status {
          if (!group_open) {
            TC_RETURN_IF_ERROR(clique->group_start());
            group_open = true;
          }
          if (current_group_pairs >= kMaxMappedPeerPairsPerNcclGroup) {
            TC_RETURN_IF_ERROR(clique->group_end());
            TC_RETURN_IF_ERROR(clique->group_start());
            group_open = true;
            current_group_pairs = 0;
          }
          return absl::OkStatus();
        };
        auto flush_group = [&]() -> absl::Status {
          if (!group_open) {
            return absl::OkStatus();
          }
          TC_RETURN_IF_ERROR(clique->group_end());
          group_open = false;
          current_group_pairs = 0;
          return absl::OkStatus();
        };
        for (const auto& transfer : program.packed_transfers) {
          if (transfer.bytes > std::numeric_limits<size_t>::max()) {
            return absl::OutOfRangeError("source-window routed packed transfer exceeds size_t limits");
          }
          auto* pack_stage = route_pack_stages[transfer.producer][transfer.consumer].get();
          auto* recv_stage = route_recv_stages[transfer.producer][transfer.consumer].get();
          if (pack_stage == nullptr || recv_stage == nullptr) {
            return absl::FailedPreconditionError("source-window routed packed transfer is missing staging buffers");
          }
          TC_RETURN_IF_ERROR(ensure_group_pair_capacity());
          TC_RETURN_IF_ERROR(clique->send_u8(
              static_cast<int>(transfer.producer),
              pack_stage->get(),
              static_cast<size_t>(transfer.bytes),
              static_cast<int>(transfer.consumer)));
          TC_RETURN_IF_ERROR(clique->recv_u8(
              static_cast<int>(transfer.consumer),
              recv_stage->get(),
              static_cast<size_t>(transfer.bytes),
              static_cast<int>(transfer.producer)));
          actual_peer_transfer_bytes += transfer.bytes;
          routed_packed_pairs += 1;
          current_group_pairs += 1;
        }
        for (const auto& piece : program.direct_remote_pieces) {
          if (piece.length > std::numeric_limits<size_t>::max()) {
            return absl::OutOfRangeError("source-window routed remote piece exceeds size_t limits");
          }
          const auto* source_base = static_cast<const std::uint8_t*>(rank_send_stages[piece.producer]->get());
          auto dst_ptr_or = resolve_routed_target_ptr(piece.target, piece.length);
          if (!dst_ptr_or.ok()) {
            return dst_ptr_or.status();
          }
          TC_RETURN_IF_ERROR(ensure_group_pair_capacity());
          TC_RETURN_IF_ERROR(clique->send_u8(
              static_cast<int>(piece.producer),
              source_base + piece.src_offset,
              static_cast<size_t>(piece.length),
              static_cast<int>(piece.consumer)));
          TC_RETURN_IF_ERROR(clique->recv_u8(
              static_cast<int>(piece.consumer),
              *dst_ptr_or,
              static_cast<size_t>(piece.length),
              static_cast<int>(piece.producer)));
          actual_peer_transfer_bytes += piece.length;
          actual_scatter_ops += 1;
          routed_remote_pieces += 1;
          current_group_pairs += 1;
        }
        TC_RETURN_IF_ERROR(flush_group());
        collective_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
      }

      {
        const auto step_start = std::chrono::steady_clock::now();
        TC_RETURN_IF_ERROR(synchronize_clique_all());
        collective_sync_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
      }

      {
        const auto step_start = std::chrono::steady_clock::now();
        const auto descriptor_build_start = step_start;
        std::vector<std::vector<SourceWindowBatchedScatterDescriptor>> descriptors_by_consumer(world_size);
        for (const auto& piece : program.packed_remote_pieces) {
          if (piece.length > std::numeric_limits<size_t>::max()) {
            return absl::OutOfRangeError("source-window routed packed scatter exceeds size_t limits");
          }
          auto* recv_stage = route_recv_stages[piece.producer][piece.consumer].get();
          if (recv_stage == nullptr) {
            return absl::FailedPreconditionError("source-window routed packed scatter is missing receive staging");
          }
          const auto* recv_base = static_cast<const std::uint8_t*>(recv_stage->get());
          auto dst_ptr_or = resolve_routed_target_ptr(piece.target, piece.length);
          if (!dst_ptr_or.ok()) {
            return dst_ptr_or.status();
          }
          descriptors_by_consumer[piece.consumer].push_back(
              SourceWindowBatchedScatterDescriptor{
                  .src_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(recv_base + piece.pack_offset)),
                  .dst_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(*dst_ptr_or)),
                  .row_bytes = piece.length,
                  .row_count = 1,
                  .source_stride_bytes = piece.length,
                  .target_stride_bytes = piece.length,
              });
        }
        routed_remote_descriptor_build_sec +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - descriptor_build_start).count();
        for (size_t consumer = 0; consumer < descriptors_by_consumer.size(); ++consumer) {
          const auto& descriptors = descriptors_by_consumer[consumer];
          const auto remote_scatter_issue_start = std::chrono::steady_clock::now();
          TC_RETURN_IF_ERROR(
              issue_scatter_descriptors_with_batched_kernel_or_memcpy(consumer, absl::MakeSpan(descriptors)));
          routed_remote_scatter_issue_sec +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() - remote_scatter_issue_start).count();
          actual_scatter_ops += descriptors.size();
          routed_remote_pieces += descriptors.size();
        }
        scatter_issue_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
      }
      return absl::OkStatus();
    }

    std::vector<std::vector<CopyPiece>> local_pieces(world_size);
    std::vector<std::vector<std::vector<CopyPiece>>> remote_pieces(world_size);
    std::vector<std::vector<std::vector<PackedRemotePiece>>> packed_remote_pieces(world_size);
    std::vector<std::vector<uint64_t>> pack_offsets(world_size, std::vector<uint64_t>(world_size, 0));
    struct PendingLocal2dCopy {
      std::uint8_t* dst_ptr{nullptr};
      const std::uint8_t* src_ptr{nullptr};
      uint64_t width{0};
      uint64_t rows{0};
      uint64_t source_pitch{0};
      uint64_t target_pitch{0};
    };
    struct PendingPacked2dCopy {
      std::uint8_t* pack_ptr{nullptr};
      const std::uint8_t* src_ptr{nullptr};
      uint64_t width{0};
      uint64_t rows{0};
      uint64_t source_pitch{0};
    };
    std::vector<std::vector<PendingLocal2dCopy>> pending_local_2d_copies_by_rank(world_size);
    std::vector<std::vector<PendingPacked2dCopy>> pending_packed_2d_copies_by_producer(world_size);
    for (auto& producer_pieces : remote_pieces) {
      producer_pieces.resize(world_size);
    }
    for (auto& producer_pieces : packed_remote_pieces) {
      producer_pieces.resize(world_size);
    }

    auto producer_for_source_offset = [&](uint64_t source_offset) -> absl::StatusOr<size_t> {
      if (source_offset < chunk.chunk_start || source_offset >= chunk.chunk_end) {
        return absl::OutOfRangeError("source-window routed source offset is outside chunk");
      }
      const uint64_t chunk_offset = source_offset - chunk.chunk_start;
      size_t producer = static_cast<size_t>(chunk_offset / chunk.stripe_bytes);
      if (producer >= world_size) {
        producer = world_size - 1;
      }
      return producer;
    };

    auto append_packed_remote_piece = [&](size_t producer,
                                          uint32_t consumer_rank,
                                          const std::uint8_t* src_ptr,
                                          std::uint8_t* dst_ptr,
                                          uint64_t length) -> absl::StatusOr<bool> {
      if (producer == consumer_rank || length == 0 || src_ptr == nullptr || dst_ptr == nullptr ||
          length > std::numeric_limits<size_t>::max()) {
        return false;
      }
      if (producer >= route_pack_stages.size() || consumer_rank >= route_pack_stages[producer].size() ||
          producer >= route_recv_stages.size() || consumer_rank >= route_recv_stages[producer].size()) {
        return false;
      }
      auto* pack_stage = route_pack_stages[producer][consumer_rank].get();
      auto* recv_stage = route_recv_stages[producer][consumer_rank].get();
      if (pack_stage == nullptr || recv_stage == nullptr) {
        return false;
      }
      uint64_t& pack_offset = pack_offsets[producer][consumer_rank];
      if (pack_offset > max_stripe_bytes || length > max_stripe_bytes - pack_offset) {
        return false;
      }
      auto* pack_ptr = static_cast<std::uint8_t*>(pack_stage->get()) + pack_offset;
      TC_RETURN_IF_ERROR(set_device_cached(participants[producer].device_id));
      TC_RETURN_IF_ERROR(
          tensorcast::cuda::memcpy_async(
              pack_ptr,
              src_ptr,
              static_cast<size_t>(length),
              cudaMemcpyDeviceToDevice,
              clique->stream(static_cast<int>(producer))));
      routed_pack_ops += 1;
      append_merged_packed_remote_piece(
          packed_remote_pieces[producer][consumer_rank],
          PackedRemotePiece{
              .pack_offset = pack_offset,
              .dst_ptr = dst_ptr,
              .length = length,
          });
      pack_offset += length;
      return true;
    };

    auto append_routed_linear_piece = [&](uint32_t consumer_rank,
                                          uint32_t storage_index,
                                          uint64_t source_begin,
                                          uint64_t target_logical_offset,
                                          uint64_t length) -> absl::Status {
      if (consumer_rank >= mapped_participants.size()) {
        return absl::InvalidArgumentError("source-window consumer rank out of bounds");
      }
      auto piece_or = resolve_target_piece_by_storage_index(
          mapped_participants[consumer_rank], storage_index, target_logical_offset, length);
      if (!piece_or.ok()) {
        return piece_or.status();
      }
      const auto& piece = *piece_or;
      target_storage_fast_path_pieces += 1;
      target_storage_fast_path_bytes += piece.length;
      uint64_t remaining = piece.length;
      uint64_t source_cursor = source_begin + piece.src_offset;
      uint64_t target_piece_offset = 0;
      while (remaining > 0) {
        auto producer_or = producer_for_source_offset(source_cursor);
        if (!producer_or.ok()) {
          return producer_or.status();
        }
        const size_t producer = *producer_or;
        const uint64_t producer_stripe_begin = static_cast<uint64_t>(producer) * chunk.stripe_bytes;
        const uint64_t producer_offset = source_cursor - chunk.chunk_start - producer_stripe_begin;
        const uint64_t producer_available = chunk.stripe_bytes - producer_offset;
        const uint64_t take =
            std::min<uint64_t>(remaining, std::min<uint64_t>(producer_available, chunk.chunk_end - source_cursor));
        if (take == 0) {
          return absl::InternalError("source-window routed linear split made no progress");
        }
        if (take > std::numeric_limits<size_t>::max()) {
          return absl::OutOfRangeError("source-window routed piece exceeds size_t limits");
        }
        const auto* src_ptr = static_cast<const std::uint8_t*>(rank_send_stages[producer]->get()) + producer_offset;
        auto* dst_ptr = piece.dst_ptr.get() + target_piece_offset;
        if (producer == consumer_rank) {
          append_merged_copy_piece(
              local_pieces[consumer_rank],
              CopyPiece{
                  .src_ptr = src_ptr,
                  .dst_ptr = dst_ptr,
                  .length = take,
              });
        } else {
          constexpr uint64_t kRoutedPackSmallPieceBytes = 1ULL << 20;
          const bool pair_already_packing = pack_offsets[producer][consumer_rank] > 0;
          auto packed_or = (pair_already_packing || take <= kRoutedPackSmallPieceBytes)
              ? append_packed_remote_piece(producer, consumer_rank, src_ptr, dst_ptr, take)
              : absl::StatusOr<bool>(false);
          if (!packed_or.ok()) {
            return packed_or.status();
          }
          if (!*packed_or) {
            append_merged_copy_piece(
                remote_pieces[producer][consumer_rank],
                CopyPiece{
                    .src_ptr = src_ptr,
                    .dst_ptr = dst_ptr,
                    .length = take,
                });
          }
        }
        remaining -= take;
        source_cursor += take;
        target_piece_offset += take;
      }
      return absl::OkStatus();
    };

    auto append_remote_packed_2d = [&](size_t producer,
                                       uint32_t consumer_rank,
                                       uint32_t storage_index,
                                       const std::uint8_t* src_ptr,
                                       uint64_t source_pitch,
                                       uint64_t dst_logical_offset,
                                       uint64_t width,
                                       uint64_t rows,
                                       uint64_t target_pitch) -> absl::StatusOr<bool> {
      if (producer == consumer_rank || rows <= 1 || width == 0 || target_pitch != width) {
        return false;
      }
      if (rows > std::numeric_limits<uint64_t>::max() / width) {
        return absl::OutOfRangeError("source-window routed packed bytes overflow");
      }
      const uint64_t packed_bytes = rows * width;
      if (packed_bytes == 0 || packed_bytes > max_stripe_bytes || packed_bytes > std::numeric_limits<size_t>::max() ||
          width > std::numeric_limits<size_t>::max() || source_pitch > std::numeric_limits<size_t>::max() ||
          rows > std::numeric_limits<size_t>::max()) {
        return false;
      }
      auto piece_or = resolve_target_piece_by_storage_index(
          mapped_participants[consumer_rank], storage_index, dst_logical_offset, packed_bytes);
      if (!piece_or.ok()) {
        return piece_or.status();
      }
      auto* pack_stage = route_pack_stages[producer][consumer_rank].get();
      if (pack_stage == nullptr) {
        return false;
      }
      uint64_t& pack_offset = pack_offsets[producer][consumer_rank];
      if (pack_offset > max_stripe_bytes || packed_bytes > max_stripe_bytes - pack_offset) {
        return false;
      }
      auto* pack_ptr = static_cast<std::uint8_t*>(pack_stage->get()) + pack_offset;
      pending_packed_2d_copies_by_producer[producer].push_back(
          PendingPacked2dCopy{
              .pack_ptr = pack_ptr,
              .src_ptr = src_ptr,
              .width = width,
              .rows = rows,
              .source_pitch = source_pitch,
          });
      append_merged_packed_remote_piece(
          packed_remote_pieces[producer][consumer_rank],
          PackedRemotePiece{
              .pack_offset = pack_offset,
              .dst_ptr = piece_or->dst_ptr.get(),
              .length = packed_bytes,
          });
      target_storage_fast_path_pieces += 1;
      target_storage_fast_path_bytes += packed_bytes;
      pack_offset += packed_bytes;
      return true;
    };

    {
      const auto step_start = std::chrono::steady_clock::now();
      const auto span_plan_start = step_start;
      for (const auto* span_ptr : chunk.consumer_spans) {
        const auto& span = *span_ptr;
        if (span.rank >= mapped_participants.size()) {
          return absl::InvalidArgumentError("source-window consumer rank out of bounds");
        }
        if (span.row_count > 1 && span.row_bytes > 0 && span.source_stride_bytes > 0 && span.target_stride_bytes > 0) {
          if (span.source_stride_bytes < span.row_bytes || span.target_stride_bytes < span.row_bytes) {
            return absl::InvalidArgumentError("source-window 2D routed span has invalid row strides");
          }
          uint64_t row = 0;
          if (chunk.chunk_start > span.source_offset) {
            row =
                std::min<uint64_t>(span.row_count, (chunk.chunk_start - span.source_offset) / span.source_stride_bytes);
            while (row < span.row_count &&
                   span.source_offset + row * span.source_stride_bytes + span.row_bytes <= chunk.chunk_start) {
              ++row;
            }
          }
          while (row < span.row_count) {
            const uint64_t row_source_begin = span.source_offset + row * span.source_stride_bytes;
            if (row_source_begin >= chunk.chunk_end) {
              break;
            }
            const uint64_t row_source_end = row_source_begin + span.row_bytes;
            const uint64_t overlap_begin = std::max<uint64_t>(row_source_begin, chunk.chunk_start);
            const uint64_t overlap_end = std::min<uint64_t>(row_source_end, chunk.chunk_end);
            if (overlap_end <= overlap_begin) {
              ++row;
              continue;
            }
            const uint64_t copy_col_offset = overlap_begin - row_source_begin;
            const uint64_t copy_width = overlap_end - overlap_begin;
            uint64_t rows_in_run = 1;
            while (row + rows_in_run < span.row_count) {
              const uint64_t next_row_begin = span.source_offset + (row + rows_in_run) * span.source_stride_bytes;
              if (next_row_begin >= chunk.chunk_end) {
                break;
              }
              const uint64_t next_overlap_begin = std::max<uint64_t>(next_row_begin, chunk.chunk_start);
              const uint64_t next_overlap_end = std::min<uint64_t>(next_row_begin + span.row_bytes, chunk.chunk_end);
              if (next_overlap_end <= next_overlap_begin || next_overlap_begin - next_row_begin != copy_col_offset ||
                  next_overlap_end - next_overlap_begin != copy_width) {
                break;
              }
              ++rows_in_run;
            }

            uint64_t run_row = 0;
            while (run_row < rows_in_run) {
              const uint64_t first_source =
                  span.source_offset + (row + run_row) * span.source_stride_bytes + copy_col_offset;
              auto producer_or = producer_for_source_offset(first_source);
              if (!producer_or.ok()) {
                return producer_or.status();
              }
              const size_t producer = *producer_or;
              const uint64_t producer_stripe_begin = static_cast<uint64_t>(producer) * chunk.stripe_bytes;
              const uint64_t producer_offset = first_source - chunk.chunk_start - producer_stripe_begin;
              if (producer_offset + copy_width > chunk.stripe_bytes) {
                const uint64_t dst_logical_offset =
                    span.target_offset + (row + run_row) * span.target_stride_bytes + copy_col_offset;
                TC_RETURN_IF_ERROR(append_routed_linear_piece(
                    span.rank, span.storage_index, first_source, dst_logical_offset, copy_width));
                run_row += 1;
                continue;
              }

              uint64_t grouped_rows = 1;
              while (run_row + grouped_rows < rows_in_run) {
                const uint64_t next_source =
                    span.source_offset + (row + run_row + grouped_rows) * span.source_stride_bytes + copy_col_offset;
                auto next_producer_or = producer_for_source_offset(next_source);
                if (!next_producer_or.ok() || *next_producer_or != producer) {
                  break;
                }
                const uint64_t next_producer_offset = next_source - chunk.chunk_start - producer_stripe_begin;
                if (next_producer_offset + copy_width > chunk.stripe_bytes) {
                  break;
                }
                ++grouped_rows;
              }

              const uint64_t dst_logical_offset =
                  span.target_offset + (row + run_row) * span.target_stride_bytes + copy_col_offset;
              const auto* src_ptr =
                  static_cast<const std::uint8_t*>(rank_send_stages[producer]->get()) + producer_offset;
              bool copied_as_group = false;
              if (producer == span.rank && grouped_rows > 1) {
                const uint64_t target_envelope_bytes = (grouped_rows - 1) * span.target_stride_bytes + copy_width;
                auto piece_or = resolve_target_piece_by_storage_index(
                    mapped_participants[span.rank], span.storage_index, dst_logical_offset, target_envelope_bytes);
                if (!piece_or.ok()) {
                  return piece_or.status();
                }
                if (copy_width <= std::numeric_limits<size_t>::max() &&
                    span.source_stride_bytes <= std::numeric_limits<size_t>::max() &&
                    span.target_stride_bytes <= std::numeric_limits<size_t>::max() &&
                    grouped_rows <= std::numeric_limits<size_t>::max()) {
                  pending_local_2d_copies_by_rank[span.rank].push_back(
                      PendingLocal2dCopy{
                          .dst_ptr = piece_or->dst_ptr.get(),
                          .src_ptr = src_ptr,
                          .width = copy_width,
                          .rows = grouped_rows,
                          .source_pitch = span.source_stride_bytes,
                          .target_pitch = span.target_stride_bytes,
                      });
                  target_storage_fast_path_pieces += 1;
                  target_storage_fast_path_bytes += copy_width * grouped_rows;
                  copied_as_group = true;
                }
              } else if (producer != span.rank && grouped_rows > 1) {
                auto packed_or = append_remote_packed_2d(
                    producer,
                    span.rank,
                    span.storage_index,
                    src_ptr,
                    span.source_stride_bytes,
                    dst_logical_offset,
                    copy_width,
                    grouped_rows,
                    span.target_stride_bytes);
                if (!packed_or.ok()) {
                  return packed_or.status();
                }
                copied_as_group = *packed_or;
              }
              if (!copied_as_group) {
                for (uint64_t offset_row = 0; offset_row < grouped_rows; ++offset_row) {
                  const uint64_t row_source =
                      span.source_offset + (row + run_row + offset_row) * span.source_stride_bytes + copy_col_offset;
                  const uint64_t row_target =
                      span.target_offset + (row + run_row + offset_row) * span.target_stride_bytes + copy_col_offset;
                  TC_RETURN_IF_ERROR(
                      append_routed_linear_piece(span.rank, span.storage_index, row_source, row_target, copy_width));
                }
              }
              run_row += grouped_rows;
            }
            row += rows_in_run;
          }
        } else {
          const uint64_t span_end = span.source_offset + span.length;
          const uint64_t overlap_begin = std::max<uint64_t>(span.source_offset, chunk.chunk_start);
          const uint64_t overlap_end = std::min<uint64_t>(span_end, chunk.chunk_end);
          if (overlap_end <= overlap_begin) {
            continue;
          }
          const uint64_t dst_logical_offset = span.target_offset + (overlap_begin - span.source_offset);
          TC_RETURN_IF_ERROR(append_routed_linear_piece(
              span.rank, span.storage_index, overlap_begin, dst_logical_offset, overlap_end - overlap_begin));
        }
      }
      routed_span_plan_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - span_plan_start).count();

      for (size_t rank = 0; rank < pending_local_2d_copies_by_rank.size(); ++rank) {
        const auto& pending_copies = pending_local_2d_copies_by_rank[rank];
        if (pending_copies.empty()) {
          continue;
        }
        const auto descriptor_build_start = std::chrono::steady_clock::now();
        std::vector<SourceWindowBatchedScatterDescriptor> descriptors;
        descriptors.reserve(pending_copies.size());
        for (const auto& pending : pending_copies) {
          descriptors.push_back(
              SourceWindowBatchedScatterDescriptor{
                  .src_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pending.src_ptr)),
                  .dst_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pending.dst_ptr)),
                  .row_bytes = pending.width,
                  .row_count = pending.rows,
                  .source_stride_bytes = pending.source_pitch,
                  .target_stride_bytes = pending.target_pitch,
              });
        }
        routed_local_descriptor_build_sec +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - descriptor_build_start).count();
        const auto local_issue_start = std::chrono::steady_clock::now();
        TC_RETURN_IF_ERROR(issue_scatter_descriptors_with_batched_kernel_or_memcpy(rank, absl::MakeSpan(descriptors)));
        routed_local_issue_sec +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - local_issue_start).count();
        actual_scatter_ops += descriptors.size();
        routed_local_2d_pieces += descriptors.size();
        routed_local_pieces += descriptors.size();
      }

      for (size_t producer = 0; producer < pending_packed_2d_copies_by_producer.size(); ++producer) {
        const auto& pending_copies = pending_packed_2d_copies_by_producer[producer];
        if (pending_copies.empty()) {
          continue;
        }
        const auto descriptor_build_start = std::chrono::steady_clock::now();
        std::vector<SourceWindowBatchedScatterDescriptor> descriptors;
        descriptors.reserve(pending_copies.size());
        for (const auto& pending : pending_copies) {
          descriptors.push_back(
              SourceWindowBatchedScatterDescriptor{
                  .src_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pending.src_ptr)),
                  .dst_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pending.pack_ptr)),
                  .row_bytes = pending.width,
                  .row_count = pending.rows,
                  .source_stride_bytes = pending.source_pitch,
                  .target_stride_bytes = pending.width,
              });
        }
        routed_pack_descriptor_build_sec +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - descriptor_build_start).count();
        const auto pack_issue_start = std::chrono::steady_clock::now();
        TC_RETURN_IF_ERROR(issue_scatter_descriptors_with_batched_kernel_or_memcpy(
            producer,
            absl::MakeSpan(descriptors),
            /*count_as_routed_pack=*/true));
        routed_pack_issue_sec +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - pack_issue_start).count();
        routed_pack_ops += descriptors.size();
        routed_deferred_2d_pack_ops += descriptors.size();
      }

      for (size_t rank = 0; rank < local_pieces.size(); ++rank) {
        const auto descriptor_build_start = std::chrono::steady_clock::now();
        std::vector<SourceWindowBatchedScatterDescriptor> descriptors;
        descriptors.reserve(local_pieces[rank].size());
        for (const auto& piece : local_pieces[rank]) {
          if (piece.length > std::numeric_limits<size_t>::max()) {
            return absl::OutOfRangeError("source-window routed local piece exceeds size_t limits");
          }
          descriptors.push_back(
              SourceWindowBatchedScatterDescriptor{
                  .src_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(piece.src_ptr)),
                  .dst_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(piece.dst_ptr)),
                  .row_bytes = piece.length,
                  .row_count = 1,
                  .source_stride_bytes = piece.length,
                  .target_stride_bytes = piece.length,
              });
        }
        routed_local_descriptor_build_sec +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - descriptor_build_start).count();
        const auto local_issue_start = std::chrono::steady_clock::now();
        TC_RETURN_IF_ERROR(issue_scatter_descriptors_with_batched_kernel_or_memcpy(rank, absl::MakeSpan(descriptors)));
        routed_local_issue_sec +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - local_issue_start).count();
        actual_scatter_ops += descriptors.size();
        routed_local_pieces += descriptors.size();
      }
      scatter_issue_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }

    {
      const auto step_start = std::chrono::steady_clock::now();
      bool group_open = false;
      size_t current_group_pairs = 0;
      auto ensure_group_pair_capacity = [&]() -> absl::Status {
        if (!group_open) {
          TC_RETURN_IF_ERROR(clique->group_start());
          group_open = true;
        }
        if (current_group_pairs >= kMaxMappedPeerPairsPerNcclGroup) {
          TC_RETURN_IF_ERROR(clique->group_end());
          TC_RETURN_IF_ERROR(clique->group_start());
          group_open = true;
          current_group_pairs = 0;
        }
        return absl::OkStatus();
      };
      auto flush_group = [&]() -> absl::Status {
        if (!group_open) {
          return absl::OkStatus();
        }
        TC_RETURN_IF_ERROR(clique->group_end());
        group_open = false;
        current_group_pairs = 0;
        return absl::OkStatus();
      };
      for (size_t producer = 0; producer < remote_pieces.size(); ++producer) {
        for (size_t consumer = 0; consumer < remote_pieces[producer].size(); ++consumer) {
          if (producer == consumer) {
            continue;
          }
          const uint64_t packed_bytes = pack_offsets[producer][consumer];
          if (packed_bytes > 0) {
            if (packed_bytes > std::numeric_limits<size_t>::max()) {
              return absl::OutOfRangeError("source-window routed packed transfer exceeds size_t limits");
            }
            auto* pack_stage = route_pack_stages[producer][consumer].get();
            auto* recv_stage = route_recv_stages[producer][consumer].get();
            if (pack_stage == nullptr || recv_stage == nullptr) {
              return absl::FailedPreconditionError("source-window routed packed transfer is missing staging buffers");
            }
            TC_RETURN_IF_ERROR(ensure_group_pair_capacity());
            TC_RETURN_IF_ERROR(clique->send_u8(
                static_cast<int>(producer),
                pack_stage->get(),
                static_cast<size_t>(packed_bytes),
                static_cast<int>(consumer)));
            TC_RETURN_IF_ERROR(clique->recv_u8(
                static_cast<int>(consumer),
                recv_stage->get(),
                static_cast<size_t>(packed_bytes),
                static_cast<int>(producer)));
            actual_peer_transfer_bytes += packed_bytes;
            routed_packed_pairs += 1;
            current_group_pairs += 1;
          }
          for (const auto& piece : remote_pieces[producer][consumer]) {
            if (piece.length == 0) {
              continue;
            }
            if (piece.length > std::numeric_limits<size_t>::max()) {
              return absl::OutOfRangeError("source-window routed remote piece exceeds size_t limits");
            }
            TC_RETURN_IF_ERROR(ensure_group_pair_capacity());
            TC_RETURN_IF_ERROR(clique->send_u8(
                static_cast<int>(producer),
                piece.src_ptr,
                static_cast<size_t>(piece.length),
                static_cast<int>(consumer)));
            TC_RETURN_IF_ERROR(clique->recv_u8(
                static_cast<int>(consumer),
                piece.dst_ptr,
                static_cast<size_t>(piece.length),
                static_cast<int>(producer)));
            actual_peer_transfer_bytes += piece.length;
            actual_scatter_ops += 1;
            routed_remote_pieces += 1;
            current_group_pairs += 1;
          }
        }
      }
      TC_RETURN_IF_ERROR(flush_group());
      collective_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(synchronize_clique_all());
      collective_sync_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    {
      const auto step_start = std::chrono::steady_clock::now();
      const auto descriptor_build_start = step_start;
      std::vector<std::vector<SourceWindowBatchedScatterDescriptor>> descriptors_by_consumer(world_size);
      for (size_t producer = 0; producer < packed_remote_pieces.size(); ++producer) {
        for (size_t consumer = 0; consumer < packed_remote_pieces[producer].size(); ++consumer) {
          if (producer == consumer || packed_remote_pieces[producer][consumer].empty()) {
            continue;
          }
          auto* recv_stage = route_recv_stages[producer][consumer].get();
          if (recv_stage == nullptr) {
            return absl::FailedPreconditionError("source-window routed packed scatter is missing receive staging");
          }
          const auto* recv_base = static_cast<const std::uint8_t*>(recv_stage->get());
          auto& descriptors = descriptors_by_consumer[consumer];
          descriptors.reserve(descriptors.size() + packed_remote_pieces[producer][consumer].size());
          for (const auto& piece : packed_remote_pieces[producer][consumer]) {
            if (piece.length > std::numeric_limits<size_t>::max()) {
              return absl::OutOfRangeError("source-window routed packed scatter exceeds size_t limits");
            }
            descriptors.push_back(
                SourceWindowBatchedScatterDescriptor{
                    .src_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(recv_base + piece.pack_offset)),
                    .dst_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(piece.dst_ptr)),
                    .row_bytes = piece.length,
                    .row_count = 1,
                    .source_stride_bytes = piece.length,
                    .target_stride_bytes = piece.length,
                });
          }
        }
      }
      routed_remote_descriptor_build_sec +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - descriptor_build_start).count();
      for (size_t consumer = 0; consumer < descriptors_by_consumer.size(); ++consumer) {
        const auto& descriptors = descriptors_by_consumer[consumer];
        const auto remote_scatter_issue_start = std::chrono::steady_clock::now();
        TC_RETURN_IF_ERROR(
            issue_scatter_descriptors_with_batched_kernel_or_memcpy(consumer, absl::MakeSpan(descriptors)));
        routed_remote_scatter_issue_sec +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - remote_scatter_issue_start).count();
        actual_scatter_ops += descriptors.size();
        routed_remote_pieces += descriptors.size();
      }
      scatter_issue_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    return absl::OkStatus();
  };

  auto execute_single_buffer_chunk = [&](const SourceWindowRuntimeChunk& chunk) -> absl::Status {
    if (chunk_uses_local_only(chunk)) {
      if (chunk.window->owner_rank >= world_size) {
        return absl::InvalidArgumentError("source-window local-only owner rank out of bounds");
      }
      const size_t owner = chunk.window->owner_rank;
      {
        const auto step_start = std::chrono::steady_clock::now();
        char* host_buffer = host_pool.buffers.front();
        TC_RETURN_IF_ERROR(read_exact(*rank_sources[owner], chunk.chunk_start, host_buffer, chunk.chunk_len));
        TC_RETURN_IF_ERROR(set_device_cached(participants[owner].device_id));
        TC_RETURN_IF_ERROR(
            tensorcast::cuda::memcpy_async(
                rank_stages[owner]->get(),
                host_buffer,
                chunk.chunk_len,
                cudaMemcpyHostToDevice,
                clique->stream(static_cast<int>(owner))));
        read_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
      }
      return issue_local_only_chunk_from_parallel_slot(chunk, /*slot=*/0, /*issue_h2d_from_host_slot=*/false);
    }
    {
      const auto step_start = std::chrono::steady_clock::now();
      std::vector<absl::Status> read_statuses(world_size);
      for (size_t rank = 0; rank < world_size; ++rank) {
        char* host_buffer = host_pool.buffers.front();
        const uint64_t stripe_start = chunk.chunk_start + static_cast<uint64_t>(rank * chunk.stripe_bytes);
        const size_t read_len = stripe_start < chunk.chunk_end
            ? static_cast<size_t>(std::min<uint64_t>(chunk.stripe_bytes, chunk.chunk_end - stripe_start))
            : 0;
        if (read_len > 0) {
          read_statuses[rank] = read_exact(*rank_sources[rank], stripe_start, host_buffer, read_len);
          if (read_statuses[rank].ok() && read_len < chunk.stripe_bytes) {
            std::memset(host_buffer + read_len, 0, chunk.stripe_bytes - read_len);
          }
        } else {
          std::memset(host_buffer, 0, chunk.stripe_bytes);
        }
        if (!read_statuses[rank].ok()) {
          break;
        }
        TC_RETURN_IF_ERROR(set_device_cached(participants[rank].device_id));
        TC_RETURN_IF_ERROR(
            tensorcast::cuda::memcpy_async(
                rank_send_stages[rank]->get(),
                host_buffer,
                chunk.stripe_bytes,
                cudaMemcpyHostToDevice,
                clique->stream(static_cast<int>(rank))));
      }
      for (const auto& status : read_statuses) {
        TC_RETURN_IF_ERROR(status);
      }
      read_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();
    }
    if (chunk_uses_consumer_routed(chunk)) {
      return issue_consumer_routed_chunk_from_parallel_slot(chunk, /*slot=*/0, /*issue_h2d_from_host_slot=*/false);
    }
    return issue_gpu_chunk_from_parallel_slot(chunk, /*slot=*/0, /*issue_h2d_from_host_slot=*/false);
  };

  if (parallel_host_buffers) {
    size_t next_launch_chunk = 0;
    const size_t initial_read_ahead = std::min(active_pipeline_slots, runtime_chunks.size());
    for (; next_launch_chunk < initial_read_ahead; ++next_launch_chunk) {
      TC_RETURN_IF_ERROR(launch_parallel_chunk_read(next_launch_chunk));
    }
    double read_job_max_sec = 0.0;
    for (size_t chunk_index = 0; chunk_index < runtime_chunks.size(); ++chunk_index) {
      const size_t slot = chunk_index % active_pipeline_slots;
      SourceWindowReadAheadResult read_result = wait_parallel_chunk_read(chunk_index);
      for (const auto& status : read_result.statuses) {
        TC_RETURN_IF_ERROR(status);
      }
      read_sec += read_result.read_sec;
      read_job_sec += read_result.read_job_sec;
      read_job_max_sec = std::max(read_job_max_sec, read_result.read_job_sec);
      bytes_read += runtime_chunks[chunk_index].chunk_len;
      chunk_count += 1;
      if (chunk_uses_consumer_routed(runtime_chunks[chunk_index])) {
        TC_RETURN_IF_ERROR(issue_consumer_routed_chunk_from_parallel_slot(
            runtime_chunks[chunk_index], slot, /*issue_h2d_from_host_slot=*/true));
      } else if (chunk_uses_local_only(runtime_chunks[chunk_index])) {
        TC_RETURN_IF_ERROR(issue_local_only_chunk_from_parallel_slot(
            runtime_chunks[chunk_index], slot, /*issue_h2d_from_host_slot=*/true));
      } else {
        TC_RETURN_IF_ERROR(
            issue_gpu_chunk_from_parallel_slot(runtime_chunks[chunk_index], slot, /*issue_h2d_from_host_slot=*/true));
      }
      if (next_launch_chunk < runtime_chunks.size()) {
        TC_RETURN_IF_ERROR(launch_parallel_chunk_read(next_launch_chunk));
        ++next_launch_chunk;
      }
    }
    LOG(INFO) << "source_window_collective_read_ahead_profile"
              << " artifact_id=" << participants.front().artifact_id << " group_id=" << plan.group.group_id
              << " chunks=" << runtime_chunks.size()
              << " active_pipeline_slots=" << active_pipeline_slots
              << " read_job_avg=" << (runtime_chunks.empty() ? 0.0 : read_job_sec / runtime_chunks.size()) << "s"
              << " read_job_max=" << read_job_max_sec << "s";
  } else {
    for (const auto& chunk : runtime_chunks) {
      bytes_read += chunk.chunk_len;
      chunk_count += 1;
      TC_RETURN_IF_ERROR(execute_single_buffer_chunk(chunk));
    }
  }
  drain_batched_scatter_host_descriptor_slots();
  log_source_window_collective_memory(
      "after_data_plane",
      participants.front().artifact_id,
      plan.group.group_id,
      capture_source_window_device_memory(device_ids),
      &memory_entry);
  evict_clique_on_exit = true;

  auto metrics = source_window_metrics_from_summary(plan.summary);
  metrics.unique_source_bytes = bytes_read;
  metrics.peer_transfer_bytes = actual_peer_transfer_bytes;
  const uint64_t gpu_stage_bytes = configured_chunk_bytes * static_cast<uint64_t>(participants.size()) +
      max_stripe_bytes * static_cast<uint64_t>(participants.size());
  const uint64_t host_stage_bytes = parallel_host_buffers ? configured_chunk_bytes *
          static_cast<uint64_t>(participants.size()) * static_cast<uint64_t>(active_pipeline_slots)
                                                          : configured_chunk_bytes;
  const uint64_t routed_stage_bytes = has_consumer_routed_windows ? 2ULL * max_stripe_bytes *
          static_cast<uint64_t>(participants.size()) * static_cast<uint64_t>(participants.size() - 1)
                                                                  : 0;
  metrics.peak_temporary_bytes = gpu_stage_bytes + host_stage_bytes + routed_stage_bytes;
  metrics.batch_count = chunk_count;
  metrics.source_window_peer_transfer_bytes = actual_peer_transfer_bytes;
  if (use_consumer_routed) {
    metrics.source_window_peer_useful_bytes = actual_peer_transfer_bytes;
    metrics.source_window_peer_waste_bytes = 0;
  } else if (actual_peer_transfer_bytes > metrics.source_window_peer_useful_bytes) {
    metrics.source_window_peer_waste_bytes = actual_peer_transfer_bytes - metrics.source_window_peer_useful_bytes;
  }
  metrics.source_window_scatter_op_count = actual_scatter_ops;

  auto join_atomic_u64 = [](const std::vector<std::atomic<uint64_t>>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i != 0) {
        absl::StrAppend(&out, ",");
      }
      absl::StrAppend(&out, values[i].load(std::memory_order_relaxed));
    }
    absl::StrAppend(&out, "]");
    return out;
  };
  auto join_rank_read_sec = [&]() {
    std::string out = "[";
    for (size_t i = 0; i < rank_read_ns.size(); ++i) {
      if (i != 0) {
        absl::StrAppend(&out, ",");
      }
      const double sec = static_cast<double>(rank_read_ns[i].load(std::memory_order_relaxed)) / 1e9;
      absl::StrAppend(&out, sec);
    }
    absl::StrAppend(&out, "]");
    return out;
  };

  const auto total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  logged_total_sec = total_sec;
  logged_success = true;
  LOG(INFO) << "source_window_collective_mapped_target timings"
            << " artifact_id=" << participants.front().artifact_id << " group_id=" << plan.group.group_id
            << " windows=" << plan.windows.size() << " chunks=" << chunk_count
            << " chunk_bytes=" << configured_chunk_bytes
            << " stripe_buffer_bytes=" << chunk_sizing.stripe_buffer_bytes
            << " max_collective_chunk_bytes=" << max_collective_chunk_bytes
            << " max_stripe_bytes=" << max_stripe_bytes
            << " parallel_host_buffers=" << (parallel_host_buffers ? 1 : 0)
            << " requested_pipeline_slots=" << requested_pipeline_slots
            << " active_pipeline_slots=" << active_pipeline_slots << " bytes_read=" << bytes_read
            << " peer_transfer_bytes=" << actual_peer_transfer_bytes << " scatter_ops=" << actual_scatter_ops
            << " target_storage_fast_path_pieces=" << target_storage_fast_path_pieces
            << " target_storage_fast_path_bytes=" << target_storage_fast_path_bytes
            << " runtime_chunk_unfiltered_consumer_span_refs=" << runtime_chunk_unfiltered_consumer_span_refs
            << " runtime_chunk_prefiltered_consumer_span_refs=" << runtime_chunk_prefiltered_consumer_span_refs
            << " batched_scatter_kernel_enabled=" << (batched_scatter_kernel_enabled ? 1 : 0)
            << " batched_scatter_kernel_launches=" << batched_scatter_kernel_launches
            << " batched_scatter_kernel_descriptors=" << batched_scatter_kernel_descriptors
            << " batched_scatter_fallback_launches=" << batched_scatter_fallback_launches
            << " batched_scatter_fallback_descriptors=" << batched_scatter_fallback_descriptors
            << " batched_scatter_capacity_fallback_chunks=" << batched_scatter_capacity_fallback_chunks
            << " batched_routed_pack_kernel_launches=" << batched_routed_pack_kernel_launches
            << " batched_routed_pack_kernel_descriptors=" << batched_routed_pack_kernel_descriptors
            << " batched_routed_pack_fallback_launches=" << batched_routed_pack_fallback_launches
            << " batched_routed_pack_fallback_descriptors=" << batched_routed_pack_fallback_descriptors
            << " batched_scatter_descriptor_build=" << batched_scatter_descriptor_build_sec << "s"
            << " batched_scatter_descriptor_host_copy=" << batched_scatter_descriptor_host_copy_sec << "s"
            << " batched_scatter_descriptor_slot_wait=" << batched_scatter_descriptor_slot_wait_sec << "s"
            << " batched_scatter_descriptor_final_drain=" << batched_scatter_descriptor_final_drain_sec << "s"
            << " batched_scatter_kernel_submit=" << batched_scatter_kernel_submit_sec << "s"
            << " batched_scatter_fallback_submit=" << batched_scatter_fallback_submit_sec << "s"
            << " routed_span_plan=" << routed_span_plan_sec << "s"
            << " routed_pack_descriptor_build=" << routed_pack_descriptor_build_sec << "s"
            << " routed_pack_issue=" << routed_pack_issue_sec << "s"
            << " routed_local_descriptor_build=" << routed_local_descriptor_build_sec << "s"
            << " routed_local_issue=" << routed_local_issue_sec << "s"
            << " routed_remote_descriptor_build=" << routed_remote_descriptor_build_sec << "s"
            << " routed_remote_scatter_issue=" << routed_remote_scatter_issue_sec << "s"
            << " routed_compiled_program_enabled=" << (compiled_routed_program_enabled ? 1 : 0)
            << " routed_compiled_program_build=" << routed_compiled_program_build_sec << "s"
            << " routed_compiled_program_build_threads=" << routed_compiled_program_build_threads
            << " routed_compiled_program_key=" << routed_compiled_program_key_sec << "s"
            << " routed_compiled_program_lookup=" << routed_compiled_program_lookup_sec << "s"
            << " routed_compiled_program_wait=" << routed_compiled_program_wait_sec << "s"
            << " routed_compiled_program_cache_store=" << routed_compiled_program_cache_store_sec << "s"
            << " routed_compiled_program_cache_eligible=" << (routed_compiled_program_cache_eligible ? 1 : 0)
            << " routed_compiled_program_cache_hit=" << (routed_compiled_program_cache_hit ? 1 : 0)
            << " routed_compiled_program_cache_waited=" << (routed_compiled_program_cache_waited ? 1 : 0)
            << " routed_compiled_program_cache_size_mismatch=" << (routed_compiled_program_cache_size_mismatch ? 1 : 0)
            << " routed_compiled_program_chunks=" << routed_compiled_program_chunks
            << " routed_compiled_program_local_descriptors=" << routed_compiled_program_local_descriptors
            << " routed_compiled_program_pack_descriptors=" << routed_compiled_program_pack_descriptors
            << " routed_compiled_program_packed_remote_pieces=" << routed_compiled_program_packed_remote_pieces
            << " routed_compiled_program_direct_remote_pieces=" << routed_compiled_program_direct_remote_pieces
            << " scatter_cuda_graph_enabled=" << (scatter_cuda_graph_enabled ? 1 : 0)
            << " scatter_cuda_graph_launches=" << scatter_cuda_graph_launches
            << " scatter_cuda_graph_descriptors=" << scatter_cuda_graph_descriptors
            << " scatter_cuda_graph_nodes=" << scatter_cuda_graph_nodes
            << " scatter_cuda_graph_fallback_chunks=" << scatter_cuda_graph_fallback_chunks
            << " scatter_cuda_graph_build=" << scatter_cuda_graph_build_sec << "s"
            << " routed_pack_ops=" << routed_pack_ops << " routed_packed_pairs=" << routed_packed_pairs
            << " routed_deferred_2d_pack_ops=" << routed_deferred_2d_pack_ops
            << " routed_local_2d_pieces=" << routed_local_2d_pieces << " routed_local_pieces=" << routed_local_pieces
            << " routed_remote_pieces=" << routed_remote_pieces << " routed_stage_bytes=" << routed_stage_bytes
            << " cuda_device_switches=" << cuda_device_switches
            << " direct_pinned_read_attempts=" << direct_pinned_read_attempts.load(std::memory_order_relaxed)
            << " direct_pinned_read_successes=" << direct_pinned_read_successes.load(std::memory_order_relaxed)
            << " direct_pinned_read_fallbacks=" << direct_pinned_read_fallbacks.load(std::memory_order_relaxed)
            << " direct_pinned_read_success_bytes=" << direct_pinned_read_success_bytes.load(std::memory_order_relaxed)
            << " direct_pinned_read_fallback_bytes="
            << direct_pinned_read_fallback_bytes.load(std::memory_order_relaxed)
            << " direct_pinned_fallback_unaligned_host="
            << direct_pinned_fallback_unaligned_host.load(std::memory_order_relaxed)
            << " direct_pinned_fallback_outside_segment="
            << direct_pinned_fallback_outside_segment.load(std::memory_order_relaxed)
            << " direct_pinned_fallback_cross_segment="
            << direct_pinned_fallback_cross_segment.load(std::memory_order_relaxed)
            << " direct_pinned_fallback_file_edge=" << direct_pinned_fallback_file_edge.load(std::memory_order_relaxed)
            << " direct_pinned_fallback_capacity=" << direct_pinned_fallback_capacity.load(std::memory_order_relaxed)
            << " clique_init=" << clique_sec << "s"
            << " clique_cache_hit=" << (clique_cache_hit ? 1 : 0) << " pinned_alloc=" << pinned_alloc_sec << "s"
            << " stage_alloc=" << stage_alloc_sec << "s"
            << " read=" << read_sec << "s"
            << " read_job_sum=" << read_job_sec << "s"
            << " rank_read_sec=" << join_rank_read_sec()
            << " rank_read_bytes=" << join_atomic_u64(rank_read_bytes)
            << " rank_read_calls=" << join_atomic_u64(rank_read_calls)
            << " rank_zero_fill_bytes=" << join_atomic_u64(rank_zero_fill_bytes)
            << " rank_zero_fill_calls=" << join_atomic_u64(rank_zero_fill_calls)
            << " h2d=" << h2d_sec << "s"
            << " collective_issue=" << collective_sec << "s"
            << " collective_sync=" << collective_sync_sec << "s"
            << " scatter_issue=" << scatter_issue_sec << "s"
            << " scatter_sync=" << scatter_sync_sec << "s"
            << " total=" << total_sec << "s";
  return metrics;
}

SourceWindowCollectiveMappedTargetLoadResult execute_source_window_group_final_admission(
    const loading::CollectiveLoadGroupHint& group,
    const std::vector<SourceWindowMappedParticipant>& participants,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options) {
  const auto plan_start = std::chrono::steady_clock::now();
  const auto input_start = std::chrono::steady_clock::now();
  auto input_or = build_source_window_collective_group_input(group, participants, options);
  const double input_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - input_start).count();
  const SourceWindowCollectiveConfig config = source_window_collective_config_from_strategy(options.strategy_config);
  auto attach_plan_cache_hit = [](SourceWindowCollectiveMappedTargetLoadResult result, bool plan_cache_hit) {
    result.plan_cache_hit = plan_cache_hit;
    return result;
  };
  if (!input_or.ok()) {
    const double plan_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - plan_start).count();
    LOG(INFO) << "source_window_collective_plan"
              << " artifact_id=" << participants.front().artifact_id << " group_id=" << group.group_id
              << " tp_size=" << group.world_size << " admitted=0"
              << " reject_reason=" << input_or.status().message() << " input_sec=" << input_sec
              << " key_sec=0 lookup_sec=0 build_sec=0 cache_store_sec=0"
              << " plan_cache_enabled=" << (options.enable_source_window_plan_cache ? 1 : 0)
              << " plan_cache_hit=0 plan_cache_key_kind=none"
              << " plan_sec=" << plan_sec;
    return source_window_group_rejected_result(config, std::string(input_or.status().message()));
  }
  const SourceWindowPreparedRealizationFactStats prepared_fact_stats =
      source_window_prepared_realization_fact_stats(*input_or);
  std::string plan_cache_key;
  std::string plan_cache_key_kind = "none";
  double key_sec = 0.0;
  SourceWindowCollectivePlan plan;
  bool plan_cache_hit = false;
  double lookup_sec = 0.0;
  if (options.enable_source_window_plan_cache) {
    const auto key_start = std::chrono::steady_clock::now();
    if (auto prepared_key =
            source_window_collective_prepared_plan_cache_key(participants.front().artifact_id, *input_or);
        prepared_key.has_value()) {
      plan_cache_key = std::move(*prepared_key);
      plan_cache_key_kind = "prepared";
    } else {
      plan_cache_key = source_window_collective_plan_cache_key(participants.front().artifact_id, *input_or);
      plan_cache_key_kind = "full_work_plan";
    }
    key_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - key_start).count();
    const auto lookup_start = std::chrono::steady_clock::now();
    plan_cache_hit = lookup_source_window_collective_plan_cache(plan_cache_key, group, &plan);
    lookup_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - lookup_start).count();
  }
  absl::Status plan_status;
  double build_sec = 0.0;
  double cache_store_sec = 0.0;
  if (!plan_cache_hit) {
    const auto build_start = std::chrono::steady_clock::now();
    auto plan_or = build_source_window_collective_plan(*input_or);
    build_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();
    if (!plan_or.ok()) {
      plan_status = plan_or.status();
      if (options.enable_source_window_plan_cache && !plan_cache_key.empty()) {
        abandon_source_window_collective_plan_cache_build(plan_cache_key);
      }
    } else {
      plan = std::move(*plan_or);
      if (options.enable_source_window_plan_cache) {
        const auto cache_store_start = std::chrono::steady_clock::now();
        store_source_window_collective_plan_cache(plan_cache_key, plan);
        cache_store_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - cache_store_start).count();
      }
    }
  }
  const double plan_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - plan_start).count();
  if (!plan_status.ok()) {
    LOG(INFO) << "source_window_collective_plan"
              << " artifact_id=" << participants.front().artifact_id << " group_id=" << group.group_id
              << " tp_size=" << group.world_size << " admitted=0"
              << " reject_reason=" << plan_status.message() << " input_sec=" << input_sec << " key_sec=" << key_sec
              << " lookup_sec=" << lookup_sec << " build_sec=" << build_sec << " cache_store_sec=" << cache_store_sec
              << " plan_cache_enabled=" << (options.enable_source_window_plan_cache ? 1 : 0)
              << " plan_cache_hit=" << (plan_cache_hit ? 1 : 0) << " plan_cache_key_kind=" << plan_cache_key_kind
              << " plan_cache_key=" << plan_cache_key
              << " prepared_realization_members=" << prepared_fact_stats.member_count
              << " prepared_realization_group_key_unique=" << prepared_fact_stats.group_key_unique
              << " prepared_realization_member_key_unique=" << prepared_fact_stats.member_key_unique
              << " prepared_realization_plan_hash_unique=" << prepared_fact_stats.realization_plan_hash_unique
              << " prepared_realization_target_layout_template_hash_unique="
              << prepared_fact_stats.target_layout_template_hash_unique
              << " prepared_realization_target_index_hash_unique=" << prepared_fact_stats.target_index_hash_unique
              << " plan_sec=" << plan_sec;
    return attach_plan_cache_hit(
        source_window_group_rejected_result(config, std::string(plan_status.message())), plan_cache_hit);
  }
  const bool admitted = plan.summary.group_final_admitted;
  LOG(INFO) << "source_window_collective_plan"
            << " artifact_id=" << participants.front().artifact_id << " group_id=" << group.group_id
            << " tp_size=" << group.world_size << " admitted=" << (admitted ? 1 : 0)
            << " reject_reason=" << plan.summary.group_reject_reason
            << " windows=" << plan.summary.source_window_window_count
            << " group_disk_read_bytes=" << plan.summary.source_window_group_disk_read_bytes
            << " rank_read_bytes_max=" << plan.summary.source_window_rank_read_bytes_max
            << " local_rank_read_bytes_max=" << plan.summary.source_window_local_rank_read_bytes_max
            << " rank_read_saving_bytes=" << plan.summary.source_window_rank_read_saving_bytes
            << " unique_payload_bytes=" << plan.summary.source_window_unique_payload_bytes
            << " target_write_bytes=" << plan.summary.source_window_target_write_bytes
            << " read_amplification_x1000=" << plan.summary.source_window_read_amplification_x1000
            << " peer_transfer_bytes=" << plan.summary.source_window_peer_transfer_bytes
            << " peer_useful_bytes=" << plan.summary.source_window_peer_useful_bytes
            << " peer_waste_bytes=" << plan.summary.source_window_peer_waste_bytes
            << " scatter_ops=" << plan.summary.source_window_scatter_op_count
            << " residual_bytes=" << plan.summary.source_window_residual_bytes << " distribution_mode="
            << runtime::ingestion::strategy::source_window_collective_distribution_mode_name(plan.distribution_mode)
            << " plan_hash=" << plan.plan_hash << " input_sec=" << input_sec << " key_sec=" << key_sec
            << " lookup_sec=" << lookup_sec << " build_sec=" << build_sec << " cache_store_sec=" << cache_store_sec
            << " plan_cache_enabled=" << (options.enable_source_window_plan_cache ? 1 : 0)
            << " plan_cache_hit=" << (plan_cache_hit ? 1 : 0) << " plan_cache_key_kind=" << plan_cache_key_kind
            << " plan_cache_key=" << plan_cache_key
            << " prepared_realization_members=" << prepared_fact_stats.member_count
            << " prepared_realization_group_key_unique=" << prepared_fact_stats.group_key_unique
            << " prepared_realization_member_key_unique=" << prepared_fact_stats.member_key_unique
            << " prepared_realization_plan_hash_unique=" << prepared_fact_stats.realization_plan_hash_unique
            << " prepared_realization_target_layout_template_hash_unique="
            << prepared_fact_stats.target_layout_template_hash_unique
            << " prepared_realization_target_index_hash_unique=" << prepared_fact_stats.target_index_hash_unique
            << " plan_sec=" << plan_sec;
  if (verbose_materialization_strategy_diagnostics(options.strategy_config)) {
    const auto tensor_stage_start = std::chrono::steady_clock::now();
    auto tensor_stage_summary_or = summarize_source_window_tensor_staged_copy(*input_or);
    const double tensor_stage_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tensor_stage_start).count();
    if (tensor_stage_summary_or.ok()) {
      const auto& summary = *tensor_stage_summary_or;
      LOG(INFO) << "source_window_tensor_staged_feasibility"
                << " artifact_id=" << participants.front().artifact_id << " group_id=" << group.group_id
                << " tp_size=" << group.world_size << " feasible=" << (summary.feasible ? 1 : 0)
                << " reject_reason=" << summary.reject_reason << " rank_count=" << summary.rank_count
                << " source_fragment_count=" << summary.source_fragment_count
                << " destination_tensor_count=" << summary.destination_tensor_count
                << " source_tensor_count=" << summary.source_tensor_count
                << " eligible_bytes=" << summary.eligible_bytes << " ineligible_bytes=" << summary.ineligible_bytes
                << " raw_copy_ops=" << summary.raw_copy_ops
                << " tensor_staged_copy_ops=" << summary.tensor_staged_copy_ops
                << " linear_copy_ops=" << summary.linear_copy_ops << " copy_2d_ops=" << summary.copy_2d_ops
                << " max_tensor_staged_copy_ops_per_rank=" << summary.max_tensor_staged_copy_ops_per_rank
                << " estimated_op_reduction_x1000=" << summary.estimated_op_reduction_x1000
                << " summary_sec=" << tensor_stage_sec;
    } else {
      LOG(INFO) << "source_window_tensor_staged_feasibility"
                << " artifact_id=" << participants.front().artifact_id << " group_id=" << group.group_id
                << " tp_size=" << group.world_size << " feasible=0"
                << " reject_reason=" << tensor_stage_summary_or.status().message()
                << " summary_sec=" << tensor_stage_sec;
    }
    uint64_t diagnostic_chunk_bytes = options.chunk_bytes;
    if (pinned_pool != nullptr && pinned_pool->slice_bytes() > 0) {
      diagnostic_chunk_bytes = diagnostic_chunk_bytes == 0
          ? static_cast<uint64_t>(pinned_pool->slice_bytes())
          : std::min<uint64_t>(diagnostic_chunk_bytes, pinned_pool->slice_bytes());
    }
    const auto batched_scatter_start = std::chrono::steady_clock::now();
    auto batched_scatter_summary_or = summarize_source_window_batched_scatter(plan, diagnostic_chunk_bytes);
    const double batched_scatter_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - batched_scatter_start).count();
    if (batched_scatter_summary_or.ok()) {
      const auto& summary = *batched_scatter_summary_or;
      LOG(INFO) << "source_window_batched_scatter_feasibility"
                << " artifact_id=" << participants.front().artifact_id << " group_id=" << group.group_id
                << " tp_size=" << group.world_size << " feasible=" << (summary.feasible ? 1 : 0)
                << " reject_reason=" << summary.reject_reason << " rank_count=" << summary.rank_count
                << " window_count=" << summary.window_count << " runtime_chunk_bytes=" << summary.runtime_chunk_bytes
                << " estimated_runtime_chunk_count=" << summary.estimated_runtime_chunk_count
                << " consumer_span_count=" << summary.consumer_span_count
                << " full_window_all_gather_windows=" << summary.full_window_all_gather_windows
                << " consumer_routed_windows=" << summary.consumer_routed_windows
                << " local_only_windows=" << summary.local_only_windows
                << " target_write_bytes=" << summary.target_write_bytes
                << " estimated_current_copy_launches=" << summary.estimated_current_copy_launches
                << " estimated_current_scatter_launches=" << summary.estimated_current_scatter_launches
                << " estimated_current_pack_launches=" << summary.estimated_current_pack_launches
                << " estimated_current_linear_copy_launches=" << summary.estimated_current_linear_copy_launches
                << " estimated_current_copy_2d_launches=" << summary.estimated_current_copy_2d_launches
                << " batched_total_copy_launches=" << summary.batched_total_copy_launches
                << " batched_scatter_launches=" << summary.batched_scatter_launches
                << " batched_pack_launches=" << summary.batched_pack_launches
                << " max_descriptors_per_batched_scatter=" << summary.max_descriptors_per_batched_scatter
                << " max_descriptors_per_batched_pack=" << summary.max_descriptors_per_batched_pack
                << " estimated_copy_launch_reduction_x1000=" << summary.estimated_copy_launch_reduction_x1000
                << " summary_sec=" << batched_scatter_sec;
    } else {
      LOG(INFO) << "source_window_batched_scatter_feasibility"
                << " artifact_id=" << participants.front().artifact_id << " group_id=" << group.group_id
                << " tp_size=" << group.world_size << " feasible=0"
                << " reject_reason=" << batched_scatter_summary_or.status().message()
                << " summary_sec=" << batched_scatter_sec;
    }
  }
  if (!admitted) {
    return attach_plan_cache_hit(
        source_window_group_rejected_result(config, plan, plan.summary.group_reject_reason), plan_cache_hit);
  }
  if (config.selection_mode == runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kDryRun) {
    SourceWindowCollectiveMappedTargetLoadResult result;
    result.handled = false;
    result.status = absl::OkStatus();
    result.metrics = source_window_metrics_from_summary(plan.summary);
    result.skip_reason = "source_window_dry_run";
    result.plan_hash = plan.plan_hash;
    result.plan_cache_hit = plan_cache_hit;
    return result;
  }
  if (pinned_pool == nullptr) {
    return attach_plan_cache_hit(
        source_window_runtime_unavailable_result(config, plan, "pinned_pool_missing"), plan_cache_hit);
  }
  if (plan.distribution_mode !=
          runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kFullWindowAllGather &&
      plan.distribution_mode != runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted &&
      plan.distribution_mode != runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kHybridWindow &&
      plan.distribution_mode != runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kLocalOnly) {
    return attach_plan_cache_hit(
        source_window_runtime_unavailable_result(config, plan, "distribution_unsupported"), plan_cache_hit);
  }
  auto metrics_or = execute_source_window_collective_mapped(plan, participants, pinned_pool, pinned_timeout, options);
  if (!metrics_or.ok()) {
    return attach_plan_cache_hit(source_window_runtime_failure_result(plan, metrics_or.status()), plan_cache_hit);
  }
  return attach_plan_cache_hit(source_window_runtime_success_result(plan, std::move(*metrics_or)), plan_cache_hit);
}

SourceWindowCollectiveMappedTargetLoadResult wait_for_source_window_mapped_group_and_maybe_execute(
    const SourceWindowCollectiveMappedTargetLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options) {
  const auto call_start = std::chrono::steady_clock::now();
  const SourceWindowCollectiveConfig config = source_window_collective_config_from_strategy(options.strategy_config);
  auto work_plan_ref = source_window_request_work_plan_ref(request);
  auto target_layout_ref = source_window_request_target_layout_ref(request);
  if (work_plan_ref == nullptr || target_layout_ref == nullptr) {
    return source_window_unhandled_or_strict_failure(config, "request_incomplete");
  }
  auto storage_spans_or = build_target_storage_spans(*target_layout_ref);
  if (!storage_spans_or.ok()) {
    return source_window_unhandled_or_strict_failure(
        config, "invalid_target_storage_layout", storage_spans_or.status());
  }
  auto parsed = SourceWindowMappedParticipant{
      .artifact_id = request.artifact_id,
      .rank = static_cast<int>(request.group.rank),
      .device_id = request.device_id,
      .disk_context = request.disk_context,
      .work_plan = std::move(work_plan_ref),
      .target_layout = std::move(target_layout_ref),
      .storage_spans = std::move(*storage_spans_or),
      .candidate_summary = request.candidate_summary,
      .source_index_digest = request.source_index_digest,
      .prepared_realization = request.prepared_realization,
  };
  const auto request_prepared = std::chrono::steady_clock::now();
  std::shared_ptr<SourceWindowMappedGroupState> state;
  {
    absl::MutexLock lock(&g_source_window_mapped_group_mu);
    auto& slot = g_source_window_mapped_groups[request.group.group_id];
    if (slot == nullptr) {
      slot = std::make_shared<SourceWindowMappedGroupState>(request.group.world_size);
    }
    state = slot;
  }
  const auto group_state_ready = std::chrono::steady_clock::now();

  bool leader = false;
  bool erase_empty_group = false;
  bool timed_out = false;
  uint32_t joined_after = 0;
  const auto assemble_start = std::chrono::steady_clock::now();
  {
    absl::MutexLock lock(&state->mu);
    if (request.group.rank >= state->world_size) {
      return {
          .handled = false,
          .status = absl::InvalidArgumentError("source-window collective rank out of range"),
          .skip_reason = "collective_rank_out_of_range",
      };
    }
    auto& slot = state->participants[request.group.rank];
    if (!slot.has_value()) {
      slot = std::move(parsed);
      state->joined += 1;
      LOG(INFO) << "source_window_collective_mapped_target join group_id=" << request.group.group_id
                << " rank=" << request.group.rank << " joined=" << state->joined << "/" << state->world_size;
    }
    joined_after = state->joined;
    if (state->joined == state->world_size) {
      state->launching = true;
      leader = true;
      state->cv.SignalAll();
      LOG(INFO) << "source_window_collective_mapped_target launching group_id=" << request.group.group_id
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
        LOG(INFO) << "source_window_collective_mapped_target timeout group_id=" << request.group.group_id
                  << " rank=" << request.group.rank << " remaining=" << state->joined;
      }
    }
  }
  const auto assemble_done = std::chrono::steady_clock::now();
  if (erase_empty_group) {
    absl::MutexLock group_lock(&g_source_window_mapped_group_mu);
    g_source_window_mapped_groups.erase(request.group.group_id);
    auto result = source_window_unhandled_or_strict_failure(config, "group_assemble_timeout");
    const auto done = std::chrono::steady_clock::now();
    LOG(INFO) << "source_window_collective_mapped_target call_profile"
              << " group_id=" << request.group.group_id << " rank=" << request.group.rank
              << " leader=0 timeout=1 joined_after=" << joined_after
              << " request_prep_sec=" << std::chrono::duration<double>(request_prepared - call_start).count()
              << " group_state_sec=" << std::chrono::duration<double>(group_state_ready - request_prepared).count()
              << " assemble_wait_sec=" << std::chrono::duration<double>(assemble_done - assemble_start).count()
              << " result_wait_sec=0 leader_execute_sec=0 total_sec="
              << std::chrono::duration<double>(done - call_start).count() << " handled=" << result.handled
              << " status_ok=" << result.status.ok() << " skip_reason=" << result.skip_reason;
    return result;
  }
  if (timed_out) {
    auto result = source_window_unhandled_or_strict_failure(config, "group_assemble_timeout");
    const auto done = std::chrono::steady_clock::now();
    LOG(INFO) << "source_window_collective_mapped_target call_profile"
              << " group_id=" << request.group.group_id << " rank=" << request.group.rank
              << " leader=0 timeout=1 joined_after=" << joined_after
              << " request_prep_sec=" << std::chrono::duration<double>(request_prepared - call_start).count()
              << " group_state_sec=" << std::chrono::duration<double>(group_state_ready - request_prepared).count()
              << " assemble_wait_sec=" << std::chrono::duration<double>(assemble_done - assemble_start).count()
              << " result_wait_sec=0 leader_execute_sec=0 total_sec="
              << std::chrono::duration<double>(done - call_start).count() << " handled=" << result.handled
              << " status_ok=" << result.status.ok() << " skip_reason=" << result.skip_reason;
    return result;
  }

  if (leader) {
    std::vector<SourceWindowMappedParticipant> participants;
    participants.reserve(state->world_size);
    const auto collect_start = std::chrono::steady_clock::now();
    {
      absl::MutexLock lock(&state->mu);
      for (auto& participant : state->participants) {
        if (!participant.has_value()) {
          state->result = {
              .handled = true,
              .status = absl::FailedPreconditionError("source-window mapped collective group is incomplete"),
              .skip_reason = "group_incomplete",
          };
          state->complete = true;
          state->cv.SignalAll();
          return state->result;
        }
        participants.push_back(std::move(*participant));
        participant.reset();
      }
    }
    const auto collect_done = std::chrono::steady_clock::now();
    std::sort(
        participants.begin(),
        participants.end(),
        [](const SourceWindowMappedParticipant& a, const SourceWindowMappedParticipant& b) { return a.rank < b.rank; });
    const auto execute_start = std::chrono::steady_clock::now();
    SourceWindowCollectiveMappedTargetLoadResult result =
        execute_source_window_group_final_admission(request.group, participants, pinned_pool, pinned_timeout, options);
    const auto execute_done = std::chrono::steady_clock::now();
    LOG(INFO) << "source_window_collective_mapped_target finished group_id=" << request.group.group_id
              << " handled=" << result.handled << " status=" << result.status << " skip_reason=" << result.skip_reason;
    {
      absl::MutexLock lock(&state->mu);
      state->result = result;
      state->complete = true;
      state->cv.SignalAll();
    }
    const auto publish_done = std::chrono::steady_clock::now();
    {
      absl::MutexLock lock(&g_source_window_mapped_group_mu);
      g_source_window_mapped_groups.erase(request.group.group_id);
    }
    const auto done = std::chrono::steady_clock::now();
    LOG(INFO) << "source_window_collective_mapped_target call_profile"
              << " group_id=" << request.group.group_id << " rank=" << request.group.rank
              << " leader=1 timeout=0 joined_after=" << joined_after
              << " request_prep_sec=" << std::chrono::duration<double>(request_prepared - call_start).count()
              << " group_state_sec=" << std::chrono::duration<double>(group_state_ready - request_prepared).count()
              << " assemble_wait_sec=" << std::chrono::duration<double>(assemble_done - assemble_start).count()
              << " participant_collect_sec=" << std::chrono::duration<double>(collect_done - collect_start).count()
              << " leader_execute_sec=" << std::chrono::duration<double>(execute_done - execute_start).count()
              << " publish_sec=" << std::chrono::duration<double>(publish_done - execute_done).count()
              << " result_wait_sec=0 total_sec=" << std::chrono::duration<double>(done - call_start).count()
              << " handled=" << result.handled << " status_ok=" << result.status.ok()
              << " skip_reason=" << result.skip_reason;
    return result;
  }

  SourceWindowCollectiveMappedTargetLoadResult result;
  const auto result_wait_start = std::chrono::steady_clock::now();
  {
    absl::MutexLock lock(&state->mu);
    while (!state->complete) {
      state->cv.Wait(&state->mu);
    }
    result = state->result;
  }
  const auto done = std::chrono::steady_clock::now();
  LOG(INFO) << "source_window_collective_mapped_target call_profile"
            << " group_id=" << request.group.group_id << " rank=" << request.group.rank
            << " leader=0 timeout=0 joined_after=" << joined_after
            << " request_prep_sec=" << std::chrono::duration<double>(request_prepared - call_start).count()
            << " group_state_sec=" << std::chrono::duration<double>(group_state_ready - request_prepared).count()
            << " assemble_wait_sec=" << std::chrono::duration<double>(assemble_done - assemble_start).count()
            << " result_wait_sec=" << std::chrono::duration<double>(done - result_wait_start).count()
            << " leader_execute_sec=0 total_sec=" << std::chrono::duration<double>(done - call_start).count()
            << " handled=" << result.handled << " status_ok=" << result.status.ok()
            << " skip_reason=" << result.skip_reason;
  return result;
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

absl::StatusOr<LocalMappedSafetensorsAutoIoDecision> choose_auto_local_mapped_safetensors_io_for_testing(
    absl::Span<const loader::SharedSafetensorsSegment> segments) {
  return choose_auto_local_mapped_safetensors_io(segments);
}

void clear_source_window_collective_plan_cache_for_testing() {
  clear_source_window_collective_plan_cache();
}

SourceWindowCollectivePlanCacheStats source_window_collective_plan_cache_stats_for_testing() {
  return source_window_collective_plan_cache_stats_snapshot();
}

void clear_source_window_routed_program_cache_for_testing() {
  clear_source_window_routed_program_cache();
}

SourceWindowCollectivePlanCacheStats source_window_routed_program_cache_stats_for_testing() {
  return source_window_routed_program_cache_stats_snapshot();
}

size_t source_window_compiled_routed_program_build_thread_count_for_testing(
    size_t chunk_count, uint32_t configured_thread_count) {
  return compiled_routed_program_build_thread_count(chunk_count, configured_thread_count);
}

absl::StatusOr<std::string> source_window_routed_program_cache_key_for_testing(
    std::string_view artifact_id,
    const SourceWindowCollectivePlan& plan,
    absl::Span<const SourceWindowCollectiveMappedTargetLoadRequest> requests,
    size_t configured_chunk_bytes,
    size_t max_collective_chunk_bytes,
    size_t max_stripe_bytes) {
  std::vector<SourceWindowMappedParticipant> participants;
  participants.reserve(requests.size());
  for (const auto& request : requests) {
    auto storage_spans_or = build_target_storage_spans(source_window_request_target_layout(request));
    if (!storage_spans_or.ok()) {
      return storage_spans_or.status();
    }
    participants.push_back(
        SourceWindowMappedParticipant{
            .artifact_id = request.artifact_id,
            .rank = static_cast<int>(request.group.rank),
            .device_id = request.device_id,
            .disk_context = request.disk_context,
            .target_layout = source_window_request_target_layout_ref(request),
            .storage_spans = std::move(*storage_spans_or),
            .source_index_digest = request.source_index_digest,
            .prepared_realization = request.prepared_realization,
        });
  }
  auto key = source_window_routed_program_cache_key(
      artifact_id,
      plan,
      absl::MakeConstSpan(participants),
      configured_chunk_bytes,
      max_collective_chunk_bytes,
      max_stripe_bytes);
  if (!key.has_value()) {
    return absl::FailedPreconditionError("source-window routed program cache key is not eligible");
  }
  return *key;
}

SourceWindowRoutedProgramCachePrepareResult prepare_source_window_routed_program_cache(
    std::string_view artifact_id,
    const SourceWindowCollectivePlan& plan,
    absl::Span<const SourceWindowCollectiveMappedTargetLoadRequest> requests,
    size_t configured_chunk_bytes,
    size_t max_collective_chunk_bytes,
    size_t max_stripe_bytes,
    uint32_t configured_build_threads) {
  SourceWindowRoutedProgramCachePrepareResult result;
  result.plan_hash = plan.plan_hash;
  if (requests.empty()) {
    result.status = absl::InvalidArgumentError("source-window routed program prepare requires requests");
    result.skip_reason = "requests_empty";
    return result;
  }
  if (configured_chunk_bytes == 0 || max_collective_chunk_bytes == 0 || max_stripe_bytes == 0) {
    result.status = absl::InvalidArgumentError("source-window routed program prepare requires non-zero chunk sizing");
    result.skip_reason = "invalid_chunk_sizing";
    return result;
  }
  const bool has_consumer_routed_windows =
      std::any_of(plan.windows.begin(), plan.windows.end(), [](const SourceWindowCollectiveWindow& window) {
        return window.distribution_mode ==
            runtime::ingestion::strategy::SourceWindowCollectiveDistributionMode::kConsumerRouted;
      });
  if (!has_consumer_routed_windows) {
    result.status = absl::OkStatus();
    result.skip_reason = "no_consumer_routed_windows";
    return result;
  }

  std::vector<SourceWindowMappedParticipant> participants;
  participants.reserve(requests.size());
  for (const auto& request : requests) {
    auto storage_spans_or = build_target_storage_spans(source_window_request_target_layout(request));
    if (!storage_spans_or.ok()) {
      result.status = storage_spans_or.status();
      result.skip_reason = "invalid_target_storage_layout";
      return result;
    }
    participants.push_back(
        SourceWindowMappedParticipant{
            .artifact_id = request.artifact_id,
            .rank = static_cast<int>(request.group.rank),
            .device_id = request.device_id,
            .disk_context = request.disk_context,
            .target_layout = source_window_request_target_layout_ref(request),
            .storage_spans = std::move(*storage_spans_or),
            .source_index_digest = request.source_index_digest,
            .prepared_realization = request.prepared_realization,
        });
  }

  auto key = source_window_routed_program_cache_key(
      artifact_id,
      plan,
      absl::MakeConstSpan(participants),
      configured_chunk_bytes,
      max_collective_chunk_bytes,
      max_stripe_bytes);
  if (!key.has_value()) {
    result.status = absl::FailedPreconditionError("source-window routed program cache key is not eligible");
    result.skip_reason = "cache_key_ineligible";
    return result;
  }

  std::vector<SourceWindowRuntimeChunk> runtime_chunks;
  uint64_t unfiltered_consumer_span_refs = 0;
  uint64_t prefiltered_consumer_span_refs = 0;
  auto runtime_chunks_or = build_source_window_runtime_chunks(
      plan,
      requests.size(),
      configured_chunk_bytes,
      max_collective_chunk_bytes,
      &unfiltered_consumer_span_refs,
      &prefiltered_consumer_span_refs);
  if (!runtime_chunks_or.ok()) {
    result.status = runtime_chunks_or.status();
    result.skip_reason = "runtime_chunk_build_failed";
    return result;
  }
  runtime_chunks = std::move(*runtime_chunks_or);
  result.runtime_chunk_count = runtime_chunks.size();

  std::vector<SourceWindowRoutedChunkProgram> cached_programs;
  auto acquire_result = acquire_source_window_routed_program_cache(*key, runtime_chunks.size());
  cached_programs = std::move(acquire_result.programs);
  if (acquire_result.cache_hit) {
    result.prepared = true;
    result.cache_hit = true;
    for (const auto& program : cached_programs) {
      if (program.compiled) {
        result.compiled_chunk_count += 1;
      }
    }
    return result;
  }
  bool cache_build_completed = false;
  auto cache_build_cleanup = absl::Cleanup([&]() {
    if (acquire_result.reserved_build && !cache_build_completed) {
      abandon_source_window_routed_program_cache_build(*key);
    }
  });

  const std::vector<ParsedMappedParticipant> mapped_participants =
      source_window_parsed_mapped_participants(participants);
  double build_sec = 0.0;
  result.program_build_threads =
      compiled_routed_program_build_thread_count(runtime_chunks.size(), configured_build_threads);
  auto programs_or = build_source_window_routed_programs(
      absl::MakeConstSpan(runtime_chunks),
      absl::MakeConstSpan(mapped_participants),
      requests.size(),
      max_stripe_bytes,
      configured_build_threads,
      &build_sec);
  if (!programs_or.ok()) {
    result.status = programs_or.status();
    result.skip_reason = "program_build_failed";
    return result;
  }
  result.program_build_sec = build_sec;
  std::vector<SourceWindowRoutedChunkProgram> programs = std::move(*programs_or);
  for (const auto& program : programs) {
    if (program.compiled) {
      result.compiled_chunk_count += 1;
    }
  }
  if (acquire_result.reserved_build) {
    complete_source_window_routed_program_cache_build(*key, std::move(programs));
    cache_build_completed = true;
  } else {
    store_source_window_routed_program_cache(*key, std::move(programs));
  }
  result.prepared = true;
  result.status = absl::OkStatus();
  return result;
}

struct SourceWindowCollectivePlanPrepareState {
  SourceWindowRoutedProgramCachePrepareResult result;
  std::vector<SourceWindowCollectiveMappedTargetLoadRequest> requests;
  SourceWindowCollectivePlan plan;
  size_t configured_chunk_bytes{0};
  size_t max_collective_chunk_bytes{0};
  size_t max_stripe_bytes{0};
};

SourceWindowCollectivePlanPrepareState prepare_source_window_collective_plan_cache_state(
    absl::Span<const SourceWindowCollectiveMappedTargetLoadRequest> requests,
    const CollectiveMappedTargetLoadOptions& options) {
  SourceWindowCollectivePlanPrepareState state;
  auto& result = state.result;
  if (requests.empty()) {
    result.status = absl::InvalidArgumentError("source-window collective routed program prepare requires requests");
    result.skip_reason = "requests_empty";
    return state;
  }

  const SourceWindowCollectiveConfig config = source_window_collective_config_from_strategy(options.strategy_config);
  if (!config.enabled) {
    result.status = absl::OkStatus();
    result.skip_reason = "strategy_disabled";
    return state;
  }
  if (config.selection_mode == runtime::ingestion::strategy::SourceWindowCollectiveSelectionMode::kDryRun) {
    result.status = absl::OkStatus();
    result.skip_reason = "source_window_dry_run";
    return state;
  }
  const auto& first = requests.front();
  if (first.group.world_size <= 1) {
    result.status = absl::OkStatus();
    result.skip_reason = "collective_group_missing";
    return state;
  }
  if (requests.size() != first.group.world_size) {
    result.status = absl::FailedPreconditionError("source-window collective prepare member count mismatch");
    result.skip_reason = "member_count_mismatch";
    return state;
  }
  if (options.chunk_bytes == 0) {
    result.status = absl::InvalidArgumentError("source-window collective routed program prepare requires chunk bytes");
    result.skip_reason = "invalid_chunk_sizing";
    return state;
  }

  std::vector<std::optional<SourceWindowMappedParticipant>> participant_slots(first.group.world_size);
  for (const auto& request : requests) {
    if (request.group.group_id != first.group.group_id || request.group.world_size != first.group.world_size) {
      result.status = absl::FailedPreconditionError("source-window collective prepare group mismatch");
      result.skip_reason = "group_mismatch";
      return state;
    }
    if (request.group.rank >= first.group.world_size) {
      result.status = absl::InvalidArgumentError("source-window collective prepare rank out of range");
      result.skip_reason = "collective_rank_out_of_range";
      return state;
    }
    if (participant_slots[request.group.rank].has_value()) {
      result.status = absl::InvalidArgumentError("source-window collective prepare duplicate rank");
      result.skip_reason = "duplicate_member_rank";
      return state;
    }
    auto work_plan_ref = source_window_request_work_plan_ref(request);
    auto target_layout_ref = source_window_request_target_layout_ref(request);
    if (work_plan_ref == nullptr || target_layout_ref == nullptr || request.artifact_id.empty() ||
        request.device_id < 0 || request.disk_context == nullptr || target_layout_ref->storages.empty() ||
        work_plan_ref->items.empty()) {
      result.status = absl::OkStatus();
      result.skip_reason = "request_incomplete";
      return state;
    }
    if (!request.candidate_summary.candidate) {
      result.status = absl::OkStatus();
      result.skip_reason = request.candidate_summary.pre_admission_reason.empty()
          ? "candidate_missing"
          : request.candidate_summary.pre_admission_reason;
      return state;
    }
    if (!request.disk_context->is_safetensors()) {
      result.status = absl::OkStatus();
      result.skip_reason = "non_safetensors_source";
      return state;
    }
    auto storage_spans_or = build_target_storage_spans(*target_layout_ref);
    if (!storage_spans_or.ok()) {
      result.status = storage_spans_or.status();
      result.skip_reason = "invalid_target_storage_layout";
      return state;
    }
    participant_slots[request.group.rank] = SourceWindowMappedParticipant{
        .artifact_id = request.artifact_id,
        .rank = static_cast<int>(request.group.rank),
        .device_id = request.device_id,
        .disk_context = request.disk_context,
        .work_plan = std::move(work_plan_ref),
        .target_layout = std::move(target_layout_ref),
        .storage_spans = std::move(*storage_spans_or),
        .candidate_summary = request.candidate_summary,
        .source_index_digest = request.source_index_digest,
        .prepared_realization = request.prepared_realization,
    };
  }

  std::vector<SourceWindowMappedParticipant> participants;
  participants.reserve(participant_slots.size());
  for (auto& participant : participant_slots) {
    if (!participant.has_value()) {
      result.status = absl::FailedPreconditionError("source-window collective prepare missing rank");
      result.skip_reason = "missing_member_rank";
      return state;
    }
    participants.push_back(std::move(*participant));
  }

  auto input_or = build_source_window_collective_group_input(first.group, participants, options);
  if (!input_or.ok()) {
    result.status = input_or.status();
    result.skip_reason = absl::StrCat("source_window_group_rejected:", input_or.status().message());
    return state;
  }

  SourceWindowCollectivePlan plan;
  bool plan_cache_hit = false;
  if (options.enable_source_window_plan_cache) {
    std::string plan_cache_key;
    if (auto prepared_key = source_window_collective_prepared_plan_cache_key(first.artifact_id, *input_or);
        prepared_key.has_value()) {
      plan_cache_key = std::move(*prepared_key);
    } else {
      plan_cache_key = source_window_collective_plan_cache_key(first.artifact_id, *input_or);
    }
    plan_cache_hit = lookup_source_window_collective_plan_cache(plan_cache_key, first.group, &plan);
    if (!plan_cache_hit) {
      auto plan_or = build_source_window_collective_plan(*input_or);
      if (!plan_or.ok()) {
        abandon_source_window_collective_plan_cache_build(plan_cache_key);
        result.status = plan_or.status();
        result.skip_reason = absl::StrCat("source_window_group_rejected:", plan_or.status().message());
        return state;
      }
      plan = std::move(*plan_or);
      store_source_window_collective_plan_cache(plan_cache_key, plan);
    }
  } else {
    auto plan_or = build_source_window_collective_plan(*input_or);
    if (!plan_or.ok()) {
      result.status = plan_or.status();
      result.skip_reason = absl::StrCat("source_window_group_rejected:", plan_or.status().message());
      return state;
    }
    plan = std::move(*plan_or);
  }
  result.plan_hash = plan.plan_hash;
  result.cache_hit = plan_cache_hit;

  LOG(INFO) << "source_window_collective_routed_program_prepare"
            << " artifact_id=" << first.artifact_id << " group_id=" << first.group.group_id
            << " tp_size=" << first.group.world_size << " admitted=" << (plan.summary.group_final_admitted ? 1 : 0)
            << " reject_reason=" << plan.summary.group_reject_reason << " distribution_mode="
            << runtime::ingestion::strategy::source_window_collective_distribution_mode_name(plan.distribution_mode)
            << " plan_hash=" << plan.plan_hash
            << " plan_cache_enabled=" << (options.enable_source_window_plan_cache ? 1 : 0)
            << " plan_cache_hit=" << (plan_cache_hit ? 1 : 0);
  if (!plan.summary.group_final_admitted) {
    result.status = absl::OkStatus();
    result.skip_reason = absl::StrCat("source_window_group_rejected:", plan.summary.group_reject_reason);
    return state;
  }

  const size_t configured_chunk_bytes = static_cast<size_t>(options.chunk_bytes);
  auto chunk_sizing_or =
      source_window_runtime_chunk_sizing(configured_chunk_bytes, requests.size(), plan.distribution_mode);
  if (!chunk_sizing_or.ok()) {
    result.status = chunk_sizing_or.status();
    result.skip_reason = "invalid_chunk_sizing";
    return state;
  }
  const auto& chunk_sizing = *chunk_sizing_or;
  state.requests.assign(requests.begin(), requests.end());
  state.plan = std::move(plan);
  state.configured_chunk_bytes = configured_chunk_bytes;
  state.max_collective_chunk_bytes = chunk_sizing.max_collective_chunk_bytes;
  state.max_stripe_bytes = chunk_sizing.max_stripe_bytes;
  result.prepared = true;
  result.status = absl::OkStatus();
  return state;
}

SourceWindowRoutedProgramCachePrepareResult prepare_source_window_collective_plan_cache(
    absl::Span<const SourceWindowCollectiveMappedTargetLoadRequest> requests,
    const CollectiveMappedTargetLoadOptions& options) {
  auto state = prepare_source_window_collective_plan_cache_state(requests, options);
  return state.result;
}

SourceWindowRoutedProgramCachePrepareResult prepare_source_window_collective_routed_program_cache(
    absl::Span<const SourceWindowCollectiveMappedTargetLoadRequest> requests,
    const CollectiveMappedTargetLoadOptions& options) {
  auto state = prepare_source_window_collective_plan_cache_state(requests, options);
  if (!state.result.status.ok() || !state.result.prepared) {
    return state.result;
  }
  if (!options.strategy_config.enable_source_window_compiled_routed_program) {
    state.result.prepared = false;
    state.result.skip_reason = "compiled_routed_program_disabled";
    return state.result;
  }
  return prepare_source_window_routed_program_cache(
      state.requests.front().artifact_id,
      state.plan,
      absl::MakeConstSpan(state.requests),
      state.configured_chunk_bytes,
      state.max_collective_chunk_bytes,
      state.max_stripe_bytes,
      options.strategy_config.source_window_compiled_program_build_threads);
}

SourceWindowRoutedProgramCachePrepareResult prepare_source_window_routed_program_cache_for_testing(
    std::string_view artifact_id,
    const SourceWindowCollectivePlan& plan,
    absl::Span<const SourceWindowCollectiveMappedTargetLoadRequest> requests,
    size_t configured_chunk_bytes,
    size_t max_collective_chunk_bytes,
    size_t max_stripe_bytes,
    uint32_t configured_build_threads) {
  return prepare_source_window_routed_program_cache(
      artifact_id,
      plan,
      requests,
      configured_chunk_bytes,
      max_collective_chunk_bytes,
      max_stripe_bytes,
      configured_build_threads);
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

SourceWindowCollectiveMappedTargetLoadResult try_source_window_collective_mapped_target_load(
    const SourceWindowCollectiveMappedTargetLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options) {
  (void)pinned_pool;
  (void)pinned_timeout;
  const SourceWindowCollectiveConfig config = source_window_collective_config_from_strategy(options.strategy_config);
  if (!config.enabled) {
    return {.handled = false, .status = absl::OkStatus(), .skip_reason = "strategy_disabled"};
  }
  const auto& request_work_plan = source_window_request_work_plan(request);
  const auto& request_target_layout = source_window_request_target_layout(request);
  if (request.group.world_size <= 1 || request.device_id < 0 || request.disk_context == nullptr ||
      request_target_layout.storages.empty() || request_work_plan.items.empty() || request.artifact_id.empty()) {
    LOG(INFO) << "source_window_collective_mapped_target skipped group_id=" << request.group.group_id
              << " reason=request_incomplete"
              << " world_size=" << request.group.world_size << " device_id=" << request.device_id
              << " disk_context=" << (request.disk_context != nullptr)
              << " storages=" << request_target_layout.storages.size()
              << " work_items=" << request_work_plan.items.size() << " artifact_id=" << (!request.artifact_id.empty());
    return source_window_unhandled_or_strict_failure(config, "request_incomplete");
  }
  if (!request.candidate_summary.candidate) {
    const std::string reason = request.candidate_summary.pre_admission_reason.empty()
        ? "candidate_missing"
        : request.candidate_summary.pre_admission_reason;
    return source_window_unhandled_or_strict_failure(config, reason);
  }
  if (!request.disk_context->is_safetensors()) {
    LOG(INFO) << "source_window_collective_mapped_target skipped group_id=" << request.group.group_id
              << " reason=non_safetensors_source";
    return source_window_unhandled_or_strict_failure(config, "non_safetensors_source");
  }
  return wait_for_source_window_mapped_group_and_maybe_execute(request, pinned_pool, pinned_timeout, options);
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
  const auto memory_before = capture_source_window_device_memory(device_ids);
  log_source_window_collective_memory(
      "clique_prewarm_before",
      /*artifact_id=*/"",
      /*group_id=*/"prewarm",
      memory_before,
      &memory_before);
  bool cache_hit = false;
  auto clique_or = get_or_create_cached_clique(device_ids, &cache_hit);
  if (!clique_or.ok()) {
    return clique_or.status();
  }
  log_source_window_collective_memory(
      "clique_prewarm_after",
      /*artifact_id=*/"",
      /*group_id=*/"prewarm",
      capture_source_window_device_memory(device_ids),
      &memory_before);
  LOG(INFO) << "collective_disk_load clique prewarm complete device_ids=" << clique_cache_key(device_ids)
            << " cache_hit=" << (cache_hit ? 1 : 0);
  return absl::OkStatus();
}

} // namespace tensorcast::store::replica
