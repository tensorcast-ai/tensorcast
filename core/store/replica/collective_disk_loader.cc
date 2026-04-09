// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/replica/collective_disk_loader.h"

#include <errno.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <nccl.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
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

constexpr std::chrono::milliseconds kGroupAssembleTimeout{2000};
constexpr size_t kWholeSourceFreeMemoryReserveBytes = 512ULL * 1024ULL * 1024ULL;
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
  return {.handled = false, .status = absl::OkStatus()};
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
  enum class Kind : uint8_t { kFull = 0, kDim0 = 1, kDim1 = 2 };

  uint64_t dst_offset{0};
  uint64_t dst_size_bytes{0};
  Kind kind{Kind::kFull};
  int64_t start{0};
  uint64_t length{0};
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
  double owner_skew_ratio{1.0};
  bool used_segment_split{false};
};

struct SegmentCopy {
  uint64_t src_offset{0};
  uint64_t dst_offset{0};
  size_t bytes{0};
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
  }

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
      const uint8_t* mapped = segment.file->mapped_base();
      if (mapped != nullptr) {
        std::memcpy(out + total, mapped + segment.data_start + within, step);
        total += step;
        cursor += step;
        continue;
      }
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

std::optional<TensorAxisRange> single_axis_range(const TensorCoordinateSpec& spec) {
  if (spec.selects_scalar || spec.axes.size() != 1) {
    return std::nullopt;
  }
  return spec.axes.front();
}

bool mapped_tensor_job_sources_match(const RepresentationWorkItem& lhs, const RepresentationWorkItem& rhs) {
  return lhs.partition_kind == rhs.partition_kind && lhs.dst_spec == rhs.dst_spec && lhs.sources == rhs.sources;
}

absl::StatusOr<MappedTensorJobBuildResult> build_mapped_tensor_jobs(
    const std::vector<ParsedMappedParticipant>& participants) {
  MappedTensorJobBuildResult result;
  if (participants.empty()) {
    return result;
  }
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
      std::vector<void*> dst_ptrs(participants.size(), nullptr);
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

  for (auto& ranges : result.handled_dst_ranges_by_rank) {
    merge_byte_ranges(&ranges);
  }
  return result;
}

absl::StatusOr<std::vector<MappedSegmentRef>> build_mapped_segment_refs(
    const std::vector<ParsedMappedParticipant>& participants) {
  std::vector<MappedSegmentRef> segments;
  for (const auto& participant : participants) {
    if (participant.collective_lane_map.num_sources != 1) {
      return absl::InvalidArgumentError("mapped collective load requires mapping.num_sources == 1");
    }
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
      segments.push_back(
          MappedSegmentRef{
              .rank = participant.rank,
              .src_offset = segment.src_offset,
              .dst_offset = segment.dst_offset,
              .length = segment.length,
          });
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

bool disable_collective_whole_source_preload() {
  return false;
}

absl::StatusOr<std::optional<std::unique_ptr<common::memory::GpuDeviceMemory>>> maybe_load_whole_source_to_root_buffer(
    const std::vector<ParsedParticipant>& participants,
    absl::Span<const loader::SharedSafetensorsSegment> segments,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    size_t chunk_bytes) {
  if (participants.empty()) {
    return absl::InvalidArgumentError("participants are empty");
  }
  if (segments.empty()) {
    return absl::InvalidArgumentError("safetensors segments are empty");
  }
  if (chunk_bytes == 0) {
    return absl::InvalidArgumentError("chunk_bytes must be > 0");
  }
  if (pinned_pool == nullptr) {
    return absl::InvalidArgumentError("pinned_pool is null");
  }

  const int root_rank = 0;
  const int root_device_id = participants[static_cast<size_t>(root_rank)].device_id;
  const uint64_t source_bytes = total_source_bytes(segments);
  if (source_bytes == 0) {
    return absl::InvalidArgumentError("collective source bytes are zero");
  }

  if (disable_collective_whole_source_preload()) {
    LOG(INFO) << "collective_whole_source skipped"
              << " prototype=1"
              << " cleanup_target=0109"
              << " reason=disabled"
              << " source_bytes=" << source_bytes;
    return std::optional<std::unique_ptr<common::memory::GpuDeviceMemory>>{};
  }

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  TC_RETURN_IF_ERROR(tensorcast::cuda::get_memory_info(&free_bytes, &total_bytes, root_device_id));
  if (source_bytes + kWholeSourceFreeMemoryReserveBytes > free_bytes) {
    LOG(INFO) << "collective_whole_source skipped"
              << " prototype=1"
              << " cleanup_target=0109"
              << " reason=insufficient_free_memory"
              << " source_bytes=" << source_bytes << " free_bytes=" << free_bytes << " total_bytes=" << total_bytes
              << " reserve_bytes=" << kWholeSourceFreeMemoryReserveBytes;
    return std::optional<std::unique_ptr<common::memory::GpuDeviceMemory>>{};
  }

  auto root_source = std::make_unique<common::memory::GpuDeviceMemory>();
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(root_device_id));
  TC_RETURN_IF_ERROR(root_source->allocate(static_cast<size_t>(source_bytes), root_device_id));

  const size_t buffer_chunk_bytes = pinned_pool->slice_bytes();
  if (buffer_chunk_bytes == 0) {
    return absl::InvalidArgumentError("pinned_pool slice_bytes must be > 0");
  }
  const size_t capacity_slices = pinned_pool->capacity_slices();
  const int concurrency = std::max<int>(1, std::min<int>(8, static_cast<int>(capacity_slices / 2)));
  const size_t num_chunks =
      std::max<size_t>(2, std::min<size_t>(capacity_slices, static_cast<size_t>(std::max(2, concurrency * 2))));
  auto streaming_buffer =
      std::make_shared<common::memory::StreamingPinnedBuffer>(num_chunks, buffer_chunk_bytes, pinned_pool);
  TC_RETURN_IF_ERROR(streaming_buffer->initialize(
      pinned_timeout,
      absl::StrCat("collective_whole_source artifact_id=", participants.front().replica_key.artifact_id)));
  loader::StreamingBufferAdapter buffer_pool(streaming_buffer);
  loader::GpuMemorySink sink(
      loader::GpuMemorySink::Options{
          .gpu_base_ptr = gsl::not_null<void*>{root_source->get()},
          .total_size = source_bytes,
          .chunk_size = buffer_chunk_bytes,
          .device_id = root_device_id,
          .allocation = nullptr,
          .gpu_sched_enabled = false,
          .gpu_sched_limit_bytes = loader::DEFAULT_GPU_SCHED_LIMIT_BYTES,
          .gpu_sched_limit_copies = loader::DEFAULT_GPU_SCHED_LIMIT_COPIES,
      });
  PreadMultiSafetensorsSource source(std::vector<loader::SharedSafetensorsSegment>(segments.begin(), segments.end()));
  const auto ranges = split_even_ranges(source_bytes, concurrency);
  TC_RETURN_IF_ERROR(
      loader::pump_ranges(
          source, sink, buffer_pool, ranges, concurrency, whole_source_load_runtime().blocking_executor()));
  TC_RETURN_IF_ERROR(sink.close());
  TC_RETURN_IF_ERROR(streaming_buffer->release());
  return std::optional<std::unique_ptr<common::memory::GpuDeviceMemory>>(std::move(root_source));
}

absl::Status execute_replicated_tensor_from_root_buffer(
    const TensorJob& job,
    const std::vector<ParsedParticipant>& participants,
    NcclClique& clique,
    const void* root_source_ptr,
    size_t chunk_bytes,
    int root_rank) {
  const auto source_base_offset_or = source_base_offset_bytes(job.source);
  if (!source_base_offset_or.ok()) {
    return source_base_offset_or.status();
  }
  const uint64_t source_base_offset = *source_base_offset_or;
  uint64_t copied = 0;
  while (copied < job.source.size_bytes) {
    const size_t step =
        static_cast<size_t>(std::min<uint64_t>(job.source.size_bytes - copied, static_cast<uint64_t>(chunk_bytes)));
    const auto* send_ptr = static_cast<const uint8_t*>(root_source_ptr) + source_base_offset + copied;
    TC_RETURN_IF_ERROR(clique.group_start());
    for (size_t idx = 0; idx < participants.size(); ++idx) {
      auto* dst_ptr = static_cast<uint8_t*>(participants[idx].gpu_ptr) + job.slices[idx].dst_offset + copied;
      const void* rank_send_ptr = (static_cast<int>(idx) == root_rank) ? send_ptr : dst_ptr;
      TC_RETURN_IF_ERROR(clique.broadcast_u8(static_cast<int>(idx), rank_send_ptr, dst_ptr, step, root_rank));
    }
    TC_RETURN_IF_ERROR(clique.group_end());
    copied += static_cast<uint64_t>(step);
  }
  TC_RETURN_IF_ERROR(clique.synchronize_all());
  return absl::OkStatus();
}

absl::Status execute_dim0_tensor_from_root_buffer(
    const TensorJob& job,
    const std::vector<ParsedParticipant>& participants,
    NcclClique& clique,
    const void* root_source_ptr,
    size_t chunk_bytes,
    int root_rank) {
  const auto source_base_offset_or = source_base_offset_bytes(job.source);
  if (!source_base_offset_or.ok()) {
    return source_base_offset_or.status();
  }
  const uint64_t source_base_offset = *source_base_offset_or;
  uint64_t per_row_bytes = job.source.elem_size;
  for (size_t dim = 1; dim < job.source.shape.size(); ++dim) {
    per_row_bytes *= static_cast<uint64_t>(job.source.shape[dim]);
  }
  const uint64_t full_bytes = job.source.size_bytes;
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
  uint64_t copied = 0;
  while (copied < full_bytes) {
    const size_t step =
        static_cast<size_t>(std::min<uint64_t>(full_bytes - copied, static_cast<uint64_t>(chunk_bytes)));
    TC_RETURN_IF_ERROR(clique.group_start());
    for (size_t idx = 0; idx < participants.size(); ++idx) {
      const auto& slice = job.slices[idx];
      const uint64_t slice_begin = static_cast<uint64_t>(slice.start) * per_row_bytes;
      const uint64_t slice_end = slice_begin + slice.dst_size_bytes;
      const uint64_t chunk_begin = copied;
      const uint64_t chunk_end = copied + step;
      const uint64_t overlap_begin = std::max<uint64_t>(slice_begin, chunk_begin);
      const uint64_t overlap_end = std::min<uint64_t>(slice_end, chunk_end);
      if (overlap_end <= overlap_begin) {
        continue;
      }
      const size_t overlap_bytes = static_cast<size_t>(overlap_end - overlap_begin);
      const uint64_t dst_off = slice.dst_offset + (overlap_begin - slice_begin);
      auto* dst_ptr = static_cast<uint8_t*>(participants[idx].gpu_ptr) + dst_off;
      const auto* src_ptr = static_cast<const uint8_t*>(root_source_ptr) + source_base_offset + overlap_begin;
      if (static_cast<int>(idx) == root_rank) {
        TC_RETURN_IF_ERROR(
            tensorcast::cuda::memcpy_async(
                dst_ptr, src_ptr, overlap_bytes, cudaMemcpyDeviceToDevice, clique.stream(root_rank)));
      } else {
        TC_RETURN_IF_ERROR(clique.send_u8(root_rank, src_ptr, overlap_bytes, static_cast<int>(idx)));
        TC_RETURN_IF_ERROR(clique.recv_u8(static_cast<int>(idx), dst_ptr, overlap_bytes, root_rank));
      }
    }
    TC_RETURN_IF_ERROR(clique.group_end());
    copied += static_cast<uint64_t>(step);
  }
  TC_RETURN_IF_ERROR(clique.synchronize_all());
  return absl::OkStatus();
}

absl::Status execute_dim1_tensor_from_root_buffer(
    const TensorJob& job,
    const std::vector<ParsedParticipant>& participants,
    NcclClique& clique,
    const void* root_source_ptr,
    size_t chunk_bytes,
    int root_rank) {
  const auto source_base_offset_or = source_base_offset_bytes(job.source);
  if (!source_base_offset_or.ok()) {
    return source_base_offset_or.status();
  }
  const uint64_t source_base_offset = *source_base_offset_or;
  if (job.source.shape.size() != 2) {
    return absl::UnimplementedError("dim1 collective tensor must be 2D");
  }
  const uint64_t rows = static_cast<uint64_t>(job.source.shape[0]);
  const uint64_t cols = static_cast<uint64_t>(job.source.shape[1]);
  const uint64_t row_bytes = cols * job.source.elem_size;
  if (row_bytes == 0) {
    return absl::InvalidArgumentError("dim1 collective tensor has zero row_bytes");
  }

  const uint64_t rows_per_chunk = std::max<uint64_t>(1, static_cast<uint64_t>(chunk_bytes) / row_bytes);
  std::vector<std::unique_ptr<common::memory::GpuDeviceMemory>> pack_buffers(participants.size());
  for (size_t idx = 0; idx < participants.size(); ++idx) {
    if (static_cast<int>(idx) == root_rank) {
      continue;
    }
    const uint64_t col_bytes = job.slices[idx].dst_size_bytes / std::max<uint64_t>(1, rows);
    if (col_bytes == 0) {
      continue;
    }
    pack_buffers[idx] = std::make_unique<common::memory::GpuDeviceMemory>();
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
    TC_RETURN_IF_ERROR(
        pack_buffers[idx]->allocate(
            static_cast<size_t>(rows_per_chunk * col_bytes), participants[static_cast<size_t>(root_rank)].device_id));
  }

  cudaStream_t pack_stream = nullptr;
  cudaEvent_t pack_ready_event = nullptr;
  cudaEvent_t pack_consumed_event = nullptr;
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&pack_stream, cudaStreamNonBlocking));
  TC_RETURN_IF_ERROR(tensorcast::cuda::event_create_with_flags(&pack_ready_event, cudaEventDisableTiming));
  TC_RETURN_IF_ERROR(tensorcast::cuda::event_create_with_flags(&pack_consumed_event, cudaEventDisableTiming));

  for (uint64_t row = 0; row < rows; row += rows_per_chunk) {
    const uint64_t chunk_rows = std::min<uint64_t>(rows_per_chunk, rows - row);
    if (row > 0) {
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_wait_event(pack_stream, pack_consumed_event));
    }
    const auto* row_base_ptr = static_cast<const uint8_t*>(root_source_ptr) + source_base_offset + row * row_bytes;

    for (size_t idx = 0; idx < participants.size(); ++idx) {
      const auto& slice = job.slices[idx];
      const uint64_t col_bytes = slice.dst_size_bytes / std::max<uint64_t>(1, rows);
      const uint64_t src_col_bytes = static_cast<uint64_t>(slice.start) * job.source.elem_size;
      auto* dst_ptr = static_cast<uint8_t*>(participants[idx].gpu_ptr) + slice.dst_offset + row * col_bytes;
      const auto* src_ptr = row_base_ptr + src_col_bytes;
      if (static_cast<int>(idx) == root_rank) {
        SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
            dst_ptr,
            static_cast<size_t>(col_bytes),
            src_ptr,
            static_cast<size_t>(row_bytes),
            static_cast<size_t>(col_bytes),
            static_cast<size_t>(chunk_rows),
            cudaMemcpyDeviceToDevice,
            clique.stream(root_rank)));
        continue;
      }
      if (pack_buffers[idx] == nullptr) {
        continue;
      }
      auto* pack_ptr = static_cast<uint8_t*>(pack_buffers[idx]->get());
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

    TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(pack_ready_event, pack_stream));
    TC_RETURN_IF_ERROR(clique.wait_stream_on_event(root_rank, pack_ready_event));
    TC_RETURN_IF_ERROR(clique.group_start());
    for (size_t idx = 0; idx < participants.size(); ++idx) {
      if (static_cast<int>(idx) == root_rank || pack_buffers[idx] == nullptr) {
        continue;
      }
      const auto& slice = job.slices[idx];
      const uint64_t col_bytes = slice.dst_size_bytes / std::max<uint64_t>(1, rows);
      const size_t send_bytes = static_cast<size_t>(chunk_rows * col_bytes);
      auto* dst_ptr = static_cast<uint8_t*>(participants[idx].gpu_ptr) + slice.dst_offset + row * col_bytes;
      TC_RETURN_IF_ERROR(clique.send_u8(root_rank, pack_buffers[idx]->get(), send_bytes, static_cast<int>(idx)));
      TC_RETURN_IF_ERROR(clique.recv_u8(static_cast<int>(idx), dst_ptr, send_bytes, root_rank));
    }
    TC_RETURN_IF_ERROR(clique.group_end());
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(pack_consumed_event, clique.stream(root_rank)));
  }

  TC_RETURN_IF_ERROR(clique.synchronize_all());
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
  TC_RETURN_IF_ERROR(tensorcast::cuda::event_destroy(pack_consumed_event));
  TC_RETURN_IF_ERROR(tensorcast::cuda::event_destroy(pack_ready_event));
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_destroy(pack_stream));
  return absl::OkStatus();
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
        const uint64_t src_col_bytes = static_cast<uint64_t>(slice.start) * job.source.elem_size;
        auto* dst_ptr = static_cast<uint8_t*>(participants[idx].gpu_ptr) + slice.dst_offset + row * col_bytes;
        const auto* src_ptr = static_cast<const uint8_t*>(root_stage_ptr) + src_col_bytes;
        if (static_cast<int>(idx) == root_rank) {
          SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
              dst_ptr,
              static_cast<size_t>(col_bytes),
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
        const size_t send_bytes = static_cast<size_t>(chunk_rows * col_bytes);
        auto* dst_ptr = static_cast<uint8_t*>(participants[idx].gpu_ptr) + slice.dst_offset + row * col_bytes;
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

absl::StatusOr<std::vector<SegmentCopy>> build_local_direct_segments(
    const std::vector<TensorJob>& jobs,
    uint64_t* direct_source_bytes,
    size_t* replicated_jobs,
    size_t* dim0_jobs) {
  if (direct_source_bytes != nullptr) {
    *direct_source_bytes = 0;
  }
  if (replicated_jobs != nullptr) {
    *replicated_jobs = 0;
  }
  if (dim0_jobs != nullptr) {
    *dim0_jobs = 0;
  }
  std::vector<SegmentCopy> segments;
  segments.reserve(jobs.size());
  for (const auto& job : jobs) {
    const auto& slice = job.slices.front();
    if (slice.dst_size_bytes == 0) {
      continue;
    }
    switch (job.distribution) {
      case TensorJob::Distribution::kReplicated: {
        segments.push_back(
            SegmentCopy{
                .src_offset = job.source.offset,
                .dst_offset = slice.dst_offset,
                .bytes = static_cast<size_t>(slice.dst_size_bytes),
            });
        if (direct_source_bytes != nullptr) {
          *direct_source_bytes += slice.dst_size_bytes;
        }
        if (replicated_jobs != nullptr) {
          *replicated_jobs += 1;
        }
        break;
      }
      case TensorJob::Distribution::kDim0Partitioned: {
        uint64_t per_row_bytes = job.source.elem_size;
        for (size_t dim = 1; dim < job.source.shape.size(); ++dim) {
          per_row_bytes *= static_cast<uint64_t>(job.source.shape[dim]);
        }
        segments.push_back(
            SegmentCopy{
                .src_offset = job.source.offset + static_cast<uint64_t>(slice.start) * per_row_bytes,
                .dst_offset = slice.dst_offset,
                .bytes = static_cast<size_t>(slice.dst_size_bytes),
            });
        if (direct_source_bytes != nullptr) {
          *direct_source_bytes += slice.dst_size_bytes;
        }
        if (dim0_jobs != nullptr) {
          *dim0_jobs += 1;
        }
        break;
      }
      case TensorJob::Distribution::kDim1Partitioned:
        break;
    }
  }
  return merge_adjacent_segments_by_src(std::move(segments));
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
    return {.handled = false, .status = absl::OkStatus()};
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

  loader::MultiSafetensorsSource backing_source(request.disk_context->safetensors_segments());
  uint64_t direct_source_bytes = 0;
  size_t replicated_jobs = 0;
  size_t dim0_jobs = 0;
  auto direct_segments_or = build_local_direct_segments(*jobs_or, &direct_source_bytes, &replicated_jobs, &dim0_jobs);
  if (!direct_segments_or.ok()) {
    if (absl::IsUnimplemented(direct_segments_or.status())) {
      return local_batched_fallback(
          request.replica_key.artifact_id, "unsupported_direct_segments", direct_segments_or.status());
    }
    return {.handled = true, .status = direct_segments_or.status()};
  }

  const auto total_start = std::chrono::steady_clock::now();
  double direct_sec = 0.0;
  if (!direct_segments_or->empty()) {
    std::vector<RemappedSource::Segment> remap;
    remap.reserve(direct_segments_or->size());
    for (const auto& segment : *direct_segments_or) {
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
    auto ranges_or = build_pump_ranges_for_copy(*direct_segments_or, /*io_threads=*/4);
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
    direct_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - direct_start).count();
  }

  std::vector<TensorJob> dim1_jobs;
  dim1_jobs.reserve(jobs_or->size());
  for (const auto& job : *jobs_or) {
    if (job.distribution == TensorJob::Distribution::kDim1Partitioned) {
      dim1_jobs.push_back(job);
    }
  }
  const auto dim1_start = std::chrono::steady_clock::now();
  {
    const absl::Status dim1_status = execute_local_dim1_jobs(
        dim1_jobs, backing_source, pinned_pool, pinned_timeout, request.gpu_ptr, request.device_id);
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
            << " direct_segments=" << direct_segments_or->size() << " replicated_jobs=" << replicated_jobs
            << " dim0_jobs=" << dim0_jobs << " dim1_jobs=" << dim1_jobs.size()
            << " direct_source_bytes=" << direct_source_bytes << " direct_sec=" << direct_sec
            << " dim1_sec=" << dim1_sec << " total=" << total_sec;
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

absl::Status execute_group_collective(
    const std::vector<ParsedParticipant>& participants,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const StrategyConfig& strategy_config) {
  const auto total_start = std::chrono::steady_clock::now();
  if (participants.empty()) {
    return absl::InvalidArgumentError("collective disk load participants are empty");
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
      LOG(INFO) << "collective_owner_file_batched plan"
                << " artifact_id=" << participants.front().replica_key.artifact_id
                << " batches=" << owner_collective_plan->batches.size()
                << " unique_source_bytes=" << owner_collective_plan->unique_source_bytes
                << " peer_transfer_bytes=" << owner_collective_plan->peer_transfer_bytes
                << " owner_skew_ratio=" << owner_collective_plan->owner_skew_ratio
                << " segment_split=" << (owner_collective_plan->used_segment_split ? 1 : 0);
    } else {
      LOG(WARNING) << "collective_owner_file_batched fallback"
                   << " artifact_id=" << participants.front().replica_key.artifact_id
                   << " status=" << owner_plan_or.status();
    }
  }
  const auto whole_source_start = std::chrono::steady_clock::now();
  std::unique_ptr<common::memory::GpuDeviceMemory> root_source;
  bool whole_source_enabled = false;
  if (participants.front().disk_context->safetensors_segments().empty()) {
    return absl::FailedPreconditionError("collective disk load requires non-empty safetensors segments");
  }
  if (!owner_file_enabled) {
    auto root_source_or = maybe_load_whole_source_to_root_buffer(
        participants,
        participants.front().disk_context->safetensors_segments(),
        pinned_pool,
        pinned_timeout,
        chunk_bytes);
    if (!root_source_or.ok()) {
      return root_source_or.status();
    }
    if (root_source_or->has_value()) {
      root_source = std::move(**root_source_or);
      whole_source_enabled = true;
      LOG(INFO) << "collective_whole_source prototype"
                << " artifact_id=" << participants.front().replica_key.artifact_id << " prototype=1"
                << " cleanup_target=0109"
                << " source_bytes=" << root_source->size();
    }
  } else {
    LOG(INFO) << "collective_whole_source skipped"
              << " artifact_id=" << participants.front().replica_key.artifact_id
              << " reason=owner_file_batched_enabled";
  }
  const auto whole_source_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - whole_source_start).count();

  PinnedBorrow host_pool;
  host_pool.pool = pinned_pool;
  double pinned_alloc_sec = 0.0;
  char* host_buffer = nullptr;
  if (!whole_source_enabled) {
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
        if (whole_source_enabled) {
          TC_RETURN_IF_ERROR(execute_replicated_tensor_from_root_buffer(
              job, participants, *clique, root_source->get(), chunk_bytes, root_rank));
        } else {
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
        }
        replicated_jobs += 1;
        replicated_source_bytes += job.source.size_bytes;
        replicated_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - job_start).count();
        break;
      case TensorJob::Distribution::kDim0Partitioned:
        if (whole_source_enabled) {
          TC_RETURN_IF_ERROR(execute_dim0_tensor_from_root_buffer(
              job, participants, *clique, root_source->get(), chunk_bytes, root_rank));
        } else {
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
        }
        dim0_jobs += 1;
        dim0_source_bytes += job.source.size_bytes;
        dim0_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - job_start).count();
        break;
      case TensorJob::Distribution::kDim1Partitioned:
        if (whole_source_enabled) {
          TC_RETURN_IF_ERROR(execute_dim1_tensor_from_root_buffer(
              job, participants, *clique, root_source->get(), chunk_bytes, root_rank));
        } else {
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
        }
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
            << " whole_source=" << (whole_source_enabled ? 1 : 0) << " whole_source_load=" << whole_source_sec << "s"
            << " whole_source_bytes=" << (root_source != nullptr ? root_source->size() : 0)
            << " build_jobs=" << jobs_sec << "s"
            << " replicated_jobs=" << replicated_jobs << " replicated_source_bytes=" << replicated_source_bytes
            << " replicated_exec=" << replicated_sec << "s"
            << " replicated_gib_s=" << replicated_gib_s << " dim0_jobs=" << dim0_jobs
            << " dim0_source_bytes=" << dim0_source_bytes << " dim0_exec=" << dim0_sec << "s"
            << " dim0_gib_s=" << dim0_gib_s << " dim1_jobs=" << dim1_jobs << " dim1_source_bytes=" << dim1_source_bytes
            << " dim1_exec=" << dim1_sec << "s"
            << " dim1_gib_s=" << dim1_gib_s << " chunk_bytes=" << chunk_bytes << " total=" << total_sec << "s";
  return absl::OkStatus();
}

absl::Status execute_group_collective_mapped(
    const std::vector<ParsedMappedParticipant>& participants,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options) {
  const auto total_start = std::chrono::steady_clock::now();
  if (participants.empty()) {
    return absl::InvalidArgumentError("mapped collective load participants are empty");
  }
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

  const size_t chunk_bytes = std::max<size_t>(1, std::min<uint64_t>(options.chunk_bytes, pinned_pool->slice_bytes()));
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

  const auto segments_start = std::chrono::steady_clock::now();
  auto segment_refs_or = build_mapped_segment_refs(participants);
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
    return absl::OkStatus();
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
  return absl::OkStatus();
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
    const absl::Status exec_status =
        execute_group_collective(participants, pinned_pool, pinned_timeout, request.strategy_config);
    LOG(INFO) << "collective_disk_load finished group_id=" << request.group.group_id << " status=" << exec_status;
    {
      absl::MutexLock lock(&state->mu);
      state->status = exec_status;
      state->complete = true;
      state->cv.SignalAll();
    }
    {
      absl::MutexLock lock(&g_group_mu);
      g_groups.erase(request.group.group_id);
    }
    return {.handled = true, .status = exec_status};
  }

  {
    absl::MutexLock lock(&state->mu);
    while (!state->complete) {
      state->cv.Wait(&state->mu);
    }
    return {.handled = true, .status = state->status};
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
    const absl::Status exec_status =
        execute_group_collective_mapped(participants, pinned_pool, pinned_timeout, options);
    LOG(INFO) << "collective_mapped_target finished group_id=" << request.group.group_id << " status=" << exec_status;
    {
      absl::MutexLock lock(&state->mu);
      state->status = exec_status;
      state->complete = true;
      state->cv.SignalAll();
    }
    {
      absl::MutexLock lock(&g_mapped_group_mu);
      g_mapped_groups.erase(request.group.group_id);
    }
    return {.handled = true, .status = exec_status};
  }

  {
    absl::MutexLock lock(&state->mu);
    while (!state->complete) {
      state->cv.Wait(&state->mu);
    }
    return {.handled = true, .status = state->status};
  }
}

} // namespace

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
  if (request.group.world_size <= 1 || request.gpu_ptr == nullptr || request.device_id < 0 ||
      request.disk_context == nullptr || request.representation_work_plan.items.empty() ||
      request.gpu_allocation == nullptr) {
    LOG(INFO) << "collective_disk_load skipped group_id=" << request.group.group_id << " reason=request_incomplete"
              << " world_size=" << request.group.world_size << " gpu_ptr=" << (request.gpu_ptr != nullptr)
              << " device_id=" << request.device_id << " disk_context=" << (request.disk_context != nullptr)
              << " work_items=" << request.representation_work_plan.items.size()
              << " gpu_allocation=" << (request.gpu_allocation != nullptr);
    return {.handled = false, .status = absl::OkStatus()};
  }
  if (!request.disk_context->is_safetensors()) {
    LOG(INFO) << "collective_disk_load skipped group_id=" << request.group.group_id << " reason=non_safetensors_source";
    return {.handled = false, .status = absl::OkStatus()};
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
