// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/replica/collective_disk_loader.h"

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
#include "core/cuda/cuda_api.h"
#include "core/cuda/error_handling.h"
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
constexpr size_t kPreferredChunkBytes = 128ULL * 1024ULL * 1024ULL;

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
  std::string source_index_json;
  std::string view_index_json;
  std::optional<loading::VariantIdentity> variant_identity;
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

ABSL_CONST_INIT absl::Mutex g_group_mu(absl::kConstInit);
absl::flat_hash_map<std::string, std::shared_ptr<GroupState>> g_groups ABSL_GUARDED_BY(g_group_mu);
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

absl::StatusOr<std::vector<TensorJob>> build_tensor_jobs(const std::vector<ParsedParticipant>& participants) {
  if (participants.empty()) {
    return std::vector<TensorJob>{};
  }
  std::vector<absl::flat_hash_map<std::string, TensorMeta>> source_by_rank;
  std::vector<absl::flat_hash_map<std::string, TensorMeta>> view_by_rank;
  source_by_rank.reserve(participants.size());
  view_by_rank.reserve(participants.size());
  for (const auto& participant : participants) {
    auto source_or = parse_index_json(participant.source_index_json);
    if (!source_or.ok()) {
      return source_or.status();
    }
    auto view_or = parse_index_json(participant.view_index_json);
    if (!view_or.ok()) {
      return view_or.status();
    }
    source_by_rank.push_back(std::move(*source_or));
    view_by_rank.push_back(std::move(*view_or));
  }

  std::vector<std::string> names;
  names.reserve(view_by_rank.front().size());
  for (const auto& [name, _] : view_by_rank.front()) {
    names.push_back(name);
  }
  std::sort(names.begin(), names.end(), [&](const std::string& a, const std::string& b) {
    return source_by_rank.front().at(a).offset < source_by_rank.front().at(b).offset;
  });

  std::vector<TensorJob> jobs;
  jobs.reserve(names.size());
  for (const auto& name : names) {
    TensorJob job;
    job.name = name;
    job.source = source_by_rank.front().at(name);
    job.slices.resize(participants.size());

    bool any_dim0 = false;
    bool any_dim1 = false;
    bool any_full = false;
    for (size_t idx = 0; idx < participants.size(); ++idx) {
      const auto source_it = source_by_rank[idx].find(name);
      const auto view_it = view_by_rank[idx].find(name);
      if (source_it == source_by_rank[idx].end() || view_it == view_by_rank[idx].end()) {
        return absl::FailedPreconditionError(absl::StrCat("collective disk load requires the same tensor set: ", name));
      }
      std::optional<materialization::view::TensorViewOps> tensor_ops;
      if (participants[idx].variant_identity.has_value() && participants[idx].variant_identity->view_spec.has_value()) {
        const auto& tensors = participants[idx].variant_identity->view_spec->tensors;
        if (auto spec_it = tensors.find(name); spec_it != tensors.end()) {
          tensor_ops = spec_it->second;
        }
      }
      auto slice_or = build_rank_tensor_slice(source_it->second, view_it->second, tensor_ops);
      if (!slice_or.ok()) {
        return slice_or.status();
      }
      job.slices[idx] = *slice_or;
      any_full = any_full || slice_or->kind == RankTensorSlice::Kind::kFull;
      any_dim0 = any_dim0 || slice_or->kind == RankTensorSlice::Kind::kDim0;
      any_dim1 = any_dim1 || slice_or->kind == RankTensorSlice::Kind::kDim1;
    }

    if (any_dim0 && (any_dim1 || any_full)) {
      return absl::UnimplementedError(absl::StrCat("mixed slice kinds for tensor are unsupported: ", name));
    }
    if (any_dim1 && any_full) {
      return absl::UnimplementedError(absl::StrCat("mixed dim1/full slices are unsupported: ", name));
    }
    if (any_dim1) {
      job.distribution = TensorJob::Distribution::kDim1Partitioned;
    } else if (any_dim0) {
      job.distribution = TensorJob::Distribution::kDim0Partitioned;
    } else {
      job.distribution = TensorJob::Distribution::kReplicated;
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

absl::Status execute_replicated_tensor(
    const TensorJob& job,
    const std::vector<ParsedParticipant>& participants,
    loader::MultiSafetensorsSource& source,
    NcclClique& clique,
    char* host_buffer,
    size_t host_buffer_bytes,
    void* root_stage_ptr,
    cudaStream_t h2d_stream,
    cudaEvent_t ready_event,
    int root_rank) {
  uint64_t copied = 0;
  while (copied < job.source.size_bytes) {
    const size_t chunk_bytes = static_cast<size_t>(
        std::min<uint64_t>(job.source.size_bytes - copied, static_cast<uint64_t>(host_buffer_bytes)));
    TC_RETURN_IF_ERROR(read_exact(source, job.source.offset + copied, host_buffer, chunk_bytes));
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
    loader::MultiSafetensorsSource& source,
    NcclClique& clique,
    char* host_buffer,
    size_t host_buffer_bytes,
    void* root_stage_ptr,
    cudaStream_t h2d_stream,
    cudaEvent_t ready_event,
    int root_rank) {
  uint64_t per_row_bytes = job.source.elem_size;
  for (size_t dim = 1; dim < job.source.shape.size(); ++dim) {
    per_row_bytes *= static_cast<uint64_t>(job.source.shape[dim]);
  }
  const uint64_t full_bytes = job.source.size_bytes;
  uint64_t copied = 0;
  while (copied < full_bytes) {
    const size_t chunk_bytes =
        static_cast<size_t>(std::min<uint64_t>(full_bytes - copied, static_cast<uint64_t>(host_buffer_bytes)));
    TC_RETURN_IF_ERROR(read_exact(source, job.source.offset + copied, host_buffer, chunk_bytes));
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
    TC_RETURN_IF_ERROR(
        tensorcast::cuda::memcpy_async(root_stage_ptr, host_buffer, chunk_bytes, cudaMemcpyHostToDevice, h2d_stream));
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(ready_event, h2d_stream));
    TC_RETURN_IF_ERROR(clique.wait_stream_on_event(root_rank, ready_event));

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
        TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[idx].device_id));
        TC_RETURN_IF_ERROR(
            tensorcast::cuda::memcpy_async(
                dst_ptr, src_ptr, overlap_bytes, cudaMemcpyDeviceToDevice, clique.stream(root_rank)));
      } else {
        TC_RETURN_IF_ERROR(clique.send_u8(root_rank, src_ptr, overlap_bytes, static_cast<int>(idx)));
        TC_RETURN_IF_ERROR(clique.recv_u8(static_cast<int>(idx), dst_ptr, overlap_bytes, root_rank));
      }
    }
    TC_RETURN_IF_ERROR(clique.group_end());
    TC_RETURN_IF_ERROR(clique.synchronize_all());
    copied += static_cast<uint64_t>(chunk_bytes);
  }
  return absl::OkStatus();
}

absl::Status execute_dim1_tensor(
    const TensorJob& job,
    const std::vector<ParsedParticipant>& participants,
    loader::MultiSafetensorsSource& source,
    NcclClique& clique,
    char* host_buffer,
    size_t host_buffer_bytes,
    void* root_stage_ptr,
    cudaStream_t h2d_stream,
    cudaEvent_t ready_event,
    int root_rank) {
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
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&pack_stream, cudaStreamNonBlocking));

  for (uint64_t row = 0; row < rows; row += rows_per_chunk) {
    const uint64_t chunk_rows = std::min<uint64_t>(rows_per_chunk, rows - row);
    const uint64_t chunk_bytes = chunk_rows * row_bytes;
    TC_RETURN_IF_ERROR(
        read_exact(source, job.source.offset + row * row_bytes, host_buffer, static_cast<size_t>(chunk_bytes)));
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
    TC_RETURN_IF_ERROR(
        tensorcast::cuda::memcpy_async(
            root_stage_ptr, host_buffer, static_cast<size_t>(chunk_bytes), cudaMemcpyHostToDevice, h2d_stream));
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(ready_event, h2d_stream));
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_wait_event(pack_stream, ready_event));

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
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(pack_stream));
    TC_RETURN_IF_ERROR(clique.group_start());
    for (size_t idx = 0; idx < participants.size(); ++idx) {
      if (static_cast<int>(idx) == root_rank) {
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
    TC_RETURN_IF_ERROR(clique.synchronize_all());
  }

  if (pack_stream != nullptr) {
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_destroy(pack_stream));
  }
  return absl::OkStatus();
}

absl::Status execute_group_collective(
    const std::vector<ParsedParticipant>& participants,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout) {
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

  PinnedBorrow host_pool;
  host_pool.pool = pinned_pool;
  const auto pinned_alloc_start = std::chrono::steady_clock::now();
  if (pinned_pool->allocate(
          pinned_pool->slice_bytes(),
          host_pool.buffers,
          pinned_timeout,
          absl::StrCat("collective_disk_load artifact_id=", participants.front().replica_key.artifact_id)) != 0 ||
      host_pool.buffers.empty()) {
    return absl::ResourceExhaustedError("failed to allocate pinned buffer for collective disk load");
  }
  const auto pinned_alloc_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - pinned_alloc_start).count();
  const size_t chunk_bytes = std::min<size_t>(kPreferredChunkBytes, pinned_pool->slice_bytes());
  char* host_buffer = host_pool.buffers.front();

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
  size_t replicated_jobs = 0;
  size_t dim0_jobs = 0;
  size_t dim1_jobs = 0;
  double replicated_sec = 0.0;
  double dim0_sec = 0.0;
  double dim1_sec = 0.0;
  for (const auto& job : *jobs_or) {
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
        dim1_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - job_start).count();
        break;
    }
  }

  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
  TC_RETURN_IF_ERROR(tensorcast::cuda::event_destroy(ready_event));
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_destroy(h2d_stream));
  const auto total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  LOG(INFO) << "collective_disk_load timings: clique_init=" << clique_sec << "s"
            << " clique_cache_hit=" << (clique_cache_hit ? 1 : 0) << " pinned_alloc=" << pinned_alloc_sec << "s"
            << " root_alloc=" << root_alloc_sec << "s"
            << " build_jobs=" << jobs_sec << "s"
            << " replicated_jobs=" << replicated_jobs << " replicated_exec=" << replicated_sec << "s"
            << " dim0_jobs=" << dim0_jobs << " dim0_exec=" << dim0_sec << "s"
            << " dim1_jobs=" << dim1_jobs << " dim1_exec=" << dim1_sec << "s"
            << " chunk_bytes=" << chunk_bytes << " total=" << total_sec << "s";
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
      .source_index_json = request.source_index_json,
      .view_index_json = request.view_index_json,
      .variant_identity = request.variant_identity,
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
      const absl::Time deadline = absl::Now() + absl::Milliseconds(kGroupAssembleTimeout.count());
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
    const absl::Status exec_status = execute_group_collective(participants, pinned_pool, pinned_timeout);
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

} // namespace

CollectiveDiskLoadResult try_collective_disk_load(
    const CollectiveDiskLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout) {
  if (request.group.world_size <= 1 || request.gpu_ptr == nullptr || request.device_id < 0 ||
      request.disk_context == nullptr || request.source_index_json.empty() || request.view_index_json.empty() ||
      request.gpu_allocation == nullptr) {
    LOG(INFO) << "collective_disk_load skipped group_id=" << request.group.group_id << " reason=request_incomplete"
              << " world_size=" << request.group.world_size << " gpu_ptr=" << (request.gpu_ptr != nullptr)
              << " device_id=" << request.device_id << " disk_context=" << (request.disk_context != nullptr)
              << " source_index=" << (!request.source_index_json.empty())
              << " view_index=" << (!request.view_index_json.empty())
              << " gpu_allocation=" << (request.gpu_allocation != nullptr);
    return {.handled = false, .status = absl::OkStatus()};
  }
  if (!request.disk_context->is_safetensors()) {
    LOG(INFO) << "collective_disk_load skipped group_id=" << request.group.group_id << " reason=non_safetensors_source";
    return {.handled = false, .status = absl::OkStatus()};
  }
  if (request.variant_identity.has_value() && request.variant_identity->view_spec.has_value()) {
    for (const auto& [_, ops] : request.variant_identity->view_spec->tensors) {
      if (ops.ops.size() > 1) {
        LOG(INFO) << "collective_disk_load skipped group_id=" << request.group.group_id << " reason=multi_op_view_spec";
        return {.handled = false, .status = absl::OkStatus()};
      }
    }
  }
  return wait_for_group_and_maybe_execute(request, pinned_pool, pinned_timeout);
}

} // namespace tensorcast::store::replica
