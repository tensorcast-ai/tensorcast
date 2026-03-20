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
constexpr size_t kPreferredChunkBytes = 128ULL * 1024ULL * 1024ULL;
constexpr size_t kWholeSourceFreeMemoryReserveBytes = 512ULL * 1024ULL * 1024ULL;
constexpr size_t kMaxMappedPeerPairsPerNcclGroup = 1024;

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
  loader::ByteRangeMap map;
  loading::IntoTargetLayout target_layout;
  std::vector<TargetStorageSpan> storage_spans;
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

absl::StatusOr<std::vector<MappedSegmentRef>> build_mapped_segment_refs(
    const std::vector<ParsedMappedParticipant>& participants) {
  std::vector<MappedSegmentRef> segments;
  for (const auto& participant : participants) {
    if (participant.map.num_sources != 1) {
      return absl::InvalidArgumentError("mapped collective load requires mapping.num_sources == 1");
    }
    for (const auto& segment : participant.map.segments) {
      if (segment.kind != loader::ByteRangeSegment::Kind::kData) {
        return absl::UnimplementedError("mapped collective load does not support pad segments");
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

// Whole-source preload is the required execution mode for collective disk
// load. If this invariant cannot be satisfied, the request must fail.
absl::StatusOr<std::unique_ptr<common::memory::GpuDeviceMemory>> load_whole_source_to_root_buffer_required(
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

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  TC_RETURN_IF_ERROR(tensorcast::cuda::get_memory_info(&free_bytes, &total_bytes, root_device_id));
  if (source_bytes + kWholeSourceFreeMemoryReserveBytes > free_bytes) {
    return absl::ResourceExhaustedError(
        absl::StrCat(
            "whole-source root buffer required but unavailable: source_bytes=",
            source_bytes,
            " free_bytes=",
            free_bytes,
            " total_bytes=",
            total_bytes,
            " reserve_bytes=",
            kWholeSourceFreeMemoryReserveBytes));
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
  return root_source;
}

absl::Status execute_replicated_tensor_from_root_buffer(
    const TensorJob& job,
    const std::vector<ParsedParticipant>& participants,
    NcclClique& clique,
    const void* root_source_ptr,
    size_t chunk_bytes,
    int root_rank) {
  uint64_t copied = 0;
  while (copied < job.source.size_bytes) {
    const size_t step =
        static_cast<size_t>(std::min<uint64_t>(job.source.size_bytes - copied, static_cast<uint64_t>(chunk_bytes)));
    const auto* send_ptr = static_cast<const uint8_t*>(root_source_ptr) + job.source.offset + copied;
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
      const auto* src_ptr = static_cast<const uint8_t*>(root_source_ptr) + job.source.offset + overlap_begin;
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
    const auto* row_base_ptr = static_cast<const uint8_t*>(root_source_ptr) + job.source.offset + row * row_bytes;

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
      TC_RETURN_IF_ERROR(read_exact(source, job.source.offset + copied, host_buffer, chunk_bytes));
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
    loader::MultiSafetensorsSource& source,
    NcclClique& clique,
    char* host_buffer,
    size_t host_buffer_bytes,
    void* root_stage_ptr,
    cudaStream_t h2d_stream,
    cudaEvent_t ready_event,
    int root_rank) {
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
    const auto alloc_start = std::chrono::steady_clock::now();
    TC_RETURN_IF_ERROR(
        pack_buffers[idx]->allocate(
            static_cast<size_t>(rows_per_chunk * col_bytes), participants[static_cast<size_t>(root_rank)].device_id));
    pack_alloc_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - alloc_start).count();
  }

  cudaStream_t pack_stream = nullptr;
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(participants[static_cast<size_t>(root_rank)].device_id));
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&pack_stream, cudaStreamNonBlocking));

  for (uint64_t row = 0; row < rows; row += rows_per_chunk) {
    chunk_count += 1;
    const uint64_t chunk_rows = std::min<uint64_t>(rows_per_chunk, rows - row);
    const uint64_t chunk_bytes = chunk_rows * row_bytes;
    {
      const auto step_start = std::chrono::steady_clock::now();
      TC_RETURN_IF_ERROR(
          read_exact(source, job.source.offset + row * row_bytes, host_buffer, static_cast<size_t>(chunk_bytes)));
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
        TC_RETURN_IF_ERROR(clique.send_u8(root_rank, pack_buffers[idx]->get(), send_bytes, static_cast<int>(idx)));
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

  if (pack_stream != nullptr) {
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
  const size_t chunk_bytes = std::min<size_t>(kPreferredChunkBytes, pinned_pool->slice_bytes());

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
  const auto whole_source_start = std::chrono::steady_clock::now();
  std::unique_ptr<common::memory::GpuDeviceMemory> root_source;
  bool whole_source_enabled = false;
  if (participants.front().disk_context->safetensors_segments().empty()) {
    return absl::FailedPreconditionError("collective disk load requires non-empty safetensors segments");
  }
  auto root_source_or = load_whole_source_to_root_buffer_required(
      participants,
      participants.front().disk_context->safetensors_segments(),
      pinned_pool,
      pinned_timeout,
      chunk_bytes);
  if (!root_source_or.ok()) {
    return root_source_or.status();
  }
  root_source = std::move(*root_source_or);
  whole_source_enabled = true;
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
  double replicated_sec = 0.0;
  double dim0_sec = 0.0;
  double dim1_sec = 0.0;
  for (const auto& job : *jobs_or) {
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
            << " whole_source=" << (whole_source_enabled ? 1 : 0) << " whole_source_load=" << whole_source_sec << "s"
            << " whole_source_bytes=" << (root_source != nullptr ? root_source->size() : 0)
            << " build_jobs=" << jobs_sec << "s"
            << " replicated_jobs=" << replicated_jobs << " replicated_exec=" << replicated_sec << "s"
            << " dim0_jobs=" << dim0_jobs << " dim0_exec=" << dim0_sec << "s"
            << " dim1_jobs=" << dim1_jobs << " dim1_exec=" << dim1_sec << "s"
            << " chunk_bytes=" << chunk_bytes << " total=" << total_sec << "s";
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
            << " build_segments_sec=" << segments_sec;

  PreadMultiSafetensorsSource source(
      std::vector<loader::SharedSafetensorsSegment>(
          participants.front().disk_context->safetensors_segments().begin(),
          participants.front().disk_context->safetensors_segments().end()));
  ChunkPrefetcher prefetcher(&source);
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
    return absl::InvalidArgumentError("mapped collective load has no chunks to materialize");
  }
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

CollectiveMappedTargetLoadResult wait_for_mapped_group_and_maybe_execute(
    const CollectiveMappedTargetLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options) {
  auto storage_spans_or = build_target_storage_spans(request.target_layout);
  if (!storage_spans_or.ok()) {
    return {.handled = false, .status = storage_spans_or.status()};
  }
  auto parsed = ParsedMappedParticipant{
      .artifact_id = request.artifact_id,
      .rank = static_cast<int>(request.group.rank),
      .device_id = request.device_id,
      .disk_context = request.disk_context,
      .map = request.map,
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
      return {.handled = false, .status = absl::InvalidArgumentError("mapped collective rank out of range")};
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
        LOG(INFO) << "collective_mapped_target timeout group_id=" << request.group.group_id
                  << " rank=" << request.group.rank << " remaining=" << state->joined;
      }
    }
  }
  if (erase_empty_group) {
    absl::MutexLock group_lock(&g_mapped_group_mu);
    g_mapped_groups.erase(request.group.group_id);
    return {.handled = false, .status = absl::OkStatus()};
  }
  if (timed_out) {
    return {.handled = false, .status = absl::OkStatus()};
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

CollectiveMappedTargetLoadResult try_collective_mapped_target_load(
    const CollectiveMappedTargetLoadRequest& request,
    const std::shared_ptr<common::memory::PinnedBufferPool>& pinned_pool,
    std::chrono::milliseconds pinned_timeout,
    const CollectiveMappedTargetLoadOptions& options) {
  if (request.group.world_size <= 1 || request.device_id < 0 || request.disk_context == nullptr ||
      request.target_layout.storages.empty() || request.map.total_bytes == 0 || request.artifact_id.empty()) {
    LOG(INFO) << "collective_mapped_target skipped group_id=" << request.group.group_id << " reason=request_incomplete"
              << " world_size=" << request.group.world_size << " device_id=" << request.device_id
              << " disk_context=" << (request.disk_context != nullptr)
              << " storages=" << request.target_layout.storages.size() << " map_bytes=" << request.map.total_bytes
              << " artifact_id=" << (!request.artifact_id.empty());
    return {.handled = false, .status = absl::OkStatus()};
  }
  if (!request.disk_context->is_safetensors()) {
    LOG(INFO) << "collective_mapped_target skipped group_id=" << request.group.group_id
              << " reason=non_safetensors_source";
    return {.handled = false, .status = absl::OkStatus()};
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
