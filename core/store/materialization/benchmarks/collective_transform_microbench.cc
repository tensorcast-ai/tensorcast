// Copyright (c) 2025-2026, TensorCast Team.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <nccl.h>
#include <torch/torch.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "core/common/logging_init.h"
#include "core/cuda/cuda_api.h"

ABSL_FLAG(
    std::string,
    mode,
    "all",
    "Benchmark mode: all|memcpy2d_sendrecv|torch_narrow_copy_sendrecv|"
    "torch_transpose_roundtrip_sendrecv|dim0_direct_sendrecv|"
    "dim0_torch_narrow_copy_sendrecv|dim0_grouped_sendrecv|"
    "source_window_allgather_scatter|source_window_allgather_scatter_graph|"
    "source_window_allgather_scatter_graph_update|"
    "source_window_allgather_scatter_graph_exec_update|source_window_allgather_no_scatter");
ABSL_FLAG(std::string, device_ids, "0,1,2,3,4,5,6,7", "Comma-separated CUDA device ids");
ABSL_FLAG(int, rows, 4096, "Rows per chunk tensor");
ABSL_FLAG(int, cols, 16384, "Cols per chunk tensor");
ABSL_FLAG(int, jobs, 1, "Number of synthetic per-tensor jobs for dim0 modes");
ABSL_FLAG(
    int,
    source_window_scatter_ops_per_rank,
    1,
    "Number of 2D scatter ops per rank for source-window scatter mode");
ABSL_FLAG(std::string, dtype, "bf16", "Tensor dtype: bf16|fp16|fp32");
ABSL_FLAG(int, warmup_iters, 3, "Warmup iterations");
ABSL_FLAG(int, iters, 10, "Measured iterations");
ABSL_FLAG(bool, check, true, "Run one correctness check per mode");

namespace tensorcast::bench {
namespace {

#define TC_RETURN_IF_ERROR(expr)     \
  do {                               \
    ::absl::Status _status = (expr); \
    if (!_status.ok()) {             \
      return _status;                \
    }                                \
  } while (false)

absl::Status cuda_status(cudaError_t rc, std::string_view what) {
  if (rc == cudaSuccess) {
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrCat(what, ": ", cudaGetErrorString(rc)));
}

absl::Status nccl_status(ncclResult_t rc, std::string_view what) {
  if (rc == ncclSuccess) {
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrCat(what, ": ", ncclGetErrorString(rc)));
}

enum class Mode {
  kMemcpy2DPackSendrecv,
  kTorchNarrowCopySendrecv,
  kTorchTransposeRoundtripSendrecv,
  kDim0DirectSendrecv,
  kDim0TorchNarrowCopySendrecv,
  kDim0GroupedSendrecv,
  kSourceWindowAllGatherScatter,
  kSourceWindowAllGatherScatterGraph,
  kSourceWindowAllGatherScatterGraphUpdate,
  kSourceWindowAllGatherScatterGraphExecUpdate,
  kSourceWindowAllGatherNoScatter,
};

std::string mode_name(Mode mode) {
  switch (mode) {
    case Mode::kMemcpy2DPackSendrecv:
      return "memcpy2d_sendrecv";
    case Mode::kTorchNarrowCopySendrecv:
      return "torch_narrow_copy_sendrecv";
    case Mode::kTorchTransposeRoundtripSendrecv:
      return "torch_transpose_roundtrip_sendrecv";
    case Mode::kDim0DirectSendrecv:
      return "dim0_direct_sendrecv";
    case Mode::kDim0TorchNarrowCopySendrecv:
      return "dim0_torch_narrow_copy_sendrecv";
    case Mode::kDim0GroupedSendrecv:
      return "dim0_grouped_sendrecv";
    case Mode::kSourceWindowAllGatherScatter:
      return "source_window_allgather_scatter";
    case Mode::kSourceWindowAllGatherScatterGraph:
      return "source_window_allgather_scatter_graph";
    case Mode::kSourceWindowAllGatherScatterGraphUpdate:
      return "source_window_allgather_scatter_graph_update";
    case Mode::kSourceWindowAllGatherScatterGraphExecUpdate:
      return "source_window_allgather_scatter_graph_exec_update";
    case Mode::kSourceWindowAllGatherNoScatter:
      return "source_window_allgather_no_scatter";
  }
  return "unknown";
}

bool is_source_window_mode(Mode mode) {
  return mode == Mode::kSourceWindowAllGatherScatter || mode == Mode::kSourceWindowAllGatherScatterGraph ||
      mode == Mode::kSourceWindowAllGatherScatterGraphUpdate ||
      mode == Mode::kSourceWindowAllGatherScatterGraphExecUpdate || mode == Mode::kSourceWindowAllGatherNoScatter;
}

absl::StatusOr<std::vector<int>> parse_device_ids(std::string_view raw) {
  std::vector<int> ids;
  std::stringstream ss{std::string(raw)};
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    int value = -1;
    if (!absl::SimpleAtoi(token, &value) || value < 0) {
      return absl::InvalidArgumentError(absl::StrCat("invalid device id: ", token));
    }
    ids.push_back(value);
  }
  if (ids.empty()) {
    return absl::InvalidArgumentError("device_ids must not be empty");
  }
  return ids;
}

absl::StatusOr<torch::ScalarType> parse_dtype(std::string_view raw) {
  if (raw == "bf16") {
    return torch::kBFloat16;
  }
  if (raw == "fp16") {
    return torch::kFloat16;
  }
  if (raw == "fp32") {
    return torch::kFloat32;
  }
  return absl::InvalidArgumentError(absl::StrCat("unsupported dtype: ", raw));
}

size_t dtype_nbytes(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kBFloat16:
    case torch::kFloat16:
      return 2;
    case torch::kFloat32:
      return 4;
    default:
      return 0;
  }
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
        (void)cudaSetDevice(rank.device_id);
        (void)cudaStreamDestroy(rank.stream);
      }
      if (rank.comm != nullptr) {
        (void)ncclCommDestroy(rank.comm);
      }
    }
  }

  NcclClique() = default;
  NcclClique(const NcclClique&) = delete;
  NcclClique& operator=(const NcclClique&) = delete;

  NcclClique(NcclClique&& other) noexcept {
    ranks_ = std::move(other.ranks_);
    for (auto& rank : other.ranks_) {
      rank.stream = nullptr;
      rank.comm = nullptr;
      rank.device_id = -1;
    }
  }

  NcclClique& operator=(NcclClique&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    for (auto& rank : ranks_) {
      if (rank.stream != nullptr) {
        (void)cudaSetDevice(rank.device_id);
        (void)cudaStreamDestroy(rank.stream);
      }
      if (rank.comm != nullptr) {
        (void)ncclCommDestroy(rank.comm);
      }
    }
    ranks_ = std::move(other.ranks_);
    for (auto& rank : other.ranks_) {
      rank.stream = nullptr;
      rank.comm = nullptr;
      rank.device_id = -1;
    }
    return *this;
  }

  static absl::StatusOr<NcclClique> create(const std::vector<int>& device_ids) {
    if (device_ids.size() <= 1) {
      return absl::InvalidArgumentError("NcclClique requires at least 2 devices");
    }
    int device_count = 0;
    TC_RETURN_IF_ERROR(cuda_status(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount"));
    for (int id : device_ids) {
      if (id < 0 || id >= device_count) {
        return absl::InvalidArgumentError(absl::StrCat("invalid device id: ", id));
      }
    }

    NcclClique clique;
    clique.ranks_.resize(device_ids.size());
    for (size_t i = 0; i < device_ids.size(); ++i) {
      clique.ranks_[i].device_id = device_ids[i];
      TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(device_ids[i]), "cudaSetDevice"));
      TC_RETURN_IF_ERROR(cuda_status(
          cudaStreamCreateWithFlags(&clique.ranks_[i].stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"));
    }
    std::vector<ncclComm_t> comms(device_ids.size(), nullptr);
    TC_RETURN_IF_ERROR(nccl_status(
        ncclCommInitAll(comms.data(), static_cast<int>(device_ids.size()), device_ids.data()), "ncclCommInitAll"));
    for (size_t i = 0; i < comms.size(); ++i) {
      clique.ranks_[i].comm = comms[i];
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

  absl::Status group_start() {
    return nccl_status(ncclGroupStart(), "ncclGroupStart");
  }

  absl::Status group_end() {
    return nccl_status(ncclGroupEnd(), "ncclGroupEnd");
  }

  absl::Status send_u8(int rank, const void* src, size_t bytes, int peer_rank) {
    return nccl_status(
        ncclSend(src, bytes, ncclUint8, peer_rank, ranks_[static_cast<size_t>(rank)].comm, stream(rank)), "ncclSend");
  }

  absl::Status recv_u8(int rank, void* dst, size_t bytes, int peer_rank) {
    return nccl_status(
        ncclRecv(dst, bytes, ncclUint8, peer_rank, ranks_[static_cast<size_t>(rank)].comm, stream(rank)), "ncclRecv");
  }

  absl::Status all_gather_u8(int rank, const void* src, void* dst, size_t send_bytes) {
    return nccl_status(
        ncclAllGather(src, dst, send_bytes, ncclUint8, ranks_[static_cast<size_t>(rank)].comm, stream(rank)),
        "ncclAllGather");
  }

  absl::Status synchronize_all() {
    for (const auto& rank : ranks_) {
      TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(rank.device_id), "cudaSetDevice"));
      TC_RETURN_IF_ERROR(cuda_status(cudaStreamSynchronize(rank.stream), "cudaStreamSynchronize"));
    }
    return absl::OkStatus();
  }

 private:
  std::vector<RankCtx> ranks_;
};

struct PhaseStats {
  double sender_transform_sec{0.0};
  double nccl_sec{0.0};
  double receiver_transform_sec{0.0};
  double graph_update_sec{0.0};
  double total_sec{0.0};
};

enum class SourceWindowScatterGraphKind {
  kNone,
  kCapturedStatic,
  kMemcpyNodeTemplate,
  kMemcpyNode2DExecUpdate,
};

struct BenchContext {
  std::vector<int> device_ids;
  torch::ScalarType dtype;
  size_t elem_bytes{0};
  int rows{0};
  int cols{0};
  int world_size{0};
  int root_rank{0};
  int shard_cols{0};
  torch::Tensor source;
  std::vector<torch::Tensor> dst;
  std::vector<torch::Tensor> dst_update_scratch;
  std::vector<torch::Tensor> pack;
  std::vector<torch::Tensor> stripe_stage;
  std::vector<torch::Tensor> allgather_stage;
  torch::Tensor transposed;
  std::vector<torch::Tensor> recv_transposed;
  cudaStream_t pack_stream{nullptr};
  std::unique_ptr<NcclClique> clique;

  struct ScatterGraph {
    int device_id{-1};
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t exec{nullptr};
    std::vector<cudaGraphNode_t> memcpy_nodes;
  };

  std::vector<ScatterGraph> source_window_scatter_graphs;
  SourceWindowScatterGraphKind source_window_scatter_graph_kind{SourceWindowScatterGraphKind::kNone};
  double source_window_scatter_graph_build_sec{0.0};
  uint64_t source_window_scatter_graph_count{0};
  uint64_t source_window_scatter_graph_node_count{0};
};

struct Dim0BenchContext {
  std::vector<int> device_ids;
  torch::ScalarType dtype;
  size_t elem_bytes{0};
  int rows{0};
  int cols{0};
  int jobs{0};
  int world_size{0};
  int root_rank{0};
  int shard_rows{0};
  std::vector<torch::Tensor> source_jobs;
  std::vector<std::vector<torch::Tensor>> dst_jobs;
  std::vector<torch::Tensor> pack_rows;
  std::vector<torch::Tensor> grouped_pack_rows;
  std::vector<torch::Tensor> grouped_recv_rows;
  cudaStream_t pack_stream{nullptr};
  std::unique_ptr<NcclClique> clique;
};

torch::Tensor make_cuda_tensor(int device_id, torch::ScalarType dtype, const std::vector<int64_t>& shape) {
  auto options = torch::TensorOptions().device(torch::kCUDA, device_id).dtype(dtype);
  return torch::empty(shape, options);
}

torch::Tensor make_cuda_byte_tensor(int device_id, int64_t bytes) {
  auto options = torch::TensorOptions().device(torch::kCUDA, device_id).dtype(torch::kUInt8);
  return torch::empty({bytes}, options);
}

absl::Status init_context(
    BenchContext& ctx,
    const std::vector<int>& device_ids,
    torch::ScalarType dtype,
    int rows,
    int cols) {
  ctx.device_ids = device_ids;
  ctx.dtype = dtype;
  ctx.elem_bytes = dtype_nbytes(dtype);
  ctx.rows = rows;
  ctx.cols = cols;
  ctx.world_size = static_cast<int>(device_ids.size());
  ctx.root_rank = 0;
  if (ctx.cols % ctx.world_size != 0) {
    return absl::InvalidArgumentError("cols must be divisible by world_size");
  }
  ctx.shard_cols = ctx.cols / ctx.world_size;
  const size_t total_bytes = static_cast<size_t>(ctx.rows) * static_cast<size_t>(ctx.cols) * ctx.elem_bytes;
  if (total_bytes % static_cast<size_t>(ctx.world_size) != 0) {
    return absl::InvalidArgumentError("rows * cols * dtype_bytes must be divisible by world_size");
  }
  const size_t stripe_bytes = total_bytes / static_cast<size_t>(ctx.world_size);

  auto clique_or = NcclClique::create(device_ids);
  if (!clique_or.ok()) {
    return clique_or.status();
  }
  ctx.clique = std::make_unique<NcclClique>(std::move(*clique_or));

  ctx.source = make_cuda_tensor(device_ids[ctx.root_rank], dtype, {ctx.rows, ctx.cols});
  ctx.transposed = make_cuda_tensor(device_ids[ctx.root_rank], dtype, {ctx.cols, ctx.rows});
  ctx.dst.reserve(device_ids.size());
  ctx.dst_update_scratch.reserve(device_ids.size());
  ctx.pack.reserve(device_ids.size());
  ctx.stripe_stage.reserve(device_ids.size());
  ctx.allgather_stage.reserve(device_ids.size());
  ctx.recv_transposed.reserve(device_ids.size());
  for (size_t i = 0; i < device_ids.size(); ++i) {
    ctx.dst.push_back(make_cuda_tensor(device_ids[i], dtype, {ctx.rows, ctx.shard_cols}));
    ctx.dst_update_scratch.push_back(make_cuda_tensor(device_ids[i], dtype, {ctx.rows, ctx.shard_cols}));
    ctx.stripe_stage.push_back(make_cuda_byte_tensor(device_ids[i], static_cast<int64_t>(stripe_bytes)));
    ctx.allgather_stage.push_back(make_cuda_tensor(device_ids[i], dtype, {ctx.rows, ctx.cols}));
    if (static_cast<int>(i) == ctx.root_rank) {
      ctx.pack.push_back(torch::Tensor());
    } else {
      ctx.pack.push_back(make_cuda_tensor(device_ids[ctx.root_rank], dtype, {ctx.rows, ctx.shard_cols}));
    }
    ctx.recv_transposed.push_back(make_cuda_tensor(device_ids[i], dtype, {ctx.shard_cols, ctx.rows}));
  }

  TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(device_ids[ctx.root_rank]), "cudaSetDevice"));
  TC_RETURN_IF_ERROR(
      cuda_status(cudaStreamCreateWithFlags(&ctx.pack_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"));

  torch::NoGradGuard no_grad;
  ctx.source.uniform_(0.0, 1.0);
  for (auto& dst : ctx.dst) {
    dst.zero_();
  }
  for (auto& dst : ctx.dst_update_scratch) {
    dst.zero_();
  }
  for (auto& recv : ctx.recv_transposed) {
    recv.zero_();
  }
  for (auto& tensor : ctx.stripe_stage) {
    tensor.zero_();
  }
  for (auto& tensor : ctx.allgather_stage) {
    tensor.zero_();
  }
  if (ctx.transposed.defined()) {
    ctx.transposed.zero_();
  }
  for (auto& pack : ctx.pack) {
    if (pack.defined()) {
      pack.zero_();
    }
  }
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(device_ids[rank]), "cudaSetDevice"));
    TC_RETURN_IF_ERROR(cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init tensors)"));
  }
  TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(device_ids[ctx.root_rank]), "cudaSetDevice"));
  auto* source_bytes = static_cast<uint8_t*>(ctx.source.data_ptr());
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    auto* stripe_ptr = static_cast<uint8_t*>(ctx.stripe_stage[static_cast<size_t>(rank)].data_ptr());
    const auto source_off = static_cast<size_t>(rank) * stripe_bytes;
    if (rank == ctx.root_rank) {
      TC_RETURN_IF_ERROR(cuda_status(
          cudaMemcpyAsync(
              stripe_ptr, source_bytes + source_off, stripe_bytes, cudaMemcpyDeviceToDevice, ctx.clique->stream(rank)),
          "cudaMemcpyAsync(source stripe)"));
    } else {
      TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(device_ids[rank]), "cudaSetDevice"));
      TC_RETURN_IF_ERROR(cuda_status(
          cudaMemcpyPeerAsync(
              stripe_ptr,
              device_ids[rank],
              source_bytes + source_off,
              device_ids[ctx.root_rank],
              stripe_bytes,
              ctx.clique->stream(rank)),
          "cudaMemcpyPeerAsync(source stripe)"));
    }
  }
  TC_RETURN_IF_ERROR(ctx.clique->synchronize_all());
  TC_RETURN_IF_ERROR(cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
  return absl::OkStatus();
}

void clear_source_window_scatter_graphs(BenchContext& ctx) {
  for (auto& graph : ctx.source_window_scatter_graphs) {
    if (graph.device_id >= 0) {
      (void)cudaSetDevice(graph.device_id);
    }
    if (graph.exec != nullptr) {
      (void)cudaGraphExecDestroy(graph.exec);
      graph.exec = nullptr;
    }
    if (graph.graph != nullptr) {
      (void)cudaGraphDestroy(graph.graph);
      graph.graph = nullptr;
    }
  }
  ctx.source_window_scatter_graphs.clear();
  ctx.source_window_scatter_graph_kind = SourceWindowScatterGraphKind::kNone;
  ctx.source_window_scatter_graph_build_sec = 0.0;
  ctx.source_window_scatter_graph_count = 0;
  ctx.source_window_scatter_graph_node_count = 0;
}

void destroy_context(BenchContext& ctx) {
  clear_source_window_scatter_graphs(ctx);
  if (ctx.pack_stream != nullptr) {
    (void)cudaSetDevice(ctx.device_ids[ctx.root_rank]);
    (void)cudaStreamDestroy(ctx.pack_stream);
  }
}

absl::Status init_dim0_context(
    Dim0BenchContext& ctx,
    const std::vector<int>& device_ids,
    torch::ScalarType dtype,
    int rows,
    int cols,
    int jobs) {
  ctx.device_ids = device_ids;
  ctx.dtype = dtype;
  ctx.elem_bytes = dtype_nbytes(dtype);
  ctx.rows = rows;
  ctx.cols = cols;
  ctx.jobs = jobs;
  ctx.world_size = static_cast<int>(device_ids.size());
  ctx.root_rank = 0;
  if (rows <= 0 || cols <= 0 || jobs <= 0) {
    return absl::InvalidArgumentError("rows, cols and jobs must be > 0");
  }
  if (ctx.rows % ctx.world_size != 0) {
    return absl::InvalidArgumentError("rows must be divisible by world_size");
  }
  ctx.shard_rows = ctx.rows / ctx.world_size;

  auto clique_or = NcclClique::create(device_ids);
  if (!clique_or.ok()) {
    return clique_or.status();
  }
  ctx.clique = std::make_unique<NcclClique>(std::move(*clique_or));

  ctx.source_jobs.reserve(static_cast<size_t>(ctx.jobs));
  ctx.dst_jobs.reserve(static_cast<size_t>(ctx.jobs));
  for (int job = 0; job < ctx.jobs; ++job) {
    ctx.source_jobs.push_back(make_cuda_tensor(device_ids[ctx.root_rank], dtype, {ctx.rows, ctx.cols}));
    std::vector<torch::Tensor> per_rank;
    per_rank.reserve(device_ids.size());
    for (int rank = 0; rank < ctx.world_size; ++rank) {
      per_rank.push_back(make_cuda_tensor(device_ids[rank], dtype, {ctx.shard_rows, ctx.cols}));
    }
    ctx.dst_jobs.push_back(std::move(per_rank));
  }

  ctx.pack_rows.reserve(device_ids.size());
  ctx.grouped_pack_rows.reserve(device_ids.size());
  ctx.grouped_recv_rows.reserve(device_ids.size());
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    if (rank == ctx.root_rank) {
      ctx.pack_rows.push_back(torch::Tensor());
    } else {
      ctx.pack_rows.push_back(make_cuda_tensor(device_ids[ctx.root_rank], dtype, {ctx.shard_rows, ctx.cols}));
    }
    ctx.grouped_pack_rows.push_back(
        make_cuda_tensor(device_ids[ctx.root_rank], dtype, {ctx.jobs * ctx.shard_rows, ctx.cols}));
    ctx.grouped_recv_rows.push_back(make_cuda_tensor(device_ids[rank], dtype, {ctx.jobs * ctx.shard_rows, ctx.cols}));
  }

  TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(device_ids[ctx.root_rank]), "cudaSetDevice"));
  TC_RETURN_IF_ERROR(
      cuda_status(cudaStreamCreateWithFlags(&ctx.pack_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"));

  {
    torch::NoGradGuard no_grad;
    for (auto& src : ctx.source_jobs) {
      src.uniform_(0.0, 1.0);
    }
    for (auto& per_job : ctx.dst_jobs) {
      for (auto& dst : per_job) {
        dst.zero_();
      }
    }
    for (auto& tensor : ctx.pack_rows) {
      if (tensor.defined()) {
        tensor.zero_();
      }
    }
    for (auto& tensor : ctx.grouped_pack_rows) {
      tensor.zero_();
    }
    for (auto& tensor : ctx.grouped_recv_rows) {
      tensor.zero_();
    }
  }
  TC_RETURN_IF_ERROR(cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
  return absl::OkStatus();
}

void destroy_dim0_context(Dim0BenchContext& ctx) {
  if (ctx.pack_stream != nullptr) {
    (void)cudaSetDevice(ctx.device_ids[ctx.root_rank]);
    (void)cudaStreamDestroy(ctx.pack_stream);
  }
}

absl::Status zero_outputs(Dim0BenchContext& ctx) {
  torch::NoGradGuard no_grad;
  for (auto& per_job : ctx.dst_jobs) {
    for (auto& dst : per_job) {
      dst.zero_();
    }
  }
  for (auto& tensor : ctx.pack_rows) {
    if (tensor.defined()) {
      tensor.zero_();
    }
  }
  for (auto& tensor : ctx.grouped_pack_rows) {
    tensor.zero_();
  }
  for (auto& tensor : ctx.grouped_recv_rows) {
    tensor.zero_();
  }
  TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(ctx.device_ids[ctx.root_rank]), "cudaSetDevice"));
  TC_RETURN_IF_ERROR(cuda_status(cudaStreamSynchronize(ctx.pack_stream), "cudaStreamSynchronize"));
  TC_RETURN_IF_ERROR(ctx.clique->synchronize_all());
  return absl::OkStatus();
}

absl::Status zero_outputs(BenchContext& ctx) {
  torch::NoGradGuard no_grad;
  for (auto& dst : ctx.dst) {
    dst.zero_();
  }
  for (auto& dst : ctx.dst_update_scratch) {
    dst.zero_();
  }
  for (auto& recv : ctx.recv_transposed) {
    recv.zero_();
  }
  for (auto& tensor : ctx.allgather_stage) {
    tensor.zero_();
  }
  if (ctx.transposed.defined()) {
    ctx.transposed.zero_();
  }
  for (auto& pack : ctx.pack) {
    if (pack.defined()) {
      pack.zero_();
    }
  }
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(ctx.device_ids[rank]), "cudaSetDevice"));
    TC_RETURN_IF_ERROR(cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize(zero outputs)"));
  }
  TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(ctx.device_ids[ctx.root_rank]), "cudaSetDevice"));
  TC_RETURN_IF_ERROR(cuda_status(cudaStreamSynchronize(ctx.pack_stream), "cudaStreamSynchronize"));
  TC_RETURN_IF_ERROR(ctx.clique->synchronize_all());
  return absl::OkStatus();
}

absl::Status run_memcpy2d_sendrecv(BenchContext& ctx, PhaseStats* stats) {
  const size_t row_bytes = static_cast<size_t>(ctx.cols) * ctx.elem_bytes;
  const size_t shard_row_bytes = static_cast<size_t>(ctx.shard_cols) * ctx.elem_bytes;
  const size_t shard_bytes = static_cast<size_t>(ctx.rows) * shard_row_bytes;
  auto* src_ptr = static_cast<uint8_t*>(ctx.source.data_ptr());

  const auto total_start = std::chrono::steady_clock::now();
  const auto sender_start = std::chrono::steady_clock::now();
  TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(ctx.device_ids[ctx.root_rank]), "cudaSetDevice"));
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    auto* dst_ptr = static_cast<uint8_t*>(ctx.dst[static_cast<size_t>(rank)].data_ptr());
    const auto src_off = static_cast<size_t>(rank) * shard_row_bytes;
    if (rank == ctx.root_rank) {
      TC_RETURN_IF_ERROR(cuda_status(
          cudaMemcpy2DAsync(
              dst_ptr,
              shard_row_bytes,
              src_ptr + src_off,
              row_bytes,
              shard_row_bytes,
              ctx.rows,
              cudaMemcpyDeviceToDevice,
              ctx.pack_stream),
          "cudaMemcpy2DAsync(root local)"));
    } else {
      auto* pack_ptr = static_cast<uint8_t*>(ctx.pack[static_cast<size_t>(rank)].data_ptr());
      TC_RETURN_IF_ERROR(cuda_status(
          cudaMemcpy2DAsync(
              pack_ptr,
              shard_row_bytes,
              src_ptr + src_off,
              row_bytes,
              shard_row_bytes,
              ctx.rows,
              cudaMemcpyDeviceToDevice,
              ctx.pack_stream),
          "cudaMemcpy2DAsync(pack)"));
    }
  }
  TC_RETURN_IF_ERROR(cuda_status(cudaStreamSynchronize(ctx.pack_stream), "cudaStreamSynchronize(pack)"));
  stats->sender_transform_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - sender_start).count();

  const auto nccl_start = std::chrono::steady_clock::now();
  TC_RETURN_IF_ERROR(ctx.clique->group_start());
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    if (rank == ctx.root_rank) {
      continue;
    }
    auto* pack_ptr = static_cast<uint8_t*>(ctx.pack[static_cast<size_t>(rank)].data_ptr());
    auto* dst_ptr = static_cast<uint8_t*>(ctx.dst[static_cast<size_t>(rank)].data_ptr());
    TC_RETURN_IF_ERROR(ctx.clique->send_u8(ctx.root_rank, pack_ptr, shard_bytes, rank));
    TC_RETURN_IF_ERROR(ctx.clique->recv_u8(rank, dst_ptr, shard_bytes, ctx.root_rank));
  }
  TC_RETURN_IF_ERROR(ctx.clique->group_end());
  TC_RETURN_IF_ERROR(ctx.clique->synchronize_all());
  stats->nccl_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - nccl_start).count();
  stats->total_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  return absl::OkStatus();
}

absl::Status run_torch_narrow_copy_sendrecv(BenchContext& ctx, PhaseStats* stats) {
  const size_t shard_bytes = static_cast<size_t>(ctx.rows) * ctx.shard_cols * ctx.elem_bytes;
  const auto total_start = std::chrono::steady_clock::now();
  const auto sender_start = std::chrono::steady_clock::now();
  {
    torch::NoGradGuard no_grad;
    for (int rank = 0; rank < ctx.world_size; ++rank) {
      auto slice = ctx.source.narrow(1, rank * ctx.shard_cols, ctx.shard_cols);
      if (rank == ctx.root_rank) {
        ctx.dst[static_cast<size_t>(rank)].copy_(slice);
      } else {
        ctx.pack[static_cast<size_t>(rank)].copy_(slice);
      }
    }
  }
  TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(ctx.device_ids[ctx.root_rank]), "cudaSetDevice"));
  TC_RETURN_IF_ERROR(cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
  stats->sender_transform_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - sender_start).count();

  const auto nccl_start = std::chrono::steady_clock::now();
  TC_RETURN_IF_ERROR(ctx.clique->group_start());
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    if (rank == ctx.root_rank) {
      continue;
    }
    auto* pack_ptr = static_cast<uint8_t*>(ctx.pack[static_cast<size_t>(rank)].data_ptr());
    auto* dst_ptr = static_cast<uint8_t*>(ctx.dst[static_cast<size_t>(rank)].data_ptr());
    TC_RETURN_IF_ERROR(ctx.clique->send_u8(ctx.root_rank, pack_ptr, shard_bytes, rank));
    TC_RETURN_IF_ERROR(ctx.clique->recv_u8(rank, dst_ptr, shard_bytes, ctx.root_rank));
  }
  TC_RETURN_IF_ERROR(ctx.clique->group_end());
  TC_RETURN_IF_ERROR(ctx.clique->synchronize_all());
  stats->nccl_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - nccl_start).count();
  stats->total_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  return absl::OkStatus();
}

absl::Status run_torch_transpose_roundtrip_sendrecv(BenchContext& ctx, PhaseStats* stats) {
  const size_t shard_bytes = static_cast<size_t>(ctx.rows) * ctx.shard_cols * ctx.elem_bytes;
  const auto total_start = std::chrono::steady_clock::now();
  const auto sender_start = std::chrono::steady_clock::now();
  {
    torch::NoGradGuard no_grad;
    ctx.transposed.copy_(ctx.source.transpose(0, 1));
    auto local_slice = ctx.transposed.narrow(0, ctx.root_rank * ctx.shard_cols, ctx.shard_cols);
    ctx.recv_transposed[static_cast<size_t>(ctx.root_rank)].copy_(local_slice);
  }
  TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(ctx.device_ids[ctx.root_rank]), "cudaSetDevice"));
  TC_RETURN_IF_ERROR(cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
  stats->sender_transform_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - sender_start).count();

  const auto nccl_start = std::chrono::steady_clock::now();
  TC_RETURN_IF_ERROR(ctx.clique->group_start());
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    if (rank == ctx.root_rank) {
      continue;
    }
    auto send_slice = ctx.transposed.narrow(0, rank * ctx.shard_cols, ctx.shard_cols);
    auto* src_ptr = static_cast<uint8_t*>(send_slice.data_ptr());
    auto* dst_ptr = static_cast<uint8_t*>(ctx.recv_transposed[static_cast<size_t>(rank)].data_ptr());
    TC_RETURN_IF_ERROR(ctx.clique->send_u8(ctx.root_rank, src_ptr, shard_bytes, rank));
    TC_RETURN_IF_ERROR(ctx.clique->recv_u8(rank, dst_ptr, shard_bytes, ctx.root_rank));
  }
  TC_RETURN_IF_ERROR(ctx.clique->group_end());
  TC_RETURN_IF_ERROR(ctx.clique->synchronize_all());
  stats->nccl_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - nccl_start).count();

  const auto receiver_start = std::chrono::steady_clock::now();
  {
    torch::NoGradGuard no_grad;
    for (int rank = 0; rank < ctx.world_size; ++rank) {
      ctx.dst[static_cast<size_t>(rank)].copy_(ctx.recv_transposed[static_cast<size_t>(rank)].transpose(0, 1));
    }
  }
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(ctx.device_ids[rank]), "cudaSetDevice"));
    TC_RETURN_IF_ERROR(cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
  }
  stats->receiver_transform_sec +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - receiver_start).count();
  stats->total_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  return absl::OkStatus();
}

absl::Status issue_source_window_scatter_copies(BenchContext& ctx, int rank) {
  const size_t row_bytes = static_cast<size_t>(ctx.cols) * ctx.elem_bytes;
  const size_t shard_row_bytes = static_cast<size_t>(ctx.shard_cols) * ctx.elem_bytes;
  const int scatter_ops_per_rank = std::max(1, absl::GetFlag(FLAGS_source_window_scatter_ops_per_rank));
  TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(ctx.device_ids[rank]), "cudaSetDevice"));
  auto* gather_ptr = static_cast<uint8_t*>(ctx.allgather_stage[static_cast<size_t>(rank)].data_ptr());
  auto* dst_ptr = static_cast<uint8_t*>(ctx.dst[static_cast<size_t>(rank)].data_ptr());
  const size_t source_offset = static_cast<size_t>(rank) * shard_row_bytes;
  for (int op = 0; op < scatter_ops_per_rank; ++op) {
    const int row_begin = (ctx.rows * op) / scatter_ops_per_rank;
    const int row_end = (ctx.rows * (op + 1)) / scatter_ops_per_rank;
    if (row_end <= row_begin) {
      continue;
    }
    const size_t row_offset = static_cast<size_t>(row_begin);
    TC_RETURN_IF_ERROR(cuda_status(
        cudaMemcpy2DAsync(
            dst_ptr + row_offset * shard_row_bytes,
            shard_row_bytes,
            gather_ptr + source_offset + row_offset * row_bytes,
            row_bytes,
            shard_row_bytes,
            static_cast<size_t>(row_end - row_begin),
            cudaMemcpyDeviceToDevice,
            ctx.clique->stream(rank)),
        "cudaMemcpy2DAsync(source window scatter)"));
  }
  return absl::OkStatus();
}

struct SourceWindowScatter1DParam {
  void* dst{nullptr};
  const void* src{nullptr};
  size_t count{0};
};

std::vector<cudaMemcpy3DParms> source_window_scatter_2d_params_for_rank(
    BenchContext& ctx,
    int rank,
    bool use_update_scratch_dst) {
  const size_t row_bytes = static_cast<size_t>(ctx.cols) * ctx.elem_bytes;
  const size_t shard_row_bytes = static_cast<size_t>(ctx.shard_cols) * ctx.elem_bytes;
  const int scatter_ops_per_rank = std::max(1, absl::GetFlag(FLAGS_source_window_scatter_ops_per_rank));
  auto* gather_ptr = static_cast<uint8_t*>(ctx.allgather_stage[static_cast<size_t>(rank)].data_ptr());
  auto& dst_tensor =
      use_update_scratch_dst ? ctx.dst_update_scratch[static_cast<size_t>(rank)] : ctx.dst[static_cast<size_t>(rank)];
  auto* dst_ptr = static_cast<uint8_t*>(dst_tensor.data_ptr());
  const size_t source_offset = static_cast<size_t>(rank) * shard_row_bytes;

  std::vector<cudaMemcpy3DParms> params;
  params.reserve(static_cast<size_t>(scatter_ops_per_rank));
  for (int op = 0; op < scatter_ops_per_rank; ++op) {
    const int row_begin = (ctx.rows * op) / scatter_ops_per_rank;
    const int row_end = (ctx.rows * (op + 1)) / scatter_ops_per_rank;
    if (row_end <= row_begin) {
      continue;
    }
    const auto row_count = static_cast<size_t>(row_end - row_begin);
    const size_t row_offset = static_cast<size_t>(row_begin);
    cudaMemcpy3DParms copy_params{};
    copy_params.srcPtr =
        make_cudaPitchedPtr(gather_ptr + source_offset + row_offset * row_bytes, row_bytes, shard_row_bytes, row_count);
    copy_params.dstPtr =
        make_cudaPitchedPtr(dst_ptr + row_offset * shard_row_bytes, shard_row_bytes, shard_row_bytes, row_count);
    copy_params.extent = make_cudaExtent(shard_row_bytes, row_count, 1);
    copy_params.kind = cudaMemcpyDeviceToDevice;
    params.push_back(copy_params);
  }
  return params;
}

std::vector<SourceWindowScatter1DParam> source_window_scatter_1d_params_for_rank(
    BenchContext& ctx,
    int rank,
    bool use_update_scratch_dst) {
  const size_t row_bytes = static_cast<size_t>(ctx.cols) * ctx.elem_bytes;
  const size_t shard_row_bytes = static_cast<size_t>(ctx.shard_cols) * ctx.elem_bytes;
  const int scatter_ops_per_rank = std::max(1, absl::GetFlag(FLAGS_source_window_scatter_ops_per_rank));
  auto* gather_ptr = static_cast<uint8_t*>(ctx.allgather_stage[static_cast<size_t>(rank)].data_ptr());
  auto& dst_tensor =
      use_update_scratch_dst ? ctx.dst_update_scratch[static_cast<size_t>(rank)] : ctx.dst[static_cast<size_t>(rank)];
  auto* dst_ptr = static_cast<uint8_t*>(dst_tensor.data_ptr());
  const size_t source_offset = static_cast<size_t>(rank) * shard_row_bytes;

  std::vector<SourceWindowScatter1DParam> params;
  params.reserve(static_cast<size_t>(ctx.rows));
  for (int op = 0; op < scatter_ops_per_rank; ++op) {
    const int row_begin = (ctx.rows * op) / scatter_ops_per_rank;
    const int row_end = (ctx.rows * (op + 1)) / scatter_ops_per_rank;
    for (int row = row_begin; row < row_end; ++row) {
      const size_t row_offset = static_cast<size_t>(row);
      params.push_back(
          SourceWindowScatter1DParam{
              .dst = dst_ptr + row_offset * shard_row_bytes,
              .src = gather_ptr + source_offset + row_offset * row_bytes,
              .count = shard_row_bytes,
          });
    }
  }
  return params;
}

absl::Status ensure_source_window_scatter_graphs(BenchContext& ctx, SourceWindowScatterGraphKind graph_kind) {
  if (graph_kind == SourceWindowScatterGraphKind::kNone) {
    return absl::InvalidArgumentError("source-window scatter graph kind must not be none");
  }
  if (!ctx.source_window_scatter_graphs.empty()) {
    if (ctx.source_window_scatter_graph_kind == graph_kind) {
      return absl::OkStatus();
    }
    clear_source_window_scatter_graphs(ctx);
  }
  const auto build_start = std::chrono::steady_clock::now();
  ctx.source_window_scatter_graphs.reserve(static_cast<size_t>(ctx.world_size));
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(ctx.device_ids[rank]), "cudaSetDevice"));
    cudaGraph_t graph = nullptr;
    std::vector<cudaGraphNode_t> memcpy_nodes;
    if (graph_kind == SourceWindowScatterGraphKind::kCapturedStatic) {
      cudaStream_t stream = ctx.clique->stream(rank);
      TC_RETURN_IF_ERROR(cuda_status(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          "cudaStreamBeginCapture(source-window scatter graph)"));
      absl::Status issue_status = issue_source_window_scatter_copies(ctx, rank);
      cudaError_t end_status = cudaStreamEndCapture(stream, &graph);
      if (!issue_status.ok()) {
        if (end_status == cudaSuccess && graph != nullptr) {
          (void)cudaGraphDestroy(graph);
        }
        return issue_status;
      }
      TC_RETURN_IF_ERROR(cuda_status(end_status, "cudaStreamEndCapture(source-window scatter graph)"));
    } else if (graph_kind == SourceWindowScatterGraphKind::kMemcpyNodeTemplate) {
      TC_RETURN_IF_ERROR(cuda_status(cudaGraphCreate(&graph, 0), "cudaGraphCreate(source-window scatter graph)"));
      auto params = source_window_scatter_1d_params_for_rank(
          ctx,
          rank,
          /*use_update_scratch_dst=*/true);
      memcpy_nodes.reserve(params.size());
      for (const auto& copy_params : params) {
        cudaGraphNode_t node = nullptr;
        cudaError_t add_status = cudaGraphAddMemcpyNode1D(
            &node, graph, nullptr, 0, copy_params.dst, copy_params.src, copy_params.count, cudaMemcpyDeviceToDevice);
        if (add_status != cudaSuccess) {
          (void)cudaGraphDestroy(graph);
          return cuda_status(add_status, "cudaGraphAddMemcpyNode1D(source-window scatter graph)");
        }
        memcpy_nodes.push_back(node);
      }
    } else {
      TC_RETURN_IF_ERROR(cuda_status(cudaGraphCreate(&graph, 0), "cudaGraphCreate(source-window scatter graph)"));
      auto params = source_window_scatter_2d_params_for_rank(
          ctx,
          rank,
          /*use_update_scratch_dst=*/true);
      memcpy_nodes.reserve(params.size());
      for (const auto& copy_params : params) {
        cudaGraphNode_t node = nullptr;
        cudaError_t add_status = cudaGraphAddMemcpyNode(&node, graph, nullptr, 0, &copy_params);
        if (add_status != cudaSuccess) {
          (void)cudaGraphDestroy(graph);
          return cuda_status(add_status, "cudaGraphAddMemcpyNode(source-window scatter exec-update graph)");
        }
        memcpy_nodes.push_back(node);
      }
    }

    size_t graph_node_count = 0;
    TC_RETURN_IF_ERROR(cuda_status(
        cudaGraphGetNodes(graph, nullptr, &graph_node_count), "cudaGraphGetNodes(source-window scatter graph)"));

    cudaGraphExec_t exec = nullptr;
    cudaError_t instantiate_status = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (instantiate_status != cudaSuccess) {
      (void)cudaGraphDestroy(graph);
      return cuda_status(instantiate_status, "cudaGraphInstantiate(source-window scatter graph)");
    }
    ctx.source_window_scatter_graphs.push_back(
        BenchContext::ScatterGraph{
            .device_id = ctx.device_ids[rank],
            .graph = graph,
            .exec = exec,
            .memcpy_nodes = std::move(memcpy_nodes),
        });
    ctx.source_window_scatter_graph_node_count += static_cast<uint64_t>(graph_node_count);
  }
  ctx.source_window_scatter_graph_kind = graph_kind;
  ctx.source_window_scatter_graph_count = static_cast<uint64_t>(ctx.source_window_scatter_graphs.size());
  ctx.source_window_scatter_graph_build_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();
  return absl::OkStatus();
}

absl::Status update_source_window_scatter_graphs(BenchContext& ctx) {
  if (ctx.source_window_scatter_graph_kind != SourceWindowScatterGraphKind::kMemcpyNodeTemplate) {
    return absl::FailedPreconditionError("source-window scatter graph is not an updatable memcpy-node template");
  }
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    if (rank >= static_cast<int>(ctx.source_window_scatter_graphs.size())) {
      return absl::FailedPreconditionError("source-window scatter graph missing for rank");
    }
    auto& graph = ctx.source_window_scatter_graphs[static_cast<size_t>(rank)];
    auto params = source_window_scatter_1d_params_for_rank(
        ctx,
        rank,
        /*use_update_scratch_dst=*/false);
    if (params.size() != graph.memcpy_nodes.size()) {
      return absl::FailedPreconditionError("source-window scatter graph update node count mismatch");
    }
    for (size_t node_index = 0; node_index < params.size(); ++node_index) {
      TC_RETURN_IF_ERROR(cuda_status(
          cudaGraphExecMemcpyNodeSetParams1D(
              graph.exec,
              graph.memcpy_nodes[node_index],
              params[node_index].dst,
              params[node_index].src,
              params[node_index].count,
              cudaMemcpyDeviceToDevice),
          "cudaGraphExecMemcpyNodeSetParams1D(source-window scatter graph)"));
    }
  }
  return absl::OkStatus();
}

absl::Status update_source_window_scatter_graphs_with_exec_update(BenchContext& ctx, bool use_update_scratch_dst) {
  if (ctx.source_window_scatter_graph_kind != SourceWindowScatterGraphKind::kMemcpyNode2DExecUpdate) {
    return absl::FailedPreconditionError("source-window scatter graph is not a 2D exec-update template");
  }
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    if (rank >= static_cast<int>(ctx.source_window_scatter_graphs.size())) {
      return absl::FailedPreconditionError("source-window scatter graph missing for rank");
    }
    auto& graph = ctx.source_window_scatter_graphs[static_cast<size_t>(rank)];
    auto params = source_window_scatter_2d_params_for_rank(ctx, rank, use_update_scratch_dst);
    if (params.size() != graph.memcpy_nodes.size()) {
      return absl::FailedPreconditionError("source-window scatter graph exec-update node count mismatch");
    }
    for (size_t node_index = 0; node_index < params.size(); ++node_index) {
      TC_RETURN_IF_ERROR(cuda_status(
          cudaGraphMemcpyNodeSetParams(graph.memcpy_nodes[node_index], &params[node_index]),
          "cudaGraphMemcpyNodeSetParams(source-window scatter exec-update graph)"));
    }
    cudaGraphExecUpdateResultInfo update_info{};
    cudaError_t update_status = cudaGraphExecUpdate(graph.exec, graph.graph, &update_info);
    if (update_status != cudaSuccess) {
      return absl::InternalError(
          absl::StrCat(
              "cudaGraphExecUpdate(source-window scatter graph): ",
              cudaGetErrorString(update_status),
              " result=",
              static_cast<int>(update_info.result)));
    }
    if (update_info.result != cudaGraphExecUpdateSuccess) {
      return absl::InternalError(
          absl::StrCat(
              "cudaGraphExecUpdate(source-window scatter graph) result=", static_cast<int>(update_info.result)));
    }
  }
  return absl::OkStatus();
}

enum class SourceWindowScatterMode {
  kNone,
  kMemcpy,
  kGraph,
  kGraphUpdate,
  kGraphExecUpdate,
};

absl::Status run_source_window_allgather(BenchContext& ctx, SourceWindowScatterMode scatter_mode, PhaseStats* stats) {
  const size_t total_bytes = static_cast<size_t>(ctx.rows) * static_cast<size_t>(ctx.cols) * ctx.elem_bytes;
  const size_t stripe_bytes = total_bytes / static_cast<size_t>(ctx.world_size);
  const auto total_start = std::chrono::steady_clock::now();

  const auto nccl_start = std::chrono::steady_clock::now();
  TC_RETURN_IF_ERROR(ctx.clique->group_start());
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    auto* stripe_ptr = static_cast<uint8_t*>(ctx.stripe_stage[static_cast<size_t>(rank)].data_ptr());
    auto* gather_ptr = static_cast<uint8_t*>(ctx.allgather_stage[static_cast<size_t>(rank)].data_ptr());
    TC_RETURN_IF_ERROR(ctx.clique->all_gather_u8(rank, stripe_ptr, gather_ptr, stripe_bytes));
  }
  TC_RETURN_IF_ERROR(ctx.clique->group_end());
  TC_RETURN_IF_ERROR(ctx.clique->synchronize_all());
  stats->nccl_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - nccl_start).count();

  if (scatter_mode != SourceWindowScatterMode::kNone) {
    const auto receiver_start = std::chrono::steady_clock::now();
    if (scatter_mode == SourceWindowScatterMode::kGraph || scatter_mode == SourceWindowScatterMode::kGraphUpdate ||
        scatter_mode == SourceWindowScatterMode::kGraphExecUpdate) {
      SourceWindowScatterGraphKind graph_kind = SourceWindowScatterGraphKind::kCapturedStatic;
      if (scatter_mode == SourceWindowScatterMode::kGraphUpdate) {
        graph_kind = SourceWindowScatterGraphKind::kMemcpyNodeTemplate;
      } else if (scatter_mode == SourceWindowScatterMode::kGraphExecUpdate) {
        graph_kind = SourceWindowScatterGraphKind::kMemcpyNode2DExecUpdate;
      }
      TC_RETURN_IF_ERROR(ensure_source_window_scatter_graphs(ctx, graph_kind));
      if (scatter_mode == SourceWindowScatterMode::kGraphUpdate) {
        const auto update_start = std::chrono::steady_clock::now();
        TC_RETURN_IF_ERROR(update_source_window_scatter_graphs(ctx));
        stats->graph_update_sec +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
      } else if (scatter_mode == SourceWindowScatterMode::kGraphExecUpdate) {
        const auto update_start = std::chrono::steady_clock::now();
        TC_RETURN_IF_ERROR(update_source_window_scatter_graphs_with_exec_update(
            ctx,
            /*use_update_scratch_dst=*/true));
        TC_RETURN_IF_ERROR(update_source_window_scatter_graphs_with_exec_update(
            ctx,
            /*use_update_scratch_dst=*/false));
        stats->graph_update_sec +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - update_start).count();
      }
      for (int rank = 0; rank < ctx.world_size; ++rank) {
        TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(ctx.device_ids[rank]), "cudaSetDevice"));
        if (rank >= static_cast<int>(ctx.source_window_scatter_graphs.size()) ||
            ctx.source_window_scatter_graphs[static_cast<size_t>(rank)].exec == nullptr) {
          return absl::FailedPreconditionError("source-window scatter graph missing for rank");
        }
        TC_RETURN_IF_ERROR(cuda_status(
            cudaGraphLaunch(ctx.source_window_scatter_graphs[static_cast<size_t>(rank)].exec, ctx.clique->stream(rank)),
            "cudaGraphLaunch(source-window scatter graph)"));
      }
    } else {
      for (int rank = 0; rank < ctx.world_size; ++rank) {
        TC_RETURN_IF_ERROR(issue_source_window_scatter_copies(ctx, rank));
      }
    }
    TC_RETURN_IF_ERROR(ctx.clique->synchronize_all());
    stats->receiver_transform_sec +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - receiver_start).count();
  }

  stats->total_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  return absl::OkStatus();
}

absl::Status run_dim0_direct_sendrecv(Dim0BenchContext& ctx, PhaseStats* stats) {
  const size_t shard_bytes = static_cast<size_t>(ctx.shard_rows) * ctx.cols * ctx.elem_bytes;
  const auto total_start = std::chrono::steady_clock::now();
  const auto nccl_start = std::chrono::steady_clock::now();
  for (int job = 0; job < ctx.jobs; ++job) {
    TC_RETURN_IF_ERROR(ctx.clique->group_start());
    for (int rank = 0; rank < ctx.world_size; ++rank) {
      auto slice = ctx.source_jobs[static_cast<size_t>(job)].narrow(0, rank * ctx.shard_rows, ctx.shard_rows);
      if (rank == ctx.root_rank) {
        ctx.dst_jobs[static_cast<size_t>(job)][static_cast<size_t>(rank)].copy_(slice);
        continue;
      }
      auto* src_ptr = static_cast<uint8_t*>(slice.data_ptr());
      auto* dst_ptr =
          static_cast<uint8_t*>(ctx.dst_jobs[static_cast<size_t>(job)][static_cast<size_t>(rank)].data_ptr());
      TC_RETURN_IF_ERROR(ctx.clique->send_u8(ctx.root_rank, src_ptr, shard_bytes, rank));
      TC_RETURN_IF_ERROR(ctx.clique->recv_u8(rank, dst_ptr, shard_bytes, ctx.root_rank));
    }
    TC_RETURN_IF_ERROR(ctx.clique->group_end());
    TC_RETURN_IF_ERROR(ctx.clique->synchronize_all());
  }
  stats->nccl_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - nccl_start).count();
  stats->total_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  return absl::OkStatus();
}

absl::Status run_dim0_torch_narrow_copy_sendrecv(Dim0BenchContext& ctx, PhaseStats* stats) {
  const size_t shard_bytes = static_cast<size_t>(ctx.shard_rows) * ctx.cols * ctx.elem_bytes;
  const auto total_start = std::chrono::steady_clock::now();
  for (int job = 0; job < ctx.jobs; ++job) {
    const auto sender_start = std::chrono::steady_clock::now();
    {
      torch::NoGradGuard no_grad;
      for (int rank = 0; rank < ctx.world_size; ++rank) {
        auto slice = ctx.source_jobs[static_cast<size_t>(job)].narrow(0, rank * ctx.shard_rows, ctx.shard_rows);
        if (rank == ctx.root_rank) {
          ctx.dst_jobs[static_cast<size_t>(job)][static_cast<size_t>(rank)].copy_(slice);
        } else {
          ctx.pack_rows[static_cast<size_t>(rank)].copy_(slice);
        }
      }
    }
    TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(ctx.device_ids[ctx.root_rank]), "cudaSetDevice"));
    TC_RETURN_IF_ERROR(cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
    stats->sender_transform_sec +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - sender_start).count();

    const auto nccl_start = std::chrono::steady_clock::now();
    TC_RETURN_IF_ERROR(ctx.clique->group_start());
    for (int rank = 0; rank < ctx.world_size; ++rank) {
      if (rank == ctx.root_rank) {
        continue;
      }
      auto* src_ptr = static_cast<uint8_t*>(ctx.pack_rows[static_cast<size_t>(rank)].data_ptr());
      auto* dst_ptr =
          static_cast<uint8_t*>(ctx.dst_jobs[static_cast<size_t>(job)][static_cast<size_t>(rank)].data_ptr());
      TC_RETURN_IF_ERROR(ctx.clique->send_u8(ctx.root_rank, src_ptr, shard_bytes, rank));
      TC_RETURN_IF_ERROR(ctx.clique->recv_u8(rank, dst_ptr, shard_bytes, ctx.root_rank));
    }
    TC_RETURN_IF_ERROR(ctx.clique->group_end());
    TC_RETURN_IF_ERROR(ctx.clique->synchronize_all());
    stats->nccl_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - nccl_start).count();
  }
  stats->total_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  return absl::OkStatus();
}

absl::Status run_dim0_grouped_sendrecv(Dim0BenchContext& ctx, PhaseStats* stats) {
  const size_t grouped_bytes = static_cast<size_t>(ctx.jobs) * static_cast<size_t>(ctx.shard_rows) *
      static_cast<size_t>(ctx.cols) * ctx.elem_bytes;
  const auto total_start = std::chrono::steady_clock::now();
  const auto sender_start = std::chrono::steady_clock::now();
  {
    torch::NoGradGuard no_grad;
    for (int rank = 0; rank < ctx.world_size; ++rank) {
      for (int job = 0; job < ctx.jobs; ++job) {
        auto src_slice = ctx.source_jobs[static_cast<size_t>(job)].narrow(0, rank * ctx.shard_rows, ctx.shard_rows);
        auto grouped_slice =
            ctx.grouped_pack_rows[static_cast<size_t>(rank)].narrow(0, job * ctx.shard_rows, ctx.shard_rows);
        grouped_slice.copy_(src_slice);
      }
    }
  }
  TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(ctx.device_ids[ctx.root_rank]), "cudaSetDevice"));
  TC_RETURN_IF_ERROR(cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
  stats->sender_transform_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - sender_start).count();

  const auto nccl_start = std::chrono::steady_clock::now();
  TC_RETURN_IF_ERROR(ctx.clique->group_start());
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    auto* src_ptr = static_cast<uint8_t*>(ctx.grouped_pack_rows[static_cast<size_t>(rank)].data_ptr());
    auto* dst_ptr = static_cast<uint8_t*>(ctx.grouped_recv_rows[static_cast<size_t>(rank)].data_ptr());
    if (rank == ctx.root_rank) {
      TC_RETURN_IF_ERROR(cuda_status(
          cudaMemcpyAsync(dst_ptr, src_ptr, grouped_bytes, cudaMemcpyDeviceToDevice, ctx.clique->stream(ctx.root_rank)),
          "cudaMemcpyAsync(grouped local)"));
      continue;
    }
    TC_RETURN_IF_ERROR(ctx.clique->send_u8(ctx.root_rank, src_ptr, grouped_bytes, rank));
    TC_RETURN_IF_ERROR(ctx.clique->recv_u8(rank, dst_ptr, grouped_bytes, ctx.root_rank));
  }
  TC_RETURN_IF_ERROR(ctx.clique->group_end());
  TC_RETURN_IF_ERROR(ctx.clique->synchronize_all());
  stats->nccl_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - nccl_start).count();

  const auto receiver_start = std::chrono::steady_clock::now();
  {
    torch::NoGradGuard no_grad;
    for (int rank = 0; rank < ctx.world_size; ++rank) {
      for (int job = 0; job < ctx.jobs; ++job) {
        auto grouped_slice =
            ctx.grouped_recv_rows[static_cast<size_t>(rank)].narrow(0, job * ctx.shard_rows, ctx.shard_rows);
        ctx.dst_jobs[static_cast<size_t>(job)][static_cast<size_t>(rank)].copy_(grouped_slice);
      }
    }
  }
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    TC_RETURN_IF_ERROR(cuda_status(cudaSetDevice(ctx.device_ids[rank]), "cudaSetDevice"));
    TC_RETURN_IF_ERROR(cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
  }
  stats->receiver_transform_sec +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - receiver_start).count();
  stats->total_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  return absl::OkStatus();
}

absl::Status verify_dim0_outputs(const Dim0BenchContext& ctx) {
  for (int job = 0; job < ctx.jobs; ++job) {
    auto reference = ctx.source_jobs[static_cast<size_t>(job)].to(torch::kCPU);
    for (int rank = 0; rank < ctx.world_size; ++rank) {
      auto want = reference.narrow(0, rank * ctx.shard_rows, ctx.shard_rows).contiguous();
      auto got = ctx.dst_jobs[static_cast<size_t>(job)][static_cast<size_t>(rank)].to(torch::kCPU);
      if (!got.equal(want)) {
        return absl::InternalError(absl::StrCat("dim0 output mismatch on job=", job, " rank=", rank));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status run_mode_once(BenchContext& ctx, Mode mode, PhaseStats* stats) {
  switch (mode) {
    case Mode::kMemcpy2DPackSendrecv:
      return run_memcpy2d_sendrecv(ctx, stats);
    case Mode::kTorchNarrowCopySendrecv:
      return run_torch_narrow_copy_sendrecv(ctx, stats);
    case Mode::kTorchTransposeRoundtripSendrecv:
      return run_torch_transpose_roundtrip_sendrecv(ctx, stats);
    case Mode::kSourceWindowAllGatherScatter:
      return run_source_window_allgather(ctx, SourceWindowScatterMode::kMemcpy, stats);
    case Mode::kSourceWindowAllGatherScatterGraph:
      return run_source_window_allgather(ctx, SourceWindowScatterMode::kGraph, stats);
    case Mode::kSourceWindowAllGatherScatterGraphUpdate:
      return run_source_window_allgather(ctx, SourceWindowScatterMode::kGraphUpdate, stats);
    case Mode::kSourceWindowAllGatherScatterGraphExecUpdate:
      return run_source_window_allgather(ctx, SourceWindowScatterMode::kGraphExecUpdate, stats);
    case Mode::kSourceWindowAllGatherNoScatter:
      return run_source_window_allgather(ctx, SourceWindowScatterMode::kNone, stats);
    case Mode::kDim0DirectSendrecv:
    case Mode::kDim0TorchNarrowCopySendrecv:
    case Mode::kDim0GroupedSendrecv:
      break;
  }
  return absl::InvalidArgumentError("unknown mode");
}

absl::Status verify_outputs(const BenchContext& ctx, Mode mode) {
  auto reference = ctx.source.to(torch::kCPU);
  if (mode == Mode::kSourceWindowAllGatherNoScatter) {
    for (int rank = 0; rank < ctx.world_size; ++rank) {
      auto got = ctx.allgather_stage[static_cast<size_t>(rank)].to(torch::kCPU);
      if (!got.equal(reference)) {
        return absl::InternalError(absl::StrCat("all-gather output mismatch on rank ", rank));
      }
    }
    return absl::OkStatus();
  }
  for (int rank = 0; rank < ctx.world_size; ++rank) {
    auto want = reference.narrow(1, rank * ctx.shard_cols, ctx.shard_cols).contiguous();
    auto got = ctx.dst[static_cast<size_t>(rank)].to(torch::kCPU);
    if (!got.equal(want)) {
      return absl::InternalError(absl::StrCat("output mismatch on rank ", rank));
    }
  }
  return absl::OkStatus();
}

absl::Status run_mode(BenchContext& ctx, Mode mode, int warmup_iters, int iters, bool run_check) {
  if (mode == Mode::kSourceWindowAllGatherScatterGraph) {
    TC_RETURN_IF_ERROR(ensure_source_window_scatter_graphs(ctx, SourceWindowScatterGraphKind::kCapturedStatic));
  } else if (mode == Mode::kSourceWindowAllGatherScatterGraphUpdate) {
    TC_RETURN_IF_ERROR(ensure_source_window_scatter_graphs(ctx, SourceWindowScatterGraphKind::kMemcpyNodeTemplate));
  } else if (mode == Mode::kSourceWindowAllGatherScatterGraphExecUpdate) {
    TC_RETURN_IF_ERROR(ensure_source_window_scatter_graphs(ctx, SourceWindowScatterGraphKind::kMemcpyNode2DExecUpdate));
  }
  PhaseStats stats;
  for (int i = 0; i < warmup_iters; ++i) {
    TC_RETURN_IF_ERROR(zero_outputs(ctx));
    PhaseStats scratch;
    TC_RETURN_IF_ERROR(run_mode_once(ctx, mode, &scratch));
  }

  TC_RETURN_IF_ERROR(zero_outputs(ctx));
  if (run_check) {
    PhaseStats scratch;
    TC_RETURN_IF_ERROR(run_mode_once(ctx, mode, &scratch));
    TC_RETURN_IF_ERROR(verify_outputs(ctx, mode));
  }

  for (int i = 0; i < iters; ++i) {
    TC_RETURN_IF_ERROR(zero_outputs(ctx));
    TC_RETURN_IF_ERROR(run_mode_once(ctx, mode, &stats));
  }

  const double iters_d = static_cast<double>(iters);
  const double total_payload_gib =
      (static_cast<double>(ctx.rows) * ctx.cols * ctx.elem_bytes) / static_cast<double>(1ull << 30);
  const double total_avg = stats.total_sec / iters_d;
  const double delivery_gib_s = total_payload_gib / total_avg;
  std::cout << "{"
            << "\"mode\":\"" << mode_name(mode) << "\","
            << "\"rows\":" << ctx.rows << ","
            << "\"cols\":" << ctx.cols << ","
            << "\"world_size\":" << ctx.world_size << ","
            << "\"dtype_bytes\":" << ctx.elem_bytes << ","
            << "\"payload_gib\":" << total_payload_gib << ","
            << "\"sender_transform_avg_sec\":" << (stats.sender_transform_sec / iters_d) << ","
            << "\"nccl_avg_sec\":" << (stats.nccl_sec / iters_d) << ","
            << "\"receiver_transform_avg_sec\":" << (stats.receiver_transform_sec / iters_d) << ","
            << "\"graph_update_avg_sec\":" << (stats.graph_update_sec / iters_d) << ","
            << "\"total_avg_sec\":" << total_avg << ","
            << "\"delivery_gib_s\":" << delivery_gib_s;
  if (is_source_window_mode(mode)) {
    std::cout << ","
              << "\"source_window_stripe_gib\":" << (total_payload_gib / static_cast<double>(ctx.world_size)) << ","
              << "\"source_window_full_window_gib\":" << total_payload_gib << ","
              << "\"source_window_group_receiver_gib\":" << (total_payload_gib * static_cast<double>(ctx.world_size))
              << ","
              << "\"source_window_scatter_ops_per_rank\":"
              << std::max(1, absl::GetFlag(FLAGS_source_window_scatter_ops_per_rank));
    if (mode == Mode::kSourceWindowAllGatherScatterGraph || mode == Mode::kSourceWindowAllGatherScatterGraphUpdate) {
      std::cout << ","
                << "\"source_window_scatter_graph_build_sec\":" << ctx.source_window_scatter_graph_build_sec << ","
                << "\"source_window_scatter_graph_count\":" << ctx.source_window_scatter_graph_count << ","
                << "\"source_window_scatter_graph_node_count\":" << ctx.source_window_scatter_graph_node_count;
    }
  }
  std::cout << "}" << std::endl;
  return absl::OkStatus();
}

absl::Status run_dim0_mode_once(Dim0BenchContext& ctx, Mode mode, PhaseStats* stats) {
  switch (mode) {
    case Mode::kDim0DirectSendrecv:
      return run_dim0_direct_sendrecv(ctx, stats);
    case Mode::kDim0TorchNarrowCopySendrecv:
      return run_dim0_torch_narrow_copy_sendrecv(ctx, stats);
    case Mode::kDim0GroupedSendrecv:
      return run_dim0_grouped_sendrecv(ctx, stats);
    case Mode::kMemcpy2DPackSendrecv:
    case Mode::kTorchNarrowCopySendrecv:
    case Mode::kTorchTransposeRoundtripSendrecv:
    case Mode::kSourceWindowAllGatherScatter:
    case Mode::kSourceWindowAllGatherScatterGraph:
    case Mode::kSourceWindowAllGatherScatterGraphUpdate:
    case Mode::kSourceWindowAllGatherScatterGraphExecUpdate:
    case Mode::kSourceWindowAllGatherNoScatter:
      break;
  }
  return absl::InvalidArgumentError("unknown dim0 mode");
}

absl::Status run_dim0_mode(Dim0BenchContext& ctx, Mode mode, int warmup_iters, int iters, bool run_check) {
  PhaseStats stats;
  for (int i = 0; i < warmup_iters; ++i) {
    TC_RETURN_IF_ERROR(zero_outputs(ctx));
    PhaseStats scratch;
    TC_RETURN_IF_ERROR(run_dim0_mode_once(ctx, mode, &scratch));
  }

  TC_RETURN_IF_ERROR(zero_outputs(ctx));
  if (run_check) {
    PhaseStats scratch;
    TC_RETURN_IF_ERROR(run_dim0_mode_once(ctx, mode, &scratch));
    TC_RETURN_IF_ERROR(verify_dim0_outputs(ctx));
  }

  for (int i = 0; i < iters; ++i) {
    TC_RETURN_IF_ERROR(zero_outputs(ctx));
    TC_RETURN_IF_ERROR(run_dim0_mode_once(ctx, mode, &stats));
  }

  const double iters_d = static_cast<double>(iters);
  const double total_payload_gib =
      (static_cast<double>(ctx.jobs) * ctx.rows * ctx.cols * ctx.elem_bytes) / static_cast<double>(1ull << 30);
  const double total_avg = stats.total_sec / iters_d;
  const double delivery_gib_s = total_payload_gib / total_avg;
  std::cout << "{"
            << "\"mode\":\"" << mode_name(mode) << "\","
            << "\"rows\":" << ctx.rows << ","
            << "\"cols\":" << ctx.cols << ","
            << "\"jobs\":" << ctx.jobs << ","
            << "\"world_size\":" << ctx.world_size << ","
            << "\"dtype_bytes\":" << ctx.elem_bytes << ","
            << "\"payload_gib\":" << total_payload_gib << ","
            << "\"sender_transform_avg_sec\":" << (stats.sender_transform_sec / iters_d) << ","
            << "\"nccl_avg_sec\":" << (stats.nccl_sec / iters_d) << ","
            << "\"receiver_transform_avg_sec\":" << (stats.receiver_transform_sec / iters_d) << ","
            << "\"total_avg_sec\":" << total_avg << ","
            << "\"delivery_gib_s\":" << delivery_gib_s << "}" << std::endl;
  return absl::OkStatus();
}

absl::Status real_main() {
  const auto device_ids_or = parse_device_ids(absl::GetFlag(FLAGS_device_ids));
  if (!device_ids_or.ok()) {
    return device_ids_or.status();
  }
  const auto dtype_or = parse_dtype(absl::GetFlag(FLAGS_dtype));
  if (!dtype_or.ok()) {
    return dtype_or.status();
  }
  const int rows = absl::GetFlag(FLAGS_rows);
  const int cols = absl::GetFlag(FLAGS_cols);
  if (rows <= 0 || cols <= 0) {
    return absl::InvalidArgumentError("rows and cols must be > 0");
  }
  const int warmup_iters = absl::GetFlag(FLAGS_warmup_iters);
  const int iters = absl::GetFlag(FLAGS_iters);
  const int jobs = absl::GetFlag(FLAGS_jobs);
  if (warmup_iters < 0 || iters <= 0) {
    return absl::InvalidArgumentError("warmup_iters >= 0 and iters > 0 are required");
  }

  const std::string mode = absl::GetFlag(FLAGS_mode);
  if (mode == "all") {
    BenchContext ctx;
    TC_RETURN_IF_ERROR(init_context(ctx, *device_ids_or, *dtype_or, rows, cols));
    auto cleanup = std::unique_ptr<void, void (*)(void*)>{
        &ctx, +[](void* ptr) { destroy_context(*static_cast<BenchContext*>(ptr)); }};
    TC_RETURN_IF_ERROR(run_mode(ctx, Mode::kMemcpy2DPackSendrecv, warmup_iters, iters, absl::GetFlag(FLAGS_check)));
    TC_RETURN_IF_ERROR(run_mode(ctx, Mode::kTorchNarrowCopySendrecv, warmup_iters, iters, absl::GetFlag(FLAGS_check)));
    TC_RETURN_IF_ERROR(
        run_mode(ctx, Mode::kTorchTransposeRoundtripSendrecv, warmup_iters, iters, absl::GetFlag(FLAGS_check)));
    TC_RETURN_IF_ERROR(
        run_mode(ctx, Mode::kSourceWindowAllGatherNoScatter, warmup_iters, iters, absl::GetFlag(FLAGS_check)));
    TC_RETURN_IF_ERROR(
        run_mode(ctx, Mode::kSourceWindowAllGatherScatter, warmup_iters, iters, absl::GetFlag(FLAGS_check)));
    TC_RETURN_IF_ERROR(
        run_mode(ctx, Mode::kSourceWindowAllGatherScatterGraph, warmup_iters, iters, absl::GetFlag(FLAGS_check)));
    TC_RETURN_IF_ERROR(
        run_mode(ctx, Mode::kSourceWindowAllGatherScatterGraphUpdate, warmup_iters, iters, absl::GetFlag(FLAGS_check)));
    TC_RETURN_IF_ERROR(run_mode(
        ctx, Mode::kSourceWindowAllGatherScatterGraphExecUpdate, warmup_iters, iters, absl::GetFlag(FLAGS_check)));
    Dim0BenchContext dim0_ctx;
    TC_RETURN_IF_ERROR(init_dim0_context(dim0_ctx, *device_ids_or, *dtype_or, rows, cols, jobs));
    auto dim0_cleanup = std::unique_ptr<void, void (*)(void*)>{
        &dim0_ctx, +[](void* ptr) { destroy_dim0_context(*static_cast<Dim0BenchContext*>(ptr)); }};
    TC_RETURN_IF_ERROR(
        run_dim0_mode(dim0_ctx, Mode::kDim0DirectSendrecv, warmup_iters, iters, absl::GetFlag(FLAGS_check)));
    TC_RETURN_IF_ERROR(
        run_dim0_mode(dim0_ctx, Mode::kDim0TorchNarrowCopySendrecv, warmup_iters, iters, absl::GetFlag(FLAGS_check)));
    TC_RETURN_IF_ERROR(
        run_dim0_mode(dim0_ctx, Mode::kDim0GroupedSendrecv, warmup_iters, iters, absl::GetFlag(FLAGS_check)));
    return absl::OkStatus();
  }
  if (mode == "memcpy2d_sendrecv") {
    BenchContext ctx;
    TC_RETURN_IF_ERROR(init_context(ctx, *device_ids_or, *dtype_or, rows, cols));
    auto cleanup = std::unique_ptr<void, void (*)(void*)>{
        &ctx, +[](void* ptr) { destroy_context(*static_cast<BenchContext*>(ptr)); }};
    return run_mode(ctx, Mode::kMemcpy2DPackSendrecv, warmup_iters, iters, absl::GetFlag(FLAGS_check));
  }
  if (mode == "torch_narrow_copy_sendrecv") {
    BenchContext ctx;
    TC_RETURN_IF_ERROR(init_context(ctx, *device_ids_or, *dtype_or, rows, cols));
    auto cleanup = std::unique_ptr<void, void (*)(void*)>{
        &ctx, +[](void* ptr) { destroy_context(*static_cast<BenchContext*>(ptr)); }};
    return run_mode(ctx, Mode::kTorchNarrowCopySendrecv, warmup_iters, iters, absl::GetFlag(FLAGS_check));
  }
  if (mode == "torch_transpose_roundtrip_sendrecv") {
    BenchContext ctx;
    TC_RETURN_IF_ERROR(init_context(ctx, *device_ids_or, *dtype_or, rows, cols));
    auto cleanup = std::unique_ptr<void, void (*)(void*)>{
        &ctx, +[](void* ptr) { destroy_context(*static_cast<BenchContext*>(ptr)); }};
    return run_mode(ctx, Mode::kTorchTransposeRoundtripSendrecv, warmup_iters, iters, absl::GetFlag(FLAGS_check));
  }
  if (mode == "source_window_allgather_no_scatter" || mode == "source_window_allgather_scatter" ||
      mode == "source_window_allgather_scatter_graph" || mode == "source_window_allgather_scatter_graph_update" ||
      mode == "source_window_allgather_scatter_graph_exec_update") {
    BenchContext ctx;
    TC_RETURN_IF_ERROR(init_context(ctx, *device_ids_or, *dtype_or, rows, cols));
    auto cleanup = std::unique_ptr<void, void (*)(void*)>{
        &ctx, +[](void* ptr) { destroy_context(*static_cast<BenchContext*>(ptr)); }};
    if (mode == "source_window_allgather_no_scatter") {
      return run_mode(ctx, Mode::kSourceWindowAllGatherNoScatter, warmup_iters, iters, absl::GetFlag(FLAGS_check));
    }
    if (mode == "source_window_allgather_scatter_graph") {
      return run_mode(ctx, Mode::kSourceWindowAllGatherScatterGraph, warmup_iters, iters, absl::GetFlag(FLAGS_check));
    }
    if (mode == "source_window_allgather_scatter_graph_update") {
      return run_mode(
          ctx, Mode::kSourceWindowAllGatherScatterGraphUpdate, warmup_iters, iters, absl::GetFlag(FLAGS_check));
    }
    if (mode == "source_window_allgather_scatter_graph_exec_update") {
      return run_mode(
          ctx, Mode::kSourceWindowAllGatherScatterGraphExecUpdate, warmup_iters, iters, absl::GetFlag(FLAGS_check));
    }
    return run_mode(ctx, Mode::kSourceWindowAllGatherScatter, warmup_iters, iters, absl::GetFlag(FLAGS_check));
  }
  if (mode == "dim0_direct_sendrecv" || mode == "dim0_torch_narrow_copy_sendrecv" || mode == "dim0_grouped_sendrecv") {
    Dim0BenchContext dim0_ctx;
    TC_RETURN_IF_ERROR(init_dim0_context(dim0_ctx, *device_ids_or, *dtype_or, rows, cols, jobs));
    auto dim0_cleanup = std::unique_ptr<void, void (*)(void*)>{
        &dim0_ctx, +[](void* ptr) { destroy_dim0_context(*static_cast<Dim0BenchContext*>(ptr)); }};
    if (mode == "dim0_direct_sendrecv") {
      return run_dim0_mode(dim0_ctx, Mode::kDim0DirectSendrecv, warmup_iters, iters, absl::GetFlag(FLAGS_check));
    }
    if (mode == "dim0_torch_narrow_copy_sendrecv") {
      return run_dim0_mode(
          dim0_ctx, Mode::kDim0TorchNarrowCopySendrecv, warmup_iters, iters, absl::GetFlag(FLAGS_check));
    }
    return run_dim0_mode(dim0_ctx, Mode::kDim0GroupedSendrecv, warmup_iters, iters, absl::GetFlag(FLAGS_check));
  }
  return absl::InvalidArgumentError(absl::StrCat("unknown mode: ", mode));
}

} // namespace
} // namespace tensorcast::bench

int main(int argc, char** argv) {
  tensorcast::common::ensure_logging_initialized();
  absl::ParseCommandLine(argc, argv);
  const absl::Status st = tensorcast::bench::real_main();
  if (!st.ok()) {
    std::cerr << st << std::endl;
    return 1;
  }
  return 0;
}
