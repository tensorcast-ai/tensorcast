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
    "dim0_torch_narrow_copy_sendrecv|dim0_grouped_sendrecv");
ABSL_FLAG(std::string, device_ids, "0,1,2,3,4,5,6,7", "Comma-separated CUDA device ids");
ABSL_FLAG(int, rows, 4096, "Rows per chunk tensor");
ABSL_FLAG(int, cols, 16384, "Cols per chunk tensor");
ABSL_FLAG(int, jobs, 1, "Number of synthetic per-tensor jobs for dim0 modes");
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
  }
  return "unknown";
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
  double total_sec{0.0};
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
  std::vector<torch::Tensor> pack;
  torch::Tensor transposed;
  std::vector<torch::Tensor> recv_transposed;
  cudaStream_t pack_stream{nullptr};
  std::unique_ptr<NcclClique> clique;
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

  auto clique_or = NcclClique::create(device_ids);
  if (!clique_or.ok()) {
    return clique_or.status();
  }
  ctx.clique = std::make_unique<NcclClique>(std::move(*clique_or));

  ctx.source = make_cuda_tensor(device_ids[ctx.root_rank], dtype, {ctx.rows, ctx.cols});
  ctx.transposed = make_cuda_tensor(device_ids[ctx.root_rank], dtype, {ctx.cols, ctx.rows});
  ctx.dst.reserve(device_ids.size());
  ctx.pack.reserve(device_ids.size());
  ctx.recv_transposed.reserve(device_ids.size());
  for (size_t i = 0; i < device_ids.size(); ++i) {
    ctx.dst.push_back(make_cuda_tensor(device_ids[i], dtype, {ctx.rows, ctx.shard_cols}));
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
  for (auto& recv : ctx.recv_transposed) {
    recv.zero_();
  }
  if (ctx.transposed.defined()) {
    ctx.transposed.zero_();
  }
  for (auto& pack : ctx.pack) {
    if (pack.defined()) {
      pack.zero_();
    }
  }
  TC_RETURN_IF_ERROR(cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
  return absl::OkStatus();
}

void destroy_context(BenchContext& ctx) {
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
  for (auto& recv : ctx.recv_transposed) {
    recv.zero_();
  }
  if (ctx.transposed.defined()) {
    ctx.transposed.zero_();
  }
  for (auto& pack : ctx.pack) {
    if (pack.defined()) {
      pack.zero_();
    }
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
    case Mode::kDim0DirectSendrecv:
    case Mode::kDim0TorchNarrowCopySendrecv:
    case Mode::kDim0GroupedSendrecv:
      break;
  }
  return absl::InvalidArgumentError("unknown mode");
}

absl::Status verify_outputs(const BenchContext& ctx) {
  auto reference = ctx.source.to(torch::kCPU);
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
    TC_RETURN_IF_ERROR(verify_outputs(ctx));
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
            << "\"total_avg_sec\":" << total_avg << ","
            << "\"delivery_gib_s\":" << delivery_gib_s << "}" << std::endl;
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
