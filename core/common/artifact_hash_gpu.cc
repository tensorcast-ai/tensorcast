// Copyright (c) 2025-2026, TensorCast Team.

#include "core/common/artifact_hash.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <openssl/sha.h>
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash_internal.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_driver_api.h"
#include "core/cuda/lazy_nvrtc.h"

namespace tensorcast::common {

namespace cuda = ::tensorcast::cuda;

namespace {

using internal::compute_chunk_lengths;
using internal::compute_tree_hash_sha256;
using internal::determine_leaf_chunk_size;
using internal::kMaxChunkSize;
using internal::sha256_bytes;
using internal::to_multibase_multihash_sha256;

std::string format_cuda_driver_version(int version) {
  if (version <= 0) {
    return "unknown";
  }
  const int major = version / 1000;
  const int minor = (version % 1000) / 10;
  const int patch = version % 10;
  if (patch == 0) {
    return absl::StrCat(major, ".", minor);
  }
  return absl::StrCat(major, ".", minor, ".", patch);
}

void log_driver_nvrtc_mismatch_hint_if_needed(const absl::Status& status) {
  if (!absl::StrContains(status.message(), "CUDA_ERROR_UNSUPPORTED_PTX_VERSION")) {
    return;
  }

  int nvrtc_major = 0;
  int nvrtc_minor = 0;
  bool has_nvrtc_version = false;
  auto nvrtc_version_or = cuda::LazyNvrtc::get().nvrtcVersion();
  if (nvrtc_version_or.ok()) {
    const nvrtcResult nvrtc_version_result = (*nvrtc_version_or)(&nvrtc_major, &nvrtc_minor);
    has_nvrtc_version = nvrtc_version_result == NVRTC_SUCCESS;
  }

  int driver_version = 0;
  const cudaError_t driver_version_result = cudaDriverGetVersion(&driver_version);
  const bool has_driver_version = driver_version_result == cudaSuccess;

  std::string hint =
      "NVRTC generated PTX that the loaded NVIDIA driver rejected (unsupported PTX version). "
      "This usually means the NVRTC toolkit is newer than the installed driver.";

  if (has_nvrtc_version) {
    absl::StrAppend(&hint, " NVRTC version ", nvrtc_major, ".", nvrtc_minor, ".");
  }

  if (has_driver_version) {
    absl::StrAppend(&hint, " Driver reports version ", format_cuda_driver_version(driver_version), ".");
  }

  if (has_nvrtc_version && has_driver_version) {
    absl::StrAppend(
        &hint,
        " Update the driver to a release that supports CUDA ",
        nvrtc_major,
        ".",
        nvrtc_minor,
        " or align NVRTC with the installed driver.");
  }

  LOG(ERROR) << hint;
}

void release_device_buffer(void** ptr) {
  if (ptr == nullptr || *ptr == nullptr) {
    return;
  }
  absl::Status status = cuda::free(*ptr);
  if (!status.ok()) {
    LOG(ERROR) << "cuda::free failed while releasing buffer: " << status;
  }
  *ptr = nullptr;
}

absl::StatusOr<std::vector<uint8_t>> compute_root_via_host_copy(
    const uint8_t* device_base,
    uint64_t total_size,
    size_t chunk_size_bytes) {
  const auto lengths = compute_chunk_lengths(total_size, chunk_size_bytes);
  if (lengths.empty()) {
    return absl::InvalidArgumentError("No data available for hashing");
  }

  std::vector<std::vector<uint8_t>> leaves;
  leaves.reserve(lengths.size());
  std::vector<uint8_t> host_buffer(chunk_size_bytes);
  uint64_t offset = 0;

  for (uint64_t length : lengths) {
    const auto copy_bytes = static_cast<size_t>(length);
    if (copy_bytes == 0) {
      return absl::InvalidArgumentError("Encountered zero-length hash chunk");
    }

    auto copy_status = cuda::memcpy(host_buffer.data(), device_base + offset, copy_bytes, cudaMemcpyDeviceToHost);
    if (!copy_status.ok()) {
      return absl::InternalError(absl::StrCat("CUDA memcpy failed at offset ", offset, ": ", copy_status.ToString()));
    }

    if (auto sync_status = cuda::device_synchronize(); !sync_status.ok()) {
      return absl::InternalError(absl::StrCat("CUDA synchronize failed after memcpy: ", sync_status.ToString()));
    }

    auto hash_result = sha256_bytes(absl::Span<const uint8_t>(host_buffer.data(), copy_bytes));
    if (!hash_result.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to hash chunk at offset ", offset, ": ", hash_result.status().ToString()));
    }
    leaves.push_back(std::move(hash_result.value()));
    offset += copy_bytes;
  }

  auto root = compute_tree_hash_sha256(leaves);
  if (!root.ok()) {
    return root.status();
  }
  return root.value();
}

constexpr char kSha256KernelName[] = "tensorcast_sha256_leaf_kernel";

constexpr char kSha256KernelSource[] = R"NVRTC(
typedef unsigned long long uint64_t;
typedef unsigned int WORD;
typedef unsigned char BYTE;

__constant__ WORD TENSORCAST_SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

__device__ __forceinline__ WORD ROTRIGHT(WORD value, WORD bits) {
  return (value >> bits) | (value << (32 - bits));
}

__device__ __forceinline__ WORD CH(WORD x, WORD y, WORD z) {
  return (x & y) ^ (~x & z);
}

__device__ __forceinline__ WORD MAJ(WORD x, WORD y, WORD z) {
  return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ WORD EP0(WORD x) {
  return ROTRIGHT(x, 2) ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22);
}

__device__ __forceinline__ WORD EP1(WORD x) {
  return ROTRIGHT(x, 6) ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25);
}

__device__ __forceinline__ WORD SIG0(WORD x) {
  return ROTRIGHT(x, 7) ^ ROTRIGHT(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ WORD SIG1(WORD x) {
  return ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ (x >> 10);
}

struct TensorcastSha256Ctx {
  BYTE data[64];
  WORD datalen;
  uint64_t bitlen;
  WORD state[8];
};

__device__ __forceinline__ void tensorcast_sha256_transform(TensorcastSha256Ctx* ctx, const BYTE* data) {
  WORD m[64];
  for (int i = 0, j = 0; i < 16; ++i, j += 4) {
    m[i] = (static_cast<WORD>(data[j]) << 24) |
           (static_cast<WORD>(data[j + 1]) << 16) |
           (static_cast<WORD>(data[j + 2]) << 8) |
           static_cast<WORD>(data[j + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
  }

  WORD a = ctx->state[0];
  WORD b = ctx->state[1];
  WORD c = ctx->state[2];
  WORD d = ctx->state[3];
  WORD e = ctx->state[4];
  WORD f = ctx->state[5];
  WORD g = ctx->state[6];
  WORD h = ctx->state[7];

  for (int i = 0; i < 64; ++i) {
    WORD t1 = h + EP1(e) + CH(e, f, g) + TENSORCAST_SHA256_K[i] + m[i];
    WORD t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

__device__ __forceinline__ void tensorcast_sha256_init(TensorcastSha256Ctx* ctx) {
  ctx->datalen = 0;
  ctx->bitlen = 0;
  ctx->state[0] = 0x6a09e667;
  ctx->state[1] = 0xbb67ae85;
  ctx->state[2] = 0x3c6ef372;
  ctx->state[3] = 0xa54ff53a;
  ctx->state[4] = 0x510e527f;
  ctx->state[5] = 0x9b05688c;
  ctx->state[6] = 0x1f83d9ab;
  ctx->state[7] = 0x5be0cd19;
}

__device__ __forceinline__ void tensorcast_sha256_update(
    TensorcastSha256Ctx* ctx,
    const BYTE* data,
    uint64_t len) {
  if (len == 0ULL) {
    return;
  }

  uint64_t offset = 0ULL;

  if (ctx->datalen > 0) {
    const uint64_t available = 64ULL - static_cast<uint64_t>(ctx->datalen);
    const uint64_t to_copy = len < available ? len : available;
    for (uint64_t i = 0; i < to_copy; ++i) {
      ctx->data[ctx->datalen + i] = data[i];
    }
    const uint64_t new_length = static_cast<uint64_t>(ctx->datalen) + to_copy;
    ctx->datalen = static_cast<WORD>(new_length);
    offset += to_copy;
    if (ctx->datalen == 64) {
      tensorcast_sha256_transform(ctx, ctx->data);
      ctx->bitlen += 512ULL;
      ctx->datalen = 0;
    }
  }

  while (offset + 64ULL <= len) {
    tensorcast_sha256_transform(ctx, data + offset);
    ctx->bitlen += 512ULL;
    offset += 64ULL;
  }

  const uint64_t remaining = len - offset;
  if (remaining == 0ULL) {
    return;
  }

  for (uint64_t i = 0; i < remaining; ++i) {
    ctx->data[ctx->datalen + i] = data[offset + i];
  }
  const uint64_t tail_length = static_cast<uint64_t>(ctx->datalen) + remaining;
  ctx->datalen = static_cast<WORD>(tail_length);
}

__device__ __forceinline__ void tensorcast_sha256_final(TensorcastSha256Ctx* ctx, BYTE* hash) {
  WORD i = ctx->datalen;
  ctx->data[i++] = 0x80;
  if (ctx->datalen < 56) {
    while (i < 56) {
      ctx->data[i++] = 0x00;
    }
  } else {
    while (i < 64) {
      ctx->data[i++] = 0x00;
    }
    tensorcast_sha256_transform(ctx, ctx->data);
    for (int j = 0; j < 56; ++j) {
      ctx->data[j] = 0x00;
    }
  }

  ctx->bitlen += static_cast<uint64_t>(ctx->datalen) * 8ULL;
  ctx->data[63] = static_cast<BYTE>(ctx->bitlen);
  ctx->data[62] = static_cast<BYTE>(ctx->bitlen >> 8U);
  ctx->data[61] = static_cast<BYTE>(ctx->bitlen >> 16U);
  ctx->data[60] = static_cast<BYTE>(ctx->bitlen >> 24U);
  ctx->data[59] = static_cast<BYTE>(ctx->bitlen >> 32U);
  ctx->data[58] = static_cast<BYTE>(ctx->bitlen >> 40U);
  ctx->data[57] = static_cast<BYTE>(ctx->bitlen >> 48U);
  ctx->data[56] = static_cast<BYTE>(ctx->bitlen >> 56U);
  tensorcast_sha256_transform(ctx, ctx->data);

  for (int j = 0; j < 4; ++j) {
    hash[j] = (ctx->state[0] >> (24 - j * 8)) & 0xff;
    hash[j + 4] = (ctx->state[1] >> (24 - j * 8)) & 0xff;
    hash[j + 8] = (ctx->state[2] >> (24 - j * 8)) & 0xff;
    hash[j + 12] = (ctx->state[3] >> (24 - j * 8)) & 0xff;
    hash[j + 16] = (ctx->state[4] >> (24 - j * 8)) & 0xff;
    hash[j + 20] = (ctx->state[5] >> (24 - j * 8)) & 0xff;
    hash[j + 24] = (ctx->state[6] >> (24 - j * 8)) & 0xff;
    hash[j + 28] = (ctx->state[7] >> (24 - j * 8)) & 0xff;
  }
}

extern "C" __global__ void tensorcast_sha256_leaf_kernel(
    const BYTE* base,
    uint64_t chunk_stride,
    uint64_t total_size,
    BYTE* digests,
    unsigned int leaf_count) {
  for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < leaf_count;
       idx += blockDim.x * gridDim.x) {
    const uint64_t offset = static_cast<uint64_t>(idx) * chunk_stride;
    if (offset >= total_size) {
      continue;
    }

    uint64_t len = total_size - offset;
    if (len > chunk_stride) {
      len = chunk_stride;
    }

    const BYTE* chunk = base + offset;
    BYTE* out = digests + static_cast<uint64_t>(idx) * 32ULL;

    TensorcastSha256Ctx ctx;
    tensorcast_sha256_init(&ctx);
    tensorcast_sha256_update(&ctx, chunk, len);
    tensorcast_sha256_final(&ctx, out);
  }
}
)NVRTC";

absl::Status cu_result_to_status(CUresult result, std::string_view context) {
  if (auto status = cuda::DriverApi::ensure_loaded(); !status.ok()) {
    return absl::InternalError(absl::StrCat(context, " - CUDA driver unavailable: ", status.message()));
  }
  return cuda::DriverApi::get().to_status(result, context);
}

absl::Status ensure_cuda_driver_initialized() {
  return cuda::DriverApi::ensure_loaded();
}

struct NvrtcSha256Kernel {
  CUmodule module = nullptr;
  CUfunction function = nullptr;

  ~NvrtcSha256Kernel() {
    if (module != nullptr) {
      if (cuda::DriverApi::ensure_loaded().ok()) {
        const cuda::DriverApi& driver = cuda::DriverApi::get();
        if (driver.cuModuleUnload != nullptr) {
          driver.cuModuleUnload(module);
        }
      }
    }
  }
};

std::string read_nvrtc_log(nvrtcProgram program) {
  auto log_size_or = cuda::LazyNvrtc::get().nvrtcGetProgramLogSize();
  if (!log_size_or.ok()) {
    return {};
  }
  auto log_or = cuda::LazyNvrtc::get().nvrtcGetProgramLog();
  if (!log_or.ok()) {
    return {};
  }
  size_t log_size = 0;
  if ((*log_size_or)(program, &log_size) != NVRTC_SUCCESS || log_size <= 1) {
    return {};
  }
  std::string log(log_size, '\0');
  if ((*log_or)(program, log.data()) != NVRTC_SUCCESS) {
    return {};
  }
  while (!log.empty() && (log.back() == '\0' || log.back() == '\n')) {
    log.pop_back();
  }
  return log;
}

std::string format_nvrtc_options(absl::Span<const char* const> options) {
  std::string formatted;
  for (const char* option : options) {
    if (!formatted.empty()) {
      absl::StrAppend(&formatted, ", ");
    }
    absl::StrAppend(&formatted, "'", option, "'");
  }
  return formatted;
}

std::string nvrtc_error_string(nvrtcResult result) {
  auto error_or = cuda::LazyNvrtc::get().nvrtcGetErrorString();
  if (!error_or.ok()) {
    return absl::StrCat("NVRTC error ", static_cast<int>(result));
  }
  const char* message = (*error_or)(result);
  return message != nullptr ? message : "NVRTC error (null)";
}

absl::StatusOr<std::shared_ptr<NvrtcSha256Kernel>> compile_nvrtc_sha256_kernel(int major, int minor) {
  auto create_program_or = cuda::LazyNvrtc::get().nvrtcCreateProgram();
  if (!create_program_or.ok()) {
    return create_program_or.status();
  }
  auto destroy_program_or = cuda::LazyNvrtc::get().nvrtcDestroyProgram();
  if (!destroy_program_or.ok()) {
    return destroy_program_or.status();
  }
  auto compile_program_or = cuda::LazyNvrtc::get().nvrtcCompileProgram();
  if (!compile_program_or.ok()) {
    return compile_program_or.status();
  }
  auto get_ptx_size_or = cuda::LazyNvrtc::get().nvrtcGetPTXSize();
  if (!get_ptx_size_or.ok()) {
    return get_ptx_size_or.status();
  }
  auto get_ptx_or = cuda::LazyNvrtc::get().nvrtcGetPTX();
  if (!get_ptx_or.ok()) {
    return get_ptx_or.status();
  }

  nvrtcProgram program;
  nvrtcResult create_result =
      (*create_program_or)(&program, kSha256KernelSource, "tensorcast_sha256.cu", 0, nullptr, nullptr);
  if (create_result != NVRTC_SUCCESS) {
    return absl::InternalError(absl::StrCat("nvrtcCreateProgram failed: ", nvrtc_error_string(create_result)));
  }
  absl::Cleanup destroy_program = absl::MakeCleanup([&]() { (*destroy_program_or)(&program); });

  std::string arch_option = absl::StrCat("--gpu-architecture=compute_", major, minor);
  std::array<const char*, 2> options{"--std=c++17", arch_option.c_str()};
  absl::Span<const char* const> options_span(options.data(), options.size());
  nvrtcResult compile_result =
      (*compile_program_or)(program, static_cast<int>(options_span.size()), options_span.data());

  const std::string log = read_nvrtc_log(program);
  if (!log.empty()) {
    VLOG(1) << "NVRTC SHA256 kernel log:\n" << log;
  }

  if (compile_result != NVRTC_SUCCESS) {
    std::string message =
        absl::StrCat("nvrtcCompileProgram failed: ", nvrtc_error_string(compile_result), " (options: ");
    absl::StrAppend(&message, format_nvrtc_options(options_span), ")");
    if (!log.empty()) {
      absl::StrAppend(&message, "; NVRTC log:\n", log);
    } else {
      absl::StrAppend(&message, "; NVRTC log empty or unavailable");
    }
    return absl::InternalError(message);
  }

  size_t ptx_size = 0;
  nvrtcResult ptx_size_result = (*get_ptx_size_or)(program, &ptx_size);
  if (ptx_size_result != NVRTC_SUCCESS) {
    return absl::InternalError(absl::StrCat("nvrtcGetPTXSize failed: ", nvrtc_error_string(ptx_size_result)));
  }
  std::string ptx(ptx_size, '\0');
  nvrtcResult ptx_result = (*get_ptx_or)(program, ptx.data());
  if (ptx_result != NVRTC_SUCCESS) {
    return absl::InternalError(absl::StrCat("nvrtcGetPTX failed: ", nvrtc_error_string(ptx_result)));
  }

  auto init_status = ensure_cuda_driver_initialized();
  if (!init_status.ok()) {
    return init_status;
  }
  const cuda::DriverApi& driver = cuda::DriverApi::get();

  CUmodule module = nullptr;
  CUfunction function = nullptr;
  auto module_status = cu_result_to_status(driver.cuModuleLoadData(&module, ptx.c_str()), "cuModuleLoadData");
  if (!module_status.ok()) {
    log_driver_nvrtc_mismatch_hint_if_needed(module_status);
    return module_status;
  }

  auto function_status =
      cu_result_to_status(driver.cuModuleGetFunction(&function, module, kSha256KernelName), "cuModuleGetFunction");
  if (!function_status.ok()) {
    driver.cuModuleUnload(module);
    log_driver_nvrtc_mismatch_hint_if_needed(function_status);
    return function_status;
  }

  auto kernel = std::make_shared<NvrtcSha256Kernel>();
  kernel->module = module;
  kernel->function = function;
  return kernel;
}

class NvrtcSha256KernelCache {
 public:
  struct CacheKey {
    CUcontext context = nullptr;
    int device_id = -1;
    int major = 0;
    int minor = 0;

    bool operator==(const CacheKey& other) const = default;

    template <typename H>
    friend H AbslHashValue(H h, const CacheKey& key) {
      return H::combine(std::move(h), reinterpret_cast<uintptr_t>(key.context), key.device_id, key.major, key.minor);
    }
  };

  absl::StatusOr<std::shared_ptr<NvrtcSha256Kernel>> get_or_compile(const CacheKey& key) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = cache_.find(key);
      if (it != cache_.end()) {
        return it->second;
      }
    }

    auto compiled_or = compile_nvrtc_sha256_kernel(key.major, key.minor);
    if (!compiled_or.ok()) {
      return compiled_or.status();
    }
    auto compiled = std::move(compiled_or.value());

    {
      std::lock_guard<std::mutex> lock(mu_);
      auto [it, inserted] = cache_.emplace(key, compiled);
      if (!inserted) {
        return it->second;
      }
    }

    return compiled;
  }

  void invalidate(const CacheKey& key) {
    std::lock_guard<std::mutex> lock(mu_);
    cache_.erase(key);
  }

 private:
  std::mutex mu_;
  absl::flat_hash_map<CacheKey, std::shared_ptr<NvrtcSha256Kernel>> cache_;
};

NvrtcSha256KernelCache& get_nvrtc_kernel_cache() {
  static auto* cache = new NvrtcSha256KernelCache();
  return *cache;
}

absl::StatusOr<CUcontext> current_cuda_context() {
  auto init_status = ensure_cuda_driver_initialized();
  if (!init_status.ok()) {
    return init_status;
  }
  const cuda::DriverApi& driver = cuda::DriverApi::get();
  CUcontext context = nullptr;
  auto ctx_status = cu_result_to_status(driver.cuCtxGetCurrent(&context), "cuCtxGetCurrent");
  if (!ctx_status.ok()) {
    return ctx_status;
  }
  if (context == nullptr) {
    return absl::FailedPreconditionError("CUDA context is null after set_device");
  }
  return context;
}

absl::StatusOr<std::vector<uint8_t>> compute_root_via_nvrtc(
    const uint8_t* device_base,
    uint64_t total_size,
    size_t chunk_size_bytes,
    int device_id) {
  if (cuda::is_fake()) {
    return absl::UnimplementedError("NVRTC hashing unavailable when FakeCuda backend is active");
  }

  if (chunk_size_bytes == 0) {
    return absl::InvalidArgumentError("chunk_size_bytes must be > 0");
  }

  cudaDeviceProp props;
  if (auto prop_status = cuda::get_device_properties(device_id, &props); !prop_status.ok()) {
    return prop_status;
  }

  auto context_or = current_cuda_context();
  if (!context_or.ok()) {
    return context_or.status();
  }
  NvrtcSha256KernelCache::CacheKey cache_key{
      .context = *context_or,
      .device_id = device_id,
      .major = props.major,
      .minor = props.minor,
  };

  auto kernel_or = get_nvrtc_kernel_cache().get_or_compile(cache_key);
  if (!kernel_or.ok()) {
    return kernel_or.status();
  }
  auto kernel = kernel_or.value();
  const auto sm_count = static_cast<unsigned int>(props.multiProcessorCount);

  auto ensure_driver_status = ensure_cuda_driver_initialized();
  if (!ensure_driver_status.ok()) {
    return ensure_driver_status;
  }
  const cuda::DriverApi& driver = cuda::DriverApi::get();

  const uint64_t leaf_count64 =
      (total_size + static_cast<uint64_t>(chunk_size_bytes) - 1ULL) / static_cast<uint64_t>(chunk_size_bytes);
  if (leaf_count64 == 0) {
    return absl::InvalidArgumentError("No leaves to hash");
  }
  if (leaf_count64 > std::numeric_limits<unsigned int>::max()) {
    return absl::InvalidArgumentError("Leaf count exceeds supported range");
  }
  auto leaf_count = static_cast<unsigned int>(leaf_count64);
  const size_t digest_bytes = static_cast<size_t>(leaf_count) * SHA256_DIGEST_LENGTH;

  void* device_digests = nullptr;
  if (auto status = cuda::malloc(&device_digests, digest_bytes); !status.ok()) {
    return status;
  }
  absl::Cleanup free_digests = absl::MakeCleanup([&]() { release_device_buffer(&device_digests); });

  cudaStream_t stream = nullptr;
  bool stream_created = false;
  auto stream_status = cuda::stream_create_with_flags(&stream, cudaStreamNonBlocking);
  if (stream_status.ok()) {
    stream_created = true;
  } else {
    VLOG(1) << "Failed to create CUDA stream for hashing, falling back to default stream: " << stream_status;
    stream = nullptr;
  }
  absl::Cleanup destroy_stream = absl::MakeCleanup([&]() {
    if (stream_created) {
      auto destroy_status = cuda::stream_destroy(stream);
      if (!destroy_status.ok()) {
        LOG(ERROR) << "cuda::stream_destroy failed: " << destroy_status;
      }
      stream_created = false;
      stream = nullptr;
    }
  });

  auto data_ptr = reinterpret_cast<CUdeviceptr>(device_base);
  auto digests_ptr = reinterpret_cast<CUdeviceptr>(device_digests);
  auto stride = static_cast<uint64_t>(chunk_size_bytes); // NOLINT(google-runtime-int)
  auto total_size_param = static_cast<uint64_t>(total_size); // NOLINT(google-runtime-int)

  void* params[] = {&data_ptr, &stride, &total_size_param, &digests_ptr, &leaf_count};

  constexpr unsigned int kThreadsPerBlock = 512;
  const unsigned int blocks = (leaf_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const unsigned int max_grid = sm_count > 0 ? sm_count * 8U : 1U;
  const unsigned int launch_blocks = blocks == 0 ? 1U : std::min(blocks, max_grid);

  CUstream launch_stream = stream_created ? reinterpret_cast<CUstream>(stream) : nullptr;

  auto launch_result = driver.cuLaunchKernel(
      kernel->function, launch_blocks, 1, 1, kThreadsPerBlock, 1, 1, 0, launch_stream, params, nullptr);
  if (launch_result == CUDA_ERROR_INVALID_HANDLE || launch_result == CUDA_ERROR_INVALID_CONTEXT) {
    LOG(WARNING) << "NVRTC kernel launch returned invalid handle/context; refreshing cache for device=" << device_id;
    get_nvrtc_kernel_cache().invalidate(cache_key);
    auto refreshed_or = get_nvrtc_kernel_cache().get_or_compile(cache_key);
    if (!refreshed_or.ok()) {
      return refreshed_or.status();
    }
    kernel = refreshed_or.value();
    launch_result = driver.cuLaunchKernel(
        kernel->function, launch_blocks, 1, 1, kThreadsPerBlock, 1, 1, 0, launch_stream, params, nullptr);
  }
  auto launch_status = cu_result_to_status(launch_result, "cuLaunchKernel(tensorcast_sha256_leaf_kernel)");
  if (!launch_status.ok()) {
    return launch_status;
  }

  std::vector<uint8_t> host_digests(digest_bytes);
  absl::Status copy_status;
  if (stream_created) {
    copy_status = cuda::memcpy_async(host_digests.data(), device_digests, digest_bytes, cudaMemcpyDeviceToHost, stream);
  } else {
    copy_status = cuda::memcpy(host_digests.data(), device_digests, digest_bytes, cudaMemcpyDeviceToHost);
  }
  if (!copy_status.ok()) {
    return copy_status;
  }

  absl::Status sync_status = stream_created ? cuda::stream_synchronize(stream) : cuda::device_synchronize();
  if (!sync_status.ok()) {
    return sync_status;
  }

  std::vector<std::vector<uint8_t>> leaves;
  leaves.reserve(leaf_count);
  for (unsigned int i = 0; i < leaf_count; ++i) {
    const uint8_t* begin = host_digests.data() + static_cast<size_t>(i) * SHA256_DIGEST_LENGTH;
    leaves.emplace_back(begin, begin + SHA256_DIGEST_LENGTH);
  }

  auto root = compute_tree_hash_sha256(leaves);
  if (!root.ok()) {
    return root.status();
  }
  return root.value();
}

} // namespace

absl::StatusOr<std::string> compute_data_multihash_from_gpu(
    void* gpu_ptr,
    uint64_t total_size,
    int device_id,
    size_t leaf_chunk_bytes) {
  if (gpu_ptr == nullptr) {
    return absl::InvalidArgumentError("GPU pointer is null");
  }

  if (total_size == 0) {
    return absl::InvalidArgumentError("Cannot hash empty GPU buffer");
  }

  if (total_size > 1ULL << 40) {
    return absl::InvalidArgumentError(absl::StrCat("GPU buffer too large for hashing: ", total_size, " bytes"));
  }

  if (leaf_chunk_bytes == 0) {
    return absl::InvalidArgumentError("leaf_chunk_bytes must be > 0");
  }

  if (leaf_chunk_bytes > kMaxChunkSize) {
    return absl::InvalidArgumentError(
        absl::StrCat("Chunk size too large: ", leaf_chunk_bytes, " (max: ", kMaxChunkSize, ")"));
  }

  if (auto st = cuda::set_device(device_id); !st.ok()) {
    return absl::InternalError(absl::StrCat("Failed to set CUDA device ", device_id, ": ", st.ToString()));
  }

  auto* device_base = static_cast<uint8_t*>(gpu_ptr);
  const size_t chunk_size = determine_leaf_chunk_size(total_size, leaf_chunk_bytes);

  absl::StatusOr<std::vector<uint8_t>> root_or;

  if (!cuda::is_fake()) {
    root_or = compute_root_via_nvrtc(device_base, total_size, chunk_size, device_id);
    if (!root_or.ok()) {
      LOG(WARNING) << "NVRTC hashing unavailable; falling back to CPU hashing: " << root_or.status();
      root_or = compute_root_via_host_copy(device_base, total_size, chunk_size);
    }
  } else {
    root_or = compute_root_via_host_copy(device_base, total_size, chunk_size);
  }

  if (!root_or.ok()) {
    return root_or.status();
  }

  std::string multihash = to_multibase_multihash_sha256(root_or.value());
  if (multihash.empty()) {
    return absl::InternalError("Failed to generate multihash");
  }

  return multihash;
}

} // namespace tensorcast::common
