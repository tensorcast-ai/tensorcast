// Copyright (c) 2026, TensorCast Team.

#include "core/store/replica/source_window_batched_scatter_kernel.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include "absl/base/no_destructor.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_driver_api.h"
#include "core/cuda/lazy_nvrtc.h"

namespace tensorcast::store::replica {
namespace {

constexpr char kKernelName[] = "tensorcast_source_window_batched_scatter_kernel";
constexpr char kKernelSource[] = R"NVRTC(
extern "C" {

typedef unsigned long long uint64_t;

struct TensorcastSourceWindowScatterDescriptor {
  uint64_t src_ptr;
  uint64_t dst_ptr;
  uint64_t row_bytes;
  uint64_t row_count;
  uint64_t source_stride_bytes;
  uint64_t target_stride_bytes;
};

__device__ __forceinline__ bool descriptor_is_u64_aligned(
    const TensorcastSourceWindowScatterDescriptor& desc) {
  return ((desc.src_ptr | desc.dst_ptr | desc.row_bytes |
           desc.source_stride_bytes | desc.target_stride_bytes) & 7ULL) == 0ULL;
}

__global__ void tensorcast_source_window_batched_scatter_kernel(
    const TensorcastSourceWindowScatterDescriptor* descs,
    unsigned int descriptor_count) {
  const unsigned int descriptor_index = blockIdx.x;
  if (descriptor_index >= descriptor_count) {
    return;
  }
  const TensorcastSourceWindowScatterDescriptor desc = descs[descriptor_index];
  if (desc.row_bytes == 0ULL || desc.row_count == 0ULL ||
      desc.src_ptr == 0ULL || desc.dst_ptr == 0ULL) {
    return;
  }

  const unsigned int block_y = blockIdx.y;
  const unsigned int blocks_y = gridDim.y;
  const unsigned int tid = threadIdx.x;
  const unsigned int stride_threads = blockDim.x * blocks_y;
  const unsigned int thread_base = tid + block_y * blockDim.x;
  const bool u64_path = descriptor_is_u64_aligned(desc);

  if (u64_path) {
    const uint64_t words_per_row = desc.row_bytes >> 3ULL;
    const uint64_t total_words = words_per_row * desc.row_count;
    unsigned long long* dst_base = reinterpret_cast<unsigned long long*>(desc.dst_ptr);
    const unsigned long long* src_base = reinterpret_cast<const unsigned long long*>(desc.src_ptr);
    const uint64_t src_stride_words = desc.source_stride_bytes >> 3ULL;
    const uint64_t dst_stride_words = desc.target_stride_bytes >> 3ULL;
    for (uint64_t word = thread_base; word < total_words; word += stride_threads) {
      const uint64_t row = words_per_row == 0ULL ? 0ULL : word / words_per_row;
      const uint64_t col = words_per_row == 0ULL ? 0ULL : word - row * words_per_row;
      dst_base[row * dst_stride_words + col] = src_base[row * src_stride_words + col];
    }
    return;
  }

  const uint64_t total_bytes = desc.row_bytes * desc.row_count;
  unsigned char* dst_base = reinterpret_cast<unsigned char*>(desc.dst_ptr);
  const unsigned char* src_base = reinterpret_cast<const unsigned char*>(desc.src_ptr);
  for (uint64_t byte = thread_base; byte < total_bytes; byte += stride_threads) {
    const uint64_t row = byte / desc.row_bytes;
    const uint64_t col = byte - row * desc.row_bytes;
    dst_base[row * desc.target_stride_bytes + col] = src_base[row * desc.source_stride_bytes + col];
  }
}

} // extern "C"
)NVRTC";

absl::Status cu_result_to_status(CUresult result, std::string_view context) {
  if (auto status = cuda::DriverApi::ensure_loaded(); !status.ok()) {
    return absl::InternalError(absl::StrCat(context, " - CUDA driver unavailable: ", status.message()));
  }
  return cuda::DriverApi::get().to_status(result, context);
}

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

std::string nvrtc_error_string(nvrtcResult result) {
  auto error_or = cuda::LazyNvrtc::get().nvrtcGetErrorString();
  if (!error_or.ok()) {
    return absl::StrCat("NVRTC error ", static_cast<int>(result));
  }
  const char* message = (*error_or)(result);
  return message != nullptr ? message : "NVRTC error (null)";
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

absl::StatusOr<CUcontext> current_cuda_context() {
  auto init_status = cuda::DriverApi::ensure_loaded();
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

struct NvrtcBatchedScatterKernel {
  CUmodule module = nullptr;
  CUfunction function = nullptr;

  ~NvrtcBatchedScatterKernel() {
    if (module == nullptr || !cuda::DriverApi::ensure_loaded().ok()) {
      return;
    }
    const cuda::DriverApi& driver = cuda::DriverApi::get();
    if (driver.cuModuleUnload != nullptr) {
      driver.cuModuleUnload(module);
    }
  }
};

struct KernelCacheKey {
  CUcontext context = nullptr;
  int device_id = -1;
  int major = 0;
  int minor = 0;

  bool operator==(const KernelCacheKey& other) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const KernelCacheKey& key) {
    return H::combine(std::move(h), reinterpret_cast<uintptr_t>(key.context), key.device_id, key.major, key.minor);
  }
};

absl::StatusOr<std::shared_ptr<NvrtcBatchedScatterKernel>> compile_kernel(int major, int minor) {
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
      (*create_program_or)(&program, kKernelSource, "tensorcast_source_window_batched_scatter.cu", 0, nullptr, nullptr);
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
  if (compile_result != NVRTC_SUCCESS) {
    std::string message = absl::StrCat(
        "nvrtcCompileProgram(source-window batched scatter) failed: ",
        nvrtc_error_string(compile_result),
        " (options: ",
        format_nvrtc_options(options_span),
        ")");
    if (!log.empty()) {
      absl::StrAppend(&message, "; NVRTC log:\n", log);
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

  auto init_status = cuda::DriverApi::ensure_loaded();
  if (!init_status.ok()) {
    return init_status;
  }
  const cuda::DriverApi& driver = cuda::DriverApi::get();

  CUmodule module = nullptr;
  CUfunction function = nullptr;
  auto module_status = cu_result_to_status(driver.cuModuleLoadData(&module, ptx.c_str()), "cuModuleLoadData");
  if (!module_status.ok()) {
    return module_status;
  }
  auto function_status = cu_result_to_status(
      driver.cuModuleGetFunction(&function, module, kKernelName),
      "cuModuleGetFunction(tensorcast_source_window_batched_scatter_kernel)");
  if (!function_status.ok()) {
    driver.cuModuleUnload(module);
    return function_status;
  }

  auto kernel = std::make_shared<NvrtcBatchedScatterKernel>();
  kernel->module = module;
  kernel->function = function;
  return kernel;
}

class KernelCache {
 public:
  absl::StatusOr<std::shared_ptr<NvrtcBatchedScatterKernel>> get_or_compile(const KernelCacheKey& key) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = cache_.find(key);
      if (it != cache_.end()) {
        return it->second;
      }
    }
    auto compiled_or = compile_kernel(key.major, key.minor);
    if (!compiled_or.ok()) {
      return compiled_or.status();
    }
    auto compiled = compiled_or.value();
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto [it, inserted] = cache_.emplace(key, compiled);
      if (!inserted) {
        return it->second;
      }
    }
    return compiled;
  }

  void invalidate(const KernelCacheKey& key) {
    std::lock_guard<std::mutex> lock(mu_);
    cache_.erase(key);
  }

 private:
  std::mutex mu_;
  absl::flat_hash_map<KernelCacheKey, std::shared_ptr<NvrtcBatchedScatterKernel>> cache_;
};

KernelCache& kernel_cache() {
  static absl::NoDestructor<KernelCache> cache;
  return *cache;
}

struct PreparedKernelCacheKey {
  CUcontext context = nullptr;
  int device_id = -1;

  bool operator==(const PreparedKernelCacheKey& other) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const PreparedKernelCacheKey& key) {
    return H::combine(std::move(h), reinterpret_cast<uintptr_t>(key.context), key.device_id);
  }
};

struct PreparedBatchedScatterKernel {
  std::shared_ptr<NvrtcBatchedScatterKernel> kernel;
  KernelCacheKey kernel_cache_key;
  uint64_t max_blocks_per_descriptor = 1;
};

absl::StatusOr<PreparedKernelCacheKey> build_prepared_kernel_cache_key(int device_id) {
  if (device_id < 0) {
    return absl::InvalidArgumentError("device_id must be >= 0");
  }
  auto context_or = current_cuda_context();
  if (!context_or.ok()) {
    return context_or.status();
  }
  return PreparedKernelCacheKey{
      .context = *context_or,
      .device_id = device_id,
  };
}

class PreparedKernelCache {
 public:
  absl::StatusOr<PreparedBatchedScatterKernel> get_or_prepare(const PreparedKernelCacheKey& key) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = cache_.find(key);
      if (it != cache_.end()) {
        return it->second;
      }
    }

    cudaDeviceProp props;
    auto prop_status = cuda::get_device_properties(key.device_id, &props);
    if (!prop_status.ok()) {
      return prop_status;
    }
    KernelCacheKey kernel_cache_key{
        .context = key.context,
        .device_id = key.device_id,
        .major = props.major,
        .minor = props.minor,
    };
    auto kernel_or = kernel_cache().get_or_compile(kernel_cache_key);
    if (!kernel_or.ok()) {
      return kernel_or.status();
    }
    PreparedBatchedScatterKernel prepared{
        .kernel = kernel_or.value(),
        .kernel_cache_key = kernel_cache_key,
        .max_blocks_per_descriptor =
            props.multiProcessorCount > 0 ? static_cast<uint64_t>(std::max(1, props.multiProcessorCount / 8)) : 1ULL,
    };
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto [it, inserted] = cache_.emplace(key, prepared);
      if (!inserted) {
        return it->second;
      }
    }
    return prepared;
  }

  void invalidate(const PreparedKernelCacheKey& key) {
    std::lock_guard<std::mutex> lock(mu_);
    cache_.erase(key);
  }

 private:
  std::mutex mu_;
  absl::flat_hash_map<PreparedKernelCacheKey, PreparedBatchedScatterKernel> cache_;
};

PreparedKernelCache& prepared_kernel_cache() {
  static absl::NoDestructor<PreparedKernelCache> cache;
  return *cache;
}

uint64_t descriptor_bytes(const SourceWindowBatchedScatterDescriptor& desc) {
  if (desc.row_bytes == 0 || desc.row_count == 0) {
    return 0;
  }
  if (desc.row_count > std::numeric_limits<uint64_t>::max() / desc.row_bytes) {
    return std::numeric_limits<uint64_t>::max();
  }
  return desc.row_count * desc.row_bytes;
}

} // namespace

absl::Status launch_source_window_batched_scatter(
    absl::Span<const SourceWindowBatchedScatterDescriptor> descriptors,
    void* device_descriptor_buffer,
    size_t device_descriptor_capacity_bytes,
    int device_id,
    cudaStream_t stream) {
  if (descriptors.empty()) {
    return absl::OkStatus();
  }
  if (cuda::is_fake()) {
    return absl::UnimplementedError("source-window batched scatter kernel unavailable with FakeCuda backend");
  }
  if (device_descriptor_buffer == nullptr) {
    return absl::InvalidArgumentError("source-window batched scatter descriptor buffer is null");
  }
  const size_t descriptor_bytes_total = descriptors.size() * sizeof(SourceWindowBatchedScatterDescriptor);
  if (descriptor_bytes_total > device_descriptor_capacity_bytes) {
    return absl::ResourceExhaustedError("source-window batched scatter descriptor buffer too small");
  }
  if (descriptors.size() > std::numeric_limits<unsigned int>::max()) {
    return absl::InvalidArgumentError("source-window batched scatter descriptor count exceeds uint32");
  }

  auto prepared_key_or = build_prepared_kernel_cache_key(device_id);
  if (!prepared_key_or.ok()) {
    return prepared_key_or.status();
  }
  const PreparedKernelCacheKey prepared_key = *prepared_key_or;
  auto prepared_or = prepared_kernel_cache().get_or_prepare(prepared_key);
  if (!prepared_or.ok()) {
    return prepared_or.status();
  }
  PreparedBatchedScatterKernel prepared = prepared_or.value();
  auto kernel = prepared.kernel;

  auto copy_status = cuda::memcpy_async(
      device_descriptor_buffer, descriptors.data(), descriptor_bytes_total, cudaMemcpyHostToDevice, stream);
  if (!copy_status.ok()) {
    return copy_status;
  }

  uint64_t max_bytes = 0;
  for (const auto& desc : descriptors) {
    max_bytes = std::max(max_bytes, descriptor_bytes(desc));
  }
  constexpr unsigned int kThreadsPerBlock = 256;
  const uint64_t bytes_per_block_wave = 64ULL * 1024ULL;
  uint64_t blocks_per_descriptor64 = (max_bytes + bytes_per_block_wave - 1ULL) / bytes_per_block_wave;
  blocks_per_descriptor64 = std::max<uint64_t>(1, blocks_per_descriptor64);
  blocks_per_descriptor64 =
      std::min<uint64_t>(blocks_per_descriptor64, std::min<uint64_t>(16, prepared.max_blocks_per_descriptor));
  const auto blocks_per_descriptor = static_cast<unsigned int>(blocks_per_descriptor64);

  CUdeviceptr descs_ptr = reinterpret_cast<CUdeviceptr>(device_descriptor_buffer);
  auto descriptor_count = static_cast<unsigned int>(descriptors.size());
  void* params[] = {&descs_ptr, &descriptor_count};
  const cuda::DriverApi& driver = cuda::DriverApi::get();
  CUstream launch_stream = reinterpret_cast<CUstream>(stream);
  CUresult launch_result = driver.cuLaunchKernel(
      kernel->function,
      descriptor_count,
      blocks_per_descriptor,
      1,
      kThreadsPerBlock,
      1,
      1,
      0,
      launch_stream,
      params,
      nullptr);
  if (launch_result == CUDA_ERROR_INVALID_HANDLE || launch_result == CUDA_ERROR_INVALID_CONTEXT) {
    kernel_cache().invalidate(prepared.kernel_cache_key);
    prepared_kernel_cache().invalidate(prepared_key);
    auto refreshed_or = prepared_kernel_cache().get_or_prepare(prepared_key);
    if (!refreshed_or.ok()) {
      return refreshed_or.status();
    }
    prepared = refreshed_or.value();
    kernel = prepared.kernel;
    launch_result = driver.cuLaunchKernel(
        kernel->function,
        descriptor_count,
        blocks_per_descriptor,
        1,
        kThreadsPerBlock,
        1,
        1,
        0,
        launch_stream,
        params,
        nullptr);
  }
  return cu_result_to_status(launch_result, "cuLaunchKernel(tensorcast_source_window_batched_scatter_kernel)");
}

absl::Status prewarm_source_window_batched_scatter_kernel_for_device(int device_id) {
  if (device_id < 0) {
    return absl::InvalidArgumentError("device_id must be >= 0");
  }
  if (cuda::is_fake()) {
    return absl::OkStatus();
  }
  auto set_status = cuda::set_device(device_id);
  if (!set_status.ok()) {
    return set_status;
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  auto memory_info_status = cuda::get_memory_info(&free_bytes, &total_bytes, device_id);
  if (!memory_info_status.ok()) {
    return memory_info_status;
  }
  auto prepared_key_or = build_prepared_kernel_cache_key(device_id);
  if (!prepared_key_or.ok()) {
    return prepared_key_or.status();
  }
  auto prepared_or = prepared_kernel_cache().get_or_prepare(*prepared_key_or);
  if (!prepared_or.ok()) {
    return prepared_or.status();
  }
  return absl::OkStatus();
}

absl::Status prewarm_source_window_batched_scatter_kernel_for_visible_devices() {
  if (cuda::is_fake()) {
    return absl::OkStatus();
  }
  int device_count = 0;
  auto count_status = cuda::get_device_count(&device_count);
  if (!count_status.ok()) {
    return count_status;
  }
  for (int device_id = 0; device_id < device_count; ++device_id) {
    auto status = prewarm_source_window_batched_scatter_kernel_for_device(device_id);
    if (!status.ok()) {
      return absl::Status(
          status.code(),
          absl::StrCat("source-window batched scatter prewarm failed for device ", device_id, ": ", status.message()));
    }
  }
  return absl::OkStatus();
}

} // namespace tensorcast::store::replica
