// Copyright (c) 2025, TensorCast Team.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/artifact_hash.h"
#include "core/common/cuda_api.h"
#include "core/store/loader/source_hash.h"
#include "gsl/gsl"

namespace tc = tensorcast;

namespace {

[[noreturn]] void fail_and_exit(absl::string_view message) {
  LOG(ERROR) << message;
  std::exit(EXIT_FAILURE);
}

constexpr uint64_t kDefaultLeafChunkBytes = tc::common::kGpuHashDefaultLeafChunkBytes;
constexpr uint64_t kMinLeafChunkBytes = 512ULL * 1024;
constexpr uint64_t kTargetLeafCount = 4096ULL;

size_t align_down(size_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const size_t remainder = value % alignment;
  return remainder == 0 ? value : value - remainder;
}

size_t determine_leaf_chunk_size(uint64_t total_size) {
  size_t chunk = align_down(kDefaultLeafChunkBytes, 64);
  if (chunk == 0) {
    chunk = 64;
  }

  auto leaf_count = [&](size_t chunk_bytes) -> uint64_t {
    return (total_size + static_cast<uint64_t>(chunk_bytes) - 1ULL) / static_cast<uint64_t>(chunk_bytes);
  };

  size_t adjusted = chunk;
  while (adjusted > kMinLeafChunkBytes && leaf_count(adjusted) < kTargetLeafCount) {
    size_t candidate = align_down(adjusted / 2, 64);
    if (candidate < kMinLeafChunkBytes) {
      break;
    }
    adjusted = candidate;
  }

  if (adjusted < kMinLeafChunkBytes) {
    adjusted = align_down(kMinLeafChunkBytes, 64);
  }

  return adjusted == 0 ? 64 : adjusted;
}

std::string format_bytes(uint64_t bytes) {
  static constexpr double kKB = 1024.0;
  static constexpr double kMB = 1024.0 * 1024.0;
  static constexpr double kGB = 1024.0 * 1024.0 * 1024.0;
  if (bytes >= static_cast<uint64_t>(kGB)) {
    return absl::StrCat(static_cast<double>(bytes) / kGB, " GiB");
  }
  if (bytes >= static_cast<uint64_t>(kMB)) {
    return absl::StrCat(static_cast<double>(bytes) / kMB, " MiB");
  }
  if (bytes >= static_cast<uint64_t>(kKB)) {
    return absl::StrCat(static_cast<double>(bytes) / kKB, " KiB");
  }
  return absl::StrCat(bytes, " B");
}

} // namespace

int main() {
  if (auto status = tc::cuda::set_device(0); !status.ok()) {
    fail_and_exit(absl::StrCat("set_device failed: ", status.ToString()));
  }

  constexpr uint64_t kWarmupBytes = 1ULL << 20;
  void* warmup_ptr = nullptr;
  if (auto status = tc::cuda::malloc(&warmup_ptr, kWarmupBytes); !status.ok()) {
    fail_and_exit(absl::StrCat("cuda::malloc warmup failed: ", status.ToString()));
  }
  auto warmup_cleanup = absl::Cleanup([&]() {
    if (warmup_ptr != nullptr) {
      auto status = tc::cuda::free(warmup_ptr);
      if (!status.ok()) {
        LOG(ERROR) << "cuda::free warmup cleanup failed: " << status;
      }
      warmup_ptr = nullptr;
    }
  });
  if (auto status = tc::cuda::memset(warmup_ptr, 0, kWarmupBytes); !status.ok()) {
    fail_and_exit(absl::StrCat("cuda::memset warmup failed: ", status.ToString()));
  }
  auto warmup_hash = tc::common::compute_data_multihash_from_gpu(warmup_ptr, kWarmupBytes, /*device_id=*/0);
  if (!warmup_hash.ok()) {
    fail_and_exit(absl::StrCat("warmup hash failed: ", warmup_hash.status().ToString()));
  }

  const std::vector<uint64_t> sizes_bytes = {
      64ULL << 10,
      512ULL << 10,
      1ULL << 20,
      8ULL << 20,
      32ULL << 20,
      128ULL << 20,
      512ULL << 20,
      1ULL << 30,
      2ULL << 30,
      4ULL << 30,
      8ULL << 30,
      16ULL << 30,
      32ULL << 30,
      48ULL << 30,
  };

  LOG(INFO)
      << "size_bytes,display_size,chunk_bytes,leaf_count,gpu_elapsed_ms,gpu_status,cpu_elapsed_ms,cpu_status,hash_match,gpu_gib_per_s,cpu_gib_per_s";

  for (uint64_t size : sizes_bytes) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    std::string gpu_status_msg = "skipped";
    std::string cpu_status_msg = "skipped";
    double gpu_elapsed_ms = 0.0;
    double cpu_elapsed_ms = 0.0;
    bool hash_match = false;
    std::string gpu_hash_value;
    std::string cpu_hash_value;
    const size_t chunk_bytes = determine_leaf_chunk_size(size);
    const uint64_t leaf_count = chunk_bytes > 0
        ? (size + static_cast<uint64_t>(chunk_bytes) - 1ULL) / static_cast<uint64_t>(chunk_bytes)
        : 0ULL;

    auto emit = [&]() {
      const double gpu_gib_per_s = (gpu_status_msg == "ok" && gpu_elapsed_ms > 0.0)
          ? (static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0)) / (gpu_elapsed_ms / 1000.0)
          : 0.0;
      const double cpu_gib_per_s = (cpu_status_msg == "ok" && cpu_elapsed_ms > 0.0)
          ? (static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0)) / (cpu_elapsed_ms / 1000.0)
          : 0.0;
      LOG(INFO) << size << ',' << format_bytes(size) << ',' << chunk_bytes << ',' << leaf_count << ',' << gpu_elapsed_ms
                << ',' << gpu_status_msg << ',' << cpu_elapsed_ms << ',' << cpu_status_msg << ','
                << (hash_match ? 1 : 0) << ',' << gpu_gib_per_s << ',' << cpu_gib_per_s;
    };

    if (auto status = tc::cuda::get_memory_info(&free_bytes, &total_bytes, /*device_id=*/0); !status.ok()) {
      gpu_status_msg = absl::StrCat("get_memory_info failed: ", status.ToString());
      emit();
      continue;
    }

    const uint64_t free_bytes_u64 = static_cast<uint64_t>(free_bytes);
    const uint64_t total_bytes_u64 = static_cast<uint64_t>(total_bytes);

    if (size > total_bytes_u64) {
      gpu_status_msg = "skipped: exceeds device total memory";
      emit();
      continue;
    }

    if (size > free_bytes_u64) {
      gpu_status_msg = absl::StrCat("skipped: insufficient free memory (free=", format_bytes(free_bytes_u64), ")");
      emit();
      continue;
    }

    void* device_ptr = nullptr;
    absl::Status alloc_status = tc::cuda::malloc(&device_ptr, static_cast<size_t>(size));
    if (!alloc_status.ok()) {
      gpu_status_msg = absl::StrCat("cuda::malloc failed: ", alloc_status.ToString());
      emit();
      continue;
    }

    auto device_cleanup = absl::Cleanup([&]() {
      if (device_ptr != nullptr) {
        auto status = tc::cuda::free(device_ptr);
        if (!status.ok()) {
          LOG(ERROR) << "cuda::free cleanup failed: " << status;
        }
        device_ptr = nullptr;
      }
    });

    if (auto memset_status = tc::cuda::memset(device_ptr, 0, static_cast<size_t>(size)); !memset_status.ok()) {
      gpu_status_msg = absl::StrCat("cuda::memset failed: ", memset_status.ToString());
      emit();
      continue;
    }

    const absl::Time start = absl::Now();
    auto gpu_hash_or =
        tc::common::compute_data_multihash_from_gpu(device_ptr, size, /*device_id=*/0, kDefaultLeafChunkBytes);
    const absl::Duration elapsed = absl::Now() - start;
    gpu_elapsed_ms = absl::ToDoubleMilliseconds(elapsed);
    if (!gpu_hash_or.ok()) {
      gpu_status_msg = absl::StrCat("hash failed: ", gpu_hash_or.status().ToString());
      emit();
      continue;
    }

    gpu_status_msg = "ok";
    gpu_hash_value = gpu_hash_or.value();

    uint8_t* host_ptr = static_cast<uint8_t*>(std::malloc(static_cast<size_t>(size)));
    if (host_ptr == nullptr) {
      cpu_status_msg = absl::StrCat("malloc failed for CPU buffer of size ", size, " bytes");
      emit();
      continue;
    }

    auto host_cleanup = absl::Cleanup([&]() { std::free(host_ptr); });

    std::memset(host_ptr, 0, static_cast<size_t>(size));

    const absl::Time cpu_start = absl::Now();
    auto cpu_hash_or = tensorcast::store::loader::compute_data_multihash_from_cpu_memory(
        gsl::make_not_null<const void*>(host_ptr), size, chunk_bytes);
    const absl::Duration cpu_elapsed = absl::Now() - cpu_start;
    cpu_elapsed_ms = absl::ToDoubleMilliseconds(cpu_elapsed);
    if (!cpu_hash_or.ok()) {
      cpu_status_msg = absl::StrCat("hash failed: ", cpu_hash_or.status().ToString());
      emit();
      continue;
    }

    cpu_status_msg = "ok";
    cpu_hash_value = cpu_hash_or.value();
    hash_match = (gpu_hash_value == cpu_hash_value);

    emit();
  }

  return EXIT_SUCCESS;
}
