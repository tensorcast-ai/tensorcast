// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <string>
#include <string_view>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "core/common/cuda_api.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/misc/ibv_wrap.h"
#include "core/communicator/transport/rdma_context.h"
#include "core/testing/test_helpers.h"

namespace {

constexpr int kGpuId = 0;
constexpr int kMrAccessFlags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;

constexpr std::string_view kEnableEnv = "TENSORCAST_RUN_RDMA_VMM_UB_TEST";

bool env_enabled(std::string_view name) {
  const std::string key(name);
  const char* value = std::getenv(key.c_str());
  if (value == nullptr) {
    return false;
  }
  std::string_view sv(value);
  return sv == "1" || sv == "true" || sv == "TRUE" || sv == "yes" || sv == "YES";
}

void log_if_error(absl::string_view operation, const absl::Status& status) {
  if (!status.ok()) {
    LOG(WARNING) << operation << " failed during cleanup: " << status;
  }
}

void log_if_ibv_error(absl::string_view operation, tensorcast::communicator::misc::result_t result) {
  if (result != tensorcast::communicator::misc::SUCCESS) {
    LOG(WARNING) << operation << " failed during cleanup";
  }
}

absl::StatusOr<size_t> get_vmm_granularity(int device_id) {
  absl::Status status = tensorcast::cuda::cu_init(0);
  if (!status.ok()) {
    return status;
  }

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t granularity = 0;
  status = tensorcast::cuda::cu_mem_get_allocation_granularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (!status.ok()) {
    return status;
  }
  if (granularity == 0) {
    return absl::InternalError("cuMemGetAllocationGranularity returned 0");
  }
  return granularity;
}

CUmemAllocationProp make_vmm_prop(int device_id) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  return prop;
}

absl::Status map_vmm_region(CUdeviceptr base, size_t bytes, CUmemGenericAllocationHandle handle, int device_id) {
  absl::Status status = tensorcast::cuda::cu_mem_map(base, bytes, /*offset=*/0, handle, /*flags=*/0);
  if (!status.ok()) {
    return status;
  }
  CUmemAccessDesc access_desc{};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = device_id;
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return tensorcast::cuda::cu_mem_set_access(base, bytes, &access_desc, /*count=*/1);
}

bool gpu_mr_registration_supported(const tensorcast::communicator::transport::net_dev_t& net_dev, size_t bytes) {
  void* gpu_ptr = nullptr;
  absl::Status alloc_status = tensorcast::cuda::malloc(&gpu_ptr, bytes);
  if (!alloc_status.ok()) {
    LOG(WARNING) << "cudaMalloc failed while probing GPUDirect RDMA support: " << alloc_status;
    return false;
  }
  absl::Cleanup free_gpu = [&]() { log_if_error("cudaFree", tensorcast::cuda::free(gpu_ptr)); };

  ibv_mr* mr = nullptr;
  auto reg_res = net_dev->reg_mr(&mr, gpu_ptr, bytes, kMrAccessFlags);
  if (reg_res != tensorcast::communicator::misc::SUCCESS || mr == nullptr) {
    return false;
  }
  log_if_ibv_error("ibv_dereg_mr", tensorcast::communicator::misc::wrap_ibv_dereg_mr(mr));
  return true;
}

} // namespace

namespace tensorcast::unittests {

TEST_CASE("UB repro: cuMemUnmap/cuMemRelease while MR alive (manual)", "[rdma][vmm][ub]") {
  if (!env_enabled(kEnableEnv)) {
    WARN(
        "UB repro is disabled by default. Enable explicitly with:\n"
        "  bazel test //core/communicator:rdma_vmm_ub_test --test_output=all \\\n"
        "    --test_env=TENSORCAST_RUN_RDMA_VMM_UB_TEST=1\n"
        "WARNING: This can crash the process or the host (undefined behavior).");
    return;
  }

  int device_count = 0;
  absl::Status device_status = tensorcast::cuda::get_device_count(&device_count);
  REQUIRE(device_status.ok());
  REQUIRE(device_count > 0);
  REQUIRE_FALSE(tensorcast::cuda::is_fake());

  REQUIRE(tensorcast::communicator::misc::wrap_ibv_symbols() == tensorcast::communicator::misc::SUCCESS);
  REQUIRE(tensorcast::cuda::set_device(kGpuId).ok());

  auto granularity_or = get_vmm_granularity(kGpuId);
  REQUIRE(granularity_or.ok());
  const size_t tensor_bytes = *granularity_or;
  const auto prop = make_vmm_prop(kGpuId);

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true, /*buffers_per_flow=*/2);
  auto server_pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(1ULL << 20),
      /*cpu_slice_bytes=*/(1ULL << 20),
      /*enable_rdma=*/true);
  auto client_pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(1ULL << 20),
      /*cpu_slice_bytes=*/(1ULL << 20),
      /*enable_rdma=*/true);
  tensorcast::communicator::engine::Communicator server(cfg, std::move(server_pools), /*channel_expire_sec=*/5);
  tensorcast::communicator::engine::Communicator client(cfg, std::move(client_pools), /*channel_expire_sec=*/5);

  const int server_port = tensorcast::testing::find_available_port(61000);
  const int client_port = tensorcast::testing::find_available_port(61100);
  REQUIRE(server_port > 0);
  REQUIRE(client_port > 0);
  REQUIRE(server.init("127.0.0.1", static_cast<uint16_t>(server_port), /*conn_count=*/2).ok());
  REQUIRE(client.init("127.0.0.1", static_cast<uint16_t>(client_port), /*conn_count=*/2).ok());

  auto rdma_ctx = std::make_shared<tensorcast::communicator::transport::RdmaContext>();
  auto net_dev = rdma_ctx->get_best_dev(kGpuId);
  REQUIRE(net_dev != nullptr);

  REQUIRE(gpu_mr_registration_supported(net_dev, tensor_bytes));

  CUdeviceptr va = 0;
  REQUIRE(tensorcast::cuda::cu_mem_address_reserve(&va, tensor_bytes, tensor_bytes, /*addr=*/0, /*flags=*/0).ok());
  absl::Cleanup free_va = [&]() {
    log_if_error("cuMemAddressFree", tensorcast::cuda::cu_mem_address_free(va, tensor_bytes));
  };

  CUmemGenericAllocationHandle handle = 0;
  REQUIRE(tensorcast::cuda::cu_mem_create(&handle, tensor_bytes, &prop, /*flags=*/0).ok());
  absl::Cleanup release_handle = [&]() { log_if_error("cuMemRelease", tensorcast::cuda::cu_mem_release(handle)); };
  REQUIRE(map_vmm_region(va, tensor_bytes, handle, kGpuId).ok());
  absl::Cleanup unmap_guard = [&]() { log_if_error("cuMemUnmap", tensorcast::cuda::cu_mem_unmap(va, tensor_bytes)); };

  uint8_t* dst_gpu = nullptr;
  REQUIRE(tensorcast::cuda::malloc(reinterpret_cast<void**>(&dst_gpu), tensor_bytes).ok());
  absl::Cleanup free_dst = [&]() { log_if_error("cudaFree", tensorcast::cuda::free(dst_gpu)); };
  REQUIRE(tensorcast::cuda::memset(dst_gpu, 0, tensor_bytes).ok());

  const auto pattern = tensorcast::testing::create_test_pattern(tensor_bytes, /*seed=*/0x5A);
  REQUIRE(
      tensorcast::cuda::memcpy(reinterpret_cast<void*>(va), pattern.data(), tensor_bytes, cudaMemcpyHostToDevice).ok());
  REQUIRE(tensorcast::cuda::device_synchronize().ok());

  tensorcast::communicator::engine::Communicator::RegisterTensorOptions opts;
  opts.register_mr = true;
  opts.needs_staging = false;
  opts.async = false;
  opts.direct_rdma_enabled = true;

  const std::string tensor_key = "RDMA_VMM_UB_TENSOR";
  absl::Status register_status = server.register_tensor_ex(
      tensor_key,
      static_cast<uint64_t>(va),
      static_cast<uint64_t>(tensor_bytes),
      tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
      kGpuId,
      opts);
  REQUIRE(register_status.ok());

  // Baseline: verify RDMA can read from the VMM tensor when mapping is valid.
  {
    auto future = client.read_tensor(
        tensor_key,
        reinterpret_cast<uint64_t>(dst_gpu),
        static_cast<uint64_t>(tensor_bytes),
        tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
        kGpuId,
        "127.0.0.1",
        static_cast<uint16_t>(server_port),
        /*remote_offset=*/0);

    constexpr auto kTimeout = std::chrono::seconds(30);
    REQUIRE(future.valid());
    REQUIRE(future.wait_for(kTimeout) == std::future_status::ready);
    auto result = future.get();
    REQUIRE(result.status.ok());

    std::vector<uint8_t> host_dst(tensor_bytes);
    REQUIRE(tensorcast::cuda::memcpy(host_dst.data(), dst_gpu, tensor_bytes, cudaMemcpyDeviceToHost).ok());
    REQUIRE(tensorcast::testing::verify_pattern(host_dst.data(), tensor_bytes, /*seed=*/0x5A));
  }

  // UB: Break the required lifetime ordering:
  // Map -> Register(MR) -> Unmap/Release -> ... (MR still alive)
  LOG(ERROR) << "[UB] About to call cuMemUnmap/cuMemRelease while MR is still registered for tensor=" << tensor_key;
  absl::Status ub_unmap_status = tensorcast::cuda::cu_mem_unmap(va, tensor_bytes);
  LOG(ERROR) << "[UB] cuMemUnmap returned: " << ub_unmap_status;
  std::move(unmap_guard).Cancel();
  absl::Status ub_release_status = tensorcast::cuda::cu_mem_release(handle);
  LOG(ERROR) << "[UB] cuMemRelease returned: " << ub_release_status;
  std::move(release_handle).Cancel();

  // Trigger an RDMA READ after unmapping to amplify the effect. Outcome is undefined:
  // - may crash the process or the host
  // - may hang
  // - may return a verbs error (WC status != SUCCESS)
  // - may appear to work (still UB)
  auto future = client.read_tensor(
      tensor_key,
      reinterpret_cast<uint64_t>(dst_gpu),
      static_cast<uint64_t>(tensor_bytes),
      tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
      kGpuId,
      "127.0.0.1",
      static_cast<uint16_t>(server_port),
      /*remote_offset=*/0);

  constexpr auto kTimeout = std::chrono::seconds(10);
  REQUIRE(future.valid());
  auto status = future.wait_for(kTimeout);
  if (status == std::future_status::ready) {
    auto result = future.get();
    WARN("UB repro completed without crashing; final read status: " + result.status.ToString());
  } else {
    WARN("UB repro did not complete within timeout; process may be hung (expected for UB).");
  }
}

} // namespace tensorcast::unittests
