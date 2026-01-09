// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/misc/ibv_wrap.h"
#include "core/communicator/transport/rdma_context.h"
#include "core/cuda/cuda_api.h"
#include "core/testing/test_helpers.h"

namespace tensorcast::communicator::engine {

class CommunicatorTestPeer {
 public:
  static auto& rdma_context(Communicator& communicator) {
    return communicator.rdma_context_;
  }
};

} // namespace tensorcast::communicator::engine

namespace {

constexpr int kGpuId = 0;
constexpr int kMrAccessFlags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;

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

absl::StatusOr<int> get_dmabuf_supported_attr(int device_id) {
  if (absl::Status init_status = tensorcast::cuda::cu_init(0); !init_status.ok()) {
    return init_status;
  }
  CUdevice cuda_device = 0;
  if (absl::Status device_status = tensorcast::cuda::cu_device_get(&cuda_device, device_id); !device_status.ok()) {
    return device_status;
  }
  int dmabuf_supported = 0;
  if (absl::Status attr_status = tensorcast::cuda::cu_device_get_attribute(
          &dmabuf_supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cuda_device);
      !attr_status.ok()) {
    return attr_status;
  }
  return dmabuf_supported;
}

} // namespace

namespace tensorcast::unittests {

TEST_CASE("CUDA VMM address mapping gates ibv_reg_dmabuf_mr", "[rdma][vmm]") {
  int device_count = 0;
  absl::Status cuda_status = tensorcast::cuda::get_device_count(&device_count);
  if (!cuda_status.ok() || device_count == 0) {
    WARN("CUDA unavailable for rdma_vmm_test: " + cuda_status.ToString());
    return;
  }
  if (tensorcast::cuda::is_fake()) {
    WARN("Skipping rdma_vmm_test: requires real CUDA backend");
    return;
  }

  auto dmabuf_supported_or = get_dmabuf_supported_attr(kGpuId);
  INFO("cu_device_get_attribute(CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED): " + dmabuf_supported_or.status().ToString());
  if (!dmabuf_supported_or.ok()) {
    WARN(
        "Skipping rdma_vmm_test: cuDeviceGetAttribute(DMA_BUF_SUPPORTED) failed: " +
        dmabuf_supported_or.status().ToString());
    return;
  }
  LOG(INFO) << "cu_device_get_attribute(CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED) status=" << dmabuf_supported_or.status()
            << " value=" << *dmabuf_supported_or;
  INFO("CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED: " + std::to_string(*dmabuf_supported_or));
  if (*dmabuf_supported_or == 0) {
    WARN("Skipping rdma_vmm_test: CUDA device does not support dma-buf (CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED=0)");
    return;
  }

  if (tensorcast::communicator::misc::wrap_ibv_symbols() != tensorcast::communicator::misc::SUCCESS) {
    WARN("Skipping rdma_vmm_test: libibverbs not available (wrap_ibv_symbols failed)");
    return;
  }

  auto set_device_status = tensorcast::cuda::set_device(kGpuId);
  REQUIRE(set_device_status.ok());

  auto granularity_or = get_vmm_granularity(kGpuId);
  if (!granularity_or.ok()) {
    WARN("Skipping rdma_vmm_test: CUDA VMM unavailable: " + granularity_or.status().ToString());
    return;
  }

  INFO("CUDA VMM granularity: " + std::to_string(*granularity_or));
  const size_t granularity = *granularity_or;
  const auto prop = make_vmm_prop(kGpuId);

  auto rdma_ctx = std::make_shared<tensorcast::communicator::transport::RdmaContext>();
  auto net_dev = rdma_ctx->get_best_dev(kGpuId);
  if (net_dev == nullptr) {
    WARN("Skipping rdma_vmm_test: no active RDMA device for GPU 0");
    return;
  }

  const long page_size = sysconf(_SC_PAGESIZE);
  INFO("Page size: " + std::to_string(page_size));
  REQUIRE(page_size > 0);
  REQUIRE(granularity % static_cast<size_t>(page_size) == 0);

  SECTION("Reserved-only VA cannot export dma-buf handle") {
    CUdeviceptr va = 0;
    auto reserve_status = tensorcast::cuda::cu_mem_address_reserve(&va, granularity, granularity, /*addr=*/0, 0);
    REQUIRE(reserve_status.ok());
    absl::Cleanup free_va = [&]() {
      log_if_error("cuMemAddressFree", tensorcast::cuda::cu_mem_address_free(va, granularity));
    };

    int dmabuf_fd = -1;
    absl::Status handle_status = tensorcast::cuda::cu_mem_get_handle_for_address_range(
        &dmabuf_fd, va, granularity, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, /*flags=*/0);
    REQUIRE_FALSE(handle_status.ok());
  }

  SECTION("Partially-mapped VA rejects handle export across unmapped pages") {
    const size_t total_bytes = granularity * 2;
    CUdeviceptr va = 0;
    auto reserve_status =
        tensorcast::cuda::cu_mem_address_reserve(&va, total_bytes, granularity, /*addr=*/0, /*flags=*/0);
    REQUIRE(reserve_status.ok());
    absl::Cleanup free_va = [&]() {
      log_if_error("cuMemAddressFree", tensorcast::cuda::cu_mem_address_free(va, total_bytes));
    };

    CUmemGenericAllocationHandle handle = 0;
    auto create_status = tensorcast::cuda::cu_mem_create(&handle, granularity, &prop, /*flags=*/0);
    REQUIRE(create_status.ok());
    absl::Cleanup release_handle = [&]() { log_if_error("cuMemRelease", tensorcast::cuda::cu_mem_release(handle)); };

    auto map_status = map_vmm_region(va, granularity, handle, kGpuId);
    REQUIRE(map_status.ok());
    absl::Cleanup unmap = [&]() { log_if_error("cuMemUnmap", tensorcast::cuda::cu_mem_unmap(va, granularity)); };

    int dmabuf_fd = -1;
    absl::Status handle_status = tensorcast::cuda::cu_mem_get_handle_for_address_range(
        &dmabuf_fd, va, total_bytes, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, /*flags=*/0);
    REQUIRE_FALSE(handle_status.ok());

    // Exporting handle for the unmapped sub-range should also fail (even if the first page is mapped).
    const CUdeviceptr unmapped_va = va + static_cast<CUdeviceptr>(granularity);
    dmabuf_fd = -1;
    handle_status = tensorcast::cuda::cu_mem_get_handle_for_address_range(
        &dmabuf_fd, unmapped_va, granularity, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, /*flags=*/0);
    REQUIRE_FALSE(handle_status.ok());

    // Once the requested range is fully backed, we can export dma-buf and register it.
    dmabuf_fd = -1;
    handle_status = tensorcast::cuda::cu_mem_get_handle_for_address_range(
        &dmabuf_fd, va, granularity, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, /*flags=*/0);
    if (!handle_status.ok() || dmabuf_fd < 0) {
      WARN("cuMemGetHandleForAddressRange(DMA_BUF_FD) not available for mapped CUDA VMM memory; skipping");
      return;
    }
    absl::Cleanup close_dmabuf_fd = [&]() {
      if (dmabuf_fd >= 0 && close(dmabuf_fd) != 0) {
        PLOG(WARNING) << "close(dmabuf_fd) failed during cleanup";
      }
    };

    ibv_mr* mr = tensorcast::communicator::misc::wrap_direct_ibv_reg_dmabuf_mr(
        net_dev->get_pd(),
        /*offset=*/0,
        granularity,
        /*iova=*/static_cast<uint64_t>(va),
        dmabuf_fd,
        kMrAccessFlags);
    if (mr == nullptr) {
      WARN("ibv_reg_dmabuf_mr failed for mapped CUDA VMM memory; skipping mapped registration check");
      return;
    }
    log_if_ibv_error("ibv_dereg_mr", tensorcast::communicator::misc::wrap_ibv_dereg_mr(mr));
  }

  SECTION("Mapped & backed VA allows ibv_reg_dmabuf_mr") {
    CUdeviceptr va = 0;
    auto reserve_status = tensorcast::cuda::cu_mem_address_reserve(&va, granularity, granularity, /*addr=*/0, 0);
    REQUIRE(reserve_status.ok());
    absl::Cleanup free_va = [&]() {
      log_if_error("cuMemAddressFree", tensorcast::cuda::cu_mem_address_free(va, granularity));
    };

    CUmemGenericAllocationHandle handle = 0;
    auto create_status = tensorcast::cuda::cu_mem_create(&handle, granularity, &prop, /*flags=*/0);
    REQUIRE(create_status.ok());
    absl::Cleanup release_handle = [&]() { log_if_error("cuMemRelease", tensorcast::cuda::cu_mem_release(handle)); };

    auto map_status = map_vmm_region(va, granularity, handle, kGpuId);
    REQUIRE(map_status.ok());
    absl::Cleanup unmap = [&]() { log_if_error("cuMemUnmap", tensorcast::cuda::cu_mem_unmap(va, granularity)); };

    int dmabuf_fd = -1;
    absl::Status handle_status = tensorcast::cuda::cu_mem_get_handle_for_address_range(
        &dmabuf_fd, va, granularity, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, /*flags=*/0);
    if (!handle_status.ok() || dmabuf_fd < 0) {
      WARN("cuMemGetHandleForAddressRange(DMA_BUF_FD) error: " + handle_status.ToString());
      WARN("cuMemGetHandleForAddressRange(DMA_BUF_FD) not available for mapped CUDA VMM memory; skipping");
      REQUIRE(false);
    }

    absl::Cleanup close_dmabuf_fd = [&]() {
      if (dmabuf_fd >= 0 && close(dmabuf_fd) != 0) {
        PLOG(WARNING) << "close(dmabuf_fd) failed during cleanup";
      }
    };

    ibv_mr* mr = tensorcast::communicator::misc::wrap_direct_ibv_reg_dmabuf_mr(
        net_dev->get_pd(),
        /*offset=*/0,
        granularity,
        /*iova=*/static_cast<uint64_t>(va),
        dmabuf_fd,
        kMrAccessFlags);
    if (mr == nullptr) {
      WARN("ibv_reg_dmabuf_mr failed for mapped CUDA VMM memory; skipping mapped registration check");
      REQUIRE(false);
    }
    log_if_ibv_error("ibv_dereg_mr", tensorcast::communicator::misc::wrap_ibv_dereg_mr(mr));
  }
}

TEST_CASE("CUDA VMM + Communicator direct RDMA read", "[rdma][vmm][integration]") {
  int device_count = 0;
  absl::Status cuda_status = tensorcast::cuda::get_device_count(&device_count);
  if (!cuda_status.ok() || device_count == 0) {
    WARN("CUDA unavailable for rdma_vmm_test: " + cuda_status.ToString());
    return;
  }
  if (tensorcast::cuda::is_fake()) {
    WARN("Skipping rdma_vmm_test: requires real CUDA backend");
    return;
  }

  auto dmabuf_supported_or = get_dmabuf_supported_attr(kGpuId);
  INFO("cu_device_get_attribute(CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED): " + dmabuf_supported_or.status().ToString());
  if (!dmabuf_supported_or.ok()) {
    WARN(
        "Skipping rdma_vmm_test: cuDeviceGetAttribute(DMA_BUF_SUPPORTED) failed: " +
        dmabuf_supported_or.status().ToString());
    return;
  }
  LOG(INFO) << "cu_device_get_attribute(CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED) status=" << dmabuf_supported_or.status()
            << " value=" << *dmabuf_supported_or;
  INFO("CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED: " + std::to_string(*dmabuf_supported_or));
  if (*dmabuf_supported_or == 0) {
    WARN("Skipping rdma_vmm_test: CUDA device does not support dma-buf (CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED=0)");
    return;
  }

  if (tensorcast::communicator::misc::wrap_ibv_symbols() != tensorcast::communicator::misc::SUCCESS) {
    WARN("Skipping rdma_vmm_test: libibverbs not available (wrap_ibv_symbols failed)");
    return;
  }

  auto set_device_status = tensorcast::cuda::set_device(kGpuId);
  REQUIRE(set_device_status.ok());

  auto granularity_or = get_vmm_granularity(kGpuId);
  if (!granularity_or.ok()) {
    WARN("Skipping rdma_vmm_test: CUDA VMM unavailable: " + granularity_or.status().ToString());
    return;
  }
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

  const int server_port = tensorcast::testing::find_available_port(60000);
  const int client_port = tensorcast::testing::find_available_port(60100);
  REQUIRE(server_port > 0);
  REQUIRE(client_port > 0);
  REQUIRE(server.init("127.0.0.1", static_cast<uint16_t>(server_port), /*conn_count=*/2).ok());
  REQUIRE(client.init("127.0.0.1", static_cast<uint16_t>(client_port), /*conn_count=*/2).ok());

  auto& rdma_ctx = tensorcast::communicator::engine::CommunicatorTestPeer::rdma_context(server);
  auto net_dev = rdma_ctx->get_best_dev(kGpuId);
  if (net_dev == nullptr) {
    WARN("Skipping rdma_vmm_test: no active RDMA device for GPU 0");
    return;
  }

  if (!gpu_mr_registration_supported(net_dev, tensor_bytes)) {
    WARN("GPUDirect RDMA not available for CUDA device memory; skipping CUDA VMM direct RDMA test");
    return;
  }

  CUdeviceptr va = 0;
  auto reserve_status =
      tensorcast::cuda::cu_mem_address_reserve(&va, tensor_bytes, tensor_bytes, /*addr=*/0, /*flags=*/0);
  REQUIRE(reserve_status.ok());
  absl::Cleanup free_va = [&]() {
    log_if_error("cuMemAddressFree", tensorcast::cuda::cu_mem_address_free(va, tensor_bytes));
  };

  uint8_t* dst_gpu = nullptr;
  auto alloc_status = tensorcast::cuda::malloc(reinterpret_cast<void**>(&dst_gpu), tensor_bytes);
  REQUIRE(alloc_status.ok());
  absl::Cleanup free_dst = [&]() { log_if_error("cudaFree", tensorcast::cuda::free(dst_gpu)); };
  REQUIRE(tensorcast::cuda::memset(dst_gpu, 0, tensor_bytes).ok());

  auto run_roundtrip = [&](uint8_t seed, CUmemGenericAllocationHandle handle) -> bool {
    auto map_status = map_vmm_region(va, tensor_bytes, handle, kGpuId);
    REQUIRE(map_status.ok());
    absl::Cleanup unmap_guard = [&]() { log_if_error("cuMemUnmap", tensorcast::cuda::cu_mem_unmap(va, tensor_bytes)); };

    const auto pattern = tensorcast::testing::create_test_pattern(tensor_bytes, seed);
    REQUIRE(
        tensorcast::cuda::memcpy(reinterpret_cast<void*>(va), pattern.data(), tensor_bytes, cudaMemcpyHostToDevice)
            .ok());
    REQUIRE(tensorcast::cuda::device_synchronize().ok());

    tensorcast::communicator::engine::Communicator::RegisterTensorOptions opts;
    opts.register_mr = true;
    opts.needs_staging = false;
    opts.async = false;
    opts.direct_rdma_enabled = true;
    const std::string tensor_key = "RDMA_VMM_TENSOR";
    auto register_status = server.register_tensor_ex(
        tensor_key,
        static_cast<uint64_t>(va),
        static_cast<uint64_t>(tensor_bytes),
        tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
        kGpuId,
        opts);
    if (!register_status.ok()) {
      WARN("GPUDirect RDMA registration for CUDA VMM memory is unavailable: " + register_status.ToString());
      return false;
    }
    absl::Cleanup unregister_guard = [&]() {
      log_if_error("client.unregister_tensor", client.unregister_tensor(tensor_key));
      log_if_error("server.unregister_tensor", server.unregister_tensor(tensor_key));
    };

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

    REQUIRE(tensorcast::cuda::device_synchronize().ok());
    std::vector<uint8_t> host_dst(tensor_bytes);
    REQUIRE(tensorcast::cuda::memcpy(host_dst.data(), dst_gpu, tensor_bytes, cudaMemcpyDeviceToHost).ok());
    REQUIRE(tensorcast::testing::verify_pattern(host_dst.data(), tensor_bytes, seed));

    // Also verify a non-zero remote_offset read works for CUDA VMM memory.
    {
      const size_t slice_offset = tensor_bytes / 4;
      const size_t slice_bytes = std::min<size_t>(tensor_bytes / 2, 4096);
      REQUIRE(slice_offset + slice_bytes <= tensor_bytes);

      REQUIRE(tensorcast::cuda::memset(dst_gpu, 0, slice_bytes).ok());
      auto slice_future = client.read_tensor(
          tensor_key,
          reinterpret_cast<uint64_t>(dst_gpu),
          static_cast<uint64_t>(slice_bytes),
          tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
          kGpuId,
          "127.0.0.1",
          static_cast<uint16_t>(server_port),
          static_cast<uint64_t>(slice_offset));

      REQUIRE(slice_future.valid());
      REQUIRE(slice_future.wait_for(kTimeout) == std::future_status::ready);
      auto slice_result = slice_future.get();
      REQUIRE(slice_result.status.ok());

      std::vector<uint8_t> host_slice(slice_bytes);
      REQUIRE(tensorcast::cuda::memcpy(host_slice.data(), dst_gpu, slice_bytes, cudaMemcpyDeviceToHost).ok());
      REQUIRE(std::equal(host_slice.begin(), host_slice.end(), pattern.begin() + slice_offset));
    }

    REQUIRE(client.unregister_tensor(tensor_key).ok());
    REQUIRE(server.unregister_tensor(tensor_key).ok());
    std::move(unregister_guard).Cancel();

    REQUIRE(tensorcast::cuda::cu_mem_unmap(va, tensor_bytes).ok());
    std::move(unmap_guard).Cancel();
    return true;
  };

  CUmemGenericAllocationHandle handle1 = 0;
  REQUIRE(tensorcast::cuda::cu_mem_create(&handle1, tensor_bytes, &prop, /*flags=*/0).ok());
  absl::Cleanup release_handle1 = [&]() { log_if_error("cuMemRelease", tensorcast::cuda::cu_mem_release(handle1)); };
  if (!run_roundtrip(/*seed=*/0x11, handle1)) {
    return;
  }

  // With the mapping removed (VA reserved only), Communicator registration should fail.
  {
    tensorcast::communicator::engine::Communicator::RegisterTensorOptions opts;
    opts.register_mr = true;
    opts.needs_staging = false;
    opts.async = false;
    opts.direct_rdma_enabled = true;
    auto status = server.register_tensor_ex(
        "RDMA_VMM_TENSOR",
        static_cast<uint64_t>(va),
        static_cast<uint64_t>(tensor_bytes),
        tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
        kGpuId,
        opts);
    REQUIRE_FALSE(status.ok());
  }

  CUmemGenericAllocationHandle handle2 = 0;
  REQUIRE(tensorcast::cuda::cu_mem_create(&handle2, tensor_bytes, &prop, /*flags=*/0).ok());
  absl::Cleanup release_handle2 = [&]() { log_if_error("cuMemRelease", tensorcast::cuda::cu_mem_release(handle2)); };
  (void)run_roundtrip(/*seed=*/0x22, handle2);
}

} // namespace tensorcast::unittests
