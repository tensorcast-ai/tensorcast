// Copyright (c) 2026, TensorCast Team.

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nccl.h>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_ipc.h"
#include "core/cuda/device_guard.h"
#include "daemon/testing/cuda_ipc_spawn_helper.h"

namespace {

absl::Status nccl_status(ncclResult_t rc, std::string_view what) {
  if (rc == ncclSuccess) {
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrCat(what, ": ", ncclGetErrorString(rc)));
}

class NcclPair final {
 public:
  NcclPair() = default;

  ~NcclPair() {
    if (stream0_ != nullptr) {
      (void)tensorcast::cuda::set_device(0);
      (void)tensorcast::cuda::stream_destroy(stream0_);
    }
    if (stream1_ != nullptr) {
      (void)tensorcast::cuda::set_device(1);
      (void)tensorcast::cuda::stream_destroy(stream1_);
    }
    if (comm0_ != nullptr) {
      (void)ncclCommDestroy(comm0_);
    }
    if (comm1_ != nullptr) {
      (void)ncclCommDestroy(comm1_);
    }
  }

  absl::Status initialize() {
    const std::array<int, 2> devices = {0, 1};
    REQUIRE(tensorcast::cuda::set_device(0).ok());
    REQUIRE(tensorcast::cuda::stream_create_with_flags(&stream0_, cudaStreamNonBlocking).ok());
    REQUIRE(tensorcast::cuda::set_device(1).ok());
    REQUIRE(tensorcast::cuda::stream_create_with_flags(&stream1_, cudaStreamNonBlocking).ok());
    std::array<ncclComm_t, 2> comms = {nullptr, nullptr};
    auto status =
        nccl_status(ncclCommInitAll(comms.data(), static_cast<int>(devices.size()), devices.data()), "ncclCommInitAll");
    if (!status.ok()) {
      return status;
    }
    comm0_ = comms[0];
    comm1_ = comms[1];
    return absl::OkStatus();
  }

  ncclComm_t comm0() const {
    return comm0_;
  }

  ncclComm_t comm1() const {
    return comm1_;
  }

  cudaStream_t stream0() const {
    return stream0_;
  }

  cudaStream_t stream1() const {
    return stream1_;
  }

 private:
  ncclComm_t comm0_{nullptr};
  ncclComm_t comm1_{nullptr};
  cudaStream_t stream0_{nullptr};
  cudaStream_t stream1_{nullptr};
};

} // namespace

TEST_CASE("nccl sendrecv can write into ipc-mapped peer memory", "[daemon][nccl][ipc]") {
  if (tensorcast::cuda::is_fake()) {
    WARN("Skipping NCCL IPC send/recv under fake CUDA backend.");
    return;
  }

  int device_count = 0;
  REQUIRE(tensorcast::cuda::get_device_count(&device_count).ok());
  if (device_count < 2) {
    return;
  }

  const auto helper_path_or = tensorcast::daemon::testing::resolve_cuda_ipc_helper_path();
  REQUIRE(helper_path_or.ok());
  auto child_or = tensorcast::daemon::testing::CudaIpcChild::Spawn(
      *helper_path_or,
      /*device_id=*/1,
      {{.size_bytes = 4096, .fill_byte = 0}});
  REQUIRE(child_or.ok());
  auto child = std::move(*child_or);
  REQUIRE(child.handle_bytes().size() == 1);

  REQUIRE(tensorcast::cuda::set_device(1).ok());
  auto dst_mapping_or =
      tensorcast::cuda::IpcMapping::open(child.handle_bytes()[0], {.flags = cudaIpcMemLazyEnablePeerAccess});
  REQUIRE(dst_mapping_or.ok());
  auto dst_mapping = std::move(*dst_mapping_or);

  tensorcast::cuda::DeviceGuard root_guard(0);
  REQUIRE(root_guard.status().ok());
  void* src_ptr = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&src_ptr, 4096).ok());
  std::vector<std::uint8_t> host_in(4096);
  for (size_t i = 0; i < host_in.size(); ++i) {
    host_in[i] = static_cast<std::uint8_t>(i % 251);
  }
  REQUIRE(tensorcast::cuda::memcpy(src_ptr, host_in.data(), host_in.size(), cudaMemcpyHostToDevice).ok());

  tensorcast::cuda::DeviceGuard dst_guard(1);
  REQUIRE(dst_guard.status().ok());
  REQUIRE(tensorcast::cuda::memset(dst_mapping.get(), 0, host_in.size()).ok());

  NcclPair pair;
  REQUIRE(pair.initialize().ok());

  REQUIRE(nccl_status(ncclGroupStart(), "ncclGroupStart").ok());
  REQUIRE(nccl_status(ncclSend(src_ptr, host_in.size(), ncclUint8, 1, pair.comm0(), pair.stream0()), "ncclSend").ok());
  REQUIRE(
      nccl_status(ncclRecv(dst_mapping.get(), host_in.size(), ncclUint8, 0, pair.comm1(), pair.stream1()), "ncclRecv")
          .ok());
  REQUIRE(nccl_status(ncclGroupEnd(), "ncclGroupEnd").ok());
  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::stream_synchronize(pair.stream0()).ok());
  REQUIRE(tensorcast::cuda::set_device(1).ok());
  REQUIRE(tensorcast::cuda::stream_synchronize(pair.stream1()).ok());

  std::vector<std::uint8_t> host_out(host_in.size(), 0);
  REQUIRE(tensorcast::cuda::memcpy(host_out.data(), dst_mapping.get(), host_out.size(), cudaMemcpyDeviceToHost).ok());
  CHECK(host_out == host_in);

  REQUIRE(tensorcast::cuda::set_device(0).ok());
  REQUIRE(tensorcast::cuda::free(src_ptr).ok());
}
