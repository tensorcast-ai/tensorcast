// Copyright (c) 2026, TensorCast Team.

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/communicator/engine/engine.h"
#include "core/testing/test_helpers.h"

namespace tensorcast::communicator::engine {

class CommunicatorTestPeer {
 public:
  static absl::StatusOr<std::shared_ptr<void>> acquire_tensor_read_lease(
      Communicator& communicator,
      const std::string& tensor_key) {
    return communicator.acquire_tensor_read_lease(tensor_key);
  }
};

} // namespace tensorcast::communicator::engine

namespace tensorcast::unittests {

TEST_CASE("unregister_tensor waits in-flight source reads and blocks new reads", "[communicator][drain]") {
  auto cfg = tensorcast::testing::make_tcp_communicator_config();
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/false);
  communicator::engine::Communicator comm(cfg, std::move(pools), 0);

  std::vector<std::uint8_t> buffer(4096, 7);
  communicator::engine::Communicator::RegisterTensorOptions options;
  options.register_mr = false;
  options.needs_staging = false;
  options.async = false;
  constexpr const char* kTensorKey = "unregister_drain_tensor";
  REQUIRE(comm.register_tensor_ex(
                  kTensorKey,
                  reinterpret_cast<std::uint64_t>(buffer.data()),
                  static_cast<std::uint64_t>(buffer.size()),
                  communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
                  -1,
                  options)
              .ok());

  auto lease_or = communicator::engine::CommunicatorTestPeer::acquire_tensor_read_lease(comm, kTensorKey);
  REQUIRE(lease_or.ok());
  std::shared_ptr<void> inflight_read = std::move(*lease_or);

  std::atomic<bool> unregister_done{false};
  absl::Status unregister_status;
  std::thread unregister_thread([&]() {
    unregister_status = comm.unregister_tensor(kTensorKey);
    unregister_done.store(true, std::memory_order_release);
  });

  absl::SleepFor(absl::Milliseconds(150));
  CHECK_FALSE(unregister_done.load(std::memory_order_acquire));

  auto blocked_lease_or = communicator::engine::CommunicatorTestPeer::acquire_tensor_read_lease(comm, kTensorKey);
  REQUIRE_FALSE(blocked_lease_or.ok());
  CHECK(blocked_lease_or.status().code() == absl::StatusCode::kUnavailable);

  inflight_read.reset();
  unregister_thread.join();

  REQUIRE(unregister_done.load(std::memory_order_acquire));
  REQUIRE(unregister_status.ok());
}

} // namespace tensorcast::unittests
