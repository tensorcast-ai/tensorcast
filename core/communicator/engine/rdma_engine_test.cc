
// Copyright (c) 2025-2026, TensorCast Team.

#include <arpa/inet.h>
#include <catch2/catch_test_macros.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#include "core/communicator/engine/engine.h"

#include "absl/synchronization/mutex.h"
#include "core/communicator/misc/utils.h"
#include "core/cuda/cuda_api.h"
#include "core/testing/test_helpers.h"

namespace tensorcast::communicator::engine {

class CommunicatorTestPeer {
 public:
  static auto& rdma_context(Communicator& communicator) {
    return communicator.rdma_context_;
  }

  static auto& pending_requests(Communicator& communicator) {
    return communicator.pending_requests_;
  }

  static auto& channels(Communicator& communicator) {
    return communicator.channels_;
  }

  static auto on_receive_response(
      Communicator& communicator,
      const channel_t& channel,
      const transport::tcp_transport_t& control,
      const engine_message_t& message) {
    return communicator.on_receive_response(channel, control, message);
  }

  static auto on_receive_request(
      Communicator& communicator,
      const channel_t& channel,
      const transport::tcp_transport_t& control,
      const engine_message_t& message) {
    return communicator.on_receive_request(channel, control, message);
  }

  static void stop_workers(Communicator& communicator) {
    communicator.stop_.store(true);
    communicator.request_queue_.stop();
  }

  static bool has_rdma_device(Communicator& communicator) {
    if (communicator.rdma_context_ == nullptr) {
      return false;
    }
    return !communicator.rdma_context_->list_devs().empty();
  }
};

} // namespace tensorcast::communicator::engine

namespace tensorcast::unittests {

#define BUF_SIZE 65536
#define CPU_KEY "RDMA_TENSOR_KEY_CPU"
#define GPU_KEY "RDMA_TENSOR_KEY_GPU"

struct RdmaTestFixture {
  communicator::engine::Communicator* server_;
  absl::Status server_init_status_;
  communicator::engine::Communicator* client_;
  absl::Status client_init_status_;
  uint32_t server_buf_[BUF_SIZE];
  uint32_t client_buf_[BUF_SIZE];
  uint32_t* server_gpu_ptr_ = nullptr;
  uint32_t* client_gpu_ptr_ = nullptr;
  size_t tensor_bytes_ = sizeof(uint32_t) * BUF_SIZE;

  RdmaTestFixture() {
    auto srv_cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
    // srv_cfg.mutable_rdma()->set_qp_count(4); // Enable multi-QP for server, default is 1
    auto srv_pools = tensorcast::testing::make_test_pinned_staging_pools(
        srv_cfg.stager().buffers_per_flow(),
        srv_cfg.transport().tcp_conn_count(),
        /*gpu_slice_bytes=*/(16ULL << 20),
        /*cpu_slice_bytes=*/(4ULL << 20),
        /*enable_rdma=*/true);
    server_ = new communicator::engine::Communicator(srv_cfg, std::move(srv_pools), 30);
    server_init_status_ = server_->init("127.0.0.1", 60000, 8);
    auto cli_cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
    // cli_cfg.mutable_rdma()->set_qp_count(4); // Enable multi-QP for client, default is 1
    auto cli_pools = tensorcast::testing::make_test_pinned_staging_pools(
        cli_cfg.stager().buffers_per_flow(),
        cli_cfg.transport().tcp_conn_count(),
        /*gpu_slice_bytes=*/(16ULL << 20),
        /*cpu_slice_bytes=*/(4ULL << 20),
        /*enable_rdma=*/true);
    client_ = new communicator::engine::Communicator(cli_cfg, std::move(cli_pools), 30);
    client_init_status_ = client_->init("127.0.0.1", 60001, 8);

    for (uint32_t i = 0; i < BUF_SIZE; i++) {
      server_buf_[i] = i;
      client_buf_[i] = 0;
    }

    int device_count = 0;
    auto device_count_status = tensorcast::cuda::get_device_count(&device_count);
    REQUIRE(device_count_status.ok());
    REQUIRE(device_count > 0);

    auto status = tensorcast::cuda::malloc(reinterpret_cast<void**>(&server_gpu_ptr_), tensor_bytes_);
    REQUIRE(status.ok());
    status = tensorcast::cuda::malloc(reinterpret_cast<void**>(&client_gpu_ptr_), tensor_bytes_);
    REQUIRE(status.ok());

    status = tensorcast::cuda::memcpy(server_gpu_ptr_, server_buf_, tensor_bytes_, cudaMemcpyHostToDevice);
    REQUIRE(status.ok());
    status = tensorcast::cuda::memset(client_gpu_ptr_, 0, tensor_bytes_);
    REQUIRE(status.ok());
  }

  ~RdmaTestFixture() {
    if (server_gpu_ptr_ != nullptr) {
      auto status = tensorcast::cuda::free(server_gpu_ptr_);
      REQUIRE(status.ok());
      server_gpu_ptr_ = nullptr;
    }
    if (client_gpu_ptr_ != nullptr) {
      auto status = tensorcast::cuda::free(client_gpu_ptr_);
      REQUIRE(status.ok());
      client_gpu_ptr_ = nullptr;
    }
    delete server_;
    delete client_;
  }
};

TEST_CASE("RDMA Communication Engine", "[rdma][communicator]") {
  RdmaTestFixture fixture;

  SECTION("Initialization") {
    REQUIRE(fixture.server_init_status_.ok());
    REQUIRE(fixture.client_init_status_.ok());
  }

  SECTION("Register tensors synchronously") {
    REQUIRE(fixture.server_init_status_.ok());
    REQUIRE(fixture.client_init_status_.ok());
    if (!communicator::engine::CommunicatorTestPeer::has_rdma_device(*fixture.server_) ||
        !communicator::engine::CommunicatorTestPeer::has_rdma_device(*fixture.client_)) {
      SUCCEED("Skipping RDMA register test: no RDMA net devices available");
      return;
    }

    // register GPU tensor
    communicator::engine::Communicator::RegisterTensorOptions o1;
    o1.register_mr = true;
    o1.needs_staging = true;
    o1.async = false;
    auto status = fixture.server_->register_tensor_ex(
        GPU_KEY,
        reinterpret_cast<uint64_t>(fixture.server_gpu_ptr_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
        0,
        o1);
    REQUIRE(status.ok());

    // register CPU tensor
    communicator::engine::Communicator::RegisterTensorOptions o2;
    o2.register_mr = true;
    o2.needs_staging = false;
    o2.async = false;
    status = fixture.server_->register_tensor_ex(
        CPU_KEY,
        reinterpret_cast<uint64_t>(fixture.server_buf_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
        -1,
        o2);
    REQUIRE(status.ok());
  }

  SECTION("Register and unregister tensors") {
    REQUIRE(fixture.server_init_status_.ok());
    REQUIRE(fixture.client_init_status_.ok());
    if (!communicator::engine::CommunicatorTestPeer::has_rdma_device(*fixture.server_) ||
        !communicator::engine::CommunicatorTestPeer::has_rdma_device(*fixture.client_)) {
      SUCCEED("Skipping RDMA register/unregister test: no RDMA net devices available");
      return;
    }

    // register/unregister CPU tensor
    communicator::engine::Communicator::RegisterTensorOptions o3;
    o3.register_mr = true;
    o3.needs_staging = false;
    o3.async = false;
    auto status = fixture.server_->register_tensor_ex(
        CPU_KEY,
        reinterpret_cast<uint64_t>(fixture.server_buf_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
        -1,
        o3);
    REQUIRE(status.ok());

    status = fixture.server_->unregister_tensor(CPU_KEY);
    REQUIRE(status.ok());

    // register/unregister GPU tensor
    communicator::engine::Communicator::RegisterTensorOptions o4;
    o4.register_mr = true;
    o4.needs_staging = true;
    o4.async = false;
    status = fixture.server_->register_tensor_ex(
        GPU_KEY,
        reinterpret_cast<uint64_t>(fixture.server_gpu_ptr_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
        0,
        o4);
    REQUIRE(status.ok());

    status = fixture.server_->unregister_tensor(GPU_KEY);
    REQUIRE(status.ok());
  }

  SECTION("Unregister non-existent tensors is idempotent OK") {
    REQUIRE(fixture.server_init_status_.ok());
    REQUIRE(fixture.client_init_status_.ok());

    auto status = fixture.server_->unregister_tensor(CPU_KEY);
    REQUIRE(status.ok());

    status = fixture.server_->unregister_tensor(GPU_KEY);
    REQUIRE(status.ok());
  }

  SECTION("Read tensors") {
    REQUIRE(fixture.server_init_status_.ok());
    REQUIRE(fixture.client_init_status_.ok());
    if (!communicator::engine::CommunicatorTestPeer::has_rdma_device(*fixture.server_) ||
        !communicator::engine::CommunicatorTestPeer::has_rdma_device(*fixture.client_)) {
      SUCCEED("Skipping RDMA read test: no RDMA net devices available");
      return;
    }

    communicator::engine::Communicator::RegisterTensorOptions o5;
    o5.register_mr = true;
    o5.needs_staging = true;
    o5.async = false;
    auto refresh_status = tensorcast::cuda::memcpy(
        fixture.server_gpu_ptr_, fixture.server_buf_, fixture.tensor_bytes_, cudaMemcpyHostToDevice);
    REQUIRE(refresh_status.ok());
    auto status = fixture.server_->register_tensor_ex(
        GPU_KEY,
        reinterpret_cast<uint64_t>(fixture.server_gpu_ptr_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
        0,
        o5);
    REQUIRE(status.ok());

    communicator::engine::Communicator::RegisterTensorOptions o6;
    o6.register_mr = true;
    o6.needs_staging = false;
    o6.async = false;
    status = fixture.server_->register_tensor_ex(
        CPU_KEY,
        reinterpret_cast<uint64_t>(fixture.server_buf_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
        -1,
        o6);
    REQUIRE(status.ok());

    // Test reading with various offsets
    for (uint32_t offset = 4096; offset < BUF_SIZE - 1; offset += 4096) {
      auto reset_status = tensorcast::cuda::memset(fixture.client_gpu_ptr_, 0, fixture.tensor_bytes_);
      REQUIRE(reset_status.ok());

      auto future_result = fixture.client_->read_tensor(
          GPU_KEY,
          reinterpret_cast<uint64_t>(fixture.client_gpu_ptr_),
          (BUF_SIZE - offset) * sizeof(uint32_t),
          communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
          0,
          "127.0.0.1",
          60000,
          offset * sizeof(uint32_t));

      auto result = future_result.get();
      REQUIRE(result.status.ok());

      auto copy_status = tensorcast::cuda::memcpy(
          fixture.client_buf_, fixture.client_gpu_ptr_, fixture.tensor_bytes_, cudaMemcpyDeviceToHost);
      REQUIRE(copy_status.ok());

      bool verify_ok = true;
      for (uint32_t i = 0; i < BUF_SIZE - offset; i++) {
        if (fixture.client_buf_[i] != fixture.server_buf_[i + offset]) {
          verify_ok = false;
        }
      }
      REQUIRE(verify_ok);
    }

    // Test reading entire buffer
    auto reset_status = tensorcast::cuda::memset(fixture.client_gpu_ptr_, 0, fixture.tensor_bytes_);
    REQUIRE(reset_status.ok());

    auto future_result = fixture.client_->read_tensor(
        GPU_KEY,
        reinterpret_cast<uint64_t>(fixture.client_gpu_ptr_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
        0,
        "127.0.0.1",
        60000);

    auto result = future_result.get();
    REQUIRE(result.status.ok());

    auto copy_status = tensorcast::cuda::memcpy(
        fixture.client_buf_, fixture.client_gpu_ptr_, fixture.tensor_bytes_, cudaMemcpyDeviceToHost);
    REQUIRE(copy_status.ok());

    bool verify_ok = true;
    for (uint32_t i = 0; i < BUF_SIZE; i++) {
      if (fixture.client_buf_[i] != fixture.server_buf_[i]) {
        verify_ok = false;
      }
    }
    REQUIRE(verify_ok);

    // Test connection close
    auto close_status = fixture.client_->close_connection("127.0.0.1", 60000);
    REQUIRE(close_status.ok());

    // Double close should fail
    close_status = fixture.client_->close_connection("127.0.0.1", 60000);
    REQUIRE_FALSE(close_status.ok());
  }
}

using namespace tensorcast::communicator::engine;

TEST_CASE("RDMA read defers until handshake completes", "[rdma][communicator][handshake]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
  using tensorcast::communicator::misc::STRNCPY;
  using tensorcast::communicator::misc::SUCCESS;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator client(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(client)) {
    CommunicatorTestPeer::stop_workers(client);
    SUCCEED("Skipping RDMA handshake test: no RDMA net devices available");
    return;
  }

  auto& rdma_ctx = CommunicatorTestPeer::rdma_context(client);
  auto net_dev = rdma_ctx->get_best_dev(/*gpu_id=*/0);
  REQUIRE(net_dev != nullptr);
  const std::string local_dev_name = net_dev->get_name();
  const std::string remote_dev_name = "peer.nic0";

  constexpr size_t local_bytes = 256;
  uint8_t* local_gpu_buffer = nullptr;
  auto alloc_status = tensorcast::cuda::malloc(reinterpret_cast<void**>(&local_gpu_buffer), local_bytes);
  REQUIRE(alloc_status.ok());

  auto local_tensor = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "rdma_handshake_tensor",
      reinterpret_cast<uint64_t>(local_gpu_buffer),
      static_cast<uint64_t>(local_bytes),
      COMMUNICATE_ENGINE_DEV_GPU,
      net_dev);
  local_tensor->set_device_id(0);
  local_tensor->register_mr(net_dev.get());
  local_tensor->set_read_ready();

  auto read_request = std::make_shared<tensorcast::communicator::transport::ReadRequest>(
      "rdma_handshake_tensor",
      "127.0.0.1",
      65000,
      local_tensor,
      /*remote_offset=*/0,
      /*request_id=*/11,
      /*rail_id=*/net_dev->get_rail_id());
  auto remote_tensor = std::make_shared<tensorcast::communicator::transport::RemotePartitionTensor>(
      "rdma_handshake_tensor", remote_dev_name, /*addr=*/0xABC0, local_tensor->get_bytes(), /*rkey=*/0x1234);
  read_request->set_remote_tensor(remote_tensor);

  CommunicatorTestPeer::pending_requests(client).put(read_request->get_key(), read_request);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65000);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);

  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/4);
  CommunicatorTestPeer::channels(client).put(read_request->get_dst_url(), channel);

  const uint32_t payload_size = sizeof(ProtoReadResponseExHeader) + sizeof(ProtoReadResponseExSeg);
  auto response = std::make_shared<EngineMessage>(ENGINE_OP_READ_RESPONSE_EX, payload_size);
  auto* hdr = response->get_payload<ProtoReadResponseExHeader>();
  STRNCPY(hdr->tensor_key, "rdma_handshake_tensor", kMaxTensorNameLen);
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->staged = 1;
  STRNCPY(hdr->nic_name, remote_dev_name, kMaxDevName);
  hdr->num_segments = 1;
  hdr->window_seq = 0;
  hdr->credit_granted = 1;
  hdr->request_offset = 0;
  hdr->more_segments = 0;
  auto* seg =
      reinterpret_cast<ProtoReadResponseExSeg*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader));
  seg->addr = remote_tensor->get_uint64_addr();
  seg->offset = 0;
  seg->bytes = static_cast<uint32_t>(local_tensor->get_bytes());
  seg->rkey = remote_tensor->get_rkey();

  auto status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, response);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  auto endpoint = channel->get_rdma_endpoint(local_dev_name, remote_dev_name);
  REQUIRE(endpoint != nullptr);

  {
    absl::MutexLock lock(&endpoint->mu);
    CHECK(endpoint->state == Channel::HandshakeState::kConnectRequested);
    CHECK(endpoint->pending_reads.size() == 1);
    CHECK(endpoint->pending_reads.front().request == read_request);
  }

  auto rdma_transport = channel->get_rdma(local_dev_name, remote_dev_name);
  REQUIRE(rdma_transport != nullptr);

  CHECK_FALSE(read_request->is_result_set());
  CHECK(read_request->expected_completions_.load() == 1);
  {
    absl::MutexLock lock(&read_request->ack_mu_);
    CHECK(read_request->pending_ack_windows_.size() == 1);
    const auto& pending_window = read_request->pending_ack_windows_.front();
    CHECK(pending_window.window_seq == 0);
    CHECK(pending_window.final_window);
    CHECK(pending_window.offsets.size() == 1);
    CHECK(pending_window.offsets[0] == 0);
  }

  auto connect_rsp = EngineMessage::make_message<ProtoRdmaConnectResponse>(ENGINE_OP_RDMA_CONNECT_RESPONSE);
  auto* connect_payload = connect_rsp->get_payload<ProtoRdmaConnectResponse>();
  STRNCPY(connect_payload->src_dev_name, local_dev_name, kMaxDevName);
  STRNCPY(connect_payload->dst_dev_name, remote_dev_name, kMaxDevName);
  CHECK(rdma_transport->get_local_info(&connect_payload->qp_info) == SUCCESS);

  status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, connect_rsp);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  CHECK(read_request->expected_completions_.load() == 1);
  {
    absl::MutexLock lock(&read_request->ack_mu_);
    CHECK(read_request->pending_ack_windows_.size() == 1);
  }
  {
    absl::MutexLock lock(&endpoint->mu);
    CHECK(endpoint->state == Channel::HandshakeState::kReady);
    CHECK(endpoint->pending_reads.empty());
  }

  CHECK(rdma_transport->ready());
  CHECK_FALSE(read_request->is_result_set());

  auto& pending_map = CommunicatorTestPeer::pending_requests(client);
  if (pending_map.exist(read_request->get_key())) {
    pending_map.del(read_request->get_key());
  }
  CommunicatorTestPeer::channels(client).del(read_request->get_dst_url());

  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(client);

  auto free_status = tensorcast::cuda::free(local_gpu_buffer);
  REQUIRE(free_status.ok());
}

TEST_CASE("RDMA handshake failure surfaces retryable error", "[rdma][communicator][handshake]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
  using tensorcast::communicator::misc::STRNCPY;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator client(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(client)) {
    CommunicatorTestPeer::stop_workers(client);
    SUCCEED("Skipping RDMA handshake failure test: no RDMA net devices available");
    return;
  }

  auto& rdma_ctx = CommunicatorTestPeer::rdma_context(client);
  auto net_dev = rdma_ctx->get_best_dev(/*gpu_id=*/0);
  REQUIRE(net_dev != nullptr);
  const std::string local_dev_name = net_dev->get_name();
  const std::string remote_dev_name = "peer.nic1";

  constexpr size_t local_bytes = 128;
  uint8_t* local_gpu_buffer = nullptr;
  auto alloc_status = tensorcast::cuda::malloc(reinterpret_cast<void**>(&local_gpu_buffer), local_bytes);
  REQUIRE(alloc_status.ok());

  auto local_tensor = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "rdma_handshake_tensor_fail",
      reinterpret_cast<uint64_t>(local_gpu_buffer),
      static_cast<uint64_t>(local_bytes),
      COMMUNICATE_ENGINE_DEV_GPU,
      net_dev);
  local_tensor->set_device_id(0);
  local_tensor->register_mr(net_dev.get());
  local_tensor->set_read_ready();

  auto read_request = std::make_shared<tensorcast::communicator::transport::ReadRequest>(
      "rdma_handshake_tensor_fail",
      "127.0.0.1",
      65001,
      local_tensor,
      /*remote_offset=*/0,
      /*request_id=*/13);
  auto remote_tensor = std::make_shared<tensorcast::communicator::transport::RemotePartitionTensor>(
      "rdma_handshake_tensor_fail", remote_dev_name, /*addr=*/0xCAF0, local_tensor->get_bytes(), /*rkey=*/0xABCD);
  read_request->set_remote_tensor(remote_tensor);

  CommunicatorTestPeer::pending_requests(client).put(read_request->get_key(), read_request);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65001);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);

  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/2,
      /*max_window_segments=*/2);
  CommunicatorTestPeer::channels(client).put(read_request->get_dst_url(), channel);

  const uint32_t payload_size = sizeof(ProtoReadResponseExHeader) + sizeof(ProtoReadResponseExSeg);
  auto response = std::make_shared<EngineMessage>(ENGINE_OP_READ_RESPONSE_EX, payload_size);
  auto* hdr = response->get_payload<ProtoReadResponseExHeader>();
  STRNCPY(hdr->tensor_key, "rdma_handshake_tensor_fail", kMaxTensorNameLen);
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->staged = 1;
  STRNCPY(hdr->nic_name, remote_dev_name, kMaxDevName);
  hdr->num_segments = 1;
  hdr->window_seq = 0;
  hdr->credit_granted = 1;
  hdr->request_offset = 0;
  hdr->more_segments = 0;
  auto* seg =
      reinterpret_cast<ProtoReadResponseExSeg*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader));
  seg->addr = remote_tensor->get_uint64_addr();
  seg->offset = 0;
  seg->bytes = static_cast<uint32_t>(local_tensor->get_bytes());
  seg->rkey = remote_tensor->get_rkey();

  auto status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, response);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  auto endpoint = channel->get_rdma_endpoint(local_dev_name, remote_dev_name);
  REQUIRE(endpoint != nullptr);

  auto connect_failed = EngineMessage::make_message<ProtoRdmaConnectFailed>(ENGINE_OP_RDMA_CONNECT_FAILED);
  auto* fail_payload = connect_failed->get_payload<ProtoRdmaConnectFailed>();
  STRNCPY(fail_payload->src_dev_name, local_dev_name, kMaxDevName);
  STRNCPY(fail_payload->dst_dev_name, remote_dev_name, kMaxDevName);

  status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, connect_failed);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  CHECK(read_request->is_result_set());
  CHECK_FALSE(read_request->status_.status.ok());
  CHECK(absl::IsUnavailable(read_request->status_.status));

  {
    absl::MutexLock lock(&endpoint->mu);
    CHECK(endpoint->state == Channel::HandshakeState::kFailed);
    CHECK(endpoint->pending_reads.empty());
    CHECK(endpoint->failure_count >= 1);
    CHECK(endpoint->next_retry_at > absl::Now());
  }

  CHECK_FALSE(CommunicatorTestPeer::pending_requests(client).exist(read_request->get_key()));

  if (CommunicatorTestPeer::channels(client).exist(read_request->get_dst_url())) {
    CommunicatorTestPeer::channels(client).del(read_request->get_dst_url());
  }

  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(client);

  auto free_status = tensorcast::cuda::free(local_gpu_buffer);
  REQUIRE(free_status.ok());
}

TEST_CASE("RDMA connect request failure sends connect-failed opcode", "[rdma][communicator][handshake]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::misc::STRNCPY;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator server(cfg, std::move(pools), /*channel_expire_sec=*/0);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  timeval timeout{};
  timeout.tv_sec = 2;
  REQUIRE(::setsockopt(sv[1], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0);

  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65003);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/2,
      /*max_window_segments=*/2);

  auto connect_req = EngineMessage::make_message<ProtoRdmaConnectRequest>(ENGINE_OP_RDMA_CONNECT_REQUEST);
  auto* req_payload = connect_req->get_payload<ProtoRdmaConnectRequest>();
  STRNCPY(req_payload->src_dev_name, "peer.nic3", kMaxDevName);
  STRNCPY(req_payload->dst_dev_name, "nonexistent.nic", kMaxDevName);

  auto status = CommunicatorTestPeer::on_receive_request(server, channel, control_transport, connect_req);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  ProtoHeader header{};
  auto header_bytes = ::recv(sv[1], &header, sizeof(header), MSG_WAITALL);
  REQUIRE(header_bytes == static_cast<ssize_t>(sizeof(header)));
  CHECK(header.prefix == kHeaderPrefix);
  CHECK(header.op == ENGINE_OP_RDMA_CONNECT_FAILED);
  CHECK(header.size == sizeof(ProtoRdmaConnectFailed));

  ProtoRdmaConnectFailed failed_payload{};
  auto payload_bytes = ::recv(sv[1], &failed_payload, sizeof(failed_payload), MSG_WAITALL);
  REQUIRE(payload_bytes == static_cast<ssize_t>(sizeof(failed_payload)));
  CHECK(std::string(failed_payload.src_dev_name) == "peer.nic3");
  CHECK(std::string(failed_payload.dst_dev_name) == "nonexistent.nic");

  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(server);
}

TEST_CASE("RDMA response lookup uses request instance key across windows", "[rdma][communicator][handshake]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
  using tensorcast::communicator::misc::STRNCPY;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator client(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(client)) {
    CommunicatorTestPeer::stop_workers(client);
    SUCCEED("Skipping RDMA response lookup test: no RDMA net devices available");
    return;
  }

  auto& rdma_ctx = CommunicatorTestPeer::rdma_context(client);
  auto net_dev = rdma_ctx->get_best_dev(/*gpu_id=*/0);
  REQUIRE(net_dev != nullptr);
  const std::string local_dev_name = net_dev->get_name();
  const std::string remote_dev_name = "peer.nic2";

  constexpr size_t local_bytes = 256;
  constexpr uint64_t response_segment_offset = 64;
  constexpr uint64_t request_id = 23;
  uint8_t* local_gpu_buffer = nullptr;
  auto alloc_status = tensorcast::cuda::malloc(reinterpret_cast<void**>(&local_gpu_buffer), local_bytes);
  REQUIRE(alloc_status.ok());

  auto local_tensor = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "rdma_window_lookup_tensor",
      reinterpret_cast<uint64_t>(local_gpu_buffer),
      static_cast<uint64_t>(local_bytes),
      COMMUNICATE_ENGINE_DEV_GPU,
      net_dev);
  local_tensor->set_device_id(0);
  local_tensor->register_mr(net_dev.get());
  local_tensor->set_read_ready();

  auto read_request = std::make_shared<tensorcast::communicator::transport::ReadRequest>(
      "rdma_window_lookup_tensor",
      "127.0.0.1",
      65002,
      local_tensor,
      /*remote_offset=*/0,
      request_id,
      net_dev->get_rail_id());
  auto remote_tensor = std::make_shared<tensorcast::communicator::transport::RemotePartitionTensor>(
      "rdma_window_lookup_tensor", remote_dev_name, /*addr=*/0xABCD, local_tensor->get_bytes(), /*rkey=*/0x1234);
  read_request->set_remote_tensor(remote_tensor);

  CommunicatorTestPeer::pending_requests(client).put(read_request->get_key(), read_request);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65002);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);

  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/4);
  CommunicatorTestPeer::channels(client).put(read_request->get_dst_url(), channel);

  const uint32_t payload_size = sizeof(ProtoReadResponseExHeader) + sizeof(ProtoReadResponseExSeg);
  auto response = std::make_shared<EngineMessage>(ENGINE_OP_READ_RESPONSE_EX, payload_size);
  auto* hdr = response->get_payload<ProtoReadResponseExHeader>();
  STRNCPY(hdr->tensor_key, "rdma_window_lookup_tensor", kMaxTensorNameLen);
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->staged = 1;
  STRNCPY(hdr->nic_name, remote_dev_name, kMaxDevName);
  hdr->num_segments = 1;
  hdr->window_seq = 1;
  hdr->credit_granted = 1;
  hdr->request_offset = 0;
  hdr->request_id = request_id;
  hdr->more_segments = 1;
  auto* seg =
      reinterpret_cast<ProtoReadResponseExSeg*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader));
  seg->addr = remote_tensor->get_uint64_addr();
  seg->offset = response_segment_offset;
  seg->bytes = static_cast<uint32_t>(local_tensor->get_bytes() - response_segment_offset);
  seg->rkey = remote_tensor->get_rkey();

  auto status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, response);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  auto endpoint = channel->get_rdma_endpoint(local_dev_name, remote_dev_name);
  REQUIRE(endpoint != nullptr);
  {
    absl::MutexLock lock(&endpoint->mu);
    CHECK(endpoint->state == Channel::HandshakeState::kConnectRequested);
    CHECK(endpoint->pending_reads.size() == 1);
    CHECK(endpoint->pending_reads.front().request == read_request);
  }
  CHECK(read_request->expected_completions_.load() == 1);

  CommunicatorTestPeer::pending_requests(client).erase_if_present(read_request->get_key());
  if (CommunicatorTestPeer::channels(client).exist(read_request->get_dst_url())) {
    CommunicatorTestPeer::channels(client).del(read_request->get_dst_url());
  }
  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(client);

  auto free_status = tensorcast::cuda::free(local_gpu_buffer);
  REQUIRE(free_status.ok());
}

TEST_CASE("RDMA connect failure cleanup tolerates missing pending request", "[rdma][communicator][handshake]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
  using tensorcast::communicator::misc::STRNCPY;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator client(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(client)) {
    CommunicatorTestPeer::stop_workers(client);
    SUCCEED("Skipping RDMA cleanup test: no RDMA net devices available");
    return;
  }

  auto& rdma_ctx = CommunicatorTestPeer::rdma_context(client);
  auto net_dev = rdma_ctx->get_best_dev(/*gpu_id=*/0);
  REQUIRE(net_dev != nullptr);
  const std::string local_dev_name = net_dev->get_name();
  const std::string remote_dev_name = "peer.nic.cleanup";

  constexpr size_t local_bytes = 128;
  uint8_t* local_gpu_buffer = nullptr;
  auto alloc_status = tensorcast::cuda::malloc(reinterpret_cast<void**>(&local_gpu_buffer), local_bytes);
  REQUIRE(alloc_status.ok());

  auto local_tensor = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "rdma_cleanup_tensor",
      reinterpret_cast<uint64_t>(local_gpu_buffer),
      static_cast<uint64_t>(local_bytes),
      COMMUNICATE_ENGINE_DEV_GPU,
      net_dev);
  local_tensor->set_device_id(0);
  local_tensor->register_mr(net_dev.get());
  local_tensor->set_read_ready();

  auto read_request = std::make_shared<tensorcast::communicator::transport::ReadRequest>(
      "rdma_cleanup_tensor", "127.0.0.1", 65002, local_tensor, /*remote_offset=*/0, /*request_id=*/17,
      net_dev->get_rail_id());
  auto remote_tensor = std::make_shared<tensorcast::communicator::transport::RemotePartitionTensor>(
      "rdma_cleanup_tensor", remote_dev_name, /*addr=*/0xBEE0, local_tensor->get_bytes(), /*rkey=*/0xBEEF);
  read_request->set_remote_tensor(remote_tensor);

  CommunicatorTestPeer::pending_requests(client).put(read_request->get_key(), read_request);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65002);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);

  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/2,
      /*max_window_segments=*/2);
  CommunicatorTestPeer::channels(client).put(read_request->get_dst_url(), channel);

  const uint32_t payload_size = sizeof(ProtoReadResponseExHeader) + sizeof(ProtoReadResponseExSeg);
  auto response = std::make_shared<EngineMessage>(ENGINE_OP_READ_RESPONSE_EX, payload_size);
  auto* hdr = response->get_payload<ProtoReadResponseExHeader>();
  STRNCPY(hdr->tensor_key, "rdma_cleanup_tensor", kMaxTensorNameLen);
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->staged = 1;
  STRNCPY(hdr->nic_name, remote_dev_name, kMaxDevName);
  hdr->request_offset = 0;
  hdr->request_id = 17;
  hdr->num_segments = 1;
  hdr->window_seq = 0;
  hdr->credit_granted = 1;
  hdr->more_segments = 0;
  auto* seg =
      reinterpret_cast<ProtoReadResponseExSeg*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader));
  seg->addr = remote_tensor->get_uint64_addr();
  seg->offset = 0;
  seg->bytes = static_cast<uint32_t>(local_tensor->get_bytes());
  seg->rkey = remote_tensor->get_rkey();

  auto status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, response);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  auto endpoint = channel->get_rdma_endpoint(local_dev_name, remote_dev_name);
  REQUIRE(endpoint != nullptr);
  {
    absl::MutexLock lock(&endpoint->mu);
    REQUIRE(endpoint->state == Channel::HandshakeState::kConnectRequested);
    REQUIRE(endpoint->pending_reads.size() == 1);
  }

  // Simulate a concurrent cleanup racing ahead of the connect-failed handler.
  REQUIRE(CommunicatorTestPeer::pending_requests(client).erase_if_present(read_request->get_key()));
  REQUIRE_FALSE(CommunicatorTestPeer::pending_requests(client).exist(read_request->get_key()));

  auto connect_failed = EngineMessage::make_message<ProtoRdmaConnectFailed>(ENGINE_OP_RDMA_CONNECT_FAILED);
  auto* fail_payload = connect_failed->get_payload<ProtoRdmaConnectFailed>();
  STRNCPY(fail_payload->src_dev_name, local_dev_name, kMaxDevName);
  STRNCPY(fail_payload->dst_dev_name, remote_dev_name, kMaxDevName);

  status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, connect_failed);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  CHECK(read_request->is_result_set());
  CHECK(absl::IsUnavailable(read_request->status_.status));
  {
    absl::MutexLock lock(&endpoint->mu);
    CHECK(endpoint->state == Channel::HandshakeState::kFailed);
    CHECK(endpoint->pending_reads.empty());
  }

  if (CommunicatorTestPeer::channels(client).exist(read_request->get_dst_url())) {
    CommunicatorTestPeer::channels(client).del(read_request->get_dst_url());
  }
  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(client);

  auto free_status = tensorcast::cuda::free(local_gpu_buffer);
  REQUIRE(free_status.ok());
}

} // namespace tensorcast::unittests
