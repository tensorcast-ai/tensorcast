
// Copyright (c) 2025-2026, TensorCast Team.

#include <arpa/inet.h>
#include <catch2/catch_test_macros.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <array>
#include <cerrno>
#include <functional>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/time/time.h"
#include "core/communicator/engine/engine.h"

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "core/communicator/misc/utils.h"
#include "core/cuda/cuda_api.h"
#include "core/testing/test_helpers.h"

namespace tensorcast::communicator::engine {

class CommunicatorTestPeer {
 public:
  using LazySourceMrTestEvent = Communicator::LazySourceMrTestEvent;
  using LazySourceMrTestHook = std::function<void(LazySourceMrTestEvent, std::string_view, int16_t)>;

  static auto& rdma_context(Communicator& communicator) {
    return communicator.rdma_context_;
  }

  static auto& pending_requests(Communicator& communicator) {
    return communicator.pending_requests_;
  }

  static auto& channels(Communicator& communicator) {
    return communicator.channels_;
  }

  static auto stable_local_backing_state(Communicator& communicator, const std::string& backing_id) {
    absl::MutexLock lock(&communicator.stable_local_backings_mu_);
    auto it = communicator.stable_local_backings_.find(backing_id);
    if (it == communicator.stable_local_backings_.end()) {
      return std::shared_ptr<Communicator::StableLocalBackingState>{};
    }
    return it->second;
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
    communicator.read_plan_admission_queue_.stop();
    communicator.read_plan_execution_queue_.stop();
    communicator.stable_local_prewarm_queue_.stop();
  }

  static bool has_rdma_device(Communicator& communicator) {
    if (communicator.rdma_context_ == nullptr) {
      return false;
    }
    return !communicator.rdma_context_->list_devs().empty();
  }

  static bool stable_local_backing_active(Communicator& communicator, const std::string& backing_id) {
    return communicator.stable_local_backing_active_for_test(backing_id);
  }

  static size_t stable_local_backing_chunk_count(
      Communicator& communicator,
      const std::string& backing_id,
      int16_t rail_id) {
    return communicator.stable_local_backing_chunk_count_for_test(backing_id, rail_id);
  }

  static bool stable_local_backing_prewarm_complete(Communicator& communicator, const std::string& backing_id) {
    return communicator.stable_local_backing_prewarm_complete_for_test(backing_id);
  }

  static bool wait_for_stable_local_backing_prewarm(
      Communicator& communicator,
      const std::string& backing_id,
      absl::Duration timeout) {
    return communicator.wait_for_stable_local_backing_prewarm_for_test(backing_id, timeout);
  }

  static auto prepare_read_plan(
      Communicator& communicator,
      const routing::ReadPlan& plan,
      uint64_t request_id,
      std::string_view tensor_key) {
    return communicator.prepare_read_plan(plan, request_id, tensor_key);
  }

  static uint32_t read_plan_admission_workers(Communicator& communicator) {
    return communicator.read_plan_admission_workers_;
  }

  static uint32_t read_plan_direct_source_workers(Communicator& communicator) {
    return communicator.read_plan_direct_source_workers_;
  }

  static auto ensure_tensor_registered_on_dev(
      Communicator& communicator,
      const std::shared_ptr<transport::PartitionTensor>& tensor,
      const transport::net_dev_t& dev) {
    return communicator.ensure_tensor_registered_on_dev(tensor, dev);
  }

  static void set_lazy_source_mr_test_hook(LazySourceMrTestHook hook) {
    communicator::engine::Communicator::set_lazy_source_mr_test_hook_for_test(std::move(hook));
  }

  static void clear_lazy_source_mr_test_hook() {
    communicator::engine::Communicator::clear_lazy_source_mr_test_hook_for_test();
  }
};

} // namespace tensorcast::communicator::engine

namespace tensorcast::communicator::transport {

class RdmaTransportTestPeer {
 public:
  static void set_post_send_hook(RdmaTransport::PostSendHook hook) {
    RdmaTransport::post_send_hook_for_tests() = std::move(hook);
  }

  static void clear_post_send_hook() {
    RdmaTransport::post_send_hook_for_tests() = nullptr;
  }

  static int qp_count(const RdmaTransport& transport) {
    return transport.qp_count_;
  }

  static int inflight_queue_size(RdmaTransport& transport, int qp_index) {
    return transport.per_qp_inflight_queues_[qp_index].size();
  }
};

} // namespace tensorcast::communicator::transport

namespace tensorcast::unittests {

#define BUF_SIZE 65536
#define CPU_KEY "RDMA_TENSOR_KEY_CPU"
#define GPU_KEY "RDMA_TENSOR_KEY_GPU"

void require_recv_timeout_or_env_restriction(int fd, const timeval& timeout) {
  errno = 0;
  const int rc = ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  REQUIRE((rc == 0 || errno == EPERM));
}

class ScopedPostSendHook {
 public:
  explicit ScopedPostSendHook(std::function<int(struct ibv_qp*, struct ibv_send_wr*, struct ibv_send_wr**)> hook) {
    tensorcast::communicator::transport::RdmaTransportTestPeer::set_post_send_hook(std::move(hook));
  }

  ~ScopedPostSendHook() {
    tensorcast::communicator::transport::RdmaTransportTestPeer::clear_post_send_hook();
  }

  ScopedPostSendHook(const ScopedPostSendHook&) = delete;
  ScopedPostSendHook& operator=(const ScopedPostSendHook&) = delete;
};

class ScopedLazySourceMrHook {
 public:
  explicit ScopedLazySourceMrHook(
      std::function<void(communicator::engine::CommunicatorTestPeer::LazySourceMrTestEvent, std::string_view, int16_t)>
          hook) {
    communicator::engine::CommunicatorTestPeer::set_lazy_source_mr_test_hook(std::move(hook));
  }

  ~ScopedLazySourceMrHook() {
    communicator::engine::CommunicatorTestPeer::clear_lazy_source_mr_test_hook();
  }

  ScopedLazySourceMrHook(const ScopedLazySourceMrHook&) = delete;
  ScopedLazySourceMrHook& operator=(const ScopedLazySourceMrHook&) = delete;
};

class DummyAckStager : public communicator::engine::MemoryStager {
 public:
  absl::StatusOr<void*> stage(
      const std::shared_ptr<communicator::transport::PartitionTensor>& /*tensor*/,
      uint64_t /*offset*/,
      uint64_t /*bytes*/,
      StageMode /*mode*/ = StageMode::kBlocking) override {
    return absl::UnimplementedError("DummyAckStager stage not used");
  }

  absl::Status release_staged_buffer(gsl::not_null<void*> exposed_ptr) override {
    released_ptrs_.push_back(exposed_ptr);
    return absl::OkStatus();
  }

  size_t get_chunk_size() const override {
    return 1;
  }

  size_t get_num_buffers() const override {
    return 1;
  }

  std::vector<void*> released_ptrs_;
};

struct AckLeaseFactory {
  std::shared_ptr<DummyAckStager> stager = std::make_shared<DummyAckStager>();
  int credit_released = 0;

  communicator::engine::StageLease make(
      communicator::engine::FlowCreditLedger& ledger,
      void* ptr,
      size_t bytes,
      const std::string& request_key,
      uint32_t window_seq,
      uint32_t segment_idx,
      uint64_t offset) {
    communicator::engine::StageLease::Metadata metadata;
    metadata.transport = communicator::engine::StageTransport::kRdma;
    metadata.request_key = request_key;
    metadata.window_seq = window_seq;
    metadata.segment_idx = segment_idx;
    metadata.offset = offset;
    metadata.bytes = bytes;
    return communicator::engine::StageLease(
        stager,
        &ledger,
        ptr,
        bytes,
        /*mr=*/nullptr,
        /*deregister_mr=*/false,
        metadata,
        [this]() { ++credit_released; });
  }
};

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
  CHECK(CommunicatorTestPeer::pending_requests(client).exist(read_request->get_key()));
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
  require_recv_timeout_or_env_restriction(sv[1], timeout);

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
      "rdma_cleanup_tensor",
      "127.0.0.1",
      65002,
      local_tensor,
      /*remote_offset=*/0,
      /*request_id=*/17,
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

TEST_CASE("READ_PLAN_REQUEST missing tensor sends READ_PLAN_FAILED", "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;

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
  require_recv_timeout_or_env_restriction(sv[1], timeout);

  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65011);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/2,
      /*max_window_segments=*/2);

  const uint32_t payload_size = sizeof(ProtoReadPlanRequestHeader) + sizeof(ProtoReadPlanSourceSlice);
  auto request = std::make_shared<EngineMessage>(ENGINE_OP_READ_PLAN_REQUEST, payload_size);
  auto* hdr = request->get_payload<ProtoReadPlanRequestHeader>();
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->rail_id = -1;
  hdr->request_id = 91;
  hdr->num_source_slices = 1;
  auto* slice =
      reinterpret_cast<ProtoReadPlanSourceSlice*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanRequestHeader));
  communicator::misc::STRNCPY(slice->tensor_key, "missing-plan-tensor", kMaxTensorNameLen);
  slice->source_slice_index = 0;
  slice->remote_offset = 0;
  slice->bytes = 64;

  auto status = CommunicatorTestPeer::on_receive_request(server, channel, control_transport, request);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  ProtoHeader response_header{};
  REQUIRE(
      ::recv(sv[1], &response_header, sizeof(response_header), MSG_WAITALL) ==
      static_cast<ssize_t>(sizeof(response_header)));
  REQUIRE(response_header.op == ENGINE_OP_READ_PLAN_FAILED);
  REQUIRE(response_header.size == sizeof(ProtoReadPlanFailed));

  ProtoReadPlanFailed response{};
  REQUIRE(::recv(sv[1], &response, sizeof(response), MSG_WAITALL) == static_cast<ssize_t>(sizeof(response)));
  REQUIRE(response.request_id == 91);
  REQUIRE(response.reason == TENSORCAST_READ_FAILED_NO_TENSOR);

  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(server);
}

TEST_CASE("read-plan worker auto sizing follows communicator defaults", "[rdma][communicator][read_plan][config]") {
  {
    auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/false);
    auto pools = tensorcast::testing::make_test_pinned_staging_pools(
        cfg.stager().buffers_per_flow(),
        cfg.transport().tcp_conn_count(),
        /*gpu_slice_bytes=*/(16ULL << 20),
        /*cpu_slice_bytes=*/(4ULL << 20),
        /*enable_rdma=*/false);
    Communicator communicator(cfg, std::move(pools), /*channel_expire_sec=*/0);
    REQUIRE(CommunicatorTestPeer::read_plan_admission_workers(communicator) == 1);
    REQUIRE(CommunicatorTestPeer::read_plan_direct_source_workers(communicator) == 1);
    CommunicatorTestPeer::stop_workers(communicator);
  }

  {
    auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
    auto pools = tensorcast::testing::make_test_pinned_staging_pools(
        cfg.stager().buffers_per_flow(),
        cfg.transport().tcp_conn_count(),
        /*gpu_slice_bytes=*/(16ULL << 20),
        /*cpu_slice_bytes=*/(4ULL << 20),
        /*enable_rdma=*/true);
    Communicator communicator(cfg, std::move(pools), /*channel_expire_sec=*/0);
    const size_t visible_rail_count = CommunicatorTestPeer::rdma_context(communicator)->list_devs().size();
    REQUIRE(
        CommunicatorTestPeer::read_plan_admission_workers(communicator) ==
        static_cast<uint32_t>(std::min<size_t>(4, std::max<size_t>(2, visible_rail_count))));
    REQUIRE(
        CommunicatorTestPeer::read_plan_direct_source_workers(communicator) ==
        static_cast<uint32_t>(std::min<size_t>(4, std::max<size_t>(1, visible_rail_count))));
    CommunicatorTestPeer::stop_workers(communicator);
  }
}

TEST_CASE("read-plan worker knobs honor explicit configuration", "[rdma][communicator][read_plan][config]") {
  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  cfg.mutable_rdma()->set_read_plan_admission_workers(3);
  cfg.mutable_rdma()->set_read_plan_direct_source_workers(4);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator communicator(cfg, std::move(pools), /*channel_expire_sec=*/0);
  REQUIRE(CommunicatorTestPeer::read_plan_admission_workers(communicator) == 3);
  REQUIRE(CommunicatorTestPeer::read_plan_direct_source_workers(communicator) == 4);
  CommunicatorTestPeer::stop_workers(communicator);
}

TEST_CASE(
    "READ_PLAN_REQUEST reuses one session rail for CPU source slices with different preferred rails",
    "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator server(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(server)) {
    CommunicatorTestPeer::stop_workers(server);
    SUCCEED("Skipping CPU multi-rail READ_PLAN_REQUEST test: no RDMA net devices available");
    return;
  }

  auto& rdma_ctx = CommunicatorTestPeer::rdma_context(server);
  const auto& devs = rdma_ctx->list_devs();
  if (devs.size() < 2) {
    CommunicatorTestPeer::stop_workers(server);
    SUCCEED("Skipping CPU multi-rail READ_PLAN_REQUEST test: fewer than two RDMA rails available");
    return;
  }

  std::string first_key;
  std::string second_key;
  tensorcast::communicator::transport::net_dev_t first_dev;
  tensorcast::communicator::transport::net_dev_t second_dev;
  for (int index = 0; index < 256 && second_dev == nullptr; ++index) {
    const std::string key = absl::StrCat("plan-cpu-rail-", index);
    auto dev = rdma_ctx->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, key);
    REQUIRE(dev != nullptr);
    if (first_dev == nullptr) {
      first_key = key;
      first_dev = dev;
      continue;
    }
    if (dev->get_name() != first_dev->get_name() || dev->get_rail_id() != first_dev->get_rail_id()) {
      second_key = key;
      second_dev = dev;
      break;
    }
  }
  if (second_dev == nullptr) {
    CommunicatorTestPeer::stop_workers(server);
    SUCCEED("Skipping CPU multi-rail READ_PLAN_REQUEST test: could not find two keys on different RDMA rails");
    return;
  }

  std::array<uint8_t, 64> buffer0{};
  std::array<uint8_t, 64> buffer1{};
  Communicator::RegisterTensorOptions opts;
  opts.register_mr = false;
  opts.needs_staging = false;
  opts.async = false;
  REQUIRE(server
              .register_tensor_ex(
                  first_key,
                  reinterpret_cast<uint64_t>(buffer0.data()),
                  static_cast<uint64_t>(buffer0.size()),
                  COMMUNICATE_ENGINE_DEV_CPU,
                  -1,
                  opts)
              .ok());
  REQUIRE(server
              .register_tensor_ex(
                  second_key,
                  reinterpret_cast<uint64_t>(buffer1.data()),
                  static_cast<uint64_t>(buffer1.size()),
                  COMMUNICATE_ENGINE_DEV_CPU,
                  -1,
                  opts)
              .ok());

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  timeval timeout{};
  timeout.tv_sec = 2;
  require_recv_timeout_or_env_restriction(sv[1], timeout);

  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65012);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/2,
      /*max_window_segments=*/2);

  const uint32_t payload_size = sizeof(ProtoReadPlanRequestHeader) + 2 * sizeof(ProtoReadPlanSourceSlice);
  auto request = std::make_shared<EngineMessage>(ENGINE_OP_READ_PLAN_REQUEST, payload_size);
  auto* hdr = request->get_payload<ProtoReadPlanRequestHeader>();
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->rail_id = -1;
  hdr->request_id = 92;
  hdr->num_source_slices = 2;
  auto* slices =
      reinterpret_cast<ProtoReadPlanSourceSlice*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanRequestHeader));
  communicator::misc::STRNCPY(slices[0].tensor_key, first_key, kMaxTensorNameLen);
  slices[0].source_slice_index = 0;
  slices[0].remote_offset = 0;
  slices[0].bytes = 16;
  communicator::misc::STRNCPY(slices[1].tensor_key, second_key, kMaxTensorNameLen);
  slices[1].source_slice_index = 1;
  slices[1].remote_offset = 0;
  slices[1].bytes = 16;

  auto status = CommunicatorTestPeer::on_receive_request(server, channel, control_transport, request);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  ProtoHeader response_header{};
  REQUIRE(
      ::recv(sv[1], &response_header, sizeof(response_header), MSG_WAITALL) ==
      static_cast<ssize_t>(sizeof(response_header)));
  REQUIRE(response_header.op == ENGINE_OP_READ_PLAN_RESPONSE_EX);
  REQUIRE(response_header.size >= sizeof(ProtoReadPlanResponseExHeader));

  std::vector<char> payload(response_header.size);
  REQUIRE(::recv(sv[1], payload.data(), payload.size(), MSG_WAITALL) == static_cast<ssize_t>(payload.size()));
  auto* response = reinterpret_cast<const ProtoReadPlanResponseExHeader*>(payload.data());
  REQUIRE(response->request_id == 92);
  REQUIRE(response->rail_id == first_dev->get_rail_id());
  REQUIRE(response->num_segments == 2);
  const auto* segs = reinterpret_cast<const ProtoReadPlanResponseExSeg*>(
      reinterpret_cast<const uint8_t*>(response) + sizeof(ProtoReadPlanResponseExHeader));
  REQUIRE(segs[0].source_slice_index == 0);
  REQUIRE(segs[1].source_slice_index == 1);

  (void)server.unregister_tensor(first_key);
  (void)server.unregister_tensor(second_key);
  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(server);
}

TEST_CASE(
    "READ_PLAN_REQUEST CPU direct source bypasses staged credit and emits one zero-copy window",
    "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator server(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(server)) {
    CommunicatorTestPeer::stop_workers(server);
    SUCCEED("Skipping CPU direct-source READ_PLAN_REQUEST test: no RDMA net devices available");
    return;
  }

  auto& rdma_ctx = CommunicatorTestPeer::rdma_context(server);
  std::string first_key;
  std::string second_key;
  tensorcast::communicator::transport::net_dev_t first_dev;
  for (int index = 0; index < 256 && second_key.empty(); ++index) {
    const std::string key = absl::StrCat("plan-cpu-direct-source-", index);
    auto dev = rdma_ctx->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, key);
    REQUIRE(dev != nullptr);
    if (first_dev == nullptr) {
      first_key = key;
      first_dev = dev;
      continue;
    }
    if (dev->get_name() == first_dev->get_name() && dev->get_rail_id() == first_dev->get_rail_id()) {
      second_key = key;
      break;
    }
  }
  if (second_key.empty()) {
    CommunicatorTestPeer::stop_workers(server);
    SUCCEED("Skipping CPU direct-source READ_PLAN_REQUEST test: could not find two keys on the same RDMA rail");
    return;
  }

  std::array<uint8_t, 64> buffer0{};
  std::array<uint8_t, 64> buffer1{};
  Communicator::RegisterTensorOptions opts;
  opts.register_mr = true;
  opts.needs_staging = false;
  opts.async = false;
  opts.direct_rdma_enabled = true;
  REQUIRE(server
              .register_tensor_ex(
                  first_key,
                  reinterpret_cast<uint64_t>(buffer0.data()),
                  static_cast<uint64_t>(buffer0.size()),
                  COMMUNICATE_ENGINE_DEV_CPU,
                  -1,
                  opts)
              .ok());
  REQUIRE(server
              .register_tensor_ex(
                  second_key,
                  reinterpret_cast<uint64_t>(buffer1.data()),
                  static_cast<uint64_t>(buffer1.size()),
                  COMMUNICATE_ENGINE_DEV_CPU,
                  -1,
                  opts)
              .ok());

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  timeval timeout{};
  timeout.tv_sec = 2;
  require_recv_timeout_or_env_restriction(sv[1], timeout);

  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65013);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/1,
      /*max_window_segments=*/1);
  auto flow_state = channel->flow_state();
  REQUIRE(flow_state != nullptr);
  REQUIRE(flow_state->registry.size() == 0);
  REQUIRE(flow_state->ledger.outstanding_credit() == 0);

  const uint32_t payload_size = sizeof(ProtoReadPlanRequestHeader) + 2 * sizeof(ProtoReadPlanSourceSlice);
  auto request = std::make_shared<EngineMessage>(ENGINE_OP_READ_PLAN_REQUEST, payload_size);
  auto* hdr = request->get_payload<ProtoReadPlanRequestHeader>();
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->rail_id = -1;
  hdr->request_id = 93;
  hdr->num_source_slices = 2;
  auto* slices =
      reinterpret_cast<ProtoReadPlanSourceSlice*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanRequestHeader));
  communicator::misc::STRNCPY(slices[0].tensor_key, first_key, kMaxTensorNameLen);
  slices[0].source_slice_index = 0;
  slices[0].remote_offset = 0;
  slices[0].bytes = static_cast<uint64_t>(buffer0.size());
  communicator::misc::STRNCPY(slices[1].tensor_key, second_key, kMaxTensorNameLen);
  slices[1].source_slice_index = 1;
  slices[1].remote_offset = 0;
  slices[1].bytes = static_cast<uint64_t>(buffer1.size());

  auto status = CommunicatorTestPeer::on_receive_request(server, channel, control_transport, request);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  ProtoHeader response_header{};
  REQUIRE(
      ::recv(sv[1], &response_header, sizeof(response_header), MSG_WAITALL) ==
      static_cast<ssize_t>(sizeof(response_header)));
  REQUIRE(response_header.op == ENGINE_OP_READ_PLAN_RESPONSE_EX);
  REQUIRE(response_header.size >= sizeof(ProtoReadPlanResponseExHeader));

  std::vector<char> payload(response_header.size);
  REQUIRE(::recv(sv[1], payload.data(), payload.size(), MSG_WAITALL) == static_cast<ssize_t>(payload.size()));
  auto* response = reinterpret_cast<const ProtoReadPlanResponseExHeader*>(payload.data());
  REQUIRE(response->request_id == 93);
  REQUIRE(response->staged == 0);
  REQUIRE(response->zero_copy == 1);
  REQUIRE(response->rail_id == first_dev->get_rail_id());
  REQUIRE(response->num_segments == 2);
  REQUIRE(response->credit_granted == 2);
  REQUIRE(response->more_segments == 0);
  REQUIRE(flow_state->registry.size() == 0);
  REQUIRE(flow_state->ledger.outstanding_credit() == 0);

  const auto* segs = reinterpret_cast<const ProtoReadPlanResponseExSeg*>(
      reinterpret_cast<const uint8_t*>(response) + sizeof(ProtoReadPlanResponseExHeader));
  REQUIRE(segs[0].source_slice_index == 0);
  REQUIRE(segs[0].source_slice_offset == 0);
  REQUIRE(segs[0].addr == reinterpret_cast<uint64_t>(buffer0.data()));
  REQUIRE(segs[0].bytes == static_cast<uint32_t>(buffer0.size()));
  REQUIRE(segs[1].source_slice_index == 1);
  REQUIRE(segs[1].source_slice_offset == 0);
  REQUIRE(segs[1].addr == reinterpret_cast<uint64_t>(buffer1.data()));
  REQUIRE(segs[1].bytes == static_cast<uint32_t>(buffer1.size()));

  (void)server.unregister_tensor(first_key);
  (void)server.unregister_tensor(second_key);
  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(server);
}

TEST_CASE(
    "READ_PLAN_REQUEST CPU direct source lazily registers source MR on the session rail",
    "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator server(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(server)) {
    CommunicatorTestPeer::stop_workers(server);
    SUCCEED("Skipping CPU lazy-MR READ_PLAN_REQUEST test: no RDMA net devices available");
    return;
  }

  auto& rdma_ctx = CommunicatorTestPeer::rdma_context(server);
  std::string first_key;
  std::string second_key;
  tensorcast::communicator::transport::net_dev_t first_dev;
  for (int index = 0; index < 512 && second_key.empty(); ++index) {
    const std::string key = absl::StrCat("plan-cpu-direct-lazy-", index);
    auto dev = rdma_ctx->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, key);
    REQUIRE(dev != nullptr);
    if (first_dev == nullptr) {
      first_key = key;
      first_dev = dev;
      continue;
    }
    if (dev->get_name() != first_dev->get_name() || dev->get_rail_id() != first_dev->get_rail_id()) {
      second_key = key;
      break;
    }
  }
  if (second_key.empty()) {
    CommunicatorTestPeer::stop_workers(server);
    SUCCEED("Skipping CPU lazy-MR READ_PLAN_REQUEST test: could not find two keys on different RDMA rails");
    return;
  }

  std::array<uint8_t, 64> buffer0{};
  std::array<uint8_t, 64> buffer1{};
  Communicator::RegisterTensorOptions opts;
  opts.register_mr = true;
  opts.needs_staging = false;
  opts.async = false;
  opts.direct_rdma_enabled = true;
  REQUIRE(server
              .register_tensor_ex(
                  first_key,
                  reinterpret_cast<uint64_t>(buffer0.data()),
                  static_cast<uint64_t>(buffer0.size()),
                  COMMUNICATE_ENGINE_DEV_CPU,
                  -1,
                  opts)
              .ok());
  REQUIRE(server
              .register_tensor_ex(
                  second_key,
                  reinterpret_cast<uint64_t>(buffer1.data()),
                  static_cast<uint64_t>(buffer1.size()),
                  COMMUNICATE_ENGINE_DEV_CPU,
                  -1,
                  opts)
              .ok());

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  timeval timeout{};
  timeout.tv_sec = 2;
  require_recv_timeout_or_env_restriction(sv[1], timeout);

  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65014);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/1,
      /*max_window_segments=*/1);
  auto flow_state = channel->flow_state();
  REQUIRE(flow_state != nullptr);

  const uint32_t payload_size = sizeof(ProtoReadPlanRequestHeader) + 2 * sizeof(ProtoReadPlanSourceSlice);
  auto request = std::make_shared<EngineMessage>(ENGINE_OP_READ_PLAN_REQUEST, payload_size);
  auto* hdr = request->get_payload<ProtoReadPlanRequestHeader>();
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->rail_id = first_dev->get_rail_id();
  hdr->request_id = 94;
  hdr->num_source_slices = 2;
  auto* slices =
      reinterpret_cast<ProtoReadPlanSourceSlice*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanRequestHeader));
  communicator::misc::STRNCPY(slices[0].tensor_key, first_key, kMaxTensorNameLen);
  slices[0].source_slice_index = 0;
  slices[0].remote_offset = 0;
  slices[0].bytes = static_cast<uint64_t>(buffer0.size());
  communicator::misc::STRNCPY(slices[1].tensor_key, second_key, kMaxTensorNameLen);
  slices[1].source_slice_index = 1;
  slices[1].remote_offset = 0;
  slices[1].bytes = static_cast<uint64_t>(buffer1.size());

  auto status = CommunicatorTestPeer::on_receive_request(server, channel, control_transport, request);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  ProtoHeader response_header{};
  REQUIRE(
      ::recv(sv[1], &response_header, sizeof(response_header), MSG_WAITALL) ==
      static_cast<ssize_t>(sizeof(response_header)));
  REQUIRE(response_header.op == ENGINE_OP_READ_PLAN_RESPONSE_EX);
  REQUIRE(response_header.size >= sizeof(ProtoReadPlanResponseExHeader));

  std::vector<char> payload(response_header.size);
  REQUIRE(::recv(sv[1], payload.data(), payload.size(), MSG_WAITALL) == static_cast<ssize_t>(payload.size()));
  auto* response = reinterpret_cast<const ProtoReadPlanResponseExHeader*>(payload.data());
  REQUIRE(response->request_id == 94);
  REQUIRE(response->staged == 0);
  REQUIRE(response->zero_copy == 1);
  REQUIRE(response->rail_id == first_dev->get_rail_id());
  REQUIRE(response->num_segments == 2);
  REQUIRE(response->credit_granted == 2);
  REQUIRE(response->more_segments == 0);
  REQUIRE(flow_state->registry.size() == 0);
  REQUIRE(flow_state->ledger.outstanding_credit() == 0);

  const auto* segs = reinterpret_cast<const ProtoReadPlanResponseExSeg*>(
      reinterpret_cast<const uint8_t*>(response) + sizeof(ProtoReadPlanResponseExHeader));
  REQUIRE(segs[0].source_slice_index == 0);
  REQUIRE(segs[0].addr == reinterpret_cast<uint64_t>(buffer0.data()));
  REQUIRE(segs[0].rkey != 0);
  REQUIRE(segs[1].source_slice_index == 1);
  REQUIRE(segs[1].addr == reinterpret_cast<uint64_t>(buffer1.data()));
  REQUIRE(segs[1].rkey != 0);

  (void)server.unregister_tensor(first_key);
  (void)server.unregister_tensor(second_key);
  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(server);
}

TEST_CASE(
    "READ_PLAN_REQUEST stable-backed CPU source uses requested-rail chunk MR without raw source registration",
    "[rdma][communicator][read_plan][stable_backing]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  cfg.mutable_rdma()->set_enable_stable_local_mr_reuse(true);
  cfg.mutable_rdma()->set_stable_local_mr_reuse_chunk_slots(2);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator server(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(server)) {
    CommunicatorTestPeer::stop_workers(server);
    SUCCEED("Skipping stable-backed source READ_PLAN_REQUEST test: no RDMA net devices available");
    return;
  }

  auto net_dev = CommunicatorTestPeer::rdma_context(server)->get_best_dev(
      COMMUNICATE_ENGINE_DEV_CPU, -1, -1, "stable-source-view");
  REQUIRE(net_dev != nullptr);

  std::array<uint8_t, 128> buffer{};
  const uint64_t slot_addr = reinterpret_cast<uint64_t>(buffer.data()) + 64;
  tensorcast::store::StableLocalBackingRef backing{
      .kind = tensorcast::store::StableLocalBackingKind::kHostSharedRegion,
      .backing_id = "region:test-stable-source-view",
      .backing_base_addr = reinterpret_cast<uint64_t>(buffer.data()),
      .backing_bytes = static_cast<uint64_t>(buffer.size()),
      .slot_bytes = 64,
      .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
      .dev_id = 0,
  };
  REQUIRE(server.activate_stable_local_backing(backing, std::make_shared<int>(1)).ok());
  REQUIRE(
      CommunicatorTestPeer::stable_local_backing_chunk_count(server, backing.backing_id, net_dev->get_rail_id()) == 0);

  const std::string view_key = "stable-source-view-key";
  Communicator::StableLocalBackingSourceView source_view{
      .tensor_key = view_key,
      .addr = slot_addr,
      .bytes = 64,
      .backing = backing,
      .keepalive = std::make_shared<int>(2),
  };
  REQUIRE(server.register_stable_local_backing_source_view(source_view).ok());

  int raw_lazy_events = 0;
  ScopedLazySourceMrHook hook([&](communicator::engine::CommunicatorTestPeer::LazySourceMrTestEvent /*event*/,
                                  std::string_view tensor_key,
                                  int16_t /*rail_id*/) {
    if (tensor_key == view_key) {
      ++raw_lazy_events;
    }
  });

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  timeval timeout{};
  timeout.tv_sec = 2;
  require_recv_timeout_or_env_restriction(sv[1], timeout);

  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65016);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/1,
      /*max_window_segments=*/1);
  auto flow_state = channel->flow_state();
  REQUIRE(flow_state != nullptr);

  const uint32_t payload_size = sizeof(ProtoReadPlanRequestHeader) + sizeof(ProtoReadPlanSourceSlice);
  auto request = std::make_shared<EngineMessage>(ENGINE_OP_READ_PLAN_REQUEST, payload_size);
  auto* hdr = request->get_payload<ProtoReadPlanRequestHeader>();
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->rail_id = net_dev->get_rail_id();
  hdr->request_id = 97;
  hdr->num_source_slices = 1;
  auto* slice =
      reinterpret_cast<ProtoReadPlanSourceSlice*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanRequestHeader));
  communicator::misc::STRNCPY(slice->tensor_key, view_key, kMaxTensorNameLen);
  slice->source_slice_index = 0;
  slice->remote_offset = 0;
  slice->bytes = 64;

  auto status = CommunicatorTestPeer::on_receive_request(server, channel, control_transport, request);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  ProtoHeader response_header{};
  REQUIRE(
      ::recv(sv[1], &response_header, sizeof(response_header), MSG_WAITALL) ==
      static_cast<ssize_t>(sizeof(response_header)));
  REQUIRE(response_header.op == ENGINE_OP_READ_PLAN_RESPONSE_EX);
  REQUIRE(response_header.size >= sizeof(ProtoReadPlanResponseExHeader));

  std::vector<char> payload(response_header.size);
  REQUIRE(::recv(sv[1], payload.data(), payload.size(), MSG_WAITALL) == static_cast<ssize_t>(payload.size()));
  auto* response = reinterpret_cast<const ProtoReadPlanResponseExHeader*>(payload.data());
  REQUIRE(response->request_id == 97);
  REQUIRE(response->staged == 0);
  REQUIRE(response->zero_copy == 1);
  REQUIRE(response->rail_id == net_dev->get_rail_id());
  REQUIRE(response->num_segments == 1);
  const auto* seg = reinterpret_cast<const ProtoReadPlanResponseExSeg*>(
      reinterpret_cast<const uint8_t*>(response) + sizeof(ProtoReadPlanResponseExHeader));
  REQUIRE(seg->source_slice_index == 0);
  REQUIRE(seg->addr == slot_addr);
  REQUIRE(seg->bytes == 64);
  REQUIRE(seg->rkey != 0);
  CHECK(raw_lazy_events == 0);
  CHECK(
      CommunicatorTestPeer::stable_local_backing_chunk_count(server, backing.backing_id, net_dev->get_rail_id()) == 1);

  REQUIRE(server.unregister_tensor(view_key).ok());
  REQUIRE(server.deactivate_stable_local_backing(backing.backing_id).ok());
  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(server);
}

TEST_CASE(
    "CPU direct-source lazy MR gate coalesces same-tensor concurrent admissions",
    "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator server(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(server)) {
    CommunicatorTestPeer::stop_workers(server);
    SUCCEED("Skipping same-tensor lazy-MR gate test: no RDMA net devices available");
    return;
  }

  auto net_dev =
      CommunicatorTestPeer::rdma_context(server)->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, "lazy-gate-same");
  REQUIRE(net_dev != nullptr);

  std::array<uint8_t, 64> buffer{};
  auto tensor = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "lazy-gate-same-tensor",
      reinterpret_cast<uint64_t>(buffer.data()),
      static_cast<uint64_t>(buffer.size()),
      COMMUNICATE_ENGINE_DEV_CPU,
      net_dev);

  absl::Mutex gate_mu;
  absl::CondVar gate_cv;
  bool owner_entered = false;
  bool waiter_joined = false;
  bool release_owner = false;
  int owner_count = 0;
  int waiter_count = 0;

  ScopedLazySourceMrHook hook([&](communicator::engine::CommunicatorTestPeer::LazySourceMrTestEvent event,
                                  std::string_view tensor_key,
                                  int16_t rail_id) {
    if (tensor_key != "lazy-gate-same-tensor" || rail_id != net_dev->get_rail_id()) {
      return;
    }
    absl::MutexLock lock(&gate_mu);
    if (event == communicator::engine::CommunicatorTestPeer::LazySourceMrTestEvent::kOwnerAcquired) {
      owner_entered = true;
      ++owner_count;
      gate_cv.SignalAll();
      while (!release_owner) {
        gate_cv.Wait(&gate_mu);
      }
      return;
    }
    if (event == communicator::engine::CommunicatorTestPeer::LazySourceMrTestEvent::kWaiterJoined) {
      waiter_joined = true;
      ++waiter_count;
      gate_cv.SignalAll();
    }
  });

  absl::Status status_owner;
  absl::Status status_waiter;
  bool owner_role = false;
  bool owner_fast_path = false;
  bool owner_waiter_role = false;
  bool waiter_role = false;
  bool waiter_fast_path = false;
  bool waiter_owner_role = false;

  std::thread owner_thread([&]() {
    auto ensure_or = CommunicatorTestPeer::ensure_tensor_registered_on_dev(server, tensor, net_dev);
    status_owner = ensure_or.status();
    if (!ensure_or.ok()) {
      return;
    }
    owner_role = ensure_or->owner;
    owner_fast_path = ensure_or->fast_path_hit;
    owner_waiter_role = ensure_or->waiter;
  });

  {
    absl::MutexLock lock(&gate_mu);
    while (!owner_entered) {
      gate_cv.Wait(&gate_mu);
    }
  }

  std::thread waiter_thread([&]() {
    auto ensure_or = CommunicatorTestPeer::ensure_tensor_registered_on_dev(server, tensor, net_dev);
    status_waiter = ensure_or.status();
    if (!ensure_or.ok()) {
      return;
    }
    waiter_role = ensure_or->waiter;
    waiter_fast_path = ensure_or->fast_path_hit;
    waiter_owner_role = ensure_or->owner;
  });

  {
    absl::MutexLock lock(&gate_mu);
    while (!waiter_joined) {
      gate_cv.Wait(&gate_mu);
    }
    release_owner = true;
    gate_cv.SignalAll();
  }

  owner_thread.join();
  waiter_thread.join();

  REQUIRE(status_owner.ok());
  REQUIRE(status_waiter.ok());
  REQUIRE(owner_role);
  REQUIRE_FALSE(owner_fast_path);
  REQUIRE_FALSE(owner_waiter_role);
  REQUIRE(waiter_role);
  REQUIRE_FALSE(waiter_fast_path);
  REQUIRE_FALSE(waiter_owner_role);
  REQUIRE(owner_count == 1);
  REQUIRE(waiter_count == 1);
  REQUIRE(tensor->has_registered_mr(net_dev));

  CommunicatorTestPeer::stop_workers(server);
}

TEST_CASE(
    "CPU direct-source lazy MR gate does not serialize different tensors behind one owner",
    "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator server(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(server)) {
    CommunicatorTestPeer::stop_workers(server);
    SUCCEED("Skipping cross-tensor lazy-MR gate test: no RDMA net devices available");
    return;
  }

  auto net_dev = CommunicatorTestPeer::rdma_context(server)->get_best_dev(
      COMMUNICATE_ENGINE_DEV_CPU, -1, -1, "lazy-gate-different");
  REQUIRE(net_dev != nullptr);

  std::array<uint8_t, 64> buffer_a{};
  std::array<uint8_t, 64> buffer_b{};
  auto tensor_a = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "lazy-gate-tensor-a",
      reinterpret_cast<uint64_t>(buffer_a.data()),
      static_cast<uint64_t>(buffer_a.size()),
      COMMUNICATE_ENGINE_DEV_CPU,
      net_dev);
  auto tensor_b = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "lazy-gate-tensor-b",
      reinterpret_cast<uint64_t>(buffer_b.data()),
      static_cast<uint64_t>(buffer_b.size()),
      COMMUNICATE_ENGINE_DEV_CPU,
      net_dev);

  absl::Mutex gate_mu;
  absl::CondVar gate_cv;
  bool tensor_a_owner_entered = false;
  bool tensor_b_owner_entered = false;
  bool release_tensor_a_owner = false;
  int tensor_a_owner_count = 0;
  int tensor_b_owner_count = 0;

  ScopedLazySourceMrHook hook([&](communicator::engine::CommunicatorTestPeer::LazySourceMrTestEvent event,
                                  std::string_view tensor_key,
                                  int16_t rail_id) {
    if (rail_id != net_dev->get_rail_id() ||
        event != communicator::engine::CommunicatorTestPeer::LazySourceMrTestEvent::kOwnerAcquired) {
      return;
    }
    absl::MutexLock lock(&gate_mu);
    if (tensor_key == "lazy-gate-tensor-a") {
      tensor_a_owner_entered = true;
      ++tensor_a_owner_count;
      gate_cv.SignalAll();
      while (!release_tensor_a_owner) {
        gate_cv.Wait(&gate_mu);
      }
      return;
    }
    if (tensor_key == "lazy-gate-tensor-b") {
      tensor_b_owner_entered = true;
      ++tensor_b_owner_count;
      gate_cv.SignalAll();
    }
  });

  absl::Status status_a;
  absl::Status status_b;
  bool tensor_a_owner_role = false;
  bool tensor_b_owner_role = false;
  bool tensor_b_waiter_role = false;

  std::thread thread_a([&]() {
    auto ensure_or = CommunicatorTestPeer::ensure_tensor_registered_on_dev(server, tensor_a, net_dev);
    status_a = ensure_or.status();
    if (!ensure_or.ok()) {
      return;
    }
    tensor_a_owner_role = ensure_or->owner;
  });

  {
    absl::MutexLock lock(&gate_mu);
    while (!tensor_a_owner_entered) {
      gate_cv.Wait(&gate_mu);
    }
  }

  std::thread thread_b([&]() {
    auto ensure_or = CommunicatorTestPeer::ensure_tensor_registered_on_dev(server, tensor_b, net_dev);
    status_b = ensure_or.status();
    if (!ensure_or.ok()) {
      return;
    }
    tensor_b_owner_role = ensure_or->owner;
    tensor_b_waiter_role = ensure_or->waiter;
  });

  {
    absl::MutexLock lock(&gate_mu);
    while (!tensor_b_owner_entered) {
      const bool timed_out = gate_cv.WaitWithTimeout(&gate_mu, absl::Seconds(2));
      REQUIRE_FALSE(timed_out);
    }
    release_tensor_a_owner = true;
    gate_cv.SignalAll();
  }

  thread_a.join();
  thread_b.join();

  REQUIRE(status_a.ok());
  REQUIRE(status_b.ok());
  REQUIRE(tensor_a_owner_role);
  REQUIRE(tensor_b_owner_role);
  REQUIRE_FALSE(tensor_b_waiter_role);
  REQUIRE(tensor_a_owner_count == 1);
  REQUIRE(tensor_b_owner_count == 1);
  REQUIRE(tensor_a->has_registered_mr(net_dev));
  REQUIRE(tensor_b->has_registered_mr(net_dev));

  CommunicatorTestPeer::stop_workers(server);
}

TEST_CASE(
    "CPU direct-source lazy MR gate bypasses coordination on MR-ready fast path",
    "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator server(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(server)) {
    CommunicatorTestPeer::stop_workers(server);
    SUCCEED("Skipping fast-path lazy-MR gate test: no RDMA net devices available");
    return;
  }

  auto net_dev =
      CommunicatorTestPeer::rdma_context(server)->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, "lazy-gate-fast");
  REQUIRE(net_dev != nullptr);

  std::array<uint8_t, 64> buffer{};
  auto tensor = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "lazy-gate-fast-tensor",
      reinterpret_cast<uint64_t>(buffer.data()),
      static_cast<uint64_t>(buffer.size()),
      COMMUNICATE_ENGINE_DEV_CPU,
      net_dev);
  tensor->register_mr(net_dev.get());
  REQUIRE(tensor->has_registered_mr(net_dev));

  int fast_path_count = 0;
  int owner_count = 0;
  int waiter_count = 0;
  ScopedLazySourceMrHook hook([&](communicator::engine::CommunicatorTestPeer::LazySourceMrTestEvent event,
                                  std::string_view tensor_key,
                                  int16_t rail_id) {
    if (tensor_key != "lazy-gate-fast-tensor" || rail_id != net_dev->get_rail_id()) {
      return;
    }
    if (event == communicator::engine::CommunicatorTestPeer::LazySourceMrTestEvent::kFastPathHit) {
      ++fast_path_count;
    } else if (event == communicator::engine::CommunicatorTestPeer::LazySourceMrTestEvent::kOwnerAcquired) {
      ++owner_count;
    } else if (event == communicator::engine::CommunicatorTestPeer::LazySourceMrTestEvent::kWaiterJoined) {
      ++waiter_count;
    }
  });

  auto ensure_or = CommunicatorTestPeer::ensure_tensor_registered_on_dev(server, tensor, net_dev);
  REQUIRE(ensure_or.ok());
  REQUIRE(ensure_or->fast_path_hit);
  REQUIRE_FALSE(ensure_or->owner);
  REQUIRE_FALSE(ensure_or->waiter);
  REQUIRE(fast_path_count == 1);
  REQUIRE(owner_count == 0);
  REQUIRE(waiter_count == 0);

  CommunicatorTestPeer::stop_workers(server);
}

TEST_CASE("READ_PLAN_FAILED resolves pending request by request id", "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator client(cfg, std::move(pools), /*channel_expire_sec=*/0);

  auto prepared = std::make_shared<tensorcast::communicator::transport::PreparedReadPlan>();
  prepared->total_bytes = 64;
  prepared->local_nic = "mlx5_plan";
  prepared->rail_id = 2;
  auto read_request = std::make_shared<tensorcast::communicator::transport::ReadRequest>(
      "plan_display",
      "127.0.0.1",
      65012,
      prepared,
      /*request_id=*/92,
      /*rail_id=*/2);
  CommunicatorTestPeer::pending_requests(client).put(read_request->get_key(), read_request);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65012);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/2,
      /*max_window_segments=*/2);

  auto failed = EngineMessage::make_message<ProtoReadPlanFailed>(ENGINE_OP_READ_PLAN_FAILED);
  auto* payload = failed->get_payload<ProtoReadPlanFailed>();
  payload->request_id = 92;
  payload->reason = TENSORCAST_READ_FAILED_NO_TENSOR;

  auto status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, failed);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);
  REQUIRE(read_request->is_result_set());
  REQUIRE(absl::IsNotFound(read_request->status_.status));
  REQUIRE_FALSE(CommunicatorTestPeer::pending_requests(client).exist(read_request->get_key()));

  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(client);
}

TEST_CASE(
    "READ_PLAN_RESPONSE_EX uses prepared plan request key and queues plan ACKs",
    "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

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
    SUCCEED("Skipping READ_PLAN_RESPONSE_EX test: no RDMA net devices available");
    return;
  }

  auto& rdma_ctx = CommunicatorTestPeer::rdma_context(client);
  auto net_dev = rdma_ctx->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, "plan-local");
  REQUIRE(net_dev != nullptr);
  const std::string local_dev_name = net_dev->get_name();
  const std::string remote_dev_name = "peer.plan.nic";

  std::array<uint8_t, 64> local_buffer{};
  auto local_tensor = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "read_plan_region",
      reinterpret_cast<uint64_t>(local_buffer.data()),
      static_cast<uint64_t>(local_buffer.size()),
      COMMUNICATE_ENGINE_DEV_CPU,
      net_dev);
  local_tensor->register_mr(net_dev.get());
  local_tensor->set_read_ready();

  auto prepared = std::make_shared<tensorcast::communicator::transport::PreparedReadPlan>();
  prepared->logical_plan.local_regions.push_back(
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer.data()),
          .bytes = static_cast<uint64_t>(local_buffer.size()),
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      });
  prepared->logical_plan.source_slices.push_back(
      tensorcast::communicator::routing::SourceSlice{
          .authority_id = "authority",
          .route =
              tensorcast::communicator::routing::ReadRouteContext{
                  .local_endpoint_id = "local",
                  .remote_endpoint_id = "remote",
                  .protocol = tensorcast::communicator::routing::ConnectionProtocol::kRdma,
                  .rail_id = net_dev->get_rail_id(),
              },
          .tensor_key = "plan-source",
          .remote_offset = 0,
          .bytes = static_cast<uint64_t>(local_buffer.size()),
      });
  prepared->logical_plan.slices.push_back(
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 0,
          .local_region_offset = 0,
          .bytes = static_cast<uint64_t>(local_buffer.size()),
      });
  prepared->placements_by_source_slice = {
      {{.local_region_index = 0,
        .local_region_offset = 0,
        .source_slice_offset = 0,
        .bytes = static_cast<uint64_t>(local_buffer.size())}}};
  prepared->local_regions.push_back(
      tensorcast::communicator::transport::PreparedLocalRegion{
          .logical_region = prepared->logical_plan.local_regions.front(),
          .rail_id = net_dev->get_rail_id(),
          .nic_name = net_dev->get_name(),
          .tensor = local_tensor,
      });
  prepared->total_bytes = static_cast<uint64_t>(local_buffer.size());
  prepared->local_nic = net_dev->get_name();
  prepared->rail_id = net_dev->get_rail_id();

  auto read_request = std::make_shared<tensorcast::communicator::transport::ReadRequest>(
      "plan_display",
      "127.0.0.1",
      65013,
      prepared,
      /*request_id=*/93,
      net_dev->get_rail_id());
  CommunicatorTestPeer::pending_requests(client).put(read_request->get_key(), read_request);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65013);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/4);
  CommunicatorTestPeer::channels(client).put(read_request->get_dst_url(), channel);

  const uint32_t payload_size = sizeof(ProtoReadPlanResponseExHeader) + sizeof(ProtoReadPlanResponseExSeg);
  auto response = std::make_shared<EngineMessage>(ENGINE_OP_READ_PLAN_RESPONSE_EX, payload_size);
  auto* hdr = response->get_payload<ProtoReadPlanResponseExHeader>();
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->staged = 1;
  communicator::misc::STRNCPY(hdr->nic_name, remote_dev_name, kMaxDevName);
  hdr->request_id = 93;
  hdr->num_segments = 1;
  hdr->window_seq = 0;
  hdr->credit_granted = 1;
  hdr->more_segments = 0;
  hdr->zero_copy = 0;
  hdr->rail_id = net_dev->get_rail_id();
  auto* seg = reinterpret_cast<ProtoReadPlanResponseExSeg*>(
      reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanResponseExHeader));
  seg->source_slice_index = 0;
  seg->source_slice_offset = 0;
  seg->addr = 0xACDC;
  seg->bytes = static_cast<uint32_t>(local_buffer.size());
  seg->rkey = 0x1234;

  auto status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, response);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  auto endpoint = channel->get_rdma_endpoint(local_dev_name, remote_dev_name);
  REQUIRE(endpoint != nullptr);
  {
    absl::MutexLock lock(&endpoint->mu);
    REQUIRE(endpoint->state == Channel::HandshakeState::kConnectRequested);
    REQUIRE(endpoint->pending_reads.size() == 1);
    REQUIRE(endpoint->pending_reads.front().request == read_request);
  }
  REQUIRE(read_request->expected_completions_.load() == 1);
  {
    absl::MutexLock lock(&read_request->ack_mu_);
    REQUIRE(read_request->pending_ack_windows_.size() == 1);
    const auto& pending_window = read_request->pending_ack_windows_.front();
    REQUIRE(pending_window.window_seq == 0);
    REQUIRE(pending_window.num_segments == 1);
    REQUIRE(pending_window.ack_kind == tensorcast::communicator::transport::ReadRequest::AckKind::kSegmentCount);
  }

  CommunicatorTestPeer::pending_requests(client).erase_if_present(read_request->get_key());
  if (CommunicatorTestPeer::channels(client).exist(read_request->get_dst_url())) {
    CommunicatorTestPeer::channels(client).del(read_request->get_dst_url());
  }
  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(client);
}

TEST_CASE("READ_PLAN_RESPONSE_EX zero-copy windows skip staged plan ACK queueing", "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

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
    SUCCEED("Skipping zero-copy READ_PLAN_RESPONSE_EX test: no RDMA net devices available");
    return;
  }

  auto& rdma_ctx = CommunicatorTestPeer::rdma_context(client);
  auto net_dev = rdma_ctx->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, "plan-local-direct");
  REQUIRE(net_dev != nullptr);
  const std::string local_dev_name = net_dev->get_name();
  const std::string remote_dev_name = "peer.plan.direct.nic";

  std::array<uint8_t, 64> local_buffer{};
  auto local_tensor = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "read_plan_region_direct",
      reinterpret_cast<uint64_t>(local_buffer.data()),
      static_cast<uint64_t>(local_buffer.size()),
      COMMUNICATE_ENGINE_DEV_CPU,
      net_dev);
  local_tensor->register_mr(net_dev.get());
  local_tensor->set_read_ready();

  auto prepared = std::make_shared<tensorcast::communicator::transport::PreparedReadPlan>();
  prepared->logical_plan.local_regions.push_back(
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer.data()),
          .bytes = static_cast<uint64_t>(local_buffer.size()),
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      });
  prepared->logical_plan.source_slices.push_back(
      tensorcast::communicator::routing::SourceSlice{
          .authority_id = "authority",
          .route =
              tensorcast::communicator::routing::ReadRouteContext{
                  .local_endpoint_id = "local",
                  .remote_endpoint_id = "remote",
                  .protocol = tensorcast::communicator::routing::ConnectionProtocol::kRdma,
                  .rail_id = net_dev->get_rail_id(),
              },
          .tensor_key = "plan-source-direct",
          .remote_offset = 0,
          .bytes = static_cast<uint64_t>(local_buffer.size()),
      });
  prepared->logical_plan.slices.push_back(
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 0,
          .local_region_offset = 0,
          .bytes = static_cast<uint64_t>(local_buffer.size()),
      });
  prepared->placements_by_source_slice = {
      {{.local_region_index = 0,
        .local_region_offset = 0,
        .source_slice_offset = 0,
        .bytes = static_cast<uint64_t>(local_buffer.size())}}};
  prepared->local_regions.push_back(
      tensorcast::communicator::transport::PreparedLocalRegion{
          .logical_region = prepared->logical_plan.local_regions.front(),
          .rail_id = net_dev->get_rail_id(),
          .nic_name = net_dev->get_name(),
          .tensor = local_tensor,
      });
  prepared->total_bytes = static_cast<uint64_t>(local_buffer.size());
  prepared->local_nic = net_dev->get_name();
  prepared->rail_id = net_dev->get_rail_id();

  auto read_request = std::make_shared<tensorcast::communicator::transport::ReadRequest>(
      "plan_display_direct",
      "127.0.0.1",
      65015,
      prepared,
      /*request_id=*/95,
      net_dev->get_rail_id());
  CommunicatorTestPeer::pending_requests(client).put(read_request->get_key(), read_request);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65015);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/4);
  CommunicatorTestPeer::channels(client).put(read_request->get_dst_url(), channel);

  const uint32_t payload_size = sizeof(ProtoReadPlanResponseExHeader) + sizeof(ProtoReadPlanResponseExSeg);
  auto response = std::make_shared<EngineMessage>(ENGINE_OP_READ_PLAN_RESPONSE_EX, payload_size);
  auto* hdr = response->get_payload<ProtoReadPlanResponseExHeader>();
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->staged = 0;
  communicator::misc::STRNCPY(hdr->nic_name, remote_dev_name, kMaxDevName);
  hdr->request_id = 95;
  hdr->num_segments = 1;
  hdr->window_seq = 7;
  hdr->credit_granted = 1;
  hdr->more_segments = 0;
  hdr->zero_copy = 1;
  hdr->rail_id = net_dev->get_rail_id();
  auto* seg = reinterpret_cast<ProtoReadPlanResponseExSeg*>(
      reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanResponseExHeader));
  seg->source_slice_index = 0;
  seg->source_slice_offset = 0;
  seg->addr = 0xBEEF;
  seg->bytes = static_cast<uint32_t>(local_buffer.size());
  seg->rkey = 0x2345;

  auto status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, response);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  auto endpoint = channel->get_rdma_endpoint(local_dev_name, remote_dev_name);
  REQUIRE(endpoint != nullptr);
  {
    absl::MutexLock lock(&endpoint->mu);
    REQUIRE(endpoint->state == Channel::HandshakeState::kConnectRequested);
    REQUIRE(endpoint->pending_reads.size() == 1);
    REQUIRE(endpoint->pending_reads.front().request == read_request);
  }
  REQUIRE(read_request->expected_completions_.load() == 1);
  CHECK_FALSE(read_request->status_.rdma_staged_response);
  CHECK(read_request->status_.rdma_zero_copy_response);
  {
    absl::MutexLock lock(&read_request->ack_mu_);
    CHECK(read_request->pending_ack_windows_.empty());
    CHECK(read_request->segment_window_queue_.empty());
  }

  CommunicatorTestPeer::pending_requests(client).erase_if_present(read_request->get_key());
  if (CommunicatorTestPeer::channels(client).exist(read_request->get_dst_url())) {
    CommunicatorTestPeer::channels(client).del(read_request->get_dst_url());
  }
  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(client);
}

TEST_CASE(
    "Communicator stable local backing lazily registers slot-aligned chunks on RDMA rails",
    "[rdma][communicator][stable_backing]") {
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  cfg.mutable_rdma()->set_enable_stable_local_mr_reuse(true);
  cfg.mutable_rdma()->set_stable_local_mr_reuse_chunk_slots(2);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator client(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(client)) {
    CommunicatorTestPeer::stop_workers(client);
    SUCCEED("Skipping stable backing preregistration test: no RDMA net devices available");
    return;
  }

  auto net_dev = CommunicatorTestPeer::rdma_context(client)->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, "mr");
  REQUIRE(net_dev != nullptr);

  std::array<uint8_t, 128> local_buffer{};
  tensorcast::store::StableLocalBackingRef backing{
      .kind = tensorcast::store::StableLocalBackingKind::kHostSharedRegion,
      .backing_id = "region:test-stable-backing",
      .backing_base_addr = reinterpret_cast<uint64_t>(local_buffer.data()),
      .backing_bytes = static_cast<uint64_t>(local_buffer.size()),
      .slot_bytes = 32,
      .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
      .dev_id = 0,
  };

  REQUIRE(client.activate_stable_local_backing(backing, std::make_shared<int>(1)).ok());
  auto state = CommunicatorTestPeer::stable_local_backing_state(client, backing.backing_id);
  REQUIRE(state != nullptr);
  CHECK(CommunicatorTestPeer::stable_local_backing_active(client, backing.backing_id));
  CHECK(
      CommunicatorTestPeer::stable_local_backing_chunk_count(client, backing.backing_id, net_dev->get_rail_id()) == 0);

  tensorcast::communicator::routing::ReadPlan plan;
  plan.local_regions = {
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer.data()),
          .bytes = 32,
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
          .stable_backing = backing,
      },
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer.data()) + 32,
          .bytes = 32,
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
          .stable_backing = backing,
      },
  };
  plan.source_slices = {
      tensorcast::communicator::routing::SourceSlice{
          .authority_id = "authority-0",
          .route =
              tensorcast::communicator::routing::ReadRouteContext{
                  .local_endpoint_id = "cpu://local",
                  .remote_endpoint_id = "cpu://remote",
                  .protocol = tensorcast::communicator::routing::ConnectionProtocol::kRdma,
                  .rail_id = net_dev->get_rail_id(),
              },
          .tensor_key = "source-tensor",
          .remote_offset = 0,
          .bytes = 64,
      },
  };
  plan.slices = {
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 0,
          .source_slice_offset = 0,
          .local_region_offset = 0,
          .bytes = 32,
      },
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 1,
          .source_slice_offset = 32,
          .local_region_offset = 0,
          .bytes = 32,
      },
  };

  auto prepared_or = CommunicatorTestPeer::prepare_read_plan(client, plan, /*request_id=*/95, "source-tensor");
  REQUIRE(prepared_or.ok());
  auto prepared = *prepared_or;
  REQUIRE(prepared != nullptr);
  CHECK(prepared->local_registration_mode == "stable_backing_reuse");
  CHECK(prepared->stable_backing_id == backing.backing_id);
  CHECK(prepared->stable_backing_chunk_bytes == 64);
  CHECK(prepared->stable_backing_chunk_count == 1);
  CHECK(prepared->stable_backing_chunk_cache_hits == 1);
  CHECK(prepared->stable_backing_chunk_cache_misses == 1);
  CHECK(prepared->stable_backing_chunk_waits == 0);
  CHECK(prepared->stable_backing_chunk_lazy_registrations == 1);
  CHECK_FALSE(prepared->stable_backing_prewarm_requested);
  CHECK_FALSE(prepared->stable_backing_prewarm_complete);
  REQUIRE(prepared->local_regions.size() == 2);
  REQUIRE(prepared->local_regions[0].mr != nullptr);
  REQUIRE(prepared->local_regions[1].mr != nullptr);
  CHECK(prepared->local_regions[0].mr == prepared->local_regions[1].mr);
  CHECK(
      CommunicatorTestPeer::stable_local_backing_chunk_count(client, backing.backing_id, net_dev->get_rail_id()) == 1);

  REQUIRE(client.deactivate_stable_local_backing(backing.backing_id).ok());
  CHECK(CommunicatorTestPeer::stable_local_backing_state(client, backing.backing_id) == nullptr);
  CommunicatorTestPeer::stop_workers(client);
}

TEST_CASE(
    "Communicator stable local backing asynchronously prewarms slot-aligned chunks on all RDMA rails",
    "[rdma][communicator][stable_backing]") {
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  cfg.mutable_rdma()->set_enable_stable_local_mr_reuse(true);
  cfg.mutable_rdma()->set_stable_local_mr_reuse_chunk_slots(2);
  cfg.mutable_rdma()->set_stable_local_mr_reuse_prewarm_workers(1);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator client(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(client)) {
    CommunicatorTestPeer::stop_workers(client);
    SUCCEED("Skipping stable backing async prewarm test: no RDMA net devices available");
    return;
  }

  std::array<uint8_t, 128> local_buffer{};
  tensorcast::store::StableLocalBackingRef backing{
      .kind = tensorcast::store::StableLocalBackingKind::kHostSharedRegion,
      .backing_id = "region:test-stable-backing-eager",
      .backing_base_addr = reinterpret_cast<uint64_t>(local_buffer.data()),
      .backing_bytes = static_cast<uint64_t>(local_buffer.size()),
      .slot_bytes = 32,
      .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
      .dev_id = 0,
  };

  REQUIRE(client.activate_stable_local_backing(backing, std::make_shared<int>(1)).ok());
  auto state = CommunicatorTestPeer::stable_local_backing_state(client, backing.backing_id);
  REQUIRE(state != nullptr);
  CHECK(CommunicatorTestPeer::wait_for_stable_local_backing_prewarm(client, backing.backing_id, absl::Seconds(5)));
  CHECK(CommunicatorTestPeer::stable_local_backing_prewarm_complete(client, backing.backing_id));

  absl::flat_hash_set<int16_t> seen_rails;
  for (const auto& dev : CommunicatorTestPeer::rdma_context(client)->list_devs()) {
    if (dev == nullptr) {
      continue;
    }
    if (!seen_rails.insert(dev->get_rail_id()).second) {
      continue;
    }
    CHECK(CommunicatorTestPeer::stable_local_backing_chunk_count(client, backing.backing_id, dev->get_rail_id()) == 2);
  }

  auto net_dev = CommunicatorTestPeer::rdma_context(client)->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, "mr");
  REQUIRE(net_dev != nullptr);
  tensorcast::communicator::routing::ReadPlan plan;
  plan.local_regions = {
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer.data()),
          .bytes = 32,
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
          .stable_backing = backing,
      },
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer.data()) + 32,
          .bytes = 32,
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
          .stable_backing = backing,
      },
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer.data()) + 64,
          .bytes = 32,
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
          .stable_backing = backing,
      },
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer.data()) + 96,
          .bytes = 32,
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
          .stable_backing = backing,
      },
  };
  plan.source_slices = {
      tensorcast::communicator::routing::SourceSlice{
          .authority_id = "authority-0",
          .route =
              tensorcast::communicator::routing::ReadRouteContext{
                  .local_endpoint_id = "cpu://local",
                  .remote_endpoint_id = "cpu://remote",
                  .protocol = tensorcast::communicator::routing::ConnectionProtocol::kRdma,
                  .rail_id = net_dev->get_rail_id(),
              },
          .tensor_key = "source-tensor",
          .remote_offset = 0,
          .bytes = 128,
      },
  };
  plan.slices = {
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 0,
          .source_slice_offset = 0,
          .local_region_offset = 0,
          .bytes = 32,
      },
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 1,
          .source_slice_offset = 32,
          .local_region_offset = 0,
          .bytes = 32,
      },
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 2,
          .source_slice_offset = 64,
          .local_region_offset = 0,
          .bytes = 32,
      },
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 3,
          .source_slice_offset = 96,
          .local_region_offset = 0,
          .bytes = 32,
      },
  };

  auto prepared_or = CommunicatorTestPeer::prepare_read_plan(client, plan, /*request_id=*/96, "source-tensor");
  REQUIRE(prepared_or.ok());
  auto prepared = *prepared_or;
  REQUIRE(prepared != nullptr);
  CHECK(prepared->local_registration_mode == "stable_backing_reuse");
  CHECK(prepared->stable_backing_id == backing.backing_id);
  CHECK(prepared->stable_backing_chunk_bytes == 64);
  CHECK(prepared->stable_backing_chunk_count == 2);
  CHECK(prepared->stable_backing_chunk_cache_hits == 4);
  CHECK(prepared->stable_backing_chunk_cache_misses == 0);
  CHECK(prepared->stable_backing_prewarm_requested);
  CHECK(prepared->stable_backing_prewarm_complete);
  CHECK(prepared->stable_backing_chunk_waits == 0);
  CHECK(prepared->stable_backing_chunk_lazy_registrations == 0);

  REQUIRE(client.deactivate_stable_local_backing(backing.backing_id).ok());
  CHECK(CommunicatorTestPeer::stable_local_backing_state(client, backing.backing_id) == nullptr);
  CommunicatorTestPeer::stop_workers(client);
}

TEST_CASE("READ_PLAN_RESPONSE_EX uses prepared direct MRs without request tensors", "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

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
    SUCCEED("Skipping direct-MR READ_PLAN_RESPONSE_EX test: no RDMA net devices available");
    return;
  }

  auto net_dev = CommunicatorTestPeer::rdma_context(client)->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, "mr");
  REQUIRE(net_dev != nullptr);
  const std::string local_dev_name = net_dev->get_name();
  const std::string remote_dev_name = "peer.plan.direct.mr.nic";

  std::array<uint8_t, 64> local_buffer{};
  auto local_tensor = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "read_plan_region_direct_mr",
      reinterpret_cast<uint64_t>(local_buffer.data()),
      static_cast<uint64_t>(local_buffer.size()),
      COMMUNICATE_ENGINE_DEV_CPU,
      net_dev);
  local_tensor->register_mr(net_dev.get());
  local_tensor->set_read_ready();
  auto* mr = local_tensor->get_mr(net_dev);
  REQUIRE(mr != nullptr);

  auto prepared = std::make_shared<tensorcast::communicator::transport::PreparedReadPlan>();
  prepared->logical_plan.local_regions.push_back(
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer.data()),
          .bytes = static_cast<uint64_t>(local_buffer.size()),
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      });
  prepared->logical_plan.source_slices.push_back(
      tensorcast::communicator::routing::SourceSlice{
          .authority_id = "authority",
          .route =
              tensorcast::communicator::routing::ReadRouteContext{
                  .local_endpoint_id = "local",
                  .remote_endpoint_id = "remote",
                  .protocol = tensorcast::communicator::routing::ConnectionProtocol::kRdma,
                  .rail_id = net_dev->get_rail_id(),
              },
          .tensor_key = "plan-source-direct-mr",
          .remote_offset = 0,
          .bytes = static_cast<uint64_t>(local_buffer.size()),
      });
  prepared->logical_plan.slices.push_back(
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 0,
          .local_region_offset = 0,
          .bytes = static_cast<uint64_t>(local_buffer.size()),
      });
  prepared->placements_by_source_slice = {
      {{.local_region_index = 0,
        .local_region_offset = 0,
        .source_slice_offset = 0,
        .bytes = static_cast<uint64_t>(local_buffer.size())}}};
  prepared->local_regions.push_back(
      tensorcast::communicator::transport::PreparedLocalRegion{
          .logical_region = prepared->logical_plan.local_regions.front(),
          .rail_id = net_dev->get_rail_id(),
          .nic_name = net_dev->get_name(),
          .tensor = nullptr,
          .mr = mr,
          .keepalive = std::make_shared<int>(1),
      });
  prepared->total_bytes = static_cast<uint64_t>(local_buffer.size());
  prepared->local_nic = net_dev->get_name();
  prepared->rail_id = net_dev->get_rail_id();

  auto read_request = std::make_shared<tensorcast::communicator::transport::ReadRequest>(
      "plan_display_direct_mr",
      "127.0.0.1",
      65016,
      prepared,
      /*request_id=*/96,
      net_dev->get_rail_id());
  CommunicatorTestPeer::pending_requests(client).put(read_request->get_key(), read_request);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65016);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/4);
  CommunicatorTestPeer::channels(client).put(read_request->get_dst_url(), channel);

  const uint32_t payload_size = sizeof(ProtoReadPlanResponseExHeader) + sizeof(ProtoReadPlanResponseExSeg);
  auto response = std::make_shared<EngineMessage>(ENGINE_OP_READ_PLAN_RESPONSE_EX, payload_size);
  auto* hdr = response->get_payload<ProtoReadPlanResponseExHeader>();
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->staged = 1;
  communicator::misc::STRNCPY(hdr->nic_name, remote_dev_name, kMaxDevName);
  hdr->request_id = 96;
  hdr->num_segments = 1;
  hdr->window_seq = 9;
  hdr->credit_granted = 1;
  hdr->more_segments = 0;
  hdr->zero_copy = 0;
  hdr->rail_id = net_dev->get_rail_id();
  auto* seg = reinterpret_cast<ProtoReadPlanResponseExSeg*>(
      reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanResponseExHeader));
  seg->source_slice_index = 0;
  seg->source_slice_offset = 0;
  seg->addr = 0xCAFE0000;
  seg->bytes = static_cast<uint32_t>(local_buffer.size());
  seg->rkey = 0x3456;

  auto status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, response);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  auto endpoint = channel->get_rdma_endpoint(local_dev_name, remote_dev_name);
  REQUIRE(endpoint != nullptr);
  {
    absl::MutexLock lock(&endpoint->mu);
    REQUIRE(endpoint->pending_reads.size() == 1);
    REQUIRE(endpoint->pending_reads.front().segments.size() == 1);
    const auto& rdma_seg = endpoint->pending_reads.front().segments[0];
    REQUIRE(rdma_seg.local_addr == reinterpret_cast<uint64_t>(local_buffer.data()));
    REQUIRE(rdma_seg.local_lkey == mr->lkey);
    REQUIRE(rdma_seg.length == local_buffer.size());
  }

  CommunicatorTestPeer::pending_requests(client).erase_if_present(read_request->get_key());
  if (CommunicatorTestPeer::channels(client).exist(read_request->get_dst_url())) {
    CommunicatorTestPeer::channels(client).del(read_request->get_dst_url());
  }
  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(client);
}

TEST_CASE(
    "READ_PLAN_RESPONSE_EX preserves per-segment local lkey across prepared local regions",
    "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

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
    SUCCEED("Skipping multi-local-region READ_PLAN_RESPONSE_EX test: no RDMA net devices available");
    return;
  }

  auto& rdma_ctx = CommunicatorTestPeer::rdma_context(client);
  auto net_dev = rdma_ctx->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, "plan-local");
  REQUIRE(net_dev != nullptr);
  const std::string local_dev_name = net_dev->get_name();
  const std::string remote_dev_name = "peer.plan.multi.nic";

  std::array<uint8_t, 48> local_buffer0{};
  std::array<uint8_t, 64> local_buffer1{};
  auto local_tensor0 = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "read_plan_region_0",
      reinterpret_cast<uint64_t>(local_buffer0.data()),
      static_cast<uint64_t>(local_buffer0.size()),
      COMMUNICATE_ENGINE_DEV_CPU,
      net_dev);
  auto local_tensor1 = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "read_plan_region_1",
      reinterpret_cast<uint64_t>(local_buffer1.data()),
      static_cast<uint64_t>(local_buffer1.size()),
      COMMUNICATE_ENGINE_DEV_CPU,
      net_dev);
  local_tensor0->register_mr(net_dev.get());
  local_tensor1->register_mr(net_dev.get());
  local_tensor0->set_read_ready();
  local_tensor1->set_read_ready();

  auto* mr0 = local_tensor0->get_mr(net_dev);
  auto* mr1 = local_tensor1->get_mr(net_dev);
  REQUIRE(mr0 != nullptr);
  REQUIRE(mr1 != nullptr);

  constexpr uint64_t kSourceSlice0Bytes = 16;
  constexpr uint64_t kSourceSlice1Bytes = 28;
  constexpr uint64_t kRegion1PlacementOffset = 8;
  constexpr uint64_t kSeg1SourceOffset = 4;
  constexpr uint64_t kSeg1Bytes = 20;

  auto prepared = std::make_shared<tensorcast::communicator::transport::PreparedReadPlan>();
  prepared->logical_plan.local_regions = {
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer0.data()),
          .bytes = static_cast<uint64_t>(local_buffer0.size()),
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      },
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer1.data()),
          .bytes = static_cast<uint64_t>(local_buffer1.size()),
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      },
  };
  prepared->logical_plan.source_slices = {
      tensorcast::communicator::routing::SourceSlice{
          .authority_id = "authority",
          .route =
              tensorcast::communicator::routing::ReadRouteContext{
                  .local_endpoint_id = "local",
                  .remote_endpoint_id = "remote",
                  .protocol = tensorcast::communicator::routing::ConnectionProtocol::kRdma,
                  .rail_id = net_dev->get_rail_id(),
              },
          .tensor_key = "plan-source-0",
          .remote_offset = 0,
          .bytes = kSourceSlice0Bytes,
      },
      tensorcast::communicator::routing::SourceSlice{
          .authority_id = "authority",
          .route =
              tensorcast::communicator::routing::ReadRouteContext{
                  .local_endpoint_id = "local",
                  .remote_endpoint_id = "remote",
                  .protocol = tensorcast::communicator::routing::ConnectionProtocol::kRdma,
                  .rail_id = net_dev->get_rail_id(),
              },
          .tensor_key = "plan-source-1",
          .remote_offset = 128,
          .bytes = kSourceSlice1Bytes,
      },
  };
  prepared->logical_plan.slices = {
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 0,
          .local_region_offset = 0,
          .bytes = kSourceSlice0Bytes,
      },
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 1,
          .local_region_index = 1,
          .local_region_offset = kRegion1PlacementOffset,
          .bytes = kSourceSlice1Bytes,
      },
  };
  prepared->placements_by_source_slice = {
      {{
          .local_region_index = 0,
          .local_region_offset = 0,
          .source_slice_offset = 0,
          .bytes = kSourceSlice0Bytes,
      }},
      {{
          .local_region_index = 1,
          .local_region_offset = kRegion1PlacementOffset,
          .source_slice_offset = 0,
          .bytes = kSourceSlice1Bytes,
      }},
  };
  prepared->local_regions = {
      tensorcast::communicator::transport::PreparedLocalRegion{
          .logical_region = prepared->logical_plan.local_regions[0],
          .rail_id = net_dev->get_rail_id(),
          .nic_name = net_dev->get_name(),
          .tensor = local_tensor0,
      },
      tensorcast::communicator::transport::PreparedLocalRegion{
          .logical_region = prepared->logical_plan.local_regions[1],
          .rail_id = net_dev->get_rail_id(),
          .nic_name = net_dev->get_name(),
          .tensor = local_tensor1,
      },
  };
  prepared->total_bytes = kSourceSlice0Bytes + kSeg1Bytes;
  prepared->local_nic = net_dev->get_name();
  prepared->rail_id = net_dev->get_rail_id();

  auto read_request = std::make_shared<tensorcast::communicator::transport::ReadRequest>(
      "plan_display_multi_region",
      "127.0.0.1",
      65014,
      prepared,
      /*request_id=*/94,
      net_dev->get_rail_id());
  CommunicatorTestPeer::pending_requests(client).put(read_request->get_key(), read_request);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65014);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/4);
  CommunicatorTestPeer::channels(client).put(read_request->get_dst_url(), channel);

  const uint32_t payload_size = sizeof(ProtoReadPlanResponseExHeader) + 2 * sizeof(ProtoReadPlanResponseExSeg);
  auto response = std::make_shared<EngineMessage>(ENGINE_OP_READ_PLAN_RESPONSE_EX, payload_size);
  auto* hdr = response->get_payload<ProtoReadPlanResponseExHeader>();
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->staged = 1;
  communicator::misc::STRNCPY(hdr->nic_name, remote_dev_name, kMaxDevName);
  hdr->request_id = 94;
  hdr->num_segments = 2;
  hdr->window_seq = 3;
  hdr->credit_granted = 1;
  hdr->more_segments = 0;
  hdr->zero_copy = 0;
  hdr->rail_id = net_dev->get_rail_id();
  auto* segs = reinterpret_cast<ProtoReadPlanResponseExSeg*>(
      reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanResponseExHeader));
  segs[0].source_slice_index = 0;
  segs[0].source_slice_offset = 0;
  segs[0].addr = 0xACDC0000;
  segs[0].bytes = static_cast<uint32_t>(kSourceSlice0Bytes);
  segs[0].rkey = 0x1234;
  segs[1].source_slice_index = 1;
  segs[1].source_slice_offset = static_cast<uint32_t>(kSeg1SourceOffset);
  segs[1].addr = 0xBEEF0000;
  segs[1].bytes = static_cast<uint32_t>(kSeg1Bytes);
  segs[1].rkey = 0x5678;

  auto status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, response);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  auto endpoint = channel->get_rdma_endpoint(local_dev_name, remote_dev_name);
  REQUIRE(endpoint != nullptr);
  {
    absl::MutexLock lock(&endpoint->mu);
    REQUIRE(endpoint->state == Channel::HandshakeState::kConnectRequested);
    REQUIRE(endpoint->pending_reads.size() == 1);
    REQUIRE(endpoint->pending_reads.front().request == read_request);
    REQUIRE(endpoint->pending_reads.front().segments.size() == 2);

    const auto& rdma_seg0 = endpoint->pending_reads.front().segments[0];
    REQUIRE(rdma_seg0.local_addr == reinterpret_cast<uint64_t>(local_buffer0.data()));
    REQUIRE(rdma_seg0.local_lkey == mr0->lkey);
    REQUIRE(rdma_seg0.length == kSourceSlice0Bytes);
    REQUIRE(rdma_seg0.remote_addr == segs[0].addr);
    REQUIRE(rdma_seg0.rkey == segs[0].rkey);
    REQUIRE(rdma_seg0.window_seq == 3);
    REQUIRE(rdma_seg0.segment_idx == 0);

    const auto& rdma_seg1 = endpoint->pending_reads.front().segments[1];
    REQUIRE(
        rdma_seg1.local_addr ==
        reinterpret_cast<uint64_t>(local_buffer1.data()) + kRegion1PlacementOffset + kSeg1SourceOffset);
    REQUIRE(rdma_seg1.local_lkey == mr1->lkey);
    REQUIRE(rdma_seg1.length == kSeg1Bytes);
    REQUIRE(rdma_seg1.remote_addr == segs[1].addr);
    REQUIRE(rdma_seg1.rkey == segs[1].rkey);
    REQUIRE(rdma_seg1.window_seq == 3);
    REQUIRE(rdma_seg1.segment_idx == 1);
  }
  REQUIRE(read_request->expected_completions_.load() == 2);
  {
    absl::MutexLock lock(&read_request->ack_mu_);
    REQUIRE(read_request->pending_ack_windows_.size() == 1);
    const auto& pending_window = read_request->pending_ack_windows_.front();
    REQUIRE(pending_window.window_seq == 3);
    REQUIRE(pending_window.num_segments == 2);
    REQUIRE(pending_window.ack_kind == tensorcast::communicator::transport::ReadRequest::AckKind::kSegmentCount);
  }

  CommunicatorTestPeer::pending_requests(client).erase_if_present(read_request->get_key());
  if (CommunicatorTestPeer::channels(client).exist(read_request->get_dst_url())) {
    CommunicatorTestPeer::channels(client).del(read_request->get_dst_url());
  }
  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(client);
}

TEST_CASE(
    "READ_PLAN_RESPONSE_EX splits one source segment across multiple prepared placements",
    "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

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
    SUCCEED("Skipping multi-placement READ_PLAN_RESPONSE_EX test: no RDMA net devices available");
    return;
  }

  auto& rdma_ctx = CommunicatorTestPeer::rdma_context(client);
  auto net_dev = rdma_ctx->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, "plan-local");
  REQUIRE(net_dev != nullptr);
  const std::string local_dev_name = net_dev->get_name();
  const std::string remote_dev_name = "peer.plan.split.nic";

  std::array<uint8_t, 32> local_buffer0{};
  std::array<uint8_t, 64> local_buffer1{};
  auto local_tensor0 = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "read_plan_split_region_0",
      reinterpret_cast<uint64_t>(local_buffer0.data()),
      static_cast<uint64_t>(local_buffer0.size()),
      COMMUNICATE_ENGINE_DEV_CPU,
      net_dev);
  auto local_tensor1 = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "read_plan_split_region_1",
      reinterpret_cast<uint64_t>(local_buffer1.data()),
      static_cast<uint64_t>(local_buffer1.size()),
      COMMUNICATE_ENGINE_DEV_CPU,
      net_dev);
  local_tensor0->register_mr(net_dev.get());
  local_tensor1->register_mr(net_dev.get());
  local_tensor0->set_read_ready();
  local_tensor1->set_read_ready();

  auto* mr0 = local_tensor0->get_mr(net_dev);
  auto* mr1 = local_tensor1->get_mr(net_dev);
  REQUIRE(mr0 != nullptr);
  REQUIRE(mr1 != nullptr);

  constexpr uint64_t kSourceBytes = 48;
  constexpr uint64_t kFirstPlacementBytes = 16;
  constexpr uint64_t kFirstPlacementLocalOffset = 4;
  constexpr uint64_t kSecondPlacementSourceOffset = 16;
  constexpr uint64_t kSecondPlacementLocalOffset = 8;
  constexpr uint64_t kResponseSourceOffset = 8;
  constexpr uint64_t kResponseBytes = 32;

  auto prepared = std::make_shared<tensorcast::communicator::transport::PreparedReadPlan>();
  prepared->logical_plan.local_regions = {
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer0.data()),
          .bytes = static_cast<uint64_t>(local_buffer0.size()),
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      },
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer1.data()),
          .bytes = static_cast<uint64_t>(local_buffer1.size()),
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      },
  };
  prepared->logical_plan.source_slices = {
      tensorcast::communicator::routing::SourceSlice{
          .authority_id = "authority",
          .route =
              tensorcast::communicator::routing::ReadRouteContext{
                  .local_endpoint_id = "local",
                  .remote_endpoint_id = "remote",
                  .protocol = tensorcast::communicator::routing::ConnectionProtocol::kRdma,
                  .rail_id = net_dev->get_rail_id(),
              },
          .tensor_key = "plan-source-split",
          .remote_offset = 0,
          .bytes = kSourceBytes,
      },
  };
  prepared->logical_plan.slices = {
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 0,
          .source_slice_offset = 0,
          .local_region_offset = kFirstPlacementLocalOffset,
          .bytes = kFirstPlacementBytes,
      },
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 1,
          .source_slice_offset = kSecondPlacementSourceOffset,
          .local_region_offset = kSecondPlacementLocalOffset,
          .bytes = kSourceBytes - kSecondPlacementSourceOffset,
      },
  };
  prepared->placements_by_source_slice = {{
      {
          .local_region_index = 0,
          .local_region_offset = kFirstPlacementLocalOffset,
          .source_slice_offset = 0,
          .bytes = kFirstPlacementBytes,
      },
      {
          .local_region_index = 1,
          .local_region_offset = kSecondPlacementLocalOffset,
          .source_slice_offset = kSecondPlacementSourceOffset,
          .bytes = kSourceBytes - kSecondPlacementSourceOffset,
      },
  }};
  prepared->local_regions = {
      tensorcast::communicator::transport::PreparedLocalRegion{
          .logical_region = prepared->logical_plan.local_regions[0],
          .rail_id = net_dev->get_rail_id(),
          .nic_name = net_dev->get_name(),
          .tensor = local_tensor0,
      },
      tensorcast::communicator::transport::PreparedLocalRegion{
          .logical_region = prepared->logical_plan.local_regions[1],
          .rail_id = net_dev->get_rail_id(),
          .nic_name = net_dev->get_name(),
          .tensor = local_tensor1,
      },
  };
  prepared->total_bytes = kSourceBytes;
  prepared->local_nic = net_dev->get_name();
  prepared->rail_id = net_dev->get_rail_id();

  auto read_request = std::make_shared<tensorcast::communicator::transport::ReadRequest>(
      "plan_display_split",
      "127.0.0.1",
      65015,
      prepared,
      /*request_id=*/95,
      net_dev->get_rail_id());
  CommunicatorTestPeer::pending_requests(client).put(read_request->get_key(), read_request);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65015);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/4);
  CommunicatorTestPeer::channels(client).put(read_request->get_dst_url(), channel);

  const uint32_t payload_size = sizeof(ProtoReadPlanResponseExHeader) + sizeof(ProtoReadPlanResponseExSeg);
  auto response = std::make_shared<EngineMessage>(ENGINE_OP_READ_PLAN_RESPONSE_EX, payload_size);
  auto* hdr = response->get_payload<ProtoReadPlanResponseExHeader>();
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->staged = 1;
  communicator::misc::STRNCPY(hdr->nic_name, remote_dev_name, kMaxDevName);
  hdr->request_id = 95;
  hdr->num_segments = 1;
  hdr->window_seq = 4;
  hdr->credit_granted = 1;
  hdr->more_segments = 0;
  hdr->zero_copy = 0;
  hdr->rail_id = net_dev->get_rail_id();
  auto* seg = reinterpret_cast<ProtoReadPlanResponseExSeg*>(
      reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanResponseExHeader));
  seg->source_slice_index = 0;
  seg->source_slice_offset = kResponseSourceOffset;
  seg->addr = 0xCAFE0000;
  seg->bytes = static_cast<uint32_t>(kResponseBytes);
  seg->rkey = 0x2468;

  auto status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, response);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  auto endpoint = channel->get_rdma_endpoint(local_dev_name, remote_dev_name);
  REQUIRE(endpoint != nullptr);
  {
    absl::MutexLock lock(&endpoint->mu);
    REQUIRE(endpoint->state == Channel::HandshakeState::kConnectRequested);
    REQUIRE(endpoint->pending_reads.size() == 1);
    REQUIRE(endpoint->pending_reads.front().request == read_request);
    REQUIRE(endpoint->pending_reads.front().segments.size() == 2);

    const auto& rdma_seg0 = endpoint->pending_reads.front().segments[0];
    REQUIRE(
        rdma_seg0.local_addr ==
        reinterpret_cast<uint64_t>(local_buffer0.data()) + kFirstPlacementLocalOffset + kResponseSourceOffset);
    REQUIRE(rdma_seg0.local_lkey == mr0->lkey);
    REQUIRE(rdma_seg0.length == 8);
    REQUIRE(rdma_seg0.remote_addr == seg->addr);
    REQUIRE(rdma_seg0.rkey == seg->rkey);
    REQUIRE(rdma_seg0.window_seq == 4);
    REQUIRE(rdma_seg0.segment_idx == 0);

    const auto& rdma_seg1 = endpoint->pending_reads.front().segments[1];
    REQUIRE(rdma_seg1.local_addr == reinterpret_cast<uint64_t>(local_buffer1.data()) + kSecondPlacementLocalOffset);
    REQUIRE(rdma_seg1.local_lkey == mr1->lkey);
    REQUIRE(rdma_seg1.length == 24);
    REQUIRE(rdma_seg1.remote_addr == seg->addr + 8);
    REQUIRE(rdma_seg1.rkey == seg->rkey);
    REQUIRE(rdma_seg1.window_seq == 4);
    REQUIRE(rdma_seg1.segment_idx == 0);
  }
  REQUIRE(read_request->expected_completions_.load() == 2);
  {
    absl::MutexLock lock(&read_request->ack_mu_);
    REQUIRE(read_request->pending_ack_windows_.size() == 1);
    const auto& pending_window = read_request->pending_ack_windows_.front();
    REQUIRE(pending_window.window_seq == 4);
    REQUIRE(pending_window.num_segments == 1);
    REQUIRE(pending_window.remaining == 2);
    REQUIRE(pending_window.ack_kind == tensorcast::communicator::transport::ReadRequest::AckKind::kSegmentCount);
  }

  CommunicatorTestPeer::pending_requests(client).erase_if_present(read_request->get_key());
  if (CommunicatorTestPeer::channels(client).exist(read_request->get_dst_url())) {
    CommunicatorTestPeer::channels(client).del(read_request->get_dst_url());
  }
  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(client);
}

TEST_CASE("READ_PLAN_RESPONSE_EX posts vectored reads across multiple QPs", "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  cfg.mutable_rdma()->set_qp_count(2);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator client(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(client)) {
    CommunicatorTestPeer::stop_workers(client);
    SUCCEED("Skipping multi-QP READ_PLAN_RESPONSE_EX test: no RDMA net devices available");
    return;
  }

  auto& rdma_ctx = CommunicatorTestPeer::rdma_context(client);
  auto net_dev = rdma_ctx->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, "plan-local");
  REQUIRE(net_dev != nullptr);
  const std::string local_dev_name = net_dev->get_name();
  const std::string remote_dev_name = "peer.plan.mqp.nic";

  std::array<uint8_t, 96> local_buffer{};
  auto local_tensor = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "read_plan_multi_qp_region",
      reinterpret_cast<uint64_t>(local_buffer.data()),
      static_cast<uint64_t>(local_buffer.size()),
      COMMUNICATE_ENGINE_DEV_CPU,
      net_dev);
  local_tensor->register_mr(net_dev.get());
  local_tensor->set_read_ready();

  auto prepared = std::make_shared<tensorcast::communicator::transport::PreparedReadPlan>();
  prepared->logical_plan.local_regions = {
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer.data()),
          .bytes = static_cast<uint64_t>(local_buffer.size()),
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      },
  };
  prepared->logical_plan.source_slices = {
      tensorcast::communicator::routing::SourceSlice{
          .authority_id = "authority",
          .route =
              tensorcast::communicator::routing::ReadRouteContext{
                  .local_endpoint_id = "local",
                  .remote_endpoint_id = "remote",
                  .protocol = tensorcast::communicator::routing::ConnectionProtocol::kRdma,
                  .rail_id = net_dev->get_rail_id(),
              },
          .tensor_key = "plan-source-0",
          .remote_offset = 0,
          .bytes = 16,
      },
      tensorcast::communicator::routing::SourceSlice{
          .authority_id = "authority",
          .route =
              tensorcast::communicator::routing::ReadRouteContext{
                  .local_endpoint_id = "local",
                  .remote_endpoint_id = "remote",
                  .protocol = tensorcast::communicator::routing::ConnectionProtocol::kRdma,
                  .rail_id = net_dev->get_rail_id(),
              },
          .tensor_key = "plan-source-1",
          .remote_offset = 64,
          .bytes = 16,
      },
      tensorcast::communicator::routing::SourceSlice{
          .authority_id = "authority",
          .route =
              tensorcast::communicator::routing::ReadRouteContext{
                  .local_endpoint_id = "local",
                  .remote_endpoint_id = "remote",
                  .protocol = tensorcast::communicator::routing::ConnectionProtocol::kRdma,
                  .rail_id = net_dev->get_rail_id(),
              },
          .tensor_key = "plan-source-2",
          .remote_offset = 128,
          .bytes = 16,
      },
  };
  prepared->logical_plan.slices = {
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 0,
          .source_slice_offset = 0,
          .local_region_offset = 0,
          .bytes = 16,
      },
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 1,
          .local_region_index = 0,
          .source_slice_offset = 0,
          .local_region_offset = 24,
          .bytes = 16,
      },
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 2,
          .local_region_index = 0,
          .source_slice_offset = 0,
          .local_region_offset = 48,
          .bytes = 16,
      },
  };
  prepared->placements_by_source_slice = {
      {{
          .local_region_index = 0,
          .local_region_offset = 0,
          .source_slice_offset = 0,
          .bytes = 16,
      }},
      {{
          .local_region_index = 0,
          .local_region_offset = 24,
          .source_slice_offset = 0,
          .bytes = 16,
      }},
      {{
          .local_region_index = 0,
          .local_region_offset = 48,
          .source_slice_offset = 0,
          .bytes = 16,
      }},
  };
  prepared->local_regions = {
      tensorcast::communicator::transport::PreparedLocalRegion{
          .logical_region = prepared->logical_plan.local_regions.front(),
          .rail_id = net_dev->get_rail_id(),
          .nic_name = net_dev->get_name(),
          .tensor = local_tensor,
      },
  };
  prepared->total_bytes = 48;
  prepared->local_nic = net_dev->get_name();
  prepared->rail_id = net_dev->get_rail_id();

  auto read_request = std::make_shared<tensorcast::communicator::transport::ReadRequest>(
      "plan_display_multi_qp",
      "127.0.0.1",
      65016,
      prepared,
      /*request_id=*/96,
      net_dev->get_rail_id());
  CommunicatorTestPeer::pending_requests(client).put(read_request->get_key(), read_request);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65016);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/4);
  CommunicatorTestPeer::channels(client).put(read_request->get_dst_url(), channel);

  const uint32_t payload_size = sizeof(ProtoReadPlanResponseExHeader) + 3 * sizeof(ProtoReadPlanResponseExSeg);
  auto response = std::make_shared<EngineMessage>(ENGINE_OP_READ_PLAN_RESPONSE_EX, payload_size);
  auto* hdr = response->get_payload<ProtoReadPlanResponseExHeader>();
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->staged = 1;
  communicator::misc::STRNCPY(hdr->nic_name, remote_dev_name, kMaxDevName);
  hdr->request_id = 96;
  hdr->num_segments = 3;
  hdr->window_seq = 5;
  hdr->credit_granted = 1;
  hdr->more_segments = 0;
  hdr->zero_copy = 0;
  hdr->rail_id = net_dev->get_rail_id();
  auto* segs = reinterpret_cast<ProtoReadPlanResponseExSeg*>(
      reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanResponseExHeader));
  segs[0].source_slice_index = 0;
  segs[0].source_slice_offset = 0;
  segs[0].addr = 0xAAA000;
  segs[0].bytes = 16;
  segs[0].rkey = 0x1010;
  segs[1].source_slice_index = 1;
  segs[1].source_slice_offset = 0;
  segs[1].addr = 0xBBB000;
  segs[1].bytes = 16;
  segs[1].rkey = 0x2020;
  segs[2].source_slice_index = 2;
  segs[2].source_slice_offset = 0;
  segs[2].addr = 0xCCC000;
  segs[2].bytes = 16;
  segs[2].rkey = 0x3030;

  auto status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, response);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  auto rdma_transport = channel->get_rdma(local_dev_name, remote_dev_name);
  REQUIRE(rdma_transport != nullptr);
  ScopedPostSendHook post_send_hook([](struct ibv_qp*, struct ibv_send_wr*, struct ibv_send_wr** bad_wr) {
    if (bad_wr != nullptr) {
      *bad_wr = nullptr;
    }
    return 0;
  });

  auto connect_rsp = EngineMessage::make_message<ProtoRdmaConnectResponse>(ENGINE_OP_RDMA_CONNECT_RESPONSE);
  auto* connect_payload = connect_rsp->get_payload<ProtoRdmaConnectResponse>();
  communicator::misc::STRNCPY(connect_payload->src_dev_name, local_dev_name, kMaxDevName);
  communicator::misc::STRNCPY(connect_payload->dst_dev_name, remote_dev_name, kMaxDevName);
  REQUIRE(rdma_transport->get_local_info(&connect_payload->qp_info) == tensorcast::communicator::misc::SUCCESS);

  status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, connect_rsp);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  REQUIRE(tensorcast::communicator::transport::RdmaTransportTestPeer::qp_count(*rdma_transport) == 2);
  REQUIRE(tensorcast::communicator::transport::RdmaTransportTestPeer::inflight_queue_size(*rdma_transport, 0) == 2);
  REQUIRE(tensorcast::communicator::transport::RdmaTransportTestPeer::inflight_queue_size(*rdma_transport, 1) == 1);
  REQUIRE(read_request->expected_completions_.load() == 3);
  REQUIRE_FALSE(read_request->is_result_set());
  {
    absl::MutexLock lock(&read_request->ack_mu_);
    REQUIRE(read_request->pending_ack_windows_.size() == 1);
    REQUIRE(read_request->pending_ack_windows_.front().remaining == 3);
  }

  CommunicatorTestPeer::pending_requests(client).erase_if_present(read_request->get_key());
  if (CommunicatorTestPeer::channels(client).exist(read_request->get_dst_url())) {
    CommunicatorTestPeer::channels(client).del(read_request->get_dst_url());
  }
  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(client);
}

TEST_CASE(
    "READ_PLAN_RESPONSE_EX partial post failure keeps only posted WRs inflight and fails request",
    "[rdma][communicator][read_plan]") {
  using tensorcast::communicator::base::CHANNEL_RDMA;
  using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;

  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true);
  cfg.mutable_rdma()->set_qp_count(1);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/true);
  Communicator client(cfg, std::move(pools), /*channel_expire_sec=*/0);
  if (!CommunicatorTestPeer::has_rdma_device(client)) {
    CommunicatorTestPeer::stop_workers(client);
    SUCCEED("Skipping partial-post READ_PLAN_RESPONSE_EX test: no RDMA net devices available");
    return;
  }

  auto& rdma_ctx = CommunicatorTestPeer::rdma_context(client);
  auto net_dev = rdma_ctx->get_best_dev(COMMUNICATE_ENGINE_DEV_CPU, -1, -1, "plan-local");
  REQUIRE(net_dev != nullptr);
  const std::string local_dev_name = net_dev->get_name();
  const std::string remote_dev_name = "peer.plan.partial.nic";

  std::array<uint8_t, 96> local_buffer{};
  auto local_tensor = std::make_shared<tensorcast::communicator::transport::PartitionTensor>(
      "read_plan_partial_region",
      reinterpret_cast<uint64_t>(local_buffer.data()),
      static_cast<uint64_t>(local_buffer.size()),
      COMMUNICATE_ENGINE_DEV_CPU,
      net_dev);
  local_tensor->register_mr(net_dev.get());
  local_tensor->set_read_ready();

  auto prepared = std::make_shared<tensorcast::communicator::transport::PreparedReadPlan>();
  prepared->logical_plan.local_regions = {
      tensorcast::communicator::routing::LocalRegion{
          .addr = reinterpret_cast<uint64_t>(local_buffer.data()),
          .bytes = static_cast<uint64_t>(local_buffer.size()),
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      },
  };
  prepared->logical_plan.source_slices = {
      tensorcast::communicator::routing::SourceSlice{
          .authority_id = "authority",
          .route =
              tensorcast::communicator::routing::ReadRouteContext{
                  .local_endpoint_id = "local",
                  .remote_endpoint_id = "remote",
                  .protocol = tensorcast::communicator::routing::ConnectionProtocol::kRdma,
                  .rail_id = net_dev->get_rail_id(),
              },
          .tensor_key = "plan-source-0",
          .remote_offset = 0,
          .bytes = 16,
      },
      tensorcast::communicator::routing::SourceSlice{
          .authority_id = "authority",
          .route =
              tensorcast::communicator::routing::ReadRouteContext{
                  .local_endpoint_id = "local",
                  .remote_endpoint_id = "remote",
                  .protocol = tensorcast::communicator::routing::ConnectionProtocol::kRdma,
                  .rail_id = net_dev->get_rail_id(),
              },
          .tensor_key = "plan-source-1",
          .remote_offset = 64,
          .bytes = 16,
      },
  };
  prepared->logical_plan.slices = {
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 0,
          .source_slice_offset = 0,
          .local_region_offset = 0,
          .bytes = 16,
      },
      tensorcast::communicator::routing::ReadPlanSlice{
          .source_slice_index = 1,
          .local_region_index = 0,
          .source_slice_offset = 0,
          .local_region_offset = 32,
          .bytes = 16,
      },
  };
  prepared->placements_by_source_slice = {
      {{
          .local_region_index = 0,
          .local_region_offset = 0,
          .source_slice_offset = 0,
          .bytes = 16,
      }},
      {{
          .local_region_index = 0,
          .local_region_offset = 32,
          .source_slice_offset = 0,
          .bytes = 16,
      }},
  };
  prepared->local_regions = {
      tensorcast::communicator::transport::PreparedLocalRegion{
          .logical_region = prepared->logical_plan.local_regions.front(),
          .rail_id = net_dev->get_rail_id(),
          .nic_name = net_dev->get_name(),
          .tensor = local_tensor,
      },
  };
  prepared->total_bytes = 32;
  prepared->local_nic = net_dev->get_name();
  prepared->rail_id = net_dev->get_rail_id();

  auto read_request = std::make_shared<tensorcast::communicator::transport::ReadRequest>(
      "plan_display_partial_post",
      "127.0.0.1",
      65017,
      prepared,
      /*request_id=*/97,
      net_dev->get_rail_id());
  CommunicatorTestPeer::pending_requests(client).put(read_request->get_key(), read_request);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65017);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/4);
  CommunicatorTestPeer::channels(client).put(read_request->get_dst_url(), channel);

  const uint32_t payload_size = sizeof(ProtoReadPlanResponseExHeader) + 2 * sizeof(ProtoReadPlanResponseExSeg);
  auto response = std::make_shared<EngineMessage>(ENGINE_OP_READ_PLAN_RESPONSE_EX, payload_size);
  auto* hdr = response->get_payload<ProtoReadPlanResponseExHeader>();
  hdr->transport_type = ENGINE_TRANSPORT_RDMA;
  hdr->staged = 1;
  communicator::misc::STRNCPY(hdr->nic_name, remote_dev_name, kMaxDevName);
  hdr->request_id = 97;
  hdr->num_segments = 2;
  hdr->window_seq = 6;
  hdr->credit_granted = 1;
  hdr->more_segments = 0;
  hdr->zero_copy = 0;
  hdr->rail_id = net_dev->get_rail_id();
  auto* segs = reinterpret_cast<ProtoReadPlanResponseExSeg*>(
      reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadPlanResponseExHeader));
  segs[0].source_slice_index = 0;
  segs[0].source_slice_offset = 0;
  segs[0].addr = 0xDDD000;
  segs[0].bytes = 16;
  segs[0].rkey = 0x4040;
  segs[1].source_slice_index = 1;
  segs[1].source_slice_offset = 0;
  segs[1].addr = 0xEEE000;
  segs[1].bytes = 16;
  segs[1].rkey = 0x5050;

  auto status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, response);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  auto rdma_transport = channel->get_rdma(local_dev_name, remote_dev_name);
  REQUIRE(rdma_transport != nullptr);
  ScopedPostSendHook post_send_hook([](struct ibv_qp*, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr) {
    if (wr == nullptr || wr->next == nullptr) {
      errno = EINVAL;
      if (bad_wr != nullptr) {
        *bad_wr = wr;
      }
      return -1;
    }
    errno = EIO;
    if (bad_wr != nullptr) {
      *bad_wr = wr->next;
    }
    return -1;
  });

  auto connect_rsp = EngineMessage::make_message<ProtoRdmaConnectResponse>(ENGINE_OP_RDMA_CONNECT_RESPONSE);
  auto* connect_payload = connect_rsp->get_payload<ProtoRdmaConnectResponse>();
  communicator::misc::STRNCPY(connect_payload->src_dev_name, local_dev_name, kMaxDevName);
  communicator::misc::STRNCPY(connect_payload->dst_dev_name, remote_dev_name, kMaxDevName);
  REQUIRE(rdma_transport->get_local_info(&connect_payload->qp_info) == tensorcast::communicator::misc::SUCCESS);

  status = CommunicatorTestPeer::on_receive_response(client, channel, control_transport, connect_rsp);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);

  REQUIRE(tensorcast::communicator::transport::RdmaTransportTestPeer::qp_count(*rdma_transport) == 1);
  REQUIRE(tensorcast::communicator::transport::RdmaTransportTestPeer::inflight_queue_size(*rdma_transport, 0) == 1);
  REQUIRE(read_request->is_result_set());
  REQUIRE_FALSE(read_request->status_.status.ok());
  REQUIRE(
      std::string(read_request->status_.status.message()).find("rdma post_send (multi) failed") != std::string::npos);
  REQUIRE_FALSE(CommunicatorTestPeer::pending_requests(client).exist(read_request->get_key()));
  {
    absl::MutexLock lock(&read_request->ack_mu_);
    REQUIRE(read_request->pending_ack_windows_.size() == 1);
    REQUIRE(read_request->pending_ack_windows_.front().remaining == 2);
  }

  if (CommunicatorTestPeer::channels(client).exist(read_request->get_dst_url())) {
    CommunicatorTestPeer::channels(client).del(read_request->get_dst_url());
  }
  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(client);
}

TEST_CASE(
    "RDMA_READ_PLAN_DONE_EX ignores zero-copy windows without staged leases",
    "[rdma][communicator][read_plan][ack]") {
  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/false);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/false);
  Communicator server(cfg, std::move(pools), /*channel_expire_sec=*/0);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65100);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      tensorcast::communicator::base::CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/4);
  auto flow_state = channel->flow_state();
  REQUIRE(flow_state != nullptr);

  AckLeaseFactory lease_factory;
  constexpr uint32_t kRequestId = 300;
  const std::string request_key = tensorcast::communicator::transport::get_read_plan_request_key(kRequestId);
  auto credit_or = flow_state->ledger.acquire(1);
  REQUIRE(credit_or.ok());
  credit_or->mark_consumed();
  flow_state->registry.put(
      StageLeaseKey{
          .request_key = request_key,
          .window_seq = 41,
          .segment_idx = 0,
      },
      lease_factory.make(
          flow_state->ledger,
          reinterpret_cast<void*>(0x500),
          /*bytes=*/8,
          request_key,
          /*window_seq=*/41,
          /*segment_idx=*/0,
          /*offset=*/0));
  CHECK(flow_state->registry.size() == 1);
  CHECK(flow_state->ledger.outstanding_credit() == 1);

  auto ack = EngineMessage::make_message<ProtoRdmaReadPlanDoneExHeader>(ENGINE_OP_RDMA_READ_PLAN_DONE_EX);
  auto* hdr = ack->get_payload<ProtoRdmaReadPlanDoneExHeader>();
  hdr->request_id = kRequestId;
  hdr->window_seq = 40;
  hdr->num_segments = 1;
  hdr->final_window = 0;

  auto status = CommunicatorTestPeer::on_receive_request(server, channel, control_transport, ack);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);
  CHECK(flow_state->registry.size() == 1);
  CHECK(flow_state->ledger.outstanding_credit() == 1);
  CHECK(lease_factory.credit_released == 0);
  CHECK(lease_factory.stager->released_ptrs_.empty());

  auto still_inflight =
      flow_state->registry.take(StageLeaseKey{.request_key = request_key, .window_seq = 41, .segment_idx = 0});
  REQUIRE(still_inflight.ok());
  CHECK(still_inflight->metadata().window_seq == 41);
  still_inflight->release();
  CHECK(flow_state->ledger.outstanding_credit() == 0);
  CHECK(flow_state->registry.size() == 0);

  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(server);
}

TEST_CASE(
    "RDMA_READ_PLAN_DONE_EX restores staged credit one window at a time",
    "[rdma][communicator][read_plan][ack]") {
  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/false);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/false);
  Communicator server(cfg, std::move(pools), /*channel_expire_sec=*/0);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65101);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      tensorcast::communicator::base::CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/4);
  auto flow_state = channel->flow_state();
  REQUIRE(flow_state != nullptr);

  AckLeaseFactory lease_factory;
  constexpr uint32_t kRequestId = 301;
  const std::string request_key = tensorcast::communicator::transport::get_read_plan_request_key(kRequestId);
  auto put_lease = [&](uint32_t window_seq, uint32_t segment_idx, uint64_t offset, void* ptr) {
    auto credit_or = flow_state->ledger.acquire(1);
    REQUIRE(credit_or.ok());
    credit_or->mark_consumed();
    flow_state->registry.put(
        StageLeaseKey{
            .request_key = request_key,
            .window_seq = window_seq,
            .segment_idx = segment_idx,
        },
        lease_factory.make(flow_state->ledger, ptr, /*bytes=*/8, request_key, window_seq, segment_idx, offset));
  };

  put_lease(/*window_seq=*/11, /*segment_idx=*/0, /*offset=*/0, reinterpret_cast<void*>(0x100));
  put_lease(/*window_seq=*/11, /*segment_idx=*/1, /*offset=*/8, reinterpret_cast<void*>(0x108));
  put_lease(/*window_seq=*/12, /*segment_idx=*/0, /*offset=*/0, reinterpret_cast<void*>(0x200));
  CHECK(flow_state->registry.size() == 3);
  CHECK(flow_state->ledger.outstanding_credit() == 3);

  auto ack = EngineMessage::make_message<ProtoRdmaReadPlanDoneExHeader>(ENGINE_OP_RDMA_READ_PLAN_DONE_EX);
  auto* hdr = ack->get_payload<ProtoRdmaReadPlanDoneExHeader>();
  hdr->request_id = kRequestId;
  hdr->window_seq = 11;
  hdr->num_segments = 2;
  hdr->final_window = 0;

  auto status = CommunicatorTestPeer::on_receive_request(server, channel, control_transport, ack);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);
  CHECK(flow_state->registry.size() == 1);
  CHECK(flow_state->ledger.outstanding_credit() == 1);
  CHECK(lease_factory.credit_released == 2);
  REQUIRE(lease_factory.stager->released_ptrs_.size() == 2);
  CHECK(lease_factory.stager->released_ptrs_[0] == reinterpret_cast<void*>(0x100));
  CHECK(lease_factory.stager->released_ptrs_[1] == reinterpret_cast<void*>(0x108));
  CHECK_FALSE(
      flow_state->registry.take(StageLeaseKey{.request_key = request_key, .window_seq = 11, .segment_idx = 0}).ok());
  CHECK_FALSE(
      flow_state->registry.take(StageLeaseKey{.request_key = request_key, .window_seq = 11, .segment_idx = 1}).ok());

  auto remaining =
      flow_state->registry.take(StageLeaseKey{.request_key = request_key, .window_seq = 12, .segment_idx = 0});
  REQUIRE(remaining.ok());
  CHECK(remaining->metadata().window_seq == 12);
  CHECK(remaining->metadata().segment_idx == 0);
  CHECK(remaining->metadata().offset == 0);
  remaining->release();
  CHECK(flow_state->ledger.outstanding_credit() == 0);
  CHECK(flow_state->registry.size() == 0);

  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(server);
}

TEST_CASE(
    "RDMA_READ_PLAN_DONE_EX keys staged release by window_seq and segment_idx",
    "[rdma][communicator][read_plan][ack]") {
  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/false);
  auto pools = tensorcast::testing::make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(16ULL << 20),
      /*cpu_slice_bytes=*/(4ULL << 20),
      /*enable_rdma=*/false);
  Communicator server(cfg, std::move(pools), /*channel_expire_sec=*/0);

  int sv[2];
  REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  auto control_ctx = std::make_shared<tensorcast::communicator::transport::TcpContext>();
  struct sockaddr_in remote_addr{};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(65102);
  remote_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  auto control_transport =
      std::make_shared<tensorcast::communicator::transport::TcpTransport>(control_ctx.get(), sv[0], remote_addr);
  auto channel = std::make_shared<Channel>(
      control_transport,
      tensorcast::communicator::base::CHANNEL_RDMA,
      /*buffers_per_flow=*/4,
      /*max_window_segments=*/4);
  auto flow_state = channel->flow_state();
  REQUIRE(flow_state != nullptr);

  AckLeaseFactory lease_factory;
  constexpr uint32_t kRequestId = 302;
  const std::string request_key = tensorcast::communicator::transport::get_read_plan_request_key(kRequestId);
  auto put_lease = [&](uint32_t window_seq, void* ptr) {
    auto credit_or = flow_state->ledger.acquire(1);
    REQUIRE(credit_or.ok());
    credit_or->mark_consumed();
    flow_state->registry.put(
        StageLeaseKey{
            .request_key = request_key,
            .window_seq = window_seq,
            .segment_idx = 0,
        },
        lease_factory.make(
            flow_state->ledger, ptr, /*bytes=*/8, request_key, window_seq, /*segment_idx=*/0, /*offset=*/0));
  };

  // The semantic source-slice metadata is intentionally identical across both
  // windows; only request-local window identity should control release.
  put_lease(/*window_seq=*/20, reinterpret_cast<void*>(0x300));
  put_lease(/*window_seq=*/21, reinterpret_cast<void*>(0x400));
  CHECK(flow_state->registry.size() == 2);
  CHECK(flow_state->ledger.outstanding_credit() == 2);

  auto ack = EngineMessage::make_message<ProtoRdmaReadPlanDoneExHeader>(ENGINE_OP_RDMA_READ_PLAN_DONE_EX);
  auto* hdr = ack->get_payload<ProtoRdmaReadPlanDoneExHeader>();
  hdr->request_id = kRequestId;
  hdr->window_seq = 21;
  hdr->num_segments = 1;
  hdr->final_window = 0;

  auto status = CommunicatorTestPeer::on_receive_request(server, channel, control_transport, ack);
  REQUIRE(status == tensorcast::communicator::misc::SUCCESS);
  CHECK(flow_state->registry.size() == 1);
  CHECK(flow_state->ledger.outstanding_credit() == 1);
  CHECK(lease_factory.credit_released == 1);
  REQUIRE(lease_factory.stager->released_ptrs_.size() == 1);
  CHECK(lease_factory.stager->released_ptrs_.front() == reinterpret_cast<void*>(0x400));
  CHECK_FALSE(
      flow_state->registry.take(StageLeaseKey{.request_key = request_key, .window_seq = 21, .segment_idx = 0}).ok());

  auto still_inflight =
      flow_state->registry.take(StageLeaseKey{.request_key = request_key, .window_seq = 20, .segment_idx = 0});
  REQUIRE(still_inflight.ok());
  CHECK(still_inflight->metadata().offset == 0);
  CHECK(still_inflight->metadata().window_seq == 20);
  still_inflight->release();
  CHECK(flow_state->ledger.outstanding_credit() == 0);
  CHECK(flow_state->registry.size() == 0);

  ::close(sv[1]);
  CommunicatorTestPeer::stop_workers(server);
}

} // namespace tensorcast::unittests
