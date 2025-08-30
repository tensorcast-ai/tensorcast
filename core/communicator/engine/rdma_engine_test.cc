
// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include "core/communicator/engine/engine.h"

namespace tensorcast::unittests {

#define BUF_SIZE 65536
#define CPU_KEY "RDMA_TENSOR_KEY_CPU"
#define GPU_KEY "RDMA_TENSOR_KEY_GPU"

struct RdmaTestFixture {
  communicator::CommunicateEngine* server_;
  absl::Status server_init_status_;
  communicator::CommunicateEngine* client_;
  absl::Status client_init_status_;
  uint32_t server_buf_[BUF_SIZE];
  uint32_t client_buf_[BUF_SIZE];

  RdmaTestFixture() {
    communicator::CommunicatorConfig srv_cfg; srv_cfg.enable_rdma = true;
    server_ = new communicator::CommunicateEngine(srv_cfg, 30);
    server_init_status_ = server_->init("127.0.0.1", 60000, 8);
    communicator::CommunicatorConfig cli_cfg; cli_cfg.enable_rdma = true;
    client_ = new communicator::CommunicateEngine(cli_cfg, 30);
    client_init_status_ = client_->init("127.0.0.1", 60001, 8);

    for (uint32_t i = 0; i < BUF_SIZE; i++) {
      server_buf_[i] = i;
      client_buf_[i] = 0;
    }
  }

  ~RdmaTestFixture() {
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

    // register GPU tensor
    auto status = fixture.server_->register_tensor(
        GPU_KEY,
        reinterpret_cast<uint64_t>(fixture.server_buf_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::COMMUNICATE_ENGINE_DEV_GPU,
        0);
    REQUIRE(status.ok());

    // register CPU tensor
    status = fixture.server_->register_tensor(
        CPU_KEY,
        reinterpret_cast<uint64_t>(fixture.server_buf_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::COMMUNICATE_ENGINE_DEV_CPU,
        -1);
    REQUIRE(status.ok());
  }

  SECTION("Register and unregister tensors") {
    REQUIRE(fixture.server_init_status_.ok());
    REQUIRE(fixture.client_init_status_.ok());

    // register/unregister CPU tensor
    auto status = fixture.server_->register_tensor(
        CPU_KEY,
        reinterpret_cast<uint64_t>(fixture.server_buf_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::COMMUNICATE_ENGINE_DEV_CPU,
        -1);
    REQUIRE(status.ok());

    status = fixture.server_->unregister_tensor(CPU_KEY);
    REQUIRE(status.ok());

    // register/unregister GPU tensor
    status = fixture.server_->register_tensor(
        GPU_KEY,
        reinterpret_cast<uint64_t>(fixture.server_buf_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::COMMUNICATE_ENGINE_DEV_GPU,
        0);
    REQUIRE(status.ok());

    status = fixture.server_->unregister_tensor(GPU_KEY);
    REQUIRE(status.ok());
  }

  SECTION("Unregister non-existent tensors fails") {
    REQUIRE(fixture.server_init_status_.ok());
    REQUIRE(fixture.client_init_status_.ok());

    auto status = fixture.server_->unregister_tensor(CPU_KEY);
    REQUIRE_FALSE(status.ok());

    status = fixture.server_->unregister_tensor(GPU_KEY);
    REQUIRE_FALSE(status.ok());
  }

  SECTION("Read tensors") {
    REQUIRE(fixture.server_init_status_.ok());
    REQUIRE(fixture.client_init_status_.ok());

    auto status = fixture.server_->register_tensor(
        GPU_KEY,
        reinterpret_cast<uint64_t>(fixture.server_buf_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::COMMUNICATE_ENGINE_DEV_GPU,
        0);
    REQUIRE(status.ok());

    status = fixture.server_->register_tensor(
        CPU_KEY,
        reinterpret_cast<uint64_t>(fixture.server_buf_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::COMMUNICATE_ENGINE_DEV_CPU,
        -1);
    REQUIRE(status.ok());

    // Test reading with various offsets
    for (uint32_t offset = 4096; offset < BUF_SIZE - 1; offset += 4096) {
      auto future_result = fixture.client_->read_tensor(
          GPU_KEY,
          reinterpret_cast<uint64_t>(fixture.client_buf_),
          (BUF_SIZE - offset) * sizeof(uint32_t),
          communicator::COMMUNICATE_ENGINE_DEV_GPU,
          0,
          "127.0.0.1",
          60000,
          offset * sizeof(uint32_t));

      auto result = future_result.get();
      REQUIRE(result.status.ok());

      bool verify_ok = true;
      for (uint32_t i = 0; i < BUF_SIZE - offset; i++) {
        if (fixture.client_buf_[i] != fixture.server_buf_[i + offset]) {
          verify_ok = false;
        }
      }
      REQUIRE(verify_ok);
    }

    // Test reading entire buffer
    auto future_result = fixture.client_->read_tensor(
        GPU_KEY,
        reinterpret_cast<uint64_t>(fixture.client_buf_),
        sizeof(uint32_t) * BUF_SIZE,
        communicator::COMMUNICATE_ENGINE_DEV_GPU,
        0,
        "127.0.0.1",
        60000);

    auto result = future_result.get();
    REQUIRE(result.status.ok());

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

} // namespace tensorcast::unittests
