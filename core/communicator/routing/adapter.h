// Copyright (c) 2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ROUTING_ADAPTER_H_
#define CORE_COMMUNICATOR_ROUTING_ADAPTER_H_

#include <memory>

#include "absl/status/status.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/routing/types.h"
#include "core/communicator/transport/request.h"

namespace tensorcast::communicator::routing {

class ConnectionAdapter {
 public:
  virtual ~ConnectionAdapter() = default;

  virtual ConnectionProtocol protocol() const = 0;
  virtual bool is_available() const = 0;

  virtual transport::future_read_result_t read_tensor(
      const ReadRequest& request,
      const EndpointBinding& local,
      const EndpointBinding& remote) = 0;

  virtual absl::Status close(const EndpointBinding& remote) = 0;
};

class EngineAdapter final : public ConnectionAdapter {
 public:
  explicit EngineAdapter(std::shared_ptr<engine::Communicator> engine);

  ConnectionProtocol protocol() const override {
    return ConnectionProtocol::kAuto;
  }

  bool is_available() const override {
    return engine_ != nullptr;
  }

  transport::future_read_result_t read_tensor(
      const ReadRequest& request,
      const EndpointBinding& local,
      const EndpointBinding& remote) override;

  absl::Status close(const EndpointBinding& remote) override;

  const std::shared_ptr<engine::Communicator>& engine() const {
    return engine_;
  }

 private:
  std::shared_ptr<engine::Communicator> engine_;
};

class NvlinkAdapter final : public ConnectionAdapter {
 public:
  explicit NvlinkAdapter(std::shared_ptr<engine::Communicator> engine = nullptr);

  ConnectionProtocol protocol() const override {
    return ConnectionProtocol::kNvlink;
  }

  bool is_available() const override {
    return engine_ != nullptr;
  }

  transport::future_read_result_t read_tensor(
      const ReadRequest& request,
      const EndpointBinding& local,
      const EndpointBinding& remote) override;

  absl::Status close(const EndpointBinding& remote) override;

 private:
  std::shared_ptr<engine::Communicator> engine_;
};

class PcieAdapter final : public ConnectionAdapter {
 public:
  explicit PcieAdapter(std::shared_ptr<engine::Communicator> engine);

  ConnectionProtocol protocol() const override {
    return ConnectionProtocol::kPcie;
  }

  bool is_available() const override {
    return engine_ != nullptr;
  }

  transport::future_read_result_t read_tensor(
      const ReadRequest& request,
      const EndpointBinding& local,
      const EndpointBinding& remote) override;

  absl::Status close(const EndpointBinding& remote) override;

 private:
  std::shared_ptr<engine::Communicator> engine_;
};

} // namespace tensorcast::communicator::routing

#endif // CORE_COMMUNICATOR_ROUTING_ADAPTER_H_
