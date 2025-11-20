// Copyright (c) 2025, TensorCast Team.

#include "core/store/runtime/runtime_env.h"

#include <utility>

namespace tensorcast::store::runtime {

RuntimeEnv::RuntimeEnv(StoreEngineOptions options)
    : options_(std::move(options)), context_(std::make_unique<RuntimeContext>(options_)) {}

RuntimeEnv::~RuntimeEnv() {
  shutdown();
}

absl::Status RuntimeEnv::initialize() {
  absl::MutexLock lock(&lifecycle_mu_);
  if (started_) {
    return absl::OkStatus();
  }
  auto status = context_->start();
  if (!status.ok()) {
    return status;
  }
  started_ = true;
  return absl::OkStatus();
}

void RuntimeEnv::shutdown() {
  {
    absl::MutexLock lock(&lifecycle_mu_);
    if (!started_) {
      return;
    }
    started_ = false;
  }

  if (context_) {
    context_->shutdown();
  }
}

RuntimeContext& RuntimeEnv::runtime_context() {
  return *context_;
}

const RuntimeContext& RuntimeEnv::runtime_context() const {
  return *context_;
}

void RuntimeEnv::update_worker_identity(
    std::string worker_id,
    std::string node_id,
    std::string node_address,
    uint32_t grpc_port,
    uint32_t p2p_port) {
  components::WorkerIdentity identity{
      .worker_id = std::move(worker_id),
      .node_id = std::move(node_id),
      .node_address = std::move(node_address),
      .grpc_port = grpc_port,
      .p2p_port = p2p_port,
  };
  context_->set_worker_identity(std::move(identity));
}

const components::WorkerIdentity& RuntimeEnv::worker_identity() const {
  return context_->worker_identity();
}

void RuntimeEnv::set_global_store_client_for_testing(std::shared_ptr<components::IGlobalStoreClient> client) {
  context_->set_global_store_client_for_testing(std::move(client));
}

} // namespace tensorcast::store::runtime
