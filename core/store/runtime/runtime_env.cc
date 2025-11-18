// Copyright (c) 2025, TensorCast Team.

#include "core/store/runtime/runtime_env.h"

#include <utility>

#include "absl/log/log.h"

namespace tensorcast::store::runtime {

RuntimeEnv::RuntimeEnv(StoreEngineOptions options)
    : options_(std::move(options)),
      component_catalog_(std::make_unique<ComponentCatalog>(options_)),
      event_hub_(std::make_unique<RuntimeEventHub>()) {}

RuntimeEnv::~RuntimeEnv() {
  Shutdown();
}

absl::Status RuntimeEnv::Initialize() {
  absl::MutexLock lock(&lifecycle_mu_);
  if (started_) {
    return absl::OkStatus();
  }
  auto status = component_catalog_->start();
  if (!status.ok()) {
    return status;
  }
  started_ = true;
  return absl::OkStatus();
}

void RuntimeEnv::Shutdown() {
  std::vector<absl::AnyInvocable<void()>> hooks;
  {
    absl::MutexLock lock(&lifecycle_mu_);
    if (!started_) {
      return;
    }
    started_ = false;
    for (auto it = shutdown_hooks_.rbegin(); it != shutdown_hooks_.rend(); ++it) {
      hooks.push_back(std::move(it->second));
    }
    shutdown_hooks_.clear();
  }

  for (auto& hook : hooks) {
    if (hook) {
      hook();
    }
  }

  if (event_hub_) {
    event_hub_->drain();
  }
  if (component_catalog_) {
    component_catalog_->shutdown();
  }
}

ComponentCatalog& RuntimeEnv::component_catalog() {
  return *component_catalog_;
}

const ComponentCatalog& RuntimeEnv::component_catalog() const {
  return *component_catalog_;
}

RuntimeEventHub& RuntimeEnv::event_hub() {
  return *event_hub_;
}

const RuntimeEventHub& RuntimeEnv::event_hub() const {
  return *event_hub_;
}

void RuntimeEnv::UpdateWorkerIdentity(
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
  component_catalog_->set_worker_identity(std::move(identity));
}

const components::WorkerIdentity& RuntimeEnv::worker_identity() const {
  return component_catalog_->worker_identity();
}

void RuntimeEnv::RegisterShutdownDependency(std::string name, absl::AnyInvocable<void()> hook) {
  absl::MutexLock lock(&lifecycle_mu_);
  shutdown_hooks_.emplace_back(std::move(name), std::move(hook));
}

void RuntimeEnv::set_global_store_client_for_testing(std::shared_ptr<components::IGlobalStoreClient> client) {
  component_catalog_->set_global_store_client_for_testing(std::move(client));
}

} // namespace tensorcast::store::runtime
