// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/worker_identity.h"
#include "core/store/runtime/component_catalog.h"
#include "core/store/runtime/runtime_event_hub.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::runtime {

class RuntimeEnv {
 public:
  explicit RuntimeEnv(StoreEngineOptions options);
  ~RuntimeEnv();

  RuntimeEnv(const RuntimeEnv&) = delete;
  RuntimeEnv& operator=(const RuntimeEnv&) = delete;

  absl::Status Initialize();
  void Shutdown();

  ComponentCatalog& component_catalog();
  const ComponentCatalog& component_catalog() const;
  RuntimeEventHub& event_hub();
  const RuntimeEventHub& event_hub() const;

  void UpdateWorkerIdentity(
      std::string worker_id,
      std::string node_id,
      std::string node_address,
      uint32_t grpc_port,
      uint32_t p2p_port);

  const components::WorkerIdentity& worker_identity() const;

  void RegisterShutdownDependency(std::string name, absl::AnyInvocable<void()> hook);
  void set_global_store_client_for_testing(std::shared_ptr<components::IGlobalStoreClient> client);

 private:
  void RunShutdownHooks();

  StoreEngineOptions options_;
  std::unique_ptr<ComponentCatalog> component_catalog_;
  std::unique_ptr<RuntimeEventHub> event_hub_;

  mutable absl::Mutex lifecycle_mu_;
  bool started_ ABSL_GUARDED_BY(lifecycle_mu_) = false;
  std::vector<std::pair<std::string, absl::AnyInvocable<void()>>> shutdown_hooks_ ABSL_GUARDED_BY(lifecycle_mu_);
};

} // namespace tensorcast::store::runtime
