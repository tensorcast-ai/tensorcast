// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/worker_identity.h"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::runtime {

class RuntimeEnv {
 public:
  explicit RuntimeEnv(StoreEngineOptions options);
  ~RuntimeEnv();

  RuntimeEnv(const RuntimeEnv&) = delete;
  RuntimeEnv& operator=(const RuntimeEnv&) = delete;

  absl::Status initialize();
  void shutdown();

  RuntimeContext& runtime_context();
  const RuntimeContext& runtime_context() const;

  void update_worker_identity(
      std::string worker_id,
      std::string node_id,
      std::string node_address,
      uint32_t grpc_port,
      uint32_t p2p_port);

  const components::WorkerIdentity& worker_identity() const;

  void set_global_store_client_for_testing(std::shared_ptr<components::IGlobalStoreClient> client);

 private:
  StoreEngineOptions options_;
  std::unique_ptr<RuntimeContext> context_;

  mutable absl::Mutex lifecycle_mu_;
  bool started_ ABSL_GUARDED_BY(lifecycle_mu_) = false;
};

} // namespace tensorcast::store::runtime
