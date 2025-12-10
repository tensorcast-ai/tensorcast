// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <filesystem>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "core/store/runtime/ingestion/ingestion_runtime.h"
#include "core/store/runtime/metadata/metadata_gateway.h"
#include "core/store/runtime/replica/replica_runtime.h"
#include "core/store/runtime/runtime_env.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::runtime::ingestion::testing {

namespace metadata = tensorcast::store::runtime::metadata;

class ScopedIngestionRuntimeTestHarness {
 public:
  explicit ScopedIngestionRuntimeTestHarness(StoreEngineOptions options) : options_(std::move(options)) {}

  ~ScopedIngestionRuntimeTestHarness() {
    shutdown();
  }

  ScopedIngestionRuntimeTestHarness(const ScopedIngestionRuntimeTestHarness&) = delete;
  ScopedIngestionRuntimeTestHarness& operator=(const ScopedIngestionRuntimeTestHarness&) = delete;

  absl::Status initialize() {
    if (env_ != nullptr) {
      return absl::FailedPreconditionError("harness already initialized");
    }
    env_ = std::make_unique<RuntimeEnv>(options_);
    auto status = env_->initialize();
    if (!status.ok()) {
      env_.reset();
      return status;
    }
    replica_runtime_ =
        std::make_unique<ReplicaRuntime>(ReplicaRuntime::Config{.runtime_context = &env_->runtime_context()});
    metadata_gateway_ = std::make_unique<metadata::MetadataGateway>(metadata::MetadataGateway::Config{
        .runtime_context = &env_->runtime_context(),
        .replica_runtime = replica_runtime_.get(),
        .artifact_chunk_bytes = options_.artifact_chunk_bytes,
        .pinned_memory_timeout = options_.pinned_memory_timeout,
        .replica_factory = {},
    });
    return absl::OkStatus();
  }

  void shutdown() {
    if (env_ != nullptr) {
      // Ensure no in-flight ingestion callbacks target torn-down subscribers.
      env_->runtime_context().drain_events();
    }
    if (replica_runtime_) {
      replica_runtime_->clear_mem();
      replica_runtime_.reset();
    }
    metadata_gateway_.reset();
    if (env_) {
      env_->shutdown();
      env_.reset();
    }
  }

  [[nodiscard]] RuntimeContext& runtime_context() {
    ABSL_CHECK(env_ != nullptr) << "initialize must be called first";
    return env_->runtime_context();
  }

  [[nodiscard]] ReplicaRuntime& replica_runtime() {
    ABSL_CHECK(replica_runtime_ != nullptr) << "initialize must be called first";
    return *replica_runtime_;
  }

  [[nodiscard]] metadata::MetadataGateway& metadata_gateway() {
    ABSL_CHECK(metadata_gateway_ != nullptr) << "initialize must be called first";
    return *metadata_gateway_;
  }

  [[nodiscard]] const StoreEngineOptions& options() const {
    return options_;
  }

  [[nodiscard]] IngestionRuntime::Config make_runtime_config(
      std::shared_ptr<const IngestionRuntimeDependencies> dependencies = nullptr) const {
    ABSL_CHECK(env_ != nullptr) << "initialize must be called first";
    ABSL_CHECK(replica_runtime_ != nullptr) << "initialize must be called first";
    ABSL_CHECK(metadata_gateway_ != nullptr) << "initialize must be called first";
    RuntimeEnv* env = env_.get();
    RuntimeContext* context = &env->runtime_context();
    IngestionRuntime::Config config{
        .runtime_context = context,
        .replica_runtime = replica_runtime_.get(),
        .metadata_gateway = metadata_gateway_.get(),
        .storage_path = std::filesystem::path(options_.storage_path),
        .artifact_chunk_bytes = options_.artifact_chunk_bytes,
        .pinned_memory_timeout = options_.pinned_memory_timeout,
        .num_threads = options_.num_thread,
        .options = &options_,
        .dependencies = std::move(dependencies),
    };
    return config;
  }

 private:
  StoreEngineOptions options_;
  std::unique_ptr<RuntimeEnv> env_;
  std::unique_ptr<ReplicaRuntime> replica_runtime_;
  std::unique_ptr<metadata::MetadataGateway> metadata_gateway_;
};

} // namespace tensorcast::store::runtime::ingestion::testing
