// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "core/common/memory/memory_location.h"
#include "core/store/communication_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/replica/replica.h"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/runtime/replica/replica_runtime.h"
#include "core/store/store_engine_options.h"
#include "gsl/pointers"

namespace tensorcast::store::materialization::runtime::pipeline {

namespace store_runtime = tensorcast::store::runtime;

enum class SourceType { kDisk, kP2P };

struct DiskSourceMetadata {
  loading::DiskSource source;
  std::filesystem::path artifact_path;
  bool descriptor_present{false};
  bool is_safetensors{false};
  std::optional<std::string> schema_version;
  std::optional<std::string> existing_index_multihash;
  std::optional<std::string> existing_data_multihash;
};

struct P2PSourceMetadata {
  P2PSource source;
};

struct VerificationState {
  std::optional<std::string> computed_index_multihash;
  std::optional<std::string> computed_data_multihash;
  std::optional<std::string> view_data_hash;
  std::optional<std::string> canonical_index_json;
  uint64_t logical_total_size{0};
};

struct IngestionContext {
  SourceType source_type;
  std::string request_id;
  std::string publish_context_id;
  loading::MaterializeMode materialize_mode{loading::MaterializeMode::AUTO};
  std::string artifact_identifier;
  loading::ReplicaTarget target;
  loading::MaterializeHints hints;
  DeviceKey target_device;
  common::memory::MemoryLocation target_location{common::memory::MemoryLocation::CPU};
  int target_device_id{-1};
  bool target_is_gpu{false};

  std::filesystem::path storage_path;
  size_t artifact_chunk_bytes{0};
  size_t tx_slice_bytes{0};
  int num_threads{0};
  std::chrono::milliseconds pinned_memory_timeout{0};
  const StoreEngineOptions* options{nullptr};
  store_runtime::ReplicaRuntime* replica_runtime{nullptr};
  store_runtime::RuntimeContext* runtime_context{nullptr};

  DiskSourceMetadata disk;
  P2PSourceMetadata p2p;
  VerificationState verification;
  std::optional<loader::ViewPlan> resolved_view_plan;

  std::shared_ptr<replica::Replica> replica;
  std::shared_future<absl::Status> load_future;

  std::chrono::steady_clock::time_point start_time;
  bool publish_to_global_store{true};
};

} // namespace tensorcast::store::materialization::runtime::pipeline
