// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/common/artifact_identity.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/cuda/cuda_ipc.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/components/stable_dram_cache_manager.h"
#include "core/store/components/stable_dram_cache_policy.h"
#include "core/store/device_types.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/memory_tier_budget.h"
#include "core/store/memory_tier_config.h"
#include "core/store/replica/replica.h"
#include "core/store/runtime/replica/replica_promotion_manager.h"
#include "core/store/store_engine_options.h"
#include "core/store/view_utils.h"
#include "gsl/pointers"

namespace tensorcast::store::runtime::metadata {

enum class ViewPlacement : uint8_t { kUnspecified = 0, kServer = 1, kClient = 2 };

using CanonicalRange = tensorcast::store::view::CanonicalRange;

enum class ViewRegistrationKind : uint8_t { kUnspecified = 0, kCanonical = 1, kPiece = 2 };

enum class RegistrationPlan : uint8_t { kCoalesced = 0, kStableDram = 1 };

struct StableDramOptions {
  bool stage_on_gpu{true};
  bool release_gpu_on_commit{true};
};

struct ViewRegistration {
  std::string view_id;
  loader::ViewSpec spec;
  ViewPlacement placement{ViewPlacement::kUnspecified};
  uint64_t canonical_size_bytes{0};
  std::vector<CanonicalRange> canonical_ranges;
  ViewRegistrationKind registration_kind{ViewRegistrationKind::kUnspecified};
};

struct ArtifactRegistration {
  std::string artifact_id;
  std::string tensor_index_key;
  std::optional<std::string> tensor_index_data;
  std::string schema_version{"v3"};
  std::string encoding{"json"};
  RegistrationPlan plan{RegistrationPlan::kCoalesced};
  StableDramOptions stable_dram;
  std::optional<components::StableDramCachePolicy> stable_cache_policy;
  int device_id{0};
  uint64_t total_size_bytes{0};
  bool enable_p2p{true};
  uint32_t ttl_ms{0};
  std::optional<std::string> client_artifact_id;
  // When set, commit will reuse the provided canonical `mi2:` artifact id
  // without recomputing index/data multihashes. Used by daemon-side local
  // stable materialization paths.
  std::optional<std::string> artifact_id_override;
  std::optional<ViewRegistration> view;
};

struct RegistrationBeginResult {
  std::string registration_id;
  cuda::IpcHandleBytes cuda_ipc_handle_bytes{};
  int device_id{0};
  uint64_t size_bytes{0};
};

struct RegistrationCpuMemfdInfo {
  loading::ReplicaKey replica_key;
  int fd{-1};
  uint64_t size_bytes{0};
  uint64_t offset_bytes{0};
};

struct RegistrationCommitResult {
  std::string registration_id;
  std::string artifact_id;
  int device_id{0};
  DeviceKey device;
  uint64_t size_bytes{0};
  bool existed{false};
  bool stable_cache_admitted{false};
  std::string index_multihash;
  std::string data_multihash;
  std::string schema_version;
  std::string encoding;
  std::optional<std::string> view_id;
  std::optional<std::string> view_data_multihash;
  std::optional<std::string> view_index_json;
  std::vector<CanonicalRange> canonical_ranges;
  ViewRegistrationKind registration_kind{ViewRegistrationKind::kUnspecified};
  common::ArtifactIdKind id_kind{common::ArtifactIdKind::kMi2};
};

struct RegistrationResources {
  gsl::not_null<components::DeviceManager*> device_manager;
  gsl::not_null<components::ReplicaRegistry*> replica_registry;
  gsl::not_null<components::MetricsCollector*> metrics_collector;
  gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> memory_pool;
  std::shared_ptr<components::CommunicationManager> communication_manager;
  std::shared_ptr<components::StableDramCacheManager> stable_cache_manager;
  std::shared_ptr<common::AsyncRuntime> async_runtime;
  std::shared_ptr<MemoryTierBudget> memory_tier_budget;
  std::optional<MemoryTierConfig> memory_tier_config;
  StoreEngineOptions::ByteMappingConfig byte_mapping_config{};
  ::tensorcast::store::runtime::ReplicaPromotionManager* promotion_manager{nullptr};
  bool cpu_shared_memory_enabled{false};
};

using ReplicaFactory = std::function<absl::StatusOr<std::shared_ptr<replica::Replica>>(const replica::ReplicaConfig&)>;

struct RegistrationPublication {
  std::string artifact_id;
  DeviceKey device;
  uint64_t size_bytes{0};
  std::optional<std::string> view_id;
  std::string tensor_index_key;
  std::vector<std::string> remote_memory_keys;
  std::vector<uint64_t> buffer_sizes;
  std::optional<std::string> tensor_index_data;
  std::string encoding{"json"};
  std::string schema_version{"v3"};
  std::optional<std::string> verification_json;
  std::string index_multihash;
  std::string data_multihash;
  common::ArtifactIdKind id_kind{common::ArtifactIdKind::kMi2};
};

class RegistrationPublisher {
 public:
  virtual ~RegistrationPublisher() = default;
  virtual absl::Status publish_registration(const RegistrationPublication& publication) = 0;
  virtual absl::Status update_view_state(const components::ViewStateUpdate& update) = 0;
};

} // namespace tensorcast::store::runtime::metadata
