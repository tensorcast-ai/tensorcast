// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/common/artifact_identity.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/memory_location.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/device_types.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/replica/replica.h"
#include "core/store/view_utils.h"
#include "gsl/pointers"

namespace tensorcast::store::runtime::metadata {

enum class ViewPlacement : uint8_t { kUnspecified = 0, kServer = 1, kClient = 2 };

using CanonicalRange = tensorcast::store::view::CanonicalRange;

struct ViewRegistration {
  std::string view_id;
  loader::ViewSpec spec;
  ViewPlacement placement{ViewPlacement::kUnspecified};
  uint64_t canonical_size_bytes{0};
  std::vector<CanonicalRange> canonical_ranges;
  bool allow_partial{false};
};

struct ArtifactRegistration {
  std::string artifact_id;
  std::string tensor_index_key;
  std::optional<std::string> tensor_index_data;
  std::string schema_version{"v3"};
  std::string encoding{"json"};
  int device_id{0};
  uint64_t total_size_bytes{0};
  bool enable_p2p{true};
  uint32_t ttl_ms{0};
  std::optional<std::string> client_artifact_id;
  std::optional<ViewRegistration> view;
};

struct RegistrationBeginResult {
  std::string registration_id;
  std::array<std::byte, sizeof(cudaIpcMemHandle_t)> cuda_ipc_handle_bytes{};
  int device_id{0};
  uint64_t size_bytes{0};
};

struct RegistrationCommitResult {
  std::string registration_id;
  std::string artifact_id;
  int device_id{0};
  DeviceKey device;
  uint64_t size_bytes{0};
  bool existed{false};
  std::string index_multihash;
  std::string data_multihash;
  std::string schema_version;
  std::string encoding;
  std::optional<std::string> view_id;
  std::optional<std::string> view_data_multihash;
  std::optional<std::string> view_index_json;
  std::vector<CanonicalRange> canonical_ranges;
  bool allow_partial{false};
  common::ArtifactIdKind id_kind{common::ArtifactIdKind::kMi2};
};

struct RegistrationResources {
  gsl::not_null<components::DeviceManager*> device_manager;
  gsl::not_null<components::ReplicaRegistry*> replica_registry;
  gsl::not_null<components::MetricsCollector*> metrics_collector;
  gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> memory_pool;
  std::shared_ptr<components::CommunicationManager> communication_manager;
  std::shared_ptr<common::AsyncRuntime> async_runtime;
};

using ReplicaFactory = std::function<absl::StatusOr<std::shared_ptr<replica::Replica>>(const replica::ReplicaConfig&)>;

struct RegistrationPublication {
  std::string artifact_id;
  DeviceKey device;
  uint64_t size_bytes{0};
  std::string tensor_index_key;
  std::vector<std::string> remote_memory_keys;
  std::vector<uint64_t> buffer_sizes;
  std::optional<std::string> tensor_index_data;
  std::string encoding{"json"};
  std::string schema_version{"v3"};
  std::optional<std::string> verification_json;
};

class RegistrationPublisher {
 public:
  virtual ~RegistrationPublisher() = default;
  virtual absl::Status publish_registration(const RegistrationPublication& publication) = 0;
  virtual absl::Status update_variant_view(const components::VariantViewUpdate& update) = 0;
};

} // namespace tensorcast::store::runtime::metadata
