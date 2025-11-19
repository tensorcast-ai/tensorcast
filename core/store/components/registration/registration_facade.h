// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/common/artifact_identity.h"
#include "core/common/memory/memory_location.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/device_types.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/replica/replica.h"
#include "core/store/runtime/global_metadata_gateway.h"
#include "core/store/view_utils.h"
#include "gsl/pointers"

namespace tensorcast::store::components {

enum class ViewPlacement : uint8_t { kUnspecified = 0, kServer = 1, kClient = 2 };

using CanonicalRange = view::CanonicalRange;

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
  gsl::not_null<DeviceManager*> device_manager;
  gsl::not_null<ReplicaRegistry*> replica_registry;
  gsl::not_null<MetricsCollector*> metrics_collector;
  gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> memory_pool;
  std::shared_ptr<CommunicationManager> communication_manager;
};

using ReplicaFactory = std::function<absl::StatusOr<std::shared_ptr<replica::Replica>>(const replica::ReplicaConfig&)>;

class RegistrationFacade {
 public:
  RegistrationFacade(
      RegistrationResources resources,
      ReplicaFactory replica_factory,
      size_t artifact_chunk_bytes,
      std::chrono::milliseconds pinned_memory_timeout,
      runtime::GlobalMetadataGateway* metadata_gateway);

  RegistrationFacade(const RegistrationFacade&) = delete;
  RegistrationFacade& operator=(const RegistrationFacade&) = delete;

  absl::StatusOr<RegistrationBeginResult> begin(const ArtifactRegistration& reg);
  absl::StatusOr<RegistrationCommitResult> commit(std::string_view registration_id);
  absl::Status abort(std::string_view registration_id);
  absl::Status keep_alive(std::string_view registration_id, uint32_t ttl_ms);
  absl::Status ingest_view_chunk(
      std::string_view registration_id,
      uint64_t view_offset,
      absl::Span<const std::byte> data);
  absl::StatusOr<uint64_t> get_view_ingested_bytes(std::string_view registration_id) const;

 private:
  struct PendingRegistrationContext;

  std::shared_ptr<PendingRegistrationContext> erase_pending(
      std::string_view registration_id,
      size_t* pending_size_after = nullptr);
  std::shared_ptr<PendingRegistrationContext> lookup_pending(std::string_view registration_id) const;
  static void release_replica_memory(
      const std::shared_ptr<replica::Replica>& replica,
      common::memory::MemoryLocation location);
  void record_pending_gauge(size_t pending_count) const;
  void record_commit_latency(const PendingRegistrationContext& ctx, std::string_view status) const;

  gsl::not_null<DeviceManager*> device_manager_;
  gsl::not_null<ReplicaRegistry*> replica_registry_;
  gsl::not_null<MetricsCollector*> metrics_collector_;
  gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> memory_pool_;
  ReplicaFactory replica_factory_;
  size_t artifact_chunk_bytes_{0};
  std::chrono::milliseconds pinned_memory_timeout_{0};
  std::shared_ptr<CommunicationManager> communication_manager_;
  runtime::GlobalMetadataGateway* metadata_gateway_;

  mutable std::mutex pending_mutex_;
  absl::flat_hash_map<std::string, std::shared_ptr<PendingRegistrationContext>> pending_regs_;
};

} // namespace tensorcast::store::components
