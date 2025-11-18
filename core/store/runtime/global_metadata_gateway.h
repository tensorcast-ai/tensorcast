// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "core/store/components/global_store_client.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/runtime/component_catalog.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/runtime/replica_runtime.h"
#include "core/store/runtime/runtime_event_hub.h"
#include "gsl/pointers"

namespace tensorcast::store::runtime {

class GlobalMetadataGateway {
 public:
  struct Config {
    ComponentCatalog* component_catalog;
    ReplicaRuntime* replica_runtime;
    RuntimeEventHub* event_hub = nullptr;
  };

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

  explicit GlobalMetadataGateway(Config config);

  [[nodiscard]] bool is_connected() const;
  void set_client_override(std::shared_ptr<components::IGlobalStoreClient> client);
  void refresh_override_endpoint();

  void handle_ingestion_result(const IngestionResultEvent& event);

  absl::Status register_replica(const loading::ReplicaKey& key, std::string_view artifact_id_override = {});
  absl::Status unregister_replica(std::string_view artifact_id, int device_id);

  absl::StatusOr<components::KeyMapping> resolve_key_mapping(std::string_view key) const;
  absl::StatusOr<std::string> get_canonical_index(std::string_view artifact_id) const;
  absl::Status upsert_key_mapping(
      std::string_view key,
      std::string_view artifact_id,
      std::string_view disk_path,
      absl::Duration ttl);
  absl::Status revoke_key_mapping(std::string_view key);

  absl::Status publish_registration(const RegistrationPublication& publication);
  absl::Status update_variant_view(const components::VariantViewUpdate& update);

 private:
  std::string worker_id() const;
  absl::StatusOr<std::shared_ptr<components::IGlobalStoreClient>> get_connected_client() const;

  gsl::not_null<ComponentCatalog*> component_catalog_;
  gsl::not_null<ReplicaRuntime*> replica_runtime_;
  RuntimeEventHub* event_hub_{nullptr};
  std::unique_ptr<RuntimeEventHub::Subscription> ingress_subscription_;
  std::shared_ptr<components::IGlobalStoreClient> override_client_;
};

} // namespace tensorcast::store::runtime
