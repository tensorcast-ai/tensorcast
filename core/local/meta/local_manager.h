// Copyright (c) 2025, TensorCast Team.

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include "absl/status/statusor.h"
#include "core/local/chunk/chunk.h"
#include "core/local/meta/artifact.h"
#include "core/local/meta/replica.h"
#include "core/local/meta/view.h"

namespace tensorcast::local::meta {

class LocalConfig {
 public:
  size_t chunk_size{2 * (1 << 20)}; // 2MB
};

class LocalManager {
 public:
  static const LocalConfig kLocalConfig;
  // LocalManager();
  // ~LocalManager();
  static void detect_devices();
  static void manual_set_devices(const std::vector<store::DeviceKey>& device_list);
  static absl::StatusOr<Artifact*> get_or_create_artifact(const std::string& artifact_id);
  static absl::StatusOr<View*> create_view(
      Artifact* artifact,
      const std::string& view_id,
      View::ViewType view_type,
      size_t size,
      bool with_chunks = false);
  // static absl::StatusOr<View*> create_view_with_chunks(Artifact* artifact, const std::string& view_id, View::ViewType
  // view_type, size_t size);
  static absl::Status bind_replica_source(const Replica& replica, void* cpu_base);
  static absl::Status bind_replica_source(const Replica& replica, std::string file);
  // TODO: implement this
  // static absl::Status bind_replica_source(const Replica& replica, common::memory::GpuDeviceMemory* gpu_memory);

  static absl::StatusOr<std::unique_ptr<ReplicaHandler>> load_replica(const Replica& replica);
  //   absl::Status create_replica(const std::string& artifact_id, const std::string& replica_id);
 private:
  static std::map<std::string, std::unique_ptr<Artifact>> kArtifacts;
  static std::vector<store::DeviceKey> kDeviceList;
};

} // namespace tensorcast::local::meta