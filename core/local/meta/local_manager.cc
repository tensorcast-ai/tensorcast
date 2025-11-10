// Copyright (c) 2025, TensorCast Team.

#include "core/local/meta/local_manager.h"
#include <memory>
#include <vector>
#include "core/local/chunk/data_chunk.h"
#include "core/local/loader/disk_chunk_loader.h"
#include "core/local/loader/dram_chunk_loader.h"
#include "core/local/meta/replica.h"
#include "core/local/meta/view.h"

// #include "absl/status/statusor.h"

namespace tensorcast::local::meta {

const LocalConfig LocalManager::kLocalConfig;
std::map<std::string, std::unique_ptr<Artifact>> LocalManager::kArtifacts;
std::vector<store::DeviceKey> LocalManager::kDeviceList;

void LocalManager::manual_set_devices(const std::vector<store::DeviceKey>& device_list) {
  kDeviceList = device_list;
}

absl::StatusOr<Artifact*> LocalManager::get_or_create_artifact(const std::string& artifact_id) {
  if (kArtifacts.find(artifact_id) != kArtifacts.end()) {
    return kArtifacts[artifact_id].get();
  }
  auto artifact = std::make_unique<Artifact>(artifact_id);
  auto artifact_ptr = artifact.get();
  kArtifacts.emplace(artifact_id, std::move(artifact));
  return artifact_ptr;
}

//     if (kArtifacts.find(artifact_id) != kArtifacts.end()) {
//         return absl::AlreadyExistsError("Artifact with this ID already exists");
//     }
//     auto artifact = std::make_unique<Artifact>(artifact_id);
//     auto artifact_ptr = artifact.get();
//     kArtifacts.emplace(artifact_id, std::move(artifact));
//     return artifact_ptr;
// }

absl::StatusOr<View*> LocalManager::create_view(
    Artifact* artifact,
    const std::string& view_id,
    View::ViewType view_type,
    size_t size,
    bool with_chunks) {
  if (artifact->get_view_by_id(view_id) != nullptr) {
    return absl::AlreadyExistsError("View with this ID already exists");
  }
  auto view = std::make_unique<View>(view_id, "", size, "", artifact, view_type, std::vector<std::shared_ptr<Chunk>>{});
  View* view_ptr = view.get();
  artifact->add_view(std::move(view));
  // return view_ptr;
  // }

  // absl::StatusOr<View*> LocalManager::create_view_with_chunks(Artifact* artifact, const std::string& view_id,
  // View::ViewType view_type, size_t size) { auto st = create_view(artifact, view_id, view_type, size); if (!st.ok()) {
  //     return st.status();
  // }
  // auto *view = st.value();
  for (off_t i = 0; i < size; i += LocalManager::kLocalConfig.chunk_size) {
    auto chunk = std::make_shared<Chunk>(LocalManager::kLocalConfig.chunk_size, artifact);
    chunk->generate_data_chunks(kDeviceList);
    auto s = view_ptr->bind_chunk_at(i, chunk);
    if (!s.ok()) {
      return s;
    }
  }
  return view_ptr;
}

absl::Status LocalManager::bind_replica_source(const Replica& replica, void* cpu_base) {
  for (auto it = replica.begin(); it != replica.end(); ++it) {
    auto* data_chunk = *it;
    void* chunk_base = static_cast<char*>(cpu_base) + it.get_offset();
    auto loader = std::make_shared<data::DramChunkLoader>(data_chunk, chunk_base);
    data_chunk->register_loader(loader, data::DataChunk::LoaderPriority::High);
    // absl::Status s = data_chunk->bind_source(chunk_base);
    // if (!s.ok()) {
    //     return s;
    // }
  }
  return absl::OkStatus();
}

absl::Status LocalManager::bind_replica_source(const Replica& replica, std::string file) {
  for (auto it = replica.begin(); it != replica.end(); ++it) {
    auto* data_chunk = *it;
    auto loader = std::make_shared<data::DiskChunkLoader>(data_chunk, file, it.get_offset());
    data_chunk->register_loader(loader, data::DataChunk::LoaderPriority::High);
  }
  return absl::OkStatus();
}

// absl::Status LocalManager::bind_replica_source(const Replica& replica, common::memory::GpuDeviceMemory* gpu_memory) {

absl::StatusOr<std::unique_ptr<ReplicaHandler>> LocalManager::load_replica(const Replica& replica) {
  // std::vector<data::DataChunk*> data_chunks;
  absl::Status s = absl::OkStatus();
  auto lease = std::make_unique<data::ChunkPinLease>();
  for (auto it = replica.begin(); it != replica.end(); ++it) {
    auto* data_chunk = *it;
    s = lease->pin(data_chunk);
    if (!s.ok()) {
      // return absl::InternalError("Failed to pin data chunk");
      return s;
    }
    s = data_chunk->load();
    if (!s.ok()) {
      // return absl::InternalError("Failed to load data chunk");
      return s;
    }
    // data_chunks.push_back(data_chunk);
  }
  auto handler = std::make_unique<ReplicaHandler>(std::move(lease), replica);
  return handler;
  // data::ChunkPinLease lease = data::ChunkPinLease(std::move(data_chunks));
}

//   for (off_t i = 0; i < view_size_; i += LocalManager::kLocalConfig.chunk_size) {
//     auto chunk = std::make_shared<Chunk>(LocalManager::kLocalConfig.chunk_size, artifact_);
//     auto s = bind_chunk_at(i, chunk);
//     assert(s.ok());
//   }
} // namespace tensorcast::local::meta