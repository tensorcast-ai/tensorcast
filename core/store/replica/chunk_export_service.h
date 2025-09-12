// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gsl/pointers"

#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/memory_location.h"
#include "core/store/communication_types.h"
#include "core/store/direct_write.h"
#include "core/store/replica/replica_memory_coordinator.h"

namespace tensorcast::store::replica {

class ChunkExportService {
 public:
  ChunkExportService(
      gsl::not_null<std::shared_ptr<ReplicaMemoryCoordinator>> uma,
      gsl::not_null<std::shared_ptr<common::memory::DistributedVirtualMemoryPool>> dvmp)
      : uma_(std::move(uma)), dvmp_(std::move(dvmp)) {}

  absl::StatusOr<CommRegistrationInfo> export_chunks(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location,
      absl::Span<const uint32_t> chunks,
      communicator::engine::CommunicateEngine& comm_engine);

  absl::Status unexport_chunks(
      const loading::ReplicaKey& key,
      const CommRegistrationInfo& info,
      communicator::engine::CommunicateEngine& comm_engine);

 private:
  static std::vector<std::pair<uint32_t, uint32_t>> coalesce_ranges(std::vector<uint32_t> chunks);

  struct ExportKey {
    loading::ReplicaKey key;
    common::memory::MemoryLocation location;

    bool operator==(const ExportKey& other) const {
      return key == other.key && location == other.location;
    }
  };

  struct ExportKeyHash {
    size_t operator()(const ExportKey& k) const {
      return absl::HashOf(
          k.key.artifact_id,
          static_cast<int>(k.key.device.type),
          k.key.device.ordinal,
          k.key.replica,
          static_cast<int>(k.location));
    }
  };

  struct ExportRecord {
    CommRegistrationInfo info;
    // Keepalive tokens for CPU leases; empty for GPU
    std::vector<DirectWriteToken> cpu_tokens;
  };

  gsl::not_null<std::shared_ptr<ReplicaMemoryCoordinator>> uma_;
  gsl::not_null<std::shared_ptr<common::memory::DistributedVirtualMemoryPool>> dvmp_;

  // Cache per (ReplicaKey, Location) to support precise unexport and lease lifetime
  std::unordered_map<ExportKey, ExportRecord, ExportKeyHash> records_;
  std::mutex records_mu_;
};

} // namespace tensorcast::store::replica
