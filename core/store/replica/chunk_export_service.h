// Copyright (c) 2025, StepCast Team. All rights reserved.

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

namespace stepcast::communicator {
class CommunicateEngine;
}

namespace stepcast::store {

class ChunkExportService {
 public:
  ChunkExportService(
      std::shared_ptr<ReplicaMemoryCoordinator> uma,
      gsl::not_null<std::shared_ptr<memory::DistributedVirtualMemoryPool>> dvmp)
      : uma_(std::move(uma)), dvmp_(std::move(dvmp)) {}

  absl::StatusOr<CommRegistrationInfo> export_chunks(
      const ReplicaKey& key,
      MemoryLocation location,
      absl::Span<const uint32_t> chunks,
      communicator::CommunicateEngine& comm_engine);

  absl::Status unexport_chunks(
      const ReplicaKey& key,
      const CommRegistrationInfo& info,
      communicator::CommunicateEngine& comm_engine);

 private:
  static std::vector<std::pair<uint32_t, uint32_t>> coalesce_ranges(std::vector<uint32_t> chunks);

  struct ExportKey {
    ReplicaKey key;
    MemoryLocation location;
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

  std::shared_ptr<ReplicaMemoryCoordinator> uma_;
  gsl::not_null<std::shared_ptr<memory::DistributedVirtualMemoryPool>> dvmp_;

  // Cache per (ReplicaKey, Location) to support precise unexport and lease lifetime
  std::unordered_map<ExportKey, ExportRecord, ExportKeyHash> records_;
  std::mutex records_mu_;
};

} // namespace stepcast::store