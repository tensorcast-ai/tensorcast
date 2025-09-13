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

#include "core/common/memory/memory_location.h"
#include "core/common/memory/virtual_address_space.h"
#include "core/store/communication_types.h"
// no direct write token needed in export service
#include "core/store/replica/unified_memory_authority.h"
// OpenTelemetry Metrics API (types used in member declarations)
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/observer_result.h"

namespace tensorcast::store::replica {

class MemoryExportRegistry {
 public:
  MemoryExportRegistry(
      gsl::not_null<std::shared_ptr<UnifiedMemoryAuthority>> uma,
      gsl::not_null<std::shared_ptr<common::memory::VirtualAddressSpace>> virtual_addr_space);

  absl::StatusOr<ExportRegistration> export_chunks(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location,
      absl::Span<const uint32_t> chunks,
      communicator::engine::Communicator& comm_engine);

  absl::Status unexport_chunks(
      const loading::ReplicaKey& key,
      const ExportRegistration& info,
      communicator::engine::Communicator& comm_engine);

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
    ExportRegistration info;
    // UMA-managed keepalive for CPU VS pin leases (nullptr for GPU)
    std::shared_ptr<void> uma_keepalive;
    // Coalesced chunk ranges used for this export (for UMA ledger updates on unexport)
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
  };

  gsl::not_null<std::shared_ptr<UnifiedMemoryAuthority>> uma_;
  gsl::not_null<std::shared_ptr<common::memory::VirtualAddressSpace>> va_space_;

  // Cache per (ReplicaKey, Location) to support precise unexport and lease lifetime
  std::unordered_map<ExportKey, ExportRecord, ExportKeyHash> records_;
  std::mutex records_mu_;

  // --- Metrics ---
  // Meter and instruments for export metrics
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> meter_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> ex_reg_total_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> ex_keepalive_gauge_;

  static void keepalive_gauge_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept;
  double keepalive_count_snapshot_ = 0.0; // last snapshot used when callback cannot lock
};

} // namespace tensorcast::store::replica
