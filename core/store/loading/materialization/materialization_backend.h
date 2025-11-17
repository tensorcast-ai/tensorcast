// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/loading/loading_spec.h"

namespace tensorcast::store::loading {

// Thin interface that exposes the ingestion primitives used by materialization
// helpers. StoreEngine implements this to keep orchestration decoupled from
// the full engine surface area.
class MaterializationBackend {
 public:
  virtual ~MaterializationBackend() = default;

  virtual absl::StatusOr<ReplicaHandle> ingest_from_p2p(
      const std::string& artifact_identifier,
      const P2PSource& source,
      const ReplicaTarget& target,
      const MaterializeHints& hints) = 0;

  virtual absl::StatusOr<ReplicaHandle> ingest_from_disk(
      const std::string& artifact_identifier,
      const DiskSource& source,
      const ReplicaTarget& target,
      const MaterializeHints& hints) = 0;

  virtual absl::Status register_replica_with_global_store(
      const ReplicaKey& key,
      std::string_view artifact_id_override) = 0;
};

} // namespace tensorcast::store::loading
