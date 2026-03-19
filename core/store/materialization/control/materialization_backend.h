// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/communication_types.h"
#include "core/store/materialization/contracts/loading_spec.h"

namespace tensorcast::store::materialization::control {

using tensorcast::store::P2PSource;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::loading::MaterializeHints;
using tensorcast::store::loading::ReplicaHandle;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::loading::ReplicaTarget;

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

  // Allow orchestrators to inject runtime-owned transport handles when they
  // build P2PSource directly from control-plane metadata.
  virtual void prepare_p2p_source(P2PSource* source) const {}

  virtual absl::StatusOr<ReplicaHandle> ingest_from_disk(
      const std::string& artifact_identifier,
      const DiskSource& source,
      const ReplicaTarget& target,
      const MaterializeHints& hints) = 0;

  virtual absl::Status register_replica_with_global_store(
      const ReplicaKey& key,
      std::string_view artifact_id_override,
      std::string_view publish_context_id = {}) = 0;
};

} // namespace tensorcast::store::materialization::control
