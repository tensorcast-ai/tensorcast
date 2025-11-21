// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>

#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/contracts/materialization_request.h"

namespace tensorcast::store {

class IArtifactLoader;

namespace components {
class IGlobalStoreClient;
} // namespace components

namespace materialization {

class IArtifactLoaderRegistry {
 public:
  virtual ~IArtifactLoaderRegistry() = default;
  virtual absl::StatusOr<std::unique_ptr<IArtifactLoader>> CreateLoader(
      const loading::ArtifactSource& source,
      const loading::MaterializeHints& hints) = 0;
};

class ArtifactSourceRouter {
 public:
  virtual ~ArtifactSourceRouter() = default;
  virtual absl::StatusOr<loading::ArtifactSource> SelectSource(
      const loading::MaterializationRequest& request,
      components::IGlobalStoreClient& gs_client) = 0;
};

} // namespace materialization
} // namespace tensorcast::store
