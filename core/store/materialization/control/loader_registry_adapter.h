// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>

#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/contracts/registry.h"

namespace tensorcast::store::materialization::control {

// Temporary adapter that instantiates loaders using the legacy dataplane
// factories while the new registry implementation lands.
class LoaderRegistryAdapter : public materialization::IArtifactLoaderRegistry {
 public:
  absl::StatusOr<std::unique_ptr<IArtifactLoader>> CreateLoader(
      const loading::ArtifactSource& source,
      const loading::MaterializeHints& hints) override;
};

} // namespace tensorcast::store::materialization::control
