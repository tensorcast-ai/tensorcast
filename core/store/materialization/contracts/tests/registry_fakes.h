// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <functional>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/registry.h"

namespace tensorcast::store::materialization::testing {

class FakeLoaderRegistry : public IArtifactLoaderRegistry {
 public:
  std::function<absl::StatusOr<std::unique_ptr<IArtifactLoader>>(
      const loading::ArtifactSource&,
      const loading::MaterializeHints&)>
      on_create;

  absl::StatusOr<std::unique_ptr<IArtifactLoader>> CreateLoader(
      const loading::ArtifactSource& source,
      const loading::MaterializeHints& hints) override {
    last_source = source;
    last_hints = hints;
    if (on_create) {
      return on_create(source, hints);
    }
    return absl::UnimplementedError("FakeLoaderRegistry::on_create not set");
  }

  std::optional<loading::ArtifactSource> last_source;
  std::optional<loading::MaterializeHints> last_hints;
};

class FakeArtifactSourceRouter : public ArtifactSourceRouter {
 public:
  std::function<
      absl::StatusOr<loading::ArtifactSource>(const loading::MaterializationRequest&, components::IGlobalStoreClient&)>
      on_select;

  absl::StatusOr<loading::ArtifactSource> SelectSource(
      const loading::MaterializationRequest& request,
      components::IGlobalStoreClient& client) override {
    last_request = request;
    if (on_select) {
      return on_select(request, client);
    }
    return absl::UnimplementedError("FakeArtifactSourceRouter::on_select not set");
  }

  std::optional<loading::MaterializationRequest> last_request;
};

} // namespace tensorcast::store::materialization::testing
