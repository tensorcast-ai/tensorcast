// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/control/loader_registry_adapter.h"

#include <utility>

#include "absl/functional/overload.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/store/materialization/dataplane/contracts/inline_buffer_loader.h"
#include "core/store/materialization/dataplane/loaders/disk_loader.h"
#include "core/store/materialization/dataplane/loaders/p2p_loader.h"

namespace tensorcast::store::materialization::control {

absl::StatusOr<std::unique_ptr<IArtifactLoader>> LoaderRegistryAdapter::CreateLoader(
    const loading::ArtifactSource& source,
    const loading::MaterializeHints& /*hints*/) {
  std::unique_ptr<IArtifactLoader> loader;
  absl::Status status = absl::OkStatus();

  try {
    status = std::visit(
        absl::Overload{
            [&](const loading::DiskSource& disk_source) -> absl::Status {
              loader = std::make_unique<DiskLoader>(disk_source);
              return absl::OkStatus();
            },
            [&](const P2PSource& p2p_source) -> absl::Status {
              loader = std::make_unique<P2PLoader>(p2p_source);
              return absl::OkStatus();
            },
            [&](const loading::InlineBufferSource& buffer_source) -> absl::Status {
              loader = std::make_unique<InlineBufferLoader>(buffer_source);
              return absl::OkStatus();
            }},
        source);
  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrCat("Failed to create loader: ", e.what()));
  }

  if (!status.ok()) {
    return status;
  }
  if (!loader) {
    return absl::InternalError("LoaderRegistryAdapter failed to create loader for source variant");
  }
  return loader;
}

} // namespace tensorcast::store::materialization::control
