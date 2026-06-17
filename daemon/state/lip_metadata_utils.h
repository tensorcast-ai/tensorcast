// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <string>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "daemon/state/types.h"

namespace tensorcast::daemon {

// Build canonical index JSON from deduplicated storage metadata and tensor aliases.
// Returns Ok("") when aliases/storages are empty.
absl::StatusOr<std::string> build_canonical_index_from_metadata(
    absl::Span<const LeaseSegMeta> segments,
    absl::Span<const RegisterStorageMeta> storages,
    absl::Span<const RegisterTensorAliasMeta> aliases,
    int device_id);

} // namespace tensorcast::daemon
