// Copyright (c) 2025, TensorCast Team.

#pragma once

#include "absl/status/status.h"
#include "core/store/materialization/runtime/pipeline/ingestion_context.h"

namespace tensorcast::store::materialization::runtime::pipeline {

class DiskSourceAdapter {
 public:
  [[nodiscard]] static absl::Status prepare(const loading::DiskSource& source, IngestionContext& ctx);
};

class P2PSourceAdapter {
 public:
  [[nodiscard]] static absl::Status prepare(const P2PSource& source, IngestionContext& ctx);
};

} // namespace tensorcast::store::materialization::runtime::pipeline
