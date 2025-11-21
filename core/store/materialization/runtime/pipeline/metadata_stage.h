// Copyright (c) 2025, TensorCast Team.

#pragma once

#include "absl/status/status.h"
#include "core/store/materialization/runtime/pipeline/ingestion_context.h"

namespace tensorcast::store::materialization::runtime::pipeline {

class MetadataStage {
 public:
  [[nodiscard]] static absl::Status process(IngestionContext& ctx);
};

} // namespace tensorcast::store::materialization::runtime::pipeline
