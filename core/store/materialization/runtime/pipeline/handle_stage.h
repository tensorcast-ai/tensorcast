// Copyright (c) 2025, TensorCast Team.

#pragma once

#include "absl/status/statusor.h"
#include "core/store/materialization/runtime/pipeline/ingestion_context.h"

namespace tensorcast::store::materialization::runtime::pipeline {

class HandleStage {
 public:
  [[nodiscard]] static absl::StatusOr<loading::ReplicaHandle> build(IngestionContext& ctx);
};

} // namespace tensorcast::store::materialization::runtime::pipeline
