// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include "absl/status/status.h"
#include "core/store/materialization/runtime/pipeline/ingestion_context.h"

namespace tensorcast::store::materialization::runtime::pipeline {

struct FullDigestDecision {
  bool should_compute{false};
  bool forced_by_hint{false};
  bool forced_by_engine_option{false};
  bool forced_by_safetensors{false};
  bool trusted_existing_data_multihash{false};
};

[[nodiscard]] FullDigestDecision resolve_full_digest_decision(const IngestionContext& ctx);
[[nodiscard]] bool should_skip_disk_verification(const IngestionContext& ctx);

class VerificationStage {
 public:
  [[nodiscard]] static absl::Status verify(IngestionContext& ctx);
};

} // namespace tensorcast::store::materialization::runtime::pipeline
