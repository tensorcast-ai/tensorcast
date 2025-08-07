// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "core/store/loader/sink.h"
#include "core/store/model/memory_manager.h"

namespace stepcast::store::loader {

class UnifiedMemorySink : public Sink {
 public:
  struct Options {
    std::shared_ptr<Sink> inner_sink;
    std::shared_ptr<store::MemoryManager> memory_manager;
    store::ModelLocation target_location;
    std::optional<int> device_id; // For GPU targets
    std::optional<std::vector<uint32_t>> chunk_indices; // For partial loads
  };

  explicit UnifiedMemorySink(Options options);
  ~UnifiedMemorySink() override = default;

  absl::Status write(const void* src, size_t bytes) override;

  absl::Status close() override;

 private:
  absl::Status update_chunk_states();

  Options options_;
  size_t bytes_written_ = 0;
  bool states_updated_ = false;
};

} // namespace stepcast::store::loader