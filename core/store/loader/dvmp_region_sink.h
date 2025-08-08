// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "core/store/loader/sink.h"
#include "core/store/model/memory_manager.h"

namespace stepcast::store::loader {

// Writes directly into the DVMP-reserved CPU region at a given offset.
// Uses DVMP::write_at to ensure chunk metadata is updated.
class DVMPRegionSink : public Sink, public PositionedSink {
 public:
  struct Options {
    std::shared_ptr<store::MemoryManager> memory_manager; // to access DVMP and model id
    uint64_t total_size = 0;
  };

  explicit DVMPRegionSink(Options options);
  ~DVMPRegionSink() override = default;

  absl::Status write(const void* src, size_t bytes) override;
  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override;
  absl::Status close() override {
    return absl::OkStatus();
  }

 private:
  Options options_;
  uint64_t current_offset_ = 0;
};

} // namespace stepcast::store::loader
