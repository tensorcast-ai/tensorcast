// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/store/loader/sink.h"
#include "core/store/model/memory_manager.h"

namespace stepcast::store::loader {

// Writes directly into the DVMP-reserved CPU region at a given offset.
// Uses DVMP::write_at to ensure chunk metadata is updated.
class DVMPRegionSink : public Sink, public PositionedSink, public DirectWritableSink {
 public:
  struct Options {
    // Per‑model DVMP region handle (preferred for writes)
    memory::DistributedVirtualMemoryPool::DvmpRegion region;
    // Optional: memory manager for capability planning (direct writes)
    std::shared_ptr<MemoryManager> memory_manager;
    uint64_t total_size = 0;
  };

  explicit DVMPRegionSink(Options options);
  ~DVMPRegionSink() override = default;

  absl::Status write(const void* src, size_t bytes) override;
  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override;
  absl::Status close() override {
    return absl::OkStatus();
  }

  // Capability: plan direct writes for provided ranges via MemoryManager
  absl::StatusOr<DirectWriteToken> plan_direct_write(absl::Span<const VaRange> ranges) override {
    if (!options_.memory_manager) {
      return absl::FailedPreconditionError("DVMPRegionSink: memory_manager is null for plan_direct_write");
    }
    return options_.memory_manager->plan_direct_write(ranges);
  }
  [[nodiscard]] uint64_t total_size() const {
    return options_.total_size;
  }

 private:
  Options options_;
  uint64_t current_offset_ = 0;
};

} // namespace stepcast::store::loader
