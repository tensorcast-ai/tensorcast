// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/store/loader/sink.h"

namespace stepcast::store::loader {

// Writes directly into the DVMP-reserved CPU region at a given offset.
// Uses DVMP::write_at to ensure chunk metadata is updated.
class DVMPRegionSink : public Sink, public PositionedSink, public DirectWritableSink {
 public:
  struct Options {
    // Per‑replica DVMP region handle (preferred for writes)
    memory::DistributedVirtualMemoryPool::DvmpRegion region;
    // Replace MemoryManager dependency with an injected callback to avoid cycles
    std::function<absl::StatusOr<DirectWriteToken>(absl::Span<const VaRange>)> plan_direct_write_fn;
    uint64_t total_size = 0;
  };

  explicit DVMPRegionSink(Options options);
  ~DVMPRegionSink() override = default;

  absl::Status write(const void* src, size_t bytes) override;
  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override;
  absl::Status close() override {
    return absl::OkStatus();
  }

  // Capability: plan direct writes for provided ranges via injected callback
  absl::StatusOr<DirectWriteToken> plan_direct_write(absl::Span<const VaRange> ranges) override {
    if (!options_.plan_direct_write_fn) {
      return absl::FailedPreconditionError("DVMPRegionSink: plan_direct_write_fn is null");
    }
    return options_.plan_direct_write_fn(ranges);
  }
  [[nodiscard]] uint64_t total_size() const {
    return options_.total_size;
  }

 private:
  Options options_;
  uint64_t current_offset_ = 0;
};

} // namespace stepcast::store::loader
