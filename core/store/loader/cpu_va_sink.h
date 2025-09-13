// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/common/memory/virtual_address_space.h"
#include "core/store/loader/sink.h"

namespace tensorcast::store::loader {

// Writes directly into the VirtualAddressSpace-reserved CPU region at a given offset.
// Uses VA write_at to ensure chunk metadata is updated.
class CpuVaSink : public Sink, public PositionedSink, public DirectWriteCapable {
 public:
  struct Options {
    // Per‑replica VA region handle (preferred for writes)
    common::memory::VirtualAddressSpace::VaRegion region;
    // Replace ReplicaLoadController dependency with an injected callback to avoid cycles
    std::function<absl::StatusOr<DirectWriteGrant>(absl::Span<const VaRange>)> plan_direct_write_fn;
    uint64_t total_size = 0;
  };

  explicit CpuVaSink(Options options);
  ~CpuVaSink() override = default;

  absl::Status write(const void* src, size_t bytes) override;
  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override;

  absl::Status close() override {
    return absl::OkStatus();
  }

  // Capability: plan direct writes for provided ranges via injected callback
  absl::StatusOr<DirectWriteGrant> plan_direct_write(absl::Span<const VaRange> ranges) override {
    if (!options_.plan_direct_write_fn) {
      return absl::FailedPreconditionError("CpuVaSink: plan_direct_write_fn is null");
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

} // namespace tensorcast::store::loader
