// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/store/materialization/contracts/stable_local_backing.h"
#include "core/store/materialization/dataplane/contracts/sink.h"
#include "gsl/pointers"

namespace tensorcast::store::loader {

struct HostTargetStorage {
  gsl::not_null<void*> base_ptr;
  uint64_t length{0};
  std::optional<StableLocalBackingRef> stable_backing;
  std::shared_ptr<void> keepalive;
};

class TargetLayoutHostSink : public Sink, public PositionedSink, public DirectWriteCapable {
 public:
  struct Options {
    std::vector<HostTargetStorage> storages;
    std::shared_ptr<void> keepalive;
  };

  explicit TargetLayoutHostSink(Options options);
  ~TargetLayoutHostSink() override = default;

  absl::Status write(const void* src, size_t bytes) override;
  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override;
  absl::StatusOr<DirectWriteGrant> plan_direct_write(absl::Span<const VaRange> ranges) override;

  absl::Status close() override {
    return overall_status_;
  }

  [[nodiscard]] uint64_t total_size() const {
    return total_size_;
  }

 private:
  struct StorageState {
    uint64_t base_offset{0};
    uint64_t length{0};
    gsl::not_null<void*> base_ptr;
    std::optional<StableLocalBackingRef> stable_backing;
  };

  size_t find_storage_index(uint64_t offset) const;

  std::vector<StorageState> storage_states_;
  std::shared_ptr<void> keepalive_;
  uint64_t total_size_{0};
  uint64_t current_offset_{0};
  absl::Status overall_status_;
};

} // namespace tensorcast::store::loader
