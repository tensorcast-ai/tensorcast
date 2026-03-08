// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/store_engine.h"

namespace tensorcast::daemon {

class BodyHandle {
 public:
  enum class Kind {
    kCoreReplica = 0,
  };

  BodyHandle() = default;

  [[nodiscard]] static absl::StatusOr<BodyHandle> create(
      store::StoreEngine& engine,
      store::loading::ReplicaHandle replica_handle);

  [[nodiscard]] bool empty() const;
  [[nodiscard]] Kind kind() const;
  [[nodiscard]] std::uint64_t size_bytes() const;
  [[nodiscard]] const store::loading::ReplicaHandle& replica_handle() const;
  [[nodiscard]] common::memory::MemoryLocation location() const;

  [[nodiscard]] absl::StatusOr<std::unique_ptr<store::IArtifactLoader>> make_loader() const;
  [[nodiscard]] absl::StatusOr<std::string> read_range(std::uint64_t offset, std::size_t bytes) const;
  [[nodiscard]] absl::StatusOr<std::string> read_all_bytes() const;
  [[nodiscard]] absl::StatusOr<std::string> compute_sha256_hex() const;
  [[nodiscard]] absl::Status retire() const;

 private:
  struct CoreBacking {
    store::StoreEngine* engine{nullptr};
    store::loading::ReplicaHandle replica_handle;
    std::uint64_t size_bytes{0};
    std::atomic<bool> retired{false};
  };

  explicit BodyHandle(std::shared_ptr<CoreBacking> backing);

  std::shared_ptr<CoreBacking> backing_;
};

} // namespace tensorcast::daemon
