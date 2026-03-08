// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "daemon/service/body_backing_types.h"
#include "daemon/service/byte_artifact_body_handle.h"
#include "daemon/service/byte_artifact_runtime_state.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class ByteArtifactBodyStore {
 public:
  struct EntrySnapshot {
    BodyDescriptor descriptor;
    BodyHandle body_handle;
    absl::Time expires_at{absl::InfinitePast()};
  };

  enum class PutOutcome {
    kCreated,
    kJoined,
    kConflict,
  };

  struct PutResult {
    PutOutcome outcome{PutOutcome::kConflict};
  };

  explicit ByteArtifactBodyStore(ByteArtifactRuntimeState& state);

  [[nodiscard]] bool exists(
      std::string_view artifact_id,
      std::uint64_t shard_id,
      std::uint64_t lease_generation,
      std::uint64_t routing_epoch,
      absl::Time now);

  [[nodiscard]] std::optional<EntrySnapshot> get(
      std::string_view artifact_id,
      std::uint64_t shard_id,
      std::uint64_t lease_generation,
      std::uint64_t routing_epoch,
      absl::Time now);

  [[nodiscard]] PutResult put_if_absent(
      std::string_view artifact_id,
      const v2::PutIfAbsentInvariant& invariant,
      const BodyDescriptor& descriptor,
      const BodyBackingObservation& observation,
      const BodyHandle& body_handle,
      std::uint64_t shard_id,
      std::uint64_t lease_generation,
      std::uint64_t routing_epoch,
      absl::Time now,
      const std::optional<std::uint64_t>& ttl_ms);

  [[nodiscard]] bool touch_ttl(
      std::string_view artifact_id,
      std::uint64_t shard_id,
      std::uint64_t lease_generation,
      std::uint64_t routing_epoch,
      absl::Time now,
      std::uint64_t ttl_ms);

 private:
  ByteArtifactRuntimeState& state_;
};

} // namespace tensorcast::daemon
