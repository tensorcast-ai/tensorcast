// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>

#include "absl/strings/ascii.h"
#include "absl/time/time.h"
#include "core/common/memory/memory_location.h"
#include "daemon/service/byte_artifact_body_handle.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

enum class BodyAccessClass : std::uint8_t {
  kHomeDefault = 0,
  kLocalGpuHot = 1,
  kTransientForward = 2,
  kSmallObject = 3,
};

enum class BodyPreferredResidency : std::uint8_t {
  kCpu = 0,
  kGpu = 1,
};

enum class BodyRetentionIntent : std::uint8_t {
  kEphemeral = 0,
  kRetained = 1,
};

enum class BodyStableRetentionRequirement : std::uint8_t {
  kNone = 0,
  kPreferStable = 1,
  kRequireStable = 2,
};

enum class BodySharingIntent : std::uint8_t {
  kPrivateLocal = 0,
  kLocalReadMostly = 1,
  kRemoteShareable = 2,
};

struct BodyBackingIntent {
  BodyPreferredResidency preferred_residency{BodyPreferredResidency::kCpu};
  BodyRetentionIntent retention_intent{BodyRetentionIntent::kRetained};
  BodyStableRetentionRequirement stable_retention_requirement{BodyStableRetentionRequirement::kNone};
  BodySharingIntent sharing_intent{BodySharingIntent::kPrivateLocal};
};

enum class BodyStableRetentionState : std::uint8_t {
  kUnknown = 0,
  kNotRequested = 1,
  kHeld = 2,
  kSkipped = 3,
};

enum class BodyCommunicatorExportState : std::uint8_t {
  kUnknown = 0,
  kNotExported = 1,
  kExported = 2,
};

enum class BodyCapabilityResolutionMode : std::uint8_t {
  kLocalBodyHandle = 0,
  kLoader = 1,
  kChunkRpcFallback = 2,
};

struct BodyDescriptor {
  std::string physical_artifact_id;
  std::string layout_id;
  std::uint64_t size_bytes{0};
  std::string payload_digest_alg;
  std::string payload_digest_hex;
  absl::Time created_at{absl::InfinitePast()};
  absl::Time verified_at{absl::InfinitePast()};
};

struct ResolvedBodyCapability {
  BodyCapabilityResolutionMode mode{BodyCapabilityResolutionMode::kLoader};
  bool local{true};
  BodyHandle body_handle;
  BodyDescriptor descriptor;
};

struct BodyBackingObservation {
  std::string physical_artifact_id;
  common::memory::MemoryLocation memory_location{common::memory::MemoryLocation::CPU};
  std::uint64_t size_bytes{0};
  bool cpu_memfd_available{false};
  bool cuda_ipc_available{false};
  BodyCommunicatorExportState communicator_export_state{BodyCommunicatorExportState::kUnknown};
  BodyStableRetentionState stable_retention_state{BodyStableRetentionState::kUnknown};
  absl::Time observed_at{absl::InfinitePast()};
};

inline std::string normalize_body_digest_value(std::string_view value) {
  std::string normalized(value);
  absl::AsciiStrToLower(&normalized);
  return normalized;
}

inline BodyDescriptor normalized_body_descriptor(BodyDescriptor descriptor) {
  descriptor.payload_digest_alg = normalize_body_digest_value(descriptor.payload_digest_alg);
  descriptor.payload_digest_hex = normalize_body_digest_value(descriptor.payload_digest_hex);
  return descriptor;
}

inline v2::PutIfAbsentInvariant body_descriptor_to_invariant(const BodyDescriptor& descriptor) {
  v2::PutIfAbsentInvariant invariant;
  invariant.set_layout_id(descriptor.layout_id);
  invariant.set_byte_length(descriptor.size_bytes);
  invariant.set_payload_digest_alg(descriptor.payload_digest_alg);
  invariant.set_payload_digest_hex(descriptor.payload_digest_hex);
  return invariant;
}

} // namespace tensorcast::daemon
