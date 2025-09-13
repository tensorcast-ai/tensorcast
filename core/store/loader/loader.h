// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
// Prefer explicit includes over forward declarations
#include "core/store/loader/source.h"

namespace tensorcast::store {

/**
 * @brief Interface for loading replica data from a source (Disk, P2P)
 *        into memory managed by a ReplicaLoadController.
 */
class IArtifactLoader {
 public:
  virtual ~IArtifactLoader() = default;

  /**
   * @brief Initialize the loader.
   * @return absl::Status OkStatus if initialization succeeds, error otherwise.
   */
  virtual absl::Status initialize() = 0;

  /**
   * @brief Get the total size of the replica data in bytes.
   * May require initialization or connection depending on the loader type.
   * @return absl::StatusOr<uint64_t> Artifact size or error.
   */
  virtual absl::StatusOr<uint64_t> get_artifact_size() = 0;

  /**
   * @brief NEW: Provide a data source handle for this loader.
   * Implementations may return a muxed source (e.g., primary + fallback).
   */
  virtual absl::StatusOr<std::unique_ptr<loader::SeekableSource>> open_source() = 0;

  // Disable copy and move semantics
  IArtifactLoader(const IArtifactLoader&) = delete;
  IArtifactLoader& operator=(const IArtifactLoader&) = delete;
  IArtifactLoader(IArtifactLoader&&) = delete;
  IArtifactLoader& operator=(IArtifactLoader&&) = delete;

 protected:
  IArtifactLoader() = default; // Constructor accessible only to derived classes
};

} // namespace tensorcast::store
