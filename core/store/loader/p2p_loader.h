// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <future>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

#include "core/common/memory/memory_location.h"
#include "core/store/communication_types.h"
#include "core/store/loader/loader.h"

namespace tensorcast::store::loader {
class SeekableSource;
}

namespace tensorcast::store {

/**
 * @brief Loader implementation for fetching replica data from a remote peer via RDMA or TCP.
 */
class P2PLoader : public IArtifactLoader {
 public:
  /**
   * @brief Constructs a P2PLoader.
   * @param source Configuration detailing the remote peer source.
   * @param comm_engine A shared communicator engine instance for performing P2P operations.
   */
  explicit P2PLoader(P2PSource source);

  ~P2PLoader() override = default;

  // Disable copy and move
  P2PLoader(const P2PLoader&) = delete;
  P2PLoader& operator=(const P2PLoader&) = delete;
  P2PLoader(P2PLoader&&) = delete;
  P2PLoader& operator=(P2PLoader&&) = delete;

  /**
   * @brief Initializes the loader (currently checks communicator).
   * @return absl::Status OkStatus if communicator is valid.
   */
  absl::Status initialize() override ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Gets the expected artifact size from the configuration.
   * Requires the loader to be initialized first.
   * @return absl::StatusOr<uint64_t> Artifact size or error if not initialized or not configured.
   */
  absl::StatusOr<uint64_t> get_artifact_size() override ABSL_LOCKS_EXCLUDED(mutex_);

  // NEW: Provide the data source (possibly muxed with disk fallback)
  absl::StatusOr<std::unique_ptr<loader::SeekableSource>> open_source() override ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  mutable absl::Mutex mutex_;

  P2PSource source_;
  bool initialized_ = false;
};

} // namespace tensorcast::store
