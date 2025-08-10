// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <future>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/model/model_location.h"

// Forward declare SeekableSource in the loader namespace to avoid heavy includes here
namespace stepcast::store::loader {
class SeekableSource;
}

namespace stepcast::store {

/**
 * @brief Interface for loading model data from a source (Disk, P2P)
 *        into memory managed by a MemoryManager.
 */
class IModelLoader {
 public:
  virtual ~IModelLoader() = default;

  /**
   * @brief Initialize the loader.
   * @return absl::Status OkStatus if initialization succeeds, error otherwise.
   */
  virtual absl::Status initialize() = 0;

  /**
   * @brief Get the total size of the model data in bytes.
   * May require initialization or connection depending on the loader type.
   * @return absl::StatusOr<uint64_t> Model size or error.
   */
  virtual absl::StatusOr<uint64_t> get_model_size() = 0;

  /**
   * @brief NEW: Provide a data source handle for this loader.
   * Implementations may return a muxed source (e.g., primary + fallback).
   */
  virtual absl::StatusOr<std::unique_ptr<loader::SeekableSource>> open_source() = 0;

  // Disable copy and move semantics
  IModelLoader(const IModelLoader&) = delete;
  IModelLoader& operator=(const IModelLoader&) = delete;
  IModelLoader(IModelLoader&&) = delete;
  IModelLoader& operator=(IModelLoader&&) = delete;

 protected:
  IModelLoader() = default; // Constructor accessible only to derived classes
};

} // namespace stepcast::store