// Copyright (c) 2025, StepCast Team. All rights reserved.

//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
#pragma once

// Replace direct CUDA dependency with abstraction header that works for both
// real CUDA and FakeCuda builds.
#include <string>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/common/cuda_api.h"

namespace stepcast::common {

// ---------------------------------------------------------------------------
// CUDA error helpers that work for both real CUDA runtime and FakeCuda.
// ---------------------------------------------------------------------------

// Return a human-readable CUDA error name.
static inline const char* cuda_error_name(cudaError_t err) {
#ifdef USE_FAKE_CUDA
  switch (err) {
    case cudaSuccess:
      return "cudaSuccess";
    case cudaErrorInvalidValue:
      return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation:
      return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError:
      return "cudaErrorInitializationError";
    case cudaErrorLaunchFailure:
      return "cudaErrorLaunchFailure";
    case cudaErrorInvalidConfiguration:
      return "cudaErrorInvalidConfiguration";
    default:
      return "cudaErrorUnknown";
  }
#else // Real CUDA runtime
  return cudaGetErrorName(err);
#endif
}

// Return a descriptive CUDA error string.
static inline const char* cuda_error_string(cudaError_t err) {
#ifdef USE_FAKE_CUDA
  // Fake backend has no additional description beyond the name.
  return cuda_error_name(err);
#else
  return cudaGetErrorString(err);
#endif
}

// ---------------------------------------------------------------------------
// Preferred modern API: Convert CUDA errors to absl::Status
// ---------------------------------------------------------------------------
[[nodiscard]] inline absl::Status cuda_as_status(cudaError_t result, const std::string& context) {
  if (result == cudaSuccess) {
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrCat(context, " - ", cuda_error_name(result), ": ", cuda_error_string(result)));
}

} // namespace stepcast::common

// Macro to evaluate a CUDA expression and RETURN the resulting absl::Status
// from the current function if the expression fails.
#define SC_RETURN_IF_CUDA_ERROR(expr)                                          \
  do {                                                                         \
    cudaError_t _cuda_err = (expr);                                            \
    absl::Status _status = stepcast::common::cuda_as_status(_cuda_err, #expr); \
    if (!_status.ok()) {                                                       \
      LOG(ERROR) << _status;                                                   \
      return _status;                                                          \
    }                                                                          \
  } while (0)
