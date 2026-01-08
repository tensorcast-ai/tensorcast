// Copyright (c) 2025-2026, TensorCast Team.

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
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::cuda {

// ---------------------------------------------------------------------------
// CUDA error helpers that work for both real CUDA runtime and FakeCuda.
// ---------------------------------------------------------------------------

// Return a human-readable CUDA error name.
static inline const char* cuda_error_name(cudaError_t err) {
  return cudaGetErrorName(err);
}

// Return a descriptive CUDA error string.
static inline const char* cuda_error_string(cudaError_t err) {
  return cudaGetErrorString(err);
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

[[nodiscard]] inline absl::Status cuda_check(cudaError_t result, const std::string& context) {
  return cuda_as_status(result, context);
}

} // namespace tensorcast::cuda

// Evaluate a CUDA expression and return the resulting status.
#define TC_CUDA_STATUS(expr) ::tensorcast::cuda::cuda_check((expr), #expr)

// Evaluate a CUDA expression and return the error status from the current function if it fails.
#define TC_CUDA_RETURN_IF_ERROR(expr)            \
  do {                                           \
    absl::Status _status = TC_CUDA_STATUS(expr); \
    if (!_status.ok()) {                         \
      LOG(ERROR) << _status;                     \
      return _status;                            \
    }                                            \
  } while (0)

// Evaluate a CUDA expression and crash on failure (use for invariants only).
#define TC_CUDA_CHECK(expr)                      \
  do {                                           \
    absl::Status _status = TC_CUDA_STATUS(expr); \
    ABSL_CHECK_OK(_status);                      \
  } while (0)

// Evaluate a CUDA expression and log a warning if it fails; also clears the last error.
#define TC_CUDA_CHECK_WARN(expr)                  \
  do {                                            \
    absl::Status _status = TC_CUDA_STATUS(expr);  \
    if (!_status.ok()) {                          \
      LOG(WARNING) << _status;                    \
      (void)::tensorcast::cuda::get_last_error(); \
    }                                             \
  } while (0)

// Clear the last CUDA error.
#define TC_CUDA_CLEAR_ERROR()                   \
  do {                                          \
    (void)::tensorcast::cuda::get_last_error(); \
  } while (0)

// Kernel launch check helper (use in functions that return absl::Status).
#define TC_CUDA_KERNEL_LAUNCH_CHECK()                            \
  do {                                                           \
    absl::Status _status = ::tensorcast::cuda::get_last_error(); \
    if (!_status.ok()) {                                         \
      LOG(ERROR) << _status;                                     \
      return _status;                                            \
    }                                                            \
  } while (0)

// Macro to evaluate a CUDA expression and RETURN the resulting absl::Status
// from the current function if the expression fails.
#define SC_RETURN_IF_CUDA_ERROR(expr)                                          \
  do {                                                                         \
    cudaError_t _cuda_err = (expr);                                            \
    absl::Status _status = tensorcast::cuda::cuda_as_status(_cuda_err, #expr); \
    if (!_status.ok()) {                                                       \
      LOG(ERROR) << _status;                                                   \
      return _status;                                                          \
    }                                                                          \
  } while (0)
