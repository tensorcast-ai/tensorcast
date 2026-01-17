// Copyright (c) 2025-2026, TensorCast Team.

// Minimal engine->gRPC status mapping helper per RFC-0010 Appendix A.
#pragma once

#include "absl/status/status.h"
#include "grpcpp/grpcpp.h"

namespace tensorcast::daemon::status_utils {

inline grpc::Status to_grpc_status(const absl::Status& s) {
  using grpc::StatusCode;
  switch (s.code()) {
    case absl::StatusCode::kInvalidArgument:
      return {StatusCode::INVALID_ARGUMENT, std::string(s.message())};
    case absl::StatusCode::kNotFound:
      return {StatusCode::NOT_FOUND, std::string(s.message())};
    case absl::StatusCode::kResourceExhausted:
      return {StatusCode::RESOURCE_EXHAUSTED, std::string(s.message())};
    case absl::StatusCode::kDeadlineExceeded:
      return {StatusCode::DEADLINE_EXCEEDED, std::string(s.message())};
    case absl::StatusCode::kFailedPrecondition:
      return {StatusCode::FAILED_PRECONDITION, std::string(s.message())};
    case absl::StatusCode::kAlreadyExists:
      return {StatusCode::ALREADY_EXISTS, std::string(s.message())};
    case absl::StatusCode::kUnimplemented:
      return {StatusCode::UNIMPLEMENTED, std::string(s.message())};
    case absl::StatusCode::kUnavailable:
      return {StatusCode::UNAVAILABLE, std::string(s.message())};
    default:
      return {StatusCode::INTERNAL, std::string(s.message())};
  }
}

} // namespace tensorcast::daemon::status_utils
