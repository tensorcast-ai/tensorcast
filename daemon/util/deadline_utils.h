// Copyright (c) 2025-2026, TensorCast Team.

// DeadlineUtils: helpers to clamp user-provided timeouts to gRPC deadlines

#pragma once

#include <algorithm>
#include <chrono>

#include "grpcpp/grpcpp.h"

namespace tensorcast::daemon {

inline std::chrono::milliseconds ClampToDeadline(
    const grpc::ServerContext& ctx,
    std::chrono::milliseconds user_timeout,
    std::chrono::milliseconds hard_cap) {
  using clock = std::chrono::system_clock;
  auto remaining = hard_cap;
  // Apply user timeout if positive
  if (user_timeout.count() > 0 && user_timeout < remaining)
    remaining = user_timeout;
  // Apply gRPC deadline if set
  const auto deadline = ctx.deadline();
  const auto now = clock::now();
  if (deadline != clock::time_point::max()) {
    if (deadline <= now) {
      remaining = std::chrono::milliseconds(0);
    } else {
      auto d = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
      if (d < remaining)
        remaining = d;
    }
  }
  return remaining;
}

} // namespace tensorcast::daemon
