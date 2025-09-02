// Copyright (c) 2025, TensorCast Team.

// All rights reserved.
#pragma once

#include <cstdint>
#include <string>

namespace tensorcast::common::trace {

class TraceScope {
 public:
  TraceScope(const std::string& replica, const std::string& stage);

  // Move-only to allow capture into lambdas / async tasks.
  TraceScope(TraceScope&& other) noexcept;
  TraceScope& operator=(TraceScope&& other) noexcept;

  TraceScope(const TraceScope&) = delete;
  TraceScope& operator=(const TraceScope&) = delete;

  ~TraceScope();

 private:
  void Finish();

  static constexpr uint64_t kInvalidSpan = static_cast<uint64_t>(-1);
  std::string artifact_id_;
  std::string request_id_;
  uint64_t id_ = kInvalidSpan;
};

} // namespace tensorcast::common::trace