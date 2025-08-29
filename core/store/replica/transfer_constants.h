// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>

namespace tensorcast::store {

// Transfer buffer size constants
constexpr size_t kDefaultMaxBufferBytes = 256ULL << 20; // 256 MB

// Device ID constants
constexpr int kCpuDeviceId = -1;

} // namespace tensorcast::store