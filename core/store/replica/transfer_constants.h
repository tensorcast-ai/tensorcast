// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstddef>

namespace stepcast::store {

// Transfer buffer size constants
constexpr size_t kDefaultMaxBufferBytes = 256ULL << 20; // 256 MB

// Device ID constants
constexpr int kCpuDeviceId = -1;

} // namespace stepcast::store