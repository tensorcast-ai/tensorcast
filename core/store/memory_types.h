// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include "core/store/concurrent_vector.h"

namespace stepcast::store {

// ══════════════════════════════════════════════════════════════════════════
// Memory Size Constants
// ══════════════════════════════════════════════════════════════════════════

constexpr size_t KB = 1024LL;
constexpr size_t MB = 1024LL * KB;
constexpr size_t GB = 1024LL * MB;

// ══════════════════════════════════════════════════════════════════════════
// Batch Structures for Memory Transfer
// ══════════════════════════════════════════════════════════════════════════

/**
 * @brief Basic batch information for memory chunks
 */
struct Batch {
  size_t chunk_id = 0;
  size_t size = 0;
} __attribute__((aligned(16)));

using BatchVector = ConcurrentVector<Batch>;

} // namespace stepcast::store