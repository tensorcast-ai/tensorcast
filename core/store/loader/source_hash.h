// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "core/store/loader/source.h"

namespace stepcast::store::loader {

/**
 * Compute data multihash (multibase base32 over Merkle tree sha2-256 root)
 * by streaming bytes from a generic SeekableSource.
 *
 * The function reads sequentially from offset 0..total_size in fixed-size
 * leaves (default 4 MiB), computes sha256 for each leaf, then reduces the
 * leaf digests pairwise until a single root remains. Finally it returns a
 * multibase(base32, lowercase) encoded multihash with code 0x12 and length 0x20.
 */
absl::StatusOr<std::string> compute_data_multihash_from_seekable_source(
    SeekableSource& source,
    uint64_t total_size,
    size_t leaf_chunk_bytes = 4ULL * 1024 * 1024);

} // namespace stepcast::store::loader
