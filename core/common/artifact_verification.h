// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tensorcast::store {

// Verification levels from least to most thorough
enum class VerificationLevel {
  KEY_POINTS = 0, // Check first, middle, last positions (near-zero overhead)
  SPARSE_SAMPLING, // Check 16 evenly distributed points
  SEGMENT_HASHES, // Check 8 segment hashes
  FULL_HASH // Check complete rolling hash
};

// Verification metadata for a replica
struct ArtifactVerificationInfo {
  uint64_t artifact_size = 0;
  uint64_t full_hash = 0; // xxHash64 of entire replica
  std::array<uint64_t, 8> segment_hashes = {}; // Hashes of 8 equal segments
  std::array<uint64_t, 16> sample_values = {}; // Values at 16 sample points
  std::array<uint64_t, 3> key_values = {}; // Values at start, middle, end

  // Serialize to/from JSON string for storage
  [[nodiscard]] std::string to_json() const;
  static absl::StatusOr<ArtifactVerificationInfo> from_json(const std::string& json_str);
} __attribute__((aligned(128)));

class ArtifactVerifier {
 public:
  static constexpr size_t CHUNK_SIZE = 1024 * 1024; // 1MB processing chunks
  // Fixed buffer size used for segment hashing in both generation and verification
  // This constant defines the protocol; do not diverge per device or call site
  static constexpr size_t SEGMENT_HASH_BUFFER_SIZE = 256 * 1024; // 256KB
  static constexpr size_t NUM_SEGMENTS = 8;
  static constexpr size_t NUM_SAMPLES = 16;

  // Generate verification info from replica data (CPU or GPU)
  // For GPU data, specify device_id >= 0
  static absl::StatusOr<ArtifactVerificationInfo> generate_verification_info(
      const std::vector<void*>& data_ptrs,
      const std::vector<size_t>& data_sizes,
      int device_id = -1, // -1 for CPU, >= 0 for GPU
      VerificationLevel max_level = VerificationLevel::FULL_HASH);

  // Verify replica data against existing verification info
  // Returns OK if verification passes at specified level
  static absl::Status verify_artifact_data(
      const std::vector<void*>& data_ptrs,
      const std::vector<size_t>& data_sizes,
      const ArtifactVerificationInfo& expected_info,
      VerificationLevel level = VerificationLevel::SEGMENT_HASHES,
      int device_id = -1);

  // Fast key-point verification (first, middle, last)
  static absl::Status verify_key_points(
      const std::vector<void*>& data_ptrs,
      const std::vector<size_t>& data_sizes,
      const ArtifactVerificationInfo& expected_info,
      int device_id = -1);

 private:
  // Helper to read data from CPU or GPU memory
  static absl::Status read_data_chunk(
      void* dest,
      const std::vector<void*>& data_ptrs,
      const std::vector<size_t>& data_sizes,
      size_t global_offset,
      size_t read_size,
      int device_id);

  // xxHash64 implementation
  static uint64_t xxhash64(const void* data, size_t len, uint64_t seed = 0);

  // Get value at specific offset (8 bytes)
  static absl::StatusOr<uint64_t> get_value_at_offset(
      const std::vector<void*>& data_ptrs,
      const std::vector<size_t>& data_sizes,
      size_t offset,
      int device_id);
};

} // namespace tensorcast::store