// Copyright (c) 2025, TensorCast Team.

#include "core/common/artifact_verification.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "core/common/cuda_api.h"
#include "core/common/error_handling.h"

// for convenience
using json = nlohmann::json;

namespace tensorcast::common {

// xxHash64 constants
static constexpr uint64_t kPrimE641 = 11400714785074694791ULL;
static constexpr uint64_t kPrimE642 = 14029467366897019727ULL;
static constexpr uint64_t kPrimE643 = 1609587929392839161ULL;
static constexpr uint64_t kPrimE644 = 9650029242287828579ULL;
static constexpr uint64_t kPrimE645 = 2870177450012600261ULL;

std::string bytes_to_hex(absl::Span<const uint8_t> bytes) {
  std::string out;
  out.reserve(bytes.size() * 3);
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (i != 0) {
      out.push_back(' ');
    }
    absl::StrAppendFormat(&out, "%02x", bytes[i]);
  }
  return out;
}

// JSON serialization using nlohmann/json
std::string ArtifactVerificationInfo::to_json() const {
  json j;
  j["version"] = "1.0";
  j["artifact_size"] = artifact_size;
  j["full_hash"] = full_hash;
  j["segment_hashes"] = segment_hashes;
  j["sample_values"] = sample_values;
  j["key_values"] = key_values;

  return j.dump(2); // Pretty print with 2-space indentation
}

// JSON deserialization using nlohmann/json
absl::StatusOr<ArtifactVerificationInfo> ArtifactVerificationInfo::from_json(const std::string& json_str) {
  try {
    json j = json::parse(json_str);

    ArtifactVerificationInfo info;

    // Parse artifact_size
    if (j.contains("artifact_size") && j["artifact_size"].is_number_unsigned()) {
      info.artifact_size = j["artifact_size"].get<uint64_t>();
    }

    // Parse full_hash
    if (j.contains("full_hash") && j["full_hash"].is_number_unsigned()) {
      info.full_hash = j["full_hash"].get<uint64_t>();
    }

    // Parse segment_hashes array
    if (j.contains("segment_hashes") && j["segment_hashes"].is_array()) {
      const auto& seg_hashes = j["segment_hashes"];
      for (size_t i = 0; i < std::min(seg_hashes.size(), info.segment_hashes.size()); ++i) {
        if (seg_hashes[i].is_number_unsigned()) {
          info.segment_hashes.at(i) = seg_hashes[i].get<uint64_t>();
        }
      }
    }

    // Parse sample_values array
    if (j.contains("sample_values") && j["sample_values"].is_array()) {
      const auto& sample_vals = j["sample_values"];
      for (size_t i = 0; i < std::min(sample_vals.size(), info.sample_values.size()); ++i) {
        if (sample_vals[i].is_number_unsigned()) {
          info.sample_values.at(i) = sample_vals[i].get<uint64_t>();
        }
      }
    }

    // Parse key_values array
    if (j.contains("key_values") && j["key_values"].is_array()) {
      const auto& key_vals = j["key_values"];
      for (size_t i = 0; i < std::min(key_vals.size(), info.key_values.size()); ++i) {
        if (key_vals[i].is_number_unsigned()) {
          info.key_values.at(i) = key_vals[i].get<uint64_t>();
        }
      }
    }

    return info;

  } catch (const json::parse_error& e) {
    return absl::InvalidArgumentError(absl::StrCat("JSON parse error: ", e.what()));
  } catch (const json::type_error& e) {
    return absl::InvalidArgumentError(absl::StrCat("JSON type error: ", e.what()));
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("JSON processing error: ", e.what()));
  }
}

// Rotate left
static inline uint64_t rotl64(uint64_t x, int r) {
  return (x << r) | (x >> (64 - r));
}

// xxHash64 implementation
uint64_t ArtifactVerifier::xxhash64(const void* data, size_t len, uint64_t seed) {
  const auto* p = static_cast<const uint8_t*>(data);
  const uint8_t* const end = p + len;
  uint64_t h64;

  if (len >= 32) {
    const uint8_t* const limit = end - 32;
    uint64_t v1 = seed + kPrimE641 + kPrimE642;
    uint64_t v2 = seed + kPrimE642;
    uint64_t v3 = seed + 0;
    uint64_t v4 = seed - kPrimE641;

    do {
      v1 = rotl64(v1 + (*reinterpret_cast<const uint64_t*>(p) * kPrimE642), 31) * kPrimE641;
      p += 8;
      v2 = rotl64(v2 + (*reinterpret_cast<const uint64_t*>(p) * kPrimE642), 31) * kPrimE641;
      p += 8;
      v3 = rotl64(v3 + (*reinterpret_cast<const uint64_t*>(p) * kPrimE642), 31) * kPrimE641;
      p += 8;
      v4 = rotl64(v4 + (*reinterpret_cast<const uint64_t*>(p) * kPrimE642), 31) * kPrimE641;
      p += 8;
    } while (p <= limit);

    h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
    h64 = (h64 ^ (rotl64(v1 * kPrimE642, 31) * kPrimE641)) * kPrimE641 + kPrimE644;
    h64 = (h64 ^ (rotl64(v2 * kPrimE642, 31) * kPrimE641)) * kPrimE641 + kPrimE644;
    h64 = (h64 ^ (rotl64(v3 * kPrimE642, 31) * kPrimE641)) * kPrimE641 + kPrimE644;
    h64 = (h64 ^ (rotl64(v4 * kPrimE642, 31) * kPrimE641)) * kPrimE641 + kPrimE644;
  } else {
    h64 = seed + kPrimE645;
  }

  h64 += len;

  while (p + 8 <= end) {
    h64 ^= rotl64(*reinterpret_cast<const uint64_t*>(p) * kPrimE642, 31) * kPrimE641;
    h64 = rotl64(h64, 27) * kPrimE641 + kPrimE644;
    p += 8;
  }

  if (p + 4 <= end) {
    h64 ^= *reinterpret_cast<const uint32_t*>(p) * kPrimE641;
    h64 = rotl64(h64, 23) * kPrimE642 + kPrimE643;
    p += 4;
  }

  while (p < end) {
    h64 ^= (*p++) * kPrimE645;
    h64 = rotl64(h64, 11) * kPrimE641;
  }

  h64 ^= h64 >> 33;
  h64 *= kPrimE642;
  h64 ^= h64 >> 29;
  h64 *= kPrimE643;
  h64 ^= h64 >> 32;

  return h64;
}

absl::Status ArtifactVerifier::read_data_chunk(
    void* dest,
    const std::vector<void*>& data_ptrs,
    const std::vector<size_t>& data_sizes,
    size_t global_offset,
    size_t read_size,
    int device_id) {
  // Basic validation
  if (dest == nullptr) {
    return absl::InvalidArgumentError("Destination pointer is null");
  }
  if (data_ptrs.empty() || data_sizes.empty()) {
    return absl::InvalidArgumentError("Empty data pointers or sizes");
  }
  if (data_ptrs.size() != data_sizes.size()) {
    return absl::InvalidArgumentError("data_ptrs and data_sizes length mismatch");
  }

  const uint64_t total_size = std::accumulate(data_sizes.begin(), data_sizes.end(), 0ULL);
  if (read_size == 0) {
    return absl::OkStatus();
  }
  if (global_offset >= total_size) {
    return absl::OutOfRangeError("Global offset exceeds total data size");
  }

  // Find which buffer contains the offset
  size_t current_offset = 0;
  size_t buffer_idx = 0;
  bool found = false;

  for (size_t i = 0; i < data_sizes.size(); ++i) {
    const size_t seg_size = data_sizes[i];
    if (global_offset < current_offset + seg_size) {
      buffer_idx = i;
      found = true;
      break;
    }
    current_offset += seg_size;
  }
  if (!found) {
    // Should not happen due to the global_offset check above, but be defensive
    return absl::OutOfRangeError("Failed to locate buffer for requested offset");
  }

  size_t bytes_read = 0;
  auto* dest_ptr = static_cast<uint8_t*>(dest);

  while (bytes_read < read_size && buffer_idx < data_ptrs.size()) {
    if (data_ptrs[buffer_idx] == nullptr) {
      return absl::FailedPreconditionError("Null source pointer in data_ptrs");
    }

    const size_t local_offset = global_offset - current_offset;
    if (local_offset >= data_sizes[buffer_idx]) {
      // Move to next buffer if local offset at end (defensive)
      current_offset += data_sizes[buffer_idx];
      ++buffer_idx;
      continue;
    }

    const size_t remaining_in_buffer = data_sizes[buffer_idx] - local_offset;
    size_t bytes_to_read = std::min(read_size - bytes_read, remaining_in_buffer);

    const uint8_t* src_ptr = static_cast<const uint8_t*>(data_ptrs[buffer_idx]) + local_offset;

    if (device_id >= 0) {
      // GPU memory copy - set device first for safety
      auto set_device_status = cuda::set_device(device_id);
      if (!set_device_status.ok()) {
        return set_device_status;
      }
      auto memcpy_status = cuda::memcpy(dest_ptr + bytes_read, src_ptr, bytes_to_read, cudaMemcpyDeviceToHost);
      if (!memcpy_status.ok()) {
        return memcpy_status;
      }
    } else {
      std::memcpy(dest_ptr + bytes_read, src_ptr, bytes_to_read);
    }

    bytes_read += bytes_to_read;
    global_offset += bytes_to_read;

    if (bytes_to_read == remaining_in_buffer) {
      // Move to next buffer
      current_offset += data_sizes[buffer_idx];
      ++buffer_idx;
    }
  }

  if (bytes_read < read_size) {
    return absl::OutOfRangeError("Not enough data to read requested size");
  }

  return absl::OkStatus();
}

absl::StatusOr<uint64_t> ArtifactVerifier::get_value_at_offset(
    const std::vector<void*>& data_ptrs,
    const std::vector<size_t>& data_sizes,
    size_t offset,
    int device_id) {
  uint64_t total_size = std::accumulate(data_sizes.begin(), data_sizes.end(), 0UL);
  if (offset >= total_size) {
    return 0; // Past end of data
  }

  // Only read the minimum needed (8 bytes for uint64_t or remaining data)
  uint64_t value = 0;
  size_t bytes_to_read = std::min(sizeof(uint64_t), total_size - offset);

  if (bytes_to_read == 0) {
    return 0;
  }

  absl::Status status = read_data_chunk(&value, data_ptrs, data_sizes, offset, bytes_to_read, device_id);
  if (!status.ok()) {
    return status;
  }

  return value;
}

absl::StatusOr<ArtifactVerificationInfo> ArtifactVerifier::generate_verification_info(
    const std::vector<void*>& data_ptrs,
    const std::vector<size_t>& data_sizes,
    int device_id,
    VerificationLevel max_level) {
  auto start_time = std::chrono::high_resolution_clock::now();

  ArtifactVerificationInfo info;
  info.artifact_size = std::accumulate(data_sizes.begin(), data_sizes.end(), 0UL);

  if (info.artifact_size == 0) {
    return absl::InvalidArgumentError("Artifact size is 0");
  }

  // Set CUDA device if needed
  if (device_id >= 0) {
    auto set_device_status = cuda::set_device(device_id);
    if (!set_device_status.ok()) {
      return set_device_status;
    }
  }

  // 1. Key values (first, middle, last) - minimal data copying
  auto first_val = get_value_at_offset(data_ptrs, data_sizes, 0, device_id);
  if (!first_val.ok()) {
    return first_val.status();
  }
  info.key_values[0] = *first_val;

  auto mid_val = get_value_at_offset(data_ptrs, data_sizes, info.artifact_size / 2, device_id);
  if (!mid_val.ok()) {
    return mid_val.status();
  }
  info.key_values[1] = *mid_val;

  auto last_val =
      get_value_at_offset(data_ptrs, data_sizes, info.artifact_size >= 8 ? info.artifact_size - 8 : 0, device_id);
  if (!last_val.ok()) {
    return last_val.status();
  }
  info.key_values[2] = *last_val;

  // 2. Sample values (16 evenly distributed points) - minimal data copying
  for (size_t i = 0; i < NUM_SAMPLES; ++i) {
    size_t offset = (info.artifact_size * i) / NUM_SAMPLES;
    auto val = get_value_at_offset(data_ptrs, data_sizes, offset, device_id);
    if (!val.ok()) {
      return val.status();
    }
    info.sample_values.at(i) = *val;
  }

  // 3. Segment hashes / full hash (only if required)
  if (max_level >= VerificationLevel::SEGMENT_HASHES) {
    // Use a fixed buffer size for segment hashing across generation and verification
    size_t buffer_size = ArtifactVerifier::SEGMENT_HASH_BUFFER_SIZE;
    std::vector<uint8_t> buffer(buffer_size);
    size_t segment_size = info.artifact_size / NUM_SEGMENTS;
    uint64_t rolling_hash = 0;

    for (size_t seg = 0; seg < NUM_SEGMENTS; ++seg) {
      size_t seg_start = seg * segment_size;
      size_t seg_end = (seg == NUM_SEGMENTS - 1) ? info.artifact_size : (seg + 1) * segment_size;

      uint64_t seg_hash = 0;
      size_t offset = seg_start;

      // Process segment in small chunks to minimize GPU->CPU copy overhead
      while (offset < seg_end) {
        size_t chunk_size = std::min(buffer_size, seg_end - offset);
        absl::Status status = read_data_chunk(buffer.data(), data_ptrs, data_sizes, offset, chunk_size, device_id);
        if (!status.ok()) {
          return status;
        }

        // Update segment hash
        seg_hash = xxhash64(buffer.data(), chunk_size, seg_hash);

        // Update rolling hash
        rolling_hash = xxhash64(buffer.data(), chunk_size, rolling_hash);

        offset += chunk_size;
      }

      info.segment_hashes.at(seg) = seg_hash;
    }

    info.full_hash = rolling_hash;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  LOG(INFO) << "Generated verification info for " << info.artifact_size << " bytes in " << duration.count() << " ms"
            << (device_id >= 0 ? " (GPU)" : " (CPU)");

  return info;
}

absl::Status ArtifactVerifier::verify_key_points(
    const std::vector<void*>& data_ptrs,
    const std::vector<size_t>& data_sizes,
    const ArtifactVerificationInfo& expected_info,
    int device_id) {
  uint64_t total_size = std::accumulate(data_sizes.begin(), data_sizes.end(), 0UL);
  if (total_size != expected_info.artifact_size) {
    return absl::FailedPreconditionError(
        absl::StrCat("Size mismatch: got ", total_size, ", expected ", expected_info.artifact_size));
  }

  // Set CUDA device if needed
  if (device_id >= 0) {
    auto set_device_status = cuda::set_device(device_id);
    if (!set_device_status.ok()) {
      return set_device_status;
    }
  }

  // Check first value - only copy 8 bytes
  auto first_val_or = get_value_at_offset(data_ptrs, data_sizes, 0, device_id);
  if (!first_val_or.ok()) {
    return first_val_or.status();
  }

  auto mid_val_or = get_value_at_offset(data_ptrs, data_sizes, total_size / 2, device_id);
  if (!mid_val_or.ok()) {
    return mid_val_or.status();
  }

  const size_t last_offset = total_size >= sizeof(uint64_t) ? static_cast<size_t>(total_size - sizeof(uint64_t)) : 0;
  auto last_val_or = get_value_at_offset(data_ptrs, data_sizes, last_offset, device_id);
  if (!last_val_or.ok()) {
    return last_val_or.status();
  }

  const uint64_t actual_first = *first_val_or;
  const uint64_t actual_mid = *mid_val_or;
  const uint64_t actual_last = *last_val_or;

  const uint64_t expected_first = expected_info.key_values[0];
  const uint64_t expected_mid = expected_info.key_values[1];
  const uint64_t expected_last = expected_info.key_values[2];

  auto log_mismatch = [&](std::string_view position, uint64_t actual, uint64_t expected) {
    LOG(ERROR) << "verify_key_points mismatch (" << position << ") total_size=" << total_size
               << " device_id=" << device_id << " actual=0x" << absl::Hex(actual, absl::kZeroPad16) << " expected=0x"
               << absl::Hex(expected, absl::kZeroPad16) << " actual[keys]=[0x"
               << absl::Hex(actual_first, absl::kZeroPad16) << ", 0x" << absl::Hex(actual_mid, absl::kZeroPad16)
               << ", 0x" << absl::Hex(actual_last, absl::kZeroPad16) << "] expected[keys]=[0x"
               << absl::Hex(expected_first, absl::kZeroPad16) << ", 0x" << absl::Hex(expected_mid, absl::kZeroPad16)
               << ", 0x" << absl::Hex(expected_last, absl::kZeroPad16) << "]";

    const size_t tail_len = static_cast<size_t>(std::min<uint64_t>(16, total_size));
    const auto tail_offset = static_cast<size_t>(total_size >= tail_len ? total_size - tail_len : 0);
    std::array<uint8_t, 16> tail_bytes{};
    absl::Status tail_status =
        read_data_chunk(tail_bytes.data(), data_ptrs, data_sizes, tail_offset, tail_len, device_id);
    if (tail_status.ok()) {
      std::string tail_hex;
      tail_hex.reserve(tail_len * 3);
      for (size_t i = 0; i < tail_len; ++i) {
        if (i != 0) {
          tail_hex.append(" ");
        }
        tail_hex.append(absl::StrFormat("%02x", tail_bytes[i]));
      }
      LOG(ERROR) << "verify_key_points tail@" << tail_offset << " len=" << tail_len << " bytes [" << tail_hex << "]";
    } else {
      LOG(ERROR) << "verify_key_points unable to capture tail bytes: " << tail_status;
    }
  };

  if (actual_first != expected_first) {
    log_mismatch("start", actual_first, expected_first);
    return absl::DataLossError(
        absl::StrCat(
            "Key point mismatch at start of replica: got 0x",
            absl::Hex(actual_first),
            ", expected 0x",
            absl::Hex(expected_first)));
  }

  if (actual_mid != expected_mid) {
    log_mismatch("middle", actual_mid, expected_mid);
    return absl::DataLossError(
        absl::StrCat(
            "Key point mismatch at middle: got 0x", absl::Hex(actual_mid), ", expected 0x", absl::Hex(expected_mid)));
  }

  if (actual_last != expected_last) {
    log_mismatch("end", actual_last, expected_last);
    return absl::DataLossError(
        absl::StrCat(
            "Key point mismatch at end: got 0x", absl::Hex(actual_last), ", expected 0x", absl::Hex(expected_last)));
  }

  VLOG(1) << "Key-point verification passed for " << total_size << " bytes" << (device_id >= 0 ? " (GPU)" : " (CPU)")
          << " - only copied 24 bytes for verification";

  return absl::OkStatus();
}

absl::Status ArtifactVerifier::verify_artifact_data(
    const std::vector<void*>& data_ptrs,
    const std::vector<size_t>& data_sizes,
    const ArtifactVerificationInfo& expected_info,
    VerificationLevel level,
    int device_id) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Always verify key points first (fastest check)
  absl::Status status = verify_key_points(data_ptrs, data_sizes, expected_info, device_id);
  if (!status.ok()) {
    return status;
  }

  if (level == VerificationLevel::KEY_POINTS) {
    return absl::OkStatus();
  }

  // Verify sparse samples
  if (level >= VerificationLevel::SPARSE_SAMPLING) {
    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
      size_t offset = (expected_info.artifact_size * i) / NUM_SAMPLES;
      auto val = get_value_at_offset(data_ptrs, data_sizes, offset, device_id);
      if (!val.ok()) {
        return val.status();
      }
      if (*val != expected_info.sample_values.at(i)) {
        return absl::DataLossError(absl::StrCat("Sample value mismatch at offset ", offset));
      }
    }
  }

  // Verify segment hashes
  if (level >= VerificationLevel::SEGMENT_HASHES) {
    // Use the same fixed buffer size as generation
    std::vector<uint8_t> buffer(ArtifactVerifier::SEGMENT_HASH_BUFFER_SIZE);
    size_t segment_size = expected_info.artifact_size / NUM_SEGMENTS;

    for (size_t seg = 0; seg < NUM_SEGMENTS; ++seg) {
      size_t seg_start = seg * segment_size;
      size_t seg_end = (seg == NUM_SEGMENTS - 1) ? expected_info.artifact_size : (seg + 1) * segment_size;

      uint64_t seg_hash = 0;
      size_t offset = seg_start;

      while (offset < seg_end) {
        size_t chunk_size = std::min(buffer.size(), seg_end - offset);
        status = read_data_chunk(buffer.data(), data_ptrs, data_sizes, offset, chunk_size, device_id);
        if (!status.ok()) {
          return status;
        }

        seg_hash = xxhash64(buffer.data(), chunk_size, seg_hash);
        offset += chunk_size;
      }

      if (seg_hash != expected_info.segment_hashes.at(seg)) {
        const size_t segment_len = seg_end - seg_start;
        const size_t sample_len = std::min<size_t>(64, segment_len);
        std::vector<uint8_t> head_sample(sample_len);
        absl::Status head_status =
            read_data_chunk(head_sample.data(), data_ptrs, data_sizes, seg_start, sample_len, device_id);
        std::vector<uint8_t> tail_sample;
        absl::Status tail_status = absl::OkStatus();
        if (segment_len > sample_len) {
          const size_t tail_len = std::min<size_t>(64, segment_len);
          tail_sample.resize(tail_len);
          tail_status =
              read_data_chunk(tail_sample.data(), data_ptrs, data_sizes, seg_end - tail_len, tail_len, device_id);
        }
        LOG(ERROR) << "ArtifactVerifier: Segment hash mismatch (segment=" << seg << ", device=" << device_id
                   << ", offset=" << seg_start << ", length=" << segment_len << ")"
                   << " actual_hash=" << seg_hash << " expected_hash=" << expected_info.segment_hashes.at(seg);
        if (head_status.ok()) {
          LOG(ERROR) << "ArtifactVerifier: segment head bytes [" << bytes_to_hex(absl::MakeConstSpan(head_sample))
                     << "]";
        } else {
          LOG(ERROR) << "ArtifactVerifier: failed to sample segment head (segment=" << seg
                     << ") status=" << head_status;
        }
        if (!tail_sample.empty()) {
          if (tail_status.ok()) {
            LOG(ERROR) << "ArtifactVerifier: segment tail bytes [" << bytes_to_hex(absl::MakeConstSpan(tail_sample))
                       << "]";
          } else {
            LOG(ERROR) << "ArtifactVerifier: failed to sample segment tail (segment=" << seg
                       << ") status=" << tail_status;
          }
        }
        return absl::DataLossError(absl::StrCat("Segment ", seg, " hash mismatch"));
      }
    }
  }

  // Verify full hash
  if (level >= VerificationLevel::FULL_HASH) {
    auto actual_info = generate_verification_info(data_ptrs, data_sizes, device_id, VerificationLevel::FULL_HASH);
    if (!actual_info.ok()) {
      return actual_info.status();
    }

    if (actual_info->full_hash != expected_info.full_hash) {
      return absl::DataLossError("Full artifact hash mismatch");
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  VLOG(1) << "Artifact verification passed at level " << static_cast<int>(level) << " in " << duration.count() << " ms";

  return absl::OkStatus();
}

} // namespace tensorcast::common
