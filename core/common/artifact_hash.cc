// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/common/artifact_hash.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include <openssl/evp.h>
#include <openssl/sha.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/common/cuda_api.h"
// NOTE: artifact_hash is intended to be loader-agnostic; the disk-dir helper is
// now implemented in core/store/loader. Keep GPU hashing and index hashing here.
// Any disk directory hashing should be performed via loader::disk_dir_hash.

namespace stepcast::store::artifact_hash {

namespace {

// Maximum allowed chunk size for hashing (64MB)
constexpr size_t kMaxChunkSize = 64ULL * 1024 * 1024;
// Default chunk size for tree hashing (4MB)
constexpr size_t kDefaultChunkSize = 4ULL * 1024 * 1024;

// Base32 RFC 4648 alphabet (lowercase without padding)
constexpr char kBase32Alphabet[] = "abcdefghijklmnopqrstuvwxyz234567";

// Base32 (lowercase) without padding, RFC 4648 alphabet
std::string base32_lower_encode(const std::vector<uint8_t>& data) {
  if (data.empty()) {
    return std::string();
  }

  const size_t in_len = data.size();
  std::string result;
  result.reserve(((in_len + 4) / 5) * 8);

  size_t i = 0;
  while (i < in_len) {
    // Process 5 bytes at a time to produce 8 base32 characters
    uint64_t buffer = 0;
    size_t bytes_to_process = std::min<size_t>(5, in_len - i);

    // Load bytes into buffer (big-endian)
    for (size_t j = 0; j < bytes_to_process; ++j) {
      buffer = (buffer << 8) | data[i + j];
    }

    // Pad remaining bits if needed
    buffer <<= (5 - bytes_to_process) * 8;

    // Extract 8 5-bit values
    for (int shift = 35; shift >= 0; shift -= 5) {
      result.push_back(kBase32Alphabet[(buffer >> shift) & 0x1F]);
    }

    i += bytes_to_process;
  }

  // Trim padding characters based on input length
  size_t output_length = ((in_len * 8 + 4) / 5); // Actual encoded length without padding
  result.resize(output_length);

  return result;
}

// RAII wrapper for EVP_MD_CTX
class EvpMdCtxWrapper {
 public:
  EvpMdCtxWrapper() : ctx_(EVP_MD_CTX_new()) {
    if (!ctx_) {
      LOG(ERROR) << "Failed to create EVP_MD_CTX";
    }
  }

  ~EvpMdCtxWrapper() {
    if (ctx_) {
      EVP_MD_CTX_free(ctx_);
    }
  }

  EVP_MD_CTX* get() {
    return ctx_;
  }
  bool valid() const {
    return ctx_ != nullptr;
  }

  // Disable copy
  EvpMdCtxWrapper(const EvpMdCtxWrapper&) = delete;
  EvpMdCtxWrapper& operator=(const EvpMdCtxWrapper&) = delete;

  // Enable move
  EvpMdCtxWrapper(EvpMdCtxWrapper&& other) noexcept : ctx_(other.ctx_) {
    other.ctx_ = nullptr;
  }

 private:
  EVP_MD_CTX* ctx_;
};

std::vector<uint8_t> hex_to_bytes(std::string_view hex) {
  std::vector<uint8_t> out;
  if (hex.size() % 2 != 0) {
    return out;
  }
  out.reserve(hex.size() / 2);
  auto hexval = [](char c) -> int {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
      return 10 + (c - 'a');
    }
    if (c >= 'A' && c <= 'F') {
      return 10 + (c - 'A');
    }
    return -1;
  };
  for (size_t i = 0; i < hex.size(); i += 2) {
    int hi = hexval(hex[i]);
    int lo = hexval(hex[i + 1]);
    if (hi < 0 || lo < 0) {
      return {};
    }
    out.push_back(static_cast<uint8_t>((hi << 4) | lo));
  }
  return out;
}

// Validate multihash format: multibase prefix + multihash function code + length
bool validate_multihash_format(const std::string& multihash) {
  if (multihash.size() < 3) {
    return false; // Too short for valid multihash
  }

  // Check multibase prefix (we use 'b' for base32)
  if (multihash[0] != 'b') {
    return false;
  }

  // The rest should be valid base32 characters
  for (size_t i = 1; i < multihash.size(); ++i) {
    char c = multihash[i];
    bool valid = (c >= 'a' && c <= 'z') || (c >= '2' && c <= '7');
    if (!valid) {
      return false;
    }
  }

  return true;
}

std::string to_multibase_multihash_sha256(const std::vector<uint8_t>& digest) {
  if (digest.size() != SHA256_DIGEST_LENGTH) {
    LOG(ERROR) << "Invalid digest size for SHA256: " << digest.size();
    return "";
  }

  std::vector<uint8_t> mh;
  mh.reserve(2 + digest.size());
  mh.push_back(0x12); // SHA256 function code
  mh.push_back(0x20); // SHA256 digest length (32 bytes)
  mh.insert(mh.end(), digest.begin(), digest.end());

  std::string b32 = base32_lower_encode(mh);
  std::string result = std::string("b") + b32;

  // Validate the generated multihash
  if (!validate_multihash_format(result)) {
    LOG(ERROR) << "Generated invalid multihash format";
    return "";
  }

  return result;
}

absl::StatusOr<std::vector<uint8_t>> sha256_bytes(absl::Span<const uint8_t> bytes) {
  if (bytes.empty()) {
    return absl::InvalidArgumentError("Cannot hash empty data");
  }

  EvpMdCtxWrapper ctx;
  if (!ctx.valid()) {
    return absl::InternalError("Failed to create digest context");
  }

  if (EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1) {
    return absl::InternalError("Failed to initialize SHA256");
  }

  if (EVP_DigestUpdate(ctx.get(), bytes.data(), bytes.size()) != 1) {
    return absl::InternalError("Failed to update SHA256");
  }

  std::vector<uint8_t> out(SHA256_DIGEST_LENGTH);
  unsigned int digest_len = 0;
  if (EVP_DigestFinal_ex(ctx.get(), out.data(), &digest_len) != 1) {
    return absl::InternalError("Failed to finalize SHA256");
  }

  if (digest_len != SHA256_DIGEST_LENGTH) {
    return absl::InternalError("Unexpected SHA256 digest length");
  }

  return out;
}

absl::StatusOr<std::vector<uint8_t>> compute_tree_hash_sha256(const std::vector<std::vector<uint8_t>>& leaf_digests) {
  if (leaf_digests.empty()) {
    return std::vector<uint8_t>(SHA256_DIGEST_LENGTH, 0);
  }

  // Validate all leaf digests are proper SHA256 digests
  for (const auto& leaf : leaf_digests) {
    if (leaf.size() != SHA256_DIGEST_LENGTH) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid leaf digest size: ", leaf.size(), " (expected ", SHA256_DIGEST_LENGTH, ")"));
    }
  }

  std::vector<std::vector<uint8_t>> level = leaf_digests;
  while (level.size() > 1) {
    std::vector<std::vector<uint8_t>> next;
    next.reserve((level.size() + 1) / 2);

    for (size_t i = 0; i < level.size(); i += 2) {
      if (i + 1 < level.size()) {
        // Combine two hashes
        std::vector<uint8_t> concat;
        concat.reserve(2 * SHA256_DIGEST_LENGTH);
        concat.insert(concat.end(), level[i].begin(), level[i].end());
        concat.insert(concat.end(), level[i + 1].begin(), level[i + 1].end());

        auto hash_result = sha256_bytes(concat);
        if (!hash_result.ok()) {
          return hash_result.status();
        }
        next.push_back(std::move(hash_result.value()));
      } else {
        // Odd node, promote directly
        next.push_back(level[i]);
      }
    }
    level.swap(next);
  }

  return level.front();
}

} // namespace

// Public wrappers for reuse across modules
std::vector<uint8_t> sha256_digest_bytes(absl::Span<const uint8_t> bytes) {
  auto result = sha256_bytes(bytes);
  if (!result.ok()) {
    LOG(ERROR) << "SHA256 failed: " << result.status();
    return std::vector<uint8_t>(SHA256_DIGEST_LENGTH, 0);
  }
  return result.value();
}

std::vector<uint8_t> compute_tree_hash_root_sha256(const std::vector<std::vector<uint8_t>>& leaf_digests) {
  auto result = compute_tree_hash_sha256(leaf_digests);
  if (!result.ok()) {
    LOG(ERROR) << "Tree hash failed: " << result.status();
    return std::vector<uint8_t>(SHA256_DIGEST_LENGTH, 0);
  }
  return result.value();
}

std::string multibase_multihash_sha256(const std::vector<uint8_t>& digest) {
  return to_multibase_multihash_sha256(digest);
}

absl::StatusOr<std::string> compute_index_multihash(
    const std::optional<std::string>& index_data,
    std::string_view index_key_hex) {
  std::vector<uint8_t> digest;

  if (index_data.has_value() && !index_data->empty()) {
    const auto bytes =
        absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(index_data->data()), index_data->size());
    auto hash_result = sha256_bytes(bytes);
    if (!hash_result.ok()) {
      return hash_result.status();
    }
    digest = std::move(hash_result.value());
  } else if (!index_key_hex.empty()) {
    auto key_bytes = hex_to_bytes(index_key_hex);
    if (key_bytes.size() != SHA256_DIGEST_LENGTH) {
      return absl::InvalidArgumentError(
          absl::StrCat("tensor_index_key must be a 32-byte sha256 hex string, got ", key_bytes.size(), " bytes"));
    }
    digest = std::move(key_bytes);
  } else {
    return absl::InvalidArgumentError("Missing tensor index data or key for multihash computation");
  }

  std::string result = to_multibase_multihash_sha256(digest);
  if (result.empty()) {
    return absl::InternalError("Failed to generate multihash");
  }

  return result;
}

absl::StatusOr<std::string> compute_data_multihash_from_gpu(void* gpu_ptr, uint64_t total_size, int device_id) {
  if (gpu_ptr == nullptr) {
    return absl::InvalidArgumentError("GPU pointer is null");
  }

  if (total_size == 0) {
    return absl::InvalidArgumentError("Cannot hash empty GPU buffer");
  }

  if (total_size > 1ULL << 40) { // 1TB limit
    return absl::InvalidArgumentError(absl::StrCat("GPU buffer too large for hashing: ", total_size, " bytes"));
  }

  // Set CUDA device with error checking
  if (auto st = cuda::set_device(device_id); !st.ok()) {
    return absl::InternalError(absl::StrCat("Failed to set CUDA device ", device_id, ": ", st.ToString()));
  }

  // Validate chunk size
  const size_t chunk_size_bytes = kDefaultChunkSize;
  if (chunk_size_bytes > kMaxChunkSize) {
    return absl::InvalidArgumentError(
        absl::StrCat("Chunk size too large: ", chunk_size_bytes, " (max: ", kMaxChunkSize, ")"));
  }

  std::vector<std::vector<uint8_t>> leaves;
  leaves.reserve(static_cast<size_t>((total_size + chunk_size_bytes - 1) / chunk_size_bytes));

  uint8_t* src = static_cast<uint8_t*>(gpu_ptr);
  std::vector<uint8_t> host_buf(chunk_size_bytes);
  uint64_t processed = 0;

  while (processed < total_size) {
    size_t to_copy = static_cast<size_t>(std::min<uint64_t>(chunk_size_bytes, total_size - processed));

    // Validate copy parameters
    if (to_copy == 0 || to_copy > chunk_size_bytes) {
      return absl::InternalError(absl::StrCat("Invalid copy size: ", to_copy));
    }

    // Perform GPU to host memory copy with comprehensive error handling
    auto st = cuda::memcpy(host_buf.data(), src + processed, to_copy, cudaMemcpyDeviceToHost);
    if (!st.ok()) {
      return absl::InternalError(absl::StrCat("CUDA memcpy failed at offset ", processed, ": ", st.ToString()));
    }

    // Synchronize to ensure copy is complete
    if (auto sync_st = cuda::device_synchronize(); !sync_st.ok()) {
      return absl::InternalError(absl::StrCat("CUDA synchronize failed after memcpy: ", sync_st.ToString()));
    }

    // Hash the chunk
    auto hash_result = sha256_bytes(absl::Span<const uint8_t>(host_buf.data(), to_copy));
    if (!hash_result.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to hash chunk at offset ", processed, ": ", hash_result.status().ToString()));
    }

    leaves.push_back(std::move(hash_result.value()));
    processed += to_copy;
  }

  // Compute tree hash from leaves
  auto root_result = compute_tree_hash_sha256(leaves);
  if (!root_result.ok()) {
    return root_result.status();
  }

  std::string multihash = to_multibase_multihash_sha256(root_result.value());
  if (multihash.empty()) {
    return absl::InternalError("Failed to generate multihash");
  }

  return multihash;
}

} // namespace stepcast::store::artifact_hash