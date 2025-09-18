// Copyright (c) 2025, TensorCast Team.

#include "core/common/artifact_hash.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <openssl/evp.h>
#include <openssl/sha.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash_internal.h"

namespace tensorcast::common {

namespace {

// Base32 RFC 4648 alphabet (lowercase without padding)
constexpr std::array<char, 32> kBase32Alphabet{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                                               'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                                               'w', 'x', 'y', 'z', '2', '3', '4', '5', '6', '7'};

std::string base32_lower_encode(const std::vector<uint8_t>& data) {
  if (data.empty()) {
    return {};
  }

  const size_t in_len = data.size();
  std::string result;
  result.reserve(((in_len + 4) / 5) * 8);

  size_t i = 0;
  while (i < in_len) {
    uint64_t buffer = 0;
    const size_t bytes_to_process = std::min<size_t>(5, in_len - i);

    for (size_t j = 0; j < bytes_to_process; ++j) {
      buffer = (buffer << 8) | data[i + j];
    }

    buffer <<= (5 - bytes_to_process) * 8;

    for (int shift = 35; shift >= 0; shift -= 5) {
      const auto idx = static_cast<size_t>((buffer >> shift) & 0x1F);
      result.push_back(kBase32Alphabet.at(idx));
    }

    i += bytes_to_process;
  }

  const size_t output_length = ((in_len * 8 + 4) / 5);
  result.resize(output_length);
  return result;
}

// RAII wrapper for EVP_MD_CTX
class EvpMdCtxWrapper {
 public:
  EvpMdCtxWrapper() : ctx_(EVP_MD_CTX_new()) {
    if (ctx_ == nullptr) {
      LOG(ERROR) << "Failed to create EVP_MD_CTX";
    }
  }

  ~EvpMdCtxWrapper() {
    if (ctx_ != nullptr) {
      EVP_MD_CTX_free(ctx_);
    }
  }

  EvpMdCtxWrapper(const EvpMdCtxWrapper&) = delete;
  EvpMdCtxWrapper& operator=(const EvpMdCtxWrapper&) = delete;

  EvpMdCtxWrapper(EvpMdCtxWrapper&& other) noexcept : ctx_(other.ctx_) {
    other.ctx_ = nullptr;
  }

  EVP_MD_CTX* get() {
    return ctx_;
  }

  [[nodiscard]] bool valid() const {
    return ctx_ != nullptr;
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
    const int hi = hexval(hex[i]);
    const int lo = hexval(hex[i + 1]);
    if (hi < 0 || lo < 0) {
      return {};
    }
    out.push_back(static_cast<uint8_t>((hi << 4) | lo));
  }
  return out;
}

bool validate_multihash_format(const std::string& multihash) {
  if (multihash.size() < 3) {
    return false;
  }

  if (multihash[0] != 'b') {
    return false;
  }

  for (size_t i = 1; i < multihash.size(); ++i) {
    char c = multihash[i];
    const bool valid = (c >= 'a' && c <= 'z') || (c >= '2' && c <= '7');
    if (!valid) {
      return false;
    }
  }

  return true;
}

size_t align_down_to(size_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const size_t remainder = value % alignment;
  return remainder == 0 ? value : value - remainder;
}

} // namespace

namespace internal {

absl::StatusOr<std::vector<uint8_t>> sha256_bytes(absl::Span<const uint8_t> bytes) {
  if (bytes.empty()) {
    return std::vector<uint8_t>(SHA256_DIGEST_LENGTH, 0);
  }

  EvpMdCtxWrapper ctx_wrapper;
  if (!ctx_wrapper.valid()) {
    return absl::InternalError("Failed to allocate EVP_MD_CTX");
  }

  EVP_MD_CTX* ctx = ctx_wrapper.get();
  const EVP_MD* sha256 = EVP_sha256();
  if (sha256 == nullptr) {
    return absl::InternalError("Failed to get SHA256 EVP_MD");
  }

  if (EVP_DigestInit_ex(ctx, sha256, nullptr) != 1) {
    return absl::InternalError("EVP_DigestInit_ex failed");
  }

  if (EVP_DigestUpdate(ctx, bytes.data(), bytes.size()) != 1) {
    return absl::InternalError("EVP_DigestUpdate failed");
  }

  std::vector<uint8_t> digest(SHA256_DIGEST_LENGTH);
  unsigned int out_len = 0;
  if (EVP_DigestFinal_ex(ctx, digest.data(), &out_len) != 1) {
    return absl::InternalError("EVP_DigestFinal_ex failed");
  }

  if (out_len != SHA256_DIGEST_LENGTH) {
    return absl::InternalError(absl::StrCat("Unexpected SHA256 digest length: ", out_len));
  }

  return digest;
}

absl::StatusOr<std::vector<uint8_t>> compute_tree_hash_sha256(const std::vector<std::vector<uint8_t>>& leaf_digests) {
  if (leaf_digests.empty()) {
    return absl::InvalidArgumentError("Leaf digests must not be empty");
  }

  for (const auto& leaf : leaf_digests) {
    if (leaf.size() != SHA256_DIGEST_LENGTH) {
      return absl::InvalidArgumentError("Leaf digest length must be 32 bytes");
    }
  }

  std::vector<std::vector<uint8_t>> level = leaf_digests;
  while (level.size() > 1) {
    std::vector<std::vector<uint8_t>> next;
    next.reserve((level.size() + 1) / 2);
    for (size_t i = 0; i < level.size(); i += 2) {
      if (i + 1 < level.size()) {
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
        next.push_back(level[i]);
      }
    }
    level.swap(next);
  }

  return level.front();
}

std::string to_multibase_multihash_sha256(const std::vector<uint8_t>& digest) {
  if (digest.size() != SHA256_DIGEST_LENGTH) {
    LOG(ERROR) << "Invalid digest size for SHA256: " << digest.size();
    return "";
  }

  std::vector<uint8_t> mh;
  mh.reserve(2 + digest.size());
  mh.push_back(0x12);
  mh.push_back(0x20);
  mh.insert(mh.end(), digest.begin(), digest.end());

  std::string b32 = base32_lower_encode(mh);
  std::string result = std::string("b") + b32;
  if (!validate_multihash_format(result)) {
    LOG(ERROR) << "Generated invalid multihash format";
    return "";
  }
  return result;
}

std::vector<uint64_t> compute_chunk_lengths(uint64_t total_size, size_t chunk_size_bytes) {
  const auto num_chunks = static_cast<size_t>((total_size + chunk_size_bytes - 1) / chunk_size_bytes);
  std::vector<uint64_t> lengths;
  lengths.reserve(num_chunks);
  uint64_t remaining = total_size;
  while (remaining > 0) {
    const uint64_t chunk = std::min<uint64_t>(chunk_size_bytes, remaining);
    lengths.push_back(chunk);
    remaining -= chunk;
  }
  return lengths;
}

size_t determine_leaf_chunk_size(uint64_t total_size, size_t requested_chunk_bytes) {
  size_t bounded = requested_chunk_bytes;
  if (bounded == 0) {
    bounded = kGpuHashDefaultLeafChunkBytes;
  }
  bounded = std::min(bounded, kMaxChunkSize);
  bounded = std::max(bounded, static_cast<size_t>(64));
  bounded = align_down_to(bounded, 64);
  if (bounded == 0) {
    bounded = 64;
  }

  if (requested_chunk_bytes != kGpuHashDefaultLeafChunkBytes) {
    return bounded;
  }

  const auto compute_leaf_count = [&](size_t chunk_bytes) -> uint64_t {
    return (total_size + static_cast<uint64_t>(chunk_bytes) - 1ULL) / static_cast<uint64_t>(chunk_bytes);
  };

  size_t adjusted = bounded;
  while (adjusted > kMinLeafChunkBytes && compute_leaf_count(adjusted) < kTargetLeafCount) {
    size_t candidate = align_down_to(adjusted / 2, 64);
    if (candidate < kMinLeafChunkBytes) {
      break;
    }
    adjusted = candidate;
  }

  if (adjusted < kMinLeafChunkBytes) {
    adjusted = align_down_to(kMinLeafChunkBytes, 64);
  }

  return adjusted == 0 ? 64 : adjusted;
}

} // namespace internal

std::vector<uint8_t> sha256_digest_bytes(absl::Span<const uint8_t> bytes) {
  auto result = internal::sha256_bytes(bytes);
  if (!result.ok()) {
    LOG(ERROR) << "SHA256 failed: " << result.status();
    return std::vector<uint8_t>(SHA256_DIGEST_LENGTH, 0);
  }
  return result.value();
}

std::vector<uint8_t> compute_tree_hash_root_sha256(const std::vector<std::vector<uint8_t>>& leaf_digests) {
  auto result = internal::compute_tree_hash_sha256(leaf_digests);
  if (!result.ok()) {
    LOG(ERROR) << "Tree hash failed: " << result.status();
    return std::vector<uint8_t>(SHA256_DIGEST_LENGTH, 0);
  }
  return result.value();
}

std::string multibase_multihash_sha256(const std::vector<uint8_t>& digest) {
  return internal::to_multibase_multihash_sha256(digest);
}

absl::StatusOr<std::string> compute_index_multihash(
    const std::optional<std::string>& index_data,
    std::string_view index_key_hex) {
  std::vector<uint8_t> digest;

  if (index_data.has_value() && !index_data->empty()) {
    const auto bytes =
        absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(index_data->data()), index_data->size());
    auto hash_result = internal::sha256_bytes(bytes);
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

  std::string result = internal::to_multibase_multihash_sha256(digest);
  if (result.empty()) {
    return absl::InternalError("Failed to generate multihash");
  }

  return result;
}

} // namespace tensorcast::common
