// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/common/model_hash.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "core/common/cuda_api.h"
// NOTE: model_hash is intended to be loader-agnostic; the disk-dir helper is
// now implemented in core/store/loader. Keep GPU hashing and index hashing here.
// Any disk directory hashing should be performed via loader::disk_dir_hash.

namespace stepcast::store::model_hash {

namespace {

// Base32 (lowercase) without padding, RFC 4648 alphabet
std::string base32_lower_encode(const std::vector<uint8_t>& data) {
  static constexpr char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz234567";
  const size_t in_len = data.size();
  if (in_len == 0) {
    return std::string();
  }

  std::string out;
  out.reserve(((in_len + 4) / 5) * 8);
  size_t i = 0;
  while (i < in_len) {
    uint64_t b0 = i < in_len ? data[i++] : 0;
    uint64_t b1 = i < in_len ? data[i++] : 0;
    uint64_t b2 = i < in_len ? data[i++] : 0;
    uint64_t b3 = i < in_len ? data[i++] : 0;
    uint64_t b4 = i < in_len ? data[i++] : 0;
    uint64_t buffer = (b0 << 32) | (b1 << 24) | (b2 << 16) | (b3 << 8) | b4;
    out.push_back(kAlphabet[(buffer >> 35) & 0x1F]);
    out.push_back(kAlphabet[(buffer >> 30) & 0x1F]);
    out.push_back(kAlphabet[(buffer >> 25) & 0x1F]);
    out.push_back(kAlphabet[(buffer >> 20) & 0x1F]);
    out.push_back(kAlphabet[(buffer >> 15) & 0x1F]);
    out.push_back(kAlphabet[(buffer >> 10) & 0x1F]);
    out.push_back(kAlphabet[(buffer >> 5) & 0x1F]);
    out.push_back(kAlphabet[(buffer >> 0) & 0x1F]);
  }

  size_t rem = in_len % 5;
  if (rem == 0) {
    return out;
  }
  size_t keep = 0;
  switch (rem) {
    case 1:
      keep = ((in_len / 5) * 8) + 2;
      break;
    case 2:
      keep = ((in_len / 5) * 8) + 4;
      break;
    case 3:
      keep = ((in_len / 5) * 8) + 5;
      break;
    case 4:
      keep = ((in_len / 5) * 8) + 7;
      break;
    default:
      keep = out.size();
      break;
  }
  out.resize(keep);
  return out;
}

inline uint32_t rotr(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32 - n));
}
inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}
inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (x & z) ^ (y & z);
}
inline uint32_t bsig0(uint32_t x) {
  return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}
inline uint32_t bsig1(uint32_t x) {
  return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}
inline uint32_t ssig0(uint32_t x) {
  return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}
inline uint32_t ssig1(uint32_t x) {
  return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

struct Sha256Ctx {
  uint32_t state[8];
  uint64_t bitlen;
  uint8_t data[64];
  size_t datalen;
};

void sha256_init(Sha256Ctx* ctx) {
  ctx->datalen = 0;
  ctx->bitlen = 0;
  ctx->state[0] = 0x6a09e667;
  ctx->state[1] = 0xbb67ae85;
  ctx->state[2] = 0x3c6ef372;
  ctx->state[3] = 0xa54ff53a;
  ctx->state[4] = 0x510e527f;
  ctx->state[5] = 0x9b05688c;
  ctx->state[6] = 0x1f83d9ab;
  ctx->state[7] = 0x5be0cd19;
}

void sha256_transform(Sha256Ctx* ctx, const uint8_t data[]) {
  static const uint32_t kSha256RoundConstants[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (data[i * 4] << 24) | (data[i * 4 + 1] << 16) | (data[i * 4 + 2] << 8) | (data[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    w[i] = ssig1(w[i - 2]) + w[i - 7] + ssig0(w[i - 15]) + w[i - 16];
  }

  uint32_t a = ctx->state[0];
  uint32_t b = ctx->state[1];
  uint32_t c = ctx->state[2];
  uint32_t d = ctx->state[3];
  uint32_t e = ctx->state[4];
  uint32_t f = ctx->state[5];
  uint32_t g = ctx->state[6];
  uint32_t h = ctx->state[7];

  for (int i = 0; i < 64; ++i) {
    uint32_t t1 = h + bsig1(e) + ch(e, f, g) + kSha256RoundConstants[i] + w[i];
    uint32_t t2 = bsig0(a) + maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t2 + t1;
  }

  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

void sha256_update(Sha256Ctx* ctx, const uint8_t* data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    ctx->data[ctx->datalen] = data[i];
    ctx->datalen++;
    if (ctx->datalen == 64) {
      sha256_transform(ctx, ctx->data);
      ctx->bitlen += 512;
      ctx->datalen = 0;
    }
  }
}

void sha256_final(Sha256Ctx* ctx, uint8_t hash[32]) {
  size_t i = ctx->datalen;
  if (ctx->datalen < 56) {
    ctx->data[i++] = 0x80;
    while (i < 56) {
      ctx->data[i++] = 0x00;
    }
  } else {
    ctx->data[i++] = 0x80;
    while (i < 64) {
      ctx->data[i++] = 0x00;
    }
    sha256_transform(ctx, ctx->data);
    std::fill(std::begin(ctx->data), std::end(ctx->data), 0);
  }
  ctx->bitlen += ctx->datalen * 8;
  ctx->data[63] = ctx->bitlen & 0xFF;
  ctx->data[62] = (ctx->bitlen >> 8) & 0xFF;
  ctx->data[61] = (ctx->bitlen >> 16) & 0xFF;
  ctx->data[60] = (ctx->bitlen >> 24) & 0xFF;
  ctx->data[59] = (ctx->bitlen >> 32) & 0xFF;
  ctx->data[58] = (ctx->bitlen >> 40) & 0xFF;
  ctx->data[57] = (ctx->bitlen >> 48) & 0xFF;
  ctx->data[56] = (ctx->bitlen >> 56) & 0xFF;
  sha256_transform(ctx, ctx->data);
  for (i = 0; i < 8; ++i) {
    hash[i * 4 + 0] = (ctx->state[i] >> 24) & 0xFF;
    hash[i * 4 + 1] = (ctx->state[i] >> 16) & 0xFF;
    hash[i * 4 + 2] = (ctx->state[i] >> 8) & 0xFF;
    hash[i * 4 + 3] = (ctx->state[i]) & 0xFF;
  }
}

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

std::string to_multibase_multihash_sha256(const std::vector<uint8_t>& digest) {
  std::vector<uint8_t> mh;
  mh.reserve(2 + digest.size());
  mh.push_back(0x12);
  mh.push_back(0x20);
  mh.insert(mh.end(), digest.begin(), digest.end());
  std::string b32 = base32_lower_encode(mh);
  return std::string("b") + b32;
}

std::vector<uint8_t> sha256_bytes(absl::Span<const uint8_t> bytes) {
  Sha256Ctx ctx;
  sha256_init(&ctx);
  sha256_update(&ctx, bytes.data(), bytes.size());
  std::vector<uint8_t> out(32);
  sha256_final(&ctx, out.data());
  return out;
}

std::vector<uint8_t> compute_tree_hash_sha256(const std::vector<std::vector<uint8_t>>& leaf_digests) {
  if (leaf_digests.empty()) {
    return std::vector<uint8_t>(32, 0);
  }
  std::vector<std::vector<uint8_t>> level = leaf_digests;
  while (level.size() > 1) {
    std::vector<std::vector<uint8_t>> next;
    next.reserve((level.size() + 1) / 2);
    for (size_t i = 0; i < level.size(); i += 2) {
      if (i + 1 < level.size()) {
        std::vector<uint8_t> concat;
        concat.reserve(64);
        concat.insert(concat.end(), level[i].begin(), level[i].end());
        concat.insert(concat.end(), level[i + 1].begin(), level[i + 1].end());
        next.push_back(sha256_bytes(concat));
      } else {
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
  return sha256_bytes(bytes);
}

std::vector<uint8_t> compute_tree_hash_root_sha256(const std::vector<std::vector<uint8_t>>& leaf_digests) {
  return compute_tree_hash_sha256(leaf_digests);
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
    digest = sha256_bytes(bytes);
  } else if (!index_key_hex.empty()) {
    auto key_bytes = hex_to_bytes(index_key_hex);
    if (key_bytes.size() != 32) {
      return absl::InvalidArgumentError("tensor_index_key must be a 32-byte sha256 hex string");
    }
    digest = std::move(key_bytes);
  } else {
    return absl::InvalidArgumentError("Missing tensor index data or key for multihash computation");
  }
  return to_multibase_multihash_sha256(digest);
}

absl::StatusOr<std::string> compute_data_multihash_from_gpu(void* gpu_ptr, uint64_t total_size, int device_id) {
  if (gpu_ptr == nullptr || total_size == 0) {
    return absl::InvalidArgumentError("Invalid GPU buffer for hashing");
  }
  if (auto st = cuda::set_device(device_id); !st.ok()) {
    return st;
  }
  const size_t chunk_size_bytes = 4ULL * 1024 * 1024; // 4 MiB leaves
  std::vector<std::vector<uint8_t>> leaves;
  leaves.reserve(static_cast<size_t>((total_size + chunk_size_bytes - 1) / chunk_size_bytes));

  uint8_t* src = static_cast<uint8_t*>(gpu_ptr);
  std::vector<uint8_t> host_buf(chunk_size_bytes);
  uint64_t processed = 0;
  while (processed < total_size) {
    size_t to_copy = static_cast<size_t>(std::min<uint64_t>(chunk_size_bytes, total_size - processed));
    auto st = cuda::memcpy(host_buf.data(), src + processed, to_copy, cudaMemcpyDeviceToHost);
    if (!st.ok()) {
      return st;
    }
    leaves.push_back(sha256_bytes(absl::Span<const uint8_t>(host_buf.data(), to_copy)));
    processed += to_copy;
  }
  std::vector<uint8_t> root = compute_tree_hash_sha256(leaves);
  return to_multibase_multihash_sha256(root);
}

absl::StatusOr<std::string> compute_data_multihash_from_disk_dir(const std::string& /*model_dir*/) {
  return absl::UnimplementedError(
      "compute_data_multihash_from_disk_dir moved to core/store/loader; use loader::compute_data_multihash_from_disk_dir");
}

} // namespace stepcast::store::model_hash
