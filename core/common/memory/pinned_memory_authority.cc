// Copyright (c) 2026, TensorCast Team.

#include "core/common/memory/pinned_memory_authority.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

namespace tensorcast::common::memory {

namespace {

constexpr uint64_t kPageBytes = 4096;

absl::Status validate_class_config(const PinnedMemoryAuthority::ClassConfig& cls) {
  if (cls.name.empty()) {
    return absl::InvalidArgumentError("pinned_memory.classes[].name must be non-empty");
  }
  if (cls.slice_bytes == 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("pinned_memory.classes[name=", cls.name, "].slice_bytes must be > 0"));
  }
  if (cls.slice_bytes % kPageBytes != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "pinned_memory.classes[name=%s].slice_bytes=%llu must be a multiple of 4096 bytes",
            cls.name,
            static_cast<unsigned long long>(cls.slice_bytes)));
  }
  if (cls.pool_bytes == 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("pinned_memory.classes[name=", cls.name, "].pool_bytes must be > 0"));
  }
  if (cls.pool_bytes % cls.slice_bytes != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "pinned_memory.classes[name=%s].pool_bytes=%llu must be a multiple of slice_bytes=%llu",
            cls.name,
            static_cast<unsigned long long>(cls.pool_bytes),
            static_cast<unsigned long long>(cls.slice_bytes)));
  }
  return absl::OkStatus();
}

} // namespace

PinnedMemoryAuthority::PinnedMemoryAuthority(Config cfg) : cfg_(std::move(cfg)) {}

absl::StatusOr<std::shared_ptr<PinnedMemoryAuthority>> PinnedMemoryAuthority::create(Config cfg) {
  auto authority = std::shared_ptr<PinnedMemoryAuthority>(new PinnedMemoryAuthority(std::move(cfg)));
  const absl::Status st = authority->validate_and_build_pools();
  if (!st.ok()) {
    return st;
  }
  return authority;
}

absl::Status PinnedMemoryAuthority::validate_and_build_pools() {
  if (cfg_.classes.empty()) {
    return absl::InvalidArgumentError("pinned_memory.classes must be non-empty");
  }

  std::unordered_map<std::string, size_t> seen;
  uint64_t sum_pool_bytes = 0;
  classes_.clear();
  pools_.clear();
  classes_.reserve(cfg_.classes.size());
  pools_.reserve(cfg_.classes.size());

  for (const auto& cls : cfg_.classes) {
    const absl::Status st = validate_class_config(cls);
    if (!st.ok()) {
      return st;
    }
    if (!seen.emplace(cls.name, classes_.size()).second) {
      return absl::InvalidArgumentError(absl::StrCat("Duplicate pinned_memory class name: ", cls.name));
    }
    sum_pool_bytes += cls.pool_bytes;
    classes_.push_back(cls);
  }

  // Phase 1 fixed-allocation: total pinned reservation is derived from class pools.
  cfg_.total_bytes = sum_pool_bytes;

  for (const auto& cls : classes_) {
    PinnedBufferPool::Options pool_opts;
    pool_opts.name = cls.name;
    pool_opts.numa_node = cls.numa_node;
    pool_opts.prefault = cls.numa_prefault;
    pools_.push_back(
        std::make_shared<PinnedBufferPool>(static_cast<size_t>(cls.pool_bytes), cls.slice_bytes, std::move(pool_opts)));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<PinnedBufferPool>> PinnedMemoryAuthority::get_class_pool(std::string_view name) const {
  for (size_t i = 0; i < classes_.size(); ++i) {
    if (classes_[i].name == name) {
      return pools_[i];
    }
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown pinned_memory class: ", name));
}

absl::StatusOr<PinnedMemoryAuthority::ClassConfig> PinnedMemoryAuthority::get_class_config(
    std::string_view name) const {
  for (const auto& cls : classes_) {
    if (cls.name == name) {
      return cls;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown pinned_memory class: ", name));
}

uint64_t PinnedMemoryAuthority::committed_bytes() const {
  uint64_t total = 0;
  for (size_t i = 0; i < classes_.size(); ++i) {
    const auto& pool = pools_[i];
    total += static_cast<uint64_t>(pool->capacity_slices() * pool->slice_bytes());
  }
  return total;
}

std::vector<std::string> PinnedMemoryAuthority::class_names() const {
  std::vector<std::string> out;
  out.reserve(classes_.size());
  for (const auto& cls : classes_) {
    out.push_back(cls.name);
  }
  return out;
}

} // namespace tensorcast::common::memory
