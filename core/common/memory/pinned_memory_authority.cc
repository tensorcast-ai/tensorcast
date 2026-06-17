// Copyright (c) 2026, TensorCast Team.

#include "core/common/memory/pinned_memory_authority.h"

#include <algorithm>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/futures/Future.h"

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

size_t startup_parallelism(size_t task_count) {
  const size_t hw = static_cast<size_t>(std::thread::hardware_concurrency());
  if (task_count == 0) {
    return 1;
  }
  if (hw == 0) {
    return task_count;
  }
  return std::max<size_t>(1, std::min(task_count, hw));
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

  pools_.resize(classes_.size());
  if (classes_.size() == 1) {
    const auto& cls = classes_.front();
    PinnedBufferPool::Options options;
    options.name = std::string(cls.name);
    options.numa_node = cls.numa_node;
    options.prefault = cls.numa_prefault;
    options.register_on_create = !cfg_.defer_host_registration;
    pools_.front() = std::make_shared<PinnedBufferPool>(static_cast<size_t>(cls.pool_bytes), cls.slice_bytes, options);
    return absl::OkStatus();
  }

  folly::CPUThreadPoolExecutor executor(startup_parallelism(classes_.size()));
  auto keep_alive = folly::getKeepAliveToken(executor);
  std::vector<folly::SemiFuture<absl::StatusOr<std::shared_ptr<PinnedBufferPool>>>> tasks;
  tasks.reserve(classes_.size());
  const bool register_on_create = !cfg_.defer_host_registration;
  for (const auto& cls : classes_) {
    tasks.push_back(
        folly::via(keep_alive.copy(), [cls, register_on_create]() -> absl::StatusOr<std::shared_ptr<PinnedBufferPool>> {
          try {
            PinnedBufferPool::Options options{
                .name = std::string(cls.name),
                .register_on_create = register_on_create,
            };
            return std::make_shared<PinnedBufferPool>(static_cast<size_t>(cls.pool_bytes), cls.slice_bytes, options);
          } catch (const std::exception& ex) {
            return absl::InternalError(
                absl::StrCat("PinnedBufferPool construction threw for class ", cls.name, ": ", ex.what()));
          }
        }));
  }

  auto results = folly::collectAll(std::move(tasks)).get();
  for (size_t idx = 0; idx < results.size(); ++idx) {
    const auto& result = results[idx];
    if (!result.hasValue()) {
      return absl::InternalError(
          absl::StrCat("PinnedBufferPool construction future failed for class ", classes_[idx].name));
    }
    auto pool_or = result.value();
    if (!pool_or.ok()) {
      return pool_or.status();
    }
    pools_[idx] = std::move(*pool_or);
  }

  return absl::OkStatus();
}

absl::Status PinnedMemoryAuthority::register_all_pools() {
  if (pools_.empty()) {
    return absl::OkStatus();
  }
  if (pools_.size() == 1) {
    return pools_.front()->register_host_memory();
  }

  folly::CPUThreadPoolExecutor executor(startup_parallelism(pools_.size()));
  auto keep_alive = folly::getKeepAliveToken(executor);
  std::vector<folly::SemiFuture<absl::Status>> tasks;
  tasks.reserve(pools_.size());
  for (size_t idx = 0; idx < pools_.size(); ++idx) {
    auto pool = pools_[idx];
    const std::string class_name = classes_[idx].name;
    tasks.push_back(folly::via(keep_alive.copy(), [pool = std::move(pool), class_name]() -> absl::Status {
      try {
        auto status = pool->register_host_memory();
        if (!status.ok()) {
          return absl::Status(
              status.code(),
              absl::StrCat("host registration failed for pinned pool ", class_name, ": ", status.message()));
        }
        return absl::OkStatus();
      } catch (const std::exception& ex) {
        return absl::InternalError(
            absl::StrCat("PinnedBufferPool host registration threw for class ", class_name, ": ", ex.what()));
      }
    }));
  }

  auto results = folly::collectAll(std::move(tasks)).get();
  for (size_t idx = 0; idx < results.size(); ++idx) {
    const auto& result = results[idx];
    if (!result.hasValue()) {
      return absl::InternalError(
          absl::StrCat("PinnedBufferPool host registration future failed for class ", classes_[idx].name));
    }
    const absl::Status status = result.value();
    if (!status.ok()) {
      return status;
    }
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
