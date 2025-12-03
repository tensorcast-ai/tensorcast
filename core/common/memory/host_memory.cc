// Copyright (c) 2025, TensorCast Team.

#include "core/common/memory/host_memory.h"

#include <fstream>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"

namespace tensorcast::common::memory {
namespace {

absl::Mutex& override_mutex() {
  static absl::Mutex mu;
  return mu;
}

std::optional<uint64_t>& override_bytes() {
  static std::optional<uint64_t> bytes;
  return bytes;
}

std::optional<uint64_t> read_cgroup_memory_max() {
  std::ifstream file("/sys/fs/cgroup/memory.max");
  if (!file.is_open()) {
    return std::nullopt;
  }
  std::string line;
  if (!std::getline(file, line)) {
    return std::nullopt;
  }
  line = std::string(absl::StripAsciiWhitespace(line));
  if (line == "max" || line.empty()) {
    return std::nullopt;
  }
  uint64_t value = 0;
  if (!absl::SimpleAtoi(line, &value)) {
    return std::nullopt;
  }
  return value;
}

std::optional<uint64_t> read_meminfo_memtotal() {
  std::ifstream file("/proc/meminfo");
  if (!file.is_open()) {
    return std::nullopt;
  }
  std::string line;
  while (std::getline(file, line)) {
    // /proc/meminfo uses lines like "MemTotal:       16343456 kB"
    if (!absl::StartsWith(line, "MemTotal:")) {
      continue;
    }
    const auto parts = absl::StrSplit(line, absl::ByAnyChar(" \t"), absl::SkipEmpty());
    for (const auto& token : parts) {
      uint64_t value_kb = 0;
      if (absl::SimpleAtoi(token, &value_kb)) {
        return value_kb * 1024ULL;
      }
    }
  }
  return std::nullopt;
}

} // namespace

absl::StatusOr<uint64_t> detect_host_memory_capacity_bytes() {
  {
    absl::MutexLock lk(&override_mutex());
    if (override_bytes().has_value()) {
      return *override_bytes();
    }
  }
  if (auto cgroup_limit = read_cgroup_memory_max(); cgroup_limit.has_value()) {
    return *cgroup_limit;
  }
  if (auto mem_total = read_meminfo_memtotal(); mem_total.has_value()) {
    return *mem_total;
  }
  return absl::UnavailableError("Unable to determine host memory capacity");
}

void set_host_memory_capacity_override_for_testing(std::optional<uint64_t> bytes) {
  absl::MutexLock lk(&override_mutex());
  override_bytes() = bytes;
}

} // namespace tensorcast::common::memory
