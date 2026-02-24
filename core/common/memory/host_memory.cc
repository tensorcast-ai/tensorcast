// Copyright (c) 2025-2026, TensorCast Team.

#include "core/common/memory/host_memory.h"

#include <fstream>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
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

std::optional<uint64_t>& override_capacity_bytes() {
  static std::optional<uint64_t> bytes;
  return bytes;
}

std::optional<uint64_t>& override_available_bytes() {
  static std::optional<uint64_t> bytes;
  return bytes;
}

std::optional<std::string> read_file_first_line(const char* path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return std::nullopt;
  }
  std::string line;
  if (!std::getline(file, line)) {
    return std::nullopt;
  }
  line = std::string(absl::StripAsciiWhitespace(line));
  return line;
}

std::optional<uint64_t> parse_uint64_line(const std::string& line) {
  if (line == "max" || line.empty()) {
    return std::nullopt;
  }
  uint64_t value = 0;
  if (!absl::SimpleAtoi(line, &value)) {
    return std::nullopt;
  }
  return value;
}

std::optional<uint64_t> read_uint64_file(const char* path) {
  auto line = read_file_first_line(path);
  if (!line.has_value()) {
    return std::nullopt;
  }
  return parse_uint64_line(*line);
}

bool cgroup_limit_is_unbounded(uint64_t limit_bytes) {
  // cgroup v1 reports a huge sentinel near ULLONG_MAX for "unlimited".
  return limit_bytes >= (1ULL << 60);
}

std::optional<uint64_t> read_meminfo_value_bytes(const char* key_prefix) {
  std::ifstream file("/proc/meminfo");
  if (!file.is_open()) {
    return std::nullopt;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (!absl::StartsWith(line, key_prefix)) {
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

std::optional<uint64_t> read_memory_stat_key(const char* path, const char* key) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return std::nullopt;
  }
  std::string stat_key;
  uint64_t stat_value = 0;
  while (file >> stat_key >> stat_value) {
    if (stat_key == key) {
      return stat_value;
    }
  }
  return std::nullopt;
}

std::optional<uint64_t> read_cgroup_v2_memory_limit() {
  auto value = read_uint64_file("/sys/fs/cgroup/memory.max");
  if (!value.has_value() || cgroup_limit_is_unbounded(*value)) {
    return std::nullopt;
  }
  return value;
}

std::optional<uint64_t> read_cgroup_v2_memory_current() {
  return read_uint64_file("/sys/fs/cgroup/memory.current");
}

std::optional<uint64_t> read_cgroup_v2_inactive_file() {
  return read_memory_stat_key("/sys/fs/cgroup/memory.stat", "inactive_file");
}

std::optional<uint64_t> read_cgroup_v1_memory_limit() {
  auto value = read_uint64_file("/sys/fs/cgroup/memory/memory.limit_in_bytes");
  if (!value.has_value() || cgroup_limit_is_unbounded(*value)) {
    return std::nullopt;
  }
  return value;
}

std::optional<uint64_t> read_cgroup_v1_memory_usage() {
  return read_uint64_file("/sys/fs/cgroup/memory/memory.usage_in_bytes");
}

std::optional<uint64_t> read_cgroup_v1_inactive_file() {
  if (auto total_inactive = read_memory_stat_key("/sys/fs/cgroup/memory/memory.stat", "total_inactive_file");
      total_inactive.has_value()) {
    return total_inactive;
  }
  return read_memory_stat_key("/sys/fs/cgroup/memory/memory.stat", "inactive_file");
}

std::optional<uint64_t> read_meminfo_memtotal() {
  return read_meminfo_value_bytes("MemTotal:");
}

std::optional<uint64_t> read_meminfo_memavailable() {
  return read_meminfo_value_bytes("MemAvailable:");
}

} // namespace

absl::StatusOr<uint64_t> detect_host_memory_capacity_bytes() {
  {
    absl::MutexLock lk(&override_mutex());
    if (override_capacity_bytes().has_value()) {
      return *override_capacity_bytes();
    }
  }
  if (auto cgroup_limit = read_cgroup_v2_memory_limit(); cgroup_limit.has_value()) {
    return *cgroup_limit;
  }
  if (auto cgroup_limit = read_cgroup_v1_memory_limit(); cgroup_limit.has_value()) {
    return *cgroup_limit;
  }
  if (auto mem_total = read_meminfo_memtotal(); mem_total.has_value()) {
    return *mem_total;
  }
  return absl::UnavailableError("Unable to determine host memory capacity");
}

absl::StatusOr<uint64_t> detect_host_memory_available_bytes() {
  {
    absl::MutexLock lk(&override_mutex());
    if (override_available_bytes().has_value()) {
      return *override_available_bytes();
    }
  }

  if (auto limit = read_cgroup_v2_memory_limit(); limit.has_value()) {
    auto current = read_cgroup_v2_memory_current();
    if (current.has_value()) {
      const uint64_t inactive = read_cgroup_v2_inactive_file().value_or(0);
      const uint64_t effective_current = (*current > inactive) ? (*current - inactive) : 0;
      return (*limit > effective_current) ? (*limit - effective_current) : 0;
    }
  }

  if (auto limit = read_cgroup_v1_memory_limit(); limit.has_value()) {
    auto usage = read_cgroup_v1_memory_usage();
    if (usage.has_value()) {
      const uint64_t inactive = read_cgroup_v1_inactive_file().value_or(0);
      const uint64_t effective_usage = (*usage > inactive) ? (*usage - inactive) : 0;
      return (*limit > effective_usage) ? (*limit - effective_usage) : 0;
    }
  }

  if (auto mem_available = read_meminfo_memavailable(); mem_available.has_value()) {
    return *mem_available;
  }
  return absl::UnavailableError("Unable to determine available host memory");
}

void set_host_memory_capacity_override_for_testing(std::optional<uint64_t> bytes) {
  absl::MutexLock lk(&override_mutex());
  override_capacity_bytes() = bytes;
}

void set_host_memory_available_override_for_testing(std::optional<uint64_t> bytes) {
  absl::MutexLock lk(&override_mutex());
  override_available_bytes() = bytes;
}

} // namespace tensorcast::common::memory
