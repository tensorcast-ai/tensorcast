// Copyright (c) 2026, TensorCast Team.

#include "daemon/app/startup_memory_preflight.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

namespace tensorcast::daemon {
namespace {

constexpr uint64_t kBytesPerGiB = 1024ULL * 1024 * 1024;
constexpr uint64_t kMaxHeadroomBytes = 10ULL * kBytesPerGiB;

absl::Mutex& override_mutex() {
  static absl::Mutex mu;
  return mu;
}

std::optional<uint64_t>& override_available_bytes() {
  static std::optional<uint64_t> bytes;
  return bytes;
}

absl::StatusOr<std::optional<std::filesystem::path>> resolve_cgroup_v2_path() {
  std::ifstream in("/proc/self/cgroup");
  if (!in.is_open()) {
    return std::optional<std::filesystem::path>{};
  }
  std::string line;
  std::string rel;
  while (std::getline(in, line)) {
    // cgroup v2 line format: 0::/some/path
    if (line.rfind("0::", 0) == 0) {
      rel = line.substr(3);
      break;
    }
  }
  if (rel.empty()) {
    return std::optional<std::filesystem::path>{};
  }
  std::filesystem::path cg_root("/sys/fs/cgroup");
  std::filesystem::path cg_path = cg_root / std::filesystem::path(rel).relative_path();
  return std::optional<std::filesystem::path>{cg_path};
}

absl::StatusOr<std::optional<uint64_t>> read_cgroup_v2_uint64_file(
    const std::filesystem::path& cg_path,
    const std::filesystem::path& file) {
  std::ifstream in(cg_path / file);
  if (!in.is_open()) {
    return std::optional<uint64_t>{};
  }
  std::string raw;
  if (!std::getline(in, raw)) {
    return std::optional<uint64_t>{};
  }
  if (raw.empty() || raw == "max") {
    return std::optional<uint64_t>{};
  }
  try {
    return std::optional<uint64_t>{std::stoull(raw)};
  } catch (...) {
    return std::optional<uint64_t>{};
  }
}

absl::StatusOr<std::optional<uint64_t>> read_cgroup_v2_inactive_file_bytes(const std::filesystem::path& cg_path) {
  std::ifstream in(cg_path / "memory.stat");
  if (!in.is_open()) {
    return std::optional<uint64_t>{};
  }
  std::string key;
  uint64_t value = 0;
  while (in >> key >> value) {
    if (key == "inactive_file") {
      return std::optional<uint64_t>{value};
    }
  }
  return std::optional<uint64_t>{};
}

absl::StatusOr<uint64_t> read_meminfo_memavailable_bytes() {
  std::ifstream in("/proc/meminfo");
  if (!in.is_open()) {
    return absl::ErrnoToStatus(errno, "Failed to open /proc/meminfo");
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("MemAvailable:", 0) != 0) {
      continue;
    }
    const char* p = line.c_str() + std::strlen("MemAvailable:");
    while (*p == ' ' || *p == '\t') {
      ++p;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long kb = std::strtoull(p, &end, 10);
    if (end == p || errno != 0) {
      return absl::InternalError("Failed to parse MemAvailable from /proc/meminfo");
    }
    return static_cast<uint64_t>(kb) * 1024ULL;
  }
  return absl::NotFoundError("MemAvailable not found in /proc/meminfo");
}

std::string format_bytes_gib(uint64_t bytes) {
  const double gib = static_cast<double>(bytes) / static_cast<double>(kBytesPerGiB);
  return std::format("{}B ({:.2f}GiB)", bytes, gib);
}

} // namespace

uint64_t compute_startup_memory_headroom_bytes(uint64_t required_bytes) {
  const uint64_t ten_percent = required_bytes / 10;
  return std::min<uint64_t>(ten_percent, kMaxHeadroomBytes);
}

absl::StatusOr<StartupMemoryAvailability> detect_startup_memory_available_bytes() {
  {
    absl::MutexLock lk(&override_mutex());
    if (override_available_bytes().has_value()) {
      StartupMemoryAvailability out;
      out.available_bytes = *override_available_bytes();
      out.source = "override";
      return out;
    }
  }

  StartupMemoryAvailability out;
  auto cg_path_or = resolve_cgroup_v2_path();
  if (!cg_path_or.ok()) {
    return cg_path_or.status();
  }
  if (cg_path_or->has_value()) {
    const std::filesystem::path& cg_path = **cg_path_or;
    auto max_or = read_cgroup_v2_uint64_file(cg_path, "memory.max");
    if (!max_or.ok()) {
      return max_or.status();
    }
    auto current_or = read_cgroup_v2_uint64_file(cg_path, "memory.current");
    if (!current_or.ok()) {
      return current_or.status();
    }
    if (max_or->has_value() && current_or->has_value()) {
      out.cgroup_memory_max_bytes = **max_or;
      out.cgroup_memory_current_bytes = **current_or;

      auto inactive_or = read_cgroup_v2_inactive_file_bytes(cg_path);
      if (!inactive_or.ok()) {
        return inactive_or.status();
      }
      const uint64_t inactive_file = inactive_or->value_or(0);
      out.cgroup_inactive_file_bytes = inactive_file;

      // Approximate "MemAvailable" for cgroups by treating inactive file cache
      // as reclaimable headroom.
      const uint64_t current = **current_or;
      const uint64_t effective_current = (current > inactive_file) ? (current - inactive_file) : 0;
      const uint64_t max = **max_or;
      out.available_bytes = (max > effective_current) ? (max - effective_current) : 0;
      out.source = "cgroup_v2";
      return out;
    }
  }

  auto memavail_or = read_meminfo_memavailable_bytes();
  if (!memavail_or.ok()) {
    return memavail_or.status();
  }
  out.available_bytes = *memavail_or;
  out.source = "meminfo";
  return out;
}

absl::Status preflight_startup_memory(uint64_t pinned_bytes, uint64_t stable_bytes) {
  const uint64_t required_bytes = pinned_bytes + stable_bytes;
  const uint64_t headroom_bytes = compute_startup_memory_headroom_bytes(required_bytes);
  const uint64_t total_required_bytes = required_bytes + headroom_bytes;

  auto avail_or = detect_startup_memory_available_bytes();
  if (!avail_or.ok()) {
    return avail_or.status();
  }
  const auto& avail = *avail_or;

  LOG(INFO) << "Startup memory preflight: source=" << avail.source
            << " available=" << format_bytes_gib(avail.available_bytes)
            << " required_pinned=" << format_bytes_gib(pinned_bytes)
            << " required_stable=" << format_bytes_gib(stable_bytes) << " headroom=" << format_bytes_gib(headroom_bytes)
            << " total_required=" << format_bytes_gib(total_required_bytes);

  if (avail.available_bytes < total_required_bytes) {
    if (avail.source == "cgroup_v2") {
      return absl::ResourceExhaustedError(
          std::format(
              "Insufficient startup memory (cgroup v2): available={} total_required={} "
              "(pinned={} stable={} headroom={}). memory.max={} memory.current={} inactive_file={}",
              format_bytes_gib(avail.available_bytes),
              format_bytes_gib(total_required_bytes),
              format_bytes_gib(pinned_bytes),
              format_bytes_gib(stable_bytes),
              format_bytes_gib(headroom_bytes),
              format_bytes_gib(avail.cgroup_memory_max_bytes.value_or(0)),
              format_bytes_gib(avail.cgroup_memory_current_bytes.value_or(0)),
              format_bytes_gib(avail.cgroup_inactive_file_bytes.value_or(0))));
    }
    return absl::ResourceExhaustedError(
        std::format(
            "Insufficient startup memory: available={} total_required={} "
            "(pinned={} stable={} headroom={})",
            format_bytes_gib(avail.available_bytes),
            format_bytes_gib(total_required_bytes),
            format_bytes_gib(pinned_bytes),
            format_bytes_gib(stable_bytes),
            format_bytes_gib(headroom_bytes)));
  }

  return absl::OkStatus();
}

void set_startup_memory_available_override_for_testing(std::optional<uint64_t> bytes) {
  absl::MutexLock lk(&override_mutex());
  override_available_bytes() = bytes;
}

} // namespace tensorcast::daemon
