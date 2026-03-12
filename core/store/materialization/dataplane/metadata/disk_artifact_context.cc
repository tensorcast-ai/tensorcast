// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <cstring>
#include <fstream>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "core/store/materialization/dataplane/metadata/safetensors_util.h"

namespace tensorcast::store::loader {
namespace {

struct CacheState {
  absl::Mutex mutex;
  absl::flat_hash_map<std::string, std::weak_ptr<const DiskArtifactContext>> contexts ABSL_GUARDED_BY(mutex);
  std::atomic<uint64_t> context_hits{0};
  std::atomic<uint64_t> context_misses{0};
  std::atomic<uint64_t> index_hits{0};
  std::atomic<uint64_t> index_misses{0};
};

CacheState& cache_state() {
  static CacheState state;
  return state;
}

std::filesystem::path normalize_artifact_path(const std::filesystem::path& artifact_path) {
  if (artifact_path.is_absolute()) {
    return artifact_path.lexically_normal();
  }
  return std::filesystem::absolute(artifact_path).lexically_normal();
}

absl::StatusOr<size_t> pread_fully(int fd, uint64_t off, void* dst, size_t bytes) {
  size_t total = 0;
  char* ptr = static_cast<char*>(dst);
  while (total < bytes) {
    const ssize_t got = ::pread(fd, ptr + total, bytes - total, static_cast<off_t>(off + total));
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      return absl::ErrnoToStatus(errno, "pread failed");
    }
    if (got == 0) {
      break;
    }
    total += static_cast<size_t>(got);
  }
  return total;
}

absl::Status validate_safetensors_header_json(int fd, uint64_t header_length) {
  std::string header;
  header.resize(static_cast<size_t>(header_length));
  auto got_or = pread_fully(fd, sizeof(uint64_t), header.data(), header.size());
  if (!got_or.ok()) {
    return got_or.status();
  }
  if (*got_or != header.size()) {
    return absl::InvalidArgumentError("Truncated safetensors header");
  }
  if (header.empty() || header.front() != '{') {
    return absl::InvalidArgumentError("Malformed safetensors header: must start with '{'");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<SharedFileHandle>> open_shared_file(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("Failed to open ", path.string()));
  }
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    const int saved_errno = errno;
    ::close(fd);
    return absl::ErrnoToStatus(saved_errno, absl::StrCat("Failed to stat ", path.string()));
  }
  return std::make_shared<SharedFileHandle>(path, fd, static_cast<uint64_t>(st.st_size));
}

absl::StatusOr<std::shared_ptr<const DiskArtifactContext>> build_disk_artifact_context(
    const std::filesystem::path& artifact_path) {
  std::error_code ec;
  const bool exists = std::filesystem::exists(artifact_path, ec);
  if (ec) {
    return absl::ErrnoToStatus(
        ec.value(), absl::StrCat("Failed to access artifact directory '", artifact_path.string(), "'"));
  }
  if (!exists) {
    return absl::NotFoundError(absl::StrCat("Artifact directory not found: ", artifact_path.string()));
  }
  const bool is_dir = std::filesystem::is_directory(artifact_path, ec);
  if (ec) {
    return absl::ErrnoToStatus(
        ec.value(), absl::StrCat("Failed to stat artifact directory '", artifact_path.string(), "'"));
  }
  if (!is_dir) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected artifact path to be a directory: ", artifact_path.string()));
  }

  bool descriptor_present = false;
  bool tensor_index_json_present = false;
  bool tensor_index_cbor_present = false;
  bool has_single_file = false;
  std::filesystem::path single_file_path;
  std::vector<std::pair<uint64_t, std::filesystem::path>> multipart_paths;
  std::vector<std::filesystem::path> safetensors_paths;

  auto parse_partition_index = [](std::string_view name) -> std::optional<uint64_t> {
    constexpr std::string_view kPrefix = "tensor.data_";
    if (!name.starts_with(kPrefix)) {
      return std::nullopt;
    }
    const std::string_view suffix = name.substr(kPrefix.size());
    if (suffix.empty()) {
      return std::nullopt;
    }
    uint64_t value = 0;
    for (char c : suffix) {
      if (!std::isdigit(static_cast<unsigned char>(c))) {
        return std::nullopt;
      }
      value = value * 10 + static_cast<uint64_t>(c - '0');
    }
    return value;
  };

  for (const auto& entry : std::filesystem::directory_iterator(artifact_path, ec)) {
    if (ec) {
      return absl::ErrnoToStatus(
          ec.value(), absl::StrCat("Failed to enumerate artifact directory '", artifact_path.string(), "'"));
    }
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (filename == "artifact_descriptor.json") {
      descriptor_present = true;
      continue;
    }
    if (filename == "tensor_index.json") {
      tensor_index_json_present = true;
      continue;
    }
    if (filename == "tensor_index.cbor") {
      tensor_index_cbor_present = true;
      continue;
    }
    if (filename == "tensor.data") {
      has_single_file = true;
      single_file_path = entry.path();
      continue;
    }
    if (auto idx = parse_partition_index(filename)) {
      multipart_paths.emplace_back(*idx, entry.path());
      continue;
    }
    if (absl::EndsWith(filename, ".safetensors")) {
      safetensors_paths.push_back(entry.path());
    }
  }

  std::vector<std::filesystem::path> partition_paths;
  std::vector<size_t> partition_sizes;
  std::vector<SharedSafetensorsSegment> safetensors_segments;
  uint64_t total_size = 0;
  bool is_safetensors = false;

  if (!multipart_paths.empty()) {
    std::ranges::sort(multipart_paths, [](const auto& a, const auto& b) {
      if (a.first != b.first) {
        return a.first < b.first;
      }
      return a.second.filename() < b.second.filename();
    });
    partition_paths.reserve(multipart_paths.size());
    partition_sizes.reserve(multipart_paths.size());
    for (const auto& [_, path] : multipart_paths) {
      const uint64_t size = std::filesystem::file_size(path);
      partition_paths.push_back(path);
      partition_sizes.push_back(static_cast<size_t>(size));
      total_size += size;
    }
  } else if (has_single_file) {
    const uint64_t size = std::filesystem::file_size(single_file_path);
    partition_paths.push_back(single_file_path);
    partition_sizes.push_back(static_cast<size_t>(size));
    total_size = size;
  } else if (!safetensors_paths.empty()) {
    is_safetensors = true;
    std::ranges::sort(safetensors_paths, [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
    partition_paths.reserve(safetensors_paths.size());
    partition_sizes.reserve(safetensors_paths.size());
    safetensors_segments.reserve(safetensors_paths.size());
    uint64_t running_base = 0;
    for (const auto& path : safetensors_paths) {
      auto handle_or = open_shared_file(path);
      if (!handle_or.ok()) {
        return handle_or.status();
      }
      auto header_or = ParseSafetensorsHeader((*handle_or)->fd());
      if (!header_or.ok()) {
        return header_or.status();
      }
      auto header_status = validate_safetensors_header_json((*handle_or)->fd(), header_or->header_length);
      if (!header_status.ok()) {
        return header_status;
      }
      partition_paths.push_back(path);
      partition_sizes.push_back(static_cast<size_t>(header_or->data_size));
      safetensors_segments.push_back(
          SharedSafetensorsSegment{
              .path = path,
              .file = std::move(*handle_or),
              .data_start = header_or->data_start,
              .data_size = header_or->data_size,
              .base_offset = running_base,
          });
      running_base += header_or->data_size;
    }
    total_size = running_base;
  } else {
    return absl::NotFoundError(
        absl::StrCat("No replica partition files found in: ", artifact_path.string(), " (also no .safetensors)"));
  }

  return std::shared_ptr<const DiskArtifactContext>(new DiskArtifactContext(
      artifact_path,
      std::move(partition_paths),
      std::move(partition_sizes),
      total_size,
      is_safetensors,
      descriptor_present,
      tensor_index_json_present,
      tensor_index_cbor_present,
      std::move(safetensors_segments)));
}

} // namespace

SharedFileHandle::SharedFileHandle(std::filesystem::path path, int fd, uint64_t file_size)
    : path_(std::move(path)), fd_(fd), file_size_(file_size) {}

SharedFileHandle::~SharedFileHandle() {
  {
    absl::MutexLock lock(&mutex_);
    if (mapped_base_ != nullptr && file_size_ > 0) {
      ::munmap(const_cast<uint8_t*>(mapped_base_), static_cast<size_t>(file_size_));
      mapped_base_ = nullptr;
    }
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

const uint8_t* SharedFileHandle::mapped_base() const {
  absl::MutexLock lock(&mutex_);
  if (mapped_base_ != nullptr) {
    return mapped_base_;
  }
  if (mmap_attempted_ || file_size_ == 0 || fd_ < 0) {
    return nullptr;
  }
  mmap_attempted_ = true;
  void* mapped = ::mmap(nullptr, static_cast<size_t>(file_size_), PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapped == MAP_FAILED) {
    const int saved_errno = errno;
    LOG(WARNING) << "SharedFileHandle mmap failed for " << path_.string() << ": " << strerror(saved_errno);
    return nullptr;
  }
  mapped_base_ = static_cast<const uint8_t*>(mapped);
  return mapped_base_;
}

DiskArtifactContext::DiskArtifactContext(
    std::filesystem::path artifact_path,
    std::vector<std::filesystem::path> partition_paths,
    std::vector<size_t> partition_sizes,
    uint64_t total_size,
    bool is_safetensors,
    bool descriptor_present,
    bool tensor_index_json_present,
    bool tensor_index_cbor_present,
    std::vector<SharedSafetensorsSegment> safetensors_segments)
    : artifact_path_(std::move(artifact_path)),
      partition_paths_(std::move(partition_paths)),
      partition_sizes_(std::move(partition_sizes)),
      total_size_(total_size),
      is_safetensors_(is_safetensors),
      descriptor_present_(descriptor_present),
      tensor_index_json_present_(tensor_index_json_present),
      tensor_index_cbor_present_(tensor_index_cbor_present),
      safetensors_segments_(std::move(safetensors_segments)) {}

absl::StatusOr<IndexInfo> DiskArtifactContext::get_index_info(int target_device_id) const {
  {
    absl::MutexLock lock(&mutex_);
    for (const auto& entry : index_cache_) {
      if (entry.first == target_device_id) {
        LOG(INFO) << "DiskArtifactContext index cache hit path=" << artifact_path_.string()
                  << " target_device_id=" << target_device_id;
        cache_state().index_hits.fetch_add(1, std::memory_order_relaxed);
        return entry.second;
      }
    }
  }

  LOG(INFO) << "DiskArtifactContext index cache miss path=" << artifact_path_.string()
            << " target_device_id=" << target_device_id;
  cache_state().index_misses.fetch_add(1, std::memory_order_relaxed);
  auto info_or = read_from_artifact_dir(artifact_path_, target_device_id);
  if (!info_or.ok()) {
    return info_or.status();
  }

  {
    absl::MutexLock lock(&mutex_);
    index_cache_.push_back({target_device_id, *info_or});
  }
  return *info_or;
}

absl::StatusOr<std::shared_ptr<const DiskArtifactContext>> get_disk_artifact_context(
    const std::filesystem::path& artifact_path) {
  const std::filesystem::path normalized = normalize_artifact_path(artifact_path);
  const std::string cache_key = normalized.string();
  auto& state = cache_state();
  {
    absl::MutexLock lock(&state.mutex);
    auto it = state.contexts.find(cache_key);
    if (it != state.contexts.end()) {
      if (auto existing = it->second.lock(); existing != nullptr) {
        LOG(INFO) << "DiskArtifactContext cache hit path=" << normalized.string();
        state.context_hits.fetch_add(1, std::memory_order_relaxed);
        return existing;
      }
      state.contexts.erase(it);
    }
  }

  auto built_or = build_disk_artifact_context(normalized);
  if (!built_or.ok()) {
    return built_or.status();
  }
  LOG(INFO) << "DiskArtifactContext cache miss path=" << normalized.string();

  {
    absl::MutexLock lock(&state.mutex);
    state.contexts[cache_key] = *built_or;
  }
  state.context_misses.fetch_add(1, std::memory_order_relaxed);
  return *built_or;
}

DiskArtifactContextCacheStats get_disk_artifact_context_cache_stats() {
  auto& state = cache_state();
  return DiskArtifactContextCacheStats{
      .context_hits = state.context_hits.load(std::memory_order_relaxed),
      .context_misses = state.context_misses.load(std::memory_order_relaxed),
      .index_hits = state.index_hits.load(std::memory_order_relaxed),
      .index_misses = state.index_misses.load(std::memory_order_relaxed),
  };
}

void reset_disk_artifact_context_cache_for_testing() {
  auto& state = cache_state();
  {
    absl::MutexLock lock(&state.mutex);
    state.contexts.clear();
  }
  state.context_hits.store(0, std::memory_order_relaxed);
  state.context_misses.store(0, std::memory_order_relaxed);
  state.index_hits.store(0, std::memory_order_relaxed);
  state.index_misses.store(0, std::memory_order_relaxed);
}

} // namespace tensorcast::store::loader
