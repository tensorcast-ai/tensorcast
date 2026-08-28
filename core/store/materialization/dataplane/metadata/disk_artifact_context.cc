// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "core/checkpoint/tensor_writer.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/materialization/dataplane/metadata/safetensors_util.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::loader {
namespace {

struct CacheState {
  absl::Mutex mutex;

  struct ContextEntry {
    std::shared_ptr<const DiskArtifactContext> context;
    uint64_t last_access{0};
  };

  absl::flat_hash_map<std::string, ContextEntry> contexts ABSL_GUARDED_BY(mutex);
  std::atomic<uint64_t> access_counter{0};
  std::atomic<uint64_t> context_hits{0};
  std::atomic<uint64_t> context_misses{0};
  std::atomic<uint64_t> index_hits{0};
  std::atomic<uint64_t> index_misses{0};
};

constexpr size_t kMaxCachedDiskArtifactContexts = 16;
constexpr int kPersistentIndexCacheSchemaVersion = 1;

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

bool is_cache_relevant_file(std::string_view filename) {
  return filename == "artifact_descriptor.json" || filename == "tensor_index.json" || filename == "tensor_index.cbor" ||
      filename == "tensor.data" || filename.starts_with("tensor.data_") || filename.ends_with(".safetensors");
}

absl::StatusOr<std::string> artifact_directory_cache_fingerprint(const std::filesystem::path& artifact_path) {
  std::vector<std::string> entries;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(artifact_path, ec)) {
    if (ec) {
      return absl::ErrnoToStatus(
          ec.value(), absl::StrCat("Failed to enumerate artifact directory '", artifact_path.string(), "'"));
    }
    struct stat st{};
    if (::stat(entry.path().c_str(), &st) != 0) {
      return absl::ErrnoToStatus(errno, absl::StrCat("Failed to stat '", entry.path().string(), "'"));
    }
    if (!S_ISREG(st.st_mode)) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (!is_cache_relevant_file(filename)) {
      continue;
    }
#if defined(__linux__)
    const int64_t mtime_ns =
        static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000LL + static_cast<int64_t>(st.st_mtim.tv_nsec);
#else
    const int64_t mtime_ns = static_cast<int64_t>(st.st_mtime) * 1000000000LL;
#endif
    entries.push_back(
        absl::StrCat(
            filename,
            ":inode=",
            static_cast<uint64_t>(st.st_ino),
            ":size=",
            static_cast<uint64_t>(st.st_size),
            ":mtime_ns=",
            mtime_ns));
  }
  std::ranges::sort(entries);
  return absl::StrJoin(entries, "|");
}

std::vector<std::shared_ptr<const DiskArtifactContext>> evict_old_contexts_locked(CacheState& state)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(state.mutex) {
  std::vector<std::shared_ptr<const DiskArtifactContext>> evicted;
  while (state.contexts.size() > kMaxCachedDiskArtifactContexts) {
    auto victim = state.contexts.end();
    for (auto it = state.contexts.begin(); it != state.contexts.end(); ++it) {
      if (victim == state.contexts.end() || it->second.last_access < victim->second.last_access) {
        victim = it;
      }
    }
    if (victim == state.contexts.end()) {
      break;
    }
    evicted.push_back(std::move(victim->second.context));
    state.contexts.erase(victim);
  }
  return evicted;
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

std::string sha256_hex(std::string_view payload) {
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));
}

std::filesystem::path persistent_index_cache_dir(const std::filesystem::path& artifact_path) {
  return artifact_path / ".tensorcast" / "metadata_cache";
}

std::string persistent_index_cache_key(
    const std::filesystem::path& artifact_path, std::string_view fingerprint, int target_device_id) {
  return sha256_hex(absl::StrCat(
      "tensorcast.disk_artifact_context.index.v1\n",
      "path=",
      artifact_path.string(),
      "\nfp=",
      fingerprint,
      "\ntarget=",
      target_device_id));
}

std::filesystem::path persistent_index_cache_path(
    const std::filesystem::path& artifact_path, std::string_view fingerprint, int target_device_id) {
  return persistent_index_cache_dir(artifact_path) /
      absl::StrCat(persistent_index_cache_key(artifact_path, fingerprint, target_device_id), ".json");
}

absl::StatusOr<IndexInfo> load_persistent_index_cache(
    const std::filesystem::path& artifact_path, std::string_view fingerprint, int target_device_id) {
  const auto path = persistent_index_cache_path(artifact_path, fingerprint, target_device_id);
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return absl::NotFoundError("persistent disk index cache not found");
  }
  try {
    nlohmann::json j;
    in >> j;
    if (!j.is_object()) {
      return absl::InvalidArgumentError("persistent disk index cache is not an object");
    }
    if (j.value("schema_version", 0) != kPersistentIndexCacheSchemaVersion) {
      return absl::InvalidArgumentError("persistent disk index cache schema mismatch");
    }
    if (j.value("artifact_path", std::string()) != artifact_path.string()) {
      return absl::InvalidArgumentError("persistent disk index cache path mismatch");
    }
    if (j.value("directory_fingerprint", std::string()) != std::string(fingerprint)) {
      return absl::InvalidArgumentError("persistent disk index cache fingerprint mismatch");
    }
    if (j.value("target_device_id", -1) != target_device_id) {
      return absl::InvalidArgumentError("persistent disk index cache target mismatch");
    }
    IndexInfo info;
    info.is_safetensors = j.value("is_safetensors", false);
    info.total_size_bytes = j.value("total_size_bytes", uint64_t{0});
    info.source_total_size_bytes = j.value("source_total_size_bytes", uint64_t{0});
    info.index_multihash = j.value("index_multihash", std::string());
    info.canonical_index_json = j.value("canonical_index_json", std::string());
    if (j.contains("source_index_json") && j["source_index_json"].is_string()) {
      info.source_index_json = j["source_index_json"].get<std::string>();
    }
    if (info.canonical_index_json.empty()) {
      return absl::InvalidArgumentError("persistent disk index cache has empty canonical_index_json");
    }
    return info;
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("failed to parse persistent disk index cache: ", e.what()));
  }
}

void maybe_write_persistent_index_cache(
    const std::filesystem::path& artifact_path,
    std::string_view fingerprint,
    int target_device_id,
    const IndexInfo& info) {
  if (!info.is_safetensors || target_device_id != 0 || info.canonical_index_json.empty()) {
    return;
  }
  const auto dir = persistent_index_cache_dir(artifact_path);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    LOG(INFO) << "DiskArtifactContext persistent index cache disabled for " << artifact_path.string()
              << ": create_directories failed: " << ec.message();
    return;
  }
  const auto final_path = persistent_index_cache_path(artifact_path, fingerprint, target_device_id);
  const auto tmp_path = final_path.string() + ".tmp." + std::to_string(::getpid());
  nlohmann::json j;
  j["schema_version"] = kPersistentIndexCacheSchemaVersion;
  j["artifact_path"] = artifact_path.string();
  j["directory_fingerprint"] = std::string(fingerprint);
  j["target_device_id"] = target_device_id;
  j["is_safetensors"] = info.is_safetensors;
  j["total_size_bytes"] = info.total_size_bytes;
  j["source_total_size_bytes"] = info.source_total_size_bytes;
  j["index_multihash"] = info.index_multihash;
  j["canonical_index_json"] = info.canonical_index_json;
  if (info.source_index_json.has_value()) {
    j["source_index_json"] = *info.source_index_json;
  }
  {
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      LOG(INFO) << "DiskArtifactContext persistent index cache disabled for " << artifact_path.string()
                << ": open failed for " << tmp_path;
      return;
    }
    out << j.dump();
    out.flush();
    if (!out.good()) {
      LOG(INFO) << "DiskArtifactContext persistent index cache write failed for " << tmp_path;
      std::filesystem::remove(tmp_path, ec);
      return;
    }
  }
  if (::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
    const int saved_errno = errno;
    LOG(INFO) << "DiskArtifactContext persistent index cache rename failed for " << final_path.string() << ": "
              << strerror(saved_errno);
    std::filesystem::remove(tmp_path, ec);
  }
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

struct StandardPartitionLayout {
  std::vector<size_t> partition_sizes;
  uint64_t total_size{0};
};

absl::StatusOr<std::optional<StandardPartitionLayout>> maybe_reconstruct_checkpoint_partition_layout(
    std::string_view layout_json,
    uint64_t logical_total_size,
    size_t partition_count) {
  CHECK(partition_count > 0);
  if (layout_json.empty()) {
    return absl::InvalidArgumentError("disk artifact index is empty for standard partition artifact");
  }

  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(layout_json, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse disk artifact index for checkpoint-style layout reconstruction: ", e.what()));
  }
  if (!parsed.is_object()) {
    return absl::InvalidArgumentError("disk artifact index must be a JSON object");
  }

  absl::flat_hash_map<uint64_t, uint64_t> max_size_by_offset;
  max_size_by_offset.reserve(parsed.size());
  for (auto it = parsed.begin(); it != parsed.end(); ++it) {
    const auto& entry = it.value();
    if (!entry.is_array() || entry.size() < 2) {
      return absl::InvalidArgumentError(
          absl::StrCat("disk artifact index entry must contain [offset,size,...] for tensor '", it.key(), "'"));
    }
    const uint64_t offset = entry[0].get<uint64_t>();
    const uint64_t size = entry[1].get<uint64_t>();
    auto existing = max_size_by_offset.find(offset);
    if (existing == max_size_by_offset.end()) {
      max_size_by_offset.emplace(offset, size);
    } else if (size > existing->second) {
      existing->second = size;
    }
  }

  if (max_size_by_offset.empty()) {
    return StandardPartitionLayout{
        .partition_sizes = std::vector<size_t>(partition_count, 0),
        .total_size = logical_total_size,
    };
  }

  std::vector<std::pair<uint64_t, uint64_t>> records;
  records.reserve(max_size_by_offset.size());
  for (const auto& [offset, size] : max_size_by_offset) {
    records.emplace_back(offset, size);
  }
  std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) {
      return a.first < b.first;
    }
    return a.second < b.second;
  });

  std::vector<size_t> partition_sizes;
  partition_sizes.reserve(partition_count);
  uint64_t current_partition_start = 0;
  uint64_t current_partition_used = 0;

  for (size_t i = 0; i < records.size(); ++i) {
    const auto [offset, size] = records[i];
    const uint64_t occupied_bytes = (i + 1 < records.size())
        ? (records[i + 1].first - offset)
        : static_cast<uint64_t>(checkpoint::TensorWriter::aligned_size(static_cast<size_t>(size)));

    if (occupied_bytes < size) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "disk artifact index layout is inconsistent: occupied bytes ",
              occupied_bytes,
              " smaller than tensor size ",
              size,
              " at offset ",
              offset));
    }

    if (offset < current_partition_start + current_partition_used) {
      return absl::FailedPreconditionError(
          absl::StrCat("disk artifact index layout overlaps within a partition at offset ", offset));
    }

    const uint64_t gap_before = offset - (current_partition_start + current_partition_used);
    const uint64_t bytes_if_kept = current_partition_used + gap_before + occupied_bytes;
    if (current_partition_used > 0 && bytes_if_kept > checkpoint::kPartitionMaxSize) {
      partition_sizes.push_back(static_cast<size_t>(offset - current_partition_start));
      current_partition_start = offset;
      current_partition_used = occupied_bytes;
      continue;
    }

    current_partition_used = bytes_if_kept;
  }

  if (logical_total_size < current_partition_start) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            "disk artifact logical total size ",
            logical_total_size,
            " is smaller than reconstructed last partition start ",
            current_partition_start));
  }

  partition_sizes.push_back(static_cast<size_t>(logical_total_size - current_partition_start));
  if (partition_sizes.size() != partition_count) {
    return std::nullopt;
  }

  return StandardPartitionLayout{
      .partition_sizes = std::move(partition_sizes),
      .total_size = logical_total_size,
  };
}

absl::StatusOr<std::optional<StandardPartitionLayout>> maybe_resolve_standard_partition_layout(
    const std::filesystem::path& artifact_path,
    const std::vector<size_t>& physical_partition_sizes) {
  CHECK(!physical_partition_sizes.empty());

  auto info_or = read_from_artifact_dir(artifact_path, /*target_device_id=*/0);
  if (!info_or.ok()) {
    if (absl::IsNotFound(info_or.status())) {
      return std::nullopt;
    }
    return info_or.status();
  }

  const IndexInfo& info = *info_or;
  if (info.is_safetensors) {
    return std::nullopt;
  }

  const std::string& layout_json =
      info.source_index_json.has_value() ? *info.source_index_json : info.canonical_index_json;
  const uint64_t logical_total_size = (info.source_index_json.has_value() && info.source_total_size_bytes > 0)
      ? info.source_total_size_bytes
      : info.total_size_bytes;
  const uint64_t physical_total_size =
      std::accumulate(physical_partition_sizes.begin(), physical_partition_sizes.end(), uint64_t{0});

  if (logical_total_size > physical_total_size) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            "disk artifact logical total size ",
            logical_total_size,
            " exceeds physical partition bytes ",
            physical_total_size,
            " for artifact ",
            artifact_path.string()));
  }

  auto checkpoint_layout_or =
      maybe_reconstruct_checkpoint_partition_layout(layout_json, logical_total_size, physical_partition_sizes.size());
  if (!checkpoint_layout_or.ok()) {
    return checkpoint_layout_or.status();
  }
  if (checkpoint_layout_or->has_value()) {
    return checkpoint_layout_or;
  }

  std::vector<size_t> resolved_partition_sizes = physical_partition_sizes;
  if (resolved_partition_sizes.empty()) {
    return StandardPartitionLayout{.partition_sizes = {}, .total_size = logical_total_size};
  }
  if (resolved_partition_sizes.size() == 1) {
    resolved_partition_sizes[0] = static_cast<size_t>(logical_total_size);
    return StandardPartitionLayout{
        .partition_sizes = std::move(resolved_partition_sizes),
        .total_size = logical_total_size,
    };
  }

  const uint64_t prefix_before_last = physical_total_size - resolved_partition_sizes.back();
  if (logical_total_size < prefix_before_last) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            "disk artifact logical total size ",
            logical_total_size,
            " falls before the final partition under numeric concatenation for ",
            artifact_path.string(),
            "; per-part logical boundaries require checkpoint-style inference or explicit metadata"));
  }

  resolved_partition_sizes.back() = static_cast<size_t>(logical_total_size - prefix_before_last);
  return StandardPartitionLayout{
      .partition_sizes = std::move(resolved_partition_sizes),
      .total_size = logical_total_size,
  };
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
  std::optional<IndexInfo> initial_index_info;
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
    std::optional<std::string> directory_fingerprint;
    auto fingerprint_or = artifact_directory_cache_fingerprint(artifact_path);
    if (fingerprint_or.ok()) {
      directory_fingerprint = *fingerprint_or;
      auto cached_index_or = load_persistent_index_cache(artifact_path, *directory_fingerprint, /*target_device_id=*/0);
      if (cached_index_or.ok()) {
        initial_index_info = *cached_index_or;
        LOG(INFO) << "DiskArtifactContext persistent index cache hit path=" << artifact_path.string();
      } else if (!absl::IsNotFound(cached_index_or.status())) {
        LOG(INFO) << "DiskArtifactContext ignored persistent index cache for " << artifact_path.string() << ": "
                  << cached_index_or.status();
      }
    }
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
      if (!initial_index_info.has_value()) {
        auto header_status = validate_safetensors_header_json((*handle_or)->fd(), header_or->header_length);
        if (!header_status.ok()) {
          return header_status;
        }
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

  if (!is_safetensors && !partition_paths.empty()) {
    auto layout_or = maybe_resolve_standard_partition_layout(artifact_path, partition_sizes);
    if (!layout_or.ok()) {
      return layout_or.status();
    }
    if (layout_or->has_value()) {
      const auto& layout = **layout_or;
      for (size_t i = 0; i < partition_sizes.size(); ++i) {
        if (partition_sizes[i] < layout.partition_sizes[i]) {
          return absl::FailedPreconditionError(
              absl::StrCat(
                  "disk artifact partition file is smaller than index-defined logical layout: path=",
                  partition_paths[i].string(),
                  " physical_size=",
                  partition_sizes[i],
                  " logical_size=",
                  layout.partition_sizes[i]));
        }
      }
      partition_sizes = layout.partition_sizes;
      total_size = layout.total_size;
    }
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
      std::move(safetensors_segments),
      std::move(initial_index_info)));
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
    std::vector<SharedSafetensorsSegment> safetensors_segments,
    std::optional<IndexInfo> initial_index_info)
    : artifact_path_(std::move(artifact_path)),
      partition_paths_(std::move(partition_paths)),
      partition_sizes_(std::move(partition_sizes)),
      total_size_(total_size),
      is_safetensors_(is_safetensors),
      descriptor_present_(descriptor_present),
      tensor_index_json_present_(tensor_index_json_present),
      tensor_index_cbor_present_(tensor_index_cbor_present),
      safetensors_segments_(std::move(safetensors_segments)) {
  if (initial_index_info.has_value()) {
    index_cache_.push_back({0, std::move(*initial_index_info)});
  }
}

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
  if (is_safetensors_ && target_device_id == 0) {
    auto fingerprint_or = artifact_directory_cache_fingerprint(artifact_path_);
    if (fingerprint_or.ok()) {
      maybe_write_persistent_index_cache(artifact_path_, *fingerprint_or, target_device_id, *info_or);
    } else {
      LOG(INFO) << "DiskArtifactContext skipped persistent index cache write for " << artifact_path_.string() << ": "
                << fingerprint_or.status();
    }
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
  auto fingerprint_or = artifact_directory_cache_fingerprint(normalized);
  if (!fingerprint_or.ok()) {
    return fingerprint_or.status();
  }
  const std::string cache_key = absl::StrCat(normalized.string(), "\n", *fingerprint_or);
  auto& state = cache_state();
  {
    absl::MutexLock lock(&state.mutex);
    auto it = state.contexts.find(cache_key);
    if (it != state.contexts.end()) {
      it->second.last_access = state.access_counter.fetch_add(1, std::memory_order_relaxed) + 1;
      LOG(INFO) << "DiskArtifactContext cache hit path=" << normalized.string();
      state.context_hits.fetch_add(1, std::memory_order_relaxed);
      return it->second.context;
    }
  }

  auto built_or = build_disk_artifact_context(normalized);
  if (!built_or.ok()) {
    return built_or.status();
  }
  LOG(INFO) << "DiskArtifactContext cache miss path=" << normalized.string();

  std::vector<std::shared_ptr<const DiskArtifactContext>> evicted_contexts;
  {
    absl::MutexLock lock(&state.mutex);
    auto it = state.contexts.find(cache_key);
    if (it != state.contexts.end()) {
      it->second.last_access = state.access_counter.fetch_add(1, std::memory_order_relaxed) + 1;
      state.context_hits.fetch_add(1, std::memory_order_relaxed);
      return it->second.context;
    }
    state.contexts[cache_key] = CacheState::ContextEntry{
        .context = *built_or, .last_access = state.access_counter.fetch_add(1, std::memory_order_relaxed) + 1};
    evicted_contexts = evict_old_contexts_locked(state);
  }
  // Release retired mmap handles outside the cache mutex.
  evicted_contexts.clear();
  state.context_misses.fetch_add(1, std::memory_order_relaxed);
  return *built_or;
}

absl::StatusOr<std::shared_ptr<const DiskArtifactContext>> get_uncached_disk_artifact_context(
    const std::filesystem::path& artifact_path) {
  return build_disk_artifact_context(normalize_artifact_path(artifact_path));
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
  state.access_counter.store(0, std::memory_order_relaxed);
  state.context_hits.store(0, std::memory_order_relaxed);
  state.context_misses.store(0, std::memory_order_relaxed);
  state.index_hits.store(0, std::memory_order_relaxed);
  state.index_misses.store(0, std::memory_order_relaxed);
}

} // namespace tensorcast::store::loader
