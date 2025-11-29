// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/dataplane/verification/verification_utils.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_verification.h"
#include "core/common/cuda_api.h"
#include "core/store/materialization/dataplane/metadata/safetensors_util.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "gsl/pointers"
#include "nlohmann/json.hpp"

namespace tensorcast::store::loader::verification {
struct VerificationMetadataGuard::Entry {
  explicit Entry(std::string artifact_id_in) : artifact_id(std::move(artifact_id_in)) {}

  std::string artifact_id;
  absl::Mutex mutex;
  int32_t ref_count = 0;
  bool warn_emitted = false;
};

namespace {

struct CachedVerification {
  common::ArtifactVerificationInfo info;
  std::string serialized;
  std::string fingerprint;
  std::filesystem::file_time_type last_write_time{};
  absl::Time cached_at;
};

ABSL_CONST_INIT absl::Mutex g_guard_map_mu(absl::kConstInit);
absl::flat_hash_map<std::string, VerificationMetadataGuard::EntryPtr> g_guard_entries ABSL_GUARDED_BY(g_guard_map_mu);

ABSL_CONST_INIT absl::Mutex g_metadata_cache_mu(absl::kConstInit);
absl::flat_hash_map<std::string, CachedVerification> g_metadata_cache ABSL_GUARDED_BY(g_metadata_cache_mu);
std::atomic<uint64_t> g_temp_file_nonce{0};
constexpr absl::Duration kGuardWarnThreshold = absl::Milliseconds(100);

std::string build_cache_key(std::string_view artifact_id, std::string_view byte_space_id) {
  if (artifact_id.empty()) {
    return absl::StrCat("unknown|", byte_space_id);
  }
  return absl::StrCat(artifact_id, "|", byte_space_id);
}

std::optional<CachedVerification> lookup_cached_metadata(std::string_view artifact_id, std::string_view byte_space_id) {
  absl::MutexLock lock(&g_metadata_cache_mu);
  const std::string key = build_cache_key(artifact_id, byte_space_id);
  auto it = g_metadata_cache.find(key);
  if (it == g_metadata_cache.end()) {
    return std::nullopt;
  }
  return it->second;
}

void store_cached_metadata(
    std::string_view artifact_id,
    std::string_view byte_space_id,
    const CachedVerification& entry) {
  absl::MutexLock lock(&g_metadata_cache_mu);
  const std::string key = build_cache_key(artifact_id, byte_space_id);
  g_metadata_cache[key] = entry;
}

void invalidate_cached_metadata(std::string_view artifact_id, std::string_view byte_space_id) {
  absl::MutexLock lock(&g_metadata_cache_mu);
  const std::string key = build_cache_key(artifact_id, byte_space_id);
  g_metadata_cache.erase(key);
}

std::string fingerprint_payload(std::string_view payload) {
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  std::string hex;
  hex.reserve(digest.size() * 2);
  for (uint8_t byte : digest) {
    absl::StrAppendFormat(&hex, "%02x", byte);
  }
  return hex;
}

absl::StatusOr<CachedVerification> load_metadata_from_disk(const std::filesystem::path& verification_path) {
  std::ifstream vf(verification_path, std::ios::binary);
  if (!vf.is_open()) {
    return absl::PermissionDeniedError(
        absl::StrCat("Failed to open verification metadata at '", verification_path.string(), "'"));
  }

  std::stringstream vbuf;
  vbuf << vf.rdbuf();
  vf.close();
  std::string payload = vbuf.str();
  if (payload.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Verification metadata at '", verification_path.string(), "' is empty"));
  }

  auto ver_or = common::ArtifactVerificationInfo::from_json(payload);
  if (!ver_or.ok()) {
    return ver_or.status();
  }

  CachedVerification entry;
  entry.info = *ver_or;
  entry.serialized = std::move(payload);
  entry.fingerprint = fingerprint_payload(entry.serialized);
  entry.cached_at = absl::Now();
  std::error_code ec;
  entry.last_write_time = std::filesystem::last_write_time(verification_path, ec);
  if (ec) {
    entry.last_write_time = std::filesystem::file_time_type::min();
  }
  return entry;
}

absl::StatusOr<std::optional<std::string>> infer_artifact_id_from_descriptor(
    const std::filesystem::path& artifact_dir) {
  const auto descriptor_path = artifact_dir / "artifact_descriptor.json";
  if (!std::filesystem::exists(descriptor_path)) {
    return std::optional<std::string>{};
  }

  std::ifstream in(descriptor_path);
  if (!in.is_open()) {
    return absl::PermissionDeniedError(absl::StrCat("DESCRIPTOR_NOT_READABLE: cannot open ", descriptor_path.string()));
  }

  try {
    nlohmann::json j;
    in >> j;
    if (!j.contains("artifact_id")) {
      return absl::FailedPreconditionError(
          absl::StrCat("artifact_descriptor.json missing artifact_id at ", descriptor_path.string()));
    }
    if (!j["artifact_id"].is_string()) {
      return absl::InvalidArgumentError(
          absl::StrCat("artifact_descriptor.json artifact_id must be string at ", descriptor_path.string()));
    }
    const std::string artifact_id = j["artifact_id"].get<std::string>();
    if (!absl::StartsWith(artifact_id, "mi2:")) {
      return absl::InvalidArgumentError(absl::StrCat("artifact_id must start with 'mi2:' (got ", artifact_id, ")"));
    }
    return std::optional<std::string>(artifact_id);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse artifact_descriptor.json at ", descriptor_path.string(), ": ", e.what()));
  }
}

std::string build_path_guard_key(const std::filesystem::path& artifact_dir) {
  std::error_code ec;
  const auto canonical_path = std::filesystem::weakly_canonical(artifact_dir, ec);
  if (!ec) {
    return canonical_path.string();
  }
  return artifact_dir.lexically_normal().string();
}

std::filesystem::path build_temp_path(const std::filesystem::path& final_path) {
  const uint64_t nonce = g_temp_file_nonce.fetch_add(1, std::memory_order_relaxed);
  const pid_t pid = getpid();
  std::filesystem::path tmp = final_path;
  tmp += absl::StrFormat(".tmp.%d.%llu", static_cast<int>(pid), static_cast<uint64_t>(nonce));
  return tmp;
}

absl::Status atomic_write_file(const std::filesystem::path& final_path, std::string_view payload) {
  const std::filesystem::path tmp_path = build_temp_path(final_path);
  const std::string tmp_path_str = tmp_path.string();
  const std::string final_path_str = final_path.string();

  int fd = ::open(tmp_path_str.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Failed to open temporary file for verification metadata: ", tmp_path_str));
  }

  const char* data = payload.data();
  size_t remaining = payload.size();
  while (remaining > 0) {
    const ssize_t written = ::write(fd, data, remaining);
    if (written < 0) {
      const int err = errno;
      ::close(fd);
      ::unlink(tmp_path_str.c_str());
      return absl::ErrnoToStatus(err, absl::StrCat("Failed to write verification metadata: ", tmp_path_str));
    }
    data += written;
    remaining -= static_cast<size_t>(written);
  }

  if (::fsync(fd) != 0) {
    const int err = errno;
    ::close(fd);
    ::unlink(tmp_path_str.c_str());
    return absl::ErrnoToStatus(err, absl::StrCat("Failed to fsync verification metadata: ", tmp_path_str));
  }

  if (::close(fd) != 0) {
    const int err = errno;
    ::unlink(tmp_path_str.c_str());
    return absl::ErrnoToStatus(err, absl::StrCat("Failed to close verification metadata temp file: ", tmp_path_str));
  }

  if (::rename(tmp_path_str.c_str(), final_path_str.c_str()) != 0) {
    const int err = errno;
    ::unlink(tmp_path_str.c_str());
    return absl::ErrnoToStatus(
        err, absl::StrCat("Failed to atomically rename verification metadata to ", final_path_str));
  }

  const std::filesystem::path parent_dir = final_path.parent_path();
  if (!parent_dir.empty()) {
    const std::string parent_dir_str = parent_dir.string();
    int dir_fd = ::open(parent_dir_str.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir_fd < 0) {
      return absl::ErrnoToStatus(errno, absl::StrCat("Failed to open parent directory for fsync: ", parent_dir_str));
    }
    if (::fsync(dir_fd) != 0) {
      const int err = errno;
      ::close(dir_fd);
      return absl::ErrnoToStatus(err, absl::StrCat("Failed to fsync parent directory: ", parent_dir_str));
    }
    ::close(dir_fd);
  }

  return absl::OkStatus();
}

std::string sanitize_view_id_for_filename(std::string_view view_id) {
  if (view_id.empty()) {
    return std::string("view");
  }
  std::string sanitized;
  sanitized.reserve(view_id.size());
  for (char ch : view_id) {
    const unsigned char u = static_cast<unsigned char>(ch);
    if ((u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') || (u >= '0' && u <= '9') || ch == '_' || ch == '-' ||
        ch == '.') {
      sanitized.push_back(ch);
    } else {
      sanitized.push_back('_');
    }
  }
  constexpr size_t kMaxSuffixLength = 160;
  if (sanitized.size() > kMaxSuffixLength) {
    sanitized.erase(0, sanitized.size() - kMaxSuffixLength);
  }
  return sanitized;
}

std::filesystem::path verification_path_for_view(
    const std::filesystem::path& artifact_path,
    std::string_view byte_space_id) {
  if (byte_space_id.empty()) {
    return artifact_path / "verification.json";
  }
  return artifact_path / absl::StrCat("verification.view_", sanitize_view_id_for_filename(byte_space_id), ".json");
}

std::vector<void*> build_pointer_vector(const MemoryView& mem) {
  if (mem.base_ptr == nullptr || mem.size_bytes == 0) {
    return {};
  }
  return {mem.base_ptr};
}

std::vector<size_t> build_size_vector(const MemoryView& mem) {
  if (mem.base_ptr == nullptr || mem.size_bytes == 0) {
    return {};
  }
  return {static_cast<size_t>(mem.size_bytes)};
}

bool should_skip_gpu_hash(const MemoryView& mem) {
  return mem.location != common::memory::MemoryLocation::GPU || mem.gpu_device_id == std::nullopt ||
      mem.base_ptr == nullptr || mem.size_bytes == 0;
}

bool should_skip_cpu_hash(const MemoryView& mem) {
  return mem.location != common::memory::MemoryLocation::CPU || mem.base_ptr == nullptr || mem.size_bytes == 0;
}

absl::Status persist_verification_json(const std::filesystem::path& verification_path, std::string_view payload) {
  return atomic_write_file(verification_path, payload);
}

absl::StatusOr<std::vector<std::vector<uint8_t>>> compute_leaf_digests_from_cpu(
    const MemoryView& mem,
    absl::Span<const uint64_t> leaf_indices,
    size_t chunk_bytes) {
  const auto* base = static_cast<const uint8_t*>(mem.base_ptr);
  if (base == nullptr) {
    return absl::FailedPreconditionError("CPU pointer unavailable for canonical leaf digests");
  }
  std::vector<std::vector<uint8_t>> digests;
  digests.reserve(leaf_indices.size());
  for (uint64_t idx : leaf_indices) {
    const uint64_t offset = idx * chunk_bytes;
    if (offset + chunk_bytes > mem.size_bytes) {
      continue;
    }
    digests.push_back(common::sha256_digest_bytes(absl::Span<const uint8_t>(base + offset, chunk_bytes)));
  }
  return digests;
}

absl::StatusOr<std::vector<std::vector<uint8_t>>> compute_leaf_digests_from_gpu(
    const MemoryView& mem,
    absl::Span<const uint64_t> leaf_indices,
    size_t chunk_bytes) {
  if (mem.gpu_device_id == std::nullopt) {
    return absl::FailedPreconditionError("GPU device id unavailable for canonical leaf digests");
  }
  if (mem.base_ptr == nullptr) {
    return absl::FailedPreconditionError("GPU pointer unavailable for canonical leaf digests");
  }
  std::vector<uint8_t> buffer(chunk_bytes);
  std::vector<std::vector<uint8_t>> digests;
  digests.reserve(leaf_indices.size());
  for (uint64_t idx : leaf_indices) {
    const uint64_t offset = idx * chunk_bytes;
    if (offset + chunk_bytes > mem.size_bytes) {
      continue;
    }
    if (auto st = tensorcast::cuda::set_device(*mem.gpu_device_id); !st.ok()) {
      return st;
    }
    auto copy_status = tensorcast::cuda::memcpy(
        buffer.data(), static_cast<uint8_t*>(mem.base_ptr) + offset, chunk_bytes, cudaMemcpyDeviceToHost);
    if (!copy_status.ok()) {
      return copy_status;
    }
    if (auto sync_status = tensorcast::cuda::device_synchronize(); !sync_status.ok()) {
      return sync_status;
    }
    digests.push_back(common::sha256_digest_bytes(absl::Span<const uint8_t>(buffer.data(), chunk_bytes)));
  }
  return digests;
}

} // namespace

void ClearVerificationMetadataCacheForTesting() {
  absl::MutexLock lock(&g_metadata_cache_mu);
  g_metadata_cache.clear();
}

VerificationMetadataGuard::ScopedLock::ScopedLock(
    std::string artifact_id,
    absl::Duration wait_duration,
    std::shared_ptr<Entry> entry_handle)
    : artifact_id_(std::move(artifact_id)), wait_duration_(wait_duration), entry_handle_(std::move(entry_handle)) {}

VerificationMetadataGuard::ScopedLock::ScopedLock(ScopedLock&& other) noexcept {
  artifact_id_ = std::move(other.artifact_id_);
  wait_duration_ = other.wait_duration_;
  entry_handle_ = std::move(other.entry_handle_);
  other.wait_duration_ = absl::ZeroDuration();
}

VerificationMetadataGuard::ScopedLock& VerificationMetadataGuard::ScopedLock::operator=(ScopedLock&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  Reset();
  artifact_id_ = std::move(other.artifact_id_);
  wait_duration_ = other.wait_duration_;
  entry_handle_ = std::move(other.entry_handle_);
  other.wait_duration_ = absl::ZeroDuration();
  return *this;
}

VerificationMetadataGuard::ScopedLock::~ScopedLock() {
  Reset();
}

void VerificationMetadataGuard::ScopedLock::Reset() {
  if (!entry_handle_) {
    return;
  }

  entry_handle_->mutex.Unlock();
  {
    absl::MutexLock lock(&g_guard_map_mu);
    auto it = g_guard_entries.find(artifact_id_);
    if (it != g_guard_entries.end()) {
      auto& entry_ptr = it->second;
      if (entry_ptr && entry_ptr->ref_count > 0) {
        entry_ptr->ref_count -= 1;
      }
      if (!entry_ptr || entry_ptr->ref_count == 0) {
        g_guard_entries.erase(it);
      }
    }
  }

  entry_handle_.reset();
  artifact_id_.clear();
  wait_duration_ = absl::ZeroDuration();
}

VerificationMetadataGuard::ScopedLock VerificationMetadataGuard::Acquire(std::string artifact_id) {
  return Acquire(std::move(artifact_id), kGuardWarnThreshold);
}

VerificationMetadataGuard::ScopedLock VerificationMetadataGuard::Acquire(
    std::string artifact_id,
    absl::Duration warn_after) {
  if (artifact_id.empty()) {
    artifact_id = "unknown";
  }

  std::shared_ptr<Entry> entry;
  {
    absl::MutexLock lock(&g_guard_map_mu);
    auto& slot = g_guard_entries[artifact_id];
    if (!slot) {
      slot = std::make_shared<Entry>(artifact_id);
    }
    entry = slot;
    entry->ref_count += 1;
  }

  const absl::Time wait_start = absl::Now();
  entry->mutex.Lock();
  const absl::Duration wait_duration = absl::Now() - wait_start;

  if (warn_after > absl::ZeroDuration() && wait_duration > warn_after && !entry->warn_emitted) {
    entry->warn_emitted = true;
    LOG(WARNING) << absl::StrFormat(
        "verification_metadata_guard_wait_exceeded artifact_id=%s wait_ms=%.2f threshold_ms=%.2f",
        entry->artifact_id,
        absl::ToDoubleMilliseconds(wait_duration),
        absl::ToDoubleMilliseconds(warn_after));
  }

  return ScopedLock(std::move(artifact_id), wait_duration, std::move(entry));
}

absl::StatusOr<std::string> compute_data_multihash(const MemoryView& mem) {
  if (mem.size_bytes == 0 || mem.base_ptr == nullptr) {
    return absl::NotFoundError("Memory region unavailable for hashing");
  }
  if (mem.location == common::memory::MemoryLocation::GPU) {
    if (should_skip_gpu_hash(mem)) {
      return absl::NotFoundError("GPU hashing requirements not met");
    }
    auto hash_or = loader::compute_data_multihash_from_gpu_memory(
        gsl::not_null<void*>{mem.base_ptr}, mem.size_bytes, *mem.gpu_device_id);
    if (!hash_or.ok()) {
      LOG(WARNING) << "compute_data_multihash_from_gpu_memory failed: " << hash_or.status();
      return hash_or.status();
    }
    return hash_or.value();
  }

  if (should_skip_cpu_hash(mem)) {
    return absl::NotFoundError("CPU hashing requirements not met");
  }
  auto hash_or =
      loader::compute_data_multihash_from_cpu_memory(gsl::not_null<const void*>{mem.base_ptr}, mem.size_bytes);
  if (!hash_or.ok()) {
    LOG(WARNING) << "compute_data_multihash_from_cpu_memory failed: " << hash_or.status();
    return hash_or.status();
  }
  return hash_or.value();
}

absl::StatusOr<ViewHashResult> compute_view_tree_hash_and_leaves(
    loader::SeekableSource& base_source,
    uint64_t total_size,
    size_t leaf_chunk_bytes) {
  if (total_size == 0) {
    return absl::InvalidArgumentError("total_size must be > 0");
  }
  if (leaf_chunk_bytes == 0) {
    return absl::InvalidArgumentError("leaf_chunk_bytes must be > 0");
  }
  std::vector<std::vector<uint8_t>> leaves;
  leaves.reserve(static_cast<size_t>((total_size + leaf_chunk_bytes - 1) / leaf_chunk_bytes));
  std::vector<uint8_t> buffer(leaf_chunk_bytes);
  uint64_t processed = 0;
  while (processed < total_size) {
    const size_t to_read = static_cast<size_t>(std::min<uint64_t>(leaf_chunk_bytes, total_size - processed));
    auto read_or = base_source.read_at(processed, buffer.data(), to_read);
    if (!read_or.ok()) {
      return read_or.status();
    }
    const size_t got = read_or.value();
    if (got == 0) {
      return absl::DataLossError("short read while computing tree hash");
    }
    leaves.push_back(common::sha256_digest_bytes(absl::Span<const uint8_t>(buffer.data(), got)));
    processed += got;
  }
  ViewHashResult out;
  out.multihash = common::multibase_multihash_sha256(common::compute_tree_hash_root_sha256(leaves));
  out.leaf_digests = std::move(leaves);
  return out;
}

absl::Status reuse_or_generate_verification_json(
    const std::filesystem::path& artifact_dir,
    std::string expected_byte_space_id,
    const MemoryView& mem) {
  auto artifact_id_or = infer_artifact_id_from_descriptor(artifact_dir);
  if (!artifact_id_or.ok()) {
    return artifact_id_or.status();
  }
  std::optional<std::string> artifact_id = *artifact_id_or;
  std::string guard_key = artifact_id.value_or(build_path_guard_key(artifact_dir));
  auto guard = VerificationMetadataGuard::Acquire(guard_key);
  const auto verification_path = verification_path_for_view(artifact_dir, expected_byte_space_id);
  const absl::Time op_start = absl::Now();
  const bool had_existing_file = std::filesystem::exists(verification_path);
  bool persist_required = !had_existing_file;

  if (mem.location == common::memory::MemoryLocation::GPU && mem.gpu_device_id.has_value()) {
    auto set_dev_status = tensorcast::cuda::set_device(*mem.gpu_device_id);
    if (!set_dev_status.ok()) {
      LOG(ERROR) << "Failed to set CUDA device before verification: " << set_dev_status;
      return set_dev_status;
    }
    auto sync_status = tensorcast::cuda::device_synchronize();
    if (!sync_status.ok()) {
      LOG(ERROR) << "device_synchronize failed before verification: " << sync_status;
      return sync_status;
    }
  }

  const std::string cache_scope = artifact_id.value_or(guard_key);
  const std::string artifact_ref = artifact_id.value_or(guard_key);

  CachedVerification metadata_entry;
  bool have_metadata = false;
  bool reused_existing = false;
  bool regenerated_metadata = false;
  bool metadata_verified = false;
  bool cache_hit = false;
  bool need_regeneration = false;
  bool persist_attempted = false;
  bool persist_success = false;
  bool persisted_cached_metadata = false;
  absl::Duration persist_duration = absl::ZeroDuration();
  absl::Status persist_status = absl::OkStatus();

  std::optional<CachedVerification> cached = lookup_cached_metadata(cache_scope, expected_byte_space_id);
  const bool verification_exists = std::filesystem::exists(verification_path);

  if (verification_exists) {
    auto disk_or = load_metadata_from_disk(verification_path);
    if (disk_or.ok()) {
      metadata_entry = *disk_or;
      metadata_entry.cached_at = absl::Now();
      have_metadata = true;
      persist_required = false;
      store_cached_metadata(cache_scope, expected_byte_space_id, metadata_entry);
    } else {
      LOG(WARNING) << "Failed to load verification metadata at '" << verification_path.string()
                   << "': " << disk_or.status();
      need_regeneration = true;
      persist_required = true;
      invalidate_cached_metadata(cache_scope, expected_byte_space_id);
    }
  } else if (cached.has_value()) {
    metadata_entry = *cached;
    metadata_entry.cached_at = absl::Now();
    have_metadata = true;
    cache_hit = true;
    persist_required = true;
  } else {
    need_regeneration = true;
  }

  const std::vector<void*> ptrs = build_pointer_vector(mem);
  const std::vector<size_t> sizes = build_size_vector(mem);

  if (have_metadata) {
    const auto& info = metadata_entry.info;
    auto fmt_hex = [](uint64_t value) { return absl::StrCat("0x", absl::Hex(value, absl::kZeroPad16)); };
    LOG(INFO) << "verification_utils: using cached metadata artifact_size=" << info.artifact_size << " key_values=["
              << fmt_hex(info.key_values[0]) << ", " << fmt_hex(info.key_values[1]) << ", "
              << fmt_hex(info.key_values[2]) << "]";
    const bool byte_space_mismatch =
        ((!expected_byte_space_id.empty() || !info.byte_space_id.empty()) &&
         info.byte_space_id != expected_byte_space_id);
    if (byte_space_mismatch) {
      LOG(WARNING) << "Verification metadata at '" << verification_path.string() << "' recorded for artifact '"
                   << artifact_ref << "' byte_space_id='" << info.byte_space_id << "', expected '"
                   << expected_byte_space_id << "'; regenerating.";
      need_regeneration = true;
      have_metadata = false;
      invalidate_cached_metadata(cache_scope, expected_byte_space_id);
    } else if (!ptrs.empty()) {
      absl::Status verify_status = common::ArtifactVerifier::verify_artifact_data(
          ptrs, sizes, info, common::VerificationLevel::SEGMENT_HASHES, mem.gpu_device_id.value_or(-1));
      if (!verify_status.ok()) {
        LOG(ERROR) << "verification failed for artifact='" << artifact_ref << "' byte_space='" << expected_byte_space_id
                   << "' status=" << verify_status;
        const auto tensor_path = artifact_dir / "tensor.data_0";
        if (std::filesystem::exists(tensor_path)) {
          std::string msg = std::string(verify_status.message());
          std::string marker = "offset ";
          size_t pos = msg.find(marker);
          if (pos != std::string::npos) {
            pos += marker.size();
            size_t end = msg.find_first_not_of("0123456789", pos);
            std::string offset_str = msg.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
            uint64_t offset = 0;
            if (!offset_str.empty()) {
              offset = static_cast<uint64_t>(std::strtoull(offset_str.c_str(), nullptr, 10));
            }
            std::array<unsigned char, 16> host_bytes{};
            std::ifstream tf(tensor_path, std::ios::binary);
            if (tf.is_open()) {
              tf.seekg(static_cast<std::streamoff>(offset));
              tf.read(reinterpret_cast<char*>(host_bytes.data()), host_bytes.size());
              if (tf.gcount() > 0) {
                std::string host_hex;
                host_hex.reserve(static_cast<size_t>(tf.gcount()) * 3);
                for (std::streamsize i = 0; i < tf.gcount(); ++i) {
                  if (i != 0) {
                    host_hex.push_back(' ');
                  }
                  absl::StrAppendFormat(&host_hex, "%02x", host_bytes[static_cast<size_t>(i)]);
                }
                LOG(ERROR) << "Host bytes at offset=" << offset << " (" << tf.gcount() << " bytes): [" << host_hex
                           << "]";
              }
            }
          }
        }
        invalidate_cached_metadata(cache_scope, expected_byte_space_id);
        return absl::DataLossError(
            absl::StrCat("ARTIFACT_ID_MISMATCH: verification failed: ", verify_status.message()));
      }
      metadata_verified = true;
      reused_existing = true;
    } else {
      metadata_verified = true;
      reused_existing = true;
    }
  }

  if (!metadata_verified && need_regeneration && !ptrs.empty()) {
    auto gen_or = common::ArtifactVerifier::generate_verification_info(
        ptrs, sizes, mem.gpu_device_id.value_or(-1), common::VerificationLevel::SEGMENT_HASHES);
    if (!gen_or.ok()) {
      invalidate_cached_metadata(cache_scope, expected_byte_space_id);
      return gen_or.status();
    }
    metadata_entry.info = *gen_or;
    auto fmt_hex = [](uint64_t value) { return absl::StrCat("0x", absl::Hex(value, absl::kZeroPad16)); };
    LOG(INFO) << "verification_utils: regenerated metadata artifact_size=" << metadata_entry.info.artifact_size
              << " key_values=[" << fmt_hex(metadata_entry.info.key_values[0]) << ", "
              << fmt_hex(metadata_entry.info.key_values[1]) << ", " << fmt_hex(metadata_entry.info.key_values[2])
              << "]";
    metadata_entry.info.byte_space_id = expected_byte_space_id;
    metadata_entry.info.refresh_metadata_signature();
    const std::string payload = metadata_entry.info.to_json();
    metadata_entry.serialized = payload;
    metadata_entry.fingerprint = fingerprint_payload(payload);
    metadata_entry.cached_at = absl::Now();
    metadata_entry.last_write_time = std::filesystem::file_time_type::min();
    store_cached_metadata(cache_scope, expected_byte_space_id, metadata_entry);
    persist_attempted = true;
    const absl::Time persist_start = absl::Now();
    persist_status = persist_verification_json(verification_path, payload);
    persist_duration = absl::Now() - persist_start;
    if (persist_status.ok()) {
      persist_success = true;
      persist_required = false;
      std::error_code ec;
      metadata_entry.last_write_time = std::filesystem::last_write_time(verification_path, ec);
      if (ec) {
        metadata_entry.last_write_time = std::filesystem::file_time_type::min();
      }
      metadata_entry.cached_at = absl::Now();
      store_cached_metadata(cache_scope, expected_byte_space_id, metadata_entry);
    }
    metadata_verified = true;
    regenerated_metadata = true;
    need_regeneration = false;
  } else if (!metadata_verified && need_regeneration && ptrs.empty()) {
    VLOG(1) << "Skipping verification metadata regeneration for artifact '" << artifact_ref << "' (byte_space_id='"
            << expected_byte_space_id << "') because no readable memory view was supplied.";
  }

  if (!persist_success && persist_required && (have_metadata || metadata_verified)) {
    metadata_entry.info.refresh_metadata_signature();
    const std::string payload = metadata_entry.info.to_json();
    metadata_entry.serialized = payload;
    metadata_entry.fingerprint = fingerprint_payload(payload);
    persist_attempted = true;
    const absl::Time persist_start = absl::Now();
    persist_status = persist_verification_json(verification_path, payload);
    persist_duration = absl::Now() - persist_start;
    if (persist_status.ok()) {
      persist_success = true;
      persisted_cached_metadata = true;
      persist_required = false;
      std::error_code ec;
      metadata_entry.last_write_time = std::filesystem::last_write_time(verification_path, ec);
      if (ec) {
        metadata_entry.last_write_time = std::filesystem::file_time_type::min();
      }
      metadata_entry.cached_at = absl::Now();
      store_cached_metadata(cache_scope, expected_byte_space_id, metadata_entry);
    }
  }

  const absl::Duration op_duration = absl::Now() - op_start;
  VLOG(1) << absl::StrFormat(
      "verification_metadata_status artifact_id=%s byte_space=%s reused=%s regenerated=%s cache_hit=%s wait_ms=%.2f duration_ms=%.2f write_attempted=%s write_ms=%.2f cached_persist=%s",
      artifact_ref,
      expected_byte_space_id,
      reused_existing ? "true" : "false",
      regenerated_metadata ? "true" : "false",
      cache_hit ? "true" : "false",
      absl::ToDoubleMilliseconds(guard.wait_duration()),
      absl::ToDoubleMilliseconds(op_duration),
      persist_attempted ? "true" : "false",
      persist_attempted ? absl::ToDoubleMilliseconds(persist_duration) : -1.0,
      persisted_cached_metadata ? "true" : "false");

  if (persist_attempted) {
    if (persist_success) {
      LOG(INFO) << absl::StrFormat(
          "verification_metadata_write_succeeded artifact=%s byte_space=%s wait_ms=%.2f write_ms=%.2f cached_persist=%s",
          artifact_ref,
          expected_byte_space_id,
          absl::ToDoubleMilliseconds(guard.wait_duration()),
          absl::ToDoubleMilliseconds(persist_duration),
          persisted_cached_metadata ? "true" : "false");
    } else {
      LOG(ERROR) << absl::StrFormat(
          "verification_metadata_write_failed artifact=%s byte_space=%s wait_ms=%.2f write_ms=%.2f cached_persist=%s error=\"%s\"",
          artifact_ref,
          expected_byte_space_id,
          absl::ToDoubleMilliseconds(guard.wait_duration()),
          absl::ToDoubleMilliseconds(persist_duration),
          persisted_cached_metadata ? "true" : "false",
          persist_status.message());
    }
  }

  return absl::OkStatus();
}

absl::Status write_descriptor_if_absent(
    const std::filesystem::path& artifact_dir,
    std::string_view index_multihash,
    std::string_view data_multihash,
    uint64_t total_size_bytes,
    std::string_view encoding) {
  const auto descriptor_path = artifact_dir / "artifact_descriptor.json";
  if (std::filesystem::exists(descriptor_path)) {
    return absl::OkStatus();
  }

  try {
    nlohmann::json j;
    j["artifact_id"] = std::string("mi2:") + std::string(index_multihash) + ":" + std::string(data_multihash);
    j["index_multihash"] = index_multihash;
    j["data_multihash"] = data_multihash;
    j["schema_version"] = "v3";
    j["encoding"] = encoding;
    j["total_size"] = total_size_bytes;
    nlohmann::json hp;
    hp["chunk_size"] = 4 * 1024 * 1024;
    hp["fanout"] = 2;
    hp["algorithm"] = "sha2-256";
    j["hash_params"] = hp;

    std::ofstream of(descriptor_path);
    if (!of.is_open()) {
      return absl::PermissionDeniedError("DESCRIPTOR_NOT_WRITABLE: cannot write artifact_descriptor.json");
    }
    of << j.dump(2);
  } catch (const std::exception& e) {
    return absl::PermissionDeniedError(std::string("DESCRIPTOR_NOT_WRITABLE: ") + e.what());
  }

  const auto index_json_path = artifact_dir / "tensor_index.json";
  if (!std::filesystem::exists(index_json_path)) {
    std::vector<std::filesystem::path> safetensors_files;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(artifact_dir, ec)) {
      if (ec) {
        LOG(WARNING) << "Failed to enumerate artifact directory '" << artifact_dir.string()
                     << "' while persisting canonical index: " << ec.message();
        break;
      }
      if (!entry.is_regular_file()) {
        continue;
      }
      const std::string name = entry.path().filename().string();
      if (absl::EndsWith(name, ".safetensors")) {
        safetensors_files.push_back(entry.path());
      }
    }
    if (!safetensors_files.empty()) {
      auto index_bytes_or = loader::BuildCanonicalIndexFromSafetensors(safetensors_files);
      if (index_bytes_or.ok()) {
        try {
          std::ofstream oj(index_json_path);
          if (!oj.is_open()) {
            return absl::PermissionDeniedError("DESCRIPTOR_NOT_WRITABLE: cannot write tensor_index.json");
          }
          oj << index_bytes_or.value();
          oj.close();
        } catch (const std::exception& e) {
          return absl::PermissionDeniedError(std::string("DESCRIPTOR_NOT_WRITABLE: ") + e.what());
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<uint8_t>>> compute_canonical_leaf_digests(
    const MemoryView& mem,
    absl::Span<const uint64_t> leaf_indices,
    size_t chunk_bytes) {
  if (leaf_indices.empty()) {
    return std::vector<std::vector<uint8_t>>{};
  }
  if (chunk_bytes == 0) {
    return absl::InvalidArgumentError("chunk_bytes must be > 0");
  }
  if (mem.location == common::memory::MemoryLocation::CPU) {
    return compute_leaf_digests_from_cpu(mem, leaf_indices, chunk_bytes);
  }
  if (mem.location == common::memory::MemoryLocation::GPU) {
    return compute_leaf_digests_from_gpu(mem, leaf_indices, chunk_bytes);
  }
  return absl::InvalidArgumentError("Unsupported memory location for canonical leaf digests");
}

} // namespace tensorcast::store::loader::verification
