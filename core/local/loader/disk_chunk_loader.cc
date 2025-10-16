// Copyright (c) 2025, TensorCast Team.

#include "core/local/loader/disk_chunk_loader.h"
#include "core/local/chunk/data_chunk.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <mutex>
#include <string>
#include <unordered_map>

#include "absl/log/log.h"
#include "absl/status/status.h"

namespace tensorcast::local::loader {

// -------------------- BackendFile --------------------

std::unordered_map<std::string, BackendFile::Ptr> BackendFile::index;
std::mutex BackendFile::index_mutex;

BackendFile::BackendFile(std::filesystem::path path) : path_(std::move(path)) {
  fd_ = ::open(path_.c_str(), O_RDONLY);
  if (fd_ < 0) {
    PLOG(ERROR) << "open failed for " << path_;
    return;
  }

  struct stat st{};
  if (::fstat(fd_, &st) != 0) {
    PLOG(ERROR) << "fstat failed for " << path_;
    ::close(fd_);
    fd_ = -1;
    return;
  }
  dev_ = st.st_dev;
  ino_ = st.st_ino;
}

BackendFile::~BackendFile() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

absl::Status BackendFile::read(void* dst, size_t size, off_t offset) const {
  if (fd_ < 0) {
    return absl::FailedPreconditionError("BackendFile not open");
  }
  if (size == 0) {
    return absl::OkStatus();
  }
  ssize_t n = ::pread(fd_, dst, size, offset);
  if (n < 0) {
    int err = errno;
    PLOG(ERROR) << "pread failed for " << path_;
    return absl::ErrnoToStatus(err, "pread failed in BackendFile::read");
  }
  if (static_cast<size_t>(n) != size) {
    LOG(ERROR) << "pread read " << n << " bytes, requested " << size;
    return absl::InternalError("partial read in BackendFile::read");
  }
  return absl::OkStatus();
}

std::string BackendFile::make_key(dev_t dev, ino_t ino) {
  return std::to_string(static_cast<uint64_t>(dev)) + ":" + std::to_string(static_cast<uint64_t>(ino));
}

absl::StatusOr<BackendFile::Ptr> BackendFile::get_or_create(const std::filesystem::path& path) {
  struct stat st{};
  if (::stat(path.c_str(), &st) != 0) {
    int err = errno;
    PLOG(ERROR) << "stat failed for " << path;
    return absl::ErrnoToStatus(err, "stat failed in BackendFile::get_or_create");
  }

  const std::string key = make_key(st.st_dev, st.st_ino);
  std::scoped_lock<std::mutex> lock(index_mutex);
  auto it = index.find(key);
  if (it != index.end()) {
    return it->second;
  }

  Ptr ptr = std::make_shared<BackendFile>(path);
  if (ptr->fd_ < 0) {
    return absl::InternalError("Failed to open file in BackendFile::get_or_create");
  }
  index.emplace(key, ptr);
  return ptr;
}

// -------------------- DiskChunkLoader --------------------

absl::Status DiskChunkLoader::_load() {
  if (chunk_ == nullptr) {
    return absl::FailedPreconditionError("DataChunk pointer is null");
  }

  if (f_path_.empty()) {
    return absl::InvalidArgumentError("DiskChunkLoader requires backing file path");
  }

  if (chunk_->size == 0) {
    return absl::InvalidArgumentError("DataChunk size must be non-zero");
  }

  if (chunk_->cpu_base == nullptr) {
    return absl::FailedPreconditionError("DataChunk CPU base not mapped");
  }

  if (backend_file_ == nullptr) {
    auto bf_or = BackendFile::get_or_create(f_path_);
    if (!bf_or.ok()) {
      return bf_or.status();
    }
    backend_file_ = *bf_or;
  }

  return backend_file_->read(chunk_->cpu_base, chunk_->size, f_offset_);
}

absl::Status DiskChunkLoader::load() {
  return this->_load();
}

std::future<absl::Status> DiskChunkLoader::load_async() {
  return std::async(std::launch::async, [this]() { return this->_load(); });
}

} // namespace tensorcast::local::loader
