// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <sys/types.h>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "core/local/loader/chunk_loader.h"

namespace tensorcast::local::data {

class BackendFile final {
 public:
  using Ptr = std::shared_ptr<BackendFile>;

  // Constructor: open file and record device and inode
  explicit BackendFile(std::filesystem::path path);
  ~BackendFile();

  BackendFile(const BackendFile&) = delete;
  BackendFile& operator=(const BackendFile&) = delete;

  // Read using pread; returns error if bytes read != size
  absl::Status read(void* dst, size_t size, off_t offset) const;

  // Static: return existing instance for the same dev+ino, or create
  static absl::StatusOr<Ptr> get_or_create(const std::filesystem::path& path);

 private:
  static std::string make_key(dev_t dev, ino_t ino);

  std::filesystem::path path_;
  int fd_{-1};
  dev_t dev_{};
  ino_t ino_{};

  // Global index: key is "dev:ino"
  static std::unordered_map<std::string, Ptr> index;
  static std::mutex index_mutex;
};

class DiskChunkLoader : public ChunkLoader {
 public:
  DiskChunkLoader(DataChunk* data_chunk, std::filesystem::path f_path, off_t f_offset)
      : ChunkLoader(data_chunk), f_path_(std::move(f_path)), f_offset_(f_offset) {
    auto bf_or = BackendFile::get_or_create(f_path_);
    if (bf_or.ok()) {
      backend_file_ = *bf_or;
    }
  }

  absl::Status load() override;
  // std::future<absl::Status> load_async() override;

 private:
  std::filesystem::path f_path_;
  off_t f_offset_{0};
  BackendFile::Ptr backend_file_{nullptr};

  // absl::Status load_();
};

} // namespace tensorcast::local::data
