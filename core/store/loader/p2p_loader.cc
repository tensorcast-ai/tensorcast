// Copyright (c) 2025, TensorCast Team.

#include "core/store/loader/p2p_loader.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/store/loader/file_partition_source.h"
#include "core/store/loader/mux_seekable_source.h"
#include "core/store/loader/remote_key_source.h"

namespace tensorcast::store {

P2PLoader::P2PLoader(P2PSource source) : source_(std::move(source)), initialized_(false) {}

absl::Status P2PLoader::initialize() {
  absl::MutexLock lock(&mutex_);

  if (initialized_) {
    return absl::OkStatus();
  }

  // Validate source configuration
  if (!source_.comm_engine) {
    return absl::InvalidArgumentError("Communicator is required");
  }

  if (source_.size_bytes == 0) {
    return absl::InvalidArgumentError("Artifact size must be greater than 0");
  }

  if (source_.memory_keys.empty()) {
    return absl::InvalidArgumentError("Memory keys are required");
  }

  LOG(INFO) << "P2PLoader initialized: size=" << source_.size_bytes << " bytes, keys=" << source_.memory_keys.size();

  initialized_ = true;
  return absl::OkStatus();
}

namespace {
absl::StatusOr<store::loader::FilePartitionSource::Options> build_fallback_disk_source_opts(
    const std::string& dir,
    size_t chunk_size,
    uint64_t expected_total) {
  namespace fs = std::filesystem;
  fs::path artifact_dir_path(dir);
  if (dir.empty() || !fs::exists(artifact_dir_path) || !fs::is_directory(artifact_dir_path)) {
    return absl::NotFoundError("Fallback replica directory not found or not a directory");
  }
  std::vector<fs::path> paths;
  std::vector<size_t> sizes;
  uint64_t total = 0;
  for (const auto& entry : fs::directory_iterator(artifact_dir_path)) {
    if (entry.is_regular_file()) {
      const std::string filename = entry.path().filename().string();
      if (filename.starts_with("tensor.data")) {
        paths.push_back(entry.path());
        size_t sz = fs::file_size(entry.path());
        sizes.push_back(sz);
        total += sz;
      }
    }
  }
  if (paths.empty()) {
    return absl::NotFoundError("No tensor.data partitions in fallback dir");
  }
  std::vector<std::pair<fs::path, size_t>> pair;
  pair.reserve(paths.size());
  for (size_t i = 0; i < paths.size(); ++i)
    pair.emplace_back(paths[i], sizes[i]);
  std::ranges::sort(pair, [](const auto& a, const auto& b) { return a.first.filename() < b.first.filename(); });
  loader::FilePartitionSource::Options opts;
  for (auto& p : pair) {
    opts.partition_paths.push_back(p.first);
    opts.partition_sizes.push_back(p.second);
  }
  opts.total_size = (expected_total > 0) ? expected_total : total;
  opts.chunk_size = chunk_size;
  opts.use_direct_io = (opts.total_size > 5ULL * 1024 * 1024 * 1024);
  return opts;
}
} // namespace

absl::StatusOr<std::unique_ptr<loader::SeekableSource>> P2PLoader::open_source() {
  if (!initialized_) {
    auto st = initialize();
    if (!st.ok()) {
      return st;
    }
  }

  // Construct primary remote source
  loader::RemoteKeySource::Options src_opts{
      .comm_engine = source_.comm_engine,
      .memory_keys = source_.memory_keys,
      .buffer_sizes = source_.buf_sizes,
      .ip = source_.ip,
      .port = source_.port,
      .total_size = source_.size_bytes};
  auto remote_src = std::make_shared<store::loader::RemoteKeySource>(src_opts);

  // Optional disk fallback via configured directory
  if (!source_.fallback_disk_dir.empty()) {
    auto disk_opts_or =
        build_fallback_disk_source_opts(source_.fallback_disk_dir, 128 * 1024 * 1024, source_.size_bytes);
    if (disk_opts_or.ok()) {
      auto file_src_ptr = std::make_shared<store::loader::FilePartitionSource>(*disk_opts_or);
      auto mux = std::make_unique<store::loader::MuxSeekableSource>(remote_src, file_src_ptr);
      return mux;
    }
    LOG(WARNING) << "P2PLoader: fallback dir set but invalid: " << disk_opts_or.status();
  }

  // Default: return remote source (unique_ptr wrapper around shared)
  struct Wrapper : public loader::SeekableSource {
    explicit Wrapper(std::shared_ptr<loader::SeekableSource> inner) : inner_(std::move(inner)) {}

    absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
      return inner_->read(dst, max_bytes);
    }

    absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
      return inner_->read_at(offset, dst, bytes);
    }

    [[nodiscard]] bool supports_direct_write() const override {
      return inner_->supports_direct_write();
    }

    absl::StatusOr<size_t> read_into(uint64_t dest_va_offset, size_t bytes, const DirectWriteToken& token) override {
      return inner_->read_into(dest_va_offset, bytes, token);
    }

   private:
    std::shared_ptr<loader::SeekableSource> inner_;
  };

  return std::make_unique<Wrapper>(remote_src);
}

absl::StatusOr<uint64_t> P2PLoader::get_artifact_size() {
  absl::MutexLock lock(&mutex_);
  if (!initialized_) {
    return absl::FailedPreconditionError("P2PLoader not initialized");
  }
  return source_.size_bytes;
}

} // namespace tensorcast::store
