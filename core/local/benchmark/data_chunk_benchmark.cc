// Copyright (c) 2025, TensorCast Team.

#include "core/local/chunk/data_chunk.h"
#include "core/local/loader/disk_chunk_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace {

using tensorcast::local::data::DataChunk;

uint64_t ParsePositiveArg(const char* arg, const char* name) {
  errno = 0;
  char* end = nullptr;
  unsigned long long value = std::strtoull(arg, &end, 10);
  if (arg == end || errno == ERANGE || value == 0ULL) {
    std::cerr << "Invalid " << name << " argument '" << arg << "'. Expected positive integer." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return value;
}

} // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <output_file_path> <file_size_gib> <chunk_size_mib>" << std::endl;
    return EXIT_FAILURE;
  }

  const std::filesystem::path output_path = argv[1];
  const uint64_t size_gib = ParsePositiveArg(argv[2], "file size (GiB)");
  const uint64_t chunk_size_mib = ParsePositiveArg(argv[3], "chunk size (MiB)");

  const long page_size_long = ::sysconf(_SC_PAGESIZE);
  if (page_size_long <= 0) {
    PLOG(ERROR) << "Failed to query system page size";
    return EXIT_FAILURE;
  }
  const uint64_t page_size = static_cast<uint64_t>(page_size_long);

  constexpr uint64_t kGiB = 1ULL << 30;
  constexpr uint64_t kMiB = 1ULL << 20;

  if (size_gib > std::numeric_limits<uint64_t>::max() / kGiB) {
    std::cerr << "file_size_gib is too large" << std::endl;
    return EXIT_FAILURE;
  }
  if (chunk_size_mib > std::numeric_limits<uint64_t>::max() / kMiB) {
    std::cerr << "chunk_size_mib is too large" << std::endl;
    return EXIT_FAILURE;
  }

  uint64_t size_bytes = size_gib * kGiB;
  uint64_t chunk_bytes = chunk_size_mib * kMiB;

  if (chunk_bytes % page_size != 0) {
    const uint64_t aligned_chunk_bytes = ((chunk_bytes + page_size - 1) / page_size) * page_size;
    LOG(INFO) << "Adjusting chunk size from " << chunk_bytes << " to page-aligned " << aligned_chunk_bytes << " bytes";
    chunk_bytes = aligned_chunk_bytes;
  }
  if (chunk_bytes == 0) {
    std::cerr << "Chunk size after alignment is zero" << std::endl;
    return EXIT_FAILURE;
  }

  if (size_bytes % page_size != 0) {
    const uint64_t aligned_size_bytes = ((size_bytes + page_size - 1) / page_size) * page_size;
    LOG(INFO) << "Adjusting file size from " << size_bytes << " to page-aligned " << aligned_size_bytes << " bytes";
    size_bytes = aligned_size_bytes;
  }

  if (size_bytes % chunk_bytes != 0) {
    const uint64_t aligned_size_bytes = ((size_bytes + chunk_bytes - 1) / chunk_bytes) * chunk_bytes;
    LOG(INFO) << "Adjusting file size from " << size_bytes << " to chunk-aligned " << aligned_size_bytes << " bytes";
    size_bytes = aligned_size_bytes;
  }

  const uint64_t chunk_count = size_bytes / chunk_bytes;
  if (chunk_count == 0) {
    std::cerr << "Chunk size exceeds file size" << std::endl;
    return EXIT_FAILURE;
  }

  if (chunk_bytes > std::numeric_limits<size_t>::max()) {
    std::cerr << "Chunk size exceeds size_t range" << std::endl;
    return EXIT_FAILURE;
  }

  // Check mlock limits
  struct rlimit memlock_limit;
  if (getrlimit(RLIMIT_MEMLOCK, &memlock_limit) == 0) {
    const uint64_t memlock_kb = memlock_limit.rlim_cur / 1024;
    const uint64_t total_lock_kb = (size_bytes * chunk_count) / 1024;
    if (total_lock_kb > memlock_kb) {
      std::cerr << "WARNING: Total memory to lock (" << total_lock_kb << " KB) exceeds "
                << "current RLIMIT_MEMLOCK (" << memlock_kb << " KB)." << std::endl;
      std::cerr << "This may cause mlock() to fail. Consider:" << std::endl;
      std::cerr << "  1. Run with sudo" << std::endl;
      std::cerr << "  2. Increase ulimit: sudo ulimit -l unlimited" << std::endl;
      std::cerr << "  3. Reduce chunk size or file size" << std::endl;
      std::cerr << "Continuing anyway..." << std::endl;
    }
  }

  int fd = ::open(output_path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0600);
  if (fd < 0) {
    PLOG(ERROR) << "Failed to create file at " << output_path;
    return EXIT_FAILURE;
  }

  // Fill the file with random content up to size_bytes, rather than creating a sparse/hole file
  std::vector<char> buffer(16 * 1024 * 1024); // 16 MiB buffer
  FILE* rand_file = std::fopen("/dev/urandom", "rb");
  if (rand_file == nullptr) {
    PLOG(ERROR) << "Failed to open /dev/urandom";
    ::close(fd);
    std::filesystem::remove(output_path);
    return EXIT_FAILURE;
  }

  uint64_t written = 0;
  while (written < size_bytes) {
    const size_t to_write = static_cast<size_t>(std::min<uint64_t>(buffer.size(), size_bytes - written));
    const size_t read_bytes = std::fread(buffer.data(), 1, to_write, rand_file);
    if (read_bytes != to_write) {
      PLOG(ERROR) << "Failed to read random data from /dev/urandom";
      std::fclose(rand_file);
      ::close(fd);
      std::filesystem::remove(output_path);
      return EXIT_FAILURE;
    }

    ssize_t rc = ::write(fd, buffer.data(), to_write);
    if (rc <= 0) {
      PLOG(ERROR) << "Failed to write random data to target file";
      std::fclose(rand_file);
      ::close(fd);
      std::filesystem::remove(output_path);
      return EXIT_FAILURE;
    }

    written += static_cast<uint64_t>(rc);
  }

  std::fclose(rand_file);
  if (::fsync(fd) != 0) {
    PLOG(WARNING) << "fsync failed on target file";
  }
  ::close(fd);

  const std::filesystem::path temp_path = output_path;

  if (chunk_count > std::numeric_limits<size_t>::max()) {
    std::cerr << "Chunk count exceeds size_t range" << std::endl;
    return EXIT_FAILURE;
  }

  const size_t chunk_bytes_size_t = static_cast<size_t>(chunk_bytes);
  const size_t chunk_count_size_t = static_cast<size_t>(chunk_count);

  LOG(INFO) << "Benchmarking DataChunk load: file='" << output_path << "' size=" << size_bytes
            << " bytes, chunks=" << chunk_count << " x " << chunk_bytes << " bytes";

  std::vector<std::shared_ptr<DataChunk>> chunks;
  chunks.reserve(chunk_count_size_t);
  std::vector<uint64_t> offsets;
  offsets.reserve(chunk_count_size_t);
  std::vector<std::shared_ptr<tensorcast::local::data::DiskChunkLoader>> loaders;
  loaders.reserve(chunk_count_size_t);

  absl::Status init_status = absl::OkStatus();
  const absl::Time init_start = absl::Now();
  try {
    for (uint64_t i = 0; i < chunk_count; ++i) {
      const uint64_t offset = i * chunk_bytes;
      auto chunk = std::make_shared<DataChunk>(nullptr, /*replica_offset=*/0, chunk_bytes_size_t);
      auto loader = std::make_shared<tensorcast::local::data::DiskChunkLoader>(
          chunk.get(), temp_path, static_cast<off_t>(offset));
      chunk->register_loader(loader, DataChunk::LoaderPriority::High);
      offsets.push_back(offset);
      loaders.push_back(loader);
      chunks.push_back(std::move(chunk));
    }
  } catch (const std::exception& ex) {
    init_status = absl::InternalError(ex.what());
  }
  const absl::Duration init_elapsed = absl::Now() - init_start;
  if (!init_status.ok()) {
    LOG(ERROR) << "Failed to initialize DataChunk vector: " << init_status;
    for (const auto& chunk : chunks) {
      if (chunk && chunk->get_base_addr() != nullptr) {
        ::munmap(chunk->get_base_addr(), chunk->get_size());
      }
    }
    std::filesystem::remove(temp_path);
    return EXIT_FAILURE;
  }

  const absl::Time load_start = absl::Now();
  for (size_t i = 0; i < chunks.size(); ++i) {
    const auto& chunk = chunks[i];
    absl::Status load_status = chunk->load();
    if (!load_status.ok()) {
      LOG(ERROR) << "DataChunk::load failed at chunk offset " << offsets[i] << ": " << load_status;
      for (size_t j = 0; j < chunks.size(); ++j) {
        const auto& prev_chunk = chunks[j];
        if (!prev_chunk) {
          continue;
        }
        absl::Status drop_status = prev_chunk->drop();
        if (!drop_status.ok()) {
          LOG(WARNING) << "drop during cleanup failed at offset " << offsets[j] << ": " << drop_status;
        }
        if (prev_chunk->get_base_addr() != nullptr) {
          ::munmap(prev_chunk->get_base_addr(), prev_chunk->get_size());
        }
      }
      std::filesystem::remove(temp_path);
      return EXIT_FAILURE;
    }
  }
  const absl::Duration load_elapsed = absl::Now() - load_start;

  const absl::Time drop_start = absl::Now();
  for (size_t i = 0; i < chunks.size(); ++i) {
    const auto& chunk = chunks[i];
    absl::Status drop_status = chunk->drop();
    if (!drop_status.ok()) {
      LOG(WARNING) << "DataChunk::drop reported error for offset " << offsets[i] << ": " << drop_status;
    }
  }
  const absl::Duration drop_elapsed = absl::Now() - drop_start;

  for (size_t i = 0; i < chunks.size(); ++i) {
    const auto& chunk = chunks[i];
    if (chunk->get_base_addr() != nullptr && chunk->get_size() > 0) {
      if (::munmap(chunk->get_base_addr(), chunk->get_size()) != 0) {
        PLOG(WARNING) << "munmap failed for chunk at offset " << offsets[i];
      }
      // chunk->get_base_addr() = nullptr;
    }
  }

  std::cout << "Benchmark Summary" << std::endl;
  std::cout << "  file_size_bytes=" << size_bytes << std::endl;
  std::cout << "  chunk_size_bytes=" << chunk_bytes << std::endl;
  std::cout << "  chunk_count=" << chunk_count << std::endl;
  std::cout << "  init_seconds=" << absl::ToDoubleSeconds(init_elapsed) << std::endl;
  std::cout << "  load_seconds=" << absl::ToDoubleSeconds(load_elapsed) << std::endl;
  std::cout << "  drop_seconds=" << absl::ToDoubleSeconds(drop_elapsed) << std::endl;

  std::error_code ec;
  std::filesystem::remove(temp_path, ec);
  if (ec) {
    LOG(WARNING) << "Failed to remove temporary file: " << temp_path << " error: " << ec.message();
  }

  return EXIT_SUCCESS;
}
