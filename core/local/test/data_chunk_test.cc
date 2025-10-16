// Copyright (c) 2025, TensorCast Team.

#include "core/local/chunk/data_chunk.h"
#include "core/local/loader/disk_chunk_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <random>
#include <system_error>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace tensorcast::local::chunk {

namespace {

struct TempFileFixture {
  explicit TempFileFixture(size_t size_bytes) : data_(size_bytes) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);
    std::generate(data_.begin(), data_.end(), [&]() { return static_cast<std::uint8_t>(dist(gen)); });

    auto temp_dir = std::filesystem::temp_directory_path();
    path_ = temp_dir / ("tensorcast-datachunk-test-" + std::to_string(rd()) + ".bin");

    std::ofstream out(path_, std::ios::binary | std::ios::trunc);
    REQUIRE(out.is_open());
    out.write(reinterpret_cast<const char*>(data_.data()), static_cast<std::streamsize>(data_.size()));
    REQUIRE(out.good());
  }

  ~TempFileFixture() {
    if (path_.empty()) {
      return;
    }
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  const std::filesystem::path& file_path() const {
    return path_;
  }

  const std::vector<std::uint8_t>& data() const {
    return data_;
  }

 private:
  std::filesystem::path path_;
  std::vector<std::uint8_t> data_;
};

std::vector<std::shared_ptr<DataChunk>> MakeChunks(size_t count) {
  std::vector<std::shared_ptr<DataChunk>> chunks;
  chunks.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    auto chunk = std::make_shared<DataChunk>();
    chunk->size = 4096;
    chunk->gpu_base = nullptr;
    chunk->cpu_base = nullptr;
    chunk->replica = nullptr;
    chunk->r_offset = 0;
    chunk->in_dram = true;
    chunk->in_gpu = false;
    chunk->preempt_level = 0;
    chunks.push_back(std::move(chunk));
  }
  return chunks;
}

std::vector<std::shared_ptr<DataChunk>> CopyRefs(const std::vector<std::shared_ptr<DataChunk>>& chunks) {
  return chunks;
}

} // namespace

TEST_CASE("ChunkPinLease sets and clears lock state for single lease", "[data_chunk]") {
  auto chunks = MakeChunks(3);

  {
    auto lease = ChunkPinLease::pin_chunks(CopyRefs(chunks));
    for (const auto& chunk : chunks) {
      REQUIRE(chunk->lock_refcnt() == 1);
      REQUIRE(chunk->is_locked());
    }
  }

  for (const auto& chunk : chunks) {
    REQUIRE(chunk->lock_refcnt() == 0);
    REQUIRE_FALSE(chunk->is_locked());
  }
}

TEST_CASE("ChunkPinLease tracks nested leases via refcount", "[data_chunk]") {
  auto chunks = MakeChunks(2);

  {
    auto lease_a = ChunkPinLease::pin_chunks(CopyRefs(chunks));
    for (const auto& chunk : chunks) {
      REQUIRE(chunk->lock_refcnt() == 1);
      REQUIRE(chunk->is_locked());
    }

    {
      auto lease_b = ChunkPinLease::pin_chunks(CopyRefs(chunks));
      for (const auto& chunk : chunks) {
        REQUIRE(chunk->lock_refcnt() == 2);
        REQUIRE(chunk->is_locked());
      }
    }

    for (const auto& chunk : chunks) {
      REQUIRE(chunk->lock_refcnt() == 1);
      REQUIRE(chunk->is_locked());
    }

    (void)lease_a;
  }

  for (const auto& chunk : chunks) {
    REQUIRE(chunk->lock_refcnt() == 0);
    REQUIRE_FALSE(chunk->is_locked());
  }
}

TEST_CASE("DataChunk constructor enforces page alignment", "[data_chunk]") {
  const int64_t page_size_long = ::sysconf(_SC_PAGESIZE);
  REQUIRE(page_size_long > 0);
  const size_t page_size = static_cast<size_t>(page_size_long);

  size_t kFileBytes = page_size * 20;
  TempFileFixture fixture(kFileBytes);

  SECTION("non-aligned length throws") {
    REQUIRE_THROWS_AS(DataChunk(nullptr, /*replica_offset=*/0, page_size + 1), std::invalid_argument);
  }

  SECTION("aligned length maps successfully") {
    const off_t file_offset = static_cast<off_t>(page_size);
    const size_t len = page_size * 2;
    REQUIRE(static_cast<size_t>(file_offset) + len <= fixture.data().size());

    DataChunk chunk(nullptr, /*replica_offset=*/0, len);

    REQUIRE(chunk.cpu_base != nullptr);
    REQUIRE(chunk.size == len);

    REQUIRE(::munmap(chunk.cpu_base, len) == 0);
    chunk.cpu_base = nullptr;
  }
}

TEST_CASE("DataChunk load reads backing file and drop releases pin", "[data_chunk]") {
  const int64_t page_size_long = ::sysconf(_SC_PAGESIZE);
  REQUIRE(page_size_long > 0);
  const size_t page_size = static_cast<size_t>(page_size_long);

  const size_t kFileBytes = page_size * 3;
  TempFileFixture fixture(kFileBytes);

  const off_t file_offset = static_cast<off_t>(page_size);
  const size_t len = page_size;
  REQUIRE(static_cast<size_t>(file_offset) + len <= fixture.data().size());

  DataChunk chunk(nullptr, /*replica_offset=*/0, len);
  REQUIRE(chunk.cpu_base != nullptr);
  REQUIRE(chunk.size == len);
  auto* mapped_bytes = static_cast<std::uint8_t*>(chunk.cpu_base);
  std::memset(mapped_bytes, 0, len);

  // Register a disk loader (high priority) to read from the file
  tensorcast::local::loader::DiskChunkLoader disk_loader(&chunk, fixture.file_path(), file_offset);
  chunk.register_loader(&disk_loader, DataChunk::LoaderPriority::High);

  auto status = chunk.load();
  REQUIRE(status.ok());
  REQUIRE(chunk.in_dram);

  REQUIRE(std::memcmp(mapped_bytes, fixture.data().data() + file_offset, len) == 0);

  auto drop_status = chunk.drop();
  REQUIRE(drop_status.ok());

  std::memset(mapped_bytes, 0, len);

  status = chunk.load();
  REQUIRE(status.ok());
  REQUIRE(chunk.in_dram);
  REQUIRE(std::memcmp(mapped_bytes, fixture.data().data() + file_offset, len) == 0);

  drop_status = chunk.drop();
  REQUIRE(drop_status.ok());

  REQUIRE(::munmap(chunk.cpu_base, len) == 0);
}

TEST_CASE("DataChunk load priority fallback works", "[data_chunk]") {
  const int64_t page_size_long = ::sysconf(_SC_PAGESIZE);
  REQUIRE(page_size_long > 0);
  const size_t page_size = static_cast<size_t>(page_size_long);

  const size_t kFileBytes = page_size * 4;
  TempFileFixture fixture(kFileBytes);

  const off_t file_offset = static_cast<off_t>(page_size);
  const size_t len = page_size * 2;
  REQUIRE(static_cast<size_t>(file_offset) + len <= fixture.data().size());

  DataChunk chunk(nullptr, /*replica_offset=*/0, len);
  REQUIRE(chunk.cpu_base != nullptr);
  auto* mapped_bytes = static_cast<std::uint8_t*>(chunk.cpu_base);
  //   std::memset(mapped_bytes, 0, len);

  // Register a bad high-priority loader first (should fail)
  std::filesystem::path bad_path = fixture.file_path();
  bad_path += ".missing";
  tensorcast::local::loader::DiskChunkLoader bad_loader(&chunk, bad_path, file_offset);
  chunk.register_loader(&bad_loader, DataChunk::LoaderPriority::High);

  // Register a valid low-priority loader
  tensorcast::local::loader::DiskChunkLoader good_loader(&chunk, fixture.file_path(), file_offset);
  chunk.register_loader(&good_loader, DataChunk::LoaderPriority::Low);

  auto status = chunk.load();
  REQUIRE(status.ok());
  REQUIRE(chunk.in_dram);
  REQUIRE(std::memcmp(mapped_bytes, fixture.data().data() + file_offset, len) == 0);

  auto drop_status = chunk.drop();
  REQUIRE(drop_status.ok());

  REQUIRE(::munmap(chunk.cpu_base, len) == 0);
}

TEST_CASE("DataChunk load_async loads data correctly", "[data_chunk]") {
  const int64_t page_size_long = ::sysconf(_SC_PAGESIZE);
  REQUIRE(page_size_long > 0);
  const size_t page_size = static_cast<size_t>(page_size_long);

  const size_t kFileBytes = page_size * 3;
  TempFileFixture fixture(kFileBytes);

  const off_t file_offset = static_cast<off_t>(page_size);
  const size_t len = page_size;
  REQUIRE(static_cast<size_t>(file_offset) + len <= fixture.data().size());

  DataChunk chunk(nullptr, /*replica_offset=*/0, len);
  REQUIRE(chunk.cpu_base != nullptr);
  auto* mapped_bytes = static_cast<std::uint8_t*>(chunk.cpu_base);
  std::memset(mapped_bytes, 0, len);

  tensorcast::local::loader::DiskChunkLoader loader(&chunk, fixture.file_path(), file_offset);
  chunk.register_loader(&loader, DataChunk::LoaderPriority::High);

  auto fut = chunk.load_async();
  auto status = fut.get();
  REQUIRE(status.ok());
  REQUIRE(chunk.in_dram);
  REQUIRE(std::memcmp(mapped_bytes, fixture.data().data() + file_offset, len) == 0);

  auto drop_status = chunk.drop();
  REQUIRE(drop_status.ok());
  REQUIRE(::munmap(chunk.cpu_base, len) == 0);
}

TEST_CASE("DataChunk load_async supports concurrent loads across chunks", "[data_chunk]") {
  const int64_t page_size_long = ::sysconf(_SC_PAGESIZE);
  REQUIRE(page_size_long > 0);
  const size_t page_size = static_cast<size_t>(page_size_long);

  const size_t kFileBytes = page_size * 6;
  TempFileFixture fixture(kFileBytes);

  const size_t len = page_size;
  const off_t offset_a = static_cast<off_t>(0);
  const off_t offset_b = static_cast<off_t>(page_size * 2);
  const off_t offset_c = static_cast<off_t>(page_size * 4);

  DataChunk a(nullptr, /*replica_offset=*/0, len);
  DataChunk b(nullptr, /*replica_offset=*/0, len);
  DataChunk c(nullptr, /*replica_offset=*/0, len);

  auto* bytes_a = static_cast<std::uint8_t*>(a.cpu_base);
  auto* bytes_b = static_cast<std::uint8_t*>(b.cpu_base);
  auto* bytes_c = static_cast<std::uint8_t*>(c.cpu_base);
  REQUIRE(bytes_a != nullptr);
  REQUIRE(bytes_b != nullptr);
  REQUIRE(bytes_c != nullptr);
  std::memset(bytes_a, 0, len);
  std::memset(bytes_b, 0, len);
  std::memset(bytes_c, 0, len);

  tensorcast::local::loader::DiskChunkLoader loader_a(&a, fixture.file_path(), offset_a);
  tensorcast::local::loader::DiskChunkLoader loader_b(&b, fixture.file_path(), offset_b);
  tensorcast::local::loader::DiskChunkLoader loader_c(&c, fixture.file_path(), offset_c);
  a.register_loader(&loader_a, DataChunk::LoaderPriority::High);
  b.register_loader(&loader_b, DataChunk::LoaderPriority::High);
  c.register_loader(&loader_c, DataChunk::LoaderPriority::High);

  auto f1 = a.load_async();
  auto f2 = b.load_async();
  auto f3 = c.load_async();

  REQUIRE(f1.get().ok());
  REQUIRE(f2.get().ok());
  REQUIRE(f3.get().ok());

  REQUIRE(std::memcmp(bytes_a, fixture.data().data() + offset_a, len) == 0);
  REQUIRE(std::memcmp(bytes_b, fixture.data().data() + offset_b, len) == 0);
  REQUIRE(std::memcmp(bytes_c, fixture.data().data() + offset_c, len) == 0);

  REQUIRE(a.drop().ok());
  REQUIRE(b.drop().ok());
  REQUIRE(c.drop().ok());

  REQUIRE(::munmap(a.cpu_base, len) == 0);
  REQUIRE(::munmap(b.cpu_base, len) == 0);
  REQUIRE(::munmap(c.cpu_base, len) == 0);
}

} // namespace tensorcast::local::chunk
