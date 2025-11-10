// Copyright (c) 2025, TensorCast Team.

#include "core/local/chunk/chunk.h"
#include "core/local/chunk/data_chunk.h"
#include "core/local/loader/disk_chunk_loader.h"
#include "core/store/device_types.h"

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
#include <stdexcept>
#include <system_error>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using namespace tensorcast::local::meta;

using namespace tensorcast::local::data;

namespace {

tensorcast::store::DeviceKey MakeCpuDeviceKey() {
  tensorcast::store::DeviceKey key;
  key.type = tensorcast::DeviceType::CPU;
  key.ordinal = -1;
  key.uuid = "cpu";
  return key;
}

struct ChunkWithData {
  std::shared_ptr<Chunk> chunk;
  std::shared_ptr<DataChunk> data_chunk;
};

class DataChunkBuilder {
 public:
  explicit DataChunkBuilder(size_t chunk_size) : chunk_size_(chunk_size), device_key_(MakeCpuDeviceKey()) {}

  ChunkWithData make() const {
    auto chunk = std::make_shared<Chunk>(chunk_size_, /*replica_ptr=*/nullptr);
    chunk->generate_data_chunks({device_key_});
    DataChunk* ptr = chunk->get_data_chunk(device_key_);
    REQUIRE(ptr != nullptr);
    return {chunk, std::shared_ptr<DataChunk>(chunk, ptr)};
  }

  const tensorcast::store::DeviceKey& device_key() const {
    return device_key_;
  }

  size_t chunk_size() const {
    return chunk_size_;
  }

 private:
  size_t chunk_size_;
  // off_t r_offset_;
  tensorcast::store::DeviceKey device_key_;
};

size_t GetPageSize() {
  const long page_size = ::sysconf(_SC_PAGESIZE);
  REQUIRE(page_size > 0);
  return static_cast<size_t>(page_size);
}

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

std::vector<std::shared_ptr<DataChunk>> MakeDataChunks(size_t count, size_t chunk_size) {
  DataChunkBuilder builder(chunk_size);
  std::vector<std::shared_ptr<DataChunk>> chunks;
  chunks.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    auto pair = builder.make();
    chunks.push_back(std::move(pair.data_chunk));
  }
  return chunks;
}

std::vector<DataChunk*> ToRawPtrs(const std::vector<std::shared_ptr<DataChunk>>& chunks) {
  std::vector<DataChunk*> raw_chunks;
  raw_chunks.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    raw_chunks.push_back(chunk.get());
  }
  return raw_chunks;
}

} // namespace

TEST_CASE("ChunkPinLease sets and clears lock state for single lease", "[data_chunk]") {
  const size_t chunk_size = GetPageSize();
  auto chunks = MakeDataChunks(3, chunk_size);

  {
    auto lease = ChunkPinLease(ToRawPtrs(chunks));
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
  const size_t chunk_size = GetPageSize();
  auto chunks = MakeDataChunks(2, chunk_size);

  {
    auto lease_a = ChunkPinLease(ToRawPtrs(chunks));
    for (const auto& chunk : chunks) {
      REQUIRE(chunk->lock_refcnt() == 1);
      REQUIRE(chunk->is_locked());
    }

    {
      auto lease_b = ChunkPinLease(ToRawPtrs(chunks));
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
    DataChunkBuilder builder(page_size + 1);
    REQUIRE_THROWS_AS(builder.make(), std::invalid_argument);
  }

  SECTION("aligned length maps successfully") {
    const off_t file_offset = static_cast<off_t>(page_size);
    const size_t len = page_size * 2;
    REQUIRE(static_cast<size_t>(file_offset) + len <= fixture.data().size());

    DataChunkBuilder builder(len);
    auto chunk_with_data = builder.make();
    auto* data_chunk = chunk_with_data.data_chunk.get();

    REQUIRE(data_chunk->get_base_addr() != nullptr);
    REQUIRE(data_chunk->get_size() == len);

    REQUIRE(::munmap(data_chunk->get_base_addr(), len) == 0);
    // data_chunk->get_base_addr() = nullptr;
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

  DataChunkBuilder builder(len);
  auto chunk_with_data = builder.make();
  auto* chunk = chunk_with_data.data_chunk.get();
  REQUIRE(chunk->get_base_addr() != nullptr);
  REQUIRE(chunk->get_size() == len);
  auto* mapped_bytes = static_cast<std::uint8_t*>(chunk->get_base_addr());
  std::memset(mapped_bytes, 0, len);

  // Register a disk loader (high priority) to read from the file
  auto disk_loader = std::make_shared<DiskChunkLoader>(chunk, fixture.file_path(), file_offset);
  chunk->register_loader(disk_loader, DataChunk::LoaderPriority::High);

  auto status = chunk->load();
  REQUIRE(status.ok());
  REQUIRE(chunk->is_loaded());

  REQUIRE(std::memcmp(mapped_bytes, fixture.data().data() + file_offset, len) == 0);

  auto drop_status = chunk->drop();
  REQUIRE(drop_status.ok());

  std::memset(mapped_bytes, 0, len);

  status = chunk->load();
  REQUIRE(status.ok());
  REQUIRE(chunk->is_loaded());
  REQUIRE(std::memcmp(mapped_bytes, fixture.data().data() + file_offset, len) == 0);

  drop_status = chunk->drop();
  REQUIRE(drop_status.ok());

  REQUIRE(::munmap(chunk->get_base_addr(), len) == 0);
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

  DataChunkBuilder builder(len);
  auto chunk_with_data = builder.make();
  auto* chunk = chunk_with_data.data_chunk.get();
  REQUIRE(chunk->get_base_addr() != nullptr);
  auto* mapped_bytes = static_cast<std::uint8_t*>(chunk->get_base_addr());
  //   std::memset(mapped_bytes, 0, len);

  // Register a bad high-priority loader first (should fail)
  std::filesystem::path bad_path = fixture.file_path();
  bad_path += ".missing";
  auto bad_loader = std::make_shared<DiskChunkLoader>(chunk, bad_path, file_offset);
  chunk->register_loader(bad_loader, DataChunk::LoaderPriority::High);

  // Register a valid low-priority loader
  auto good_loader = std::make_shared<DiskChunkLoader>(chunk, fixture.file_path(), file_offset);
  chunk->register_loader(good_loader, DataChunk::LoaderPriority::Low);

  auto status = chunk->load();
  REQUIRE(status.ok());
  REQUIRE(chunk->is_loaded());
  REQUIRE(std::memcmp(mapped_bytes, fixture.data().data() + file_offset, len) == 0);

  auto drop_status = chunk->drop();
  REQUIRE(drop_status.ok());

  REQUIRE(::munmap(chunk->get_base_addr(), len) == 0);
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

  DataChunkBuilder builder(len);
  auto chunk_with_data = builder.make();
  auto* chunk = chunk_with_data.data_chunk.get();
  REQUIRE(chunk->get_base_addr() != nullptr);
  auto* mapped_bytes = static_cast<std::uint8_t*>(chunk->get_base_addr());
  std::memset(mapped_bytes, 0, len);

  auto loader = std::make_shared<DiskChunkLoader>(chunk, fixture.file_path(), file_offset);
  chunk->register_loader(loader, DataChunk::LoaderPriority::High);

  auto fut = chunk->load_async();
  auto status = fut.get();
  REQUIRE(status.ok());
  REQUIRE(chunk->is_loaded());
  REQUIRE(std::memcmp(mapped_bytes, fixture.data().data() + file_offset, len) == 0);

  auto drop_status = chunk->drop();
  REQUIRE(drop_status.ok());
  REQUIRE(::munmap(chunk->get_base_addr(), len) == 0);
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

  DataChunkBuilder builder_a(len);
  DataChunkBuilder builder_b(len);
  DataChunkBuilder builder_c(len);

  auto awd = builder_a.make();
  auto bwd = builder_b.make();
  auto cwd = builder_c.make();

  auto* a = awd.data_chunk.get();
  auto* b = bwd.data_chunk.get();
  auto* c = cwd.data_chunk.get();

  auto* bytes_a = static_cast<std::uint8_t*>(a->get_base_addr());
  auto* bytes_b = static_cast<std::uint8_t*>(b->get_base_addr());
  auto* bytes_c = static_cast<std::uint8_t*>(c->get_base_addr());
  REQUIRE(bytes_a != nullptr);
  REQUIRE(bytes_b != nullptr);
  REQUIRE(bytes_c != nullptr);
  std::memset(bytes_a, 0, len);
  std::memset(bytes_b, 0, len);
  std::memset(bytes_c, 0, len);

  auto loader_a = std::make_shared<DiskChunkLoader>(a, fixture.file_path(), offset_a);
  auto loader_b = std::make_shared<DiskChunkLoader>(b, fixture.file_path(), offset_b);
  auto loader_c = std::make_shared<DiskChunkLoader>(c, fixture.file_path(), offset_c);
  a->register_loader(loader_a, DataChunk::LoaderPriority::High);
  b->register_loader(loader_b, DataChunk::LoaderPriority::High);
  c->register_loader(loader_c, DataChunk::LoaderPriority::High);

  auto f1 = a->load_async();
  auto f2 = b->load_async();
  auto f3 = c->load_async();

  REQUIRE(f1.get().ok());
  REQUIRE(f2.get().ok());
  REQUIRE(f3.get().ok());

  REQUIRE(std::memcmp(bytes_a, fixture.data().data() + offset_a, len) == 0);
  REQUIRE(std::memcmp(bytes_b, fixture.data().data() + offset_b, len) == 0);
  REQUIRE(std::memcmp(bytes_c, fixture.data().data() + offset_c, len) == 0);

  REQUIRE(a->drop().ok());
  REQUIRE(b->drop().ok());
  REQUIRE(c->drop().ok());

  REQUIRE(::munmap(a->get_base_addr(), len) == 0);
  REQUIRE(::munmap(b->get_base_addr(), len) == 0);
  REQUIRE(::munmap(c->get_base_addr(), len) == 0);
}
