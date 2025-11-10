// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <filesystem>
#include <vector>

#include "core/local/chunk/chunk.h"
#include "core/local/chunk/data_chunk.h" // for DataChunk
#include "core/local/loader/disk_chunk_loader.h" // for BackendFile
#include "core/store/device_types.h"
#include "core/testing/common.h"

namespace fs = std::filesystem;

using tensorcast::local::data::BackendFile;
using tensorcast::local::data::DataChunk;
using tensorcast::local::data::DiskChunkLoader;
using tensorcast::testing::create_dummy_file;
using tensorcast::testing::read_file_content;

using tensorcast::local::meta::Chunk;

namespace {

tensorcast::store::DeviceKey MakeCpuDeviceKey() {
  tensorcast::store::DeviceKey key;
  key.type = tensorcast::DeviceType::CPU;
  key.ordinal = -1;
  key.uuid = "cpu";
  return key;
}

struct ChunkWithData {
  std::shared_ptr<tensorcast::local::meta::Chunk> chunk;
  std::shared_ptr<DataChunk> data_chunk;
};

size_t GetPageSize() {
  const long page_size = ::sysconf(_SC_PAGESIZE);
  REQUIRE(page_size > 0);
  return static_cast<size_t>(page_size);
}

class DataChunkBuilder {
 public:
  explicit DataChunkBuilder(size_t chunk_size) : chunk_size_(chunk_size), device_key_(MakeCpuDeviceKey()) {}

  ChunkWithData make() const {
    auto chunk = std::make_shared<tensorcast::local::meta::Chunk>(chunk_size_, /*replica_ptr=*/nullptr);
    chunk->generate_data_chunks({device_key_});
    DataChunk* ptr = chunk->get_data_chunk(device_key_);
    REQUIRE(ptr != nullptr);
    return {chunk, std::shared_ptr<DataChunk>(chunk, ptr)};
  }

 private:
  size_t chunk_size_;
  // off_t r_offset_;
  tensorcast::store::DeviceKey device_key_;
};

void ReleaseChunk(DataChunk* chunk) {
  if (chunk == nullptr) {
    return;
  }
  (void)chunk->drop();
  if (chunk->get_base_addr() != nullptr) {
    ::munmap(chunk->get_base_addr(), chunk->get_size());
    // chunk->get_base_addr() = nullptr;
  }
}

} // namespace

TEST_CASE("BackendFile get_or_create deduplicates by dev+ino and supports pread", "[backendfile]") {
  fs::path tmpdir = fs::temp_directory_path() / "backendfile_test";
  if (fs::exists(tmpdir)) {
    fs::remove_all(tmpdir);
  }
  fs::create_directories(tmpdir);

  const fs::path file_path = tmpdir / "data.bin";
  const size_t file_size = 4096; // 4 KiB
  REQUIRE(create_dummy_file(file_path, file_size, 'A'));

  // Prepare expected content
  const auto content = read_file_content(file_path);
  REQUIRE(content.size() == file_size);

  // First get_or_create
  auto bf_or = BackendFile::get_or_create(file_path);
  REQUIRE(bf_or.ok());
  auto bf1 = *bf_or;
  REQUIRE(bf1 != nullptr);

  // Second get_or_create should return the same instance (by dev+ino key)
  auto bf_or2 = BackendFile::get_or_create(file_path);
  REQUIRE(bf_or2.ok());
  auto bf2 = *bf_or2;
  REQUIRE(bf2 != nullptr);
  REQUIRE(bf1.get() == bf2.get());

  // Read full file using pread
  std::vector<char> buf(file_size);
  REQUIRE(bf1->read(buf.data(), buf.size(), /*offset=*/0).ok());
  REQUIRE(buf == content);

  // Read partial segment with offset
  const size_t half = file_size / 2;
  std::vector<char> buf2(half);
  REQUIRE(bf1->read(buf2.data(), buf2.size(), /*offset=*/half).ok());
  REQUIRE(std::vector<char>(content.begin() + half, content.end()) == buf2);

  // Cleanup
  fs::remove_all(tmpdir);
}

TEST_CASE("BackendFile returns error for missing path", "[backendfile]") {
  fs::path nonexist = fs::temp_directory_path() / "backendfile_test_missing" / "missing.bin";
  auto bf_or = BackendFile::get_or_create(nonexist);
  REQUIRE_FALSE(bf_or.ok());
}

// TEST_CASE("DiskChunkLoader load_async reads full file", "[disk_chunk_loader][async]") {
//   fs::path tmpdir = fs::temp_directory_path() / "disk_chunk_loader_async_test";
//   if (fs::exists(tmpdir)) {
//     fs::remove_all(tmpdir);
//   }
//   fs::create_directories(tmpdir);

//   const fs::path file_path = tmpdir / "data.bin";
//   const size_t file_size = GetPageSize();
//   REQUIRE(create_dummy_file(file_path, file_size, 'K'));

//   const auto expected = read_file_content(file_path);
//   REQUIRE(expected.size() == file_size);

//   DataChunkBuilder builder(file_size);
//   auto chunk_with_data = builder.make();
//   auto* chunk = chunk_with_data.data_chunk.get();
//   auto* buf_ptr = static_cast<char*>(chunk->get_base_addr());
//   REQUIRE(buf_ptr != nullptr);

//   std::memset(buf_ptr, 0, file_size);

//   DiskChunkLoader loader(chunk, file_path, /*f_offset=*/0);
//   auto fut = loader.load_async();
//   REQUIRE(fut.valid());
//   auto st = fut.get();
//   REQUIRE(st.ok());
//   REQUIRE(std::memcmp(buf_ptr, expected.data(), file_size) == 0);

//   ReleaseChunk(chunk);

//   fs::remove_all(tmpdir);
// }

// TEST_CASE("DiskChunkLoader load_async supports offset and repeated calls", "[disk_chunk_loader][async]") {
//   fs::path tmpdir = fs::temp_directory_path() / "disk_chunk_loader_async_offset_test";
//   if (fs::exists(tmpdir)) {
//     fs::remove_all(tmpdir);
//   }
//   fs::create_directories(tmpdir);

//   const fs::path file_path = tmpdir / "data2.bin";
//   const size_t page_size = GetPageSize();
//   const size_t file_size = page_size * 2;
//   REQUIRE(create_dummy_file(file_path, file_size, 'Z'));

//   const auto expected = read_file_content(file_path);
//   const size_t half = page_size;

//   // First: read second half using offset
//   DataChunkBuilder half_builder(half);
//   auto half_chunk_with_data = half_builder.make();
//   auto* half_chunk = half_chunk_with_data.data_chunk.get();
//   REQUIRE(half_chunk->get_base_addr() != nullptr);

//   {
//     DiskChunkLoader loader(half_chunk, file_path, /*f_offset=*/static_cast<off_t>(half));
//     auto fut = loader.load_async();
//     auto st = fut.get();
//     REQUIRE(st.ok());
//   }
//   REQUIRE(std::memcmp(half_chunk->get_base_addr(), expected.data() + half, half) == 0);

//   // Second: reuse different buffer with full read via new loader
//   DataChunkBuilder full_builder(file_size);
//   auto full_chunk_with_data = full_builder.make();
//   auto* full_chunk = full_chunk_with_data.data_chunk.get();
//   REQUIRE(full_chunk->get_base_addr() != nullptr);

//   {
//     DiskChunkLoader loader(full_chunk, file_path, /*f_offset=*/0);
//     auto st = loader.load_async().get();
//     REQUIRE(st.ok());
//   }
//   REQUIRE(std::memcmp(full_chunk->get_base_addr(), expected.data(), file_size) == 0);

//   ReleaseChunk(half_chunk);
//   ReleaseChunk(full_chunk);

//   fs::remove_all(tmpdir);
// }
