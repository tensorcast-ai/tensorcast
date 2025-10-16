// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <vector>

#include "core/local/chunk/data_chunk.h" // for DataChunk
#include "core/local/loader/disk_chunk_loader.h" // for BackendFile
#include "core/testing/common.h"

namespace fs = std::filesystem;

using tensorcast::local::chunk::DataChunk;
using tensorcast::local::loader::BackendFile;
using tensorcast::local::loader::DiskChunkLoader;
using tensorcast::testing::create_dummy_file;
using tensorcast::testing::read_file_content;

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

TEST_CASE("DiskChunkLoader load_async reads full file", "[disk_chunk_loader][async]") {
  fs::path tmpdir = fs::temp_directory_path() / "disk_chunk_loader_async_test";
  if (fs::exists(tmpdir)) {
    fs::remove_all(tmpdir);
  }
  fs::create_directories(tmpdir);

  const fs::path file_path = tmpdir / "data.bin";
  const size_t file_size = 4096;
  REQUIRE(create_dummy_file(file_path, file_size, 'K'));

  const auto expected = read_file_content(file_path);
  REQUIRE(expected.size() == file_size);

  std::vector<char> buf(file_size, 0);
  DataChunk chunk;
  chunk.size = file_size;
  chunk.cpu_base = buf.data();

  DiskChunkLoader loader(&chunk, file_path, /*f_offset=*/0);
  auto fut = loader.load_async();
  REQUIRE(fut.valid());
  auto st = fut.get();
  REQUIRE(st.ok());
  REQUIRE(buf == expected);

  fs::remove_all(tmpdir);
}

TEST_CASE("DiskChunkLoader load_async supports offset and repeated calls", "[disk_chunk_loader][async]") {
  fs::path tmpdir = fs::temp_directory_path() / "disk_chunk_loader_async_offset_test";
  if (fs::exists(tmpdir)) {
    fs::remove_all(tmpdir);
  }
  fs::create_directories(tmpdir);

  const fs::path file_path = tmpdir / "data2.bin";
  const size_t file_size = 2048;
  REQUIRE(create_dummy_file(file_path, file_size, 'Z'));

  const auto expected = read_file_content(file_path);
  const size_t half = file_size / 2;

  // First: read second half using offset
  std::vector<char> half_buf(half, 0);
  DataChunk chunk;
  chunk.size = half;
  chunk.cpu_base = half_buf.data();
  {
    DiskChunkLoader loader(&chunk, file_path, /*f_offset=*/static_cast<off_t>(half));
    auto fut = loader.load_async();
    auto st = fut.get();
    REQUIRE(st.ok());
  }
  REQUIRE(std::vector<char>(expected.begin() + half, expected.end()) == half_buf);

  // Second: reuse different buffer with full read via new loader
  std::vector<char> full_buf(file_size, 0);
  chunk.size = file_size;
  chunk.cpu_base = full_buf.data();
  {
    DiskChunkLoader loader(&chunk, file_path, /*f_offset=*/0);
    auto st = loader.load_async().get();
    REQUIRE(st.ok());
  }
  REQUIRE(full_buf == expected);

  fs::remove_all(tmpdir);
}
