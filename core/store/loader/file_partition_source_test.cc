// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "core/store/loader/file_partition_source.h"

using namespace tensorcast::store::loader;
namespace fs = std::filesystem;

class TempFileFixture {
 public:
  TempFileFixture() {
    temp_dir_ = fs::temp_directory_path() / ("test_file_partition_" + std::to_string(getpid()));
    fs::create_directories(temp_dir_);
  }

  ~TempFileFixture() {
    fs::remove_all(temp_dir_);
  }

  fs::path create_file(const std::string& name, size_t size, char fill_char = 'A') {
    fs::path file_path = temp_dir_ / name;
    std::ofstream file(file_path, std::ios::binary);

    std::vector<char> data(size, fill_char);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<char>(fill_char + (i % 26));
    }

    file.write(data.data(), size);
    file.close();

    return file_path;
  }

  fs::path create_aligned_file(const std::string& name, size_t size, size_t alignment = 512) {
    // Create file with size aligned to specified boundary
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    return create_file(name, aligned_size);
  }

  [[nodiscard]] const fs::path& temp_dir() const {
    return temp_dir_;
  }

 private:
  fs::path temp_dir_;
};

TEST_CASE("FilePartitionSource basic functionality", "[file_partition_source]") {
  TempFileFixture fixture;

  SECTION("Single partition read") {
    size_t file_size = 10 * 1024; // 10KB
    auto file_path = fixture.create_file("test.bin", file_size);

    FilePartitionSource::Options options;
    options.partition_paths = {file_path};
    options.partition_sizes = {file_size};
    options.total_size = file_size;
    options.io_batch_bytes = 1024;
    options.use_direct_io = false; // Disable for simplicity

    FilePartitionSource source(options);

    // Read entire file
    std::vector<char> buffer(file_size);
    auto result = source.read(buffer.data(), file_size);

    REQUIRE(result.ok());
    REQUIRE(result.value() == file_size);

    // Verify content
    for (size_t i = 0; i < file_size; ++i) {
      REQUIRE(buffer[i] == static_cast<char>('A' + (i % 26)));
    }
  }

  SECTION("Multiple partitions sequential read") {
    size_t part1_size = 5 * 1024;
    size_t part2_size = 3 * 1024;
    size_t part3_size = 2 * 1024;

    auto file1 = fixture.create_file("part1.bin", part1_size, 'A');
    auto file2 = fixture.create_file("part2.bin", part2_size, 'B');
    auto file3 = fixture.create_file("part3.bin", part3_size, 'C');

    FilePartitionSource::Options options;
    options.partition_paths = {file1, file2, file3};
    options.partition_sizes = {part1_size, part2_size, part3_size};
    options.total_size = part1_size + part2_size + part3_size;
    options.io_batch_bytes = 1024;
    options.use_direct_io = false;

    FilePartitionSource source(options);

    // Read all data
    std::vector<char> buffer(options.total_size);
    size_t total_read = 0;

    while (total_read < options.total_size) {
      auto result = source.read(buffer.data() + total_read, 1024);
      REQUIRE(result.ok());
      if (result.value() == 0)
        break; // EOF
      total_read += result.value();
    }

    REQUIRE(total_read == options.total_size);

    // Verify each partition's content
    for (size_t i = 0; i < part1_size; ++i) {
      REQUIRE(buffer[i] == static_cast<char>('A' + (i % 26)));
    }
    for (size_t i = 0; i < part2_size; ++i) {
      REQUIRE(buffer[part1_size + i] == static_cast<char>('B' + (i % 26)));
    }
    for (size_t i = 0; i < part3_size; ++i) {
      REQUIRE(buffer[part1_size + part2_size + i] == static_cast<char>('C' + (i % 26)));
    }
  }

  SECTION("Read at specific offset") {
    size_t file_size = 10 * 1024;
    auto file_path = fixture.create_file("test.bin", file_size);

    FilePartitionSource::Options options;
    options.partition_paths = {file_path};
    options.partition_sizes = {file_size};
    options.total_size = file_size;
    options.use_direct_io = false;

    FilePartitionSource source(options);

    // Read from middle of file
    size_t offset = 5 * 1024;
    size_t read_size = 2 * 1024;
    std::vector<char> buffer(read_size);

    auto result = source.read_at(offset, buffer.data(), read_size);

    REQUIRE(result.ok());
    REQUIRE(result.value() == read_size);

    // Verify content
    for (size_t i = 0; i < read_size; ++i) {
      REQUIRE(buffer[i] == static_cast<char>('A' + ((offset + i) % 26)));
    }
  }

  SECTION("Read across partition boundaries") {
    size_t part1_size = 5 * 1024;
    size_t part2_size = 5 * 1024;

    auto file1 = fixture.create_file("part1.bin", part1_size, 'A');
    auto file2 = fixture.create_file("part2.bin", part2_size, 'B');

    FilePartitionSource::Options options;
    options.partition_paths = {file1, file2};
    options.partition_sizes = {part1_size, part2_size};
    options.total_size = part1_size + part2_size;
    options.use_direct_io = false;

    FilePartitionSource source(options);

    // Read across boundary
    size_t offset = 4 * 1024; // Start 1KB before end of first partition
    size_t read_size = 2 * 1024; // Read 2KB (spans both partitions)
    std::vector<char> buffer(read_size);

    auto result = source.read_at(offset, buffer.data(), read_size);

    REQUIRE(result.ok());
    REQUIRE(result.value() == read_size);

    // Verify content from first partition
    for (size_t i = 0; i < 1024; ++i) {
      REQUIRE(buffer[i] == static_cast<char>('A' + ((offset + i) % 26)));
    }
    // Verify content from second partition
    for (size_t i = 1024; i < read_size; ++i) {
      REQUIRE(buffer[i] == static_cast<char>('B' + ((i - 1024) % 26)));
    }
  }
}

TEST_CASE("FilePartitionSource error handling", "[file_partition_source]") {
  TempFileFixture fixture;

  SECTION("Non-existent file") {
    FilePartitionSource::Options options;
    options.partition_paths = {fixture.temp_dir() / "nonexistent.bin"};
    options.partition_sizes = {1024};
    options.total_size = 1024;

    FilePartitionSource source(options);

    std::vector<char> buffer(1024);
    auto result = source.read(buffer.data(), 1024);

    REQUIRE(!result.ok());
  }

  SECTION("Mismatched sizes") {
    auto file = fixture.create_file("test.bin", 1024);

    FilePartitionSource::Options options;
    options.partition_paths = {file};
    options.partition_sizes = {2048}; // Wrong size
    options.total_size = 2048;

    FilePartitionSource source(options);

    std::vector<char> buffer(2048);
    auto result = source.read(buffer.data(), 2048);

    // Should read only what's available
    if (result.ok()) {
      REQUIRE(result.value() <= 1024);
    }
  }

  SECTION("Empty partition list") {
    FilePartitionSource::Options options;
    options.partition_paths = {};
    options.partition_sizes = {};
    options.total_size = 0;

    FilePartitionSource source(options);

    std::vector<char> buffer(1024);
    auto result = source.read(buffer.data(), 1024);

    REQUIRE(result.ok());
    REQUIRE(result.value() == 0); // EOF
  }

  SECTION("Read beyond file size") {
    size_t file_size = 1024;
    auto file = fixture.create_file("test.bin", file_size);

    FilePartitionSource::Options options;
    options.partition_paths = {file};
    options.partition_sizes = {file_size};
    options.total_size = file_size;

    FilePartitionSource source(options);

    // Try to read at offset beyond file
    std::vector<char> buffer(1024);
    auto result = source.read_at(2048, buffer.data(), 1024);

    REQUIRE(result.ok());
    REQUIRE(result.value() == 0); // EOF
  }
}

TEST_CASE("FilePartitionSource thread safety", "[file_partition_source]") {
  TempFileFixture fixture;

  SECTION("Concurrent reads with read()") {
    size_t file_size = 100 * 1024; // 100KB
    auto file = fixture.create_file("test.bin", file_size);

    FilePartitionSource::Options options;
    options.partition_paths = {file};
    options.partition_sizes = {file_size};
    options.total_size = file_size;
    options.use_direct_io = false;

    FilePartitionSource source(options);

    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<std::vector<char>> buffers(num_threads);
    std::atomic<size_t> total_read{0};

    // Each thread reads sequentially
    for (int i = 0; i < num_threads; ++i) {
      buffers[i].resize(1024);
      threads.emplace_back([&source, &buffers, i, &total_read]() {
        for (int j = 0; j < 10; ++j) {
          auto result = source.read(buffers[i].data(), 1024);
          if (result.ok() && result.value() > 0) {
            total_read += result.value();
          }
        }
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    // All reads should complete successfully
    REQUIRE(total_read.load() > 0);
    REQUIRE(total_read.load() <= file_size);
  }

  SECTION("Concurrent reads with read_at()") {
    size_t file_size = 100 * 1024;
    auto file = fixture.create_file("test.bin", file_size);

    FilePartitionSource::Options options;
    options.partition_paths = {file};
    options.partition_sizes = {file_size};
    options.total_size = file_size;
    options.use_direct_io = false;

    FilePartitionSource source(options);

    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<bool> success(num_threads);

    // Each thread reads from different offsets
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back([&source, &success, i, file_size]() {
        std::vector<char> buffer(1024);
        size_t offset = (i * file_size) / 4;

        auto result = source.read_at(offset, buffer.data(), 1024);
        success[i] = result.ok() && result.value() > 0;

        // Verify content
        if (success[i]) {
          for (size_t j = 0; j < result.value(); ++j) {
            char expected = static_cast<char>('A' + ((offset + j) % 26));
            if (buffer[j] != expected) {
              success[i] = false;
              break;
            }
          }
        }
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    // All reads should succeed
    for (bool s : success) {
      REQUIRE(s);
    }
  }

  SECTION("Mixed read() and read_at() concurrency") {
    size_t file_size = 50 * 1024;
    auto file = fixture.create_file("test.bin", file_size);

    FilePartitionSource::Options options;
    options.partition_paths = {file};
    options.partition_sizes = {file_size};
    options.total_size = file_size;
    options.use_direct_io = false;

    FilePartitionSource source(options);

    std::atomic<bool> all_ok{true};
    std::vector<std::thread> threads;

    // Thread using read()
    threads.emplace_back([&source, &all_ok]() {
      std::vector<char> buffer(1024);
      for (int i = 0; i < 10; ++i) {
        auto result = source.read(buffer.data(), 1024);
        if (!result.ok() || result.value() == 0) {
          all_ok = false;
          break;
        }
      }
    });

    // Threads using read_at()
    for (int i = 0; i < 3; ++i) {
      threads.emplace_back([&source, &all_ok, i]() {
        std::vector<char> buffer(1024);
        size_t offset = i * 10 * 1024;
        auto result = source.read_at(offset, buffer.data(), 1024);
        if (!result.ok() || result.value() == 0) {
          all_ok = false;
          return;
        }
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    REQUIRE(all_ok.load());
  }
}

TEST_CASE("FilePartitionSource Direct I/O", "[file_partition_source]") {
  TempFileFixture fixture;

  SECTION("Direct I/O alignment handling") {
    // Direct I/O requires aligned sizes
    size_t aligned_size = 4 * 512; // 2KB, aligned to 512 bytes
    auto file = fixture.create_aligned_file("test.bin", aligned_size, 512);

    FilePartitionSource::Options options;
    options.partition_paths = {file};
    options.partition_sizes = {aligned_size};
    options.total_size = aligned_size;
    options.io_batch_bytes = 512;
    options.use_direct_io = true; // Request Direct I/O

    FilePartitionSource source(options);

    // Note: Direct I/O may not be supported on all filesystems
    // The implementation should fall back gracefully

    std::vector<char> buffer(aligned_size);
    auto result = source.read(buffer.data(), aligned_size);

    REQUIRE(result.ok());
    // Either we read the full amount, or Direct I/O wasn't supported
    if (result.value() > 0) {
      REQUIRE(result.value() == aligned_size);
    }
  }

  SECTION("Unaligned read with Direct I/O") {
    size_t file_size = 10 * 1024 + 100; // Intentionally unaligned
    auto file = fixture.create_file("test.bin", file_size);

    FilePartitionSource::Options options;
    options.partition_paths = {file};
    options.partition_sizes = {file_size};
    options.total_size = file_size;
    options.use_direct_io = true;

    FilePartitionSource source(options);

    // Should handle unaligned reads gracefully
    std::vector<char> buffer(1234); // Unaligned size
    auto result = source.read(buffer.data(), 1234);

    // Should either succeed or fail gracefully
    if (result.ok()) {
      REQUIRE(result.value() <= 1234);
    }
  }
}
