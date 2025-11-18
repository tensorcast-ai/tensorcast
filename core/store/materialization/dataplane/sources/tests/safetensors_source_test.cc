// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/dataplane/sources/safetensors_source.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include "absl/strings/str_format.h"

using tensorcast::store::loader::SafetensorsSource;

namespace {

std::filesystem::path make_temp_file_path(const std::string& prefix) {
  auto base = std::filesystem::temp_directory_path();
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dist;
  auto name = absl::StrFormat("%s-%016x.safetensors", prefix, dist(gen));
  return base / name;
}

void write_u64_le(std::ofstream& out, uint64_t v) {
  unsigned char buf[8];
  for (int i = 0; i < 8; ++i) {
    buf[i] = static_cast<unsigned char>((v >> (8 * i)) & 0xFF);
  }
  out.write(reinterpret_cast<const char*>(buf), 8);
}

std::filesystem::path create_safetensors_file(
    const std::vector<unsigned char>& payload,
    const std::string& header_json = "{}") {
  std::filesystem::path p = make_temp_file_path("st_single");
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  REQUIRE(out.is_open());
  write_u64_le(out, static_cast<uint64_t>(header_json.size()));
  out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
  if (!payload.empty()) {
    out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
  }
  out.close();
  return p;
}

} // namespace

TEST_CASE("SafetensorsSource reads payload correctly", "[safetensors]") {
  std::vector<unsigned char> data(1024);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<unsigned char>(i & 0xFF);
  }
  auto path = create_safetensors_file(data, "{\"x\":{\"dtype\":\"U8\",\"shape\":[1024],\"data_offsets\":[0,1024]}}");

  SafetensorsSource src(path);
  std::vector<unsigned char> buf(256);

  // Read first 256 bytes at offset 0
  auto got0 = src.read_at(0, buf.data(), buf.size());
  REQUIRE(got0.ok());
  REQUIRE(*got0 == buf.size());
  for (size_t i = 0; i < buf.size(); ++i) {
    REQUIRE(buf[i] == data[i]);
  }

  // Read spanning end
  auto got1 = src.read_at(900, buf.data(), buf.size());
  REQUIRE(got1.ok());
  REQUIRE(*got1 == 124);
  for (size_t i = 0; i < static_cast<size_t>(*got1); ++i) {
    REQUIRE(buf[i] == data[900 + i]);
  }

  // EOF past end
  auto got2 = src.read_at(1024, buf.data(), buf.size());
  REQUIRE(got2.ok());
  REQUIRE(*got2 == 0);

  // Cleanup
  std::filesystem::remove(path);
}

TEST_CASE("SafetensorsSource rejects malformed headers", "[safetensors]") {
  // Header says 10 bytes but file contains fewer
  auto p = make_temp_file_path("st_bad");
  {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    REQUIRE(out.is_open());
    write_u64_le(out, 10);
    out.write("{123", 4); // malformed and truncated
  }

  SafetensorsSource src(p);
  unsigned char b;
  auto st = src.read_at(0, &b, 1);
  REQUIRE_FALSE(st.ok());
  std::filesystem::remove(p);
}
