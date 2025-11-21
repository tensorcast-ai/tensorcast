// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/dataplane/sources/multi_safetensors_source.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include "absl/strings/str_format.h"

using tensorcast::store::loader::MultiSafetensorsSource;

namespace {

std::filesystem::path make_temp_dir(const std::string& prefix) {
  auto base = std::filesystem::temp_directory_path();
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dist;
  auto name = absl::StrFormat("%s-%016x", prefix, dist(gen));
  auto dir = base / name;
  std::filesystem::create_directories(dir);
  return dir;
}

void write_u64_le(std::ofstream& out, uint64_t v) {
  unsigned char buf[8];
  for (int i = 0; i < 8; ++i) {
    buf[i] = static_cast<unsigned char>((v >> (8 * i)) & 0xFF);
  }
  out.write(reinterpret_cast<const char*>(buf), 8);
}

std::filesystem::path create_safetensors_file(
    const std::filesystem::path& dir,
    const std::string& filename,
    const std::vector<unsigned char>& payload,
    const std::string& header_json = "{}") {
  std::filesystem::path p = dir / filename;
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

TEST_CASE("MultiSafetensorsSource concatenates payloads", "[safetensors]") {
  std::vector<unsigned char> a(128), b(256);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<unsigned char>(i);
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<unsigned char>(0x80 + (i & 0x7F));
  }

  auto dir = make_temp_dir("st_multi_src");
  auto p1 = create_safetensors_file(
      dir, "0001.safetensors", a, "{\"t0\":{\"dtype\":\"U8\",\"shape\":[128],\"data_offsets\":[0,128]}}");
  auto p2 = create_safetensors_file(
      dir, "0002.safetensors", b, "{\"t1\":{\"dtype\":\"U8\",\"shape\":[256],\"data_offsets\":[0,256]}}");

  MultiSafetensorsSource src({p1, p2});
  std::vector<unsigned char> buf(512);

  // Read across boundary: last 16 of a + first 64 of b
  auto got = src.read_at(112, buf.data(), 80);
  REQUIRE(got.ok());
  REQUIRE(*got == 80);
  for (size_t i = 0; i < 16; ++i) {
    REQUIRE(buf[i] == a[112 + i]);
  }
  for (size_t i = 0; i < 64; ++i) {
    REQUIRE(buf[16 + i] == b[i]);
  }

  // EOF beyond total size
  auto eof = src.read_at(128 + 256, buf.data(), buf.size());
  REQUIRE(eof.ok());
  REQUIRE(*eof == 0);

  std::filesystem::remove(p1);
  std::filesystem::remove(p2);
  std::filesystem::remove(dir);
}
