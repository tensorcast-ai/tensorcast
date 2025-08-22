// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/disk_loader.h"
#include "core/store/loader/multi_safetensors_source.h"
#include "core/store/loader/safetensors_source.h"
#include "core/store/loader/source.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

using stepcast::store::DiskLoader;
using stepcast::store::DiskSource;
using stepcast::store::loader::SeekableSource;

namespace {

std::filesystem::path make_tmp_dir(const std::string& prefix) {
  auto base = std::filesystem::temp_directory_path() / prefix;
  std::filesystem::create_directories(base);
  return base;
}

void write_u64_le(std::ofstream& out, uint64_t v) {
  unsigned char buf[8];
  for (int i = 0; i < 8; ++i) {
    buf[i] = static_cast<unsigned char>((v >> (8 * i)) & 0xFF);
  }
  out.write(reinterpret_cast<const char*>(buf), 8);
}

void create_st_file(
    const std::filesystem::path& path,
    const std::string& header_json,
    const std::vector<unsigned char>& payload) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  REQUIRE(out.is_open());
  write_u64_le(out, static_cast<uint64_t>(header_json.size()));
  out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
  if (!payload.empty()) {
    out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
  }
}

} // namespace

TEST_CASE("DiskLoader detects single .safetensors", "[safetensors]") {
  auto dir = make_tmp_dir("st_disk_single");
  std::vector<unsigned char> payload(64);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<unsigned char>(i);
  }
  auto f = dir / "weights.safetensors";
  create_st_file(f, "{\"t\":{\"dtype\":\"U8\",\"shape\":[64],\"data_offsets\":[0,64]}}", payload);

  DiskLoader dl(DiskSource{.path = dir, .expected_size = std::nullopt});
  auto st = dl.initialize();
  REQUIRE(st.ok());

  auto size_or = dl.get_artifact_size();
  REQUIRE(size_or.ok());
  REQUIRE(*size_or == 64);

  auto src_or = dl.open_source();
  REQUIRE(src_or.ok());
  std::unique_ptr<SeekableSource> src = std::move(*src_or);
  // Ensure DiskLoader selected the single-file Safetensors source
  auto* st_src = dynamic_cast<stepcast::store::loader::SafetensorsSource*>(src.get());
  REQUIRE(st_src != nullptr);

  std::filesystem::remove(f);
  std::filesystem::remove(dir);
}

TEST_CASE("DiskLoader detects multiple .safetensors", "[safetensors]") {
  auto dir = make_tmp_dir("st_disk_multi");
  std::vector<unsigned char> a(16);
  std::vector<unsigned char> b(32);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<unsigned char>(i);
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<unsigned char>(0x80 + (i & 0x7F));
  }
  auto f1 = dir / "part1.safetensors";
  auto f2 = dir / "part2.safetensors";
  create_st_file(f1, "{\"a\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":[0,16]}}", a);
  create_st_file(f2, "{\"b\":{\"dtype\":\"U8\",\"shape\":[32],\"data_offsets\":[0,32]}}", b);

  DiskLoader dl(DiskSource{.path = dir, .expected_size = std::nullopt});
  REQUIRE(dl.initialize().ok());
  auto size_or = dl.get_artifact_size();
  REQUIRE(size_or.ok());
  REQUIRE(*size_or == 48);

  auto src_or = dl.open_source();
  REQUIRE(src_or.ok());
  std::unique_ptr<SeekableSource> src = std::move(*src_or);
  // Ensure DiskLoader selected the multi-file Safetensors source
  auto* ms_src = dynamic_cast<stepcast::store::loader::MultiSafetensorsSource*>(src.get());
  REQUIRE(ms_src != nullptr);

  std::filesystem::remove(f1);
  std::filesystem::remove(f2);
  std::filesystem::remove(dir);
}
