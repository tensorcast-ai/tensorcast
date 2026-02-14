// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/metadata/index_reader.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using tensorcast::store::loader::build_from_safetensors;
using tensorcast::store::loader::canonicalize_from_raw_json;
using tensorcast::store::loader::IndexInfo;
using tensorcast::store::loader::read_from_artifact_dir;

namespace {

fs::path make_temp_dir(const std::string& prefix) {
  fs::path base = fs::temp_directory_path() / prefix;
  fs::create_directories(base);
  return base;
}

void write_text(const fs::path& path, const std::string& content) {
  std::ofstream out(path, std::ios::trunc);
  REQUIRE(out.is_open());
  out << content;
}

std::string canonical_index_sample() {
  // tensorB precedes tensorA in raw form to test ordering logic.
  return R"({
    "tensorB": [16, 8, [2], [1], "torch.float32", 0],
    "tensorA": [0, 16, [4], [1], "torch.float32", 0]
  })";
}

// Helper to write a simple safetensors file with header-only (no payload required for index building).
void create_safetensors_file(
    const fs::path& path,
    const std::string& tensor_name,
    uint64_t size_bytes,
    const std::string& dtype = "U8") {
  // Basic safetensors layout: [header_size (u64 little-endian)][header_json][payload...]
  const std::string header_json =
      nlohmann::json({{tensor_name, {{"dtype", dtype}, {"shape", {size_bytes}}, {"data_offsets", {0, size_bytes}}}}})
          .dump();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  REQUIRE(out.is_open());
  uint64_t header_size = header_json.size();
  for (int i = 0; i < 8; ++i) {
    unsigned char byte = static_cast<unsigned char>((header_size >> (8 * i)) & 0xFF);
    out.put(static_cast<char>(byte));
  }
  out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
  std::vector<char> payload(static_cast<size_t>(size_bytes), '\0');
  if (!payload.empty()) {
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  }
}

} // namespace

TEST_CASE("IndexReader loads canonical index from tensor_index.json", "[index_reader]") {
  fs::path dir = make_temp_dir("index_reader_json");
  write_text(dir / "tensor_index.json", canonical_index_sample());

  auto info_or = read_from_artifact_dir(dir, /*target_device_id=*/0);
  REQUIRE(info_or.ok());
  const IndexInfo& info = info_or.value();

  CHECK_FALSE(info.canonical_index_json.empty());
  CHECK(info.total_size_bytes == 24);
  CHECK_FALSE(info.index_multihash.empty());
  CHECK_FALSE(info.is_safetensors);

  nlohmann::json parsed = nlohmann::json::parse(info.canonical_index_json);
  std::vector<std::string> keys;
  for (auto it = parsed.begin(); it != parsed.end(); ++it) {
    keys.push_back(it.key());
  }
  REQUIRE(keys.size() == 2);
  CHECK(keys[0] == "tensorA");
  CHECK(keys[1] == "tensorB");

  fs::remove_all(dir);
}

TEST_CASE("IndexReader rebuilds canonical JSON ordering from raw bytes", "[index_reader]") {
  const std::string raw = canonical_index_sample();
  auto info_or = canonicalize_from_raw_json(raw, /*target_device_id=*/0);
  REQUIRE(info_or.ok());
  const IndexInfo& info = info_or.value();
  CHECK(info.total_size_bytes == 24);
  // canonicalize_from_raw_json marks safetensors false
  CHECK_FALSE(info.is_safetensors);
  nlohmann::json parsed = nlohmann::json::parse(info.canonical_index_json);
  auto it = parsed.begin();
  REQUIRE(it != parsed.end());
  CHECK(it.key() == "tensorA");
}

TEST_CASE("IndexReader builds canonical index from safetensors files", "[index_reader][safetensors]") {
  fs::path dir = make_temp_dir("index_reader_st");
  create_safetensors_file(dir / "part0.safetensors", "weights", /*size_bytes=*/32);
  create_safetensors_file(dir / "part1.safetensors", "bias", /*size_bytes=*/16);

  auto info_or = read_from_artifact_dir(dir, /*target_device_id=*/0);
  REQUIRE(info_or.ok());
  const IndexInfo& info = info_or.value();
  CHECK(info.is_safetensors);
  CHECK(info.total_size_bytes == 48);
  CHECK_FALSE(info.canonical_index_json.empty());
  REQUIRE(info.source_index_json.has_value());
  CHECK(info.source_total_size_bytes == 48);

  nlohmann::json canonical = nlohmann::json::parse(info.canonical_index_json);
  nlohmann::json source = nlohmann::json::parse(*info.source_index_json);
  CHECK(canonical["bias"][0] == 0);
  CHECK(canonical["weights"][0] == 16);
  CHECK(source["weights"][0] == 0);
  CHECK(source["bias"][0] == 32);

  // Ensure build_from_safetensors works when invoked directly.
  std::vector<fs::path> files{dir / "part0.safetensors", dir / "part1.safetensors"};
  auto st_info_or = build_from_safetensors(files, std::nullopt);
  REQUIRE(st_info_or.ok());
  CHECK(st_info_or->is_safetensors);
  CHECK(st_info_or->total_size_bytes == 48);
  REQUIRE(st_info_or->source_index_json.has_value());
  CHECK(st_info_or->source_total_size_bytes == 48);

  fs::remove_all(dir);
}

TEST_CASE("IndexReader prefers tensor_index.json over safetensors headers", "[index_reader][safetensors]") {
  fs::path dir = make_temp_dir("index_reader_prefers_json");
  write_text(dir / "tensor_index.json", R"({"from_index":[0,8,[1],[1],"torch.uint8",0]})");
  create_safetensors_file(dir / "part0.safetensors", "from_safetensors", /*size_bytes=*/16);

  auto info_or = read_from_artifact_dir(dir, /*target_device_id=*/0);
  REQUIRE(info_or.ok());
  const IndexInfo& info = info_or.value();
  CHECK_FALSE(info.is_safetensors);

  nlohmann::json parsed = nlohmann::json::parse(info.canonical_index_json);
  CHECK(parsed.contains("from_index"));
  CHECK_FALSE(parsed.contains("from_safetensors"));

  fs::remove_all(dir);
}

TEST_CASE("IndexReader supports float8 safetensors dtypes", "[index_reader][safetensors]") {
  fs::path dir = make_temp_dir("index_reader_float8_dtype");
  create_safetensors_file(dir / "e4m3.safetensors", "e4m3", /*size_bytes=*/16, /*dtype=*/"F8_E4M3");
  create_safetensors_file(dir / "e5m2.safetensors", "e5m2", /*size_bytes=*/16, /*dtype=*/"F8_E5M2");

  std::vector<fs::path> files{dir / "e4m3.safetensors", dir / "e5m2.safetensors"};
  auto info_or = build_from_safetensors(files, std::nullopt);

  REQUIRE(info_or.ok());
  REQUIRE(info_or->source_index_json.has_value());
  const auto source = nlohmann::json::parse(*info_or->source_index_json);
  CHECK(source.at("e4m3")[4].get<std::string>() == "torch.float8_e4m3fn");
  CHECK(source.at("e5m2")[4].get<std::string>() == "torch.float8_e5m2");

  fs::remove_all(dir);
}

TEST_CASE("IndexReader reports unsupported safetensors dtype details", "[index_reader][safetensors]") {
  fs::path dir = make_temp_dir("index_reader_unsupported_dtype");
  create_safetensors_file(dir / "weights.safetensors", "weights", /*size_bytes=*/16, /*dtype=*/"F8_UNKNOWN");

  std::vector<fs::path> files{dir / "weights.safetensors"};
  auto info_or = build_from_safetensors(files, std::nullopt);

  REQUIRE_FALSE(info_or.ok());
  const std::string message = std::string(info_or.status().message());
  CHECK(message.find("Unsupported safetensors dtype") != std::string::npos);
  CHECK(message.find("F8_UNKNOWN") != std::string::npos);
  CHECK(message.find("weights") != std::string::npos);
  CHECK(message.find("weights.safetensors") != std::string::npos);
  CHECK(message.find("Supported dtype tokens") != std::string::npos);

  fs::remove_all(dir);
}
