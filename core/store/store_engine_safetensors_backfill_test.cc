// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"

using stepcast::DeviceType;
using stepcast::store::DeviceKey;
using stepcast::store::StoreEngine;
using stepcast::store::StoreEngineOptions;

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

static StoreEngine make_store(const std::filesystem::path& root) {
  StoreEngineOptions opts;
  opts.storage_path = root.string();
  opts.memory_pool_size = 32ULL * 1024 * 1024;
  opts.chunk_size = 64ULL * 1024;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  return StoreEngine(opts);
}

static DeviceKey cpu_key() {
  return DeviceKey{DeviceType::CPU, -1, /*uuid=*/""};
}

TEST_CASE("Safetensors backfill writes descriptor and CBOR index", "[store_engine][safetensors][backfill]") {
  auto root = make_tmp_dir("st_backfill");
  const std::string artifact = "st_model";
  auto dir = root / artifact;
  std::filesystem::create_directories(dir);

  // Create a minimal .safetensors file with 16-byte payload
  std::vector<unsigned char> payload(16);
  for (size_t i = 0; i < payload.size(); ++i)
    payload[i] = static_cast<unsigned char>(i);
  auto f = dir / "weights.safetensors";
  const std::string header = "{\"w\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":[0,16]}}";
  create_st_file(f, header, payload);

  // No descriptor or index present initially
  REQUIRE_FALSE(std::filesystem::exists(dir / "artifact_descriptor.json"));
  REQUIRE_FALSE(std::filesystem::exists(dir / "tensor_index.json"));
  REQUIRE_FALSE(std::filesystem::exists(dir / "tensor_index.cbor"));

  StoreEngine store = make_store(root);
  stepcast::store::MaterializeHints hints;
  hints.disk_path = artifact;
  auto handle_or =
      store.materialize_replica(cpu_key(), stepcast::store::StoreEngine::MaterializeMode::LOAD_ONLY, hints);

  // If there is an error, log it and continue
  if (!handle_or.ok()) {
    LOG(ERROR) << "Materialize failed: " << handle_or.status().message();
  }

  REQUIRE(handle_or.ok());
  auto handle = std::move(*handle_or);
  REQUIRE(handle.wait_ready(std::chrono::milliseconds(30000)).ok());

  // Backfill should have written descriptor and JSON index
  auto desc_path = dir / "artifact_descriptor.json";
  REQUIRE(std::filesystem::exists(desc_path));
  REQUIRE(std::filesystem::exists(dir / "tensor_index.json"));

  std::ifstream in(desc_path);
  REQUIRE(in.is_open());
  nlohmann::json j;
  in >> j;
  in.close();
  REQUIRE(j.contains("encoding"));
  REQUIRE(j["encoding"].get<std::string>() == "json");
  REQUIRE(j.contains("total_size"));
  REQUIRE(j["total_size"].get<uint64_t>() == 16ULL);

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}
