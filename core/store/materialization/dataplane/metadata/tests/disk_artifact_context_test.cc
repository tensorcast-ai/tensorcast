// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace {

std::filesystem::path make_temp_dir(const std::string& prefix) {
  auto base = std::filesystem::temp_directory_path();
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dist;
  auto dir = base / (prefix + "-" + std::to_string(dist(gen)));
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

TEST_CASE("DiskArtifactContext reuses cached safetensors scan and index info", "[disk_artifact_context][safetensors]") {
  using tensorcast::store::loader::get_disk_artifact_context;
  using tensorcast::store::loader::get_disk_artifact_context_cache_stats;
  using tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing;

  reset_disk_artifact_context_cache_for_testing();

  auto dir = make_temp_dir("disk-artifact-context");
  const auto file = dir / "weights.safetensors";
  create_st_file(
      file, "{\"t\":{\"dtype\":\"U8\",\"shape\":[64],\"data_offsets\":[0,64]}}", std::vector<unsigned char>(64, 7));

  auto ctx1_or = get_disk_artifact_context(dir);
  REQUIRE(ctx1_or.ok());
  auto ctx2_or = get_disk_artifact_context(dir);
  REQUIRE(ctx2_or.ok());

  REQUIRE(ctx1_or->get() == ctx2_or->get());
  REQUIRE((*ctx1_or)->is_safetensors());
  REQUIRE((*ctx1_or)->safetensors_segments().size() == 1);

  auto stats = get_disk_artifact_context_cache_stats();
  REQUIRE(stats.context_misses == 1);
  REQUIRE(stats.context_hits >= 1);

  auto index1_or = (*ctx1_or)->get_index_info(/*target_device_id=*/0);
  REQUIRE(index1_or.ok());
  auto index2_or = (*ctx2_or)->get_index_info(/*target_device_id=*/0);
  REQUIRE(index2_or.ok());
  REQUIRE(index1_or->canonical_index_json == index2_or->canonical_index_json);

  stats = get_disk_artifact_context_cache_stats();
  REQUIRE(stats.index_misses == 1);
  REQUIRE(stats.index_hits >= 1);

  ctx1_or = absl::StatusOr<std::shared_ptr<const tensorcast::store::loader::DiskArtifactContext>>(
      std::shared_ptr<const tensorcast::store::loader::DiskArtifactContext>());
  ctx2_or = absl::StatusOr<std::shared_ptr<const tensorcast::store::loader::DiskArtifactContext>>(
      std::shared_ptr<const tensorcast::store::loader::DiskArtifactContext>());
  reset_disk_artifact_context_cache_for_testing();
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
