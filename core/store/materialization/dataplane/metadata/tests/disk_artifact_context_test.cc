// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "core/checkpoint/tensor_writer.h"

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

void write_text(const std::filesystem::path& path, const std::string& content) {
  std::ofstream out(path, std::ios::trunc);
  REQUIRE(out.is_open());
  out << content;
}

void create_sparse_file(const std::filesystem::path& path, uint64_t size) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  REQUIRE(out.is_open());
  if (size > 0) {
    out.seekp(static_cast<std::streamoff>(size - 1));
    out.put('\0');
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

TEST_CASE("DiskArtifactContext orders multipart partitions numerically", "[disk_artifact_context][partitioned]") {
  using tensorcast::store::loader::get_disk_artifact_context;
  using tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing;

  reset_disk_artifact_context_cache_for_testing();

  auto dir = make_temp_dir("disk-artifact-context-partitions");
  {
    std::ofstream out(dir / "tensor.data_10", std::ios::binary | std::ios::trunc);
    REQUIRE(out.is_open());
    out << "BBBB";
  }
  {
    std::ofstream out(dir / "tensor.data_2", std::ios::binary | std::ios::trunc);
    REQUIRE(out.is_open());
    out << "AA";
  }

  auto ctx_or = get_disk_artifact_context(dir);
  REQUIRE(ctx_or.ok());
  REQUIRE_FALSE((*ctx_or)->is_safetensors());
  REQUIRE((*ctx_or)->partition_paths().size() == 2);
  CHECK((*ctx_or)->partition_paths()[0].filename() == std::filesystem::path("tensor.data_2"));
  CHECK((*ctx_or)->partition_paths()[1].filename() == std::filesystem::path("tensor.data_10"));

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST_CASE(
    "DiskArtifactContext uses logical standard-partition sizes from tensor_index",
    "[disk_artifact_context][standard]") {
  using tensorcast::store::loader::get_disk_artifact_context;
  using tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing;

  reset_disk_artifact_context_cache_for_testing();

  auto dir = make_temp_dir("disk-artifact-context-standard");
  create_sparse_file(dir / "tensor.data_0", /*size=*/4096);
  write_text(dir / "tensor_index.json", R"({"weights":[0,128,[32],[1],"torch.float32",0]})");

  auto ctx_or = get_disk_artifact_context(dir);
  REQUIRE(ctx_or.ok());

  CHECK((*ctx_or)->partition_sizes().size() == 1);
  CHECK((*ctx_or)->partition_sizes()[0] == 128);
  CHECK((*ctx_or)->total_size() == 128);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST_CASE(
    "DiskArtifactContext orders safetensors lexicographically and ignores nested files",
    "[disk_artifact_context][safetensors][scope]") {
  using tensorcast::store::loader::get_disk_artifact_context;
  using tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing;

  reset_disk_artifact_context_cache_for_testing();

  auto dir = make_temp_dir("disk-artifact-context-safetensors-order");
  std::filesystem::create_directories(dir / "nested");
  create_st_file(
      dir / "b.safetensors",
      "{\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}",
      std::vector<unsigned char>(4, 2));
  create_st_file(
      dir / "a.safetensors",
      "{\"a\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}",
      std::vector<unsigned char>(2, 1));
  create_st_file(
      dir / "nested" / "z.safetensors",
      "{\"z\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}",
      std::vector<unsigned char>(8, 3));
  {
    std::ofstream out(dir / "nested" / "tensor.data_0", std::ios::binary | std::ios::trunc);
    REQUIRE(out.is_open());
    out << "ignored";
  }

  auto ctx_or = get_disk_artifact_context(dir);
  REQUIRE(ctx_or.ok());
  REQUIRE((*ctx_or)->is_safetensors());
  REQUIRE((*ctx_or)->partition_paths().size() == 2);
  CHECK((*ctx_or)->partition_paths()[0].filename() == std::filesystem::path("a.safetensors"));
  CHECK((*ctx_or)->partition_paths()[1].filename() == std::filesystem::path("b.safetensors"));
  CHECK((*ctx_or)->total_size() == 6);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST_CASE(
    "DiskArtifactContext reconstructs multipart logical layout despite physical tail padding",
    "[disk_artifact_context][standard][multipart]") {
  using tensorcast::store::loader::get_disk_artifact_context;
  using tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing;

  reset_disk_artifact_context_cache_for_testing();

  auto dir = make_temp_dir("disk-artifact-context-multipart");
  constexpr uint64_t kFirstTensorSize = tensorcast::checkpoint::kPartitionMaxSize - 4092;
  constexpr uint64_t kSecondTensorOffset =
      tensorcast::checkpoint::TensorWriter::aligned_size(static_cast<size_t>(kFirstTensorSize));
  constexpr uint64_t kSecondTensorSize = 4096;

  create_sparse_file(dir / "tensor.data_0", /*size=*/tensorcast::checkpoint::kPartitionMaxSize);
  create_sparse_file(dir / "tensor.data_1", /*size=*/kSecondTensorSize);
  write_text(
      dir / "tensor_index.json",
      std::string("{") + "\"weights\":[" + std::to_string(0) + "," + std::to_string(kFirstTensorSize) +
          ",[1],[1],\"torch.uint8\",0]," + "\"bias\":[" + std::to_string(kSecondTensorOffset) + "," +
          std::to_string(kSecondTensorSize) + ",[1],[1],\"torch.uint8\",0]}");

  auto ctx_or = get_disk_artifact_context(dir);
  REQUIRE(ctx_or.ok());

  REQUIRE((*ctx_or)->partition_sizes().size() == 2);
  CHECK((*ctx_or)->partition_sizes()[0] == kSecondTensorOffset);
  CHECK((*ctx_or)->partition_sizes()[1] == kSecondTensorSize);
  CHECK((*ctx_or)->total_size() == kSecondTensorOffset + kSecondTensorSize);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST_CASE(
    "DiskArtifactContext prefers partitioned layout over safetensors when mixed",
    "[disk_artifact_context][mixed]") {
  using tensorcast::store::loader::get_disk_artifact_context;
  using tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing;

  reset_disk_artifact_context_cache_for_testing();

  auto dir = make_temp_dir("disk-artifact-context-mixed");
  {
    std::ofstream out(dir / "tensor.data", std::ios::binary | std::ios::trunc);
    REQUIRE(out.is_open());
    out << "payload";
  }
  create_st_file(
      dir / "weights.safetensors",
      "{\"w\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}",
      std::vector<unsigned char>(4, 4));

  auto ctx_or = get_disk_artifact_context(dir);
  REQUIRE(ctx_or.ok());
  REQUIRE_FALSE((*ctx_or)->is_safetensors());
  REQUIRE((*ctx_or)->partition_paths().size() == 1);
  CHECK((*ctx_or)->partition_paths()[0].filename() == std::filesystem::path("tensor.data"));
  CHECK((*ctx_or)->total_size() == 7);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST_CASE(
    "DiskArtifactContext preserves multipart numeric concatenation when index lacks per-part boundaries",
    "[disk_artifact_context][standard][multipart]") {
  using tensorcast::store::loader::get_disk_artifact_context;
  using tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing;

  reset_disk_artifact_context_cache_for_testing();

  auto dir = make_temp_dir("disk-artifact-context-multipart-fallback");
  create_sparse_file(dir / "tensor.data_0", /*size=*/16);
  create_sparse_file(dir / "tensor.data_1", /*size=*/8);
  write_text(dir / "tensor_index.json", R"({"payload":[0,24,[24],[1],"torch.uint8",0]})");

  auto ctx_or = get_disk_artifact_context(dir);
  REQUIRE(ctx_or.ok());

  REQUIRE((*ctx_or)->partition_sizes().size() == 2);
  CHECK((*ctx_or)->partition_sizes()[0] == 16);
  CHECK((*ctx_or)->partition_sizes()[1] == 8);
  CHECK((*ctx_or)->total_size() == 24);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST_CASE(
    "DiskArtifactContext trims only the final multipart tail padding under numeric concatenation",
    "[disk_artifact_context][standard][multipart]") {
  using tensorcast::store::loader::get_disk_artifact_context;
  using tensorcast::store::loader::reset_disk_artifact_context_cache_for_testing;

  reset_disk_artifact_context_cache_for_testing();

  auto dir = make_temp_dir("disk-artifact-context-multipart-tail");
  create_sparse_file(dir / "tensor.data_0", /*size=*/16);
  create_sparse_file(dir / "tensor.data_1", /*size=*/4096);
  write_text(dir / "tensor_index.json", R"({"payload":[0,24,[24],[1],"torch.uint8",0]})");

  auto ctx_or = get_disk_artifact_context(dir);
  REQUIRE(ctx_or.ok());

  REQUIRE((*ctx_or)->partition_sizes().size() == 2);
  CHECK((*ctx_or)->partition_sizes()[0] == 16);
  CHECK((*ctx_or)->partition_sizes()[1] == 8);
  CHECK((*ctx_or)->total_size() == 24);

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
