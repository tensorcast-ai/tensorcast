// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <unordered_map>

#include "core/checkpoint/checkpoint.h"

namespace fs = std::filesystem;
using namespace tensorcast::checkpoint;

TEST_CASE("save / load tensors that share same buffer", "[checkpoint][dedup]") {
  // ------------------------------------------------------------------
  // Prepare shared buffer and metadata
  // ------------------------------------------------------------------
  constexpr size_t kTensorBytes = 256; // 256-byte dummy tensor
  std::vector<uint8_t> shared_storage(kTensorBytes);
  for (size_t i = 0; i < kTensorBytes; ++i) {
    shared_storage[i] = static_cast<uint8_t>(i % 256);
  }

  const char* buffer_ptr = reinterpret_cast<const char*>(shared_storage.data());

  std::vector<std::string> tensor_names = {"tensorA", "tensorAlias"};
  std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> tensor_data;
  for (const auto& n : tensor_names) {
    tensor_data[n] = {reinterpret_cast<uint64_t>(buffer_ptr), kTensorBytes};
  }

  // ------------------------------------------------------------------
  // Save tensors
  // ------------------------------------------------------------------
  fs::path tmp_dir = fs::temp_directory_path() / "duplicate_buffer_test";
  if (fs::exists(tmp_dir))
    fs::remove_all(tmp_dir);
  fs::create_directories(tmp_dir);

  auto offsets = save_tensors(tensor_names, tensor_data, tmp_dir.string());

  // Both tensors must have identical offset (deduplicated)
  REQUIRE(offsets.size() == 2);
  REQUIRE(offsets["tensorA"] == offsets["tensorAlias"]);

  // The on-disk file contains one record. It is 8-byte aligned, and may be padded to page size (e.g. 4096 bytes) when
  // O_DIRECT is used.
  const fs::path data_file = tmp_dir / "tensor.data_0";
  REQUIRE(fs::exists(data_file));
  const uint64_t expected_min_size = ((kTensorBytes + 7ULL) / 8ULL) * 8ULL;
  const uint64_t actual_size = fs::file_size(data_file);
  REQUIRE(actual_size % 8ULL == 0ULL);
  REQUIRE(actual_size >= expected_min_size);

  // ------------------------------------------------------------------
  // Restore tensors via checkpoint API (CPU path)
  // ------------------------------------------------------------------
  std::unordered_map<std::string, std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::string, uint64_t>>
      meta_state;
  for (const auto& n : tensor_names) {
    meta_state[n] = {
        std::vector<int64_t>{static_cast<int64_t>(kTensorBytes)}, /*strides*/ {1}, "torch.uint8", /*alignment*/ 0};
  }

  // Build per-tensor offset map required by restore routine
  std::unordered_map<std::string, uint64_t> tensor_device_offsets;
  for (const auto& n : tensor_names) {
    tensor_device_offsets[n] = offsets[n];
  }

  auto state_dict = restore_tensors_from_disk(meta_state, tmp_dir.string(), tensor_device_offsets, /*device_id*/ -1);
  REQUIRE(state_dict.size() == 2);

  const auto& t0 = state_dict.at("tensorA");
  const auto& t1 = state_dict.at("tensorAlias");

  // Storage aliasing check – they must share the same data pointer
  void* ptr0 = t0.data_ptr<uint8_t>();
  void* ptr1 = t1.data_ptr<uint8_t>();
  REQUIRE(ptr0 == ptr1);

  // Content integrity check
  const uint8_t* data_ptr = t0.data_ptr<uint8_t>();
  REQUIRE(std::memcmp(data_ptr, shared_storage.data(), kTensorBytes) == 0);

  // Cleanup
  fs::remove_all(tmp_dir);
}