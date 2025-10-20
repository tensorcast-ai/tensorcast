// Copyright (c) 2025, TensorCast Team.

#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/common/artifact_verification.h"
#include "core/store/loader/verification_utils.h"

namespace fs = std::filesystem;

namespace {

tensorcast::store::loader::verification::MemoryView make_memory_view(std::vector<uint8_t>& data) {
  tensorcast::store::loader::verification::MemoryView view;
  view.location = tensorcast::common::memory::MemoryLocation::CPU;
  view.base_ptr = data.data();
  view.size_bytes = data.size();
  return view;
}

fs::path make_temp_dir(const std::string& suffix) {
  fs::path dir = fs::temp_directory_path() / suffix;
  fs::create_directories(dir);
  return dir;
}

std::string read_file(const fs::path& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return {};
  }
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE("Verification metadata guard serializes concurrent GPU loads", "[store][verification][race]") {
  fs::path base_dir = make_temp_dir("verification_multi_gpu");
  const int kThreads = 6;
  const int kRounds = 5;

  std::vector<uint8_t> payload(1024 * 64);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<uint8_t>((i * 37) & 0xFF);
  }
  auto view = make_memory_view(payload);

  for (int round = 0; round < kRounds; ++round) {
    fs::path artifact_dir = base_dir / ("round_" + std::to_string(round));
    fs::create_directories(artifact_dir);
    const fs::path verification_path = artifact_dir / "verification.json";

    tensorcast::store::loader::verification::ClearVerificationMetadataCacheForTesting();

    std::barrier sync_point(kThreads);
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    std::atomic<int> failures{0};

    for (int i = 0; i < kThreads; ++i) {
      workers.emplace_back([&, artifact_dir]() {
        sync_point.arrive_and_wait();
        auto status = tensorcast::store::loader::verification::reuse_or_generate_verification_json(
            artifact_dir,
            /*expected_byte_space_id=*/"",
            view);
        if (!status.ok()) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }

    for (auto& t : workers) {
      t.join();
    }

    REQUIRE(failures.load(std::memory_order_relaxed) == 0);
    CHECK(fs::exists(verification_path));

    const std::string contents = read_file(verification_path);
    REQUIRE_FALSE(contents.empty());

    auto info_or = tensorcast::common::ArtifactVerificationInfo::from_json(contents);
    REQUIRE(info_or.ok());
    CHECK(info_or->artifact_size == view.size_bytes);
  }

  fs::remove_all(base_dir);
}
