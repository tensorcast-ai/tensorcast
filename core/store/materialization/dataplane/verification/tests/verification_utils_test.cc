// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/verification/verification_utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include "absl/log/log.h"
#include "absl/log/log_sink_registry.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_verification.h"
#include "core/cuda/cuda_api.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/testing/common.h"

namespace fs = std::filesystem;
using tensorcast::store::loader::verification::MemoryView;
using tensorcast::store::loader::verification::ViewHashResult;

namespace {

class ScopedTempDir {
 public:
  explicit ScopedTempDir(const std::string& prefix) : path_(make_unique_path(prefix)) {
    fs::create_directories(path_);
  }

  ~ScopedTempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }

  const fs::path& path() const {
    return path_;
  }

 private:
  static fs::path make_unique_path(const std::string& prefix) {
    static std::atomic<uint64_t> counter{0};
    const auto ticks = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    return fs::temp_directory_path() /
        (prefix + "_" + std::to_string(ticks) + "_" + std::to_string(counter.fetch_add(1)));
  }

  fs::path path_;
};

MemoryView make_cpu_view(std::vector<uint8_t>& data) {
  MemoryView view;
  view.location = tensorcast::common::memory::MemoryLocation::CPU;
  view.base_ptr = data.data();
  view.size_bytes = data.size();
  return view;
}

MemoryView make_gpu_view(void* ptr, size_t bytes, int device_id) {
  MemoryView view;
  view.location = tensorcast::common::memory::MemoryLocation::GPU;
  view.base_ptr = ptr;
  view.size_bytes = bytes;
  view.gpu_device_id = device_id;
  return view;
}

class VectorSource : public tensorcast::store::loader::SeekableSource {
 public:
  explicit VectorSource(std::vector<uint8_t> data) : data_(std::move(data)) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return data_.size();
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    const size_t to_copy = std::min(max_bytes, data_.size() - cursor_);
    std::memcpy(dst, data_.data() + cursor_, to_copy);
    cursor_ += to_copy;
    return to_copy;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= data_.size()) {
      return static_cast<size_t>(0);
    }
    const size_t to_copy = std::min<size_t>(bytes, data_.size() - static_cast<size_t>(offset));
    std::memcpy(dst, data_.data() + offset, to_copy);
    return to_copy;
  }

 private:
  std::vector<uint8_t> data_;
  size_t cursor_{0};
};

} // namespace

TEST_CASE("VerificationUtils computes CPU multihash", "[verification][cpu]") {
  std::vector<uint8_t> data(16);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>(i * 3);
  }
  MemoryView view = make_cpu_view(data);
  auto hash_or = tensorcast::store::loader::verification::compute_data_multihash(view);
  REQUIRE(hash_or.ok());

  auto expected_or = tensorcast::store::loader::compute_data_multihash_from_cpu_memory(
      gsl::not_null<const void*>{data.data()}, data.size());
  REQUIRE(expected_or.ok());
  CHECK(hash_or.value() == expected_or.value());
}

TEST_CASE("VerificationUtils computes GPU multihash when CUDA available", "[verification][gpu]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping GPU verification test.");
    return;
  }

  constexpr size_t kBytes = 64;
  std::vector<uint8_t> host_data(kBytes);
  for (size_t i = 0; i < host_data.size(); ++i) {
    host_data[i] = static_cast<uint8_t>(0xAA ^ i);
  }

  REQUIRE(tensorcast::cuda::set_device(0).ok());
  void* device_ptr = nullptr;
  REQUIRE(tensorcast::cuda::malloc(&device_ptr, kBytes).ok());
  REQUIRE(tensorcast::cuda::memcpy(device_ptr, host_data.data(), kBytes, cudaMemcpyHostToDevice).ok());

  MemoryView view = make_gpu_view(device_ptr, kBytes, /*device_id=*/0);
  auto hash_or = tensorcast::store::loader::verification::compute_data_multihash(view);
  REQUIRE(hash_or.ok());

  auto expected_or = tensorcast::store::loader::compute_data_multihash_from_cpu_memory(
      gsl::not_null<const void*>{host_data.data()}, host_data.size());
  REQUIRE(expected_or.ok());
  CHECK(hash_or.value() == expected_or.value());

  REQUIRE(tensorcast::cuda::free(device_ptr).ok());
}

TEST_CASE("VerificationUtils generates and reuses verification.json", "[verification][metadata]") {
  ScopedTempDir dir("verification_utils");

  std::vector<uint8_t> data(128, 0x5A);
  MemoryView view = make_cpu_view(data);

  // When file absent, it should be generated.
  auto status = tensorcast::store::loader::verification::reuse_or_generate_verification_json(
      dir.path(),
      /*expected_byte_space_id=*/"canonical",
      view);
  CHECK(status.ok());
  fs::path verification_path = dir.path() / "verification.view_canonical.json";
  CHECK(fs::exists(verification_path));

  std::ifstream in(verification_path);
  REQUIRE(in.is_open());
  std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  auto info_or = tensorcast::common::ArtifactVerificationInfo::from_json(contents);
  REQUIRE(info_or.ok());
  CHECK(info_or->byte_space_id == "canonical");

  // Reuse should succeed without rewriting when byte_space_id matches.
  status = tensorcast::store::loader::verification::reuse_or_generate_verification_json(
      dir.path(),
      /*expected_byte_space_id=*/"canonical",
      view);
  CHECK(status.ok());

  // Mismatched byte_space_id should trigger regeneration into a new file.
  fs::path mismatch_path = dir.path() / "verification.view_variant.json";
  status = tensorcast::store::loader::verification::reuse_or_generate_verification_json(
      dir.path(),
      /*expected_byte_space_id=*/"variant",
      view);
  CHECK(status.ok());
  CHECK(fs::exists(mismatch_path));

  std::ifstream in2(mismatch_path);
  REQUIRE(in2.is_open());
  std::string updated((std::istreambuf_iterator<char>(in2)), std::istreambuf_iterator<char>());
  auto updated_info = tensorcast::common::ArtifactVerificationInfo::from_json(updated);
  REQUIRE(updated_info.ok());
  CHECK(updated_info->byte_space_id == "variant");
}

TEST_CASE("VerificationUtils detects tampering even when metadata was cached", "[verification][metadata][tamper]") {
  ScopedTempDir dir("verification_tamper");
  std::vector<uint8_t> data(256, 0x7A);
  MemoryView view = make_cpu_view(data);

  auto status = tensorcast::store::loader::verification::reuse_or_generate_verification_json(
      dir.path(),
      /*expected_byte_space_id=*/"",
      view);
  REQUIRE(status.ok());

  // Warm the cache with a second successful reuse.
  status = tensorcast::store::loader::verification::reuse_or_generate_verification_json(
      dir.path(),
      /*expected_byte_space_id=*/"",
      view);
  REQUIRE(status.ok());

  const fs::path verification_path = dir.path() / "verification.json";
  REQUIRE(fs::exists(verification_path));

  // Tamper with key values and rewrite the file with a matching signature.
  std::ifstream vf(verification_path);
  REQUIRE(vf.is_open());
  std::string payload((std::istreambuf_iterator<char>(vf)), std::istreambuf_iterator<char>());
  vf.close();

  auto info_or = tensorcast::common::ArtifactVerificationInfo::from_json(payload);
  REQUIRE(info_or.ok());
  tensorcast::common::ArtifactVerificationInfo tampered = *info_or;
  tampered.key_values = {0ULL, 0ULL, 0ULL};
  tampered.refresh_metadata_signature();
  std::ofstream out(verification_path);
  REQUIRE(out.is_open());
  out << tampered.to_json();
  out.close();

  status = tensorcast::store::loader::verification::reuse_or_generate_verification_json(
      dir.path(),
      /*expected_byte_space_id=*/"",
      view);
  REQUIRE_FALSE(status.ok());
  CHECK(status.code() == absl::StatusCode::kDataLoss);
}

TEST_CASE("VerificationUtils writes descriptor when absent", "[verification][descriptor]") {
  ScopedTempDir dir("verification_descriptor");

  const std::string index_mh = "mindex";
  const std::string data_mh = "mdata";
  const uint64_t total_size = 42;

  auto status = tensorcast::store::loader::verification::write_descriptor_if_absent(
      dir.path(), index_mh, data_mh, total_size, "json");
  CHECK(status.ok());

  fs::path descriptor = dir.path() / "artifact_descriptor.json";
  CHECK(fs::exists(descriptor));

  std::ifstream in(descriptor);
  REQUIRE(in.is_open());
  nlohmann::json j;
  in >> j;
  CHECK(j["index_multihash"] == index_mh);
  CHECK(j["data_multihash"] == data_mh);
  CHECK(j["total_size"] == total_size);

  // Second call should no-op.
  status = tensorcast::store::loader::verification::write_descriptor_if_absent(
      dir.path(), index_mh, data_mh, total_size, "json");
  CHECK(status.ok());
}

TEST_CASE("VerificationUtils atomic persistence prevents partial JSON reads", "[verification][atomic]") {
  ScopedTempDir dir("verification_atomic");
  std::vector<uint8_t> data(1024 * 32, 0x42);
  MemoryView view = make_cpu_view(data);
  const fs::path verification_path = dir.path() / "verification.view_canonical.json";
  const int iterations = 32;

  std::atomic<bool> writer_done{false};
  std::atomic<int> parse_failures{0};

  std::thread reader([&]() {
    while (!writer_done.load(std::memory_order_acquire)) {
      if (std::filesystem::exists(verification_path)) {
        std::ifstream in(verification_path);
        if (!in.is_open()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }
        std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (contents.empty()) {
          // Reader may observe file absence between unlink and rename; try again.
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }
        auto info_or = tensorcast::common::ArtifactVerificationInfo::from_json(contents);
        if (!info_or.ok()) {
          parse_failures.fetch_add(1, std::memory_order_relaxed);
          return;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });

  for (int i = 0; i < iterations; ++i) {
    std::error_code ec;
    std::filesystem::remove(verification_path, ec);
    auto status = tensorcast::store::loader::verification::reuse_or_generate_verification_json(
        dir.path(),
        /*expected_byte_space_id=*/"canonical",
        view);
    REQUIRE(status.ok());
    tensorcast::store::loader::verification::ClearVerificationMetadataCacheForTesting();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  writer_done.store(true, std::memory_order_release);
  reader.join();

  CHECK(parse_failures.load(std::memory_order_relaxed) == 0);
  CHECK(std::filesystem::exists(verification_path));
}

class CollectingLogSink : public absl::LogSink {
 public:
  void Send(const absl::LogEntry& entry) override {
    absl::MutexLock lock(&mu_);
    messages_.push_back(std::string(entry.text_message()));
  }

  bool Contains(absl::string_view needle) const {
    absl::MutexLock lock(&mu_);
    for (const auto& msg : messages_) {
      if (msg.find(needle) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

 private:
  mutable absl::Mutex mu_;
  std::vector<std::string> messages_ ABSL_GUARDED_BY(mu_);
};

TEST_CASE("VerificationUtils emits structured logs on metadata write", "[verification][logging]") {
  ScopedTempDir dir("verification_logging");

  std::vector<uint8_t> data(256, 0x11);
  MemoryView view = make_cpu_view(data);
  tensorcast::store::loader::verification::ClearVerificationMetadataCacheForTesting();

  CollectingLogSink sink;
  absl::AddLogSink(&sink);
  auto status = tensorcast::store::loader::verification::reuse_or_generate_verification_json(
      dir.path(),
      /*expected_byte_space_id=*/"",
      view);
  absl::RemoveLogSink(&sink);

  REQUIRE(status.ok());
  CHECK(sink.Contains("verification_metadata_write_succeeded"));
}

TEST_CASE("VerificationUtils computes view tree hash and leaf digests", "[verification][view]") {
  std::vector<uint8_t> data(32);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>(i);
  }
  VectorSource source(data);
  auto result_or = tensorcast::store::loader::verification::compute_view_tree_hash_and_leaves(
      source,
      /*total_size=*/data.size(),
      /*leaf_chunk_bytes=*/8);
  REQUIRE(result_or.ok());
  const ViewHashResult& result = result_or.value();
  CHECK(result.leaf_digests.size() == 4);
  CHECK_FALSE(result.multihash.empty());
}

TEST_CASE("VerificationUtils computes canonical leaf digests for CPU", "[verification][leaf]") {
  std::vector<uint8_t> data(32);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<uint8_t>(0x10 + i);
  }
  MemoryView view = make_cpu_view(data);
  std::array<uint64_t, 2> indices = {0, 1};
  auto digests_or = tensorcast::store::loader::verification::compute_canonical_leaf_digests(
      view,
      absl::MakeSpan(indices),
      /*chunk_bytes=*/16);
  REQUIRE(digests_or.ok());
  CHECK(digests_or->size() == indices.size());
}
