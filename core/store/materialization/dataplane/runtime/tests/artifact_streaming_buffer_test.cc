// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/testing/common.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <vector>

#include <nlohmann/json.hpp>
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/common/artifact_hash.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/cuda/cuda_api.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/sources/file_partition_source.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"

namespace fs = std::filesystem;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::common::memory::PinnedBufferPool;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::replica::MemoryState;
using tensorcast::store::replica::Replica;
using tensorcast::store::replica::ReplicaConfig;
using namespace tensorcast::testing;

namespace {

class ScopedTempDir {
 public:
  explicit ScopedTempDir(const std::string& prefix) : path_(fs::temp_directory_path() / prefix) {
    std::error_code ec;
    fs::remove_all(path_, ec);
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
  fs::path path_;
};

// Write minimal RFC-0007 metadata for a standard partition directory:
// - tensor_index.json: one dummy entry covering [0, total_size)
// - artifact_descriptor.json: with artifact_id = "mi2:<index_mh>:<data_mh>"
absl::Status write_descriptor_and_index(const std::filesystem::path& dir, uint64_t total_size) {
  using nlohmann::json;
  // 1) Build minimal canonical index JSON
  json idx = json::object();
  json entry = json::array();
  entry.push_back(0); // offset
  entry.push_back(total_size); // size
  entry.push_back(json::array()); // shape
  entry.push_back(json::array()); // stride
  entry.push_back("torch.uint8"); // dtype
  entry.push_back(static_cast<int64_t>(0)); // storage_offset
  idx["__dummy__"] = std::move(entry);

  const auto index_json_path = dir / "tensor_index.json";
  {
    std::ofstream of(index_json_path);
    if (!of.is_open()) {
      return absl::InternalError("Failed to open tensor_index.json for writing");
    }
    of << idx.dump();
  }

  // 2) Compute index/data multihash
  auto index_mh_or = tensorcast::common::compute_index_multihash(std::optional<std::string>(idx.dump()), "");
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }
  // Use unified SeekableSource hashing pipeline instead of ad-hoc disk dir hashing
  tensorcast::store::loader::FilePartitionSource::Options opts;
  // Collect all standard partition files (tensor.data, tensor.data_0, tensor.data_1, ...)
  std::vector<std::filesystem::path> parts;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto& name = entry.path().filename().string();
    if (name == "tensor.data" || name.starts_with("tensor.data_")) {
      parts.push_back(entry.path());
    }
  }
  std::ranges::sort(parts, [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
  if (parts.empty()) {
    // Fallback to the common single-file name
    parts.push_back(dir / "tensor.data_0");
  }
  uint64_t size_sum = 0;
  for (const auto& p : parts) {
    const auto sz = static_cast<size_t>(std::filesystem::file_size(p));
    opts.partition_paths.push_back(p);
    opts.partition_sizes.push_back(sz);
    size_sum += sz;
  }
  opts.total_size = size_sum;
  tensorcast::store::loader::FilePartitionSource src(std::move(opts));
  auto data_mh_or = tensorcast::store::loader::compute_data_multihash_from_seekable_source(src, size_sum);
  if (!data_mh_or.ok()) {
    return data_mh_or.status();
  }

  // 3) Write artifact_descriptor.json
  const auto descriptor_path = dir / "artifact_descriptor.json";
  json desc;
  desc["artifact_id"] = std::string("mi2:") + *index_mh_or + ":" + *data_mh_or;
  desc["index_multihash"] = *index_mh_or;
  desc["data_multihash"] = *data_mh_or;
  desc["schema_version"] = "v3";
  desc["encoding"] = "json";
  desc["total_size"] = size_sum;
  json hp;
  hp["chunk_size"] = 4 * 1024 * 1024;
  hp["fanout"] = 2;
  hp["algorithm"] = "sha2-256";
  desc["hash_params"] = hp;
  {
    std::ofstream of(descriptor_path);
    if (!of.is_open()) {
      return absl::InternalError("Failed to open artifact_descriptor.json for writing");
    }
    of << desc.dump(2);
  }

  return absl::OkStatus();
}
} // namespace

TEST_CASE("Streaming Disk Load to GPU", "[replica][disk][streaming]") {
  // Setup dummy replica with two partitions
  const std::string artifact_id = "streaming_disk_artifact";
  const std::string artifact_dir_name = "streaming_model_files";
  const std::string p0 = "tensor.data_0";
  const std::string p1 = "tensor.data_1";
  const size_t size0 = 1024 * 4;
  const size_t size1 = 1024 * 3;
  const size_t total_size = size0 + size1;

  // Skip if CUDA is not available on this system
  if (!is_cuda_available()) {
    SKIP("CUDA not available. Skipping streaming GPU test.");
  }

  ScopedTempDir base("streaming_disk_test");
  fs::create_directories(base.path() / artifact_dir_name);

  fs::path path0 = base.path() / artifact_dir_name / p0;
  fs::path path1 = base.path() / artifact_dir_name / p1;
  REQUIRE(create_dummy_file(path0, size0, 'S'));
  REQUIRE(create_dummy_file(path1, size1, 'T'));

  // Combined original data
  std::vector<char> data0 = read_file_content(path0);
  std::vector<char> data1 = read_file_content(path1);
  std::vector<char> combined;
  combined.reserve(total_size);
  combined.insert(combined.end(), data0.begin(), data0.end());
  combined.insert(combined.end(), data1.begin(), data1.end());
  REQUIRE(combined.size() == total_size);

  // RFC-0007: Write minimal canonical index + descriptor for standard partitions
  {
    auto st = write_descriptor_and_index(base.path() / artifact_dir_name, static_cast<uint64_t>(total_size));
    REQUIRE(st.ok());
  }

  // Pinned pool for streaming
  const size_t pool_total = 1024 * 1024;
  const size_t pool_chunk = 4096; // Use 4 KiB-aligned chunk size per alignment requirements
  auto pool = std::make_shared<PinnedBufferPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);
  auto async_runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  REQUIRE(async_runtime != nullptr);

  // Unified UMA is managed internally; no manual CPU arena injection needed.
  // Use new DiskSource
  DiskSource disk_src;
  disk_src.path = base.path() / artifact_dir_name;

  // Use aggregate initialization for ReplicaConfig
  ReplicaConfig cfg{
      .source = disk_src,
      .artifact_identifier = artifact_id,
      .device_type = ::tensorcast::DeviceType::GPU,
      .local_device_id = 0,
      .pinned_buffer_pool = pool,
      .async_runtime = async_runtime,
      .expected_artifact_size = std::nullopt,
      .max_buffer_bytes = 1024 * 2 // 2 KB buffer to force streaming
  };

  auto mstatus = Replica::create(cfg);
  REQUIRE(mstatus.ok());
  auto replica = std::move(*mstatus);

  // Perform streaming load directly to GPU
  REQUIRE(replica->get_memory_state(MemoryLocation::GPU) <= MemoryState::UNALLOCATED);
  auto fut = replica->ensure_loaded_async(MemoryLocation::GPU);
  REQUIRE(fut.valid());
  replica->wait_until_loaded(MemoryLocation::GPU, absl::Seconds(30)).IgnoreError();
  REQUIRE(replica->get_memory_state(MemoryLocation::GPU) == MemoryState::LOADED);

  // Get GPU data pointer and copy back
  auto gpu_ptrs = replica->get_data_pointer(MemoryLocation::GPU);
  REQUIRE(gpu_ptrs.size() == 1);
  void* gpu_ptr = gpu_ptrs[0];
  REQUIRE(gpu_ptr != nullptr);

  std::vector<char> host_buf(total_size);
  absl::Status copy_status = tensorcast::cuda::memcpy(host_buf.data(), gpu_ptr, total_size, cudaMemcpyDeviceToHost);
  REQUIRE(copy_status.ok());

  // Verify data matches
  REQUIRE(host_buf.size() == total_size);
  REQUIRE(host_buf == combined);

  // Cleanup
  replica.reset();
  pool.reset();
}
